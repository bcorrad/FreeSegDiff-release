import time
from config import *
from typing import Union
import torchvision.transforms as T
import torch
from tqdm import tqdm
import cv2
from PIL import Image
from models import pipe, attn_hook
from BLIPcaptioner import BLIPImageCaptioner
from segment import imageSegmentor

BLIP_DT = 0.0

def runSegmentation(imgsDir: Union[os.PathLike, str]):
    # text conditioning
    encs_list = []
    total_time_list = []
    IMAGE_PROC_DT_LIST = []
    for i in range(len(segmentationClasses)):
        if len(segmentationClasses[i].split()) > 1:
            batch_encs_list = []
            for el in segmentationClasses[i].split():
                batch_encs_spec = pipe._encode_prompt(f"A photo of {el}", pipe.device, 1, False, None) # encode segmentation classes into text embeddings
                batch_encs_list.append(batch_encs_spec)
            synonimsEncoding = torch.cat(batch_encs_list, dim=0)
            encs_list.append(synonimsEncoding.mean(0)[None])
        else:
            batch_encs = pipe._encode_prompt(f"A photo of {segmentationClasses[i]}", pipe.device, 1, False, None) # encode segmentation classes into text embeddings
            encs_list.append(batch_encs)
    encs = torch.cat(encs_list, dim=0)
    # pool into single embedding. we use mean pooling on used tokens
    lengths = [len(pipe.tokenizer(c)['input_ids']) for c in segmentationClasses]
    text_embeddings = torch.stack([encs[i, 1:lengths[i]].mean(0) for i in range(len(encs))])
    torch.save(text_embeddings, embeddingPath)
    if not candidateClassesBottleneck:
        attn_hook.set_text_embeddings(text_embeddings[None]) # none to add batch dimension
    null_embeddings = pipe._encode_prompt('', pipe.device, 1, False, None) # encode null text that is used during unet forward
    timesteps = torch.Tensor(timesteps_vector).long().to(DEVICE) # 0 = no noise, 1000 = full noise
    prep = T.Compose([T.Resize((img_display_size, img_display_size))])

    evalMetrics_matrix = []
    source = os.listdir(imgsDir)
    
    for img in tqdm(source):
        image_timer_st = time.time()
        pipeline_dts = {} 
        imageName = os.path.splitext(img)[0]
        if evaluateImages and not os.path.exists(os.path.join(dataset.dataset_maps_npz_folder, f'{imageName}.npz')):
            print(f"{imageName} skipped: {os.path.join(dataset.dataset_maps_npz_folder, imageName+'.npz')} does not exist.")
            continue 
        if not os.path.exists(os.path.join(pred_maps_folder, imageName)):
            os.mkdir(os.path.join(pred_maps_folder, imageName))
            img_pred_maps_folder = os.path.join(pred_maps_folder, imageName)
        origImgToBeMasked = cv2.cvtColor(cv2.imread(os.path.join(imgsDir, img)), cv2.COLOR_BGR2RGB)  
        H, W, C = origImgToBeMasked.shape
        pilImage = Image.open(os.path.join(imgsDir, img)).convert('RGB') # load image
        image = T.ToTensor()(prep(pilImage))[None].mul(2).sub(1).half().to(DEVICE) # compute latent image
        if inferCandidateClasses:
            if openVocab:
                caption, filteredWordsList, candidateClasses, BLIP_DT = BLIPImageCaptioner(image=pilImage, classesEncodings=None, img_pred_maps_folder=img_pred_maps_folder, pipe=pipe)
            else:
                caption, filteredWordsList, candidateClasses, BLIP_DT = BLIPImageCaptioner(image=pilImage, classesEncodings=encs_list, img_pred_maps_folder=img_pred_maps_folder, pipe=pipe)
        else:
            candidateClasses = [i for i in range(len(segmentationClasses))]
        if candidateClassesBottleneck:
            encs_list = []
            for i in range(len(segmentationClasses)):
                if len(segmentationClasses[i].split()) > 1:
                    batch_encs_list = []
                    for el in segmentationClasses[i].split():
                        batch_encs_spec = pipe._encode_prompt(f'A photo of a {el}', pipe.device, 1, False, None) # encode segmentation classes into text embeddings
                        batch_encs_list.append(batch_encs_spec)
                    synonimsEncoding = torch.cat(batch_encs_list, dim=0)
                    encs_list.append(synonimsEncoding.mean(0)[None])
                else:
                    batch_encs = pipe._encode_prompt(f'A photo of a {segmentationClasses[i]}', pipe.device, 1, False, None) # encode segmentation classes into text embeddings
                    encs_list.append(batch_encs)
            encs = torch.cat(encs_list, dim=0)
            # pool into single embedding. we use mean pooling on used tokens
            lengths = [len(pipe.tokenizer(c)['input_ids']) for c in segmentationClasses]
            text_embeddings = torch.stack([encs[i, 1:lengths[i]].mean(0) for i in range(len(encs))])
            attn_hook.set_text_embeddings(text_embeddings[None])

        init_latents = pipe.vae.encode(image).latent_dist.sample()
        init_latents = 0.18215 * init_latents
        noise = torch.randn_like(init_latents) # compute noisy samples
        noisy_latents = pipe.scheduler.add_noise(init_latents, noise, timesteps)
        print(attn_hook.selectBlock)

        SDM_st = time.time()
        
        for i in range(len(timesteps)): # run unet and get output from the attention hook
            noise_pred = pipe.unet(noisy_latents[i:i+1], timesteps[i:i+1], encoder_hidden_states=null_embeddings).sample
            if collectUnetAttention:
                attention_scores_resized, attention_scores_dict = attn_hook.computeAttentionMaps()
            if collectUnetFeatures:
                feature_maps_resized = attn_hook.computeFeatureMaps()

        SDM_et = time.time()
        SDM_dt = SDM_et - SDM_st
        pipeline_dts['SDM'] = SDM_dt

        if collectUnetFeatures and collectUnetAttention:
            af = np.concatenate((a, f), axis=0)
            if adaptiveK > 0:
                SEGMENTATION_START = time.time()
                imageSegmentor(af[None], pngNamePrefix=f'af', evalMetrics_matrix=evalMetrics_matrix, img_pred_maps_folder=img_pred_maps_folder, imageName=imageName, origImgToBeMasked=origImgToBeMasked, candidateClasses=candidateClasses, bestK=len(candidateClasses)+adaptiveK, segmentationClasses_=filteredWordsList, savePredictedMasks=savePredictedMasks)
                SEGMENTATION_END = time.time()
                SEGMENTATION_DT = SEGMENTATION_END - SEGMENTATION_START
                print(f"Segmentation time: {SEGMENTATION_DT:.2f} seconds")
            else:
                SEGMENTATION_START = time.time()
                imageSegmentor(af[None], pngNamePrefix=f'af', evalMetrics_matrix=evalMetrics_matrix, img_pred_maps_folder=img_pred_maps_folder, imageName=imageName, origImgToBeMasked=origImgToBeMasked, candidateClasses=candidateClasses, segmentationClasses_=filteredWordsList, savePredictedMasks=savePredictedMasks)
                SEGMENTATION_END = time.time()
                SEGMENTATION_DT = SEGMENTATION_END - SEGMENTATION_START
                print(f"Segmentation time: {SEGMENTATION_DT:.2f} seconds")
        elif collectUnetFeatures and not collectUnetAttention:
            feature_store = feature_maps_resized
            f = feature_store.reshape(np.prod(feature_store.shape[:-2]), -1)
            if adaptiveK > 0:
                SEGMENTATION_START = time.time()
                imageSegmentor(f[None], pngNamePrefix=f'f', evalMetrics_matrix=evalMetrics_matrix, img_pred_maps_folder=img_pred_maps_folder, imageName=imageName, origImgToBeMasked=origImgToBeMasked, candidateClasses=candidateClasses, bestK=len(candidateClasses)+adaptiveK, segmentationClasses_=filteredWordsList, savePredictedMasks=savePredictedMasks)
                SEGMENTATION_END = time.time()
                SEGMENTATION_DT = SEGMENTATION_END - SEGMENTATION_START
                print(f"Segmentation time: {SEGMENTATION_DT:.2f} seconds")
            else:
                SEGMENTATION_START = time.time()
                imageSegmentor(f[None], pngNamePrefix=f'f', evalMetrics_matrix=evalMetrics_matrix, img_pred_maps_folder=img_pred_maps_folder, imageName=imageName, origImgToBeMasked=origImgToBeMasked, candidateClasses=candidateClasses, segmentationClasses_=filteredWordsList, savePredictedMasks=savePredictedMasks)
                SEGMENTATION_END = time.time()
                SEGMENTATION_DT = SEGMENTATION_END - SEGMENTATION_START
                print(f"Segmentation time: {SEGMENTATION_DT:.2f} seconds")
        elif collectUnetAttention and not collectUnetFeatures:
            attention_store = attention_scores_resized
            a = attention_store.reshape(np.prod(attention_store.shape[:-2]), -1)
            if adaptiveK > 0:
                SEGMENTATION_START = time.time()
                imageSegmentor(a[None], pngNamePrefix=f'a', evalMetrics_matrix=evalMetrics_matrix, img_pred_maps_folder=img_pred_maps_folder, imageName=imageName, origImgToBeMasked=origImgToBeMasked, candidateClasses=candidateClasses, bestK=len(candidateClasses)+adaptiveK, segmentationClasses_=filteredWordsList, savePredictedMasks=savePredictedMasks)
                SEGMENTATION_END = time.time()
                SEGMENTATION_DT = SEGMENTATION_END - SEGMENTATION_START
                print(f"Segmentation time: {SEGMENTATION_DT:.2f} seconds")
            else:    
                SEGMENTATION_START = time.time()
                imageSegmentor(a[None], pngNamePrefix=f'a', evalMetrics_matrix=evalMetrics_matrix, img_pred_maps_folder=img_pred_maps_folder, imageName=imageName, origImgToBeMasked=origImgToBeMasked, candidateClasses=candidateClasses, segmentationClasses_=filteredWordsList, savePredictedMasks=savePredictedMasks)
                SEGMENTATION_END = time.time()
                SEGMENTATION_DT = SEGMENTATION_END - SEGMENTATION_START
                print(f"Segmentation time: {SEGMENTATION_DT:.2f} seconds")
        
        pilImage.close()
        
        PIPELINE_DT = BLIP_DT + SDM_dt + SEGMENTATION_DT
        IMAGE_PROC_DT_LIST.append(PIPELINE_DT)
        print(f"Pipeline time: {PIPELINE_DT:.2f} seconds")
        
    # Save the average time for all images
    avg_time = sum(total_time_list) / len(total_time_list)
    print(f"Average time for all images: {avg_time:.2f} seconds")
    image_timer_avg = sum(IMAGE_PROC_DT_LIST) / len(IMAGE_PROC_DT_LIST)
    print(f"Average image processing time: {image_timer_avg:.2f} seconds")