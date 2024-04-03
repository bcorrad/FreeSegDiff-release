import os
import time
import spacy
import numpy as np
import PIL.Image as Image
from scipy.spatial import distance
from lavis.models import load_model_and_preprocess
from config import DEVICE, segmentationClasses, openVocab

def BLIPImageCaptioner(image: Image, classesEncodings: list, img_pred_maps_folder: os.PathLike, pipe):
    """AI is creating summary for BLIPImageCaptioner

    Args:
        image (Image): [description]
        classesEncodings (list): [description]
        img_pred_maps_folder (os.PathLike): [description]
        pipe ([type]): [description]

    Returns:
        [type]: [description]
    """
    classesWordlist = []
    filteredWordsEncsList = []   # (classes, 768)
    if openVocab:
        candidateClasses = []
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)
        BLIP_st = time.time()
        image = vis_processors["eval"](image).unsqueeze(0).to(DEVICE)
        caption = model.generate({"image": image})[0]
        BLIP_et = time.time()
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(caption)
        filteredWordsList = []
        for token in doc:
            if not token.is_stop and (token.pos_ == "NOUN" or token.pos_ == "ADJ"):
                if token.pos_ == "ADJ":
                    noun_chunk = token.head
                    filteredWordsList.append(f"{token.text} {noun_chunk.text}")
                else:
                    filteredWordsList.append(token.text)
        classesWordlist = segmentationClasses + [item for item in filteredWordsList if item not in segmentationClasses]
        candidateClasses = [i for i, word in enumerate(classesWordlist) if word in filteredWordsList]
        BLIP_dt = BLIP_et - BLIP_st

        with open(os.path.join(img_pred_maps_folder, f'caption.txt'), 'w') as file:
            file.write(str(filteredWordsList) + "\n" + caption + " --> " + ", ".join(classesWordlist))
            
        return caption, classesWordlist, candidateClasses, BLIP_dt
    else:
        candidateClasses = []
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)
        BLIP_st = time.time()
        image = vis_processors["eval"](image).unsqueeze(0).to(DEVICE)
        caption = model.generate({"image": image})[0]
        BLIP_et = time.time()
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(caption)
        filteredWordsList = []
        for token in doc:
            if not token.is_stop and (token.pos_ == "NOUN" or token.pos_ == "ADJ"):
                if token.pos_ == "ADJ":
                    noun_chunk = token.head
                else:
                    noun_chunk = token.text
                filteredWordsList.append(noun_chunk)
        for word in filteredWordsList:
            filteredWordsEncsList.append(pipe._encode_prompt(f"{word}", pipe.device, 1, False, None).cpu().mean(1).squeeze(0)) # 768
        distCorrMatrix = np.zeros((len(filteredWordsEncsList), len(classesEncodings)))
        for fwi, fwe in enumerate(filteredWordsEncsList):
            for cei, ce in enumerate(classesEncodings):
                distCorrMatrix[fwi][cei] = distance.correlation(fwe, ce.mean(1).cpu().squeeze(0))
        args = np.argmin(distCorrMatrix, axis=1)
        vals = np.min(distCorrMatrix, axis=1)
        for i in range(len(args)):
            valsMean = np.mean(vals)
            if vals[i] <= valsMean:
                candidateClasses.append(args[i])
        candidateClassesString = f', '.join(segmentationClasses[i] for i in np.unique(candidateClasses))
        print(f'Candidate classes <= {valsMean} -> {candidateClassesString}')
        BLIP_dt = BLIP_et - BLIP_st

        with open(os.path.join(img_pred_maps_folder, f'caption.txt'), 'w') as file:
            file.write(str(filteredWordsList) + "\n" + caption + " --> " + ", ".join(segmentationClasses[c] for c in np.unique(candidateClasses)))

        return caption, segmentationClasses, np.unique(candidateClasses), BLIP_dt