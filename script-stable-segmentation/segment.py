import os
import time
import csv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from crf import CRF
from pamr import BinaryPamr
from models import CLIP_processor, CLIP_model, refiner
from utils import savePredictedMask, clustering_on_tensor
from evaluation import metricsEvaluation
from config import dataset, evaluateImages, segmentation_supercategories, segmentation_supercategories_names, colors, \
                    segmentation_classes, segmentationClasses, maskRefinement, DEVICE, reports_folder, MIN_N_CLUSTERS, MAX_N_CLUSTERS

def imageSegmentor(tensorToCluster: np.array, 
                   evalMetrics_matrix: list,
                   origImgToBeMasked: np.array,
                   img_pred_maps_folder: os.PathLike,
                   imageName: str,
                   candidateClasses: list,
                   pngNamePrefix: str='', 
                   bestK: int=None, 
                   segmentationClasses_: list=None,
                   savePredictedMasks: bool=True):
    pipeline_dts = {} # dictionary to store the time taken by each step of the pipeline (BLIP, CLIP, mask refinement)
    
    H, W, C = origImgToBeMasked.shape

    if segmentationClasses_ is not None:
        multipleChoiceCLIP = segmentationClasses_
    else:
        multipleChoiceCLIP = segmentationClasses
    try:
        plt.clf()
    except Exception as e:
        print(e)
    if bestK is None:
        N_CLUSTERS = range(MIN_N_CLUSTERS, MAX_N_CLUSTERS)
    else:
        N_CLUSTERS = [bestK]
    for j, K in enumerate(N_CLUSTERS):
        start_time = time.time()
        clusteredTensorK, _, silhouette_avg = clustering_on_tensor(tensorToCluster, K, 'kmeans', True)
        clusteredTensor = clusteredTensorK + 1
        clusteredMap = clusteredTensor
        predictedSegmentMap = np.zeros_like(clusteredTensor)
        predictedSegmentMap = cv2.resize(predictedSegmentMap, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        clustersIds = np.unique(clusteredMap)
        clusteredMapResized = cv2.resize(clusteredMap, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        outputsScoresMatrixForImage = np.zeros(shape=(len(multipleChoiceCLIP), len(clustersIds)), dtype=np.float32)
        CLIP_st = time.time()
        for cluster_idx, cluster_id in enumerate(clustersIds):
            clusterBinaryMask = np.where(clusteredMapResized == cluster_id, 1, 0).astype(np.uint8)
            origImageMasked = (clusterBinaryMask[:, :, np.newaxis] * origImgToBeMasked)
            inputs = CLIP_processor(text=multipleChoiceCLIP, images=origImageMasked, return_tensors="pt", padding=True).to(DEVICE)
            outputs = CLIP_model(**inputs)
            outputsProbs = [float(a) for a in outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()]
            outputsScoresMatrixForImage[:, cluster_idx] = outputsProbs
        CLIP_et = time.time()
        CLIP_dt = CLIP_et - CLIP_st
        pipeline_dts['CLIP_dt'] = CLIP_dt
        m = outputsScoresMatrixForImage
        mask = np.ones(m.shape[0], dtype=bool)
        mask[candidateClasses] = False
        m[mask] = 0
        m[m < np.max(m, axis=1, keepdims=True)] = 0
        assignment = np.argmax(m, axis=0)

        for clust, clas in enumerate(assignment):
            print(multipleChoiceCLIP[clas], clust, outputsScoresMatrixForImage[clas][clust])
            if outputsScoresMatrixForImage[clas][clust] != 0:
                clusterBinaryMask = np.where(clusteredMapResized == clust + 1, 1, 0).astype(np.uint8)
                if maskRefinement == "CRF":
                    maskRefinement_st = time.time()
                    output = (CRF(origImgToBeMasked, clusterBinaryMask)*255)[:,:,0]
                    maskRefinement_et = time.time()
                    maskRefinement_dt = maskRefinement_et - maskRefinement_st
                    pipeline_dts['refine'] = maskRefinement_dt
                elif maskRefinement == "PSP":
                    maskRefinement_st = time.time()
                    output = refiner.refine(Image.fromarray(origImgToBeMasked), (clusterBinaryMask*255), fast=False, L=900) 
                    maskRefinement_et = time.time()
                    maskRefinement_dt = maskRefinement_et - maskRefinement_st
                    pipeline_dts['refine'] = maskRefinement_dt
                elif maskRefinement == "PAMR":
                    maskRefinement_st = time.time()
                    data = torch.from_numpy(origImgToBeMasked).float().to(DEVICE)
                    data = data.permute(2,0,1)[None]
                    sal = torch.from_numpy(clusterBinaryMask).float().to(DEVICE)[None]
                    sal = sal.repeat(1,3,1,1)
                    output = BinaryPamr(data, sal, binary=0.5).cpu().numpy().squeeze(0)
                    output = (output * 255)[0,:,:]
                    maskRefinement_et = time.time()
                    maskRefinement_dt = maskRefinement_et - maskRefinement_st
                    pipeline_dts['refine'] = maskRefinement_dt
                else:
                    maskRefinement_st = time.time()
                    output = clusterBinaryMask*255
                    maskRefinement_et = time.time()
                    maskRefinement_dt = maskRefinement_et - maskRefinement_st
                output = np.where((output != 0) & (predictedSegmentMap == 0), 1, 0).astype(np.uint8)
                try:
                    classLabel = segmentation_supercategories[clas+1]
                except Exception as e:
                    print(e)
                    classLabel = clas+1
                predictedSegmentMap += output * (classLabel)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        pipeline_dtsCsvFile = os.path.join(reports_folder, f'pipeline_dts_{pngNamePrefix}_{K}.csv')
        with open(pipeline_dtsCsvFile, 'a', newline='') as file_csv:
            writer = csv.DictWriter(file_csv, fieldnames=pipeline_dts.keys())
            if not os.path.getsize(pipeline_dtsCsvFile):
                writer.writeheader()
            writer.writerow(pipeline_dts)

        legendClasses = ['unlabeled'] + multipleChoiceCLIP
        if evaluateImages:
            try:
                evaluationClasses = len(np.unique(list(segmentation_supercategories.values())))+1
                legendClasses = ['unlabeled'] + segmentation_supercategories_names
            except Exception as e:
                evaluationClasses = len(segmentation_classes)
                legendClasses = ['unlabeled'] + segmentationClasses_
            # Evaluate image
            evaluationMetrics = metricsEvaluation(predictedSegmentMap, 
                                                    evaluationClasses, 
                                                    os.path.join(dataset.dataset_maps_npz_folder, f'{imageName}.npz'), 
                                                    imageName, 
                                                    segmentation_supercategories)
            evaluationMetrics['mode'] = pngNamePrefix
            evaluationMetrics['time'] = elapsed_time
            evaluationMetrics['n_clusters'] = K
            evaluationMetrics['silhouette_avg'] = silhouette_avg
            metricsCsvFile = os.path.join(reports_folder, f'{pngNamePrefix}_{K}_metrics.csv')
            with open(metricsCsvFile, 'a', newline='') as file_csv:
                writer = csv.DictWriter(file_csv, fieldnames=evaluationMetrics.keys())
                if not os.path.getsize(metricsCsvFile):
                    writer.writeheader()
                writer.writerow(evaluationMetrics)
            evalMetrics_matrix.append([evaluationMetrics['n_clusters'], 
                                    evaluationMetrics['mode'], 
                                    evaluationMetrics['time'], 
                                    evaluationMetrics['PixelAccuracy'], 
                                    evaluationMetrics['MeanAccuracy'], 
                                    evaluationMetrics['MeanIoU']])
            mat = np.array(evalMetrics_matrix, dtype=object)
            metricsMatrix = mat[np.where((mat[:, 0] == K) & (mat[:, 1] == pngNamePrefix))][:, 3:]
            evalMetrics_nanmean = np.nanmean(metricsMatrix, 0, keepdims=True).flatten()
            metrics = {
                'nimgs': metricsMatrix.shape[0],
                'pix_acc': evalMetrics_nanmean[0], 
                'mean_acc': evalMetrics_nanmean[1], 
                'mean_iou': evalMetrics_nanmean[2]
            }
            datasetMetricCsvFile = os.path.join(reports_folder, f'{pngNamePrefix}_{int(K)}.csv')
            with open(datasetMetricCsvFile, 'a', newline='') as file_csv:
                writer = csv.DictWriter(file_csv, fieldnames=metrics.keys())
                if not os.path.getsize(datasetMetricCsvFile):
                    writer.writeheader()
                writer.writerow(metrics)
            # Save segmentation maps
            plotTitle = ', '.join([''.join(multipleChoiceCLIP[C].split()[0]) for C in candidateClasses])+f" \nmIoU = {float(evaluationMetrics['MeanIoU']):.3f}"
        else:
            plotTitle = ', '.join([''.join(multipleChoiceCLIP[C].split()[0]) for C in candidateClasses])
        if savePredictedMasks:
            # savePredictedMask(predictedSegmentMap,
            #                     origImgToBeMasked,
            #                     os.path.join(img_pred_maps_folder, f"{pngNamePrefix}_{imageName}_assigned_{K}.png"),
            #                     colors,
            #                     legendClasses,
            #                     title=''.join(plotTitle))
            savePredictedMask(predictedSegmentMap,
                                origImgToBeMasked,
                                os.path.join(img_pred_maps_folder, f"{pngNamePrefix}_{imageName}_no_legend_assigned_{K}.png"),
                                colors,
                                legendClasses,
                                paper_labeling=True)
            savePredictedMask(cv2.resize(clusteredMap, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8), 
                                None,
                                os.path.join(img_pred_maps_folder, f"{pngNamePrefix}_{imageName}_clusters_{K}.png"))

        return pipeline_dts