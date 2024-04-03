import os
from typing import Union
import numpy as np

def metricsEvaluation(segmentMapPred: np.array, 
                      nClasses: int, 
                      npzSegmentMapTrue: Union[os.PathLike, str], 
                      imageName: Union[os.PathLike, str],
                      superCategoriesDict: dict=None):
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    with np.load(npzSegmentMapTrue) as data:
        segmentMapTrue = data.f.arr_0

    # Replacement with supercategories if defined
    if superCategoriesDict is not None:
        segmentMapTrueSuper = np.zeros_like(segmentMapTrue)
        for value in np.unique(segmentMapTrue):
            if value not in [0, 255]:
                try:
                    superClas = superCategoriesDict[value]
                    print(value, superCategoriesDict[value])
                    segmentMapTrueSuper += np.where(segmentMapTrue==value, superCategoriesDict[value], 0).astype(np.uint8)
                except Exception as e:
                    segmentMapTrueSuper = np.where(segmentMapTrue==value, 0, segmentMapTrueSuper).astype(np.uint8)
                
        segmentMapTrueSuper = np.where(segmentMapTrueSuper==255, 0, segmentMapTrueSuper)
        segmentMapTrue_ = segmentMapTrueSuper
    else:
        segmentMapTrue_ = segmentMapTrue
    hist = np.zeros((nClasses, nClasses))
    for lt, lp in zip(segmentMapTrue_, segmentMapPred):
        hist += _fast_hist(lt.flatten(), lp.flatten(), nClasses)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    print(f'Valid {np.where(valid==True)}')
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(nClasses), iu))
    mean_ok = ((np.nansum(iu)/(np.unique(segmentMapTrue)).shape[0]) == mean_iu) # Metrics coherency
    return {
        "ImageName": imageName,
        "LabelsTrue": np.unique(segmentMapTrue),
        "SuperLabelsTrue": np.unique(segmentMapTrueSuper) if superCategoriesDict is not None else np.unique(segmentMapTrue),
        "LabelsPreds": np.unique(segmentMapPred),
        "PixelAccuracy": acc,
        "MeanAccuracy": acc_cls,
        "FrequencyWeightedIoU": fwavacc,
        "MeanIoU": mean_iu,
        "ClassIoU": cls_iu,
        "MeanIoUtruth": mean_ok
    }