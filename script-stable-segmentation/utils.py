import os, random
from typing import Union, Literal
import numpy as np

def checkGPUsAvailable():
    """ Check if GPUs are available """
    import torch
    num_gpus = torch.cuda.device_count()
    cudaDevs = [torch.cuda.device(i) for i in range(num_gpus)]
    if len(cudaDevs) > 0:
        print(f'--- Nice! GPUs are accessible {cudaDevs} ---')
        for gpu_index in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(gpu_index)
            print(f"GPU {gpu_index}: {gpu_name}")
        chosen_gpu_index = random.randint(0, gpu_index)
        return f'cuda:{chosen_gpu_index}'
    else:
        print(f'--- No GPUs found {cudaDevs} ---')

def findExtension(dir):
    import re
    import numpy as np
    files = os.listdir(dir)
    ext = np.unique([re.findall(r'\.(.+)', file)[0] for file in files])
    # remove dot from extension
    ext = [e.replace('.', '') for e in ext]
    return ext[0]

def initializeExperiment(root: os.PathLike, 
                        experimentID: str=''):
    """Function to initialize an experiment folder and subfolders.

    Args:
        root (Union[os.PathLike, str], optional): [description]. Defaults to '.'.
        class_folders (list, optional): [description]. Defaults to [].

    Returns:
        [type]: [description]
    """
    import time
    # Create experiment folder
    experiment_folder = time.strftime("%Y%m%d_%H-%M-%S")
    experiment_folder = f"{experiment_folder}_{experimentID}" if experimentID is not None else experiment_folder
    os.makedirs(os.path.join(root, experiment_folder))
    experiment_folder = os.path.join(root, experiment_folder)
    # Create "maps" subfolder
    pred_maps_folder = os.path.join(root, experiment_folder, "pred_maps")
    os.makedirs(pred_maps_folder)
    # Create a folder for csv reports
    reports_folder = os.path.join(root, experiment_folder, "reports")
    os.makedirs(reports_folder)
    return experiment_folder, \
            pred_maps_folder, \
            reports_folder

# Function to copy the configuration file in the experiment folder
def copyConfigFile(experiment_folder, projectRoot, projectSubdir):
    import shutil
    shutil.copy(os.path.join(projectRoot, projectSubdir, 'config.py'), os.path.join(experiment_folder, 'config.py'))
    shutil.copy(os.path.join(projectRoot, projectSubdir, 'attentionHook.py'), os.path.join(experiment_folder, 'attentionHook.py'))
    shutil.copy(os.path.join(projectRoot, projectSubdir, 'main.py'), os.path.join(experiment_folder, 'main.py'))

def calcola_media_mean_iou(file_csv='/home/corradini/zeroshot_segmentation_coco/20230904_19-33-02/dataset_metrics_5.csv'):
    import csv
    # Lista per memorizzare i valori di MeanIoU
    mean_iou_values = []
    try:
        # Apre il file CSV in modalità lettura
        with open(file_csv, mode='r') as file:
            # Crea un oggetto CSV reader
            csv_reader = csv.DictReader(file)

            # Itera attraverso ogni riga del file CSV
            for row in csv_reader:
                # Estrae il valore della colonna "MeanIoU" dalla riga corrente e lo converte in float
                mean_iou = float(row['mean_iou'])

                # Aggiunge il valore alla lista
                mean_iou_values.append(mean_iou)

        # Calcola la media dei valori di MeanIoU
        if len(mean_iou_values) > 0:
            mean_iou_avg = sum(mean_iou_values) / len(mean_iou_values)
            return mean_iou_avg
        else:
            return None  # Nessun valore di MeanIoU trovato nel file
    except FileNotFoundError:
        return None  # Gestisce il caso in cui il file non esista
    
def savePredictedMask(predictedMask: Union[np.array, None], 
                      originalImage: Union[np.array, None],
                      imagePath: Union[os.PathLike, str], 
                      colorPalette: Union[list, None]=None,
                      legendLabels: Union[list, None]=None,
                      eng: Literal['matplotlib', 'cv2']='matplotlib',
                      **kwargs):
    if predictedMask is not None and len(predictedMask.shape) == 2:
        predictedMaskRGB = np.repeat(predictedMask[:, :, np.newaxis], 3, axis=2)
        threeChannelMap = np.zeros_like(predictedMaskRGB)
        if colorPalette is None:
            threeChannelMap = predictedMask
    elif originalImage is not None:
        threeChannelMap = originalImage
    alpha = 1.
    if eng == 'matplotlib':
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        if colorPalette is not None:
            for c in np.unique(predictedMask):
                threeChannelMap = np.where(predictedMaskRGB == c, colorPalette[c], threeChannelMap)
        plt.figure()
        if originalImage is not None:
            plt.imshow(originalImage)
            alpha = 0.7
            plt.imshow(threeChannelMap, alpha=alpha)
        # Clustering only
        elif originalImage is None:
            plt.imshow(threeChannelMap)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            plt.axis('off')
            plt.savefig(imagePath, bbox_inches='tight', pad_inches=0)
            plt.clf()
            plt.close()
        # Print legend
        if legendLabels is not None and colorPalette is not None and 'paper_labeling' not in kwargs: 
            try:
                patches = [mpatches.Patch(color=colorPalette[i]/255, label=legendLabels[i].capitalize()) for i in np.unique(predictedMask)]
                plt.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15))
            except Exception as e:
                print(e)
            if 'title' in kwargs:
                plt.title(kwargs['title'])
            plt.axis('off')
            plt.savefig(imagePath)
            plt.clf()
            plt.close()
            return
        # Print object class on the object
        if legendLabels is not None and colorPalette is not None and 'paper_labeling' in kwargs:
            for n in np.unique(predictedMask):
                if n != 0:
                    points = np.where(predictedMask==n)
                    y = points[0].mean()
                    x = points[1].mean()
                    plt.text(x + 2, y + 2, legendLabels[n].capitalize(), color='black', fontsize=13, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            plt.axis('off')
            plt.savefig(imagePath, bbox_inches='tight', pad_inches=0)
            plt.clf()
            plt.close()
            return
        # if 'title' in kwargs:
        #     plt.title(kwargs['title'])
        # plt.axis('off')
        # plt.savefig(imagePath)
        # plt.clf()
        # plt.close()
    elif eng == 'cv2':
        import cv2
        if colorPalette is not None:
            threeChannelMap = np.zeros_like(predictedMaskRGB)
            for c in np.unique(predictedMask):
                threeChannelMap = np.where(predictedMaskRGB == c, colorPalette[c], threeChannelMap)
        else:
            for c in np.unique(predictedMask):
                threeChannelMap = np.where(predictedMaskRGB == c, np.random.randint(0, 255, size=3), threeChannelMap)
        cv2.imwrite(imagePath, threeChannelMap)
        cv2.destroyAllWindows()

# VOC-C evaluation function
def calculate_average_mIoU_for_few_classes(filename):
    import pandas as pd
    print(filename)
    df = pd.read_csv(filename)
    imageNamesList = []
    miou = 0.0
    count = 0
    a = [i.replace('[','').replace(']','') for i in df['LabelsTrue']]
    a_ = [list(b) for b in a]
    for i, b in enumerate(a_):
        while " " in b:
            b.remove(" ")
        if len(b) <= 2:
            count += 1
            miou += df['MeanIoU'][i]
            imageNamesList.append(df['ImageName'][i])
    
    print(f"miou={miou/count}")
    return miou/count, count, miou #,imageNamesList

def clustering_on_tensor(tensor, 
                        n_clusters, 
                        clustering: Literal['kmeans', 'agglomerative']='kmeans',
                        returnLabels: bool=True):
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.neighbors import NearestCentroid
    from sklearn.metrics import silhouette_score
    if len(tensor.shape) == 4:
        N, C, W, H = tensor.shape
    elif len(tensor.shape) == 3 and tensor.shape[-1] == 32**2:
        N, C, W, H = tensor.shape[0], tensor.shape[1], 32, 32
        tensor = tensor.reshape((N, C, W, H))
    elif len(tensor.shape) == 3 and tensor.shape[-1] == 64**2:
        N, C, W, H = tensor.shape[0], tensor.shape[1], 64, 64
        tensor = tensor.reshape((N, C, W, H))
    elif len(tensor.shape) == 3 and tensor.shape[-1] == 16**2:
        N, C, W, H = tensor.shape[0], tensor.shape[1], 16, 16
        tensor = tensor.reshape((N, C, W, H))
    elif len(tensor.shape) == 3 and tensor.shape[-1] == 8**2:
        N, C, W, H = tensor.shape[0], tensor.shape[1], 8, 8
        tensor = tensor.reshape((N, C, W, H))
    elif len(tensor.shape) == 3 and tensor.shape[-1] == 128**2:
        N, C, W, H = tensor.shape[0], tensor.shape[1], 128, 128
        tensor = tensor.reshape((N, C, W, H))
    elif len(tensor.shape) == 3 and tensor.shape[-1] == 512**2:
        N, C, W, H = tensor.shape[0], tensor.shape[1], 512, 512
        tensor = tensor.reshape((N, C, W, H))
    if len(tensor.shape) in [3, 4]:
        tensor_2d = tensor.transpose((1, 0, 2, 3))
        tensor_2d = tensor_2d.reshape((C, -1))
        tensor_2d = tensor_2d.transpose(1, 0)
    elif len(tensor.shape) == 2:
        import math
        C, WH = tensor.shape
        N = 1
        W, H = int(math.sqrt(WH)), int(math.sqrt(WH))
        tensor_2d = tensor.transpose(1, 0)
    if clustering == 'kmeans':
        # cluster_ids_x, cluster_centers = kmeans(X=tensor_2d, num_clusters=n_clusters, distance='euclidean', device=device)
        kmeans_ = KMeans(n_clusters=n_clusters).fit(X=tensor_2d)
        cluster_ids_x = kmeans_.labels_
        cluster_centers = kmeans_.cluster_centers_
    elif clustering == 'agglomerative':
        agglomerativeClustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X=tensor_2d)
        clf = NearestCentroid()
        clf.fit(tensor_2d, agglomerativeClustering.labels_)
        cluster_centers = clf.centroids_
        cluster_ids_x = agglomerativeClustering.labels_
    else:
        raise ValueError("Invalid clustering algorithm. Supported options: 'kmeans', 'agglomerative'")
    labels = cluster_ids_x.reshape((N, W, H))
    silhouette_avg = silhouette_score(tensor_2d, labels.reshape(-1))
    if returnLabels:
        labels = labels.transpose(1,2,0)
        labels = labels.squeeze(-1)
        return labels, cluster_centers, silhouette_avg
    else:
        return None, cluster_centers, silhouette_avg
    
def calculate_label_accuracy(csv_file):
    import pandas as pd
    # Leggi il file CSV
    data = pd.read_csv(csv_file)
    true_labels, pred_labels = [], []

    # Estrai le colonne 'LabelsTrue' e 'LabelsPreds' come liste
    for labelList in data['LabelsTrue']:
        labelList = labelList.split(' ')
        lab = [int(item.strip('[] ').strip()) for item in labelList if item.strip('[] ').strip()]
        true_labels.append(lab)

    for labelList in data['LabelsPreds']:
        labelList = labelList.split(' ')
        lab = [int(item.strip('[] ').strip()) for item in labelList if item.strip('[] ').strip()]
        pred_labels.append(lab)

    # Inizializza una variabile per tenere traccia del numero di corrispondenze
    num_matches = 0

    # Calcola la percentuale di corrispondenza tra le due liste
    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            num_matches += 1

    accuracy = (num_matches / len(true_labels)) * 100

    return accuracy

def plotAttentionMaps(attention_scores_dict_resized: dict, originalImagePath: os.PathLike, destImagePath: os.PathLike):
    import matplotlib.pyplot as plt
    ncols = len(attention_scores_dict_resized)
    width_ratio = [4]
    for _ in range(ncols):
        width_ratio.append(2)
    fig, ax = plt.subplots(nrows=1, ncols=ncols+1, gridspec_kw={'width_ratios': width_ratio}, figsize=(12, 3))
    ax[0].imshow(plt.imread(originalImagePath))
    ax[0].set_axis_off()
    index = 1
    for key, value in attention_scores_dict_resized.items():
        ax[index].imshow(value.mean(0))
        ax[index].set_title(key, rotation=90, fontsize=10)
        ax[index].set_axis_off()
        index += 1
    plt.tight_layout()
    plt.savefig(os.path.join(destImagePath, 'attHooks.png'))