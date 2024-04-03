import matplotlib
matplotlib.use(backend='Agg')
from utils import checkGPUsAvailable, copyConfigFile, findExtension, initializeExperiment

checkGPUsAvailable()

import os
import numpy as np
from datasetsManager import segmentationDataset, ade20kManager, COCODataset

img_display_size = 512  # int, possible choices: 512, 1024. Size to which test images are resized before segmentation.
mapSize = 16 if img_display_size == 512 else 32  # int, calculated based on img_display_size value
map_display_size = 224  # int. Size to which test images are resized before metrics evaluation.

segmentation_classes = None

batch_size = None # Union[int, None]. Size of the subset of the overall dataset to segment. If None, takes all the images.
LDMPath = '/home/corradini/.models/stable-diffusion-v1-4/'  # str. Path of the latent diffusion model .json file.

projectRoot = '/home/corradini/FreeSegDiff-release'  # str. Project root folder.
projectSubdir = 'script-stable-segmentation'  # str. Project subfolder.
embeddingPath = os.path.join(projectRoot, 'embeddings', 'segmentation_embeddings.pt') # Union[os.PathLike, str, None]. Path of the text embedding file. If None, it defaults to projectRoot/embeddings/segmentation_embeddings.pt
colorsPath = os.path.join(projectRoot, 'colors', 'colormap.csv') # Union[os.PathLike, str, None]. Path of the color .npz file. If None, it defaults to projectRoot/colors/color_palette.npz
DEVICE = 'cuda:0'  # str. Device to be used (cpu or gpu).
N = 1  # Union[int, None]. Number of denoising inference timesteps.
N_amount = 0
imgsDir =  None # "/home/corradini/FreeSegDiff/images"
mapsDir = None

randomImages = False # Pick random images from the dataset
randomBatchSize = None # Number of random images

imgsDir_ext = None
mapsDir_ext = None

clusteringCriterion = "kmeans" # Literal['kmeans', 'agglomerative']
dataset_name = "pascalvoc"  # str. Dataset name. Literal['coco', 'coco-2017', 'coco27-2017', 'pascalvoc']
adaptiveK = 0 # int. Number to add to the number of predicted classes for adaptive-K clustering
MIN_N_CLUSTERS = 4
MAX_N_CLUSTERS = 5

collectUnetFeatures = True
collectUnetAttention = False
attnHooksResolutions = [16]
mapsSize = 32

inferCandidateClasses = True # Category prefiltering via BLIP
candidateClassesBottleneck = False # Use only candidate classes as input for the SDM

openVocab = True # Open vocabulary segmentation
maskRefinement = "CRF" # Literal['CRF', 'PSP', 'PAMR', None]
savePredictedMasks = True

pipeline_dts = {} #{'BLIP', 'SDM', 'CLIP', 'refine', 'Tot'}

evaluationMode = "miou"  # Literal['jaccard', 'miou']
evaluateImages = True if imgsDir is None else False
evaluateImages = False if openVocab is True else evaluateImages

atnPrefix = ''.join([str(a) for a in attnHooksResolutions])
experimentID = ''.join(['f' if collectUnetFeatures else '', 
                        'a' if collectUnetAttention else '', 
                        f'_{dataset_name}',
                        f'_atn{mapsSize}-', atnPrefix, 
                        f'_openVocab' if openVocab else '_closedVocab',
                        f'_adaptK' if adaptiveK else f'_K_{MIN_N_CLUSTERS}-{MAX_N_CLUSTERS}',
                        f'_rand' if randomImages else '',
                        f'_candBott-' if candidateClassesBottleneck else '',
                        f'_noise-{N_amount}' if N_amount else '',
                        f'_refine-{maskRefinement}' if maskRefinement else '',
                        f'_ds-{batch_size}' if batch_size is not None else '_ds-all']) # Union[str, None]. Suffix added to the experiment folder, if needed.
print(experimentID)

def noiseVectorGenerator(N, seed=17, amount=500, reproduce=True):
    import numpy as np
    if reproduce:
        return list(amount*(np.ones(N)))
    else:
        # np.random.seed(seed)
        return [100, 300, 500]

timesteps_vector = noiseVectorGenerator(N=N, amount=N_amount) 
print(timesteps_vector)

# Create experiment folder
experiment_folder, pred_maps_folder, reports_folder = initializeExperiment(projectRoot, experimentID)
# If colormap.csv file does not exist in the colors folder, create a palette with 250 colors
colors_folder = os.path.join(projectRoot, 'colors')
if not os.path.exists(os.path.join(colors_folder)):
    os.mkdir(os.path.join(colors_folder))
if not os.path.exists(os.path.join(colors_folder, 'colormap.csv')):
    colors = [list(np.random.randint(1, 255, 3)) for _ in range(200)]
    colors.insert(0, [0,0,0])
    colors = np.array(colors)
    with open(os.path.join(colors_folder, 'colormap.csv'), 'w') as file:
        for color in colors:
            file.write(','.join([str(c) for c in color]) + "\n")
else:
    with open(os.path.join(colors_folder, 'colormap.csv'), 'r') as file:
        lines = [line.strip().split(',') for line in file.readlines()]
    colors = [[int(value) for value in line] for line in lines]
    colors = np.array(colors)
copyConfigFile(experiment_folder, projectRoot, projectSubdir)
# Save images and maps 
if dataset_name == 'coco-2017':
    datasetPath = '/home/corradini/.datasets/coco-2017/images/val2017/'
    annotationsTxt = '/home/corradini/.datasets/coco-2017/labels.txt'
    annotationsDirPath = '/home/corradini/.datasets/coco-2017/annotations/SegClass_val2017_gray/'
    classNamesFilePath = '/home/corradini/.datasets/coco-2017/labels.txt'
    supercategoriesFilePath = None
    colorMapFile = os.path.join(experiment_folder, 'colormap.csv')
    dataset = segmentationDataset(datasetName=dataset_name,
                                datasetDirectoryPath=datasetPath, 
                                datasetImageNamesFile=None, 
                                annotationsDirectoryPath=annotationsDirPath, 
                                classNamesFilePath=annotationsTxt, 
                                colormapFilePath=None, 
                                batchSize=batch_size,
                                nClasses=None,
                                randomImages=randomImages,
                                supercategoriesFilePath=supercategoriesFilePath)
    segmentation_classes = dataset.classes if segmentation_classes is None else segmentation_classes
    segmentation_supercategories, segmentation_supercategories_names = dataset.categorySupercatDict, dataset.superCategories

if dataset_name == 'coco27-2017':
    datasetPath = '/home/corradini/.datasets/coco-2017/images/val2017/'
    annotationsTxt = '/home/corradini/.datasets/coco-2017/labels.txt'
    annotationsDirPath = '/home/corradini/.datasets/coco-2017/annotations/SegClass_val2017_gray/'
    classNamesFilePath = '/home/corradini/.datasets/coco-2017/labels.txt'
    supercategoriesFilePath = '/home/corradini/.datasets/coco-2017/classesSupercategories.csv' 
    # supercategoriesFilePath = None
    colorMapFile = os.path.join(experiment_folder, 'colormap.csv')
    dataset = segmentationDataset(datasetName=dataset_name,
                                datasetDirectoryPath=datasetPath, 
                                datasetImageNamesFile=None, 
                                annotationsDirectoryPath=annotationsDirPath, 
                                classNamesFilePath=annotationsTxt, 
                                colormapFilePath=None, 
                                batchSize=batch_size,
                                nClasses=None,
                                randomImages=randomImages,
                                supercategoriesFilePath=supercategoriesFilePath)
    segmentation_classes = dataset.classes if segmentation_classes is None else segmentation_classes
    segmentation_supercategories, segmentation_supercategories_names = dataset.categorySupercatDict, dataset.superCategories

elif dataset_name == 'pascalvoc':
    datasetDirectoryPath = '/home/corradini/.datasets/pascalVOC/VOCdevkit/VOC2012/JPEGImages/'
    datasetImageNamesFile = '/home/corradini/.datasets/pascalVOC/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    annotationsDirectoryPath = '/home/corradini/.datasets/pascalVOC/VOCdevkit/VOC2012/SegmentationClass'
    classNamesFile = '/home/corradini/.datasets/pascalVOC/classes.txt'
    colorMapFile = '/home/corradini/.datasets/pascalVOC/colormap.txt'
    dataset = segmentationDataset(dataset_name, 
                                    datasetDirectoryPath, 
                                    datasetImageNamesFile, 
                                    annotationsDirectoryPath, 
                                    classNamesFile, 
                                    colorMapFile, 
                                    batch_size, 
                                    randomImages=randomImages,
                                    randomBatchSize=randomBatchSize)
    segmentation_classes = dataset.classes if segmentation_classes is None else segmentation_classes
    segmentation_supercategories, segmentation_supercategories_names = None, None

elif dataset_name == 'ade20k':
    datasetDirectoryPath = '/home/corradini/.datasets/ade20k/ADE20K_2021_17_01/images/ADE/validation'
    annotationsDirectoryPath = None
    classNamesFile = '/home/corradini/.datasets/ade20k/ADE20K_2021_17_01/objects.txt'
    dataset = ade20kManager(datasetDirectoryPath, annotationsDirectoryPath, classNamesFile, batchSize=batch_size)
    segmentation_classes = dataset.getSegmentationCategories() if segmentation_classes is None else segmentation_classes

elif dataset_name is None and imgsDir is not None:
    datasetDirectoryPath = imgsDir

# Add unlabeled class on top of the list of segmentation classes
if 'unlabeled' not in segmentation_classes:
    segmentation_classes.insert(0, 'unlabeled')

print('config.segmentation_classes: ', segmentation_classes)

segmentationClasses = segmentation_classes[1:]
print('config.segmentationClasses: ', segmentationClasses)

# Save images and masks 
if mapsDir is None and imgsDir is None:
    if dataset_name == "coco" and isinstance(dataset, COCODataset):
        val_gen = dataset.dataGeneratorCoco(filterClasses=segmentation_classes[1:], batch_size=batch_size)
        dataset.saveImgsAndMaps(val_gen)
        imgsDir = dataset.dataset_imgs_folder
        mapsDir = dataset.dataset_maps_folder
        imgsDir_ext = findExtension(imgsDir)
        mapsDir_ext = findExtension(mapsDir)
    elif isinstance(dataset, segmentationDataset):
        dataset.saveImgsAndMaps()
        imgsDir = dataset.dataset_imgs_folder
        mapsDir = dataset.dataset_maps_folder
        imgsDir_ext, mapsDir_ext = dataset.getExtensions() 
    elif isinstance(dataset, ade20kManager):
        dataset.saveImgsAndMaps()
        colors = dataset.getColorPalette()
        imgsDir = dataset.dataset_imgs_folder
        mapsDir = dataset.dataset_maps_folder
        imgsDir_ext, mapsDir_ext = dataset.getExtensions() 
elif mapsDir is None and imgsDir is not None:
    imgsDir_ext = findExtension(imgsDir)
else:
    imgsDir_ext = findExtension(imgsDir)
    mapsDir_ext = findExtension(mapsDir) 

