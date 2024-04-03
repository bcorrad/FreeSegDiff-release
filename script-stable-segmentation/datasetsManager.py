import numpy as np
import os, csv, pickle
import cv2
from tqdm import tqdm
from config import *
from typing import Union, Literal
import random
import torch

class segmentationDataset:
    def __init__(self,
            datasetName: Union[str, None],
            datasetDirectoryPath: Union[str, os.PathLike],
            datasetImageNamesFile: Union[str, os.PathLike, None], 
            annotationsDirectoryPath: Union[str, os.PathLike],
            classNamesFilePath: Union[str, os.PathLike, None],
            colormapFilePath: Union[str, os.PathLike, None],
            batchSize: int=4,
            nClasses: int=None,
            randomImages: bool=False,
            randomBatchSize: int=None,
            supercategoriesFilePath: Union[str, os.PathLike, None]=None
            ):
        
        self.datasetName = datasetName
        self.datasetDirectoryPath = datasetDirectoryPath
        self.datasetImageNamesFile = datasetImageNamesFile
        self.annotationsDirectoryPath = annotationsDirectoryPath
        self.classNamesFilePath = classNamesFilePath
        self.colormapFilePath = colormapFilePath
        self.imagesFilenames = None
        self.batchSize = batchSize
        self.randomImages = randomImages
        if batchSize is None:
            self.batchSize = len(self._get_filenames())
        self.randomBatchSize = randomBatchSize if randomBatchSize is not None else self.batchSize
        self.supercategoriesFilePath = supercategoriesFilePath

        self.dataset_imgs_folder = None
        self.dataset_maps_folder = None
        self.dataset_maps_npz_folder = None
        self.dataset_imgs_npz_folder = None
        self.initializeDatasetDirs()
        self.classes = self.getSegmentationCategories()[:nClasses] if nClasses is not None else self.getSegmentationCategories()
        if self.supercategoriesFilePath is not None:
            self.categorySupercatDict, self.superCategories = self.getSegmentationSupercategories() 
        else: 
            self.categorySupercatDict = None 
            self.superCategories = None
        if self.colormapFilePath is not None:
            self.colormap = self.getColorMap()

    def initializeDatasetDirs(self):
        from config import experiment_folder
        self.dataset_imgs_folder = os.path.join(experiment_folder, "dataset_imgs")
        os.makedirs(self.dataset_imgs_folder)
        self.dataset_maps_folder = os.path.join(experiment_folder, "dataset_maps")
        os.makedirs(self.dataset_maps_folder)
        self.dataset_maps_npz_folder = os.path.join(experiment_folder, "dataset_maps_npz")
        os.makedirs(self.dataset_maps_npz_folder)
        self.dataset_imgs_npz_folder = os.path.join(experiment_folder, "dataset_imgs_npz")
        os.makedirs(self.dataset_imgs_npz_folder)

    def _get_filenames(self):
        if self.datasetImageNamesFile is not None and os.path.isfile(self.datasetImageNamesFile): 
            with open(self.datasetImageNamesFile, 'r') as f:
                imagesSource = f.read().splitlines()
                imagesSource = [os.path.splitext(img)[0] for img in imagesSource] # remove extension from images names
            imagesSource = [os.path.join(self.datasetDirectoryPath, img) for img in imagesSource]
        else:
            imagesSource = [os.path.join(self.datasetDirectoryPath, img) for img in os.listdir(self.datasetDirectoryPath)]
        return imagesSource
        
    def saveImgsAndMaps(self):
        if self.datasetImageNamesFile is None: # if self.datasetImageNamesFile is None, search images in self.datasetDirectoryPath
            imagesSource = os.listdir(self.datasetDirectoryPath)
        elif os.path.isfile(self.datasetImageNamesFile): # else, read images names from self.datasetImageNamesFile and build the paths for images and maps
            with open(self.datasetImageNamesFile, 'r') as f:
                imagesSource = f.read().splitlines()
                imagesSource = [os.path.splitext(img)[0] for img in imagesSource] # remove extension from images names
            imagesSource = [os.path.join(self.datasetDirectoryPath, img) for img in imagesSource]
        else:
            raise ValueError("datasetImageNamesFile must be a valid file or None")
        self.imgs_ext = os.path.splitext(os.listdir(self.datasetDirectoryPath)[0])[1]
        self.maps_ext = os.path.splitext(os.listdir(self.annotationsDirectoryPath)[0])[1]
        
        if self.randomImages:
            random.seed(len(imagesSource))
            imgsRandomIndexes = list(range(len(imagesSource)))
            random.shuffle(imgsRandomIndexes)
            source = [imagesSource[i] for i in imgsRandomIndexes[:self.batchSize]]
        else:
            source = imagesSource[:self.batchSize]
        
        imgs_counter = 0
        for filename in tqdm(source):
            if not filename.endswith(self.imgs_ext):
                filename = filename + self.imgs_ext
            img_name = os.path.splitext(os.path.basename(filename))[0]
            map_name = os.path.join(self.annotationsDirectoryPath, f"{img_name}{self.maps_ext}")
            try:
                img = cv2.imread(os.path.join(self.datasetDirectoryPath, filename))
                cv2.imwrite(os.path.join(self.dataset_imgs_folder, f"{img_name}{self.maps_ext}"), img)
            except:
                continue
            try:
                map = cv2.imread(map_name)
                if map.shape[-1] == 3:
                    if self.colormapFilePath is not None or self.datasetName == 'pascalvoc':
                        map = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(os.path.join(self.dataset_maps_folder, f"{img_name}{self.maps_ext}"), map)
                        map_class = self.oneChannelMap(map) # WxHx1
                    elif self.colormapFilePath is None or self.datasetName == 'cocostuff':
                        # map[np.all(map == 255, axis=2)] = [0,0,0]
                        map_class = map[:, :, 0]
                        cv2.imwrite(os.path.join(self.dataset_maps_folder, f"{img_name}{self.maps_ext}"), map)
                cv2.destroyAllWindows()
                np.savez_compressed(os.path.join(self.dataset_maps_npz_folder, img_name), map_class)
                imgs_counter += 1
            except:
                print(f'{img_name} not found')
                continue

    def oneChannelMap(self, threeChannelMap, target_color=(224,224,192), replacement_color=(0,0,0)):
        # border is 224, 224, 192 in bgr mode
        # if cv2.isColor(threeChannelMap, cv2.COLOR_BGR2RGB):
        target_color = np.array(target_color)
        # Trova i pixel che corrispondono al colore target
        borders = np.all(threeChannelMap == target_color, axis=2)
        # Rimpiazza il colore target con il colore di sostituzione
        map = np.zeros_like(threeChannelMap)
        map[borders] = replacement_color
        for class_index, color in enumerate(self.colormap):
            borders = np.all(threeChannelMap == color, axis=2) 
            map[borders] = [class_index, class_index, class_index]
        return map[:,:,0]
    
    def threeChannelMap(self, oneChannelMap, colormap):
        threeChannelMap = np.zeros_like(oneChannelMap)
        for class_index, color in enumerate(colormap):
            threeChannelMap[np.all(oneChannelMap == class_index, axis=2)] = color
        return threeChannelMap

    def getExtensions(self):
        return self.imgs_ext, self.maps_ext
    
    def getSegmentationCategories(self):
        # Read rows of the txt classNamesFilePath
        classes = []
        with open(self.classNamesFilePath, 'r') as f:
            rows = f.readlines()
            rows = [row.replace('\n','') for row in rows]
            # Check if class with index 0 is background or unlabeled
            try:
                rows.pop(rows.index('background'))
            except Exception as e:
                print(e)
            try:
                rows.pop(rows.index('unlabeled'))
            except Exception as e:
                print(e)
            classes = rows
        return classes
    
    def getSegmentationSupercategories(self):
        cat_supercat_dict = {}
        supercategories_dict = {}
        supercategories_list = []
        with open(self.supercategoriesFilePath, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                class_id = int(row['classId'])
                supercategory_index = int(row['supercategory_index'])
                supercategory = row['supercategory']
                cat_supercat_dict[class_id] = supercategory_index
                supercategories_dict[supercategory_index] = supercategory
        for chiave, valore in dict(sorted(supercategories_dict.items())).items():
            supercategories_list.append(valore)
        return cat_supercat_dict, supercategories_list
    
    def getColorMap(self):
        # Read rows of the txt colormapFilePath. The format is R G B
        if os.path.exists(self.colormapFilePath):
            with open(self.colormapFilePath, 'r') as f:
                rows = f.readlines()
                rows = [row.replace('\n','') for row in rows]
            return [[int(color) for color in row.split()] for row in rows]    

from typing import Union, Literal, List
import os
import cv2
import numpy as np
import json

class ade20kManager:
    def __init__(self,
            datasetDirectoryPath: Union[str, os.PathLike],
            annotationsDirectoryPath: Union[str, os.PathLike],
            classNamesFilePath: Union[str, os.PathLike],
            n_classes: Union[int, None]=150,
            batchSize: Union[int, None]=None
            ) -> None:
        
        self.datasetDirectoryPath = datasetDirectoryPath
        self.annotationsDirectoryPath = annotationsDirectoryPath
        self.classNamesFilePath = classNamesFilePath
        self.n_classes = n_classes
        self.classes, self.colors = self.read_classes_file(self.n_classes) if n_classes is not None else self.read_classes_file()
        self.batchSize = batchSize

        self.mode_path = dict()
        self.npz_path = dict()

        self.json_files = []

        self.dataset_imgs_folder = None
        self.dataset_maps_folder = None
        self.dataset_maps_npz_folder = None
        self.dataset_imgs_npz_folder = None

        self.img_ext = None
        self.maps_ext = None

        if annotationsDirectoryPath is not None:
            self.initializeDatasetDirs_global()
        else:
            self.initializeDatasetDirs()
        self.scan_subfolders_for_json(self.datasetDirectoryPath, self.json_files)

    def initializeDatasetDirs_global(self):
        self.dataset_imgs_folder = os.path.join(self.annotationsDirectoryPath, "dataset_imgs")
        if not os.path.exists(self.dataset_imgs_folder):
            os.makedirs(self.dataset_imgs_folder)
        self.dataset_maps_folder = os.path.join(self.annotationsDirectoryPath, "dataset_maps")
        if not os.path.exists(self.dataset_maps_folder):
            os.makedirs(self.dataset_maps_folder)
        self.dataset_maps_npz_folder = os.path.join(self.annotationsDirectoryPath, "dataset_maps_npz")
        if not os.path.exists(self.dataset_maps_npz_folder):
            os.makedirs(self.dataset_maps_npz_folder)
        self.dataset_imgs_npz_folder = os.path.join(self.annotationsDirectoryPath, "dataset_imgs_npz")
        if not os.path.exists(self.dataset_imgs_npz_folder):
            os.makedirs(self.dataset_imgs_npz_folder)

    def initializeDatasetDirs(self):
        from config import experiment_folder
        # Create subfolders for maps and images
        self.dataset_imgs_folder = os.path.join(experiment_folder, "dataset_imgs")
        os.makedirs(self.dataset_imgs_folder)
        self.dataset_maps_folder = os.path.join(experiment_folder, "dataset_maps")
        os.makedirs(self.dataset_maps_folder)
        # Create subfolder for maps npz
        self.dataset_maps_npz_folder = os.path.join(experiment_folder, "dataset_maps_npz")
        os.makedirs(self.dataset_maps_npz_folder)
        # Create subfolder for imgs npz
        self.dataset_imgs_npz_folder = os.path.join(experiment_folder, "dataset_imgs_npz")
        os.makedirs(self.dataset_imgs_npz_folder)

    # Read the classes file (separator is \t) and save the first column in a list (classes)
    def read_classes_file(self, n_classes=None):
        with open(self.classNamesFilePath) as f:
            classes = [line.split('\t')[0] for line in f][:n_classes] if n_classes is not None else [line.split('\t')[0] for line in f]
        # generate a list of colors
        colors = [tuple(np.random.randint(0, 255, 3)) for _ in range(len(classes))]
        # add unlabeled class and color
        classes.insert(0, 'unlabeled')
        colors.insert(0, (0,0,0))
        return classes, colors
    
    def getSegmentationCategories(self):
        return self.classes[1:]
    
    def getColorPalette(self):
        return self.colors[1:]
    
    def getExtensions(self):
        return self.img_ext, self.maps_ext

    def scan_subfolders_for_json(self, root, file_paths):
        if os.path.isfile(root) and root.endswith('.json'):
            if self.batchSize is not None:
                if len(file_paths) < self.batchSize:
                    file_paths.append(root)
                    print('Saving json file: ', root, ' -> Number of json files: ', len(file_paths), self.batchSize)
                else:
                    return
            elif self.batchSize is None:
                file_paths.append(root)
                print('Saving json file: ', root, ' -> Number of json files: ', len(file_paths))
        elif os.path.isdir(root):
            for name in os.listdir(root):
                path = os.path.join(root, name)
                self.scan_subfolders_for_json(path, file_paths)
        
    def parse_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data
    
    def saveImgsAndMaps(self, batchsize: Union[int, None]=None, modes: List[Literal['rgb', 'gray']]=['rgb', 'gray'], saveimage: bool=True, savetarget: bool=True, savenpz: bool=True):
        if self.annotationsDirectoryPath is not None and not os.path.exists(self.annotationsDirectoryPath):
            os.makedirs(self.annotationsDirectoryPath)
        for mode in modes:
            self.mode_path[mode] = os.path.join(self.dataset_maps_folder, mode)
            if not os.path.exists(self.mode_path[mode]):
                os.makedirs(self.mode_path[mode])
            if savenpz:
                self.npz_path[mode] = os.path.join(self.mode_path[mode], 'npz')
                if not os.path.exists(self.npz_path[mode]):
                    os.makedirs(self.npz_path[mode])
        if savetarget:
            target_path = os.path.join(self.dataset_maps_folder, 'target')
            if not os.path.exists(target_path):
                os.makedirs(target_path)
        json_files_batched = self.json_files[:batchsize] if batchsize is not None else self.json_files
        for json_file in json_files_batched:
            data_json = self.parse_json(json_file)
            data = data_json['annotation']
            if self.img_ext is None:
                self.img_ext = data['filename'].split('.')[-1]
            if savetarget:
                segmentation_mask_path = os.path.join(os.path.dirname(json_file), f"{data['filename'][:-4]}_seg.png") # Get path of segmentation mask
                segmentation_mask = cv2.imread(segmentation_mask_path)
                cv2.imwrite(os.path.join(target_path, data['filename'][:-4] + '_seg.png'), segmentation_mask)
            if saveimage:
                image_path = os.path.join(os.path.dirname(json_file), data['filename']) # Get path of segmentation mask
                image = cv2.imread(image_path)
                cv2.imwrite(os.path.join(self.dataset_imgs_folder, data['filename']), image)
                # cv2.imwrite(os.path.join(self.dataset_imgs_folder, data['filename'][:-4] + '.png'), image)
            for mode in modes:
                final_mask = np.zeros(data['imsize'], dtype=np.uint8) if mode == 'rgb' else np.zeros(data['imsize'][:-1], dtype=np.uint8) # Create empty mask
                for obj in data['object']:
                    if self.maps_ext is None:
                        self.maps_ext = obj['instance_mask'].split('.')[-1]
                    instance_mask_path = os.path.join(os.path.dirname(json_file), obj['instance_mask']) # Get path of instance mask
                    instance_mask = cv2.imread(instance_mask_path) if mode == 'rgb' else cv2.imread(instance_mask_path, cv2.IMREAD_GRAYSCALE)
                    # print(obj['id'] , obj['name'], obj['name_ndx'])
                    if obj['parts']['part_level'] != 0:
                        part_level = obj['parts']['part_level']
                        part_id = obj['parts']['ispartof']
                        obj_clas = [part['name'] for i, part in enumerate(data['object']) if part['id'] == part_id][0]
                    else:
                        obj_clas = obj['name']

                    if obj_clas in self.classes:
                        clas = self.classes.index(obj_clas)
                    else:
                        clas = self.classes.index('unlabeled')
                        # print(f"obj_clas {obj_clas} not in class list => replaced with unlabeled class")
                    color = self.colors[clas] if mode == 'rgb' else clas
                    # print(f'Coloring class {self.classes[clas]} with color {color}')
                    final_mask = np.where(instance_mask == 255, color, final_mask)
                final_mask_path = os.path.join(self.mode_path[mode], data['filename'][:-4] + '.png')
                cv2.imwrite(final_mask_path, final_mask)
                if savenpz and mode == 'gray':
                    final_mask_path = os.path.join(self.dataset_maps_npz_folder, data['filename'][:-4] + '.npz')
                    np.savez_compressed(final_mask_path, final_mask)
                elif savenpz and mode == 'rgb':
                    final_mask_path = os.path.join(self.npz_path[mode], data['filename'][:-4] + '.npz')
                    np.savez_compressed(final_mask_path, final_mask)

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import os
import cv2
from tqdm import tqdm
from typing import Union, Literal

# Class for managing the COCO dataset. Takes care of loading the images and the annotations from paths and of displaying them
class COCODataset:
    def __init__(self,
            dataDir: Union[str, os.PathLike],
            annFile: Union[str, os.PathLike],
            dataType: Union[Literal['train'], Literal['val'], Literal['test']]='val'
            ):
        """ Initialize COCO api for instance annotations
        
        Args:
            dataDir (Union[str, os.PathLike]): Path to the directory containing the COCO dataset
            annFile (Union[str, os.PathLike]): Path to the annotations file
            dataType (Union[Literal['train'], Literal['val'], Literal['test']], optional): Type of the dataset. Defaults to 'val'.
            
            Returns:
                COCO: COCO api for instance annotations
        """
        
        self.dataDir = dataDir
        self.annFile = annFile
        self.dataType = dataType
        self.images = None

        # initialize the COCO api for instance annotations
        self.coco=COCO(self.annFile)
        # Get the list of categories and supercategories
        self.catIDs = self.coco.getCatIds()
        self.cats = self.coco.loadCats(self.catIDs)
        
        self.dataset_imgs_folder = None
        self.dataset_maps_folder = None
        self.dataset_maps_npz_folder = None
        self.dataset_imgs_npz_folder = None
        self.initializeCocoDirs()
        
    def initializeCocoDirs(self):
        from config import experiment_folder
        # Create subfolders for maps and images
        self.dataset_imgs_folder = os.path.join(experiment_folder, "dataset_imgs")
        os.makedirs(self.dataset_imgs_folder)
        self.dataset_maps_folder = os.path.join(experiment_folder, "dataset_maps")
        os.makedirs(self.dataset_maps_folder)
        # Create subfolder for maps npz
        self.dataset_maps_npz_folder = os.path.join(experiment_folder, "dataset_maps_npz")
        os.makedirs(self.dataset_maps_npz_folder)
        # Create subfolder for imgs npz
        self.dataset_imgs_npz_folder = os.path.join(experiment_folder, "dataset_imgs_npz")
        os.makedirs(self.dataset_imgs_npz_folder)

    def getCOCOCategories(self):
        """ Display COCO categories and supercategories
        
        Returns:
            None
        """

        nms=[cat['name'] for cat in self.cats]
        print(len(nms),'COCO categories: \n{}\n'.format(' '.join(nms)))
        return nms
        
    def getCOCOSupercategories(self):
        """ Display COCO supercategories
        
        Returns:
            list of supercategories
        """

        nms = set([cat['supercategory'] for cat in self.cats])
        print(len(nms),'COCO supercategories: \n{}'.format(' '.join(nms)))
        return nms

    def getClassName(self, 
                    classID: int,
                    cats: list):
        """ Get the name of the class given its ID and the list of categories
        """
        for i in range(len(self.cats)):
            if self.cats[i]['id'] == classID:
                return self.cats[i]['name']
        return None
    
    def loadCOCOImages(self,
            filterClasses: Union[list, None],
            comb: bool=True):
        """ Load and display a random image from the COCO dataset

        Args:
            filterClasses (list, optional): List of classes to filter the images. Defaults to ['laptop', 'tv', 'cell phone'].
            comb (bool, optional): If True, take images showing at least one of filterClasses element. Defaults to True.

        Returns:
            None
        """
        # list of all images containing given categories
        images = []
        if filterClasses!=None:
            # iterate for each individual class in the list
            for className in filterClasses:
                # get all images containing given class
                catIds = self.coco.getCatIds(catNms=className)
                imgIds = self.coco.getImgIds(catIds=catIds)
                images += self.coco.loadImgs(imgIds)
        else:
            imgIds = self.coco.getImgIds()
            images = self.coco.loadImgs(imgIds)
            
        # Now, filter out the repeated images    
        unique_images = []
        for i in range(len(images)):
            if images[i] not in unique_images:
                unique_images.append(images[i])

        self.dataset_size = len(unique_images)
        # save the list of images containing the filter classes
        self.images = unique_images

        print("Number of images containing the filter classes:", self.dataset_size)
        return self.images

    def getImage(self,
                imageObj,
                image_folder,
                input_image_size: tuple=(224,224)):
        self.img_ext = os.path.join(image_folder, imageObj['file_name']).split('.')[-1]
        # Read and normalize an image
        train_img = io.imread(os.path.join(image_folder, imageObj['file_name']))/255.0
        # Resize
        train_img = cv2.resize(train_img, input_image_size, interpolation=cv2.INTER_NEAREST)
        if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
            return train_img
        else: # To handle a black and white image, increase dimensions to 3
            stacked_img = np.stack((train_img,)*3, axis=-1)
            return stacked_img

    def getNormalMask(self,
                    imageObj, 
                    classes, 
                    catIds, 
                    input_image_size):
        
        annIds = self.coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        cats = self.coco.loadCats(catIds)
        train_mask = np.zeros(input_image_size)
        for a in range(len(anns)):
            className = self.getClassName(anns[a]['category_id'], cats)
            pixel_value = classes.index(className)+1 
            new_mask = cv2.resize(self.coco.annToMask(anns[a])*pixel_value, input_image_size, interpolation=cv2.INTER_NEAREST)
            train_mask = np.maximum(new_mask, train_mask)

        # Add extra dimension for parity with train_img size [X * X * 3]
        train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
        return train_mask  
        
    def getBinaryMask(self,
                    imageObj, 
                    catIds, 
                    input_image_size):
         
        annIds = self.coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        train_mask = np.zeros(input_image_size)
        for a in range(len(anns)):
            new_mask = cv2.resize(self.coco.annToMask(anns[a]), input_image_size)
            #Threshold because resizing may cause extraneous values
            new_mask[new_mask >= 0.5] = 1
            new_mask[new_mask < 0.5] = 0
            train_mask = np.maximum(new_mask, train_mask)
            train_mask = np.where(train_mask > 0.5, 1.0, 0.0)

        # Add extra dimension for parity with train_img size [X * X * 3]
        train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
        return train_mask
    
    def dataGeneratorCoco(
            self,
            filterClasses: Union[list, None]=None,  
            input_image_size: tuple=(224,224), 
            batch_size: Union[int, None]=None, 
            mask_type: Union[Literal['binary'], Literal['normal']]='normal'):
        import random
        img_folder = self.dataDir
        images = self.loadCOCOImages(filterClasses=filterClasses)
        dataset_size = len(self.images)
        catIds = self.coco.getCatIds(catNms=filterClasses)
        batch_size = dataset_size if batch_size is None else batch_size

        c = 0
        while(True):
            
            img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
            mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float') 
            # \ if mask_type == 'binary' else np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
            filename = ['']*batch_size
            print(f"Retrieving COCO images and masks")
            for i in tqdm(range(c, c+batch_size)): #initially from 0 to batch_size, when c = 0
                imageObj = images[i]
                # Retrieve Image
                train_img = self.getImage(imageObj, img_folder)
                # Create Mask
                if mask_type=="binary":
                    train_mask = self.getBinaryMask(imageObj, catIds, input_image_size)
                elif mask_type=="normal":
                    train_mask = self.getNormalMask(imageObj, filterClasses, catIds, input_image_size)   
                # Add to respective batch sized arrays
                img[i-c] = train_img
                mask[i-c] = train_mask
                filename[i-c] = imageObj['file_name']
                
            c+=batch_size
            if(c + batch_size >= dataset_size):
                c=0
                random.shuffle(self.images)
            yield img, mask, filename

    def saveImgsAndMaps(self,
                        gen):

        img, mask, filename = next(gen)
        batch = len(mask)

        print(f"Saving COCO images and masks")
        # Save mask with the same name and extension as the image
        for m in tqdm(range(batch)):
            img_name = os.path.splitext(filename[m])[0]
            # As values of arr_img are floating point between 0 and 1, multiply by 255 and cast to int
            arr_img = (img[m] * 255).astype(np.uint8) 
            # As values of arr_mask are in the range from 0 to 4, just convert to int
            arr_mask = (mask[m]).astype(np.uint8) 
            # Transform to 3 channels image
            rgb_mask = cv2.cvtColor(arr_mask, cv2.COLOR_GRAY2RGB)
            # Save masks as npz files in the mask npz folder
            # np.savez_compressed(os.path.join(self.dataset_maps_npz_folder, img_name), arr_mask)

            # Replace unique tuples with colors in the range 0-255 (for 3 channels)
            from config import colors
            for i in range(len(colors)):
                rgb_mask[np.where((rgb_mask  == [i,i,i]).all(axis = 2))] = colors[i]

            # Save masks
            masks_path = os.path.join(self.dataset_maps_folder, f'{img_name}.png')
            # Save masks as png files
            cv2.imwrite(masks_path, rgb_mask)
            # Save images
            imgs_path = os.path.join(self.dataset_imgs_folder, f'{img_name}.png')
            # Convert to BGR for cv2
            arr_img = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)
            # Save images as npz files in the image npz folder
            np.savez_compressed(os.path.join(self.dataset_imgs_npz_folder, img_name), arr_img)
            cv2.imwrite(imgs_path, arr_img)
                 
        print(f'Images saved in {self.dataset_imgs_folder}')
        print(f'Maps saved in {self.dataset_maps_folder}')
