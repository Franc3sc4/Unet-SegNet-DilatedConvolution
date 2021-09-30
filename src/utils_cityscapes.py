import os
import cv2
import numpy as np
import albumentations as album
from tensorflow import keras



LABELS_name = {'unlabeled' : (  0,  0,  0),
    'ego vehicle'          : (  0,  0,  0),
    'rectification border' : (  0,  0,  0),
    'out of roi'           : (  0,  0,  0),
    'static'               : (  0,  0,  0),
     'dynamic'              : (111, 74,  0),
     'ground'               : ( 81,  0, 81),
     'road'                 : (128, 64,128),
     'sidewalk'             : (244, 35,232),
     'parking'              : (250,170,160),
     'rail track'           : (230,150,140),
     'building'             : ( 70, 70, 70),
     'wall'                 : (102,102,156),
     'fence'                : (190,153,153),
     'guard rail'           : (180,165,180),
     'bridge'               : (150,100,100),
     'tunnel'               : (150,120, 90),
     'pole'                 : (153,153,153),
     'polegroup'            : (153,153,153),
     'traffic light'        : (250,170, 30),
     'traffic sign'         : (220,220,  0),
     'vegetation'           : (107,142, 35),
     'terrain'              : (152,251,152),
     'sky'                  : ( 70,130,180),
     'person'               : (220, 20, 60),
     'rider'                : (255,  0,  0),
     'car'                  : (  0,  0,142),
     'truck'                : (  0,  0, 70),
     'bus'                  : (  0, 60,100),
     'caravan'              : (  0,  0, 90),
     'trailer'              : (  0,  0,110),
     'train'                : (  0, 80,100),
     'motorcycle'           : (  0,  0,230),
     'bicycle'              : (119, 11, 32),
     'license plate'        : (  0,  0,142)
}



LABELS = {0 : (0, 0, 0),
         1 : (0,  0,  0),
         2 : (0,  0,  0),
         3 : (0,  0,  0),
         4 : (0,  0,  0),
         5  : (111, 74,  0),
         6  : ( 81,  0, 81),
         7  : (128, 64, 128),
         8  : (244, 35, 232),
         9  : (250, 170, 160),
         10 : (230, 150, 140),
         11 : ( 70, 70, 70),
         12 : (102, 102, 156),
         13 : (190, 153, 153),
         14 : (180, 165, 180),
         15 : (150, 100, 100),
         16 : (150, 120, 90),
         17 : (153, 153, 153),
         18 : (153, 153, 153),
         19 : (250, 170, 30),
         20 : (220, 220,  0),
         21 : (107, 142, 35),
         22 : (152, 251, 152),
         23 : (70, 130, 180),
         24 : (220, 20, 60),
         25 : (255, 0, 0),
         26 : (0, 0,142),
         27 : (0, 0, 70),
         28 : (0, 60, 100),
         29 : (0, 0, 90),
         30 : (0, 0, 110),
         31 : (0, 80, 100),
         32 : (0, 0, 230),
         33 : (119, 11, 32),
         -1 : (0, 0, 142)
}

grouped_labels = [[0, 1,2,3,4,5,6],[7,8,9,10],[11,12,13,14,15,16],[17,18,19,20],[21,22],[24,25], [26,27,28,29,32,33, -1]]
mapping_labels = {0: ((0,0,0),[0,1,2,3,4,5,6]), 1: ((128,64,128),[7,8,9,10]), 
                  2: ((70,70,70), [11,12,13,14,15,16]), 3:((153,153,153), [17,18,19,20]),
                  4: ((107,142,35), [21,22]), 5:((70,130,180),[23]), 6:((220,20,60), [24,25]),
                  7:((0,0,142), [26,27,28,29,32,33, -1])}


def preprocess_cityscape(images_paths, mask_images_paths, dim, augmentation_fn=None):
    """
        Load cityscape images and resize
    """
    set_images = []
    set_masks = []
    for img_path, mask_path in zip(images_paths, mask_images_paths):
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (dim[0], dim[1]))
        mask = cv2.resize(mask, (dim[0], dim[1]), interpolation=cv2.INTER_NEAREST_EXACT).astype('float')
        if augmentation_fn is not None:
            result_aug = augmentation_fn(image=image, mask=mask)
            image = result_aug['image']
            mask = result_aug['mask']

        set_images.append(image)
        set_masks.append(mask)

    return set_images, set_masks

# Perform colour coding on the reverse-one-hot outputs

def colour_code_segmentation(image):
    """
    colorization of the segmented image
    """
    w = image.shape[0]
    h = image.shape[1]
    new_img = np.zeros((w,h,3))
    
    for clm in range(h):
        for row in range(w):
            value = int(image[row,clm])
            pixels =  mapping_labels[value][0]  # get ids
            new_img[row,clm,0] = pixels[0]
            new_img[row,clm,1] = pixels[1]
            new_img[row,clm,2] = pixels[2]

    return new_img.astype(int)

class Dataset:
    """
    @param: classes_values: class labels
    """
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_values=grouped_labels,
            augmentation=None, 
            preprocessing=None,
            resize=(256,256)
    ):
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.masks_fps = list(filter(lambda mask: mask.find("labelIds") != -1, self.masks_fps))
        
        self.class_values = class_values
        
        self.resize=resize
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resize[0], self.resize[1]))
        mask = cv2.imread(self.masks_fps[i], 0) #0: grey scale mode
        mask = cv2.resize(mask, (self.resize[0], self.resize[1]),
                          interpolation=cv2.INTER_NEAREST_EXACT).astype('float')
        
        # extract certain classes from mask (e.g. cars)
        masks = [np.isin(mask,np.array(v)) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.images_fps)
    
    def get_all_data(self):
        images = []
        masks = []
        for i in range(len(self.images_fps)):
            image, mask = self.__getitem__(i)
            images.append(image)
            masks.append(mask)
        return np.array(images), np.array(masks)

    def convert_to_single_mask(self, mask):
        new_mask = np.zeros((mask.shape[0], mask.shape[1]))
        for class_pos in range(mask.shape[2]): 
            tmp = mask[:, :, class_pos].copy()
            tmp [tmp==1] = class_pos
            new_mask = new_mask + tmp
        return new_mask
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return (batch[0], batch[1])
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
            
def get_preprocessing(preprocessing_fn):
    
    
    _transform = [
        album.Lambda(image=preprocessing_fn),
    ]
    return album.Compose(_transform)

def get_training_augmentation(resize):
    train_transform = [

        album.HorizontalFlip(p=0.5),

        album.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        album.PadIfNeeded(min_width=resize[0], min_height=resize[1], always_apply=True, border_mode=0),
        album.RandomCrop(width=resize[0], height=resize[1], always_apply=True),

        album.IAAAdditiveGaussianNoise(p=0.2),
        album.IAAPerspective(p=0.5),

        album.OneOf(
            [
                album.CLAHE(p=1),
                album.RandomBrightness(p=1),
                album.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        album.OneOf(
            [
                album.IAASharpen(p=1),
                album.Blur(blur_limit=3, p=1),
                album.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        album.OneOf(
            [
                album.RandomContrast(p=1),
                album.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        album.Lambda(mask=round_clip_0_1)
    ]
    return album.Compose(train_transform)

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)