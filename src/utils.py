import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from patchify import patchify
import albumentations as album

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def get_images_paths(images_dir, masks_dir):
    image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
    mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
    return image_paths, mask_paths

def get_patches(image, patch_width, patch_heigth, channels):
    patches = patchify(image, (patch_width, patch_heigth, channels), step=patch_width) #patchify returns an ndarray matrix of images. i.e [3, 3, 1, 500, 500, 3]
    #thus a reshape is needed, i.e [9, 500, 500, 3] shape as list of images .
    patches = patches.reshape((patches.shape[0]*patches.shape[1], patch_width, patch_heigth, channels))
    return patches

# helper function for data visualization
def visualize(**images, path_to_save=None):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def load_preprocess_images(images_paths, mask_images_paths, patch_size, dim, mask_channels, class_rgb_values, augmentation_fn=None):
    """
        Load images and rescales them to dim 
    """
    set_images = []
    set_masks = []
    for img_path, mask_path in zip(images_paths, mask_images_paths):
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, class_rgb_values).astype('float')
        if patch_size is not None:
            img_patches = get_patches(image, patch_size[0], patch_size[1], 3) #[500, 500, 3]
            mask_patches = get_patches(mask, patch_size[0], patch_size[1], mask_channels)
            for i in range(img_patches.shape[0]):
                if dim is not None:
                    image_patch = cv2.resize(img_patches[i], (dim[0],dim[1]))
                    mask_patch = cv2.resize(mask_patches[i], (dim[0],dim[1]), interpolation=cv2.INTER_NEAREST_EXACT)
                else:
                    image_patch = img_patches[i]
                    mask_patch = mask_patches[i]

                if augmentation_fn is not None:
                    result_aug = augmentation_fn(image=image_patch, mask=mask_patch)
                    image_patch = result_aug['image']
                    mask_patch = result_aug['mask']

                set_images.append(image_patch)
                set_masks.append(mask_patch)
        else:
            image = cv2.resize(image, (dim[0],dim[1]))
            mask = cv2.resize(mask, (dim[0],dim[1]), interpolation=cv2.INTER_NEAREST_EXACT)
            if augmentation_fn is not None:
                result_aug = augmentation_fn(image=image, mask=mask)
                image = result_aug['image']
                mask = result_aug['mask']
            
    return set_images, set_masks


def get_training_augmentation(prob=0.5):
    train_transform = [
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
                #album.ShiftScaleRotate(p=1, scale_limit=1.0)
            ],
            p=prob,
        ),
    ]
    return album.Compose(train_transform)

################################################## AUGMENTATION #####################

def get_validation_augmentation():   
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)