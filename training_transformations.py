#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
from PIL import Image

# define the brightness transformation function (from Python/digital_image_processing GitHub repo):

def change_brightness(img: Image.Image, level: float) -> Image.Image:
    """
    Change the brightness of a PIL Image to a given level.
    """
    def brightness(c: int) -> float:
        """
        Fundamental Transformation/Operation that'll be performed on
        every bit.
        """
        return 128 + level + (c - 128)
    if not -255.0 <= level <= 255.0:
        raise ValueError("level must be between -255.0 (black) and 255.0 (white)")
    return img.point(brightness)

# define the contrast transformation function (from Python/digital_image_processing GitHub repo):

def change_contrast(img: Image.Image, level: int) -> Image.Image:
    """
    Function to change contrast
    """
    if not -255 <= level <= 255:
        raise ValueError("Contrast level must be between -255 and 255")

    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c: int) -> int:
        """
        Fundamental Transformation/Operation that'll be performed on
        every bit.
        """
        return int(128 + factor * (c - 128))

    return img.point(contrast)


# define a function to transform the original Tiny ImageNet training images through the specified brightness levels that we want

def transform_train_image_brightness(train_root, out_root_base, brightness_levels):
    """
    Function that walks through each class folder in the Tiny ImageNet training subset and
    performs brightness transformations at each specified level. Saves the transformed images
    to the path specified in out_root_base
    train_root: Path to original training root (contains one subfolder per class)
    out_root_base: Base directory under which transformed folders will be made
    brightness_levels: List of brightness percentages(we used [-100, -50, 50, 100])

    The for loop iterates through each brightness level, builds the path of the output directory for that level,
    iterates through each image in each class subfolder in the training subset (Tiny ImageNet training folder is organized with
    subfolders for each object class), applies the brightness transformations to each image, and saves the transformed images in
    appropriate directory
    """
    for level in brightness_levels:
        destination_base = os.path.join(out_root_base, f"train_bright_{int(level)}")
        
        for class_name in os.listdir(train_root):
            src_class_dir = os.path.join(train_root, class_name, "images")
            destination_class_dir = os.path.join(destination_base, class_name)
            os.makedirs(destination_class_dir, exist_ok=True)

            for fname in os.listdir(src_class_dir):
                src_path = os.path.join(src_class_dir, fname)
                destination_path = os.path.join(destination_class_dir, fname)

                with Image.open(src_path) as img:
                    bright = change_brightness(img, level)
                    bright.save(destination_path)

# calling the transform_train_image_brightness function. File paths are local

if __name__ == "__main__":
    train_root    = "/Users/charlotteimbert/Documents/tiny-imagenet-200/train"
    out_root_base = "/Users/charlotteimbert/Documents/"
    brightness_levels = [-100, -50, 50, 100]  
    transform_train_image_brightness(train_root, out_root_base, brightness_levels)

# carry out a similar process for the contrast transformations

def transform_train_image_contrast(train_root, out_root_base, contrast_levels):
    """
    Function that walks through each class folder in the Tiny ImageNet training subset and
    performs contrast transformations at each specified level. Saves the transformed images
    to the path specified in out_root_base
    train_root: Path to original training root (contains one subfolder per class)
    out_root_base: Base directory under which transformed folders will be made
    contrast_levels: List of contrast percentages(we used [-100, -50, 50, 100])
    """
    """
    The for loop iterates through each brightness level, builds the path of the output directory for that level,
    iterates through each image in each class subfolder in the training subset (Tiny ImageNet training folder is organized with
    subfolders for each object class), applies the contrast transformations to each image, and saves the transformed images in
    appropriate directory
    """
    for level in contrast_levels:
        destination_base = os.path.join(out_root_base, f"train_contrast_{int(level)}")
        
        for class_name in os.listdir(train_root):
            src_class_dir = os.path.join(train_root, class_name, "images")
            destination_class_dir = os.path.join(destination_base, class_name)
            os.makedirs(destination_class_dir, exist_ok=True)

            for fname in os.listdir(src_class_dir):
                src_path = os.path.join(src_class_dir, fname)
                destination_path = os.path.join(destination_class_dir, fname)

                with Image.open(src_path) as img:
                    contrast = change_contrast(img, level)
                    contrast.save(destination_path)
                    
# calling the transform_train_image_contrast function. File paths are local

if __name__ == "__main__":
    train_root = "/Users/charlotteimbert/Documents/tiny-imagenet-200/train"
    out_root_base = "/Users/charlotteimbert/Documents/"
    contrast_levels = [-100, -50, 50, 100]

    transform_train_image_contrast(train_root, out_root_base, contrast_levels)

