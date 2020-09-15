# -*- coding: utf-8 -*-
"""
Methods used to augment images.
The augmentations are initialized from a dictionary specifying the magnitude of each operation

@author: Daniel Schneider
"""

from PIL import Image, ImageDraw
import numpy as np
import Augmentor
from typing import List, Tuple, Dict

from datasetGenerator.augmentation.randomNoiseAugmentor import RandomNoise, RandomSaltAndPepper
from datasetGenerator.augmentation.transformAugmentor import ShearFloatPrecision, Scale


defaultParams = {
        'color': 'none',        # none, small, medium, large
        'distortion': 'none',   # none, small, medium, large
        'noise': 'none',        # none, small, medium, large
        'rotate': 'none',       # none, standard, standard_crop
        'scale': 'none',        # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'none'         # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }


def init_augmentor(images: List[List[np.array]], augment_params: Dict[str, str]) -> Augmentor.DataPipeline:
    """
    Initializes the augmentor pipeline for a given list of images (as numpy arrays)
    and the specified augmentation parameters
    
    # Arguments
        images: List[List[np.array]]. a list of images to apply augmentations to (as numpy arrays).
                                For each image additional mask images like bounding boxes can be specified
                                which will recieve the same augmentations as the image itself
        augment_params: Dict[str, str]. the augmentations to be performed.
                                        Allowed keys are:
                                        'color':        'none', 'small', 'medium', 'large'
                                        'distortion':   'none', 'small', 'medium', 'large'
                                        'noise':        'none', 'small', 'medium', 'large'
                                        'rotate':       'none', 'standard', 'standard_crop'
                                        'scale':        'none', 'x_small', 'x_medium', 'x_large',
                                                                'both_small', 'both_medium', 'both_large'
                                        'shear':        'none', 'x_small', 'x_medium', 'x_large',
                                                                'y_small', 'y_medium', 'y_large',
                                                                'both_small', 'both_medium', 'both_large'
    
    # Returns
        DataPipeline. the augmentation pipeline used to generate augmented images
         
    """
    
    params = {**defaultParams, **augment_params}
    
    p = Augmentor.DataPipeline(images)
    
    # color, brightness, contrast
    if params['color'] == 'small':
        p.random_color(0.3, min_factor=0.95, max_factor=1.05)
        p.random_brightness(0.3, min_factor=0.95, max_factor=1.05)
        p.random_contrast(0.3, min_factor=0.95, max_factor=1.05)
    elif params['color'] == 'medium':
        p.random_color(0.4, min_factor=0.9, max_factor=1.1)
        p.random_brightness(0.4, min_factor=0.9, max_factor=1.1)
        p.random_contrast(0.4, min_factor=0.9, max_factor=1.1)
    elif params['color'] == 'large':
        p.random_color(0.4, min_factor=0.85, max_factor=1.15)
        p.random_brightness(0.4, min_factor=0.9, max_factor=1.1)
        p.random_contrast(0.4, min_factor=0.85, max_factor=1.15)

    # distortion
    if params['distortion'] == 'small':
        p.random_distortion(0.4, grid_width=4, grid_height=4, magnitude=1)
    elif params['distortion'] == 'medium':
        p.random_distortion(0.4, grid_width=10, grid_height=5, magnitude=2)
    elif params['distortion'] == 'large':
        p.random_distortion(0.4, grid_width=15, grid_height=5, magnitude=2)

    # noise, salt&pepper
    if params['noise'] == 'small':
        p.add_operation(RandomNoise(0.25, 0.002))
        p.add_operation(RandomSaltAndPepper(0.2, 0.001, 0.001))
    elif params['noise'] == 'medium':
        p.add_operation(RandomNoise(0.25, 0.006))
        p.add_operation(RandomSaltAndPepper(0.2, 0.003, 0.003))
    elif params['noise'] == 'large':
        p.add_operation(RandomNoise(0.25, 0.01))
        p.add_operation(RandomSaltAndPepper(0.2, 0.005, 0.005))

    # rotate
    if params['rotate'] == 'standard':
        p.rotate_without_crop(0.4, max_left_rotation=2, max_right_rotation=2, expand=True)
    elif params['rotate'] == 'standard_crop':
        p.rotate_without_crop(0.4, max_left_rotation=2, max_right_rotation=2, expand=False)
    
    # scale
    if params['scale'] == 'x_small':
        p.add_operation(Scale(0.3, min_scale_x=0.92, max_scale_x=1.02, min_scale_y=0.99, max_scale_y=1.01))
    elif params['scale'] == 'x_medium':
        p.add_operation(Scale(0.3, min_scale_x=0.85, max_scale_x=1.1, min_scale_y=0.95, max_scale_y=1.05))
    elif params['scale'] == 'x_large':
        p.add_operation(Scale(0.3, min_scale_x=0.8, max_scale_x=1.2, min_scale_y=0.95, max_scale_y=1.05))
    
    elif params['scale'] == 'both_small':
        p.add_operation(Scale(0.3, min_scale_x=0.95, max_scale_x=1.05, min_scale_y=0.95, max_scale_y=1.05))
    elif params['scale'] == 'both_medium':
        p.add_operation(Scale(0.3, min_scale_x=0.9, max_scale_x=1.1, min_scale_y=0.9, max_scale_y=1.1))
    elif params['scale'] == 'both_large':
        p.add_operation(Scale(0.3, min_scale_x=0.8, max_scale_x=1.2, min_scale_y=0.8, max_scale_y=1.2))

    # shear, skew
    if params['shear'] == 'x_small':
        p.skew_left_right(0.3, magnitude=0.005)
        p.add_operation(ShearFloatPrecision(0.4, max_shear_left=1.0, max_shear_right=0.05))
    elif params['shear'] == 'x_medium':
        p.skew_left_right(0.3, magnitude=0.01)                                                  # final
        p.add_operation(ShearFloatPrecision(0.4, max_shear_left=2.0, max_shear_right=0.25))
    elif params['shear'] == 'x_large':
        p.skew_left_right(0.3, magnitude=0.015)
        p.add_operation(ShearFloatPrecision(0.4, max_shear_left=3.0, max_shear_right=0.5))
    
    elif params['shear'] == 'y_small':
        p.add_operation(ShearFloatPrecision(0.4, max_shear_left=0.05, max_shear_right=1.0))     # measure, border
    elif params['shear'] == 'y_medium':
        p.add_operation(ShearFloatPrecision(0.4, max_shear_left=0.25, max_shear_right=2.0))
    elif params['shear'] == 'y_large':
        p.add_operation(ShearFloatPrecision(0.4, max_shear_left=0.5, max_shear_right=3.0))
    
    elif params['shear'] == 'both_small':
        p.skew_left_right(0.3, magnitude=0.005)
        p.add_operation(ShearFloatPrecision(0.4, max_shear_left=1.0, max_shear_right=1.0))      # part
    elif params['shear'] == 'both_medium':
        p.skew_left_right(0.3, magnitude=0.01)
        p.add_operation(ShearFloatPrecision(0.4, max_shear_left=2.0, max_shear_right=2.0, keep_size=True))
    elif params['shear'] == 'both_large':
        p.skew_left_right(0.3, magnitude=0.015)
        p.add_operation(ShearFloatPrecision(0.4, max_shear_left=3.0, max_shear_right=3.0, keep_size=True))

    return p
    
    
def augment_images(img_aug: Image.Image, augment_num:int, augment_params: Dict[str, str]) -> List[Image.Image]:
    """
    Performs augmentations specified by the given parameters on a given image and samples the specified number of images.
    
    # Arguments
        img: Image. the image to augment
        augmentNum: int.  number of augmentation samples to create
        augmentParams: Dict[str, str]. the augmentations to be performed
    
    # Returns
        List[Image]. the augmented images
    """
    
    images = [[np.asarray(img_aug)]]
    
    p = init_augmentor(images, augment_params)
    
    sample_x = p.sample(augment_num)
    output_images = []
    for s in sample_x:
        img_aug = Image.fromarray(s[0])
        # display(img)
        output_images.append(img_aug)
    
    return output_images


def augment_image(img: Image.Image, augment_params: Dict[str, str]) -> Image.Image:
    """
    Performs augmentations specified by the given parameters on a given image.
    
    # Arguments
        img: Image. the image to augment
        augmentParams: Dict[str, str]. the augmentations to be performed
    
    # Returns
        Image. the augmented image
    """
    
    images = [[np.asarray(img)]]
    
    p = init_augmentor(images, augment_params)
    
    sample_x = p.sample(1)
    sample = sample_x[0][0]
    img_aug = Image.fromarray(sample)
    
    return img_aug


def augment_image_bboxes(img: Image.Image, bboxes:List[Tuple[int, int, int, int]], augment_params: Dict[str, str]) \
        -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    """
    Performs augmentations specified by the given parameters on the given image and its bounding boxes.
    
    # Arguments
        img: Image. the image to augment
        bboxes: List[Tuple[int,int,int,int]]. the bounding boxes for each element in the image
        augmentParams: Dict[str, str]. the augmentations to be performed
    
    # Returns
        Image. the augmented image
        List[Tuple[int,int,int,int]]]. the updated bounding boxes for the augmented image
    """

    bbox_imgs = []
    
    bboxes_aug = [[[] for _ in range(len(i))] for i in bboxes]
    
    for bbox_row in bboxes:
        for b in bbox_row:
            bbox_img = Image.new('L', (img.width, img.height), 0)
            draw = ImageDraw.Draw(bbox_img)
            draw.rectangle(b, fill=255)
            bbox_imgs.append(bbox_img)

    images = [[np.asarray(i) for i in [img] + bbox_imgs]]
    
    p = init_augmentor(images, augment_params)
    
    sample_x = p.sample(1)
    sample = sample_x[0][0]
    img_aug = Image.fromarray(sample)
    
    sample_boxes = sample_x[0][1:]

    row_idx = 0
    col_idx = 0
    for s in sample_boxes:
        while row_idx < len(bboxes_aug) and col_idx >= len(bboxes_aug[row_idx]):
            row_idx += 1
            col_idx = 0

        idx = np.where(s > 150)
        if idx[0].size == 0:
            idx[0] = np.array([0, img.height])
        if idx[1].size == 0:
            idx[1] = np.array([0, img.width])
        
        bboxes_aug[row_idx][col_idx] = ((min(idx[1]), min(idx[0])), (max(idx[1]), max(idx[0])))
        col_idx += 1
    
    return img_aug, bboxes_aug


def augment_background(img: Image.Image) -> Image.Image:
    """
    Augmentation on a background image (no color, large distortions)
    
    # Arguments
        img: Image. the image to augment
    
    # Returns
        Image. the augmented image
    """
    
    params = {
        'color': 'none',            # none, small, medium, large
        'distortion': 'large',      # none, small, medium, large
        'noise': 'none',            # none, small, medium, large
        'rotate': 'standard_crop',  # none, standard, standard_crop
        'scale': 'both_medium',     # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'both_medium'      # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }
    
    return augment_image(img, params)


def augment_border(img: Image.Image) -> Image.Image:
    """
    Augmentation on a background border image (no color, small distortions)
    
    # Arguments
        img: Image. the image to augment
    
    # Returns
        Image. the augmented image
    """
    
    params = {
        'color': 'none',            # none, small, medium, large
        'distortion': 'small',      # none, small, medium, large
        'noise': 'none',            # none, small, medium, large
        'rotate': 'none',           # none, standard, standard_crop
        'scale': 'both_medium',     # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'y_small'          # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }
    
    return augment_image(img, params)


def augment_background_color(img: Image.Image) -> Image.Image:
    """
    Augmentation on the final background image (color, very small distortions)
    
    # Arguments
        img: Image. the image to augment
    
    # Returns
        Image. the augmented image
    """
    
    params = {
        'color': 'small',           # none, small, medium, large
        'distortion': 'small',      # none, small, medium, large
        'noise': 'small',           # none, small, medium, large
        'rotate': 'none',           # none, standard, standard_crop
        'scale': 'none',            # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'none'             # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }
    
    return augment_image(img, params)


def augment_measure_line(img: Image.Image) -> Image.Image:
    """
    Augmentation on a measure line image (color, small distortions)
    
    # Arguments
        img: Image. the image to augment
    
    # Returns
        Image. the augmented image
    """
    
    params = {
        'color': 'small',           # none, small, medium, large
        'distortion': 'small',      # none, small, medium, large
        'noise': 'none',            # none, small, medium, large
        'rotate': 'none',           # none, standard, standard_crop
        'scale': 'both_medium',     # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'y_small'          # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }
    
    return augment_image(img, params)

    
def augment_image_element(img: Image.Image) -> Image.Image:
    """
    Augmentation on a source image (color, small distortions)
    
    # Arguments
        img: Image. the image to augment
    
    # Returns
        Image. the augmented image
    """
    
    params = {
        'color': 'small',           # none, small, medium, large
        'distortion': 'small',      # none, small, medium, large
        'noise': 'none',            # none, small, medium, large
        'rotate': 'standard',       # none, standard, standard_crop
        'scale': 'x_large',         # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'both_small'       # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }
    
    return augment_image(img, params)


def augment_image_distortion_bb(img: Image.Image, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    """
    Augmentation on a foreground-image with bounding boxes
    
    # Arguments
        img: Image. the image to augment
        bboxes: List[Tuple[int,int,int,int]]. the bounding boxes for each element in the image
        
    # Returns
        Image. the augmented image
        List[Tuple[int,int,int,int]]]. the updated bounding boxes for the augmented image
    """
    
    params = {
        'color': 'none',            # none, small, medium, large
        'distortion': 'medium',     # none, small, medium, large
        'noise': 'none',            # none, small, medium, large
        'rotate': 'none',           # none, standard, standard_crop
        'scale': 'x_small',         # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'x_medium'         # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }
    
    return augment_image_bboxes(img, bboxes, params)


def augment_image_color_noise(img: Image.Image) -> Image.Image:
    """
    Final augmentations on a generated image (color, large distortions, noise)
    
    # Arguments
        img: Image. the image to augment
    
    # Returns
        Image. the augmented image
    """
    
    params = {
        'color': 'large',           # none, small, medium, large
        'distortion': 'large',      # none, small, medium, large
        'noise': 'medium',          # none, small, medium, large
        'rotate': 'none',           # none, standard, standard_crop
        'scale': 'none',            # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'none'             # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }
    
    return augment_image(img, params)


def augment_images_dataset(img: Image.Image, augmentation_num: int) -> List[Image.Image]:
    """
    Augmentations on a generated or real image creating a specified number of augmented images
    (color, large distortions, noise)
    
    # Arguments
        img: Image. the image to augment
        augmentation_num: int.  number of augmentation samples to create
    
    # Returns
        List[Image]. the augmented images
    """
    
    params = {
        'color': 'large',           # none, small, medium, large
        'distortion': 'large',      # none, small, medium, large
        'noise': 'medium',          # none, small, medium, large
        'rotate': 'none',           # none, standard, standard_crop
        'scale': 'x_medium',        # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'x_medium'         # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }
    
    return augment_images(img, augmentation_num, params)
