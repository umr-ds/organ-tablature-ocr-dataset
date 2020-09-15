# -*- coding: utf-8 -*-
"""
Classes for Noise and Deletion  augmentations

@author: Daniel Schneider
"""

import numpy as np
import random
from Augmentor.Operations import Operation
from PIL import Image, ImageFilter
from typing import List


class RandomSaltAndPepper(Operation):
    """
    Class for a random salt and pepper augmentation operation (expands the Operation superclass).
    Skips black and white images because they could be mask images that would be ruined by noise augmentation
         
    # Arguments
        probability: float. the probability for the augmentation to take part
        black_probability: float. the probability for switching a pixel to black
        black_probability: float. the probability for switching a pixel to white
    """
    
    def __init__(self, probability: float, black_probability: float = 0, white_probability: float = 0):
        Operation.__init__(self, probability)
        self.black_offset = black_probability
        self.white_offset = 1 - white_probability

    def perform_operation(self, images: List[Image.Image]) -> List[Image.Image]:
        augmented_images = []
        for image in images:
            augmented_images.append(self.add_random_salt_and_pepper(image))

        return augmented_images

    def add_random_salt_and_pepper(self, image: Image.Image) -> Image.Image:
        """
        Randomly sets pixels to black (0,0,0) or white (255,255,255) according to the specified probabilities
        
        # Arguments
            image: Image. the image to augment
            
        # Returns
            Image. the augmented image
        """
        
        if image.mode == "L":   # skip mask images
            return image
        
        img_array = np.array(image.convert('RGBA'))
        rand_array = np.random.random(img_array.shape[0:2])
        
        img_array[rand_array < self.black_offset] = [0, 0, 0, 255]          # black
        img_array[rand_array > self.white_offset] = [255, 255, 255, 255]    # white
        
        return Image.fromarray(img_array)


class RandomNoise(Operation):
    """
    Class for a random noise augmentation operation (expands the Operation superclass).
    Skips black and white images because they could be mask images that would be ruined by noise augmentation
         
    # Arguments
        probability: float. the probability for the augmentation to take part
        random_color_probability: float. the probability for switching a pixel to a random color
    """

    def __init__(self, probability: float, random_color_probability: float = 0):
        Operation.__init__(self, probability)
        self.randomColor_offset = random_color_probability * 255

    def perform_operation(self, images: List[Image.Image]) -> List[Image.Image]:
        augmented_images = []
        for image in images:
            augmented_images.append(self.add_random_noise(image))

        return augmented_images

    def add_random_noise(self, image: Image.Image) -> Image.Image:
        """
        Randomly sets pixels to a random color according to the specified probability
        
        # Arguments
            image: Image. the image to augment
            
        # Returns
            Image. the augmented image
        """
        
        if image.mode == "L":   # skip mask images
            return image
        
        img_array = np.array(image.convert('RGBA'))
        rand_array_rgba = np.random.uniform(0, 255, img_array.shape)
        
        rand_array_mask = rand_array_rgba[:, :, 3] < self.randomColor_offset
        rand_array_rgba[:, :, 3] = 255
        
        img_array[rand_array_mask] = rand_array_rgba[rand_array_mask]  # random color
        
        return Image.fromarray(img_array)


class RandomDeletion(Operation):
    """
    Class for a random deletion augmentation operation (expands the Operation superclass).
    Skips black and white images because they could be mask images that would be ruined by this kind of augmentation
         
    # Arguments
        probability: float. the probability for the augmentation to take part
        min_size: int. the minimum deletion size 
        max_size: int. the maximum deletion size
        min_blend_range: int. the minimum blend range between deleted and not deleted areas
        max_blend_range: int. the maximum blend range between deleted and not deleted areas
    """

    def __init__(self, probability: float, min_size: int = 0, max_size: int = 9,
                 min_blend_range: int = 0, max_blend_range: int = 9):
        Operation.__init__(self, probability)
        self.size = (min_size, max_size)
        self.blend_range = (min_blend_range, max_blend_range)

    def perform_operation(self, images: List[Image.Image]) -> List[Image.Image]:
        augmented_images = []
        for image in images:
            augmented_images.append(self.add_random_alpha_variation(image))

        return augmented_images
    
    def add_random_alpha_variation(self, image: Image.Image) -> Image.Image:
        """
        Randomly deletes areas of the image by setting their alpha channel to zero
        
        # Arguments
            image: Image. the image to augment
            
        # Returns
            Image. the augmented image
        """
        
        if image.mode == "L":   # skip mask images
            return image
        
        img_array = np.array(image.convert('RGBA'))
        
        center_y = np.random.randint(0, img_array.shape[0])
        center_x = np.random.randint(0, img_array.shape[1])
        
        alpha_array = np.full_like(img_array[:, :, 3], 255)
        alpha_array[center_y, center_x] = 0
        
        alpha_img = Image.fromarray(alpha_array, mode='L')
        r_filter_size = random.randint(self.size[0], self.size[1])
        if r_filter_size % 2 != 1:
            r_filter_size += 1
        r_blur_size = random.randint(self.blend_range[0], self.blend_range[1])
        alpha_img = alpha_img.filter(ImageFilter.MinFilter(r_filter_size)).filter(ImageFilter.BoxBlur(r_blur_size))
        
        new_alpha_array = np.rint(img_array[:, :, 3]*(np.asarray(alpha_img)/255))
        img_array[:, :, 3] = new_alpha_array

        return Image.fromarray(img_array)


class RandomAlphaVariation(Operation):
    """
    Class for a random alpha variation augmentation operation (expands the Operation superclass).
    Skips black and white images because they could be mask images that would be ruined by this kind of augmentation
         
    # Arguments
        probability: float. the probability for the augmentation to take part
        max_filter_size: int. the filter size for the max filter  
        blur_size: int. the filter size for the blur filter  
    """

    def __init__(self, probability: float, max_filter_size: int = 7, blur_size: int = 3):
        Operation.__init__(self, probability)
        self.max_filter_size = max_filter_size
        self.blur_size = blur_size

    def perform_operation(self, images: List[Image.Image]) -> List[Image.Image]:
        augmented_images = []
        for image in images:
            augmented_images.append(self.add_random_alpha_variation(image))

        return augmented_images
    
    def add_random_alpha_variation(self, image: Image.Image) -> Image.Image:
        """
        Randomly changes the alpha value of the image by applying a blur and max filter to a randomized array
        and setting the result array as new alpha channel
        
        # Arguments
            image: Image. the image to augment
            
        # Returns
            Image. the augmented image
        """
        
        if image.mode == "L":   # skip mask images
            return image
        
        img_array = np.array(image.convert('RGBA'))
        rand_alpha_array = np.random.randint(0, 255, img_array.shape[:2])
        
        rand_alpha_img = Image.fromarray(rand_alpha_array, mode='L')
        rand_alpha_img = rand_alpha_img.filter(ImageFilter.MaxFilter(self.max_filter_size))\
            .filter(ImageFilter.BoxBlur(self.blur_size))
        
        new_alpha_array = np.rint(img_array[:, :, 3] * (np.asarray(rand_alpha_img)/255))
        img_array[:, :, 3] = new_alpha_array

        return Image.fromarray(img_array, mode='RGBA')
