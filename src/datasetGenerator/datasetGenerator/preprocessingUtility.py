# -*- coding: utf-8 -*-
"""
Methods for pre-processing of the images used for the generator.
Methods for checking for and correcting errors in groundtruth values.

@author: Daniel Schneider
"""

import re
import numpy as np
import random
import os
from PIL import Image
from typing import List, Tuple


# Background image preprocessing #######

def enlarge_backgrounds(width: int, height: int, backgrounds: List[Image.Image], blend_range: int = 25) \
        -> List[Image.Image]:
    """
    Enlarges background images by pasting and blending the same image multiple times up to the wanted size

    # Arguments
        width: int. the wanted image width
        height: int. the wanted image height
        backgrounds: List[Image]. the images to enlarge
        blendRange:int. the amount of overlap and blending

    # Returns
        List[Image] the enlarged images
    """

    backgrounds_out = []

    for background in backgrounds:
        background_out = Image.new('RGBA', (width, height), 0)

        background_exp = expand_image(background, blend_range)

        for x in range(0, width, background.width):
            for y in range(0, height, background.height):
                background_out.paste(background, (x, y))

        for x in range(0, width, background.width):
            for y in range(0, height, background.height):
                background_exp_off = Image.new('RGBA', (width, height), 0)
                background_exp_off.paste(background_exp, (x - blend_range, y - blend_range))

                background_out = Image.alpha_composite(background_out, background_exp_off)

        backgrounds_out.append(background_out)

    return backgrounds_out


def expand_image(img: Image.Image, blend_range: int, blend_inner_range: int = 3) -> Image.Image:
    """
    Expands an image by adding a border that fades to transparent over a specified range

    # Arguments
        img: Image. the image to expand
        blend_range: int. the size of the blending border
        blend_inner_range: int. the maximum number of pixels from the inner image used for blending

    # Returns
        Image. the expanded image
    """

    output = Image.new('RGBA', (img.width + 2 * blend_range, img.height + 2 * blend_range), 0)
    output.paste(img, (blend_range, blend_range))
    output_array = np.array(output)

    blend_alpha = int(blend_inner_range / 2)

    for x in range(blend_range, output_array.shape[0] - blend_range):
        for y in range(blend_range + blend_alpha):

            if y < blend_range:
                blend_inner = random.randint(1, min(blend_range - y, blend_inner_range))

                weights = [random.uniform(0, 10) for _ in range(blend_inner)]
                output_array[x, y] = np.average(output_array[x, blend_range:blend_range + blend_inner],
                                                axis=0, weights=weights)

                weights = [random.uniform(0, 10) for _ in range(blend_inner)]
                output_array[x, output_array.shape[1] - 1 - y] = np.average(
                    output_array[x, output_array.shape[1] - blend_range - blend_inner: output_array.shape[1] - blend_range],
                    axis=0, weights=weights)

            output_array[x, y, 3] = output_array[x, output_array.shape[1] - 1 - y, 3] \
                = calculate_alpha(y, blend_range + blend_alpha)

    for y in range(0, output_array.shape[1]):
        for x in range(blend_range + blend_alpha):

            if x < blend_range:
                blend_inner = random.randint(1, min(blend_range - x, blend_inner_range))

                weights = [random.uniform(0, 10) for _ in range(blend_inner)]
                output_array[x, y] = np.average(output_array[blend_range:blend_range + blend_inner, y],
                                                axis=0, weights=weights)

                weights = [random.uniform(0, 10) for _ in range(blend_inner)]
                output_array[output_array.shape[0] - 1 - x, y] = np.average(
                    output_array[output_array.shape[0] - blend_range - blend_inner:output_array.shape[0] - blend_range, y],
                    axis=0, weights=weights)

            if y < blend_range:
                output_array[x, y, 3] = output_array[output_array.shape[0] - 1 - x, y, 3] \
                    = calculate_alpha(x * y, blend_range * blend_range + blend_alpha)
            elif y > output_array.shape[1] - blend_range:
                output_array[x, y, 3] = output_array[output_array.shape[0] - 1 - x, y, 3] \
                    = calculate_alpha(x * (output_array.shape[1] - y), blend_range * blend_range + blend_alpha)
            else:
                output_array[x, y, 3] = output_array[output_array.shape[0] - 1 - x, y, 3] \
                    = calculate_alpha(x, blend_range + blend_alpha)

    output_array[blend_range:output_array.shape[0] - blend_range, blend_range:output_array.shape[1] - blend_range, 3] = 0

    return Image.fromarray(output_array)


def calculate_alpha(x: int, blend_range: int) -> float:
    """
    Calculates the blending alpha value for a given distance
    
    # Arguments
        x: int. the distance to calculate the apha value for
        blend_range: int. the size of the blending border
    
    # Returns
        float. the calculated alpha value 
    """
    
    return x*x * 127 / (blend_range * blend_range)


# Character Image preprocessing #######

def crop_image_to_non_transparent_areas_folder(folder_path: str, subfolder_old: str, subfolder_new: str) -> None:
    """
    Crops transparent areas from all train images in a given folder structure
    and places the results in another subfolder
    
    # Arguments
        folder_path: str. the base path of the source images
        subfolder_old: str. the subfolder to load the images from
        subfolder_new: str. the subfolder to save the cropped images to
    """
    
    folder_file_list = os.listdir(folder_path)
    
    if subfolder_old in folder_file_list:
        f_path = os.path.join(folder_path, subfolder_old)
        o_path = os.path.join(folder_path, subfolder_new)
        if not os.path.exists(o_path):
            os.makedirs(o_path)
            
        for ff in os.listdir(f_path):
            ff_path = os.path.join(f_path, ff)
            oo_path = os.path.join(o_path, ff)
            
            if not(os.path.isdir(ff_path)) and not(ff.startswith('_')):
                crop_image_to_non_transparent_areas(ff_path, oo_path)
    else:
        for f in folder_file_list:
            f_path = os.path.join(folder_path, f)
            if not(f.startswith('_')) and os.path.isdir(f_path):
                crop_image_to_non_transparent_areas_folder(f_path, subfolder_old, subfolder_new)


def crop_image_to_non_transparent_areas(path_in: str, path_out: str) -> None:
    """
    Crops transparent areas from an image and saves the result to the specified path
    
    # Arguments
        path_in: str. the path to load the image file from
        path_out:str. the path to save the cropped image to
    """
    
    img = Image.open(path_in)
    non_transparent = np.where(np.asarray(img)[:, :, 3] > 0)
    img_out = img.crop([min(non_transparent[1]), min(non_transparent[0]),
                        max(non_transparent[1]), max(non_transparent[0])])

    print(path_in, path_out)
    img_out.save(path_out)


# Check for and correct errors in groundtruth values ########

def add_gt_leading_and_trailing_spaces(input_path: str, indices: Tuple[int, int],
                                       output_path: str, files_per_folder: int = 1000) -> None:
    """ Updates the groundtruth files and adds a leading and trailing space to them
        
    # Arguments
        input_path: str. the path to load the groundtruth data from
        indices: Tuple[int, int]. the start and stop index of the images to load
        output_path: str. the path to save the updated groundtruth data to
        files_per_folder: int. the number of files in the subfolders the dataset is grouped into
    """
    
    for i in range(indices[0], indices[1]):
        
        file_path = os.path.join(str((i // files_per_folder) * files_per_folder), str(i) + ".txt")
        with open(os.path.join(output_path, file_path), 'w') as fo:
            with open(os.path.join(input_path, file_path), 'r') as fi:
                fo.writelines(["".join([" ", x.strip(), " ", "\n"]) for x in fi.readlines()])


def find_incorrect_gt_files(input_folder: str, input_ids: List[int], labels: Tuple[List[str], List[str]],
                            files_per_folder: int) -> None:
    """
    Searches for incorrect gt-files (unknown labels or double spaces).
    Outputs the id and the incorrect label of each error found.

    # Arguments
        input_folder: the folder to search gt-files in
        input_ids: List[int]. the ids of the files to check
        labels: Tuple[List[str], List[str]]. the labels to check against
        files_per_folder: int. the number of files in each subfolder of the dataset
    """

    for i in input_ids:
        file = os.path.join(input_folder, str((i // files_per_folder) * files_per_folder), str(i) + ".txt")

        with open(file, 'r') as f:
            lines = f.read().splitlines()

            top_line_array = list(filter(None, re.split("(\s+)", lines[0])))
            bot_line_array = list(filter(None, re.split("(\s+)", lines[1])))

            for t in top_line_array:
                if t not in labels[0]:
                    print(i, "top:", t)

            if '  ' in top_line_array:
                print(i, "top:", 'double space')

            for b in bot_line_array:
                if b not in labels[1]:
                    print(i, "bot:", b)

            if '  ' in bot_line_array:
                print(i, "bot:", 'double space')
