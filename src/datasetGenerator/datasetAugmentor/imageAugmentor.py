# -*- coding: utf-8 -*-
"""
Methods for generating datasets from existing tablature images using data augmentation

@author: Daniel Schneider
"""

from PIL import Image
from typing import Tuple, Optional, Dict
import os
import shutil
import random

from datasetGenerator.augmentation.augmentationUtility import augment_images
from datasetGenerator.datasetGenerator.generatorUtility import display_progress


def load_image_from_dataset(input_folder: str, index: int, files_per_folder: int = 1000) \
        -> Tuple[str, Optional[Image.Image]]:
    """
    Loads an image with given index from a dataset folder

    # Arguments
        input_folder: str. the dataset folder
        index: int. the id of the image to load
        files_per_folder: int. the number of files in each subfolder of the dataset

    # Returns
        Image. the loaded image or None if the image is not found
    """
    subfolder_index = str(files_per_folder * (index // files_per_folder))
    input_subfolder = os.path.join(input_folder, subfolder_index)

    input_path = os.path.join(input_subfolder, str(index))

    if os.path.exists(input_path + '.png'):
        with open(input_path + '.png', 'rb') as im:
            img = Image.open(im)
            return input_path, img.copy().convert('RGBA')

    return input_path, None


def save_image_to_dataset(image: Image.Image, groundtruth_path: str,
                          index: int, output_folder: str, files_per_folder: int = 1000) -> None:
    """
    Saves an image and its groundtruth data to a given dataset folder

    # Arguments
        image: Image. the image to save
        meatadata_path: str. the path of the metadata file
        index: int. the index to save the image at
        output_folder: str. the dataset folder
        files_per_folder: int. the number of files in each subfolder of the dataset
    """

    subfolder_index = str(files_per_folder * (index // files_per_folder))
    output_subfolder = os.path.join(output_folder, subfolder_index)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder, exist_ok=True)

    output_path = os.path.join(output_subfolder, str(index))

    image.save(output_path + '.png')
    if os.path.exists(groundtruth_path + '.txt'):
        shutil.copy2(groundtruth_path + '.txt', output_path + '.txt')


def generate_augmented_dataset(input_folder: str, input_indices: Tuple[int, int],
                               output_folder: str, output_index_start: int,
                               augment_num: int, augment_params: Dict[str, str] = {},
                               output_size: Tuple[int, int] = (2000, 128),
                               src_img_max_cutoff: Tuple[int, int, int, int] = (40, 40, 5, 15),
                               files_per_folder: int = 1000, show_progress_bar: bool = False) -> None:
    """
    Loads images and groundtruth data from a given path and generates a dataset from them.
    The images are cropped to the given size and data augmentation is used to generate multiple instances of each image.
    The augmented images are saved to a specified dataset folder.
        
    # Arguments
        input_folder: str. the path to load the images and groundtruth data from
        indices: Tuple[int, int]. the indices of the images to load
        output_folder: str. the path to save the augmented images to
        output_index_start: int. the starting index for numbering the augmented images
        augment_num: int. the number of augmented images to generate from each image
        augment_params: Dict[str, str]. the augmentations to be performed
        output_size: Tuple[int, int]. the size of the output images
        src_img_max_cutoff: Tuple[int, int, int, int]. the max amount of pixels to be cut off during random placement
                                                       of the source images (top, bottom, left, right)
        files_per_folder: int. the number of files in each subfolder of the dataset
        show_progress_bar: bool. indicates whether to display a progress bar in stdout
    """

    print("Augmentation of {} images".format(input_indices[1] - input_indices[0]))

    for i in range(input_indices[0], input_indices[1]):
        if show_progress_bar:
            display_progress(i - input_indices[0], input_indices[1] - input_indices[0], ('', 1, 0))

        input_path, img = load_image_from_dataset(input_folder, i, files_per_folder)

        if img:
            img_aug = augment_images(img, augment_num, augment_params)
            # img_aug = augment_images_dataset(img, augment_num)

            for j in range(augment_num):
                r_i = output_index_start + j

                img_out = Image.new('RGB', output_size, (255, 255, 255))
                x_pos = round(random.uniform(-src_img_max_cutoff[0],
                                             output_size[0] - img_aug[j].width + src_img_max_cutoff[1]))
                y_pos = round(random.uniform(-src_img_max_cutoff[2],
                                             output_size[1] - img_aug[j].height + src_img_max_cutoff[3]))
                img_out.paste(img_aug[j], (x_pos, y_pos), img_aug[j])

                save_image_to_dataset(img_out, input_path, r_i, output_folder, files_per_folder)

            output_index_start += augment_num

    if show_progress_bar:
        display_progress(input_indices[1] - input_indices[0], input_indices[1] - input_indices[0], ('', 1, 0))
        print()
    print("Augmentation finished")


def generate_non_augmented_dataset(input_folder: str, input_indices: Tuple[int, int],
                                   output_folder: str, output_index_start: int, output_size: Tuple[int, int] = (2000, 128),
                                   files_per_folder: int = 1000, show_progress_bar: bool = False) -> None:
    """
    Loads images and groundtruth data from a given path and generates a dataset from them.
    The images are only cropped to the given size and saved to a specified dataset folder.
        
    # Arguments
        input_folder: str. the path to load the images and groundtruth data from
        input_indices: Tuple[int, int]. the start and stop index of the images to load
        output_folder: str. the path to save the final images to
        output_index_start: int. the starting index for numbering the output images
        output_size: Tuple[int,int]. the size of the output images
        files_per_folder: int. the number of files in the subfolders the generated dataset is grouped into
        show_progress_bar: bool. indicates whether to display a progress bar in stdout
    """

    print("Cropping of {} images".format(input_indices[1] - input_indices[0]))

    for i in range(input_indices[0], input_indices[1]):
        if show_progress_bar:
            display_progress(i - input_indices[0], input_indices[1] - input_indices[0], ('', 1, 0))

        input_path, img = load_image_from_dataset(input_folder, i, files_per_folder)

        if img:
            img_out = Image.new('RGB', output_size, (255, 255, 255))

            x_pos = int(output_size[0] / 2 - img.width / 2)
            y_pos = int(output_size[1] / 2 - img.height / 2)

            img_out.paste(img, (x_pos, y_pos), img)

            r_i = output_index_start + i

            save_image_to_dataset(img_out, input_path, r_i, output_folder, files_per_folder)

    if show_progress_bar:
        display_progress(input_indices[1] - input_indices[0], input_indices[1] - input_indices[0], ('', 1, 0))
        print()
    print("Cropping finished")


def rename_generated_files(input_folder: str, input_indices: Tuple[int, int],
                           output_folder: str, output_index_start: int,
                           files_per_folder: int, copy: bool = True) -> None:
    """
    Renames the files in a given folder and puts them into the dataset folder structure
    
    # Arguments
        input_folder: str. the folder to load the images from
        input_indices: Tuple[int, int]. the start and stop index of the images to load
        output_folder: str. the folder to save  the images to
        output_index_start: int. the start index for renaming the files
        files_per_folder: int. the number of files per folder in the structure to be created
        copy: bool. indicates whether to keep the original unrenamed files upon folder structure creation
    """

    output_index = output_index_start

    for i in range(input_indices[0], input_indices[1]):

        old_path = os.path.join(input_folder, str(i))

        output_subfolder = os.path.join(output_folder, str(files_per_folder * (output_index // files_per_folder)))
        if not (os.path.exists(output_subfolder)):
            os.makedirs(output_subfolder)

        new_path = os.path.join(output_subfolder, str(output_index))

        if os.path.isfile(old_path + '.png'):
            if copy:
                shutil.copy2(old_path + '.png', new_path + '.png')
            else:
                os.rename(old_path + '.png', new_path + '.png')

        if os.path.isfile(old_path + '.txt'):
            if copy:
                shutil.copy2(old_path + '.txt', new_path + '.txt')
            else:
                os.rename(old_path + '.txt', new_path + '.txt')

        output_index += 1
