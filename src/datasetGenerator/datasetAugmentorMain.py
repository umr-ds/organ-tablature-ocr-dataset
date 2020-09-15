# -*- coding: utf-8 -*-
"""
Main method for starting the dataset augmentation process.
The parameters for the augmentation are specified here

@author: Daniel Schneider
"""

import sys
from typing import Tuple

from datasetGenerator.datasetAugmentor.imageAugmentor import generate_augmented_dataset, generate_non_augmented_dataset


def augment_dataset(input_folder: str, input_indices: Tuple[int, int],
                    output_folder: str, augment_num: int, output_index_start: int) -> None:
    """ performs augmentation on the images with given indices in a folder and saves the results to another folder
        augmentation steps are: color, brightness, contrast, distortion, noise, saltAndPepper, zoom, skew, shear
    
    # Arguments
        input_folder: str. the folder to load the images to augment from
        input_indices: Tuple[int, int]. the indices of the images to augment
        output_folder: str. the folder to save the augmented images to
        augment_num: int. the number of augmentation samples to create per image
        output_index_start: int. the index to start numbering the augmented images by
    """

    output_size = (2000, 128)
    src_img_max_cutoff = (40, 40, 5, 15)
    files_per_folder = 1000
    show_progress_bar = True

    augment_params = {
        'color': 'large',       # none, small, medium, large
        'distortion': 'large',  # none, small, medium, large
        'noise': 'medium',      # none, small, medium, large
        'rotate': 'none',       # none, standard, standard_crop
        'scale': 'x_medium',    # none, x_small, x_medium, x_large, both_small, both_medium, both_large
        'shear': 'x_medium'     # none, x_small, x_medium, x_large, y_small, y_medium, y_large, both_small, both_medium, both_large
    }

    if augment_num > 0:
        generate_augmented_dataset(input_folder=input_folder, input_indices=input_indices,
                                   output_folder=output_folder, output_index_start=output_index_start,
                                   augment_num=augment_num, augment_params=augment_params, output_size=output_size,
                                   src_img_max_cutoff=src_img_max_cutoff, files_per_folder=files_per_folder,
                                   show_progress_bar=show_progress_bar)
    else:
        generate_non_augmented_dataset(input_folder=input_folder, input_indices=input_indices,
                                       output_folder=output_folder, output_index_start=output_index_start,
                                       output_size=output_size, files_per_folder=files_per_folder,
                                       show_progress_bar=show_progress_bar)

                    
if __name__ == "__main__":
    """
    # Arguments
        1. input_folder: str. the folder to load the images to augment from
        2. input_indices_start: int. the start index of the images to augment
        3. input_indices_stop: int. the stop index of the images to augment
        4. output_folder: str. the folder to save the augmented images to
        5. augment_num: int. the number of augmentation samples to create per image
        6. output_index_start: int. the index to start numbering the augmented images by
    """

    augment_dataset(input_folder=sys.argv[1],
                    input_indices=(int(sys.argv[2]), int(sys.argv[3])),
                    output_folder=sys.argv[4], augment_num=int(sys.argv[5]),
                    output_index_start=int(sys.argv[6]))
