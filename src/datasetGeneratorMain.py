# -*- coding: utf-8 -*-
"""
Main method for starting the organ tablature dataset generator.
The parameters for the generator are specified here

@author: Daniel Schneider
"""

import os
import sys

from datasetGenerator.datasetGenerator.generatorUtility import load_labelset, load_labelset_and_metadata
from datasetGenerator.datasetGenerator.randomImageGenerator import RandomImageGenerator


def generate_dataset(input_folder: str, output_folder: str,
                     generate_num: int, output_index_start: int, final_augment: bool = False) -> None:
    """
    Generates a batch of random organ tablature rows using existing symbols loaded from a given folder structure.
    The generated images and corresponding groundtruth data are saved in the specified folder (numbering them consecutive)
        
    # Arguments
        input_folder: str. the folder to load the labelsets from
        output_folder: str. the folder to save the generated dataset to
        generate_num: int. number of images to generate
        output_index_start: int. number to start numbering the generated images at
        final_augment: bool. indicates whether the final augmentation step (noise, rotation) shall be done
    """
    
    # defaults
    width = (1900, 2000)    # min and max width
    height = (128, 140)     # min and max height
    gt = "simple"           # 'simple': label-string, 'boxfile': bounding boxes (may lead to errors with augmentation),
                            # 'boxDisplay': bounding boxes + visualization during creation
    num_of_voices = 4       #

    empty_measure_chance = 3                # 1 in x chance
    empty_measure_width = (0, 120)          # min and max width
    filled_measure_width = (240, 440)       # min and max width

    side_measure_offset_x = (40, 80)        # left and right side
    level_offset_y = (55, 105)              # first and second level end
    block_offset_xy = ((2, 20), (5, 20))    # x and y direction min and max width
    symbol_offset_xy = (2, 2)               # x and y direction offset

    background_offset = 20                  # all sides margin
    border_image_chance = 5                 # 1 in x chance
    prev_background_blend = (0.0, 0.25)     # min and max blend

    background_images = load_labelset(os.path.join(input_folder, 'backgrounds'))

    duration_labelset = load_labelset_and_metadata(input_folder, 'duration')
    note_labelset = load_labelset_and_metadata(input_folder, 'note')
    rest_labelset = load_labelset_and_metadata(input_folder, 'rest')
    special_labelset = load_labelset_and_metadata(input_folder, 'special')

    files_per_folder = 1000
    show_progress_bar = True

    generator = RandomImageGenerator(image_width=width, image_height=height, num_of_voices=num_of_voices,
                                     background_images=background_images,
                                     duration_labelset=duration_labelset,
                                     note_labelset=note_labelset,
                                     rest_labelset=rest_labelset,
                                     special_labelset=special_labelset,
                                     background_offset=background_offset, border_image_chance=border_image_chance,
                                     prev_background_blend=prev_background_blend,
                                     empty_measure_chance=empty_measure_chance, empty_measure_width=empty_measure_width,
                                     filled_measure_width=filled_measure_width,
                                     side_measure_offset_x=side_measure_offset_x, level_offset_y=level_offset_y,
                                     block_offset_xy=block_offset_xy,
                                     symbol_offset_xy=symbol_offset_xy,
                                     final_augment=final_augment, gt=gt)

    generator.generate_random_rows_batch(generate_num=generate_num, output_index_start=output_index_start,
                                         output_folder=output_folder, files_per_folder=files_per_folder,
                                         show_progress_bar=show_progress_bar)


if __name__ == "__main__":
    """
    # Arguments
        1. input_folder: str. the folder to load the labelsets from
        2. output_folder: str. the folder to save the generated dataset to
        3. generate_num: int. number of images to generate
        4. output_index_start: int. number to start numbering the generated images at
        5. final_augment: bool. indicates whether the final augmentation step (noise, rotation) shall be done
    """

    generate_dataset(input_folder=sys.argv[1], output_folder=sys.argv[2],
                     generate_num=int(sys.argv[3]), output_index_start=int(sys.argv[4]),
                     final_augment=bool(sys.argv[5]))
