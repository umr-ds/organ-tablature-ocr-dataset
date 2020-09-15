# -*- coding: utf-8 -*-
"""
The Generator-Class consisting of the methods for running an organ tablature generation

@author: Daniel Schneider
"""

from PIL import Image
from typing import List, Dict, Tuple, Optional
import os
import random
import numpy as np

from datasetGenerator.datasetGenerator.generatorUtility import \
    visualize_bounding_boxes, split_image_to_voices_with_given_center, load_random_label, display_progress
from datasetGenerator.datasetGenerator.generatorParts import \
    generate_background, generate_measure_content, generate_measure_line
from datasetGenerator.augmentation.augmentationUtility import \
    augment_image_distortion_bb, augment_image_color_noise


class RandomImageGenerator:
    """
    Class for generating random organ tablature rows from given images of single tablature characters

    """

    def __init__(self, image_width: Tuple[int, int], image_height: Tuple[int, int], num_of_voices: int,
                 background_images: Dict[str, List[Image.Image]],
                 duration_labelset: Tuple[Dict[str, List[Image.Image]], Dict[str, Dict[str, int]]],
                 note_labelset: Tuple[Dict[str, List[Image.Image]], Dict[str, Dict[str, int]]],
                 rest_labelset: Tuple[Dict[str, List[Image.Image]], Dict[str, Dict[str, int]]],
                 special_labelset: Tuple[Dict[str, List[Image.Image]], Dict[str, Dict[str, int]]],
                 background_offset: int, border_image_chance: int,
                 prev_background_blend: Tuple[float, float],
                 empty_measure_chance: int, empty_measure_width: Tuple[int, int],
                 filled_measure_width: Tuple[int, int],
                 side_measure_offset_x: Tuple[int, int], level_offset_y: Tuple[int, int],
                 block_offset_xy: Tuple[Tuple[int, int], Tuple[int, int]],
                 symbol_offset_xy: Tuple[int, int],
                 final_augment: bool = True, gt: str = 'simple'):

        """
        Initializes the generator

        # Arguments
            width: Tuple[int,int]. the min and max width of the output image
            height: Tuple[int,int]. the min and max height of the output image
            num_of_voices:int. the number of voices to generate
            background_images: Dict[str,List[Image]]. the background images and border images to be used
            duration_labelset: Tuple[Dict[str, List[Image.Image]], Dict[str, Dict[str, int]]]. the duration images and metadata
            note_labelset: Tuple[Dict[str, List[Image.Image]], Dict[str, Dict[str, int]]]. the note pitch images and metadata
            rest_labelset: Tuple[Dict[str, List[Image.Image]], Dict[str, Dict[str, int]]]. the rest images and metadata
            special_labelset: Tuple[Dict[str, List[Image.Image]], Dict[str, Dict[str, int]]]. the special images and metadata
            background_offset:int. the offset of the background images (to prevent cutoff)
            border_image_chance: int. the chance at which border images are to be generated (1 in x)
            prev_background_blend: Tuple[int, int]. the min and max rate for mixing the previous background in
            empty_measure_chance: int. the chance at which empty measures are to be generated (1 in x)
            empty_measure_width: Tuple[int, int]. the min and max width for empty measures
            filled_measure_width: Tuple[int, int]. the min and max width for non-empty measures
            side_measure_offset_x: Tuple[int, int]. the offsets for the first and last measure
            level_offset_y: Tuple[int, int]. the offset of each layer end position
                                             (0: top layer end, 1: bottom layer end = voice height)
            block_offset_xy: Tuple[Tuple[int, int], Tuple[int, int]]. the min and max offsets of generated blocks
                                                                      in x and y direction
            symbol_offset_xy: Tuple[int, int]. the offset randomization for character placement (in x and y direction)
            final_augment: bool. indicates whether the final augmentation step (noise, rotation) shall be done
            gt: str. the style of groundtruth values to save
                     'simple' for a label-strings,
                     'boxfile' for bounding boxes,
                     'boxDisplay' for bounding boxes (with visualization during creation)
        """

        self.image_width = image_width
        self.image_height = image_height
        self.num_of_voices = num_of_voices

        self.background_images = background_images

        self.duration_images = duration_labelset[0]
        self.duration_metadata = duration_labelset[1]
        self.duration_weights = [int(self.duration_metadata[t]['weight']) for t in self.duration_images.keys()]

        self.note_images = note_labelset[0]
        self.note_metadata = note_labelset[1]
        self.note_weights = [int(self.note_metadata[t]['weight']) for t in self.note_images.keys()]

        self.rest_images = rest_labelset[0]
        self.rest_metadata = rest_labelset[1]
        self.rest_weights = [int(self.rest_metadata[t]['weight']) for t in self.rest_images.keys()]

        self.special_images = special_labelset[0]
        self.special_metadata = special_labelset[1]
        self.special_weights = [int(self.special_metadata[t]['weight']) for t in self.special_images.keys()]

        self.duration_images[''] = []   # dummy entry for rest instead of duration symbol
        self.duration_metadata[''] = {}
        self.duration_weights += [sum(self.rest_weights)]

        self.background_offset = background_offset
        self.border_image_rate = border_image_chance
        self.prev_background_blend = prev_background_blend

        self.empty_measure_rate = empty_measure_chance
        self.empty_measure_width = empty_measure_width
        self.filled_measure_width = filled_measure_width

        self.side_measure_offset_left = side_measure_offset_x[0]
        self.side_measure_offset_right = side_measure_offset_x[1]

        self.level_offset_y = level_offset_y
        self.block_offset_x = block_offset_xy[0]
        self.block_offset_y = block_offset_xy[1]
        self.symbol_offset_x = symbol_offset_xy[0]
        self.symbol_offset_y = symbol_offset_xy[1]

        self.final_augment = final_augment
        self.gt = gt

    def generate_random_row(self, prev_background: Image.Image) \
            -> Tuple[Image.Image, List[str], List[int], Optional[Image.Image]]:
        """
        Generates a random organ tablature row with a specified size using existing symbols.
        Also generates matching groundtruth data

        # Arguments
            prev_background: Image. the last generated image (to be faded into the new image)

        # Returns
            Image. the generated image
            List[str]: the groundtruth data for every voice (as a string in the specified format))
            List[int]: the center position of each generated voices
            Image. the generated image with visualized bounding boxes (or None if none are generated)
        """

        width = random.randint(self.image_width[0], self.image_width[1])
        height = self.image_height[0] * self.num_of_voices + self.level_offset_y[1]

        output_content = Image.new('RGBA', (width, height), 0)
        output_gt = []

        bboxes = [[] for _ in range(self.num_of_voices * 2)]
        letter_list = [[] for _ in range(self.num_of_voices * 2)]
        voice_centers = [[] for _ in range(self.num_of_voices)]

        if self.gt == "simple":
            for letter in letter_list:
                letter.append(' ')

        content_end_x = width - self.side_measure_offset_right

        #########################################################

        # Background
        output_background = generate_background(width=width, height=height,
                                                backgrounds=self.background_images,
                                                prev_background=prev_background,
                                                background_offset=self.background_offset,
                                                border_image_rate=self.border_image_rate,
                                                prev_background_blend=self.prev_background_blend)

        # pre-calculate voice-positions
        y = self.level_offset_y[0]
        voice_positions_y = []
        for v in range(self.num_of_voices):
            y += random.randint(self.block_offset_y[0], self.block_offset_y[1]) + self.level_offset_y[1]   # random voice offset
            voice_positions_y.append(y)
            voice_centers[v] = y - int(self.level_offset_y[1] / 2)

        #########################################################

        # place measures and special chars
        measure_end_x = self.side_measure_offset_left + random.randint(self.empty_measure_width[0],
                                                                       self.empty_measure_width[1])
        measure_begin_x = 0
        empty_measure = True
        no_measure_line = False

        r = random.randint(0, self.side_measure_offset_left)
        if r == 0:
            no_measure_line = True

        r_special = load_random_label(self.special_images, self.special_weights, self.special_metadata)

        while measure_end_x <= content_end_x:

            # fill measure
            if not empty_measure:

                output_content, bboxes, letter_list = generate_measure_content(
                    measure_begin_x, measure_end_x, self.num_of_voices, self.gt,
                    self.block_offset_x, self.symbol_offset_x, self.symbol_offset_y,
                    voice_positions_y, (self.level_offset_y[1] - self.level_offset_y[0]),
                    self.duration_images, self.duration_weights, self.duration_metadata,
                    self.note_images, self.note_weights, self.note_metadata,
                    self.rest_images, self.rest_weights, self.rest_metadata,
                    output_content, bboxes, letter_list
                )

                # select special char
                r_special = load_random_label(self.special_images, self.special_weights, self.special_metadata)

            if not no_measure_line:
                output_content, bboxes, letter_list, measure_end_x = generate_measure_line(
                    r_special, self.num_of_voices, self.gt, measure_end_x, content_end_x,
                    self.symbol_offset_x, self.symbol_offset_y, voice_positions_y,
                    output_content, bboxes, letter_list
                )

            measure_begin_x = measure_end_x
            no_measure_line = False
            empty_measure = False

            r = random.randint(0, self.empty_measure_rate)
            if r == 0:
                last_special = r_special[0]

                # end of image reached -> one measure without closing line
                if measure_end_x + self.filled_measure_width[0] < content_end_x < measure_end_x + self.filled_measure_width[1]:
                    measure_end_x = content_end_x - random.randint(self.empty_measure_width[0], self.empty_measure_width[1])
                    no_measure_line = True

                # empty measure
                else:
                    empty_measure = True

                    r_special = load_random_label(self.special_images, self.special_weights, self.special_metadata)
                    next_special = r_special[0]

                    if last_special == 'm' and next_special == 'm':
                        # two measures, keep distance like normal measure
                        if measure_end_x + self.filled_measure_width[0] < content_end_x < measure_end_x + self.filled_measure_width[1]:
                            measure_end_x = content_end_x - random.randint(self.empty_measure_width[0], self.empty_measure_width[1])
                        else:
                            measure_end_x += random.randint(self.filled_measure_width[0], self.filled_measure_width[1])

                    else:
                        # at least one special element, shorter distance
                        if measure_end_x + self.empty_measure_width[0] < content_end_x < measure_end_x + self.empty_measure_width[1]:
                            measure_end_x = content_end_x - random.randint(self.empty_measure_width[0], self.empty_measure_width[1])
                        else:
                            measure_end_x += random.randint(self.empty_measure_width[0], self.empty_measure_width[1])
            else:
                if measure_end_x + self.filled_measure_width[0] < content_end_x < measure_end_x + self.filled_measure_width[1]:
                    measure_end_x = content_end_x - random.randint(self.empty_measure_width[0], self.empty_measure_width[1])
                else:
                    measure_end_x += random.randint(self.filled_measure_width[0], self.filled_measure_width[1])

        #########################################################

        # Augmentation on result image (with list of bounding boxes)
        (output_img_aug, outputBoxes) = augment_image_distortion_bb(output_content, bboxes)   # +[lineBboxes])

        # Blending of Foreground and Background
        output_img_aug_arr = np.array(output_img_aug)
        output_img_aug_arr[:, :, 3] = output_img_aug_arr[:, :, 3]*0.9
        output_img_aug = Image.fromarray(output_img_aug_arr, mode='RGBA')
        # display(output_img_aug)

        output_content_blend = Image.new('RGBA', (width, height), 0)
        pos_x = round((width - output_img_aug.width)/2)
        pos_y = round((height - output_img_aug.height)/2)
        output_content_blend.paste(output_img_aug, (pos_x, pos_y), output_img_aug)
        output_final = Image.alpha_composite(output_background, output_content_blend)
        # display(output_background)

        if self.final_augment:
            output_final_aug = augment_image_color_noise(output_final)
        else:
            output_final_aug = output_final

        #########################################################

        # Ground Truth Output
        for r in range(self.num_of_voices):
            gt_row = ''
            for i in range(2):
                if self.gt == 'boxfile' or self.gt == 'boxDisplay':
                    gt_row += '--Row {:d}--\n'.format(i)

                for (l, letter) in enumerate(letter_list[2*r+i]):
                    gt_row += letter
                    if (self.gt == 'boxfile' or self.gt == 'boxDisplay') and (' ' not in letter):
                        box = outputBoxes[2*r+i][l]
                        gt_row += ' {:d} {:d} {:d} {:d} {:d} \n'.format(box[0][0], box[0][1], box[1][0], box[1][1], 0)

                if self.gt == 'simple':
                    gt_row += '\n'

            output_gt.append(gt_row)

        # Bounding Box Visualization
        if self.gt == 'boxDisplay':
            output_img_box_viz = visualize_bounding_boxes(output_final_aug, outputBoxes)
        else:
            output_img_box_viz = None

        return output_final_aug, output_gt, voice_centers, output_img_box_viz

    def generate_random_rows_batch(self, generate_num: int, output_index_start: int,
                                   output_folder: str, files_per_folder: int = 1000,
                                   show_progress_bar: bool = False) -> None:
        """
        generates a batch of random organ tablature rows with specified size using existing symbols.
        Saves the generated images and corresponding groundtruth data in the specified folder.
        The generated images are numbered with ids ranging from start_n to start_n + n

        # Arguments
            n: int. number of images to generate
            start_n: int. the index to start generation at
            output_folder: str. the folder to output the result images to
            files_per_folder: int. the number of files in each subfolder of the dataset
            show_progress_bar: bool. indicates whether to display a progress bar in stdout
        """

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_img_aug = None

        print("Generation of {}k images".format(round(generate_num / 1000)))

        for i in range(output_index_start, output_index_start + generate_num, self.num_of_voices):
            if show_progress_bar:
                display_progress(i - output_index_start, generate_num, ('k', 1000, 1))

            (output_img_aug, output_gt, voice_positions, _) = self.generate_random_row(prev_background=output_img_aug)

            output_img_split = split_image_to_voices_with_given_center(output_img_aug, self.image_height,
                                                                       voice_positions)

            for v in range(self.num_of_voices):
                output_index = i + v

                subfolder_index = str(files_per_folder * (output_index // files_per_folder))
                output_subfolder = os.path.join(output_folder, subfolder_index)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder, exist_ok=True)

                output_path = os.path.join(output_subfolder, str(i + v))
                output_img_split[v].save(output_path+".png")
                with open(output_path+".txt", 'w') as f:
                    f.write(output_gt[v])
                    f.close()

        if show_progress_bar:
            display_progress(generate_num, generate_num, ('k', 1000, 1))
            print()
        print("Generation finished")
