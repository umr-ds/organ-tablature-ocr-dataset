# -*- coding: utf-8 -*-
"""
Methods used for different parts during the organ tablature generation

@author: Daniel Schneider
"""

from PIL import Image, ImageFilter
# from IPython.display import display
from typing import List, Dict, Tuple, Optional
import random
import math
import numpy as np

from datasetGenerator.augmentation.augmentationUtility import augment_background, augment_border, \
    augment_background_color, augment_image_element, augment_measure_line
from datasetGenerator.datasetGenerator.generatorUtility import load_random_label


def generate_background(width: int, height: int,
                        backgrounds: Dict[str, List[Image.Image]],
                        prev_background: Optional[Image.Image] = None,
                        background_offset: int = 20,
                        border_image_rate: int = 10,
                        prev_background_blend: Tuple[float, float] = (0.0, 0.25)
                        ) -> Image.Image:
    """
    Generates the background for a generated image of given size by selecting a random image from the given list
    and blending it with the previous image.
        
    # Arguments
        width: int, height: int. the size of the output image
        backgrounds: Dict[str,List[Image]]. the background images and border images to be used
        prev_background: Image. the last generated image (to be faded into the new image)
        background_offset: int. the offset used for the background images (to prevent cutoff)
        border_image_rate: int. the rate at which border images are to be generated (1 in x)
        prev_background_blend: Tuple[int, int]. the min and max rate for mixing the previous background in
    
    # Returns
        Image. the generated image
    """

    output_background = Image.new('RGBA', (width, height), 0)

    # background image
    r_back = random.choice(backgrounds['b']).copy()
    r_back_aug = augment_background(r_back)

    b_offset_x = -random.randint(background_offset, abs(r_back_aug.width - background_offset - width))
    b_offset_y = -random.randint(background_offset, abs(r_back_aug.height - background_offset - height))

    output_background.paste(r_back_aug, (b_offset_x, b_offset_y), r_back_aug)

    # left and right border images
    rand = random.randint(-border_image_rate, border_image_rate)

    if rand == 0 or rand == 1:
        r_left = random.choice(backgrounds['l'])
        r_left_aug = augment_border(r_left)

        r_left_aug_arr = np.array(r_left_aug)
        r_left_aug_arr[:, :, 3] = r_left_aug_arr[:, :, 3] * 0.9
        r_left_aug = Image.fromarray(r_left_aug_arr, mode='RGBA')

        l_offset_y = -random.randint(0, abs(r_left_aug.height - height))
        output_background.paste(r_left_aug, (0, l_offset_y), r_left_aug)

    if rand == -1 or rand == 0:
        r_right = random.choice(backgrounds['r'])
        r_right_aug = augment_border(r_right)

        r_right_aug_arr = np.array(r_right_aug)
        r_right_aug_arr[:, :, 3] = r_right_aug_arr[:, :, 3] * 0.9
        r_right_aug = Image.fromarray(r_right_aug_arr, mode='RGBA')

        r_offset_y = -random.randint(0, abs(r_right_aug.height - height))
        output_background.paste(r_right_aug, (width - r_right_aug.width, r_offset_y), r_right_aug)

    # mix with previous background
    if prev_background:
        prev_background_transformed = prev_background.transpose(Image.FLIP_LEFT_RIGHT).filter(ImageFilter.BoxBlur(3))
        blend_rand = random.uniform(prev_background_blend[0], prev_background_blend[1])

        prev_background_blend = prev_background_transformed.resize((width, height))
        output_background = Image.blend(output_background, prev_background_blend, blend_rand)

    output_background_aug = augment_background_color(output_background)

    # display(output_background)
    return output_background_aug


def generate_measure_content(measure_begin_x: int, measure_end_x: int, num_of_voices: int, gt: str,
                             block_offset_x: Tuple[int, int], symbol_offset_x: int, symbol_offset_y: int,
                             voice_positions_y: List[int], level_height_bot: int,
                             duration_images: Dict[str, List[Image.Image]],
                             duration_weights: List[int], duration_metadata: Dict[str, Dict[str, int]],
                             note_images: Dict[str, List[Image.Image]],
                             note_weights: List[int], note_metadata: Dict[str, Dict[str, int]],
                             rest_images: Dict[str, List[Image.Image]],
                             rest_weights: List[int], rest_metadata: Dict[str, Dict[str, int]],
                             output_content: Image.Image, bboxes: List[List[Tuple[int, int, int, int]]],
                             letter_list: List[List[str]]
                             ) -> Tuple[Image.Image, List[List[Tuple[int, int, int, int]]], List[List[str]]]:
    """
    Generates the contents of one voice image of given size by selecting a random image from the given list
    and mixing it with the previous image at a random blend range.
        
    # Arguments
        measure_begin_x: int. the start position of the measure to fill,
        measure_end_x: int. the end position of the measure to fill
        num_of_voices: int. the number of voices to generate content for
        gt: str. the style of groundtruth values to save
                 'simple' for a label-strings,
                 'boxfile' for bounding boxes,
                 'boxDisplay' for bounding boxes (with visualization during creation)
        block_offset_x: Tuple[int, int]. the min and max offsets of generated blocks in x direction
        symbol_offset_x: int. the symbol position randomization in x-direction
        symbol_offset_y: int. the symbol position randomization in y-direction
        voice_positions_y: List[int]. the y-position of each voice
        level_height_bot: int. the height of the bottom level area
        duration_images: Dict[str,List[Image]], the images for the duration labelset
        duration_weights: List[int], the weights for the duration labelset
        duration_metadata: Dict[str,Dict[str,int]], the metadata for the duration labelset
        note_images: Dict[str,List[Image]], the images for the note labelset
        note_weights: List[int], the weights for the note labelset
        note_metadata: Dict[str,Dict[str,int]], the metadata for the note labelset
        rest_images: Dict[str,List[Image]], the images for the rest labelset
        rest_weights: List[int], the weights for the rest labelset
        rest_metadata: Dict[str,Dict[str,int]], the metadata for the rest labelset
        output_content: Image. the image to paste the generated content onto
        bboxes: List[List[Tuple[int,int,int,int]]]. the bounding boxes for each voice (separated into top and bottom layer)
        letter_list: List[List[str]]. the label string-sequences for each voice (separated into top and bottom layer)
        
    # Returns
        Image. the updated image
        List[List[Tuple[int,int,int,int]]]. the updated bounding boxes for each voice (separated into top and bottom layer)
        List[List[str]]. the updated label string-sequences for each voice (separated into top and bottom layer)
    """

    for r in range(num_of_voices):
        y = voice_positions_y[r]

        x = measure_begin_x
        image_block_list = []
        rest_width = 0

        while x < measure_end_x:
            output_img_block, letters_block, bbox_block, block_width = generate_image_block(
                gt, level_height_bot, symbol_offset_x, symbol_offset_y,
                duration_images, duration_weights, duration_metadata,
                note_images, note_weights, note_metadata,
                rest_images, rest_weights, rest_metadata
            )

            # print(x, block_width, xMeasureEnd)
            if x + block_width + block_offset_x[1] < measure_end_x:
                image_block_list.append(output_img_block)
                letter_list[2 * r] += letters_block[0]
                letter_list[2 * r + 1] += letters_block[1]
                bboxes[2 * r] += bbox_block[0]
                bboxes[2 * r + 1] += bbox_block[1]

                random_offset = random.randint(block_offset_x[0], block_offset_x[1])
                rest_width += random_offset
                x = x + block_width + random_offset

            else:
                if len(image_block_list) > 0:
                    rest_width += measure_end_x - x
                    block_dist = int(round(rest_width / (len(image_block_list) + 1)))
                    block_dist_rand = int(round(block_dist / len(image_block_list)))

                    x = measure_begin_x
                    img_blend = Image.new('RGBA', output_content.size, 0)

                    for img in image_block_list:
                        x = x + block_dist + random.randint(-block_dist_rand, block_dist_rand)

                        img_blend.paste(img, (x, y - img.height), img)

                        x += img.width

                    # display(img_blend)
                    output_content = Image.alpha_composite(output_content, img_blend)

                    # display(outputContent)
                break

    return output_content, bboxes, letter_list


def generate_image_block(gt: str,
                         level_height_bot: int, symbol_offset_x: int, symbol_offset_y: int,
                         duration_images: Dict[str, List[Image.Image]],
                         duration_weights: List[int], duration_metadata: Dict[str, Dict[str, int]],
                         note_images: Dict[str, List[Image.Image]],
                         note_weights: List[int], note_metadata: Dict[str, Dict[str, int]],
                         rest_images: Dict[str, List[Image.Image]],
                         rest_weights: List[int], rest_metadata: Dict[str, Dict[str, int]],
                         ) -> Tuple[Image.Image, List[str], List[Tuple[int, int, int, int]], int]:
    """ generates an image block (either one duration symbol with the according number of note symbols or one rest symbol).
        returns the generated image with groundtruth values in the specified format 
        
    # Arguments
        gt: str. the style of groundtruth values to save
                 'simple' for a label-strings,
                 'boxfile' for bounding boxes,
                 'boxDisplay' for bounding boxes (with visualization during creation)
        level_height_bot: int. the height of the bottom level area
        symbol_offset_x: int. the symbol position randomization in x-direction
        symbol_offset_y: int. the symbol position randomization in y-direction
        voice_positions_y: List[int]. the y-position of each voice
        level_offset_y: Tuple[int,int]. the offset of each layer end position
                        (0: top layer end, 1: bottom layer end = voice height)
        duration_images: Dict[str,List[Image]], the images for the duration labelset
        duration_weights: List[int], the weights for the duration labelset
        duration_metadata: Dict[str,Dict[str,int]], the metadata for the duration labelset
        note_images: Dict[str,List[Image]], the images for the note labelset
        note_weights: List[int], the weights for the note labelset
        note_metadata: Dict[str,Dict[str,int]], the metadata for the note labelset
        rest_images: Dict[str,List[Image]], the images for the rest labelset
        rest_weights: List[int], the weights for the rest labelset
        rest_metadata: Dict[str,Dict[str,int]], the metadata for the rest labelset


    # Returns
        Image. the generated image block
        List[str]. the label string-sequences for top and bottom layer 
        List[Tuple[int,int,int,int]]. the bounding boxes for top and bottom layer (empty tuples if none are generated)
        int. width of the generated block 
    """

    r_duration = load_random_label(duration_images, duration_weights, duration_metadata)

    # note und duration
    if r_duration[0] != '':
        letters_top = [r_duration[0]]
        if gt == "simple":
            letters_top.append(' ')

        # chose random duration image
        r_duration_image = random.choice(r_duration[1])
        r_duration_image_aug = augment_image_element(r_duration_image)  # augmentation on each random sample-image (without bounding box)

        num_of_notes = r_duration[2]['numOfMatches']
        r_duration_img_width = r_duration_image_aug.width + 2 * r_duration[2]['offsetSide']
        r_duration_img_height = r_duration_image_aug.height + r_duration[2]['offsetBot']

        # find random note images for duration
        note_list = []
        letters_bot = []

        # note_width = 0
        note_widths = []
        note_height = 0
        note_heights = []

        for _ in range(num_of_notes):
            r_note = load_random_label(note_images, note_weights, note_metadata)

            r_note_image = random.choice(r_note[1])
            r_note_image_aug = augment_image_element(r_note_image)  # augmentation on each random sample-image (without bounding box)

            letters_bot.append(r_note[0])
            if gt == "simple":
                letters_bot.append(' ')

            note_list.append(r_note_image_aug)
            # note_width += r_note_image_aug.width
            note_widths.append(r_note_image_aug.width + 2 * r_note[2]['offsetSide'])
            note_heights.append(r_note_image_aug.height + r_note[2]['offsetBot'])
            if note_heights[-1] > note_height:
                note_height = note_heights[-1]

        note_width = sum(note_widths)

        # calculate block_width
        if note_width < r_duration_img_width:
            block_width = int(r_duration_img_width * 1.1)
        else:
            block_width = int(note_width * 1.1)

        block_height_top = int(r_duration_img_height * 1.05)
        block_height_bot = int(note_height * 1.05)

        output_img = Image.new('RGBA', (block_width, block_height_top + level_height_bot), 0)

        # place duration image
        (imgTop, bbox_top) = paste_images([r_duration_image_aug], gt,
                                          [r_duration_img_width], block_width, symbol_offset_x,
                                          [r_duration_img_height], block_height_top, symbol_offset_y)
        output_img.paste(imgTop, (0, 0), imgTop)

        # place note images
        (imgBot, bboxBot) = paste_images(note_list, gt,
                                         note_widths, block_width, symbol_offset_x,
                                         note_heights, block_height_bot, symbol_offset_y)
        output_img.paste(imgBot, (0, output_img.height - block_height_bot), imgBot)

    # rest
    else:
        r_rest = load_random_label(rest_images, rest_weights, rest_metadata)
        r_rest_image = random.choice(r_rest[1])
        r_rest_image_aug = augment_image_element(r_rest_image)  # augmentation on each random sample-image (without bounding box)

        r_rest_img_height = r_rest_image_aug.height + r_rest[2]['offsetBot']
        r_rest_img_width = r_rest_image_aug.width + 2 * r_rest[2]['offsetSide']

        letters_top = []
        bbox_top = []
        letters_bot = [r_rest[0]]
        if gt == "simple":
            letters_bot.append(' ')

        block_width = int(r_rest_img_width * 1.1)
        block_height_bot = int(r_rest_img_height * 1.05)

        # place rest image
        (output_img, bboxBot) = paste_images([r_rest_image_aug], gt,
                                             [r_rest_img_width], block_width, symbol_offset_x,
                                             [r_rest_img_height], block_height_bot, symbol_offset_y)

    # display(output_img)

    return output_img, [letters_top, letters_bot], [bbox_top, bboxBot], block_width


def paste_images(images: List[Image.Image], gt: str,
                 img_widths: List[int], block_width: int, symbol_offset_x: int,
                 img_heights: List[int], block_height: int, symbol_offset_y: int,
                 ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Places a series of images in a given output area and randomizes their positions.
    Also generates groundtruth (bounding box) values
        
    # Arguments
        images: List[Image]. the images to be pasted
        img_widths: List[int]. the widths of the images (including margins)
        block_width: int. the width of the block to paste the images in
        symbol_offset_x: int. the symbol position randomization in x-direction
        img_heights: List[int]. the heights of the images (including margins)
        block_height: int. the height of the block to paste the images in
        symbol_offset_y: int. the symbol position randomization in y-direction
        gt: str. the style of groundtruth values to save
                'simple' for a label-strings,
                'boxfile' for bounding boxes,
                'boxDisplay' for bounding boxes (with visualization during creation)
    
    # Returns
        Image. the generated image block
        Tuple[int,int,int,int]. the bounding boxes (empty tuples if none are generated)
    """

    content_width = sum(img_widths)
    content_height = max(img_heights)

    x_offset_center = int((block_width - content_width) / (len(images) + 1))
    if x_offset_center < symbol_offset_x:
        x_offset_center = symbol_offset_x

    symbol_offset_x = int(math.ceil(x_offset_center / 2))
    x_offset = [x_offset_center - symbol_offset_x, x_offset_center + symbol_offset_x]

    y_offset_center = int((block_height - content_height) / 2)
    if y_offset_center < symbol_offset_y:
        y_offset_center = symbol_offset_y

    symbol_offset_y = int(math.ceil(y_offset_center / 2))
    y_offset = [y_offset_center - symbol_offset_y, y_offset_center + symbol_offset_y]

    # numOfImages = len(images)
    bboxes = []

    x = random.randint(x_offset[0], x_offset[1])
    y = block_height - random.randint(y_offset[0], y_offset[1])

    output_img = Image.new('RGBA', (block_width, block_height), 0)
    for (i, img) in enumerate(images):

        xi = x + int(img_widths[i] / 2 - img.width / 2)
        yi = y - img_heights[i]

        output_img.paste(img, (xi, yi), img)

        if gt == 'boxfile' or gt == 'boxDisplay':
            bboxes.append(((xi, yi), (xi + img_widths[i], yi + img_heights[i])))

        x += img_widths[i] + random.randint(x_offset[0], x_offset[1])

    # display(output_img)

    return output_img, bboxes


def generate_measure_line(r_special: Tuple[str, List[Image.Image], Dict[str, int]],
                          num_of_voices: int, gt: str,
                          position_x: int, content_end_x: int,
                          symbol_offset_x: int, symbol_offset_y: int, voice_positions_y: List[int],
                          output_content: Image.Image, bboxes: List[List[Tuple[int, int, int, int]]],
                          letter_list: List[List[str]]
                          ) -> Tuple[Image.Image, List[List[Tuple[int, int, int, int]]], List[List[str]], int]:
    """
    Places a measure line image at the specified x-coordinate and randomizes its positions.
    Also generates groundtruth (bounding box) values

    # Arguments
        r_special: Dict[str, List[Image.Image], Dict[str, int]]. the special character information (key, images, metadata)
        num_of_voices: int. the number of voices to generate content for
        gt: str. the style of groundtruth values to save
                'simple' for a label-strings,
                'boxfile' for bounding boxes,
                'boxDisplay' for bounding boxes (with visualization during creation)
        position_x: int. the x-coordinate to place the measure line at
        content_end_x: int. the maximum x-position to witch the measure line can reach
        symbol_offset_x: int. the symbol position randomization in x-direction
        symbol_offset_y: int. the symbol position randomization in y-direction
        voice_positions_y: List[int]. the y-position of each voice
        output_content: Image. the image to paste the generated content onto
        bboxes: List[List[Tuple[int,int,int,int]]]. the bounding boxes for each voice (separated into top and bottom layer)
        letter_list: List[List[str]]. the label string-sequences for each voice (separated into top and bottom layer)
    
    # Returns
        Image. the updated image with the pasted measure line
        List[List[Tuple[int,int,int,int]]]. the updated bounding boxes for each voice
        List[List[str]]. the updated label-string-sequences for each voice
        int. the x-position after pasting the image 
    """

    # breakAfter = False
    height = output_content.height

    r_special_image = random.choice(r_special[1])

    img_blend = Image.new('RGBA', output_content.size, 0)

    if r_special[0] == 'm':
        r_special_image_aug = augment_measure_line(r_special_image)

        if position_x + r_special_image_aug.width < content_end_x:
            y = -random.randint(0, abs(r_special_image_aug.height - height))
            img_blend.paste(r_special_image_aug, (position_x, y), r_special_image_aug)

            if gt == "simple":
                for r in range(num_of_voices):
                    letter_list[2 * r].append('|')
                    letter_list[2 * r + 1].append('|')

                    letter_list[2 * r].append(' ')
                    letter_list[2 * r + 1].append(' ')

            position_x += r_special_image.width

    else:
        r_special_image_aug = augment_image_element(r_special_image)
        r_special_img_width = r_special_image_aug.width + 2 * r_special[2]['offsetSide']
        r_special_img_height = r_special_image_aug.height + r_special[2]['offsetBot']

        if position_x + r_special_image_aug.width < content_end_x:

            if r_special[0] == 't':
                block_width = int(r_special_img_width * 1.05)
                block_height = int(r_special_img_height * 1.1)

                img, _ = paste_images([r_special_image_aug], gt,
                                      [r_special_img_width], block_width, symbol_offset_x,
                                      [r_special_img_height], block_height, symbol_offset_y)
                img_blend.paste(img, (position_x, voice_positions_y[-1] - block_height), img)

            else:
                r = random.randint(0, num_of_voices - 1)
                y = voice_positions_y[r]

                letter_list[2 * r].append(r_special[0])
                if gt == "simple":
                    letter_list[2 * r].append(' ')

                block_width = int(r_special_img_width * 1.05)
                block_height = int(r_special_img_height * 1.05)
                # place image
                img, bbox = paste_images([r_special_image_aug], gt,
                                         [r_special_img_width], block_width, symbol_offset_x,
                                         [r_special_img_height], block_height, symbol_offset_y)

                img_blend.paste(img, (position_x, y - block_height), img)

                bboxes[2 * r] += bbox

            position_x += block_width

    output_content = Image.alpha_composite(output_content, img_blend)

    return output_content, bboxes, letter_list, position_x
