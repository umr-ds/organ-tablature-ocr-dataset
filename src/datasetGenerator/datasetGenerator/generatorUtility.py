# -*- coding: utf-8 -*-
"""
@author: Daniel Schneider
"""

import os
import random
import sys
import requests
import zipfile

from PIL import Image, ImageDraw
from typing import List, Dict, Tuple, TextIO


def load_labelset(input_path: str, subfolder: str = "cutT") \
        -> Dict[str, List[Image.Image]]:
    """
    Loads a labelset from the specified folder and returns it as a dictionary
    where each label references a list of available images for that label
    
    # Arguments
        input_path: str. the folder to load the labelset from
                         data will be loaded from all subfolders (excluding those starting with '_')
        subfolder: str. the subfolder the images from each category should be loaded from
        
    # Returns
        Dict[str, List[Image]]. the labelset images as a dictionary of labels and corresponding lists of available images
    """

    file_list = os.listdir(input_path)

    train_data = {}

    for f in file_list:
        f_path = os.path.join(input_path, f)
        if not(f.startswith('_')):
            if os.path.isdir(f_path):
                # folder
                folder_file_list = os.listdir(f_path)

                # use cut out versions if available
                if subfolder in folder_file_list:
                    f_path = os.path.join(f_path, folder_file_list[folder_file_list.index(subfolder)])
                    folder_file_list = os.listdir(f_path)

                for ff in folder_file_list:
                    ff_path = os.path.join(f_path, ff)

                    if not(os.path.isdir(ff_path)) and not(ff.startswith('_')):
                        with open(ff_path, 'rb') as i:
                            img = Image.open(i)

                            key = ff.split('_')[0]
                            if key not in train_data:
                                train_data[key] = []

                            train_data[key].append(img.copy())
            else:
                with open(f_path, 'rb') as i:
                    img = Image.open(i)

                    key = f.split('_')[0]
                    if key not in train_data:
                        train_data[key] = []

                    train_data[key].append(img.copy())

    return train_data


def load_labelset_metadata(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Loads the labelset metadata from the specified file and returns it as a dictionary
    where each label references the metadata information (weights, margin, ...) for that label
    
    # Arguments
        file_path: str. the path to the file to load the metadata from
    
    # Returns
        Dict[str, Dict[str, int]]. the labelset metadata as dictionary of labels and corresponding metadata information
    """

    duration_matches = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                s = line.split(':')
                keys = s[1].strip().split(',')

            else:
                s = line.split(':')
                s1s = s[1].strip().split(',')
                metadata_dict = {}
                for (k, key) in enumerate(keys):
                    metadata_dict[key] = int(s1s[k])

                duration_matches[s[0]] = metadata_dict

    return duration_matches


def load_labelset_and_metadata(input_folder: str, labelset_name: str) \
        -> Tuple[Dict[str, List[Image.Image]], Dict[str, Dict[str, int]]]:
    """
    Loads a labelset and its metadata from the specified folder and returns it as two separate dictionaries

    # Arguments
        input_folder: str. the folder to load the labelset from
        labelset_name: str. the labelset name (and name of the subfolder to load the labelset from)

    # Returns
        Dict[str, List[Image]]. the labelset images as a dictionary of labels and corresponding lists of available images
        Dict[str, Dict[str, int]]. the labelset metadata as dictionary of labels and corresponding metadata information
    """

    input_folder = os.path.join(input_folder, labelset_name)

    data = load_labelset(input_folder)
    metadata = load_labelset_metadata(os.path.join(input_folder, '_'+labelset_name+'_metadata.txt'))

    return data, metadata


def load_random_label(images: Dict[str, List[Image.Image]], weights: List[int], metadata: Dict[str, Dict[str, int]]) \
        -> Tuple[str, List[Image.Image], Dict[str, int]]:
    """
    Selects a random label from a given labelset and returns it's available images and the metadata information

    # Arguments
        images: Dict[str, List[Image.Image]]. the image dictionary of the labelset
        weights: List[int]. the label weights of the labelset
        metadata: Dict[str, Dict[str, int]]. the metadata dictionary of the labelset

    # Returns
        str. the randomly selected label
        List[Image.Image]. the available images for the selected label
        Dict[str, int]]. the metadata information for the selected label
    """

    r = random.choices(list(images.keys()), weights=weights, k=1)[0]
    return r, images[r], metadata[r]


def visualize_bounding_boxes(img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> Image.Image:
    """
    Utility method for displaying bounding boxes on an image
    
    # Arguments
        img: Image. the image to display bounding boxes on
        boxes: List[List[int]] a list of bounding boxes (as tuples of edge points)
    
    # Returns
        Image. the input image with rendered bounding boxes
    """

    output = Image.new('RGBA', (img.width, img.height), 0)
    output.paste(img, (0, 0))

    # outputBox = Image.new('RGBA', (img.width, img.height), 0)
    draw = ImageDraw.Draw(output)
    for box_row in boxes:
        for b in box_row:
            draw.rectangle(b, outline="red")

    return output


def split_image_to_voices_with_given_center(image: Image.Image, image_height: Tuple[int, int],
                                            voice_center_position_y: List[int]) -> List[Image.Image]:
    """
    Splits an image into separate images for each voice centered on the given y-coordinates
    
    # Arguments
        image: Image. the image to split into voices
        image_height: Tuple[int, int]. the min and max height to crop each voice image to
        voice_positions: List[int]. the y-positions to center the voice images on
    
    # Returns
        List[Image]. the split voice images
    """

    img_rows = []

    for y in voice_center_position_y:
        result_height = random.randint(image_height[0], image_height[1])
        half_height = int(result_height / 2)

        y_start = y - half_height
        y_end = y + half_height

        if y_start < 0:
            y_start = 0
            y_end = result_height
        if y_end > image.height:
            y_start = image.height - result_height
            y_end = image.height

        img_rows.append(image.crop([0, y_start, image.width, y_end]))

    return img_rows


def download_source_images(data_url: str, data_size: int,
                           data_output_path: str = "../data",
                           data_zip_path: str = "../data/realdataSources/data.zip") -> None:
    """
    Downloads a zip file with source images from a specified url and extracts it to the data folder

    # Arguments
        data_url: str. the url to download the zip file from
        data_size: int. the size of the file to download (needed for download progress bar)
        data_output_path: str. the directory to extract the archive to
        data_zip_path: str. the path to save the downloaded zip archive at
    """

    data_zip_folder = os.path.split(data_zip_path)[0]
    if not os.path.exists(data_zip_folder):
        os.makedirs(data_zip_folder)

    with open(data_zip_path, 'wb') as f:
        response = requests.get(data_url, stream=True)
        total = response.headers.get('content-length')
        if total is not None:
            data_size = total

        print("Download of {}MB".format(round(data_size / (1024 * 1024))))

        downloaded = 0
        for data in response.iter_content(chunk_size=max(int(data_size / 1000), 1024 * 1024)):
            downloaded += len(data)
            f.write(data)

            display_progress(downloaded, data_size, ('MB', 1024 * 1024, 0))

    print()
    print("Download complete")

    with zipfile.ZipFile(data_zip_path, 'r') as z:
        z.extractall(data_output_path)

    print("Extraction complete")


def display_progress(current: int, total: int, unit: Tuple[str, int, int],
                     output: TextIO = sys.stdout, size: int = 100) -> None:
    """
    Displays a progress bar in a text output stream

    # Arguments
         current: int. the current value of the progress bar
         total: int. the max value of the progress bar
         unit: Tuple[str, int, int]. the unit the progress is measured at:
                                     (unit measurement string, unit measurement factor, number of digits after comma)
         output: TextIO. the output stream to write the progress bar to (stdout by default)
         size: int. the number of progression units on the progress bar
    """

    done = int(size * current / total)

    format_string = '\r[{:s}{:s}] ({:.'+str(unit[2])+'f} / {:.'+str(unit[2])+'f}{:s})'

    output.write(format_string.format(
        '█' * done,
        '.' * (size - done),
        current / unit[1],
        total / unit[1],
        unit[0]))
    output.flush()
