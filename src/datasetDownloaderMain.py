# -*- coding: utf-8 -*-
"""
Main method for starting the organ tablature dataset generator.
The parameters for the generator are specified here

@author: Daniel Schneider
"""

import sys

from datasetGenerator.datasetGenerator.generatorUtility import download_source_images


def download_dataset(zip_path: str, output_path: str, delete_zip_file: bool = True) -> None:
    """
    Downloads a zip file with source images and extracts it to a specified folder
        
    # Arguments
        zip_path: str. the path to save the downloaded zip archive at
        output_path: str. the directory to extract the archive to
        delete_zip_file: bool. indicates whether the zip file shall be deleted after extraction
    """

    url = "https://box.uni-marburg.de/index.php/s/MENZtcfuWDeDHi8/download"
    data_size = 770034652  # for progress bar visualization only

    download_source_images(url, data_size, zip_path, output_path, delete_zip_file)


if __name__ == "__main__":
    """
    # Arguments
        1. zip_path: str. the path to save the downloaded zip file to
        2. output_path: str. the folder to extract the files to
        3. delete_zip_file: bool. indicates whether the zip file shall be deleted after extraction
    """

    download_dataset(zip_path=sys.argv[1], output_path=sys.argv[2], delete_zip_file=(sys.argv[3] == 'True'))
