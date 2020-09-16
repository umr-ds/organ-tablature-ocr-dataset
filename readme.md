# Organ Tablature OCR Dataset

This repository provides the organ tablature ocr data set used in the paper `Recognition of Organ Tablature Music Notation with Deep Neural Networks`.
The data set consists of a training, validation and testing subset.


### Data Sources
The data consists on the one hand of real tablature staves extracted from the scans of two organ tablature books: 
* "Orgel oder Instrument Tabulaturbuch" (’Organ or InstrumentTablature Book’) from 1583 
* "Ein  new künstlich Tabulaturbuch" (’A new artificial tablaturebook’)  from  1575 

Both books were written by German Organist and Composer Elias Nikolaus Ammerbach.
1,200 staves from each book were manually annotated with label sequences. 

On the other hand the data set is made up of artificially generated tablatures that were produced by a data generator.


### Data Generator
The whole data set requires almost 300GB of disc space, which is why instead of the whole data set the random data generator is distributed along with the required source images.
The annotated real tablature staves are made available in a cloud storage and can be downloaded from there.

The `datasets.ipynb` ipython notebook is provided for building the data set locally.
It automatically downloads the required data and runs the data generator and data augmentor with the appropriate parameters to create the training, validation and test set.
Further instructions are provided in the notebook itself.

The following packages need to be installed to run the generator:
* **Jupyter**: `pip install jupyter`
* **Numpy**: `pip install numpy`
* **Pillow**: `pip install Pillow`
* **Augmentor**: `pip install Augmentor`


## Folder Structure
The `src` folder contains all the python code of the data generator and data augmentor program.

The `data` directory contains all tablature images and is structured as follows:
* `generatorSources`: Contains source files for the tablature generator
    * `backgrounds`: source images for backgrounds and image borders
    * `duration`: source images for duration tablature characters
    * `note`: source images for note pitch tablature characters
    * `rest`: source images for rest tablature characters
    * `special`: source images for special tablature characters (measure lines, repetition signs, text blocks, ...)
* `realdataSources`: Will be created during data set building. Will contain annotated real organ tablature staves from the two tablature books
* `generatorOutput`: Will be created during data set building. Will serve as output directory for the data generator
* `datasetOutput`: Will be created during data set building. Will serve as output directory for the final data sets

