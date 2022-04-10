# logic for prepping cubs and coco datasets for use with the model
import torch.utils.data as data
from PIL import Image
import os
import pandas as pd


# function to place cropped images and preprocessed captions in their proper places
# then we can refer to train/test splits separately using the imageCaptionDataset class below
def setupCUB(sourceImageDir, sourceCaptionsDir, boundingBoxFilePath, trainTestSplitPath):
    # creating holder folders for train and test images and captions
    if (not os.path.isdir('../CUBS Dataset')):
        os.mkdir('../CUBS Dataset')
    dataDir = '../CUBS Dataset'
    trainDir = os.path.join(dataDir,'Train')
    testDir = os.path.join(dataDir,'Test')
    if (not os.path.isdir(trainDir)):
        os.mkdir(trainDir)
    if (not os.path.isdir(testDir)):
        os.mkdir(testDir)
    if (not os.path.isdir(os.path.join(trainDir,'Images'))):
        os.mkdir(os.path.join(trainDir,'Images'))
    if (not os.path.isdir(os.path.join(trainDir,'Captions'))):
        os.mkdir(os.path.join(trainDir,'Captions'))
    if (not os.path.isdir(os.path.join(testDir, 'Images'))):
        os.mkdir(os.path.join(testDir, 'Images'))
    if (not os.path.isdir(os.path.join(testDir, 'Captions'))):
        os.mkdir(os.path.join(testDir, 'Captions'))

    # reading the image index designation and updating a dictionary

    # reading the class id file for each image class and updating a dictionary

    # reading the train/test split file and updating a dictionary

    # reading the bounding box file and updating a dictionary

    # for each image, crop it based on the bounding box, place it in the images folder
    # process its captions, update the dictionary with processed captions and place them in the captions folder
    # naming scheme: image is named by its image id
    # captions for each image is designated by the imageID.txt
    # i.e the processed captions for imageID 1 will be placed in 1.txt

    # we can update the dictionary and save it in case it becomes handy for faster training
    pass


# coco requires less setup, just place train images and correlated
def setupCOCO():
    pass


# cubs has 10 descriptions per image, and coco has 5
# we basically need to match up descriptions to images
# cubs also includes bounding boxes for images
class imageCaptionDataset(data.Dataset):
    def __init__(self, captionsDir, imagesDir):
        pass


setupCUB('', '', '', '')
