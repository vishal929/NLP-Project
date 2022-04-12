# logic for prepping cubs and coco datasets for use with the model
import random

import torch
import torch.utils.data as data
from PIL import Image
import os
import pickle
# may not need pandas here
import pandas as pd
import nltk.tokenize.regexp as RegexTokenizer
from random import randint
import numpy as np
# need to import torchvision for transformations on images
import torchvision.transforms as transforms
torch.manual_seed(17)

# establish vocabulary takes in a list of tokenized captions
# returns a mapping from words to an index in vocab, and vice versa, in that order
def establishVocabulary(captionList):
    # adding words to vocabulary
    vocab = []
    wordExists = {}
    for caption in captionList:
        for word in caption:
            if word not in wordExists:
                wordExists[word] = 1
                vocab.append(word)
    # generating mapping from word index to word and from word to word index
    wordToIndex = {}
    indexToWord = {}

    for index,w in vocab:
        wordToIndex[w] = index
        indexToWord[index] = w

    return wordToIndex, indexToWord





#setupCUB builds 2 dictionaries with the following format of information, for both test and train
# (ImageID, ImageNameFolder/ImageFileName.jpg, ImageClass, (x,y,width,height), [processedCaptions])

# in addition, a word mapping dictionaries are created, where words from both train and test are indexed
# the first dictionary is a mapping from word to index
# the second dictionary is a mapping from index to word

# function to place cropped images and preprocessed captions in their proper places
# then we can refer to train/test splits separately using the imageCaptionDataset class below
# imageIDPath is the path to the file mapping filenames of images to imageIDs
# imageClassPath is the path to the file mapping filenames of images to their classes
def setupCUB(sourceImageDir, sourceCaptionsDir, boundingBoxFilePath, trainTestSplitPath,
             imageIDPath, imageClassPath):
    # creating holder folders for train and test images and captions
    # NOTE, WE MAY NOT NEED THE BELOW DIRECTORIES, WE JUST HAVE TO HAVE A STANDARDIZED LOCATION FOR CUB IMAGES AND CAPTIONS
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

    # dictionary holding caption and image data formatted as:
    textImageData = {}
    trainTextImageData = {}
    testTextImageData = {}

    # reading the image index designation and updating a dictionary
    imageIDFile = open(imageIDPath,"r")
    imageIDMappings = imageIDFile.readlines()
    for line in imageIDMappings:
        # space seperated
        tokens = line.split(' ')
        # first token is imageID and second token is filename
        # mapping image ID to filename for the image
        textImageData[int(tokens[0])] = [tokens[1]]
    imageIDFile.close()

    # reading the class id file for each image class and updating a dictionary
    classIDFile = open(imageClassPath,"r")
    imageClassMappings = classIDFile.readlines()
    for line in imageClassMappings:
        # space separated
        tokens = line.split(' ')
        # first token is the image ID and the second token is the class mapping
        textImageData[int(tokens[0])].append(int(tokens[1]))
    classIDFile.close()

    # reading the train/test split file and updating a dictionary
    trainTestSplitFile = open(trainTestSplitPath,"r")
    imageTrainTestMappings = trainTestSplitFile.readlines()
    for line in imageTrainTestMappings:
        # space separated
        tokens = line.split(' ')
        # first token is image ID and the second token is 0 if its in test and 1 if its in training
        if int(tokens[1])==0:
            # its in test set
            testTextImageData[int(tokens[0])] = textImageData[int(tokens[0])]
        else:
            # its in training set
            trainTextImageData[int(tokens[0])] = textImageData[int(tokens[0])]
    trainTestSplitFile.close()

    # reading the bounding box file and updating a dictionary
    boundingBoxFile = open(boundingBoxFilePath,"r")
    boundingBoxFileLines = boundingBoxFile.readlines()
    for line in boundingBoxFileLines:
        # space separated
        tokens = line.split(' ')
        # first token is imageID
        # next tokens are x, y, width, and height respectively
        if int(tokens[0]) in trainTextImageData:
            # we update train dictionary
            # appending tuple of bounding box values
            trainTextImageData[int(tokens[0])].append((float(tokens[1]),
                                                       float(tokens[2]),
                                                       float(tokens[3]),
                                                       float(tokens[4])))
        elif int(tokens[0]) in testTextImageData:
            # we update test dictionary
            # appending tuple of bounding box values
            testTextImageData[int(tokens[0])].append((float(tokens[1]),
                                                      float(tokens[2]),
                                                      float(tokens[3]),
                                                      float(tokens[4])))
    boundingBoxFile.close()

    # for each image, crop it based on the bounding box, place it in the respective images folder
    # process its captions, update the dictionary with processed captions and place them in the captions folder
    # naming scheme: image is named by its image id
    # captions for each image is designated by the imageID.txt
    # i.e the processed captions for imageID 1 will be placed in 1.txt

    # processing training images and captions
    captionMapping= {}
    for trainImageID in trainTextImageData:
        # cropping the image and saving it into images
        '''
        img = Image.open(os.path.join(sourceImageDir,trainTextImageData[trainImageID][0])).convert('RGB')
        x = int(trainTextImageData[trainImageID][3][0])
        y = int(trainTextImageData[trainImageID][3][1])
        w = int(trainTextImageData[trainImageID][3][2])
        h = int(trainTextImageData[trainImageID][3][3])
        img = img.crop((x,y,x+w,y-h))
        # saving image under imageID
        img.save(os.path.join(os.path.join(trainDir,'Images'),trainTextImageData[trainImageID][1] + '.txt'))
        '''
        # tokenizing captions
        processedCaptions=[]
        captionFile = open(os.path.join(sourceCaptionsDir,trainTextImageData[trainImageID][0])[:-3]+'txt')
        captions = captionFile.readlines()
        for caption in captions:
            # tokenize captions
            tokenizedCaption=[]
            # paper uses a regexp tokenizer for pattern r'\w+'
            # this just means to separate all alphanumeric words and dont consider special characters like '$'
            tokenizer = RegexTokenizer(r'\w+')
            words = tokenizer.tokenize(caption.lower())
            for word in words:
                if len(word)>0:
                    tokenizedCaption.append(word)
            processedCaptions.append((trainImageID,tokenizedCaption))
        # need a mapping so we know which caption to associate with which image
        captionMapping[trainImageID] = processedCaptions
        captionFile.close()

    # processing test images and captions
    for testImageID in testTextImageData:
        '''
        img = Image.open(os.path.join(sourceImageDir, testTextImageData[testImageID][0])).convert('RGB')
        x = int(testTextImageData[testImageID][3][0])
        y = int(testTextImageData[testImageID][3][1])
        w = int(testTextImageData[testImageID][3][2])
        h = int(testTextImageData[testImageID][3][3])
        img = img.crop((x, y, x + w, y - h))
        # saving image under imageID
        img.save(os.path.join(os.path.join(testDir, 'Images'), testTextImageData[testImageID][1] + '.txt'))
        '''
        # tokenize captions
        processedCaptions=[]
        captionFile = open(os.path.join(sourceCaptionsDir, testTextImageData[testImageID][0])[:-3] + 'txt')
        captions = captionFile.readLines()
        for caption in captions:
            # tokenize caption
            tokenizedCaption = []
            # paper uses a regexp tokenizer for pattern r'\w+'
            # this just means to separate all alphanumeric words and dont consider special characters like '$'
            tokenizer = RegexTokenizer(r'\w+')
            words = tokenizer.tokenize(caption.lower())
            for word in words:
                if len(word) > 0:
                    tokenizedCaption.append(word)
            processedCaptions.append((testImageID,tokenizedCaption))
        captionMapping[testImageID]= processedCaptions
        captionFile.close()

    # create word to index mapping and index to word mapping and transform captions in that way
    wordToIndex, indexToWord = establishVocabulary(captionMapping.values())

    # transforming captions for both test and train to adhere to the vocab mappings
    for imgID in captionMapping:
        wordCaptions = captionMapping[imgID]
        intCaptions = []
        for caption in wordCaptions:
            tokenizedByInts = []
            for word in caption:
                tokenizedByInts.append(wordToIndex[word])
            intCaptions.append(tokenizedByInts)
        # now we have a representation of captions for an image by ints, lets set it in the train/test dictionary
        if imgID in trainTextImageData:
            # update train dictionary
            trainTextImageData[imgID].append(intCaptions)
        else:
            pass
            # update test dictionary
            testTextImageData[imgID].append(intCaptions)


    # save dictionary mappings for training
    with open('trainImagesCUB.pickle','wb') as handle:
        pickle.dump(trainTextImageData,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open('testImagesCUB.pickle','wb') as handle:
        pickle.dump(testTextImageData,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open('CUBWordToIndex','wb') as handle:
        pickle.dump(wordToIndex, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('CUBIndexToWord','wb') as handle:
        pickle.dump(indexToWord, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return trainTextImageData, testTextImageData


# coco requires less setup, just place train images and correlated captions
def setupCOCO():
    pass


# cubs has 10 descriptions per image, and coco has 5
# we basically need to match up descriptions to images
# cubs also includes bounding boxes for images
# the original paper normalizes images in the dataset
# the original paper caps the tokens in a caption to 18
class imageCaptionDataset(data.Dataset):
    def __init__(self, captionsDir, imagesDir, captionToImageRatio=10, imageTransform=None, split='train', maxCaptionLength = 18):
        # load pickle serializations for data grabbing
        # image data includes a list of captions for each image
        self.captionsDir = captionsDir
        self.imagesDir = imagesDir
        self.imageData={}
        self.wordToIndex = {}
        self.indexToWord = {}
        self.captionToImageRatio = captionToImageRatio
        self.imageTransform = imageTransform
        # normalizes each channel with mean specified in the first tuple, and standard deviation from the second tuple
        self.normalize = torch.nn.Sequential(
            transforms.toTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        )
        self.maxCaptionLength = maxCaptionLength

    def __len__(self):
        # we will return the number of images
        return len(self.imageData)

    def __getitem__(self,index):
        # randomly select an image
        imgData = random.choice(list(self.imageData.values()))
        # getting class identifier
        classID = imgData[1]
        # grab a random caption from the list of captions
        randCaption = random.choice(imgData[3])
        # load pixels of image into memory
        img = Image.open(os.path.join(self.imagesDir,imgData[0]))
        # perform any transforms needed on the image
        if imgData[2] is not None:
            bbox=imgData[2]
            # then bounding boxes, we have cub data
            width,height = img.size
            # TODO: understand how below code works with bounding boxes
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])

        if self.imageTransform is not None:
            img = self.imageTransform(img)

        # normalizing channels of the image
        img = self.normalize(img)

        # need to do just a tiny bit of work to the caption before returning
        numWords = len(randCaption)
        # just padding with zeroes
        padded = np.zeros((self.maxCaptionLength,1))

        # if the text is longer than 18, we need to rearrange the sentence so it makes sense with 18 tokens
        # so, all 18 tokens should be in order

        if numWords <= self.maxCaptionLength:
            # no issues here, our caption is less than the max
            padded[:numWords,0] = randCaption
        else:
            # grabbing indices of words as a list
            indices = np.arange(numWords)
            # shuffling the indices
            np.random.shuffle(indices)
            # limiting the indices
            indices = indices[:self.maxCaptionLength]
            # sorting the indices so the sentence order is preserved
            indices = np.sort(indices)
            # now just setting the pad based on the index
            for i in indices:
                padded[i,0] = randCaption[i]

        return img, padded, max(numWords,self.maxCaptionLength), classID








setupCUB('', '', '', '')
