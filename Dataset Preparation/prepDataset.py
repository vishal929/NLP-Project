# logic for prepping cubs and coco datasets for use with the model
import torch.utils.data as data
from PIL import Image
import os
import pickle
# may not need pandas here
import pandas as pd
import nltk.tokenize.regexp as RegexTokenizer

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
            # update test dictionary
            testTextImageData[imgID].append(intCaptions)


    # we can update the dictionary and save it in case it becomes handy for faster training
    with open('trainCUB.pickle','wb') as handle:
        pickle.dump(trainTextImageData,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open('testCUB.pickle','wb') as handle:
        pickle.dump(testTextImageData,handle,protocol=pickle.HIGHEST_PROTOCOL)

    return trainTextImageData, testTextImageData


# coco requires less setup, just place train images and correlated captions
def setupCOCO():
    pass


# cubs has 10 descriptions per image, and coco has 5
# we basically need to match up descriptions to images
# cubs also includes bounding boxes for images
class imageCaptionDataset(data.Dataset):
    def __init__(self, captionsDir, imagesDir):
        pass


setupCUB('', '', '', '')
