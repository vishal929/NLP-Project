# function for generating images based on user input and interactivity
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

import DatasetPreparation.prepDataset as prepData
from Losses.Loss import adv_D, adv_G, calculateAttentionMatchingScoreBatchWrapper, totalLoss
from Modules.Discriminator import Discriminator
from Modules.Generator import Generator
from TextEncoder.RNNEncoder import bilstmEncoder
from ImageEncoder.CNNEncoder import cnnEncoder
from tqdm import tqdm
import PIL.Image as Image
import nltk.tokenize.regexp as RegexTokenizer

# transforms user input into a valid padded caption for feeding into the text encoder
def transformUserInput(userInput, wordToIndexVocab):
    padded = np.zeros(18)
    newText = []
    numWords = 0
    # paper uses a regexp tokenizer for pattern r'\w+'
    # this just means to separate all alphanumeric words and dont consider special characters like '$'
    tokenizer = RegexTokenizer.RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(userInput.lower())
    for word in words:
        if len(word) > 0:
            if word in wordToIndexVocab:
                # only append words that exist in vocabulary
                newText.append(wordToIndexVocab[word])
                numWords += 1
    if numWords>18:
        # problem here, we need to randomly pick some words, but preserve word order
        indices = np.arange(newText)
        # shuffling indices
        np.random.shuffle(indices)
        indices = indices[:18]
        # preserving word order
        indices = np.sort(indices)
        for index,actualIndex in enumerate(indices):
            padded[index] = newText[actualIndex]
        # capping numWords at 18
        numWords = 18
    else:
        # no problem, we can fit all the words in the caption
        padded[:numWords] = newText
    return torch.tensor(padded), torch.tensor(numWords)

def evaluate(generator,textEncoder, device, wordToIndexVocab):

    # evaluating based on user input
    # grabbing user caption:
    while True:
        with torch.no_grad():
            encodedCaption, captionLength = transformUserInput(
                input("enter a caption! (preferably about birds from CUB and <= 18 words)"),
                wordToIndexVocab)

            # goes from [18] to [1,18] for both length and caption
            encodedCaption = encodedCaption.unsqueeze(0).type(torch.LongTensor)
            captionLength = captionLength.unsqueeze(0).type(torch.LongTensor)

            # sending data to gpu
            encodedCaption = encodedCaption.to(device)
            captionLength = captionLength.to(device)

            # resetting bilstm encoder hidden state
            newHidden = textEncoder.initHiddenStates(1)

            wordEmbeddings=None
            sentEmbeddings=None

            # getting embeddings
            wordEmbeddings, sentEmbeddings = textEncoder(encodedCaption,captionLength,newHidden)

            # Make standard gaussian noise
            z_shape = (1, 100)
            z = torch.normal(0.0, torch.ones(z_shape)).to(device)

            # Forward pass for generator to get generated imgs conditioned on embedded text
            x_fake = generator(z, sentEmbeddings)
            x_fake = (x_fake + 1.0) * 127.5
            x_fake = x_fake.squeeze(0).cpu().numpy().astype(np.uint8)
            x_fake = x_fake.transpose(1,2,0)
            Image.fromarray(x_fake).show('Generated Image')



if __name__ == '__main__':
    # transform for images
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(256 * 76 / 64)),
            torchvision.transforms.RandomCrop(256),
            torchvision.transforms.RandomHorizontalFlip()])

    # grab dataset
    cubImageDir = os.path.join(os.getcwd(),'CUBS Dataset','Cubs-2011','cub-200-2011-20220408T185459Z-001',
                               'cub-200-2011','CUB_200_2011','CUB_200_2011','images')
    '''
    cubImageDir = '../CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/CUB_200_2011/CUB_200_2011/' \
                  'images'
    '''
    cubCaptionDir = os.path.join(os.getcwd(),'CUBS Dataset','Cubs-2011','bird_metadata','birds','text','text')

    # place where cub metadata is
    cubBoundingBoxFile = os.path.join(os.getcwd(),'CUBS Dataset','Cubs-2011','cub-200-2011-20220408T185459Z-001',
                               'cub-200-2011','CUB_200_2011','CUB_200_2011','bounding_boxes.txt')
    '''
    cubBoundingBoxFile = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                         'CUB_200_2011/CUB_200_2011/bounding_boxes.txt'
    '''
    '''
    cubFileMappings = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                      'CUB_200_2011/CUB_200_2011/images.txt'
    '''
    cubFileMappings = os.path.join(os.getcwd(), 'CUBS Dataset', 'Cubs-2011',
                                      'cub-200-2011-20220408T185459Z-001',
                                      'cub-200-2011', 'CUB_200_2011', 'CUB_200_2011', 'images.txt')
    cubClasses = os.path.join(os.getcwd(), 'CUBS Dataset', 'Cubs-2011',
                                   'cub-200-2011-20220408T185459Z-001',
                                   'cub-200-2011', 'CUB_200_2011', 'CUB_200_2011', 'image_class_labels.txt')
    '''
    cubClasses = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                 './CUB_200_2011/image_class_labels.txt'
    '''
    cubTrainTestSplit = os.path.join(os.getcwd(), 'CUBS Dataset', 'Cubs-2011',
                              'cub-200-2011-20220408T185459Z-001',
                              'cub-200-2011', 'CUB_200_2011', 'CUB_200_2011', 'train_test_split.txt')
    '''
    cubTrainTestSplit = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                        'CUB_200_2011/CUB_200_2011/train_test_split.txt'
    '''

    pickleDir = os.path.join(os.getcwd(),'DatasetPreparation','CUBMetadata')

    # setting up dataset if not setup already (i.e pickles dont exist)
    #prepData.setupCUB(cubCaptionDir,cubBoundingBoxFile,cubTrainTestSplit,cubFileMappings,cubClasses,pickleDir)

    # grabbing dataset
    # we need the dataset in order to define our vocabulary
    cubDataset = prepData.imageCaptionDataset(cubCaptionDir, cubImageDir, pickleDir,10, transform, 'train', 18)


    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initializing modules needed in framework
    # z_shape = (batchSize, 100)
    z_shape = (64,100)
    generator = Generator(z_shape=z_shape).to(device)

    # vocab_size,
    # input length (caption words are sent to an embedding of this length),
    # dropout probability
    # hidden features of both directions of the bilstm
    # stepSize is the length of caption inputs -> 18
    text_encoder = bilstmEncoder(len(cubDataset.indexToWord), 300, 0.5, 256, 18)

    # loading pre trained parameters of the text_encoder for cub
    cubTextEncoderPath = os.path.join(os.getcwd(), 'CubDamsm', 'text_encoder200.pth')
    #cubTextEncoderPath = "../CubDamsm/text_encoder200.pth"
    textEncoderState = torch.load(cubTextEncoderPath, map_location=lambda storage, loc: storage)
    modifiedState = OrderedDict()
    for key, values in textEncoderState.items():
        if key[:7] == 'encoder':
            # replace this name
            key = 'embedding' + key[7:]
        elif key[:3] == 'rnn':
            key = 'LSTM' + key[3:]
            # replace this name
        modifiedState[key] = values

    # loading saved model params to the text encoder
    text_encoder.load_state_dict(modifiedState)

    # removing references
    del textEncoderState
    del modifiedState

    # sending text encoder to gpu
    text_encoder.to(device)

    # setting parameters to be fixed
    for param in text_encoder.parameters():
        param.requires_grad = False

    text_encoder.eval()

    # loading training status for cub
    trainFile = os.path.join(os.getcwd(),'DatasetPreparation','WithDAMSMTrainingCheckpoint.pth')
    #trainFile = 'cubTrainingCheckpoint.pth'
    if os.path.isfile(trainFile):
        print('loaded state!')
        checkpoint = torch.load(trainFile)

        generator.load_state_dict(checkpoint['generator_state_dict'])

        # removing reference
        del checkpoint

    # setting generator to eval
    generator.eval()


    # test loop, interactively asking user to input some text into standard in
    evaluate(generator, text_encoder, device, cubDataset.wordToIndex)



