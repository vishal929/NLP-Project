# file for testing the generator on the cub dataset
import torch
import os
from collections import OrderedDict

from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

import DatasetPreparation.prepDataset as prepData
from Losses.Loss import adv_D, adv_G, calculateAttentionMatchingScoreBatchWrapper, totalLoss
from Modules.Discriminator import Discriminator
from Modules.Generator import Generator
from TextEncoder.RNNEncoder import bilstmEncoder
from ImageEncoder.CNNEncoder import cnnEncoder
from tqdm import tqdm

# dont need discriminator or image encoder for testing, we are just saving generated images basically
def test(dataloader, generator, textEncoder, device, batch_size):

    # testing, or "saving" generated images

    # summary writer for keeping track of testing
    writer = SummaryWriter('testStuff')
    dataIterator = iter(dataloader)
    for step in tqdm(range(len(dataIterator))):
        with torch.no_grad():
            testImages, testCaptions, captionLengths, classID  = dataIterator.next()
            testCaptions = testCaptions.type(torch.LongTensor)
            captionLengths = captionLengths.type(torch.LongTensor)

            # squeezing labels so they are (batchSize, 18) instead of (batchsize,18,1)
            testCaptions = testCaptions.squeeze()

            # sending data to gpu
            #trainImages = trainImages.to(device)
            testCaptions = testCaptions.to(device)
            captionLengths = captionLengths.to(device)

            # resetting bilstm encoder hidden state
            newHidden = textEncoder.initHiddenStates(batch_size)

            wordEmbeddings=None
            sentEmbeddings=None

            # getting embeddings
            wordEmbeddings, sentEmbeddings = textEncoder(testCaptions,captionLengths,newHidden)

            # detaching so gradient doesnt flow back to textEncoder
            #wordEmbeddings = wordEmbeddings.detach()
            #sentEmbeddings = sentEmbeddings.detach()

            # Make standard gaussian noise
            z_shape = (batch_size, 100)
            #with torch.no_grad():
            z = torch.normal(0.0, torch.ones(z_shape)).to(device)

            # Forward pass for generator to get generated imgs conditioned on embedded text
            x_fake = generator(z, sentEmbeddings)

            # saving generatedImage, realImage, and associated caption

            #img = trainImages[0].to(device)

            #write_images_losses(writer, img, x_fake, L_advD, L_advD, genLoss, finalLoss, epoch)

if __name__ == '__main__':
    # grab dataset
    cubImageDir = os.path.join(os.getcwd(),'..','CUBS Dataset','Cubs-2011','cub-200-2011-20220408T185459Z-001',
                               'cub-200-2011','CUB_200_2011','CUB_200_2011','images')
    '''
    cubImageDir = '../CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/CUB_200_2011/CUB_200_2011/' \
                  'images'
    '''
    cubCaptionDir = os.path.join(os.getcwd(),'..','CUBS Dataset','Cubs-2011','bird_metadata','birds','text','text')

    # place where cub metadata is
    cubBoundingBoxFile = os.path.join(os.getcwd(),'..','CUBS Dataset','Cubs-2011','cub-200-2011-20220408T185459Z-001',
                               'cub-200-2011','CUB_200_2011','CUB_200_2011','bounding_boxes.txt')
    '''
    cubBoundingBoxFile = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                         'CUB_200_2011/CUB_200_2011/bounding_boxes.txt'
    '''
    '''
    cubFileMappings = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                      'CUB_200_2011/CUB_200_2011/images.txt'
    '''
    cubFileMappings = os.path.join(os.getcwd(), '..', 'CUBS Dataset', 'Cubs-2011',
                                      'cub-200-2011-20220408T185459Z-001',
                                      'cub-200-2011', 'CUB_200_2011', 'CUB_200_2011', 'images.txt')
    cubClasses = os.path.join(os.getcwd(), '..', 'CUBS Dataset', 'Cubs-2011',
                                   'cub-200-2011-20220408T185459Z-001',
                                   'cub-200-2011', 'CUB_200_2011', 'CUB_200_2011', 'image_class_labels.txt')
    '''
    cubClasses = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                 './CUB_200_2011/image_class_labels.txt'
    '''
    cubTrainTestSplit = os.path.join(os.getcwd(), '..', 'CUBS Dataset', 'Cubs-2011',
                              'cub-200-2011-20220408T185459Z-001',
                              'cub-200-2011', 'CUB_200_2011', 'CUB_200_2011', 'train_test_split.txt')
    '''
    cubTrainTestSplit = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                        'CUB_200_2011/CUB_200_2011/train_test_split.txt'
    '''

    pickleDir = os.path.join(os.getcwd(),'CUBMetadata')

    # setting up dataset if not setup already (i.e pickles dont exist)
    prepData.setupCUB(cubCaptionDir,cubBoundingBoxFile,cubTrainTestSplit,cubFileMappings,cubClasses,pickleDir)

    # grabbing dataset
    batchSize = 4
    cubDataset = prepData.imageCaptionDataset(cubCaptionDir, cubImageDir, pickleDir,10, None, 'test', 18)

    cubDataLoader = DataLoader(cubDataset,batch_size=batchSize, drop_last=True, shuffle=True, num_workers=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initializing modules needed in framework
    # z_shape = (batchSize, 100)
    z_shape = (64,100)
    in_dim = (batchSize, 256, 256, 3)
    generator = Generator(z_shape=z_shape).to(device)
    #discriminator = Discriminator(in_dim).to(device)
    #discriminator = Discriminator().to(device)

    # vocab_size,
    # input length (caption words are sent to an embedding of this length),
    # dropout probability
    # hidden features of both directions of the bilstm
    # stepSize is the length of caption inputs -> 18
    text_encoder = bilstmEncoder(len(cubDataset.indexToWord), 300, 0.5, 256, 18)

    # loading pre trained parameters of the text_encoder for cub
    cubTextEncoderPath = "../CubDamsm/text_encoder200.pth"
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

    # sending text encoder to gpu
    text_encoder.to(device)

    # setting parameters to be fixed
    for param in text_encoder.parameters():
        param.requires_grad = False

    text_encoder.eval()

    # try loading the image encoder
    '''
    image_encoder = cnnEncoder()

    # loading pre trained weights to the image encoder
    cubImageEncoderPath = "../CubDamsm/image_encoder200.pth"
    imageEncoderState = torch.load(cubImageEncoderPath, map_location=lambda storage, loc: storage)
    imageModifiedState = OrderedDict()
    for key, values in imageEncoderState.items():
        if key[0:13] == 'Conv2d_1a_3x3':
            key = 'inceptionUpToLocalFeatures.0' + key[13:]
        elif key[0:13] == 'Conv2d_2a_3x3':
            key = 'inceptionUpToLocalFeatures.1' + key[13:]
        elif key[0:13] == 'Conv2d_2b_3x3':
            key = 'inceptionUpToLocalFeatures.2' + key[13:]
        elif key[0:13] == 'Conv2d_3b_1x1':
            key = 'inceptionUpToLocalFeatures.4' + key[13:]
        elif key[0:13] == 'Conv2d_4a_3x3':
            key = 'inceptionUpToLocalFeatures.5' + key[13:]
        elif key[0:8] == 'Mixed_5b':
            key = 'inceptionUpToLocalFeatures.7' + key[8:]
        elif key[0:8] == 'Mixed_5c':
            key = 'inceptionUpToLocalFeatures.8' + key[8:]
        elif key[0:8] == 'Mixed_5d':
            key = 'inceptionUpToLocalFeatures.9' + key[8:]
        elif key[0:8] == 'Mixed_6a':
            key = 'inceptionUpToLocalFeatures.10' + key[8:]
        elif key[0:8] == 'Mixed_6b':
            key = 'inceptionUpToLocalFeatures.11' + key[8:]
        elif key[0:8] == 'Mixed_6c':
            key = 'inceptionUpToLocalFeatures.12' + key[8:]
        elif key[0:8] == 'Mixed_6d':
            key = 'inceptionUpToLocalFeatures.13' + key[8:]
        elif key[0:8] == 'Mixed_6e':
            key = 'inceptionUpToLocalFeatures.14' + key[8:]
        elif key[0:8] == 'Mixed_7a':
            key = 'restOfInception.0' + key[8:]
        elif key[0:8] == 'Mixed_7b':
            key = 'restOfInception.1' + key[8:]
        elif key[0:8] == 'Mixed_7c':
            key = 'restOfInception.2' + key[8:]
        elif key[0:12] == 'emb_features':
            key = 'embedLocalFeatures' + key[12:]
        elif key[0:12] == 'emb_cnn_code':
            key = 'embedGlobalFeatures' + key[12:]
        imageModifiedState[key] = values

    image_encoder.load_state_dict(imageModifiedState)

    image_encoder = image_encoder.to(device)

    for param in image_encoder.parameters():
        param.requires_grad = False

    image_encoder.eval()

    numEpoch = 0

    # setting adam optimizer for generator and discriminator
    optimizerG = torch.optim.Adam(generator.parameters(),lr=0.0001, betas=(0.0,0.9))
    optimizerD = torch.optim.Adam(discriminator.parameters(),lr=0.0004, betas=(0.0, 0.9))
    '''

    # loading training status for cub
    trainFile = 'cubTrainingCheckpoint.pth'

    if (os.path.isfile(os.path.join(os.getcwd(),trainFile))):
        print('loaded state!')
        checkpoint = torch.load('cubTrainingCheckpoint.pth')

        generator.load_state_dict(checkpoint['generator_state_dict'])
        #discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        #optimizerG.load_state_dict(checkpoint['optimizer_generator_state_dict'])
        #optimizerD.load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
        generatorLoss = checkpoint['gLoss']
        discriminatorLoss = checkpoint['dLoss']
        numEpoch = checkpoint['epoch']

    discriminatorLoss = None
    generatorLoss = None

    # training loop
    # saving state every 10 epochs
    test(cubDataLoader, generator, text_encoder,  device,  batchSize)


