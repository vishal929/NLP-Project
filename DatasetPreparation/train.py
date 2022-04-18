# simply file for local training
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import prepDataset
from Losses.Loss import adv_D, adv_G
from Modules.Discriminator import Discriminator
from Modules.Generator import  Generator
from TextEncoder.RNNEncoder import bilstmEncoder
from ImageEncoder.CNNEncoder import cnnEncoder



def train(dataloader, generator, discriminator, textEncoder, imageEncoder, device, optimizerG, optimizerD,
          epochNum, batch_size, discriminatorLoss, generatorLoss):

    # training

    # saving training status at certain intervals
    torch.save({
        'epoch': epochNum,
        'generator_state_dict': generator.state_dict(),
        'optimizer_generator_state_dict': optimizerG.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_discriminator_state_dict': optimizerD.state_dict(),
        'gLoss': generatorLoss,
        'dLoss': discriminatorLoss
    }, 'trainingCheckpoint.pth')
    return 0

# transform for images
transform = transforms.Compose([
        transforms.Resize(int(256 * 76 / 64)),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip()])

# grab dataset
cubImageDir = '../CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/CUB_200_2011/CUB_200_2011/' \
              'images'
cubCaptionDir = '../CUBS Dataset/Cubs-2011/bird_metadata/birds/text/text'

# place where cub metadata is
cubBoundingBoxFile = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                     'CUB_200_2011/CUB_200_2011/bounding_boxes.txt'
cubFileMappings = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                  'CUB_200_2011/CUB_200_2011/images.txt'
cubClasses = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
             './CUB_200_2011/image_class_labels.txt'
cubTrainTestSplit = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                    'CUB_200_2011/CUB_200_2011/train_test_split.txt'

# setting up dataset if not setup already (i.e pickles dont exist)

# grabbing dataset
cubDataset = prepDataset.imageCaptionDataset(cubCaptionDir, cubImageDir, 10, transform, 'train', 18)

cubDataLoader = DataLoader(cubDataset,batch_size=24, drop_last=True, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initializing modules needed in framework

generator = Generator().to(device)
discriminator = Discriminator().to(device)

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
text_encoder.cuda()

# setting parameters to be fixed
for param in text_encoder.parameters():
    param.requires_grad = False

text_encoder.eval()

# try loading the image encoder
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

image_encoder = image_encoder.cuda()

for param in image_encoder.parameters():
    param.requires_grad = False

image_encoder.eval()

numEpoch = 0

# setting adam optimizer for generator and discriminator
optimizerG = torch.optim.Adam(generator.parameters(),lr=0.0001, betas=(0.0,0.9))
optimizerD = torch.optim.Adam(discriminator.parameters(),lr=0.0004, betas=(0.0, 0.9))

# loading training status
checkpoint = torch.load('trainingCheckpoint.pth')

generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
optimizerG.load_state_dict(checkpoint['optimizer_generator_state_dict'])
optimizerD.load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
generatorLoss = checkpoint['gLoss']
discriminatorLoss = checkpoint['dLoss']
numEpoch = checkpoint['epoch']

# training loop
train(cubDataLoader, generator, discriminator,text_encoder, image_encoder, device, optimizerG, optimizerD,
      numEpoch, 24, discriminatorLoss, generatorLoss)



