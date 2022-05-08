import pathlib

import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image

# pytorch is not the recommended way to calculate IS score please use this repo instead (OpenAI code implementation):
#https://github.com/openai/improved-gan/blob/master/inception_score/model.py

# calculating Inception Score for GANs

# idea is to compute KL divergence between image label distribution and the generated marginal distribution

# IS = exp(Expected_x KL(p(y|x) || p(y)) )

# p(y) -> y is the label
# p(y|x) -> label given an image

# split images into groups
# the paper just uses the openAI code for calculating IS

# grabbing the inception v3 model for evaluating inception score
def getInceptionModel():
    # we will allow the inception model to transform input in this case
    inceptionModel = torchvision.models.inception_v3(pretrained=True)
    inceptionModel.eval()
    return inceptionModel


def inceptionScore(conditionalProb, epsilon=1e-8):
    marginalProb = conditionalProb.mean(dim=0)

    # calculating kl divergence between marginal and conditional probs for each image
    kl = conditionalProb * (torch.log(conditionalProb + epsilon) - torch.log(marginalProb + epsilon))

    # sum kl divergence over labels and average for each image
    kl = torch.mean(kl.sum(dim=1))
    return torch.exp(kl)


def getFileNames(directory):
    # just gets filenames for all png and jpg in a directory
    path = pathlib.Path(directory)
    images = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    return images

def loadImages(filenames):
    # just loads image data into a tensor
    imgs = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        imgs.append(img)
    # returning tensor
    return torch.tensor(imgs)



# IS score only operates on the generated images, not the real ones
# images arguments here is just a list of filenames from getFileNames function
def isScore(images,device):
    inceptionV3 = getInceptionModel().to(device)
    inceptionV3.eval()

    conditionalProbs = []
    batchSize = 30
    for batchNum in tqdm(range(batchSize)):
        batchFileNames = images[(batchNum*batchSize):(batchNum*batchSize)+batchSize]
        # loading images from a batch of filenames
        imgs = loadImages(batchFileNames)
        conditionalProb = inceptionV3(imgs)
        conditionalProbs.append(conditionalProb)

    #conditionalProbs = inceptionV3(images)

    # concatenating tensors by batch dim
    conditionalProbs = torch.cat(conditionalProbs,dim=0)


    # splitting into 10 parts for evaluation, as is the methodology in the Inception Score paper
    # NOTE: the author of this paper uses 3 as the split size
    split = 3

    splitSize = torch.floor(conditionalProbs.shape[0] / split)

    conditionalProbs = torch.split(conditionalProbs, split_size_or_sections=splitSize, dim=0)

    # asserting we should have exactly split parts

    if len(conditionalProbs) != split:
        print('IS Score dimension assert triggered!')

    allScores = []
    for conditionalProbBatch in conditionalProbs:
        batchScore = inceptionScore(conditionalProbBatch)
        allScores.append(batchScore)

    scoreAvg, scoreSTD = np.mean(allScores), np.std(allScores)

    return scoreAvg, scoreSTD



