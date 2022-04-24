import torch
import torchvision
import numpy as np


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


# IS score only operates on the generated images, not the real ones
def isScore(images):
    # I think we should have enough vram to process all the generated images at once, lets try this
    inceptionV3 = getInceptionModel()
    inceptionV3.eval()

    conditionalProbs = inceptionV3(images)

    # splitting into 10 parts for evaluation, as is the methodology in the Inception Score paper

    splitSize = torch.floor(conditionalProbs.shape[0] / 10)

    conditionalProbs = torch.split(conditionalProbs, split_size_or_sections=splitSize)

    # asserting we should have exactly 10 parts

    if len(conditionalProbs) != 10:
        print('IS Score dimension assert triggered!')

    allScores = []
    for conditionalProbBatch in conditionalProbs:
        batchScore = inceptionScore(conditionalProbBatch)
        allScores.append(batchScore)

    scoreAvg, scoreSTD = np.mean(allScores), np.std(allScores)
