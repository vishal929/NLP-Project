# calculating Frechet Inception Distance for GAN
# FID score uses the inception v3 model classification pretrained on imagenet
import os
import pathlib

import torch
import torchvision
import numpy as np
import scipy.linalg._matfuncs_sqrtm as sqrtm
from tqdm import tqdm
from PIL import Image

# pytorch is not the official way to calculate FID score
# please use this repo instead (from original authors of FID Score):
# https://github.com/bioinf-jku/TTUR

def getInceptionV3():
    # we will let inception transform our images before evaluation
    states = torchvision.models.inception_v3(pretrained=True).state_dict()
    for state in states:
        print(state)
    '''
    inceptionPoolModel = torch.nn.Sequential(*list(
        torchvision.models.inception_v3(pretrained=True)
            .children()
    )[:-1])
    '''
    inceptionPoolModel = torchvision.models.inception_v3(pretrained=True)

    # eval for no dropout
    inceptionPoolModel.eval()

    return inceptionPoolModel

def getFileNames(directory):
    # just gets filenames for all png and jpg in a directory
    path = pathlib.Path(directory)
    images = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    return images

# loads a batch of images by filename
def loadFilenames(filenames):
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),
        torchvision.transforms.CenterCrop(299),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # just loads image data into a tensor
    imgs = []
    for filename in filenames:
        img = Image.open(filename).convert('RGB')
        imgs.append(preprocess(img))
    # returning tensor as a batch of image data
    return torch.stack(imgs,dim=0)

# calculating fid given mean and covariance features of the generated set, and of the real set
def fidScore(realMeanFeature, generatedMeanFeature, realGeneratedCovariance, generatedRealCovariance):
    # ||mu_x-mu_y||^2
    realMeanFeature = realMeanFeature.numpy()
    generatedMeanFeature = generatedMeanFeature.numpy()
    realGeneratedCovariance = realGeneratedCovariance.numpy()
    generatedRealCovariance = generatedRealCovariance.numpy()

    meanDistance = np.sum((generatedMeanFeature - realMeanFeature) ** 2)
    # Cov_x + Cov_y - 2 sqrt(Cov_x Cov_y)

    covSqrt = sqrtm(realGeneratedCovariance.dot(generatedRealCovariance))

    # handling imaginary elements from the matrix sqrt
    covSqrt = covSqrt.real

    covarianceTerm = (realGeneratedCovariance + generatedRealCovariance) - (2 * covSqrt)
    return meanDistance + torch.trace(covarianceTerm)


def getActivations(realImages, generatedImages, inceptionPoolModel):
    # loading the inceptionV3 Model but without the fully connected layer (only up to avg pool )


    # sending both real images and generated images through inception to get pool representations
    # may need to interpolate to 299x299 before sending images through inception

    realActivations = inceptionPoolModel(realImages)
    fakeActivations = inceptionPoolModel(generatedImages)

    # returning both realActivations and fakeActivations
    return realActivations, fakeActivations

# realImages and generatedImages here are just filenames from the getFileNames function
def getFIDScore(realImages,generatedImages,device):
    # split realImages and generatedImages into batches of splitSize
    #splitSize= 4
    #(splits into equal 100 splits of the original # of examples)
    # i.e each index here represents (100,255,255,3) where we have 100 examples of 255x255 images
    #print(realImages)
    #realImages = np.array_split(realImages,splitSize)
    #realImages = torch.split(realImages,split_size_or_sections=splitSize)

    #generatedImages = np.array_split(generatedImages,splitSize)
    #generatedImages = torch.split(generatedImages, split_size_or_sections=splitSize)

    # getting inception v3 model for activations
    inceptionPoolModel = getInceptionV3().to(device)

    allRealActivations = []
    allFakeActivations = []

    batchSize = 2
    realBatches = len(realImages) // batchSize
    fakeBatches = len(generatedImages) // batchSize

    # getting real activations
    for i in tqdm(range(realBatches)):
        realFileNames = realImages[i*batchSize:(i*batchSize)+batchSize]
        # reading images as a batch
        realImageBatch = loadFilenames(realFileNames).type(torch.FloatTensor).to(device)
        realActivations = inceptionPoolModel(realImageBatch)
        # shape should be (numExamples,2048) where 2048 is the feature vector length
        allRealActivations.append(realActivations.to('cpu'))

    # getting fake activations
    for i in tqdm(range(fakeBatches)):
        generatedFileNames = generatedImages[i*batchSize:(i*batchSize)+batchSize]
        # reading images as a batch
        generatedImageBatch = loadFilenames(generatedFileNames).type(torch.FloatTensor).to(device)

        generatedActivations= inceptionPoolModel(generatedImageBatch)
        # shape should be (numExamples,2048) where 2048 is the feature vector length
        allFakeActivations.append(generatedActivations)

    # stacking real and fake activations into the overall matrix where first
    allRealActivations = torch.cat(allRealActivations,dim=0)
    allFakeActivations = torch.cat(allFakeActivations,dim=0)

    # running statistics on the activations to get mean and covariance required for FID score
    allRealMean = torch.mean(allRealActivations,dim=0)
    allFakeMean = torch.mean(allFakeActivations, dim=0)
    allRealCov = torch.cov(allRealActivations)
    allFakeCov = torch.cov(allFakeActivations)

    return fidScore(allRealMean,allFakeMean,allRealCov,allFakeCov)

if __name__ == '__main__':
    fakeDir = os.path.join(os.getcwd(),'..','test_fake')
    realDir = os.path.join(os.getcwd(),'..','test_real')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    realImages = getFileNames(realDir)
    fakeImages = getFileNames(fakeDir)
    getFIDScore(realImages,fakeImages,device)