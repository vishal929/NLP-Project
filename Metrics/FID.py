# calculating Frechet Inception Distance for GAN
# FID score uses the inception v3 model classification pretrained on imagenet

import torch
import torchvision
import numpy as np
import scipy.linalg.sqrtm as sqrtm


def getInceptionV3():
    inceptionPoolModel = torch.nn.Sequential(*list(
        torchvision.models.inception_v3(pretrained=True, transform_input=False)
            .children()
    )[:-1])

    # eval for no dropout
    inceptionPoolModel.eval()

    return inceptionPoolModel

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

    realImages = torch.nn.functional.interpolate(realImages, size=(299, 299), mode='bilinear', align_corners=False)
    realActivations = inceptionPoolModel(realImages)
    fakeActivations = inceptionPoolModel(generatedImages)

    # returning both realActivations and fakeActivations
    return realActivations, fakeActivations

def getFIDScore(realImages,generatedImages):
    # split realImages and generatedImages into batches of splitSize
    splitSize= 100
    #(splits into equal 100 splits of the original # of examples)
    # i.e each index here represents (100,255,255,3) where we have 100 examples of 255x255 images
    realImages = torch.split(realImages,split_size_or_sections=splitSize)
    generatedImages = torch.split(generatedImages, split_size_or_sections=splitSize)

    # getting inception v3 model for activations
    inceptionPoolModel = getInceptionV3()

    allRealActivations = []
    allFakeActivations = []

    for i in range(len(realImages)):
        realActivations, fakeActivations = getActivations(realImages[i],generatedImages[i],inceptionPoolModel)
        # shape should be (numExamples,2048) where 2048 is the feature vector length
        allRealActivations.append(realActivations)
        allFakeActivations.append(allFakeActivations)
    # stacking real and fake activations into the overall matrix where first
    allRealActivations = torch.cat(allRealActivations,dim=0)
    allFakeActivations = torch.cat(allFakeActivations,dim=0)

    # running statistics on the activations to get mean and covariance required for FID score
    allRealMean = torch.mean(allRealActivations,dim=0)
    allFakeMean = torch.mean(allFakeActivations, dim=0)
    allRealCov = torch.cov(allRealActivations)
    allFakeCov = torch.cov(allFakeActivations)

    return fidScore(allRealMean,allFakeMean,allRealCov,allFakeCov)

