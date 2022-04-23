# calculating Frechet Inception Distance for GAN
# FID score uses the inception v3 model classification pretrained on imagenet

import torch
import torchvision


# calculating fid given mean and covariance features of the generated set, and of the real set
def fidScore(realMeanFeature, generatedMeanFeature, realGeneratedCovariance, generatedRealCovariance):
    # ||mu_x-mu_y||^2
    meanDistance = pow(abs(generatedMeanFeature - realMeanFeature), 2)
    # Cov_x + Cov_y - 2 Cov_x Cov_y
    covarianceTerm = (realGeneratedCovariance + generatedRealCovariance) - 2 * (
                realGeneratedCovariance * generatedRealCovariance)
    return meanDistance - torch.trace(covarianceTerm)


def getFIDScore(realImages, generatedImages):
    # loading the inceptionV3 Model but without the fully connected layer (only up to avg pool)
    inceptionPoolModel = torch.nn.Sequential(*list(
        torchvision.models.inception_v3(pretrained=True,transform_input=False)
        .children()
        )[:-1])

    # sending both real images and generated images through inception to get pool representations
    # may need to interpolate to 299x299 before sending images through inception

    realActivations = inceptionPoolModel(realImages)
    fakeActivations = inceptionPoolModel(generatedImages)

    # batchwise mean
    realMeanFeatures = realActivations.mean(dim=0)
    fakeMeanFeatures = fakeActivations.mean(dim=0)

    #

    pass