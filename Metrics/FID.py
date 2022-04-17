# calculating Frechet Inception Distance for GAN
# FID score uses the inception v3 model classification pretrained on imagenet

import torch


# calculating fid given mean and covariance features of the generated set, and of the real set
def fidScore(realMeanFeature, generatedMeanFeature, realGeneratedCovariance, generatedRealCovariance):
    # ||mu_x-mu_y||^2
    meanDistance = pow(abs(generatedMeanFeature - realMeanFeature), 2)
    # Cov_x + Cov_y - 2 Cov_x Cov_y
    covarianceTerm = (realGeneratedCovariance + generatedRealCovariance) - 2 * (
                realGeneratedCovariance * generatedRealCovariance)
    return meanDistance - torch.trace(covarianceTerm)
