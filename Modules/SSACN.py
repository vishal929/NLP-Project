# implementation of SSACN block module in pytorch

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
# from sync_batchnorm import SynchronizedBatchNorm2d

# Block that takes text feature vector and image feature maps from last SSACN block as input
# Outputs new image feature maps
# Each SSACN has an upsample block, a mask predictor, a SSCBN and a residual block
class SSACNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, predict_mask=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = f.interpolate()
        self.mask_predictor = MaskPredictor(self.out_channels)
        self.sscbn = SSCBN(self.in_channels)

    def forward(self, image_features, text_features, mask=None):
        image_features = self.upsample(image_features)
        mask = self.mask_predictor(image_features)
        output_image_features = self.sscbn(image_features, text_features, mask)

        return output_image_features

# Mask predictor used in SSACN
class MaskPredictor(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2D(out_ch, 100, 3, 1, 1)
        # TODO: replace this batchnorm with SSCBN
        self.batchnorm = nn.BatchNorm2d(100) 
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(100, 1, 1, 1, 0)
        self.sigmoid = nn.sigmoid()

    def forward(self, upsampled_features):
        mask = self.conv1(upsampled_features)
        mask = self.batchnorm(mask)
        mask = self.relu(mask)
        mask = self.conv2(mask)
        mask = self.sigmoid(mask)

        return(mask)

# SSCBN used in SSACN (includes residual connection)
class SSCBN(nn.Module):
    def __init__(self, num_features_output):
        super().__init__()

        self.batch_norm2d = nn.BatchNorm2d(num_features_output, affine=False)

        gamma_linear1 = nn.Linear(256, 256)
        gamma_relu = nn.ReLU(inplace=True)
        gamma_linear2 = nn.Linear(256, num_features_output)

        nn.init.zeros_(gamma_linear2.weight.data)
        nn.init.zeros_(gamma_linear2.bias.data)
     
        beta_linear1 = nn.Linear(256, 256)
        beta_relu = nn.ReLU(inplace=True)
        beta_linear2 = nn.Linear(256, num_features_output)

        nn.init.zeros_(beta_linear2.weight.data)
        nn.init.zeros_(beta_linear2.bias.data)

        self.fc_gamma = nn.Sequential([
            gamma_linear1,
            gamma_relu,
            gamma_linear2
        ])
        self.fc_beta = nn.Sequential([
            beta_linear1,
            beta_relu,
            beta_linear2
        ])

    def forward(self, upsampled_image_features, text_features, predicted_mask):
        bn_upsampled_image_features = self.batch_norm2d(upsampled_image_features)
        weight = self.fc_gamma(text_features)
        bias = self.fc_beta(text_features)

        return predicted_mask * (weight * bn_upsampled_image_features + bias) + bn_upsampled_image_features