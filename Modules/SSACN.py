# implementation of SSACN block module in pytorch

import torch
from torch._C import UnionType
import torch.nn as nn
import numpy as np
import torch.nn.functional as f

# TODO: Correct Input / output channels
# TODO: learnable shortcut if the input and output channels don't match
# TODO: look at learnable_sc
# Block that takes text feature vector and image feature maps from last SSACN block as input
# Outputs new image feature maps
# Each SSACN has an upsample block, a mask predictor, a SSCBN and a residual block
# batch size, height, width, channels
class SSACNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.learnable_sc = (in_channels != out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_predictor = MaskPredictor(self.out_channels)
        self.sscbn = SSCBN(self.out_channels, self.learnable_sc)
        self.diffChannels = in_channels != out_channels
        if self.diffChannels:
            self.channelScaler = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

# check scale_factor
    def forward(self, image_features, text_features, mask=None, scale = 2):
        if self.diffChannels:
            image_features = self.channelScaler(image_features)
        (b_s, c, h, w) = image_features.shape
        if scale != 1:
          upscaled_image_features = f.interpolate(image_features, mode='bilinear',
                                    size=(scale*h, scale*w),align_corners=True)
        else:
          upscaled_image_features = image_features
        #print(upscaled_image_features.shape)
        mask = self.mask_predictor(upscaled_image_features)
        output_image_features = self.sscbn(upscaled_image_features, text_features, mask)

        return output_image_features

# Mask predictor used in SSACN
class MaskPredictor(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(out_ch, 100, 3, 1, 1)
        self.batchnorm = nn.BatchNorm2d(100) 
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(100, 1, 1, 1, 0)

    def forward(self, upsampled_features):
        #print(upsampled_features.shape)
        mask = self.conv1(upsampled_features)
        #print(mask.shape)
        mask = self.batchnorm(mask)
        #print(mask.shape)
        mask = self.relu(mask)
        #print(mask.shape)
        mask = self.conv2(mask)
        #print(mask.shape)
        mask = torch.sigmoid(mask)

        return(mask)

# SSCBN used in SSACN (includes residual connection)
class SSCBN(nn.Module):
    def __init__(self, num_features_output, learnable_sc):
        super().__init__()

        # self.learnable_sc = learnable_sc
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

        self.fc_gamma = nn.Sequential(
            gamma_linear1,
            gamma_relu,
            gamma_linear2
        )
        self.fc_beta = nn.Sequential(
            beta_linear1,
            beta_relu,
            beta_linear2
        )

        # if self.learnable_sc:
        # self.conv_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def forward(self, upsampled_image_features, text_features, predicted_mask):
        bn_upsampled_image_features = self.batch_norm2d(upsampled_image_features)
        weight = self.fc_gamma(text_features)
        bias = self.fc_beta(text_features)

        #print("Predicted Mask shape: ", predicted_mask.shape)
        #print("BN_upsampled shape: ", bn_upsampled_image_features.shape)
        #print("Weight shape: ", weight.shape)
        #print("Bias shape: ",bias.shape)
        w_dim = weight.size()
        x_dim = bn_upsampled_image_features.size()
        new_view = (w_dim[0], w_dim[1], 1, 1)
        new_rep = (1, 1, x_dim[-2],x_dim[-1] )

        weight = weight.view(new_view).repeat(new_rep)
        bias = bias.view(new_view).repeat(new_rep)
        sscbn = predicted_mask * (weight * bn_upsampled_image_features + bias) + bn_upsampled_image_features
        

        return sscbn