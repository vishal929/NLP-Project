# implementation of downblock as a module from the paper
# this is used by the discriminator to downsample generator output into features
# specifically consists of downsampling operation + residual block
import torch

class DownBlock(torch.nn.Module):
    def __init__(self, numFeaturesIncoming, numFeaturesOutgoing):
        super().__init__()
        # transformations applied to residual before adding to output
        self.residualConv = torch.nn.Sequential(
            torch.nn.Conv2d(numFeaturesIncoming,numFeaturesOutgoing,4,2,1,bias=False),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(numFeaturesOutgoing, numFeaturesOutgoing, 3, 1, 1, bias =False),
            torch.nn.LeakyReLU(0.2)
        )

        # scaler parameter to weight the residual
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        # 1x1 convolution to project input features to output feature dimension, in case they do not match
        self.dimConv = torch.nn.Conv2d(numFeaturesIncoming,numFeaturesOutgoing,1)

        # boolean indicating whether we need to apply the above convolution or not
        self.needDimensionTransform = numFeaturesOutgoing != numFeaturesIncoming

        # downsampling operation
        self.downsample = torch.nn.AvgPool2d(2)

    def forward(self,x):
        downsampled = None
        if self.needDimensionTransform:
            downsampled = self.downsample(self.dimConv(x))
        else:
            downsampled = self.downsample(x)
        return downsampled + (self.gamma * self.residualConv(x))



