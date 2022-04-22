# testing our ssacn block vs theirs on random input
from collections import OrderedDict

import torch
from Modules.SSACN import SSACNBlock

class affine(torch.nn.Module):

    def __init__(self, num_features):
        super(affine, self).__init__()

        self.batch_norm2d = torch.nn.BatchNorm2d(num_features, affine=False)

        self.fc_gamma = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(256, 256)),
            ('relu1', torch.nn.ReLU(inplace=True)),
            ('linear2', torch.nn.Linear(256, num_features)),
        ]))
        self.fc_beta = torch.nn.Sequential(OrderedDict([
            ('linear1', torch.nn.Linear(256, 256)),
            ('relu1', torch.nn.ReLU(inplace=True)),
            ('linear2', torch.nn.Linear(256, num_features)),
        ]))
        self._initialize()

    def _initialize(self):
        torch.nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        torch.nn.init.zeros_(self.fc_gamma.linear2.bias.data)
        torch.nn.init.zeros_(self.fc_beta.linear2.weight.data)
        torch.nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None, fusion_mask=None):
        x = self.batch_norm2d(x)
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        weight = weight * fusion_mask + 1
        bias = bias * fusion_mask
        return weight * x + bias

# their defined ssacn block
class G_Block(torch.nn.Module):

    def __init__(self, in_ch, out_ch, num_w=256, predict_mask=True):
        super(G_Block, self).__init__()

        self.learnable_sc = in_ch != out_ch
        self.predict_mask = predict_mask
        self.c1 = torch.nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = torch.nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.affine0 = affine(in_ch)
        #self.affine1 = affine(in_ch)
        self.affine2 = affine(out_ch)
        #self.affine3 = affine(out_ch)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = torch.nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

        if self.predict_mask:
            self.conv_mask = torch.nn.Sequential(torch.nn.Conv2d(out_ch, 100, 3, 1, 1),
                                           torch.nn.BatchNorm2d(100),
                                           torch.nn.ReLU(),
                                           torch.nn.Conv2d(100, 1, 1, 1, 0))

    def forward(self, x, y=None, fusion_mask=None):
        out = self.shortcut(x) + self.gamma * self.residual(x, y, fusion_mask)

        if self.predict_mask:
            mask = self.conv_mask(out)
        else:
            mask = None

        return out, mask

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, y=None, fusion_mask=None):
        h = self.affine0(x, y, fusion_mask)
        h = torch.nn.ReLU(inplace=True)(h)
        h = self.c1(h)

        h = self.affine2(h, y, fusion_mask)
        h = torch.nn.ReLU(inplace=True)(h)
        return self.c2(h)




# setting device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generating some random input to use ssacn blocks
numFeatures = 64
inScale = 8
outScale = 8
# first block takes as input image of (512,4,4) where 512 channels 4 height and 4 width
imageDim = 4
randomImageInput = torch.rand((1,numFeatures*inScale,imageDim,imageDim),dtype=torch.float).to(device)
randomTextInput = torch.rand((1,256),dtype=torch.float).to(device)
inChannel = inScale * numFeatures
outChannel = outScale * numFeatures
torch.manual_seed(0)
ourSSACN = SSACNBlock(inChannel,outChannel).to(device)
torch.manual_seed(0)
conv_mask= torch.nn.Sequential(torch.nn.Conv2d(8 * numFeatures, 100, 3, 1, 1),
                                       torch.nn.BatchNorm2d(100),
                                       torch.nn.ReLU(),
                                       torch.nn.Conv2d(100, 1, 1, 1, 0)).to(device)
theirSSACN = G_Block(inChannel,outChannel).to(device)



ourResult = ourSSACN(randomImageInput,randomTextInput)

stage_mask = conv_mask(randomImageInput)
fusion_mask = torch.sigmoid(stage_mask)
theirResult = theirSSACN(randomImageInput, randomTextInput, fusion_mask)

print('our result:' + str(ourResult))
print('their result:' + str(theirResult))
