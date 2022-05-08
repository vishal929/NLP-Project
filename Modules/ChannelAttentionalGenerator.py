# generator implementing channel attention ideas from the following paper:
# https://arxiv.org/pdf/1805.08318.pdf

import torch
from Modules.SSACN import *
from Modules.FCBlock import FCBlock
from Modules.ChannelAttentionalBlock import ChannelAttention

class ChannelAttentionalGenerator(torch.nn.Module):
    def __init__(self, n_features=64, z_shape=(1, 100), ssacn_block_scales=None):
        super(ChannelAttentionalGenerator, self).__init__()

        self.dense = FCBlock()  # Linear with shape (z_shape[1], n_features*128)
        # NOTE: The above dense layer auto-reshapes output into
        # shape: (batch_size, 8*n_features, 4, 4)

        # Use default scales for SSACNBlocks if none provided
        if (ssacn_block_scales is None):
            ssacn_block_scales = [(8, 8), (8, 8), (8, 8), (8, 8), (8, 4), (4, 2), (2, 1)]

        self.SSACN_blocks = torch.nn.ModuleList()
        for i, (in_scale, out_scale) in enumerate(ssacn_block_scales):
            curr_block = SSACNBlock(in_scale * n_features, out_scale * n_features)
            self.SSACN_blocks.append(curr_block)

        self.BN = torch.nn.BatchNorm2d(n_features)
        self.leakyR = torch.nn.LeakyReLU(0.2, inplace=True)


        # attentional operations
        ''' THIS USES TOO MUCH MEMORY!'''
        '''
        self.block4SelfAttention = ChannelAttention(256,64,64,True)
        self.block4CrossAttention = ChannelAttention(256,64,64,False)
        '''

        self.block5SelfAttention = ChannelAttention(128,128,128,True)
        self.block5CrossAttention = ChannelAttention(128,128,128,False)


        # Initialize last conv layer to turn generated img into shape: (batch_size, 3, 256, 256)
        self.final_conv = torch.nn.Conv2d(n_features, 3, 3, 1, 1)
        self.tanh = torch.nn.Tanh()

    def forward(self, z, s, w, padMask):
        """
        Forward pass of generator to generate fake images
        Args:
          z (torch.Tensor): Standard gaussian noise of shape: (batch_size, z_shape[1])
          s (torch.Tensor,requires_grad=T): Sentence-embeddings used by SSACNBlock
          w (torch.Tensor): word embeddings generated by text encoder
          padMask: padding values on the word embeddings (max length of word embeddings is 18)
        Return:
          x (torch.Tensor,requires_grad=T): Final generated image of shape: (batch_size, 3, 256,256)
        """
        x = self.dense(z)  # Shape: (batch_size, 8*n_features, 4, 4)
        for i, ssacn_block in enumerate(self.SSACN_blocks):
            scale = 2
            if (i == 0):
                scale = 1
            x = ssacn_block(x, s, scale=scale)
            # we will run self and cross attention after ssacn blocks indices 4 and 5 (after block 6 is too much RAM usage! around 50 gb for 1 example)
            if (i== 4):
                #x = self.block4SelfAttention(x)
                #x = self.block4CrossAttention(x,w,padMask)
                pass
            elif (i==5):
                x = self.block5SelfAttention(x)
                x = self.block5CrossAttention(x,w,padMask)
        x = self.BN(x)
        x = self.leakyR(x)
        # shape at this point is (batchSize, 64,256,256)

        x = self.final_conv(x)
        x = self.tanh(x)
        return x
