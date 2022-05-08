# doing 1x1 conv to get query, key, and value for self attention
# in the case of cross attention, doing 1x1 conv to just get query

import torch

class ChannelAttention(torch.nn.Module):
    def __init__(self, inputChannels, inputWidth, inputHeight, isSelfAttention, channelScalar = 8, numHeads=1, wordDim = 256):
        super().__init__()
        self.isSelfAttention = isSelfAttention
        channelDim = inputChannels
        self.channelScaler = channelScalar
        # h x w
        self.embedDim = inputWidth * inputHeight
        self.wordDim = wordDim
        if isSelfAttention:
            # 1x1 conv to get query, key of shape (batch,C/scalar,HxW), and value respectively of (batch,C,HxW)
            self.queryConv = torch.nn.Conv2d(channelDim, channelDim//channelScalar, 1)
            self.keyConv = torch.nn.Conv2d(channelDim, channelDim // channelScalar, 1)
            self.valueConv = torch.nn.Conv2d(channelDim, channelDim, 1)
            # self.attention = torch.nn.MultiHeadedAttention(self.embedDim,num_heads = numHeads)
        else:
            # 1x1 conv to get query of shape (batch, C/scalar, HxW)
            self.queryConv = torch.nn.Conv2d(channelDim, channelDim//channelScalar, 1)
            #self.keyConv = torch.nn.Conv2d(channelDim, channelDim // channelScalar, 1)
            #self.valueConv = torch.nn.Conv2d(channelDim, channelDim, 1)
            # linear layer to project words to the embed dim
            self.wordLinear = torch.nn.Linear(self.wordDim,self.embedDim)
            self.keyLinear = torch.nn.Linear(self.wordDim,self.embedDim)
            # self.attention = torch.nn.MultiHeadedAttention(self.embedDim,num_heads = numHeads)
            # cross attention layer

        # need a gamma for combination weight with residual, like in ssacn block
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        # need a softmax operation along the last dimension (pixel dimension)
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, x, y=None, padMask = None):
        batch,numChannels,width,height = x.shape
        # transpose for matrix multiplication
        #print('xshape:' + str(x.shape))
        query = self.queryConv(x).flatten(start_dim=-2,end_dim=-1)
        #print(query.shape)
        if self.isSelfAttention:
            key = self.keyConv(x).flatten(start_dim=-2,end_dim=-1)
            value = self.valueConv(x).flatten(start_dim=-2,end_dim=-1)
            #value = self.selfAttention(query, key, value)
        else:
            pad = torch.zeros((batch,256,numChannels-18)).cuda()
            # expanding y to be of numChannels
            y = torch.cat((y,pad),dim=-1)
            #y = y.expand(2,256,numChannels)
            # projecting words to embed dim
            key = self.wordLinear(y[:,:,:numChannels//self.channelScaler].transpose(-2,-1))
            y = self.wordLinear(y.transpose(-2,-1))
            #print(key.shape)
            value = y
            #print(value.shape)
            #value = self.crossAttention(query,y,y,key_padding_mask = padMask)

        # attention operation
        query = query.transpose(-1,-2)
        query = torch.bmm(query, key)
        query = self.softmax(query)
        value = torch.bmm(value, query.transpose(-1, -2)).view(batch,numChannels,width,height)

        # multi headed attention operation
        '''
        if self.isSelfAttention:
            value = self.attention(query,key,value)
        else:
            value = self.attention(query,y,y,key_padding_mask=padMask)
        '''

        # parameter weighting operation
        x = (self.gamma*value) + x

        return x


# test channel attention
'''
testInput = torch.randn((1,64,100,100)).to('cuda')

attnTest = ChannelAttention(64,100,100,True).to('cuda')

res = attnTest(testInput)
'''