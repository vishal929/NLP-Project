import torch
from SSACN import *
from FCBlock import FCBlock

class Generator(torch.nn.Module):
  """
  Implementation of the SSA-GAN: https://arxiv.org/pdf/2104.00567.pdf
  """
  def __init__(self, n_features=64 , z_shape=(1, 100),ssacn_block_scales = None):
    super(Generator, self).__init__()

    self.dense = FCBlock() #Linear with shape (z_shape[1], n_features*128)
    #NOTE: The above dense layer auto-reshapes output into 
    #shape: (batch_size, 8*n_features, 4, 4)

    #Use default scales for SSACNBlocks if none provided
    if(ssacn_block_scales is None):
      ssacn_block_scales = [(8,8), (8,8), (8,8), (8,8), (8,4), (4,2), (2,1)]

    self.SSACN_blocks = torch.nn.ModuleList()
    for i, (in_scale, out_scale) in enumerate(ssacn_block_scales):
      curr_block = SSACNBlock(in_scale*n_features, out_scale*n_features)
      self.SSACN_blocks.append(curr_block)

    self.BN = torch.nn.BatchNorm2d(n_features)
    self.leakyR = torch.nn.LeakyReLU(0.2, inplace=True)
    #Initialize last conv layer to turn generated img into shape: (batch_size, 3, 256, 256)
    self.final_conv = torch.nn.Conv2d(n_features, 3, 3, 1, 1)
    self.tanh = torch.nn.Tanh()

  def forward(self, z, s):
    """
    Forward pass of generator to generate fake images
    Args:
      z (torch.Tensor): Standard gaussian noise of shape: (batch_size, z_shape[1])
      s (torch.Tensor,requires_grad=T): Sentence-embeddings used by SSACNBlock 
    Return:
      x (torch.Tensor,requires_grad=T): Final generated image of shape: (batch_size, 3, 256,256)
    """
    x = self.dense(z) #Shape: (batch_size, 8*n_features, 4, 4)
    for i, ssacn_block in enumerate(self.SSACN_blocks):
      scale = 2
      if(i==0):
        scale = 1
      x = ssacn_block(x, s, scale = scale )
    x = self.BN(x)
    x = self.leakyR(x)
    x = self.final_conv(x)
    x = self.tanh(x)
    return x

