import torch
from FCBlock import FCBlock
from SSACN import SSACNBlock
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from AttentionalBlock import AttentionalBlock

class AttentionalGenerator(torch.nn.Module):

  def __init__(self, n_features=64 , attentionFeatures = 768 ,patchLength = 32,ssacn_block_scales = None):
    super(AttentionalGenerator, self).__init__()

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
    # Initialize last conv layer to turn generated img into shape: (batch_size, 3, 256, 256)
    self.final_conv = torch.nn.Conv2d(n_features, 3, 3, 1, 1)

    # splitting our (batch_size,3,256,256) to patches of (32x32x3) where 3 is for channel and 32 is height and width
    # then sending each (32x32x3) patch to some embedding through a linear layer

    patchDim = patchLength*patchLength*3
    numPatches = (256 // patchLength) * (256 // patchLength)

    self.transformPatches = torch.nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patchLength, p2 = patchLength),
            torch.nn.Linear(patchDim, attentionFeatures)
    )


    # need positional embedding for patches
    self.imagePositions = torch.nn.Parameter(torch.randn(1,numPatches,attentionFeatures))

    # need positional embedding for words
    # there is max sequence length of 18 and word embeddings have dim 256
    self.wordPositions = torch.nn.Parameter(torch.randn(18,256))
    # 3 self attention layers
    self.selfAttentionLayers = torch.nn.Sequential(
            AttentionalBlock(embedDim=attentionFeatures,isSelfAttention=True,numPatches=numPatches),
            AttentionalBlock(embedDim=attentionFeatures, isSelfAttention=True, numPatches=numPatches),
            AttentionalBlock(embedDim=attentionFeatures, isSelfAttention=True, numPatches=numPatches)
    )

    # 3 cross attention layers
    self.crossAttentionLayers = torch.nn.Sequential(
          AttentionalBlock(embedDim=attentionFeatures, isSelfAttention=False, numPatches=numPatches),
          AttentionalBlock(embedDim=attentionFeatures, isSelfAttention=False, numPatches=numPatches),
          AttentionalBlock(embedDim=attentionFeatures, isSelfAttention=False, numPatches=numPatches)
    )

    # sending our patch representations back to the patch dimension and reconstructing them into original shape
    self.reconstructPatches = torch.nn.Sequential(torch.nn.Linear(attentionFeatures,patchDim),
                                                  Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patchLength,
                                                            p2=patchLength)
                                                  )

    # final tanh to get values between -1 and 1
    self.tanh = torch.nn.Tanh()


  def forward(self, z, s, w, wMask):
    """
        Forward pass of generator to generate fake images
        Args:
          z (torch.Tensor): Standard gaussian noise of shape: (batch_size, z_shape[1])
          s (torch.Tensor,requires_grad=T): Sentence-embeddings used by SSACNBlock
          w (torch.Tensor,requires_grad=T): word-embeddings used by transformer blocks
          wMask (torch.Tensor,requires_grad=F): simply a mask to ignore padding tokens during attention operation
        Return:
          x (torch.Tensor,requires_grad=T): Final generated image of shape: (batch_size, 3, 256,256)
        """
    x = self.dense(z)  # Shape: (batch_size, 8*n_features, 4, 4)
    for i, ssacn_block in enumerate(self.SSACN_blocks):
      scale = 2
      if (i == 0):
        scale = 1
      x = ssacn_block(x, s, scale=scale)
    x = self.BN(x)
    x = self.leakyR(x)
    x = self.final_conv(x)

    # attention stuff, inspired by vision transformer

    # getting patches and sending them to attentionalBlock dimension
    x = self.transformPatches(x)

    # adding positional embedding to patches
    x = x + self.imagePositions

    # adding positional embeddings to words
    w = w + self.wordPositions

    # run self attention layers
    x = self.selfAttentionLayers(x)

    # run cross attention layers between image and word representations
    x = self.crossAttentionLayers(x,w,wMask)

    # reconstruct original dimension
    x = self.reconstructPatches(x)

    x = self.tanh(x)

    return x

