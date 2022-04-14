import torch
from DownBlock import DownBlock

class Discriminator(torch.nn.Module):
      """
  Implementation of DF-gan's discriminator: https://arxiv.org/pdf/2008.05865.pdf
  """

  def __init__(self, in_dim, downblock_scales=None, dim_text=256):
      """
      Args:
        in_dim (tuple): Shape of input images of format (batch_size, height, width, channels)
        downblock_scales (Optional list): List of how much to scale the number of out-going features for each downblock
        dim_text (int): Dimension of each text caption
      """
      super().__init__()

      n_c = in_dim[-1]
      #First 1x1 convolution with stride 1 -> (batch_size, height, width, n_c)
      self.conv1 = torch.nn.Conv2d(n_c, n_c, 1, 1)

      #If the amount to scale the outgoing number of features is not provided, use default
      if(downblock_scales is None):
        downblock_scales = [2,4,8, 16, 16, 16, 2, 2]
      
      #Initialize downblocks
      self.downblocks_l = []
      prev_scale = 1
      for i in range(len(downblock_scales)-2):
        curr_scale = downblock_scales[i]
        self.downblocks_l.append(DownBlock(n_c*prev_scale, n_c*curr_scale))
        prev_scale = curr_scale
      
      #Initialize for Decide section
      n_rep1 = downblock_scales[-2]
      n_rep2 = downblock_scales[-1]
      self.conv_rep1 = torch.nn.Conv2d(n_c*prev_scale, n_c*n_rep1, 3, 1,1, bias=False)
      self.conv_rep2 = torch.nn.Conv2d(n_c*n_rep1, n_c*n_rep2, 4, 1, bias=False)
      self.leakyR = torch.nn.LeakyReLU(0.2)

  def forward(self, x, s):
    """
    Gives the discriminator decision (scores/logits) for a pair of given batch of images and corresponding text

    Args:
      x (torch.Tensor): Input batch of images with shape (batch_size, height, width, channels)
      s (torch.Tensor): Input text caption with shape (batch_size, dim_text)
    Return:
      x (torch.Tensor): Output decision or logits
    """
    x = self.conv1(x)
    for downblock_layer in self.downblocks_l:
      x = downblock_layer(x)

    #Process text input and concatenate to image features along height
    s = None
    x = torch.cat(x, s, dim=1)

    #Run rest of conv layers on concatenated output
    x = self.conv_rep1(x)
    x = self.leakyR(x)
    x = self.conv_rep2(x)

    return x
