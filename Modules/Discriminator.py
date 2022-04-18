import torch
from DownBlock import DownBlock

class Discriminator(torch.nn.Module):
  """
  Implementation of DF-gan's discriminator: https://arxiv.org/pdf/2008.05865.pdf
  """
  def __init__(self, in_dim, n_c = 64, downblock_scales=None, text_shape=(1,256)):
      """
      Args:
        in_dim (tuple): Shape of input images of format (batch_size, height, width, channels)
        n_c (int) : Number of channels for output of first conv layer; all other downblock channel sizes are scaled to this.
        downblock_scales (Optional list): List of how much to scale the number of out-going features for each downblock
        text_shape (tuple): Shape of each embedded text (batch_size, embedded_text_length)
      """
      super(Discriminator,self).__init__()
      self.text_shape = text_shape
      #First 3x3 convolution with stride 1 -> (batch_size, height, width, n_c)
      self.conv1 = torch.nn.Conv2d(in_dim[-1], n_c, 3, 1)

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
      self.conv_rep1 = torch.nn.Conv2d(n_c*prev_scale + (text_shape[-1]), n_c*n_rep1, 3, 1,1, bias=False)
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
    #Input text is shape (batch_size, padded_text_length=18)
    #Image should now have shape (batch_size, 4, 4, n_c*16=1024)
    h,w = x.size()[1], x.size()[2]
    #Replicate text across channels, maintaining same batch_size
    s.view((-1, 1, 1, self.text_shape[-1])).repeat((1, h, w, 1))
    #Concatenate along the channels dimension
    x = torch.cat(x, s, dim=-1)

    #Run rest of conv layers on concatenated output
    x = self.conv_rep1(x)
    x = self.leakyR(x)
    x = self.conv_rep2(x)

    return x
