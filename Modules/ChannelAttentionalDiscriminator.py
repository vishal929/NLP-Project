# generator implementing channel attention ideas from the following paper:
# https://arxiv.org/pdf/1805.08318.pdf

# implementing attentional ideas to our disccriminator in order to better improve performance
import torch
from Modules.ChannelAttentionalBlock import ChannelAttention
from Modules.DownBlock import DownBlock

class ChannelAttentionalDiscriminator(torch.nn.Module):
    def __init__(self, in_dim, n_c=64, downblock_scales=None, text_shape=(1, 256)):
        """
        Args:
          in_dim (tuple): Shape of input images of format (batch_size, height, width, channels)
          n_c (int) : Number of channels for output of first conv layer; all other downblock channel sizes are scaled to this.
          downblock_scales (Optional list): List of how much to scale the number of out-going features for each downblock
          text_shape (tuple): Shape of each embedded text (batch_size, embedded_text_length)
        """
        super(ChannelAttentionalDiscriminator, self).__init__()
        self.text_shape = text_shape
        # First 3x3 convolution with stride 1 -> (batch_size, height, width, n_c)
        self.conv1 = torch.nn.Conv2d(3, n_c, 3, 1, 1)

        # If the amount to scale the outgoing number of features is not provided, use default
        if (downblock_scales is None):
            downblock_scales = [1, 2, 4, 8, 16, 16, 16, 2, 1]

        # Initialize downblocks
        self.downblocks_l = torch.nn.ModuleList()
        prev_scale = 1
        for i in range(len(downblock_scales) - 3):
            curr_scale = downblock_scales[i]
            next_scale = downblock_scales[i + 1]
            self.downblocks_l.append(DownBlock(n_c * curr_scale, n_c * next_scale))
            prev_scale = curr_scale

        # Initialize for Decide section
        n_rep1 = downblock_scales[-2]
        n_rep2 = downblock_scales[-1]
        self.conv_rep1 = torch.nn.Conv2d(n_c * prev_scale + (text_shape[-1]), n_c * n_rep1, 3, 1, 1, bias=False)
        self.leakyR = torch.nn.LeakyReLU(0.2, inplace=True)

        # attention operations BEFORE the final convolution
        # the shape at this stage is (batch,128,4,4)
        self.selfAttention = ChannelAttention(128,4,4,True)
        self.crossAttention = ChannelAttention(128,4,4,False)

        self.conv_rep2 = torch.nn.Conv2d(n_c * n_rep1, n_rep2, 4, 1, 0, bias=False)

    def forward(self, x, s, w, padMask):
        """
        Gives the discriminator decision (scores/logits) for a pair of given batch of images and corresponding text

        Args:
          x (torch.Tensor): Input batch of images with shape (batch_size, height, width, channels)
          s (torch.Tensor): Input text caption with shape (batch_size, dim_text)
          w (torch.Tensor): Input word embeddings with shape (batch_size,18,dim_text)
          padMask : Input padding mask for valid word indices (i.e if caption length is 16, then indices 17 and 18 are padding)
        Return:
          x (torch.Tensor): Output decision or logits
        """
        x = self.conv1(x)
        for i, downblock_layer in enumerate(self.downblocks_l):
            x = downblock_layer(x)

        # Process text input and concatenate to image features along height

        # image actually (batch_size, 1024,4,4)
        # sentence dimension is (batch_size, 256)
        # print(x.shape)
        # print(s.shape)
        # Replicate text across channels, maintaining same batch_size
        # unsqueezing sentence to add dimensions after features -> (batchsize,256,1,1)
        s = s.unsqueeze(2).unsqueeze(3).repeat(1, 1, 4, 4)
        # print(s.shape)
        # Concatenate along the channels dimension (last 2 dims are h,w so -3 dim is channel)
        x = torch.cat((x, s), dim=-3)
        # print("Concat Ours: ", x[0,:,0,:])

        # Run rest of conv layers on concatenated output
        x = self.conv_rep1(x)
        x = self.leakyR(x)

        # attentional operations
        x = self.selfAttention(x)
        # NOTE IF CROSS ATTENTION IN DISCRIMINATOR IS FAILING -> TRY CONCATENATING WORDS TO THE CHANNELS
        # # LIKE WE DO WITH THE SENTENCE EMBEDDING
        x = self.crossAttention(x,w,padMask)

        x = self.conv_rep2(x)
        # print("Our shape: ", x.shape)
        # print("Joint conv Ours: ", x[0,:,0,:])

        return x