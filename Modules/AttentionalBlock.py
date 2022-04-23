# our implementation of a block to replace the SSACN block
import torch


# pytorch module for our class

# will this work?
# we start with noise of shape (3,255,255)
# then we split each channel into 16x16 chunks and put them together into one tensor
# perform self attention between the chunks in each channel to get some representation for each chunk
# # (ideally, this provides a representation of patches that still "need to be filled out")
# perform cross attention between chunks of all channels and 18 word embeddings
# # (this would provide the "stuffing" for patches that still "need to be filled out")
class AttentionalBlock(torch.nn.Module):

    # # attentional block
    def __init__(self):
        super.__init__()
        '''
        torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False,
                                    add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None,
                                    dtype=None)
        '''
        # self attention layer will send 16x16 patches to a new representation
        # resulting embed_dim=256 is split across 4 heads and then concatenated to send each patch to a vector
        # # of size 256
        self.selfAttentionLayer = torch.nn.MultiheadAttention(embed_dim=256,num_heads=4)


        self.crossAttentionLayer = torch.nn.MultiheadAttention()


    # x is the image channel data
    # y is the word embeddings
    def forward(self,x,y):
        # grabbing 16x16 patches from input x
        size = 16
        stride =16
        patches = x.unfold(1,size,stride).unfold(2,size,stride).unfold(3,size,stride)

        # flattening 16x16 patches


        # self attention between patches of image to determine "which need to still be filled out"
        patchRepresentations = self.selfAttentionLayer(patches,patches,patches)

        pass
