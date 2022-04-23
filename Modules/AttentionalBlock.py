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
        self.selfAttentionLayer = torch.nn.MultiheadAttention(embed_dim=256,num_heads=4,batch_first=True)


        self.crossAttentionLayer = torch.nn.MultiheadAttention(embed_dim=256,num_heads=4,batch_first=True)


    # x is the image channel data
    # y are the word embeddings
    def forward(self,x,y):
        # grabbing 16x16 patches from input x
        size = 16
        stride =16
        patches = x.unfold(2,size,stride).unfold(3,size,stride)

        #(batchSize,3*numPatches)

        # flattening 16x16 patches
        patches = patches.flatten(start_dim=-2,end_dim=-1)
        # flattening middle representations to get just a sequence of raw patches
        patches = patches.flatten(start_dim=1,end_dim=-2)


        # self attention between patches of image to determine "which need to still be filled out"
        patches = self.selfAttentionLayer(patches,patches,patches)

        # now we have representations of patches based on self attention
        # lets " fill" these patches using cross attention with word embeddings
        # keep in mind words embeddings have dim 256, and our patches have dim 256, so we can do cross attention here

        # cross attention between our patches and word embeddings
        patches = self.crossAttentionLayer(patches,y,y)

        # reshaping patches back to (batch,3,256,256)
        patches = patches.reshape(x.shape)

        # send along output to the next block for further "selective" filling
        return patches

