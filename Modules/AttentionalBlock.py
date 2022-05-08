# our implementation of a block to replace the SSACN block
import torch


# pytorch module for our class
# attentional module expecting patch representations as input
class AttentionalBlock(torch.nn.Module):

    # # attentional block
    # patchSizeLength **2 needs to be divisible by numHeads
    def __init__(self, numHeads = 8, numPatches=1, embedDim=256, textDim=256, linearScaling = 4, isSelfAttention=True ,dropout=0.1):
        super().__init__()
        '''
        torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False,
                                    add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None,
                                    dtype=None)
        '''
        # embedding dim for the attention layer
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.isSelfAttention = isSelfAttention
        self.dropout = dropout
        # intermediate scaling for mlp part of self attention
        self.linearScaling = linearScaling
        # self attention layer will send 16x16 patches to a new representation
        # resulting embed_dim=256 is split across 4 heads and then concatenated to send each patch to a vector
        # # of size 256

        # define an attention layer
        if isSelfAttention:
            self.attention = torch.nn.MultiheadAttention(embed_dim=self.embedDim, num_heads=self.numHeads, batch_first=True, dropout=self.dropout)
        else:
            # key and value will be of a different dimension possibly (256 for word embeddings)
            self.attention = torch.nn.MultiheadAttention(embed_dim=self.embedDim, num_heads=self.numHeads, batch_first=True, dropout=self.dropout, kdim=textDim,vdim=textDim)

        # feedforward part after attention
        self.feedforward = torch.nn.Sequential(torch.nn.Linear(self.embedDim,self.linearScaling*self.embedDim),
                                               torch.nn.GELU(),
                                               torch.nn.Dropout(p=self.dropout),
                                               torch.nn.Linear(self.linearScaling*self.embedDim,self.embedDim))
        # need 2 layernorms
        self.norm1 = torch.nn.LayerNorm((numPatches,self.embedDim))
        self.norm2 = torch.nn.LayerNorm((numPatches,self.embedDim))

    # x are the patch embeddings
    # y are word features for cross attention
    # may need a mask if there are padding tokens
    def forward(self,x,y=None, mask=None):
        # keep a shortcut
        shortcut = x
        if self.isSelfAttention:
            # we just do self attention operations
            # run through multi headed attention layer
            x, weights = self.attention(x,x,x)

        else:
            # doing cross attention operations
            # masking padding words beyond sequence length
            x, weights = self.attention(x,y,y, key_padding_mask = mask)

        # add shortcut and layernorm
        x = self.norm1(shortcut + x)
        # keep a shortcut
        shortcut = x
        # feedforward
        x = self.feedforward(x)
        # add shortcut and layernorm
        x = self.norm2(shortcut + x)

        # just return the new representations for the patches -> we can reshape in the generator to the output
        return x