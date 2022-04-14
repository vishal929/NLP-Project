# our implementation of a block to replace the SSACN block
import torch


# pytorch module for our class

class AttentionalBlock(torch.nn.Module):
    # predict_mask means that we apply convolutions with relu to the obtained mask to pass to the next
    # # attentional block
    def __init__(self,in_channel, out_channel, predict_mask=True):
        super.__init__()

    # x is the image channel data
    # y is the text encoding
    def forward(self,x,y):
        # self attention between channels here to get a channel representation for the mask
        # then "filling" by using the obtained mask and the outputs of the text encoding perceptron
        pass
