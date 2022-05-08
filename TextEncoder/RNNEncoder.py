# class for the rnn bi-lstm text encoder
# this is pretrained with the image encoder on CUB and COCO before use with GAN for CUB and COCO respectively
# # so that text representations are aligned with image representations

import torch

class bilstmEncoder(torch.nn.Module):

    def __init__(self, numTokens, embeddingSize, dropoutProb, numHiddenFeatures, stepSize):
        super().__init__()
        # stepsize or length of words in a caption -> 18 words
        self.stepSize = stepSize
        # size of vocabulary (dictionary)
        self.numTokens = numTokens
        # size of each embedding of input
        self.embeddingSize = embeddingSize
        # dropout probability
        self.dropoutProb = dropoutProb
        # feature space of hidden states
        # # dividing by 2 because we are bidirectional, meaning we end with with total numHiddenFeatures after
        # # 2 directional passes
        self.numHiddenFeatures = numHiddenFeatures // 2
        # number of recurrent layers
        # below parameter not needed, default is already 1
        #self.numLayers = numLayers

        # defining an embedding
        self.embedding = torch.nn.Embedding(self.numTokens,self.embeddingSize)
        # dropout layer
        self.dropoutLayer = torch.nn.Dropout(self.dropoutProb)
        # defining the lstm
        # using default # of layers =1, i.e we are NOT using a stacked lstm here
        self.LSTM = torch.nn.LSTM(self.embeddingSize, self.numHiddenFeatures, batch_first = True,
                                  dropout=self.dropoutProb, bidirectional=True)

        # initializing weights
        self.initializeWeights()

    def initializeWeights(self):
        # paper initializes embedding matrix weights uniformly from -0.1 to 0.1
        # rnn parameters are already initialized by pytorch
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, captionLengths, hidden):
        # embedding the input captions
        embedded = self.dropoutLayer(self.embedding(captions))

        # packing the caption lengths to use with the lstm
        captionLengthData = captionLengths.data.tolist()
        packedEmbeddings = torch.nn.utils.rnn.pack_padded_sequence(embedded,captionLengthData, batch_first = True,
                                                                   enforce_sorted=False)

        # getting output and final hidden states based on last hidden state
        output, hiddens = self.LSTM(packedEmbeddings,hidden)

        # output is returned as a packed sequence, so we need to unpack it -> give padding to 18 elements
        output, lengths = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True,total_length=18)

        # sentence encoding will be the final hidden states of directions of the lstm
        # we transpose to change element orientation from (2,batchSize,hiddenDimension) to (batchsize,2,hiddenDimension)
        # need to reshape in memory with contiguous so that we can get a view as a sentence encoding
        # todo: check if we need contiguous in line below
        sentenceEncoding = hiddens[0].transpose(0,1)
        # flattening along 2nd and 3rd dimension to get sentence encodings
        sentenceEncoding = torch.flatten(sentenceEncoding,start_dim=1)

        # getting individual word encodings
        # output has shape (batchSize, length, 2*hiddenDimension)
        # we want (batchsize, 2*hiddenDimension, length) so that matching loss is computed using columns as word
        # # representations
        wordEncoding = output.transpose(1,2)

        return wordEncoding, sentenceEncoding

    # need to zero out current hidden states and cell states for the next step so that
    # # one caption in the dataset does not depend on the previous one that was run through the lstm
    def initHiddenStates(self, batchSize):
        # gets a reference to some parameter (this is just for getting data type and device for the parameter easily)
        weightDataType = next(self.parameters()).data

        finalHiddenStateParams = weightDataType.new(2,batchSize,self.numHiddenFeatures).zero_()
        cellStateParams = weightDataType.new(2,batchSize,self.numHiddenFeatures).zero_()

        return finalHiddenStateParams, cellStateParams
