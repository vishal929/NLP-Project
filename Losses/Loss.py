import torch
# we have normalized adversarial loss and weighted damsm loss components

# sentenceEmbedding is the sentence feature vector
# wordMatrixEmbedding is a matrix where the i-th column is the feature vector of the i-th word
# localImagePerceptionOutput is result of feeding embedding of subregion of image through perception layers
# globalImagePerceptionOutput is same as above, but with the global feature from last average pooling layer of
# # Inception v3
# local image subregions are obtained by reshaping feature maps in shape of 768 from output of
# # "mixed 6e layer" of Inception-v3
# gammaOne is an attention factor
# gammaTwo is another attention factor
# gammaThree is a smoothing factor
def damsmLoss(sentenceEmbedding, wordMatrixEmbedding, localImagePerceptronOutput, globalImagePerceptronOutput,
              gammaOne, gammaTwo, gammaThree):
    # get similarity matrix of words and subregions
    # s = e^T v
    s = torch.matmul(wordMatrixEmbedding.transpose() ,localImagePerceptronOutput)

    # get global

    #  normalize similarity matrix
    s = torch.exp(s)
    s = s / torch.sum(s)

    # get region context vectors as a matrix
    # this is the result of an alpha matrix elementwise multiplication with the word vectors
    # then we sum everything together

    alphaMatrix = torch.exp(gammaOne * s)
    # summing over each column
    alphaMatrix = alphaMatrix / torch.sum(alphaMatrix,1)

    # multiplying each a_j with v_j by matrix multiplication to get a sub-region word context matrix by broadcasting
    contextMatrix = torch.matmul(alphaMatrix,localImagePerceptronOutput.transpose())

    # grabbing word relevance to image with cosine similarity between each context vector and the i-th word embedding
    relevanceMatrix = (torch.matmul(contextMatrix,wordMatrixEmbedding))
    # element wise division by the norms of each context vector, and for each word vector
    relevanceMatrix /= torch.matmul(contextMatrix,contextMatrix.transpose())
    relevanceMatrix /= torch.matmul(wordMatrixEmbedding.transpose(), wordMatrixEmbedding)









def adversarialLoss():
    pass

def totalLoss(damsmWeight, adversarialLoss, damsmLoss):
    return adversarialLoss + (damsmWeight * damsmLoss)