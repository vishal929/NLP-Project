import torch
import sys
sys.path.append('../Modules')
from Discriminator import Discriminator
# we have normalized adversarial loss and weighted damsm loss components
# LOOK OVER SCALING HERE , NOT SURE ABOUT THAT

# this function calculates a "matching score" between a given sentence and an image
# a locally derived score and a global score is returned
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
def calculateAttentionMatchingScore(sentenceEmbedding, wordMatrixEmbedding, localImagePerceptronOutput, globalImagePerceptronOutput,
              gammaOne=5, gammaTwo=5):
    # get similarity matrix of words and subregions
    # s = e^T v
    s = torch.matmul(wordMatrixEmbedding.transpose() ,localImagePerceptronOutput)

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
    # recall that sub regions are columns in the localImagePerceptronOutput
    # so, we take the transpose before multiplication
    contextMatrix = torch.matmul(alphaMatrix,localImagePerceptronOutput.transpose())

    # grabbing word relevance to image with cosine similarity between each context vector and the i-th word embedding
    # its a column vector of dot products basically
    relevanceVector = (torch.matmul(contextMatrix,wordMatrixEmbedding))
    # element wise division by the norms of each context vector, and for each word vector
    # basically just elementwise square, then sum along column, and then take sqrt
    relevanceVector /= torch.sqrt(torch.sum(torch.pow(contextMatrix,2)))
    # need to sum along row here since the i-th feature vector of the i-th word is the i-th column
    relevanceVector /= torch.sqrt(torch.sum(torch.pow(wordMatrixEmbedding,2),1))

    # getting entire attention driven matching score between
    # entire image (Q) and whole text description (D)
    RQD = torch.log(
        torch.sum(
            torch.exp(gammaTwo * relevanceVector)
        )
    ) ** (1/gammaTwo)

    # getting a global score between the image and sentence without local components
    globalAttentionScore = torch.dot(globalImagePerceptronOutput,sentenceEmbedding)/ \
                      (torch.norm(sentenceEmbedding) * torch.norm(globalImagePerceptronOutput))
    return RQD, globalAttentionScore

# we are given a score matrix where the i,jth element is the attention driven matching score
# between the i-th image and the j-th sentence
# we are also given a matrix for the same thing as above, but with global consideration only
# since we have both M images and sentences in a batch, these will be MxM matrices
# gammaThree is a smoothing factor
def calculateDAMSMLoss(localAttentionDrivenScoreMatrix, globalAttentionDrivenScoreMatrix
                       , gammaThree=10):
    # calculating P(D_i|Q_i) matrix -> matching description to image
    powered = torch.exp(gammaThree * localAttentionDrivenScoreMatrix)
    # need to sum along each column and scale the row by the entire row sum
    pDQ =  powered / torch.sum(powered,1)
    # calculating P(Q_i | D_i) matrix -> matching image to description
    # need to sum along rows here, and scale the column by the entire column sum
    pQD = powered/torch.sum(powered,0)

    # losses for local parts
    # just a negative log probability of an image being matched with its description
    localLossOne =  - torch.trace(torch.log(pDQ))
    # vice versa log of a description being matched with an image
    localLossTwo = -torch.trace(torch.log(pQD))

    # doing the same as above, but for the global components
    globalPowered = torch.exp(gammaThree * globalAttentionDrivenScoreMatrix)
    globalPDQ = globalPowered / torch.sum(globalPowered,1)
    globalPQD = globalPowered/torch.sum(globalPowered,0)

    # losses for global parts
    # same as above basically
    globalLossOne = -torch.trace(torch.log(globalPDQ))
    globalLossTwo = -torch.trace(torch.log(globalPQD))

    # adding up all the losses to form damsm
    return localLossOne + localLossTwo + globalLossOne + globalLossTwo

def adv_D(D, x, x_hat, s, s_hat, lambda_MA, p):
  """
  Computes the advesarial loss for the Discriminator
  Args:
    D (Discriminator object): Discriminator
    x (torch.Tensor, requires_grad=T): True image with shape (batch_size, height, width, channels)
    x_hat (torch.Tensor): Fake generated image with same shape as x
    s (torch.Tensor, requires_grad=T): True text description with shape (batch_size, T_s)
    s_hat (torch.Tensor): Mismatched text description with same shape as s
    lambda_MA (float) : Weight of MA-GP loss
    p (float) : p hyperparameter
  Return:
    Loss_adv_D (float / torch.Tensor): Computed adversarial loss for Discriminator
    D_fake (torch.Tensor, requires_grad=T): Discriminator decisions for fake generated images with shape (batch_size, UNKNOWN_YET)
  """
  D_real = D(x,s)
  D_fake = D(x_hat,s)

  #Expected loss for failing to recognize real images (real loss)
  E_real = torch.maximum(0.0, 1.0 - D_real).mean()
  #Expected loss for getting fooled by generator
  E_fool = torch.maximum(0.0, 1.0 + D_fake).mean()
  #Expected loss for getting wrong text pairing
  E_mismatch = torch.maximum(0.0, 1.0 + D(x, s_hat)).mean()

  #Calculate gradients of decision w.r.t. both x and s
  D_real.backward(torch.ones_like(D_real))
  #Obtain calculated gradients and use them to get Eucl norms
  grad_D_real_x = x.grad
  grad_D_real_s = s.grad
  grad_norm_x = torch.dot(grad_D_real_x.T, grad_D_real_x)
  grad_norm_s = torch.dot(grad_D_real_s.T, grad_D_real_s)
  #Expected MA-GP loss
  E_MA = torch.pow(grad_norm_x + grad_norm_s, p).mean()

  #Final weighted adv loss for discriminator
  Loss_adv_D = E_real + ((E_fool + E_mismatch) / 2.0) + lambda_MA*(E_MA)
  return Loss_adv_D, D_fake

def adv_G(D_fake):
    """
    Computes the adversarial loss for the generator
    Args:
        D_fake (torch.Tensor, requires_grad=T): discriminator decisions for fake generated images
    Return:
        Loss_adv_G (float / torch.Tensor): Computed adversarial loss for generator
    """
    return -1.0*(D_fake).mean()

# total loss for the generator consists of an adversarial loss + a weighted damsm loss
def totalLoss(damsmWeight, adversarialLoss, damsmLoss):
    return adversarialLoss + (damsmWeight * damsmLoss)