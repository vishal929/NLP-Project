import torch
import sys
sys.path.append('../Modules')
import Modules.Discriminator as Discriminator


#from Discriminator import Discriminator
# we have normalized adversarial loss and weighted damsm loss components
# LOOK OVER SCALING HERE , NOT SURE ABOUT THAT

def calculateAttentionMatchingScoreBatchWrapper(sentenceBatch, wordMatrixBatch, localImageBatch, globalImageBatch,
                                                match_labels, sequenceLengths,
                                                gammaOne=5, gammaTwo = 5):
    # basically we take a matrix, broadcast it to match captionBatches, then calculate the scores
    # this will compute matrix to batch multiplication
    # we need this to calculate dissimilarity scores
    localScores=[]
    globalScores=[]
    #print(sequenceLengths)
    for i in range(len(sentenceBatch)):
        # rqdvector will become row of local score matrix
        # globalAttentionVector will become row of global score matrix
        RQDVector, globalAttentionVector = calculateAttentionMatchingScore(sentenceBatch[i].unsqueeze(0)
                                                                           , wordMatrixBatch[i].unsqueeze(0)
                                                                           , localImageBatch
                                                                           , sequenceLengths[i]
                                                                           , globalImageBatch
                                                                           , gammaOne, gammaTwo)
        '''
        RQDVector, globalAttentionVector =calculateAttentionMatchingScore(sentenceBatch,wordMatrixBatch
                                                                          ,localImageBatch[i].unsqueeze(0)
                                                                          ,globalImageBatch[i].unsqueeze(0)
                                                                          ,gammaOne, gammaTwo)
        '''
        localScores.append(RQDVector)
        globalScores.append(globalAttentionVector)
        #print(globalAttentionVector)

    # now we have score matrices for local and global matching
    # we can compute directly the damsm loss
    # concatenating localscores and global scores so each element is a column
    localMatrix = torch.stack(localScores,dim=1)
    #print(localMatrix)
    globalMatrix = torch.stack(globalScores,dim=1)
    #print(globalMatrix)
    return calculateDAMSMLoss(localMatrix,globalMatrix, match_labels)



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
# sentence embeddings have shape: (batch,features)
# word embeddings have shape: (batch, features, sequenceLength) -> i.e word embeddings are columns
# localImage output has shape: (batch, numFeatures, 17 , 17)
# globalImage output has shape: (batch,numFeatures)
def calculateAttentionMatchingScore(sentenceEmbedding, wordMatrixEmbedding, localImagePerceptronOutput,
              sequenceLength,
              globalImagePerceptronOutput,
              gammaOne=5, gammaTwo=5):
    # get similarity matrix of words and subregions
    # s = e^T v
    #print(wordMatrixEmbedding.transpose(1,2).shape)
    #print(localImagePerceptronOutput.flatten(start_dim=2).shape)

    # important! we need to calculate matching score between an image and every caption

    # sentenceEmbedding is of shape (1,256)
    # wordMatrixEmbedding is of shape(1,256,sequenceLength)

    # reducing the sequenceLength of word embedding to the true value
    wordMatrixEmbedding = wordMatrixEmbedding[:,:,:sequenceLength]
    # repeating word matrices and sentence embeddings to match image batch
    #sentenceEmbedding = sentenceEmbedding.repeat(globalImagePerceptronOutput.shape[0],1)
    wordMatrixEmbedding = wordMatrixEmbedding.repeat(localImagePerceptronOutput.shape[0],1,1)

    # transposing the matrix of word embeddings for each batch to have shape (batch,sequenceLength,features)
    # flattening localImage to have shape (batch,numFeatures,289)
    localImagePerceptronOutput = localImagePerceptronOutput.flatten(start_dim=2)
    s = torch.matmul(wordMatrixEmbedding.transpose(1,2) ,localImagePerceptronOutput)
    #s = torch.matmul(localImagePerceptronOutput.transpose(dim0=1,dim1=2),wordMatrixEmbedding)

    #  normalize similarity matrix batch for each matrix in the batch
    # s has shape ( batch, sequenceLength,289)
    # summing along columns
    s = torch.softmax(s,dim=1)
    '''
    s = torch.exp(s)
    s = s / torch.sum(s,dim=1,keepdim=True)
    '''
    #print(s)



    # get region context vectors as a matrix
    # this is the result of an alpha matrix elementwise multiplication with the word vectors
    # then we sum everything together
    alphaMatrix = gammaOne * s
    alphaMatrix = torch.softmax(alphaMatrix,dim=2)
    #alphaMatrix = torch.exp(gammaOne * s)
    # summing over each column
    '''
    print(alphaMatrix.shape)
    print(torch.sum(alphaMatrix,dim =1))
    print(torch.sum(alphaMatrix,dim=1,keepdim=True))
    print(torch.sum(alphaMatrix, dim=1).shape)
    print(torch.sum(alphaMatrix, dim=1, keepdim=True).shape)
    '''
    #alphaMatrix = alphaMatrix / torch.sum(alphaMatrix,dim = 2,keepdim=True)
    #print(alphaMatrix)

    # multiplying each a_j with v_j by matrix multiplication to get a sub-region word context matrix by broadcasting
    # recall that sub regions are columns in the localImagePerceptronOutput
    # so, we take the transpose before multiplication
    # alphaMatrix has shape (batch, sequenceLength, 289)
    # localImagePerceptron ouptut has shape (batch, features, 289) so we transpose dims 1 and 2
    contextMatrix = torch.matmul(alphaMatrix,localImagePerceptronOutput.transpose(1,2))
    #print(contextMatrix)

    # context matrix has shape (batch,sequenceLength, features)
    #print(contextMatrix.shape)

    # grabbing word relevance to image with cosine similarity between each context vector and the i-th word embedding
    # its a column vector of dot products basically
    # since we only care about matching context i to word i, we take the diagonal
    relevanceVector =  torch.matmul(contextMatrix,wordMatrixEmbedding).diagonal(dim1=-2, dim2=-1)

    # relevance vector currently has shape (batch, sequenceLength, sequenceLength)
    #print(relevanceVector.shape)
    # element wise division by the norms of each context vector, and for each word vector
    # basically just elementwise square, then sum along column, and then take sqrt
    #print(torch.norm(contextMatrix,dim=2).shape)
    # clamping to avoid divide by 0
    relevanceVector /= torch.norm(contextMatrix,dim=2).clamp(1e-8)
    #relevanceVector /= torch.sqrt(torch.sum(torch.pow(contextMatrix,2),dim=1,keepdim=True))
    # norming along columns, since columns hold features
    #print(torch.norm(wordMatrixEmbedding,dim=1).shape)
    # clamping to avoid divide by 0
    relevanceVector /= torch.norm(wordMatrixEmbedding,dim=1).clamp(1e-8)
    # need to sum along row here since the i-th feature vector of the i-th word is the i-th column
    #relevanceVector /= torch.sqrt(torch.sum(torch.pow(wordMatrixEmbedding,2),dim=2,keepdim=True))

    #print(relevanceVector.shape)

    #print(torch.log(torch.sum(torch.exp(5.0*relevanceVector),dim=1)))
    RQD = torch.logsumexp(relevanceVector*gammaTwo,dim=1)
    #print(RQD.shape)
    #RQD = torch.log(torch.sum(torch.exp(5.0*relevanceVector),dim=1))

    # getting entire attention driven matching score between
    # entire image (Q) and whole text description (D)
    #RQD = torch.pow(torch.logsumexp(relevanceVector*gammaTwo,dim=1),1/gammaTwo)
    '''
    RQD = torch.log(
        torch.sum(
            torch.exp(gammaTwo * relevanceVector),dim=1
        )
    ) ** (1/gammaTwo)
    '''
    #print(RQD)

    # getting a global score between the image and sentence without local components
    #print(globalImagePerceptronOutput.shape)
    #print(sentenceEmbedding.shape)
    #print(globalImagePerceptronOutput)
    globalAttentionScore = torch.matmul(globalImagePerceptronOutput,sentenceEmbedding.transpose(0,1)).squeeze(1)
    #print(globalAttentionScore.shape)
    globalAttentionScore /= torch.norm(globalImagePerceptronOutput,dim=1).clamp(1e-8)
    globalAttentionScore /= torch.norm(sentenceEmbedding,dim=1).clamp(1e-8)
    return RQD, globalAttentionScore

# we are given a score matrix where the i,jth element is the attention driven matching score
# between the i-th image and the j-th sentence
# we are also given a matrix for the same thing as above, but with global consideration only
# since we have both M images and sentences in a batch, these will be MxM matrices
# gammaThree is a smoothing factor
def calculateDAMSMLoss(localAttentionDrivenScoreMatrix, globalAttentionDrivenScoreMatrix, match_labels
                       , gammaThree=10):
    # calculating P(D_i|Q_i) matrix -> matching description to image
    powered = torch.exp(gammaThree * localAttentionDrivenScoreMatrix)
    #powered = gammaThree * localAttentionDrivenScoreMatrix
    # just doing softmax with identity matrix as labels
    #localLossOne = torch.nn.CrossEntropyLoss()(powered,match_labels)
    powered1 = powered.transpose(0,1)
    #localLossTwo = torch.nn.CrossEntropyLoss()(powered.transpose(0,1),match_labels)
    # need to sum along each column and scale the row by the entire row sum
    #print(powered)
    #print(torch.sum(powered,dim=1,keepdim=True))
    pDQ =  powered / torch.sum(powered,dim=1,keepdim=True)
    #print(pDQ)

    # calculating P(Q_i | D_i) matrix -> matching image to description
    # need to sum along rows here, and scale the column by the entire column sum
    #print(torch.sum(powered,dim=0,keepdim=True))
    pQD = powered/ torch.sum(powered,dim=0,keepdim=True)
    #print(pQD)

    # losses for local parts
    # just a negative log probability of an image being matched with its description
    localLossOne =  - torch.trace(torch.log(pDQ))
    # vice versa log of a description being matched with an image
    localLossTwo = -torch.trace(torch.log(pQD))

    # doing the same as above, but for the global components
    '''
    globalAttentionDrivenScoreMatrix = gammaThree * globalAttentionDrivenScoreMatrix
    globalLossOne = torch.nn.CrossEntropyLoss()(globalAttentionDrivenScoreMatrix,match_labels)
    globalLossTwo = torch.nn.CrossEntropyLoss()(globalAttentionDrivenScoreMatrix.transpose(0,1),match_labels)
    '''
    globalPowered = torch.exp(gammaThree * globalAttentionDrivenScoreMatrix)
    globalPDQ = globalPowered / torch.sum(globalPowered,dim=1)
    globalPQD = globalPowered/torch.sum(globalPowered,dim=0)



    # losses for global parts
    # same as above basically
    globalLossOne = -torch.trace(torch.log(globalPDQ))
    globalLossTwo = -torch.trace(torch.log(globalPQD))

    # returning losses
    return localLossOne , localLossTwo , globalLossOne ,globalLossTwo


def adv_D(D, opt, x, x_hat, s, lambda_MA, p, device):
    """
    Computes the advesarial loss for the Discriminator
    Args:
      D (Discriminator object): Discriminator
      opt (Discriminator Optimizer): Optimizer
      x (torch.Tensor, requires_grad=T): True image with shape (batch_size, channels,  height, width)
      x_hat (torch.Tensor): Fake generated image with same shape as x
      s (torch.Tensor, requires_grad=T): True text embedding with shape (batch_size, T_s)
      lambda_MA (float) : Weight of MA-GP loss
      p (float) : p hyperparameter
    Return:
      Loss_adv_D (float / torch.Tensor): Computed adversarial loss for Discriminator
      D_fake (torch.Tensor, requires_grad=T): Discriminator decisions for fake generated images with shape (batch_size, UNKNOWN_YET)
    """
    #print(x.shape)
    D_real = D(x, s)
    D_fake = D(x_hat, s)

    # Expected loss for failing to recognize real images (real loss)
    E_real = torch.maximum(torch.zeros(D_real.shape).to(device), 1.0 - D_real).mean()
    # Expected loss for getting fooled by generator
    E_fool = torch.maximum(torch.zeros(D_fake.shape).to(device), 1.0 + D_fake).mean()
    # Expected loss for getting wrong text pairing
    # Mismatched text
    s_hat = s[:(s.size(0) - 1)]
    x_mismatch = x[1:]
    mismatchedErrors = D(x_mismatch,s_hat)
    E_mismatch = torch.maximum(torch.zeros(mismatchedErrors.shape).to(device), 1.0 + mismatchedErrors).mean()

    # Calculate gradients of decision w.r.t. both x and s
    D_real.backward(torch.ones_like(D_real).to(device),retain_graph=True)
    #print(D_real.grad)
    #testGrad = [torch.autograd.grad(outputs=out,inputs=x[0],retain_graph=True)[0][i] for i,out in enumerate(D_real[0])]
    #print(testGrad)
    # Obtain calculated gradients and use them to get Eucl norms
    grad_D_real_x = x.grad.flatten(start_dim=1)
    grad_D_real_s = s.grad
    #print(grad_D_real_s.shape)
    #print(grad_D_real_x.shape)
    opt.zero_grad()
    grad_norm_x = torch.norm(grad_D_real_x)
    grad_norm_s = torch.norm(grad_D_real_s)
    # Expected MA-GP loss
    E_MA = torch.pow(grad_norm_x + grad_norm_s, p).mean()

    # Final weighted adv loss for discriminator
    Loss_adv_D = E_real + ((E_fool + E_mismatch) / 2.0) + lambda_MA * (E_MA)
    return Loss_adv_D, D_fake
'''
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
'''

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
def totalLoss( generatorLoss, damsmLoss, damsmWeight = 0.05):
    #print(generatorLoss.shape)
    #print(damsmLoss.shape)
    return generatorLoss + (damsmWeight * (damsmLoss[0]+damsmLoss[1]+damsmLoss[2]+damsmLoss[3]))