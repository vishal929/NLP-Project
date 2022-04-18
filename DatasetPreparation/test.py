import pickle
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torchvision
import sys
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import Losses.Loss as loss

''' below is only for notebook '''
'''
parent_path = "/content/drive/My Drive/NLP-Project/"
sys.path.append("{}DatasetPreparation".format(parent_path))
sys.path.append("{}Losses".format(parent_path))
sys.path.append("{}Metrics".format(parent_path))
sys.path.append("{}Modules".format(parent_path))
sys.path.append("{}TextEncoder".format(parent_path))
'''
# import prepDataset as prepD
# from Loss import adv_D, adv_G
# from Discriminator import Discriminator as D

from TextEncoder.RNNEncoder import bilstmEncoder
from ImageEncoder.CNNEncoder import cnnEncoder
import DatasetPreparation.prepDataset as prepD

'''function definitions that the author provides'''
''' this is for testing purposes with our implementation'''

def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = torch.nn.Softmax()(attn)  # Eq. (8)
    #print(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = torch.nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)
    #print(weightedContext)

    return weightedContext, attn.view(batch_size, -1, ih, iw)

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def sent_loss(cnn_code, rnn_code, labels, class_ids,batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = class_ids == class_ids[i]
            mask[i] = 0
            masks.append(mask.view(1, -1))
        # masks: batch_size x batch_size
        masks = torch.cat(masks, dim=0)
        #if cfg.CUDA:
        masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    #print(cnn_code)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * 10

    #print(scores0)

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    #print(scores0)
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = torch.nn.CrossEntropyLoss()(scores0, labels)
        loss1 = torch.nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1

def words_loss(img_features, words_emb, labels,cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i])
            mask[i] = 0
            masks.append(mask.view(1, -1))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        weiContext, attn = func_attention(word, context, 5.0)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)
        #print(row_sim)

        # Eq. (10)
        #print(torch.log(torch.sum(torch.exp(5.0*row_sim),dim=1)))
        row_sim.mul_(5.0).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)


        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    #print(similarities)
    if class_ids is not None:
        # masks: batch_size x batch_size
        masks = torch.cat(masks, dim=0)
        #if cfg.CUDA:
        masks = masks.cuda()

    similarities = similarities * 10.0
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = torch.nn.CrossEntropyLoss()(similarities, labels)
        loss1 = torch.nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps
if __name__ == '__main__':
    # image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop(256),
        torchvision.transforms.RandomHorizontalFlip()])

    # grab dataset
    cubImageDir = '../CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/CUB_200_2011/CUB_200_2011/' \
                  'images'
    cubCaptionDir = '../CUBS Dataset/Cubs-2011/bird_metadata/birds/text/text'

    # place where cub metadata is
    cubBoundingBoxFile = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                         'CUB_200_2011/CUB_200_2011/bounding_boxes.txt'
    cubFileMappings = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                      'CUB_200_2011/CUB_200_2011/images.txt'
    cubClasses = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                 './CUB_200_2011/image_class_labels.txt'
    cubTrainTestSplit = './CUBS Dataset/Cubs-2011/cub-200-2011-20220408T185459Z-001/cub-200-2011/' \
                        'CUB_200_2011/CUB_200_2011/train_test_split.txt'

    cubDataset = prepD.imageCaptionDataset(cubCaptionDir, cubImageDir, 10, transform, 'train', 18)

    with open('./CUBMetadata/TheirCaptions/captions.pickle', 'rb') as f:
        x = pickle.load(f)
        wordtoix = x[3]
        for word in cubDataset.wordToIndex:
            if word not in wordtoix:
                print(word)
    # vocab_size,
    # input length (caption words are sent to an embedding of this length),
    # dropout probability
    # hidden features of both directions of the bilstm
    # stepSize is the length of caption inputs -> 18
    text_encoder = bilstmEncoder(len(cubDataset.indexToWord), 300, 0.5, 256, 18)



    # loading pre trained parameters of the text_encoder for cub
    cubTextEncoderPath = "../CubDamsm/text_encoder200.pth"
    textEncoderState = torch.load(cubTextEncoderPath, map_location=lambda storage,loc: storage)
    modifiedState = OrderedDict()
    for key, values in textEncoderState.items():
        if key[:7] == 'encoder':
            # replace this name
            key = 'embedding' + key[7:]
        elif key[:3] == 'rnn':
            key = 'LSTM' + key[3:]
            # replace this name
        modifiedState[key] = values

    # loading saved model params to the text encoder
    text_encoder.load_state_dict(modifiedState)

    # sending text encoder to gpu
    text_encoder.cuda()

    # setting parameters to be fixed
    for param in text_encoder.parameters():
        param.requires_grad = False

    text_encoder.eval()

    # try loading the image encoder
    image_encoder = cnnEncoder()

    # loading pre trained weights to the image encoder
    cubImageEncoderPath = "../CubDamsm/image_encoder200.pth"
    imageEncoderState = torch.load(cubImageEncoderPath, map_location=lambda storage, loc: storage)
    imageModifiedState= OrderedDict()
    for key, values in imageEncoderState.items():
        if key[0:13] == 'Conv2d_1a_3x3':
            key = 'inceptionUpToLocalFeatures.0' + key[13:]
        elif key[0:13] == 'Conv2d_2a_3x3':
            key = 'inceptionUpToLocalFeatures.1' + key[13:]
        elif key[0:13] == 'Conv2d_2b_3x3':
            key = 'inceptionUpToLocalFeatures.2' + key[13:]
        elif key[0:13] == 'Conv2d_3b_1x1':
            key = 'inceptionUpToLocalFeatures.4' + key[13:]
        elif key[0:13] == 'Conv2d_4a_3x3':
            key = 'inceptionUpToLocalFeatures.5' + key[13:]
        elif key[0:8] == 'Mixed_5b':
            key = 'inceptionUpToLocalFeatures.7' + key[8:]
        elif key[0:8] == 'Mixed_5c':
            key = 'inceptionUpToLocalFeatures.8' + key[8:]
        elif key[0:8] == 'Mixed_5d':
            key = 'inceptionUpToLocalFeatures.9' + key[8:]
        elif key[0:8] == 'Mixed_6a':
            key = 'inceptionUpToLocalFeatures.10' + key[8:]
        elif key[0:8] == 'Mixed_6b':
            key = 'inceptionUpToLocalFeatures.11' + key[8:]
        elif key[0:8] == 'Mixed_6c':
            key = 'inceptionUpToLocalFeatures.12' + key[8:]
        elif key[0:8] == 'Mixed_6d':
            key = 'inceptionUpToLocalFeatures.13' + key[8:]
        elif key[0:8] == 'Mixed_6e':
            key = 'inceptionUpToLocalFeatures.14' + key[8:]
        elif key[0:8] == 'Mixed_7a':
            key = 'restOfInception.0' + key[8:]
        elif key[0:8] == 'Mixed_7b':
            key = 'restOfInception.1' + key[8:]
        elif key[0:8] == 'Mixed_7c':
            key = 'restOfInception.2' + key[8:]
        elif key[0:12] == 'emb_features':
            key = 'embedLocalFeatures' + key[12:]
        elif key[0:12] == 'emb_cnn_code':
            key = 'embedGlobalFeatures' + key[12:]
        imageModifiedState[key] = values

    image_encoder.load_state_dict(imageModifiedState)

    image_encoder = image_encoder.cuda()

    for param in image_encoder.parameters():
        param.requires_grad = False

    image_encoder.eval()

    # loading data
    dataloader = torch.utils.data.DataLoader(
        cubDataset, batch_size = 5, drop_last = True,
        shuffle=True, num_workers = 2
    )

    # grabbing some random batch of 5 images and captions
    train_images, train_captions, captionLengths, classID = next(iter(dataloader))
    train_captions = train_captions.type(torch.LongTensor)
    captionLengths = captionLengths.type(torch.LongTensor)
    print(train_images.shape)
    print(train_captions.shape)
    print(captionLengths.shape)
    print(classID.shape)

    #print(train_captions)

    # squeezing labels so they are (batchSize, 18) instead of (batchsize,18,1)
    train_captions = train_captions.squeeze()
    print(train_captions.shape)

    # sending data to gpu
    train_captions = train_captions.cuda()
    train_images = train_images.cuda()
    captionLengths = captionLengths.cuda()

    # resetting text encoder state
    newHidden = text_encoder.initHiddenStates(5)

    wordEmbeddings, sentenceEmbeddings = text_encoder(train_captions,captionLengths,newHidden)

    print('word embeddings shape:' + str(wordEmbeddings.shape))
    print('sentence embeddings shape:' + str(sentenceEmbeddings.shape))

    # getting image encodings

    localImageFeatures, globalImageFeatures = image_encoder(train_images)

    print('local image features shape: ' + str(localImageFeatures.shape))
    print('global image features shape: ' + str(globalImageFeatures.shape))

    # authors prepare real labels and fake labels, with size batch_size
    real_labels = torch.FloatTensor(5).fill_(1).cuda()
    fake_labels = torch.FloatTensor(5).fill_(0).cuda()
    match_labels = torch.LongTensor(5).cuda()

    # getting value of sentence and word matching losses
    #words_loss(img_features, words_emb, labels,cap_lens, class_ids, batch_size)
    localHolder = words_loss(localImageFeatures, wordEmbeddings, match_labels,captionLengths, classID, 5)

    #sent_loss(cnn_code, rnn_code, labels, class_ids,batch_size, eps=1e-8)
    globalHolder = sent_loss(globalImageFeatures,sentenceEmbeddings,match_labels,classID,5)

    loss0Local = localHolder[0]
    loss1Local = localHolder[1]
    loss1Global = globalHolder[0]
    loss2Global = globalHolder[1]

    print('loss 0 local shape: ' + str(loss0Local.shape))
    print('loss 1 local shape: ' + str(loss1Local.shape))
    print('loss 0 global shape: ' + str(loss1Global.shape))
    print('loss 1 global shape: ' + str(loss1Global.shape))
    # values
    print('loss 0 local : ' + str(loss0Local))
    print('loss 1 local : ' + str(loss1Local))
    print('loss 0 global : ' + str(loss1Global))
    print('loss 1 global : ' + str(loss2Global))

    # our implementation
    ourLoss0Local, ourLoss1Local, ourLoss0Global, ourLoss1Global = loss.calculateAttentionMatchingScoreBatchWrapper(sentenceEmbeddings, wordEmbeddings, localImageFeatures, globalImageFeatures,match_labels)
    print('our l0 local: ' + str(ourLoss0Local))
    print('our l1 local: ' + str(ourLoss1Local))
    print('our l0 global: ' + str(ourLoss0Global))
    print('our l1 global: ' + str(ourLoss1Global))



