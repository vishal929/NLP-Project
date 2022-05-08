# NLP-Project

## Environment Setup:

For this project, we used the following additional modules in a Conda Python 3.8 environment: 
* Pytorch Version 1.11
* Torchvision 0.12
* Numpy Version 1.21.5
* Pillow Version 9.0.1
* NLTK Version 3.7
* TQDM Version 4.64.0
* Tensorboard Version 2.6.0

## SSA-GAN: 

This is an implementation of the SSA-GAN provided by this paper: https://arxiv.org/pdf/2104.00567.pdf

This repo is NOT an official implementation, just an implementation for a class project

## Setup:

You can find the official implementation here: https://github.com/wtliao/text2image

There, you can find the birds dataset (CUB) and the pretrained text encoder and pretrained image encoder (DAMSM). 
In addition, their metadata includes the text descriptions, so make sure you download those as well. 

From there, you can place the pretrained pth files in the CubDamsm folder.
You can place the CUB dataset wherever you wish, just make sure the paths to the data files are correct in train.py, or whichever training file you use.

If this is first time setup, make sure you uncomment 'prepDataset.setupCub' in train.py or whichever training file you are using. 
This will generate metadata needed to setup dataloaders and get started with training.

## Evaluation:

You can run test.py to evaluate. Make sure the torch.load() matches up with the filename of your saved state!
test.py will sample from the test set 30,000 times and generate images according to the captions save them to a directory
called test_fake. The caption is saved into a directory test_captions. The transformed real images are saved to a 
directory called test_real. This is so computing FID is possible. 

To compute IS score, please use the author's original IS.py at the repo linked above. This uses the OpenAI Tensorflow
implementation to calculate IS, which is the recommended way to compute IS scores. 

To compute the FID score, please use this repo, which the original authors have also linked: 
https://github.com/bioinf-jku/TTUR

## Our Checkpoints:

Our checkpoints turned out to be not so great compared to the original author's implementation, but regardless 
we provide both the DAMSM and non-DAMSM checkpoints below to download: 
* DAMSM:
* non-DAMSM: 