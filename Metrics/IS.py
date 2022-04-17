# calculating Inception Score for GANs

# idea is to compute KL divergence between image label distribution and the generated marginal distribution

# IS = exp(Expected_x KL(p(y|x) || p(y)) )

# p(y) -> y is the label
# p(y|x) -> label given an image

# split images into groups
# the paper just uses the openAI code for calculating IS
def isScore(images, split):
    pass


