# fully connected block to reshape input of gaussian noise vector
import torch

# fully connected block to basically reshape gaussian noise vector from dim 100 to 4x4x512
class FCBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fullyConnected = torch.nn.Linear(100,8192)

    def forward(self,x):
        # running fully connected, and then reshaping
        # reshaping to batch size x 512 x 4 x 4
        return self.fullyConnected(x).reshape(x.size(0), 512, 4, 4)