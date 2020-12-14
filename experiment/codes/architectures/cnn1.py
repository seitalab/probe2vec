import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn1(nn.Module):

    def __init__(self, params):
        super(cnn1, self).__init__()

        # Conv parameters are for MNIST (hard coded)

        h_dim = params.h_dim
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64

        self.dropout1 = nn.Dropout2d()
        self.dense1 = nn.Linear(12 * 12 * 64, h_dim)
        self.dropout2 = nn.Dropout2d()
        self.dense2 = nn.Linear(h_dim, params.o_dim)

    def forward(self, X: torch.tensor):
        """
        Args:
            X (torch.tensor): Torch tensor of size [num_batch, w, h] (w, h = 28).
        Returns:
            hx (torch.tenor): Torch tensor of size [num_batch, o_dim] (o_dim=10).
        """
        bs = X.size(0)
        X = X.unsqueeze(1)

        hx = F.relu(self.conv1(X))
        hx = self.pool(F.relu(self.conv2(hx)))
        hx = self.dropout1(hx)

        hx = hx.view(bs, -1)
        hx = F.relu(self.dense1(hx))
        hx = self.dropout2(hx)
        hx = self.dense2(hx)
        return hx
