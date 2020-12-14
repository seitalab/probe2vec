import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp1(nn.Module):

    def __init__(self, params):
        super(mlp1, self).__init__()

        h_dim = params.h_dim
        i_dim = params.i_dim * params.i_dim # h28 x w28 -> 784dim
        self.dense1 = nn.Linear(i_dim, h_dim)
        self.dense2 = nn.Linear(h_dim, h_dim)
        self.dense3 = nn.Linear(h_dim, params.o_dim)

    def forward(self, X: torch.tensor):
        """
        Args:
            X (torch.tensor): Torch tensor of size [num_batch, w, h] (w, h = 28).
        Returns:
            hx (torch.tenor): Torch tensor of size [num_batch, o_dim] (o_dim=10).
        """
        bs = X.size(0)
        X = X.view(bs, -1)

        hx = F.relu(self.dense1(X))
        hx = F.relu(self.dense2(hx))
        hx = self.dense3(hx)
        return hx
