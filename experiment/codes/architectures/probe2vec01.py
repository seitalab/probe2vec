import torch
import torch.nn as nn
import torch.nn.functional as F

class probe2vec01(nn.Module):

    def __init__(self, params):
        super(probe2vec01, self).__init__()

        num_embed = params.i_dim
        h_dim = params.h_dim
        self.embed = nn.Embedding(num_embed, h_dim)

    def forward(self, X: torch.tensor):
        """
        Args:
            X (torch.tensor): Torch tensor of size [num_batch].
        Returns:
            hx (torch.tenor): Torch tensor of size [num_batch, h_dim].
        """
        hx = self.embed(X)
        return hx
