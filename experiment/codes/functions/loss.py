import torch
import torch.nn as nn
import torch.nn.functional as F

class W2VLoss(nn.Module):

    def __init__(self):
        super(W2VLoss, self).__init__()

    def forward(self, pair1, pair2, label):

        pair1 = pair1.unsqueeze(1)
        pair2 = pair2.unsqueeze(-1)

        dot = torch.tensordot(pair1, pair2)
        dot = dot.squeeze(-1)

        loss = (label - torch.sigmoid(dot)).sum()
        return loss
