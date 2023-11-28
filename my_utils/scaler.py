import torch
import torch.nn as nn

class MinMaxScalerLayer(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0, eps=1e-5):
        super(MinMaxScalerLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    def forward(self, x):
        x_min = x.min().detach()
        x_max = x.max().detach()
        x_norm = (x - x_min) / (x_max - x_min + self.eps)
        return x_norm * (self.max_val - self.min_val) + self.min_val