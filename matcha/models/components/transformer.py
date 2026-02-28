import torch.nn as nn


class BasicTransformerBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x
