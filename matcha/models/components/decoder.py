import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x


class Block1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ResnetBlock1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Downsample1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Upsample1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
