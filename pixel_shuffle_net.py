import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class co_pixelshuffle(nn.Module):
    def __init__(self, in_channel, ratio=2):
        super().__init__()
        self.in_channel = in_channel
        self.ratio = ratio
        self.up_sample = pixelshuffle(in_channel, ratio)
        self.down_sample = pixelunshuffle(in_channel, ratio)
        self.co_weight = nn.Parameter(
            torch.normal(mean=0, std=0.001, size=(in_channel, in_channel, 1, 1))
        )
        self.co_bias = nn.Parameter(
            torch.normal(mean=0, std=0.001, size=(in_channel,))
        )

    def down_forward(self, x):
        x = F.conv2d(x, weight=self.co_weight, bias=self.co_bias)
        x = F.relu(x)
        x = self.down_sample(x)
        return x

    def up_forward(self, x):
        x = F.conv2d(x, weight=self.co_weight, bias=self.co_bias)
        x = F.relu(x)
        x = self.up_sample(x)
        return x


class pixelshuffle(nn.Module):
    def __init__(self, in_channel, ratio):
        super().__init__()
        self.ratio = ratio
        self.conv = nn.Conv2d(in_channel, in_channel * ratio**2, 1, 1)
        constant_init(self.conv, val=0.)

    def forward(self, x):
        x = self.conv(x)
        out = F.pixel_shuffle(x, self.ratio)
        return out

class pixelunshuffle(nn.Module):
    def __init__(self, in_channel, ratio):
        super().__init__()
        self.ratio = ratio
        self.conv = nn.Conv2d(in_channel, in_channel // ratio**2, 1, 1)
        constant_init(self.conv, val=0.)


    def forward(self, x):
        x = self.conv(x)
        out = F.pixel_unshuffle(x, self.ratio)
        return out
    
if __name__ == '__main__':
    x = torch.rand(size = (8, 96, 32, 32))
    print(x.size())
    co_sample = co_pixelshuffle(96)
    x1 = co_sample.up_forward(x)
    print(x1.size())
    print(co_sample.down_forward(x1).size())

