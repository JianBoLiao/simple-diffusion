import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from udsample import Dy_DownSample, Dy_UpSample



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


class Tune_kernel(nn.Module):
    def __init__(self, num_in, num_out, dropout=0.0):
        super().__init__()
        self.tune = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_in, num_out)
        )

    def forward(self, x):
        x = self.tune(x)
        return x

class Tune_kernel_relu(nn.Module):
    def __init__(self, num_in, num_out, dropout=0.0):
        super().__init__()
        self.tune = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_in, num_out)
        )

    def forward(self, x):
        x = self.tune(x)
        x = F.relu(x)
        return x

class Co_dysample(nn.Module):
    def __init__(self, c_in, groups=4, dyscope=True):
        super().__init__()
        self.c_in = c_in
        self.dy_downSample = Dy_DownSample(c_in, groups=groups, dyScope=dyscope)
        self.dy_upSample = Dy_UpSample(c_in, groups=groups, dyScope=dyscope)
        self.register_buffer("downkernel", self.dy_downSample.offset_weight)    # 这玩意只用一次
        self.register_buffer("upkernel", self.dy_upSample.offset_weight)


        up_num = self.upkernel.numel()
        down_num = self.downkernel.numel()
        x1, x2, _, _ = self.downkernel.shape
        x3, x4, _, _ = self.upkernel.shape
        self.rearr2down = Rearrange('(x1 x2) -> x1 x2 1 1', x1=x1, x2=x2)
        self.rearr2up = Rearrange('(x3 x4) -> x3 x4 1 1', x3=x3, x4=x4)
        self.d2u = Tune_kernel(down_num, up_num, 0.1)
        self.u2d = Tune_kernel(up_num, down_num, 0.1)
        self.gated2u = 0.5
        self.gateu2d = 0.5

    def down_forward(self, x):
        tune = self.rearr2down(self.u2d(self.upkernel.clone().flatten()))
        x = self.dy_downSample(x=x, tune=tune)
        return x
            
    

    def up_forward(self, x):
        tune = self.rearr2up(self.d2u(self.dy_downSample.offset_weight.flatten()))
        x = self.dy_upSample(x=x, tune=tune)
        self.upkernel = self.dy_upSample.offset_weight
        return x

class Co_dysample_relu(nn.Module):
    def __init__(self, c_in, groups=4, dyscope=True):
        super().__init__()  
        self.c_in = c_in
        self.dy_downSample = Dy_DownSample(c_in, groups=groups, dyScope=dyscope)
        self.dy_upSample = Dy_UpSample(c_in, groups=groups, dyScope=dyscope)
        self.register_buffer("downkernel", self.dy_downSample.offset_weight)    # 这玩意只用一次
        self.register_buffer("upkernel", self.dy_upSample.offset_weight)


        up_num = self.upkernel.numel()
        down_num = self.downkernel.numel()
        x1, x2, _, _ = self.downkernel.shape
        x3, x4, _, _ = self.upkernel.shape
        self.rearr2down = Rearrange('(x1 x2) -> x1 x2 1 1', x1=x1, x2=x2)
        self.rearr2up = Rearrange('(x3 x4) -> x3 x4 1 1', x3=x3, x4=x4)
        self.d2u = Tune_kernel_relu(down_num, up_num, 0.1)
        self.u2d = Tune_kernel_relu(up_num, down_num, 0.1)
        self.gated2u = 0.5
        self.gateu2d = 0.5

    def down_forward(self, x):
        tune = self.rearr2down(self.u2d(self.upkernel.clone().flatten()))
        x = self.dy_downSample(x=x, tune=tune)
        return x
            
    

    def up_forward(self, x):
        tune = self.rearr2up(self.d2u(self.dy_downSample.offset_weight.flatten()))
        x = self.dy_upSample(x=x, tune=tune)
        self.upkernel = self.dy_upSample.offset_weight
        return x


if __name__ == '__main__':
    x = torch.randn(size=(8, 32, 100, 240))


    # my_dysample = Dy_UpSample(32, style='lp', groups=4)
    # dysample = DySample(32, style='lp', groups=4)
    # x1 = my_dysample(x)
    # x2 = dysample(x)
    # print(my_dysample.offset.weight.size)
    # print(dysample.offset.weight.size)
    # print(torch.allclose(my_dysample.offset.weight, dysample.offset.weight, 1e-2, 1e-2))
    # print(torch.allclose(x1, x2, 1e-2, 1e-2))

    co_dysample = Co_dysample(32)
    x1 = co_dysample.down_forward(x)
    print(x1.size())
    print(co_dysample.up_forward(x1).size())