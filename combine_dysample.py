import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import einops



class non_co_dysample(nn.Module):
    def __init__(self, c_in, groups=4, dyscope=True): # writer=None
        super().__init__()
        self.c_in = c_in
        # self.writer = writer
        self.Dy_DownSample = Dy_DownSample(c_in, groups=groups, dyScope=dyscope)
        self.Dy_UpSample = Dy_UpSample(c_in, groups=groups, dyScope=dyscope)


    def down_forward(self, x):
        x, self.down_offset = self.Dy_DownSample(x)
        return x

    def up_forward(self, x):
        self.down_offset *= self.c_in / 16
        x = self.Dy_UpSample(x, self.down_offset)
        return x


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


class Dy_UpSample(nn.Module):
    def __init__(self, c_in, style='lp', ratio=2, groups=4, dyScope=True):
        super(Dy_UpSample, self).__init__()
        self.ratio = ratio
        self.style = style
        self.groups = groups
        self.dySample = dyScope
        # self.weight = nn.Parameter(torch.ones(1))
        assert style in ['lp', 'pl']
        assert c_in % groups == 0

        # upsampling 是分为 linear+pixel-shuffle 和 pixel-shuffle+linear
        # downsampling 分为 linear+pixel-unshuffle 和 pixel-unshuffle
        if style == 'lp':
            c_out = int(2 * groups * ratio ** 2)
            c_mid = c_in + 2 * self.groups
        else:
            assert c_in >= groups * ratio ** 2
            c_out = 2 * groups
            c_in = int(c_in // ratio ** 2)

        if dyScope:
            self.scope = nn.Conv2d(c_mid, c_out, kernel_size=1)
            constant_init(self.scope, val=0.)

        self.offset = nn.Conv2d(c_mid, c_out, kernel_size=1)
        normal_init(self.offset, std=0.001)

    def Sample(self, x, offset):
        _, _, h, w = offset.size()
        x = einops.rearrange(x, 'b (c grp) h w -> (b grp) c h w', grp=self.groups)
        offset = einops.rearrange(offset, 'b (grp two) h w -> (b grp) h w two',
                                  two=2, grp=self.groups)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 1, 1, 2)

        h = torch.linspace(0.5, h - 0.5, h)
        w = torch.linspace(0.5, w - 0.5, w)
        pos = torch.stack(torch.meshgrid(w, h, indexing='xy')).to(x.device)
        pos = einops.rearrange(pos, 'two h w -> 1 h w two')
        pos = 2 * (pos + offset) / normalizer - 1

        out = F.grid_sample(x, pos, align_corners=False, mode='bilinear', padding_mode="border")
        out = einops.rearrange(out, '(b grp) c h w -> b (c grp) h w', grp=self.groups)
        return out

    def forward_lp(self, x, off):
        fusion = torch.cat([x, off], dim=1)
        offset = self.offset(fusion)
        if self.dySample:
            offset = F.sigmoid(self.scope(fusion)) * 0.5 * offset
        else:
            offset = 0.25 * offset
        offset = F.pixel_shuffle(offset, upscale_factor=self.ratio)
        return self.Sample(x, offset)

    def forward_pl(self, x):
        y = F.pixel_shuffle(x, upscale_factor=self.ratio)
        offset = self.offset(y)
        if self.dySample:
            offset = F.sigmoid(self.scope(y)) * 0.5 * offset
        else:
            offset = 0.25 * offset
        return self.Sample(x, offset)

    def forward(self, x, off):
        if self.style == 'lp':
            return self.forward_lp(x, off)
        return self.forward_pl(x)



class Dy_DownSample(nn.Module):
    def __init__(self, c_in, style='lp', ratio=2, groups=1, dyScope=True):
        super(Dy_DownSample, self).__init__()
        self.ratio = ratio
        self.style = style
        self.groups = groups
        self.dySample = dyScope
        assert style in ['lp', 'pl']
        assert c_in % groups == 0

        # upsampling 是分为 linear+pixel-shuffle 和 pixel-shuffle+linear
        # downsampling 分为 linear+pixel-unshuffle 和 pixel-unshuffle+linear
        # if ratio > 1:
        if style == 'lp':
            assert 2 * groups % ratio ** 2 == 0
            c_out = 2 * int(groups / ratio ** 2)
        else:
            assert c_in >= groups / ratio ** 2
            c_out = 2 * groups
            c_in = c_in * ratio ** 2

        if dyScope:
            self.scope = nn.Conv2d(c_in, c_out, kernel_size=1)
            constant_init(self.scope, val=0.)

        self.offset = nn.Conv2d(c_in, c_out, kernel_size=1)
        normal_init(self.offset, std=0.001)

    def Sample(self, x, offset):
        _, _, h, w = offset.size()
        x = einops.rearrange(x, 'b (c grp) h w -> (b grp) c h w', grp=self.groups)
        offset = einops.rearrange(offset, 'b (grp two) h w -> (b grp) h w two',
                                  two=2, grp=self.groups)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 1, 1, 2)

        h = torch.linspace(0.5, h - 0.5, h)
        w = torch.linspace(0.5, w - 0.5, w)
        pos = torch.stack(torch.meshgrid(w, h, indexing='xy')).to(x.device)
        pos = einops.rearrange(pos, 'two h w -> 1 h w two')
        pos = 2 * (pos + offset) / normalizer - 1
        off = (2 * offset / normalizer) - 1
        off = einops.rearrange(off, '(b grp) h w two-> b (grp two) h w', grp=self.groups)

        out = F.grid_sample(x, pos, align_corners=False, mode='bilinear', padding_mode="border")
        out = einops.rearrange(out, '(b grp) c h w -> b (c grp) h w', grp=self.groups)
        return out, off

    def forward_lp(self, x):
        offset = self.offset(x)
        if self.dySample:
            offset = F.sigmoid(self.scope(x)) * 0.5 * offset
        else:
            offset = 0.25 * offset
        offset = F.pixel_unshuffle(offset, downscale_factor=self.ratio)
        return self.Sample(x, offset)

    def forward_pl(self, x):
        y = F.pixel_unshuffle(x, downscale_factor=self.ratio)
        offset = self.offset(y)
        if self.dySample:
            offset = F.sigmoid(self.scope(y)) * 0.5 * offset
        else:
            offset = 0.25 * offset
        return self.Sample(x, offset)

    def forward(self, x):
        _, _, h, w = x.size()
        padh = self.ratio - h % self.ratio if h % self.ratio else 0
        padw = self.ratio - w % self.ratio if w % self.ratio else 0
        # padh = padw = 0
        x = F.pad(x, (padw // 2, padw - padw // 2, padh // 2, padh - padh // 2), mode='replicate')
        if self.style == 'lp':
            return self.forward_lp(x)
        return self.forward_pl(x)


if __name__ == '__main__':
    x = torch.zeros(size=(8, 32, 100, 240))
    co_dysample = non_co_dysample(32)
    x = co_dysample.down_forward(x)
    print(x.shape)
    x = co_dysample.up_forward(x)
    print(x.shape)

