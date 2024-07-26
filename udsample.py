import torch
# import wandb
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import einops



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



class Dy_DownSample(nn.Module):
    def __init__(self, c_in, style='lp', gate=0.5, ratio=2, groups=1, dyScope=True, module_id=0):
        super().__init__()
        self.gate = gate
        self.ratio =ratio
        self.style = style
        self.groups = groups
        self.dySample = dyScope
        self.module_id = module_id
        assert style in ['lp', 'pl']
        assert c_in % groups == 0

        # upsampling 是分为 linear+pixel-shuffle 和 pixel-shuffle+linear
        # downsampling 分为 linear+pixel-unshuffle 和 pixel-unshuffle+linear
        # if ratio > 1:
        if style == 'lp':
            assert 2 * groups % ratio**2 == 0
            c_out = 2 * int(groups / ratio**2)
        else:
            assert c_in >= groups / ratio**2
            c_out = 2 * groups
            c_in = c_in * ratio**2


        if dyScope:
            self.scope = nn.Conv2d(c_in, c_out, kernel_size=1)
            constant_init(self.scope, val=0.)

        self.offset_weight = nn.Parameter(
            torch.normal(mean=0, std=0.001, size=(c_out, c_in, 1, 1))
        )
        self.offset_bias = nn.Parameter(
            torch.normal(mean=0, std=0.001, size=(c_out,))
        )


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

    def forward_lp(self, x, tune):

        mean1 = tune.mean()
        std1 = tune.std()
        mean2 = self.offset_weight.mean()
        std2 = self.offset_weight.std()
        tune = (tune - mean1) / std1 * std2 + mean2



        weight = self.offset_weight * self.gate + (1 - self.gate) * tune
        offset = F.conv2d(x, weight=weight, bias=self.offset_bias)

        if self.dySample:
            offset = F.sigmoid(self.scope(x)) * offset * 0.5
            # wandb.log({f"{self.module_id}.offset": wandb.Histogram(offset.data.cpu().clone().detach())})
        else:
            offset = 0.25 * offset


        offset = F.pixel_unshuffle(offset, downscale_factor=self.ratio)       # b (grp two) h w
        
        return self.Sample(x, offset)
    
    def forward_pl(self, x, tune):
        raise NotImplementedError
        y = F.pixel_unshuffle(x, downscale_factor=self.ratio)
        weight = self.offset_weight * self.gate + (1 - self.gate) * tune
        offset = F.conv2d(x, weight=weight, bias=self.offset_bias)
        offset = F.sigmoid(offset)

        if self.dySample:
            offset = F.sigmoid(self.scope(y)) * 0.5 * offset

        else:
            offset = 0.25 * offset
        return self.Sample(x, offset)

    def forward(self, x, tune):
        _, _, h, w = x.size()
        padh = self.ratio - h % self.ratio if h % self.ratio else 0
        padw = self.ratio - w % self.ratio if w % self.ratio else 0
        # padh = padw = 0
        x = F.pad(x, (padw//2, padw-padw//2, padh//2, padh-padh//2), mode='replicate')
        if self.style == 'lp':
            return self.forward_lp(x, tune)
        return self.forward_pl(x, tune)




class Dy_UpSample(nn.Module):
    def __init__(self, c_in, style='lp', gate=0.5, ratio=2, groups=4, dyScope=True, module_id=0):
        super().__init__()
        self.gate = gate
        self.ratio =ratio
        self.style = style
        self.groups = groups
        self.dySample = dyScope
        self.module_id = module_id
        assert style in ['lp', 'pl']
        assert c_in % groups == 0

        # upsampling 是分为 linear+pixel-shuffle 和 pixel-shuffle+linear
        # downsampling 分为 linear+pixel-unshuffle 和 pixel-unshuffle
        if style == 'lp':
            c_out = int(2 * groups * ratio**2)
        else:
            assert c_in >= groups * ratio**2
            c_out = 2 * groups
            c_in = int(c_in // ratio**2)
        

        if dyScope:
            self.scope = nn.Conv2d(c_in, c_out, kernel_size=1)
            constant_init(self.scope, val=0.)

        self.offset_weight = nn.Parameter(
            torch.normal(mean=0, std=0.001, size=(c_out, c_in, 1, 1))
        )
        self.offset_bias = nn.Parameter(
            torch.normal(mean=0, std=0.001, size=(c_out,))
        )


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

    def forward_lp(self, x, tune):
        # offset = self.offset(x)


        mean1 = tune.mean()
        std1 = tune.std()
        mean2 = self.offset_weight.mean()
        std2 = self.offset_weight.std()
        tune = (tune - mean1) / std1 * std2 + mean2


        weight = self.offset_weight * self.gate + (1 - self.gate) * tune
        offset = F.conv2d(x, weight=weight, bias=self.offset_bias)  # b (grp two ratio ratio) h w


        if self.dySample:
            offset = F.sigmoid(self.scope(x)) * offset * 0.5
        else:
            offset = 0.25 * offset
        offset = F.pixel_shuffle(offset, upscale_factor=self.ratio)

        # offset = einops.rearrange(offset, 'b (grp two) h w -> b grp two h w', grp=self.groups)
        

        return self.Sample(x, offset)
    
    def forward_pl(self, x, tune):
        y = F.pixel_shuffle(x, upscale_factor=self.ratio)
        # offset = self.offset(y)
        weight = self.offset_weight * self.gate + (1 - self.gate) * tune
        offset = F.conv2d(x, weight=weight, bias=self.offset_bias)


        if self.dySample:
            offset = F.sigmoid(self.scope(y)) * 0.5 * offset
        else:
            offset = 0.25 * offset
        return self.Sample(x, offset)

    def forward(self, x, tune):
        if self.style == 'lp':
            return self.forward_lp(x, tune)
        return self.forward_pl(x, tune)

if __name__ == '__main__':
    x = torch.randn(1, 16, 32, 32)
    dy_downsample = Dy_DownSample(16, style='lp', ratio=2, groups=4, dyScope=True)
    print(dy_downsample(x, tune=torch.rand(size=(1,1,1,1))).size())
    # dy_upsample = Dy_UpSample(16, style='lp', ratio=2, groups=4, dyScope=True)
    # print(dy_upsample(x, tune=torch.rand(size=(1,1,1,1))).size())