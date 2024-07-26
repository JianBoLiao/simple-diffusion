import torch.nn as nn
from diffusion_utilities import *
from combine_dysample import non_co_dysample
from kernel_normalized import Co_dysample, Co_dysample_relu
from pixel_shuffle_net import co_pixelshuffle

# from co_dysample import co_dysample
import pdb

class ContextUnet1(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_cfeat = 10, height = 16):
        super(ContextUnet1, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        # self.co_dysample1 = co_dysample(n_feat)
        # self.co_dysample2 = co_dysample(n_feat)
        self.down1 = UnetDown1(n_feat, n_feat)       # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown1(n_feat, 2 * n_feat)   # down2 #[10, 236, 4, 4]

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(4),
            nn.GELU()
        )

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )
        self.up1 = UnetUp1(4 * n_feat, n_feat)
        self.up2 = UnetUp1(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1)
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat) : time step
        c : (batch, n_classes) : context label
        """
        # pdb.set_trace()
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)

        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.concat((up3, x), 1))
        return out

class ContextUnet2(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_cfeat = 10, height = 16):
        super(ContextUnet2, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        # self.co_dysample1 = co_dysample(n_feat)
        # self.co_dysample2 = co_dysample(n_feat)
        self.down1 = UnetDown2(n_feat, n_feat)       # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown2(n_feat, 2 * n_feat)   # down2 #[10, 236, 4, 4]

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(4),
            nn.GELU()
        )

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )
        self.up1 = UnetUp2(4 * n_feat, n_feat)
        self.up2 = UnetUp2(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1)
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat) : time step
        c : (batch, n_classes) : context label
        """
        # pdb.set_trace()
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)

        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.concat((up3, x), 1))
        return out

class ContextUnet3(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_cfeat = 10, height = 16):
        super(ContextUnet3, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.co_dysample1 = Co_dysample(n_feat)
        self.co_dysample2 = Co_dysample(n_feat)
        self.down1 = UnetDown3(n_feat, n_feat, self.co_dysample1)       # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown3(n_feat, 2 * n_feat, self.co_dysample2)   # down2 #[10, 236, 4, 4]

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(4),
            nn.GELU()
        )

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )
        self.up1 = UnetUp3(4 * n_feat, n_feat, self.co_dysample2)
        self.up2 = UnetUp3(2 * n_feat, n_feat,self.co_dysample1)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1)
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat) : time step
        c : (batch, n_classes) : context label
        """
        # pdb.set_trace()
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)

        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.concat((up3, x), 1))
        return out

class ContextUnet4(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_cfeat = 10, height = 16):
        super(ContextUnet4, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.co_dysample1 = non_co_dysample(n_feat)
        self.co_dysample2 = non_co_dysample(n_feat)
        self.down1 = UnetDown3(n_feat, n_feat, self.co_dysample1)       # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown3(n_feat, 2 * n_feat, self.co_dysample2)   # down2 #[10, 236, 4, 4]

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(4),
            nn.GELU()
        )

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )
        self.up1 = UnetUp3(4 * n_feat, n_feat, self.co_dysample2)
        self.up2 = UnetUp3(2 * n_feat, n_feat,self.co_dysample1)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1)
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat) : time step
        c : (batch, n_classes) : context label
        """
        # pdb.set_trace()
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)

        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.concat((up3, x), 1))
        return out

class ContextUnet5(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_cfeat = 10, height = 16):
        super(ContextUnet5, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.co_pixelshuffle1 = co_pixelshuffle(n_feat)
        self.co_pixelshuffle2 = co_pixelshuffle(n_feat)
        self.down1 = UnetDown3(n_feat, n_feat, self.co_pixelshuffle1)       # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown3(n_feat, 2 * n_feat, self.co_pixelshuffle2)   # down2 #[10, 236, 4, 4]

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(4),
            nn.GELU()
        )

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )
        self.up1 = UnetUp3(4 * n_feat, n_feat, self.co_pixelshuffle2)
        self.up2 = UnetUp3(2 * n_feat, n_feat,self.co_pixelshuffle1)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1)
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat) : time step
        c : (batch, n_classes) : context label
        """
        # pdb.set_trace()
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)

        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.concat((up3, x), 1))
        return out
    
class ContextUnet6(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_cfeat = 10, height = 16):
        super(ContextUnet6, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.co_dysample_relu1 = Co_dysample_relu(n_feat)
        self.co_dysample_relu2 = Co_dysample_relu(n_feat)
        self.down1 = UnetDown3(n_feat, n_feat, self.co_dysample_relu1)       # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown3(n_feat, 2 * n_feat, self.co_dysample_relu2)   # down2 #[10, 236, 4, 4]

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(4),
            nn.GELU()
        )

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )
        self.up1 = UnetUp3(4 * n_feat, n_feat, self.co_dysample_relu2)
        self.up2 = UnetUp3(2 * n_feat, n_feat,self.co_dysample_relu1)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1)
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat) : time step
        c : (batch, n_classes) : context label
        """
        # pdb.set_trace()
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)

        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)
        
        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.concat((up3, x), 1))
        return out