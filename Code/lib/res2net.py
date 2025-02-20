
import torch
import timm

import numpy as np

import timm

import torch

import torch.nn.functional as F

from torch import nn

from torch.utils.checkpoint import checkpoint

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
	channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.convatt0 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, bins):
        m = self.avg_pool(x)
        x0 = x
        y =  self.avg_pool(x0)
        y = self.convatt0(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        x0 = x0 * y.expand_as(x0)

        x = (x0) + x
        return x

class eca(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gba1 = eca_layer(1, c)
        self.gba2 = eca_layer(1, c)
        self.gba3 = eca_layer(1, c)
        self.gba4 = eca_layer(1, c)
        self.gba5 = eca_layer(1, c)
    def forward(self, feats, bin):
        x1 = self.gba1(feats[0], bin)
        x2 = self.gba2(feats[1], bin)
        x3 = self.gba3(feats[2], bin)
        x4 = self.gba4(feats[3], bin)
        x5 = self.gba5(feats[4], bin)
        return x1, x2, x3, x4, x5

class GBARES(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_encoder = timm.create_model(model_name="res2net50_26w_4s", pretrained=True, in_chans=3, features_only=True)
        #self.depth_encoder = timm.create_model(model_name="res2net50_26w_4s", pretrained=True, in_chans=1, features_only=True)
        #in_chans=3
        self.depth_encoder = timm.create_model(model_name="res2net50_26w_4s", pretrained=True, in_chans=3, features_only=True)
        ####################################
        self.gba_model_x = eca(3)
        self.gba_model_d = eca(3)

    def forward(self, x, d, bin):#no GBA
        rgb_feats = self.rgb_encoder(x)
        if bin.numel()!=0:
            print("error!no bin")
            exit()
        depth_feats = self.depth_encoder(d)
        if bin.numel()!=0:
            print("error!no bin")
            exit()
        return rgb_feats, depth_feats


# model=GBARES()
# p1=sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters: {p1} ")