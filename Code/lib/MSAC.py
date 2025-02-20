import torch
import torch.nn as nn


class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,padding=kernel_size // 2, groups=in_channels)
        #self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.dw_large = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,padding=kernel_size // 2, groups=in_channels),
            # 空洞卷积
            nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3),
            nn.GroupNorm(num_groups=8, num_channels=in_channels)   
        )
        self.dw_small = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,padding=1, groups=in_channels),
            # 空洞卷积
            #nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3),
            nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3),

            nn.GroupNorm(num_groups=8, num_channels=in_channels)           
        )
        self.dw_max = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(in_channels, in_channels, 1 ),

            nn.GroupNorm(num_groups=8, num_channels=in_channels)           
        )

        #self.bn = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_large = self.dw_large(x)
        x_small = self.dw_small(x)
        ########################
        x_max = self.dw_max(x)
        return self.relu(x_large + x_small +x_max)


    def initialize(self):
        weight_init(self)

class MSACBlock(nn.Module):
    def __init__(self, channels, kernel_size):#Cs2=64, K1=9
        super(MSACBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.depth_conv = DepthwiseConvBlock(channels, kernel_size)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)#1*1
        x = self.depth_conv(x)
        x = self.conv2(x)#1*1
        x = self.bn(x+residual)###跳跃链接
        x = self.relu(x)
        return x
    def initialize(self):
        weight_init(self)