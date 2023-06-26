import torch
import torch.nn as nn

# Convolution + BatchNorm
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

# A single residual block for ResNeXt (no repititions)
class ResBlock(nn.Module):
    def __init__(self, in_channels, layer, first=False):
        super(ResBlock, self).__init__()

        self.projection = None
        stride = 1
        inter_channels = layer[1]
        out_channels = layer[2]
        cardinality = layer[3]

        if first:
            stride = 2
            self.projection = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, groups=1)

        self.cb1 = ConvBlock(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.cb2 = ConvBlock(in_channels=inter_channels, out_channels=inter_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        self.cb3 = ConvBlock(in_channels=inter_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.relu(self.cb1(x))
        x = self.relu(self.cb2(x))
        x = self.cb3(x)

        if self.projection is not None:
            identity = self.projection(identity)
        
        x += identity
        x = self.relu(x)
        return x