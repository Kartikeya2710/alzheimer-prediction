import torch
import torch.nn as nn
from graphs.models.EfficientNet.stochasticdepth import StochasticDepth

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=0.99
        )
        self.silu = nn.SiLU() if activation else nn.Identity()
    
    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(
            in_features=in_channels,
            out_features=in_channels//reduction_ratio
        )
        self.fc2 = nn.Linear(
            in_features=in_channels//reduction_ratio,
            out_features=in_channels
        )
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor):
        identity = x
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x[:, :, None, None]
        weighted = x * identity
        return weighted

class MBConv(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size,
            stride, 
            padding, 
            expansion_ratio,
            reduction_ratio=24
        ):
        super(MBConv, self).__init__()

        # We will not use the residual connection in the first occurance of every block (where stride=2 is used and/or in_channels != out_channels) 
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        inter_channels = in_channels * expansion_ratio
        self.expand = (in_channels != inter_channels)
        # print(f"inter_channels = {inter_channels}, expansion_ratio = {expansion_ratio}")
        self.cb1 = ConvBlock(in_channels, inter_channels, 1, 1, 0) if self.expand else nn.Identity()
        self.cb2 = ConvBlock(inter_channels, inter_channels, kernel_size, stride, padding, groups=inter_channels)
        self.se = SEBlock(inter_channels, reduction_ratio)
        self.cb3 = ConvBlock(inter_channels, out_channels, 1, 1, 0)

        self.sd = StochasticDepth()

    def forward(self, x):
        f = self.cb1(x)
        f = self.cb2(f)
        f = self.se(f)
        f = self.cb3(f)

        if self.use_residual:
            f += x
        
        f = self.sd(f)
        
        return f

# if __name__ == "__main__":
#     x = torch.randn(1, 128, 224, 224)
#     block = MBConv(128, 32)
#     print(block(x).shape)