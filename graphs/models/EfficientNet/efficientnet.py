import torch
import torch.nn as nn
from graphs.models.EfficientNet.mbconv import MBConv, ConvBlock
from math import ceil

class EfficientNet(nn.Module):
    def __init__(self, config, version="B0"):
        super(EfficientNet, self).__init__()
        phi, resolution, p = config.phis[version]
        self.conv_layers = nn.ModuleList()
        self.in_channels = config.in_channels
        self.layers = config.layers
        self.num_classes = config.num_classes
        self._calculate_coef(phi)

        # First Conv layer
        self._add_layer(self.layers[0])

        # Layer 2 to 8
        for idx in range(1, len(self.layers)-1):
            # layers[idx].append(reduction_factor)
            self._add_layer(self.layers[idx])

        # Last Conv Layer
        self._add_layer(self.layers[-1])

        # Avg pooling and FC layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.dropout = nn.Dropout(p=p)


    def _add_layer(self, layer):
        # *args will only have the optional expansion_ratio for the MBConv block or nothing for the ConvBlock
        block_type, out_channels, kernel_size, stride, occurances, *args = layer
        # Scaling up for different models using the compound scaling method
        occurances, out_channels = self._update_dim(occurances, out_channels)
        padding = kernel_size // 2
        block = globals()[block_type](self.in_channels, out_channels, kernel_size, stride, padding, *args)
        self.in_channels = out_channels

        self.conv_layers.append(block)

        for i in range(occurances-1):
            block = globals()[block_type](self.in_channels, out_channels, kernel_size, 1, padding, *args)
            # No need to update self.in_channels because they will remain the same for this layer
            self.conv_layers.append(block)


    def _calculate_coef(self, phi, alpha=1.2, beta=1.1):
        self.d_scale = alpha ** phi
        self.w_scale = beta ** phi

    def _update_dim(self, occurances, out_channels):
        return int(occurances * self.d_scale), int(out_channels * self.w_scale)


    def forward(self, x):
        for idx, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# if __name__ == "__main__":
#     x = torch.randn(1, 3, 260, 260)
#     version = "B1"
#     _, res, _ = phis[version]
#     model = EfficientNet_B1(3, 1000).to("cpu")
#     model.eval()
#     print(model)
#     print(summary(model, (3, res, res), device="cpu"))