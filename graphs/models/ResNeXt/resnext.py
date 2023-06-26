import torch.nn as nn
from utils.dictionary import ConfigDict
from graphs.models.ResNeXt.residualblock import ConvBlock, ResBlock

class ResNeXt(nn.Module):
    def __init__(self, config: ConfigDict, model_version="50"):
        super(ResNeXt, self).__init__()
        self.layers = config.layers[model_version]
        self.in_channels = config.in_channels
        self.num_classes = config.num_classes
        self.cb1 = ConvBlock(self.in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = 64
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, self.num_classes)

        self._make_residual_layers()

    def _make_residual_layers(self):
        self.blocks = nn.ModuleList()
        for layer in self.layers:
            for idx in range(layer[0]):
                self.blocks.append(ResBlock(self.in_channels, layer, idx==0))
                self.in_channels = layer[2]

    def forward(self, x):
        x = self.relu(self.cb1(x))
        x = self.pool1(x)

        for block in self.blocks:
            x = block(x)

        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
# if __name__ == "__main__":
    # model = ResNeXt50(1, 4).to('cpu')
    # x = torch.randn(32, 1, 224, 224)
    # output = model(x)
    # print(output)
    # print(summary(model, (1, 224, 224), device='cpu'))