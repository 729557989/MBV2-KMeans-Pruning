import torch
import torch.nn as nn
import torch.nn.functional as F
from torchprofile import profile_macs


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(Bottleneck, self).__init__()
        
        # Depthwise convolution expansion ratio applied
        DW_num_channels = expansion_factor * in_channels
        # Whether to use residual structure or not
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        if expansion_factor == 1:
            self.conv = nn.Sequential(
                # Depthwise convolution, b/c groups parameter was specified
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(num_features=in_channels, momentum=0.1),
                nn.ReLU6(inplace=True),
                # 1x1 pointwise convolution
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.1),
            )
        else:
            self.conv = nn.Sequential(
                # 1x1 pointwise convolution
                nn.Conv2d(in_channels, DW_num_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=DW_num_channels, momentum=0.1),
                nn.ReLU6(inplace=True),
                # Depthwise convolution, b/c groups parameter was specified
                nn.Conv2d(DW_num_channels, DW_num_channels, kernel_size=3, stride=stride, padding=1, groups=DW_num_channels, bias=False),
                nn.BatchNorm2d(num_features=DW_num_channels, momentum=0.1),
                nn.ReLU6(inplace=True),
                # 1x1 pointwise convolution
                nn.Conv2d(DW_num_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.1),
            )
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
        
# Just make sure that predicts 200 classes
class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        
        self.input_dim = 224
        num_classes = 200
        bottleneck_setting = [
            # (expansion factor), (number of output channels), (repeat times), (stride)
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.network = []
        
        self.first_output_channel = 32
        
        # First Layer
        self.first_layer = nn.Sequential(
                # Depthwise convolution, b/c groups parameter was specified
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=self.first_output_channel),
                nn.ReLU6(inplace=True),
            )
        self.network.append(self.first_layer)
        
        # Bottleneck Layers
        for _, (t, c, n, s) in enumerate(bottleneck_setting):
            for i in range(n):
                # "The first layer of each sequence has a stride s and all others use stride 1."
                if i == 0:
                    bottleneck = Bottleneck(
                        in_channels = self.first_output_channel,
                        out_channels = c,
                        stride = s,
                        expansion_factor = t
                    )
                    self.first_output_channel = c
                else:
                    bottleneck = Bottleneck(
                        in_channels = self.first_output_channel,
                        out_channels = c,
                        stride = 1,
                        expansion_factor = t
                    )
                
                self.network.append(bottleneck)
        
        # pointwise convolution
        self.network.append(
            nn.Sequential(
                # Depthwise convolution, b/c groups parameter was specified
                nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=1280),
                nn.ReLU6(inplace=True),
            )
        )
        
        # 7x7 maxpooling
        self.network.append(
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        
        self.network = nn.Sequential(*self.network)
        
        # classification
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        assert x.size()[-1] == self.input_dim and x.size()[-2] == self.input_dim
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = MobileNetV2()
    model.eval()
    input = torch.randn(1, 3, 224, 224)
    macs = profile_macs(model, input)
    print("macs: ", macs) # MBV2 should have ~300M MACs
    out = model(input)
    print(out.shape)