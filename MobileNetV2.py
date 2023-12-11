import torch
import timeit
import torch
import torch.nn as nn
from torchprofile import profile_macs
from helpers import visualize_weights

"""
# Optimizer
optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.008, weight_decay=0.0001)
# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=16, gamma=0.3)

前两个block不prune，后面的层只prune depthwise的卷积层（做kmeabs），
不prune pointwise的卷积层(除了修改input dimension)，就算调整 pointwise卷积层也只是为了不出现dimension mismatch的问题

只针对 prune output 的层做 KMeans Clustering，也就是 depthwise convolutional layer, pointwise只prune input dimension

play with: torch.optim.lr_scheduler.OneCycleLR

Best Epoch: 76
    Best Train Loss: 0.0408 Acc: 0.9936
    Best Test Loss: 0.5729 Acc: 0.8680
    
    
    No, in a depthwise separable convolutional layer with group convolution, each filter F_i is designed to convolve only with its corresponding input channel I_i. 
"""

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
                bottleneck = Bottleneck(
                    in_channels = self.first_output_channel,
                    out_channels = c,
                    stride = s if i == 0 else 1,
                    expansion_factor = t
                )
                self.first_output_channel = c
                
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert x.size()[-1] == self.input_dim and x.size()[-2] == self.input_dim
        x = self.network(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def profile_test(self, num_iterations=150):
        self.eval()
        input_tensor = torch.randn(1, 3, 224, 224)

        # Warm-up the model (optional but recommended)
        with torch.no_grad():
            _ = self.forward(input_tensor)

        def inference_time():
            with torch.no_grad():
                self.forward(input_tensor)

        # Measure inference time and calculate average
        total_time = timeit.timeit(inference_time, number=num_iterations)
        average_time = total_time / num_iterations * 1000  # Convert to milliseconds

        print(f"Average Inference Time: {round(average_time, 4)} ms")
        
        macs = profile_macs(self, input_tensor)
        m_macs = macs / 1000000
        print("Million Macs: ", m_macs)
        
if __name__ == '__main__':
    model = MobileNetV2()
    model.profile_test()

    visualize_weights(model)