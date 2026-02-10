import torch
import torch.nn as nn
import torch.nn.functional as F


# Squeeze-and-Excitation模块
class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeAndExcitation, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


# 深度可分离卷积块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(DepthwiseSeparableConv, self).__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * expansion, in_channels * expansion, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels * expansion, bias=False),
            nn.BatchNorm2d(in_channels * expansion),
            nn.ReLU(inplace=True)
        )
        self.se = SqueezeAndExcitation(in_channels * expansion)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.skip:
            return self.pointwise(self.se(self.depthwise(self.expand(x)))) + x
        else:
            return self.pointwise(self.se(self.depthwise(self.expand(x))))


# EfficientNet-B0
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(32, 16, stride=1, expansion=1),
            DepthwiseSeparableConv(16, 24, stride=2, expansion=6),
            DepthwiseSeparableConv(24, 40, stride=2, expansion=6),
            DepthwiseSeparableConv(40, 80, stride=2, expansion=6),
            DepthwiseSeparableConv(80, 112, stride=1, expansion=6),
            DepthwiseSeparableConv(112, 192, stride=2, expansion=6),
            DepthwiseSeparableConv(192, 320, stride=1, expansion=6)
        )
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Example usage
if __name__ == "__main__":
    model = EfficientNetB0(num_classes=100)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.size())