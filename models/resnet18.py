# models/resnet18.py

import torch.nn as nn
from torchvision import models


class ResNet18Custom(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, pretrained=False):
        super(ResNet18Custom, self).__init__()
        # 加载预训练的 ResNet18 模型
        self.model = models.resnet18(pretrained=pretrained)

        # 如果输入通道数不是3，调整第一层卷积
        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # 调整最后的全连接层以匹配类别数
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
