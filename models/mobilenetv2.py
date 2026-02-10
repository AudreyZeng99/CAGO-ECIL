# models/mobilenetv2.py

import torch.nn as nn
from torchvision import models


class MobileNetV2Custom(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, pretrained=False):
        super(MobileNetV2Custom, self).__init__()
        # 加载预训练的 MobileNetV2 模型
        self.model = models.mobilenet_v2(pretrained=pretrained)

        # 如果输入通道数不是3，调整第一层卷积
        if input_channels != 3:
            self.model.features[0][0] = nn.Conv2d(
                input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
            )

        # 调整分类器的最后一层以匹配类别数
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
