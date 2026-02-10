# models/model_factory.py

import torch.nn as nn
from torchvision import models
from models.resnet32 import ResNet32Custom
from models.resnet18 import ResNet18Custom
from models.efficientnet_b0_1 import EfficientNetB0
from models.mobilenetv2 import MobileNetV2Custom
from models.simple_network import MyMLP

def get_model(model_name, num_classes, input_channels=3, pretrained=False, device='cpu'):
    """
    根据模型名称返回相应的模型实例，并调整最后的分类层。

    Args:
        model_name (str): 模型名称，如 'resnet18', 'resnet32', 'efficientnet_b0', 'mobilenet_v2'
        num_classes (int): 类别数
        input_channels (int): 输入通道数
        pretrained (bool): 是否使用预训练权重
        device (str): 设备类型

    Returns:
        model (nn.Module): 调整后的模型
    """
    if model_name.lower() == 'resnet18':
        model = ResNet18Custom(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)

    elif model_name.lower() == 'mlp':
        model = MyMLP(input_channels=input_channels, num_classes=num_classes)
    elif model_name.lower() == 'resnet32':
        model = ResNet32Custom(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)

    elif model_name.lower() == 'efficientnet_b0':
        model = EfficientNetB0(num_classes=num_classes)

    elif model_name.lower() == 'mobilenet_v2':
        model = MobileNetV2Custom(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model.to(device)


def get_model_pretrained(model_name, num_classes, input_channels=3, pretrained=True, device='cpu'):
    """
    根据模型名称返回相应的模型实例，并调整最后的分类层。

    Args:
        model_name (str): 模型名称，如 'resnet18', 'resnet32', 'efficientnet_b0', 'mobilenet_v2'
        num_classes (int): 类别数
        input_channels (int): 输入通道数
        pretrained (bool): 是否使用预训练权重
        device (str): 设备类型

    Returns:
        model (nn.Module): 调整后的模型
    """
    if model_name.lower() == 'resnet18':
        model = ResNet18Custom(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)

    elif model_name.lower() == 'resnet32':
        model = ResNet32Custom(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)

    elif model_name.lower() == 'efficientnet_b0':
        model = EfficientNetB0Custom(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)

    elif model_name.lower() == 'mobilenet_v2':
        model = MobileNetV2Custom(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model.to(device)