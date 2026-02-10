# datasets/data_loader.py

import numpy as np
from torchvision import datasets
from datasets.custom_dataset import CustomTrainDataset, CustomTestDataset
from .transforms import get_transform
import torch

def load_dataset(dataset_name, subset_ratio=1.0):
    """
    加载指定的数据集。

    Args:
        dataset_name (str): 数据集名称，例如 'MNIST', 'CIFAR-10', 'CIFAR-100'
        subset_ratio (float): 训练集和测试集的比例，默认为1.0（使用全部数据）

    Returns:
        tuple: (x_train, y_train, x_test, y_test)
    """
    if dataset_name == 'MNIST':
        from torchvision import datasets
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    elif dataset_name == 'CIFAR-10':
        from torchvision import datasets
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    elif dataset_name == 'CIFAR-100':
        from torchvision import datasets
        train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
        test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=None)
    elif dataset_name.lower() == "tinyimagenet":
        return TinyImageNetDataset(root_dir=dataset_path, split=split, transform=transform)
    # elif dataset_name.lower() == "xxx":
    #     return AnotherDataset(...)
    else:
        raise ValueError("Unsupported dataset")

    # 将数据转换为 NumPy 数组
    x_train = train_set.data
    y_train = np.array(train_set.targets)
    x_test = test_set.data
    y_test = np.array(test_set.targets)

    # 根据 subset_ratio 进行数据集的子集化
    if subset_ratio < 1.0:
        num_train = int(len(x_train) * subset_ratio)
        num_test = int(len(x_test) * subset_ratio)
        x_train = x_train[:num_train]
        y_train = y_train[:num_train]
        x_test = x_test[:num_test]
        y_test = y_test[:num_test]

    return x_train, y_train, x_test, y_test


