from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def ensure_pil(data):
    if isinstance(data, np.ndarray):
        return Image.fromarray(data)
    elif torch.is_tensor(data):
        return transforms.ToPILImage()(data)
    else:
        raise TypeError("Unrecognized type for data conversion to PIL Image")

class CustomTrainDataset(Dataset):
    def __init__(self, x, y, input_ratios, transform=None):
        self.x = x
        self.y = y
        self.transform = transform or CustomTrainDataset.default_transform()
        self.input_ratios = input_ratios

        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = sum(counts)
        self.sample_indices = []

        for class_index in unique_classes:
            num_samples = int(total_samples * (input_ratios[class_index] / sum(input_ratios[:len(unique_classes)])))
            indices = np.where(y == class_index)[0]
            sampled_indices = np.random.choice(indices, num_samples, replace=True)
            self.sample_indices.extend(sampled_indices)

        np.random.shuffle(self.sample_indices)

    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        data = self.x[self.sample_indices[idx]]
        label = self.y[self.sample_indices[idx]]
        data = ensure_pil(data)  # Convert data to PIL Image correctly based on type
        data = self.transform(data)
        return data, label

class CustomTestDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform or CustomTestDataset.default_transform()

    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = self.x[idx]
        label = self.y[idx]
        data = ensure_pil(data)  # Convert data to PIL Image correctly based on type
        data = self.transform(data)
        return data, label


# # datasets/custom_dataset.py
#
# from torch.utils.data import Dataset
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# from PIL import Image
#
#
#
#
# class CustomTrainDataset(Dataset):
#     def __init__(self, x, y, input_ratios, transform=None):
#         self.x = x
#         self.y = y
#         self.transform = transform
#         self.input_ratios = input_ratios
#         print(f"input_ratios_temp in CustomDataset:\n{input_ratios}")
#
#         # Calculating the sample allocations per class based on input_ratios
#         unique_classes, counts = np.unique(y, return_counts=True)
#         total_samples = sum(counts)
#         self.sample_indices = []
#
#         for class_index in unique_classes:
#             num_samples = int(total_samples * (input_ratios[class_index] / sum(input_ratios[:len(unique_classes)])))
#             indices = np.where(y == class_index)[0]
#             # sampled_indices = np.random.choice(indices, num_samples, replace=False)
#             sampled_indices = np.random.choice(indices, num_samples, replace=True)
#             self.sample_indices.extend(sampled_indices)
#
#         # Shuffle the indices to ensure mixed batches
#         np.random.shuffle(self.sample_indices)
#
#     @staticmethod
#     def default_transform():
#         return transforms.Compose([
#             transforms.ToPILImage(),  # 转换为 PIL 图像
#             transforms.RandomHorizontalFlip(),  # 随机水平翻转
#             transforms.ToTensor(),  # 转换为张量
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
#         ])
#
#     def __len__(self):
#         return len(self.sample_indices)
#
#     def __getitem__(self, idx):
#         # Get the sample index
#         # sample_index = self.sample_indices[idx]
#         # data = self.x[sample_index]
#         # label = self.y[sample_index]
#         #
#         # if self.transform:
#         #     data = self.transform(data)
#         #
#         # return data, label
#         data = self.x[self.sample_indices[idx]]
#         label = self.y[self.sample_indices[idx]]
#
#         # 转换 numpy 数组为 PIL 图像
#         data = Image.fromarray(data)
#
#         if self.transform:
#             data = self.transform(data)
#
#         return data, label
#
#
#
# class CustomTestDataset(Dataset):
#     def __init__(self, x, y, transform=None):
#         self.x = x
#         self.y = y
#         self.transform = transform
#
#     @staticmethod
#     def default_transform():
#         return transforms.Compose([
#             transforms.ToPILImage(),  # 转换为 PIL 图像
#             transforms.ToTensor(),  # 转换为张量
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
#         ])
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         data = self.x[idx]
#         label = self.y[idx]
#
#         # 转换 numpy 数组为 PIL 图像
#         data = Image.fromarray(data)
#
#         if self.transform:
#             data = self.transform(data)
#
#         return data, label
