import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class TinyImageNetDataset(Dataset):
    """
    处理Tiny-ImageNet数据集的自定义Dataset。
    - root_dir: tiny-imagenet-200 数据集所在的根目录
    - split: 'train' / 'val' / 'test'
    - transform: 可选的图像增广或预处理
    """

    def __init__(self, root_dir, split='train', transform=None):
        super(TinyImageNetDataset, self).__init__()
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # 获取所有的类标签ID
        self.wnids = []
        with open(os.path.join(root_dir, 'wnids.txt'), 'r') as f:
            for line in f:
                self.wnids.append(line.strip())

        # 构建 label->index 映射表
        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        if self.split == 'train':
            self._load_train_data()
        elif self.split == 'val':
            self._load_val_data()
        elif self.split == 'test':
            # Tiny-ImageNet test 没有 label，这里仅作演示
            self._load_test_data()
        else:
            raise ValueError("split must be 'train', 'val' or 'test'.")

    def _load_train_data(self):
        # 读取 train/<wnid>/images 目录下所有图片
        train_dir = os.path.join(self.root_dir, 'train')
        for wnid in os.listdir(train_dir):
            wnid_folder = os.path.join(train_dir, wnid)
            if not os.path.isdir(wnid_folder):
                continue
            label = self.wnid_to_idx[wnid]
            images_folder = os.path.join(wnid_folder, 'images')
            for img_name in os.listdir(images_folder):
                if img_name.endswith('.JPEG') or img_name.endswith('.jpg') or img_name.endswith('.png'):
                    self.image_paths.append(os.path.join(images_folder, img_name))
                    self.labels.append(label)

    def _load_val_data(self):
        # val 数据需要先从 val_annotations.txt 获取图像到标签的映射
        val_dir = os.path.join(self.root_dir, 'val')
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')

        img_to_wnid = {}
        with open(annotations_file, 'r') as f:
            for line in f:
                line_split = line.strip().split('\t')
                filename, wnid = line_split[0], line_split[1]
                img_to_wnid[filename] = wnid

        images_folder = os.path.join(val_dir, 'images')
        for img_name in os.listdir(images_folder):
            if not img_name.endswith('.JPEG'):
                continue
            img_path = os.path.join(images_folder, img_name)
            wnid = img_to_wnid[img_name]
            label = self.wnid_to_idx[wnid]
            self.image_paths.append(img_path)
            self.labels.append(label)

    def _load_test_data(self):
        # test 集没有官方标签，这里仅加载图像
        test_dir = os.path.join(self.root_dir, 'test')
        images_folder = os.path.join(test_dir, 'images')
        for img_name in os.listdir(images_folder):
            if img_name.endswith('.JPEG'):
                img_path = os.path.join(images_folder, img_name)
                self.image_paths.append(img_path)
                # 测试集无 label，用 -1 代替
                self.labels.append(-1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 打开图像
        image = Image.open(img_path).convert('RGB')

        # 图像预处理/增广
        if self.transform:
            image = self.transform(image)

        return image, label


def get_tinyimagenet_dataloader(root_dir, split='train', transform=None, batch_size=64, shuffle=True, num_workers=4):
    """
    返回Tiny-ImageNet的DataLoader
    """
    dataset = TinyImageNetDataset(root_dir=root_dir, split=split, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, pin_memory=True)
    return loader
