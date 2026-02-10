# datasets/transforms.py

from torchvision import transforms
#
# def get_transform():
#     """
#     定义通用的数据转换。
#
#     Returns:
#         transform (transforms.Compose): 组合的转换函数
#     """
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((224, 224)),  # 根据模型需求调整大小
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 的均值和标准差
#                              std=[0.229, 0.224, 0.225])
#     ])
#     return transform
# datasets/transforms.py


def get_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 根据需要调整均值和标准差
    ])
