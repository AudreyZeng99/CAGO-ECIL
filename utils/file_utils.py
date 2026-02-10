# utils/file_utils.py

import os


def create_exp_dir(prefix, dataset_name, model_name, increment, nepoch, learning_rate, existing_dir):
    """
    创建实验目录结构。

    Args:
        prefix(str): 区别模型是否为预训练模型
        dataset_name (str): 数据集名称
        model_name (str): 模型名称
        increment (int): 每次增加的类别数
        nepoch (int): 每个训练阶段的epoch数
        learning_rate (float): 学习率
        existing_dir (str): 已存在的目录
    """
    # 使用 os.path.join 进行路径拼接，确保路径的正确性
    if prefix == "train":
        existing_dir_exp = os.path.join(existing_dir, "experiments/train")
    elif prefix == "pretrained":
        existing_dir_exp = os.path.join(existing_dir, "experiments/pretrained")

    exp_dir = os.path.join(existing_dir_exp,
                           f"{dataset_name}_{model_name}_inc{increment}_epochs{nepoch}_lr{learning_rate}")

    # 创建实验目录及其子目录，如果不存在
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)  # 使用 exist_ok=True 避免多次创建时报错
        os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
        print(f"Created experiment directory at: {exp_dir}")
    else:
        print(f"Experiment directory already exists at: {exp_dir}")

    return exp_dir


def ensure_directory_exists(directory):
    """
    确保指定的目录存在，如果不存在则创建。

    Args:
        directory (str): 需要确保存在的目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    print(f"Directory confirmed: {directory}")