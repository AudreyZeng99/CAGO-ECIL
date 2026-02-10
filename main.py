# main.py

import argparse
import yaml
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.model_factory import get_model
from datasets.data_loader import load_dataset
from datasets.transforms import get_transform
from ga.genetic_algorithm import GA
from utils.logging_utils import setup_logging, log_memory_usage
from utils.checkpoint import load_checkpoint
from utils.file_utils import create_exp_dir  # 导入 create_exp_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/cifar10_resnet18.yaml', help="Path to the config file.")
    args = parser.parse_args()

    # 打印当前工作目录
    print(f"Current working directory: {os.getcwd()}")

    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {script_dir}")

    # 构建配置文件的绝对路径
    config_path = os.path.join(script_dir, args.config)
    print(f"Loading config file from: {config_path}")

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"Error: Configuration file does not exist at {config_path}")
        return

    # 加载配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    prefix = config.get('pretrained', 'train')
    dataset = config.get('dataset', 'CIFAR-100')
    nepoch = config.get('nepoch', 10)
    learning_rate = config.get('learning_rate', 0.001)
    increment = config.get('increment', 5)
    model_name = config.get('model_name', 'resnet18')
    select_rate = config.get('select_rate', 0.2)
    population_size = config.get('population', 50)
    num_generations = config.get('generation', 40)
    mutation_rate = config.get('mutation_rate', 0.8)
    resume = config.get('resume', False)
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoint')
    seed = config.get('seed', 2023)

    # 设置随机种子以确保可复现性
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 加载数据集
    x_train, y_train, x_test, y_test = load_dataset(dataset, subset_ratio=1.0)
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # 定义数据转换
    transform = get_transform()

    # 创建实验目录，基于脚本所在的目录
    exp_dir = create_exp_dir(prefix, dataset, model_name, increment, nepoch, learning_rate, script_dir)

    # 设置日志记录
    setup_logging(exp_dir)

    # 设备选择逻辑：CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
    elif device == 'mps':
        print("MPS device is available and being used.")
    else:
        print("Using CPU.")

    # 创建模型
    if dataset == 'MNIST':
        input_channels = 1
        num_classes = 10

    elif dataset == 'CIFAR-10':
        input_channels = 3
        num_classes = 10
    elif dataset == 'CIFAR-100':
        input_channels = 3
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset")

    model = get_model(model_name, num_classes, input_channels=input_channels, pretrained=True, device=device)

    # 打印模型所在设备
    print(f"Model is on device: {next(model.parameters()).device}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 运行遗传算法，传递 device 参数
    GA(
        model=model,
        dataset_name=dataset,
        nepoch=nepoch,
        select_rate=select_rate,
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        increment=increment,
        resume=resume,
        learning_rate=learning_rate,
        args=args,
        exp_dir=exp_dir,
        criterion=criterion,
        optimizer=optimizer,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        transform=transform,
        device=device  # 传递 device
    )



if __name__ == "__main__":
    main()
