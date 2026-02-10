<div align="center">

# CAGO-ECIL: Cloud-Assisted Genetic Optimization for Edge-Class Incremental Learning with training acceleration

[English](README.md) | [简体中文](README_zh.md)

</div>

## 论文介绍

本项目是论文 **"CAGO-ECIL: Cloud-Assisted Genetic Optimization for Edge-Class Incremental Learning with training acceleration"** 的官方实现。

该论文提出了一种云辅助的遗传优化方法，用于边缘类增量学习，旨在加速训练过程并保持模型性能。

## 项目简介

本项目实现了一种基于遗传算法的增量学习方法，支持ResNet系列模型，以及可支持多种轻量级模型（ResNet18、EfficientNet、MobileNetV2）和数据集（MNIST、CIFAR-10、CIFAR-100）。通过遗传算法优化模型的训练过程，实现对新类别的逐步学习，同时保持对旧类别的记忆。



## 目录结构

## 使用方法

### 环境设置

1. **创建虚拟环境（可选）**：

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **安装依赖**：

    ```bash
    pip install -r requirements.txt
    ```

### 配置

编辑 `config/config.yaml` 文件，设置所需的参数：

```yaml
dataset: CIFAR-100
nepoch: 10
learning_rate: 0.001
increment: 5
model_name: resnet18
select_rate: 0.2
population: 50
generation: 40
mutation_rate: 0.8
resume: false
checkpoint_dir: checkpoint
seed: 2023
```
如果想要快速尝试实验过程，运行项目之前，可以将config/config.yaml的population 和 generation参数修改为更小的值，比如`population=10`, `generation=2`，以进行快速开始。

### 运行项目

```bash
python main.py --config config/config.yaml
```

### 恢复训练

设置 `resume: true` 并指定 `checkpoint_dir`，以从上一个检查点恢复训练。

```yaml
resume: true
checkpoint_dir: checkpoint
```

然后运行：

```bash
python main.py --config config/config.yaml
```
## 引用

如果您觉得本项目对您的研究有帮助，请考虑引用我们的论文：

```bibtex
@article{ZENG2026108021,
 title = {CAGO-ECIL: Cloud-Assisted Genetic Optimization for Edge-Class Incremental Learning with training acceleration},
 journal = {Future Generation Computer Systems},
 volume = {174},
 pages = {108021},
 year = {2026},
 issn = {0167-739X},
 doi = {https://doi.org/10.1016/j.future.2025.108021},
 url = {https://www.sciencedirect.com/science/article/pii/S0167739X25003164},
 author = {Huayue Zeng and Wangbo Shen and Haijie Wu and Min Dong and Weiwei Lin and C.L. Philip Chen}
}
```