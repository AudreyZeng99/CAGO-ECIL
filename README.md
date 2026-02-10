<div align="center">

# CAGO-ECIL: Cloud-Assisted Genetic Optimization for Edge-Class Incremental Learning with training acceleration

[English](README.md) | [简体中文](README_zh.md)

</div>

## Paper Introduction

This project is the official implementation of the paper **"CAGO-ECIL: Cloud-Assisted Genetic Optimization for Edge-Class Incremental Learning with training acceleration"**.

This paper proposes a cloud-assisted genetic optimization method for edge-class incremental learning, aiming to accelerate the training process while maintaining model performance.

## Repo Introduction

This repo implements a genetic algorithm-based incremental learning method, supporting ResNet backboned models, also lightweight models (EfficientNet, MobileNet) and datasets (MNIST, CIFAR-10, CIFAR-100). By optimizing the model training process through genetic algorithms, it achieves incremental learning of new classes while retaining memory of old classes.

## Citation

If you find this project useful for your research, please consider citing our paper:

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

## Directory Structure

## Usage

### Environment Setup

1. **Create Virtual Environment (Optional)**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Edit the `config/config.yaml` file to set the required parameters:

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

### Run Project

```bash
python main.py --config config/config.yaml
```

### Resume Training

Set `resume: true` and specify `checkpoint_dir` to resume training from the last checkpoint.

```yaml
resume: true
checkpoint_dir: checkpoint
```

Then run:

```bash
python main.py --config config/config.yaml
```
