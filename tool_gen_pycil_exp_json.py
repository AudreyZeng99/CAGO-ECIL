import json
import itertools

# 定义字段列表
methods = ['pass', "icarl", "bic", "gem", "memo", "lwf", "ewc","aper"]
etrs = ["1", "5", "10"]
datasets = ["mnist", "cifar10", "cifar100"]
stages_10_classes = ["10", "5"]
stages_100_classes = ["20", "10", "5"]


# 定义一个函数，根据method生成不同的键值对
def generate_content(method):
    # dataset_name = dataset_name
    # generate file name
    if method == 'pass':
        return {
            "prefix": "ETR_1",
            "epochs" : 11,
            "dataset": "cifar100",
            "memory_size": 0,
            "shuffle": True,
            "init_cls": 10,
            "increment": 10,
            "model_name": "pass",
            "convnet_type": "resnet18_cbam",
            "device": ["0","1"],
            "seed": [1993],
            "lambda_fkd":10,
            "lambda_proto":10,
            "temp":0.1,
            "lr" : 0.001,
            "batch_size" : 64,
            "weight_decay" : 2e-4,
            "step_size":45,
            "gamma":0.1,
            "num_workers" : 8,
            "T" : 2
            }
    elif method == 'icarl':
        return {
            "prefix": "ETR_1",
            "nepochs": 10,
            "dataset": "mnist",
            "memory_size": 2000,
            "memory_per_class": 20,
            "fixed_memory": False,
            "shuffle": True,
            "init_cls": 1,
            "increment": 1,
            "model_name": "icarl",
            "convnet_type": "resnet18_cbam",
            "device": ["0","1"],
            "seed": [1993]
            }
    elif method == 'bic':
        return {
            "prefix": "ETR_1",
            "nepochs": 10,
            "dataset": "cifar100",
            "memory_size": 2000,
            "memory_per_class": 20,
            "fixed_memory": False,
            "shuffle": True,
            "init_cls": 10,
            "increment": 10,
            "model_name": "bic",
            "convnet_type": "resnet32",
            "device": ["0", "1"],
            "seed": [1993]
        }
    elif method == "gem":
        return {
            "prefix": "ETR_5",
            "nepochs": 50,
            "dataset": "cifar100",
            "memory_size": 2000,
            "memory_per_class": 20,
            "fixed_memory": False,
            "shuffle": True,
            "init_cls": 20,
            "increment": 20,
            "model_name": "gem",
            "convnet_type": "resnet32",
            "device": ["0", "1"],
            "seed": [1993]

        }
    elif method == "memo":
        return {
            "prefix": "ETR_5",
            "init_epoch": 50,
            "epochs": 50,
            "dataset": "cifar100",
            "init_cls": 20,
            "increment": 20,
            "memory_size": 2000,
            "memory_per_class": 20,
            "fixed_memory": False,
            "shuffle": True,
            "model_name": "memo",
            "convnet_type": "memo_resnet32",
            "train_base": True,
            "train_adaptive": False,
            "debug": False,
            "skip": False,
            "device": ["0", "1"],
            "seed": [1993],
            "scheduler": "steplr",
            "t_max": "null",
            "init_lr": 0.1,
            "init_weight_decay": 5e-4,
            "init_lr_decay": 0.1,
            "init_milestones": [60, 120, 170],
            "milestones": [80, 120, 150],
            "lrate": 0.1,
            "batch_size": 128,
            "weight_decay": 2e-4,
            "lrate_decay": 0.1,
            "alpha_aux": 1.0

        }
    elif method == "lwf":
        return {
            "prefix": "reproduce",
            "nepochs": 10,
            "dataset": "cifar10",
            "memory_size": 2000,
            "memory_per_class": 20,
            "fixed_memory": False,
            "shuffle": True,
            "init_cls": 2,
            "increment": 2,
            "model_name": "lwf",
            "convnet_type": "resnet32",
            "device": ["0", "1"],
            "seed": [1993]

        }
    elif method == "ewc":
        return {
            "prefix": "reproduce",
            "nepochs": 10,
            "dataset": "cifar10",
            "memory_size": 2000,
            "memory_per_class": 20,
            "fixed_memory": False,
            "shuffle": True,
            "init_cls": 2,
            "increment": 2,
            "model_name": "ewc",
            "convnet_type": "resnet32",
            "device": ["0", "1"],
            "seed": [1993]

        }
    else: # method == "aper"
        return {
            "prefix": "reproduce",
            "dataset": "cifar100",
            "memory_size": 2000,
            "memory_per_class": 20,
            "fixed_memory": False,
            "shuffle": True,
            "init_cls": 10,
            "increment": 10,
            "model_name": "aper_finetune",
            "convnet_type": "cosine_resnet32",
            "device": ["0"],
            "trained_epoch": 200,
            "tuned_epoch": 20,
            "optimizer": "sgd",
            "init_weight_decay": 0.05,
            "weight_decay": 0.05,
            "finetune_lr": 0.005,
            "seed": [1993]
        }


# 整理好所有要生成的filename
def get_combinations():
    combinations = []  # 存放所有可能的组合
    # generate content

    combinations_1 = list(itertools.product(methods, "mnist", etrs, stages_10_classes))
    combinations_2 = list(itertools.product(methods, "cifar10", etrs, stages_10_classes))
    combinations_3 = list(itertools.product(methods, "cifar100", etrs, stages_100_classes))
    combinations.append(combinations_1)
    combinations.append(combinations_2)
    combinations.append(combinations_3)

    return combinations



# For different method, the adding contents are different.
def add_content(method, dataset, etr, stage, content):
    #先改统一的内容
    # ETR
    if etr == "1":
        content["prefix"] = "ETR_1"
        nepochs
    elif etr =="5":
        content["prefix"] = "ETR_5"
    elif etr == "10":
        content["prefix"] = "ETR_10"

    # Stage
    if dataset =="cifar100":
        if stage == "20":
            content["init_cls"] = 5
            content["increment"]  = 5
        elif stage == "10" :
            content["init_cls"] = 10
            content["increment"] = 10
        elif stage =="5":
            content["init_cls"] = 20
            content["increment"] = 20
    else:# datset == mnsit or cifar10
        if stage == "5":
            content["init_cls"] = 2
            content["increment"] = 2
        else:
            content["init_cls"] = 1
            content["increment"] = 1


    # nepochs
    if method =="aper":
        content["nepochs"] = 10

    elif method == "pass":



if __name__=="__main__":
    # for method in methods:
    #     for dataset_name in datasets:
    #         combination = generate_content(method, dataset_name)
    #         method, etr, stage = combination
    # 遍历所有组合，生成JSON文件

    combinations = get_combinations()
    for combination in combinations:
        method, dataset, etr, stage = combination
        content = generate_content(method)

        # 根据etr和stage添加额外的键值对
        add_content(method, dataset, etr, stage, content)
        content["etr"] = etr
        content["stage"] = stage

        # 构建文件名
        filename = f"{method}_{etr}_{stage}.json"

        # 写入JSON文件
        with open(filename, 'w') as json_file:
            json.dump(content, json_file, indent=4)

        print(f"Generated {filename} with content:")
        print(json.dumps(content, indent=4))