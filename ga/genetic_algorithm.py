import os
import json
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logging_utils import log_memory_usage
from models.model_factory import get_model
from datasets.data_loader import load_dataset
from datasets.transforms import get_transform
from ga.population import create_population, select_top_individuals, regenerate_population
from utils.file_utils import create_exp_dir
from datasets.custom_dataset import CustomTrainDataset, CustomTestDataset
from torch.utils.data import DataLoader
from utils.file_utils import ensure_directory_exists
from ga.utils import calculate_valleys_and_recovery
from ga.utils import fitness_function

def GA(model, dataset_name, nepoch, select_rate, population_size, num_generations, mutation_rate, increment, resume, learning_rate, args, exp_dir, criterion, optimizer, x_train, y_train, x_test, y_test, transform, device):
    """
    遗传算法主函数。
    """
    if dataset_name in ['MNIST', 'CIFAR-10']:
        num_classes = 10
    elif dataset_name == 'CIFAR-100':
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset")

    if resume:
        loaded_data = load_checkpoint(checkpoint_dir=os.path.join(exp_dir, 'checkpoint'), model=model)
        if loaded_data and loaded_data[1] is not None:
            model, population, fitness_scores, start_generation, best_fitness = loaded_data
            print(f"Resumed from generation {start_generation}.")
        else:
            print("Unable to load checkpoint, starting a new training session.")
            population = create_population(population_size, num_classes, increment)
            fitness_scores = []
            start_generation = 0
            best_fitness = []
    else:
        population = create_population(population_size, num_classes, increment)
        fitness_scores = []
        start_generation = 0
        best_fitness = []

    bestI_by_fitness = []
    bestI_by_aa = []
    bestI_by_fm = []
    best_fitness_list = []
    best_aa_list = []
    best_fm_list = []

    for i in range(start_generation, num_generations):
        generation_number = i + 1
        print(f"Generation # {generation_number} is starting...")
        fitness_scores_list, max_fitness, fit_index, acc_up2now_scores_list, max_aa, aa_index, fm_scores_list, min_fm , fm_index= get_fitness_list_of_generation(
            population, model, num_classes, nepoch, increment, generation_number, criterion, optimizer, x_train, y_train, x_test, y_test, transform, device, learning_rate,exp_dir  # 修改这里，传递 device
        )
        best_individual, top_individuals, selected_indices = select_top_individuals(population, "Fitness", fitness_scores_list, select_rate)

        remaining_population = [ind for idx, ind in enumerate(population) if idx not in selected_indices]
        new_population = regenerate_population(top_individuals, population_size, remaining_population, mutation_rate, num_classes, increment)
        population = new_population

        best_individual_aa = select_top_individuals(population, "AA", acc_up2now_scores_list, select_rate)
        best_individual_fm = select_top_individuals(population, "FM", fm_scores_list, select_rate)

        bestI_by_fitness.append(best_individual)
        bestI_by_aa.append(best_individual_aa)
        bestI_by_fm.append(best_individual_fm)

        best_fitness_list.append(max_fitness)
        best_aa_list.append(max_aa)
        best_fm_list.append(min_fm)
        save_checkpoint(model, population, fitness_scores_list, best_fitness_list, best_aa_list, best_fm_list, generation_number, exp_dir)

        #### 测试代码
        print(f"Generation # {generation_number}:")
        print("Selected indices:", selected_indices, "#", len(selected_indices))
        print("len of top individuals:", len(top_individuals))
        print("Remaining pop size:", len(remaining_population))
        print("New population size:", len(new_population))

        # 清理会话和内存
        clear_session()
        log_memory_usage()

    best_individual_list_dir = os.path.join(exp_dir, 'results', f'best_I{fit_index}_fitness_gen{generation_number}.json')
    with open(best_individual_list_dir, 'w') as file:
        json.dump(bestI_by_fitness, file, default=default_converter)

    best_individual_aa_list_dir = os.path.join(exp_dir, 'results', f'best_I{aa_index}_aa_gen{generation_number}.json')
    with open(best_individual_aa_list_dir, 'w') as file:
        json.dump(bestI_by_aa, file, default=default_converter)

    best_individual_fm_list_dir = os.path.join(exp_dir, 'results', f'best_I{fm_index}_fm_gen{generation_number}.json')
    with open(best_individual_fm_list_dir, 'w') as file:
        json.dump(bestI_by_fm, file, default=default_converter)

    print("GA execution completed.")

def get_fitness_list_of_generation(population, model, num_classes, nepoch, increment, current_generation, criterion, optimizer, x_train, y_train, x_test, y_test, transform, device, learning_rate,exp_dir):
    fit_score = []
    acc_up2now_score = []
    fm_score = []

    for i in range(len(population)):

        # 计算个体的适应度
        fitness, acc_up2now, FM = calculate_individual_fitness(x_train, y_train, x_test, y_test, model, population[i], current_generation, i,
                                         num_classes, nepoch, increment, criterion, optimizer, transform, device,
                                         learning_rate,exp_dir)
        fit_score.append(fitness)
        acc_up2now_score.append(acc_up2now)
        fm_score.append(FM)


    # 转换为 NumPy 数组并计算最大适应度
    fit_score_array = np.array(fit_score)
    max_fit = np.max(fit_score_array)
    max_fit_index = np.argmax(fit_score_array)

    acc_up2now_array = np.array(acc_up2now_score)
    max_acc_up2now = np.max(acc_up2now_array)
    max_aa_index = np.argmax(acc_up2now_array)

    fm_array = np.array(fm_score)
    min_fm = np.min(fm_array)
    min_fm_index = np.argmin(fm_array)

    return fit_score_array, max_fit, max_fit_index, acc_up2now_array, max_acc_up2now, max_aa_index, fm_array, min_fm,min_fm_index


def select_samples(x_train, y_train, old_classes, new_classes, min_per_old_class=20, total_old=2000):
    selected_indices = []
    total_selected = 0

    if old_classes:
        per_class_allocation = max(min_per_old_class, total_old // max(1, len(old_classes)))
        for class_id in old_classes:
            class_indices = np.where(y_train == class_id)[0]
            np.random.shuffle(class_indices)
            alloc_num = min(len(class_indices), per_class_allocation)
            selected_indices.extend(class_indices[:alloc_num])
            total_selected += alloc_num

        while total_selected < total_old:
            added_any = False
            for class_id in old_classes:
                if total_selected >= total_old:
                    break
                class_indices = np.where(y_train == class_id)[0]
                np.random.shuffle(class_indices)
                additional_indices = [idx for idx in class_indices if idx not in selected_indices][:1]
                if additional_indices:
                    selected_indices.extend(additional_indices)
                    total_selected += len(additional_indices)
                    added_any = True
            if not added_any:
                print("Warning: Not enough samples to reach total_old.")
                break

    for class_id in new_classes:
        class_indices = np.where(y_train == class_id)[0]
        selected_indices.extend(class_indices)

    return x_train[selected_indices], y_train[selected_indices]


# ------------------------------
# 假设性辅助函数、类（请根据项目实际情况进行替换）
# ------------------------------

def ensure_directory_exists(path):
    """
    如果目录不存在，则创建目录。
    """
    if not os.path.exists(path):
        os.makedirs(path)



def test_on_classes(model, x_test, y_test, classes, transform, device):
    """
    用于对指定的类（classes）进行测试，并返回Top-1 accuracy。
    """
    model.eval()
    indices = np.isin(y_test, classes)
    x_test_part = x_test[indices]
    y_test_part = y_test[indices]
    dataset_part = CustomTestDataset(x_test_part, y_test_part, transform=transform)
    loader_part = DataLoader(dataset_part, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader_part:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0
    return correct / total


# --------------------------------
# 1) 增量训练封装：individual_incremental_learning
# --------------------------------
def individual_incremental_learning(
        x_train, y_train,
        x_test, y_test,
        model,
        input_ratios,
        num_classes,
        nepoch,
        increment,
        criterion,
        optimizer,
        transform,
        device
):
    """
    该函数包含增量学习的核心逻辑：
    - 按照 context（增量阶段）进行迭代；
    - 对当前增量的类进行训练和测试；
    - 记录训练过程中的中间指标（loss、accuracy 等）；
    - 返回记录在 history 中的各种指标，以及辅助用于后续计算 Forgetting 的变量。

    参数：
    --------
    x_train, y_train, x_test, y_test : 训练/测试数据
    model : PyTorch 模型
    input_ratios : 每个 context 的输入比例
    num_classes : 数据集的总类别数
    nepoch : 当前 context 的训练轮数
    increment : 增量步长，每个 context 要学习多少类
    criterion : 损失函数
    optimizer : 优化器
    transform : 数据增强或预处理
    device : cuda 或 cpu

    返回：
    --------
    history : dict, 记录了训练过程中的各种指标
    best_up2now_acc_list : list, 用于后续计算 Forgetting 的最佳历史准确率
    final_up2now_acc_list : list, 用于后续计算 Forgetting 的最终准确率
    new_context_test_accuracy_list : list, 每个 context 完成后，对“当前新类”的测试准确率（a_k_k）
    """
    import torch
    from torch.utils.data import DataLoader

    context_num = int(num_classes // increment)

    # 初始化一个history字典，保存中间训练结果
    history = {
        'accuracy': [],  # 每个epoch的训练准确率(针对当前context的训练集)
        'loss': [],  # 每个epoch的训练损失(针对当前context的训练集)
        'average_accuracy': [],  # 每学完一个context后，对已学类的平均准确率
        'acc_up2now': [],  # 对[0, i+increment) 这些类的测试准确率
        'top1_accuracy': [],  # 当前context新类（[i, i+increment)）的top1准确率
        'avg_acc_by_mean': [],  # 逐context累积的平均值
        'FM':[],
        'BWT':[]
    }

    # 用于计算Forgetting的辅助列表
    previous_accuracies = []  # 记录各context新类的top1准确率, 例如 top1_accuracy
    final_up2now_acc_list = []  # 存储最后一个context时，所有旧context的accuracy
    best_up2now_acc_list = [0] * context_num  # 每个context的历史最佳准确率（针对旧context）

    # 用于后续计算 BWT 的辅助：保存每个context在其新类上的测试准确率 a_k_k
    new_context_test_accuracy_list = []

    past_test_past_list = []  # 会随着增量学习动态变长，并且需要找最大past_test_max_a_ij(i<k,j<k)
    past_test_max_a_ij = 0

    FM = 0
    BWT = 0

    # ==============
    # 开始增量训练
    # ==============
    for k, i in enumerate(range(0, num_classes, increment)):
        # ===========
        # 训练数据准备
        # input: new_classes, old_classes, input_ratios_temp
        # output: train_loader_current
        # ===========

        # 当前context所含的新类
        new_classes = list(range(i, min(i + increment, num_classes)))
        # 当前context之前的所有旧类
        old_classes = list(range(i))

        print("New classes list:", new_classes)
        print("Old classes list:", old_classes)

        # 从原始训练集里选择当前context要学习的样本
        x_train_current, y_train_current = select_samples(x_train, y_train, old_classes, new_classes)
        print(f"Shape of x_train_current: {x_train_current.shape}")
        print(f"Shape of y_train_current: {y_train_current.shape}")

        # 获取输入比例（若需要）
        input_ratios_temp = input_ratios[k]
        print("input_ratios_temp:\n", input_ratios_temp)

        # 构建当前context的DataLoader
        # train_dataset_current = CustomDataset(x_train_current, y_train_current, transform=transform)
        train_dataset_current = CustomTrainDataset(x_train_current, y_train_current, input_ratios=input_ratios_temp,
                                              transform=transform)
        train_loader_current = DataLoader(train_dataset_current, batch_size=64, shuffle=True, num_workers=0,
                                          pin_memory=True)
        """
        具体来说，train_loader_current 是一个迭代器，它每次迭代返回一个数据批次。每个批次包括一对 inputs 和 labels：
        
        inputs：这是一个包含批量输入数据的张量（Tensor）。在这里，它是来自 CustomTrainDataset 的 x_train_current 的一个子集，已经通过任何定义的变换（如图像预处理）转换为适合模型输入的格式。
        labels：这是一个包含批量标签数据的张量。它对应于输入数据的真实标签，也是从 CustomTrainDataset 的 y_train_current 的一个子集。

        """


        # ===========
        # 训练阶段
        # ===========
        for epoch in range(nepoch):
            running_loss = 0.0
            correct = 0
            total = 0
            model.train()

            for inputs, labels in train_loader_current:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total if total > 0 else 0.0
            epoch_acc = correct / total if total > 0 else 0.0
            print(f"Increment {k}, Epoch {epoch + 1}/{nepoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            # 保存每个 epoch 的指标（仅针对训练集）
            history['accuracy'].append(epoch_acc)
            history['loss'].append(epoch_loss)

        # ===========
        # 测试阶段(当前context的新类 top-1 accuracy)
        # ===========

        model.eval()

        # 当前context的新类测试集
        x_test_current = x_test[np.isin(y_test, np.arange(i, i + increment))]
        y_test_current = y_test[np.isin(y_test, np.arange(i, i + increment))]

        test_dataset_current = CustomTestDataset(x_test_current, y_test_current, transform=transform)
        test_loader_current = DataLoader(test_dataset_current, batch_size=64, shuffle=False, num_workers=0,
                                         pin_memory=True)
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader_current:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, test_pred = torch.max(outputs, 1)
                test_correct += (test_pred == labels).sum().item()
                test_total += labels.size(0)
        top1_accuracy = test_correct / test_total if test_total > 0 else 0.0

        # 记录当前context新类的 top1 accuracy
        history['top1_accuracy'].append(top1_accuracy)
        previous_accuracies.append(top1_accuracy)

        # 逐context的平均值
        avg_acc_by_mean = np.mean(previous_accuracies) if len(previous_accuracies) > 0 else 0.0
        history["avg_acc_by_mean"].append(avg_acc_by_mean)

        # up2now accuracy (测试所有已学类[0 ~ i+increment))
        acc_up2now = test_on_classes(model, x_test, y_test, list(range(0, i + increment)), transform, device)
        history['acc_up2now'].append(acc_up2now)

        # =============
        # 评估旧类表现
        # =============
        average_accuracy_list = []
        ####
        # 收集当前context测试每个旧类的准确率
        current_test_on_each_old = []


        if not old_classes:
            print("No old classes provided.")
        else:

            cur_on_past_min = 1
            fm_list = []
            bwt_left = 0
            bwt_right = 0
            # increment 步长每次加上 increment，再对 old_classes 进行测试
            for idx, k2 in enumerate(range(0, len(old_classes), increment)):
                # 对过去的每个context所学习的类进行测试
                seen_classes = list(range(k2, k2 + increment))
                print(f"idx:{idx}, k2:{k2}")
                print(f"seen classes:{seen_classes}")
                test_accuracy = test_on_classes(model, x_test, y_test, seen_classes, transform, device) #在前面某个context测试
                current_test_on_each_old.append(test_accuracy)
                if test_accuracy < cur_on_past_min:
                    cur_on_past_min = test_accuracy

                if k == context_num - 1:
                    final_up2now_acc_list.append(test_accuracy) # 用来求平均准确率的！
                else:
                    if test_accuracy > best_up2now_acc_list[idx]:
                        print("Best_up2now_acc_list(Updated!) idx={}: {:.4f} -> {:.4f}".format(
                            idx, best_up2now_acc_list[idx], test_accuracy
                        ))
                        best_up2now_acc_list[idx] = test_accuracy

                # 1. 计算当前的fm_temp
                fm_temp = past_test_max_a_ij - cur_on_past_min
                fm_list.append(fm_temp)

                # 2. 计算当前的bwt_tmp
                test_accuracy_new = test_on_classes(model, x_test, y_test, new_classes, transform, device)
                bwt_right += test_accuracy_new
                bwt_left += test_accuracy

                average_accuracy_list.append(test_accuracy)


            # 计算FM
            FM = np.mean(fm_temp)
            history['FM'].append(FM)
            print(f"Context {k} has FM_k = {FM}")

            # 计算BWT
            BWT = np.mean(bwt_left - bwt_right)
            history['BWT'].append(BWT)
            print(f"Context {k} has BWT_k = {BWT}")
        # =============
        # 评估新类表现
        # =============
        if not new_classes:
            print("No new classes provided.")
        else:
            test_accuracy_new = test_on_classes(model, x_test, y_test, new_classes, transform, device)
            average_accuracy_list.append(test_accuracy_new)

        past_test_past_list.extend(current_test_on_each_old)
        past_test_past_list.append(test_accuracy_new)
        past_test_max_a_ij = max(past_test_past_list)

        # 计算平均accuracy
        average_accuracy = np.mean(average_accuracy_list) if len(average_accuracy_list) > 0 else 0.0
        history['average_accuracy'].append(average_accuracy)

        # 用于后续 BWT 计算：记录当前模型 h(k) 在它自己的新类上的 accuracy (a_k_k)
        new_context_test_accuracy_list.append(top1_accuracy)

        # 打印中间结果
        print(f"Context {k}\n")
        print(f"Top-1 Accuracy on current new classes: {top1_accuracy}")
        print(f"Average Accuracy on old classes: {average_accuracy}")
        print(f"Acc up2now: {acc_up2now}")
        print(f"Avg Acc by Mean: {avg_acc_by_mean}")
        print(f"FM:{FM}")
        print(f"BWT:{BWT}")


        # 释放不需要的变量，节省显存和内存
        del x_train_current, y_train_current
        gc.collect()

    return history, best_up2now_acc_list, final_up2now_acc_list, new_context_test_accuracy_list


# --------------------------------
# 2) 判断模型是否收敛的示例逻辑
# --------------------------------
def check_convergence(accuracy_list, last_n=3, threshold=0.001):
    """
    简易示例逻辑：判断模型是否收敛。
    - 取最后 last_n 个 epoch 的准确率，如果两两之差均小于 threshold，则认为已经收敛。

    参数:
    -----
    accuracy_list : List[float]
        训练过程中记录下的 accuracy 序列
    last_n : int
        用于判断收敛时，取末尾多少个 epoch
    threshold : float
        收敛阈值

    返回:
    -----
    bool : 是否收敛
    """
    if len(accuracy_list) < last_n:
        return False  # epoch 不够多，暂时认为未收敛

    # 取最后 last_n 个 accuracy
    recent_acc = accuracy_list[-last_n:]

    # 两两比较，若差值绝对值都小于 threshold，则收敛
    for i in range(len(recent_acc) - 1):
        if abs(recent_acc[i + 1] - recent_acc[i]) > threshold:
            return False
    return True


# --------------------------------
# 3) 计算FM和BWT + 绘图 + 判断收敛
# --------------------------------
def calculate_individual_fitness(
        x_train, y_train,
        x_test, y_test,
        model,
        input_ratios,
        generta,
        individual_id,
        num_classes,
        nepoch,
        increment,
        criterion,
        optimizer,
        transform,
        device,
        learning_rate,
        exp_dir
):
    """
    主函数：负责
    1. 调用 individual_incremental_learning 做增量学习
    2. 计算 average_forgetting, FM, BWT 等指标
    3. 额外：绘制增量学习收敛曲线、判断是否收敛
    4. 保存结果到文件
    5. 返回 fitness_score
    """
    import torch
    import gc
    from torch.utils.data import DataLoader

    # 1) 调用封装好的增量学习
    history, best_up2now_acc_list, final_up2now_acc_list, new_context_test_accuracy_list = \
        individual_incremental_learning(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            model=model,
            input_ratios=input_ratios,
            num_classes=num_classes,
            nepoch=nepoch,
            increment=increment,
            criterion=criterion,
            optimizer=optimizer,
            transform=transform,
            device=device
        )

    context_num = int(num_classes // increment)

    # ================================
    # (c) 判断模型是否收敛
    # ================================
    # 在此示例中，我们以 history['accuracy'] 中记录的最后若干 epoch 训练准确率
    # 为判断依据。您也可以改为看 loss、或看最后一个 context 的 accuracy 等。
    model_converged = check_convergence(history['accuracy'], last_n=5, threshold=0.001)
    print(f"Model Convergence Check => {model_converged}")

    # ================================
    # (d) 释放显存，保存 JSON 结果
    # ================================
    torch.cuda.empty_cache()
    gc.collect()

    ensure_directory_exists(os.path.join(exp_dir, 'results'))
    ensure_directory_exists(os.path.join(exp_dir, 'checkpoint'))
    ensure_directory_exists(os.path.join(exp_dir, 'plots'))

    # 保存各种指标到文件
    train_acc_dir = os.path.join(exp_dir, 'results', f'training_acc_gen{generta}_{individual_id}.json')
    top1_acc_dir = os.path.join(exp_dir, 'results', f'top1_accuracy_gen{generta}_{individual_id}.json')
    avg_acc_dir = os.path.join(exp_dir, 'results', f'avg_accuracy_gen{generta}_{individual_id}.json')
    up2now_acc_dir = os.path.join(exp_dir, 'results', f'acc_up2now_gen{generta}_{individual_id}.json')
    avg_acc_dir_1 = os.path.join(exp_dir, 'results', f'avg_acc_by_mean_gen{generta}_{individual_id}.json')
    # avg_forget_dir = os.path.join(exp_dir, 'results', f'average_forgetting_gen{generta}_{individual_id}.json')
    fm_dir = os.path.join(exp_dir, 'results', f'FM_gen{generta}_{individual_id}.json')
    bwt_dir = os.path.join(exp_dir, 'results', f'BWT_gen{generta}_{individual_id}.json')
    converge_dir = os.path.join(exp_dir, 'results', f'convergence_gen{generta}_{individual_id}.json')  # 收敛结果

    with open(train_acc_dir, 'w') as file:
        json.dump(history['accuracy'], file)

    with open(top1_acc_dir, 'w') as file:
        json.dump(history['top1_accuracy'], file)

    with open(avg_acc_dir, 'w') as file:
        json.dump(history['average_accuracy'], file)

    with open(up2now_acc_dir, 'w') as file:
        json.dump(history['acc_up2now'], file)

    with open(avg_acc_dir_1, 'w') as file:
        json.dump(history['avg_acc_by_mean'], file)

    # with open(avg_forget_dir, 'w') as file:
    #     json.dump(history['average_forgetting'], file)

    with open(fm_dir, 'w') as file:
        json.dump(history['FM'], file)

    with open(bwt_dir, 'w') as file:
        json.dump(history['BWT'], file)

    # 保存模型是否收敛
    with open(converge_dir, 'w') as file:
        json.dump({"model_converged": model_converged}, file)

    # ================================
    # (e) 绘制“增量学习收敛曲线”并保存
    # ================================
    # 这里示例：将所有 epoch 的 accuracy 画在一张图中。
    # 您也可针对各个context分段绘制，或另作其他可视化。
    train_acc_plot_dir = os.path.join(exp_dir, 'plots', f'training_acc_curve_gen{generta}_{individual_id}.png')

    plt.figure(figsize=(5, 4))
    plt.plot(history['accuracy'], label='Train Accuracy (all epochs)')
    plt.title('Incremental Learning Convergence Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(train_acc_plot_dir)
    plt.close()

    # 也可以绘制平均精度曲线（针对每个context训练完后的测试）
    avg_acc_plot_dir = os.path.join(exp_dir, 'plots', f'avg_accuracy_curve_gen{generta}_{individual_id}.png')
    num_contexts = len(history['average_accuracy'])
    context_ticks = range(num_contexts)

    plt.figure(figsize=(5, 4))
    plt.plot(history['average_accuracy'], marker='o', label='Avg Accuracy (per context)')
    plt.title('Incremental Learning - Average Accuracy per Context')
    plt.xlabel('Context Index')
    plt.ylabel('Average Accuracy')
    plt.xticks(context_ticks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(avg_acc_plot_dir)
    plt.close()

    # ================================
    # (f) 计算并返回 fitness_score
    # ================================
    valleys, recoveries = calculate_valleys_and_recovery(history['accuracy'])
    fitness_score = fitness_function(valleys, recoveries)

    last_fm = history["FM"][-1]
    last_aa = history["acc_up2now"][-1]



    return fitness_score,last_aa, last_fm

def clear_session():
    """
    清理会话，释放内存。
    """
    torch.cuda.empty_cache()
    gc.collect()

def default_converter(o):
    if isinstance(o, np.integer):
        return int(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
