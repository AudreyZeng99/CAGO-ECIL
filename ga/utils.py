# ga/utils.py
# 实现GA的辅助算法，如适应度函数的自定义设计部分！

import numpy as np

def calculate_valleys_and_recovery(accuracies, window_size=2):
    valleys = []
    recoveries = []
    for i in range(1, len(accuracies) - 1):
        if is_valley(accuracies, i, window_size):
            left_peak_index, _ = find_peak(accuracies, i, direction=-1)
            right_peak_index, _ = find_peak(accuracies, i, direction=1)
            depth = max(accuracies[left_peak_index], accuracies[right_peak_index]) - accuracies[i]
            valleys.append(depth)
            recovery = right_peak_index - i
            recoveries.append(recovery)
    return valleys, recoveries

def is_valley(accuracies, i, window_size=2):
    for offset in range(1, window_size + 1):
        if i - offset < 0 or i + offset >= len(accuracies):
            return False  # Boundary check
        if accuracies[i] >= accuracies[i - offset] or accuracies[i] >= accuracies[i + offset]:
            return False
    return True

def find_peak(accuracies, start_index, direction):
    peak_value = accuracies[start_index]
    peak_index = start_index
    for i in range(start_index, len(accuracies) if direction > 0 else -1, direction):
        if accuracies[i] >= peak_value:
            peak_value = accuracies[i]
            peak_index = i
        else:
            break
    return peak_index, peak_value

def fitness_function(valley_depths, recovery_epochs, weights=None):
    if weights is None:
        weights = {'mean': 1.5, 'max': 2.0, 'std_dev': 0.5}

    if len(valley_depths) == 0 or len(recovery_epochs) == 0:
        # 如果没有山谷或恢复，返回适应度分数为0
        return 0

    mean_depth = np.mean(valley_depths)
    max_depth = np.max(valley_depths)
    std_dev_depth = np.std(valley_depths)
    print('mean_depth:', mean_depth, 'max_depth:', max_depth, 'std_dev_depth:', std_dev_depth)
    mean_recovery = np.mean(recovery_epochs)
    max_recovery = np.max(recovery_epochs)
    std_dev_recovery = np.std(recovery_epochs)
    print('mean_recovery:', mean_recovery, 'max_recovery:', max_recovery, 'std_dev_recovery:', std_dev_recovery)
    fitness_score = (weights['mean'] / (1 + mean_depth) +
                     weights['max'] / (1 + max_depth) +
                     weights['std_dev'] / (1 + std_dev_depth)) + (weights['mean'] / (1 + mean_recovery) +
                                                                  weights['max'] / (1 + max_recovery) +
                                                                  weights['std_dev'] / (1 + std_dev_recovery))
    print('score:', fitness_score)
    return fitness_score
