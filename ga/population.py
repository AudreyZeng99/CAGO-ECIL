# ga/population.py
# 管理种群的创建和更新

import numpy as np
import copy
import json
import os

def create_population(population_size, num_classes, increment):
    population = []
    for p in range(population_size):
        individual = []
        for i in range(0, num_classes, increment):
            current_length = min(i + increment, num_classes)
            ratios = np.round(np.random.uniform(1, 10, current_length), 2)
            ratios = list(ratios) + [0] * (num_classes - current_length)
            individual.append(ratios)
        population.append(individual)
    print("Type of population:", type(population))
    print("Shape of population:", get_shape_of_nested_list(population))
    return population

def get_shape_of_nested_list(nested_list):
    if not isinstance(nested_list, list):
        return []
    if not nested_list:
        return [0]
    shape = [len(nested_list)]
    while isinstance(nested_list[0], list):
        shape.append(len(nested_list[0]))
        nested_list = nested_list[0]
    return shape


def select_top_individuals(population, selector, scores, select_rate): # selector =

    if selector == 'Fitness':
        num_selected = int(len(population) * select_rate)
        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = sorted_indices[:num_selected]
        top_individuals = [population[idx] for idx in selected_indices]

        best_indice = sorted_indices[0]
        best_individual = population[best_indice]
        return best_individual, top_individuals, selected_indices
    elif selector == "AA":
        num_selected = 1
        sorted_indices = np.argsort(selector)[::-1]
        selected_indices = sorted_indices[:num_selected]
        best_indice = sorted_indices[0]
        best_individual = population[best_indice]
        return best_individual
    elif selector == "FM":
        num_selected = 1
        sorted_indices = np.argsort(selector)
        selected_indices = sorted_indices[:num_selected]
        best_indice = sorted_indices[0]
        best_individual = population[best_indice]
        return best_individual


def regenerate_population(top_individuals, population_size, remaining_population, mutation_rate, num_classes, increment):
    num_new_individuals = population_size - len(top_individuals)
    if num_new_individuals <= 0:
        print("No new individuals needed.")
        return top_individuals + remaining_population

    parents = select_parents(top_individuals, len(top_individuals))
    offspring = crossover(parents, int(num_new_individuals) // 2)
    new_individuals = mutate(offspring, mutation_rate)
    if len(new_individuals) < num_new_individuals:
        offspring_new = create_population(num_new_individuals - len(new_individuals), num_classes, increment)
        new_population = top_individuals + new_individuals + offspring_new
    else:
        new_population = top_individuals + new_individuals
    return new_population

def select_parents(selected_population, num_parents):
    selected_indices = np.random.choice(len(selected_population), size=num_parents, replace=False)
    parents_list = [selected_population[idx] for idx in selected_indices]
    return parents_list

def crossover(parents, num_new_individuals, crossover_rate=0.5):
    offspring = []
    if not parents:
        print("     No parents available for crossover.")
        return offspring
    while len(offspring) < num_new_individuals:
        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                break
            parent1 = copy.deepcopy(parents[i])
            parent2 = copy.deepcopy(parents[i + 1])
            child1, child2 = parent1, parent2
            for ratio in range(len(parent1[-1])):
                if np.random.rand() < crossover_rate:
                    for context in range(len(parent1)):
                        if child1[context][ratio] == 0:
                            continue
                        child1[context][ratio] = parent2[context][ratio]
                        child2[context][ratio] = parent1[context][ratio]
            offspring.extend([child1, child2])
            if len(offspring) >= num_new_individuals:
                break
    return offspring[:num_new_individuals]

def mutate(offspring, mutation_rate):
    for child in offspring:
        for ratio in range(len(child[-1])):
            if np.random.rand() < mutation_rate:
                mutated_gene = np.random.randint(1, 11)
                for context in range(len(child)):
                    if child[context][ratio] == 0:
                        continue
                    child[context][ratio] = mutated_gene
    return offspring
