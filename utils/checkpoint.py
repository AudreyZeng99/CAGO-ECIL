# utils/checkpoint.py

import torch
import json
import os

def save_checkpoint(model, population, fitness_scores, best_fitness,best_aa, best_fm, generation, exp_dir):
    checkpoint_dir = os.path.join(exp_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
    ga_state_dir = os.path.join(checkpoint_dir, f'ga_state_gen{generation}.json')

    try:
        torch.save(model.state_dict(), model_checkpoint_path)
        state = {
            'population': population,
            'fitness_scores': [float(score) for score in fitness_scores],
            'generation': generation,
            'best_fitness': [float(fit) for fit in best_fitness],
            'best_aa': [float(aa) for aa in best_aa],
            'best_fm': [float(fm) for fm in best_fm ]

        }
        with open(ga_state_dir, 'w') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        print(f"Checkpoint saved for generation {generation} at {model_checkpoint_path}")
    except Exception as e:
        print(f"Failed to save checkpoint for generation {generation}: {e}")

def load_checkpoint(checkpoint_dir, model):
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint directory does not exist:", checkpoint_dir)
        return None, None, None, 0, None

    try:
        # 找到最新的生成代数
        ga_states = [f for f in os.listdir(checkpoint_dir) if f.startswith('ga_state_gen')]
        if not ga_states:
            print("No GA state files found in checkpoint directory:", checkpoint_dir)
            return None, None, None, 0, None
        latest_gen = max([int(f.split('_')[-1].split('.json')[0]) for f in ga_states])
        model_checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')
        ga_state_dir = os.path.join(checkpoint_dir, f'ga_state_gen{latest_gen}.json')

        if not os.path.exists(model_checkpoint_path) or not os.path.exists(ga_state_dir):
            print("Checkpoint files do not exist for the latest generation:", latest_gen)
            return None, None, None, 0, None

        # 加载模型状态
        model.load_state_dict(torch.load(model_checkpoint_path))
        model.to(device)

        # 加载 GA 状态
        with open(ga_state_dir, 'r') as f:
            state = json.load(f)
        population = state['population']
        fitness_scores = state['fitness_scores']
        generation = state['generation']
        best_fitness = state.get('best_fitness', [])

        print(f"Loaded checkpoint for generation {generation} from {checkpoint_dir}")
        return model, population, fitness_scores, generation, best_fitness
    except Exception as e:
        print("Failed to load model or state from the specified directory:", str(e))
        return None, None, None, 0, None
