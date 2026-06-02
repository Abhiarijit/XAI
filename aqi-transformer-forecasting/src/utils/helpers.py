def load_config(config_file):
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, config_file):
    import json
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)

def create_directory(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_seed(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]