import os
import random
import yaml
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# Base default configuration
CONFIG = {
    "seed": 42,
    "train_stories": 20000,
    "val_stories": 1000,
    "test_stories": 1000,
    "train_stories_subset": 3000,
    "val_stories_subset": 500,
    "test_stories_subset": 500,
    "max_vocab_size": 10000,
    "min_freq": 2,
    "max_seq_len": 128,
    "embed_dim": 256,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "epochs": 15,
    "patience": 2,
}

# Load overrides from config.yaml if available
config_yaml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
if os.path.exists(config_yaml_path):
    try:
        with open(config_yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if isinstance(yaml_config, dict):
                CONFIG.update(yaml_config)
    except Exception:
        pass
