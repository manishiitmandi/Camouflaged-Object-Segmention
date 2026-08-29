"""Configuration and constants for DSS."""

import random
import os
import numpy as np
import torch


def setup_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# Paper hyperparameters (from Table 1)
HYPERPARAMETERS = {
    # Stage 1: Feature-coherent Object Discovery (FOD)
    'leiden_resolution': 0.5,
    'pc_energy_threshold': 1.0,      # ε
    'correlation_threshold': 0.95,   # τ
    'top_k': 5,                      # K

    # Processing parameters
    'target_size': 1120,
    'patch_size': 14,

    # Stage 2: SAM Segmentation
    'sam_model': 'sam2_vit_l',

    # Stage 3: Semantic-driven Mask Selection (SMS)
    'qwen_model': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'max_tokens': 512,
}

# Dataset configurations
DATASETS = {
    'CHAMELEON': {
        'num_images': 76,
        'description': 'Chameleon dataset',
    },
    'CAMO': {
        'num_images': 250,
        'description': 'CAMO dataset',
    },
    'COD10K': {
        'num_images': 2026,
        'description': 'COD10K dataset',
    },
    'NC4K': {
        'num_images': 4121,
        'description': 'NC4K dataset',
    },
}

# Hardware configuration
DEVICE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 1,
    'num_workers': 4,
    'fp16': torch.cuda.is_available(),
}

# Model paths (update after downloading)
MODEL_PATHS = {
    'dinov2': 'models/dinov2',
    'sam2_vit_l': 'models/sam2_vit_l',
    'qwen2.5_vl': 'models/qwen2.5_vl',
}

# Evaluation metrics
METRICS = ['Sα', 'Eφ', 'Fwβ', 'MAE']

# Inference timing (from paper)
INFERENCE_TIMING = {
    'FOD': 7.74,     # seconds per image
    'SAM': 3.63,     # seconds per image
    'SMS': 30.59,    # seconds per image
    'total': 41.96,  # seconds per image
}