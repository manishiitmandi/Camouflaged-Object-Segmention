"""
DSS - Modular Implementation

Discover, Segment, and Select: A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation
"""

__version__ = "1.0.0"
__author__ = "DSS Project Team"

# Re-export configuration and seeds
from .config import HYPERPARAMETERS, DATASETS, DEVICE_CONFIG, MODEL_PATHS, METRICS
from .utils.general import setup_seed

# Re-export core datasets, models, and inference pipelines
from .datasets import Pseudo_Dataset, infer_Dataset, Sam_refine_Dataset
from .models import DINOv2PatchClassifier, IoULoss
from .inference import run_qwen_inference, run_drs_inference

__all__ = [
    'HYPERPARAMETERS', 'DATASETS', 'DEVICE_CONFIG', 'MODEL_PATHS', 'METRICS',
    'setup_seed',
    'Pseudo_Dataset', 'infer_Dataset', 'Sam_refine_Dataset',
    'DINOv2PatchClassifier', 'IoULoss',
    'run_qwen_inference', 'run_drs_inference',
]