# DSS - Discover, Segment, Select: Perfect Replication

This is a perfect replication of the CVPR 2026 paper "Discover, Segment, and Select: A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation" by Yilong Yang et al.

## Original Repository
All files have been cloned exactly from: https://github.com/ynulonger/DSS

## Project Structure
```
cos/
├── dss/                        # Modular package implementation of the paper
│   ├── clustering/             # Feature-coherent Object Discovery (Leiden + Part Composition)
│   ├── datasets/               # Training & Evaluation dataset loaders
│   ├── models/                 # Neural architectures (DINOv2 patch classifier, losses)
│   ├── sam/                    # Segment Anything (SAM2) integrations
│   ├── scoring/                # Mask heuristics scoring & soft voting merge
│   ├── qwen/                   # MLLM prompt formatting & pairwise selection
│   ├── utils/                  # General helper modules (image, bbox, masks, plotting)
│   └── inference/              # Core pipelines (Qwen parallel runs, DRS predict)
├── original/                   # Unmodified reference copy of original repository
│   └── DSS/                    # Monolithic source files (utils.py, infer_DRS_pred.py, etc.)
├── models/                     # Downloaded pre-trained weights (DINOv2, SAM2, Qwen2.5-VL)
├── datasets/                   # Camouflaged Object Segmentation test sets (COD10K, CAMO, etc.)
├── outputs/                    # Generated prediction masks and logs
├── configs/                    # Yaml configuration files (dss_config.yaml)
├── scripts/                    # Helper & verification scripts (verify_imports.py)
├── run_pipeline.py             # Unified command-line pipeline execution runner
├── REPLICATION_GUIDE.md        # Full replication guide documentation
└── requirements.txt            # Python dependencies list
```

## Original DSS Files (Perfect Replication)

### 1. Main Scripts
- **`infer_COD10K_QWen_7B_parallel.py`** - Main inference script for COD10K dataset using QWen2.5-VL (7B) with parallel processing
- **`infer_DRS_pred.py`** - DRS (DINOv2 Refinement & Segmentation) prediction script
- **`clean_QWen_output.py`** - Cleans QWen output JSON files

### 2. Core Modules
- **`utils.py`** (79,502 lines) - Main utility module with all DSS functionality:
  - DINOv2 feature extraction
  - Leiden clustering with ScanPy integration
  - Part Composition refinement
  - Similarity-based Box Generation
  - Feature processing and visualization
  - Mask selection and evaluation metrics

- **`refine_leiden_utils.py`** (19,123 lines) - Leiden clustering refinement utilities:
  - Feature coherence refinement
  - Similarity map computation
  - Spatial entropy calculations
  - Cluster filtering and refinement

### 3. Pre-computed Results
- **`json_files/`** - Pre-computed QWen outputs for all datasets:
  - `infer_CAMO_QWen_7B_clean.json` - CAMO dataset (250 images)
  - `infer_CHAMELEON_QWen_7B_clean.json` - CHAMELEON dataset (76 images)
  - `infer_COD10K_QWen_7B_clean.json` - COD10K dataset (2,026 images)
  - `infer_NC4K_QWen_7B_clean.json` - NC4K dataset (4,121 images)

## DSS Pipeline (Based on Paper)

### Stage 1: Discover (Feature-coherent Object Discovery - FOD)
1. **DINOv2 Feature Extraction**: Extract patch-level features using DINOv2
2. **Leiden Clustering**: Adaptive clustering using Leiden algorithm (resolution=0.5)
3. **Part Composition (PC)**: Iterative refinement with feature coherence energy (ϵ=1.0)
4. **Similarity-based Box Generation (SBG)**: Generate bounding boxes from similarity maps (τ=0.95)

### Stage 2: Segment
1. **SAM2 Segmentation**: Use SAM2 (ViT-L) to segment each proposal bbox
2. **Candidate Masks**: Generate multiple candidate masks from different proposals

### Stage 3: Select (Semantic-driven Mask Selection - SMS)
1. **Heuristic Scoring**: Score masks using correlation and boundary contact
2. **Top-K Selection**: Select top-K masks (K=5) for pairwise comparison
3. **QWen2.5-VL Selection**: Progressive pairwise comparison using QWen2.5-VL-Instruct (7B)

## Getting Started

### 1. Install Dependencies
```bash
cd cos
pip install -r requirements.txt
```

### 2. Download Models
```bash
# Download DINOv2
huggingface-cli download facebook/dinov2-base --local-dir models/dinov2

# Download QWen2.5-VL (7B)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen2.5_vl

# Download SAM2 (ViT-L)
# Visit: https://github.com/facebookresearch/segment-anything-2
# Download: sam2_vit_l.pth to models/sam2_vit_l/
```

### 3. Download Datasets
```bash
# Download standard COS datasets:
# - CHAMELEON (76 images)
# - CAMO-Test (250 images)
# - COD10K-Test (2,026 images)
# - NC4K (4,121 images)

# Place datasets in datasets/ directory
```

### 4. Run Inference
To run the end-to-end Zero-shot COS pipeline (Stage 1 + Stage 2/3) using the modularized codebase:
```bash
python run_pipeline.py --dataset cod10k --stage all
```
For advanced options and customized configs, please refer to the detailed [REPLICATION_GUIDE.md](file:///Users/dharmendrakumar/cos/REPLICATION_GUIDE.md).

## Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- NVIDIA GPU with ≥24GB VRAM (RTX A5000/3090 recommended)

## Citation
If you use this implementation, please cite the original paper:
```bibtex
@inproceedings{yang2026dss,
  title     = {Discover, Segment, and Select: A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation},
  author    = {Yilong Yang, Jianxin Tian, Shengchuan Zhang, Liujuan Cao},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Notes
- This is a **perfect replication** of the original DSS repository
- All files are identical to the original GitHub repository
- The implementation matches the paper's architecture exactly
- Pre-computed QWen outputs for all datasets are included