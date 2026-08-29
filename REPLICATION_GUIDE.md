# DSS Replication Guide

This guide outlines the steps to replicate the results of the paper **"Discover, Segment, and Select: A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation"** (CVPR 2026) using this modularized codebase.

---

## 1. Install Dependencies
Ensure your environment meets the requirements and install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Download Pre-trained Models
DSS is a training-free framework that leverages pre-trained backbones. Download the following models:

### A. DINOv2 (Feature Extractor)
Download the model from Hugging Face:
```bash
huggingface-cli download facebook/dinov2-base --local-dir models/dinov2
```

### B. Qwen2.5-VL-7B-Instruct (MLLM Selection & Localization)
Download the model from Hugging Face:
```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen2.5_vl
```

### C. SAM2 - ViT-Large (Segmenter)
1. Install `segment-anything-2` by following [Meta's SAM2 installation instructions](https://github.com/facebookresearch/segment-anything-2).
2. Download the weights file `sam2.1_hiera_large.pt` and the model config `sam2.1_hiera_l.yaml`.
3. Save them inside the `models/sam2_vit_l/` directory.

---

## 3. Download the Datasets
Replication results are evaluated on four standard zero-shot Camouflaged Object Segmentation (COS) test sets:
* **CHAMELEON** (76 images)
* **CAMO** (250 images)
* **COD10K** (2,026 images)
* **NC4K** (4,121 images)

Download the datasets and extract them inside the `datasets/` directory of your workspace (e.g. `datasets/COD10K/Image/` and `datasets/COD10K/GT/`).

---

## 4. Verify Configuration File
A master configuration file is defined at [configs/dss_config.yaml](file:///Users/dharmendrakumar/cos/configs/dss_config.yaml) where you can update directories and adjust hyperparameters matching the paper:
- **`model_paths`**: Directories containing downloaded weights.
- **`datasets`**: Root folders for each zero-shot test set.
- **`hyperparameters`**: Table 1 paper parameters (Leiden resolution scale $r = 0.5$, PC energy threshold $\epsilon = 1.0$, PC weights $\alpha = 1.0$, $\beta = 6.0$, $\gamma = 0.4$, SBG correlation threshold $\tau = 0.95$, target SAM size $T = 1022$, and Top-$K$ selection count $K = 5$).
- **`execution`**: Active GPUs (e.g. `"0,1"`) and seed configurations.

---

## 5. Run Stage 1 — MLLM Bounding Box Localization
Execute Qwen2.5-VL to find coordinates of hidden/camouflaged targets across test images. You can write a small runner script in the workspace root:

```python
# run_localization.py
from dss import run_qwen_inference

---

## 5. Run the End-to-End Pipeline
We have provided a unified entrypoint script [run_pipeline.py](file:///Users/dharmendrakumar/cos/run_pipeline.py) which automatically loads configuration parameters from [configs/dss_config.yaml](file:///Users/dharmendrakumar/cos/configs/dss_config.yaml) and executes the zero-shot camouflaged object segmentation.

To execute both Stage 1 (Qwen Bbox Localization) and Stage 2/3 (DRS Segment & Select) sequentially:
```bash
./run_pipeline.py --dataset cod10k --stage all
```

### Advanced Command Line Flags:
* **`--stage`**: `1` (run localization only), `2` (run DRS segment/select only), or `all` (run full pipeline sequentially).
* **`--dataset`**: Choose dataset to process (`chameleon`, `camo`, `cod10k`, or `nc4k`).
* **`--gpus`**: Override GPU IDs to use (e.g. `--gpus "0,1,2"`).
* **`--config`**: Specify a custom YAML configuration file path.

Example: If you already have bounding boxes computed and only want to run DRS segment/select on the CAMO dataset:
```bash
./run_pipeline.py --dataset camo --stage 2
```
*This generates predictions in `outputs/preds/CAMO/refine+True_merge+True_include+False/`.*

---

## 6. Compute Metrics
Compare your generated predictions against the ground truth labels in `datasets/COD10K/GT/` to calculate the evaluation metrics reported in the paper:
* **Structure-measure ($S_\alpha$)**
* **Enhanced-alignment measure ($E_\phi$)**
* **Weighted F-measure ($F_\beta^w$)**
* **Mean Absolute Error ($MAE$)**
