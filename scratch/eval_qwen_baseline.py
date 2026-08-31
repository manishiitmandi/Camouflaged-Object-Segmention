#!/usr/bin/env python3
import os
import sys
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Load config
config_path = "configs/dss_config.yaml"
if not os.path.exists(config_path):
    print(f"Error: Config not found at {config_path}")
    sys.exit(1)

with open(config_path, "r") as f:
    import yaml
    config = yaml.safe_load(f)

model_paths = config.get("model_paths", {})
sam2_checkpoint = model_paths.get("sam2_vit_l", {}).get("checkpoint")
sam2_cfg = model_paths.get("sam2_vit_l", {}).get("config")

# Import SAM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from dss.scoring.scoring import boundary_contact
from dss.utils import clean_mask

# Output directory for baseline masks
out_dir = "outputs/preds/CAMO/baseline_qwen_sam2"
os.makedirs(out_dir, exist_ok=True)

# Load Qwen JSON
json_path = "outputs/CAMO/infer_CAMO_Qwen2.5-VL-7B_clean.json"
if not os.path.exists(json_path):
    print(f"Error: Qwen JSON not found at {json_path}")
    sys.exit(1)

with open(json_path, "r") as f:
    qwen_data = json.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Loading SAM2 model on {device}...")
sam2 = build_sam2(sam2_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2)

print(f"Generating Qwen+SAM2 baseline predictions for {len(qwen_data)} images...")

for entry in tqdm(qwen_data, desc="Segmenting"):
    img_path = entry.get("image", "")
    result = entry.get("result", {})
    
    # Try local path mapping
    if not os.path.exists(img_path):
        img_path_local = os.path.join("datasets/CAMO/Image", os.path.basename(img_path))
        if os.path.exists(img_path_local):
            img_path = img_path_local
        else:
            continue
            
    cv2_img = cv2.imread(img_path)
    if cv2_img is None:
        continue
        
    H, W = cv2_img.shape[:2]
    predictor.set_image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    
    bboxes = []
    if isinstance(result, dict) and "bbox_2d" in result:
        raw_box = result["bbox_2d"]
        if isinstance(raw_box, list):
            if len(raw_box) > 0 and isinstance(raw_box[0], list):
                bboxes = raw_box
            elif len(raw_box) == 4 and all(isinstance(x, (int, float)) for x in raw_box):
                bboxes = [raw_box]
                
    qwen_masks = []
    for box in bboxes:
        xmin, ymin, xmax, ymax = box
        xmin_px = int(xmin * W / 1000.0)
        ymin_px = int(ymin * H / 1000.0)
        xmax_px = int(xmax * W / 1000.0)
        ymax_px = int(ymax * H / 1000.0)
        
        xmin_px = max(0, min(xmin_px, W - 1))
        ymin_px = max(0, min(ymin_px, H - 1))
        xmax_px = max(xmin_px + 1, min(xmax_px, W))
        ymax_px = max(ymin_px + 1, min(ymax_px, H))
        
        sam_box = np.array([xmin_px, ymin_px, xmax_px, ymax_px])
        try:
            mask, _, _ = predictor.predict(box=sam_box, multimask_output=False)
            qwen_masks.append(mask[0])
        except Exception:
            pass
            
    if len(qwen_masks) > 0:
        qwen_pred = np.any(qwen_masks, axis=0).astype(np.uint8)
    else:
        qwen_pred = np.zeros((H, W), dtype=np.uint8)
        
    if boundary_contact(qwen_pred, n=10) > 0.75:
        qwen_pred = 1 - qwen_pred
    qwen_pred = clean_mask(qwen_pred, min_size=100)
    
    # Save mask
    save_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
    save_path = os.path.join(out_dir, save_name)
    cv2.imwrite(save_path, (qwen_pred * 255).astype(np.uint8))

print(f"\n✓ Baseline masks saved to: {out_dir}")
print("Running evaluation on Qwen+SAM2 baseline...")

# Run evaluation script directly
import subprocess
cmd = [
    sys.executable,
    "scripts/evaluate_predictions.py",
    "--pred_dir", out_dir,
    "--gt_dir", "datasets/CAMO/GT"
]
subprocess.run(cmd)
