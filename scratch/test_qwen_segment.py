#!/usr/bin/env python3
import os
import sys
import json
import torch
import cv2
import numpy as np

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
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load Qwen JSON
json_path = "outputs/CAMO/infer_CAMO_Qwen2.5-VL-7B_clean.json"
if not os.path.exists(json_path):
    print(f"Error: Qwen JSON not found at {json_path}")
    sys.exit(1)

with open(json_path, "r") as f:
    qwen_data = json.load(f)

# Find first valid entry
entry = None
for item in qwen_data:
    if isinstance(item.get("result"), dict) and "bbox_2d" in item["result"]:
        entry = item
        break

if not entry:
    print("Error: No valid JSON entries found.")
    sys.exit(1)

img_path = entry["image"]
bboxes = entry["result"]["bbox_2d"]
# Qwen coordinates are in flat list [xmin, ymin, xmax, ymax]
print(f"Image: {img_path}")
print(f"Qwen bbox: {bboxes}")

# Load image
cv2_img = cv2.imread(img_path)
if cv2_img is None:
    # Try mapping to local path
    img_path_local = os.path.join("datasets/CAMO/Image", os.path.basename(img_path))
    cv2_img = cv2.imread(img_path_local)
    if cv2_img is None:
         print(f"Error: Image not found at {img_path}")
         sys.exit(1)
    img_path = img_path_local

H, W = cv2_img.shape[:2]
print(f"Image dimensions (H, W): ({H}, {W})")

# Load SAM
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sam2 = build_sam2(sam2_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2)

predictor.set_image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

# Run segmenter fallback
qwen_masks = []
# Ensure list of lists
if isinstance(bboxes, list) and not isinstance(bboxes[0], list):
    bboxes = [bboxes]

for box in bboxes:
    xmin, ymin, xmax, ymax = box
    xmin_px = max(0, min(int(xmin), W - 1))
    ymin_px = max(0, min(int(ymin), H - 1))
    xmax_px = max(xmin_px + 1, min(int(xmax), W))
    ymax_px = max(ymin_px + 1, min(int(ymax), H))
    
    sam_box = np.array([xmin_px, ymin_px, xmax_px, ymax_px])
    print(f"SAM prompted box coordinates (xmin, ymin, xmax, ymax): {sam_box}")
    
    mask, iou_scores, _ = predictor.predict(
        box=sam_box,
        multimask_output=False
    )
    print(f"SAM IoU Score: {iou_scores[0]:.4f}")
    qwen_masks.append(mask[0])

if len(qwen_masks) > 0:
    qwen_pred_full = np.any(qwen_masks, axis=0).astype(np.uint8)
else:
    qwen_pred_full = np.zeros((H, W), dtype=np.uint8)

print(f"Generated mask sum (number of foreground pixels): {qwen_pred_full.sum()}")
print("Successfully verified dynamic segmentation.")
