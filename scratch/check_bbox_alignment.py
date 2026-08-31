#!/usr/bin/env python3
import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm

json_path = "outputs/CAMO/infer_CAMO_Qwen2.5-VL-7B_clean.json"
gt_dir = "datasets/CAMO/GT"
img_dir = "datasets/CAMO/Image"

if not os.path.exists(json_path):
    print(f"Error: JSON not found at {json_path}")
    sys.exit(1)

with open(json_path, "r") as f:
    qwen_data = json.load(f)

def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

ious_order1 = [] # [xmin, ymin, xmax, ymax] scaled to 1000
ious_order2 = [] # [ymin, xmin, ymax, xmax] scaled to 1000
ious_order3 = [] # raw unscaled pixels [x1, y1, x2, y2]
ious_order4 = [] # raw unscaled pixels [y1, x1, y2, x2]

empty_count = 0
valid_count = 0

for item in qwen_data:
    img_path = item.get("image", "")
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    gt_path = os.path.join(gt_dir, base_name + ".png")
    
    if not os.path.exists(gt_path):
        continue
        
    gt_mask = cv2.imread(gt_path, 0)
    if gt_mask is None or (gt_mask > 0).sum() == 0:
        continue
        
    H, W = gt_mask.shape[:2]
    y_indices, x_indices = (gt_mask > 0).nonzero()
    gt_box = [x_indices.min(), y_indices.min(), x_indices.max(), y_indices.max()]
    
    result = item.get("result", {})
    if not isinstance(result, dict) or "bbox_2d" not in result:
        empty_count += 1
        continue
        
    raw_box = result["bbox_2d"]
    if isinstance(raw_box, list) and len(raw_box) > 0 and isinstance(raw_box[0], list):
        box = raw_box[0]
    elif isinstance(raw_box, list) and len(raw_box) == 4:
        box = raw_box
    else:
        empty_count += 1
        continue
        
    v1, v2, v3, v4 = box
    valid_count += 1
    
    # Interpretation 1: [xmin, ymin, xmax, ymax] scaled to 1000
    b1 = [int(v1 * W / 1000.0), int(v2 * H / 1000.0), int(v3 * W / 1000.0), int(v4 * H / 1000.0)]
    ious_order1.append(compute_iou(b1, gt_box))
    
    # Interpretation 2: [ymin, xmin, ymax, xmax] scaled to 1000
    b2 = [int(v2 * W / 1000.0), int(v1 * H / 1000.0), int(v4 * W / 1000.0), int(v3 * H / 1000.0)]
    ious_order2.append(compute_iou(b2, gt_box))
    
    # Interpretation 3: Raw pixel [x1, y1, x2, y2]
    b3 = [int(v1), int(v2), int(v3), int(v4)]
    ious_order3.append(compute_iou(b3, gt_box))
    
    # Interpretation 4: Raw pixel [y1, x1, y2, x2]
    b4 = [int(v2), int(v1), int(v4), int(v3)]
    ious_order4.append(compute_iou(b4, gt_box))

print("=" * 60)
print(f"Total valid boxes evaluated: {valid_count} (Empty / missing: {empty_count})")
print("=" * 60)
print(f"Mean IoU (Interpretation 1: [xmin, ymin, xmax, ymax] / 1000): {np.mean(ious_order1):.4f}")
print(f"Mean IoU (Interpretation 2: [ymin, xmin, ymax, xmax] / 1000): {np.mean(ious_order2):.4f}")
print(f"Mean IoU (Interpretation 3: Raw pixels [xmin, ymin, xmax, ymax]): {np.mean(ious_order3):.4f}")
print(f"Mean IoU (Interpretation 4: Raw pixels [ymin, xmin, ymax, xmax]): {np.mean(ious_order4):.4f}")
print("=" * 60)
