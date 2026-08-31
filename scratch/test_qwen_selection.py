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
qwen_model_path = model_paths.get("qwen2.5_vl")

# Import packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers import AutoModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dss.utils import load_and_pad_image, merge_sim_maps
from dss.clustering.leiden import get_patch_level_hierarchical_labels
from dss.sam.sam_utils import get_MAX_IoU_mask_from_SAM
from dss.qwen.qwen_utils import query_qwen_for_selection, clean_and_parse_json

# Load Qwen JSON
json_path = "outputs/CAMO/infer_CAMO_Qwen2.5-VL-7B_clean.json"
if not os.path.exists(json_path):
    print(f"Error: Qwen JSON not found at {json_path}")
    sys.exit(1)

with open(json_path, "r") as f:
    qwen_data = json.load(f)

entry = qwen_data[0]
img_path = entry["image"]
bboxes = entry["result"]["bbox_2d"]

# Try local path mapping
img_path_local = os.path.join("datasets/CAMO/Image", os.path.basename(img_path))
if os.path.exists(img_path_local):
    img_path = img_path_local

print(f"Image: {img_path}")

# Load models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Loading models...")
dino_model = AutoModel.from_pretrained(model_paths.get("dinov2"), output_hidden_states=True).to(device)
patch_size = dino_model.config.patch_size

sam2 = build_sam2(sam2_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2)

QWen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": device},
)
QWen_processor = AutoProcessor.from_pretrained(qwen_model_path)

# 1. Run DINOv2 feature extraction
image, image_tensor, (pad_W, pad_H), (W, H) = load_and_pad_image(img_path, patch_size=patch_size, target_size=1022)
dino_model.eval()
with torch.no_grad():
    outputs = dino_model(image_tensor.to(device))
    hidden_states = outputs.hidden_states
patch_tokens_2last = hidden_states[-1][:, 1:, :]
_, _, pH, pW = image_tensor.shape
num_patch_w = pW // patch_size
num_patch_h = pH // patch_size
feature_map_2last = patch_tokens_2last.reshape(1, num_patch_h, num_patch_w, -1).permute(0, 3, 1, 2)

candidate_masks, candidate_fgs, sim_maps_leiden, sim_maps_refine, padded_candidate_masks, low_res_mask, leiden_map, pca_data = get_patch_level_hierarchical_labels(
    data=feature_map_2last[0].cpu().numpy(),
    pad_H=pad_H, pad_W=pad_W,
    n_pca=16,
    resolution=0.5,
    alpha=1, beta=6, gamma=0.4
)
cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

merged_sim_maps = merge_sim_maps(sim_maps_leiden+sim_maps_refine, thresh=0.95, visualisation=False)
candidate_masks_final, candidate_scores_final = get_MAX_IoU_mask_from_SAM(merged_sim_maps, cv2_img, predictor, k=5)

# 2. Generate on-the-fly Qwen mask
predictor.set_image(cv2_img)
qwen_masks = []
if isinstance(bboxes, list) and not isinstance(bboxes[0], list):
    bboxes = [bboxes]
for box in bboxes:
    xmin, ymin, xmax, ymax = box
    xmin_px = max(0, min(int(xmin), W - 1))
    ymin_px = max(0, min(int(ymin), H - 1))
    xmax_px = max(xmin_px + 1, min(int(xmax), W))
    ymax_px = max(ymin_px + 1, min(int(ymax), H))
    
    sam_box = np.array([xmin_px, ymin_px, xmax_px, ymax_px])
    mask, _, _ = predictor.predict(box=sam_box, multimask_output=False)
    qwen_masks.append(mask[0])

qwen_pred = np.any(qwen_masks, axis=0).astype(np.uint8)
qwen_pred = cv2.resize(qwen_pred, dsize=(sim_maps_leiden[0].shape[1], sim_maps_leiden[0].shape[0]), interpolation=cv2.INTER_NEAREST)
from dss.scoring.scoring import boundary_contact
if boundary_contact(qwen_pred, n=10) > 0.75:
    qwen_pred = 1 - qwen_pred
from dss.utils import clean_mask
qwen_pred = clean_mask(qwen_pred, min_size=100)

# Add Qwen to candidates
candidate_masks_final.append(qwen_pred)
candidate_scores_final.append(np.array(candidate_scores_final).mean())

print(f"Number of final candidates: {len(candidate_masks_final)}")
print(f"Candidates heuristic scores: {candidate_scores_final}")

# 3. Simulate first comparison between Mask 1 and Mask 2
mask1, score1 = candidate_masks_final[0], candidate_scores_final[0]
mask2, score2 = candidate_masks_final[1], candidate_scores_final[1]

numpy_image = np.array(image)
masked_img1 = mask1[:, :, np.newaxis] * numpy_image
masked_img2 = mask2[:, :, np.newaxis] * numpy_image

print("\n--- Simulating query_qwen_for_selection ---")
# Inline the query logic to print the exact response text
from dss.qwen.qwen_utils import numpy_to_base64
from concurrent.futures import ThreadPoolExecutor

def encode_img(img):
    return numpy_to_base64(img)

with ThreadPoolExecutor(max_workers=3) as executor:
    img64, mask1_b64, mask2_b64 = executor.map(encode_img, [numpy_image, masked_img1, masked_img2])

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "The image is this."},
            {"type": "image", "image": f"data:image/png;base64,{img64}"},
            {"type": "text", "text": "The MASK A is this."},
            {"type": "image", "image": f"data:image/png;base64,{mask1_b64}"},
            {"type": "text", "text": "The MASK B is this."},
            {"type": "image", "image": f"data:image/png;base64,{mask2_b64}"},
            {"type": "text", "text": """
            CAMOUFLAGE MASK COMPARISON TASK
            IMAGE: The image may contain a few animal/insect or human whose shape, color, texture, pattern and movement closely resemble its surroundings.
            MASK A: Current best mask
            MASK B: New candidate mask

            MASK INTERPRETATION:
            - White areas = potential camouflaged objects
            - Black areas = background  

            CRITICAL REQUIREMENT: The chosen mask MUST match your object analysis.

            STEP-BY-STEP PROCESS:

            1. OBJECT IDENTIFICATION:
            - Carefully examine the image
            - Identify all hidden/concealed objects and their exact locations

            2. MASK EVALUATION:
            Mask A: Does the white region cover all identified objects? How much extra background?
            Mask B: Does the white region cover all identified objects? How much extra background?

            3. SELECTION CRITERIA:
            - PRIMARY: Choose the mask that covers ALL identified objects completely
            - SECONDARY: Among masks that meet primary criterion, choose the one with least background
            - If no mask covers all objects, choose the one that covers the most objects

            4. CONSISTENCY CHECK:
            - Ensure the chosen mask's white regions align with your identified objects
            - If analysis says "2 camouflaged insects", the mask should have 2 corresponding white regions

            OUTPUT JSON (DO NOT ADD ANY EXTRA INFO, JUST JSON!):
            [{
                "better_mask": "Mask A" / "Mask B", 
            }]
            """},
        ],
    } 
]

text = QWen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = [numpy_image, masked_img1, masked_img2], None

inputs = QWen_processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(device)

print("Calling Qwen generate...")
with torch.no_grad():
    generated_ids = QWen_model.generate(
        **inputs, 
        max_new_tokens=64, 
        output_scores=True, 
        do_sample=True,
        temperature=0.5, 
        return_dict_in_generate=True
    )
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids.sequences)]
output_text = QWen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(f"\nQwen Raw Response Text:\n{output_text}\n")

# Try to parse it
try:
    data = clean_and_parse_json(output_text)
    print("Parsed JSON data successfully:", data)
except Exception as e:
    print("JSON parsing failed with error:", e)
