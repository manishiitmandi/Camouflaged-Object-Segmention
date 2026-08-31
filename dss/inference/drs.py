"""DINOv2 Refinement and Segmentation (DRS) inference pipeline runner."""

import os
import json
import time
import torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
import cv2

from transformers import AutoModel, AutoImageProcessor, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from dss.utils import load_and_pad_image, save_mask_as_color, clean_mask, merge_sim_maps
from dss.clustering.leiden import get_patch_level_hierarchical_labels
from dss.sam.sam_utils import get_MAX_IoU_mask_from_SAM
from dss.scoring.scoring import boundary_contact
from dss.qwen.qwen_utils import pairwise_selection_from_QWen


def get_img_list(json_file, pred_dir, local_dataset_dir=None):
    """Retrieve lists of image paths and corresponding bounding boxes to process."""
    pred_list = glob(os.path.join(pred_dir, "*.png"))
    pred_list = [os.path.basename(pred).replace(".png", "") for pred in pred_list]
    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    img_list = []
    bbox_list = []
    print(f'{len(pred_list)} imgs finished, left {len(results)-len(pred_list)} imgs to process')

    # Build local path map if local_dataset_dir is provided or exists
    path_map = {}
    if not local_dataset_dir:
        # Infer dataset name from json filename or path if possible
        for dataset_key in ["camo", "cod10k", "chameleon", "nc4k"]:
            if dataset_key in json_file.lower():
                local_dataset_dir = os.path.join("datasets", dataset_key.upper())
                break
    
    if local_dataset_dir and os.path.exists(local_dataset_dir):
        print(f"Building local image path mapping from: {local_dataset_dir}")
        for root, _, files in os.walk(local_dataset_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path_map[f.lower()] = os.path.join(root, f)
    else:
        print(f"Warning: local_dataset_dir '{local_dataset_dir}' not found. Using paths from JSON file directly.")

    for item in tqdm(results):
        confidence = []
        img_path = item["image"]
        
        # Map path locally
        basename = os.path.basename(img_path).lower()
        if basename in path_map:
            img_path = path_map[basename]
            
        img_name = os.path.basename(img_path).split(".")[0]
        if img_name in pred_list:
            continue
        result = item["result"]
        if isinstance(result, str):
            continue
        elif isinstance(result, list):
            bboxes = []
            for obj in result:
                if isinstance(obj, dict) and "confidence" in obj:
                    confidence.append(obj["confidence"])
                if isinstance(obj, dict) and "bbox_2d" in obj:
                    bboxes.append(obj["bbox_2d"])
        elif isinstance(result, dict):
            bboxes = []
            if "description" in result.keys():
                img = cv2.imread(img_path, flags=0)
                if img is not None:
                    bboxes.append([0, img.shape[0], 0, img.shape[1]])
                else:
                    bboxes.append([0, 1022, 0, 1022])
            for obj in result:
                if obj == "confidence":
                    confidence.append(result["confidence"])
                if obj == "bbox_2d":
                    bboxes.append(result["bbox_2d"])

        bboxes_np = np.array(bboxes)
        if len(bboxes_np.shape) == 3:
            bboxes_np = bboxes_np[0]
        bbox_list.append(bboxes_np)
        img_list.append(img_path)

    print(f'num of images: {len(img_list)}, {len(bbox_list)}')
    return img_list, bbox_list



def worker(
    rank, gpu_id, dataset, img_chunk, bbox_chunk, pred_dir, 
    refine, merge, include_Qwen_pred, 
    dino_model_path, qwen_model_path, sam2_checkpoint, sam2_cfg
):
    """
    Multiprocessing worker function executing the DRS pipeline on a slice of images.
    """
    FOD_time = 0
    Seg_time = 0
    SMS_time = 0
    total_time = 0
    pbar = tqdm(
        total=len(img_chunk),
        desc=f"Worker {rank}, GPU: {gpu_id}",
        position=rank,
        leave=False
    )
    
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    QWen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    QWen_processor = AutoProcessor.from_pretrained(qwen_model_path)
    QWen_model.eval()
    
    print(f"Worker {rank} loading models on GPU {gpu_id}...")
    dino_model = AutoModel.from_pretrained(dino_model_path, output_hidden_states=True).to(device)
    patch_size = dino_model.config.patch_size
    
    sam2 = build_sam2(sam2_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2)

    if "COD10K" in dataset:
        gt_dir = "/data/yilong/Datasets/COD10K-v3/Test/GT_Object/"
    else:
        gt_dir = f"/data/yilong/Datasets/{dataset}/GT/"

    for image_path, bboxes_qwen in zip(img_chunk, bbox_chunk):
        start_time = time.time()
        sam_pred_path = f'/data/yilong/QWen_main/baseline/{dataset}/{image_path.split("/")[-1].replace("jpg", "png")}'

        target_size = 1022
        mask_save_path = os.path.join(pred_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.png")
        image, image_tensor, (pad_W, pad_H), (W, H) = load_and_pad_image(image_path, patch_size=patch_size, target_size=target_size)

        resolution = 0.5
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
            resolution=resolution,
            n_neighbors=None,
            alpha=1,
            beta=6, gamma=0.4
        )
        cv2_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        if refine and merge:
            merged_sim_maps = merge_sim_maps(sim_maps_leiden + sim_maps_refine, thresh=0.95, visualisation=False)
        elif refine and not merge:
            merged_sim_maps = sim_maps_refine + sim_maps_leiden
        elif not refine and not merge:
            merged_sim_maps = sim_maps_leiden
        elif not refine and merge:
            merged_sim_maps = merge_sim_maps(sim_maps_leiden, thresh=0.95, visualisation=False)

        # On user systems, sam_pred_path may not exist. We fallback to image shape if file is missing.
        orig_size = (H, W)
        if include_Qwen_pred:
            if os.path.exists(sam_pred_path):
                qwen_pred = cv2.imread(sam_pred_path, flags=0) > 0
                orig_size = qwen_pred.shape
                qwen_pred = cv2.resize(qwen_pred.astype(np.uint8), dsize=(sim_maps_leiden[0].shape[1], 
                            sim_maps_leiden[0].shape[0]), 
                            interpolation=cv2.INTER_NEAREST)

                if boundary_contact(qwen_pred, n=10) > 0.75:
                    qwen_pred = 1 - qwen_pred
                qwen_pred = clean_mask(qwen_pred, min_size=100)
            else:
                # Fallback: compute Qwen prediction mask on-the-fly using SAM and Stage 1 bboxes
                predictor.set_image(cv2_img)
                qwen_masks = []
                # bboxes_qwen has shape [N, 4] where N >= 0. Format: [ymin, xmin, ymax, xmax] scaled to 1000.
                if len(bboxes_qwen) > 0 and bboxes_qwen.ndim >= 2:
                    for box in bboxes_qwen:
                        xmin, ymin, xmax, ymax = box
                        xmin_px = max(0, min(int(xmin), W - 1))
                        ymin_px = max(0, min(int(ymin), H - 1))
                        xmax_px = max(xmin_px + 1, min(int(xmax), W))
                        ymax_px = max(ymin_px + 1, min(int(ymax), H))
                        
                        sam_box = np.array([xmin_px, ymin_px, xmax_px, ymax_px])
                        try:
                            mask, _, _ = predictor.predict(
                                box=sam_box,
                                multimask_output=False
                            )
                            qwen_masks.append(mask[0])
                        except Exception as e:
                            print(f"SAM predict failed for box {sam_box}: {e}")
                
                if len(qwen_masks) > 0:
                    qwen_pred_full = np.any(qwen_masks, axis=0).astype(np.uint8)
                else:
                    qwen_pred_full = np.zeros((H, W), dtype=np.uint8)
                
                qwen_pred = cv2.resize(qwen_pred_full, dsize=(sim_maps_leiden[0].shape[1], 
                            sim_maps_leiden[0].shape[0]), 
                            interpolation=cv2.INTER_NEAREST)

                if boundary_contact(qwen_pred, n=10) > 0.75:
                    qwen_pred = 1 - qwen_pred
                qwen_pred = clean_mask(qwen_pred, min_size=100)

        end_time1 = time.time()
        FOD_time += (end_time1 - start_time)
        candidate_masks_final, candidate_scores_final = get_MAX_IoU_mask_from_SAM(merged_sim_maps, cv2_img, predictor, k=5)
        
        end_time2 = time.time()
        Seg_time += (end_time2 - end_time1)
        if include_Qwen_pred:
            candidate_masks_final.append(qwen_pred)
            candidate_scores_final.append(np.array(candidate_scores_final).mean())

        best_mask = pairwise_selection_from_QWen(np.array(image), candidate_masks_final, candidate_scores_final, QWen_model, QWen_processor, device=device)        
        final_pred = cv2.resize(best_mask.astype(np.uint8), dsize=(orig_size[1], orig_size[0]))
        end_time3 = time.time()
        SMS_time += (end_time3 - end_time2)
        save_mask_as_color(final_pred, mask_save_path)
        pbar.update(1)
        pbar.set_postfix({"current": image_path.split('/')[-1]})
        
    pbar.close()
    total_time += (end_time3 - start_time)
    print(f"Worker {rank} on GPU {gpu_id} finished in {total_time:.2f} seconds. FOD: {FOD_time:.2f}, Seg: {Seg_time:.2f}, SMS: {SMS_time:.2f}")


def run_drs_inference(
    dataset_name, pred_dir_base, json_file, gpus="0", processes_per_gpu=1,
    refine=True, merge=True, include_qwen=False,
    dino_model_path='/data/yilong/hf_dinov2',
    qwen_model_path='/data/yilong/hf_Qwen2.5-VL-7B-Instruct',
    sam2_checkpoint='/data/yilong/sam2/checkpoints/sam2.1_hiera_large.pt',
    sam2_cfg='//data/yilong/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
    local_dataset_dir=None
):
    """
    Launch multiprocessing DRS prediction pipeline.
    """
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    pred_dir = os.path.join(pred_dir_base, dataset_name, f'refine+{refine}_merge+{merge}_include+{include_qwen}')
    print(f'Predictions will be saved to: {pred_dir}')
    os.makedirs(pred_dir, exist_ok=True)

    img_list, bbox_list = get_img_list(json_file, pred_dir, local_dataset_dir=local_dataset_dir)
    gpu_ids = [int(id) for id in gpus.split(',')]
    num_gpus = len(gpu_ids)
    total_processes = num_gpus * processes_per_gpu
    chunk_size = len(img_list) // total_processes + 1
    
    processes = []
    for i in range(total_processes):
        gpu_id = gpu_ids[i % num_gpus]
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(img_list))
        chunk_imgs = img_list[start_idx:end_idx]
        chunk_bboxes = bbox_list[start_idx:end_idx]

        if not chunk_imgs:
            continue

        p = mp.Process(
            target=worker,
            args=(i, gpu_id, dataset_name, chunk_imgs, chunk_bboxes, pred_dir, refine, merge, include_qwen,
                  dino_model_path, qwen_model_path, sam2_checkpoint, sam2_cfg)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("All DRS processes completed successfully!")
