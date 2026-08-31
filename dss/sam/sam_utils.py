"""Segment Anything (SAM) utility functions for DSS."""

import cv2
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy.stats import pearsonr

from dss.utils import adaptive_dual_threshold_growth, merge_boxes_iterative, clean_mask, draw_bboxes_watershed
from dss.scoring.scoring import boundary_contact


def get_candidate_mask_from_SAM(sim_maps, image, predictor):
    """Generate candidate segmentations from SAM given similarity maps."""
    predictor.set_image(image)
    structure = generate_binary_structure(2, 2)
    predictions = []
    binary_maps = []
    
    for index, heatmap in enumerate(sim_maps):
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0) 
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        binary_map, _, _ = adaptive_dual_threshold_growth(heatmap)
        
        binary_maps.append(binary_map)
        labeled_array, num_features = label(binary_map, structure=structure)
        filtered_bboxes = []
        for idx in range(1, num_features + 1):
            coords = np.column_stack(np.where(labeled_array == idx))
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)
            if (max_x-min_x)*(max_y-min_y) >= heatmap.shape[0]*heatmap.shape[0]*0.001:
                filtered_bboxes.append(bbox)
        
        if len(filtered_bboxes) >= 1:
            filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)

        if len(filtered_bboxes) <= 12 and len(filtered_bboxes) > 0:
            PRED = np.empty([0, image.shape[0], image.shape[1]])
            for bbox in filtered_bboxes:
                mask, iou_scores, low_res_masks = predictor.predict(
                        box=bbox, 
                        mask_input=cv2.resize(heatmap, dsize=(256, 256))[None, ...],
                        multimask_output=False
                )
                PRED = np.concatenate([PRED, mask], axis=0)
            combined_mask = np.any(PRED, axis=0)
            mask = combined_mask
            mask = clean_mask(mask, min_size=100)
            bcr = boundary_contact(mask, n=10)
            if bcr > 0.75:
                continue
            predictions.append(mask)
    return predictions, binary_maps


def top_k_masks(masks, scores, k=3):
    """Sort and retrieve the top K highest scoring masks."""
    scores = np.array(scores)
    topk_indices = scores.argsort()[::-1][:k]
    topk_masks = [masks[i] for i in topk_indices]
    topk_scores = [scores[i] for i in topk_indices]
    return topk_masks, topk_scores


def get_MAX_IoU_mask_from_SAM(sim_maps, image, predictor, k=3):
    """Generate SAM candidate masks and rank them based on correlation & edge contact."""
    predictor.set_image(image)
    structure = generate_binary_structure(2, 2)
    predictions = []
    binary_maps = []
    scores = []
    
    for index, heatmap in enumerate(sim_maps):
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0) 
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        binary_map, _, _ = adaptive_dual_threshold_growth(heatmap)
        
        binary_maps.append(binary_map)
        labeled_array, num_features = label(binary_map, structure=structure)
        filtered_bboxes = []
        for idx in range(1, num_features + 1):
            coords = np.column_stack(np.where(labeled_array == idx))
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)

            if (max_x-min_x)*(max_y-min_y) >= heatmap.shape[0]*heatmap.shape[0]*0.01:
                filtered_bboxes.append(bbox)
        
        if len(filtered_bboxes) >= 1:
            filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)
        filtered_bboxes = np.array(filtered_bboxes)

        if len(filtered_bboxes) > 0:
            mask, iou_scores, low_res_masks = predictor.predict(
                    box=filtered_bboxes, 
                    mask_input=cv2.resize(heatmap, dsize=(256, 256))[None, ...],
                    multimask_output=False
            )
            combined_mask = np.any(mask, axis=0)
            if len(combined_mask.shape) == 3:
                mask = combined_mask[0]
            else:
                mask = combined_mask
            bcr = boundary_contact(mask, n=10)
            if bcr > 0.75:
                mask = 1 - mask
                heatmap = 1 - heatmap

            mask = clean_mask(mask, min_size=100)
            if mask.sum() == 0:
                continue
            corr, p_val = pearsonr(heatmap.flatten(), mask.flatten())
            bc = 1 - boundary_contact(mask, n=10)
            score = corr + bc
            scores.append(score)
            predictions.append(mask)     
    topk_preds, topk_scores = top_k_masks(predictions, scores, k=k) 
    return topk_preds, topk_scores


def get_MAX_IoU_mask_from_SAM_bbox_from_binary_map(refined_masks, candidate_fgs, leiden_map, image, predictor):
    """Generate SAM masks from refined predictions and Leiden clusters."""
    predictor.set_image(image)
    structure = generate_binary_structure(2, 2)
    predictions = []
    
    for refined_mask in refined_masks:
        labeled_array_1, num_features_1 = label(refined_mask, structure=structure)
        filtered_bboxes = []

        for idx in range(1, num_features_1+1):
            coords = np.column_stack(np.where(labeled_array_1 == idx))
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)

            if (max_x-min_x)*(max_y-min_y) >= image.shape[0]*image.shape[0]*0.01:
                filtered_bboxes.append(bbox)
        
        if len(filtered_bboxes) >= 1:
            filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)
        filtered_bboxes = np.array(filtered_bboxes)

        if len(filtered_bboxes) > 0:
            mask, iou_scores, low_res_masks = predictor.predict(
                    box=filtered_bboxes, 
                    multimask_output=False
            )
            combined_mask = np.any(mask, axis=0)
            if len(combined_mask.shape) == 3:
                mask = combined_mask[0]
            else:
                mask = combined_mask
            bcr = boundary_contact(mask, n=10)
            if bcr > 0.75:
                mask = 1 - mask

            mask = clean_mask(mask, min_size=100)
            if mask.sum() == 0:
                continue
            predictions.append(mask)  

    for candidate_fg in np.unique(leiden_map):
        labeled_array_2, num_features_2 = label(leiden_map==candidate_fg, structure=structure)
        filtered_bboxes = []
        for idx in range(1, num_features_2+1):
            coords = np.column_stack(np.where(labeled_array_2 == idx))
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)

            if (max_x-min_x)*(max_y-min_y) >= image.shape[0]*image.shape[0]*0.01:
                filtered_bboxes.append(bbox)
        
        if len(filtered_bboxes) >= 1:
            filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)
        filtered_bboxes = np.array(filtered_bboxes)

        if len(filtered_bboxes) > 0:
            mask, iou_scores, low_res_masks = predictor.predict(
                    box=filtered_bboxes, 
                    multimask_output=False
            )
            combined_mask = np.any(mask, axis=0)
            if len(combined_mask.shape) == 3:
                mask = combined_mask[0]
            else:
                mask = combined_mask
            bcr = boundary_contact(mask, n=10)
            if bcr > 0.75:
                mask = 1 - mask

            mask = clean_mask(mask, min_size=100)
            if mask.sum() == 0:
                continue
            predictions.append(mask)  

    return predictions, None


def get_one_candidate_mask_from_SAM(sim_map, image, Candidate_mask, predictor):
    """Helper method to construct a single combined candidate mask prediction from SAM."""
    predictor.set_image(image)
    structure = generate_binary_structure(2, 2)
    heatmap = cv2.GaussianBlur(sim_map, (15, 15), 0) 
    heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
    binary_map, _, _ = adaptive_dual_threshold_growth(heatmap)
    labeled_array, num_features = label(binary_map, structure=structure)
    
    filtered_bboxes = []
    for idx in range(1, num_features + 1):
        coords = np.column_stack(np.where(labeled_array == idx))
        count = len(coords)
        if count >= 14*14*3:
            min_y, max_y = coords[:, 0].min(), coords[:, 0].max()
            min_x, max_x = coords[:, 1].min(), coords[:, 1].max()
            bbox = (min_x, min_y, max_x, max_y)
            filtered_bboxes.append(bbox)
    
    _, watershed_bboxes = draw_bboxes_watershed(sim_map, threshold_ratio=0.55, min_area=28, show=False)
    filtered_bboxes.extend(watershed_bboxes)
    filtered_bboxes = merge_boxes_iterative(np.array(filtered_bboxes), iou_thresh=0.9)
    filtered_bboxes = np.array(filtered_bboxes)
    
    if len(filtered_bboxes) > 20:
        return None
        
    mask, _, _ = predictor.predict(
            box=filtered_bboxes, 
            mask_input=cv2.resize(heatmap, dsize=(256, 256))[None, ...],
            multimask_output=False
    )
    if len(mask.shape) == 3:
        mask = mask[None, ...]
    combined_mask = np.any(mask, axis=0)[0]
    return combined_mask
