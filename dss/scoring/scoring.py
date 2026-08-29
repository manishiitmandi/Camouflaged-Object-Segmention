"""Scoring and metric voting algorithms for DSS candidate masks selection."""

import numpy as np
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy, label, regionprops

from dss.utils.general import binary_spatial_entropy_map, heatmap_spatial_entropy_map_fast, softmax
from dss.utils.masks import calculate_iou_matrix, clean_mask


def boundary_contact(mask, n=10):
    """
    Calculate the ratio of the mask contacting the image boundaries.
    """
    H, W = mask.shape
    n = min(n, H // 2, W // 2)
    top_edge = mask[0:n, :]
    bottom_edge = mask[H-n:H, :]
    left_edge = mask[n:H-n, 0:n]
    right_edge = mask[n:H-n, W-n:W]
    
    top_contact = np.sum(top_edge > 0)
    bottom_contact = np.sum(bottom_edge > 0)
    left_contact = np.sum(left_edge > 0)
    right_contact = np.sum(right_edge > 0)
    
    contact_pixels = top_contact + bottom_contact + left_contact + right_contact
    total_edge_pixels = (n * W) + (n * W) + (n * (H - 2 * n)) + (n * (H - 2 * n))
    return contact_pixels / total_edge_pixels


def texture_scores(mask, image, P=8, R=1.0):
    """Calculate texture scores based on Local Binary Patterns (LBP) entropy."""
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    inside_vals = gray[mask > 0]
    outside_vals = gray[mask == 0]

    if len(inside_vals) == 0 or len(outside_vals) == 0:
        return 0.0

    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    inside_entropy = shannon_entropy(lbp[mask > 0])
    outside_entropy = shannon_entropy(lbp[mask == 0])
    lbp_score = inside_entropy / (outside_entropy + 1e-6)
    return lbp_score


def compare_masks(masks, image):
    """Compare a list of masks based on LBP texture scores."""
    results = []
    for i, m in enumerate(masks):
        var_s = texture_scores(m, image)
        results.append((i, var_s))
    return results


def scoring(mask, image):
    """Compute overall heuristic score vector [boundary_contact, entropy, texture_scores]."""
    bc = boundary_contact(mask, n=10)
    entropy_map, num_comp = binary_spatial_entropy_map(mask, window_size=7)
    ts = texture_scores(mask, image)
    entropy = np.mean(entropy_map[mask == 1])
    score = [bc, entropy, ts]
    return score


def calculate_spatial_chaos(mask):
    """Determine spatial chaos score of mask combining fragmentation and spatial entropy."""
    if np.sum(mask) == 0:
        return 0.0, 1.0
    try:
        entropy_map = heatmap_spatial_entropy_map_fast(mask, window_size=5)
        if np.any(np.isnan(entropy_map)):
            entropy = 0.5
        else:
            if np.any(mask == 1):
                entropy_foreground = np.mean(entropy_map[mask == 1])
            else:
                entropy_foreground = 0
                
            if np.any(mask == 0):
                entropy_background = np.mean(entropy_map[mask == 0])
            else:
                entropy_background = 0
                
            entropy = entropy_foreground + entropy_background
            entropy = min(max(entropy, 0), 1)

        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)
        
        n_components = len(regions)
        fragmentation = min(n_components / 20.0, 1.0)
        chaos_score = 0.5 * fragmentation + 0.5 * entropy
        if np.isnan(chaos_score) or np.isinf(chaos_score):
            chaos_score = 0.5
        chaos_score = min(max(chaos_score, 0), 1)
        confidence = 1 - chaos_score
        
        if np.isnan(confidence) or np.isinf(confidence):
            confidence = 0.5
            chaos_score = 0.5
        return confidence, chaos_score
        
    except Exception as e:
        print(f"Error calculating spatial chaos: {e}")
        return 0.5, 0.5


def calculate_all_confidences(masks):
    """Compute confidence and chaos scores for a list of masks."""
    confidences = []
    chaos_scores = []
    for mask in masks:
        confidence, chaos = calculate_spatial_chaos(mask)
        confidences.append(confidence)
        chaos_scores.append(chaos)
    return np.array(confidences), np.array(chaos_scores)


def merge_masks_with_soft_voting(masks, heatmaps, image, iou_threshold=0.5, confidence_threshold=0.0):
    """Merge highly overlapping masks based on weighted soft voting."""
    n_masks = len(masks)
    if n_masks == 0:
        return [], [], []
    
    iou_matrix = calculate_iou_matrix(masks)
    confidences, chaos_scores = calculate_all_confidences(masks)
    confidences = softmax(confidences)
    parent = list(range(n_masks))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            if confidences[root_x] > confidences[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_x] = root_y
    
    for i in range(n_masks):
        for j in range(i + 1, n_masks):
            if (iou_matrix[i, j] > iou_threshold and 
                confidences[i] > confidence_threshold and 
                confidences[j] > confidence_threshold):
                union(i, j)
    
    groups = {}
    for i in range(n_masks):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    merged_masks = []
    merge_groups = []
    
    for group_indices in groups.values():
        if len(group_indices) == 0:
            continue
        merged_mask = soft_vote_merge(masks, heatmaps, image, group_indices, confidences)
        if merged_mask is None:
            continue
        merged_masks.append(merged_mask)
        merge_groups.append(group_indices)
    
    return merged_masks, merge_groups, confidences, iou_matrix


def soft_vote_merge(masks, heatmaps, image, group_indices, confidences):
    """Merge a cluster of masks weighted by their respective confidences."""
    if not group_indices:
        return np.zeros_like(masks[0], dtype=np.float32), 0.0
    
    H, W = masks[0].shape
    vote_map = np.zeros((H, W), dtype=np.float32)
    
    group_confidences = np.array([confidences[i] for i in group_indices])
    normalized_weights = softmax(group_confidences)
    total_confidence = 0.0
    
    for idx, weight in zip(group_indices, normalized_weights):
        vote_map += masks[idx].astype(np.float32) * weight
        total_confidence += weight
        
    if total_confidence > 0:
        vote_map /= total_confidence
        
    merged_mask = vote_map > 0.5
    cleaned_mask = clean_mask(merged_mask, min_size=100)

    if len(np.unique(cleaned_mask)) == 1:
        return None
    else:
        return cleaned_mask
