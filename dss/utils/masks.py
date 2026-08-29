"""Mask processing and scoring helpers for DSS."""

import numpy as np
from skimage import measure


def clean_mask(mask, min_size=100):
    """Clean mask by removing small connected components and filling small holes."""
    cleaned_mask = remove_small_objects(mask, min_size=min_size)
    cleaned_mask = fill_small_holes(cleaned_mask, area_threshold=min_size)
    return cleaned_mask


def remove_small_objects(mask, min_size=100):
    """Remove small connected components (islands) from binary mask."""
    labeled_mask = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    cleaned_mask = np.zeros_like(mask, dtype=bool)
    for region in regions:
        if region.area >= min_size:
            cleaned_mask[region.coords[:, 0], region.coords[:, 1]] = True
    return cleaned_mask


def fill_small_holes(mask, area_threshold=100):
    """Fill small black holes inside a binary mask."""
    inverted_mask = np.logical_not(mask)
    labeled_holes = measure.label(inverted_mask, connectivity=2)
    regions = measure.regionprops(labeled_holes)
    filled_mask = mask.copy()
    for region in regions:
        if region.area <= area_threshold:
            filled_mask[region.coords[:, 0], region.coords[:, 1]] = True
    return filled_mask


def calculate_iou_matrix(masks):
    """
    Vectorized calculation of IoU between all pairs of masks (only foreground).
    """
    n_masks = len(masks)
    if n_masks == 0:
        return np.array([])
    masks_array = np.array([mask.astype(np.float32).flatten() for mask in masks])
    intersection = np.dot(masks_array, masks_array.T)
    areas = np.sum(masks_array, axis=1)
    union = areas[:, None] + areas[None, :] - intersection
    iou_matrix = np.zeros((n_masks, n_masks))
    mask_nonzero = union > 0
    iou_matrix[mask_nonzero] = intersection[mask_nonzero] / union[mask_nonzero]
    np.fill_diagonal(iou_matrix, 1.0)
    return iou_matrix


def compute_iou(mask1, mask2):
    """Compute standard intersection over union between two binary masks."""
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return inter / union


def find_best_mask_by_iou(mask_list, gt_mask):
    """
    Find the mask in mask_list that has highest IoU with gt_mask.
    Returns: best_mask, best_iou, best_index, list_of_all_ious
    """
    gt_mask = (gt_mask > 0).astype(np.uint8)
    best_iou = -1
    best_mask = None
    best_index = -1
    all_ious = []
    for i, candidate_mask in enumerate(mask_list):
        candidate_mask_binary = (candidate_mask > 0).astype(np.uint8)
        iou = compute_iou(candidate_mask_binary, gt_mask)
        all_ious.append(iou)
        if iou > best_iou:
            best_iou = iou
            best_mask = candidate_mask_binary
            best_index = i
    return best_mask, best_iou, best_index, all_ious


def bboxes_to_mask(bboxes, img_size):
    """Convert a list of bounding boxes [x1, y1, x2, y2] into a binary mask."""
    mask = np.zeros(img_size, dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
    return mask


def remove_small_regions(mask, min_size=20):
    """
    Clean small regions in a binary mask, return int array.
    """
    cleaned_mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    return cleaned_mask.astype(int)
