"""Bounding box operations for DSS."""

import cv2
import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage import label, generate_binary_structure
from skimage import morphology
from skimage.morphology import disk


def get_rotated_box(mask):
    """Find rotated bounding boxes for each connected contour in grey mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rotated_boxes = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        rect = cv2.minAreaRect(cnt)  # (center, (w,h), angle)
        box = cv2.boxPoints(rect)    # 4 vertices
        box = box.astype(int)
        box = box.reshape((-1, 2))
        rotated_boxes.append(box)
    return rotated_boxes


def filter_bboxes(bboxes, iou_threshold=0.9):
    """Filter out boxes that are covered by another box by more than iou_threshold."""
    def box_area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def intersection_area(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right <= x_left or y_bottom <= y_top:
            return 0
        return (x_right - x_left) * (y_bottom - y_top)

    keep = []
    for i, box1 in enumerate(bboxes):
        area1 = box_area(box1)
        remove = False
        for j, box2 in enumerate(bboxes):
            if i == j:
                continue
            inter_area = intersection_area(box1, box2)
            if area1 > 0 and (inter_area / area1) >= iou_threshold:
                remove = True
                break
        if not remove:
            keep.append(box1)
    return keep


def get_peaks(heatmap, bbox, threshold_abs=0.75, min_distance=56):
    """Detect local maxima (peaks) in the specified bbox area of a heatmap."""
    if bbox is not None:
        x1, y1, x2, y2 = bbox
    else:
        x1 = 0
        y1 = 0
        x2 = heatmap.shape[0]
        y2 = heatmap.shape[1]
        
    bbox_heatmap = heatmap[y1:y2, x1:x2]
    peaks = peak_local_max(
        bbox_heatmap,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=True
    )
    global_peaks = peaks + np.array([y1, x1])  # (y, x) order
    global_peaks = np.flip(global_peaks, axis=1)  # flip to (x, y)
    return global_peaks


def expand_bbox(bbox, img_width=None, img_height=None, ratio=0.05):
    """Expand bounding box by a given ratio, optionally clamping to image bounds."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    dw = int(w * ratio)
    dh = int(h * ratio)

    new_x1 = x1 - dw
    new_y1 = y1 - dh
    new_x2 = x2 + dw
    new_y2 = y2 + dh

    return [new_x1, new_y1, new_x2, new_y2]


def adaptive_dual_threshold_growth(heatmap):
    """Adaptive dual-threshold region growing using morphology.reconstruction."""
    hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    low_ratio = 0.65
    high_ratio = 0.8
    strong_mask = hm_norm >= high_ratio
    weak_mask   = hm_norm >= low_ratio

    final_mask = morphology.reconstruction(
        seed=strong_mask.astype(np.uint8),
        mask=weak_mask.astype(np.uint8),
        method='dilation'
    )
    radius = 5
    kernel = disk(radius)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    return final_mask.astype(np.uint8), strong_mask.astype(np.uint8), weak_mask.astype(np.uint8)


def bbox_from_sim_maps(sim_maps):
    """Generate list of bounding boxes from list of similarity maps."""
    structure = generate_binary_structure(2, 2)
    all_boxes = []
    for heatmap in sim_maps:
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
        
        filtered_bboxes = filter_bboxes(filtered_bboxes, iou_threshold=0.9)
        if len(filtered_bboxes) < 20:
            all_boxes.append(filtered_bboxes)
            
    all_boxes = [bbox for group in all_boxes for bbox in group]
    return all_boxes


def compute_iou_matrix(boxes):
    """Compute IoU matrix between all pairs of bounding boxes."""
    n = len(boxes)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            box1 = boxes[i]
            box2 = boxes[j]
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            
            if x_right < x_left or y_bottom < y_top:
                iou = 0.0
            else:
                inter = (x_right - x_left) * (y_bottom - y_top)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                iou = inter / (area1 + area2 - inter + 1e-8)
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    return iou_matrix


def merge_boxes_iterative(boxes, iou_thresh=0.9):
    """Merge overlapping bounding boxes iteratively."""
    if len(boxes) <= 1:
        return boxes
    merged_boxes = []
    used = set()
    for i in range(len(boxes)):
        if i in used:
            continue
        curr_box = list(boxes[i])
        used.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            # Calculate overlapping IoU
            x_left = max(curr_box[0], boxes[j][0])
            y_top = max(curr_box[1], boxes[j][1])
            x_right = min(curr_box[2], boxes[j][2])
            y_bottom = min(curr_box[3], boxes[j][3])
            
            if x_right >= x_left and y_bottom >= y_top:
                inter = (x_right - x_left) * (y_bottom - y_top)
                area1 = (curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])
                area2 = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                union = area1 + area2 - inter
                iou = inter / (union + 1e-8)
                if iou > iou_thresh:
                    # Merge box
                    curr_box[0] = min(curr_box[0], boxes[j][0])
                    curr_box[1] = min(curr_box[1], boxes[j][1])
                    curr_box[2] = max(curr_box[2], boxes[j][2])
                    curr_box[3] = max(curr_box[3], boxes[j][3])
                    used.add(j)
        merged_boxes.append(curr_box)
    return merged_boxes


def find_best_match_box(bboxes_ref, candidates, img_size):
    """
    Find which candidate bbox list has the maximum IoU coverage against bboxes_ref.
    """
    from dss.utils.masks import bboxes_to_mask, compute_iou
    mask_ref = bboxes_to_mask(bboxes_ref, img_size)
    best_idx, best_iou = -1, 0.0
    scores = []
    for i, bboxes in enumerate(candidates):
        mask_cand = bboxes_to_mask(bboxes, img_size)
        iou = compute_iou(mask_ref, mask_cand)
        scores.append(iou)
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx, best_iou, scores


def resize_bbox(bbox, orig_size, new_size):
    """Map bbox coordinates from original image size to new image size."""
    x1, y1, x2, y2 = bbox
    w, h = orig_size
    new_w, new_h = new_size
    scale_x = new_w / w
    scale_y = new_h / h
    new_x1 = x1 * scale_x
    new_y1 = y1 * scale_y
    new_x2 = x2 * scale_x
    new_y2 = y2 * scale_y
    return [new_x1, new_y1, new_x2, new_y2]
    
    
def resize_bboxes(bboxes: list, original_size: tuple, new_size: tuple) -> list:
    """Resize a list of bounding boxes."""
    return [resize_bbox(bbox, original_size, new_size) for bbox in bboxes]
