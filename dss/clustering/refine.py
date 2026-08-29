"""Part composition and clustering refinement utilities for DSS."""

import math
import numpy as np
import cv2
from scipy.ndimage import generic_filter

from dss.utils.general import mode_filter, min_max_normalize


def draw_star_point(img, centers, labels):
    """Draw star markings representing positive/negative labels on the visualization image."""
    print(f"num of points sampled: {len(labels)}, {img.shape}")
    outer_radius = 10
    inner_radius = 3
    n_points = 5
    for center, label in zip(centers, labels):
        points = []
        for i in range(2 * n_points):
            angle = i * math.pi / n_points
            radius = inner_radius if i % 2 == 1 else outer_radius
            x = center[0] + radius * math.cos(angle - math.pi/2)
            y = center[1] + radius * math.sin(angle - math.pi/2)
            points.append((int(x), int(y)))

        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(img, [pts], color=(255, 0, 0) if label == 1 else (0, 255, 0))
    return img


def spatial_smoothness_8n(labels):
    """Calculate the 8-neighborhood spatial smoothness of patch label mappings."""
    height, width = labels.shape
    total = 0
    same = 0

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

    for i in range(height):
        for j in range(width):
            for dx, dy in neighbors:
                ni, nj = i + dx, j + dy
                if 0 <= ni < height and 0 <= nj < width:
                    total += 1
                    if labels[i, j] == labels[ni, nj]:
                        same += 1

    return same / total if total > 0 else 0


def compute_fisher_score(X, labels):
    """Compute Fisher score between two label classes."""
    X1 = X[labels == 0]
    X2 = X[labels == 1]
    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)
    var1 = X1.var(axis=0)
    var2 = X2.var(axis=0)
    numerator = np.square(mean1 - mean2)
    denominator = var1 + var2 + 1e-8
    fisher_scores = numerator / denominator
    average_fisher_score = np.mean(fisher_scores)
    return average_fisher_score


def compute_energy(y, X, knn_indices, alpha=1.0, beta=1.0):
    """Compute the joint spatial-coherence and intra/inter-class energy of labels."""
    fg_mask = (y > 0.5)
    bg_mask = (y < 0.5)
    energy = 0.0
    
    inter_class_sep = 0.0
    if np.any(fg_mask) and np.any(bg_mask):
        center_fg = X[fg_mask].mean(axis=0)
        center_bg = X[bg_mask].mean(axis=0)
        inter_class_sep = alpha * np.linalg.norm(center_fg - center_bg)
        energy -= inter_class_sep
    
    inner_class = 0.0
    for i in range(len(y)):
        neighbor_diff = np.mean(np.abs(y[i] - y[knn_indices[i]]))
        inner_class += neighbor_diff
    energy += beta * inner_class / len(y)
    return energy


def calculate_candidate_mask_score(data, sort_directions):
    """Calculate normalized sum rank scores based on custom feature sorting directions."""
    data = min_max_normalize(data)
    n_samples, n_features = data.shape
    scores = np.zeros_like(data, dtype=float)
    if len(sort_directions) != n_features:
        raise ValueError("sort_directions length must match features number")
    for feature_idx in range(n_features):
        feature_values = data[:, feature_idx]
        ascending = sort_directions[feature_idx]
        if ascending:
            scores[:, feature_idx] = 1 - feature_values
        else:
            scores[:, feature_idx] = feature_values    
    scores = scores.sum(axis=1)
    return scores


def smooth_and_unpad(labels_map, pad_H, pad_W, size=5, smooth=True):
    """Resize label map to original resolution and unpad it to exact image boundaries."""
    if smooth:
        smoothed_label_image_low_res = generic_filter(labels_map, function=mode_filter, size=size)
    else:
        smoothed_label_image_low_res = labels_map
    smoothed_label_image = cv2.resize(smoothed_label_image_low_res, dsize=None, fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
    
    if pad_H == 0:
        if pad_W == 0:
            smoothed_unpad_label = smoothed_label_image
        else:
            smoothed_unpad_label = smoothed_label_image[:, :-pad_W]
    else:
        if pad_W == 0:
            smoothed_unpad_label = smoothed_label_image[:-pad_H, :]
        else:
            smoothed_unpad_label = smoothed_label_image[:-pad_H, :-pad_W]
    return smoothed_unpad_label, smoothed_label_image_low_res, smoothed_label_image
