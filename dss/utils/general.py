"""General utilities for DSS."""

import os
import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
from sklearn.metrics.pairwise import cosine_similarity


def setup_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def mode_filter(x):
    """Find the mode (most frequent value) of an array."""
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts)]


def min_max_normalize(data):
    """Linearly scale each column to [0,1] range."""
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    return (data - mins) / (maxs - mins + 1e-8)


def binary_spatial_entropy_map(binary_image, window_size=3):
    """
    Calculate the local spatial entropy map of a binary image using OpenCV.
    Returns (entropy, num_features).
    """
    structure = generate_binary_structure(2, 2)
    img = binary_image.astype(np.uint8)
    labeled_array, num_features = label(img, structure=structure)
    
    # Fast convolution using OpenCV's boxFilter
    count_1 = cv2.boxFilter(img.astype(np.float32), -1, (window_size, window_size), 
                           borderType=cv2.BORDER_CONSTANT)
    count_1 *= window_size * window_size  # Convert average back to count
    
    total_pixels = window_size * window_size
    count_0 = total_pixels - count_1
    
    p0 = count_0 / total_pixels
    p1 = count_1 / total_pixels
    entropy = np.zeros_like(p0, dtype=np.float32)
    
    mask = (p0 > 0) & (p1 > 0)
    entropy[mask] = -p0[mask] * np.log2(p0[mask]) - p1[mask] * np.log2(p1[mask])
    return entropy, num_features


def heatmap_spatial_entropy_map_fast(heatmap, window_size=9, bins=16):
    """
    Efficient calculation of a local spatial entropy map for a heatmap.
    """
    h, w = heatmap.shape
    entropy_map = np.zeros((h, w), dtype=np.float32)

    heatmap = heatmap.astype(np.float32)
    # Normalize to [0, bins-1]
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap_norm = (heatmap - h_min) / (h_max - h_min + 1e-6)
    heatmap_idx = np.floor(heatmap_norm * (bins - 1)).astype(np.uint8)

    total_pixels = window_size * window_size

    probs = np.zeros((h, w, bins), dtype=np.float32)
    for b in range(bins):
        mask = (heatmap_idx == b).astype(np.float32)
        count = cv2.boxFilter(mask, ddepth=-1, ksize=(window_size, window_size), 
                              normalize=False, borderType=cv2.BORDER_REFLECT)
        probs[:, :, b] = count / total_pixels

    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.log2(probs + 1e-12)
        entropy_map = -np.sum(probs * logp, axis=-1)

    return entropy_map


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    """Compute sigmoid values for x."""
    return 1 / (1 + np.exp(-x))


def mask_sim(list_of_masks):
    """Compute cosine similarity matrix between a list of flattened masks."""
    flattened_masks = np.array([mask.ravel() for mask in list_of_masks])
    cosine_sim_matrix = cosine_similarity(flattened_masks)
    return cosine_sim_matrix


def sim_of_sim_maps(sim_maps):
    """Plot and return similarity of similarity maps matrix."""
    M = mask_sim(sim_maps) 
    plt.figure(figsize=(10, 10))
    plt.imshow(M, cmap='jet')
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            text_color = 'white' if M[i, j] < 0.5 else 'black'
            _ = plt.text(j, i, f'{M[i, j]:.3f}', 
                    ha='center', va='center', 
                    color=text_color, 
                    fontsize=10)
    plt.show()
    return M


def strict_components_from_similarity(M, thresh=0.5):
    """
    Find components of similarity where all pairwise elements exceed thresh.
    """
    n = M.shape[0]
    comps = []
    used = set()

    for i in range(n):
        if i in used:
            continue
        comp = [i]
        for j in range(n):
            if j != i and j not in used:
                if all(M[j, k] > thresh for k in comp):
                    comp.append(j)
        for k in comp:
            used.add(k)
        comps.append(comp)

    return comps


def heatmap_correlation_matrix(heatmaps, show=False):
    """Calculate the correlation coefficient matrix for a list of heatmaps."""
    flat_maps = np.array([hm.flatten() for hm in heatmaps])  
    corr_matrix = np.corrcoef(flat_maps)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(corr_matrix, cmap='jet')
        rows, cols = corr_matrix.shape
        for i in range(rows):
            for j in range(cols):
                text_color = 'white' if corr_matrix[i, j] < 0.5 else 'black'
                _ = plt.text(j, i, f'{corr_matrix[i, j]:.3f}', 
                        ha='center', va='center', 
                        color=text_color, 
                        fontsize=10)
        plt.show()
    return corr_matrix


def merge_sim_maps(sim_maps, thresh=0.89, visualisation=False):
    """Merge similarity maps based on their correlation matrix."""
    M = heatmap_correlation_matrix(sim_maps, show=visualisation) 
    pairs = strict_components_from_similarity(M, thresh=thresh)
    merged_sim_maps = []
    if visualisation:
        fig, axs = plt.subplots(1, len(pairs), figsize=(15, 8))
        if len(pairs) == 1:
            axs = [axs]
        else:
            axs = axs.flatten()
        for ax, pair in zip(axs, pairs):
            maps_to_merge = [sim_maps[idx] for idx in pair]
            max_map = np.max(maps_to_merge, axis=0)
            merged_sim_maps.append(max_map)
            _ = ax.imshow(max_map, cmap='jet')
            _ = ax.set_title(f"ids: {pair}")
        plt.show()
    else:
        for pair in pairs:
            maps_to_merge = [sim_maps[idx] for idx in pair]
            max_map = np.max(maps_to_merge, axis=0)
            merged_sim_maps.append(max_map)
    return merged_sim_maps
