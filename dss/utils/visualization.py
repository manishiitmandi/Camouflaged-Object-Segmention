"""Visualization and plotting utilities for DSS."""

import math
import cv2
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label, generate_binary_structure


def show_image(fig, ax, img, title="", show_bar=True):
    """Plot clustering images with a colorbar."""
    img = np.squeeze(img)
    n_classes = len(np.unique(img))
    colors = matplotlib.colormaps["Set3"]
    cmap = ListedColormap(colors(np.linspace(0, 1, n_classes)))
    img_plot = ax.imshow(img, cmap=cmap, vmin=0, vmax=n_classes-1)
    
    if show_bar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(img_plot, cax=cax)
        cbar.set_ticks(range(n_classes))
        cbar.set_label("Clusters")
        
    ax.set_title(title)
    ax.axis("off")


def visualize_merging_process(original_masks, image, merged_masks, merge_groups, confidences):
    """Visualize the mask merging process in a grid."""
    from dss.utils.general import softmax
    structure = generate_binary_structure(2, 2)
    confidences = softmax(confidences)
    
    n_original = len(original_masks)
    n_merged = len(merged_masks)
    
    fig, axes = plt.subplots(math.ceil((n_merged+n_original+1)/6), 6, figsize=(24, 15))
    axes = axes.flatten()
    
    axes[0].imshow(image)
    axes[0].axis('off')
    
    for i in range(1, n_original+n_merged+1):
        if i < n_original+1:
            axes[i].imshow(original_masks[i-1], cmap='jet')
            labeled_array, num_features = label(original_masks[i-1], structure=structure)
            axes[i].set_title(f'Mask {i-1}, {num_features} comps')
        else:
            axes[i].imshow(merged_masks[i-n_original-1], cmap='jet')
            labeled_array, num_features = label(merged_masks[i-n_original-1], structure=structure)
            group_str = ','.join(map(str, merge_groups[i-n_original-1]))
            axes[i].set_title(f'Group {i-n_original-1}\n({group_str}, {num_features} comps)')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def draw_bboxes_watershed(heatmap, threshold_ratio=0.6, min_area=42, show=True):
    """
    Apply watershed segmentation to extract bounding boxes around high confidence regions.
    """
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    thresh_val = int(threshold_ratio * heatmap_norm.max())
    _, binary = cv2.threshold(heatmap_norm, int(thresh_val), 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(heatmap_norm, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    img_with_boxes = img_color.copy()
    bboxes = []
    for label_idx in range(2, num_labels+1):
        mask = (markers == label_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h < min_area:
                continue
            bboxes.append((x, y, x+w, y+h))
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title("Watershed Result")
        plt.axis("off")
        plt.show()

    return img_with_boxes, bboxes


def save_mask_as_color(mask, save_path, colormap='jet'):
    """Save binary mask as a color image (saved as white on black)."""
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = mask.astype(np.uint8)
    Image.fromarray(mask * 255).save(save_path)


def sample_heatmap_points(heatmap, step=24, high_thresh=0.7, low_thresh=0.3):
    """Grid sample heatmap points into positive/negative coordinates."""
    H, W = heatmap.shape
    coords = []
    labels = []
    for y in range(0, H, step):
        for x in range(0, W, step):
            val = heatmap[y, x]
            if val > high_thresh:
                coords.append((x, y))
                labels.append(1)
            elif val < low_thresh:
                coords.append((x, y))
                labels.append(0)
    return np.array(coords), np.array(labels)


def sample_heatmap_points_with_bboxes(heatmap, bboxes, step=42, high_thresh=0.8, low_thresh=0.3):
    """Sparse sample heatmap points matching bounding box inclusion criteria."""
    H, W = heatmap.shape
    coords = []
    labels = []

    def in_any_bbox(x, y):
        for (x1, y1, x2, y2) in bboxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    for y in range(0, H, step):
        for x in range(0, W, step):
            val = heatmap[y, x]
            inside = in_any_bbox(x, y)

            if inside:
                if val > high_thresh:
                    coords.append((x, y))
                    labels.append(1)
                elif val < low_thresh:
                    coords.append((x, y))
                    labels.append(0)

    return np.array(coords), np.array(labels)
