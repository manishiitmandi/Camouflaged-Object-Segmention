"""Leiden clustering utilities for DSS."""

import time
import numpy as np
import cv2
import torch
import anndata
import scanpy as sc
import torch.nn.functional as F
from scipy.ndimage import generic_filter, label
from sklearn.metrics.pairwise import cosine_similarity

from dss.utils.general import mode_filter
from dss.clustering.refine import smooth_and_unpad, compute_energy


def get_leiden_label(data, leiden_key, pad_H, pad_W, resolution=0.5):
    """
    Perform PCA-based Leiden clustering on DINOv2 feature data.
    """
    n_features, h, w = data.shape
    C, H, W = data.shape
    n_neighbors = int((H*W)**0.5)
    n_pca = 64
    coords = np.mgrid[0:H, 0:W].reshape(2, -1).T
    X = data.reshape(n_features, -1).T
    
    print(f'resolution:{resolution}, n_neighbors:{n_neighbors}, n_pca:{n_pca}')
    adata = anndata.AnnData(X)
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=64)
    adata.obsm['spatial'] = coords
    
    sc.pp.neighbors(adata, 
        use_rep='X_pca', 
        n_neighbors=n_neighbors, 
        n_pcs=n_pca,
        method='gauss',
        metric='euclidean'
    )

    sc.tl.leiden(adata, resolution=resolution, key_added=leiden_key)
    labels = adata.obs[leiden_key].astype(int).values
    labels_map = labels.reshape(h, w)
    smoothed_label_image = generic_filter(labels_map, function=mode_filter, size=5)
    smoothed_label_image = cv2.resize(smoothed_label_image, dsize=None, fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
    
    if pad_H == 0:
        if pad_W == 0:
            return smoothed_label_image, adata
        else:
            return smoothed_label_image[:, :-pad_W], adata
    else:
        if pad_W == 0:
            return smoothed_label_image[:-pad_H, :], adata
        else:
            return smoothed_label_image[:-pad_H, :-pad_W], adata


def get_candidate_fg_clusters(image):
    """Identify potential foreground clusters based on component connectivity."""
    unique_values = np.unique(image)
    structure_8 = np.ones((3, 3), dtype=int)
    candidate_clusters = []
    for target_value in unique_values:
        mask = (image == target_value).astype(int)
        labeled_array, num_component = label(mask, structure=structure_8)
        if num_component < 50:
            candidate_clusters.append(target_value)
    return candidate_clusters


def get_sim_map(adata, label_img, fg_cluster, H, W, pad_H, pad_W):
    """Generate normalized cosine similarity map for a given cluster."""
    X_PCA = torch.from_numpy(adata.obsm['X_pca'].copy())
    if fg_cluster is None:
        idx_bool = (label_img.ravel() != fg_cluster)
        X_PCA_filtered = X_PCA
    else:
        idx_bool = (label_img.ravel() == fg_cluster)
        X_PCA_filtered = X_PCA[idx_bool]
    regional_mean_feat = X_PCA_filtered.mean(dim=0, keepdim=True)
    similarity = F.cosine_similarity(regional_mean_feat, X_PCA).reshape(H, W).numpy()
    sim_map, _, _ = smooth_and_unpad(similarity, pad_H, pad_W, smooth=False)
    sim_map_norm = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    return sim_map_norm


def get_leiden_labels_sim_maps(data, pad_H, pad_W, resolution=0.5, n_neighbors=None, n_pca=64):
    """Compute initial similarity maps directly from Leiden clusters."""
    C, H, W = data.shape
    X = data.reshape(C, -1).T
    sim_maps = []

    if n_neighbors is None:
        n_neighbors = int((H * W) ** 0.5)

    adata = anndata.AnnData(X.copy())
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_pca)
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=n_neighbors, n_pcs=n_pca, method='gauss')
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden_init')
    
    initial_labels = adata.obs['leiden_init'].astype(int).values
    labels_map = initial_labels.reshape(H, W)
    smoothed_label_image, init_smoothed_label_image_low_res, smoothed_padded = smooth_and_unpad(labels_map, pad_H, pad_W)
    candidate_fgs = get_candidate_fg_clusters(smoothed_label_image)
    
    for fg_cluster in candidate_fgs:
        sim_map_leiden = get_sim_map(adata, init_smoothed_label_image_low_res, fg_cluster, H, W, pad_H, pad_W)
        sim_maps.append(sim_map_leiden)
    return sim_maps


def get_patch_level_hierarchical_labels(
    data,
    pad_H, pad_W,
    resolution=0.5,
    n_neighbors=None,
    n_pca=64,
    alpha=1.0,
    beta=1.5,
    gamma=0.5
):
    """
    Perform hierarchical patch-level optimization to refine clustering foregrounds.
    """
    from dss.utils.masks import remove_small_regions

    C, H, W = data.shape
    X = data.reshape(C, -1).T
    coords = np.mgrid[0:H, 0:W].reshape(2, -1).T
    N = H * W
    
    leiden_iter_labels = []
    leiden_iter_labels_low_res = []
    smoothed_padded_masks = []
    sim_maps_leiden_list = []
    sim_maps_refine_list = []

    energies = []
    if n_neighbors is None:
        n_neighbors = int((H * W) ** 0.5)

    adata = anndata.AnnData(X.copy())
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=n_pca)
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=n_neighbors, n_pcs=n_pca, method='gauss')
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden_init', flavor="igraph")

    initial_labels = adata.obs['leiden_init'].astype(int).values
    labels_map = initial_labels.reshape(H, W)
    leiden_map, init_smoothed_label_image_low_res, smoothed_padded = smooth_and_unpad(labels_map, pad_H, pad_W)
    candidate_fgs = get_candidate_fg_clusters(leiden_map)
    threshold = int(init_smoothed_label_image_low_res.shape[0]*init_smoothed_label_image_low_res.shape[1]*0.01)

    leiden_iter_labels_low_res.append(init_smoothed_label_image_low_res)
    smoothed_padded_masks.append(smoothed_padded)

    dist_spatial = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    spatial_weight = np.exp(-dist_spatial ** 2 / (2 * (beta)**2))
    feat_sim = cosine_similarity(X)
    sim_joint = feat_sim * spatial_weight
    np.fill_diagonal(sim_joint, -1)
    knn_indices = np.argsort(sim_joint, axis=1)[:, -int(np.sqrt(N)):]

    for fg_cluster in candidate_fgs:
        y = (init_smoothed_label_image_low_res.reshape(-1) == fg_cluster).astype(float)
        delta = 10 
        it = 1
        while delta > 1:
            fg_mask = (y > 0.5)
            bg_mask = (y < 0.5)
            
            if np.any(fg_mask) and np.any(bg_mask):
                center_fg = np.average(adata.obsm['X_pca'][fg_mask], axis=0, weights=y[fg_mask])
                center_bg = np.average(adata.obsm['X_pca'][bg_mask], axis=0, weights=1 - y[bg_mask])
            else:
                center_fg = adata.obsm['X_pca'][fg_mask].mean(axis=0) if np.any(fg_mask) else adata.obsm['X_pca'].mean(axis=0)
                center_bg = adata.obsm['X_pca'][bg_mask].mean(axis=0) if np.any(bg_mask) else adata.obsm['X_pca'].mean(axis=0)

            d_fg = np.linalg.norm(adata.obsm['X_pca'] - center_fg, axis=1)
            d_bg = np.linalg.norm(adata.obsm['X_pca'] - center_bg, axis=1)
            score_feat = d_bg - d_fg

            neighbor_labels = y[knn_indices]
            score_smooth = neighbor_labels.mean(axis=1)

            total_score = alpha * score_feat + beta * score_smooth
            y_new = 1.0 / (1.0 + np.exp(-total_score))
            y = y_new

            energy = compute_energy(y, X, knn_indices)
            energies.append(energy)
            it += 1
            if len(energies) >= 2:
                delta = np.abs(energies[-1] - energies[-2])
                if delta < 1:
                    binary_labels = (y > 0.5).astype(int)
                    labels_map_it = binary_labels.reshape(H, W)
                    labels_map_it = remove_small_regions(labels_map_it, min_size=threshold)

                    smoothed_label_image, smoothed_label_image_low_res, smoothed_padded = smooth_and_unpad(labels_map_it, pad_H, pad_W)
                    
                    if len(np.unique(smoothed_label_image_low_res)) == 1:
                        continue
                    
                    sim_map_leiden = get_sim_map(adata, init_smoothed_label_image_low_res, fg_cluster, H, W, pad_H, pad_W)
                    sim_map_leiden = cv2.GaussianBlur(sim_map_leiden, (5, 5), sigmaX=0)
                    sim_maps_leiden_list.append(sim_map_leiden)
                    
                    candidate_fgs_updated.append(fg_cluster)
                    leiden_iter_labels.append(smoothed_label_image)
                    leiden_iter_labels_low_res.append(smoothed_label_image_low_res)
                    smoothed_padded_masks.append(smoothed_padded)

                    sim_map_opt = get_sim_map(adata, smoothed_label_image_low_res, 1, H, W, pad_H, pad_W)
                    sim_map_opt = cv2.GaussianBlur(sim_map_opt, (5, 5), sigmaX=0)
                    sim_maps_refine_list.append(sim_map_opt)

    pca_data = np.reshape(adata.obsm['X_pca'], [H, W, n_pca])
    return leiden_iter_labels, candidate_fgs_updated, sim_maps_leiden_list, sim_maps_refine_list, smoothed_padded_masks, leiden_iter_labels_low_res, leiden_map, pca_data
