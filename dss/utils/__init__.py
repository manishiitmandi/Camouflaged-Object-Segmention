"""Utility modules for DSS."""

from .general import (
    setup_seed,
    mode_filter,
    min_max_normalize,
    binary_spatial_entropy_map,
    heatmap_spatial_entropy_map_fast,
    softmax,
    sigmoid,
    mask_sim,
    sim_of_sim_maps,
    strict_components_from_similarity,
    heatmap_correlation_matrix,
    merge_sim_maps,
)

from .image import (
    ResizeLongestSide,
    load_and_pad_image,
    resize_and_pad_1,
    resize_and_pad,
    numpy_to_base64,
    combine_images_cv2_advanced,
)

from .masks import (
    clean_mask,
    remove_small_objects,
    fill_small_holes,
    calculate_iou_matrix,
    compute_iou,
    find_best_mask_by_iou,
    bboxes_to_mask,
    remove_small_regions,
)

from .bbox import (
    get_rotated_box,
    filter_bboxes,
    get_peaks,
    expand_bbox,
    adaptive_dual_threshold_growth,
    bbox_from_sim_maps,
    compute_iou_matrix,
    merge_boxes_iterative,
    find_best_match_box,
    resize_bbox,
    resize_bboxes,
)

from .visualization import (
    show_image,
    visualize_merging_process,
    draw_bboxes_watershed,
    save_mask_as_color,
    sample_heatmap_points,
    sample_heatmap_points_with_bboxes,
)
