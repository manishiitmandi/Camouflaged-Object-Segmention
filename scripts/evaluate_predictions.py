#!/usr/bin/env python3
"""
Evaluation Script for Zero-shot Camouflaged Object Segmentation.
Calculates Structure-measure (S_alpha), Enhanced-alignment measure (E_phi), 
Weighted F-measure (F_beta^w), and Mean Absolute Error (MAE) using PySODMetrics.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm

try:
    from pysodmetrics import Smeasure, Emeasure, Fmeasure, MAE, WeightedFmeasure
except ImportError:
    try:
        from py_sod_metrics import Smeasure, Emeasure, Fmeasure, MAE, WeightedFmeasure
    except ImportError:
        print("Error: 'pysodmetrics' library is required to calculate evaluation metrics.")
        print("Please install it using: pip install pysodmetrics")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted camouflaged object segmentation masks.")
    parser.add_argument(
        "--pred_dir", 
        type=str, 
        default="outputs/preds/CAMO/refine+True_merge+True_include+True", 
        help="Path to directory containing predicted binary masks (.png)"
    )
    parser.add_argument(
        "--gt_dir", 
        type=str, 
        default="datasets/CAMO/GT", 
        help="Path to directory containing ground truth masks"
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir
    gt_dir = args.gt_dir

    if not os.path.exists(pred_dir):
        print(f"Error: Prediction directory '{pred_dir}' does not exist.")
        sys.exit(1)
    if not os.path.exists(gt_dir):
        print(f"Error: GT directory '{gt_dir}' does not exist.")
        sys.exit(1)

    # Find prediction files
    pred_files = [f for f in os.listdir(pred_dir) if f.lower().endswith('.png')]
    if not pred_files:
        print(f"No prediction (.png) images found in '{pred_dir}'")
        sys.exit(1)

    print(f"Evaluating {len(pred_files)} prediction masks against ground truth...")

    # Initialize metric accumulators
    fm = Fmeasure()
    sm = Smeasure()
    em = Emeasure()
    mae = MAE()
    wfm = WeightedFmeasure()

    successful_count = 0

    for fname in tqdm(pred_files, desc="Evaluating"):
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        
        # Fallback to match different extensions if needed (e.g. .jpg GT vs .png pred)
        if not os.path.exists(gt_path):
            base = os.path.splitext(fname)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                temp_path = os.path.join(gt_dir, base + ext)
                if os.path.exists(temp_path):
                    gt_path = temp_path
                    break

        if not os.path.exists(gt_path):
            print(f"Warning: No matching GT mask found for prediction '{fname}' (looked in: {gt_path})")
            continue

        # Read images in grayscale
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred_mask is None or gt_mask is None:
            print(f"Warning: Failed to load '{fname}' or its ground truth.")
            continue

        # Resize prediction to GT size if they differ
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Normalize masks to binary (0 and 255)
        _, gt_binary = cv2.threshold(gt_mask, 128, 255, cv2.THRESH_BINARY)
        _, pred_binary = cv2.threshold(pred_mask, 128, 255, cv2.THRESH_BINARY)

        # Update metrics
        fm.step(pred=pred_binary, gt=gt_binary)
        sm.step(pred=pred_binary, gt=gt_binary)
        em.step(pred=pred_binary, gt=gt_binary)
        mae.step(pred=pred_binary, gt=gt_binary)
        wfm.step(pred=pred_binary, gt=gt_binary)
        
        successful_count += 1

    if successful_count == 0:
        print("Error: No image pairs were successfully matched and evaluated.")
        sys.exit(1)

    # Get results
    fm_res = fm.get_results()
    sm_res = sm.get_results()
    em_res = em.get_results()
    mae_res = mae.get_results()
    wfm_res = wfm.get_results()

    # Retrieve specific metric variants
    s_alpha = sm_res['sm']
    mae_val = mae_res['mae']
    
    # E-measure (adp stands for adaptive threshold, curve is the curve of thresholded values)
    e_mean_adp = em_res['em']['adp']
    e_max = em_res['em']['curve'].max()
    
    # F-measure and Weighted F-measure
    f_max = fm_res['fm']['curve'].max()
    weighted_f_beta = wfm_res['wfm']

    print("\n" + "=" * 55)
    print("Zero-Shot Camouflaged Object Segmentation Evaluation")
    print("=" * 55)
    print(f"Structure-measure (S_alpha)   : {s_alpha:.4f}")
    print(f"Mean Absolute Error (MAE)     : {mae_val:.4f}")
    print(f"Enhanced-alignment (E_mean)   : {e_mean_adp:.4f}")
    print(f"Enhanced-alignment (E_max)    : {e_max:.4f}")
    print(f"Weighted F-measure (Fw_beta)  : {weighted_f_beta:.4f}")
    print(f"Max F-measure (F_max)         : {f_max:.4f}")
    print("=" * 55)
    print(f"Successfully evaluated {successful_count} mask pairs.")


if __name__ == "__main__":
    main()
