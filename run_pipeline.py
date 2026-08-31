#!/usr/bin/env python3
"""
End-to-End Replication Pipeline Runner for DSS.
Loads configuration from configs/dss_config.yaml and runs Stages 1, 2, and 3.
"""

import os
import argparse
import sys

# Ensure dss package can be imported from current working directory
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import yaml
except ImportError:
    print("Error: 'PyYAML' is required to parse the configuration file.")
    print("Please install it using: pip install pyyaml")
    sys.exit(1)

from dss import run_qwen_inference, run_drs_inference


def load_config(config_path):
    """Load config from a YAML file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="DSS zero-shot Camouflaged Object Segmentation replication pipeline.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/dss_config.yaml", 
        help="Path to dss_config.yaml"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="cod10k", 
        choices=["chameleon", "camo", "cod10k", "nc4k"], 
        help="Dataset name to process"
    )
    parser.add_argument(
        "--stage", 
        type=str, 
        default="all", 
        choices=["1", "2", "all"], 
        help="Stage to execute: 1 (Qwen Bbox Localization), 2 (DRS Segment & Select), or all"
    )
    parser.add_argument(
        "--gpus", 
        type=str, 
        default=None, 
        help="GPUs list (override config value, e.g., '0,1')"
    )
    args = parser.parse_args()

    # Load parameters
    config = load_config(args.config)
    
    # Resolve paths and configs
    model_paths = config.get("model_paths", {})
    datasets = config.get("datasets", {})
    hyperparams = config.get("hyperparameters", {})
    exec_config = config.get("execution", {})

    # Extract target dataset info
    dataset_key = args.dataset.lower()
    if dataset_key not in datasets:
        print(f"Error: Dataset '{args.dataset}' is not defined in datasets configuration.")
        sys.exit(1)
    
    dataset_path = datasets[dataset_key]
    dataset_name = args.dataset.upper()  # chameleon -> CHAMELEON, etc.

    # GPU config overrides
    gpus = args.gpus if args.gpus is not None else str(exec_config.get("gpus", "0"))
    processes_per_gpu = exec_config.get("processes_per_gpu", 1)
    seed = exec_config.get("seed", 42)

    # Output details
    output_dir_base = "outputs"
    localization_json_dir = os.path.join(output_dir_base, dataset_name)
    localization_json_file = os.path.join(localization_json_dir, f"infer_{dataset_name}_Qwen2.5-VL-7B_clean.json")

    print("=" * 70)
    print("DSS Zero-shot COS Pipeline Execution")
    print("=" * 70)
    print(f"Dataset : {dataset_name} (path: {dataset_path})")
    print(f"Stage   : {args.stage}")
    print(f"GPUs    : {gpus}")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Stage 1: MLLM Bounding Box Localization (Qwen Inference)
    # -----------------------------------------------------------------
    if args.stage in ["1", "all"]:
        print("\n>>> Running Stage 1: MLLM Bounding Box Localization...")
        
        # In the original implementation, the parallel script output is named infer_{dataset}_Qwen_7B_clean.json
        # We will use localization_json_file as the output target.
        # Ensure directories exist
        os.makedirs(localization_json_dir, exist_ok=True)
        
        # Call Qwen inference
        run_qwen_inference(
            image_dir=os.path.join(dataset_path, "Image") if os.path.exists(os.path.join(dataset_path, "Image")) else dataset_path,
            model_dir=model_paths.get("qwen2.5_vl"),
            output_dir=localization_json_dir,
            dataset=dataset_name,
            gpus=gpus,
            seed=seed
        )
        
        # Locate the clean output file produced by the runner
        generated_clean_file = os.path.join(localization_json_dir, f"infer_{dataset_name}_QWen_7B_clean.json")
        if os.path.exists(generated_clean_file):
            # Rename or copy to keep name standard
            if os.path.exists(localization_json_file):
                os.remove(localization_json_file)
            os.rename(generated_clean_file, localization_json_file)
            print(f"✓ Stage 1 Complete. Localization JSON saved to: {localization_json_file}")
        else:
            print("Warning: Could not find generated localization JSON.")

    # -----------------------------------------------------------------
    # Stage 2 & 3: DRS Pipeline (FOD + SAM + SMS)
    # -----------------------------------------------------------------
    if args.stage in ["2", "all"]:
        print("\n>>> Running Stage 2 & 3: DRS Pipeline (Segment & Select)...")
        
        if not os.path.exists(localization_json_file):
            # Try fallback to standard output names if Stage 1 was skipped
            alternative_file = os.path.join(localization_json_dir, f"infer_{dataset_name}_QWen_7B_clean.json")
            if os.path.exists(alternative_file):
                localization_json_file = alternative_file
            else:
                print(f"Error: Localization JSON not found at {localization_json_file}")
                print("Please run Stage 1 first to generate the bounding boxes.")
                sys.exit(1)

        # Call DRS inference pipeline
        run_drs_inference(
            dataset_name=dataset_name,
            pred_dir_base=os.path.join(output_dir_base, "preds"),
            json_file=localization_json_file,
            gpus=gpus,
            processes_per_gpu=processes_per_gpu,
            refine=True,
            merge=True,
            include_qwen=True,
            dino_model_path=model_paths.get("dinov2"),
            qwen_model_path=model_paths.get("qwen2.5_vl"),
            sam2_checkpoint=model_paths.get("sam2_vit_l", {}).get("checkpoint"),
            sam2_cfg=model_paths.get("sam2_vit_l", {}).get("config"),
            local_dataset_dir=dataset_path
        )
        print("✓ Stage 2 & 3 Complete. Predictions generated successfully.")

    print("\n" + "=" * 70)
    print("DSS PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
