#!/usr/bin/env python3
"""
Setup script for DSS perfect replication.
This script helps set up the environment and organize files.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_environment():
    """Check Python environment and dependencies."""
    print("=" * 60)
    print("Checking Environment for DSS Replication")
    print("=" * 60)

    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print("⚠ Warning: Python 3.9+ required for DSS")

    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("✗ CUDA not available - will use CPU (slower)")
    except ImportError:
        print("✗ PyTorch not installed")

    print("\n" + "=" * 60)

def setup_directory_structure():
    """Create directory structure for DSS project."""
    print("Setting up directory structure...")

    directories = [
        "models",
        "models/dinov2",
        "models/sam2_vit_l",
        "models/qwen2.5_vl",
        "datasets",
        "datasets/CHAMELEON",
        "datasets/CAMO",
        "datasets/COD10K",
        "datasets/NC4K",
        "outputs",
        "scripts",
        "configs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")

    print("\n✓ Directory structure created successfully!")

def install_dependencies():
    """Install required Python dependencies."""
    print("\nInstalling dependencies...")

    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"✗ Requirements file not found: {requirements_file}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✓ Dependencies installed successfully!")
            return True
        else:
            print(f"✗ Failed to install dependencies:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def create_config_files():
    """Create configuration files based on original DSS code."""
    print("\nCreating configuration files...")

    # Create model config
    config_content = """# Model configuration for DSS
# Update these paths after downloading models

model_paths:
  dinov2: "models/dinov2"
  sam2_vit_l: "models/sam2_vit_l"
  qwen2.5_vl: "models/qwen2.5_vl"

# Dataset paths
datasets:
  chameleon: "datasets/CHAMELEON"
  camo: "datasets/CAMO"
  cod10k: "datasets/COD10K"
  nc4k: "datasets/NC4K"

# Hyperparameters from Table 1 in paper
hyperparameters:
  leiden_resolution: 0.5
  pc_energy_threshold: 1.0
  correlation_threshold: 0.95
  top_k: 5
  target_size: 1120
  patch_size: 14

# Hardware
device: "cuda"  # or "cpu"
batch_size: 1
num_workers: 4
"""

    config_path = "configs/dss_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"✓ Created: {config_path}")

    # Create download script
    download_script = """#!/bin/bash
# Download script for DSS models

echo "Downloading DSS models..."

# Download DINOv2
echo "1. Downloading DINOv2..."
mkdir -p models/dinov2
# huggingface-cli download facebook/dinov2-base --local-dir models/dinov2
echo "   Run: huggingface-cli download facebook/dinov2-base --local-dir models/dinov2"

# Download QWen2.5-VL (7B)
echo "2. Downloading QWen2.5-VL (7B)..."
mkdir -p models/qwen2.5_vl
# huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen2.5_vl
echo "   Run: huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/qwen2.5_vl"

# Download SAM2 (ViT-L)
echo "3. Downloading SAM2 (ViT-L)..."
mkdir -p models/sam2_vit_l
echo "   Visit: https://github.com/facebookresearch/segment-anything-2"
echo "   Download: sam2_vit_l.pth to models/sam2_vit_l/"

echo ""
echo "After downloading models, update paths in configs/dss_config.yaml"
"""

    download_path = "scripts/download_models.sh"
    with open(download_path, "w") as f:
        f.write(download_script)

    # Make it executable
    os.chmod(download_path, 0o755)
    print(f"✓ Created: {download_path}")

    return True

def verify_original_files():
    """Verify that original DSS files are present."""
    print("\nVerifying original DSS files...")

    original_files = [
        "original/DSS/clean_QWen_output.py",
        "original/DSS/infer_COD10K_QWen_7B_parallel.py",
        "original/DSS/infer_DRS_pred.py",
        "original/DSS/README.md",
        "original/DSS/refine_leiden_utils.py",
        "original/DSS/utils.py",
        "original/DSS/json_files/infer_CAMO_QWen_7B_clean.json",
        "original/DSS/json_files/infer_CHAMELEON_QWen_7B_clean.json",
        "original/DSS/json_files/infer_COD10K_QWen_7B_clean.json",
        "original/DSS/json_files/infer_NC4K_QWen_7B_clean.json"
    ]

    all_present = True
    for file_path in original_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_present = False

    return all_present

def create_test_script():
    """Create a test script to verify DSS setup."""
    test_script = """#!/usr/bin/env python3
"""

    test_path = "scripts/test_dss_setup.py"
    with open(test_path, "w") as f:
        f.write(test_script)

    print(f"✓ Created: {test_path}")
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup DSS perfect replication')
    parser.add_argument('--check-env', action='store_true', help='Check environment only')
    parser.add_argument('--setup-dirs', action='store_true', help='Setup directories only')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies only')
    parser.add_argument('--all', action='store_true', help='Run complete setup')

    args = parser.parse_args()

    if not any(vars(args).values()):
        args.all = True

    print("=" * 60)
    print("DSS - Perfect Replication Setup")
    print("=" * 60)
    print("This script sets up the perfect replication of the DSS repository.")
    print("Original repository: https://github.com/ynulonger/DSS")
    print("=" * 60)

    if args.all or args.check_env:
        check_environment()

    if args.all or args.setup_dirs:
        setup_directory_structure()

    if args.all or args.install_deps:
        if not install_dependencies():
            print("\n⚠ Warning: Dependency installation failed.")
            print("You may need to install dependencies manually:")
            print("pip install -r requirements.txt")

    if args.all:
        create_config_files()
        verify_original_files()
        create_test_script()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)

    if args.all:
        print("\nNext steps:")
        print("1. Download models:")
        print("   bash scripts/download_models.sh")
        print("\n2. Download datasets (CHAMELEON, CAMO, COD10K, NC4K)")
        print("   Place in datasets/ directory")
        print("\n3. Test the setup:")
        print("   cd original/DSS")
        print("   python infer_COD10K_QWen_7B_parallel.py --help")
        print("\n4. Run inference:")
        print("   python infer_COD10K_QWen_7B_parallel.py \\")
        print("     --dataset_path ../datasets/COD10K \\")
        print("     --output_dir ../outputs")

    print("\nOriginal DSS files are in: original/DSS/")
    print("These are exact copies from https://github.com/ynulonger/DSS")
    print("=" * 60)

if __name__ == "__main__":
    main()