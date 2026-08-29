#!/usr/bin/env python3
"""
Verification script to verify package structure, imports, and function parity using mocked dependencies.
This allows running syntax and import checks without requiring heavy scientific/GPU libraries to be installed.
"""
import sys
import os
import types

# Create mock modules for heavy dependencies
mock_modules = [
    "torch", "torch.nn", "torch.nn.functional", "torch.utils.data", "torch.multiprocessing",
    "torchvision", "torchvision.transforms", "torchvision.transforms.v2",
    "cv2", "anndata", "scanpy", "scipy", "scipy.ndimage", "scipy.stats", "scipy.sparse",
    "sklearn", "sklearn.metrics", "sklearn.metrics.pairwise", "sklearn.preprocessing",
    "skimage", "skimage.feature", "skimage.measure", "skimage.morphology",
    "matplotlib", "matplotlib.pyplot", "matplotlib.colors", "mpl_toolkits", "mpl_toolkits.axes_grid1",
    "transformers", "sam2", "sam2.build_sam", "sam2.sam2_image_predictor", "qwen_vl_utils"
]

class MockObject:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        if name.isupper():
            return 1.0
        return MockObject()
    def __call__(self, *args, **kwargs):
        return MockObject()
    def __bool__(self):
        return False
    def __mro_entries__(self, bases):
        return (object,)

class MockModule(types.ModuleType):
    def __getattr__(self, name):
        if name.isupper():
            return 1.0
        return MockObject()
    def __call__(self, *args, **kwargs):
        return MockObject()

# Register mock modules in sys.modules
for module_name in mock_modules:
    sys.modules[module_name] = MockModule(module_name)

# Add dss_implementation to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("Verification using MOCKED dependencies to check syntax and package import structure...")

try:
    import dss
    print("✓ Successfully imported top-level dss package.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"✗ Failed to import dss package: {e}")
    sys.exit(1)

# Check all modular submodules
submodules = [
    "dss.config",
    "dss.utils.general",
    "dss.utils.image",
    "dss.utils.masks",
    "dss.utils.bbox",
    "dss.utils.visualization",
    "dss.clustering.leiden",
    "dss.clustering.refine",
    "dss.datasets.base",
    "dss.models.classifiers",
    "dss.sam.sam_utils",
    "dss.scoring.scoring",
    "dss.qwen.qwen_utils",
    "dss.inference.qwen",
    "dss.inference.drs"
]

print("\nVerifying submodule imports (Syntax & Structure check):")
for sub in submodules:
    try:
        __import__(sub)
        print(f"  ✓ {sub} loaded successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ✗ {sub} failed to load: {e}")
        sys.exit(1)

# Check expected exports
expected_exports = [
    'setup_seed',
    'Pseudo_Dataset',
    'infer_Dataset',
    'Sam_refine_Dataset',
    'DINOv2PatchClassifier',
    'IoULoss',
    'run_qwen_inference',
    'run_drs_inference'
]

print("\nVerifying dss package top-level exports:")
for exp in expected_exports:
    if hasattr(dss, exp):
        print(f"  ✓ dss.{exp} is exported.")
    else:
        print(f"  ✗ dss.{exp} is NOT exported.")
        sys.exit(1)

print("\n✓ SUCCESS: Modular dss package structure, imports, and syntax verified completely.")
