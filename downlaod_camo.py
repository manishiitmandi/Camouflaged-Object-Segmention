import os
from datasets import load_dataset

# 1. Load the dataset test split from Hugging Face
print("Downloading and loading PassbyGrocer/CAMO (test split)...")
ds = load_dataset("PassbyGrocer/CAMO", split="test")
print("Dataset loaded successfully! Columns:", ds.column_names)

# 2. Define and create output directories
image_dir = "datasets/CAMO/Image"
gt_dir = "datasets/CAMO/GT"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

# 3. Detect column names (usually 'image' and 'label' or 'mask')
image_col = 'image'
mask_col = 'label' if 'label' in ds.column_names else None
if not mask_col:
    for col in ds.column_names:
        if col != 'image':
            mask_col = col
            break

print(f"Extracting files to '{image_dir}' and '{gt_dir}'...")

# 4. Save images and masks preserving their original names
for idx, item in enumerate(ds):
    img = item[image_col]
    mask = item[mask_col]
    
    # Try to extract the original filename (crucial for matching Qwen JSON predictions)
    filename = None
    if hasattr(img, 'filename') and img.filename:
        filename = os.path.basename(img.filename)
    elif 'filename' in item:
        filename = os.path.basename(item['filename'])
    elif 'file_name' in item:
        filename = os.path.basename(item['file_name'])
    
    # Fallback to index-based naming if original filename is not stored
    if not filename:
        filename = f"camo_{idx:05d}.jpg"
        
    mask_filename = os.path.splitext(filename)[0] + ".png"
    
    # Save the PIL Images
    img.save(os.path.join(image_dir, filename))
    mask.save(os.path.join(gt_dir, mask_filename))

print(f"Successfully saved {len(ds)} images and masks!")
