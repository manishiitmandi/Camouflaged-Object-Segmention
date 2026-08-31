#!/usr/bin/env python3
import os
from tqdm import tqdm
from datasets import load_dataset, Image

def main():
    dataset_name = "nobg/COD10K"
    split_name = "test"
    output_dir = "datasets/COD10K"
    image_dir = os.path.join(output_dir, "Image")
    gt_dir = os.path.join(output_dir, "GT")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print(f"Loading '{dataset_name}' ({split_name} split) from Hugging Face...")
    
    # We load the dataset with decode=True to get PIL Images, but we also load it with decode=False
    # (or cast the column) to extract the original file paths/filenames.
    # To keep it extremely simple and fast, we can load the dataset normally (decode=True)
    # and load a path-only version with decode=False in parallel.
    ds_data = load_dataset(dataset_name, split=split_name)
    ds_paths = ds_data.cast_column("image", Image(decode=False)) # Cast to Image(decode=False) to avoid decoding and get path dict

    num_samples = len(ds_data)
    print(f"Downloading and extracting {num_samples} test images and ground truth masks...")

    for idx in tqdm(range(num_samples), desc="Saving files"):
        item_data = ds_data[idx]
        item_path = ds_paths[idx]

        # Extract filename (e.g. 'COD10K-CAM-1-Aquatic-1-BatFish-2.jpg')
        original_path = item_path["image"]["path"]
        filename = os.path.basename(original_path)
        base_name, _ = os.path.splitext(filename)

        # Save Image
        img = item_data["image"]
        img.save(os.path.join(image_dir, filename))

        # Save GT Mask (ensure png format)
        mask = item_data["mask"]
        mask.save(os.path.join(gt_dir, base_name + ".png"))

    print(f"\n✓ Download and extraction complete!")
    print(f"Images saved to: {image_dir}")
    print(f"GT masks saved to: {gt_dir}")

if __name__ == "__main__":
    main()
