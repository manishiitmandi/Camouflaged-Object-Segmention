#!/usr/bin/env python3
import os
from tqdm import tqdm
from datasets import load_dataset, Image

def main():
    dataset_path = "datasets/NC4K"
    image_dir = os.path.join(dataset_path, "Image")
    gt_dir = os.path.join(dataset_path, "GT")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print(f"Loading local NC4K dataset...")
    # Load all splits: train, validation, test
    splits = ["train", "validation", "test"]
    
    total_images = 0
    for split in splits:
        print(f"Processing split: {split}...")
        ds_data = load_dataset(dataset_path, split=split)
        ds_paths = ds_data.cast_column("image", Image(decode=False))

        num_samples = len(ds_data)
        for idx in tqdm(range(num_samples), desc=f"Saving {split} split"):
            item_data = ds_data[idx]
            item_path = ds_paths[idx]

            # Extract filename (e.g. '2856.jpg')
            original_path = item_path["image"]["path"]
            filename = os.path.basename(original_path)
            base_name, _ = os.path.splitext(filename)

            # Save Image
            img = item_data["image"]
            img.save(os.path.join(image_dir, filename))

            # Save GT Mask (ensure png format)
            mask = item_data["gt"]
            mask.save(os.path.join(gt_dir, base_name + ".png"))
            
            total_images += 1

    print(f"\n✓ NC4K extraction complete!")
    print(f"Total extracted images: {total_images}")
    print(f"Images saved to: {image_dir}")
    print(f"GT masks saved to: {gt_dir}")

if __name__ == "__main__":
    main()
