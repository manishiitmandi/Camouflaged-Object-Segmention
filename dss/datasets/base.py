"""Dataset classes for DSS model training and inference."""

import os
import random
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2

from dss.utils.image import resize_and_pad
from dss.models.classifiers import mask_to_hard_labels_vectorized


class Pseudo_Dataset(Dataset):
    """
    Dataset wrapper for training DINOv2PatchClassifier on pseudo mask targets.
    """
    def __init__(self, image_dir, pseudo_mask_dir, target_size=1022, is_train=False):
        train_ratio = 0.8
        self.image_dir = image_dir
        self.mask_dir = pseudo_mask_dir
        self.is_train = is_train
        
        self.items = []
        for fname in sorted(os.listdir(image_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(self.image_dir, fname)
                mask_path = os.path.join(self.mask_dir, base + ".png")
                if os.path.exists(mask_path):
                    self.items.append((img_path, mask_path, base))
        
        shuffled_items = self.items.copy()
        random.shuffle(shuffled_items)
        split_idx = int(len(shuffled_items) * train_ratio)
        if is_train:
            self.items = shuffled_items[:split_idx]
        else:
            self.items = shuffled_items[split_idx:]

        print(f"Loaded {len(self.items)} image-mask pairs")
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if self.is_train:
            import albumentations as A
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_REFLECT),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ])

        self.final_transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, base_name = self.items[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask_np_binary = (np.array(mask) > 128).astype(np.int32)
        mask_pil = Image.fromarray(mask_np_binary)

        image_pil, mask_pil, pad_info = resize_and_pad(image, mask_pil, 1022, ignore_value=2)
        mask_np = np.array(mask_pil)
        
        labels = mask_to_hard_labels_vectorized(
            mask_np,
            patch_size=14,
            ignore_value=2,
            thresh_pos=0.3,
        )
        
        if self.is_train:
            augmented = self.transform(image=np.array(image_pil), mask=labels)
            image_pil = augmented['image']
            labels = augmented['mask']

        image_tensor = self.final_transform(image_pil)
        downsampled_labels = cv2.resize(labels.astype(np.uint8), (0, 0), fx=1/14, fy=1/14, interpolation=cv2.INTER_NEAREST)
        labels = torch.from_numpy(downsampled_labels).long()

        return image_tensor, labels, pad_info, base_name


class infer_Dataset(Dataset):
    """
    Inference dataset wrapper for evaluation datasets.
    """
    def __init__(self, image_dir, target_size=1022):
        self.image_dir = image_dir
        self.items = []
        for fname in sorted(os.listdir(image_dir)):
            if "NonCAM" in fname:
                break
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(self.image_dir, fname)
                self.items.append((img_path, base))

        print(f"Loaded {len(self.items)} inference images")
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.final_transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, base_name = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        image_pil, _, pad_info = resize_and_pad(image, image, 1022, ignore_value=2)
        image_tensor = self.final_transform(image_pil)
        return image_tensor, pad_info, base_name, np.array(image.size)


class Sam_refine_Dataset(Dataset):
    """
    SAM refinement dataset loading original images and pseudo masks.
    """
    def __init__(self, image_dir, pseudo_mask_dir):
        self.image_dir = image_dir
        self.mask_dir = pseudo_mask_dir

        self.items = []
        for fname in sorted(os.listdir(self.image_dir)):
            if "NonCAM" in fname:
                break
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                base = os.path.splitext(fname)[0]
                img_path = os.path.join(self.image_dir, fname)
                mask_path = os.path.join(self.mask_dir, base + ".jpg")
                if os.path.exists(mask_path):
                    self.items.append((img_path, mask_path, base))

        print(f"Loaded {len(self.items)} SAM refinement pairs")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, base_name = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        mask = Image.open(mask_path).convert("L")
        mask_np_binary = (np.array(mask) > 128).astype(np.int32)
        return image, mask_np_binary, base_name
