"""Image processing utilities for DSS."""

import io
import math
import base64
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms as T


def ResizeLongestSide(image, target_size=980):
    """
    Resize image so that its longest side is target_size, keeping aspect ratio.
    """
    width, height = image.size
    original_longest = max(height, width)
    scale = target_size / original_longest
    
    if height > width:
        new_height = target_size
        new_width = int(width * scale)
    else:
        new_width = target_size
        new_height = int(height * scale)
        
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


def load_and_pad_image(image_path, patch_size=14, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], target_size=1120):
    """
    Load an image, resize it keeping aspect ratio, and pad it to be a multiple of patch_size.
    Returns: (resized_image, padded_tensor, pad_W_and_H, original_size)
    """
    image = Image.open(image_path).convert("RGB")
    Orig_W, Orig_H = image.size
    image = ResizeLongestSide(image, target_size=target_size)
    
    W, H = image.size
    pad_W = (patch_size - W % patch_size) % patch_size
    pad_H = (patch_size - H % patch_size) % patch_size
    
    pad_transform = transforms.Pad((0, 0, pad_W, pad_H), fill=0)
    normalize = transforms.Normalize(mean=mean, std=std)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        pad_transform
    ])
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, pH, pW]
    return image, image_tensor, (pad_W, pad_H), (Orig_W, Orig_H)


def resize_and_pad_1(image: Image.Image, target_size=1022, is_mask=True):
    """
    Resize and pad a single image (or mask).
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    ignore_value = 2 if is_mask else 0
    image = image.resize((new_w, new_h), resample=resample)
    
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    right = target_size - new_w - left
    bottom = target_size - new_h - top
    pad_info = np.array([left, top, right, bottom])
    image = T.Pad((left, top, right, bottom), fill=ignore_value)(image)
    return image, pad_info


def resize_and_pad(image: Image.Image, mask: Image.Image, target_size=1022, ignore_value=2):
    """
    Resize and pad both image and mask keeping aspect ratio.
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    image = image.resize((new_w, new_h), resample=Image.BILINEAR)
    mask = mask.resize((new_w, new_h), resample=Image.NEAREST)
    
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    right = target_size - new_w - left
    bottom = target_size - new_h - top
    pad_info = np.array([left, top, right, bottom])
    
    image = T.Pad((left, top, right, bottom), fill=0)(image)
    mask = T.Pad((left, top, right, bottom), fill=ignore_value)(mask)
    return image, mask, pad_info


def numpy_to_base64(img_array):
    """Convert numpy array (image) to base64 encoded PNG string."""
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def combine_images_cv2_advanced(images_dict, spacing=10, background_color=255, paths=None):
    """
    Concatenate images in a grid, drawing each mask into a single canvas.
    """
    n = len(images_dict)
    if n == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
        
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    first_mask = list(images_dict.values())[0]
    h, w = first_mask.shape[:2]

    total_width = cols * w + (cols - 1) * spacing
    total_height = rows * h + (rows - 1) * spacing
    
    new_image = np.ones((total_height, total_width, 3), dtype=np.uint8)
    new_image[:, :] = (background_color, background_color, background_color)
    
    x_start = 0    
    for i, (key, mask) in enumerate(images_dict.items()):
        row = i // cols
        col = i % cols
        
        y_start = row * (h + spacing)
        x_offset = col * (w + spacing)
        y_offset = 0
        
        y1 = y_start + y_offset
        y2 = y1 + mask.shape[0]
        x1 = x_start + x_offset
        x2 = x1 + mask.shape[1]

        if len(mask.shape) == 3:
            mask_bgr = mask
        else:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        if y2 <= new_image.shape[0] and x2 <= new_image.shape[1]:
            new_image[y1:y2, x1:x2] = mask_bgr
        else:
            print(f"Warning: Image {key} exceeds boundaries")
            
    return new_image
