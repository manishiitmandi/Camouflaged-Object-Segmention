"""DINOv2 neural classifiers and IoU/CrossEntropy losses for DSS training."""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


def mask_to_hard_labels_vectorized(mask_np, patch_size=14, ignore_value=2, thresh_pos=0.3):
    """
    Downsample a pixel-level binary mask into patch-level hard labels.
    """
    H, W = mask_np.shape
    Ph, Pw = H // patch_size, W // patch_size

    patches = mask_np.reshape(Ph, patch_size, Pw, patch_size)
    patches = patches.transpose(0, 2, 1, 3)  # -> (Ph, Pw, 14, 14)
    patches = patches.reshape(Ph, Pw, -1)    # (Ph, Pw, 196)

    valid = (patches != ignore_value)
    has_valid = valid.sum(axis=-1) > 0

    values = patches.copy()
    values[~valid] = 0
    ratio = (values == 1).sum(axis=-1).astype(np.float32) / (valid.sum(axis=-1) + 1e-8)

    labels = np.full((Ph, Pw), fill_value=ignore_value, dtype=np.int32)
    labels[has_valid & (ratio >= thresh_pos)] = 1
    labels[has_valid & (ratio <= thresh_pos)] = 0
    upsampled_labels = cv2.resize(labels.astype(np.uint8), (0, 0), fx=14, fy=14, interpolation=cv2.INTER_NEAREST)
    return upsampled_labels


def evaluate(model, test_loader, device, criterion_ce, criterion_iou, num_classes=2, ignore_index=2):
    """
    Evaluate patch classifier model performance metrics (Accuracy, IoU).
    """
    model.eval()
    preds_all = []
    labels_all = []
    loss_ce = 0
    loss_iou = 0
    loss_total = 0

    with torch.no_grad():
        for images, labels, pad_info, names in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss_ce += criterion_ce(logits, labels).item()
            loss_iou += criterion_iou(logits, labels).item()
            loss_total += loss_ce + loss_iou

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            preds_all.append(preds.cpu())
            labels_all.append(labels.to('cpu'))

    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    valid_mask = (labels_all != ignore_index)
    preds_valid = preds_all[valid_mask]
    labels_valid = labels_all[valid_mask]
    
    acc = (preds_valid == labels_valid).float().mean().item()
    iou_list = []
    for cls in range(num_classes):
        pred_class = (preds_valid == cls)
        label_class = (labels_valid == cls)
        intersection = (pred_class & label_class).sum().float()
        union = (pred_class | label_class).sum().float()
        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection + 1e-6) / (union + 1e-6)
        iou_list.append(iou)

    iou_tensor = torch.tensor(iou_list)
    iou_score = iou_tensor.nanmean().item()

    return preds_all, acc, iou_list[1], loss_ce/len(test_loader), loss_iou/len(test_loader), loss_total/len(test_loader)


class IoULoss(nn.Module):
    """
    Soft intersection-over-union loss function supporting ignore values.
    """
    def __init__(self, smooth=1e-6, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
            target = target * valid_mask.long()
        else:
            valid_mask = None

        num_classes = logits.shape[1]
        target = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        pred = F.softmax(logits, dim=1)

        if valid_mask is not None:
            pred = pred * valid_mask.unsqueeze(1)
            target = target * valid_mask.unsqueeze(1)

        intersection = (pred * target).sum(dim=(2, 3))
        union = (pred + target - pred * target).sum(dim=(2, 3))

        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou
        loss = loss.mean()

        return loss


class DINOv2PatchClassifier(nn.Module):
    """
    A freeze-backbone patch-level binary classifier running over DINOv2 representations.
    """
    def __init__(self, num_classes=2, model_name='/data/yilong/hf_dinov2'):
        super().__init__()
        self.model_name = model_name
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.dinov2 = AutoModel.from_pretrained(model_name)

        for param in self.dinov2.parameters():
            param.requires_grad = False

        print(f"Number of frozen parameters: {sum(p.numel() for p in self.dinov2.parameters() if not p.requires_grad)}")

        hidden_size = self.dinov2.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

        print(f"Number of trainable parameters: {sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)}")

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.dinov2(pixel_values)
        patch_features = outputs.last_hidden_state[:, 1:, :]

        H = W = 73
        patch_features = patch_features.view(-1, H, W, patch_features.shape[-1])

        logits = self.classifier(patch_features)
        return logits.permute(0, 3, 1, 2)
