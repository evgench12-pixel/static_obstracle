"""Losses for binary BEV occupancy with 255-ignore mask."""
import torch
import torch.nn.functional as F

from src.config import IGNORE_INDEX


def masked_bce_with_logits(logits, target, ignore_index=IGNORE_INDEX, pos_weight=None):
    """BCE-with-logits where target == ignore_index gets zero weight.

    logits: (B, 1, H, W) float
    target: (B, 1, H, W) long with values {0, 1, ignore_index}
    pos_weight: scalar weight on the occupied (positive) class — >1 pushes
        recall on obstacles, useful since the metric averages free + occupied IoU.
    """
    mask = (target != ignore_index).float()
    target_f = target.float() * mask  # set 255→0 for safe BCE input
    pw = None if pos_weight is None else torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target_f, reduction="none", pos_weight=pw)
    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
    return loss


def soft_iou_loss(logits, target, ignore_index=IGNORE_INDEX, eps=1e-6):
    """Soft-IoU loss on the occupied class. Optional addition to BCE."""
    probs = torch.sigmoid(logits)
    mask = (target != ignore_index).float()
    target_f = target.float() * mask
    probs = probs * mask
    inter = (probs * target_f).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target_f.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return 1.0 - iou.mean()


def soft_dice_loss(logits, target, ignore_index=IGNORE_INDEX, eps=1e-6):
    """Per-batch averaged soft Dice on the occupied class."""
    probs = torch.sigmoid(logits)
    mask = (target != ignore_index).float()
    target_f = target.float() * mask
    probs = probs * mask
    inter = (probs * target_f).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target_f.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits, target, ignore_index=IGNORE_INDEX, pos_weight=None,
                  dice_weight=1.0):
    """BCE + Dice combined. Dice handles class imbalance directly; BCE keeps
    pixel-level calibration."""
    bce = masked_bce_with_logits(logits, target, ignore_index, pos_weight=pos_weight)
    dice = soft_dice_loss(logits, target, ignore_index)
    return bce + dice_weight * dice
