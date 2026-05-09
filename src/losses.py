"""Losses for binary BEV occupancy with 255-ignore mask."""
import torch
import torch.nn.functional as F

from src.config import IGNORE_INDEX


def masked_bce_with_logits(logits, target, ignore_index=IGNORE_INDEX):
    """BCE-with-logits where target == ignore_index gets zero weight.

    logits: (B, 1, H, W) float
    target: (B, 1, H, W) long with values {0, 1, ignore_index}
    """
    mask = (target != ignore_index).float()
    target_f = target.float() * mask  # set 255→0 for safe BCE input
    loss = F.binary_cross_entropy_with_logits(logits, target_f, reduction="none")
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
