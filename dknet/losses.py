"""Training losses and class-weight helpers for DK-Net."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss, including the background class."""

    def __init__(self, num_classes: int, smooth: float = 1e-5) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits [N, C, H, W], got {tuple(logits.shape)}.")
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                "Target must have shape [N, H, W] matching logits; "
                f"got logits={tuple(logits.shape)}, target={tuple(target.shape)}."
            )

        probabilities = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target.long(), self.num_classes).permute(0, 3, 1, 2).float()
        dimensions = (0, 2, 3)
        intersection = torch.sum(probabilities * target_one_hot, dim=dimensions)
        denominator = torch.sum(probabilities.square() + target_one_hot.square(), dim=dimensions)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


def make_class_weights(
    num_classes: int,
    target_class: int | None = None,
    target_class_weight: float = 1.0,
) -> torch.Tensor | None:
    """Build optional CrossEntropy class weights for a targeted experiment."""

    if target_class is None:
        return None
    if not 0 <= target_class < num_classes:
        raise ValueError(f"target_class must be in [0, {num_classes - 1}], got {target_class}.")
    if target_class_weight <= 0:
        raise ValueError("target_class_weight must be positive.")

    weights = torch.ones(num_classes, dtype=torch.float32)
    weights[target_class] = float(target_class_weight)
    return weights
