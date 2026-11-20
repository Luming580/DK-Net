"""Segmentation metrics with no MedPy dependency."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def binary_dice(prediction: np.ndarray, target: np.ndarray, empty_value: float = 1.0) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    total = prediction.sum() + target.sum()
    if total == 0:
        return float(empty_value)
    return float(2.0 * np.logical_and(prediction, target).sum() / total)


def binary_jaccard(prediction: np.ndarray, target: np.ndarray, empty_value: float = 1.0) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return float(empty_value)
    return float(np.logical_and(prediction, target).sum() / union)


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance for two nonempty masks."""

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() or not target.any():
        raise ValueError("hd95 requires two nonempty masks.")

    prediction_surface = np.logical_xor(prediction, binary_erosion(prediction))
    target_surface = np.logical_xor(target, binary_erosion(target))
    distances_to_target = distance_transform_edt(~target_surface)[prediction_surface]
    distances_to_prediction = distance_transform_edt(~prediction_surface)[target_surface]
    return float(np.percentile(np.concatenate((distances_to_target, distances_to_prediction)), 95))


def synapse_class_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Match the empty-mask convention used by common Synapse baselines."""

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.any() and target.any():
        return binary_dice(prediction, target), hd95(prediction, target)
    if prediction.any() and not target.any():
        return 1.0, 0.0
    return 0.0, 0.0
