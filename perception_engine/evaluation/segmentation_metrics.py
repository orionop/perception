"""
Segmentation quality metrics.

Computes pixel accuracy, per-class IoU, mean IoU, and a confusion matrix
for evaluating segmentation prediction quality against ground-truth masks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Compute standard segmentation evaluation metrics.

    Args:
        prediction: ``(H, W)`` integer array of predicted class IDs.
        ground_truth: ``(H, W)`` integer array of ground-truth class IDs.
        num_classes: Total number of semantic classes.
        class_names: Optional mapping from class ID to human-readable name,
            used for per-class IoU reporting.

    Returns:
        Dict with keys:
            - ``pixel_accuracy``: overall pixel accuracy (float).
            - ``mean_iou``: macro-averaged IoU across classes (float).
            - ``per_class_iou``: dict of ``{class_name_or_id: iou}`` pairs.
            - ``confusion_matrix``: ``(C, C)`` numpy array.
    """
    # Flatten for vectorised operations.
    pred_flat = prediction.ravel().astype(np.int64)
    gt_flat = ground_truth.ravel().astype(np.int64)

    # --- Confusion matrix ---
    confusion = _confusion_matrix(pred_flat, gt_flat, num_classes)

    # --- Pixel accuracy ---
    correct = np.diag(confusion).sum()
    total = confusion.sum()
    pixel_accuracy = float(correct / total) if total > 0 else 0.0

    # --- Per-class IoU ---
    per_class_iou: Dict[str, float] = {}
    iou_values = []

    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = tp + fp + fn

        iou = float(tp / denom) if denom > 0 else float("nan")

        label = (
            class_names.get(c, str(c)) if class_names else str(c)
        )
        per_class_iou[label] = iou

        if np.isfinite(iou):
            iou_values.append(iou)

    # --- Mean IoU (ignoring classes not present in GT or pred) ---
    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0

    # --- Frequency-weighted IoU (fwIoU) ---
    # Weights each class IoU by its frequency in the ground truth.
    # Useful for imbalanced datasets where one class dominates.
    class_freq = confusion.sum(axis=1)  # per-class pixel count in GT
    total_pixels = class_freq.sum()
    fw_iou = 0.0
    if total_pixels > 0:
        for c in range(num_classes):
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            denom = tp + fp + fn
            if denom > 0:
                iou_c = tp / denom
                fw_iou += (class_freq[c] / total_pixels) * iou_c
        fw_iou = float(fw_iou)

    # --- Dice coefficient (macro-averaged) ---
    dice_values = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        if denom > 0:
            dice_values.append(float(2 * tp / denom))
    mean_dice = float(np.mean(dice_values)) if dice_values else 0.0

    logger.info(
        "Metrics: pixel_acc=%.4f, mIoU=%.4f, fwIoU=%.4f, dice=%.4f",
        pixel_accuracy, mean_iou, fw_iou, mean_dice,
    )

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "frequency_weighted_iou": fw_iou,
        "dice_coefficient": mean_dice,
        "per_class_iou": per_class_iou,
        "confusion_matrix": confusion,
    }


def _confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Build a (num_classes, num_classes) confusion matrix.

    Element ``[i, j]`` counts the number of pixels with ground-truth
    class ``i`` predicted as class ``j``.
    """
    assert pred.shape == gt.shape, (
        f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
    )

    # Mask out any invalid class IDs.
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    gt_valid = gt[valid]
    pred_valid = pred[valid]

    # Use np.bincount for a fast histogram.
    indices = gt_valid * num_classes + pred_valid
    cm = np.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)
