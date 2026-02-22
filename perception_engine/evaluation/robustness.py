"""
Robustness evaluation under image perturbations.

Applies configurable perturbations (brightness, blur, noise, contrast)
to the input image, re-runs inference, and measures the performance
drop relative to the clean baseline.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from perception_engine.data_types import SegmentationOutput
from perception_engine.engine.inference_engine import InferenceEngine
from perception_engine.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------
# Each function takes an (H, W, 3) uint8 numpy array and returns
# a perturbed copy of the same shape and dtype.
# ---------------------------------------------------------------


def _brightness_shift(image: np.ndarray, factor: float = 1.4) -> np.ndarray:
    """Increase or decrease brightness by a multiplicative factor."""
    pil = Image.fromarray(image)
    enhanced = ImageEnhance.Brightness(pil).enhance(factor)
    return np.array(enhanced)


def _gaussian_blur(image: np.ndarray, radius: int = 3) -> np.ndarray:
    """Apply Gaussian blur with the given kernel radius."""
    pil = Image.fromarray(image)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred)


def _gaussian_noise(
    image: np.ndarray, std: float = 25.0
) -> np.ndarray:
    """Add zero-mean Gaussian noise with the given standard deviation."""
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def _contrast_shift(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Reduce or increase contrast by a multiplicative factor."""
    pil = Image.fromarray(image)
    enhanced = ImageEnhance.Contrast(pil).enhance(factor)
    return np.array(enhanced)


# Registry: perturbation name → callable.
PERTURBATION_REGISTRY: Dict[str, Callable[..., np.ndarray]] = {
    "brightness": _brightness_shift,
    "blur": _gaussian_blur,
    "noise": _gaussian_noise,
    "contrast": _contrast_shift,
}

# Default parameters for each perturbation.
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "brightness": {"factor": 1.4},
    "blur": {"radius": 3},
    "noise": {"std": 25.0},
    "contrast": {"factor": 0.5},
}


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------


class RobustnessEvaluator:
    """Evaluates model robustness under image perturbations.

    For each enabled perturbation:
        1. Apply the perturbation to the input image.
        2. Re-run inference via the :class:`InferenceEngine`.
        3. Compute segmentation metrics against ground truth.
        4. Measure the mIoU drop relative to the clean baseline.

    Attributes:
        engine: The inference engine tied to a specific model.
        perturbations: List of perturbation names to apply.
        num_classes: Number of segmentation classes (for metric computation).
    """

    def __init__(
        self,
        engine: InferenceEngine,
        perturbations: List[str],
        num_classes: int,
        class_names: Optional[Dict[int, str]] = None,
        perturbation_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.engine = engine
        self.perturbations = perturbations
        self.num_classes = num_classes
        self.class_names = class_names
        self.perturbation_params = perturbation_params or {}

    def evaluate(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        baseline_miou: float,
    ) -> Dict[str, Dict[str, Any]]:
        """Run robustness evaluation on all configured perturbations.

        Args:
            image: Clean ``(H, W, 3)`` uint8 RGB image.
            ground_truth: ``(H, W)`` ground-truth class IDs.
            baseline_miou: mIoU of the clean (unperturbed) prediction.

        Returns:
            Dict mapping perturbation names to metric dicts containing
            ``pixel_accuracy``, ``mean_iou``, ``miou_drop``, and
            ``per_class_iou``.
        """
        results: Dict[str, Dict[str, Any]] = {}

        for name in self.perturbations:
            fn = PERTURBATION_REGISTRY.get(name)
            if fn is None:
                logger.warning(
                    "Unknown perturbation '%s' — skipping. "
                    "Available: %s",
                    name,
                    list(PERTURBATION_REGISTRY.keys()),
                )
                continue

            logger.info("Applying perturbation: '%s'", name)
            params = {**DEFAULT_PARAMS.get(name, {}), **self.perturbation_params.get(name, {})}
            perturbed = fn(image, **params)

            # Re-run inference.
            seg_output: SegmentationOutput = self.engine.run(perturbed)

            # Resize ground truth to match output mask if needed.
            gt_resized = _resize_gt(ground_truth, seg_output.mask.shape)

            # Compute metrics on perturbed output.
            metrics = compute_segmentation_metrics(
                prediction=seg_output.mask,
                ground_truth=gt_resized,
                num_classes=self.num_classes,
                class_names=self.class_names,
            )

            miou_drop = baseline_miou - metrics["mean_iou"]

            result = {
                "pixel_accuracy": metrics["pixel_accuracy"],
                "mean_iou": metrics["mean_iou"],
                "miou_drop": miou_drop,
                "per_class_iou": metrics["per_class_iou"],
                "inference_time_ms": seg_output.inference_time_ms,
            }
            results[name] = result

            logger.info(
                "Perturbation '%s': mIoU=%.4f (drop=%.4f)",
                name,
                metrics["mean_iou"],
                miou_drop,
            )

        return results


def _resize_gt(
    ground_truth: np.ndarray,
    target_shape: tuple,
) -> np.ndarray:
    """Resize ground truth mask to match prediction shape using nearest
    neighbour interpolation (to preserve class IDs)."""
    if ground_truth.shape == target_shape:
        return ground_truth

    from PIL import Image as PILImage

    gt_pil = PILImage.fromarray(ground_truth.astype(np.uint8))
    gt_resized = gt_pil.resize(
        (target_shape[1], target_shape[0]),
        resample=PILImage.NEAREST,
    )
    return np.array(gt_resized).astype(np.int32)
