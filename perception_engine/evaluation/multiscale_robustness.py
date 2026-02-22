"""
Multi-scale robustness evaluation.

Sweeps perturbation severity levels and records how model performance
degrades.  Produces degradation curve data suitable for plotting
mIoU vs. severity for each perturbation type.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from perception_engine.data_types import SegmentationOutput
from perception_engine.engine.inference_engine import InferenceEngine
from perception_engine.evaluation.robustness import PERTURBATION_REGISTRY
from perception_engine.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)

logger = logging.getLogger(__name__)


# Default severity sweeps per perturbation.
DEFAULT_SWEEPS: Dict[str, Dict[str, List]] = {
    "brightness": {"factor": [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 2.0]},
    "blur": {"radius": [1, 2, 3, 5, 7, 10]},
    "noise": {"std": [5.0, 10.0, 15.0, 25.0, 40.0, 60.0]},
    "contrast": {"factor": [0.2, 0.4, 0.5, 0.7, 1.0, 1.5]},
}


class MultiScaleRobustnessEvaluator:
    """Evaluate model robustness at multiple perturbation severities.

    For each perturbation type, sweeps through a list of severity values,
    re-runs inference, and records the resulting metrics to build
    degradation curves.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        num_classes: int,
        class_names: Optional[Dict[int, str]] = None,
        sweeps: Optional[Dict[str, Dict[str, List]]] = None,
    ) -> None:
        self.engine = engine
        self.num_classes = num_classes
        self.class_names = class_names
        self.sweeps = sweeps or DEFAULT_SWEEPS

    def evaluate(
        self,
        image: np.ndarray,
        ground_truth: np.ndarray,
        perturbations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run multi-scale robustness evaluation.

        Args:
            image: Clean ``(H, W, 3)`` uint8 RGB image.
            ground_truth: ``(H, W)`` ground-truth class IDs.
            perturbations: List of perturbation names to evaluate.
                Defaults to all keys in sweeps.

        Returns:
            Dict mapping perturbation names to curve data dicts containing
            ``param_name``, ``param_values``, ``miou_values``, and
            ``pixel_acc_values``.
        """
        if perturbations is None:
            perturbations = list(self.sweeps.keys())

        results: Dict[str, Any] = {}

        for name in perturbations:
            fn = PERTURBATION_REGISTRY.get(name)
            if fn is None:
                logger.warning("Unknown perturbation '%s' — skipping.", name)
                continue

            sweep = self.sweeps.get(name)
            if not sweep:
                logger.warning("No sweep defined for '%s' — skipping.", name)
                continue

            param_name = list(sweep.keys())[0]
            param_values = sweep[param_name]

            miou_values = []
            pixel_acc_values = []

            logger.info(
                "Multi-scale sweep: '%s' — %s = %s",
                name, param_name, param_values,
            )

            for val in param_values:
                perturbed = fn(image, **{param_name: val})
                seg_output: SegmentationOutput = self.engine.run(perturbed)

                from perception_engine.evaluation.robustness import _resize_gt
                gt_resized = _resize_gt(ground_truth, seg_output.mask.shape)

                metrics = compute_segmentation_metrics(
                    prediction=seg_output.mask,
                    ground_truth=gt_resized,
                    num_classes=self.num_classes,
                    class_names=self.class_names,
                )

                miou_values.append(float(metrics["mean_iou"]))
                pixel_acc_values.append(float(metrics["pixel_accuracy"]))

            results[name] = {
                "param_name": param_name,
                "param_values": param_values,
                "miou_values": miou_values,
                "pixel_acc_values": pixel_acc_values,
            }

            logger.info(
                "  '%s' sweep done — mIoU range: [%.4f, %.4f]",
                name,
                min(miou_values),
                max(miou_values),
            )

        return results


def save_degradation_curves(
    multiscale_data: Dict[str, Any],
    output_path: str,
    model_name: str = "model",
) -> str:
    """Save degradation curve plots as a single multi-panel PNG.

    Args:
        multiscale_data: Output of :meth:`MultiScaleRobustnessEvaluator.evaluate`.
        output_path: File path (PNG) to save the chart.
        model_name: Label for the plot title.

    Returns:
        Path to the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(multiscale_data)
    if n == 0:
        return output_path

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, (pert_name, data) in enumerate(multiscale_data.items()):
        ax = axes[0][idx]
        param_name = data["param_name"]
        param_vals = data["param_values"]
        miou_vals = data["miou_values"]

        ax.plot(param_vals, miou_vals, "o-", color="#2ecc71", linewidth=2, markersize=6)
        ax.set_xlabel(f"{param_name}", fontsize=11)
        ax.set_ylabel("mIoU", fontsize=11)
        ax.set_title(f"{pert_name}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(miou_vals) * 1.2 if miou_vals else 1.0)

    fig.suptitle(
        f"Robustness Degradation — {model_name}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Degradation curves saved to %s", output_path)
    return output_path
