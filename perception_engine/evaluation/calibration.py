"""
Confidence calibration analysis.

Computes Expected Calibration Error (ECE) and generates reliability
diagram data to assess whether model confidence scores are trustworthy
for safety scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_ece(
    confidence_map: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    num_bins: int = 15,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error and reliability diagram data.

    ECE measures how well a model's predicted confidence matches its
    actual accuracy.  A perfectly calibrated model has ECE = 0.

    Args:
        confidence_map: ``(H, W)`` float array of per-pixel max softmax
            probabilities (the model's confidence in its prediction).
        prediction: ``(H, W)`` integer predicted class IDs.
        ground_truth: ``(H, W)`` integer ground-truth class IDs.
        num_bins: Number of bins for the reliability diagram.

    Returns:
        Dict with:
            - ``ece``: scalar Expected Calibration Error.
            - ``bin_edges``: list of bin boundary values.
            - ``bin_accuracies``: per-bin actual accuracy.
            - ``bin_confidences``: per-bin mean confidence.
            - ``bin_counts``: number of pixels per bin.
    """
    conf_flat = confidence_map.ravel().astype(np.float64)
    pred_flat = prediction.ravel().astype(np.int64)
    gt_flat = ground_truth.ravel().astype(np.int64)

    correct = (pred_flat == gt_flat).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    total = len(conf_flat)

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == num_bins - 1:
            mask = (conf_flat >= lo) & (conf_flat <= hi)
        else:
            mask = (conf_flat >= lo) & (conf_flat < hi)

        count = mask.sum()
        bin_counts.append(int(count))

        if count == 0:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            continue

        acc = float(correct[mask].mean())
        conf = float(conf_flat[mask].mean())
        bin_accuracies.append(acc)
        bin_confidences.append(conf)

        ece += (count / total) * abs(acc - conf)

    logger.info("ECE = %.4f (%d bins)", ece, num_bins)

    return {
        "ece": float(ece),
        "num_bins": num_bins,
        "bin_edges": bin_edges.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }


def save_reliability_diagram(
    calibration_data: Dict[str, Any],
    output_path: str,
    model_name: str = "model",
) -> str:
    """Save a reliability diagram as a PNG image.

    Args:
        calibration_data: Output of :func:`compute_ece`.
        output_path: Path (file) to save the diagram.
        model_name: Label for the plot title.

    Returns:
        Path to the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bin_edges = calibration_data["bin_edges"]
    bin_accs = calibration_data["bin_accuracies"]
    bin_confs = calibration_data["bin_confidences"]
    ece = calibration_data["ece"]
    num_bins = calibration_data["num_bins"]

    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(num_bins)]
    width = 1.0 / num_bins * 0.8

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Perfect calibration line.
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)

    # Accuracy bars.
    ax.bar(
        centers, bin_accs, width=width, alpha=0.7,
        color="#4a90d9", edgecolor="white", label="Accuracy",
    )

    # Confidence line.
    ax.plot(
        centers, bin_confs, "o-", color="#e74c3c",
        markersize=5, label="Avg confidence",
    )

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(
        f"Reliability Diagram — {model_name}\nECE = {ece:.4f}",
        fontsize=13,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Reliability diagram saved to %s", output_path)
    return output_path
