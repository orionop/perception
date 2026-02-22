"""
Report export utilities.

Serializes ``BenchmarkReport`` dataclasses to JSON and CSV formats
for downstream analysis, paper tables, and experiment tracking.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from perception_engine.core.data_types import BenchmarkReport

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# NumPy-aware JSON encoder
# ------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types and infinity."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isinf(obj):
                return "Infinity"
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ------------------------------------------------------------------
# BenchmarkReport → serializable dict
# ------------------------------------------------------------------

def report_to_dict(report: BenchmarkReport) -> Dict[str, Any]:
    """Convert a ``BenchmarkReport`` to a JSON-friendly dict.

    Large arrays (masks, confidence maps, cost maps) are summarised
    with shape and dtype rather than serialised in full.
    """
    seg = report.segmentation_output
    nav = report.navigation_result

    out: Dict[str, Any] = {
        "model_name": report.model_name,
        "inference_time_ms": seg.inference_time_ms,
        "segmentation": {
            "mask_shape": list(seg.mask.shape),
            "num_classes": int(seg.probabilities.shape[0]),
            "confidence_mean": float(seg.confidence_map.mean()),
            "confidence_std": float(seg.confidence_map.std()),
            "class_distribution": _class_distribution(seg.mask),
        },
        "navigation": {
            "path_found": nav.path_found,
            "path_cost": _safe_float(nav.path_cost),
            "path_length": len(nav.path) if nav.path else 0,
        },
    }

    # Safety.
    if report.safety_report:
        sr = report.safety_report
        out["safety"] = {
            "safety_score": sr.safety_score,
            "obstacle_overlap_pct": sr.obstacle_overlap_pct,
            "avg_confidence": sr.avg_confidence,
            "path_cost": _safe_float(sr.path_cost),
        }

    # Metrics.
    if report.metrics:
        m = {**report.metrics}
        # Convert confusion matrix to list-of-lists if present.
        if "confusion_matrix" in m:
            cm = m["confusion_matrix"]
            m["confusion_matrix"] = cm.tolist() if isinstance(cm, np.ndarray) else cm
        out["metrics"] = m

    # Robustness.
    if report.robustness:
        out["robustness"] = report.robustness

    return out


# ------------------------------------------------------------------
# Export functions
# ------------------------------------------------------------------

def export_json(
    reports: List[BenchmarkReport],
    output_path: str | Path,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export benchmark reports to a structured JSON file.

    Args:
        reports: List of model benchmark reports.
        output_path: Path to the output JSON file.
        config: Optional experiment config to embed for reproducibility.

    Returns:
        Path to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "num_models": len(reports),
        "reports": [report_to_dict(r) for r in reports],
    }

    if config is not None:
        # Strip large or non-serializable entries.
        safe_config = {
            k: v for k, v in config.items()
            if k not in ("_internal",)
        }
        payload["config"] = safe_config

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, cls=_NumpyEncoder)

    logger.info("JSON report exported to %s", path)
    return path


def export_csv(
    reports: List[BenchmarkReport],
    output_path: str | Path,
) -> Path:
    """Export a comparison table of all models to CSV.

    One row per model with key metrics. Suitable for pasting into
    a paper table or spreadsheet.

    Args:
        reports: List of model benchmark reports.
        output_path: Path to the output CSV file.

    Returns:
        Path to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "model",
        "mean_iou",
        "pixel_accuracy",
        "inference_time_ms",
        "path_found",
        "path_cost",
        "path_length",
        "safety_score",
        "obstacle_overlap_pct",
        "avg_confidence",
    ]

    # Per-class IoU columns (discover from first report with metrics).
    class_iou_keys: List[str] = []
    for r in reports:
        pci = r.metrics.get("per_class_iou", {})
        if pci:
            class_iou_keys = [f"iou_{k}" for k in pci.keys()]
            break

    # Robustness columns.
    rob_keys: List[str] = []
    for r in reports:
        if r.robustness:
            rob_keys = [f"rob_{p}_miou_drop" for p in r.robustness.keys()]
            break

    all_headers = headers + class_iou_keys + rob_keys

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_headers)
        writer.writeheader()

        for r in reports:
            row: Dict[str, Any] = {
                "model": r.model_name,
                "mean_iou": _fmt(r.metrics.get("mean_iou")),
                "pixel_accuracy": _fmt(r.metrics.get("pixel_accuracy")),
                "inference_time_ms": f"{r.segmentation_output.inference_time_ms:.2f}",
                "path_found": r.navigation_result.path_found,
                "path_cost": _safe_float(r.navigation_result.path_cost),
                "path_length": len(r.navigation_result.path) if r.navigation_result.path else 0,
                "safety_score": _fmt(
                    r.safety_report.safety_score if r.safety_report else None
                ),
                "obstacle_overlap_pct": _fmt(
                    r.safety_report.obstacle_overlap_pct if r.safety_report else None
                ),
                "avg_confidence": _fmt(
                    r.safety_report.avg_confidence if r.safety_report else None
                ),
            }

            # Per-class IoU.
            pci = r.metrics.get("per_class_iou", {})
            for cls_name, iou in pci.items():
                row[f"iou_{cls_name}"] = _fmt(iou)

            # Robustness.
            if r.robustness:
                for pert, data in r.robustness.items():
                    row[f"rob_{pert}_miou_drop"] = _fmt(data.get("miou_drop"))

            writer.writerow(row)

    logger.info("CSV report exported to %s", path)
    return path


def export_per_class_csv(
    reports: List[BenchmarkReport],
    output_path: str | Path,
) -> Path:
    """Export detailed per-class IoU breakdown to CSV.

    One row per (model, class) pair — useful for the problem statement's
    "visually similar classes" analysis requirement.

    Args:
        reports: List of model benchmark reports.
        output_path: Path to the output CSV file.

    Returns:
        Path to the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "class", "iou"])

        for r in reports:
            pci = r.metrics.get("per_class_iou", {})
            for cls_name, iou in pci.items():
                writer.writerow([r.model_name, cls_name, _fmt(iou)])

    logger.info("Per-class CSV exported to %s", path)
    return path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _class_distribution(mask: np.ndarray) -> Dict[str, float]:
    """Compute the fraction of pixels belonging to each class."""
    total = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    return {str(int(u)): round(float(c / total), 6) for u, c in zip(unique, counts)}


def _safe_float(val: Any) -> Any:
    """Convert infinity to string for JSON compatibility."""
    if val is None:
        return None
    if isinstance(val, float) and np.isinf(val):
        return "Infinity"
    return val


def _fmt(val: Any, decimals: int = 6) -> str:
    """Format a numeric value for CSV output."""
    if val is None:
        return ""
    if isinstance(val, float):
        if np.isinf(val) or np.isnan(val):
            return str(val)
        return f"{val:.{decimals}f}"
    return str(val)
