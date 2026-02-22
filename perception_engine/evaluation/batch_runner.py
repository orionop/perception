"""
Batch evaluation runner.

Iterates over a directory of images (and optional ground-truth masks),
runs the full benchmark pipeline per image, and aggregates metrics
across the entire dataset into a summary report.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from perception_engine.data_types import BenchmarkReport
from perception_engine.engine.mask_remapping import (
    build_mapping_from_config,
    remap_mask,
)
from perception_engine.evaluation.benchmarking import BenchmarkRunner
from perception_engine.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class BatchRunner:
    """Run the benchmark pipeline across a directory of images.

    Aggregates per-image metrics into dataset-level statistics and
    produces summary reports in JSON and CSV.

    Attributes:
        runner: The underlying single-image benchmark runner.
        config: Full experiment configuration dict.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        config: Dict[str, Any],
        device: str,
    ) -> None:
        self.runner = BenchmarkRunner(registry, config, device)
        self.config = config
        self._mask_mapping = build_mapping_from_config(config)

    def run(
        self,
        image_dir: str | Path,
        gt_dir: Optional[str | Path] = None,
        image_ext: str = ".png",
        gt_ext: str = ".png",
        max_samples: Optional[int] = None,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run batch evaluation.

        Args:
            image_dir: Directory containing input RGB images.
            gt_dir: Optional directory of ground-truth masks. File names
                must match (stem-wise) those in ``image_dir``.
            image_ext: Image file extension filter.
            gt_ext: Ground-truth mask file extension.
            max_samples: Cap on the number of images to process.
                ``None`` means process all.
            output_dir: Directory to save reports and optional
                per-image visualizations.

        Returns:
            Aggregate report dict with dataset-level metrics per model.
        """
        image_dir = Path(image_dir)
        gt_path = Path(gt_dir) if gt_dir else None

        # Discover image files.
        image_files = sorted(image_dir.glob(f"*{image_ext}"))
        if max_samples is not None:
            image_files = image_files[:max_samples]

        if not image_files:
            raise FileNotFoundError(
                f"No *{image_ext} files found in {image_dir}"
            )

        logger.info(
            "Batch evaluation: %d images from %s", len(image_files), image_dir
        )

        # Accumulators: model_name → metric_name → list of values.
        accumulators: Dict[str, Dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        robustness_acc: Dict[str, Dict[str, Dict[str, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for idx, img_path in enumerate(image_files):
            logger.info(
                "[%d/%d] Processing %s",
                idx + 1,
                len(image_files),
                img_path.name,
            )

            image = self._load_image(img_path)
            gt = self._load_gt(img_path, gt_path, gt_ext) if gt_path else None

            reports: List[BenchmarkReport] = self.runner.run(image, gt)

            for report in reports:
                name = report.model_name
                acc = accumulators[name]

                acc["inference_time_ms"].append(
                    report.segmentation_output.inference_time_ms
                )

                if report.metrics:
                    acc["mean_iou"].append(report.metrics.get("mean_iou", 0.0))
                    acc["pixel_accuracy"].append(
                        report.metrics.get("pixel_accuracy", 0.0)
                    )
                    # Per-class IoU.
                    for cls, iou in report.metrics.get(
                        "per_class_iou", {}
                    ).items():
                        if np.isfinite(iou):
                            acc[f"iou_{cls}"].append(iou)

                acc["path_found"].append(report.navigation_result.path_found)
                if report.navigation_result.path_found:
                    acc["path_cost"].append(report.navigation_result.path_cost)

                if report.safety_report:
                    acc["safety_score"].append(
                        report.safety_report.safety_score
                    )

                # Robustness.
                if report.robustness:
                    for pert, data in report.robustness.items():
                        r_acc = robustness_acc[name][pert]
                        r_acc["mean_iou"].append(data.get("mean_iou", 0.0))
                        r_acc["miou_drop"].append(data.get("miou_drop", 0.0))

        # Aggregate.
        aggregate = self._aggregate(
            accumulators, robustness_acc, len(image_files)
        )

        # Save reports.
        if output_dir:
            self._save_reports(aggregate, Path(output_dir))

        # Print summary.
        self._print_summary(aggregate)

        return aggregate

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"))

    def _load_gt(
        self, img_path: Path, gt_dir: Path, gt_ext: str
    ) -> Optional[np.ndarray]:
        """Load and optionally remap a ground-truth mask."""
        gt_file = gt_dir / f"{img_path.stem}{gt_ext}"
        if not gt_file.exists():
            logger.warning("GT mask not found for %s — skipping.", img_path.name)
            return None

        from PIL import Image

        gt = np.array(Image.open(gt_file)).astype(np.int32)

        if self._mask_mapping is not None:
            gt = remap_mask(gt, self._mask_mapping)

        return gt

    @staticmethod
    def _aggregate(
        accumulators: Dict[str, Dict[str, list]],
        robustness_acc: Dict[str, Dict[str, Dict[str, list]]],
        total_images: int,
    ) -> Dict[str, Any]:
        """Compute mean ± std for all accumulated metrics."""
        result: Dict[str, Any] = {"total_images": total_images, "models": {}}

        for model_name, acc in accumulators.items():
            model_stats: Dict[str, Any] = {}

            for key, values in acc.items():
                if key == "path_found":
                    model_stats["path_found_pct"] = (
                        100.0 * sum(values) / len(values) if values else 0.0
                    )
                elif values:
                    arr = np.array(values, dtype=np.float64)
                    model_stats[key] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }

            # Robustness aggregate.
            if model_name in robustness_acc:
                rob: Dict[str, Any] = {}
                for pert, r_vals in robustness_acc[model_name].items():
                    rob[pert] = {}
                    for metric, vals in r_vals.items():
                        arr = np.array(vals, dtype=np.float64)
                        rob[pert][metric] = {
                            "mean": float(np.mean(arr)),
                            "std": float(np.std(arr)),
                        }
                model_stats["robustness"] = rob

            result["models"][model_name] = model_stats

        return result

    @staticmethod
    def _save_reports(aggregate: Dict[str, Any], output_dir: Path) -> None:
        """Save aggregate report as JSON and a comparison CSV."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON.
        json_path = output_dir / "batch_report.json"

        def _default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(json_path, "w") as f:
            json.dump(aggregate, f, indent=2, default=_default)
        logger.info("Batch report saved to %s", json_path)

        # CSV summary.
        csv_path = output_dir / "batch_summary.csv"
        models = aggregate.get("models", {})
        if models:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "model", "mean_iou_mean", "mean_iou_std",
                    "pixel_acc_mean", "latency_mean_ms",
                    "safety_mean", "path_found_pct",
                ])
                for name, stats in models.items():
                    writer.writerow([
                        name,
                        f"{stats.get('mean_iou', {}).get('mean', 0):.4f}",
                        f"{stats.get('mean_iou', {}).get('std', 0):.4f}",
                        f"{stats.get('pixel_accuracy', {}).get('mean', 0):.4f}",
                        f"{stats.get('inference_time_ms', {}).get('mean', 0):.2f}",
                        f"{stats.get('safety_score', {}).get('mean', 0):.4f}",
                        f"{stats.get('path_found_pct', 0):.1f}",
                    ])
            logger.info("Batch CSV saved to %s", csv_path)

    @staticmethod
    def _print_summary(aggregate: Dict[str, Any]) -> None:
        """Print a formatted dataset-level summary."""
        sep = "=" * 80
        print(f"\n{sep}")
        print("  BATCH EVALUATION SUMMARY")
        print(f"  Images evaluated: {aggregate['total_images']}")
        print(sep)
        print()

        header = (
            f"  {'Model':<20} {'mIoU':>12} {'PixAcc':>12} "
            f"{'Latency(ms)':>14} {'Safety':>10} {'PathFound%':>12}"
        )
        print(header)
        print("  " + "-" * 76)

        for name, stats in aggregate.get("models", {}).items():
            miou = stats.get("mean_iou", {})
            pix = stats.get("pixel_accuracy", {})
            lat = stats.get("inference_time_ms", {})
            saf = stats.get("safety_score", {})
            pf = stats.get("path_found_pct", 0)

            print(
                f"  {name:<20} "
                f"{miou.get('mean', 0):>6.4f}±{miou.get('std', 0):<5.4f}"
                f"{pix.get('mean', 0):>6.4f}±{pix.get('std', 0):<5.4f}"
                f"{lat.get('mean', 0):>10.1f}±{lat.get('std', 0):<4.1f}"
                f"{saf.get('mean', 0):>6.4f}±{saf.get('std', 0):<4.4f}"
                f"{pf:>10.1f}%"
            )

        print(sep)

        # Robustness sub-table.
        has_rob = any(
            "robustness" in s for s in aggregate.get("models", {}).values()
        )
        if has_rob:
            print(f"\n  {'Model':<20} {'Perturbation':<15} "
                  f"{'mIoU Drop':>12}")
            print("  " + "-" * 50)
            for name, stats in aggregate.get("models", {}).items():
                for pert, pdata in stats.get("robustness", {}).items():
                    drop = pdata.get("miou_drop", {})
                    print(
                        f"  {name:<20} {pert:<15} "
                        f"{drop.get('mean', 0):>6.4f}±{drop.get('std', 0):.4f}"
                    )
            print(sep)
