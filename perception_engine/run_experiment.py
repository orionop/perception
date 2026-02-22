#!/usr/bin/env python3
"""
Perception Evaluation Engine — Experiment Runner (CLI Entry Point).

Usage (single image):
    python -m perception_engine.run_experiment \\
        --config perception_engine/configs/experiment.yaml \\
        --image  path/to/input.png \\
        [--ground-truth path/to/gt_mask.png]

Usage (batch mode):
    python -m perception_engine.run_experiment \\
        --config perception_engine/configs/experiment.yaml \\
        --batch

Batch mode reads image_dir and gt_dir from the config's dataset section.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from perception_engine.configs.config_loader import get_device, load_config
from perception_engine.engine.mask_remapping import (
    build_mapping_from_config,
    remap_mask,
)
from perception_engine.evaluation.batch_runner import BatchRunner
from perception_engine.evaluation.benchmarking import BenchmarkRunner
from perception_engine.evaluation.calibration import (
    compute_ece,
    save_reliability_diagram,
)
from perception_engine.evaluation.export import (
    export_csv,
    export_json,
    export_per_class_csv,
)
from perception_engine.evaluation.explanation import (
    generate_explanation,
    print_explanation,
    save_explanation,
)
from perception_engine.evaluation.multiscale_robustness import (
    MultiScaleRobustnessEvaluator,
    save_degradation_curves,
)
from perception_engine.models.registry import ModelRegistry
from perception_engine.visualization.overlays import (
    overlay_confidence,
    overlay_mask,
    overlay_path,
)


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configure structured logging with optional JSON file output."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [
        logging.StreamHandler(),
    ]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        import json
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    "time": self.formatTime(record),
                    "name": record.name,
                    "level": record.levelname,
                    "message": record.getMessage(),
                })
        file_handler.setFormatter(JSONFormatter())
        handlers.append(file_handler)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Perception Evaluation Engine — Benchmark Runner",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML experiment configuration file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input RGB image (single-image mode).",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Path to ground-truth segmentation mask (optional).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch evaluation over the dataset directory in config.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Override ground-truth directory for batch mode.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max images to evaluate in batch mode (for quick runs).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this model (by name) instead of all registered models.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to structured JSON log file for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()
    if not args.batch and args.image is None:
        parser.error("Either --image or --batch must be specified.")
    if args.batch and args.image is not None:
        parser.error("--image and --batch are mutually exclusive.")
    return args


def load_image(path: str) -> np.ndarray:
    """Load an RGB image from disk.

    Args:
        path: Filesystem path to the image.

    Returns:
        ``(H, W, 3)`` uint8 numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = Image.open(p).convert("RGB")
    return np.array(img)


def load_ground_truth(path: str) -> np.ndarray:
    """Load a ground-truth mask from disk.

    The mask file should be a single-channel image where pixel values
    are class IDs.

    Args:
        path: Filesystem path to the mask.

    Returns:
        ``(H, W)`` int32 numpy array of class IDs.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ground-truth mask not found: {p}")
    mask = Image.open(p)
    return np.array(mask).astype(np.int32)


def save_visualizations(
    reports,
    image: np.ndarray,
    config: dict,
    output_dir: Path,
) -> None:
    """Save visualization images for each benchmark report.

    Args:
        reports: List of ``BenchmarkReport`` instances.
        image: Original input image.
        config: Experiment configuration dict.
        output_dir: Directory to save outputs into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = None
    if "class_names" in config:
        names = config["class_names"]
        if isinstance(names, list):
            class_names = {i: n for i, n in enumerate(names)}
        else:
            class_names = names

    for report in reports:
        name = report.model_name

        # Mask overlay.
        overlay_mask(
            image=image,
            mask=report.segmentation_output.mask,
            num_classes=config["models"][0]["num_classes"],
            class_names=class_names,
            save_path=str(output_dir / f"{name}_mask_overlay.png"),
        )

        # Path overlay.
        overlay_path(
            image=image,
            cost_map=report.navigation_result.cost_map,
            path=report.navigation_result.path,
            save_path=str(output_dir / f"{name}_path_overlay.png"),
        )

        # Confidence map.
        overlay_confidence(
            confidence_map=report.segmentation_output.confidence_map,
            save_path=str(output_dir / f"{name}_confidence.png"),
        )

    logging.getLogger(__name__).info(
        "Visualizations saved to %s", output_dir
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(verbose=args.verbose, log_file=args.log_file)
    logger = logging.getLogger(__name__)

    # 1. Load config.
    logger.info("Loading config from %s", args.config)
    config = load_config(args.config)

    # 2. Resolve device.
    device = get_device(config)
    logger.info("Using device: %s", device)

    # 3. Build model registry (optionally filtered to single model).
    registry = ModelRegistry.from_config(config, device)
    if args.model:
        if args.model not in registry.list_models():
            logger.error(
                "Model '%s' not found. Available: %s",
                args.model, registry.list_models(),
            )
            sys.exit(1)
        # Filter to single model for quick eval.
        config["models"] = [
            m for m in config["models"]
            if m["name"] == args.model
        ]
        registry = ModelRegistry.from_config(config, device)
    logger.info("Model registry: %s", registry.list_models())

    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("output_dir", "outputs"))

    if args.batch:
        # ---- Batch mode ----
        dataset_cfg = config.get("dataset", {})
        image_dir = dataset_cfg.get("image_dir")
        gt_dir = args.gt_dir or dataset_cfg.get("gt_dir")
        image_ext = dataset_cfg.get("image_ext", ".png")
        gt_ext = dataset_cfg.get("gt_ext", ".png")
        max_samples = args.max_samples or dataset_cfg.get("max_samples")

        if not image_dir:
            logger.error(
                "Batch mode requires 'dataset.image_dir' in config or "
                "override via --image-dir."
            )
            sys.exit(1)

        batch = BatchRunner(registry, config, device)
        batch.run(
            image_dir=image_dir,
            gt_dir=gt_dir,
            image_ext=image_ext,
            gt_ext=gt_ext,
            max_samples=max_samples,
            output_dir=output_dir,
        )
    else:
        # ---- Single-image mode ----
        image = load_image(args.image)
        logger.info("Input image loaded: shape=%s", image.shape)

        ground_truth: Optional[np.ndarray] = None
        if args.ground_truth:
            ground_truth = load_ground_truth(args.ground_truth)
            logger.info(
                "Ground-truth mask loaded: shape=%s", ground_truth.shape
            )

            mask_mapping = build_mapping_from_config(config)
            if mask_mapping is not None:
                ground_truth = remap_mask(ground_truth, mask_mapping)
                logger.info(
                    "Ground-truth remapped: %d raw values → %d classes",
                    len(mask_mapping),
                    len(set(mask_mapping.values())),
                )

        runner = BenchmarkRunner(registry, config, device)
        reports = runner.run(image, ground_truth)

        if output_cfg.get("save_visualizations", True):
            save_visualizations(reports, image, config, output_dir)

        # Export structured reports.
        export_json(
            reports,
            output_dir / "benchmark_report.json",
            config=config,
        )
        export_csv(reports, output_dir / "benchmark_comparison.csv")
        export_per_class_csv(reports, output_dir / "per_class_iou.csv")

        # Generate and display explanation.
        explanation = generate_explanation(reports, config)
        print_explanation(explanation)
        save_explanation(explanation, output_dir)

        logger.info(
            "Experiment complete. %d model(s) evaluated.", len(reports)
        )


if __name__ == "__main__":
    main()
