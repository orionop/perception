"""
Multi-model benchmarking runner.

Orchestrates the full evaluation pipeline — inference, navigation, safety,
metrics, and robustness — across all registered models and collates the
results into structured benchmark reports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from perception_engine.core.data_types import BenchmarkReport, SegmentationOutput
from perception_engine.engine.inference_engine import InferenceEngine
from perception_engine.evaluation.robustness import RobustnessEvaluator
from perception_engine.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)
from perception_engine.models.registry import ModelRegistry
from perception_engine.navigation.cost_mapping import build_cost_map
from perception_engine.navigation.planner import planner_factory
from perception_engine.navigation.safety import compute_safety_report

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs the full evaluation pipeline across multiple models.

    Attributes:
        registry: Model registry containing all models to benchmark.
        config: Full experiment configuration dict.
        device: Active PyTorch device string.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        config: Dict[str, Any],
        device: str,
    ) -> None:
        self.registry = registry
        self.config = config
        self.device = device

    def run(
        self,
        image: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
    ) -> List[BenchmarkReport]:
        """Run the full benchmark across all registered models.

        For each model:
            1. Inference → SegmentationOutput
            2. Segmentation metrics (if ground_truth provided)
            3. Cost map → A* planning → NavigationResult
            4. Safety report
            5. Robustness evaluation (if enabled)

        Args:
            image: ``(H, W, 3)`` uint8 RGB input image.
            ground_truth: Optional ``(H, W)`` ground-truth class IDs.

        Returns:
            List of :class:`BenchmarkReport`, one per model.
        """
        model_names = self.registry.list_models()
        reports: List[BenchmarkReport] = []

        logger.info(
            "Starting benchmark across %d model(s): %s",
            len(model_names),
            model_names,
        )

        for name in model_names:
            logger.info("=" * 60)
            logger.info("Evaluating model: '%s'", name)
            logger.info("=" * 60)

            report = self._evaluate_single_model(name, image, ground_truth)
            reports.append(report)

        self._print_comparison(reports)
        return reports

    def _evaluate_single_model(
        self,
        model_name: str,
        image: np.ndarray,
        ground_truth: Optional[np.ndarray],
    ) -> BenchmarkReport:
        """Run the full pipeline for a single model."""

        model = self.registry.get(model_name)
        engine = InferenceEngine(model, self.config, self.device)

        # 1. Inference.
        seg_output: SegmentationOutput = engine.run(image)

        # 2. Segmentation metrics.
        metrics: Dict[str, Any] = {}
        if ground_truth is not None:
            gt_resized = self._resize_gt(ground_truth, seg_output.mask.shape)
            metrics = compute_segmentation_metrics(
                prediction=seg_output.mask,
                ground_truth=gt_resized,
                num_classes=model.num_classes,
                class_names=self._class_names(),
            )

        # 3. Cost map + planning.
        cost_map = build_cost_map(
            mask=seg_output.mask,
            cost_mapping=self.config["cost_mapping"],
            cost_values=self.config.get("cost_values", {}),
        )

        planner = planner_factory(self.config)
        start, goal = self._resolve_start_goal(cost_map.shape)
        nav_result = planner.plan(cost_map, start, goal)

        # 4. Safety report.
        safety_report = compute_safety_report(
            navigation_result=nav_result,
            mask=seg_output.mask,
            confidence_map=seg_output.confidence_map,
            cost_mapping=self.config["cost_mapping"],
            safety_cfg=self.config.get("safety", {}),
        )

        # 5. Robustness evaluation.
        robustness_results: Optional[Dict[str, Dict[str, Any]]] = None
        robustness_cfg = self.config.get("robustness", {})
        if robustness_cfg.get("enabled", False) and ground_truth is not None:
            perturbations = robustness_cfg.get("perturbations", [])
            if perturbations:
                evaluator = RobustnessEvaluator(
                    engine=engine,
                    perturbations=perturbations,
                    num_classes=model.num_classes,
                    class_names=self._class_names(),
                    perturbation_params=robustness_cfg.get("params", {}),
                )
                baseline_miou = metrics.get("mean_iou", 0.0)
                robustness_results = evaluator.evaluate(
                    image=image,
                    ground_truth=ground_truth,
                    baseline_miou=baseline_miou,
                )

        return BenchmarkReport(
            model_name=model_name,
            segmentation_output=seg_output,
            navigation_result=nav_result,
            safety_report=safety_report,
            metrics=metrics,
            robustness=robustness_results,
        )

    def _resolve_start_goal(self, shape: tuple) -> tuple:
        """Resolve planner start/goal from config, defaulting to corners."""
        planner_cfg = self.config.get("planner", {})
        start = planner_cfg.get("start")
        if start is None:
            start = (shape[0] - 1, 0)  # Bottom-left (ground level).
        else:
            start = tuple(start)
        goal = planner_cfg.get("goal")
        if goal is None:
            goal = (shape[0] - 1, shape[1] - 1)  # Bottom-right.
        else:
            goal = tuple(goal)
        return start, goal

    def _class_names(self) -> Optional[Dict[int, str]]:
        """Build class ID → name mapping from config."""
        names = self.config.get("class_names")
        if names is None:
            return None
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        return names

    @staticmethod
    def _resize_gt(
        ground_truth: np.ndarray, target_shape: tuple
    ) -> np.ndarray:
        """Resize GT to match prediction shape via nearest-neighbour."""
        if ground_truth.shape == target_shape:
            return ground_truth

        from PIL import Image

        gt_pil = Image.fromarray(ground_truth.astype(np.uint8))
        gt_resized = gt_pil.resize(
            (target_shape[1], target_shape[0]),
            resample=Image.NEAREST,
        )
        return np.array(gt_resized).astype(np.int32)

    @staticmethod
    def _print_comparison(reports: List[BenchmarkReport]) -> None:
        """Print a formatted comparison table to stdout."""
        separator = "-" * 90
        print("\n" + separator)
        print(f"{'Model':<20} {'mIoU':>8} {'PixAcc':>8} "
              f"{'Latency(ms)':>12} {'PathCost':>10} {'Safety':>8}")
        print(separator)

        for r in reports:
            miou = r.metrics.get("mean_iou", float("nan"))
            pix_acc = r.metrics.get("pixel_accuracy", float("nan"))
            latency = r.segmentation_output.inference_time_ms
            path_cost = r.navigation_result.path_cost
            safety = (
                r.safety_report.safety_score
                if r.safety_report
                else float("nan")
            )

            print(
                f"{r.model_name:<20} {miou:>8.4f} {pix_acc:>8.4f} "
                f"{latency:>12.2f} {path_cost:>10.2f} {safety:>8.4f}"
            )

        print(separator)

        # Robustness sub-table (if any model has it).
        has_robustness = any(r.robustness for r in reports)
        if has_robustness:
            print(f"\n{'Model':<20} {'Perturbation':<15} "
                  f"{'mIoU':>8} {'Drop':>8}")
            print(separator)
            for r in reports:
                if r.robustness:
                    for pert_name, pert_data in r.robustness.items():
                        print(
                            f"{r.model_name:<20} {pert_name:<15} "
                            f"{pert_data['mean_iou']:>8.4f} "
                            f"{pert_data['miou_drop']:>8.4f}"
                        )
            print(separator)
