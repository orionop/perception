"""
Core data structures for the Perception Evaluation Engine.

All structured outputs exchanged between pipeline stages are defined here
as frozen-safe dataclasses. This ensures type safety, clarity, and
serialization readiness across the entire framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SegmentationOutput:
    """Result of a single segmentation inference pass.

    Attributes:
        mask: (H, W) int array of predicted class IDs (argmax of softmax).
        confidence_map: (H, W) float array of per-pixel max softmax
            probability, representing model confidence.
        probabilities: (C, H, W) float array of full softmax distribution
            across all classes.
        inference_time_ms: Wall-clock forward-pass latency in milliseconds,
            measured on the active device.
    """

    mask: np.ndarray
    confidence_map: np.ndarray
    probabilities: np.ndarray
    inference_time_ms: float


@dataclass
class NavigationResult:
    """Result of the A* path-planning stage.

    Attributes:
        cost_map: (H, W) float array of traversability costs derived from
            the segmentation mask.
        path: Ordered list of (row, col) grid coordinates from start to goal.
            ``None`` if no path was found.
        path_cost: Total accumulated cost along the path.
            ``float('inf')`` if no path exists.
        path_found: Whether the planner successfully found a route.
    """

    cost_map: np.ndarray
    path: Optional[List[Tuple[int, int]]]
    path_cost: float
    path_found: bool


@dataclass
class SafetyReport:
    """Navigation safety assessment for a given path.

    Attributes:
        obstacle_overlap_pct: Percentage of path cells that coincide with
            obstacle-class pixels (0.0–100.0).
        avg_confidence: Mean model confidence along the planned path.
        path_cost: Total traversability cost of the path (same as
            ``NavigationResult.path_cost``).
        safety_score: Composite safety score in [0, 1].
            Computed via a configurable weighted formula.
    """

    obstacle_overlap_pct: float
    avg_confidence: float
    path_cost: float
    safety_score: float


@dataclass
class BenchmarkReport:
    """Complete evaluation report for a single model.

    Aggregates inference output, navigation analysis, safety assessment,
    segmentation quality metrics, and optional robustness testing.

    Attributes:
        model_name: Human-readable identifier matching the YAML config entry.
        segmentation_output: Raw inference results.
        navigation_result: Path-planning results derived from the
            segmentation mask.
        safety_report: Safety assessment for the planned path.
            ``None`` if no path was found.
        metrics: Segmentation quality metrics dict containing keys such as
            ``pixel_accuracy``, ``mean_iou``, ``per_class_iou``,
            ``confusion_matrix``.
        robustness: Mapping of perturbation names to metric dicts showing
            performance under each perturbation. ``None`` if robustness
            testing is disabled.
    """

    model_name: str
    segmentation_output: SegmentationOutput
    navigation_result: NavigationResult
    safety_report: Optional[SafetyReport]
    metrics: Dict[str, Any] = field(default_factory=dict)
    robustness: Optional[Dict[str, Dict[str, Any]]] = None
