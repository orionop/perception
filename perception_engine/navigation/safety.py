"""
Navigation safety metrics.

Computes a composite safety report for a planned path by analysing obstacle
overlap, model confidence along the route, and total path cost.  The
weighting formula for the final score is fully configurable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from perception_engine.data_types import NavigationResult, SafetyReport

logger = logging.getLogger(__name__)


def compute_safety_report(
    navigation_result: NavigationResult,
    mask: np.ndarray,
    confidence_map: np.ndarray,
    cost_mapping: Dict[str, List[int]],
    safety_cfg: Dict[str, Any],
) -> Optional[SafetyReport]:
    """Compute a safety report for a planned navigation path.

    The composite safety score is::

        score = 1.0
              - w_obstacle * (obstacle_overlap_pct / 100)
              - w_confidence * (1 - avg_confidence)
              - w_cost * min(path_cost / max_cost, 1.0)

    All weights and ``max_cost`` are read from ``safety_cfg``.

    Args:
        navigation_result: Output of the A* planner.
        mask: ``(H, W)`` segmentation mask (class IDs).
        confidence_map: ``(H, W)`` model confidence map.
        cost_mapping: Category-to-class-ID mapping from config.
        safety_cfg: Safety score configuration dict with keys
            ``weight_obstacle``, ``weight_confidence``, ``weight_cost``,
            ``max_acceptable_cost``.

    Returns:
        :class:`SafetyReport`, or ``None`` if no path was found.
    """
    if not navigation_result.path_found or navigation_result.path is None:
        logger.warning("No path available — safety report is None.")
        return None

    path = navigation_result.path
    obstacle_ids = set(cost_mapping.get("obstacle", []))

    # --- Obstacle overlap ---
    obstacle_count = sum(
        1 for r, c in path if int(mask[r, c]) in obstacle_ids
    )
    obstacle_overlap_pct = 100.0 * obstacle_count / len(path)

    # --- Average confidence along path ---
    confidences = [float(confidence_map[r, c]) for r, c in path]
    avg_confidence = float(np.mean(confidences))

    # --- Path cost ---
    path_cost = navigation_result.path_cost

    # --- Composite safety score ---
    w_obs = safety_cfg.get("weight_obstacle", 0.4)
    w_conf = safety_cfg.get("weight_confidence", 0.3)
    w_cost = safety_cfg.get("weight_cost", 0.3)
    max_cost = safety_cfg.get("max_acceptable_cost", 1000.0)

    safety_score = (
        1.0
        - w_obs * (obstacle_overlap_pct / 100.0)
        - w_conf * (1.0 - avg_confidence)
        - w_cost * min(path_cost / max_cost, 1.0)
    )
    # Clamp to [0, 1].
    safety_score = float(np.clip(safety_score, 0.0, 1.0))

    report = SafetyReport(
        obstacle_overlap_pct=obstacle_overlap_pct,
        avg_confidence=avg_confidence,
        path_cost=path_cost,
        safety_score=safety_score,
    )

    logger.info(
        "Safety: overlap=%.1f%%, confidence=%.3f, cost=%.1f, score=%.3f",
        obstacle_overlap_pct,
        avg_confidence,
        path_cost,
        safety_score,
    )

    return report
