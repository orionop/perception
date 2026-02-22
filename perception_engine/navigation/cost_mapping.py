"""
Traversability cost mapping.

Converts a segmentation mask into a 2-D cost map by mapping each class ID
to a navigation cost based on the YAML configuration.  The cost map drives
downstream path planning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def build_cost_map(
    mask: np.ndarray,
    cost_mapping: Dict[str, List[int]],
    cost_values: Dict[str, float],
) -> np.ndarray:
    """Convert a segmentation mask into a traversability cost map.

    Each pixel in the mask carries a class ID.  The ``cost_mapping`` dict
    groups class IDs by semantic category (``traversable``, ``obstacle``,
    ``soft``, ``ignored``).  The ``cost_values`` dict assigns a numeric
    cost to each category.

    Args:
        mask: ``(H, W)`` integer array of predicted class IDs.
        cost_mapping: Maps category names to lists of class IDs, e.g.
            ``{"traversable": [8, 2], "obstacle": [6, 7, 1], ...}``.
        cost_values: Maps category names to cost scalars, e.g.
            ``{"traversable": 1.0, "obstacle": inf, "soft": 5.0, ...}``.

    Returns:
        ``(H, W)`` float64 array of traversability costs.
    """
    # Default cost for any unmapped class ID is infinity (impassable).
    cost_map = np.full(mask.shape, fill_value=float("inf"), dtype=np.float64)

    # Build a flat lookup: class_id → cost.
    class_to_cost: Dict[int, float] = {}
    for category, class_ids in cost_mapping.items():
        cost = cost_values.get(category, float("inf"))
        for cid in class_ids:
            class_to_cost[cid] = cost

    # Vectorised assignment via a lookup array for speed.
    max_class_id = max(max(class_to_cost.keys()), int(mask.max())) + 1
    lut = np.full(max_class_id, fill_value=float("inf"), dtype=np.float64)
    for cid, cost in class_to_cost.items():
        if cid < max_class_id:
            lut[cid] = cost

    # Clip mask values to the LUT range (defensive).
    safe_mask = np.clip(mask, 0, max_class_id - 1)
    cost_map = lut[safe_mask]

    mapped_count = np.isfinite(cost_map).sum()
    total = cost_map.size
    logger.info(
        "Cost map built: %d / %d pixels mapped (%.1f%% traversable or soft)",
        mapped_count,
        total,
        100.0 * mapped_count / total,
    )

    return cost_map


def get_obstacle_mask(
    mask: np.ndarray,
    cost_mapping: Dict[str, List[int]],
) -> np.ndarray:
    """Return a boolean mask indicating obstacle pixels.

    Args:
        mask: ``(H, W)`` predicted class IDs.
        cost_mapping: Category-to-class-ID mapping.

    Returns:
        ``(H, W)`` boolean array — ``True`` where the pixel is an obstacle.
    """
    obstacle_ids = set(cost_mapping.get("obstacle", []))
    return np.isin(mask, list(obstacle_ids))
