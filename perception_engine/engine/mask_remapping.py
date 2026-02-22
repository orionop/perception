"""
Mask remapping utilities.

Handles the conversion between raw dataset mask pixel values
(e.g., 100, 200, 300, ..., 10000) and contiguous model class indices
(0, 1, 2, ..., N-1).  This is necessary because many segmentation
datasets encode class labels as non-contiguous integers, while models
output predictions in a contiguous [0, num_classes) range.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_remap_lut(
    mask_value_mapping: Dict[int, int],
    ignore_index: int = 255,
) -> np.ndarray:
    """Build a dense lookup table for fast mask value remapping.

    Args:
        mask_value_mapping: Maps raw pixel values to contiguous class
            indices, e.g. ``{100: 0, 200: 1, ..., 10000: 9}``.
        ignore_index: Value assigned to any raw pixel value not present
            in the mapping.

    Returns:
        1-D numpy array of shape ``(max_raw_value + 1,)`` where
        ``lut[raw_value] = class_index``.
    """
    max_val = max(mask_value_mapping.keys()) + 1
    lut = np.full(max_val, fill_value=ignore_index, dtype=np.int32)

    for raw_val, class_idx in mask_value_mapping.items():
        lut[raw_val] = class_idx

    logger.debug(
        "Built remap LUT: %d entries, %d mapped classes",
        max_val,
        len(mask_value_mapping),
    )
    return lut


def remap_mask(
    mask: np.ndarray,
    mask_value_mapping: Dict[int, int],
    ignore_index: int = 255,
) -> np.ndarray:
    """Remap a raw ground-truth mask to contiguous class indices.

    Args:
        mask: ``(H, W)`` integer array with raw pixel values.
        mask_value_mapping: Raw value → contiguous index mapping.
        ignore_index: Index for unmapped values.

    Returns:
        ``(H, W)`` int32 array with contiguous class indices.
    """
    lut = build_remap_lut(mask_value_mapping, ignore_index)

    # Values outside the LUT range get the ignore index directly.
    out_of_range = (mask < 0) | (mask >= len(lut))
    safe_mask = np.clip(mask, 0, len(lut) - 1)
    result = lut[safe_mask]
    result[out_of_range] = ignore_index
    return result


def build_mapping_from_config(
    config: Dict[str, Any],
) -> Optional[Dict[int, int]]:
    """Extract the mask_value_mapping from experiment config.

    If the config contains a ``mask_value_mapping`` section (list of
    ``{raw: X, index: Y}`` or a flat ``{raw_val: index}`` dict),
    return it as a ``Dict[int, int]``.  Returns ``None`` if no
    mapping is configured (assumes mask values are already contiguous).

    Args:
        config: Full experiment configuration dict.

    Returns:
        Mapping dict or ``None``.
    """
    mapping = config.get("mask_value_mapping")
    if mapping is None:
        return None

    # Support two formats:
    #   1. Dict:  {100: 0, 200: 1, ...}
    #   2. List:  [{raw: 100, index: 0}, ...]
    if isinstance(mapping, dict):
        return {int(k): int(v) for k, v in mapping.items()}

    if isinstance(mapping, list):
        return {
            int(entry["raw"]): int(entry["index"]) for entry in mapping
        }

    logger.warning(
        "Unrecognised mask_value_mapping format: %s", type(mapping)
    )
    return None
