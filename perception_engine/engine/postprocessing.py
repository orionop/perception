"""
Inference output postprocessing.

Converts raw model logits into human-interpretable segmentation outputs:
softmax probabilities, argmax class masks, and pixel-wise confidence maps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from perception_engine.data_types import SegmentationOutput


def postprocess_logits(
    logits: torch.Tensor,
    inference_time_ms: float,
) -> SegmentationOutput:
    """Convert raw model logits to a structured segmentation output.

    Args:
        logits: Raw model output of shape ``(1, C, H, W)``.
        inference_time_ms: Measured forward-pass latency in milliseconds.

    Returns:
        A :class:`SegmentationOutput` containing the argmax mask,
        confidence map, full probability distribution, and timing info.
    """
    # Softmax across the class dimension → (1, C, H, W).
    probabilities = F.softmax(logits, dim=1)

    # Per-pixel max probability = confidence.
    confidence_map, mask = torch.max(probabilities, dim=1)  # both (1, H, W)

    # Move to numpy, squeeze batch dim.
    mask_np = mask.squeeze(0).cpu().numpy().astype(np.int32)
    confidence_np = confidence_map.squeeze(0).cpu().numpy().astype(np.float32)
    probs_np = probabilities.squeeze(0).cpu().numpy().astype(np.float32)

    return SegmentationOutput(
        mask=mask_np,
        confidence_map=confidence_np,
        probabilities=probs_np,
        inference_time_ms=inference_time_ms,
    )
