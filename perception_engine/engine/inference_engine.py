"""
Inference engine.

Orchestrates the full inference pipeline: preprocessing → forward pass →
postprocessing, with accurate latency measurement.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

from perception_engine.core.data_types import SegmentationOutput
from perception_engine.engine.postprocessing import postprocess_logits
from perception_engine.engine.preprocessing import preprocess_from_config
from perception_engine.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Runs the complete inference pipeline for a given model.

    Handles preprocessing, device-aware forward pass, postprocessing,
    and latency measurement in a single ``run()`` call.

    Attributes:
        model: The segmentation model to run.
        config: Full experiment configuration dict.
        device: Active PyTorch device string.
    """

    def __init__(
        self,
        model: BaseModel,
        config: Dict[str, Any],
        device: str,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device

    def run(
        self,
        image: np.ndarray | Image.Image,
    ) -> SegmentationOutput:
        """Execute the full inference pipeline on a single image.

        Steps:
            1. Preprocess the image (resize, normalise, to tensor).
            2. Forward pass through the model.
            3. Measure wall-clock inference latency.
            4. Postprocess logits into mask, confidence, probabilities.

        Args:
            image: Raw RGB input image.

        Returns:
            :class:`SegmentationOutput` with mask, confidence, probs,
            and timing.
        """
        # Step 1: Preprocess.
        tensor = preprocess_from_config(image, self.config, self.device)

        # Step 2 & 3: Forward pass with timing.
        # Use torch.cuda.synchronize for accurate GPU timing when applicable.
        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        logits = self.model.forward(tensor)

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "Inference for '%s': %.2f ms",
            self.model.name,
            elapsed_ms,
        )

        # Step 4: Postprocess.
        return postprocess_logits(logits, inference_time_ms=elapsed_ms)
