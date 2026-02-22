"""
Weight loading utilities.

Handles loading ``state_dict`` checkpoint files into model networks with
device mapping, strict/non-strict modes, and descriptive error reporting
for missing or unexpected keys.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from perception_engine.models.base_model import BaseModel

logger = logging.getLogger(__name__)


def load_weights(
    model: "BaseModel",
    weights_path: str,
    device: str,
    strict: bool = False,
) -> None:
    """Load a state_dict checkpoint into a model's network.

    Args:
        model: ``BaseModel`` instance whose ``.network`` will receive
            the weights.
        weights_path: Filesystem path to a ``.pth`` / ``.pt`` checkpoint.
        device: Device to map tensors onto during loading.
        strict: If ``True``, require an exact key match between the
            checkpoint and the model.  Defaults to ``False`` so that
            partial fine-tuned checkpoints can be loaded without error.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If ``strict=True`` and keys do not match.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Weights file not found: {path}  "
            f"(model: '{model.name}')"
        )

    logger.info(
        "Loading weights for '%s' from %s (strict=%s)",
        model.name,
        path,
        strict,
    )

    state_dict = torch.load(
        path,
        map_location=torch.device(device),
        weights_only=True,
    )

    # Handle checkpoints that wrap the state_dict inside a dict
    # (e.g. {"model_state_dict": ..., "optimizer_state_dict": ...}).
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    result = model.network.load_state_dict(state_dict, strict=strict)

    if result.missing_keys:
        logger.warning(
            "Missing keys in checkpoint for '%s': %s",
            model.name,
            result.missing_keys,
        )
    if result.unexpected_keys:
        logger.warning(
            "Unexpected keys in checkpoint for '%s': %s",
            model.name,
            result.unexpected_keys,
        )

    logger.info("Weights loaded successfully for '%s'.", model.name)
