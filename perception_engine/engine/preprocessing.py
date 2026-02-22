"""
Image preprocessing pipeline.

Converts raw RGB images (numpy arrays or PIL images) into normalised
PyTorch tensors suitable for segmentation model input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


def preprocess_image(
    image: np.ndarray | Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    mean: List[float] | None = None,
    std: List[float] | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Convert a raw image into a batched, normalised tensor.

    Pipeline: PIL conversion → resize → float32 [0, 1] → normalise → tensor
    → add batch dim → device transfer.

    Args:
        image: Input RGB image as an ``(H, W, 3)`` numpy array or
            a ``PIL.Image``.
        target_size: ``(height, width)`` to resize the image to.
        mean: Per-channel mean for normalisation.  Defaults to ImageNet values.
        std: Per-channel std for normalisation.  Defaults to ImageNet values.
        device: Target PyTorch device string.

    Returns:
        Tensor of shape ``(1, 3, H, W)`` on the requested device.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    # Ensure PIL Image.
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    # Resize using bilinear interpolation.
    image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)

    # Convert to float32 array in [0, 1].
    arr = np.array(image, dtype=np.float32) / 255.0

    # Normalise per channel.
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean_arr) / std_arr

    # HWC → CHW and add batch dimension.
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    return tensor.to(device)


def preprocess_from_config(
    image: np.ndarray | Image.Image,
    config: Dict[str, Any],
    device: str = "cpu",
) -> torch.Tensor:
    """Convenience wrapper that reads preprocessing params from config.

    Args:
        image: Raw input image.
        config: Full experiment config dict.
        device: Target device.

    Returns:
        Preprocessed tensor.
    """
    prep = config.get("preprocessing", {})
    return preprocess_image(
        image=image,
        target_size=tuple(prep.get("target_size", [512, 512])),
        mean=prep.get("normalize", {}).get("mean"),
        std=prep.get("normalize", {}).get("std"),
        device=device,
    )
