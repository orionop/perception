"""
Abstract base model wrapper.

Provides a uniform interface for any segmentation architecture supported
by ``segmentation_models_pytorch``.  Concrete model instances are created
via :func:`from_config` and always placed into eval mode on the resolved
device.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseModel:
    """Thin wrapper around a segmentation_models_pytorch model.

    Handles architecture instantiation, device placement, eval mode,
    and a clean forward interface that accepts raw tensors.

    Attributes:
        name: Human-readable model identifier from config.
        network: The underlying ``torch.nn.Module``.
        device: PyTorch device string the model lives on.
        num_classes: Number of output segmentation classes.
    """

    def __init__(
        self,
        name: str,
        network: nn.Module,
        device: str,
        num_classes: int,
    ) -> None:
        self.name = name
        self.device = device
        self.num_classes = num_classes
        self.network = network.to(device).eval()
        logger.info(
            "Model '%s' loaded on device '%s' (%d classes)",
            name,
            device,
            num_classes,
        )

    @classmethod
    def from_config(
        cls, model_cfg: Dict[str, Any], device: str
    ) -> "BaseModel":
        """Instantiate a model from a YAML model definition.

        Supported architectures (via ``segmentation_models_pytorch``):
            - ``unet``
            - ``unetplusplus``
            - ``deeplabv3``
            - ``deeplabv3plus``
            - ``fpn``
            - ``pspnet``
            - ``linknet``
            - ``manet``
            - ``pan``

        Args:
            model_cfg: Dict with keys ``name``, ``architecture``,
                ``backbone``, ``num_classes``, and optionally ``encoder_weights``.
            device: Target device string (``"cuda"``, ``"mps"``, ``"cpu"``).

        Returns:
            Initialised ``BaseModel`` instance ready for inference.

        Raises:
            ValueError: If the architecture is not recognised.
        """
        import segmentation_models_pytorch as smp

        arch = model_cfg["architecture"].lower()
        backbone = model_cfg["backbone"]
        num_classes = model_cfg["num_classes"]
        encoder_weights = model_cfg.get("encoder_weights", "imagenet")

        # Map config strings to smp constructor functions.
        architecture_map = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3": smp.DeepLabV3,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "linknet": smp.Linknet,
            "manet": smp.MAnet,
            "pan": smp.PAN,
        }

        if arch not in architecture_map:
            raise ValueError(
                f"Unknown architecture '{arch}'. "
                f"Supported: {list(architecture_map.keys())}"
            )

        network = architecture_map[arch](
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

        return cls(
            name=model_cfg["name"],
            network=network,
            device=device,
            num_classes=num_classes,
        )

    @torch.no_grad()
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return raw logits.

        Args:
            tensor: Input tensor of shape (B, 3, H, W) already on the
                correct device.

        Returns:
            Logits tensor of shape (B, C, H, W).
        """
        return self.network(tensor)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BaseModel(name='{self.name}', device='{self.device}', "
            f"num_classes={self.num_classes})"
        )
