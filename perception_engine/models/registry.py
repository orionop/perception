"""
Model registry.

Provides registration and lazy retrieval of segmentation models.
Models are registered from YAML config entries and instantiated on
first access so that weight loading and device transfers happen only
when actually needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from perception_engine.models.base_model import BaseModel
from perception_engine.models.loaders import load_weights

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for named segmentation model instances.

    Usage::

        registry = ModelRegistry(device="cpu")
        registry.register(model_cfg)
        model = registry.get("deeplab_v1")

    Models are lazily constructed: the config is stored at
    :meth:`register` time and the network is built only on the first
    :meth:`get` call.
    """

    def __init__(self, device: str) -> None:
        """
        Args:
            device: PyTorch device string for all registered models.
        """
        self._device = device
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._models: Dict[str, BaseModel] = {}

    def register(self, model_cfg: Dict[str, Any]) -> None:
        """Register a model config for later instantiation.

        Args:
            model_cfg: Dict with at least ``name``, ``architecture``,
                ``backbone``, and ``num_classes``.

        Raises:
            ValueError: If a model with the same name is already registered.
        """
        name = model_cfg["name"]
        if name in self._configs:
            raise ValueError(f"Model '{name}' is already registered.")
        self._configs[name] = model_cfg
        logger.info("Registered model config: '%s'", name)

    def get(self, name: str) -> BaseModel:
        """Retrieve (and build if necessary) a model by name.

        On the first call for a given name the model network is
        instantiated, weights are loaded (if a ``weights`` path is
        provided in the config), and the model is cached.

        Args:
            name: Model identifier matching a previously registered config.

        Returns:
            Ready-to-use ``BaseModel`` instance.

        Raises:
            KeyError: If no config was registered under ``name``.
        """
        if name in self._models:
            return self._models[name]

        if name not in self._configs:
            raise KeyError(
                f"Model '{name}' not found. "
                f"Available: {list(self._configs.keys())}"
            )

        cfg = self._configs[name]
        model = BaseModel.from_config(cfg, self._device)

        # Load pretrained weights if a path was provided.
        weights_path = cfg.get("weights")
        if weights_path:
            load_weights(model, weights_path, self._device)

        self._models[name] = model
        return model

    def list_models(self) -> List[str]:
        """Return the names of all registered models."""
        return list(self._configs.keys())

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        device: str,
    ) -> "ModelRegistry":
        """Build a registry and register all models from an experiment config.

        Args:
            config: Full experiment config dict (must contain ``"models"``).
            device: PyTorch device string.

        Returns:
            Populated ``ModelRegistry``.
        """
        registry = cls(device=device)
        for model_cfg in config["models"]:
            registry.register(model_cfg)
        return registry
