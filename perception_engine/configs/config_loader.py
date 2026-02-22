"""
Configuration loader with validation.

Reads YAML experiment configs, validates required sections, and provides
typed accessors. Fails fast on missing or malformed configuration to
prevent silent downstream errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Sections that MUST exist in every experiment config.
_REQUIRED_SECTIONS = {"models", "class_names", "cost_mapping", "planner"}

# Keys required on every model definition.
_REQUIRED_MODEL_KEYS = {"name", "architecture", "backbone", "num_classes"}


class ConfigValidationError(Exception):
    """Raised when the experiment config is invalid."""


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate a YAML experiment configuration.

    Args:
        path: Filesystem path to the YAML config file.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigValidationError: If required sections or keys are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ConfigValidationError("Config file is empty.")

    _validate(config)
    config = _apply_defaults(config)
    logger.info("Configuration loaded from %s", config_path)
    return config


def _validate(config: Dict[str, Any]) -> None:
    """Validate the top-level structure and model definitions."""
    missing = _REQUIRED_SECTIONS - set(config.keys())
    if missing:
        raise ConfigValidationError(
            f"Missing required config sections: {missing}"
        )

    # Validate each model entry.
    models: List[Dict[str, Any]] = config["models"]
    if not isinstance(models, list) or len(models) == 0:
        raise ConfigValidationError(
            "'models' must be a non-empty list of model definitions."
        )

    for idx, model_def in enumerate(models):
        missing_keys = _REQUIRED_MODEL_KEYS - set(model_def.keys())
        if missing_keys:
            raise ConfigValidationError(
                f"Model at index {idx} ('{model_def.get('name', '?')}') "
                f"is missing keys: {missing_keys}"
            )

    # Validate cost_mapping categories.
    cost_mapping = config["cost_mapping"]
    required_cost_keys = {"traversable", "obstacle"}
    missing_cost = required_cost_keys - set(cost_mapping.keys())
    if missing_cost:
        raise ConfigValidationError(
            f"cost_mapping must contain at least: {required_cost_keys}. "
            f"Missing: {missing_cost}"
        )


def _apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply sensible defaults for optional configuration sections."""

    # Device defaults to auto-detect.
    config.setdefault("device", "auto")

    # Preprocessing defaults.
    preprocessing = config.setdefault("preprocessing", {})
    preprocessing.setdefault("target_size", [512, 512])
    preprocessing.setdefault(
        "normalize",
        {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    )

    # Planner defaults.
    planner = config["planner"]
    planner.setdefault("allow_diagonal", False)
    planner.setdefault("start", [0, 0])
    planner.setdefault("goal", None)  # None → bottom-right corner

    # Cost values.
    cost_values = config.setdefault("cost_values", {})
    cost_values.setdefault("traversable", 1.0)
    cost_values.setdefault("obstacle", float("inf"))
    cost_values.setdefault("soft", 5.0)
    cost_values.setdefault("ignored", float("inf"))

    # Robustness defaults.
    robustness = config.setdefault("robustness", {})
    robustness.setdefault("enabled", False)
    robustness.setdefault("perturbations", [])

    # Safety score weights.
    safety = config.setdefault("safety", {})
    safety.setdefault("weight_obstacle", 0.4)
    safety.setdefault("weight_confidence", 0.3)
    safety.setdefault("weight_cost", 0.3)
    safety.setdefault("max_acceptable_cost", 1000.0)

    # Output settings.
    output = config.setdefault("output", {})
    output.setdefault("save_visualizations", True)
    output.setdefault("output_dir", "outputs")

    return config


def get_device(config: Dict[str, Any]) -> str:
    """Resolve the compute device from config.

    Auto-detect order: CUDA → MPS → CPU.

    Args:
        config: Loaded experiment config.

    Returns:
        PyTorch device string (e.g., ``"cuda"``, ``"mps"``, ``"cpu"``).
    """
    import torch

    requested = config.get("device", "auto")

    if requested != "auto":
        logger.info("Using explicitly configured device: %s", requested)
        return requested

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info("Auto-detected device: %s", device)
    return device
