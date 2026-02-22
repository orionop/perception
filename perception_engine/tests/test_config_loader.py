"""Tests for the configuration loader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from perception_engine.config_loader import (
    ConfigValidationError,
    get_device,
    load_config,
)


def _write_yaml(data: dict) -> str:
    """Write a dict as YAML to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.dump(data, f)
    return path


def _minimal_valid_config() -> dict:
    """Return the smallest valid config that passes validation."""
    return {
        "models": [
            {
                "name": "test_model",
                "architecture": "unet",
                "backbone": "resnet18",
                "num_classes": 3,
            }
        ],
        "class_names": ["a", "b", "c"],
        "cost_mapping": {
            "traversable": [0],
            "obstacle": [1],
        },
        "planner": {"allow_diagonal": False},
    }


def test_load_valid_config():
    """Loading the example experiment.yaml should succeed."""
    config_path = (
        Path(__file__).parent.parent / "configs" / "experiment.yaml"
    )
    config = load_config(str(config_path))
    assert "models" in config
    assert len(config["models"]) >= 1


def test_missing_required_section():
    """Omitting a required section should raise ConfigValidationError."""
    bad = _minimal_valid_config()
    del bad["cost_mapping"]
    path = _write_yaml(bad)

    with pytest.raises(ConfigValidationError, match="cost_mapping"):
        load_config(path)

    os.unlink(path)


def test_missing_model_keys():
    """A model missing 'backbone' should fail validation."""
    bad = _minimal_valid_config()
    del bad["models"][0]["backbone"]
    path = _write_yaml(bad)

    with pytest.raises(ConfigValidationError, match="backbone"):
        load_config(path)

    os.unlink(path)


def test_defaults_applied():
    """Defaults should be filled in for optional sections."""
    path = _write_yaml(_minimal_valid_config())
    config = load_config(path)

    # device default
    assert config["device"] == "auto"
    # preprocessing defaults
    assert config["preprocessing"]["target_size"] == [512, 512]
    # safety defaults
    assert "weight_obstacle" in config["safety"]
    # robustness defaults
    assert config["robustness"]["enabled"] is False

    os.unlink(path)


def test_device_auto_detect():
    """get_device should return a valid device string."""
    config = {"device": "auto"}
    device = get_device(config)
    assert device in ("cpu", "cuda", "mps")


def test_device_explicit():
    """Explicit device should be returned as-is."""
    config = {"device": "cpu"}
    assert get_device(config) == "cpu"


def test_file_not_found():
    """Loading a nonexistent config should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_empty_config():
    """An empty YAML file should raise ConfigValidationError."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)  # empty file

    with pytest.raises(ConfigValidationError, match="empty"):
        load_config(path)

    os.unlink(path)
