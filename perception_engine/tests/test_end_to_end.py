"""End-to-end tests — full pipeline with real models on synthetic data.

These tests instantiate actual segmentation models (with random weights
on ImageNet-initialised encoders), run inference on synthetic images,
and validate the complete pipeline output structure.

NOTE: These tests require `segmentation_models_pytorch` and PyTorch.
They run on CPU and use small images (64×64) for speed.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from perception_engine.configs.config_loader import get_device, load_config
from perception_engine.core.data_types import BenchmarkReport
from perception_engine.engine.inference_engine import InferenceEngine
from perception_engine.engine.postprocessing import postprocess_logits
from perception_engine.engine.preprocessing import preprocess_image
from perception_engine.evaluation.benchmarking import BenchmarkRunner
from perception_engine.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)
from perception_engine.models.base_model import BaseModel
from perception_engine.models.registry import ModelRegistry
from perception_engine.navigation.cost_mapping import build_cost_map
from perception_engine.navigation.planner import AStarPlanner
from perception_engine.navigation.safety import compute_safety_report
from perception_engine.visualization.overlays import (
    overlay_confidence,
    overlay_mask,
    overlay_path,
)


# ----- Fixtures -----

@pytest.fixture(scope="module")
def device() -> str:
    return "cpu"


@pytest.fixture(scope="module")
def synthetic_image() -> np.ndarray:
    """64×64 RGB image with random pixel values."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def synthetic_gt() -> np.ndarray:
    """64×64 ground-truth mask with 3 classes."""
    np.random.seed(42)
    return np.random.randint(0, 3, (64, 64), dtype=np.int32)


@pytest.fixture(scope="module")
def small_config() -> dict:
    """Minimal valid config for E2E tests with tiny images."""
    return {
        "models": [
            {
                "name": "test_unet",
                "architecture": "unet",
                "backbone": "resnet18",
                "num_classes": 3,
                "encoder_weights": "imagenet",
            },
        ],
        "class_names": ["background", "terrain", "obstacle"],
        "cost_mapping": {
            "traversable": [0, 1],
            "obstacle": [2],
        },
        "cost_values": {
            "traversable": 1.0,
            "obstacle": float("inf"),
        },
        "preprocessing": {
            "target_size": [64, 64],
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "planner": {
            "allow_diagonal": False,
            "start": [0, 0],
            "goal": None,
        },
        "safety": {
            "weight_obstacle": 0.4,
            "weight_confidence": 0.3,
            "weight_cost": 0.3,
            "max_acceptable_cost": 1000.0,
        },
        "robustness": {
            "enabled": False,
            "perturbations": [],
        },
        "device": "cpu",
        "output": {
            "save_visualizations": False,
            "output_dir": "test_outputs",
        },
    }


# ----- E2E Tests -----


def test_full_pipeline_synthetic(
    small_config, synthetic_image, synthetic_gt, device
):
    """Full pipeline on a single model with synthetic data."""
    # 1. Build model.
    model_cfg = small_config["models"][0]
    model = BaseModel.from_config(model_cfg, device)

    # 2. Run inference.
    engine = InferenceEngine(model, small_config, device)
    seg_output = engine.run(synthetic_image)

    assert seg_output.mask.shape == (64, 64)
    assert seg_output.confidence_map.shape == (64, 64)
    assert seg_output.probabilities.shape == (3, 64, 64)
    assert seg_output.inference_time_ms > 0

    # 3. Segmentation metrics.
    metrics = compute_segmentation_metrics(
        prediction=seg_output.mask,
        ground_truth=synthetic_gt,
        num_classes=3,
        class_names={0: "background", 1: "terrain", 2: "obstacle"},
    )
    assert "mean_iou" in metrics
    assert "pixel_accuracy" in metrics
    assert 0.0 <= metrics["pixel_accuracy"] <= 1.0
    assert 0.0 <= metrics["mean_iou"] <= 1.0

    # 4. Cost map.
    cost_map = build_cost_map(
        seg_output.mask,
        small_config["cost_mapping"],
        small_config["cost_values"],
    )
    assert cost_map.shape == (64, 64)

    # 5. Path planning.
    planner = AStarPlanner(allow_diagonal=False)
    nav = planner.plan(cost_map, start=(0, 0), goal=(63, 63))
    # Path may or may not be found depending on random mask — both are valid.
    assert isinstance(nav.path_found, bool)

    # 6. Safety report.
    if nav.path_found:
        safety = compute_safety_report(
            nav,
            seg_output.mask,
            seg_output.confidence_map,
            small_config["cost_mapping"],
            small_config["safety"],
        )
        assert safety is not None
        assert 0.0 <= safety.safety_score <= 1.0


def test_benchmark_two_models(small_config, synthetic_image, synthetic_gt, device):
    """BenchmarkRunner should produce a report per model."""
    # Add a second model.
    config = {**small_config}
    config["models"] = [
        {
            "name": "model_a",
            "architecture": "unet",
            "backbone": "resnet18",
            "num_classes": 3,
            "encoder_weights": "imagenet",
        },
        {
            "name": "model_b",
            "architecture": "fpn",
            "backbone": "resnet18",
            "num_classes": 3,
            "encoder_weights": "imagenet",
        },
    ]

    registry = ModelRegistry.from_config(config, device)
    runner = BenchmarkRunner(registry, config, device)
    reports = runner.run(synthetic_image, synthetic_gt)

    assert len(reports) == 2
    assert all(isinstance(r, BenchmarkReport) for r in reports)
    assert reports[0].model_name == "model_a"
    assert reports[1].model_name == "model_b"
    # Both should have metrics.
    for r in reports:
        assert "mean_iou" in r.metrics
        assert r.segmentation_output.inference_time_ms > 0


def test_visualization_outputs_saved(
    small_config, synthetic_image, device
):
    """Overlay functions should write image files to disk."""
    model_cfg = small_config["models"][0]
    model = BaseModel.from_config(model_cfg, device)
    engine = InferenceEngine(model, small_config, device)
    seg_output = engine.run(synthetic_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        mask_path = os.path.join(tmpdir, "mask.png")
        conf_path = os.path.join(tmpdir, "conf.png")
        path_path = os.path.join(tmpdir, "path.png")

        overlay_mask(
            image=synthetic_image,
            mask=seg_output.mask,
            num_classes=3,
            save_path=mask_path,
        )
        overlay_confidence(
            confidence_map=seg_output.confidence_map,
            save_path=conf_path,
        )

        cost_map = build_cost_map(
            seg_output.mask,
            small_config["cost_mapping"],
            small_config["cost_values"],
        )
        planner = AStarPlanner(allow_diagonal=False)
        nav = planner.plan(cost_map, start=(0, 0), goal=(63, 63))
        overlay_path(
            image=synthetic_image,
            cost_map=nav.cost_map,
            path=nav.path,
            save_path=path_path,
        )

        assert os.path.isfile(mask_path), "Mask overlay not saved"
        assert os.path.isfile(conf_path), "Confidence map not saved"
        assert os.path.isfile(path_path), "Path overlay not saved"

        # Files should have non-zero size.
        for p in [mask_path, conf_path, path_path]:
            assert os.path.getsize(p) > 0, f"Empty file: {p}"


def test_e2e_with_robustness_enabled(
    small_config, synthetic_image, synthetic_gt, device
):
    """Full benchmark with robustness enabled should populate robustness field."""
    config = {**small_config}
    config["robustness"] = {
        "enabled": True,
        "perturbations": ["brightness", "noise"],
    }

    registry = ModelRegistry.from_config(config, device)
    runner = BenchmarkRunner(registry, config, device)
    reports = runner.run(synthetic_image, synthetic_gt)

    assert len(reports) == 1
    report = reports[0]

    # Robustness field should be populated.
    assert report.robustness is not None
    assert "brightness" in report.robustness
    assert "noise" in report.robustness

    for pert_name, pert_data in report.robustness.items():
        assert "mean_iou" in pert_data
        assert "miou_drop" in pert_data
        assert "pixel_accuracy" in pert_data
        assert 0.0 <= pert_data["mean_iou"] <= 1.0
        assert pert_data["inference_time_ms"] > 0

