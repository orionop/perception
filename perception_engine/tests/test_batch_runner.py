"""Tests for the batch evaluation runner."""

import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from perception_engine.evaluation.batch_runner import BatchRunner
from perception_engine.models.registry import ModelRegistry


@pytest.fixture(scope="module")
def batch_config():
    """Minimal config for batch tests."""
    return {
        "models": [
            {
                "name": "batch_unet",
                "architecture": "unet",
                "backbone": "resnet18",
                "num_classes": 3,
                "encoder_weights": "imagenet",
            },
        ],
        "class_names": ["bg", "terrain", "obstacle"],
        "cost_mapping": {
            "traversable": [0, 1],
            "obstacle": [2],
        },
        "cost_values": {
            "traversable": 1.0,
            "obstacle": float("inf"),
        },
        "preprocessing": {
            "target_size": [32, 32],
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "planner": {"allow_diagonal": False, "start": [0, 0], "goal": None},
        "safety": {
            "weight_obstacle": 0.4,
            "weight_confidence": 0.3,
            "weight_cost": 0.3,
            "max_acceptable_cost": 1000.0,
        },
        "robustness": {"enabled": False, "perturbations": []},
        "device": "cpu",
    }


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Create a temp directory with 3 synthetic images + GT masks."""
    tmpdir = tempfile.mkdtemp()
    img_dir = Path(tmpdir) / "images"
    gt_dir = Path(tmpdir) / "masks"
    img_dir.mkdir()
    gt_dir.mkdir()

    np.random.seed(42)
    for i in range(3):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        mask = np.random.randint(0, 3, (32, 32), dtype=np.uint8)

        Image.fromarray(img).save(img_dir / f"img_{i:03d}.png")
        Image.fromarray(mask).save(gt_dir / f"img_{i:03d}.png")

    yield str(img_dir), str(gt_dir), tmpdir

    # Cleanup.
    import shutil
    shutil.rmtree(tmpdir)


def test_batch_runs_all_images(batch_config, synthetic_dataset):
    """Batch runner should process all images and return aggregate stats."""
    img_dir, gt_dir, _ = synthetic_dataset

    registry = ModelRegistry.from_config(batch_config, "cpu")
    runner = BatchRunner(registry, batch_config, "cpu")

    result = runner.run(
        image_dir=img_dir,
        gt_dir=gt_dir,
        image_ext=".png",
        gt_ext=".png",
    )

    assert result["total_images"] == 3
    assert "batch_unet" in result["models"]

    stats = result["models"]["batch_unet"]
    assert "mean_iou" in stats
    assert "pixel_accuracy" in stats
    assert "inference_time_ms" in stats
    assert 0.0 <= stats["mean_iou"]["mean"] <= 1.0


def test_batch_max_samples(batch_config, synthetic_dataset):
    """max_samples should cap the number of images evaluated."""
    img_dir, gt_dir, _ = synthetic_dataset

    registry = ModelRegistry.from_config(batch_config, "cpu")
    runner = BatchRunner(registry, batch_config, "cpu")

    result = runner.run(
        image_dir=img_dir,
        gt_dir=gt_dir,
        max_samples=2,
    )

    # Only 2 images processed even though 3 exist.
    assert result["total_images"] == 2


def test_batch_saves_reports(batch_config, synthetic_dataset):
    """JSON and CSV reports should be created in the output directory."""
    img_dir, gt_dir, tmpdir = synthetic_dataset
    out_dir = Path(tmpdir) / "output"

    registry = ModelRegistry.from_config(batch_config, "cpu")
    runner = BatchRunner(registry, batch_config, "cpu")

    runner.run(
        image_dir=img_dir,
        gt_dir=gt_dir,
        output_dir=out_dir,
    )

    json_path = out_dir / "batch_report.json"
    csv_path = out_dir / "batch_summary.csv"

    assert json_path.exists(), "batch_report.json not created"
    assert csv_path.exists(), "batch_summary.csv not created"

    # JSON should be valid.
    with open(json_path) as f:
        data = json.load(f)
    assert data["total_images"] == 3
    assert "batch_unet" in data["models"]

    # CSV should have header + 1 data row.
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 2  # header + 1 model
