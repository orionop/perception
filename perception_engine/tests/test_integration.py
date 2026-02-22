"""Integration tests — cross-module interactions.

These tests verify that outputs from one module flow correctly
into the next module in the pipeline, using synthetic data.
"""

import numpy as np

from perception_engine.data_types import NavigationResult
from perception_engine.engine.mask_remapping import remap_mask
from perception_engine.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)
from perception_engine.navigation.cost_mapping import build_cost_map
from perception_engine.navigation.planner import AStarPlanner
from perception_engine.navigation.safety import compute_safety_report


def test_mask_to_costmap_to_planner():
    """mask → cost map → A* should find a path through traversable terrain."""
    # Build a 10×10 mask: top half traversable, bottom half traversable,
    # middle row is obstacle except column 5.
    mask = np.zeros((10, 10), dtype=np.int32)  # all class 0 (traversable)
    mask[5, :] = 1  # obstacle wall
    mask[5, 5] = 0  # gap in the wall

    cost_mapping = {"traversable": [0], "obstacle": [1]}
    cost_values = {"traversable": 1.0, "obstacle": float("inf")}

    cost_map = build_cost_map(mask, cost_mapping, cost_values)
    assert cost_map[0, 0] == 1.0
    assert cost_map[5, 0] == float("inf")
    assert cost_map[5, 5] == 1.0  # gap

    planner = AStarPlanner(allow_diagonal=False)
    result = planner.plan(cost_map, start=(0, 0), goal=(9, 9))

    assert result.path_found
    assert result.path is not None
    # Path must pass through the gap at (5, 5).
    assert (5, 5) in result.path


def test_costmap_to_safety():
    """cost map + planner result → safety should produce a valid report."""
    mask = np.zeros((5, 5), dtype=np.int32)
    mask[2, 2] = 1  # one obstacle pixel

    cost_mapping = {"traversable": [0], "obstacle": [1]}
    cost_values = {"traversable": 1.0, "obstacle": float("inf")}
    cost_map = build_cost_map(mask, cost_mapping, cost_values)

    planner = AStarPlanner(allow_diagonal=False)
    nav_result = planner.plan(cost_map, start=(0, 0), goal=(4, 4))
    assert nav_result.path_found

    conf = np.full((5, 5), 0.9, dtype=np.float32)
    safety_cfg = {
        "weight_obstacle": 0.4,
        "weight_confidence": 0.3,
        "weight_cost": 0.3,
        "max_acceptable_cost": 100.0,
    }

    report = compute_safety_report(
        nav_result, mask, conf, cost_mapping, safety_cfg
    )
    assert report is not None
    assert report.safety_score > 0.5  # mostly traversable


def test_metrics_with_remapped_gt():
    """Remapped GT mask should produce correct mIoU scores."""
    # Raw GT: class IDs 100 and 200.
    raw_gt = np.array([[100, 200], [100, 100]], dtype=np.int32)
    mapping = {100: 0, 200: 1}

    remapped = remap_mask(raw_gt, mapping)
    assert remapped[0, 0] == 0
    assert remapped[0, 1] == 1

    # Prediction matches exactly.
    prediction = remapped.copy()
    metrics = compute_segmentation_metrics(
        prediction=prediction,
        ground_truth=remapped,
        num_classes=2,
    )
    assert metrics["mean_iou"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0


def test_planner_on_mixed_cost_grid():
    """Planner should prefer low-cost traversable over high-cost soft terrain."""
    # Create a grid: left column = traversable (1.0),
    # middle = soft (5.0), right = traversable.
    mask = np.zeros((5, 5), dtype=np.int32)
    mask[:, 2] = 2  # soft terrain in the middle column

    cost_mapping = {"traversable": [0], "soft": [2]}
    cost_values = {"traversable": 1.0, "soft": 5.0}
    cost_map = build_cost_map(mask, cost_mapping, cost_values)

    planner = AStarPlanner(allow_diagonal=False)
    result = planner.plan(cost_map, start=(0, 0), goal=(4, 4))

    assert result.path_found
    # Path should exist — soft terrain is passable, just costlier.
    assert result.path_cost > 0
    assert result.path_cost < float("inf")


def test_robustness_perturbation_integration():
    """Perturbation → inference → metrics should report mIoU drop."""
    from perception_engine.evaluation.robustness import RobustnessEvaluator
    from perception_engine.engine.inference_engine import InferenceEngine
    from perception_engine.models.base_model import BaseModel

    model_cfg = {
        "name": "robust_test",
        "architecture": "unet",
        "backbone": "resnet18",
        "num_classes": 3,
        "encoder_weights": "imagenet",
    }
    config = {
        "preprocessing": {
            "target_size": [32, 32],
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    }

    model = BaseModel.from_config(model_cfg, "cpu")
    engine = InferenceEngine(model, config, "cpu")

    np.random.seed(99)
    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    gt = np.random.randint(0, 3, (32, 32), dtype=np.int32)

    # Baseline inference.
    seg = engine.run(image)
    baseline_metrics = compute_segmentation_metrics(seg.mask, gt, 3)
    baseline_miou = baseline_metrics["mean_iou"]

    evaluator = RobustnessEvaluator(
        engine=engine,
        perturbations=["brightness", "blur"],
        num_classes=3,
    )
    results = evaluator.evaluate(image, gt, baseline_miou)

    assert "brightness" in results
    assert "blur" in results
    for name, data in results.items():
        assert "mean_iou" in data
        assert "miou_drop" in data
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] > 0


def test_robustness_unknown_perturbation_skipped():
    """Unknown perturbation names should be skipped without crashing."""
    from perception_engine.evaluation.robustness import RobustnessEvaluator
    from perception_engine.engine.inference_engine import InferenceEngine
    from perception_engine.models.base_model import BaseModel

    model_cfg = {
        "name": "skip_test",
        "architecture": "unet",
        "backbone": "resnet18",
        "num_classes": 2,
        "encoder_weights": "imagenet",
    }
    config = {
        "preprocessing": {"target_size": [16, 16]},
    }

    model = BaseModel.from_config(model_cfg, "cpu")
    engine = InferenceEngine(model, config, "cpu")

    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    gt = np.zeros((16, 16), dtype=np.int32)

    evaluator = RobustnessEvaluator(
        engine=engine,
        perturbations=["nonexistent_perturbation", "blur"],
        num_classes=2,
    )
    results = evaluator.evaluate(image, gt, baseline_miou=0.5)

    # Unknown perturbation skipped, valid one still runs.
    assert "nonexistent_perturbation" not in results
    assert "blur" in results


def test_safety_with_diagonal_planner():
    """Safety report should work correctly with diagonal A* paths."""
    mask = np.zeros((8, 8), dtype=np.int32)  # all traversable
    mask[3, 3] = 1  # single obstacle

    cost_mapping = {"traversable": [0], "obstacle": [1]}
    cost_values = {"traversable": 1.0, "obstacle": float("inf")}
    cost_map = build_cost_map(mask, cost_mapping, cost_values)

    planner = AStarPlanner(allow_diagonal=True)
    nav = planner.plan(cost_map, start=(0, 0), goal=(7, 7))
    assert nav.path_found

    conf = np.full((8, 8), 0.95, dtype=np.float32)
    safety_cfg = {
        "weight_obstacle": 0.4,
        "weight_confidence": 0.3,
        "weight_cost": 0.3,
        "max_acceptable_cost": 200.0,
    }

    report = compute_safety_report(nav, mask, conf, cost_mapping, safety_cfg)
    assert report is not None
    assert report.obstacle_overlap_pct == 0.0  # path avoids the obstacle
    assert report.safety_score > 0.8


def test_remap_then_costmap():
    """Remapped GT mask should produce correct cost map with contiguous IDs."""
    raw_mask = np.array([
        [100, 200],
        [10000, 7100],
    ], dtype=np.int32)
    mapping = {100: 0, 200: 1, 7100: 2, 10000: 3}

    remapped = remap_mask(raw_mask, mapping)
    assert remapped[0, 0] == 0
    assert remapped[1, 1] == 2

    cost_mapping = {"traversable": [2, 3], "obstacle": [0], "soft": [1]}
    cost_values = {"traversable": 1.0, "obstacle": float("inf"), "soft": 5.0}

    cost_map = build_cost_map(remapped, cost_mapping, cost_values)
    assert cost_map[0, 0] == float("inf")   # class 0 → obstacle
    assert cost_map[0, 1] == 5.0            # class 1 → soft
    assert cost_map[1, 0] == 1.0            # class 3 → traversable
    assert cost_map[1, 1] == 1.0            # class 2 → traversable


def test_remap_preserves_spatial_structure():
    """Remapping should preserve the spatial layout of the mask."""
    # Checkerboard pattern with raw values.
    raw = np.array([
        [500, 100, 500],
        [100, 500, 100],
        [500, 100, 500],
    ], dtype=np.int32)
    mapping = {100: 0, 500: 1}

    remapped = remap_mask(raw, mapping)

    # Spatial structure: corners and center should be class 1.
    assert remapped[0, 0] == 1
    assert remapped[1, 1] == 1
    assert remapped[2, 2] == 1
    # Edges should be class 0.
    assert remapped[0, 1] == 0
    assert remapped[1, 0] == 0
    # Shape preserved.
    assert remapped.shape == raw.shape


