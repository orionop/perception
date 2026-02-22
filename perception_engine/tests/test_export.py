"""Tests for the export module."""

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from perception_engine.data_types import (
    BenchmarkReport,
    NavigationResult,
    SafetyReport,
    SegmentationOutput,
)
from perception_engine.evaluation.export import (
    export_csv,
    export_json,
    export_per_class_csv,
    report_to_dict,
)


def _make_report(name: str = "test_model") -> BenchmarkReport:
    """Build a minimal BenchmarkReport for testing."""
    seg = SegmentationOutput(
        mask=np.array([[0, 1], [2, 0]], dtype=np.int32),
        confidence_map=np.array([[0.9, 0.8], [0.7, 0.95]], dtype=np.float32),
        probabilities=np.random.rand(3, 2, 2).astype(np.float32),
        inference_time_ms=15.3,
    )
    nav = NavigationResult(
        cost_map=np.ones((2, 2)),
        path=[(0, 0), (0, 1), (1, 1)],
        path_cost=3.0,
        path_found=True,
    )
    safety = SafetyReport(
        obstacle_overlap_pct=10.0,
        avg_confidence=0.85,
        path_cost=3.0,
        safety_score=0.82,
    )
    return BenchmarkReport(
        model_name=name,
        segmentation_output=seg,
        navigation_result=nav,
        safety_report=safety,
        metrics={
            "pixel_accuracy": 0.75,
            "mean_iou": 0.60,
            "per_class_iou": {"tree": 0.5, "rock": 0.7, "sky": 0.6},
            "confusion_matrix": np.eye(3, dtype=np.int32),
        },
        robustness={
            "brightness": {"mean_iou": 0.55, "miou_drop": 0.05, "inference_time_ms": 12.0},
            "noise": {"mean_iou": 0.50, "miou_drop": 0.10, "inference_time_ms": 14.0},
        },
    )


def test_report_to_dict_structure():
    """report_to_dict should return a complete, JSON-friendly dict."""
    d = report_to_dict(_make_report())

    assert d["model_name"] == "test_model"
    assert d["inference_time_ms"] == 15.3
    assert d["segmentation"]["num_classes"] == 3
    assert d["navigation"]["path_found"] is True
    assert d["navigation"]["path_length"] == 3
    assert d["safety"]["safety_score"] == 0.82
    assert d["metrics"]["mean_iou"] == 0.60
    assert "robustness" in d
    assert d["robustness"]["brightness"]["miou_drop"] == 0.05


def test_export_json_creates_valid_file():
    """JSON export should create a parseable file with config embedded."""
    reports = [_make_report("model_a"), _make_report("model_b")]
    config = {"device": "cpu", "models": [{"name": "a"}]}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_json(reports, Path(tmpdir) / "report.json", config=config)

        assert path.exists()
        with open(path) as f:
            data = json.load(f)

        assert data["num_models"] == 2
        assert len(data["reports"]) == 2
        assert data["reports"][0]["model_name"] == "model_a"
        assert "config" in data
        assert data["config"]["device"] == "cpu"


def test_export_csv_creates_valid_file():
    """CSV export should have header + 1 row per model."""
    reports = [_make_report("m1"), _make_report("m2")]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_csv(reports, Path(tmpdir) / "compare.csv")

        assert path.exists()
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["model"] == "m1"
        assert rows[1]["model"] == "m2"
        # Per-class IoU columns should be present.
        assert "iou_tree" in rows[0]
        assert "iou_rock" in rows[0]
        # Robustness columns.
        assert "rob_brightness_miou_drop" in rows[0]
        assert "rob_noise_miou_drop" in rows[0]


def test_export_per_class_csv():
    """Per-class CSV should have one row per (model, class) pair."""
    reports = [_make_report("m1")]  # 3 classes

    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_per_class_csv(reports, Path(tmpdir) / "perclass.csv")

        assert path.exists()
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # header + 3 data rows (tree, rock, sky)
        assert len(rows) == 4
        assert rows[0] == ["model", "class", "iou"]
        assert rows[1][0] == "m1"
        assert rows[1][1] == "tree"


def test_export_no_safety_no_robustness():
    """Export should handle reports with no safety or robustness gracefully."""
    seg = SegmentationOutput(
        mask=np.zeros((2, 2), dtype=np.int32),
        confidence_map=np.ones((2, 2), dtype=np.float32),
        probabilities=np.zeros((2, 2, 2), dtype=np.float32),
        inference_time_ms=5.0,
    )
    nav = NavigationResult(
        cost_map=np.ones((2, 2)),
        path=None,
        path_cost=float("inf"),
        path_found=False,
    )
    report = BenchmarkReport(
        model_name="bare",
        segmentation_output=seg,
        navigation_result=nav,
        safety_report=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = export_json([report], Path(tmpdir) / "bare.json")
        csv_path = export_csv([report], Path(tmpdir) / "bare.csv")

        assert json_path.exists()
        assert csv_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert "safety" not in data["reports"][0]
        assert data["reports"][0]["navigation"]["path_cost"] == "Infinity"
