"""Tests for the explanation generator."""

import json
import tempfile
from pathlib import Path

import numpy as np

from perception_engine.core.data_types import (
    BenchmarkReport,
    NavigationResult,
    SafetyReport,
    SegmentationOutput,
)
from perception_engine.evaluation.explanation import (
    generate_explanation,
    print_explanation,
    save_explanation,
)


def _make_report(name="test_model", miou=0.72, path_found=True):
    seg = SegmentationOutput(
        mask=np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32),
        confidence_map=np.full((2, 3), 0.85, dtype=np.float32),
        probabilities=np.random.rand(3, 2, 3).astype(np.float32),
        inference_time_ms=25.0,
    )
    nav = NavigationResult(
        cost_map=np.ones((2, 3)),
        path=[(0, 0), (0, 1), (1, 2)] if path_found else None,
        path_cost=3.0 if path_found else float("inf"),
        path_found=path_found,
    )
    safety = SafetyReport(
        obstacle_overlap_pct=5.0,
        avg_confidence=0.85,
        path_cost=3.0,
        safety_score=0.88,
    ) if path_found else None

    return BenchmarkReport(
        model_name=name,
        segmentation_output=seg,
        navigation_result=nav,
        safety_report=safety,
        metrics={
            "pixel_accuracy": 0.80,
            "mean_iou": miou,
            "per_class_iou": {"tree": 0.65, "rock": 0.80, "sky": 0.70},
        },
        robustness={
            "brightness": {"mean_iou": 0.68, "miou_drop": 0.04, "inference_time_ms": 26.0},
            "noise": {"mean_iou": 0.60, "miou_drop": 0.12, "inference_time_ms": 28.0},
        },
    )


def test_explanation_structure():
    """Explanation should have overview, models, and sections."""
    reports = [_make_report()]
    config = {"class_names": ["tree", "rock", "sky"]}

    expl = generate_explanation(reports, config)

    assert "title" in expl
    assert "overview" in expl
    assert len(expl["models"]) == 1

    model_expl = expl["models"][0]
    assert model_expl["model_name"] == "test_model"
    titles = [s["title"] for s in model_expl["sections"]]

    assert any("scene" in t.lower() or "understand" in t.lower() for t in titles)
    assert any("fast" in t.lower() or "speed" in t.lower() for t in titles)
    assert any("safe" in t.lower() for t in titles)
    assert any("confident" in t.lower() or "confidence" in t.lower() for t in titles)


def test_explanation_comparison():
    """Multi-model should produce a comparison section."""
    reports = [_make_report("model_a", 0.72), _make_report("model_b", 0.65)]
    expl = generate_explanation(reports)

    assert "comparison" in expl
    assert "model_a" in expl["comparison"]["best_accuracy"]


def test_explanation_no_path():
    """No-path should explain why navigation failed."""
    reports = [_make_report(path_found=False)]
    expl = generate_explanation(reports)

    nav_section = [
        s for s in expl["models"][0]["sections"]
        if "path" in s["title"].lower()
    ]
    assert len(nav_section) == 1
    assert "❌" in nav_section[0]["metric"]


def test_save_explanation_files():
    """Should create both explanation.json and explanation.txt."""
    reports = [_make_report()]
    expl = generate_explanation(reports)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = save_explanation(expl, tmpdir)

        assert json_path.exists()
        assert (Path(tmpdir) / "explanation.txt").exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["title"] == "Perception Engine — Results Explained"


def test_print_explanation_runs(capsys):
    """print_explanation should produce stdout output."""
    reports = [_make_report()]
    expl = generate_explanation(reports)
    print_explanation(expl)

    captured = capsys.readouterr()
    assert "Results Explained" in captured.out
    assert "test_model" in captured.out
