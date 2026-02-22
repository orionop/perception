"""Tests for core data structures."""

from perception_engine.data_types import (
    BenchmarkReport,
    NavigationResult,
    SafetyReport,
    SegmentationOutput,
)

import numpy as np


def test_segmentation_output_creation():
    out = SegmentationOutput(
        mask=np.zeros((4, 4), dtype=np.int32),
        confidence_map=np.ones((4, 4), dtype=np.float32),
        probabilities=np.zeros((3, 4, 4), dtype=np.float32),
        inference_time_ms=12.5,
    )
    assert out.mask.shape == (4, 4)
    assert out.inference_time_ms == 12.5


def test_navigation_result_no_path():
    nav = NavigationResult(
        cost_map=np.ones((4, 4)),
        path=None,
        path_cost=float("inf"),
        path_found=False,
    )
    assert nav.path is None
    assert not nav.path_found


def test_safety_report():
    report = SafetyReport(
        obstacle_overlap_pct=5.0,
        avg_confidence=0.9,
        path_cost=50.0,
        safety_score=0.85,
    )
    assert 0.0 <= report.safety_score <= 1.0


def test_benchmark_report_defaults():
    seg = SegmentationOutput(
        mask=np.zeros((2, 2), dtype=np.int32),
        confidence_map=np.ones((2, 2), dtype=np.float32),
        probabilities=np.zeros((3, 2, 2), dtype=np.float32),
        inference_time_ms=10.0,
    )
    nav = NavigationResult(
        cost_map=np.ones((2, 2)),
        path=[(0, 0), (1, 1)],
        path_cost=2.0,
        path_found=True,
    )
    report = BenchmarkReport(
        model_name="test_model",
        segmentation_output=seg,
        navigation_result=nav,
        safety_report=None,
    )
    assert report.metrics == {}
    assert report.robustness is None
