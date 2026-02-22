"""Tests for segmentation metrics."""

import numpy as np

from perception_engine.evaluation.segmentation_metrics import (
    compute_segmentation_metrics,
)


def test_perfect_prediction():
    """Metrics should be 1.0 when prediction matches GT exactly."""
    mask = np.array([
        [0, 1, 2],
        [0, 1, 2],
    ], dtype=np.int32)

    metrics = compute_segmentation_metrics(
        prediction=mask,
        ground_truth=mask,
        num_classes=3,
    )

    assert metrics["pixel_accuracy"] == 1.0
    assert metrics["mean_iou"] == 1.0
    for iou in metrics["per_class_iou"].values():
        assert iou == 1.0


def test_completely_wrong_prediction():
    """mIoU should be 0 when prediction is fully incorrect."""
    gt = np.zeros((4, 4), dtype=np.int32)       # all class 0
    pred = np.ones((4, 4), dtype=np.int32)       # all class 1

    metrics = compute_segmentation_metrics(
        prediction=pred,
        ground_truth=gt,
        num_classes=2,
    )

    assert metrics["pixel_accuracy"] == 0.0
    assert metrics["mean_iou"] == 0.0


def test_confusion_matrix_shape():
    """Confusion matrix should be (num_classes, num_classes)."""
    gt = np.zeros((3, 3), dtype=np.int32)
    pred = np.zeros((3, 3), dtype=np.int32)

    metrics = compute_segmentation_metrics(
        prediction=pred,
        ground_truth=gt,
        num_classes=5,
    )

    assert metrics["confusion_matrix"].shape == (5, 5)


def test_class_names_in_output():
    """Per-class IoU keys should use class names when provided."""
    gt = np.array([[0, 1]], dtype=np.int32)
    pred = np.array([[0, 1]], dtype=np.int32)

    metrics = compute_segmentation_metrics(
        prediction=pred,
        ground_truth=gt,
        num_classes=2,
        class_names={0: "sand", 1: "rock"},
    )

    assert "sand" in metrics["per_class_iou"]
    assert "rock" in metrics["per_class_iou"]
