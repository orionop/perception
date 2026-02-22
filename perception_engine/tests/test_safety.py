"""Tests for navigation safety metrics."""

import numpy as np

from perception_engine.core.data_types import NavigationResult
from perception_engine.navigation.safety import compute_safety_report


def _make_nav_result(
    path: list, cost_map: np.ndarray, path_cost: float = 10.0
) -> NavigationResult:
    return NavigationResult(
        cost_map=cost_map,
        path=path,
        path_cost=path_cost,
        path_found=True,
    )


def _default_safety_cfg() -> dict:
    return {
        "weight_obstacle": 0.4,
        "weight_confidence": 0.3,
        "weight_cost": 0.3,
        "max_acceptable_cost": 1000.0,
    }


def test_safety_score_all_traversable():
    """A clean path with no obstacles and high confidence → score ≈ 1.0."""
    mask = np.zeros((5, 5), dtype=np.int32)       # all class 0
    conf = np.ones((5, 5), dtype=np.float32)       # full confidence
    cost_map = np.ones((5, 5), dtype=np.float64)
    path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
    cost_mapping = {"obstacle": [1]}  # class 0 not an obstacle

    nav = _make_nav_result(path, cost_map, path_cost=5.0)
    report = compute_safety_report(
        nav, mask, conf, cost_mapping, _default_safety_cfg()
    )

    assert report is not None
    assert report.obstacle_overlap_pct == 0.0
    assert report.avg_confidence == 1.0
    # score = 1.0 - 0 - 0 - 0.3*(5/1000) ≈ 0.9985
    assert report.safety_score > 0.99


def test_safety_score_high_obstacle_overlap():
    """100% obstacle overlap → score is heavily penalised."""
    mask = np.ones((5, 5), dtype=np.int32)         # all class 1
    conf = np.full((5, 5), 0.5, dtype=np.float32)
    cost_map = np.ones((5, 5), dtype=np.float64)
    path = [(0, 0), (1, 0), (2, 0)]
    cost_mapping = {"obstacle": [1]}  # 100% overlap

    nav = _make_nav_result(path, cost_map, path_cost=3.0)
    report = compute_safety_report(
        nav, mask, conf, cost_mapping, _default_safety_cfg()
    )

    assert report is not None
    assert report.obstacle_overlap_pct == 100.0
    # score = 1.0 - 0.4*1.0 - 0.3*0.5 - 0.3*(3/1000) ≈ 0.4491
    assert report.safety_score < 0.5


def test_safety_no_path():
    """No path found → safety report is None."""
    nav = NavigationResult(
        cost_map=np.ones((3, 3)),
        path=None,
        path_cost=float("inf"),
        path_found=False,
    )
    mask = np.zeros((3, 3), dtype=np.int32)
    conf = np.ones((3, 3), dtype=np.float32)

    report = compute_safety_report(
        nav, mask, conf, {"obstacle": [1]}, _default_safety_cfg()
    )
    assert report is None


def test_safety_score_clamped():
    """Safety score should always be in [0, 1] even with extreme values."""
    mask = np.ones((3, 3), dtype=np.int32)
    conf = np.zeros((3, 3), dtype=np.float32)  # zero confidence
    cost_map = np.ones((3, 3), dtype=np.float64)
    path = [(0, 0), (1, 0), (2, 0)]
    cost_mapping = {"obstacle": [1]}

    nav = _make_nav_result(path, cost_map, path_cost=99999.0)
    report = compute_safety_report(
        nav, mask, conf, cost_mapping, _default_safety_cfg()
    )

    assert report is not None
    assert 0.0 <= report.safety_score <= 1.0


def test_custom_weights():
    """Different weight configs should produce different scores."""
    mask = np.array([[0, 1], [0, 0]], dtype=np.int32)
    conf = np.full((2, 2), 0.8, dtype=np.float32)
    cost_map = np.ones((2, 2), dtype=np.float64)
    path = [(0, 0), (0, 1)]
    cost_mapping = {"obstacle": [1]}
    nav = _make_nav_result(path, cost_map, path_cost=2.0)

    cfg_a = {
        "weight_obstacle": 0.8,   # heavily penalise obstacles
        "weight_confidence": 0.1,
        "weight_cost": 0.1,
        "max_acceptable_cost": 100.0,
    }
    cfg_b = {
        "weight_obstacle": 0.1,   # barely penalise obstacles
        "weight_confidence": 0.1,
        "weight_cost": 0.8,
        "max_acceptable_cost": 100.0,
    }

    report_a = compute_safety_report(nav, mask, conf, cost_mapping, cfg_a)
    report_b = compute_safety_report(nav, mask, conf, cost_mapping, cfg_b)

    assert report_a is not None and report_b is not None
    # With 50% obstacle overlap, cfg_a penalises harder.
    assert report_a.safety_score < report_b.safety_score
