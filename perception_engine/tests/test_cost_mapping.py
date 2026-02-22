"""Tests for the cost mapping module."""

import numpy as np

from perception_engine.navigation.cost_mapping import build_cost_map, get_obstacle_mask


def test_build_cost_map_basic():
    """Verify basic class-to-cost mapping."""
    mask = np.array([
        [0, 1, 2],
        [3, 4, 0],
    ], dtype=np.int32)

    cost_mapping = {
        "traversable": [0, 2],
        "obstacle": [1, 3],
        "soft": [4],
    }
    cost_values = {
        "traversable": 1.0,
        "obstacle": float("inf"),
        "soft": 5.0,
    }

    cost_map = build_cost_map(mask, cost_mapping, cost_values)

    assert cost_map[0, 0] == 1.0   # class 0 → traversable
    assert cost_map[0, 1] == float("inf")  # class 1 → obstacle
    assert cost_map[0, 2] == 1.0   # class 2 → traversable
    assert cost_map[1, 0] == float("inf")  # class 3 → obstacle
    assert cost_map[1, 1] == 5.0   # class 4 → soft
    assert cost_map[1, 2] == 1.0   # class 0 → traversable


def test_build_cost_map_unmapped_goes_to_inf():
    """Verify that unmapped class IDs default to infinity."""
    mask = np.array([[99]], dtype=np.int32)
    cost_mapping = {"traversable": [0]}
    cost_values = {"traversable": 1.0}

    cost_map = build_cost_map(mask, cost_mapping, cost_values)
    assert cost_map[0, 0] == float("inf")


def test_get_obstacle_mask():
    """Verify the boolean obstacle mask."""
    mask = np.array([
        [0, 1],
        [2, 3],
    ], dtype=np.int32)
    cost_mapping = {"obstacle": [1, 3]}

    obs = get_obstacle_mask(mask, cost_mapping)
    assert obs[0, 0] is np.bool_(False)
    assert obs[0, 1] is np.bool_(True)
    assert obs[1, 0] is np.bool_(False)
    assert obs[1, 1] is np.bool_(True)


def test_cost_map_large_sparse_class_ids():
    """LUT should handle large sparse IDs (e.g., 7100, 10000) efficiently."""
    mask = np.array([
        [100, 10000],
        [7100, 200],
    ], dtype=np.int32)

    cost_mapping = {
        "traversable": [10000, 7100],
        "obstacle": [100],
        "soft": [200],
    }
    cost_values = {
        "traversable": 1.0,
        "obstacle": float("inf"),
        "soft": 5.0,
    }

    cost_map = build_cost_map(mask, cost_mapping, cost_values)

    assert cost_map[0, 0] == float("inf")   # 100 → obstacle
    assert cost_map[0, 1] == 1.0            # 10000 → traversable
    assert cost_map[1, 0] == 1.0            # 7100 → traversable
    assert cost_map[1, 1] == 5.0            # 200 → soft
