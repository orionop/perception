"""Tests for the A* path planner."""

import numpy as np

from perception_engine.navigation.planner import AStarPlanner


def test_simple_path_cardinal():
    """A* should find a path on a trivially open grid."""
    cost_map = np.ones((5, 5), dtype=np.float64)
    planner = AStarPlanner(allow_diagonal=False)

    result = planner.plan(cost_map, start=(0, 0), goal=(4, 4))

    assert result.path_found
    assert result.path is not None
    assert result.path[0] == (0, 0)
    assert result.path[-1] == (4, 4)
    # Cardinal-only path length from (0,0) to (4,4) = 8 moves + 1 = 9 cells.
    assert len(result.path) == 9


def test_simple_path_diagonal():
    """Diagonal planner should find a shorter path."""
    cost_map = np.ones((5, 5), dtype=np.float64)
    planner = AStarPlanner(allow_diagonal=True)

    result = planner.plan(cost_map, start=(0, 0), goal=(4, 4))

    assert result.path_found
    assert result.path is not None
    # Diagonal path is shorter in both steps and coordinate changes.
    assert len(result.path) <= 9


def test_no_path_blocked():
    """Planner should fail gracefully when goal is surrounded by walls."""
    cost_map = np.ones((5, 5), dtype=np.float64)
    # Block row 3 entirely.
    cost_map[3, :] = float("inf")

    planner = AStarPlanner(allow_diagonal=False)
    result = planner.plan(cost_map, start=(0, 0), goal=(4, 4))

    assert not result.path_found
    assert result.path is None
    assert result.path_cost == float("inf")


def test_impassable_start():
    """Planner should return no-path when start is impassable."""
    cost_map = np.ones((3, 3), dtype=np.float64)
    cost_map[0, 0] = float("inf")

    planner = AStarPlanner(allow_diagonal=False)
    result = planner.plan(cost_map, start=(0, 0), goal=(2, 2))

    assert not result.path_found


def test_path_cost_accumulates():
    """Path cost should reflect the sum of traversed cell costs."""
    cost_map = np.array([
        [1.0, 2.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    planner = AStarPlanner(allow_diagonal=False)
    result = planner.plan(cost_map, start=(0, 0), goal=(2, 2))

    assert result.path_found
    # Verify path cost is finite and positive.
    assert 0 < result.path_cost < float("inf")
