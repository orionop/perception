"""
Grid-based path planners.

Provides a common ``BasePlanner`` interface and concrete implementations:
    - **AStarPlanner** — optimal shortest path on a cost grid
    - **PotentialFieldPlanner** — gradient-descent on an artificial potential field
    - **RRTStarPlanner** — sampling-based planner for complex obstacle fields

All planners are selectable via the ``planner.strategy`` config key.
"""

from __future__ import annotations

import abc
import heapq
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from perception_engine.data_types import NavigationResult

logger = logging.getLogger(__name__)

class BasePlanner(abc.ABC):
    """Abstract base class for all path planners."""

    @abc.abstractmethod
    def plan(
        self,
        cost_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> NavigationResult:
        """Compute a path from *start* to *goal* on the cost grid."""

    @staticmethod
    def _no_path(cost_map: np.ndarray) -> NavigationResult:
        return NavigationResult(
            cost_map=cost_map,
            path=None,
            path_cost=float("inf"),
            path_found=False,
        )

    @staticmethod
    def _in_bounds(
        pos: Tuple[int, int], rows: int, cols: int
    ) -> bool:
        return 0 <= pos[0] < rows and 0 <= pos[1] < cols


# Movement directions: (delta_row, delta_col, cost_multiplier).
_CARDINAL = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
]

_DIAGONAL = [
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
]


class AStarPlanner(BasePlanner):
    """A* planner on a 2-D traversability cost grid."""

    def __init__(self, allow_diagonal: bool = False) -> None:
        self.allow_diagonal = allow_diagonal
        self._moves = _CARDINAL + (_DIAGONAL if allow_diagonal else [])

    def plan(
        self,
        cost_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> NavigationResult:
        rows, cols = cost_map.shape

        if not self._in_bounds(start, rows, cols):
            logger.error("Start %s is out of bounds (%d, %d).", start, rows, cols)
            return self._no_path(cost_map)
        if not self._in_bounds(goal, rows, cols):
            logger.error("Goal %s is out of bounds (%d, %d).", goal, rows, cols)
            return self._no_path(cost_map)

        if not np.isfinite(cost_map[start]):
            logger.warning("Start cell %s is impassable (cost=inf).", start)
            return self._no_path(cost_map)
        if not np.isfinite(cost_map[goal]):
            logger.warning("Goal cell %s is impassable (cost=inf).", goal)
            return self._no_path(cost_map)

        open_set: list = []
        counter = 0
        heapq.heappush(open_set, (0.0, counter, start))

        g_score = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed = set()

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current == goal:
                path = self._reconstruct(came_from, current)
                total_cost = g_score[current]
                logger.info(
                    "Path found: %d cells, total cost %.2f",
                    len(path),
                    total_cost,
                )
                return NavigationResult(
                    cost_map=cost_map,
                    path=path,
                    path_cost=total_cost,
                    path_found=True,
                )

            if current in closed:
                continue
            closed.add(current)

            for dr, dc, move_cost in self._moves:
                nr, nc = current[0] + dr, current[1] + dc

                if not self._in_bounds((nr, nc), rows, cols):
                    continue
                if (nr, nc) in closed:
                    continue

                cell_cost = cost_map[nr, nc]
                if not np.isfinite(cell_cost):
                    continue

                tentative_g = g_score[current] + cell_cost * move_cost

                if tentative_g < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic((nr, nc), goal)
                    came_from[(nr, nc)] = current
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))

        logger.warning("No path found from %s to %s.", start, goal)
        return self._no_path(cost_map)

    def _heuristic(
        self, a: Tuple[int, int], b: Tuple[int, int]
    ) -> float:
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        if self.allow_diagonal:
            return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)
        return float(dr + dc)

    @staticmethod
    def _reconstruct(
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AStarPlanner":
        planner_cfg = config.get("planner", {})
        return cls(allow_diagonal=planner_cfg.get("allow_diagonal", False))


# ------------------------------------------------------------------
# Potential Field Planner
# ------------------------------------------------------------------


class PotentialFieldPlanner(BasePlanner):
    """Artificial potential field planner.

    Uses an attractive potential toward the goal and a repulsive
    potential from obstacles to generate a gradient the robot follows.
    """

    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 100.0,
        repulsive_range: int = 10,
        step_size: int = 1,
        max_iterations: int = 5000,
    ) -> None:
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.repulsive_range = repulsive_range
        self.step_size = step_size
        self.max_iterations = max_iterations

    def plan(
        self,
        cost_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> NavigationResult:
        rows, cols = cost_map.shape

        if not self._in_bounds(start, rows, cols) or not self._in_bounds(goal, rows, cols):
            return self._no_path(cost_map)
        if not np.isfinite(cost_map[start]):
            logger.warning("PotentialField: start %s is impassable.", start)
            return self._no_path(cost_map)

        # Precompute obstacle mask.
        obstacle_mask = ~np.isfinite(cost_map)

        path = [start]
        current = start
        visited = {start}

        for _ in range(self.max_iterations):
            if current == goal:
                total_cost = sum(
                    cost_map[p] for p in path if np.isfinite(cost_map[p])
                )
                logger.info(
                    "PotentialField path found: %d cells, cost %.2f",
                    len(path), total_cost,
                )
                return NavigationResult(
                    cost_map=cost_map,
                    path=path,
                    path_cost=total_cost,
                    path_found=True,
                )

            # Compute forces on neighbors.
            best_next = None
            best_potential = float("inf")

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = current[0] + dr, current[1] + dc
                    if not self._in_bounds((nr, nc), rows, cols):
                        continue
                    if not np.isfinite(cost_map[nr, nc]):
                        continue
                    if (nr, nc) in visited:
                        continue

                    # Attractive potential = distance to goal.
                    att = self.attractive_gain * math.hypot(
                        nr - goal[0], nc - goal[1]
                    )

                    # Repulsive potential from nearby obstacles.
                    rep = 0.0
                    rr = self.repulsive_range
                    r_min = max(0, nr - rr)
                    r_max = min(rows, nr + rr + 1)
                    c_min = max(0, nc - rr)
                    c_max = min(cols, nc + rr + 1)
                    patch = obstacle_mask[r_min:r_max, c_min:c_max]
                    if patch.any():
                        obs_coords = np.argwhere(patch)
                        obs_coords[:, 0] += r_min
                        obs_coords[:, 1] += c_min
                        dists = np.sqrt(
                            (obs_coords[:, 0] - nr) ** 2
                            + (obs_coords[:, 1] - nc) ** 2
                        )
                        dists = np.maximum(dists, 0.5)
                        rep = self.repulsive_gain * np.sum(1.0 / dists)

                    total = att + rep
                    if total < best_potential:
                        best_potential = total
                        best_next = (nr, nc)

            if best_next is None:
                break  # Stuck in local minimum.

            current = best_next
            visited.add(current)
            path.append(current)

        logger.warning("PotentialField: no path found (stuck or max iterations).")
        return self._no_path(cost_map)


# ------------------------------------------------------------------
# RRT* Planner (sampling-based)
# ------------------------------------------------------------------


class RRTStarPlanner(BasePlanner):
    """RRT* sampling-based planner for complex obstacle fields.

    Builds a rapidly-exploring random tree and rewires it for
    near-optimal paths.
    """

    def __init__(
        self,
        max_iterations: int = 5000,
        step_size: int = 5,
        goal_bias: float = 0.1,
        rewire_radius: int = 15,
    ) -> None:
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius

    def plan(
        self,
        cost_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> NavigationResult:
        rows, cols = cost_map.shape

        if not self._in_bounds(start, rows, cols) or not self._in_bounds(goal, rows, cols):
            return self._no_path(cost_map)
        if not np.isfinite(cost_map[start]):
            logger.warning("RRT*: start %s is impassable.", start)
            return self._no_path(cost_map)

        # Tree as dict: node → parent.
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        cost_to: Dict[Tuple[int, int], float] = {start: 0.0}
        nodes = [start]

        for iteration in range(self.max_iterations):
            # Sample random point (with goal bias).
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = (random.randint(0, rows - 1), random.randint(0, cols - 1))

            # Find nearest node.
            nearest = min(nodes, key=lambda n: _dist(n, sample))

            # Steer toward sample.
            new_node = self._steer(nearest, sample, self.step_size)
            nr, nc = new_node

            if not self._in_bounds(new_node, rows, cols):
                continue
            if not np.isfinite(cost_map[nr, nc]):
                continue
            if not self._collision_free(cost_map, nearest, new_node):
                continue

            # Find best parent in neighborhood.
            neighbors = [
                n for n in nodes
                if _dist(n, new_node) <= self.rewire_radius
                and self._collision_free(cost_map, n, new_node)
            ]

            best_parent = nearest
            best_cost = cost_to[nearest] + self._edge_cost(cost_map, nearest, new_node)

            for nb in neighbors:
                c = cost_to[nb] + self._edge_cost(cost_map, nb, new_node)
                if c < best_cost:
                    best_cost = c
                    best_parent = nb

            parent[new_node] = best_parent
            cost_to[new_node] = best_cost
            nodes.append(new_node)

            # Rewire neighbors.
            for nb in neighbors:
                c = best_cost + self._edge_cost(cost_map, new_node, nb)
                if c < cost_to.get(nb, float("inf")):
                    parent[nb] = new_node
                    cost_to[nb] = c

            # Check if we reached the goal.
            if _dist(new_node, goal) <= self.step_size:
                if self._collision_free(cost_map, new_node, goal) and np.isfinite(
                    cost_map[goal]
                ):
                    parent[goal] = new_node
                    cost_to[goal] = best_cost + self._edge_cost(
                        cost_map, new_node, goal
                    )
                    path = self._extract_path(parent, goal)
                    logger.info(
                        "RRT* path found: %d cells, cost %.2f",
                        len(path), cost_to[goal],
                    )
                    return NavigationResult(
                        cost_map=cost_map,
                        path=path,
                        path_cost=cost_to[goal],
                        path_found=True,
                    )

        logger.warning("RRT*: no path found in %d iterations.", self.max_iterations)
        return self._no_path(cost_map)

    @staticmethod
    def _steer(
        from_node: Tuple[int, int],
        to_node: Tuple[int, int],
        step_size: int,
    ) -> Tuple[int, int]:
        d = _dist(from_node, to_node)
        if d <= step_size:
            return to_node
        ratio = step_size / d
        r = int(from_node[0] + ratio * (to_node[0] - from_node[0]))
        c = int(from_node[1] + ratio * (to_node[1] - from_node[1]))
        return (r, c)

    @staticmethod
    def _collision_free(
        cost_map: np.ndarray,
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> bool:
        """Check if the straight line from a to b is obstacle-free."""
        steps = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        if steps == 0:
            return True
        for i in range(steps + 1):
            t = i / steps
            r = int(a[0] + t * (b[0] - a[0]))
            c = int(a[1] + t * (b[1] - a[1]))
            if not np.isfinite(cost_map[r, c]):
                return False
        return True

    @staticmethod
    def _edge_cost(
        cost_map: np.ndarray,
        a: Tuple[int, int],
        b: Tuple[int, int],
    ) -> float:
        d = _dist(a, b)
        avg_cost = (cost_map[a] + cost_map[b]) / 2.0
        return d * avg_cost

    @staticmethod
    def _extract_path(
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [goal]
        node = goal
        while parent.get(node) is not None:
            node = parent[node]
            path.append(node)
        path.reverse()
        return path


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def planner_factory(config: Dict[str, Any]) -> BasePlanner:
    """Create a planner instance from config.

    Config key: ``planner.strategy`` — one of ``astar``, ``potential_field``,
    ``rrt_star``.  Defaults to ``astar``.
    """
    planner_cfg = config.get("planner", {})
    strategy = planner_cfg.get("strategy", "astar")
    allow_diagonal = planner_cfg.get("allow_diagonal", False)

    if strategy == "astar":
        return AStarPlanner(allow_diagonal=allow_diagonal)
    elif strategy == "potential_field":
        params = planner_cfg.get("potential_field", {})
        return PotentialFieldPlanner(
            attractive_gain=params.get("attractive_gain", 1.0),
            repulsive_gain=params.get("repulsive_gain", 100.0),
            repulsive_range=params.get("repulsive_range", 10),
            max_iterations=params.get("max_iterations", 5000),
        )
    elif strategy == "rrt_star":
        params = planner_cfg.get("rrt_star", {})
        return RRTStarPlanner(
            max_iterations=params.get("max_iterations", 5000),
            step_size=params.get("step_size", 5),
            goal_bias=params.get("goal_bias", 0.1),
            rewire_radius=params.get("rewire_radius", 15),
        )
    else:
        raise ValueError(f"Unknown planner strategy: {strategy}")
