"""Tests for the pole/wall shape filter — synthetic point clouds, no I/O.

Run with: python -m unittest discover -s tests
"""

import unittest

import numpy as np

from tree4cfd.config import ShapeFilter
from tree4cfd.shapes import (
    classify_pole_wall,
    compute_tree_stats,
    filter_pole_wall_trees,
)

RNG = np.random.default_rng(0)


def _box(n, cx, cy, dx, dy, z0, z1, label):
    """n random points filling a box, plus their label array."""
    pts = np.column_stack([
        RNG.uniform(cx - dx / 2, cx + dx / 2, n),
        RNG.uniform(cy - dy / 2, cy + dy / 2, n),
        RNG.uniform(z0, z1, n),
    ])
    return pts, np.full(n, label, dtype=np.int32)


# A normal tree (wide crown), a pole (tall + thin), a wall (tall + thin in one
# axis, low aspect), and a low bush (short).
TREE = _box(500, 0, 0, 6.0, 6.0, 0, 8, 1)
POLE = _box(60, 50, 50, 0.5, 0.5, 0, 8, 2)
WALL = _box(200, 100, 0, 0.3, 2.0, 0, 8, 3)
BUSH = _box(200, 0, 100, 4.0, 4.0, 0, 1.5, 4)


def _combine(*clouds):
    pts = np.vstack([c[0] for c in clouds])
    labels = np.concatenate([c[1] for c in clouds])
    return pts, labels


class ComputeStats(unittest.TestCase):
    def test_stats_capture_geometry(self):
        pts, labels = _combine(TREE, POLE)
        stats = compute_tree_stats(pts, labels)
        self.assertEqual(set(stats), {1, 2})
        self.assertGreater(stats[1]["crown_diam"], 4.0)   # wide tree
        self.assertLess(stats[2]["crown_diam"], 1.0)      # thin pole
        self.assertLess(stats[2]["mid_span"], 1.0)
        self.assertEqual(stats[1]["n"], 500)

    def test_ignores_unassigned(self):
        pts, labels = _combine(TREE)
        labels[:50] = 0
        self.assertEqual(set(compute_tree_stats(pts, labels)), {1})


class ClassifyAndFilter(unittest.TestCase):
    def setUp(self):
        self.cfg = ShapeFilter(enabled=True)

    def test_pole_and_wall_classified(self):
        pts, labels = _combine(TREE, POLE, WALL, BUSH)
        poles, walls = classify_pole_wall(compute_tree_stats(pts, labels), self.cfg)
        self.assertIn(2, poles)
        self.assertIn(3, walls)
        # Tree and bush survive both categories.
        self.assertNotIn(1, poles | walls)
        self.assertNotIn(4, poles | walls)

    def test_filter_removes_pole_and_wall_only(self):
        pts, labels = _combine(TREE, POLE, WALL, BUSH)
        out, n_poles, n_walls = filter_pole_wall_trees(labels.copy(), pts, self.cfg)
        self.assertEqual((n_poles, n_walls), (1, 1))
        self.assertEqual(set(np.unique(out[out > 0])), {1, 4})

    def test_point_count_cap_spares_dense_narrow_tree(self):
        # A tall narrow-ish column but with many points: the pole cap (n<200)
        # keeps it, guarding against deleting genuine columnar trees.
        dense = _box(400, 200, 200, 2.0, 2.0, 0, 9, 9)
        out, n_poles, n_walls = filter_pole_wall_trees(
            dense[1].copy(), dense[0], self.cfg
        )
        self.assertEqual((n_poles, n_walls), (0, 0))

    def test_no_trees(self):
        labels = np.zeros(10, dtype=np.int32)
        pts = RNG.random((10, 3))
        out, n_poles, n_walls = filter_pole_wall_trees(labels.copy(), pts, self.cfg)
        self.assertEqual((n_poles, n_walls), (0, 0))


if __name__ == "__main__":
    unittest.main()
