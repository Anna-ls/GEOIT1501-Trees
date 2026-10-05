"""Tests for crown meshing — small-component pruning. No I/O.

Run with: python -m unittest discover -s tests
"""

import unittest

import numpy as np

from tree4cfd.crown import (
    kept_component_mask,
    prune_small_components,
    segment_to_marching_cubes,
)

RNG = np.random.default_rng(0)


class PruneSmallComponents(unittest.TestCase):
    def _field(self):
        # Normalised density field: one large blob + one tiny far-away blob,
        # both above the iso level (0.15).
        f = np.zeros((30, 30, 30), dtype=np.float32)
        f[5:15, 5:15, 5:15] = 1.0     # large component (1000 voxels)
        f[25:27, 25:27, 25:27] = 0.5  # tiny component (8 voxels)
        return f

    def test_drops_tiny_component(self):
        out = prune_small_components(self._field(), iso_level=0.15, min_frac=0.1)
        self.assertGreater(out[5:15, 5:15, 5:15].max(), 0)  # large kept
        self.assertEqual(out[25:27, 25:27, 25:27].max(), 0)  # tiny gone

    def test_disabled_keeps_everything(self):
        out = prune_small_components(self._field(), iso_level=0.15, min_frac=0.0)
        self.assertGreater(out[25:27, 25:27, 25:27].max(), 0)

    def test_single_component_untouched(self):
        f = np.zeros((20, 20, 20), dtype=np.float32)
        f[5:15, 5:15, 5:15] = 1.0
        out = prune_small_components(f, iso_level=0.15, min_frac=0.1)
        np.testing.assert_array_equal(out, f)

    def test_preserves_subiso_background(self):
        # A smooth sub-iso halo around the large blob must survive pruning —
        # zeroing it would roughen the meshed surface.
        f = self._field()
        f[4, 5:15, 5:15] = 0.10  # below iso (0.15), part of the background
        out = prune_small_components(f, iso_level=0.15, min_frac=0.1)
        self.assertAlmostEqual(float(out[4, 10, 10]), 0.10, places=5)
        self.assertEqual(out[25:27, 25:27, 25:27].max(), 0)  # tiny still gone


def _fill(cx, cy, cz, half, step=0.4):
    """Dense uniform lattice of points filling a cube — equal local density."""
    g = np.arange(-half, half + 1e-9, step)
    X, Y, Z = np.meshgrid(g, g, g)
    return np.column_stack([X.ravel() + cx, Y.ravel() + cy, Z.ravel() + cz])


class SegmentToMarchingCubes(unittest.TestCase):
    def test_stray_cluster_excluded_from_mesh(self):
        # A big crown (10 m cube) + a small same-density stray blob (2 m cube)
        # 30 m away. Both exceed the iso level; only the stray is small enough
        # to be pruned.
        pts = np.vstack([_fill(0, 0, 5, 5.0), _fill(30, 30, 5, 1.0)])

        v_on, _ = segment_to_marching_cubes(pts, min_component_frac=0.1)
        v_off, _ = segment_to_marching_cubes(pts, min_component_frac=0.0)

        self.assertGreater(len(v_on), 0)
        near_stray_on = (np.linalg.norm(v_on - [30, 30, 5], axis=1) < 5).sum()
        near_stray_off = (np.linalg.norm(v_off - [30, 30, 5], axis=1) < 5).sum()
        self.assertEqual(near_stray_on, 0)        # pruned away
        self.assertGreater(near_stray_off, 0)     # present without pruning

    def test_main_crown_unchanged_by_pruning(self):
        # The kept crown must be identical with/without pruning — pruning only
        # removes the stray, it must not alter (roughen) the main surface.
        pts = np.vstack([_fill(0, 0, 5, 5.0), _fill(30, 30, 5, 1.0)])
        v_on, _ = segment_to_marching_cubes(pts, min_component_frac=0.1)
        v_off, _ = segment_to_marching_cubes(pts, min_component_frac=0.0)
        # Main crown vertices (near origin) come first; they must match exactly.
        self.assertGreater(len(v_on), 0)
        self.assertTrue(np.allclose(v_on, v_off[: len(v_on)]))


class KeptComponentMask(unittest.TestCase):
    def test_drops_stray_cluster_points(self):
        # Dense crown + a small stray cluster 30 m away; the stray points should
        # be excluded so a coarse-LoD extent matches the pruned hull.
        crown = _fill(0, 0, 5, 5.0)
        stray = _fill(30, 30, 5, 1.0)
        pts = np.vstack([crown, stray])
        mask = kept_component_mask(pts, 0.5, 1.5, 0.15, min_frac=0.1)
        # No kept point should be near the stray cluster.
        near_stray = np.linalg.norm(pts[mask] - [30, 30, 5], axis=1) < 5
        self.assertFalse(near_stray.any())
        self.assertTrue(mask[: len(crown)].mean() > 0.5)   # main crown retained

    def test_disabled_keeps_all(self):
        pts = np.vstack([_fill(0, 0, 5, 5.0), _fill(30, 30, 5, 1.0)])
        self.assertTrue(kept_component_mask(pts, 0.5, 1.5, 0.15, min_frac=0.0).all())


if __name__ == "__main__":
    unittest.main()
