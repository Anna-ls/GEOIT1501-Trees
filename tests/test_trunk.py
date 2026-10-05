"""Tests for trunk parameter estimation — inventory DBH override.

Run with: python -m unittest discover -s tests
"""

import unittest

import numpy as np

from tree4cfd.trunk import estimate_trunk_params

RNG = np.random.default_rng(0)


def _tree_and_dtm():
    # A crown of points 0–10 m tall around (5, 5); flat ground at z = 0.
    pts = np.column_stack([
        RNG.uniform(3, 7, 200), RNG.uniform(3, 7, 200), RNG.uniform(0, 10, 200),
    ])
    dtm = np.zeros((10, 10))
    return pts, dtm


class TrunkDbhOverride(unittest.TestCase):
    def test_override_sets_radius(self):
        pts, dtm = _tree_and_dtm()
        tp = estimate_trunk_params(pts, dtm, 0, 0, 1.0, dbh_override_m=1.0)
        self.assertIsNotNone(tp)
        self.assertAlmostEqual(tp["dbh"], 1.0)
        self.assertAlmostEqual(tp["r_base"], 0.5)

    def test_allometry_used_without_override(self):
        pts, dtm = _tree_and_dtm()
        allom = estimate_trunk_params(pts, dtm, 0, 0, 1.0,
                                      allom_a=0.03, allom_b=1.2)
        forced = estimate_trunk_params(pts, dtm, 0, 0, 1.0,
                                       allom_a=0.03, allom_b=1.2, dbh_override_m=1.0)
        self.assertNotAlmostEqual(allom["dbh"], 1.0)   # allometric value differs
        self.assertAlmostEqual(forced["dbh"], 1.0)

    def test_zero_override_falls_back(self):
        pts, dtm = _tree_and_dtm()
        a = estimate_trunk_params(pts, dtm, 0, 0, 1.0, dbh_override_m=0.0)
        b = estimate_trunk_params(pts, dtm, 0, 0, 1.0)
        self.assertAlmostEqual(a["dbh"], b["dbh"])     # 0 → allometry


class TrunkForceXY(unittest.TestCase):
    def test_force_xy_pins_base(self):
        pts, dtm = _tree_and_dtm()       # crown around (5, 5)
        tp = estimate_trunk_params(pts, dtm, 0, 0, 1.0, force_xy=(2.0, 3.0))
        self.assertIsNotNone(tp)
        self.assertAlmostEqual(tp["x"], 2.0)
        self.assertAlmostEqual(tp["y"], 3.0)
        # Crown-derived height is still used (z from the points, not the xy).
        self.assertGreater(tp["height_total"], 8.0)


def _floating_blob():
    # A crown 20–30 m above flat ground (dtm=0): total height ~30 m, long trunk.
    pts = np.column_stack([
        RNG.uniform(3, 7, 200), RNG.uniform(3, 7, 200), RNG.uniform(20, 30, 200),
    ])
    return pts, np.zeros((10, 10))


class HeightSanityLimits(unittest.TestCase):
    def test_too_tall_total_height_rejected(self):
        pts, dtm = _floating_blob()
        self.assertIsNotNone(estimate_trunk_params(pts, dtm, 0, 0, 1.0))  # no limit
        self.assertIsNone(
            estimate_trunk_params(pts, dtm, 0, 0, 1.0, max_total_height=25.0))

    def test_too_long_trunk_rejected(self):
        pts, dtm = _floating_blob()  # crown base ~21 m above ground
        self.assertIsNone(
            estimate_trunk_params(pts, dtm, 0, 0, 1.0, max_trunk_height=12.0))

    def test_normal_tree_passes_limits(self):
        pts, dtm = _tree_and_dtm()   # 0–10 m tall, on the ground
        self.assertIsNotNone(estimate_trunk_params(
            pts, dtm, 0, 0, 1.0, max_total_height=30.0, max_trunk_height=12.0))


if __name__ == "__main__":
    unittest.main()
