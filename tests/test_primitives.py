"""Tests for tree primitives and species→shape mapping. No I/O.

Run with: python -m unittest discover -s tests
"""

import unittest

import numpy as np

from tree4cfd.primitives import (
    block_mesh,
    cone_mesh,
    crown_primitive,
    crown_shape_for_species,
    ellipsoid_mesh,
)


def _valid_mesh(v, f):
    return len(v) > 0 and len(f) > 0 and int(f.max()) < len(v) and int(f.min()) >= 0


class SpeciesShape(unittest.TestCase):
    def test_conifer_to_cone(self):
        self.assertEqual(crown_shape_for_species("Pinus pinea"), "cone")
        self.assertEqual(crown_shape_for_species("Cupressus sempervirens"), "cone")

    def test_palm_to_cylinder(self):
        self.assertEqual(crown_shape_for_species("Phoenix dactylifera"), "cylinder")
        self.assertEqual(crown_shape_for_species(None, "Palmera de Canàries"), "cylinder")

    def test_broadleaf_and_unknown_to_ellipsoid(self):
        self.assertEqual(crown_shape_for_species("Platanus x hispanica"), "ellipsoid")
        self.assertEqual(crown_shape_for_species(None), "ellipsoid")


class Meshes(unittest.TestCase):
    def test_cone_valid_and_apex_height(self):
        v, f = cone_mesh(0, 0, 5, 12, r=3, n_sides=12)
        self.assertTrue(_valid_mesh(v, f))
        self.assertAlmostEqual(v[:, 2].max(), 12.0)   # apex
        self.assertAlmostEqual(v[:, 2].min(), 5.0)    # base

    def test_ellipsoid_valid_and_bounds(self):
        v, f = ellipsoid_mesh(1, 2, 10, a=2, b=3, c=4)
        self.assertTrue(_valid_mesh(v, f))
        self.assertAlmostEqual(v[:, 2].max(), 14.0, places=5)  # cz + c
        self.assertAlmostEqual(v[:, 2].min(), 6.0, places=5)   # cz - c
        self.assertLessEqual(np.abs(v[:, 0] - 1).max(), 2.0 + 1e-6)  # within a

    def test_crown_primitive_dispatch(self):
        for shape in ("ellipsoid", "cone", "cylinder"):
            v, f = crown_primitive(shape, 0, 0, 5, 12, dx=4, dy=4)
            self.assertTrue(_valid_mesh(v, f), shape)

    def test_block_spans_full_height(self):
        v, f = block_mesh(0, 0, 0, 15, dx=5, dy=5)
        self.assertTrue(_valid_mesh(v, f))
        self.assertAlmostEqual(v[:, 2].min(), 0.0)
        self.assertAlmostEqual(v[:, 2].max(), 15.0)


if __name__ == "__main__":
    unittest.main()
