"""Tests for the OSM building filter — pure geometry, no network.

Run with: python -m unittest discover -s tests
"""

import unittest

import numpy as np
from shapely.geometry import Polygon

from tree4cfd.buildings import (
    _rings_from_overpass,
    filter_trees_in_buildings,
    trees_inside_buildings,
)


# A 10 x 10 m square footprint with its lower-left corner at (0, 0).
SQUARE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


class TreesInsideBuildings(unittest.TestCase):
    def test_centroid_inside_is_flagged(self):
        cents = np.array([[5.0, 5.0]])
        mask = trees_inside_buildings(cents, [SQUARE], buffer=0.0)
        self.assertTrue(mask[0])

    def test_centroid_far_outside_is_kept(self):
        cents = np.array([[50.0, 50.0]])
        mask = trees_inside_buildings(cents, [SQUARE], buffer=0.5)
        self.assertFalse(mask[0])

    def test_buffer_catches_nearby_tree(self):
        # 0.3 m outside the edge: kept with no buffer, removed with 0.5 m buffer.
        cents = np.array([[10.3, 5.0]])
        self.assertFalse(trees_inside_buildings(cents, [SQUARE], buffer=0.0)[0])
        self.assertTrue(trees_inside_buildings(cents, [SQUARE], buffer=0.5)[0])

    def test_buffer_excludes_beyond_distance(self):
        cents = np.array([[10.8, 5.0]])  # 0.8 m outside, beyond a 0.5 m buffer
        self.assertFalse(trees_inside_buildings(cents, [SQUARE], buffer=0.5)[0])

    def test_empty_inputs(self):
        self.assertEqual(
            trees_inside_buildings(np.empty((0, 2)), [SQUARE], 0.5).shape, (0,)
        )
        self.assertFalse(
            trees_inside_buildings(np.array([[5.0, 5.0]]), [], 0.5).any()
        )


class FilterTreesInBuildings(unittest.TestCase):
    def _monkeypatch_polys(self, polys):
        """Replace the network footprint loader so tests stay offline."""
        import tree4cfd.buildings as b
        self._orig = b.load_building_polygons
        b.load_building_polygons = lambda *a, **k: polys
        self.addCleanup(lambda: setattr(b, "load_building_polygons", self._orig))

    def test_removes_only_trees_in_footprint(self):
        # Three labelled trees: tree 1 inside the square, trees 2 and 3 outside.
        # translation shifts local coords into the footprint's CRS.
        translation = (0.0, 0.0, 0.0)
        pts_xy = np.array(
            [[5.0, 5.0], [5.0, 5.0],      # tree 1 -> centroid (5, 5) INSIDE
             [50.0, 50.0], [50.0, 50.0],  # tree 2 -> centroid (50, 50) OUTSIDE
             [80.0, 20.0], [80.0, 20.0]]  # tree 3 -> centroid (80, 20) OUTSIDE
        )
        labels = np.array([1, 1, 2, 2, 3, 3], dtype=np.int32)
        self._monkeypatch_polys([SQUARE])

        out, n_removed = filter_trees_in_buildings(
            labels.copy(), pts_xy, translation, epsg=25831, buffer=0.5
        )
        self.assertEqual(n_removed, 1)
        self.assertEqual(set(np.unique(out[out > 0])), {2, 3})
        self.assertTrue((out[labels == 1] == 0).all())

    def test_translation_offset_applied(self):
        # Local centroid (5,5); translation puts it at (105,105) — outside SQUARE.
        translation = (100.0, 100.0, 0.0)
        pts_xy = np.array([[5.0, 5.0], [5.0, 5.0]])
        labels = np.array([1, 1], dtype=np.int32)
        self._monkeypatch_polys([SQUARE])

        out, n_removed = filter_trees_in_buildings(
            labels.copy(), pts_xy, translation, epsg=25831, buffer=0.5
        )
        self.assertEqual(n_removed, 0)

    def test_no_crs_skips(self):
        labels = np.array([1, 1], dtype=np.int32)
        pts_xy = np.array([[5.0, 5.0], [5.0, 5.0]])
        out, n_removed = filter_trees_in_buildings(
            labels.copy(), pts_xy, (0, 0, 0), epsg=None, buffer=0.5
        )
        self.assertEqual(n_removed, 0)


class OverpassParsing(unittest.TestCase):
    def test_parses_ways_and_relation_outer_members(self):
        data = {
            "elements": [
                {"type": "way", "geometry": [
                    {"lon": 2.1, "lat": 41.4}, {"lon": 2.2, "lat": 41.4},
                    {"lon": 2.2, "lat": 41.5}, {"lon": 2.1, "lat": 41.4},
                ]},
                {"type": "relation", "members": [
                    {"type": "way", "role": "outer", "geometry": [
                        {"lon": 2.3, "lat": 41.4}, {"lon": 2.4, "lat": 41.4},
                        {"lon": 2.4, "lat": 41.5},
                    ]},
                    {"type": "way", "role": "inner", "geometry": [
                        {"lon": 2.31, "lat": 41.41},
                    ]},
                ]},
                {"type": "node", "lon": 2.0, "lat": 41.0},  # ignored
            ]
        }
        rings = _rings_from_overpass(data)
        self.assertEqual(len(rings), 2)  # one way + one outer member; inner skipped
        self.assertEqual(rings[0][0], (2.1, 41.4))


if __name__ == "__main__":
    unittest.main()
