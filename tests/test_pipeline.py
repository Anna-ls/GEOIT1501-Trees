"""Tests for tile mesh assembly — inventory-seeded blob splitting. No I/O.

Run with: python -m unittest discover -s tests
"""

import unittest

import numpy as np

import tempfile
from pathlib import Path

from tree4cfd.config import Config
from tree4cfd.pipeline import (
    TileMeshes,
    _build_tile_meshes,
    _connected_trunk_top,
    _filter_region,
    _merge_tiles,
    _write_meshes,
    parse_lod,
)

RNG = np.random.default_rng(0)


def _cluster(cx, cy, n=200, half=1.5, z0=0, z1=8):
    return np.column_stack([
        RNG.uniform(cx - half, cx + half, n),
        RNG.uniform(cy - half, cy + half, n),
        RNG.uniform(z0, z1, n),
    ])


class BlobSplitting(unittest.TestCase):
    def setUp(self):
        # One labelled "blob" = two clusters 12 m apart; flat ground.
        self.pts = np.vstack([_cluster(5, 5), _cluster(17, 5)])
        self.labels = np.ones(len(self.pts), dtype=np.int32)
        self.dtm = np.zeros((25, 25))
        self.georef = (0.0, 0.0, 1.0)
        self.cfg = Config()

    def _count(self, inv_local, inv_recs):
        tm = _build_tile_meshes(
            self.pts, self.labels, self.dtm, self.georef, self.cfg,
            inv_local, inv_recs)
        return tm.n_crowns, tm.n_trunks

    def test_two_seeds_split_one_blob_into_two(self):
        inv = np.array([[5.0, 5.0], [17.0, 5.0]])
        recs = [{"dbh_cm": None}, {"dbh_cm": None}]
        n_crowns, n_trunks = self._count(inv, recs)
        self.assertEqual(n_crowns, 2)        # blob split into two instances
        self.assertEqual(n_trunks, 2)        # one trunk each

    def test_no_inventory_keeps_single_estimated_tree(self):
        n_crowns, n_trunks = self._count(None, None)
        self.assertEqual((n_crowns, n_trunks), (1, 1))   # one whole crown + estimated trunk

    def test_one_seed_no_split(self):
        inv = np.array([[5.0, 5.0]])
        n_crowns, n_trunks = self._count(inv, [{"dbh_cm": None}])
        self.assertEqual((n_crowns, n_trunks), (1, 1))


class LevelOfDetail(unittest.TestCase):
    def setUp(self):
        self.pts = np.vstack([_cluster(5, 5), _cluster(17, 5)])
        self.labels = np.ones(len(self.pts), dtype=np.int32)
        self.dtm = np.zeros((25, 25))
        self.georef = (0.0, 0.0, 1.0)
        self.inv = np.array([[5.0, 5.0], [17.0, 5.0]])
        self.recs = [{"dbh_cm": None, "species": "Pinus pinea"},
                     {"dbh_cm": None, "species": "Platanus x hispanica"}]

    def _faces(self, lod):
        cfg = Config()
        cfg.lod = lod
        tm = _build_tile_meshes(
            self.pts, self.labels, self.dtm, self.georef, cfg, self.inv, self.recs)
        return len(tm.crown_f) + len(tm.trunk_f), tm.n_crowns, tm.n_trunks

    def test_parse_lod(self):
        self.assertEqual(parse_lod(1), (1, False, "1"))
        self.assertEqual(parse_lod(1.1), (1, True, "1.1"))
        self.assertEqual(parse_lod(3), (3, False, "3"))
        self.assertEqual(parse_lod(3.1), (3, True, "3.1"))

    def test_integer_lods_have_no_trunk(self):
        for lod in (1, 2, 3):
            _, n_crowns, n_trunks = self._faces(lod)
            self.assertEqual(n_crowns, 2, lod)   # split into two crowns
            self.assertEqual(n_trunks, 0, lod)   # crown only

    def test_point1_lods_add_trunks(self):
        for lod in (1.1, 2.1, 3.1):
            _, n_crowns, n_trunks = self._faces(lod)
            self.assertEqual((n_crowns, n_trunks), (2, 2), lod)  # crown + trunk each

    def test_face_counts_increase_with_detail(self):
        self.assertLess(self._faces(1)[0], self._faces(2)[0])    # block < primitive
        self.assertLess(self._faces(2)[0], self._faces(3)[0])    # primitive < hull
        self.assertLess(self._faces(2)[0], self._faces(2.1)[0])  # trunk adds faces


class RegionFilter(unittest.TestCase):
    def test_keeps_only_trees_within_radius(self):
        # Local centroids; translation maps to projected coords; POI in projected.
        translation = (1000.0, 2000.0, 0.0)
        pts = np.array([[5.0, 5.0], [5.0, 5.0],        # tree 1 -> (1005, 2005)
                        [405.0, 5.0], [405.0, 5.0]])   # tree 2 -> (1405, 2005)
        labels = np.array([1, 1, 2, 2], dtype=np.int32)
        poi = (1005.0, 2005.0, 0.0)
        out, n_kept = _filter_region(labels.copy(), pts, translation, poi, radius=100.0)
        self.assertEqual(n_kept, 1)
        self.assertEqual(set(np.unique(out[out > 0])), {1})   # tree 2 (400 m away) dropped

    def test_all_within_radius(self):
        translation = (0.0, 0.0, 0.0)
        pts = np.array([[1.0, 1.0], [2.0, 2.0]])
        labels = np.array([1, 2], dtype=np.int32)
        _, n_kept = _filter_region(labels.copy(), pts, translation, (0.0, 0.0, 0.0), 100.0)
        self.assertEqual(n_kept, 2)


def _tm(seed):
    rng = np.random.default_rng(seed)
    return TileMeshes(rng.random((4, 3)), np.array([[0, 1, 2]]),
                      rng.random((3, 3)), np.array([[0, 1, 2]]), 1, 1)


class MergeAndWrite(unittest.TestCase):
    def test_merge_offsets_and_counts(self):
        m = _merge_tiles([_tm(0), _tm(1)])
        self.assertEqual((m.n_crowns, m.n_trunks), (2, 2))
        self.assertEqual(len(m.crown_v), 8)
        self.assertEqual(len(m.trunk_v), 6)
        np.testing.assert_array_equal(m.crown_f[1], [4, 5, 6])   # 2nd tile offset by 4
        np.testing.assert_array_equal(m.trunk_f[1], [3, 4, 5])   # 2nd tile offset by 3

    def test_write_separate_files(self):
        with tempfile.TemporaryDirectory() as d:
            _write_meshes(Path(d) / "x_lod2.1", _tm(0), separate=True)
            names = sorted(p.name for p in Path(d).glob("*.obj"))
            self.assertEqual(names, ["x_lod2.1_crown.obj", "x_lod2.1_trunk.obj"])

    def test_write_grouped_file(self):
        with tempfile.TemporaryDirectory() as d:
            _write_meshes(Path(d) / "x_lod2.1", _tm(0), separate=False)
            p = Path(d) / "x_lod2.1.obj"
            self.assertTrue(p.exists())
            text = p.read_text()
            self.assertIn("g crown", text)
            self.assertIn("g trunk", text)


class ConnectedTrunkTop(unittest.TestCase):
    def test_raises_top_to_reach_crown(self):
        # Crown mesh bottom (10) above the 10th-percentile (6) → trunk extended.
        self.assertAlmostEqual(_connected_trunk_top(6.0, 10.0, 0.5), 10.5)

    def test_keeps_top_when_already_overlapping(self):
        # Crown extends below the 10th-percentile → keep the higher trunk top.
        self.assertAlmostEqual(_connected_trunk_top(6.0, 4.0, 0.5), 6.0)


if __name__ == "__main__":
    unittest.main()
