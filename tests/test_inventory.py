"""Tests for inventory: universal CSV IO, record parsing, and matching. No network.

Run with: python -m unittest discover -s tests
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyproj import Transformer

from tree4cfd.config import Inventory
from tree4cfd.inventory import (
    UNIVERSAL_FIELDS,
    _barcelona_record,
    _paris_record,
    annotate_and_write,
    build_inventory_index,
    crown_footprint,
    crown_inventory_indices,
    inventory_local_xy,
    load_inventory_csv,
    match_to_inventory,
    snap_under_crown,
    write_universal_csv,
)


class RecordParsing(unittest.TestCase):
    def test_paris_record_has_lonlat_and_dbh(self):
        r = _paris_record({
            "idbase": 42, "libellefrancais": "Platane",
            "genre": "Platanus", "espece": "x acerifolia",
            "hauteurenm": 12, "circonferenceencm": 157, "domanialite": "Alignement",
            "geo_point_2d": {"lon": 2.35, "lat": 48.85},
        })
        self.assertEqual((r["lon"], r["lat"]), (2.35, 48.85))
        self.assertEqual(r["species"], "Platanus x acerifolia")
        self.assertAlmostEqual(r["dbh_cm"], 50.0, places=1)
        self.assertEqual(r["id"], 42)

    def test_barcelona_record_lonlat_and_age(self):
        from datetime import date
        r = _barcelona_record({
            "codi": "X1", "longitud": "2.16", "latitud": "41.43",
            "cat_nom_castella": "Pino", "cat_nom_cientific": "Pinus pinea",
            "categoria_arbrat": "EXEMPLAR", "data_plantacio": "2008-01-01",
        })
        self.assertEqual((r["lon"], r["lat"]), (2.16, 41.43))
        self.assertEqual(r["age_years"], date.today().year - 2008)
        self.assertIsNone(r["dbh_cm"])


class UniversalCsvIO(unittest.TestCase):
    def test_roundtrip_and_numeric_parsing(self):
        recs = [
            {"id": "1", "lon": 2.35, "lat": 48.85, "common_name": "Platane",
             "species": "Platanus", "height_m": 12, "dbh_cm": 50.0,
             "age_years": None, "category": "Alignement"},
            {"id": "2", "lon": 2.36, "lat": 48.86, "common_name": None,
             "species": None, "height_m": None, "dbh_cm": None,
             "age_years": 7, "category": None},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "inv.csv"
            write_universal_csv(p, recs)
            with p.open() as f:
                self.assertEqual(next(csv.reader(f)), UNIVERSAL_FIELDS)  # header
            xy, loaded = load_inventory_csv(p)

        self.assertEqual(xy.shape, (2, 2))
        self.assertAlmostEqual(xy[0, 0], 2.35)
        self.assertEqual(loaded[0]["dbh_cm"], 50.0)        # parsed to float
        self.assertIsNone(loaded[1]["dbh_cm"])             # blank -> None
        self.assertEqual(loaded[1]["age_years"], 7.0)

    def test_minimal_private_csv(self):
        # A user's private file with only lon/lat/species is accepted.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "private.csv"
            p.write_text("lon,lat,species\n2.35,48.85,Quercus\n")
            xy, recs = load_inventory_csv(p)
        self.assertEqual(xy.shape, (1, 2))
        self.assertEqual(recs[0]["species"], "Quercus")
        self.assertIsNone(recs[0]["dbh_cm"])

    def test_rows_without_coords_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("lon,lat\n2.0,41.0\n,41.0\nbad,1\n")
            xy, recs = load_inventory_csv(p)
        self.assertEqual(len(recs), 1)


class Matching(unittest.TestCase):
    def test_nearest_within_radius(self):
        cents = np.array([[0.0, 0.0], [100.0, 100.0]])
        invxy = np.array([[0.5, 0.0]])
        idx, dist = match_to_inventory(cents, invxy, 3.0)
        self.assertEqual(idx[0], 0)
        self.assertAlmostEqual(dist[0], 0.5)
        self.assertEqual(idx[1], -1)

    def test_annotate_reprojects_and_writes(self):
        # Inventory tree at a known lon/lat; put a segmented tree at its EPSG:2154
        # location so it must match after reprojection.
        epsg, lon, lat = 2154, 2.35, 48.85
        X, Y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        translation = (X - 5.0, Y - 5.0, 0.0)        # local origin near the tree
        pts_xy = np.array([[5.0, 5.0], [5.0, 5.0]])  # centroid -> (X, Y)
        labels = np.array([1, 1], dtype=np.int32)
        inv = (np.array([[lon, lat]]),
               [{"id": "7", "common_name": "Platane", "species": "Platanus",
                 "height_m": 12.0, "dbh_cm": 50.0, "age_years": None,
                 "category": "Alignement"}])

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t_trees.csv"
            n_matched, n_trees, dbh = annotate_and_write(
                out, labels, pts_xy, translation, epsg, inv, match_dist=3.0)
            with out.open() as f:
                rows = list(csv.DictReader(f))

        self.assertEqual((n_matched, n_trees), (1, 1))
        self.assertAlmostEqual(dbh[1], 0.50, places=3)     # DBH metres for trunk
        self.assertEqual(rows[0]["inv_species"], "Platanus")
        self.assertEqual(rows[0]["matched"], "1")

    def test_annotate_no_crs(self):
        inv = (np.array([[2.35, 48.85]]), [{"dbh_cm": 50.0}])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.csv"
            n_matched, n_trees, dbh = annotate_and_write(
                out, np.array([1, 1], dtype=np.int32), np.zeros((2, 2)),
                (0, 0, 0), None, inv, 3.0)
        self.assertEqual((n_matched, dbh), (0, {}))


class SpatialHelpers(unittest.TestCase):
    def test_inventory_local_xy_reprojects_and_shifts(self):
        epsg, lon, lat = 2154, 2.35, 48.85
        X, Y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
        out = inventory_local_xy(np.array([[lon, lat]]), epsg, (X - 10.0, Y - 20.0, 0.0))
        self.assertAlmostEqual(out[0, 0], 10.0, places=3)
        self.assertAlmostEqual(out[0, 1], 20.0, places=3)

    def test_inventory_local_xy_no_crs(self):
        out = inventory_local_xy(np.array([[2.0, 41.0]]), None, (0, 0, 0))
        self.assertEqual(out.shape, (0, 2))

    def test_crown_connection_gates_on_nearest_point(self):
        # Dense filled crown so interior points exist (the gate needs real points).
        gx, gy = np.meshgrid(np.arange(0, 10.1, 1.0), np.arange(0, 10.1, 1.0))
        crown = np.column_stack([gx.ravel(), gy.ravel()])
        inv = np.array([[5.0, 5.0],     # on a crown point
                        [50.0, 50.0],   # far away
                        [10.6, 5.0]])   # 0.6 m beyond the canopy edge
        st = build_inventory_index(inv)
        self.assertEqual(crown_inventory_indices(crown, inv, st, 0.0), [0])
        self.assertEqual(sorted(crown_inventory_indices(crown, inv, st, 1.0)), [0, 2])

    def test_crown_connection_concavity_excluded(self):
        # Two clusters with a gap; an inventory point in the (hull-filled) gap
        # must NOT connect — nothing is actually above it.
        left = np.random.default_rng(0).uniform([0, 0], [3, 10], (200, 2))
        right = np.random.default_rng(1).uniform([17, 0], [20, 10], (200, 2))
        crown = np.vstack([left, right])
        inv = np.array([[10.0, 5.0]])  # mid-gap: inside convex hull, far from points
        st = build_inventory_index(inv)
        self.assertEqual(crown_inventory_indices(crown, inv, st, 2.0), [])

    def test_crown_connection_no_inventory(self):
        self.assertEqual(crown_inventory_indices(np.zeros((4, 2)), None, None, 1.0), [])

    def test_snap_keeps_inside_point(self):
        crown = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        fp = crown_footprint(crown)
        self.assertEqual(snap_under_crown((5.0, 5.0), fp), (5.0, 5.0))  # already inside

    def test_snap_pulls_outside_point_in(self):
        crown = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
        fp = crown_footprint(crown)
        sx, sy = snap_under_crown((12.0, 5.0), fp, inset=0.3)  # 2 m to the right
        self.assertAlmostEqual(sx, 10.0 - 0.3, places=5)       # pulled inside the edge
        self.assertAlmostEqual(sy, 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
