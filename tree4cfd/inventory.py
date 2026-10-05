"""Tree-inventory extraction and matching — two decoupled stages.

1. **Extract** (``tree4cfd inventory``): pull a city's open tree inventory for
   the area covered by the tiles and write a *universal* inventory CSV — one row
   per tree:

       id, lon, lat, common_name, species, height_m, dbh_cm, age_years, category

   ``lon``/``lat`` (WGS84) are the only required columns, so any city source — or
   a user's own private inventory — can produce a compatible file.

2. **Match** (during ``tree4cfd run``): load that CSV, reproject to each tile's
   CRS, and match every segmented tree to the nearest inventory entry within
   ``match_dist`` metres, writing a per-tile ``<tile>_trees.csv`` join and
   feeding matched DBH to the trunk model.

Built-in extraction sources: ``paris_ods`` (OpenDataSoft) and ``barcelona_ckan``
(CKAN downloadable CSVs). Responses are cached on disk.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point
from shapely.strtree import STRtree

from .config import Inventory
from .io import find_tiles, read_header_bounds

_HEADERS = {"User-Agent": "tree4cfd/0.1 (LiDAR tree meshing; inventory lookup)"}

# Universal inventory CSV schema. lon/lat are required; the rest are optional.
UNIVERSAL_FIELDS = ["id", "lon", "lat", "common_name", "species",
                    "height_m", "dbh_cm", "age_years", "category"]
_NUMERIC = {"lon", "lat", "height_m", "dbh_cm", "age_years"}


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _age_from_date(s: Optional[str]) -> Optional[int]:
    """Years since a planting date string containing a 4-digit year."""
    if not s:
        return None
    m = re.search(r"(?:18|19|20)\d{2}", s)
    if not m:
        return None
    age = date.today().year - int(m.group())
    return age if 0 <= age <= 300 else None


# ----------------------------------------------------------------------
# Stage 1 — extraction to the universal CSV
# ----------------------------------------------------------------------

def _paris_record(row: dict) -> dict:
    g = row.get("geo_point_2d") or {}
    circ = row.get("circonferenceencm")
    genre, espece = row.get("genre"), row.get("espece")
    return {
        "id": row.get("idbase"), "lon": g.get("lon"), "lat": g.get("lat"),
        "common_name": row.get("libellefrancais"),
        "species": " ".join(p for p in (genre, espece) if p) or None,
        "height_m": row.get("hauteurenm"),
        "dbh_cm": round(circ / math.pi, 1) if circ else None,
        "age_years": None, "category": row.get("domanialite"),
    }


def fetch_paris(bbox_ll, cfg: Inventory) -> List[dict]:
    """Records (lon/lat) from the Paris OpenDataSoft API for a lat/lon bbox."""
    s, w, n, e = bbox_ll
    cache = Path(cfg.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    key = "paris_" + hashlib.md5(f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}".encode()).hexdigest()
    cache_file = cache / f"{key}.json"
    if cache_file.exists():
        rows = json.loads(cache_file.read_text())
    else:
        url = f"{cfg.ods_base}/catalog/datasets/{cfg.ods_dataset}/exports/json"
        params = {"where": f"in_bbox(geo_point_2d,{s},{w},{n},{e})",
                  "select": "idbase,libellefrancais,genre,espece,hauteurenm,"
                            "circonferenceencm,domanialite,geo_point_2d"}
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=300)
        resp.raise_for_status()
        rows = resp.json()
        cache_file.write_text(json.dumps(rows))
    return [r for r in (_paris_record(x) for x in rows) if r["lon"] is not None]


def _barcelona_record(row: dict) -> dict:
    return {
        "id": row.get("codi"),
        "lon": _num(row.get("longitud")), "lat": _num(row.get("latitud")),
        "common_name": row.get("cat_nom_castella"),
        "species": row.get("cat_nom_cientific"),
        "height_m": None, "dbh_cm": None,
        "age_years": _age_from_date(row.get("data_plantacio")),
        "category": row.get("categoria_arbrat"),
    }


def _download_ckan_csv(cfg: Inventory, dataset: str) -> Path:
    cache = Path(cfg.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    csv_file = cache / f"{dataset}.csv"
    if csv_file.exists():
        return csv_file
    meta = requests.get(f"{cfg.ckan_base}/data/api/3/action/package_show",
                        params={"id": dataset}, headers=_HEADERS, timeout=120).json()
    url = next((r["url"] for r in meta["result"]["resources"]
                if r.get("format") == "CSV" and "zip" not in (r.get("name") or "").lower()),
               None)
    if not url:
        raise RuntimeError(f"No CSV resource for CKAN dataset {dataset!r}")
    resp = requests.get(url, headers=_HEADERS, timeout=600)
    resp.raise_for_status()
    csv_file.write_bytes(resp.content)
    return csv_file


def fetch_barcelona(bbox_ll, cfg: Inventory) -> List[dict]:
    """Records (lon/lat) from the Barcelona CKAN CSVs, filtered to a lat/lon bbox."""
    s, w, n, e = bbox_ll
    out: List[dict] = []
    for dataset in cfg.ckan_datasets:
        with open(_download_ckan_csv(cfg, dataset), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rec = _barcelona_record(row)
                if rec["lon"] is None or rec["lat"] is None:
                    continue
                if w <= rec["lon"] <= e and s <= rec["lat"] <= n:
                    out.append(rec)
    return out


def _fetch_records(bbox_ll, cfg: Inventory) -> List[dict]:
    if cfg.source == "paris_ods":
        return fetch_paris(bbox_ll, cfg)
    if cfg.source == "barcelona_ckan":
        return fetch_barcelona(bbox_ll, cfg)
    raise ValueError(f"Unknown inventory source: {cfg.source!r}")


def _tiles_bbox_lonlat(tiles) -> Optional[Tuple[float, float, float, float]]:
    """Union lat/lon bbox (s, w, n, e) over the tiles' headers."""
    s = w = n = e = None
    for t in tiles:
        (xmn, ymn, xmx, ymx), epsg = read_header_bounds(t)
        if epsg is None:
            continue
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
            [xmn, xmx, xmn, xmx], [ymn, ymn, ymx, ymx])
        smn, wmn, nmx, emx = min(lat), min(lon), max(lat), max(lon)
        s = smn if s is None else min(s, smn)
        w = wmn if w is None else min(w, wmn)
        n = nmx if n is None else max(n, nmx)
        e = emx if e is None else max(e, emx)
    return None if s is None else (s, w, n, e)


def write_universal_csv(path, records: Sequence[dict]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIVERSAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in UNIVERSAL_FIELDS})


def _roi_bbox_lonlat(poi, radius, epsg):
    """Lat/lon bbox (s, w, n, e) for the POI ± radius square in CRS ``epsg``."""
    t = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = t.transform(
        [poi[0] - radius, poi[0] + radius, poi[0] - radius, poi[0] + radius],
        [poi[1] - radius, poi[1] - radius, poi[1] + radius, poi[1] + radius])
    return min(lat), min(lon), max(lat), max(lon)


def extract_inventory(cfg) -> Tuple[Path, int]:
    """Fetch the inventory and write the universal CSV.

    Covers just the influence region (POI ± radius) when one is set, otherwise
    the full extent of the tiles.
    """
    tiles = find_tiles(cfg.paths.tiles_dir)
    if not tiles:
        raise FileNotFoundError(f"No tiles in {cfg.paths.tiles_dir}")
    if not cfg.inventory.csv_path:
        raise ValueError("inventory.csv_path is not set")

    if cfg.influence_region > 0 and cfg.point_of_interest is not None:
        _, epsg = read_header_bounds(tiles[0])
        bbox = _roi_bbox_lonlat(cfg.point_of_interest, cfg.influence_region, epsg) \
            if epsg else _tiles_bbox_lonlat(tiles)
    else:
        bbox = _tiles_bbox_lonlat(tiles)
    if bbox is None:
        raise RuntimeError("Tiles have no CRS — cannot determine area for inventory")
    records = _fetch_records(bbox, cfg.inventory)
    write_universal_csv(cfg.inventory.csv_path, records)
    return Path(cfg.inventory.csv_path), len(records)


# ----------------------------------------------------------------------
# Stage 2 — load the universal CSV and match per tile
# ----------------------------------------------------------------------

def load_inventory_csv(path) -> Tuple[np.ndarray, List[dict]]:
    """Load a universal inventory CSV → ``(lonlat (M, 2), records)``.

    Tolerant of missing optional columns; rows without a valid lon/lat are
    dropped. Numeric fields are parsed to floats (or ``None``).
    """
    lon, lat, recs = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            x, y = _num(row.get("lon")), _num(row.get("lat"))
            if x is None or y is None:
                continue
            lon.append(x); lat.append(y)
            recs.append({k: (_num(row.get(k)) if k in _NUMERIC else (row.get(k) or None))
                         for k in UNIVERSAL_FIELDS})
    xy = np.column_stack([lon, lat]) if lon else np.empty((0, 2))
    return xy, recs


def inventory_local_xy(inv_lonlat: np.ndarray, epsg: Optional[int],
                       translation: Sequence[float]) -> np.ndarray:
    """Reproject inventory lon/lat to the tile CRS and shift to local coords.

    Returns an ``(M, 2)`` array aligned with the translated vegetation points
    (empty if no CRS or no inventory).
    """
    if epsg is None or len(inv_lonlat) == 0:
        return np.empty((0, 2))
    X, Y = Transformer.from_crs(4326, epsg, always_xy=True).transform(
        inv_lonlat[:, 0], inv_lonlat[:, 1])
    return np.column_stack([X, Y]) - np.asarray(translation[:2], dtype=float)


def build_inventory_index(inv_local: np.ndarray):
    """STRtree over inventory points (local coords), or ``None`` if empty."""
    if inv_local is None or len(inv_local) == 0:
        return None
    return STRtree([Point(float(x), float(y)) for x, y in inv_local])


def crown_inventory_indices(
    crown_xy: np.ndarray, inv_local: np.ndarray, strtree, buffer: float
) -> List[int]:
    """Indices of inventory points that genuinely sit under a crown.

    The convex hull (+ buffer) is only a fast pre-filter; each candidate is then
    gated by its distance to the nearest *actual* crown point being ≤ ``buffer``,
    so points in hull concavities or just beyond the canopy don't get a trunk
    with nothing above it.
    """
    if strtree is None or len(crown_xy) == 0:
        return []
    crown_xy = np.asarray(crown_xy)
    hull = MultiPoint(crown_xy).convex_hull
    cand = [int(i) for i in strtree.query(
        hull.buffer(buffer) if buffer > 0 else hull, predicate="intersects")]
    if not cand:
        return []
    d, _ = cKDTree(crown_xy).query(inv_local[cand])
    return [cand[k] for k in range(len(cand)) if d[k] <= buffer]


def crown_footprint(crown_xy: np.ndarray):
    """Convex-hull footprint of a crown's points (for snapping trunks under it)."""
    return MultiPoint(np.asarray(crown_xy)).convex_hull


def snap_under_crown(xy, footprint, inset: float = 0.3) -> Tuple[float, float]:
    """Return ``xy`` unchanged if under the crown footprint, else snap just inside.

    Keeps the inventory stem position when it already sits under the canopy; when
    it's at/beyond the edge, projects it onto the footprint boundary and nudges
    ``inset`` metres toward the centroid so the trunk connects to the crown mesh.
    """
    x, y = float(xy[0]), float(xy[1])
    pt = Point(x, y)
    if footprint.geom_type != "Polygon" or footprint.contains(pt):
        return x, y
    proj = footprint.exterior.interpolate(footprint.exterior.project(pt))
    c = footprint.centroid
    dx, dy = c.x - proj.x, c.y - proj.y
    n = math.hypot(dx, dy)
    if n == 0:
        return proj.x, proj.y
    f = min(inset, n) / n
    return proj.x + dx * f, proj.y + dy * f


def match_to_inventory(centroids, inv_xy, max_dist) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest inventory entry per tree within ``max_dist`` (same CRS as inputs)."""
    n = len(centroids)
    idx = np.full(n, -1, dtype=int)
    dist = np.full(n, np.inf)
    if n == 0 or len(inv_xy) == 0:
        return idx, dist
    d, i = cKDTree(inv_xy).query(centroids, distance_upper_bound=max_dist)
    hit = np.isfinite(d)
    idx[hit] = i[hit]; dist[hit] = d[hit]
    return idx, dist


_MATCHED_HEADER = ["tree_id", "x", "y", "matched", "match_dist_m", "inv_id",
                   "inv_common_name", "inv_species", "inv_height_m", "inv_dbh_cm",
                   "inv_age_years", "inv_category"]


def annotate_and_write(
    csv_path, labels: np.ndarray, pts_xy: np.ndarray, translation: Sequence[float],
    epsg: Optional[int], inventory: Tuple[np.ndarray, List[dict]], match_dist: float,
) -> Tuple[int, int, dict]:
    """Match surviving trees to the loaded inventory; write the per-tile CSV.

    Returns ``(n_matched, n_trees, dbh_by_label)``; ``dbh_by_label`` maps tree
    label → DBH in metres for matched trees that carry a diameter.
    """
    inv_lonlat, inv_recs = inventory
    uniq = np.unique(labels[labels > 0])
    if len(uniq) == 0:
        return 0, 0, {}

    tx, ty = float(translation[0]), float(translation[1])
    cnt = np.bincount(labels)
    cx = np.bincount(labels, weights=pts_xy[:, 0])[uniq] / cnt[uniq] + tx
    cy = np.bincount(labels, weights=pts_xy[:, 1])[uniq] / cnt[uniq] + ty
    centroids = np.column_stack([cx, cy])

    if epsg is None or len(inv_lonlat) == 0:
        inv_xy = np.empty((0, 2))
    else:
        X, Y = Transformer.from_crs(4326, epsg, always_xy=True).transform(
            inv_lonlat[:, 0], inv_lonlat[:, 1])
        inv_xy = np.column_stack([X, Y])
    idx, dist = match_to_inventory(centroids, inv_xy, match_dist)

    def blank(v):
        return "" if v is None else v

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_MATCHED_HEADER)
        for k, lbl in enumerate(uniq):
            base = [int(lbl), round(float(centroids[k, 0]), 3), round(float(centroids[k, 1]), 3)]
            if idx[k] >= 0:
                r = inv_recs[idx[k]]
                writer.writerow(base + [1, round(float(dist[k]), 2), blank(r.get("id")),
                    blank(r.get("common_name")), blank(r.get("species")),
                    blank(r.get("height_m")), blank(r.get("dbh_cm")),
                    blank(r.get("age_years")), blank(r.get("category"))])
            else:
                writer.writerow(base + [0, "", "", "", "", "", "", "", ""])

    n_matched = int((idx >= 0).sum())
    dbh_by_label = {int(uniq[k]): inv_recs[idx[k]]["dbh_cm"] / 100.0
                    for k in range(len(uniq))
                    if idx[k] >= 0 and inv_recs[idx[k]].get("dbh_cm")}
    return n_matched, len(uniq), dbh_by_label
