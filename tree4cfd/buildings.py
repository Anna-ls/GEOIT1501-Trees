"""Drop trees that sit inside OpenStreetMap building footprints.

CHM segmentation occasionally promotes rooftop greenery, facade planting, or
misclassified roof structure into a "tree". Such artefacts have their horizontal
centroid on or inside a building footprint, so we remove them before meshing.

Footprints come from the OpenStreetMap Overpass API, queried per tile bounding
box and cached on disk. All geometry tests run in the tile's projected CRS
(metres), so ``buffer`` is a true metric distance.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Public mirrors tried in turn when the primary endpoint is overloaded.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
# Overpass rejects the default python-requests User-Agent (HTTP 406).
_HEADERS = {"User-Agent": "tree4cfd/0.1 (LiDAR tree meshing; OSM building filter)"}


def _overpass_query(query: str, overpass_url: str, max_retries: int = 5) -> dict:
    """POST a query to Overpass with retries and mirror rotation.

    Transient failures (timeouts, 429/5xx) are common on the public servers, so
    each attempt rotates to a different endpoint and backs off exponentially.
    Raises the last error if every attempt fails.
    """
    endpoints = [overpass_url] + [m for m in OVERPASS_MIRRORS if m != overpass_url]
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        url = endpoints[attempt % len(endpoints)]
        try:
            resp = requests.post(
                url, data={"data": query}, headers=_HEADERS, timeout=300
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as err:
            last_err = err
            if attempt == max_retries - 1:
                break
            wait = min(60, 5 * 2 ** attempt)
            print(f"    buildings: Overpass {url} failed ({err}); retry in {wait}s")
            time.sleep(wait)
    raise last_err


def _rings_from_overpass(data: dict) -> List[List[Tuple[float, float]]]:
    """Extract building outer rings as lists of ``(lon, lat)`` from Overpass JSON.

    Handles ``way`` elements (with inline ``geometry`` from ``out geom``) and the
    outer members of ``relation`` (multipolygon) elements. Inner holes are
    ignored — a centroid in a courtyard is a rare, acceptable miss.
    """
    rings: List[List[Tuple[float, float]]] = []
    for el in data.get("elements", []):
        if el.get("type") == "way" and "geometry" in el:
            rings.append([(p["lon"], p["lat"]) for p in el["geometry"]])
        elif el.get("type") == "relation":
            for m in el.get("members", []):
                if m.get("type") == "way" and m.get("role") == "outer" and "geometry" in m:
                    rings.append([(p["lon"], p["lat"]) for p in m["geometry"]])
    return rings


def fetch_footprints_lonlat(
    south: float, west: float, north: float, east: float,
    overpass_url: str = OVERPASS_URL,
    cache_dir: str | Path = "DATA/osm_cache",
) -> List[List[Tuple[float, float]]]:
    """Return building outer rings (lon/lat) for a lat/lon bbox, cached on disk.

    The raw Overpass response is cached keyed by the rounded bbox so reruns are
    offline and reproducible.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(
        f"{south:.5f},{west:.5f},{north:.5f},{east:.5f}".encode()
    ).hexdigest()
    cache_file = cache / f"osm_{key}.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        query = (
            "[out:json][timeout:180];"
            f'(way["building"]({south},{west},{north},{east});'
            f'relation["building"]({south},{west},{north},{east}););'
            "out geom;"
        )
        data = _overpass_query(query, overpass_url)
        cache_file.write_text(json.dumps(data))

    return _rings_from_overpass(data)


def _bbox_to_lonlat(
    x_min: float, y_min: float, x_max: float, y_max: float, epsg: int
) -> Tuple[float, float, float, float]:
    """Project a metric bbox to an Overpass ``(south, west, north, east)`` bbox."""
    t = Transformer.from_crs(epsg, 4326, always_xy=True)
    lon, lat = t.transform([x_min, x_max, x_min, x_max], [y_min, y_min, y_max, y_max])
    return min(lat), min(lon), max(lat), max(lon)


def load_building_polygons(
    bounds: Tuple[float, float, float, float], epsg: int,
    overpass_url: str = OVERPASS_URL, cache_dir: str | Path = "DATA/osm_cache",
) -> List[Polygon]:
    """Footprints overlapping ``bounds`` (projected CRS), reprojected to that CRS.

    ``bounds`` is ``(x_min, y_min, x_max, y_max)`` in the tile's projected CRS.
    """
    south, west, north, east = _bbox_to_lonlat(*bounds, epsg)
    rings = fetch_footprints_lonlat(south, west, north, east, overpass_url, cache_dir)
    if not rings:
        return []

    t = Transformer.from_crs(4326, epsg, always_xy=True)
    polys: List[Polygon] = []
    for ring in rings:
        if len(ring) < 3:
            continue
        lon = [p[0] for p in ring]
        lat = [p[1] for p in ring]
        xs, ys = t.transform(lon, lat)
        poly = Polygon(zip(xs, ys))
        if poly.is_valid:
            polys.append(poly)
        else:  # self-touching rings happen; zero-width buffer repairs them
            repaired = poly.buffer(0)
            if not repaired.is_empty:
                polys.append(repaired)
    return polys


def trees_inside_buildings(
    centroids_xy: np.ndarray, polygons: Sequence[Polygon], buffer: float = 0.5
) -> np.ndarray:
    """Boolean mask: ``True`` where a centroid lies within ``buffer`` m of a footprint.

    Pure geometry — no network. ``centroids_xy`` is ``(K, 2)`` in the same CRS as
    ``polygons``.
    """
    n = len(centroids_xy)
    mask = np.zeros(n, dtype=bool)
    if n == 0 or not polygons:
        return mask

    geoms = [p.buffer(buffer) for p in polygons] if buffer else list(polygons)
    tree = STRtree(geoms)
    for i, xy in enumerate(centroids_xy):
        if len(tree.query(Point(xy), predicate="intersects")) > 0:
            mask[i] = True
    return mask


def filter_trees_in_buildings(
    labels: np.ndarray, pts_xy: np.ndarray, translation: Sequence[float],
    epsg: Optional[int], buffer: float = 0.5,
    overpass_url: str = OVERPASS_URL, cache_dir: str | Path = "DATA/osm_cache",
) -> Tuple[np.ndarray, int]:
    """Zero out labels of trees whose centroid sits inside a buffered OSM footprint.

    ``pts_xy`` is ``(N, 2)`` vegetation point XY in **local** coordinates (with
    ``translation`` already subtracted). Returns the updated ``labels`` and the
    number of trees removed.
    """
    if epsg is None:
        print("    buildings: tile has no CRS — skipping footprint filter")
        return labels, 0

    unique = np.unique(labels[labels > 0])
    if len(unique) == 0:
        return labels, 0

    tx, ty = float(translation[0]), float(translation[1])
    # Per-tree centroid in projected coords (O(N) via bincount).
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    sum_x = np.bincount(labels, weights=pts_xy[:, 0], minlength=counts.size)
    sum_y = np.bincount(labels, weights=pts_xy[:, 1], minlength=counts.size)
    centroids = np.column_stack(
        [sum_x[unique] / counts[unique] + tx, sum_y[unique] / counts[unique] + ty]
    )

    bounds = (
        pts_xy[:, 0].min() + tx, pts_xy[:, 1].min() + ty,
        pts_xy[:, 0].max() + tx, pts_xy[:, 1].max() + ty,
    )
    polys = load_building_polygons(bounds, epsg, overpass_url, cache_dir)
    if not polys:
        return labels, 0

    inside = trees_inside_buildings(centroids, polys, buffer)
    removed = unique[inside]
    if len(removed):
        labels[np.isin(labels, removed)] = 0
    return labels, int(len(removed))
