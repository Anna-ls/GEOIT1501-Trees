"""LAS/LAZ reading, point filtering, and OBJ writing."""

from __future__ import annotations

from pathlib import Path
from typing import List

import laspy
import numpy as np

from .config import Filtering


def find_tiles(tiles_dir: Path) -> List[Path]:
    """Return all ``.laz`` then ``.las`` tiles in ``tiles_dir``, sorted."""
    tiles_dir = Path(tiles_dir)
    return sorted(tiles_dir.glob("*.laz")) + sorted(tiles_dir.glob("*.las"))


def read_las(path: Path):
    """Read a LAS/LAZ file fully into memory and return the laspy object."""
    with laspy.open(path) as f:
        return f.read()


def read_header_bounds(path: Path):
    """Return ``((x_min, y_min, x_max, y_max), epsg)`` from the LAS header only.

    Cheap (no point reading) — used to compute the tiles' extent for inventory
    extraction. ``epsg`` is ``None`` if the file has no CRS.
    """
    with laspy.open(path) as f:
        h = f.header
        crs = h.parse_crs()
        return (h.x_min, h.y_min, h.x_max, h.y_max), (crs.to_epsg() if crs else None)


def filter_vegetation(las, cfg: Filtering) -> np.ndarray:
    """Select high-vegetation points as an ``(N, 3)`` xyz array.

    Uses the configured ASPRS class codes when given, otherwise falls back to
    an NDVI threshold. Optionally keeps only multi-return points (pulses that
    penetrate the canopy).
    """
    if cfg.high_veg_classes:
        mask = np.isin(np.asarray(las.classification, dtype=np.uint8), cfg.high_veg_classes)
    else:
        try:
            nir = np.asarray(las.nir, dtype=np.float64)
            red = np.asarray(las.red, dtype=np.float64)
            ndvi = (nir - red) / (nir + red + 1e-8)
            mask = ndvi > cfg.ndvi_threshold
        except AttributeError:
            print(
                "No high_veg_classes configured and LAS file has no NIR/Red bands for NDVI"
            )

    if cfg.multi_return_only:
        mask &= np.asarray(las.number_of_returns) > 1

    return np.column_stack((las.x[mask], las.y[mask], las.z[mask]))


def filter_ground(las, cfg: Filtering) -> np.ndarray:
    """Select ground points (``cfg.ground_classes``, default class 2) as ``(N, 3)``."""
    mask = np.isin(np.asarray(las.classification, dtype=np.uint8), cfg.ground_classes)
    return np.column_stack((las.x[mask], las.y[mask], las.z[mask]))


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a Wavefront ``.obj`` mesh (1-based face indices)."""
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def write_obj_grouped(path: Path, verts: np.ndarray, groups) -> None:
    """Write a Wavefront ``.obj`` with named ``g`` groups sharing one vertex list.

    ``groups`` is a list of ``(name, faces)`` where ``faces`` indexes ``verts``.
    Lets downstream tools (and the CFD mesher) separate e.g. crown vs trunk.
    """
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for name, faces in groups:
            if len(faces) == 0:
                continue
            f.write(f"g {name}\n")
            for tri in faces:
                f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def write_offset(path: Path, translation) -> None:
    """Record the translation so output coordinates can be mapped back to UTM."""
    with open(path, "w") as f:
        f.write(f"x_offset = {translation[0]}\n")
        f.write(f"y_offset = {translation[1]}\n")
        f.write(f"z_offset = {translation[2]}\n")
        f.write("# Subtract this from .obj coordinates to recover original UTM\n")
