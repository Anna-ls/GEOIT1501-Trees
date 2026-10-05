"""Step 1 — profile classification codes across tiles.

Confirms which class code holds high vegetation before running the pipeline.
High-vegetation points typically show a high multi-return rate (> 80 %),
NDVI > 0.2, and a wide Z range.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .io import read_las

# Catalan LiDAR classification scheme (ICGC), a superset of standard ASPRS.
ASPRS = {
    1: "Default (no classificats)",
    2: "Ground (terreny)",
    3: "Low vegetation (vegetació baixa)",
    4: "Medium vegetation (vegetació mitjana)",
    5: "High vegetation (vegetació alta)",
    6: "Building (edificis)",
    7: "Low points (punts baixos)",
    8: "Model key points (punts clau)",
    9: "Water (aigua)",
    12: "Overlapping (solapament)",
    17: "Bridge (ponts)",
    18: "Air points (punts aeris)",
    75: "Other ground (possible terreny)",
    76: "Under building (sota edificis)",
    77: "Other building (damunt edificis)",
}


def profile_classes(tiles: List[Path]) -> dict:
    """Aggregate per-class statistics across all ``tiles``.

    Returns a mapping ``class_code -> {count, z_min, z_max, ndvi_sum,
    multi_return}``.
    """
    class_stats: dict = {}
    for tile in tiles:
        las = read_las(tile)

        has_nir = hasattr(las, "nir")
        z = np.asarray(las.z, dtype=np.float32)
        ret = np.asarray(las.number_of_returns, dtype=np.uint8)
        cls = np.asarray(las.classification, dtype=np.uint8)

        if has_nir:
            nir = np.asarray(las.nir, dtype=np.float64)
            red = np.asarray(las.red, dtype=np.float64)
            ndvi = (nir - red) / (nir + red + 1e-8)

        for code in np.unique(cls):
            mask = cls == code
            s = class_stats.setdefault(
                int(code),
                {"count": 0, "z_min": np.inf, "z_max": -np.inf,
                 "ndvi_sum": 0.0, "multi_return": 0},
            )
            s["count"] += int(mask.sum())
            s["z_min"] = min(s["z_min"], float(z[mask].min()))
            s["z_max"] = max(s["z_max"], float(z[mask].max()))
            s["multi_return"] += int((ret[mask] > 1).sum())
            if has_nir:
                s["ndvi_sum"] += float(ndvi[mask].sum())

    return class_stats


def print_class_table(class_stats: dict) -> None:
    """Print the per-class summary table to stdout."""
    print(
        f"\n{'Code':>6}  {'Name':<38}  {'Points':>12}  "
        f"{'Z range':>16}  {'NDVI mean':>10}  {'Multi-return':>14}"
    )
    print("-" * 110)
    for code, s in sorted(class_stats.items()):
        n = s["count"]
        ndvi_mean = s["ndvi_sum"] / n if n else 0
        multi_pct = s["multi_return"] / n * 100 if n else 0
        name = ASPRS.get(code, "(unknown — not in Catalan scheme)")
        print(
            f"{code:>6}  {name:<38}  {n:>12,}  "
            f"[{s['z_min']:5.1f}, {s['z_max']:6.1f}]  "
            f"{ndvi_mean:>10.3f}  {multi_pct:>13.1f}%"
        )
