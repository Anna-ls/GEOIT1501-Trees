"""Shape-based filtering of segmented trees: drop pole- and wall-like blobs.

CHM segmentation occasionally turns lamp posts, sign poles, narrow facade
slivers, or other thin vertical structures into "trees". These are tall but
horizontally narrow, unlike a real crown which spreads out. This module flags
them from per-tree geometry — overall extent plus the median XY span through the
middle height slices — and zeroes their labels before meshing.

Adapted from the SegmentAnyTree post-processing pole/wall filter
(github.com/Jackson513ye/sat_pp), reworked for tree4cfd's per-point label model.
Stats use coordinate *differences* only, so local (translated) coords are fine.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .config import ShapeFilter


def compute_tree_stats(
    pts_xyz: np.ndarray, labels: np.ndarray
) -> Dict[int, dict]:
    """Per-tree geometric statistics, keyed by label (only labels > 0).

    Each entry holds point count, vertical range, horizontal extents, the mean
    crown diameter, aspect (crown_diam / z_range), and the median XY span across
    the middle three of five equal height slices (``mid_span`` over the mean of
    the two horizontal extents, ``mid_min_span`` over their minimum).
    """
    stats: Dict[int, dict] = {}
    for lbl in np.unique(labels[labels > 0]):
        m = labels == lbl
        ix, iy, iz = pts_xyz[m, 0], pts_xyz[m, 1], pts_xyz[m, 2]
        n = int(m.sum())
        z_min, z_max = float(iz.min()), float(iz.max())
        z_range = z_max - z_min
        dx = float(ix.max() - ix.min())
        dy = float(iy.max() - iy.min())
        crown_diam = (dx + dy) / 2.0

        # Median span across the middle 3 of 5 height slices.
        if z_range > 1.0 and n >= 10:
            edges = np.linspace(z_min, z_max, 6)
            spans, min_spans = [], []
            for si in range(5):
                s = (iz >= edges[si]) & (iz < edges[si + 1])
                if s.sum() > 1:
                    sdx = float(ix[s].max() - ix[s].min())
                    sdy = float(iy[s].max() - iy[s].min())
                    spans.append((sdx + sdy) / 2.0)
                    min_spans.append(min(sdx, sdy))
                else:
                    spans.append(0.0)
                    min_spans.append(0.0)
            mid_span = float(np.median(spans[1:4]))
            mid_min_span = float(np.median(min_spans[1:4]))
        else:
            mid_span = crown_diam
            mid_min_span = min(dx, dy)

        stats[int(lbl)] = {
            "n": n,
            "z_range": z_range,
            "dx": dx,
            "dy": dy,
            "crown_diam": crown_diam,
            "min_xy": min(dx, dy),
            "aspect": crown_diam / z_range if z_range > 0 else 999.0,
            "mid_span": mid_span,
            "mid_min_span": mid_min_span,
        }
    return stats


def classify_pole_wall(
    stats: Dict[int, dict], cfg: ShapeFilter
) -> Tuple[set, set]:
    """Split tree labels into pole-like and wall-like sets given the thresholds."""
    poles, walls = set(), set()
    for lbl, s in stats.items():
        if (
            s["z_range"] > cfg.pole_z_range_min
            and s["crown_diam"] < cfg.pole_crown_diam_max
            and s["mid_span"] < cfg.pole_mid_span_max
            and s["n"] < cfg.pole_max_points
        ):
            poles.add(lbl)
        elif (
            s["z_range"] > cfg.wall_z_range_min
            and s["min_xy"] < cfg.wall_min_xy_max
            and s["mid_min_span"] < cfg.wall_mid_min_span_max
            and s["aspect"] < cfg.wall_aspect_max
        ):
            walls.add(lbl)
    return poles, walls


def filter_pole_wall_trees(
    labels: np.ndarray, pts_xyz: np.ndarray, cfg: ShapeFilter
) -> Tuple[np.ndarray, int, int]:
    """Zero out labels of pole-like and wall-like segments.

    ``pts_xyz`` is ``(N, 3)`` vegetation points; ``labels`` the matching per-point
    tree ids. Returns updated ``labels`` and the pole / wall removal counts.
    """
    stats = compute_tree_stats(pts_xyz, labels)
    if not stats:
        return labels, 0, 0
    poles, walls = classify_pole_wall(stats, cfg)
    removed = poles | walls
    if removed:
        labels[np.isin(labels, list(removed))] = 0
    return labels, len(poles), len(walls)
