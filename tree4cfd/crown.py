"""Per-tree crown meshing via a Gaussian-smoothed voxel grid + marching cubes."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, label as cc_label
from skimage.measure import marching_cubes

# 26-connectivity so diagonally touching voxels count as one crown.
_CONNECTIVITY = np.ones((3, 3, 3), dtype=int)


def _smoothed_norm_grid(pts, voxel_size, sigma):
    """Occupancy grid → Gaussian-smoothed, max-normalised.

    Returns ``(field, min_b, idx, max_density)`` where ``idx`` is each point's
    voxel index and ``min_b`` the grid origin. Shared by the crown hull (LoD 3)
    and the point-level component mask (LoD 1-2).
    """
    padding = 3.0 * sigma * voxel_size
    min_b = pts.min(axis=0) - padding
    dims = np.ceil((pts.max(axis=0) + padding - min_b) / voxel_size).astype(int)
    idx = np.clip(np.floor((pts - min_b) / voxel_size).astype(int), 0, dims - 1)
    grid = np.zeros(dims, dtype=np.float32)
    np.add.at(grid, tuple(idx.T), 1.0)
    field = gaussian_filter(grid, sigma=sigma)
    mx = field.max()
    if mx > 0:
        field /= mx
    return field, min_b, idx, mx


def _component_keep(field, iso_level, min_frac):
    """Label isosurface components and flag which to keep.

    Returns ``(comp, keep, n)``: the per-voxel component labels, a per-label
    boolean of components ≥ ``min_frac`` of the largest (``keep[0]`` = background
    = ``False``), and the component count.
    """
    comp, n = cc_label(field >= iso_level, structure=_CONNECTIVITY)
    sizes = np.bincount(comp.ravel()) if n else np.array([0])
    sizes[0] = 0  # background
    keep = sizes >= max(1.0, min_frac * sizes.max()) if n else sizes > 0
    keep[0] = False
    return comp, keep, n


def prune_small_components(field: np.ndarray, iso_level: float, min_frac: float) -> np.ndarray:
    """Zero out isosurface components smaller than ``min_frac`` of the largest.

    Stray points within a segment form little high-density islands that marching
    cubes would otherwise emit as floating blobs. Only small *above-iso*
    components are suppressed; the sub-iso background gradient is left untouched
    (marching cubes needs it to position the surface smoothly).
    """
    if min_frac <= 0:
        return field
    comp, keep, n = _component_keep(field, iso_level, min_frac)
    if n <= 1:
        return field
    remove = ~keep
    remove[0] = False  # never touch the background gradient
    out = field.copy()
    out[remove[comp]] = 0.0
    return out


def kept_component_mask(
    pts: np.ndarray, voxel_size: float = 0.75, sigma: float = 1.5,
    iso_level: float = 0.15, min_frac: float = 0.1,
) -> np.ndarray:
    """Boolean mask of points belonging to the crown's *kept* components.

    The point-level counterpart of :func:`prune_small_components`, so coarser
    LoDs can take their extent from the same cleaned crown the marching-cubes
    hull uses (no taller-than-LoD3 crowns). Background points (below the
    isosurface) are excluded. Returns all-True when pruning is disabled or would
    remove everything.
    """
    n = len(pts)
    if n < 10 or min_frac <= 0:
        return np.ones(n, dtype=bool)
    field, _, idx, mx = _smoothed_norm_grid(pts, voxel_size, sigma)
    if mx == 0:
        return np.ones(n, dtype=bool)
    comp, keep, ncomp = _component_keep(field, iso_level, min_frac)
    if ncomp <= 1:
        return np.ones(n, dtype=bool)
    mask = keep[comp[idx[:, 0], idx[:, 1], idx[:, 2]]]
    return mask if mask.any() else np.ones(n, dtype=bool)


def segment_to_marching_cubes(
    pts: np.ndarray,
    voxel_size: float = 0.75,
    sigma: float = 1.5,
    iso_level: float = 0.15,
    min_component_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a watertight crown mesh for one tree's points.

    Rasterises points into an occupancy voxel grid, smooths it with a Gaussian,
    normalises, drops small disconnected blobs (see ``prune_small_components``),
    and extracts an isosurface with marching cubes. Returns empty arrays when
    there are too few points or no surface can be extracted.
    """
    if len(pts) < 10:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    field, min_b, _, mx = _smoothed_norm_grid(pts, voxel_size, sigma)
    if mx == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    field = prune_small_components(field, iso_level, min_component_frac)
    try:
        verts, faces, _, _ = marching_cubes(field, level=min(iso_level, 0.99))
    except ValueError:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    verts = verts * voxel_size + min_b
    return verts, faces
