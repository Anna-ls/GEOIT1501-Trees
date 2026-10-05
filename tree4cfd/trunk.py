"""Trunk geometry: DTM sampling, allometric parameter estimation, cone mesh."""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample_dtm(dtm, x_min, y_min, cell_size, x, y):
    """Bilinear sample of the DTM raster at world coordinates ``(x, y)``.

    ``x`` and ``y`` may be scalars or arrays. Coordinates are clamped to the
    raster bounds. Returns the ground elevation.
    """
    ny, nx = dtm.shape
    fx = (np.asarray(x) - x_min) / cell_size
    fy = (np.asarray(y) - y_min) / cell_size
    fx = np.clip(fx, 0, nx - 1.001)
    fy = np.clip(fy, 0, ny - 1.001)
    x0 = np.floor(fx).astype(int); x1 = x0 + 1
    y0 = np.floor(fy).astype(int); y1 = y0 + 1
    wx = fx - x0; wy = fy - y0
    return (
        (1 - wx) * (1 - wy) * dtm[y0, x0]
        + wx * (1 - wy) * dtm[y0, x1]
        + (1 - wx) * wy * dtm[y1, x0]
        + wx * wy * dtm[y1, x1]
    )


def estimate_trunk_params(
    tree_pts: np.ndarray,
    dtm,
    x_min,
    y_min,
    cell_size,
    q_base: float = 0.20,
    q_cb: float = 0.10,
    allom_a: float = 0.012,
    allom_b: float = 1.2,
    taper_ratio: float = 0.4,
    dbh_override_m: Optional[float] = None,
    max_trunk_height: float = 0.0,
    max_total_height: float = 0.0,
    force_xy: Optional[tuple] = None,
) -> Optional[dict]:
    """Estimate trunk geometry for one segmented crown.

    Locates the trunk base in XY from the lowest ``q_base`` fraction of crown
    points, looks up ground elevation from the DTM, then fits a tapered cone
    from ground to crown base. The diameter is ``dbh_override_m`` (e.g. a matched
    inventory DBH, in metres) when given, otherwise an allometric DBH model.

    Returns a geometry dict, or ``None`` if the geometry is implausible —
    including trees taller than ``max_total_height`` above ground or with a trunk
    (ground to crown base) longer than ``max_trunk_height``, which flag floating
    roof-level blobs and DTM errors (both ``0`` to disable).

    ``force_xy`` pins the trunk base to a given ``(x, y)`` (e.g. a matched
    inventory tree position) instead of deriving it from the lowest crown points.
    """
    if len(tree_pts) < 20:
        return None

    z = tree_pts[:, 2]
    if force_xy is not None:
        x_t, y_t = float(force_xy[0]), float(force_xy[1])
    else:
        z_thresh = np.quantile(z, q_base)
        low_mask = z <= z_thresh
        if low_mask.sum() < 3:
            return None
        x_t = float(tree_pts[low_mask, 0].mean())
        y_t = float(tree_pts[low_mask, 1].mean())

    z_base = float(sample_dtm(dtm, x_min, y_min, cell_size, x_t, y_t))
    z_crown_base = float(np.quantile(z, q_cb))
    # Minimum trunk height of 1.5 m above ground (handles tall narrow crowns
    # like cypresses where the 10th-percentile point is near ground level).
    z_crown_base = max(z_crown_base, z_base + 1.5)
    z_top = float(z.max())

    h_trunk = z_crown_base - z_base
    h_total = z_top - z_base
    if h_trunk <= 0.3 or h_total <= 1.0:
        return None
    if max_trunk_height > 0 and h_trunk > max_trunk_height:
        return None
    if max_total_height > 0 and h_total > max_total_height:
        return None

    if dbh_override_m and dbh_override_m > 0:
        dbh = dbh_override_m
    else:
        dbh = allom_a * (h_total ** allom_b)
    r_base = dbh / 2.0
    r_top = taper_ratio * r_base

    return {
        "x": x_t, "y": y_t,
        "z_base": z_base, "z_crown_base": z_crown_base, "z_top": z_top,
        "height_total": h_total, "height_trunk": h_trunk,
        "dbh": dbh, "r_base": r_base, "r_top": r_top,
    }


def make_trunk_mesh(
    x, y, z_base, z_top, r_base, r_top, n_sides: int = 12, close_caps: bool = True
):
    """Closed truncated cone aligned with +Z.

    Returns ``verts (V, 3)`` and ``faces (F, 3)`` with outward-facing normals;
    watertight when ``close_caps`` is True.
    """
    theta = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    ring_bot = np.column_stack(
        [x + r_base * cos_t, y + r_base * sin_t, np.full(n_sides, z_base)]
    )
    ring_top = np.column_stack(
        [x + r_top * cos_t, y + r_top * sin_t, np.full(n_sides, z_top)]
    )
    verts = [ring_bot, ring_top]

    faces = []
    for k in range(n_sides):
        j = (k + 1) % n_sides
        faces.append([k, j, j + n_sides])
        faces.append([k, j + n_sides, k + n_sides])

    if close_caps:
        c_bot = np.array([[x, y, z_base]])
        c_top = np.array([[x, y, z_top]])
        c_bot_idx = 2 * n_sides
        c_top_idx = 2 * n_sides + 1
        verts.extend([c_bot, c_top])
        for k in range(n_sides):
            j = (k + 1) % n_sides
            faces.append([c_bot_idx, j, k])               # bottom cap (-Z normal)
            faces.append([c_top_idx, k + n_sides, j + n_sides])  # top cap (+Z normal)

    return np.vstack(verts), np.array(faces, dtype=np.int32)
