"""Canopy Height Model (CHM) based individual-tree segmentation.

Builds a DTM (from ground points) and a DSM (from vegetation points) on a
common grid, differences them into a CHM, finds tree tops as local maxima,
and grows crowns with a watershed.
"""

from __future__ import annotations

from typing import Tuple, List, Dict

import numpy as np
import logging
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.cluster import DBSCAN

def analyse_cluster(cluster_pts: np.ndarray) -> Dict[str, float]:
    """
    Performs PCA on the 2D points of a cluster.

    Returns a dictionary containing elongation and offset-ratio metrics
    """
    if len(cluster_pts) < 2:
        return {
            "elongation": 999.0,
            "offset_ratio": 0.0,
            "std_minor": 0.0,
            "std_major": 0.0,
            "z_range": 0.0 if len(cluster_pts) == 0 else float(np.ptp(cluster_pts[:, 2])),
        }

    xy = cluster_pts[:, :2]
    z = cluster_pts[:, 2]
    centroid = np.mean(xy, axis=0)

    # PCA Eigenvalue decomposition
    cov = np.cov(xy, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)

    std_minor = np.sqrt(max(evals[0], 0))
    std_major = np.sqrt(max(evals[1], 0))

    elongation = std_major / std_minor if std_minor > 0 else 999.0

    max_z_idx = np.argmax(z)
    highest_xy = xy[max_z_idx]
    dist_to_centroid = np.linalg.norm(highest_xy - centroid)

    offset_ratio = dist_to_centroid / std_major if std_major > 0 else 0.0

    return {
        "elongation": elongation,
        "offset_ratio": offset_ratio,
        "std_minor": std_minor,
        "std_major": std_major,
        "z_range": np.ptp(z),
    }

def find_secondary_peak(
        cluster_pts: np.ndarray,
        num_angles: int = 4,
        bin_size: float = 0.5,
        smooth_sigma: float = 2.0,
        peak_min_dist_m: float = 3.0,
        d_euclidean_thresh: float = 2.5,
        d_margin_thresh: float = 1.5
) -> np.ndarray:
    xy = cluster_pts[:, :2]
    z = cluster_pts[:, 2]

    dominant_idx = np.argmax(z)
    dominant_peak = cluster_pts[dominant_idx]

    hull = ConvexHull(xy)
    hull_eqs = hull.equations

    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    candidate_peaks = []
    min_dist_bins = max(1, int(peak_min_dist_m / bin_size))

    for angle in angles:
        # Project XY points onto the 2D vector defined by the angle
        p = xy[:, 0] * np.cos(angle) + xy[:, 1] * np.sin(angle)

        # Shift distances to start at 0 for bins
        p_shifted = p - p.min()
        bin_indices = (p_shifted / bin_size).astype(int)

        num_bins = bin_indices.max() + 1
        profile = np.zeros(num_bins)
        max_pt_idx_per_bin = np.full(num_bins, -1, dtype=int)

        # Extract the highest Z value for each bin
        for i in range(len(p)):
            b = bin_indices[i]
            if z[i] > profile[b]:
                profile[b] = z[i]
                max_pt_idx_per_bin[b] = i

        # Smooth
        smooth_profile = gaussian_filter(profile, sigma=smooth_sigma)

        # Find peak in profile
        peaks_idx, _ = find_peaks(smooth_profile, distance=min_dist_bins)

        for idx in peaks_idx:
            pt_idx = max_pt_idx_per_bin[idx]
            if pt_idx != -1:
                candidate_peaks.append(cluster_pts[pt_idx])

    if not candidate_peaks:
        return np.array([dominant_peak])

    candidate_peaks = np.array(candidate_peaks)

    # Clustering merges duplicate detection across different angles
    cluster_labels = DBSCAN(eps=peak_min_dist_m * 0.75, min_samples=1).fit_predict(candidate_peaks[:, :2])

    # Filter treetops using Euclidean distance to dominant peak and margin to the edge
    final_peaks = []
    for label in np.unique(cluster_labels):
        if label == -1:
            continue
        cluster_cands = candidate_peaks[cluster_labels == label]
        best_peak = cluster_cands[np.argmax(cluster_cands[:, 2])]

        dist_to_dominant = np.linalg.norm(best_peak[:2] - dominant_peak[:2])
        if dist_to_dominant < 1e-3:
            final_peaks.append(best_peak)
            continue

        d_euclidean = dist_to_dominant
        d_margin = float('inf')
        if hull_eqs is not None:
            dist_to_edges = np.abs(np.dot(hull_eqs[:,:2], best_peak[:2]) + hull_eqs[:,2])
            d_margin = np.min(dist_to_edges)

        if d_euclidean >= d_euclidean_thresh and d_margin >= d_margin_thresh:
            final_peaks.append(best_peak)

    return np.array(final_peaks)

def find_peaks_variable_window(
        chm_smooth: np.ndarray,
        window_bins: list,
        cell_size: float
):
    detected_peaks = []
    chm_search = chm_smooth.copy()
    ny, nx = int(chm_smooth.shape[0]), int(chm_smooth.shape[1])

    for h_min, h_max, w_radius in window_bins:
        w_radius = max(1, int(round(w_radius / cell_size)))

        y, x = np.ogrid[-w_radius:w_radius+1, -w_radius:w_radius+1]
        footprint = x**2 + y**2 <= w_radius**2

        local_max = maximum_filter(chm_search, footprint=footprint)
        peak_mask = (chm_search == local_max) & (chm_search >= h_min) & (chm_search <= h_max)
        peaks_y, peaks_x = np.where(peak_mask)

        for py, px in zip(peaks_y, peaks_x):
            py, px = int(py), int(px)

            if chm_search[py, px] == 0:
                continue

            detected_peaks.append((py, px))
            y_start, y_end = max(0, py - w_radius), min(ny, py + w_radius + 1)
            x_start, x_end = max(0, px - w_radius), min(nx, px + w_radius + 1)

            rr, cc = np.ogrid[y_start:y_end, x_start:x_end]
            dist_sq = (rr - py)**2 + (cc - px)**2
            mask_area = dist_sq <= w_radius**2

            chm_search[y_start:y_end, x_start:x_end][mask_area] = 0

    if not detected_peaks:
        return np.empty((0, 2), dtype=int)

    return np.array(detected_peaks)

def generate_window_bins(area_type: str, min_height: float = 2.5) -> list:
    # Equation: radius = a + b*h + c*h^2
    # Curves are formulas taken from the paper bc too much work to compute them ourselves (I think)
    curves = {
        # Low Built-up / Parks (High canopy overlap, intricate multi-layered structure)
        # Mapped to Plot 6: Complex, Multi-Layered Mixed forest (95-121 pts/m2)
        "low_built_up": {"a": 3.061, "b": -0.052, "c": 0.003},

        # Mid Built-up / Suburban (Moderate overlap, mix of gardens and street trees)
        # Mapped to Plot 5: Complex, Single-Layered Mixed forest (30 pts/m2)
        "mid_built_up": {"a": 1.344, "b": 0.090, "c": -0.0001},

        # Densely Built-up (Isolated street trees, constrained crowns, clear boundaries)
        # Mapped to Plot 2: Simple, Mixed forest (11 pts/m2 - clear crown boundaries)
        "dense_built_up": {"a": 2.759, "b": -0.078, "c": 0.004}
    }

    coeffs = curves.get(area_type, curves["mid_built_up"])

    height_tiers = [
        (20.0, 999.0),
        (10.0, 20.0),
        (min_height, 10.0)
    ]

    window_bins = []

    for h_min, h_max in height_tiers:
        h_eval = h_min if h_max == 999.0 else (h_min + h_max) / 2.0
        r_calc = coeffs["a"] + (coeffs["b"] * h_eval) + (coeffs["c"] * (h_eval ** 2))

        r_final = max(1.0, float(r_calc))

        window_bins.append((h_min, h_max, r_final))
    return window_bins

def segment_trees_chm(
    #paramters for CHM segmentation
    pts_veg: np.ndarray,
    pts_ground: np.ndarray,
    cell_size: float = 1.0,
    smooth_sigma: float = 1.55,
    min_height: float = 2.5,
    peak_min_dist_m: float = 3.0,
    area_type: str = "mid_built_up",
    min_tree_points: int = 100,
    resolve_multi_trees: bool = True,
    multi_tree_kwargs: dict = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Segment individual trees from a CHM.

    Returns
    -------
    labels : (N,) int32
        Per-vegetation-point tree id; 0 = unassigned, 1..K = tree ids.
    dtm : (ny, nx) float64
        Digital Terrain Model raster (used later for trunk ground lookup).
    (x_min, y_min, cell_size) :
        Georeferencing of the raster.
    """
    if multi_tree_kwargs is None:
        multi_tree_kwargs = {}
    max_elongation = multi_tree_kwargs.pop("max_elongation", 3.0)
    max_offset_ratio = multi_tree_kwargs.pop("max_offset_ratio", 1.5)

    if len(pts_veg) == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros((1, 1), dtype=np.float64),
            (0.0, 0.0, cell_size)
        )

    x_min = pts_veg[:, 0].min()
    y_min = pts_veg[:, 1].min()
    x_max = pts_veg[:, 0].max()
    y_max = pts_veg[:, 1].max()

    if len(pts_ground) > 0:
        x_min = min(x_min, pts_ground[:, 0].min())
        y_min = min(y_min, pts_ground[:, 1].min())
        x_max = max(x_max, pts_ground[:, 0].max())
        y_max = max(y_max, pts_ground[:, 1].max())

    nx = int(np.ceil((x_max - x_min) / cell_size)) + 1
    ny = int(np.ceil((y_max - y_min) / cell_size)) + 1

    def to_rc(pts):
        c = np.clip(((pts[:, 0] - x_min) / cell_size).astype(int), 0, nx - 1)
        r = np.clip(((pts[:, 1] - y_min) / cell_size).astype(int), 0, ny - 1)
        return r, c

    # ---------------------------------------------------
    # DSM: max vegetation Z per cell
    # ---------------------------------------------------
    flat_dsm = np.full(ny * nx, -np.inf, dtype=np.float64)
    r_v, c_v = to_rc(pts_veg)
    np.maximum.at(flat_dsm, r_v * nx + c_v, pts_veg[:, 2])
    dsm = flat_dsm.reshape(ny, nx)
    dsm[dsm == -np.inf] = np.nan

    # ---------------------------------------------------
    # DTM: mean ground Z per cell, gaps filled by nearest valid cell
    # ---------------------------------------------------
    dtm_sum = np.zeros(ny * nx, dtype=np.float64)
    dtm_cnt = np.zeros(ny * nx, dtype=np.int32)
    r_g, c_g = to_rc(pts_ground)
    np.add.at(dtm_sum, r_g * nx + c_g, pts_ground[:, 2])
    np.add.at(dtm_cnt, r_g * nx + c_g, 1)
    dtm = np.where(
        dtm_cnt > 0,
        dtm_sum / np.where(dtm_cnt > 0, dtm_cnt, 1),
        np.nan,
    ).reshape(ny, nx)
    nan_mask = np.isnan(dtm)
    if nan_mask.all():
        dtm[:] = pts_veg[:, 2].min() if len(pts_veg) > 0 else 0.0
    elif nan_mask.any():
        fill_idx = distance_transform_edt(
            nan_mask, return_distances=False, return_indices=True
        )
        dtm = dtm[tuple(fill_idx)]

    # ---------------------------------------------------
    # CHM = DSM - DTM, clamped to >= 0
    # ---------------------------------------------------
    chm = np.where(~np.isnan(dsm), dsm - dtm, 0.0)
    chm = np.maximum(chm, 0.0)
    chm_masked = np.where(chm >= min_height, chm, 0.0)
    chm_smooth = gaussian_filter(chm_masked, sigma=smooth_sigma)

    window_bins = generate_window_bins(area_type, min_height)
    logging.info("Generated window bins")

    peaks = find_peaks_variable_window(chm_smooth, window_bins, cell_size=cell_size)
    tree_mask = chm_smooth >= min_height

    if len(peaks) == 0:
        logging.warning("CHM: no peaks found — check min_height or point density")
        return (
            np.zeros(len(pts_veg), dtype=np.int32),
            dtm,
            (x_min, y_min, cell_size),
        )

    # WATERSHED
    markers = np.zeros((ny, nx), dtype=np.int32)
    markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)
    labels_2d = watershed(-chm_smooth, markers, mask=tree_mask)

    labels = labels_2d[r_v, c_v].astype(np.int32)
    unique_lbls, counts = np.unique(labels, return_counts=True)

    next_new_label = labels.max() + 1
    suspect_count = 0
    small_dropped = 0
    added_trees = 0
    pole_wall_dropped = 0

    # FILTERING
    for lbl, cnt in zip(unique_lbls, counts):
        if lbl == 0:
            continue

        # Drop tiny segments
        if cnt < min_tree_points:
            labels[labels == lbl] = 0
            small_dropped += 1
            continue

        mask = labels == lbl
        cluster_pts = pts_veg[mask]

        cluster_r = r_v[mask]
        cluster_c = c_v[mask]

        height_above_ground = cluster_pts[:, 2] - dtm[cluster_r, cluster_c]

        if np.min(height_above_ground) > 7.0:
            labels[mask] = 0
            continue

        metrics = analyse_cluster(cluster_pts)

        # Drop poles and walls
        is_pole = metrics["z_range"] > 3.0 and metrics["std_major"] < 0.3
        is_wall = metrics["z_range"] > 3.0 and metrics["std_minor"] < 0.3 and metrics["elongation"] > 4.0

        if is_pole or is_wall:
            labels[mask] = 0
            pole_wall_dropped += 1
            continue

        # Further segment multi-tree clusters
        if resolve_multi_trees:
            is_elongated = metrics["elongation"] > max_elongation
            is_uncentered = metrics["offset_ratio"] > max_offset_ratio

            if is_elongated or is_uncentered:
                suspect_count += 1

                peaks_3d = find_secondary_peak(cluster_pts, **multi_tree_kwargs)

                # Assign points to the nearest (2D Euclidean distance) secondary peak identified
                if len(peaks_3d) > 1:
                    dists = cdist(cluster_pts[:, :2], peaks_3d[:, :2])

                    sub_labels = np.argmin(dists, axis=1)
                    original_indices = np.where(mask)[0]

                    for peak_idx in range(1, len(peaks_3d)):
                        new_label_mask = (sub_labels == peak_idx)
                        indices_to_change = original_indices[new_label_mask]

                        labels[indices_to_change] = next_new_label
                        next_new_label += 1

                    added_trees += (len(peaks_3d) - 1)

    logging.info(f"CHM: {len(peaks)} initial peaks found.")
    logging.info(f"Filter: {small_dropped} tiny segments (< {min_tree_points} pts) and {pole_wall_dropped} pole or wall like segments dropped.")
    logging.info(f"PCA: {suspect_count} multi-tree clusters detected, adding {added_trees} new trees.")

    return labels, dtm, (x_min, y_min, cell_size)
