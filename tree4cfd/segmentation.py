"""Canopy Height Model (CHM) based individual-tree segmentation.

Builds a DTM (from ground points) and a DSM (from vegetation points) on a
common grid, differences them into a CHM, finds tree tops as local maxima,
and grows crowns with a watershed.
"""

from __future__ import annotations

from typing import Tuple, List, Dict

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
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
        smooth_sigma: float = 1.55,
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

def segment_trees_chm(
    #paramters for CHM segmentation
    pts_veg: np.ndarray,
    pts_ground: np.ndarray,
    cell_size: float = 1.0,
    smooth_sigma: float = 1.55,
    min_height: float = 2.5,
    peak_min_dist_m: float = 3.0,
    min_tree_points: int = 60,
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

    x_min = min(pts_veg[:, 0].min(), pts_ground[:, 0].min())
    y_min = min(pts_veg[:, 1].min(), pts_ground[:, 1].min())
    x_max = max(pts_veg[:, 0].max(), pts_ground[:, 0].max())
    y_max = max(pts_veg[:, 1].max(), pts_ground[:, 1].max())

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
    if nan_mask.any():
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

    tree_mask = chm_smooth >= min_height
    min_dist_cells = max(1, int(peak_min_dist_m / cell_size))
    peaks = peak_local_max(
        chm_smooth, min_distance=min_dist_cells, labels=tree_mask
    )

    if len(peaks) == 0:
        print("    CHM: no peaks found — check min_height or point density")
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

        mask = labels == lbl
        cluster_pts = pts_veg[mask]
        metrics = analyse_cluster(cluster_pts)

        # Drop tiny segments
        if cnt < min_tree_points:
            labels[labels == lbl] = 0
            small_dropped += 1
            continue

        # Drop poles and walls
        is_pole = metrics["z_range"] > 3.0 and metrics["std_major"] < 0.6
        is_wall = metrics["z_range"] > 3.0 and metrics["std_minor"] < 0.4 and metrics["elongation"] > 4.0

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

    print(f"    CHM: {len(peaks)} initial peaks found.")
    print(f"    Filter: {small_dropped} tiny segments (< {min_tree_points} pts) and {pole_wall_dropped} pole or wall like segments dropped.")
    print(f"    PCA: {suspect_count} multi-tree clusters detected, adding {added_trees} new trees.")

    return labels, dtm, (x_min, y_min, cell_size)
