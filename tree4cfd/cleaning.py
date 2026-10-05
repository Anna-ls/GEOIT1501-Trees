"""Point-cloud cleaning: statistical outlier removal and voxel downsampling."""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree


def remove_outliers_sor(pts: np.ndarray, n_neighbors: int, std_ratio: float) -> np.ndarray:
    """Statistical Outlier Removal.

    Drops points whose mean distance to their ``n_neighbors`` nearest neighbours
    exceeds ``mean + std_ratio * std`` over the whole cloud.
    """
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=n_neighbors + 1, workers=-1)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    return pts[mean_dists < threshold]


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one representative point per occupied voxel."""
    voxel_ids = np.floor(pts / voxel_size).astype(np.int32)
    _, unique_idx = np.unique(voxel_ids, axis=0, return_index=True)
    return pts[unique_idx]
