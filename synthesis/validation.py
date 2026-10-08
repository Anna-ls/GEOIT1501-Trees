import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import json
import geopandas as gpd
import laspy
import numpy as np
from scipy.spatial import cKDTree

from tree4cfd.buildings import filter_trees_in_buildings
from tree4cfd.segmentation import segment_trees_chm

# ---- CONFIGURATION ---------------------------------------------------
def configuration(case_name):
    PARK_NAME = case_name
    with open(root_dir / "DATA" / "IN" / case_name / "validation_config.json", "r") as f:
        config = json.load(f)
    PARK_POLYGON_PATH = root_dir / config["PARK_POLYGON_PATH"]

    AHN_INPUT_TILES = [
        root_dir / tile for tile in config["AHN_INPUT_TILES"]
    ]

    AHN_CLIPPED_OUT = root_dir / config["AHN_CLIPPED_OUT"]
    TREE_REGISTRY_PARQUET = root_dir / config["TREE_REGISTRY_PARQUET"]

    CLIP_BUFFER_M = 3.0
    CRS = "EPSG:28992"  # RD New

    return PARK_NAME, PARK_POLYGON_PATH, AHN_INPUT_TILES, AHN_CLIPPED_OUT, TREE_REGISTRY_PARQUET, CLIP_BUFFER_M, CRS

def load_park_polygon(path, crs):
    """Loads local GeoPackage boundary."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Ensure {path} exists.")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("Park polygon has no CRS set")

    gdf = gdf.to_crs(crs)
    if len(gdf) > 1:
        print(f"  [!] {len(gdf)} features matched -- dissolving into one polygon")
        unbuffered = gdf.geometry.union_all()
    else:
        unbuffered = gdf.geometry.iloc[0]

    return unbuffered

def extract_detected_trees(pts_veg: np.ndarray, labels: np.ndarray, crs) -> gpd.GeoDataFrame:
    """Converts segmented tree point labels into GeoPandas point centroids."""
    tree_records = []
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        if lbl == 0:
            continue
        cluster = pts_veg[labels == lbl]
        
        # Take highest point (tree top) as point geometry
        max_idx = np.argmax(cluster[:, 2])
        top_x, top_y, top_z = cluster[max_idx]
        
        tree_records.append({
            "tree_id": int(lbl),
            "geometry": gpd.points_from_xy([top_x], [top_y])[0],
            "z_max": float(top_z),
            "point_count": len(cluster)
        })

    return gpd.GeoDataFrame(tree_records, crs=crs)

def count_registry_trees(polygon, registry_path, PARK_NAME, crs):
    """Counts ground-truth registered trees within the unbuffered park boundary."""
    if not registry_path.exists():
        print(f"  [!] Registry file not found at '{registry_path}'. Skipping registry count.")
        return None

    trees = gpd.read_file(registry_path)
    if trees.crs is None:
        raise ValueError("Registry file has no CRS set")

    trees = trees.to_crs(crs)
    inside = trees[trees.geometry.within(polygon)]
    print(f"  [✓] Registry reference count inside {PARK_NAME}: {len(inside)}")
    return inside

def compare_counts(registry_inside, detected_inside, PARK_NAME):
    """Prints evaluation statistics comparing detected trees against registry ground truth."""
    if registry_inside is None or detected_inside is None:
        print("  [!] Skipping comparison: missing dataset.")
        return

    n_ref = len(registry_inside)
    n_det = len(detected_inside)
    diff = n_det - n_ref
    pct = (diff / n_ref * 100) if n_ref else float("nan")

    print(f"\n==========================================")
    print(f"   {PARK_NAME.upper()} VALIDATION SUMMARY")
    print(f"==========================================")
    print(f"  Municipal Registry (Reference): {n_ref} trees")
    print(f"  Anna's CHM Segmentation:       {n_det} trees")
    print(f"  Delta Count:                    {diff:+d} ({pct:+.1f}%)")
    print(f"==========================================\n")

def evaluate_spatial_accuracy(detected_gdf: gpd.GeoDataFrame, registry_gdf: gpd.GeoDataFrame, dist_thresh: float = 2.5):
    """Calculates Precision, Recall, and F1-score against reference points within a buffer."""
    if len(detected_gdf) == 0 or len(registry_gdf) == 0:
        print("  [!] Cannot compute spatial accuracy: one of the input point sets is empty.")
        return 0.0, 0.0, 0.0

    ref_coords = np.array(list(zip(registry_gdf.geometry.x, registry_gdf.geometry.y)))
    det_coords = np.array(list(zip(detected_gdf.geometry.x, detected_gdf.geometry.y)))

    tree_kd = cKDTree(ref_coords)
    distances, indices = tree_kd.query(det_coords, distance_upper_bound=dist_thresh)

    matched_mask = distances < dist_thresh
    matched_ref_indices = set(indices[matched_mask])

    tp = len(matched_ref_indices)
    fp = len(detected_gdf) - tp
    fn = len(registry_gdf) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n==========================================")
    print("      DELFT SPATIAL VALIDATION RESULTS    ")
    print("==========================================")
    print(f"  Reference Registry Count : {len(registry_gdf)}")
    print(f"  Detected Trees Count     : {len(detected_gdf)}")
    print(f"  True Positives  (TP)     : {tp}")
    print(f"  False Positives (FP)     : {fp}")
    print(f"  False Negatives (FN)     : {fn}")
    print(f"  ----------------------------------------")
    print(f"  Precision                : {precision:.3f}")
    print(f"  Recall                   : {recall:.3f}")
    print(f"  F1-Score                 : {f1:.3f}")
    print("==========================================\n")

    return precision, recall, f1


# ---- MAIN EXECUTION PIPELINE -----------------------------------------

def validation_pipeline(case_name, parameters: dict):
    PARK_NAME, PARK_POLYGON_PATH, AHN_INPUT_TILES, AHN_CLIPPED_OUT, TREE_REGISTRY_PARQUET, CLIP_BUFFER_M, CRS = configuration(case_name)

    multi_tree_kwargs = {
        "max_elongation": parameters["max_elongation"],
        "max_offset_ratio": parameters["max_offset_ratio"],
        "num_angles": parameters["num_angles"],
        "bin_size": parameters["bin_size"],
        "peak_min_dist_m": parameters["min_peak_dist_3d"],
        "d_euclidean_thresh": parameters["d_euclidean_thresh"],
        "d_margin_thresh": parameters["d_margin_thresh"]
    }

    print(f"--- Running Validation Pipeline for {PARK_NAME} ---")

    unbuffered = load_park_polygon(PARK_POLYGON_PATH, CRS)
    # ---------------------------------------------------
    # 3. Read LAZ and extract vegetation/ground coordinates
    # ---------------------------------------------------
    print(f"  Loading point cloud data...")
    las = laspy.read(str(AHN_CLIPPED_OUT))
    total_clipped_points = len(las.points)

    unique_classes, counts = np.unique(
    las.classification,
    return_counts=True
    )

    print("\n  Point classifications:")
    for cls, count in zip(unique_classes, counts):
        print(f"    Class {cls:2d}: {count:,} points")

    is_ground = las.classification == 2
    is_veg = las.classification == 1

    pts_ground = np.column_stack((
        las.x[is_ground],
        las.y[is_ground],
        las.z[is_ground]
    ))
    pts_veg = np.column_stack((
        las.x[is_veg],
        las.y[is_veg],
        las.z[is_veg]
    ))

    print(
        f"  Loaded {len(pts_veg):,} vegetation points "
        f"and {len(pts_ground):,} ground points."
    )

    # ---------------------------------------------------
    # 4. Call external segmentation module
    # ---------------------------------------------------
    print("  Executing additional CHM segmentation...")
    labels, _, _ = segment_trees_chm(
        pts_veg,
        pts_ground,
        cell_size=parameters["cell_size"],
        smooth_sigma=parameters["smooth_sigma"],
        min_height=parameters["min_height"],
        peak_min_dist_m=parameters["peak_min_dist_m"],
        min_tree_points=parameters["min_tree_points"],
        resolve_multi_trees=parameters["resolve_multi_trees"],
        multi_tree_kwargs=multi_tree_kwargs
    )

    print("  Filtering trees inside building footprints...")
    epsg_code = CRS.to_epsg() if hasattr(CRS, "to_epsg") else int(CRS.split(":")[-1])
    labels, removed_count = filter_trees_in_buildings(
        labels=labels,
        pts_xy=pts_veg[:, :2],
        translation=(0.0, 0.0),
        epsg=epsg_code,
        buffer=0.5
    )
    print(f"    Removed {removed_count} trees overlapping with buildings.")

    # ---------------------------------------------------
    # 5. Extract tree centroids and count trees inside unbuffered park boundary
    # ---------------------------------------------------
    detected_gdf = extract_detected_trees(pts_veg, labels, CRS)
    detected_gdf.to_file(f"../DATA/OUT/{case_name}_detected_trees.gpkg", driver="GPKG")
    detected_inside = detected_gdf[detected_gdf.geometry.within(unbuffered)]

    # ---------------------------------------------------
    # 6. Load ground truth registry and compare counts
    # ---------------------------------------------------
    registry_inside = count_registry_trees(unbuffered, TREE_REGISTRY_PARQUET, PARK_NAME, CRS)
    compare_counts(registry_inside, detected_inside, PARK_NAME)

    precision, recall, f1_score = evaluate_spatial_accuracy(detected_gdf, registry_inside, dist_thresh = 2.5)
    with open(f"../DATA/OUT/f1_scores.txt", "a") as f:
        f.write(f"{PARK_NAME}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}\n")

    return registry_inside, detected_inside, precision, recall, f1_score, total_clipped_points

if __name__ == "__main__":
    with open(f"./parameters_segmentation.json", "r") as f:
        params = json.load(f)


    validation_pipeline("vondelpark", params)