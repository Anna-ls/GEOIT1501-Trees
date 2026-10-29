import geopandas as gpd
from pathlib import Path
import warnings
import pandas as pd
from synthesis.validation import evaluate_spatial_accuracy

def validate_against_inventories(input_polygon, detected_trees_gdf, inventory_dir: Path, hulls_gpkg: Path, crs="EPSG:28992"):
    if not hulls_gpkg.exists():
        warnings.warn("Inventory hulls not found. Skipping spatial validation.")
        return None, None, None

    hulls_gdf = gpd.read_file(hulls_gpkg)
    if hulls_gdf.crs != crs:
        hulls_gdf = hulls_gdf.to_crs(crs)

    intersecting_hulls = hulls_gdf[hulls_gdf.geometry.intersects(input_polygon)]

    if intersecting_hulls.empty:
        print("No ground truth inventory available for this polygon. Skipping evaluation.")
        return None, None, None

    combined_hull = intersecting_hulls.geometry.union_all()
    if not input_polygon.within(combined_hull):
        warnings.warn("The input polygon partially extends outside the bounds of the ground truth inventory.")
        return None, None, None

    relevant_inventories = []
    for idx, row in intersecting_hulls.iterrows():
        inv_path = inventory_dir / f"{row['inventory_name']}.parquet"
        if inv_path.exists():
            inv_gdf = gpd.read_parquet(inv_path)
            if inv_gdf.crs != crs:
                inv_gdf = inv_gdf.to_crs(crs)
            relevant_inventories.append(inv_gdf)

    if not relevant_inventories:
        return None, None, None

    combined_points = gpd.GeoDataFrame(pd.concat(relevant_inventories, ignore_index=True), crs=crs)
    clipped_inventory = combined_points[combined_points.geometry.within(input_polygon)]

    precision, recall, f1_score = evaluate_spatial_accuracy(
        detected_trees_gdf,
        clipped_inventory,
        dist_thresh=2.5
    )

    return precision, recall, f1_score