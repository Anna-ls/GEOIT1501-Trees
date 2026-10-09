from pathlib import Path
import geopandas as gpd
import laspy
import numpy as np
from shapely import box, contains_xy

def clip_and_merge_pointclouds(
        tiles_folder: str,
        output_folder: str,
        gpkg_path: str,
        tile_ids: list[str] = None,
        output_filename: str = "clipped_output.laz",
        max_workers: int = None
):
    tiles_dir = Path(tiles_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # ------------ Find GKPG and buffer it --------------
    # ---------------------------------------------------
    if not Path(gpkg_path).exists():
        raise FileNotFoundError(f"AOI file not found at {gpkg_path}")

    gdf = gpd.read_file(gpkg_path)
    unbuffered = gdf.geometry.union_all() if len(gdf) > 1 else gdf.geometry.iloc[0]
    buffered_polygon = unbuffered.buffer(0.5)
    p_minx, p_miny, p_maxx, p_maxy = buffered_polygon.bounds

    # ---------------------------------------------------
    # -- Check if the tile has been downloaded before ---
    # ---------------------------------------------------
    if tile_ids:
        laz_files = []
        for tile_id in tile_ids:
            tile_folder = tiles_dir / "tiles" / tile_id
            if tile_folder.exists() and tile_folder.is_dir():
                laz_files.extend(list(tile_folder.rglob("*.laz")))
    else:
        laz_files = list(tiles_dir.rglob("*.laz"))

    output_laz = out_dir / output_filename
    laz_files = [f for f in laz_files if f.resolve() != output_laz.resolve()]
    print(f"Found {len(laz_files)} .laz file(s) to process.")
    if not laz_files:
        raise FileNotFoundError("No .laz files found to process.")

    # ---------------------------------------------------
    # --------- Clip each LAZ file to the AOI -----------
    # ---------------------------------------------------
    output_laz = out_dir / output_filename

    candidate_tiles = []
    for f in laz_files:
        with laspy.open(str(f)) as r:
            t_minx, t_miny, _ = r.header.mins
            t_maxx, t_maxy, _ = r.header.maxs

            # Skip tile entirely if bounding boxes do not overlap
            if not (t_maxx < p_minx or t_minx > p_maxx or t_maxy < p_miny or t_miny > p_maxy):
                candidate_tiles.append(f)

    with laspy.open(str(laz_files[0])) as first_las:
        clean_header = laspy.LasHeader(version=first_las.header.version, point_format=first_las.header.point_format)
        clean_header.scales = first_las.header.scales
        clean_header.offsets = first_las.header.offsets

        for vlr in first_las.header.vlrs:
            if getattr(vlr, "user_id", "") != "copc":
                clean_header.vlrs.append(vlr)

        for evlr in first_las.header.evlrs:
            if getattr(evlr, "user_id", "") != "copc":
                clean_header.evlrs.append(evlr)

    points_written = 0
    all_mins = [np.inf, np.inf, np.inf]
    all_maxs = [-np.inf, -np.inf, -np.inf]

    with laspy.open(str(output_laz), mode="w", header=clean_header) as writer:
        for tile_path in candidate_tiles:
            with laspy.open(str(tile_path)) as las:
                t_minx, t_miny, _ = las.header.mins
                t_maxx, t_maxy, _ = las.header.maxs
                tile_box = box(t_minx, t_miny, t_maxx, t_maxy)
                is_fully_contained = buffered_polygon.contains(tile_box)

                for chunk in las.chunk_iterator(1_000_000):
                    if is_fully_contained:
                        matched = chunk
                    else:
                        x, y = chunk.x, chunk.y

                        # Vectorized bounding box pre-screen
                        bbox_mask = (x >= p_minx) & (x <= p_maxx) & (y >= p_miny) & (y <= p_maxy)
                        if not np.any(bbox_mask):
                            continue

                        # Precise shapely check on candidate subset
                        cand_idx = np.flatnonzero(bbox_mask)
                        poly_mask = contains_xy(buffered_polygon, x[cand_idx], y[cand_idx])

                        if not np.any(poly_mask):
                            continue

                        matched = chunk[cand_idx[poly_mask]]

                    writer.write_points(matched)
                    points_written += len(matched)

                    all_mins = np.minimum(all_mins, [matched.x.min(), matched.y.min(), matched.z.min()])
                    all_maxs = np.maximum(all_maxs, [matched.x.max(), matched.y.max(), matched.z.max()])

    if points_written == 0:
        output_laz.unlink(missing_ok=True)
        raise ValueError("No points found within the buffered geometry.")

    with laspy.open(str(output_laz), mode="a") as dst:
        dst.header.mins = all_mins
        dst.header.maxs = all_maxs

    print(f"Merged & clipped point cloud saved (laspy chunked) -> {output_laz}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    target_folder = script_dir.parent / "data" / "in" / "all-files" / "ams-mid-2"

    clip_and_merge_pointclouds(target_folder)