from pathlib import Path
from shapely import contains_xy

import geopandas as gpd
import laspy
import numpy as np


def clip_and_merge_pointclouds(
        tiles_folder: str,
        output_folder: str,
        gpkg_path: str,
        tile_ids: list[str] = None,
        output_filename: str = "clipped_output.laz"
):
    tiles_dir = Path(tiles_folder)
    out_dir = Path(output_folder)

    # ---------------------------------------------------
    # ------------ Find GKPG and buffer it --------------
    # ---------------------------------------------------
    if not Path(gpkg_path).exists():
        raise FileNotFoundError(f"AOI file not found at {gpkg_path}")

    gdf = gpd.read_file(gpkg_path)

    if len(gdf) > 1:
        unbuffered = gdf.geometry.union_all()
    else:
        unbuffered = gdf.geometry.iloc[0]

    buffered_polygon = unbuffered.buffer(0.5)

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

    laz_files = [f for f in laz_files if f.name != output_filename]
    print(f"Found {len(laz_files)} .laz file(s) to process.")

    if not laz_files:
        raise FileNotFoundError("No .laz files found to process.")

    # ---------------------------------------------------
    # --------- Clip each LAZ file to the AOI -----------
    # ---------------------------------------------------
    output_laz = out_dir / output_filename

    points_list = []
    clean_header = None

    for tile_path in laz_files:
        las = laspy.read(str(tile_path))
        if clean_header is None:
            clean_header = laspy.LasHeader(version=las.header.version, point_format=las.header.point_format)
            clean_header.scales = las.header.scales
            clean_header.offsets = las.header.offsets

            for vlr in las.header.vlrs:
                if getattr(vlr, "user_id", "") != "copc":
                    clean_header.vlrs.append(vlr)

            for evlr in las.header.evlrs:
                if getattr(evlr, "user_id", "") != "copc":
                    clean_header.evlrs.append(evlr)
        # Make sure at the next laz file it uses that offset
        else:
            las.change_scaling(scales=clean_header.scales, offsets=clean_header.offsets)

        mask = contains_xy(buffered_polygon, las.x, las.y)
        points_list.append(las.points[mask].array)

    if not points_list:
        raise ValueError("No points found within the buffered geometry.")

    concatenated_array = np.concatenate(points_list)

    out = laspy.LasData(clean_header)
    out.points = laspy.ScaleAwarePointRecord(
        concatenated_array,
        point_format=clean_header.point_format,
        scales=clean_header.scales,
        offsets=clean_header.offsets
    )
    out.write(str(output_laz))
    print(f"Merged & clipped point cloud saved (laspy) -> {output_laz}")



if __name__ == "__main__":
    script_dir = Path(__file__).parent
    target_folder = script_dir.parent / "data" / "in" / "all-files" / "ams-mid-2"

    clip_and_merge_pointclouds(target_folder)