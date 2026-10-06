import json
import subprocess
import copy
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from shapely import contains_xy


def clip_and_merge_pointclouds(folder_path: str, outpout_filename: str = "clipped_output.laz"):
    folder = Path(folder_path)

    gpkg_files = [f for f in folder.glob("*.gpkg") if not f.name.endswith("-trees.gpkg")]
    if not gpkg_files:
        raise FileNotFoundError(f"No .gpkg AOI file found in {folder_path}")
    gpkg_path = gpkg_files[0]

    gdf = gpd.read_file(gpkg_path)

    if len(gdf) > 1:
        unbuffered = gdf.geometry.union_all()
    else:
        unbuffered = gdf.geometry.iloc[0]

    buffered_polygon = unbuffered.buffer(0.5)

    laz_files = [f for f in folder.rglob("*.laz") if f.name != outpout_filename]
    print(f"Found {len(laz_files)} .laz file(s) to process.")

    if not laz_files:
        raise FileNotFoundError("No .laz files found to process.")

    output_laz = folder / outpout_filename
    wkt = buffered_polygon.wkt

    pipeline_steps = [str(tile) for tile in laz_files]
    pipeline_steps.append({"type": "filters.crop", "polygon": wkt})
    pipeline_steps.append(str(output_laz))

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