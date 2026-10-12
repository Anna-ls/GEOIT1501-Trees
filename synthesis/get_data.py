import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
import time
import requests
import zipfile

from CFTree.config import get_config, setup_logger
from CFTree.download_geotiles import download_tile
from clip_and_merge import clip_and_merge_pointclouds
from clip_and_merge import clip_and_merge_objs

# ---------------------------------------------------------------
# ----------------------- Tile worker ---------------------------
# ---------------------------------------------------------------
def download_ahn_tiles(tile_id: str, output_dir: Path, base_url: str, overwrite: bool) -> dict:
    """Download raw AHN tile"""
    try:
        # Check if tile is already downloaded
        tile_folder = output_dir / "tiles" / tile_id
        if not overwrite and tile_folder.exists() and tile_folder.is_dir():
            existing_laz = list(tile_folder.glob("*.laz"))
            if existing_laz:
                return {
                    "tile_id": tile_id,
                    "status": "ok (cached)",
                    "paths": {
                        "raw": str(existing_laz[0]),
                    },
                }
        # If not, download raw tile
        result_dl = download_tile(tile_id, output_dir, base_url, overwrite=overwrite)
        laz_path = result_dl.get("paths", {}).get("laz")

        if result_dl["status"] != "ok" or not laz_path or not Path(laz_path).exists():
            return {"tile_id": tile_id, "status": "download_failed"}

        # Return tile summary
        return {
            "tile_id": tile_id,
            "status": "ok",
            "paths": {
                "raw": str(laz_path),
            },
        }

    except Exception as e:
        logging.exception(f"[{tile_id}] Unexpected error: {e}")
        return {"tile_id": tile_id, "status": f"error: {e}"}

def download_bag_obj(download_url: str, output_dir: Path, tile_id: str, overwrite: bool) -> dict:
    """Download BAG 3D obj tile"""
    try:
        if not download_url:
            return {"tile_id": tile_id, "status": "no_url"}

        bag_dir = Path(output_dir) / "3dbag_tiles"
        url_parts = download_url.split("/")
        filename = url_parts[-1]
        zip_path = bag_dir / filename

        nested_folder_parts = url_parts[-4:-1]
        expected_dir = bag_dir.joinpath(*nested_folder_parts)

        if not overwrite and expected_dir.exists():
            return {"tile_id": tile_id, "status": "ok (cached)", "path": str(expected_dir)}

        zip_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(download_url, stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if zip_path.suffix == '.zip':
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(bag_dir)
            zip_path.unlink()
            return {"tile_id": tile_id, "status": "ok (extracted from zip)", "path": str(expected_dir)}
        else:
            return {"tile_id": tile_id, "status": "ok", "path": str(zip_path)}

    except Exception as e:
        logging.exception(f"[3D BAG {tile_id}] Unexpected error: {e}")
        return {"tile_id": tile_id, "status": f"error: {e}"}

# ---------------------------------------------------------------
# ----------------- Process the GeoJSON file --------------------
# ---------------------------------------------------------------
def process_geojson(
        geojson_path: str | Path,
        buffer_distance: float = 10.0,
        overwrite: bool = False,
        log_level: str = "INFO",
        download_buildings: bool = False,
        bag_lod: str = "LoD22",
):
    """Run get_data pipeline for a specific GeoJSON area of interest."""
    geojson_path = Path(geojson_path).resolve()
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    case = geojson_path.stem

    cfg = get_config(case_name=case)
    n_cores = cfg.get("default_cores", 1)

    setup_logger(case, "get_data", log_level)

    logging.info(f"Starting data download for the defined AOI")
    # ---------------------------------------------------
    # -------------- Define directories -----------------
    # ---------------------------------------------------
    root_dir = Path(__file__).resolve().parent.parent
    resources_dir = root_dir / "resources"

    output_dir = geojson_path.parent
    tiles_dir = root_dir / "DATA" / "OUT"

    buffered_aoi_path = output_dir / f"{case}_buffered.geojson"
    gpkg_aoi_path = output_dir / f"{case}_aoi.gpkg"

    # ---------------------------------------------------
    # ------------- Load and buffer AOI -----------------
    # ---------------------------------------------------
    logging.info(f"Loading AOI from {geojson_path}")
    aoi = gpd.read_file(geojson_path).to_crs(cfg["crs"])
    aoi["geometry"] = aoi.buffer(buffer_distance)

    aoi.to_file(buffered_aoi_path, driver="GeoJSON")
    aoi.to_file(gpkg_aoi_path, driver="GPKG")
    logging.info(f"Buffered AOI saved as GPKG and GeoJSON")

    # ---------------------------------------------------
    # ---------------- Find AHN tile IDs ----------------
    # ---------------------------------------------------
    logging.info("Loading AHN index to find intersections...")
    tiles = gpd.read_file(resources_dir / "bladwijzer_AHN6.gpkg", layer="bladindeling").to_crs(cfg["crs"])
    intersecting = tiles[tiles.intersects(aoi.union_all())]
    tile_ids = intersecting.apply(
        lambda row: f"{int(row['left']):06d}_{int(row['bottom']):06d}",
        axis=1
    ).unique().tolist()

    if not tile_ids:
        logging.info("No intersecting tiles found.")
        return 0.0, 0.0
    logging.info(f"Found {len(tile_ids)} intersecting tiles: {tile_ids if len(tile_ids) <= 10 else '...'}")

    base_url = "https://fsn1.your-objectstorage.com/hwh-ahn/AHN5_KM/01_LAZ"

    # ---------------------------------------------------
    # ---------------- Find BAG tile IDs ----------------
    # ---------------------------------------------------
    bag_index_path = resources_dir / "tile_index.fgb"
    bag_downloads = []

    if download_buildings:
        logging.info(f"Loading 3D BAG index to find intersections...")
        bag_tiles = gpd.read_file(bag_index_path).to_crs(cfg["crs"])
        intersecting_bag = bag_tiles[bag_tiles.intersects(aoi.union_all())]

        if not intersecting_bag.empty:
            bag_downloads = intersecting_bag[["tile_id", "obj_download"]].dropna().to_dict("records")

        if not bag_downloads:
            logging.info("No intersecting 3D BAG tiles found.")
        else:
            logging.info(f"Found {len(bag_downloads)} intersecting 3D BAG tiles")

    # ---------------------------------------------------
    # ----------------- Download tiles ------------------
    # ---------------------------------------------------
    t_start_download = time.perf_counter()
    if n_cores > 1:
        logging.info(f"Processing {len(tile_ids)} AHN tiles and {len(bag_downloads)} 3D BAG tiles in parallel using {n_cores} cores.")
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            ahn_futures = {
                pool.submit(download_ahn_tiles, tid, tiles_dir, base_url, overwrite): tid
                for tid in tile_ids
            }
            for f in as_completed(ahn_futures):
                tid = ahn_futures[f]
                try:
                    result = f.result()
                    logging.info(f"[{tid}] {result['status'].upper()}")
                except Exception as e:
                    logging.warning(f"[{tid}] Exception: {e}")

            if download_buildings:
                bag_futures = {
                    pool.submit(download_bag_obj, item["obj_download"], tiles_dir, item["tile_id"], overwrite): item["tile_id"]
                    for item in bag_downloads
                }
                for f in as_completed(bag_futures):
                    tid = bag_futures[f]
                    try:
                        result = f.result()
                        logging.info(f"[3D BAG {tid}] {result['status'].upper()}")
                    except Exception as e:
                        logging.warning(f"[3D BAG {tid}] Exception: {e}")
    else:
        logging.info("Running serial mode.")
        for tid in tile_ids:
            result = download_ahn_tiles(tid, tiles_dir, base_url, overwrite)
            logging.info(f"[AHN {tid}] {result['status'].upper()}")
        if download_buildings:
            for item in bag_downloads:
                result = download_bag_obj(item["obj_download"], tiles_dir, item["tile_id"], overwrite)
                logging.info(f"[3D BAG {item['tile_id']}] {result['status'].upper()}")
    download_time = time.perf_counter() - t_start_download

    # ---------------------------------------------------
    # ---------------- Clip and merge -------------------
    # ---------------------------------------------------
    t_start_clip = time.perf_counter()
    logging.info("Starting clip and merge of AHN tiles...")
    try:
        clip_and_merge_pointclouds(str(tiles_dir), str(output_dir), gpkg_path=str(gpkg_aoi_path), tile_ids=tile_ids, output_filename="clipped_output.laz")
        logging.info("AHN clip and merge process completed.")
    except Exception as e:
        logging.exception(f"AHN clip and merge process failed. Error: {e}")

    ahn_clip_time = time.perf_counter() - t_start_clip

    bag_clip_time = 0.0

    if download_buildings:
        bag_start_time = time.perf_counter()
        logging.info("Starting clip and merge of 3D BAG tiles...")
        try:
            bag_dir = Path(tiles_dir) / "3dbag_tiles"
            target_objs = list(bag_dir.rglob(f"*-{bag_lod}-3D.obj"))

            clipped_buildings_path = Path(output_dir) / "clipped_buildings.obj"

            clip_and_merge_objs(
                obj_paths=target_objs,
                aoi_gpkg_path=str(gpkg_aoi_path),
                output_obj_path=clipped_buildings_path,
            )

        except Exception as e:
            logging.exception(f"3D BAG clip and merge process failed. Error: {e}")

        bag_clip_time = time.perf_counter() - bag_start_time


    total_time = ahn_clip_time + bag_clip_time + download_time
    logging.info(f"Completed data collection in {total_time:.1f} seconds.")

    return download_time, ahn_clip_time, bag_clip_time


# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_data.py case_name")
        sys.exit(1)

    input_geojson = sys.argv[1]
    process_geojson(input_geojson)
