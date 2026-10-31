import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
import time
import requests
import gzip
import shutil

from CFTree.config import get_config, setup_logger
from CFTree.download_tiles import download_tile
from clip_and_merge import clip_and_merge_pointclouds, clip_and_merge_cityjson, cityjson_to_obj

# ---------------------------------------------------------------
# ----------------------- Tile worker ---------------------------
# ---------------------------------------------------------------
def download_ahn_tiles(tile_id: str, output_dir: Path, base_urls: list[str], overwrite: bool) -> dict:
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
        for base_url in base_urls:
            result_dl = download_tile(tile_id, output_dir, base_url, overwrite=overwrite)
            laz_path = result_dl.get("paths", {}).get("laz")

            if result_dl["status"] == "ok" and laz_path and Path(laz_path).exists():
                return {"tile_id": tile_id, "status": "ok", "paths": {"raw": str(laz_path),}, "source_url": base_url}

        # Return tile summary
        return {
            "tile_id": tile_id,
            "status": "download_failed",
        }

    except Exception as e:
        logging.exception(f"[{tile_id}] Unexpected error: {e}")
        return {"tile_id": tile_id, "status": f"error: {e}"}

def download_bag_cityjson(download_url: str, output_dir: Path, tile_id: str, overwrite: bool) -> dict:
    """Download BAG 3D CityJSON tile"""
    try:
        if not download_url:
            return {"tile_id": tile_id, "status": "no_url"}

        bag_dir = Path(output_dir) / "3dbag_tiles"
        bag_dir.mkdir(parents=True, exist_ok=True)

        filename = download_url.split("/")[-1]
        is_gzipped = filename.endswith(".gz")

        final_filename = filename[:-3] if is_gzipped else filename
        json_path = bag_dir / final_filename
        gz_path = bag_dir / filename

        if not overwrite and json_path.exists():
            return {"tile_id": tile_id, "status": "ok (cached)", "path": str(json_path)}

        response = requests.get(download_url, stream=True, headers={'Accept-Encoding': 'identity'})
        response.raise_for_status()

        target_path = gz_path if is_gzipped else json_path
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if is_gzipped:
            with gzip.open(gz_path, 'rb') as f_check:
                magic = f_check.read(2)
                if magic == b'\x1f\x8b':
                    # File is actually gzipped; decompress it
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(json_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    gz_path.unlink()
                else:
                    # requests auto-decompressed it; rename the file directly
                    shutil.move(gz_path, json_path)

        return {"tile_id": tile_id, "status": "ok", "path": str(json_path)}

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
        bag_lod: str = "2.2",
        translation: list[float] = None
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

    ahn_base_urls = [
        "https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/01_LAZ",
        "https://fsn1.your-objectstorage.com/hwh-ahn/AHN5_KM/01_LAZ"
    ]

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
            bag_downloads = intersecting_bag[["tile_id", "cj_download"]].dropna().to_dict("records")

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
                pool.submit(download_ahn_tiles, tid, tiles_dir, ahn_base_urls, overwrite): tid
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
                    pool.submit(download_bag_cityjson, item["cj_download"], tiles_dir, item["tile_id"], overwrite): item["tile_id"]
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
            result = download_ahn_tiles(tid, tiles_dir, ahn_base_urls, overwrite)
            logging.info(f"[AHN {tid}] {result['status'].upper()}")
        if download_buildings:
            for item in bag_downloads:
                result = download_bag_cityjson(item["cj_download"], tiles_dir, item["tile_id"], overwrite)
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
            target_json = list(bag_dir.rglob("*.city.json"))

            clipped_buildings_path = Path(output_dir) / "clipped_buildings.city.json"

            success = clip_and_merge_cityjson(
                cj_paths=target_json,
                aoi_gpkg_path=str(gpkg_aoi_path),
                output_cj_path=clipped_buildings_path,
                lod=bag_lod
            )

            if success:
                final_obj_path = Path(output_dir) / "buildings_obj.obj"
                cityjson_to_obj(clipped_buildings_path, final_obj_path, translation)
            else:
                logging.warning(f"Clip and merge process failed.")

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
