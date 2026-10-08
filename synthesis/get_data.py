import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import geopandas as gpd
import time

from CFTree.config import get_config, setup_logger
from CFTree.download_geotiles import download_tile
from clip_and_merge import clip_and_merge_pointclouds

# ---------------------------------------------------------------
# ----------------------- Tile worker ---------------------------
# ---------------------------------------------------------------
def process_tile(tile_id: str, output_dir: Path, base_url: str, overwrite: bool) -> dict:
    """Download raw tile"""
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

# ---------------------------------------------------------------
# ----------------- Process the GeoJSON file --------------------
# ---------------------------------------------------------------
def process_geojson(
        geojson_path: str | Path,
        buffer_distance: float = 20.0,
        overwrite: bool = False,
        log_level: str = "INFO",
):
    """Run get_data pipeline for a specific GeoJSON area of interest."""
    geojson_path = Path(geojson_path).resolve()
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

    case = geojson_path.stem

    cfg = get_config(case_name=case)
    n_cores = cfg.get("default_cores", 1)

    setup_logger(case, "get_data", log_level)

    logging.info(f"Starting get_data for case: {case}")
    logging.info(f"Input GeoJSON: {geojson_path}")
    logging.info(f"Parallel workers: {n_cores}")
    logging.info(f"Buffer distance: {buffer_distance} m")

    # ---------------------------------------------------
    # -------------- Define directories -----------------
    # ---------------------------------------------------
    root_dir = Path(__file__).resolve().parent.parent
    cftree_dir = root_dir / "CFTree"
    resources_dir = cftree_dir / "resources"

    output_dir = geojson_path.parent
    tiles_dir = root_dir / "DATA" / "OUT"

    buffered_aoi_path = output_dir / f"{case}_buffered.geojson"
    gpkg_aoi_path = output_dir / f"{case}_aoi.gpkg"

    # ---------------------------------------------------
    # ------------- Load and buffer AOI -----------------
    # ---------------------------------------------------
    logging.info(f"Loading AOI from {geojson_path}")
    logging.info(f"CRS: {cfg['crs']}")
    aoi = gpd.read_file(geojson_path).to_crs(cfg["crs"])
    aoi["geometry"] = aoi.buffer(buffer_distance)

    aoi.to_file(buffered_aoi_path, driver="GeoJSON")
    logging.info(f"Buffered AOI saved to {buffered_aoi_path}")

    aoi.to_file(gpkg_aoi_path, driver="GPKG")
    logging.info(f"Saved AOI as GPKG to {gpkg_aoi_path}")

    # ---------------------------------------------------
    # ---------------- Find tile IDs --------------------
    # ---------------------------------------------------
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
    # ----------------- Download tiles ------------------
    # ---------------------------------------------------
    t_start_download = time.time()
    if n_cores > 1:
        logging.info(f"Running {len(tile_ids)} tiles in parallel using {n_cores} cores.")
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            futures = {
                pool.submit(process_tile, tid, tiles_dir, base_url, overwrite): tid
                for tid in tile_ids
            }
            for f in as_completed(futures):
                tid = futures[f]
                try:
                    result = f.result()
                    logging.info(f"[{tid}] {result['status'].upper()}")
                except Exception as e:
                    logging.warning(f"[{tid}] Exception: {e}")
    else:
        logging.info("Running serial mode.")
        for tid in tile_ids:
            result = process_tile(tid, tiles_dir, base_url, overwrite)
            logging.info(f"[{tid}] {result['status'].upper()}")
    download_time = time.time() - t_start_download

    # ---------------------------------------------------
    # ---------- Clip and merge pointclouds -------------
    # ---------------------------------------------------
    t_start_clip = time.time()
    logging.info("Starting clip and merge process...")
    try:
        clip_and_merge_pointclouds(str(tiles_dir), str(output_dir), gpkg_path=str(gpkg_aoi_path), tile_ids=tile_ids, output_filename="clipped_output.laz")
        logging.info("Clip and merge process is completed.")
    except Exception as e:
        logging.exception(f"Clip and merge process failed. Error: {e}")
    clip_time = time.time() - t_start_clip
    logging.info(f"Completed get_data for case: {case}")

    return download_time, clip_time


# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_data.py case_name")
        sys.exit(1)

    input_geojson = sys.argv[1]
    process_geojson(input_geojson)
