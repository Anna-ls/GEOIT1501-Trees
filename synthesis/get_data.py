import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import geopandas as gpd
from pathlib import Path

from CFTree.config import get_config, setup_logger
from CFTree.download_geotiles import download_tile
from clip_and_merge import clip_and_merge_pointclouds

# Tile worker (must be top-level for multiprocessing)
def process_tile(tile_id: str, output_dir: Path, base_url: str, overwrite: bool) -> dict:
    """Download raw tile"""
    try:
        # 1. Download raw tile
        result_dl = download_tile(tile_id, output_dir, base_url, overwrite=overwrite)
        laz_path = result_dl.get("paths", {}).get("laz")

        if result_dl["status"] != "ok" or not laz_path or not Path(laz_path).exists():
            return {"tile_id": tile_id, "status": "download_failed"}

        # 2. Return tile summary
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


# Runner main
def main():
    parser = argparse.ArgumentParser(description="Run get_data pipeline for a case.")
    parser.add_argument("--case", type=str, help="Case name (default from config)")
    parser.add_argument("--n-cores", type=int, default=None, help="Number of parallel workers (default from config)")
    parser.add_argument("--overwrite", action="store_true", help="Re-download tiles if they exist")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    parser.add_argument("--dry-run", action="store_true", help="Only list tiles to be processed")
    parser.add_argument("--buffer", type=float, default=20.0, help="Buffer in meters around AOI")
    args = parser.parse_args()

    # Load configuration
    cfg = get_config(case_name=args.case, n_cores=args.n_cores)
    case = cfg["case"]
    n_cores = cfg["default_cores"]

    setup_logger(case, "get_data", args.log_level)

    logging.info(f"Starting get_data for case: {case}")
    logging.info(f"Parallel workers: {n_cores} (from {'CLI' if args.n_cores else 'config'})")
    logging.info(f"Buffer distance: {args.buffer} m")

    root_dir = Path(__file__).resolve().parent.parent
    cftree_dir = root_dir / "CFTree"
    case_dir = cftree_dir / "cases" / case

    aoi_path = case_dir / "case_area.geojson"
    buffered_aoi_path = case_dir / "case_area_buffered.geojson"
    resources_dir = cftree_dir / "resources"
    output_dir = case_dir

    # Step 1: Load and buffer AOI
    logging.info(f"Loading AOI from {aoi_path}")
    logging.info(f"CRS: {cfg['crs']}")
    aoi = gpd.read_file(aoi_path).to_crs(cfg["crs"])
    aoi["geometry"] = aoi.buffer(args.buffer)
    aoi.to_file(buffered_aoi_path, driver="GeoJSON")
    logging.info(f"Buffered AOI saved to {buffered_aoi_path}")

    # Save the buffered AOI as a GPKG into the output directory.
    gpkg_aoi_path = output_dir / f"{case}_aoi.gpkg"
    aoi.to_file(gpkg_aoi_path, driver="GPKG")
    logging.info(f"Saved AOI as GPKG to {gpkg_aoi_path}")

    # Step 2: Determine intersecting tiles
    tiles = gpd.read_file(resources_dir / "AHN_subunits_GeoTiles" / "AHN_subunits_GeoTiles.shp").to_crs(cfg["crs"])
    intersecting = tiles[tiles.intersects(aoi.union_all())]
    tile_ids = intersecting["GT_AHNSUB"].unique().tolist()

    if not tile_ids:
        logging.info("No intersecting tiles found.")
        return

    logging.info(f"Found {len(tile_ids)} intersecting tiles: {tile_ids if len(tile_ids) <= 10 else '...'}")

    if args.dry_run:
        logging.info("[DRY RUN] Exiting before downloads.")
        return

    # Step 3: Per-tile pipeline (download)
    base_url = "https://geotiles.citg.tudelft.nl/AHN5_T"

    if n_cores > 1:
        logging.info(f"Running {len(tile_ids)} tiles in parallel using {n_cores} cores.")
        with ProcessPoolExecutor(max_workers=n_cores) as pool:
            futures = {
                pool.submit(process_tile, tid, output_dir, base_url, args.overwrite): tid
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
            result = process_tile(tid, output_dir, base_url, args.overwrite)
            logging.info(f"[{tid}] {result['status'].upper()}")

    # Step 4: Clip and merge
    logging.info("Starting clip and merge process...")
    try:
        clip_and_merge_pointclouds(str(output_dir), "clipped_output.laz")
        logging.info("Clip and merge process is completed.")
    except Exception as e:
        logging.exception(f"Clip and merge process failed. Error: {e}")

    logging.info(f"Completed get_data for case: {case}")

# Entry point
if __name__ == "__main__":
    main()
