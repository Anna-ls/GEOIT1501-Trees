# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

import logging
from pathlib import Path
import urllib.request
import urllib.error
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def download_tile(tile_id: str, output_dir: Path, base_url: str, overwrite: bool = False) -> dict:
    """Download LAZ for one tile."""
    tile_folder = output_dir / "tiles" / tile_id
    tile_folder.mkdir(parents=True, exist_ok=True)

    laz_url = f"{base_url}/AHN5_C_{tile_id}.COPC.LAZ"
    laz_path = tile_folder / "raw.laz"

    try:
        if overwrite or not laz_path.exists():
            logging.info(f"[{tile_id}] Downloading COPC.LAZ")
            urllib.request.urlretrieve(laz_url, str(laz_path))
        else:
            logging.info(f"[{tile_id}] Skipping existing LAZ")

        return {
            "tile_id": tile_id,
            "status": "ok",
            "paths": {"laz": laz_path},
        }

    except urllib.error.URLError as e:
        logging.warning(f"[{tile_id}] Download failed: {e}")
        return {
            "tile_id": tile_id,
            "status": "failed",
            "paths": {"laz": laz_path},
        }
