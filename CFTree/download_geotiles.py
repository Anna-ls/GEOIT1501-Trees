# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/download_geotiles.py

import subprocess
import logging
from pathlib import Path
import urllib.request
import urllib.error
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def download_tile(tile_id: str, output_dir: Path, base_url: str, overwrite: bool = False) -> dict:
    """Download LAZ and LAX for one tile."""
    tile_folder = output_dir / "tiles" / tile_id
    tile_folder.mkdir(parents=True, exist_ok=True)

    tile_folder = output_dir / "tiles" / tile_id
    tile_folder.mkdir(parents=True, exist_ok=True)

    laz_url = f"{base_url}/{tile_id}.LAZ"
    lax_url = f"{base_url}/{tile_id}.LAX"
    laz_path = tile_folder / "raw.laz"
    lax_path = tile_folder / "raw.lax"

    try:
        if overwrite or not laz_path.exists():
            logging.info(f"[{tile_id}] Downloading LAZ")
            urllib.request.urlretrieve(laz_url, str(laz_path))
        else:
            logging.info(f"[{tile_id}] Skipping existing LAZ")

        if overwrite or not lax_path.exists():
            logging.info(f"[{tile_id}] Downloading LAX")
            urllib.request.urlretrieve(lax_url, str(lax_path))
        else:
            logging.info(f"[{tile_id}] Skipping existing LAX")

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
