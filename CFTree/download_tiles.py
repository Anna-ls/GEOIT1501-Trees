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
    laz_path = tile_folder / "raw.laz"

    if "AHN6" in base_url:
        ahn_versions = ["AHN6_2025", "AHN6_2026"]
    else:
        ahn_versions = ["AHN5"]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    if not overwrite and laz_path.exists():
        logging.info(f"[{tile_id}] Skipping existing LAZ")
        return {
            "tile_id": tile_id,
            "status": "ok",
            "paths": {"laz": laz_path},
        }

    for version in ahn_versions:
        laz_url = f"{base_url}/{version}_C_{tile_id}.COPC.LAZ"
        try:
            logging.info(f"[{tile_id}] Attempting download: {laz_url}")
            req = urllib.request.Request(laz_url, headers=headers)

            with urllib.request.urlopen(req) as response, open(str(laz_path), 'wb') as out_file:
                out_file.write(response.read())

            return {
                "tile_id": tile_id,
                "status": "ok",
                "paths": {"laz": laz_path},
            }
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            else:
                logging.warning(f"[{tile_id}] HTTP error on {version}: {e}")
        except urllib.error.URLError as e:
            logging.warning(f"[{tile_id}] Download failed on {version}: {e}")

    return {
        "tile_id": tile_id,
        "status": "failed",
        "paths": {"laz": laz_path},
    }
