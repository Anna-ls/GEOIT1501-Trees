"""Command-line interface — drive the pipeline from a JSON config file.

Examples
--------
    python -m tree4cfd inspect   --config config.json
    python -m tree4cfd inventory --config config.json
    python -m tree4cfd run       --config config.json
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .inspect_laz import print_class_table, profile_classes
from .inventory import extract_inventory
from .io import find_tiles
from .pipeline import run_pipeline


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "-c", "--config", default="config.json",
        help="Path to the JSON config file (default: config.json)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tree4cfd",
        description="LiDAR point clouds to individual tree meshes for CFD.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser(
        "inspect", help="Profile classification codes across tiles (Step 1)."
    )
    _add_config_arg(p_inspect)

    p_inv = sub.add_parser(
        "inventory",
        help="Extract the tree inventory for the tiles' area to a universal CSV.",
    )
    _add_config_arg(p_inv)

    p_run = sub.add_parser(
        "run", help="Run the full per-tile meshing pipeline."
    )
    _add_config_arg(p_run)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "inspect":
        tiles = find_tiles(cfg.paths.tiles_dir)
        print(f"Found {len(tiles)} tiles")
        if not tiles:
            return 1
        print_class_table(profile_classes(tiles))
    elif args.command == "inventory":
        path, n = extract_inventory(cfg)
        print(f"Wrote {n:,} inventory trees → {path}")
    elif args.command == "run":
        run_pipeline(cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
