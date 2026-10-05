"""Configuration loading and validation.

All pipeline inputs (paths and parameters) live in a single JSON file. This
module maps that JSON onto typed dataclasses so the rest of the package never
touches raw dictionaries and gets clear errors for missing or misspelled keys.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, get_type_hints


@dataclass
class Paths:
    """Input/output locations."""
    tiles_dir: Path = Path("bcn_new_pointcloud")
    output_dir: Path = Path("bcn_canopy_objs_Li")

    def __post_init__(self) -> None:
        self.tiles_dir = Path(self.tiles_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class Filtering:
    """High-vegetation selection.

    ``high_veg_classes`` lists ASPRS classification codes to treat as high
    vegetation (5 = High Vegetation in the standard scheme). Leave it empty to
    fall back to an NDVI threshold instead.
    """
    high_veg_classes: List[int] = field(default_factory=lambda: [5])
    ground_classes: List[int] = field(default_factory=lambda: [2])
    ndvi_threshold: float = 0.2
    multi_return_only: bool = True


@dataclass
class Cleaning:
    """Point-cloud cleaning applied to vegetation points only."""
    sor_neighbors: int = 20       # 0 disables statistical outlier removal
    sor_std_ratio: float = 1.5
    voxel_size: float = 1.0       # 0 disables voxel downsampling


@dataclass
class Segmentation:
    """CHM-based individual-tree segmentation."""
    cell_size: float = 0.5        # m — DTM/DSM/CHM grid resolution
    smooth_sigma: float = 2.0     # grid cells — Gaussian smooth before peaks
    min_height: float = 3.0       # m — excludes low vegetation / rooftop shrubs
    peak_min_dist: float = 5.0    # m — minimum spacing between tree tops
    min_tree_pts: int = 150       # discard segments with fewer points


@dataclass
class Trunk:
    """Allometric trunk-cone model.

    DBH = ``allom_a`` * H ** ``allom_b`` (H and DBH in metres). Defaults are a
    generic mixed-species fit; replace with species-specific values when an
    inventory is available.
    """
    q_base: float = 0.20          # crown-z quantile for trunk-base centroid
    q_crown: float = 0.10         # crown-z quantile used as crown-base elevation
    allom_a: float = 0.03
    allom_b: float = 1.2
    taper: float = 0.4            # r_top / r_base
    n_sides: int = 12             # polygon sides per trunk cone
    # Sanity limits (m): drop a tree whose total height above ground exceeds
    # max_total_height, or whose trunk (ground to crown base) exceeds
    # max_trunk_height — these flag floating roof blobs / DTM errors. 0 disables.
    max_total_height: float = 0.0
    max_trunk_height: float = 0.0
    # Extend the trunk top this far into the crown mesh so the cone connects to
    # the canopy with no vertical gap.
    crown_overlap: float = 0.5


@dataclass
class Crown:
    """Per-tree crown meshing (marching cubes)."""
    voxel_size: float = 0.75
    sigma: float = 1.5
    iso_level: float = 0.15
    # Drop crown mesh components smaller than this fraction of the largest one,
    # removing floating blobs from stray points. 0 disables.
    min_component_frac: float = 0.1


@dataclass
class ShapeFilter:
    """Drop pole-like and wall-like segments by their shape.

    Removes thin vertical structures (lamp posts, sign poles, facade slivers)
    that CHM segmentation can mistake for trees. A segment is a *pole* if it is
    tall, horizontally compact, and narrow through its mid-height slices; a
    *wall* if it is tall, thin in one horizontal axis, and taller than it is
    wide (low aspect). Disabled by default — tune the point-count limit for
    your point density (defaults assume voxel-downsampled clouds).
    """
    enabled: bool = False
    pole_z_range_min: float = 3.0     # m — min vertical extent for a pole
    pole_crown_diam_max: float = 3.0  # m — max mean horizontal extent
    pole_mid_span_max: float = 1.0    # m — max median mid-slice span
    pole_max_points: int = 200        # safety cap so big narrow trees survive
    wall_z_range_min: float = 3.0     # m — min vertical extent for a wall
    wall_min_xy_max: float = 1.5      # m — max of the thinner horizontal axis
    wall_mid_min_span_max: float = 0.5  # m — max median mid-slice thin span
    wall_aspect_max: float = 0.5      # crown_diam / z_range below this = wall


@dataclass
class Inventory:
    """Match segmented trees to a tree inventory via a universal CSV.

    Decoupled into two stages. ``tree4cfd inventory`` fetches a city's inventory
    (``source``) and writes the universal CSV at ``csv_path`` (schema: ``id, lon,
    lat, common_name, species, height_m, dbh_cm, age_years, category``; lon/lat
    required). ``tree4cfd run`` then loads ``csv_path``, matches each segmented
    tree to the nearest entry within ``match_dist`` metres, and writes a per-tile
    ``<tile>_trees.csv``. Supply your own ``csv_path`` (private data) to skip the
    fetch entirely — any file in that schema works.
    """
    enabled: bool = False
    csv_path: str = ""            # universal inventory CSV (read by `run`)
    match_dist: float = 3.0       # m — max tree-centroid ↔ inventory distance
    connect_buffer: float = 2.0   # m — inventory tree within this of a crown's
    #                               footprint places a trunk there (drives trunks)
    # --- extraction (`tree4cfd inventory`) ---
    source: str = ""             # "paris_ods" | "barcelona_ckan" | "" (private CSV)
    cache_dir: str = "DATA/inventory_cache"
    # Paris OpenDataSoft
    ods_base: str = "https://opendata.paris.fr/api/explore/v2.1"
    ods_dataset: str = "les-arbres"
    # Barcelona CKAN (unioned over all listed datasets)
    ckan_base: str = "https://opendata-ajuntament.barcelona.cat"
    ckan_datasets: List[str] = field(
        default_factory=lambda: ["arbrat-viari", "arbrat-zona", "arbrat-parcs"]
    )


@dataclass
class Buildings:
    """Drop trees whose centroid falls inside an OSM building footprint.

    Footprints are fetched from the OpenStreetMap Overpass API per tile and
    cached on disk. ``buffer`` dilates each footprint (metres) before testing,
    so trees within that distance of a building are also removed.
    """
    enabled: bool = True
    buffer: float = 0.5
    cache_dir: str = "DATA/osm_cache"
    overpass_url: str = "https://overpass-api.de/api/interpreter"


@dataclass
class Output:
    """How meshes are written out."""
    merge_tiles: bool = False          # combine all tiles into one .obj (per LoD)
    separate_crown_trunk: bool = False  # write crown and trunk as separate .obj files


@dataclass
class Config:
    """Top-level configuration aggregating every section."""
    paths: Paths = field(default_factory=Paths)
    filtering: Filtering = field(default_factory=Filtering)
    cleaning: Cleaning = field(default_factory=Cleaning)
    segmentation: Segmentation = field(default_factory=Segmentation)
    trunk: Trunk = field(default_factory=Trunk)
    crown: Crown = field(default_factory=Crown)
    shape_filter: ShapeFilter = field(default_factory=ShapeFilter)
    buildings: Buildings = field(default_factory=Buildings)
    inventory: Inventory = field(default_factory=Inventory)
    output: Output = field(default_factory=Output)
    # Level of detail. Integer part = crown shape: 1 = block, 2 = species-driven
    # primitive, 3 = marching-cubes hull. Each crown spans the canopy (crown base
    # → top), not the ground. A ``.1`` suffix adds the trunk down to the ground:
    # 1, 1.1, 2, 2.1, 3, 3.1.
    lod: float = 3.1
    # Region of interest: keep only trees whose centroid is within
    # ``influence_region`` metres (horizontal) of ``point_of_interest`` (x, y, z
    # in the tile's projected CRS). ``influence_region`` = 0 disables it.
    point_of_interest: Optional[Tuple[float, float, float]] = None
    influence_region: float = 0.0
    # Translation subtracted from all output coordinates (UTM easting,
    # northing, z). Recover original UTM by adding it back.
    translation: Tuple[float, float, float] = (429692.0, 4582890.0, 0.0)

    def __post_init__(self) -> None:
        self.translation = tuple(float(v) for v in self.translation)
        self.lod = float(self.lod)
        if self.point_of_interest is not None:
            self.point_of_interest = tuple(float(v) for v in self.point_of_interest)


def _build(cls: type, data: Any):
    """Instantiate dataclass ``cls`` from ``data``, validating unknown keys."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object for '{cls.__name__}', got {type(data).__name__}")

    # Resolve string annotations (PEP 563) to real types for nested dataclasses.
    hints = get_type_hints(cls)
    unknown = set(data) - set(hints)
    if unknown:
        raise ValueError(
            f"Unknown key(s) for '{cls.__name__}': {sorted(unknown)}. "
            f"Allowed: {sorted(hints)}"
        )

    kwargs = {}
    for name, value in data.items():
        ftype = hints[name]
        kwargs[name] = _build(ftype, value) if is_dataclass(ftype) else value
    return cls(**kwargs)


_COMMENT_RE = re.compile(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/', re.DOTALL)


def _strip_json_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments, preserving them inside strings."""
    return _COMMENT_RE.sub(lambda m: m.group(1) or "", text)


def load_config(path: str | Path) -> Config:
    """Load and validate a JSONC config file into a :class:`Config`.

    ``//`` and ``/* */`` comments are allowed (stripped before parsing). Any
    section omitted from the file falls back to its documented defaults.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = json.loads(_strip_json_comments(path.read_text()))
    return _build(Config, data)
