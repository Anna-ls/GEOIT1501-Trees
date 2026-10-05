# tree4cfd — Architecture

> Created 2026-06-18. The `.github` instructions require reading this file before
> any code change; it did not previously exist, so it was created alongside the
> OSM building-filter feature and should be kept current.

## Purpose

Convert city LiDAR point clouds into individual tree meshes (crown + trunk) for
CFD, writing one combined `.obj` per tile. Driven entirely by a JSON config.

## Data layout (local, gitignored)

```
DATA/
├── IN/
│   ├── Barcelona/   # ICGC LiDAR, EPSG:25831, has NIR (NDVI available)
│   ├── Paris/       # IGN LiDAR HD, EPSG:2154, no NIR (class-based only)
│   ├── Rotterdam/
│   └── tud/
├── OUT/             # <city>/ subfolders with .obj + offset.txt + <tile>_trees.csv
├── inventory/       # universal inventory CSVs (paris_trees.csv, ...)
├── inventory_cache/ # cached raw inventory fetches (ODS json, CKAN csv)
└── osm_cache/       # cached Overpass responses (per tile bbox)
```

CRS is read from each LAS header (`header.parse_crs()`), not hard-coded.

## Database schema

N/A — this project has no database. Inputs are LAS/LAZ files; outputs are `.obj`
meshes plus an `offset.txt`. State is purely the on-disk cache in `DATA/osm_cache`.

## Module map

| Module            | Responsibility                                                    |
| ----------------- | ---------------------------------------------------------------- |
| `config.py`       | JSON → typed dataclasses (`Paths`, `Filtering`, `Cleaning`, `Segmentation`, `Trunk`, `Crown`, `Buildings`). |
| `io.py`           | LAS read, vegetation/ground filtering, OBJ + offset writing.     |
| `cleaning.py`     | Statistical outlier removal + voxel downsample.                  |
| `segmentation.py` | CHM (DSM−DTM) tree segmentation → per-point labels + DTM.        |
| `shapes.py`       | Per-tree geometric stats + pole/wall filter: drop tall, horizontally narrow segments (lamp posts, facade slivers). |
| `buildings.py`    | OSM Overpass footprint fetch (cached) + drop trees whose centroid sits inside a buffered footprint. |
| `inventory.py`    | Two stages: `extract_inventory` writes a universal CSV (`id,lon,lat,common_name,species,height_m,dbh_cm,age_years,category`) from a city source (Paris OpenDataSoft / Barcelona CKAN); `load_inventory_csv` + `annotate_and_write` match segmented trees to it per tile. Private CSVs in the same schema work too. |
| `trunk.py`        | DTM sampling, allometric trunk params (or inventory DBH; `force_xy` pins the base to an inventory position), trunk cone mesh; rejects implausibly tall trees / over-long trunks (floating roof blobs). |
| `crown.py`        | Marching-cubes crown mesh (LoD 3); prunes small disconnected blobs (stray points). |
| `primitives.py`   | Parametric tree solids for coarse LoD: LoD 1 enclosing block; LoD 2 species-driven crown (ellipsoid/cone/cylinder) + trunk. |
| `inspect.py`      | Step-1 classification-code profiling.                            |
| `pipeline.py`     | Per-tile orchestration.                                          |
| `cli.py`          | `inspect` / `inventory` / `run` subcommands.                     |

## Per-tile pipeline (`pipeline.process_tile`)

1. **Load** tile; read CRS (EPSG) from header.
2. **Filter** high-vegetation + ground points (`io.filter_vegetation/ground`).
3. **Clean** vegetation (SOR + voxel downsample).
4. **Translate** all coordinates by `config.translation` to a local origin.
5. **Segment** trees via CHM → per-point `labels`, `dtm`, georef. Then, if
   `influence_region` > 0, drop trees whose centroid is beyond that radius of
   `point_of_interest` (and tiles fully outside the radius are skipped upfront).
6. **Shape filter** (`shapes.filter_pole_wall_trees`, optional): zero out labels of
   pole-like / wall-like segments using per-tree geometry (vertical extent,
   horizontal extents, mid-height-slice spans). Translation-invariant.
7. **Building filter** (`buildings.filter_trees_in_buildings`): zero out labels of
   trees whose horizontal centroid falls inside an OSM footprint buffered by
   `buildings.buffer` metres. Geometry tests run in the projected CRS so the
   buffer is a true metric distance.
8. **Inventory match** (`inventory.annotate_and_write`, optional): the universal
   CSV is loaded once in `run_pipeline`; per tile, its lon/lat are reprojected to
   the tile CRS and each surviving tree matched to the nearest entry within
   `inventory.match_dist`, writing `<tile>_trees.csv`. (The CSV is produced
   separately by `tree4cfd inventory`, or supplied by the user.)
9. **Mesh** each surviving tree (crown + trunk). Inventory trees connecting to a
   crown (footprint + `connect_buffer`) **split** it into one instance per tree —
   points partitioned by nearest snapped inventory seed, each meshed as its own
   crown + trunk (inventory DBH where available). Crowns with no connected
   inventory stay whole with the LiDAR-estimated trunk. The crown (canopy only,
   crown base → top) follows `lod` base 1/2/3 = block / species primitive
   (ellipsoid/cone/cylinder) / marching-cubes hull; a `.1` suffix adds the trunk
   (ground → canopy, top extended `trunk.crown_overlap` into the crown, no gap).
   Files: `<tile>_lod<level>.obj`. Concatenate.
10. **Write** one combined `.obj` + `offset.txt`.

## Coordinate handling

Points are shifted to a local origin by subtracting `translation` (x, y, z).
To test against OSM the centroids are shifted back (`+translation[:2]`) into the
projected CRS; the tile bbox is reprojected to lon/lat for the Overpass query,
and the returned footprints are reprojected back to the projected CRS via
`pyproj`. OBJ coordinates stay in local space; `offset.txt` records how to
recover the original projected coordinates.

## External dependencies

`laspy[laszip]`, `numpy`, `scipy`, `scikit-image` (core); `shapely`, `pyproj`,
`requests` (building filter / OSM). Tests use stdlib `unittest`. Runtime env:
conda `wrappy2` (the `base` env lacks `laspy`).
