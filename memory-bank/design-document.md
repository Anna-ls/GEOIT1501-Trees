# tree4cfd — Design Document

> Created 2026-06-18 alongside the OSM building-filter feature.

## Goal

Refactor of `barcelona_wrap_voxel_Li21.ipynb` into a configurable, testable
package that turns LiDAR tiles into per-tree CFD meshes. No visualization; all
inputs via one JSON config per city.

## Design principles (from `.github` instructions)

- Prefer simple solutions; keep modules under ~200–300 lines.
- DRY: reuse existing helpers; don't introduce parallel patterns.
- Each pipeline stage is a small module over plain NumPy `(N, 3)` arrays.
- Mock data only in tests, never in dev/prod.
- One tile in memory at a time; `del` between heavy stages.

## OSM building-filter feature

**Motivation.** CHM segmentation occasionally promotes rooftop greenery, facade
planting, or misclassified roof structure into "trees". Such artefacts sit on or
inside building footprints. Removing them yields a cleaner CFD tree set.

**Approach.**
1. After segmentation, compute each tree's horizontal centroid (mean x, y of its
   points) in the projected CRS.
2. Fetch building footprints for the tile's bbox from the OpenStreetMap Overpass
   API (`way["building"]` + `relation["building"]`, `out geom`). Responses are
   cached on disk keyed by rounded bbox so reruns are offline and reproducible.
3. Reproject footprints to the tile's projected CRS; buffer them outward by
   `buildings.buffer` metres (default 0.5).
4. Drop any tree whose centroid intersects a buffered footprint (label → 0), then
   mesh only the survivors.

**Why Overpass + shapely + pyproj** (not osmnx/geopandas): a lighter dependency
footprint for the PyPI goal, and the network/geometry logic stays small and
unit-testable. The geometry predicate (`buildings.trees_inside_buildings`) is a
pure function tested with mock polygons; the network fetch is isolated.

**Config.** New `[buildings]` section: `enabled`, `buffer`, `cache_dir`,
`overpass_url`. CRS is taken from the LAS header, not configured.

## Per-city configs

`config_barcelona.json`, `config_paris.json`: high-vegetation class 5 only,
city-appropriate `translation` (tile min corner), `DATA/IN/<city>` →
`DATA/OUT/<city>`, building filter enabled with 0.5 m buffer.

## Testing

`tests/test_buildings.py` (stdlib `unittest`): centroid-in-footprint, buffer
boundary, multi-tree label removal, and Overpass-JSON ring parsing — all with
mock geometry, no network.
