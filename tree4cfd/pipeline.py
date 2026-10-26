"""Per-tile orchestration: filter → clean → segment → mesh → write .obj."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import NamedTuple
import geopandas as gpd
from shapely.geometry import shape, box
import math
root_dir = Path(__file__).resolve().parent.parent

import numpy as np
import logging

from .buildings import filter_trees_in_buildings
from synthesis.building_filtering_bag import filter_trees_in_buildings_bag
from .cleaning import remove_outliers_sor, voxel_downsample
from .config import Config
from .crown import kept_component_mask, segment_to_marching_cubes
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from synthesis.determine_lcz_of_area import determine_lcz_of_area
from scipy.ndimage import distance_transform_edt
from shapely.affinity import translate as translate_geometry
import shapely
from synthesis.cgal_wrap import wrap_and_simplify

from .inventory import (
    build_inventory_index,
    crown_footprint,
    crown_inventory_indices,
    snap_under_crown,
)
from .io import (
    filter_ground,
    filter_vegetation,
    find_tiles,
    read_header_bounds,
    read_las,
    write_obj,
    write_obj_grouped,
    write_offset,
)
from .primitives import block_mesh, crown_primitive, crown_shape_for_species
from .segmentation import segment_trees_chm
from .trunk import estimate_trunk_params, make_trunk_mesh

def get_outlines(tiles):
    if not tiles:
        logging.warning("No tiles provided to calculate outlines")

    global_min_x, global_min_y = float('inf'), float('inf')
    global_max_x, global_max_y = float('-inf'), float('-inf')
    tile_epsg = None

    for tile in tiles:
        bounds, epsg = read_header_bounds(tile)
        minx, miny, maxx, maxy = bounds

        if epsg and tile_epsg is None:
            tile_epsg = epsg

        if minx < global_min_x:
            global_min_x = minx
        if miny < global_min_y:
            global_min_y = miny
        if maxx > global_max_x:
            global_max_x = maxx
        if maxy > global_max_y:
            global_max_y = maxy

    bounding_box = box(global_min_x, global_min_y, global_max_x, global_max_y)

    if not tile_epsg:
        tile_epsg = 28992

    gdf = gpd.GeoDataFrame([{"geometry": bounding_box}], crs=f"EPSG:{tile_epsg}")
    bounding_box_4326 = gdf.to_crs("EPSG:4326").geometry.iloc[0]

    return {"geometry": bounding_box_4326}

def split_polygon(polygon, cell_size=500):
    if isinstance(polygon, (str, Path)):
        gdf = gpd.read_file(polygon)
        geom = gdf.geometry.iloc[0]
        out_dir = Path(polygon).parent
    else:
        geom = polygon["geometry"]
        if not isinstance(geom, shapely.geometry.base.BaseGeometry):
            geom = shape(geom)
        out_dir = Path(f"{root_dir}/DATA/OUT")

    gdf = gpd.GeoDataFrame(
        [{"geometry": geom}],
        crs="EPSG:4326"
    ).to_crs("EPSG:28992")

    geom = gdf.geometry.iloc[0]
    minx, miny, maxx, maxy = geom.bounds

    width = maxx - minx
    height = maxy - miny

    # Number of cells needed in each direction
    x_boxes = max(1, math.ceil(width / cell_size))
    y_boxes = max(1, math.ceil(height / cell_size))

    # Stretch cells evenly to fit the bounding box
    cell_width = width / x_boxes
    cell_height = height / y_boxes

    pieces = []

    for i in range(x_boxes):
        for j in range(y_boxes):
            x1 = minx + i * cell_width
            y1 = miny + j * cell_height

            cell = box(
                x1,
                y1,
                minx + (i + 1) * cell_width,
                miny + (j + 1) * cell_height
            )

            intersection = geom.intersection(cell)

            for poly in extract_polygons(intersection):
                if poly.area > 0:
                    pieces.append({"geometry": poly})

    result = gpd.GeoDataFrame(
        pieces,
        geometry="geometry",
        crs=gdf.crs
    )

    result.to_file(
        Path(f"{out_dir}/splitted_polygon.gpkg"),
        driver="GPKG"
    )

    return result


def combine_tile_results(tile_results, n_points):
    """Combine polygon segmentation results while preserving point order."""

    if not tile_results:
        raise ValueError("tile_results is empty")

    cell_size = tile_results[0]["georef"][2]

    for result in tile_results:
        if not np.isclose(result["georef"][2], cell_size):
            raise ValueError("All DTM tiles must use the same cell size")

    # Global DTM extent.
    x_min = min(r["georef"][0] for r in tile_results)
    y_min = min(r["georef"][1] for r in tile_results)

    x_max = max(
        r["georef"][0] + (r["dtm"].shape[1] - 1) * cell_size
        for r in tile_results
    )
    y_max = max(
        r["georef"][1] + (r["dtm"].shape[0] - 1) * cell_size
        for r in tile_results
    )

    nx = int(round((x_max - x_min) / cell_size)) + 1
    ny = int(round((y_max - y_min) / cell_size)) + 1

    dtm_sum = np.zeros((ny, nx), dtype=np.float64)
    dtm_count = np.zeros((ny, nx), dtype=np.uint32)

    # Crucial: labels are indexed by the original point-cloud order.
    labels_all = np.zeros(n_points, dtype=np.int32)
    next_label = 1

    for result in tile_results:
        local_labels = np.asarray(
            result["labels"], dtype=np.int32
        ).ravel()

        point_indices = np.asarray(
            result["point_indices"], dtype=np.int64
        ).ravel()

        if len(local_labels) != len(point_indices):
            raise ValueError(
                "Label/index mismatch: "
                f"{len(local_labels)} labels vs "
                f"{len(point_indices)} point indices"
            )

        if np.any(point_indices < 0) or np.any(point_indices >= n_points):
            raise ValueError("Point index out of bounds")

        # Assign globally unique IDs to positive labels.
        positive = local_labels > 0
        if positive.any():
            local_labels = local_labels.copy()
            local_labels[positive] += next_label - 1
            next_label = int(local_labels.max()) + 1

        # Restore labels to their original vegetation point positions.
        labels_all[point_indices] = local_labels

        # Mosaic this polygon's DTM.
        dtm = np.asarray(result["dtm"], dtype=np.float64)
        tx, ty, _ = result["georef"]

        col = int(round((tx - x_min) / cell_size))
        row = int(round((ty - y_min) / cell_size))

        h, w = dtm.shape
        target = (slice(row, row + h), slice(col, col + w))

        valid = np.isfinite(dtm)
        dtm_sum[target] += np.where(valid, dtm, 0.0)
        dtm_count[target] += valid.astype(np.uint32)

    # Average overlapping valid DTM cells.
    dtm_all = np.full((ny, nx), np.nan, dtype=np.float64)
    valid = dtm_count > 0
    dtm_all[valid] = dtm_sum[valid] / dtm_count[valid]

    # Fill gaps with nearest valid terrain values.
    if valid.any() and (~valid).any():
        nearest = distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        nearest_dtm = dtm_all[tuple(nearest)]
        dtm_all[~valid] = nearest_dtm[~valid]

    return labels_all, dtm_all, (x_min, y_min, cell_size)


def filter_points_by_polygon(pts, polygon):
    point_geometries = shapely.points(pts[:, 0], pts[:, 1])
    mask = shapely.contains(polygon, point_geometries)
    return pts[mask]

def extract_polygons(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [
            p for part in geom.geoms
            for p in extract_polygons(part)
        ]
    return []


def _tile_name(tile: Path) -> str:
    """Derive a short tile name (the embedded 6-digit id, else the stem)."""
    m = re.search(r"(\d{6})", tile.stem)
    return m.group(1) if m else tile.stem


def _trunk_for(tree_pts, dtm, georef, cfg: Config, dbh_m=None, force_xy=None):
    """Estimate trunk params for one crown (optionally pinned to ``force_xy``)."""
    gx, gy, gcs = georef
    return estimate_trunk_params(
        tree_pts, dtm, gx, gy, gcs,
        q_base=cfg.trunk.q_base, q_cb=cfg.trunk.q_crown,
        allom_a=cfg.trunk.allom_a, allom_b=cfg.trunk.allom_b,
        taper_ratio=cfg.trunk.taper, dbh_override_m=dbh_m,
        max_trunk_height=cfg.trunk.max_trunk_height,
        max_total_height=cfg.trunk.max_total_height, force_xy=force_xy,
    )


def parse_lod(lod):
    """Split a LoD value into ``(base, with_trunk, label)``.

    Integer part is the crown level (1/2/3); a ``.1`` fractional part adds the
    trunk. e.g. ``2.1`` → ``(2, True, "2.1")``; ``3`` → ``(3, False, "3")``.
    """
    base = int(float(lod))
    with_trunk = (float(lod) - base) > 0.05
    return base, with_trunk, (f"{base}.1" if with_trunk else f"{base}")


def _connected_trunk_top(z_crown_base, crown_min_z, overlap):
    """Raise the trunk top to reach into the crown mesh, closing any vertical gap.

    The crown mesh bottom (``crown_min_z``) can sit above the points' 10th
    percentile (``z_crown_base``); extend the trunk to ``overlap`` m inside it.
    """
    return max(z_crown_base, crown_min_z + overlap)


class TileMeshes(NamedTuple):
    """Crown and trunk geometry for a tile, as separate watertight groups."""
    crown_v: np.ndarray
    crown_f: np.ndarray
    trunk_v: np.ndarray
    trunk_f: np.ndarray
    crown_groups: list
    trunk_groups: list
    n_crowns: int
    n_trunks: int

    def is_empty(self) -> bool:
        return len(self.crown_v) == 0 and len(self.trunk_v) == 0

    def combined(self):
        """Single ``(verts, faces)`` with crown then trunk (offset applied)."""
        vs, fs, off = [], [], 0
        for v, f in ((self.crown_v, self.crown_f), (self.trunk_v, self.trunk_f)):
            if len(v):
                vs.append(v)
                fs.append(f + off)
                off += len(v)
        if not vs:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        return np.vstack(vs), np.vstack(fs)


def _build_tile_meshes(pts_veg, labels, dtm, georef, cfg: Config,
                       inv_local=None, inv_recs=None):
    """Crown mesh + trunk cones for every labelled tree; concatenate them.

    Trunk placement prefers the inventory. If inventory trees connect to a crown
    (inside its footprint + ``inventory.connect_buffer``), the crown is **split**
    into one separate instance per connected tree — its points partitioned by
    nearest inventory seed (snapped under the canopy) — each instance meshed as
    its own crown + trunk, sized by inventory DBH where available. Crowns with no
    connected inventory are kept whole with the LiDAR-estimated trunk (allometric
    size). Any instance that yields no valid trunk (e.g. a floating blob) is
    dropped entirely.
    """
    unique_lbls = np.unique(labels[labels > 0])
    strtree = build_inventory_index(inv_local)
    inv_recs = inv_recs or []
    claimed: set = set()

    crown_v, crown_f, trunk_v, trunk_f = [], [], [], []
    crown_groups, trunk_groups = [], []
    cof = tof = 0  # separate vertex offsets for the crown and trunk groups
    n_crowns = n_trunks = 0

    def _add_crown(tree_id, v, f):
        nonlocal cof
        crown_v.append(v)
        crown_f.append(f + cof)
        crown_groups.append((f"tree_{tree_id}_crown", f + cof))
        cof += len(v)

    def _add_trunk(tree_id, v, f):
        nonlocal tof
        trunk_v.append(v)
        trunk_f.append(f + tof)
        trunk_groups.append((f"tree_{tree_id}_trunk", f + tof))
        tof += len(v)

    base, with_trunk, _ = parse_lod(cfg.lod)

    def emit(tree_id, sub_pts, tp, shape="ellipsoid"):
        """Mesh one tree instance at the configured LoD; append if valid.

        The crown spans the canopy (crown base → top); the trunk (ground →
        canopy) is added only for the ``.1`` LoD variants.
        """
        nonlocal n_crowns, n_trunks
        z_cb = tp["z_crown_base"]

        if base <= 2:               # parametric crown — size from the *pruned*
            # points so the extent matches the LoD-3 hull (same component pruning)
            mask = kept_component_mask(
                sub_pts, cfg.crown.voxel_size, cfg.crown.sigma,
                cfg.crown.iso_level, cfg.crown.min_component_frac)
            cpts = sub_pts[mask] if mask.sum() >= 10 else sub_pts
            cx, cy = float(cpts[:, 0].mean()), float(cpts[:, 1].mean())
            dx, dy = float(np.ptp(cpts[:, 0])), float(np.ptp(cpts[:, 1]))
            z_top = max(float(cpts[:, 2].max()), z_cb + 0.5)
            if base <= 1:           # block of just the crown volume = LoD1
                v_crown, f_crown = block_mesh(cx, cy, z_cb, z_top, dx, dy,
                                              n_sides=cfg.trunk.n_sides)
            else:                   # species-driven crown primitive = LoD2
                v_crown, f_crown = crown_primitive(shape, cx, cy, z_cb, z_top, dx, dy,
                                                   n_sides=cfg.trunk.n_sides)
        else:                       # marching-cubes canopy hull = LoD3
            v_crown, f_crown = segment_to_marching_cubes(
                sub_pts, voxel_size=cfg.crown.voxel_size, sigma=cfg.crown.sigma,
                iso_level=cfg.crown.iso_level,
                min_component_frac=cfg.crown.min_component_frac)
            if cfg.crown.wrap:
                v_crown, f_crown = wrap_and_simplify(v_crown, f_crown, offset_ratio=cfg.crown.offset_ratio)

        if len(v_crown) == 0 or len(f_crown) == 0:
            print(f"[CGAL ERROR] Empty mesh for tree={tree_id}")
            return

        if not np.isfinite(v_crown).all():
            raise ValueError(f"Non-finite CGAL vertices for tree={tree_id}")

        if f_crown.min() < 0 or f_crown.max() >= len(v_crown):
            raise ValueError(f"Invalid CGAL face indices for tree={tree_id}")
        if len(v_crown) == 0:
            return
        _add_crown(tree_id, v_crown, f_crown)
        n_crowns += 1

        if with_trunk:
            z_trunk_top = _connected_trunk_top(
                z_cb, float(v_crown[:, 2].min()), cfg.trunk.crown_overlap)
            _add_trunk(tree_id, *make_trunk_mesh(
                tp["x"], tp["y"], tp["z_base"], z_trunk_top,
                tp["r_base"], tp["r_top"], n_sides=cfg.trunk.n_sides))
            n_trunks += 1

    for lbl in unique_lbls:
        tree_pts = pts_veg[labels == lbl]

        # Inventory trees connecting to this crown (not already claimed elsewhere).
        connected = [i for i in crown_inventory_indices(
            tree_pts[:, :2], inv_local, strtree, cfg.inventory.connect_buffer)
            if i not in claimed]

        if connected:  # split the blob into one instance per inventory tree
            footprint = crown_footprint(tree_pts[:, :2])
            seeds = [snap_under_crown(inv_local[i], footprint) for i in connected]
            for i in connected:
                claimed.add(i)
            if len(seeds) == 1:
                assign = np.zeros(len(tree_pts), dtype=int)
            else:
                assign = cKDTree(np.array(seeds)).query(tree_pts[:, :2])[1]
            for k, (idx, seed) in enumerate(zip(connected, seeds)):
                sub = tree_pts[assign == k]
                if len(sub) == 0:
                    continue
                rec = inv_recs[idx]
                dbh_cm = rec.get("dbh_cm")
                tp = _trunk_for(sub, dtm, georef, cfg,
                                dbh_m=(dbh_cm / 100.0 if dbh_cm else None),
                                force_xy=seed)
                if tp is not None:
                    instance_lbl = f"{lbl}_inv_{idx}"
                    emit(instance_lbl, sub, tp,
                         crown_shape_for_species(rec.get("species"), rec.get("common_name")))
        else:  # keep the whole crown with the LiDAR-estimated trunk
            tp = _trunk_for(tree_pts, dtm, georef, cfg)
            if tp is not None:
                emit(lbl, tree_pts, tp)  # no species → default ellipsoid crown at LoD 2

    def _stack(vs, fs):
        return (np.vstack(vs), np.vstack(fs)) if vs else (
            np.zeros((0, 3)), np.zeros((0, 3), dtype=int))

    cv, cf = _stack(crown_v, crown_f)
    tv, tf = _stack(trunk_v, trunk_f)
    return TileMeshes(cv, cf, tv, tf, crown_groups, trunk_groups, n_crowns, n_trunks)

def group_touching_crowns(pts, labels, threshold=0.5):
    """Merges labels of crowns if their points are within the distance threshold."""
    unique_lbls = np.unique(labels[labels > 0])
    if len(unique_lbls) <= 1:
        return labels

    voxel_size = threshold / 2.0
    coords = np.floor(pts / voxel_size).astype(int)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)

    ds_pts = pts[unique_idx]
    ds_labels = labels[unique_idx]

    valid = ds_labels > 0
    ds_pts = ds_pts[valid]
    ds_labels = ds_labels[valid]

    tree = cKDTree(ds_pts)
    pairs = tree.query_pairs(r=threshold)

    lbl_to_idx = {lbl: i for i, lbl in enumerate(unique_lbls)}
    idx_to_lbl = {i: lbl for i, lbl in enumerate(unique_lbls)}
    n_lbls = len(unique_lbls)

    adj = lil_matrix((n_lbls, n_lbls), dtype=int)
    for i, j in pairs:
        l1, l2 = ds_labels[i], ds_labels[j]
        if l1 != l2:
            u, v = lbl_to_idx[l1], lbl_to_idx[l2]
            adj[u, v] = 1
            adj[v, u] = 1

    _, comp_labels = connected_components(adj, directed=False)

    new_labels = labels.copy()
    for i, comp in enumerate(comp_labels):
        orig_lbl = idx_to_lbl[i]
        new_labels[labels == orig_lbl] = comp + 1  # +1 keeps the background at 0

    return new_labels


def extract_detected_trees(pts_veg: np.ndarray, labels: np.ndarray, crs) -> gpd.GeoDataFrame:
    """Converts segmented tree point labels into GeoPandas point centroids."""
    tree_records = []
    unique_labels = np.unique(labels)

    for lbl in unique_labels:
        if lbl == 0:
            continue
        cluster = pts_veg[labels == lbl]

        # Take highest point (tree top) as point geometry
        max_idx = np.argmax(cluster[:, 2])
        top_x, top_y, top_z = cluster[max_idx]

        tree_records.append({
            "tree_id": int(lbl),
            "geometry": gpd.points_from_xy([top_x], [top_y])[0],
            "z_max": float(top_z),
            "point_count": len(cluster)
        })

    return gpd.GeoDataFrame(tree_records, crs=crs)

def build_tile(tile: Path, cfg: Config, progress_callback=None, polygon=None):
    """Process one tile end to end and return its crown/trunk meshes.

    ``inventory`` is a preloaded ``(lonlat, records)`` tuple from the universal
    CSV, or ``None`` to skip inventory matching. Returns ``None`` if the tile is
    skipped (too few points) or yields no meshes. Writing is handled by the
    caller (``run_pipeline``) so tiles can be merged.
    """
    tile_name = _tile_name(tile)
    translation = np.array(cfg.translation)

    # ---------------------------------------------------
    # -- Load tile once; extract vegetation and ground --
    # ---------------------------------------------------
    las = read_las(tile)
    crs = las.header.parse_crs()
    epsg = crs.to_epsg() if crs else None
    pts_veg = filter_vegetation(las, cfg.filtering)
    pts_ground = filter_ground(las, cfg.filtering)
    del las
    logging.info(f"Found {len(pts_veg):>10,} vegetation pts and {len(pts_ground):>10,} ground pts (for DTM)")

    if len(pts_ground) < 100:
        logging.exception("Too few ground points to build DTM, skipping.")
        return None

    # ---------------------------------------------------
    # -------------- Clean vegetation -------------------
    # ---------------------------------------------------
    if cfg.cleaning.sor_neighbors > 0:
        pts_veg = remove_outliers_sor(
            pts_veg, cfg.cleaning.sor_neighbors, cfg.cleaning.sor_std_ratio
        )
        logging.info(f"After SOR: {len(pts_veg):>10,} pts remaining")
    logging.info(cfg.cleaning.voxel_size)
    if cfg.cleaning.voxel_size > 0:
        pts_veg = voxel_downsample(pts_veg, cfg.cleaning.voxel_size)
        logging.info(f"After voxel downsample: {len(pts_veg):>10,} pts remaining")

    if len(pts_veg) < 100:
        logging.exception("Too few vegetation points, skipping.")
        return None

    pts_veg -= translation
    pts_ground -= translation

    t0 = time.perf_counter()

    if progress_callback:
        progress_callback("Running CHM segmentation...")

    t_start_segment = time.perf_counter()
    splitted_polygon = split_polygon(polygon)

    # -------------------------------------------------------------------
    # --- CHM segmentation (returns the DTM for trunk ground lookup) ----
    # -------------------------------------------------------------------

    temp_results = []
    for poly in splitted_polygon.geometry:
        poly_local = translate_geometry(
                poly,
                xoff=-translation[0],
                yoff=-translation[1],
            )

        # Keep the polygon as a Shapely geometry.
        logging.info("Filtering points for polygon...")
        shapely.prepare(poly_local)

        veg_mask = shapely.contains_xy(
            poly_local,
            pts_veg[:, 0],
            pts_veg[:, 1]
        )

        veg_indices = np.flatnonzero(veg_mask)
        temp_pts_veg = pts_veg[veg_mask]

        ground_mask = shapely.contains_xy(
            poly_local,
            pts_ground[:, 0],
            pts_ground[:, 1]
        )

        temp_pts_ground = pts_ground[ground_mask]

        logging.info("Starting segmentation for sub-polygon...")
        labels, dtm, georef = segment_trees_chm(
            temp_pts_veg,
            temp_pts_ground,
            cell_size=cfg.segmentation.cell_size,
            smooth_sigma=cfg.segmentation.smooth_sigma,
            min_height=cfg.segmentation.min_height,
            peak_min_dist_m=cfg.segmentation.peak_min_dist,
            min_tree_points=cfg.segmentation.min_tree_pts,
            area_type=determine_lcz_of_area(
                gpd.GeoDataFrame(
                    geometry=[poly],
                    crs=splitted_polygon.crs,
                )
            ),
        )

        temp_results.append({
            "labels": labels,
            "point_indices": veg_indices,
            "dtm": dtm,
            "georef": georef,
        })

    labels, dtm, georef = combine_tile_results(
        temp_results,
        n_points=len(pts_veg),
    )

    logging.info("Merged segmentation results")

    # labels, dtm, georef = segment_trees_chm(
    #     pts_veg, pts_ground,
    #     cell_size=cfg.segmentation.cell_size,
    #     smooth_sigma=cfg.segmentation.smooth_sigma,
    #     min_height=cfg.segmentation.min_height,
    #     peak_min_dist_m=cfg.segmentation.peak_min_dist,
    #     min_tree_points=cfg.segmentation.min_tree_pts,
    #     area_type="mid_built_up"
    # )
    logging.info(f"-> Segmentation done in: {time.perf_counter() - t0:.1f} s")
    del pts_ground

    if getattr(cfg.output, 'merge_touching_crowns', False):
        if progress_callback:
            progress_callback("Merging touching crowns...")
        n_before = len(np.unique(labels[labels > 0]))
        labels = group_touching_crowns(pts_veg, labels, threshold=0.5)
        n_after = len(np.unique(labels[labels > 0]))
        logging.info(f"Merged touching crowns: reduced from {n_before} to {n_after} unique crowns")

    segmentation_time = time.perf_counter() - t_start_segment
    t_start_clean = time.perf_counter()

    # 5. Drop trees whose centroid sits inside an OSM building footprint
    if cfg.buildings.enabled:
        if epsg == 7415:
            labels, n_removed = filter_trees_in_buildings_bag(
                labels, pts_veg[:, :2], cfg.translation, epsg,
                buffer=cfg.buildings.buffer
            )
            logging.info(f"Buildings (BAG): removed {n_removed} trees inside footprints")

        else:
            labels, n_removed = filter_trees_in_buildings(
                labels, pts_veg[:, :2], cfg.translation, epsg,
                buffer=cfg.buildings.buffer,
                overpass_url=cfg.buildings.overpass_url,
                cache_dir=cfg.buildings.cache_dir,
            )
            logging.info(f"Buildings (OSM): removed {n_removed} trees inside footprints")

    if progress_callback:
        progress_callback("Exporting tree locations...")

    trees_gdf = extract_detected_trees(pts_veg, labels, crs)
    if not trees_gdf.empty:
        out_file = cfg.paths.output_dir / f"{tile_name}_tree_locations.gpkg"
        trees_gdf.to_file(out_file, driver="GPKG")
        logging.info(f"Exported {len(trees_gdf)} tree locations to {out_file.name}")

    # 6. Match surviving trees to the loaded inventory (CSV sidecar + trunk source)
    inv_local = inv_recs = None
    # if cfg.inventory.enabled and inventory is not None:
    #     inv_csv = cfg.paths.output_dir / f"{tile_name}_trees.csv"
    #     n_matched, n_alive, n_dbh = annotate_and_write(
    #         inv_csv, labels, pts_veg[:, :2], cfg.translation, epsg,
    #         inventory, cfg.inventory.match_dist
    #     )
    #     print(
    #         f"    inventory: matched {n_matched}/{n_alive} trees, "
    #         f"{len(n_dbh)} with DBH -> {inv_csv.name}"
    #     )
    #     inv_lonlat, inv_recs = inventory
    #     inv_local = inventory_local_xy(inv_lonlat, epsg, cfg.translation)

    if progress_callback:
        progress_callback(f"Reconstructing 3D meshes...")

    cleaning_time = time.perf_counter() - t_start_clean
    t_start_mesh = time.perf_counter()

    # 7. Mesh every tree (crown + inventory-driven trunks)
    tm = _build_tile_meshes(pts_veg, labels, dtm, georef, cfg, inv_local, inv_recs)
    del pts_veg, dtm

    if tm.is_empty():
        logging.warning("No meshes produced, skipping.\n")
        return None
    n_faces = len(tm.crown_f) + len(tm.trunk_f)
    logging.info(f"{tm.n_crowns} trees -> {len(tm.crown_v) + len(tm.trunk_v):,} vertices {n_faces:,} faces  ({time.perf_counter() - t0:.1f} s total)")

    meshing_time = time.perf_counter() - t_start_mesh
    return tm, segmentation_time, cleaning_time, meshing_time


def _write_meshes(stem: Path, tm: TileMeshes, separate: bool) -> None:
    """Write ``tm`` to OBJ at ``stem`` — separate crown/trunk files or one grouped."""
    if separate:
        if len(tm.crown_v):
            write_obj_grouped(stem.with_name(stem.name + "_crown.obj"), tm.crown_v, tm.crown_groups)
        if len(tm.trunk_v):
            write_obj_grouped(stem.with_name(stem.name + "_trunk.obj"), tm.trunk_v, tm.trunk_groups)
    else:
        verts = np.vstack([v for v in (tm.crown_v, tm.trunk_v) if len(v)])
        offset = len(tm.crown_v)
        shifted_trunk_groups = [(name, f + offset) for name, f in tm.trunk_groups]

        all_groups = tm.crown_groups + shifted_trunk_groups

        write_obj_grouped(stem.with_name(stem.name + ".obj"), verts, all_groups)


def _merge_tiles(metas) -> TileMeshes:
    """Concatenate per-tile TileMeshes into one (separate crown/trunk offsets)."""
    cv, cf, tv, tf = [], [], [], []
    cg, tg = [], []
    nc = nt = 0
    cof = tof = 0
    for tm in metas:
        if len(tm.crown_v):
            cv.append(tm.crown_v); cf.append(tm.crown_f + cof)
            cg.extend([(name, f + cof) for name, f in tm.crown_groups])
            cof += len(tm.crown_v)
        if len(tm.trunk_v):
            tv.append(tm.trunk_v); tf.append(tm.trunk_f + tof)
            tg.extend([(name, f + tof) for name, f in tm.trunk_groups])
            tof += len(tm.trunk_v)
        nc += tm.n_crowns; nt += tm.n_trunks

    def stk(vs, fs):
        return (np.vstack(vs), np.vstack(fs)) if vs else (
            np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    cvv, cff = stk(cv, cf)
    tvv, tff = stk(tv, tf)
    return TileMeshes(cvv, cff, tvv, tff, cg, tg, nc, nt)


def run_pipeline(cfg: Config, polygon: Path = None, progress_callback=None):
    """Process every tile in the configured tiles directory."""
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    tiles = find_tiles(cfg.paths.tiles_dir)
    logging.info(f"Processing {len(tiles)} tile -> {cfg.paths.output_dir}/")

    write_offset(cfg.paths.output_dir / "offset.txt", cfg.translation)

    # # Load the universal inventory CSV once, shared across all tiles.
    # inventory = None
    # if cfg.inventory.enabled:
    #     path = Path(cfg.inventory.csv_path)
    #     if cfg.inventory.csv_path and path.exists():
    #         inventory = load_inventory_csv(path)
    #         print(f"Inventory: {len(inventory[1]):,} trees from {path}\n")
    #     else:
    #         print(
    #             f"Inventory enabled but CSV not found ({cfg.inventory.csv_path or '<unset>'}); "
    #             "run `python -m tree4cfd inventory` first — skipping matching.\n"
    #         )

    label = parse_lod(cfg.lod)[2]
    sep = cfg.output.separate_crown_trunk
    merge = cfg.output.merge_tiles
    out_dir = cfg.paths.output_dir

    total_seg = 0.0
    total_clean = 0.0
    total_mesh = 0.0

    metas = []  # collected meshes for merging
    seen = set()

    for i, tile in enumerate(tiles, 1):
        name = _tile_name(tile)
        if name in seen:
            logging.info(f"{tile.name}  ->  duplicate of {name}, skipping")
            continue
        seen.add(name)

        stem = out_dir / f"{name}_lod{label}"
        if not merge:
            primary = stem.with_name(stem.name + ("_crown.obj" if sep else ".obj"))
            if primary.exists():
                logging.info(f"{tile.name}  ->  already done, skipping")
                continue
        if polygon is None:
            polygon = get_outlines(tiles)
            result = build_tile(tile, cfg, progress_callback, polygon)
        else:
            result = build_tile(tile, cfg, progress_callback, polygon)
        if result is None:
            continue

        tm, seg_time, clean_time, mesh_time = result
        total_seg += seg_time
        total_clean += clean_time
        total_mesh += mesh_time

        if merge:
            metas.append(tm)
        else:
            _write_meshes(stem, tm, sep)
            logging.info(f"→ {stem.name}.obj\n")

    if merge and metas:
        merged = _merge_tiles(metas)
        stem = out_dir / f"merged_lod{label}"
        _write_meshes(stem, merged, sep)
        logging.info(f"Merged {len(metas)} tiles -> {merged.n_crowns} trees, {len(merged.crown_f) + len(merged.trunk_f):,} faces -> {stem.name}*.obj")

    return total_seg, total_clean, total_mesh