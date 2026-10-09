"""Per-tile orchestration: filter → clean → segment → mesh → write .obj."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from .buildings import filter_trees_in_buildings
from synthesis.building_filtering_bag import filter_trees_in_buildings_bag
from .cleaning import remove_outliers_sor, voxel_downsample
from .config import Config
from .crown import kept_component_mask, segment_to_marching_cubes
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

from .inventory import (
    annotate_and_write,
    build_inventory_index,
    crown_footprint,
    crown_inventory_indices,
    inventory_local_xy,
    load_inventory_csv,
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
from .shapes import filter_pole_wall_trees
from .trunk import estimate_trunk_params, make_trunk_mesh



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
    cof = tof = 0  # separate vertex offsets for the crown and trunk groups
    n_crowns = n_trunks = 0

    def _add_crown(v, f):
        nonlocal cof
        crown_v.append(v); crown_f.append(f + cof); cof += len(v)

    def _add_trunk(v, f):
        nonlocal tof
        trunk_v.append(v); trunk_f.append(f + tof); tof += len(v)

    base, with_trunk, _ = parse_lod(cfg.lod)

    def emit(sub_pts, tp, shape="ellipsoid"):
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
            if base <= 1:           # block of just the crown volume
                v_crown, f_crown = block_mesh(cx, cy, z_cb, z_top, dx, dy,
                                              n_sides=cfg.trunk.n_sides)
            else:                   # species-driven crown primitive
                v_crown, f_crown = crown_primitive(shape, cx, cy, z_cb, z_top, dx, dy,
                                                   n_sides=cfg.trunk.n_sides)
        else:                       # marching-cubes canopy hull
            v_crown, f_crown = segment_to_marching_cubes(
                sub_pts, voxel_size=cfg.crown.voxel_size, sigma=cfg.crown.sigma,
                iso_level=cfg.crown.iso_level,
                min_component_frac=cfg.crown.min_component_frac)
        if len(v_crown) == 0:
            return
        _add_crown(v_crown, f_crown)
        n_crowns += 1

        if with_trunk:
            z_trunk_top = _connected_trunk_top(
                z_cb, float(v_crown[:, 2].min()), cfg.trunk.crown_overlap)
            _add_trunk(*make_trunk_mesh(
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
                    emit(sub, tp,
                         crown_shape_for_species(rec.get("species"), rec.get("common_name")))
        else:  # keep the whole crown with the LiDAR-estimated trunk
            tp = _trunk_for(tree_pts, dtm, georef, cfg)
            if tp is not None:
                emit(tree_pts, tp)  # no species → default ellipsoid crown at LoD 2

    def _stack(vs, fs):
        return (np.vstack(vs), np.vstack(fs)) if vs else (
            np.zeros((0, 3)), np.zeros((0, 3), dtype=int))

    cv, cf = _stack(crown_v, crown_f)
    tv, tf = _stack(trunk_v, trunk_f)
    return TileMeshes(cv, cf, tv, tf, n_crowns, n_trunks)

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


def build_tile(tile: Path, cfg: Config, progress_callback=None):
    """Process one tile end to end and return its crown/trunk meshes.

    ``inventory`` is a preloaded ``(lonlat, records)`` tuple from the universal
    CSV, or ``None`` to skip inventory matching. Returns ``None`` if the tile is
    skipped (too few points) or yields no meshes. Writing is handled by the
    caller (``run_pipeline``) so tiles can be merged.
    """
    tile_name = _tile_name(tile)
    print(f"  {tile.name}")
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
    print(f"    vegetation: {len(pts_veg):>10,} pts    ground: {len(pts_ground):>10,} pts")

    if len(pts_ground) < 100:
        print("    Too few ground points to build DTM, skipping.\n")
        return None

    # ---------------------------------------------------
    # -------------- Clean vegetation -------------------
    # ---------------------------------------------------
    if cfg.cleaning.sor_neighbors > 0:
        pts_veg = remove_outliers_sor(
            pts_veg, cfg.cleaning.sor_neighbors, cfg.cleaning.sor_std_ratio
        )
        print(f"    after SOR: {len(pts_veg):>10,} pts")
    if cfg.cleaning.voxel_size > 0:
        pts_veg = voxel_downsample(pts_veg, cfg.cleaning.voxel_size)
        print(f"    after voxel downsample: {len(pts_veg):>10,} pts")

    if len(pts_veg) < 100:
        print("    Too few vegetation points, skipping.\n")
        return None

    pts_veg -= translation
    pts_ground -= translation

    t0 = time.perf_counter()

    if progress_callback:
        progress_callback("Running CHM segmentation...")

    t_start_segment = time.perf_counter()

    # -------------------------------------------------------------------
    # --- CHM segmentation (returns the DTM for trunk ground lookup) ----
    # -------------------------------------------------------------------
    labels, dtm, georef = segment_trees_chm(
        pts_veg, pts_ground,
        cell_size=cfg.segmentation.cell_size,
        smooth_sigma=cfg.segmentation.smooth_sigma,
        min_height=cfg.segmentation.min_height,
        peak_min_dist_m=cfg.segmentation.peak_min_dist,
        min_tree_points=cfg.segmentation.min_tree_pts,
    )
    print(f"    -> Segmentation done in: {time.perf_counter() - t0:.1f} s")
    del pts_ground

    if getattr(cfg.output, 'merge_touching_crowns', False):
        if progress_callback:
            progress_callback("Merging touching crowns...")
        n_before = len(np.unique(labels[labels > 0]))
        labels = group_touching_crowns(pts_veg, labels, threshold=0.5)
        n_after = len(np.unique(labels[labels > 0]))
        print(f"    merged touching: reduced from {n_before} to {n_after} unique crowns")

    segmentation_time = time.perf_counter() - t_start_segment
    t_start_clean = time.perf_counter()

    # 5. Drop trees whose centroid sits inside an OSM building footprint
    if cfg.buildings.enabled:
        if epsg == 7415:
            labels, n_removed = filter_trees_in_buildings_bag(
                labels, pts_veg[:, :2], cfg.translation, epsg,
                buffer=cfg.buildings.buffer
            )
            print(f"    buildings (BAG): removed {n_removed} trees inside footprints")

        else:
            labels, n_removed = filter_trees_in_buildings(
                labels, pts_veg[:, :2], cfg.translation, epsg,
                buffer=cfg.buildings.buffer,
                overpass_url=cfg.buildings.overpass_url,
                cache_dir=cfg.buildings.cache_dir,
            )
            print(f"    buildings (OSM): removed {n_removed} trees inside footprints")

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
        print("    No meshes produced, skipping.\n")
        return None
    n_faces = len(tm.crown_f) + len(tm.trunk_f)
    print(
        f"    {tm.n_crowns} trees → {len(tm.crown_v) + len(tm.trunk_v):,} vertices  "
        f"{n_faces:,} faces  ({time.perf_counter() - t0:.1f} s total)"
    )

    meshing_time = time.perf_counter() - t_start_mesh
    return tm, segmentation_time, cleaning_time, meshing_time


def _write_meshes(stem: Path, tm: TileMeshes, separate: bool) -> None:
    """Write ``tm`` to OBJ at ``stem`` — separate crown/trunk files or one grouped."""
    if separate:
        if len(tm.crown_v):
            write_obj(stem.with_name(stem.name + "_crown.obj"), tm.crown_v, tm.crown_f)
        if len(tm.trunk_v):
            write_obj(stem.with_name(stem.name + "_trunk.obj"), tm.trunk_v, tm.trunk_f)
    else:
        verts = np.vstack([v for v in (tm.crown_v, tm.trunk_v) if len(v)])
        write_obj_grouped(stem.with_name(stem.name + ".obj"), verts, [
            ("crown", tm.crown_f),
            ("trunk", tm.trunk_f + len(tm.crown_v)),
        ])


def _merge_tiles(metas) -> TileMeshes:
    """Concatenate per-tile TileMeshes into one (separate crown/trunk offsets)."""
    cv, cf, tv, tf, nc, nt = [], [], [], [], 0, 0
    cof = tof = 0
    for tm in metas:
        if len(tm.crown_v):
            cv.append(tm.crown_v); cf.append(tm.crown_f + cof); cof += len(tm.crown_v)
        if len(tm.trunk_v):
            tv.append(tm.trunk_v); tf.append(tm.trunk_f + tof); tof += len(tm.trunk_v)
        nc += tm.n_crowns; nt += tm.n_trunks

    def stk(vs, fs):
        return (np.vstack(vs), np.vstack(fs)) if vs else (
            np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    cvv, cff = stk(cv, cf)
    tvv, tff = stk(tv, tf)
    return TileMeshes(cvv, cff, tvv, tff, nc, nt)


def run_pipeline(cfg: Config, progress_callback=None):
    """Process every tile in the configured tiles directory."""
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)

    tiles = find_tiles(cfg.paths.tiles_dir)
    print(f"Processing {len(tiles)} tile -> {cfg.paths.output_dir}/\n")

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
        print(f"Processing: ", end=" ")
        name = _tile_name(tile)
        if name in seen:
            print(f"  {tile.name}  ->  duplicate of {name}, skipping")
            continue
        seen.add(name)

        stem = out_dir / f"{name}_lod{label}"
        if not merge:
            primary = stem.with_name(stem.name + ("_crown.obj" if sep else ".obj"))
            if primary.exists():
                print(f"  {tile.name}  ->  already done, skipping")
                continue
        result = build_tile(tile, cfg, progress_callback)
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
            print(f"    → {stem.name}.obj\n")

    if merge and metas:
        merged = _merge_tiles(metas)
        stem = out_dir / f"merged_lod{label}"
        _write_meshes(stem, merged, sep)
        print(f"\nMerged {len(metas)} tiles → {merged.n_crowns} trees, "
              f"{len(merged.crown_f) + len(merged.trunk_f):,} faces → {stem.name}*.obj")

    return total_seg, total_clean, total_mesh
