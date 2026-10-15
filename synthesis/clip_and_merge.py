import geopandas as gpd
import laspy
import numpy as np
import trimesh
import logging
import json

from shapely import box, contains_xy, Point
from pathlib import Path
from cjio import cityjson

def clip_and_merge_pointclouds(
        tiles_folder: str,
        output_folder: str,
        gpkg_path: str,
        tile_ids: list[str] = None,
        output_filename: str = "clipped_output.laz",
):
    tiles_dir = Path(tiles_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # ------------ Find GKPG and buffer it --------------
    # ---------------------------------------------------
    if not Path(gpkg_path).exists():
        logging.exception(f"AOI file not found at {gpkg_path}")

    gdf = gpd.read_file(gpkg_path)
    unbuffered = gdf.geometry.union_all() if len(gdf) > 1 else gdf.geometry.iloc[0]
    buffered_polygon = unbuffered.buffer(0.5)
    p_minx, p_miny, p_maxx, p_maxy = buffered_polygon.bounds

    # ---------------------------------------------------
    # -- Check if the tile has been downloaded before ---
    # ---------------------------------------------------
    if tile_ids:
        laz_files = []
        for tile_id in tile_ids:
            tile_folder = tiles_dir / "tiles" / tile_id
            if tile_folder.exists() and tile_folder.is_dir():
                laz_files.extend(list(tile_folder.rglob("*.laz")))
    else:
        laz_files = list(tiles_dir.rglob("*.laz"))

    output_laz = out_dir / output_filename
    laz_files = [f for f in laz_files if f.resolve() != output_laz.resolve()]
    logging.info(f"Found {len(laz_files)} .laz file(s) to process.")
    if not laz_files:
        logging.exception("No .laz files found to process.")

    # ---------------------------------------------------
    # --------- Clip each LAZ file to the AOI -----------
    # ---------------------------------------------------
    output_laz = out_dir / output_filename

    candidate_tiles = []
    for f in laz_files:
        with laspy.open(str(f)) as r:
            t_minx, t_miny, _ = r.header.mins
            t_maxx, t_maxy, _ = r.header.maxs

            # Skip tile entirely if bounding boxes do not overlap
            if not (t_maxx < p_minx or t_minx > p_maxx or t_maxy < p_miny or t_miny > p_maxy):
                candidate_tiles.append(f)

    with laspy.open(str(laz_files[0])) as first_las:
        clean_header = laspy.LasHeader(version=first_las.header.version, point_format=first_las.header.point_format)
        clean_header.scales = first_las.header.scales
        clean_header.offsets = first_las.header.offsets

        for vlr in first_las.header.vlrs:
            if getattr(vlr, "user_id", "") != "copc":
                clean_header.vlrs.append(vlr)

        for evlr in first_las.header.evlrs:
            if getattr(evlr, "user_id", "") != "copc":
                clean_header.evlrs.append(evlr)

    points_written = 0
    all_mins = [np.inf, np.inf, np.inf]
    all_maxs = [-np.inf, -np.inf, -np.inf]
    logging.info("Opening laz file")
    with laspy.open(str(output_laz), mode="w", header=clean_header) as writer:
        for tile_path in candidate_tiles:
            with laspy.open(str(tile_path), laz_backend=laspy.LazBackend.Laszip) as las:
                t_minx, t_miny, _ = las.header.mins
                t_maxx, t_maxy, _ = las.header.maxs
                tile_box = box(t_minx, t_miny, t_maxx, t_maxy)
                is_fully_contained = buffered_polygon.contains(tile_box)
                for chunk in las.chunk_iterator(1_000_000):
                    if is_fully_contained:
                        matched = chunk
                    else:
                        x, y = chunk.x, chunk.y

                        # Vectorized bounding box pre-screen
                        bbox_mask = (x >= p_minx) & (x <= p_maxx) & (y >= p_miny) & (y <= p_maxy)
                        if not np.any(bbox_mask):
                            continue

                        # Precise shapely check on candidate subset
                        cand_idx = np.flatnonzero(bbox_mask)
                        poly_mask = contains_xy(buffered_polygon, x[cand_idx], y[cand_idx])

                        if not np.any(poly_mask):
                            continue

                        matched = chunk[cand_idx[poly_mask]]

                    logging.info(f"writing {len(matched)} points")

                    writer.write_points(matched)
                    points_written += len(matched)

                    all_mins = np.minimum(all_mins, [matched.x.min(), matched.y.min(), matched.z.min()])
                    all_maxs = np.maximum(all_maxs, [matched.x.max(), matched.y.max(), matched.z.max()])

    if points_written == 0:
        output_laz.unlink(missing_ok=True)
        logging.exception("No points found within the buffered geometry.")

    with laspy.open(str(output_laz), mode="a") as dst:
        dst.header.mins = all_mins
        dst.header.maxs = all_maxs

    logging.info(f"Merged & clipped point cloud saved")

def clip_and_merge_objs(obj_paths: list, aoi_gpkg_path: str, output_obj_path: Path):
    """Merges multiple .obj files into a single .obj file"""
    if not obj_paths:
        logging.exception("No .obj files found for the selected LoD.")
        return False

    aoi_gdf = gpd.read_file(aoi_gpkg_path)
    aoi_polygon = aoi_gdf.union_all()
    buffered_polygon = aoi_polygon.buffer(0.5)
    p_minx, p_miny, p_maxx, p_maxy = buffered_polygon.bounds

    all_kept_meshes: list[trimesh.Trimesh] = []

    for obj_path in obj_paths:
        try:
            mesh = trimesh.load(obj_path, force='mesh')

            if not isinstance(mesh, trimesh.Trimesh):
                continue
            if getattr(mesh, 'is_empty', True):
                continue

            x = mesh.vertices[:, 0]
            y = mesh.vertices[:, 1]
            bbox_mask = (x >= p_minx) & (x <= p_maxx) & (y >= p_miny) & (y <= p_maxy)

            if not np.any(bbox_mask):
                continue

            can_idx = np.flatnonzero(bbox_mask)
            poly_mask = contains_xy(buffered_polygon, x[can_idx], y[can_idx])

            if not np.any(poly_mask):
                continue

            v_valid = np.zeros(len(mesh.vertices), dtype=bool)
            v_valid[can_idx[poly_mask]] = True

            face_mask = v_valid[mesh.faces].any(axis=1)

            if np.any(face_mask):
                clipped = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[face_mask], process=False)
                clipped.remove_unreferenced_vertices()
                all_kept_meshes.append(clipped)

        except Exception as e:
            logging.exception(f"Error processing {Path(obj_path).name}: {e}")

    if all_kept_meshes:
        merged_mesh = trimesh.util.concatenate(all_kept_meshes)

        if not getattr(merged_mesh, 'is_empty', True):
            merged_mesh.export(str(output_obj_path))
            logging.info("3D BAG clip and merge process completed.")
            return True

    logging.warning("Something went wrong when merging .obj files")
    return False

def clip_and_merge_cityjson(cj_paths: list, aoi_gpkg_path: str, output_cj_path: Path, lod: str = "2.2"):
    if not cj_paths:
        logging.exception("No CityJSON files found.")
        return False

    aoi_gdf = gpd.read_file(aoi_gpkg_path)
    buffered_polygon = aoi_gdf.union_all().buffer(0.5)

    out_scale = [0.001, 0.001, 0.001]
    out_translate = [0.0, 0.0, 0.0]

    merged_cj = {
        "type": "CityJSON",
        "version": "2.0",
        "transform": {"scale": out_scale, "translate": out_translate},
        "CityObjects": {},
        "vertices": []
    }

    vertex_offset = 0

    for cj_path in cj_paths:
        with open(cj_path, "r") as f:
            cj_data = json.load(f)

        transform = cj_data.get("transform", {"scale": [1, 1, 1], "translate": [0, 0, 0]})
        scale = transform["scale"]
        translate = transform["translate"]

        vertices = cj_data.get("vertices", [])

        kept_objects = {}
        used_indices_set = set()

        for obj_id, city_obj in cj_data.get("CityObjects", {}).items():
            if city_obj["type"] not in ["Building", "BuildingPart"]:
                continue

            filter_geometries = [
                g for g in city_obj.get("geometry", [])
                if str(g.get("lod")) == lod
            ]

            if not filter_geometries:
                continue

            def flatten(lst):
                for item in lst:
                    if isinstance(item, list):
                        yield from flatten(item)
                    else:
                        yield item

            geom = filter_geometries[0]
            indices = list(flatten(geom["boundaries"]))

            if not indices:
                continue

            xs = [vertices[idx][0] * scale[0] + translate[0] for idx in indices]
            ys = [vertices[idx][1] * scale[1] + translate[1] for idx in indices]

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            building_box = box(min_x, min_y, max_x, max_y)

            if buffered_polygon.intersects(building_box):
                kept_objects[obj_id] = city_obj
                for g in filter_geometries:
                    used_indices_set.update(list(flatten(g["boundaries"])))

        if not kept_objects:
            continue

        old_to_new_mapping = {}
        current_merged_vertex_count = len(merged_cj["vertices"])

        for new_idx, old_idx in enumerate(sorted(used_indices_set)):
            old_to_new_mapping[old_idx] = current_merged_vertex_count + new_idx

            v = vertices[old_idx]
            gx = v[0] * scale[0] + translate[0]
            gy = v[1] * scale[1] + translate[1]
            gz = v[2] * scale[2] + translate[2]

            ix = int(round((gx - out_translate[0]) / out_scale[0]))
            iy = int(round((gy - out_translate[1]) / out_scale[1]))
            iz = int(round((gz - out_translate[2]) / out_scale[2]))

            merged_cj["vertices"].append([ix, iy, iz])

        # 3. Update the boundaries of the kept objects with the mapped indices
        for obj_id, city_obj in kept_objects.items():
            def remap_indices(lst):
                if isinstance(lst, list):
                    return [remap_indices(item) for item in lst]
                return old_to_new_mapping[lst]

            filtered_geoms = [g for g in city_obj.get("geometry", []) if str(g.get("lod")) == lod]

            for g in filtered_geoms:
                g["boundaries"] = remap_indices(g["boundaries"])

            city_obj["geometry"] = filtered_geoms
            merged_cj["CityObjects"][obj_id] = city_obj

    if merged_cj["CityObjects"]:
        with open(output_cj_path, "w") as f:
            json.dump(merged_cj, f)
        logging.info("3D BAG CityJSON clip and merge process completed.")
        return True

    logging.warning("Something went wrong when merging CityJSON files")
    return False


def cityjson_to_obj(cityjson_path: Path | str, output_obj_path: Path | str, translation: list[float] = None):
    """Converts CityJSON to OBJ without requiring the 'triangle' or 'cjio' packages."""
    cityjson_path = Path(cityjson_path)
    if not cityjson_path.exists():
        logging.warning(f"File {cityjson_path.name} does not exist. Skipping OBJ conversion.")
        return False

    try:
        with open(cityjson_path, "r") as f:
            cj_data = json.load(f)

        vertices = cj_data.get("vertices", [])
        if not vertices:
            logging.warning("No vertices found in CityJSON file")
            return False

        transform = cj_data.get("transform", {"scale": [1, 1, 1], "translate": [0, 0, 0]})
        scale = transform["scale"]
        translate_cj = transform["translate"]

        # 1. Convert local integer coordinates to global real-world floats
        raw_vertices = [
            (
                v[0] * scale[0] + translate_cj[0],
                v[1] * scale[1] + translate_cj[1],
                v[2] * scale[2] + translate_cj[2],
            )
            for v in vertices
        ]

        # 2. Apply the dynamic shared offset to pin buildings to the exact same origin as the trees
        if translation and len(translation) >= 2:
            off_x, off_y = translation[0], translation[1]
            off_z = translation[2] if len(translation) > 2 else 0.0
        else:
            off_x = min(v[0] for v in raw_vertices)
            off_y = min(v[1] for v in raw_vertices)
            off_z = 0.0

        local_vertices = [
            (v[0] - off_x, v[1] - off_y, v[2] - off_z)
            for v in raw_vertices
        ]

        def flatten_face(lst):
            for item in lst:
                if isinstance(item, list):
                    yield from flatten_face(item)
                else:
                    yield item

        def extract_faces(boundaries, geom_type):
            faces = []
            if geom_type in ["MultiSurface", "CompositeSurface"]:
                for surface in boundaries:
                    if surface:
                        faces.append(list(flatten_face(surface[0])))
            elif geom_type == "Solid":
                for shell in boundaries:
                    for surface in shell:
                        if surface:
                            faces.append(list(flatten_face(surface[0])))
            elif geom_type == "MultiSolid":
                for solid in boundaries:
                    for shell in solid:
                        for surface in shell:
                            if surface:
                                faces.append(list(flatten_face(surface[0])))
            return faces

        # 3. Write directly to OBJ
        used_indices = set()
        object_to_write = []

        for obj_id, city_obj in cj_data.get("CityObjects", {}).items():
            if city_obj["type"] not in ["Building", "BuildingPart"]:
                continue
            obj_faces = []
            for geom in city_obj.get("geometry", []):
                faces = extract_faces(geom["boundaries"], geom["type"])
                for face in faces:
                    used_indices.update(face)
                    obj_faces.append(face)
            if obj_faces:
                object_to_write.append((obj_id, obj_faces))

        old_to_new = {}
        new_idx = 1

        with open(output_obj_path, "w") as f:
            f.write(f"# Native OBJ Aligned to offset X: -{off_x}, Y: -{off_y}, Z: -{off_z}\n")

            for old_idx in sorted(used_indices):
                old_to_new[old_idx] = new_idx
                v = local_vertices[old_idx]
                f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
                new_idx += 1

            for obj_id, obj_faces in object_to_write:
                f.write(f"o {obj_id}\n")
                for face in obj_faces:
                    obj_face_indices = [str(old_to_new[idx]) for idx in face]
                    f.write(f"f {' '.join(obj_face_indices)}\n")

        logging.info("Successfully converted CityJSON to OBJ natively.")
        return True

    except Exception as e:
        logging.exception(f"Error converting CityJSON natively: {e}")
        return False

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    target_folder = script_dir.parent / "data" / "in" / "all-files" / "ams-mid-2"

    clip_and_merge_pointclouds(target_folder)