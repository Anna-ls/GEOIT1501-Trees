import sys
import pandas as pd
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import streamlit as st
import folium
import json
import geopandas as gpd

from folium.plugins import Draw
from streamlit_folium import st_folium
from datetime import datetime

from get_data import process_geojson
from tree4cfd.pipeline import run_pipeline
from tree4cfd.config import load_config
from rendering import render_3d_objs

st.title("Urban Tree Reconstruction")
st.write("Draw or upload a polygon and generate tree meshes within that area.")

if "out_dir" not in st.session_state:
    st.session_state.out_dir = None

# Adds option to either draw a polygon or upload one
input_method = st.radio("Choose input method:", ("Draw on Map", "Upload GeoJSON"))

st.subheader("Tree Settings")
lod_choice = st.radio("Choose Level of Detail (LoD):", ("LoD2", "LoD3"))
lod_value = 2.1 if lod_choice == "LoD2" else 3.1

if lod_choice == "LoD3":
    merge_touching = st.checkbox("Merge touching crowns into a single mesh", value=False)
    alpha_wrap = st.checkbox("Alpha Wrap the resulting geometries in order to simplify them", value=False)
else:
    merge_touching = False
    alpha_wrap = False

if alpha_wrap:
    offset_ratio = st.slider("Select feature complexity (lower = less triangles = smaller file size):", 20, 600, 100, 10)
else:
    offset_ratio = 50

st.subheader("Building Settings")
include_buildings = st.checkbox("Download and include 3D BAG buildings", value=False)
bag_lod = "0"
if include_buildings:
    bag_lod_choice = st.radio("Choose Building Level of Detail:", ("LoD1.2", "LoD1.3", "LoD2.2"))
    lod_map = {"LoD1.2": "1.2", "LoD1.3": "1.3", "LoD2.2": "2.2"}
    bag_lod = lod_map[bag_lod_choice]

st_data = None
uploaded_file = None

# ---------------------------------------------------------------
# --- Show either the map or the uploader based on the choice ---
# ---------------------------------------------------------------
if input_method == "Draw on Map":
    m = folium.Map(location=[51.9244, 4.4777], zoom_start=13, tiles=None)
    folium.TileLayer("OpenStreetMap", overlay=True, name="OpenStreetMap").add_to(m)

    draw = Draw(
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': False}
    )
    m.add_child(draw)
    st_data = st_folium(m, width=700, height=500)
else:
    uploaded_file = st.file_uploader("Upload a GeoJSON file defining your AOI", type=["geojson", "json"])

execution_times = {}
scores = {}
total_time = 0

# ---------------------------------------------------------------
# ---------------------- Pipeline execution ---------------------
# ---------------------------------------------------------------
if st.button("Run Reconstruction Pipeline"):
    case_name = "target_region"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root_dir / "DATA" / "OUT" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = out_dir / f"{case_name}.geojson"

    valid_input = False
    # ---------------------------------------------------
    # ------------- Retrieve drawn polygon --------------
    # ---------------------------------------------------
    if input_method == "Draw on Map":
        if st_data and st_data.get("last_active_drawing"):
            geojson_geometry = st_data["last_active_drawing"]
            geojson_feature = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature","properties": {},"geometry": geojson_geometry["geometry"],}
                ]
            }
            with open(geojson_path, 'w') as f:
                json.dump(geojson_feature, f)
            valid_input = True
        else:
            st.warning("Please draw a polygon on the map before running the pipeline")
    # ---------------------------------------------------
    # ------------ Retrieve uploaded file ---------------
    # ---------------------------------------------------
    elif input_method == "Upload GeoJSON":
        if uploaded_file is not None:
            with open(geojson_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_file.seek(0)
            geojson_geometry = json.load(uploaded_file)
            valid_input = True
        else:
            st.warning("Please upload a GeoJSON file before running the pipeline")

    if valid_input:
        try:
            # If clipped_output.laz exist, unlink it so it will overwrite
            laz_output = out_dir / "clipped_output.laz"
            if laz_output.exists():
                laz_output.unlink()

            aoi_gdf = gpd.read_file(geojson_path)

            if aoi_gdf.crs is None:
                aoi_gdf = aoi_gdf.set_crs(epsg=4326)
            if aoi_gdf.crs.to_epsg() != 7415:
                aoi_gdf = aoi_gdf.to_crs(epsg=7415)

            minx, miny, maxx, maxy = aoi_gdf.total_bounds
            translation = [float(minx), float(miny), 0.0]

            # ---------------------------------------------------
            # ------------- Run data collection -----------------
            # ---------------------------------------------------
            with st.spinner('Downloading, clipping, and merging AHN tiles...'):
                dl_time, ahn_clip_time, bag_clip_time = process_geojson(
                    geojson_path,
                    buffer_distance=10.0,
                    overwrite=False,
                    download_buildings=include_buildings,
                    bag_lod=bag_lod,
                    translation=translation
                )

            execution_times['1. Download LiDAR Tiles'] = dl_time
            execution_times['2. Clip & Merge Pointcloud'] = ahn_clip_time
            execution_times['3. Clip & Merge 3D buildings'] = bag_clip_time

            # ---------------------------------------------------
            # ---- Generate the config file for the pipeline ----
            # ---------------------------------------------------
            config_json_path = out_dir / "temp_config.json"
            config_data = {
                "paths": {
                    "tiles_dir": str(out_dir),
                    "output_dir": str(out_dir)
                },
                "translation": translation,
                "lod": lod_value,
                "filtering": {
                    "high_veg_classes": [1],
                    "multi_return_only": True
                },
                "cleaning": {
                    "sor_neighbors": 20,
                    "sor_std_ratio": 1.5,
                    "voxel_size": 0
                },
                "segmentation": {
                    "cell_size": 0.4,
                    "smooth_sigma": 2.8,
                    "min_height": 2.0,
                    "peak_min_dist": 1.8,
                    "min_tree_pts": 60,
                    "resolve_multi_trees": True,
                    "multi_tree_kwargs": {
                        "max_elongation": 3.0,
                        "max_offset_ratio": 1.5,
                        "num_angles": 4,
                        "bin_size": 0.5,
                        "smooth_sigma": 2.8
                    }
                },
                "trunk": {
                    "q_base": 0.20,
                    "q_crown": 0.10,
                    "allom_a": 0.03,
                    "allom_b": 1.2,
                    "taper": 0.4,
                    "n_sides": 12
                },
                "crown": {
                    "voxel_size": 0.5,
                    "sigma": 1.5,
                    "iso_level": 0.10,
                    "min_component_frac": 0.3,
                    "wrap": alpha_wrap,
                    "offset_ratio": offset_ratio
                },
                "buildings": {
                    "enabled": True,
                    "buffer": 2
                },
                "output": {
                    "merge_tiles": True,
                    "separate_crown_trunk": True,
                    "merge_touching_crowns": merge_touching
                },
                "inventory": {
                    "enabled": True,
                }
            }
            with open(config_json_path, "w") as f:
                json.dump(config_data, f)
            cfg = load_config(config_json_path)

            # ---------------------------------------------------
            # -------------- Execute pipeline.py ----------------
            # ---------------------------------------------------
            status_text = st.empty()

            def update_ui(msg):
                status_text.info(msg)

            with st.spinner('Reconstructing tree crowns and trunks...'):
                seg_time, clean_time, mesh_time, precision, recall, f1_score = run_pipeline(cfg, polygon=Path(geojson_path), progress_callback=update_ui)
            execution_times['4. Tree Segmentation'] = seg_time
            execution_times['5. Filter & Clean'] = clean_time
            execution_times['6. Reconstruct 3D Meshes'] = mesh_time

            scores['Precision'] = precision
            scores['Recall'] = recall
            scores['F1 Score'] = f1_score

            scores_data = {type: value for type, value in scores.items()}
            with open(out_dir / "scores.json", "w") as f:
                json.dump(scores_data, f, indent=4)

            status_text.empty()
            st.success(f"Tree reconstruction complete! Download output files using the buttons")

            total_time = sum(execution_times.values())

            json_data = {step: round(time, 2) for step, time in execution_times.items()}
            json_data["Total Execution Time"] = round(total_time, 2)

            with open(out_dir / "time-summary.json", "w") as f:
                json.dump(json_data, f, indent=4)

            st.session_state.out_dir = out_dir

            st.subheader("Execution Time Summary")
            # Format the dictionary for a clean Streamlit table
            summary_data = {
                "Pipeline Step": list(execution_times.keys()) + ["Total Execution Time"],
                "Duration (Seconds)": [
                    "Not enabled" if k == '3. Clip & Merge 3D buildings' and v == 0
                    else str(round(v, 2))
                    for k, v in execution_times.items()
                ] + [str(round(total_time, 2))]
            }
            st.table(pd.DataFrame(summary_data))

            if not precision is None or recall is None or f1_score is None:
                st.subheader("Precision, Recall and F1-Score")
                scores_data = {
                    "Type": list(scores.keys()),
                    "Value": [v for v in scores.values()]
                }

        except Exception as e:
            st.error(f"An error occurred while running the pipeline: {e}")


if st.session_state.out_dir:
    out_dir = st.session_state.out_dir
    # Expose the resulting .obj file for download
    generated_objs = list(out_dir.glob("*.obj"))
    output_crown_obj_path = out_dir / f"merged_lod{int(lod_value)}.1_crown.obj"
    output_trunk_obj_path = out_dir / f"merged_lod{int(lod_value)}.1_trunk.obj"
    output_bag_path = out_dir / "buildings_obj.obj"

    if generated_objs:
        st.subheader("**Download meshes**")
        for obj_path in generated_objs:
            if obj_path == output_crown_obj_path:
                label = "Download Meshed Trees (.obj)"
            elif obj_path == output_trunk_obj_path:
                label = "Download Trunk Geometries (.obj)"
            elif obj_path == output_bag_path:
                label = "Download 3DBAG Geometries (.obj)"
            else:
                label = f"Download {obj_path.name}"
            with open(obj_path, "rb") as file:
                st.download_button(
                    label=label,
                    data=file,
                    mime="model/obj",
                    key=obj_path.name
                )
        st.subheader("**3D Preview**")
        with st.spinner("Loading 3D viewer..."):
            render_3d_objs(generated_objs)

    else:
        st.warning(f"Pipeline finished, but no .obj files were found.")