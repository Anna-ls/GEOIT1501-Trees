import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from datetime import datetime
import json

from get_data import process_geojson
from tree4cfd.pipeline import run_pipeline
from tree4cfd.config import load_config

st.title("Urban Tree Reconstruction")
st.write("Draw a polygon on the map to define the region for AHN tile extraction and 3D tree meshing.")

# Adds option to either draw a polygon or upload one
input_method = st.radio("Choose input method:", ("Draw on Map", "Upload GeoJSON"))

st_data = None
uploaded_file = None

# ---------------------------------------------------------------
# --- Show either the map or the uploader based on the choice ---
# ---------------------------------------------------------------
if input_method == "Draw on Map":
    m = folium.Map(location=[51.9244, 4.4777], zoom_start=13)
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
            valid_input = True
        else:
            st.warning("Please upload a GeoJSON file before running the pipeline")

    if valid_input:
        try:
            # If clipped_output.laz exist, unlink it so it will overwrite
            laz_output = out_dir / "clipped_output.laz"
            if laz_output.exists():
                laz_output.unlink()

            # ---------------------------------------------------
            # ------------- Run data collection -----------------
            # ---------------------------------------------------
            with st.spinner('Downloading, clipping, and merging AHN tiles...'):
                process_geojson(geojson_path, buffer_distance=20.0, overwrite=False)

            # ---------------------------------------------------
            # ---- Generate the config file for the pipeline ----
            # ---------------------------------------------------
            config_json_path = out_dir / "temp_config.json"
            config_data = {
                "paths": {
                    "tiles_dir": str(out_dir),
                    "output_dir": str(out_dir)
                },
                "translation": [0.0, 0.0, 0.0],
                "lod": 3.0,
                "point_of_interest": [0.0, 0.0, 0.0],
                "influence_region": 1250,
                "filtering": {
                    "high_veg_classes": [1],
                    "multi_return_only": False
                },
                "cleaning": {
                    "sor_neighbors": 20,
                    "sor_std_ratio": 1.5,
                    "voxel_size": 0
                },
                "segmentation": {
                    "cell_size": 0.5,
                    "smooth_sigma": 1.5,
                    "min_height": 2.0,
                    "peak_min_dist": 3.0,
                    "min_tree_pts": 50
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
                    "iso_level": 0.15,
                    "min_component_frac": 0.3
                },
                "shape_filter": {
                    "enabled": False
                },
                "buildings": {
                    "enabled": False,
                    "buffer": 1
                },
                "inventory": {
                    "enabled": False,
                    "source": "ods",
                    "csv_path": "../DATA/inventory/trees.csv",
                    "match_dist": 3.0
                },
                "output": {
                    "merge_tiles": True,
                    "separate_crown_trunk": True
                }
            }
            with open(config_json_path, "w") as f:
                json.dump(config_data, f)
            cfg = load_config(config_json_path)

            # ---------------------------------------------------
            # -------------- Execute pipeline.py ----------------
            # ---------------------------------------------------
            with st.spinner('Reconstructing tree crowns and trunks...'):
                run_pipeline(cfg)

            st.success("Tree reconstruction complete!")

            # Expose the resulting .obj file for download
            output_obj_path = out_dir / "merged_lod2.obj"
            if output_obj_path.exists():
                with open(output_obj_path, "rb") as file:
                    st.download_button(
                        label="Download Meshed Trees (.obj)",
                        data=file,
                        file_name="merged_trees.obj",
                        mime="model/obj"
                    )
            else:
                st.warning(f"Pipeline finished, but {output_obj_path.name} was not found in {out_dir}.")

        except Exception as e:
            st.error(f"An error occurred while running the pipeline: {e}")
