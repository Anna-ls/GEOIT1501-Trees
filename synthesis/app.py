import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import json

from get_data import process_geojson
from tree4cfd.pipeline import run_pipeline
from tree4cfd.config import load_config

st.title("Urban Tree Reconstruction")
st.write("Draw a polygon on the map to define the region for AHN tile extraction and 3D tree meshing.")

# STEP 1: Initialise the map
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

# STEP 2: Pipeline execution
if st.button("Run Reconstruction Pipeline"):
    if st_data["last_active_drawing"]:

        case_name = "target_region"
        geojson_path = root_dir / "DATA" / "OUT" / f"{case_name}.geojson"
        out_dir = root_dir / "DATA" / "OUT"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save the drawn polygon
        geojson_geometry = st_data["last_active_drawing"]
        geojson_feature = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature","properties": {},"geometry": geojson_geometry["geometry"],}
            ]
        }
        with open(geojson_path, 'w') as f:
            json.dump(geojson_feature, f)

        try:
            laz_output = out_dir / "clipped_output.laz"
            if laz_output.exists():
                laz_output.unlink()

            # Run data collection directly
            with st.spinner('Downloading, clipping, and merging AHN tiles...'):
                process_geojson(geojson_path, buffer_distance=20.0, overwrite=True)

            # Generate the configuration for the pipeline
            config_json_path = out_dir / "temp_config.json"
            config_data = {
                "paths": {
                    "tiles_dir": str(out_dir),
                    "output_dir": str(out_dir)
                },
                "translation": [0.0, 0.0, 0.0],
                "lod": 2.0,
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
                    "n_sides": 12,
                    "max_total_height": 0.0,
                    "max_trunk_height": 15.0
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

            # Execute pipeline.py
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

    else:
        st.warning("Please draw a polygon on the map before running the pipeline.")
