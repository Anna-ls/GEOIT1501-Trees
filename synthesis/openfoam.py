import pyvista as pv
import numpy as np
import gzip

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

pv.global_theme.anti_aliasing = 'ssaa'

def get_u_mag_array(folder_name, sample_surface):
    reader = pv.OpenFOAMReader(f"{root_dir}/DATA/IN/javaeiland/{folder_name}/case.foam")
    reader.enable_all_cell_arrays()
    if reader.time_values:
        reader.set_active_time_value(reader.time_values[-1])

    data = reader.read()
    internal_volume = data["internalMesh"].cell_data_to_point_data()
    resampled = sample_surface.sample(internal_volume)

    if "U" in resampled.point_data:
        return np.linalg.norm(resampled.point_data["U"], axis=1)
    else:
        return resampled.point_data[resampled.array_names[0]]

def create_morph_video():
    reader = pv.OpenFOAMReader(f"{root_dir}/DATA/IN/javaeiland/javaeiland_210_zomer/case.foam")
    reader.enable_all_cell_arrays()
    reader.enable_all_patch_arrays()

    if hasattr(reader, "reader") and hasattr(reader.reader, "SetReadZones"):
        reader.reader.SetReadZones(True)

    if reader.time_values:
        last_time = reader.time_values[-1]
        reader.set_active_time_value(last_time)

    data = reader.read()
    boundaries = data["boundary"]
    ground_patch = boundaries["ground_in"]
    brug_patch = boundaries["brug_dek"]

    ground_surface = ground_patch.merge(brug_patch)
    closed_ground = ground_surface.delaunay_2d(alpha=5.0)
    sample_surface = closed_ground.translate([0.0,0.0,1.75], inplace=False)

    u_no_trees = get_u_mag_array("javaeiland_210_no_trees", sample_surface)
    u_winter = get_u_mag_array("javaeiland_210_winter", sample_surface)
    u_zomer = get_u_mag_array("javaeiland_210_zomer", sample_surface)

    resampled_plane = sample_surface.copy()
    resampled_plane.point_data["U_mag"] = u_no_trees

    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

    plotter.add_mesh(
        resampled_plane,
        scalars="U_mag",
        cmap="jet",
        clim=[0.0, 10.0],
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "U (m/s)", "vertical": False, "position_x": 0.05,
            "position_y": 0.05, "width": 0.4, "height": 0.08,
            "title_font_size": 28, "label_font_size": 22, "color": "white", "fmt": "%.1f"
        },
    )

    for key in boundaries.keys():
        if "building" in key.lower():
            plotter.add_mesh(boundaries[key], color="#e0e0e0", lighting=True, smooth_shading=True)

    tree_actor = None
    cellzones_file = root_dir / "DATA/IN/javaeiland/javaeiland_210_zomer/constant/polyMesh/cellZones.gz"
    if cellzones_file.exists():
        cell_ids = []
        with gzip.open(cellzones_file, "rt", encoding="utf-8") as f:
            in_groencell = False
            in_list = False
            for line in f:
                line = line.strip()
                if "groencell" in line: in_groencell = True
                if in_groencell:
                    if line == "(" and not in_list:
                        in_list = True
                    elif line == ")" and in_list:
                        break
                    elif in_list:
                        try:
                            cell_ids.append(int(line))
                        except ValueError:
                            pass
        if cell_ids:
            tree_volume = data["internalMesh"].extract_cells(cell_ids)
            tree_actor = plotter.add_mesh(tree_volume, style="wireframe", color="black", opacity=0.0, line_width=0.5)

    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera_position = [(153.80, -502.03, 2615.57), (153.80, -502.03, 31.90), (0.0, 1.0, 0.0)]
    plotter.camera.parallel_scale = 150.0
    plotter.set_background("black")

    movie_path = root_dir / "DATA" / "OUT" / "CFD" / "movie-black.mp4"
    plotter.open_movie(str(movie_path), framerate=30)

    def morph_frames(u_start, u_end, tree_op_start, tree_op_end, duration_sec=2.0):
        frames = int(duration_sec * 30)
        for i in range(frames):
            frac = i / float(frames - 1)
            resampled_plane.point_data["U_mag"] = (1.0 - frac) * u_start + (frac * u_end)
            if tree_actor is not None:
                tree_actor.prop.opacity = (1.0 - frac) * tree_op_start + (frac * tree_op_end)
            plotter.write_frame()

    morph_frames(u_no_trees, u_no_trees, 0.0, 0.0, 1.0)
    morph_frames(u_no_trees, u_winter, 0.0, 0.1, 2.0)
    morph_frames(u_winter, u_winter, 0.1, 0.1, 1.0)
    morph_frames(u_winter, u_zomer, 0.1, 0.3, 2.0)
    morph_frames(u_zomer, u_zomer, 0.3, 0.3, 1.5)

    plotter.close()


def plot_simulation(folder_name, text_colour):
    reader = pv.OpenFOAMReader(f"{root_dir}/DATA/IN/javaeiland/{folder_name}/case.foam")

    reader.enable_all_cell_arrays()
    reader.enable_all_patch_arrays()

    if hasattr(reader, "reader") and hasattr(reader.reader, "SetReadZones"):
        reader.reader.SetReadZones(True)

    if reader.time_values:
        last_time = reader.time_values[-1]
        reader.set_active_time_value(last_time)

    data = reader.read()

    boundaries = data["boundary"]

    ground_patch = boundaries["ground_in"]
    brug_patch = boundaries["brug_dek"]

    ground_surface = ground_patch.merge(brug_patch)
    closed_ground = ground_surface.delaunay_2d(alpha=5.0)
    sample_surface = closed_ground.translate([0.0,0.0,1.75], inplace=False)

    internal_volume = data["internalMesh"].cell_data_to_point_data()
    resampled_plane = sample_surface.sample(internal_volume)

    if "U" in resampled_plane.point_data:
        U_vectors = resampled_plane.point_data["U"]
        resampled_plane.point_data["U_mag"] = np.linalg.norm(U_vectors, axis=1)
        scalar_field = "U_mag"
    else:
        scalar_field = resampled_plane.array_names[0]

    iso_levels = np.arange(3.0, 9.5, 2.0)
    contours = resampled_plane.contour(isosurfaces=iso_levels, scalars=scalar_field)

    plotter = pv.Plotter(off_screen=True, window_size=[3840, 2160])

    legend_format = {
        "title": "U (m/s)",
        "vertical": False,
        "position_x": 0.05,
        "position_y": 0.05,
        "width": 0.4,
        "height": 0.08,
        "title_font_size": 40,
        "label_font_size": 30,
        "color": text_colour,
        "fmt": "%.1f",
    }

    plotter.add_mesh(
        resampled_plane,
        scalars=scalar_field,
        cmap="jet",
        clim=[0.0, 10.0],
        show_scalar_bar=True,
        scalar_bar_args=legend_format,
    )

    if contours.n_cells > 0:
        plotter.add_mesh(
            contours,
            color="black",
            line_width=3.0,
        )

    for key in boundaries.keys():
        if "building" in key.lower():
            plotter.add_mesh(
                boundaries[key],
                color="#e0e0e0",
                lighting=True,
                smooth_shading=True,
            )

    internal_mesh = data["internalMesh"]

    cellzones_file = root_dir / f"DATA/IN/javaeiland/{folder_name}/constant/polyMesh/cellZones.gz"

    cell_ids = []

    if cellzones_file.exists():
        print("Starting tree cell extraction")
        with gzip.open(cellzones_file, "rt", encoding="utf-8") as f:
            in_groencell = False
            in_list = False

            for line in f:
                line = line.strip()

                if "groencell" in line:
                    in_groencell = True

                if in_groencell:
                    if line == "(" and not in_list:
                        in_list = True
                    elif line == ")" and in_list:
                        break
                    elif in_list:
                        try:
                            cell_ids.append(int(line))
                        except ValueError:
                            pass
        if cell_ids:
            print(f"Found {len(cell_ids)} tree cells")

            tree_volume = internal_mesh.extract_cells(cell_ids)

            plotter.add_mesh(
                tree_volume,
                style="wireframe",
                color="black",
                opacity=0.2,
                line_width=0.3,
            )

        else:
            print(f"No cell array found for {cell_ids}")

    fixed_camera = [(153.80, -502.03, 2615.57), (153.80, -502.03, 31.90), (0.0, 1.0, 0.0)]

    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.camera_position = fixed_camera
    plotter.set_background("white")

    plotter.camera.parallel_scale = 150.0

    output_dir = root_dir / "DATA" / "OUT" / "CFD" / f"{folder_name}_{text_colour}-text.png"

    plotter.screenshot(output_dir, transparent_background=True)

    plotter.close()

def main():
    # folder_names = ['javaeiland_210_winter', 'javaeiland_210_zomer', 'javaeiland_210_no_trees']
    #
    # for folder_name in folder_names:
    #     plot_simulation(folder_name, "white")
    #     plot_simulation(folder_name, "black")
    #
    create_morph_video()


if __name__ == "__main__":
    main()
