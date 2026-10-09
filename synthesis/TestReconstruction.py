import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tree4cfd.cleaning import remove_outliers_sor, voxel_downsample
from tree4cfd.config import Config
from tree4cfd.crown import kept_component_mask, segment_to_marching_cubes
from tree4cfd.io import read_las
from tree4cfd.primitives import block_mesh, crown_primitive
from tree4cfd.trunk import estimate_trunk_params, make_trunk_mesh

from tqdm import tqdm

import time


def parse_lod(lod):
    # print(f"Requested LOD: {lod}")
    value = float(lod)
    base = int(value)
    with_trunk = (value - base) > 0.05
    return base, with_trunk


def reconstruct_tree(laz_path: str | Path, cfg: Config, number_of_runs: int = 1, output_file: str | Path = None, t0 = None):
    """
    Reconstruct one tree from a LAZ containing only that tree.

    Returns:
        crown_vertices, crown_faces, trunk_vertices, trunk_faces
    """

    laz_path = Path(laz_path)

    # ---------------------------------------------------------
    # 1. Read the tree
    # ---------------------------------------------------------

    las = read_las(laz_path)

    pts = np.column_stack((
        las.x,
        las.y,
        las.z,
    )).astype(np.float64)

    # print(f"{laz_path.name}: {len(pts):,} points")

    if len(pts) < 100:
        print("Too few points.")
        return None

    # ---------------------------------------------------------
    # 2. Clean
    # ---------------------------------------------------------

    if cfg.cleaning.sor_neighbors > 0:
        pts = remove_outliers_sor(
            pts,
            cfg.cleaning.sor_neighbors,
            cfg.cleaning.sor_std_ratio,
        )

    if cfg.cleaning.voxel_size > 0:
        pts = voxel_downsample(
            pts,
            cfg.cleaning.voxel_size,
        )

    print(f"After cleaning: {len(pts):,} points")

    # Apply the same coordinate translation used elsewhere.
    pts -= np.asarray(cfg.translation)

    base_lod, with_trunk = parse_lod(cfg.lod)
    print(f"Base LOD: {base_lod}, with trunk: {with_trunk}")

    print(f"Reconstructing {number_of_runs} times for {laz_path.name}...")
    print(f"Points: {len(pts):,}")
    for i in tqdm(range(number_of_runs)):
        # ---------------------------------------------------------
        # 3. Reconstruct crown
        # ---------------------------------------------------------
        if i%1000 == 0 and i > 0:
            with open(output_file, "a") as f:
                f.write(f"{time.perf_counter() - t0:.2f}\n")
        base_lod, with_trunk = parse_lod(cfg.lod)

        if base_lod <= 2:

            # Remove small disconnected components before
            # estimating the crown dimensions.
            mask = kept_component_mask(
                pts,
                cfg.crown.voxel_size,
                cfg.crown.sigma,
                cfg.crown.iso_level,
                cfg.crown.min_component_frac,
            )

            crown_pts = pts[mask] if mask.sum() >= 10 else pts

            cx = float(crown_pts[:, 0].mean())
            cy = float(crown_pts[:, 1].mean())

            dx = float(np.ptp(crown_pts[:, 0]))
            dy = float(np.ptp(crown_pts[:, 1]))

            z_min = float(crown_pts[:, 2].min())
            z_max = float(crown_pts[:, 2].max())

            if base_lod == 1:

                crown_v, crown_f = block_mesh(
                    cx,
                    cy,
                    z_min,
                    z_max,
                    dx,
                    dy,
                    n_sides=cfg.trunk.n_sides,
                )

            else:

                crown_v, crown_f = crown_primitive(
                    "ellipsoid",
                    cx,
                    cy,
                    z_min,
                    z_max,
                    dx,
                    dy,
                    n_sides=cfg.trunk.n_sides,
                )

        else:

            crown_v, crown_f = segment_to_marching_cubes(
                pts,
                voxel_size=cfg.crown.voxel_size,
                sigma=cfg.crown.sigma,
                iso_level=cfg.crown.iso_level,
                min_component_frac=cfg.crown.min_component_frac,
            )

        # ---------------------------------------------------------
        # 4. Trunk
        # ---------------------------------------------------------

        # if with_trunk:
        #     # This is the only part that still depends on how
        #     # estimate_trunk_params() obtains ground elevation.
        #     #
        #     # For an isolated tree, this should be adapted to use
        #     # the lowest/base portion of `pts` instead of a DTM.

        #     raise NotImplementedError(
        #         "Adapt estimate_trunk_params() for an isolated tree "
        #         "without a DTM."
        #     )

        trunk_v = np.zeros((0, 3))
        trunk_f = np.zeros((0, 3), dtype=int)

    return (
        crown_v,
        crown_f,
        trunk_v,
        trunk_f,
    )

def write_tree_mesh(mesh, output_path):
    """Write crown and trunk meshes to a single OBJ file."""

    crown_v, crown_f, trunk_v, trunk_f = mesh

    with open(output_path, "w") as f:

        # -----------------------------------------------------
        # Crown
        # -----------------------------------------------------

        for v in crown_v:
            f.write(
                f"v {v[0]:.6f} "
                f"{v[1]:.6f} "
                f"{v[2]:.6f}\n"
            )

        f.write("g crown\n")

        # OBJ uses 1-based vertex indexing
        for face in crown_f:
            f.write(
                f"f {face[0] + 1} "
                f"{face[1] + 1} "
                f"{face[2] + 1}\n"
            )

        # -----------------------------------------------------
        # Trunk
        # -----------------------------------------------------

        trunk_offset = len(crown_v)

        for v in trunk_v:
            f.write(
                f"v {v[0]:.6f} "
                f"{v[1]:.6f} "
                f"{v[2]:.6f}\n"
            )

        if len(trunk_v):
            f.write("g trunk\n")

            for face in trunk_f:
                f.write(
                    f"f {face[0] + trunk_offset + 1} "
                    f"{face[1] + trunk_offset + 1} "
                    f"{face[2] + trunk_offset + 1}\n"
                )

    print(f"  Mesh written to: {output_path}")

def main():

    number_of_runs = 10000
    LOD = [2.0, 3.0]

    

    output_dir = Path(
        r"data/out"
    )

    cfg = []
    for i in range(4):
        cfg.append(Config(f"data/in/Trees/config_tree{i + 1}.json"))

    laz_paths = [Path("data/in/Trees/tree1/Tree1.laz"),
                 Path("data/in/Trees/tree2/Tree2.laz"),
                 Path("data/in/Trees/tree3/Tree3.laz"),
                 Path("data/in/Trees/tree4/Tree4.laz")]
    for lod in LOD:
        t0 = time.perf_counter()
        output_file = Path(
                            f"data/out/reconstruction_times_lod{lod}.txt"
                        )
        with open(output_file, "w") as f:
            f.write("")
        for i in range(4):
            
            
            cfg[i].lod = lod
            laz_path = laz_paths[i]
            #print(f"Processing {laz_path}...")
            mesh = reconstruct_tree(laz_path, cfg[i], number_of_runs, output_file, t0)

            output_path = output_dir / f"{laz_path.stem}_reconstructed.obj"

            # write_tree_mesh(
            #     mesh,
            #     output_path,
            # )
        print(f"Total time for {number_of_runs} runs with LOD{lod}: {time.perf_counter() - t0:.2f} seconds")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    file1 = Path("data/out/reconstruction_times_lod2.0.txt")
    file2 = Path("data/out/reconstruction_times_lod3.0.txt")

    data1 = np.loadtxt(file1)
    data2 = np.loadtxt(file2)

    x1 = np.arange(1, len(data1) + 1)
    x2 = np.arange(1, len(data2) + 1)


    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        x1,
        data1,
        color="#00D9FF",
        linewidth=2.5,
        label=file1.stem,
    )

    ax.plot(
        x2,
        data2,
        color="#FF4D8D",
        linewidth=2.5,
        label=file2.stem,
    )


    # Points
    ax.scatter(x1, data1, color="#00D9FF", s=25, alpha=0.8)
    ax.scatter(x2, data2, color="#FF4D8D", s=25, alpha=0.8)


    # ---------------------------------------------------------
    # Styling
    # ---------------------------------------------------------

    ax.set_title(
        "Comparison",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )

    ax.set_xlabel("Observation (x1.000)", fontsize=13)
    ax.set_ylabel("Seconds", fontsize=13)

    ax.grid(
        True,
        color="white",
        alpha=0.08,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()
   

if __name__ == "__main__":
    main()