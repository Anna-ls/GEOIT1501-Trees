import numpy as np
from pathlib import Path
from CGAL import CGAL_Alpha_wrap_3
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
import trimesh
import tempfile

root_dir = Path(__file__).resolve().parent.parent


def wrap_and_simplify(vertices, alpha_ratio=5, offset_ratio=500):
    """
    Temporary-file bridge to CGAL Alpha Wrap.
    alpha_ratio and offset_ratio are relative to the crown's bounding-box diagonal.
    """
    vertices = np.asarray(vertices, dtype=float)

    if len(vertices) < 4:
        raise ValueError("Mesh is too small to wrap")

    origin = vertices.mean(axis=0)
    vertices = vertices - origin

    cgal_points = [Point_3(float(x), float(y), float(z)) for x, y, z in vertices]
    wrapped = Polyhedron_3()

    extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    if extent <= 0:
        raise ValueError("Point cluster has zero size")

    alpha = extent / alpha_ratio
    offset = extent / offset_ratio

    CGAL_Alpha_wrap_3.alpha_wrap_3(cgal_points, alpha, offset, wrapped)

    output_path = Path(f"{root_dir}/DATA/OUT/cgal_raw_output.off")
    wrapped.write_to_file(str(output_path))

    # Read the OFF result back into arrays.
    with open(output_path, "r") as f:
        lines = [line.strip() for line in f
                 if line.strip() and not line.lstrip().startswith("#")]

    if lines[0] != "OFF":
        raise ValueError("Unexpected OFF output format")

    nv, nf, _ = map(int, lines[1].split()[:3])
    out_vertices = np.array(
        [[float(v) for v in lines[i + 2].split()[:3]]
         for i in range(nv)],
        dtype=float,
    )

    out_faces = []
    for line in lines[nv + 2:nv + 2 + nf]:
        parts = list(map(int, line.split()))
        if parts[0] != 3:
            raise ValueError("Expected triangular output faces")
        out_faces.append(parts[1:4])
    out_vertices = out_vertices + origin

    return out_vertices, np.asarray(out_faces, dtype=np.int64)
