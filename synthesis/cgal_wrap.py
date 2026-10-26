import numpy as np
from pathlib import Path
from CGAL import CGAL_Alpha_wrap_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3

root_dir = Path(__file__).resolve().parent.parent


def wrap_and_simplify(vertices, faces, alpha_ratio=5, offset_ratio=500):
    """
    Temporary-file bridge to CGAL Alpha Wrap.
    alpha_ratio and offset_ratio are relative to the crown's bounding-box diagonal.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)

    if len(vertices) < 4 or len(faces) < 4:
        raise ValueError("Mesh is too small to wrap")

    # Write a temporary OFF triangle mesh.
    # This is a simple first test; ensure face indices are valid.
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Face indices out of range")

    # Translate mesh near the origin for CGAL
    origin = vertices.mean(axis=0)
    vertices = vertices - origin

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.off"

        with open(input_path, "w") as f:
            f.write("OFF\n")
            f.write(f"{len(vertices)} {len(faces)} 0\n")
            for x, y, z in vertices:
                f.write(f"{x:.17g} {y:.17g} {z:.17g}\n")
            for a, b, c in faces:
                f.write(f"3 {a} {b} {c}\n")

        source = Polyhedron_3(str(input_path))
        wrapped = Polyhedron_3()

        extent = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        if extent <= 0:
            raise ValueError("Mesh has zero size")

        alpha = extent / alpha_ratio
        offset = extent / offset_ratio

        CGAL_Alpha_wrap_3.alpha_wrap_3(
            source, alpha, offset, wrapped
        )
        output_path = Path(f"{root_dir}/DATA/OUT/cgal_raw_output.off")
        wrapped.write_to_file(str(output_path))

        # Read the OFF result back into arrays.
        with open(output_path, "r") as f:
            lines = [line.strip() for line in f
                     if line.strip() and not line.lstrip().startswith("#")]

        # OFF files may include comments or a header on the counts line;
        # this basic parser expects the standard CGAL OFF output.
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
