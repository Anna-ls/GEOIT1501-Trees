import easy3d
import numpy as np
import trimesh
import tempfile
import os

def simplify_geometry(verts: np.ndarray, faces: np.ndarray, target_ratio: float = 0.5):
    if len(faces) == 0:
        return verts, faces

    cleaner = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    cleaner.merge_vertices(merge_tex=False, merge_norm=False)
    verts = cleaner.vertices
    faces = cleaner.faces

    fd, tmp_path = tempfile.mkstemp(suffix='.obj')
    os.close(fd)

    try:
        # Export the cleaned mesh to the temp file
        cleaner.export(tmp_path)

        # Let easy3d read the mesh directly
        mesh = easy3d.SurfaceMeshIO.load(tmp_path)

        # 3. Simplify
        target_faces = max(4, int(len(cleaner.faces) * target_ratio))
        simplifier = easy3d.SurfaceMeshSimplification(mesh)
        simplifier.simplify(target_faces)

        # 4. Extract the simplified geometry
        simp_verts = []
        v_map = {}
        for i, v in enumerate(mesh.vertices()):
            pt = mesh.position(v)
            simp_verts.append([pt.x, pt.y, pt.z])
            v_map[v.idx()] = i

        simp_faces = []
        for f in mesh.faces():
            try:
                simp_faces.append([v_map[v.idx()] for v in mesh.vertices(f)])
            except KeyError:
                pass

        return np.array(simp_verts), np.array(simp_faces)

    finally:
        # Clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


