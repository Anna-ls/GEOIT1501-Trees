import easy3d
import numpy as np
import trimesh

def simplify_geometry(verts: np.ndarray, faces: np.ndarray, target_ratio: float = 0.5):
    if len(faces) == 0:
        return verts, faces

    cleaner = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    cleaner.merge_vertices(merge_tex=False, merge_norm=False,
                           tol=1e-3)
    cleaner.update_faces(cleaner.nondegenerate_faces())

    verts = cleaner.vertices
    faces = cleaner.faces

    mesh = easy3d.SurfaceMesh()

    vertex_handles = []
    for v in verts:
        vertex_handles.append(mesh.add_vertex(easy3d.vec3(v[0], v[1], v[2])))

    for f in faces:
        mesh.add_triangle(vertex_handles[f[0]], vertex_handles[f[1]], vertex_handles[f[2]])

    target_faces = max(4, int(len(faces) * target_ratio))

    simplifier = easy3d.SurfaceMeshSimplification(mesh)

    simplifier.simplify(target_faces)

    simp_verts = []
    v_map = {}

    for i, v in enumerate(mesh.vertices()):
        pt = mesh.position(v)
        simp_verts.append([pt.x, pt.y, pt.z])
        v_map[v.idx()] = i

    simp_faces = []
    for f in mesh.faces():
        f_verts = [v_map[v.idx()] for v in mesh.vertices(f)]
        simp_faces.append(f_verts)

    return np.array(simp_verts), np.array(simp_faces)


