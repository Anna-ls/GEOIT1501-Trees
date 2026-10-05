"""Parametric tree primitives for coarse levels of detail (LoD 1-2).

LoD 3 uses the marching-cubes canopy hull (see :mod:`tree4cfd.crown`). The
coarser levels approximate each crown with a simple watertight solid:

* **LoD 1** — a single block (cylinder) enclosing the whole tree.
* **LoD 2** — a species-driven crown primitive on a trunk:
    * ``ellipsoid``  — broadleaves / rounded crowns (default),
    * ``cone``       — conifers,
    * ``cylinder``   — palms / columnar forms.

Shapes are sized from the per-tree extents the pipeline already computes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .trunk import make_trunk_mesh

# Genus → crown shape (lowercased genus). Anything else falls back to ellipsoid.
_CONE_GENERA = {
    "pinus", "cupressus", "cedrus", "picea", "abies", "taxus", "thuja",
    "juniperus", "chamaecyparis", "cupressocyparis", "sequoia", "araucaria",
}
_CYLINDER_GENERA = {  # palms / strongly columnar
    "phoenix", "washingtonia", "trachycarpus", "chamaerops", "brahea", "jubaea",
}


def crown_shape_for_species(species: Optional[str] = None,
                            common_name: Optional[str] = None) -> str:
    """Map a species (or common name) to a crown primitive shape."""
    genus = (species or "").strip().lower().split(" ")[0]
    text = " ".join(t for t in (species, common_name) if t).lower()
    if genus in _CYLINDER_GENERA or "palm" in text or "palmera" in text:
        return "cylinder"
    if genus in _CONE_GENERA or "pino" in text or "cypress" in text or "ciprés" in text:
        return "cone"
    return "ellipsoid"


def cone_mesh(cx, cy, z0, z1, r, n_sides: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Closed cone: circular base at ``z0`` (radius ``r``), apex at ``z1``."""
    th = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    ring = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th), np.full(n_sides, z0)])
    verts = np.vstack([ring, [[cx, cy, z1]], [[cx, cy, z0]]])
    apex, base_c = n_sides, n_sides + 1
    faces = []
    for k in range(n_sides):
        j = (k + 1) % n_sides
        faces.append([k, j, apex])        # side (outward)
        faces.append([base_c, j, k])      # base (downward)
    return verts, np.array(faces, dtype=np.int32)


def ellipsoid_mesh(cx, cy, cz, a, b, c, n_lat: int = 8,
                   n_lon: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Closed UV ellipsoid centred at ``(cx, cy, cz)`` with radii ``a, b, c``."""
    rings = []
    verts = []
    for i in range(1, n_lat):                 # latitude rings (poles added below)
        theta = np.pi * i / n_lat
        zc, rr = np.cos(theta), np.sin(theta)
        ring = []
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            ring.append(len(verts))
            verts.append([cx + a * rr * np.cos(phi), cy + b * rr * np.sin(phi), cz + c * zc])
        rings.append(ring)
    top = len(verts); verts.append([cx, cy, cz + c])
    bot = len(verts); verts.append([cx, cy, cz - c])

    faces = []
    for i in range(len(rings) - 1):           # quad strips between rings
        up, dn = rings[i], rings[i + 1]
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            faces.append([up[j], up[j2], dn[j2]])
            faces.append([up[j], dn[j2], dn[j]])
    for j in range(n_lon):                     # pole caps
        j2 = (j + 1) % n_lon
        faces.append([top, rings[0][j2], rings[0][j]])
        faces.append([bot, rings[-1][j], rings[-1][j2]])
    return np.asarray(verts, dtype=float), np.array(faces, dtype=np.int32)


def crown_primitive(shape: str, cx, cy, z0, z1, dx, dy,
                    n_sides: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Build a crown solid of ``shape`` spanning ``z0``→``z1`` with plan size ``dx, dy``."""
    a, b = max(dx / 2, 0.5), max(dy / 2, 0.5)
    if shape == "cone":
        return cone_mesh(cx, cy, z0, z1, max(a, b), n_sides)
    if shape == "cylinder":
        r = (a + b) / 2
        return make_trunk_mesh(cx, cy, z0, z1, r, r, n_sides=n_sides)
    cz = (z0 + z1) / 2
    return ellipsoid_mesh(cx, cy, cz, a, b, max((z1 - z0) / 2, 0.5))


def block_mesh(cx, cy, z0, z1, dx, dy, n_sides: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """LoD-1 whole-tree block: a vertical cylinder enclosing the tree."""
    r = max(max(dx, dy) / 2, 0.5)
    return make_trunk_mesh(cx, cy, z0, z1, r, r, n_sides=n_sides)
