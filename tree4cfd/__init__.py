"""tree4cfd — LiDAR point clouds to individual tree meshes for CFD.

Pipeline: filter high-vegetation points, clean (SOR + voxel downsample),
segment individual trees via a Canopy Height Model, then mesh each tree
crown (marching cubes) with an allometric trunk cone, writing one combined
``.obj`` per tile.

All inputs (paths and parameters) are supplied through a JSON config file;
see :mod:`tree4cfd.config` and ``config.json``.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
