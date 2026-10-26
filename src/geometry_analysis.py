"""
geometry_analysis.py
====================
Computational geometry module for calculating clinical morphometric properties
(Volume, Surface Area, Sphericity/Compactness) from 3D watertight bone meshes.
"""

from typing import Dict, Any
import numpy as np
import trimesh


def compute_mesh_metrics(mesh: trimesh.Trimesh, name: str = "") -> Dict[str, Any]:
    """
    Computes clinical morphometric properties for a given watertight 3D bone mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Reconstructed 3D triangle mesh.
    name : str, optional
        Anatomical identifier for the bone (e.g. 'L3', 'T10').

    Returns
    -------
    dict
        Dictionary containing vertices, faces, volume (mm³), surface area (mm²),
        compactness/sphericity, and watertight verification status.
    """
    volume = abs(mesh.volume)
    surface_area = mesh.area

    # Sphericity / Compactness metric: (36 * pi * V^2)^(1/3) / Area
    # A perfect sphere has compactness = 1.0; complex vertebrae typically have 0.45 - 0.55
    if surface_area > 0:
        compactness = (36.0 * np.pi * (volume ** 2)) ** (1.0 / 3.0) / surface_area
    else:
        compactness = 0.0

    return {
        "vertebra": name,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "volume_mm3": round(volume, 2),
        "surface_area_mm2": round(surface_area, 2),
        "compactness": round(compactness, 4),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
    }
