"""
mesh_extractor.py
=================
Core module for extracting and cleaning watertight 3D triangle meshes
from volumetric medical segmentation masks using Marching Cubes and Trimesh.
"""

from typing import Tuple, Optional
import numpy as np
from skimage import measure
import trimesh


def extract_and_clean_mesh(
    seg_volume: np.ndarray,
    label_id: int,
    voxel_spacing: Tuple[float, float, float],
    output_path: Optional[str] = None,
    smooth_iterations: int = 3,
    smooth_lambda: float = 0.5,
) -> Optional[trimesh.Trimesh]:
    """
    Extracts a watertight 3D mesh for a specific vertebra label using Marching Cubes.

    Parameters
    ----------
    seg_volume : np.ndarray
        3D numpy array representing the multi-class segmentation mask.
    label_id : int
        Integer label ID corresponding to the target anatomical vertebra.
    voxel_spacing : tuple of float
        (sx, sy, sz) physical spacing in millimeters along each axis.
    output_path : str, optional
        Destination filepath to export the reconstructed mesh as an STL file.
    smooth_iterations : int, default=3
        Number of Laplacian smoothing passes to eliminate voxel staircase artifacts.
    smooth_lambda : float, default=0.5
        Relaxation factor for Laplacian smoothing.

    Returns
    -------
    trimesh.Trimesh or None
        Cleaned, watertight 3D triangle mesh in millimeter coordinates.
    """
    binary_mask = (seg_volume == label_id).astype(np.float32)

    # Skip empty or negligible masks
    if binary_mask.sum() < 100:
        return None

    spacing_tuple = tuple(float(s) for s in voxel_spacing)

    # Marching Cubes isosurface extraction
    verts, faces, normals, _ = measure.marching_cubes(
        binary_mask,
        level=0.5,
        spacing=spacing_tuple
    )

    # Build Trimesh representation
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Post-processing: Laplacian smoothing (voxel de-stepping)
    if smooth_iterations > 0:
        mesh = trimesh.smoothing.filter_laplacian(
            mesh, lamb=smooth_lambda, iterations=smooth_iterations
        )

    # Clean non-manifold features, duplicate/degenerate faces
    mesh.process(validate=True)

    # Seal boundaries to ensure watertight manifold for volume calculation & 3D printing
    trimesh.repair.fill_holes(mesh)

    # Export to STL if path provided
    if output_path:
        mesh.export(output_path)

    return mesh
