"""
visualizer.py
=============
Visualization module for rendering publication-quality 3D spine models,
orthogonal 2D cross-sections, and animated 360-degree orbital GIF assets.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
import trimesh

# Spinal region color palette
REGION_COLORS = {
    **{i: [0.20, 0.60, 0.90] for i in range(1, 8)},    # Cervical: Blue
    **{i: [0.18, 0.80, 0.44] for i in range(8, 20)},   # Thoracic: Emerald Green
    **{i: [0.91, 0.30, 0.24] for i in range(20, 26)},  # Lumbar:   Coral Red
}


def plot_orthogonal_slices(
    seg_volume: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5)
) -> None:
    """Plots Sagittal, Coronal, and Axial cross-sectional views of a 3D medical volume."""
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    mid = [s // 2 for s in seg_volume.shape]

    # Sagittal (Side Profile)
    axs[0].imshow(seg_volume[mid[0], :, :].T, cmap="tab20", origin="lower")
    axs[0].set_title("Sagittal View (Side Profile)", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Anterior-Posterior (Y)")
    axs[0].set_ylabel("Inferior-Superior (Z)")

    # Coronal (Frontal Profile)
    axs[1].imshow(seg_volume[:, mid[1], :].T, cmap="tab20", origin="lower")
    axs[1].set_title("Coronal View (Frontal Profile)", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Left-Right (X)")
    axs[1].set_ylabel("Inferior-Superior (Z)")

    # Axial (Top-Down Profile)
    axs[2].imshow(seg_volume[:, :, mid[2]], cmap="tab20", origin="lower")
    axs[2].set_title("Axial View (Top-Down Profile)", fontsize=12, fontweight="bold")
    axs[2].set_xlabel("Left-Right (X)")
    axs[2].set_ylabel("Anterior-Posterior (Y)")

    plt.suptitle("Spine Segmentation Mask — 3 Canonical Orthogonal Cross-Sections", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_single_mesh(
    mesh: trimesh.Trimesh,
    title: str = "3D Vertebra Mesh",
    color: str = "#3498db",
    output_path: Optional[str] = None
) -> None:
    """Renders a single 3D mesh in millimeter coordinates."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.92, facecolor=color, edgecolor="none")
    ax.add_collection3d(poly)

    v = mesh.vertices
    ax.set_xlim(v[:, 0].min(), v[:, 0].max())
    ax.set_ylim(v[:, 1].min(), v[:, 1].max())
    ax.set_zlim(v[:, 2].min(), v[:, 2].max())

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=22, azim=48)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_full_spine(
    meshes_dict: Dict[str, trimesh.Trimesh],
    label_map: Dict[int, str],
    title: str = "Full Spine 3D Reconstruction",
    output_path: Optional[str] = None
) -> None:
    """Assembles all reconstructed meshes into a unified multi-color 3D plot."""
    fig = plt.figure(figsize=(10, 14))
    ax = fig.add_subplot(111, projection="3d")

    name_to_label = {v: k for k, v in label_map.items()}

    for name, mesh in meshes_dict.items():
        label_id = name_to_label.get(name, 0)
        color = REGION_COLORS.get(label_id, [0.7, 0.7, 0.7])
        poly = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.88, facecolor=color, edgecolor="none")
        ax.add_collection3d(poly)

    all_verts = np.concatenate([m.vertices for m in meshes_dict.values()])
    ax.set_xlim(all_verts[:, 0].min(), all_verts[:, 0].max())
    ax.set_ylim(all_verts[:, 1].min(), all_verts[:, 1].max())
    ax.set_zlim(all_verts[:, 2].min(), all_verts[:, 2].max())

    ax.set_xlabel("X (mm)", fontsize=11)
    ax.set_ylabel("Y (mm)", fontsize=11)
    ax.set_zlabel("Z (mm)", fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold")

    legend_elements = [
        Patch(facecolor=[0.18, 0.80, 0.44], label="Thoracic Vertebrae (T1–T12)"),
        Patch(facecolor=[0.91, 0.30, 0.24], label="Lumbar Vertebrae (L1–L5)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)
    ax.view_init(elev=18, azim=42)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def generate_rotating_gif(
    meshes_dict: Dict[str, trimesh.Trimesh],
    label_map: Dict[int, str],
    output_path: str = "spine_rotating.gif",
    fps: int = 14,
    dpi: int = 110
) -> None:
    """Generates a 360-degree orbital rotation animation saved as a GIF."""
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(111, projection="3d")

    name_to_label = {v: k for k, v in label_map.items()}

    for name, mesh in meshes_dict.items():
        label_id = name_to_label.get(name, 0)
        color = REGION_COLORS.get(label_id, [0.7, 0.7, 0.7])
        poly = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.88, facecolor=color, edgecolor="none")
        ax.add_collection3d(poly)

    all_verts = np.concatenate([m.vertices for m in meshes_dict.values()])
    ax.set_xlim(all_verts[:, 0].min(), all_verts[:, 0].max())
    ax.set_ylim(all_verts[:, 1].min(), all_verts[:, 1].max())
    ax.set_zlim(all_verts[:, 2].min(), all_verts[:, 2].max())
    ax.set_axis_off()

    def rotate(angle):
        ax.view_init(elev=15, azim=angle)
        return []

    anim = animation.FuncAnimation(
        fig,
        rotate,
        frames=np.arange(0, 360, 4),
        interval=70,
        blit=False
    )
    anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close()
