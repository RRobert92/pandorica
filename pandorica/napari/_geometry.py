#######################################################################
#  Pandorica - Analytical tools for cryo-electron microscopy          #
#                                                                     #
#  https://github.com/RRobert92                                       #
#                                                                     #
#  Robert Kiewisz                                                     #
#  PolyForm Noncommercial License 1.0.0 - see LICENSE                 #
#######################################################################
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Robert Kiewisz

"""
napari display glue: pose ↔ napari geometry and boundary-face images.

The stitching pipeline emits 2-D in-plane poses ``{Angle, Tx, Ty, Scale}`` on
``[id, x, y, z]`` graphs (physical units); napari shows arrays/points in
``(z, y, x)`` order. These helpers are used only by the napari widgets — they
apply a pose to a graph's XY, reorder coordinates for point layers, build the
napari affine that aligns a volume layer without resampling, pick per-section
overlay colors, and make a cheap 2-D face image per section for the GT view.
"""

from typing import List, Optional

import numpy as np

from pandorica.stitch.io import Dataset
from pandorica.stitch.transform.solver import Pose, apply_pose


def apply_pose_to_coords(pose: Pose, coords: np.ndarray) -> np.ndarray:
    """Return a copy of ``[N, 4]`` (or ``[N, 3]``) with the pose applied to XY."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return coords.copy()
    out = coords.copy()
    xcol = 1 if coords.shape[1] == 4 else 0
    out[:, xcol : xcol + 2] = apply_pose(pose, coords[:, xcol : xcol + 2])
    return out


def coords_to_points_zyx(coords: np.ndarray) -> np.ndarray:
    """``[N, 4]/[N, 3]`` graph → ``[N, 3]`` point coordinates in napari ``(z, y, x)``."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return np.empty((0, 3))
    xyz = coords[:, 1:4] if coords.shape[1] == 4 else coords
    return xyz[:, [2, 1, 0]]  # (x, y, z) -> (z, y, x)


def napari_affine(pose: Pose) -> np.ndarray:
    """
    4×4 napari affine (data ``(z, y, x)`` order) equivalent to ``apply_pose`` on XY.

    Used to align a section's **volume** image layer without resampling: set the
    image ``scale = pixel_size`` (voxel → physical) and this ``affine`` (physical
    pose). The Z axis is left identity — the warp is in-plane.
    """
    a = np.deg2rad(pose["Angle"])
    c, s, sc = np.cos(a), np.sin(a), pose["Scale"]
    m = np.eye(4)
    # apply_pose: x' = sc(cx - sy) + Tx ; y' = sc(sx + cy) + Ty.  Axes: (z, y, x).
    m[1, 1], m[1, 2], m[1, 3] = sc * c, sc * s, pose["Ty"]  # y'
    m[2, 1], m[2, 2], m[2, 3] = -sc * s, sc * c, pose["Tx"]  # x'
    return m


def napari_affine_2d(pose: Pose) -> np.ndarray:
    """3×3 napari affine (data ``(y, x)`` order) equivalent to ``apply_pose`` on XY.

    The 2-D analogue of :func:`napari_affine`, for driving a section's boundary-face
    **image** layer live in the GT recorder's 2-D view (set ``scale = pixel_size``,
    then this ``affine``).
    """
    a = np.deg2rad(pose["Angle"])
    c, s, sc = np.cos(a), np.sin(a), pose["Scale"]
    # apply_pose: x' = sc(cx - sy) + Tx ; y' = sc(sx + cy) + Ty.  Axes: (y, x).
    return np.array(
        [
            [sc * c, sc * s, pose["Ty"]],  # y'
            [-sc * s, sc * c, pose["Tx"]],  # x'
            [0.0, 0.0, 1.0],
        ]
    )


def boundary_face_images(
    dataset: Dataset, downscale: int = 8
) -> List[Optional[np.ndarray]]:
    """
    One 2-D representative face image per section (mean-Z projection), downsampled.

    A proxy for the pipeline's per-section boundary-face image (used for the A–P
    sign hint + intensity check). The pipeline expects a single image per section;
    a mean-Z projection of the (downsampled) volume is a robust, cheap stand-in at
    the validation stage. Sections without a volume contribute ``None``.
    """
    images: List[Optional[np.ndarray]] = []
    for s in dataset.sections:
        if not s.has_volume():
            images.append(None)
            continue
        vol = s.load_volume(downscale=downscale)
        images.append(vol.mean(axis=0).astype(np.float32))
        s.drop_volume()
    return images


# Distinct colors cycled per section for the overlay layers.
SECTION_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#76b7b2",
    "#edc948",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]


def section_color(index: int) -> str:
    return SECTION_COLORS[index % len(SECTION_COLORS)]
