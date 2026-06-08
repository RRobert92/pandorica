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

from pandorica.stitch.dataset import Dataset
from pandorica.stitch.transform.solver import Pose, apply_pose, linear_part


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


def coords_to_paths_zyx(coords: np.ndarray) -> List[np.ndarray]:
    """``[N, 4]`` (id, x, y, z) graph → list of ``[Mi, 3]`` polylines in napari ``(z, y, x)``.

    Each output array is the ordered point sequence of one filament/segment.
    Order is preserved within a segment (the input row order is assumed to
    reflect the along-filament order, matching the convention used by
    ``pandorica.io.amira.read_segmented_points``). Single-point segments are
    dropped — a path needs at least 2 points to render as a line in napari.

    Used by the napari widget to render MT/spatial-graph filaments as splines
    via ``viewer.add_shapes(path)`` instead of unstructured points.
    """
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0 or coords.shape[1] < 4:
        return []
    ids = coords[:, 0]
    xyz = coords[:, 1:4]
    # Group by id while preserving in-segment order. np.unique with return_index
    # would re-sort by id; we want sorted-by-id but stable within each segment.
    sort_idx = np.argsort(ids, kind="stable")
    sorted_ids = ids[sort_idx]
    sorted_xyz = xyz[sort_idx]
    # Find segment boundaries
    boundaries = np.where(np.diff(sorted_ids) != 0)[0] + 1
    segments_xyz = np.split(sorted_xyz, boundaries)
    paths: List[np.ndarray] = []
    for seg in segments_xyz:
        if len(seg) < 2:
            continue
        # (x, y, z) -> (z, y, x) for napari
        paths.append(seg[:, [2, 1, 0]])
    return paths


def napari_affine(pose: Pose) -> np.ndarray:
    """
    4×4 napari affine (data ``(z, y, x)`` order) equivalent to ``apply_pose`` on XY.

    Used to align a section's **volume** image layer without resampling: set the
    image ``scale = pixel_size`` (voxel → physical) and this ``affine`` (physical
    pose). The Z axis is left identity — the warp is in-plane.

    The linear block is the pose's full 2×2 ``L`` (reordered to napari's (y, x)
    axes), so anisotropy/shear display correctly; for an isotropic pose ``L = Scale·R``
    and this reduces to the rotation+scale form.
    """
    L = linear_part(pose)
    m = np.eye(4)
    # apply_pose: x' = L00·x + L01·y + Tx ; y' = L10·x + L11·y + Ty.  Axes: (z, y, x).
    m[1, 1], m[1, 2], m[1, 3] = L[1, 1], L[1, 0], pose["Ty"]  # y'
    m[2, 1], m[2, 2], m[2, 3] = L[0, 1], L[0, 0], pose["Tx"]  # x'
    return m


def napari_affine_2d(pose: Pose) -> np.ndarray:
    """3×3 napari affine (data ``(y, x)`` order) equivalent to ``apply_pose`` on XY.

    The 2-D analogue of :func:`napari_affine`, for driving a section's boundary-face
    **image** layer live in the GT recorder's 2-D view (set ``scale = pixel_size``,
    then this ``affine``). The linear block is the pose's full 2×2 ``L`` (reordered to
    napari's (y, x) axes), so anisotropy/shear display correctly.
    """
    L = linear_part(pose)
    # apply_pose: x' = L00·x + L01·y + Tx ; y' = L10·x + L11·y + Ty.  Axes: (y, x).
    return np.array(
        [
            [L[1, 1], L[1, 0], pose["Ty"]],  # y'
            [L[0, 1], L[0, 0], pose["Tx"]],  # x'
            [0.0, 0.0, 1.0],
        ]
    )


# --------------------------------------------------------------------------- #
# Anisotropic (Sx, Sy) display helpers — GT recorder only
# --------------------------------------------------------------------------- #
# Plastic-section EM often suffers anisotropic compression along the knife's
# cutting axis (~5-15% typical). The CoarseGTWidget exposes Sx, Sy independently
# so the operator can record this. These helpers transform points and build
# napari affines for the *display* path — the pipeline's Pose type stays
# isotropic by design (see project_export_perf_landscape memory).
#
# Operation order, matching centroid_pose's semantics generalised:
#   x'' = R · diag(sx, sy) · (x − c) + c + (tx, ty)
# where c is the rotation/scaling centre and R is the CCW rotation by angle.


def apply_anisotropic_xy(
    points_xy: np.ndarray,
    angle_deg: float,
    tx: float,
    ty: float,
    sx: float,
    sy: float,
    center_xy: np.ndarray,
) -> np.ndarray:
    """
    Apply anisotropic scale + rotation about a centre + translation to ``[N, 2]`` XY.

    Drops to the same math as ``centroid_pose`` + ``apply_pose`` when ``sx == sy``.
    """
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    p = np.asarray(points_xy, dtype=float)
    if len(p) == 0:
        return p.copy()
    cx, cy = float(center_xy[0]), float(center_xy[1])
    x_local = p[:, 0] - cx
    y_local = p[:, 1] - cy
    x_out = c * sx * x_local - s * sy * y_local + cx + tx
    y_out = s * sx * x_local + c * sy * y_local + cy + ty
    return np.column_stack([x_out, y_out])


def napari_affine_2d_anisotropic(
    angle_deg: float,
    tx: float,
    ty: float,
    sx: float,
    sy: float,
    center_xy: np.ndarray,
) -> np.ndarray:
    """3×3 napari affine (data ``(y, x)`` order) for anisotropic scale + rotation + translation.

    Equivalent to :func:`napari_affine_2d` of a ``centroid_pose`` when
    ``sx == sy``; with independent Sx, Sy the linear part becomes the general
    ``R · diag(sx, sy)``. Used to drive the GT recorder's moving-face image
    layer without resampling.
    """
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    # Translation in the output frame: centre stays put under the linear part,
    # then add (tx, ty). Derived from x_out / y_out above.
    tx_const = cx + tx - c * sx * cx + s * sy * cy
    ty_const = cy + ty - s * sx * cx - c * sy * cy
    return np.array(
        [
            [c * sy, s * sx, ty_const],  # y'
            [-s * sy, c * sx, tx_const],  # x'
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
