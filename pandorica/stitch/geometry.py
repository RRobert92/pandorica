#######################################################################
#  Serial Stitcher - An Automatic tool for tomograms stitching        #
#                                                                     #
#  https://github.com/RRobert92                                       #
#                                                                     #
#  Robert Kiewisz                                                     #
#  PolyForm Noncommercial License 1.0.0 - see LICENSE                 #
#######################################################################
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Robert Kiewisz

"""
Geometry helpers for the serial-section stitching pipeline.

The pipeline works on ``[id, x, y, z]`` graphs and 2-D in-plane poses
``{Angle, Tx, Ty, Scale}`` (physical units). This module:

* extracts the boundary-face endpoints used as cross-gap landmarks (top face =
  high-Z, bottom face = low-Z — the ``core`` registration convention),
* projects the boundary slices of a volume to a 2-D face image,
* builds a centroid-anchored pose for the manual GT recorder, and
* converts a physical-unit pose to the pixel-unit pose the volume applier needs.

The napari-only display glue (pose→napari affine, point reordering, overlay
colors) lives in ``napari_stitch._geometry``.
"""

from typing import Dict, List

import numpy as np

from pandorica.stitch.transform.scale import boundary_landmarks
from pandorica.stitch.transform.solver import Pose

# Intuitive Z-up face → extract_boundary_endpoints' (backwards) labels, mirroring
# core._FACE_LABEL: "top" = high-Z face, "bottom" = low-Z face.
_FACE_LABEL = {"top": "bottom", "bottom": "top"}


def face_endpoints(
    coords: np.ndarray, face: str, z_band_fraction: float = 0.15
) -> List[Dict]:
    """Boundary endpoints of the intuitive ``face`` ('top' = high-Z, 'bottom' = low-Z)."""
    if len(coords) == 0:
        return []
    return boundary_landmarks(coords, _FACE_LABEL[face], z_band_fraction)


def endpoints_xy(endpoints: List[Dict]) -> np.ndarray:
    """``[M, 2]`` XY positions of boundary endpoints (empty if none)."""
    return (
        np.array([e["pos"][:2] for e in endpoints], dtype=float)
        if endpoints
        else np.empty((0, 2))
    )


def pose_to_pixel(pose: Pose, pixel_size: float) -> Pose:
    """Convert a physical-unit pose to pixel units (rotation/scale unchanged)."""
    px = pixel_size if pixel_size else 1.0
    return {
        "Angle": pose["Angle"],
        "Tx": pose["Tx"] / px,
        "Ty": pose["Ty"] / px,
        "Scale": pose["Scale"],
    }


def zmax_face(
    volume: np.ndarray, face: str, n_slices: int = 10, invert_z: bool = False
) -> np.ndarray:
    """
    Z-max projection of the ``n_slices`` boundary slices of a ``[Z, Y, X]`` volume.

    With the Z-up convention (array index increasing with Z) the **top** face is the
    high-Z end (last slices) and the **bottom** face the low-Z end (first slices).
    ``invert_z`` swaps the two ends for stacks stored high-Z-first.

    :return: a ``[Y, X]`` maximum-intensity projection of the chosen face slab.
    """
    volume = np.asarray(volume)
    z = volume.shape[0]
    n = max(1, min(int(n_slices), z))
    high_is_top = not invert_z
    take_high = (face == "top") == high_is_top
    sl = volume[-n:] if take_high else volume[:n]
    return sl.max(axis=0)


def centroid_pose(
    angle_deg: float,
    tx: float,
    ty: float,
    center_xy: np.ndarray,
    scale: float = 1.0,
) -> Pose:
    """
    Pose that scales by ``scale`` and rotates by ``angle_deg`` about ``center_xy``,
    then translates by ``(tx, ty)``.

    Natural parametrisation for the GT recorder: rotating and scaling the moving
    face about its own centroid keeps it near the fixed face while the operator
    dials in angle / scale, then ``(tx, ty)`` nudges it into place. Equivalent to
    ``x' = scale·R·(x − c) + c + (tx, ty)`` — a centred similarity transform.
    Default ``scale=1.0`` preserves the prior signature (rigid-only callers
    don't need to opt in).
    """
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.asarray(center_xy, dtype=float)
    s = float(scale)
    t = c - s * (R @ c) + np.array([tx, ty], dtype=float)
    return {
        "Angle": float(angle_deg),
        "Tx": float(t[0]),
        "Ty": float(t[1]),
        "Scale": s,
    }
