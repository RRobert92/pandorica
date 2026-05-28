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
Dense-intensity verification — "splines match, intensity verifies".

The MT-spline alignment can be confidently *wrong* (a spurious rotation the point
clouds happen to fit — the sec12→13 lesson, and the MaleMeiosis tangent misfire).
This independent layer checks the alignment against the actual IMAGE content: it
warps the moving boundary-face image by the proposed rotation and asks whether
that **improves image agreement with the reference** AND **beats the 180° flip**.
If intensity and splines disagree, the interface is flagged.

Image warping reuses ``applier`` (same pose convention as the whole pipeline), so
the rotation here is in the identical sense as ``rotation_search`` / the solver.

Most informative for large rotations (a 90° turn dramatically changes agreement);
for near-0° interfaces the test is weak and naturally returns "verified" (the
identity baseline already agrees) — it is a guard against gross errors, not a
sub-degree refiner.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pandorica.stitch.transform.applier import (
    make_inverse_map,
    warp_volume_slicewise,
)
from pandorica.stitch.transform.solver import Pose


def _center_rotation_pose(angle_deg: float, center: np.ndarray) -> Pose:
    """A pose rotating about ``center`` (x, y) by ``angle_deg`` (no translation)."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    t = center - R @ center
    return {
        "Angle": float(angle_deg),
        "Tx": float(t[0]),
        "Ty": float(t[1]),
        "Scale": 1.0,
    }


def _transform_image(img: np.ndarray, pose: Pose) -> np.ndarray:
    """Warp a 2-D image by ``pose`` (pixel units), reusing the applier."""
    inv = make_inverse_map(pose)
    out = warp_volume_slicewise(
        np.asarray(img, dtype=float)[None], inv, out_hw=img.shape, dtype=np.float32
    )
    return out[0]


def image_similarity(
    a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """Zero-mean normalised cross-correlation of two same-shape images (optionally masked)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if mask is not None:
        m = np.asarray(mask) > 0
        a, b = a[m], b[m]
    else:
        a, b = a.ravel(), b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


@dataclass
class IntensityVerification:
    """Result of verifying a rotation against image content."""

    score: float  # similarity at the proposed rotation
    flip_score: float  # similarity at the 180°-flipped rotation
    identity_score: float  # similarity with no rotation
    verified: bool


def verify_rotation(
    ref_image: np.ndarray,
    mov_image: np.ndarray,
    angle_deg: float,
    center: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    margin: float = 0.02,
) -> IntensityVerification:
    """
    Verify a spline-derived rotation against the boundary-face images.

    :param ref_image / mov_image: same-shape 2-D boundary-face images.
    :param angle_deg: the proposed rotation (mov→ref, same convention as the
        rest of the pipeline).
    :param center: rotation centre (x, y); default = image centre.
    :param mask: optional region mask (exclude frame / nucleus).
    :param margin: how much the rotation must beat the flip / identity to verify.
    :return: an ``IntensityVerification``; ``verified`` iff the rotation agrees
        with the image at least as well as identity AND beats the 180° flip.
    """
    mov_image = np.asarray(mov_image, dtype=float)
    if center is None:
        center = np.array([mov_image.shape[1] / 2.0, mov_image.shape[0] / 2.0])
    aligned = _transform_image(mov_image, _center_rotation_pose(angle_deg, center))
    flipped = _transform_image(
        mov_image, _center_rotation_pose(angle_deg + 180.0, center)
    )
    s = image_similarity(ref_image, aligned, mask)
    s_flip = image_similarity(ref_image, flipped, mask)
    s_id = image_similarity(ref_image, mov_image, mask)
    verified = (s >= s_flip + margin) and (s >= s_id - margin)
    return IntensityVerification(s, s_flip, s_id, verified)
