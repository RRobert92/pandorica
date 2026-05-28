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
Coarse rigid bootstrap for serial-section stitching.

Supplies the rotation that manual landmarks used to provide in Amira — REQUIRED
in general (the example sec09-13 is an easy, near-registered outlier; raw data is
not). The method is landmark-free:

    1. Principal-axis estimate per section (PCA of the in-plane point cloud).
    2. Align the moving axis onto the fixed axis. The axis is sign-ambiguous, so
       both the base rotation and base+180° are tried (the "oriented" in oriented
       ICP).
    3. Each seed is refined by nearest-neighbour ICP (reusing the existing SVD
       Procrustes fit), and the lower-Chamfer-residual orientation wins.

Output is the ``(Angle, Tx, Ty, Scale)`` mov->ref convention of
``matching.mt_transform.fit_rigid_transform_2d`` (``ref = scale * R @ mov + t``),
so it drops straight into the existing transform machinery.

Caveat: point-only 180° disambiguation can be unreliable on near-symmetric
clouds; the hybrid coarse (global rotation search + image A–P-polarity
sign) is what resolves that.
"""

from typing import Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pandorica.stitch.matching.mt_transform import (
    fit_rigid_transform_2d,
)


def _rot(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _principal_angle(xy: np.ndarray) -> float:
    """Angle (deg) of the dominant principal axis of an ``[M, 2]`` cloud."""
    c = xy - xy.mean(axis=0)
    cov = c.T @ c
    _, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
    v = vecs[:, -1]  # largest-eigenvalue direction
    return float(np.degrees(np.arctan2(v[1], v[0])))


def apply_rigid_xy(xy: np.ndarray, angle_deg, tx, ty, scale) -> np.ndarray:
    """Apply a mov->ref rigid+scale transform to ``[M, 2]`` points."""
    R = _rot(angle_deg)
    return scale * (np.asarray(xy, float) @ R.T) + np.array([tx, ty])


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Mean nearest-neighbour distance from ``a`` to ``b`` (one-directional)."""
    nn = NearestNeighbors(n_neighbors=1).fit(b)
    d, _ = nn.kneighbors(a)
    return float(d.mean())


def _icp_nn(ref_xy, mov_xy, init, refine_iters, allow_scale):
    """
    Nearest-neighbour ICP from an initial transform.

    Each iteration re-derives correspondences from the current positions, then
    re-fits the full transform from the *original* mov points to the matched ref
    points (so errors don't compound across iterations).
    """
    angle, tx, ty, scale = init
    if refine_iters <= 0 or len(ref_xy) < 2:
        return angle, tx, ty, scale
    nn = NearestNeighbors(n_neighbors=1).fit(ref_xy)
    for _ in range(refine_iters):
        cur = apply_rigid_xy(mov_xy, angle, tx, ty, scale)
        _, idx = nn.kneighbors(cur)
        matched_ref = ref_xy[idx[:, 0]]
        angle, tx, ty, scale = fit_rigid_transform_2d(
            matched_ref, mov_xy, allow_scale=allow_scale
        )
    return angle, tx, ty, scale


def coarse_align(
    ref_xy: np.ndarray,
    mov_xy: np.ndarray,
    allow_scale: bool = False,
    refine_iters: int = 5,
) -> Dict:
    """
    Landmark-free coarse rigid alignment of ``mov_xy`` onto ``ref_xy``.

    :param ref_xy: ``[M, 2]`` fixed-section in-plane points.
    :param mov_xy: ``[K, 2]`` moving-section in-plane points.
    :param allow_scale: Also estimate an isotropic scale.
    :param refine_iters: Nearest-neighbour ICP iterations per orientation seed.
    :return: ``{Angle, Tx, Ty, Scale, residual}`` (mov->ref); ``residual`` is the
        winning Chamfer distance (useful as a coarse-quality signal).
    """
    ref_xy = np.asarray(ref_xy, dtype=float)
    mov_xy = np.asarray(mov_xy, dtype=float)
    if len(ref_xy) < 2 or len(mov_xy) < 2:
        return {"Angle": 0.0, "Tx": 0.0, "Ty": 0.0, "Scale": 1.0, "residual": np.inf}

    base = _principal_angle(ref_xy) - _principal_angle(mov_xy)
    ref_mean, mov_mean = ref_xy.mean(0), mov_xy.mean(0)
    if allow_scale:
        var_ref = np.sum((ref_xy - ref_mean) ** 2)
        var_mov = np.sum((mov_xy - mov_mean) ** 2)
        scale0 = float(np.sqrt(var_ref / var_mov)) if var_mov > 1e-8 else 1.0
    else:
        scale0 = 1.0

    best = None
    for rot in (base, base + 180.0):
        R = _rot(rot)
        t = ref_mean - scale0 * (R @ mov_mean)
        init = (rot, float(t[0]), float(t[1]), scale0)
        angle, tx, ty, scale = _icp_nn(ref_xy, mov_xy, init, refine_iters, allow_scale)
        resid = chamfer(apply_rigid_xy(mov_xy, angle, tx, ty, scale), ref_xy)
        if best is None or resid < best["residual"]:
            best = {
                "Angle": float(angle),
                "Tx": float(tx),
                "Ty": float(ty),
                "Scale": float(scale),
                "residual": resid,
            }
    return best
