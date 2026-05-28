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
Global rigid-pose refinement for serial-section stitching.

Replaces the legacy greedy center-out accumulation. Greedy chaining compounds
per-interface error along the stack; the headline case is **scale**, which it
multiplies section-to-section and so drifts exponentially.

Formulation: each section k has an absolute pose ``P_k`` (rotation + translation,
optional isotropic scale); section 0 is the gauge anchor ``P_0 = I``. For each
interface k the matched endpoints ``A_k`` (on section k) and ``B_k`` (on section
k+1) should coincide in the global frame, so we minimise

    Σ_k ‖ P_k(A_k) − P_{k+1}(B_k) ‖²

over ``P_1 … P_{n-1}`` with ``scipy.optimize.least_squares``, initialised from the
greedy chain (a good, cheap starting point).

**Important (no loop closure, no multi-gap MTs):** the unregularised objective
*decouples* per interface — section k contributes its bottom endpoints to one
gap and its top endpoints to the other (different physical points, no shared
landmark), so ``{P_k} ↔ {relative transforms}`` is a free bijection and the
greedy Procrustes chain already attains the global optimum. Pose-only refinement
therefore equals greedy unless a coupling is added. We add it as **priors**:

    * a **scale→1 prior** that pulls each section's scale to 1 — directly counters
      the multiplicative scale drift greedy accumulates (the real defect here);
    * an optional **pose-smoothness prior** (second-difference penalty) that pulls
      the pose trajectory toward constant inter-section motion, damping random
      far-section wander.

These couple the sections so the global solve genuinely differs from and beats
greedy on drift. They introduce a mild modelling bias (sections assumed to vary
smoothly / near unit scale); a constant-step chain satisfies them exactly, so
clean data is still recovered without bias.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from pandorica.stitch.matching.mt_transform import (
    fit_rigid_transform_2d,
)

Pose = Dict[str, float]  # {Angle (deg), Tx, Ty, Scale}
Interface = Tuple[np.ndarray, np.ndarray]  # (A_k on section k, B_k on section k+1)

IDENTITY: Pose = {"Angle": 0.0, "Tx": 0.0, "Ty": 0.0, "Scale": 1.0}


def _R(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def apply_pose(pose: Pose, xy: np.ndarray) -> np.ndarray:
    """Map ``[m, 2]`` points through a pose: ``scale * R @ x + t``."""
    return pose["Scale"] * (np.asarray(xy, float) @ _R(pose["Angle"]).T) + np.array(
        [pose["Tx"], pose["Ty"]]
    )


def compose_poses(outer: Pose, inner: Pose) -> Pose:
    """Pose equivalent to applying ``inner`` then ``outer``."""
    R_o = _R(outer["Angle"])
    t_i = np.array([inner["Tx"], inner["Ty"]])
    t = outer["Scale"] * (R_o @ t_i) + np.array([outer["Tx"], outer["Ty"]])
    return {
        "Angle": outer["Angle"] + inner["Angle"],
        "Tx": float(t[0]),
        "Ty": float(t[1]),
        "Scale": outer["Scale"] * inner["Scale"],
    }


def invert_pose(pose: Pose) -> Pose:
    """Analytic inverse of a rigid+scale pose (``x = P⁻¹(P(x))``)."""
    Ri = _R(pose["Angle"]).T
    inv_s = 1.0 / pose["Scale"]
    t = inv_s * (Ri @ np.array([pose["Tx"], pose["Ty"]]))
    return {
        "Angle": -pose["Angle"],
        "Tx": float(-t[0]),
        "Ty": float(-t[1]),
        "Scale": inv_s,
    }


def greedy_chain(
    interfaces: Sequence[Interface], allow_scale: bool = False
) -> List[Pose]:
    """
    Greedy baseline: chain per-interface rigid fits (the legacy accumulation, but on
    raw correspondences). Each interface fits ``B_k -> A_k`` and composes onto the
    running pose. This is the drift-prone reference the global solve must beat.
    """
    poses = [dict(IDENTITY)]
    for A, B in interfaces:
        if len(A) >= 2:
            ang, tx, ty, sc = fit_rigid_transform_2d(A, B, allow_scale=allow_scale)
            rel = {"Angle": ang, "Tx": tx, "Ty": ty, "Scale": sc}
        else:
            rel = dict(IDENTITY)
        poses.append(compose_poses(poses[-1], rel))
    return poses


def total_residual(poses: Sequence[Pose], interfaces: Sequence[Interface]) -> float:
    """RMS cross-interface endpoint discrepancy under the given poses."""
    sq, count = 0.0, 0
    for k, (A, B) in enumerate(interfaces):
        if len(A) == 0:
            continue
        diff = apply_pose(poses[k], A) - apply_pose(poses[k + 1], B)
        sq += float((diff**2).sum())
        count += diff.shape[0]
    return float(np.sqrt(sq / count)) if count else 0.0


def _pack(poses: Sequence[Pose], allow_scale: bool) -> np.ndarray:
    """Flatten poses 1..n-1 into an optimiser vector (section 0 is the gauge)."""
    rows = []
    for p in poses[1:]:
        row = [np.deg2rad(p["Angle"]), p["Tx"], p["Ty"]]
        if allow_scale:
            row.append(np.log(p["Scale"]))
        rows.append(row)
    return np.array(rows, dtype=float).ravel()


def _unpack(vec: np.ndarray, n_sections: int, allow_scale: bool) -> List[Pose]:
    """Inverse of ``_pack`` — rebuild the pose list with section 0 = identity."""
    width = 4 if allow_scale else 3
    poses = [dict(IDENTITY)]
    for k in range(n_sections - 1):
        seg = vec[k * width : (k + 1) * width]
        poses.append(
            {
                "Angle": float(np.degrees(seg[0])),
                "Tx": float(seg[1]),
                "Ty": float(seg[2]),
                "Scale": float(np.exp(seg[3])) if allow_scale else 1.0,
            }
        )
    return poses


def global_pose_refine(
    interfaces: Sequence[Interface],
    init_poses: Sequence[Pose] = None,
    allow_scale: bool = False,
    lambda_scale: float = 1.0,
    lambda_smooth: float = 0.0,
    weights: Sequence[float] = None,
) -> List[Pose]:
    """
    Jointly refine absolute section poses from matched cross-interface endpoints,
    with priors that couple the sections (see module docstring).

    :param interfaces: list of ``(A_k, B_k)`` for k = 0 … n-2, where ``A_k`` are
        matched endpoints on section k and ``B_k`` the partners on section k+1,
        each ``[m, 2]`` in that section's own coordinates.
    :param init_poses: optional initial absolute poses (default: greedy chain).
        Section 0 is always forced to identity (gauge anchor).
    :param allow_scale: also refine a per-section isotropic scale.
    :param lambda_scale: weight of the scale→1 prior (only active with
        ``allow_scale``); counters multiplicative scale drift.
    :param lambda_smooth: weight of the pose-smoothness (second-difference) prior;
        0 disables it.
    :param weights: optional per-interface confidence weights. A low-confidence
        interface (sparse / ambiguous matches) then imposes its transform less
        strongly, letting the smoothness prior and well-determined neighbours
        dominate — the cure for a single noisy gap forcing a spurious rotation.
        Each interface residual is scaled by √w.
    :return: list of ``n = len(interfaces) + 1`` absolute poses (section 0 = I).
    """
    n_sections = len(interfaces) + 1
    if n_sections < 2:
        return [dict(IDENTITY)] * n_sections

    if init_poses is None:
        init_poses = greedy_chain(interfaces, allow_scale=allow_scale)
    p0 = _pack(init_poses, allow_scale)
    width = 4 if allow_scale else 3
    if weights is None:
        weights = [1.0] * len(interfaces)
    sqrt_w = [float(np.sqrt(max(w, 0.0))) for w in weights]

    def residuals(vec):
        poses = _unpack(vec, n_sections, allow_scale)
        parts = []
        for k, (A, B) in enumerate(interfaces):
            if len(A) == 0:
                continue
            r = (apply_pose(poses[k], A) - apply_pose(poses[k + 1], B)).ravel()
            parts.append(sqrt_w[k] * r)

        # Priors operate on the raw param rows (section 0 = identity = zeros).
        rows = np.vstack([np.zeros(width), vec.reshape(n_sections - 1, width)])
        if allow_scale and lambda_scale > 0:
            parts.append(np.sqrt(lambda_scale) * rows[1:, 3])  # log-scale -> 0
        if lambda_smooth > 0 and n_sections >= 3:
            second_diff = rows[2:] - 2.0 * rows[1:-1] + rows[:-2]
            parts.append(np.sqrt(lambda_smooth) * second_diff.ravel())

        return np.concatenate(parts) if parts else np.zeros(1)

    sol = least_squares(residuals, p0, method="trf")
    return _unpack(sol.x, n_sections, allow_scale)
