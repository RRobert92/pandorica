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
from scipy.linalg import logm
from scipy.optimize import least_squares

from pandorica.stitch.matching.mt_transform import (
    fit_rigid_transform_2d,
)

Pose = Dict[str, float]  # {Angle (deg), Tx, Ty, Scale} + optional linear part L00..L11
Interface = Tuple[np.ndarray, np.ndarray]  # (A_k on section k, B_k on section k+1)

# The 2x2 linear part L = [[L00, L01], [L10, L11]] (so x' = L @ x + t) is the SOURCE OF
# TRUTH for apply/compose/invert: it composes EXACTLY (matrix multiply) and carries
# anisotropy + shear, neither of which a single scalar Scale can. Angle and Scale are kept
# as the derived polar-decomposition VIEW (Angle = polar rotation, Scale = sqrt|det|) so
# every existing pose["Angle"]/["Scale"] reader stays valid. A pose WITHOUT L* keys (a legacy
# 4-key dict, or a rigid/similarity constructor) is read as L = Scale * R(Angle) -- so old
# poses and old call sites keep working unchanged.
IDENTITY: Pose = {"Angle": 0.0, "Tx": 0.0, "Ty": 0.0, "Scale": 1.0,
                  "L00": 1.0, "L01": 0.0, "L10": 0.0, "L11": 1.0}


def _R(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def linear_part(pose: Pose) -> np.ndarray:
    """The 2x2 linear map ``L`` (``x' = L @ x + t``).

    The truth when the ``L*`` keys are present; otherwise reconstructed from the
    similarity view as ``Scale * R(Angle)`` (legacy 4-key / rigid poses).
    """
    if "L00" in pose:
        return np.array([[pose["L00"], pose["L01"]],
                         [pose["L10"], pose["L11"]]], float)
    return float(pose.get("Scale", 1.0)) * _R(pose["Angle"])


def _polar_angle_scale(L: np.ndarray) -> Tuple[float, float]:
    """Derived similarity VIEW of a 2x2: ``(polar-rotation angle deg, sqrt|det| scale)``.

    Lossy by design — anisotropy/shear cannot be captured by one angle + one scale; the
    full geometry lives in the ``L*`` keys. This view exists only so back-compat readers
    (logging, the MT angle cross-check) get a representative rotation and magnitude.
    """
    U, _s, Vt = np.linalg.svd(L)
    Rp = U @ Vt
    angle = float(np.degrees(np.arctan2(Rp[1, 0], Rp[0, 0])))
    scale = float(np.sqrt(abs(np.linalg.det(L))))
    return angle, scale


def pose_from_matrix(L: np.ndarray, t: np.ndarray) -> Pose:
    """Build a Pose from a 2x2 linear map ``L`` and translation ``t`` (the affine truth).

    ``Angle``/``Scale`` are filled as the derived similarity view for back-compat.
    """
    L = np.asarray(L, float)
    t = np.asarray(t, float)
    angle, scale = _polar_angle_scale(L)
    return {"Angle": angle, "Tx": float(t[0]), "Ty": float(t[1]), "Scale": scale,
            "L00": float(L[0, 0]), "L01": float(L[0, 1]),
            "L10": float(L[1, 0]), "L11": float(L[1, 1])}


def make_pose(angle: float = 0.0, tx: float = 0.0, ty: float = 0.0,
              scale: float = 1.0) -> Pose:
    """Construct a similarity Pose (rotation + isotropic scale + translation) with a
    consistent ``L`` populated. The drop-in replacement for the old 4-key dict literals."""
    return pose_from_matrix(scale * _R(angle), np.array([tx, ty], float))


def apply_pose(pose: Pose, xy: np.ndarray) -> np.ndarray:
    """Map ``[m, 2]`` points through a pose: ``L @ x + t``."""
    L = linear_part(pose)
    return np.asarray(xy, float) @ L.T + np.array([pose["Tx"], pose["Ty"]])


def compose_poses(outer: Pose, inner: Pose) -> Pose:
    """Pose equivalent to applying ``inner`` then ``outer`` (exact: ``L_o @ L_i``)."""
    L_o, L_i = linear_part(outer), linear_part(inner)
    t_i = np.array([inner["Tx"], inner["Ty"]])
    t = L_o @ t_i + np.array([outer["Tx"], outer["Ty"]])
    return pose_from_matrix(L_o @ L_i, t)


def invert_pose(pose: Pose) -> Pose:
    """Analytic inverse of an affine pose (``x = P⁻¹(P(x))``)."""
    L = linear_part(pose)
    Li = np.linalg.inv(L)
    t = -Li @ np.array([pose["Tx"], pose["Ty"]])
    return pose_from_matrix(Li, t)


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
            rel = make_pose(ang, tx, ty, sc)
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


def _affine_tangent(row: np.ndarray) -> np.ndarray:
    """Map a raw affine param row ``[L00, L01, L10, L11, Tx, Ty]`` to its tangent
    ``[logm(L).ravel(), Tx, Ty]``.

    Smoothness is measured here rather than on raw ``L`` so that a constant-step
    affine chain is linear in the section index: ``logm`` turns a constant relative
    rotation/scale step ``M`` into a constant additive step ``logm(M)``, giving zero
    second-difference. On raw ``L`` a large constant rotation curves (``cos(kθ)``) and
    would be wrongly penalised. ``.real`` drops the tiny imaginary part ``logm`` can
    carry for near-rotations; reflections (det < 0) are out of scope for this soft prior.
    """
    L = np.array([[row[0], row[1]], [row[2], row[3]]], float)
    logL = np.real(logm(L))
    return np.array([logL[0, 0], logL[0, 1], logL[1, 0], logL[1, 1], row[4], row[5]])


def _param_width(allow_scale: bool, allow_affine: bool) -> int:
    """Optimiser params per section: 6 affine (raw L + t), 4 similarity, 3 rigid."""
    if allow_affine:
        return 6
    return 4 if allow_scale else 3


def _pack(poses: Sequence[Pose], allow_scale: bool, allow_affine: bool = False) -> np.ndarray:
    """Flatten poses 1..n-1 into an optimiser vector (section 0 is the gauge).

    Affine mode packs the raw linear part directly as ``[L00, L01, L10, L11, Tx, Ty]`` —
    no angle/scale parameterisation — so anisotropy and shear are first-class unknowns.
    """
    rows = []
    for p in poses[1:]:
        if allow_affine:
            L = linear_part(p)
            rows.append([L[0, 0], L[0, 1], L[1, 0], L[1, 1], p["Tx"], p["Ty"]])
            continue
        row = [np.deg2rad(p["Angle"]), p["Tx"], p["Ty"]]
        if allow_scale:
            row.append(np.log(p["Scale"]))
        rows.append(row)
    return np.array(rows, dtype=float).ravel()


def _unpack(
    vec: np.ndarray, n_sections: int, allow_scale: bool, allow_affine: bool = False
) -> List[Pose]:
    """Inverse of ``_pack`` — rebuild the pose list with section 0 = identity."""
    width = _param_width(allow_scale, allow_affine)
    poses = [dict(IDENTITY)]
    for k in range(n_sections - 1):
        seg = vec[k * width : (k + 1) * width]
        if allow_affine:
            L = np.array([[seg[0], seg[1]], [seg[2], seg[3]]], float)
            poses.append(pose_from_matrix(L, np.array([seg[4], seg[5]])))
            continue
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
    allow_affine: bool = False,
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
    :param allow_affine: refine a full per-section affine (raw 2x2 ``L`` + ``t``,
        6 DOF) instead of a similarity. Subsumes ``allow_scale`` and lets each
        section carry anisotropy ``(sx, sy)`` and shear. Existing callers that
        omit this flag keep the rigid/similarity behaviour exactly.
    :param lambda_scale: weight of the rotation prior. In similarity mode this is
        the scale→1 prior (active with ``allow_scale``). In affine mode it becomes
        the **pull-to-rotation** prior ``√λ·(LᵀL−I)`` per section, which is zero on
        any pure rotation and so penalises only the anisotropy/shear/scale a clean
        rotation chain does not need — the affine generalisation of scale→1.
    :param lambda_smooth: weight of the pose-smoothness (second-difference) prior;
        0 disables it. In affine mode the second difference is taken in the
        matrix-log tangent (``logm(L)`` ‖ ``t``) so a constant-step affine chain —
        including large constant rotations — has zero curvature and is unbiased.
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
        init_poses = greedy_chain(interfaces, allow_scale=allow_scale or allow_affine)
    p0 = _pack(init_poses, allow_scale, allow_affine)
    width = _param_width(allow_scale, allow_affine)
    if weights is None:
        weights = [1.0] * len(interfaces)
    sqrt_w = [float(np.sqrt(max(w, 0.0))) for w in weights]

    # Identity anchor row in param space: zeros for the similarity log-params,
    # the identity affine [1,0,0,1,0,0] for raw-L params.
    anchor = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) if allow_affine else np.zeros(width)

    def residuals(vec):
        poses = _unpack(vec, n_sections, allow_scale, allow_affine)
        parts = []
        for k, (A, B) in enumerate(interfaces):
            if len(A) == 0:
                continue
            r = (apply_pose(poses[k], A) - apply_pose(poses[k + 1], B)).ravel()
            parts.append(sqrt_w[k] * r)

        rows = np.vstack([anchor, vec.reshape(n_sections - 1, width)])
        if allow_affine:
            if lambda_scale > 0:  # pull-to-rotation: zero iff each L is orthogonal
                for seg in rows[1:]:
                    L = np.array([[seg[0], seg[1]], [seg[2], seg[3]]])
                    parts.append(np.sqrt(lambda_scale) * (L.T @ L - np.eye(2)).ravel())
            if lambda_smooth > 0 and n_sections >= 3:
                tan = np.array([_affine_tangent(r) for r in rows])
                second_diff = tan[2:] - 2.0 * tan[1:-1] + tan[:-2]
                parts.append(np.sqrt(lambda_smooth) * second_diff.ravel())
        else:
            if allow_scale and lambda_scale > 0:
                parts.append(np.sqrt(lambda_scale) * rows[1:, 3])  # log-scale -> 0
            if lambda_smooth > 0 and n_sections >= 3:
                second_diff = rows[2:] - 2.0 * rows[1:-1] + rows[:-2]
                parts.append(np.sqrt(lambda_smooth) * second_diff.ravel())

        return np.concatenate(parts) if parts else np.zeros(1)

    sol = least_squares(residuals, p0, method="trf")
    return _unpack(sol.x, n_sections, allow_scale, allow_affine)
