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
Direction-aware endpoint matcher with outlier rejection and confidence.

This is the *enhanced Hungarian* matcher: it keeps a one-to-one Hungarian
assignment and adds three things on top of a plain assignment:

    * **ρ-normalised thresholds** — spatial gates are expressed in units of the
      median nearest-neighbour spacing ρ, not raw pixels, so they port
      across voxel sizes. (The legacy matcher hard-coded 500 px.)
    * **vMF direction term** — tangent agreement enters the cost as a von
      Mises-Fisher-style likelihood ``1 - |cos Δθ|`` (sign-agnostic, since MT
      polarity is ambiguous), rather than a linear angle penalty.
    * **outlier/duplicate rejection** — near-coincident endpoints are deduped
      and geometric outliers are dropped via a robust rigid-fit residual gate,
      *before* any warp. Clustered/duplicate correspondences were the Amira
      whirlpool trigger, so this is the single most important prevention input.

It also reports a **confidence** record (match fraction, mean cost, shift
coherence) for the QC gate to threshold.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from pandorica.stitch.matching.mt_transform import (
    fit_rigid_transform_2d,
)


def _vmf_direction_cost(d1: np.ndarray, d2: np.ndarray) -> float:
    """
    von Mises-Fisher-style direction cost for two (in-plane) tangents.

    Returns ``1 - |cos Δθ|`` in ``[0, 1]``: 0 when (anti-)parallel, 1 when
    perpendicular. Sign-agnostic because MT polarity is not meaningful here.
    """
    a, b = d1[:2], d2[:2]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return 1.0 - abs(cos_a)


def dedupe_endpoints(
    endpoints: List[Dict], rho: float, dup_frac: float = 0.1
) -> List[Dict]:
    """
    Greedily drop near-coincident endpoints (within ``dup_frac * rho``).

    Clustered/duplicate landmarks are the whirlpool trigger; removing them up
    front is W1's cheapest, highest-value step.
    """
    if not endpoints:
        return []
    thresh = dup_frac * rho
    kept: List[Dict] = []
    kept_xy: List[np.ndarray] = []
    for ep in endpoints:
        xy = ep["pos"][:2]
        if all(np.linalg.norm(xy - k) > thresh for k in kept_xy):
            kept.append(ep)
            kept_xy.append(xy)
    return kept


def _cost_matrix(ref, mov, rho, max_dist_rho, max_angle_deg, w_dist):
    """Pairwise cost; np.inf where the distance/angle hard gates are exceeded."""
    max_dist = max_dist_rho * rho
    n_ref, n_mov = len(ref), len(mov)
    cost = np.full((n_ref, n_mov), np.inf)
    for i, r in enumerate(ref):
        for j, m in enumerate(mov):
            d = np.linalg.norm(r["pos"][:2] - m["pos"][:2])
            if d > max_dist:
                continue
            dir_cost = _vmf_direction_cost(r["dir"], m["dir"])
            # |cos Δθ| <= cos(max_angle) ⇔ dir_cost >= 1 - cos(max_angle)
            if dir_cost > 1.0 - np.cos(np.deg2rad(max_angle_deg)):
                continue
            cost[i, j] = w_dist * (d / max_dist) + (1.0 - w_dist) * dir_cost
    return cost


def _assign(cost) -> List[Tuple[int, int, float]]:
    """Hungarian one-to-one assignment, keeping only finite-cost pairs."""
    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        return []
    big = finite.max() * 10 + 100
    filled = np.where(np.isfinite(cost), cost, big)
    row, col = linear_sum_assignment(filled)
    return [
        (int(r), int(c), float(cost[r, c]))
        for r, c in zip(row, col)
        if np.isfinite(cost[r, c])
    ]


def reject_outliers(ref_xy, mov_xy, rho, max_resid_rho=2.0, iters=2):
    """
    Robust geometric outlier rejection via the rigid-fit residual.

    Fits a rigid transform, drops matches whose residual exceeds
    ``max_resid_rho * rho``, and refits. Returns a boolean keep-mask.
    """
    keep = np.ones(len(ref_xy), dtype=bool)
    if len(ref_xy) < 3:
        return keep
    thresh = max_resid_rho * rho
    for _ in range(iters):
        if keep.sum() < 3:
            break
        a, tx, ty, sc = fit_rigid_transform_2d(ref_xy[keep], mov_xy[keep])
        theta = np.deg2rad(a)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pred = sc * (mov_xy @ R.T) + np.array([tx, ty])
        resid = np.linalg.norm(pred - ref_xy, axis=1)
        new_keep = resid <= thresh
        if new_keep.sum() == keep.sum() and np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return keep


def _confidence(ref_xy, mov_xy, costs, n_ref, n_mov, rho) -> Dict:
    """Confidence signals for the QC gate (all cheap, threshold downstream)."""
    n = len(ref_xy)
    denom = max(1, min(n_ref, n_mov))
    shifts = ref_xy - mov_xy
    incoherence = (
        float(np.linalg.norm(np.std(shifts, axis=0)) / rho) if n > 1 else np.inf
    )
    return {
        "n_matches": int(n),
        "match_fraction": float(n / denom),
        "mean_cost": float(np.mean(costs)) if n else np.inf,
        "shift_incoherence_rho": incoherence,
    }


def match_sections(
    ref_endpoints: List[Dict],
    mov_endpoints: List[Dict],
    rho: float,
    max_dist_rho: float = 5.0,
    max_angle_deg: float = 30.0,
    w_dist: float = 0.7,
    dup_frac: float = 0.1,
    max_resid_rho: float = 2.0,
):
    """
    Match two boundary-endpoint sets with ρ-scaled gates, vMF direction, and W1.

    :param ref_endpoints / mov_endpoints: lists of ``{id, pos, dir}`` (see
        ``extract_boundary_endpoints`` / ``scale.boundary_landmarks``).
    :param rho: median NN spacing of the section (length unit for all gates).
    :param max_dist_rho: max XY distance for a candidate pair, in ρ.
    :param max_angle_deg: max tangent angle difference (sign-agnostic).
    :param w_dist: weight of the distance term vs the direction term.
    :param dup_frac: dedupe radius as a fraction of ρ.
    :param max_resid_rho: outlier residual gate, in ρ.
    :return: ``(matches, ref_xy, mov_xy, confidence)`` where ``matches`` is a list
        of ``(ref_idx, mov_idx, cost)`` into the *deduped* endpoint lists.
    """
    ref = dedupe_endpoints(ref_endpoints, rho, dup_frac)
    mov = dedupe_endpoints(mov_endpoints, rho, dup_frac)
    empty = (np.empty((0, 2)), np.empty((0, 2)))
    if len(ref) == 0 or len(mov) == 0:
        return [], *empty, _confidence(*empty, [], len(ref), len(mov), rho)

    cost = _cost_matrix(ref, mov, rho, max_dist_rho, max_angle_deg, w_dist)
    matches = _assign(cost)
    if not matches:
        return [], *empty, _confidence(*empty, [], len(ref), len(mov), rho)

    ref_xy = np.array([ref[r]["pos"][:2] for r, _, _ in matches])
    mov_xy = np.array([mov[c]["pos"][:2] for _, c, _ in matches])

    keep = reject_outliers(ref_xy, mov_xy, rho, max_resid_rho)
    matches = [m for m, k in zip(matches, keep) if k]
    ref_xy, mov_xy = ref_xy[keep], mov_xy[keep]

    costs = [c for _, _, c in matches]
    conf = _confidence(ref_xy, mov_xy, costs, len(ref), len(mov), rho)
    return matches, ref_xy, mov_xy, conf
