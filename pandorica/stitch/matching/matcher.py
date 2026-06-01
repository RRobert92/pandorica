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


def _resolve_max_dist(
    rho: float,
    max_dist_rho: float,
    min_dist_A: float,
    max_dist_A: float,
) -> float:
    """Compute the effective XY distance gate in physical units (Å).

    ``rho`` is the median nearest-neighbour spacing of endpoints, in Å. The
    raw ρ-scaled gate ``max_dist_rho * rho`` adapts to local density but is
    unbounded; in dense bundles it can land within neighbour-confusion
    territory, and in sparse regions it can grow to physically impossible
    values (a microtubule does not continue micrometres laterally between
    serial sections). The clamp ``[min_dist_A, max_dist_A]`` keeps the gate
    within physically plausible MT-continuation distances regardless of
    local density.
    """
    return float(np.clip(max_dist_rho * rho, min_dist_A, max_dist_A))


def _cost_matrix(ref, mov, rho, max_dist, max_angle_deg, w_dist):
    """Pairwise cost; np.inf where the distance/angle hard gates are exceeded.

    ``max_dist`` is in the same units as the endpoint positions (Å in the
    pipeline). Resolve it via :func:`_resolve_max_dist` before calling.
    """
    n_ref, n_mov = len(ref), len(mov)
    cost = np.full((n_ref, n_mov), np.inf)
    if max_dist <= 0.0:
        return cost
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


def _segments_cross_2d(p1, p2, q1, q2) -> bool:
    """True iff open segments p1p2 and q1q2 properly cross in 2D.

    Used by the uncrosser to spot the "rung swap": two close MT stubs whose
    Hungarian assignment is the X-cross rather than the parallel pairing.
    """

    def _orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _pair_direction_cost(r_dir, m_dir) -> float:
    """Plain ``1 - |cos Δθ|`` between two 2D tangents (0 = best, 1 = worst).

    Same metric the matcher already uses; reused here at the per-pair level.
    """
    return _vmf_direction_cost(np.asarray(r_dir), np.asarray(m_dir))


def uncross_pairs(matches, ref, mov, rho, neighbour_rho=2.0):
    """
    Locally uncross X-crossed Hungarian pairs by direction continuity.

    Hungarian minimises *total* cost; near-coincident pairs with similar
    direction can come out crossed (A↔B', B↔A') because the swap raises the
    sum elsewhere. This pass examines each pair against close neighbours
    (within ``neighbour_rho * rho``), checks whether their connecting segments
    cross in XY, and if so evaluates the swap: the configuration with lower
    *direction-only* cost wins. Distance is intentionally excluded — that was
    Hungarian's tiebreaker and is what gets fooled in the rung-swap case.

    :param matches: list of ``(ref_idx, mov_idx, cost)`` from Hungarian, indexing
        the deduped endpoint lists ``ref`` / ``mov``.
    :param ref / mov: deduped endpoint lists (each entry has ``pos`` and ``dir``).
    :param rho: median endpoint NN spacing.
    :param neighbour_rho: search radius in ρ for cross checks.
    :return: an uncrossed match list (same length, indices possibly reassigned).
    """
    if len(matches) < 2:
        return list(matches)

    out = [list(m) for m in matches]
    radius = neighbour_rho * rho
    radius2 = radius * radius
    changed = True
    iters = 0
    # Bounded iteration: a single swap can unlock its neighbours, but the
    # process is monotone in direction-cost so it converges quickly. Two passes
    # are enough on real data; cap at four to be defensive.
    while changed and iters < 4:
        changed = False
        iters += 1
        for i in range(len(out)):
            ri, mi, ci = out[i]
            p_r = ref[ri]["pos"][:2]
            p_m = mov[mi]["pos"][:2]
            for j in range(i + 1, len(out)):
                rj, mj, cj = out[j]
                q_r = ref[rj]["pos"][:2]
                # Only consider neighbour pairs where the ref endpoints are
                # geometrically close — the rung-swap is a local pathology.
                d2 = (p_r[0] - q_r[0]) ** 2 + (p_r[1] - q_r[1]) ** 2
                if d2 > radius2:
                    continue
                q_m = mov[mj]["pos"][:2]
                if not _segments_cross_2d(p_r, p_m, q_r, q_m):
                    continue
                # Direction-only cost of the two configurations.
                cur = _pair_direction_cost(
                    ref[ri]["dir"], mov[mi]["dir"]
                ) + _pair_direction_cost(ref[rj]["dir"], mov[mj]["dir"])
                swp = _pair_direction_cost(
                    ref[ri]["dir"], mov[mj]["dir"]
                ) + _pair_direction_cost(ref[rj]["dir"], mov[mi]["dir"])
                if swp + 1e-9 < cur:
                    out[i][1], out[j][1] = mj, mi
                    # Costs become stale after swap — recompute from the
                    # matcher's full distance + direction metric so callers
                    # see a coherent number. Mirror ``_cost_matrix``.
                    out[i][2] = _swap_cost(ref[ri], mov[mj], rho)
                    out[j][2] = _swap_cost(ref[rj], mov[mi], rho)
                    p_m = mov[out[i][1]]["pos"][:2]
                    changed = True
    return [tuple(m) for m in out]


def _swap_cost(r, m, rho):
    """Distance+direction cost for one pair, in the same units the cost
    matrix uses (without the gate). Used to refresh costs after an uncross
    swap; does not affect outlier rejection (which uses the rigid residual).
    """
    d = float(np.linalg.norm(r["pos"][:2] - m["pos"][:2]))
    dir_c = _pair_direction_cost(r["dir"], m["dir"])
    # Match the default weighting in ``_cost_matrix`` (w_dist=0.7) at the
    # nominal max_dist; this is a sanity number, not a gate.
    return 0.7 * (d / (5.0 * rho)) + 0.3 * dir_c


def filter_pair_smoothness(
    matches,
    ref,
    mov,
    *,
    rho: float = 1.0,
    max_tangent_deg: float = 30.0,
    max_chord_tangent_deg: float = 45.0,
    chord_check_rho: float = 0.5,
    min_xy: float = 0.2,
):
    """
    Drop pairs that would join their MTs at an implausible angle.

    Two XY-plane gates (both must pass, both sign-agnostic):

    * **tangent ↔ tangent** — ``|cos Δθ_tt| ≥ cos(max_tangent_deg)``. Two
      stubs whose local tangents disagree are not the same fiber.

    * **chord ↔ tangent** — the chord ``mov_pos − ref_pos`` (the line we'd
      actually draw to connect the two endpoints) must also lie along the
      natural continuation of each stub: ``|cos Δ_chord| ≥
      cos(max_chord_tangent_deg)`` against both tangents. This catches the
      sideways-offset failure mode that tangent-tangent alone misses — two
      stubs with **parallel** tangents but at different lateral positions
      have a chord that is **perpendicular** to those tangents, requiring a
      sharp lateral jump to chain. The chord threshold is intentionally
      *looser* than tangent-tangent so genuinely curved MTs survive (the
      chord deviates from each local tangent by the curvature accumulated
      between the two stubs' sampling windows).

    The chord check is only applied when the chord is at least
    ``chord_check_rho * rho`` long. Short chords (near-coincident stubs after
    coarse alignment, or genuine same-MT continuations across a thin cut) have
    a chord *direction* that is dominated by tracing noise — applying an
    angular gate there would drop legitimate matches. The matcher's distance
    gate already trusts those pairs as close enough; tangent-tangent is the
    only signal we add at that scale.

    Near-vertical pairs — both XY-tangents below ``min_xy`` — are skipped:
    the in-plane signal is too noisy to give a verdict, and the matcher's
    cost gate is the only available judge in that regime.

    :param matches: list of ``(ref_idx, mov_idx, cost)``.
    :param ref / mov: deduped endpoint lists.
    :param rho: median NN spacing of the boundary endpoints (sets the chord
        length unit).
    :param max_tangent_deg: max acceptable angle between matched XY tangents.
    :param max_chord_tangent_deg: max acceptable angle between the chord and
        *either* tangent. Higher = more MT curvature allowed.
    :param chord_check_rho: minimum chord length, in ρ, for the chord–tangent
        gate to be applied. Below this the chord is treated as noise-dominated
        and the chord check is skipped.
    :param min_xy: minimum XY-tangent magnitude on *both* sides for either
        verdict to be considered reliable.
    :return: filtered list of matches.
    """
    if not matches:
        return list(matches)
    cos_max_tt = float(np.cos(np.deg2rad(max_tangent_deg)))
    cos_max_ct = float(np.cos(np.deg2rad(max_chord_tangent_deg)))
    chord_check_thresh = float(chord_check_rho) * float(rho)
    out = []
    for r, c, cost in matches:
        rd = np.asarray(ref[r]["dir"])[:2]
        md = np.asarray(mov[c]["dir"])[:2]
        nr, nm = float(np.linalg.norm(rd)), float(np.linalg.norm(md))
        if nr < min_xy or nm < min_xy:
            # Near-vertical on at least one side: both verdicts are
            # uninformative; trust Hungarian's cost gate.
            out.append((r, c, cost))
            continue
        cos_tt = float(np.clip(np.dot(rd, md) / (nr * nm), -1.0, 1.0))
        if abs(cos_tt) < cos_max_tt:
            continue  # tangent ↔ tangent fails
        # Chord direction (XY): mov endpoint relative to ref endpoint.
        rp = np.asarray(ref[r]["pos"])[:2]
        mp = np.asarray(mov[c]["pos"])[:2]
        chord = mp - rp
        chord_norm = float(np.linalg.norm(chord))
        if chord_norm < chord_check_thresh:
            # Short chord: direction is noise-dominated. Trust the matcher's
            # distance gate + the tangent-tangent check above.
            out.append((r, c, cost))
            continue
        cos_cr = abs(float(np.dot(chord, rd))) / (chord_norm * nr)
        cos_cm = abs(float(np.dot(chord, md))) / (chord_norm * nm)
        if cos_cr >= cos_max_ct and cos_cm >= cos_max_ct:
            out.append((r, c, cost))
    return out


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
    min_dist_A: float = 500.0,
    max_dist_A: float = 2500.0,
    max_angle_deg: float = 30.0,
    w_dist: float = 0.7,
    dup_frac: float = 0.1,
    max_resid_rho: float = 2.0,
    uncross_neighbour_rho: float = 2.0,
    smoothness_max_tangent_deg: float = 45.0,
    smoothness_max_chord_tangent_deg: float = 60.0,
    smoothness_chord_check_rho: float = 1.0,
    smoothness_min_xy: float = 0.2,
):
    """
    Match two boundary-endpoint sets with ρ-scaled gates, vMF direction, and W1.

    Pipeline inside this function::

        dedupe → Hungarian → uncross (rung swap) → per-pair smoothness gate
               → rigid-residual outlier rejection

    The two new steps (uncross + smoothness) sit *before* outlier rejection on
    purpose: they fix wrong correspondences before the rigid fit sees them.
    The same matches drive both the warp fitter (alignment) and the downstream
    chain builder (cross-section MT identity), so cleaning them once here
    benefits both.

    :param ref_endpoints / mov_endpoints: lists of ``{id, pos, dir}`` (see
        ``extract_boundary_endpoints`` / ``scale.boundary_landmarks``).
    :param rho: median NN spacing of the section (length unit for all gates).
    :param max_dist_rho: ρ-scaled XY distance gate. The *effective* gate is
        ``clip(max_dist_rho * rho, min_dist_A, max_dist_A)`` — pure
        ρ-scaling can be physically unreasonable both in very dense bundles
        (loose, neighbour-confusion territory) and very sparse regions
        (micrometres of lateral drift between sections, impossible). The
        clamp bounds it to plausible MT-continuation distances.
    :param min_dist_A: physical floor for the distance gate, in Å. Default
        500 Å (50 nm, ≈ 2 MT diameters) — comfortable margin for tracing
        noise + warp residual in dense data.
    :param max_dist_A: physical ceiling for the distance gate, in Å. Default
        2500 Å (250 nm, ≈ 10 MT diameters) — a real same-MT continuation
        should land within this even if warp residual is sizeable.
    :param max_angle_deg: max tangent angle difference (sign-agnostic).
    :param w_dist: weight of the distance term vs the direction term.
    :param dup_frac: dedupe radius as a fraction of ρ.
    :param max_resid_rho: outlier residual gate, in ρ.
    :param uncross_neighbour_rho: local radius (ρ) for the X-cross check.
    :param smoothness_max_tangent_deg: per-pair tangent-vs-tangent gate (deg).
    :param smoothness_max_chord_tangent_deg: per-pair chord-vs-tangent gate
        (deg). Higher = more MT curvature allowed across the cut. The default
        (45°) is looser than the tangent-vs-tangent gate (30°) so genuinely
        bending MTs survive while sideways-offset misjoins are still rejected.
    :param smoothness_chord_check_rho: minimum chord length (in ρ) for the
        chord–tangent gate to fire. Short chords are noise-dominated and only
        the tangent–tangent gate applies.
    :param smoothness_min_xy: minimum XY-tangent magnitude for the smoothness
        verdict to be considered reliable (near-vertical pairs are skipped).
    :return: ``(matches, ref_xy, mov_xy, confidence, id_pairs)``. ``matches`` is
        the list of ``(ref_idx, mov_idx, cost)`` into the *deduped* endpoint
        lists; ``id_pairs`` is the parallel ``(ref_mt_id, mov_mt_id, cost)``
        carrying original MT IDs for downstream chain building.
    """
    ref = dedupe_endpoints(ref_endpoints, rho, dup_frac)
    mov = dedupe_endpoints(mov_endpoints, rho, dup_frac)
    empty = (np.empty((0, 2)), np.empty((0, 2)))
    if len(ref) == 0 or len(mov) == 0:
        return [], *empty, _confidence(*empty, [], len(ref), len(mov), rho), []

    max_dist = _resolve_max_dist(rho, max_dist_rho, min_dist_A, max_dist_A)
    cost = _cost_matrix(ref, mov, rho, max_dist, max_angle_deg, w_dist)
    matches = _assign(cost)
    if not matches:
        return [], *empty, _confidence(*empty, [], len(ref), len(mov), rho), []

    matches = uncross_pairs(matches, ref, mov, rho, neighbour_rho=uncross_neighbour_rho)
    matches = filter_pair_smoothness(
        matches,
        ref,
        mov,
        rho=rho,
        max_tangent_deg=smoothness_max_tangent_deg,
        max_chord_tangent_deg=smoothness_max_chord_tangent_deg,
        chord_check_rho=smoothness_chord_check_rho,
        min_xy=smoothness_min_xy,
    )
    if not matches:
        return [], *empty, _confidence(*empty, [], len(ref), len(mov), rho), []

    ref_xy = np.array([ref[r]["pos"][:2] for r, _, _ in matches])
    mov_xy = np.array([mov[c]["pos"][:2] for _, c, _ in matches])

    keep = reject_outliers(ref_xy, mov_xy, rho, max_resid_rho)
    matches = [m for m, k in zip(matches, keep) if k]
    ref_xy, mov_xy = ref_xy[keep], mov_xy[keep]

    costs = [c for _, _, c in matches]
    conf = _confidence(ref_xy, mov_xy, costs, len(ref), len(mov), rho)
    id_pairs = [(int(ref[r]["id"]), int(mov[c]["id"]), float(cost)) for r, c, cost in matches]
    return matches, ref_xy, mov_xy, conf, id_pairs
