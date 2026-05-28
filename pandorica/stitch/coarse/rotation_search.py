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
Global MT-endpoint rotation search — the primary coarse rotation.

Replaces the local PCA seed (``coarse.coarse_align``), which could not recover
large / ambiguous rotations. A *global* sweep over θ ∈ [0,360°) — rotate the
moving endpoints (centroid-aligned), match, and take the angle that maximises the
matched fraction — recovers magnitude *and sign* when the endpoint constellation
is asymmetric. Validated on two datasets: FemalePN sec12→13 = −90° (auto, correct
sign) and MaleMeiosis sec4→5 = +40°.

It is NOT trusted blindly: the bundled-spindle case produces a near-collinear
constellation with a **180° flip ambiguity** (confirmed on MaleMeiosis: every
interface's second basin sits at ~180°). So this returns a ``RotationEstimate``
carrying **degeneracy / confidence diagnostics** — peak margin, the ±180°
flip-ratio, PCA anisotropy, angular uniformity — so the
caller (the hybrid coarse: image A–P polarity + stack-wide sign continuity + ABSTAIN)
can detect when the rotation/sign is *not* trustworthy and resolve it elsewhere
rather than silently mis-stitching.
"""

from dataclasses import dataclass

import numpy as np

from pandorica.stitch.matching.matcher import match_sections
from pandorica.stitch.coarse.cpd import (
    cpd_rotation_search,
    _consensus,
    _median_nn,
)

# Degeneracy thresholds (tunable; defaults from the two validation datasets).
_FLIP_RATIO_MIN = 1.3  # peak/180°-lag below this ⇒ 180° (sign) ambiguous
_ANISOTROPY_MAX = 12.0  # PCA λmax/λmin above this ⇒ near-collinear (bundle)
_ANGULAR_UNIFORMITY_MAX = 0.85  # axial resultant above this ⇒ strongly oriented
_MARGIN_MIN = 1.5  # peak/2nd-basin below this ⇒ weakly determined
_MATCH_MIN = 0.3  # absolute matched-fraction floor


def _xy(eps) -> np.ndarray:
    return np.array([e["pos"][:2] for e in eps], dtype=float)


def _rotate_endpoints(eps, angle_deg, center, target):
    """Rotate endpoints about ``center`` by ``angle_deg``, then shift center→target."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = []
    for e in eps:
        p = e["pos"].copy()
        d = e["dir"].copy()
        p[:2] = R @ (p[:2] - center) + target
        d[:2] = R @ d[:2]
        out.append({**e, "pos": p, "dir": d})
    return out


def _regime(xy: np.ndarray):
    """(PCA anisotropy λmax/λmin, axial angular-uniformity resultant) of a cloud."""
    c = xy - xy.mean(0)
    w = np.linalg.eigvalsh(c.T @ c)
    aniso = float(w[-1] / max(w[0], 1e-9))
    th = np.arctan2(c[:, 1], c[:, 0])
    ang_u = float(np.hypot(np.cos(2 * th).mean(), np.sin(2 * th).mean()))
    return aniso, ang_u


@dataclass
class RotationEstimate:
    """Coarse rotation + degeneracy/confidence diagnostics (mov→ref, degrees)."""

    angle: float
    match_fraction: float
    peak_margin: float  # peak / 2nd-best basin (≥40° away)
    flip_ratio: float  # peak / 180°-lag basin (low ⇒ sign-ambiguous)
    anisotropy: float
    angular_uniformity: float
    n_ref: int
    n_mov: int

    @property
    def degenerate(self) -> bool:
        """True if the constellation can't trustworthily fix the rotation/sign."""
        return (
            self.flip_ratio < _FLIP_RATIO_MIN
            or self.anisotropy > _ANISOTROPY_MAX
            or self.angular_uniformity > _ANGULAR_UNIFORMITY_MAX
        )

    @property
    def confident(self) -> bool:
        # Gate on the matched fraction + the robust degeneracy flags (which include
        # the 180° flip-ratio). peak_margin is reported but NOT gated on: with the
        # loose coarse gate its chance floor is high, so it is advisory
        # only. Tight-gate inlier-significance is a later matcher-hardening step.
        return self.match_fraction >= _MATCH_MIN and not self.degenerate


def _rot_consensus(mov_xy, ref_xy, angle_deg, mc, rc, tol):
    """Rotation-only consensus inliers at ``angle_deg`` (centroid-aligned)."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    mapped = (mov_xy - mc) @ R.T + rc
    n, _ = _consensus(mapped, ref_xy, tol)
    return n


# A CPD basin margin below this (best residual vs the next rotation basin) means a
# rival rotation fits about as well ⇒ the angle is not trustworthy.
_CPD_BASIN_MARGIN_MIN = 1.5


def _cpd_estimate(ref_eps, mov_eps, rxy, w, n_seeds, quantile) -> RotationEstimate:
    """RotationEstimate from the multi-seed CPD search, with flip diagnostics."""
    mxy = _xy(mov_eps)
    est = cpd_rotation_search(rxy, mxy, w=w, n_seeds=n_seeds, inlier_quantile=quantile)
    aniso, ang_u = _regime(rxy)
    # flip_ratio: consensus at the chosen angle vs its 180° branch (the bundle
    # sign-ambiguity diagnostic the hybrid fusion gates on).
    tol = 0.6 * _median_nn(rxy)
    rc, mc = rxy.mean(0), mxy.mean(0)
    n_peak = _rot_consensus(mxy, rxy, est.angle, mc, rc, tol)
    n_flip = _rot_consensus(mxy, rxy, est.angle + 180.0, mc, rc, tol)
    denom = max(1, min(len(ref_eps), len(mov_eps)))
    flip_ratio = n_peak / max(n_flip, 1e-6)
    # If a NON-180° rival basin fits about as well (low basin margin), the rotation
    # is ambiguous (heavy-jitter near-degeneracy) — collapse flip_ratio so the
    # estimate reads degenerate and the hybrid flags rather than silently accepts.
    if est.basin_margin < _CPD_BASIN_MARGIN_MIN:
        flip_ratio = min(flip_ratio, 1.0)
    return RotationEstimate(
        angle=est.angle,
        match_fraction=n_peak / denom,
        peak_margin=est.basin_margin,
        flip_ratio=flip_ratio,
        anisotropy=aniso,
        angular_uniformity=ang_u,
        n_ref=len(ref_eps),
        n_mov=len(mov_eps),
    )


def global_rotation_search(
    ref_eps,
    mov_eps,
    rho: float,
    step: float = 5.0,
    coarse_gate_rho: float = 8.0,
    refine: bool = True,
    use_cpd: bool = False,
    cpd_w: float = 0.4,
    cpd_seeds: int = 12,
    cpd_quantile: float = 0.5,
) -> RotationEstimate:
    """
    Coarse rotation (mov→ref) by global tight-gate inlier-count maximisation over θ.

    :param ref_eps / mov_eps: boundary-endpoint lists ``{id, pos, dir}``.
    :param rho: endpoint scale (median NN spacing).
    :param step: coarse sweep step (deg); a ±2·step residual fine-refine follows.
    :param coarse_gate_rho: TIGHT matcher gate (in ρ) for the inlier-count sweep —
        tight so decoys can't be captured at the wrong rotation. Must be ≳ the
        worst true-match offset at the coarse ``step`` (a few ρ); 3 works for the
        validated data.
    :param use_cpd: estimate the angle with the multi-seed CPD search instead
        of the gated sweep. Decisively more decoy-robust and
        recovers ±90° from a cold start; the degeneracy diagnostics (anisotropy,
        angular_uniformity, flip_ratio) are still produced for the hybrid fusion.
    :param cpd_w / cpd_seeds: CPD outlier weight / # of seed angles.
    :return: a ``RotationEstimate`` (``.angle`` signed in (−180,180]).
    """
    n_ref, n_mov = len(ref_eps), len(mov_eps)
    if n_ref < 3 or n_mov < 3:
        return RotationEstimate(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, n_ref, n_mov)

    rxy = _xy(ref_eps)
    if use_cpd:
        return _cpd_estimate(ref_eps, mov_eps, rxy, cpd_w, cpd_seeds, cpd_quantile)
    rc, mc = rxy.mean(0), _xy(mov_eps).mean(0)
    angles = np.arange(0.0, 360.0, step)
    denom = max(1, min(n_ref, n_mov))

    # Coarse sweep objective = TIGHT-gate inlier COUNT, not loose-gate fraction.
    # Decoys (non-continuing MTs) find within-gate partners at WRONG rotations
    # under a loose gate (decoy capture → C was worse than A in the benchmark);
    # at a tight gate they don't form tight matches, so only the true rotation
    # accumulates many inliers. Count (not fraction) avoids the decoy-inflated
    # denominator.
    def inlier_count(th, gate):
        movc = _rotate_endpoints(mov_eps, th, mc, rc)
        _, rx, _, _ = match_sections(ref_eps, movc, rho, max_dist_rho=gate)
        return len(rx)

    counts = np.array([float(inlier_count(th, coarse_gate_rho)) for th in angles])
    i = int(counts.argmax())
    peak_th = float(angles[i])
    peak_count = float(counts[i])

    sep = np.abs((angles - peak_th + 180.0) % 360.0 - 180.0)
    second = float(counts[sep >= 40.0].max()) if np.any(sep >= 40.0) else 0.0
    j = int(np.argmin(np.abs((angles - (peak_th + 180.0)) % 360.0)))
    flip = float(counts[j])

    if refine:
        # The count is flat across the basin (it saturates), so it finds the basin
        # but a coarse angle. Localise the angle by MINIMISING the matched-pair
        # RESIDUAL over the basin (count finds the basin, residual finds the angle).
        fine_gate = min(coarse_gate_rho, 3.0)  # tight gate → precise angle

        def fine_residual(th):
            movc = _rotate_endpoints(mov_eps, th, mc, rc)
            _, rx, mxc, _ = match_sections(ref_eps, movc, rho, max_dist_rho=fine_gate)
            if len(rx) < 3:
                return np.inf
            return float(np.sqrt(((rx - mxc) ** 2).sum(1).mean()))

        window = np.arange(peak_th - 2 * step, peak_th + 2 * step + 0.5, 1.0)
        resids = [fine_residual(th) for th in window]
        peak_th = float(window[int(np.argmin(resids))])

    aniso, ang_u = _regime(rxy)
    angle = float(((peak_th + 180.0) % 360.0) - 180.0)
    # Reported match_fraction = tight inliers at the final angle / min(set sizes).
    final_count = float(inlier_count(angle, coarse_gate_rho))
    return RotationEstimate(
        angle=angle,
        match_fraction=final_count / denom,
        peak_margin=peak_count / max(second, 1e-6),
        flip_ratio=peak_count / max(flip, 1e-6),
        anisotropy=aniso,
        angular_uniformity=ang_u,
        n_ref=n_ref,
        n_mov=n_mov,
    )
