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
Coherent Point Drift (rigid) matcher — numpy self-port (the BCPD rigid core).

The gate-based matcher is decoy-/jitter-fragile: no fixed distance gate is robust
to both. CPD replaces the hard one-to-one gated
Hungarian assignment with a **probabilistic GMM-EM fit + an explicit uniform
outlier component**, so non-continuing MTs (decoys) are absorbed by the outlier
term rather than forced into (wrong) matches — no gate to tune. It returns **soft
correspondences** (a posterior matrix) → a calibrated confidence, plus the rigid
transform (rotation + translation + optional scale).

This is the rigid CPD of Myronenko & Song (2010), the algorithmic foundation of
BCPD (Hirose 2021). The Bayesian/variational and non-rigid "drift" layers of full
BCPD are a further extension; the rigid core is what the measured matcher gap
needs (the non-rigid stage is handled separately by the guarded TPS warp).

Numerics: points are internally centred/scaled to O(1) (the Gaussian responsibilities
underflow at raw Å magnitudes ~10⁵); the transform is de-normalised on return.
"""

import os

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np


@dataclass
class CPDResult:
    """Rigid CPD fit aligning ``Y`` (moving) onto ``X`` (fixed)."""

    R: np.ndarray  # 2x2 rotation
    t: np.ndarray  # 2 translation
    s: float  # isotropic scale
    P: np.ndarray  # [M, N] soft correspondence posteriors (rows = Y, cols = X)
    sigma2: float  # final GMM variance
    iterations: int

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """Apply the fitted transform to ``[M, 2]`` moving points."""
        return self.s * (np.asarray(Y, float) @ self.R.T) + self.t


def cpd_rigid(
    X: np.ndarray,
    Y: np.ndarray,
    w: float = 0.3,
    allow_scale: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> CPDResult:
    """
    Rigid Coherent Point Drift: align moving ``Y`` onto fixed ``X``.

    :param X: ``[N, 2]`` fixed (reference) points.
    :param Y: ``[M, 2]`` moving points.
    :param w: assumed outlier fraction in ``[0, 1)`` — the weight of the uniform
        background component. Higher = more robust to decoys / partial overlap.
    :param allow_scale: also estimate an isotropic scale.
    :param max_iter / tol: EM stopping controls.
    :return: a ``CPDResult`` (transform mov→ref + soft correspondences ``P``).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N, D = X.shape
    M = Y.shape[0]

    # Centre/scale both sets by X's stats (preserves the rigid relation).
    xm = X.mean(0)
    scale = np.sqrt((((X - xm) ** 2).sum(1)).mean()) or 1.0
    Xn = (X - xm) / scale
    Yn = (Y - xm) / scale

    R = np.eye(D)
    t = np.zeros(D)
    s = 1.0
    TY = s * (Yn @ R.T) + t
    # initial variance = mean squared all-pairs distance
    sigma2 = (M * (Xn**2).sum() + N * (Yn**2).sum() - 2 * Xn.sum(0) @ Yn.sum(0)) / (
        D * M * N
    )
    sigma2 = max(float(sigma2), 1e-9)
    c_const = (w / (1.0 - w)) * (M / N) if w < 1.0 else 0.0

    P = np.zeros((M, N))
    it = 0
    for it in range(1, max_iter + 1):
        # E-step: responsibilities P[m, n] = p(y_m generated x_n)
        diff = Xn[None, :, :] - TY[:, None, :]  # [M, N, D]
        dist2 = (diff**2).sum(-1)  # [M, N]
        P = np.exp(-dist2 / (2.0 * sigma2))
        c = (2.0 * np.pi * sigma2) ** (D / 2.0) * c_const
        denom = P.sum(0)[None, :] + c
        denom[denom == 0] = 1e-12
        P = P / denom

        # M-step (rigid, closed form)
        Np = P.sum()
        if Np < 1e-9:
            break
        P1 = P.sum(1)  # [M]
        Pt1 = P.sum(0)  # [N]
        muX = Xn.T @ Pt1 / Np  # [D]
        muY = Yn.T @ P1 / Np
        Xhat = Xn - muX
        Yhat = Yn - muY
        A = Xhat.T @ P.T @ Yhat  # [D, D]
        U, Sg, Vt = np.linalg.svd(A)
        Cd = np.ones(D)
        Cd[-1] = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag(Cd) @ Vt
        trAR = float((Sg * Cd).sum())
        if allow_scale:
            s = trAR / float((P1 * (Yhat**2).sum(1)).sum())
        t = muX - s * (R @ muY)
        TY = s * (Yn @ R.T) + t

        sigma2_new = (float((Pt1 * (Xhat**2).sum(1)).sum()) - s * trAR) / (Np * D)
        sigma2_new = max(sigma2_new, 1e-9)
        if abs(sigma2_new - sigma2) < tol:
            sigma2 = sigma2_new
            break
        sigma2 = sigma2_new

    # De-normalise the transform back to real coordinates: real maps Y→X.
    # Xn = s R Yn + t, with Xn=(X-xm)/scale, Yn=(Y-xm)/scale ⇒
    #   X = s R Y + (xm - s R xm + scale·t)
    R_real = R
    s_real = s
    t_real = xm - s * (R @ xm) + scale * t
    return CPDResult(
        R_real, t_real, float(s_real), P, float(sigma2 * scale * scale), it
    )


def _angle_of(R: np.ndarray) -> float:
    """Signed rotation angle (deg, in (−180,180]) of a 2x2 rotation."""
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


@dataclass
class CPDRotation:
    """Multi-seed CPD rotation estimate (mov→ref)."""

    angle: float
    n_confident: int  # consensus inliers at the winning seed
    match_fraction: float
    result: CPDResult
    basin_margin: float = (
        1.0  # 2nd-best-basin residual / best residual (≥1; ~1 = ambiguous)
    )


def cpd_rotation_search(
    ref_xy: np.ndarray,
    mov_xy: np.ndarray,
    w: float = 0.4,
    n_seeds: int = 12,
    allow_scale: bool = False,
    inlier_quantile: float = 0.5,
) -> CPDRotation:
    """
    Robust coarse rotation by **multi-seed CPD**, the decoy-resistant replacement
    for the gate-based inlier-count sweep (which captured decoys at wrong angles).

    CPD's EM is non-convex, so a single identity start cannot recover large / ±90°
    rotations. Instead we seed CPD from ``n_seeds`` angles spanning [0,360°) — each
    seed pre-rotates the moving cloud about its centroid, then CPD-rigid refines
    *robustly* (the uniform outlier term absorbs decoys, no gate). The winner is the
    seed with the lowest ``inlier_quantile``-th nearest-neighbour residual: a low
    percentile residual requires a *majority* of points to be well-aligned, which
    only the true rigid map achieves (a spurious basin aligns a coincidental
    minority). See ``basin_margin`` for the ambiguity guard.

    :param ref_xy / mov_xy: ``[N,2]`` / ``[M,2]`` endpoint coordinates.
    :param w: outlier weight (decoy robustness).
    :param n_seeds: # of equispaced seed angles over [0,360°).
    :param inlier_quantile: quantile of NN residuals used to rank seeds (≈ expected
        inlier fraction). q=0.5 (median) tolerates ~50% decoys; lower q is more
        decoy-robust but noisier with few points.
    :return: a ``CPDRotation`` (best ``.angle`` mov→ref, with the underlying fit).
    """
    ref_xy = np.asarray(ref_xy, float)
    mov_xy = np.asarray(mov_xy, float)
    if len(ref_xy) < 3 or len(mov_xy) < 3:
        return CPDRotation(0.0, 0, 0.0, cpd_rigid(ref_xy, mov_xy, w=w))

    mc = mov_xy.mean(0)
    denom = max(1, min(len(ref_xy), len(mov_xy)))
    # Seed ranking = the q-th PERCENTILE of nearest-neighbour residuals (minimised),
    # tol-free. This is the crux of decoy-AND-jitter robustness, and is NOT a fixed
    # tolerance (which slides the decoy↔jitter failure along one knob) nor a trimmed
    # mean of the best fraction (which rewards a wrong basin that aligns ANY subset).
    # A low percentile residual requires a MAJORITY of points to be well-aligned —
    # only the true rigid map achieves that; a spurious basin aligns a coincidental
    # minority and so its q-th residual is large. q (≈ the inlier fraction) sets the
    # outlier breakdown: q=0.5 tolerates ~50% decoys while jitter only lifts the
    # residual mildly. Reported confidence still uses an NN-tol consensus count.
    tol = 1.0 * _median_nn(ref_xy)
    seeds = np.linspace(0.0, 360.0, n_seeds, endpoint=False)

    def _eval_seed(seed):  # -> (score, angle, n_inl, res)
        a = np.deg2rad(seed)
        Rs = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        seeded = (mov_xy - mc) @ Rs.T + mc
        res = cpd_rigid(ref_xy, seeded, w=w, allow_scale=allow_scale)
        # CPD proposes the transform (seed pre-rotation composed in); the percentile
        # residual scores it. res maps the SEEDED mov onto ref.
        mapped = res.transform(seeded)
        nn = np.sqrt(((mapped[:, None, :] - ref_xy[None, :, :]) ** 2).sum(-1)).min(1)
        score = float(np.percentile(nn, 100.0 * inlier_quantile))
        n_inl, _ = _consensus(mapped, ref_xy, tol)
        angle = float(((_angle_of(res.R @ Rs) + 180.0) % 360.0) - 180.0)
        return (score, angle, n_inl, res)

    # The seeds are independent and the per-seed CPD EM is GIL-releasing numpy
    # (dense [M,N] E-step), so a thread per seed is real parallelism. ``map``
    # preserves input order, so ``cands`` is byte-identical to the serial sweep.
    n_workers = min(int(n_seeds), os.cpu_count() or 1)
    if n_workers > 1 and n_seeds > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            cands = list(ex.map(_eval_seed, seeds))
    else:
        cands = [_eval_seed(s) for s in seeds]

    cands.sort(key=lambda c: c[0])  # lowest residual first
    best_score, angle, n_inl, res = cands[0]
    # Basin margin: best residual vs the best residual among DIFFERENT basins (angle
    # ≥40° away). Near 1 ⇒ a rival rotation fits about as well ⇒ the angle is not
    # trustworthy (the heavy-jitter flip near-degeneracy). The caller flags this
    # rather than accept a confident-looking wrong basin.
    rival = next(
        (c[0] for c in cands[1:] if abs(((c[1] - angle + 180) % 360) - 180) >= 40.0),
        None,
    )
    margin = float(rival / best_score) if rival and best_score > 1e-9 else float("inf")
    return CPDRotation(angle, int(n_inl), n_inl / denom, res, basin_margin=margin)


def _median_nn(xy: np.ndarray) -> float:
    """Median nearest-neighbour spacing of a point cloud (fallback if <2 pts)."""
    if len(xy) < 2:
        return 1.0
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1))) or 1.0


def _consensus(mapped: np.ndarray, ref_xy: np.ndarray, tol: float):
    """
    Greedy one-to-one consensus: # of mapped points within ``tol`` of a distinct
    ref point, and the RMS of those inlier distances. Decoy-/collapse-robust.
    """
    d = np.sqrt(((mapped[:, None, :] - ref_xy[None, :, :]) ** 2).sum(-1))
    used = np.zeros(ref_xy.shape[0], bool)
    dists = []
    for m in np.argsort(d.min(1)):  # closest pairs first
        order = np.argsort(d[m])
        for n in order:
            if d[m, n] > tol:
                break
            if not used[n]:
                used[n] = True
                dists.append(d[m, n])
                break
    if not dists:
        return 0, np.inf
    return len(dists), float(np.sqrt(np.mean(np.square(dists))))


def correspondences(result: CPDResult, prob_threshold: float = 0.5):
    """
    Hard correspondences from the soft posterior: for each fixed point (column),
    the most-likely moving point if that probability exceeds ``prob_threshold``.

    :return: ``(pairs, confidence)`` — ``pairs`` is a list of ``(mov_idx, fix_idx,
        prob)``; ``confidence`` is the mean accepted posterior (a calibrated score).
    """
    P = result.P
    if P.size == 0:
        return [], 0.0
    pairs = []
    for n in range(P.shape[1]):
        m = int(P[:, n].argmax())
        p = float(P[m, n])
        if p >= prob_threshold:
            pairs.append((m, n, p))
    conf = float(np.mean([p for _, _, p in pairs])) if pairs else 0.0
    return pairs, conf
