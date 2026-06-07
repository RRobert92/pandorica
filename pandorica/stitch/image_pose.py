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
Image-only coarse poses (no microtubules).

Same two-stage method as the MT stitcher, but driven by **image patches** instead
of microtubule landmarks:

* **coarse** — a rotation sweep scored by *weighted RANSAC inlier support*: rotate the
  moving boundary-face over candidate angles, block-match onto the fixed face, fit a
  rigid transform by confidence-weighted RANSAC, and keep the angle with the most inlier
  support (a wrong 180° flip leaves few consistent cells, so its support collapses; a
  central-disk NCC breaks a residual tie). RANSAC carries the translation too;
* **fine** — the same guarded image-patch warp (``image_residual_warps``) used for
  MT-free regions, applied here over the whole (mask-free) face.

The moving face is rotated with the *same* pose operator the exporter applies, so
the returned angle plugs straight into the v2 pose convention. Per-interface
relatives compose into absolute poses (section 0 = gauge). Memory-safe: volumes
are read two at a time, downsampled.
"""

from typing import List, Optional

import numpy as np

from pandorica.stitch.transform.solver import (
    IDENTITY,
    Pose,
    compose_poses,
    invert_pose,
    pose_from_matrix,
)
from pandorica.stitch.transform.applier import (
    make_inverse_map,
    warp_volume_slicewise,
)
from pandorica.stitch import geometry as geo
from pandorica.stitch.match import block_match, _prep, _mi, _METHODS
from pandorica.stitch.dataset import Dataset
from pandorica.stitch.contour_rotation import contour_rotation

# Coarse pose robustness (image_only_poses): the two faces are different cut surfaces, so
# ~half the block-match cells decorrelate; a confidence-weighted RANSAC fits the rigid
# transform on the rest and we ABSTAIN when too few cells are inliers (also catches a bad
# flip, which leaves few consistent cells). See tmp/coarse_warp/DISCOVERIES.md.
_GATE_FRAC = 0.12   # min rigid-inlier fraction to trust the block-match pose
_MIN_CELLS = 12     # and at least this many correspondences, for a stable fit

# Anisotropic/shear refine (image_only_poses): once the angle is locked, a small-window
# block-match + affine RANSAC recovers a per-interface (sx, sy)/shear the rigid fit cannot.
# Probe (tmp/aniso_policy_probe.py, tmp/verify_affine_refine.py) on a real EM face:
# the fit returns ~identity on isotropic data (0.07% spurious aniso over a 10-section
# chain) and recovers a planted aniso to <0.5% with a small window; a large window biases
# the magnitude ~+5pp (within-window deformation). So we use a SMALL window here and commit
# the affine only when the residual stretch clears a gate (else keep the validated rigid).
_AFFINE_HALF = 20      # small match window: minimises within-window-deformation aniso bias
_ANISO_GATE = 0.02     # commit aniso/shear only if max(sx,sy)/min(sx,sy)-1 exceeds this

# Physical guard on the committed residual-affine. _ANISO_GATE bounds the anisotropy
# RATIO but NOT the absolute AREA, so the affine RANSAC can overfit a both-axes stretch
# two adjacent EM sections cannot physically show (Monopoles sec12->sec13: det 1.12, an
# 8.9%/3.1% stretch) — left in, it warps the volume corner by thousands of Å. We clamp
# A_res's singular values and determinant into a physical band (keeping its orientation),
# so a gross stretch is reined to a physical one. The band's low end must clear the
# largest *real* compression in the stack (Monopoles sec10->sec11 sv 0.917, det 0.937 —
# a genuine aniso that HELPS the MTs), so it only bites the overfit outlier. The MT
# scale-gate (gate_coarse_scale) does the precise per-interface validation on top.
_AFFINE_SV_BAND = (0.85, 1.15)    # per-axis stretch magnitude rail (gross single-axis)
_AFFINE_DET_BAND = (0.90, 1.10)   # area change |det(A_res)| (the both-axes inflation)

# Rotation = the angle with the most weighted RANSAC inlier SUPPORT (not a central-disk
# NCC peak), which resolves the 180° branch by image consensus: a wrong flip leaves few
# rigidly consistent cells so its support collapses.
_SWEEP_STEP = 15.0    # coarse angle step (deg); one local refine at step/5 follows
_BRANCH_FRAC = 0.85   # flag + NCC-tiebreak only when the opposite (~180°) branch keeps
                      # >= this fraction of the winner's support (a true near-tie); a
                      # clear margin trusts the RANSAC winner over the weaker NCC.
_ROT_TOL = 10.0       # sweep-chosen vs contour angle gap (deg) above this = disagree

# MT<->image dual-chain cross-check (reconcile_image_mt): the image RANSAC pose is an
# independent second estimate; keep MT unless the image disagrees AND is more certain.
# Translation override is GATED on the image's own confidence (silent on abstain) so the
# cross-check stays low-noise. See tmp/coarse_warp/DISCOVERIES.md.
_XCHK_SHIFT_FLOOR_PX = 8.0   # min |Δcenter-shift| (face px) to call a translation conflict
_XCHK_SHIFT_FRAC = 0.25      # ...or this fraction of the drift magnitude (whichever larger)
_XCHK_MT_MATCH_MIN = 0.30    # MT match-fraction below this = MT translation not trusted


def _mean_face(volume, face: str, n_slices: int, invert_z: bool):
    """Z-MEAN projection of the boundary slab (same slab as ``geo.zmax_face``).

    The contour/segmentation rotation wants the smooth membrane signal: a mean
    projection averages down noise and preserves the nuclear envelope, whereas the
    Z-MIP (``zmax_face``) keeps the brightest speckle per column and breaks it.
    """
    volume = np.asarray(volume)
    n = max(1, min(int(n_slices), volume.shape[0]))
    take_high = (face == "top") == (not invert_z)
    sl = volume[-n:] if take_high else volume[:n]
    return sl.mean(axis=0).astype(np.float32)


def _rotate_face(face, angle, center):
    """Resample a 2-D face by ``angle`` about ``center`` (px) — same op the exporter uses."""
    pose = geo.centroid_pose(angle, 0.0, 0.0, center)
    inv = make_inverse_map(pose)
    return warp_volume_slicewise(face[None], inv, out_hw=face.shape, dtype=np.float32)[
        0
    ]


def _similarity(a, b, valid, metric):
    aa, bb = a[valid], b[valid]
    if aa.size < 100:
        return -1e9
    if metric == "mi":
        return _mi(aa, bb)
    am, bm = aa - aa.mean(), bb - bb.mean()
    den = np.sqrt((am * am).sum() * (bm * bm).sum())
    return float((am * bm).sum() / den) if den > 0 else -1e9


def _rigid_from_pairs(src: np.ndarray, dst: np.ndarray, w: Optional[np.ndarray] = None):
    """Weighted least-squares 2-D rigid fit (rotation + translation, no scale).

    Closed-form 2-D Kabsch for ``dst ≈ R(θ)·src + t``. Per-pair weights ``w`` (the
    block-match confidence) tilt the fit toward the cells most likely to be true
    correspondences, so a few high-information landmarks set the transform.

    :return: ``(theta_rad, t_xy)``.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    wsum = float(w.sum())
    if len(src) < 2 or wsum <= 0:
        return 0.0, np.zeros(2)
    sc = (w[:, None] * src).sum(0) / wsum
    dc = (w[:, None] * dst).sum(0) / wsum
    s, d = src - sc, dst - dc
    num = float((w * (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0])).sum())
    den = float((w * (s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1])).sum())
    theta = float(np.arctan2(num, den))
    c, sn = np.cos(theta), np.sin(theta)
    t = dc - np.array([[c, -sn], [sn, c]]) @ sc
    return theta, t


def _ransac_rigid(
    src: np.ndarray, dst: np.ndarray, conf: np.ndarray,
    *, tol: float, iters: int = 200, seed: int = 0,
):
    """Confidence-weighted RANSAC 2-D rigid fit (rotation + translation).

    A rigid transform is fixed by **two** point correspondences, so the minimal
    sample is 2 pairs — drawn with probability ∝ ``conf`` so the most matchable
    cells seed the hypotheses. The model with the largest confidence-weighted inlier
    support (Σ ``conf`` over pairs within ``tol`` px of the model) wins, then a
    weighted refit on that inlier set. Picking by *weighted support* (not a raw
    agreeing-cell count) is the per-cell-information weighting: a handful of confident,
    geometrically consistent landmarks beats a larger spray of low-information cells,
    and a wrong 180° flip leaves few consistent pairs so it still loses on support.

    :return: ``(theta_rad, t_xy, inlier_mask, support)`` — ``support`` = Σ conf over
        inliers (``0`` when < 2 correspondences or no model gathers ≥ 2 inliers).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    conf = np.asarray(conf, float)
    n = len(src)
    if n < 2:
        return 0.0, np.zeros(2), np.zeros(n, bool), 0.0
    p = conf / conf.sum() if conf.sum() > 0 else None
    rng = np.random.default_rng(seed)
    iters = min(int(iters), n * (n - 1) // 2)
    best_support, best_inl = -1.0, None
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False, p=p)
        if np.linalg.norm(src[i] - src[j]) < tol:
            continue  # too short a baseline -> the 2-point angle is unstable
        theta, t = _rigid_from_pairs(src[[i, j]], dst[[i, j]])
        c, sn = np.cos(theta), np.sin(theta)
        pred = src @ np.array([[c, sn], [-sn, c]]) + t  # R·src + t
        inl = np.linalg.norm(pred - dst, axis=1) < tol
        support = float(conf[inl].sum())
        if support > best_support:
            best_support, best_inl = support, inl
    if best_inl is None or int(best_inl.sum()) < 2:
        return 0.0, np.zeros(2), np.zeros(n, bool), 0.0
    theta, t = _rigid_from_pairs(src[best_inl], dst[best_inl], conf[best_inl])
    return theta, t, best_inl, float(conf[best_inl].sum())


def _match_rigid(fixed, moving, ang, center, *, metric, grid, search, tol, workers,
                 half=64):
    """Rotate ``moving`` by ``ang``, block-match, then weighted-RANSAC a rigid refine.

    The block-match gives cell correspondences each with a confidence (peakiness);
    :func:`_ransac_rigid` fits a rigid transform on the most matchable, geometrically
    consistent cells. The sweep angle and the RANSAC residual rotation compose into a
    single angle about ``center`` plus a net translation (the exporter's pose
    convention), so the branch record stays ``(rot, shift)``.

    :return: ``dict(rot, shift, agree, ncell, support)`` — ``support`` = weighted Σ conf
        over rigid inliers (the rotation score, where the per-cell-information weighting
        lives); ``agree`` = plain fraction of correspondences that are rigid inliers (an
        unweighted, hard-to-game floor for the abstain gate); ``ncell`` = correspondences
        found.
    """
    moving_rot = _rotate_face(moving, ang, center)
    src, dst, conf = block_match(
        fixed, moving_rot, None, method=metric, grid=grid,
        half=half, search=search, workers=workers,
    )
    ncell = len(src)
    dtheta, t, inl, support = _ransac_rigid(src, dst, conf, tol=tol)
    agree = float(inl.sum()) / ncell if ncell else 0.0
    c, sn = np.cos(dtheta), np.sin(dtheta)
    center = np.asarray(center, float)
    shift = np.array([[c, -sn], [sn, c]]) @ center + t - center
    return dict(
        rot=float(ang + np.degrees(dtheta)),
        ang=float(ang),
        shift=np.asarray(shift, float),
        agree=float(agree),
        ncell=int(ncell),
        support=float(support),
    )


def _fit_affine(src: np.ndarray, dst: np.ndarray, w: Optional[np.ndarray] = None):
    """Weighted least-squares full 2-D affine fit ``dst ≈ A·src + t`` (A is 2x2).

    Solved per output axis (the x- and y-equations decouple): each is a weighted
    linear fit of ``[src_x, src_y, 1]`` to the target coordinate. Unlike the rigid
    :func:`_rigid_from_pairs` this leaves the 2x2 free, so anisotropy and shear are
    captured. :return: ``(A, t)``.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    w = np.ones(n) if w is None else np.asarray(w, float)
    sw = np.sqrt(np.clip(w, 0.0, None))
    M = np.column_stack([src[:, 0], src[:, 1], np.ones(n)]) * sw[:, None]
    cx, *_ = np.linalg.lstsq(M, dst[:, 0] * sw, rcond=None)
    cy, *_ = np.linalg.lstsq(M, dst[:, 1] * sw, rcond=None)
    return np.array([[cx[0], cx[1]], [cy[0], cy[1]]]), np.array([cx[2], cy[2]])


def _ransac_affine(
    src: np.ndarray, dst: np.ndarray, conf: np.ndarray,
    *, tol: float, iters: int = 300, seed: int = 0,
):
    """Confidence-weighted RANSAC full-affine fit (minimal sample = 3 pairs).

    A 2-D affine is fixed by **three** correspondences, so the minimal sample is 3
    (drawn ∝ ``conf``; near-collinear triples are skipped as degenerate). The model
    with the largest confidence-weighted inlier support wins, then a weighted refit on
    that inlier set. Affine inliers (not the rigid ones) are required because under real
    anisotropy a rigid model only agrees near the centre, biasing the fit inward.

    :return: ``(A, t, inlier_mask, support)`` — ``support`` = Σ conf over inliers.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    conf = np.asarray(conf, float)
    n = len(src)
    if n < 3:
        return np.eye(2), np.zeros(2), np.zeros(n, bool), 0.0
    p = conf / conf.sum() if conf.sum() > 0 else None
    rng = np.random.default_rng(seed)
    best_support, best_inl = -1.0, None
    for _ in range(int(iters)):
        idx = rng.choice(n, size=3, replace=False, p=p)
        d1, d2 = src[idx[1]] - src[idx[0]], src[idx[2]] - src[idx[0]]
        if abs(d1[0] * d2[1] - d1[1] * d2[0]) < 1.0:
            continue  # near-collinear triple -> degenerate affine
        A, t = _fit_affine(src[idx], dst[idx])
        if not np.isfinite(A).all():
            continue
        inl = np.linalg.norm(src @ A.T + t - dst, axis=1) < tol
        support = float(conf[inl].sum())
        if support > best_support:
            best_support, best_inl = support, inl
    if best_inl is None or int(best_inl.sum()) < 3:
        return np.eye(2), np.zeros(2), np.zeros(n, bool), 0.0
    A, t = _fit_affine(src[best_inl], dst[best_inl], conf[best_inl])
    return A, t, best_inl, float(conf[best_inl].sum())


def _clamp_affine(A: np.ndarray):
    """Bound a residual-affine into a physical stretch/area band, keeping orientation.

    SVD ``A = U·diag(s)·Vt``; clamp each singular value into :data:`_AFFINE_SV_BAND`
    (per-axis stretch rail), then clamp ``det`` into :data:`_AFFINE_DET_BAND` by an
    isotropic rescale — which leaves the anisotropy *direction and ratio* intact while
    reining the absolute AREA the ratio-only :data:`_ANISO_GATE` never bounded. So a
    genuine mild aniso passes through unchanged and only a gross over-stretch is pulled
    back to the physical edge. :return: ``(A_clamped, changed)``.
    """
    U, s, Vt = np.linalg.svd(A)
    s_c = np.clip(s, _AFFINE_SV_BAND[0], _AFFINE_SV_BAND[1])
    A_c = (U * s_c) @ Vt
    det = float(np.linalg.det(A_c))
    lo, hi = _AFFINE_DET_BAND
    if det > 0 and (det < lo or det > hi):
        A_c = A_c * float(np.sqrt(np.clip(det, lo, hi) / det))
    return A_c, not np.allclose(A_c, A, atol=1e-6)


def _affine_refine(fixed, moving, ang, center, *, metric, grid, search, workers,
                   half=_AFFINE_HALF):
    """Recover a per-interface anisotropic/shear linear part once the angle is locked.

    Mirrors :func:`_match_rigid` at the committed sweep angle ``ang`` — rotate ``moving``
    by ``ang`` about ``center``, block-match (but with the SMALL :data:`_AFFINE_HALF`
    window, which minimises the within-window-deformation bias on the aniso magnitude),
    confidence-weighted affine-RANSAC — then builds the rel pose's linear part exactly
    like production's rigid formula with the residual rigid ``R(dtheta)`` replaced by the
    residual affine ``A_res``::

        L = A_res · R(ang)            (rigid limit A_res = R(dtheta) ⇒ L = R(ang+dtheta))
        shift = A_res·center + t − center

    On a rigid interface ``A_res ≈ I`` and this reproduces the rigid pose to <0.3 px
    (validated, tmp/verify_affine_refine.py). :return: ``(L, shift_px, (sx, sy), n_inl)``
    — ``(sx, sy)`` = singular values of ``A_res`` (residual stretch magnitudes), or
    ``None`` when too few correspondences for a stable affine.
    """
    moving_rot = _rotate_face(moving, ang, center)
    src, dst, conf = block_match(
        fixed, moving_rot, None, method=metric, grid=grid,
        half=half, search=search, workers=workers,
    )
    if len(src) < _MIN_CELLS:
        return None
    tol = max(6.0, search / 8.0)
    A, t, inl, _support = _ransac_affine(src, dst, conf, tol=tol)
    if int(inl.sum()) < _MIN_CELLS:
        return None
    A, _ = _clamp_affine(A)  # physical guard: rein a non-physical over-stretch / area
    a = np.deg2rad(ang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    center = np.asarray(center, float)
    L = A @ R
    shift = A @ center + t - center
    s = np.linalg.svd(A, compute_uv=False)
    return L, np.asarray(shift, float), (float(s.min()), float(s.max())), int(inl.sum())


def _angle_gap(a: float, b: float) -> float:
    """Smallest absolute angular difference (deg) in ``[0, 180]`` (branch-aware)."""
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _agree_rotation(fixed, moving, center, *, mk, step: float = _SWEEP_STEP):
    """Rotation with the largest weighted RANSAC support, via a two-stage sweep.

    Stage 1 ranks every angle in ``[-180, 180)`` by a *cheap* confidence-weighted RANSAC
    fit (half-resolution faces + coarse block-match grid) — just enough to locate the
    basin and the opposite 180° branch. Stage 2 recomputes support at the full grid /
    resolution around those two seeds, so the committed pose never inherits the coarse
    ranking grid. The score is the weighted **inlier support** (summed confidence of
    cells consistent with one rigid transform): a few confident, geometrically consistent
    landmarks beat a spray of low-information cells, and a wrong 180° flip loses because
    it leaves few consistent pairs. The ranking pass is ~10× cheaper per angle — full
    grid × every swept angle was the cost of this stage (see tmp/coarse_warp).

    :return: ``(win, opp)`` — branch records ``dict(rot, shift, agree, ncell, support)``;
        ``win`` = best angle, ``opp`` = best angle in the OPPOSITE (>90°) branch (``None``
        if only one branch swept). ``opp`` support near ``win``'s = ambiguous 180° flip.
    """
    # Stage 1 — cheap RANKING sweep (coarse grid; half-res when the face is big enough
    # for the coarse block-match geometry — half=64 + search). Support only ORDERS
    # angles; every committed value is recomputed full-res in stage 2, so the ranking
    # grid never leaks into the result. Tiny faces (unit tests) stay full-res.
    search_c = max(32, int(mk["search"]) // 2)
    ds_c = 2 if min(fixed.shape) // 2 >= 2 * (64 + search_c) + 8 else 1
    fx_c, mv_c = fixed[::ds_c, ::ds_c], moving[::ds_c, ::ds_c]
    center_c = np.asarray(center, float) / ds_c
    mk_c = dict(
        mk, grid=max(4, int(mk["grid"]) // 2),
        search=search_c, tol=max(6.0, float(mk["tol"]) / 2.0), half=64,
    )
    coarse = {
        round(float(a), 3): _match_rigid(fx_c, mv_c, a, center_c, **mk_c)["support"]
        for a in np.arange(-180.0, 180.0, step)
    }
    c_best = max(coarse, key=coarse.get)
    opp_pool = [a for a in coarse if _angle_gap(a, c_best) > 90.0]
    c_opp = max(opp_pool, key=coarse.get) if opp_pool else None

    # Stage 2 — full grid/res: refine around the winner, and score the opposite branch's
    # coarse winner once (so win vs opp compare on the same full-grid support scale).
    scored = {}  # sweep angle -> dict(rot, shift, agree, ncell, support)

    def at(ang):
        ang = round(float(ang), 3)
        if ang not in scored:
            scored[ang] = _match_rigid(fixed, moving, ang, center, **mk)
        return scored[ang]

    for ang in np.arange(c_best - step, c_best + step + 1e-9, step / 5.0):
        at(ang)
    if c_opp is not None:
        at(c_opp)

    best = max(scored, key=lambda a: scored[a]["support"])
    opp_cands = [a for a in scored if _angle_gap(a, best) > 90.0]
    opp = scored[max(opp_cands, key=lambda a: scored[a]["support"])] if opp_cands else None
    return scored[best], opp


def rotation_search(
    fixed,
    moving,
    *,
    metric: str = "ncc",
    full_range: float = 180.0,
    coarse_step: float = 5.0,
):
    """
    Coarse inter-section rotation from images (the MT rotation-search analog).

    Rotates the moving face over ``[-full_range, full_range)`` in ``coarse_step``
    increments (then two refinement passes) and keeps the angle maximising image
    similarity to the fixed face. ``metric`` selects the pre-processing + score
    (``'ncc'`` CLAHE, ``'grad'`` edges, ``'mi'`` mutual information) — robust to
    contrast / blur, like the fine matcher.

    :return: best rotation (deg) to apply to ``moving`` to align it onto ``fixed``.
    """
    prep, score_metric = _METHODS.get(metric, (None, "ncc"))
    fp = _prep(fixed, prep)
    mp = _prep(moving, prep)
    h, w = fixed.shape
    center = np.array([w / 2.0, h / 2.0])
    # Score over a FIXED central disk that stays in-frame under any rotation, so
    # every candidate angle is compared on the same pixels (NCC/MI over an
    # angle-dependent overlap is not comparable and lets a 180°-flip win).
    yy, xx = np.ogrid[:h, :w]
    radius = 0.45 * min(h, w)
    disk = (xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2 <= radius * radius
    fixed_valid = disk & (fp != 0)

    def score(angle):
        r = _rotate_face(mp, angle, center)
        return _similarity(fp, r, fixed_valid & (r != 0), score_metric)

    best, best_s = 0.0, -1e9
    grid = list(np.arange(-full_range, full_range, coarse_step))
    for span, step in [
        (0.0, 0.0),
        (coarse_step, coarse_step / 5.0),
        (coarse_step / 5.0, coarse_step / 25.0),
    ]:
        cands = (
            grid if step == 0.0 else np.arange(best - span, best + span + 1e-9, step)
        )
        for ang in cands:
            s = score(float(ang))
            if s > best_s:
                best_s, best = s, float(ang)
    return best


def image_only_poses(
    dataset: Dataset,
    *,
    load_downscale: int = 2,
    n_slices: int = 10,
    invert_z: bool = False,
    metric: str = "ncc",
    match_grid: int = 12,
    match_search: Optional[int] = None,
    min_inlier_frac: float = _GATE_FRAC,
    workers: int = 1,
    on_interface=None,
    log=print,
) -> List[Pose]:
    """
    Absolute per-section rigid poses from images alone (no microtubules).

    Per interface both the **rotation** and its **translation** come from one
    confidence-weighted RANSAC rigid fit on a large-window block-match, evaluated over a
    full-range angle sweep (:func:`_agree_rotation`): the angle kept is the one with the
    most weighted inlier SUPPORT — the cells that see continuing structure fit a common
    rigid transform while ~half fall on the gap-decorrelated cut surface, and a wrong
    180° flip leaves few rigidly consistent cells. The weighted support is the rotation
    score (a few high-confidence, geometrically consistent landmarks outweigh a larger
    spray of low-information cells); the plain rigid-inlier FRACTION is the translation
    confidence (the abstain gate). Choosing the angle by this downstream signal (not a
    central-disk NCC peak) resolves the 180° branch by image consensus. *Too few inlier*
    cells **abstains** to a rotation-only rigid (zero shift) rather than inject noise into
    the pose chain. The relative rigid is composed onto the running absolute pose
    (section 0 = gauge); fine deformation is added later by ``image_residual_warps``.

    :func:`.contour_rotation.contour_rotation` still runs as an independent geometry
    cross-check: an interface is **flagged** needs-review when the sweep angle and the
    contour angle disagree, when the two 180° branches tie in weighted support (a
    central-disk NCC then breaks the tie by image evidence, but the interface stays
    flagged), when contour itself is unsure, or when the translation abstained — never
    silently committed. On abstain the rotation-only rigid **keeps the RANSAC rotation**
    (it already beat its 180° flip on weighted support); only the translation is dropped
    to zero. Contour stays a cross-check (the review flag), not a rotation override.

    :param load_downscale: volume decimation for pose estimation (2 sections held).
    :param metric: image similarity for the block-match that scores both the rotation
        sweep and the committed translation.
    :param match_search: block-match search radius (px) for the translation; ``None``
        auto-sizes it to the face so the bulk drift is covered (FFT-NCC makes a large
        window cheap). Too small a window is what made the old plain-median shift fail.
    :param min_inlier_frac: abstain gate — below this rigid-inlier fraction the
        block-match pose is untrusted, so the interface gets a rotation-only rigid
        and is flagged for review.
    :param on_interface: optional inspection hook ``f(info: dict)`` called once per
        interface with the committed result (keys: ``label, fixed, moving`` MIP faces,
        ``rot`` deg, ``shift`` px, ``center`` px, ``agree``, ``ncell``, ``trans_ok``,
        ``flagged``, ``tag``, ``rot_geom`` = the contour cross-check angle, or
        ``None`` when the nucleus can't be segmented; ``support`` = committed-branch
        weighted RANSAC inlier support, ``branch_ambiguous`` = the 180° branches tied,
        ``opp_rot`` = opposite-branch angle deg or ``None``, ``pxf`` = Å per face pixel;
        ``sx``, ``sy`` = the residual-affine singular values (in-plane stretch
        magnitudes; ``(1, 1)`` when the angle abstained or the affine refine failed),
        ``aniso_committed`` = whether that ``(sx, sy)``/shear was actually committed
        into the pose, i.e. it cleared :data:`_ANISO_GATE`).
        Pure side-channel for diagnostics/visualisation and the MT cross-check
        (:func:`reconcile_image_mt`) — it does not affect the returned poses. ``None``
        (default) disables it.
    :return: list of ``n`` absolute poses (Å, section 0 = identity).
    """
    n = len(dataset.sections)
    poses: List[Pose] = [dict(IDENTITY)]
    if n < 2:
        return poses
    ns = max(1, n_slices // max(load_downscale, 1))
    flagged: List[str] = []

    prev = dataset.sections[0].load_volume(downscale=load_downscale)
    for k in range(n - 1):
        s_next = dataset.sections[k + 1]
        cur = s_next.load_volume(downscale=load_downscale)
        pxf = s_next.pixel_size  # Å per face pixel at load_downscale
        label = dataset.interface_label(k)

        # MIP faces drive the translation block-match (unchanged); mean faces drive
        # the geometry rotation (segmentation needs the smooth membrane, not a MIP).
        fixed = np.asarray(geo.zmax_face(prev, "top", ns, invert_z), np.float32)
        moving = np.asarray(geo.zmax_face(cur, "bottom", ns, invert_z), np.float32)
        fixed_mean = _mean_face(prev, "top", ns, invert_z)
        moving_mean = _mean_face(cur, "bottom", ns, invert_z)
        center = np.array([moving.shape[1] / 2.0, moving.shape[0] / 2.0])  # (x, y) px

        hf, wf = fixed.shape
        # Large search so the bulk drift is inside the window (FFT-NCC ~ free); a
        # too-small window was what made the old plain-median shift return noise.
        search = match_search or int(np.clip(min(hf, wf) // 8, 64, 256))
        tol = max(12.0, search / 6.0)  # scales with search -> noise floor stays ~2%
        # Block-match window scaled to the face so each translation cell sees a
        # CONSTANT physical area across load_downscale: a fixed 64 px half covered only
        # ~half the structure at a 2048 px (ds=2) face vs a 1024 px (ds=4) one, halving
        # the RANSAC inlier agreement and destabilising the translation (tmp/diag_
        # translation.py: agree 0.16 -> 0.3). The 64 floor leaves <=1024 px faces (the
        # validated ds=4 size, and every unit-test face) byte-for-byte unchanged.
        match_half = max(64, min(hf, wf) // 16)
        mk = dict(metric=metric, grid=match_grid, search=search, tol=tol,
                  workers=workers, half=match_half)

        # Rotation = angle with most weighted RANSAC support (see _agree_rotation);
        # contour_rotation runs as an independent geometry cross-check (review flag only).
        est = contour_rotation(fixed_mean, moving_mean)
        win, opp = _agree_rotation(fixed, moving, center, mk=mk)
        rot, shift, agree, ncell = win["rot"], win["shift"], win["agree"], win["ncell"]
        committed_ang = win["ang"]  # sweep angle of the committed branch (for affine refine)
        rot_geom = est.angle if est is not None else None  # contour cross-check (diagnostics)
        win_count = win["support"]
        opp_count = opp["support"] if opp is not None else 0.0
        # 180° branches tie when the opposite keeps nearly the winner's support
        # (near-symmetric content — sign unrecoverable from consensus alone).
        branch_ambiguous = opp is not None and opp_count >= _BRANCH_FRAC * win_count

        # Tie-break ONLY here with the central-disk NCC (fixed in-frame disk, so a flip
        # can't win on overlap): commit the nearer branch. Stays flagged either way.
        rot_ncc = None
        if branch_ambiguous:
            rot_ncc = rotation_search(fixed, moving, metric=metric)
            if _angle_gap(opp["rot"], rot_ncc) < _angle_gap(rot, rot_ncc):
                rot, shift, agree, ncell = (
                    opp["rot"], opp["shift"], opp["agree"], opp["ncell"]
                )
                committed_ang = opp["ang"]
                win_count, opp_count = opp_count, win_count

        trans_ok = ncell >= _MIN_CELLS and agree >= min_inlier_frac
        contour_disagree = est is not None and _angle_gap(rot, est.angle) > _ROT_TOL
        is_flagged = bool(branch_ambiguous or contour_disagree
                          or (est is not None and est.flagged))
        cross = f"vs contour {est.angle:+.1f}°" if est is not None else "no contour"
        tie = f" tie->ncc {rot_ncc:+.1f}°" if rot_ncc is not None else ""
        tag = (
            f"ransac-sweep rot={rot:+.1f}° (flip {win_count:.0f}v{opp_count:.0f}{tie}) "
            f"{cross}{' DISAGREE' if contour_disagree else ''}"
        )
        # Abstain: too few inlier cells -> drop the shift to zero (don't inject noise)
        # but KEEP the RANSAC rotation (it already beat its 180° flip on support). Flag.
        if not trans_ok:
            shift = np.zeros(2)
        if is_flagged or not trans_ok:
            flagged.append(label)

        # Default to the validated rigid pose. Then, with the angle locked, fit a
        # small-window affine on its own RANSAC inliers and COMMIT the (sx, sy)/shear it
        # finds — but only when the residual stretch clears _ANISO_GATE (isotropic serial
        # sections fit ~identity, so this is a no-op there; see the probe constants above).
        sx = sy = 1.0
        aniso_committed = False
        rel = geo.centroid_pose(
            rot, float(shift[0]) * pxf, float(shift[1]) * pxf, center * pxf
        )
        if trans_ok:
            # The affine window AND cell grid scale with the face so each cell covers a
            # CONSTANT physical area and the cell DENSITY is constant across
            # load_downscale: the validated (_AFFINE_HALF, match_grid) config is for a
            # ~1024 px face (ds=4); at 2048 px (ds=2) a 20 px window over a 12-cell grid
            # returned too few affine inliers to recover the stretch at all (tmp/diag_
            # aniso.py). Floors keep <=1024 px faces (ds=4, unit tests) unchanged.
            aff_scale = min(fixed.shape) / 1024.0
            aff_half = max(_AFFINE_HALF, round(_AFFINE_HALF * aff_scale))
            aff_grid = max(match_grid, round(match_grid * aff_scale))
            ref = _affine_refine(fixed, moving, committed_ang, center,
                                 metric=metric, grid=aff_grid, half=aff_half,
                                 search=search, workers=workers)
            if ref is not None:
                L, shift_aff, (sx, sy), n_aff = ref
                stretch = max(sx, sy) / max(min(sx, sy), 1e-9) - 1.0
                if stretch > _ANISO_GATE and n_aff >= _MIN_CELLS:
                    c_phys = center * pxf
                    t_rel = c_phys - L @ c_phys + shift_aff * pxf
                    rel = pose_from_matrix(L, t_rel)
                    aniso_committed = True
        poses.append(compose_poses(poses[-1], rel))
        reasons = [r for r, bad in (("sign", is_flagged), ("shift", not trans_ok)) if bad]
        review = f"  NEEDS REVIEW ({'+'.join(reasons)})" if reasons else ""
        aniso_note = f"  aniso=({sx:.3f},{sy:.3f})" if aniso_committed else ""
        log(
            f"  [image-pose] {label}: rot={rot:+.1f}°  "
            f"shift=({shift[0]:+.0f},{shift[1]:+.0f})px  "
            f"matches={ncell} agree={agree:.0%}{aniso_note}  [{tag}]{review}"
        )
        if on_interface is not None:
            on_interface(dict(
                label=label, fixed=fixed, moving=moving, rot=rot,
                shift=np.asarray(shift, float), center=center, agree=agree,
                ncell=ncell, trans_ok=trans_ok, flagged=is_flagged, tag=tag,
                rot_geom=rot_geom, support=float(win_count),
                branch_ambiguous=bool(branch_ambiguous),
                opp_rot=(float(opp["rot"]) if opp is not None else None),
                pxf=float(pxf), sx=float(sx), sy=float(sy),
                aniso_committed=bool(aniso_committed),
            ))

        dataset.sections[k].drop_volume()
        prev = cur
    dataset.sections[-1].drop_volume()

    if flagged:
        log(
            f"  [image-pose] {len(flagged)}/{n - 1} interface(s) flagged for review "
            f"(sign and/or shift): {', '.join(flagged)}"
        )
        log(
            "  [image-pose] NOTE: 'sign' = the 180° flip is weak on near-circular "
            "faces; 'shift' = too few cells agreed, so the interface kept a "
            "rotation-only rigid (decorrelated gap or a wrong rotation) — verify "
            "flagged interfaces visually."
        )
    return poses


def _pose_center_shift(pose: Pose, center_xy: np.ndarray) -> np.ndarray:
    """Net displacement (Å) of the point ``center_xy`` under ``pose`` — the inverse of
    :func:`.geometry.centroid_pose`'s ``(tx, ty)`` parametrisation.

    A 2-D similarity ``{Angle, Tx, Ty, Scale}`` can be re-expressed as "rotate+scale
    about ``center_xy``, then shift by ``t``". This returns that ``t`` =
    ``P(center) - center``, so two poses can be compared on *where they move the same
    face-centre point* regardless of how each parametrised its translation.
    """
    a = np.deg2rad(float(pose["Angle"]))
    s = float(pose.get("Scale", 1.0))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.asarray(center_xy, float)
    return np.array([pose["Tx"], pose["Ty"]], float) - c + s * (R @ c)


def reconcile_image_mt(
    dataset: Dataset,
    mt_poses: List[Pose],
    mt_rows: List[dict],
    *,
    load_downscale: int = 2,
    n_slices: int = 10,
    invert_z: bool = False,
    metric: str = "ncc",
    match_grid: int = 12,
    match_search: Optional[int] = None,
    min_inlier_frac: float = _GATE_FRAC,
    workers: int = 1,
    image_candidates: Optional[List[dict]] = None,
    log=print,
):
    """
    Dual-chain cross-check: reconcile MT poses with an independent image estimate.

    When a stack has BOTH microtubule graphs and volumes, the image-only RANSAC pose
    (:func:`image_only_poses`) is a *second* estimate of every interface — the "two
    good candidates for rotation and translation" idea. This runs that image path
    (harvesting its per-interface candidate + confidence via the ``on_interface``
    hook, no re-implementation), then per interface keeps the MT pose **unless** the
    image disagrees and is the more certain side:

    * **rotation / sign** — agreement within ``_ROT_TOL`` keeps MT. On disagreement
      the side that is *sign-confident* wins (MT: ``qc_ok`` and not ``hybrid_flag``;
      image: ``trans_ok`` and not ``branch_ambiguous``); if both or neither are
      confident, the larger inlier fraction breaks it, defaulting to MT. The sign is
      where MT is weakest (the ``intensity_qc`` lesson) and the RANSAC support is
      strongest, so this is the high-value override.
    * **translation** — *gated* on the image's own confidence: if the image abstained
      (``trans_ok`` False) the cross-check stays silent and keeps MT (this is what
      makes it low-noise — the image has no opinion across a decorrelated gap, so it
      does not flag one). Only a confident image shift that differs by more than a
      drift-scaled tolerance overrides MT, and only adopts the image when MT itself is
      weak or the larger inlier fraction favours the image.

    The chosen components rebuild that interface's relative pose (about the face
    centre, preserving MT's scale) and the absolute chain is recomposed; interfaces
    where nothing changed keep the exact MT pose. Committed poses stay MT-only when
    the image is silent or agrees, so a good MT result can never be made worse by a
    low-confidence image guess.

    :param mt_poses: the absolute MT poses (section 0 = gauge), length ``n``.
    :param mt_rows: per-interface dicts (length ``n-1``) carrying MT confidence —
        keys ``match_frac``, ``qc_ok``, ``hybrid_flag`` (e.g. ``stitch.interface_rows``).
    :param image_candidates: optional pre-harvested image candidates (the
        ``on_interface`` payloads from :func:`image_only_poses`, length ``n-1``). When
        ``None`` (default) the image path is run internally to produce them; supply it
        to skip a second volume pass or to unit-test the reconcile logic without volumes.
    :return: ``(poses, reports)`` — reconciled absolute poses and a per-interface
        report dict list for logging (sources picked, gaps, confidences, flags).
    """
    n = len(dataset.sections)
    poses = [dict(p) for p in mt_poses]
    if n < 2:
        return poses, []

    # Harvest the image candidate per interface by running the validated image path with
    # a collector hook. Its verbose logging is silenced, but the hook streams a one-line
    # progress note as each interface finishes — the harvest is the slow part, so without
    # this the stage looks frozen (no output until the whole cross-check table dumps).
    img_info: List[dict] = image_candidates
    if img_info is None:
        img_info = []

        def _collect(info):
            img_info.append(info)
            log(
                f"  [xcheck] image pose {len(img_info)}/{n - 1}: {info['label']}  "
                f"rot={info['rot']:+.1f}°  agree={info['agree']:.0%}"
            )

        image_only_poses(
            dataset, load_downscale=load_downscale, n_slices=n_slices,
            invert_z=invert_z, metric=metric, match_grid=match_grid,
            match_search=match_search, min_inlier_frac=min_inlier_frac,
            workers=workers, on_interface=_collect, log=lambda *a, **k: None,
        )

    # MT relatives straight from the committed absolute poses (robust to any global
    # smoothing inside the solver — recomposing unchanged relatives reproduces poses).
    rels = [compose_poses(invert_pose(poses[k]), poses[k + 1]) for k in range(n - 1)]
    reports: List[dict] = []
    for k in range(n - 1):
        info = img_info[k]
        pxf = float(info["pxf"])
        c = np.asarray(info["center"], float) * pxf  # face centre, Å

        rel_mt = rels[k]
        th_mt = float(rel_mt["Angle"])
        s_mt = float(rel_mt.get("Scale", 1.0))
        t_mt = _pose_center_shift(rel_mt, c)
        th_img = float(info["rot"])
        t_img = np.asarray(info["shift"], float) * pxf

        match_frac = float(mt_rows[k].get("match_frac", 0.0))
        mt_qc = bool(mt_rows[k].get("qc_ok", False))
        mt_flag = bool(mt_rows[k].get("hybrid_flag", False))
        agree = float(info["agree"])
        trans_ok = bool(info["trans_ok"])
        branch_amb = bool(info["branch_ambiguous"])
        mt_sign_ok = mt_qc and not mt_flag
        img_sign_ok = trans_ok and not branch_amb

        # --- rotation / sign ---
        rot_gap = _angle_gap(th_mt, th_img)
        rot_src = "mt"
        rot_conflict = rot_gap > _ROT_TOL
        if rot_conflict:
            if img_sign_ok and not mt_sign_ok:
                rot_src = "img"
            elif mt_sign_ok and not img_sign_ok:
                rot_src = "mt"
            elif img_sign_ok and mt_sign_ok:
                rot_src = "img" if agree > match_frac else "mt"
            else:
                rot_src = "mt"
        th_pick = th_img if rot_src == "img" else th_mt

        # --- translation (gated on the image's own confidence) ---
        drift = max(float(np.linalg.norm(t_mt)), float(np.linalg.norm(t_img)))
        dt = float(np.linalg.norm(t_img - t_mt))
        t_floor = _XCHK_SHIFT_FLOOR_PX * pxf
        t_disagree = dt > max(t_floor, _XCHK_SHIFT_FRAC * drift)
        mt_t_ok = mt_qc and match_frac >= _XCHK_MT_MATCH_MIN
        t_src = "mt"
        t_conflict = False
        if trans_ok and t_disagree:
            t_conflict = True
            if mt_t_ok:
                t_src = "img" if agree > match_frac else "mt"
            else:
                t_src = "img"
        t_pick = t_img if t_src == "img" else t_mt

        if rot_src == "img" or t_src == "img":
            rels[k] = geo.centroid_pose(
                th_pick, float(t_pick[0]), float(t_pick[1]), c, scale=s_mt
            )

        reports.append(dict(
            label=info["label"], rot_mt=th_mt, rot_img=th_img, rot_gap=rot_gap,
            rot_src=rot_src, rot_conflict=rot_conflict, dt_px=dt / pxf, t_src=t_src,
            t_conflict=t_conflict, flagged=bool(rot_conflict or t_conflict),
            match_frac=match_frac, agree=agree, mt_flag=mt_flag,
            branch_ambiguous=branch_amb, trans_ok=trans_ok,
            rot_final=float(rels[k]["Angle"]),
            rot_geom=info.get("rot_geom"),
        ))

    new_poses = [dict(mt_poses[0])]
    for k in range(n - 1):
        new_poses.append(compose_poses(new_poses[-1], rels[k]))
    return new_poses, reports
