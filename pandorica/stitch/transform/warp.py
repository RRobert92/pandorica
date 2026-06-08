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
Guarded thin-plate-spline warp for serial-section stitching.

It replaces Amira's rigid-MLS — the whirlpool source — with a regularised TPS
that is *guarded* by the field diagnostics.

A TPS/RBF displacement field is fit from matched correspondences (mov -> ref).
The field is then sampled on a grid and checked against the diffeomorphism
invariant: **det J ≥ ε AND |curl u| ≤ Ω_max**, in units of ρ. If it
violates the invariant, the smoothing is escalated and the field re-fit; if no
smoothing in the allowed ladder yields a safe field, the warp is **rejected**
(``accepted=False``) — an unsafe warp is never applied.

The vorticity bound matters as much as det J: a swirl can keep det J > 0 yet be
pathological (see ``test_diagnostics``), so both gates are enforced.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from pandorica.stitch.transform.diagnostics import FieldCertificate

# Smoothing ladder, in ρ-normalised units (see fit_guarded_warp). 0 = exact
# interpolation; escalate toward smoother (more diffeomorphism-friendly) fits if
# the guard trips. The ladder is **capped**: a genuine foldover/vortex needs far
# more smoothing than this to satisfy the vorticity bound (it would have to be
# flattened away entirely), so it is rejected rather than silently over-smoothed.
_DEFAULT_SMOOTHINGS = (0.0, 1.0, 5.0, 20.0, 100.0)


def _median_nn(pts: np.ndarray) -> float:
    """Median nearest-neighbour distance of a point set (the warp's ρ fallback)."""
    if len(pts) < 2:
        return 1.0
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(d[:, 1])) or 1.0


@dataclass
class GuardedWarp:
    """
    A fitted, guard-checked TPS displacement field (mov -> ref).

    :param certificate: the diffeomorphism certificate of the accepted (or
        last-tried) field.
    :param smoothing: the TPS smoothing that produced this field.
    :param accepted: whether a field satisfying the invariant was found.
    """

    certificate: FieldCertificate
    smoothing: float
    accepted: bool
    _rbf: Optional[RBFInterpolator] = None
    _center: Optional[np.ndarray] = None  # ρ-normalisation centre
    _scale: float = 1.0  # ρ-normalisation scale

    def displacement(self, xy: np.ndarray) -> np.ndarray:
        """Displacement ``u(x)`` at ``[M, 2]`` points (zero if identity warp)."""
        xy = np.asarray(xy, dtype=float)
        if self._rbf is None:
            return np.zeros_like(xy)
        # Fit lives in ρ-normalised coordinates; map in, evaluate, scale back out.
        return self._rbf((xy - self._center) / self._scale) * self._scale

    def apply_xy(self, xy: np.ndarray) -> np.ndarray:
        """Map ``[M, 2]`` points through the warp: ``f(x) = x + u(x)``."""
        xy = np.asarray(xy, dtype=float)
        return xy + self.displacement(xy)


def _identity_warp() -> GuardedWarp:
    cert = FieldCertificate(
        min_det_j=1.0,
        max_abs_vorticity=0.0,
        ow_min=0.0,
        eps=0.0,
        omega_max=np.inf,
        passed=True,
    )
    return GuardedWarp(
        certificate=cert,
        smoothing=0.0,
        accepted=True,
        _rbf=None,
        _center=np.zeros(2),
        _scale=1.0,
    )


def _tangent_augment(srcn, dstn, src_tan, dst_tan, weight, step, min_xy):
    """Extra (src→dst) correspondences that make matched MT stubs continue more smoothly
    across the seam ("minimise stairs"). For each pair with a RELIABLE in-plane tangent on
    BOTH sides (``|tan_xy| > min_xy`` — near-vertical stubs have no usable in-plane direction
    and are skipped), step a short distance (``step``, ρ-normalised) along the moving stub's
    tangent and map it to the same step along a tangent pulled partway (by ``weight``) toward
    the reference stub's direction. A TPS fit through these extra points rotates the local
    Jacobian so the warped moving tangent swings toward the reference's. The pull is SOFT
    (``weight`` < 1) and the result still rides the vorticity/detJ guard — so it never forces a
    whirlpool, and it never fully erases a kink (a wrongly-matched pair keeps a residual stair
    rather than being smoothed into a convincing fake). Returns ``(aug_srcn, aug_dispn)``.
    """
    if src_tan is None or dst_tan is None or weight <= 0.0:
        return np.empty((0, 2)), np.empty((0, 2))
    st = np.asarray(src_tan, dtype=float)[:, :2]
    dt = np.asarray(dst_tan, dtype=float)[:, :2]
    sn = np.linalg.norm(st, axis=1)
    dn = np.linalg.norm(dt, axis=1)
    ok = (sn > min_xy) & (dn > min_xy)
    if not ok.any():
        return np.empty((0, 2)), np.empty((0, 2))
    su = st[ok] / sn[ok, None]
    du = dt[ok] / dn[ok, None]
    tgt = (1.0 - weight) * su + weight * du
    tgt = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-12)
    aug_src = srcn[ok] + step * su
    aug_dst = dstn[ok] + step * tgt
    return aug_src, aug_dst - aug_src


def fit_guarded_warp(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    rho: Optional[float] = None,
    eps: float = 0.05,
    omega_max: float = 1.0,
    grid_n: int = 48,
    pad: float = 0.1,
    smoothings: Sequence[float] = _DEFAULT_SMOOTHINGS,
    src_tan: Optional[np.ndarray] = None,
    dst_tan: Optional[np.ndarray] = None,
    tangent_weight: float = 0.0,
    tangent_step: float = 0.2,
    tangent_min_xy: float = 0.2,
) -> GuardedWarp:
    """
    Fit a foldover/vorticity-guarded TPS warp mapping ``src_xy`` onto ``dst_xy``.

    The fit is done in **ρ-normalised, centred coordinates** (divide by ρ, the
    correspondence spacing). This is essential: in raw coordinates (which may be
    ~10⁵ Å) the thin-plate-spline kernel is enormous and the ``smoothing``
    parameter is negligible, so the regularisation ladder does nothing. Working
    in ρ units makes smoothing meaningful. det J / curl are scale-invariant, so
    the certificate computed on the normalised grid is valid for the real map.

    :param src_xy: ``[m, 2]`` source (moving) correspondence points.
    :param dst_xy: ``[m, 2]`` destination (reference) correspondence points.
    :param rho: correspondence length scale for normalisation (default: median
        nearest-neighbour spacing of ``src_xy``).
    :param eps: det J floor (diffeomorphism invariant).
    :param omega_max: vorticity bound (diffeomorphism invariant).
    :param grid_n: grid resolution for sampling the field's diagnostics.
    :param pad: bounding-box padding fraction for the diagnostic grid.
    :param smoothings: ascending smoothing ladder (ρ-normalised); first safe wins.
    :param src_tan / dst_tan: ``[m, 2]`` in-plane tangents of the matched moving / reference
        stubs (same order as ``src_xy`` / ``dst_xy``) for the optional tangent-continuity term.
        ``None`` disables it.
    :param tangent_weight: strength of the tangent pull (``0`` = off; ~``0.5`` = gentle). Soft,
        so it never forces a whirlpool past the guard nor fully erases a kink.
    :param tangent_step: tangent-step length as a fraction of ρ (the Jacobian lever; ~``0.2``).
    :param tangent_min_xy: minimum in-plane tangent magnitude to use a pair (skips near-vertical
        stubs, whose in-plane direction is unreliable).
    :return: a ``GuardedWarp``. ``accepted=False`` means no safe field was found
        within the (capped) ladder and the warp must not be applied.
    """
    src = np.asarray(src_xy, dtype=float)
    dst = np.asarray(dst_xy, dtype=float)
    # Thin-plate-spline in 2-D needs >= 3 non-degenerate points; below that a
    # warp is not identifiable — fall back to identity (trivially safe).
    if len(src) < 4:
        return _identity_warp()

    scale = float(rho) if rho else _median_nn(src)
    center = src.mean(0)
    srcn = (src - center) / scale
    dstn = (dst - center) / scale
    dispn = dstn - srcn

    # Optional tangent-continuity term: extra correspondences (fit only — they do not enter
    # the diagnostic grid or the query) that pull reliable in-plane tangents toward continuity.
    fit_srcn, fit_dispn = srcn, dispn
    aug_src, aug_disp = _tangent_augment(
        srcn, dstn, src_tan, dst_tan, tangent_weight, tangent_step, tangent_min_xy
    )
    if len(aug_src):
        fit_srcn = np.vstack([srcn, aug_src])
        fit_dispn = np.vstack([dispn, aug_disp])

    # Diagnostic grid over the padded (normalised) source bounding box (original points only).
    lo, hi = srcn.min(0), srcn.max(0)
    ext = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    lo, hi = lo - pad * ext, hi + pad * ext
    x = np.linspace(lo[0], hi[0], grid_n)
    y = np.linspace(lo[1], hi[1], grid_n)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])

    last: Optional[GuardedWarp] = None
    for s in smoothings:
        # Degenerate correspondences (≥4 points but ~collinear / duplicated) make the
        # TPS system singular. A higher smoothing adds diagonal regularisation and may
        # resolve it, so skip to the next rung rather than crashing; if every rung is
        # singular the warp is not identifiable -> identity (trivially safe).
        try:
            rbf = RBFInterpolator(fit_srcn, fit_dispn, kernel="thin_plate_spline", smoothing=s)
        except np.linalg.LinAlgError:
            continue
        U = rbf(grid_pts).reshape(grid_n, grid_n, 2)
        cert = FieldCertificate.from_field(U, dx, dy, eps=eps, omega_max=omega_max)
        last = GuardedWarp(
            certificate=cert,
            smoothing=float(s),
            accepted=cert.passed,
            _rbf=rbf,
            _center=center,
            _scale=scale,
        )
        if cert.passed:
            return last
    return last if last is not None else _identity_warp()
