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
Warp-field diagnostics for serial-section stitching.

These are the quantitative guards behind the whirlpool-prevention invariant:
every composed warp must be an orientation-preserving diffeomorphism,
which means **det J >= eps AND |curl u| <= omega_max** — det J > 0 alone is *not*
sufficient, because a foldover-free field can still swirl pathologically.

Everything operates on a displacement field ``u(x)`` sampled on a regular 2-D
grid as ``U`` of shape ``[H, W, 2]`` (``u_x, u_y``) with spacings ``dx, dy``.
Deriving the metrics from a *sampled* field (not an analytic map) is deliberate:
real warps arrive as discrete displacement grids, so the same code validates
synthetic test fields and production warps alike.

The map under test is ``f(x) = x + u(x)``; its Jacobian is ``J = I + grad u``.
Gradients of displacement are dimensionless (displacement and coordinate share a
length unit), so **det J and curl are already scale-invariant** — only the
inverse-consistency residual, a length, is normalised by ρ.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

import numpy as np

# A map xy->xy may be passed as a plain callable or anything with ``apply_xy``
# (e.g. a Deformation object).
MapLike = Union[Callable[[np.ndarray], np.ndarray], object]


# --------------------------------------------------------------------------- #
# Grid sampling
# --------------------------------------------------------------------------- #
def displacement_grid(
    disp: MapLike, x: np.ndarray, y: np.ndarray, is_displacement: bool = True
) -> np.ndarray:
    """
    Sample a displacement (or map) onto the regular grid ``x`` × ``y``.

    :param disp: A callable (or object with ``displacement``/``apply_xy``) giving
        either the displacement ``u(x)`` (``is_displacement=True``) or the map
        ``f(x)`` (``is_displacement=False``) at ``[M, 2]`` points.
    :param x: 1-D array of grid x-coordinates (columns).
    :param y: 1-D array of grid y-coordinates (rows).
    :param is_displacement: Whether ``disp`` returns ``u`` (else the map ``f``).
    :return: ``U`` of shape ``[len(y), len(x), 2]``.
    """
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float))  # 'xy' indexing
    pts = np.column_stack([X.ravel(), Y.ravel()])
    fn = _resolve(disp, is_displacement)
    vals = np.asarray(fn(pts), dtype=float)
    u = vals if is_displacement else vals - pts
    return u.reshape(Y.shape[0], Y.shape[1], 2)


def _resolve(disp: MapLike, is_displacement: bool) -> Callable:
    """Return the right callable from a callable / Deformation-like object."""
    if callable(disp):
        return disp
    if is_displacement and hasattr(disp, "displacement"):
        return disp.displacement
    if not is_displacement and hasattr(disp, "apply_xy"):
        return disp.apply_xy
    raise TypeError("disp must be callable or expose displacement/apply_xy")


# --------------------------------------------------------------------------- #
# Core differential quantities
# --------------------------------------------------------------------------- #
def _partials(U: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, ...]:
    """Return (dux_dx, dux_dy, duy_dx, duy_dy) via central differences."""
    U = np.asarray(U, dtype=float)
    if U.ndim != 3 or U.shape[2] != 2:
        raise ValueError(f"U must be [H, W, 2]; got shape {U.shape}")
    ux, uy = U[..., 0], U[..., 1]
    # np.gradient on a 2-D array → [d/d(axis0=rows=y), d/d(axis1=cols=x)].
    dux_dy, dux_dx = np.gradient(ux, dy, dx)
    duy_dy, duy_dx = np.gradient(uy, dy, dx)
    return dux_dx, dux_dy, duy_dx, duy_dy


def jacobian_det(U: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """
    Determinant of the map Jacobian ``J = I + grad u`` at every grid point.

    ``det J <= 0`` marks a foldover (the warp is locally non-invertible).
    """
    dux_dx, dux_dy, duy_dx, duy_dy = _partials(U, dx, dy)
    return (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx


def curl(U: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Scalar vorticity ``omega = du_y/dx - du_x/dy`` (dimensionless)."""
    _, dux_dy, duy_dx, _ = _partials(U, dx, dy)
    return duy_dx - dux_dy


def okubo_weiss(U: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """
    Okubo-Weiss field ``W = s_n^2 + s_s^2 - omega^2``.

    ``W < 0`` is rotation-dominated (a vortex core); ``W > 0`` strain-dominated.
    """
    dux_dx, dux_dy, duy_dx, duy_dy = _partials(U, dx, dy)
    s_n = dux_dx - duy_dy  # normal strain
    s_s = duy_dx + dux_dy  # shear strain
    omega = duy_dx - dux_dy  # vorticity
    return s_n**2 + s_s**2 - omega**2


def _trim(a: np.ndarray, border: int) -> np.ndarray:
    """Drop a ``border``-wide frame to avoid one-sided edge-difference artefacts."""
    if border <= 0 or min(a.shape[:2]) <= 2 * border:
        return a
    return a[border:-border, border:-border]


def min_det_j(
    U: np.ndarray, dx: float = 1.0, dy: float = 1.0, border: int = 1
) -> float:
    """Minimum det J over the interior (most negative = worst foldover)."""
    return float(_trim(jacobian_det(U, dx, dy), border).min())


def vorticity_max(
    U: np.ndarray, dx: float = 1.0, dy: float = 1.0, border: int = 1
) -> float:
    """Maximum absolute vorticity over the interior."""
    return float(np.abs(_trim(curl(U, dx, dy), border)).max())


# --------------------------------------------------------------------------- #
# Inverse consistency
# --------------------------------------------------------------------------- #
def inverse_consistency(
    fwd: MapLike, inv: MapLike, points: np.ndarray, rho: float = 1.0
) -> Dict[str, float]:
    """
    Round-trip residual of ``inv(fwd(x))`` against identity, in units of ρ.

    A genuine forward/inverse pair returns points to themselves (residual ~0); a
    bad inverse leaves a residual. ``fwd``/``inv`` are maps ``f(x)`` (callables or
    objects exposing ``apply_xy``).
    """
    points = np.asarray(points, dtype=float)
    f = _resolve(fwd, is_displacement=False)
    g = _resolve(inv, is_displacement=False)
    res = np.linalg.norm(g(f(points)) - points, axis=1)
    return {"max_rho": float(res.max() / rho), "mean_rho": float(res.mean() / rho)}


# --------------------------------------------------------------------------- #
# Certificate
# --------------------------------------------------------------------------- #
@dataclass
class FieldCertificate:
    """
    Pass/fail record for a warp field against the diffeomorphism invariant.

    ``passed`` is True iff ``min_det_j >= eps`` AND ``max_abs_vorticity <=
    omega_max``. ``ow_min < 0`` flags the presence of a rotation-dominated
    (vortex) region; it is informational, not part of the gate.
    """

    min_det_j: float
    max_abs_vorticity: float
    ow_min: float
    eps: float
    omega_max: float
    passed: bool

    @classmethod
    def from_field(
        cls,
        U: np.ndarray,
        dx: float = 1.0,
        dy: float = 1.0,
        eps: float = 0.05,
        omega_max: float = 1.0,
        border: int = 1,
    ) -> "FieldCertificate":
        """Compute the certificate from a sampled displacement field ``U``."""
        mdj = min_det_j(U, dx, dy, border)
        vmax = vorticity_max(U, dx, dy, border)
        ow_min = float(_trim(okubo_weiss(U, dx, dy), border).min())
        passed = (mdj >= eps) and (vmax <= omega_max)
        return cls(mdj, vmax, ow_min, eps, omega_max, passed)
