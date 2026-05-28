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
Synthetic perturbation helpers for the serial-section stitching tests.

Test-only utilities: perturb a clean spatial graph by a known transform, run an
aligner through it, and score the recovery against the known answer. This is the
generator + scorer shared across the stitching test suite.

Deformations act on the in-plane ``(x, y)`` coordinates of a ``[N, 4]``
``[id, x, y, z]`` spatial graph (serial-section warps are 2-D-per-slice); the MT
id (column 0) and z (column 3) pass through untouched.

Four deformation families:
    * **rigid**  — rotation about Z + in-plane translation.
    * **scale**  — isotropic in-plane scaling.
    * **smooth** — a diffeomorphic RBF (sum-of-Gaussians) displacement field.
    * **vortex** — the whirlpool stressor. Modelled as a tangential displacement
      ``u(x) = strength · decay(r) · J90 · (x - c)`` added to identity. Unlike a
      pure twist (which is area-preserving, det J = 1), this folds the plane
      (det J < 0) for large ``strength`` — the case the QC gate MUST reject.

Each deformation is an analytic map, so the field diagnostics can sample its
Jacobian/curl on a grid.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# In-plane 90° rotation, used to build the tangential (swirl) displacement.
_J90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def _rot2d(angle_deg: float) -> np.ndarray:
    """2x2 in-plane rotation matrix for ``angle_deg`` degrees (CCW)."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


@dataclass
class Deformation:
    """
    An analytic in-plane deformation ``f: R^2 -> R^2``.

    :param kind: One of ``'rigid'``, ``'scale'``, ``'smooth'``, ``'vortex'``.
    :param params: The defining parameters (also the recovery ground truth for
        the parametric families).
    :param fn: The map itself, applied to an ``[M, 2]`` array of ``(x, y)``.
    """

    kind: str
    params: Dict
    fn: Callable[[np.ndarray], np.ndarray] = field(repr=False)

    def apply_xy(self, xy: np.ndarray) -> np.ndarray:
        """Apply the map to an ``[M, 2]`` array of in-plane points."""
        return self.fn(np.asarray(xy, dtype=float))

    def displacement(self, xy: np.ndarray) -> np.ndarray:
        """Displacement field ``u(x) = f(x) - x`` at ``[M, 2]`` points."""
        xy = np.asarray(xy, dtype=float)
        return self.apply_xy(xy) - xy


# --------------------------------------------------------------------------- #
# Deformation constructors
# --------------------------------------------------------------------------- #
def rigid(angle_deg: float, t: "tuple[float, float]") -> Deformation:
    """Rotation about Z by ``angle_deg`` then in-plane translation ``t``."""
    R = _rot2d(angle_deg)
    t = np.asarray(t, dtype=float)

    def fn(xy):
        return xy @ R.T + t

    return Deformation("rigid", {"angle": float(angle_deg), "t": t.copy()}, fn)


def scale(s: float, center: Optional["tuple[float, float]"] = None) -> Deformation:
    """Isotropic in-plane scaling by ``s`` about ``center`` (default origin)."""
    c = np.zeros(2) if center is None else np.asarray(center, dtype=float)

    def fn(xy):
        return c + s * (xy - c)

    return Deformation("scale", {"scale": float(s), "center": c.copy()}, fn)


def smooth_rbf(centers: np.ndarray, amps: np.ndarray, sigma: float) -> Deformation:
    """
    Smooth sum-of-Gaussians displacement field (a diffeomorphism for modest amps).

    :param centers: ``[K, 2]`` Gaussian-bump centres.
    :param amps: ``[K, 2]`` per-bump displacement amplitudes.
    :param sigma: Gaussian width (same units as coordinates).
    """
    centers = np.asarray(centers, dtype=float)
    amps = np.asarray(amps, dtype=float)
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    def fn(xy):
        # [M, K] weights, then [M, 2] displacement.
        d2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        w = np.exp(-d2 * inv2s2)
        return xy + w @ amps

    return Deformation(
        "smooth",
        {"centers": centers.copy(), "amps": amps.copy(), "sigma": float(sigma)},
        fn,
    )


def vortex(
    center: "tuple[float, float]", strength: float, radius: float
) -> Deformation:
    """
    Whirlpool stressor: tangential displacement decaying with radius.

    ``u(x) = strength · exp(-(r/radius)^2) · J90 · (x - c)``, added to identity.
    Folds the plane (det J < 0) once ``strength`` is large enough — the case the
    diffeomorphism/QC gate must detect and reject.

    :param center: Swirl centre ``(cx, cy)``.
    :param strength: Peak tangential gain (dimensionless).
    :param radius: Decay radius (coordinate units).
    """
    c = np.asarray(center, dtype=float)

    def fn(xy):
        y = xy - c
        r2 = (y**2).sum(-1)
        decay = np.exp(-r2 / (radius * radius))
        tangential = y @ _J90.T  # rotate (x - c) by +90°
        return xy + strength * decay[:, None] * tangential

    return Deformation(
        "vortex",
        {"center": c.copy(), "strength": float(strength), "radius": float(radius)},
        fn,
    )


# --------------------------------------------------------------------------- #
# Applying a deformation to a spatial graph
# --------------------------------------------------------------------------- #
def apply(coords: np.ndarray, deformation: Deformation) -> np.ndarray:
    """
    Apply a deformation to the in-plane coords of a ``[N, 4]`` spatial graph.

    Column 0 (id) and column 3 (z) are preserved; only ``(x, y)`` move.

    :param coords: ``[N, 4]`` ``[id, x, y, z]`` spatial graph.
    :param deformation: The deformation to inject.
    :return: A new ``[N, 4]`` array (input is not modified).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"coords must be [N, 4]; got shape {coords.shape}")
    out = coords.copy()
    out[:, 1:3] = deformation.apply_xy(coords[:, 1:3])
    return out


# Convenience wrappers: return (perturbed_coords, ground_truth_deformation).
def apply_rigid(coords, angle_deg, t):
    d = rigid(angle_deg, t)
    return apply(coords, d), d


def apply_scale(coords, s, center=None):
    d = scale(s, center)
    return apply(coords, d), d


def apply_vortex(coords, center, strength, radius):
    d = vortex(center, strength, radius)
    return apply(coords, d), d


def apply_smooth_nonrigid(
    coords: np.ndarray,
    rng: np.random.Generator,
    n_bumps: int = 6,
    amplitude: float = 1.0,
    sigma: Optional[float] = None,
):
    """
    Inject a random (seeded) smooth diffeomorphic deformation.

    Bump centres are drawn within the in-plane bounding box; amplitudes are
    drawn uniformly in ``[-amplitude, amplitude]`` per axis. ``sigma`` defaults
    to a fraction of the in-plane extent so the field is smooth.
    """
    coords = np.asarray(coords, dtype=float)
    xy = coords[:, 1:3]
    lo, hi = xy.min(0), xy.max(0)
    extent = float(np.linalg.norm(hi - lo))
    if sigma is None:
        sigma = 0.15 * extent
    centers = rng.uniform(lo, hi, size=(n_bumps, 2))
    amps = rng.uniform(-amplitude, amplitude, size=(n_bumps, 2))
    d = smooth_rbf(centers, amps, sigma)
    return apply(coords, d), d


# --------------------------------------------------------------------------- #
# Recovery scoring
# --------------------------------------------------------------------------- #
def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles, in degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def recovery_error(
    estimated: Dict, ground_truth: Dict, rho: float = 1.0
) -> Dict[str, float]:
    """
    Compare a recovered rigid(+scale) transform to the known truth.

    Both dicts may carry ``angle`` (deg), ``t`` (2-vector), and ``scale``.
    Translation error is reported in units of ρ so it is dataset-portable.

    :return: ``{'rot_deg', 'trans_rho', 'scale_err'}`` (keys present only for
        parameters supplied in ``ground_truth``).
    """
    out: Dict[str, float] = {}
    if "angle" in ground_truth:
        out["rot_deg"] = _angle_diff_deg(
            float(estimated.get("angle", 0.0)), float(ground_truth["angle"])
        )
    if "t" in ground_truth:
        et = np.asarray(estimated.get("t", (0.0, 0.0)), dtype=float)
        gt = np.asarray(ground_truth["t"], dtype=float)
        out["trans_rho"] = float(np.linalg.norm(et - gt) / rho)
    if "scale" in ground_truth:
        out["scale_err"] = abs(
            float(estimated.get("scale", 1.0)) - float(ground_truth["scale"])
        )
    return out


# --------------------------------------------------------------------------- #
# Case container + generator
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticCase:
    """A labelled (clean, perturbed, truth) triple for benchmarking."""

    name: str
    clean: np.ndarray
    perturbed: np.ndarray
    deformation: Deformation

    @property
    def ground_truth(self) -> Dict:
        """The known parameters that produced ``perturbed`` from ``clean``."""
        return self.deformation.params


def amplitude_sweep(
    clean: np.ndarray,
    rng: np.random.Generator,
    rigid_angles=(2.0, 5.0, 15.0),
    rigid_shift: float = 1.0,
    vortex_strengths=(0.05, 0.2),
) -> List[SyntheticCase]:
    """
    A small labelled benchmark suite over rigid magnitudes and vortex strengths.

    ``rigid_shift`` and the vortex radius are expressed as fractions of the
    in-plane extent so the suite scales with the data.
    """
    clean = np.asarray(clean, dtype=float)
    xy = clean[:, 1:3]
    lo, hi = xy.min(0), xy.max(0)
    extent = float(np.linalg.norm(hi - lo))
    center = tuple((lo + hi) / 2.0)

    cases: List[SyntheticCase] = []
    for ang in rigid_angles:
        shift = rng.uniform(-1, 1, size=2) * rigid_shift
        perturbed, d = apply_rigid(clean, ang, shift)
        cases.append(SyntheticCase(f"rigid_{ang:g}deg", clean, perturbed, d))
    for st in vortex_strengths:
        perturbed, d = apply_vortex(clean, center, st, radius=0.3 * extent)
        cases.append(SyntheticCase(f"vortex_{st:g}", clean, perturbed, d))
    return cases
