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
Image-only inter-section rotation from the eggshell / cortex tangent.

The eggshell is by far the most rigid landmark in a C. elegans embryo
plastic-section EM image: a high-contrast outer curve whose tangent direction
at the section's visible boundary changes by <1° between adjacent ~250 nm cuts.
When present, a *single* eggshell tangent segment pins the rotation; the
inter-section rotation is the difference of the two tangent angles.

Pipeline per face:

1. **Edge detection** — Canny on a contrast-stretched, mildly-blurred copy of
   the face; the eggshell shows as a long, smooth, high-contrast curve.
2. **Contour extraction** — connected edge curves via ``cv2.findContours``.
3. **Eggshell selection** — keep only contours that touch / pass close to the
   image border (the eggshell extends beyond the FOV, so its in-frame trace
   always meets the frame edge), then take the LONGEST such contour. If no
   contour qualifies, return ``None`` — the eggshell is not in this section.
4. **Tangent angle** — PCA on the contour pixels; the dominant eigenvector
   is the eggshell tangent direction, expressed in ``(-90°, 90°]`` (a line's
   orientation is mod 180° by construction).

The inter-section rotation (mov→ref, deg, mod 180°) is then
``angle_fixed - angle_moving``. The sign / full 360° resolution must come
from a complementary estimator — this primitive alone is 2-fold ambiguous
(the tangent is signed only if you also know which way along the curve is
"forward", which a simple PCA does not).

This is a *complementary* rotation source for the cases where the nuclear
contour fails: round pronuclei, lost organelle constellation. It will be
``None`` on faces that genuinely have no eggshell visible (interior cuts of
the gonad, plant-tissue specimens), and that is fine — the caller falls back
to contour_rotation.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class EggshellTangent:
    """Eggshell tangent direction in a single face (deg in (-90°, 90°])."""

    angle: float  # tangent direction (mod 180°)
    n_pixels: int  # length of the selected contour (more = more reliable)
    border_overlap: float  # fraction of contour points within ``border_px`` of frame


def _stretch(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Percentile contrast stretch → uint8."""
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def _border_overlap(contour: np.ndarray, h: int, w: int, border_px: int) -> float:
    """Fraction of contour pixels within ``border_px`` of any frame edge."""
    pts = contour.reshape(-1, 2)
    near = (
        (pts[:, 0] < border_px)
        | (pts[:, 0] >= w - border_px)
        | (pts[:, 1] < border_px)
        | (pts[:, 1] >= h - border_px)
    )
    return float(near.mean())


def _tangent_angle_pca(contour: np.ndarray) -> float:
    """
    Principal axis angle of a contour pixel set, in ``(-90°, 90°]``.

    PCA on the (x, y) points gives the dominant direction of variance — for a
    smooth curve segment this matches the tangent direction averaged along
    the curve. Returns the angle (deg) of the principal eigenvector with the
    positive x-axis, collapsed mod 180°.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    pts = pts - pts.mean(axis=0)
    # Covariance eigendecomposition (analytic for 2x2 — avoids scipy).
    cov = pts.T @ pts / max(len(pts) - 1, 1)
    a, b, c, d = cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
    # Eigenvector of larger eigenvalue: principal direction
    trace = a + d
    det = a * d - b * c
    disc = max(trace * trace / 4.0 - det, 0.0)
    lam1 = trace / 2.0 + np.sqrt(disc)
    # Eigenvector for lam1: solve (cov - lam1·I) v = 0
    if abs(b) > 1e-12:
        vx, vy = b, lam1 - a
    elif abs(c) > 1e-12:
        vx, vy = lam1 - d, c
    else:
        vx, vy = (1.0, 0.0) if a >= d else (0.0, 1.0)
    angle = np.degrees(np.arctan2(vy, vx))
    return float(((angle + 90.0) % 180.0) - 90.0)


def eggshell_tangent(
    face: np.ndarray,
    *,
    blur_sigma: float = 1.5,
    canny_lo: int = 40,
    canny_hi: int = 120,
    border_px: int = 20,
    min_border_touches: int = 8,
    min_pixels: int = 200,
) -> Optional[EggshellTangent]:
    """
    Detect the eggshell in ``face`` and return its tangent direction.

    Returns ``None`` when no contour matches the eggshell heuristics (long
    enough, sufficiently border-touching). The caller treats this as "no
    eggshell visible in this section" and falls back to another estimator.

    :param face: 2D image (mean-projected boundary slab).
    :param blur_sigma: Gaussian blur σ (px) before Canny. Suppresses speckle.
    :param canny_lo / canny_hi: Canny hysteresis thresholds.
    :param border_px: a contour is considered "edge-touching" if at least
        ``min_border_touches`` of its points lie within this many pixels of
        any frame edge. The eggshell always crosses the FOV boundary because
        its physical extent is much larger than a section, but the trace is
        a closed loop around a thin curve so only the entry/exit endpoints
        actually touch the border.
    :param min_border_touches: minimum number of points within ``border_px``
        of any frame edge. Higher → stricter (rejects internal curves).
    :param min_pixels: minimum contour length in pixels.
    """
    g = _stretch(face)
    g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    edges = cv2.Canny(g, canny_lo, canny_hi)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    h, w = face.shape
    candidates = []
    for c in contours:
        if c.shape[0] < min_pixels:
            continue
        pts = c.reshape(-1, 2)
        near = (
            (pts[:, 0] < border_px)
            | (pts[:, 0] >= w - border_px)
            | (pts[:, 1] < border_px)
            | (pts[:, 1] >= h - border_px)
        )
        n_touches = int(near.sum())
        if n_touches < min_border_touches:
            continue
        candidates.append((c, n_touches))
    if not candidates:
        return None
    # Longest qualifying contour → the eggshell
    c, n_touches = max(candidates, key=lambda t: t[0].shape[0])
    return EggshellTangent(
        angle=_tangent_angle_pca(c),
        n_pixels=int(c.shape[0]),
        border_overlap=float(n_touches) / float(c.shape[0]),
    )


def eggshell_rotation(fixed: np.ndarray, moving: np.ndarray) -> Optional[float]:
    """
    Inter-section rotation (mov→ref, mod 180°) from eggshell tangent agreement.

    Returns ``None`` if either face has no detectable eggshell. The result
    is rotation modulo 180° — the eggshell tangent is a line, not a vector,
    so a complementary primitive must resolve the sign (FM Stage B, the
    nuclear-contour estimator, or the AP-polarity signal).

    The PCA tangent returns the line angle modulo 180°. Naive subtraction of
    two such angles is ill-defined (a line at +89° and one at -89° point in
    the same direction, but their numerical difference is 178°). The
    *double-angle* representation ``exp(i·2θ)`` maps lines to a well-defined
    point on the unit circle; the rotation between two lines is
    ``arg(z_f · conj(z_m)) / 2``, recovered cleanly mod 180°.
    """
    t_f = eggshell_tangent(fixed)
    t_m = eggshell_tangent(moving)
    if t_f is None or t_m is None:
        return None
    a_f = np.deg2rad(t_f.angle) * 2.0
    a_m = np.deg2rad(t_m.angle) * 2.0
    z = np.exp(1j * (a_f - a_m))
    d = np.degrees(np.angle(z)) / 2.0
    return float(((d + 90.0) % 180.0) - 90.0)
