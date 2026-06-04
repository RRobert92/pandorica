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
Image-only inter-section rotation from cell geometry (no microtubules).

A geometry **cross-check** for the image-only pose stage (``image_pose``). That stage
picks rotation by image consensus (the swept angle whose block-match cells best fit one
rigid transform), which is reliable where the cells carry a rotational cue but can be
fooled on a near-circular (≈2-fold-symmetric) cross-section where a 180° flip scores
almost as well. So ``image_pose`` cross-checks its angle against the independent estimate
here and **flags** the interface on disagreement — it keeps its own RANSAC rotation
either way (a review signal, not an override). Independence is the point: a plain
boundary-face NCC ``argmax`` can sit at the noise floor at every angle (the failure that
motivated this module — see tmp/coarse_warp/DISCOVERIES.md), so a pixel-similarity peak
alone can't be trusted to pick the angle — geometry is steadier.

It recovers rotation from **geometry**, the same principle the MT path uses (match a
constellation, not pixels) but with image-derived structures:

* **magnitude** — the nuclear-envelope outline. The nucleus is segmented (the large,
  smooth, low-local-variance central region), its boundary taken as a radial
  signature ``r(φ)`` about the centroid, and the inter-section rotation read from the
  **circular cross-correlation** of the two signatures. This recovers the rotation
  *magnitude* reliably (≈5–13° of the MT truth on all four FemalePN interfaces) —
  but a near-circular (≈2-fold-symmetric) outline leaves a **180° branch ambiguity**;
* **sign / 180° flip** — the **organelle constellation**. The largest dark
  cytoplasmic organelles (mitochondria / vesicles) form an *asymmetric* point set;
  rotating the moving set by each contour branch and counting matches to the fixed
  set votes for the correct flip. The vote margin is the ``flip_ratio`` confidence.

The flip is the hard part: on near-symmetric specimens the organelle vote is weak
(``flip_ratio`` ≈ 1.1–1.4) and can pick the wrong branch — a wrong flip turns a
section upside-down. So interfaces with a weak vote are **flagged for review**
(``flagged=True``); they are never silently committed. This mirrors the MT path's
"recover the magnitude, gate the sign, abstain rather than mis-sign" design.

NOTE: the thresholds here are tuned on a single 5-section FemalePN stack; treat them
as starting points. Feed this **mean-projected** faces (not the Z-MIP): a MIP keeps
the brightest speckle per column and breaks the smooth membrane the segmentation
needs.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree


def _wrap(a: float) -> float:
    """Wrap an angle to (−180°, 180°]."""
    return ((float(a) + 180.0) % 360.0) - 180.0


def _segment_nucleus(face: np.ndarray) -> Optional[np.ndarray]:
    """
    Segment the nucleus = the large, smooth (low-local-variance) central blob.

    The nucleoplasm is texture-poor next to the granular cytoplasm, so a local
    standard-deviation threshold isolates it; the largest low-variance component
    whose centroid sits in the central third of the frame is taken as the nucleus.

    :return: a uint8 mask, or ``None`` if no plausible central nucleus is found.
    """
    lo, hi = np.percentile(face, [1.0, 99.0])
    g = np.clip((face - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 3)
    mean = cv2.boxFilter(g, -1, (25, 25))
    var = cv2.boxFilter(g * g, -1, (25, 25)) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    smooth = (std < np.percentile(std, 35)).astype(np.uint8)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(smooth)
    h, w = face.shape
    best, best_area = 0, 0
    for i in range(1, n):
        cx, cy = cent[i]
        central = abs(cx - w / 2) < w * 0.3 and abs(cy - h / 2) < h * 0.3
        if central and stats[i, cv2.CC_STAT_AREA] > best_area:
            best, best_area = i, stats[i, cv2.CC_STAT_AREA]
    # A real nucleus is a bounded central blob (≈¼ of the frame) ringed by
    # cytoplasm — reject a sprawling region (no smooth nucleus: e.g. blank / pure
    # texture) by area and by touching the frame border.
    if not best or not (0.04 * h * w <= best_area <= 0.40 * h * w):
        return None
    mask = (lab == best).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    ff = mask.copy()  # flood-fill holes from a corner
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    mask = (mask | (1 - ff)).astype(np.uint8)
    b = 4
    if mask[:b].any() or mask[-b:].any() or mask[:, :b].any() or mask[:, -b:].any():
        return None
    return mask


def _radial_signature(mask: np.ndarray, nbins: int = 360) -> Optional[np.ndarray]:
    """Outline of ``mask`` as a normalised radial profile ``r(φ)`` about its centroid."""
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] <= 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    ang = np.degrees(np.arctan2(cnt[:, 1] - cy, cnt[:, 0] - cx)) % 360.0
    rad = np.hypot(cnt[:, 0] - cx, cnt[:, 1] - cy)
    sig = np.full(nbins, np.nan)
    bins = (ang / (360.0 / nbins)).astype(int) % nbins
    for b in range(nbins):
        sel = rad[bins == b]
        if sel.size:
            sig[b] = sel.max()
    idx = np.arange(nbins)
    good = ~np.isnan(sig)
    if good.sum() < nbins // 2:
        return None
    sig = np.interp(idx, idx[good], sig[good], period=nbins)
    return sig / sig.mean()


def _contour_branches(
    sig_fixed: np.ndarray, sig_moving: np.ndarray
) -> Tuple[List[float], float]:
    """
    Two 180°-separated rotation candidates + shape-match quality, from the signatures.

    The circular cross-correlation of the two radial signatures peaks at the rotation
    aligning ``moving`` onto ``fixed``. A near-2-fold-symmetric outline yields two
    comparable peaks ≈180° apart — both are returned as branches for the flip vote.
    """
    n = len(sig_fixed)
    a = sig_fixed - sig_fixed.mean()
    b = sig_moving - sig_moving.mean()
    corr = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), n)
    corr /= np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    k1 = int(corr.argmax())
    sep = np.abs(((np.arange(n) - k1 + n // 2) % n) - n // 2)
    k2 = int(np.where(sep >= 90, corr, -9.0).argmax())
    branches = [_wrap(k1 * 360.0 / n), _wrap(k2 * 360.0 / n)]
    return branches, float(corr[k1])


def _detect_organelles(
    face: np.ndarray,
    nucleus: np.ndarray,
    topn: int = 45,
    area_min: int = 120,
    area_max: int = 6000,
    border: int = 50,
) -> np.ndarray:
    """
    Centroids of the ``topn`` largest dark cytoplasmic organelles (an asymmetric set).

    Dark blobs are thresholded outside the (dilated) nucleus and away from the frame
    border, filtered by area, and the largest kept — a stable, distinctive
    constellation rather than every speck.
    """
    lo, hi = np.percentile(face, [1.0, 99.0])
    g = np.clip((face - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 2)
    dark = (g < np.percentile(g, 10)).astype(np.uint8)
    dark[cv2.dilate(nucleus, np.ones((31, 31), np.uint8)) > 0] = 0
    h, w = face.shape
    dark[:border] = dark[-border:] = dark[:, :border] = dark[:, -border:] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(dark)
    cand = [
        (stats[i, cv2.CC_STAT_AREA], cent[i])
        for i in range(1, n)
        if area_min <= stats[i, cv2.CC_STAT_AREA] <= area_max
    ]
    cand.sort(key=lambda t: -t[0])
    return np.array([c for _, c in cand[:topn]], dtype=float)


def _blob_inliers(pf: np.ndarray, pb: np.ndarray, angle: float, tol: float) -> int:
    """Rotate moving blobs by ``angle`` (centroid-aligned to fixed); count NN matches ≤ tol."""
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    moved = (pb - pb.mean(0)) @ R.T + pf.mean(0)
    d, _ = cKDTree(pf).query(moved)
    return int((d <= tol).sum())


def _resolve_flip(
    branches: List[float], pf: np.ndarray, pb: np.ndarray
) -> Tuple[float, float]:
    """Pick the contour branch best matching the organelle constellation; return (angle, flip_ratio)."""
    spacing = (
        float(np.median(cKDTree(pf).query(pf, k=2)[0][:, 1])) if len(pf) > 2 else 50.0
    )
    tol = 1.5 * spacing
    counts = [_blob_inliers(pf, pb, b, tol) for b in branches]
    i = int(np.argmax(counts))
    other = max((c for j, c in enumerate(counts) if j != i), default=0)
    return branches[i], counts[i] / max(other, 1e-6)


@dataclass
class ContourRotationEstimate:
    """Geometry-based image-only rotation (mov→ref, deg) + sign-confidence / flag."""

    angle: float  # resolved rotation to apply to moving to align onto fixed
    branches: Tuple[float, float]  # the two 180°-separated contour candidates
    shape_corr: float  # contour-signature correlation (magnitude-match quality)
    flip_ratio: float  # organelle-vote margin between branches (sign confidence)
    n_blobs: Tuple[int, int]  # organelles used (fixed, moving)
    flagged: bool  # weak flip / few blobs → needs human review
    source: str  # "contour+blob" | "contour+weakflip"


def contour_rotation(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    topn_blobs: int = 45,
    flip_ratio_min: float = 1.3,
    min_blobs: int = 6,
) -> Optional[ContourRotationEstimate]:
    """
    Image-only inter-section rotation from nuclear shape + organelle constellation.

    Feed **mean-projected** boundary faces (see module note). Returns ``None`` when
    the geometry is unusable (nucleus not segmentable / no contour), so the caller can
    fall back. Otherwise returns a :class:`ContourRotationEstimate`; ``.flagged`` is
    True when the 180° flip vote is weak (``flip_ratio < flip_ratio_min``) or there are
    too few organelles to vote — the magnitude is trustworthy but the **sign needs
    review**.

    :param topn_blobs: number of largest organelles kept per face for the flip vote.
    :param flip_ratio_min: organelle-vote margin below which the sign is flagged.
    :param min_blobs: minimum organelles per face to attempt a flip vote.
    """
    mask_f = _segment_nucleus(fixed)
    mask_m = _segment_nucleus(moving)
    if mask_f is None or mask_m is None:
        return None
    sig_f = _radial_signature(mask_f)
    sig_m = _radial_signature(mask_m)
    if sig_f is None or sig_m is None:
        return None
    branches, shape_corr = _contour_branches(sig_f, sig_m)

    pf = _detect_organelles(fixed, mask_f, topn_blobs)
    pb = _detect_organelles(moving, mask_m, topn_blobs)
    if len(pf) < min_blobs or len(pb) < min_blobs:
        # No constellation to vote: keep the higher-correlation branch, flag the sign.
        return ContourRotationEstimate(
            angle=branches[0],
            branches=(branches[0], branches[1]),
            shape_corr=shape_corr,
            flip_ratio=1.0,
            n_blobs=(len(pf), len(pb)),
            flagged=True,
            source="contour+weakflip",
        )
    angle, flip_ratio = _resolve_flip(branches, pf, pb)
    flagged = flip_ratio < flip_ratio_min
    return ContourRotationEstimate(
        angle=angle,
        branches=(branches[0], branches[1]),
        shape_corr=shape_corr,
        flip_ratio=flip_ratio,
        n_blobs=(len(pf), len(pb)),
        flagged=flagged,
        source="contour+blob" if not flagged else "contour+weakflip",
    )
