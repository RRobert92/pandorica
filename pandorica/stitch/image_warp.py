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
Image-derived **fill** warp for the MT-free regions of an interface.

The MT spline warp only constrains the deformation where microtubules are; in
regions without MTs the TPS just extrapolates and the image content there is
left poorly aligned. This module fits, **per interface**, a second guarded TPS
warp from **image** correspondences — masked, large-window, subpixel block
matching (:mod:`.match`) restricted to the MT-free area — that "taps" the empty
regions into register while leaving the MT-anchored regions to the spline warp.

The result is one ``GuardedWarp`` per interface, in the **same section-k local Å
frame** as the MT warp, so :func:`.stitch.export_stitched` sums the two and
carries them through the identical Z-blend. With an empty MT mask the same
machinery does pure image stitching (no MTs needed).

Matching is parallelised over grid cells (``workers``); memory stays bounded
because the workers operate only on the small **downsampled** face images, never
on volume-sized arrays.

**Caveat:** the two faces are different physical cut surfaces, so image matches
are reliable only for large continuing features; featureless / ambiguous patches
are dropped by the confidence and variance filters rather than forced.
"""

from typing import List, Optional, Sequence

import numpy as np
from scipy.ndimage import binary_dilation

from pandorica.stitch.transform.solver import (
    Pose,
    apply_pose,
    compose_poses,
    invert_pose,
    IDENTITY,
)
from pandorica.stitch.transform.warp import (
    fit_guarded_warp,
    GuardedWarp,
)
from pandorica.stitch.transform.applier import (
    make_inverse_map,
    warp_volume_slicewise,
)
from pandorica.stitch import geometry as geo
from pandorica.stitch.stitch import _FramedWarp
from pandorica.stitch.io import Dataset
from pandorica.stitch.match import block_match, block_match_ncc

__all__ = [
    "block_match",
    "block_match_ncc",
    "image_residual_warps",
]


def _mt_mask(pts_px: np.ndarray, shape_hw, radius_px: int) -> np.ndarray:
    """Boolean mask (True = MT present) from endpoint pixel positions, dilated."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=bool)
    if len(pts_px):
        ix = np.clip(np.round(pts_px[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(pts_px[:, 1]).astype(int), 0, h - 1)
        mask[iy, ix] = True
        if radius_px > 0:
            mask = binary_dilation(mask, iterations=int(radius_px))
    return mask


def _dedupe(src: np.ndarray, dst: np.ndarray, tol: float = 1.0):
    """Drop near-coincident source points (within ``tol`` Å) — a TPS RBF system is
    singular if any control points duplicate (image matches landing on / repeated
    MT anchors). Keeps the first of each."""
    if len(src) == 0:
        return src, dst
    key = np.round(src / tol).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    return src[idx], dst[idx]


def _faces(dataset: Dataset, n_slices: int, proj_downscale: int, invert_z: bool):
    """Per-section (top_proj, bottom_proj, px_face); loads each volume once."""
    out = []
    ns = max(1, n_slices // max(proj_downscale, 1))
    for s in dataset.sections:
        if not s.has_volume():
            out.append((None, None, 1.0))
            continue
        vol = s.load_volume(downscale=proj_downscale)
        top = np.asarray(geo.zmax_face(vol, "top", ns, invert_z), np.float32)
        bot = np.asarray(geo.zmax_face(vol, "bottom", ns, invert_z), np.float32)
        out.append((top, bot, s.pixel_size))
        s.drop_volume()
    return out


def _interface_prep(dataset, poses, mt_warps, faces, k, mask_radius_px):
    """Resample interface k's moving face into section-k frame + build MT mask/anchors.

    :return: ``(fixed, moving_aligned, mask, anchors_A, pxf)`` or ``None`` if a face
        is missing. ``mask`` True = MT present; ``anchors_A`` are MT endpoints (Å,
        section-k frame) used both for the mask and as zero-displacement warp anchors.
    """
    fixed, _, pxf = faces[k]
    _, moving, _ = faces[k + 1]
    if fixed is None or moving is None:
        return None
    rel = compose_poses(invert_pose(poses[k]), poses[k + 1])  # k+1 -> k (Å)
    mtw = None
    if mt_warps is not None and mt_warps[k] is not None and mt_warps[k].accepted:
        mtw = _FramedWarp(mt_warps[k], dict(IDENTITY), coord_to_A=pxf)
    inv = make_inverse_map(geo.pose_to_pixel(rel, pxf), warp=mtw)
    moving_al = warp_volume_slicewise(
        moving[None], inv, out_hw=fixed.shape, dtype=np.float32
    )[0]
    anchors = []
    top_ep = geo.endpoints_xy(geo.face_endpoints(dataset.sections[k].coords, "top"))
    if len(top_ep):
        anchors.append(top_ep)
    bot_ep = geo.endpoints_xy(
        geo.face_endpoints(dataset.sections[k + 1].coords, "bottom")
    )
    if len(bot_ep):
        anchors.append(apply_pose(rel, bot_ep))
    anchors_A = np.vstack(anchors) if anchors else np.empty((0, 2))
    mask = _mt_mask(anchors_A / pxf, fixed.shape, mask_radius_px)
    return fixed, moving_al, mask, anchors_A, pxf


def image_residual_warps(
    dataset: Dataset,
    poses: Sequence[Pose],
    mt_warps: Optional[Sequence] = None,
    *,
    method: str = "mi",
    n_slices: int = 10,
    proj_downscale: int = 4,
    invert_z: bool = False,
    grid: int = 16,
    half: int = 64,
    search: int = 16,
    min_peakiness: float = 4.0,
    mask_radius_px: int = 24,
    omega_max: float = 0.3,
    workers: int = 1,
    progress=None,
) -> List[GuardedWarp]:
    """
    Per-interface image-fill warps (length ``n-1``), in each section-k Å frame.

    The moving bottom face of section ``k+1`` is resampled into section ``k``'s
    frame under the rigid pose + MT warp, then masked block-matched (``method`` —
    see :func:`.match.block_match`) to section ``k``'s top face over the MT-free
    region. The residual shifts, plus the MT endpoints pinned to zero (so the fill
    vanishes at the MTs), are fit to a guarded TPS. Too few matches → identity.

    :param method: ``'mi'`` / ``'grad'`` / ``'ncc'`` matching metric.
    :param mt_warps: MT warps to pre-align before matching (``None`` = image-only).
    :param workers: processes for cell matching (memory-safe — faces are small).
    :param omega_max: vorticity bound for the guarded fit (same softness knob).
    :return: list of ``GuardedWarp`` (some identity), one per interface.
    """
    n = len(dataset.sections)
    faces = _faces(dataset, n_slices, proj_downscale, invert_z)
    warps: List[GuardedWarp] = []
    empty = fit_guarded_warp(np.empty((0, 2)), np.empty((0, 2)))
    for k in range(n - 1):
        if progress is not None:
            progress(f"image-fill {dataset.interface_label(k)}", k / max(n - 1, 1))
        prep = _interface_prep(dataset, poses, mt_warps, faces, k, mask_radius_px)
        if prep is None:
            warps.append(empty)
            continue
        fixed, moving_al, mask, anchors_A, pxf = prep
        src, dst, _ = block_match(
            fixed,
            moving_al,
            mask,
            method=method,
            grid=grid,
            half=half,
            search=search,
            min_peakiness=min_peakiness,
            workers=workers,
        )
        if len(src) < 4:
            warps.append(empty)
            continue
        fit_src, fit_dst = _dedupe(
            np.vstack([src * pxf, anchors_A]), np.vstack([dst * pxf, anchors_A])
        )
        if len(fit_src) < 4:
            warps.append(empty)
            continue
        warps.append(fit_guarded_warp(fit_src, fit_dst, omega_max=omega_max))
    return warps
