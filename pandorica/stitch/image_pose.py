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

* **coarse** — a *rotation search* (the image analog of the MT coarse rotation
  search): rotate the moving boundary-face over candidate angles and keep the one
  that maximises image similarity (NCC / MI / edge) to the fixed face, then a
  global block-match for the coarse translation;
* **fine** — the same guarded image-patch warp (``image_residual_warps``) used for
  MT-free regions, applied here over the whole (mask-free) face.

The moving face is rotated with the *same* pose operator the exporter applies, so
the returned angle plugs straight into the v2 pose convention. Per-interface
relatives compose into absolute poses (section 0 = gauge). Memory-safe: volumes
are read two at a time, downsampled.
"""

from typing import List

import numpy as np

from pandorica.stitch.transform.solver import (
    IDENTITY,
    Pose,
    compose_poses,
)
from pandorica.stitch.transform.applier import (
    make_inverse_map,
    warp_volume_slicewise,
)
from pandorica.stitch import geometry as geo
from pandorica.stitch.match import block_match, _prep, _mi, _METHODS
from pandorica.stitch.dataset import Dataset
from pandorica.stitch.contour_rotation import contour_rotation


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
    workers: int = 1,
    log=print,
) -> List[Pose]:
    """
    Absolute per-section rigid poses from images alone (no microtubules).

    Per interface the inter-section **rotation** comes from cell geometry
    (:func:`.contour_rotation.contour_rotation` — nuclear-envelope shape for the
    magnitude, organelle constellation for the 180° sign), a global block-match
    gives the residual **translation**, and the relative rigid is composed onto the
    running absolute pose (section 0 = gauge). Fine deformation is added later by
    ``image_residual_warps`` over the (mask-free) faces.

    The geometry estimator recovers the rotation magnitude reliably but the sign /
    180° flip is weak on near-circular cross-sections, so any interface with a weak
    flip vote (or where the nucleus can't be segmented) is **flagged** in the log as
    needs-review — never silently committed. When the geometry is unusable the
    estimator falls back to the dense-intensity :func:`rotation_search` (also flagged).

    :param load_downscale: volume decimation for pose estimation (2 sections held).
    :param metric: image similarity for the translation block-match (and the rotation
        fallback).
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

        # coarse rotation from cell geometry; dense-intensity search as the fallback
        est = contour_rotation(fixed_mean, moving_mean)
        if est is not None:
            rot, is_flagged = est.angle, est.flagged
            tag = (
                f"{est.source}  shape_corr={est.shape_corr:.2f}  "
                f"flip_ratio={est.flip_ratio:.2f}  blobs={est.n_blobs[0]}/{est.n_blobs[1]}"
                f"  branches=[{est.branches[0]:+.0f},{est.branches[1]:+.0f}]"
            )
        else:
            rot, is_flagged = rotation_search(fixed, moving, metric=metric), True
            tag = "geometry-unusable -> dense-NCC fallback"
        if is_flagged:
            flagged.append(label)

        moving_rot = _rotate_face(moving, rot, center)
        src, dst, _ = block_match(
            fixed, moving_rot, None, method=metric, grid=match_grid, workers=workers
        )
        shift = np.median(dst - src, axis=0) if len(src) else np.zeros(2)  # (x, y) px

        rel = geo.centroid_pose(
            rot, float(shift[0]) * pxf, float(shift[1]) * pxf, center * pxf
        )
        poses.append(compose_poses(poses[-1], rel))
        log(
            f"  [image-pose] {label}: rot={rot:+.1f}°  "
            f"shift=({shift[0]:+.0f},{shift[1]:+.0f})px  matches={len(src)}  "
            f"[{tag}]{'  NEEDS REVIEW (sign)' if is_flagged else ''}"
        )

        dataset.sections[k].drop_volume()
        prev = cur
    dataset.sections[-1].drop_volume()

    if flagged:
        log(
            f"  [image-pose] {len(flagged)}/{n - 1} interface(s) flagged for sign review: "
            f"{', '.join(flagged)}"
        )
        log(
            "  [image-pose] NOTE: rotation MAGNITUDE is reliable; the 180° flip is "
            "weak on near-circular faces — verify flagged interfaces visually."
        )
    return poses
