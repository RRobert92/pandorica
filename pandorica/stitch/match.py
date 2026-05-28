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
Dependency-light block-matching primitives (cv2 + numpy only).

Kept free of any ``tardis_em`` import so the multiprocessing workers (spawned on
macOS) start cheaply — they re-import only this module, not the whole DL stack.
Provides the pluggable, contrast-robust, masked, subpixel matcher used by
:mod:`.image_warp` for image-fill, parallelised over grid cells.
"""

import multiprocessing as _mproc
from typing import Optional

import cv2
import numpy as np

# Per-method config: (face pre-processing, similarity metric).
_METHODS = {"ncc": ("clahe", "ncc"), "grad": ("grad", "ncc"), "mi": (None, "mi")}


def _prep(img: np.ndarray, mode: Optional[str]) -> np.ndarray:
    """Pre-process a face: 'clahe' (contrast-eq), 'grad' (edges), or raw."""
    img = np.asarray(img, np.float32)
    if mode == "grad":
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return np.hypot(gx, gy).astype(np.float32)
    if mode == "clahe":
        lo, hi = np.percentile(img, [1.0, 99.0])
        u = np.clip((img - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        return cv2.createCLAHE(2.0, (8, 8)).apply(u).astype(np.float32)
    return img


def _mi(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Mutual information of two equal-size patches (contrast/modality-robust)."""
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    s = h.sum()
    if s <= 0:
        return 0.0
    pxy = h / s
    px, py = pxy.sum(1), pxy.sum(0)
    nz = pxy > 0
    hxy = -(pxy[nz] * np.log(pxy[nz])).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return float(hx + hy - hxy)


def _metric_map(tmpl, region, metric, tmask=None) -> np.ndarray:
    """Similarity map of ``tmpl`` slid over ``region`` (NCC via cv2, or MI loop)."""
    if metric == "mi":
        ph, pw = tmpl.shape
        hh, ww = region.shape[0] - ph + 1, region.shape[1] - pw + 1
        ts = tmpl[::2, ::2]  # subsample for histogram speed
        fm = tmask[::2, ::2].astype(bool) if tmask is not None else None
        tflat = ts[fm] if fm is not None else ts
        m = np.empty((hh, ww), np.float32)
        for i in range(hh):
            for j in range(ww):
                cand = region[i : i + ph : 2, j : j + pw : 2]
                m[i, j] = _mi(tflat, cand[fm] if fm is not None else cand)
        return m
    if tmask is not None:
        res = cv2.matchTemplate(region, tmpl, cv2.TM_CCORR_NORMED, mask=tmask)
        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
    return cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)


def _subpix(m: np.ndarray, py: int, px: int):
    """Parabolic subpixel offset (dy, dx) of a similarity-map peak at (py, px)."""

    def off(vm, vc, vp):
        d = vm - 2 * vc + vp
        return 0.0 if abs(d) < 1e-9 else 0.5 * (vm - vp) / d

    dy = (
        off(m[py - 1, px], m[py, px], m[py + 1, px]) if 0 < py < m.shape[0] - 1 else 0.0
    )
    dx = (
        off(m[py, px - 1], m[py, px], m[py, px + 1]) if 0 < px < m.shape[1] - 1 else 0.0
    )
    return dy, dx


# --- parallel cell worker (globals set per-process via the pool initializer) --- #
_CTX: dict = {}


def _cell_init(
    fix_p, mov_p, mov_raw, mask, metric, half, search, min_pk, min_std, min_free
):
    _CTX.update(
        fix_p=fix_p,
        mov_p=mov_p,
        mov_raw=mov_raw,
        mask=mask,
        metric=metric,
        half=half,
        search=search,
        min_pk=min_pk,
        min_std=min_std,
        min_free=min_free,
    )


def _cell_work(cyx):
    """Match one grid cell using the shared context; returns (src, dst, conf) or None."""
    c = _CTX
    cy, cx = cyx
    half, search = c["half"], c["search"]
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    tmask = None
    if c["mask"] is not None:
        free = ~c["mask"][sl]
        if free.mean() < c["min_free"] or c["mov_raw"][sl][free].std() < c["min_std"]:
            return None
        tmask = free.astype(np.float32)
    elif c["mov_raw"][sl].std() < c["min_std"]:
        return None
    tmpl = c["mov_p"][sl]
    y0, x0 = cy - half - search, cx - half - search
    region = c["fix_p"][y0 : cy + half + search, x0 : cx + half + search]
    if region.shape[0] <= tmpl.shape[0] or region.shape[1] <= tmpl.shape[1]:
        return None
    m = _metric_map(tmpl, region, c["metric"], tmask)
    _, peak, _, loc = cv2.minMaxLoc(m)
    peakiness = (peak - float(np.median(m))) / (float(m.std()) + 1e-6)
    if peakiness < c["min_pk"]:
        return None
    dy, dx = _subpix(m, loc[1], loc[0])
    return [cx, cy], [x0 + loc[0] + dx + half, y0 + loc[1] + dy + half], peakiness


def block_match(
    fixed: np.ndarray,
    moving: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "ncc",
    grid: int = 16,
    half: int = 64,
    search: int = 16,
    min_peakiness: float = 4.0,
    min_std: float = 5.0,
    min_free: float = 0.3,
    workers: int = 1,
):
    """
    Large-window, subpixel, masked block-match with a pluggable contrast-robust metric.

    ``method``: ``'ncc'`` (CLAHE + normalized cross-correlation), ``'grad'`` (NCC on
    gradient/edge maps), ``'mi'`` (mutual information). Confidence is a
    metric-agnostic **peakiness** ``(peak − median)/std``. When ``mask`` (True = MT)
    is given a big window may overlap MTs — similarity uses only its MT-free pixels
    (cell kept if ≥ ``min_free`` free with texture ≥ ``min_std``).

    ``workers > 1`` matches grid cells across processes. Memory is bounded: the
    workers operate on the (downsampled) face images only — each process holds one
    copy of those small arrays, nothing volume-sized.

    :return: ``(src, dst, conf)`` source / matched ``(x, y)`` (subpixel) + peakiness.
    """
    prep, metric = _METHODS.get(method, (None, "ncc"))
    fix_p = _prep(fixed, prep)
    fix_p = fix_p - float(fix_p.mean())  # zero-mean: masked CCORR / MI stability
    mov_p = _prep(moving, prep)
    mov_p = mov_p - float(mov_p.mean())
    mov_raw = np.asarray(moving, np.float32)
    h, w = fixed.shape
    cys = np.linspace(half + search, h - half - search, grid).astype(int)
    cxs = np.linspace(half + search, w - half - search, grid).astype(int)
    cells = [(int(cy), int(cx)) for cy in cys for cx in cxs]
    args = (
        fix_p,
        mov_p,
        mov_raw,
        mask,
        metric,
        half,
        search,
        min_peakiness,
        min_std,
        min_free,
    )

    if workers and workers > 1 and len(cells) > workers:
        ctx = _mproc.get_context("spawn")
        chunk = max(1, len(cells) // (workers * 4))
        with ctx.Pool(workers, initializer=_cell_init, initargs=args) as pool:
            results = pool.map(_cell_work, cells, chunksize=chunk)
    else:
        _cell_init(*args)
        results = [_cell_work(c) for c in cells]

    src, dst, conf = [], [], []
    for r in results:
        if r is not None:
            src.append(r[0])
            dst.append(r[1])
            conf.append(r[2])
    return (
        np.array(src, float).reshape(-1, 2),
        np.array(dst, float).reshape(-1, 2),
        np.array(conf, float),
    )


def block_match_ncc(
    fixed, moving, mask=None, grid=24, half=32, search=24, min_ncc=0.3, min_std=5.0
):
    """Simple integer-peak NCC matcher (kept for tests / quick raw correlation)."""
    fixed = np.asarray(fixed, np.float32)
    moving = np.asarray(moving, np.float32)
    h, w = fixed.shape
    cys = np.linspace(half + search, h - half - search, grid).astype(int)
    cxs = np.linspace(half + search, w - half - search, grid).astype(int)
    src, dst, conf = [], [], []
    for cy in cys:
        for cx in cxs:
            sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
            if mask is not None and mask[sl].any():
                continue
            tmpl = moving[sl]
            if tmpl.shape[0] < 2 * half or tmpl.std() < min_std:
                continue
            y0, x0 = cy - half - search, cx - half - search
            region = fixed[y0 : cy + half + search, x0 : cx + half + search]
            if region.shape[0] < tmpl.shape[0] or region.shape[1] < tmpl.shape[1]:
                continue
            res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
            _, peak, _, loc = cv2.minMaxLoc(res)
            if peak < min_ncc:
                continue
            src.append([cx, cy])
            dst.append([x0 + loc[0] + half, y0 + loc[1] + half])
            conf.append(peak)
    return (
        np.array(src, float).reshape(-1, 2),
        np.array(dst, float).reshape(-1, 2),
        np.array(conf, float),
    )
