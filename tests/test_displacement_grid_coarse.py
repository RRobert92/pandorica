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
Tests for ``stitch._displacement_grid_coarse`` — the coarse-grid + bilinear
upsample optimisation that replaces per-canvas-pixel TPS evaluation in the
export path.

Two invariants matter:

* On a realistic canvas with a non-trivial smooth TPS, the coarse evaluation
  agrees with full per-pixel evaluation to sub-pixel error (the diffeomorphism
  guard guarantees the field is smooth, so bilinear interpolation between
  coarse samples is faithful).
* On a tiny canvas (smaller than the configured spacing), the helper falls
  back to full evaluation instead of producing a degenerate grid.
"""

import numpy as np

from pandorica.stitch.stitch import _displacement_grid_coarse, _FramedWarp
from pandorica.stitch.transform import warp


def _make_canvas_pts(hc: int, wc: int) -> np.ndarray:
    """``[Hc*Wc, 2]`` ``(x, y)`` integer canvas coords, matching the export."""
    yy, xx = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)


def _identity_frame_pose():
    """Frame pose that makes ``_FramedWarp`` a pass-through to the base warp."""
    return {"Angle": 0.0, "Tx": 0.0, "Ty": 0.0, "Scale": 1.0}


def _fit_smooth_canvas_warp(hc: int, wc: int, rng_seed: int = 0):
    """Fit a guarded TPS over a sparse grid of correspondences on the canvas."""
    rng = np.random.default_rng(rng_seed)
    # Sparse correspondence grid in canvas coords (well inside the bounds).
    xs = np.linspace(0.1 * wc, 0.9 * wc, 8)
    ys = np.linspace(0.1 * hc, 0.9 * hc, 8)
    XX, YY = np.meshgrid(xs, ys)
    src = np.column_stack([XX.ravel(), YY.ravel()])
    # A few px of smooth, low-curvature displacement (diffeomorphic by design).
    dst = src + rng.normal(scale=2.5, size=src.shape)
    return warp.fit_guarded_warp(src, dst)


def test_coarse_matches_full_eval_within_subpixel():
    """
    On a 512×512 canvas with a smooth guarded TPS, ``coarse_px=8`` matches
    full per-pixel evaluation to < 0.05 px (median) and < 0.5 px (max).

    The L∞ bound is loose because bilinear interpolation can pick up small
    error near correspondence points where the field bends; the L1 median is
    the metric the export warp actually sees.
    """
    hc, wc = 512, 512
    base = _fit_smooth_canvas_warp(hc, wc)
    assert base.accepted, "test setup: TPS must be accepted (diffeomorphic)"

    fw = _FramedWarp(base, _identity_frame_pose(), coord_to_A=1.0)
    out_pts = _make_canvas_pts(hc, wc)

    d_full = _displacement_grid_coarse(fw, (hc, wc), out_pts, coarse_px=0)
    d_coarse = _displacement_grid_coarse(fw, (hc, wc), out_pts, coarse_px=8)

    err = np.linalg.norm(d_full - d_coarse, axis=1)
    assert np.median(err) < 0.05, f"median displacement error {np.median(err):.4f} px"
    assert np.max(err) < 0.5, f"max displacement error {np.max(err):.4f} px"


def test_coarse_eval_returns_correct_shape_and_dtype():
    """Output is ``[M, 2]`` float32 regardless of the optimisation branch taken."""
    hc, wc = 200, 320
    base = _fit_smooth_canvas_warp(hc, wc)
    fw = _FramedWarp(base, _identity_frame_pose(), coord_to_A=1.0)
    out_pts = _make_canvas_pts(hc, wc)

    for coarse_px in (0, 4, 8, 16):
        d = _displacement_grid_coarse(fw, (hc, wc), out_pts, coarse_px=coarse_px)
        assert d.shape == (hc * wc, 2), f"shape wrong at coarse_px={coarse_px}"
        assert d.dtype == np.float32, f"dtype wrong at coarse_px={coarse_px}"
        assert np.isfinite(d).all(), f"non-finite at coarse_px={coarse_px}"


def test_coarse_px_zero_is_exact_passthrough():
    """``coarse_px=0`` must equal calling the warp directly — no smoothing."""
    hc, wc = 128, 128
    base = _fit_smooth_canvas_warp(hc, wc)
    fw = _FramedWarp(base, _identity_frame_pose(), coord_to_A=1.0)
    out_pts = _make_canvas_pts(hc, wc)

    d_helper = _displacement_grid_coarse(fw, (hc, wc), out_pts, coarse_px=0)
    d_direct = np.asarray(fw.displacement(out_pts), dtype=np.float32)
    np.testing.assert_allclose(d_helper, d_direct, atol=1e-6)


def test_small_canvas_falls_back_to_full_eval():
    """
    When ``coarse_px ≥ min(Hc, Wc)`` the coarse grid would be degenerate — the
    helper must fall back to full evaluation so callers don't have to special-case.
    """
    hc, wc = 16, 16  # smaller than the default 8-px spacing × a meaningful grid
    base = _fit_smooth_canvas_warp(hc, wc, rng_seed=1)
    fw = _FramedWarp(base, _identity_frame_pose(), coord_to_A=1.0)
    out_pts = _make_canvas_pts(hc, wc)

    # coarse_px = canvas size → fallback to full
    d_fallback = _displacement_grid_coarse(fw, (hc, wc), out_pts, coarse_px=wc)
    d_full = _displacement_grid_coarse(fw, (hc, wc), out_pts, coarse_px=0)
    np.testing.assert_allclose(d_fallback, d_full, atol=1e-6)


def test_identity_warp_is_zero_everywhere():
    """A warp with no correspondences (identity) gives zero displacement everywhere."""
    hc, wc = 64, 96
    base = warp._identity_warp()
    fw = _FramedWarp(base, _identity_frame_pose(), coord_to_A=1.0)
    out_pts = _make_canvas_pts(hc, wc)

    for coarse_px in (0, 8, 32):
        d = _displacement_grid_coarse(fw, (hc, wc), out_pts, coarse_px=coarse_px)
        assert np.allclose(d, 0.0), f"identity warp not zero at coarse_px={coarse_px}"
