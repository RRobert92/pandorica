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
Tests for the guarded TPS warp (``warp.py``).

The two critical behaviours: recover a benign smooth non-rigid field, and
**refuse** a vortex (foldover) field rather than apply it.
"""

import numpy as np

from pandorica.stitch.transform import warp
from tests import serial_stitching_utils as syn


def _grid_xy(n=12, lo=-8.0, hi=8.0):
    x = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(x, x)
    return np.column_stack([X.ravel(), Y.ravel()])


# --------------------------------------------------------------------------- #
# Benign smooth field: accepted and recovered
# --------------------------------------------------------------------------- #
def test_recovers_benign_smooth_field():
    src = _grid_xy()
    d = syn.smooth_rbf(
        centers=np.array([[0.0, 0.0]]), amps=np.array([[0.6, -0.4]]), sigma=5.0
    )
    dst = d.apply_xy(src)
    w = warp.fit_guarded_warp(src, dst)
    assert w.accepted
    assert w.certificate.passed
    # At smoothing 0 the TPS interpolates the correspondences exactly.
    recovered = w.apply_xy(src)
    rms = np.sqrt(((recovered - dst) ** 2).sum(1).mean())
    assert rms < 1e-6


# --------------------------------------------------------------------------- #
# Vortex: refused
# --------------------------------------------------------------------------- #
def test_refuses_vortex_foldover():
    src = _grid_xy()
    d = syn.vortex(center=(0.0, 0.0), strength=5.0, radius=3.0)
    dst = d.apply_xy(src)
    # Bounded smoothing ladder: a genuine foldover cannot be made safe without
    # destroying the fit, so the warp must be rejected, not silently applied.
    w = warp.fit_guarded_warp(src, dst, smoothings=(0.0, 0.5, 2.0))
    assert not w.accepted
    assert not w.certificate.passed


def test_unguarded_vortex_field_folds():
    """Sanity: at smoothing 0 the vortex correspondences do produce det J < 0."""
    src = _grid_xy()
    dst = syn.vortex((0.0, 0.0), strength=5.0, radius=3.0).apply_xy(src)
    w = warp.fit_guarded_warp(src, dst, smoothings=(0.0,))
    assert w.certificate.min_det_j < 0.0


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #
def test_too_few_points_is_identity():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    w = warp.fit_guarded_warp(src, src + 0.1)
    assert w.accepted
    assert np.allclose(w.displacement(src), 0.0)


# --------------------------------------------------------------------------- #
# Tangent-continuity term (Part 2: minimise stairs)
# --------------------------------------------------------------------------- #
def _warped_tan(w, p, t, eps=1e-2):
    d = w.apply_xy((p + eps * t)[None])[0] - w.apply_xy((p - eps * t)[None])[0]
    return d / np.linalg.norm(d)


def test_tangent_term_reduces_kink_and_stays_diffeomorphism():
    # Identity position map, but every stub's in-plane tangent is 30° off the reference.
    # The soft tangent term swings the warped tangent toward the reference while the field
    # stays a clean diffeomorphism.
    src = _grid_xy(n=7, lo=-6.0, hi=6.0)
    dst = src.copy()
    a = np.deg2rad(30.0)
    mov_tan = np.tile([np.sin(a), np.cos(a)], (len(src), 1))
    ref_tan = np.tile([0.0, 1.0], (len(src), 1))
    w0 = warp.fit_guarded_warp(src, dst)
    w1 = warp.fit_guarded_warp(src, dst, src_tan=mov_tan, dst_tan=ref_tan,
                               tangent_weight=0.8, tangent_step=0.2)
    p = np.array([0.0, 0.0])
    ang0 = np.degrees(np.arccos(abs(_warped_tan(w0, p, mov_tan[0]) @ ref_tan[0])))
    ang1 = np.degrees(np.arccos(abs(_warped_tan(w1, p, mov_tan[0]) @ ref_tan[0])))
    assert ang0 > 25.0           # no term: the ~30° kink is intact
    assert ang1 < ang0 - 3.0     # term: the kink is reduced
    assert w1.accepted and w1.certificate.passed


def test_tangent_term_skips_unreliable_vertical_tangents():
    # Near-vertical stubs (tiny |tan_xy|) have no usable in-plane direction and must be
    # skipped -> no augmentation -> identity warp -> a probe tangent is left unchanged;
    # shallow stubs DO get augmented -> the probe tangent swings toward the reference.
    src = _grid_xy(n=7, lo=-6.0, hi=6.0)
    dst = src.copy()
    a = np.deg2rad(30.0)
    mov = np.tile([np.sin(a), np.cos(a)], (len(src), 1))   # reliable (|xy| = 1)
    ref = np.tile([0.0, 1.0], (len(src), 1))
    vert = np.tile([0.1, 0.1], (len(src), 1))              # |xy| ≈ 0.14 < min_xy 0.2

    def kink(w):
        wt = _warped_tan(w, np.array([0.0, 0.0]), mov[0])
        return np.degrees(np.arccos(abs(wt @ ref[0])))

    w_vert = warp.fit_guarded_warp(src, dst, src_tan=vert, dst_tan=vert, tangent_weight=0.8)
    w_shallow = warp.fit_guarded_warp(src, dst, src_tan=mov, dst_tan=ref, tangent_weight=0.8)
    assert kink(w_vert) > 27.0                  # vertical gated out -> ~30° kink intact
    assert kink(w_shallow) < kink(w_vert) - 3.0  # shallow augmented -> kink reduced


def test_tangent_term_off_by_default():
    src = _grid_xy(n=7, lo=-6.0, hi=6.0)
    dst = src + 0.1
    tan = np.tile([0.0, 1.0], (len(src), 1))
    w_off = warp.fit_guarded_warp(src, dst)
    w_zero = warp.fit_guarded_warp(src, dst, src_tan=tan, dst_tan=tan, tangent_weight=0.0)
    assert np.allclose(w_off.displacement(src), w_zero.displacement(src))
