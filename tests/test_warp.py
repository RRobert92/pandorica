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
