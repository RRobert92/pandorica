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

"""Tests for the rigid Coherent Point Drift matcher (``cpd.py``)."""

import numpy as np
import pytest

from pandorica.stitch.coarse import cpd


def _R(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _cloud(seed=0, n=40):
    return np.random.default_rng(seed).uniform(-50, 50, size=(n, 2))


def test_recovers_known_rigid_transform():
    X = _cloud()
    Y = X @ _R(20.0).T + np.array([5.0, -3.0])  # X = R(-20) (Y - t) ...
    # Y = R(20) X + t ⇒ aligning Y→X needs R(-20), t recovered accordingly.
    res = cpd.cpd_rigid(X, Y, w=0.0)
    aligned = res.transform(Y)
    rms = np.sqrt(((aligned - X) ** 2).sum(1).mean())
    assert rms < 1e-3


def test_soft_correspondences_are_correct():
    X = _cloud(1)
    Y = X @ _R(15.0).T + np.array([2.0, 2.0])
    res = cpd.cpd_rigid(X, Y, w=0.1)
    pairs, conf = cpd.correspondences(res)
    # row m should map to the same index column (X and Y are index-aligned here)
    correct = sum(1 for m, n, _ in pairs if m == n)
    assert correct >= int(0.9 * len(X))
    assert conf > 0.8


def test_robust_to_outliers():
    # The key BCPD/CPD benefit: decoys (non-corresponding points) added to BOTH
    # sets must not break the transform — the uniform outlier term absorbs them.
    rng = np.random.default_rng(2)
    base = rng.uniform(-50, 50, size=(30, 2))
    Y = base @ _R(25.0).T + np.array([4.0, -6.0])
    X = np.vstack([base, rng.uniform(-50, 50, size=(20, 2))])  # +20 decoys in X
    Y = np.vstack([Y, rng.uniform(-50, 50, size=(15, 2))])  # +15 decoys in Y
    res = cpd.cpd_rigid(X, Y, w=0.5)  # high outlier weight
    aligned = res.transform(Y[:30])  # the true (continuing) subset
    rms = np.sqrt(((aligned - base) ** 2).sum(1).mean())
    assert rms < 2.0  # recovers the true transform despite ~40-50% decoys


def test_recovers_scale():
    X = _cloud(3)
    Y = (X @ _R(10.0).T) * 0.8 + np.array([1.0, 1.0])  # scale 0.8 in mov→...
    res = cpd.cpd_rigid(X, Y, w=0.0, allow_scale=True)
    aligned = res.transform(Y)
    rms = np.sqrt(((aligned - X) ** 2).sum(1).mean())
    assert rms < 1e-2


def test_no_scale_keeps_unit():
    X = _cloud(4)
    Y = X @ _R(30.0).T
    res = cpd.cpd_rigid(X, Y, w=0.0, allow_scale=False)
    assert res.s == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("ang", [20.0, 90.0, -90.0, 150.0])
def test_rotation_search_cold_start_large_angle(ang):
    # Multi-seed CPD recovers large/±90° rotations from a cold start (single-start
    # CPD EM cannot — non-convex). ref = R(ang) mov ⇒ recover +ang.
    rng = np.random.default_rng(5)
    mov = rng.uniform(-50, 50, size=(30, 2))
    ref = mov @ _R(ang).T
    est = cpd.cpd_rotation_search(ref, mov)
    assert abs(((est.angle - ang) + 180) % 360 - 180) < 3.0


def test_rotation_search_robust_to_decoys():
    # The headline benefit: recover rotation with 100% decoys (no cross-section
    # partner) added to both clouds — where a gated count captures decoys.
    rng = np.random.default_rng(6)
    base = rng.uniform(-50, 50, size=(30, 2))
    ref = base @ _R(35.0).T
    ref = np.vstack([ref, rng.uniform(-50, 50, size=(30, 2))])
    mov = np.vstack([base, rng.uniform(-50, 50, size=(30, 2))])
    est = cpd.cpd_rotation_search(ref, mov, w=0.5)
    assert abs(((est.angle - 35.0) + 180) % 360 - 180) < 4.0
    assert est.n_confident >= 20  # the continuing subset is found as consensus


def test_rotation_search_reports_high_basin_margin_when_unambiguous():
    # A clean, well-determined rotation has a clear best basin: the margin to the
    # next rotation basin is comfortably > 1 (the ambiguity guard stays silent).
    rng = np.random.default_rng(8)
    mov = rng.uniform(-50, 50, size=(30, 2))
    ref = mov @ _R(40.0).T
    est = cpd.cpd_rotation_search(ref, mov)
    assert est.basin_margin > 1.5
