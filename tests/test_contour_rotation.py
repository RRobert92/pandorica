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

"""Tests for the geometry-based image-only rotation (``contour_rotation``).

The magnitude (contour shape) and the sign (organelle constellation) are tested as
units on controlled inputs — masks and point sets — so a failure pinpoints the stage,
and the tests don't depend on the blob detector reproducing a realistic EM texture.
A lenient end-to-end test then checks magnitude recovery on a synthetic face.
"""

import cv2
import numpy as np

from pandorica.stitch import contour_rotation as cr


def _wrap(a):
    return ((a + 180.0) % 360.0) - 180.0


def _ellipse_mask(angle_deg, axes=(120, 75), h=400, w=400):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), axes, int(angle_deg), 0, 360, 1, -1)
    return m


# --------------------------------------------------------------------------- #
# magnitude: nuclear-contour shape
# --------------------------------------------------------------------------- #
def test_contour_branches_recover_rotation_magnitude():
    sig_f = cr._radial_signature(_ellipse_mask(0))
    sig_m = cr._radial_signature(_ellipse_mask(30))  # same ellipse, rotated +30°
    branches, corr = cr._contour_branches(sig_f, sig_m)
    assert corr > 0.9  # same shape -> strong signature match
    # one branch undoes the +30° (≈ -30°); the two branches are ≈180° apart (the
    # ellipse is 2-fold symmetric, hence the flip ambiguity the blobs must resolve).
    assert min(abs(_wrap(b - (-30.0))) for b in branches) < 5.0
    assert abs(abs(_wrap(branches[0] - branches[1])) - 180.0) < 15.0


# --------------------------------------------------------------------------- #
# sign: organelle-constellation flip vote
# --------------------------------------------------------------------------- #
def test_resolve_flip_discriminates_asymmetric_constellation():
    rng = np.random.default_rng(0)
    pf = rng.uniform(-150, 150, (20, 2))  # asymmetric cloud (not centrally symmetric)
    deg = 40.0
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pb = pf @ R.T  # moving = fixed rotated by +40° about the (≈origin) centroid
    # candidates: the correct undo (-40°) and its 180° flip; the vote must pick -40°.
    angle, flip_ratio = cr._resolve_flip([-deg, _wrap(-deg + 180)], pf, pb)
    assert abs(_wrap(angle - (-deg))) < 1.0
    assert flip_ratio > 1.5  # asymmetric cloud -> the flip clearly loses


def test_resolve_flip_is_ambiguous_for_centrally_symmetric_constellation():
    # a centrally symmetric cloud (p and -p both present) cannot break the flip.
    half = np.array([[100.0, 20.0], [40.0, 90.0], [-30.0, 70.0], [80.0, -60.0]])
    pf = np.vstack([half, -half])
    deg = 40.0
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pb = pf @ R.T
    _, flip_ratio = cr._resolve_flip([-deg, _wrap(-deg + 180)], pf, pb)
    assert flip_ratio < 1.2  # both branches match equally -> weak vote (would flag)


# --------------------------------------------------------------------------- #
# segmentation gates
# --------------------------------------------------------------------------- #
def test_segment_nucleus_finds_central_blob():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:400, 0:400]
    face = (140.0 + 40.0 * np.sin(xx / 2.5) * np.sin(yy / 2.5)).astype(np.float32)
    cv2.circle(face, (200, 200), 90, 175.0, -1)  # smooth (low-variance) nucleus
    face[face == 175.0] += rng.normal(0, 2, int((face == 175.0).sum()))
    mask = cr._segment_nucleus(face)
    assert mask is not None
    area_frac = mask.sum() / face.size
    assert 0.1 < area_frac < 0.35  # the circle, not a sprawling region


def test_segment_nucleus_rejects_pure_noise():
    noise = np.random.default_rng(1).normal(140, 25, (400, 400)).astype(np.float32)
    assert cr._segment_nucleus(noise) is None  # no bounded central smooth blob


# --------------------------------------------------------------------------- #
# end-to-end (lenient: magnitude must be recovered; sign is gated separately)
# --------------------------------------------------------------------------- #
def _face(rot_deg, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:400, 0:400]
    img = (140.0 + 40.0 * np.sin(xx / 2.5) * np.sin(yy / 2.5)).astype(np.float32)
    img += rng.normal(0, 5, (400, 400)).astype(np.float32)
    img[_ellipse_mask(20 + rot_deg) > 0] = 175.0 + rng.normal(0, 2, 1)
    return img


def test_contour_rotation_recovers_magnitude_end_to_end():
    est = cr.contour_rotation(_face(0.0), _face(35.0))  # cell rotated +35°
    assert est is not None
    # a branch undoes the +35° (≈ -35°); the magnitude is recovered even if the
    # sign vote is weak (then ``flagged`` -> needs review, which is acceptable here).
    assert min(abs(_wrap(b - (-35.0))) for b in est.branches) < 8.0


def test_contour_rotation_returns_none_without_nucleus():
    noise = np.random.default_rng(2).normal(140, 25, (400, 400)).astype(np.float32)
    assert cr.contour_rotation(noise, noise.copy()) is None
