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

"""Tests for dense-intensity verification (``intensity_qc.py``)."""

import numpy as np

from pandorica.stitch.pipeline import intensity_qc as iq


def _polar_image(n=120, blob_offset=(34, 0)):
    """Disk 'cell' + off-centre bright blob → asymmetric, rotation-sensitive."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = n / 2
    img = np.full((n, n), 0.5, dtype=float)
    img[(xx - c) ** 2 + (yy - c) ** 2 < (n * 0.4) ** 2] = 0.5
    bx, by = c + blob_offset[0], c + blob_offset[1]
    img[(xx - bx) ** 2 + (yy - by) ** 2 < (n * 0.12) ** 2] = 1.0
    return img


def _center():
    return np.array([60.0, 60.0])


def test_self_similarity_is_one():
    img = _polar_image()
    assert iq.image_similarity(img, img) > 0.999


def test_correct_rotation_verifies_and_beats_flip():
    ref = _polar_image()
    # mov = ref rotated by −40° about centre; applying +40° should recover ref.
    mov = iq._transform_image(ref, iq._center_rotation_pose(-40.0, _center()))
    v = iq.verify_rotation(ref, mov, 40.0, center=_center())
    assert v.verified
    assert v.score > v.flip_score  # the +40 rotation beats the +220 (flipped)


def test_wrong_sign_is_rejected():
    ref = _polar_image()
    mov = iq._transform_image(ref, iq._center_rotation_pose(-40.0, _center()))
    # Claiming the OPPOSITE sign (−40, i.e. the flip of the true +40) must fail:
    v = iq.verify_rotation(ref, mov, -140.0, center=_center())  # 180° off the truth
    assert not v.verified
    assert v.flip_score > v.score  # the flip (the true +40) agrees better


def test_symmetric_image_is_inconclusive():
    # Centred blob → rotationally symmetric → rotation can't be verified vs flip.
    ref = _polar_image(blob_offset=(0, 0))
    mov = iq._transform_image(ref, iq._center_rotation_pose(-40.0, _center()))
    v = iq.verify_rotation(ref, mov, 40.0, center=_center())
    assert abs(v.score - v.flip_score) < 0.05  # cannot distinguish → ~equal
