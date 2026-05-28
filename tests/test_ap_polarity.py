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

"""Tests for the image A–P-polarity rotation hint (``ap_polarity.py``)."""

import numpy as np
import pytest
from scipy.ndimage import rotate

from pandorica.stitch.coarse import ap_polarity as ap


def _polar_image(n=120, blob_offset=(30, 0), bright=True):
    """A disk 'cell' with an off-centre dense blob → a clear A–P direction."""
    yy, xx = np.mgrid[0:n, 0:n]
    c = n / 2
    img = np.full((n, n), 0.5, dtype=float)  # background
    disk = (xx - c) ** 2 + (yy - c) ** 2 < (n * 0.4) ** 2
    img[disk] = 0.5
    bx, by = c + blob_offset[0], c + blob_offset[1]
    blob = (xx - bx) ** 2 + (yy - by) ** 2 < (n * 0.12) ** 2
    img[blob] = 1.0 if bright else 0.0
    return img


def test_polarity_vector_points_toward_dense_blob():
    # bright blob to the +x side; dense_is_dark=False → vector points +x (angle ~0).
    img = _polar_image(blob_offset=(30, 0), bright=True)
    vec, ang = ap.density_polarity_vector(img, dense_is_dark=False)
    assert vec[0] > 0 and abs(vec[1]) < abs(vec[0])
    assert abs(ang) < 15.0


def _R(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


@pytest.mark.parametrize("scipy_deg", [30.0, -30.0, 90.0, -90.0])
def test_hint_aligns_mov_polarity_to_ref(scipy_deg):
    # Convention-independent: the returned hint, applied as R(θ) to the moving
    # polarity vector, must align it with the reference's (this is exactly what
    # coarse_fusion needs — same R(θ) convention as rotation_search).
    ref = _polar_image(blob_offset=(35, 0), bright=True)
    mov = rotate(ref, scipy_deg, reshape=False, order=1, mode="nearest")
    hint = ap.ap_rotation_hint(ref, mov, dense_is_dark=False)
    assert hint is not None
    assert abs(hint) == pytest.approx(abs(scipy_deg), abs=6.0)  # magnitude
    v_ref, _ = ap.density_polarity_vector(ref, dense_is_dark=False)
    v_mov, _ = ap.density_polarity_vector(mov, dense_is_dark=False)
    aligned = _R(hint) @ v_mov
    cos = np.dot(aligned, v_ref) / (np.linalg.norm(aligned) * np.linalg.norm(v_ref))
    assert cos > 0.99  # R(hint) rotates mov polarity onto ref polarity


def test_symmetric_image_gives_no_hint():
    # centred blob → no A–P asymmetry → weak vector → None (abstain).
    img = _polar_image(blob_offset=(0, 0), bright=True)
    hint = ap.ap_rotation_hint(img, img, dense_is_dark=False, min_magnitude=2.0)
    assert hint is None
