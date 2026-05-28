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
Tests for the ρ scale unit and boundary-landmark extraction (``scale.py``).

All cases use small hand-built synthetic graphs so the expected values are
exact; no dataset required.
"""

import numpy as np
import pytest

from pandorica.stitch.transform.scale import rho, boundary_landmarks


def _line_mt(mt_id, xy, z_vals):
    """One straight MT: column-0 id, fixed (x, y), varying z → [n, 4]."""
    x, y = xy
    return np.array([[mt_id, x, y, z] for z in z_vals], dtype=float)


# --------------------------------------------------------------------------- #
# rho
# --------------------------------------------------------------------------- #
def test_rho_uniform_spacing():
    """
    On a 1-D chain spaced 2.0 apart every nearest-neighbour distance is 2.0,
    so ρ (the mean NN distance) is exactly 2.0.
    """
    coords = np.array(
        [
            [0, 0.0, 0.0, 0.0],
            [0, 2.0, 0.0, 0.0],
            [0, 4.0, 0.0, 0.0],
            [0, 6.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    assert rho(coords) == pytest.approx(2.0)


def test_rho_accepts_xyz_only():
    """rho works on a bare [N, 3] cloud as well as the [N, 4] contract."""
    xyz = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=float)
    assert rho(xyz) == pytest.approx(3.0)


def test_rho_rejects_bad_shape():
    with pytest.raises(ValueError):
        rho(np.zeros((5, 2)))


# --------------------------------------------------------------------------- #
# boundary_landmarks
# --------------------------------------------------------------------------- #
def _two_mts():
    """Two vertical MTs spanning z ∈ [0, 10]."""
    z_vals = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    return np.vstack([_line_mt(0, (0.0, 0.0), z_vals), _line_mt(1, (5.0, 5.0), z_vals)])


def test_boundary_bottom_picks_high_z():
    """'bottom' = high-Z face: both MT endpoints land in the high-Z band."""
    coords = _two_mts()
    lms = boundary_landmarks(coords, boundary="bottom", z_band_fraction=0.15)
    assert len(lms) == 2
    assert {lm["id"] for lm in lms} == {0, 1}
    # z_max=10, band=1.5 → threshold 8.5; endpoints must be at the top.
    for lm in lms:
        assert lm["pos"][2] >= 8.5


def test_boundary_top_picks_low_z():
    """'top' = low-Z face: endpoints land in the low-Z band."""
    coords = _two_mts()
    lms = boundary_landmarks(coords, boundary="top", z_band_fraction=0.15)
    assert len(lms) == 2
    for lm in lms:
        assert lm["pos"][2] <= 1.5


def test_boundary_landmark_has_unit_direction():
    """Each landmark carries a (roughly) unit tangent vector."""
    lms = boundary_landmarks(_two_mts(), boundary="bottom")
    assert len(lms) >= 1
    for lm in lms:
        assert np.linalg.norm(lm["dir"]) == pytest.approx(1.0, abs=1e-6)


def test_boundary_rejects_bad_face():
    with pytest.raises(ValueError):
        boundary_landmarks(_two_mts(), boundary="sideways")
