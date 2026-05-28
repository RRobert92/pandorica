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
Tests for the global MT-endpoint rotation search (``rotation_search.py``).

Confirms it recovers arbitrary rotations incl. sign on ASYMMETRIC constellations,
and FLAGS the degenerate cases (collinear bundle → 180° flip ambiguity; radial →
angular-uniform) instead of silently returning a confident wrong answer.
"""

import numpy as np
import pytest

from pandorica.stitch.coarse import rotation_search as rs


def _eps(xy, ids=None):
    """Build endpoint dicts (zero dir → distance-only matching) from [M,2] xy."""
    xy = np.asarray(xy, float)
    ids = range(len(xy)) if ids is None else ids
    return [
        {"id": int(i), "pos": np.array([p[0], p[1], 0.0]), "dir": np.zeros(3)}
        for i, p in zip(ids, xy)
    ]


def _rot(xy, deg, about):
    a = np.deg2rad(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(xy, float) - about) @ R.T + about


def _rho(xy):
    from sklearn.neighbors import NearestNeighbors

    d, _ = NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)
    return float(np.median(d[:, 1]))


def _asymmetric_cloud(seed=0, n=40):
    """A scattered, clearly non-symmetric point set (an 'L' + blob)."""
    rng = np.random.default_rng(seed)
    arm = np.column_stack([np.linspace(0, 100, 20), rng.normal(0, 2, 20)])
    spur = np.column_stack([rng.normal(0, 2, 12), np.linspace(5, 60, 12)])
    blob = rng.normal([80, 40], 5, size=(8, 2))
    return np.vstack([arm, spur, blob])


# --------------------------------------------------------------------------- #
# Recovery on asymmetric constellations — magnitude AND sign
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("true_deg", [0.0, 25.0, 90.0, -90.0, 150.0])
def test_recovers_signed_rotation_on_asymmetric_cloud(true_deg):
    ref = _asymmetric_cloud()
    c = ref.mean(0)
    mov = _rot(
        ref, true_deg, c
    )  # mov = ref rotated by true_deg → align needs -true_deg
    est = rs.global_rotation_search(_eps(ref), _eps(mov), _rho(ref))
    # search rotates mov to match ref, so it should recover -true_deg (mod 360).
    err = ((est.angle - (-true_deg)) + 180) % 360 - 180
    assert abs(err) <= 3.0
    assert est.match_fraction > 0.9
    assert est.confident


def test_asymmetric_is_not_flagged_degenerate():
    ref = _asymmetric_cloud(seed=1)
    mov = _rot(ref, 40.0, ref.mean(0))
    est = rs.global_rotation_search(_eps(ref), _eps(mov), _rho(ref))
    assert est.flip_ratio > 1.3  # sign well-resolved
    assert not est.degenerate


# --------------------------------------------------------------------------- #
# Degeneracy detection — must FLAG, not silently mis-solve
# --------------------------------------------------------------------------- #
def test_collinear_bundle_flags_180_ambiguity():
    # A near-collinear "bundle": 180° flip maps it onto itself → sign ambiguous.
    x = np.linspace(0, 100, 30)
    line = np.column_stack([x, np.random.default_rng(0).normal(0, 1.0, 30)])
    mov = _rot(line, 30.0, line.mean(0))
    est = rs.global_rotation_search(_eps(line), _eps(mov), _rho(line))
    assert est.anisotropy > rs._ANISOTROPY_MAX or est.flip_ratio < rs._FLIP_RATIO_MIN
    assert est.degenerate
    assert not est.confident


def test_too_few_endpoints_is_safe():
    est = rs.global_rotation_search(_eps(np.zeros((2, 2))), _eps(np.zeros((2, 2))), 1.0)
    assert est.angle == 0.0 and not est.confident
