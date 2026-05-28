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
Tests for the coarse rigid bootstrap (``coarse.py``).

An asymmetric "L" cloud is used so the 180° disambiguation has a signal; the
moving cloud is a known rigid transform of it, recovered to ~0 residual.
"""

import numpy as np
import pytest

from pandorica.stitch.coarse import coarse


def _L_cloud():
    """Asymmetric L: horizontal arm + a short vertical arm at one end."""
    arm_x = np.column_stack([np.linspace(0, 10, 40), np.zeros(40)])
    arm_y = np.column_stack([np.zeros(12), np.linspace(0.3, 3.0, 12)])
    return np.vstack([arm_x, arm_y])


def _move(xy, angle_deg, t, scale=1.0):
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return scale * (xy @ R.T) + np.asarray(t, float)


# --------------------------------------------------------------------------- #
# Principal axis
# --------------------------------------------------------------------------- #
def test_principal_angle_along_axes():
    x_line = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
    y_line = np.column_stack([np.zeros(50), np.linspace(0, 10, 50)])
    assert abs(coarse._principal_angle(x_line)) % 180 == pytest.approx(0.0, abs=1e-6)
    assert coarse._principal_angle(y_line) % 180 == pytest.approx(90.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Coarse recovery
# --------------------------------------------------------------------------- #
def test_coarse_recovers_rotation_and_translation():
    ref = _L_cloud()
    mov = _move(ref, angle_deg=40.0, t=(5.0, -3.0))
    tf = coarse.coarse_align(ref, mov)
    recovered = coarse.apply_rigid_xy(mov, tf["Angle"], tf["Tx"], tf["Ty"], tf["Scale"])
    rms = np.sqrt(((recovered - ref) ** 2).sum(1).mean())
    assert rms < 1e-6
    assert tf["residual"] < 1e-6


def test_coarse_handles_large_angle_with_180_disambiguation():
    ref = _L_cloud()
    # A near-180° misalignment is the worst case for axis-sign ambiguity.
    mov = _move(ref, angle_deg=170.0, t=(-4.0, 6.0))
    tf = coarse.coarse_align(ref, mov)
    recovered = coarse.apply_rigid_xy(mov, tf["Angle"], tf["Tx"], tf["Ty"], tf["Scale"])
    rms = np.sqrt(((recovered - ref) ** 2).sum(1).mean())
    assert rms < 1e-6


def test_coarse_recovers_scale_when_enabled():
    ref = _L_cloud()
    mov = _move(ref, angle_deg=25.0, t=(1.0, 2.0), scale=1.5)
    # mov = 1.5 * R @ ref + t  →  recovering mov->ref needs scale 1/1.5.
    tf = coarse.coarse_align(ref, mov, allow_scale=True)
    assert tf["Scale"] == pytest.approx(1.0 / 1.5, rel=1e-3)
    recovered = coarse.apply_rigid_xy(mov, tf["Angle"], tf["Tx"], tf["Ty"], tf["Scale"])
    rms = np.sqrt(((recovered - ref) ** 2).sum(1).mean())
    assert rms < 1e-4


def test_coarse_degenerate_input_returns_identity():
    tf = coarse.coarse_align(np.zeros((1, 2)), np.zeros((1, 2)))
    assert tf["Angle"] == 0.0 and tf["Scale"] == 1.0
    assert tf["residual"] == np.inf
