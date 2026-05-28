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
Tests for the warp-field diagnostics (``diagnostics.py``).

Every metric is checked against a closed-form analytic field so the expected
value is exact, then the vortex case is confirmed to trip the foldover gate.
"""

import numpy as np
import pytest

from pandorica.stitch.transform import diagnostics as dg
from tests import serial_stitching_utils as syn


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _grid(n=41, lo=-5.0, hi=5.0):
    x = np.linspace(lo, hi, n)
    y = np.linspace(lo, hi, n)
    return x, y, x[1] - x[0], y[1] - y[0]


def _field_from_uxy(x, y, ux_fn, uy_fn):
    """Build U[H, W, 2] from analytic u_x(x, y), u_y(x, y)."""
    X, Y = np.meshgrid(x, y)
    return np.stack([ux_fn(X, Y), uy_fn(X, Y)], axis=-1)


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def test_identity_field():
    x, y, dx, dy = _grid()
    U = np.zeros((len(y), len(x), 2))
    assert np.allclose(dg.jacobian_det(U, dx, dy), 1.0)
    assert np.allclose(dg.curl(U, dx, dy), 0.0)
    assert np.allclose(dg.okubo_weiss(U, dx, dy), 0.0)
    cert = dg.FieldCertificate.from_field(U, dx, dy, eps=0.05, omega_max=1.0)
    assert cert.passed


# --------------------------------------------------------------------------- #
# Pure rotation: det J == 1, curl == 2 sin(theta)
# --------------------------------------------------------------------------- #
def test_pure_rotation():
    x, y, dx, dy = _grid()
    angle = 10.0
    d = syn.rigid(angle, (0.0, 0.0))
    U = dg.displacement_grid(d, x, y, is_displacement=True)
    assert np.allclose(dg.jacobian_det(U, dx, dy), 1.0, atol=1e-6)
    expected_curl = 2.0 * np.sin(np.deg2rad(angle))
    assert np.allclose(dg.curl(U, dx, dy), expected_curl, atol=1e-6)


# --------------------------------------------------------------------------- #
# Pure (incompressible) strain: det J == 1 - a^2, curl == 0, OW == 4 a^2 > 0
# --------------------------------------------------------------------------- #
def test_pure_strain_is_strain_dominated():
    x, y, dx, dy = _grid()
    a = 0.1
    U = _field_from_uxy(x, y, lambda X, Y: a * X, lambda X, Y: -a * Y)
    assert np.allclose(dg.jacobian_det(U, dx, dy), 1.0 - a**2, atol=1e-6)
    assert np.allclose(dg.curl(U, dx, dy), 0.0, atol=1e-6)
    ow = dg.okubo_weiss(U, dx, dy)
    assert np.allclose(ow, 4.0 * a**2, atol=1e-6)
    assert ow.min() > 0.0  # strain-dominated everywhere


# --------------------------------------------------------------------------- #
# Simple shear: det J == 1, curl == -k, OW == 0
# --------------------------------------------------------------------------- #
def test_simple_shear():
    x, y, dx, dy = _grid()
    k = 0.2
    U = _field_from_uxy(x, y, lambda X, Y: k * Y, lambda X, Y: np.zeros_like(X))
    assert np.allclose(dg.jacobian_det(U, dx, dy), 1.0, atol=1e-6)
    assert np.allclose(dg.curl(U, dx, dy), -k, atol=1e-6)
    assert np.allclose(dg.okubo_weiss(U, dx, dy), 0.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Vortex: the foldover gate MUST reject a strong swirl
# --------------------------------------------------------------------------- #
def test_strong_vortex_folds_and_fails_certificate():
    x, y, dx, dy = _grid(n=81, lo=-8.0, hi=8.0)
    d = syn.vortex(center=(0.0, 0.0), strength=5.0, radius=3.0)
    U = dg.displacement_grid(d, x, y, is_displacement=True)

    assert dg.min_det_j(U, dx, dy) < 0.0  # genuine foldover
    assert dg.okubo_weiss(U, dx, dy).min() < 0.0  # rotation-dominated core

    cert = dg.FieldCertificate.from_field(U, dx, dy, eps=0.05, omega_max=1.0)
    assert not cert.passed
    assert cert.ow_min < 0.0


def test_moderate_swirl_passes_detj_but_vorticity_gate_catches_it():
    """
    The whole reason curl is in the gate (report 13): a swirl can keep det J > 0
    yet still be a pathological rotation. det J alone would wave it through; the
    vorticity bound is what rejects it.
    """
    x, y, dx, dy = _grid(n=81, lo=-8.0, hi=8.0)
    d = syn.vortex(center=(0.0, 0.0), strength=1.5, radius=3.0)
    U = dg.displacement_grid(d, x, y, is_displacement=True)

    assert dg.min_det_j(U, dx, dy) > 0.05  # no foldover — det J gate would pass
    assert dg.vorticity_max(U, dx, dy) > 1.0  # but it swirls hard

    cert = dg.FieldCertificate.from_field(U, dx, dy, eps=0.05, omega_max=1.0)
    assert not cert.passed  # rejected by the vorticity bound, not det J


def test_weak_smooth_field_passes_certificate():
    x, y, dx, dy = _grid()
    # A gentle smooth bump — diffeomorphic, low vorticity.
    U = _field_from_uxy(
        x,
        y,
        lambda X, Y: 0.05 * np.exp(-(X**2 + Y**2) / 8.0),
        lambda X, Y: np.zeros_like(X),
    )
    cert = dg.FieldCertificate.from_field(U, dx, dy, eps=0.0, omega_max=1.0)
    assert cert.passed
    assert cert.min_det_j > 0.0


# --------------------------------------------------------------------------- #
# Inverse consistency
# --------------------------------------------------------------------------- #
def test_inverse_consistency_true_inverse_is_zero():
    angle, t = 12.0, np.array([2.0, -1.0])
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def fwd(p):
        return p @ R.T + t

    def inv(p):
        return (p - t) @ R  # R^T applied on the right == inverse rotation

    pts = np.random.default_rng(0).uniform(-5, 5, size=(50, 2))
    res = dg.inverse_consistency(fwd, inv, pts, rho=2.0)
    assert res["max_rho"] == pytest.approx(0.0, abs=1e-9)


def test_inverse_consistency_wrong_inverse_is_nonzero():
    d = syn.rigid(20.0, (3.0, 3.0))
    pts = np.random.default_rng(1).uniform(-5, 5, size=(50, 2))
    res = dg.inverse_consistency(d.apply_xy, lambda p: p, pts, rho=1.0)
    assert res["max_rho"] > 1.0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_partials_rejects_bad_shape():
    with pytest.raises(ValueError):
        dg.jacobian_det(np.zeros((4, 4)))


def test_resolve_rejects_unusable_object():
    with pytest.raises(TypeError):
        dg.inverse_consistency(object(), object(), np.zeros((3, 2)))
