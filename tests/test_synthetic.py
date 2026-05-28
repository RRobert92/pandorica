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
Tests for the synthetic perturbation harness (``synthetic.py``).

Strategy: inject a known transform, recover it, assert the recovery error is
~0. The det J < 0 assertion for the vortex case lives in the diagnostics
test; here we verify the swirl *geometry*.
"""

import numpy as np
import pytest

from tests import serial_stitching_utils as syn


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _grid_graph(n=7, step=2.0):
    """An n×n in-plane grid as a [N, 4] [id, x, y, z] graph at z=0."""
    xs, ys = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    ids = np.arange(xy.shape[0])
    z = np.zeros(xy.shape[0])
    return np.column_stack([ids, xy[:, 0], xy[:, 1], z]).astype(float)


def estimate_rigid_2d(src_coords, dst_coords):
    """
    Kabsch recovery of the rigid map taking ``src`` xy → ``dst`` xy (same order).
    Returns ``{'angle': deg, 't': [tx, ty]}``.
    """
    P = src_coords[:, 1:3]
    Q = dst_coords[:, 1:3]
    Pm, Qm = P.mean(0), Q.mean(0)
    H = (P - Pm).T @ (Q - Qm)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    t = Qm - Pm @ R.T
    return {"angle": float(angle), "t": t}


# --------------------------------------------------------------------------- #
# Rigid / scale recovery
# --------------------------------------------------------------------------- #
def test_rigid_roundtrip_recovers_exactly():
    clean = _grid_graph()
    perturbed, gt = syn.apply_rigid(clean, angle_deg=15.0, t=(3.0, -4.0))
    est = estimate_rigid_2d(clean, perturbed)
    err = syn.recovery_error(est, gt.params, rho=2.0)
    assert err["rot_deg"] == pytest.approx(0.0, abs=1e-9)
    assert err["trans_rho"] == pytest.approx(0.0, abs=1e-9)


def test_identity_rigid_is_noop():
    clean = _grid_graph()
    perturbed, _ = syn.apply_rigid(clean, angle_deg=0.0, t=(0.0, 0.0))
    assert np.allclose(perturbed, clean)


def test_scale_roundtrip():
    clean = _grid_graph()
    perturbed, gt = syn.apply_scale(clean, s=1.3)
    # Recover scale from mean radius ratio about origin.
    r_clean = np.linalg.norm(clean[:, 1:3], axis=1)
    r_pert = np.linalg.norm(perturbed[:, 1:3], axis=1)
    mask = r_clean > 0
    est_s = float(np.mean(r_pert[mask] / r_clean[mask]))
    err = syn.recovery_error({"scale": est_s}, gt.params)
    assert err["scale_err"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Structural invariants
# --------------------------------------------------------------------------- #
def test_apply_preserves_id_and_z():
    clean = _grid_graph()
    perturbed, _ = syn.apply_rigid(clean, 20.0, (1.0, 1.0))
    assert np.array_equal(perturbed[:, 0], clean[:, 0])
    assert np.array_equal(perturbed[:, 3], clean[:, 3])


def test_apply_rejects_bad_shape():
    with pytest.raises(ValueError):
        syn.apply(np.zeros((5, 3)), syn.rigid(0.0, (0.0, 0.0)))


def test_displacement_matches_apply():
    d = syn.vortex(center=(6.0, 6.0), strength=0.1, radius=5.0)
    xy = _grid_graph()[:, 1:3]
    assert np.allclose(d.displacement(xy), d.apply_xy(xy) - xy)


# --------------------------------------------------------------------------- #
# Vortex geometry (det J < 0 assertion lives in the diagnostics test)
# --------------------------------------------------------------------------- #
def test_vortex_is_tangential_and_decays():
    center = np.array([6.0, 6.0])
    d = syn.vortex(center=tuple(center), strength=0.3, radius=4.0)

    # Displacement at the centre is zero.
    assert np.allclose(d.displacement(center[None, :]), 0.0, atol=1e-12)

    # Displacement is perpendicular to the radial direction (pure swirl).
    pts = np.array([[9.0, 6.0], [6.0, 9.0], [8.0, 8.0]])
    u = d.displacement(pts)
    radial = pts - center
    dots = np.einsum("ij,ij->i", u, radial)
    assert np.allclose(dots, 0.0, atol=1e-9)

    # Far outside the radius the swirl has decayed away.
    far = center + np.array([50.0, 0.0])
    assert np.linalg.norm(d.displacement(far[None, :])) < 1e-6


def test_vortex_strength_scales_displacement():
    center = (6.0, 6.0)
    pt = np.array([[9.0, 6.0]])
    weak = syn.vortex(center, 0.1, 4.0).displacement(pt)
    strong = syn.vortex(center, 0.4, 4.0).displacement(pt)
    assert np.linalg.norm(strong) > np.linalg.norm(weak)


# --------------------------------------------------------------------------- #
# Smooth non-rigid + sweep
# --------------------------------------------------------------------------- #
def test_smooth_nonrigid_is_reproducible():
    clean = _grid_graph()
    a, _ = syn.apply_smooth_nonrigid(clean, np.random.default_rng(0))
    b, _ = syn.apply_smooth_nonrigid(clean, np.random.default_rng(0))
    assert np.allclose(a, b)


def test_smooth_nonrigid_is_bounded():
    clean = _grid_graph()
    perturbed, _ = syn.apply_smooth_nonrigid(
        clean, np.random.default_rng(1), amplitude=0.5
    )
    disp = np.linalg.norm(perturbed[:, 1:3] - clean[:, 1:3], axis=1)
    # n_bumps=6, amplitude 0.5 per axis → bounded well below the grid extent.
    assert disp.max() < 6 * 0.5 * np.sqrt(2)


def test_amplitude_sweep_labels_and_truth():
    clean = _grid_graph()
    cases = syn.amplitude_sweep(clean, np.random.default_rng(2))
    kinds = {c.deformation.kind for c in cases}
    assert {"rigid", "vortex"} <= kinds
    for c in cases:
        assert c.clean.shape == c.perturbed.shape
        assert isinstance(c.ground_truth, dict)
