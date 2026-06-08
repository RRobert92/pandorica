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
Tests for global pose refinement with priors (``solver.py``).

Key facts under no-loop / no-multigap constraints:
  * the *unregularised* pose solve equals the greedy chain (the objective
    decouples per interface) — locked in below as documentation;
  * the **priors** are what reduce drift: the scale→1 prior kills greedy's
    multiplicative scale drift, and the smoothness prior damps far-section wander.
"""

import numpy as np
import pytest

from pandorica.stitch.transform import solver as sv


def _invert(pose):
    Ri = sv._R(pose["Angle"]).T
    t = np.array([pose["Tx"], pose["Ty"]])
    ti = -(1.0 / pose["Scale"]) * (Ri @ t)
    return {
        "Angle": -pose["Angle"],
        "Tx": float(ti[0]),
        "Ty": float(ti[1]),
        "Scale": 1.0 / pose["Scale"],
    }


def _build_chain(n=6, m=8, noise=0.0, scale_noise=0.0, seed=0):
    """GT poses (constant step, unit scale) + matched interfaces (+ optional noise)."""
    rng = np.random.default_rng(seed)
    gt = [dict(sv.IDENTITY)]
    # Realistic serial-section step: small rotation (<1°, report 04) so the
    # absolute-pose trajectory is near-linear (what the smoothness prior assumes).
    step = {"Angle": 0.5, "Tx": 6.0, "Ty": -3.0, "Scale": 1.0}
    for _ in range(n - 1):
        gt.append(sv.compose_poses(gt[-1], step))

    interfaces = []
    for k in range(n - 1):
        B = rng.uniform(-50, 50, size=(m, 2))
        glob = sv.apply_pose(gt[k + 1], B)
        A = sv.apply_pose(_invert(gt[k]), glob)
        if scale_noise > 0:  # inject a per-interface scale error (drives scale drift)
            c = A.mean(0)
            A = c + (1.0 + rng.normal(0, scale_noise)) * (A - c)
        if noise > 0:
            A = A + rng.normal(0, noise, A.shape)
        interfaces.append((A, B))
    return gt, interfaces


# --------------------------------------------------------------------------- #
# Clean data: recovered exactly (GT satisfies the priors)
# --------------------------------------------------------------------------- #
def test_noiseless_recovers_ground_truth():
    # Priors off (defaults: rigid → scale prior inactive, lambda_smooth=0): the
    # solve must fit clean data exactly, with no prior bias.
    gt, interfaces = _build_chain(noise=0.0)
    poses = sv.global_pose_refine(interfaces)
    assert sv.total_residual(poses, interfaces) < 1e-5
    assert poses[-1]["Angle"] == pytest.approx(gt[-1]["Angle"], abs=1e-4)
    assert poses[-1]["Tx"] == pytest.approx(gt[-1]["Tx"], abs=1e-3)


def test_gauge_anchor_is_identity():
    _, interfaces = _build_chain(noise=1.0)
    assert sv.global_pose_refine(interfaces)[0] == sv.IDENTITY


def test_compose_and_apply_consistent():
    a = {"Angle": 20.0, "Tx": 3.0, "Ty": -1.0, "Scale": 1.0}
    b = {"Angle": -10.0, "Tx": -2.0, "Ty": 5.0, "Scale": 1.0}
    xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(
        sv.apply_pose(a, sv.apply_pose(b, xy)),
        sv.apply_pose(sv.compose_poses(a, b), xy),
    )


# --------------------------------------------------------------------------- #
# Anisotropic / shear affine core: apply == L@x+t, compose == matmul, invert
# is analytic — all exact outside the similarity group (the generalisation that
# lets a per-section (sx, sy)/shear be carried end-to-end).
# --------------------------------------------------------------------------- #
# An aniso + shear linear part (det != 0, sx != sy, off-diagonal shear): the
# regime that a single scalar Scale cannot represent.
_L_ANISO = np.array([[1.20, 0.15], [-0.05, 0.85]])
_T_ANISO = np.array([12.0, -7.0])


def _rand_pts(seed=0, m=24):
    return np.random.default_rng(seed).normal(size=(m, 2)) * 50.0


def test_apply_pose_matches_known_affine():
    P = sv.pose_from_matrix(_L_ANISO, _T_ANISO)
    xy = _rand_pts()
    assert np.allclose(sv.apply_pose(P, xy), xy @ _L_ANISO.T + _T_ANISO)
    assert np.allclose(sv.linear_part(P), _L_ANISO)


def test_compose_exact_under_anisotropy():
    A = sv.pose_from_matrix(np.array([[0.9, -0.2], [0.1, 1.1]]), np.array([3.0, 4.0]))
    B = sv.pose_from_matrix(np.array([[1.05, 0.07], [-0.03, 0.98]]), np.array([-2.0, 5.0]))
    xy = _rand_pts(seed=1)
    # compose(A, B) == "apply B then A"; and its linear part is the matrix product.
    assert np.allclose(
        sv.apply_pose(sv.compose_poses(A, B), xy),
        sv.apply_pose(A, sv.apply_pose(B, xy)),
    )
    assert np.allclose(
        sv.linear_part(sv.compose_poses(A, B)),
        sv.linear_part(A) @ sv.linear_part(B),
    )


def test_invert_round_trips_under_anisotropy():
    P = sv.pose_from_matrix(_L_ANISO, _T_ANISO)
    xy = _rand_pts(seed=2)
    assert np.allclose(sv.apply_pose(sv.invert_pose(P), sv.apply_pose(P, xy)), xy)


def test_legacy_pose_without_L_reads_as_similarity():
    # A 4-key dict (no L* keys) must behave exactly as Scale * R(Angle) — so old
    # poses and old call sites keep working unchanged.
    legacy = {"Angle": 30.0, "Tx": 5.0, "Ty": -3.0, "Scale": 1.4}
    L_exp = 1.4 * sv._R(30.0)
    xy = _rand_pts(seed=3)
    assert np.allclose(sv.linear_part(legacy), L_exp)
    assert np.allclose(sv.apply_pose(legacy, xy), xy @ L_exp.T + np.array([5.0, -3.0]))


def test_make_pose_is_similarity_drop_in():
    # make_pose() must reproduce the old 4-key similarity literal exactly, and the
    # derived Angle/Scale view must round-trip the constructor arguments.
    P = sv.make_pose(angle=25.0, tx=1.0, ty=2.0, scale=1.3)
    L_exp = 1.3 * sv._R(25.0)
    xy = _rand_pts(seed=4)
    assert np.allclose(sv.apply_pose(P, xy), xy @ L_exp.T + np.array([1.0, 2.0]))
    assert P["Angle"] == pytest.approx(25.0)
    assert P["Scale"] == pytest.approx(1.3)


def test_polar_view_is_rotation_and_sqrt_det():
    # The derived (Angle, Scale) view of an aniso L is its polar rotation and
    # sqrt|det| — the lossy similarity readout back-compat consumers see.
    P = sv.pose_from_matrix(_L_ANISO, _T_ANISO)
    assert P["Scale"] == pytest.approx(np.sqrt(abs(np.linalg.det(_L_ANISO))))
    U, _s, Vt = np.linalg.svd(_L_ANISO)
    Rp = U @ Vt
    assert P["Angle"] == pytest.approx(np.degrees(np.arctan2(Rp[1, 0], Rp[0, 0])))


# --------------------------------------------------------------------------- #
# The finding: unregularised pose solve == greedy (objective decouples)
# --------------------------------------------------------------------------- #
def test_unregularised_solve_equals_greedy():
    _, interfaces = _build_chain(n=8, m=10, noise=2.0, seed=3)
    greedy = sv.greedy_chain(interfaces)
    refined = sv.global_pose_refine(interfaces, lambda_scale=0.0, lambda_smooth=0.0)
    assert sv.total_residual(refined, interfaces) == pytest.approx(
        sv.total_residual(greedy, interfaces), rel=1e-4
    )


# --------------------------------------------------------------------------- #
# Priors reduce drift
# --------------------------------------------------------------------------- #
def test_scale_prior_controls_scale_drift():
    gt, interfaces = _build_chain(n=8, m=12, scale_noise=0.03, seed=1)
    greedy = sv.greedy_chain(interfaces, allow_scale=True)
    refined = sv.global_pose_refine(interfaces, allow_scale=True, lambda_scale=50.0)
    # GT scale is 1 everywhere; greedy compounds the per-interface scale errors.
    assert abs(refined[-1]["Scale"] - 1.0) < abs(greedy[-1]["Scale"] - 1.0)


# --------------------------------------------------------------------------- #
# Affine mode (allow_affine=True): recover a per-section (sx, sy)/shear chain,
# and the priors as their affine generalisations.
# --------------------------------------------------------------------------- #
def _build_affine_chain(n=6, m=10, step_L=None, step_t=None, noise=0.0, seed=0):
    """GT poses from a constant affine step + matched interfaces (+ optional noise)."""
    rng = np.random.default_rng(seed)
    if step_L is None:
        step_L = np.array([[1.04, 0.06], [-0.03, 0.97]])  # aniso + shear + slight rot
    if step_t is None:
        step_t = np.array([5.0, -2.0])
    step = sv.pose_from_matrix(step_L, step_t)
    gt = [dict(sv.IDENTITY)]
    for _ in range(n - 1):
        gt.append(sv.compose_poses(gt[-1], step))

    interfaces = []
    for k in range(n - 1):
        B = rng.uniform(-50, 50, size=(m, 2))
        glob = sv.apply_pose(gt[k + 1], B)
        A = sv.apply_pose(sv.invert_pose(gt[k]), glob)
        if noise > 0:
            A = A + rng.normal(0, noise, A.shape)
        interfaces.append((A, B))
    return gt, interfaces


def test_affine_recovers_anisotropic_chain():
    # Priors off: the 6-DOF solve must fit a genuinely anisotropic+shear chain
    # exactly — the regime a similarity (allow_scale) solve structurally cannot.
    gt, interfaces = _build_affine_chain(noise=0.0)
    poses = sv.global_pose_refine(
        interfaces, allow_affine=True, lambda_scale=0.0, lambda_smooth=0.0
    )
    assert sv.total_residual(poses, interfaces) < 1e-5
    assert np.allclose(sv.linear_part(poses[-1]), sv.linear_part(gt[-1]), atol=1e-4)
    assert np.allclose(
        [poses[-1]["Tx"], poses[-1]["Ty"]], [gt[-1]["Tx"], gt[-1]["Ty"]], atol=1e-3
    )


def _aniso_total(poses):
    """Σ ‖LᵀL − I‖ over non-anchor sections — how far the chain is from a rotation."""
    return float(
        sum(
            np.linalg.norm(L.T @ L - np.eye(2))
            for L in (sv.linear_part(p) for p in poses[1:])
        )
    )


def test_pull_to_rotation_prior_reduces_anisotropy():
    # GT is a pure-rotation chain (orthogonal L); noise tempts the free affine fit
    # into spurious anisotropy/shear. The √λ(LᵀL−I) prior pulls each L back toward
    # orthogonal — the affine generalisation of scale→1.
    rng = np.random.default_rng(7)
    step = sv.make_pose(angle=0.5, tx=6.0, ty=-3.0, scale=1.0)
    gt = [dict(sv.IDENTITY)]
    for _ in range(7):
        gt.append(sv.compose_poses(gt[-1], step))
    interfaces = []
    for k in range(7):
        B = rng.uniform(-50, 50, size=(10, 2))
        glob = sv.apply_pose(gt[k + 1], B)
        A = sv.apply_pose(sv.invert_pose(gt[k]), glob) + rng.normal(0, 2.0, (10, 2))
        interfaces.append((A, B))

    free = sv.global_pose_refine(interfaces, allow_affine=True, lambda_scale=0.0)
    pulled = sv.global_pose_refine(interfaces, allow_affine=True, lambda_scale=50.0)
    assert _aniso_total(pulled) < _aniso_total(free)


def test_affine_smoothness_unbiased_on_large_constant_rotation():
    # Isolate the matrix-log claim on the LINEAR part. A big CONSTANT per-section
    # rotation is a constant additive step under logm, so its second-difference is
    # zero and the smoothness prior must NOT bias it even at high weight. (Raw-L
    # smoothness WOULD, since cos(kθ) curves in k.) The translation step is zero so
    # the absolute-translation trajectory does not spiral — translation smoothness
    # is a separate small-rotation assumption (see _build_chain) and is not under
    # test here.
    step = sv.make_pose(angle=20.0, tx=0.0, ty=0.0, scale=1.0)
    gt = [dict(sv.IDENTITY)]
    for _ in range(6):
        gt.append(sv.compose_poses(gt[-1], step))
    rng = np.random.default_rng(3)
    interfaces = []
    for k in range(6):
        B = rng.uniform(-40, 40, size=(12, 2))
        glob = sv.apply_pose(gt[k + 1], B)
        interfaces.append((sv.apply_pose(sv.invert_pose(gt[k]), glob), B))

    poses = sv.global_pose_refine(
        interfaces, allow_affine=True, lambda_scale=0.0, lambda_smooth=100.0
    )
    assert sv.total_residual(poses, interfaces) < 1e-4
    assert np.allclose(sv.linear_part(poses[-1]), sv.linear_part(gt[-1]), atol=1e-3)


def test_affine_flag_off_is_unchanged_similarity():
    # Guard the back-compat contract: with allow_affine=False the result is the
    # legacy similarity solve, byte-for-byte with what allow_scale alone produced.
    _, interfaces = _build_chain(n=7, m=10, scale_noise=0.02, seed=5)
    base = sv.global_pose_refine(interfaces, allow_scale=True, lambda_scale=10.0)
    same = sv.global_pose_refine(
        interfaces, allow_scale=True, allow_affine=False, lambda_scale=10.0
    )
    for a, b in zip(base, same):
        assert a["Scale"] == pytest.approx(b["Scale"])
        assert a["Angle"] == pytest.approx(b["Angle"])
        assert a["Tx"] == pytest.approx(b["Tx"])


def test_smoothness_prior_reduces_far_drift_on_average():
    def far_err(poses, gt):
        return np.linalg.norm(
            [poses[-1]["Tx"] - gt[-1]["Tx"], poses[-1]["Ty"] - gt[-1]["Ty"]]
        )

    g_errs, r_errs = [], []
    for seed in range(20):
        gt, interfaces = _build_chain(n=8, m=10, noise=3.0, seed=seed)
        g_errs.append(far_err(sv.greedy_chain(interfaces), gt))
        r_errs.append(
            far_err(sv.global_pose_refine(interfaces, lambda_smooth=100.0), gt)
        )
    assert np.mean(r_errs) < np.mean(g_errs)
