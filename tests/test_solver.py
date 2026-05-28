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
