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
Tests for the registration core (``core.register_section_stack``).

A synthetic serial-section stack with known ground-truth poses and genuine
cross-gap continuity is built; the engine must run with zero manual input,
certify the clean interfaces, and recover the poses. A data-marked test exercises
the real sec09-13 graphs end-to-end.
"""

import numpy as np
import pytest

from pandorica.stitch.pipeline import core
from pandorica.stitch.transform.solver import (
    IDENTITY,
    apply_pose,
    invert_pose,
    compose_poses,
)


def _make_stack(n_sections=4, m=10, seed=0):
    """GT poses + section graphs whose cross-gap endpoints truly correspond."""
    rng = np.random.default_rng(seed)
    gt = [dict(IDENTITY)]
    step = {"Angle": 0.5, "Tx": 6.0, "Ty": -3.0, "Scale": 1.0}
    for _ in range(n_sections - 1):
        gt.append(compose_poses(gt[-1], step))

    rows = [[] for _ in range(n_sections)]
    next_id = [0]

    def add_vertical(sec, xy, z0, z1):
        for z in np.linspace(z0, z1, 6):
            rows[sec].append([next_id[0], xy[0], xy[1], z])
        next_id[0] += 1

    for k in range(n_sections - 1):
        glob = rng.uniform(-40, 40, size=(m, 2))  # global crossing points
        loc_k = apply_pose(invert_pose(gt[k]), glob)  # section k local (bottom face)
        loc_k1 = apply_pose(invert_pose(gt[k + 1]), glob)  # section k+1 local (top)
        for xy in loc_k:
            add_vertical(k, xy, 5.0, 10.0)  # high-z half
        for xy in loc_k1:
            add_vertical(k + 1, xy, 0.0, 5.0)  # low-z half

    return gt, [np.array(r, dtype=float) for r in rows]


def test_pipeline_runs_and_certifies_clean_stack():
    gt, coords_list = _make_stack()
    result = core.register_section_stack(coords_list)

    assert len(result.poses) == len(coords_list)
    assert result.poses[0] == IDENTITY
    assert result.accepted
    for iface in result.interfaces:
        assert iface.warp.accepted
        assert iface.qc.accepted


def test_pipeline_recovers_ground_truth_poses():
    gt, coords_list = _make_stack(n_sections=5, m=12, seed=2)
    result = core.register_section_stack(coords_list)
    # Gauge-anchored at section 0; far-section pose should match GT.
    assert result.poses[-1]["Angle"] == pytest.approx(gt[-1]["Angle"], abs=0.2)
    assert result.poses[-1]["Tx"] == pytest.approx(gt[-1]["Tx"], abs=1.0)
    assert result.poses[-1]["Ty"] == pytest.approx(gt[-1]["Ty"], abs=1.0)


def _rotated_two_section(angle_deg=90.0, m=16, seed=0):
    """Two sections whose continuing MTs differ by a known in-plane rotation."""
    rng = np.random.default_rng(seed)
    glob = rng.uniform(-40, 40, size=(m, 2))
    c = glob.mean(0)
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot = (glob - c) @ R.T + c  # section-1 local = section-0 rotated about centroid

    rows0, rows1 = [], []
    for i, (g, r) in enumerate(zip(glob, rot)):
        for z in np.linspace(5.0, 10.0, 6):  # sec0 top face (high-z)
            rows0.append([i, g[0], g[1], z])
        for z in np.linspace(0.0, 5.0, 6):  # sec1 bottom face (low-z)
            rows1.append([i, r[0], r[1], z])
    return [np.array(rows0, float), np.array(rows1, float)]


def test_coarse_angle_seed_recovers_large_rotation():
    # sec1 is rotated +90° from sec0; aligning sec1→sec0 needs −90°.
    coords = _rotated_two_section(angle_deg=90.0)
    res = core.register_section_stack(coords, coarse_angles=[-90.0])
    assert res.interfaces[0].confidence["match_fraction"] > 0.8
    assert res.poses[1]["Angle"] == pytest.approx(-90.0, abs=2.0)


def test_without_coarse_angle_large_rotation_is_unreliable():
    # Documents the failure mode: with no image hint the MT-only pipeline cannot
    # reliably recover a 90° rotation (it is rotationally ambiguous from points).
    coords = _rotated_two_section(angle_deg=90.0)
    res = core.register_section_stack(coords)  # no coarse_angles
    assert res.poses[1]["Angle"] != pytest.approx(-90.0, abs=2.0)


def test_single_section_is_trivial():
    result = core.register_section_stack([np.zeros((4, 4))])
    assert len(result.poses) == 1
    assert result.accepted


@pytest.mark.data
def test_runs_on_real_sections(section_graphs):
    """Zero-manual-input run over the real sec09-13 graphs (smoke + structure)."""
    coords_list = list(section_graphs.values())
    result = core.register_section_stack(coords_list)
    assert len(result.poses) == len(coords_list)
    assert result.poses[0] == IDENTITY
    # Every interface produces a QC record (accepted or flagged with reasons).
    assert len(result.interfaces) == len(coords_list) - 1
    for iface in result.interfaces:
        assert iface.qc.accepted or len(iface.qc.reasons) > 0
