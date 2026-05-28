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

"""Tests for the full stitch pipeline (``stitcher.stitch_sections``)."""

import numpy as np
import pytest

from pandorica.stitch.pipeline import stitcher
from pandorica.stitch.transform.solver import (
    IDENTITY,
    apply_pose,
    invert_pose,
    compose_poses,
)


def _make_stack(n_sections=4, m=12, seed=0):
    """Serial sections with genuine cross-gap continuity (small known rotations)."""
    rng = np.random.default_rng(seed)
    gt = [dict(IDENTITY)]
    step = {"Angle": 0.5, "Tx": 6.0, "Ty": -3.0, "Scale": 1.0}
    for _ in range(n_sections - 1):
        gt.append(compose_poses(gt[-1], step))
    rows = [[] for _ in range(n_sections)]
    nid = [0]

    def add(sec, xy, z0, z1):
        for z in np.linspace(z0, z1, 6):
            rows[sec].append([nid[0], xy[0], xy[1], z])
        nid[0] += 1

    for k in range(n_sections - 1):
        glob = rng.uniform(-40, 40, size=(m, 2))
        for xy in apply_pose(invert_pose(gt[k]), glob):
            add(k, xy, 5.0, 10.0)
        for xy in apply_pose(invert_pose(gt[k + 1]), glob):
            add(k + 1, xy, 0.0, 5.0)
    return gt, [np.array(r, float) for r in rows]


def test_candidate_c_runs_and_certifies_clean_stack():
    gt, coords = _make_stack()
    res = stitcher.stitch_sections(coords)
    assert len(res.poses) == len(coords)
    assert res.poses[0] == IDENTITY
    assert res.accepted  # no images → intensity not gated; base accepts
    # hybrid coarse produced one record per interface
    assert len(res.hybrid.records) == len(coords) - 1
    assert res.intensity == [None] * (len(coords) - 1)


def test_candidate_c_recovers_poses():
    gt, coords = _make_stack(n_sections=5, m=14, seed=2)
    res = stitcher.stitch_sections(coords)
    assert res.poses[-1]["Angle"] == pytest.approx(gt[-1]["Angle"], abs=1.5)


def test_single_section_trivial():
    res = stitcher.stitch_sections([np.zeros((4, 4))])
    assert len(res.poses) == 1
