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

"""Tests for the public stitch entry points: the registration core
(``core.register_section_stack``) and the full pipeline (``stitcher.stitch_sections``).
"""

import numpy as np

from pandorica.stitch.pipeline.core import register_section_stack
from pandorica.stitch.pipeline.stitcher import stitch_sections


def _R(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _stack(rel_angles, m=30, seed=0, jitter=0.0):
    """Serial sections whose continuing MTs rotate by the given relative angles."""
    rng = np.random.default_rng(seed)
    G = rng.uniform(-50, 50, size=(m, 2))
    cum = np.cumsum([0.0] + list(rel_angles))
    coords = []
    for a in cum:
        loc = G @ _R(-a).T
        if jitter:
            loc = loc + rng.normal(0, jitter, loc.shape)
        rows = []
        for nid, xy in enumerate(loc):
            for z in np.linspace(0.0, 10.0, 6):
                rows.append([nid, xy[0], xy[1], z])
        coords.append(np.array(rows, float))
    return cum, coords


def test_core_runs_and_returns_poses():
    cum, coords = _stack([3.0, -2.0, 1.0], seed=1)
    res = register_section_stack(coords)
    assert len(res.poses) == len(coords)
    assert res.poses[0]["Angle"] == 0.0  # gauge anchor
    assert isinstance(res.accepted, bool)


def test_stitch_sections_cpd_default():
    cum, coords = _stack([2.0, 90.0, -2.0], seed=2)
    res = stitch_sections(coords)  # cpd_coarse defaults True
    far = abs(((res.poses[-1]["Angle"] - cum[-1]) + 180) % 360 - 180)
    assert far < 5.0  # recovers the +90° interface via CPD coarse
    assert len(res.intensity) == len(coords) - 1


def test_stitch_sections_gated_path_still_available():
    cum, coords = _stack([3.0, -2.0, 1.0], seed=3)
    res = stitch_sections(coords, cpd_coarse=False)
    assert len(res.poses) == len(coords)


def test_qc_threshold_param_threads_through():
    # An impossible match-fraction floor must flip acceptance to False — proving
    # the flat QC knob actually reaches qc.assess_interface.
    _, coords = _stack([3.0, -2.0, 1.0], seed=4)
    ok = register_section_stack(coords, qc_min_match_fraction=0.3)
    strict = register_section_stack(coords, qc_min_match_fraction=1.01)
    assert ok.accepted and not strict.accepted


def test_matcher_param_threads_through():
    # A near-zero distance gate starves the matcher → interfaces fail to certify.
    # Needs jitter: on noise-free data the PCA coarse aligns clouds to < 1e-6·ρ and
    # would match even at a tiny gate, masking the param's effect.
    # Disable the physical floor (``min_dist_A=0``) so the starved gate isn't
    # rescued by the absolute-units clamp the matcher applies by default.
    _, coords = _stack([3.0, -2.0, 1.0], seed=5, jitter=2.0)
    ok = register_section_stack(coords, max_dist_rho=8.0)
    starved = register_section_stack(
        coords, max_dist_rho=1e-6, min_dist_A=0.0, max_dist_A=0.0
    )
    assert ok.accepted and not starved.accepted
