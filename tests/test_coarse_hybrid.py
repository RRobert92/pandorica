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

"""Tests for the hybrid coarse orchestrator + continuity (``coarse_hybrid.py``)."""

import numpy as np
import pytest

from pandorica.stitch.coarse import coarse_hybrid as ch
from pandorica.stitch.coarse.coarse_fusion import ResolvedRotation
from pandorica.stitch.coarse.rotation_search import RotationEstimate


def _est(angle, flip_ratio=3.0):
    return RotationEstimate(angle, 0.8, 2.0, flip_ratio, 2.0, 0.3, 50, 50)


# --------------------------------------------------------------------------- #
# Continuity pass (pure logic)
# --------------------------------------------------------------------------- #
def test_continuity_resolves_small_flagged_toward_trend():
    # Confident neighbours ~0–2°; a flagged interface at angle 2° (flip = 182°)
    # → continuity picks 2°, not the 180°-off branch.
    recs = [
        ResolvedRotation(1.0, "confident", False, _est(1.0)),
        ResolvedRotation(2.0, "abstain", True, _est(2.0, flip_ratio=1.05)),
        ResolvedRotation(0.0, "confident", False, _est(0.0)),
    ]
    ch._continuity_resolve(recs, tol=30.0)
    assert recs[1].source == "continuity" and not recs[1].flagged
    assert recs[1].angle == pytest.approx(2.0)


def test_continuity_leaves_ambiguous_large_flagged():
    # Both branches of a 90° interface (90 and −90) are far from a ~0 trend →
    # stays flagged (we don't guess a genuine large rotation's sign).
    recs = [
        ResolvedRotation(0.0, "confident", False, _est(0.0)),
        ResolvedRotation(90.0, "abstain", True, _est(90.0, flip_ratio=1.05)),
        ResolvedRotation(1.0, "confident", False, _est(1.0)),
    ]
    ch._continuity_resolve(recs, tol=30.0)
    assert recs[1].flagged and recs[1].source == "abstain"


# --------------------------------------------------------------------------- #
# End-to-end on a synthetic stack (no images → MT-only + continuity)
# --------------------------------------------------------------------------- #
def _R(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _make_rot_stack(rel_angles, m=28, seed=0):
    """Sections whose continuing MTs are an asymmetric set rotated by known rels."""
    rng = np.random.default_rng(seed)
    G = rng.uniform(-50, 50, size=(m, 2))  # asymmetric global crossing points
    cum = np.cumsum([0.0] + list(rel_angles))  # absolute angle per section
    coords = []
    for k, a in enumerate(cum):
        loc = G @ _R(-a).T  # section-k-local positions
        rows = []
        for i, xy in enumerate(loc):
            for z in np.linspace(0.0, 10.0, 6):  # vertical MT → top & bottom faces
                rows.append([i, xy[0], xy[1], z])
        coords.append(np.array(rows, float))
    return coords


def test_hybrid_recovers_relative_rotations_no_images():
    rels = [3.0, 90.0, -2.0]
    coords = _make_rot_stack(rels)
    res = ch.hybrid_coarse(coords)
    assert len(res.angles) == 3
    for got, want in zip(res.angles, rels):
        err = ((got - want) + 180) % 360 - 180
        # coarse-level tolerance (downstream fine rigid fit refines to sub-degree)
        assert abs(err) <= 6.0, (got, want)
    # asymmetric constellation → all confident, none flagged
    assert all(not r.flagged for r in res.records)
