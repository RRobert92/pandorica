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

"""Tests for coarse-rotation sign fusion + ABSTAIN (``coarse_fusion.py``)."""

import pytest

from pandorica.stitch.coarse.coarse_fusion import (
    resolve_interface,
    resolve_stack_rotations,
)
from pandorica.stitch.coarse.rotation_search import RotationEstimate


def _est(angle, flip_ratio=3.0, aniso=2.0, ang_u=0.3, mf=0.8):
    """A RotationEstimate; defaults are non-degenerate (confident)."""
    return RotationEstimate(
        angle=angle,
        match_fraction=mf,
        peak_margin=2.0,
        flip_ratio=flip_ratio,
        anisotropy=aniso,
        angular_uniformity=ang_u,
        n_ref=50,
        n_mov=50,
    )


def _ambiguous(angle):
    """An ambiguous estimate (180° flip-ratio too low → not confident)."""
    return _est(angle, flip_ratio=1.05)


# --------------------------------------------------------------------------- #
def test_confident_estimate_accepted_as_is():
    r = resolve_interface(_est(-90.0))
    assert r.source == "confident" and not r.flagged
    assert r.angle == pytest.approx(-90.0)


def test_ambiguous_resolved_by_ap_polarity_to_correct_branch():
    # MT magnitude 90° but sign ambiguous; A–P hint says ~−88° → pick −90 branch.
    r = resolve_interface(_ambiguous(90.0), ap_angle=-88.0)
    assert r.source == "ap_polarity" and not r.flagged
    assert r.angle == pytest.approx(-90.0, abs=1e-6)  # flipped to −90


def test_ambiguous_ap_confirms_unflipped_branch():
    r = resolve_interface(_ambiguous(90.0), ap_angle=92.0)
    assert r.source == "ap_polarity"
    assert r.angle == pytest.approx(90.0)


def test_ambiguous_without_ap_abstains():
    r = resolve_interface(_ambiguous(90.0), ap_angle=None)
    assert r.source == "abstain" and r.flagged


def test_ambiguous_ap_disagrees_with_both_branches_abstains():
    # A–P hint near 0° agrees with neither 90° nor −90° within tolerance.
    r = resolve_interface(_ambiguous(90.0), ap_angle=2.0, ap_tol_deg=45.0)
    assert r.source == "abstain" and r.flagged


def test_stack_resolution_mixed():
    ests = [_est(2.0), _ambiguous(90.0), _ambiguous(40.0), _est(1.0)]
    aps = [None, -89.0, None, None]
    out = resolve_stack_rotations(ests, ap_angles=aps)
    assert [o.source for o in out] == [
        "confident",
        "ap_polarity",
        "abstain",
        "confident",
    ]
    assert out[1].angle == pytest.approx(-90.0)
    assert out[2].flagged


def test_stack_length_mismatch_raises():
    with pytest.raises(ValueError):
        resolve_stack_rotations([_est(1.0)], ap_angles=[1.0, 2.0])
