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

"""Tests for the enhanced Hungarian matcher (``matcher.py``)."""

import numpy as np
import pytest

from pandorica.stitch.matching import matcher as mt


def _ep(mt_id, pos, direction=(1.0, 0.0, 0.0)):
    return {
        "id": mt_id,
        "pos": np.array(pos, dtype=float),
        "dir": np.array(direction, dtype=float),
    }


def _grid_endpoints(n=5, step=10.0, direction=(1.0, 0.0, 0.0)):
    eps = []
    k = 0
    for i in range(n):
        for j in range(n):
            eps.append(_ep(k, (i * step, j * step, 0.0), direction))
            k += 1
    return eps


# --------------------------------------------------------------------------- #
# vMF direction cost
# --------------------------------------------------------------------------- #
def test_vmf_cost_extremes():
    assert mt._vmf_direction_cost(
        np.array([1, 0, 0]), np.array([1, 0, 0])
    ) == pytest.approx(0.0)
    assert mt._vmf_direction_cost(
        np.array([1, 0, 0]), np.array([-1, 0, 0])
    ) == pytest.approx(0.0)
    assert mt._vmf_direction_cost(
        np.array([1, 0, 0]), np.array([0, 1, 0])
    ) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Dedupe (W1a)
# --------------------------------------------------------------------------- #
def test_dedupe_removes_near_coincident():
    eps = [_ep(0, (0, 0, 0)), _ep(1, (0.01, 0, 0)), _ep(2, (5, 0, 0))]
    kept = mt.dedupe_endpoints(eps, rho=1.0, dup_frac=0.1)
    assert len(kept) == 2  # the 0.01-apart duplicate is dropped


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def test_matches_known_correspondence_under_small_shift():
    ref = _grid_endpoints()
    rho = 10.0
    # moving = same grid shifted by less than the distance gate
    mov = [_ep(e["id"], e["pos"] + np.array([1.0, 1.0, 0.0]), e["dir"]) for e in ref]
    matches, ref_xy, mov_xy, conf, id_pairs = mt.match_sections(ref, mov, rho)
    assert conf["match_fraction"] == pytest.approx(1.0)
    # every matched pair is the same grid node (ref == mov - shift)
    assert np.allclose(ref_xy, mov_xy - np.array([1.0, 1.0]))
    # id_pairs parallel to matches: same length, same costs
    assert len(id_pairs) == len(matches)


def test_distance_gate_blocks_far_points():
    # Disable the physical clamp so this test exercises pure ρ-scaling.
    rho = 1.0
    ref = [_ep(0, (0, 0, 0))]
    mov = [_ep(0, (100, 0, 0))]  # way beyond max_dist_rho * rho
    matches, ref_xy, _, conf, _ = mt.match_sections(
        ref, mov, rho,
        max_dist_rho=5.0, min_dist_A=0.0, max_dist_A=10.0,
    )
    assert matches == []
    assert conf["n_matches"] == 0


def test_distance_gate_clamps_to_physical_max():
    # ρ-scaling alone would allow a 5000 Å match (5 × ρ), but the physical
    # ceiling caps it at max_dist_A. Pair at 3000 Å should be REJECTED when
    # max_dist_A=2500.
    rho = 1000.0
    ref = [_ep(0, (0, 0, 0))]
    mov = [_ep(0, (3000, 0, 0))]
    matches, *_, _ = mt.match_sections(
        ref, mov, rho,
        max_dist_rho=5.0, min_dist_A=500.0, max_dist_A=2500.0,
    )
    assert matches == []


def test_distance_gate_clamps_to_physical_min():
    # ρ-scaling alone would gate at 50 Å (5 × ρ) but the physical floor
    # (default 500 Å) lifts the gate to 500 Å. Pair at 200 Å should be KEPT.
    rho = 10.0
    ref = [_ep(0, (0, 0, 0))]
    mov = [_ep(0, (200, 0, 0))]
    matches, *_, _ = mt.match_sections(
        ref, mov, rho,
        max_dist_rho=5.0, min_dist_A=500.0, max_dist_A=2500.0,
    )
    assert len(matches) == 1


def test_angle_gate_blocks_misaligned_directions():
    rho = 1.0
    ref = [_ep(0, (0, 0, 0), (1, 0, 0))]
    mov = [_ep(0, (0.5, 0, 0), (0, 1, 0))]  # perpendicular tangent
    matches, *_ = mt.match_sections(ref, mov, rho, max_angle_deg=30.0)
    assert matches == []


# --------------------------------------------------------------------------- #
# W1b outlier rejection
# --------------------------------------------------------------------------- #
def test_outlier_match_is_rejected():
    ref = _grid_endpoints()
    rho = 10.0
    mov = [_ep(e["id"], e["pos"] + np.array([1.0, 1.0, 0.0]), e["dir"]) for e in ref]
    # Corrupt one moving endpoint so its post-fit residual is large but it still
    # falls inside the distance gate of its true partner's neighbourhood.
    mov[12]["pos"] = mov[12]["pos"] + np.array([6.0, -6.0, 0.0])
    matches, ref_xy, mov_xy, conf, _ = mt.match_sections(ref, mov, rho, max_resid_rho=0.5)
    # The corrupted pair should be dropped; the rest survive.
    assert conf["n_matches"] >= len(ref) - 2
    # No surviving pair has a large residual to the consensus shift.
    shifts = ref_xy - mov_xy
    assert np.std(shifts, axis=0).max() < 1.0


# --------------------------------------------------------------------------- #
# orientation (sign) gate: reject fold-back (anti-parallel) matches
# --------------------------------------------------------------------------- #
def test_sign_gate_rejects_fold_back():
    # Anti-parallel in-plane tangents = a fold-back (two different MTs). The
    # |cos| angle gate accepts it (reads as aligned); the orientation gate rejects.
    rho = 1.0
    ref = [_ep(0, (0, 0, 0), (1, 0, 0))]
    mov = [_ep(0, (1, 0, 0), (-1, 0, 0))]      # within distance, anti-parallel
    matches, *_ = mt.match_sections(ref, mov, rho)
    assert matches == []                       # rejected by the orientation gate
    # Disabling the orientation gate (very negative threshold) lets it match again,
    # confirming it was the sign gate (not distance/angle) that dropped it.
    matches2, *_ = mt.match_sections(ref, mov, rho, sign_min_cos=-2.0)
    assert len(matches2) == 1


def test_sign_gate_exempts_near_vertical():
    # Near-vertical stubs have negligible, noisy in-plane tangents, so the sign
    # gate must NOT judge them — they match on position as before.
    rho = 1.0
    ref = [_ep(0, (0, 0, 0), (0.1, 0.0, 0.99))]
    mov = [_ep(0, (1, 0, 0), (-0.1, 0.0, 0.99))]  # in-plane anti-parallel but |d_xy|<0.2
    matches, *_ = mt.match_sections(ref, mov, rho)
    assert len(matches) == 1                   # exempt from the orientation gate


# --------------------------------------------------------------------------- #
# geometry-default uncrosser: resolve parallel rung-swaps
# --------------------------------------------------------------------------- #
def test_uncrosser_fixes_parallel_rung_swap():
    # Two parallel MTs whose Hungarian assignment came out CROSSED. Direction can't
    # tell crossed from uncrossed (both parallel), so the old direction-only rule
    # (margin 0) leaves it; the geometry-default rule uncrosses it.
    ref = [_ep(0, (0, 0, 0), (0, 1, 0)), _ep(1, (20, 0, 0), (0, 1, 0))]
    mov = [_ep(0, (0, 5, 0), (0, 1, 0)), _ep(1, (20, 5, 0), (0, 1, 0))]
    crossed = [(0, 1, 0.0), (1, 0, 0.0)]       # A->Q, B->P  (segments cross)
    rho = 20.0
    kept = mt.uncross_pairs(crossed, ref, mov, rho, dir_veto_margin=0.0)
    assert [(r, c) for r, c, _ in kept] == [(0, 1), (1, 0)]   # old rule: unchanged
    fixed = mt.uncross_pairs(crossed, ref, mov, rho, dir_veto_margin=0.2)
    assert [(r, c) for r, c, _ in fixed] == [(0, 0), (1, 1)]  # geometry: uncrossed


def test_uncrosser_keeps_a_genuine_crossing():
    # When the tangents CLEARLY prefer the crossed pairing (each stub points at its
    # crossed partner), the veto wins and the crossing is preserved.
    ref = [_ep(0, (0, 0, 0), (1, 1, 0)), _ep(1, (20, 0, 0), (-1, 1, 0))]
    mov = [_ep(0, (0, 5, 0), (-1, 1, 0)), _ep(1, (20, 5, 0), (1, 1, 0))]
    crossed = [(0, 1, 0.0), (1, 0, 0.0)]       # A->Q, B->P: tangents continue across
    kept = mt.uncross_pairs(crossed, ref, mov, 20.0, dir_veto_margin=0.2)
    assert [(r, c) for r, c, _ in kept] == [(0, 1), (1, 0)]   # veto: stays crossed


def test_confidence_separates_good_from_incoherent():
    ref = _grid_endpoints()
    rho = 10.0
    good = [_ep(e["id"], e["pos"] + np.array([1.0, 1.0, 0.0]), e["dir"]) for e in ref]
    _, _, _, conf_good, _ = mt.match_sections(ref, good, rho)
    # Incoherent: random per-endpoint shifts inside the gate.
    rng = np.random.default_rng(0)
    bad = [
        _ep(e["id"], e["pos"] + np.r_[rng.uniform(-15, 15, 2), 0.0], e["dir"])
        for e in ref
    ]
    _, _, _, conf_bad, _ = mt.match_sections(ref, bad, rho, max_resid_rho=100.0)
    assert conf_good["shift_incoherence_rho"] < conf_bad["shift_incoherence_rho"]
