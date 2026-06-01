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

"""Tests for chain.chain_filaments and the matcher's uncrosser / smoothness."""

import numpy as np
import pytest

from pandorica.stitch.matching import matcher as mt
from pandorica.stitch.chain import (
    chain_filaments,
    compute_chain_labels,
    orient_chain_blocks,
    split_chains_at_joints,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ep(mt_id, pos, direction=(1.0, 0.0, 0.0)):
    return {
        "id": int(mt_id),
        "pos": np.array(pos, dtype=float),
        "dir": np.array(direction, dtype=float),
    }


# --------------------------------------------------------------------------- #
# Uncrosser
# --------------------------------------------------------------------------- #
def test_uncrosser_swaps_rung_swap():
    # Two MT stubs close together. Ground truth pairing is parallel (a->a', b->b'),
    # but a permutation that crosses (a->b', b->a') is feasible. The pairing has
    # to be disambiguable by *direction continuity* — sign-agnostic vmf can't
    # tell ±X apart, so use orthogonal tangents:
    #
    #     ref: a=(0, 0) dir +X,    b=(0, 2) dir +Y
    #     mov: a'=(1, 0) dir +X,   b'=(1, 2) dir +Y
    #
    # Parallel (a->a', b->b') has perfect direction continuity (cost 0+0).
    # Crossed (a->b', b->a') pairs a +X tangent with a +Y one → cost 1+1.
    ref = [_ep(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
           _ep(1, (0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]
    mov = [_ep(10, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
           _ep(11, (1.0, 2.0, 0.0), (0.0, 1.0, 0.0))]
    # Start from the X-crossed assignment: (0->11), (1->10).
    crossed = [(0, 1, 0.5), (1, 0, 0.5)]
    out = mt.uncross_pairs(crossed, ref, mov, rho=1.0, neighbour_rho=5.0)
    # After uncrossing we expect the parallel pairing.
    pairing = {r: c for r, c, _ in out}
    assert pairing[0] == 0
    assert pairing[1] == 1


def test_uncrosser_leaves_good_pairing_alone():
    ref = [_ep(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
           _ep(1, (0.0, 2.0, 0.0), (0.0, 1.0, 0.0))]
    mov = [_ep(10, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
           _ep(11, (1.0, 2.0, 0.0), (0.0, 1.0, 0.0))]
    parallel = [(0, 0, 0.5), (1, 1, 0.5)]
    out = mt.uncross_pairs(parallel, ref, mov, rho=1.0)
    assert {(r, c) for r, c, _ in out} == {(0, 0), (1, 1)}


# --------------------------------------------------------------------------- #
# Per-pair smoothness filter
# --------------------------------------------------------------------------- #
def test_smoothness_drops_kinked_pair():
    # Two pairs: one with aligned in-plane tangents (kept), one with
    # perpendicular ones (dropped).
    ref = [
        _ep(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        _ep(1, (10.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ]
    mov = [
        _ep(10, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        _ep(11, (10.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # 90° kink
    ]
    matches = [(0, 0, 0.1), (1, 1, 0.1)]
    out = mt.filter_pair_smoothness(matches, ref, mov, max_tangent_deg=30.0)
    survivors = {(r, c) for r, c, _ in out}
    assert (0, 0) in survivors
    assert (1, 1) not in survivors


def test_smoothness_drops_sideways_offset_pair():
    # Two stubs with parallel tangents but the chord between them is
    # perpendicular to those tangents — the "sideways jump" failure mode.
    # tangent-tangent agreement alone would (wrongly) accept it.
    ref = [_ep(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))]
    mov = [_ep(10, (0.0, 5.0, 0.0), (1.0, 0.0, 0.0))]
    matches = [(0, 0, 0.1)]
    out = mt.filter_pair_smoothness(matches, ref, mov)
    assert out == []  # chord (+Y) perpendicular to tangents (+X) → rejected


def test_smoothness_keeps_gently_curved_pair():
    # Two stubs whose tangents differ by ~15° and whose chord differs from
    # each tangent by ~10°. A real curved MT crossing a cut.
    import math
    ang_a = math.radians(0.0)
    ang_b = math.radians(15.0)
    ref = [_ep(0, (0.0, 0.0, 0.0), (math.cos(ang_a), math.sin(ang_a), 0.0))]
    mov = [_ep(10, (3.0, 0.4, 0.0), (math.cos(ang_b), math.sin(ang_b), 0.0))]
    matches = [(0, 0, 0.1)]
    out = mt.filter_pair_smoothness(matches, ref, mov)
    assert out == matches  # curvature within budget → kept


def test_smoothness_skips_near_vertical_pair():
    # Near-vertical means tiny XY-tangent magnitude → the smoothness verdict
    # is uninformative. The filter must NOT drop such pairs (Hungarian's
    # cost gate is the only available judge in that regime).
    ref = [_ep(0, (0.0, 0.0, 0.0), (0.01, 0.0, 1.0))]  # almost pure Z
    mov = [_ep(10, (0.5, 0.0, 0.0), (0.0, 0.01, 1.0))]  # almost pure Z, different XY
    matches = [(0, 0, 0.1)]
    out = mt.filter_pair_smoothness(matches, ref, mov, max_tangent_deg=30.0)
    assert out == matches  # near-vertical → kept


# --------------------------------------------------------------------------- #
# Chain builder
# --------------------------------------------------------------------------- #
def test_chain_linear_three_sections():
    # 3 sections, one MT continuing through all of them as:
    #   section 0 id 5  ↔  section 1 id 7  ↔  section 2 id 11
    sections = [[5], [7], [11]]
    pairs = [[(5, 7, 0.1)], [(7, 11, 0.1)]]
    accepted = [True, True]
    id_map, n = chain_filaments(sections, pairs, accepted)
    assert n == 1
    assert id_map[(0, 5)] == id_map[(1, 7)] == id_map[(2, 11)]


def test_chain_breaks_at_flagged_interface():
    # Same chain as above but interface 0↔1 is flagged. Section 0 must remain
    # disconnected from sections 1-2 (which still chain across iface 1↔2).
    sections = [[5], [7], [11]]
    pairs = [[(5, 7, 0.1)], [(7, 11, 0.1)]]
    accepted = [False, True]
    id_map, n = chain_filaments(sections, pairs, accepted)
    assert n == 2
    assert id_map[(0, 5)] != id_map[(1, 7)]
    assert id_map[(1, 7)] == id_map[(2, 11)]


def test_chain_lonely_mts_get_own_ids():
    # Section 0 has one un-matched MT. It must still appear in id_map with
    # its own unique global id.
    sections = [[5, 6], [7]]
    pairs = [[(5, 7, 0.1)]]  # MT 6 has no partner
    accepted = [True]
    id_map, n = chain_filaments(sections, pairs, accepted)
    assert n == 2
    assert id_map[(0, 5)] == id_map[(1, 7)]
    assert id_map[(0, 6)] != id_map[(0, 5)]


def test_chain_contiguous_global_ids():
    # Global IDs must be a contiguous 0..n_filaments-1 range.
    sections = [[1, 2, 3], [4, 5], [6]]
    pairs = [[(1, 4, 0.1), (3, 5, 0.1)], [(4, 6, 0.1)]]
    accepted = [True, True]
    id_map, n = chain_filaments(sections, pairs, accepted)
    assert sorted(set(id_map.values())) == list(range(n))


def test_chain_empty_section_list():
    id_map, n = chain_filaments([], [], [])
    assert id_map == {}
    assert n == 0


def test_chain_input_length_mismatch_raises():
    with pytest.raises(ValueError):
        chain_filaments([[1], [2]], [], [True])


# --------------------------------------------------------------------------- #
# End-to-end: matcher returns id_pairs that the chain can consume
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# orient_chain_blocks (zigzag prevention)
# --------------------------------------------------------------------------- #
def _coords(rows):
    """Helper: rows = [(gid, x, y, z), ...] → np.float64 array."""
    return np.array(rows, dtype=float)


def test_orient_preserves_trace_order_for_bendy_mt():
    # Single MT in one section, NON-monotonic Z. Trace order must survive
    # untouched — sorting by Z (the old behaviour) would scramble these.
    rows = [
        (0, 0.0, 0.0, 10.0),
        (0, 1.0, 0.0, 11.0),
        (0, 2.0, 0.0, 9.0),   # MT bends back in Z
        (0, 3.0, 0.0, 12.0),
        (0, 4.0, 0.0, 13.0),
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 0, 0, 0])
    out = orient_chain_blocks(coords, sections)
    assert np.allclose(out, coords)  # exactly the input — no reorder


def test_orient_reverses_sub_block_when_endpoint_far():
    # gid 0 spans 2 sections. Section 0 trace ends at z=10 (high). Section 1
    # trace is stored in REVERSE: it starts at z=20 and ends at z=11.
    # The join should reverse section 1 so its z=11 endpoint connects to
    # section 0's z=10 tail.
    rows = [
        (0, 0.0, 0.0, 8.0),
        (0, 0.0, 0.0, 9.0),
        (0, 0.0, 0.0, 10.0),    # tail of section 0
        (0, 0.0, 0.0, 20.0),    # head of section 1 (far from tail)
        (0, 0.0, 0.0, 15.0),
        (0, 0.0, 0.0, 11.0),    # tail of section 1 (close to section-0 tail)
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 0, 1, 1, 1])
    out = orient_chain_blocks(coords, sections)
    # First 3 rows unchanged (section 0 single sub still 'tail closest').
    assert np.allclose(out[:3, 3], [8.0, 9.0, 10.0])
    # Section 1 must be reversed: 11, 15, 20.
    assert np.allclose(out[3:, 3], [11.0, 15.0, 20.0])


def test_orient_chooses_first_block_orientation_against_second():
    # Section 0's MT is stored in trace order with z descending (10 -> 8),
    # but section 1's nearer endpoint is at z=11. The first block should be
    # reversed so its tail (z=10) connects to section 1's z=11 head.
    rows = [
        (0, 0.0, 0.0, 10.0),    # head of section 0 (close to section 1's head)
        (0, 0.0, 0.0, 9.0),
        (0, 0.0, 0.0, 8.0),     # tail of section 0 (far from section 1)
        (0, 0.0, 0.0, 11.0),    # head of section 1
        (0, 0.0, 0.0, 15.0),
        (0, 0.0, 0.0, 20.0),
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 0, 1, 1, 1])
    out = orient_chain_blocks(coords, sections)
    # Section 0 must be reversed: 8, 9, 10.
    assert np.allclose(out[:3, 3], [8.0, 9.0, 10.0])
    # Section 1 kept as-is: 11, 15, 20.
    assert np.allclose(out[3:, 3], [11.0, 15.0, 20.0])


def test_orient_three_sections_running_tail():
    # Three sections, each in trace order natural for joining.
    # Verify the running-tail logic keeps the chain monotonic.
    rows = [
        (0, 0.0, 0.0, 0.0),
        (0, 0.0, 0.0, 1.0),
        (0, 0.0, 0.0, 2.0),     # tail of section 0
        (0, 0.0, 0.0, 3.0),
        (0, 0.0, 0.0, 4.0),     # tail of section 1
        (0, 0.0, 0.0, 5.0),
        (0, 0.0, 0.0, 6.0),     # tail of section 2
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 0, 1, 1, 2, 2])
    out = orient_chain_blocks(coords, sections)
    # No reversals needed: the chain is already continuous.
    assert np.allclose(out[:, 3], np.arange(7, dtype=float))


def test_orient_multiple_chains_independent():
    # Two chains: gid 0 spans sections 0 and 1, gid 1 lives in section 0 only.
    rows = [
        (0, 0.0, 0.0, 0.0),     # chain 0, section 0, head
        (0, 0.0, 0.0, 5.0),     # chain 0, section 0, tail
        (1, 10.0, 0.0, 0.0),    # chain 1, section 0, head
        (1, 10.0, 0.0, 100.0),  # chain 1, section 0, tail (a bendy MT!)
        (1, 10.0, 0.0, 50.0),
        (0, 0.0, 0.0, 6.0),     # chain 0, section 1, head
        (0, 0.0, 0.0, 9.0),     # chain 0, section 1, tail
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 0, 0, 0, 1, 1])
    # Caller sort-by-gid would produce:
    gid_order = np.argsort(coords[:, 0].astype(int), kind="stable")
    coords_s = coords[gid_order]
    sections_s = sections[gid_order]
    out = orient_chain_blocks(coords_s, sections_s)
    # gid 0 (4 rows): preserved trace order 0,5,6,9.
    assert np.allclose(out[:4, 3], [0.0, 5.0, 6.0, 9.0])
    # gid 1 (3 rows): single section, bendy MT — trace order untouched.
    assert np.allclose(out[4:, 3], [0.0, 100.0, 50.0])


# --------------------------------------------------------------------------- #
# split_chains_at_joints (post-chain overall-direction splitter)
# --------------------------------------------------------------------------- #
def test_split_keeps_chain_with_lateral_shift():
    # A single MT cut at the section boundary and shifted +Y by 5 units.
    # LOCAL bend at the joint is ~90°, but the overall direction of each
    # half is +X — the chain must SURVIVE.
    rows = [
        (0, 0.0, 0.0, 0.0),
        (0, 5.0, 0.0, 0.0),
        (0, 10.0, 0.0, 0.0),    # tail of section 0, pointing +X
        (0, 10.0, 5.0, 0.0),    # head of section 1 — lateral jump +Y
        (0, 15.0, 5.0, 0.0),
        (0, 20.0, 5.0, 0.0),    # tail of section 1, pointing +X
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 0, 1, 1, 1])
    out = split_chains_at_joints(coords, sections, max_angle_deg=45.0)
    # All rows stay the same gid — overall direction agrees (+X on both sides).
    assert np.all(out[:, 0] == 0)


def test_split_breaks_chain_with_perpendicular_overall_direction():
    # Two unrelated MTs that happened to be joined: section 0 points +X,
    # section 1 points +Y. Overall directions disagree by 90° → SPLIT.
    rows = [
        (0, 0.0, 0.0, 0.0),
        (0, 10.0, 0.0, 0.0),    # tail of section 0, overall +X
        (0, 10.0, 0.0, 1.0),
        (0, 10.0, 10.0, 1.0),   # section 1, overall +Y
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 1, 1])
    out = split_chains_at_joints(coords, sections, max_angle_deg=45.0)
    # gid changes at the joint: two different filaments now.
    assert out[0, 0] == out[1, 0]      # section 0 keeps gid 0
    assert out[2, 0] == out[3, 0]      # section 1 has its own gid
    assert out[0, 0] != out[2, 0]


def test_split_handles_three_section_chain_with_one_bad_joint():
    # Three sections, all pointing roughly +X, except section 1 points +Y.
    # The two bad joints (0↔1 and 1↔2) BOTH split → 3 filaments.
    rows = [
        (0, 0.0, 0.0, 0.0),     # section 0, +X
        (0, 5.0, 0.0, 0.0),
        (0, 0.0, 5.0, 1.0),     # section 1, +Y
        (0, 0.0, 10.0, 1.0),
        (0, 10.0, 0.0, 2.0),    # section 2, +X
        (0, 15.0, 0.0, 2.0),
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 1, 1, 2, 2])
    out = split_chains_at_joints(coords, sections, max_angle_deg=45.0)
    # Three distinct gids: original 0 keeps first sub-block, two new ids.
    gids = out[:, 0]
    assert len({gids[0], gids[2], gids[4]}) == 3


def test_split_skips_short_sub_blocks():
    # A 1-point sub-block can't give a direction — joint is skipped, chain stays.
    rows = [
        (0, 0.0, 0.0, 0.0),
        (0, 1.0, 0.0, 0.0),
        (0, 0.0, 100.0, 1.0),   # only 1 point in section 1
        (0, 100.0, 0.0, 2.0),   # only 1 point in section 2
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 1, 2])
    out = split_chains_at_joints(coords, sections, max_angle_deg=45.0)
    assert np.all(out[:, 0] == 0)  # nothing to split — direction undefined


def test_split_independent_chains():
    # Two chains side by side. One should split, the other should survive.
    rows = [
        # Chain 0: shifts laterally but keeps direction → keep.
        (0, 0.0, 0.0, 0.0),
        (0, 10.0, 0.0, 0.0),
        (0, 10.0, 5.0, 1.0),
        (0, 20.0, 5.0, 1.0),
        # Chain 1: 90° turn at the joint → split.
        (1, 100.0, 0.0, 0.0),
        (1, 110.0, 0.0, 0.0),
        (1, 110.0, 0.0, 1.0),
        (1, 110.0, 10.0, 1.0),
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    out = split_chains_at_joints(coords, sections, max_angle_deg=45.0)
    # Chain 0 untouched (lateral shift, same overall direction).
    assert out[0, 0] == out[3, 0]
    # Chain 1 split at joint.
    assert out[4, 0] != out[6, 0]


# --------------------------------------------------------------------------- #
# compute_chain_labels (diagnostic output)
# --------------------------------------------------------------------------- #
def test_labels_single_section_mt_is_clean():
    rows = [
        (0, 0.0, 0.0, 0.0),
        (0, 1.0, 0.0, 0.0),
        (0, 2.0, 0.0, 0.0),
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 0])
    L = compute_chain_labels(coords, sections)
    assert L["edge_chain_length"].tolist() == [1]
    assert L["edge_n_joints"].tolist() == [0]
    assert L["edge_max_joint_angle_deg"].tolist() == [0.0]
    assert L["point_at_joint"].tolist() == [0, 0, 0]


def test_labels_two_section_straight_chain_low_angle():
    # An almost-straight chain crossing one cut → small joint angle.
    rows = [
        (0, 0.0, 0.0, 0.0),
        (0, 10.0, 0.0, 0.0),    # section 0 direction +X
        (0, 11.0, 0.0, 1.0),
        (0, 21.0, 0.0, 1.0),    # section 1 direction +X
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 1, 1])
    L = compute_chain_labels(coords, sections)
    assert L["edge_chain_length"].tolist() == [2]
    assert L["edge_n_joints"].tolist() == [1]
    assert L["edge_max_joint_angle_deg"][0] < 1.0  # essentially straight
    # Both rows on each side of the joint get marked.
    assert L["point_at_joint"].tolist() == [0, 1, 1, 0]


def test_labels_two_section_perpendicular_chain_high_angle():
    rows = [
        (0, 0.0, 0.0, 0.0),
        (0, 10.0, 0.0, 0.0),    # section 0 direction +X
        (0, 10.0, 0.0, 1.0),
        (0, 10.0, 10.0, 1.0),   # section 1 direction +Y
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 1, 1])
    L = compute_chain_labels(coords, sections)
    assert L["edge_chain_length"].tolist() == [2]
    assert L["edge_max_joint_angle_deg"][0] == pytest.approx(90.0, abs=1.0)
    # Joint points report the angle for both rows.
    assert L["point_joint_angle_deg"][1] == pytest.approx(90.0, abs=1.0)
    assert L["point_joint_angle_deg"][2] == pytest.approx(90.0, abs=1.0)


def test_labels_local_vs_overall_distinguish():
    # A chain where the OVERALL direction agrees (+X on both sides) but the
    # LOCAL geometry at the joint makes a sharp +Y excursion at the end of
    # section 0 and a sharp +Y entry at section 1 — visually it's a sharp
    # triangle at the joint. Local should be ~90°, overall should be ~0°.
    rows = [
        (0,  0.0, 0.0, 0.0),
        (0,  5.0, 0.0, 0.0),
        (0,  9.0, 0.0, 0.0),
        (0,  10.0, 2.0, 0.0),    # last pre-joint point: kicks +Y
        (0,  10.0, 5.0, 1.0),    # first post-joint point: starts at +Y offset
        (0,  11.0, 5.0, 1.0),
        (0,  15.0, 5.0, 1.0),
        (0,  20.0, 5.0, 1.0),
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    L = compute_chain_labels(coords, sections, local_window=2)
    # Overall direction +X on both sides (small +Y mixed in) → small overall angle.
    assert L["edge_max_joint_overall_deg"][0] < 30.0
    # Local: last 2 of pre-block = (9,0)→(10,2) direction (1,2), first 2 of
    # post = (10,5)→(11,5) direction (1,0). Angle between ≈ 63°. Local
    # captures the visual sharpness.
    assert L["edge_max_joint_angle_deg"][0] > 45.0


def test_labels_was_split_marks_split_descendants():
    # Two filaments after a split; gid 0 was original (pre-split), gid 7 is fresh.
    rows = [
        (0, 0.0, 0.0, 0.0),
        (0, 1.0, 0.0, 0.0),
        (7, 5.0, 5.0, 1.0),
        (7, 6.0, 5.0, 1.0),
    ]
    coords = _coords(rows)
    sections = np.array([0, 0, 1, 1])
    pre_split = np.array([0, 0, 0, 0])  # both used to be gid 0; split made gid 7
    L = compute_chain_labels(coords, sections, pre_split)
    assert L["edge_was_split"].tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# End-to-end: matcher returns id_pairs that the chain can consume
# --------------------------------------------------------------------------- #
def test_matcher_id_pairs_feed_into_chain():
    # Two sections, three MTs each, identity correspondence with small shift.
    ref = [_ep(i, (i * 10.0, 0.0, 0.0)) for i in (101, 102, 103)]
    mov = [_ep(i, (i_x * 10.0 + 0.5, 0.0, 0.0))
           for i, i_x in zip((201, 202, 203), (101, 102, 103))]
    # Match section 0 → section 1.
    matches, _, _, conf, id_pairs = mt.match_sections(ref, mov, rho=10.0)
    assert conf["match_fraction"] > 0
    assert len(id_pairs) == len(matches)

    # Feed into chain builder.
    sections = [[101, 102, 103], [201, 202, 203]]
    id_map, n = chain_filaments(sections, [id_pairs], [True])
    # Three MTs continuing across one interface → exactly three filaments.
    assert n == 3
    # And the ref/mov of every pair share the same global id.
    for r_id, m_id, _ in id_pairs:
        assert id_map[(0, r_id)] == id_map[(1, m_id)]
