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
Tests for ``core.register_warps_to_coarse`` — the *fine* half of the coarse→fine
pipeline (project_coarse_fine_architecture).

The contract under test, which encodes the architecture decision:

  * the supplied image-coarse pose chain passes through **unchanged** — the global
    pose is the image coarse, never re-fit from MT correspondences;
  * with a perfectly coarse-aligned stack the MT residual warp is ~zero;
  * a planted **spatially-varying** residual is absorbed by the **warp**, while the
    poses stay exactly the coarse poses (the deformation does NOT leak into a global
    transform — this is the whole reason MT-global-affine was reverted);
  * an MT-free interface is **pose-accepted** with no warp (image-fill covers it),
    not failed.
"""

import numpy as np
import pytest

from pandorica.stitch.pipeline import core
from pandorica.stitch.pipeline.stitcher import stitch_sections
from pandorica.stitch.transform.solver import (
    IDENTITY,
    apply_pose,
    invert_pose,
    compose_poses,
    make_pose,
)

_STEP = {"Angle": 0.5, "Tx": 6.0, "Ty": -3.0, "Scale": 1.0}


def _coarse_chain(n_sections):
    gt = [dict(IDENTITY)]
    for _ in range(n_sections - 1):
        gt.append(compose_poses(gt[-1], _STEP))
    return gt


def _stack(gt, m=14, seed=0, residual=None, spread=(40.0, 40.0)):
    """Section graphs whose cross-gap endpoints correspond under ``gt``.

    ``residual(glob)->[M,2]`` (optional) is a smooth spatially-varying field added
    to the **moving** crossing points before they are mapped into the next
    section's local frame — so after the coarse pose aligns the gap, that field is
    exactly the leftover the fine warp must capture.

    ``spread`` is the half-extent of the crossing-point cloud per axis; a wide
    (asymmetric) cloud makes a *wrong* rotation genuinely collapse the cross-gap
    match (a square cloud is ~symmetric under 90° and self-matches spuriously).
    """
    rng = np.random.default_rng(seed)
    n = len(gt)
    rows = [[] for _ in range(n)]
    next_id = [0]

    def add_vertical(sec, xy, z0, z1):
        for z in np.linspace(z0, z1, 6):
            rows[sec].append([next_id[0], xy[0], xy[1], z])
        next_id[0] += 1

    for k in range(n - 1):
        glob = rng.uniform([-spread[0], -spread[1]], [spread[0], spread[1]], size=(m, 2))
        glob_mov = glob if residual is None else glob + residual(glob)
        loc_k = apply_pose(invert_pose(gt[k]), glob)          # ref: sec k top (high-z)
        loc_k1 = apply_pose(invert_pose(gt[k + 1]), glob_mov)  # mov: sec k+1 bot (low-z)
        for xy in loc_k:
            add_vertical(k, xy, 5.0, 10.0)
        for xy in loc_k1:
            add_vertical(k + 1, xy, 0.0, 5.0)
    return [np.array(r, dtype=float) for r in rows]


def _poses_match(got, want):
    for p, g in zip(got, want):
        for key in ("Angle", "Tx", "Ty", "Scale"):
            assert p[key] == pytest.approx(g[key], abs=1e-9), key


def _ref_xy(coords_k):
    return core._xy(core._face(coords_k, "top", 0.15))


# --------------------------------------------------------------------------- #
# passthrough + zero residual on a perfectly coarse-aligned stack
# --------------------------------------------------------------------------- #
def test_passthrough_and_zero_residual_on_clean_stack():
    gt = _coarse_chain(4)
    coords = _stack(gt, m=16, seed=1)
    res = core.register_warps_to_coarse(coords, gt)

    assert len(res.poses) == len(gt)
    _poses_match(res.poses, gt)            # the coarse pose chain is untouched
    assert res.accepted
    for k, iface in enumerate(res.interfaces):
        assert iface.warp.accepted
        # nothing left for the warp once the coarse pose aligns the gap
        disp = iface.warp.displacement(_ref_xy(coords[k]))
        assert np.abs(disp).max() < 1.0


# --------------------------------------------------------------------------- #
# a spatially-varying residual lands in the WARP, not the pose
# --------------------------------------------------------------------------- #
def test_spatially_varying_residual_goes_to_warp_not_pose():
    gt = _coarse_chain(3)

    def field(glob):  # smooth, ~±5 units, well within the match gate
        return 5.0 * np.column_stack(
            [np.sin(glob[:, 1] / 40.0), np.cos(glob[:, 0] / 40.0)]
        )

    coords = _stack(gt, m=20, seed=2, residual=field)
    res = core.register_warps_to_coarse(coords, gt)

    # KEY: the deformation did NOT leak into a global transform.
    _poses_match(res.poses, gt)
    assert res.accepted
    for k, iface in enumerate(res.interfaces):
        assert iface.warp.accepted
        disp = iface.warp.displacement(_ref_xy(coords[k]))
        # the warp captured a real, non-trivial field (vs <1.0 on the clean stack)
        assert np.abs(disp).max() > 2.0


# --------------------------------------------------------------------------- #
# the warp bootstrap recovers pairs a single rigid-frame match drops
# --------------------------------------------------------------------------- #
def test_bootstrap_recovers_pairs_a_single_match_drops():
    # A residual large enough (a few ρ) that the matcher's rigid-residual gate drops
    # the displaced-but-correct pairs on a single match — the exact failure the
    # bootstrap targets. Iteratively pre-warping the moving endpoints pulls those
    # partners back inside the (unchanged) gate, so the converged match fraction
    # clears the first-pass one, the warp stays accepted, and the pose never leaves
    # the coarse chain. ``match_fraction_single`` is the un-bootstrapped first pass.
    gt = _coarse_chain(2)

    def field(glob):  # smooth, several ρ in amplitude
        return 34.0 * np.column_stack(
            [np.sin(glob[:, 1] / 80.0), np.cos(glob[:, 0] / 80.0)]
        )

    coords = _stack(gt, m=45, seed=5, residual=field, spread=(60.0, 60.0))
    res = core.register_warps_to_coarse(coords, gt)

    _poses_match(res.poses, gt)                       # pose stays the coarse chain
    conf = res.interfaces[0].confidence
    single = conf["match_fraction_single"]
    final = conf["match_fraction"]
    assert single < 0.9                               # a single match really dropped pairs
    assert final >= single + 0.1                      # the bootstrap recovered a real chunk
    assert res.interfaces[0].warp.accepted


# --------------------------------------------------------------------------- #
# an MT-free interface is pose-accepted (image-fill covers it), not failed
# --------------------------------------------------------------------------- #
def test_mt_free_interface_is_pose_accepted_not_failed():
    gt = _coarse_chain(2)
    ref = _stack(gt, m=12, seed=3)[0]      # section 0 keeps its MTs
    coords = [ref, np.empty((0, 4), dtype=float)]  # section 1 has no MTs at all
    res = core.register_warps_to_coarse(coords, gt)

    _poses_match(res.poses, gt)            # pose still comes from the image coarse
    assert res.accepted                    # not a failure
    iface = res.interfaces[0]
    assert iface.qc.accepted               # the interface is pose-accepted...
    assert not iface.warp.accepted         # ...with no MT residual warp
    assert iface.warp.displacement(np.zeros((3, 2))).shape == (3, 2)  # safe to call


# --------------------------------------------------------------------------- #
# fit=False keeps the supplied seed verbatim (no rigid re-fit)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# MT rotation rescue: when the image coarse rotation fails, the dense MTs fix it
# --------------------------------------------------------------------------- #
def test_rescue_corrects_a_wrong_coarse_rotation():
    # The MTs correspond at the TRUE rotation, but the coarse pose of section 1 is
    # spun 90° wrong (the sec01->sec02 failure). We inject the match collapse the
    # real bundles show (a dense *synthetic* cloud finds ~30% spurious partners under
    # any rotation, which is a fixture limitation, not the rescue's); the part under
    # test is the rescue's own MT rotation search, which runs on the true coords.
    gt = _coarse_chain(2)
    coords = _stack(gt, m=40, seed=7)        # vertical fibres -> confident MT search
    bad = [dict(gt[0]), compose_poses(gt[1], make_pose(90.0))]

    fixed, rescues = core.rescue_coarse_poses(
        coords, bad, [0.0], match_gate=0.3, search_kwargs={"use_cpd": True},
    )
    assert len(rescues) == 1 and rescues[0][0] == 0
    # the MTs re-estimate the true relative rotation, not the 90°-off image guess
    assert abs(fixed[1]["Angle"] - gt[1]["Angle"]) < 5.0
    # and the corrected pose actually matches the MTs
    base = core.register_warps_to_coarse(coords, fixed)
    assert base.interfaces[0].qc.match_fraction >= 0.3


def test_rescue_leaves_a_good_coarse_alone():
    # No interface should be rescued when the image coarse already matches well —
    # the rescue must not perturb a healthy stack.
    gt = _coarse_chain(3)
    coords = _stack(gt, m=24, seed=11)
    base = core.register_warps_to_coarse(coords, gt)
    mf = [it.qc.match_fraction for it in base.interfaces]
    fixed, rescues = core.rescue_coarse_poses(
        coords, gt, mf, match_gate=0.3, search_kwargs={"use_cpd": True},
    )
    assert rescues == []
    _poses_match(fixed, gt)


def test_stitch_sections_runs_rescue_path_clean():
    # Wiring: the coarse_poses path executes the rescue stage and, on a healthy
    # stack, rescues nothing and leaves the poses untouched.
    gt = _coarse_chain(3)
    coords = _stack(gt, m=30, seed=8)
    res = stitch_sections(coords, coarse_poses=gt)
    assert isinstance(res.rescues, list) and res.rescues == []
    _poses_match(res.poses, gt)


def test_evaluate_seed_fit_false_keeps_seed_as_rel():
    gt = _coarse_chain(2)
    coords = _stack(gt, m=16, seed=4)
    rel = compose_poses(invert_pose(gt[0]), gt[1])
    ref_eps = core._face(coords[0], "top", 0.15)
    mov_eps = core._face(coords[1], "bottom", 0.15)
    rho = core._endpoint_rho(ref_eps)
    out = core._evaluate_seed(rel, ref_eps, mov_eps, rho, False, {}, fit=False)
    assert out is not None
    for key in ("Angle", "Tx", "Ty", "Scale"):
        assert out["rel"][key] == pytest.approx(rel[key], abs=1e-9)
