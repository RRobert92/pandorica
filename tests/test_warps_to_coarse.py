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
    pose_from_matrix,
    linear_part,
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


# --------------------------------------------------------------------------- #
# scale-gate: drop an overfit image scale the MTs reject (rotation-only matches better)
# --------------------------------------------------------------------------- #
def test_scale_gate_drops_an_overfit_image_scale():
    # The MTs correspond under a RIGID gt (rotation + shift). The image coarse pose for
    # section 1 carries an extra anisotropic over-stretch (det ~1.12) the MTs reject:
    # rotation-only then matches better, so the gate must drop the scale and re-accumulate
    # the chain back to ~rotation-only. This is the sec12->sec13 failure in miniature.
    gt = _coarse_chain(2)
    coords = _stack(gt, m=45, seed=5, spread=(60.0, 60.0))
    # An anisotropic scale about a center FAR from the cloud (as on real data, where the
    # section centre is tens of thousands of Å from the origin): it both stretches AND
    # mis-maps the whole cloud, collapsing the match — exactly what the MTs reject.
    R1, t1 = linear_part(gt[1]), np.array([gt[1]["Tx"], gt[1]["Ty"]])
    S, c0 = np.diag([1.16, 0.97]), np.array([5000.0, 0.0])  # det(S) ~1.125
    bad = pose_from_matrix(R1 @ S, R1 @ ((np.eye(2) - S) @ c0) + t1)

    new_poses, gated = core.gate_coarse_scale(coords, [dict(gt[0]), bad])

    assert len(gated) == 1 and gated[0][0] == 0
    assert gated[0][3] > gated[0][2]                       # rot-only beat the full scale
    rel = compose_poses(invert_pose(new_poses[0]), new_poses[1])
    sv = np.linalg.svd(linear_part(rel), compute_uv=False)
    assert max(abs(sv[0] - 1.0), abs(sv[1] - 1.0)) < 0.05  # scale dropped


def test_scale_gate_keeps_a_genuine_aniso():
    # The MTs correspond under a genuine ANISO gt, so the full pose (with that scale)
    # matches best — the gate must NOT drop it (the knife-compression case that HELPS).
    L = np.diag([1.12, 0.96]) @ linear_part(make_pose(0.6))
    gt = [dict(IDENTITY), pose_from_matrix(L, np.array([6.0, -3.0]))]
    coords = _stack(gt, m=45, seed=6, spread=(60.0, 60.0))

    new_poses, gated = core.gate_coarse_scale(coords, gt)

    assert gated == []                                     # genuine aniso kept
    _poses_match(new_poses, gt)                            # poses untouched


def test_fit_affine_2d_recovers_known_affine():
    rng = np.random.default_rng(0)
    src = rng.uniform(-50, 50, size=(30, 2))
    L_true = np.array([[1.10, 0.05], [-0.03, 0.90]])
    t_true = np.array([3.0, -2.0])
    L, t = core._fit_affine_2d(src, src @ L_true.T + t_true)
    assert np.allclose(L, L_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-6)


def test_rescue_recovers_anisotropy_when_image_failed():
    # Image coarse failed (section 1 spun 90° wrong) AND the dense MTs correspond under a
    # genuine ANISOTROPIC gt. The rescue must recover BOTH the rotation and the anisotropy,
    # so the gap a rigid-only rescue would leave is closed in the COARSE pose.
    L = np.diag([1.12, 0.96]) @ linear_part(make_pose(0.6))
    gt = [dict(IDENTITY), pose_from_matrix(L, np.array([6.0, -3.0]))]
    coords = _stack(gt, m=45, seed=6, spread=(60.0, 60.0))
    bad = [dict(gt[0]), compose_poses(gt[1], make_pose(90.0))]

    fixed, rescues = core.rescue_coarse_poses(
        coords, bad, [0.0], match_gate=0.3, search_kwargs={"use_cpd": True},
    )
    assert len(rescues) == 1
    aniso = rescues[0][5]
    assert aniso is not None                          # anisotropy was recovered
    assert aniso[0] / aniso[1] > 1.08                 # ~the planted 1.12/0.96 = 1.17
    sv = np.linalg.svd(
        linear_part(compose_poses(invert_pose(fixed[0]), fixed[1])), compute_uv=False
    )
    assert sv[0] / sv[1] > 1.08                        # the corrected pose carries the aniso


def test_rescue_recovers_no_anisotropy_when_isotropic():
    # Isotropic gt: the rescue fixes the rotation but must NOT invent anisotropy (gate safety).
    gt = _coarse_chain(2)
    coords = _stack(gt, m=40, seed=7)
    bad = [dict(gt[0]), compose_poses(gt[1], make_pose(90.0))]

    fixed, rescues = core.rescue_coarse_poses(
        coords, bad, [0.0], match_gate=0.3, search_kwargs={"use_cpd": True},
    )
    assert len(rescues) == 1
    assert rescues[0][5] is None                       # no spurious anisotropy


def test_cut_vertical_jog_drops_only_vertical_local_outlier():
    # 16 pairs on a grid sharing a smooth coarse displacement, EXCEPT one NEAR-VERTICAL pair
    # whose displacement deviates from its neighbourhood -> a local outlier -> cut. A shallow
    # outlier (reliable tangent) is left for the direction-based split; a true pair in a
    # uniformly-displaced (deformed) field is NOT cut (it agrees with its neighbours).
    pos = np.array([[x, y] for x in range(4) for y in range(4)], float) * 3.0
    shallow, vert = [1.0, 0.0], [0.05, 0.05]
    out_pos = 5
    id_pairs = [(i, i, 0.1) for i in range(16)]

    dirs = np.tile(shallow, (16, 1)); dirs[out_pos] = vert
    disp = np.tile([1.0, 0.0], (16, 1)); disp[out_pos] = [1.0, 6.0]   # deviates ~6ρ
    out = core._cut_vertical_jog(id_pairs, pos + disp, pos, dirs, dirs, 1.0, 2.0, k=6)
    assert out_pos not in [p[0] for p in out] and len(out) == 15      # vertical outlier cut

    dirs2 = np.tile(shallow, (16, 1))                                 # same outlier, shallow
    out2 = core._cut_vertical_jog(id_pairs, pos + disp, pos, dirs2, dirs2, 1.0, 2.0, k=6)
    assert len(out2) == 16                                            # split judges shallow

    smooth = np.tile([2.0, 0.0], (16, 1))                             # uniform 2ρ deformation
    out3 = core._cut_vertical_jog(id_pairs, pos + smooth, pos, dirs, dirs, 1.0, 2.0, k=6)
    assert len(out3) == 16                                            # no local outlier -> kept

    assert len(core._cut_vertical_jog(id_pairs, pos + disp, pos, dirs, dirs, 1.0, 0.0, k=6)) == 16


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
