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
Logic tests for the napari stitch-validator plugin (no Qt / no GL).

Covers the non-UI surface the dock widgets depend on: the pose↔napari geometry
glue, dataset model, pipeline adapters, and the stitched volume + microtubule
export (warp canvas sizing, Z-stacking, graph merge). Widget wiring itself needs
a live napari Viewer (OpenGL) and is exercised manually / offscreen elsewhere.
"""

import os

import numpy as np
import pytest

from pandorica.stitch import geometry as geo
from pandorica.napari import _geometry as npg
from pandorica.stitch import dataset as io
from pandorica.stitch import stitch as st
from pandorica.stitch.pipeline.stitcher import stitch_sections
from pandorica.stitch import image_warp as iw
from pandorica.stitch import accel as accel
from pandorica.stitch import image_pose as ip
from pandorica.stitch.transform.solver import apply_pose, IDENTITY


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_napari_affine_matches_apply_pose():
    pose = {"Angle": 37.0, "Tx": 12.0, "Ty": -5.0, "Scale": 1.3}
    xy = np.array([[3.0, 4.0], [10.0, -2.0], [0.0, 0.0]])
    A = npg.napari_affine(pose)
    zyx1 = np.column_stack([np.zeros(len(xy)), xy[:, 1], xy[:, 0], np.ones(len(xy))])
    got = (A @ zyx1.T).T[:, [2, 1]]  # (x', y')
    assert np.allclose(got, apply_pose(pose, xy))


def test_pose_to_pixel_scales_translation_only():
    pose = {"Angle": 20.0, "Tx": 100.0, "Ty": -40.0, "Scale": 1.0}
    p = geo.pose_to_pixel(pose, pixel_size=10.0)
    assert p["Angle"] == 20.0 and p["Scale"] == 1.0
    assert p["Tx"] == 10.0 and p["Ty"] == -4.0


def test_apply_pose_to_coords_keeps_id_and_z():
    coords = np.array([[7.0, 1.0, 2.0, 99.0]])
    out = npg.apply_pose_to_coords(
        {"Angle": 90.0, "Tx": 0, "Ty": 0, "Scale": 1.0}, coords
    )
    assert out[0, 0] == 7.0 and out[0, 3] == 99.0
    assert np.allclose(out[0, 1:3], [-2.0, 1.0])  # 90° rotation of (1, 2)


def test_coords_to_points_zyx_reorders():
    coords = np.array([[0, 1.0, 2.0, 3.0]])
    assert np.allclose(npg.coords_to_points_zyx(coords), [[3.0, 2.0, 1.0]])


def test_centroid_pose_fixes_center():
    c = np.array([5.0, 5.0])
    pose = geo.centroid_pose(45.0, 0.0, 0.0, c)
    assert np.allclose(apply_pose(pose, c[None])[0], c)  # centre is invariant


def test_napari_affine_2d_matches_apply_pose():
    pose = {"Angle": -28.0, "Tx": 7.0, "Ty": 3.0, "Scale": 0.9}
    xy = np.array([[2.0, 9.0], [-4.0, 1.0]])
    A = npg.napari_affine_2d(pose)
    yx1 = np.column_stack([xy[:, 1], xy[:, 0], np.ones(len(xy))])
    got = (A @ yx1.T).T[:, [1, 0]]  # (x', y')
    assert np.allclose(got, apply_pose(pose, xy))


def test_zmax_face_picks_correct_end():
    vol = np.zeros((6, 2, 2), dtype=np.uint8)
    vol[-1] = 9  # high-Z slice (top)
    vol[0] = 5  # low-Z slice (bottom)
    assert geo.zmax_face(vol, "top", n_slices=2).max() == 9
    assert geo.zmax_face(vol, "bottom", n_slices=2).max() == 5
    # invert_z swaps the two ends
    assert geo.zmax_face(vol, "top", n_slices=2, invert_z=True).max() == 5


# --------------------------------------------------------------------------- #
# section naming — must stay unique (coarse_gt.json keys depend on it)
# --------------------------------------------------------------------------- #
def test_derive_names_secNN():
    stems = [
        "T0619_Grid11A_FemalePN_sec09.rec_flattrim.corrected",
        "T0619_Grid11A_FemalePN_sec10.rec_flattrim.corrected",
    ]
    assert io._derive_names(stems) == ["sec09", "sec10"]


def test_derive_names_trailing_index_unique():
    # MaleMeiosis convention: distinguishing token is the trailing _N
    stems = [f"T0391_worm13_metaphase02_{n}" for n in (4, 5, 6, 7, 8)]
    names = io._derive_names(stems)
    assert names == ["4", "5", "6", "7", "8"]
    assert len(set(names)) == len(names)  # no collision


def test_derive_names_fallback_unique_on_collision():
    names = io._derive_names(["same", "same"])
    assert len(set(names)) == 2


# --------------------------------------------------------------------------- #
# export: synthetic in-memory volumes + graph merge
# --------------------------------------------------------------------------- #
def _mem_section(name, i, vol, px, coords):
    s = io.Section(
        name=name,
        index=i,
        image_path="mem",
        coord_path=None,
        coords=coords,
        pixel_size=px,
    )
    s._volume = vol
    return s


def test_export_stitched_volume_and_graph(tmp_path, monkeypatch):
    # keep preset in-memory volumes alive (export drops + would otherwise reload)
    monkeypatch.setattr(io.Section, "drop_volume", lambda self: None)

    vol = np.zeros((4, 30, 40), np.uint8)
    vol[:, 10:20, 12:28] = 200
    c = np.array([[0, 24.0, 30.0, 2.0], [0, 30.0, 30.0, 4.0]])
    ds = io.Dataset(
        folder=str(tmp_path),
        sections=[
            _mem_section("s0", 0, vol, 2.0, c),
            _mem_section("s1", 1, vol.copy(), 2.0, c.copy()),
        ],
    )
    poses = [dict(IDENTITY), {"Angle": 20.0, "Tx": 30.0, "Ty": -10.0, "Scale": 1.0}]
    out = str(tmp_path / "stitched_output")
    written = st.export_stitched(ds, poses, out, downscale=1, write_volume=True)

    assert os.path.isfile(written["volume"])
    assert os.path.isfile(written["graph"])
    # stitched volume Z must equal the sum of section thicknesses (4 + 4).
    head = open(written["volume"], "rb").read(3000).decode("latin1")
    lattice = next(ln for ln in head.splitlines() if "define Lattice" in ln)
    z = int(lattice.split()[-1])
    assert z == 8


class _MockWarp:
    """Stand-in GuardedWarp with a known displacement field, for frame tests."""

    def __init__(self, field, accepted=True):
        self.accepted = accepted
        self._field = field  # callable [M,2]->[M,2]

    def displacement(self, xy):
        return self._field(np.asarray(xy, float))


def test_framed_warp_maps_through_frame_pose():
    # base displacement defined in the prev-section local frame
    base = _MockWarp(lambda u: u * 0.1)
    P = {"Angle": 30.0, "Tx": 50.0, "Ty": -20.0, "Scale": 1.0}
    fw = st._FramedWarp(base, P, coord_to_A=1.0)
    u = np.array([[3.0, 7.0], [-4.0, 2.0]])
    y = apply_pose(P, u)  # query point in the output frame
    rot = {"Angle": P["Angle"], "Tx": 0.0, "Ty": 0.0, "Scale": P["Scale"]}
    expected = apply_pose(rot, base.displacement(u))  # rotate disp into frame
    assert np.allclose(fw.displacement(y), expected)
    assert fw.accepted


def test_framed_warp_coord_to_A_scaling():
    base = _MockWarp(lambda u: np.full_like(u, 4.0))  # constant 4 Å displacement
    P = dict(IDENTITY)
    fw = st._FramedWarp(base, P, coord_to_A=2.0)  # 2 Å per working unit
    # identity frame: 4 Å displacement -> 2 working units
    assert np.allclose(fw.displacement(np.zeros((1, 2))), [[2.0, 2.0]])


def test_export_warp_changes_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(io.Section, "drop_volume", lambda self: None)
    vol = np.zeros((4, 30, 40), np.uint8)
    vol[:, 8:22, 10:30] = 200
    c = np.array([[0, 24.0, 30.0, 2.0], [0, 30.0, 30.0, 4.0]])

    def make_ds():
        return io.Dataset(
            folder=str(tmp_path),
            sections=[
                _mem_section("s0", 0, vol.copy(), 2.0, c.copy()),
                _mem_section("s1", 1, vol.copy(), 2.0, c.copy()),
            ],
        )

    poses = [dict(IDENTITY), dict(IDENTITY)]
    # section 1 gets a constant-shift "warp" (10 Å); section 0 has none.
    warps = [_MockWarp(lambda u: np.full_like(u, 10.0))]

    st.export_stitched(make_ds(), poses, str(tmp_path / "rigid"), write_volume=True)
    st.export_stitched(
        make_ds(), poses, str(tmp_path / "warped"), write_volume=True, warps=warps
    )
    rigid = open(str(tmp_path / "rigid" / "stitched_volume.am"), "rb").read()
    warped = open(str(tmp_path / "warped" / "stitched_volume.am"), "rb").read()
    assert rigid != warped  # the TPS warp visibly changes the output


def test_zblend_slice_endpoints():
    # bright pixel at (y=10, x=10); bottom slice shifts +3 in x, top slice no shift
    vol = np.zeros((3, 20, 20), np.uint8)
    vol[:, 10, 10] = 255
    yy, xx = np.meshgrid(np.arange(20), np.arange(20), indexing="ij")
    out_pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    b_grid = np.tile([3.0, 0.0], (out_pts.shape[0], 1))  # +3 x at bottom (k=0)
    t_grid = np.zeros_like(out_pts)  # none at top
    out = st._warp_volume_zblend(vol, dict(IDENTITY), (20, 20), out_pts, b_grid, t_grid)
    # bottom slice: pixel moved to x=13; top slice: stays at x=10
    assert np.unravel_index(out[0].argmax(), out[0].shape) == (10, 13)
    assert np.unravel_index(out[-1].argmax(), out[-1].shape) == (10, 10)


def test_rotation_search_recovers_angle():
    # asymmetric pattern (breaks 180° symmetry) rotated by a known angle
    fixed = np.zeros((200, 200), np.float32)
    fixed[60:140, 80:120] = 200.0  # off-centre bar
    fixed[60:80, 80:160] = 200.0  # + arm -> L-shape (asymmetric)
    center = np.array([100.0, 100.0])
    moving = ip._rotate_face(fixed, 25.0, center)  # rotate by +25°
    found = ip.rotation_search(fixed, moving, metric="ncc")
    # to align moving back onto fixed, rotate by -25°
    assert abs(((found - (-25.0)) + 180) % 360 - 180) < 2.0


def test_rigid_from_pairs_recovers_transform():
    # Closed-form weighted 2-D rigid (Kabsch): dst = R(theta)*src + t exactly.
    rng = np.random.default_rng(2)
    src = rng.standard_normal((12, 2)) * 80.0
    th = np.deg2rad(-23.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = np.array([-6.0, 11.0])
    dst = src @ R.T + t
    dth, t_hat = ip._rigid_from_pairs(src, dst)
    assert abs(np.rad2deg(dth) - (-23.0)) < 1e-6
    assert np.allclose(t_hat, t, atol=1e-6)


def test_ransac_rigid_recovers_transform_with_outliers():
    # Weighted RANSAC fits a rigid (rotation+translation) from cell correspondences and
    # rejects gross outliers; the minimal sample is "2 good candidates" (2 point pairs).
    rng = np.random.default_rng(1)
    src = rng.standard_normal((24, 2)) * 100.0
    th = np.deg2rad(18.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = np.array([7.0, -4.0])
    dst = src @ R.T + t
    dst[:9] = rng.standard_normal((9, 2)) * 400.0  # 9/24 gross outliers
    conf = np.ones(24)
    dth, t_hat, inl, support = ip._ransac_rigid(src, dst, conf, tol=3.0)
    assert abs(np.rad2deg(dth) - 18.0) < 0.5  # rotation recovered through the outliers
    assert np.allclose(t_hat, t, atol=0.5)  # translation recovered
    assert not inl[:9].any() and inl[9:].all()  # exactly the 15 clean pairs are inliers
    assert support == 15.0  # weighted support = sum conf over inliers (conf == 1)


def test_agree_rotation_picks_true_branch_by_support():
    # Image-only rotation ranks each swept angle by weighted RANSAC inlier SUPPORT (sum
    # of match confidence over cells consistent with ONE rigid transform), not a raw
    # agreeing-cell count. A wrong 180° flip leaves few rigidly consistent cells, so its
    # support collapses while the true angle's stays high. Realistically-scaled blocky
    # texture (raw std ≳ 5, ~0–255) so block_match's std / peakiness gates pass, and the
    # 8-px blocks survive the rotation interpolation.
    rng = np.random.default_rng(0)
    fixed = (np.kron(rng.standard_normal((32, 32)), np.ones((8, 8))) * 50 + 128).astype(
        np.float32
    )  # 256×256
    h, w = fixed.shape
    center = np.array([w / 2.0, h / 2.0])
    true = 15.0
    moving = ip._rotate_face(fixed, true, center)  # moving = fixed rotated by +15°
    mk = dict(metric="ncc", grid=8, search=24, tol=8.0, workers=1)
    win, opp = ip._agree_rotation(fixed, moving, center, mk=mk)
    # recovers the true branch (≈ -15°, aligning moving onto fixed), NOT the +165° flip
    assert ip._angle_gap(win["rot"], -true) < 5.0
    assert opp is not None
    # the flip keeps few rigidly consistent cells, so it loses on weighted SUPPORT
    assert win["support"] > opp["support"]


def test_gpu_warp_matches_cpu_zblend():
    # torch grid_sample (CPU) must reproduce the scipy Z-blend warp
    rng = np.random.default_rng(0)
    vol = (rng.random((3, 40, 50)) * 255).astype(np.uint8)
    hc, wc = 40, 50
    yy, xx = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    out_pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    b = np.tile([2.0, -1.0], (out_pts.shape[0], 1))  # low-Z shift
    t = np.zeros_like(out_pts)  # high-Z none
    inv = dict(IDENTITY)
    cpu = st._warp_volume_zblend(vol, inv, (hc, wc), out_pts, b, t)
    gpu = accel.warp_volume_torch(
        vol, inv, (hc, wc), out_pts, b, t, device="cpu", chunk=2
    )
    assert cpu.shape == gpu.shape
    assert np.mean(np.abs(cpu.astype(int) - gpu.astype(int))) < 2.0


def test_export_zblend_differs_from_uniform(tmp_path, monkeypatch):
    monkeypatch.setattr(io.Section, "drop_volume", lambda self: None)
    vol = np.zeros((4, 30, 40), np.uint8)
    vol[:, 8:22, 10:30] = 200
    c = np.array([[0, 24.0, 30.0, 2.0], [0, 30.0, 30.0, 4.0]])

    def make_ds():
        return io.Dataset(
            folder=str(tmp_path),
            sections=[
                _mem_section("s0", 0, vol.copy(), 2.0, c.copy()),
                _mem_section("s1", 1, vol.copy(), 2.0, c.copy()),
            ],
        )

    poses = [dict(IDENTITY), dict(IDENTITY)]
    warps = [_MockWarp(lambda u: np.full_like(u, 10.0))]
    st.export_stitched(make_ds(), poses, str(tmp_path / "uni"), warps=warps)
    st.export_stitched(
        make_ds(), poses, str(tmp_path / "zb"), warps=warps, warp_zblend=True
    )
    uni = open(str(tmp_path / "uni" / "stitched_volume.am"), "rb").read()
    zb = open(str(tmp_path / "zb" / "stitched_volume.am"), "rb").read()
    assert uni != zb  # Z-blend (symmetric, per-slice) differs from uniform warp


# --------------------------------------------------------------------------- #
# image-fill warp (NCC block-match + MT mask)
# --------------------------------------------------------------------------- #
def test_block_match_recovers_known_shift():
    rng = np.random.default_rng(0)
    fixed = rng.random((256, 256)).astype(np.float32) * 100
    # moving = fixed shifted by (+6 x, -4 y): moving[y,x] = fixed[y+4, x-6]
    moving = np.roll(fixed, shift=(4, -6), axis=(0, 1))
    src, dst, conf = iw.block_match_ncc(
        fixed, moving, grid=6, half=24, search=12, min_ncc=0.5
    )
    assert len(src) > 0
    shift = (dst - src).mean(0)  # should recover (+6, -4)
    assert np.allclose(shift, [6.0, -4.0], atol=1.0)


def test_mt_mask_excludes_matches():
    rng = np.random.default_rng(1)
    fixed = rng.random((256, 256)).astype(np.float32) * 100
    moving = np.roll(fixed, shift=(3, 3), axis=(0, 1))
    full = np.zeros((256, 256), bool)  # no MTs -> matches allowed
    masked = np.ones((256, 256), bool)  # all MT -> everything excluded
    n_open = len(iw.block_match_ncc(fixed, moving, full, grid=6, half=24, search=10)[0])
    n_blocked = len(
        iw.block_match_ncc(fixed, moving, masked, grid=6, half=24, search=10)[0]
    )
    assert n_open > 0 and n_blocked == 0


def test_sum_warp_adds_displacements():
    a = _MockWarp(lambda u: np.full_like(u, 3.0))
    b = _MockWarp(lambda u: np.full_like(u, 4.0))
    s = st._SumWarp([a, b, None, _MockWarp(None, accepted=False)])
    assert s.accepted
    assert np.allclose(s.displacement(np.zeros((2, 2))), 7.0)
    assert not st._SumWarp([]).accepted


def test_export_graph_only(tmp_path):
    c = np.array([[0, 1.0, 2.0, 0.0], [0, 5.0, 6.0, 1.0]])
    ds = io.Dataset(
        folder=str(tmp_path),
        sections=[
            io.Section("s0", 0, None, None, c.copy()),
            io.Section("s1", 1, None, None, c.copy()),
        ],
    )
    poses = [dict(IDENTITY), dict(IDENTITY)]
    written = st.export_stitched(ds, poses, str(tmp_path / "out"), write_volume=False)
    assert "volume" not in written
    assert os.path.isfile(written["graph"])


# --------------------------------------------------------------------------- #
# real-data path (skipped if the example dataset is absent)
# --------------------------------------------------------------------------- #
_DATA = os.environ.get(
    "TARDIS_TEST_DATA", "/Users/robertkiewisz/Downloads/C.elegans_FemalePN"
)


@pytest.mark.skipif(not os.path.isdir(_DATA), reason="example dataset not present")
def test_real_dataset_load_run_rows():
    ds = io.load_dataset(_DATA)
    assert len(ds) >= 2
    coords = ds.coords_list()
    res_c = stitch_sections(coords)
    assert len(res_c.poses) == len(ds)
    rows = st.interface_rows(res_c, ds)
    assert len(rows) == len(ds) - 1
    assert {"interface", "hybrid_deg", "qc_ok"} <= set(rows[0])
