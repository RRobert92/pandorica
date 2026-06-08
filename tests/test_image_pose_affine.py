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
Tests for the anisotropic/shear affine refine in ``image_pose`` (task #13).

Three layers:
  * ``_fit_affine`` / ``_ransac_affine`` recover a known affine and reject outliers;
  * ``_affine_refine`` on a synthetic textured face injects ~no anisotropy on a pure
    rotation (the iso primary case) and recovers a planted ``(sx, sy)`` on an aniso
    plant — both validated against real EM faces in tmp/verify_affine_refine.py, here
    pinned with reproducible synthetic texture;
  * an end-to-end ``image_only_poses`` run commits the planted anisotropy into the
    pose and surfaces ``(sx, sy)`` through ``on_interface``.
"""

import cv2
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pandorica.stitch import dataset as io
from pandorica.stitch import image_pose as ip
from pandorica.stitch.transform.solver import apply_pose, linear_part


def _texture(size=256, seed=0):
    """High-frequency textured face block_match can lock onto."""
    rng = np.random.default_rng(seed)
    t = gaussian_filter(rng.normal(0, 1, (size, size)).astype(np.float32), 2.0)
    t = (t - t.min()) / (np.ptp(t) + 1e-9) * 255.0
    return t.astype(np.float32)


def _render(fixed, L, t, noise=4.0, seed=1):
    """moving(y) = fixed(L^-1 (y - t)); block_match(fixed, moving) then recovers L."""
    h, w = fixed.shape
    Linv = np.linalg.inv(L)
    M = np.hstack([Linv, (-Linv @ np.asarray(t, float)).reshape(2, 1)]).astype(np.float64)
    mov = cv2.warpAffine(fixed, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    rng = np.random.default_rng(seed)
    return (mov + rng.normal(0, noise, mov.shape)).astype(np.float32)


def _R(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# --------------------------------------------------------------------------- #
# fitters
# --------------------------------------------------------------------------- #
def test_fit_affine_recovers_known():
    rng = np.random.default_rng(0)
    A = np.array([[1.1, 0.08], [-0.05, 0.93]])
    t = np.array([4.0, -7.0])
    src = rng.normal(0, 20, (30, 2))
    dst = src @ A.T + t
    A_hat, t_hat = ip._fit_affine(src, dst)
    assert np.allclose(A_hat, A, atol=1e-9)
    assert np.allclose(t_hat, t, atol=1e-9)


def test_ransac_affine_rejects_outliers():
    rng = np.random.default_rng(3)
    A = np.array([[0.95, 0.04], [0.02, 1.07]])
    t = np.array([6.0, 3.0])
    src = rng.normal(0, 25, (60, 2))
    dst = src @ A.T + t
    out = rng.random(60) < 0.3            # 30% gross outliers
    dst[out] += rng.uniform(-80, 80, (out.sum(), 2))
    conf = np.ones(60)
    A_hat, t_hat, inl, _ = ip._ransac_affine(src, dst, conf, tol=2.0)
    assert np.allclose(A_hat, A, atol=0.02)
    assert inl.sum() >= (~out).sum() - 2   # recovers ~all true inliers
    assert not inl[out].any()              # and excludes every planted outlier


# --------------------------------------------------------------------------- #
# _clamp_affine: physical guard on the committed residual-affine
# --------------------------------------------------------------------------- #
def test_clamp_affine_reins_over_area():
    # A both-axes inflation (det 1.32 — the overfit two adjacent EM sections cannot
    # show) is pulled back into the det band, keeping its orientation and anisotropy
    # DIRECTION (the ratio is preserved by the isotropic rescale).
    A = np.diag([1.20, 1.10])
    A_c, changed = ip._clamp_affine(A)
    assert changed
    assert np.linalg.det(A_c) <= ip._AFFINE_DET_BAND[1] + 1e-9
    # still anisotropic (a real stretch direction survives, only the area is reined)
    sv = np.linalg.svd(A_c, compute_uv=False)
    assert sv.max() / sv.min() > 1.0


def test_clamp_affine_leaves_physical_aniso_untouched():
    # A genuine mild aniso/compression (det 0.94, both singular values inside the band
    # — the Monopoles sec10->sec11 regime that HELPS the MTs) passes through unchanged.
    A = _R(7.0) @ np.diag([1.02, 0.92])
    A_c, changed = ip._clamp_affine(A)
    assert not changed
    assert np.allclose(A_c, A, atol=1e-6)


def test_affine_refine_clamps_extreme_stretch():
    # End-to-end through the refine: an extreme planted stretch (det 1.44) is committed
    # only up to the physical band, not at its raw overfit magnitude.
    fixed = _texture(256, seed=8)
    moving = _render(fixed, np.diag([1.20, 1.20]), np.array([5.0, -3.0]))
    out = ip._affine_refine(fixed, moving, 0.0, np.array([128.0, 128.0]), **_MK)
    assert out is not None
    L, _shift, _sv, _n = out
    assert np.linalg.det(L) <= ip._AFFINE_DET_BAND[1] + 1e-6


# --------------------------------------------------------------------------- #
# _affine_refine on a textured face (ang=0 isolates the fit from angle search)
# --------------------------------------------------------------------------- #
_MK = dict(metric="ncc", grid=12, search=40, workers=2)


def test_affine_refine_no_spurious_aniso_on_pure_rotation():
    # A pure rotation has singular values (1, 1): the refine must NOT manufacture
    # anisotropy from a rotation (this is what protects the isotropic primary case).
    fixed = _texture(256, seed=1)
    moving = _render(fixed, _R(9.0), np.array([8.0, -5.0]))
    out = ip._affine_refine(fixed, moving, 9.0, np.array([128.0, 128.0]), **_MK)
    assert out is not None
    _L, _shift, (sx, sy), _n = out
    assert max(sx, sy) / min(sx, sy) - 1.0 < 0.02   # essentially isotropic


def test_affine_refine_recovers_planted_anisotropy():
    # Pure axis-aligned stretch at zero rotation -> singular values recover (sx, sy).
    fixed = _texture(256, seed=2)
    moving = _render(fixed, np.diag([1.12, 0.90]), np.array([6.0, -4.0]))
    out = ip._affine_refine(fixed, moving, 0.0, np.array([128.0, 128.0]), **_MK)
    assert out is not None
    _L, _shift, (sx, sy), _n = out
    assert sorted((sx, sy)) == pytest.approx([0.90, 1.12], abs=0.04)


# --------------------------------------------------------------------------- #
# end-to-end: image_only_poses commits the aniso and surfaces (sx, sy)
# --------------------------------------------------------------------------- #
def _mem_section(name, i, vol, px):
    s = io.Section(name=name, index=i, image_path="mem", coord_path=None,
                   coords=np.empty((0, 4)), pixel_size=px)
    s._volume = vol
    return s


def test_image_only_poses_commits_anisotropy(monkeypatch):
    monkeypatch.setattr(io.Section, "drop_volume", lambda self: None)
    # Two sections; section-1 bottom face = anisotropic warp of section-0 top face.
    face0 = _texture(256, seed=5)
    L_true = np.diag([1.12, 0.90])
    face1 = _render(face0, L_true, np.array([7.0, -4.0]), noise=3.0)
    vol0 = np.repeat(face0[None], 3, axis=0)   # constant-Z stacks -> face == the texture
    vol1 = np.repeat(face1[None], 3, axis=0)
    ds = io.Dataset(folder="mem", sections=[
        _mem_section("s0", 0, vol0, 1.0),
        _mem_section("s1", 1, vol1, 1.0),
    ])

    seen = []
    poses = ip.image_only_poses(
        ds, load_downscale=1, n_slices=3, match_search=40,
        on_interface=lambda info: seen.append(info), log=lambda *a, **k: None,
    )
    assert len(seen) == 1
    info = seen[0]
    assert info["aniso_committed"] is True
    assert sorted((info["sx"], info["sy"])) == pytest.approx([0.90, 1.12], abs=0.05)
    # the committed relative pose's linear part must be anisotropic (not a similarity)
    rel_L = linear_part(poses[1])
    svals = np.linalg.svd(rel_L, compute_uv=False)
    assert svals.max() / svals.min() - 1.0 > 0.10
