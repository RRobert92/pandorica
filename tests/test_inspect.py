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

"""Tests for the napari inspection-bundle writer (``inspect.py``)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pandorica.stitch import inspect as insp
from pandorica.stitch.transform.solver import make_pose


class _FakeWarp:
    """Minimal warp: a constant in-plane shear u=(a·y, 0) → |curl| = a everywhere."""

    def __init__(self, a=0.3, accepted=True):
        self._rbf = object()  # not None → displacement is evaluated
        self.accepted = accepted
        self._a = a
        self.certificate = SimpleNamespace(max_abs_vorticity=0.5, min_det_j=0.7)

    def displacement(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.column_stack([self._a * xy[:, 1], np.zeros(len(xy))])


# --------------------------------------------------------------------------- #
# _framed_displacement
# --------------------------------------------------------------------------- #
def test_framed_displacement_identity_is_base():
    w = _FakeWarp(a=0.3)
    xy = np.array([[0.0, 10.0], [5.0, 0.0]])
    d = insp._framed_displacement(w, make_pose(), xy)
    assert np.allclose(d, w.displacement(xy))


def test_framed_displacement_zero_when_no_rbf():
    w = _FakeWarp()
    w._rbf = None
    xy = np.array([[1.0, 2.0]])
    assert np.allclose(insp._framed_displacement(w, make_pose(), xy), 0.0)


# --------------------------------------------------------------------------- #
# _curl_grid
# --------------------------------------------------------------------------- #
def test_curl_grid_recovers_shear_curl():
    w = _FakeWarp(a=0.3)
    curl, extent, pts, disp = insp._curl_grid(
        w, make_pose(), np.array([0.0, 0.0]), np.array([100.0, 100.0]), grid_n=16
    )
    assert curl.shape == (16, 16)
    assert extent.shape == (4,)
    # u=(0.3 y, 0) → curl = ∂u_y/∂x − ∂u_x/∂y = −0.3 → |curl| ≈ 0.3 everywhere.
    assert curl.mean() == pytest.approx(0.3, abs=1e-6)


# --------------------------------------------------------------------------- #
# write_inspection_bundle round-trip
# --------------------------------------------------------------------------- #
def _fake_face(coords, face, zbf):
    base = 0.0 if face == "bottom" else 10.0
    return [
        {"id": i, "pos": np.array([i * 100.0, base, 0.0]), "dir": np.array([0.0, 0.0, 1.0])}
        for i in range(3)
    ]


def test_write_bundle_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(insp, "_face", _fake_face)
    iface = SimpleNamespace(
        warp=_FakeWarp(a=0.3, accepted=False),  # rejected warp, still rendered
        qc=SimpleNamespace(chainable=True, accepted=False, match_fraction=0.74,
                           reasons=["warp not a diffeomorphism"]),
        id_pairs=[(0, 0, 0.1), (1, 1, 0.2), (2, 2, 0.3)],
    )
    result = SimpleNamespace(base=SimpleNamespace(interfaces=[iface]))
    coords = [np.zeros((3, 4)), np.zeros((3, 4))]  # ignored by _fake_face
    poses = [make_pose(), make_pose()]

    out = str(tmp_path / "inspect.npz")
    insp.write_inspection_bundle(result, coords, poses, ["s0", "s1"], out, grid_n=16)

    z = np.load(out, allow_pickle=False)
    man = json.loads(str(z["manifest"]))
    assert man["n_interfaces"] == 1
    m = man["interfaces"][0]
    assert m["name"] == "s0->s1"
    assert m["chainable"] and not m["qc_accepted"]   # the decouple case
    assert m["n_matches"] == 3
    assert z["if0_match_ref"].shape == (3, 2)
    assert z["if0_match_mov"].shape == (3, 2)
    assert z["if0_match_cost"].shape == (3,)
    assert z["if0_curl"].shape == (16, 16)
    assert z["if0_extent"].shape == (4,)
    # warp NOT applied to the rendered match lines (rejected) → mov stays at base y=0.
    assert np.allclose(z["if0_match_mov"][:, 1], 0.0)
