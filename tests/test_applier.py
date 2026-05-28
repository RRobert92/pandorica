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

"""Tests for the slice-wise volume warp applier (``applier.py``)."""

import numpy as np

from pandorica.stitch.transform import applier


def test_identity_map_roundtrips():
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 256, size=(3, 8, 8), dtype=np.uint8)
    out = applier.warp_volume_slicewise(vol, lambda p: p)
    assert out.dtype == np.uint8
    assert np.array_equal(out, vol)


def test_pose_translation_shifts_feature():
    vol = np.zeros((2, 9, 9), dtype=np.uint8)
    vol[1, 3, 2] = 200  # marker at (z=1, y=3, x=2)
    pose = {"Angle": 0.0, "Tx": 2.0, "Ty": 0.0, "Scale": 1.0}  # moving x -> x + 2
    inv = applier.make_inverse_map(pose)
    out = applier.warp_volume_slicewise(vol, inv)
    # Feature should land at x = 4 in the reference frame.
    assert out[1, 3, 4] == 200
    assert out[1, 3, 2] == 0


def test_output_dtype_is_uint8_from_float_input():
    vol = np.full((2, 5, 5), 130.7, dtype=np.float32)
    out = applier.warp_volume_slicewise(vol, lambda p: p)
    assert out.dtype == np.uint8
    assert out[0, 0, 0] == 130  # cast, not the int8-wrapped negative


def test_memmap_io_is_slicewise(tmp_path):
    # Both input and output as on-disk memmaps — the applier must never need the
    # whole stack in RAM.
    src = np.memmap(tmp_path / "in.dat", dtype=np.uint8, mode="w+", shape=(4, 6, 6))
    src[:] = 77
    src.flush()
    dst = np.memmap(tmp_path / "out.dat", dtype=np.uint8, mode="w+", shape=(4, 6, 6))
    out = applier.warp_volume_slicewise(src, lambda p: p, output=dst)
    assert out.shape == (4, 6, 6)
    assert np.all(np.asarray(out) == 77)
