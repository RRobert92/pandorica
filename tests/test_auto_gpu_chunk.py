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
Tests for ``accel.auto_gpu_chunk`` — the adaptive Z-slice chunk sizer.

The helper has to be safe in three configurations: no GPU available (CPU),
GPU available with a memory-query API (CUDA), GPU available without one
(MPS). We test the device-agnostic bounds and the math by passing the device
identifier as a plain string and monkeypatching the memory query.
"""

import importlib
import pytest

from pandorica.stitch import accel


# --------------------------------------------------------------------------- #
# Per-slice bytes formula
# --------------------------------------------------------------------------- #
def test_per_slice_bytes_grows_with_canvas_area():
    # With Hin·Win held constant, the canvas term (32·Hc·Wc) scales with area;
    # at small input size the canvas dominates, so a 4× canvas-area increase
    # roughly quadruples the per-slice cost.
    a = accel._per_slice_bytes((1000, 1000), (256, 256))
    b = accel._per_slice_bytes((2000, 2000), (256, 256))
    assert b > 3 * a


def test_per_slice_bytes_includes_input_term():
    """Doubling Hin·Win at fixed canvas must still increase the per-slice cost."""
    a = accel._per_slice_bytes((1000, 1000), (1024, 1024))
    b = accel._per_slice_bytes((1000, 1000), (2048, 2048))
    assert b > a


# --------------------------------------------------------------------------- #
# Device dispatch / fallback paths
# --------------------------------------------------------------------------- #
def test_cpu_device_returns_fallback():
    """CPU has no VRAM to budget against — must return the fallback."""
    assert accel.auto_gpu_chunk("cpu", (4000, 4000), (4096, 4096), fallback=7) == 7


def test_mps_returns_fallback(monkeypatch):
    """MPS has no free-memory API — must return the fallback."""
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: None)
    assert accel.auto_gpu_chunk("mps", (4000, 4000), (4096, 4096), fallback=11) == 11


def test_query_failure_returns_fallback(monkeypatch):
    """Even a 'cuda' device returns fallback when mem_get_info fails."""
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: None)
    assert accel.auto_gpu_chunk("cuda", (4000, 4000), (4096, 4096), fallback=3) == 3


# --------------------------------------------------------------------------- #
# Math: chunk = budget // per_slice
# --------------------------------------------------------------------------- #
def test_chunk_sized_from_free_memory(monkeypatch):
    """24 GB free with 50% safety on a 4k×4k canvas → expect a meaningful chunk."""
    free = 24 * 1024 ** 3  # 24 GB
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: free)

    out_hw = (4000, 4000)
    in_hw = (4096, 4096)
    chunk = accel.auto_gpu_chunk("cuda", out_hw, in_hw, safety_frac=0.5, max_chunk=64)

    per_slice = accel._per_slice_bytes(out_hw, in_hw)
    expected = (free * 0.5) // per_slice
    assert chunk == max(1, min(64, int(expected)))
    # Sanity: 24 GB / (32·16M + 4·16M ≈ 576 MB) × 0.5 ≈ 20 slices → clamped at 20.
    assert 10 <= chunk <= 64


def test_chunk_shrinks_as_canvas_grows(monkeypatch):
    """A bigger canvas eats more per-slice memory → smaller chunk for same VRAM."""
    free = 16 * 1024 ** 3
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: free)
    small = accel.auto_gpu_chunk("cuda", (2000, 2000), (4096, 4096), max_chunk=256)
    big = accel.auto_gpu_chunk("cuda", (8000, 8000), (4096, 4096), max_chunk=256)
    assert big < small


def test_chunk_grows_with_more_free_memory(monkeypatch):
    """Same canvas, more VRAM → larger chunk (until max_chunk caps it)."""
    out_hw = (4000, 4000)
    in_hw = (4096, 4096)
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: 8 * 1024 ** 3)
    small = accel.auto_gpu_chunk("cuda", out_hw, in_hw, max_chunk=256)
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: 80 * 1024 ** 3)
    big = accel.auto_gpu_chunk("cuda", out_hw, in_hw, max_chunk=256)
    assert big > small


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #
def test_chunk_clamped_to_max_on_huge_memory(monkeypatch):
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: 1024 * 1024 ** 3)  # 1 TB
    chunk = accel.auto_gpu_chunk("cuda", (1000, 1000), (1024, 1024), max_chunk=32)
    assert chunk == 32


def test_chunk_clamped_to_min_on_tiny_memory(monkeypatch):
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: 1024)  # 1 KB
    chunk = accel.auto_gpu_chunk("cuda", (8000, 8000), (4096, 4096), min_chunk=1)
    assert chunk == 1


def test_safety_frac_is_honored(monkeypatch):
    """Smaller safety_frac → smaller chunk for the same free memory."""
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: 20 * 1024 ** 3)
    out_hw = (4000, 4000)
    in_hw = (4096, 4096)
    conservative = accel.auto_gpu_chunk("cuda", out_hw, in_hw, safety_frac=0.2, max_chunk=256)
    aggressive = accel.auto_gpu_chunk("cuda", out_hw, in_hw, safety_frac=0.8, max_chunk=256)
    assert aggressive >= 3 * conservative


def test_safety_frac_clamped_to_safe_range(monkeypatch):
    """safety_frac outside (0.05, 0.95) is clamped to the bounds."""
    monkeypatch.setattr(accel, "device_free_bytes", lambda _dev: 16 * 1024 ** 3)
    out_hw, in_hw = (4000, 4000), (4096, 4096)
    high = accel.auto_gpu_chunk("cuda", out_hw, in_hw, safety_frac=10.0, max_chunk=256)
    capped = accel.auto_gpu_chunk("cuda", out_hw, in_hw, safety_frac=0.95, max_chunk=256)
    assert high == capped
    low = accel.auto_gpu_chunk("cuda", out_hw, in_hw, safety_frac=-1.0, max_chunk=256)
    floored = accel.auto_gpu_chunk("cuda", out_hw, in_hw, safety_frac=0.05, max_chunk=256)
    assert low == floored


# --------------------------------------------------------------------------- #
# Live device query (only when CUDA is actually present)
# --------------------------------------------------------------------------- #
def test_device_free_bytes_returns_int_or_none_on_real_device():
    """``device_free_bytes`` returns an int or None, never raises."""
    for dev in ("cpu", "mps", "cuda"):
        free = accel.device_free_bytes(dev)
        assert free is None or (isinstance(free, int) and free > 0)
