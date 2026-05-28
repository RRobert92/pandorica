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
Slice-wise, memmap-friendly, dtype-safe volume warp applier.

Replaces an eager full-stack RAM load with a streaming applier that reads and
writes one Z-slice at a time, so peak memory is one slice — not the whole
multi-GB stack.

The serial-section warp is **2-D in-plane and identical across Z** within a
section (EM-confirmed: the warp aligns the XY plane), so the (inverse) sampling
coordinates are computed **once** and reused for every slice.

Output dtype is **uint8** by default, matching the byte-only ``to_am``
contract — never int8 (which corrupted voxels ≥ 128).
"""

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from pandorica.stitch.transform.solver import (
    Pose,
    apply_pose,
    invert_pose,
)


def make_inverse_map(pose: Pose, warp=None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build the output→input in-plane sampling map for resampling.

    The forward map (moving → reference) is ``f(x) = warp(pose(x))``. To fill an
    output (reference-frame) pixel ``y`` we need ``f⁻¹(y) = pose⁻¹(warp⁻¹(y))``.
    The pose inverse is analytic; the TPS inverse uses the small-displacement
    approximation ``warp⁻¹(y) ≈ y − u(y)``, valid because the warp guard
    guarantees a small, smooth (diffeomorphic) field.

    :param pose: the section's rigid+scale pose (moving → reference).
    :param warp: optional ``GuardedWarp``; ignored unless ``warp.accepted``.
    :return: callable mapping ``[M, 2]`` output ``(x, y)`` to input ``(x, y)``.
    """
    inv_pose = invert_pose(pose)

    def inverse_map(out_xy: np.ndarray) -> np.ndarray:
        y = np.asarray(out_xy, dtype=float)
        if warp is not None and getattr(warp, "accepted", False):
            y = y - warp.displacement(y)  # warp⁻¹ (small-displacement)
        return apply_pose(inv_pose, y)

    return inverse_map


def warp_volume_slicewise(
    volume,
    inverse_map: Callable[[np.ndarray], np.ndarray],
    out_hw: Optional[Tuple[int, int]] = None,
    output=None,
    order: int = 1,
    dtype=np.uint8,
    cval: float = 0.0,
):
    """
    Apply a 2-D in-plane ``inverse_map`` to every Z-slice of a ``[Z, Y, X]`` volume.

    Reads and writes one slice at a time, so ``volume`` and ``output`` may be
    ``np.memmap`` and the full stack is never materialised in RAM.

    :param volume: input ``[Z, Y, X]`` array or memmap.
    :param inverse_map: callable ``[M, 2]`` output ``(x, y)`` → input ``(x, y)``.
    :param out_hw: output ``(H, W)`` (default: same as input slice).
    :param output: optional preallocated ``[Z, H, W]`` array/memmap to write into.
    :param order: spline interpolation order for ``map_coordinates``.
    :param dtype: output dtype (default uint8 — the ``to_am`` byte contract).
    :param cval: fill value outside the input.
    :return: the warped ``[Z, H, W]`` volume (``output`` if supplied).
    """
    z, in_h, in_w = volume.shape
    h, w = out_hw if out_hw is not None else (in_h, in_w)

    if output is None:
        output = np.empty((z, h, w), dtype=dtype)

    # Output grid (x, y), mapped once — identical for every slice.
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    out_pts = np.column_stack([xx.ravel(), yy.ravel()])
    src = inverse_map(out_pts)  # input (x, y)
    coords = np.vstack([src[:, 1], src[:, 0]])  # map_coordinates wants (row=y, col=x)

    for k in range(z):
        sl = np.asarray(volume[k])  # one slice into RAM
        warped = map_coordinates(sl, coords, order=order, mode="constant", cval=cval)
        output[k] = warped.reshape(h, w).astype(dtype)
    return output
