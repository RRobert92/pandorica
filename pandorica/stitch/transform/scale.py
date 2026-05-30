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
Scale unit (ρ) and boundary-landmark helpers for serial-section stitching.

This module is the single, stable surface that later stitching stages import for
two foundational primitives:

    * ``rho(coords)`` — the median nearest-neighbour spacing of a section. ρ is the
      natural length unit of the data; expressing every downstream threshold in
      units of ρ makes the pipeline portable across voxel sizes (no raw-nm magic
      numbers).
    * ``boundary_landmarks(coords, boundary)`` — the microtubule endpoints (and
      local tangents) at one Z face of a section, the cross-gap matching landmarks.

Both functions thinly wrap existing, tested primitives so callers depend on one
import rather than reaching into ``utils`` and ``matching.mt_endpoints`` directly.
"""

from typing import List, Dict

import numpy as np

from pandorica.utils.pointcloud import pc_median_dist
from pandorica.stitch.matching.mt_endpoints import (
    extract_boundary_endpoints,
)


def _xyz(coords: np.ndarray) -> np.ndarray:
    """
    Return the spatial ``[x, y, z]`` columns from a point array.

    Accepts the ``[N, 4]`` ``[id, x, y, z]`` spatial-graph contract or a bare
    ``[N, 3]`` point cloud.

    :param coords: Point array of shape ``[N, 4]`` or ``[N, 3]``.
    :type coords: np.ndarray
    :return: Spatial coordinates of shape ``[N, 3]``.
    :rtype: np.ndarray
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] not in (3, 4):
        raise ValueError(f"coords must be [N, 3] or [N, 4]; got shape {coords.shape}")
    return coords[:, 1:4] if coords.shape[1] == 4 else coords


def rho(coords: np.ndarray) -> float:
    """
    Median nearest-neighbour spacing (ρ) of a section, in coordinate units.

    ρ is the dataset-derived length unit used to normalise every spatial
    threshold downstream, replacing voxel-scale-dependent constants.

    :param coords: Spatial graph ``[N, 4]`` (``[id, x, y, z]``) or point cloud
        ``[N, 3]``.
    :type coords: np.ndarray
    :return: Median nearest-neighbour distance.
    :rtype: float
    """
    return float(pc_median_dist(_xyz(coords)))


def boundary_landmarks(
    coords: np.ndarray,
    boundary: str = "bottom",
    z_band_fraction: float = 0.15,
    min_direction_pts: int = 3,
) -> List[Dict]:
    """
    Microtubule endpoints and local tangents at one Z face of a section.

    These endpoints are the landmarks matched across the gap between two serial
    sections. ``boundary='bottom'`` selects the high-Z face, ``'top'`` the
    low-Z face.

    :param coords: Spatial graph ``[N, 4]`` (``[id, x, y, z]``). Points sharing
        column 0 form one microtubule.
    :type coords: np.ndarray
    :param boundary: ``'bottom'`` (high-Z face) or ``'top'`` (low-Z face).
    :type boundary: str
    :param z_band_fraction: Fraction of the total Z-range defining the boundary
        zone; only MTs whose endpoint falls inside it are returned.
    :type z_band_fraction: float
    :param min_direction_pts: Minimum points used for the local direction estimate.
    :type min_direction_pts: int
    :return: List of dicts with keys ``id`` (MT id), ``pos`` (``[x, y, z]``
        endpoint), ``dir`` (unit tangent toward the boundary).
    :rtype: list[dict]
    """
    if boundary not in ("bottom", "top"):
        raise ValueError(f"boundary must be 'bottom' or 'top'; got {boundary!r}")

    return extract_boundary_endpoints(
        np.asarray(coords),
        boundary=boundary,
        z_band_fraction=z_band_fraction,
        min_direction_pts=min_direction_pts,
    )
