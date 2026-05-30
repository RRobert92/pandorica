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
Small point-cloud helpers used by the stitcher.

Currently exposes :func:`pc_median_dist`, the dataset-derived length unit
(ρ) used throughout :mod:`pandorica.stitch.transform.scale`. The function
returns the **mean of nearest-neighbour distances** — the name follows the
upstream tardis_em_analysis convention even though the statistic is a mean,
not a median. Renaming would silently shift every ρ-scaled threshold in the
stitcher, so we keep the historical name and behaviour.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


def pc_median_dist(
    pc: np.ndarray,
    avg_over: bool = False,
    box_size: float = 0.15,
) -> float:
    """Mean nearest-neighbour distance of a point cloud.

    :param pc: ``(N, 2)`` or ``(N, 3)`` point cloud.
    :param avg_over: if ``True``, restrict the computation to points inside a
        bounding box of side ``2 * box_size`` × the per-axis extent, centred
        on the median position. Useful for getting a robust local scale on
        clouds with heavy outliers.
    :param box_size: half-width of the local box as a fraction of the
        per-axis extent. Only used when ``avg_over`` is ``True``.
    :return: mean of the nearest-neighbour distances. Returns ``1.0`` if the
        cloud has fewer than 3 points (degenerate for kNN).

    The behaviour matches ``tardis_em_analysis.utils.pc_median_dist`` for the
    arguments we use in the stitcher — :func:`pandorica.stitch.transform
    .scale.rho` calls it with default arguments on ``[N, 3]`` arrays, so the
    same downstream ρ-scaled thresholds remain valid.
    """
    pc = np.asarray(pc)

    if avg_over:
        box_dim = pc.shape[1]
        if box_dim not in (2, 3):
            offset_x = offset_y = 0.0
        else:
            offset_x = (pc[:, 0].max() - pc[:, 0].min()) * box_size
            offset_y = (pc[:, 1].max() - pc[:, 1].min()) * box_size
        offset_z = (
            (pc[:, 2].max() - pc[:, 2].min()) * box_size if box_dim == 3 else 0.0
        )

        cx = float(np.median(pc[:, 0]))
        cy = float(np.median(pc[:, 1]))
        cz = float(np.median(pc[:, 2])) if box_dim == 3 else 0.0
        # Inherited quirk: the upstream gates Z-filtering on
        # ``points.shape[0] == 3`` (number of points, not dimension), so 3D
        # clouds with ≠ 3 rows are filtered as 2D columns through all of Z.
        # Replicated for behavioural parity. Pandorica never exercises this
        # branch — `rho()` calls the default ``avg_over=False`` path.
        use_z = pc.shape[0] == 3 and box_dim == 3
        mask = _point_in_bb(
            pc,
            cx - offset_x,
            cx + offset_x,
            cy - offset_y,
            cy + offset_y,
            cz - offset_z if use_z else None,
            cz + offset_z if use_z else None,
        )
        pc = pc[mask]

    if pc.shape[0] < 3:
        return 1.0

    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(pc)
    distances, _ = nn.kneighbors(pc)
    return float(np.mean(distances[:, 1]))


def _point_in_bb(
    points: np.ndarray,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: Optional[float] = None,
    max_z: Optional[float] = None,
) -> np.ndarray:
    """Boolean mask: points strictly inside ``[min, max]`` on each axis."""
    bound = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound &= np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    if points.shape[1] >= 3 and min_z is not None and max_z is not None:
        bound &= np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)
    elif points.shape[1] >= 3:
        bound &= np.ones(points.shape[0], dtype=bool)
    return bound


__all__ = ["pc_median_dist"]
