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
Hybrid coarse-rotation orchestrator — ties the pieces into ``coarse_angles``.

For each serial-section interface it runs the MT global rotation search
(``rotation_search``), optionally computes an image A–P-polarity sign hint
(``ap_polarity``), resolves the sign / abstains (``coarse_fusion``), and finally
applies a **stack-wide continuity** pass: a flagged (ambiguous) interface is
resolved toward the trend of the confident interfaces when one of its two 180°
branches clearly matches and the other doesn't — but a genuinely ambiguous large
rotation (both branches far from the trend) stays flagged rather than guessed.

Output is a per-interface angle list ready for
``core.register_section_stack(coords_list, coarse_angles=...)`` plus the full
per-interface records (source, flagged, diagnostics) for QC.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from pandorica.stitch.coarse.rotation_search import (
    RotationEstimate,
    global_rotation_search,
)
from pandorica.stitch.coarse.coarse_fusion import (
    ResolvedRotation,
    resolve_stack_rotations,
    _wrap,
)
from pandorica.stitch.coarse.ap_polarity import ap_rotation_hint
from pandorica.stitch.pipeline.core import _face, _endpoint_rho


@dataclass
class HybridCoarseResult:
    """Coarse-rotation solution for a stack."""

    angles: List[float]  # per interface (mov→ref), feed to register_section_stack
    records: List[ResolvedRotation]
    estimates: List[RotationEstimate]


def _continuity_resolve(records: List[ResolvedRotation], tol: float) -> None:
    """Upgrade flagged interfaces toward the confident trend, in place."""
    anchored = [r.angle for r in records if r.source in ("confident", "ap_polarity")]
    trend = float(np.median(anchored)) if anchored else 0.0
    for r in records:
        if not r.flagged:
            continue
        cands = [_wrap(r.estimate.angle), _wrap(r.estimate.angle + 180.0)]
        d = [abs(_wrap(c - trend)) for c in cands]
        i = int(np.argmin(d))
        # Resolve only when ONE branch clearly matches the trend (the small-vs-
        # 180°-off case); a genuinely ambiguous large rotation stays flagged.
        if d[i] <= tol and d[1 - i] > tol:
            r.angle, r.source, r.flagged = cands[i], "continuity", False


def hybrid_coarse(
    coords_list,
    section_images: Optional[Sequence[np.ndarray]] = None,
    z_band_fraction: float = 0.15,
    continuity_tol: float = 30.0,
    search_kwargs: Optional[dict] = None,
    ap_kwargs: Optional[dict] = None,
    progress: Optional[Callable[[int, int, float], None]] = None,
) -> HybridCoarseResult:
    """
    Compute stack coarse rotations by the hybrid (MT search + A–P sign + continuity).

    :param coords_list: per-section spatial graphs ``[N, 4]``, in stack order.
    :param section_images: optional one 2-D image per section (e.g. a projection)
        for the A–P-polarity sign hint; ``None`` → MT-only + continuity.
    :param continuity_tol: trend tolerance (deg) for resolving flagged interfaces.
    :param search_kwargs / ap_kwargs: forwarded to ``global_rotation_search`` /
        ``ap_rotation_hint``.
    :param progress: optional ``(k_done, n_total, angle_for_k)`` callback fired
        once per interface after its rotation estimate completes. ``k_done`` runs
        ``0..n_total-1``; ``angle_for_k`` is the just-computed raw estimate (deg).
        The final resolved angles in ``HybridCoarseResult.angles`` may differ
        from the per-interface estimates after the stack-wide A–P / continuity
        pass — callers wanting the final angles should read them off the return.
    :return: a ``HybridCoarseResult``.
    """
    sk = dict(search_kwargs or {})
    ak = dict(ap_kwargs or {})
    n = len(coords_list)
    estimates: List[RotationEstimate] = []
    ap_angles: List[Optional[float]] = []

    for k in range(n - 1):
        ref = _face(coords_list[k], "top", z_band_fraction)
        mov = _face(coords_list[k + 1], "bottom", z_band_fraction)
        rho = _endpoint_rho(ref) if len(ref) >= 2 else 1.0
        est = global_rotation_search(ref, mov, rho, **sk)
        estimates.append(est)
        if section_images is not None:
            ap_angles.append(
                ap_rotation_hint(section_images[k], section_images[k + 1], **ak)
            )
        else:
            ap_angles.append(None)
        if progress is not None:
            progress(k, n - 1, float(est.angle))

    records = resolve_stack_rotations(estimates, ap_angles)
    _continuity_resolve(records, continuity_tol)
    return HybridCoarseResult(
        angles=[r.angle for r in records], records=records, estimates=estimates
    )
