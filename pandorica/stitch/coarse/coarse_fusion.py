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
Coarse-rotation sign resolution + ABSTAIN.

The global MT-endpoint search (``rotation_search``) gives a precise rotation
*magnitude* per interface, but on bundled/symmetric constellations the **sign**
(the 180° flip) is ambiguous (confirmed on the MaleMeiosis spindle: every
interface's second basin sat at ~180°). The reliable sign authority is the
**image A–P polarity** — a
signed, every-section biological axis — *not* the MT endpoints.

This module fuses the two: it keeps the MT search's precise angle but uses an
optional per-interface **A–P-polarity rotation hint** to pick the correct 180°
branch. When an interface is already confidently resolved by the MT search alone
(high flip-ratio, non-degenerate), it is accepted as-is. When it is ambiguous and
no trustworthy A–P hint resolves it, the interface **ABSTAINS** (is flagged) — it
is never silently mis-signed. (The downstream global solve / operator then
handles flagged interfaces; stack-wide A–P-sign continuity is layered on top.)
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from pandorica.stitch.coarse.rotation_search import RotationEstimate


def _wrap(a: float) -> float:
    """Wrap an angle to (−180°, 180°]."""
    return ((float(a) + 180.0) % 360.0) - 180.0


@dataclass
class ResolvedRotation:
    """A sign-resolved coarse rotation for one interface."""

    angle: float
    source: str  # "confident" | "ap_polarity" | "abstain"
    flagged: bool
    estimate: RotationEstimate


def resolve_interface(
    est: RotationEstimate,
    ap_angle: Optional[float] = None,
    ap_tol_deg: float = 45.0,
) -> ResolvedRotation:
    """
    Resolve one interface's coarse rotation (sign included).

    :param est: the MT global-rotation-search estimate.
    :param ap_angle: optional image-derived A–P-polarity rotation hint (deg,
        signed, mov→ref) — the sign authority for ambiguous interfaces.
    :param ap_tol_deg: max allowed disagreement between the chosen MT branch and
        the A–P hint; beyond this the interface abstains.
    :return: a ``ResolvedRotation``.
    """
    if est.confident:
        return ResolvedRotation(est.angle, "confident", False, est)

    # Ambiguous: the two 180°-flip branches of the (precise) MT magnitude.
    candidates = [_wrap(est.angle), _wrap(est.angle + 180.0)]
    if ap_angle is not None:
        diffs = [abs(_wrap(c - ap_angle)) for c in candidates]
        i = int(np.argmin(diffs))
        if diffs[i] <= ap_tol_deg:
            return ResolvedRotation(candidates[i], "ap_polarity", False, est)

    # No trustworthy resolver → flag rather than guess the sign.
    return ResolvedRotation(est.angle, "abstain", True, est)


def resolve_stack_rotations(
    estimates: Sequence[RotationEstimate],
    ap_angles: Optional[Sequence[Optional[float]]] = None,
    ap_tol_deg: float = 45.0,
) -> List[ResolvedRotation]:
    """
    Resolve coarse rotations for a whole serial-section stack.

    :param estimates: per-interface ``RotationEstimate`` (k → mov section k+1).
    :param ap_angles: optional per-interface A–P-polarity rotation hints (same
        length); ``None`` entries fall back to MT-only / ABSTAIN.
    :return: list of ``ResolvedRotation`` (one per interface).
    """
    aps = list(ap_angles) if ap_angles is not None else [None] * len(estimates)
    if len(aps) != len(estimates):
        raise ValueError("ap_angles must match estimates length")
    return [resolve_interface(e, a, ap_tol_deg) for e, a in zip(estimates, aps)]
