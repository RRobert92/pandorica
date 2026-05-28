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
Per-interface QC certificate and accept/flag gate.

Each serial-section interface gets a certificate that fuses three independent
signals — none sufficient alone, all cheap:

    * **warp field** — the diffeomorphism certificate (det J ≥ ε AND
      |curl| ≤ Ω_max). The hard whirlpool guard.
    * **matcher confidence** — match fraction and shift coherence from the matcher.
      Too few or incoherent matches means the alignment is untrustworthy even if
      the warp is technically smooth.
    * **MT tangent continuity (ADVISORY)** — reported but not gated: on real data
      it reads high even at the correct rotation (boundary MTs cross the gap
      near-perpendicular, so their in-plane tangent doesn't reliably continue).
      The dense-intensity verification is the image-based check used instead.

The gate is conservative: an interface is **accepted only if every (gating) check
passes**; otherwise it is **flagged** with reasons, never silently stitched.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from pandorica.stitch.transform.diagnostics import FieldCertificate


def tangent_discontinuity_deg(
    ref_dirs: np.ndarray, mov_dirs: np.ndarray, min_xy: float = 0.2
) -> float:
    """
    XY-magnitude-weighted mean angle (deg) between matched tangents, sign-agnostic.

    Both arrays are ``[m, >=2]`` matched *unit* direction vectors in a common
    frame (moving tangents already mapped by the pose). 0° = perfect continuity.

    Boundary MTs that cross the gap run ~perpendicular to the section (steep in Z),
    so their in-plane (XY) tangent is tiny and dominated by tracing noise — the
    near-perpendicular boundary MTs. We therefore **weight each pair by its XY-tangent
    magnitude** (a unit 3-vector's XY norm = how in-plane the MT is): near-vertical
    tangents contribute ~nothing. If even the most in-plane tangent is below
    ``min_xy`` (essentially all MTs are near-vertical), the metric is uninformative
    and returns 0.0 — so it never *misfires* on a specimen whose boundary tangents
    are all steep-in-Z (which spuriously flagged a correct +40° interface before).
    """
    ref = np.asarray(ref_dirs, dtype=float)
    mov = np.asarray(mov_dirs, dtype=float)
    if len(ref) == 0:
        return 0.0
    angles, weights = [], []
    for r, m in zip(ref, mov):
        nr, nm = np.linalg.norm(r[:2]), np.linalg.norm(m[:2])
        if nr < 1e-8 or nm < 1e-8:
            continue
        cos_a = np.clip(np.dot(r[:2], m[:2]) / (nr * nm), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(abs(cos_a))))
        weights.append(nr * nm)  # both ~unit-3D ⇒ XY product in [0,1]
    if not weights or max(weights) < min_xy:
        return 0.0  # no reliably in-plane tangents → uninformative
    return float(np.average(angles, weights=weights))


@dataclass
class InterfaceQC:
    """QC record for one serial-section interface."""

    accepted: bool
    field_passed: bool
    match_fraction: float
    shift_incoherence_rho: float
    tangent_discontinuity_deg: float
    reasons: List[str] = field(default_factory=list)


def assess_interface(
    field_certificate: FieldCertificate,
    confidence: dict,
    ref_dirs: np.ndarray,
    mov_dirs: np.ndarray,
    min_match_fraction: float = 0.3,
    max_shift_incoherence_rho: float = 2.5,
    max_tangent_deg: float = 20.0,
) -> InterfaceQC:
    """
    Fuse warp / matcher / biology signals into an accept-or-flag certificate.

    :param field_certificate: the warp's diffeomorphism certificate.
    :param confidence: matcher confidence dict (``match_fraction``,
        ``shift_incoherence_rho``).
    :param ref_dirs / mov_dirs: matched tangents in a common frame (biology check).
    :param min_match_fraction: minimum acceptable matched fraction.
    :param max_shift_incoherence_rho: maximum acceptable shift incoherence (ρ).
    :param max_tangent_deg: maximum acceptable mean tangent discontinuity (deg).
    :return: an ``InterfaceQC``; ``accepted`` iff every check passes.
    """
    reasons: List[str] = []

    if not field_certificate.passed:
        reasons.append(
            f"warp not a diffeomorphism (min detJ={field_certificate.min_det_j:.3g}, "
            f"max|curl|={field_certificate.max_abs_vorticity:.3g})"
        )

    match_fraction = float(confidence.get("match_fraction", 0.0))
    if match_fraction < min_match_fraction:
        reasons.append(
            f"low match fraction ({match_fraction:.2f} < {min_match_fraction})"
        )

    # shift_incoherence is the spread of matched-pair shifts around a single
    # rigid transform — i.e. the non-rigid residual the guarded warp absorbs.
    # The bound is therefore generous (≈1.5ρ); det J / vorticity / tangent are
    # the gates that catch genuinely bad alignments.
    incoherence = float(confidence.get("shift_incoherence_rho", np.inf))
    if incoherence > max_shift_incoherence_rho:
        reasons.append(
            f"incoherent shifts ({incoherence:.2g}ρ > {max_shift_incoherence_rho}ρ)"
        )

    # MT tangent continuity is ADVISORY only, not a gate. On real data (FemalePN
    # sec12→13, MaleMeiosis sec4→5) it reads high (80°, 40°) at the *correct*
    # rotation, because boundary MTs cross the gap near-perpendicular and their
    # in-plane tangent direction does not reliably continue.
    # It is reported for inspection but does not flag the interface; the dense-
    # intensity verification is the image-based check that replaces it.
    tangent = tangent_discontinuity_deg(ref_dirs, mov_dirs)
    _ = max_tangent_deg  # retained for API/back-compat; intentionally not gated

    return InterfaceQC(
        accepted=len(reasons) == 0,
        field_passed=field_certificate.passed,
        match_fraction=match_fraction,
        shift_incoherence_rho=incoherence,
        tangent_discontinuity_deg=tangent,
        reasons=reasons,
    )
