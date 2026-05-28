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
Serial-section stitch orchestrator (pragmatic global).

Integrates the production coarse + verification stages on top of the
registration core's matching / warp / solve machinery (``register_section_stack``):

    hybrid_coarse (MT global rotation search + A–P-polarity sign + stack continuity
    + ABSTAIN)  →  register_section_stack (match → relative rigid → guarded warp → global
    pose solve)  →  dense-intensity verification (splines match, intensity verifies)

vs the bare registration core, this swaps the fragile point-cloud coarse for the
hybrid coarse and
adds the intensity QC layer. The **CPD coarse rotation** (``cpd_coarse=True``) is
decoy-robust and recovers ±90° rotations from a cold start, routing the coarse
stage through the multi-seed CPD search.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from pandorica.stitch.coarse.coarse_hybrid import (
    hybrid_coarse,
    HybridCoarseResult,
)
from pandorica.stitch.pipeline.core import (
    register_section_stack,
    StitchResult,
)
from pandorica.stitch.pipeline.intensity_qc import (
    verify_rotation,
    IntensityVerification,
)

# Below this |rotation| the dense-intensity test is uninformative (identity already
# agrees), so it is not used to gate the interface.
_INTENSITY_MIN_ANGLE_DEG = 15.0


@dataclass
class SerialStitchResult:
    """Stitch result: the base stitch + hybrid-coarse + intensity records."""

    poses: List[dict]
    base: StitchResult
    hybrid: HybridCoarseResult
    intensity: List[Optional[IntensityVerification]]
    accepted: bool = field(default=False)


def stitch_sections(
    coords_list,
    section_images: Optional[Sequence] = None,
    z_band_fraction: float = 0.15,
    allow_scale: bool = False,
    lambda_scale: float = 1.0,
    lambda_smooth: float = 0.0,
    cpd_coarse: bool = True,
    cpd_w: float = 0.4,
    cpd_seeds: int = 12,
    cpd_quantile: float = 0.5,
    continuity_tol: float = 30.0,
    intensity_margin: float = 0.02,
    intensity_min_angle: float = _INTENSITY_MIN_ANGLE_DEG,
    warp_eps: float = 0.05,
    warp_omega_max: float = 1.0,
    warp_grid_n: int = 48,
    warp_pad: float = 0.1,
    qc_min_match_fraction: float = 0.3,
    qc_max_shift_incoherence_rho: float = 2.5,
    qc_max_tangent_deg: float = 20.0,
    **match_kwargs,
) -> SerialStitchResult:
    """
    Run the stitch over a serial-section stack.

    :param coords_list: per-section spatial graphs ``[N, 4]`` in stack order.
    :param section_images: optional one 2-D boundary-face image per section, used
        for the A–P-polarity sign hint AND the dense-intensity verification.
    :param cpd_coarse: use the multi-seed CPD search for the coarse rotation
        (decoy-robust, ±90° cold-start). **Default True** — it dominates the gated
        inlier sweep on realistic data and flags ambiguous interfaces rather than
        mis-stitching. Set ``False`` for the legacy gated sweep.
    :param cpd_w / cpd_seeds / cpd_quantile: CPD outlier weight / # seed angles /
        residual quantile used to rank seeds (only used when ``cpd_coarse``).
    :param continuity_tol: stack-continuity trend tolerance (deg) for resolving
        flagged coarse interfaces.
    :param intensity_margin: how much the rotation must beat the flip/identity in
        the dense-intensity check to verify.
    :param intensity_min_angle: only gate on intensity for ``|angle|`` above this.
    :param warp_* / qc_*: TPS-warp and per-interface QC thresholds, forwarded to the
        base pipeline.
    :param z_band_fraction / allow_scale / lambda_scale / lambda_smooth /
        match_kwargs: forwarded to the coarse + base pipeline.
    :return: a ``SerialStitchResult``. ``accepted`` requires the base pipeline to accept,
        no flagged coarse interface, AND every significant-rotation interface to be
        intensity-verified (when images are supplied).
    """
    search_kwargs = None
    if cpd_coarse:
        search_kwargs = {
            "use_cpd": True,
            "cpd_w": cpd_w,
            "cpd_seeds": cpd_seeds,
            "cpd_quantile": cpd_quantile,
        }
    hybrid = hybrid_coarse(
        coords_list,
        section_images,
        z_band_fraction=z_band_fraction,
        continuity_tol=continuity_tol,
        search_kwargs=search_kwargs,
    )
    base = register_section_stack(
        coords_list,
        coarse_angles=hybrid.angles,
        z_band_fraction=z_band_fraction,
        allow_scale=allow_scale,
        lambda_scale=lambda_scale,
        lambda_smooth=lambda_smooth,
        warp_eps=warp_eps,
        warp_omega_max=warp_omega_max,
        warp_grid_n=warp_grid_n,
        warp_pad=warp_pad,
        qc_min_match_fraction=qc_min_match_fraction,
        qc_max_shift_incoherence_rho=qc_max_shift_incoherence_rho,
        qc_max_tangent_deg=qc_max_tangent_deg,
        **match_kwargs,
    )

    intensity: List[Optional[IntensityVerification]] = []
    intensity_ok = True
    for k in range(len(base.interfaces)):
        v = None
        if section_images is not None and k + 1 < len(section_images):
            v = verify_rotation(
                section_images[k],
                section_images[k + 1],
                hybrid.angles[k],
                margin=intensity_margin,
            )
            # Only gate on intensity for interfaces with a significant rotation,
            # where the test is meaningful (a wrong 90° turn is image-detectable).
            if abs(hybrid.angles[k]) >= intensity_min_angle and not v.verified:
                intensity_ok = False
        intensity.append(v)

    # A flagged coarse interface = the rotation is not trustworthy (ambiguous /
    # unresolved by continuity). Do not silently accept it.
    coarse_ok = not any(rec.flagged for rec in hybrid.records)
    accepted = base.accepted and intensity_ok and coarse_ok
    return SerialStitchResult(
        poses=base.poses,
        base=base,
        hybrid=hybrid,
        intensity=intensity,
        accepted=accepted,
    )
