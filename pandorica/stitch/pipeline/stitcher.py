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
from typing import Callable, List, Optional, Sequence

from pandorica.stitch.coarse.coarse_hybrid import (
    hybrid_coarse,
    HybridCoarseResult,
)
from pandorica.stitch.pipeline.core import (
    register_section_stack,
    register_warps_to_coarse,
    rescue_coarse_poses,
    StitchResult,
)
from pandorica.stitch.pipeline.intensity_qc import (
    verify_rotation,
    IntensityVerification,
)
from pandorica.stitch.transform.solver import (
    Pose,
    compose_poses,
    invert_pose,
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
    # MT rotation rescues (image-coarse path only): per entry
    # ``(k, image_angle, mt_angle, image_match, mt_match)``.
    rescues: List = field(default_factory=list)


def stitch_sections(
    coords_list,
    section_images: Optional[Sequence] = None,
    coarse_poses: Optional[List[Pose]] = None,
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
    progress: Optional[Callable[[str, int, int], None]] = None,
    **match_kwargs,
) -> SerialStitchResult:
    """
    Run the stitch over a serial-section stack.

    :param coords_list: per-section spatial graphs ``[N, 4]`` in stack order.
    :param section_images: optional one 2-D boundary-face image per section, used
        for the A–P-polarity sign hint AND the dense-intensity verification.
    :param coarse_poses: optional absolute per-section image-coarse poses (length
        ``n``). When given, this is the **coarse→fine default**: the global pose
        (translation + rotation + anisotropic scale) is taken from the image and the
        MTs only fit the fine residual warp relative to it (no MT pose-solve, no
        ``hybrid_coarse``). ``None`` (default) runs the legacy MT-pose path.
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
    :param progress: optional ``(phase, k, n_interfaces)`` callback, fired once per
        interface as it is solved. ``phase`` is ``"coarse"`` (rotation search) then
        ``"register"`` (match + warp); lets a caller stream live progress for a long
        stack instead of going silent until the whole solve returns.
    :param z_band_fraction / allow_scale / lambda_scale / lambda_smooth /
        match_kwargs: forwarded to the coarse + base pipeline.
    :return: a ``SerialStitchResult``. ``accepted`` requires the base pipeline to accept,
        no flagged coarse interface, AND every significant-rotation interface to be
        intensity-verified (when images are supplied).
    """
    rescues: List = []
    if coarse_poses is not None:
        # Image-coarse mode (the coarse→fine default): the global pose — translation,
        # rotation, anisotropic scale — is already fixed by the image and applied to
        # the MTs upstream. Skip the MT point-cloud coarse and the MT pose-solve; the
        # MTs only drive the fine residual warp here (register_warps_to_coarse). The
        # "coarse rotation" we report per interface is just the relative image-pose
        # angle, so interface_rows / continuity logging keep working.
        warp_kw = dict(
            z_band_fraction=z_band_fraction,
            warp_eps=warp_eps,
            warp_omega_max=warp_omega_max,
            warp_grid_n=warp_grid_n,
            warp_pad=warp_pad,
            qc_min_match_fraction=qc_min_match_fraction,
            qc_max_shift_incoherence_rho=qc_max_shift_incoherence_rho,
            qc_max_tangent_deg=qc_max_tangent_deg,
        )
        base = register_warps_to_coarse(
            coords_list, coarse_poses,
            progress=(lambda k, ntot: progress("warp", k, ntot)) if progress else None,
            **warp_kw, **match_kwargs,
        )
        # MT rotation rescue: on any interface where the image coarse rotation failed
        # (the MT match collapsed below the gate), re-estimate the rotation from the
        # dense MTs and adopt it only when it confidently beats the gate; then re-warp.
        rescue_sk = (
            {"use_cpd": True, "cpd_w": cpd_w, "cpd_seeds": cpd_seeds,
             "cpd_quantile": cpd_quantile}
            if cpd_coarse else None
        )
        # Trigger the rescue on the UN-bootstrapped single-pass match: the warp
        # bootstrap can lift a collapsed interface above the gate, which would hide a
        # grossly wrong rotation from the rescue (and the rescue compares single-pass
        # MT vs single-pass image, so the signals must match).
        coarse_poses, rescues = rescue_coarse_poses(
            coords_list, list(coarse_poses),
            [it.confidence.get("match_fraction_single", it.qc.match_fraction)
             for it in base.interfaces],
            z_band_fraction=z_band_fraction, match_gate=qc_min_match_fraction,
            search_kwargs=rescue_sk, **match_kwargs,
        )
        if rescues:
            base = register_warps_to_coarse(
                coords_list, coarse_poses,
                progress=(
                    (lambda k, ntot: progress("rescue-warp", k, ntot))
                    if progress else None
                ),
                **warp_kw, **match_kwargs,
            )
        rels = [
            compose_poses(invert_pose(coarse_poses[k]), coarse_poses[k + 1])
            for k in range(len(coarse_poses) - 1)
        ]
        hybrid = HybridCoarseResult(
            angles=[float(r["Angle"]) for r in rels], records=[], estimates=[]
        )
    else:
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
            progress=(
                (lambda k, ntot, ang: progress("coarse", k, ntot)) if progress else None
            ),
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
            progress=(
                (lambda k, ntot: progress("register", k, ntot)) if progress else None
            ),
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
        rescues=rescues,
    )
