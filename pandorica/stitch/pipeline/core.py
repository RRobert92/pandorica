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
Registration core (faithful, no-landmark automation).

Wires the stages into an end-to-end **automatic, no-landmark, whirlpool-safe**
serial-section stitcher, operating on spatial graphs
(``[N, 4]`` ``[id, x, y, z]`` per section):

    load → coarse (spline-PCA + oriented ICP) → match (ρ-scaled + vMF + outlier reject)
         → relative rigid fit → guarded TPS warp → per-interface QC
         → global pose refinement (with priors)

For each consecutive interface it matches the **top face of section n to the
bottom face of section n+1**, fits the relative rigid transform, estimates a
foldover-guarded residual warp, and certifies the result. Absolute section poses
come from the global solve (gauge-anchored at section 0). Volume warping is a
separate step (``applier.warp_volume_slicewise``); this module produces the
transforms + QC.

**Input requirement / convention (Z-up):** all tomograms are assumed *correctly
flipped* so that increasing Z is "up" and the stack advances with section number.
Under that convention a section's **top face = high-Z** and **bottom face =
low-Z**, and the physically-adjacent pair across the gap between section n and
n+1 is ``top(n) ↔ bottom(n+1)``. (Empirically validated on sec09-13: this pairing
gives the best cross-gap MT-tangent continuity. NB the underlying
``extract_boundary_endpoints`` uses the *opposite* labels — its ``"bottom"`` is
high-Z — so the mapping is handled in ``_face`` below.)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from pandorica.stitch.transform.scale import boundary_landmarks
from pandorica.stitch.coarse.coarse import coarse_align
from pandorica.stitch.matching.matcher import match_sections
from pandorica.stitch.transform.warp import (
    fit_guarded_warp,
    GuardedWarp,
)
from pandorica.stitch.pipeline.qc import (
    assess_interface,
    InterfaceQC,
    tangent_discontinuity_deg,
)
from pandorica.stitch.transform.solver import (
    Pose,
    IDENTITY,
    apply_pose,
    invert_pose,
    global_pose_refine,
)
from pandorica.stitch.matching.mt_transform import (
    fit_rigid_transform_2d,
)


@dataclass
class InterfaceResult:
    """Per-interface outcome (section k → k+1).

    ``id_pairs`` carries the surviving MT correspondences for this interface as
    ``(ref_mt_id, mov_mt_id, cost)`` — the input by which the downstream chain
    builder joins per-section spline IDs into single cross-section filaments.
    Empty when the interface failed or was rejected.
    """

    coarse: Pose
    relative: Pose
    warp: GuardedWarp
    qc: InterfaceQC
    confidence: dict
    id_pairs: List[Tuple[int, int, float]] = field(default_factory=list)


@dataclass
class StitchResult:
    """Full registration-core result over a section stack."""

    poses: List[Pose]  # absolute per-section poses (section 0 = identity)
    interfaces: List[InterfaceResult]
    accepted: bool = field(default=False)


# Intuitive Z-up face names → extract_boundary_endpoints' (backwards) labels.
# "top" = high-Z face, "bottom" = low-Z face (see module docstring requirement).
_FACE_LABEL = {"top": "bottom", "bottom": "top"}


def _face(coords, face: str, z_band_fraction: float):
    """Boundary endpoints of the intuitive ``face`` ('top'=high-Z, 'bottom'=low-Z)."""
    return boundary_landmarks(coords, _FACE_LABEL[face], z_band_fraction)


def _xy(endpoints) -> np.ndarray:
    return (
        np.array([e["pos"][:2] for e in endpoints]) if endpoints else np.empty((0, 2))
    )


def _apply_pose_to_endpoints(endpoints, pose: Pose):
    """Copy endpoints with pos (full transform) and dir (rotation only) posed."""
    out = []
    for e in endpoints:
        pos = e["pos"].copy()
        d = e["dir"].copy()
        pos[:2] = apply_pose(pose, pos[:2][None, :])[0]
        d[:2] = apply_pose(
            {**pose, "Tx": 0.0, "Ty": 0.0, "Scale": 1.0}, d[:2][None, :]
        )[0]
        out.append({**e, "pos": pos, "dir": d})
    return out


def _endpoint_rho(endpoints) -> float:
    """
    Median nearest-neighbour spacing of boundary *endpoints* (not section points).

    Matching operates on one endpoint per MT, so the relevant scale is the
    spacing *between* endpoints — typically thousands of Å — not the dense
    along-spline point spacing (~100 Å) that the full-section ρ would give. Using
    the wrong scale shrinks the matcher's distance gate by ~20× and starves it of
    matches.
    """
    xy = _xy(endpoints)
    if len(xy) < 2:
        return 1.0
    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    d, _ = nn.kneighbors(xy)
    return float(np.median(d[:, 1]))


def _dirs_for(xy: np.ndarray, endpoints) -> np.ndarray:
    """Look up the tangent of each matched point by exact-nearest endpoint."""
    if len(xy) == 0 or not endpoints:
        return np.empty((0, 2))
    pos = np.array([e["pos"][:2] for e in endpoints])
    dirs = np.array([e["dir"][:2] for e in endpoints])
    _, idx = NearestNeighbors(n_neighbors=1).fit(pos).kneighbors(xy)
    return dirs[idx[:, 0]]


def _centroid_rotation_seed(angle, mov_eps) -> Pose:
    """A seed pose that rotates mov endpoints by ``angle`` about their centroid.

    Used to inject an image-derived coarse rotation: rotating in place keeps the
    cloud near the reference so the MT matcher's distance gate still engages, then
    matching recovers the residual translation.
    """
    c = _xy(mov_eps).mean(0)
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    t = c - R @ c
    return {"Angle": float(angle), "Tx": float(t[0]), "Ty": float(t[1]), "Scale": 1.0}


def _evaluate_seed(seed, ref_eps, mov_eps, rho, allow_scale, match_kwargs):
    """
    Match under a coarse ``seed`` and return its quality + relative transform.

    Returns a dict with ``score`` (post-rigid residual + MT-tangent break),
    ``residual`` (ρ), ``match_fraction``, the matched points, and the recovered
    relative rigid ``rel`` (mov-local → ref-local). ``None`` if too few matches.
    """
    mov_c = _apply_pose_to_endpoints(mov_eps, seed)
    _, rx, mxc, cf, id_pairs = match_sections(ref_eps, mov_c, rho, **match_kwargs)
    if len(rx) < 3:
        return None
    b_local = apply_pose(invert_pose(seed), mxc)
    ang, tx, ty, sc = fit_rigid_transform_2d(rx, b_local, allow_scale=allow_scale)
    rel = {"Angle": ang, "Tx": tx, "Ty": ty, "Scale": sc}
    residual = float(
        np.sqrt(((apply_pose(rel, b_local) - rx) ** 2).sum(1).mean()) / rho
    )
    rot = {"Angle": ang, "Tx": 0.0, "Ty": 0.0, "Scale": 1.0}
    tang = tangent_discontinuity_deg(
        _dirs_for(rx, ref_eps), apply_pose(rot, _dirs_for(mxc, mov_c))
    )
    return {
        "score": residual + tang / 45.0,
        "residual": residual,
        "match_fraction": cf["match_fraction"],
        "seed": seed,
        "mov_c": mov_c,
        "rx": rx,
        "mxc": mxc,
        "cf": cf,
        "rel": rel,
        "id_pairs": id_pairs,
    }


def _failed_interface(reason: str) -> InterfaceResult:
    from pandorica.stitch.transform.diagnostics import FieldCertificate

    cert = FieldCertificate(0.0, np.inf, 0.0, 0.05, 1.0, passed=False)
    qc = InterfaceQC(False, False, 0.0, np.inf, 0.0, reasons=[reason])
    return InterfaceResult(
        dict(IDENTITY), dict(IDENTITY), GuardedWarp(cert, 0.0, False, None), qc, {}, []
    )


def register_section_stack(
    coords_list,
    allow_scale: bool = False,
    lambda_scale: float = 1.0,
    lambda_smooth: float = 0.0,
    z_band_fraction: float = 0.15,
    coarse_angles=None,
    warp_eps: float = 0.05,
    warp_omega_max: float = 1.0,
    warp_grid_n: int = 48,
    warp_pad: float = 0.1,
    qc_min_match_fraction: float = 0.3,
    qc_max_shift_incoherence_rho: float = 2.5,
    qc_max_tangent_deg: float = 20.0,
    progress: Optional[Callable[[int, int], None]] = None,
    **match_kwargs,
) -> StitchResult:
    """
    Run the registration core over a list of section spatial graphs.

    :param coords_list: list of ``[N, 4]`` ``[id, x, y, z]`` section graphs, in
        stack order.
    :param allow_scale / lambda_scale / lambda_smooth: passed to the global solve.
        ``lambda_smooth`` defaults to 0 (data-driven): a non-zero value smooths
        the pose trajectory and can suppress a *genuine* large inter-section
        rotation, so only raise it for interfaces independently confirmed noisy
        (e.g. by an image cross-check), not to tidy a tilt that is actually real.
    :param z_band_fraction: boundary-band fraction for endpoint extraction.
    :param coarse_angles: optional per-interface image-derived coarse rotations
        (deg, length ``len(coords_list)-1``; see ``image_coarse``). When given for
        an interface, the matcher is seeded with that rotation instead of the
        point-cloud coarse — essential for large/ambiguous rotations the MT
        endpoints cannot resolve from a cold start. ``None`` entries fall back to
        the point-cloud policy.
    :param progress: optional ``(k, n_interfaces)`` callback fired at the start of
        each interface's solve, so a long stack can stream live progress.
    :param match_kwargs: forwarded to ``matcher.match_sections`` (e.g. thresholds).
    :return: a ``StitchResult`` with absolute poses, per-interface records, and an
        overall accept flag (all interfaces certified).
    """
    n = len(coords_list)
    if n < 2:
        return StitchResult(
            poses=[dict(IDENTITY)] * max(n, 1), interfaces=[], accepted=n <= 1
        )

    solver_interfaces = []
    weights: List[float] = []
    results: List[InterfaceResult] = []

    for k in range(n - 1):
        if progress is not None:
            progress(k, n - 1)
        # Convention: top face of section k ↔ bottom face of section k+1.
        ref_eps = _face(coords_list[k], "top", z_band_fraction)
        mov_eps = _face(coords_list[k + 1], "bottom", z_band_fraction)
        if len(ref_eps) < 2 or len(mov_eps) < 2:
            results.append(_failed_interface("too few boundary endpoints"))
            solver_interfaces.append((np.empty((0, 2)), np.empty((0, 2))))
            weights.append(0.0)
            continue

        rho_k = _endpoint_rho(ref_eps)
        # The effective distance gate is now clipped to physical (Å) bounds —
        # see ``matcher._resolve_max_dist``. No pipeline-level override is
        # needed; the matcher's defaults are calibrated for MT physiology.

        img_angle = coarse_angles[k] if coarse_angles is not None else None
        if img_angle is not None:
            # Image-derived coarse rotation: the reliable arbiter for large /
            # rotationally-ambiguous interfaces the MT endpoints can't resolve cold.
            # Seed with it; MT matching refines translation. Fall back to identity
            # only if the image seed fails to match.
            seed = _centroid_rotation_seed(img_angle, mov_eps)
            best = _evaluate_seed(
                seed, ref_eps, mov_eps, rho_k, allow_scale, match_kwargs
            )
            if best is None or best["match_fraction"] < 0.3:
                ident = _evaluate_seed(
                    dict(IDENTITY), ref_eps, mov_eps, rho_k, allow_scale, match_kwargs
                )
                if ident and (
                    best is None or ident["match_fraction"] > best["match_fraction"]
                ):
                    best = ident
        else:
            # No image hint: PREFER the identity seed (no coarse). For
            # already-registered data it is correct, and PCA-aligning two
            # *different* endpoint clouds can impose a spurious centroid shift that
            # games the residual. Fall back to PCA only when identity is inadequate
            # — but PCA can't resolve large rotations either; supply coarse_angles.
            ident = _evaluate_seed(
                dict(IDENTITY), ref_eps, mov_eps, rho_k, allow_scale, match_kwargs
            )
            if ident and ident["match_fraction"] >= 0.3 and ident["residual"] <= 1.5:
                best = ident
            else:
                pca = coarse_align(_xy(ref_eps), _xy(mov_eps), allow_scale=allow_scale)
                pca_seed = {kk: pca[kk] for kk in ("Angle", "Tx", "Ty", "Scale")}
                cand = [
                    r
                    for r in (
                        ident,
                        _evaluate_seed(
                            pca_seed, ref_eps, mov_eps, rho_k, allow_scale, match_kwargs
                        ),
                    )
                    if r
                ]
                best = min(cand, key=lambda r: r["score"]) if cand else None

        if best is None:
            results.append(_failed_interface("too few matches"))
            solver_interfaces.append((np.empty((0, 2)), np.empty((0, 2))))
            weights.append(0.0)
            continue

        coarse_pose = best["seed"]
        mov_eps_c = best["mov_c"]
        ref_xy, mov_xy_c, conf, rel = best["rx"], best["mxc"], best["cf"], best["rel"]
        if len(ref_xy) < 2:
            results.append(_failed_interface("too few matches"))
            solver_interfaces.append((np.empty((0, 2)), np.empty((0, 2))))
            weights.append(0.0)
            continue

        # Matched moving points back to section k+1 local coordinates.
        B = apply_pose(invert_pose(coarse_pose), mov_xy_c)
        A = ref_xy
        solver_interfaces.append((A, B))
        # Confidence weight: a low-match / incoherent interface should not impose
        # its (possibly ambiguous) transform as strongly in the global solve.
        weights.append(conf.get("match_fraction", 0.0) ** 2)

        # Residual warp on top of the selected relative rigid.
        warp = fit_guarded_warp(
            apply_pose(rel, B),
            A,
            rho=rho_k,
            eps=warp_eps,
            omega_max=warp_omega_max,
            grid_n=warp_grid_n,
            pad=warp_pad,
        )

        ref_dirs = _dirs_for(ref_xy, ref_eps)
        mov_dirs = _dirs_for(mov_xy_c, mov_eps_c)
        rot_only = {"Angle": rel["Angle"], "Tx": 0.0, "Ty": 0.0, "Scale": 1.0}
        mov_dirs_aligned = apply_pose(rot_only, mov_dirs) if len(mov_dirs) else mov_dirs

        qc = assess_interface(
            warp.certificate,
            conf,
            ref_dirs,
            mov_dirs_aligned,
            min_match_fraction=qc_min_match_fraction,
            max_shift_incoherence_rho=qc_max_shift_incoherence_rho,
            max_tangent_deg=qc_max_tangent_deg,
        )
        results.append(
            InterfaceResult(coarse_pose, rel, warp, qc, conf, best.get("id_pairs", []))
        )

    poses = global_pose_refine(
        solver_interfaces,
        allow_scale=allow_scale,
        lambda_scale=lambda_scale,
        lambda_smooth=lambda_smooth,
        weights=weights,
    )
    accepted = all(r.qc.accepted for r in results)
    return StitchResult(poses=poses, interfaces=results, accepted=accepted)
