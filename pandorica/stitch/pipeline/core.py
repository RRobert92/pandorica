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
    compose_poses,
    global_pose_refine,
    make_pose,
    linear_part,
    pose_from_matrix,
)
from pandorica.stitch.matching.mt_transform import (
    fit_rigid_transform_2d,
)
from pandorica.stitch.coarse.rotation_search import global_rotation_search


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
    return make_pose(float(angle), float(t[0]), float(t[1]))


def _evaluate_seed(seed, ref_eps, mov_eps, rho, allow_scale, match_kwargs, fit=True):
    """
    Match under a coarse ``seed`` and return its quality + relative transform.

    Returns a dict with ``score`` (post-rigid residual + MT-tangent break),
    ``residual`` (ρ), ``match_fraction``, the matched points, and the recovered
    relative rigid ``rel`` (mov-local → ref-local). ``None`` if too few matches.

    ``fit`` (default ``True``) re-fits a rigid/similarity ``rel`` from the MT
    correspondences. Pass ``fit=False`` to keep the supplied ``seed`` *as* the
    committed relative pose — used by the image-coarse path, where ``seed`` is
    the per-interface image pose (translation + rotation + anisotropic scale in
    its L-matrix) and the MT residual belongs in the warp, not in a re-fit rigid
    (fitting one from MT endpoints overfits the spatially-varying baking field —
    see project_coarse_fine_architecture).
    """
    mov_c = _apply_pose_to_endpoints(mov_eps, seed)
    _, rx, mxc, cf, id_pairs = match_sections(ref_eps, mov_c, rho, **match_kwargs)
    if len(rx) < 3:
        return None
    b_local = apply_pose(invert_pose(seed), mxc)
    if fit:
        ang, tx, ty, sc = fit_rigid_transform_2d(rx, b_local, allow_scale=allow_scale)
        rel = make_pose(ang, tx, ty, sc)
    else:
        rel = dict(seed)
    residual = float(
        np.sqrt(((apply_pose(rel, b_local) - rx) ** 2).sum(1).mean()) / rho
    )
    rot = make_pose(rel["Angle"])
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
    qc = InterfaceQC(False, False, False, 0.0, np.inf, 0.0, reasons=[reason])
    return InterfaceResult(
        dict(IDENTITY), dict(IDENTITY), GuardedWarp(cert, 0.0, False, None), qc, {}, []
    )


def _warpless_interface(rel: Pose) -> InterfaceResult:
    """An interface the image coarse aligns but the MTs don't match across.

    Unlike :func:`_failed_interface` this is **not** a failure: the global pose is
    already set by the image coarse, so the interface's QC passes (``accepted``);
    there is simply no MT residual warp here, and the image-fill stage covers the
    fine deformation. The warp is left un-accepted (``_rbf`` ``None`` → zero
    displacement) so the exporter ignores it and uses the image-fill warp instead.
    """
    from pandorica.stitch.transform.diagnostics import FieldCertificate

    cert = FieldCertificate(1.0, 0.0, 0.0, 0.05, 1.0, passed=False)
    qc = InterfaceQC(True, False, False, 0.0, 0.0, 0.0,
                     reasons=["image-coarse only (no MT match)"])
    return InterfaceResult(
        dict(rel), dict(rel), GuardedWarp(cert, 0.0, False, None), qc, {}, []
    )


# Near-vertical stubs (|in-plane tangent| below this) have no usable in-plane direction, so
# split_chains cannot judge whether they continue — it judges by direction. A wrong such match
# is two different MTs joined by a lateral jog. We catch those by POSITION instead.
_CUT_VERTICAL_MIN_XY = 0.2


def _cut_vertical_jog(id_pairs, mov_xy, ref_xy, mov_dirs, ref_dirs, rho, jog_rho, k=8):
    """Drop near-vertical matched pairs whose coarse displacement DISAGREES with its local
    neighbourhood by more than ``jog_rho`` × ρ — the split-blind wrong matches.

    A wrong match (two different MTs joined) is a LOCAL OUTLIER: its coarse displacement
    ``mov − ref`` deviates from the smooth field its neighbours follow. The *absolute* jog does
    NOT separate wrong from right — a true pair in a deformed region also has a large coarse jog
    (and the warp bootstrap recovers it), so cutting on absolute jog also cuts good pairs (even
    on healthy stacks). Only the deviation from the LOCAL consensus is specific to a bad match.
    Restricted to NEAR-VERTICAL pairs: a reliable in-plane tangent would let the direction-based
    chain split judge the pair instead. Conservative; ``jog_rho <= 0`` disables it.
    """
    n = len(id_pairs)
    if jog_rho <= 0 or n < k + 2 or len(mov_xy) != n or len(ref_xy) != n:
        return id_pairs
    ref = np.asarray(ref_xy, float)
    disp = (np.asarray(mov_xy, float) - ref) / max(rho, 1e-9)
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(ref).kneighbors(ref)
    resid = np.array([
        np.linalg.norm(disp[i] - np.median(disp[idx[i, 1:]], axis=0)) for i in range(n)
    ])
    sv = (np.linalg.norm(np.asarray(mov_dirs, float), axis=1)
          if len(mov_dirs) == n else np.ones(n))
    dv = (np.linalg.norm(np.asarray(ref_dirs, float), axis=1)
          if len(ref_dirs) == n else np.ones(n))
    drop = ((sv < _CUT_VERTICAL_MIN_XY) | (dv < _CUT_VERTICAL_MIN_XY)) & (resid > jog_rho)
    return [ip for ip, d in zip(id_pairs, drop) if not d]


def register_section_stack(
    coords_list,
    allow_scale: bool = False,
    lambda_scale: float = 1.0,
    lambda_smooth: float = 0.0,
    z_band_fraction: float = 0.15,
    coarse_angles=None,
    warp_eps: float = 0.05,
    warp_omega_max: float = 1.0,
    warp_tangent_weight: float = 0.0,
    cut_vertical_jog_rho: float = 2.0,
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
                pca_seed = make_pose(pca["Angle"], pca["Tx"], pca["Ty"], pca["Scale"])
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

        ref_dirs = _dirs_for(ref_xy, ref_eps)
        mov_dirs = _dirs_for(mov_xy_c, mov_eps_c)
        rot_only = make_pose(rel["Angle"])
        mov_dirs_aligned = apply_pose(rot_only, mov_dirs) if len(mov_dirs) else mov_dirs

        # Residual warp on top of the selected relative rigid (with the gentle, guarded
        # tangent-continuity term on reliable shallow-MT stubs).
        warp = fit_guarded_warp(
            apply_pose(rel, B),
            A,
            rho=rho_k,
            eps=warp_eps,
            omega_max=warp_omega_max,
            grid_n=warp_grid_n,
            pad=warp_pad,
            src_tan=mov_dirs_aligned,
            dst_tan=ref_dirs,
            tangent_weight=warp_tangent_weight,
        )

        qc = assess_interface(
            warp.certificate,
            conf,
            ref_dirs,
            mov_dirs_aligned,
            min_match_fraction=qc_min_match_fraction,
            max_shift_incoherence_rho=qc_max_shift_incoherence_rho,
            max_tangent_deg=qc_max_tangent_deg,
        )
        cut_pairs = _cut_vertical_jog(
            best.get("id_pairs", []), apply_pose(rel, B), A,
            mov_dirs_aligned, ref_dirs, rho_k, cut_vertical_jog_rho,
        )
        results.append(
            InterfaceResult(coarse_pose, rel, warp, qc, conf, cut_pairs)
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


_BOOTSTRAP_MAX_PASSES = 6
_BOOTSTRAP_MIN_GAIN = 0.01


def _apply_warp_to_eps(endpoints, warp):
    """Copy boundary endpoints with their xy displaced by a fitted warp (id/dir kept)."""
    out = []
    for e in endpoints:
        pos = e["pos"].copy()
        pos[:2] = pos[:2] + warp.displacement(pos[:2][None, :])[0]
        out.append({**e, "pos": pos})
    return out


def _bootstrap_correspondences(ref_eps, mov_c0, rho, match_kwargs, warp_kw):
    """Discover the converged cross-gap MT correspondences by iteratively pre-warping
    the moving endpoints, returning the best ``(ref_xy, conf, id_pairs)`` seen.

    When the coarse pose leaves a spatially-varying residual, a single rigid-frame
    match drops the displaced-but-correct pairs: the matcher's rigid-residual and
    smoothness gates are tuned for tight (<1ρ) co-location, so partners sitting at
    2–3ρ are rejected even though they are real continuations. Each pass fits the
    guarded TPS warp on the matches it *did* find and applies it, which pulls the
    remaining true partners back toward <1ρ so the SAME tight gates now accept them —
    recovering both the chain and the warp's support **without loosening any gate**
    (a false neighbour is not pulled coherently by a smooth field, so it still fails).

    The loop stops when the match stops improving (so a tight coarse pose costs a
    single pass — a no-op), when the guarded warp can find no safe field, or at the
    pass cap. The caller re-fits one guard-safe warp from these pairs in the ORIGINAL
    coarse frame, so the export still carries a single field per interface.

    Returns ``(ref_xy, conf, id_pairs, first_frac)`` — the best match seen plus the
    UN-bootstrapped first-pass fraction, which the caller keeps so the rotation rescue
    still detects a collapse the bootstrap would otherwise mask above its gate.
    """
    movc = [dict(e) for e in mov_c0]
    best = (np.empty((0, 2)), None, [])
    best_frac, prev, first_frac = -1.0, -1.0, 0.0
    for i in range(_BOOTSTRAP_MAX_PASSES):
        _, rx, mxc, conf, idp = match_sections(ref_eps, movc, rho, **match_kwargs)
        frac = conf.get("match_fraction", 0.0) if conf else 0.0
        if i == 0:
            first_frac = frac
        if frac > best_frac:
            best, best_frac = (rx, conf, idp), frac
        if len(rx) < 4 or frac - prev < _BOOTSTRAP_MIN_GAIN:
            break
        prev = frac
        w = fit_guarded_warp(mxc, rx, rho=rho, **warp_kw)
        if not w.accepted:
            break
        movc = _apply_warp_to_eps(movc, w)
    return (*best, first_frac)


def register_warps_to_coarse(
    coords_list,
    coarse_poses: List[Pose],
    z_band_fraction: float = 0.15,
    warp_eps: float = 0.05,
    warp_omega_max: float = 1.0,
    warp_grid_n: int = 48,
    warp_pad: float = 0.1,
    qc_min_match_fraction: float = 0.3,
    qc_max_shift_incoherence_rho: float = 2.5,
    qc_max_tangent_deg: float = 20.0,
    warp_tangent_weight: float = 0.0,
    cut_vertical_jog_rho: float = 2.0,
    progress: Optional[Callable[[int, int], None]] = None,
    **match_kwargs,
) -> StitchResult:
    """
    Fit the per-interface MT residual warp RELATIVE to a supplied coarse pose chain.

    This is the *fine* half of the coarse→fine pipeline (see
    project_coarse_fine_architecture). The image coarse
    (:func:`.image_pose.image_only_poses`) already fixes every section's **global**
    pose — translation + rotation + anisotropic scale, carried in the L-matrix —
    and that one pose is applied to BOTH the volume and the microtubule graph. This
    stage adds only the spatially-varying remainder: for each gap it matches the
    boundary MTs *in the already-coarse-aligned frame* and fits a foldover-guarded
    TPS warp on the leftover (the baking deformation a single global affine cannot
    represent).

    Crucially, **no rigid/affine is re-fit from the MT correspondences** — fitting
    one would force the spatially-varying field into a global transform and overfit
    the anisotropy (the reverted MT-global-affine mistake). The committed relative
    pose IS the image coarse, so the absolute poses returned are exactly
    ``coarse_poses``; the matched moving MTs land in the reference frame through that
    pose and the warp captures their residual displacement onto the reference MTs.

    :param coords_list: per-section spatial graphs ``[N, 4]`` ``[id, x, y, z]``.
    :param coarse_poses: absolute per-section image-coarse poses (length ``n``;
        section 0 = gauge). Applied to image and MTs alike upstream.
    :param warp_* / qc_*: TPS-warp and per-interface QC thresholds (as
        :func:`register_section_stack`).
    :param progress: optional ``(k, n_interfaces)`` callback fired per interface.
    :param match_kwargs: forwarded to ``matcher.match_sections``.
    :return: a ``StitchResult`` whose ``poses`` are ``coarse_poses`` unchanged and
        whose per-interface ``warp`` holds the MT residual. MT-free interfaces get a
        warp-less (but pose-accepted) record — the image-fill stage covers them.
    """
    n = len(coords_list)
    if n < 2:
        base = [dict(p) for p in coarse_poses] if coarse_poses else [dict(IDENTITY)]
        return StitchResult(poses=base, interfaces=[], accepted=n <= 1)

    rels = [
        compose_poses(invert_pose(coarse_poses[k]), coarse_poses[k + 1])
        for k in range(n - 1)
    ]
    results: List[InterfaceResult] = []
    for k in range(n - 1):
        if progress is not None:
            progress(k, n - 1)
        rel = rels[k]
        ref_eps = _face(coords_list[k], "top", z_band_fraction)
        mov_eps = _face(coords_list[k + 1], "bottom", z_band_fraction)
        if len(ref_eps) < 2 or len(mov_eps) < 2:
            results.append(_warpless_interface(rel))
            continue

        rho_k = _endpoint_rho(ref_eps)
        mov_c0 = _apply_pose_to_endpoints(mov_eps, rel)
        warp_kw = dict(eps=warp_eps, omega_max=warp_omega_max,
                       grid_n=warp_grid_n, pad=warp_pad)
        # Match in the already-coarse-aligned frame, bootstrapping the correspondences
        # so a spatially-varying residual doesn't cost real cross-gap pairs (see
        # _bootstrap_correspondences). A tight coarse pose converges in one pass.
        ref_xy, conf, id_pairs, first_frac = _bootstrap_correspondences(
            ref_eps, mov_c0, rho_k, match_kwargs, warp_kw
        )
        if conf is None or len(ref_xy) < 2:
            results.append(_warpless_interface(rel))
            continue

        # Re-fit ONE guard-safe warp from the converged pairs in the ORIGINAL coarse
        # frame (map each match back to its rel-posed moving position by MT id), so the
        # export carries a single field per interface — the bootstrap only DISCOVERS
        # the pairs; the warp and QC are then computed on the true, un-pre-warped
        # residual. ref endpoints are never pre-warped, so ref_xy is already correct.
        id_to_xy = {int(e["id"]): np.asarray(e["pos"][:2], float) for e in mov_c0}
        mov_xy_c = np.array([id_to_xy[int(ip[1])] for ip in id_pairs])
        # In the pre-warped frame the matched shift is ~0 (it would trivially pass the
        # incoherence gate); recompute it on the real residual instead.
        shifts = ref_xy - mov_xy_c
        conf = {
            **conf,
            "match_fraction_single": float(first_frac),
            "shift_incoherence_rho": (
                float(np.linalg.norm(np.std(shifts, axis=0)) / rho_k)
                if len(ref_xy) > 1 else np.inf
            ),
        }
        ref_dirs = _dirs_for(ref_xy, ref_eps)
        mov_dirs = _dirs_for(mov_xy_c, mov_c0)
        rot_only = make_pose(rel["Angle"])
        mov_dirs_aligned = apply_pose(rot_only, mov_dirs) if len(mov_dirs) else mov_dirs

        # The tangent-continuity term (gentle, guarded) nudges matched shallow-MT stubs toward
        # continuity across the seam; near-vertical stubs (unreliable in-plane tangent) are
        # skipped inside fit_guarded_warp, and the vorticity/detJ guard stays authoritative.
        warp = fit_guarded_warp(
            mov_xy_c, ref_xy, rho=rho_k,
            src_tan=mov_dirs_aligned, dst_tan=ref_dirs,
            tangent_weight=warp_tangent_weight, **warp_kw,
        )

        qc = assess_interface(
            warp.certificate,
            conf,
            ref_dirs,
            mov_dirs_aligned,
            min_match_fraction=qc_min_match_fraction,
            max_shift_incoherence_rho=qc_max_shift_incoherence_rho,
            max_tangent_deg=qc_max_tangent_deg,
        )
        # Cut split-blind wrong matches: near-vertical pairs with a large residual lateral jog
        # (split_chains can't judge them by direction). Conservative — only the genuine outliers.
        id_pairs = _cut_vertical_jog(
            id_pairs, mov_xy_c, ref_xy, mov_dirs_aligned, ref_dirs, rho_k,
            cut_vertical_jog_rho,
        )
        results.append(
            InterfaceResult(dict(rel), dict(rel), warp, qc, conf, id_pairs)
        )

    accepted = all(r.qc.accepted for r in results)
    return StitchResult(
        poses=[dict(p) for p in coarse_poses], interfaces=results, accepted=accepted
    )


# Anisotropy recovery in the MT rescue. When the image coarse FAILED on an interface (so it
# never fit that interface's per-axis stretch — unlike every interface where the image
# committed aniso=(sx,sy)), the dense MTs are the only remaining source. If the MTs show a
# residual stretch above this gate, recover an anisotropic affine from the correspondences and
# adopt it (the symmetric inverse of gate_coarse_scale, which DROPS a bad image scale). Gated
# high so a healthy residual never trips it: where the image already applied its aniso the
# residual is ~unit (ratio ~1.01 on Monopoles); only a genuine missed stretch (rescued
# sec01->sec02, ratio ~1.10, closes ~72% of the boundary gap) clears it.
_RESCUE_ANISO_GATE = 0.05
_RESCUE_ANISO_MIN_PAIRS = 12


def _fit_affine_2d(src: np.ndarray, dst: np.ndarray):
    """Least-squares anisotropic affine ``(L, t)`` mapping ``src -> dst`` (``dst ≈ L@src + t``)."""
    m = np.column_stack([src, np.ones(len(src))])
    a, *_ = np.linalg.lstsq(m, dst, rcond=None)        # (3, 2): dst ≈ m @ a
    return a[:2].T, a[2]


def _rescue_relative(coords_ref, coords_mov, z_band_fraction, search_kwargs,
                     match_kwargs):
    """Re-estimate one interface's relative pose from the MTs (rotation search +
    rigid fit, then optional anisotropy recovery), to rescue an interface where the
    image coarse rotation failed.

    Adoption is decided by the DIRECT criterion — does the re-estimated rotation make
    the MTs actually match (its post-fit match fraction, the caller's gate) — not by
    the rotation search's own ``confident`` flag, which is an unreliable proxy here: on
    the real Monopoles sec01→sec02 the CPD angle was flagged *not* confident
    (flip_ratio 1.00) yet it produced the best production match and was the correct
    rescue. A wrong rotation cannot produce a good MT match, so the gate is the safety.

    Returns ``(rel, match_fraction, mt_angle)`` or ``None`` when too few endpoints or
    the search yields no usable fit. (The decoy-robust multi-seed CPD is the default
    search; the full gated sweep is left out — it is prohibitively slow on a
    thousand-endpoint bundle and the match gate already rejects a bad estimate.)
    """
    ref_eps = _face(coords_ref, "top", z_band_fraction)
    mov_eps = _face(coords_mov, "bottom", z_band_fraction)
    if len(ref_eps) < 3 or len(mov_eps) < 3:
        return None
    rho = _endpoint_rho(ref_eps)
    est = global_rotation_search(ref_eps, mov_eps, rho,
                                 **(search_kwargs or {"use_cpd": True}))
    seed = _centroid_rotation_seed(est.angle, mov_eps)
    best = _evaluate_seed(seed, ref_eps, mov_eps, rho, False, match_kwargs, fit=True)
    if best is None:
        return None
    rel, mf, aniso = best["rel"], float(best["match_fraction"]), None
    # Recover this interface's anisotropy from the MTs (the image coarse failed, so it never
    # could). The stretch lives in the PERIPHERAL MTs, which only match once the warp bootstrap
    # pulls them in — the single-pass rescue match is the central, near-aligned subset and
    # under-reads the aniso (sec01->sec02: 1.02 single-pass vs 1.10 dense). So fit the affine on
    # the DENSE register_warps_to_coarse correspondences, and adopt it only when the stretch is
    # genuine (> gate) AND it does not hurt the (bootstrapped) match — a spurious affine
    # overfitting the spatially-varying field would not hold its match when re-applied.
    rr = register_warps_to_coarse(
        [coords_ref, coords_mov], [dict(IDENTITY), dict(rel)],
        z_band_fraction=z_band_fraction, **match_kwargs,
    )
    pairs = rr.interfaces[0].id_pairs if rr.interfaces else []
    if len(pairs) >= _RESCUE_ANISO_MIN_PAIRS:
        ref_by = {int(e["id"]): np.asarray(e["pos"][:2], float) for e in ref_eps}
        mov_by = {int(e["id"]): np.asarray(e["pos"][:2], float) for e in mov_eps}
        pr = [ref_by[int(ri)] for ri, mi, _ in pairs if int(ri) in ref_by and int(mi) in mov_by]
        pm = [mov_by[int(mi)] for ri, mi, _ in pairs if int(ri) in ref_by and int(mi) in mov_by]
        if len(pr) >= _RESCUE_ANISO_MIN_PAIRS:
            lin, t = _fit_affine_2d(np.array(pm), np.array(pr))   # mov-local -> ref-local
            sv = np.linalg.svd(lin, compute_uv=False)
            m_cur = float(rr.interfaces[0].confidence.get("match_fraction", mf))
            if sv[1] > 1e-9 and sv[0] / sv[1] - 1.0 > _RESCUE_ANISO_GATE:
                rel_aff = pose_from_matrix(lin, t)
                m_aff = _seed_match_fraction(
                    coords_ref, coords_mov, rel_aff, z_band_fraction, match_kwargs
                )
                if m_aff >= m_cur:
                    rel, mf, aniso = rel_aff, m_aff, (float(sv[0]), float(sv[1]))
    return rel, mf, float(est.angle), aniso


def rescue_coarse_poses(
    coords_list,
    coarse_poses,
    match_fractions,
    *,
    z_band_fraction: float = 0.15,
    match_gate: float = 0.3,
    search_kwargs=None,
    **match_kwargs,
):
    """Rescue interfaces where the image coarse rotation failed, using the MTs.

    The image-driven coarse is reliable almost everywhere, but on an occasional
    hard interface (large drift, low overlap, and — on MT-bundle samples — no
    nuclear contour to cross-check) it can commit a grossly wrong rotation. The MT
    match fraction is the detector: *a wrong rotation collapses the match*. Where
    that match is below ``match_gate``, this re-estimates the rotation from the
    (dense) MTs and **adopts it only when the MT estimate is confident AND it
    clears the gate AND it beats the image's match** — so a rescue can never make
    an interface worse. A genuinely sparse interface, or a correct large rotation
    that already matched well, is left untouched.

    :param match_fractions: per-interface MT match fraction from a first
        :func:`register_warps_to_coarse` pass (length ``n-1``).
    :param search_kwargs: forwarded to :func:`global_rotation_search` (e.g.
        ``{"use_cpd": True, ...}`` for the decoy-robust multi-seed search).
    :return: ``(poses, rescues)`` — the corrected absolute pose chain (unchanged
        where nothing was rescued) and a list of
        ``(k, image_angle, mt_angle, image_match, mt_match, aniso)`` for logging,
        where ``aniso`` is the recovered ``(sx, sy)`` or ``None``.
    """
    n = len(coords_list)
    if n < 2:
        return [dict(p) for p in coarse_poses], []
    rels = [
        compose_poses(invert_pose(coarse_poses[k]), coarse_poses[k + 1])
        for k in range(n - 1)
    ]
    rescues = []
    for k in range(n - 1):
        if k >= len(match_fractions) or match_fractions[k] >= match_gate:
            continue
        out = _rescue_relative(
            coords_list[k], coords_list[k + 1], z_band_fraction, search_kwargs,
            match_kwargs,
        )
        if out is None:
            continue
        rel_mt, mf_mt, ang_mt, aniso_mt = out
        if mf_mt >= match_gate and mf_mt > match_fractions[k]:
            img_ang = float(rels[k].get("Angle", 0.0))
            rels[k] = rel_mt
            rescues.append(
                (k, img_ang, ang_mt, float(match_fractions[k]), mf_mt, aniso_mt)
            )
    if not rescues:
        return [dict(p) for p in coarse_poses], []
    new_poses = [dict(coarse_poses[0])]
    for k in range(n - 1):
        new_poses.append(compose_poses(new_poses[-1], rels[k]))
    return new_poses, rescues


# Below this singular-value deviation from 1 a relative pose carries no meaningful scale,
# so the scale-gate skips it (the ~unit-scale majority of interfaces costs nothing). The
# floor clears the real per-axis stretch of the *helpful* aniso interfaces on Monopoles
# (max dev ~0.083) so those still get scored — and kept, because the full pose wins.
_SCALE_GATE_MIN_SV_DEV = 0.05
# Rotation-only must beat the full image scale by at least this MT-match margin to drop
# the scale — well clear of match noise, so a genuinely helpful aniso is never dropped.
_SCALE_GATE_MARGIN = 0.10


def _seed_match_fraction(coords_ref, coords_mov, seed, z_band_fraction, match_kwargs):
    """Production MT match fraction of one interface under a relative ``seed`` pose.

    Scores through the SAME path the export uses —
    :func:`register_warps_to_coarse` on the two sections, i.e. the warp **bootstrap** —
    not a single match. This matters: a single match badly under-reads rotation-only (it
    keeps the scale-fit translation and never recovers the residual), so a single-pass
    comparison would wrongly favour the overfit scale; the bootstrap recovers the residual
    and reveals rotation-only's true quality. ``0.0`` when the interface yields no warp.
    """
    r = register_warps_to_coarse(
        [coords_ref, coords_mov], [dict(IDENTITY), dict(seed)],
        z_band_fraction=z_band_fraction, **match_kwargs,
    )
    if not r.interfaces:
        return 0.0
    return float(r.interfaces[0].confidence.get("match_fraction", 0.0))


def gate_coarse_scale(
    coords_list,
    coarse_poses,
    *,
    z_band_fraction: float = 0.15,
    match_gate: float = 0.3,
    scale_margin: float = _SCALE_GATE_MARGIN,
    min_sv_dev: float = _SCALE_GATE_MIN_SV_DEV,
    **match_kwargs,
):
    """Drop an interface's image-coarse scale where the dense MTs match clearly better
    WITHOUT it.

    The image coarse is reliable almost everywhere, but on an occasional interface the
    affine refine overfits an extreme anisotropic stretch (a registration artifact — e.g.
    a 12% area inflation two adjacent EM sections cannot physically show) that warps the
    volume corner by thousands of Å. The MTs are the detector: *a wrong scale collapses
    the match while rotation-only recovers it*. For each interface whose relative pose
    carries a non-trivial scale (singular-value deviation > ``min_sv_dev``), this scores
    the MT match under the full image pose vs rotation-only; when rotation-only clears
    ``match_gate`` AND beats the full pose by ``scale_margin`` it drops the scale to
    rotation-only (translation kept) and re-accumulates the absolute chain. A genuine
    aniso — which makes the FULL pose match better — is left untouched, so the gate can
    never make an interface worse. This is the SCALE analogue of
    :func:`rescue_coarse_poses` (which rescues a wrong rotation); the two target disjoint
    interfaces (rescue: match below gate; gate: rotation-only beats a committed scale).

    :return: ``(poses, gated)`` — the corrected absolute pose chain (unchanged where
        nothing was gated) and a list of ``(k, det_before, full_match, rot_match)`` for
        logging.
    """
    n = len(coords_list)
    if n < 2:
        return [dict(p) for p in coarse_poses], []
    rels = [
        compose_poses(invert_pose(coarse_poses[k]), coarse_poses[k + 1])
        for k in range(n - 1)
    ]
    gated = []
    for k in range(n - 1):
        rel = rels[k]
        sv = np.linalg.svd(linear_part(rel), compute_uv=False)
        if max(abs(sv[0] - 1.0), abs(sv[1] - 1.0)) <= min_sv_dev:
            continue  # ~unit scale: nothing to gate
        # Rotation-only seed = rotate the moving cloud IN PLACE about its centroid (the
        # rescue's seed). Keeping the scale-fit translation instead misplaces the section
        # (that translation was fit FOR the bad scale); rotating in place lets the
        # register recover the true translation, the same way the export will.
        mov_eps = _face(coords_list[k + 1], "bottom", z_band_fraction)
        if len(mov_eps) < 3:
            continue
        rot_rel = _centroid_rotation_seed(float(rel["Angle"]), mov_eps)
        m_full = _seed_match_fraction(
            coords_list[k], coords_list[k + 1], rel, z_band_fraction, match_kwargs
        )
        m_rot = _seed_match_fraction(
            coords_list[k], coords_list[k + 1], rot_rel, z_band_fraction, match_kwargs
        )
        if m_rot >= match_gate and m_rot > m_full + scale_margin:
            gated.append((k, float(np.linalg.det(linear_part(rel))), m_full, m_rot))
            rels[k] = rot_rel
    if not gated:
        return [dict(p) for p in coarse_poses], []
    new_poses = [dict(coarse_poses[0])]
    for k in range(n - 1):
        new_poses.append(compose_poses(new_poses[-1], rels[k]))
    return new_poses, gated
