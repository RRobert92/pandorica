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
Result accessors + stitched-output writer.

A per-interface QC summariser for the results table, and
:func:`export_stitched` — which applies the solved per-section poses to the
volumes (streaming, slice-wise, via the package's :mod:`applier`) and to the
microtubule graphs, writing a single stitched ``.am`` volume + merged
``*_spatialGraph.am`` in the same coordinate frame.
"""

from os import makedirs
from os.path import join
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

from pandorica.stitch.transform.solver import (
    Pose,
    apply_pose,
    invert_pose,
)
from pandorica.stitch.geometry import pose_to_pixel
from pandorica.stitch.dataset import Dataset

ProgressCb = Optional[Callable[[str, float], None]]


# --------------------------------------------------------------------------- #
# Result accessors
# --------------------------------------------------------------------------- #
def result_poses(result) -> List[Pose]:
    """Absolute per-section poses from a stitch result."""
    return list(result.poses)


def result_warps(result) -> List:
    """Per-interface guarded TPS warps from a stitch result (length n-1)."""
    return [it.warp for it in result.base.interfaces]


class _FramedWarp:
    """
    Adapt an interface-local guarded warp to a section's absolute output frame.

    Each interface warp is fit in the **previous** section's local Å frame
    (mapping the rigidly-aligned moving face onto the fixed reference face). To
    apply it in the export's output frame we map a query point back through that
    section's absolute ``frame_pose``, evaluate the base displacement, then rotate
    the displacement (by the frame's rotation+scale) into the output frame.

    ``make_inverse_map`` only needs ``.accepted`` and ``.displacement(xy)``, with
    ``xy`` in the same units as the paired pose. ``coord_to_A`` converts those
    working-unit coordinates to/from the Å the base warp expects (= pixel size for
    the volume path, 1.0 for the Å graph path).
    """

    def __init__(self, base, frame_pose: Pose, coord_to_A: float = 1.0):
        self.accepted = bool(getattr(base, "accepted", False))
        self._base = base
        self._inv = invert_pose(frame_pose)
        self._rot = {
            "Angle": frame_pose["Angle"],
            "Tx": 0.0,
            "Ty": 0.0,
            "Scale": frame_pose["Scale"],
        }
        self._k = coord_to_A

    def displacement(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        local = apply_pose(self._inv, xy) * self._k  # prev-section local, Å
        d = self._base.displacement(local)  # Å displacement
        return apply_pose(self._rot, d) / self._k  # back to working unit


class _SumWarp:
    """Sum the displacements of several warps that share one (interface-local) frame.

    Used to combine an interface's MT warp with its image-fill warp: both are fit
    in the same section-k Å frame, so their residual displacements add. ``accepted``
    is True if any constituent is.
    """

    def __init__(self, warps):
        self._warps = [
            w for w in warps if w is not None and getattr(w, "accepted", False)
        ]
        self.accepted = len(self._warps) > 0

    def displacement(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        d = np.zeros_like(xy)
        for w in self._warps:
            d = d + w.displacement(xy)
        return d


def interface_rows(result, dataset: Dataset) -> List[Dict]:
    """
    Flatten a stitch result into per-interface rows for the QC table.

    Core columns come from the registration ``base``; the hybrid coarse angle/flag
    and the intensity-verification verdict are added when available.
    """
    base = result.base
    rows: List[Dict] = []
    for k, iface in enumerate(base.interfaces):
        row = {
            "interface": dataset.interface_label(k),
            "coarse_deg": iface.coarse.get("Angle", 0.0),
            "relative_deg": iface.relative.get("Angle", 0.0),
            "match_frac": iface.qc.match_fraction,
            "incoherence_rho": iface.qc.shift_incoherence_rho,
            "tangent_deg": iface.qc.tangent_discontinuity_deg,
            "warp_ok": bool(iface.warp.accepted),
            "qc_ok": bool(iface.qc.accepted),
            "chained": bool(iface.qc.chainable),
            "reasons": "; ".join(iface.qc.reasons),
        }
        rec = result.hybrid.records[k] if k < len(result.hybrid.records) else None
        ang = result.hybrid.angles
        row["hybrid_deg"] = ang[k] if k < len(ang) else float("nan")
        row["hybrid_flag"] = bool(rec.flagged) if rec is not None else False
        v = result.intensity[k] if k < len(result.intensity) else None
        row["intensity_ok"] = None if v is None else bool(v.verified)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Stitched volume + microtubule export
# --------------------------------------------------------------------------- #
def _displacement_grid_coarse(
    framed_warp,
    out_hw: Tuple[int, int],
    out_pts: np.ndarray,
    coarse_px: int = 8,
) -> np.ndarray:
    """
    Evaluate ``framed_warp.displacement`` on a coarse canvas grid and
    bilinearly upsample to every output pixel.

    The TPS field is smooth by construction (diffeomorphism-guarded), so
    sampling it on a coarse grid and bilinearly interpolating to canvas
    resolution introduces sub-pixel error (~(coarse_px)² × local field
    curvature) while replacing the dominant cost of full-canvas RBF
    evaluation. ``coarse_px=0`` (or any ≥ min(Hc, Wc)) → falls back to full
    per-pixel evaluation.

    The grid is **canvas-relative**, so spacing stays at ``coarse_px``
    regardless of input size: an 8000-px-wide canvas uses ~1000 samples
    across; a 256-px-wide canvas uses ~32 — no special small-data branch.

    :param framed_warp: object exposing ``.displacement(xy)``; ``xy`` is
        ``[M, 2]`` ``(x, y)`` in canvas pixels.
    :param out_hw: output canvas ``(Hc, Wc)``.
    :param out_pts: ``[M, 2]`` query points (``(x, y)``, canvas pixels).
    :param coarse_px: target sample spacing in canvas pixels. ``0`` or a
        value at least as large as ``min(Hc, Wc)`` → no optimization.
    :return: ``[M, 2]`` ``(dx, dy)`` displacements at ``out_pts``.
    """
    hc, wc = out_hw
    if coarse_px <= 0 or coarse_px >= min(hc, wc):
        return np.asarray(framed_warp.displacement(out_pts), dtype=np.float32)

    # Coarse-grid dimensions: ceil so spacing never exceeds coarse_px. +1 puts
    # both endpoints (0 and Hc-1 / Wc-1) on the grid, so the bilinear
    # interpolator never extrapolates.
    n_y = max(2, int(np.ceil((hc - 1) / coarse_px)) + 1)
    n_x = max(2, int(np.ceil((wc - 1) / coarse_px)) + 1)
    y_c = np.linspace(0.0, hc - 1, n_y, dtype=np.float64)
    x_c = np.linspace(0.0, wc - 1, n_x, dtype=np.float64)
    xx, yy = np.meshgrid(x_c, y_c, indexing="xy")  # both (n_y, n_x)
    coarse_xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    coarse_d = np.asarray(framed_warp.displacement(coarse_xy)).reshape(n_y, n_x, 2)

    # map_coordinates wants (row, col) = (y_idx, x_idx) in coarse-grid index
    # units; out_pts is (x, y) in canvas pixels, so divide each by the
    # coarse-grid spacing along its axis.
    y_step = (hc - 1) / (n_y - 1)
    x_step = (wc - 1) / (n_x - 1)
    yi = out_pts[:, 1].astype(np.float64) / y_step
    xi = out_pts[:, 0].astype(np.float64) / x_step
    coords = np.stack([yi, xi], axis=0)  # (2, M)

    dx = map_coordinates(coarse_d[..., 0], coords, order=1, mode="nearest")
    dy = map_coordinates(coarse_d[..., 1], coords, order=1, mode="nearest")
    return np.column_stack([dx, dy]).astype(np.float32)


def _warp_volume_zblend(
    volume,
    inv_pose: Pose,
    out_hw: Tuple[int, int],
    out_pts: np.ndarray,
    b_grid: np.ndarray,
    t_grid: np.ndarray,
    dtype=np.uint8,
):
    """
    Warp a ``[Z, Y, X]`` volume with a **Z-varying** in-plane displacement.

    The displacement at slice ``k`` is a linear blend of the bottom field
    ``b_grid`` (full at the low-Z face, ``k=0``) and the top field ``t_grid``
    (full at the high-Z face) — so each section's two faces carry their own
    interface's half-residual and the deformation interpolates between them
    through Z, instead of one warp applied uniformly across the section.

    When ``b_grid is t_grid`` (same object — the uniform-warp regime used by
    the ``warp_zblend=False`` export path), the per-slice ``disp`` and ``src``
    are constant in Z and are computed once outside the loop.

    :param inv_pose: inverse of the section's absolute (pixel, canvas-offset) pose.
    :param out_pts: ``[M, 2]`` output-grid ``(x, y)`` points (M = Hc*Wc).
    :param b_grid / t_grid: ``[M, 2]`` displacements at the bottom / top face.
    """
    z = volume.shape[0]
    hc, wc = out_hw
    out = np.empty((z, hc, wc), dtype=dtype)
    uniform = b_grid is t_grid
    if uniform:
        src = apply_pose(inv_pose, out_pts - b_grid)
        coords = np.vstack([src[:, 1], src[:, 0]])
    for k in range(z):
        if not uniform:
            a = 1.0 - (k / (z - 1)) if z > 1 else 1.0  # 1 at bottom (k=0) -> 0 at top
            disp = a * b_grid + (1.0 - a) * t_grid
            src = apply_pose(inv_pose, out_pts - disp)  # input (x, y)
            coords = np.vstack([src[:, 1], src[:, 0]])  # map_coordinates wants (row, col)
        out[k] = (
            map_coordinates(volume[k], coords, order=1, mode="constant", cval=0.0)
            .reshape(hc, wc)
            .astype(dtype)
        )
    return out


def _corner_bbox(
    poses_px: Sequence[Pose], h: int, w: int
) -> Tuple[int, int, np.ndarray]:
    """Common output canvas (Hc, Wc) and pixel offset for pixel-poses over an (h, w) frame."""
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)  # (x, y)
    pts = np.vstack([apply_pose(p, corners) for p in poses_px])
    mn, mx = pts.min(0), pts.max(0)
    wc = int(np.ceil(mx[0] - mn[0]))
    hc = int(np.ceil(mx[1] - mn[1]))
    return hc, wc, -mn  # offset (ox, oy) maps bbox-min -> (0, 0)


def _mt_bbox(
    dataset: Dataset,
    poses_px: Sequence[Pose],
    px: float,
    pad_frac: float = 0.05,
    fallback_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, np.ndarray]:
    """
    Canvas ``(Hc, Wc)`` and offset sized to enclose every microtubule after pose
    transform, padded by ``pad_frac`` of the MT-bbox span on each side.

    The corner-bbox (:func:`_corner_bbox`) sizes the canvas to fit every input
    frame after pose transform, which is conservative: with N sections drifting
    apart, most of the canvas is empty corners. Trimming to the MT bbox cuts the
    output volume to just the spatial region that carries microtubules, and the
    warp / disk-write cost drops in proportion.

    MT coords on a section are stored in physical units (Å, columns 1 and 2 of
    the ``[id, x, y, z]`` graph); ``poses_px`` is in canvas pixels, so we divide
    by ``px`` first to get pixel-frame inputs.

    :param dataset: loaded sections (their ``.coords`` are read here).
    :param poses_px: pixel-frame absolute poses (length == n sections).
    :param px: pixel size in physical units (Å/pixel).
    :param pad_frac: padding as a fraction of the MT-bbox span (per axis). ``0.05``
        gives 5 % context on every side, scaling with the stack.
    :param fallback_hw: ``(h, w)`` to use for the corner-bbox fallback when no
        section has microtubules. ``None`` → return ``(0, 0, zeros(2))``.
    :return: ``(Hc, Wc, offset)`` in the same shape as :func:`_corner_bbox`.
    """
    all_xy = []
    for i, s in enumerate(dataset.sections):
        c = getattr(s, "coords", None)
        if c is None or len(c) == 0:
            continue
        xy_px = c[:, 1:3] / max(px, 1e-12)  # MT (x, y) in input pixel coords
        all_xy.append(apply_pose(poses_px[i], xy_px))

    if not all_xy:
        if fallback_hw is not None:
            return _corner_bbox(poses_px, fallback_hw[0], fallback_hw[1])
        return 0, 0, np.zeros(2)

    pts = np.vstack(all_xy)
    mn, mx = pts.min(0), pts.max(0)
    pad = (mx - mn) * float(max(pad_frac, 0.0))
    mn = mn - pad
    mx = mx + pad
    wc = int(np.ceil(mx[0] - mn[0]))
    hc = int(np.ceil(mx[1] - mn[1]))
    return hc, wc, -mn


def export_stitched(
    dataset: Dataset,
    poses: Sequence[Pose],
    output_dir: str,
    downscale: int = 1,
    progress: ProgressCb = None,
    write_volume: bool = True,
    warps: Optional[Sequence] = None,
    warp_zblend: bool = False,
    image_warps: Optional[Sequence] = None,
    use_gpu: bool = False,
    gpu_chunk: Optional[int] = None,
    warp_coarse_px: int = 8,
    trim_to_mts: bool = False,
    mt_pad_frac: float = 0.05,
    interface_id_pairs: Optional[Sequence] = None,
    interface_accepted: Optional[Sequence[bool]] = None,
    chain_split_max_angle_deg: float = 45.0,
) -> Dict[str, str]:
    """
    Write a stitched volume + merged microtubule graph under the solved poses.

    The solved poses are physical-unit, in-plane, with section 0 the gauge anchor.
    Volumes are warped one section at a time into a shared canvas (sized from the
    section-0 frame transformed by every pose) and stacked along Z; microtubule
    graphs get the same in-plane pose plus the matching canvas offset and a
    cumulative Z shift, so volume and graph land in one coordinate frame.

    :param dataset: the loaded sections (volumes are read on demand here).
    :param poses: absolute per-section poses (length == number of sections).
    :param output_dir: directory to create and write outputs into.
    :param downscale: integer decimation of the volumes (and pixel size) — use >1
        to validate the stitch on a multi-GB stack quickly.
    :param progress: optional ``(message, fraction)`` callback for a progress bar.
    :param write_volume: if False, only merge/export the spatial graph (fast).
    :param warps: optional per-interface guarded warps (length ``n-1``, from
        :func:`result_warps`). When given, section ``i`` (the moving side of
        interface ``i-1``) additionally gets that interface's **local TPS
        deformation** applied — to both its volume slices and its microtubules —
        so the export reflects the fine alignment, not just the rigid poses. The
        reference (top) face of each section is intentionally not warped, matching
        how the warp was fit (moving→ref).
    :param warp_zblend: **Z-varying symmetric warp** (requires ``warps``).
        Instead of one warp applied uniformly across a section's Z, each interface
        residual is split half/half between its two adjacent faces, which meet at
        the mid-plane: section ``i``'s low-Z (bottom) face gets ``+½ W[i-1]`` and
        its high-Z (top) face ``-½ W[i]``, linearly blended through Z. This
        decouples the interfaces (warping one join no longer perturbs the next) at
        the cost of deforming every face including the gauge. Applied to both the
        volume slices and the microtubules. When False, the uniform warp above is
        used.
    :param image_warps: optional per-interface **image-fill** warps (length
        ``n-1``, from :func:`.image_warp.image_residual_warps`) aligning the
        MT-free regions from image content. Summed with the MT warp per interface
        (same section-k frame) and carried through identically, including the
        Z-blend. May be given with or without ``warps`` (latter = image-only).
    :param trim_to_mts: if True, size the canvas to the bounding box of the
        microtubules (with ``mt_pad_frac`` padding) instead of every section's
        corners. Drops empty-corner pixels — speeds the warp + reduces output
        size by the same factor. Falls back to the corner-bbox when no section
        has microtubules. Default ``False`` (no behaviour change for callers
        that did not opt in).
    :param mt_pad_frac: padding fraction of the MT-bbox span on each axis when
        ``trim_to_mts=True``. ``0.05`` = 5 % context on each side.
    :param interface_id_pairs: optional per-interface MT correspondences
        (length ``n-1``). When provided together with ``interface_accepted``,
        the merged spatial graph chains MTs across accepted joints into single
        global filaments — so a microtubule crossing four sections appears as
        one connected spline, not four. Chains break at flagged interfaces.
        When omitted the legacy per-section id-bump is used (each section's
        MTs stay independent in the output).
    :param interface_accepted: parallel ``[n-1]`` booleans (one per interface)
        telling the chain builder which joints to extend across. Required
        whenever ``interface_id_pairs`` is given.
    :param chain_split_max_angle_deg: post-chain joint check — after building
        chains, every joint is examined and the chain is split there if the
        OVERALL direction of the sub-block on each side disagrees by more
        than this angle (deg). Default 45°. A real MT cut and laterally
        shifted at a section boundary has a sharp local bend but the same
        overall direction on each side, so it survives this check; two
        unrelated MTs that got joined will have different overall directions
        and the chain breaks there.
    :return: dict of written paths (``volume``, ``graph``, ``log`` as available).
    """
    from pandorica.io.amira import (
        write_amira_volume_streamed,
        write_spatial_graph,
    )

    def _tick(msg: str, frac: float) -> None:
        if progress is not None:
            progress(msg, frac)

    makedirs(output_dir, exist_ok=True)
    n = len(dataset.sections)
    assert n == len(poses), f"poses ({len(poses)}) must match sections ({n})"
    try:
        from pandorica._version import version as _PV
    except Exception:  # noqa: BLE001
        _PV = "?"
    log: List[str] = [
        f"PANDORICA serial-section stitch export  (pandorica v{_PV})",
        f"folder:     {dataset.folder}",
        f"sections:   {n}",
        f"downscale:  {downscale}",
        "",
    ]
    written: Dict[str, str] = {}

    # -- pixel size + nominal frame from the first section that has a volume ---
    px = 1.0
    h = w = None
    for s in dataset.sections:
        if s.has_volume():
            v0 = s.load_volume(downscale=downscale)
            px = s.pixel_size
            h, w = v0.shape[1], v0.shape[2]
            s.drop_volume()
            break
    poses_px = [pose_to_pixel(p, px) for p in poses]

    n_iface = (len(warps) if warps is not None else 0) or (
        len(image_warps) if image_warps is not None else 0
    )

    def _iface_warp(k: int):
        """Combined MT + image-fill warp for interface k, in section k's frame."""
        if k < 0 or k >= n_iface:
            return None
        cands = []
        if warps is not None and k < len(warps):
            cands.append(warps[k])
        if image_warps is not None and k < len(image_warps):
            cands.append(image_warps[k])
        sw = _SumWarp(cands)
        return sw if sw.accepted else None

    def _section_warp(i: int):
        """Interface (i-1) warp for moving section i (its low-Z/bottom face)."""
        return _iface_warp(i - 1)

    def _upper_warp(i: int):
        """Interface i warp for section i's high-Z/top face (top boundary)."""
        return _iface_warp(i)

    # ----------------------------- volume -------------------------------------
    z_thickness = [0] * n  # slice counts (for the graph's Z stacking)
    if write_volume and h is not None:
        if trim_to_mts:
            hc, wc, offset = _mt_bbox(
                dataset,
                poses_px,
                px=px,
                pad_frac=mt_pad_frac,
                fallback_hw=(h, w),
            )
            full_hc, full_wc, _ = _corner_bbox(poses_px, h, w)
            full_area = max(full_hc * full_wc, 1)
            ratio = (hc * wc) / full_area
            delta_pct = abs(100.0 * (1.0 - ratio))
            verb = "smaller" if ratio <= 1.0 else "larger"
            log += [
                f"pixel size: {px:g}",
                f"canvas HxW: {hc} x {wc}  (MT-trim, pad={mt_pad_frac:.0%})",
                f"  vs corner: {full_hc} x {full_wc}  -> {delta_pct:.1f}% {verb}",
                f"offset px:  ({offset[0]:.1f}, {offset[1]:.1f})",
                "",
            ]
        else:
            hc, wc, offset = _corner_bbox(poses_px, h, w)
            log += [
                f"pixel size: {px:g}",
                f"canvas HxW: {hc} x {wc}",
                f"offset px:  ({offset[0]:.1f}, {offset[1]:.1f})",
                "",
            ]

        def _frame_off(j: int) -> Pose:
            """Section j's absolute pixel pose, shifted by the canvas offset."""
            fp = dict(poses_px[j])
            fp["Tx"] += float(offset[0])
            fp["Ty"] += float(offset[1])
            return fp

        # Canvas grid is needed by every warp path (zblend OR uniform; GPU OR CPU)
        # because both _warp_volume_zblend and warp_volume_torch consume out_pts
        # alongside b_grid / t_grid. float32: ~(Hc·Wc·8) bytes — the dominant
        # non-volume buffer — and the GPU warp casts to float32 anyway. Canvas
        # pixel coords are exact in float32, so no precision loss.
        yy, xx = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
        out_pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
        _zero_grid = np.zeros_like(out_pts)

        gpu_device = None
        effective_chunk = int(gpu_chunk) if gpu_chunk is not None else 4
        if use_gpu:
            from pandorica.stitch import accel as _accel

            gpu_device = _accel.pick_device(True)
            log.append(f"GPU warp device: {gpu_device}")
            if gpu_chunk is None and gpu_device not in (None, "cpu"):
                effective_chunk = _accel.auto_gpu_chunk(
                    gpu_device,
                    out_hw=(hc, wc),
                    in_hw=(h, w),
                )
                free_bytes = _accel.device_free_bytes(gpu_device)
                free_str = (
                    f"{free_bytes / 1e9:.1f} GB free"
                    if free_bytes is not None
                    else "no memory query"
                )
                log.append(
                    f"GPU chunk    : auto -> {effective_chunk} slices  ({free_str})"
                )
            else:
                log.append(f"GPU chunk    : {effective_chunk} slices  (manual)")

        full_z = 0
        temp = []
        for i, s in enumerate(dataset.sections):
            _tick(f"warping {s.name}", i / max(n, 1))
            if not s.has_volume():
                log.append(f"  [{i}] {s.name}: no volume — skipped")
                continue
            vol = s.load_volume(downscale=downscale)
            z_thickness[i] = vol.shape[0]
            wb, wt = _section_warp(i), _upper_warp(i)
            # Stream the warped section straight to a raw temp via f.write — regular
            # file I/O goes through the OS page cache, NOT process RSS, so nothing
            # larger than one GPU chunk is held in memory (a memmap, by contrast,
            # keeps its pages resident). Assembled into the .am by to_am_streamed.
            tf = join(output_dir, f"_tmp_sec{i:02d}.dat")
            with open(tf, "wb") as fout:
                # Build (b_grid, t_grid) for this section. Two regimes:
                #   * warp_zblend: bottom face carries +½·W[i-1] (its own interface),
                #     top face carries -½·W[i] (the next interface). The k-blend in
                #     _warp_volume_zblend / warp_volume_torch then linearly varies
                #     between them through Z.
                #   * uniform (warp_zblend=False): only the lower interface (W[i-1])
                #     applies to this section, identically at every Z. b_grid ==
                #     t_grid collapses the linear blend to a constant — same math
                #     as the old make_inverse_map(pose, warp=vw) path, but evaluated
                #     on the coarse grid and warped on the GPU when available.
                if warp_zblend:
                    b_grid = (
                        (
                            0.5
                            * _displacement_grid_coarse(
                                _FramedWarp(wb, _frame_off(i - 1), px),
                                (hc, wc),
                                out_pts,
                                coarse_px=warp_coarse_px,
                            )
                        ).astype(np.float32)
                        if wb is not None
                        else _zero_grid
                    )
                    t_grid = (
                        (
                            -0.5
                            * _displacement_grid_coarse(
                                _FramedWarp(wt, _frame_off(i), px),
                                (hc, wc),
                                out_pts,
                                coarse_px=warp_coarse_px,
                            )
                        ).astype(np.float32)
                        if wt is not None
                        else _zero_grid
                    )
                else:
                    b_grid = (
                        _displacement_grid_coarse(
                            _FramedWarp(wb, _frame_off(i - 1), px),
                            (hc, wc),
                            out_pts,
                            coarse_px=warp_coarse_px,
                        ).astype(np.float32)
                        if wb is not None
                        else _zero_grid
                    )
                    t_grid = b_grid

                inv_i = invert_pose(_frame_off(i))
                if use_gpu and gpu_device not in (None, "cpu"):
                    from pandorica.stitch import accel as _accel

                    _accel.warp_volume_torch(
                        vol,
                        inv_i,
                        (hc, wc),
                        out_pts,
                        b_grid,
                        t_grid,
                        device=gpu_device,
                        chunk=effective_chunk,
                        out=fout,
                    )
                else:
                    warped = _warp_volume_zblend(
                        vol, inv_i, (hc, wc), out_pts, b_grid, t_grid
                    )
                    fout.write(np.ascontiguousarray(warped).tobytes())

                has_warp = wb is not None or (warp_zblend and wt is not None)
                on_gpu = use_gpu and gpu_device not in (None, "cpu")
                if not has_warp:
                    tag = "  (+GPU)" if on_gpu else ""
                elif warp_zblend:
                    tag = "  (+Zblend/GPU)" if on_gpu else "  (+Zblend)"
                else:
                    tag = "  (+TPS/GPU)" if on_gpu else "  (+TPS)"
            z = vol.shape[0]
            s.drop_volume()
            temp.append((tf, z))
            full_z += z
            log.append(f"  [{i}] {s.name}: warped -> {(z, hc, wc)}{tag}")

        _tick("assembling volume", 0.9)
        vol_path = join(output_dir, "stitched_volume.am")
        # Concatenate the raw section slabs into the .am (file -> file via
        # copyfileobj): no full volume in RAM, no resident memmap pages.
        write_amira_volume_streamed(
            vol_path, [tf for tf, _ in temp], (full_z, hc, wc), px
        )
        written["volume"] = vol_path
        log.append(f"\nstitched volume: {(full_z, hc, wc)} -> {vol_path}")
        for tf, _ in temp:
            try:
                __import__("os").remove(tf)
            except OSError:
                pass
    else:
        # No volume export: still need an XY offset/px for graph placement.
        offset = np.zeros(2)
        # Without volumes there are no slice counts to stack the graph in Z, so
        # every section would land at Z-offset 0 and collapse onto one plane.
        # Use each section's own microtubule Z-extent as its thickness. Coords
        # are physical (Å); divide by px so it stays in the slice-equivalent
        # units the `z_off_slices * px` stack below expects (px == 1.0 here when
        # no section carries a volume, so this is exactly the Å span).
        for i, s in enumerate(dataset.sections):
            if len(s.coords):
                zc = s.coords[:, 3]
                z_thickness[i] = float(zc.max() - zc.min()) / px

    # --------------------------- microtubules ---------------------------------
    # When per-interface id_pairs + accept flags are supplied, build a global
    # filament-ID map so MTs that physically continue across *accepted*
    # interfaces become a single connected spline in the output (not one
    # disjoint spline per section they cross). Chains break at any interface
    # whose QC did not accept — losing connectivity is the correct response to
    # an untrustworthy joint. Without those inputs we fall back to the legacy
    # per-section id-bump (each section's MTs stay independent).
    _tick("merging microtubules", 0.95)
    use_chain = interface_id_pairs is not None and interface_accepted is not None
    id_map = None
    n_filaments = None
    if use_chain:
        from pandorica.stitch.chain import chain_filaments

        sections_mt_ids = [
            ([] if not len(s.coords) else sorted({int(x) for x in s.coords[:, 0]}))
            for s in dataset.sections
        ]
        id_map, n_filaments = chain_filaments(
            sections_mt_ids,
            list(interface_id_pairs),
            [bool(x) for x in interface_accepted],
        )

    merged = []
    section_idx_per_row: List[np.ndarray] = []
    id_off = 0
    z_off_slices = 0
    for i, s in enumerate(dataset.sections):
        if len(s.coords):
            c = s.coords.copy()
            xy = apply_pose(poses[i], c[:, 1:3])  # in-plane pose (physical, Å)
            wb, wt = _section_warp(i), _upper_warp(i)
            if n_iface and warp_zblend:
                # Z-varying: +½ bottom warp at low-Z, -½ top warp at high-Z.
                zc = c[:, 3]
                z0, z1 = float(zc.min()), float(zc.max())
                alpha = (1.0 - (zc - z0) / (z1 - z0)) if z1 > z0 else np.ones_like(zc)
                b = (
                    0.5 * _FramedWarp(wb, poses[i - 1], 1.0).displacement(xy)
                    if wb is not None
                    else np.zeros_like(xy)
                )
                t = (
                    -0.5 * _FramedWarp(wt, poses[i], 1.0).displacement(xy)
                    if wt is not None
                    else np.zeros_like(xy)
                )
                xy = xy + alpha[:, None] * b + (1.0 - alpha)[:, None] * t
            elif wb is not None:
                # Uniform: same TPS deformation as the volume, non-offset Å frame.
                xy = xy + _FramedWarp(wb, poses[i - 1], 1.0).displacement(xy)
            c[:, 1:3] = xy
            c[:, 1] += float(offset[0]) * px  # canvas offset (px -> Å)
            c[:, 2] += float(offset[1]) * px
            c[:, 3] += z_off_slices * px  # cumulative Z stack (Å)
            if use_chain:
                local_ids = c[:, 0].astype(int)
                gids = np.fromiter(
                    (id_map[(i, int(lid))] for lid in local_ids),
                    dtype=np.int64,
                    count=local_ids.size,
                )
                c[:, 0] = gids.astype(c.dtype)
                section_idx_per_row.append(np.full(c.shape[0], i, dtype=np.int64))
            else:
                c[:, 0] += id_off
                id_off = int(c[:, 0].max()) + 1
            merged.append(c)
        z_off_slices += z_thickness[i]

    if merged:
        merged_arr = np.concatenate(merged, axis=0)
        write_fields: Dict[str, Dict[str, np.ndarray]] = {}
        # Defined here so the non-chain branch also has access to densify ids
        # without changing point counts.
        if use_chain:
            # Stable-sort by gid (only): preserves the natural section order
            # for same-gid rows (sections are appended in stack order) AND
            # the original trace order within each section block. We then
            # reverse per-section sub-blocks as needed so the chain reads
            # continuously across joints. **Do NOT sort by Z** — a real MT
            # bends through 3D space and its Z is not monotonic; Z-sorting
            # produces zigzags in any non-Z-monotonic spline.
            # After orientation, a second pass breaks chains at joints where
            # the OVERALL direction on each side disagrees (the matcher's
            # filter only sees a coarse 20%-of-MT tangent, so it misses
            # cases visible only in the final transformed geometry).
            from pandorica.stitch.chain import (
                orient_chain_blocks,
                split_chains_at_joints,
                compute_chain_labels,
            )

            sec_arr = np.concatenate(section_idx_per_row, axis=0)
            order = np.argsort(merged_arr[:, 0].astype(np.int64), kind="stable")
            merged_arr = merged_arr[order]
            sec_arr = sec_arr[order]
            merged_arr = orient_chain_blocks(merged_arr, sec_arr)
            pre_split_gid = merged_arr[:, 0].astype(np.int64).copy()
            merged_arr = split_chains_at_joints(
                merged_arr, sec_arr, max_angle_deg=chain_split_max_angle_deg
            )
            # Re-sort by the (possibly new) gids so the writer's same-gid
            # contiguity rule still holds. Splits assigned fresh ids to
            # *suffixes* of existing chain blocks, which can leave the
            # array's gid sequence non-monotonic.
            order = np.argsort(merged_arr[:, 0].astype(np.int64), kind="stable")
            merged_arr = merged_arr[order]
            sec_arr = sec_arr[order]
            pre_split_gid = pre_split_gid[order]
            # Drop single-point filaments here (with our tracking arrays in
            # lockstep) so the writer doesn't drop them again and silently
            # misalign our per-point / per-edge labels.
            _ids = merged_arr[:, 0].astype(np.int64)
            _uniq, _counts = np.unique(_ids, return_counts=True)
            _keep_ids = set(_uniq[_counts > 1].tolist())
            _keep_mask = np.fromiter(
                (int(i) in _keep_ids for i in _ids), dtype=bool, count=_ids.size,
            )
            merged_arr = merged_arr[_keep_mask]
            sec_arr = sec_arr[_keep_mask]
            pre_split_gid = pre_split_gid[_keep_mask]
            n_filaments = int(np.unique(merged_arr[:, 0]).size)

            # Diagnostic labels — let the user colour chains by their score
            # in napari and tell us whether the metric correlates with what
            # they think is right or wrong.
            labels = compute_chain_labels(merged_arr, sec_arr, pre_split_gid)
            write_fields = {
                "point_int_fields": {
                    "SectionIdx": labels["point_section_idx"],
                    "AtJoint": labels["point_at_joint"],
                },
                "point_float_fields": {
                    "JointAngleDeg": labels["point_joint_angle_deg"],
                    "JointOverallDeg": labels["point_joint_overall_deg"],
                },
                "edge_int_fields": {
                    "ChainLength": labels["edge_chain_length"],
                    "NJoints": labels["edge_n_joints"],
                    "WasSplit": labels["edge_was_split"],
                },
                "edge_float_fields": {
                    "MaxJointAngleDeg": labels["edge_max_joint_angle_deg"],
                    "MaxJointOverallDeg": labels["edge_max_joint_overall_deg"],
                },
            }
        graph_path = join(output_dir, "stitched_spatialGraph.am")
        write_spatial_graph(graph_path, merged_arr, **write_fields)
        written["graph"] = graph_path
        n_units = (
            n_filaments if use_chain else int(merged_arr[:, 0].max()) + 1
        )
        label = "filaments" if use_chain else "MTs"
        log.append(
            f"merged graph: {merged_arr.shape[0]} pts / "
            f"{n_units} {label} -> {graph_path}"
        )

    log_path = join(output_dir, "stitch_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log) + "\n")
    written["log"] = log_path
    _tick("done", 1.0)
    return written
