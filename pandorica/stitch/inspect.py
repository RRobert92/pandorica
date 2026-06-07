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
Inspection bundle: dump the per-interface MT matches + warp field so the napari
``WarpMatchInspectorWidget`` can render them without re-running the (slow) stitch.

The expensive registration runs once in the CLI; ``--save-inspect`` writes a
portable ``.npz`` that the plugin loads and overlays. Everything is stored in the
**graph output frame** (each section placed by its absolute pose, in Å), the same
frame the exported spatial graph lands in — so a match line's length is the real
residual misalignment and the warp/curl sit where the artefact is.

Bundle layout (``np.savez_compressed``)::

    manifest                     JSON str: n_interfaces, per-interface QC, frame
    if{k}_match_ref   (M, 2)     matched reference endpoints (x, y) Å
    if{k}_match_mov   (M, 2)     matched moving endpoints (x, y) Å, posed (+warp if applied)
    if{k}_match_cost  (M,)       matcher cost per pair (lower = better)
    if{k}_vec_pts     (G, 2)     warp sample points (x, y) Å
    if{k}_vec_dir     (G, 2)     warp displacement (dx, dy) Å at those points
    if{k}_curl        (gn, gn)   |curl| of the warp field on a regular grid
    if{k}_extent      (4,)       [xmin, xmax, ymin, ymax] of the curl grid, Å

Coordinates are physical Å in natural ``(x, y)`` order; the plugin swaps to
napari's ``(row, col) = (y, x)`` on display.
"""

import json
from typing import List, Sequence

import numpy as np

from pandorica.stitch.pipeline.core import _face
from pandorica.stitch.transform.solver import Pose, apply_pose, invert_pose


def _framed_displacement(warp, frame_pose: Pose, xy: np.ndarray) -> np.ndarray:
    """Warp displacement at output-frame points (same math as stitch._FramedWarp, Å)."""
    xy = np.asarray(xy, dtype=float)
    if getattr(warp, "_rbf", None) is None:
        return np.zeros_like(xy)
    local = apply_pose(invert_pose(frame_pose), xy)
    d = warp.displacement(local)
    rot = {"Angle": frame_pose["Angle"], "Tx": 0.0, "Ty": 0.0,
           "Scale": frame_pose["Scale"]}
    return apply_pose(rot, d)


def _curl_grid(warp, frame_pose: Pose, lo, hi, grid_n: int):
    """Sample |curl| of the warp displacement on a regular grid over [lo, hi] (Å)."""
    ext = np.where(hi - lo > 1e-9, hi - lo, 1.0)
    lo, hi = lo - 0.05 * ext, hi + 0.05 * ext
    xs = np.linspace(lo[0], hi[0], grid_n)
    ys = np.linspace(lo[1], hi[1], grid_n)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    U = _framed_displacement(warp, frame_pose, pts).reshape(grid_n, grid_n, 2)
    curl = np.gradient(U[:, :, 1], dx, axis=1) - np.gradient(U[:, :, 0], dy, axis=0)
    extent = np.array([lo[0], hi[0], lo[1], hi[1]], dtype=float)
    return np.abs(curl), extent, pts, U.reshape(-1, 2)


def write_inspection_bundle(
    result,
    coords_list: Sequence[np.ndarray],
    poses: Sequence[Pose],
    section_names: Sequence[str],
    out_path: str,
    z_band_fraction: float = 0.15,
    grid_n: int = 48,
    vec_stride: int = 3,
) -> str:
    """
    Write a napari inspection bundle from a completed stitch.

    :param result: ``SerialStitchResult`` (uses ``result.base.interfaces``).
    :param coords_list: per-section MT coords ``[N, >=4]`` (gid, x, y, z, ...).
    :param poses: absolute per-section poses (length == n_sections).
    :param section_names: per-section names, for interface labels.
    :param out_path: ``.npz`` path to write.
    :return: ``out_path``.
    """
    interfaces = result.base.interfaces
    arrays = {}
    meta: List[dict] = []

    for k, iface in enumerate(interfaces):
        ref_eps = _face(coords_list[k], "top", z_band_fraction)
        mov_eps = _face(coords_list[k + 1], "bottom", z_band_fraction)
        ref_by_id = {int(e["id"]): np.asarray(e["pos"][:2], float) for e in ref_eps}
        mov_by_id = {int(e["id"]): np.asarray(e["pos"][:2], float) for e in mov_eps}

        warp_applied = bool(iface.warp.accepted)
        m_ref, m_mov, m_cost = [], [], []
        for ref_id, mov_id, cost in iface.id_pairs:
            ri, mi = int(ref_id), int(mov_id)
            if ri not in ref_by_id or mi not in mov_by_id:
                continue
            r_w = apply_pose(poses[k], ref_by_id[ri][None, :])[0]
            m_w = apply_pose(poses[k + 1], mov_by_id[mi][None, :])[0]
            if warp_applied:
                m_w = m_w + _framed_displacement(iface.warp, poses[k + 1], m_w[None, :])[0]
            m_ref.append(r_w)
            m_mov.append(m_w)
            m_cost.append(float(cost))
        m_ref = np.asarray(m_ref, float).reshape(-1, 2)
        m_mov = np.asarray(m_mov, float).reshape(-1, 2)

        # Warp field + |curl| over the matched footprint (the reference endpoints).
        if len(m_ref):
            lo, hi = m_ref.min(0), m_ref.max(0)
        else:  # no matches — fall back to the posed reference cloud
            rc = apply_pose(poses[k], np.array(list(ref_by_id.values())).reshape(-1, 2))
            lo, hi = (rc.min(0), rc.max(0)) if len(rc) else (np.zeros(2), np.ones(2))
        curl, extent, vpts, vdir = _curl_grid(iface.warp, poses[k + 1], lo, hi, grid_n)

        arrays[f"if{k}_match_ref"] = m_ref.astype(np.float32)
        arrays[f"if{k}_match_mov"] = m_mov.astype(np.float32)
        arrays[f"if{k}_match_cost"] = np.asarray(m_cost, np.float32)
        arrays[f"if{k}_vec_pts"] = vpts[::vec_stride].astype(np.float32)
        arrays[f"if{k}_vec_dir"] = vdir[::vec_stride].astype(np.float32)
        arrays[f"if{k}_curl"] = curl.astype(np.float32)
        arrays[f"if{k}_extent"] = extent.astype(np.float32)

        c = iface.warp.certificate
        meta.append({
            "k": k,
            "name": f"{section_names[k]}->{section_names[k + 1]}",
            "warp_accepted": warp_applied,
            "chainable": bool(iface.qc.chainable),
            "qc_accepted": bool(iface.qc.accepted),
            "max_curl": float(c.max_abs_vorticity),
            "min_detj": float(c.min_det_j),
            "match_fraction": float(iface.qc.match_fraction),
            "n_matches": int(len(m_cost)),
            "reasons": "; ".join(iface.qc.reasons),
        })

    manifest = {
        "n_interfaces": len(interfaces),
        "frame": "graph_output_A (x, y)",
        "z_band_fraction": z_band_fraction,
        "interfaces": meta,
    }
    np.savez_compressed(out_path, manifest=np.array(json.dumps(manifest)), **arrays)
    return out_path
