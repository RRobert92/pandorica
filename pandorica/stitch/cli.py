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
Headless, production entry point for the serial-section stitcher.

``run_stitch`` drives the full validated pipeline end-to-end with auto defaults —
load → coarse (CPD) + MT match + guarded TPS warp + global solve → optional
image-fill of MT-free regions → GPU/CPU volume warp + microtubule merge — and
writes a stitched ``.am`` volume, merged ``*_spatialGraph.am``, and a detailed log
(settings used, per-interface transforms + QC, global poses, device, and a compute
time breakdown). No napari / Qt; safe to call from a terminal CLI or a cluster job.

GPU is auto-selected (CUDA → MPS → CPU) and memory-bounded (slice-chunked warp,
lazy volume loading), so the same call runs on a workstation, a Mac, or a CUDA box.
"""

import time
from datetime import datetime
from os import makedirs
from os.path import join
from typing import Optional

try:
    from pandorica._version import version as _VERSION
except Exception:  # noqa: BLE001
    _VERSION = "?"

from pandorica.stitch.io import load_dataset
from pandorica.stitch import stitch as stitch
from pandorica.stitch import image_warp as imgwarp
from pandorica.stitch import accel as accel
from pandorica.stitch.pipeline.stitcher import (
    stitch_sections,
    _INTENSITY_MIN_ANGLE_DEG,
)


def _fmt_pose(p):
    return (
        f"angle={p['Angle']:+8.3f}°  Tx={p['Tx']:+10.1f}  "
        f"Ty={p['Ty']:+10.1f}  scale={p['Scale']:.4f}"
    )


def _ascii_volume_summary(sections):
    """
    Identify ASCII Amira volumes among ``sections`` and report on them.

    Returns ``(n_ascii, total_bytes)``. ASCII ``.am`` images are ~200× slower
    to parse than BINARY-LITTLE-ENDIAN — without this audit, users see a long
    silent wait and assume the pipeline hung.
    """
    from os.path import getsize

    n_ascii = 0
    total_bytes = 0
    for s in sections:
        if not s.has_volume() or s.image_path is None:
            continue
        try:
            with open(s.image_path, "rb") as f:
                head = f.read(200)
        except OSError:
            continue
        if b"AmiraMesh 3D ASCII" in head:
            n_ascii += 1
            try:
                total_bytes += getsize(s.image_path)
            except OSError:
                pass
    return n_ascii, total_bytes


def run_stitch(
    input_dir: str,
    output_dir: Optional[str] = None,
    *,
    image_fill: bool = True,
    method: str = "mi",
    warp_omega: float = 0.3,
    zblend: bool = True,
    cpd_coarse: bool = True,
    downscale: int = 1,
    use_gpu: Optional[bool] = None,
    gpu_chunk: int = 4,
    workers: Optional[int] = None,
    log=print,
) -> dict:
    """
    Stitch a folder of serial-section tomograms and write the output.

    :param input_dir: folder of section ``.am`` images + ``*_spatialGraph.am``
        (microtubules). Discovered/paired by ``sort_tomogram_files``.
    :param output_dir: output folder (default ``<input_dir>/stitched_output``).
    :param image_fill: also align MT-free regions from image content (needs volumes).
    :param method: image-fill matching metric — ``'mi'`` / ``'grad'`` / ``'ncc'``.
    :param warp_omega: TPS vorticity bound (lower = smoother / fewer whirlpools).
    :param zblend: Z-varying symmetric warp (else one warp per section).
    :param cpd_coarse: CPD multi-seed coarse rotation search (decoy-/±90°-robust).
    :param downscale: integer volume decimation (1 = full resolution).
    :param use_gpu: GPU warp; ``None`` = auto (CUDA→MPS→CPU), ``False`` = force CPU.
    :param gpu_chunk: GPU Z-slice chunk (caps device memory).
    :param workers: CPU processes for image-fill matching (``None`` = cpu_count − 2).
    :param log: line logger (default ``print``); receives each report line.
    :return: dict with output paths and a timing/settings report.
    """
    import os as _os

    t0 = time.perf_counter()
    out = output_dir or join(input_dir, "stitched_output")
    makedirs(out, exist_ok=True)
    if workers is None:
        workers = max(1, (_os.cpu_count() or 4) - 2)
    device = "cpu" if use_gpu is False else accel.pick_device(True)
    gpu_on = device != "cpu"

    report = [
        "#" * 71,
        "#  PANDORICA serial-section stitcher",
        f"#  pandorica v{_VERSION}   {datetime.now():%Y-%m-%d %H:%M:%S}",
        "#" * 71,
        "",
        "--- Settings ---",
        f"input_dir   : {input_dir}",
        f"output_dir  : {out}",
        f"downscale   : {downscale}",
        f"coarse      : {'CPD multi-seed' if cpd_coarse else 'gated sweep'}",
        f"warp_omega  : {warp_omega}  (vorticity bound)",
        f"z-blend     : {zblend}",
        f"image_fill  : {image_fill}  (metric={method})",
        f"GPU warp    : {gpu_on}  (device={device})",
        f"match workers: {workers}",
        "",
    ]
    for line in report:
        log(line)

    def _say(msg):
        report.append(msg)
        log(msg)

    # --- load -------------------------------------------------------------- #
    t = time.perf_counter()
    ds = load_dataset(input_dir)
    coords = ds.coords_list()
    has_vol = all(s.has_volume() for s in ds.sections)
    n_graph = sum(1 for c in coords if len(c) > 0)
    use_mt = n_graph == len(ds)  # every section has a microtubule graph
    t_load = time.perf_counter() - t
    _say(f"--- Loaded {len(ds)} sections ({', '.join(ds.names)}) ---")
    for s in ds.sections:
        _say(
            f"  {s.name:>16}: {s.n_mts:5d} MTs   volume={'yes' if s.has_volume() else 'NO'}"
        )

    # ASCII Amira volumes parse at ~25 MB/s (vs ~5 GB/s for binary) and the
    # pipeline reads each volume twice (image-fill + export). Warn up front so
    # the long wait doesn't look like a hang. Estimate is rough — kept as a
    # range so it doesn't look more precise than it is.
    n_ascii, ascii_bytes = _ascii_volume_summary(ds.sections)
    if n_ascii:
        gb = ascii_bytes / (1024 ** 3)
        # ~25 MB/s parse rate, each volume read ~2× by the pipeline.
        est_s = (ascii_bytes / (25 * 1024 ** 2)) * 2
        est_min = est_s / 60
        _say(
            f"  NOTE: {n_ascii}/{len(ds)} volume(s) are ASCII Amira (~{gb:.1f} GB total)."
        )
        _say(
            f"        ASCII parsing is ~200x slower than binary; expect roughly "
            f"{est_min * 0.7:.0f}-{est_min * 1.3:.0f} min of extra parse time."
        )
        _say(
            "        Re-export volumes as 'AmiraMesh BINARY-LITTLE-ENDIAN' "
            "for ~200x faster loads.\n"
        )

    if not use_mt and not has_vol:
        raise ValueError(
            "Nothing to stitch: sections lack both complete microtubule graphs "
            "AND volumes. Provide *_spatialGraph.am for every section (MT-based "
            "stitching) or .am volumes (image-only stitching)."
        )
    mode = "MT-based" if use_mt else "image-only"
    _say(f"  mode: {mode}\n")

    # --- poses: MT-based or image-only ------------------------------------- #
    t = time.perf_counter()
    if use_mt:
        result = stitch_sections(
            coords, warp_omega_max=warp_omega, cpd_coarse=cpd_coarse
        )
        poses = stitch.result_poses(result)
        mt_warps = stitch.result_warps(result)
        _say(
            "--- Per-interface (coarse / rel / match / warp / QC / coarse-flag / intensity) ---"
        )
        rows = stitch.interface_rows(result, ds)
        for row in rows:
            iv = row["intensity_ok"]
            iv_s = "n/a" if iv is None else ("ok" if iv else "FAIL")
            _say(
                f"  {row['interface']:>16}: coarse={row['coarse_deg']:+7.1f}°  "
                f"rel={row['relative_deg']:+7.2f}°  match={row['match_frac'] * 100:4.0f}%"
                f"  warp_ok={row['warp_ok']!s:5}  qc_ok={row['qc_ok']!s:5}"
                f"  coarse_flag={row['hybrid_flag']!s:5}  intensity={iv_s:4}"
                + (f"  [{row['reasons']}]" if row["reasons"] else "")
            )
        _say(f"  overall accepted: {result.accepted}")
        # Decompose the accept gate so a False is explained, not just reported.
        # accepted = base.accepted AND no coarse-flagged interface AND every
        # large-rotation interface intensity-verified.
        if not result.accepted:
            flagged = [r["interface"] for r in rows if r["hybrid_flag"]]
            int_fail = [
                r["interface"]
                for r in rows
                if r["intensity_ok"] is False
                and abs(r.get("hybrid_deg", 0.0)) >= _INTENSITY_MIN_ANGLE_DEG
            ]
            _say("  why not accepted:")
            if not result.base.accepted:
                _say("    - base QC rejected an interface (see warp_ok / qc_ok above)")
            if flagged:
                _say(
                    "    - coarse rotation flagged as ambiguous (sign / 180° "
                    f"unresolved): {', '.join(flagged)}"
                )
            if int_fail:
                _say(
                    "    - dense-intensity did not confirm a large rotation "
                    f"(≥{_INTENSITY_MIN_ANGLE_DEG:g}°): {', '.join(int_fail)}"
                )
            if not (flagged or int_fail or not result.base.accepted):
                _say("    - (no gating interface failed; check above)")
            _say(
                "    NOTE: flagged = needs human review, NOT proven wrong "
                "(large rotations the pipeline can't auto-verify are flagged)."
            )
    else:
        from pandorica.stitch.image_pose import image_only_poses

        _say("!" * 71)
        _say("! WARNING: image-only stitching (no microtubule graphs).")
        _say("! The inter-section ROTATION is recovered from image content alone.")
        _say("! For symmetric / near-circular cross-sections the image carries no")
        _say("! rotational cue, so the rotation can be unreliable — VERIFY VISUALLY,")
        _say("! and prefer providing *_spatialGraph.am (MT-based) when possible.")
        _say("!" * 71)
        _say("--- Image-only coarse poses (rotation search + block-match shift) ---")
        poses = image_only_poses(
            ds, metric=method if method != "mi" else "ncc", workers=workers, log=_say
        )
        mt_warps = None
    t_stitch = time.perf_counter() - t
    _say("")
    _say("--- Global per-section poses (section 0 = gauge) ---")
    for name, p in zip(ds.names, poses):
        _say(f"  {name:>16}: {_fmt_pose(p)}")
    _say("")

    # --- image-fill (MT-free regions; also the fine warp for image-only) --- #
    image_warps = None
    t_fill = 0.0
    if image_fill and not has_vol:
        _say("--- Image-fill skipped (no volumes) ---\n")
    if image_fill and has_vol:
        t = time.perf_counter()
        image_warps = imgwarp.image_residual_warps(
            ds,
            poses,
            mt_warps=mt_warps,
            method=method,
            workers=workers,
            omega_max=warp_omega,
            progress=lambda m, fr: None,
        )
        t_fill = time.perf_counter() - t
        n_fill = sum(1 for w in image_warps if getattr(w, "accepted", False))
        _say(
            f"--- Image-fill: {n_fill}/{len(image_warps)} interfaces filled "
            f"(metric={method}) ---\n"
        )

    # --- export ------------------------------------------------------------ #
    t = time.perf_counter()
    written = stitch.export_stitched(
        ds,
        poses,
        out,
        downscale=downscale,
        write_volume=has_vol,
        warps=mt_warps,
        warp_zblend=zblend,
        image_warps=image_warps,
        use_gpu=gpu_on,
        gpu_chunk=gpu_chunk,
        progress=lambda m, fr: log(f"  export: {m} ({fr * 100:.0f}%)"),
    )
    t_export = time.perf_counter() - t

    total = time.perf_counter() - t0
    _say("")
    _say("--- Outputs ---")
    for k, v in written.items():
        _say(f"  {k:>8}: {v}")
    _say("")
    _say("--- Compute time ---")
    _say(f"  load          : {t_load:7.1f} s")
    _say(f"  register      : {t_stitch:7.1f} s")
    _say(f"  image-fill    : {t_fill:7.1f} s")
    _say(f"  export/warp   : {t_export:7.1f} s")
    _say(f"  TOTAL         : {total:7.1f} s")

    log_path = join(out, "stitch_log.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(report) + "\n")
    written["log"] = log_path
    written["report"] = "\n".join(report)
    written["seconds"] = total
    return written
