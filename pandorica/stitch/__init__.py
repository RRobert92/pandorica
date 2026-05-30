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
Serial-section tomogram stitcher (MT-based, with image-only fallback).

This is the self-contained stitching core. The headless entry point is
``run_stitch``; the napari plugin in ``pandorica.napari`` is a thin UI that
calls into here and can be removed without affecting the CLI.

Layout:
    cli           headless ``run_stitch`` orchestration + reporting
    dataset       serial-section data model (``Dataset``, ``Section``,
                  ``load_dataset``); format-level Amira I/O lives in
                  ``pandorica.io.amira``.
    stitch        result accessors + stitched-output export (``export_stitched``)
    image_pose    image-only coarse poses (no-MT fallback)
    image_warp    image-fill residual warps for MT-free regions
    geometry      pose <-> pixel math, boundary landmarks
    match         image block-matching metrics
    accel         GPU/CPU device selection
    coarse/       coarse rotation search (CPD, sweep, AP-polarity, fusion, hybrid)
    matching/     endpoint matcher + low-level MT-endpoint geometry
    transform/    rigid solve, guarded TPS warp, slice applier, scale, diagnostics
    pipeline/     core.register_section_stack (engine), stitcher.stitch_sections
                  (full pipeline), QC
"""

__all__ = ["run_stitch"]


def __getattr__(name):
    # Lazy so ``import ...serial_stitch`` stays cheap (run_stitch pulls torch/IO).
    if name == "run_stitch":
        from pandorica.stitch.cli import run_stitch

        return run_stitch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
