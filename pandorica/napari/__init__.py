#######################################################################
#  Pandorica - Analytical tools for cryo-electron microscopy          #
#                                                                     #
#  https://github.com/RRobert92                                       #
#                                                                     #
#  Robert Kiewisz                                                     #
#  PolyForm Noncommercial License 1.0.0 - see LICENSE                 #
#######################################################################
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Robert Kiewisz

"""
napari plugin for *visually validating* the serial-section stitching pipeline on
real ``.am`` datasets, and for recording coarse-alignment **ground truth** by hand.

Two dock widgets are contributed (see ``napari.yaml``):

* **Serial Section Stitcher** (:class:`._widget.StitchValidatorWidget`) — load a
  folder of section ``.am`` image + ``*_spatialGraph.am`` files, run the stitch,
  overlay the raw vs. aligned microtubules (and optionally the volumes), inspect
  the per-interface QC, and export the stitched volume + merged microtubules.
* **Coarse GT Recorder** (:class:`._widget.CoarseGTWidget`) — step through each
  interface n→n+1, manually rotate/translate the moving bottom-face over the fixed
  top-face until the microtubules line up, and save the per-interface coarse
  ``{angle, tx, ty}`` as ground truth (``coarse_gt.json``).

Run it (the ``tardis`` conda env has both napari and tardis_em)::

    conda run -n tardis napari            # then Plugins → TARDIS Serial Section Stitcher ...
    # or headless launch with a folder pre-loaded:
    conda run -n tardis python -m pandorica.napari <dataset_dir>
"""

__all__ = ["StitchValidatorWidget", "CoarseGTWidget"]


def __getattr__(name):
    # Lazy so importing the headless submodules (_io, _stitch, _image_warp, ...)
    # never pulls in Qt — the production runner must stay headless on a cluster.
    if name in __all__:
        from pandorica.napari import _widget

        return getattr(_widget, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
