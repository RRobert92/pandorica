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
Headless launcher: open napari with both stitch widgets docked, optionally
pre-pointing them at a dataset folder.

    conda run -n tardis python -m pandorica.napari [dataset_dir]
"""

import sys

import napari

from pandorica.napari._widget import (
    CoarseGTWidget,
    StitchValidatorWidget,
)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    folder = argv[0] if argv else None

    viewer = napari.Viewer(title="TARDIS Serial Section Stitcher")
    validator = StitchValidatorWidget(viewer)
    gt = CoarseGTWidget(viewer)
    viewer.window.add_dock_widget(
        validator, name="Serial Section Stitcher", area="right"
    )
    viewer.window.add_dock_widget(gt, name="Coarse GT Recorder", area="right")

    if folder:
        validator.folder_lbl.setText(folder)
        gt.folder_lbl.setText(folder)
        validator._load()

    napari.run()


if __name__ == "__main__":
    main()
