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
Napari reader plugin for AmiraMesh ``.am`` files.

Registered via the manifest under ``contributions.readers``; napari calls
:func:`napari_get_reader` when the user opens (or drag-drops) a ``.am`` file.
Returns ``None`` for any file we can't read, so napari falls through to other
readers.

Classification (matches :func:`pandorica.io.amira.sort_tomogram_files`):

* ``Lattice`` in the header → image volume → returned as an Image layer.
* ``VERTEX`` / ``EDGE`` / ``HxSpatialGraph`` (or filename ending in
  ``_spatialGraph.am``) → spatial graph → returned as a Shapes layer with
  each filament rendered as a ``"path"``.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


PathOrPaths = Union[str, List[str]]
LayerData = Tuple[object, dict, str]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]


def _classify(path: str) -> str:
    """Return ``"graph"`` for spatial graphs, ``"image"`` for volumes, ``""`` if neither."""
    lp = path.lower()
    if lp.endswith("_spatialgraph.am"):
        return "graph"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(8192)
    except OSError:
        return ""
    if "Lattice" in head:
        return "image"
    if "VERTEX" in head or "EDGE" in head or "HxSpatialGraph" in head:
        return "graph"
    return ""


def _read_graph_layer(path: str) -> LayerData:
    """Read a spatial graph; return a Shapes layer with one path per filament."""
    from pandorica.io.amira import read_segmented_points
    from pandorica.napari._geometry import coords_to_paths_zyx

    coords = read_segmented_points(path)
    if coords is None or len(coords) == 0:
        # Still return a (visible) empty Shapes layer so the user knows the
        # file was recognised; nothing to draw.
        return ([], {"name": os.path.basename(path), "shape_type": "path"}, "shapes")
    paths = coords_to_paths_zyx(coords)
    if not paths:
        return ([], {"name": os.path.basename(path), "shape_type": "path"}, "shapes")
    return (
        paths,
        {
            "name": os.path.basename(path),
            "shape_type": "path",
            "edge_color": "yellow",
            "edge_width": 2.0,
            "opacity": 0.9,
        },
        "shapes",
    )


def _read_volume_layer(path: str) -> LayerData:
    """Read an AmiraMesh image lattice; return an Image layer with isotropic scale."""
    from pandorica.io.amira import read_amira_volume

    img, px_A, _physical, _transform = read_amira_volume(path)
    # napari axes are (z, y, x); scale is per-axis in physical units (Å here).
    return (
        img,
        {
            "name": os.path.basename(path),
            "scale": (float(px_A), float(px_A), float(px_A)),
            "colormap": "gray",
            "blending": "additive",
            "opacity": 0.85,
        },
        "image",
    )


def _read_one(path: str) -> Optional[LayerData]:
    kind = _classify(path)
    if kind == "graph":
        return _read_graph_layer(path)
    if kind == "image":
        return _read_volume_layer(path)
    return None


def _reader_function(path: PathOrPaths) -> List[LayerData]:
    paths = [path] if isinstance(path, str) else list(path)
    out: List[LayerData] = []
    for p in paths:
        layer = _read_one(p)
        if layer is not None:
            out.append(layer)
    return out


def napari_get_reader(path: PathOrPaths) -> Optional[ReaderFunction]:
    """npe2 entry point. Return ``_reader_function`` if we can handle ``path``."""
    paths = [path] if isinstance(path, str) else list(path)
    if not paths:
        return None
    # Only claim ``.am`` files we can actually classify. Stay silent on the rest
    # so other reader plugins get a chance.
    for p in paths:
        if not str(p).lower().endswith(".am"):
            return None
        if not _classify(p):
            return None
    return _reader_function
