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
Shared pytest fixtures for the tomogram-stitching test suite.

Design rules:
    * Tests must run green WITHOUT the C.elegans dataset (CI has no data) — any
      test that needs it is marked ``@pytest.mark.data`` and skipped when absent.
    * Spatial graphs ONLY are loaded here. The multi-GB ``.rec`` volumes are
      never opened in tests.
"""

import os
from glob import glob
from os.path import isdir, join, basename

import numpy as np
import pytest

from tardis_em_analysis.utils import pc_median_dist

# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
# sec09–13 dual-axis plastic-ET spatial graphs. Override with the
# TARDIS_TEST_DATA env var; otherwise fall back to the known local path.
# CI leaves this unset / absent → data-marked tests skip cleanly.
_DEFAULT_DATA_DIR = "/Users/robertkiewisz/Downloads/C.elegans_FemalePN"
DATA_DIR = os.environ.get("TARDIS_TEST_DATA", _DEFAULT_DATA_DIR)

# A fixed seed so synthetic cases and any randomized assertions are reproducible.
SEED = 1729


def pytest_configure(config):
    """Register custom markers so unknown-marker warnings don't fail CI."""
    config.addinivalue_line(
        "markers",
        "data: test requires the C.elegans_FemalePN dataset (skipped if absent)",
    )


def _data_available() -> bool:
    return isdir(DATA_DIR) and len(_spatial_graph_paths()) > 0


def _spatial_graph_paths():
    """Sorted sec09–13 spatial-graph .am paths (volumes are NOT touched)."""
    return sorted(glob(join(DATA_DIR, "*_spatialGraph.am")))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded NumPy random generator — reproducible across the whole session."""
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def section_graphs() -> "dict[str, np.ndarray]":
    """
    Load the sec09–13 spatial graphs as ``{section_name: [N, 4]}`` arrays of
    ``[id, x, y, z]``. Skips the whole test if the dataset is unavailable.

    Loads graphs only — ``ImportDataFromAmira`` is called with the spatial-graph
    path alone, so the multi-GB volume is never read.
    """
    if not _data_available():
        pytest.skip(f"dataset not found at {DATA_DIR} (set TARDIS_TEST_DATA)")

    # Imported lazily so collection doesn't hard-depend on tardis_em at import time.
    from tardis_em.utils.load_data import ImportDataFromAmira

    graphs = {}
    for path in _spatial_graph_paths():
        name = basename(path).replace("_spatialGraph.am", "")
        coords = ImportDataFromAmira(src_am=path).get_segmented_points()
        if coords is not None and len(coords) > 0:
            graphs[name] = np.asarray(coords)
    if not graphs:
        pytest.skip("no non-empty spatial graphs found in dataset")
    return graphs


@pytest.fixture(scope="session")
def one_graph(section_graphs) -> np.ndarray:
    """A single representative spatial graph ([N, 4]) for lightweight tests."""
    return next(iter(section_graphs.values()))


@pytest.fixture(scope="session")
def rho(one_graph) -> float:
    """
    Median nearest-neighbour spacing (ρ) of one section, in coordinate units.
    ρ is the scale unit that makes downstream thresholds dataset-portable.
    """
    return float(pc_median_dist(one_graph[:, 1:4]))
