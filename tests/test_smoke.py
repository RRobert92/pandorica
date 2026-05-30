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
Smoke tests — confirm the package imports and the test harness is wired.
These run green with or without the C.elegans dataset.
"""

import numpy as np
import pytest


def test_package_imports():
    """The package and its core reusables import cleanly."""
    import pandorica.io.amira  # noqa: F401
    from pandorica.utils.pointcloud import pc_median_dist  # noqa: F401
    from pandorica.stitch.matching.mt_endpoints import (  # noqa: F401
        extract_boundary_endpoints,
    )


def test_rng_is_seeded(rng):
    """The seeded RNG fixture is deterministic."""
    a = rng.random(5)
    other = np.random.default_rng(1729).random(5)
    assert np.allclose(a, other)


@pytest.mark.data
def test_dataset_graphs_load(section_graphs, rho):
    """When the dataset is present, graphs load as [N, 4] and ρ is positive."""
    assert len(section_graphs) >= 1
    for name, coords in section_graphs.items():
        assert coords.ndim == 2 and coords.shape[1] == 4, name
    assert rho > 0.0
