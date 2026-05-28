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

"""Tests for the per-interface QC certificate (``qc.py``)."""

import numpy as np
import pytest

from pandorica.stitch.pipeline import qc
from pandorica.stitch.transform.diagnostics import FieldCertificate


def _cert(passed=True):
    return FieldCertificate(
        min_det_j=0.8 if passed else -0.5,
        max_abs_vorticity=0.2 if passed else 5.0,
        ow_min=0.0 if passed else -3.0,
        eps=0.05,
        omega_max=1.0,
        passed=passed,
    )


def _good_conf():
    return {"match_fraction": 0.9, "shift_incoherence_rho": 0.1}


_ALIGNED = (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]))


# --------------------------------------------------------------------------- #
# Tangent discontinuity
# --------------------------------------------------------------------------- #
def test_tangent_discontinuity_values():
    assert qc.tangent_discontinuity_deg([[1, 0]], [[1, 0]]) == pytest.approx(0.0)
    assert qc.tangent_discontinuity_deg([[1, 0]], [[-1, 0]]) == pytest.approx(0.0)
    assert qc.tangent_discontinuity_deg([[1, 0]], [[0, 1]]) == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# Accept / flag
# --------------------------------------------------------------------------- #
def test_clean_interface_accepted():
    r = qc.assess_interface(_cert(True), _good_conf(), *_ALIGNED)
    assert r.accepted
    assert r.reasons == []


def test_bad_warp_is_flagged():
    r = qc.assess_interface(_cert(False), _good_conf(), *_ALIGNED)
    assert not r.accepted
    assert any("diffeomorphism" in s for s in r.reasons)


def test_low_match_fraction_flagged():
    conf = {"match_fraction": 0.1, "shift_incoherence_rho": 0.1}
    r = qc.assess_interface(_cert(True), conf, *_ALIGNED)
    assert not r.accepted
    assert any("match fraction" in s for s in r.reasons)


def test_incoherent_shifts_flagged():
    conf = {"match_fraction": 0.9, "shift_incoherence_rho": 5.0}
    r = qc.assess_interface(_cert(True), conf, *_ALIGNED)
    assert not r.accepted
    assert any("incoherent" in s for s in r.reasons)


def test_tangent_is_advisory_not_gated():
    # A large tangent discontinuity is REPORTED but no longer flags the interface
    # (boundary MT tangents are unreliable — see qc docstring / red-team).
    ref = np.array([[1.0, 0.0]])
    mov = np.array([[0.0, 1.0]])  # 90° off
    r = qc.assess_interface(_cert(True), _good_conf(), ref, mov)
    assert r.accepted  # not flagged on tangent alone
    assert r.tangent_discontinuity_deg == pytest.approx(90.0)  # but still reported
    assert not any("tangent" in s for s in r.reasons)


def test_incoherence_threshold_passes_real_nonrigid():
    # ~1ρ incoherence (real non-rigid the warp absorbs) must pass the 2.5ρ gate.
    conf = {"match_fraction": 0.7, "shift_incoherence_rho": 1.3}
    r = qc.assess_interface(_cert(True), conf, *_ALIGNED)
    assert r.accepted
