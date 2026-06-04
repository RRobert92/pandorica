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
Tests for the MT<->image dual-chain cross-check (``image_pose.reconcile_image_mt``).

The reconcile logic is exercised through the ``image_candidates`` injection seam, so no
volumes are loaded: we fabricate the image candidate payload (the ``on_interface`` dict)
and the MT confidence rows directly, then assert on which source each component is taken
from and whether the interface is flagged. Two algebra round-trips lock in the pose
surgery the reconcile relies on (``_pose_center_shift`` inverts ``centroid_pose``; an
agreeing cross-check reproduces the MT chain exactly).
"""

import numpy as np
import pytest

from pandorica.stitch import geometry as geo
from pandorica.stitch.transform import solver as sv
from pandorica.stitch import image_pose as ip


class _DS:
    """Minimal stand-in: reconcile_image_mt only reads ``len(dataset.sections)``."""

    def __init__(self, n):
        self.sections = list(range(n))


def _cand(label, rot, shift, agree, trans_ok, branch_amb, center=(100.0, 80.0), pxf=1.0):
    return dict(
        label=label, rot=float(rot), shift=np.asarray(shift, float),
        center=np.asarray(center, float), agree=float(agree),
        trans_ok=bool(trans_ok), branch_ambiguous=bool(branch_amb),
        pxf=float(pxf), rot_geom=None,
    )


def _mt_poses(rels):
    """Absolute MT poses from per-interface ``(angle, shift_px, center)`` triples."""
    poses = [dict(sv.IDENTITY)]
    for ang, sh, c in rels:
        poses.append(sv.compose_poses(
            poses[-1], geo.centroid_pose(float(ang), float(sh[0]), float(sh[1]),
                                         np.asarray(c, float))))
    return poses


def _one(*, mt_angle, mt_shift, img_rot, img_shift, agree, trans_ok=True,
         branch_amb=False, match=0.8, qc=True, flag=False, center=(100.0, 80.0)):
    """Run reconcile on a single fabricated interface; return the lone report dict."""
    ds = _DS(2)
    poses = _mt_poses([(mt_angle, mt_shift, center)])
    rows = [dict(match_frac=match, qc_ok=qc, hybrid_flag=flag)]
    cands = [_cand("s0->s1", img_rot, img_shift, agree, trans_ok, branch_amb, center)]
    new_poses, reports = ip.reconcile_image_mt(ds, poses, rows, image_candidates=cands)
    return new_poses, reports[0]


# --------------------------------------------------------------------------- #
# Algebra the reconcile depends on
# --------------------------------------------------------------------------- #
def test_pose_center_shift_inverts_centroid_pose():
    rng = np.random.default_rng(0)
    for _ in range(20):
        ang = rng.uniform(-180, 180)
        tx, ty = rng.uniform(-50, 50, 2)
        c = rng.uniform(-200, 200, 2)
        s = rng.uniform(0.8, 1.2)
        p = geo.centroid_pose(ang, tx, ty, c, scale=s)
        assert np.allclose(ip._pose_center_shift(p, c), [tx, ty], atol=1e-9)


def test_reconcile_noop_reproduces_mt_poses_when_image_agrees():
    # Image candidate equals the MT relative exactly -> no conflict -> the recomposed
    # chain must reproduce the input MT poses (to float precision).
    rels = [(2.0, (5.0, -3.0), (100.0, 80.0)),
            (-1.5, (-4.0, 7.0), (90.0, 110.0)),
            (3.2, (6.0, 2.0), (120.0, 70.0))]
    ds = _DS(len(rels) + 1)
    poses = _mt_poses(rels)
    rows = [dict(match_frac=0.8, qc_ok=True, hybrid_flag=False) for _ in rels]
    cands = [_cand(f"s{k}->s{k+1}", ang, sh, agree=0.5, trans_ok=True, branch_amb=False,
                   center=c) for k, (ang, sh, c) in enumerate(rels)]
    new_poses, reports = ip.reconcile_image_mt(ds, poses, rows, image_candidates=cands)
    assert all(r["rot_src"] == "mt" and r["t_src"] == "mt" and not r["flagged"]
               for r in reports)
    for a, b in zip(new_poses, poses):
        assert a["Angle"] == pytest.approx(b["Angle"], abs=1e-7)
        assert a["Tx"] == pytest.approx(b["Tx"], abs=1e-6)
        assert a["Ty"] == pytest.approx(b["Ty"], abs=1e-6)


# --------------------------------------------------------------------------- #
# Rotation / sign component
# --------------------------------------------------------------------------- #
def test_rotation_within_tol_keeps_mt():
    _, r = _one(mt_angle=2.0, mt_shift=(5.0, -3.0), img_rot=3.0, img_shift=(5.0, -3.0),
                agree=0.5)
    assert r["rot_src"] == "mt" and not r["rot_conflict"]
    assert r["rot_final"] == pytest.approx(2.0, abs=1e-6)


def test_rotation_conflict_image_rescues_flagged_mt():
    # MT flipped (+91) and flagged; image is sign-confident -> image wins.
    new, r = _one(mt_angle=91.0, mt_shift=(5.0, -3.0), img_rot=-89.0,
                  img_shift=(5.0, -3.0), agree=0.5, flag=True)
    assert r["rot_src"] == "img" and r["rot_conflict"] and r["flagged"]
    assert r["rot_final"] == pytest.approx(-89.0, abs=1e-6)


def test_rotation_conflict_both_confident_defaults_to_mt_on_low_image_fraction():
    # Both sign-confident, image inlier fraction (0.16) < MT match (0.9) -> keep MT,
    # but raise the flag (the conservative calibration bias).
    _, r = _one(mt_angle=91.0, mt_shift=(5.0, -3.0), img_rot=-89.0, img_shift=(5.0, -3.0),
                agree=0.16, match=0.9, flag=False)
    assert r["rot_src"] == "mt" and r["rot_conflict"] and r["flagged"]
    assert r["rot_final"] == pytest.approx(91.0, abs=1e-6)


def test_rotation_conflict_both_confident_image_wins_on_higher_fraction():
    # Both sign-confident; here the image fraction beats MT's match -> image wins.
    _, r = _one(mt_angle=91.0, mt_shift=(5.0, -3.0), img_rot=-89.0, img_shift=(5.0, -3.0),
                agree=0.5, match=0.1, flag=False)
    assert r["rot_src"] == "img" and r["flagged"]
    assert r["rot_final"] == pytest.approx(-89.0, abs=1e-6)


def test_rotation_conflict_neither_confident_keeps_mt():
    # MT flagged (sign not ok) AND image branch-ambiguous (sign not ok) -> keep MT.
    _, r = _one(mt_angle=91.0, mt_shift=(5.0, -3.0), img_rot=-89.0, img_shift=(5.0, -3.0),
                agree=0.5, branch_amb=True, flag=True)
    assert r["rot_src"] == "mt" and r["rot_conflict"] and r["flagged"]
    assert r["rot_final"] == pytest.approx(91.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Translation component (gated on the image's own confidence)
# --------------------------------------------------------------------------- #
def test_translation_image_abstain_is_silent():
    # Image abstained (trans_ok False): even a big shift difference must NOT flag.
    _, r = _one(mt_angle=2.0, mt_shift=(5.0, -3.0), img_rot=2.0, img_shift=(40.0, 50.0),
                agree=0.05, trans_ok=False)
    assert r["t_src"] == "mt" and not r["t_conflict"] and not r["flagged"]


def test_translation_overrides_when_mt_weak():
    # Confident image, big shift gap, MT match below the trust floor -> image wins.
    new, r = _one(mt_angle=2.0, mt_shift=(5.0, -3.0), img_rot=2.0, img_shift=(40.0, 50.0),
                  agree=0.5, match=0.2)
    assert r["t_src"] == "img" and r["t_conflict"] and r["flagged"]
    # committed center-shift must equal the image shift.
    c = np.array([100.0, 80.0])
    rel = sv.compose_poses(sv.invert_pose(new[0]), new[1])
    assert np.allclose(ip._pose_center_shift(rel, c), [40.0, 50.0], atol=1e-6)


def test_translation_kept_but_flagged_when_mt_strong():
    # Confident image, big shift gap, but MT strong and image fraction lower -> keep MT
    # translation, raise the flag for review.
    new, r = _one(mt_angle=2.0, mt_shift=(5.0, -3.0), img_rot=2.0, img_shift=(40.0, 50.0),
                  agree=0.16, match=0.8)
    assert r["t_src"] == "mt" and r["t_conflict"] and r["flagged"]
    c = np.array([100.0, 80.0])
    rel = sv.compose_poses(sv.invert_pose(new[0]), new[1])
    assert np.allclose(ip._pose_center_shift(rel, c), [5.0, -3.0], atol=1e-6)
