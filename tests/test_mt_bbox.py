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
Tests for ``stitch._mt_bbox`` — the MT bounding-box canvas sizer used by
``export_stitched(trim_to_mts=True)`` to drop empty-corner pixels.

What must hold:

* The trimmed canvas encloses every microtubule after pose transform (no MT is
  lost off-canvas).
* For a stack with section-to-section drift, the MT bbox is meaningfully
  smaller than the section corner bbox (this is the whole point).
* The fallback for an MT-free stack returns the corner bbox.
* ``mt_pad_frac`` is applied symmetrically on both sides of each axis.
"""

import numpy as np

from pandorica.stitch.dataset import Dataset, Section
from pandorica.stitch.stitch import _corner_bbox, _mt_bbox
from pandorica.stitch.transform.solver import apply_pose


def _make_section(name: str, idx: int, coords_xy_A: np.ndarray, px: float = 1.0) -> Section:
    """Build a Section with the given MT (x, y) coords in Å; id and z are stubs."""
    coords = np.zeros((len(coords_xy_A), 4), dtype=float)
    coords[:, 0] = np.arange(len(coords_xy_A))  # id
    coords[:, 1:3] = coords_xy_A
    coords[:, 3] = 0.0
    return Section(
        name=name,
        index=idx,
        image_path=None,
        coord_path=None,
        coords=coords,
        pixel_size=px,
    )


def _pose_px(angle_deg: float = 0.0, tx: float = 0.0, ty: float = 0.0, scale: float = 1.0):
    return {"Angle": angle_deg, "Tx": tx, "Ty": ty, "Scale": scale}


def _drifted_three_section_dataset():
    """
    Three sections with MTs clustered near the centre, poses drifting outward.

    Each section has 100 MT points in a 200×200 px patch; section 1 sits at the
    origin, section 2 shifted (+3000, -2000) px, section 3 shifted (+1500, +1500).
    With a section frame of 4096×4096, the corner bbox is much larger than the
    MT bbox — the difference is exactly the savings the trim option targets.
    """
    rng = np.random.default_rng(42)
    px = 10.0  # Å/pixel — so coords in Å are 10× the pixel coords
    base_xy_px = rng.uniform(-100, 100, size=(100, 2)) + np.array([2048, 2048])
    coords_A = base_xy_px * px

    s0 = _make_section("sec0", 0, coords_A, px)
    s1 = _make_section("sec1", 1, coords_A, px)
    s2 = _make_section("sec2", 2, coords_A, px)
    ds = Dataset(folder="/tmp/fake", sections=[s0, s1, s2])

    poses_px = [
        _pose_px(tx=0.0, ty=0.0),
        _pose_px(tx=3000.0, ty=-2000.0),
        _pose_px(tx=1500.0, ty=1500.0),
    ]
    return ds, poses_px, px


def test_mt_bbox_encloses_every_transformed_mt():
    ds, poses_px, px = _drifted_three_section_dataset()
    hc, wc, offset = _mt_bbox(ds, poses_px, px=px, pad_frac=0.0)

    for s, p in zip(ds.sections, poses_px):
        if len(s.coords) == 0:
            continue
        xy_px = s.coords[:, 1:3] / px
        warped = apply_pose(p, xy_px) + offset  # canvas coords
        assert (warped[:, 0] >= 0).all() and (warped[:, 0] <= wc).all(), (
            f"section {s.name}: MTs x outside [0, {wc}]"
        )
        assert (warped[:, 1] >= 0).all() and (warped[:, 1] <= hc).all(), (
            f"section {s.name}: MTs y outside [0, {hc}]"
        )


def test_mt_bbox_smaller_than_corner_bbox_on_drifted_stack():
    ds, poses_px, px = _drifted_three_section_dataset()
    # 4096-px section frame — typical real tomogram face.
    full_hc, full_wc, _ = _corner_bbox(poses_px, h=4096, w=4096)
    mt_hc, mt_wc, _ = _mt_bbox(ds, poses_px, px=px, pad_frac=0.05)
    full_area = full_hc * full_wc
    mt_area = mt_hc * mt_wc
    assert mt_area < full_area, (
        f"trim did not shrink canvas: corner={full_hc}x{full_wc} mt={mt_hc}x{mt_wc}"
    )
    # MTs occupy ≲ 10 % of a drifted multi-section canvas in this fixture.
    # If the fixture changes, the constant should track — keep the assertion
    # loose enough that small changes don't make it flake.
    assert mt_area < 0.5 * full_area, (
        f"trim should have produced a much smaller canvas: ratio = {mt_area / full_area:.2f}"
    )


def test_mt_bbox_pad_frac_scales_canvas_symmetrically():
    ds, poses_px, px = _drifted_three_section_dataset()
    hc0, wc0, _ = _mt_bbox(ds, poses_px, px=px, pad_frac=0.0)
    hc1, wc1, _ = _mt_bbox(ds, poses_px, px=px, pad_frac=0.10)
    # 10 % pad on each side → each axis grows by ~20 % (10 % per side).
    assert hc1 > hc0 and wc1 > wc0
    ratio_h = hc1 / max(hc0, 1)
    ratio_w = wc1 / max(wc0, 1)
    assert 1.18 < ratio_h < 1.22, f"H pad ratio {ratio_h:.3f} not ≈1.20"
    assert 1.18 < ratio_w < 1.22, f"W pad ratio {ratio_w:.3f} not ≈1.20"


def test_mt_bbox_falls_back_to_corner_when_no_mts():
    """All sections have empty graphs → fall back to corner bbox."""
    px = 10.0
    empty = np.zeros((0, 4), dtype=float)
    sections = [
        Section(name=f"sec{i}", index=i, image_path=None, coord_path=None, coords=empty, pixel_size=px)
        for i in range(3)
    ]
    ds = Dataset(folder="/tmp/fake_empty", sections=sections)
    poses_px = [_pose_px(), _pose_px(tx=2000.0), _pose_px(tx=-1500.0)]

    fallback_hw = (256, 320)
    hc_corner, wc_corner, off_corner = _corner_bbox(poses_px, *fallback_hw)
    hc_mt, wc_mt, off_mt = _mt_bbox(
        ds, poses_px, px=px, pad_frac=0.05, fallback_hw=fallback_hw
    )
    assert (hc_mt, wc_mt) == (hc_corner, wc_corner)
    np.testing.assert_allclose(off_mt, off_corner)


def test_mt_bbox_no_fallback_returns_zero_when_no_mts():
    px = 10.0
    sections = [
        Section(name="empty", index=0, image_path=None, coord_path=None,
                coords=np.zeros((0, 4)), pixel_size=px)
    ]
    ds = Dataset(folder="/tmp/empty", sections=sections)
    poses_px = [_pose_px()]
    hc, wc, offset = _mt_bbox(ds, poses_px, px=px)
    assert (hc, wc) == (0, 0)
    np.testing.assert_array_equal(offset, np.zeros(2))


def test_mt_bbox_handles_section_with_no_mts_among_others():
    """One empty section in a stack does not break the bbox over the others."""
    ds, poses_px, px = _drifted_three_section_dataset()
    ds.sections[1].coords = np.zeros((0, 4), dtype=float)  # blank out the middle one

    hc, wc, offset = _mt_bbox(ds, poses_px, px=px, pad_frac=0.0)
    # Every non-empty section's MTs must still fit inside the bbox.
    for s, p in zip(ds.sections, poses_px):
        if len(s.coords) == 0:
            continue
        xy_px = s.coords[:, 1:3] / px
        warped = apply_pose(p, xy_px) + offset
        assert (warped[:, 0] >= 0).all() and (warped[:, 0] <= wc).all()
        assert (warped[:, 1] >= 0).all() and (warped[:, 1] <= hc).all()
