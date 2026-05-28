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
Image A–P-polarity rotation hint — the biological SIGN authority.

The C. elegans embryo carries a global,
every-section anterior–posterior polarity — an asymmetric distribution of dense
cytoplasmic material (yolk / organelles / P-granules), a posteriorly-displaced
spindle, etc. Unlike the MT-endpoint constellation (locally symmetric at asters /
bundles) and unlike Fourier-magnitude image methods (180°-ambiguous), this A–P
asymmetry is a **signed** direction present in essentially every section, so it
is the reliable cue for resolving the rotation SIGN that ``rotation_search`` can't.

Implementation: a low-frequency **density-weighted centroid offset** — the vector
from the specimen's geometric centroid to the centroid of its dense material is a
robust proxy for the A–P axis. The angle between two sections' vectors is an
independent (sign-resolved) rotation hint, consumed by ``coarse_fusion``.

CAVEATS (tune on real EM): the "dense = dark" convention (``dense_is_dark``)
depends on stain/contrast; the frame and the ~circular nucleus should be MASKED
(they are symmetric and dilute the signal); work on a low-pass / downsampled
image so the cue reflects bulk cytoplasmic asymmetry, not fine texture. Masking
and the density definition still need tuning on real EM volumes.
"""

from typing import Optional, Tuple

import numpy as np


def _wrap(a: float) -> float:
    return ((float(a) + 180.0) % 360.0) - 180.0


def density_polarity_vector(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dense_is_dark: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Density-weighted centroid offset (the A–P polarity proxy) of one section.

    :param image: 2-D section image (e.g. a Z-projection / boundary-face slice).
    :param mask: optional bool/float mask of the specimen region (exclude frame +
        nucleus). Defaults to the whole image.
    :param dense_is_dark: if True, low intensity = dense material (typical for
        stained EM); else high intensity = dense.
    :return: ``(vector[x, y], angle_deg)`` — offset from the geometric centroid to
        the density-weighted centroid, and its in-plane angle.
    """
    img = np.asarray(image, dtype=float)
    w = (img.max() - img) if dense_is_dark else (img - img.min())
    if mask is not None:
        w = w * np.asarray(mask, dtype=float)
    tot = w.sum()
    if tot <= 0:
        return np.zeros(2), 0.0
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    dcx, dcy = (w * xs).sum() / tot, (w * ys).sum() / tot
    if mask is not None and np.asarray(mask).sum() > 0:
        m = np.asarray(mask, dtype=float)
        gcx, gcy = (m * xs).sum() / m.sum(), (m * ys).sum() / m.sum()
    else:
        gcx, gcy = xs.mean(), ys.mean()
    vec = np.array([dcx - gcx, dcy - gcy])
    return vec, float(np.degrees(np.arctan2(vec[1], vec[0])))


def ap_rotation_hint(
    image_ref: np.ndarray,
    image_mov: np.ndarray,
    mask_ref: Optional[np.ndarray] = None,
    mask_mov: Optional[np.ndarray] = None,
    dense_is_dark: bool = True,
    min_magnitude: float = 1.0,
) -> Optional[float]:
    """
    A–P-polarity rotation hint (deg, signed, mov→ref) for one interface.

    The rotation that aligns the moving section to the reference rotates the
    moving A–P vector onto the reference's, i.e. ``hint = φ_ref − φ_mov``.

    :param min_magnitude: if either section's polarity vector is shorter than this
        (in px), the A–P cue is too weak/symmetric to trust → return ``None`` (the
        caller then abstains / relies on other cues).
    :return: signed rotation hint, or ``None`` if the cue is unreliable.
    """
    v_ref, phi_ref = density_polarity_vector(image_ref, mask_ref, dense_is_dark)
    v_mov, phi_mov = density_polarity_vector(image_mov, mask_mov, dense_is_dark)
    if np.linalg.norm(v_ref) < min_magnitude or np.linalg.norm(v_mov) < min_magnitude:
        return None
    return _wrap(phi_ref - phi_mov)
