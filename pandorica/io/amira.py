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
Native AmiraMesh I/O for pandorica.

Replaces the previous dependency on ``tardis_em.utils.load_data`` /
``export_data`` with a self-contained reader/writer for the two file kinds
the stitcher touches:

    * **Image lattice** (``.am`` volume) — :func:`read_amira_volume`,
      :func:`write_amira_volume_streamed`.
    * **Spatial graph** (``*_spatialGraph.am``) — :func:`read_spatial_graph`,
      :func:`write_spatial_graph`, with the :class:`SpatialGraph` container.

Both the reader and writer are **lossless** with respect to the rich
AmiraMesh feature set: any int/float field declared per ``VERTEX``,
``EDGE`` or ``POINT`` is decoded into a typed dict on read, and either
written back verbatim (when handed a :class:`SpatialGraph`) or supplied
explicitly via the writer's ``*_int_fields`` / ``*_float_fields`` kwargs.

The parser is single-pass and confines the ``@N`` lookup to the data
section, which removes a class of misclassification bugs the older code
was prone to: Amira's ``Parameters`` block embeds literal strings like
``@5`` in its TCL history log, and a naïve ``list.index("@5")`` could
pick those up before the real data marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from os import listdir
from os import path as _osp
from os.path import isfile, join, split, splitext
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Headers mix ASCII with a single ISO-8859-1 byte (0xC5 == "Å"). Reading in
# 8859-1 means that byte round-trips cleanly without surrogate hand-holding.
_ENC = "iso-8859-1"

_DATA_MARKER_TXT = "# Data section follows"
_DATA_MARKER_BIN = b"# Data section follows\n"

# Field declaration: ``VERTEX|EDGE|POINT { float|int[K]? Name } @N``.
_DECL_RE = re.compile(
    r"^\s*(VERTEX|EDGE|POINT)\s*\{\s*"
    r"(float|int)(?:\[(\d+)\])?\s+(\w+)\s*\}\s*@(\d+)\s*$"
)
_DEFINE_RE = re.compile(r"^\s*define\s+(VERTEX|EDGE|POINT)\s+(\d+)\s*$")
_AT_LINE_RE = re.compile(r"^@(\d+)\s*$")


class AmiraFormatError(ValueError):
    """Raised when a ``.am`` file does not match the expected schema."""


# ---------------------------------------------------------------------------
# SpatialGraph container
# ---------------------------------------------------------------------------


@dataclass
class SpatialGraph:
    """In-memory representation of an AmiraMesh ``*_spatialGraph.am`` file.

    The four core fields (``vertices``, ``edge_connectivity``,
    ``num_edge_points``, ``point_coordinates``) match the Amira data model.
    Every optional per-element scalar declared in the file lands in a typed
    dict — ``vertex_int_fields``, ``edge_float_fields`` etc. — so reads are
    lossless and writes preserve whatever the caller chooses to keep.

    Coordinates are normalised to **Ångströms** on read regardless of the
    file's declared ``Coordinates`` unit; writes always emit ``"Å"``.
    """

    vertices: np.ndarray  # (V, 3) float64 — Å, upcast from any source dtype
    edge_connectivity: np.ndarray  # (E, 2) int32
    num_edge_points: np.ndarray  # (E,) int32
    point_coordinates: np.ndarray  # (P, 3) float64 — Å
    vertex_int_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    vertex_float_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    edge_int_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    edge_float_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    point_int_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    point_float_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    coordinate_unit: str = "A"

    # ----- convenience ------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.num_edge_points.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.point_coordinates.shape[0])

    def segmented_points(self) -> np.ndarray:
        """Return ``[seg_id, x, y, z]`` per polyline point as ``(P, 4)``.

        Mirrors the legacy ``ImportDataFromAmira.get_segmented_points`` shape
        used throughout the stitcher.
        """
        if self.n_edges == 0:
            return np.zeros((0, 4), dtype=float)
        seg = np.repeat(
            np.arange(self.n_edges, dtype=float),
            self.num_edge_points.astype(int),
        )
        if seg.shape[0] != self.n_points:
            raise AmiraFormatError(
                f"NumEdgePoints sum ({seg.shape[0]}) does not match POINT "
                f"count ({self.n_points})"
            )
        return np.column_stack((seg, self.point_coordinates.astype(float)))

    @classmethod
    def from_segmented_points(cls, coords: np.ndarray) -> "SpatialGraph":
        """Build a chain-only graph (one polyline per segment) from ``(N, 4)``.

        Each unique segment id becomes one edge with degree-1 endpoints
        (two new vertices per segment, indexed ``2k`` and ``2k+1``). Used by
        the stitcher to turn its merged ``(N, 4)`` array into a writeable
        graph object without losing topology.
        """
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(
                f"coords must be (N, 4) [seg_id, x, y, z]; got {coords.shape}"
            )
        coords = _drop_single_point_filaments(coords)
        coords = _reorder_segments_id(coords)

        if coords.shape[0] == 0:
            return cls(
                vertices=np.zeros((0, 3), dtype=np.float32),
                edge_connectivity=np.zeros((0, 2), dtype=np.int32),
                num_edge_points=np.zeros((0,), dtype=np.int32),
                point_coordinates=np.zeros((0, 3), dtype=np.float32),
            )

        seg_ids = coords[:, 0].astype(np.int64)
        change = np.flatnonzero(np.diff(seg_ids)) + 1
        start_idx = np.concatenate(([0], change))
        end_idx = np.concatenate((change, [coords.shape[0]]))
        n_segments = int(start_idx.shape[0])

        # Vertices: start_xyz, end_xyz, interleaved per segment.
        seg_starts = coords[start_idx, 1:4]
        seg_ends = coords[end_idx - 1, 1:4]
        vertices = np.empty((n_segments * 2, 3), dtype=np.float32)
        vertices[0::2] = seg_starts
        vertices[1::2] = seg_ends

        edge_connectivity = np.column_stack(
            (np.arange(0, 2 * n_segments, 2), np.arange(1, 2 * n_segments, 2))
        ).astype(np.int32)
        num_edge_points = (end_idx - start_idx).astype(np.int32)
        point_coordinates = coords[:, 1:4].astype(np.float32)

        return cls(
            vertices=vertices,
            edge_connectivity=edge_connectivity,
            num_edge_points=num_edge_points,
            point_coordinates=point_coordinates,
        )


# ---------------------------------------------------------------------------
# Header parsing helpers
# ---------------------------------------------------------------------------


def _peek_header(p: str, n: int = 16384) -> str:
    """Decode the first ``n`` bytes as ISO-8859-1 for format sniffing."""
    with open(p, "rb") as f:
        return f.read(n).decode(_ENC, errors="ignore")


def _parse_header_lines(
    lines: List[str],
) -> Tuple[Dict[str, int], "Dict[int, Tuple[str, int, np.dtype, str]]"]:
    """Return ``(counts, specs)`` for every ``@N`` block declared in ``lines``.

    ``counts`` maps ``VERTEX|EDGE|POINT`` to its declared row count.
    ``specs[N]`` is ``(section, ncols, dtype, field_name)``.
    """
    counts: Dict[str, int] = {}
    specs: Dict[int, Tuple[str, int, np.dtype, str]] = {}
    for line in lines:
        m = _DEFINE_RE.match(line)
        if m:
            counts[m.group(1)] = int(m.group(2))
            continue
        m = _DECL_RE.match(line)
        if m:
            section, base, k, name, n = m.group(1, 2, 3, 4, 5)
            ncols = int(k) if k else 1
            dtype = np.dtype("<f4") if base == "float" else np.dtype("<i4")
            specs[int(n)] = (section, ncols, dtype, name)
    return counts, specs


def _coordinate_unit(lines: List[str]) -> str:
    """Best-effort read of the ``Coordinates "<unit>"`` field. Defaults to ``A``."""
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("Coordinates"):
            m = re.search(r'Coordinates\s+"([^"]+)"', stripped)
            if m:
                u = m.group(1).strip()
                if u in ("\xc5", "Å", "A", "angstrom", "Angstrom"):
                    return "A"
                if u in ("nm", "nanometer"):
                    return "nm"
                if u == "m":
                    return "m"
                if u in ("um", "μm", "\xb5m", "micron"):
                    return "um"
                return u
    return "A"


def _scale_to_A(coords: np.ndarray, unit: str) -> np.ndarray:
    """Convert ``coords`` to Ångströms if a non-Å unit is declared."""
    if unit == "A":
        return coords
    if unit == "nm":
        return coords * 10.0
    if unit == "um":
        return coords * 1.0e4
    if unit == "m":
        return coords * 1.0e10
    return coords  # unknown unit — leave as-is


# ---------------------------------------------------------------------------
# ASCII spatial-graph reader
# ---------------------------------------------------------------------------


def _read_ascii_sg(p: str) -> SpatialGraph:
    raw = open(p, "r", encoding=_ENC).read()
    try:
        header_end = raw.index(_DATA_MARKER_TXT)
    except ValueError as e:
        raise AmiraFormatError(
            f"{_osp.basename(p)}: '{_DATA_MARKER_TXT}' marker not found"
        ) from e

    header_lines = [ln for ln in raw[:header_end].splitlines() if ln.strip()]
    counts, specs = _parse_header_lines(header_lines)
    unit = _coordinate_unit(header_lines)

    # Body = everything after the marker line. Walk it once, slicing each
    # block according to the row count its spec implies; a stray "@N"
    # literal in the (already-past) Parameters block can't reach us.
    body_lines = raw[header_end:].splitlines()[1:]
    blocks: Dict[int, np.ndarray] = {}
    i = 0
    n_body = len(body_lines)
    while i < n_body:
        line = body_lines[i].strip()
        if not line:
            i += 1
            continue
        m = _AT_LINE_RE.match(line)
        if not m:
            i += 1
            continue
        n = int(m.group(1))
        spec = specs.get(n)
        if spec is None:
            i += 1
            while i < n_body and not _AT_LINE_RE.match(body_lines[i].strip()):
                i += 1
            continue
        section, ncols, dtype, _name = spec
        rows = counts.get(section, 0)
        # Parse ASCII floats into float64: the wire dtype (<f4) is only a hint
        # about the original binary precision, not the precision of the textual
        # decimal in the file. Storing into a float32 buffer truncates by ~7
        # decimal digits, which shows up against tardis_em's float64 reader as
        # ~1e-13 disagreement on small coordinates.
        parse_dtype = np.dtype(np.float64) if dtype.kind == "f" else np.dtype(np.int64)
        rows_buf = np.empty((rows, ncols), dtype=parse_dtype)
        r = 0
        i += 1
        while r < rows and i < n_body:
            tok_line = body_lines[i].strip()
            if not tok_line:
                i += 1
                continue
            if _AT_LINE_RE.match(tok_line):
                raise AmiraFormatError(
                    f"{_osp.basename(p)}: block @{n} ({section}) declared "
                    f"{rows} rows but found {r} before next @N"
                )
            parts = tok_line.split()
            if len(parts) < ncols:
                raise AmiraFormatError(
                    f"{_osp.basename(p)}: block @{n} row {r} has "
                    f"{len(parts)} columns; expected {ncols}"
                )
            try:
                rows_buf[r] = [
                    (float(parts[j]) if parse_dtype.kind == "f" else int(parts[j]))
                    for j in range(ncols)
                ]
            except ValueError as e:
                raise AmiraFormatError(
                    f"{_osp.basename(p)}: malformed value in block @{n} "
                    f"row {r}: {tok_line!r}"
                ) from e
            r += 1
            i += 1
        if r != rows:
            raise AmiraFormatError(
                f"{_osp.basename(p)}: block @{n} ({section}) declared "
                f"{rows} rows but only {r} parsed"
            )
        blocks[n] = rows_buf

    return _assemble_spatial_graph(p, counts, specs, blocks, unit)


# ---------------------------------------------------------------------------
# Binary spatial-graph reader
# ---------------------------------------------------------------------------


def _read_binary_sg(p: str) -> SpatialGraph:
    with open(p, "rb") as f:
        raw = f.read()
    pos = raw.find(_DATA_MARKER_BIN)
    if pos < 0:
        raise AmiraFormatError(
            f"{_osp.basename(p)}: binary '{_DATA_MARKER_TXT}' marker not found"
        )
    header_text = raw[:pos].decode(_ENC, errors="ignore")
    header_lines = [ln for ln in header_text.splitlines() if ln.strip()]
    counts, specs = _parse_header_lines(header_lines)
    unit = _coordinate_unit(header_lines)

    body = raw[pos + len(_DATA_MARKER_BIN):]
    blocks: Dict[int, np.ndarray] = {}
    p_idx = 0
    n_body = len(body)
    while p_idx < n_body:
        if body[p_idx:p_idx + 1] == b"\n":
            p_idx += 1
            continue
        if body[p_idx:p_idx + 1] != b"@":
            break
        nl = body.find(b"\n", p_idx)
        if nl < 0:
            break
        try:
            n = int(body[p_idx + 1:nl])
        except ValueError:
            break
        p_idx = nl + 1
        spec = specs.get(n)
        if spec is None:
            break
        section, ncols, dtype, _name = spec
        rows = counts.get(section, 0)
        nbytes = rows * ncols * dtype.itemsize
        if p_idx + nbytes > n_body:
            raise AmiraFormatError(
                f"{_osp.basename(p)}: block @{n} truncated "
                f"(expected {nbytes} bytes, body has {n_body - p_idx})"
            )
        arr = np.frombuffer(body[p_idx:p_idx + nbytes], dtype=dtype)
        blocks[n] = arr.reshape(rows, ncols)
        p_idx += nbytes

    return _assemble_spatial_graph(p, counts, specs, blocks, unit)


def _assemble_spatial_graph(
    p: str,
    counts: Dict[str, int],
    specs: Dict[int, Tuple[str, int, np.dtype, str]],
    blocks: Dict[int, np.ndarray],
    unit: str,
) -> SpatialGraph:
    """Common reducer: extract the four core fields and route everything else."""

    def _pick(name: str) -> Optional[np.ndarray]:
        for n, (_sec, _nc, _dt, nm) in specs.items():
            if nm == name and n in blocks:
                return blocks[n]
        return None

    vertices = _pick("VertexCoordinates")
    edge_conn = _pick("EdgeConnectivity")
    num_pts = _pick("NumEdgePoints")
    pt_coords = _pick("EdgePointCoordinates")
    missing = [
        nm
        for nm, arr in (
            ("VertexCoordinates", vertices),
            ("EdgeConnectivity", edge_conn),
            ("NumEdgePoints", num_pts),
            ("EdgePointCoordinates", pt_coords),
        )
        if arr is None
    ]
    if missing:
        raise AmiraFormatError(
            f"{_osp.basename(p)}: missing required field(s): {', '.join(missing)}"
        )

    if vertices.shape[0] != counts.get("VERTEX", vertices.shape[0]):
        raise AmiraFormatError(
            f"{_osp.basename(p)}: VERTEX count mismatch "
            f"({vertices.shape[0]} vs {counts.get('VERTEX')})"
        )
    if edge_conn.shape[0] != counts.get("EDGE", edge_conn.shape[0]):
        raise AmiraFormatError(
            f"{_osp.basename(p)}: EDGE count mismatch "
            f"({edge_conn.shape[0]} vs {counts.get('EDGE')})"
        )
    if pt_coords.shape[0] != counts.get("POINT", pt_coords.shape[0]):
        raise AmiraFormatError(
            f"{_osp.basename(p)}: POINT count mismatch "
            f"({pt_coords.shape[0]} vs {counts.get('POINT')})"
        )

    # Upcast to float64 regardless of source: binary blocks are float32 on the
    # wire, ASCII blocks are parsed as float64 directly. Coordinate arrays end
    # up at the same precision either way, which keeps round-trips bit-equal
    # against tardis_em's loader.
    vertices = _scale_to_A(vertices.astype(np.float64), unit)
    pt_coords = _scale_to_A(pt_coords.astype(np.float64), unit)

    vertex_int: Dict[str, np.ndarray] = {}
    vertex_float: Dict[str, np.ndarray] = {}
    edge_int: Dict[str, np.ndarray] = {}
    edge_float: Dict[str, np.ndarray] = {}
    point_int: Dict[str, np.ndarray] = {}
    point_float: Dict[str, np.ndarray] = {}
    consumed = {
        "VertexCoordinates",
        "EdgeConnectivity",
        "NumEdgePoints",
        "EdgePointCoordinates",
    }
    for n, (section, ncols, dtype, name) in specs.items():
        if name in consumed or n not in blocks:
            continue
        arr = blocks[n]
        if ncols == 1:
            arr = arr.ravel()
        # Promote floats to float64 for consistency across binary / ASCII
        # sources; ints stay int32 (their wire type) since precision isn't
        # at stake.
        if dtype.kind == "f":
            arr = arr.astype(np.float64)
        if section == "VERTEX":
            (vertex_int if dtype.kind == "i" else vertex_float)[name] = arr
        elif section == "EDGE":
            (edge_int if dtype.kind == "i" else edge_float)[name] = arr
        elif section == "POINT":
            (point_int if dtype.kind == "i" else point_float)[name] = arr

    return SpatialGraph(
        vertices=vertices,
        edge_connectivity=edge_conn.astype(np.int32),
        num_edge_points=num_pts.astype(np.int32).ravel(),
        point_coordinates=pt_coords,
        vertex_int_fields=vertex_int,
        vertex_float_fields=vertex_float,
        edge_int_fields=edge_int,
        edge_float_fields=edge_float,
        point_int_fields=point_int,
        point_float_fields=point_float,
        coordinate_unit="A",
    )


def read_spatial_graph(p: str) -> SpatialGraph:
    """Load an Amira spatial-graph ``.am`` file (ASCII or binary).

    All declared per-element fields are decoded into the typed dicts on the
    returned :class:`SpatialGraph`. Coordinates are normalised to Ångströms.
    """
    head = _peek_header(p)
    if "AmiraMesh BINARY-LITTLE-ENDIAN" in head:
        return _read_binary_sg(p)
    if ("AmiraMesh 3D ASCII" in head) or ("# ASCII Spatial Graph" in head):
        return _read_ascii_sg(p)
    raise AmiraFormatError(
        f"{_osp.basename(p)}: not a recognised AmiraMesh spatial-graph "
        f"header (first bytes: {head[:64]!r})"
    )


def read_segmented_points(p: str) -> np.ndarray:
    """Convenience: return ``(N, 4)`` ``[seg_id, x, y, z]`` in Å.

    Mirrors the contract of the historical
    ``ImportDataFromAmira.get_segmented_points()`` when called without a
    paired image volume — i.e. no transformation offset is subtracted and
    no pixel-size scaling is applied.
    """
    return read_spatial_graph(p).segmented_points()


# ---------------------------------------------------------------------------
# Volume (image lattice) reader
# ---------------------------------------------------------------------------


_LATTICE_DEF_RE = re.compile(r"^\s*define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
_BBOX_RE = re.compile(
    r"^\s*BoundingBox\s+"
    r"(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+"
    r"(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+"
    r"(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s*,?\s*$"
)


def _parse_volume_header(text: str, p: str):
    lines = text.splitlines()
    nx = ny = nz = None
    bbox = None
    coordinate_unit = "A"
    dtype: Optional[np.dtype] = None
    is_ascii = "AmiraMesh 3D ASCII" in text
    for line in lines:
        m = _LATTICE_DEF_RE.match(line)
        if m:
            nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))
            continue
        m = _BBOX_RE.match(line)
        if m:
            bbox = [float(g) for g in m.groups()]
            continue
        stripped = line.lstrip()
        if stripped.startswith("Coordinates"):
            mm = re.search(r'Coordinates\s+"([^"]+)"', stripped)
            if mm:
                u = mm.group(1).strip()
                if u in ("\xc5", "Å", "A", "angstrom", "Angstrom"):
                    coordinate_unit = "A"
                elif u == "nm":
                    coordinate_unit = "nm"
                elif u == "m":
                    coordinate_unit = "m"
                elif u in ("um", "μm", "\xb5m", "micron"):
                    coordinate_unit = "um"
                else:
                    coordinate_unit = u
        if "Lattice { byte Data }" in line:
            dtype = np.dtype(np.uint8)
        elif "Lattice { sbyte Data }" in line:
            dtype = np.dtype(np.int8)
        elif "Lattice { float Data }" in line:
            dtype = np.dtype("<f4")

    if nx is None or ny is None or nz is None:
        raise AmiraFormatError(
            f"{_osp.basename(p)}: 'define Lattice nx ny nz' not found"
        )
    if dtype is None:
        raise AmiraFormatError(
            f"{_osp.basename(p)}: unsupported or missing Lattice data type "
            f"(expected byte, sbyte or float)"
        )
    return nx, ny, nz, bbox, dtype, coordinate_unit, is_ascii


def _pixel_size_A(
    nx: int, bbox: Optional[List[float]], unit: str
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return ``(pixel_size_A, physical_size, transformation)``.

    Pixel size is the X-extent of the BoundingBox divided by ``nx - 1``,
    optionally scaled into Å. Mirrors the historical derivation.
    """
    if bbox is None or len(bbox) != 6:
        return 1.0, np.zeros(3), np.zeros(3)
    tx, x_hi, ty, y_hi, tz, z_hi = bbox
    physical_size = np.array([x_hi, y_hi, z_hi], dtype=float)
    transformation = np.array([tx, ty, tz], dtype=float)
    px = (x_hi - tx) / max(nx - 1, 1)
    if unit == "nm":
        px *= 10.0
    elif unit == "um":
        px *= 1.0e4
    elif unit == "m":
        px *= 1.0e10
    return round(float(px), 3), physical_size, transformation


def read_amira_volume(
    p: str,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Load an AmiraMesh image lattice from ``p``.

    Supports ASCII and binary ``byte`` / ``sbyte`` / ``float`` lattices.
    Returns ``(image, pixel_size_A, physical_size, transformation)``.
    """
    head = _peek_header(p, n=32768)
    nx, ny, nz, bbox, dtype, unit, is_ascii = _parse_volume_header(head, p)
    px, physical_size, transformation = _pixel_size_A(nx, bbox, unit)
    voxels = int(nx) * int(ny) * int(nz)

    if is_ascii:
        data_offset = _find_ascii_lattice_offset(p)
        if data_offset < 0:
            raise AmiraFormatError(
                f"{_osp.basename(p)}: ASCII data section (@1) not found"
            )
        with open(p, "rb") as f:
            f.seek(data_offset)
            img = np.fromfile(f, dtype=dtype, sep=" ", count=voxels)
    else:
        if dtype == np.uint8 or dtype == np.int8:
            img = np.fromfile(p, dtype=dtype)
        else:
            offset = head.find("\n@1\n")
            if offset < 0:
                raise AmiraFormatError(
                    f"{_osp.basename(p)}: binary '@1' data marker not found"
                )
            offset += 4
            img = np.fromfile(p, dtype=dtype, offset=offset)

    if img.size < voxels:
        raise AmiraFormatError(
            f"{_osp.basename(p)}: lattice has {img.size} voxels, expected {voxels}"
        )
    img = img[-voxels:]
    if dtype == np.int8:
        img = (img.astype(np.uint8) + 128).astype(np.uint8)
    img = img.reshape((nz, ny, nx))
    return img, px, physical_size, transformation


def _find_ascii_lattice_offset(p: str) -> int:
    """Byte offset to the start of an ASCII Lattice ``@1`` data section.

    Streams the file in 1 MiB chunks looking for ``\\n@1\\n`` so a multi-GB
    body is never loaded into RAM more than once.
    """
    marker = b"\n@1\n"
    keep = len(marker) - 1
    pos = 0
    buf = b""
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                return -1
            buf += chunk
            idx = buf.find(marker)
            if idx >= 0:
                return pos + idx + len(marker)
            if len(buf) > keep:
                pos += len(buf) - keep
                buf = buf[-keep:]


# ---------------------------------------------------------------------------
# Small array helpers (used by the writer's V2 synthesis path)
# ---------------------------------------------------------------------------


def _reorder_segments_id(coord: np.ndarray) -> np.ndarray:
    """Rewrite the segment-ID column to a dense ``0..N-1`` range, preserving order."""
    if coord.shape[0] == 0:
        return coord
    _, inv = np.unique(coord[:, 0], return_inverse=True)
    out = coord.copy()
    out[:, 0] = inv.astype(out.dtype)
    return out


def _drop_single_point_filaments(coord: np.ndarray) -> np.ndarray:
    """Remove segments with only one polyline point (not a valid Amira edge)."""
    if coord.shape[0] == 0:
        return coord
    uniq, counts = np.unique(coord[:, 0], return_counts=True)
    return coord[np.isin(coord[:, 0], uniq[counts > 1])]


# ---------------------------------------------------------------------------
# Spatial-graph writer
# ---------------------------------------------------------------------------


def _try_pandorica_version() -> str:
    try:
        from pandorica._version import version as v  # type: ignore

        return str(v)
    except Exception:  # noqa: BLE001
        return "?"


def _default_sg_banner(binary: bool = False) -> List[str]:
    year = datetime.now().year
    magic = (
        "# AmiraMesh BINARY-LITTLE-ENDIAN 3.0"
        if binary
        else "# ASCII Spatial Graph"
    )
    return [
        magic,
        "# PANDORICA - Serial Stitcher",
        f"# pandorica v{_try_pandorica_version()}",
        f"# PolyForm Noncommercial 1.0.0 * 2026-{year} * Robert Kiewisz",
    ]


def _merge_extra_fields(
    sg_dict: Dict[str, np.ndarray],
    extra: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Right-biased merge: ``extra`` overrides ``sg_dict``."""
    out = dict(sg_dict)
    if extra:
        for k, v in extra.items():
            out[k] = np.asarray(v)
    return out


def _validate_field_lengths(
    sg: SpatialGraph,
    vertex_int: Dict[str, np.ndarray],
    vertex_float: Dict[str, np.ndarray],
    edge_int: Dict[str, np.ndarray],
    edge_float: Dict[str, np.ndarray],
    point_int: Dict[str, np.ndarray],
    point_float: Dict[str, np.ndarray],
) -> None:
    def _check(scope: str, expected: int, fields: Dict[str, np.ndarray]) -> None:
        for name, arr in fields.items():
            if arr.shape[0] != expected:
                raise ValueError(
                    f"{scope} field {name!r}: length {arr.shape[0]} "
                    f"does not match {scope} count {expected}"
                )

    _check("VERTEX", sg.n_vertices, vertex_int)
    _check("VERTEX", sg.n_vertices, vertex_float)
    _check("EDGE", sg.n_edges, edge_int)
    _check("EDGE", sg.n_edges, edge_float)
    _check("POINT", sg.n_points, point_int)
    _check("POINT", sg.n_points, point_float)


def write_spatial_graph(
    p: str,
    data: Union[np.ndarray, SpatialGraph],
    *,
    binary: bool = False,
    history: Optional[List[str]] = None,
    vertex_int_fields: Optional[Dict[str, np.ndarray]] = None,
    vertex_float_fields: Optional[Dict[str, np.ndarray]] = None,
    edge_int_fields: Optional[Dict[str, np.ndarray]] = None,
    edge_float_fields: Optional[Dict[str, np.ndarray]] = None,
    point_int_fields: Optional[Dict[str, np.ndarray]] = None,
    point_float_fields: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """Write a spatial-graph ``.am`` file at ``p``.

    ---------------------------------------------------------------------
    HOW TO FORMAT YOUR INPUT
    ---------------------------------------------------------------------

    **1. Pick a shape for** ``data``::

        # Option A — segmented polyline points, one row per sample:
        #   (N, 4) float array, columns = [segment_id, x, y, z]
        #   Coordinates in Ångströms. Rows of the same segment_id must
        #   be contiguous and ordered along the spline. The writer
        #   synthesises chain endpoints (two vertices per segment).
        #
        #   Use this when you have a "merged points" array from the
        #   stitcher or a custom pipeline.
        coords = np.array([
            [0, x0, y0, z0],   # segment 0, sample 0
            [0, x1, y1, z1],   # segment 0, sample 1
            [1, x0, y0, z0],   # segment 1, sample 0
            ...
        ])
        write_spatial_graph("out.am", coords)

        # Option B — full SpatialGraph (round-trips read_spatial_graph
        # losslessly, preserves arbitrary VERTEX/EDGE topology):
        sg = read_spatial_graph("in.am")
        write_spatial_graph("out.am", sg)

    **2. Add labels / scores via the** ``*_fields`` **kwargs.**

    Every kwarg is a ``{name: 1-D array}`` dict. The array length must
    match the count of its scope (VERTEX / EDGE / POINT). Names become
    Amira field names verbatim — keep them ``[A-Za-z_][A-Za-z0-9_]*``.

    +----------------------+----------+----------+--------------------------+
    | kwarg                | one per  | dtype    | typical use              |
    +======================+==========+==========+==========================+
    | vertex_int_fields    | VERTEX   | int      | endpoint kind / category |
    | vertex_float_fields  | VERTEX   | float    | per-end scalar           |
    | edge_int_fields      | EDGE     | int      | per-spline class flag    |
    | edge_float_fields    | EDGE     | float    | tortuosity, length, etc. |
    | point_int_fields     | POINT    | int      | boundary / region flag   |
    | point_float_fields   | POINT    | float    | thickness, quality, etc. |
    +----------------------+----------+----------+--------------------------+

    In chain-only graphs (the V2 schema produced from an ``(N, 4)``
    array) every VERTEX is degree-1, laid out as
    ``[start_e0, end_e0, start_e1, end_e1, …]``. Index even VERTEX rows
    for "start of spline" labels, odd rows for "end of spline".

    **3. Worked example — labels at all three levels.**

    ::

        from pandorica.io.amira import (
            SpatialGraph, write_spatial_graph, read_spatial_graph,
        )

        coords = ...                              # (N, 4) [seg, x, y, z]
        sg = SpatialGraph.from_segmented_points(coords)

        write_spatial_graph(
            "labelled.am", sg,
            edge_int_fields   = {"IsGood":    np.array([1, 0, 1, ...])},
            edge_float_fields = {"Tortuosity": np.array([1.05, 1.22, ...])},
            point_int_fields  = {"BoundaryFlag": np.array([1, 0, 0, ...])},
            point_float_fields= {"thickness":  np.array([2.5, 2.6, ...])},
            vertex_int_fields = {"EndpointKind": np.tile([1, 2], sg.n_edges)},
            history           = ["pandorica labelled run on 2026-05-30"],
        )

    **4. Field merging.**

    When ``data`` is a SpatialGraph that already carries fields (e.g.
    from a fresh read), kwargs **override** matching names and **add**
    new ones. Pass an empty dict to keep existing values untouched.

    **5. ASCII vs binary.**

    Default is ASCII (``# ASCII Spatial Graph`` magic, ``.17e`` float
    text — round-trip-safe for float64). Set ``binary=True`` for
    ``AmiraMesh BINARY-LITTLE-ENDIAN`` output: float fields downcast to
    ``float32`` and ints to ``int32`` on the wire, matching Amira's
    binary convention. Binary → binary round-trip is bit-equal.
    ASCII → binary loses precision below float32 (~7 digits) by design.

    **6. Pitfalls.**

    - Path must end in ``.am``.
    - ``(N, 4)`` rows must be grouped by ``segment_id`` (contiguous
      blocks). The writer drops segments with only one point and
      densifies IDs to ``0..n_segments-1`` automatically.
    - Field array dtype is enforced by which kwarg you use, not by the
      array's own dtype. A float array passed to ``edge_int_fields`` is
      coerced to ``int32`` (loss is silent — pick the right kwarg).
    - History lines are emitted verbatim as ``# <line>`` comments at
      the top of the header; keep them ASCII-safe.

    :param p: output ``.am`` path (must end in ``.am``).
    :param data: ``(N, 4)`` ``[seg_id, x, y, z]`` array, or a
        :class:`SpatialGraph`.
    :param binary: emit ``AmiraMesh BINARY-LITTLE-ENDIAN`` if ``True``,
        ASCII otherwise (default).
    :param history: optional ``# <line>`` comments stamped into the
        header (e.g. provenance / processing notes).
    :param vertex_int_fields: ``{name: (V,) int}`` per-vertex labels.
    :param vertex_float_fields: ``{name: (V,) float}`` per-vertex scalars.
    :param edge_int_fields: ``{name: (E,) int}`` per-spline labels.
    :param edge_float_fields: ``{name: (E,) float}`` per-spline scalars.
    :param point_int_fields: ``{name: (P,) int}`` per-point labels.
    :param point_float_fields: ``{name: (P,) float}`` per-point scalars.
    """
    if not p.endswith(".am"):
        raise ValueError(f"spatial graph path must end in .am: {p!r}")

    sg = (
        data
        if isinstance(data, SpatialGraph)
        else SpatialGraph.from_segmented_points(data)
    )

    v_int = _merge_extra_fields(sg.vertex_int_fields, vertex_int_fields)
    v_flt = _merge_extra_fields(sg.vertex_float_fields, vertex_float_fields)
    e_int = _merge_extra_fields(sg.edge_int_fields, edge_int_fields)
    e_flt = _merge_extra_fields(sg.edge_float_fields, edge_float_fields)
    p_int = _merge_extra_fields(sg.point_int_fields, point_int_fields)
    p_flt = _merge_extra_fields(sg.point_float_fields, point_float_fields)
    _validate_field_lengths(sg, v_int, v_flt, e_int, e_flt, p_int, p_flt)

    # Decl list: (block_id, section, base, name, source_array). Same shape
    # is consumed by both writers below.
    decl_blocks: List[Tuple[int, str, str, str, np.ndarray]] = []
    next_id = 5
    for nm, arr in v_int.items():
        decl_blocks.append((next_id, "VERTEX", "int", nm, arr))
        next_id += 1
    for nm, arr in v_flt.items():
        decl_blocks.append((next_id, "VERTEX", "float", nm, arr))
        next_id += 1
    for nm, arr in e_int.items():
        decl_blocks.append((next_id, "EDGE", "int", nm, arr))
        next_id += 1
    for nm, arr in e_flt.items():
        decl_blocks.append((next_id, "EDGE", "float", nm, arr))
        next_id += 1
    for nm, arr in p_int.items():
        decl_blocks.append((next_id, "POINT", "int", nm, arr))
        next_id += 1
    for nm, arr in p_flt.items():
        decl_blocks.append((next_id, "POINT", "float", nm, arr))
        next_id += 1

    header_text = _build_sg_header_text(
        sg, v_int, v_flt, e_int, e_flt, p_int, p_flt, decl_blocks, binary, history
    )

    # The text header is iso-8859-1 (because of the "Å" byte). Open the file
    # binary and emit either ASCII data lines (iso-8859-1) or raw little-
    # endian bytes after the data marker; both paths use the same encoding
    # for the header.
    with open(p, "wb") as f:
        f.write(header_text.encode(_ENC))
        if binary:
            _emit_binary_data_section(f, sg, decl_blocks)
        else:
            _emit_ascii_data_section(f, sg, decl_blocks)


def _build_sg_header_text(
    sg: SpatialGraph,
    v_int: Dict[str, np.ndarray],
    v_flt: Dict[str, np.ndarray],
    e_int: Dict[str, np.ndarray],
    e_flt: Dict[str, np.ndarray],
    p_int: Dict[str, np.ndarray],
    p_flt: Dict[str, np.ndarray],
    decl_blocks: List[Tuple[int, str, str, str, np.ndarray]],
    binary: bool,
    history: Optional[List[str]],
) -> str:
    """Return the full text header, ending with ``"# Data section follows\\n"``."""
    out: List[str] = []
    for line in _default_sg_banner(binary=binary):
        out.append(line)
    if history:
        out.append("")
        out.append("# PANDORICA history:")
        for h in history:
            out.append("# " + h if not h.startswith("#") else h)
    out.append("")
    out.append(f"define VERTEX {sg.n_vertices}")
    out.append(f"define EDGE {sg.n_edges}")
    out.append(f"define POINT {sg.n_points}")
    out.append("")
    out.append("Parameters {")
    out.append("    Units {")
    out.append('        Coordinates "\xc5"')
    out.append("    }")
    _append_units_block(out, "Vertex", list(v_int.keys()) + list(v_flt.keys()))
    _append_units_block(out, "Edge", list(e_int.keys()) + list(e_flt.keys()))
    out.append("    SpatialGraphUnitsPoint {")
    for nm in {"thickness", *p_int.keys(), *p_flt.keys()}:
        out.append(f"        {nm} {{")
        out.append("            Unit -1,")
        out.append("            Dimension -1")
        out.append("        }")
    out.append("    }")
    # Label-group blocks (one per int-typed VERTEX field — Amira's color group).
    for idx, nm in enumerate(v_int.keys()):
        out.append(f"    {nm} {{")
        out.append("        Label0 {")
        out.append("            Color 1 0.5 0.5,")
        out.append(f"            Id {idx + 1}")
        out.append("        }")
        out.append("        Id 0,")
        out.append("        Color 1 0 0")
        out.append("    }")
    out.append('    ContentType "HxSpatialGraph"')
    out.append("}")
    out.append("")
    out.append("VERTEX { float[3] VertexCoordinates } @1")
    out.append("EDGE { int[2] EdgeConnectivity } @2")
    out.append("EDGE { int NumEdgePoints } @3")
    out.append("POINT { float[3] EdgePointCoordinates } @4")
    for n, section, base, nm, _arr in decl_blocks:
        out.append(f"{section} {{ {base} {nm} }} @{n}")
    out.append("")
    out.append("# Data section follows")
    return "\n".join(out) + "\n"


def _append_units_block(out: List[str], kind: str, names: List[str]) -> None:
    out.append(f"    SpatialGraphUnits{kind} {{")
    for nm in names:
        out.append(f"        {nm} {{")
        out.append("            Unit -1,")
        out.append("            Dimension -1")
        out.append("        }")
    out.append("    }")


def _emit_ascii_data_section(
    f,
    sg: SpatialGraph,
    decl_blocks: List[Tuple[int, str, str, str, np.ndarray]],
) -> None:
    """Write the ASCII data blocks (text lines, iso-8859-1 bytes)."""

    def w(line: str) -> None:
        f.write((line + "\n").encode(_ENC))

    w("@1")
    for row in sg.vertices:
        w(f"{float(row[0]):.17e} {float(row[1]):.17e} {float(row[2]):.17e}")
    w("")
    w("@2")
    for row in sg.edge_connectivity:
        w(f"{int(row[0])} {int(row[1])}")
    w("")
    w("@3")
    for v in sg.num_edge_points:
        w(f"{int(v)}")
    w("")
    w("@4")
    for row in sg.point_coordinates:
        w(f"{float(row[0]):.17e} {float(row[1]):.17e} {float(row[2]):.17e}")
    for n, _section, base, _nm, arr in decl_blocks:
        w("")
        w(f"@{n}")
        flat = np.asarray(arr).ravel()
        if base == "int":
            for v in flat:
                w(f"{int(v)}")
        else:
            for v in flat:
                w(f"{float(v):.17e}")


def _emit_binary_data_section(
    f,
    sg: SpatialGraph,
    decl_blocks: List[Tuple[int, str, str, str, np.ndarray]],
) -> None:
    """Write little-endian binary blocks; each ``@N`` followed by raw bytes.

    Float fields are downcast to ``<f4`` and int fields to ``<i4`` so the
    output is round-trip-compatible with :func:`read_spatial_graph` (which
    reads each block at the on-wire dtype declared in the header — and the
    declared types are always ``float[K]`` / ``int[K]``).
    """

    def block(n: int, arr: np.ndarray, kind: str, ncols: int) -> None:
        f.write(f"\n@{n}\n".encode(_ENC))
        if kind == "float":
            buf = np.ascontiguousarray(arr.reshape(-1, ncols)).astype("<f4")
        else:
            buf = np.ascontiguousarray(arr.reshape(-1, ncols)).astype("<i4")
        f.write(buf.tobytes())

    # The four core blocks first. The leading "\n" before "@1" in block()
    # mirrors the separator the parser tolerates between blocks; the bytes
    # immediately after "# Data section follows\n" are "\n@1\n..." which
    # matches what `_read_binary_sg` walks.
    block(1, sg.vertices, "float", 3)
    block(2, sg.edge_connectivity, "int", 2)
    block(3, sg.num_edge_points, "int", 1)
    block(4, sg.point_coordinates, "float", 3)
    for n, _section, base, _nm, arr in decl_blocks:
        flat = np.asarray(arr)
        ncols = flat.shape[1] if flat.ndim > 1 else 1
        block(n, flat, base, ncols)


# ---------------------------------------------------------------------------
# Volume writer (streamed byte lattice)
# ---------------------------------------------------------------------------


def _default_lattice_banner() -> List[str]:
    year = datetime.now().year
    return [
        "# AmiraMesh BINARY-LITTLE-ENDIAN 3.0",
        "# PANDORICA - Serial Stitcher",
        f"# pandorica v{_try_pandorica_version()}",
        f"# PolyForm Noncommercial 1.0.0 * 2026-{year} * Robert Kiewisz",
    ]


def _lattice_header_lines(
    nz: int,
    ny: int,
    nx: int,
    pixel_size_A: float,
    header: Optional[List[str]] = None,
) -> List[str]:
    xLen = (nx - 1) * pixel_size_A
    yLen = (ny - 1) * pixel_size_A
    zLen = (nz - 1) * pixel_size_A
    out = list(_default_lattice_banner())
    if header:
        out += ["# " + h if not h.startswith("#") else h for h in header]
    out += [
        "",
        "",
        f"define Lattice {nx} {ny} {nz}",
        "",
        "Parameters {",
        "    Units {",
        '        Coordinates "\xc5"',
        "    }",
        '    DataWindow "0.000000 255.000000",',
        f'    Content "{nx}x{ny}x{nz} byte, uniform coordinates",',
        f"    BoundingBox 0 {xLen} 0 {yLen} 0 {zLen},",
        '    CoordType "uniform"',
        "}",
        "",
        "Lattice { byte Data } @1",
        "",
        "# Data section follows",
        "@1",
    ]
    return out


def write_amira_volume_streamed(
    p: str,
    slabs: Iterable[str],
    shape: Tuple[int, int, int],
    pixel_size_A: float,
    *,
    header: Optional[List[str]] = None,
) -> None:
    """Write a binary AmiraMesh byte lattice by concatenating raw ``uint8`` slabs.

    Each path in ``slabs`` is a raw C-order ``uint8`` dump of a contiguous
    block of Z-slices; concatenated in order they form the ``(nz, ny, nx)``
    lattice. Peak memory is a small constant — no array is materialised in
    RAM.
    """
    import shutil

    if not p.endswith(".am"):
        raise ValueError(f"volume path must end in .am: {p!r}")
    nz, ny, nx = shape
    head_lines = _lattice_header_lines(nz, ny, nx, pixel_size_A, header)
    with open(p, "w", encoding="utf-8") as f:
        for line in head_lines:
            f.write(line + "\n")
    with open(p, "ab") as out:
        for sp in slabs:
            with open(sp, "rb") as src:
                shutil.copyfileobj(src, out, length=1 << 20)
        out.write(b"\n")


# ---------------------------------------------------------------------------
# Folder discovery — pair Amira images with their spatial graphs
# ---------------------------------------------------------------------------


def sort_tomogram_files(path: str) -> Tuple[List[str], List[Optional[str]]]:
    """Discover and pair image + spatial-graph files in a folder.

    Returns ``(image_paths, coord_paths)`` lists where ``coord_paths[i]`` is
    the spatial graph paired with ``image_paths[i]`` (or ``None`` if no
    matching graph was found). Pairing is by filename-stem containment: a
    graph is paired with the image whose stem appears in the graph's stem.

    ``.am`` files are classified by **header content**, not by the format
    string (which varies — ``AmiraMesh`` / ``Avizo``, BINARY / ASCII, 2.x /
    3.x). A file that defines a ``Lattice`` is treated as an image volume;
    anything that defines ``VERTEX`` / ``EDGE`` / ``HxSpatialGraph`` is
    treated as a microtubule graph. Non-Amira images (``.tif`` / ``.mrc`` /
    ``.rec``) and coords (``.csv``) are accepted by extension.

    If the folder has only graphs (no image volumes), the image list comes
    back empty and the caller can fall back to a graph-only mode.
    """
    image_exts = {".tif", ".tiff", ".mrc", ".rec", ".am"}
    coord_exts = {".csv", ".am"}

    images: List[str] = []
    coordinates: List[str] = []

    for fname in listdir(path):
        fpath = join(path, fname)
        if not isfile(fpath):
            continue

        ext = splitext(fname)[1].lower()

        # Normal image types (non-.am)
        if ext in image_exts and ext != ".am":
            images.append(fpath)
            continue

        # Normal coordinate types (non-.am)
        if ext in coord_exts and ext != ".am":
            coordinates.append(fpath)
            continue

        # Handle .am files (Amira/Avizo) — classify by header CONTENT.
        if ext == ".am":
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(8192).lower()
            except OSError:
                continue  # skip unreadable files

            is_graph = (
                "hxspatialgraph" in head
                or "define vertex" in head
                or "define edge" in head
            )
            if "define lattice" in head and not is_graph:
                images.append(fpath)
            else:
                coordinates.append(fpath)

    # Graph-only folder: hand the graphs back so the caller can build
    # volume-free sections.
    if not images:
        return [], sorted(coordinates)

    img_path_list: List[str] = []
    coord_path_list: List[Optional[str]] = []
    for img in sorted(images):
        img_path_list.append(img)

        stem = splitext(split(img)[-1])[0].lower()
        matches = [p for p in coordinates if stem in splitext(split(p)[-1])[0].lower()]

        coord_path_list.append(matches[0] if matches else None)

    return img_path_list, coord_path_list


__all__ = [
    "AmiraFormatError",
    "SpatialGraph",
    "read_spatial_graph",
    "read_segmented_points",
    "read_amira_volume",
    "write_spatial_graph",
    "write_amira_volume_streamed",
    "sort_tomogram_files",
]
