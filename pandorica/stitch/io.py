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
Dataset discovery + lazy loading for the stitch-validator plugin.

A *dataset* is a folder of serial sections, each with an Amira image ``.am`` and a
matching ``*_spatialGraph.am`` (microtubules), discovered/paired by the package's
own :func:`sort_tomogram_files`. Spatial graphs are small and loaded eagerly;
volumes are multi-GB and loaded **only on demand** (and may be downsampled for
display), so opening a dataset never pulls a gigabyte into RAM.
"""

import re
from dataclasses import dataclass, field
from os import listdir
from os.path import basename, commonprefix, isdir, isfile, join, split, splitext
from typing import List, Optional, Tuple

import numpy as np

# Shown when a folder has files but none can be paired into stitchable sections.
# Kept short (≤ a few lines) so it renders cleanly inside the TARDIS logo box.
_EXPECTED_LAYOUT = (
    "Expected per section — rename so the image and its graph pair:\n"
    "  <name>_sec01.am               image  (defines 'Lattice')\n"
    "  <name>_sec01_spatialGraph.am  graph  (defines 'VERTEX'/'EDGE')\n"
    "The graph name must contain the image name; add a 'secNN' index."
)


def _raise_no_sections(folder: str, load_errors: Optional[List[str]] = None) -> None:
    """Raise a friendly, actionable error explaining why no sections were found."""
    if not isdir(folder):
        raise FileNotFoundError(f"Dataset folder does not exist: {folder!r}")

    all_files = [f for f in listdir(folder) if isfile(join(folder, f))]
    am_files = [f for f in all_files if f.lower().endswith(".am")]

    lines = ["No stitchable sections found in this folder:", f"  {folder}", ""]
    if not all_files:
        lines.append("The folder is empty.")
    elif not am_files:
        lines.append(f"Found {len(all_files)} file(s) but no Amira '.am' files.")
    else:
        lines.append(f"Found {len(am_files)} '.am' file(s), but none define an image")
        lines.append("'Lattice', so nothing could be paired into sections.")
        if load_errors:
            lines.append("(Some '.am' files also failed to parse as graphs.)")
    lines.append("")
    lines.append(_EXPECTED_LAYOUT)
    raise FileNotFoundError("\n".join(lines))


def sort_tomogram_files(path):
    image_exts = {".tif", ".tiff", ".mrc", ".rec", ".am"}
    coord_exts = {".csv", ".am"}

    images = []
    coordinates = []

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

        # Handle .am files (Amira/Avizo) → classify by header CONTENT, not by the
        # exact format string. Both image volumes and microtubule spatial graphs
        # are "AmiraMesh"/"Avizo" files and the format line varies
        # ("AmiraMesh" vs "Avizo", BINARY vs ASCII, 2.x vs 3.x), so sniff what the
        # file *defines*: a uniform ``Lattice`` (image volume) vs
        # ``VERTEX``/``EDGE``/``HxSpatialGraph`` (the MT graph).
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

    # Graph-only folder (no detectable image volumes): still return the graphs so
    # the caller can build volume-free sections.
    if not images:
        return [], sorted(coordinates)

    # Pair each image with the coord file whose name contains the image stem.
    img_path_list, coord_path_list = [], []
    for img in sorted(images):
        img_path_list.append(img)

        img = splitext(split(img)[-1])[0].lower()

        matches = [p for p in coordinates if img in splitext(split(p)[-1])[0].lower()]

        if len(matches) == 0:
            coord_path_list.append(None)
        else:
            coord_path_list.append(matches[0])

    return img_path_list, coord_path_list


def _stem(path: Optional[str], index: int) -> str:
    """Filename stem without the ``_spatialGraph`` suffix (``sectionNN`` if no path)."""
    if path is None:
        return f"section{index:02d}"
    stem = splitext(basename(path))[0]
    if stem.endswith("_spatialGraph"):
        stem = stem[: -len("_spatialGraph")]
    return stem


def _derive_names(stems: List[str]) -> List[str]:
    """
    Compact, **unique** per-section labels from their filename stems.

    Prefers a ``secNN`` token when every section has one (e.g. ``sec09``); otherwise
    strips the longest common prefix + suffix so only the distinguishing part
    remains (e.g. ``…_metaphase02_4`` → ``4``). Falls back to index-suffixing if
    anything still collides — so interface labels (and ``coarse_gt.json`` keys) are
    never ambiguous.
    """
    secs = [re.search(r"sec\d+", s, re.IGNORECASE) for s in stems]
    if stems and all(secs):
        names = [m.group() for m in secs]
    elif len(stems) > 1:
        pre = commonprefix(stems)
        suf = commonprefix([s[::-1] for s in stems])[::-1]
        names = []
        for s in stems:
            end = len(s) - len(suf)
            core = s[len(pre) : end] if len(pre) < end else s
            names.append(core.strip("._- ") or s)
    else:
        names = list(stems)

    seen: dict = {}
    out: List[str] = []
    for i, n in enumerate(names):
        n = n or f"section{i:02d}"
        if n in seen:
            n = f"{n}_{i}"
        seen[n] = True
        out.append(n)
    return out


@dataclass
class Section:
    """One serial section: its graph (eager) and volume (lazy)."""

    name: str
    index: int
    image_path: Optional[str]
    coord_path: Optional[str]
    coords: np.ndarray  # [N, 4] [id, x, y, z] in physical units (Å), or empty
    pixel_size: float = 1.0  # reliable only after the volume header is read
    _volume: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def n_mts(self) -> int:
        return int(self.coords[:, 0].max()) + 1 if len(self.coords) else 0

    def has_volume(self) -> bool:
        return self.image_path is not None

    def load_volume(self, downscale: int = 1, force: bool = False) -> np.ndarray:
        """
        Load (and cache) this section's ``[Z, Y, X]`` volume, optionally downsampled.

        ``downscale`` decimates all three axes by that integer step (cheap, lossy)
        — use it to preview/export a multi-GB stack quickly. The true pixel size is
        read from the image header here and written back to :attr:`pixel_size`
        (scaled by ``downscale``).
        """
        if self.image_path is None:
            raise ValueError(f"section {self.name!r} has no image volume")
        if self._volume is not None and not force:
            return self._volume

        from tardis_em.utils.load_data import ImportDataFromAmira

        am = ImportDataFromAmira(
            src_am=self.coord_path or self.image_path, src_img=self.image_path
        )
        vol, px = am.get_image()
        vol = np.asarray(vol)
        if downscale > 1:
            vol = vol[::downscale, ::downscale, ::downscale]
        if px:
            self.pixel_size = float(px) * downscale
        self._volume = vol
        return vol

    def drop_volume(self) -> None:
        self._volume = None


@dataclass
class Dataset:
    """An ordered stack of :class:`Section`."""

    folder: str
    sections: List[Section]

    def __len__(self) -> int:
        return len(self.sections)

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.sections]

    def coords_list(self) -> List[np.ndarray]:
        return [s.coords for s in self.sections]

    def interface_label(self, k: int) -> str:
        return f"{self.sections[k].name}->{self.sections[k + 1].name}"


def load_dataset(folder: str) -> Dataset:
    """
    Discover and load a serial-section dataset folder.

    Graphs are read eagerly (``ImportDataFromAmira.get_segmented_points``);
    volumes are left unloaded. Sections without a graph get an empty ``[0, 4]``
    coords array so they still appear in the stack (volume-only stitching is still
    possible, but the MT pipelines need graphs).

    :param folder: directory containing ``.am`` images and ``*_spatialGraph.am``.
    :return: a :class:`Dataset` in stack order.
    """
    from tardis_em.utils.load_data import ImportDataFromAmira

    image_paths, coord_paths = sort_tomogram_files(folder)
    if not image_paths and not coord_paths:
        _raise_no_sections(folder)

    # sort_tomogram_files pairs by image; if there are graphs but no images
    # (graph-only folder) fall back to pairing on the graph list.
    pairs: List[Tuple[Optional[str], Optional[str]]]
    if image_paths:
        pairs = list(zip(image_paths, coord_paths))
    else:
        pairs = [(None, c) for c in sorted(coord_paths)]

    names = _derive_names(
        [_stem(coord or img, i) for i, (img, coord) in enumerate(pairs)]
    )
    sections: List[Section] = []
    load_errors: List[str] = []
    for i, (img, coord) in enumerate(pairs):
        coords = np.empty((0, 4))
        if coord is not None and coord.endswith(".am"):
            try:
                c = ImportDataFromAmira(src_am=coord).get_segmented_points()
            except Exception as e:  # noqa: BLE001 — malformed / misclassified .am
                load_errors.append(f"{basename(coord)}: {e}")
                c = None
            if c is not None and len(c):
                coords = np.asarray(c, dtype=float)
        sections.append(
            Section(
                name=names[i], index=i, image_path=img, coord_path=coord, coords=coords
            )
        )

    # Files were found but none yielded usable content (no image volume and no
    # parseable microtubule graph) — explain and suggest the expected layout.
    if not any(s.image_path is not None or len(s.coords) for s in sections):
        _raise_no_sections(folder, load_errors)

    return Dataset(folder=folder, sections=sections)
