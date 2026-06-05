# Changelog

All notable changes to pandorica are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.6] — 2026-06-05

### Changed

- **The CPD coarse rotation search now runs its seeds in parallel.** The
  multi-seed CPD search is the dominant cost of MT-based stitching (profiling
  put it at ~85% of the solve), and its angle seeds were evaluated one after
  another. Because each seed's CPD EM is GIL-releasing numpy, the seed sweep
  in `pandorica.stitch.coarse.cpd.cpd_rotation_search` now fans the seeds out
  across a `ThreadPoolExecutor` (one worker per seed, capped at
  `os.cpu_count()`). Results are **byte-identical** to the old serial sweep —
  the seeds are independent, the EM is deterministic, and `map` preserves
  order — so no stitch outcome changes. On an 11-section test stack
  (10 interfaces, 12 seeds) the coarse stage dropped from 33.2 to 7.3 s per
  interface (4.6×) and the full MT solve from 391 to 134 s (2.9×); machines
  with more cores scale further, up to the seed count.

### Added

- **Live per-interface progress for the MT solve.** `stitch_sections` and
  `register_section_stack` gained an optional `progress` callback that fires
  once per interface, tagged with the current phase; the CLI now prints
  `[coarse] interface k/n` then `[register] interface k/n` as the solve runs,
  so a long stack streams progress instead of going silent until it returns.

## [1.1.5] — 2026-06-04

### Added

- **Cross-section filament chaining.** New `pandorica/stitch/chain.py` reuses
  the matcher's per-interface microtubule pairings to merge the per-section
  spline IDs of one microtubule into a single global filament ID (union-find
  over `(section, mt_id)`, unioning only across accepted interfaces and
  breaking at flagged ones), plus block orientation, joint splitting, and
  per-point/edge diagnostic labels (`chain_filaments`, `orient_chain_blocks`,
  `split_chains_at_joints`, `compute_chain_labels`). The exported spatial
  graph now has one connected spline per microtubule instead of one disjoint
  spline per section. CLI and `StitchValidatorWidget` forward
  `interface_id_pairs` / `interface_accepted` to the exporters.
- **Spatial-graph inspector for napari.** A `SpatialGraphInspectorWidget`
  (`pandorica/napari/_widget.py`, registered in `napari.yaml`) for inspecting
  and filtering filaments and joints; the reader attaches per-edge properties
  to the Shapes layer and can show joints as Points coloured by
  `JointAngleDeg`.
- **Image↔MT dual-chain cross-check.** `reconcile_image_mt` compares the
  independent image-pose and MT (spatial-graph) rotation estimates per
  interface and selectively overwrites an MT pose when the image is more
  certain (gated translation overrides, with a detailed report). A
  boundary-contour estimate in `contour_rotation.py` adds a second,
  geometry-based opinion.
- **Register compute-time breakdown and cross-check progress.** The CLI
  splits the register stage into `mt-solve` / `cross-check` (MT path) or
  `image-pose` (image-only) and reports each in the compute-time summary, and
  the image-candidate harvest now prints a progress line per interface so the
  cross-check no longer looks frozen.

### Changed

- **Image-pose stage rewrite (~5× faster on the test stack).** The image-only
  coarse pose now uses a confidence-weighted RANSAC rigid fit that picks the
  rotation by inlier support, abstains from the translation when the inlier
  fraction is too low, and breaks branch ties with a central-disk NCC. The
  block-matcher (`match.block_match`) switched from a per-call multiprocessing
  pool to a `ThreadPoolExecutor`, avoiding process spawn / pickling overhead.

### Fixed

- **`pandorica stitch` crashed when run with no arguments.** The Click option
  builder set both `required=True` and `default=None`, but Click does not
  enforce `required` when a default is present, so a bare `pandorica stitch`
  (or one missing `--input-dir`) reached the solver with `input_dir=None` and
  crashed instead of showing usage. Required parameters now omit the default,
  and the bare command prints help (`no_args_is_help`).

## [1.1.0] — 2026-05-31

### Added

- **napari reader plugin for AmiraMesh `.am` files.** Drag-and-drop or
  `File → Open` of `.am` files now opens them as the right layer type:
  spatial graphs (`*_spatialGraph.am`, or any file whose header declares
  `VERTEX` / `EDGE` / `HxSpatialGraph`) become a Shapes layer with each
  filament rendered as a connected `path`; volumes (`Lattice` header)
  become Image layers with isotropic Å scale. Implemented in
  `pandorica/napari/_reader.py`; registered via the `napari.yaml` manifest.
- **`Browse files…` button** in `CoarseGTWidget` / `StitchValidatorWidget`
  for picking individual `.am` files (volumes + spatial graphs) instead
  of pointing at a folder. Auto-pairs volumes with their matching
  `_spatialGraph.am` by stem containment, falls back to graph-only
  sections for unpaired graphs.
- **Spatial graphs render as splines** by default in the napari widget
  (toggleable via `Render spatial graphs as splines` checkbox; falls
  back to the prior point-cloud display when off). New helper
  `pandorica.napari._geometry.coords_to_paths_zyx(coords)` groups
  `[N, 4]` per-segment-id, preserves in-segment order, drops
  single-point segments, and converts to napari `(z, y, x)` order.
- **`allow_scale: bool = False` and `lambda_scale: float = 1.0` kwargs**
  exposed on `pandorica.stitch.cli.run_stitch`. Forwards to
  `stitch_sections` so per-section isotropic scale estimation and the
  scale→1 prior are now opt-in from the CLI / `tardis_stitch` (the
  flags were already in `stitch_sections` but were never wired through).
- **`rich.progress.Progress` bar for export warping.** Replaces the
  prior N-line text progress with a single live-updating bar (description,
  percent, elapsed/remaining). Falls back to plain prints when stdout
  isn't a TTY (piped logs, CI capture) so log files stay clean.

### Reverted before release

A short-lived `Pose.Flip` plumbing (flip-aware Procrustes + Fourier-Mellin
flip enumeration + napari widget flip checkbox + GT JSON `"flip"` field)
was prototyped against this version but proved too unreliable on real
EM cross-sections during field testing. Removed from production before
release; the prototype code is preserved at `tmp/flip_apparatus/` for
possible future revival. See that folder's `README.md` and project
memory `project_image_only_ceiling.md` for the empirical findings.

### Notes for tardis_em integration

The `run_stitch` signature grew from 15 to 17 kwargs (`allow_scale`,
`lambda_scale`). `tardis_em`'s introspection wrapper picks these up
automatically after `pip install -U pandorica`; no manifest changes
needed on the tardis side.

## [1.0.3] — 2026-05-30

### Changed

- **File reorganization** to separate format-level I/O and general utilities
  from stitcher-domain code, so future pandorica tools can reuse them
  without importing from `pandorica.stitch`:
  - `pandorica.stitch.amira` → `pandorica.io.amira` (new package).
    `sort_tomogram_files` joined it from `pandorica.stitch.io` because it
    is Amira-folder discovery, not stitcher logic.
  - `pandorica.stitch._pointcloud` → `pandorica.utils.pointcloud` (new
    package).
  - `pandorica.stitch.io` → `pandorica.stitch.dataset` (rename only; the
    file is the stitcher's `Section`/`Dataset` data model, not general
    I/O — the old name was misleading).
- No public-API signature changes; only import paths moved. Downstream
  callers update their imports:

  | Was | Becomes |
  | --- | --- |
  | `from pandorica.stitch.amira import …` | `from pandorica.io.amira import …` |
  | `from pandorica.stitch._pointcloud import pc_median_dist` | `from pandorica.utils.pointcloud import pc_median_dist` |
  | `from pandorica.stitch.io import Dataset, load_dataset, Section` | `from pandorica.stitch.dataset import Dataset, load_dataset, Section` |
  | `from pandorica.stitch.io import sort_tomogram_files` | `from pandorica.io.amira import sort_tomogram_files` |

## [1.0.2] — 2026-05-30

### Added

- **Native AmiraMesh I/O** at `pandorica.stitch.amira`:
  `read_spatial_graph`, `read_segmented_points`, `read_amira_volume`,
  `write_spatial_graph`, `write_amira_volume_streamed`, plus a
  `SpatialGraph` dataclass for lossless round-trip. Supports ASCII and
  binary spatial graphs (read **and** write) with arbitrary per-vertex /
  per-edge / per-point int and float label/scalar fields. Validated
  bit-equal against the previous `tardis_em` readers on Monopoles_test
  and C.elegans_FemalePN, and against the previous `tardis_em` writers
  for V2-schema synthesis.
- **`pc_median_dist`** ported into `pandorica.stitch._pointcloud`.
  Bit-equal on the `avg_over=False` path (the only path the stitcher
  exercises).
- **`pandorica` console script** with rich-styled terminal UI:
  `pandorica stitch ...` is a click subcommand whose flags are
  auto-derived from `inspect.signature(pandorica.stitch.cli.run_stitch)`
  and whose help text is pulled from the function's docstring. Adding a
  kwarg to `run_stitch` makes a new flag appear in `pandorica stitch
  --help` with no changes to the CLI code. Rendering covers startup
  banner panel, cyan section rules, in-stream `ok`/`FAIL` colouring,
  yellow warning panel for the image-only branch, red error panel for
  cannot-stitch failures, and a green summary panel with output paths
  and elapsed time.
- **Saved log header** (`stitch_log.txt` next to the stitched volume)
  now includes pandorica version, run date, project URL, license, the
  full BibTeX citation, and every reproducible kwarg (15 fields in the
  Settings block — input/output dirs, downscale, all warp/GPU/coarse/MT
  flags, workers). Header content is identical whether invoked via
  `pandorica stitch`, `tardis_stitch`, or `run_stitch` from a script.

### Changed

- ASCII float emission in `write_spatial_graph` upgraded to `%.17e`
  (IEEE-754 round-trip-safe for float64) — fixes a 1-ULP drift on
  ~0.6 % of values that `%.15e` could not round-trip.
- ASCII spatial-graph parser now reads float fields directly into
  float64 (was: dropped through float32 and lost ~7 decimal digits
  before upcasting). Vertex and point coordinates are returned as
  float64 regardless of source dtype.
- `Section.load_volume` no longer triggers a wasted spatial-graph parse
  as a side-effect of loading the image volume.
- Citation guidance softened in `README.md` and `pandorica/stitch
  /README.md`: removed the inaccurate claim that prior-work citations
  (Lindow 2021 / Weber 2014) are required by upstream licenses.
  Pandorica is a from-scratch reimplementation that does not include
  code from those projects; those citations are scholarly courtesy for
  positioning context, not a legal obligation imposed by pandorica.
- BibTeX `version` field in `README.md` updated to match the package
  version.

### Removed

- Runtime dependencies on `tardis_em` and `tardis_em_analysis`. Replaced
  module-for-module:
  - `tardis_em.utils.load_data.ImportDataFromAmira` →
    `pandorica.stitch.amira.read_segmented_points` and
    `pandorica.stitch.amira.read_amira_volume`.
  - `tardis_em.utils.export_data.to_am_streamed` →
    `pandorica.stitch.amira.write_amira_volume_streamed`.
  - `tardis_em.utils.export_data.NumpyToAmira.export_amiraV2` →
    `pandorica.stitch.amira.write_spatial_graph`.
  - `tardis_em_analysis.utils.pc_median_dist` →
    `pandorica.stitch._pointcloud.pc_median_dist`.
- User-facing strings that enumerated `tardis_em` / `tardis_stitch` as a
  consumer of pandorica (a downstream wrapper detail leaking into
  upstream docs). Removed from `pandorica/stitch/__init__.py`,
  `pandorica/stitch/stitch.py` log header, `pandorica/stitch/README.md`,
  and `pandorica/napari/README.md`.
- Stale `requires_tardis_em` skipif markers from
  `tests/test_napari_stitch.py` (their guarded tests no longer depend
  on `tardis_em`).

### Dependencies

- Added `click >= 8.0`, `rich >= 13.0`, `docstring_parser >= 0.15` for
  the new `pandorica` CLI.

### Companion change (in `tardis_em`)

- The `tardis_stitch` console-script wrapper in `tardis_em` was rewritten
  to derive its flag set from `inspect.signature(run_stitch)` at import
  time. Adding a kwarg in pandorica's `run_stitch` now surfaces in
  `tardis_stitch --help` after a `pip install -U pandorica` — no
  `tardis_em` release required. (Change lives in the `tardis_em` repo;
  it depends on pandorica via the `[stitch]` extra with `docstring_parser`
  added.)

## [1.0.0] — 2026-05-30

### Added

- GPU chunk auto-sizing (`gpu_chunk=None` → sizes from free CUDA VRAM,
  clamped to `[1, 64]` slices; falls back to 4 on MPS).
- Coarse warp displacement grid (`warp_coarse_px=8`) with bilinear
  upsample, replacing per-pixel scipy `RBFInterpolator` evaluation —
  ~21× faster on the export path, sub-pixel error.
- `trim_to_mts` option (with `mt_pad_frac` padding) to size the export
  canvas to the microtubule bounding box instead of the section corner
  bbox.

### Changed

- Project migrated to PEP-621 `pyproject.toml` layout; assets added.

## [Initial commit] — 2026-05-28

- Forked out of `tardis_em_analysis.serial_stitch` to relicense under
  PolyForm Noncommercial 1.0.0. Same author, same algorithmic
  pipeline; rename only.
