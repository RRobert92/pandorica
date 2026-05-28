# `serial_stitch` — serial-section tomogram stitcher

Automatic, no-landmark, whirlpool-safe stitching of a stack of serial-section
electron tomograms into one volume with a merged microtubule (MT) network.

The pipeline is **MT-graph-driven**: it aligns sections by matching the
microtubule endpoints that cross each section-to-section gap, with an image-only
fallback when graphs are absent. It is self-contained (no napari / Qt) and runs
headless on a workstation, a Mac (MPS), or a CUDA box.

```python
from pandorica.stitch.cli import run_stitch
run_stitch("path/to/sections")          # writes <dir>/stitched_output/
```

(also exposed as the `tardis_stitch` console script in `tardis_em`).

---

## Data model and units

* A section is an `[N, 4]` spatial graph of `[id, x, y, z]` points (microtubule
  centerlines); the in-plane warp acts on `(x, y)`, while `id` and `z` pass
  through untouched.
* All thresholds are expressed in **ρ**, the median nearest-neighbour spacing of
  a section (`transform/scale.py`), not raw pixels/nm. This makes the pipeline
  **portable across voxel sizes** — there are no hard-coded length constants.
* Convention (**Z-up**): a section's *top face* = high-Z, *bottom face* = low-Z;
  the physically-adjacent pair across the gap between section *n* and *n+1* is
  `top(n) ↔ bottom(n+1)`.

## Method

End to end, for a stack of *n* sections (`pipeline/stitcher.py` →
`pipeline/core.py`):

1. **Coarse rotation (per interface).** A *global* MT-endpoint rotation search
   sweeps θ ∈ [0, 360°) and takes the angle maximising the matched fraction,
   recovering large rotations including sign — where a local PCA seed cannot
   (`coarse/rotation_search.py`). A rigid **Coherent Point Drift** matcher
   (`coarse/cpd.py`) provides a decoy-robust, gate-free multi-seed search that
   recovers ±90° from a cold start. The search reports **degeneracy diagnostics**
   (peak margin, ±180° flip-ratio, PCA anisotropy, angular uniformity) so an
   ambiguous interface is *known* to be ambiguous.

2. **Sign resolution + ABSTAIN.** Bundled/symmetric endpoint constellations have
   a 180° flip ambiguity the point clouds can't resolve. The biological
   **anterior–posterior polarity** of the specimen — a signed, every-section
   density asymmetry (`coarse/ap_polarity.py`) — is used as the sign authority
   to pick the correct branch (`coarse/coarse_fusion.py`). When neither the MT
   search nor the A–P hint resolves an interface confidently, it **abstains**
   (is flagged) rather than silently mis-signing. A stack-wide **continuity**
   pass resolves a flagged interface toward the trend of the confident ones only
   when one of its branches clearly fits (`coarse/coarse_hybrid.py`).

3. **Endpoint matching.** An *enhanced Hungarian* matcher
   (`matching/matcher.py`) keeps a one-to-one assignment but adds: **ρ-normalised
   gates** (voxel-size portable), a **von Mises–Fisher** tangent term
   (`1 − |cos Δθ|`, sign-agnostic since MT polarity is ambiguous), and
   **outlier/duplicate rejection** via a robust rigid-fit residual gate *before*
   any warp. It reports a confidence record (match fraction, shift coherence).

4. **Relative rigid fit + global pose solve.** Each interface yields a relative
   rigid transform; absolute section poses come from a **global least-squares
   solve** (`transform/solver.py`, gauge-anchored at section 0) rather than
   greedy center-out chaining. Greedy chaining drifts — most damagingly in
   **scale**, which it multiplies section-to-section. The global solve adds a
   **scale→1 prior** and an optional **pose-smoothness prior** that couple the
   sections and beat greedy on accumulated drift.

5. **Guarded non-rigid warp.** A regularised **thin-plate-spline** displacement
   field is fit from the correspondences and **guarded by a diffeomorphism
   invariant** (`transform/warp.py`, `transform/diagnostics.py`): the field is
   sampled on a grid and accepted only if **det J ≥ ε AND |curl u| ≤ Ω**, both in
   ρ units. If it violates the invariant, smoothing is escalated and re-fit; if
   no allowed smoothing yields a safe field, the warp is **rejected** — an unsafe
   field is never applied. The curl bound matters as much as det J: a swirl can
   keep det J > 0 yet still be pathological.

6. **Verification + QC certificate.** Each interface gets a certificate
   (`pipeline/qc.py`) fusing the warp diffeomorphism gate and the matcher
   confidence, plus an **independent dense-intensity check** ("splines match,
   intensity verifies", `pipeline/intensity_qc.py`): the proposed rotation is
   applied to the moving boundary-face image and accepted only if it improves
   image agreement with the reference *and* beats the 180° flip. The gate is
   conservative — an interface is **accepted only if every gating check passes**,
   otherwise **flagged with reasons**, never silently stitched.

7. **Image-only fallback.** Sections without MT graphs are coarsely posed from
   image content alone (`image_pose.py`). Rotation comes from cell **geometry**
   (`contour_rotation.py`): the nuclear-envelope contour fixes the magnitude and the
   organelle constellation votes on the 180° sign. The sign is weak on near-circular
   cross-sections, so weak-flip interfaces are **flagged for review** (never silently
   committed); translation is then a block-match as in the MT path.

8. **Volume export.** Solved poses (and the guarded warps) are applied
   slice-wise into a shared canvas with optional Z-blended warp and image-fill of
   MT-free regions (`stitch.py`, `transform/applier.py`, `image_warp.py`); GPU is
   auto-selected and memory-bounded (`accel.py`). Outputs: a stitched `.am`
   volume, a merged `*_spatialGraph.am`, and a detailed log.

## Module map

```
cli              headless run_stitch orchestration + reporting
io               dataset discovery / loading (Dataset, load_dataset)
stitch           result accessors + stitched-output export
image_pose       image-only coarse poses (no-MT fallback)
contour_rotation image-only rotation from nuclear contour + organelle constellation
image_warp       image-fill residual warps for MT-free regions
geometry         pose <-> pixel math, boundary landmarks
match            image block-matching metrics (multiprocessing-safe)
accel            GPU/CPU device selection, memory-capped warp
coarse/          rotation search (CPD, sweep, A–P polarity, fusion, hybrid)
matching/        endpoint matcher + low-level MT-endpoint geometry
transform/       rigid solve, guarded TPS warp, slice applier, scale, diagnostics
pipeline/        registration core, serial-section orchestrator, QC + verification
```

## Related work

`serial_stitch` is a from-scratch Python reimplementation in the serial-section
EM lineage. Each entry below combines the citation with what `serial_stitch`
does differently. Cite the two ZIB papers when publishing — required by their
upstream licenses; the others are recommended for positioning context.

### Direct predecessor — SerialSectionAligner

Lindow, Brünig, Dercksen, Fabig, Kiewisz, Redemann, Müller-Reichert &
Prohaska, *"Semi-automatic stitching of filamentous structures in image
stacks from serial-section electron tomography,"* *Journal of Microscopy*
(2021), [doi:10.1111/jmi.13039](https://doi.org/10.1111/jmi.13039) —
[`zibamira/SerialSectionAligner`](https://github.com/zibamira/SerialSectionAligner).

The direct ancestor: per-interface MT ends + 3D directions → 2D
corresponding-point pairs → non-rigid alignment — the same core recipe used
here. SerialSectionAligner is *semi-automatic* (operator validates and
corrects); `serial_stitch` automates that workflow.

| | SerialSectionAligner (Amira rigid-MLS) | `serial_stitch` |
|---|---|---|
| Deformation | rigid Moving Least Squares — can fold/swirl ("whirlpools") | regularised TPS guarded by `det J ≥ ε ∧ \|curl\| ≤ Ω`; unsafe fields rejected, never applied |
| Thresholds | hard-coded pixel constants | everything in **ρ** (median NN spacing) → voxel-size portable |
| Bad correspondences | clustered/duplicate matches triggered whirlpools | explicit outlier/duplicate rejection before warping |
| Section poses | greedy chaining → exponential scale drift | global least-squares solve with scale→1 + smoothness priors |
| Coarse rotation | local seed; large/ambiguous rotations missed | global endpoint search + rigid CPD with degeneracy diagnostics |
| Sign ambiguity | resolved implicitly / can mis-stitch | biological A–P-polarity authority + ABSTAIN (never silently mis-signs) |
| Trust | operator review | per-interface QC certificate + independent dense-intensity verification |

### Earlier ZIB tool — microtubulestitching

Weber et al., *PLoS ONE* (2014) —
[`zibamira/microtubulestitching`](https://github.com/zibamira/microtubulestitching).

The boundary MT-endpoint **projection** approach in
`matching/mt_endpoints.py` follows this paper — the only external method
referenced in source.

### Dominant existing baseline — IMOD `etomo --join`

Mastronarde & Held, *"Automated tilt series alignment and tomographic
reconstruction in IMOD,"* *Journal of Structural Biology* **197**(2),
102–113 (2017),
[doi:10.1016/j.jsb.2016.07.011](https://doi.org/10.1016/j.jsb.2016.07.011) —
[bio3d.colorado.edu/imod](https://bio3d.colorado.edu/imod/). License: GPL-2.0.

The dominant serial-section tomogram joining workflow today. Image-only:
`xfalign` for cross-correlation coarse alignment, `midas` for manual GUI
fix-up, `tomojoin` for rigid join with optional linear stretch.
`serial_stitch` adds an MT-graph prior (biology-aware, not only image-driven),
a global pose solve instead of per-section chaining, a regularised TPS warp
guarded by the diffeomorphism invariant (rather than rigid + linear stretch),
A–P-polarity as sign authority, and ABSTAIN + per-interface QC instead of
relying on operator review in `midas`.

### Contemporary Python sibling — msemalign

Watkins, Jelli & Briggman, *"msemalign: a pipeline for serial section
multibeam scanning electron microscopy volume alignment,"* *Frontiers in
Neuroscience* (2023),
[doi:10.3389/fnins.2023.1281098](https://doi.org/10.3389/fnins.2023.1281098)
— [`mpinb/msemalign`](https://github.com/mpinb/msemalign). License: GPL-3.0.

The most prominent contemporary Python serial-section aligner. Image-only,
petabyte-scale mSEM — different EM modality and scale from `serial_stitch`'s
tomograms. Shares the "register then refine" structure; does not address the
rotational sign degeneracies that motivate `serial_stitch`'s A–P-polarity
and ABSTAIN machinery, since those don't arise in mSEM imaging the same way.

### napari-EM precedent — Okapi-EM

Perdigão, Ho, Cheng, Yee, Glen, Wu, Grange, Dumoux, Basham & Darrow,
*"Okapi-EM: A napari plugin for processing and analyzing cryogenic serial
focused ion beam/scanning electron microscopy images,"* *Biological Imaging*
**3**, e9 (2023),
[doi:10.1017/S2633903X23000119](https://doi.org/10.1017/S2633903X23000119) —
[`rosalindfranklininstitute/okapi-em`](https://github.com/rosalindfranklininstitute/okapi-em).
License: Apache-2.0.

Closest napari-EM precedent for `pandorica.napari`. Handles cryo-FIB/SEM
slice-to-slice alignment via SIFT plus charging-artefact filters specific to
FIB-SEM. The alignment problem (consecutive ablated FIB-SEM slices, microns
thick) is upstream of serial-section tomogram stitching — slice spacing,
deformation modes, and biological priors all differ.

### Algorithmic ingredients

- **Coherent Point Drift** — Myronenko & Song, 2010. `coarse/cpd.py` is an
  independent NumPy reimplementation of the *rigid* CPD core (GMM-EM with
  an explicit uniform-outlier term — no external CPD code is used). It is
  used here only as a robust, gate-free coarse *matcher*; the non-rigid
  stage is the guarded TPS, not the CPD/BCPD non-rigid extension (Hirose,
  2021), so the matching keeps CPD's probabilistic robustness while the
  deformation is made provably safe rather than only smooth.
- **Thin-plate splines** — Bookstein, 1989. The non-rigid warp basis.
- **scipy primitives** — `optimize.least_squares` (global pose solve),
  `optimize.linear_sum_assignment` (Hungarian matching), `ndimage`
  (resampling).

## Future work

- [ ] **Tune A–P-polarity on real EM volumes** — the density definition
  (`dense_is_dark`), frame/nucleus masking, and low-pass scale in
  `coarse/ap_polarity.py` are still calibrated on limited data and need
  validation across stains/contrasts.
- [ ] **Resolve flagged interfaces beyond ABSTAIN** — add a stronger sign
  authority and/or an operator-assisted path so a genuinely ambiguous large
  rotation is corrected, not just flagged.
- [ ] **Large-rotation canvas efficiency** — a genuine large inter-section
  rotation (e.g. ~90°, when tomograms are mounted/imaged rotated) is *correct*,
  but it produces a large, partly-empty output canvas. This is **not** an error to
  guard against — large rotations are often real. The streaming export already
  keeps RAM bounded; the remaining cost (disk + warp time over empty regions)
  could be cut by allocating/writing only the occupied canvas tiles instead of the
  full bounding box.
- [ ] **Loop-closure / multi-gap MT constraints** — the global pose solve
  currently decouples per interface (no shared landmarks across gaps); coupling
  MTs that span multiple sections would let the solve do more than priors.
- [x] **Stream volume writing** — the warp output streams to raw temps via
  `f.write` and the `.am` is assembled with `to_am_streamed` (`copyfileobj`), so
  the full stitched volume is never held in RAM (export peak ~3.5 GB at full res,
  down from ~11 GB). The remaining peak is the input section volume (loaded whole)
  + canvas grids — see "lazy input slabs" below.
- [ ] **Lazy input slabs** — load each section volume in Z-slabs aligned to the
  warp chunk instead of the whole multi-GB volume up front, removing the
  input-volume term from peak memory.
- [x] **Image-only fallback rotation** — recover rotation from cell *geometry*
  instead of dense intensity (`contour_rotation.py`): the nuclear-envelope contour
  gives the magnitude (≈5–13° of the MT truth on FemalePN, where the old NCC search
  was wrong by 120–170°), and the organelle constellation votes on the 180° sign.
  The sign vote is weak on near-circular faces, so weak-flip interfaces are **flagged
  for review**, never silently committed. (Tuning the flip vote / segmentation across
  more specimens remains open.)
- [ ] **Broader validation** — exercise across more datasets, voxel sizes, and
  section counts; expand the regression suite with real-data fixtures.
