# napari Serial Section Stitcher

A napari plugin to **visually validate** the serial-section stitching pipeline on
real `.am` datasets, and to **record coarse-alignment ground truth** by hand.

Runs in any conda env with napari installed.

## Two widgets

### Serial Section Stitcher
1. **Browse folder** → a directory of section `.am` images + `*_spatialGraph.am`
   (paired by `sort_tomogram_files`). Optionally tick *Load volumes* (sets a
   *display downscale*, since volumes are multi-GB).
2. **Run stitch** → solves per-section poses, overlays the **aligned**
   microtubules (toggle the `raw:*` layers to compare), aligns the volume image
   layers in place (via layer affine — no resampling), and fills a per-interface
   **QC table** (match fraction, incoherence, warp/QC pass, plus the hybrid
   coarse angle + intensity verdict).
3. **Export** → warps the volumes and graphs under the solved poses into
   `<folder>/stitched_output/` (`stitched_volume.am`, `stitched_spatialGraph.am`,
   `stitch_log.txt`). Use *Export downscale* > 1 to validate a multi-GB stack fast.

**Warp options (export box):**
- *Apply TPS warp* — the fine guarded deformation (vorticity-bounded; softness via
  the *Warp vorticity max* knob in the Run box — lower = smoother).
- *Z-blend warp* — Z-varying symmetric warp (each section's two faces carry
  half their interface residual, blended through Z) instead of one warp per section.
- *Image-fill MT-free regions* — fits an extra image-derived warp (masked block
  matching) only in the gaps with no microtubules, confined there (zero-anchored at
  MTs) and carried through the same Z-blend. *Image-fill metric*: `mi` (mutual
  information — default, contrast/blur-robust), `grad` (edges), `ncc` (CLAHE).

**Performance (export box):**
- *GPU warp (mps/cuda)* — runs the volume resampling on the GPU (`torch
  grid_sample`, Z-chunked so memory is bounded; ~7× at full resolution). Auto-on
  when a GPU is present; leave off for downscaled previews (overhead-bound on small
  data). Device order: CUDA → MPS → CPU.
- *Match workers* — CPU processes for the image-fill matching (memory-safe: workers
  see only the small downsampled faces). ~2–3× on the slow `mi` metric.

### Coarse GT Recorder
Step through each interface *n→n+1*: the **fixed** top-face endpoints of *n* (blue)
and the **moving** bottom-face endpoints of *n+1* (orange ×) are shown. Dial in the
**angle** (slider / spinbox, rotation about the moving centroid) and **translation**
(spinboxes, or tick *Grab* and drag in the canvas) until the microtubules line up,
then **Save GT for interface**. Writes/updates `<folder>/coarse_gt.json`:

**Boundary-face images** (recommended for precise alignment): *Load face
projections (slow)* reads each volume once and makes a **Z-max projection** of the
top *N* slices of section *n* (shown **blue**, `bop blue`) and the bottom *N* slices
of *n+1* (shown **orange**, `bop orange`), overlaid additively. The moving (orange)
projection rotates/translates live with the controls — line up the image content,
not just the endpoints. Tune *# boundary slices*, *projection XY downscale*, and
*Invert Z* (if your stack is stored high-Z-first). These layers sit under the
endpoint markers; toggle/hide them like any napari layer, and flip a layer's
contrast in napari if features read inverted.

```json
{ "sec10->sec11": { "angle": -42.0, "tx": 130.5, "ty": -88.2, "n_fixed": 61, "n_moving": 58 } }
```

*Prefill from auto coarse* seeds the angle from the hybrid coarse estimate so you
correct rather than start cold.

## Launch

```bash
# Plugins menu (after registering the manifest — see below):
conda run -n tardis napari        # Plugins → Serial Section Stitcher / Coarse GT Recorder

# Headless, both widgets docked, optional folder pre-loaded:
conda run -n tardis python -m pandorica.napari /path/to/dataset
```

For the **Plugins menu** to find the widgets, the `napari.manifest` entry point
must be registered — re-run an editable install once after pulling this code:

```bash
conda run -n tardis pip install -e . --no-deps
```

## Conventions
- Graphs stay in **physical units (Å)**; image layers use `scale = pixel_size`
  so volume + microtubules overlay. napari axis order is `(z, y, x)`.
- Poses are 2-D **in-plane** `{Angle, Tx, Ty, Scale}`, section 0 = gauge anchor.
  Volume warping converts `Tx, Ty` to pixels (÷ pixel size).
- Face convention: **top = high-Z**, **bottom = low-Z**; the
  physically-adjacent pair is `top(n) ↔ bottom(n+1)`.
