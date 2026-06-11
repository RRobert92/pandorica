# Brief — Tomogram alignment + MT cross-section correspondence

Confirmed by user 2026-06-11. Deliverable: **review then ideate** (audit current pipeline,
then a ranked, validated shortlist of better approaches). Scope: **full rethink on the table.**

## Problem

Pandorica stitches serial cryo-ET sections of *dense* microtubule (MT) networks. Each section is
a tomogram in which MTs are traced as splines (polylines / spatial-graph edges). Two goals,
currently produced by **one endpoint-Hungarian matcher per interface**:

1. **Align** consecutive tomograms — rotation / translation / anisotropic scale / fine warp.
2. **Connect** each MT spline to its true continuation in the next section (cross-section
   identity), so the exported spatial graph has one filament per MT instead of one stub per
   section.

Because a single Hungarian assignment of boundary endpoints feeds **both** the alignment warp and
the chain builder (union-find), the two goals poison each other: a bad match warps the volume
wrongly, and a bad warp moves endpoints so the match fails. Chicken-and-egg with no independent
signal to break the loop.

## Observed failure triad (from user screenshots)

- **Misses (recall).** MT stubs that obviously continue across the cut are left unconnected.
- **Whirlpools (alignment).** Soft swirl / foldover-like distortion in the fused warp field.
- **Wrong joins (precision).** Occasional kinked chains where two different MTs got merged.

## Success criteria

- **Robust with NO manual ground truth** and no per-dataset tuning.
- High connection **precision AND recall** (both miss and wrong-join must drop).
- **Alignment quality**: no whirlpool / foldover, low residual.
- **Honest abstention** where genuinely ambiguous (better a clean break than a wrong join).
- GT exists but is noisy: Amira-curated, and because MT positions drift after stitching the GT
  itself needs re-matching — so GT can only be an *optional spot-check*, never the training/main
  loop. Core method must be GT-free.

## Hard constraints

- Signals available: MT spline geometry (endpoints, tangents, full polylines, local density)
  **plus the underlying EM image volume** (both sections' grayscale).
- Must keep working through native Amira `.am` I/O, the napari plugin, and the CLI.
- Must run on CUDA (Windows, primary) and MPS (Mac, dev).
- Must stay compatible with the `tardis_em` coupling (tardis pins pandorica>=1.0.3).

## Out of scope / already ruled out

- Image-only **flip** auto-detection (proven unreliable in practice, reverted before 1.1.0).
- Requiring hand-labelled GT in the main loop.

## Current architecture (one line)

load → coarse (spline-PCA + oriented ICP, optional image rotation seed) → **match** (enhanced
Hungarian: ρ-scaled distance gate + vMF tangent term + dedupe + uncross + smoothness gate +
rigid-residual outlier reject) → relative rigid fit → guarded TPS warp → per-interface QC →
global pose solve. The **same** match pairs drive `chain.py` union-find for MT identity.
