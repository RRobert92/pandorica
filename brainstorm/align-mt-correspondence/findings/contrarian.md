# Contrarian / Reframe — attack the problem statement

Read the code, not just the brief. The brief says "ONE Hungarian matcher does BOTH align
and connect." That was true in `register_section_stack`. But the repo has **already half-escaped
that frame**: `image_only_poses` (image_pose.py) fixes every section's global pose from the
*image*, and `register_warps_to_coarse` (core.py) now fits the fine warp *relative to that image
coarse* and explicitly refuses to re-fit a rigid/affine from MT matches ("MTs don't carry the
global aniso — the image does"). `image_warp.py` already taps the MT-free regions from image
correspondences. So the real situation is sharper than the brief:

**The remaining poison is not "alignment vs connection share a matcher." It is that the FINE warp
and the CHAINS still share ONE Hungarian assignment (`id_pairs` feeds both `fit_guarded_warp` and
`chain_filaments`), and the rescue/gate paths (`_rescue_relative`, `gate_coarse_scale`) still let
MT correspondences leak back into the POSE.** The architecture started decoupling and stopped
two-thirds of the way. The contrarian move is to finish the divorce, and then notice the leftover
"matcher" has almost nothing left to do.

---

## The sharpest reframe (one sentence)

**Stop matching endpoints. Solve a single global geometric field (alignment) from the image +
the *whole* MT field as a deformation prior, then read correspondence off the field as a
by-product — and where the field is locally ambiguous, emit an honest break instead of a guess.**

Correspondence is not the input to alignment. It is the *output* you sample after alignment is
already done. Every wrong-join, whirlpool, and miss in the triad is a symptom of running that
arrow backwards.

---

## The assumption most worth deleting

> **"Each MT has a continuation in the next section, and our job is to find which one."**

Delete it. Reframe the unit of truth as the **deformation field**, not the **match**. The biologist
does not need "filament 47 = filament 312 next door"; they need (a) a correctly aligned stack and
(b) chains they can *trust*. A continuation either falls out unambiguously from the field (commit
it), or it doesn't (break it, honestly). "Find the partner for every stub" is the assumption that
manufactures wrong joins and whirlpools — because Hungarian is **forced to assign**, and a forced
assignment in a dense bundle is a coin flip dressed as a decision. The endpoint match was never the
goal; it was a proxy we mistook for the goal.

---

## How to GUARANTEE the failure triad (then negate each)

Mechanisms that *produce* misses / whirlpools / wrong joins — i.e. the recipe for failure, inverted
into design rules:

- **To guarantee WRONG JOINS:** force a 1:1 assignment over every stub (Hungarian assigns even the
  hopeless), then *trust the warp to that assignment*. → **Rule: never force assignment. Correspondence
  must be allowed to output ∅. Commit only matches whose posterior partner mass is concentrated; the
  rest are breaks.** (Today `_assign` keeps every finite-cost pair; the gates *subtract* afterward —
  wrong order. Gate the *probability*, don't post-filter the assignment.)
- **To guarantee WHIRLPOOLS:** drive a flexible warp (TPS) from sparse, possibly-wrong landmark pairs,
  in regions where the landmarks cluster or contradict. A few swapped/duplicate pairs put opposing
  vectors into a smooth interpolant → swirl/foldover. → **Rule: the warp must not be driven by
  discrete matches at all. Drive it from a DENSE field (image flow + MT density/orientation field),
  where no single bad correspondence can curl it.** The current `dedupe`+`reject_outliers`+foldover
  guard are *patches* on a warp that is structurally vulnerable because its support is discrete.
- **To guarantee MISSES:** make the matcher's gates depend on the very alignment it is trying to
  produce (distance gate in ρ, tight rigid-residual gate). A slightly-off coarse pose displaces a
  true partner past the gate → dropped. → **Rule: correspondence must be read in a frame already
  aligned by an INDEPENDENT signal (image), so the gate is generous and the true partner is always
  inside it.** The `_bootstrap_correspondences` loop is the codebase admitting this: it iteratively
  pre-warps to drag partners back under the tight gate. That is a workaround for matching in the
  wrong frame.

Negation summary: **abstention-by-default, dense-field warp, image-aligned frame.** All three failures
share one root — *the matcher both consumes and produces the alignment.* Cut that loop and the triad
loses its mechanism.

---

## Ideas (ranked: delete-the-problem first)

### Field-first alignment, correspondence as a readout
- **Pitch:** Alignment is solved entirely from image + MT *field* (density/orientation), never from
  discrete matches; each stub's continuation is then just "where does this endpoint land, and is one
  partner unambiguous?" — a nearest-neighbour readout with an abstain option.
- **Mechanism:** (1) Image coarse pose (already exists: `image_only_poses`). (2) Fine warp from a DENSE
  objective: image optical-flow/block-match field + an MT-orientation-field consistency term (the
  warped moving tangent field should align to the fixed tangent field everywhere, not at endpoints).
  (3) After warp, for each fixed-face endpoint compute the *distribution* over candidate partners (soft
  assignment / entropy). Commit only low-entropy ones to chains; the rest are breaks.
- **Inspiration:** Optical-flow / Demons registration drives deformation from the whole image, not
  landmarks; correspondence-free ICP variants (CPD as a density-to-density fit — `cpd.py` already here).
- **Key assumption:** The MT *field* (orientation/density), unlike individual endpoints, is a stable,
  dense, alignment-grade signal even across the lost slab.
- **Kill-test:** On Monopoles sec01→02 (the hard interface) and sec09→13, fit the fine warp with ZERO
  endpoint matches (image + orientation-field only); measure residual + foldover vs the current
  match-driven warp. If field-only warp is within noise and never swirls, the matcher is not needed
  for alignment.

### Abstention as the product (precision-first fragments, not speculative filaments)
- **Pitch:** Attack the success criterion "one filament per MT." Default to **high-precision fragments**;
  only fuse across a cut when the continuation is near-certain. A clean break is a feature, not a defect.
- **Mechanism:** Replace the hard Hungarian + post-gates with a per-stub *continuation posterior*. Emit
  a chain edge only when posterior mass on the top partner exceeds a margin AND no runner-up is close.
  Everything else exports as an honest stub with a "break: ambiguous" tag (the chain QC fields already
  exist — `edge_was_split`, `JointOverallDeg`). Ship a single knob: target precision; recall floats.
- **Inspiration:** Multi-object tracking's "tracklet" philosophy — short confident tracklets beat long
  hallucinated tracks; conservative data association (gating + abstain) over greedy global assignment.
- **Key assumption:** Biologists prefer a fragment they can trust to a long filament that might be two
  MTs. (The brief literally says "better a clean break than a wrong join" — this just makes it the
  default policy instead of a fallback.)
- **Kill-test:** Show the user current long-chain output vs precision-first fragments on a wrong-join
  case from their screenshots. If they pick the fragments, the "one filament per MT" target is wrong.

### Delete the section boundary: trace the stack as ONE volume, never split
- **Pitch:** The section-by-section framing IS the mistake. If the MT field were aligned as a volume
  first and *then* traced through the aligned stack, there is no cross-section correspondence problem —
  a filament is one connected component, full stop.
- **Mechanism:** (1) Coarsely align the raw volumes (image-only, already feasible — `image_only_poses`).
  (2) Stitch into one volume with a Z-gap mask where the slab was lost. (3) Trace MTs through the fused
  volume with a tracer that *bridges short masked Z-gaps* using local tangent extrapolation. Chains are
  then connected components, not matches. Correspondence is deleted, not solved.
- **Inspiration:** Whole-volume neuron tracing (FFN/flood-filling, automated EM connectomics) traces
  *through* the volume rather than matching per-section segmentations; serial-section reconstruction by
  3D segmentation, not 2D-then-link.
- **Key assumption:** A tracer can be trusted to bridge the lost-slab gap from tangent continuity better
  than the endpoint matcher can — i.e. the *gap-bridging* sub-problem is genuinely easier than the
  *global-assignment* sub-problem. (Plausible: a tracer uses the local field; the matcher fights global
  density confusion.) Compatibility caveat: tardis_em hands pandorica *pre-traced* per-section splines,
  so this needs a re-trace stage or a tardis change — flag, don't assume.
- **Kill-test:** Take two sections, fuse into one masked volume, run a simple tangent-extrapolation
  bridge across the gap; compare the connected components to the Hungarian chains on a labelled stub.
  If bridging matches/beats Hungarian on the dense interface, per-section matching is obsolete.

### Cut the last coupling wire: chains and warp must use DIFFERENT evidence
- **Pitch:** Even keeping per-section matching, the warp and the chains must not be the SAME `id_pairs`.
  Warp from the dense/image signal; let chains be a *separate, stricter* read on the already-warped field.
- **Mechanism:** `register_warps_to_coarse` keeps fitting the warp from `image_warp` + orientation field
  (no endpoint matches). Chains are computed *after* warp by a separate, conservative readout with its
  own abstain threshold. A bad chain can no longer corrupt the warp; a smooth warp can no longer be
  blamed for a missed chain. The rescue/gate paths (`_rescue_relative`, `gate_coarse_scale`) stop
  feeding MT-fit affines back into the POSE — that backflow is the last place a wrong match becomes a
  wrong alignment.
- **Inspiration:** Bundle adjustment separates the *geometry estimate* from the *feature-track gating*;
  you never let a single bad track bend the global solve.
- **Key assumption:** The image + orientation field alone is a sufficient warp driver (same assumption
  as idea 1; this is the minimal-change version of it).
- **Kill-test:** Ablation — drop `id_pairs` from the warp fit, keep it only for chains. If alignment QC
  (foldover, residual) is unchanged or better, the warp never needed the matches.

### Anti-idea (state it to kill it): keep buffing the Hungarian
- **Pitch (to reject):** Add more gates — better vMF, smarter uncross, a learned cost. The current file
  already has dedupe + uncross + smoothness(×2 gates) + signed-foldback + rigid-residual + vertical-jog +
  bootstrap. Seven correction layers on a 1:1 assignment is the tell: the *base operation is wrong*, not
  under-tuned. Each gate trades a wrong-join for a miss; you slide along the precision/recall line, never
  off it. **Reject all incremental Hungarian work.** The only move that leaves the line is changing the
  object (field, not endpoints) and the policy (abstain, not assign).

---

## Direct answers to the inversion prompts

1. **Right object?** The **field** (MT orientation/density + image), not the boundary endpoint. The
   endpoint is the *most* impoverished view of an MT — one point, one noisy tangent, polarity-ambiguous —
   and it lives exactly at the cut where signal is worst. Matching the richest object's poorest feature.
2. **Delete it?** Yes, two ways: (a) image-only alignment → correspondence becomes trivial NN-with-abstain;
   (b) trace-the-volume → correspondence is connected components, gone entirely. Section-by-section is the
   mistake.
3. **Easier moot problem?** Per-stub continuation *distribution* + commit-only-the-unambiguous. Warp from
   image/field, matches are a readout. Both already half-built in this repo.
4. **Attack "one filament per MT"?** It manufactures wrong joins. Replace with "as long a filament as the
   evidence certifies; break otherwise." Precision-first fragments.

## What I'd build first (cheapest decisive test)
The ablation in **idea 4** + the field-only warp in **idea 1** share one kill-test and need no new infra
(image_warp + cpd + orientation fields all exist): **fit the fine warp with zero endpoint matches on the
two hard Monopoles interfaces.** If it holds, the entire pairwise-endpoint-Hungarian frame is demoted from
"core algorithm" to "optional chain readout you're allowed to abstain from."
