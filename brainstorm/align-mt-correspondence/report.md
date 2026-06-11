# Brainstorm report — Tomogram alignment + MT cross-section correspondence

Workspace entry point: `MAP.md`. Brief: `brief.md`. This report is validation-adjusted — the
ranking below is NOT the pre-validation enthusiasm; two front-runners were demoted by evidence.

## What the pipeline established (the reframe)

1. **Full decoupling is impossible — the matcher stays.** The most popular landscape idea ("align
   on the image, drop the MT matcher") was empirically refuted: image-coarse alignment leaves a
   ~1.18ρ residual on the hard interface, the MT fine warp is what pulls it to 0.15ρ (8×), and the
   image-fill warp itself depends on the MT warp. MT correspondences are necessary for *alignment*,
   not just chaining.

2. **The three symptoms have distinct, located causes.** Misses ← real pairs rejected by the
   rigid-residual/smoothness gates because the ~1ρ coarse residual displaces true partners outside
   tight gates (37%→94% matches when gates open), NOT the z-band (loosening it = +3..17 pairs).
   Whirlpools ← bad correspondences feeding the TPS (the guard then rejects the warp, so you lose
   it). Wrong joins ← forced 1:1 assignment + loose gates on dense parallel stubs **and** —
   validation's key addition — MTs that should have had NO partner (nucleation/termination) being
   confidently linked anyway.

3. **The deepest architectural risk:** every confidence signal in the natural design (warp
   residual, geometry, image, consistency) is computed in the *same warped frame*, so they fail
   *together* on hard interfaces. Robustness requires at least one signal independent of that frame.

## Ranked shortlist (validation-adjusted, against: robust-without-GT, precision+recall, clean alignment, honest abstention)

### 1. Honest no-partner / birth–death model  — the precision fix that was missing
- Pitch: stop forcing every MT to have a continuation; model "this MT correctly has no partner"
  (it terminated, or its neighbour was freshly nucleated) as a first-class outcome.
- Mechanism: dummy-augmented / unbalanced assignment where the no-match cost encodes a birth/death
  prior; commit a partner only on a margin over the no-match option; abstain-as-group when two
  partners sit within the measured jump noise δJ.
- Provenance: red-team #6 (the missed failure mode), first-principles (group-when-aliased),
  contrarian (abstention-as-product), impl/cross-domain (MHT births/deaths, partial OT, Ultrack).
- Verdict: **PROMISING**, elevated by validation. Plausibly addresses the *real* source of your
  screenshot "wrong joins."
- Kill-test (do this FIRST — it reorders everything below): take the existing wrong joins and
  classify each into (a) neighbour-swap, (b) link to an MT that should have had no partner
  (birth/death), (c) gate-rejected miss elsewhere. The dominant bucket decides where effort goes.

### 2. Triple-overlap / cycle consistency  — the one check independent of a single warp frame
- Pitch: a wrong join across k→k+1 that survives one interface creates a curvature/cycle
  contradiction across k→k+1→k+2; use that as a free correctness check.
- Mechanism: extend `split_chains_at_joints` from one joint to a 3-section window; require
  curvature coherence (`|Δcurvature|<κ_max`) and pose-cycle closure across the triple.
- Provenance: DD3 (triple-overlap), red-team (named the *separable, valuable half* of C4).
- Verdict: **PROMISING.** Partly escapes the correlated-confidence trap because it spans three
  frames, not one. Cheap relative to full global flow.
- Kill-test: on a stack with known-bad chains, does the triple check flag them while leaving smooth
  chains intact?

### 3. Two-witness confidence — ONLY with the annular off-MT image patch
- Pitch: certify each join with geometry AND a genuinely independent image witness.
- Mechanism: L1 = ballistic-residual likelihood (centreline). L2 = grayscale continuity in an
  ANNULAR patch around the MT (mask the inner ~12 nm core) so it samples the surrounding texture
  the trace discarded — not the MT itself. Combine as log-odds; gate on score + margin.
- Provenance: DD3 (two-witness), assumption-probe A1 (the annular fix + the `|ρ|<0.3` independence
  kill-test).
- Verdict: **PROMISING with the fix; WEAK without it** (on-MT patch just re-reads the MT → ~1.5
  witnesses, miscalibrated).
- Kill-test: regress L2 residual on L1 residual across many joins; `|ρ|>0.3` ⇒ not independent,
  certificate invalid.

### 4. Bi-directional partial decoupling (revised)
- Pitch: split the two consumers of `id_pairs` — but drop low-confidence pairs from BOTH the warp
  and the chain, not just the chain.
- Mechanism: warp uses only high-confidence geometric matches (so a wrong pair can't anchor the
  TPS → kills whirlpools at the root); chain commits on the two-witness + triple-consistency score.
- Provenance: contrarian (cut-the-last-wire), DD1 (forces partial-not-full), red-team (the
  one-directional incompleteness).
- Verdict: **PROMISING.** This is the whirlpool fix; finishes the `qc.chainable` half-decoupling.
- Kill-test: refit the warp dropping the lowest-confidence decile of pairs on a whirlpool interface
  — does foldover/curl drop without alignment residual rising?

### 5. Ballistic tangent as a cost term (downgraded — not the hero)
- Pitch: use the tangent-predicted landing as the matcher's query/cost, with guards.
- Verdict: **WEAK as billed.** The 8× is within-section; cross-gap value is *untested* (needs
  aligned frames) and there is no `gap_z/dz` estimator; on high-tilt MTs it can land worse than the
  raw endpoint and fail *dangerously*; `_bootstrap_correspondences` already does much of the
  loop-breaking. Adopt only with (a) a gap-thickness estimator and (b) a tilt-gated fallback to the
  raw endpoint — then it's a useful cost term, not a standalone fix.
- Kill-test: on aligned frames, measure cross-gap landing error of tangent-shifted vs raw endpoint,
  stratified by MT tilt; find the tilt above which it hurts.

### 6. Crossing-penalty QAP (downgraded — only inside global flow)
- Verdict: **WEAK standalone.** Must beat the existing `uncross_pairs`+`filter_pair_smoothness`,
  not vanilla Weber; near-lattice MT bundles trap IPFP in automorphic local optima at exactly the
  hard spacing `d~δJ`. Only worth it wrapped in multi-section flow (which breaks the automorphism
  via trajectory context). Revisit only if 1–4 don't kill enough wrong joins.

## Graveyard (do not re-propose without new evidence)
- **Full decoupling / drop the matcher / warp from image only** — REFUTED empirically (image coarse
  1.18ρ; MT warp needed for 0.15ρ; image-fill depends on MT warp).
- **z-band loosening as the misses fix** — WEAK (+3..17 pairs; the gates downstream are the cause).
- **Ballistic re-centering as a standalone 80/20 hero** — WEAK (within-section only; dangerous on
  tilt; partly redundant with the existing bootstrap loop).
- **"QAP alone breaks dense-parallel degeneracy"** — FALSE for regular bundles (automorphism traps).
- **Gromov-Wasserstein as the primary solver** — impractical (N⁴ memory, lattice automorphisms).
- **Weber 2014 MRF/BP as-is** — its smoothness factor is symmetric under a neighbour swap → can't
  disambiguate parallel neighbours; the user's own lineage, and its documented density failure is
  the bar to beat.
- **Two-witness with an on-MT image patch** — not independent (re-reads the traced MT).
- **Learned matcher (SuperGlue/LightGlue)** — needs training data the GT-free constraint forbids;
  SuperGlue also non-commercial.
- **Trace the whole stack as one volume (delete the section boundary)** — out of pandorica scope
  (a TARDIS-upstream tracing change); parked, not killed.

## Open questions validation could not settle
- **Birth/death vs neighbour-swap: which dominates the real wrong joins?** The #1 kill-test settles
  it and reorders the list. Until measured, effort allocation between candidate 1 and candidate 6
  is a guess.
- **Does cross-gap ballistic prediction beat the raw endpoint at all?** Untested (chicken-and-egg:
  needs aligned frames). Needs a `gap_z/dz` estimator first.
- **Does the annular off-MT witness pass `|ρ|<0.3` AND have usable contrast** on real tomograms?
- **How many of the four confidence signals are genuinely warp-frame-independent?** (annular patch
  and the triple-cycle are the only candidates; if neither holds, the correlated-confidence risk is
  unmitigated.)

## The single recommended next action
Run the wrong-join classification (candidate 1's kill-test) on an existing stitched stack. It is
cheap, needs no new code beyond a diagnostic, and it decides whether this is a *precision/abstention*
problem (births/deaths + consistency) or a *matching* problem (crossing/QAP) — which is the fork the
entire roadmap hinges on.
