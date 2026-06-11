# Validation (Phase 5)

Two fresh validators (no ownership bias): a red-team on the whole stack (opus) and an
assumption-probe on the two load-bearing claims (sonnet). Full files: findings/val-redteam.md,
findings/val-assumptions.md.

## Per-candidate verdicts (red-team)

| Cand | Verdict | Strongest objection |
|------|---------|---------------------|
| C1 ballistic re-centering | **WEAK** | The 8× is WITHIN-section (dd-empirical.md:167 confirms cross-gap is *untested*, needs aligned frames). No `gap_z/dz` estimator exists anywhere in `stitch/`; `dir` is a crude 20%-chord, not a tangent. Error grows `gap_z·sec²θ·δθ` → on high-tilt MTs lands WORSE than raw endpoint, and fails *dangerously* (confident wrong join) vs the raw miss's safe fail. `_bootstrap_correspondences` already breaks the 1.18ρ loop (37%→56%) with a smooth-field guard C1 discards → partly redundant. |
| C2 two-witness + abstain | **PROMISING** | Abstention machinery sound. Risk: L1⊥L2 additivity — both degrade together on bad-warp regions → naive-Bayes overconfidence; L2 goes mute on low-contrast tomograms exactly where it's needed. |
| C3 crossing-QAP | **WEAK** | Must beat existing `uncross_pairs` + `filter_pair_smoothness` (already asymmetric crossing resolution), NOT vanilla Weber. QAP/IPFP local-optima are provably dense on repetitive (near-lattice) structure; DD2's own kill-test pre-admits "QAP = baseline in this regime." |
| C4 multi-section flow | **PROMISING but mis-scoped** | The triple-overlap curvature check is the valuable, *separable* half. The min-cost flow itself *wants to commit* (one-trajectory constraint + coalescence on parallel tracks) and structurally fights C2/C5 abstention. |
| C5 partial decoupling | **PROMISING, incomplete** | One-directional: warp keeps EVERY pair, so it patches chain precision but leaves the whirlpool intact — a wrong pair still anchors the TPS. Must also drop low-confidence pairs from the warp. |

## Stack verdict: PROMISING but mis-ordered and internally contradictory

**Most dangerous objection (the architectural one):** all four "confidences" (warp residual, L1
geometry, L2 image, triple-curvature) are **functions of the same warped frame**, so on a hard
interface they fail TOGETHER — the stack is confidently wrong exactly where the baseline already
struggled, and confidently right where it already worked. Mitigation = at least one signal that is
NOT derived from the single warped frame (annular off-MT image patch; cross-3-section cycle).

**The thing everyone missed (#6): no birth/death model.** MT nucleation/termination presents
identically to a miss. Every candidate will confidently link a freshly-nucleated MT to a ballistic
"ancestor," and NO consistency check catches it — there's no contradiction, just a wrong join to an
MT that should have had no partner. **Plausibly the real source of the screenshot "wrong joins,"
not neighbour-swaps.** This is a missing CANDIDATE, not a tweak.

## Assumption probe

- **A1 two-witness independence — PROMISING (not SOLID).** The image witness as proposed (on-MT
  patch / intensity-along-MT) RE-READS the MT's own density — the same signal the centreline was
  traced from → NOT independent. **Fix: annular OFF-MT patch** (mask the inner ~12 nm core, sample
  only surrounding ribosome/vesicle texture), then run the `|ρ|<0.3` kill-test. Without it the
  certificate is ~1.5 witnesses and miscalibrated. The abstention machinery (margin gate,
  permutation null) survives regardless.
- **A2 crossing-QAP beats Weber on density — PROMISING (overclaims).** Crossing penalty is
  genuinely asymmetric for `d ≫ δJ` (clear neighbours) but collapses to noise at `d ~ δJ` (the hard
  case). Near-lattice hexagonal MT arrays have large automorphism groups → IPFP lands on arbitrary
  crossing-free permutations → wrong joins at ~random within the lattice symmetry class. The
  multi-section flow is the *necessary outer wrapper* that breaks the automorphic degeneracy via
  trajectory context. "QAP alone breaks `d<δJ`" is FALSE for regular bundles.

Both load-bearing claims survive for the INTEGRATED stack; neither survives as a standalone
per-component claim.
