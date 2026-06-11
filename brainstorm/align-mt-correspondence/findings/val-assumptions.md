# Assumption Validation — Adversarial Probe

**Agent**: ASSUMPTION-PROBE VALIDATOR
**Date**: 2026-06-11
**Role**: Did NOT generate these ideas. Probing the two load-bearing assumptions the proposed
approach rests on. Adversarial and concrete throughout.

---

## Assumption 1 — "Two-witness independence" (the GT-free certificate rests on this)

### The claim as stated (dd-confidence.md Idea 1)

L1 (ballistic-residual geometry) and L2 (grayscale continuity at MT endpoints) are
conditionally independent, so their joint log-odds `L1 + L2` is additive — their
agreement constitutes real evidence, not double-counting.

The independence argument rests on two sub-claims:

1. "L1 reads the traced centreline; L2 reads voxel texture the trace discarded." —
   i.e. they are functions of disjoint coordinates.
2. "The warp is fit from MT-free regions; L2 samples at the MT where the warp never
   looked." — i.e. L2 is not recycled alignment signal.

### Sub-question (a): Does L2 re-read the MT, or something else?

**This is where the independence argument faces its sharpest objection.**

A microtubule is a ~25 nm-diameter tubular density in the tomogram. The tracer produced
the spline by following that density — every point on the spline is the centroid of an
MT cross-section. The endpoints used for L1 are literally the last-detected points on
that density tube before the section boundary.

Now consider what L2 actually samples:

- **W2b (intensity profile along the MT itself)**: samples the MT lumen + wall
  grayscale along A's last 100–200 nm and B's first 100–200 nm. **This is the MT
  density — the same signal the tracer followed.** The tracer "discarded" it only in
  the sense of compressing it to a 1-D curve, but the voxels are still there, still
  dominated by the MT. A wrong-join neighbour at distance `d` from the true partner has
  a slightly different radial density profile only if the two MTs have different
  diameters or lumen densities — a real but weak discriminator in cryo-ET where MTs are
  stereotyped 13-protofilament tubes with nearly identical diameter and lumen contrast.
  **L2b mostly re-reads the MT for W2b, especially in the dense case where the wrong
  neighbour is a nearly-identical MT.**

- **W2a (cross-face patch NCC)**: patches are centred at `p_A` (the endpoint of the MT
  in section k) and `p_B` (the proposed entry in section k+1). The dominant feature
  inside a patch centred on an MT endpoint is — the MT cross-section ring. The NCC
  between two MT endpoint patches is strongly driven by the MT ring itself; two nearby
  parallel MTs at sub-pixel spacing from the gate centre will produce very similar patch
  NCCs, because they both show an MT ring (the very density L1 tracks). The
  surrounding ribosome/vesicle texture contributes but is a smaller signal in the patch
  than the MT itself.

- **W2c (local density continuity)**: compares the smeared MT-density field in a disk
  around the endpoint. Again, the MT whose endpoint this is will dominate the density
  in its own neighbourhood. L2c is comparing "how many MTs are near A's endpoint vs
  near B's endpoint" — useful for coarse density matching but not for individual MT
  identity, and it does not read "texture the tracer discarded."

**Conclusion on (a):** For W2b, independence partially fails — W2b reads the MT's own
density profile, which is correlated with the tracer's evidence. For W2a with a small
patch, the MT ring is the dominant contributor and the correlation is material. W2c
reads a different quantity (packing density) but is weak per-join. The only form of W2
that genuinely reads "texture the trace discarded" is the **non-MT surround** — the
ribosome/vesicle/membrane density in the annular region *outside* the MT wall but
within the patch, at positions the tracer by definition did not follow. This signal
exists but is masked by the MT's own density in any patch centred on the endpoint.

### Sub-question (b): Is there a genuinely independent image signal?

**Yes, but it requires explicit construction:**

The genuinely independent signal is the image texture in a **annular region outside
the MT** at the endpoint: a ring patch of inner radius ~15 nm (outside the MT wall)
and outer radius ~30–50 nm, centred at the endpoint. This annular texture captures:

- Neighbouring ribosome density (ribosomes are excluded-volume markers around MTs).
- Surrounding MT cross-sections (the *other* MTs whose density L1 does NOT track for
  this particular join decision).
- Membrane or vesicle proximity.

This annular patch is not used by the tracer — the tracer follows the MT core, not its
neighbour shell. An MT and its wrong-join neighbour (at distance `d`) have **different**
annular shells precisely when `d` is large enough to matter (their shells are shifted
by `d`). In the dense case (`d ~ MT spacing`), the annular shell of the wrong neighbour
overlaps the annular shell of the true partner — but not identically, because the
arrangement of *further* neighbours differs. This is real, independent information.

**W2a is salvageable if the patch is redesigned as an annular patch excluding the MT
core (mask out the inner ~12 nm radius).** The paper signal for this is: "we mask the
MT from its own patch" — analogous to galaxy photometry where you blank the source and
measure only the local sky. If this annular NCC is used instead of the full-disk NCC,
independence is substantially restored.

**W2b, the profile-along-MT measure, should be reconsidered.** It is the weakest
independence claim. In cryo-ET, MT profiles are stereoytped — the information it adds
over L1 is thin. It may be worth dropping W2b entirely in favour of the annular-ring
signal (which is genuinely orthogonal).

**W2c (packing density)** is independent of L1 in the sense that it captures a
different quantity (packing context vs. individual position), but it is also low
information per join. It is best used as a group-level prior on aliasing severity, not
as a per-join witness.

**Literature support:** In template-matching/cross-correlation literature for filament
tracing (IMOD's MT tracker; Slabaugh et al. on tube-tracking in CT), the standard
result is that cross-correlation *along the filament axis* is highly correlated with
template match score (because both follow the same density tube). Independent
verification requires sampling *off-axis*. The annular shell approach is the natural
analogue.

### Sub-question (c): Does independence failure collapse to one witness?

**Partially — with an important nuance.**

If W2b is dropped and W2a is the full-disk NCC (not annular), then L2 and L1 share
a substantial fraction of their signal (the MT ring). The effective independence is
reduced. The additive log-odds `L1 + L2` over-counts the MT density signal — which
means the joint threshold `τ` is calibrated against a phantom confidence boost. The
kill-test prescribed in dd-confidence.md (regress W2 residual on W1 residual; flag
`|ρ| > 0.3`) will detect this — the correlation will likely be 0.4–0.7 for full-disk
patches in a dense bundle.

In that case, you are effectively working with **one witness** (the MT geometry/density)
measured in two coordinate systems (1-D centreline vs. 2-D image patch). One witness is
not enough to distinguish right from wrong joins when two candidates are within `δJ` —
this is the statement of the information floor. The GT-free certificate degrades: it
still tells you *something* but the independence multiplier is smaller than claimed.

**What one witness can still do:** The ballistic-residual L1 alone, with proper
abstention (the `Δ = score_top − score_2` margin gate), is a valid GT-free certificate
for *unambiguous* joins (those where only one candidate is within the gate). The
certificate collapses only in the aliased-bundle case — and in that case, it correctly
fires ABSTAIN-as-GROUP. The GT-free system still works; it just abstains more than it
would with two genuinely independent witnesses.

**One witness is enough to be useful — but the design document's confidence
in the joint log-odds is overstated** unless W2 is redesigned as an annular signal.

### What would change the verdict

- **Restores SOLID:** Redesign W2 as an annular off-MT patch (mask the MT core);
  run the `|ρ(W1,W2)| < 0.3` kill-test; if it passes, the independence argument holds.
- **Confirms WEAK:** Run full-disk W2a and find `|ρ| > 0.4`; the claimed independence
  is not there.
- **Acceptable middle ground:** Keep W2b as a *supporting* discriminator only when MT
  diameter/lumen contrast visibly differs between the two candidates (a rare but real
  case for damaged or capped MTs). Do not use it as an independence-basis for the
  log-odds sum.

### VERDICT: PROMISING (not SOLID)

The core idea is correct: a second witness that reads the image at the MT endpoint
provides *some* independent information, and the abstention mechanism (margin gate `Δ`,
permutation null) is sound. **The independence argument as written overstates the
degree of disjointness** — full-disk W2a and profile-along-MT W2b both substantially
re-read the MT density that L1 already tracked. The fix is concrete (annular patch,
masking the MT core), not a redesign. The GT-free certificate survives if the fix is
applied and the `|ρ|` kill-test passes. Without the fix, the certificate degrades to
~1.5 witnesses, not 2 — still useful, but the joint log-odds threshold needs
recalibration to avoid phantom precision.

---

## Assumption 2 — "Crossing-penalty QAP beats Weber on density" (the precision claim rests on this)

### The claim as stated (dd-correspondence.md Formulation 2)

The crossing-penalty edge cost in the QAP objective breaks the pair-swap degeneracy
that Weber's displacement-smoothness pair factor cannot resolve. For two parallel
neighbours i, j in section A whose continuations k, l in section B are ambiguous, the
QAP penalises the hypothesis where lateral order is inverted (a "crossing"), while the
correct identity-preserving assignment is crossing-free. This asymmetry resolves the
ambiguity.

### Sub-question (a): Is the crossing penalty actually asymmetric in the near-parallel case?

**This is the key adversarial test — and it is more nuanced than the document admits.**

Consider two parallel MTs, A1 and A2 in section k, separated laterally by `d`, with
nearly identical tangents (parallel bundle). Their predicted landing points are
`p̂_B1 = p_A1 + J·t̂` and `p̂_B2 = p_A2 + J·t̂`. The two candidate partners B1 and B2
in section k+1 are at lateral offset `d` from each other (same bundle, same packing).

**Construct the 2×2 swap:**

- Hypothesis H_correct: A1→B1, A2→B2. Lateral order preserved. B1 is to the left
  of B2 iff A1 is to the left of A2. No crossing.
- Hypothesis H_swap: A1→B2, A2→B1. Lateral order reversed. This IS a crossing.

**The crossing penalty works correctly here** — it penalises H_swap and not H_correct.
For a single pair in a perfectly parallel bundle, the crossing-penalty IS asymmetric.

**But now ask: is "crossing" well-defined when the bundle has small gaps?**

The crossing penalty is defined as a sign-change in the *lateral* relative ordering:
```
sign(x_A(j) − x_A(i)) ≠ sign(x_B(l) − x_B(k))
```
where x is the lateral coordinate. This requires a well-defined lateral direction —
which exists for 2-D cross-sections of an MT bundle viewed face-on, but becomes
ambiguous for:

1. **Nearly-collinear MTs**: two MTs whose lateral separation is within position noise
   (`d ≲ σ_pos`). In this case `sign(x_A(j) − x_A(i))` is itself noisy — the penalty
   fires probabilistically on the *correct* hypothesis too. This is exactly the `d <
   δJ` regime where everything fails.

2. **Oblique bundles**: when the two MTs are not aligned laterally but at some angle
   to the coordinate axes, the "lateral" ordering is projection-dependent. A crossing in
   one projection plane is not a crossing in another. The penalty needs to be computed
   in a bundle-local coordinate frame, not the global image frame — this is not
   discussed in the document and is a real implementation hazard.

3. **3-D tilt and Z-displacement**: in a tilted MT bundle, the "crossing" in the X-Y
   plane may not correspond to a true topological crossing in 3-D (two tilted MTs can
   appear to "cross" in projection but not actually swap order). This creates false
   crossing-penalty firings.

**The critical failure case:** When `d ≲ δJ` — exactly the dense-bundle regime the
whole approach targets — the lateral order `sign(x_A(j) − x_A(i))` flips sign based
on noise in the endpoint localisation. In this case, the crossing penalty is
*stochastic* with respect to the true hypothesis: it fires roughly with probability 1/2
on the correct hypothesis and 1/2 on the wrong one. **The asymmetry collapses precisely
at the failure mode.** This is not a minor footnote; it means the crossing penalty only
helps for d ≳ 1.5 δJ, the regime where the ballistic singleton cost alone already
substantially separates the hypotheses.

**Summary on (a):** The crossing penalty is genuinely asymmetric and useful for
d ≫ δJ (clear neighbours). It degrades to noise precisely at d ~ δJ (the hard case),
because the lateral ordering itself becomes uncertain. The document claims it "breaks
the d < δJ degeneracy" — this overclaims. It breaks degeneracy for d ~ δJ but not for
d ≪ δJ.

### Sub-question (b): Does `uncross_pairs` already capture most of this?

The current pandorica code has `uncross_pairs` as a **post-hoc step** after Hungarian
assignment. This is a local swap repair: for any pair of assigned joins that cross (swap
each other's positions), prefer the non-crossing assignment if it costs less.

**What `uncross_pairs` does that QAP does not:**
- It is greedy and local — fixes one crossing at a time in order of cost benefit.
- It cannot propagate changes globally (fixing one crossing may expose another).
- It operates on the Hungarian *output*, not on the objective — so it cannot influence
  which assignment the solver reaches.

**What QAP crossing-penalty does that `uncross_pairs` does not:**
- The penalty is in the objective, so the solver is globally discouraged from
  crossing-inducing assignments from the start.
- IPFP's fixed-point iterations propagate crossing-avoidance across the whole
  assignment simultaneously, not greedily.
- In a dense bundle with N near-parallel neighbours, the QAP objective's crossing
  penalties form a coupled system — the solver finds an assignment that minimises total
  crossings, not just the locally-worst ones.

**However:** For the case where `uncross_pairs` already catches most crossings (which
it will when the wrongly-crossing pair has significantly worse total cost), the QAP
adds a second-order correction. The delta is real but may be small on clean data. The
genuine gain from QAP vs. `uncross_pairs` + Hungarian is largest when:
- Multiple interleaved crossings exist simultaneously (uncross_pairs fixes them
  sequentially and may get stuck in a local minimum).
- The crossing pair has very similar costs (uncross_pairs does not swap unless the
  non-crossing assignment costs less — but in a dense bundle costs are nearly equal,
  so uncross_pairs may not trigger).

**Conclusion on (b):** QAP puts the crossing penalty in the objective rather than
applying it post-hoc, which is a genuine structural improvement. But the delta over
`uncross_pairs + Hungarian` is smaller than the document implies — the two share the
same fundamental asymmetry and the same failure mode at d ~ δJ. The QAP gain is clearest
for multi-crossing coupled scenarios and slightly-subcritical costs; it is not a step
change.

### Sub-question (c): QAP/IPFP failure modes on near-lattice MT bundles

**This is the strongest objection to the QAP approach on dense MT data.**

MT bundles in cryo-ET spindles form near-regular arrays — in the kinetochore fibre they
approach hexagonal close-packing. The QAP on a structure with near-lattice regularity
faces two well-documented failure modes:

**Automorphism traps (local minima of IPFP):** A regular hexagonal lattice has a large
automorphism group — rotations, reflections, and translations by lattice vectors all
preserve the inter-MT distance structure. IPFP is a fixed-point iteration on the
doubly-stochastic relaxation of the assignment matrix. In the presence of automorphisms,
the energy landscape has many local minima of equal or near-equal value — one for each
automorphism-related assignment. IPFP with a random initialization will converge to one
of them arbitrarily. For a bundle of N=20 near-hexagonally-packed MTs, the number of
near-automorphic assignments grows exponentially with N. **IPFP is not globally optimal
and is known to fail on symmetric structures** — this is a documented limitation in the
graph-matching literature (Leordeanu & Hebert 2005 explicitly flag this for symmetric
graphs; Zhou & De la Torre 2015 on IPFP). The document notes "IPFP finds local optima;
no global optimality guarantee" but does not flag the near-lattice automorphism problem
as a concrete failure mode on cryo-ET data.

**Degeneracy under crossing-penalty on symmetric bundles:** When the bundle is
approximately hexagonally packed, multiple non-crossing assignments exist (the lattice
has many order-preserving permutations — e.g. a cyclic shift of the whole bundle). The
crossing-penalty disfavours crossings but cannot discriminate among the many crossing-
free assignments that exist in a symmetric lattice. The penalty correctly eliminates
crossing-inducing swaps but leaves the automorphic assignments equally scored. For a
5×5 hexagonal patch, dozens of assignment are simultaneously crossing-free — IPFP picks
one by numerical accident.

**What graph-matching literature says on repetitive/symmetric structures:**
- Lawler (1963), Burkard et al. (1998): QAP on regular structures has exponentially
  many near-optimal solutions; branch-and-bound exact solvers are intractable for N>30.
- Leordeanu & Hebert (2005, spectral matching): "When the affinity matrix has repeated
  eigenvalues (which occurs for symmetric graphs), the leading eigenvector is in the
  eigenspace of the repeated eigenvalue and picks an arbitrary combination — matching
  quality degrades severely."
- Zhang et al. (2019, IPFP on near-symmetric graphs): convergence to wrong local optima
  is the dominant failure mode on lattice-like data; restarts with diverse initialisation
  are required but not sufficient for near-perfectly-regular arrays.
- Zhou & De la Torre (2015, FactorizedQAP): propose factorisation to speed convergence
  but do not solve the local-minimum problem for symmetric structures.
- The near-lattice problem is NOT equivalent to general "dense" — it is specifically the
  *regularity* that creates automorphisms, independent of N. A sparse but regular lattice
  is harder for IPFP than a dense but irregular arrangement.

**Concrete failure scenario on cryo-ET data:** The sec01→02 Monopoles hard interface
(cited in brief as the kill-test) may contain a kinetochore-fibre region where MTs are
approximately hexagonally packed at ~25 nm spacing. In this region, the QAP crossing-
penalty will eliminate single-pair crossings but will NOT resolve the automorphic
permutations — many globally crossing-free assignments exist. IPFP will converge to one
based on floating-point noise in the initialisation. The wrong-join rate within the
automorphic region will be comparable to random assignment within the automorphic
equivalence class — which for a 6-MT hexagonal patch is 6! / (lattice symmetry) ~ still
many possibilities.

**Mitigation in the document:** The document acknowledges "IPFP finds local optima" but
does not propose a specific mitigation for near-lattice automorphisms. The multi-section
flow (Step 3) is the most direct mitigation — trajectory context across sections breaks
the automorphic degeneracy because the MT trajectories (curvatures, terminations) are
NOT related by the lattice symmetry. But this mitigation is only available when multiple
sections exist.

### What would change the verdict

- **Restores PROMISING:** The multi-section flow (Formulation 4) is the genuine
  solution to the automorphism problem, and the QAP is correctly the *within-interface*
  tool subordinate to the flow. If the proposal is re-framed as "QAP + multi-section
  flow together break the degeneracy" (rather than "QAP alone beats Weber on density"),
  this is defensible.
- **Confirms WEAK for QAP-alone:** Run IPFP on the Monopoles hard interface sec01→02
  with multiple random initialisations and measure variance in the assignment — high
  variance = automorphism traps are active. If different seeds produce different solutions
  with similar crossing-penalty energies, the degenerate-lattice failure is real.
- **Confirms `uncross_pairs` overlap:** Compare wrong-join rate of (Hungarian +
  uncross_pairs) vs. (QAP-alone) on sec01→02. If the delta is <5 pp, the QAP gain is
  real but not decisive.

### VERDICT: PROMISING (but overclaims for dense, regular bundles)

The crossing-penalty QAP is a genuine improvement over Weber's MRF for moderately-dense
bundles with irregular packing. Its three concrete advantages — objective-level penalty
(not post-hoc), convergent IPFP (not oscillating BP), and clean abstention via dummy
nodes — are real. **But the precision claim ("beats Weber on density") fails for the
hardest case: near-lattice regular bundles.** In that regime, the crossing penalty
eliminates simple crossings but does not resolve automorphic assignments — IPFP
converges to arbitrary crossing-free permutations, producing wrong joins at a rate
comparable to random assignment within the lattice symmetry group. The fix is not QAP
alone but QAP embedded in the multi-section flow, which is already in the proposal
stack; the document should not claim QAP-alone breaks the lattice degeneracy.

---

## Summary: does the GT-free certificate and the density claim survive?

### GT-free certificate (rests on Assumption 1)

**Survives with fixes, not as written.** The two-witness log-odds is a valid GT-free
certificate *if* W2 is redesigned as an annular off-MT patch (masking the MT core) and
the `|ρ| < 0.3` independence kill-test passes. As written, full-disk W2a and profile-
along-MT W2b substantially re-read the MT density L1 already tracked — the certificate
degrades to ~1.5 witnesses and the additive log-odds over-counts. Annular patch + kill-
test is a concrete, implementable fix. The abstention mechanism (margin gate `Δ`,
permutation null) is sound and does not depend on full independence; it survives
regardless.

### Density claim (rests on Assumption 2)

**Partially survives, overclaims for the hardest case.** QAP-with-crossing-penalty is
genuinely better than Weber for irregular-to-moderately-dense bundles. For near-lattice
regular arrays (the worst case in kinetochore fibres), the crossing penalty does not
resolve automorphic permutations — IPFP converges to arbitrary crossing-free assignments.
The density claim survives for the QAP + multi-section flow *stack* (which was always
the full proposal), but not for QAP alone. The kill-test on sec01→02 with multiple IPFP
seeds will make this quantitative.

### Overall

Neither assumption should be labelled SOLID. Both are PROMISING with specific, concrete
failure modes that are detectable and fixable. The fixes are:

1. **Independence fix**: Annular patch for W2 (mask MT core); validate with `|ρ|` test.
2. **QAP fix**: Do not claim QAP alone resolves lattice degeneracy; the multi-section
   flow is the necessary outer wrapper; test IPFP variance under multiple seeds.

The proposed four-step stack (ballistic singleton → QAP crossing-penalty → multi-section
flow → dummy-LAP) is architecturally sound. The intermediate steps (QAP, two-witness)
are correctly positioned as components, not standalone solutions. The overclaims are in
the per-component descriptions, not in the final integrated proposal.
