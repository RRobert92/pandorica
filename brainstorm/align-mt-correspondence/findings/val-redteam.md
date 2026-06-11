# Red-Team Validation — align-mt-correspondence stack

**Seat:** adversarial validator. Did NOT generate the candidates. Job: find the *constructible*
failure, grounded in code + the empirical numbers + prior-art failures.

**Date:** 2026-06-11
**Code read:** `matcher.py`, `mt_endpoints.py`, `core.py::_bootstrap_correspondences`,
`register_warps_to_coarse`. **Numbers:** `dd-empirical.md` (H1/H2/H3). **Prior art:** Weber 2014
(PMC4249889), Track-Coalescence MHT/JPDA/BP (arXiv 2308.06326), QAP-LON density (arXiv 1107.4161),
naive-Bayes dependent-fusion (Kuncheva/ScienceDirect).

The five candidates lean on **three** empirical legs: H3 (ballistic 8×), H1 (warp necessary),
H2-relax (94% matches exist behind quality gates). Every leg has a measured caveat the candidates
quietly drop. I attack the drop.

---

## The single fact that reframes everything

`dd-empirical.md` H3 measured the 8× collapse **WITHIN a section** (bisect at 50 % Z, predict the
bottom-half XY from the top-half tangent). The candidates pitch it **ACROSS the lost slab.** The
same file says, in its own words:

> "**Cross-gap test (no alignment):** … ~50 % improvement rate with **near-zero median gain** —
> uninformative without alignment."

So the headline 8× number is **measured in the one regime where ballistic prediction trivially
works** (continuous traced curve, no missing material, shared coordinate frame) and is **explicitly
uninformative in the regime C1 actually runs in** (across the gap, before the warp converges). The
candidates inherited a within-section number and stapled it to a cross-section mechanism. That is
the load-bearing sleight in the whole stack.

And the mechanism needs a quantity the codebase does not have: `p̂ = p_xy + t_xy·(gap_z/dz)`. I
grepped the entire `stitch/` tree — **there is no lost-slab-thickness (`gap_z`) estimator, no tilt
(`tan θ`) model, nothing.** `mt_endpoints.py` emits `{id, pos, dir}` and `dir` is a crude
`seg[-1]-seg[0]` chord over the last ~20 % of points (line 84-90), not a fitted tangent. C1's own
"strongest open question" (estimating `gap_z`/`dz`) and DD3's `δJ²=ε²+(g·sec²θ·δθ)²` both hide the
fact that **every term in the ballistic prediction is an estimate stacked on an estimate.**

---

## C1 — Ballistic gate re-centering
**VERDICT: WEAK**

- **Strongest objection:** The 8× number is within-section; across the gap the same file measured
  *near-zero gain*. The prediction `p̂ = p_xy + t_xy·(gap_z/dz)` multiplies a noisy chord-tangent
  by an **uncalibrated, unmeasured `gap_z/dz`** — and the error grows as `gap_z·tan θ`. For a
  tilted MT (the only MTs that move laterally — vertical MTs jump ~0 and don't need C1) a tangent
  error `δθ` produces a landing error `gap_z·sec²θ·δθ`. At the high-tilt MTs that *most* need
  re-centering, `sec²θ` blows up and ballistic prediction lands **further from the true partner
  than the raw endpoint** — and worse, lands confidently *in a neighbour's lane*.

- **Evidence:** (1) Weber 2014 — the **parent algorithm pandorica forks** — states the straight-line
  cross-boundary assumption "may be problematic for microtubules parallel or nearly parallel to the
  boundary, and results should be carefully evaluated" there (PMC4249889). The original authors
  flagged exactly this failure. (2) H3's own cross-gap test: ~50 %/near-zero. (3) Code: `dir` is a
  20 %-window chord, and H3 itself reports **11 % of matched pairs on sec01→02 have |d·d| < 0.70
  (>45° tangent disagreement)** vs 2 % on healthy interfaces — i.e. on the hard interface the
  tangent is *already unreliable for 1 in 9 pairs*, and those are precisely the pairs C1 would
  extrapolate most aggressively. A 45° tangent error over a 300–350 nm slab (Weber's section
  thickness) is a ~300–350 nm landing error ≈ 0.4–0.5 ρ of *new* displacement — it manufactures a
  wrong neighbour hit rather than fixing a miss. (4) **Sign asymmetry of the harm:** a raw-endpoint
  miss fails *safe* (no join, abstain). A confident ballistic mis-landing fails *dangerous* (a
  wrong join with a small residual → survives the residual gate → feeds the TPS → whirlpool). C1
  converts safe misses into dangerous wrong joins for the tilted-MT subset.

- **What would change the verdict:** A cross-gap measurement *after* coarse alignment (not the
  uninformative no-alignment one) showing median ballistic residual beats raw endpoint **stratified
  by tilt angle** — specifically that it still wins for MTs with |tilt| > 30°. Plus a real `gap_z`
  estimator validated against section metadata. If C1 is **gated to fire only when in-plane tangent
  magnitude is small-to-moderate** (low tilt, where H3's within-section win plausibly transfers) and
  **disabled for high-tilt MTs**, it could move to PROMISING. As pitched (fire on all MTs, trust an
  unmeasured `gap_z`), it is WEAK — and it is partly **redundant** (see attack #3 / C5).

---

## C2 — Two-witness per-join confidence + honest abstention
**VERDICT: PROMISING**

- **Strongest objection:** The independence of L1⊥L2 is *assumed additive* (`score = L1 + L2`) but
  the realistic failure modes are **positively correlated**, so the product-of-FP-rates argument
  over-counts confidence — naive-Bayes overconfidence, the textbook failure of fused dependent
  classifiers. Both witnesses are evaluated *in the warped frame*; the warp is fit from the MT
  matches; a region where the warp is bad (high residual, the hard interface) is **simultaneously**
  where L1's ballistic residual is inflated AND where the face image is mis-registered so L2's NCC
  is degraded. The nuisance variable (local warp quality) drives *both*. That is exactly the
  `|ρ|>0.3` regime DD3's own kill-test names — and DD3 admits it would invalidate additivity.

- **Evidence:** Kuncheva / naive-Bayes-fusion literature: dependent classifiers fused under an
  independence assumption "produce wildly overconfident probabilities" and the designed-for gain
  "could be high only if the classifiers are statistically independent." The cryo-ET-specific
  killer DD3 itself concedes: featureless/knife-damaged tomograms make **L2 mute** (AUC ≤ 0.65),
  collapsing the two-witness system to **W1-only** — i.e. back to geometry, which is exactly the
  signal that failed. So on the datasets that are hard *because they are low-contrast*, C2 quietly
  degenerates to the baseline with a wider abstain band. The "fall back and abstain more" is honest
  but it means C2's precision gain is **anti-correlated with how much you need it.**

- **What would change the verdict:** Run DD3's own independence kill-test (regress L2 residual on
  L1 residual) on sec01→02 and report `ρ`. If `|ρ| < 0.3` empirically, additivity is safe and this
  goes SOLID. Plus an AUC(W2) measurement on the *real* Monopoles faces (binary sections — these
  may be especially L2-mute). Without those two numbers C2 is a well-reasoned design resting on two
  untested empirical assumptions. PROMISING because the *abstention machinery* (margin μ, group-as-
  centroid-to-warp + break-to-chain) is sound and decoupled even if L2 turns out weak.

---

## C3 — Crossing-penalty QAP matcher
**VERDICT: WEAK**

- **Strongest objection:** It must beat a baseline that **already does crossing resolution
  geometrically** — `uncross_pairs` (matcher.py:192) detects X-crosses among neighbours within
  `2ρ` and uncrosses-by-default, *plus* `filter_pair_smoothness` rejects the sideways-offset
  misjoin. The candidate frames C3 as "the density-breaker Weber lacks," but pandorica is not
  Weber-with-symmetric-smoothness; it is **LAP + ballistic-able + an explicit asymmetric uncrosser
  + a chord-tangent gate.** C3's only genuinely new lever over this is *global* (vs neighbour-local)
  crossing optimization — and DD2 itself says the crossing penalty "becomes noisy when d ≈ δJ" and
  IPFP "finds local optima; no global optimality guarantee." So at the exact `d < δJ` floor where
  density actually bites, the QAP edge term is as degenerate as everything else, and it pays days of
  complexity + a new CUDA/MPS dependency (pygmtools) for it.

- **Evidence:** (1) QAP local-optima networks are **provably dense** for QAP instances and "search
  difficulty increases with problem dimension" (arXiv 1107.4161) — IPFP on N~hundreds of
  near-identical parallel stubs lands in a local optimum with no guarantee it is the right
  permutation; the repetitive structure is the *worst case* for QAP, not the best. (2) DD2's own
  kill-test concedes the escape hatch: "If both produce the same wrong-join rate, … the ordering is
  already preserved by displacement coherence alone — in which case QAP = MRF in this regime and the
  baseline already works." The existing `uncross_pairs` *is* a displacement/order-preserving pass.
  So DD2 has pre-admitted the most likely outcome. (3) The crossing penalty assumes a **consistent
  lateral order is preserved across the slab** — but the slab is exactly where MTs the tracer
  couldn't see may have re-ordered; you cannot validate order-preservation in the unimaged gap.

- **What would change the verdict:** A head-to-head on sec01→02: `LAP+ballistic+uncross+smoothness`
  (the real baseline, not vanilla Hungarian) vs `+QAP-crossing`, measured wrong-join rate. If QAP
  cuts wrong joins by a margin that survives the GT-noise, SOLID. Per the brainstorm's own
  sequencing C3 is correctly deferred ("only if C1+C2 don't kill enough"); my objection is that the
  baseline it must beat is much stronger than "Weber," so the bar is higher than the candidate
  admits. WEAK = likely redundant, not wrong.

---

## C4 — Multi-section flow + triple-overlap consistency
**VERDICT: PROMISING (but mis-scoped as "global robustness")**

- **Strongest objection:** Min-cost flow **wants to commit for global consistency** — a
  detection/stub participates in exactly one trajectory and the objective *prefers* a complete,
  conserved flow. That structurally **fights C5/C2's per-join abstain**, which *wants to break*.
  DD2's own correspondence dive admits the limiting case: "if ALL sections have identically
  ambiguous interfaces … inside the region it still reduces to a pairwise coin-flip per interface."
  So in a uniform dense bundle — the hard case — flow does not add information; it adds a *global
  pressure to pick something*, which is the opposite of honest abstention. The triple-overlap
  curvature check is the genuinely valuable half and it does **not** require the flow machinery.

- **Evidence:** Network-flow tracking literature: "constraints ensure that a detection participates
  in one and only one trajectory" and the model uses unary enter/exit + pairwise link factors — the
  formulation is built to *link*, with abstention only as a costed sink. Combine with Track-
  Coalescence (arXiv 2308.06326): global association methods (MHT/JPDA/BP) **coalesce parallel /
  small-angle tracks** — exactly dense MTs — biasing two true tracks into one. A global flow over a
  dense MT bundle inherits this coalescence bias *and* the commit pressure. The free triple-overlap
  curvature-coherence check (`|Δcurvature| < κ_max` across k,k+1,k+2) is real and cheap — but it is
  separable from the flow and should be lifted out and run on the *pairwise* chains.

- **What would change the verdict:** Decouple. Adopt **triple-overlap curvature coherence as a
  standalone post-hoc chain validator** (it costs O(chain length), needs no solver, no Ultrack, no
  Gurobi) → that piece is PROMISING-to-SOLID. The **full min-cost flow** stays WEAK until there is
  evidence that real Monopoles/spindle stacks have *non-uniform* enough trajectories that flow beats
  pairwise+triple — DD2 itself says it only helps "at the edges of the ambiguous region."

---

## C5 — Partial decoupling: warp keeps MTs, chain uses the witness
**VERDICT: PROMISING**

- **Strongest objection:** The decoupling is **asymmetric in a way that can corrupt alignment.**
  C5 says: warp-trust a pair (geometric match) while chain-abstaining it (low two-witness). But if a
  pair is genuinely wrong (a dense-neighbour mis-join), C5 *still feeds it to the warp* — it only
  withholds it from the chain. So C5 fixes the *chain* precision (good) while leaving the **whirlpool
  failure mode fully intact**, because the bad pair still anchors the TPS. The brief lists three
  failures (miss / whirlpool / wrong-join); C5 addresses wrong-join-in-chain only, and by *keeping*
  the suspect pair in the warp it does nothing for whirlpool and may even legitimize a pair the old
  pipeline would have dropped at the residual gate. The "warp is tolerant to 1-to-many" claim is
  true for *centroid* anchors but a wrong *individual* pair is a coherent-looking outlier the rigid-
  residual gate (2ρ) may pass — and that is the whirlpool seed.

- **Evidence:** `register_warps_to_coarse` fits the warp from `id_pairs` directly (core.py:611-612);
  `chain.py` consumes the same `id_pairs`. C5 splits the *consumer* but both still read the *same
  matched set*. The independence the design needs (chain evidence ⟂ warp evidence) only holds if the
  chain's witness (C2/L2) is itself independent of the warp — and attack on C2 above shows L2 is
  *not* independent of warp quality. So C5 inherits C2's correlation bug: on a bad-warp region, both
  the warp anchor and the chain witness are simultaneously degraded, and C5's "trust warp / distrust
  chain" split is made on correlated evidence.

- **What would change the verdict:** Make the decoupling *symmetric* — allow a pair to be **chain-
  trusted AND warp-excluded** (a robust-loss / soft-weight on suspect pairs in the TPS), not just
  the reverse. Then C5 attacks whirlpool too. With that, and with C2's independence kill-test
  passing, C5 goes SOLID. As pitched (one-directional split, warp keeps everything) it is a chain-
  precision patch that leaves the alignment failure untouched — PROMISING, incomplete.

---

## Verdict on the STACK as a whole: **PROMISING but mis-ordered and internally contradictory**

1. **The 80/20 (C1+C2) rests its headline on the wrong measurement.** C1's 8× is within-section;
   the cross-section number in the *same file* is "near-zero gain." Until there is a post-alignment,
   tilt-stratified cross-gap measurement, the "highest-leverage, empirically validated" label on C1
   is not earned. The honest lead candidate is **C2's abstention machinery**, not C1's ballistic
   prediction.

2. **C1 may be redundant with what already works.** `_bootstrap_correspondences` (core.py:483)
   *already* recovers the displaced-but-correct pairs by iteratively re-warping the moving endpoints
   so the tight gates re-accept them — and dd-empirical measured this engine doing real work
   (sec01→02 **37 %→56 %**, +20pp). The 1.18 ρ residual is **not** an unbroken chicken-and-egg loop;
   the bootstrap already breaks it. C1 pre-shifts the query to do in one pass what bootstrap does in
   five — but bootstrap pulls pairs with a *smooth field* (a false neighbour "is not pulled
   coherently" — core.py:493), whereas C1 pulls with a *per-MT tangent* that has no such coherence
   guard. So C1 trades the bootstrap's built-in wrong-join resistance for speed. That is a *worse*
   precision posture, not a better one.

3. **C2/C5's abstain fights C4's flow.** Flow is built to commit (one detection → one trajectory);
   C5 is built to break. Named contradiction. Resolution: keep flow out, keep triple-overlap in.

4. **The compounding-overconfidence trap.** Each stage that adds a "confidence" or "consistency"
   score assumes its evidence is independent of the prior stages'. The warp residual, L1, L2, and
   the triple-curvature check are **all functions of the same warped frame**. On a bad interface
   they fail *together*. A stack that multiplies four correlated confidences will be **confidently
   wrong exactly where it matters** (the hard interface) and confidently right where it's easy
   (where the baseline already worked). This is the naive-Bayes overconfidence failure at the
   architecture level.

### #6 — What everyone missed (absent from brief.md AND candidates.md)

**MT termination / nucleation is treated as noise, but it is the dominant precision trap in
spindles — and it is invisible to every witness in the stack.** A real MT that simply *ends* in the
slab (depolymerization, a true minus-end, a plus-end that didn't reach the next face) presents
*identically* to a "miss": a stub with no partner. But a *new* MT that *nucleates* in section k+1
presents as a B-endpoint with no A-partner that **every candidate will try to explain**. C1 will
ballistically reach for the nearest A; C2's L1 will find a small residual to some neighbour; the
flow (C4) will *prefer* to link it (commit pressure) rather than open a trajectory mid-stack.
Nothing in the stack has a **birth/death model** — the network-flow literature has explicit
enter/exit costs precisely for this, and Weber's MRF had termination handling, but the candidates'
two-witness + QAP framing has *no node for "this MT is born/dies here."* In a dynamic spindle the
true MT count changes section to section (nucleation, catastrophe); forcing 1:1-ish matching onto a
population with real births and deaths manufactures wrong joins that **no consistency check can
catch, because there is no contradiction — just a confident link to an MT that should have had no
partner at all.** Density makes it worse: a freshly nucleated MT in a dense bundle always has a
plausible-looking ballistic ancestor. This is the failure mode the screenshots' "wrong joins" may
*actually* be — not neighbour-swaps (which uncross handles) but termination/nucleation
mis-attribution (which nothing in the stack models).

---

*Written 2026-06-11 by the red-team validator. Grounded against matcher.py, mt_endpoints.py,
core.py; numbers from dd-empirical.md; prior failures from Weber 2014 (PMC4249889), Track-
Coalescence (arXiv 2308.06326), QAP-LON (arXiv 1107.4161), dependent-classifier fusion (Kuncheva).*
