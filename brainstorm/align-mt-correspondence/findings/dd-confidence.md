# Deep-Dive — The intrinsic, GT-free signal that decides a cross-section MT join

Seat: hardest-reasoning. One job: design a signal, computable from the data alone,
that says **CORRECT / WRONG / ABSTAIN** for each cross-section join, and the
consistency structure that makes the whole stack self-checking — with no labels.

Grounded against live code (paths are load-bearing):
- `matcher.py::_confidence` returns only **per-interface aggregates** (`n_matches`,
  `match_fraction`, `mean_cost`, `shift_incoherence_rho`). **There is no per-join
  confidence today.** That is the hole this design fills.
- `intensity_qc.py::image_similarity` already computes zero-mean NCC of two face
  images; `match.py::block_match_ncc` does masked sub-pixel NCC with a
  `min_peakiness` (peak-vs-background ratio). The image witness is *half-built* —
  it is used as a coarse rotation guard, never per join.
- `mt_endpoints.py::extract_boundary_endpoints` keeps only `{id, pos, dir}`, but the
  **full polyline** (`coords`) is available upstream — so ballistic prediction and
  trajectory descriptors are recoverable without new I/O.
- `chain.py` breaks chains only at **QC-rejected whole interfaces**; there is no
  per-join abstain. `split_chains_at_joints` is a pairwise overall-direction check —
  the seed for cycle consistency, but it never spans more than one interface.
- `transform/scale.py::rho` gives median NN spacing per section — the density unit
  the null model needs.

The two upstream agents already named the right pieces (ballistic residual; geometry
× image two-witness; group-when-aliased; neighbour fingerprint). My job is to make
them a **single calibrated decision** with the independence argument, the threshold
quantity, the null model, and the kill-test each. Nothing here is "add a confidence
score."

---

## 0. The one equation everything hangs on

For a candidate join (A in section k, B in section k+1), after the image-led
alignment has put both faces in a common Å frame, define the **ballistic landing
point** of A:

```
    p̂_B = p_A + (g / t_z) · (t_x, t_y)          # tangent-predicted re-entry
    r    = ‖p_B − p̂_B‖                          # ballistic residual (Å)
```

`g` = lost-slab thickness (estimated per interface, §2), `t` = A's exit tangent
(from `dir`, or better from the last-N polyline points). This `r` is the spine. Every
witness below is, in effect, an **independent way of asking whether `r` is small for
the right reason** — not by accident of a dense neighbour sitting where A's true
partner should be.

The disease the brief describes — *"a wrong join in a dense parallel bundle looks
locally fine: distance small, tangents parallel"* — is precisely the case where `r`
is small but the join is wrong, because a **neighbour** of the true partner also sits
near `p̂_B`. So `r` alone is necessary, not sufficient. We need a second witness that
**does not see geometry at all**, and an abstention rule that fires exactly when two
candidates are both inside `r`'s noise.

---

## Idea 1 — Two-witness confidence: geometry × grayscale, and *why* they are independent

### Pitch
Trust a join only when **two witnesses built from disjoint evidence** agree:
(W1) the geometric ballistic residual `r` is small relative to the *measured* jump
noise, and (W2) the **EM grayscale continues across the cut along the predicted
tangent** above the local background-continuity floor. Independent agreement is real
evidence; one witness alone is not. The output is a **per-join log-odds**, not a
boolean — so the chain builder can threshold it at any operating point.

### Mechanism (the actual computation)

**Witness W1 — geometric ballistic log-likelihood.**
Model the verified-join residual as a 2-D Gaussian cloud of width `δJ` (the jump
uncertainty, §2). For candidate partner B:
```
    L1(B) = −‖p_B − p̂_B‖² / (2 δJ²)          # log-likelihood of B being A's landing
```
This is the geometry witness. It uses **endpoint position + tangent** only.

**Witness W2 — grayscale continuity along the predicted tangent.**
This must use *different pixels than the warp used*. Concretely, extract a small
oriented patch and compute continuity, in three escalating forms (pick the cheapest
that passes its kill-test on a given dataset):

- **W2a — cross-face patch NCC at the join.** Take a `(2h+1)²` patch of the section-k
  *top* face centred at `p_A` and a same-size patch of the section-(k+1) *bottom* face
  centred at `p_B`, both at the 5th–10th slice in from the cut (the outermost slices
  are knife-damaged — genomics' "don't trust the contig end" lesson, cross-domain
  Field 7). Reuse `intensity_qc.image_similarity` / `block_match_ncc`. Value =
  zero-mean NCC.
- **W2b — intensity-profile continuity along the MT itself.** Sample the grayscale
  *along* A's last 100–200 nm and along B's first 100–200 nm (the MT lumen + wall has
  a characteristic radial density profile). A true continuation has a **continuous
  intensity profile** across the splice; a wrong join splices two profiles with a
  step. Value = 1 − |profile step| / profile std. This is the strongest discriminator
  because it is the MT's *own* signature, not the surround.
- **W2c — local density continuity.** Compare the smeared MT-density field (Gaussian
  KDE of endpoints) in a disk around `p_A` vs around `p_B`. A true partner lands in a
  neighbourhood of the same local packing; a wrong join often lands in denser/sparser
  territory.

`W2 = ` the supremum over {a,b,c} that clears its dataset's background floor (§ null
model). Convert to log-odds via a logistic fit against the **background patch
correlation distribution** (NCC of A's patch against *random aligned-neighbour*
patches at the same interface — this is computed for free while scanning candidates):
```
    L2(B) = log [ P(NCC | continuation) / P(NCC | random neighbour) ]
```
The denominator is the empirical histogram of NCC between A and the *non-partner*
endpoints within the gate — a per-interface null that needs no labels.

**Combine.** Because W1 and W2 are conditionally independent given the true partner
identity (argument below), the joint log-odds is **additive**:
```
    score(B) = L1(B) + L2(B)                    # per-join confidence (log-odds)
```
Commit the top-scoring B **only if** (i) `score(B_top)` exceeds a fixed log-odds
threshold `τ` AND (ii) the margin to the runner-up `score(B_top) − score(B_2)` exceeds
a separation `μ` (this second clause is the abstention trigger, §2). Everything else
→ ABSTAIN.

### Why W1 and W2 are statistically independent (the load-bearing argument)

This is the crux the brief demands, and it has two layers:

1. **Different physical quantities.** W1 is a function of *endpoint coordinates and
   tangents* — geometry of the traced centreline. W2 is a function of *voxel
   intensities* in patches that the tracer reduced away. The trace is a lossy
   projection of the volume to a 1-D curve; the residual intensity texture (lumen
   contrast, surrounding vesicles/ribosomes/membrane, local density of *other* MTs) is
   exactly the information **thrown away** when the spline was fit. Two functions of
   disjoint coordinates of the data are independent of each other's *noise*.

2. **Different — and crucially, decoupled — failure modes.** The geometry witness
   fails when a *neighbour* happens to sit at the ballistic landing point (dense
   parallel bundle): a purely **positional** coincidence. The image witness fails when
   the cut-face grayscale is featureless or knife-damaged: a **textural** failure.
   For these to *jointly* fail by chance, a wrong neighbour would have to *both* land
   at `p̂_B` *and* carry a grayscale patch that matches A's surround better than the
   true partner's does. Those two events are driven by unrelated nuisance variables
   (lateral packing geometry vs. cut-surface texture), so their joint false-positive
   probability is the **product**, not the max — which is why agreement *multiplies*
   confidence and a single witness cannot.

   The independence is **conditional on the alignment** (both witnesses are evaluated
   in the same warped frame), but — and this is the subtle, essential point — **W2
   must not be the same signal that drove the warp.** The image-led warp (per the
   first-principles/contrarian agents) is fit from `block_match` over the **MT-free**
   regions (`image_warp.py` masks out the MTs). W2 reads the grayscale **at the MT
   endpoints** — the masked-out region the warp never used. So W2 is independent of
   the warp *by construction of where it samples*. If you ever fit the warp from MT
   patches, this independence collapses and W2 becomes double-counting — that is the
   trap to avoid.

### Inspiration
Genomics long-read spanning + read-depth (two orthogonal evidences for a contig join,
cross-domain Field 7); two-modality fingerprint matching; the codebase's own
"splines match, **intensity verifies**" doctrine in `intensity_qc.py` — generalised
from per-interface rotation to per-join.

### Key assumption
The cut-face grayscale at the 5th–10th slice carries MT-distinguishing structure (the
W2b lumen/wall profile is the safest bet; W2a surround is dataset-dependent). For
cryo-ET this is the real risk — featureless tomograms make W2 mute, in which case the
system **falls back to W1-only and must abstain more** (it cannot manufacture the
second witness). That is honest, not a failure.

### Kill-test
On a held-out interface (e.g. Monopoles `sec01→02`), build the NCC/profile-step
distribution for **mutual-nearest geometric pairs** (proxy positives) vs **random
aligned-neighbour pairs** (proxy negatives). If `AUC(W2) ≤ ~0.65`, the image witness
carries no independent information on this data → disable W2, ship W1-only with a
**wider** abstain band, and *say so in the report*. If `AUC(W2) ≥ 0.75`, W2 is real
and the two-witness product is justified. **Second kill-test for independence
itself:** regress W2's residual on W1's residual across all candidate pairs; if the
correlation `|ρ| > 0.3`, the witnesses are *not* independent (likely the warp leaked
MT patches) and the additive log-odds over-counts — fix the sampling, don't ship.

---

## Idea 2 — The abstention calculus: group-when-aliased, with a permutation null

### Pitch
Formalise the brief's "honest abstention" as a **measurement**, not a policy choice.
When two candidate partners are both inside the jump-noise cloud, the *correct* output
is ABSTAIN (emit a clean break / mark the pair as a merge-ambiguous group), because
the data is information-theoretically incapable of distinguishing them. The decision
thresholds a single quantity — a **partner-identifiability margin** — and is
calibrated against a **permutation null** that says how often a join this good would
arise by pure chance at the local density.

### Mechanism

**The quantity to threshold — identifiability margin `Δ`.**
For endpoint A with candidate partners sorted by `score` (Idea 1):
```
    Δ(A) = score(B_top) − score(B_2)          # log-odds gap to the runner-up
```
Decision rule:
```
    if score(B_top) < τ:                       ABSTAIN  (no good partner — real break or miss)
    elif Δ(A) < μ:                             ABSTAIN-as-GROUP  (aliased neighbours)
    else:                                      COMMIT B_top
```
`τ` and `μ` are **two fixed log-odds knobs** (not per-dataset tuning) because the
log-odds is already normalised by the per-interface null (Idea 1's denominator). The
user gets one dial: *target precision* → sets `τ`; recall floats (the contrarian
agent's "abstention as the product" made concrete).

**The aliasing geometry that drives `μ`.** From the first-principles bound: two
parallel neighbours separated laterally by `d` are unresolvable when `d ≲ δJ`, the
jump uncertainty. `δJ` is **measured, not assumed**, per interface:
```
    δJ² = ε² + (g · sec²θ · δθ)²
```
- `ε` = post-alignment endpoint localisation noise = the **median ballistic residual
  of the high-confidence (mutual-NN, high-W2) joins** at this interface. Self-calibrating.
- `g` = slab thickness = the **median lateral jump of confident tilted-vs-vertical
  pairs** (vertical MTs jump ~0, tilted MTs jump `g·tanθ`; the slope recovers `g`).
- `δθ` = tangent-estimation scatter from the polyline fit.

When `d < δJ` for the top-2 partners, `Δ` is automatically small (their scores nearly
tie) → the rule fires ABSTAIN-as-GROUP **for free**, because the same `δJ` that makes
them physically unresolvable makes their log-odds indistinguishable. The geometry and
the decision rule are the *same* threshold seen twice.

**ABSTAIN-as-GROUP is not a dead end — it still feeds alignment.** When A's partner is
ambiguous between {B_top, B_2}, commit the **group centroid** as an alignment anchor
(alignment is tolerant to 1-to-many — first-principles §1) and emit a per-MT
"unresolved within group {B_top,B_2}" tag to the chain (a clean break with a reason),
**not** a coin-flip pick. This is the exact decoupling the contrarian agent demands:
the warp gets its anchor, the chain gets its honesty.

**The permutation / null model — "how good is this join, really?"**
A small `r` is only impressive if it beats chance at the *local density*. Build the
null by **label permutation within the gate**: for endpoint A, take the set of all
endpoints in section k+1 within the distance gate, and ask — if I assigned A a partner
*at random* from that set, what is the distribution of `r` (and of `score`)? Because
endpoint positions are fixed and only the *identity assignment* is permuted, this is a
true null for "joins explainable by density alone."

```
    p_chance(A) = fraction of random in-gate partners with score ≥ score(B_top)
    false-join rate floor ≈ (n_in_gate − 1) · P(random partner lands within δJ of p̂_B)
                          ≈ (n_in_gate − 1) · (π δJ²) · ρ_local⁻²      # expected aliasing count
```
A committed join must have `p_chance(A)` below a small α (e.g. 0.01) — i.e. the
observed join is **better than the best of `n_in_gate` random draws**. In a dense
bundle `n_in_gate` is large and `π δJ²·ρ⁻²` approaches 1 → the null says "you can't
beat chance here" → the rule abstains. In a sparse field `n_in_gate→1` → any decent
join trivially beats the null → commit. **This is the formal statement of why dense
bundles are hard and sparse fields are easy, and it falls out of one integral.**

### Inspiration
JPDA marginal association + abstention-on-tie (radar, Field 1); genome assembly
tangles left unresolved (Field 7); the first-principles `d < δJ` information bound,
turned into a runnable permutation test.

### Key assumption
`δJ` is estimable per-interface from the confident-pair statistics (it is — vertical
vs tilted jump slope recovers `g`; high-W2 residuals recover `ε`). If an interface has
**too few** confident pairs to bootstrap `δJ`, fall back to the global `δJ` and widen
the abstain band (more conservative, never less).

### Kill-test
Take a flagged ABSTAIN-as-GROUP set and **force** per-MT pairing inside it. Against any
held-out check (even the noisy GT, used only as a spot-check), forced precision inside
flagged groups should be **~50%** (coin-flip — confirming they were genuinely
unresolvable). If forced precision is high (say >75%), `δJ` is **over-estimated** and
the system is abstaining when it could commit — lower `δJ` until forced-precision-in-
groups drops to chance. This is a *self-calibrating* knob: the right `δJ` is the one at
which forced intra-group pairing is exactly a coin flip.

---

## Idea 3 — Stack self-consistency: triple-overlap gives a FREE correctness check

### Pitch
An MT crossing sections k, k+1, k+2 is constrained **twice** — join (k,k+1) and join
(k+1,k+2) — by two *independent* interface estimates. The chain through it must be
geometrically coherent across both. This redundancy is a correctness check that
pairwise matching **structurally cannot have**, and it costs almost nothing because the
joins are already computed. It is the GT-free analogue of bundle-adjustment loop
closure: the only way three pairwise estimates agree is if they're individually right.

### Mechanism

**The check — chain-curvature coherence across a sliding window.**
For a filament with sub-blocks in sections k, k+1, k+2, define the **transport
residual**: the ballistic prediction from section k, propagated *through* the section
k+1 sub-block, should land on the section k+2 entry within `δJ`. Equivalently, the
three centreline chords (k-block, (k+1)-block, k+2-block) must form a path with
turning angle within the MT curvature bound at *both* joints, AND the two joints'
turns must be **consistent in sign/magnitude with a single smooth fiber** rather than
two coincidences stacked.

Concretely, extending `split_chains_at_joints` (which already compares overall
direction across **one** joint) to a **window of 3**:
```
    triple_ok(k) =  turn(block_k, block_{k+1}) < θ_max
                AND turn(block_{k+1}, block_{k+2}) < θ_max
                AND  | curvature(k→k+1) − curvature(k+1→k+2) |  < κ_max
```
The third clause is the new information: a real MT has **slowly-varying curvature**
(persistence length ≫ section, first-principles §3), so the bend it makes at joint k
should resemble the bend at joint k+1. Two stacked wrong-joins produce **uncorrelated**
bends → the curvature-consistency clause catches them even when each *individual* joint
passes the pairwise angle gate. **This is the free correctness check pairwise matching
lacks:** a wrong join that fools one interface rarely fools two in a curvature-coherent
way.

**Cycle consistency (when the stack has any redundancy).** If alignment is also
computed for the **skip interface** (k → k+2 directly, image-only, cheap), then the
composition of joins must close:
```
    T(k→k+1) ∘ T(k+1→k+2)  ≈  T(k→k+2)        # pose cycle-closure residual
```
A large closure residual flags that one of the three estimates is wrong **without
knowing which join is the culprit** — exactly bundle-adjustment loop closure. For the
*correspondence*, the analogue is: A→B→C (forward chaining) must equal A→C (direct
match across the skip). Disagreement = at least one wrong join in the triple → demote
the whole triple to ABSTAIN at the weakest link (lowest Idea-1 score).

**Cost.** Joins (k,k+1) and (k+1,k+2) already exist. The triple check is O(chain
length) angle/curvature arithmetic — negligible. The skip-interface image alignment is
one extra `block_match` per window — cheap and only needed if you want the stronger
cycle test. **No new matching, no new tracing.**

### Inspiration
Bundle-adjustment loop closure; MHT's "confirm a track only after M frames"
(N-scan confirmation, Field 1); ice-core wiggle-matching where a third core
cross-validates a two-core tie (Field 4).

### Key assumption
MTs persist across **≥3** sections often enough that triple windows are common (true
for dense bundles — the hard case — and the regime where the check is most needed).
For MTs spanning only 2 sections the triple check is silent and we fall back to the
two-witness pairwise score; that's fine — short fibers are where redundancy genuinely
doesn't exist.

### Kill-test
Inject a **synthetic wrong-join** at interface k (swap A's true partner for its nearest
neighbour). The triple curvature-consistency clause should flag the k-window
(`|Δcurvature| > κ_max`) **even when the pairwise angle gate at k passes**. If it does
not — i.e. a single planted wrong-join survives the triple window as often as it
survives the pairwise gate — then triple-overlap adds no information on this data and
should be dropped. Conversely, measure: of joins that the pairwise gate accepts but the
triple check rejects, what fraction are *true* (false-rejection rate)? If high, `κ_max`
is too tight (real curving MTs being broken) — loosen it.

---

## Idea 4 — The GT-free evaluation metric (rank approaches without labels)

### Pitch
To choose between candidate pipelines with **no ground truth**, you need numbers that
are (a) computable from data alone, (b) monotone in "correctness", and (c) hard to game
by trivially abstaining or trivially committing. Four such numbers; the **first three
are the ones to show a skeptic.**

### The numbers

**M1 — Two-witness agreement rate (and its calibration curve).**
Among committed joins, the fraction where W1 (geometry) and W2 (image) *independently*
clear their thresholds. Plot **commit-rate vs. realised two-witness agreement** as `τ`
sweeps: a good method's curve stays high-agreement as it commits more; a bad method's
agreement collapses as it reaches for recall. This is GT-free because the two witnesses
**validate each other** — high agreement under *independent* evidence is precisely the
thing labels would otherwise tell you. **Headline number: agreement rate at the
operating `τ`.**

**M2 — Bimodality / separation of the ballistic-residual distribution.**
The first-principles kill-test, promoted to an evaluation metric. Histogram the
ballistic residual `r` over all candidate joins. A correct pipeline produces a
**bimodal** distribution — a tight ballistic mode (`r ≪ ρ`, true joins) and a broad
mode (forced/ambiguous). Quantify with the **dip statistic** or the ratio
`(σ_broad / σ_tight)`. A method whose `r` is unimodal-broad has no separating power; a
method with a sharp tight mode and a clean valley is resolving real continuations.
**Headline number: tight-mode fraction × mode separation.**

**M3 — Abstention-calibration curve (the honesty metric).**
The single number that catches a method **cheating by abstaining**. As you force the
ABSTAIN set to commit (lower `τ`), measure the **triple-overlap consistency rate**
(Idea 3) of the newly-committed joins. A *well-calibrated* abstainer abstained exactly
on the joins that would have failed triple-consistency: forcing them in drops
consistency steeply. A *lazy* abstainer abstained on good joins: forcing them in keeps
consistency high (it was over-conservative). The **slope of triple-consistency vs.
commit-rate at the operating point** is the calibration: steep = honest abstention,
flat = either over- or under-cautious. This directly answers the brief's "honest
abstention where genuinely ambiguous" without a single label.

**M4 — Chain-smoothness energy (alignment-side cross-check).**
Total curvature energy `Σ (turn angle)²` summed over all joints of all committed
chains, normalised by chain count. A whirlpool/foldover warp or a batch of wrong-joins
inflates this; a clean stitch minimises it. Cheap, global, and catches the
*alignment* failure (whirlpool) that M1–M3 (correspondence-focused) might miss. Use as
a **regression guard**, not a headline (it can be gamed by abstaining on all curvy MTs
— which is why M3 must be reported alongside it).

### The three numbers to convince a skeptic
1. **M1** — two-witness agreement at the operating point (are commits corroborated by
   independent evidence?).
2. **M2** — ballistic-residual bimodality (is the method *separating* true from false,
   or just thresholding noise?).
3. **M3** — abstention-calibration slope (is the abstention honest, or is it hiding
   failures / wasting recall?).

If all three move the right way going from the current Hungarian to a candidate
pipeline, the candidate is better — **provably, without labels**. The noisy GT is used
only as an optional fourth spot-check (precision/recall on the subset where GT is
trustworthy), never to drive the ranking.

### Inspiration
Self-supervised model selection via agreement of independent views (co-training); DTW
path-width as confidence (Field 4); the brief's own success criteria turned into
measurable, gameable-resistant quantities.

### Key assumption
The three metrics are not all gamed by the *same* trivial strategy. They aren't:
M1 punishes uncorroborated commits, M3 punishes lazy abstention, M2 punishes
noise-thresholding — committing-everything fails M1, abstaining-everything fails M3,
random-scoring fails M2. The triangle is closed.

### Kill-test
Construct three adversarial pipelines — (i) commit-all, (ii) abstain-all, (iii)
random-score — and confirm each is killed by at least one of M1/M2/M3. If any
adversary scores well on all three, the metric set is incomplete and needs a fourth
guard. (Predicted: commit-all dies on M1, abstain-all dies on M3, random dies on M2 —
if so, the triangle holds.)

---

## The whole thing in one paragraph

After image-led alignment, each candidate join gets a **per-join log-odds** =
`L1` (geometry: ballistic residual vs measured jump-noise `δJ`) + `L2` (image:
grayscale continuity along the predicted tangent, normalised by the per-interface
random-neighbour null). The two are **independent because they read disjoint
coordinates of the data** — L1 reads the traced centreline, L2 reads the voxel texture
the trace discarded, sampled *at the MT* where the warp (fit from MT-free regions) never
looked — so their failure modes (positional aliasing vs. textural dropout) are driven
by unrelated nuisances and their false-positives **multiply**. Commit only when the top
partner clears a log-odds floor `τ` **and** beats its runner-up by margin `μ`; when two
partners sit within the measured jump-noise `δJ` (so `μ` can't be met), **abstain-as-
group** — commit the centroid to alignment, emit an honest break to the chain — which
is the operational form of the information bound `d < δJ`, validated by a permutation
null that says dense bundles can't beat chance. Triple-overlap across sections k,k+1,k+2
adds a **free** correctness check (curvature must be coherent across two joints; pose
cycle must close), catching stacked wrong-joins one interface can't. And the whole thing
is **ranked without labels** by three numbers — two-witness agreement (M1), ballistic-
residual bimodality (M2), and abstention-calibration slope (M3) — a closed triangle no
trivial commit-all / abstain-all / random strategy can pass.

*Written 2026-06-11 by the Deep-Dive agent. Grounded against `matcher.py`,
`intensity_qc.py`, `image_warp.py`, `match.py`, `mt_endpoints.py`, `chain.py`,
`transform/scale.py`.*
