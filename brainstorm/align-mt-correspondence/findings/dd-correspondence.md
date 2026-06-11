# Deep-Dive: Correspondence Formulation for Dense, Parallel MT Bundles (GT-Free)

**Agent**: DEEP-DIVE  
**Date**: 2026-06-11  
**Assumption baked in**: Image-driven alignment is already solved — both faces are in a common
frame to within << MT spacing. All five formulations are evaluated *solely* on the
correspondence sub-problem: decide per-boundary MT its continuation, or abstain.

---

## Framing: what "density failure" actually is

Before comparing formulations, we must be precise about *why* dense parallel arrays break
matching. From the first-principles analysis (first-principles.md §3):

Two parallel MTs separated laterally by `d` each jump by `J = g·tan(θ)` across the gap.
The jump uncertainty (position noise on where a single MT lands) is `δJ`. When `d < δJ`,
the displacement clouds of the two candidates overlap — their two hypotheses (A1→B1,A2→B2
vs. A1→B2,A2→B1) are *information-theoretically equiprobable* given only local endpoint
data. No assignment algorithm, however sophisticated, can beat this floor with local
information alone. The failure of Weber 2014 on dense arrays is not a bug in belief
propagation — it is hitting this floor.

This means the correct question for each formulation is:

1. **Does it exploit non-local information** to push the floor down? (Context from the rest
   of the bundle, multi-section consistency, image patch across the gap.)
2. **Does it abstain correctly** when it hits the floor, rather than guessing and poisoning
   the chain?
3. **Does it scale to N~hundreds on CUDA/MPS** without a Gurobi license or C++ build?

---

## Formulation 1: Pairwise-consistency MRF / Belief Propagation (Weber 2014)

### Why it fails on density — the mechanism

Weber's MRF has two kinds of factors:
- **Singleton**: exponential cost on (position distance, axis-projected distance, angle
  deviation). This is essentially Hungarian cost dressed up as a Gibbs potential.
- **Pair factor**: penalty when two adjacent assignments disagree on displacement —
  i.e. `||shift(i→k) - shift(j→l)||` is large for neighbouring edge pairs (i,j) and (k,l).

The pair factor is doing *displacement smoothness*: nearby MTs should jump similarly. In
a smooth warp field, this is correct and powerful. It prevents individual wrong-joins from
bending the field.

**But in a dense parallel array, this factor fires in BOTH directions simultaneously.**
When MTs are tightly packed (d << ρ), the two assignment hypotheses (correct vs. swapped)
produce *nearly identical displacement fields* for the pair factor to evaluate. The
coherent bundle jumps by J; the swapped assignment also produces a jump of approximately J
for each pair (because the partner is at J+d ≈ J). The pair factor therefore receives
nearly equal messages from both hypotheses, and BP oscillates — which is exactly what
Weber reports ("non-convergence resolved by manual intervention at critical nodes").

**The formal failure**: the pair potential is measuring displacement *smoothness*, which is
preserved whether you swap two parallel neighbours or not. What it cannot measure is
*identity* — which specific stub at position x+J is the continuation of the stub at x. For
that you need a signal that breaks the degeneracy of the two hypotheses.

### What would concretely fix it

- **Better singleton potential**: replace endpoint distance with *ballistic-residual
  distance* — cost on `||B − (A + J·t̂)||` instead of `||B − A||`. This shifts the gate
  to the *predicted* landing, and the two parallel neighbours are now at (±d/2) from
  the prediction rather than at (d) and (0) from the raw endpoint. It does not
  break degeneracy when d < δJ, but it substantially reduces wrong-joins for d ≈ δJ.

- **Bundle-topology pair potential**: instead of displacement coherence, penalise when the
  *relative order* of MTs changes — if i is to the left of j in section A, k should be
  to the left of l in section B. This is topological order preservation (a
  *permutation-sign* constraint, not a smooth-displacement constraint). It can reverse
  a crossing that displacement coherence cannot detect.

- **Global BP vs local BP**: loopy BP on a dense proximity graph is notorious for
  oscillation when many cycles exist (exactly the dense bundle graph). Mean-field
  variational inference converges monotonically but is a weaker approximation. Neither
  fixes the fundamental degeneracy; they only affect convergence of the same bad objective.

### Mechanism against neighbour confusion

Displacement coherence — but THIS IS EXACTLY WHAT FAILS when neighbours are equidistant
from the prediction. The pair factor needs a topological or ordering term, not just a
smoothness term.

### GT-free?

Weber's parameters were fit to expert-labelled examples. The pair potential weight λ can be
set self-consistently via empirical Bayes on high-confidence pairs (first-principles §5c),
but this is fiddly. **Marginal GT-free**.

### CUDA/MPS feasible?

BP itself is parallel over edges; implementable in PyTorch. No dedicated GPU library for
general factor graphs. libDai (Weber/Lindow's implementation) is unmaintained C++.
**Reimplementation needed; feasible but medium effort**.

### Failure mode on density

When d < δJ: oscillates or converges to the wrong assignment with equal probability. No
abstention mechanism — forces one of the two swapped assignments. **Fails silently.**

### Summary rating

**3/5** — the pairwise coherence idea is correct and necessary. The implementation with
displacement-smoothness pair factors cannot break the degeneracy of parallel equidistant
pairs. Requires a topological ordering term AND a ballistic-residual singleton to improve
materially. Still fails at the d < δJ floor.

---

## Formulation 2: Quadratic Assignment Problem (QAP) with displacement-coherence edge cost

### The QAP objective for MT correspondence

QAP takes two graphs G_A and G_B (nodes = MT endpoints, edges = proximity) and seeks a
node permutation π that minimises:

```
E(π) = Σ_i  c_node(i, π(i))                       [singleton: endpoint cost]
      + Σ_{i,j} c_edge(i→j, π(i)→π(j))             [pair: edge cost]
```

The pair edge cost is typically:

```
c_edge(i→j, k→l) = ||shift(i,k) - shift(j,l)||    where shift(a,b) = pos_B(b) - pos_A(a)
```

This is the displacement-coherence term — exactly what the MRF pair factor was doing, but
now formulated as a quadratic program rather than a message-passing inference.

### Does the QAP edge term resolve parallel ambiguity that LAP cannot?

**Partially, but not fundamentally.** The edge cost is the same displacement-coherence
signal as Weber's pair factor; the mathematical object (QAP objective) is the same as the
MRF MAP objective at zero temperature. Solving the QAP is therefore equivalent to running
BP at T→0 (argmax vs. soft marginal).

The concrete advantages over Weber's BP:

1. **Approximate discrete solvers (IPFP) are guaranteed to not increase the objective** at
   each step — they converge, unlike loopy BP which can oscillate. On dense arrays this
   matters: you get *a* deterministic answer, not oscillation.
2. **Spectral relaxation** (Leordeanu-Hebert) gives a soft assignment from the leading
   eigenvector of the affinity matrix, which can be thresholded for abstention.
3. **The formulation naturally encodes abstention** via dummy nodes in the QAP graph:
   augment G_B with dummy nodes at cost `τ`; the QAP can assign A-endpoints to dummies.

### The ordering constraint as an edge cost

An important QAP-specific option: set the edge cost to:

```
c_edge(i→j, k→l) = indicator(sign(x_A(j) - x_A(i)) ≠ sign(x_B(l) - x_B(k)))
```

i.e. penalise relative-order reversals (crossings). This is NOT computable in a linear
LAP, but IS natural as a QAP edge. On a parallel bundle, this ordering constraint breaks
the d < δJ degeneracy: the two swapped assignments have OPPOSITE relative orders for every
nearest-neighbour pair, so the ordering penalty strongly disfavours one of them.

**This is the key mechanism QAP can exploit that MRF/LAP cannot: a relative-order /
crossing-penalty edge cost breaks degeneracy without GT.**

### Cost and scaling at N~hundreds on CUDA

- IPFP (integer projected fixed point): O(N²) per iteration, typically <50 iterations for
  N=200. With PyTorch on CUDA: milliseconds for N=200. For N=1000: ~seconds. **Feasible**.
- Spectral relaxation: eigendecomposition of an N²×N² affinity matrix (after vectorising
  pairs, that is ~N²=40000 for N=200). This is the bottleneck; on CUDA with batched
  eigensolver: **feasible for N≤200, marginal for N~1000**.
- pygmtools backends: PyTorch (CUDA) native; MPS via PyTorch's MPS backend (untested but
  should work given it's all tensor ops).

### Failure mode

Crossing-penalty alone does not help if the bundle has many near-parallel, near-equal-spacing
MTs with no consistent lateral order (e.g. a hexagonal lattice rotated by 60°). Still fails
at the `d < δJ` floor because the crossing penalty itself becomes noisy when d ≈ δJ.
IPFP finds local optima; no global optimality guarantee.

### Summary rating

**4/5** — genuinely superior to Weber for dense arrays because the crossing-penalty edge
cost breaks the pair degeneracy that displacement-smoothness cannot. The abstention via dummy
nodes is clean. CUDA-feasible. Main limitation: still approximates a hard problem, and the
ordering constraint only works when the bundle has a consistent lateral arrangement (which
real spindle MT bundles largely do, but not always).

---

## Formulation 3: Gromov-Wasserstein Optimal Transport

### What GW preserves and why it might matter

Gromov-Wasserstein (GW) finds a transport plan π that minimises the distortion of pairwise
inter-point distances:

```
GW(A, B) = min_π  Σ_{i,j,k,l} |d_A(i,j) - d_B(k,l)|² · π(i,k) · π(j,l)
```

where d_A(i,j) is the distance between MT endpoints i and j *within section A*, and
d_B(k,l) within section B. GW does NOT use the absolute positions (no transport across the
gap directly) — it matches on the *structure* of the inter-MT distance matrix.

### Is structure-preservation the right inductive bias for a coherent bundle?

**Yes, but with a crucial caveat.** For a rigidly-translated bundle (the whole spindle
shifts by a vector), the inter-MT distance matrix d_A ≈ d_B exactly, and GW would find a
perfect matching that is identity-consistent. This is the "right" inductive bias for a
global rigid jump.

**The caveat**: GW is *invariant to permutation* — it does not know which MT is "first."
This means in a perfectly regular hexagonal lattice, GW finds the transport plan that
minimises distance distortion, which could assign A1 to B7 if the lattice has a symmetry
that makes them equivalent. GW finds the nearest *automorphism*, not the *identity map*.

For real MT bundles, which are *not* perfectly regular (local density fluctuations,
occasional isolated MTs), this automorphism problem is less severe — unique MTs (different
local density, edge position) anchor the matching. But for the inner fibres of a dense
kinetochore bundle, GW suffers exactly the same degeneracy as everything else.

### Partial/unbalanced GW for abstention

Partial GW (arXiv 2410.16718 is on point) allows only a fraction of the total mass to be
transported — unmatched endpoints leave their mass in the marginal. This is the right
mechanism for abstention: an MT at a section boundary with no plausible partner routes its
mass to the unmatched slack. The abstention threshold is the mass fraction parameter `m`
(or equivalently the KL divergence weight in unbalanced GW).

This is cleaner than dummy nodes in the QAP / LAP because the mass budget relaxation is
continuous and can be cross-validated on the total number of MTs per interface (physical
prior: section B should have ≈ same MT count as section A, modulo terminations).

### CUDA/MPS feasible?

POT's `ot.gromov.gromov_wasserstein` uses a mirror-descent / projected gradient loop.
The cost tensor is N×N×N×N (for N=100, that's 10^8 elements — 400 MB at float32). For
N=200 this is 1.6 GB — tight but feasible on a modern GPU. For N=1000 this breaks.

POT has PyTorch GPU backend (`ot.backend.TorchBackend`); MPS compatible via PyTorch MPS.
Partial GW and unbalanced GW are also implemented in POT.

**Scaling verdict: feasible for N≤200 on CUDA (tight memory); impractical for N≥500.**

### Failure mode

Lattice automorphisms in perfectly regular bundles. The N^4 cost tensor memory wall at
N≥500. Slow convergence (many Frank-Wolfe iterations) when the distance matrices are
similar (degenerate gradient). No ordering constraint analogue — it is purely structure
(distance-matrix) matching.

### Summary rating

**3/5** — interesting inductive bias (structure preservation rather than absolute position
matching) and the cleanest abstention mechanism via mass budget. But hits the automorphism
degeneracy in perfectly regular grids, and memory-scales badly beyond N~200. Not the right
primary formulation; potentially useful as a fast *pre-screening* that identifies plausible
group-level correspondences before per-MT identity assignment.

---

## Formulation 4: Min-cost Network Flow / Global Multi-Section

### Ultrack and micron/mtrack: how directly reusable?

**Ultrack** (royerlab, Nature Methods 2025) solves multi-frame tracking via a min-cost flow
on a directed graph where each candidate segment in each frame is a node, and edges
connect candidates across frames. The flow conservation ensures consistent long-range
trajectories. The key features for MT correspondence:

- **Multi-section joint optimisation**: rather than solving section k↔k+1 independently,
  the flow over the graph (k, k+1, k+2, ...) enforces that each MT track is internally
  consistent. A wrong join at k↔k+1 that contradicts evidence at k+1↔k+2 can be
  rejected globally.
- **Multiple candidates per stub**: generate K nearest partners for each endpoint (not
  committed to one), let the flow select the globally consistent set.
- **Abstention**: route unmatched flow through a dummy sink at cost τ — same as
  dummy-augmented LAP but across the full stack.

**Why it beats pairwise approaches on density**: the multi-section constraint provides
the non-local information needed to break local degeneracy. If MTs A1 and A2 are
locally indistinguishable at the k↔k+1 interface, but their trajectories over sections
k-2 to k+3 diverge (one curves left, one right), the global flow can resolve the
k↔k+1 ambiguity without GT. This is the only formulation that uses the *temporal trajectory
context* across multiple sections.

**Ultrack reusability**: Ultrack's graph builder and ILP solver are designed for blob-like
cells, not oriented filaments. The cost function (node affinity) would need to be replaced
with the MT endpoint cost (ballistic residual + direction). The graph builder would need
to generate K nearest candidates in the ballistic-predicted metric, not Euclidean. This
is a non-trivial but tractable adaptation — the graph structure and flow solver can be
reused, the cost function must be reimplemented. `pip install ultrack`; pure Python
orchestration around a C++ min-cost flow (OR-Tools or similar). No CUDA dependency in the
solver itself, but cost matrix computation (ballistic residual for N×N×K candidates per
interface) is CUDA-friendly.

**micron/mtrack (Funke lab)**: designed for within-volume MT tracing (ILP over voxel
candidates). The ILP formulation is the right shape (flow conservation = each MT is a
path) but the node graph is over voxels, not endpoint abstractions. **Not directly
reusable** for cross-section correspondence — would need a full reimplementation of the
graph. The Gurobi dependency (even academic) is a real constraint for deployment.

### Does solving the whole stack at once beat pairwise?

**Yes, provably, under one condition**: the long-range trajectory context must carry
information beyond the local interface. For MTs in a cryo-ET spindle this is almost always
true — MTs that curve over 10 sections, MTs that terminate at precise kinetochore
attachment points, MTs that belong to specific half-spindle populations. The joint flow can
exploit all of this; pairwise k↔k+1 cannot.

**The failure mode**: if ALL sections in a stack have identically ambiguous interfaces (a
perfectly uniform bundle throughout), multi-section consistency helps only at the edges of
the ambiguous region. Inside the region it still reduces to a pairwise coin-flip per
interface. The multi-section gain is real but not magical.

### CUDA/MPS feasibility

Min-cost flow is a combinatorial problem — no GPU-accelerated exact solver exists that is
generally available. The typical path is OR-Tools (Google, open source, CPU) or
NetworkX (pure Python, slow). For a stack of 10 sections × 200 MTs × K=5 candidates each,
the flow graph has ~10,000 nodes and ~50,000 edges — OR-Tools solves this in <1 second on
CPU. **This is not a CUDA problem; it is a graph problem.** CUDA is useful only for
computing the N×N×K cost matrices per interface (batched ballistic-residual distances).

### Summary rating

**4.5/5 for multi-section datasets** — the only formulation that genuinely uses multi-section
trajectory context to resolve per-interface ambiguity. Not a magic bullet for perfectly
uniform bundles, but the best available for real, imperfectly-regular MT bundles. The cost
is adaptation effort on Ultrack's graph builder. **2/5 for two-section-only problems**
where there is no trajectory context to exploit — reduces to a dummy-augmented LAP with
a min-cost flow frontend.

---

## Formulation 5: Dummy-Augmented LAP / Unbalanced Sinkhorn

### The mechanism — and what it is actually doing

**Dummy-augmented LAP** (Jaqaman 2008 TrackMate): augment the cost matrix with
`n_B` dummy columns (at cost τ) and `n_A` dummy rows; solve the enlarged square LAP.
Endpoints assigned to dummies are abstentions. This is mathematically equivalent to
adding an abstain option with a fixed utility threshold: "I'd rather leave this MT
unmatched than pay >τ to match it."

**Unbalanced Sinkhorn OT** (POT `ot.unbalanced.sinkhorn_unbalanced`): replace the
doubly-marginal constraint (every endpoint must be assigned somewhere) with KL-penalised
marginals that allow mass to accumulate in the unmatched slack. The soft transport plan
that results is an N×N matrix of soft assignments; round it to a hard 1:1 matching to
extract the correspondence. The KL weight plays the role of τ.

### Does it suffice alone to beat Weber?

**No.** The dummy-augmented LAP does not add any pairwise information — it is still solving
a linear (node-only) cost. What it adds is *honest abstention*: it will correctly refuse to
commit on ambiguous pairs when τ is calibrated to the ballistic-residual cost of a wrong
join. But it will still *produce wrong joins* for ambiguous pairs when both candidates have
costs below τ (i.e. when the two parallel neighbours are within the ballistic uncertainty).

The dummy-LAP is a necessary component of any formulation that claims honest abstention.
It is not sufficient alone: it needs a richer cost function or a pairwise constraint to
*separate* right from wrong joins before deciding whether to abstain.

**The right use**: dummy-augmented LAP as the *final rounding layer* on top of a richer soft
assignment (Sinkhorn, QAP spectral relaxation, or GW plan). The upstream formulation
provides a soft scoring over all candidate pairs; the dummy-LAP rounds it to a hard 1:1
with principled abstentions.

### Unbalanced Sinkhorn as a GT-free alignment estimator

A separate, very useful role: run unbalanced Sinkhorn on the full endpoint sets to estimate
the *smooth displacement field* without committing to hard correspondences. The soft
transport plan T (N_A × N_B matrix) implicitly encodes a weighted nearest-neighbour field.
Use the barycentre shift to estimate a coarse alignment warp. This is exactly CPD's E-step,
but formulated as OT instead of GMM. It inherits CPD's degeneracy on parallel arrays but
can be regularised with an entropic term (set the Sinkhorn regularisation ε to be larger
than MT spacing → mass diffuses across neighbours → gives a smooth field even in ambiguous
regions). **Useful for alignment confirmation, not for individual MT identity**.

### Summary rating

**2/5 as a standalone formulation. 5/5 as a necessary layer in any stack.**
Every formulation above ultimately needs dummy-augmented LAP or partial OT for the final
hard rounding. None of them work without a principled abstention mechanism at the end.

---

## Head-to-head comparison on density

| Formulation | Breaks d<δJ degeneracy? | Mechanism | Abstention | CUDA/MPS | Maturity/code |
|---|---|---|---|---|---|
| MRF/BP (Weber) | No — displacement coherence preserves both hypotheses | Displacement smoothness pair factor | None (oscillates) | Reimplementation needed | libDai (dead), reimplementation medium-effort |
| QAP with ordering penalty | **Yes** — crossing penalty disfavours one hypothesis | Relative-order / crossing-free edge cost | Dummy nodes (natural) | pygmtools PyTorch CUDA | Good (pygmtools pip-installable) |
| Gromov-Wasserstein | Partial — lattice automorphisms remain | Distance-matrix structure preservation | Mass budget (clean) | POT CUDA, memory-limited N≤200 | Good (POT pip-installable), memory wall |
| Min-cost flow / multi-section | **Yes, with trajectory context** | Multi-section consistency breaks local degeneracy | Dummy sink (natural) | CPU solver (ms), CUDA for cost matrix | Ultrack adaptable, medium-effort |
| Dummy-LAP / Sinkhorn | No — not a pairwise formulation | N/A | Yes — this IS the abstention mechanism | scipy/POT, feasible | Very high — already in pandorica family |

---

## The recommended formulation: QAP-with-ordering + multi-section flow + dummy-LAP stack

### Why this beats Weber on density — the precise mechanism

Weber's MRF fails on dense parallel bundles because:

1. The pair potential (displacement smoothness) is *symmetric under swap* of two parallel
   equidistant neighbours — both hypotheses score identically.
2. BP oscillates when the factor graph has dense cycles, which is unavoidable on a
   close-packed bundle.
3. There is no abstention; when BP oscillates, the MAP decode is arbitrary.

The proposed stack fixes each failure:

**Step 1 — Ballistic singleton cost** (replaces raw endpoint distance):
```
c_node(i, k) = ||(pos_B(k) - (pos_A(i) + J·t̂(i)))|| / σ_J
```
Recentre the gate on the predicted landing. Reduces neighbour confusion for d ≈ δJ.
Under-used in Weber; near-zero implementation cost in pandorica.

**Step 2 — QAP with crossing-free / relative-order edge cost** (replaces BP):
```
c_edge(i→j, k→l) = α·||shift(i,k) - shift(j,l)||       [displacement coherence]
                  + β·1[lateral_order(i,j) ≠ lateral_order(k,l)]  [crossing penalty]
```
The crossing penalty is the key new term. In a parallel bundle, the two swap hypotheses
have OPPOSITE crossing patterns for all nearest-neighbour pairs simultaneously — so the
crossing penalty strongly penalises the wrong hypothesis even when displacement coherence
cannot distinguish them. Solved by IPFP (pygmtools, PyTorch CUDA): deterministic,
convergent, seconds for N≤300.

**Step 3 — Multi-section min-cost flow** (wraps the per-interface QAP):
For each interface, generate K=5 candidate matches per endpoint (the top-5 by QAP
node cost); build a directed graph across all sections; run min-cost flow (OR-Tools).
Flow conservation resolves per-interface ambiguities that remain after QAP using
trajectory context from adjacent interfaces.

**Step 4 — Dummy-augmented LAP round** (the final abstention gate):
QAP + flow produce a soft ranking per endpoint. Round to hard 1:1 via dummy-augmented LAP
at threshold τ calibrated to the ballistic residual of a "wrong join at d = δJ" — the
physical abstention level.

### Idea: QAP-flow stack (full proposal)

- **Pitch**: Replace Weber's MRF/BP + Hungarian with a QAP (crossing-free edge cost) per
  interface, wrapped in a multi-section min-cost flow, with dummy-LAP for final hard
  rounding. GT-free; CUDA for the cost computation; CPU for the combinatorial solve.

- **Mechanism**: (1) Ballistic singleton cost recentres the distance gate on the predicted
  landing point, eliminating false nearest-neighbour matches. (2) QAP crossing-penalty
  edge cost breaks the pair-swap degeneracy that displacement-smoothness cannot. (3)
  Multi-section flow provides trajectory context to resolve ambiguities that remain local.
  (4) Dummy-LAP rounds to hard 1:1 with principled abstentions at the physical resolution
  limit.

- **Inspiration**: pygmtools QAP (Thinklab-SJTU); Ultrack multi-hypothesis flow (royerlab);
  TrackMate dummy-augmented LAP (Jaqaman 2008); TSOAX local-topology consistency;
  first-principles §3 information bound.

- **Key assumption**: The bundle has a *consistent lateral ordering* of MTs across sections
  (i.e. nearby MTs in A remain nearby in B, and their relative left-right order is
  preserved). This is true for any section gap small enough that MTs do not cross each
  other in the unimaged slab — physically guaranteed for section thickness < MT-MT
  spacing / tan(θ_max), which holds for all practical cryo-ET conditions.

- **Kill-test**: On the sec01→02 Monopoles hard interface, the crossing-penalty edge cost
  should reduce wrong-join rate vs. displacement-smoothness-only QAP. If both produce
  the same wrong-join rate, the bundle lacks consistent lateral ordering (or the ordering
  is already preserved by displacement coherence alone — in which case QAP = MRF in this
  regime and the baseline already works).

---

## Implementation priority and effort

| Step | Code path | Effort | Dependency |
|---|---|---|---|
| Ballistic singleton cost | Replace `dist` in `matcher.py` cost function | Low (hours) | None |
| Dummy-augmented LAP | Augment cost matrix in `matcher.py` | Low (hours) | scipy (already present) |
| QAP crossing-penalty | `pygmtools.ipfp_solver` wrapper | Medium (days) | `pip install pygmtools` |
| Multi-section flow | Adapt Ultrack graph builder + OR-Tools | High (weeks) | `pip install ultrack ortools` |

**For the next sprint, in order of leverage/effort:**

1. **Ballistic singleton cost** — cheapest gain; attacks the core under-used signal.
2. **Dummy-augmented LAP** — principled abstention; critical for honest misses.
3. **QAP crossing-penalty** — the key mechanism that beats Weber on density.
4. **Multi-section flow** — the correct architecture for long stacks; defer until 1-3 are validated.

GW is *not* recommended as a primary formulation: the automorphism degeneracy and memory
wall make it worse than QAP for this problem. It could be used as a fast group-level
pre-screening to cluster MTs into bundles before applying the per-MT QAP within each
bundle.

---

## What GW could still contribute (secondary role)

Run GW at the bundle level: cluster A-endpoints into spatial groups (Voronoi or DBScan),
cluster B-endpoints similarly, and use GW to establish group-level correspondences (which
A-bundle maps to which B-bundle). Then run QAP within each matched bundle pair. This two-
level decomposition reduces the N in the QAP from (total MTs) to (MTs per bundle), making
IPFP cheaper and the crossing-penalty more local. The GW group-matching is memory-feasible
because the number of groups is ~10-30, not 200-1000.

---

## References

All citations are to prior agent findings; no new papers added. Key papers for this analysis:

- Weber B. et al. (2014) PMC4249889 — MRF/BP prior art and failure analysis.
- Sarlin P.E. et al. (2020) SuperGlue — GNN + Sinkhorn architecture for abstention.
- pygmtools (Thinklab-SJTU) — QAP solvers with PyTorch CUDA backend.
- POT Python Optimal Transport — Sinkhorn, GW, partial GW implementations.
- Ultrack (royerlab, Nature Methods 2025) — multi-hypothesis min-cost flow.
- Jaqaman K. et al. (2008) Nature Methods — dummy-augmented LAP for abstention.
- TSOAX — local network-topology consistency constraint at junctions.
