# Implementation Landscape — Existing Tools for MT Stitching Problems

> Agent: IMPLEMENTATION LANDSCAPE | Date: 2026-06-11
> Task: find software that solves the alignment + MT correspondence problem, or a same-shaped problem in another domain.

---

## TL;DR rating key

| Rating | Meaning |
|--------|---------|
| **USABLE-AS-IS** | drop-in, no adaptation needed |
| **ADAPTABLE** | right algorithm, needs API wrapping or signal substitution |
| **INSPIRATION** | right idea, wrong modality/structure — steal the mechanism |
| **DEAD-END** | wrong problem shape or fatal constraint |

---

## 1. The only prior art that does EXACTLY this problem

### SerialSectionAligner (ZIB / Lindow et al. 2021)

- **Pitch:** The only published tool that stitches filamentous structures across serial-section tomograms. Robert is a co-author. Solves alignment (CPD nonrigid warp on MT point cloud) + correspondence (MRF with belief propagation on endpoint factor graph).
- **Mechanism:** Coarse rigid alignment via DCG (distance compatibility graph, finds large point-set cliques), then fine elastic warp via CPD on MT endpoints, then endpoint matching via pairwise MRF / belief propagation (libDai). Interactive correction loop.
- **Key assumption:** Assumes correspondences should be closer after alignment; breaks for dense parallel arrays where every MT looks like every other.
- **Kill-test:** Paper explicitly states results "were not satisfactory for the microtubule arrays" of dense spindles — the exact regime pandorica targets. Alignment failure in regular parallel grids. No Python API; C++ Amira plugin; MIT-like license but requires Amira with XPand extension to compile.
- **Status:** INSPIRATION — the MRF/BP formulation for endpoint correspondence is the best published prior; the dense-MT failure is precisely the gap pandorica is trying to close. The explicit pairwise coherence penalty (neighbouring pairs should have similar displacement vectors) is worth stealing.
- **GitHub:** https://github.com/zibamira/SerialSectionAligner

### Automated MT Stitching — Kiewisz et al. 2014 / Weber et al. (PMC4249889)

- **Pitch:** Earliest dedicated automated approach; formulates endpoint matching as MRF + belief propagation on a factor graph with singleton distance/angle factors and pairwise coherence factors.
- **Mechanism:** DCG clique → rigid alignment (SVD) → CPD nonrigid elastic → MRF belief propagation; falls back to manual when BP oscillates.
- **Key assumption:** Corresponding endpoints are spatially closer; neighbouring matches have coherent displacement.
- **Kill-test:** Dense parallel arrays fail. No code release. libDai dependency (unmaintained C++).
- **Status:** INSPIRATION — the pairwise coherence factor (two adjacent matches should agree on displacement) is the right mechanism to prevent whirlpools; pandorica's current code has no pairwise inter-match constraint.

---

## 2. EM serial-section alignment tools (image-based)

### TrakEM2 (Fiji/ImageJ plugin; Cardona et al. 2012)

- **Pitch:** Full serial-section EM stitching pipeline: tile stitching, section-to-section alignment, non-linear elastic warp. Used for connectomics volumes.
- **Mechanism:** Image-based feature detection (SIFT), block-matching, RANSAC, TPS warp per section. Works on 2D section images.
- **Key assumption:** Images have rich texture (contrast features). Operates on rasterised images, not spline/point-cloud data.
- **Kill-test:** Completely image-based; spline/MT data is not an input. Can produce registered images that pandorica could then use as the image signal for alignment, but offers no MT correspondence. No Python API; Jython scriptable from Fiji.
- **Status:** DEAD-END for MT correspondence; ADAPTABLE as image-alignment pre-processor if pandorica wants image-driven coarse align.

### AlignTK / Alignment_Projects (Janelia, Karsh)

- **Pitch:** Petascale ssEM pipeline using local cross-correlation for image tile alignment and section registration.
- **Mechanism:** Block-matching on image tiles, deformable mesh alignment, distributed computing on HPC. Operates exclusively on images.
- **Kill-test:** Image-only; no spline data; C++ codebase; no Python API.
- **Status:** DEAD-END.

### SWiFT-IR / SwiftIR (Python image registration)

- **Pitch:** Python-callable image registration tool from the Janelia/Allen ecosystem; uses image-based alignment.
- **Mechanism:** Wavelets + image cross-correlation for 2D section alignment.
- **Kill-test:** Image only; MTs are not inputs.
- **Status:** DEAD-END.

### bUnwarpJ (Fiji; Arganda-Carreras et al.)

- **Pitch:** B-spline elastic image registration; bidirectional consistency constraint.
- **Mechanism:** Spline-parameterised warp minimising image similarity + bending energy; no point cloud input.
- **Kill-test:** Image only; Fiji plugin, not Python callable.
- **Status:** DEAD-END.

---

## 3. The best directly analogous tool: min-cost network-flow particle tracking

### TrackMate (Fiji; Tinevez et al. 2017)

- **Pitch:** Fiji plugin for particle tracking; LAP (Linear Assignment Problem) two-step linker — frame-to-frame links then gap closing. Handles appearance/disappearance (abstention by leaving a particle unlinked if the cost exceeds a threshold).
- **Mechanism:** Frame-to-frame: LAP with Jaqaman cost matrix (distance + feature terms + dummy rows/columns for birth/death). Gap-closing: second LAP to link track ends to track starts across gaps.
- **Key assumption:** Costs can be decomposed: the birth/death dummy row trick sets a maximum cost at which a detection is preferentially left unlinked rather than forced into a bad match. This is precisely the "honest abstention" mechanism.
- **Kill-test for MT use:** Designed for blobs in microscopy images, not oriented line endpoints; the cost function would need to incorporate tangent direction. Fiji plugin, not Python; LAP cost matrix is 2D Euclidean by default.
- **Steal this:** The **dummy row/column trick** in the LAP cost matrix makes abstention the mathematically correct choice for low-confidence matches — a match is made only when its cost is below the cost of leaving both endpoints unmatched. This is far cleaner than a post-hoc distance gate. Pandorica currently uses `np.inf` gating before the Hungarian; the TrackMate formulation is the proper way to do it.
- **Status:** INSPIRATION — steal the birth/death dummy-augmented LAP for honest abstention.

### trackpy (Python; Allan et al.)

- **Pitch:** Pure Python/NumPy implementation of Crocker-Grier nearest-neighbour particle tracking. `trackpy.link` solves the assignment problem frame-by-frame using a KD-tree.
- **Mechanism:** Nearest-neighbour search within a search radius; LAP extension available via `trackpy.link(..., adaptive_stop=...)`. Scipy `linear_sum_assignment` under the hood.
- **Kill-test:** No orientation/tangent signal; purely spatial; no GPU.
- **Status:** DEAD-END for MT correspondence; already essentially what pandorica does for the spatial gate.

### btrack (Python/C++; Lowe et al. 2022)

- **Pitch:** Bayesian multi-object tracker. Builds tracklets from confident detections, then links tracklets using a hypothesis-based second pass. Motion model is a Kalman filter (constant velocity or constant acceleration).
- **Mechanism:** Kalman predict → Bayesian posterior assignment; tracklet hypothesis testing to handle splits/merges/gaps.
- **Key assumption:** Objects move with approximate constant velocity (or acceleration) between frames; the Kalman filter encodes this prior.
- **Kill-test for MT:** MTs don't "move" between sections — they end and continue. The velocity prior is wrong (MT orientation ≠ velocity). The Kalman model would need to be replaced with a geometry + density prior. No GPU. PyPI installable (`pip install btrack`).
- **Steal this:** The **two-pass architecture** (form confident tracklets first, then link tracklets in a second pass) breaks the chicken-and-egg cleanly: short confident tracklets (well-isolated MT stubs) anchor the alignment; the second pass chains the rest.
- **Status:** INSPIRATION — two-pass tracklet hypothesis model.

### Ultrack (royerlab; Nature Methods 2025)

- **Pitch:** Tracks cells under segmentation uncertainty by considering multiple segmentation hypotheses jointly and selecting the temporally consistent set via an integer program (min-cost flow on a directed graph).
- **Mechanism:** Generates overlapping candidate segments per time point; builds a graph over candidates; solves min-cost flow to select the globally consistent trajectory set. Scales to terabyte datasets; Fiji + napari plugins; PyPI installable (`pip install ultrack`).
- **Key assumption:** Multiple candidate segmentations exist per cell; the right one is selected by temporal consistency. For MTs: multiple endpoint hypotheses per stub could be generated.
- **Kill-test for MT:** Designed for isotropic cells in 2D/3D timelapses; no line/orientation model; the graph construction assumes blob-like objects. The cross-section correspondence problem maps to the time-lapse problem (section = time frame), but orientation/tangent cannot enter the cost without modifying the graph builder.
- **Steal this:** The **min-cost flow formulation with multiple candidate hypotheses** is exactly the right architecture for MT correspondence with abstention: generate K candidate matches per endpoint (K nearest viable partners), set up a flow graph where each endpoint is a node with supply/demand 1, and edges carry the match cost. Unmatched flow routes through a dummy sink at a fixed penalty (= the abstention cost). This is GT-free and naturally handles many-to-one rejection.
- **GitHub:** https://github.com/royerlab/ultrack
- **Status:** INSPIRATION — steal the multi-hypothesis min-cost-flow architecture.

---

## 4. Point-cloud registration (for the alignment sub-problem)

### Open3D (Intel; MIT license)

- **Pitch:** Full point cloud processing library: ICP (point-to-point, point-to-plane, colored), FPFH + RANSAC global registration, TSDF, tensor API.
- **Mechanism:** FPFH features describe local geometry around each point; RANSAC finds correspondences; ICP refines. For oriented data: point-to-plane ICP uses normals (= MT tangents in pandorica's case).
- **Key assumption:** Point set has surface normals (tangents). MT endpoints have tangent vectors — these can be used as normals for point-to-plane ICP.
- **Kill-test:** Dense parallel MT arrays are degenerate for FPFH (all points look identical locally); FPFH + RANSAC global registration will fail. Point-to-plane ICP for fine alignment is still viable after a coarse seed. GPU support: tensor ICP has CUDA; no MPS.
- **Steal this:** Point-to-plane ICP where "normals" = MT endpoint tangents is a better coarse-alignment objective than distance-only ICP — it adds the tangent direction signal.
- **Status:** ADAPTABLE for fine alignment (point-to-plane ICP with tangent-as-normal); DEAD-END for global/coarse registration.

### probreg (neka-nat; MIT license) / pycpd (siavashk; MIT license)

- **Pitch:** Python implementations of Coherent Point Drift — probabilistic point set registration that models one set as Gaussian mixture centroids and aligns the other as observations via EM. Supports rigid, affine, and nonrigid (TPS kernel) variants.
- **Mechanism:** EM loop: E-step assigns soft correspondences (posterior probabilities), M-step updates the transformation to maximise expected likelihood. Nonrigid CPD uses a low-rank RBF approximation. Scales to ~1000 points reasonably.
- **Key assumption:** Points are exchangeable; the GMM treats all pairings as soft hypotheses. No orientation signal in standard CPD.
- **Kill-test:** Dense parallel MT grid is degenerate — every MT endpoint looks the same to the GMM. Orientation-extended CPD exists (Fisher-Mises distribution on tangents, used in SerialSectionAligner) but is not in probreg/pycpd. No GPU.
- **Steal this:** The **soft correspondence + re-estimation loop** decouples alignment from hard correspondence — run "soft CPD" to get alignment, then solve hard correspondence on the aligned result. Avoids the chicken-and-egg.
- **Status:** ADAPTABLE as the alignment sub-problem solver if extended with a tangent term (orientation-CPD as in Lindow 2021).

### TEASER++ (MIT-SPARK; MIT license)

- **Pitch:** Certifiably-robust global point cloud registration; solves rotation, translation, scale separately using graduated non-convexity; >99% outlier robust.
- **Mechanism:** Decoupled rotation/translation/scale estimation using truncated least-squares; certifies global optimality. Works on correspondences or point sets.
- **Kill-test for MT:** Designed for unstructured 3D rigid scenes; the MT spindle has near-degenerate structure (parallel MTs → rotation is ambiguous). Requires C++ build with Python bindings; no PyPI wheel; no MPS.
- **Status:** DEAD-END — degenerate geometry makes it unreliable on dense parallel arrays.

### FilterReg (neka-nat, via probreg; BSD license)

- **Pitch:** Fast probabilistic registration using Student-t mixture model; more robust to outliers than CPD; solves for rigid or deformable transforms.
- **Mechanism:** EM with Student-t likelihood (heavy-tailed → outlier robust); supports point-to-point and point-to-surface (using normals).
- **Kill-test:** Same degeneracy problem as CPD on dense parallel MT fields; no GPU.
- **Status:** DEAD-END.

---

## 5. Graph and assignment solvers

### scipy.optimize.linear_sum_assignment (SciPy; BSD)

- **Pitch:** Exact O(n³) Hungarian algorithm; already used in pandorica. Returns a one-to-one assignment.
- **Kill-test:** Pure one-to-one; no native abstention; no pairwise inter-match constraints. For n=200 endpoints this is ~8ms — not the bottleneck.
- **Status:** USABLE-AS-IS for the assignment step; the limitation is the problem formulation around it, not the solver.

### lapjv / lapjvsp (Jonker-Volgenant; MIT)

- **Pitch:** ~10x faster than Hungarian for dense cost matrices (O(n²·log n) average). Python package `lapjv` available on PyPI.
- **Kill-test:** Same problem formulation limitations as Hungarian; no abstention built in.
- **Status:** ADAPTABLE — drop-in replacement for `linear_sum_assignment` when n > 500.

### pygmtools (Thinklab-SJTU; MIT license; PyPI)

- **Pitch:** Python Graph Matching Toolkit; solves quadratic assignment problems (QAP) with pairwise edge costs in addition to node costs. Backends: NumPy, PyTorch (CUDA), JAX, Paddle.
- **Mechanism:** Classic solvers: Hungarian (linear), spectral graph matching (quadratic, relaxed), integer projected fixed point (IPFP, quadratic, discrete). Neural: NGM, GCAN, etc. (require GT for training).
- **Key assumption for QAP:** Two graphs are given (MT endpoints + their neighbourhood structure on each side); node cost = endpoint distance/direction; edge cost = displacement coherence between pairs. The QAP objective naturally encodes the pairwise coherence constraint that the MRF/BP approach used in Lindow 2021 — it's the same mathematical object.
- **Kill-test:** QAP is NP-hard; spectral relaxation is approximate; IPFP is discrete but slow for n>100. The pairwise "edge cost" requires defining a neighbourhood graph over MT endpoints — straightforward (kNN), but adds a design decision.
- **Steal this:** QAP with pairwise coherence cost is the correct formulation to replace the current Hungarian + post-hoc uncross. The pairwise edge cost = `||shift_ij - shift_kl||` (displacement coherence) directly prevents whirlpools because a locally inconsistent match raises the total QAP objective.
- **GitHub:** https://github.com/Thinklab-SJTU/pygmtools
- **Status:** ADAPTABLE — use `pygmtools.ipfp_solver` or spectral solver with a displacement-coherence edge cost; PyTorch backend runs on CUDA (not MPS natively, but PyTorch MPS fallback may work).

### POT — Python Optimal Transport (PythonOT; MIT license; PyPI)

- **Pitch:** Full OT library: earth-mover distance, Sinkhorn (entropic regularisation), unbalanced OT (KL marginal relaxation), partial OT. PyTorch, JAX, NumPy backends; Sinkhorn on GPU via `ot.sinkhorn_torch`.
- **Mechanism:** Sinkhorn gives a soft transport plan (doubly stochastic matrix) — not one-to-one, but allows natural abstention via the marginal relaxation. Unbalanced OT handles cases where not all mass needs to be transported.
- **Key assumption:** Unbalanced Sinkhorn with KL marginal relaxation allows endpoints to be "left unmatched" if no good partner exists — this is the abstention mechanism.
- **Kill-test:** Soft plan needs to be rounded to one-to-one for chain building; rounding introduces error. GPU via PyTorch backend on CUDA; MPS not tested.
- **Steal this:** Unbalanced Sinkhorn as a **soft pre-alignment** step: compute a fuzzy transport plan from MT endpoints section A → section B, use the weighted centroid shifts to estimate alignment, then solve hard 1:1 assignment on the aligned points. This breaks the chicken-and-egg by separating soft alignment (Sinkhorn) from hard correspondence (Hungarian on aligned data).
- **Status:** ADAPTABLE — use `ot.unbalanced.sinkhorn_unbalanced` for the soft alignment stage.

---

## 6. Learned feature matchers (EM image signal)

### LightGlue (CVG ETH; Apache-2.0; PyPI-installable)

- **Pitch:** Learned keypoint matcher using a transformer GNN; takes two sets of keypoints + descriptors, returns confident one-to-one correspondences with explicit confidence scores per match. Supports multiple feature backends (SuperPoint, DISK, ALIKED, SIFT).
- **Mechanism:** Attentional graph neural network; alternates self-attention (within each image's keypoints) and cross-attention (between images); outputs a partial assignment (abstains on uncertain matches).
- **Key assumption for MT:** It is designed for visual feature descriptors from images; it could operate on EM image crops around each MT endpoint as descriptors — each endpoint gets a small image patch, LightGlue matches patches across sections.
- **Kill-test:** LightGlue was trained on photographic images (MegaDepth/ScanNet); EM grayscale is out-of-distribution. More importantly: two adjacent MTs have nearly identical local image content (both show a ~25 nm ring cross-section in the same ice matrix). The discriminability requirement — that each endpoint's local patch is distinguishable from its neighbours — fails for dense parallel arrays.
- **Steal this:** The **adaptive early-exit** mechanism (LightGlue stops attention iterations when confidence is sufficient) and the **explicit confidence-based abstention** are architecture patterns worth applying to a domain-specific matcher. The partial assignment output (some matches, some abstentions) is exactly what pandorica needs.
- **Status:** DEAD-END for direct use on MT endpoint patches; INSPIRATION for the confidence-gated partial assignment architecture.

### SuperGlue (Magic Leap; academic non-commercial only)

- **Pitch:** Predecessor to LightGlue; same GNN architecture but less efficient; restrictive license.
- **Kill-test:** Non-commercial license; same modality-mismatch problem as LightGlue.
- **Status:** DEAD-END.

### LoFTR (ZJU; Apache-2.0)

- **Pitch:** Dense feature matcher using a transformer on coarse feature maps; matches without keypoint detection.
- **Kill-test:** Designed for full-image matching; does not naturally give per-endpoint confidences. Dense MT image patches are near-identical; LoFTR cannot disambiguate structurally identical neighbours.
- **Status:** DEAD-END.

---

## 7. Domain-specific MT tracking tools

### micron / mtrack (nilsec, Funke lab; MIT license)

- **Pitch:** Python + ILP tool for tracking microtubules **within** a 3D EM volume (not across sections). Formulates MT tracing as a constrained ILP: candidate voxel graph → select paths via integer program. Gurobi required (free academic license).
- **Mechanism:** Candidate edge graph over voxels; ILP enforces flow conservation (each MT is a path); biological priors penalise sharp bends.
- **Kill-test:** Designed for within-volume reconstruction, not cross-section correspondence. Gurobi dependency (commercial solver, though academic license free). Not applicable as-is.
- **Steal this:** The **ILP flow formulation** for chain building: model each section's MT stubs as nodes, cross-section matches as edges with costs, and solve a min-cost flow that respects "at most one continuation per stub" — an alternative to the current union-find that can encode the abstention cost natively.
- **Status:** INSPIRATION — ILP/flow formulation for the chain-building step.

### TSOAX (Penn State; open source C++)

- **Pitch:** Tracks dynamic biopolymer networks (actin, MTs in light microscopy time-lapse) across frames using a k-partite graph matching approach.
- **Mechanism:** Extracts filament curves per frame (SOAX snake algorithm); constructs k-partite directed graph; finds minimum-cost path cover; enforces local network-topology consistency at junctions.
- **Key assumption:** Filaments are extracted by the snake algorithm (requires good image quality + manually tuned parameters); tracking across frames uses exponential decay with spatial distance threshold.
- **Kill-test:** C++ only, no Python API; requires re-implementation of the curve extraction; designed for fluorescence imaging, not cryo-ET. Dense parallel MT arrays have nearly identical local curve neighbourhoods → k-partite assignment is underdetermined.
- **Steal this:** The **local network-topology consistency constraint** at junctions (neighbouring filament trajectories should be locally consistent) is the k-partite analogue of the displacement-coherence pairwise term — and maps directly to the QAP edge cost or the MRF pairwise factor.
- **Status:** INSPIRATION — local topology-consistency constraint.

### TAMiT (yeast MT tracking; PMC10296093)

- **Pitch:** Semi-automated MT tracking in yeast fluorescence images across time frames; uses nearest-neighbour + greedy assignment.
- **Kill-test:** Fluorescence; not cryo-ET; greedy NN is weaker than Hungarian; no abstention mechanism.
- **Status:** DEAD-END.

---

## 8. MEDPC

Searched extensively; no tool named "MEDPC" exists in the cryo-ET or EM stitching literature. "MED-PC" is a behavioural neuroscience software package (operant conditioning). The term may be a mis-remembering; no result found that is relevant.

---

## 9. Other tools checked (brief verdicts)

| Tool | Status | One-line reason |
|------|--------|----------------|
| Fiji "Register Virtual Stack Slices" | DEAD-END | Image-only; Fiji plugin |
| webKnossos skeleton tools (knossos-utils) | DEAD-END | Manual annotation only; no automated matching |
| NeuroMorph (Blender) | DEAD-END | Morphology visualisation; no correspondence algorithm |
| AMfinder | DEAD-END | Arbuscular mycorrhiza fungal segmentation; wrong biology |
| Cytoseg | DEAD-END | Cell membrane segmentation; no filament tracking |
| Frangi vesselness | DEAD-END | Segmentation filter; no cross-section correspondence |
| AllenAI VAST | DEAD-END | Connectomics annotation; no automated MT matching |

---

## Key ideas (in the structured format)

### Idea A — Dummy-augmented LAP for honest abstention (from TrackMate)

- **Pitch:** Replace the current hard `np.inf` gate + post-hoc rejection with a LAP that has explicit dummy rows/columns whose cost equals the "abstain" penalty. A match is accepted only if it beats leaving both endpoints unmatched.
- **Mechanism:** Augment the cost matrix with `n_ref` dummy columns (for unmatched moving endpoints) and `n_mov` dummy rows (for unmatched reference endpoints) at a fixed cost `τ`. Solve the square augmented LAP; pairs assigned to dummies are abstentions.
- **Inspiration:** TrackMate (Jaqaman 2008 Nature Methods); also standard in particle tracking literature.
- **Key assumption:** `τ` can be set from the ρ-scaled distance gate without GT supervision — e.g. `τ = w_dist * 1.0 + w_dir * 1.0 = 1.0` (full gate = abstain).
- **Kill-test:** τ still needs to be set; too small → over-abstain (misses); too large → too many wrong joins. But τ is interpretable and can be set from physical priors.

### Idea B — Pairwise-coherence QAP to prevent whirlpools (from pygmtools / MRF literature)

- **Pitch:** Replace the current Hungarian (no pairwise term) with a Quadratic Assignment Problem that has an edge cost penalising displacement incoherence between adjacent match pairs. This directly prevents whirlpool/foldover by making a locally inconsistent set of matches expensive.
- **Mechanism:** Build a kNN graph over reference endpoints. Edge cost between pairs (i,j) and (k,l) is `||shift(i,j) - shift(k,l)||` where `shift = mov_pos - ref_pos`. Use `pygmtools.ipfp_solver` (PyTorch, CUDA-capable) or a spectral relaxation.
- **Inspiration:** pygmtools QAP; MRF pairwise factor in Lindow 2021 / Weber 2014.
- **Key assumption:** Displacement field is locally smooth — valid for rigid + slow warp. Breaks only at section boundaries with catastrophic deformation.
- **Kill-test:** QAP is NP-hard; IPFP is approximate. For n=200 endpoints, IPFP typically converges in <1s. The main risk is local optima in the IPFP solver.

### Idea C — Two-pass architecture: coarse high-confidence tracklets first (from btrack)

- **Pitch:** Run matching only on isolated/unambiguous endpoints first (confidence gate on the local density / tangent uniqueness), use those to anchor the alignment, then run a second matching pass on the remaining endpoints in the aligned coordinate frame.
- **Mechanism:** Pass 1: match only endpoints with no nearby competitor within 2ρ (low-ambiguity, high-confidence pairs). Pass 2: fit alignment from pass-1 pairs; match remaining endpoints in aligned space with looser cost.
- **Inspiration:** btrack two-pass tracklet architecture; also resembles the RANSAC paradigm.
- **Key assumption:** Enough unambiguous MTs exist per interface to anchor the alignment (~5-10% suffices for a rigid fit). Fails if all MTs are dense and ambiguous (uniform spindle cross-section with no isolated outliers).
- **Kill-test:** The densest spindle regions (kinetochore fibres in bundles) may have no isolated MTs → pass 1 yields 0 confident pairs → alignment fails.

### Idea D — Soft OT alignment + hard correspondence on aligned data (from POT / SerialSectionAligner CPD)

- **Pitch:** Run unbalanced Sinkhorn optimal transport as the alignment step (soft, no hard assignment) to estimate the displacement field; solve hard 1:1 correspondence (Hungarian + Idea A abstention) only on the already-aligned endpoints.
- **Mechanism:** Compute soft transport plan `T` between endpoint point clouds (MT endpoints weighted by local density); use `T` to compute expected displacement field; warp mov endpoints; run hard assignment on warped positions.
- **Inspiration:** POT `ot.unbalanced.sinkhorn_unbalanced`; CPD's EM loop (E-step = soft assignment, M-step = transform).
- **Key assumption:** The soft transport plan can estimate alignment without hard correspondences. Works when MTs are exchangeable enough that population-level shifts are estimable.
- **Kill-test:** Dense uniform spindle → soft OT assigns mass uniformly → displacement field is flat and correct on average but reveals no fine structure. Fine alignment still needs the hard correspondence pass.

### Idea E — Min-cost network flow for chain building (from Ultrack / micron)

- **Pitch:** Replace the current union-find (which just propagates the Hungarian's matches) with a min-cost flow over the multi-section graph, where abstention has a fixed cost and each MT stub is a source/sink with unit supply. The flow solver maximises global consistency across all interfaces jointly.
- **Mechanism:** Build a directed graph: source → each MT stub node (section k) → candidate match nodes → each MT stub node (section k+1) → sink. Edge costs = match costs (distance + direction + pairwise coherence). Abstention edge from stub to sink has cost τ. Solve min-cost flow (e.g. `networkx.min_cost_flow` or OR-Tools).
- **Inspiration:** Ultrack min-cost flow; Jaqaman LAP gap-closing; nilsec/micron ILP.
- **Key assumption:** The pairwise interface costs are correct enough that joint flow across all interfaces improves on greedy pairwise assignments. GT-free (costs are geometric).
- **Kill-test:** Min-cost flow on a multi-section stack (10+ sections, 200 MTs each) may be slow without a dedicated solver. `networkx` is too slow for >1000 nodes; needs `OR-Tools` or `scipy` flow.

---

## Summary table

| Tool / Idea | Rating | Why |
|-------------|--------|-----|
| SerialSectionAligner (ZIB/Lindow) | INSPIRATION | Only prior art; MRF+BP endpoint matching; fails dense arrays |
| Weber 2014 MRF/BP paper | INSPIRATION | Factor graph with pairwise displacement coherence is the right math |
| TrackMate dummy-LAP | INSPIRATION ★ | **Best single steal**: abstention built into the assignment matrix |
| Ultrack min-cost flow | INSPIRATION | Multi-hypothesis flow with abstention cost; correct architecture |
| pygmtools QAP | ADAPTABLE | Pairwise coherence as QAP edge cost; CUDA backend |
| POT unbalanced Sinkhorn | ADAPTABLE | Soft alignment stage to break chicken-and-egg |
| probreg/CPD (orientation-extended) | ADAPTABLE | Orientation-weighted CPD for fine alignment |
| Open3D point-to-plane ICP | ADAPTABLE | Tangent-as-normal ICP for fine rigid alignment |
| btrack two-pass | INSPIRATION | Confident-tracklet-first architecture |
| lapjv | USABLE-AS-IS | Drop-in faster LAP for large n |
| LightGlue | DEAD-END | Discriminability fails for near-identical MT cross-sections |
| micron/nilsec ILP | INSPIRATION | ILP flow formulation for chains |
| AlignTK / TrakEM2 / bUnwarpJ | DEAD-END | Image-only |
| TEASER++ | DEAD-END | Degenerate geometry on parallel arrays |
| TSOAX/SOAX | INSPIRATION | Local topology-consistency constraint at junctions |

---

## Single best "steal this" candidate

**TrackMate's dummy-augmented LAP** (Jaqaman et al. 2008, Nature Methods doi:10.1038/nmeth.1237).

The current pandorica code gates with `np.inf` in the cost matrix and then applies post-hoc outlier rejection. TrackMate's formulation flips this: augment the cost matrix with dummy rows/columns at cost `τ` (the abstain penalty), solve the now-square LAP, and endpoints assigned to dummies are abstentions by the same optimal assignment that determines the real matches. This means:
- Abstention is jointly optimal with the matches (no post-hoc inconsistency).
- `τ` is interpretable as "I'd rather leave this unmatched than pay this cost."
- It naturally handles the miss (false negative) / wrong-join (false positive) tradeoff via a single tunable parameter.
- Implementation is 10 lines around the existing `linear_sum_assignment` call.

This is the lowest-risk, highest-leverage change to the current matcher — it doesn't require changing the cost function or the warp architecture, just the assignment problem structure.
