# Cross-Domain Analogies — MT correspondence + tomogram alignment

Fields examined: multi-target radar/sonar tracking, cell/particle tracking, fingerprint minutiae matching,
optical flow on repetitive textures, ice/sediment core cross-dating, particle-physics track reconstruction,
cartographic conflation, genome contig joining, star-catalog matching.

---

## Field 1 — Multi-target radar/sonar tracking (MHT / JPDA)

**The shape match.** A sensor sweep sees N blips. Between sweeps, objects move unpredictably; some
disappear, some appear. The system must re-identify blips across the gap while simultaneously estimating
target kinematics — exactly the lateral-jump + identity problem.

**Mechanisms.**

### MHT — Multiple Hypothesis Tracking (Reid 1979 / Blackman 2004)

- **Pitch**: Instead of committing to one global assignment, enumerate a tree of consistent assignment
  hypotheses. Each leaf is a complete (all-tracks, all-detections) assignment. Prune by joint likelihood;
  keep the top K trees. Commit only after M frames of confirmatory evidence.
- **Mechanism**: The joint likelihood of a hypothesis = product of per-track motion likelihoods × missed-
  detection penalties × false-alarm priors. Gating (Mahalanobis ellipsoid) prunes the tree before it
  explodes. N-scan pruning (dropping decisions older than N frames) keeps memory linear.
- **Inspiration**: Defer the final bijection. Carry ambiguous hypotheses across the gap and let the *next*
  section confirm or kill them.
- **Key assumption**: A kinematic model (here: MT tangent direction + expected lateral jump distribution)
  that makes hypotheses distinguishable by probability.
- **Kill-test**: If >50% of MT pairs share near-identical joint likelihoods (true for perfectly parallel,
  same-density bundles) pruning stalls. Need some texture/grayscale side-channel to break ties.

### JPDA — Joint Probabilistic Data Association

- **Pitch**: Instead of hard assignment, compute for every (track, detection) pair a *marginal association
  probability*, then update each track as a *weighted centroid* of all detections.
- **Mechanism**: Computes exact marginal probabilities under the permutation polytope — equivalent to
  computing the permanent of a likelihood matrix (approximated by Murty's algorithm or belief
  propagation). Each track is updated by a soft, weighted blend.
- **Inspiration**: Feed a soft-assignment matrix into the alignment solver instead of a binary Hungarian
  result. The alignment sees a weighted point cloud, not a set of hard-matched pairs.
- **Key assumption**: Detections are conditionally independent given the global state (false for our
  whirlpool case, where a bad match deforms the grid for everyone — but a two-pass variant handles it).
- **Kill-test**: When two MTs are truly indistinguishable (same tangent, same density neighbourhood), JPDA
  gives each 50/50 → their track updates average to a phantom midpoint. Need abstention logic on top.

**What transfers to MT correspondence.**
The MHT hypothesis-tree idea directly addresses the chicken-and-egg: generate K candidate alignment
transforms + K candidate assignment bijections jointly, score them by a combined cost (alignment residual
× correspondence likelihood), and prune. The top surviving pair is the committed answer. This avoids
fixing alignment before correspondence or vice versa.

The JPDA soft-assignment idea transfers more simply: replace the hard Hungarian with a soft-assignment
matrix (approximated permanent via belief propagation or Sinkhorn), use it as a *weighted* set of
correspondences to fit the alignment warp, then re-sharpen. This is one EM-style iteration, and it is
already used in robust ICP (see below).

---

## Field 2 — Particle-physics track reconstruction (Hough + Kalman through detector layers)

**The shape match.** A collider event fires thousands of charged particles through onion-ring detector
layers. Each layer records *hits* (position only). The system must reconstruct tracks — continuous
trajectories — across gaps with no identity label, with many near-parallel, near-identical tracks in the
dense inner layers. No ground truth per event.

**Mechanisms.**

### Hough Transform track seeding + Kalman extension

- **Pitch**: In track parameter space (curvature, angle, impact parameter), each hit votes for all
  tracks it could belong to. Dense clusters in parameter space = candidate tracks. Then a Kalman filter
  extends each candidate hit-by-hit, accepting/rejecting each layer's hit by Mahalanobis gate.
- **Mechanism**: The Hough accumulator converts an underdetermined combinatorial problem (which hit goes
  to which track?) into a density-estimation problem in a 3–5D parameter space. Near-parallel tracks
  remain separate because they occupy *distinct cells* in parameter space even when they nearly overlap
  in position space.
- **Inspiration**: MTs can be parameterised similarly: (endpoint position, tangent angle, curvature).
  Build a Hough-like accumulator over MT-pair hypotheses across the gap. Dense clusters = likely
  continuations. Near-parallel neighbours occupy adjacent but distinct cells → separability in parameter
  space even when spatially confusable.
- **Key assumption**: Tracks are smooth (low curvature per layer). MT tangents change slowly → same
  assumption holds.
- **Kill-test**: Two perfectly parallel MTs at the same spacing have identical Hough votes. Need a
  distinguishing signal (local density gradient, grayscale) to break the cell tie.

### Combinatorial Kalman filter / graph network (TrackML challenge solution, e.g. Exa.TrkX)

- **Pitch**: Build a bipartite graph (hits in layer L, hits in layer L+1), score edges by a learned or
  heuristic compatibility score, then find a consistent set of edge paths (no node used twice, maximum
  total score) by belief propagation or graph-neural-network message passing.
- **Mechanism**: Acyclic-path constraint = no node visited twice = bijection constraint. Belief
  propagation iteratively marginalises over competing paths. The graph can be *weighted* (not binary) so
  ambiguous edges get low weight rather than a hard keep/drop decision — equivalent to soft assignment.
- **Inspiration**: Model the MT-continuation problem as a bipartite graph (section A endpoints, section B
  endpoints), add edges scored by (Euclidean distance + tangent dot product + grayscale consistency).
  Run BP or min-cost-flow to find the bijection. Min-cost-flow gives exact global optimum in polynomial
  time and naturally handles abstention (unmatched nodes) via a dummy sink with finite cost.
- **Key assumption**: The graph is sparse enough after Mahalanobis gating to run in real time. For
  ~hundreds of MTs per section this is trivially true.
- **Kill-test**: BP / min-cost-flow does not solve the chicken-and-egg by itself — it needs alignment to
  pre-gate edges. Must be wrapped in an EM/RANSAC outer loop.

**What transfers.**
The Hough-in-parameter-space idea is the most direct mechanism for the parallel-neighbour ambiguity:
represent MT-pair hypotheses as points in (Δx, Δy, Δθ) space, find the dense cluster (= the dominant
alignment transform), and seed correspondences from that cluster before any global optimisation. This is
essentially a domain-specific RANSAC but more principled for line-like objects.

---

## Field 3 — Fingerprint minutiae matching (global transform + dense near-identical features)

**The shape match.** A fingerprint has ~60–100 minutiae (ridge endings, bifurcations). Two impressions
of the same finger are related by an unknown similarity transform (translation + rotation + small elastic
distortion), but minutiae are locally near-identical: ridge angle is nearly the same for nearby minutiae,
and dense regions contain many nearly-identical local descriptors. No ground truth.

**Mechanisms.**

### Generalised Hough + consistency graph (Jiang & Yau 2000, MCC descriptor)

- **Pitch**: For every pair (query minutia q, template minutia t), hypothesise the global similarity
  transform T(q→t). Collect all hypotheses into a 4D accumulator (Δx, Δy, θ, scale). The peak is the
  correct transform. Then, conditioned on the peak transform, solve correspondences greedily.
- **Mechanism**: Even if individual minutiae are locally ambiguous (same ridge angle, similar local
  density), their *global transform votes* cluster only if they are correct. Wrong pairs scatter their
  votes. The accumulator is essentially a RANSAC with dense voting rather than random sampling.
- **Inspiration**: Do exactly this for MTs. For every pair (MT endpoint in section A, MT endpoint in
  section B), compute the implied rigid transform. Accumulate votes in (Δx, Δy, Δθ, Δs) space. The peak
  is the dominant alignment. This is robust to a majority of outlier pairs (wrong neighbours) — the
  correct pairs vote consistently, the wrong ones scatter. Then solve correspondence given the peak.
- **Key assumption**: A majority of MT-pair hypotheses vote correctly (i.e., >50% of MT pairs are
  distinguishable enough to cast a consistent vote). If the network is truly uniform, the accumulator
  will be flat — but real EM data has density gradients and grayscale texture that break perfect symmetry.
- **Kill-test**: If the bundle is perfectly uniform and all transforms are degenerate (translation only,
  all MTs shift by the same vector), then every pair votes for the same transform and the peak gives no
  information about which pairs are correct. But in that case, correspondence is also trivial (nearest
  neighbour in aligned space), so it does not matter.

### FingerCode / orientation-field global descriptor

- **Pitch**: Before matching minutiae, compute a global orientation-field descriptor (the ridge flow
  pattern). Align the orientation fields first (coarse registration), then refine with minutiae.
- **Mechanism**: The orientation field is a dense signal (not sparse points) that encodes the global
  topology of the print. Matching it is more robust than matching sparse points in a clutter field.
- **Inspiration**: The MT density field (a 2D image of MT endpoint positions, smeared with a Gaussian)
  is a dense signal. Aligning the two sections' density fields (cross-correlation or phase correlation)
  gives a coarse (Δx, Δy, Δθ) seed *before* any individual MT assignment. This breaks the chicken-and-egg
  without touching the identity problem at all.
- **Key assumption**: The density field has enough non-uniform structure (gradients, holes, clusters) to
  make cross-correlation well-posed. For uniformly packed bundles this may fail — but then a global
  translation is the correct answer, and cross-correlation still finds it.
- **Kill-test**: A perfectly hexagonally-packed bundle has a periodic density field; cross-correlation
  will have multiple equal peaks (aliasing). Mitigation: use the *non-periodic* boundary of the bundle
  (the edge) as a landmark.

**What transfers.**
The generalised Hough transform over pairwise transform hypotheses is the single most directly
transferable mechanism. It separates alignment estimation from correspondence, is intrinsically robust to
outliers, requires no ground truth, and handles the parallel-neighbour ambiguity because correct pairs
vote coherently while confused neighbours scatter.

---

## Field 4 — Ice-core / sediment-core cross-dating (sequence alignment with missing interval)

**The shape match.** Two cores from nearby sites record the same climate signal (annual layer thickness,
isotope ratio, dust peaks) but may have slightly different depths, a missing interval (erosion,
disturbance), and local stretching. The task: align the two sequences, estimate the gap size, and
establish layer-to-layer correspondence. No ground truth; robustness to outliers (volcanic tephra,
bioturbation) is critical.

**Mechanisms.**

### Dynamic Time Warping (DTW) with gap penalty

- **Pitch**: DTW finds the monotone alignment path through the cost matrix (similarity between sequence
  elements at every pair of positions) that minimises total cost. A gap penalty controls how much
  insertions (missing layers) are penalised. The result is a correspondence plus a warp function.
- **Mechanism**: The cost matrix is filled bottom-up; the path is backtracked. Affine gap penalties
  (open + extend) model the physical structure of a missing slab (one contiguous erasure, not random
  drops). Confidence is read from the width of the path: a narrow path = confident alignment, wide path
  = ambiguous region.
- **Inspiration**: Model the MT-endpoint sequences (sorted by, e.g., angle around the bundle centre or
  by spatial position) as 1D or 2D sequences. Run DTW with affine gap penalty on descriptor vectors
  (tangent angle, local density, grayscale patch). The aligned path gives a soft correspondence + warp
  jointly. The path width gives a per-MT confidence for abstention.
- **Key assumption**: MTs have a well-defined ordering (e.g. angular position around a bundle axis or
  a 2D grid ordering) that is preserved across sections. If the bundle has a stable topology this holds;
  if it is a loose, tangled mesh it does not.
- **Kill-test**: DTW is O(N²) in sequence length and requires a 1D ordering. For a 2D cloud of MT
  endpoints with no canonical ordering, DTW generalises to 2D but loses its efficient DP guarantee.

### Wiggle-matching / template RANSAC (Baillie & Pilcher, dendrochronology)

- **Pitch**: Slide a template sequence over the target sequence; at each lag, compute a correlation score
  (or χ²). The peak lag is the offset. Uncertainty = width of the correlation peak. Outlier layers are
  detected by high per-layer residuals.
- **Mechanism**: The template is the reference dataset; the target is the unknown sample. Because the
  signal has correlated structure (long-range climate patterns), the correlation peak is sharp even with
  a moderate fraction of outlier layers. Confidence is the ratio of peak height to second-highest peak
  (a signal-to-noise for uniqueness).
- **Inspiration**: If we can extract a 1D summary of each section's MT bundle (e.g. the histogram of
  tangent angles, or the inter-MT spacing spectrum), sliding-window cross-correlation gives a
  translation/rotation seed robust to individual MT errors. This is a coarser version of the
  orientation-field idea above but works even on non-dense bundles.
- **Key assumption**: The 1D summary is informative and stable — e.g. tangent-angle histogram is
  meaningful for a well-organised bundle (cilia, centrioles) but may be flat for an isotropic network.
- **Kill-test**: Isotropic MT networks have flat histograms → cross-correlation is flat → no seed.

**What transfers.**
DTW with affine gap penalty transfers the concept of *sequence correspondence with an explicit missing-
interval model*. The gap penalty models the knife-cut slab directly. The path width gives a per-pair
confidence score for abstention — something the current Hungarian gives no access to.

---

## Field 5 — Star-catalog matching / astrometry (global transform from many nearly-identical points)

**The shape match.** A telescope image contains thousands of stars. A reference catalog contains millions.
The image-to-catalog transform is unknown (translation, rotation, scale, possibly distortion). Stars
within a field are nearly identical (point sources; only brightness and colour distinguish them). The
task: find the transform and establish star-by-star correspondence. No ground truth per image.

**Mechanisms.**

### Triangle hashing / geometric hashing (Groth 1986, astrometry.net)

- **Pitch**: For every triple of stars in the image, compute scale- and rotation-invariant triangle
  descriptors (angle ratios, side ratios). Hash them into a lookup table built from the catalog. A
  matching triple gives a full similarity transform hypothesis. Vote across all triples; the winner is
  the correct transform.
- **Mechanism**: The hash table is pre-built from the catalog. Query triples are looked up in O(1) per
  triple. Because the descriptor is invariant, no assumption about the transform is needed. After the
  transform is found, full correspondence is done by nearest-neighbour in aligned space.
- **Inspiration**: Build MT-triple descriptors: for every triple of MT endpoints, compute invariant
  descriptors (relative angles, spacing ratios, local density context). Hash or vote across all triples
  in section A vs section B. The winning transform is the alignment. This is RANSAC-with-better-sampling:
  instead of random 3-point samples, the hash lookup finds consistent triples immediately.
- **Key assumption**: At least one non-degenerate triple (non-collinear, non-uniform) exists in each
  section. For a bundle with any non-trivial structure this holds. For a perfectly hexagonal lattice,
  all triples are similar → hash collision → ambiguity. Mitigation: use 4- or 5-point hashes (quads).
- **Kill-test**: If the bundle is a perfect hexagonal lattice with identical spacing, all quad hashes
  are the same → the hash table gives no discrimination. But this is exactly the case where any alignment
  consistent with the lattice is correct, so ambiguity does not matter operationally.

### astrometry.net quad-hash + index (Lang et al. 2010)

- **Pitch**: Extend triangle hashing to 4-star quads. Encode each quad as a 4D code in [0,1]^4 (two
  interior-point positions relative to the bounding pair). Build a K-d tree index over all catalog quads.
  A single image quad lookup gives a transform candidate in milliseconds.
- **Mechanism**: 4D code makes collisions rare even in dense fields. The index is O(log N) lookup. A
  verify step checks that the full set of image stars aligns to catalog stars under the candidate
  transform (RANSAC-style verification with a loose inlier threshold).
- **Inspiration**: Build an offline index of MT-quad descriptors computed from the training stack (or
  bootstrapped from the first section pair). At run time, query the new section's quads against the
  index. This bootstraps alignment from the stack's own geometry without ground truth.
- **Key assumption**: The dataset is large enough that a useful quad index can be built from the stack
  itself (not just a single interface). For a 10-section stack with ~200 MTs each, this is marginal;
  for a 100-section stack it becomes viable.
- **Kill-test**: Single-interface use (no stack) provides no index. Must fall back to pairwise voting.

**What transfers.**
Triangle/quad hashing is a direct drop-in for the alignment-seed step. It requires no initialisation,
is intrinsically robust to misses and outliers (votes, not fits), and is well-tested on the exact
problem of finding a rigid transform among near-identical points in a cluttered field.

---

## Field 6 — Cartographic conflation / road-network matching

**The shape match.** Two map datasets (OpenStreetMap vs a government cadastral layer, or old map vs new)
represent the same road network but with geometric offset, different precision, missing streets, and
topological inconsistencies. The task: align the two networks and establish road-segment-to-road-segment
correspondence. Near-parallel roads in a grid city are the hard case.

**Mechanisms.**

### Topological graph matching + iterative closest trajectory (Savino et al., FMM)

- **Pitch**: Road segments are matched not just by proximity but by *network topology*: degree of nodes,
  connected component structure, turn sequences. A segment is more likely to match a counterpart if its
  network neighbourhood (the graph around it) also matches.
- **Mechanism**: Compute a graph similarity score that combines geometric distance (Fréchet or Hausdorff
  between polylines) with structural similarity (degree sequence, turn-angle sequence, sub-graph
  isomorphism). Run an alternating optimisation: fix correspondence → update alignment → fix alignment →
  update correspondence.
- **Inspiration**: Each MT has a *network context*: how many MTs are nearby, what their tangent angles
  are, whether the local bundle is straight or bent. Use the local bundle topology (not just the
  individual MT endpoint) as the matching descriptor. Two MTs that are geometrically similar but have
  different local-neighbourhood graphs can be distinguished.
- **Key assumption**: The local neighbourhood is stable across the knife cut (same bundle topology on
  both sides). This is a good assumption for well-organised structures (axoneme, centriole) but weaker
  for disordered networks.
- **Kill-test**: If two parallel MTs have identical local topologies (same number of neighbours at the
  same angles), the topology descriptor gives no discrimination — same as the fingerprint uniform-field
  failure.

### Map-conflation iterative RANSAC with polyline descriptors

- **Pitch**: Represent each road segment by a descriptor vector (length, orientation, curvature,
  neighbourhood density). Run RANSAC: sample a small set of segment pairs, estimate the transform,
  count inliers. Iterate. The best transform + inlier set gives the seeded correspondences.
- **Mechanism**: Polyline descriptor = tangent histogram + local curvature + length. RANSAC samples
  2–3 pairs, estimates similarity transform by horn's method, counts all pairs with descriptor distance
  below threshold at the estimated transform. Standard RANSAC convergence guarantees apply.
- **Inspiration**: Exactly applicable to MTs. MT descriptor = (endpoint tangent angle, tangent curvature,
  local density). RANSAC over 2 MT pairs + horn's method → fast, robust, no initialisation. This is the
  simplest complete pipeline for the alignment-seed step.
- **Key assumption**: At least a few percent of random 2-pairs give a correct transform hypothesis.
  For ~200 MTs, even 5% correct pairs → RANSAC needs ~50 iterations, trivially fast.
- **Kill-test**: If all MTs are identical descriptors, RANSAC generates no discrimination between
  transform hypotheses. Votes are uniformly distributed. Same failure as all other descriptor-based
  methods.

**What transfers.**
The local-neighbourhood graph descriptor idea directly addresses the parallel-neighbour ambiguity: two
MTs with the same tangent but different local topologies (one is at the bundle edge, one is interior;
or one has a gap-neighbour pattern different from the other) can be distinguished by their neighbourhood
descriptor. This is an additional signal the current pipeline ignores entirely.

---

## Field 7 — Genome scaffold/contig joining (sequence assembly with a gap)

**The shape match.** A genome assembler produces contigs (long, nearly-contiguous DNA sequences) that
must be ordered, oriented, and gap-estimated to build a scaffold. Two adjacent contigs share no sequence
(the gap is unknown length) but their *ends* share structural signals: read-depth drops, long-read
spanning reads, Hi-C contact frequency. Near-identical repeat regions cause wrong joins.

**Mechanisms.**

### Long-read spanning + gap-size estimation (ONT/PacBio scaffold bridging)

- **Pitch**: Long reads that span the gap between two contigs are direct evidence of adjacency and
  orientation. Their alignment positions at both ends give the gap size. If multiple long reads agree,
  confidence is high; if they disagree (indicating a repeat), abstain.
- **Mechanism**: Map all reads to all contigs. For every pair of contigs, count reads that map uniquely
  to the end of contig A and the start of contig B. Threshold on count (≥3 reads) and consistency
  (variance of implied gap size < σ threshold). Wrong joins flagged by low count or high variance.
- **Inspiration**: The EM grayscale signal is the analog of long reads. If MT A in section 1 and MT B
  in section 2 are a true pair, the grayscale *just inside* each section's cut face should look similar
  (same local density, same surrounding MT pattern). Sample a small patch around each endpoint and
  compare — high similarity = evidence for the pair. Multiple supporting signals (tangent, density,
  grayscale) = high confidence; disagreement = abstain.
- **Key assumption**: The grayscale near the cut face is informative (not corrupted by the knife damage
  or carbon contamination). For cryo-ET this is a real concern — the outermost slices are often the
  noisiest. Use the 5–10th slice from the face rather than the outermost.
- **Kill-test**: If the cut face grayscale is uniformly noisy (bad section quality), this signal
  provides no discrimination. Detect by checking variance of the grayscale signal; fall back to
  geometry-only if noisy.

### Repeat-aware contig graph (Bandage / metaFlye graph simplification)

- **Pitch**: Build an assembly graph where edges represent possible joins. Disambiguate repeat-induced
  edges by *coverage depth* and *paired-end constraint*: a true join has depth ≈ mean; a repeat join
  has depth ≈ 2× mean. Tangles (multiple equally likely paths) are left unresolved (abstention).
- **Mechanism**: Coverage depth is a global signal that is *not* affected by the local sequence
  ambiguity. Similarly, paired-end reads give physical-distance constraints. Combined, they let the
  assembler assign a confidence to each edge in the graph and safely abstain where confidence is low.
- **Inspiration**: MT density is the analog of coverage depth. If a region has twice the expected MT
  density, it may contain two overlapping bundles that are individually ambiguous. Flag high-density
  regions for lower-confidence abstention rather than forcing a match.
- **Key assumption**: Expected MT density is known (or estimable from the bulk of the section). True
  for any section with a reasonable number of MTs.
- **Kill-test**: A genuine local density increase (not a repeat/overlap artefact) would trigger false
  abstentions. Mitigate by comparing the density in both sections at the interface region.

**What transfers.**
The grayscale-patch similarity as evidence for MT pair identity is the clearest transfer. It introduces
an independent signal (not derived from spline geometry) that can confirm or veto a geometric match,
directly attacking the chicken-and-egg: run geometry-first to generate candidates, then score each
candidate by grayscale patch similarity, then commit.

---

## Summary of the best transferable mechanisms

| Rank | Mechanism | From field | Handles lateral-jump? | Handles parallel-neighbour? |
|------|-----------|-----------|----------------------|----------------------------|
| 1 | **Generalised Hough / pairwise transform voting** | Fingerprint, astrometry | Yes — votes cluster at the true jump | Yes — wrong pairs scatter votes |
| 2 | **MHT hypothesis tree (defer commitment)** | Radar tracking | Yes — carries jump-size distribution | Yes — ambiguous nodes stay branched |
| 3 | **Grayscale-patch cross-validation** | Genomics (spanning reads) | Neutral (geometry gives the jump) | Yes — patch similarity breaks tie |
| 4 | **Min-cost-flow / BP on bipartite graph** | Particle physics | Neutral | Yes — abstention via dummy sink |
| 5 | **Local-neighbourhood graph descriptor** | Cartographic conflation | Neutral | Yes — topology differs even when geometry matches |
| 6 | **DTW with affine gap penalty** | Ice-core dating | Yes — gap modelled explicitly | Partial — path width = per-pair confidence |

---

## Concrete ideas

### Idea A — Pairwise-transform voting (steal from fingerprint / astrometry)

- **Pitch**: For every (MT_i in A, MT_j in B) pair, compute the implied rigid transform (Δx, Δy, Δθ).
  Accumulate votes in a 3D histogram. Peak = alignment seed. Correspondences = pairs whose implied
  transform falls within σ of the peak. Alignment and correspondence solved jointly without initialisation.
- **Mechanism**: Generalised Hough transform over MT-pair hypotheses. Bin width ≈ expected position noise
  (~5 nm) and angle noise (~2°). Sub-peak localisation by centroid of the top bin cluster.
- **Inspiration**: Fingerprint minutiae Hough (Jiang & Yau 2000); astrometry.net (Lang et al. 2010).
- **Key assumption**: The correct alignment transform is the modal vote (plurality winner). True if >~20%
  of pairs are true matches, which holds unless the network is nearly uniformly hexagonal.
- **Kill-test**: Build a synthetic perfect hexagonal bundle and check whether the vote histogram has a
  single sharp peak or is flat. If flat, add a 4th dimension (local density) to break ties.

### Idea B — Multi-hypothesis alignment + correspondence (steal from MHT)

- **Pitch**: Generate the top-K alignment transforms from the Hough vote. For each transform, compute
  the greedy correspondence (nearest neighbour in aligned space). Score each (transform, correspondence)
  pair by a joint cost: alignment residual + sum of descriptor distances + fraction abstained. Commit to
  the top-1 pair; abstain on individual MTs that are in the top-2 with close scores.
- **Mechanism**: K = 5–20 hypotheses (cheap to generate from Hough top-K bins). The joint cost avoids
  degenerate solutions where a wrong transform perfectly aligns a subset.
- **Inspiration**: Reid (1979) MHT, Murty's algorithm for K-best assignments.
- **Key assumption**: The correct (transform, correspondence) pair has strictly lower joint cost than
  any wrong pair. Empirically verified by running on the test stack.
- **Kill-test**: Measure joint cost of the correct vs second-best hypothesis on Monopoles_test sec01→02.
  If the gap is <5%, the scoring function needs an additional term.

### Idea C — Grayscale patch cross-check (steal from long-read scaffolding)

- **Pitch**: For every candidate MT correspondence pair (i, j), extract a 3D grayscale patch (radius r
  voxels) around MT_i's endpoint in section A and MT_j's endpoint in section B. Compute normalised
  cross-correlation (NCC) or SSIM. Use NCC as a multiplicative weight in the assignment cost.
- **Mechanism**: NCC is computed on the already-aligned (or approximately aligned) volumes. r = ~5–15
  voxels (tuneable). Pairs with NCC < threshold are vetoed (abstain). This is a second-stage filter
  after geometry-based candidate generation.
- **Inspiration**: ONT long-read spanning reads; also multi-modal fingerprint matching.
- **Key assumption**: The grayscale near the cut face is informative — not dominated by knife damage or
  deposition artefacts. Validated by checking that NCC between true pairs is higher than between random
  pairs on the test stack.
- **Kill-test**: On Monopoles_test sec01→02, extract NCC scores for all known true pairs (from the
  noisy GT). If the NCC distribution for true vs wrong pairs overlaps substantially (AUC < 0.65),
  the signal is too weak and this filter should be disabled or down-weighted.

### Idea D — Local bundle topology descriptor (steal from cartographic conflation)

- **Pitch**: For each MT endpoint, compute a descriptor of its local neighbourhood: sorted list of
  (distance, relative tangent angle) for its K nearest MT neighbours. Compare descriptors across
  sections using a ranked-list distance (Kendall τ or Hausdorff on the K-NN set). Use as an additional
  term in the assignment cost.
- **Mechanism**: Descriptor is computed on each section independently; it is invariant to global
  translation and approximately invariant to small rotations. Two parallel MTs with different neighbour
  configurations get different descriptors even if their own tangent is identical.
- **Inspiration**: SHOT descriptor (3D shape context), SURF on road networks.
- **Key assumption**: The local neighbourhood is preserved across the knife cut (same K neighbours on
  both sides). True for well-organised bundles; weaker for loose networks.
- **Kill-test**: Compute descriptor self-similarity between the same MT's neighbours in consecutive
  sections of the same dataset. If mean Kendall τ > 0.6, the descriptor is stable; if < 0.4, discard.

### Idea E — Min-cost-flow assignment with abstention (steal from particle physics)

- **Pitch**: Replace the Hungarian assignment with a min-cost-flow formulation on a bipartite graph
  (section A endpoints, section B endpoints, plus source/sink nodes). Each MT-pair edge has cost =
  descriptor distance. Unmatched nodes flow through a dummy sink with cost = abstention_penalty.
  The solver gives the globally optimal assignment including abstentions.
- **Mechanism**: Min-cost-flow is polynomial (Successive Shortest Paths, O(N³) worst case, O(N² log N)
  practical). It naturally handles 1:1 + abstention without Lagrange multipliers or tuning.
  abstention_penalty is a single global parameter (set by CV on the training stack or by the expected
  fraction of missing MTs).
- **Inspiration**: TrackML challenge winner (Exa.TrkX), LAP tracker (Jaqaman et al. 2008 Nature Methods).
- **Key assumption**: abstention_penalty can be set without per-dataset tuning. Set it as a quantile
  of the within-section inter-MT descriptor distances.
- **Kill-test**: Run on sec01→02 interface with the known GT. Measure precision/recall vs the current
  Hungarian. If the min-cost-flow gives lower recall than Hungarian (because abstention_penalty is too
  high), measure sensitivity to the penalty and set it adaptively.

### Idea F — Sequence DTW on bundle cross-section (steal from ice-core dating)

- **Pitch**: Order MT endpoints in each section by their angular position around the estimated bundle
  centroid (or by x-position for a planar section). Treat the ordered sequence of (tangent angle, local
  density) vectors as a 1D time series. Run DTW with affine gap penalty between the two sequences.
  The DTW path gives correspondences; the gap cost gives the expected number of missing MTs.
- **Mechanism**: DTW cost matrix is N×M (N, M = MT counts per section). Affine gap: open = expected
  cost of one missing MT, extend = 0 (contiguous gap). Path backtrack gives correspondence list.
  Path width at each position gives per-pair confidence.
- **Inspiration**: Baillie & Pilcher wiggle-matching; also used in cell lineage alignment.
- **Key assumption**: The bundle has a stable angular ordering across sections (no scrambling of the
  circular order). True for well-organised bundles (axoneme: 9+2); weaker for disordered networks.
- **Kill-test**: Compute the fraction of Monopoles_test MT pairs that violate the circular ordering.
  If >20% violate it, DTW ordering assumption is broken; fall back to 2D grid ordering or skip this idea.

---

*Written 2026-06-11 by the Cross-Domain Analogist agent.*
