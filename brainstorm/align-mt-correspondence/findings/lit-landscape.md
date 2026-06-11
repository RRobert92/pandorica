# Literature Landscape: Alignment and MT Correspondence Across Serial Cryo-ET Sections

**Agent**: LITERATURE LANDSCAPE  
**Date**: 2026-06-11  
**Problem**: Pandorica must (1) align consecutive tomograms and (2) connect MT splines across section boundaries. Currently a single Hungarian endpoint-matcher feeds both goals, causing mutual poisoning. Need a survey of families of approaches to both sub-problems.

---

## Family 1: Direct MT Stitching in Serial-Section EM (Most Relevant Prior Art)

### Core mechanism
Two papers address nearly this exact problem. Both treat MTs as oriented line-segment sets and solve alignment + correspondence jointly but sequentially rather than simultaneously.

**Weber et al. (2014)** — "Automated Stitching of Microtubule Centerlines across Serial Electron Tomograms", PLoS One, PMC4249889:
- **Alignment (coarse)**: Distance Compatibility Graph (DCG). Pairs of endpoints that preserve mutual distance ratios across sections form a graph; maximal cliques (Bron-Kerbosch) give point correspondences; SVD (Umeyama) gives the optimal rigid transform. Orientation of MT axes reduces graph complexity.
- **Alignment (fine, linear)**: Two EM variants — one treating MT angles as Fisher-Mises distributions, one factorising position (Gaussian) + orientation (Fisher-Mises). Solved with IPOpt.
- **Alignment (elastic)**: CPD variant where each endpoint gets its own translation vector, with a smoothness regularizer; solved via variational calculus.
- **Correspondence**: Markov Random Field (MRF) / factor graph with belief propagation. Singleton factors = exponential cost on (a) direct endpoint distance, (b) projected distance along MT axis extension, (c) angle deviation. Pair factors = penalty when neighbouring MT assignments disagree spatially. Solved by max-a-posteriori belief propagation. Non-convergence resolved by manual intervention at "critical nodes".
- **GT dependency**: Parameters set via maximum-likelihood from expert-labelled examples; τ=0.1 placeholder.
- **Failure cases**: Failed completely on sub-pellicular arrays in T. brucei (highly parallel, uniform spacing → DCG cliques degenerate). Required manual input for ambiguous cases.

**Lindow et al. (2021)** — "Semi-automatic stitching of filamentous structures in image stacks from serial-section electron tomography", J. Microscopy 2021 (incl. Robert Kiewisz as co-author):
- Interactive/semi-automatic; IMOD-like landmark placement with real-time TPS preview.
- Alignment: non-linear moving-least-squares (MLS) warping based on manually selected MT-pair landmarks.
- Correspondence: also manual; software assists by suggesting candidates.
- Handles very large spindle datasets (C. elegans, X. laevis) that Weber's fully-automatic method handled only partially.
- Key insight: for dense, ambiguous bundles, human judgment at "hard interface" sections is still needed.

**Maturity**: High. This is the state of the art for this exact domain.  
**Sub-problem**: Both (alignment + correspondence).  
**Why it matters for pandorica**: Pandorica is the successor; the literature confirms that sequential solve (align first, then correspond) with orientation-aware costs and pair-factor smoothing is the right structure, but that fully automated methods struggle on dense parallel arrays without GT — the same failure Pandorica exhibits.

---

## Family 2: Probabilistic Point-Set Registration — CPD / GMM / TPS-RPM

### Core mechanism
Treat one endpoint set as GMM centroids, the other as data. Maximize likelihood jointly over correspondences and transformation. Soft-assign avoids hard Hungarian step.

**Coherent Point Drift (CPD)** — Myronenko & Song, NIPS 2006 / TPAMI 2010:
- GMM centroids (set A) move coherently (regularized smoothness) to fit data (set B).
- E-step: soft posterior P(m|x) = responsibility of centroid m for point x.
- M-step: update transformation using weighted point sums.
- Non-rigid variant: displacement field in RKHS, regularized by bandwidth; bandwidth annealed from coarse to fine.
- Outlier class: uniform GMM component absorbs non-matching points without hard rejection.
- Already used in Weber 2014 (elastic alignment step).

**TPS-RPM** — Chui & Rangarajan 2003, revisited Yang et al. 2011:
- Softassign + deterministic annealing + TPS deformation field.
- Simultaneous registration and correspondence by annealing temperature; at high T, all correspondences are soft/uniform; as T decreases, correspondences harden and TPS refines.
- Double-sided outlier handling (Yang 2011): both sets can have outliers, not just one.

**Structured Analytic CPD (SA-CPD)** — arXiv 2605.00934 (2025):
- Extends CPD for large-deformation scenarios with analytic gradient.

**DeepGMR** — arXiv 2008.09088:
- Learns the GMM structure via a neural network; reduces to a closed-form registration.

**Maturity**: Very high for general point sets. Biological filament applications (Weber 2014) exist.  
**Sub-problem**: Both (soft correspondence + non-rigid warp).  
**Key assumption for MT use**: Points must have some spatial spread (not all on a grid/regular lattice). For extremely parallel dense bundles, GMM centroids collapse — same failure mode as DCG cliques.  
**Kill-test**: Run CPD non-rigid on a single hard interface from the Monopoles dataset; does it whirlpool or converge to a stable warp?

---

## Family 3: Joint Registration + Correspondence — EM-ICP Variants

### Core mechanism
EM-ICP integrates correspondence uncertainty directly into the M-step transformation update. Unlike hard-ICP (reassign to nearest), EM-ICP uses weighted contributions from all possible correspondences.

- E-step: P(correspondence j → i) ∝ exp(−||x_i − T(y_j)||² / σ²).
- M-step: compute T that minimises weighted sum over all pairs.
- σ annealed: starts wide (all-soft, alignment-independent) and narrows; avoids local minima better than hard ICP.
- Key property: at high σ, transformation is driven by global centroid alignment; at low σ, fine local correspondence.

**Variants with orientation**:
- Fisher-Mises ICP (Weber 2014 fine alignment): joints position and direction; useful when MTs are not spatially symmetric.
- Semantic ICP (Parkison & Gan 2018): adds semantic class labels to E-step; analogous to using MT tangent direction as a "semantic" feature.

**Maturity**: Very high.  
**Sub-problem**: Alignment primarily; correspondence emerges but is not the final goal.  
**Why promising for MT**: σ-annealing avoids whirlpool by starting from a global rigid alignment. Orientation terms can be bolted on without GT.

---

## Family 4: Multi-Object Tracking and Data Association (MOT / LAP / MHT)

### Core mechanism
Tracking literature addresses correspondence across time; cross-section MT linking is equivalent to tracking across two "frames" (sections).

**LAP / Hungarian** (what pandorica already does):
- Bipartite cost matrix, solved in O(n³). Minimises global assignment cost in one shot.
- Drawback: hard decisions, no uncertainty propagation; a bad cost function produces a bad unique assignment.

**Multi-Hypothesis Tracking (MHT)** — Reid 1979; Cham & Rehg 1999:
- Instead of committing to one assignment per frame, branch a tree of hypotheses.
- Prune by likelihood; defer commitment until future evidence resolves ambiguity.
- Particularly useful when local ambiguity is resolvable with downstream context (MT trajectory continuing further into the section).
- Complexity: tree grows exponentially; pruned by N-best or track score thresholds.

**JPDAF** — Bar-Shalom:
- Marginalise over all hypotheses to get soft association probabilities; each "track" gets a weighted update.
- Can be approximated by belief propagation (Williams & Lau 2014 arXiv:1209.6299) → connection to Weber's MRF.

**LAP-tracker (biology focus)** — Jaqaman et al. 2008, Nat. Methods:
- Two-step LAP: frame-to-frame linking + gap-closing. Used widely for particle/filament tracking in fluorescence microscopy.
- Gap-closing stage explicitly handles merge/split/break events — directly analogous to MT tips that terminate inside a section (truncation) vs. true cross-section boundary.

**Maturity**: Very high in computer vision and fluorescence biology. Under-applied to cryo-ET spline data.  
**Sub-problem**: Correspondence (cross-section identity).  
**Key assumption for MT use**: Cost function must encode MT endpoint position + tangent direction + local density context. LAP-tracker gap-closing is the right abstraction for handling truncated/partially-visible MTs.

---

## Family 5: MRF / Factor Graph with Pair Consistency Factors

### Core mechanism
Model the assignment of each MT pair (A_i → B_j) as a discrete random variable. Add pairwise potentials that reward spatially consistent assignments (nearby MTs in A should map to nearby MTs in B, preserving local topology). Solve by belief propagation or message passing.

This is exactly what Weber 2014 did for the correspondence step, and it is the natural extension of vanilla Hungarian to include context.

**Connection to modern methods**:
- SuperGlue (Sarlin et al., CVPR 2020) is essentially this: GNN with self-attention (within-section context) + cross-attention (between-section context) + Sinkhorn (differentiable soft assignment). Its matching head solves the same MRF but via a learned, differentiable approximation.
- LightGlue (Lindenberger et al., ICCV 2023): faster transformer-based variant; adapts depth dynamically; achieves LoFTR accuracy at 8× speed.

**Key difference from vanilla Hungarian**: pair factors prevent whirlpool by penalising topologically inconsistent assignments; the same constraint that prevents two nearby MTs from "crossing" their assignments.

**Maturity**: Weber 2014 = mature, hand-crafted, needs GT for parameter estimation. SuperGlue/LightGlue = learned, needs training pairs but no explicit GT per-pair (self-supervised on synthetic or real transformations).  
**Sub-problem**: Correspondence (pair consistency also stabilises alignment if warp is co-optimised).

---

## Family 6: Spectral Graph Matching and Optimal Transport

### Core mechanism
Represent each endpoint set as a graph (nodes = MT endpoints, edges = proximity or MT-connectivity). Match graphs by aligning their spectral structure (eigenvectors of the Laplacian) or by solving optimal transport between node distributions.

**Spectral methods** — Shapiro & Brady 1992; Leordeanu & Hebert 2005:
- Build affinity matrix A where A[(i,j),(k,l)] = compatibility of (A_i→B_k) with (A_j→B_l).
- Leading eigenvector of A gives soft assignment scores.
- Cheap, differentiable, but ignores global topology.

**Graph matching via OT** — arXiv 2111.05366:
- Cast as Gromov-Wasserstein problem: find transport plan π that minimally distorts pairwise distances.
- Partial OT (arXiv 2410.16718) handles missing nodes (MTs truncated at section boundary).

**OT-GM for 3D retinal OCT** — arXiv 2203.00069:
- Graph matching applied to 3D biomedical images; nodes are vascular bifurcations (analogous to MT endpoints).

**Maturity**: Medium. OT theory is mature; application to oriented filaments is sparse.  
**Sub-problem**: Correspondence (and implicitly alignment via the transport cost).  
**Key assumption for MT use**: Gromov-Wasserstein preserves internal distance structure; valid if the warp between sections is smooth enough that pairwise inter-MT distances are approximately preserved. Breaks for severe local deformation.  
**Kill-test**: Compute the Gromov-Wasserstein distance between two matched sections with known GT and compare to distance under whirlpool warp.

---

## Family 7: Learned Sparse Feature Matching (SuperGlue / LightGlue Paradigm)

### Core mechanism
Encode each "keypoint" (here: MT endpoint) with a descriptor that captures local context. Use a GNN with self-attention (within one section) and cross-attention (between sections) to let each endpoint gather information about the global configuration before the matching step. Solve matching via Sinkhorn (differentiable soft assignment with dustbin for unmatched points).

**SuperGlue** — Sarlin et al., CVPR 2020 (oral):
- Descriptors: hand-crafted (SIFT) or learned (SuperPoint). For MTs: tangent direction, local curvature, density, distance from section boundary.
- GNN: alternating self and cross attention layers (context aggregation).
- Sinkhorn: optimal partial transport; dustbin handles MTs with no valid continuation.
- Trained on synthetic homographies + real image pairs. No per-image GT needed at test time.

**LightGlue** — Lindenberger et al., ICCV 2023:
- Simplifies SuperGlue; early-exit for easy matches; adaptive depth.
- More accurate and faster than SuperGlue; competitive with dense matchers (LoFTR).

**Biological adaptation**:
- The MT endpoint descriptor would be: (position, tangent, curvature, local MT density, distance to section face, image patch around endpoint tip).
- Self-attention aggregates within-section context (nearby MTs inform each MT's expected continuation).
- Cross-attention propagates between-section evidence (does the image texture on both sides agree?).
- No external GT needed: train on synthetic deformations of real MT graphs (rotate/translate/warp one section, treat original as GT).

**Maturity**: High in computer vision. Zero published cryo-ET applications found; biological microscopy applications are sparse but feasible.  
**Sub-problem**: Correspondence primarily; alignment can be recovered from matched pairs.  
**Key assumption for MT use**: Sufficient distinguishing context per endpoint (not all endpoints look the same). Dense parallel bundles may be ambiguous even with context — the fundamental limit.  
**Kill-test**: Train on 10 synthetic augmentations of a real section pair; does Sinkhorn recall exceed Hungarian by >10%?

---

## Family 8: Dense Image-Based Registration (Mutual Information / Cross-Correlation / Deep)

### Core mechanism
Use the grayscale image volumes directly to estimate the alignment warp, independent of the MT annotations.

**Classical**: Normalised cross-correlation (NCC) or mutual information (MI) optimised over rigid/affine/TPS parameters. Used for tilt-series alignment; phase correlation in Fourier domain for translation.

**Deep non-rigid**: VoxelMorph (Balakrishnan et al. 2019, CVPR) and successors — U-Net predicts dense displacement field from a pair of volumes; trained on deformation plausibility (Jacobian regularity loss) without explicit GT correspondences. Can be applied to 2D projection images or 3D subvolume pairs.

**Application to cryo-ET cross-section alignment**: The grayscale projection of each section shows MT cross-sections (dots in 2D), membrane texture, and ice/carbon background. Image-based registration does not require any spline annotation and is therefore independent of MT tracing quality.

**Pandorica already does some image-based coarse alignment** (from project memory: image-only coarse warp IS estimable cross-gap via big-window block_match + RANSAC inliers). The literature supports expanding this.

**Maturity**: Very high for general image registration. Moderate for cryo-ET specifically (fiducial-free alignment literature exists: Marker-free image registration — Sorzano et al., PMC2694187).  
**Sub-problem**: Alignment only (does not solve correspondence directly, but provides a warp to pre-align before correspondence).  
**Key assumption for MT use**: Image texture (MT cross-section dots, membrane patterns) is reproducible enough across sections to anchor the warp. Heavily contaminated or featureless sections will fail.  
**Kill-test**: Apply VoxelMorph fine-tuned on MT tomogram pairs; measure displacement field smoothness vs. current warp at known-bad interfaces.

---

## Family 9: Decoupled / Staged Pipelines (Align-Then-Correspond vs. Alternate)

### Core mechanism
The literature broadly acknowledges the "chicken-and-egg" problem (correspondences needed for alignment; alignment needed for correspondence). Three resolution strategies exist:

1. **Sequential (align first, then correspond)**: Current pandorica approach; Weber 2014 also does this. Fragile if alignment is wrong.
2. **Alternating / EM-style**: Use current correspondences to estimate transform (M-step); use current transform to update soft correspondences (E-step). CPD, EM-ICP, TPS-RPM all do this. Converges to local optimum; avoids hard chicken-and-egg by softening both.
3. **Coarse-to-fine with separate signals**: Use image-based alignment (family 8) for coarse warp independent of MTs; use MT geometry only for fine correspondence. The two signals are decoupled and do not poison each other. This is the architectural principle behind pandorica's own coarse→fine design (project memory).

**Key insight from literature**: The alternating / EM approach (family 3) with soft correspondence is strictly more robust than a single-shot Hungarian. The decoupled approach (separate image signal for alignment, MT geometry for correspondence) is the cleanest architectural split.

---

## Family 10: Probabilistic / Bayesian Chain Tracking (Gap-Closing)

### Core mechanism
Model each MT as a "track" that can be born at one section face and die at another. Each endpoint at the bottom face of section k is a "detection" at time k; each endpoint at the top face of section k+1 is a "detection" at time k+1. The correspondence problem becomes a tracking problem.

**Jaqaman et al. 2008 LAP-tracker** (Nat. Methods):
- Frame-to-frame LAP for linking.
- Second LAP for gap-closing: MTs can merge, split, or have a gap (section without a detection due to truncation or tracing failure). Costs encode position + direction continuity.
- Directly handles the case where a MT is traced in section k-1 and k+1 but has no detection in section k.

**Kaplan et al. / KiT (kinetochore tracker)**: Bayesian tracking applied to kinetochore spots in fluorescence; relevant because kinetochores sit at MT plus-ends — the tracking geometry is similar.

**MHT for ambiguous dense bundles** (Cham & Rehg 1999): Defer commitment when multiple MTs are equidistant from a candidate continuation. Only assign once trajectory context resolves the ambiguity. Cost: exponential tree; need aggressive pruning.

**Maturity**: High in fluorescence tracking, low in cryo-ET spline domain.  
**Sub-problem**: Correspondence (chain-building across sections, including gap-closing for truncated MTs).  
**Key assumption for MT use**: MT direction is roughly preserved across a thin section gap (true for section thicknesses < 200 nm). Failure if MTs curve sharply within one section.

---

## Ideas Derived from the Landscape

### Idea A: Soft-Assign EM Loop (CPD-style) with Orientation-Augmented Metric
- **Pitch**: Replace the single-shot Hungarian with an annealed EM loop that simultaneously softens the correspondence and refines the warp, using orientation + position jointly.
- **Mechanism**: Treat MT endpoints as GMM centroids. Augment the Mahalanobis distance with MT tangent direction (Fisher-Mises component), local MT density (bandwidth term), and distance-from-face (prior on matching probability). Run E-step (soft responsibilities) → M-step (update TPS warp) → decrease σ → repeat. Dustbin class absorbs unmatched endpoints. Final correspondences = argmax of responsibilities.
- **Inspiration**: CPD (Myronenko & Song 2010), Weber 2014 elastic step, TPS-RPM (Chui & Rangarajan 2003).
- **Key assumption**: The local spatial coherence of MT positions is preserved modulo a smooth warp; i.e., the deformation field is smooth enough that nearby MTs co-move.
- **Kill-test**: Run on the sec01→02 hard Monopoles interface; does σ-annealing converge to a non-whirlpool warp where the direct Hungarian diverges?

### Idea B: Decoupled Two-Signal Architecture (Image Warp + MT Correspondence Separately)
- **Pitch**: Compute the section-to-section warp from the raw image volumes alone (no MTs), then solve MT correspondence on the pre-aligned point clouds with a simple, high-recall matcher.
- **Mechanism**: Step 1 — dense image registration (VoxelMorph or block-match + RANSAC, as in current coarse stage) on the grayscale tomogram pair to get a displacement field D. Step 2 — apply D to all MT endpoints in section B, transforming them into section A's coordinate frame. Step 3 — solve MT correspondence on the now pre-aligned endpoints using a low-threshold cost (position + tangent only; image warp already removed most of the gap). Step 4 — run fine MT-driven warp refinement on inlier correspondences only. Steps 1 and 3 never feed each other, eliminating the poison loop.
- **Inspiration**: Pandorica's own coarse→fine architecture (project memory: coarse image similarity → fine MT+image warp); VoxelMorph; ssTEM registration pipelines.
- **Key assumption**: The image signal is informative enough to produce a coarse warp that reduces endpoint displacement below the ambiguity radius (i.e., after image alignment, no two candidate MT pairs are within the same distance as the MT-to-MT spacing).
- **Kill-test**: After image-only alignment, measure the residual endpoint displacement distribution; if the 90th percentile is below the mean inter-MT spacing, the matcher will operate in an unambiguous regime.

### Idea C: Pair-Consistency MRF with Topology Preservation (Weber++, No GT)
- **Pitch**: Replace the single-shot Hungarian with an MRF where pair factors enforce that nearby MTs in A map to nearby MTs in B, estimating the regularisation weight from data (no GT).
- **Mechanism**: Build a spatial proximity graph on MT endpoints within each section (k-NN or Delaunay). For each edge (i,j) in section A and each pair of candidate assignments (i→k, j→l), add a pair potential penalising |displacement(i,k) − displacement(j,l)|. Singleton potentials = position + tangent cost. Solve by loopy BP or mean-field. Regularisation weight λ estimated by maximising consistency of the assignments on easy (low-ambiguity) pairs first, then applying to hard pairs (empirical Bayes / self-calibration).
- **Inspiration**: Weber 2014 MRF; SuperGlue GNN context aggregation; MRF pedestrian tracking (MDPI Sensors 2020).
- **Key assumption**: Local MT topology (which MTs are neighbours) is approximately preserved across sections. Fails if sections have dramatically different local MT organisation (e.g. new MTs nucleated, or large-scale rotation of a spindle half).
- **Kill-test**: On a labelled interface, does enforcing pair consistency increase precision without reducing recall vs. vanilla Hungarian? (Bootstrap estimate of λ from top-50% confidence matches.)

### Idea D: GNN Endpoint Descriptor + Sinkhorn Assignment (SuperGlue-for-MTs)
- **Pitch**: Train a graph neural network to produce orientation- and context-aware descriptors for MT endpoints, then solve correspondence via Sinkhorn optimal transport.
- **Mechanism**: Each MT endpoint gets an initial feature vector: (x, y, z, tangent_x, tangent_y, tangent_z, curvature, local_density, dist_to_section_face). Build a k-NN graph within each section. Apply L layers of alternating self-attention (within-section) and cross-attention (between sections). Feed the resulting descriptors into a Sinkhorn optimal transport layer (with dustbin) to produce a soft assignment matrix. Threshold to get final correspondences. Train on synthetically augmented MT graph pairs (random rigid + random smooth warp of real section annotations); no per-pair GT labels needed — supervision is the known synthetic transformation.
- **Inspiration**: SuperGlue (Sarlin et al., CVPR 2020); LightGlue (Lindenberger et al., ICCV 2023).
- **Key assumption**: Enough contextual variation per endpoint (position, density, tangent) to distinguish true pairs from impostors in dense bundles. In perfectly uniform parallel arrays, even context is degenerate.
- **Kill-test**: Does the GNN descriptor space (UMAP) show separable clusters for paired vs. non-paired endpoints on real data?

### Idea E: MHT with MT Trajectory Priors (Defer Hard Decisions)
- **Pitch**: When two or more MTs in section B are equally plausible continuations of an MT in section A, defer the assignment and carry multiple hypotheses forward until a third section (or image evidence) resolves the ambiguity.
- **Mechanism**: Enumerate top-K assignment hypotheses per MT (K=3 typically sufficient for local ambiguity). Score each by (position cost + tangent cost + image cross-correlation at tip). When processing section k+1, propagate each hypothesis and extend its score. Prune using N-best or a score threshold. Commit to the surviving hypothesis after min(3, remaining sections) steps.
- **Inspiration**: MHT (Reid 1979); Cham & Rehg 1999; Jaqaman 2008 LAP-tracker gap-closing; "Novel MHT particle tracking" PMC4373089.
- **Key assumption**: Ambiguous pairs at one interface are resolved by trajectory context from adjacent interfaces. Fails if the ambiguity persists over multiple sections (e.g. parallel bundle with identical spacing throughout the dataset).
- **Kill-test**: For known-wrong-join errors in the current output, do the correct candidates appear in the top-K hypotheses at the time of commitment?

---

## Summary of Coverage

| Family | Align | Correspond | Maturity | GT needed? |
|--------|-------|------------|----------|------------|
| 1. Direct MT stitching (Weber, Lindow) | Yes | Yes | High | Yes (params) / Manual |
| 2. CPD / GMM / TPS-RPM | Yes | Soft | Very high | No |
| 3. EM-ICP + orientation | Yes | Soft | Very high | No |
| 4. LAP / MHT / JPDAF | No | Yes | Very high | No (cost fn) |
| 5. MRF + pair consistency | Indirect | Yes | High | Calibration only |
| 6. Spectral / OT graph matching | Indirect | Yes | Medium | No |
| 7. SuperGlue / LightGlue | Indirect | Yes | High | Synthetic |
| 8. Image-based registration | Yes | No | Very high | No |
| 9. Decoupled staged pipeline | Yes | Yes | Medium | No |
| 10. Probabilistic chain / gap-closing | No | Yes | High | No |

---

## Most Promising Threads

1. **Decoupled two-signal architecture (Idea B)**: The cleanest fix to the poison loop — image signal drives alignment independently of MT signal. Already partially in pandorica. Literature (image registration, coarse-to-fine) fully supports it. Low risk.

2. **Annealed soft-assign EM loop with orientation (Idea A)**: Replaces the brittle single-shot Hungarian with a provably more robust alternating optimisation. CPD + orientation is well-understood. Can be done without any GT. Medium complexity.

3. **Pair-consistency MRF (Idea C)**: Directly addresses whirlpool and wrong-joins by enforcing local topology preservation. Weber 2014 validated this works for MTs. Key innovation needed: GT-free parameter estimation. Medium complexity.

4. **SuperGlue-for-MTs (Idea D)**: Most powerful long-term, but requires training infrastructure and a reasonable dataset of section pairs. Highest upside, highest setup cost.

5. **MHT gap-closing (Idea E)**: Addresses missed connections without changing the alignment. Low implementation cost if added on top of an improved matcher. Directly targets the "MISS" failure mode.

---

## References

- Weber B., Greenan G., Prohaska S., Baum D. et al. (2014). "Automated Stitching of Microtubule Centerlines across Serial Electron Tomograms." *PLoS One*. PMC4249889.
- Lindow N., Brünig F.N., Dercksen V.J., Fabig G., Kiewisz R., Redemann S., Müller-Reichert T., Prohaska S., Baum D. (2021). "Semi-automatic stitching of filamentous structures in image stacks from serial-section electron tomography." *Journal of Microscopy*. DOI:10.1111/jmi.13039.
- Myronenko A., Song X. (2010). "Point Set Registration: Coherent Point Drift." *TPAMI*. arXiv:0905.2635.
- Chui H., Rangarajan A. (2003). "A new point matching algorithm for non-rigid registration." *CVIU* 89(2–3).
- Yang J. et al. (2011). "The TPS-RPM algorithm: A revisit." *Pattern Recognition Letters* 32(7).
- Sarlin P.E., DeTone D., Malisiewicz T., Rabinovich A. (2020). "SuperGlue: Learning Feature Matching with Graph Neural Networks." *CVPR 2020* (oral). arXiv:1911.11763.
- Lindenberger P., Sarlin P.E., Pollefeys M. (2023). "LightGlue: Local Feature Matching at Light Speed." *ICCV 2023*. arXiv:2306.13643.
- Jaqaman K., Loerke D., Mettlen M. et al. (2008). "Robust single-particle tracking in live-cell time-lapse sequences." *Nature Methods* 5, 695–702.
- Balakrishnan G. et al. (2019). "VoxelMorph: A Learning Framework for Deformable Medical Image Registration." *CVPR 2019 / IEEE TMI*.
- Kiewisz R. et al. (2023/2024). "Accurate and fast segmentation of filaments and membranes in micrographs and tomograms with TARDIS." *bioRxiv* 2024.12.19.629196.
- Alvarez-Gonzalez B. et al. (2023). "Optimal Transport-based Graph Matching for 3D retinal OCT image registration." arXiv:2203.00069.
- Leordeanu M., Hebert M. (2005). "A spectral technique for correspondence problems using pairwise constraints." *ICCV 2005*.
- Bar-Shalom Y., Daum F., Huang J. (2009). "The probabilistic data association filter." *IEEE Control Systems Magazine* 29(6).
- Williams J.L., Lau R. (2014). "Approximate evaluation of marginal association probabilities with belief propagation." arXiv:1209.6299.
