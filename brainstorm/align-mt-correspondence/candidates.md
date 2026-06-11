# Candidates (Phase 4 synthesis)

The deep dives reframed the problem. The landscape's favourite — *full decoupling, drop the
matcher, warp from image only* — was **empirically refuted** (DD1/H1: image coarse leaves 1.18ρ;
the MT warp is needed to reach 0.15ρ; image-fill depends on the MT warp). So the matcher stays —
but it can be made ballistic, confident, density-aware, and honestly abstaining. The five
candidates below **compose** into one stack (a ladder by cost/leverage), they do not compete.

Core reframed diagnosis: the coarse (image) alignment has ~1ρ residual; the matcher's gates are
tight relative to that residual, so true pairs fall *outside* the gates (misses) while forced 1:1
assignment in dense parallel bundles manufactures wrong joins, and the bad pairs that survive feed
the TPS (whirlpools). Every candidate attacks one link of that chain.

---

## C1 — Ballistic gate re-centering  [hours; highest leverage; empirically validated]
- Pitch: centre the matcher's distance gate and cost on the tangent-PREDICTED landing point, not the raw endpoint.
- Mechanism: for each boundary MT, predict `p̂ = p_xy + t_xy·(gap_z/dz)` and run the distance gate + cost on `p̂`. DD1 measured this collapses the effective post-coarse residual 8× (0.39ρ→0.05ρ, 98% of MTs), so true partners land inside tight gates (un-misses them) and parallel neighbours separate (fewer wrong joins). The gap is ballistic because MT persistence length ≫ section thickness.
- Provenance: first-principles (ballistic-jump), DD1/H3 (measured 8×), cross-domain (Kalman/Hough seeding). Code: `dir` already carries the tangent; `coords` recoverable for a better tangent.
- Strongest open question: estimating `gap_z` (lost-slab thickness) and `dz` robustly when section/cut thickness is uncertain or varies.

## C2 — Two-witness per-join confidence + honest abstention  [days; fixes wrong joins + gives breaks]
- Pitch: score every candidate join by geometry AND an independent image witness; commit only on score + margin; abstain-as-group when ambiguous.
- Mechanism: per-join `score = L1 + L2`. L1 = ballistic-residual log-likelihood `−‖p_B−p̂‖²/2δJ²`. L2 = grayscale continuity along the predicted tangent at the endpoints (cross-face NCC / lumen-intensity step), normalised by a random-neighbour null. Commit top partner iff `score>τ AND margin>μ`; if two partners within measured `δJ`, abstain → commit centroid to alignment, emit honest break to chain. Fills the per-join hole (`_confidence` is interface-aggregate only today).
- Provenance: first-principles (two-witness), DD3 (full model + independence argument), contrarian (abstention-as-product), impl (dummy-LAP / unbalanced Sinkhorn).
- Strongest open question: does L1⊥L2 independence actually hold? (DD3 kill-test: regress L2 on L1 residual; `|ρ|>0.3` ⇒ the warp leaked MT patches and additivity is invalid.) And is L2 computable AT MT positions without contaminating the warp.

## C3 — Crossing-penalty QAP matcher  [days; the density-breaker Weber lacks]
- Pitch: replace the linear Hungarian with a quadratic assignment whose pairwise edge cost penalises CROSSING pairings.
- Mechanism: objective = unary ballistic cost (C1) + pairwise crossing penalty; solve with pygmtools IPFP (PyTorch/CUDA). The crossing term is *asymmetric* under a parallel-neighbour swap (the swap creates a crossing the correct pairing doesn't), so it resolves the degeneracy Weber's *symmetric* displacement-smoothness factor cannot. This is the principled generalisation of the existing post-hoc `uncross_pairs`.
- Provenance: DD2 (identified as THE mechanism that beats Weber on density), impl (pygmtools QAP), lit (Weber++ MRF), existing `uncross_pairs`.
- Strongest open question: does QAP beat `LAP + ballistic + uncross` enough to justify the complexity, or is the cheap stack already sufficient on real data? (primary validation target)

## C4 — Multi-section flow + triple-overlap consistency  [weeks; global robustness, long stacks]
- Pitch: solve cross-section MT identity over the whole stack jointly, with a free k,k+1,k+2 consistency check.
- Mechanism: min-cost flow over `(section, MT)` nodes across all interfaces (replaces pairwise union-find); triple-overlap curvature-coherence (`|Δcurvature|<κ_max`) + pose-cycle-closure prune stacked wrong joins that no single interface can catch. micron/mtrack (Funke lab, MIT, built for MT tracking in EM) is the reusable substrate.
- Provenance: DD2 (rank 1 for trajectory context), DD3 (triple-overlap free check), impl (micron / Ultrack min-cost flow), cross-domain (combinatorial Kalman across layers).
- Strongest open question: cost/complexity at scale; does pairwise + triple-overlap already capture most of the gain without full global flow?

## C5 — Partial decoupling: warp keeps MTs, CHAIN uses the independent witness  [refactor; survives the H1 refutation]
- Pitch: keep MT matches driving the warp (proven necessary), but make the chain COMMIT on the two-witness score, decoupled from the warp's match cost — so a pair can be warp-trusted yet chain-abstained.
- Mechanism: split the two consumers of `id_pairs`. The warp uses geometric matches (as now); the chain commit uses the per-join two-witness score (C2) + triple consistency (C4). This is the form of "cut the last coupling wire" that survives DD1: full decoupling is impossible, but the *chain decision* can use different evidence than the *warp anchor*. `qc.chainable` is already half-decoupled from `qc.accepted`; this finishes the job at per-join granularity.
- Provenance: contrarian (cut-the-last-wire), DD1 (H1 refutation forces partial-not-full), DD3 (witness independence), codebase (`chainable` half-decoupled, chains break only per-interface today).
- Strongest open question: recall cost — do we break chains the warp legitimately trusted?

---

## How they compose (the recommended path)
C1 first (cheap, decisive, validated) → C2 (precision + abstention, fills the per-join hole) →
C5 (wire C2 into the chain, leaving the warp alone) → C3 only if C1+C2 don't kill enough wrong
joins on dense data → C4 for long-stack global robustness. C1+C2 are the 80/20.

## GT-free evaluation (so the user can rank without labels) — from DD3
- **M1** two-witness agreement rate at the operating point (commit-all dies here).
- **M2** ballistic-residual bimodality: tight-mode fraction × mode separation (random dies here).
- **M3** abstention-calibration slope: triple-consistency vs commit-rate (abstain-all dies here).
