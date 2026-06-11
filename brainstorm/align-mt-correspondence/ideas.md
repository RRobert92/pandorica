# Idea ledger

All ideas from the Phase 1 landscape, grouped by underlying MECHANISM (not by source angle).
Full idea blocks live in the per-angle findings files; this is the index + clustering.

## Cluster 1 — Decouple by modality: IMAGE aligns, MTs only identify
The single most convergent theme (5 of 6 angles, and the code is ~80% there already).
- contrarian: *Field-first alignment, correspondence as a readout* (findings/contrarian.md:78)
- first-principles: *Decouple-by-modality* (first-principles.md:215)
- lit: *Idea B — Decoupled Two-Signal Architecture* (lit-landscape.md:247)
- impl: *Idea D — Soft OT alignment + hard correspondence on aligned data* (impl-landscape.md:281)
- codebase: alignment already from `image_only_poses`; only the fine warp + chain still share `id_pairs`.

## Cluster 2 — Soft/probabilistic assignment with HONEST ABSTENTION (don't force 1:1)
- contrarian: *Abstention as the product* (contrarian.md:96)
- first-principles: *Group-when-aliased, abstain-honestly* (first-principles.md:248)
- impl: *Idea A — Dummy-augmented LAP* (impl-landscape.md:257); *Idea E — Min-cost network flow* (289)
- cross-domain: *Idea E — Min-cost-flow with abstention* (cross-domain.md:450); MHT; JPDA
- lit: *Idea A — Soft-Assign EM (CPD-style)* (lit-landscape.md:240); *Idea E — MHT defer decisions* (268)
- learned matchers: SuperGlue/LightGlue dustbin (Sinkhorn + no-match bin)

## Cluster 3 — Geometric CONSISTENCY of the whole bundle (QAP / MRF / OT-structure), not lone endpoints
This is the one that beats DENSE PARALLEL neighbour confusion.
- impl: *Idea B — Pairwise-coherence QAP* (impl-landscape.md:265)
- lit: *Idea C — Pair-Consistency MRF (Weber++)* (lit-landscape.md:254); Gromov-Wasserstein OT
- cross-domain: *Idea D — Local bundle topology descriptor* (cross-domain.md:435); *Idea A — pairwise-transform voting* (392); generalised Hough + consistency graph (120)
- first-principles: *Trajectory + bundle descriptor* (first-principles.md:238)

## Cluster 4 — BALLISTIC tangent-predicted matching (the gap is ballistic; tangent predicts the landing)
Cheap, high-value, under-used signal already sitting in `dir`.
- first-principles: *Ballistic-jump matching* (first-principles.md:228)
- cross-domain: combinatorial Kalman / Hough seeding (cross-domain.md:86, 68)

## Cluster 5 — TWO-WITNESS confidence: geometry AND image must independently agree on a join
Breaks the last coupling wire — chains certified by DIFFERENT evidence than the warp used.
- first-principles: *Two-witness join confidence* (first-principles.md:259)
- cross-domain: *Idea C — Grayscale patch cross-check* (cross-domain.md:419)
- contrarian: *Cut the last coupling wire: chains and warp must use DIFFERENT evidence* (contrarian.md:131)

## Cluster 6 — DELETE the problem: trace the stack as ONE volume, never split into sections
Radical; lives upstream in TARDIS tracing, likely out of pandorica scope. Keep for graveyard/note.
- contrarian: *Delete the section boundary* (contrarian.md:111)

## Cluster 7 — GLOBAL multi-section flow (solve the whole stack at once, not pairwise)
- impl: *Idea E — Min-cost network flow for chain building* (Ultrack/micron) (impl-landscape.md:289)
- cross-domain: combinatorial Kalman across layers; min-cost-flow (cross-domain.md:86, 450)
- Directly relevant tool: **micron / mtrack (Funke lab, MIT)** — min-cost-flow MT tracking in EM (impl-landscape.md:210)

## Cluster 8 — Fix the MISSES at the source (not an "idea", a defect)
- codebase: `z_band_fraction=0.15` hard gate drops MTs before the matcher; tangent estimate crude.

## Anti-idea (named to kill it)
- contrarian: *keep buffing the Hungarian* — 7 correction layers already stacked on a 1:1 assignment is the tell that the base operation is wrong (contrarian.md:147).
