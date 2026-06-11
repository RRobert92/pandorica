# MAP — Tomogram alignment + MT correspondence brainstorm

Single source of truth for what is known. One line per file/topic. Status: open / answered / dead-end.
Entry point: read `brief.md` for the confirmed problem.

## Phase status

- Phase 0 (confirm): **done** — brief.md written, user confirmed "Yes, run it".
- Phase 1 (landscape): **done** — 6 agents returned; ideas.md populated; 8 mechanism clusters.
- Phase 2 (deep dives): **done** — 3 threads returned; central hypothesis (full decoupling) REFUTED, reframed.
- Phase 4 (synthesis): **done** — candidates.md, 5 composing candidates C1–C5.
- Phase 5 (validate): **done** — red-team + assumption-probe; see validation.md.
- Phase 6 (report): **done** — report.md (validation-adjusted ranking).

## FINAL VERDICT (see report.md)
Full decoupling REFUTED (matcher necessary). Validation demoted the two front-runners (ballistic
C1 → WEAK; crossing-QAP C3 → WEAK standalone) and surfaced the missing candidate: a **birth/death /
no-partner model** (MT nucleation/termination presents as a miss and is plausibly the REAL source
of the screenshot wrong-joins). Deepest risk = all confidences share one warped frame → fail
together; only the annular off-MT image witness and the triple-section cycle escape it.
Ranked: (1) no-partner/birth-death abstention, (2) triple-overlap consistency, (3) two-witness w/
annular patch, (4) bi-directional partial decoupling (whirlpool fix), (5) ballistic-as-cost-term
[guarded], (6) crossing-QAP [only inside global flow]. NEXT ACTION = classify existing wrong joins
into swap / birth-death / miss — decides the whole roadmap fork.

## DEEP-DIVE VERDICTS (the reframe)

- **DD1 (empirical): full decoupling REFUTED.** Image coarse leaves 1.18ρ on sec01→02; MT fine
  warp → 0.15ρ (8×); image-fill depends on MT warp. Matcher is NECESSARY. Misses are NOT the
  z-band (loosening = +3..17 pairs) — they're the rigid-residual/smoothness gates rejecting REAL
  pairs displaced by the ~1ρ coarse residual (37%→94% with gates open). Ballistic tangent predicts
  landing to 0.05ρ vs 0.39ρ raw (8×, 98% of MTs) — cheapest highest-leverage fix. → findings/dd-empirical.md
- **DD2 (correspondence): crossing-penalty QAP beats Weber on density.** Weber's smoothness factor
  is symmetric under parallel-neighbour swap → can't disambiguate; a crossing penalty is asymmetric
  → breaks it. Rank: multi-section flow > QAP-crossing > dummy-LAP(layer) > GW(impractical) >
  Weber-MRF. Stack: ballistic→dummy-LAP→QAP→flow. → findings/dd-correspondence.md
- **DD3 (confidence): two-witness + abstain + triple-overlap.** L1 ballistic residual ⊥ L2 image
  continuity at endpoints; commit on score+margin, abstain-as-group within δJ; k,k+1,k+2 cycle is a
  free check. GT-free metrics M1/M2/M3. Per-join signal is the hole (`_confidence` is aggregate). → findings/dd-confidence.md

## Findings index

| File | Covers | Status |
|------|--------|--------|
| findings/lit-landscape.md | Literature families; 5 ideas (Soft-Assign EM, Decoupled, Weber++ MRF, GNN+Sinkhorn, MHT) | answered |
| findings/impl-landscape.md | Tools; best steals: dummy-LAP, pygmtools QAP, POT unbalanced Sinkhorn, micron/mtrack | answered |
| findings/first-principles.md | Decomp: align⊥correspondence (one-way); ballistic gap; z-gate; 5 ideas | answered |
| findings/cross-domain.md | 7 fields; best transfer = generalised-Hough vote alignment; 6 ideas A–F | answered |
| findings/contrarian.md | Reframe: field-first, correspondence-as-readout; abstention-as-product; 4 ideas | answered |
| findings/codebase-archaeology.md | Flow map; failure origins (z-band / TPS-inputs / forced-1:1); reusable assets | answered |

## Landscape verdict (what the board shows)

Strong convergence: (1) DECOUPLE — image aligns, MTs only identify (code already ~80% there);
(2) ABSTAIN — stop forcing 1:1 Hungarian; (3) bundle-level geometric CONSISTENCY beats dense
parallel confusion; (4) the gap is BALLISTIC so the tangent predicts the landing; (5) certify
joins with a SECOND witness (image) independent of the geometry that drove the warp. Prior art =
Weber 2014 (the user's own lineage) which EXPLICITLY fails on dense parallel arrays — the rethink
must beat Weber on density specifically.

## Open questions (refined for Phase 2)

- OQ1 [DD1, empirical]: Does the fine warp hold with ZERO MT matches on the hard Monopoles
  interfaces? (kill-test for full decoupling = Cluster 1). And how many MTs does the z-band gate
  silently drop (= Cluster 8 magnitude)? — answered → see findings/dd-empirical.md
- OQ2 [DD2, lit re-task]: Which correspondence formulation survives DENSE PARALLEL bundles GT-free
  and beats Weber's density failure — Weber++/MRF vs Gromov-Wasserstein OT vs QAP vs min-cost-flow?
  Is micron/mtrack (Funke) directly reusable? — answered → findings/dd-correspondence.md
- OQ3 [DD3, first-principles re-task]: The intrinsic GT-free two-witness confidence model +
  abstention calculus (group-vs-break) + does k,k+1,k+2 triple-overlap give a free consistency
  check? — answered → findings/dd-confidence.md
- OQ4 (open, for synthesis): is "one filament per MT" even the right target, or do high-precision
  fragments serve the biologist better? (decide with user)
- OQ5 (open): Cluster 6 (trace-as-one-volume) is a TARDIS-upstream change — in or out of scope?

## Idea ledger

See ideas.md — 8 mechanism clusters + 1 anti-idea.

## Graveyard

(populated in validation)
