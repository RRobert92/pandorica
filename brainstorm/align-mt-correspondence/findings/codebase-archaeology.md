# Codebase Archaeology — pandorica stitcher

(Written by orchestrator from the read-only Explore agent's returned report.)

## 1. Full data flow (stage : file → function)

| Stage | File | Entry point | Notes |
|---|---|---|---|
| Load | `stitch/dataset.py` | `Dataset.coords_list()` | `[N,4]` `[id,x,y,z]` per section |
| Coarse (MT path) | `stitch/coarse/coarse_hybrid.py` | `hybrid_coarse()` | `global_rotation_search` → CPD/gated-sweep; A-P polarity; ABSTAIN flag |
| Coarse (image path) | `stitch/image_pose.py` | `image_only_poses()` | RANSAC-sweep block-match; absolute per-section poses; or `reconcile_image_mt` |
| Match | `stitch/matching/matcher.py` | `match_sections()` | dedupe → Hungarian → uncross → smoothness → rigid-residual reject; returns `(matches, ref_xy, mov_xy, conf, id_pairs)` |
| Endpoint extraction | `stitch/matching/mt_endpoints.py` | `extract_boundary_endpoints()` | via `scale.boundary_landmarks()` |
| Relative rigid fit | `stitch/matching/mt_transform.py` | `fit_rigid_transform_2d()` | SVD Procrustes, 2-D similarity, **isotropic scale only** |
| Guarded TPS warp | `stitch/transform/warp.py` | `fit_guarded_warp()` | ρ-normalised RBFInterpolator; smoothing ladder; detJ + vorticity guard |
| Field certificate | `stitch/transform/diagnostics.py` | `FieldCertificate.from_field()` | detJ ≥ eps AND ‖curl‖ ≤ omega_max |
| Per-interface QC | `stitch/pipeline/qc.py` | `assess_interface()` | fuses warp cert + matcher conf; `chainable` decoupled from `accepted` |
| Global pose solve | `stitch/transform/solver.py` | `global_pose_refine()` | gauge-anchored weighted LSQ |
| Core orchestrator | `stitch/pipeline/core.py` | `register_section_stack()` | seeds coarse angles; match/warp/qc/solve |
| Fine warp on supplied coarse | `stitch/pipeline/core.py` | `register_warps_to_coarse()` | `_bootstrap_correspondences`; no MT rigid re-fit; returns `id_pairs` for chain |
| MT rotation rescue | `stitch/pipeline/core.py` | `rescue_coarse_poses()` | collapsed-match detect → `global_rotation_search` + re-warp |
| Image fill warp | `stitch/image_warp.py` | `image_residual_warps()` | masked block-match on MT-free regions; `omega_max=0.3` |
| Full orchestrator | `stitch/pipeline/stitcher.py` | `stitch_sections()` | branches on `coarse_poses`; rescue + scale-gate; intensity QC |
| Chain / union-find | `stitch/chain.py` | `chain_filaments()` | unions on `qc.chainable` |
| Chain orient + split | `stitch/chain.py` | `orient_chain_blocks()`, `split_chains_at_joints()` | break kinks by overall-direction gate (45°, XY only) |
| Export | `stitch/stitch.py` | `export_stitched()` | applies poses + warp; calls `chain_filaments` (`cli.py:428`) |

**One match set feeds both warp and chain — CONFIRMED.** `core.py:529` returns `id_pairs`; `cli.py:423` `chain_pairs = [iface.id_pairs ...]`, `cli.py:428` `chain_accepted = [qc.chainable ...]`.

**Decoupling already partial:** `qc.chainable` (`qc.py:149-151`) is decoupled from the warp cert — an interface with a failed warp but coherent matches (fraction ≥ 0.3 AND incoherence ≤ 2.5ρ) still chains; the warp-rejected volume falls back to coarse. But the **same match set still drives both**, so a bad match still poisons both.

## 2. Endpoint extraction — MISSES suspect

`mt_endpoints.py:23-96` `extract_boundary_endpoints()`. Groups by MT id; picks endpoint Z-closest to the face; **z-band gate `z_band_fraction=0.15`** → only MTs whose endpoint is in the outermost 15% of the section Z-range are returned. Tangent = first→last chord over outermost `max(3, 20%)` points.
**MISS origin:** the z-band gate is HARD and has no fallback — an MT that truly continues but ends slightly inside the band is **never presented to the matcher**, so no matcher improvement can recover it. Secondary: coarse first→last tangent on short/curved MTs; near-vertical sign-gate vetoes.

## 3. The warp — WHIRLPOOLS suspect

`warp.py:135-235` `fit_guarded_warp()`: `RBFInterpolator(kernel="thin_plate_spline")`, inputs ρ-normalised+centred, smoothing ladder `(0,1,5,20,100)` in ρ-units, first rung passing the cert wins. Guards: **detJ ≥ 0.05 AND |curl| ≤ 1.0** on a 48×48 grid; if none pass, `accepted=False`, warp **never applied**. Optional `_tangent_augment` (default off).
**WHIRLPOOL origin:** the guard DETECTS and REJECTS swirls (accepted warps are certified diffeomorphic). Upstream cause = bad input correspondences: near-coincident/duplicate endpoints (underdetermined TPS → curl spikes; `dedupe_endpoints` `dup_frac=0.1ρ` is first defence — docstring names duplicates as the "Amira whirlpool trigger") and spatially isolated outlier matches surviving `reject_outliers` (2ρ rigid-residual gate) that make the TPS pull a tent. Image-fill warp uses tighter `omega_max=0.3`.

## 4. Matcher gates — MISSES vs WRONG JOINS

Pipeline: dedupe → cost matrix → Hungarian → uncross → smoothness → rigid-residual reject.
Gates & defaults: distance `clip(5ρ, 500Å, 2500Å)`; angle 30° (sign-agnostic); signed orientation `sign_min_cos=0` (fold-back reject); cost `0.7·dist + 0.3·dir`; dedupe `0.1ρ`; 1:1 Hungarian; uncross `2ρ` radius, `0.2` margin; smoothness tangent–tangent 45°, chord–tangent 60°, chord-check ≥1ρ; outlier reject 2ρ×2; near-vertical jog cut 2ρ (`core.py:256-282`).
**MISS gates:** z-band (upstream), distance gate (kills offset-but-real pairs before warp converges), chord-tangent 60° (kills laterally-shifted real continuations), outlier reject (ejects good pairs when few matches).
**WRONG-JOIN gates:** 30° angle gate too loose for parallel different-MTs at close range; uncross can't disambiguate parallel stubs by direction; backstop is `split_chains_at_joints` (45° XY) which is near-vertical-blind.

## 5. QC + chain — abstention

`assess_interface`: `accepted = warp.cert.passed AND match_fraction ≥ 0.3 AND incoherence ≤ 2.5ρ`. `chainable = match_fraction ≥ 0.3 AND incoherence ≤ 2.5ρ` (decoupled, `qc.py:149-151`). `tangent_discontinuity_deg` computed but NOT gated. Chain breaking: `split_chains_at_joints` (45° overall-direction, XY); `orient_chain_blocks`; `compute_chain_labels` for diagnostics. Abstention already exists: warp-reject → coarse fallback; `chainable=False` → chain breaks; `_failed_interface`/`_warpless_interface`; rescue/scale-gate margins; intensity QC.

## 6. Image path — how much alignment is already image-driven

`image_only_poses()` (`image_pose.py:477-687`): MIP faces; two-stage ±180° rotation sweep (RANSAC support ranking → full-res refine, resolves 180° branch); translation from RANSAC rigid; affine anisotropy via small-window block-match + RANSAC, committed only if stretch > 2% (`_ANISO_GATE=0.02`, SV∈[0.85,1.15], det∈[0.90,1.10]); contour cross-check; abstain below 12-cell/12%-inlier gate. `reconcile_image_mt()` cross-checks MT vs image poses.
**In the production coarse→fine path, global translation+rotation+anisotropic scale ALL come from `image_only_poses`; MTs only drive the fine residual TPS warp + chaining.** Alignment is **already substantially image-driven**. (MT-only path uses image only for A-P polarity, intensity verification, reconcile.)

## 7. Half-built / reusable / abandoned

- `tmp/pose_rethink/01-04_*.md` — Fourier-Mellin rotation redesign (log-polar magnitude phase-correlation, permutation-null confidence, cross-pair flip test). **Design only, not implemented.** Translation-invariant rotation; block-match demoted to verifier+warp source.
- `tmp/proto_selective_chain.py` — **per-pair selective chaining** prototype: skip only outlier pairs vs their k-NN neighbourhood, not whole interfaces. Half-built; not integrated into `chain_filaments`.
- `_bootstrap_correspondences()` (`core.py:483-522`) — production iterative warp→re-match loop that recovers misaligned-but-real correspondences without loosening gates. Reusable.
- `tmp/validate_decouple.py` — validated the `chainable` decoupling on Monopoles (fix already in prod).
- `stitch/coarse/cpd.py` `cpd_rotation_search()` — multi-seed decoy-robust CPD rotation, default, already ~independent of endpoint-match quality.
- `tmp/diag_*.py` / `tmp/validate_*.py` — diagnostics: wrong connections, warp curl, connection quality, anisotropy, coarse stability, matcher gate contributions.
- `tmp/flip_apparatus/` — abandoned FM flip detection (reverted, too unreliable). `tmp/knowledge/` — FM/log-polar math foundation.

## Failure origins (one line each)
- **MISSES** → `mt_endpoints.py:53-80` z-band 15% hard gate drops MTs before the matcher; + distance gate centred on un-jumped endpoint.
- **WHIRLPOOLS** → `warp.py` guard rejects them, but duplicate/isolated-outlier correspondences feed the TPS; decouple warp from matches to kill at root.
- **WRONG JOINS** → forced 1:1 Hungarian + loose 30° gate let close-parallel different-MTs through; `split_chains_at_joints` post-hoc backstop is near-vertical-blind.

**Biggest reusable asset for a rethink:** alignment is already image-driven (`image_only_poses`); the only remaining poison is the fine warp + chain sharing one `id_pairs`. `_bootstrap_correspondences()` + the FM redesign blueprint + `proto_selective_chain.py` are the building blocks.
