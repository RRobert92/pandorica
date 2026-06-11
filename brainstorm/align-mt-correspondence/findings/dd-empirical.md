# Empirical Validation — alignment / MT correspondence decoupling
Date: 2026-06-11
Dataset: Monopoles_test (11 sections, sec01→sec02 = hard interface)
Code: pandorica v1.1.6, image coarse at downscale=4

---

## H1: Full decoupling holds — the fine warp does NOT need MT matches

**Claim:** if you run the fine warp with zero MT correspondences (image-only / coarse-only), alignment
residual and foldover stay acceptable.

### Measurements

After image-only coarse (no MT fine warp), the residual displacement between matched MT endpoint pairs:

| Interface | n_pairs | median_disp | max_disp | median_rho | max_rho |
|---|---|---|---|---|---|
| sec05→sec06 | 590 | 914 Å | 2394 Å | **1.40 rho** | 3.67 rho |
| sec06→sec07 | 941 | 279 Å | 1519 Å | 0.41 rho | 2.20 rho |
| sec07→sec08 | 934 | 304 Å | 1596 Å | 0.44 rho | 2.31 rho |
| sec08→sec09 | 828 | 554 Å | 2425 Å | 0.80 rho | 3.49 rho |
| sec09→sec10 | 981 | 327 Å | 1539 Å | 0.48 rho | 2.28 rho |
| sec10→sec11 | 971 | 308 Å | 1481 Å | 0.44 rho | 2.13 rho |
| sec11→sec12 | 979 | 227 Å | 1580 Å | 0.31 rho | 2.14 rho |
| sec12→sec13 | 886 | 535 Å | 1906 Å | 0.70 rho | 2.48 rho |
| sec13→sec01 | 768 | 632 Å | 1985 Å | 0.84 rho | 2.65 rho |
| **sec01→sec02** | 415 | **882 Å** | 2025 Å | **1.18 rho** | **2.72 rho** |

MT TPS warp effect on sec01→sec02 (directly measured):
- After image coarse only: median residual = **882 Å (1.18 rho)**
- After MT fine TPS warp (pass 1): median residual = **108 Å (0.15 rho)** → 8x improvement
- The warp was accepted (detJ_min=0.40, curl=0.78, smoothing=1) — no foldover

Bootstrap gain (iterative re-warp, 5 passes):
| Interface | Pass 1 | Pass 5 | Gain |
|---|---|---|---|
| sec01→sec02 | 37% | 56% | **+20pp** |
| sec10→sec11 | 80% | 81% | +1pp |
| sec12→sec13 | 75% | 82% | +7pp |

From cf_smoke logs (C.elegans dataset): image_fill=False → match 29–37%; image_fill=True → 60–68%. The image fill warp requires the MT TPS warp as input (it masks MT regions and fills the gap), so the MT warp is upstream of the fill, not replaceable by it.

### Verdict: **REFUTED**

Full decoupling does NOT hold. The image coarse at operational downscale leaves 1.18 rho median
residual on the hard interface (sec01→sec02) and 0.4–1.4 rho across normal interfaces. The MT fine
TPS warp reduces this 8x to 0.15 rho. The image fill warp (`image_residual_warps`) depends on the
MT warp upstream and cannot substitute for it. 

Caveat: residual at healthy interfaces (sec10–sec11: 0.44 rho) is within the distance gate radius
(5 rho), so chaining could survive without fine warp there — but alignment precision would degrade.
The hard interface (sec01→sec02) requires the MT fine warp for both correct alignment and chaining.

---

## H2: z-band gate causes misses

**Claim:** the `z_band_fraction=0.15` gate in `mt_endpoints.py` silently drops MTs before the matcher.
Loosening the band should recover missed continuations.

### Measurements

**Dropout rate (all Monopoles interfaces):**

| Interface | Total MTs | @15% band | Dropped | Drop% | @30% | Extra | @50% | Extra |
|---|---|---|---|---|---|---|---|---|
| sec05→sec06 | 1387 | 1171 | 216 | **16%** | 1210 | +39 | 1286 | +115 |
| sec06→sec07 | 1429 | 1219 | 210 | 15% | 1254 | +35 | 1322 | +103 |
| sec07→sec08 | 1508 | 1194 | 314 | **21%** | 1257 | +63 | 1358 | +164 |
| sec08→sec09 | 1500 | 1209 | 291 | 19% | 1277 | +68 | 1377 | +168 |
| sec09→sec10 | 1568 | 1267 | 301 | 19% | 1325 | +58 | 1412 | +145 |
| sec10→sec11 | 1602 | 1263 | 339 | **21%** | 1356 | +93 | 1437 | +174 |
| sec11→sec12 | 1515 | 1230 | 285 | 19% | 1280 | +50 | 1368 | +138 |
| sec12→sec13 | 1516 | 1240 | 276 | 18% | 1308 | +68 | 1401 | +161 |
| sec13→sec01 | 1475 | 1224 | 251 | 17% | 1281 | +57 | 1353 | +129 |
| **sec01→sec02** | 1398 | 1150 (ref) | 248 | **18%** | 1212 | +62 | 1326 | +176 |

**Character of dropped MTs (sec01, top face):**

| Top-Z fraction (of Z-range) | N dropped | Median Z-span |
|---|---|---|
| 0.00 – 0.50 | 107 | 463 Å |
| 0.50 – 0.70 | 79 | 1080 Å |
| 0.70 – 0.80 | 34 | 1432 Å |
| 0.80 – 0.85 | **28** | **1582 Å** |
| In-band (≥0.85) | 1150 | 1852 Å |

The 107 MTs with top-Z < 50% are almost certainly not crossing the gap (median span 463 Å, half the
total section depth). The 28 near-band MTs (top-Z in 0.80–0.85 range, median span 1582 Å, within
~50 Å of the threshold) are plausible misses. In-band MTs have median span 1852 Å.

**Does widening the band recover new matched pairs?**

Matched pair counts when z-band is widened (absolute pairs, not fraction):

| Interface | @15%: pool/matched | @30%: pool/matched | @50%: pool/matched | +matches(30vs15) |
|---|---|---|---|---|
| sec01→sec02 | 1000 / 232 | 1087 / 235 | 1156 / 248 | **+3** |
| sec12→sec13 | 1188 / 278 | 1251 / 270 | 1339 / 274 | **-8** |
| sec10→sec11 | 1183 / 258 | 1250 / 261 | 1349 / 269 | **+3** |
| sec08→sec09 | 1209 / 288 | 1277 / 301 | 1377 / 312 | **+13** |
| sec05→sec06 | 1167 / 288 | 1210 / 305 | 1286 / 332 | **+17** |

### Verdict: **PARTIALLY SUPPORTED, but weak**

The z-band gate does drop 15–21% of MTs across all interfaces (measured). However, widening the
band from 15% to 30% recovers only +3 to +17 additional matched pairs per interface — negligible
relative to ~250–970 existing matches. On sec01→sec02 specifically, widening adds only 3 pairs.
The gate is not a primary driver of misses.

Key insight on dropout character: the majority of dropped MTs (107/248 on sec01) end deep inside
the section (top-Z < 50%), suggesting they genuinely do not reach the face. Only 28 MTs are within
50 Å of the 15% threshold and could be borderline misses.

**The gate does NOT explain the low match fraction on sec01→sec02 (37%).** The gate relaxation
experiment (diag_which_gate logic) shows the match fraction can reach 94% on that interface when ALL
gates are blown open — so the problem is the quality gates (rigid-residual, smoothness), not the
z-band pre-filter.

Gate relaxation results for sec01→sec02:
- Baseline: **37%**
- dist+angle only: 33% (no gain, distance gate not the cause)
- resid_rho(2→12): 54% (+17pp — residual outlier gate is primary culprit)
- smooth_off: 48% (+11pp — smoothness gate secondary)
- all loose: **94%** (the matches exist, the quality gates drop them)

This means: ~57pp of sec01→sec02 matches exist but are rejected by quality gates because the
spatially-varying residual (1.18 rho median before TPS warp) places real pairs outside the 2-rho
rigid-residual gate. Loosening the z-band adds no value here.

---

## H3: Ballistic tangent — tangent-predicted landing is closer than raw endpoint

**Claim:** endpoint + tangent * gap lands closer to the true partner than the raw endpoint.

### Measurements

**Intra-section geometry test** (bisect each section at 50% Z; predict bottom-half XY from
top-half tangent; compare to ground-truth cross-bisect position):

| Section | N pairs tested | rho | Raw XY error (median) | Tangent-predicted (median) | Fraction improved | Gain |
|---|---|---|---|---|---|---|
| sec01 | 494 | 726 Å | 285 Å (0.39 rho) | **36 Å (0.05 rho)** | **98%** | **8x** |
| sec12 | 488 | 721 Å | 310 Å (0.43 rho) | **51 Å (0.07 rho)** | **98%** | **6x** |
| sec10 | 483 | 656 Å | 310 Å (0.47 rho) | **44 Å (0.07 rho)** | **97%** | **7x** |

For the large-error subset (raw error > 1 rho): tangent prediction helps 100% of those cases,
reducing median raw 1077 Å (1.49 rho) → predicted 361 Å (0.50 rho).

**Cross-gap test (no alignment):**
Without alignment the cross-gap test is noisy (sections are in independent coordinate systems). The
result was ~50% improvement rate with near-zero median gain — uninformative without alignment.

**Tangent agreement in matched pairs (hard vs healthy):**

| Interface | n pairs | |d_ref · d_mov| median | >0.90 fraction | <0.70 fraction |
|---|---|---|---|---|---|
| sec01→sec02 (hard) | 415 | 0.960 | 74% | **11%** |
| sec10→sec11 (healthy) | 971 | 0.966 | 92% | 2% |
| sec12→sec13 (weak) | 886 | 0.988 | 93% | 2% |

On the hard interface, 11% of matched pairs have |d · d| < 0.70 (direction disagreement > 45°)
versus only 2% on healthy interfaces. This suggests wrong joins are more prevalent on sec01→sec02.

### Verdict: **STRONGLY SUPPORTED** (within sections; cross-gap needs aligned frames)

Within a section, the tangent predicts the continuation point to within **0.05–0.07 rho** versus
0.39–0.47 rho for the raw endpoint. This is a 7–8x improvement and applies to 97–98% of MTs.

The implication for the brainstorm: rather than matching raw endpoints across the gap, a tangent-
extrapolated anchor (shifted by gap_z / dz * dx, dy) should be used as the query position. This
would effectively pre-correct for the geometric XY offset introduced by the section gap, tightening
the effective distance the matcher needs to span and reducing the dependence on accurate coarse
alignment.

The current pipeline does not do this: `match_sections` matches raw endpoints after coarse-pose
application. The `_tangent_augment` term in `warp.py` pulls tangent continuity post-match (fit
only), but does not shift the query point. A tangent-shifted query is a different mechanism that
could be added upstream of the distance gate.

---

## Summary of measured numbers

| Hypothesis | Key number | Verdict |
|---|---|---|
| **H1 (decouple)** | Image coarse leaves 1.18 rho median residual; MT TPS warp reduces it 8x to 0.15 rho on hard interface | **REFUTED** — MT fine warp is necessary for alignment |
| **H2 (z-band gate)** | z-band drops 15–21% of MTs; widening 15→30% recovers +3 to +17 pairs; quality gates (not z-band) explain sec01→sec02's 37% match (94% with all gates open) | **WEAK** — gate exists, not primary miss driver |
| **H3 (ballistic tangent)** | Within-section: tangent predicts partner position to 0.05 rho vs 0.39 rho raw (8x), 98% improvement rate | **STRONGLY SUPPORTED** — tangent-shifted query is a real lever |

**Single most important number:** On sec01→sec02, gate-relaxation recovers 94% match (vs 37%
baseline) — proving the matches physically exist and the problem is quality-gate placement, not
absence of continuations or z-band dropout. The residual gate (2 rho → 12 rho) alone recovers +17pp.
This failure mode is created by the 1.18 rho post-coarse residual that the tight gates then reject.
