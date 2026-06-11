# First-Principles Decomposition — Align + MT Correspondence

Grounding note: confirmed against the live code. `match_sections`
(`pandorica/stitch/matching/matcher.py`) returns `(matches, ref_xy, mov_xy, conf,
id_pairs)`. `ref_xy/mov_xy` → `fit_rigid_transform_2d` → pose → warp; `id_pairs` →
union-find chain (`chain.py`). The endpoint object is `{id, pos(3), dir(3)}` — ONE
point + ONE local tangent per MT face (`mt_endpoints.py`). `_bootstrap_correspondences`
in `pipeline/core.py` already runs an iterative re-warp-then-rematch loop, which is the
codebase tacitly admitting the chicken-and-egg.

---

## 1. What does ALIGNMENT require vs. what does CORRESPONDENCE require?

**Alignment** needs a *low-dimensional field over the whole plane*: given a point
`(x,y)` on face A, where is the same physical material on face B? It needs **coverage and
smoothness**, not per-MT identity. A small number of *correct anchors spread across the
field* fully determines a global affine (3–4 points) or a coarse warp (tens of points,
TPS/B-spline). Alignment is **tolerant to missing MTs** (it interpolates), **tolerant to
1-to-many ambiguity** (any consistent representative suffices), but **intolerant to wrong
anchors** (one outlier bends the field — the whirlpool).

**Correspondence** needs the *exact per-object pairing*: this spline → that spline. It
needs **precision per item**, must say "no partner" honestly, and is **intolerant to
density ambiguity** (parallel neighbours within the jump radius are individually
indistinguishable even when the field is perfect).

**Same or different information?** Different, and asymmetrically dependent:

```
   ALIGNMENT  ──needs──>  SPARSE, ROBUST, SMOOTH, OUTLIER-FREE anchor field
                          (a handful of confident pairs + image block-matches)

   CORRESPONDENCE ─needs─> DENSE, PRECISE, PER-OBJECT identity
                          (every MT, with honest nulls)
```

The brief's whole disease is forcing these two needs through **one** Hungarian matrix.
Alignment wants to *throw away* ambiguous MTs (keep only the confident, well-spread few);
correspondence wants to *resolve* them. The matcher cannot do both: its outlier-rejection
(good for alignment) silently *deletes correspondences* (bad for recall); its 1:1
assignment (needed for chains) over-commits ambiguous pairs into the warp (bad for
alignment → whirlpool).

**Can one be solved without the other?**
- **Alignment without correspondence: YES, and this is the key unlock.** The EM grayscale
  volume is a *dense, identity-free* signal. Cross-correlation / block-matching of the two
  image faces gives a displacement field with NO need to know which MT is which. MTs as
  *texture* (density of tubes, local orientation field) also align without identity. The
  current code already block-matches in MT-free regions — it just doesn't let image alone
  carry the alignment where MTs are ambiguous.
- **Correspondence without alignment: NO, not robustly.** Identity requires the two faces
  in a common frame to within ~one MT spacing. Below that, you cannot tell neighbour from
  continuation. So the *honest* dependency is **one-directional**: solve alignment FIRST
  (image-led, identity-free), THEN solve correspondence in the aligned frame. The current
  bidirectional coupling is an artifact of using endpoints for both.

---

## 2. Where does the real difficulty actually live?

Ranked by how much each *causes* the observed MISS / WHIRLPOOL / WRONG-JOIN failures:

1. **The coupling, via shared 1:1 assignment** — *root cause of whirlpools and the
   chicken-and-egg.* It forces alignment to inherit correspondence's per-item commitment.
   A single wrong pair is simultaneously a chain error AND a warp anchor that the TPS bends
   toward. This is the highest-leverage thing to break.
2. **Density ambiguity is the irreducible floor for correspondence** (see §3). Once
   alignment is clean, *wrong-joins and some misses are information-limited*, not
   algorithm-limited — no matcher can beat the lateral-jump-vs-spacing ratio. This caps
   achievable recall/precision and is exactly what *honest abstention* must encode.
3. **The endpoint representation is lossy** — *cause of avoidable wrong-joins.* One point +
   one short tangent discards the part of the spline most diagnostic of identity: its
   *parallax history* (where it came from over the last few hundred nm) and the local
   bundle it travels in. Two parallel neighbours have near-identical (pos, tangent) but
   different multi-hundred-nm trajectories and different neighbour-sets.
4. **The missing-material gap** (§3) — *cause of some misses being real breaks.* The knife
   removes a slab; an MT genuinely terminates inside it for some fraction of cases. A miss
   is not always an error. The model has no explicit gap, so it cannot distinguish "I
   failed to join" from "there is nothing to join."
5. **Alignment itself is the *least* hard part** once it's image-led. Knife compression is
   ~global-affine, baking is smooth-spatially-varying; both are *low-frequency* and
   estimable from image texture without MT identity.

**Single deepest difficulty:** the coupling is the *engineering* villain, but underneath it
the **irreducible** difficulty is *correspondence under density ambiguity across a
material-destroying gap* — because that is the one thing no amount of decoupling can fully
remove. Decoupling converts an algorithm bug (whirlpool) into a clean abstention; it cannot
manufacture information the cut destroyed.

---

## 3. Minimal model of the gap, and the information bound

**Physical picture.** Sectioning removes a slab of thickness `g` (lost to the knife kerf +
the part of the section not imaged). An MT exits face A at `(x_A, y_A, z=A)` with 3D tangent
`t`, travels (unseen) through the slab, and re-enters face B at `(x_B, y_B)`. If MTs were
straight through the gap, the *predicted* re-entry is a pure ballistic extrapolation:

```
   (x_B, y_B) = (x_A, y_A) + (g / t_z) * (t_x, t_y)     [the parallax jump]
```

**Minimal model = ballistic continuation + bounded curvature.** Two parameters:
- `g`/`t_z`-driven **expected lateral jump** `J ≈ g * tan(θ)` where θ is the tilt of the MT
  from the section normal. For near-vertical MTs (θ→0) J→0; for shallow MTs J grows fast.
- a **curvature/wander bound** `σ_J`: the MT may bend within the gap. Persistence length of
  a microtubule is ~mm — *enormous* relative to a section (tens of nm) — so over a gap the
  MT is essentially **straight**: σ_J is small and curvature-driven jump uncertainty is
  sub-spacing. **Falsifiable claim:** the residual scatter of *verified* continuations
  around the ballistic prediction should be << median MT spacing ρ. If it isn't, either the
  alignment is still wrong or g is being underestimated.

**Implication: the jump is *predictable from the tangent*, and that prediction is the
single most under-used signal.** The current matcher uses `dir` only as an angular gate
(`|cosΔθ|`) and a sideways-offset chord gate — it never *uses the tangent to predict the
displaced position*. It should: the partner of A is expected at A + ballistic jump, and the
cost should be the residual to *that* point, not to A itself. This alone removes much of the
distance-gate's neighbour confusion.

**Information-theoretic limit on two parallel neighbours.** Take two parallel MTs separated
laterally by `d`, both tilted so each jumps `J`. After alignment the residual localisation
noise per endpoint is `ε` (tracing + warp residual). Resolving which B-stub continues which
A-stub is a 2-hypothesis discrimination; the log-likelihood-ratio separation scales as
`(d/ε)`. The crossover where the pairing becomes a coin-flip is:

```
   d  <  ~ε·√2     →  individually UNRESOLVABLE (must abstain or pair as a group)
   d  ≳  several·ε →  resolvable
```

But there is a *second*, sharper bound from the jump itself: if the **jump uncertainty**
(scatter of where a single MT lands) is `δJ`, then **two neighbours are unresolvable
whenever `d ≲ δJ`** — the displacement clouds overlap. Since `δJ` grows with tilt
(`δJ ≈ g·sec²θ·δθ` for tangent-estimation error `δθ`), **shallow, tilted, tightly-packed
bundles are the provably-hard regime**, and *near-vertical bundles are easy* (J≈0, clouds
collapse onto the origin → distance alone resolves them). **Falsifiable:** wrong-join rate
should correlate with `tan(θ)·(δθ)/d`, not with raw density. A dense but *vertical* bundle
should chain cleanly; a sparse but *shallow* one should fail.

**Consequence for representation:** when `d < δJ` you must stop pretending the matchable
object is a single MT. The resolvable unit becomes the **bundle/group** (match group↔group,
distribute identity by within-group order), or you **abstain per-MT but still align** using
the group centroid. This is the formal justification for *honest abstention*: it is not
timidity, it is reporting that `d < δJ`.

---

## 4. Conventional assumptions of the current design that deserve doubt

| Assumption (in code) | Doubt | Verdict |
|---|---|---|
| **1:1 Hungarian per interface** | MTs branch, terminate in-gap, fork at the knife; many faces have *no* true partner. 1:1 forces a partner where none exists (wrong-join) or deletes a real one to balance the assignment. | **Doubt hard.** Replace with *0/1-to-0/1 with explicit null*, or many-to-one for merging bundles. |
| **Endpoint (1 pt + short tangent) is the matchable object** | Throws away the most identity-rich part of the spline (its trajectory + bundle context). Two neighbours are aliased. | **Doubt hard.** Match on a *trajectory descriptor* (last N points as a curve, predicted forward, + local-neighbour fingerprint), not a point. |
| **One assignment serves BOTH align and chain** | The core disease. Align wants sparse-robust-smooth; chain wants dense-precise-honest. | **Doubt hardest — this is the thing to break.** Two different estimators consuming two different (overlapping) evidence sets. |
| **Alignment driven by MT correspondences** | The dense identity-free image signal is the natural alignment driver; MT pairs should *refine*, not *seed*. Current code seeds with MTs and patches MT-free gaps with image — backwards. | **Doubt.** Invert: image-led coarse field FIRST, MTs refine where confident. |
| **Rigid + isotropic-scale fit (`fit_rigid_transform_2d`), then TPS** | Knife compression is *anisotropic* (one axis); isotropic scale mis-models it and dumps the residual into the warp, which can fold. The memory notes anisotropic sx,sy is the target — code still fits isotropic. | **Doubt.** Fit anisotropic affine (the slab physics says so) before any free-form warp. |
| **Distance gate centred on the un-jumped endpoint** | The true partner is at A + ballistic jump, not at A. Centring the gate on A both misses tilted MTs and admits the wrong (closer, un-jumped) neighbour. | **Doubt.** Centre the gate/cost on the *tangent-predicted* position. |
| **Outlier rejection = deletion** | Good for the warp, *silently destroys recall* for the chain (a dropped pair is an unrecorded correspondence). | **Doubt.** "Outlier for the warp" ≠ "wrong correspondence." Separate the two consumers (see §1). |

**Most-doubtable single assumption:** *one Hungarian assignment serving both alignment and
chain.* Everything else is downstream of it. Break it and the warp stops being poisoned by
chain commitments, and the chain stops being thinned by warp outlier-rejection.

---

## 5. Cheapest INTRINSIC (GT-free) signal separating right-join from wrong, and real-miss
   from real-break

Three cheap, falsifiable intrinsic signals, none needing labels:

**(a) Ballistic residual + curvature continuity — separates right join from wrong join.**
After alignment, a *correct* join's B-endpoint sits within `δJ` of the tangent-predicted
landing point AND the *concatenated* spline (A-trajectory + B-trajectory) has curvature
within the MT bound (no kink). A *wrong* join (two different MTs) shows a kink or a residual
≫ δJ. Cost = ballistic residual + curvature jump at the splice. **Kill-test:** the residual
distribution for matched pairs should be *bimodal* (a tight ballistic mode = true joins, a
broad mode = forced wrong joins) — if it's unimodal, this signal carries no information here.

**(b) Mutual-nearest + cross-image-patch agreement — confirms a join with a second
independent modality.** A join is trustworthy when (i) A→B is also B→A (mutual NN in the
predicted-jump metric) AND (ii) the *grayscale patch* around A's exit correlates with the
patch around B's entry above the local background-patch-correlation. The image is a *fully
independent* witness to the geometric match; agreement of two independent signals is the
cheapest GT-free confidence. **Kill-test:** patch correlation at true joins must exceed
correlation at random aligned-neighbour patches by a margin > noise; if MT patches all look
alike (featureless tubes), modality (ii) is uninformative and only (i) survives.

**(c) Local-neighbourhood consistency (the bundle fingerprint) — distinguishes real miss
from real break, and resolves the d<δJ aliasing.** An MT travels with companions. Encode
each endpoint by the *relative arrangement of its k nearest MT neighbours* (a small
rotation-invariant descriptor of the local bundle). A true continuation preserves its
neighbourhood across the gap; a wrong join lands it among strangers.
- If a stub has NO plausible partner AND its neighbours *all* found partners → it is a
  **real break/termination** (it ended in the gap; everyone else continued). Honest miss.
- If a stub has no partner AND its whole neighbourhood also lost partners → likely a **real
  miss caused by local alignment failure or a tracing dropout**, flag the *region*, don't
  force joins.
- If `d < δJ` (aliased neighbours): the descriptor matches *the group*, so commit the group
  correspondence (align) and abstain on within-group identity (chain) — exactly the §3
  bound, made operational.
**Kill-test:** randomly delete a known-continuing MT and check the neighbourhood descriptor
of its (now partnerless) neighbours still finds *their* partners — if deleting one MT
destabilises its neighbours' descriptors, the fingerprint is too brittle to localise breaks.

**Cheapest of all:** signal (a)'s *ballistic residual* — it's one tangent extrapolation and
one subtraction per candidate pair, reuses `dir` already computed, and directly attacks the
neighbour-confusion that causes wrong-joins. It is the minimal change with the most leverage
if a full rethink is not on the table.

---

## Concrete ideas

### Decouple-by-modality (image aligns, MTs identify)
- **Pitch:** Two separate estimators — image block-match drives the alignment field;
  MT correspondence runs *in the aligned frame* and never feeds the warp.
- **Mechanism:** Coarse anisotropic-affine + smooth warp from dense grayscale
  cross-correlation (identity-free); MT pairs only *refine* the warp where image is
  textureless, and even then via a robust influence cap so no single pair can fold it.
- **Inspiration:** §1 asymmetry — alignment is solvable without identity, identity is not
  solvable without alignment.
- **Key assumption:** the EM faces share enough low-frequency texture to register without
  MTs (true in MT-free stroma; risky in fully MT-packed fields).
- **Kill-test:** on a known-aligned pair, image-only field residual vs. MT-driven field
  residual — if image-only is not within tracing noise of MT-driven, image cannot carry it.

### Ballistic-jump matching (predict where the MT lands)
- **Pitch:** Match on the tangent-extrapolated landing point, not the raw endpoint.
- **Mechanism:** cost = ‖B − (A + g·tanθ·t̂_xy)‖ + curvature-splice penalty; the distance
  gate recenters on the prediction, killing the un-jumped-neighbour false match.
- **Inspiration:** §3 — MT persistence length ≫ section, so the gap is ballistic.
- **Key assumption:** gap thickness g (or g/t_z effectively) is estimable globally per
  interface (it is — it's the median jump of confident vertical-vs-tilted pairs).
- **Kill-test:** verified-join residuals around the ballistic prediction must be << ρ; if
  not, MTs are not straight across the gap (or g is wrong).

### Trajectory + bundle descriptor (stop matching points)
- **Pitch:** Replace the (pos,dir) endpoint with a short trajectory curve + a
  rotation-invariant neighbourhood fingerprint.
- **Mechanism:** descriptor distance fuses curve-shape match and neighbour-arrangement
  match; aliased parallel neighbours separate by neighbourhood even when (pos,dir) coincide.
- **Inspiration:** §3 d<δJ bound — when points alias, context disambiguates.
- **Key assumption:** the local bundle is itself coherent across the gap (companions also
  continue) — true except at bundle edges.
- **Kill-test:** delete-one-MT robustness of neighbours' descriptors (§5c kill-test).

### Group-when-aliased, abstain-honestly (formal abstention)
- **Pitch:** When lateral spacing d < jump uncertainty δJ, match group↔group and abstain on
  per-MT identity instead of guessing.
- **Mechanism:** cluster endpoints whose pairwise d<δJ; commit the group correspondence to
  the aligner (centroid anchor), emit per-MT "unresolved" to the chain.
- **Inspiration:** §3 information bound — `d<δJ` is *provably* unresolvable.
- **Key assumption:** δJ is estimable per-region from tangent-noise × tilt (it is).
- **Kill-test:** forced per-MT pairing inside a flagged group should show ~50% precision
  against any held-out check; if it's high, the group was actually resolvable and δJ is
  overestimated.

### Two-witness join confidence (geometry × image agreement)
- **Pitch:** Trust a join only when geometry (ballistic residual) and image (patch
  correlation across the gap) independently agree; abstain on disagreement.
- **Mechanism:** confidence = AND of geometric mutual-NN and supra-background patch
  correlation; report it per-join for honest QC.
- **Inspiration:** §5b — two independent modalities, GT-free corroboration.
- **Key assumption:** MT-exit grayscale patches carry distinguishing structure (the tube +
  its surround), not pure noise.
- **Kill-test:** true-join patch correlation must beat random aligned-neighbour patch
  correlation by > noise margin; else the image witness is mute on MTs.

---

## Irreducible decomposition (the one-paragraph core)

Alignment = a *smooth, identity-free, outlier-allergic field*, best driven by the dense EM
image; Correspondence = *per-object identity with honest nulls*, only solvable once
alignment exists. They share data but not requirements, and the current design fuses them
through one 1:1 Hungarian assignment so each poisons the other. The gap is ballistic
(persistence length ≫ section), so the tangent *predicts* the jump — the most under-used
signal — and the only truly irreducible difficulty is that when lateral spacing falls below
the tangent-driven jump uncertainty (`d < δJ`), parallel neighbours are *information-
theoretically* unresolvable, which makes honest abstention a measurement, not a cop-out.
