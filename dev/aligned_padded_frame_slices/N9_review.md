# N9 review: the documentation

Reviewer: Claude Opus 5.5, 2026-11-10. I reviewed the uncommitted working tree on top of 5b188bd7 against:
- the plan: §7, D-40, D-42, D-48, D-49;
- `N9_handoff.md` and J-51…J-54;
- the session's CI rules: the `-W` build, both environments' doctests, and stable doctests with no printed floats.

## Summary

The documentation is accurate, and the J-51 correction is mathematically right. I verified Proposition 1 line by
line and numerically, and I reproduced the T3 stationarity measurement independently on both layers. The fixed point
is a stationary point and a saddle of the captured energy `E = ‖Π_{T(F)} S‖²`.

Moreover, **per store it is a MINIMUM of E**: every one-store rotation raises E. So the step is not block-coordinate
ascent of E. The user doc and the ragged docstring state this correctly ("stationary; generally not a maximum; such
rotations typically raise it"), and none of the text claims block-coordinate ascent of E. One phrase in the
internals note ("block coordinate ascent") does not say what is ascended, and should (finding 4).

Five MINOR findings, all wording or two-line additions:
- the proposition's proof needs two tightenings (one hypothesis is used only implicitly);
- the math note cites nothing (no References; "the note" is undefined in the body);
- one docstring states a measured local maximum as a consequence;
- the internals-note phrase above;
- one docstring sentence misdescribes the uniform-hint path.

Everything else checks:
- defaults and keywords against `inspect.signature`;
- the jit-or-eager figures, traceable to `verification_records.md` and the plan;
- "store" linked on first use, and the anchor exists;
- the math note compiles (9 pages) and every function it cites exists;
- the CHANGELOG respects the non-breaking rule.

Builds and doctests:
- the `-W` docs build exits 0;
- the module doctests pass on both environments (241 + 241);
- the markdown doctests, run with CI's own command, pass on both environments;
- `test_padded_frame.py` passes.

## (1) J-51: what the fixed point satisfies

### Proposition 1, line by line (`docs/padded_frame_cone.tex`, §5.1)

- **Eq. (energy).** `Π_T = I − (I−U'U'ᵀ)(·)(I−V'V'ᵀ)` is the tangent projector at a point with frames `U' = [U P]`,
  `V' = [V Q]`. Then `‖Π_T S‖² = ‖S‖² − ‖(I−Π_T)S‖²`. And `(I−U'U'ᵀ) = (I−PPᵀ)(I−UUᵀ)` because `PᵀU = 0`, so
  `(I−Π_T)S = (I−PPᵀ)N(I−QQᵀ)`. **Correct.** Numerically the identity holds to 0.0 against the direct
  `‖U'U'ᵀS + SV'V'ᵀ − U'U'ᵀSV'V'ᵀ‖²`.
- **Stationarity.** With `Q` at the top-k right vectors, `R = ‖M‖² − ‖PᵀM‖²` with `M = N(I−QQᵀ) = Σ_{j>k} σ_j u_j v_jᵀ`.
  At `P = [u_1…u_k]`, `PᵀM = 0` is the minimum of `‖PᵀM‖² ≥ 0`, so the P-gradient vanishes; symmetrically in `Q`.
  So the point is jointly stationary. **Correct.**
- **"Rotating P alone … increases E at second order"** is slightly too strong. It is true for any rotation with a
  component along some `u_j`, `j > k`, `σ_j > 0`. A rotation toward a direction in `N`'s left null space (which
  exists when the complement of `U` is larger than `N`'s rank) leaves `E` unchanged. The proof's own argument shows
  only "does not decrease" in general. **MINOR (finding 1a).**
- **The saddle's descent direction.** The proof evaluates only the END of the paired rotation (`u_k→u_{k+1}`,
  `v_k→v_{k+1}` completed). A saddle needs local descent. Along the paired rotation by `θ`:

      R(θ) = Σ_{j>k+1} σ_j² + (σ_{k+1} + (σ_k − σ_{k+1}) sin²θ)²

  This increases at second order, at rate `2σ_{k+1}(σ_k − σ_{k+1})θ²`, exactly when `σ_k > σ_{k+1} > 0`. With
  `σ_{k+1} = 0` the decrease of `E` is only fourth order. So the hypothesis `σ_{k+1} > 0` is used precisely here, and
  the proof should show it. **MINOR (finding 1b).** The statement itself is TRUE.

### Numerically (`n9_stationary.py`)

**Matrix** (12×10, `r = k = 2`), at the top-k pair:
- `P` alone toward `u_{k+1}`: `ΔE = +9.72e-6 / +9.72e-8` at `h = 1e-3 / 1e-4` (ratio 100.0, odd part 0);
- `Q` alone: the same;
- `P` alone toward a random direction orthogonal to `[U P]`: `+4.62e-6 / +4.62e-8`;
- paired toward `(u_{k+1}, v_{k+1})`: `−9.86e-6 / −9.86e-8` (ratio 100.0).

So it is a saddle, and a per-block MINIMUM of `E`. `E`'s value there is 86.34; with `Q` moved to `v_{k+1..2k}` it is
101.63 (the rank-2k capture).

**T3** (d = 4, shape `(4,5,4,3)`, old `(2,2,2,2)/(1,2,2,2,1)`, target `(3,3,3,2)/(1,3,3,2,1)`), converged:
- I rotated every store with room (8 of 10; two completions fill their unfolding) by `h = 1e-3, 1e-4, −1e-3`.
- The ratio `ΔE(1e-3)/ΔE(1e-4)` lies in **[99.4, 100.1]**; `ΔE` is **always positive**; the odd part is ≤ 2.4e-3 of
  `ΔE`.
- This holds for the ragged loop with a dense hint and with a T3 hint, and for the **uniform** loop's read-back. The
  uniform and ragged fixed points agree to 1.0e-14.

**The implementer's measurement reproduces.**

### Block-coordinate ascent? Which statement the docs make

- Given the others, a store's leading-directions completion is a **minimum** of `E` along that store's rotations: the
  matrix proof's own computation, and every one of my T3 rotations.
- So the alignment is NOT block-coordinate ascent of `E`. What the leading-directions property maximizes is the
  store's captured CONTENT `‖P⁺ᵀY‖²` for its own content `Y`. `Y` depends on the store's previous completion, so that
  is a self-consistency condition, not the argmax of a fixed objective.
- **The docs:**
  - The user doc says "stationary … generally *not* a maximum of it … such rotations typically raise it". The ragged
    docstring says the same, with "Galerkin-type stationarity". Both are **correct**.
  - Neither claims ascent of `E`.
  - The internals note's "Two-colour passes … are block coordinate ascent with a projection in between" names no
    objective. It must not be read as ascent of `E` (finding 4).
  - The local maxima are attributed to the cone merit, and the docs mark them as "measured". The uniform docstring is
    the exception (finding 3).

## (2) Findings

### 1. MINOR: Proposition 1's proof needs two tightenings

`docs/padded_frame_cone.tex`, §5.1. Both are described in (1).

**Fix.**
- (a) Replace "increases E under any rotation of P alone" with "does not decrease E, and increases it at second order
  for any rotation with a component along `u_j`, `j > k`".
- (b) Add the `R(θ)` line above, to show the second-order decrease along the paired rotation and where `σ_{k+1} > 0`
  enters.
- Rebuild the pdf.

### 2. MINOR: the math note cites nothing

`docs/padded_frame_cone.tex`.

**Problem.**
- There is no references section.
- The body speaks of "the note", "the note's stopping criterion" and "the note's merit", but only a source comment
  says which note that is: the maintainer's "Completing frames by truncating A + εS, with projection-only access".
- The T3 frames, tangent spaces and orthogonal representations the note uses are T4S's (Alger, Christierson, Chen &
  Ghattas, arXiv:2603.21141: Appendix A, the doubled-rank tangent representation eqs (49)–(52), Algorithm 11). The
  house rule is to cite by the arXiv version's numbering.

**Fix.** Add a short References section with T4S (arXiv version numbering) and the maintainer's note (an unpublished
note, by title), and name the latter at its first use in the body.

### 3. MINOR: the uniform docstring presents a measured local maximum as implied

`ufv_aligned_padded_frame` docstring: "every store's completion spans the leading directions of its content given
the others, **which makes it** a stationary point of the captured energy … and a local maximum of the note's cone
merit".

**Problem.** The leading-directions property gives stationarity (proved for matrices, measured on T3s). The merit's
local MAXIMUM is a measurement on small trees, which the ragged docstring correctly says ("measured to be a strict
local maximum").

**Fix.** "…a stationary point of the captured energy, and, as measured, a local maximum of the note's cone merit (one
of many)".

### 4. MINOR: "block coordinate ascent" in the internals note names no objective

`docs/contributor/padded_frame_internals.md`, the Transport paragraph: "Two-colour passes (`'UL'`, then `'RO'`) are
block coordinate ascent with a projection in between."

**Problem.** Read as ascent of the captured energy it is false: per store, the fixed point minimizes `E`. Ascent of the
merit is not established for trees.

**Fix.** Say what is meant, e.g. "alternating (Gauss–Seidel) updates of the two colours, with a projection in
between".

### 5. MINOR: the ragged docstring misdescribes the uniform-hint path

The ragged `fv_aligned_padded_frame` docstring, under Parameters: "A `UniformTuckerTensorTrain` hint is converted
with `.to_t3()`."

**Problem.** That reads as an automatic conversion. The code raises a `TypeError` that asks the caller to convert.

**Fix.** "…must be converted first (`hint.to_t3()`); passing one raises a `TypeError` naming the conversion."

### 6. NIT: two summaries state the merit's local maximum as a fact

The CHANGELOG's "Known limitations" ("The alignment reaches a local maximum of its merit") and the
`numerical_contracts.md` row ("a local maximum of its merit") both state it without qualification.

**Fix.** "measured to be a local maximum", or "a fixed point (a local maximum of the merit in the measurements)".

### 7. NIT: the oversampling doctest has an 11 % margin over a representation-dependent start

The ragged docstring's last doctest asserts `captured(F5) > captured(F0)` at 3 passes. On this machine that is
16 730.7 against 15 012.1, a margin of 11 %. The start is BLAS-dependent (D-49), so another LAPACK's sign choices
change both runs.

It passed on the three environments run (the implementer's, `tttt`, and floor126). The other doctests have wide
margins:
- `< 1e-8` against measured 8e-11 … 2e-11;
- `> 2×` against 2.72×;
- the 300-pass cap against 227 passes.

**Fix.** Optional: a case with a wider margin, or move the comparison into `tests/` and keep the doctest's prose.

## (3) The rest of the docs review

- **Keywords and defaults against `inspect.signature`.**
  - The loop twins (frontend and backend): `oversampling=5`, `oversampled_passes=2`, `max_passes=10`, `tol=0.05`,
    `alternate=True`, `seed=0`, `use_jit=False`.
  - `newton_cg`: the fourteen keywords exactly as documented; `g0_reference=None`, `use_jit=False`; `gtol_rel`
    exists (default 1e-8; the docs RECOMMEND about 1e-4 with `'cone'`, correctly worded as a recommendation).
  - `gradient_descent` (backend): the six `cone_*`.
  - `T3Frame.padded` / `UT3Frame.padded`, `T3Tangent.align_padded`, `padded_start` and `oversampled_ranks`: as
    documented.
- **The jit-or-eager figures.**
  - Ragged eager first step at d = 4: 932 compilations, ~22 s, 0.16 s after; jitted, ~2 s per program. These are
    the plan's §2.3 / D-39 measurements.
  - Uniform whole loop: "about 9 s at d = 6 and d = 24, then 0.02–0.08 s" matches `verification_records.md` (8.8 s /
    9.0 s; 0.02–0.08 s).
  - 8,000–9,500 memory mappings: N8's measurement. "About seven levels" is the arithmetic `65 530 / ~9 000`, and the
    text says "can", which is right.
  - The Darcy figures (49 → 21–33 Newton; 2–5×; ~40 % and 2.4× at 11 → 16; 45 ms against 1.1 s) are the plan's §2.3
    and §10 numbers. They are marked as predating D-41.
- **"Store" linked on first use.**
  - The user doc links `frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert`, and the heading exists
    (`## Four rank stores — and why excess rank is inert`).
  - The ragged docstring links `:doc:/frame_variations` with the section name.
- **The math note.**
  - It compiles with pdflatex in scratch: 9 pages, the same size as the committed pdf. The only warning is the
    first-pass "rerun for cross-references".
  - All 50 `\code{}` names resolve: functions, members or keywords that exist (e.g. `backend.optimizers._cone_reference`,
    `padded_frames._seed_rows`, `linalg.pad_safe_pinned_completion`).
  - Citations: finding 2.
- **The CHANGELOG under the non-breaking rule.**
  - `[2026.4.0] — unreleased`, with "Added" only.
  - "Changed: None. The default Newton reference stays `'zero'`, and `fv_zero_point_frame` is unchanged." That is
    consistent with my bitwise G-bit checks in N7 and N8b.
  - "Removed: none released" is correct: the template-signature `fv_padded_frame` never shipped.
  - `pinned_completion` / `pinned_leading_columns` are correctly listed as new, since S1 post-dates 2026.3.0.
  - "Known limitations" covers D-49, the merit (NIT 6) and the NaN record.
- **Existing pages.** The rank-continuation section (the snippet is a plain ```python block, not a doctest), the
  `g0_reference` table row, the three optional members in the `Geometry` protocol text, and the `numerical_contracts`
  rows are all consistent with the code.

## What I ran

1. `n9_stationary.py`: the matrix case of Proposition 1 and the T3 rotation measurement (ragged dense hint, ragged T3
   hint, uniform read-back).
2. The doctest margins of the ragged docstring (numbers in NIT 7).
3. `inspect.signature` on every documented entry point.
4. pdflatex of the math note in scratch, and the `\code{}` name resolution.
5. **Docs.** `python -m sphinx -W --keep-going -b dummy docs <scratch>`: **exit 0**.
6. **Module doctests** (`pytest --doctest-modules t3toolbox`): **241 passed** in env `t3toolbox` (numpy 2.x,
   `-n 3`), and **241 passed** in `tttt` (py3.9, numpy 1.22.4).
7. **Markdown doctests** (CI's command, `python -m doctest docs/*.md docs/contributor/*.md` without
   `doctest_style.md`): **exit 0, no failures**, in both environments.
8. **`tests/backend/test_padded_frame.py`** (the module docstring rewrite, J-54): **50 passed**.

## For the maintainer

1. **J-51 is right; update the plan's §7 / D-48 wording.**
   - The fixed point is stationary for `E = ‖Π_{T(F)} S‖²` and a saddle of it. Per store it is a MINIMUM of `E`
     (rotations raise `E`), so the step is not block-coordinate ascent of `E`.
   - The "many strict local maxima" belong to the cone merit, and were measured on small trees.
   - §2.2's "block coordinate ascent with a projection between the halves" needs the same care (finding 4).
2. **The math note should cite T4S and your note** (finding 2) before the pdf is treated as the reference.

## Counts

MAJOR 0, MINOR 5, NIT 2.

VERDICT: REVISE
