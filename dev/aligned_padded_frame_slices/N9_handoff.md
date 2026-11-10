# N9 handoff: the documentation (plan §7, D-48, D-49)

## D-49 applied (document, no code change)

The representation dependence is stated in the ragged and uniform loop docstrings, `T3Frame.padded` /
`UT3Frame.padded`, the user doc ("Representation dependence"), `numerical_contracts.md`, the CHANGELOG ("Known
limitations"), the internals note and the math note (§7). The content: the hint-free start is a function of the old
frame's representation, slot signs included, because Gaussian seeds do not flip with a slot's sign. The layers'
starts of one point differ, and a start can differ across BLAS builds. Tucker completions are unaffected. The aligned
output is representation-independent to `tol`.

**Washout re-measured on the Gaussian convention** (`<scratchpad>/n2/n9_washout.py`, d = 4, generic / `t3svd` /
shared points, three seeds each). The starts differ by 0.95–1.00 in span. The final frames differ by 7e-4–3e-2 at the
default tol, with equal pass counts, and by 8e-13–6e-11 at tol 1e-10, with pass counts within 2. These replace the
coordinate-convention numbers (2e-5…5e-3 / 3e-14…5e-10) in the docs.

## Finding: what the fixed point maximizes (J-51; plan wording to fix)

The plan's §7 / D-48 call the fixed point "a stationary point of `‖Π_{T(F)} g‖²`" and speak of "many local maxima".
Stationarity holds, but the fixed point is **not** a maximum of that energy:

- **Measured on a T3** (`<scratchpad>/n2/stationarity.py`, d = 4, converged to 1e-12). Rotating any one of ten stores'
  completions out of its span by h = 1e-3 and 1e-4 changes E in the ratio 100.0 (second order), and the change is
  **always upward**.
- **Proved for matrices** (math note, Proposition 1). E = ‖S‖² − ‖(I−PPᵀ)N(I−QQᵀ)‖². The top-k singular pair is a
  saddle of E: one-sided rotations raise E, and paired rotations lower it. E counts rank-2k content.

The many strict local maxima belong to the note's cone merit. The docstrings, the user doc, the CHANGELOG and
`numerical_contracts.md` say exactly this.

## What was written

1. **Docstrings.**
   - `fv_aligned_padded_frame` (frontend): purpose; method and schedule; the fixed-point statement; properties;
     parameters and the return value; gotchas (eager jax: 22 s, the memory mappings, seven levels; the cap counts the
     truncating pass; the pair stop; NaN; `cone_max_passes=0` ≠ `'zero'`).
   - Doctests, all inequalities or structural facts:
     - one more pass moves nothing (< 1e-8);
     - a single-store pass moves nothing for each of U, L, R, O;
     - the aligned frame captures more than 2× what the start captures;
     - oversampling on vs off at 3 passes.
   - The uniform twin: what is uniform about it (block layout, compaction, the NaN record); the whole-loop jit and
     its figures. Its doctest: from differing starts, the uniform loop reaches the ragged fixed point (captured
     energy equal to 1e-8) and a NaN-padded record.
   - `padded` (both layers), `align_padded` (both layers), `padded_start` and the `padded_frames` module: the store
     link, the convention, links to the user doc.
   - The `Geometry` protocol docstring now lists the three optional members.
2. **User doc**, `docs/aligned_padded_frame.md` (toctree: design notes). It covers §7's list, defines "store" with an
   anchor link to `frame_variations.md`'s section, and gives the measured figures.
3. **Contributor notes.**
   - New `contributor/padded_frame_internals.md`, covering every item §7 lists.
   - The `functools.partial` no-precedent paragraph in `scan_body_principles.md`.
   - The seeded-completion QR in `uniform_svd_prefix_orthogonalization.md`.
   - A new section in `verification_records.md` (acceptance deferred to N10).
   - Existing pages per §7: `rank_continuation.md`, `fitting_and_optimization.md`, `numerical_contracts.md` and
     `api_reference.rst`.
4. **CHANGELOG** `[2026.4.0] — unreleased` (Added; Changed: none; Removed: none released; Known limitations), plus a
   release-notes "Upgrading from 2026.3.0" section.
5. **Math note** `docs/padded_frame_cone.tex` + pdf: 9 pages, clean pdflatex. It contains Nick's note specialized to
   T3s, Proposition 1 and five algorithm boxes citing the implemented functions.
6. **J-rows**: J-7..J-10 (the plan's), J-11..J-50 (every handoff call, N1–N8b) and J-51..J-54 (N9's).

## Judgment calls (J-51..J-54)

- **J-51**: the fixed-point wording above.
- **J-52**: the math note is named `padded_frame_cone.tex` per the brief; the plan says `aligned_padded_frame.tex`.
- **J-53**: three geometry members are documented; §7 says "two".
- **J-54**: the stale pre-D-41 T5 summary in `tests/backend/test_padded_frame.py`'s module docstring was rewritten.
  No test changed. **The file is not named for N9.**

## Verification

- **Module doctests**, the whole package: t3toolbox 241 passed, tttt (numpy 1.22) 241 passed. The touched modules in
  the floor126 venv (pip OpenBLAS): 71 passed.
- **Markdown doctests** (the CI command): pass.
- **`-W` docs build**: succeeded, and was rerun after the last edits (the anchor link resolves).
- **Full suite at `-n 3`**: 1,338 passed, 4 skipped, 48,697 subtests (14 m 48 s), exit 0.
- After the suite, only prose changed: two docstrings' wording, the user doc's zero-pass rule (the seed is exempt) and
  the math note's gap condition (`σ_k > σ_{k+1} > 0`). The edited modules' doctests were rerun on both envs, along with
  the touched markdown doctests and the `-W` build. All pass.

## Round 2 (`N9_review.md`: REVISE, five MINOR, two NITs)

**The fixed-point statement**, word for word in both twin docstrings, the user doc, the internals note, the CHANGELOG
and the math note's §5 summary (checked by script; only the math markup differs):

> At a fixed point, every store's completion spans the leading new directions of its own content, given the other
> stores' completions, so one more pass moves nothing. The completions are then a stationary point of the captured
> hint energy `‖Π_{T(F)} hint‖²` (proved for matrices, measured on T3s) but generally not a maximum of it: rotating one
> store's completion alone never lowers that energy and generically raises it at second order. In measurements on
> small trees, they are a strict local maximum of the cone merit, one of many.

The fixes:

1. **The proof.** One-sided rotations: "never decreases, and raises at second order with a component along `u_j`,
   `j > k`" (a left-null-space direction leaves E unchanged). The paired rotation:
   `R(θ) = Σ_{j>k+1} σ_j² + (σ_{k+1} + (σ_k − σ_{k+1}) sin²θ)² = R(0) + 2σ_{k+1}(σ_k − σ_{k+1})θ² + O(θ⁴)`, which is
   where `σ_{k+1} > 0` enters. "What the loop climbs" became "what the fixed points maximize locally, in the
   measurements".
2. **References.** T4S (arXiv:2603.21141v2: App. A.1, A.3, Algorithm 11; eqs. (46)–(52); App. A.3.1) and the
   maintainer's note, by title. The note is named at its first use. The pdf is 10 pages, clean.
3. **The uniform docstring** carries the statement above; the "which makes it" sentence is gone.
4. **What is ascended.** "Block coordinate ascent" is now "alternating (Gauss–Seidel) updates", in the internals note
   and the math note §8, with one addition: each store's update maximizes `‖P⁺ᵀY‖²` for its own content `Y` given the
   others; this is a self-consistency condition, not ascent of one objective; and the joint E is a saddle there (per
   store, a minimum). J-8's wording is corrected the same way.
5. **The ragged docstring.** A uniform hint must be converted first; passing one raises a `TypeError` naming the
   conversion.

The NITs:

- The CHANGELOG (via the statement) and `numerical_contracts` now hedge the local maximum ("in the measurements").
- The oversampling doctest now runs d = 5 with a low-rank hint at 2 passes and asserts more than 1.2× the
  non-oversampled capture. Across 18 cases (3 points × 6 start seeds, `<scratchpad>/n2/oversample_margin.py`) the
  ratio was 1.45–4.03, against 1.04–1.67 for the old example.

**Reruns.** Module doctests: 241 + 241 (both envs); the twins in floor126: 23. Markdown doctests on both envs: pass.
The `-W` build: exit 0.
