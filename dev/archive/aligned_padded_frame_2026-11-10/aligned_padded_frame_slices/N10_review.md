# N10 review: acceptance (plan §11), the final review

Reviewed: T3Toolbox at 9bacdaa5 (HEAD b017cbea differs only in the plan), the uncommitted `N10_handoff.md`, and
T3Polynomial eced8e3 as a diff (read-only: nothing written there; my runs wrote to scratch only).

**Summary.**
- The library passes every check I ran, and I found nothing in T3Toolbox that blocks the release.
- The adoption commit has one real bug: the hint-free start through the ladder raises on the first level, which breaks
  d10's `START=hint_free` (finding 1).
- One §11 criterion is not met: the laptop Newton comparison was not run. That needs Nick to decide whether to run it
  or decouple it (§3 below).

## 1. The §11 / CI checklist (spot checks)

| check | result |
|---|---|
| `-W` docs build (`sphinx -W --keep-going -b dummy docs <scratch>`, t3toolbox env) | **build succeeded, exit 0** (39 s) |
| `test_import_graph` + `test_dispatch` + `test_contractions_interpreter` (`-n 3`) | **99 passed, 2,766 subtests** (8 m 34 s), equal to the handoff's count |
| dispatch rows | every new array function has rows in `test_dispatch.py`: `pinned_completion`, `pinned_leading_columns`, `pad_safe_pinned_completion`, `fv_empty_frame`, `fv_padded_frame`, `tv_align_padded_frame`, `fv_aligned_padded_frame`, the five `ufv_*`/`utv_*`, `ufv_frame_widths`, `padded_start`. `oversampled_ranks` is integer bookkeeping, so it needs no row |
| leftover helpers (`git grep` over `t3toolbox/ tests/ docs/`) | none of `_coordinate_seed_order`, `_haar_seed_rows`, `_seeded_completion`, "coordinate convention", "unit-vector scatter" or "outside/inside" survives in the package or the docs. One stale test comment remains (NIT 4) |
| plan labels in the code added since v2026.3.0 (`git diff v2026.3.0 9bacdaa5 -- t3toolbox`, added lines) | none (the hits are `N0`/`N2` mode sizes and the prose "slice"/"plan" in unrelated senses). The contributor note cites D-/T-labels, which matches the older contributor docs (I-1, I-22 in `curvature_internals.md`) |
| `__all__` | every public module-level function and constant in the 14 touched modules is in `__all__` (AST scan against `__all__`). The only names outside are type aliases (`FrameData`, `UFrameData`, …), which follows the existing `Cores`/`Tangent`/`Point` convention |
| CHANGELOG / release notes | `[2026.4.0] — unreleased`. I diffed every public `def`/`class`/constant added since the `v2026.3.0` tag, and the CHANGELOG names all of them: the three geometry members, `pinned_completion` / `pinned_leading_columns` / `pad_safe_pinned_completion` (none of them existed at the tag), `padded_start`, the four `fv_*` / `tv_*` plus the `ALIGN_*` constants, the five uniform ones, `ufv_frame_widths`, `oversampled_ranks` ×3, `padded` ×2, the frontend loops ×2, `align_padded` ×2, `local_model(frame=)`, and all 14 `newton_cg` keywords plus the `cone_*` keywords on `gradient_descent` (checked against `inspect.signature`). `docs/release_notes.md` has "Upgrading from 2026.3.0" and `{include}`s the CHANGELOG. `pyproject.toml` and the `__init__` fallback still say 2026.3.0, which matches the precedent: ad578f63 bumped both in the release commit |
| user-doc cross-links | `frame_variations.md#four-rank-stores--and-why-excess-rank-is-inert` resolves (heading at line 68; `myst_heading_anchors = 3`, and `-W` passes). `padded_frame_cone.{tex,pdf}`, `scan_body_principles.md` and the five test paths the internals note names all exist. Both new docs are in a toctree (`design_notes.rst`, `contributor_guide.rst`) |
| the N9 findings | all five MINORs are fixed in cc7d6fc6: the R(θ) expansion in Proposition 1's proof; a bibliography with T4S (arXiv:2603.21141) and Nick's note; the uniform docstring scoped to "a strict local maximum of the cone merit"; the internals note's "not coordinate ascent of … E"; the ragged docstring now says a uniform hint "must be converted first" |

**The capture baselines, reproduced.** I ran `g06 PART=capture LEVEL=1 SEEDS=0,1 AMB_GNORM=81.050` with output to
scratch:

- library .1893;
- hint-free .0550 / .0709;
- aligned default .4508 / .4638 (10 passes each; last movements .50 / .17);
- 10 fixed passes identical to the default;
- frame defect ≤ 8.9e-16.

All of these match the handoff's 1 → 6 row.

## 2. T3Polynomial eced8e3

**The API use is correct and minimal.**
- The keywords used are `padded_from`, `g0_reference='cone'` and `padded_max_passes`, plus g06's `padded_start(prob, x0,
  x_old, seed=, max_passes=, tol=)`. All of them match `inspect.signature` of the frontend `newton_cg` and the backend
  `padded_start`. Every `padded_*` / `cone_*` keyword is keyword-only with default `None`.
- On the uniform backends `padded_from` is repacked with `from_t3(..., N=ux0.N)`. That is exactly the conversion the
  library's error message prescribes.
- The records read `history[0]['g0_reference' / 'g0norm']`, which both 2026.3.0 and 2026.4.0 write. They read
  `start_frame` / `start_movements` / `cone_movements` only when present.
- The default path is untouched: `padded_start=False`, and `kw.setdefault` lets an explicit `padded_from` win.

**My checks.**
- A spy on `_newton_fit` over a 3-level ladder confirms that each level after the first passes exactly the previous
  level's point. `‖X0 − padded_from‖` is 0.0, at ranks (1,1,1) → (2,2,2) → (3,3,2).
- The `uniform_jax` backend, which the new test does not cover, works with `padded_start=True` and `'cone'` (87 s; numpy
  `padded_from` beside a jax `x0`).
- `tests/test_fitting_selection.py`: 8 passed.

**The shim.** It raises `ImportError` at import, with a pointer to `newton_cg(padded_from=, g0_reference='cone')`,
`padded_start` and the user doc. That is appropriate: g04 now fails loudly instead of running retired code. See NIT 5
for the wording.

**The pin.** It is not a pin. `dependencies = ["numpy"]`; the `>= 2026.3.0` is a comment, because t3toolbox is
installed separately.
- My recommendation is to keep 2026.3.0 as the floor, since the default path runs there. Amend the comment to say
  "2026.4.0 for `padded_start` / `g0_reference='cone'`".
- On 2026.3.0 the opt-in path fails loudly (unknown keyword), so a hard pin buys nothing until 2026.4.0 is released.
- Environment note: in the `t3toolbox` env, `pip show t3toolbox` and `t3toolbox.__version__` report **2026.0.0**,
  because the editable install's metadata is stale. Any future runtime version check would misfire until
  `pip install -e .` is re-run.

## 3. The §11 acceptance criteria

| criterion | status |
|---|---|
| T3Polynomial adopts `padded_from` and `g0_reference='cone'` in its ladder; `cone.py` retired | **Met**, apart from finding 1. eced8e3 is not pushed (`origin/main` ahead 1) |
| CI gates: full suite on both legs, floor126 padded-frame files, doctests, markdown doctests, `-W` | **Met** (the handoff's numbers; my spot checks agree) |
| The hint-free capture baselines re-recorded on the Gaussian convention (D-41; §2.3 / §11) | **Met** (1 → 6 reproduced above) |
| **Laptop Darcy o01, levels 1 → 6 and 6 → 11:** library / hint-free / aligned at the default, 4 and 40 passes / aligned without oversampling; Newton and CG counts, objective and test error; two seeds; compile logging and peak memory on one level; the baseline beside them | **NOT met.** Not run. The handoff lists it under "Not run in N10". The one shipped-path Newton run (d10, aligned + `'cone'`, one seed, `gtol_ref` 3e-4) is not this comparison, and its level 2 stopped at `max_newton` 30 with ‖g‖ 0.124 against a stop of 0.0169, so the shipped path's Newton value at level 2 is still unmeasured. The d10 knobs cannot express the 4-pass, 40-pass or no-oversampling arms; g06 `PART=newton` adapted to `padded_start` (the handoff's suggestion) would be the natural vehicle. The hint-free arm needs finding 1 fixed first |
| Runpod 20 × 1000 | Not run. **Decoupled from the release** by the plan, so it does not block |

So, on the library side the release is ready from this review. N10 itself is complete only once the laptop
comparison is run, or Nick decouples it the way the Runpod item is.

A reading for Nick: under D-41 the cheap hint-free start captures less at 1 → 6 than the library's own padded frame
(.055 / .071 against .189; the coordinate convention had .104). Only the Newton comparison will say whether that
matters.

## Findings

### MAJOR

1. **`padded_start=True` with any `padded_*` option in `newton_kwargs` raises on the first level, so d10's
   `START=hint_free` cannot run.**
   - **Cause.** The ladder sets `padded_from` only when a previous level exists. `newton_kwargs` applies to every
     level, so level 0 receives `padded_max_passes=0` without `padded_from`, and the library rejects that by design:
     `ValueError: newton_cg: padded_max_passes set without padded_from`.
   - **Reproduced** on `ragged` and `uniform` with the selection test's toy problem; it fails before the first Newton
     run completes.
   - **Scope.** The `padded_start` comment and docstring advertise exactly this route ("its padded_* options pass
     through newton_kwargs; padded_max_passes=0 = the hint-free start"). d10 builds
     `nk = {..., 'padded_max_passes': 0}` for `START=hint_free` and passes it to every level.
   - **Not affected.** The handoff's numbers: g06 calls `padded_start` directly, and the d10 verification used
     `START=aligned`.
   - **Fix (T3Polynomial).** The ladder owns `padded_from`, so let it own whether the `padded_*` options apply: on a
     level with no `padded_from`, drop the `padded_*` keys, unless the caller passed `padded_from` explicitly.
     Alternatively, document that they must go through a per-level callable. Add a ladder test with
     `newton_kwargs={'padded_max_passes': 0}`; the new test passes only `'cone'`, which is why it missed this.

### MINOR

2. **T3Polynomial's `dev/HANDOFF.md` living status is stale after the adoption.** Its top block still says "Pending
   Nick: adopt the cone reference downstream", points at the superseded `padded_frame_cone_impl_plan_2026-11-02.md`,
   and describes `cone.py` as live. Add a block for eced8e3: the `padded_start` / `START` / `G0_REF` knobs, `cone.py`
   retired, and the Gaussian capture baselines.
3. **The adoption's own test covers only the `'cone'` route** and only the `ragged` / `uniform` backends. Finding 1 asks
   for the hint-free route. `uniform_jax` with `padded_start` works (checked above) but has no test; one level at tiny
   ranks would do, or a recorded decision to leave it to the d10 runs.

### NIT

4. `tests/test_dispatch.py:177`: "the padded start's in-block completion runs this path under jit: its coordinate
   seeds can lie in span(P)". The coordinate convention is gone (D-41). Say "its seeds" (chained starts once reused
   in-span rows), or drop the clause.
5. The shim says "T3Toolbox 2026.4.0 ships it". The library ships successors, not `cone.py`'s functions:
   - `cone_reference` → `g0_reference='cone'` (`stats['history'][0]['g0norm']`);
   - `tucker_cone_completion` → `padded_from` (Tucker and TT stores, not Tucker alone);
   - `cone_frame_zero_start` → `padded_from` at a rank-0 `x_old`.

   Naming the three replacements would let g04's next reader port it without reading the old file.
6. The plan's §2.2 (line 309) still calls a UL pass followed by an RO pass "block coordinate ascent with a projection
   between the halves". D-50 (round 9) superseded that reading. This is a dev document only.

## What I ran

1. `sphinx -W --keep-going -b dummy docs <scratch>` (t3toolbox env): exit 0.
2. `pytest -n 3 tests/test_import_graph.py tests/test_dispatch.py tests/test_contractions_interpreter.py`: 99 passed, 2,766 subtests.
3. `git grep` for the retired seed helpers and wording in `t3toolbox/ tests/ docs/`, and for plan labels in the lines
   added since `v2026.3.0`.
4. An AST scan of the 14 touched modules against their `__all__`; a diff of public definitions since `v2026.3.0`
   against the CHANGELOG; `inspect.signature` of `newton_cg` and `padded_start`.
5. Link targets and anchors of the user and internals docs; the toctree entries.
6. T3Polynomial (read-only):
   - `git show eced8e3`;
   - `tests/test_fitting_selection.py` (8 passed; `PYTHONDONTWRITEBYTECODE`, no cache);
   - a scratch ladder script with a `_newton_fit` spy on `uniform_jax` (`'cone'`), on `ragged` and on `uniform` with
     `padded_max_passes=0` (finding 1);
   - `g06 PART=capture LEVEL=1 SEEDS=0,1` with `G0_OUT` in scratch.

## Counts

MAJOR 1, MINOR 2, NIT 3. The MAJOR is in T3Polynomial; nothing in T3Toolbox blocks the release. Before N10 closes,
Nick has to rule on the unmet laptop Newton comparison: run it, or decouple it.

VERDICT: REVISE
