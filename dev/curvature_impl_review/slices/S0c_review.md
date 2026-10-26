# S0c review (reviewer), base b6295d0, uncommitted working tree

Status: COMPLETE -- VERDICT: APPROVE (no BLOCKER/MAJOR; 2 MINOR deferred with notes; 5 NIT optional)

## Scope read
- diff: 7 files, +468/-41 (CHANGELOG, uniform_internals.md, uniform_fitting.py, ut3_sampling.py, utv_sampling.py, 2 test files)
- handoff: impl/S0c_handoff.md

## Findings

### Coverage (item 1) -- verified
- Exactly 18 wrappers re-masked: six kinds' `transpose` (uniform_fitting.py:115/136/164/187/210/240, via
  `_UniformKind._remask` = `ufv_variation_masks_over_stack(frame_data[5], C, K+C)` + `ufv_apply_variations_masks`);
  six public `utv_*transpose` (utv_sampling.py:451-567 via `_remasked_transpose_output`, masks = the already
  attached `_gauge_masks_over_Knew`); six `ut3_*_corewise_*transpose` (ut3_sampling.py:119-275 via
  `_remasked_core_gradient` = `ut3_apply_masks` with the point's masks). The 12 `*_from_sweep` seams are untouched
  (bare). Only those 3 source files differ from b6295d0 (md5 over every `t3toolbox/**/*.py`).
- Frontends delegate (no reimplementation): `UT3Tangent.*_transpose` -> `utv_*` (uniform_manifold.py:672-830),
  `UniformTuckerTensorTrain.*_corewise_*transpose` -> `ut3_*` (uniform_tucker_tensor_train.py:536-668).
- No other uniform transpose wrapper that holds masks exists today (grep of `def .*transpose` across `backend/u*`
  and the uniform frontends).

### Bit-identity + mask exactness (item 2), reviewer's own check -- PASS
Scripts: review/scripts/r1_dump.py (dump all 18 wrappers' outputs, same seeded inputs, run once against a
`git archive b6295d0` tree and once against the working tree) and review/scripts/r2_compare.py (compare).
Matrix: minimal; Tucker-slack `(10,11,12),(4,4,4),(1,2,2,1)`; TT-slack `(6,7,8),(3,3,3),(1,3,9,1)` (right ranks
[1,3,3,1] < left [1,3,9,1]); C=(2,3) forced pad; varying-rank C=(2,); varying-rank C=(2,) mixing a TT-slack and a
Tucker-slack point; x K in {(), (3,), (2,2)} x sum_over_probes in {T,F} x {UNIFORM_MANIFOLD, UNIFORM_COREWISE} frames.
Expected real extent derived from the RAGGED points' / `t3m.MANIFOLD|COREWISE.frame(x)` core shapes.
Result: **1728 outputs, 0 failures** -- real block bitwise == pre-S0c, every padded slot exactly 0, utv attached
masks == the ragged-derived expectation AND unchanged from base. Pre-S0c padding garbage was > 0 for all 18
wrappers (6.0e1 .. 7.4e3 on these small cases), so the check is not vacuous. The kinds handle multi-axis K=(2,2)
and C=(2,3) correctly (the mask broadcast over K+C is right).

- Also d = 1 `(7,),(3,),(1,1)` forced pad and d = 2 C=(2,) forced pad (`R1_SET=d1`): 576 outputs, 0 failures.

### Under jit (item 2, jax 0.12 installed) -- PASS
review/scripts/r3_jit_dump.py: every one of the 18 wrappers `jax.jit`-compiled with the frame / point SUPERCORES
traced and shape + masks static (closed over); the kinds also run `precompute` inside the trace. Structures
tt_slack, varyC_slack, C23_pad; K=(3,); both sop; both geometries. Run against base and working tree, compared with
r2: **x64: 288 outputs, 0 failures; x32 (jax default float32): 288 outputs, 0 failures** -- real block bitwise ==
the base tree's jitted output, padding exactly 0. No tracer masks (both re-mask chokepoints call
`require_concrete_masks`, which would raise on a tracer -- none did). Each jitted wrapper traced exactly once over
two calls with new data. No pytree / aux / hash code touched (diff is confined to the three wrapper modules).
(jit vs eager numpy differ at 7e-16 relative -- expected, different engines; not part of the claim.)

### Tests (item 3) -- adequate
- `TestTransposeOutputHygiene` (test_uniform_probing.py, 12 wrappers, 528 subtests) and
  `TestKindTransposeOutputHygiene` (test_uniform_fitting.py, 6 kinds x 2 frame types, 144 subtests): exact masks
  derived from RAGGED core shapes (non-circular); padding == 0; real block bitwise == the bare body; 1e6*(1-mask)
  garbage in frame/point padding -> identical output; non-vacuity asserted (bare padding > 0 per wrapper).
  Per-method seeding (7/8/9/10/12/13/14; 21/22/23), subTest over case x K x op x sop, numpy only. 3.0 s together.
- Mutations reproduced (review/scripts/r4_mutations.py, in-process monkeypatch, no file edits):
  kinds no-re-mask / all-true / unshifted left-right -> 145 failing subtests each (KILLED); utv no-re-mask and
  correct-masks-but-all-true-re-mask -> 337 (KILLED); ut3 no-re-mask and tt-mask shifted -> 193 (KILLED).
  Two kinds mutants SURVIVE: right := left[1:] and down := up. Both are **equivalent mutants**: they widen the mask
  only over slots the frame's own (masked-on-entry) padding forces to exactly 0 (r5 below), so no output test can
  see them; the kinds return bare pairs, so there is no attached mask to assert. Not a test gap.
- test_dispatch: all 12 utv/ut3 wrappers are already jitted (test_jit_uniform_probing / _corewise_transpose /
  _derivatives, via the frontends); the kinds' `transpose` is jitted inside test_jit_uniform_optimizer_wholestep
  and _weighted_gn_model, and `_remask` is shared by all six. No new dispatch test needed (agree with handoff #8).

### Judgment calls (item 6)
- (a) frame vs point masks -- the plan's choice (frame masks, `frame_data[5]`) is what `_remask` uses. The
  reasoning is right, verified by review/scripts/r5_structural_zero.py: on TT-slack (54 dG slots) and Tucker-slack
  (44 dU slots) MANIFOLD frames, the bare output on point-real-but-frame-padded slots is **exactly 0**, and the
  bare garbage sits only in the padded boundary-bond slots of cores 0 and d-1, which both mask sets exclude. So the
  two choices are output-equivalent today; the frame choice is the one that stays right if a frame's masks ever
  stop being a subset of the point's (F23), and it keeps the kinds geometry-agnostic. Accept.
- (c) confirmed: `UniformApplyKind.forward` on a K=(3,) variation raises `ValueError: cannot reshape array of size
  12 into shape (3,3,4,1)` (`_UniformKind._variations` hands C-only masks to K+C supercores), while the ragged
  `APPLY.forward` returns W+K = (6, 3). Pre-existing (the line is untouched), a ragged/uniform parity gap. Not S0c's
  scope (S0c is output hygiene). It must be fixed before any uniform K-stacked hvp (plan §D row `LocalModel.hvp`,
  p K+C) -- i.e. by S8 at the latest (or S4 if its T-tests run a uniform K-stack). The fix is the one-line mirror of
  `_remask`: `ufv_variation_masks_over_stack(frame_data[5], frame_data[0].shape[1:-2], v_sc[0].shape[1:-2])` in
  `_variations`, plus a K-stacked uniform forward/hvp vs ragged test. **Deferred with a note** (see must-fix list:
  record it durably -- the handoff is ephemeral).

- (b) overhead: see the c39 run below. The plan's 0.08-0.61 % band timed the multiply alone; the implementer's
  1.1 % (small, eager numpy) includes the ~12 us host-side `ufv_variation_masks_over_stack` build, which under jit
  is folded away entirely (constants). Eager small-size runs are overhead-dominated anyway, and the utv wrappers
  already built these masks before S0c (only the multiply is new there). No fast path needed; a K=() fast path
  would save ~10 us eager and nothing jitted. Accept; record the honest number (MINOR m2 below, deferred to S13).

### House style (item 4) -- clean
- Masks stay host `np` (built by `ufv_variation_masks_over_stack` / `prefix_mask`); supercores flow through `*` /
  `xnp.einsum` on the data only; no `use_jax` threaded; no closures added (the `_point` lambda is pre-existing).
- `_remasked_core_gradient` carries the module's typed three-column signature; `_remasked_transpose_output`
  matches its neighbour `_gauge_masks_over_Knew` (the module's comment-only style). `_remask` has no per-argument
  shape comments, matching its siblings `_variations` / `_point` in `_UniformKind` (its docstring states the
  stacks) -- NIT n1 at most.
- No formatter churn: every hunk is on-topic; added lines stay within each file's pre-existing max line length;
  the one re-indented continuation (`UniformProbeKind.transpose`) keeps its alignment (`v_sc = ` and `return `
  are the same width).
- Docstrings/comments: the old "padding is NOT guaranteed clean" comment (ut3_sampling.py) is rewritten; the seam
  section comment and `utv_apply_transpose_from_sweep` docstring now say "bare by contract" (the plan's C.0 claim
  that the seam docstrings "already say" so was only true of that one docstring, and it described the pre-S0c
  behaviour -- handoff conflict #2 resolved correctly). The only callers of the six `utv_*transpose*_from_sweep`
  seams are the six kinds (grep), so "their only callers, the kinds, re-mask" is true.

### CHANGELOG / uniform_internals.md (item 5) -- accurate
- `## [2026.3.0] — unreleased` / `### Changed — breaking` is the §G placement and §H's "CHANGELOG [2026.3.0]
  header" for S0c. Wording is accurate (wrapper list, "real block bit-identical", "breaks only code that read the
  padding", seams stay bare). It renders into `docs/release_notes.md` via `{include} ../CHANGELOG.md` and has no
  cross-references, so it cannot add a sphinx `-W` warning (sphinx not installed here; not run).
- The `uniform_internals.md` bullet (handoff #3, beyond §G's list) keeps the documented masking schedule complete
  and is accurate (r5 confirms the garbage is exactly the boundary-bond squash's). Good call.
- Observation (NIT n3): "bit-identical" is verified on numpy and XLA:CPU (x32 and x64). `ut3_apply_masks` is a
  `jnp.einsum` that jax lowers to `dot_general` (5 in the StableHLO), which XLA:CPU strength-reduces to one
  `multiply`; accelerators were not checked. This is the same chokepoint mask-on-entry already uses everywhere, so
  it is not new risk from S0c; no action for this slice.

### Runs (item 7)
- Touched files: `pytest tests/backend/test_uniform_fitting.py tests/test_uniform_probing.py -n 2` -> **64 passed,
  872 subtests passed** (matches the handoff).
- `pytest --doctest-modules t3toolbox -n 2` -> **194 passed**. `python -m doctest` on
  docs/contributor/uniform_internals.md, docs/uniform_equivalence_contract.md, docs/release_notes.md (which
  includes the CHANGELOG) -> OK.
- `python -I -B dev/curvature_impl_review/checks/c39_remask_overhead.py` (after): max padded output **0.0e+00** on
  all four (size x geometry) cases, real block bit-identical **True**; the now-redundant second multiply costs
  0.45 / 0.42 % (small, COREWISE / MANIFOLD) and 0.06 / 0.06 % (large). Consistent with the handoff's after-numbers.
- Full re-mask cost (review/scripts timeit, best of 5 x 2000): `_remask` = mask build 12.5 us + multiply 10-12 us
  = **24 us (small) / 27 us (large)** = ~1.1 % / ~0.13 % of the eager transpose (2.2 ms / ~20 ms). Matches the
  handoff's 22 us / 1.1 %. Zero under jit.

## Findings summary (severity, citation, fix)

No BLOCKER. No MAJOR.

- **MINOR m1 -- record the found-not-fixed K-stacked `forward` gap durably (deferred, coordinator action).**
  `backend/uniform_fitting.py:89-92` (`_UniformKind._variations`, `def` at :90, untouched by S0c) gives C-only masks to K+C
  variation supercores, so every uniform kind's `forward` raises on a K-stacked variation (reproduced: `ValueError:
  cannot reshape array of size 12 into shape (3,3,4,1)`), while the ragged `APPLY.forward` returns W+K. The plan's
  §D needs `LocalModel.hvp` on p K+C, so this must be fixed by the first slice that runs a uniform K-stacked hvp
  (S8 at the latest). Out of S0c's scope; it should not hold this commit. Fix: add a line to the plan's §H S8 row
  (or §J) and to dev/HANDOFF.md -- "`_UniformKind._variations`: use `ufv_variation_masks_over_stack(frame_data[5],
  frame_data[0].shape[1:-2], v_sc[0].shape[1:-2])` (the `_remask` mirror) + a K-stacked uniform forward/hvp vs
  ragged test". The S0c handoff is ephemeral and is not that record.
- **MINOR m2 -- the plan's S0c overhead figure understates the eager cost (deferred to S13).** C.0 / F34 / §E
  quote 0.08-0.61 % (the multiply alone). Including the host mask build the kinds' re-mask is ~24 us = ~1.1 % of
  a small eager transpose (~0.13 % large; 0 under jit). Fix: when S13 writes verification_records.md (F34),
  record "multiply 0.06-0.5 %; with the mask build ~1.1 % small / ~0.13 % large, eager numpy; folded under jit".
  No code change.
- NIT n1 -- `_UniformKind._remask` (uniform_fitting.py:94-102, `def` at :95) has no per-argument shape comments; its siblings
  `_variations` / `_point` have none either and the docstring states the stacks. Optional: `v_sc  # bare
  variation supercore pair, (d,)+K+C+(...)` / `frame_data  # UT3Frame .data, stack C`.
- NIT n2 -- `TestKindTransposeOutputHygiene._points` has no Tucker-slack case and no multi-axis C / K (the
  probing class has C=(2,3)). The reviewer's r1/r2 matrix covers Tucker slack, C=(2,3), K=(2,2), d=1, d=2 and a
  mixed-slack varying-C stack and passes, so this is optional hardening: add
  `t3.TuckerTensorTrain.randn((10, 11, 12), (4, 4, 4), (1, 2, 2, 1))` and `K=(2, 2)`.
- NIT n3 -- "bit-identical" is verified on numpy and XLA:CPU only (`ut3_apply_masks` lowers to `dot_general`,
  which XLA:CPU rewrites to a multiply). Pre-existing chokepoint; no action.
- NIT n4 -- `backend/ut3_operations.py:86` still says nonzero boundary padding is "exactly what a corewise
  gradient step leaves behind"; after S0c the library's own corewise gradients are clean (the mask on entry is
  still right, as defence against user-supplied padding). Optional reword at S13 ("what a raw gradient or
  user-supplied supercore may leave").
- NIT n5 -- the re-wrapped `ut3_sampling.py` module docstring leaves a short line ("wraps. The algorithm story is
  in"). Cosmetic; re-flow if touched again.

**Must-fix before commit:** none in the implementation. m1 needs a durable note (plan §H S8 row and/or
dev/HANDOFF.md) and can go in the same commit or the coordinator's next dev/ update. m2 and the NITs are deferred
or optional.

## Full suite
`PYTHONPATH=$PWD pytest tests/ -q -n 2` (reviewer's run, log review/full_suite_review.log): **901 passed, 1 skipped,
43,211 subtests passed, EXIT 0** (5 min 55 s). Identical counts to the implementer's full_suite.log (= the
2026.2.0 baseline plus the three new test methods). Existing tests: the test diffs are additions only (0 removed
lines), so §H's "every existing uniform test unchanged" holds. Working tree untouched by the reviewer (`git status`
still lists exactly the implementer's 7 files, +468/-41).

## Verdict
The implementation does what C.0 / T27 say: 18 wrappers re-masked with the right masks (frame masks for the kinds,
the attached gauge-shifted masks for `utv_*`, the point masks for `ut3_*`), seams bare. The K+C broadcast is
correct, including multi-axis K and C. The real block is bitwise identical to b6295d0, both eager and under jit
(x32 and x64). Padding is exactly 0. The tests are non-circular, non-vacuous and kill the meaningful mutants.
The docs and CHANGELOG are accurate. The two MINORs are deferred with notes (m1: a durable record of the K-stacked
`forward` gap for S8; m2: the honest overhead number at S13). The NITs are optional.

VERDICT: APPROVE
