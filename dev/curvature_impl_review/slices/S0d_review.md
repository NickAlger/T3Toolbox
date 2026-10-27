# S0d review (reviewer): the zero reference, the zero-point frame, the stats keys

Repo /home/claude/t3toolbox, main @ 2e184ed. The implementer's tree is uncommitted (21 modified files + new
`tests/backend/test_zero_reference.py`). Nothing in the repo was edited by the reviewer. Mutation runs used a scratch
copy (`review_checks/mut/`), and the HEAD comparison used a `git archive` export (`review_checks/head/`).
Reviewer scripts: `/tmp/claude-0/-home-claude/29728d3c-c433-5028-9de9-1af7ead3c123/scratchpad/impl/review_checks/`.

**VERDICT: APPROVE.** There is no BLOCKER and no MAJOR. The two MINORs are each a one-line fix; fold them in before the
commit. The NITs are optional.

## Must-fix list (trivial; fold into the S0d commit)

1. **MINOR-1.** In `docs/contributor/sharing_internals.md:111-113`, make the stall claim historical or drop it.
2. **MINOR-2.** In `tests/backend/test_zero_reference.py:433`, assert equal trajectory lengths before comparing
   losses. Verified: the assertion passes today.

## Findings

### MINOR

- **MINOR-1 (doc accuracy).** `docs/contributor/sharing_internals.md:111-113` now reads: "with the warm start as the
  reference the fit stalls at the target level and continuation over-grows (reproduced in the tests)".
  - No current test reproduces that claim.
  - At the test's own seed (`setUp` → `np.random.seed(0)`), the 14h body with `g0_reference='warm_start'` and no pin
    does not stall. It reaches the true shared ranks (3,3,3) at 1.1e-10 in 23/22/10 Newton steps. The default run
    takes 24/22/8 steps to 3.1e-10, and the old pin 24/23/10 to 1.2e-10 (`review_checks/sharing_14h_variants.py`).
  - The 2026.2.0 sentence was historical ("reproduced, then fixed"); the new present tense overclaims.
  - **Fix:** write e.g. "(reproduced while building the 2026.1.0 shared continuation)", or drop the parenthetical.
    `docs/rank_continuation.md:168-169` is already in the past tense and is fine.
- **MINOR-2 (test strength).** `tests/backend/test_zero_reference.py:433-435` (`test_seeded_pairs_track`) compares
  only the common prefix `m = min(len(lr), len(lu))`.
  - A layer mismatch in the reference first shows up as a different **stop iteration**, and the min-prefix hides that.
  - The O2 sweep's `SM.rel` cannot compare arrays of different lengths, so the sweep effectively requires equal ones.
  - **Fix:** add `self.assertEqual(len(lr), len(lu))` before the norm check. It passes on the current tree (verified
    on the scratch copy).

### NIT (optional)

- **NIT-1.** The explicit `g0_reference='zero'` + `g0norm_newton` branch is untested. In that branch `zero_point_frame`
  is called once, only to validate; the run records `'override'`; and it raises on a corewise geometry. I verified the
  behaviour (`count_cost.py`). Suggested test: one row in `test_overrides_win_and_skip_the_computation`,
  `run(g0_reference='zero', g0norm_newton=5.0)` → `calls == 1`, `'override'`.
- **NIT-2.** `warnings.warn(..., stacklevel=3)` points at the caller of the *backend* optimizer. From the frontend, the
  RuntimeWarning is attributed to `t3toolbox/optimizers.py:442` / `:262` rather than to user code. Harmless; the house
  has no stacklevel convention.
- **NIT-3.** `examples/fit_hilbert_tensor_newton_cg.py` is also a hand-rolled loop with the warm-start reference. Only
  `fit_varied_rank_tensor_newton_cg.py` got the "does not inherit library fixes" note, which is all §G asked for.
- **NIT-4.** `gradient_descent` does not record which reference it resolved (its stats have no history). The plan
  does not ask for it.
- **NIT-5.** `tests/test_dispatch.py:1077` imports `ufv_operations` inside the test body; the file's other imports are
  at module top.
- **Pre-existing, not S0d (for HANDOFF).** `examples/fit_hilbert_uniform_newton_cg.py` closes by printing "the true
  errors match the ragged fit_hilbert_tensor_newton_cg.py run to numerical tolerance".
  - That was already false at HEAD: the true error at level 4 is 1.557e-2 vs 1.733e-2, and the best levels differ
    (4 vs 3). The two scripts run different optimizers, the library's vs a hand-rolled one.
  - S0d moves the uniform numbers again: level 4 is now 1.715e-2 and the best level is 3.
  - The plan's audit marks this pairing "n/a", so it needs no S0d action. A prose fix belongs in S13.

## Item-by-item verification

**1. The recipe (§C.2) — matches exactly.**
- `_haar_rows` matches the plan text line for line:
  - `RandomState([seed, k_g]).standard_normal((n_g, N_g))`;
  - `qr(G.T)`;
  - the sign fix `np.where(diag(R) >= 0, 1, -1)`;
  - transpose.
- `key[i] = min(group)`, and the factor is replicated over the group.
- The TT cores are identity prefixes, with `L[d-1] = R[0] = 0`.
- The result is cast with `xnp.asarray(..., dtype=template dtype)`.
- A stacked template raises `NotImplementedError` and names `_require_unstacked`, on both layers.

| independent check | result |
|---|---|
| plan-text recipe, written from the plan alone (`indep_recipe.py`), vs the library | identical (0.0) on 10 structures: d = 1–4, maximal Tucker, varied ranks, shared incl. non-contiguous groups `((0,2),(1,))` and `((1,3),(0,2))` |
| validity | orthogonality residual ≤ 7.8e-16; base point exactly 0 (both reconstructions); Π idempotent to 5e-16 |
| ragged vs uniform real blocks | 0.0 |
| rank independence | bitwise: d = 4, 12 rank/size perturbations, plus a shared group |
| nesting | 0.0 (c41 ladder); 1.9e-16 on a d = 4, N = 9 ladder (fp, within the tests' 1e-15) |
| dtype / backend | float32 and jax templates give float32 / jax output, equal to the numpy result to 2.6e-8 |
| tracer safety | the draw reads only shapes, so it even works under `jax.jit` |

Re-runs from the repo root:

| script | result | matches |
|---|---|---|
| `c41` | (1) True/True; (2) 0.0; (3) non-decreasing on all 4 ladders; (4) 4.740e+01; (5) 44 O2 pairs, references 3.4e-16, losses 3.5e-12 | F37 exactly |
| `c38` | trailing-supported data: LAPACK 0 / identity prefix 0 / Haar 1.596e1 (seeds 0–5: 9.5–24); generic data: 6.985e1 (LAPACK) vs Haar 4.41e1–7.03e1 | F32 exactly |
| implementer's `repro_c41_c38_c29.py` | identical to `logs/repro.txt`, except c41(3)'s ladder values | (3) uses unseeded global-rng targets; monotone in both runs |

The implementer's repro also has:
- c38 under the shipped recipe: 1.867e1 on trailing-supported data;
- c29 (GN): warm 39/15 steps, LAPACK zero reference as an override 36/13 (= F26), canonical default 37/13.

**2. The resolution (§C.15, all five steps) — correct.**
- `None` resolves to `'zero'` on the manifold geometries (ragged, uniform and shared) and to `'warm_start'` on the
  corewise ones and on a custom geometry without the member.
- An explicit `'zero'` with no zero-point frame raises a ValueError. The message names either the missing member or
  the corewise reason.
- Computation happens iff the resolved reference is `'zero'` and `g0norm_newton is None`. Counted against
  `'warm_start'` (`count_cost.py`):
  - the default costs exactly +1 `kind.precompute`, +1 `kind.transpose`, +1 `kind.point_forward`, +1
    `geom.precompute`/`project`/`inner` and +1 `zero_point_frame`;
  - there is no extra `geom.frame`, so no orthogonalisation;
  - that holds on ragged and uniform, at `max_newton` 1 and 3;
  - with `g0norm_newton` given, nothing extra is computed.
- There is no second compile:
  - `chunk_size='auto'` is resolved once in `_setup`;
  - the reference is computed eagerly, outside `_cg_core`'s jit;
  - on uniform_jax in float32 (`UNIFORM_MANIFOLD` and shared), `use_jit=True` gives the same reference as numpy to
    1e-8 and 5e-8.
- Overrides win, with the chained fallback. `NewtonInfo.g0_reference` and `cg_reference` follow m32 exactly; with
  `g0norm_newton` alone, `cg_reference` is `'override'`, which equals `g0_reference`.
- Non-positive and NaN overrides raise.
- A vanishing **default** reference falls back to the warm start with a RuntimeWarning:
  - all-zero data from a nonzero start converges with `'gradient'`;
  - from a zero start it stops at iteration 0 with `'gradient'`, g0norm 1.0.
- A vanishing **explicit** `'zero'` raises ValueError. The warning and error texts are the plan's.
- The forcing term is unchanged: `eta == min(0.5, (gnorm/ref)^0.5)` on every row. The default power is still 0.5.
- The converged display line names the reference, and a hand-built row prints as before (doctest + test).

**3. Stats (§C.9) — correct.**
- `stats` keys are exactly `{losses, newton_steps, newton_accepted, stop_reason, history}`. `'newton'` is removed,
  not aliased.
- `stop_reason` covers all three values, each driven by a test. The Armijo-exhausted `'stop'` gives
  `'line_search'` with `(newton_steps, newton_accepted) = (1, 0)` and `x` unchanged. `'accept'` counts as accepted.
- The new `NewtonInfo` fields are documented in the class docstring.
- Grep for `['newton']`, `.newton` and `stats[` across t3toolbox/, examples/, tests/ and docs/: no stale consumer
  remains. The hand-rolled examples' dicts were renamed too.

**4. Pairing audit — holds.**
- `git diff --stat tests/oracle_sweeps` is empty.
- The O2 sweep passes in both tiers (results under 9).
- The C.15 audit sites pass in the full suite: U7a, `test_uniform_fitting` newton_cg/gd_matches_ragged, test_sharing
  :1281.
- Extra check: the ragged vs uniform reference is layer-independent from a zero start, a zero-padded restart and a
  TT-slack start (rel 0 / 0 / 1.1e-16).

**5. Tests T26/T29 — non-vacuous and non-circular.**
- The pinned draw equals the plan recipe computed independently (0.0). Its `R` diagonal has a negative entry, so the
  sign fix is exercised.
- Eight mutations on a scratch copy; every one is caught:

  | mutation | caught by |
  |---|---|
  | no sign fix | `test_pinned_draw` |
  | one sequential stream | independence, nesting, group-key and pinned tests (4 failures) |
  | v5 `N×n` draw | nesting + pinned |
  | key = max(g) | group-key test |
  | coordinate-aligned U | non-collapse test |
  | packed masks returned | uniform-twin test |
  | `L[d-1] ≠ 0` | validity test |
  | seed drift | pinned test |

- Tests are per-test seeded, the numerics are numpy-only, and jax dispatch is in `test_dispatch.py`.
- Gaps: MINOR-2 and NIT-1.

**6. Docs (I-29) — complete for S0d.**
- §5, rank_continuation, sharing and the numerical_contracts caveat row all state:
  - what the reference is;
  - that it is a convention, good to an O(1) factor;
  - why Haar and not coordinate-aligned (the exact-zero trap);
  - that it is identical across layers and nested across levels.
- The doc's ~1.7× figure matches the shipped recipe's seed spread: 1.62× generic, 1.69× trailing.
- The §5 doctest runs.
- The CHANGELOG has §G's lines; its table is G0P §0 verbatim.
- Nothing promised for S13 is claimed as done: curvature_internals, deferred_and_rejected, release_notes, the
  catalog and CLAUDE.md are left for S13, as the handoff says.

**7. Judgment calls.**
- **(a) Accepted.** Plan §G "Examples" names exactly these two sites, and keeping them would contradict the rewritten
  docs. I re-ran the example: the shared best level is unchanged; "35% lower" (was 30%) is inside the documented
  ~25–35%, at its edge. 14h reaches 3.1e-10 at the true ranks.
- **(b) Accepted.** C.10 says "the output masks are the template's". The prefix check is a correct structural guard
  (frames from orthogonalisation always carry prefix masks), and `seed=` mirrors the ragged signature.
- **(c) Accepted.** Both raises are structural. The admissibility conditions are necessary for any orthonormal frame,
  so they cannot reject a valid one.
- **(d) Consistent** with §C.9 ("the final converged row carries it too") and C.14 ("on the final row").
- **(e) Consistent** with §A, C.13 and C.15. `g0norm_cg` has no meaning without CG.
- **(f) Confirmed.** The draw reads only shapes and never touches the global RNG. The result is cast to the
  template's dtype and backend, and the uniform_jax path works.

**8. House style — clean.** There is no formatter churn. The signature comments follow the grammar. The frontend
`**kwargs` is left in place for S0e, with only `g0_reference` added explicitly. `backend/optimizers.py` stays T3-free:
only `warnings` was added.

**9. Gates (reviewer runs).**

| gate | result |
|---|---|
| touched backend tests (test_zero_reference, test_optimizers, test_optimizer_display) | 61 passed, 80 subtests |
| `pytest --doctest-modules t3toolbox` | 196 passed |
| page doctests: every `docs/*.md` + `docs/contributor/*.md` (minus doctest_style.md) + `getting_started.rst` | all pass (`fitting_and_optimization.md` 62 examples, getting_started 81) |
| full suite `-n 2`, half 1 (`tests/backend` + `tests/test_[a-m]*`) | 370 passed, 4,292 subtests, 183 s |
| full suite `-n 2`, half 2 (`tests/test_[n-z]*`) | 551 passed, 1 skipped, 38,955 subtests, 140 s |
| **full suite total** | **921 passed / 1 skipped / 43,247 subtests** (= the implementer's) |
| `T3TOOLBOX_SLOW_TESTS=1 pytest tests/test_oracle_sweep.py -n 2` | 5 passed / 12,790 subtests (= the implementer's) |
| changed examples (`fit_hilbert_uniform_newton_cg`, `fit_hilbert_tensor_newton_cg`, `fit_hilbert_uniform_probe_derivatives_newton_cg`, `fit_varied_rank_tensor_newton_cg`, `fit_shared_factors_jetted_probes`) | all run; the first three byte-identical to the implementer's logs |
