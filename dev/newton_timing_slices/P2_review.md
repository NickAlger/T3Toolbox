# P2 review: the `newton_cg` side of phase timing (Newton timing, 2026.5.0)

Reviewer's record. Base HEAD `89ac203f` (P1 committed in `cbcdc1d6`); P2 in the working tree, uncommitted. Read:
`CLAUDE.md`; the HANDOFF top block; the plan (draft 4) and its reviews r1, r2; `P1_handoff.md`, `P1_review.md`;
`P2_handoff.md`; the switch plan §3; `signature_style.md`, `naming_rules.md`, `parameters_not_closures.md`,
`testing_strategy.md`; `TuckerTensorTrain.squash_tails` / `resize` and `t3_squash_tails` / `t3_segment` (code shape);
the whole diff (`git diff -w` for `newton_cg`'s re-indented body) and the code around it (`Problem.local_model`,
`_alignment_options`, `_resolve_g0_reference`, `_lift_diagnostics`, `_initial_radius`, jax 0.10.2's and 0.4.30's
`jax/_src/monitoring.py`).

## Round 1

### What I ran (my own, one job at a time; current env py3.11 / numpy 2.4.6 / jax 0.10.2 unless stated)

Scripts and logs: the session scratchpad, `p2_review/`.

| check | result |
|---|---|
| **Bitwise vs HEAD** (`battery.py`, `compare.py`; a worktree of `89ac203f` and the working tree). Records x, the whole `stats` dict, every `NewtonInfo` row the callback sees (all fields but `timing`, with the row's `LocalModel` frame / sweep / residual / companion / curvature), the display's records and text (the `%.2fs` cell masked), `stats` keys and each history row's keys in order; container types, dict key order, dtypes and float bits (`tobytes`). numpy, ragged and uniform, unshared and shared `(0, 1, 1, 0)`, probe and probe-derivative kinds: `local_model` (GN, `frame=`, exact, exact with `'EXACT'` + damping; parts plus gradient, `hvp`, quadratic, misfit, objective, a retraction); `newton_cg` GN line search, GN trust region, `padded_from` + `'cone'`, `padded_*` options, `cone_*` options, a given `start_frame`, warm start + `g0norm_cg`, `g0norm_newton`, Armijo exhausted under `'stop'` and `'accept'`, converged, `max_newton=0`, exact TR with rejections, exact line search, exact + `padded_from`, exact + `padded_from` + `'cone'`; `padded_start` (defaults, `tol=0`, the zero tensor); `gradient_descent` plain and `'cone'`. `test_optimizers`' problem: manifold GN / exact TR / exact `'zero'`, corewise GN / exact, regularized GN / exact, `_OffsetProblem` driven to `trust_region_floor` (19 rejections), `'cone'`, each also without a callback. Nine structural-error messages (the corewise `padded_from` / `padded_start` / bad-option precedence / `'cone'` paths; `local_model`'s three). jax x64: `use_jit` GN, `padded_from` + `'cone'`, exact TR with a rejection, both layers, both kinds; eager ragged GN, `'cone'`, exact, exact `local_model` | **numpy 222 entries, 23,641 arrays, 0 mismatches; jax 17 entries, 1,615 arrays, 0 mismatches.** The cases do exercise what they name (rejections on every exact-TR case, `ls_failed` rows on both exhausted policies, converged rows, the floor). A planted one-ulp change is caught |
| `_local_model_phases` vs the former `local_model`, by reading | the body is the old one with `self` -> `problem`; `_mark(None)` / `_lap(None, ...)` return before any reading or synchronization. The battery's `local_model` entries (both modes, every geometry) and every `newton_cg` row's model parts are bitwise HEAD's. The timed path passes the same `hessian` / `curvature_damping` the untimed `model_options` carry (`newton_cg` normalizes `hessian` first) |
| **Timing on jax** (`timing_checks.py`): ragged and uniform `use_jit` exact TR + `padded_from` + `'cone'` + a rejection, first on empty caches then cached; eager ragged GN | per row Σ phases == `window` exactly; no negative remainder; the `total` identity exact; **compile ≤ seconds in every phase of every row** (no double counting, consistent with the three events not nesting); **the compile attributed by the run (Σ `stats['timing']['compile']`) equals an independent listener's total over the call** (40.1671 s ragged, 44.6679 s uniform; 0 on the cached runs), so every compile second is attributed exactly once. Under the uniform whole-loop jit `alignment_passes` is `None` and the alignment is timed whole (16.2 s, 15.6 s compile) |
| what the model laps wait on, jax | every leaf of `frame`, `sweep`, `residual`, the `CurvatureState` and the whole `LocalModel` is a jax array (`Problem`, `LocalModel`, `CurvatureState` are registered pytrees; the uniform frame's host masks and ints are its only non-jax leaves), so no lap's wait is a silent no-op |
| listener hygiene, jax | after every run the listener count is the baseline: also after a nested timed `use_jit` run inside a callback (inner listener gone when it returns; the outer's row-1 `previous_callback` = 7.578 s, compile 6.198 s, against the inner run's `total` 7.577 s and compile 6.198 s: the documented process-global caveat, exactly), after a padded start that raises inside the `try` (corewise `padded_from`), after `max_newton=0` and after a run that converges on row 0 (whose record carries the one-offs) |
| **Mutations** (`mutate.py`, on a scratch copy, one at a time, the timing test classes rerun; mine, in addition to the implementer's 33) | numpy 15: **8 caught** (laps not reset at a close; the next lap not restarted at the close; trust-region trial lap waits on nothing; `previous_callback` from row 2; the reference lap closed before the cone runs; the override guard reversed; the first TR radius lapped as `cg`; `gradient_descent` asking for timing), **7 missed** (below). jax 3: **1 caught** (the compile remainder against the last lap), **2 missed** (below) |

### Plan fidelity

Every §7 P2 row is built as specified: `PhaseSeconds` (fourteen fields in run order, `tracking` after
`trial_objective`, always `None`) and `NewtonTiming` in `__all__`; `NewtonInfo.timing` last, kept out of
`_NEWTON_SCALAR_FIELDS` (equal to 2026.4.0's tuple, checked by T1 and by my battery); `_CompileSeconds`, `_PhaseClock`;
`_local_model_phases` with `Problem.local_model` delegating (docstring and doctest stay on the method);
`_padded_start`; `_cone_reference(timing)` returning the triple, `gradient_descent` passing `False`; the laps;
`stats['timing']` with §3.2's keys in §3.2's order, added last in `stats`. The override guard is the plan's
`type(problem).local_model is Problem.local_model`; the member guard names `timing` explicitly (positional-or-keyword
or keyword-only; `**kwargs` and an unreadable signature count as absent). Nothing from P3 is pulled forward:
`optimizer_display.py` and the frontend `optimizers.py` are untouched. Nothing public is renamed; the new private
helpers (`_alignment_member`, `_aligned_padded_frame`, `_mark`, `_lap`, `_phase_seconds`, `_timing_stats`,
`_CompileSeconds.register` / `unregister` / `EVENTS`) sit where the plan is silent.

**The fifteen judgement calls.** All sound; none needs to go back. Several are plan readings the session should record
as draft 5 (list in the session reply).

1. *The clock's private interface* (`close(t)`, `mark()` without `t`, `row(t, ...)`, the constructor taking the entry
   reading). Sound and better than the sketch: one mechanism ends every stretch, so `setup`, `last_callback` and each
   `remainder` are one computation, and `window` is computed from the same two readings as `wall_time`, so T2's
   `window == wall_time` holds with `==`. My jax runs: tiling and the `total` identity exact.
2. *`_mark` / `_lap` accepting `clock=None`.* Sound: it is what makes `local_model`'s call the same operations in the
   same order, and the eighteen call sites stay one line each.
3. *The listener registered just after `_prepare_jit_inputs`.* Sound: §2.3's "at entry" and "`tree_contains_jax(x0)`
   after `_prepare_jit_inputs`" cannot both hold literally; registering where the test can be applied is the only
   reading. `setup_compile` then covers registration to the loop (the conversion is a `device_put`, no compile; my
   independent listener agrees to 4 decimals).
4. *`previous_callback` lapped at the top of the next iteration*, so it also holds the stop test and `x, lm = x_trial,
   None`. Sound (microseconds), and `last_callback` is the same stretch for the last row.
5. *`try` / `finally` around the body.* Sound and necessary. `git diff -w` shows only the real changes; the comment
   columns survive the re-indentation (42 of 44 commented lines at their HEAD column; the two that moved were
   staircases at HEAD).
6. *The first TR radius as a second `forms` lap before `cg`.* Sound: W-7 puts `_initial_radius` in `forms`; laps sum
   by phase.
7. *What each lap waits on.* Sound, and confirmed on jax (above).
8. *`trials` = `bt + 1`; `accepted_retraction` the last trial's lap when accepted* (so `'accept'` after an exhausted
   search reports it). Sound; see M3 for the test.
9. *The compile column mirrors `seconds`' `None` pattern.* Sound.
10. *`padded_start` keeps its error precedence* (`_alignment_member` first, `_padded_start` repeats it for
    `newton_cg`). Sound: my battery's corewise `padded_start(max_passes=-1)` raises today's member message, and
    skipping the second `_alignment_options` pass on `newton_cg`'s path is safe (the function is idempotent on its
    own output and the first pass already raised for every condition the second checks).
11. *The converged row's window read only with timing.* Sound (non-breaking).
12. *Backend docstrings in P2.* Sound: a keyword that ships between slices should be documented; §6's pages, the
    frontend docstring and the CHANGELOG stay P3's. P3 should extend, not rewrite, the `newton_cg` **Timing** paragraph.
13. *Test placement and cost.* Sound; the in-method `sys.path.insert` in `test_dispatch.py` has nine precedents in
    that file.
14. *The two extra tests.* Valuable: they are what catch mis-attribution (`test_each_phase_is_its_own_work`) and
    which outputs are waited on. M2 asks for one more property of the second.
15. *The example* (`newton_cg`'s docstring problem; §4.13 does not exist yet). Sound.

### Non-breaking

Confirmed by my own battery (above). With `timing=False`: no reading, no synchronization (`_mark` / `_lap` return at
once; the setup wait and the listener are under `if timing`), `problem.local_model` called exactly as today, the
member called without the keyword, `stats` and every history row bitwise HEAD's with HEAD's keys in HEAD's order, the
display text and records unchanged, the structural-error messages unchanged.

### Correctness of the timing

- **The laps partition each window.** Every stretch of code between a close and the next close is either inside a
  lap (from a `_mark` or the previous lap's synchronized end) or in `remainder`; nothing is counted twice (the pass
  tuples are breakdowns, never added; `alignment` is a lap of the setup stretch, not of `setup`). The `remainder`
  holds exactly §2.1's list (the forcing term, `-g`, `‖x‖`, `step_rel`, the TR radius update and the one data norm,
  bookkeeping), which `test_each_phase_is_its_own_work` pins by tracing `cw.corewise_norm` / `corewise_scale`.
- The accepted trial's share, the callback (`previous_callback`, and the last row's in `last_callback`), an
  overridden `local_model` (whole build in `frame`, the split phases `None`), the padded start (`alignment`, its passes
  P1's tuple), the cone (row 0's `reference`, `reference_passes` its breakdown): each where §2.1-§2.2 put it.
- **Synchronization** only under `timing=True`, at every lap that ends on arrays; the laps ending in `float(...)`
  pass nothing. The order inside `lap` is right (wait, then read), but nothing tests it (M2).
- **Compile accounting**: the three events, one number per phase, the remainder's twin; `None` on numpy and on 0.4.30
  (no `unregister_event_duration_listener` there; the `hasattr` check precedes registration); unregistered in the
  `finally` on every exit (no early `return` exists inside the `try`). The nesting claim holds empirically (compile ≤
  seconds everywhere; attributed == independent total).
- **`stats['timing']`** has §3.2's shape; `NewtonInfo.timing` is `None` when off and absent from the history.

Considered and not raised: `unregister_event_duration_listener` asserts membership, so a callback that calls
`jax.monitoring.clear_event_listeners()` would make the `finally` raise; contrived, and swallowing it would hide a real
misuse.

### House style

Signatures shape-annotated and aligned (`_local_model_phases`, `_padded_start`, `_aligned_padded_frame`, the clock's
methods, `_timing_stats` with its long `last` row in its own group); the two records' field comments aligned like
`NewtonInfo`'s; `inspect` imported in order; `__all__` updated; the clock is passed to `_local_model_phases` as a
parameter, never closed over; the two classes say why `parameters_not_closures.md` does not apply. No debugging
residue. Docstrings say what each piece does and when it does nothing. One cosmetic NIT (N3).

## Findings

### MINOR

**M1. The temporary mirror exemption is not self-expiring** (`tests/test_optimizers_frontend.py`,
`test_newton_cg_keywords_mirror_backend`). The session accepts the temporary line on the condition that it fails the
suite once P3 adds the frontend keyword without deleting it. As written, `not_yet` only skips `timing`, so a P3 that
adds the keyword and forgets the line stays green. **Fix:** also assert the exemption is still needed, e.g.
`for name in not_yet: self.assertNotIn(name, front, '%s is now a frontend keyword: delete not_yet (the timing plan, P3)' % name)`,
and give `P2_handoff.md` a short **"P3 must delete"** list naming this line (and the test that will fail if it stays).
Check: adding `timing: bool = False` to the frontend `newton_cg` signature on a scratch copy must turn this test red.

**M2. No test sees a lap that reads the clock before it waits.** Moving `tree_block_until_ready(outputs)` after the
`perf_counter()` reading in `_PhaseClock.lap` keeps every test green, numpy and jax (my mutation; on jax it is exactly
the bug the waits exist to prevent: the async tail lands in the next lap). `test_each_phase_waits_on_its_outputs`
checks which trees are passed, not that the wait is inside the lap. Three branches are also outside that test: the
line search's `trial_retraction` (only exact + TR is run), the `alignment` lap, and an override's `frame` lap
(mutations "waits on nothing" for each: missed). **Fix (suggested):** in that test, a mock that sleeps a fixed δ (say
20-50 ms) on every call and records the calls; assert every waiting phase's seconds ≥ δ × its waits on that row and
the `remainder` < δ (so no wait leaks out of its lap); add a Gauss-Newton line-search run, a `padded_from` run and an
`_ScaledModelProblem` run, so the three branches are covered. Criterion: the four mutations above fail.

**M3. `accepted_retraction` on a backtracked step is not pinned.** Reporting the whole row's `trial_retraction` as the
accepted share passes, because `check_timing` asserts only `0 < accepted ≤ trial_retraction` and equality satisfies
it. **Fix:** on a row with `trials > 1` and an accepted step, assert `accepted_retraction < seconds.trial_retraction`
(strict; every lap is positive). The `'exhausted, accept'` case (40 trials, the last accepted) already exercises it.

**M4. The listener's unregistration on success is not tested.** In `TestNewtonTiming.test_eager_jax` (`test_dispatch.py`)
the baseline count is taken after the first timed run, so a `newton_cg` that unregisters only on an exception (my
mutation: `finally` -> `except BaseException: ...; raise`) passes. **Fix:** take the count before the first timed
run and assert it after the successful run as well as after the raising one.

### NIT

**N1.** `_aligned_padded_frame`'s "an unreadable signature counts as absent" branch is untested (removing the
`try`/`except` stays green). One case: a member object whose `__signature__` is not a `Signature` (so
`inspect.signature` raises `TypeError`) wrapping the stock member; the timed run equals the untimed one with
`alignment_passes is None`.

**N2.** No timed run converges on row 0, so the converged row's `**(one_offs if it == 0 else {})` is untested (dropping
it stays green). Adding e.g. `'converged at row 0': (dict(), dict(gtol_rel=10.0))` to `TestNewtonTiming.CASES` lets
`check_timing` see the one-offs on a converged row 0.

**N3.** `# timing: this trial's share` sits on the line-search `retraction = _lap(...)` only, as a staircase (column
76 in a block at 73); its trust-region twin has none. Drop it, or put it on both at the block's column.

**N4 (optional, the implementer's call).** The override guard compares against `Problem.local_model` as it is at call
time, so a class-level patch of `Problem.local_model` (`unittest.mock.patch.object(Problem, 'local_model', ...)`)
passes the guard and is bypassed under timing. Capturing the stock function once after the class
(`_STOCK_LOCAL_MODEL = Problem.local_model`) and comparing with that would honour it too. An instance-level override is
already impossible (`Problem` is frozen). Not a plan change either way; leave as is if you prefer the plan's literal
form.

VERDICT (round 1): REVISE (four MINOR, all test-side; no library code path needs to change)

## Round 1, continued: the runs finished after the findings went out

| check | result |
|---|---|
| full suite on the round-0 tree (a byte-identical scratch copy, so the implementer could edit meanwhile), `pytest tests/ -q -n 3` | **1,368 passed, 5 skipped**, 48,856 subtests, 0 failed (22 min); the one warning is the pre-existing adam-on-manifold one |

## Round 2 (the implementer's `P2_handoff.md` §6-§7)

What changed: tests, the handoff, and one library line (N4). I diffed the tree against my round-0 copy file by file: in
`t3toolbox/` only `_STOCK_LOCAL_MODEL = Problem.local_model` (after the class), the guard comparing with it, and N3's
dropped comment.

| finding | resolution | checked by me |
|---|---|---|
| M1 | `for name in not_yet: self.assertNotIn(name, front, '%s is now a frontend keyword: delete not_yet (the timing plan, P3)')`; `P2_handoff.md` §6 "P3 must delete" names the line, the loop and the `or name in not_yet` | a scratch copy with `timing: bool = False` added after the frontend's `cone_seed`: **red, at the `assertNotIn`** with that message |
| M2 | `test_each_phase_waits_on_its_outputs` rewritten: the mocked wait sleeps 30 ms and is recorded with the lap it ends (rows split at each `close`); every phase ≥ 30 ms per wait it ended, each row's `remainder` < 30 ms, `setup` ≥ its own wait and < 2 waits with a padded start; four runs (exact + TR with the full per-row lap sequence, GN line search, a padded start, the override) | **caught**: the sync after the reading, the line-search trial / alignment / override frame lap waiting on nothing. The jax class alone still misses the sync-order mutation; the numpy test covers that shared code path, so no further test is needed |
| M3 | `check_timing`: `accepted_retraction < trial_retraction` on accepted rows with several trials, `==` with one | the whole-sum mutation: **caught** |
| M4 | the listener count taken before the first timed run, asserted after it and after the raising run | `finally` -> `except BaseException: ...; raise`: **caught** (jax) |
| N1 | `_UnreadableMember` (a member whose `__signature__` is not a `Signature`) joins `test_a_member_without_timing` | no `try`/`except` around the signature read: **caught** |
| N2 | `'converged at row 0'` (`gtol_rel=10.0`) in `TestNewtonTiming.CASES` | the converged row without the one-offs: **caught** |
| N3 | the comment dropped | read |
| N4 | taken: `_STOCK_LOCAL_MODEL`, read on the timed path only; `test_a_patched_local_model_is_called_whole` | the guard against the live attribute: **caught**. This departs from §2.5's literal expression for §2.5's own rule ("timing never bypasses an override"): a draft-5 line |

Round-2 runs (mine, one job at a time, on the round-1 tree):

| check | result |
|---|---|
| mutations (`mutate_r1.py`, a fresh copy) | **numpy 17 of 17 caught** (round 1's 15, the N4 guard, the M1 self-expiry); **jax 2 of 3 caught** (M4's; the compile remainder), the third being the sync order the numpy test catches |
| the bitwise battery again (N4 touched the library) | numpy **222 entries, 23,641 arrays, 0 mismatches** vs HEAD |
| `test_dispatch.py -k TestNewtonTiming` (current env) | **3 passed**, 11 subtests (88 s) |
| touched files, current env, `-n 3` (`test_optimizers`, `test_padded_start`, `test_optimizers_frontend`, `test_dispatch`, `test_jax_absent`) | **212 passed, 1 skipped** (the slow row), 1,739 subtests (10.5 min) |
| slow tier, `T3TOOLBOX_SLOW_TESTS=1 ... test_the_remainder_is_small` | **passed** |
| module doctests, `pytest --doctest-modules t3toolbox -n 3` | **241 passed** |
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, no warning**; `autoapi/.../optimizers/{NewtonTiming,PhaseSeconds}.html` rendered, the private helpers absent |
| floor env `tttt` (py3.9, numpy 1.22.4, jax 0.4.30), single process: the same five files | **211 passed, 2 skipped** (26 min): the slow row, and the pre-existing `test_eager_jax_compiles_nothing_twice` (0.4.30 cannot unregister a listener). `TestNewtonTiming` runs there on its `None` branch |

The full suite ran on the round-0 tree. Round 1 changed the library by one line, read only on the timed path, and the
touched files were rerun on both envs. The session's pre-commit full suite covers the rest.

**For the session (plan text, draft 5):** P2's readings: the clock's private interface (`close(t)`; `mark()` without
`t`; `row(t, ...)`, so `window` comes from `wall_time`'s own readings) and the helpers; the listener registered just
after `_prepare_jit_inputs`; `previous_callback` also holding the stop test and the accepted bookkeeping; the first TR
radius as a second `forms` lap; the override guard against `_STOCK_LOCAL_MODEL`; the backend docstrings written in P2
(P3 extends them); the temporary, self-expiring mirror exemption that P3 deletes with the frontend keyword; the tests
beyond §5; the review stop's example.

**Files for the P2 commit:** `t3toolbox/backend/optimizers.py`, `tests/backend/test_optimizers.py`,
`tests/backend/test_padded_start.py`, `tests/test_dispatch.py`, `tests/test_optimizers_frontend.py`,
`dev/newton_timing_slices/P2_handoff.md`, `dev/newton_timing_slices/P2_review.md`.

VERDICT: APPROVE
