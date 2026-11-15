# W1 review: the backend of `newton_cg(hessian='switch')` (the Gauss-Newton-to-exact switch, 2026.5.0)

Reviewer's record. Base HEAD `c1a42c24`; W1 in the working tree, uncommitted (`t3toolbox/backend/optimizers.py`,
`tests/backend/test_optimizers.py`, `tests/test_dispatch.py`, `tests/test_optimizers_frontend.py`, this directory).
Read: `CLAUDE.md`; the HANDOFF top block (state, the machine rule, G-1..G-7 and the nine switch answers); the plan
draft 3 (§1, §2, §3, §5, §8 W1), its reviews r1 / r2 and the source notes; `W1_handoff.md`; the timing feature's
`P2_review.md` / `P3_review.md` (method); `signature_style.md`, `naming_conventions.md`, `parameters_not_closures.md`,
`testing_strategy.md`; `Problem.local_model` / `_local_model_phases`, `LocalModel`, `_initial_radius`,
`_resolution_eps`, `_cg_solve` and the whole `newton_cg` body; the shared geometry's `project`.

## Round 1

### What I ran (my own, one job at a time; current env py3.11 / numpy 2.4.6 / jax 0.10.2 unless stated)

Scripts and logs: the session scratchpad, `w1_review/`. The reference is a `git worktree` of `c1a42c24` (`wt_ref`).

| check | result |
|---|---|
| **Bitwise vs `c1a42c24`** (`battery.py` = the P3 reviewer's battery, the five switch fields dropped from each row's record and asserted `None` on every row, the four `switch_*` parameters left out of the signature record; plus my own section: **timed runs** on numpy -- GN line search / trust region, exact TR with rejections, exact line search, corewise exact, converged, exhausted `'accept'`, `_OffsetProblem` under GN-TR and exact, padded + cone exact on both layers -- recording x, `stats` but `timing`, every row, each record's `None` pattern, `stats['timing']`'s keys, rows' keys and phases' keys; timed `use_jit` GN-TR / exact TR on both layers; and **27 structural-error messages**: `newton_cg`'s hessian (bad / bool / `None`), damping with GN, the radius rules (line search, negative, NaN), `'accept'` with TR, step control (bad, int), `on_line_search_failure`, `g0_reference`, `g0norm_newton`, padded / cone options without their group, `start_frame` with `padded_from`, a kind without `corewise_hessian` and a geometry without `lift` under `'exact'` (in `newton_cg` and `local_model`), `local_model`'s hessian (bad, `'switch'`, bool) and damping, `_check_step_control`, `_check_hessian`'s default message; the frontend's factory / `least_squares_problem` / `newton_cg` with `'switch'`, `switch_tau=` and damping) | **numpy 347 entries, 30,276 arrays: 3 mismatches, all the one expected message** (`newton_cg`'s invalid `hessian` -- a string, a bool, `None` -- now listing `'switch'`). **jax 31 entries, 2,443 arrays: 0 mismatches.** Every non-switch row's five switch fields are `None`; the history keys, `stats` keys and their order, display texts and every other message are bitwise HEAD's. A planted one-ulp change is caught |
| **The view is today's Gauss-Newton model** (`view_check.py`): `_gauss_newton_view(local_model(x, 'exact', damping))` vs `local_model(x)`, ragged / uniform × manifold / corewise / shared, three regularized cases, damping `None` and `1e-2` (not corewise): 15 checks each | **0 failures**: gradient (the exact build's and the view's), `hvp`, `quadratic`, `gn_quadratic`, `retract`, misfit, regularization, objective, the first radius, a CG solve, all bitwise; the jax treedef equal; the jitted CG result bitwise; the exact `hvp` differs (sanity) |
| By reading | `_local_model_phases` builds frame, sweep, companion and residual identically in both modes; exact mode only adds the `CurvatureState`. `dc.replace(curvature=None)` keeps every other field, so `hvp` takes the GN branch, the gauge / regularizer / companion are the GN model's and the damping lives only in the dropped state. `switch_tau=0` is therefore today's GN run with the same step control, and T1 shows it bitwise on 7 cases × 2 step controls |
| **Compile budget** (`compile_count.py`, fresh process, x64, `use_jit`; jax monitoring listener on every backend compile) | a firing switch run at rank 2: **+2 CG programs** (GN + exact), 279 backend compiles cold; the same run again: **+0 programs, 0 compiles**; the zero-padded rank-3 level: **+2**, again **+0**; a `switch_tau=0` run afterwards **+0**. At most two per level, none repeated |
| **The printed run** (`example.py`, numpy) | reproduced: `switched_at = 2`, fired on row 1 (ratios 0.0128, 0.0030), 6 Newton steps; GN-TR 25, GN line search 29, exact at power 1.0 5, exact at 0.5 6; CG iterations 30 vs GN-TR's 90 vs exact's 27; the count recomputed from the rows' own `e_N`, `e_GN`, `tr_accepted` agrees on every row |
| **Mutations** (`mutate.py`, a scratch copy of the W1 tree, one at a time, the targeted tests rerun; mine, beside the implementer's 47) | **36 run: 29 caught, 7 missed.** Missed: three real gaps (M1, M2), one T5-alone (M3; the suite catches it), one performance-only equivalent (N1), two equivalent mutants (a `tr_accepted` term implied by `accepted`, judgement call 14; `not switch_tau` for `switch_tau == 0`). Caught: the view without the regularizer, with a geometry-data-free curvature (numpy: `is_exact`; jit: 2 programs), GN `pHp` from the tracking model; the gate deleted (suite), on `e_n` only, `min` for `max`; the floor at float64's ε, the floor passed as 0; a rejected trial / backtrack / fallback / forced accept counted; boundary CG resetting; `count += 2`; no reset across a non-countable row; `switched_at = it + 2`, `None` with `k = 0`; the firing row unmarked; the exact phase reverting to the view; the exact-phase rows carrying a count, the converged rows carrying `e_gn` / a count, `switch_fired` `None` on exact rows; untracked rows reporting `e_n`; `stats['switch_track']` never `None`, the keys reordered; `_objective_resolution` without the data norm (the TR floor test); the GN phase at the switch power |
| **The `check_timing` ruling** (`helper_mut.py`): 12 library mutations of the timing code (6 aimed at the two generalized lines -- curvature lapped as gradient, curvature lapped on GN builds, the forms / the first radius lapped as tracking, a tracking lap on every non-switch step row, the curvature lap skipped -- and 6 of the P2 reviewer's that still apply), each run against the pre-W1 timing tests (`TestNewtonTiming`, `TestNewtonTimingAtTheSeam`) twice: with W1's helper and with the helper's two lines reverted to `c1a42c24`'s | **12 / 12 IDENTICAL** (same exit code and the same failing tests in both): all caught by both. By reading: `ran[...]` is a `bool`, so on a row whose `switch_pKp` and `switch_fired` are `None` (every row of every pre-W1 test, which my battery confirms) the new expressions reduce to the old ones exactly. **No existing assertion is weakened: the session's condition holds** |

The suite, module doctests, `sphinx -W` and the floor env run after this round's findings went out (a snapshot worktree,
`wt_w1`, so the implementer's edits cannot race them); results below in the session reply and round 2.

### Plan fidelity

Everything §8 assigns W1 is built as §1-§3 specify, and nothing of W2 / W4 is pulled forward (the display and the
frontend are untouched; the frontend still refuses `'switch'` through `_check_hessian`'s default modes).

- **Options.** `hessian='switch'` (`_NEWTON_HESSIAN_MODES`, case-insensitive); `switch_tau` (None = 0.25, `[0, 1)`,
  0 allowed), `switch_k` (None = 2, integral `>= 0`, 0 allowed, not a bool), `switch_cg_forcing_power` (None = 1.0,
  unchecked, the exact phase's; `cg_forcing_power` the GN phase's), `switch_track` (None = `'countable'`, `'all'`,
  case-insensitive, a non-`str` raises). Every `switch_*` raises outside switch mode; the four unread combinations raise
  (so `(0, 0)` either way); all before `_prepare_jit_inputs` (T7's spy). `_require_exact_capable` at the call, naming
  `'switch'`. `_check_step_control` maps `None` to the trust region for `'switch'`.
- **The loop.** One exact build per GN-phase point; CG, `_initial_radius`, `pHp`, ρ, the radius, the retraction, the
  lift diagnostics (`None`) and the row's `lm` on the view; `g` read from the build (one transpose:
  `test_one_transpose_per_point`). `e_GN = |Δf − predicted|` on every GN-phase step row; on a tracked row one
  `lm_track.curvature_quadratic(p)` and `e_N = |Δf − (predicted + ½α²·pKp)|` (α² verified on backtracked `'all'` rows).
  The gate is `max(e_GN, e_N) > 100 · 10 · _objective_resolution(_resolution_eps(residual), ‖ω⊙y‖, misfit, f)`: both
  errors, the library's dtype-aware scale, the source's constant. Countable is §1.2 verbatim; the count resets on
  exactly §1.2's list and not on a boundary step (a capped step: M2). The switch is one-way, lands on a fresh build,
  carries the radius; `Δ0` / `Δ_max` stay iteration 0's (T3). Tracking on countable rows by default, on every GN-phase
  step row under `'all'`, the switching row identical (T4, `test_reset_rows`).
- **Reporting.** The five `NewtonInfo` fields `None` outside switch mode and in the history only in switch mode
  (`_NEWTON_SCALAR_FIELDS` equals 2026.4.0's tuple); per-row `hessian`; the converged and exact-phase rows' fields as
  §3.1; `stats` gains `hessian`, `switch_track` (`None` with `k = 0`), `switched_at` (`it + 1`; `len(history)` when it
  fired on the last row; 0 with `k = 0`; `None` if never). `PhaseSeconds.tracking` laps the product on tracked rows;
  the GN phase's exact build laps `curvature` (and its transpose `gradient`).

### The judgement calls

All fourteen are sound.

1. *`_require_exact_capable(..., hessian='exact')`.* Sound: the plan asks for a message naming `'switch'` and gives no
   signature; a defaulted trailing parameter keeps `'exact'`'s two messages byte-identical (battery: the kind and the
   geometry paths, `newton_cg` and `local_model`).
2. *The invalid-`hessian` message lists `'switch'`.* Inherent; the only battery difference (I added `None` as a third
   input: same message).
3. *The damping message kept.* Sound by the non-breaking rule. Note for the maintainer: the frontend test's regex
   (`test_optimizers_frontend.py:726`) is a prefix, so a widened message ("... with hessian='exact' or 'switch'") would
   not break it; the wording is purely his call.
4. *Option types.* Sound (`numbers.Real` / `Integral`, bool refused, NaN out of range; the power passed through as
   `cg_forcing_power` is).
5. *Placement.* Sound.
6. *Backend docstrings now.* Sound (P2's precedent); one accuracy fix, M4.
7. *`check_timing`.* Acceptable under the session's condition, which my comparison confirms (12 / 12 identical). A
   cleaner form is N2.
8. *T4's two devices.* Sound. `forced_hits` replaces only the hit decision; the tracking decision, `e_GN` / `e_N`, the
   `countable` term and the count update are the real code, which is why my reset and count mutations are caught. The
   CG stubs produce real backtracks and real fallbacks (the library's own branches run). What the device cannot show is
   a miss, since every tracked row is a hit: M1.
9. *T5's recorded floor.* Sound for what it targets (the dtype-aware floor: my float64-ε mutation is caught), but see
   M3.
10. *T6 on `newton_cg`'s docstring problem.* Sound (the same structure as §4.12's; W4's doctest uses §4.12's).
11. *G-bit as the battery.* Sound (digests would fail on CI's other legs); my own battery agrees.
12. *The extra tests.* Valuable (the transpose count is what catches the gradient read from the view).
13. *T9 split by layer.* Sound.
14. *The equivalent mutants.* Confirmed: my `tr_accepted → True` alone is equivalent; dropping both terms is caught.

### House style

Signatures shape-annotated and aligned (`_check_hessian`, `_require_exact_capable`, `_check_switch_options` with its
return block, `_objective_resolution`, `_switch_sample`, `_gauss_newton_view`); the new `newton_cg` group spelled and
aligned as the `padded_*` / `cone_*` groups; the `NewtonInfo` field comments in their column; `numbers` imported in
order; a banner for the switch's constants and helpers; every new private helper has a docstring saying what it does
and why (the view's says why its `gradient` is not read). `_SWITCH_FIELDS` sits beside `_NEWTON_SCALAR_FIELDS` and is
excluded from it; `_newton_scalar_record` appends it iff `switch_fired` is a bool. No debugging residue. The
self-expiring mirror exemption fails the suite as soon as any one of the four keywords reaches the frontend
(`assertNotIn` per name) and is listed in the handoff's §6 with the frontend regex (N3: its line number).

## Findings

No MAJOR finding. The algorithm is the plan's, non-breaking holds bitwise, the compile budget holds.

### MINOR

**M1. The count's reset on a countable miss is untested in the loop.** Two of my mutations stay green: "a miss on a
countable row does not reset" (`... else (count if countable else 0)`) and "an uninformative countable row does not
reset". `forced_hits` makes every tracked row a hit, `test_switch_sample` tests the sample alone, and the unmocked runs
never put a miss after a hit (their counts are 0 until the firing pair). "Consecutive" is the rule's core.
**Fix:** a scripted `_switch_sample` (the `forced_hits` pattern, one `(informative, hit)` per tracked row): hit, miss,
hit, uninformative, hit, hit on `_pd_problem(0)` from `_rank2_start()` (all six rows countable under the default trust
region; checked: counts `[1, 0, 1, 0, 1, 2]`, fires on row 5, `switched_at = 6`). Criterion: both mutations fail.

**M2. A capped CG step is not tested to count.** §1.2's table puts "CG stopped on the boundary, truncated, or at its
cap" under "no special effect"; T4 tests the boundary only, and "capped CG resets" (`... and (cg_converged or
cg_boundary)`) stays green. (Truncation needs nonpositive curvature without a radius, which a GN model does not reach,
so the cap is the testable case.) **Fix:** in `test_what_counts_and_what_resets`, a forced-hit run with
`cg_maxiter=2, g0norm_cg=1e12` under each step control (checked: every row capped -- not converged, not on the
boundary, not truncated -- accepted, and the count climbs 1..6). Criterion: the mutation fails.

**M3. T5 does not test the gate it is named for.** Deleting the gate from `_switch_sample` leaves T5 green (my run:
"gate deleted, T5 alone" passes); `test_switch_sample` catches it, so the suite does. T5's spy records only the
`floor` argument. The plan's criterion is vacuous here (the handoff's §3 shows why), and the spy is one step from
making it real. **Fix:** the spy also records the real sample's `informative`; on every tracked row assert it is
`False` where `max(e_GN, e_N) <= 0.99 · expected` and `True` where it exceeds `1.01 · expected` (the margin absorbs
the float64-vs-float32 `‖y‖`), and that at least one row is uninformative. Criterion: "gate deleted, T5 alone" fails.

**M4. Two docstring sentences overstate the boundary cases.** `newton_cg`'s switch paragraph: "``switch_tau=0`` tracks
and never switches (the run is the Gauss-Newton run, bit for bit, plus the tracking); ``switch_k=0`` is exact from
iteration 0 (the exact run)". With the defaults neither holds: switch mode's step control defaults to the trust region
(Gauss-Newton's to the line search), and `switch_k=0` runs at `switch_cg_forcing_power` = 1.0 (exact's default is
0.5). W4 will lift this text into §4.14. **Fix:** "(the Gauss-Newton run with the same step control, bit for bit,
plus the tracking)" and "(the exact run at ``cg_forcing_power=switch_cg_forcing_power``)".

### NIT

**N1. Dropping the companion from the view is invisible** (`dc.replace(..., geom_aux=None)` stays green): the
geometries recompute it when `aux` is `None`, so the numbers are equal and only the cost changes, an extra companion
per projection. `test_one_transpose_per_point` could also count `geom.precompute` calls (one per point in both runs)
on the shared case. Optional.

**N2. `check_timing` could take the mode from its caller** (`switch=False`: tracking iff `switch and switch_pKp is not
None`, curvature on `hessian == 'exact' or switch`), so the pre-W1 callers' assertions are the old ones by
construction rather than because their rows' switch fields are `None`. The present form is acceptable (above);
optional.

**N3. The handoff's §6 points at `test_optimizers_frontend.py:732`**; in the W1 tree the regex is at line 736 (W1's
four exemption lines sit above it). Name the test, `TestFrontendExactNewton.
test_options_with_no_effect_and_unbuilt_geometries_raise`, rather than a line.

### For the maintainer (not findings)

- `curvature_damping`'s message with Gauss-Newton still says "only meaningful with hessian='exact'" (now also
  meaningful with `'switch'`); kept byte-identical by the non-breaking rule; widening it breaks no test. *(Superseded:
  the session ruled to widen it in W1; round 2.)*
- The backend's invalid-`hessian` message now lists `'switch'` (the frontend's in W2).
- The printed run is consistent with the rule: two clean accepted trust-region steps with `e_N / e_GN` = 0.013 and
  0.003, so it fires on row 1 and row 2 is the first exact row. The warm start (five Gauss-Newton steps) already sits
  in Gauss-Newton's linear phase, where the curvature term predicts Gauss-Newton's error almost exactly; from the
  random start the first three countable rows are misses (the exact model 3.5-9.6× worse), as the source predicts for
  the transient.

VERDICT (round 1): REVISE (four MINOR: three test-side, one docstring wording; no library code path needs to change)

## Round 2 (the implementer's `W1_handoff.md` §7, and the session's ruling on the damping message)

What changed: tests, one docstring paragraph (M4; re-wrapped to 120 columns, whitespace beyond the two clauses) and,
by the session's ruling, one error message. No library code path. I diffed a `git worktree` of `c1a42c24` with the
round-2 diff applied (`wt_r2`) against my round-1 snapshot (`wt_w1`) file by file; the frontend test file is
unchanged, and N4 (below) touched one test after it.

| finding | resolution | checked by me |
|---|---|---|
| M1 | `test_a_miss_resets`: a scripted `_switch_sample`; after N4, `switch_k=3` and hit, hit, miss, hit, hit, uninformative, hit, hit, hit: counts `[1, 2, 0, 1, 2, 0, 1, 2, 3]`, fired on row 8 only, `switched_at = 9`, every row asserted countable at run time | "a miss does not reset", "an uninformative row does not reset": **caught** |
| M2 | a forced-hit `'capped CG'` subtest per step control (`cg_maxiter=2`, `g0norm_cg=1e12`, `switch_k=50`; the trust region with `tr_radius0=1e3`): every row at the cap, counts 1..6. My round-1 "checked" was wrong for the trust region: with the default radius row 0 stops on the boundary (my own printout showed it); the implementer caught it and widened the radius | "capped CG resets": **caught** |
| M3 | the T5 spy records the real sample's verdict: uninformative at or below `0.99·floor`, informative above `1.01·floor`, at least one uninformative row | "gate deleted, T5 alone": **caught**. (The source's single gate `e_GN > floor` stays green on T5 alone, correctly: §1.3 says its counting equals ours, and the label difference needs a row with `e_GN <= floor < e_N`, which this run has none of; `test_switch_sample` pins the label) |
| M4 | "(the run is the Gauss-Newton run with the same step control, bit for bit, plus the tracking)"; "(the exact run with ``cg_forcing_power=switch_cg_forcing_power``)"; the paragraph re-wrapped | read; no line over 120 |
| N1 | `test_one_transpose_per_point` also counts `ManifoldGeometryOps.precompute`, ragged and shared manifold | "the view drops the companion": **caught** |
| N2 | `check_timing(..., switch=False)`; the pre-W1 callers' two assertions are the old ones by construction; the switch's timed test passes `switch=True` | read: `switch and ...` is `False` and `... or switch` is `hessian == 'exact'` for every pre-W1 caller |
| N3 | §6 names `TestFrontendExactNewton.test_options_with_no_effect_and_unbuilt_geometries_raise` | read |
| session ruling | backend `newton_cg`, `curvature_damping` with Gauss-Newton: `"newton_cg: curvature_damping is only meaningful with hessian='exact'"` → `"newton_cg: curvature_damping is only meaningful with hessian='exact' or 'switch'"`; `Problem.local_model`'s and the `fitting` factories' unchanged (`'switch'` is not a model's mode); T7 pins it; the before / after are in the handoff's §7 | my battery (below); the frontend test's prefix regex and `test_curvature.py`'s (on `local_model`) pass |
| **N4** (new this round, NIT) | `test_a_miss_resets` reset only from a count of 1, so a leaky count passed it (my "count halves" stayed green; "a miss decrements" would by construction). Rewritten as in the M1 row | "count halves", "a miss decrements", and again "a miss does not reset", "an uninformative row does not reset", "firing at `k + 1`": **5 / 5 caught** |

Round-2 runs (mine, one job at a time; `wt_r2` unless stated):

| check | result |
|---|---|
| my battery vs `c1a42c24` (numpy) | **347 entries, 30,276 arrays: 5 mismatching entries, two messages, all expected**: the invalid `hessian` (string, bool, `None`) and the `curvature_damping` message (the backend call and the frontend's pass-through of it). `local_model`'s damping message unchanged. jax not rerun (the round's library change is a message string; round 1: 0 mismatches) |
| my round-2 mutations | 7 run: **5 caught**; the 2 green are the source's single gate on T5 alone (correctly not a T5 property, above) and "count halves", which N4 then closed. N4's: **5 / 5 caught** on the working tree |
| `TestSwitchNewton` on the working tree (with N4) | **14 passed**, 115 subtests |
| floor env `tttt` (py3.9.19, numpy 1.22.4, jax 0.4.30), single process: `test_optimizers.py`, `test_optimizers_frontend.py`, `test_dispatch.py::TestSwitchNewton`, `::TestNewtonTiming` | **118 passed, 1 skipped** (the slow-tier timing row; 5.5 min); N4's test alone on the working tree: **passed** |
| module doctests, `-n 3` | **241 passed** |
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, no warning** (15 min); `newton_cg` / `NewtonInfo` render, the private helpers stay out |
| full suite, `-n 3` (round 1's snapshot) | **1,397 passed, 5 skipped**, 49,003 subtests (22 min); the one warning is the pre-existing adam-on-manifold one. Not rerun on round 2 (tests, one message, one docstring paragraph; the touched files ran on both envs); the session's pre-commit run covers it |

**Mutation totals (mine):** round 1 36 run, 29 caught, 4 non-gaps (2 equivalent, 1 performance-only then closed by
N1, 1 T5-alone then closed by M3), 3 real gaps (M1, M2), all closed; round 2 and N4 12 run, 10 caught, 2 accounted
for above. Every mutation that changes behaviour the plan specifies is now caught by the suite.

**For the session (plan text, draft 4):** W1's readings: `_require_exact_capable(kind, geom, hessian='exact')` (the
mode named in its two messages); `_check_hessian`'s message built from `modes` (byte-identical for the default);
the backend messages that change: the invalid `hessian` (lists `'switch'`) and, by the session's ruling,
`curvature_damping` with Gauss-Newton (`... with hessian='exact' or 'switch'`); option types (`numbers.Real` /
`Integral`, bools refused, NaN out of range, the power unconverted); the five fields appended after `timing`, the
three `stats` keys after the optional keys and before `timing`; the floor's arithmetic in `remainder`, only the
product in `tracking`; backend docstrings in W1, no doctest (W4's §4.14); `check_timing(..., switch=False)`; T4's
devices (`forced_hits`, a scripted sample, the CG stubs); T5's spy (the recorded floor and verdict) supplementing the
plan's criterion, which is vacuous at the float32 floor (`e_N >= e_GN` there); T6 on `newton_cg`'s docstring problem;
G-bit as the battery, not digests; the tests beyond §5 (one transpose and one companion per point, the defaults, the
regularized case, the capped CG step, consecutiveness from a count of 2); T9 split by layer; the equivalent mutants
kept as §1.2 writes them. **W2 must** delete the frontend mirror test's `not_yet` exemption (it fails by itself once
any switch keyword reaches the frontend) and update the `hessian` regex in
`TestFrontendExactNewton.test_options_with_no_effect_and_unbuilt_geometries_raise` when the frontend checks the three
modes.

VERDICT: APPROVE
