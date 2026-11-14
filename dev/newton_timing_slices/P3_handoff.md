# P3 handoff: the display, the frontend keyword, the figures, the docs (Newton timing, 2026.5.0)

Implementer's record for the session. Base: HEAD `a03cb43c` (the plan at draft 5; P1 `cbcdc1d6`, P2 `f8cbb442`).
Nothing is committed. Scope: plan §7 row P3, exactly --

- the display (`format_newton_iter`'s `time` line, `make_newton_display`'s closing table);
- the frontend keyword, its validation and docstring;
- the overhead and compile figures, measured on the maintainer's laptop (§6);
- the docs: `fitting_and_optimization.md` §4.13, `aligned_padded_frame.md`, `scan_body_principles.md`, the docstrings
  §6 lists, CHANGELOG `[2026.5.0]` `### Added`;
- tests T8, T9 (the mirror test's exemption DELETED, `P2_handoff.md` §6), doctests, `sphinx -W`, T10.

**Changed:** `t3toolbox/backend/optimizer_display.py`, `t3toolbox/optimizers.py`, `t3toolbox/backend/optimizers.py`
(docstrings only), `tests/backend/test_optimizer_display.py`, `tests/test_optimizers_frontend.py`,
`docs/fitting_and_optimization.md`, `docs/aligned_padded_frame.md`, `docs/contributor/scan_body_principles.md`,
`CHANGELOG.md`. **New:** this file. No backend code path changed (`backend/optimizers.py` is three docstring edits).

## 1. What was built

| where | what |
|---|---|
| `optimizer_display.py` | `_format_timing(info)`: the `time` line (below). `format_newton_iter` inserts it between the header and the error table when `info.timing` is set; otherwise the string is today's. `_format_timing_table(rows)`: the closing table. `make_newton_display`'s callback keeps `(info.timing, info.cg_iters)` per timed row (cleared at `iteration == 0`, so a callback reused across levels starts a new table) and prints the table through `print_fn` after a run's last row (`converged`, or `stop_reason` set). Records unchanged (`_newton_scalar_record` skips `timing`). Module docstring, `make_newton_display` docstring and a `format_newton_iter` doctest (a hand-built timed row) say so. |
| frontend `optimizers.newton_cg` | `timing: bool = False`, keyword-only, after `cone_seed`. A non-bool raises `TypeError('newton_cg: timing must be a bool, ...')` before the setup work (beside `_check_hessian`); `numpy.bool_` is accepted. Forwarded as `timing=True` only when set (a separate `timed` dict beside `seam`), so an untimed call reaches the backend exactly as in 2026.4.0. Docstring: a **Timing a run** section (purpose; every phase in run order; `stats['timing']`; the display and `NewtonInfo.timing`; the cost with the measured range; the compile column under `use_jit` and eager with measured figures and where it is `None`; the process-global listener; what is not timed, W-5) and a doctest-safe example (the exact-Hessian example's problem one level up: `padded_from` + `'cone'`; checks: rows == history, the phases tile each window, `reference` on row 0 only, `compile is None` on numpy, pass count == movements, the untimed run's losses equal). |
| `backend/optimizers.py` | Docstrings only: the `Geometry` protocol names `newton_cg(timing=True)` as the only caller that asks and that it reports the pass tuples `None` otherwise; `newton_cg`'s **Timing** paragraph extended (the display, a pointer to §4.13), not rewritten; `NewtonInfo`: the display prints `timing` as the `time` line. |
| `docs/fitting_and_optimization.md` | **§4.13 "Timing a Newton run"**: purpose; a platform-stable doctest of the plan's worked example (P2's review-stop run, through the frontend: keys, the window tiling, `window == wall_time`, the `total` identity, `compile is None` on numpy, the untimed run's equal losses); the phases table (plan §2.1); the record and the one-offs; the `stats['timing']` keys table; the display (the laptop's printed run, a plain block); **what it costs** (the overhead table, below); **jit or eager, as the compile column shows it** (printed `use_jit` time lines cold/warm, the cold-run compile table, what the warm split showed); the caveats (compile `None`; process-global listener; `previous_callback` a row late; async work charged to the next waiting phase; an overriding `local_model` timed whole; a member without `timing`; the frontend setup not timed). §2.1's diagnostics bullet points to §4.13. |
| `docs/aligned_padded_frame.md` | "Timing the passes", one paragraph at the end of "Jit or eager": the members' `timing`, synchronized per pass, host loops only, `None` under the uniform whole-loop jit, the public loops and `padded_start` without it. |
| `docs/contributor/scan_body_principles.md` | The partial-binding paragraph names the eager-only `clock` slot, why it must never reach a jitted body, and that the kernel's `ValueError` is the guard (draft 4's clause). |
| `CHANGELOG.md` | `[2026.5.0]` gains `### Added` after `### Changed` (the section's order; the switch appends): the keyword, `NewtonInfo.timing` / `NewtonTiming` / `PhaseSeconds`, `stats['timing']`, the display, the members' `timing`, `tree_block_until_ready`, and the measured cost. |
| tests | T8 `test_optimizer_display.py::TestTimingDisplay` (7 tests): the time line only with timing (the timed block is the untimed block plus one line after the header); every cell by the record that sets it (exact strings on hand-built records: the model sum and brackets, `curv`, the per-iteration CG time less its compile, `accepted` only on a backtracked row, `track`, a reused model, the converged row, no CG iteration); row 0's one-offs and pass counts; the compile cell (none without a listener, `compile 0`, rounding below 1 µs, the one-offs' compile); the closing table on hand-built rows (phases that ran only, % of the summed windows, mean over the rows run, compile column, one-off rows, pass counts and CG per iteration, with and without a listener); end to end (capped and converged runs: one time line per block, the table once and last, the untimed text equals the timed text without them, wall clocks masked; the same callback reused: the second table counts the second run's rows only); a replayed converged row without a stop reason closes the table. T9 `test_optimizers_frontend.py`: the exemption lines deleted; `test_newton_cg_forwards_timing_only_when_set` (True / `np.True_` forwarded as `True`; False / `np.False_` / default not forwarded); `test_newton_cg_timing_must_be_a_bool` (`'yes'`, 1, 0, None, 1.0: `TypeError`, problem never built); `test_newton_cg_timing_runs` (ragged and uniform: `stats['timing']` per row, the same numbers as untimed, `verbose=True` prints one time line per row and the table once, after the last row; `diagnostics` unchanged). |

### The `time` line and the closing table

```
  time  model 3.6e-03 [frame 4.9e-04 sweep 9.9e-04 comp 3.3e-06 resid 1.0e-03 curv 1.1e-03] | grad 2.8e-03 | cg 4.5e-02 (1.5e-02/it) | forms 1.2e-02 | trials 1.9e-03 ×1 [retr 7.9e-04 obj 1.2e-03] | cb 2.9e-04 | rest 1.8e-04
```

Row 0 adds `ref … ×9` after `grad` and `setup …`, `align … ×5` after `rest`; a jax run ends with `compile <phase> <s>
…` (or `compile 0`). The closing table (the §4.13 example on numpy) is in §4.13 and in §4 below.

## 2. Judgement calls

1. **Number format: `%.1e` seconds in every cell** (the display's fixed 7-character cell, as in the error table),
   not the plan sketch's `.12`. Phases span µs to minutes (P2's example: `companion` 3e-6 s, `cg` 0.2 s), so fixed
   decimals print most of a numpy row as `.00`; `%.1e` keeps two digits at every scale and aligns rows that carry the
   same cells. The CG per-iteration figure uses the same cell, `(1.5e-02/it)`, not "33 ms/it".
2. **One line, indented two spaces** like the table's `  rel err` legend; not wrapped (the header is one long line too).
3. **The `trials` cell has a bracket** `[retr … obj …]`, like `model`'s, so retraction and objective cost are both
   visible; `(accepted …)`, the accepted trial's retraction, is shown only on a backtracked row (`trials > 1`), where
   it differs from `retr`. The plan sketch had `trials .21 ×3 (accepted .07)` only.
4. **A phase that did not run is left out** (not `—`): the converged row has no `cg`/`forms`/`trials`, a row after a
   rejected trial no `model`/`grad`, row 0 no `cb`. `rest` is always shown.
5. **Cell order**: the row's own work in run order (`model`, `grad`, `ref`, `cg`, `forms`, `trials`, `track`), then
   `cb` and `rest` as the plan sketch has them, then row 0's one-offs (`setup`, `align`), which lie outside the window,
   then `compile`. The docstring states this order.
6. **The compile cell** lists phases whose compile wait exceeds 1 µs (`_COMPILE_SHOWN`: the remainder's difference
   can carry float rounding), the one-offs' compile included, `compile 0` when nothing compiled, and no cell without
   a listener (numpy, jax 0.4.30) -- so "not measured" and "nothing compiled" read differently.
7. **CG per iteration excludes CG's compile wait**, on the line and in the table, matching `stats['timing']
   ['cg_per_iteration']`.
8. **The closing table**: full `PhaseSeconds` names (the keys of `stats['timing']['phases']`), phases that never ran
   omitted (`curvature` under Gauss-Newton, `tracking`), the mean over the rows a phase ran on (`reference` runs once),
   `setup` / `alignment` as rows with `—` for the window share and mean, then one line with the pass counts and CG's
   seconds per iteration. Printed by a separate `print_fn` call after the last row's block.
9. **Frontend validation**: `timing` must be a `bool` (or `numpy.bool_`), else `TypeError` before the setup work. "Forwarded
   only when True" otherwise has no clear reading for `timing=1` or `'yes'` (silently untimed, or truthy-timed); the
   structural rule (always raise) decides it. The backend keeps no check (P2's; a raw-`.data` caller passing a truthy
   value gets a timed run).
10. **Docs example names** (`probes`, `truth`, `noisy`, `x_old`, `x_new`, `level`, `T`, `rows`, `plain`): the page is one
    doctest namespace and §5 later reuses `ww`, `data`, `x0`, `x0c`, so §4.13 must not rebind them. The example is
    P2's review-stop run (plan row 5.5) through the frontend: the same numbers (6 rows, 31 CG iterations).
11. **The frontend docstring example reuses the exact-Hessian example's problem** (the same problem as §4.13: its
    `x0` is the rank-2 fit) rather than repeating the setup in a docstring that already builds it.
12. **No glossary entry**: §6 does not ask for one; the 2026.5.0 glossary/docs pass (bundle item C) comes after both
    features and can add "phase timing" / `stats['timing']` then. The `stats` entry there lists `start_frame` etc. but
    not `timing` -- a one-word addition for that pass.
13. **No `docs/release_notes.md` line**: its 2026.5.0 section ("Upgrading from 2026.4.0") lists no additions (plan §6's
    condition). **The CHANGELOG's `[2026.5.0]` intro paragraph still names only the store rename** as the headline: I
    did not touch it (the release commit re-cuts the notes).
14. **§2.1's diagnostics bullet** gained half a sentence pointing to §4.13 (the display is documented there), and the
    stats table in §2.1 was left alone (it does not list 2026.4.0's `start_frame` keys either).
15. **Measurement protocol** (below): one `(config, layer, backend)` per process (eager jax adds thousands of memory
    mappings per structure; separate processes also give every cold run empty caches); a cold timed run, an untimed
    warm-up, then interleaved (off, on) pairs, 3 on jax and 7 on numpy (the first numpy batch with 3 pairs showed
    +13 %; a direct measurement of the waits showed 0.1 %, so it was noise, and 7 pairs settle it); jax in x64 so the
    iterations equal numpy's. The d = 6 case is the default Gauss-Newton + line search (the example is exact + trust
    region + alignment), so both step controls are measured.

## 3. Where the plan could not be followed literally

Nothing blocked. Readings, all recorded above: the `%.1e` cells and the `trials` bracket (calls 1, 3); "the
alignment's whole-loop program sits in `alignment_compile`" holds for the uniform layer (measured 9.0 s of 9.4 s),
while the ragged alignment's compile is its jitted steps plus eager projections (15.8 s) -- §4.13 says both.
Plan §6 says eager jax "spreads many per-operation compiles over row 0's phases": measured, eager phases also compile
on a later row when they first run there (row 1's first `geom.frame` after a padded start; the display's first error
table in `cb`), under `use_jit` too; §4.13 says "the first time they run: mostly on row 0".

**A finding for the maintainer (not acted on):** under `use_jit` on this small problem the eager phases dominate
(warm, ragged: `forms` -- the exact `pᵀHp` -- 42 % of the rows, the `'cone'` loop 28 %, `gradient` 12 %, CG 0.9 %), and
the run is 4.7x slower than numpy (3.5 s vs 0.74 s). §4.13 states it as what the split is for; whether to jit more of
the step is a design question outside this plan.

## 4. The figures (the maintainer's laptop)

Config: AMD Ryzen 7 3700U (4 cores / 8 threads), 13 GB, CPU only; Python 3.11.15, numpy 2.4.6 (scipy-openblas
0.3.31), jax 0.10.2 in x64. Scripts: the session scratchpad `p3/` (`configs.py`, `measure.py`, `measure_all.sh`,
`measure_rest.sh`, `wait_cost.py`, `display_example.py`); raw results `p3/measure.jsonl`, `p3/measure2.jsonl`.

- **A, the §4.13 example**: probe derivatives of order 1, d = 3, shape (5, 5, 5), 40 probes, 1 % noise, rank-3 truth;
  `x_old` = 5 Gauss-Newton steps at rank 2, zero-padded to rank 3; `hessian='exact'`, trust region,
  `cg_forcing_power=1.0`, `padded_from=x_old`, `g0_reference='cone'`, `max_newton=40` (6 rows, 31 CG iterations).
- **B, d = 6**: shape `(8,)*6`, 200 probes, order 1, 1 % noise, rank-4 truth, fitted at rank 3 from a random start
  by the default Gauss-Newton + line search, `max_newton=10` (10 rows, 27 CG iterations).

Each built through the frontend's `least_squares_problem` / `_old_point_data`, timing the backend call.

**Overhead**, seconds per call, off → on (median; 3 interleaved pairs after a warm-up, 7 on numpy):

| case | layer | numpy | jax `use_jit` | jax eager |
|---|---|---|---|---|
| A | ragged | 0.768 → 0.738 (−3.9 %) | 3.487 → 3.545 (+1.7 %) | 13.984 → 13.824 (−1.1 %) |
| A | uniform | 0.861 → 0.860 (−0.1 %) | 0.819 → 0.813 (−0.8 %) | 3.208 → 3.243 (+1.1 %) |
| B | ragged | 0.805 → 0.806 (+0.2 %) | 3.958 → 3.970 (+0.3 %) | 9.639 → 9.676 (+0.4 %) |
| B | uniform | 1.021 → 1.022 (+0.1 %) | 0.946 → 0.912 (−3.6 %) | 1.265 → 1.278 (+1.0 %) |

All within the run-to-run spread (single numpy runs of one call: up to 15 %). The waits on numpy, measured directly
(A, mocked wait timed): 71 waits, 10–13 µs each, 0.1 % of the run. Not measured: a GPU (the docs say to measure there).

**Compile**, the cold timed run on empty caches (seconds; the compile column summed; a rerun at the same structure
compiled 0 in every case):

| case | layer | backend | cold run | compile | largest cells | rerun |
|---|---|---|---|---|---|---|
| A | ragged | `use_jit` | 35.8 | 30.8 | alignment 15.8, reference 4.5, cg 2.9, trial_retraction 2.6, curvature 2.1 | 3.5 |
| A | uniform | `use_jit` | 38.5 | 36.1 | alignment 9.0 (of 9.35; passes `None`), reference 8.0, trial_retraction 5.9, frame 4.2, curvature 3.5, cg 1.6 | 0.82 |
| A | ragged | eager | 71.3 | 54.1 | alignment 32.7, reference 13.2, trial_retraction 2.2, curvature 2.1, cg 1.7 | 14.0 |
| A | uniform | eager | 80.4 | 72.6 | alignment 34.9, reference 28.0, curvature 3.3, trial_retraction 2.1, cg 1.9 | 3.2 |
| B | ragged | `use_jit` | 17.2 | 12.3 | gradient 3.8, trial_retraction 2.7, cg 2.1, sweep 1.4, frame 1.4 | 4.0 |
| B | uniform | `use_jit` | 17.3 | 15.5 | trial_retraction 5.4, frame 5.4, gradient 1.4, reference 1.1, cg 0.8 | 0.95 |
| B | ragged | eager | 21.2 | 10.9 | gradient 3.7, trial_retraction 2.7, frame 1.4, sweep 1.4, cg 1.0 | 9.6 |
| B | uniform | eager | 18.0 | 15.8 | frame 5.5, trial_retraction 5.5, gradient 1.4, reference 1.2, cg 1.1 | 1.3 |

Warm CG per iteration (less compile): A ragged numpy 15 ms, `use_jit` 0.9 ms, eager 305 ms; A uniform 17 / 2.0 / 57 ms;
B ragged 17 / 2.6 / 216 ms; B uniform 20 / 4.1 / 19 ms.

## 5. Verification

One job at a time, current env `t3toolbox` (py3.11, numpy 2.4.6, jax 0.10.2) unless stated; scripts and logs in the
session scratchpad `p3/`.

| check | result |
|---|---|
| **Non-breaking, bitwise against HEAD `a03cb43c`** (a scratchpad worktree, since removed): `p3/battery.py` = the P2 reviewer's battery (every `NewtonInfo` row the callback sees with its model parts, `x`, the whole `stats` with key order and float bits, `make_newton_display`'s records and text with the `%.2fs` cell masked, `local_model`, `padded_start`, `gradient_descent`, the structural-error messages; numpy ragged and uniform, unshared and shared, probe and probe-derivative kinds; jax x64 `use_jit` and eager) **plus 12 frontend entries**: `topt.newton_cg` ragged and uniform, plain, `verbose=True`, `verbose` with validation data, exact + `padded_from` + `'cone'` verbose, exact trust region with rejections verbose, and an explicit `timing=False` verbose run (absent at HEAD): `x`, `stats` (with `diagnostics`), the printed text, the stats keys; `p3/compare.py` | **numpy 234 entries, 23,755 arrays, 0 mismatches; jax 17 entries, 1,615 arrays, 0 mismatches.** A planted one-character change in a frontend display text is caught |
| **Mutation testing** of the new tests (`p3/mutate.py`: each mutation on a fresh copy of the tree; only `TestTimingDisplay` or the frontend timing / mirror tests run) | **36 of 36 caught**: the line never added / after the table; the model sum / without `curv`; `accepted` on one trial / never; CG per iteration with its compile / without the iteration guard; no `track` / `cb` / `grad` / `setup` cell; the pass counts dropped (ref, align); the one-offs before the window; the compile rounding shown; the compile without the one-offs; no `compile 0`; the table with never-run phases, % of the last window, the mean over all rows, no compile column, no one-off rows, no pass counts, CG per iteration with its compile; the table on every row / only on converged rows / only on stop rows; no reset at row 0; the frontend forwarding always / never / as given, no validation, `numpy.bool_` refused, validation after the setup, default `True` |
| new tests | `TestTimingDisplay` 7 passed (2.4 s); the three T9 tests and the mirror test pass |
| full suite, `pytest tests/ -q -n 3` | **1,379 passed, 5 skipped**, 48,890 subtests (17 min 48 s); P2's 1,369 + the 10 new tests; the one warning is the pre-existing adam-on-manifold one |
| module doctests, `pytest --doctest-modules t3toolbox -n 3` | **241 passed** (the count is per docstring; the new examples are in existing docstrings) |
| the quickstart and every docs page, as CI (`python -m doctest docs/getting_started.rst`; `docs/*.md docs/contributor/*.md` but `doctest_style.md`) | **pass** |
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, no warning**; the §4.13 anchor resolves (`fitting_and_optimization.html#timing-a-newton-run`, linked from `aligned_padded_frame.html`) |
| floor env `tttt` (py3.9.19, numpy 1.22.4, jax 0.4.30), single process: `test_optimizer_display.py`, `test_optimizers_frontend.py`; the doctests of the three touched modules; the three touched pages | **68 passed**; **6 passed**; **pass** |
| the CI examples command (every `examples/*.py`, current env) | **14 of 14 pass** |

Two edits came after the suite started, both whitespace or docstring only (a signature comment column in
`optimizer_display.py`, a rewrapped frontend docstring paragraph); the touched test files were rerun after them on both
envs (68 passed each) and the module doctests after them.

## 6. For the maintainer's FINAL STOP

1. **The display's look** (calls 1–8): `%.1e` cells, one long line, the `trials` bracket, omitted cells, the cell
   order, the compile cell's `0`, and the closing table. Printed examples: §4.13 (numpy run; `use_jit` cold and warm
   time lines).
2. **The measured cost** (§4): no measurable overhead on this CPU, numpy or jax; a GPU unmeasured. The docs and the
   frontend docstring say so and tell GPU users to measure there.
3. **The `use_jit` finding** (§3): on small problems the eager phases (`forms`' exact `pᵀHp`, the cone loop,
   `gradient`) dominate a `use_jit` run; jitting more of the step is a separate design question.
4. **The frontend `TypeError` for a non-bool `timing`** (call 9): the only new raise.
5. **Release text**: the CHANGELOG `[2026.5.0]` intro still headlines only the rename; no `release_notes.md` line and
   no glossary entry (calls 12–13) -- both for the bundle's docs pass / release commit.
6. **The self-expiring exemption is gone** (`tests/test_optimizers_frontend.py`); the mirror test passes with the
   frontend keyword.

## 7. Round 1 (review `P3_review.md`: REVISE, two MINOR, six NIT)

The reviewer's machine runs were in progress, so this round ran only the two touched test files, targeted mutations and
single doctests, all single process.

| finding | change | checked |
|---|---|---|
| **M1** two crash regressions untested | `test_a_timed_run` gains a third run, `'converged at row 0'` (`gtol_rel=10.0`): one block with its time line, then a one-row table with no CG note (it ends with the `setup` row); every run asserts the CG note is present iff some row ran CG. New `test_a_silent_display_of_a_timed_run`: a records-only display (`print_fn=None`) on a timed run (capped; converged at row 0) prints nothing, does not raise, and records exactly the untimed run's records (keys in order; arrays and values equal; `wall_time` aside) | the reviewer's two mutations (no `print_fn is not None` before the table; no `iterations > 0` guard): **caught** |
| **M2** "bitwise" overstated | "the `wall_time` clock readings aside" added in the frontend docstring, §4.13's intro and the CHANGELOG entry; the CHANGELOG's `timing=False` sentence also says "(clock readings aside)" | read; §4.13 and the frontend module doctests pass |
| **N1** three test gaps | `test_the_closing_table`: the listened row 0 has `setup_compile=2e-4`, `alignment_compile=0.4`, and both one-off rows' compile cells are asserted; `test_the_compile_cell`: `frame=5e-5` must show as `frame 5.0e-05`; `test_newton_cg_forwards_timing_only_when_set`: the last two frontend parameters are `cone_seed`, `timing`, and `timing` is `KEYWORD_ONLY` | mutations (the one-off compile cells swapped; `_COMPILE_SHOWN = 1e-3`; `timing` moved before the `*`): **3 of 3 caught**, the last on the intended assertion |
| **N2** return comments one column left | `_format_timing`: both `#` at column 37; `_format_timing_table`: both at 83 (whitespace only) | measured |
| **N3** §4.13's cold-run table | the column is now "the four largest compile cells; `cg`": the four largest, recomputed from `p3/measure*.jsonl`, then `cg`'s when it is not among them (A uniform `use_jit`: `frame` 4.2 in, `curvature` out, `; cg 1.6`; A uniform eager: `trial_retraction` 2.1 added; d = 6 `use_jit` ragged: `sweep` 1.4 for `frame` 1.35; d = 6 uniform: `reference` 1.1 / 1.2 added); `38.5` → `39`; the lead sentence says how the column reads | read against the raw results |
| **N4** "in run order" | "in the order the `time` line shows them" | read |
| **N5** `timing  1 rows` | `_format_timing_table` writes `1 row` (library, timed path only); the row-0 runs assert `'  timing  1 row, '` | mutation (always "rows"): **caught** |
| **N6** | "the uniform loop when run eagerly" | read |

Runs this round (current env, single process): `TestTimingDisplay` + the frontend timing / mirror tests **12 passed**;
both touched test files in full **69 passed**, 86 subtests (83 s); the round-1 mutations (`p3/mutate_r1.py`) **6 of 6
caught**; the doctests of `fitting_and_optimization.md`, `aligned_padded_frame.md`, `optimizers.py`,
`optimizer_display.py` pass. Library changes this round: `optimizer_display.py` (N2's whitespace; N5's `row`/`rows`,
reached only with timing on) and the frontend docstring (M2, N4); nothing on an untimed path, so round 0's battery
stands by construction (the reviewer re-runs it). Not rerun this round: the full suite, sphinx, the floor env, the
examples (the reviewer's runs held the machine).
