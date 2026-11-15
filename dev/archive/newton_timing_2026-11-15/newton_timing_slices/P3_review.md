# P3 review: the display, the frontend keyword, the figures, the docs (Newton timing, 2026.5.0)

Reviewer's record. Base HEAD `a03cb43c` (the plan at draft 5; P1 `cbcdc1d6`, P2 `f8cbb442`); P3 in the working tree,
uncommitted. Read: `CLAUDE.md`; the HANDOFF top block; the plan (draft 5, §1-§9); `P1_handoff.md`, `P1_review.md`,
`P2_handoff.md`, `P2_review.md`, `P3_handoff.md`; `signature_style.md`; `probing.py` and `tucker_tensor_train.py` for
signature shape; the whole diff and the code around it (`format_newton_iter`, `_format_header`, `_grid`, `_num`,
`make_newton_display`, the `newton_cg` loop's row records and `_timing_stats`, the frontend `newton_cg` and its mirror
test); the glossary's optimization section; `release_notes.md`'s 2026.5.0 section; the CHANGELOG's earlier `### Added`
sections; the implementer's measurement JSON (`p3/measure*.jsonl`).

## Round 1

### What I ran (my own, one job at a time; current env py3.11 / numpy 2.4.6 / jax 0.10.2 unless stated)

Scripts and logs: the session scratchpad, `p3_review/`. Everything ran on a byte copy of the working tree
(`p3_review/wt0`, checked identical to the tree before and after), so the implementer could edit meanwhile.

| check | result |
|---|---|
| **Bitwise vs HEAD** (`battery.py`, `compare.py`; a worktree of `a03cb43c` against the copy). The P2 reviewer's backend battery unchanged (every `NewtonInfo` row with its model parts, `x`, the whole `stats` with key order and float bits, the display's records and text with the `%.2fs` cell masked, `local_model`, `padded_start`, `gradient_descent`, the structural-error messages; numpy ragged and uniform, unshared and shared, probe and probe-derivative kinds; jax x64 `use_jit` and eager), plus a frontend section of my own: the frontend's signature minus `timing` (names, kinds, defaults, annotations) and `optimizer_display.__all__`; `format_newton_iter` on hand-built untimed rows (GN, regularized, exact TR rejected, converged, a stop row) at all three table layouts; frontend `newton_cg` on both layers, every run with its printed text, `stats` (with `diagnostics`) and, for one, a recording callback: `apply` / `probe` with validation, `probe_derivatives` with validation + regularizer, a residual weight, exact line search, exact TR with rejections, `padded_from` + `'cone'` + options, Armijo exhausted under `'stop'` and `'accept'`, converged at row 0, `max_newton=0`, warm-start overrides, and in the tree an explicit `timing=False` / `numpy.False_`; shared and corewise geometries; seven frontend error messages per layer; jax x64 frontend `use_jit` GN and exact + `padded_from` + `'cone'` on both layers | **numpy 292 entries, 24,157 arrays, 0 mismatches; jax 25 entries, 1,649 arrays, 0 mismatches.** A planted one-character change in one display line is caught (6 entries) |
| **The display, by eye and by arithmetic** (`display_check.py`): the §4.13 example through the frontend, `verbose=True`, `timing=True`: numpy; jax x64 `use_jit` ragged cold then warm; `use_jit` uniform cold (`display_*.out`) | every closing-table cell equals `stats['timing']` formatted (seconds, `% window`, compile, mean over the rows run, setup, alignment); `% window` sums to 100.0 (99.9 once, rounding); Σ phases == Σ window and `total == setup + alignment + Σ window + last_callback`, exactly (difference 0.0); no negative compile cell; the uniform run has no pass counts (`None` under the whole-loop jit) |
| edge paths (current env) | a silent display (`print_fn=None`) on a timed run; converged at row 0 (a one-row table, no CG note); an Armijo stop on row 0 (`trials … ×40`, no `accepted`); exact TR with two rejections (the reused rows start `time  cg`); `max_newton=0` (no rows, no table) -- all print or stay silent as they should |
| **Overhead re-run** (`overhead.py`, the §4.13 example, 7 interleaved pairs on numpy, 3 on jax) | numpy ragged 0.720 → 0.723 (+0.4 %); numpy uniform 0.848 → 0.847 (−0.1 %); `use_jit` ragged 3.507 → 3.514 (+0.2 %), cold 36.2 s. Inside the docs' −4 % to +2 % |
| **Compile figures** | the `use_jit` cold runs above reproduce the docs: ragged 35.8 s, 30.8 s compiling (alignment 15.8, CG 2.9), 3.57 s again; uniform 38.2 s, 35.7 s compiling (alignment 9.0 of 9.4). The warm split: CG 0.91 ms per iteration, `forms` 41.2 %, `reference` 29.1 %, `gradient` 12.6 % (docs: 0.9 ms, 42 / 28 / 12 %). Eager cold not re-run (plausible: see below). Every cell of the cold-run table checked against the implementer's raw JSON (N3) |
| **Mutations** (`mutate.py`, mine, distinct from the implementer's 36; a fresh copy each; T8 or T9 rerun) | **15 of 20 caught**: `%.2e` cells; the trials sum without the objective; retr/obj swapped; `accepted` showing the whole retraction; the compile cell sorted by label; no time line on a converged row; the reset after the append; the header counting CG rows only; `% window` of the total with the one-offs; `cb` before `cg`; the time line on row 0 only; the table before the last block; the frontend accepting ints; a `ValueError`; forwarding `bool(timing)` always. **Missed: 5** (M1, N1) |
| full suite, `pytest tests/ -q -n 3` | **1,379 passed, 5 skipped**, 48,890 subtests (23 min 22 s); the one warning is the pre-existing adam-on-manifold one |
| module doctests, `pytest --doctest-modules t3toolbox -n 3` | **241 passed** |
| the quickstart and every docs page (CI's command) | **pass**; §4.13's 19 examples ran (`-v`) |

### Plan fidelity

Every §7 P3 row and every §6 item is built as the plan says:

- **Display** (§4): `format_newton_iter` inserts one `time` line between the header and the error table when
  `info.timing` is set, from a private `_format_timing`; the header (κ segment included) and the table are untouched
  (the battery's untimed texts, and T8's "timed block = untimed block plus one line"). `make_newton_display` keeps
  `(timing, cg_iters)` per timed row, clears at `iteration == 0`, and prints `_format_timing_table` once after the
  run's last row (`converged`, or `stop_reason` set, which `newton_cg` sets on every run's last row: the gradient
  test, the cap, the line search, both floors). Records unchanged (`_newton_scalar_record` skips `timing`).
- **Frontend** (§3.3): `timing: bool = False` keyword-only, last, after `cone_seed`, its comment in the cone group's
  column; forwarded as `timing=True` only when set (the untimed call is 2026.4.0's: the battery's signature entry and
  every frontend run bitwise); the docstring covers each item §3.3 lists (what is timed, synchronization, compile
  accounting and where it is `None`, W-5, an example, `verbose=True`).
- **Docs** (§6): §4.13 with every sub-item (phases, one-offs, `stats['timing']`, the display, the costs with config,
  jit-or-eager with the compile column, the caveats, a platform-stable doctest); the `aligned_padded_frame.md`
  paragraph; the `scan_body_principles.md` clause (draft 4's wording, in substance); the three backend docstrings
  extended, not rewritten; CHANGELOG `[2026.5.0]` `### Added`, created after `### Changed` (2026.3.0's order; the switch
  appends). No `release_notes.md` line: its 2026.5.0 section lists no additions, which is §6's condition.
- **The exemption** is deleted (the three lines and the `or name in not_yet`), and the mirror test passes with the
  keyword (suite, below).
- **Nothing renamed**: the diff's removed lines are the four rewrapped docstring sentences, the `return` and the
  `**seam)` call line that P3 extends, and the exemption. `backend/optimizers.py` changes docstrings only.

**The judgement calls.** All sound; none needs to go back.

1. *`%.1e` cells.* Sound: the error table's own cell, seven characters at every scale; P2's example spans 3e-6 s
   (`companion`) to 0.2 s (`cg`), so fixed decimals would print most of a numpy row as `.00`.
2. *One line.* Sound and consistent with the header (also one long line). Its length is for the maintainer's eye
   (below).
3. *The `[retr … obj …]` bracket* and `(accepted …)` only when `trials > 1`. Sound: it parallels `model`'s bracket, and
   on a one-trial row the accepted share equals `retr`, so showing it would repeat a number.
4. *A phase that did not run is left out.* Sound: the line then reads as the row's work (a row after a rejected trial
   starts `time  cg …`; the converged row has no `cg`/`forms`/`trials`), and `None` never prints as a number.
5. *Cell order* (run order, then `cb`, `rest`, the one-offs, `compile`). Sound; the docstring says so.
6. *`compile 0` vs no cell; the 1 µs threshold.* Sound: "not measured" and "nothing compiled" read differently, and the
   threshold only hides the remainder's twin's rounding. (The table shows the raw sums, `0.0e+00` where nothing
   compiled; consistent.)
7. *CG per iteration less its compile wait*, on the line and in the table, equal to `stats['timing']['cg_per_iteration']`
   (checked on both jax runs below).
8. *The closing table* (full field names, never-run phases omitted, mean over the rows run, the one-offs with `—`, a
   notes line; printed once, after the last block; a reused callback starts again at row 0). Sound; my runs checked
   every cell against `stats['timing']` (below).
9. *The frontend `TypeError`* (`'newton_cg: timing must be a bool, got %r'`, before the setup work, `numpy.bool_`
   accepted). Sound and in the house style (`'newton_cg: start_frame must be a ... got ...'`): "forwarded only when
   True" has no clear reading for `1` or `'yes'`, and a type is structural. The backend stays unchecked (P2; a
   raw-`.data` caller's truthy value times the run, like its other flags): an asymmetry for the maintainer to see,
   not a defect.
10-15. *Doc example names, the docstring example's reuse of the exact-Hessian problem, no glossary entry (see below),
   no release-notes line, §2.1's half sentence, the measurement protocol.* Sound.

### Non-breaking

Confirmed by my own battery (above): with `timing=False` every frontend and backend run, every printed display text
(wall clock masked), every `stats` key and order, every recording callback's row, the frontend's signature (timing
aside), `optimizer_display.__all__` and the structural-error messages are bitwise HEAD's. `tests/test_dispatch.py` is
untouched by P3 and green in the suite.

### The display

Read against `format_newton_iter`'s house style: two-space indent and a two-word lead like the table's `  rel err`, ` | `
separators like the header, `%.1e` cells like the table. By eye on numpy and on jax `use_jit` (ragged cold and warm,
uniform cold; `p3_review/display_*.out`), and by arithmetic on all three: every closing-table cell equals
`stats['timing']` formatted (seconds, `% window`, compile, mean over the rows run, the one-offs), the `% window` column
sums to 100 %, Σ phases equals Σ window exactly, and `total == setup + alignment + Σ window + last_callback` exactly.
The edge paths print sensibly: converged at row 0 (a one-row table, no CG note), an Armijo stop on row 0 (`trials …
×40`, no `accepted`), exact trust region with two rejections (the reused rows start `time  cg`), `max_newton=0` (no
table), a silent display (`print_fn=None`, no table, no error).

### The docstring and the docs

Each claim checked against the code or a run: the phases and what they cover (§2.1's table), the one-offs, the
`stats['timing']` keys, `total`'s identity, the compile column (`None` on numpy and on jax 0.4.30), the process-global
listener, `previous_callback` a row late, the async caveat, the override and member caveats, W-5. The figures: the
overhead re-run (numpy ragged, numpy uniform, `use_jit` ragged) lands inside the docs' −4 % to +2 %; the `use_jit`
cold run reproduces exactly (36 s, 31 s compiling, 16 s in the alignment, 2.9 s CG's program; 3.5 s again), and the
uniform one too (38 s, 36 s compiling, 9.0 s of 9.4 s in the alignment); the warm split (CG 0.9 ms per iteration,
`forms` 41 %, `reference` 29 %, `gradient` 13 %) matches the docs' 42 / 28 / 12 %. The eager cold figures (71 s, 54 s
compiling) are plausible against these and against `aligned_padded_frame.md`'s eager numbers (an eager alignment
compiles every operation; about twice the jitted alignment's compile here). The GPU caveat is in the docstring, §4.13
and the CHANGELOG. The CHANGELOG entry is in the file's style (2026.4.0's `### Added`). Two wording points: M2 and
N3-N6.

**The glossary.** §6 does not ask for an entry, and not adding one in P3 is right: the glossary pass is bundle item
(C), after both features, so their entries are written once. But the glossary's optimization section has one entry
per optimizer feature (2026.4.0 added *padded start* and *oversampled ranks*, and the *stats* entry lists
`start_frame`, `start_movements`, `cone_movements`), so (C) should add: **phase timing** (`newton_cg(timing=True)`;
`NewtonTiming`, `PhaseSeconds`, `stats['timing']`; also: phase, window, one-offs, compile wait), linking §4.13, and
`timing` in the **stats** entry. "Lap" appears only in private code (`_PhaseClock`), so it needs none. The session
should record this under (C) so it is not lost.

### Tests

T8 (`TestTimingDisplay`, 7 tests) and T9 (three tests plus the mirror test without the exemption) are present and
meaningful: exact strings on hand-built records for every cell, the end-to-end capped and converged runs with the
untimed text as the oracle, the reuse case, the forwarding spy, the validation before the setup. My 20 mutations
(distinct from the implementer's 36): 15 caught, 5 missed (M1, N1).

## Findings

No MAJOR finding. Non-breaking holds bitwise, the display adds up, the figures reproduce.

### MINOR

**M1. Two crash regressions on documented paths are untested** (my mutations, both missed by T8):

- *A silent display on a timed run.* Dropping `print_fn is not None` before the closing table stays green; the
  regression is a `TypeError` (`'NoneType' object is not callable`) at the run's last row for anyone who builds a
  records-only display (`make_newton_display(..., print_fn=None)`, a documented option) and times the run.
- *A timed run with no CG iteration* (converged at row 0). Dropping the table's `iterations > 0` guard stays green; the
  regression is a `ZeroDivisionError` at the last row.

**Fix:** in `test_a_timed_run` (or beside it), a timed run with `print_fn=None` whose records equal the untimed run's
(wall clocks masked), and a converged-at-row-0 subtest (`gtol_rel=10.0`): one block with its time line, then a one-row
table without the CG note. Criterion: both mutations fail.

**M2. "Bitwise" overstates by one field.** The frontend docstring ("`x` and every other `stats` field are bitwise the
untimed run's"), §4.13's intro and the CHANGELOG entry say every other `stats` field is bitwise. `stats['history']`'s
`wall_time` (and `stats['diagnostics']`'s) is a clock reading and differs, timed or not. The plan's §1 says "the
`wall_time` clock readings aside". **Fix:** add that clause in the three places.

### NIT

**N1. Three smaller test gaps** (mutations missed): the closing table's one-off compile cells (swapping `setup_compile`
and `alignment_compile` stays green, since the listened rows leave both `None`: give them distinct values and assert
the two rows' compile cells); the 1 µs threshold (`_COMPILE_SHOWN = 1e-3` stays green: one compile wait between, say
`frame=5e-5`, shown as `frame 5.0e-05`); the keyword-only placement (moving `timing` before the `*` stays green, since
the mirror test checks names and defaults only: assert `front['timing'].kind is KEYWORD_ONLY` and that it is last).

**N2. The two new signatures' return comments sit one column left of the argument's** (`_format_timing`: `#` at
column 36 on the `info` line, 35 on `) -> str:`; `_format_timing_table`: 82 and 81). The exemplar
(`probing._xi_step`) aligns them. Whitespace only.

**N3. §4.13's cold-run table**: the column is "largest compile cells", but it always lists `cg`, also where `cg` is not
among the largest (A uniform `use_jit`: `forms` 1.8 and `gradient` 1.6 left out, `cg` 1.6 in; A uniform eager:
`trial_retraction` 2.1 out, `cg` 1.9 in; d = 6 uniform `use_jit` / eager: `reference` 1.1 / 1.2 out). Name the column
"largest compile cells, and `cg`", or list the largest. The cold-run column has one `38.5` among whole seconds.

**N4.** The frontend docstring's "A row's phases, in run order" lists `previous_callback` last; in run order it is
first (as `PhaseSeconds` declares it). "In the order the `time` line shows them" would be exact.

**N5.** A one-row run's closing table reads `timing  1 rows` (converged at row 0, an Armijo stop on row 0).

**N6.** `aligned_padded_frame.md`: "the ragged loop (eager, or around its jitted step) and the uniform loop run
eagerly" reads as a verb; "the uniform loop when run eagerly".

### For the maintainer (not findings)

- The `time` line is long: about 230 characters on numpy, about 330 on a cold jax row 0 with its compile cell (the
  header is about 170). It carries no unit; the header's wall clock reads `0.17s`, the table's column says `seconds`.
- The frontend rejects a non-bool `timing`; the backend treats any truthy value as on (P2, unchanged).

VERDICT (round 1): REVISE (two MINOR: one test-side, one wording; no library code path needs to change)

## Round 2 (the implementer's `P3_handoff.md` §7)

What changed: tests, docs and three library lines. I diffed a fresh copy of the tree (`p3_review/wt1`) against my
round-0 copy file by file. Under `t3toolbox/`: the two signature comment columns (whitespace), the closing table's
header (`row` / `rows`, reached only with timing on), and the frontend docstring's two sentences.

| finding | resolution | checked by me |
|---|---|---|
| M1 | `test_a_timed_run` gains `'converged at row 0'` (one block, then a one-row table ending at `setup`, no CG note; every run asserts the CG note iff a row ran CG); new `test_a_silent_display_of_a_timed_run` (`print_fn=None`, capped and converged at row 0: silent, no raise, records equal the untimed run's key by key, `wall_time` aside) | both mutations: **caught** |
| M2 | "the `wall_time` clock readings aside" in the frontend docstring, §4.13's intro and the CHANGELOG (whose `timing=False` sentence also says "clock readings aside") | read |
| N1 | distinct `setup_compile` / `alignment_compile` with both cells asserted; `frame=5e-5` shown as `frame 5.0e-05`; `timing` asserted last and `KEYWORD_ONLY` | the three mutations: **caught** |
| N2 | both return comments at the argument's column (37 / 83, 1-based) | measured |
| N3 | the column is "the four largest compile cells; `cg`", the lead sentence says how it reads, `38.5` → `39`. Checking every cell against `p3/measure*.jsonl` found two double-rounded values (A ragged eager `trial_retraction` 2.2465 printed 2.3; d = 6 uniform eager `gradient` 1.4486 printed 1.5), also in the handoff's §4; both fixed to 2.2 / 1.4 in a follow-up (docs and handoff only) | every cell: the top four and the values match the raw results |
| N4 | "in the order the `time` line shows them" | read |
| N5 | `1 row` (library, timed path only); the row-0 runs assert it | my "always `rows`" mutation: **caught** |
| N6 | "the uniform loop when run eagerly" | read |

Round-2 runs (mine, one job at a time, on the round-1 copy `wt1`; the follow-up changed two markdown table cells only,
checked identical otherwise):

| check | result |
|---|---|
| the bitwise battery again (the library changed) | numpy **292 entries, 24,157 arrays, 0 mismatches** vs HEAD |
| my mutations (`mutate.py`, plus "the header always plural") | **21 of 21 caught** |
| touched test files, current env | **69 passed**, 86 subtests |
| module doctests, `-n 3` | **241 passed** |
| the quickstart and every docs page, current env | **pass** |
| floor env `tttt` (py3.9.19, numpy 1.22.4, jax 0.4.30): the touched test files; the doctests of `optimizer_display.py`, `optimizers.py`, `backend/optimizers.py`; the quickstart and every docs page (CI's command) | **69 passed**; **6 passed**; **pass** |
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, no warning**; §4.13 renders, `aligned_padded_frame.html` links `#timing-a-newton-run` and §4.13 links back to `#jit-or-eager`; `NewtonTiming` / `PhaseSeconds` pages render |
| the CI examples command (every `examples/*.py`, current env) | **14 of 14 pass** |

The full suite ran on the round-0 tree. Round 1 changed the library by a whitespace column, the table header's plural
(timed path only) and two docstring sentences; the touched files were rerun on both envs. The session's pre-commit
full suite covers the rest.

**For the session (plan text, draft 6):** P3's readings: the `time` line in `%.1e` cells (not the sketch's `.12`), one
line, the cell order (run order, then `cb`, `rest`, the one-offs, `compile`), a phase that did not run left out, the
`trials` cell's `[retr … obj …]` bracket with `(accepted …)` only on a backtracked row, CG per iteration less its
compile wait (line and table), the compile cell above 1 µs with `compile 0` vs no cell; the closing table's columns
(seconds, % window, compile, mean over the rows run), its one-off rows and notes line, printed once after the last row
(converged or `stop_reason`) and reset at row 0; the frontend's `TypeError` for a non-bool `timing` (`numpy.bool_`
accepted), checked before the setup; §4.13's example = P2's review-stop run through the frontend; the measured figures
and their configs as in §4.13; no `release_notes.md` line (§6's condition) and the CHANGELOG intro left to the release
commit; **the glossary entries deferred to item (C)**: a **phase timing** entry (`newton_cg(timing=True)`;
`NewtonTiming`, `PhaseSeconds`, `stats['timing']`; also: phase, window, one-offs, compile wait) and `timing` in the
**stats** entry; the tests beyond §5's T8/T9 rows (the silent display, converged at row 0, the keyword-only
assertion).

**For the maintainer's FINAL STOP:** (1) the display's look: `docs/fitting_and_optimization.md` lines 734-775 (a
numpy run: time lines and the closing table) and 810-815 (`use_jit` cold rows 0-1, then the warm row 0), and the
hand-built row in `format_newton_iter`'s docstring (`backend/optimizer_display.py` lines 386-407): `%.1e` cells, one
long line without a unit, the `[retr … obj …]` bracket, cells left out, `compile 0`, the table once at the end; (2) the
cost: nothing measurable on the laptop CPU (my re-run +0.4 / −0.1 / +0.2 %), no GPU measured; (3) the `use_jit`
finding: on this small problem the eager phases around CG dominate (`forms` 41 %, the cone 29 %, `gradient` 13 %, CG
1 %), a design question outside the plan; (4) the frontend's `TypeError` for a non-bool `timing` (the backend
unchecked); (5) release text: the CHANGELOG's `[2026.5.0]` intro still headlines only the rename, no
`release_notes.md` line, the glossary entries deferred to (C); (6) the mirror exemption is gone.

VERDICT: APPROVE
