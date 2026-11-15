# W2 review: the display segment and the frontend of `newton_cg(hessian='switch')` (2026.5.0)

Reviewer's record. Base HEAD `3dc15d96` (W1 committed as `ca13b37c`; the plan at draft 4); W2 in the working tree,
uncommitted (`t3toolbox/backend/optimizer_display.py`, `t3toolbox/optimizers.py`, `t3toolbox/backend/optimizers.py`,
`tests/backend/test_optimizer_display.py`, `tests/test_optimizers_frontend.py`, `W2_handoff.md`). Read: `CLAUDE.md`;
the HANDOFF top block; the plan (§2, §2.1, §2.3, §3, §5, §8, CHANGES 2.6 and 4.3) with reviews r1 / r2; `W1_handoff.md`,
`W1_review.md`, `W2_handoff.md`; `P3_handoff.md`, `P3_review.md` (method); `signature_style.md`,
`naming_conventions.md`, `testing_strategy.md`; the whole diff and the code around it (`_format_header`,
`_curvature_segment`, `_num`, `format_newton_iter`, `make_newton_display`, the frontend `newton_cg` /
`least_squares_problem` / `_resolve_chunk_size`, the backend `_check_hessian` / `_check_switch_options` /
`NewtonInfo` / `PhaseSeconds`, the existing mirror / forwarding / timing tests).

## Round 1

### What I ran (my own, one job at a time; current env py3.11 / numpy 2.4.6 / jax 0.10.2 unless stated)

Scripts and logs: the session scratchpad, `w2_review/`. Everything ran on a byte copy of the working tree (`wt0`,
checked identical to the tree's `t3toolbox/` and `tests/`), against `wt_ref`, a `git archive` of `3dc15d96` (no
worktree, `.git` untouched).

| check | result |
|---|---|
| **Bitwise vs `3dc15d96`** (`battery.py` = the W1 reviewer's battery, itself the P3 / P2 reviewers', with the switch fields now recorded on every row and the four new frontend names left out of the frontend signature record; plus my own section `battery_w2.py`). Every printed text goes through a STRICT strip of the segment (`' \| sw (—\|d.de±dd) ×n( ⇒ exact)?'` followed by ` \| ` or the line end; any residue of ` \| sw ` fails the run), so a malformed segment cannot hide. My section: **backend switch runs with the display**, plain and timed (defaults, `'SWITCH'` + line search, `'All'`, `tau=0`, `k=0`, `k=3` + power, `tau=0.9 k=1`, damping + radius, corewise, regularized, `_OffsetProblem` under both tracks, exhausted `'accept'` / `'stop'`, converged early, `max_newton=1`, warm overrides; ragged / uniform × unshared / shared, each plain, padded + cone, `'all'` + line search); **hand-built rows** through `format_newton_iter` (hit, fired, untracked reset, exact phase, none; two table layouts); **cross-tree frontend switch runs**: on the reference, the REFERENCE frontend called with `hessian='exact'` (so its `least_squares_problem` sizes `'exact'`, as W2 maps `'switch'`) and its backend call rewritten to the switch mode and keywords, so its display, diagnostics and rewrap are the reference's own -- ragged warm defaults, random start `'ALL'` + validation, line search `k=3` + power, `tau=0`, `k=0` + power, corewise, a custom callback, silent, uniform `chunk_size='auto'` (the estimator), uniform `None` + validation + `'all'`, shared ragged, padded + cone on both layers, timed + verbose on both layers (header lines, `stats` but `timing`); **errors on non-switch paths** (`timing` type, damping with GN, radius with LS, `'accept'` with TR, bad step control, bad `g0_reference`, a misspelt `swich_tau`, `val_data` alone, three `least_squares_problem` errors). jax x64: the inherited `use_jit` / eager runs, plus frontend switch `use_jit` runs (TR; `'all'` + line search; timed) on both layers and backend switch `use_jit` (both layers) and eager-jax runs | **numpy 432 entries, 47,187 arrays: 11 mismatches, 10 expected + 1 artifact of my battery.** Expected: the frontend `newton_cg`'s invalid-`hessian` message lists `'switch'` (ragged, uniform, and with a bad `timing` too); `least_squares_problem`'s likewise; `least_squares_problem(hessian='switch')` and frontend `newton_cg(hessian='switch')` raised and now run; a frontend `switch_tau=` / `switch_k=` without the mode was `TypeError` (unknown keyword), now `ValueError("newton_cg: switch_tau set without hessian='switch'")` (two entries × type and text). Artifact: one `least_squares_problem` message embeds a geometry's `repr` with its memory address; equal with the address masked. **jax 40 entries, 3,252 arrays: 0 mismatches.** Every backend switch run (rows with all five switch fields, `x`, `stats`, records, the stripped text) and every cross-tree frontend switch run (`x`, `stats` with `diagnostics`, the stripped text, a custom callback's rows) is bitwise the reference's |
| **The segment by arithmetic** (`check_segments.py`, new tree): each printed header of 29 numpy and 3 jax runs against its own row | 252 rows, **201 segments** (12 of them `—`), **11 marks**, **0 problems**: the segment is exactly `sw %.1e(e_N/e_GN) ×count` (`—` iff `e_N` is None), the mark iff `switch_fired`, one mark per run iff `switched_at` is set; it follows the wall clock and precedes ` \| stop: …`; never on a converged, exact-phase or non-switch row; no `κ` on a Gauss-Newton-phase row; every exact-phase row with `pKp` shows `κ` |
| **The display by eye**: the handoff's §4, my numpy texts (`_OffsetProblem` rejections with `—` and `stop: max_newton`; the random start with validation's `train\|val` tables) and the jax `use_jit` texts | reads well; see "The display" |
| **The compile claim** in the frontend docstring (`jit_claim.py`, x64): a DEFAULT Gauss-Newton run (line search) then switch runs at the same structure | GN line search: +1 CG program; a `switch_tau=0` run after it: **+0**; a firing run (`switched_at=2`, 6 steps): **+1**; again: **+0**. The claim holds even against the line-search GN run (the radius is traced) |
| **Mutations** (`mutate.py`, mine, distinct from the implementer's 29; a fresh copy each; the display test file or the frontend switch / keyword / exact classes rerun) | **21 run: 17 caught, 4 missed** -- two real gaps (M2, N1), two equivalent mutants (the segment suppressed when a `κ` segment exists: no row carries both; the switch keywords dropped outside switch mode: the early check raises first). Caught: the mark when `switch_fired is not None`; the segment on the converged line; `e_GN = 0` printing `0.0e+00`; an untracked row printing a ratio; no ` \| ` separator; an ASCII `x`; the mark before the count; a signed ratio; the segment dropped on a reset row; the early check in switch mode always / only in switch mode / after `least_squares_problem`; the switch power overriding `cg_forcing_power`; the lowered mode forwarded to the backend / to `least_squares_problem`; the `'switch'` → `'exact'` map only in the uniform branch / only when spelt lower-case |
| floor env `tttt` (py3.9, numpy 1.22.4, jax 0.4.30), single process: `test_optimizer_display.py`, `test_optimizers_frontend.py`; the doctests of `optimizers.py`, `backend/optimizer_display.py`, `backend/optimizers.py` | **78 passed** (the sizing test's jax branch included); **6 passed** (the new frontend example and the `format_newton_iter` switch row included) |
| module doctests, `pytest --doctest-modules t3toolbox -n 3` | **241 passed** |

The full suite and `sphinx -W` are run after this round's findings went out, on the same snapshot; results in the session
reply and below.

### House style

Signatures: the four frontend keywords sit in their own blank-line group after `timing`, in the backend's order, their
`#` column the `padded_*` / `cone_*` / `timing` column (checked by column), each comment a short default-and-meaning
note in the backend's words. The edited comments of `hessian`, `curvature_damping`, `step_control` and
`least_squares_problem`'s `hessian` stay in their columns. `_switch_segment` mirrors `_curvature_segment` (a pure
`info -> Optional[str]`, `getattr` reads, `_num` cells, a docstring that says what each cell is and when it is `—`);
`_format_header` gains three lines beside the curvature segment's three. The frontend `switching` dict mirrors `timed`
and its comment says why ("likewise": forwarded only when set). No debugging residue; no name changed (every removed
line is an extended sentence, comment or call; the exemption; P3's assertion). Two blank-line slips in the test file
(N2).

### Plan fidelity

Everything §8 assigns W2 is built, and nothing of W3 / W4 is pulled forward (no docs page, no CHANGELOG, no contributor
record; the backend `newton_cg` docstring untouched).

- **Display (§3.2, T10).** `_switch_segment(info)`: `None` unless `switch_count` is set (a Gauss-Newton-phase row);
  `sw <e_N/e_GN> ×<count>` in `%.1e`, `—` when `switch_e_n` is None, either error is non-finite, or `e_GN = 0`; the
  firing row appends ` ⇒ exact`, read from `switch_fired` (G-3). Exact-phase rows keep today's `κ σ ε` segment;
  `_curvature_segment` is untouched (the `κ` line, `pKp / (pHp − pKp)`, reads `pKp` only, which a GN-phase row never
  carries). Outside switch mode every text is byte-identical (battery; T10's unchanged-text test). The
  `format_newton_iter` docstring has the one switch-row example §3.2 asks for.
- **Frontend (§2.1, §2.3, T7, T11).** The four keywords with the backend's defaults (`None`; G-1 names), after
  `timing`, keyword-only; `_check_hessian(..., _NEWTON_HESSIAN_MODES)` then `_check_switch_options` on the lower-cased
  mode, both before `least_squares_problem` (T7's spy; my "check after the setup" mutation fails 16 subtests); zero
  values pass the frontend (G-2; T11 forwards `switch_tau=0.0` and `switch_k=0` alone); `switch_track` values are the
  backend's, case-insensitive, forwarded as given (G-7); each keyword forwarded only when not None, so a call without
  them reaches the backend exactly as at `3dc15d96` (battery; the unedited `test_newton_cg_forwards_each_keyword`
  asserts the whole kwargs dict) and the backend's defaults rule. `least_squares_problem` accepts the three modes and
  sizes `'switch'` as `'exact'`; `estimate_chunk_size` and `_resolve_chunk_size` keep two modes.
- **The `not_yet` exemption** is deleted (the tuple, its loop, the `or name in not_yet`) and the `hessian` regex
  updated (CHANGES 4.3).
- **`backend/optimizers.py`** changes one docstring clause in `NewtonInfo` (judgement call 8): outside §8's W2 file
  list but inside §7's docstring list (`NewtonInfo`: "the five fields"), docstring-only, the P3 precedent (P3 extended
  `NewtonInfo`'s docstring for `timing`'s display line). Accepted; its wording is N4.

### The plan conflict (the session's ruling: ACCEPT) -- verified

§2.1 appends the switch group after `timing` "identical" on both signatures, so P3's `list(front)[-2:] == ['cone_seed',
'timing']` cannot hold. The narrowed assertion, `names[names.index('timing') - 1] == 'cone_seed'` with the unchanged
`KEYWORD_ONLY` assertion, keeps what P3's test was for: `timing` keyword-only (P3's N1 mutation, "`timing` before the
`*`", still fails it) and directly after `cone_seed`. "Last" was incidental to P3 (it was the newest keyword), and the
new `test_the_keywords` pins the rest of the order (`list(front)[-5:] == list(back)[-5:] == ['timing', *switch]`).
The docstring's "the last keyword" is updated with it. No other existing assertion changed: the test-file diff is the
exemption (W1's required delete), the regex (W1's required update, a message change), this assertion and its
docstring, and additions.

### The judgement calls

All eleven are sound; two get a test-side follow-up.

1. *`—` on either error non-finite.* Sound: an infinite `e_GN` would otherwise print `0.0e+00`. (Checking only `e_GN`
   would suffice, since `_num` prints a non-finite ratio as `—`; both is clearer.)
2. *After the `κ σ ε` position, before `stop:`.* Sound: the same slot as `κ`, which is the other per-row model
   diagnostic; a row never carries both.
3. *The count after the row, `×0` included, no `/k`.* Sound: it reads `×1 ×2 ⇒ exact`, a reset is visible, and
   `switch_k` is not on the row. The mark's source is M2.
4. *Lower-cased mode for the checks, the given string forwarded.* Sound (the existing forwarding test pins `'Exact'`
   reaching `least_squares_problem`; my two "lowered mode forwarded" mutations fail).
5. *Forwarded as given.* Sound; the test that claims it is N1.
6. *The map inside `least_squares_problem`.* Sound (§2.3: the estimator keeps two modes).
7. *A full docstring section.* Accepted per the brief; its claims checked below; one stale neighbour, M1.
8. *The `NewtonInfo` clause.* Accepted (above).
9. *The message changes.* Inherent. The new `ValueError` reads `newton_cg: switch_tau set without hessian='switch'`
   (several names sorted and comma-joined), the house form of the `padded_*` group's `newton_cg: padded_tol set
   without padded_from`: the function prefix, "set without", the group's trigger named, `ValueError` for an option
   combination. It is the backend's own message, checked at the call, so frontend and backend users read one text.
10. *No frontend kind-without-two-hole case.* Sound: the frontend takes kind names, and every built-in kind has the
    two-hole; W1's backend T7 covers it.
11. *Test problems.* Sound.

### The display

Read against `format_newton_iter`'s house style: ` | `-separated segments; a lowercase label then its cells, as
`κ … σ … ε …`, `CG …`, `TR Δ …`; `%.1e` cells, the error table's and `κ`'s width (the ratio is unsigned, so `%.1e`,
not `κ`'s `%+.1e`); `—` for "no number" as everywhere in the display; `×n` as the timing line's `trials … ×n`; the mark
` ⇒ exact` says what changes next and is the only non-cell text, once per run. On the printed runs (the handoff's §4;
my numpy and jax texts) the segment is short, ends the line, and the eye finds the switch: the ratio falling from
`~1e+00` (misses, `×0`) to `~1e-02` (`×1`, `×2 ⇒ exact`), then the `κ σ ε` cells on the exact rows. Rejected rows
read `sw — ×0` under the default track and `sw 1.0e+00 ×0` under `'all'`; a last row reads `… | sw — ×0 | stop:
max_newton`. The segment appears only when a row carries `switch_count` (by arithmetic, above).

### The frontend docstring

Each claim checked against W1's backend or a run: the rule and its defaults (the four-line literal block; "four times
better, on two clean steps in a row"); what counts and resets (W1's §1.2, the boundary / truncated / capped steps
counting); the gate `1000·ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)` = `_SWITCH_ROUNDING · _TR_RESOLUTION_FACTOR · _objective_resolution`
with `ε` = `_resolution_eps` (the dtype's `finfo.eps`); the x64 advice; each option and the unread-option rules (the
backend's messages, T7); TR default, radius carried, line search allowed; damping on every exact build; each call
starting in Gauss-Newton; the no-fire cases (§1.6); the cost (one transpose per point, `curvature_precompute`, one
product per tracked step); the compile claim (measured above); the rows / `stats` keys (W1's §3); the two sample
header lines (the handoff's run, abridged with `...`). The example is doctest-safe on both envs (it asserts structure:
the three `stats` values, the rows' `hessian` around `switched_at`, the firing row's fields, `newton_steps` under half
of Gauss-Newton's -- 6 against 29). One stale paragraph beside it: M1.

### Tests

T7 (frontend), T10, T11 present and meaningful: 17 rejection cases, each with its exact backend message (`re.escape`)
and the spy (so each case isolates the check it names, `testing_strategy.md`'s rule 1); forwarding as given and only
when set, zeros included, none in every mode when unset; the sizing spy on both layers plus the estimator itself; the
cross-layer frontend-vs-backend equality with the switch fields; the segment's three forms, the mark, the `κ`-free GN
row, the placement and the converged line on hand-built rows; the end-to-end text with the `switch_tau=0` run equal to
GN-TR's minus the segment. T12: the three edited existing assertions are the plan's (two) and the conflict's (one).
My mutations: M2 and N1 below.

## Findings

No MAJOR finding. Non-breaking holds bitwise, apart from the expected messages; the display adds up row by row.

### MINOR

**M1. The frontend `newton_cg`'s timing section is stale for the switch.** The section W2 sits beside lists "a row's
phases, in the order the `time` line shows them" (`optimizers.py` ~716-727): no `tracking` bullet (since W1 the switch's
tracking product, the time line's `track` cell), and `curvature` is "with `hessian='exact'` the curvature data" (since
W1 every switch-mode build, the Gauss-Newton phase's included, laps `curvature`). W2 makes the switch reachable from
this function, so its own docstring should list the phase. **Fix:** a `tracking` bullet after `trial_retraction`,
`trial_objective` (e.g. "``tracking`` -- ``hessian='switch'``: a tracked Gauss-Newton-phase row's curvature product
(``switch_pKp``); ``None`` on every other row"), and `curvature` "with ``hessian='exact'`` or ``'switch'``". (For W4,
not W2: `docs/fitting_and_optimization.md` §4.13's phase table says `tracking` is "reserved for a Gauss-Newton-to-exact
switch | always, for now" and `curvature` is `None` under "Gauss-Newton" -- both stale since W1; §4.14's slice should
fix them.)

**M2. The mark's source (G-3) is untested.** `" ⇒ exact" if count >= 2` (the mark from the count, not from
`switch_fired`) passes every test: the hand-built fired row has count 2, and the end-to-end runs use `switch_k = 2`, so
count 2 and `switch_fired` coincide everywhere. G-3 made `switch_fired` a row field precisely so the mark need not
infer it. **Fix:** in `test_the_segment`, a row with `switch_count=2, switch_fired=False` (a `switch_k=3` run's second
hit) prints `sw … ×2` with no mark, and a row with `switch_count=1, switch_fired=True` (`switch_k=1`) prints the mark.
Criterion: the mutation fails.

### NIT

**N1. The "as given" type check cannot fail on builtin values.** `test_forwarded_as_given_and_only_when_set` asserts
`type(forwarded[name]) is type(value)`, but with `0.0`, `0`, `0.875`, `'all'` a `float()` / `int()` conversion keeps
the type: my "power forwarded as `float()`" mutation passes. A conversion would matter for the power (W1 passes it
unconverted so `switch_k=0` is the exact run with that power bit for bit; under numpy 2 a Python float to an
`np.float32` power changes the forcing term's dtype). **Fix:** give the alone-values numpy types (`np.float64(0.0)`,
`np.int64(0)`, `np.float32(0.875)`); `float(np.float64(x))` is a builtin `float`, so the check then bites.

**N2. Blank lines** in `tests/test_optimizers_frontend.py`: three before `def _switch_problem` and one before
`if __name__ == "__main__":` (the file had two; PEP 8's two).

**N3. "Ends each Gauss-Newton-phase header" is slightly loose** (module docstring, `make_newton_display`'s **Switch
rows**, the frontend's *What it reports*): a run's last row ends with ` | stop: …` after the segment, and a converged
Gauss-Newton-phase row prints its short line with no segment. "each Gauss-Newton-phase step row's header carries ... (before a last row's
`stop:`)" would be exact. Likewise `_switch_segment`'s "(`None` on every other row: only those carry `switch_count`)":
a converged Gauss-Newton-phase row carries the count too (its segment is never printed because the converged line
returns first).

**N4. `NewtonInfo`'s clause** "(the display prints a Gauss-Newton-phase row's as its ``sw`` segment, …)" leaves
"row's" without a noun on first read; e.g. "(the display prints them as a Gauss-Newton-phase row's ``sw`` segment:
…)".

Round 1, completed after the findings went out (same snapshot): **full suite `pytest tests/ -q -n 3`: 1,407 passed, 5
skipped, 49,057 subtests (23 min 46 s)**; the one warning is the pre-existing adam-on-manifold one.

VERDICT (round 1): REVISE

## Round 2

The implementer's round 2 (`W2_handoff.md` §7): docstrings, tests and whitespace. Checked on a fresh byte copy of the
tree (`wt1`).

| finding | change | my check |
|---|---|---|
| **M1** | the frontend timing section: `curvature` "with ``hessian='exact'`` or ``'switch'``"; a `tracking` bullet after the trials ("a tracked Gauss-Newton-phase row's curvature product (``switch_pKp``); ``None`` on every other row"); `gradient`'s "(in both modes)" → "(in every mode)", the same staleness one line down | read against `PhaseSeconds` (W1): accurate, in the time line's order |
| **M2** | `test_the_segment`: count 2 with `switch_fired=False` prints `sw 2.5e-01 ×2`, no mark; count 1 with `switch_fired=True` prints `×1 ⇒ exact` | "mark from the count (>= 2)": **caught** |
| **N1** | the alone-values `np.float64(0.0)`, `np.int64(0)`, `np.float32(0.875)`, `'all'` | "power forwarded as `float()`", "`switch_k` as `int()`", "`switch_tau` as `float()`": **all caught** |
| **N2** | two blank lines each side | read |
| **N3** | module docstring, `make_newton_display`'s **Switch rows**, the frontend's *What it reports*: each Gauss-Newton-phase **step** row's header carries the segment, before a last row's stop reason; `_switch_segment`: the converged row carries the count but its short line never reaches the segments | read: exact |
| **N4** | "(the display prints them as a Gauss-Newton-phase row's ``sw`` segment: …)" | read |

| run (round 2, one job at a time) | result |
|---|---|
| library code unchanged apart from docstrings (`ast_check.py`: each touched module's AST with docstrings blanked, round 1 vs round 2) | **equal** for all three modules, so round 1's batteries (numpy, jax), segment check and compile check stand |
| my six round-1 survivors and the two new forwarding mutations (`mutate_r2.log`) | the three real gaps **caught**; the two equivalent mutants still pass, as they must (no row carries both `κ` and `sw`; the early check raises before any switch keyword could be dropped) |
| touched tests (`test_optimizer_display.py`, `test_optimizers_frontend.py`), `-n 3` | **78 passed**, 135 subtests |
| module doctests, `-n 3` | **241 passed** |
| floor env `tttt`, the same two files; the three touched modules' doctests | **78 passed**; **6 passed** |
| `sphinx -W --keep-going -b html docs` | **exit 0, no warning** (15 min); the frontend `newton_cg` page renders the switch section, its literal blocks, the new `tracking` bullet and the example |

No new finding. The round-1 full suite covers the round-2 library (identical code); the two edited test files pass on
both envs.

VERDICT: APPROVE
