# W4 review: the docs of `newton_cg(hessian='switch')` (2026.5.0)

Reviewer's record. Base: HEAD `a1fb9923` (W1 `ca13b37c`, W2 `c39b30bc`, the W3 record); W4 in the working tree. Read:
`CLAUDE.md`; `dev/HANDOFF.md`'s top block; the plan (draft 5: §0-§3, §6, §7, §8's W4 row, CHANGES 2.3 and 5.4) and its
reviews; `W1_handoff.md`, `W1_review.md`, `W2_handoff.md`, `W2_review.md`, `W3_handoff.md` (§6 the spliced text), the
two summaries in T3Polynomial `data/w03_switch_2026_11/{w03_summary.md,gpu/summary.md}`, the source notes; the code as
built (`backend/optimizers.py` `NewtonInfo`, `_objective_resolution`, `_check_switch_options`, `_switch_sample`,
`_gauss_newton_view`, the loop 2023-2295; `backend/optimizer_display.py` `_switch_segment`, `_format_header`, the
time line; the frontend `newton_cg` and `least_squares_problem`); nicks-optimizers at `1ab10dc` on GitHub (public;
`experiments/gn_newton_switch/`, `dev/TODO.md` exist there). Every run below is mine, one job at a time.

## Style

- **The three equation blocks** (§4.14 lines 888-890, 900-903, 933-936), checked by code-point column: the `=` / `:`
  column and each comment column are aligned in every block (cols 9 / 47 / 80; 7 / 47; 12 / 45); no wide characters.
  Hand-made, compact, and they read with the prose.
- **The tables**: pipe tables like the rest of the page; the long rows are tables and display lines only, as in §4.13
  (`git diff --check` clean; nothing else over 120 columns).
- **Voice**: the page's (short declaratives, the measured fact beside each claim, "Here it differs in six ways").
  §4.14 is long (≈ 250 lines) but each part is one of plan §7's items; nothing is padding.
- **Re-wraps** (§2.1, §4.13's legend, §5, `rank_continuation.md`): word-diffed, no word changed but the insertions.
- **CHANGELOG**: appended to `### Added` after the timing entry, nothing reordered; the entry follows the timing
  entry's form (bold lead, sub-bullets). The timing entry's two edits are in place.
- **Anchor** (judgement call 1): MyST 5.1's `default_slugify` gives
  `414-starting-with-gauss-newton-and-switching-to-the-exact-hessian`, the same rule that resolves the glossary's
  `#412-...` links, so the glossary pass can link it. Nothing links it yet (the new references are "§4.14" text, as for
  §4.13).

## Correctness (§4.14 and the touched rows, against the code and the plan)

All checked true unless listed under Findings:

- The two models and the rule: `e_gn = |Δf − predicted|`, `e_n = |Δf − (predicted + ½α²·pKp_track)|` (`_switch_sample`
  with `curvature = 0.5·α²·switch_pKp`); `predicted` is `α·slope + ½α²·pHp` (LS) / `slope + ½·pHp` (TR); hit iff
  informative and `e_N < τ·e_GN`; `count ≥ k` switches, `switched_at = it + 1`; the `|δ − c| < τ|δ|` form is algebra.
- Countable = `accepted and not fallback and (tr_accepted | ls_steps == 0 and not ls_failed)`: the table's four reset
  rows (rejected trial, backtrack, `−g` fallback, forced `'accept'`), the boundary / truncated / capped-CG row (no
  special effect: `countable` reads no CG flag), the converged row (count carried). Tracking from iteration 0.
- The gate: `floor = _SWITCH_ROUNDING (100) · _TR_RESOLUTION_FACTOR (10) · ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)`,
  `ε = _resolution_eps(residual)`; informative = both finite and `max > floor`; the three cases; float32's
  `1000·1.19e-7 ≈ 1.2e-4`. "Testing both moves the count exactly as testing `e_GN` alone" holds: with `e_GN ≤ floor` a
  hit would need `e_N < τ·floor < floor`.
- At and after the switch: one-way; the next point built exact; radius carried, `Δ0` / `Δ_max` and both references
  from iteration 0; `power = switch_power if switched else cg_forcing_power`; each call starts at count 0.
- Options: names, `None` defaults (0.25, 2, 1.0, `'countable'`), types (`numbers.Real` in `[0, 1)`, `Integral ≥ 0`
  not bool, the `str` set case-insensitive), the unread-option rules, `_require_exact_capable` at the call, TR default,
  `curvature_damping` legal, factories / `Problem.local_model` reject, `least_squares_problem` sizes as `'exact'`.
- Reporting: the five fields and per-row `hessian` (where each is set, matching the loop and the `NewtonInfo`
  docstring); history keys only in switch mode (`_newton_scalar_record`); the three `stats` keys and `switched_at`'s
  three special values; the `sw` segment's three forms (`—`, `×n`, `⇒ exact`) and its place before `stop:`, the
  converged line without it; `tracking` / `track` and `curvature` on every switch build (`_local_model_phases` with
  `model_hessian='exact'`).
- Compile budget "at most two programs per call; two at each level of a zero-padded continuation, none again" = W1
  review's measurement; the curvature record's "none for `switch_tau=0` after GN-TR" likewise; "a test counts the
  `kind.transpose` calls" = `test_one_transpose_per_point`.
- §4.13's rows (CHANGES 5.4 and the W2 review's W4 to-do): `curvature`, `gradient`, `tracking` now true; the legend's
  `track` sits where the time line prints it (after the trials, before `cb`).
- "Where the rule comes from": the URL is public, `1ab10dc` exists (2026-11-12 00:34 UTC = 11-11 local), the six
  differences each match the source notes (gate float64 `100·10·ε·max(1,|f|)` on `e_GN`; D10's 0.5; source counts
  backtracked LS steps and has no forced accept; products on every GN iteration incl. rejected trials; `0 < τ < 1`,
  `k ≥ 1`; `resume=`). The ledger's Rosenbrock / Fletcher–Xu / NL2SOL / delayed-start items are the source's
  `dev/TODO.md` and plan.
- **The worked example, re-run** (numpy, `w4r/rerun.py` in my scratch): every printed header cell of both display
  blocks is identical to the page except the per-row seconds (0.05/0.04/… here against the laptop's 0.03/0.03/…; the
  block says it is not checked). Ratios 1.275e-02 (1/78.4) and 2.971e-03 (1/336.6): the prose's 78× and 337×;
  switch 6 steps / 30 CG / `switched_at 2`; GN-TR 25 / 90 (`objective_floor`); GN line search 29; exact at 1.0 5 / 27;
  the random start under `'all'` and under the default: identical headers (seconds masked), `switched_at 5`, ratios
  9.6, 8.1, 3.5, 8.1e-02, 8.9e-02, every GN-phase row countable, row 0 on the boundary and counted.

## The numbers (traced, not recomputed)

- §4.14 "Measured downstream" and §4.13's GPU paragraph are **verbatim** W3 §6.1 / §6.2 (checked by substring).
- The table against `gpu/summary.md` §1 (library): GN 1,625 / 24,682 / 24,682 / 427 / CAP 4; GN-TR 1,618 / 27,023 /
  27,023 / 436 / 4; exact 267 / 10,751 / 33,309 / 275 / 0; switch 223 / 206 + 6,097 = 6,303 / 19,794 / 177 / 0. "Net" =
  total − compile − callback, the column's "less compile and display". `switched_at` 5, 8, 8, 7, 6, 5 = "5–8";
  "3 %" = 5.8 / 177 = 3.3 % (W3 §4.3); aligned 303 vs 231; 20 × 1000 287 / 291 and 308 / 288 = "even".
- §4.13: −1.4 % (d = 6 ragged jit) to +3.1 % (Darcy GN eager); spreads to 9.9 %; 1.62 vs 0.82 s; CG 1.2–3.5 ms/it
  under `use_jit`; Darcy L3 `cg` 66 %; cold compile 43.1 / 42.6 s vs §4.13's own 31 / 36 s. CHANGELOG's timing
  sentence = W3 §6.2's.
- "can hold off at a zero-padded start … fired on every zero-padded level": true. Every level ≥ 1 of both starts is a
  zero-padded `resize` (T3Polynomial `fitting.py:643`; the aligned start adds `padded_from`), and all five switch runs
  switched on every level (W3 §2.5, §5.5; `gpu/summary.md` bottom line).

## Findings

### MINOR

**M1. The frontend `newton_cg` docstring now contradicts the measurement** (the item left for review, W3-3).
`optimizers.py:745-747`: "A GPU overlaps far more than a laptop CPU, so it can lose more: measure there before
trusting its split." The A100 measured −1.4 % to +3.1 %, within the noise, and §4.13 and the CHANGELOG now say so.
**Fix:** replace that sentence with "On an A100 GPU the same cases and a downstream level differed by -1.4 % to
+3.1 %, also within the noise." (ASCII minus, as the sentence before it). Plain prose; rerun the two modules' doctests
on both envs.

**M2. "its errors were no worse than Gauss-Newton's" holds at the selected level only** (§4.14 line ~1085, spliced from
W3 §6.1). At the selected level 4: val_fsr 0.3697 vs 0.3736, test p50 0.0675 vs 0.0689. At the last level, which is
also each arm's best test p50, the switch is worse: 0.0632 vs GN's 0.0614 (W3 §3 says so; `gpu/summary.md` qualifies
it "At the selected level"). **Fix:** "… its tracking took 3 % of its time, and at the rank level the validation
error selected its errors were no worse than Gauss-Newton's." (W3_handoff §6.1 stays as the record of what was
proposed.)

**M3. The CHANGELOG's `switch_track` clause says no product counts under `'all'`.** "`'all'`: the product on every
Gauss-Newton-phase step, reported there but never counted" — under `'all'` the countable steps' products count as
before; only the extra ones do not. **Fix:** "`'all'`: the product on every Gauss-Newton-phase step, the non-countable
ones reported but never counted".

**M4. Two durable W3 findings belong in the ledger** (CLAUDE.md's routing rule; W3 §4.1 names this file for the
jitted product). `deferred_and_rejected.md` I-36 still reads "still missing are GPU timings", and the deferred item
"*Thinning or jitting the tracking*" gives no reason a jitted product would matter. **Fix:** (a) one sentence after
I-36's "Built in 2026.5.0" text: "Measured at the release on the downstream A100 Darcy ladder (user guide §4.14,
*Measured downstream*): from the library start the switch took 223 Newton steps against exact's 267 and Gauss-Newton's
1,625 (at the cap on 4 of 6 levels); from the aligned start (`padded_from`) exact took 231 against the switch's 303, so
which mode a level should start in stays open." (b) in the thinning/jitting item: "on an A100 at 10 points × 100
directions the eager product was dispatch-bound, about 80 ms at every rank (12–32 Gauss-Newton applies, against about 2
on a CPU), yet at most 3.3 % of a switch run's net wall: a jitted product is the candidate if a problem tracks much
longer".

### NIT

**N1.** §4.14 "It may never fire": "a zero-padded start, whose new levels the lift drops" → "whose new directions the
lift drops" ("level" is a continuation level on these pages; `rank_continuation.md` says "the restart's new
directions").

**N2.** `deferred_and_rejected.md`: "`dev/gn_newton_switch_plan_2026-11-13.md`, archived after the release" goes stale
when it is archived (call 13). The file's own precedent (its line 41: "the plan, archived under `dev/archive/` with the
2026.3.0 release") never does: "(the switch plan, archived under `dev/archive/` with the 2026.5.0 release)".

**N3.** Optional: `chunking.md:186` names only `hessian='exact'`, and §4.14 sends the reader there for `'switch'`'s
sizing: "`newton_cg(..., hessian='exact')` (or `'switch'`) passes its mode". (A touched page more; its doctests are
seconds.)

## Completeness

- Plan §7: every item present (the rule in math and words, the counting table, the gate, the keywords in §4.12's
  table style, the defaults, cost, reporting, the never-fires cases with the x64 advice, each call in GN, the doctest,
  §5's bullet, the CHANGELOG items, I-36 and the left-out list, the GN-view paragraph). The doctest on §4.13's problem
  (call 2) is the better choice: it is W1/W2's measured run, and it passed on both envs; its `< gntr / 2` claim has a
  wide margin (6 vs 25).
- The W2 review's W4 to-do (§4.13's `tracking` / `curvature` rows): done, and `gradient` too.
- **The CHANGELOG without switch figures** (call W3-2): right for `### Added`. The file's precedent for a downstream
  ladder result is 2026.4.0's **Measured** paragraph in the version intro, not an Added entry (feature-intrinsic costs,
  like timing's overhead, go in the entry). So the release commit's intro, which headlines the three features, should
  carry a short **Measured** paragraph for the switch (A100 Darcy 10 × 100, library start: switch 223 Newton / 19,794
  apply-equivalents / 177 s net, exact 267 / 33,309 / 275 s, Gauss-Newton 1,625 / 24,682 / 427 s at the cap on 4 of 6
  levels; aligned start: exact cheaper, 231 vs 303 Newton; 20 × 1000: even).
- **Glossary (deferred to the bundle's pass)**: the implementer's list (call 10) misses some. Complete list, stale:
  `step control` (trust region is the `'switch'` default too), `forcing term` (`switch_cg_forcing_power`, the exact
  phase's power), `history` (per-row `hessian`; the five `switch_*` fields, switch mode only; `timing` not a history
  field), `stats` (`hessian`, `switch_track`, `switched_at`; `timing`), `verbose display` (the `sw` segment; the
  `time` line and closing table), `newton_cg` (three Hessian modes), `local model` (one Hessian: rejects `'switch'`),
  `exact Hessian` (pointer to the switch); new: the switch (`hessian='switch'`, `switch_tau` / `switch_k`),
  countable step, tracking product (`switch_pKp`, `switch_track`), model errors `e_GN` / `e_N`, rounding gate,
  `switched_at`, Gauss-Newton view (contributor-facing; the user term is "the row's model"); timing: phase timing
  (`timing=True`, `PhaseSeconds`, `NewtonTiming`), the `timing` stats key, the compile column. The §4.14 anchor above
  is the link target.

## Runs (mine, one job at a time, laptop)

| check | result |
|---|---|
| the four touched pages' doctests, current env (`t3toolbox`: py3.11, numpy 2.4, jax 0.10.2) | **pass** (`fitting_and_optimization.md` 108 examples, §4.14's included, `-v` checked) |
| the same four pages, floor env (`tttt`: py3.9.19, numpy 1.22.4, jax 0.4.30) | **pass** |
| CI's full page set (`docs/*.md docs/contributor/*.md` but `doctest_style.md`), current env | **pass** |
| module doctests `t3toolbox/optimizers.py`, `t3toolbox/backend/optimizers.py`, both envs | **5 passed** each |
| `python -m sphinx -W --keep-going -b html docs <scratch>` | **exit 0, build succeeded, no warning** (18 min). Rendered: §4.14's `id="starting-with-gauss-newton-and-switching-to-the-exact-hessian"`; the W3 placeholder comment gone; the frontend `newton_cg` page's new sentence links `fitting_and_optimization.html` |
| the worked example and the comparison runs, re-measured (numpy) | every number as written (above) |
| `git diff --check`; added lines over 120 columns | clean; tables and display lines only |

## Round 2 (the implementer's `W4_handoff.md` §8)

| finding | change, read against the tree | |
|---|---|---|
| M1 | `optimizers.py:745-747`: the A100 sentence, ASCII minus; no line starts with `-` / `+` (no accidental RST list) | resolved |
| M2 | §4.14 lines 1085-1086: "at the rank level the validation error selected its errors were no worse" | resolved |
| M3 | `CHANGELOG.md:144-145`: "the non-countable ones reported but never counted" | resolved |
| M4 | the ledger, I-36 (lines 220-223: the figures, with "which mode … stays open" folded in, said once) and the thinning / jitting item (12–32 GN applies = W3 §4.1's `m_t` 12.61–32.00; "about 2" = the laptop's 1.94–2.28; 3.3 % = W3 §4.3's largest share) | resolved |
| N1-N3 | "new directions"; line 41's archived-plan form; `chunking.md:186` "(or `'switch'`)" | adopted |

| run (mine, one job at a time) | result |
|---|---|
| the five touched pages (now with `chunking.md`), current env and floor env | **pass**, both |
| module doctests of both optimizer modules, both envs | **5 passed** each |
| `git diff --check` | clean |
| `sphinx -W` | not rerun: the round-2 edits are plain prose and inline code, no directive, role, anchor or list markup (round 1's build stands) |

No new finding.

VERDICT: APPROVE
