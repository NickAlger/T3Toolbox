# W4 handoff: the docs of `newton_cg(hessian='switch')` (the Gauss-Newton-to-exact switch, 2026.5.0)

Implementer's record for the session. Base: HEAD `c97c754f` (W1 `ca13b37c`, W2 `c39b30bc`; the plan
`dev/gn_newton_switch_plan_2026-11-13.md` at draft 5). Nothing is committed. Scope: the plan's §8 row W4 (§7's docs,
CHANGELOG and contributor records) plus CHANGES 5.4 (§4.13's stale `tracking` / `curvature` rows). **Written under
the machine rule** (W3's benchmark running): files read and edited only; no pytest, doctest, sphinx or Python run.
Every check is listed in §4, to run on "GO".

**Changed:** `docs/fitting_and_optimization.md`, `docs/rank_continuation.md`, `docs/api_reference.rst`,
`docs/contributor/curvature_internals.md`, `docs/contributor/deferred_and_rejected.md`, `CHANGELOG.md`,
`t3toolbox/optimizers.py` and `t3toolbox/backend/optimizers.py` (one pointer sentence each, docstrings only).
**New:** this file. Not touched: `docs/glossary.md` (§2, call 10), `docs/release_notes.md` and the CHANGELOG's
`[2026.5.0]` intro paragraph (the release commit), `dev/HANDOFF.md`.

## 1. What was written

| where | what |
|---|---|
| `fitting_and_optimization.md` **§4.14 "Starting with Gauss-Newton and switching to the exact Hessian"** (new, after §4.13; nothing renumbered) | Opens from §4.12. **The two models** (a hand-aligned block: `Δf`, `m_GN(s)`, `m_N(s)` and where each lives on the row; `K_N`, `L`, the corewise case). **The rule** (block: `e_GN`, `e_N`, hit, switch, with the defaults), the sentence in bold, why it works (`O(‖s‖²)` vs `O(‖s‖³)`), and the equivalent form `\|δ − c\| < τ·\|δ\|` (the source note's observation). **What counts, what resets** (countable defined; plan §1.2's table in user terms). **The rounding gate** (block: resolution, floor `= 100·10·resolution`, informative, hit; the three cases; float32's `1.2e-4` and the x64 advice). **At and after the switch** (one-way; a fresh linearization; GN run step for step before, the exact run resumed after; radius carried, `Δ0` / `Δ_max` iteration 0's; references once; the two forcing powers; each call starts in GN). **The options** (§4.12's table style: `hessian` + the four keywords, G-1..G-7 as decided; the unread-option rule; what is exact mode's; `local_model` / factories reject `'switch'`; `least_squares_problem` sizing). **A worked run** (doctest, §3 below; then W2 §4's frontend `verbose=True` headers, copied by `sed` from `W2_handoff.md`, not retyped; the numbers in prose; then W2's random-start transient headers). **What it reports** (the six row fields as a table, the `None` / history-key rules, exact- vs GN-phase rows, the row invariant, `info.lm` the GN view; the three `stats` keys). **The display** (`sw`, `—`, `×`, `⇒ exact`, before `stop:`; the timing cross-reference: `tracking` / `track`, `curvature` on every switch build, a row's overhead `curvature + tracking`). **What it costs** (per point, per tracked step, qualitative overhead; the compile budget, measured; eager tracking under `use_jit`). The **W3 placeholder** `<!-- W3 BENCHMARK TABLE: filled by the session from dev/gn_newton_switch_slices/W3_handoff.md -->`, its own paragraph after "What it costs". **When to use it, and when not** (nonzero residual; each level restarts in GN and pays `k` tracked steps; never-fires cases: `d = 1`, zero residual, first-order retraction near small σ / at a padded start, float32; the rows cannot say which). **Where the rule comes from** (nicks-optimizers, cited as the source note does: URL + commit `1ab10dc`, 2026-11-11; six differences). |
| §4.13 (CHANGES 5.4) | Phase table: `curvature` -- "with `hessian='exact'`, and with `'switch'` on every build, the Gauss-Newton phase's included (§4.14)", `None` under `hessian='gauss_newton'`; `gradient` -- "(in every mode; exact and switch mode run it inside the build)" (was "in both modes"); `tracking` -- "`hessian='switch'`: the tracking product `⟨p, Lᵀ K_N L p⟩` on a tracked Gauss-Newton-phase row (its `switch_pKp`, §4.14)", `None` on "every other row" (was "reserved ... always, for now"). The display legend gains "`track` the switch's tracking product (§4.14)" (paragraph re-wrapped). |
| §2.1, §4.12, §5 | §2.1: the diagnostics bullet names the `sw` segment (§4.14); a `hessian='switch'` bullet after `hessian='exact'`'s. §4.12 "When it helps": "... or `hessian='switch'` picks the moment to change inside one call (§4.14)". §5 "The Newton schedule is yours" names the automatic `'switch'` (plan §7). |
| `rank_continuation.md` | "The per-level schedule is yours too" names `hessian='switch'` ("each level starts afresh in Gauss-Newton, §4.14"); the paragraph re-wrapped. |
| `api_reference.rst` | One clause after the exact-Hessian sentence. |
| frontend `newton_cg` docstring | One sentence after the switch section's display example: "The rule, the options, the cost, when to use it and a worked run: :doc:`/fitting_and_optimization` §4.14." |
| backend `newton_cg` docstring | One sentence ending the switch paragraph: "The user guide, with a worked run: ``docs/fitting_and_optimization.md`` §4.14." (the timing paragraph's pattern). |
| `CHANGELOG.md` `[2026.5.0]` `### Added` | Appended after the timing entry: the switch (the rule, the resets, the defaults, step control, each call in GN, the trajectory statement, the source, §4.14) with sub-bullets: the four options and the unread-option rule; the rows (`hessian` per row, the five fields, the GN view, the row invariant); `stats`; the display and the `tracking` phase; `least_squares_problem` sizing and the model factories; cost and the bitwise statement; the three changed messages. The timing entry's "(`PhaseSeconds.tracking` is reserved for a Gauss-Newton-to-exact switch.)" now reads "(`PhaseSeconds.tracking` times the `hessian='switch'` tracking product, below.)". |
| `contributor/deferred_and_rejected.md` | I-36 gains "**Built in 2026.5.0: `hessian='switch'`**, the in-run schedule ..." (when to switch is settled inside one call; which mode a level starts in stays open). A new Deferred entry, "The Gauss-Newton-to-exact switch: what 2026.5.0 left out" (plan §4 + §9): switching back, a delayed tracking start, a no-product trigger (Fletcher–Xu), a secant `S`, carrying the state across calls, thinning / jitting the tracking (G-4), public helpers (G-6), lift diagnostics on GN-phase rows, the first-order optimizers. |
| `contributor/curvature_internals.md` | "Model-level rules" gains **The Gauss-Newton view** (plan §7's paragraph: `dc.replace(lm, curvature=None)`, bitwise GN, GN treedef via M3's derived mode; one build, one transpose (M2; the transpose-count test); two CG programs per structure, measured; the tracking product as `switch_pKp`, outside `pHp`, so `κ` is untouched; eager in the host loop). The trust region section: the resolution is `_objective_resolution`, which the switch's gate reads at a hundred times the floor. |

## 2. Judgement calls

1. **The heading uses a hyphen**, "Gauss-Newton", not the brief's en dash: every other "Gauss-Newton" in the page,
   the glossary and the anchors uses the hyphen, and MyST's slug drops an en dash (`gaussnewton`). The link slug is
   `#414-starting-with-gauss-newton-and-switching-to-the-exact-hessian` (as §4.13's `#413-...`; the HTML id renders
   without the number). The plan's §7 title ("Switching from Gauss-Newton to the
   exact Hessian") yields to the brief's.
2. **The doctest runs on §4.13's problem, not §4.12's** (plan §7 says §4.12's; W1's review read it the same way).
   §4.13's `probes` / `noisy` / `x_old` are exactly `newton_cg`'s docstring problem (same seeds, same draw order;
   row 0's objective and error table match), so the doctest IS W2 §4's printed run, the docs' worked example per the
   brief, and its claims are the ones W1 and W2 measured and the frontend docstring already asserts in CI on both
   numpy generations. §4.12's `(6, 7, 8)` problem would have been a new, unmeasured outcome, written blind under the
   machine rule. Same structure as the plan names (noisy rank-3 target, rank-2 fit, probe derivatives of order 1).
3. **The doctest's step claim is `sw['newton_steps'] < gntr['newton_steps'] / 2`** against a Gauss-Newton
   trust-region run (the plan: "≤ GN-TR's"): 6 against 25 measured, so the margin is wide, and it says what the
   section claims ("a fraction"). It also shows `fired['pKp'], fired['switch_pKp'] is not None` -> `(None, True)` (the
   tracking product is not in the row's `pHp`). New names only (`warm`, `sw`, `j`, `fired`, `gntr`): checked that
   nothing later on the page uses them.
4. **The expected outputs of the doctest are predicted, not pasted**: the machine rule forbids running it. Every one
   is a structural value or a comparison whose value W1/W2 measured on this run (`switched_at = 2`, count 2, ratio
   3.0e-03, 6 vs 25 steps, `stop_reason 'gradient'`). The GO run confirms them; if any line differs, the GO run's
   output decides, never the prediction.
5. **The display blocks are header lines only** (W2's own random-start block is headers only; the full blocks with
   error tables would add 35 lines of the same table). The random-start block is W2's `switch_track='all'` run; the
   prose says so and why the default prints the same (every GN-phase row there is countable; W1's default-track run
   printed the same ratios and `switched_at = 5`).
6. **The cost paragraph does not repeat the plan §1.5's "30–40 %"**: that is the source's estimate where an exact
   product costs about a Gauss-Newton one; the source note itself says T3Toolbox's 3–4× makes it more. The text says
   "a few percent when CG runs many iterations per step and a large fraction when it runs one or two", and leaves the
   numbers to W3's table. The compile budget is stated as measured (W1 review: two programs per level, none on a
   repeat, none for `switch_tau=0` after GN-TR).
7. **"Where the rule comes from" lists six differences from the source** (gate, post-switch power, more resets,
   tracked rows, `τ = 0` / `k = 0`, no resume), each from the source note; the brief asked for the cross-reference,
   and a reader of both needs the differences. The experiments directory is named, not summarized; the claude.ai study
   is not cited (not public).
8. **Cross-references beyond §7's list**: §2.1 (two sentences), §4.12 (one clause), `rank_continuation.md` (one
   clause: the continuation page is where "each level starts in Gauss-Newton" matters), `api_reference.rst` (one
   clause), and one pointer sentence in each `newton_cg` docstring (as the timing feature's point to §4.13). The two
   docstring sentences mean the module doctests rerun at GO.
9. **§4.13: the `gradient` row too** ("in both modes" -> "in every mode"), the same staleness W2's review M1 fixed in
   the frontend docstring; and the display legend gains `track`. CHANGES 5.4 named only `tracking` and `curvature`.
10. **No glossary edit**: §7 names no glossary term. For the bundle's glossary pass, entries now stale or missing:
    `step control` ("trust region (exact default)": the switch's default too), `forcing term` (the two phases'
    powers), `history` (the switch fields), `stats` (the three switch keys, and timing's key), and candidates for new
    entries: the switch (`hessian='switch'`), countable step, tracking product, rounding gate.
11. **The CHANGELOG entry carries the three changed messages as its last sub-bullet**, not as a `### Changed`
    bullet (the brief: append to `### Added`); the timing entry's "reserved" parenthetical is updated in place (same
    unreleased section). No `release_notes.md` line (its 2026.5.0 section lists no additions; P3's call 13).
12. **The W3 placeholder is an HTML comment**, invisible in the rendered page if left unfilled, placed as its own
    paragraph between "What it costs" and "When to use it". For the session: replace it with a short "**Measured
    downstream.**" paragraph and the table (§4.12 calls T3Polynomial's fits "downstream"), and, if W3 shows a padded
    level holding off, one clause in the "It may never fire" bullet.
13. **`deferred_and_rejected.md` cites the plan by its `dev/` path "archived after the release"**: when the session
    archives the plan, that path needs its `dev/archive/` name. "Carrying the state across calls" is recorded as
    decided against for continuation levels (Nick's answer 8), the rest as deferred.
14. **Numbers in the prose** (6 steps / 30 CG; GN-TR 25 / 90; GN line search 29; exact at 1.0 from the start 5 / 27;
    ratios 1.3e-02 and 3.0e-03, i.e. 78× and 337×; the transient's 3.5–9.6×) are from the W1 / W2 records (W1's
    reviewer reproduced them), not re-measured here; GO check 5 re-measures them.

## 3. The doctest (§4.14), for reading

```python
>>> warm = dict(order=1, max_newton=40, g0_reference='warm_start')
>>> _, sw = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', probes, noisy, x_old, hessian='switch', **warm)
>>> sw['hessian'], sw['switch_track'], sw['stop_reason'] in ('gradient', 'objective_floor')
('switch', 'countable', True)
>>> j = sw['switched_at']
>>> [row['hessian'] for row in sw['history'][j - 1:j + 1]]
['gauss_newton', 'exact']
>>> fired = sw['history'][j - 1]
>>> fired['switch_fired'], fired['switch_count'], fired['switch_e_n'] < 0.25 * fired['switch_e_gn']
(True, 2, True)
>>> fired['pKp'], fired['switch_pKp'] is not None
(None, True)
>>> _, gntr = topt.newton_cg(..., x_old, step_control='trust_region', **warm)
>>> sw['newton_steps'] < gntr['newton_steps'] / 2
True
```

## 4. Verification to run on "GO" (one job at a time; nothing run yet)

Envs: current `t3toolbox` (py3.11, numpy 2.4, jax 0.10.2) and floor `tttt` (py3.9, numpy 1.22, jax 0.4.30); from
the repo root with `PYTHONPATH=$PWD`.

1. **Page doctests, current env**: the touched pages first (`python -m doctest docs/fitting_and_optimization.md
   docs/rank_continuation.md docs/contributor/curvature_internals.md docs/contributor/deferred_and_rejected.md`),
   then CI's full set (`python -m doctest $(ls docs/*.md docs/contributor/*.md | grep -v doctest_style.md)`).
2. **Page doctests, floor env**: the same four pages.
3. **Module doctests, both envs**: `pytest --doctest-modules t3toolbox/optimizers.py t3toolbox/backend/optimizers.py
   -q -p no:cacheprovider` (two docstrings gained a prose sentence; no example changed).
4. **`python -m sphinx -W --keep-going -b html docs <scratch>/html`** (current env): exit 0, no warning; then check
   the rendered page: §4.14's anchor, the three hand-aligned blocks and the five tables render; the HTML comment does
   not show; the `:doc:` link in the frontend docstring resolves.
5. **The prose numbers**, a scratch script on §4.13's problem (numpy): switch 6 steps / 30 CG / `switched_at 2` /
   ratios 1.3e-02, 3.0e-03; GN-TR 25 / 90; GN line search 29; exact at 1.0 from the start 5 / 27; the random-start
   `'all'` run's ratios 9.6, 8.1, 3.5, 8.1e-02, 8.9e-02 and `switched_at 5`. Any difference: fix the prose.
6. `git diff --check` and the 120-column scan of added lines (done before GO: clean, the display lines excepted).
7. Not run: the CI examples command (no example or library code path changed), the full suite (no code changed;
   the two docstring sentences are covered by 3), the slow tier (release gate).

## 5. For the maintainer's REVIEW STOP (the bundle's last stop before the release commit)

1. **§4.14 as a whole**, especially the three hand-aligned blocks, the counting table, and the "When to use it, and
   when not" bullets (the guidance users will act on).
2. **The doctest on §4.13's problem** instead of §4.12's (call 2), and its `< gntr / 2` claim (call 3).
3. **"Where the rule comes from"**: calling nicks-optimizers "the maintainer's", the URL and commit, and the six
   differences (call 7).
4. **The cost wording** without the plan's 30–40 % (call 6); the W3 numbers now stand where the placeholder was (§7).
5. **The cross-references beyond §7** (call 8), including the two docstring sentences.
6. **The CHANGELOG entry** (its length; the messages sub-bullet, call 11; the timing entry's parenthetical).
7. **The deferred ledger**: I-36's addition and the left-out list, with "carrying the state" recorded as decided
   against (call 13).
8. **The glossary items for the bundle's pass** (call 10).
9. Still for the release commit (not W4): the CHANGELOG `[2026.5.0]` intro to headline all three features; the
   version bump; re-cut `release_notes_2026.5.0.md`; archive the plan (and fix call 13's path).
10. **The W3 splice (§7)**: §4.14's "Measured downstream" paragraph and table, §4.13's GPU paragraph, the adopted
    clause in "It may never fire", and the CHANGELOG timing entry's GPU sentence. Also: the frontend `newton_cg`
    docstring's timing section still ends "A GPU overlaps far more than a laptop CPU, so it can lose more: measure
    there before trusting its split" (not in the splice brief; left as is, call W3-3).

## 6. Verification results (GO, 2026-11-14; one job at a time, nothing fixed: every check passed first time)

| check | result |
|---|---|
| the four touched pages' doctests, current env (`t3toolbox`: py3.11, numpy 2.4, jax 0.10.2) | **pass** (`fitting_and_optimization.md`: 108 examples, §4.14's included) |
| the same four pages, floor env (`tttt`: py3.9, numpy 1.22, jax 0.4.30) | **pass** |
| CI's full page set, current env (`docs/*.md docs/contributor/*.md` but `doctest_style.md`) | **pass** |
| module doctests of `t3toolbox/optimizers.py`, `t3toolbox/backend/optimizers.py`, both envs | **5 passed** each |
| `python -m sphinx -W --keep-going -b html docs <scratch>` (current env, 17 min) | **exit 0, build succeeded, no warning**. Rendered: §4.14's section (`id="starting-with-gauss-newton-and-switching-to-the-exact-hessian"`), its tables, the W3 comment present in the HTML source and absent from the visible text; the frontend `newton_cg` page's new sentence links to `fitting_and_optimization.html` |
| the prose numbers, re-measured (scratch `remeasure.py`, numpy, §4.13's problem) | **all as written**: switch 6 steps / 30 CG / `switched_at 2` / ratios 1.3e-02, 3.0e-03; GN-TR 25 / 90 (stop `objective_floor`); GN line search 29; exact at 1.0 from the start 5 / 27; the random start: ratios 9.6, 8.1, 3.5, 8.1e-02, 8.9e-02, `switched_at 5`, identical under `'all'` and the default `'countable'` (call 5's claim confirmed) |
| `git diff --check`; added lines over 120 columns | clean (the display lines excepted) |

The W3 placeholder was left unfilled at this point (filled afterwards from the W3 record, §7). Nothing is committed.

## 7. The W3 splice (after GO; the session's request, W3 record `a1fb9923`)

| where | what |
|---|---|
| §4.14, the placeholder | Replaced by W3 §6.1's text verbatim (copied by `sed` from `W3_handoff.md`, not retyped): **Measured downstream.** (the Darcy ladder, A100, x64, `use_jit`), the four-arm table (Newton steps, CG iterations, apply-equivalents, wall seconds less compile and display, levels at the cap), and the two sentences (fired on every level after 5–8 GN steps, tracking 3 % of its time, errors no worse; the aligned start's 303 vs 231 steps; 20 × 1000 even). It sits between "What it costs" and "When to use it". |
| §4.13 | "A GPU overlaps far more ... measure on the GPU before reading its split." replaced by W3 §6.2's paragraph verbatim (the A100 overhead −1.4 % to +3.1 %, the dispatch-bound small cases, CG's 66 % downstream, the cold compiles). |
| §4.14 "It may never fire" (call 12, W3's optional clause) | **Adopted**, as "... and the switch can hold off until the new singular values have grown, though on the downstream ladders above it fired on every zero-padded level)". "holds off" became "can hold off": W3 §5.5 found no hold-off on any zero-padded level of five switch runs, so the bullet now states the mechanism as a possibility beside the evidence. The bullet re-wrapped. |
| `CHANGELOG.md`, the timing entry's **Cost** | "A GPU overlaps more of the host's work with the device's and can lose more." replaced by W3's sentence: "On an A100 GPU the same cases and a downstream level differed by −1.4 % to +3.1 %, also within the noise." |

Judgement calls:

- **W3-1.** The numbers are W3's, unrecomputed; the tables are pipe tables like the rest of the page (no padded
  columns to align). The §4.14 "What it costs" paragraph's qualitative wording ("a large fraction when CG runs one
  or two") stays: the measured 3 % is a whole-ladder share with many CG iterations per step, which the paragraph's
  "a few percent when CG runs many iterations" already covers.
- **W3-2.** No switch measurement was added to the CHANGELOG's switch entry (W3 offered none; the "matching CHANGELOG
  sentence" is read as the timing entry's GPU caveat). A one-line ladder result (19,794 apply-equivalents against
  24,682 for Gauss-Newton and 33,309 for exact) would fit the entry's **Cost** sub-bullet if the maintainer wants it.
- **W3-3.** The frontend `newton_cg` docstring's timing section keeps its "measure there before trusting its split"
  GPU sentence: outside the splice brief, and a docstring edit reruns the module doctests. For the review stop.

Checks after the splice (one job at a time): `git diff --check` clean, no new prose line over 120 columns; the
fitting page's doctests **108 passed** on the current env and **108 passed** on the floor env; `sphinx -W` **exit 0,
no warning** (the table and the three spliced passages render; the placeholder comment is gone).

## 8. Round 1 (review `W4_review.md`: REVISE, four MINOR, three NIT; all adopted)

| finding | change |
|---|---|
| **M1** | `t3toolbox/optimizers.py`, the timing section's cost paragraph: "A GPU overlaps far more than a laptop CPU, so it can lose more: measure there before trusting its split." -> "On an A100 GPU the same cases and a downstream level differed by -1.4 % to +3.1 %, also within the noise." (ASCII minus, as the sentence before it; resolves W3-3) |
| **M2** | §4.14 *Measured downstream*: "and its errors were no worse than Gauss-Newton's" -> "and at the rank level the validation error selected its errors were no worse than Gauss-Newton's" (the paragraph re-wrapped; W3_handoff §6.1 stays as proposed) |
| **M3** | CHANGELOG, `switch_track`: "`'all'`: the product on every Gauss-Newton-phase step, the non-countable ones reported but never counted" (the sub-bullet re-wrapped) |
| **M4** | `deferred_and_rejected.md`: (a) the reviewer's sentence after I-36's "Built in 2026.5.0" text, verbatim (223 / 267 / 1,625 at the library start; 231 vs 303 at the aligned start), with the preceding sentence's "which mode ... stays open" moved into it, so it is said once; (b) the thinning / jitting item gains the reviewer's reason verbatim (the eager product dispatch-bound at about 80 ms, 12–32 GN applies against about 2 on a CPU, at most 3.3 % of a switch run's net wall) |
| **N1** | §4.14 "It may never fire": "whose new levels the lift drops" -> "whose new directions the lift drops" |
| **N2** | the ledger's plan pointer: "the plan archived under `dev/archive/` with the 2026.5.0 release" (line 41's precedent; resolves call 13's stale path) |
| **N3** | `docs/chunking.md:186`: "`newton_cg(..., hessian='exact')` (or `'switch'`) passes its mode" |

Runs (one job at a time): `git diff --check` clean, no new line over 120 columns; the touched pages
(`fitting_and_optimization.md`, `rank_continuation.md`, `chunking.md`, `contributor/curvature_internals.md`,
`contributor/deferred_and_rejected.md`) **pass** on the current env and on the floor env; the module doctests of
`t3toolbox/optimizers.py` and `t3toolbox/backend/optimizers.py` **5 passed** on each env. No markup changed (plain
prose only), so no docs rebuild. Review-stop item 10's frontend-docstring note and call 13's path are resolved by M1
and N2.
