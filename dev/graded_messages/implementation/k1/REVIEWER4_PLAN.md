# K1 part 1: the fourth reviewer's plan

Reviewer 4 (Claude), 2026-11-26, written before part 1's handover, from the committed tree `78a34d77` (part 0) only; the
working tree was not reviewed (it was being built). Tools: `k1/reviewer_tools/` (section 1). Scratch:
`scratchpad/reviewer4/` (a `git archive` of `78a34d77` in `tree/`; may be cleared).

## 1. My tools, and what each showed on `78a34d77`

| Tool | What it does | On `78a34d77` |
|---|---|---|
| `ref_dense_pins.py --api old\|new` | My own dense model, no library code: the network's dense tensor and the dense tangent built by loops; `z`, `Jp` of the three kinds at input jets by explicit sums over the slots of every mode (clipped to the box, truncated to the order); `M = b**2 f**2` entry by entry. Against it: the reference's `w_tilde`, `f`, `r_block_sumsq`, `Jp_norm_sq`, and `g`, `JtJp` **entry by entry of every block** (each unit direction's dense tangent paired with `M r` or `M Jp`). Weights: none, block, full exact, full with every `W` axis 1, one `W` axis of 1 (two `W` axes), both, both with `W` of 1, an entries block as a Python float. Sizes coincide (`d = 3 = n + 1`, `F = W = (3,)`, every `N_i = 3`; at `(2, 1)` `F = W = (3, 2)`); input jets of three different jet sizes, one beyond the box; tails on both end bonds; random tables. One degenerate configuration (order `(0,)`, a mode of size 1, `W = (1,)`: "exact" is 1 there). Also: each of the 16 seeded functions with `(y, weights)` equals itself with `w_tilde=` **my** seed (each function's own passing of the weights); the 16 that read no weight ignore them. `--api new` adds the refusals (block: `(d,)` at `order=(n,)`, `F` alone, the old `blocks_to_weight` form, a tuple and a list (TypeError); full: an edge of 1, a jet axis of 1, a mode axis of 1, `d-1` arrays, ranks; `y`: `W` of 1, one `W` axis of 1, an edge or jet axis of 1), each in six readers; and prints what the reference does with modes whose full weights disagree in their `W` pattern (INFO). `--jax`: jax in, jax out, same numbers. The old API gets `b f` written out by loops as its one weight. | 1,473 checks, 0 failed (15 s); `--jax` 10, 0 failed. A mutation (the weight applied after the turn in `_dual_jet`) gives 408 failures: the pins can fail. |
| `ref_independence.py --api old\|new [--tests ...]` | (1) static `ast` scan of the reference's code (docstrings dropped): imports, every attribute read off an imported module, any engine or `jets` name; allowed: `probing.dense_probe`, `apply.dense_apply`, `entries.dense_entries`. (2) a profile hook over all 37 functions with both weights: every library function entered outside the reference; `jets`, the engine, the steps, the fitting and uniform layers are FORBIDDEN. (3) the tests: per file, the helpers that reach the reference (call-graph closure) and, per test, its direct reference calls, the reference-reaching helpers it uses, its library calls; tests using such a helper are marked READ. | independent: only the three dense functions, `common`, `contractions._read_order` (structural) and `probing._dense_jet`/`_dense_contract_one_mode` are entered. READ list today: 7 tests (`at`/`gradient` helpers in `test_sampling_definitions.py`, where the module-level brute-force `at` collides by name with a nested helper that calls the reference: a false positive to read past; `dense_sums` in `test_block_sums.py`; `reference` in `test_sampling_variables.py`). |
| `part1_textcheck.py reference\|files\|prose` | `reference`: unit by unit against `78a34d77` **after applying the intended substitutions** (note §4.1's two signature lines, its `y` line, the weights in `_the_seed`, `_dual_jet`, `_weight_squared` calls); prints the residual hunks (the lines to read), signature `#` columns before and after per function, new lines over 120. `files`: every changed hunk of a test file classified REFLOW (same words, other breaks), COMMENTS (comments added/removed, by `tokenize`), CODE. `prose`: every "After" block of `prose_edits.md` §6 searched whitespace-normalized in the new reference (`<to be pasted>` and `...` match anything). | Self-tested on a scratch repository with three planted defects (a shifted comment column, a block dropped in `Jp_norm_sq`, a re-flowed docstring): all three shown. `prose` parses 8 blocks. |
| `audit_c_attack.py AUDIT` | 14 planted conversions in a scratch repository laid out like this one, the implementer's `diff_audit.py --part 1` run on each alone: right ones must PASS, wrong ones FAIL, blind spots are listed. | On a snapshot of the implementer's in-progress audit (not reviewed, only used to test the harness): 14 of 14 as a sound audit should be. Rerun on the delivered audit. |
| `ref_mutations.py TREE SCRATCH` | Driver: one patch per mutant on a scratch copy, my pins (and, on my turn, the part's tests with `-x -n 2`). The intents M1 to M11 are in its docstring; the exact patches are written once the built text is known. | Empty list by design. |

## 2. What part 1's review must establish

From the carry-over, the note's §2.2 (part 1), §2.3 and §6, round 2's finding 12, and my own reading.

1. **The reference reads the two weights right**, by my pins (`--api new`): every case, every kind, both jet shapes,
   the degenerate configuration; the refusals; jax. Then the mutations M1 to M11 on the built text.
2. **Independence.** `ref_independence.py --api new`: no `jets`, no engine, no steps entered; the product
   `b**2 * f**2` formed by the reference's own code (read `_weight_squared`, `_check_block_weight`, `_check_data_space`).
3. **The part's own pins are brute force.** For each new pin of the reference (note §2.2 part 1): its expected side
   is loops (`np.ndindex`) over a dense `z`/`Jp` of the tests' own route, never a reference function. In particular
   **`g`'s pin "through `<g, p> = d/dt f`" must take `d/dt f` as `sum M r Jp(p)` by loops, not as a finite difference
   of `sdef.f`**: `f` and `g` share `_weight_squared`, so a misreading there cancels in `<g, p> = d/dt sdef.f` (see
   §3, Q1). Likewise `JtJp`'s `<JtJp, q> = sum M Jp Jq` by loops.
4. **The test helpers that place a block row** (`ring.weighted` in `test_sampling_definitions.py`, the new
   `weighted_residual_two` / `weighted_two` of `oracle.py`) do it by their own code (an explicit loop over the blocks,
   `np.repeat`), not by a reshape copied from the reference; `oracle.py` imports nothing of the library. Today
   `ring.weighted` relies on numpy's equal-rank broadcasting, which right-aligns a block row: it must change.
5. **The engine comparisons pin the reference's new reading by the old engine:** the reference gets `(b, f)`, the
   engine `weight=jets.blocks_to_weight(..., b, y)`, the full weight written out, or their product; never a weight
   derived from the reference's pair by a test helper that could share its misreading. Read every rewritten
   engine-vs-reference call (the list from `ref_independence.py (3)` and the diff).
6. **Round 2, finding 12:** every test that handed the reference a now-refused form (`y` constant in `W`; weights with
   a structural axis of 1; `TestAgainstTheReference` at `tests/test_engine_data_space.py:139-158`; the
   `TestTheWeightIsOfTheTypeOfZ` patterns 'per slot', 'one per mode', 'edge and first jet axis') is rewritten: the
   pair `(block, full)`, or written out by `np.broadcast_to` with identical values; the old patterns kept once as
   refusals. No assertion or tolerance changed (audit (a), and `part1_textcheck.py files`).
7. **The scripted rewrite of the 37 signatures** changed only the intended lines: `part1_textcheck.py reference`
   leaves residuals only in the docstrings of `prose_edits.md` §6 and the private helpers of §6.7; one `#` column per
   signature, as before; `prose` finds every block. Doctest outputs pasted from a run (rerun the module's doctests).
   The blank line before every `>>>` block after an edited paragraph (carry-over lesson).
8. **The weights audit** (`diff_audit.py --part 1`): rerun, compare with the implementer's `part1_audit.txt`; my
   attack on the delivered audit; then read by hand: every `full_weight_reasons.tsv` line (apply and entries first:
   there a block passed as a full weight gives the same numbers), every `W-GONE` (the audit passes a dropped weight:
   the test may have lost its subject), every `B-NOREASON`/unresolved site. The audit's callee is the bare attribute
   name (`f`, not `sdef.f`): two callees of one name in a scope are not told apart. A weight passed by a string key
   (`**{'full_weight': x}`) is invisible to it: grep the changed tests for `'block_weight'`/`'full_weight'` strings.
9. **No engine code changed** (`git diff --stat` on `t3toolbox/` shows `sampling_definitions.py` only), the suite
   green at the part's end, the per-test tally against `k1/part0_suite_v.log.gz` (new nodes named; none lost but
   by rename), the run's validity (mtimes of changed files against the log's start).
10. **`typ.Any`:** the reference is exempt (`K_BRIEF.md` §5): not counted.

## 3. Questions on the approved design of part 1

- **Q1 (should be settled before the build is reviewed, cheap): `g`'s pin.** Note §2.2 part 1 pins `g` "through
  `<g, p> = d/dt f`, exact along one variation". If `d/dt f` is the reference's own `f` (a finite difference, or `f`
  at a moved point), the pin is circular in the weight reading: `f` (`_weight_squared` directly) and `g`
  (`_the_seed` -> `_dual_jet` -> `_weight_squared`) share the one helper part 1 rewrites, so a block row placed on the
  wrong axes passes it. The existing `test_against_brute_force` does it right (`pair_g += omega2 * r * Jp` by loops);
  the new pins should keep that form. My tool pins `g` whole (every entry) independently anyway.
- **Q2 (minor): the seeded functions.** The note's pins name six variables. Fifteen other public functions read the
  weights through the seed (the tilde and tilde-dot variables, `Kp`, `V_tilde`, `H_tilde`, `V_tilde_dot`,
  `H_tilde_dot`); a slip in one function's own call (`_the_seed(S, sample, y, None, full_weight, ...)`) is caught
  only where an engine-vs-reference test runs that name with a block weight. My law check covers each; the part's
  tests should too (the existing `test_every_name_against_the_reference` does if it keeps a block weight for every
  name), or one law test per function.
- **Q3 (minor): no engine-vs-reference case with both weights in part 1.** The old engine can take `blocks_to_weight(b)
  * f` (rounding-equal to `b**2 f**2`); one such case would pin the reference's product by independent code, beside
  the brute-force pins. Optional.
- **Q4 (minor): the signature line of note §4.1 is 124 characters** (`block_weight: ... # one per block: probe (d,)+F,
  apply F, entries (); None = ones`), in 37 functions; the module keeps to 120 but for three lines. A shorter comment
  (e.g. `# probe (d,)+F, apply F, entries (); None = ones`) keeps the column and the limit. The implementer's choice;
  I will not count it against the prose review unless the text changes meaning.
- **Q5 (informational): mode agreement.** The plan puts the rule "the `d` arrays of a full weight share one pattern of
  size-1 `W` axes" on the engine, the checkers, `pack_batch` and the batch, not on the reference; read mode by mode,
  the reference needs no such rule, so accepting is consistent. If its docstring says "z's shape except that each
  sample axis may be 1" without the agreement, that is true of the reference; no finding unless it claims the rule.
- Nothing in part 1's design looks wrong to me otherwise. The degenerate case ("exact" is z's size, which may be 1) is
  worth one test of the part's own; my tool has it.

## 4. Order of my review (when the handover arrives)

1. `git diff --stat 78a34d77` (only `sampling_definitions.py` in `t3toolbox/`), then read `_weight_squared`,
   `_check_block_weight`, `_check_data_space`, `_modes_of_data`, `_dual_jet`, `_the_seed`, `f`, `r_block_sumsq`,
   `Jp_norm_sq`, `JtJp` whole, before the handover.
2. `ref_dense_pins.py --api new` and `--jax` on the working tree; `ref_independence.py --api new --tests <changed>`.
3. Fill `ref_mutations.py`'s patches from the built text; run the pins on each; on my turn to run tests, the part's
   tests too.
4. `part1_textcheck.py reference`, `prose`, `files <changed tests>`; the module's doctests.
5. The audit: rerun, `audit_c_attack.py` on it, read the TSV, the W-GONE and the unresolved.
6. Read every rewritten test of §6.1's part-1 rows against §2 items 3 to 6.
7. The suite log, the tally, the validity of the run. Then `K1_part1_REVIEW_r1.md`.
