# R3 review — the letter API (plan draft 25, §7 row R3; §1.3, §1.4 L7, §4.5, §5.2, §5.3, D-2 form (c))

Reviewer, 2026-11-12. Reviewed: the working tree on `45f82f09` (HEAD; its code tree is R2's `5153a0be`), which holds R3
uncommitted, and `R3_handoff.md` with `R3_name_hunks.tsv`. Every check below is my own, run independently of the
implementer's runs. My scripts and outputs are in the session scratchpad under `r3rev/`, not in the repo.

## (a) The inversion at every `in update` site and every loop constant

- **Rows.** I extracted the 130 LETTER-API rows from the inventories myself (inv 2: 37 `LETTER-API`; inv 1: 93 `LA`).
  `R1_hit_table.tsv` carries no LETTER-API rows: they live in inv 1 and inv 2 only. HEAD's line numbers equal the
  inventories' (R2 kept every line count).
- **Line alignment HEAD → tree** (`difflib`, per file), against those rows:
  - every changed HEAD line is an LA row, except 15 lines that are rewrap continuations of an LA row's sentence
    (`padded_frames.py:14, 753`, `_padded_frame_oracles.py:129`, `test_padded_frame.py:42, 43, 904, 981, 983`,
    `test_dispatch.py:2059–2061, 2425–2426, 2457–2458`), the second line of the removed three-line check
    (`% (update,))`, twice) and the uniform module's import line. None of those lines carries an inventory row of any
    tier (checked against both inventories);
  - the 20 LA rows left unchanged are 14 pass-throughs (`update=update`, `subTest(update=update)`, …) and the six
    `'L'`/`'R'` gates. That matches Appendix A;
  - the five §5.2 test files have insertions only.
- **The `in update` sites** (my grep of HEAD and the tree): HEAD has 14. The three Tucker-store gates became `'D'`
  (`padded_frames.py:640`, beside `nD1 - nD0`; `uniform_padded_frames.py:566`, beside `PhiD`/`XiD`; the oracle's
  `:93`, beside `Dc`/`nD`). The three TT-shaped-store gates became `'U'` (`padded_frames.py:671`, beside `nU1 - nU0`;
  `uniform_padded_frames.py:601`, beside `PhiU`/`XiU`; the oracle's `:119`, beside `Uc`/`nU`). The six `'L'`/`'R'`
  gates and the two `letter not in update` checks are unchanged. Each new letter matches the widths and products on its
  own line.
- **Old letter strings left** (`'ULRO'`, `'UOLR'`, `'UL'`, `'RO'`, any 2–4-letter `[DULRO]` string, and unquoted
  `ULRO|UL|RO|LRO|ULO` words) in `t3toolbox/`, `tests/` and `examples/`: only inside `_update_stores`' message and
  the two refusal tests, on purpose.
- **Loop constants.** `_STORE_LETTERS` at the truncation checks (both steps) and the all-store passes (oversampled,
  truncating, simultaneous; both loops), `_COLOURS` / `_COLOURS[0]` / `_COLOURS[1]` at the pairs and the final pass.
  `_COLOURS` is `(('D', 'L'), ('U', 'R'))`: 2026.4.0's `('UL', 'RO')` with `U→D`, `O→U`, in the same pass order.
- **Tests' letter arithmetic, by hand.** `('LRO', 'ULO')` → `('U', 'L', 'R')`, `('D', 'U', 'L')`; `test_padded_frame.py:731`'s
  pairs `('D', 'down'), ('U', 'up')`; the oracle's `zip(('D', 'L', 'R', 'U'), dense_a, dense_b)` follows the dense
  frames' own order `(D, phi, psi, U)` and only `.values()` is ever read (all six callers); the recorded schedules are
  canonical literal tuples, never the module constants (so a wrong constant cannot pass its own test).

## (b) `_update_stores`, run myself

The helper and its constants are byte-identical to §1.3's block (one extra blank line before the `def`). Measured:

| input | result |
|---|---|
| `'ULRO'`, `'UL'`, `'DL'`, `'D'`, `''` | `ValueError`: base message + the 2026.4.0 sentence |
| `('U', 'O')`, `['L', 'O']` | `ValueError`: base message + the 2026.4.0 sentence |
| `('D', 'D')`, `()`, `('X',)`, `('d', 'u', 'l', 'r')` | `ValueError`: base message only |
| `('U', 'L')`, `['U', 'L']` | accepted: `('U', 'L')` (up and left, the new meaning) |
| `('D',)` | `('D',)` |
| `('L', 'D')`, `('D', 'L')`, `{'D', 'L'}`, `frozenset('LD')` | `('D', 'L')` |
| a generator | `TypeError` at `len` |

The texts are §1.3's verbatim: "update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without
repeats (D down, U up, L left, R right), e.g. ('D', 'L'); got %r", plus "; that is the 2026.4.0 form, whose 'U' was
the Tucker store (now 'D') and 'O' the TT-shaped store (now 'U'): swap each letter and pass a tuple, e.g. 'UL' -> ('D',
'L'), not ('U', 'L'), which now means up and left".

**Judgement call 1 (one sentence vs. two messages): the implementer is right.** The plan's FINAL text is not
ambiguous. §1.3's code appends ONE sentence under `if isinstance(update, str) or 'O' in update`; its prose says "every
`str` gets the 2026.4.0 sentence above" and "only a collection holding `'O'` raises"; D-2 says "a `str` is refused,
with the 2026.4.0 sentence for every `str`; a 2026.4.0 tuple of letters raises only if it holds `'O'`". The sentence
already names `'O'`. The two-message reading comes from sample batch 3's edit 3, which quoted draft 18's `'O'`-only
message; batch 3 §12 item 2 and the session's §13 ("the pointed sentence goes on EVERY `str`") replaced it with the
present one. §4.5 also asks for `'DL'` with the sentence, and the test pins it.

Two notes for later slices, not for R3:
- R5's release notes must quote §1.3's message, not sample edit 3's draft-18 text.
- For a `str` of NEW letters (`'DL'`, `'D'`) the sentence says "that is the 2026.4.0 form": true of the type, not of
  the letters. The plan asks for exactly this (§4.5), so it stays.

## (c) Canonicalization

- **Battery adapter.** B7 runs all 15 subsets plus the permuted `('L', 'D')` (key `s2+s0`) through `upd()` for both
  layers at both targets. In my run, every `s2+s0` array equals its `s0+s2` twin bitwise (ragged 13 arrays and uniform
  10, at the minimal and the slack target), and both equal 2026.4.0's string calls bitwise (d).
- **Direct calls.** `T3Tangent.align_padded` with `('L', 'D')` and `('D', 'L')`: bitwise equal (and different from
  `('U', 'R')`). `UT3Tangent.align_padded` with `('L', 'D')`, `('D', 'L')`, `['L', 'D']`: the kernel receives
  `('D', 'L')` all three times (one static), frames and movements bitwise equal.
- **The jitted ragged step** (`_jitted(tv_align_padded_frame, _STEP_STATICS)`) keys its cache on the caller's
  `update` before the step canonicalizes it: `('D', 'L')` twice, then `('L', 'D')`, gives cache sizes 1, 1, 2 (same
  numbers). Every library caller passes the constants, so "one jit static per set" holds in the library; only a
  direct user of this private wrapper could compile twice. No change needed.

## (d) Battery

`run` with the working tree under test and `--b9 battery_ref/b9`, then `compare battery_ref <mine>`:
**PASS (bitwise)**, 2,190 arrays and 856 texts, 0 errors, adapter flags `NEW=True`, `LETTERS_NEW=True`. B7 holds
16 keys per layer and target (the 15 subsets and the permuted one). The battery script is unchanged since R1.

## (e) §5.2: one assertion per silent site

All 44 sites have an assertion, at `xS`'s structure (down `(3, 6, 2)`, up `(2, 4, 2)`) and the widths
`(8, 7, 5, 6, 3)`. Every expected value comes from `F.data` / `V.data` shapes, the ragged cores' content in the
supercores' leading slots, or the supercore shapes. None is read from another renamed name, and none compares a mask
with ranks computed from it.

**Mutation test.** I undid each silent swap in turn, in a scratch copy of `t3toolbox/`, and ran the seven new tests:

- property names swapped (the `def` lines only): `down_ranks`/`up_ranks` on `T3Frame`, `T3FrameWeights`, `UT3Frame`,
  `UT3Variations`, `UT3FrameWeights`; `nD`/`nU` on `UT3Frame`, `UT3Variations`, `UT3FrameWeights`;
- field pairs swapped package-wide (so the library stays consistent and only the public name flips):
  `down_weights`, `down_mask`, `variations_down_mask`, `down_weight_supercore` with their `up_` twins;
- keyword pairs swapped within the function: `nD`/`nU` in the eight width functions, and `down_ranks`/`up_ranks` in
  `ufv_make_frame_masks` and `frame_has_minimal_ranks`.

21 of the 22 mutations fail at least one new test, which covers 42 of the 44 sites. The one that passes is
`frame_has_minimal_ranks`' rank keywords, as the handoff says: the function returns `False` unless
`down_ranks == up_ranks` and then reads only `down_ranks`, so no verdict can show a swapped call. Its test pins the
keywords and their physical sources, which is all that can be pinned. **No site is missing and none is tautological.**

## (f) The gate-tool changes (`dev/store_rename_check.py`)

I ran HEAD's version of the tool on the tree as well. Its K3 FAILS with 66 hits: 8 inside the refusal functions,
17 on the docs' LETTER-API rows and 41 word tuples on code lines (`_STORE_NAMES`, `_seed_rows('down', …)`, message
labels, the tests' `STORES`). Change by change:

| change | verdict |
|---|---|
| `K3_OLD_LETTERS` (`'ULRO'`, `'UOLR'`, `'UL'`, `'RO'` anywhere in the LA files) | **a strengthening.** HEAD's tool caught only the `'O'`-holding strings; this adds `'UL'` and reaches docstrings and comments. Every multi-letter string is now refused, so it could be `['"][DULRO]{2,4}['"]` (NIT 2) |
| `K3_REFUSALS`, carved out by function | **justified.** `_update_stores` must test `'O' in update` and name the old form; the two refusal tests pass the old forms on purpose. The three names are unique in their files. The plan should name the carve-out (handoff §8.1) |
| the word-tuple pattern in docstrings and comments only | **justified, and it is the plan's text** ("in docs and docstrings"); comments are added because L7 converts comment rows. The 41 code-line hits are B2's store-name literals, not `update` values, and must not change. A code-level word tuple passed as `update` is refused at run time, so the suite would catch it |
| K3-docs split off and gated at R5 | **justified.** The docs' LA rows belong to R5 (§1.4), so R3 cannot clean them. 17 hits, all R5's rows |
| K14 from R3: no DNT token lost in a hunk; R2's tag count repeated with the slice's additions subtracted | **a justified narrowing** of "identical, no exceptions": new test code necessarily adds DNT names (`randn`, `T3Frame`; 291 tokens). A loss still fails, and the tag recount shows none anywhere. The plan's K14 text needs the amendment |
| K15 from R3: difflib of NAME streams against `--prev`, every hunk listed | **plan-conformant, with one hole:** §5.3's cross-check (old − new ⊆ map sources, new − old ⊆ map images ∪ the handoffs' new names) is not implemented (MINOR 2). The TSV is the tool's own output, so the gate is only as good as the reading of it; I read all 36 hunks (helper, constants, annotations, `''.join` → `tuple`, new tests) |

My K15 cross-check by hand, library: old − new = {`not`, `len`, `set`, `raise`, `ValueError`} (the removed check,
rewritten inside `_update_stores`); new − old = {`_update_stores`, `_STORE_LETTERS`, `_COLOURS`, `Collection`,
`Tuple`, `typ`} plus the helper's ordinary Python (`isinstance`, `message`, `tuple`, `c`, …). Tests: the
`join` → `tuple` swap and new code only. Nothing outside the handoff's list.

The tool's R3 run, mine: K0, K3, K8b, K9-converse, K11, K13, K14, K15, K17 (and its shape gates) PASS; K3-docs
REPORT 17 (R5); the rest REPORT or PENDING, as in the handoff.

## (g) The rewrapped prose rows

- Every rewrapped line is ≤ 120 characters. The lines over 120 are exactly the ten the handoff lists in §7.6: the
  helper's signature (§1.3 verbatim), two signature comments, one comment in `padded_frames.py`, one in
  `uniform_padded_frames.py`, four trailing comments in `test_padded_frame.py` and two dispatch rows. The files
  already had many such lines (`padded_frames.py` 17, `test_dispatch.py` 22). The two over-120 lines R3 could easily
  avoid are NIT 1.
- **R4a's locator.** R3 rewrote, rewrapped or adapted LA rows, so §1.4's "a line already equal to its row's proposed
  text after L7" will not hold for them, and the locator will fail loudly on them. That is the safe outcome, but R4a
  needs the rule "skip every LETTER-API row (R3 owns them; Appendix A records them)". The non-LA lines R3 rewrapped
  carry no inventory row of any tier (checked), and tier D is a regex pass. So every non-LA row of R4a will still be
  located. For the session (S1).

## (h) Suite, doctests, docs, floor env

- **Suite** (`pytest tests/ -q -n 3`, t3toolbox env): **1,347 passed, 4 skipped, 48,724 subtests passed, 0 failed**
  (20 min 20 s). The one warning is the intended adam-on-manifold one.
- **CI's three doctest commands:** `pytest --doctest-modules t3toolbox` **241 passed**; `getting_started.rst` pass;
  every `docs/*.md` and `docs/contributor/*.md` page except `doctest_style.md` pass.
- **`sphinx -W --keep-going`**: build succeeded, exit 0, no warning, both on the R3 tree (25 min) and on the
  revision-1 tree (16 min).
- **Floor env** (`tttt`: Python 3.9.19, numpy 1.22.4, jax 0.4.30; plain pytest, three serial shards), the 11 touched
  test modules: `backend/test_padded_frame` + `backend/test_uniform_padded_frame` 91 passed;
  `backend/test_padded_start`, `backend/test_ranks`, `test_aligned_padded_frame_frontend`,
  `test_frame_variations_format`, `test_uniform_frame_variations_format`, `test_weighted`, `test_uniform_weighted`,
  `test_uniform_manifold` 343 passed; `test_dispatch` 78 passed, 1 skipped. **512 passed, 1 skipped, 0 failed.**

## (i) Fenced blocks

All 35 fenced blocks in the 18 changed files are byte-identical to HEAD (whole lines, located through the
HEAD → tree line alignment); the other files are unchanged. The tool's K11 against `v2026.4.0` also passes.

Doctest examples in the changed files: 685; **0 expected outputs changed**; one source changed
(`frame_variations_format.py:1850`, the LA doctest row).

## Findings

**MAJOR.** None.

**MINOR 1 (resolved in revision 1) — the frontend loop's Method section lost its only legend for the letters**
(`frame_variations_format.py:1765`). HEAD read "a ``'UL'`` pass (up and left stores) and an ``'RO'`` pass (right and
down)". The inv 1 row dropped the gloss because its word tuples (`('down', 'left')`) explained themselves; with L7's
letter tuples they no longer do. This is the main user-facing account of the method, and no legend for `D`/`U`/`L`/`R`
appears anywhere in that docstring. R3 owns the row, so if R3 leaves it out, no later slice will restore it. Proposed
text, in the new meaning:
"- then two-colour pairs: a ``('D', 'L')`` pass (the down and left stores) and a ``('U', 'R')`` pass (the up and
right stores), the other colour transported. …". Refill the bullet's two lines as needed (≤ 120). Its store words
join Appendix C's K9 list. The uniform twin points to this docstring, and `align_padded`'s signature comment already
spells the letters out, so nothing else needs it.

**MINOR 2 (resolved in revision 1) — K15's cross-check is missing from `k14_k15_after_r2`.** §5.3's K15 is "difflib … every unequal hunk
listed" **and** "old − new ⊆ map sources and new − old ⊆ map images ∪ the handoffs' new names". R3 passes it (my hand
computation above), but R4a, R4b and R5 will rely on the tool, and R4a's hunks are renames, where the cross-check is
the guard. Minimal form: over the replace and delete hunks (insert hunks are new code, read through the TSV), report
old − new and new − old per file. FAIL any name that is neither a hit-table source/image nor on a `--names` list the
handoff supplies. R3's list: `if`, `not`, `or`, `len`, `set`, `update`, `raise`, `ValueError`, `join` removed;
`_update_stores`, `tuple` added. Show the R3 run in the handoff.

**NIT 1 (resolved).** Two of R3's own new lines could be ≤ 120 cheaply:
- `padded_frames.py:768` (121): move "fits;" to the next comment line;
- `uniform_padded_frames.py:793` (129): e.g. `# the two colours' statics, ('D', 'L') then ('U', 'R'), at the target structure`.

The rest wait for R4a's realignment.

**NIT 2 (resolved).** `K3_OLD_LETTERS` could be `['"][DULRO]{2,4}['"]` in the LA files (form (c) refuses every `str`, so `'DL'`
or `'LR'` would be as stale as `'UL'`); the tree has none outside the carve-outs today.

## For the session (plan amendments; not the implementer's)

- **S1.** §1.4: R4a's locator skips every LETTER-API row of inv 1/inv 2 (R3 applied them; rewrapped/adapted rows do
  not equal "row text after L7"). Appendix A is the record.
- **S2.** §5.3: K3 carves out `_update_stores` and the two refusal tests by function; its word-tuple pattern reads
  docs, docstrings and comments; its docs half gates at R5. K14: "none lost; DNT names that listed new code adds are
  allowed". K15's cross-check admits a slice's listed removed names (code deletions are not map sources).
- **S3.** §5.2's module list adds `test_uniform_manifold` (`from_t3tangent`), as placed.
- **S4.** K9 at R4b: R3's new-meaning lines (Appendix C, 20 lines, plus MINOR 1's) match none of K9's clauses
  (a)–(c); R4b's K9 needs a clause or keep list for them, or it fails on correct text.
- **S5.** R5: the release notes quote §1.3's message, not sample batch 3 edit 3's draft-18 text.

## Pass 2 (revision 1, handoff §0)

All four findings were taken. I re-checked each in the tree:

- **MINOR 1: resolved.** `frame_variations_format.py:1765–1766` reads "a ``('D', 'L')`` pass (the down and left
  stores) and a ``('U', 'R')`` pass (the up and right stores), the other colour transported. …", in two lines of 115
  and 120 characters. The implementer's K9 reading is right: the new store words sit where HEAD had the opposite
  words, which is clause (b).
- **MINOR 2: resolved.** `k15_cross_check` (`K15-cross`, `--names`) reads the replace and delete hunks per file:
  old − new must be hit-table sources and new − old hit-table images, unless listed. My runs: with no list it FAILs on
  exactly the 20 names of my hand computation; with `--names
  if,not,or,len,set,update,raise,ValueError,join,_update_stores,tuple` it PASSes. Note for R4: sources and images are
  global sets, not per-file, which is looser than a per-file map but enough for a cross-check.
- **NIT 1: resolved.**
  - `padded_frames.py:768–770`: the comment now spans three lines, each ≤ 120, the third on the `while` line. Its `#`
    is at column 64, beside 65 above it: the same one-column raggedness as the neighbouring `if not stop` comment in
    HEAD, so it is R4a's realignment.
  - `uniform_padded_frames.py:793` is 117 characters.
- **NIT 2: resolved.** `K3_OLD_LETTERS` is now `['"][DULRO]{2,4}['"]`. K3 still passes.

Re-run on the revised tree:

- gate tool at `--stage R3`: K0, K3, K8b, K9-converse, K11, K13, K14, K15, K15-cross (with the list) and K17 PASS;
- fenced blocks: all 35 in the changed files still byte-identical to HEAD;
- line alignment: the revision touched two more non-LA lines, `padded_frames.py:754` (HEAD; now the `while` line's
  comment) and `frame_variations_format.py:1766` (the bullet's continuation). Neither carries an inventory row;
- `R3_name_hunks.tsv` unchanged, and the NAME stream unchanged by the revision.

## Verdict

No MAJOR or MINOR remains. S1–S5 are for the session.

VERDICT: APPROVE
