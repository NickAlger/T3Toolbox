# R3 handoff — the letter API (plan draft 25, §7 row R3; §1.3, §1.4 L7, §4.5, §5.2, D-2 form (c))

Implementer, 2026-11-12. **Not committed.** The working tree holds R3 on top of `45f82f09` (HEAD; its code tree is
R2's `5153a0be`, the commits since touch `dev/HANDOFF.md` only): 6 library files, 12 test files, and the gate tool
`dev/store_rename_check.py` (the K3 extension, the K14/K15 modes from R3 on). New: this file and
`dev/store_rename_slices/R3_name_hunks.tsv` (K15's listed hunks).

## 0. Revision 1 (after `R3_review.md`, pass 1: no MAJOR, two MINORs, two NITs; all four taken)

| item | what changed |
|---|---|
| MINOR 1 | `frame_variations_format.py:1765–1766`: the Method bullet keeps a legend for the letters, in the new meaning: "a ``('D', 'L')`` pass (the down and left stores) and a ``('U', 'R')`` pass (the up and right stores), the other colour transported. …" (the reviewer's text, refilled at ≤ 120; still two lines). K9 does not flag it (its store words align with HEAD's opposite words, clause (b)), so Appendix C is unchanged |
| MINOR 2 | `dev/store_rename_check.py`: §5.3's K15 cross-check, `K15-cross` (`k15_cross_check`): over the replace and delete hunks, per file, old − new must be hit-table sources and new − old hit-table images, unless listed by `--names`. R3's run, unlisted: FAIL with exactly the reviewer's 20 names; with `--names if,not,or,len,set,update,raise,ValueError,join,_update_stores,tuple`: **PASS** (§6) |
| NIT 1 | `padded_frames.py:768–770`: the comment now runs over three lines (the third on the `while` line), each ≤ 120 (the reviewer's "fits;" move alone gives 125); `uniform_padded_frames.py:793`: `# the ('D', 'L') and ('U', 'R') statics, at the target structure` (117; the suggested text measured 134) |
| NIT 2 | `K3_OLD_LETTERS` is now `['"][DULRO]{2,4}['"]` in the letter-API files: K3 still PASS (the only hits are inside the three carve-outs) |

Re-run after the revision (the reviewer's scope; no test changed): the touched modules' doctests (`frame_variations_format`,
`padded_frames`, `uniform_padded_frames`): 18 passed; doctest wants against HEAD: 0 changed; the gate tool at `--stage
R3` (all checks; K3, K11, K14, K15, K15-cross, K17 PASS); `sphinx -W --keep-going`: **build succeeded**, exit 0, no warning (a first re-run crashed with `docs/autoapi/…/adam.rst does not exist`, a race with a concurrent build on the shared `docs/autoapi`; the clean re-run passed). Appendices A–C regenerated.

## 1. What R3 did

1. **`update` is form (c)** (§1.3, verbatim): `backend/padded_frames.py` gains, beside `_STORE_NAMES`,
   `_STORE_LETTERS = ('D', 'U', 'L', 'R')`, `_COLOURS = (('D', 'L'), ('U', 'R'))` and `_update_stores`;
   `backend/uniform_padded_frames.py` imports the three (the D-46 sibling convention; no uniform copies). Both steps
   (`tv_align_padded_frame`, `utv_align_padded_frame`) take `update: typ.Collection[str] = ('D', 'U', 'L', 'R')`,
   call `update = _update_stores(update)` first (replacing the old three-line check), and so pass the canonical tuple
   on: the ragged step to its gates, the uniform step to `_utv_align_kernel` as its static. The truncation check keys
   on the letter: `zip(_STORE_NAMES, _STORE_LETTERS, …)` / `zip(('down', …), _STORE_LETTERS, …)`.
2. **The simultaneous inversion** at every `in update` site: `'U'→'D'` (the Tucker store's gate: ragged
   `_down_edge` map, uniform step 2's `_u_content_d`, oracle `:93`), `'O'→'U'` (the TT-shaped store's: ragged
   `_up_edge` map, uniform `:601`, oracle `:119`); `'L'`, `'R'` keep their spelling. The loops pass
   `_STORE_LETTERS` (oversampled, truncating and simultaneous passes), `_COLOURS` / `_COLOURS[0]` (pairs, final
   pass); the uniform loop binds `at(_STORE_LETTERS)`, `(at(_COLOURS[0]), at(_COLOURS[1]))`, `at(_COLOURS[0])`.
   `_aligning_pass(update: typ.Tuple[str, ...])` and `_utv_align_kernel(update: typ.Tuple[str, ...])` take the
   canonical tuple. The frontend twins `T3Tangent.align_padded`, `UT3Tangent.align_padded` take the same default and
   forward it (the backend canonicalizes); the frontend loops take no `update`.
3. **Every LETTER-API row** of inv 2 (37) and inv 1 (93) is converted or, where it is a pass-through or keeps its
   letter (`'L'`, `'R'`), left as is: Appendix A maps all 130 rows to their text in the tree (110 in changed blocks,
   20 unchanged: 14 pass-throughs, 6 `'L'`/`'R'` gates). Prose rows take the row's text with L7; the recorded
   schedules become canonical letter tuples (`(('D', 'U', 'L', 'R'), t)`, `(('D', 'L'), t)`, `(('U', 'R'), t)`);
   the doctest `frame_variations_format.py:1850` iterates `('D', 'U', 'L', 'R')` with `update=(store,)` (output
   unchanged).
4. **Tests of the letter API** (§4.5): the invalid cases `()`, `('D', 'D')`, `('X',)`, `('D', 'U', 'L', 'R', 'X')`,
   `('d', 'u', 'l', 'r')` (no 2026.4.0 sentence), and `'DL'`, `'ULRO'`, `'D'`, `('U', 'O')`, `['L', 'O']` (with it),
   the full `'ULRO'` message pinned verbatim; `('U', 'L')` accepted with the NEW meaning (the down store transported,
   its completion keeping the start's row span to 1e-12, while `('D', 'L')` moves it); canonicalization (helper
   output; the step bitwise across `('L', 'U')`, a list, a set, a frozenset); the uniform step hands the kernel the
   canonical tuple for four spellings, bitwise the same frames.
5. **§5.2's unequal-store assertions**: one per silent site, 44 of 44 (§4 below), at `xS`'s structure (down
   `(3, 6, 2)`, up `(2, 4, 2)`) and `from_t3frame(F, 8, 7, 5, 6, 3)` (`nD = 7`, `nU = 5`), against the physical
   frame (`data[0]` rows, `data[1]` middle legs), the ragged source and the supercore widths.

Nothing outside the letter API and §5.2 was touched: no realignment beyond keeping a comment column where the new
type still fits it, no free-prose store word, no fenced block (K11), no diagram.

## 2. The helper and its messages (verbatim)

```python
_STORE_NAMES = ('down', 'up', 'left', 'right')
_STORE_LETTERS = ('D', 'U', 'L', 'R')                       # a frame's stores in physical order (T3Frame.data)
_COLOURS = (('D', 'L'), ('U', 'R'))                         # the two-colour passes (alternate=True), in pass order


def _update_stores(
        update:  typ.Collection[str],   # single store letters, any order, e.g. ('D', 'L'): D down, U up, L left, R right
) -> typ.Tuple[str, ...]:               # the same letters in _STORE_LETTERS order: one jit static per set
    """``update`` validated and canonicalized; ``ValueError`` otherwise, naming the letters."""
    if (isinstance(update, str) or not update or len(set(update)) != len(update)
            or not set(update) <= set(_STORE_LETTERS)):
        message = ("update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats "
                   "(D down, U up, L left, R right), e.g. ('D', 'L'); got %r" % (update,))
        if isinstance(update, str) or 'O' in update:      # every str is a 2026.4.0 call
            message += ("; that is the 2026.4.0 form, whose 'U' was the Tucker store (now 'D') and 'O' the "
                        "TT-shaped store (now 'U'): swap each letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), "
                        "not ('U', 'L'), which now means up and left")
        raise ValueError(message)
    return tuple(c for c in _STORE_LETTERS if c in update)
```

Messages as raised (measured):

- `update=('X',)`: `update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats (D down,
  U up, L left, R right), e.g. ('D', 'L'); got ('X',)`
- `update='ULRO'` (every `str`, and a collection holding `'O'`, e.g. `('U', 'O')`): `update must be a non-empty tuple
  of the store letters 'D', 'U', 'L', 'R' without repeats (D down, U up, L left, R right), e.g. ('D', 'L'); got 'ULRO';
  that is the 2026.4.0 form, whose 'U' was the Tucker store (now 'D') and 'O' the TT-shaped store (now 'U'): swap
  each letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), not ('U', 'L'), which now means up and left`
- `update=('U', 'L')`: accepted, updates up and left (2026.4.0 read it as Tucker and left). A generator raises
  `TypeError` at `len`.

## 3. Every changed site

Appendix A is the row-by-row record of the 130 LETTER-API rows (inventory line at HEAD, its text, where it is now,
the new text). Appendix B lists every changed hunk of the letter-API files (`git diff -U0 HEAD`, before → after, line
numbers old → new). The §5.2 tests are additions only (§4, and the hunks in `R3_name_hunks.tsv`). The NAME-token
changes are exactly the 36 hunks of `R3_name_hunks.tsv` (§5): in the library, the helper and its constants, the
`typ.Collection` / `typ.Tuple` annotations, `_update_stores(update)` replacing the old check, and
`_STORE_LETTERS` / `_COLOURS` at the eleven loop and check sites; in the tests, `''.join` → `tuple` at
`test_padded_frame.py:731` and the new test code. Every other R3 change is inside a STRING or COMMENT token.

## 4. §5.2: the 44 silent sites, one assertion each

All at `xS = TuckerTensorTrain.randn((5, 6, 7), (3, 6, 2), (1, 2, 5, 1))` (seed 0): `F.data[0]` the Tucker factors
`(3, 5), (6, 6), (2, 7)`, so down `(3, 6, 2)`; `F.data[1]` the TT-shaped cores, middle legs up `(2, 4, 2)`; the
widths `(N, nD, nU, rL, rR) = (8, 7, 5, 6, 3)`, so the down supercore is `(3, 7, 8)`, the up `(3, 6, 5, 3)`, the
Tucker variations `(3, 5, 8)` (rows `nU`), the TT variations `(3, 6, 7, 3)` (middle `nD`). Expected values are read
from the physical frame, the ragged source or the supercores, never from another renamed name (no mask-vs-ranks
tautology); each uniform object is built four ways where the API allows (positional, keyword, backend packer by
keyword, dataclass/mask fields by keyword) and must give the same layout.

| § 2 site | count | test (new) | the assertion |
|---|---|---|---|
| `T3Frame.down_ranks`, `.up_ranks` | 2 | `test_frame_variations_format.TestFrameVariationsFormat.test_stores_are_named_by_side` | `== tuple(c.shape[0] for c in F.data[0])` (3, 6, 2), `== tuple(c.shape[1] for c in F.data[1])` (2, 4, 2); `data[0]` are the `(n, N)` Tucker factors |
| `T3FrameWeights.down_weights`, `.up_weights` (fields, keywords) | 2 | `test_weighted.TestT3FrameWeights.test_stores_are_named_by_side` | built by keyword from per-store vectors sized by `F.data[0]` rows / `F.data[1]` middle legs: `W.data[0] is down`, `W.data[1] is up`; `W.is_consistent_with(v)` (down ↔ the TT variations' middle leg, up ↔ the Tucker variations' rows) |
| `T3FrameWeights.down_ranks`, `.up_ranks` | 2 | same | `== (3, 6, 2)` / `(2, 4, 2)` from the physical frame and from `v`'s `tt_variations` middle / `tucker_variations` rows |
| `UT3Frame.down_ranks`, `.up_ranks` | 2 | `test_uniform_frame_variations_format.TestUT3Frame.test_stores_are_named_by_side` | `.tolist() == [3, 6, 2]` / `[2, 4, 2]` from `F.data`; the ragged cores sit in the supercores' leading slots |
| `UT3Frame.nD`, `.nU` (properties) | 2 | same | `(7, 5)`; supercore shapes `(3, 7, 8)`, `(3, 6, 5, 3)` |
| `UT3FrameMasks.down_mask`, `.up_mask` (fields, keywords) | 2 | same | built by keyword (`'mask keywords'` case): widths `(7, 5)`, sums `[3, 6, 2]`/`[2, 4, 2]`, equal to `ufv_make_frame_masks`' |
| `UT3Frame.from_t3frame(nD=, nU=)` | 2 | same | keyword call == positional call `(F, 8, 7, 5, 6, 3)` layout |
| `ufv_conversions.t3frame_to_ut3frame(nD=, nU=)` | 2 | same | `'backend keywords'` case, the same layout |
| `ufv_masking.ufv_make_frame_masks(down_ranks=, up_ranks=)` | 2 | same | ranks from the physical frame; masks equal every frame's `.masks.data` |
| `ufv_masking.ufv_make_frame_masks(nD=, nU=)` | 2 | same | mask widths 7 / 5 |
| `UT3Variations.down_ranks`, `.up_ranks` | 2 | `…TestUT3Variations.test_stores_are_named_by_side` | `[3, 6, 2]` (the TT variations' middle legs) / `[2, 4, 2]` (the Tucker variations' rows), equal to the frame's `data[0]` rows / `data[1]` middles |
| `UT3Variations.nD`, `.nU` | 2 | same | `(7, 5)`; supercores `(3, 5, 8)`, `(3, 6, 7, 3)`; the ragged variations in the leading slots |
| `UT3VariationsMasks.variations_down_mask`, `.variations_up_mask` (fields, keywords) | 2 | same | built by keyword: widths `(7, 5)`, sums `[3, 6, 2]`/`[2, 4, 2]` |
| `UT3Variations.from_t3variations(nD=, nU=)` | 2 | same | keyword == positional layout |
| `ufv_conversions.t3variations_to_ut3variations(nD=, nU=)` | 2 | same | `'backend keywords'` case |
| `UT3FrameWeights.down_weight_supercore`, `.up_weight_supercore` (fields, keywords) | 2 | `test_uniform_weighted.TestUT3FrameWeights.test_stores_are_named_by_side` | `'field keywords'` case; shapes `(3, 7)` / `(3, 5)`; the ragged per-store vectors in the leading slots |
| `UT3FrameWeights.down_ranks`, `.up_ranks` | 2 | same | `[3, 6, 2]` / `[2, 4, 2]` from the physical frame |
| `UT3FrameWeights.nD`, `.nU` | 2 | same | `(7, 5)` |
| `UT3FrameWeights.from_t3frameweights(nD=, nU=)` | 2 | same | keyword == positional `(W, 7, 5, 6, 3)` |
| `ufv_conversions.t3frameweights_to_ut3frameweights(nD=, nU=)` | 2 | same | the packer's supercores |
| `UT3Tangent.from_t3tangent(nD=, nU=)` | 2 | `test_uniform_manifold.TestCrossLayerConverters.test_from_t3tangent_widths_are_named_by_side` | keyword == positional; frame supercores `(3, 7, 8)`/`(3, 6, 5, 3)`, variations `(3, 5, 8)`/`(3, 6, 7, 3)`; the ragged cores in place; dense equal to 1e-12 |
| `ranks.frame_has_minimal_ranks(down_ranks=, up_ranks=)` | 2 | `backend/test_ranks.TestSharedFrameHasMinimalRanks.test_stores_are_named_by_side` | by keyword from the physical frames: `False` at `xS`, `True` at `(3, 5, 3)/(1, 3, 3, 1)`; `down_ranks` are the Tucker ranks; the four stores equal `compute_orthogonal_representation_ranks` |

18 attributes + 4 rank keywords + 16 width keywords + 6 width properties = 44. **Note:** `frame_has_minimal_ranks` is
symmetric in `down_ranks`/`up_ranks` (it returns `False` whenever they differ), so no verdict can show a swapped
call; its test pins the keywords and their physical source only.

## 5. The gate tool (`dev/store_rename_check.py`; the plan's R3 gates)

- **K3, with the R3 extension.** In the letter-API files any string of two to four store letters
  (`['"][DULRO]{2,4}['"]`: 2026.4.0's `'ULRO'`, `'UL'`, `'RO'` and new letters such as `'DL'` alike, since form (c)
  refuses every `str`) is now a K3 pattern too, in code, docstrings and comments (`K3_OLD_LETTERS`); §5.3's
  word-tuple pattern `\('(?:down|up|left|right)'` reads the docs and the code files' docstrings and comments only (see
  §7 item 13); `K3_REFUSALS` carves out `padded_frames.py::_update_stores`, `test_padded_frame.py::test_bad_update`
  and `test_uniform_padded_frame.py::test_update`, by function (ast ranges). K3 reports the code tree (gate R3) and
  the docs (`K3-docs`, gate R5) separately.
- **K14 and K15 from R3 on** (`--stage R3` and later; `--prev REF`, default `HEAD`; `--hunks TSV`; `--write-hunks
  TSV`). K15 diffs the NAME streams (code and doctest examples, the apply script's tokenizer) of the previous slice's
  commit and the tree with `difflib` and requires the unequal hunks to be exactly the listed ones
  (`R3_name_hunks.tsv`; I read every one: §3). K14: no DO-NOT-TOUCH token lost or replaced in any hunk (scope-aware,
  per side), and R2's tag comparison repeated on every changed file — the previous slice's counts against
  `v2026.4.0` with the hit table's images, then the tree's against the previous slice's with this slice's additions
  — so every listed occurrence still stands where R2 left it. New code may add DNT names (public names such as
  `randn`, `T3Frame` in the new tests): 291 such tokens, each in a listed hunk. Before R3 the tool behaves exactly as
  in R2 (`k14_k15` is unchanged and still runs for `--stage R1/R2`).
- **K15's cross-check** (`K15-cross`, revision 1): §5.3's "old − new ⊆ map sources and new − old ⊆ map images ∪ the
  handoffs' new names", over the replace and delete hunks (insert hunks are new code, read through the listed TSV),
  per file; a name outside the hit table's NAME sources/images fails unless the handoff lists it (`--names a,b,…`).

## 6. Gates, battery, suite, doctests, docs, floor env

All in the `t3toolbox` env (Python 3.11, numpy 2.x, jax 0.10) unless stated; tree = the working tree as handed over.

**Battery** (`PYTHONPATH=$PWD python dev/store_rename_battery.py run --out <scratch> --b9 dev/store_rename_slices/battery_ref/b9`,
then `compare dev/store_rename_slices/battery_ref <scratch>`): **PASS (bitwise)**, 2,190 arrays and 856 texts, B9's
cross-version loads included, adapter flags `NEW=True`, **`LETTERS_NEW=True`** (the `update` default is now a
tuple). B7 runs all 15 non-empty store subsets through `upd()` (letter tuples) at the minimal and the slack target,
plus the permuted `('L', 'D')`, for `T3Tangent.align_padded`, `UT3Tangent.align_padded` and the backend steps and
loops: every array bitwise equal to 2026.4.0's string calls (run twice, before and after the last docstring edits).

**Gates** (`python dev/store_rename_check.py --stage R3 --hits dev/store_rename_slices --hunks
dev/store_rename_slices/R3_name_hunks.tsv --names if,not,or,len,set,update,raise,ValueError,join,_update_stores,tuple`,
prev = HEAD `45f82f09`; re-run after revision 1):

| gate | result |
|---|---|
| K3 (code tree, gated from R3) | **PASS**: 0 hits (no `update: str`, no `update='…'`, no `'O' in update` outside the helper, no `'O'`-holding letter string or tuple, no string of 2–4 store letters in the letter-API files outside the three carve-outs, no word tuple in a code docstring or comment) |
| K3-docs (gated from R5) | REPORT: 17, all R5's docs rows (`aligned_padded_frame.md:60, 65, 66, 138`, `glossary.md:366`, `padded_frame_internals.md:46, 116, 122`) |
| K11 | **PASS**: every fenced span byte-identical to `v2026.4.0` (located by text) |
| K14 | **PASS**: 0 DNT tokens lost or replaced in a hunk; the tag comparison on the 18 changed files reproduces R2's (0 differences) with R3's additions subtracted; contract/einsum strings identical; 291 DNT tokens added, all by new test code (public names such as `randn`, `from_t3`, `T3Frame`) |
| K15 | **PASS**: 36 NAME hunks against HEAD, exactly the listed ones (`R3_name_hunks.tsv`): 23 library (the helper and constants, `typ.Collection`/`typ.Tuple`, `_update_stores(update)`, the eleven `_STORE_LETTERS`/`_COLOURS` uses), 13 tests (`''.join`→`tuple` at `test_padded_frame.py:731`, the new tests, two imports). Old−new names: the removed three-line check's tokens (`len`, `set`, `ValueError`, …), which reappear in `_update_stores`; new−old: `_update_stores`, `_STORE_LETTERS`, `_COLOURS`, `Collection`, `Tuple`, and the new tests' names |
| K15-cross (revision 1) | **PASS** with `--names if,not,or,len,set,update,raise,ValueError,join,_update_stores,tuple`. Per file, over the replace/delete hunks: `padded_frames.py` and `uniform_padded_frames.py` old − new {`ValueError`, `if`, `len`, `not`, `or`, `raise`, `set`, `update`}, new − old {`_update_stores`} (the old check, now inside the helper); `test_padded_frame.py` old − new {`join`}, new − old {`tuple`}. Without `--names` it fails on exactly these 20 |
| K17 (+ shape) | **PASS**: 891 checked, 0 bound from a variation; shape-frame 0, shape-converse 0 (shape-point 5, R4b's, unchanged) |
| K0, K8b, K9-converse, K13 | PASS (unchanged from R2) |
| K1, K2, K4–K9, K16 | REPORT, R4a–R5's to-do lists; unchanged from R2 except K9 (293 → 312, §8 item 6, Appendix C) |

**Suite** (`PYTHONPATH=$PWD python -m pytest tests/ -q -n 3`): **1,347 passed, 4 skipped, 48,724 subtests passed, 0
failed** (17 min 35 s; R2: 1,338 — the 9 new tests). The one warning is the intended adam-on-manifold one. The two
files whose test lines I re-wrapped afterwards (`backend/test_ranks.py`, `test_uniform_frame_variations_format.py`,
line splits only) were re-run: 116 passed, 440 subtests.

**Doctests** (CI's three commands): `pytest --doctest-modules t3toolbox` **241 passed**; `python -m doctest
docs/getting_started.rst` **pass**; `python -m doctest` over every `docs/*.md` and `docs/contributor/*.md` except
`doctest_style.md` **pass**. Static comparison of every docstring example (source, want) in the changed files: against
HEAD, 685 examples, **0 wants changed**, 1 source changed (`frame_variations_format.py:1850`, the letter tuple);
against `v2026.4.0`, 2,758 examples in R2's and R3's changed files, **0 wants changed**.

**Docs**: `python -m sphinx -W --keep-going -b html docs <scratch>`: **build succeeded**, exit 0, no warning.

**Floor env** (`tttt`: Python 3.9.19, numpy 1.22.4, jax 0.4.30; plain serial pytest, no `xdist`/`subtests`): the
11 touched test modules (`backend/test_padded_frame`, `backend/test_uniform_padded_frame`,
`backend/test_padded_start`, `backend/test_ranks`, `test_aligned_padded_frame_frontend`,
`test_frame_variations_format`, `test_uniform_frame_variations_format`, `test_weighted`, `test_uniform_weighted`,
`test_uniform_manifold`, `test_dispatch`; `_padded_frame_oracles` is imported by them): **512 passed, 1 skipped, 0
failed** (29 min 49 s); `pytest --doctest-modules t3toolbox`: **241 passed**.

## 7. Judgement calls

1. **One 2026.4.0 sentence, §1.3 verbatim.** The brief says "every `str` gets the 2026.4.0 sentence; a `str` holding
   `'O'` gets the pointed one; see §1.3 and sample edit 3-3". §1.3 (binding, draft 20 on) has ONE appended sentence,
   for every `str` and for any collection holding `'O'`; its text already names `'O'` (the TT-shaped store, now
   `'U'`). Sample edit 3-3's CHANGELOG text (`'O' was the 2026.4.0 letter for …`) is draft 18's message, which batch
   3 §12 item 2 and the session's §13 replaced by the every-`str` sentence. I implemented §1.3 exactly. If the session
   wants a second, `'O'`-only clause on top, it is one `if` in `_update_stores` and one test line.
2. **Placement.** The constants and the helper sit directly under `_STORE_NAMES` (§1.3 / sample edit 4: "beside");
   the uniform module imports them (`from …padded_frames import (…, _STORE_LETTERS, _COLOURS, _update_stores)`).
3. **Signature comments.** The two steps and `T3Tangent.align_padded` carry sample edit 4's form-(c) comment
   (`# store letters, any order: 'D' down, 'U' up, 'L' left, 'R' right; the rest transported`), not inv 1/inv 2's
   row texts (form (b): "the stores updated (store words)" / "the stores updated"); the uniform twins keep their
   `# as …` / `# static; as …`; the kernel says `(canonical letters)` where the row said `(store words)`;
   `_aligning_pass`'s says `# all four stores | ('D', 'L') | ('U', 'R')` (its row with L7). The comment column is
   kept where the new type and default still fit before it (uniform step, uniform frontend, kernel,
   `_aligning_pass`); where they do not (the ragged step, `T3Tangent.align_padded`) the comment follows after three
   spaces, as in sample edit 4. R4a realigns.
4. **Annotations.** Public steps and frontends: `typ.Collection[str]` (any collection is accepted and canonicalized);
   the internal `_aligning_pass` and `_utv_align_kernel`: `typ.Tuple[str, ...]` (they receive the canonical tuple,
   the jit static).
5. **The truncation check keeps its letter column** (`_STORE_LETTERS`), as §1.3 says; inv 2's rows `:581`, `:669`
   (form (b)) dropped it. The uniform check's word tuple `('down', 'up', 'left', 'right')` (B2's literal) stays.
6. **Prose rows: the row's text with L7, then a minimal rewrap when a line would pass 120 characters** (sample batch
   3 edit 6 rewrapped `:2058` the same way; Nick 2026-11-12: alignment and layout within reason, readability first).
   Rewrapped: `padded_frames.py:13` (" The" moved to `:14`), `:768–770` (HEAD `:752–753`; the comment now over three lines, the third on the `while` line),
   `uniform_padded_frames.py:798` (the one-line docstring now two), `manifold.py:186` (+ a line "truncate."),
   `_padded_frame_oracles.py:126–129` (paragraph refilled, 4 → 5 lines), `test_padded_frame.py:40–44` (module
   docstring refilled), `:903–904`, `:912`, `:980–983`, `:1006`, `test_dispatch.py:2058–2061`, `:2424–2426`,
   `:2456–2458`. No line I rewrapped carries another inventory row. **For R4a's locator:** these rows are no longer
   one line equal to "row text + L7"; the locator should skip every LETTER-API row (R3 owns them, §1.4) using
   Appendix A. Left longer than 120 on purpose: the helper's signature line (§1.3 verbatim, 121), the ragged step's
   signature line and `T3Tangent.align_padded`'s (signature comments, 160/164, as sample edit 4),
   `test_padded_frame.py:1029–1032` (trailing comments, 122–126), `test_dispatch.py:2019, 2025` (121, as sample edit
   6 says).
7. **Prose rows whose text I did not take literally:** `frame_variations_format.py:1765` keeps a legend (revision 1,
   the reviewer's MINOR 1): the row dropped HEAD's gloss because its word tuples explained themselves, letter tuples
   do not, so it reads "a ``('D', 'L')`` pass (the down and left stores) and a ``('U', 'R')`` pass (the up and right
   stores)". `uniform_padded_frames.py:793`'s signature comment is shortened (NIT 1). The
   form-(b) phrases "store words" (kernel `:522`, steps) are replaced as in item 3. "all-store" is used where the rows
   say so (`:762`, `:776`, `:833`, `:1805`, the oracle, `test_padded_frame.py:912, 923`, `test_dispatch.py:2424`).
8. **Case labels:** `test_dispatch.py:2020` `'shared, UL'` → `'shared, DL'` (the session's ruling, sample batch 3
   §13); `:2465, 2480` `'the final UL'` → `'the final DL'` by the same token (inv 1's row said `'the final
   down+left'`); `:2008` "(R and O transported)" → "(right and up transported)" (inv 1's row; R3 owns it, §1.4).
9. **The oracle's distance keys** `zip('ULRO', dense_a, dense_b)` → `zip(('D', 'L', 'R', 'U'), …)`: the swapped
   letters in the dense frames' own order `(D, phi, psi, U)`, not canonical (a zip with that tuple; only `.values()`
   is read); its docstring `:176` likewise.
10. **Invalid cases.** 2026.4.0's `''`, `'UU'`, `'X'`, `'ULROX'` become `()`, `('D', 'D')`, `('X',)`,
    `('D', 'U', 'L', 'R', 'X')`; `'ulro'` (a case test) becomes `('d', 'u', 'l', 'r')`; the strings join the
    sentence group with `'DL'`, `'ULRO'`, `'D'`, `('U', 'O')`, `['L', 'O']`. The uniform `test_update`: `()`,
    `('D', 'D')`, `('D', 'L', 'X')`, then `'DL'`, `'ULRO'`, `('U', 'O')` with the sentence.
11. **Tests beyond §4.5's minimum** (the canonicalization and kernel-static tests, the meaning pin by the down store's
    row span): they are what makes "any order, canonicalized before it becomes a jit static" and "the new meaning"
    checkable; each is a few lines.
12. **§5.2 placement.** `UT3Tangent.from_t3tangent` (in §2's silent list, not in §5.2's module list) goes to its
    class's module, `test_uniform_manifold`; the backend packers and `ufv_make_frame_masks` to the module of the
    class they build (`test_uniform_frame_variations_format` gains `import …ufv_conversions`, `test_uniform_weighted`
    too); `frame_has_minimal_ranks` to `backend/test_ranks` with function-local frontend imports (the module's top
    imports stay backend-only). Fixture names avoid the DNT/uniform letters `U`, `B`, `UF`: `uF`, `uV`, `uW`, `uv`.
13. **The gate tool** (§5): K3's word-tuple pattern now reads code files' docstrings and comments only, as §5.3 says
    ("in docs and docstrings"); R1's tool applied it to code lines, where it flagged 41 hits on B2's store-name literals in eight
    files (`_STORE_NAMES`, `_seed_rows('down', …)`, message labels), which are not `update` values and must not change.
    K3 carves out, by function, `_update_stores` (its own `'O' in update`, which §5.3's pattern would otherwise flag)
    and the two refusal tests. K3 is split: the code tree gated from R3, the docs (`K3-docs`) from R5 (their
    LETTER-API rows are R5's, §1.4).

## 8. Where the plan could not be followed as written (reported, not improvised beyond §7)

1. **§5.3 K3 vs §1.3.** K3's pattern `['"]O['"]\s+(?:not\s+)?in\s+update` matches §1.3's own helper
   (`'O' in update`), and §4.5's required refusal tests pass `'ULRO'` and `('U', 'O')` on purpose: carved out by
   function (§5). The plan should name the carve-out.
2. **§5.3 K3's scope.** "In docs and docstrings: a word tuple" — the R1 tool applied it to every code line; corrected
   to docstrings and comments (§7 item 13). Comments are included because L7 converts comment rows too
   (`padded_frames.py:752, 759`).
3. **§1.4's R4a locator** ("accepts a line already equal to its row's proposed text after L7"): the rewrapped rows of
   §7 item 6 and the rows whose text I adapted (§7 items 3, 7) will not match it; Appendix A is their record. R4a
   should skip every LETTER-API row.
4. **§5.2's module list** omits `UT3Tangent.from_t3tangent`'s module (`test_uniform_manifold`), which §2's silent list
   includes; placed there (§7 item 12).
5. **The brief's "pointed" message for a `str` holding `'O'`** has no separate text in §1.3 (§7 item 1).
6. **K9's residue grows by 19** (293 → 312; gate R4b): 20 new lines are R3's own text in the new meaning — the helper's
   message and comment, the new tests' docstrings and comments, the `T3Tangent.align_padded` comment, the dispatch
   docstring's "(right and up transported)" — and one is an alignment artifact of the K9 tool (`_STORE_NAMES = ('down',
   'up', …)`, unchanged since R2, re-aligned against the tag after the lines inserted below it); 2 R2-residue lines are
   gone (rewritten by R3). Listed in Appendix C for R4a/R4b, who own K9: none of them needs a word changed. Revision
   1's legend at `frame_variations_format.py:1765–1766` passes K9 (clause (b)).
7. **§5.3's K15 cross-check** was missing from the R1 tool's later-slice mode; added in revision 1 (`K15-cross`). Code
   a slice deletes is no map source, so the plan should say that the cross-check admits the handoff's listed names.

## Appendix A. The 130 LETTER-API rows (inv 2: 37, inv 1: 93), HEAD line → tree

Generated from the inventories and `difflib` line alignment HEAD → tree. "unchanged": a pass-through or a gate whose letter keeps its spelling (`'L'`, `'R'`). A tree range that holds several rows is the rewritten block they share.

| file:line (HEAD) | HEAD text | tree line(s) | tree text |
|---|---|---|---|
| `t3toolbox/backend/padded_frames.py:13` | `one fixed tensor, a truncating pass to the target ranks, then two-colour passes in ``('UL', 'RO')`` pairs. The` | 13-14 | `one fixed tensor, a truncating pass to the target ranks, then two-colour passes in ``(('D', 'L'), ('U', 'R'))`` pairs. / The ``ALIGN_*`` constants are the loop's defaults, named once here. Everything runs eagerly unless the loop is asked for` |
| `t3toolbox/backend/padded_frames.py:554` | `update:            str = 'ULRO',                             # stores updated: 'U' up, 'L' left, 'R' right, 'O' down; the rest transported` | 572-572 | `update:            typ.Collection[str] = ('D', 'U', 'L', 'R'),   # store letters, any order: 'D' down, 'U' up, 'L' left, 'R' right; the rest transported` |
| `t3toolbox/backend/padded_frames.py:560` | `if not update or len(set(update)) != len(update) or not set(update) <= set('ULRO'):` | 578-578 | `update = _update_stores(update)` |
| `t3toolbox/backend/padded_frames.py:561` | `raise ValueError("update must be a non-empty set of the stores 'U', 'L', 'R', 'O' without repeats; got %r"` | 578-578 | `update = _update_stores(update)` |
| `t3toolbox/backend/padded_frames.py:581` | `for store, letter, old, new, own in zip(_STORE_NAMES, 'UOLR', (nD0, nU0, rL0, rR0), out, (nD, nU, rL, rR)):` | 597-597 | `for store, letter, old, new, own in zip(_STORE_NAMES, _STORE_LETTERS, (nD0, nU0, rL0, rR0), out, (nD, nU, rL, rR)):` |
| `t3toolbox/backend/padded_frames.py:624` | `tuple(nD1[g[0]] - nD0[g[0]] for g in groups), ('U' in update,) * len(groups)))` | 640-640 | `tuple(nD1[g[0]] - nD0[g[0]] for g in groups), ('D' in update,) * len(groups)))` |
| `t3toolbox/backend/padded_frames.py:636` | `tuple(XiL[i] for i in js), tuple(rL1[i + 1] - rL0[i + 1] for i in js), ('L' in update,) * len(js)))` | 652 (unchanged) | |
| `t3toolbox/backend/padded_frames.py:646` | `tuple(XiR[i] for i in js), tuple(rR1[i] - rR0[i] for i in js), ('R' in update,) * len(js)))` | 662 (unchanged) | |
| `t3toolbox/backend/padded_frames.py:655` | `tuple(nU1[i] - nU0[i] for i in range(d)), ('O' in update,) * d))` | 671-671 | `tuple(nU1[i] - nU0[i] for i in range(d)), ('U' in update,) * d))` |
| `t3toolbox/backend/padded_frames.py:687` | `update:     str,                                      # 'ULRO' \| 'UL' \| 'RO'` | 703-703 | `update:     typ.Tuple[str, ...],                      # all four stores \| ('D', 'L') \| ('U', 'R')` |
| `t3toolbox/backend/padded_frames.py:750` | `frame, movement = _aligning_pass(step, frame, hint, project, old, out, sharing, 'ULRO')` | 766-766 | `frame, movement = _aligning_pass(step, frame, hint, project, old, out, sharing, _STORE_LETTERS)` |
| `t3toolbox/backend/padded_frames.py:752` | `if alternate:                                               # ('UL', 'RO') pairs while a whole pair fits; the stop` | 768-771 | `if alternate:                                               # (('D', 'L'), ('U', 'R')) pairs while a whole pair / stop = False                                            #   fits; the stop is tested after each pair (both / while not stop and len(movements) + 2 <= max_passes:   #   below tol) / for update in _COLOURS:` |
| `t3toolbox/backend/padded_frames.py:755` | `for update in ('UL', 'RO'):` | 768-771 | `if alternate:                                               # (('D', 'L'), ('U', 'R')) pairs while a whole pair / stop = False                                            #   fits; the stop is tested after each pair (both / while not stop and len(movements) + 2 <= max_passes:   #   below tol) / for update in _COLOURS:` |
| `t3toolbox/backend/padded_frames.py:759` | `if not stop and len(movements) < max_passes:           # an odd remainder: one final 'UL' pass` | 775-776 | `if not stop and len(movements) < max_passes:           # an odd remainder: one final ('D', 'L') pass / frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, _COLOURS[0])` |
| `t3toolbox/backend/padded_frames.py:760` | `frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, 'UL')` | 775-776 | `if not stop and len(movements) < max_passes:           # an odd remainder: one final ('D', 'L') pass / frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, _COLOURS[0])` |
| `t3toolbox/backend/padded_frames.py:762` | `else:                                                       # 'ULRO' passes, the stop tested after each (the` | 778-778 | `else:                                                       # all-store passes, the stop tested after each (the` |
| `t3toolbox/backend/padded_frames.py:765` | `frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, 'ULRO')` | 781-781 | `frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, _STORE_LETTERS)` |
| `t3toolbox/backend/uniform_padded_frames.py:522` | `update:    str,                       # static: the stores updated; the rest transported` | 522-522 | `update:    typ.Tuple[str, ...],       # static: the stores updated (canonical letters); the rest transported` |
| `t3toolbox/backend/uniform_padded_frames.py:566` | `Y = _u_content_d(g_old, g_new, PhiD[..., Na:], XiD) if 'U' in update else g_old` | 566-566 | `Y = _u_content_d(g_old, g_new, PhiD[..., Na:], XiD) if 'D' in update else g_old` |
| `t3toolbox/backend/uniform_padded_frames.py:582` | `L1, oL, PsiL, mvL = _sweep(xscan, _u_left_edge_step if 'L' in update else _u_left_edge_transport_step, zero_edge, (` | 582 (unchanged) | |
| `t3toolbox/backend/uniform_padded_frames.py:586` | `R1, oR, PsiR, mvR = _sweep(xscan, _u_right_edge_step if 'R' in update else _u_right_edge_transport_step, zero_edge, (` | 586 (unchanged) | |
| `t3toolbox/backend/uniform_padded_frames.py:601` | `Y = _u_content_d(g_old, g_new, PhiU[..., Na:], XiU) if 'O' in update else g_old` | 601-601 | `Y = _u_content_d(g_old, g_new, PhiU[..., Na:], XiU) if 'U' in update else g_old` |
| `t3toolbox/backend/uniform_padded_frames.py:621` | `update:           str = 'ULRO',                                      # static; as tv_align_padded_frame` | 621-621 | `update:           typ.Collection[str] = ('D', 'U', 'L', 'R'),        # static; as tv_align_padded_frame` |
| `t3toolbox/backend/uniform_padded_frames.py:627` | `if not update or len(set(update)) != len(update) or not set(update) <= set('ULRO'):` | 627-627 | `update = _update_stores(update)` |
| `t3toolbox/backend/uniform_padded_frames.py:628` | `raise ValueError("update must be a non-empty set of the stores 'U', 'L', 'R', 'O' without repeats; got %r"` | 627-627 | `update = _update_stores(update)` |
| `t3toolbox/backend/uniform_padded_frames.py:669` | `for store, letter, o, s1, s in zip(('down', 'up', 'left', 'right'), 'UOLR', old, out, own):` | 667-667 | `for store, letter, o, s1, s in zip(('down', 'up', 'left', 'right'), _STORE_LETTERS, old, out, own):` |
| `t3toolbox/backend/uniform_padded_frames.py:776` | `statics:  typ.Tuple,   # as _aligning_pass, at the working structure, 'ULRO'` | 774-774 | `statics:  typ.Tuple,   # as _aligning_pass, at the working structure, all four stores` |
| `t3toolbox/backend/uniform_padded_frames.py:788` | `) -> NDArray:                # 0-d bool: run another ('UL', 'RO') pair` | 786-786 | `) -> NDArray:                # 0-d bool: run another (('D', 'L'), ('U', 'R')) pair` |
| `t3toolbox/backend/uniform_padded_frames.py:795` | `statics:  typ.Tuple[typ.Tuple, typ.Tuple],   # the 'UL' and the 'RO' passes' statics, at the target structure` | 793-793 | `statics:  typ.Tuple[typ.Tuple, typ.Tuple],   # the ('D', 'L') and ('U', 'R') statics, at the target structure` |
| `t3toolbox/backend/uniform_padded_frames.py:798` | `"""One ('UL', 'RO') pair; the stop: both movements below tol (the max over the stack; a NaN never stops)."""` | 796-797 | `"""One (('D', 'L'), ('U', 'R')) pair; the stop: both movements below tol (the max over the stack; a NaN never / stops)."""` |
| `t3toolbox/backend/uniform_padded_frames.py:808` | `statics:  typ.Tuple,   # the 'UL' pass's statics` | 807-807 | `statics:  typ.Tuple,   # the ('D', 'L') pass's statics` |
| `t3toolbox/backend/uniform_padded_frames.py:811` | `"""The odd remainder's final 'UL' pass."""` | 810-810 | `"""The odd remainder's final ('D', 'L') pass."""` |
| `t3toolbox/backend/uniform_padded_frames.py:833` | `statics:  typ.Tuple,   # the 'ULRO' pass's statics, at the target structure` | 832-832 | `statics:  typ.Tuple,   # the all-store pass's statics, at the target structure` |
| `t3toolbox/backend/uniform_padded_frames.py:894` | `statics = (project, hint, shape, masks, split, (None, None), sharing, 'ULRO')` | 893-893 | `statics = (project, hint, shape, masks, split, (None, None), sharing, _STORE_LETTERS)` |
| `t3toolbox/backend/uniform_padded_frames.py:901` | `state = xwhile(_pairs_cond, ft.partial(_pairs_body, (at('UL'), at('RO'))), state, use_jit)` | 900-901 | `state = xwhile(_pairs_cond, ft.partial(_pairs_body, (at(_COLOURS[0]), at(_COLOURS[1]))), state, use_jit) / state = _cond(_single_cond(state), ft.partial(_final_pass, at(_COLOURS[0])), _no_pass, state, use_jit)` |
| `t3toolbox/backend/uniform_padded_frames.py:902` | `state = _cond(_single_cond(state), ft.partial(_final_pass, at('UL')), _no_pass, state, use_jit)` | 900-901 | `state = xwhile(_pairs_cond, ft.partial(_pairs_body, (at(_COLOURS[0]), at(_COLOURS[1]))), state, use_jit) / state = _cond(_single_cond(state), ft.partial(_final_pass, at(_COLOURS[0])), _no_pass, state, use_jit)` |
| `t3toolbox/backend/uniform_padded_frames.py:905` | `state = xwhile(_single_cond, ft.partial(_single_body, at('ULRO')), state, use_jit)` | 904-904 | `state = xwhile(_single_cond, ft.partial(_single_body, at(_STORE_LETTERS)), state, use_jit)` |
| `t3toolbox/frame_variations_format.py:1765` | `- then two-colour pairs: a ``'UL'`` pass (up and left stores) and an ``'RO'`` pass (right and down), the other` | 1765-1766 | `- then two-colour pairs: a ``('D', 'L')`` pass (the down and left stores) and a ``('U', 'R')`` pass (the up and / right stores), the other colour transported. A simultaneous update can cycle with period 2, which the pairs avoid.` |
| `t3toolbox/frame_variations_format.py:1805` | `- ``alternate``: the two-colour switch. ``False`` runs simultaneous ``'ULRO'`` passes, stopping on one movement.` | 1805-1805 | `- ``alternate``: the two-colour switch. ``False`` runs simultaneous all-store passes, stopping on one movement.` |
| `t3toolbox/frame_variations_format.py:1850` | `>>> [bool(float(tangent.align_padded(*old, update=store)[1]) < 1e-8) for store in 'ULRO']` | 1850-1850 | `>>> [bool(float(tangent.align_padded(*old, update=(store,))[1]) < 1e-8) for store in ('D', 'U', 'L', 'R')]` |
| `t3toolbox/manifold.py:175` | `update:            str = 'ULRO',                            # the stores updated ('U', 'L', 'R', 'O'); the rest transported` | 175-175 | `update:            typ.Collection[str] = ('D', 'U', 'L', 'R'),   # store letters, any order: 'D' down, 'U' up, 'L' left, 'R' right; the rest transported` |
| `t3toolbox/manifold.py:186` | `the others (``'UL'`` / ``'RO'`` are the two colours of the loop); output ranks below the frame's truncate.` | 186-187 | `the others (``('D', 'L')`` / ``('U', 'R')`` are the two colours of the loop); output ranks below the frame's / truncate.` |
| `t3toolbox/manifold.py:207` | `sharing=sharing, update=update)` | 208 (unchanged) | |
| `t3toolbox/uniform_frame_variations_format.py:1611` | ```('UL', 'RO')`` pairs).` | 1611-1611 | ```(('D', 'L'), ('U', 'R'))`` pairs).` |
| `t3toolbox/uniform_manifold.py:152` | `update:        str = 'ULRO',                                              # as T3Tangent.align_padded` | 152-152 | `update:        typ.Collection[str] = ('D', 'U', 'L', 'R'),                # as T3Tangent.align_padded` |
| `t3toolbox/uniform_manifold.py:180` | `self.frame.data, self.variations.data, split, tucker_ranks, tt_ranks, sharing=sharing, update=update)` | 180 (unchanged) | |
| `tests/backend/_padded_frame_oracles.py:65` | `def dense_pass(old_dense, cur_dense, Z, stores, update='ULRO', groups=()):` | 65-65 | `def dense_pass(old_dense, cur_dense, Z, stores, update=('D', 'U', 'L', 'R'), groups=()):` |
| `tests/backend/_padded_frame_oracles.py:93` | `W = choose(Q, np.concatenate(blocks, axis=1), Dc[i0][:, s:], nD[i0] - s, 'U' in update)` | 93-93 | `W = choose(Q, np.concatenate(blocks, axis=1), Dc[i0][:, s:], nD[i0] - s, 'D' in update)` |
| `tests/backend/_padded_frame_oracles.py:103` | `phi.append(np.concatenate([phio[k + 1], choose(Q, Y, phic[k + 1][:, r0:], rL[k + 1] - r0, 'L' in update)], 1))` | 103 (unchanged) | |
| `tests/backend/_padded_frame_oracles.py:111` | `psi[k] = np.concatenate([psio[k], choose(Q, Y, psic[k][:, r0:], rR[k] - r0, 'R' in update)], 1)` | 111 (unchanged) | |
| `tests/backend/_padded_frame_oracles.py:119` | `U.append(np.concatenate([Uo[i], choose(Q, content, Uc[i][:, r0:], nU[i] - r0, 'O' in update)], 1))` | 119-119 | `U.append(np.concatenate([Uo[i], choose(Q, content, Uc[i][:, r0:], nU[i] - r0, 'U' in update)], 1))` |
| `tests/backend/_padded_frame_oracles.py:126` | `(``alternate``) ('UL', 'RO') pairs while a whole pair fits under ``max_passes``, stopping after a pair whose two` | 126-130 | `(``alternate``) (('D', 'L'), ('U', 'R')) pairs while a whole pair fits under ``max_passes``, stopping after a / pair whose two movements are below ``tol``, and one final ('D', 'L') pass in a remaining slot unless stopped; or / (not ``alternate``) all-store passes, stopping after the first pass from the truncating one on whose movement is / below ``tol``. Every pass is against ``dense_project(D, S)``. Returns the dense frames and the movement after / every pass."""` |
| `tests/backend/_padded_frame_oracles.py:127` | `movements are below ``tol``, and one final 'UL' pass in a remaining slot unless stopped; or (not ``alternate``)` | 126-130 | `(``alternate``) (('D', 'L'), ('U', 'R')) pairs while a whole pair fits under ``max_passes``, stopping after a / pair whose two movements are below ``tol``, and one final ('D', 'L') pass in a remaining slot unless stopped; or / (not ``alternate``) all-store passes, stopping after the first pass from the truncating one on whose movement is / below ``tol``. Every pass is against ``dense_project(D, S)``. Returns the dense frames and the movement after / every pass."""` |
| `tests/backend/_padded_frame_oracles.py:128` | `'ULRO' passes, stopping after the first pass from the truncating one on whose movement is below ``tol``. Every` | 126-130 | `(``alternate``) (('D', 'L'), ('U', 'R')) pairs while a whole pair fits under ``max_passes``, stopping after a / pair whose two movements are below ``tol``, and one final ('D', 'L') pass in a remaining slot unless stopped; or / (not ``alternate``) all-store passes, stopping after the first pass from the truncating one on whose movement is / below ``tol``. Every pass is against ``dense_project(D, S)``. Returns the dense frames and the movement after / every pass."""` |
| `tests/backend/_padded_frame_oracles.py:140` | `run(working if it < n_over else target, 'ULRO')` | 141-141 | `run(working if it < n_over else target, ('D', 'U', 'L', 'R'))` |
| `tests/backend/_padded_frame_oracles.py:144` | `run(target, 'UL')` | 145-146 | `run(target, ('D', 'L')) / run(target, ('U', 'R'))` |
| `tests/backend/_padded_frame_oracles.py:145` | `run(target, 'RO')` | 145-146 | `run(target, ('D', 'L')) / run(target, ('U', 'R'))` |
| `tests/backend/_padded_frame_oracles.py:148` | `run(target, 'UL')` | 149-149 | `run(target, ('D', 'L'))` |
| `tests/backend/_padded_frame_oracles.py:151` | `run(target, 'ULRO')` | 152-152 | `run(target, ('D', 'U', 'L', 'R'))` |
| `tests/backend/_padded_frame_oracles.py:176` | `('U' leaf -> core, 'L' rightward, 'R' leftward, 'O' core -> leaf); 1.0 where the widths differ."""` | 177-177 | `('D' leaf -> core, 'L' rightward, 'R' leftward, 'U' core -> leaf); 1.0 where the widths differ."""` |
| `tests/backend/_padded_frame_oracles.py:178` | `for name, a, b in zip('ULRO', dense_a, dense_b):` | 179-179 | `for name, a, b in zip(('D', 'L', 'R', 'U'), dense_a, dense_b):` |
| `tests/backend/test_padded_frame.py:40` | `stop rule on prescribed movements (the stop tested only after a ``('UL', 'RO')`` pair, never on the truncating` | 40-43 | `stop rule on prescribed movements (the stop tested only after a ``(('D', 'L'), ('U', 'R'))`` pair, never on the / truncating pass nor across two pairs; an odd remainder running one final ('D', 'L') only when not stopped; the stack / maximum; the cap). Movements are a tuple of ``C`` arrays. ``use_jit=True`` without jax warns once and runs eagerly on / numpy (the optimizers' jax-absent policy). Eager = jit and T13 (a), the compile count, need jax with x64 and live in` |
| `tests/backend/test_padded_frame.py:41` | `pass nor across two pairs; an odd remainder running one final 'UL' only when not stopped; the stack maximum; the` | 40-43 | `stop rule on prescribed movements (the stop tested only after a ``(('D', 'L'), ('U', 'R'))`` pair, never on the / truncating pass nor across two pairs; an odd remainder running one final ('D', 'L') only when not stopped; the stack / maximum; the cap). Movements are a tuple of ``C`` arrays. ``use_jit=True`` without jax warns once and runs eagerly on / numpy (the optimizers' jax-absent policy). Eager = jit and T13 (a), the compile count, need jax with x64 and live in` |
| `tests/backend/test_padded_frame.py:513` | `def assert_matches_dense(self, start, variations, old, F, movement, stores, update='ULRO', groups=(), tol=1e-9):` | 513-514 | `def assert_matches_dense(self, start, variations, old, F, movement, stores, update=('D', 'U', 'L', 'R'), groups=(), / tol=1e-9):` |
| `tests/backend/test_padded_frame.py:547` | `def check(self, shape, old_ranks, work_ranks, out_ranks, options, update='ULRO'):` | 548-548 | `def check(self, shape, old_ranks, work_ranks, out_ranks, options, update=('D', 'U', 'L', 'R')):` |
| `tests/backend/test_padded_frame.py:554` | `*(out_ranks or (None, None)), sharing=sharing, update=update)` | 555 (unchanged) | |
| `tests/backend/test_padded_frame.py:574` | `for update in ('UL', 'RO', 'LRO', 'ULO', 'U', 'O'):` | 575-575 | `for update in (('D', 'L'), ('U', 'R'), ('U', 'L', 'R'), ('D', 'U', 'L'), ('D',), ('U',)):` |
| `tests/backend/test_padded_frame.py:575` | `with self.subTest(update=update):` | 576 (unchanged) | |
| `tests/backend/test_padded_frame.py:576` | `self.check(*args, update=update)` | 577 (unchanged) | |
| `tests/backend/test_padded_frame.py:723` | `self.step(update='UL')` | 724-724 | `self.step(update=('D', 'L'))` |
| `tests/backend/test_padded_frame.py:730` | `for missing, store in (('U', 'down'), ('O', 'up'), ('L', 'left'), ('R', 'right')):` | 731-732 | `for missing, store in (('D', 'down'), ('U', 'up'), ('L', 'left'), ('R', 'right')): / update = tuple(c for c in ('D', 'U', 'L', 'R') if c != missing)` |
| `tests/backend/test_padded_frame.py:731` | `update = ''.join(c for c in 'ULRO' if c != missing)` | 731-732 | `for missing, store in (('D', 'down'), ('U', 'up'), ('L', 'left'), ('R', 'right')): / update = tuple(c for c in ('D', 'U', 'L', 'R') if c != missing)` |
| `tests/backend/test_padded_frame.py:733` | `self.step(out=self.target, update=update)` | 734 (unchanged) | |
| `tests/backend/test_padded_frame.py:736` | `for update in ('', 'UU', 'X', 'ULROX', 'ulro'):` | 737-773 | `"""Not a non-empty collection of distinct store letters: the message names the letters. Every str (the 2026.4.0 / annotated type, refused whatever it holds) and a collection holding the 2026.4.0 letter 'O' add the 2026.4.0 / sentence.""" / for update in ((), ('D', 'D'), ('X',), ('D', 'U', 'L', 'R', 'X'), ('d', 'u', 'l', 'r')): / with self.subTest(update=update): / with self.assertRaisesRegex(ValueError, 'without repeats') as raised: / self.step(update=update) / self.assertNotIn('2026.4.0', str(raised.exception)) / for update in ('DL', 'ULRO', 'D', ('U', 'O'), ['L', 'O']): / with self.subTest(update=update): / with self.assertRaisesRegex(ValueError, "without repeats .*; that is the 2026.4.0 form, whose 'U' was " / r"the Tucker store \(now 'D'\)"): / self.step(update=update) / with self.assertRaises(ValueError) as raised: / self.step(update='ULRO') / self.assertEqual(str(raised.exception), / "update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats " / "(D down, U up, L left, R right), e.g. ('D', 'L'); got 'ULRO'; that is the 2026.4.0 form, " / "whose 'U' was the Tucker store (now 'D') and 'O' the TT-shaped store (now 'U'): swap each " / "letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), not ('U', 'L'), which now means up and " / "left") /  / def test_update_letters_in_any_order_with_the_new_meaning(self): / """Any collection of the letters, in any order, is one canonical set (``_STORE_LETTERS`` order), so the frames / agree bitwise. A 2026.4.0 tuple without 'O' runs in the new meaning: ('U', 'L') updates the up and the left / stores and transports the down store, whose completion keeps its span; ('D', 'L') moves it.""" / self.assertEqual(padded_frames._update_stores(('R', 'U', 'D')), ('D', 'U', 'R')) / F, movement = self.step(update=('U', 'L')) / for update in (('L', 'U'), ['U', 'L'], {'U', 'L'}, frozenset(('U', 'L'))): / with self.subTest(update=update): / G, movement_G = self.step(update=update) / self.assertTrue(all(np.array_equal(a, b) for f, g in zip(F, G) for a, b in zip(f, g))) / self.assertTrue(np.array_equal(movement, movement_G)) / F_down, _ = self.step(update=('D', 'L')) / off_span = lambda frame: [np.linalg.norm(a - a @ s.T @ s) for a, s in zip(frame[0], self.start[0])] / self.assertLessEqual(max(off_span(F)), 1e-12)          # the down store transported: the start's row spans / self.assertGreater(max(off_span(F_down)), 1e-3)        # the down store aligned: moved` |
| `tests/backend/test_padded_frame.py:737` | `with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'without repeats'):` | 737-773 | `"""Not a non-empty collection of distinct store letters: the message names the letters. Every str (the 2026.4.0 / annotated type, refused whatever it holds) and a collection holding the 2026.4.0 letter 'O' add the 2026.4.0 / sentence.""" / for update in ((), ('D', 'D'), ('X',), ('D', 'U', 'L', 'R', 'X'), ('d', 'u', 'l', 'r')): / with self.subTest(update=update): / with self.assertRaisesRegex(ValueError, 'without repeats') as raised: / self.step(update=update) / self.assertNotIn('2026.4.0', str(raised.exception)) / for update in ('DL', 'ULRO', 'D', ('U', 'O'), ['L', 'O']): / with self.subTest(update=update): / with self.assertRaisesRegex(ValueError, "without repeats .*; that is the 2026.4.0 form, whose 'U' was " / r"the Tucker store \(now 'D'\)"): / self.step(update=update) / with self.assertRaises(ValueError) as raised: / self.step(update='ULRO') / self.assertEqual(str(raised.exception), / "update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats " / "(D down, U up, L left, R right), e.g. ('D', 'L'); got 'ULRO'; that is the 2026.4.0 form, " / "whose 'U' was the Tucker store (now 'D') and 'O' the TT-shaped store (now 'U'): swap each " / "letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), not ('U', 'L'), which now means up and " / "left") /  / def test_update_letters_in_any_order_with_the_new_meaning(self): / """Any collection of the letters, in any order, is one canonical set (``_STORE_LETTERS`` order), so the frames / agree bitwise. A 2026.4.0 tuple without 'O' runs in the new meaning: ('U', 'L') updates the up and the left / stores and transports the down store, whose completion keeps its span; ('D', 'L') moves it.""" / self.assertEqual(padded_frames._update_stores(('R', 'U', 'D')), ('D', 'U', 'R')) / F, movement = self.step(update=('U', 'L')) / for update in (('L', 'U'), ['U', 'L'], {'U', 'L'}, frozenset(('U', 'L'))): / with self.subTest(update=update): / G, movement_G = self.step(update=update) / self.assertTrue(all(np.array_equal(a, b) for f, g in zip(F, G) for a, b in zip(f, g))) / self.assertTrue(np.array_equal(movement, movement_G)) / F_down, _ = self.step(update=('D', 'L')) / off_span = lambda frame: [np.linalg.norm(a - a @ s.T @ s) for a, s in zip(frame[0], self.start[0])] / self.assertLessEqual(max(off_span(F)), 1e-12)          # the down store transported: the start's row spans / self.assertGreater(max(off_span(F_down)), 1e-3)        # the down store aligned: moved` |
| `tests/backend/test_padded_frame.py:738` | `self.step(update=update)` | 737-773 | `"""Not a non-empty collection of distinct store letters: the message names the letters. Every str (the 2026.4.0 / annotated type, refused whatever it holds) and a collection holding the 2026.4.0 letter 'O' add the 2026.4.0 / sentence.""" / for update in ((), ('D', 'D'), ('X',), ('D', 'U', 'L', 'R', 'X'), ('d', 'u', 'l', 'r')): / with self.subTest(update=update): / with self.assertRaisesRegex(ValueError, 'without repeats') as raised: / self.step(update=update) / self.assertNotIn('2026.4.0', str(raised.exception)) / for update in ('DL', 'ULRO', 'D', ('U', 'O'), ['L', 'O']): / with self.subTest(update=update): / with self.assertRaisesRegex(ValueError, "without repeats .*; that is the 2026.4.0 form, whose 'U' was " / r"the Tucker store \(now 'D'\)"): / self.step(update=update) / with self.assertRaises(ValueError) as raised: / self.step(update='ULRO') / self.assertEqual(str(raised.exception), / "update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats " / "(D down, U up, L left, R right), e.g. ('D', 'L'); got 'ULRO'; that is the 2026.4.0 form, " / "whose 'U' was the Tucker store (now 'D') and 'O' the TT-shaped store (now 'U'): swap each " / "letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), not ('U', 'L'), which now means up and " / "left") /  / def test_update_letters_in_any_order_with_the_new_meaning(self): / """Any collection of the letters, in any order, is one canonical set (``_STORE_LETTERS`` order), so the frames / agree bitwise. A 2026.4.0 tuple without 'O' runs in the new meaning: ('U', 'L') updates the up and the left / stores and transports the down store, whose completion keeps its span; ('D', 'L') moves it.""" / self.assertEqual(padded_frames._update_stores(('R', 'U', 'D')), ('D', 'U', 'R')) / F, movement = self.step(update=('U', 'L')) / for update in (('L', 'U'), ['U', 'L'], {'U', 'L'}, frozenset(('U', 'L'))): / with self.subTest(update=update): / G, movement_G = self.step(update=update) / self.assertTrue(all(np.array_equal(a, b) for f, g in zip(F, G) for a, b in zip(f, g))) / self.assertTrue(np.array_equal(movement, movement_G)) / F_down, _ = self.step(update=('D', 'L')) / off_span = lambda frame: [np.linalg.norm(a - a @ s.T @ s) for a, s in zip(frame[0], self.start[0])] / self.assertLessEqual(max(off_span(F)), 1e-12)          # the down store transported: the start's row spans / self.assertGreater(max(off_span(F_down)), 1e-3)        # the down store aligned: moved` |
| `tests/backend/test_padded_frame.py:899` | `self.assertEqual(calls, [('ULRO', t), ('UL', t), ('RO', t), ('UL', t)])` | 934-934 | `self.assertEqual(calls, [(('D', 'U', 'L', 'R'), t), (('D', 'L'), t), (('U', 'R'), t), (('D', 'L'), t)])` |
| `tests/backend/test_padded_frame.py:903` | `"""Two oversampled passes, the truncating pass, then 'UL' and 'RO' in turn; the simultaneous schedule` | 938-939 | `"""Two oversampled passes, the truncating pass, then ('D', 'L') and ('U', 'R') in turn; the simultaneous / schedule updates every store on every pass."""` |
| `tests/backend/test_padded_frame.py:907` | `self.assertEqual(calls, [('ULRO', None), ('ULRO', None), ('ULRO', t), ('UL', t), ('RO', t), ('UL', t)])` | 942-943 | `self.assertEqual(calls, [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t), / (('D', 'L'), t), (('U', 'R'), t), (('D', 'L'), t)])` |
| `tests/backend/test_padded_frame.py:909` | `self.assertEqual(calls, [('ULRO', None), ('ULRO', None), ('ULRO', t), ('ULRO', t)])` | 945-946 | `self.assertEqual(calls, [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t), / (('D', 'U', 'L', 'R'), t)])` |
| `tests/backend/test_padded_frame.py:912` | `"""max_passes = 2 with no oversampled phase: the 'ULRO' pass, then one slot, no pair: the final 'UL'."""` | 949-950 | `"""max_passes = 2 with no oversampled phase: the all-store pass, then one slot, no pair: the final / ('D', 'L')."""` |
| `tests/backend/test_padded_frame.py:914` | `self.assertEqual(calls, [('ULRO', self.target[0]), ('UL', self.target[0])])` | 952-952 | `self.assertEqual(calls, [(('D', 'U', 'L', 'R'), self.target[0]), (('D', 'L'), self.target[0])])` |
| `tests/backend/test_padded_frame.py:919` | `self.assertEqual(calls, [('ULRO', None), ('ULRO', None), ('ULRO', self.target[0])])` | 957-958 | `self.assertEqual(calls, [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), / (('D', 'U', 'L', 'R'), self.target[0])])` |
| `tests/backend/test_padded_frame.py:923` | `"""max_passes = 1: one truncating pass from the working start, or one 'ULRO' pass at the target."""` | 962-962 | `"""max_passes = 1: one truncating pass from the working start, or one all-store pass at the target."""` |
| `tests/backend/test_padded_frame.py:927` | `self.assertEqual(calls, [('ULRO', self.target[0])])` | 966-966 | `self.assertEqual(calls, [(('D', 'U', 'L', 'R'), self.target[0])])` |
| `tests/backend/test_padded_frame.py:980` | `after the truncating one come in ('UL', 'RO') pairs (3-4, 5-6, 7-8) and the stop is tested after each pair,` | 1019-1022 | `after the truncating one come in (('D', 'L'), ('U', 'R')) pairs (3-4, 5-6, 7-8) and the stop is tested after / each pair, both movements below tol; never on the truncating pass, never across two pairs; the odd slot left by / the cap (pass 9) runs one final ('D', 'L') only when not stopped. Simultaneous: the first pass from the / truncating one on below tol. Oversampled passes never stop. The stack maximum decides; the cap stops."""` |
| `tests/backend/test_padded_frame.py:982` | `(pass 9) runs one final 'UL' only when not stopped. Simultaneous: the first pass from the truncating one on` | 1019-1022 | `after the truncating one come in (('D', 'L'), ('U', 'R')) pairs (3-4, 5-6, 7-8) and the stop is tested after / each pair, both movements below tol; never on the truncating pass, never across two pairs; the odd slot left by / the cap (pass 9) runs one final ('D', 'L') only when not stopped. Simultaneous: the first pass from the / truncating one on below tol. Oversampled passes never stop. The stack maximum decides; the cap stops."""` |
| `tests/backend/test_padded_frame.py:990` | `self.assertEqual(self.stop_after([hi] * 9 + [lo]), 10)                          # the cap: 3 pairs, a final 'UL'` | 1029-1032 | `self.assertEqual(self.stop_after([hi] * 9 + [lo]), 10)                          # the cap: 3 pairs, a final ('D', 'L') / self.assertEqual(self.stop_after([hi] * 9, max_passes=9), 9)                    # an even cap: no final ('D', 'L') / self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final ('D', 'L') / self.assertEqual(self.stop_after([hi] * 6, max_passes=6), 6)                    # not stopped: the final ('D', 'L')` |
| `tests/backend/test_padded_frame.py:991` | `self.assertEqual(self.stop_after([hi] * 9, max_passes=9), 9)                    # an even cap: no final 'UL'` | 1029-1032 | `self.assertEqual(self.stop_after([hi] * 9 + [lo]), 10)                          # the cap: 3 pairs, a final ('D', 'L') / self.assertEqual(self.stop_after([hi] * 9, max_passes=9), 9)                    # an even cap: no final ('D', 'L') / self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final ('D', 'L') / self.assertEqual(self.stop_after([hi] * 6, max_passes=6), 6)                    # not stopped: the final ('D', 'L')` |
| `tests/backend/test_padded_frame.py:992` | `self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final 'UL'` | 1029-1032 | `self.assertEqual(self.stop_after([hi] * 9 + [lo]), 10)                          # the cap: 3 pairs, a final ('D', 'L') / self.assertEqual(self.stop_after([hi] * 9, max_passes=9), 9)                    # an even cap: no final ('D', 'L') / self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final ('D', 'L') / self.assertEqual(self.stop_after([hi] * 6, max_passes=6), 6)                    # not stopped: the final ('D', 'L')` |
| `tests/backend/test_padded_frame.py:993` | `self.assertEqual(self.stop_after([hi] * 6, max_passes=6), 6)                    # not stopped: the final 'UL'` | 1029-1032 | `self.assertEqual(self.stop_after([hi] * 9 + [lo]), 10)                          # the cap: 3 pairs, a final ('D', 'L') / self.assertEqual(self.stop_after([hi] * 9, max_passes=9), 9)                    # an even cap: no final ('D', 'L') / self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final ('D', 'L') / self.assertEqual(self.stop_after([hi] * 6, max_passes=6), 6)                    # not stopped: the final ('D', 'L')` |
| `tests/backend/test_padded_frame.py:1006` | `"""Recorded: a stop after the first pair leaves (oversampled x2, truncating, 'UL', 'RO') and no more."""` | 1045-1046 | `"""Recorded: a stop after the first pair leaves (oversampled x2, truncating, ('D', 'L'), ('U', 'R')) and no / more."""` |
| `tests/backend/test_padded_frame.py:1015` | `self.assertEqual(updates, ['ULRO', 'ULRO', 'ULRO', 'UL', 'RO'])` | 1055-1056 | `self.assertEqual(updates, [('D', 'U', 'L', 'R'), ('D', 'U', 'L', 'R'), ('D', 'U', 'L', 'R'), ('D', 'L'), / ('U', 'R')])` |
| `tests/backend/test_padded_start.py:480` | `dense_tied_project(start_dense, G, groups, x0.to_dense()), stores, 'ULRO', groups)` | 480-481 | `dense_tied_project(start_dense, G, groups, x0.to_dense()), stores, / ('D', 'U', 'L', 'R'), groups)` |
| `tests/backend/test_uniform_padded_frame.py:360` | `def check(self, shape, old_ranks, work_ranks, out_ranks, options, update='ULRO'):` | 360-360 | `def check(self, shape, old_ranks, work_ranks, out_ranks, options, update=('D', 'U', 'L', 'R')):` |
| `tests/backend/test_uniform_padded_frame.py:366` | `sharing=sharing, update=update)` | 366 (unchanged) | |
| `tests/backend/test_uniform_padded_frame.py:369` | `update=update)` | 369 (unchanged) | |
| `tests/backend/test_uniform_padded_frame.py:388` | `"""Each store family transported, and the two colours ('UL', 'RO')."""` | 388-388 | `"""Each store family transported, and the two colours (('D', 'L'), ('U', 'R'))."""` |
| `tests/backend/test_uniform_padded_frame.py:390` | `for update in ('UL', 'RO', 'LRO', 'ULO', 'U', 'O'):` | 390-390 | `for update in (('D', 'L'), ('U', 'R'), ('U', 'L', 'R'), ('D', 'U', 'L'), ('D',), ('U',)):` |
| `tests/backend/test_uniform_padded_frame.py:391` | `with self.subTest(update=update):` | 391 (unchanged) | |
| `tests/backend/test_uniform_padded_frame.py:392` | `self.check(*args, update=update)` | 392 (unchanged) | |
| `tests/backend/test_uniform_padded_frame.py:514` | `self.call(update='UL', out=(None, None))` | 514-514 | `self.call(update=('D', 'L'), out=(None, None))` |
| `tests/backend/test_uniform_padded_frame.py:517` | `for update in ('', 'UU', 'ULX'):` | 517-519 | `"""The ragged step's validation (``padded_frames._update_stores``): bad letters name the letters; every str / and a collection holding the 2026.4.0 letter 'O' add the 2026.4.0 sentence.""" / for update in ((), ('D', 'D'), ('D', 'L', 'X')):` |
| `tests/backend/test_uniform_padded_frame.py:518` | `with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'update must be'):` | 520 (unchanged) | |
| `tests/backend/test_uniform_padded_frame.py:519` | `self.call(update=update)` | 521 (unchanged) | |
| `tests/backend/test_uniform_padded_frame.py:547` | `self.call(update='RO')` | 567-567 | `self.call(update=('U', 'R'))` |
| `tests/backend/test_uniform_padded_frame.py:771` | `[('ULRO', None), ('ULRO', None), ('ULRO', t), ('UL', None), ('RO', None), ('UL', None)])` | 791-792 | `[(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t), / (('D', 'L'), None), (('U', 'R'), None), (('D', 'L'), None)])` |
| `tests/backend/test_uniform_padded_frame.py:773` | `[('ULRO', None), ('ULRO', None), ('ULRO', t), ('ULRO', None)])` | 794-797 | `[(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t), / (('D', 'U', 'L', 'R'), None)]) / self.assertEqual(self.run_recorded(oversampling=0, max_passes=2, tol=0.0), / [(('D', 'U', 'L', 'R'), t), (('D', 'L'), None)])` |
| `tests/backend/test_uniform_padded_frame.py:774` | `self.assertEqual(self.run_recorded(oversampling=0, max_passes=2, tol=0.0), [('ULRO', t), ('UL', None)])` | 794-797 | `[(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t), / (('D', 'U', 'L', 'R'), None)]) / self.assertEqual(self.run_recorded(oversampling=0, max_passes=2, tol=0.0), / [(('D', 'U', 'L', 'R'), t), (('D', 'L'), None)])` |
| `tests/backend/test_uniform_padded_frame.py:776` | `[('ULRO', None), ('ULRO', None), ('ULRO', t)])` | 799-799 | `[(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t)])` |
| `tests/backend/test_uniform_padded_frame.py:800` | `self.assertEqual(self.stop_after([hi] * 10), 10)                                # 3 pairs and a final 'UL'` | 823-823 | `self.assertEqual(self.stop_after([hi] * 10), 10)                                # 3 pairs and a final ('D', 'L')` |
| `tests/backend/test_uniform_padded_frame.py:802` | `self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final 'UL'` | 825-825 | `self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final ('D', 'L')` |
| `tests/test_aligned_padded_frame_frontend.py:125` | `UF1, umv = uv.align_padded(split, update='UL')` | 125-126 | `UF1, umv = uv.align_padded(split, update=('D', 'L')) / UFb, umvb = uniform_padded_frames.utv_align_padded_frame(UF.data, uv.variations.data, split, update=('D', 'L'))` |
| `tests/test_aligned_padded_frame_frontend.py:126` | `UFb, umvb = uniform_padded_frames.utv_align_padded_frame(UF.data, uv.variations.data, split, update='UL')` | 125-126 | `UF1, umv = uv.align_padded(split, update=('D', 'L')) / UFb, umvb = uniform_padded_frames.utv_align_padded_frame(UF.data, uv.variations.data, split, update=('D', 'L'))` |
| `tests/test_dispatch.py:2008` | `two-colour half (R and O transported), the zero tensor's empty frame, and a stack C = (2,)."""` | 2008-2008 | `two-colour half (right and up transported), the zero tensor's empty frame, and a stack C = (2,)."""` |
| `tests/test_dispatch.py:2019` | `((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, 'ULRO'),` | 2019-2021 | `((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, ('D', 'U', 'L', 'R')), / 'shared, DL': (bvf.t3_orthogonal_representations(x_tied)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)), / ((3, 3, 3, 3), (1, 3, 4, 3, 1)), (None, None), (0, 1, 0, 1), ('D', 'L')),` |
| `tests/test_dispatch.py:2020` | `'shared, UL': (bvf.t3_orthogonal_representations(x_tied)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),` | 2019-2021 | `((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, ('D', 'U', 'L', 'R')), / 'shared, DL': (bvf.t3_orthogonal_representations(x_tied)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)), / ((3, 3, 3, 3), (1, 3, 4, 3, 1)), (None, None), (0, 1, 0, 1), ('D', 'L')),` |
| `tests/test_dispatch.py:2021` | `((3, 3, 3, 3), (1, 3, 4, 3, 1)), (None, None), (0, 1, 0, 1), 'UL'),` | 2019-2021 | `((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, ('D', 'U', 'L', 'R')), / 'shared, DL': (bvf.t3_orthogonal_representations(x_tied)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)), / ((3, 3, 3, 3), (1, 3, 4, 3, 1)), (None, None), (0, 1, 0, 1), ('D', 'L')),` |
| `tests/test_dispatch.py:2023` | `((2, 3, 3, 2), (1, 2, 3, 2, 1)), (None, None), None, 'ULRO'),` | 2023-2023 | `((2, 3, 3, 2), (1, 2, 3, 2, 1)), (None, None), None, ('D', 'U', 'L', 'R')),` |
| `tests/test_dispatch.py:2025` | `((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, 'ULRO'),` | 2025-2025 | `((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, ('D', 'U', 'L', 'R')),` |
| `tests/test_dispatch.py:2041` | `tt_ranks=out_ranks[1], sharing=sharing, update=update)` | 2041 (unchanged) | |
| `tests/test_dispatch.py:2058` | `every leaf -- two oversampled passes, the truncating pass and one 'UL' / 'RO' pair (tol 0, max_passes 5), at` | 2058-2061 | `every leaf -- two oversampled passes, the truncating pass and one ('D', 'L') / ('U', 'R') pair (tol 0, / max_passes 5), at d = 2, with the dense projection of a hint of the matching array type -- eagerly, and with / use_jit=True (the memoized jitted step; numpy in is converted to jax, the numpy hint handed on as is, jax out). / The movements equal numpy's (x64 on in this module; they do not depend on the gauge)."""` |
| `tests/test_dispatch.py:2226` | `captured.update(dynamic=dynamic, skeleton=skeleton, split=split_, update=update)` | 2226 (unchanged) | |
| `tests/test_dispatch.py:2345` | `# capped at 12 passes: oversampled and truncating passes, pairs, the final 'UL' (12 = 3 + 4 pairs + 1 and` | 2345-2345 | `# capped at 12 passes: oversampled and truncating passes, pairs, the final ('D', 'L') (12 = 3 + 4 pairs + 1 and` |
| `tests/test_dispatch.py:2395` | `self.assertEqual((lengths['two-colour, p=5'], lengths['two-colour, p=0']), (12, 12))   # the final 'UL'` | 2395-2395 | `self.assertEqual((lengths['two-colour, p=5'], lengths['two-colour, p=0']), (12, 12))   # the final ('D', 'L')` |
| `tests/test_dispatch.py:2416` | `self.assertEqual(seen, [(u, on_jax) for u in ('ULRO', 'ULRO', 'ULRO', 'UL')])` | 2416-2416 | `self.assertEqual(seen, [(u, on_jax) for u in (('D', 'U', 'L', 'R'),) * 3 + (('D', 'L'),)])` |
| `tests/test_dispatch.py:2424` | `oversampled 'ULRO', the truncating 'ULRO', 'UL', 'RO' -- and a second run at the same structure, other values,` | 2424-2427 | `oversampled all-store, the truncating all-store, ('D', 'L'), ('U', 'R') -- and a second run at the same / structure, other values, compiles none. A structure no other test uses, so the count is this run's. The wrapper / is memoized (one object), the house ``_jitted`` contract; jax 0.10 would share the programs across wrappers / anyway."""` |
| `tests/test_dispatch.py:2456` | `"""T12's cases, plus the odd remainder's final 'UL' (tol 0, cap 10: 2 + 1 + 3 pairs + 1): the numpy (eager)` | 2457-2459 | `"""T12's cases, plus the odd remainder's final ('D', 'L') (tol 0, cap 10: 2 + 1 + 3 pairs + 1): the numpy / (eager) loop and the jitted loop run the same passes (NaN in the same slots), frames to 1e-12, movements to / 1e-10; early stops included (the defaults stop at 9)."""` |
| `tests/test_dispatch.py:2465` | `cases['the final UL'] = cases['defaults'][:4] + ({'tol': 0.0},)` | 2466-2466 | `cases['the final DL'] = cases['defaults'][:4] + ({'tol': 0.0},)` |
| `tests/test_dispatch.py:2480` | `self.assertEqual((lengths['defaults'], lengths['the final UL']), (9, 10))` | 2481-2481 | `self.assertEqual((lengths['defaults'], lengths['the final DL']), (9, 10))` |

## Appendix B. Every changed hunk of the letter-API files (`git diff -U0 HEAD`; old lines → new lines)

The §5.2 test files are additions only (Appendix D).

#### `t3toolbox/backend/padded_frames.py` (13 hunks)

- **:13-14 -> :13-14**
  ```
  - one fixed tensor, a truncating pass to the target ranks, then two-colour passes in ``('UL', 'RO')`` pairs. The
  - ``ALIGN_*`` constants are the loop's defaults, named once here. Everything runs eagerly unless the loop is asked for
  + one fixed tensor, a truncating pass to the target ranks, then two-colour passes in ``(('D', 'L'), ('U', 'R'))`` pairs.
  + The ``ALIGN_*`` constants are the loop's defaults, named once here. Everything runs eagerly unless the loop is asked for
  ```
- **:after 51 -> :52-69**
  ```
  + _STORE_LETTERS = ('D', 'U', 'L', 'R')                       # a frame's stores in physical order (T3Frame.data)
  + _COLOURS = (('D', 'L'), ('U', 'R'))                         # the two-colour passes (alternate=True), in pass order
  + 
  + 
  + def _update_stores(
  +         update:  typ.Collection[str],   # single store letters, any order, e.g. ('D', 'L'): D down, U up, L left, R right
  + ) -> typ.Tuple[str, ...]:               # the same letters in _STORE_LETTERS order: one jit static per set
  +     """``update`` validated and canonicalized; ``ValueError`` otherwise, naming the letters."""
  +     if (isinstance(update, str) or not update or len(set(update)) != len(update)
  +             or not set(update) <= set(_STORE_LETTERS)):
  +         message = ("update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats "
  +                    "(D down, U up, L left, R right), e.g. ('D', 'L'); got %r" % (update,))
  +         if isinstance(update, str) or 'O' in update:      # every str is a 2026.4.0 call
  +             message += ("; that is the 2026.4.0 form, whose 'U' was the Tucker store (now 'D') and 'O' the "
  +                         "TT-shaped store (now 'U'): swap each letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), "
  +                         "not ('U', 'L'), which now means up and left")
  +         raise ValueError(message)
  +     return tuple(c for c in _STORE_LETTERS if c in update)
  ```
- **:554 -> :572**
  ```
  -         update:            str = 'ULRO',                             # stores updated: 'U' up, 'L' left, 'R' right, 'O' down; the rest transported
  +         update:            typ.Collection[str] = ('D', 'U', 'L', 'R'),   # store letters, any order: 'D' down, 'U' up, 'L' left, 'R' right; the rest transported
  ```
- **:560-562 -> :578**
  ```
  -     if not update or len(set(update)) != len(update) or not set(update) <= set('ULRO'):
  -         raise ValueError("update must be a non-empty set of the stores 'U', 'L', 'R', 'O' without repeats; got %r"
  -                          % (update,))
  +     update = _update_stores(update)
  ```
- **:581 -> :597**
  ```
  -     for store, letter, old, new, own in zip(_STORE_NAMES, 'UOLR', (nD0, nU0, rL0, rR0), out, (nD, nU, rL, rR)):
  +     for store, letter, old, new, own in zip(_STORE_NAMES, _STORE_LETTERS, (nD0, nU0, rL0, rR0), out, (nD, nU, rL, rR)):
  ```
- **:624 -> :640**
  ```
  -         tuple(nD1[g[0]] - nD0[g[0]] for g in groups), ('U' in update,) * len(groups)))
  +         tuple(nD1[g[0]] - nD0[g[0]] for g in groups), ('D' in update,) * len(groups)))
  ```
- **:655 -> :671**
  ```
  -         tuple(nU1[i] - nU0[i] for i in range(d)), ('O' in update,) * d))
  +         tuple(nU1[i] - nU0[i] for i in range(d)), ('U' in update,) * d))
  ```
- **:687 -> :703**
  ```
  -         update:     str,                                      # 'ULRO' | 'UL' | 'RO'
  +         update:     typ.Tuple[str, ...],                      # all four stores | ('D', 'L') | ('U', 'R')
  ```
- **:750 -> :766**
  ```
  -         frame, movement = _aligning_pass(step, frame, hint, project, old, out, sharing, 'ULRO')
  +         frame, movement = _aligning_pass(step, frame, hint, project, old, out, sharing, _STORE_LETTERS)
  ```
- **:752-755 -> :768-771**
  ```
  -     if alternate:                                               # ('UL', 'RO') pairs while a whole pair fits; the stop
  -         stop = False                                            #   is tested after each pair (both below tol)
  -         while not stop and len(movements) + 2 <= max_passes:
  -             for update in ('UL', 'RO'):
  +     if alternate:                                               # (('D', 'L'), ('U', 'R')) pairs while a whole pair
  +         stop = False                                            #   fits; the stop is tested after each pair (both
  +         while not stop and len(movements) + 2 <= max_passes:   #   below tol)
  +             for update in _COLOURS:
  ```
- **:759-760 -> :775-776**
  ```
  -         if not stop and len(movements) < max_passes:           # an odd remainder: one final 'UL' pass
  -             frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, 'UL')
  +         if not stop and len(movements) < max_passes:           # an odd remainder: one final ('D', 'L') pass
  +             frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, _COLOURS[0])
  ```
- **:762 -> :778**
  ```
  -     else:                                                       # 'ULRO' passes, the stop tested after each (the
  +     else:                                                       # all-store passes, the stop tested after each (the
  ```
- **:765 -> :781**
  ```
  -             frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, 'ULRO')
  +             frame, movement = _aligning_pass(step, frame, hint, project, old, target, sharing, _STORE_LETTERS)
  ```

#### `t3toolbox/backend/uniform_padded_frames.py` (17 hunks)

- **:36 -> :36**
  ```
  -                                              _right_message_step, _jitted)
  +                                              _right_message_step, _jitted, _STORE_LETTERS, _COLOURS, _update_stores)
  ```
- **:522 -> :522**
  ```
  -         update:    str,                       # static: the stores updated; the rest transported
  +         update:    typ.Tuple[str, ...],       # static: the stores updated (canonical letters); the rest transported
  ```
- **:566 -> :566**
  ```
  -     Y = _u_content_d(g_old, g_new, PhiD[..., Na:], XiD) if 'U' in update else g_old
  +     Y = _u_content_d(g_old, g_new, PhiD[..., Na:], XiD) if 'D' in update else g_old
  ```
- **:601 -> :601**
  ```
  -     Y = _u_content_d(g_old, g_new, PhiU[..., Na:], XiU) if 'O' in update else g_old
  +     Y = _u_content_d(g_old, g_new, PhiU[..., Na:], XiU) if 'U' in update else g_old
  ```
- **:621 -> :621**
  ```
  -         update:           str = 'ULRO',                                      # static; as tv_align_padded_frame
  +         update:           typ.Collection[str] = ('D', 'U', 'L', 'R'),        # static; as tv_align_padded_frame
  ```
- **:627-629 -> :627**
  ```
  -     if not update or len(set(update)) != len(update) or not set(update) <= set('ULRO'):
  -         raise ValueError("update must be a non-empty set of the stores 'U', 'L', 'R', 'O' without repeats; got %r"
  -                          % (update,))
  +     update = _update_stores(update)
  ```
- **:669 -> :667**
  ```
  -     for store, letter, o, s1, s in zip(('down', 'up', 'left', 'right'), 'UOLR', old, out, own):
  +     for store, letter, o, s1, s in zip(('down', 'up', 'left', 'right'), _STORE_LETTERS, old, out, own):
  ```
- **:776 -> :774**
  ```
  -         statics:  typ.Tuple,   # as _aligning_pass, at the working structure, 'ULRO'
  +         statics:  typ.Tuple,   # as _aligning_pass, at the working structure, all four stores
  ```
- **:788 -> :786**
  ```
  - ) -> NDArray:                # 0-d bool: run another ('UL', 'RO') pair
  + ) -> NDArray:                # 0-d bool: run another (('D', 'L'), ('U', 'R')) pair
  ```
- **:795 -> :793**
  ```
  -         statics:  typ.Tuple[typ.Tuple, typ.Tuple],   # the 'UL' and the 'RO' passes' statics, at the target structure
  +         statics:  typ.Tuple[typ.Tuple, typ.Tuple],   # the ('D', 'L') and ('U', 'R') statics, at the target structure
  ```
- **:798 -> :796-797**
  ```
  -     """One ('UL', 'RO') pair; the stop: both movements below tol (the max over the stack; a NaN never stops)."""
  +     """One (('D', 'L'), ('U', 'R')) pair; the stop: both movements below tol (the max over the stack; a NaN never
  +     stops)."""
  ```
- **:808 -> :807**
  ```
  -         statics:  typ.Tuple,   # the 'UL' pass's statics
  +         statics:  typ.Tuple,   # the ('D', 'L') pass's statics
  ```
- **:811 -> :810**
  ```
  -     """The odd remainder's final 'UL' pass."""
  +     """The odd remainder's final ('D', 'L') pass."""
  ```
- **:833 -> :832**
  ```
  -         statics:  typ.Tuple,   # the 'ULRO' pass's statics, at the target structure
  +         statics:  typ.Tuple,   # the all-store pass's statics, at the target structure
  ```
- **:894 -> :893**
  ```
  -     statics = (project, hint, shape, masks, split, (None, None), sharing, 'ULRO')
  +     statics = (project, hint, shape, masks, split, (None, None), sharing, _STORE_LETTERS)
  ```
- **:901-902 -> :900-901**
  ```
  -         state = xwhile(_pairs_cond, ft.partial(_pairs_body, (at('UL'), at('RO'))), state, use_jit)
  -         state = _cond(_single_cond(state), ft.partial(_final_pass, at('UL')), _no_pass, state, use_jit)
  +         state = xwhile(_pairs_cond, ft.partial(_pairs_body, (at(_COLOURS[0]), at(_COLOURS[1]))), state, use_jit)
  +         state = _cond(_single_cond(state), ft.partial(_final_pass, at(_COLOURS[0])), _no_pass, state, use_jit)
  ```
- **:905 -> :904**
  ```
  -         state = xwhile(_single_cond, ft.partial(_single_body, at('ULRO')), state, use_jit)
  +         state = xwhile(_single_cond, ft.partial(_single_body, at(_STORE_LETTERS)), state, use_jit)
  ```

#### `t3toolbox/manifold.py` (2 hunks)

- **:175 -> :175**
  ```
  -             update:            str = 'ULRO',                            # the stores updated ('U', 'L', 'R', 'O'); the rest transported
  +             update:            typ.Collection[str] = ('D', 'U', 'L', 'R'),   # store letters, any order: 'D' down, 'U' up, 'L' left, 'R' right; the rest transported
  ```
- **:186 -> :186-187**
  ```
  -         the others (``'UL'`` / ``'RO'`` are the two colours of the loop); output ranks below the frame's truncate.
  +         the others (``('D', 'L')`` / ``('U', 'R')`` are the two colours of the loop); output ranks below the frame's
  +         truncate.
  ```

#### `t3toolbox/uniform_manifold.py` (1 hunks)

- **:152 -> :152**
  ```
  -             update:        str = 'ULRO',                                              # as T3Tangent.align_padded
  +             update:        typ.Collection[str] = ('D', 'U', 'L', 'R'),                # as T3Tangent.align_padded
  ```

#### `t3toolbox/frame_variations_format.py` (3 hunks)

- **:1765-1766 -> :1765-1766**
  ```
  -     - then two-colour pairs: a ``'UL'`` pass (up and left stores) and an ``'RO'`` pass (right and down), the other
  -       colour transported. A simultaneous update can cycle with period 2, which the pairs avoid.
  +     - then two-colour pairs: a ``('D', 'L')`` pass (the down and left stores) and a ``('U', 'R')`` pass (the up and
  +       right stores), the other colour transported. A simultaneous update can cycle with period 2, which the pairs avoid.
  ```
- **:1805 -> :1805**
  ```
  -     - ``alternate``: the two-colour switch. ``False`` runs simultaneous ``'ULRO'`` passes, stopping on one movement.
  +     - ``alternate``: the two-colour switch. ``False`` runs simultaneous all-store passes, stopping on one movement.
  ```
- **:1850 -> :1850**
  ```
  -     >>> [bool(float(tangent.align_padded(*old, update=store)[1]) < 1e-8) for store in 'ULRO']
  +     >>> [bool(float(tangent.align_padded(*old, update=(store,))[1]) < 1e-8) for store in ('D', 'U', 'L', 'R')]
  ```

#### `t3toolbox/uniform_frame_variations_format.py` (1 hunks)

- **:1611 -> :1611**
  ```
  -     ``('UL', 'RO')`` pairs).
  +     ``(('D', 'L'), ('U', 'R'))`` pairs).
  ```

#### `tests/backend/_padded_frame_oracles.py` (10 hunks)

- **:65 -> :65**
  ```
  - def dense_pass(old_dense, cur_dense, Z, stores, update='ULRO', groups=()):
  + def dense_pass(old_dense, cur_dense, Z, stores, update=('D', 'U', 'L', 'R'), groups=()):
  ```
- **:93 -> :93**
  ```
  -         W = choose(Q, np.concatenate(blocks, axis=1), Dc[i0][:, s:], nD[i0] - s, 'U' in update)
  +         W = choose(Q, np.concatenate(blocks, axis=1), Dc[i0][:, s:], nD[i0] - s, 'D' in update)
  ```
- **:119 -> :119**
  ```
  -         U.append(np.concatenate([Uo[i], choose(Q, content, Uc[i][:, r0:], nU[i] - r0, 'O' in update)], 1))
  +         U.append(np.concatenate([Uo[i], choose(Q, content, Uc[i][:, r0:], nU[i] - r0, 'U' in update)], 1))
  ```
- **:126-129 -> :126-130**
  ```
  -     (``alternate``) ('UL', 'RO') pairs while a whole pair fits under ``max_passes``, stopping after a pair whose two
  -     movements are below ``tol``, and one final 'UL' pass in a remaining slot unless stopped; or (not ``alternate``)
  -     'ULRO' passes, stopping after the first pass from the truncating one on whose movement is below ``tol``. Every
  -     pass is against ``dense_project(D, S)``. Returns the dense frames and the movement after every pass."""
  +     (``alternate``) (('D', 'L'), ('U', 'R')) pairs while a whole pair fits under ``max_passes``, stopping after a
  +     pair whose two movements are below ``tol``, and one final ('D', 'L') pass in a remaining slot unless stopped; or
  +     (not ``alternate``) all-store passes, stopping after the first pass from the truncating one on whose movement is
  +     below ``tol``. Every pass is against ``dense_project(D, S)``. Returns the dense frames and the movement after
  +     every pass."""
  ```
- **:140 -> :141**
  ```
  -         run(working if it < n_over else target, 'ULRO')
  +         run(working if it < n_over else target, ('D', 'U', 'L', 'R'))
  ```
- **:144-145 -> :145-146**
  ```
  -             run(target, 'UL')
  -             run(target, 'RO')
  +             run(target, ('D', 'L'))
  +             run(target, ('U', 'R'))
  ```
- **:148 -> :149**
  ```
  -             run(target, 'UL')
  +             run(target, ('D', 'L'))
  ```
- **:151 -> :152**
  ```
  -             run(target, 'ULRO')
  +             run(target, ('D', 'U', 'L', 'R'))
  ```
- **:176 -> :177**
  ```
  -     ('U' leaf -> core, 'L' rightward, 'R' leftward, 'O' core -> leaf); 1.0 where the widths differ."""
  +     ('D' leaf -> core, 'L' rightward, 'R' leftward, 'U' core -> leaf); 1.0 where the widths differ."""
  ```
- **:178 -> :179**
  ```
  -     for name, a, b in zip('ULRO', dense_a, dense_b):
  +     for name, a, b in zip(('D', 'L', 'R', 'U'), dense_a, dense_b):
  ```

#### `tests/backend/test_padded_frame.py` (20 hunks)

- **:40-43 -> :40-43**
  ```
  -   stop rule on prescribed movements (the stop tested only after a ``('UL', 'RO')`` pair, never on the truncating
  -   pass nor across two pairs; an odd remainder running one final 'UL' only when not stopped; the stack maximum; the
  -   cap). Movements are a tuple of ``C`` arrays. ``use_jit=True`` without jax warns once and runs eagerly on numpy
  -   (the optimizers' jax-absent policy). Eager = jit and T13 (a), the compile count, need jax with x64 and live in
  +   stop rule on prescribed movements (the stop tested only after a ``(('D', 'L'), ('U', 'R'))`` pair, never on the
  +   truncating pass nor across two pairs; an odd remainder running one final ('D', 'L') only when not stopped; the stack
  +   maximum; the cap). Movements are a tuple of ``C`` arrays. ``use_jit=True`` without jax warns once and runs eagerly on
  +   numpy (the optimizers' jax-absent policy). Eager = jit and T13 (a), the compile count, need jax with x64 and live in
  ```
- **:513 -> :513-514**
  ```
  -     def assert_matches_dense(self, start, variations, old, F, movement, stores, update='ULRO', groups=(), tol=1e-9):
  +     def assert_matches_dense(self, start, variations, old, F, movement, stores, update=('D', 'U', 'L', 'R'), groups=(),
  +                              tol=1e-9):
  ```
- **:547 -> :548**
  ```
  -     def check(self, shape, old_ranks, work_ranks, out_ranks, options, update='ULRO'):
  +     def check(self, shape, old_ranks, work_ranks, out_ranks, options, update=('D', 'U', 'L', 'R')):
  ```
- **:574 -> :575**
  ```
  -         for update in ('UL', 'RO', 'LRO', 'ULO', 'U', 'O'):
  +         for update in (('D', 'L'), ('U', 'R'), ('U', 'L', 'R'), ('D', 'U', 'L'), ('D',), ('U',)):
  ```
- **:723 -> :724**
  ```
  -         self.step(update='UL')
  +         self.step(update=('D', 'L'))
  ```
- **:730-731 -> :731-732**
  ```
  -         for missing, store in (('U', 'down'), ('O', 'up'), ('L', 'left'), ('R', 'right')):
  -             update = ''.join(c for c in 'ULRO' if c != missing)
  +         for missing, store in (('D', 'down'), ('U', 'up'), ('L', 'left'), ('R', 'right')):
  +             update = tuple(c for c in ('D', 'U', 'L', 'R') if c != missing)
  ```
- **:736-738 -> :737-773**
  ```
  -         for update in ('', 'UU', 'X', 'ULROX', 'ulro'):
  -             with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'without repeats'):
  -                 self.step(update=update)
  +         """Not a non-empty collection of distinct store letters: the message names the letters. Every str (the 2026.4.0
  +         annotated type, refused whatever it holds) and a collection holding the 2026.4.0 letter 'O' add the 2026.4.0
  +         sentence."""
  +         for update in ((), ('D', 'D'), ('X',), ('D', 'U', 'L', 'R', 'X'), ('d', 'u', 'l', 'r')):
  +             with self.subTest(update=update):
  +                 with self.assertRaisesRegex(ValueError, 'without repeats') as raised:
  +                     self.step(update=update)
  +                 self.assertNotIn('2026.4.0', str(raised.exception))
  +         for update in ('DL', 'ULRO', 'D', ('U', 'O'), ['L', 'O']):
  +             with self.subTest(update=update):
  +                 with self.assertRaisesRegex(ValueError, "without repeats .*; that is the 2026.4.0 form, whose 'U' was "
  +                                                         r"the Tucker store \(now 'D'\)"):
  +                     self.step(update=update)
  +         with self.assertRaises(ValueError) as raised:
  +             self.step(update='ULRO')
  +         self.assertEqual(str(raised.exception),
  +                          "update must be a non-empty tuple of the store letters 'D', 'U', 'L', 'R' without repeats "
  +                          "(D down, U up, L left, R right), e.g. ('D', 'L'); got 'ULRO'; that is the 2026.4.0 form, "
  +                          "whose 'U' was the Tucker store (now 'D') and 'O' the TT-shaped store (now 'U'): swap each "
  +                          "letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), not ('U', 'L'), which now means up and "
  +                          "left")
  + 
  +     def test_update_letters_in_any_order_with_the_new_meaning(self):
  +         """Any collection of the letters, in any order, is one canonical set (``_STORE_LETTERS`` order), so the frames
  +         agree bitwise. A 2026.4.0 tuple without 'O' runs in the new meaning: ('U', 'L') updates the up and the left
  +         stores and transports the down store, whose completion keeps its span; ('D', 'L') moves it."""
  +         self.assertEqual(padded_frames._update_stores(('R', 'U', 'D')), ('D', 'U', 'R'))
  +         F, movement = self.step(update=('U', 'L'))
  +         for update in (('L', 'U'), ['U', 'L'], {'U', 'L'}, frozenset(('U', 'L'))):
  +             with self.subTest(update=update):
  +                 G, movement_G = self.step(update=update)
  +                 self.assertTrue(all(np.array_equal(a, b) for f, g in zip(F, G) for a, b in zip(f, g)))
  +                 self.assertTrue(np.array_equal(movement, movement_G))
  +         F_down, _ = self.step(update=('D', 'L'))
  +         off_span = lambda frame: [np.linalg.norm(a - a @ s.T @ s) for a, s in zip(frame[0], self.start[0])]
  +         self.assertLessEqual(max(off_span(F)), 1e-12)          # the down store transported: the start's row spans
  +         self.assertGreater(max(off_span(F_down)), 1e-3)        # the down store aligned: moved
  ```
- **:899 -> :934**
  ```
  -         self.assertEqual(calls, [('ULRO', t), ('UL', t), ('RO', t), ('UL', t)])
  +         self.assertEqual(calls, [(('D', 'U', 'L', 'R'), t), (('D', 'L'), t), (('U', 'R'), t), (('D', 'L'), t)])
  ```
- **:903-904 -> :938-939**
  ```
  -         """Two oversampled passes, the truncating pass, then 'UL' and 'RO' in turn; the simultaneous schedule
  -         updates every store on every pass."""
  +         """Two oversampled passes, the truncating pass, then ('D', 'L') and ('U', 'R') in turn; the simultaneous
  +         schedule updates every store on every pass."""
  ```
- **:907 -> :942-943**
  ```
  -         self.assertEqual(calls, [('ULRO', None), ('ULRO', None), ('ULRO', t), ('UL', t), ('RO', t), ('UL', t)])
  +         self.assertEqual(calls, [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t),
  +                                  (('D', 'L'), t), (('U', 'R'), t), (('D', 'L'), t)])
  ```
- **:909 -> :945-946**
  ```
  -         self.assertEqual(calls, [('ULRO', None), ('ULRO', None), ('ULRO', t), ('ULRO', t)])
  +         self.assertEqual(calls, [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t),
  +                                  (('D', 'U', 'L', 'R'), t)])
  ```
- **:912 -> :949-950**
  ```
  -         """max_passes = 2 with no oversampled phase: the 'ULRO' pass, then one slot, no pair: the final 'UL'."""
  +         """max_passes = 2 with no oversampled phase: the all-store pass, then one slot, no pair: the final
  +         ('D', 'L')."""
  ```
- **:914 -> :952**
  ```
  -         self.assertEqual(calls, [('ULRO', self.target[0]), ('UL', self.target[0])])
  +         self.assertEqual(calls, [(('D', 'U', 'L', 'R'), self.target[0]), (('D', 'L'), self.target[0])])
  ```
- **:919 -> :957-958**
  ```
  -         self.assertEqual(calls, [('ULRO', None), ('ULRO', None), ('ULRO', self.target[0])])
  +         self.assertEqual(calls, [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None),
  +                                  (('D', 'U', 'L', 'R'), self.target[0])])
  ```
- **:923 -> :962**
  ```
  -         """max_passes = 1: one truncating pass from the working start, or one 'ULRO' pass at the target."""
  +         """max_passes = 1: one truncating pass from the working start, or one all-store pass at the target."""
  ```
- **:927 -> :966**
  ```
  -                 self.assertEqual(calls, [('ULRO', self.target[0])])
  +                 self.assertEqual(calls, [(('D', 'U', 'L', 'R'), self.target[0])])
  ```
- **:980-983 -> :1019-1022**
  ```
  -         after the truncating one come in ('UL', 'RO') pairs (3-4, 5-6, 7-8) and the stop is tested after each pair,
  -         both movements below tol; never on the truncating pass, never across two pairs; the odd slot left by the cap
  -         (pass 9) runs one final 'UL' only when not stopped. Simultaneous: the first pass from the truncating one on
  -         below tol. Oversampled passes never stop. The stack maximum decides; the cap stops."""
  +         after the truncating one come in (('D', 'L'), ('U', 'R')) pairs (3-4, 5-6, 7-8) and the stop is tested after
  +         each pair, both movements below tol; never on the truncating pass, never across two pairs; the odd slot left by
  +         the cap (pass 9) runs one final ('D', 'L') only when not stopped. Simultaneous: the first pass from the
  +         truncating one on below tol. Oversampled passes never stop. The stack maximum decides; the cap stops."""
  ```
- **:990-993 -> :1029-1032**
  ```
  -         self.assertEqual(self.stop_after([hi] * 9 + [lo]), 10)                          # the cap: 3 pairs, a final 'UL'
  -         self.assertEqual(self.stop_after([hi] * 9, max_passes=9), 9)                    # an even cap: no final 'UL'
  -         self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final 'UL'
  -         self.assertEqual(self.stop_after([hi] * 6, max_passes=6), 6)                    # not stopped: the final 'UL'
  +         self.assertEqual(self.stop_after([hi] * 9 + [lo]), 10)                          # the cap: 3 pairs, a final ('D', 'L')
  +         self.assertEqual(self.stop_after([hi] * 9, max_passes=9), 9)                    # an even cap: no final ('D', 'L')
  +         self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final ('D', 'L')
  +         self.assertEqual(self.stop_after([hi] * 6, max_passes=6), 6)                    # not stopped: the final ('D', 'L')
  ```
- **:1006 -> :1045-1046**
  ```
  -         """Recorded: a stop after the first pair leaves (oversampled x2, truncating, 'UL', 'RO') and no more."""
  +         """Recorded: a stop after the first pair leaves (oversampled x2, truncating, ('D', 'L'), ('U', 'R')) and no
  +         more."""
  ```
- **:1015 -> :1055-1056**
  ```
  -         self.assertEqual(updates, ['ULRO', 'ULRO', 'ULRO', 'UL', 'RO'])
  +         self.assertEqual(updates, [('D', 'U', 'L', 'R'), ('D', 'U', 'L', 'R'), ('D', 'U', 'L', 'R'), ('D', 'L'),
  +                                    ('U', 'R')])
  ```

#### `tests/backend/test_uniform_padded_frame.py` (12 hunks)

- **:360 -> :360**
  ```
  -     def check(self, shape, old_ranks, work_ranks, out_ranks, options, update='ULRO'):
  +     def check(self, shape, old_ranks, work_ranks, out_ranks, options, update=('D', 'U', 'L', 'R')):
  ```
- **:388 -> :388**
  ```
  -         """Each store family transported, and the two colours ('UL', 'RO')."""
  +         """Each store family transported, and the two colours (('D', 'L'), ('U', 'R'))."""
  ```
- **:390 -> :390**
  ```
  -         for update in ('UL', 'RO', 'LRO', 'ULO', 'U', 'O'):
  +         for update in (('D', 'L'), ('U', 'R'), ('U', 'L', 'R'), ('D', 'U', 'L'), ('D',), ('U',)):
  ```
- **:514 -> :514**
  ```
  -         self.call(update='UL', out=(None, None))
  +         self.call(update=('D', 'L'), out=(None, None))
  ```
- **:517 -> :517-519**
  ```
  -         for update in ('', 'UU', 'ULX'):
  +         """The ragged step's validation (``padded_frames._update_stores``): bad letters name the letters; every str
  +         and a collection holding the 2026.4.0 letter 'O' add the 2026.4.0 sentence."""
  +         for update in ((), ('D', 'D'), ('D', 'L', 'X')):
  ```
- **:after 519 -> :522-539**
  ```
  +         for update in ('DL', 'ULRO', ('U', 'O')):
  +             with self.subTest(update=update), self.assertRaisesRegex(ValueError, 'update must be .*2026.4.0 form'):
  +                 self.call(update=update)
  + 
  +     def test_update_reaches_the_kernel_canonical(self):
  +         """Any order or collection of the letters reaches the kernel, whose jit static it is, in ``_STORE_LETTERS``
  +         order, one static per set; the frames agree bitwise."""
  +         kernel, seen = uniform_padded_frames._utv_align_kernel, []
  + 
  +         def capture(dynamic, skeleton, split, update):
  +             seen.append(update)
  +             return kernel(dynamic, skeleton, split, update)
  +         with unittest.mock.patch.object(uniform_padded_frames, '_utv_align_kernel', side_effect=capture):
  +             runs = [self.call(update=u, out=(None, None)) for u in (('D', 'L'), ('L', 'D'), ['L', 'D'], {'D', 'L'})]
  +         self.assertEqual(seen, [('D', 'L')] * 4)
  +         (F0, movement0), others = runs[0], runs[1:]
  +         for F, movement in others:
  +             self.assertTrue(all(np.array_equal(a, b) for a, b in zip(F[:4] + (movement,), F0[:4] + (movement0,))))
  ```
- **:547 -> :567**
  ```
  -             self.call(update='RO')
  +             self.call(update=('U', 'R'))
  ```
- **:771 -> :791-792**
  ```
  -                          [('ULRO', None), ('ULRO', None), ('ULRO', t), ('UL', None), ('RO', None), ('UL', None)])
  +                          [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t),
  +                           (('D', 'L'), None), (('U', 'R'), None), (('D', 'L'), None)])
  ```
- **:773-774 -> :794-797**
  ```
  -                          [('ULRO', None), ('ULRO', None), ('ULRO', t), ('ULRO', None)])
  -         self.assertEqual(self.run_recorded(oversampling=0, max_passes=2, tol=0.0), [('ULRO', t), ('UL', None)])
  +                          [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t),
  +                           (('D', 'U', 'L', 'R'), None)])
  +         self.assertEqual(self.run_recorded(oversampling=0, max_passes=2, tol=0.0),
  +                          [(('D', 'U', 'L', 'R'), t), (('D', 'L'), None)])
  ```
- **:776 -> :799**
  ```
  -                          [('ULRO', None), ('ULRO', None), ('ULRO', t)])
  +                          [(('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), None), (('D', 'U', 'L', 'R'), t)])
  ```
- **:800 -> :823**
  ```
  -         self.assertEqual(self.stop_after([hi] * 10), 10)                                # 3 pairs and a final 'UL'
  +         self.assertEqual(self.stop_after([hi] * 10), 10)                                # 3 pairs and a final ('D', 'L')
  ```
- **:802 -> :825**
  ```
  -         self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final 'UL'
  +         self.assertEqual(self.stop_after([hi, hi, hi, lo, lo, hi], max_passes=6), 5)    # stopped: no final ('D', 'L')
  ```

#### `tests/backend/test_padded_start.py` (1 hunks)

- **:480 -> :480-481**
  ```
  -                                      dense_tied_project(start_dense, G, groups, x0.to_dense()), stores, 'ULRO', groups)
  +                                      dense_tied_project(start_dense, G, groups, x0.to_dense()), stores,
  +                                      ('D', 'U', 'L', 'R'), groups)
  ```

#### `tests/test_aligned_padded_frame_frontend.py` (1 hunks)

- **:125-126 -> :125-126**
  ```
  -         UF1, umv = uv.align_padded(split, update='UL')
  -         UFb, umvb = uniform_padded_frames.utv_align_padded_frame(UF.data, uv.variations.data, split, update='UL')
  +         UF1, umv = uv.align_padded(split, update=('D', 'L'))
  +         UFb, umvb = uniform_padded_frames.utv_align_padded_frame(UF.data, uv.variations.data, split, update=('D', 'L'))
  ```

#### `tests/test_dispatch.py` (12 hunks)

- **:2008 -> :2008**
  ```
  -         two-colour half (R and O transported), the zero tensor's empty frame, and a stack C = (2,)."""
  +         two-colour half (right and up transported), the zero tensor's empty frame, and a stack C = (2,)."""
  ```
- **:2019-2021 -> :2019-2021**
  ```
  -                            ((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, 'ULRO'),
  -             'shared, UL': (bvf.t3_orthogonal_representations(x_tied)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
  -                            ((3, 3, 3, 3), (1, 3, 4, 3, 1)), (None, None), (0, 1, 0, 1), 'UL'),
  +                            ((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, ('D', 'U', 'L', 'R')),
  +             'shared, DL': (bvf.t3_orthogonal_representations(x_tied)[0].data, ((2, 2, 2, 2), (1, 2, 2, 2, 1)),
  +                            ((3, 3, 3, 3), (1, 3, 4, 3, 1)), (None, None), (0, 1, 0, 1), ('D', 'L')),
  ```
- **:2023 -> :2023**
  ```
  -                            ((2, 3, 3, 2), (1, 2, 3, 2, 1)), (None, None), None, 'ULRO'),
  +                            ((2, 3, 3, 2), (1, 2, 3, 2, 1)), (None, None), None, ('D', 'U', 'L', 'R')),
  ```
- **:2025 -> :2025**
  ```
  -                            ((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, 'ULRO'),
  +                            ((3, 4, 4, 3), (1, 3, 4, 3, 1)), ((3, 3, 3, 3), (1, 3, 3, 3, 1)), None, ('D', 'U', 'L', 'R')),
  ```
- **:2058-2061 -> :2058-2061**
  ```
  -         every leaf -- two oversampled passes, the truncating pass and one 'UL' / 'RO' pair (tol 0, max_passes 5), at
  -         d = 2, with the dense projection of a hint of the matching array type -- eagerly, and with use_jit=True (the
  -         memoized jitted step; numpy in is converted to jax, the numpy hint handed on as is, jax out). The movements
  -         equal numpy's (x64 on in this module; they do not depend on the gauge)."""
  +         every leaf -- two oversampled passes, the truncating pass and one ('D', 'L') / ('U', 'R') pair (tol 0,
  +         max_passes 5), at d = 2, with the dense projection of a hint of the matching array type -- eagerly, and with
  +         use_jit=True (the memoized jitted step; numpy in is converted to jax, the numpy hint handed on as is, jax out).
  +         The movements equal numpy's (x64 on in this module; they do not depend on the gauge)."""
  ```
- **:2345 -> :2345**
  ```
  -     # capped at 12 passes: oversampled and truncating passes, pairs, the final 'UL' (12 = 3 + 4 pairs + 1 and
  +     # capped at 12 passes: oversampled and truncating passes, pairs, the final ('D', 'L') (12 = 3 + 4 pairs + 1 and
  ```
- **:2395 -> :2395**
  ```
  -         self.assertEqual((lengths['two-colour, p=5'], lengths['two-colour, p=0']), (12, 12))   # the final 'UL'
  +         self.assertEqual((lengths['two-colour, p=5'], lengths['two-colour, p=0']), (12, 12))   # the final ('D', 'L')
  ```
- **:2416 -> :2416**
  ```
  -                 self.assertEqual(seen, [(u, on_jax) for u in ('ULRO', 'ULRO', 'ULRO', 'UL')])
  +                 self.assertEqual(seen, [(u, on_jax) for u in (('D', 'U', 'L', 'R'),) * 3 + (('D', 'L'),)])
  ```
- **:2424-2426 -> :2424-2427**
  ```
  -         oversampled 'ULRO', the truncating 'ULRO', 'UL', 'RO' -- and a second run at the same structure, other values,
  -         compiles none. A structure no other test uses, so the count is this run's. The wrapper is memoized (one
  -         object), the house ``_jitted`` contract; jax 0.10 would share the programs across wrappers anyway."""
  +         oversampled all-store, the truncating all-store, ('D', 'L'), ('U', 'R') -- and a second run at the same
  +         structure, other values, compiles none. A structure no other test uses, so the count is this run's. The wrapper
  +         is memoized (one object), the house ``_jitted`` contract; jax 0.10 would share the programs across wrappers
  +         anyway."""
  ```
- **:2456-2458 -> :2457-2459**
  ```
  -         """T12's cases, plus the odd remainder's final 'UL' (tol 0, cap 10: 2 + 1 + 3 pairs + 1): the numpy (eager)
  -         loop and the jitted loop run the same passes (NaN in the same slots), frames to 1e-12, movements to 1e-10;
  -         early stops included (the defaults stop at 9)."""
  +         """T12's cases, plus the odd remainder's final ('D', 'L') (tol 0, cap 10: 2 + 1 + 3 pairs + 1): the numpy
  +         (eager) loop and the jitted loop run the same passes (NaN in the same slots), frames to 1e-12, movements to
  +         1e-10; early stops included (the defaults stop at 9)."""
  ```
- **:2465 -> :2466**
  ```
  -         cases['the final UL'] = cases['defaults'][:4] + ({'tol': 0.0},)
  +         cases['the final DL'] = cases['defaults'][:4] + ({'tol': 0.0},)
  ```
- **:2480 -> :2481**
  ```
  -         self.assertEqual((lengths['defaults'], lengths['the final UL']), (9, 10))
  +         self.assertEqual((lengths['defaults'], lengths['the final DL']), (9, 10))
  ```

## Appendix C. K9's residue, HEAD → tree (R3's own text in the new meaning; one alignment artifact, `padded_frames.py:51`)

- NEW `t3toolbox/backend/padded_frames.py:51`: _STORE_NAMES = ('down', 'up', 'left', 'right')
- NEW `t3toolbox/backend/padded_frames.py:57`: update:  typ.Collection[str],   # single store letters, any order, e.g. ('D', 'L'): D down, U up, L left, R right
- NEW `t3toolbox/backend/padded_frames.py:63`: "(D down, U up, L left, R right), e.g. ('D', 'L'); got %r" % (update,))
- NEW `t3toolbox/backend/padded_frames.py:67`: "not ('U', 'L'), which now means up and left")
- NEW `t3toolbox/manifold.py:175`: update:            typ.Collection[str] = ('D', 'U', 'L', 'R'),   # store letters, any order: 'D' down, 'U' up, 'L' left, 'R' right; the rest transported
- NEW `tests/backend/test_padded_frame.py:754`: "(D down, U up, L left, R right), e.g. ('D', 'L'); got 'ULRO'; that is the 2026.4.0 form, "
- NEW `tests/backend/test_padded_frame.py:756`: "letter and pass a tuple, e.g. 'UL' -> ('D', 'L'), not ('U', 'L'), which now means up and "
- NEW `tests/backend/test_padded_frame.py:761`: agree bitwise. A 2026.4.0 tuple without 'O' runs in the new meaning: ('U', 'L') updates the up and the left
- NEW `tests/backend/test_padded_frame.py:762`: stores and transports the down store, whose completion keeps its span; ('D', 'L') moves it."""
- NEW `tests/backend/test_padded_frame.py:772`: self.assertLessEqual(max(off_span(F)), 1e-12)          # the down store transported: the start's row spans
- NEW `tests/backend/test_padded_frame.py:773`: self.assertGreater(max(off_span(F_down)), 1e-3)        # the down store aligned: moved
- NEW `tests/backend/test_ranks.py:436`: physical frames: down = the Tucker factors' rows, up = the TT-shaped cores' middle legs. A frame whose stores
- NEW `tests/backend/test_ranks.py:437`: all differ is not minimal; a minimal one is. (The verdict is symmetric in down/up; the keywords are pinned.)"""
- NEW `tests/test_dispatch.py:2008`: two-colour half (right and up transported), the zero tensor's empty frame, and a stack C = (2,)."""
- NEW `tests/test_frame_variations_format.py:540`: whose four stores all differ: the down store is ``data[0]``, the Tucker factors ``(n, N)`` below each Tucker
- NEW `tests/test_frame_variations_format.py:541`: edge; the up store is ``data[1]``, the TT-shaped cores above it, counted by their middle leg."""
- NEW `tests/test_uniform_frame_variations_format.py:165`: supercores: the down supercore holds the Tucker factors at width nD, the up supercore the TT-shaped cores at
- NEW `tests/test_uniform_frame_variations_format.py:279`: Tucker variations' rows are the up store's (width nU), the TT variations' middle leg the down store's (nD)."""
- NEW `tests/test_uniform_manifold.py:1091`: all differ: nD pads the down store (the Tucker factors, the TT variations' middle leg), nU the up store (the
- NEW `tests/test_uniform_weighted.py:807`: (``from_t3weights`` copies one Tucker vector into both): the down weights, sized by the Tucker factors' rows,
- NEW `tests/test_uniform_weighted.py:808`: go to the down supercore at width nD; the up weights, sized by the TT-shaped cores' middle legs, to width nU."""
- GONE (HEAD) `t3toolbox/backend/padded_frames.py:554`: update:            str = 'ULRO',                             # stores updated: 'U' up, 'L' left, 'R' right, 'O' down; the rest transported
- GONE (HEAD) `t3toolbox/frame_variations_format.py:1765`: - then two-colour pairs: a ``'UL'`` pass (up and left stores) and an ``'RO'`` pass (right and down), the other

## Appendix D. The §5.2 additions (pure insertions; `git diff -U0 HEAD` headers)

- `tests/test_frame_variations_format.py` -537,0 +538,11 (in class TestFrameVariationsFormat(unittest.TestCase):)
- `tests/test_weighted.py` -386,0 +387,19 (in class TestT3FrameWeights(unittest.TestCase):)
- `tests/test_uniform_frame_variations_format.py` -17,0 +18 (in import t3toolbox.backend.ufv_masking as ufv_masking)
- `tests/test_uniform_frame_variations_format.py` -160,0 +162,32 (in class TestUT3Frame(unittest.TestCase):)
- `tests/test_uniform_frame_variations_format.py` -242,0 +276,28 (in class TestUT3Variations(unittest.TestCase):)
- `tests/test_uniform_weighted.py` -578,0 +579 (in import t3toolbox.backend.utv_operations as utv_operations)
- `tests/test_uniform_weighted.py` -801,0 +803,25 (in class TestUT3FrameWeights(unittest.TestCase):)
- `tests/backend/test_ranks.py` -433,0 +434,20 (in class TestSharedFrameHasMinimalRanks(unittest.TestCase):)
- `tests/test_uniform_manifold.py` -1088,0 +1089,20 (in class TestCrossLayerConverters(unittest.TestCase):)
