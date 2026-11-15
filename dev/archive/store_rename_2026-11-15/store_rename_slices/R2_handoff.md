# R2 handoff — the scripted pass (plan draft 24, agreed round 25; §7 row R2)

Implementer, 2026-11-11. **Not committed.** The working tree holds R2's result: 81 files rewritten by the script
(32 in `t3toolbox/`, 34 in `tests/`, 15 docs incl. `README.md`/`CLAUDE.md`; `examples/` 0) plus eleven §4.6 hand edits, and
`dev/store_rename_apply.py` gains `--apply`.

## 1. What was run

```
PY=/home/nick/miniconda3/envs/t3toolbox/bin/python
$PY dev/store_rename_apply.py --apply --i-am-r2 --out <scratch>/r2 --scratch <scratch>/r2tree
```

`--apply` (new in R2) refuses unless every target file equals `HEAD`, runs PASS 1 (old tokens → sentinels
`ZQX%04d_XQZ`) and PASS 2 (sentinels → new tokens) into the scratch trees exactly as R1's dry run, writes **the single
reviewable pass-1 diff** (`dev/store_rename_slices/R2_pass1.diff`, 532 KB, `-U0`: every sentinel placed, against
`HEAD`; uncommitted — commit it or not, the session's call; difflib's `-U0` still prints 78 unchanged lines inside
its replace blocks, so it is not a pure zero-context diff, though every change in it is a sentinel), copies pass 2 over the tree, and asserts zero sentinels in
every written file.

- **Sentinels:** 4,896 placed by pass 1, 4,896 removed by pass 2, **0 left** in the tree (K0). 283 keys.
- **The hit table applied is byte-identical to R1's** `dev/store_rename_slices/R1_hit_table.tsv` (`cmp`): R2 applied
  exactly the reviewed table, nothing more.
- **Bijection:** 0 errors; the 12 deviations are the explained one-way/scoped ones of R1.
- **Fences:** every §4A fenced span was skipped (K11 below); the letter strings (`'ULRO'`, `'U' in update`) stay old
  on purpose until R3; no realignment (the plan's R2 row has none; §4.10's signature, body and doctest realignment is
  R4a).

## 2. Counts: hits applied per tier per file

Tier totals: A1 493, A1-alias 6, A1-label 11, A2 636, A3 1,170, A3-explicit 58, A3s 189, A3x 32, B1 271, B2 50, B3 19,
B4 42, B5 6, B6 9, C 922, C-B 232, C-D 63, C-1.2a-B 85, C-1.2a-D 7, C-VH 243, C-sig 34, C-site 314, T-typo 4 —
**4,896**. By context: NAME 3,546, doctest NAME 225, COMMENT 785, docstring prose 132, STRING 41, exact literal 107,
DOC 48, doctest comment 12. Per file (the R1 hit table, applied):

| file | total | by tier |
|---|---|---|
| `t3toolbox/backend/sampling_derivatives.py` | 522 | A1 79, A3 172, A3x 26, C 189, C-1.2a-B 8, C-VH 24, C-sig 15, C-site 9 |
| `t3toolbox/uniform_frame_variations_format.py` | 332 | A1 23, A1-label 4, A2 90, A3 142, B1 29, B2 6, B3 4, C 30, C-site 4 |
| `t3toolbox/backend/padded_frames.py` | 321 | A2 25, A3 100, A3s 32, B1 22, B2 6, C 136 |
| `t3toolbox/backend/uniform_padded_frames.py` | 283 | A2 4, A3 2, A3s 84, B1 4, B2 14, C 175 |
| `t3toolbox/backend/probing.py` | 256 | A1 75, A3 103, C 44, C-1.2a-B 4, C-VH 27, C-site 3 |
| `tests/test_uniform_frame_variations_format.py` | 218 | A1 8, A2 67, A3 54, A3-explicit 38, B1 22, B3 9, C 8, C-D 4, C-site 8 |
| `t3toolbox/backend/utv_operations.py` | 195 | A1 1, A2 78, A3 86, C 14, C-1.2a-B 11, C-VH 5 |
| `t3toolbox/frame_variations_format.py` | 174 | A1 44, A1-label 1, A2 47, A3 19, A3-explicit 8, B2 4, C 21, C-site 30 |
| `t3toolbox/backend/ufv_conversions.py` | 170 | A1 6, A2 87, A3 61, C 16 |
| `t3toolbox/backend/tv_operations.py` | 145 | A1 61, A2 2, A3 19, C 36, C-1.2a-B 11, C-VH 16 |
| `t3toolbox/backend/fv_operations.py` | 144 | A1 27, A2 13, A3 44, B1 43, C 17 |
| `t3toolbox/backend/ufv_operations.py` | 140 | A1 4, A2 50, A3 61, B1 17, C 8 |
| `tests/test_frame_variations_format.py` | 135 | A1 10, A2 32, A3 18, B1 7, C-site 68 |
| `tests/backend/test_curvature.py` | 106 | B1 8, B6 6, C-B 6, C-1.2a-B 7, C-sig 5, C-site 74 |
| `tests/backend/test_padded_frame.py` | 96 | A2 14, A3 10, B1 41, B2 10, C 21 |
| `t3toolbox/backend/ufv_masking.py` | 95 | A1 19, A2 32, A3 31, A3s 2, C 11 |
| `tests/backend/_uniform_padded_frame_testing.py` | 90 | A3 14, A3s 51, B1 4, C 21 |
| `tests/test_probe_derivatives.py` | 88 | A3 6, A3-explicit 8, B1 4, C-1.2a-B 12, C-VH 58 |
| `t3toolbox/backend/t3_linalg.py` | 82 | A1 7, A3 10, C 10, C-B 51, C-VH 4 |
| `tests/test_jet_recurrence.py` | 80 | A3 5, B4 42, C 25, C-VH 8 |
| `tests/backend/_padded_frame_oracles.py` | 70 | A3 6, B1 15, C 49 |
| `t3toolbox/backend/apply.py` | 68 | A1 24, A3 7, C 6, C-1.2a-B 4, C-VH 12, C-sig 11, C-site 4 |
| `tests/test_tucker_tensor_train.py` | 68 | B5 6, C-B 62 |
| `tests/test_sharing.py` | 65 | A2 2, A3 11, C-B 6, C-D 31, C-1.2a-D 3, C-site 12 |
| `tests/backend/_curvature_oracles.py` | 62 | B6 3, C-B 39, C-1.2a-B 9, C-site 11 |
| `t3toolbox/backend/fv_conversions.py` | 59 | A1 25, A1-alias 6, A2 14, C 10, T-typo 4 |
| `t3toolbox/backend/sharing.py` | 57 | A1 4, A2 4, A3 29, A3-explicit 1, C-1.2a-B 11, C-1.2a-D 4, C-site 4 |
| `tests/_jet_trs_twins.py` | 54 | A1 12, C 29, C-VH 10, C-sig 3 |
| `t3toolbox/manifold.py` | 44 | A1 2, A2 6, A3 2, C-1.2a-B 8, C-VH 26 |
| `tests/test_manifold.py` | 44 | A1 4, A2 4, A3 4, B1 4, C-VH 4, C-site 24 |
| `tests/backend/test_uniform_curvature.py` | 42 | A2 3, A3 28, A3-explicit 3, C-site 8 |
| `tests/test_uniform_manifold.py` | 39 | A1 6, A2 13, A3 20 |
| `t3toolbox/backend/geometry.py` | 38 | A2 3, A3 7, B1 6, B2 4, C 8, C-B 10 |
| `tests/backend/test_uniform_padded_frame.py` | 38 | A2 1, A3 4, A3s 20, C 13 |
| `tests/test_dispatch.py` | 38 | A3 8, B1 8, B3 4, C-B 10, C-D 4, C-site 4 |
| `t3toolbox/backend/utv_sampling.py` | 35 | A3 2, C-VH 33 |
| `tests/test_weighted.py` | 34 | A1 4, A2 6, A3 4, B1 2, B3 2, C-B 6, C-D 10 |
| `t3toolbox/uniform_manifold.py` | 32 | A2 10, A3 16, B1 6 |
| `tests/test_uniform_probing.py` | 30 | A1 2, A2 2, A3 10, B1 8, C-D 4, C-site 4 |
| `t3toolbox/backend/entries.py` | 29 | A1 19, A3 5, C 3, C-site 2 |
| `tests/backend/test_padded_start.py` | 28 | B2 4, C 22, C-site 2 |
| `t3toolbox/backend/ranks.py` | 22 | A1-label 2, A2 7, A3 4, B1 9 |
| `t3toolbox/backend/t3_orthogonalization.py` | 20 | A1 9, C-B 7, C-site 4 |
| `tests/backend/test_ranks.py` | 18 | A1 2, B1 8, C-D 4, C-site 4 |
| `tests/backend/test_optimizers.py` | 17 | C-B 17 |
| `tests/backend/test_uniform_fitting.py` | 17 | A1 2, A3 3, B1 4, C-D 4, C-site 4 |
| `tests/backend/test_from_sweep_reuse.py` | 16 | C-VH 16 |
| `tests/backend/test_tv_operations.py` | 13 | A1 3, A3 2, C-site 8 |
| `tests/test_uniform_weighted.py` | 13 | A2 3, A3 8, B2 2 |
| `t3toolbox/tucker_tensor_train.py` | 8 | C-B 8 |
| `docs/glossary.md` | 7 | A1 2, A2 2, A3 3 |
| `t3toolbox/backend/ut3_linalg.py` | 7 | C-site 7 |
| `docs/frame_variations.md` | 6 | A2 4, A3 2 |
| `t3toolbox/backend/ut3_sampling.py` | 6 | C-site 6 |
| `t3toolbox/uniform_tucker_tensor_train.py` | 6 | C-B 4, C-site 2 |
| `tests/test_aligned_padded_frame_frontend.py` | 6 | A1 4, C-D 2 |
| `docs/contributor/padded_frame_internals.md` | 5 | A2 5 |
| `t3toolbox/backend/ut3_orthogonalization.py` | 5 | C-site 5 |
| `docs/contributor/sharing_internals.md` | 4 | A3 4 |
| `t3toolbox/optimizers.py` | 4 | A3x 4 |
| `tests/backend/test_geometry.py` | 4 | A3 4 |
| `tests/oracle_sweeps/sweep_models.py` | 4 | C-B 4 |
| `CLAUDE.md` | 3 | A1 1, A1-label 2 |
| `docs/batching_and_stacking.md` | 3 | A1 1, A1-label 2 |
| `docs/contributor/curvature_internals.md` | 3 | A3 3 |
| `docs/reading_signatures.md` | 3 | A3 3 |
| `docs/weighting.md` | 3 | A2 2, A3 1 |
| `t3toolbox/shared_geometry.py` | 3 | A1 3 |
| `tests/backend/test_t3_orthogonalization.py` | 3 | C-site 3 |
| `docs/chunking.md` | 2 | A3 2 |
| `docs/contributor/parameters_not_closures.md` | 2 | A3 2 |
| `docs/contributor/signature_style.md` | 2 | A3 2 |
| `docs/contributor/weighted_internals.md` | 2 | A3 2 |
| `docs/numerical_contracts.md` | 2 | A2 2 |
| `t3toolbox/backend/t3_operations.py` | 2 | C-B 2 |
| `tests/backend/test_fv_operations.py` | 2 | A3 2 |
| `tests/oracle_sweeps/o1_uniform.py` | 2 | A3 2 |
| `tests/test_chunk_size_estimator.py` | 2 | A3x 2 |
| `docs/sharing.md` | 1 | A3 1 |
| `t3toolbox/backend/optimizers.py` | 1 | A2 1 |
| `tests/oracle_sweeps/o1_ragged.py` | 1 | A2 1 |
| **total** | **4896** | A1 493, A1-alias 6, A1-label 11, A2 636, A3 1170, A3-explicit 58, A3s 189, A3x 32, B1 271, B2 50, B3 19, B4 42, B5 6, B6 9, C 922, C-B 232, C-D 63, C-1.2a-B 85, C-1.2a-D 7, C-VH 243, C-sig 34, C-site 314, T-typo 4 |

## 3. The hand edits (§4.6; the session: the four failing message tests only, "the `%s` literals by script, the prose words and the test regexes by hand")

The R1 preview showed exactly four failures after the scripted pass: message tests whose regex names a store the
flipped message no longer uses. B2 already flipped the `%s` store names (`_STORE_NAMES`) and the tests' `replaced('…')`
literals; the regexes and two prose store words in the messages they match are hand edits. Each was asserted against
its exact current text before replacing it.

| file:line | before | after | why |
|---|---|---|---|
| `tests/backend/test_padded_frame.py:405` | `'old_frame down\\[1\\]'` | `'old_frame up\\[1\\]'` | the test now replaces `'up'` (B2), and the message names it by `%s` |
| `tests/backend/test_padded_frame.py:409` | `'old_frame down\\[2\\]'` | `'old_frame up\\[2\\]'` | the same |
| `tests/backend/test_padded_frame.py:760` | `'frame down\\[1\\]'` | `'frame up\\[1\\]'` | `bad_frame` replaces `up[1]` (B1), named by `%s` |
| `tests/backend/test_uniform_padded_frame.py:303` | `'the down masks count'` | `'the up masks count'` | the test corrupts `masks[1]`, the up mask (`mU`, A3 scoped) |
| `t3toolbox/backend/padded_frames.py:102` (message) | `an orthogonal representation with up widths %r` | `… with down widths %r` | it formats `down_widths` (A2), the Tucker store's |
| `t3toolbox/backend/uniform_padded_frames.py:196` (message) | `… with these up and '` | `… with these down and '` | the layout is computed from `old[0]`, the down (Tucker) counts |

**The rest of §4.6 (the session's ruling after the first hand-back: "§4.6 puts store-naming messages and their tests
in R2; apply them now").** Three more messages called the Tucker store "up" in prose; each reads the down (Tucker)
masks or counts, and two tests match them:

| file:line | before | after | why |
|---|---|---|---|
| `t3toolbox/backend/uniform_padded_frames.py:225` (message) | `unequal old up masks or Tucker counts` | `unequal old down masks or Tucker counts` | the check compares `mD0` (A3 scoped), the down masks |
| `t3toolbox/backend/uniform_padded_frames.py:657` (message) | `which its up and left '` | `which its down and left '` | the layout comes from `counts[0]`, the down counts |
| `t3toolbox/backend/uniform_padded_frames.py:682` (message) | `unequal Tucker stores or old up masks;` | `unequal Tucker stores or old down masks;` | the check compares `mD`, the down masks |
| `tests/backend/test_uniform_padded_frame.py:320` | `'unequal old up masks'` | `'unequal old down masks'` | matches `:225` (its test is `test_unequal_old_down_masks_in_a_group`, A2) |
| `tests/backend/test_uniform_padded_frame.py:551` | `'unequal Tucker stores or old up masks'` | `'unequal Tucker stores or old down masks'` | matches `:682` |

After these, no `raise` message in the padded-frame and frame modules calls the Tucker store "up"; K9 lists none of the
eleven edited lines (its free-prose residue: 293 lines, R4's); K9-converse, K14 and K15 PASS; the four §4.6 test files
(`test_uniform_padded_frame`, `test_padded_frame`, `test_padded_start`, `test_frame_variations_format`): 140 passed,
1,791 subtests; the battery: PASS (bitwise) again.

Not hand-edited in R2 (R4a's free prose and labels): the subtest label `'a down width'`
(`test_padded_frame.py:405`), the comment at `test_uniform_padded_frame.py:301`, and the docstring at
`padded_frames.py:86`. See §6 for the other §4.6 messages.

## 4. Gates

`python dev/store_rename_check.py --stage R2 --hits dev/store_rename_slices --pass1 <scratch>/r2tree/pass1 --realized`
on the working tree (after the hand edits):

| gate | result |
|---|---|
| K0 | **PASS**: 0 sentinels in the tree (4,896 in the pass-1 tree, the hit count) |
| K11 | **PASS**: all 300 fenced spans byte-identical to `v2026.4.0`, located by text; no sentinel inside a fence in the pass-1 tree |
| K13 | **PASS** (fenced spans skipped): 0 shape lines whose width letter disagrees with the entry; 14 generic `tucker_cores`/`tt_cores` lines reported (§4 item 12, R4b) |
| K14 | **PASS**: 1,393 DNT rows, per-file multisets identical (images excluded); every `contract`/`einsum` string identical |
| K15 | **PASS**: 0 NAME positions changed outside the hit table; the converse (every NAME hit realized) PASS |
| K17 | **PASS**: 891 bare-frame-letter images, none bound from a variation; K17-shape-frame 0, K17-shape-converse 0 (215 name/array pairs); K17-shape-point 5, the corewise kernels item 12's `n` fixes in R4a/R4b (gated from R4b) |
| K8b, K9-converse | PASS |
| K1–K9, K16 | REPORT (the R3–R5 to-do lists; K1/K2's residue is all inside fenced blocks) |

**Battery** (`dev/store_rename_battery.py`, the tree under test = the working tree): **PASS (bitwise)** against the
`v2026.4.0` reference, 2,190 arrays and 856 texts (reprs through `map_text_a_tiers`), B9's cross-version loads
included; adapter flags recorded `NEW=True`, `LETTERS_NEW=False`.

**Doctest outputs unchanged:** 0 of 2,923 examples in the 81 touched files changed their expected output (every
`want`, compared statically against the tag).

**Suite** (`PYTHONPATH=$PWD python -m pytest tests/ -q -n 3`, t3toolbox env: Python 3.11, numpy 2.x, jax 0.10):
**1,338 passed, 4 skipped, 48,697 subtests passed, 0 failed** (19 min 28 s). The four message tests pass after the
hand edits.

**Doctests:** module doctests (`pytest --doctest-modules t3toolbox`) **241 passed**; `docs/getting_started.rst`
**pass**; every `docs/*.md` and `docs/contributor/*.md` page except `doctest_style.md` **pass** (tier A ran over the
docs, so the markdown doctests use the new names).

**Docs build:** `python -m sphinx -W -b html docs <scratch>` — **build succeeded**, no warning.

**Floor env** (`tttt`: Python 3.9.19, numpy 1.22.4, jax 0.4.30; serial, no `xdist`): the 27 touched test modules
plus `tests/test_oracle_sweep.py` (it drives the touched sweep modules): **1,066 passed, 5 skipped, 0 failed**
(38 min 47 s; 4 warnings, the library's intended ones such as adam-on-manifold). The files after `test_dispatch` were
also run as two parallel shards to finish sooner (265 passed; 274 passed, 2 skipped), redundant with the serial run.

## 5. Judgement calls

1. **The two message prose edits** (`padded_frames.py:102`, `uniform_padded_frames.py:196`) are the "prose words" of
   the session's instruction: without them the four fixed tests would pass on a message that calls the Tucker store
   "up". Nothing else was edited by hand.
2. **`--apply` writes pass 2 from the scratch trees** rather than sentinelizing the working tree in place: the result
   is the same bytes (pass 2 of pass 1), the working tree never holds sentinels, and the pass-1 diff is the review
   artifact §1.3 asks for.
3. **No realignment.** The R2 row lists none; §4 item 10 (signature blocks, bodies, doctests) is in R4a's row. The
   tree therefore carries the ±2-column drift the plan expects (e.g. field comments beside renamed fields).

## 6. What the plan could not be followed on, or should record (report, not improvised)

1. **§4.6's other messages: resolved** by the session's ruling and applied in R2 (§3's second table).
2. **Draft 24's text lacks three rows (the writer is recording them; they stay applied) the reviewer approved in R1's third pass and the committed script applies:**
   `tv_operations.tv_oblique_gauge_projection`'s `dG1→H1`, and the TT assembly twins' `dG_tilde(s)→H_tilde(s)`
   (`probing._assemble_tt_variations_core`, `assemble_tt_variations`, `sampling_derivatives._assemble_tt_variation_jets_dense`).
   The writer should record them.
3. **The floor env** (`tttt`: Python 3.9, numpy 1.22, jax 0.4.30) has neither `pytest-xdist` nor `pytest-subtests`,
   so its run is serial with plain pytest.
