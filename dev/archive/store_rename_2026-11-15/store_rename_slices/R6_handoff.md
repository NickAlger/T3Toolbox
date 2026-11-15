# R6 handoff: acceptance (store rename, 2026.5.0)

Implementer's record for the session. Base: HEAD `1ddeadcf` (R5 = `6062f3d2`, plus the handoff and plan-draft-28
commits); during R6 the session added five `dev/`-only commits (now `e659d6ab`; no code, test or docs file changed). Nothing is committed. Scope (plan §7 row R6, §5, §6, the brief):

- §5 end to end, one heavy job at a time;
- the alignment sweep;
- the T3Polynomial follow-up, prepared there and not committed;
- the root release notes, checked;
- this file.

**Changed in the tree:** 56 files under `t3toolbox/`, `tests/`, `examples/` and `docs/`, whitespace only (the
alignment sweep, §2). **New:** `dev/store_rename_slices/R6_alignment_sweep.md`; `dev/store_rename_slices/r6_align_scan.py`
(the sweep's scan, one-off, like the other gate tools); this file. **Elsewhere:**
`/home/nick/repos/T3Polynomial/scripts/g01_g0_common.py` (§3), uncommitted. The version string is not bumped:
`pyproject.toml`, `CITATION.cff` and the `__init__` fallback still read 2026.4.0.

**Result.** Every §5 item passes. Nothing fails, and no fix outside the rows was made. Two notes for the session (§5):

- the floor-env battery's `compare` step needs Python 3.11, because it imports `store_rename_apply.py`;
- the one-line T3Polynomial change has to be two lines.

## 1. §5 end to end

Runs were one at a time. Environments: `t3toolbox` (py3.11.15, numpy 2.4.6, jax 0.10.2) and the floor `tttt` (py3.9.19,
numpy 1.22.4, jax 0.4.30, pytest 8.4.1 without xdist or subtests). The runner, logs and outputs are under the session
scratchpad `r6/`.

### 1.1 The gates (`dev/store_rename_check.py --stage R6 --hits dev/store_rename_slices --prev HEAD`, on the final tree)

| gate | result |
|---|---|
| K0, K1, K2, K3, K3-docs, K4, K5, K6, K7, K8, K8b | **PASS (0)** over the code tree, the docs and the notes |
| K9, K9-converse, K9-registered; K9-docs, -converse, -registered | **PASS** |
| K10/K10b | **PASS**: 37 in place, 23 re-laid out, 0 pending |
| K11 | **PASS**: 94 fenced spans byte-identical to the tag; 206 rewritten blocks excepted and checked by K12 |
| K12 | **PASS: 301 of 301** blocks filled, every new text the tree verbatim |
| K13 | **PASS** (K13-generic INFO 2, K13-unrecognised INFO 160, as at R5) |
| K14 | **PASS**: 0 DNT tokens lost or added; the contract/einsum strings are identical |
| K15 / K15-cross | **0 NAME hunks** against HEAD (the sweep is whitespace only); K15-cross **PASS** |
| K16 | REPORT 21 LaTeX width forms, all signed off at R5 (unchanged) |
| K17 | **PASS** (891 NAME hits to a bare frame letter, 0 bound from a variation core) |
| K17 shape checks, both directions | **PASS**: K17-shape-frame (no `D`/`dD`/`U`/`dU` on a V/H-shaped array), K17-shape-converse (no V/H name on a core-shaped array) and K17-shape-point, 223 NAME/array pairs resolved |
| `store_rename_diagram_check.py all` | self-test **PASS**; K10/K10b **PASS**; K11 **PASS**; K12 **PASS** (301/301) |

**K15 over the whole arc against `v2026.4.0`: 0 unlisted hunks.**

- A direct `difflib` of the tag's NAME streams against the tree is not meaningful: R2's 4,896 in-place renames merge
  with R3/R4's hunks into 3,237 mixed hunks.
- So the arc is checked as the chain it is. Each commit was extracted with `git archive` and gated against its
  predecessor with that slice's own listing.

| step | check | result |
|---|---|---|
| tag → R2 `5153a0be` | positional against `R1_hit_table.tsv` (`--realized`) | K15 **PASS** (0 positions outside the table), K15-realized **PASS**, K14 **PASS** |
| R2 → R3 `74683a39` | `R3_name_hunks.tsv`, R3's `--names` | **36 hunks, all listed**, K15-cross PASS, K14 PASS |
| R3 → R4a `e3085085` | `R4a_name_hunks.tsv`, `--names w,o` | **40 hunks, all listed**, K15-cross PASS, K14 PASS |
| R4a → R4b `f5ef5a8c` | `R4b_name_hunks.tsv`, `--names dGxi,Hxi,rsh,rsig_hat` | **36 hunks, all listed**, K15-cross PASS, K14 PASS |
| R4b → R5 `6062f3d2` | — | **0 hunks**, K14 PASS |
| R5 → R6 tree | — | **0 hunks**, K14 PASS |

For the R2 and R3 steps the hit table omits its 24 `R4a-row` rows (the session's R4a rulings, realized at R4a). With
them included, R2's K14/K15-realized and R3's base recount flag exactly those not-yet-realized rows: `udots`, and the
eight comprehension `V`/`H`.

**K14, the DNT names against the tag:**

- PASS at every step of the chain.
- At R2 it compares the DNT multisets with the tag directly, the hit-table images subtracted.
- From R3 on, every added DNT token belongs to listed new code: 291 at R3, 24 at R4a, 26 at R4b.

**K17, both directions:** the shape checks above (frame forms on variations and variation names on cores), plus K17's
binding check.

### 1.2 The bitwise battery

| environment | reference | result |
|---|---|---|
| `t3toolbox` | R1's `dev/store_rename_slices/battery_ref` (the tag, this env), B9 from `battery_ref/b9` | **PASS (bitwise)**: 2,190 arrays, 856 texts, 0 errors |
| floor `tttt` | the tag worktree (`<scratch>/t3_v2026.4.0`, `e364fe01`) run in the floor env: `b9-save` then `run`; the tree then `run` with those B9 files | **PASS (bitwise)**: 2,190 arrays, 856 texts, 0 errors in both runs |

Bitwise equality holds within one environment (plan §9), so the floor env gets its own tag reference.

- **Note.** The floor `compare` step stops at `store_rename_apply.py`'s `assert sys.version_info[:2] == (3, 11)`; it
  imports that file for the text map.
- The two floor-env records were therefore compared with the `t3toolbox` Python.
- `compare` only reads the two saved records (`np.array_equal`), so both arrays still come from the floor env.

### 1.3 Suite, slow tier, doctests, examples, docs

| run | result |
|---|---|
| suite, `t3toolbox`, `pytest tests/ -q -n 3` | **1,347 passed, 4 skipped, 48,724 subtests passed**, 0 failed (19 min 31 s); the one warning is the intended adam-on-manifold one |
| slow tier, `T3TOOLBOX_SLOW_TESTS=1 pytest tests/test_oracle_sweep.py tests/backend/test_uniform_curvature.py -q -p no:cacheprovider`, serial, `t3toolbox` | **66 passed, 13,173 subtests passed** (17 min 32 s); two adam-on-manifold warnings, intended |
| module doctests, `pytest --doctest-modules t3toolbox` | **241 passed** in `t3toolbox`; **241 passed** in the floor env |
| `python -m doctest docs/getting_started.rst` | **81/81** in both envs |
| every `docs/*.md` and `docs/contributor/*.md` page except `doctest_style.md` (CI's command) | **pass** in both envs |
| expected outputs vs the tag (static: per docstring for `.py`, per page for `.md`/`.rst`, every non-`dev/` file changed since the tag) | **3,177 examples at the tag, 3,177 in the tree, 0 expected outputs changed**; 153 sources changed (renames, letter tuples, and the sweep's comment realignments: 20 in the first pass, 37 after Revision 1) |
| every example (CI's examples loop, `t3toolbox` env with optax 0.2.8 and scipy) | **14 of 14 PASS** (447 s; the longest is `fit_shared_factors_jetted_probes.py` at 130 s) |
| `python -m sphinx -W --keep-going -b html docs <scratch>/html` | **build succeeded**, exit 0, no warning (16 min 27 s) |
| suite, floor env, single process, `pytest tests/ -q -p no:cacheprovider` | **1,346 passed, 5 skipped**, 0 failed (38 min 8 s); the extra skip against the current env is by design, `test_dispatch.py::test_eager_jax_compiles_nothing_twice` ("jax 0.4.30: jax.monitoring cannot unregister a listener", confirmed with `-rs`); the 4 warnings are the intended adam-on-manifold ones |

**Leftover 2026.4.0 store words in the rendered pages.** I grepped the `-W` build's HTML text (tags stripped; `_sources`,
`_static` skipped) for `up_tucker_cores`, `down_tt_cores`, `nO` (word-bounded), `'ULRO'` and "outer-orthogonal". Hits
are in three pages, all carved out (my grep's words; the reviewer's wider grep adds the two items after the list):

- `glossary.html`: the "(2026.4.0, names reversed: …, `up_tucker_cores`, `down_tt_cores`)" retired-spellings alias
  group, and the orthogonal-core entry's "also: outer-orthogonal" (N17's alias).
- `batching_and_stacking.html`: the §8 paper ↔ code map's T4S line, "`O` (outer-orthogonal) = `up_tt_cores` (our `U`,
  up-orthogonal)".
- `release_notes.html`: "Upgrading from 2026.4.0". This covers the old → new table, the migration snippet's `SWAP` map,
  and the quoted `update` message with `'ULRO'`. These are records that name the old spellings on purpose (plan §5.3).

Also carved out (review N4):

- `entries_apply_probe.html` carries `Oᵢ` (`entries_apply_probe.md:260–273`), inside its §8 paper ↔ code map.
- `release_notes.html` also carries the past releases' CHANGELOG records: [2026.4.0]'s `('UL', 'RO')` and
  [2026.2.0]'s "Tucker up-SVD".

There are **0** `nO` hits and **0** in any API page.

**Timing.** All library edits (the sweep) were in place before the first run started (15:39). One exception is
`utv_operations.py:484`, where a comment's whitespace was moved at 15:46, during the suite. That line is
token-identical (K15 0 hunks). Everything from the slow tier on started after it.

## 2. The alignment sweep (`R6_alignment_sweep.md`, every fix and every judgement)

- **Revision 1 (the R6 review):** the first pass paired adjacent lines only, so its "0 near-misses / 0 drifted"
  meant adjacent pairs. The reviewer's bridge scan (columns split by one or two comment-less lines) found 74 more
  pairs. 62 lines moved (A1–A4, the NIT); the 19 bridged pairs left are separate groups (J3b). Bridged drift against
  the tag is now 0. `r6_align_scan.py --bridge` lists them, report only.
- **What changed:** 277 lines in 56 files, whitespace only. `git diff -w` is empty, the token streams equal HEAD's
  (docstring whitespace aside), K15 finds 0 NAME hunks, and no doctest output changed (37 doctest source lines moved
  a comment).
- **What remains:** over the 109 `.py`/`.md`/`.rst` files the arc touched, no comment or doctest column is within two
  columns of its neighbour's, adjacent or bridged within one indent.
- **Drift:** 0 tag alignments broken by the arc, adjacent (3,334 pairs) or bridged.
- **The known items:**
  - R4a's named examples: `optimizers.py:228`'s `frame:`, `utv_operations.py:276–279`, `:486–487`, `:300–302`,
    `ufv_conversions.py:407–408` and `test_uniform_manifold.py:111–112`;
  - R3's `test_frame_variations_format.py:544–545`;
  - R5's N3, `numerical_contracts.md:105–107`.
- **Fix classes:**
  - 114 chain moves to the block's majority column;
  - 38 re-pads to a line that sticks out by 1–2;
  - 39 type-column slips, 35 of them a `frame:` argument one or two columns off its block (a pattern copied across
    the backend signatures);
  - 24 hand fixes;
  - 62 bridged fixes (Revision 1).
- **Line origin:** 42 of the changed lines were touched by the arc; 235 were as at the tag.
- **Judgements (J1–J9):**
  - Left: long-name type stick-outs (24); stick-outs ≥ 3 columns (455, as in the reference); separate groups (26 adjacent, 19 bridged:
    return tuples against `]:`, arguments against return lines, doctest sub-blocks); unpadded assignments beside
    aligned pairs (6).
  - Two tag pairings were resolved toward their block: `NewtonInfo.misfit` and `uniform_least_squares_problem`'s
    `sharing`. These are the scan's only "drift" entries, made on purpose.
  - Seven already-long lines grow by 1–2 characters, to at most 139.
  - `t3_svd.py:45` was reverted to the tag.
  - No inventoried block was touched.
- **Rerunning the check:** `python dev/store_rename_slices/r6_align_scan.py [--bridge]`.
- **The runs in §1:** they predate Revision 1. Revision 1 is whitespace only and token-identical (K15 0 hunks, 0
  doctest wants changed); the reviewer re-runs the heavy jobs.

## 3. The T3Polynomial follow-up (prepared, NOT committed there)

`scripts/g01_g0_common.py`, `zero_base`:

```
-    up, down, left, right = frame
+    down, up, left, right = frame
     left = list(left); right = list(right)
     left[-1] = np.zeros_like(left[-1]); right[0] = np.zeros_like(right[0])
-    return tuple(up), tuple(down), tuple(left), tuple(right)
+    return tuple(down), tuple(up), tuple(left), tuple(right)
```

- **Two lines, not one.** Changing only `:121` would hand back `(up, down, …)` and swap the stores. The return line
  renames with it, so the function is unchanged (plan §2: "its locals are renamed after the release").
- **Check:** `zero_base` run on `fvc.t3_orthogonal_representations(...)[0]` of the battery's unequal-store `xS` returns
  `data[0]`, `data[1]` unchanged (Tucker factors `(3,5),(6,6),(2,7)`), with `left[-1]`, `right[0]` zeroed.
- **Suite:** `PYTHONPATH=/home/nick/repos/T3Toolbox pytest -q -n 2` from that repo's root (t3toolbox env): **69 passed,
  33 skipped, 547 subtests passed, 0 failed** (5 min 58 s).
- **One collection error, unrelated and pre-existing:** `scripts/s02_probe_smoke_test.py` imports `ufl` (the fenicsx
  env). The repo has no pytest config, so a root run collects that script.

## 4. Release notes

- **The file and the CHANGELOG:** `release_notes_2026.5.0.md` (repo root, untracked) equals the CHANGELOG `[2026.5.0] —
  unreleased` section body with its boundary blank lines stripped. This is exactly the 2026.4.0 precedent:
  `release_notes_2026.4.0.md` is its section the same way.
- **Version:** not bumped. The release commit stamps the CHANGELOG date and bumps `pyproject.toml` / `CITATION.cff` /
  the `__init__` fallback; Nick tags.

## 5. For the session

- **Commit by name:**
  - the 56 swept files (`git diff --name-only`);
  - `dev/store_rename_slices/R6_alignment_sweep.md`, `R6_handoff.md` and `r6_align_scan.py`.
  Leave out the untracked `release_notes_*.md` files, `dev/check_t3svd_retraction_second_order.py`,
  `dev/newton_timing_plan_2026-11-10.md`, `dev/gn_newton_switch_plan_2026-11-13.md` (appeared during R6, not
  mine) and `dev/store_rename_slices/R2_pass1.diff`.
- **T3Polynomial:** commit only after the release (§3).
- **Tool note:** the battery's `compare` cannot run on the floor Python, because `store_rename_apply.py` asserts 3.11.
  The plan's "R6 repeats it in the floor environment" holds for the runs; the comparison of their records ran on 3.11.
- **After R6** (plan §6): the 601-row docs-consistency pass, then K2–K9 again.
