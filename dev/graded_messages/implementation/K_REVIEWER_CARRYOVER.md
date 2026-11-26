# Reviewer carryover (K arc)

Reviewer 3 (Claude), 2026-11-25, about 23:45 CT, at the end of K1 part 0's build review, round 1. This page holds what a
successor needs that the slice records do not. Read it with `K_BRIEF.md`, then `BRIEF.md` ("What the reviewer does").
Reviewer 2's carryover (written before K1) is in git history; its K0 lessons and its "carried to later slices" list are
kept below.

## Where things stand

- **K0:** done, committed (`282b46a9`).
- **K1's design:** signed off by me in two rounds (`K1_DESIGN_REVIEW_r1.md`: NOT YET; `_r2.md`: SIGN OFF), approved by
  the orchestrator, with rulings 22 to 29 in `K_ARC_DECISIONS.md`. The note (`K1_DESIGN_NOTE.md`, revised after round 2's
  minor findings, `K1_DESIGN_RESPONSE_r2.md`) is the specification of the build; `k1/prose_edits.md` holds the texts (parts 0 and 1 in full;
  parts 2 to 5 get a prose round before each is built: the orchestrator's rule).
- **K1 part 0** (the kind and the order move into the batch): built on the local branch `k1-wip`, uncommitted. My
  review: `K1_part0_REVIEW_r1.md`, SIGN OFF with five minor findings; my full `-v` rerun of the suite on the final
  tree (1,887 passed, the same nodes as the implementer's log) is in `scratchpad/reviewer3/suite_v.log.gz`.
- **Next:** part 0's round 2 if needed, then part 1 (the dense reference with two weights). Part 1's prose is already
  in full (`prose_edits.md` §6); my round-2 finding 12 (part 1 must rewrite the tests that hand the reference a now-refused
  form, e.g. `tests/test_engine_data_space.py:139-158`) is the thing to check first in part 1.

## My tools (`implementation/k1/reviewer_tools/`, kept; scratch `scratchpad/reviewer3/` may be cleared)

| Tool | What it does |
|---|---|
| `dense_checks.py` | 41 dense loop checks of K1's mathematics (left alignment with coinciding sizes, bitwise single-weight against today's `jets.data_metric`, the bridge, `k!` at `(3,)` and `(2, 1)`, `<y, M y>` with `C` of 1). Imports the OLD `jets` functions: run it on a tree before part 5 |
| `batch_model.py` | a model of the batch class (design phase) |
| `real_batch_attack.py` | 31 attacks on the BUILT `LeastSquaresBatch` and the fieldless oracle (old positional forms, order normalization and treedef, `replace`, copies, placeholder leaves, one trace per kind/order, drawn batches, the subclass hash). Runs on both environments; adapt it in part 4 (`weight=` becomes `block_weight=`/`full_weight=`) |

Reviewer 2's tools in `k0/reviewer2_tools/` that I used and that work on K1: `tally2.py OLDLOG NEWLOG OLDROOT NEWROOT`
(per-test tally against the static status of each test), `unitdiff.py OLDROOT NEWROOT [--comments] files` (unit by
unit, exact text, comment blocks). `prose_check.py` does NOT parse K1's `prose_edits.md` (its headings carry no full
paths): I used a whitespace-normalized paragraph search instead (the review's §7 says how).

How I checked part 0, step by step (a model for the parts to come):
1. Rerun `k1/diff_audit.py BASE --part N --out ...` and diff against the implementer's `part<N>_audit.txt`; read every
   non-PAIRED entry against `git show HEAD:file`. A scope-by-scope `ast` scan (old oracle pairs against new batch
   pairs in the same function) is a second, independent check; a per-test-function scan of the helper calls likewise.
2. The battery: `git archive 282b46a9 t3toolbox tests | tar -x -C scratch/old`; `cd scratch/old && PYTHONPATH=$PWD
   python -B .../k1/battery.py ../bat/old old`; the tree's record with the form of the part; `battery.py --compare
   bat/old bat/<part> --expect part0|new`.
3. Mutations of each new guard or check in a scratch copy of `t3toolbox/` and `tests/` (a small script replacing one
   exact text, running the part's tests with `-x -n 2`, restoring).
4. Readers: grep for `getattr`/`hasattr` with the names, attribute reads, the protocol's text.
5. The tally with `tally2.py`; an `ast` check that no test changed class.
6. Texts: `unitdiff.py` of the library modules; the test diff filtered for lines that are not constructions or moved
   calls; trailing-comment columns of every changed line against its old counterpart and its neighbours (found a
   broken pair in a page doctest); the blank line before every doctest (found one lost).
7. **The suite log's validity:** compare the mtimes of the changed files with the `-v` log's start (its end less its
   duration). In part 0 two library files were edited during the run; I reran the whole suite.

## Lessons from K1 so far

- The implementer's audits are good and its tools reproduce exactly; the value of the review is in reading what they
  list, and in the independent scans (scope-by-scope, per test function), not in rerunning them.
- An audit that keys on class names misses a construction through a variable (`for cls in (...): cls(...)`).
- A docstring paragraph edited in a part can swallow the blank line before a doctest; doctest still passes, Sphinx
  `-W` is silent, the rendering is wrong. Check the line before every `>>>` block that follows an edited paragraph.
- Untested branches of a structural check show up only by mutation (`_require_draw`'s non-integer branch).

## Carried from reviewer 2 (still open)

- **K5:** the hygiene negative control's three patches (`_UNIFORM_MASKS`, `_mask_variable`,
  `ufv_masking.ufv_apply_variations_masks`) pointed at K5's masking points; the apply and entries cases of
  `TestKpAtTTSlack` added before the masking changes (`K_ARC_DECISIONS.md` 11); the scan floors re-measured (vocabulary
  `> 60`: 84 at K0; call sites `> 90`: 113; subscript literals `> 100`: 208; jet literals `>= 29`: 38; sharding cases
  `> 150`: 228; backend modules `> 30`: 47).
- **P17:** `sum_over_probes` in `tests/_derivative_data.py`; `chunk_size` in the translation;
  `tests/oracle_sweeps/oracle.py:146`'s "until the old kinds go".
- **P19:** the pages' prose naming deleted code; `test_uniform_curvature.py`'s docstring naming
  `uniform_least_squares_problem`.
- The dense reference forms `ω²` itself and imports no `jets`: keep it so through K1 (part 1).

## Fourth reviewer: prepared for part 1 (2026-11-26, about 01:30 CT)

Reviewer 4 (Claude) takes K1 from part 1 on. Oriented on the briefs, the plan (§1, §2, §4.3, §5, §7 K1, §8), the note,
both design reviews, the prose of part 1 (`prose_edits.md` §6), the prenotes, rulings 22 to 35, part 0's review; traced
`TuckerTensorTrain.probe` -> `probing.t3_probe` -> `sampling_variables` (and the data-space steps of `sampling_steps`
with `jets.data_metric`, `_omega_squared`, `blocks_to_weight`), `probe_corewise_transpose` ->
`t3_probe_corewise_transpose` (the held `Jp` route), `__add__` -> `t3_linalg`; then the reference whole in its
weighted parts and the tests that compare the engine with it. Nothing of the working tree was reviewed.

- **The plan of my review: `k1/REVIEWER4_PLAN.md`** (what part 1's review must establish, five questions on the
  design, the order of the review). Read it first.
- **My tools, in `k1/reviewer_tools/`:** `ref_dense_pins.py` (my independent dense model of the reference's weighted
  variables; `--api old` validates it on any tree before part 1: 1,473 checks pass on `78a34d77`; `--api new` is the
  check of part 1, with the refusals; `--jax`), `ref_independence.py` (static imports, a profile hook over the 37
  functions, the tests' call graph to the reference), `part1_textcheck.py` (the scripted rewrite against the base
  after the intended substitutions; re-flowed prose; the prose of §6 found in the code), `audit_c_attack.py` (14
  planted conversions against the implementer's weights audit), `ref_mutations.py` (a driver; the patches are
  written from the built text). What each showed is in the plan's §1. Reviewer 3's tools above still hold
  (`dense_checks.py` imports the old `jets`: run it before part 5).
- **Things to know:** the reference imports `probing`, `apply`, `entries` for their dense functions only (which import
  the engine module but never call it: the profile hook shows it); the audit's callee key is the bare attribute name
  (`f`, a reason line must say `f`, not `sdef.f`); the audit passes a dropped weight (`W-GONE`): read each; note
  §4.1's signature line is 124 characters.

### Part 1 reviewed (2026-11-26, 01:05 to 01:20 CT)

`K1_part1_REVIEW_r1.md`: SIGN OFF, no blocking or should-fix finding, notes for parts 2 and 5 in its §9. How to rerun
what I ran:
- `PYTHONPATH=<repo> python -B k1/reviewer_tools/ref_dense_pins.py --api new` gives 1,968 checks; `--jax` gives 25.
  `--api old` works on trees before part 1 only.
- `ref_independence.py --api new --tests <files>`.
- `part1_textcheck.py --base 78a34d77 reference|prose|files ...`. Its `NEW_W` is now the built two-line comment.
- `audit_c_attack.py k1/diff_audit.py` gives 14 of 14.
- The audit rerun needs `--scope-map TestTheWeightIsOfTheTypeOfZ=TestTheTwoWeights`.
- `ref_mutations.py <repo> SCRATCH [--tests ...]`. The patches M1 to M11 match the text at part 1; refit them if
  part 2 touches the reference.

For part 2, the reference is the anchor. The engine's refusal of probe modes whose size-1 `W` patterns disagree has
no counterpart in the reference, which accepts them, so test it as a refusal on its own.

### Where things stand (2026-11-26, 01:30 CT, reviewer 4)

- **Part 1:** signed off (`K1_part1_REVIEW_r1.md`) and committed on `k1-wip` as `31e3d100`. Rulings: the reference's
  own tests get a loop placement of their own in part 2, not part 5; the modes' `W` pattern is tested at the engine as
  a refusal on its own.
- **Part 2's prose** (`k1/prose_edits.md` §3) is reviewed in `K1_part2_PROSE_REVIEW_r1.md`: **NOT YET**. Two
  should-fix findings:
  - `jets._metric`'s dtype rule: today `r` is cast to the lowest precision of `r` and the weight; the draft would give
    float64 where today gives float32;
  - the `Rule` rows: use the table's existing continuation form, decided now.

  Four minor findings besides. **Round 2 (`K1_part2_PROSE_REVIEW_r2.md`): SIGN OFF** (01:36 CT). The dtype rule
  was checked against my own model and today's `data_metric` (12 pairings, integers included). The `Rule` rows were
  checked against the real columns (60, 91, 110). Two line re-fills (A, B) are given there for the build to write.
  **Part 2's build starts from this prose when the session resumes.**
- **What part 2's build review must check first:** the list at the end of that review: the dtype; the engine against
  the reference with `K` and coinciding sizes; the refusals cell by cell; junk in the padding (finite); the mixed node
  `test_block_sums::test_engine_and_oracle`; the `Rule` rows.
- **My tools:** `k1/reviewer_tools/` (see "Fourth reviewer" above). `ref_dense_pins.py --api new` is the anchor on the
  reference; the engine side is for part 2's tests. `ref_mutations.py`'s patches are for the reference at `31e3d100`.
- **My context** at this point: about three quarters used. A fresh reviewer may be needed for part 2's build review.

## Context

At the end of part 0's round 1 I have used most of my context (roughly three quarters). I can take part 0's round 2
if it is small; a fresh reviewer should take part 1 onwards, starting from this page.
