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

## Context

At the end of part 0's round 1 I have used most of my context (roughly three quarters). I can take part 0's round 2
if it is small; a fresh reviewer should take part 1 onwards, starting from this page.
