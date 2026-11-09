# N1 review: cleanup and the ragged start

Reviewer: Claude Opus 5.5, 2026-11-09. Reviewed the uncommitted working tree against
`dev/aligned_padded_frame_impl_plan_2026-11-09.md` (§3.1–3.3, §4.2, §5.1, §5.4, §6 T5/T16/T18/T19, §8 N1, §12), the
implementer's `N1_handoff.md`, and the session's rulings (no binary fixtures; the `tt_ranks` boundary check; no plan
labels in shipped code; `VariationsData` only if referenced; the stale S1 dispatch comment).

## Summary

The library code is good. `padded_frames.py` follows the gold standard: pure functions on tuples of arrays, ints and
static int tuples; structure as `Sequence[int]` with `len=` comments; §5.1's signatures verbatim; one-line docstring
on the twinned function, a doctest on the untwinned `fv_empty_frame` (§3.6); no contraction at all, so no raw einsum
and no HISTORICAL rows needed; downward imports only (`common`, `linalg`, `ranks`, `sharing`, `fv_operations` for
`ZERO_FRAME_SEED`); exported through `__all__` and picked up by autoapi like every backend module. The old frame is
read for its shapes only because it is used (its cores are the pinned blocks, §3.3). The construction equals S2's
bitwise on all 380 arrays of the 30 frozen cases, run live from commit 92640502 on this machine. The cleanup is
complete. `oversampled_ranks` is widened per §5.4. T18 detects a planted cycle.

What blocks the commit is the test file. It still loads the gitignored npz, so T5 would fail on any clean checkout,
including CI. Removing that test, as the ruling requires, exposes a coverage gap: two mutations of the Tucker
completion at a nonzero old factor survive the rest of T5. The ruling's points (the boundary check, the plan label,
the alias, the stale comment) are not yet applied.

## Findings

### 1. MAJOR: T5 loads the gitignored npz, so the committed test fails on a clean checkout

`tests/backend/test_padded_frame.py:40` (`REFERENCE`), `:135–167` (`TestFrozenReference`), and the docstring at
`:8–13`.

**Problem.** `tests/data/padded_frame_s2_reference.npz` is ignored by `.gitignore:159` (`*.npz`, "regenerable, never
fixtures"). It is the only file in `tests/data/`, and nothing there is tracked. If the slice is committed as it
stands, `np.load(REFERENCE)` raises `FileNotFoundError` in CI and on every other machine. The session's ruling also
makes the bitwise comparison with S2 a one-time migration check, not a committed test.

**Fix.**
- Delete `TestFrozenReference`, the `REFERENCE` constant, `import pathlib`, and `qr_computed`, which only that test
  uses. `inside_count` stays: `test_in_block_columns_come_from_coordinate_seeds` uses it.
- Rewrite the module docstring's first bullet. T5 now asserts the platform-independent facts only (see finding 2).
- Delete the untracked `tests/data/` directory.
- Finding 8 covers the dev script and the handoff.

### 2. MAJOR: once the npz test is gone, T5 does not pin the Tucker completion at a nonzero old factor

`tests/backend/test_padded_frame.py`, `TestConvention` (`:283` ff.).

**Problem.** The ruling requires T5 to assert this structural fact of §4.2: the Tucker completion is the keyed Haar
stream `RandomState([seed, min(g)])`, made orthonormal against the old factor. The remaining tests check it only at
the zero tensor, where `s = 0`, through `fv_zero_point_frame`. At a nonzero old factor they check nesting and that
the seed has some effect, but nothing pins *which* stream rows are used or under which key.

I ran a mutation check on a scratch copy of the package with `TestFrozenReference` removed. Both of the following
leave the file green (19 passed):
- draw the stream with an offset, `_haar_seed_rows(n, N, key, seed)[s:]`;
- use the key `key + 1` only when `s > 0`.

The handoff's six mutations are all still caught without the npz test (layout check, below-the-old-frame check,
group-store check, column-major outside order, group key, reversed inside seeds).

**Fix.** Add a test against an independent reference, at a nonzero old frame, unshared and shared (`(0, 0, 1)`), with
two seeds. For each group `g` with first mode `i`:
- `s = U0.shape[0]`;
- `G = np.random.RandomState([seed, i]).standard_normal((n - s, N))`;
- `Q, R = np.linalg.qr(np.concatenate([U0.T, G.T], axis=1))`;
- `ref = (Q[:, s:] * np.where(np.diag(R)[s:] >= 0, 1.0, -1.0)).T`;
- assert `up[i][s:]` equals `ref` to ≤ 1e-14.

This reference is numpy's QR directly, not `pinned_completion`. I checked it on `(5,6,7)`, `(2,2,2)→(3,4,3)`, seed 3:
the difference is 0.0 here. Re-run the two mutations above and confirm they turn the file red.

### 3. MINOR: no `tt_ranks` boundary check (ruling; also §3.3, "with boundary bonds 1 (structural)")

`t3toolbox/backend/padded_frames.py:244`.

**Problem.** `compute_orthogonal_representation_ranks` silently maps any boundary rank to 1. For example,
`fv_padded_frame(fv_empty_frame((5,6,7)), (2,3,2), (2,2,2,3))` is accepted (verified). Handoff judgment call 7 left
this open; the session ruled that the check should be there.

**Fix.**
- After `new_stores` is computed, so that a length error keeps its own message, add
  `if tt_ranks[0] != 1 or tt_ranks[-1] != 1: raise ValueError(...)`, with a message that names the boundary ranks.
- Add an isolated case to `TestStructuralErrors` (the good target with only `tt_ranks[0]` changed, and one with only
  `tt_ranks[-1]` changed).

### 4. MINOR: unused alias `VariationsData`

`t3toolbox/backend/padded_frames.py:32–33`.

**Problem.** No shape comment or docstring in the module references it.

**Fix.** Delete both lines. N2 adds them back with `tv_align_padded_frame`, whose signature uses the alias.

### 5. MINOR: a plan label in shipped code

`t3toolbox/backend/padded_frames.py:35`, the comment `# the loops' defaults (D-6), named once; ...`.

**Fix.** Drop `(D-6)`. For example: `# the alignment loops' defaults, named once; the geometry members and the frontend
twins reference them`.

### 6. MINOR: stale comment in S1's dispatch row (ruling)

`tests/test_dispatch.py:165`, "(S3's jitted constructor runs this path)".

**Problem.** S3's constructor is gone. The substance still holds: `fv_padded_frame` runs `pinned_completion` under
jit, and its in-block coordinate seeds can lie in the span of the old columns.

**Fix.** Reword the comment, for example: `# "valid for any seeds" on jax's own QR (the padded start's in-block
completion runs this path under jit: its coordinate seeds can lie in span(P)): ...`.

### 7. MINOR: the array path of `oversampled_ranks` does not check lengths before splicing the boundaries

`t3toolbox/backend/ranks.py:265–277`.

**Problem.** `plus_tt` rebuilds the boundary entries before `compute_minimal_ranks` checks the lengths. A length-1
`tt_ranks` array therefore becomes length 2. At `d = 1` this passes silently:
`oversampled_ranks((7,), np.array([[2, 3]]), np.array([[1, 1]]), 1)` returns a `(2, 2)` TT array for a `(1, 2)`
input. The sequence path raises `len(tt_ranks) = 1 != d + 1 = 2` on the same input. Verified both.

**Fix.**
- After the dtype check, check `tucker_ranks.shape[:1] == (d,)` and `tt_ranks.shape[:1] == (d + 1,)`, with the
  sequence path's messages.
- Add the `d = 1` case to `test_array_structural_errors`. Its existing `tt[:3]` case at `d = 3` cannot see this.

### 8. MINOR: the dev script and the handoff still treat the npz as a fixture to commit

`dev/aligned_padded_frame_slices/N1_freeze_s2_reference.py`, `N1_handoff.md` ("For the session": `git add -f`, or a
`.gitignore` exception).

**Fix.**
- Make the script the migration check itself: run S2 from `git show 92640502:...` and the new `fv_padded_frame` on
  the 30 cases, and print the bitwise count (380/380 here). Do not write into `tests/`.
- In the handoff, record the result and remove the `git add -f` / `.gitignore` advice.
- My scratch script, `compare_s2.py` (below), already does this, so the script can import `CASES` and compare
  directly.

### 9. NIT: T5 checks the in-block completion only with one in-block column

`tests/backend/test_padded_frame.py:314–337`.

**Problem.** Every case has `k_in = 1`, so only the first coordinate seed is pinned. A mutation that swaps the 2nd and
3rd in-block seeds survives.

**Fix.** Add a `k_in = 2` case: `(4,4,4,4)`, old `(2,2,2,2)/(1,2,2,2,1)`, target `(2,2,2,2)/(1,2,4,2,1)`. There
`left[1]` goes from `(2,2,2)` to `(2,2,4)` and `right[2]` from `(2,2,2)` to `(4,2,2)` (verified). Compare both
completion columns with `in_block_columns(A0, 2)`.

### 10. NIT, for the session and not N1: N3's planned binary fixture

Plan §6 T4, §8 row N3 and Appendix A put the cycle regression in `tests/data/padded_frame_cycle_case.npz`. That
collides with the no-binary-fixtures ruling, and `*.npz` is ignored anyway. Decide before N3 how T4 gets its data,
for example by regenerating the case in the test from its seed and structure, or by storing the few arrays as a
small text or Python literal. Similarly, `docs/api_reference.rst`'s hand-written backend list will need
`padded_frames` in N9.

## Verified and not findings

- **Signatures.** `fv_empty_frame`, `fv_padded_frame` and the `ALIGN_*` constants match §5.1 line by line, as does
  `oversampled_ranks` against §5.4.
- **Backend style.** Private helpers have module-level, annotated, one-line docstrings. No result classes, no dict of
  shapes, no template object. `None` appears only internally (`factors`), never for the zero tensor.
- **No private import from `fv_operations`** (judgment call 4). This follows §3.2 and is fine. `pinned_completion`
  with `s = 0` reproduces `_haar_rows` bitwise, because numpy's QR copies its input to the same Fortran buffer and
  the sign rule multiplies by ±1, which is exact.
- **`validate_sharing` rather than `canonical_groups`** (call 9). Groups come back ascending, so `group[0] = min(g)`,
  the plan's key.
- **Stacks (C).** One batched QR. Per-element equality is tested to 1e-14, and per element against
  `fv_zero_point_frame` bitwise at the zero tensor.
- **Sharing labels.** Arbitrary hashables work (`('a','a','b')` is tested). One array per group, checked with `is`.
- **Cleanup.**
  - `git diff efe5e3d -- t3toolbox/backend/fv_operations.py` is empty.
  - The grep for `_frame_stores`, `_outside_first`, `_empty_old_frame`, `tv_to_t3(template`, `hinted` and
    `fv_operations.fv_padded_frame` in `t3toolbox/ tests/ docs/ CHANGELOG.md` finds only the new private
    `_old_frame_stores` and the test docstring's provenance line, which goes with finding 1.
  - `git diff efe5e3d -- tests/test_contractions_interpreter.py` is S1's two strings only.
  - S1's dispatch row (`test_pinned_primitives_dispatch`) and `tests/backend/test_linalg.py` are intact.
  - `geometry.py` is unmodified (S4 discarded).
  - No docs or CHANGELOG change since efe5e3d.
- **T16.** `test_empty_frame_dispatch` and `test_padded_frame_dispatch` cover numpy in → numpy out, jax in → jax
  out, jit with static ranks, sharing and seed equal to numpy at 1e-12, the shared empty frame, the in-block case,
  and C = (2,).
- **T18.** It builds the graph on module files with `ast` and counts lazy imports. I planted
  `import t3toolbox.backend.padded_frames` inside a function of `fv_operations.py` in a scratch copy of the package.
  `test_the_backend_graph_is_acyclic` went red with the cycle
  `['…fv_operations', '…padded_frames', '…fv_operations']`, and the baseline copy was green. The repository was not
  touched.
- **T19.** Sixty random structures × three stacks × three values of `p`, half with sharing, each equal to the
  per-element sequence calls. The structural errors are isolated.

## What I ran (env `t3toolbox`, py3.11, numpy 2.x, jax 0.10.2; repo root, `PYTHONPATH=$PWD`)

1. **S2 equivalence, live.** A scratch script (`compare_s2.py`) imports the freeze script's `CASES`, runs
   `fv_padded_frame` of commit 92640502 from git and the new `padded_frames.fv_padded_frame` (with `fv_empty_frame`
   for the zero cases), and compares them. Result: **30 cases, 380 arrays, 380 bitwise equal to the live S2 output
   and 380 bitwise equal to the npz**; worst difference 0.0; one Tucker array per group in every shared case.
2. **Doctests.** `pytest --doctest-modules` on `padded_frames.py`, `ranks.py`, `fv_operations.py` and `linalg.py`:
   **19 passed**.
3. **Targeted tests.** `pytest tests/backend/test_padded_frame.py tests/backend/test_padded_start.py
   tests/test_dispatch.py tests/test_contractions_interpreter.py tests/test_import_graph.py
   tests/backend/test_linalg.py -q -n 4`: **126 passed, 4,237 subtests passed, 0 failed** (8 min 41 s). The npz is
   still present on this machine, so `TestFrozenReference` passes here; that is finding 1.
4. **T18 planted cycle.** A scratch copy of `t3toolbox/` with the lazy cycle `fv_operations → padded_frames`: red, as
   above. The unmodified copy: 5 passed.
5. **Mutation run.** A scratch copy with `TestFrozenReference` removed, nine mutations of `padded_frames.py`: six
   red (the handoff's six), three green (finding 2's two, finding 9's one).
6. **Probes** (`python -`):
   - the boundary-rank target is accepted (finding 3);
   - `oversampled_ranks` accepts a `d = 1` length-1 TT array (finding 7);
   - list inputs give tuples out;
   - a float32 old frame gives float32 out;
   - a partially-zero target from the empty frame is structurally consistent;
   - the independent Tucker reference of finding 2 matches to 0.0.

The full suite was not run. The verdict is REVISE, and the implementer reported it green (1,216 passed, 4 skipped),
so the session should rerun it on the revised tree.

## Counts

MAJOR 2, MINOR 6, NIT 2.


## Round 2 (re-check of the implementer's fixes, 2026-11-09)

I re-checked the diff item by item:

- **Finding 1, fixed.** `TestFrozenReference`, `REFERENCE`, `pathlib` and `qr_computed` are gone. No `npz` appears
  anywhere in the test file. `tests/data/` does not exist. The docstring cites the one-time check instead.
- **Finding 2, fixed.** `test_tucker_completion_is_the_keyed_stream_completed` checks unshared and `(0, 0, 1)`, seeds
  0 and 3. It compares against numpy's sign-fixed QR of `[U0ᵀ | Gᵀ]` directly, checks the old rows bitwise, and
  checks one array per group with `is`.
- **Finding 3, fixed.** The boundary `ValueError` (`padded_frames.py:243`) comes after the length checks. It has an
  isolated test, `test_target_boundary_ranks`, covering each boundary.
- **Findings 4 and 5, fixed.** `VariationsData` is removed and `(D-6)` dropped. No plan label is left in the module.
- **Finding 6, fixed.** The comment at `test_dispatch.py:165` was reworded.
- **Finding 7, fixed.** The array path checks lengths before splicing the boundaries. The `d = 1` case is tested
  (`test_padded_start.py:169`).
- **Finding 8, fixed.** `N1_check_s2_equivalence.py` writes nothing. The handoff no longer gives `git add -f` advice.
- **Finding 9, fixed.** Two `k_in = 2` cases were added (left and right).
- **Unchanged:** `git diff efe5e3d -- t3toolbox/backend/fv_operations.py` is still empty.

What I ran:

- `N1_check_s2_equivalence.py`: **30 cases, 380/380 bitwise**.
- `test_padded_frame.py`, `test_padded_start.py`, `test_import_graph.py` and `test_contractions_interpreter.py`:
  **49 passed, 3,600 subtests**.
- The `test_dispatch.py` rows `-k "padded_frame or empty_frame or pinned"`: **3 passed**.
- Doctests of `padded_frames.py` and `ranks.py`: **2 passed**.
- Mutation re-run against the revised T5, on a scratch copy: **all 10 mutations red**. These are the six from the
  handoff, my three survivors from round 1, and deleting the boundary check.

I did not run the full suite myself; the implementer reports 1,217 passed, 4 skipped, 0 failed.

New findings: none. Note for the session when staging: `dev/aligned_padded_frame_slices/__pycache__/` exists. It is
gitignored, but stage by name anyway.

VERDICT: APPROVE
