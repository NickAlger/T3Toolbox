# N1 handoff: cleanup and the ragged start

> **Superseded in part by D-41 (2026-11-10, slice N8b).** The hint-free start's convention changed to Gaussian seeds
> in every store, slot-indexed. The S2-parity record below (380/380 arrays bitwise against S2), the coordinate-seed
> convention, and T5's "zero tensor = `fv_zero_point_frame`" check describe the convention before D-41. They are kept
> as history; see `N8b_handoff.md`.

Slice N1 of `dev/aligned_padded_frame_impl_plan_2026-11-09.md` (§8). Implementer: Claude Opus 5.5, 2026-11-09.
Revised after `N1_review.md` (REVISE; round 2 below). Everything is uncommitted in the working tree; nothing outside
the repository was touched.

## Order of work

1. **Recorded the replaced construction first.** Before anything was deleted, the hint-free outputs of the
   committed `fv_operations.fv_padded_frame` (commit 92640502's file; its hint-free path is 2562264a's, unchanged by
   S3) were computed on 30 cases: every case of the S2 tests (the zero tensor × 5 structures × seeds 0 and 5; the
   stacked zero template; the six resized points; sharing; the three in-block cases; the outside-then-in-block case;
   nesting; the four stacked cases), the deleted dispatch row's two cases, and one extra seed. The construction was
   run from git, never from the working tree.
   - **The session ruled that `.npz` files are never fixtures** (`.gitignore`: "regenerable, never fixtures"), so this
     is a one-time migration check, not a committed test. `dev/aligned_padded_frame_slices/N1_check_s2_equivalence.py`
     runs S2 live from `git show 92640502:...` and the new `fv_padded_frame` on the same 30 cases, and prints the
     bitwise count. It writes nothing.
   - **Result: 30 cases, 380 of 380 output arrays bitwise equal; one Tucker array per group in every shared case.**
     The reviewer reproduced it independently.
   - The script asserts that each S2 template's stores equal `compute_orthogonal_representation_ranks(shape, target
     ranks)`, so every case is expressible in the new signature.
2. **Discarded S4**: `git checkout --` on exactly the four named files. The two untracked files
   (`dev/check_t3svd_retraction_second_order.py`, `release_notes_2026.3.0.md`) were not touched.
3. **Cleanup.** `fv_operations.py` restored to efe5e3d (`git diff efe5e3d -- t3toolbox/backend/fv_operations.py`
   is empty; the index was left untouched). S2/S3 tests and dispatch rows deleted, S1's dispatch row kept. S3's ten
   HISTORICAL strings and their comment removed; S1's two kept. `test_padded_start.py`'s docstring refreshed.
4. **Built** `backend/padded_frames.py`, widened `ranks.oversampled_ranks`, wrote T5, T16 rows, T18, T19.

## Files

| file | change |
|---|---|
| `t3toolbox/backend/padded_frames.py` | **new.** `fv_empty_frame`, `fv_padded_frame` as §5.1 (signatures and shape comments verbatim), the five `ALIGN_*` constants and the `FrameData` alias. Private, module-level: `_frame_core_shapes` (the frame's twin of `t3_operations.t3_core_shapes`), `_old_frame_stores` (stores from the up/left widths plus the layout check), `_haar_seed_rows` and `_coordinate_seed_order` (the host bookkeeping §4.2 names, for N4 to share), `_embed` (`pad_or_truncate`), `_seeded_completion`, `_padded_core`, `_padded_tucker_factor`. Imports `common`, `linalg`, `ranks`, `sharing` and `fv_operations` (`ZERO_FRAME_SEED` only). |
| `t3toolbox/backend/fv_operations.py` | restored to efe5e3d. |
| `t3toolbox/backend/ranks.py` | `oversampled_ranks` widened (§5.4 signature verbatim). Sequences: the unchanged path. Arrays: host numpy, `(d,)+C` / `(d+1,)+C` in and out, lengths checked before the boundaries are spliced, then one call of `compute_minimal_ranks`. Docstring updated; one doctest added. |
| `tests/backend/test_padded_frame.py` | rewritten: T5 (below). The S2/S3 tests are gone. |
| `tests/backend/test_padded_start.py` | docstring refreshed (T19 now, T14 with the seam); T19: `test_arrays_are_per_element_calls` (60 random structures × 3 stacks × p, half with sharing), `test_array_structural_errors`. |
| `tests/test_dispatch.py` | S2's `test_padded_frame_dispatch` and S3's `test_padded_frame_hinted_dispatch` deleted. New T16 rows: `test_empty_frame_dispatch` (numpy by default, `use_jax=True` gives jax, jits with all arguments static; C = () and (2,)); `test_padded_frame_dispatch` (numpy in, numpy out; jax in, jax out; jit with ranks, sharing and seed static equals numpy to 1e-12; the shared empty frame, an in-block nonzero frame, a stack C = (2,)). S1 row's stale comment reworded. |
| `tests/test_contractions_interpreter.py` | S3's ten strings and comment removed; the diff against efe5e3d is now S1's two strings only. |
| `tests/test_import_graph.py` | **new**, T18 (below). |
| `dev/aligned_padded_frame_slices/N1_check_s2_equivalence.py` | the one-time S2 equivalence check (above). |

## The tests

- **T5** (`test_padded_frame.py`, 21 tests, platform-independent):
  - *Zero tensor.* At the empty frame the start equals `fv_zero_point_frame` **bitwise** on all four stores, for
    the five S2 structures × two seeds (sharing included), and per element for a C = (2, 3) stack.
  - *Tucker completion at a nonzero old factor* (new in round 2). Each group's new rows equal numpy's sign-fixed QR
    of `[U0ᵀ | RandomState([seed, min(g)]) rowsᵀ]`, computed in the test directly, not through `pinned_completion`
    (≤ 1e-14; 0.0 here). Unshared and shared `(0, 0, 1)`, seeds 0 and 3; the old rows bitwise; one array per group.
  - *Empty frame.* Its shapes equal those of `t3_orthogonal_representations` of the rank-0 T3, d = 1..4 ×
    C ∈ {(), (2,), (2,3)}; every core is empty, numpy, float64. A start at the zero tensor's own ranks returns it.
  - *Convention.* Valid frames at ten resized points (incl. d = 1, d = 4, the in-block cases and a non-minimal
    target) and from the empty frame. No increase returns the old frame bitwise. Sharing gives one Tucker array per
    group. Outside seeds are exact unit vectors in row-major order (left, right, down). In-block columns are the
    coordinate-seeded completion, ≤ 1e-14, with one in-block column (three cases) and two (`left[1]` and `right[2]`
    of `(4,4,4,4)`, `(1,2,2,2,1) → (1,2,4,2,1)`). Nesting. The seed moves only the Tucker completions. A stack
    equals per-element calls (≤ 1e-14). Sign-gauge covariance (projectors ≤ 1e-12). "Valid" takes the shapes from
    the library's own frame of the zero-padded point (an independent source).
  - *Structural errors*, each isolated (the good input is run, the bad one differs in one core or one argument, and
    the message is matched): five old-frame layout cases and the store-length case; the target below the old frame
    (two); the target's boundary TT ranks (each end alone); target lengths; sharing (unequal Tucker stores in the old
    frame only, in the target only; label count; mode sizes).
  - *Mutation check*, run and the file restored: each of ten mutations of `padded_frames.py` turns T5 red.
    - Checks deleted: the layout check, the below-the-old-frame check, the group-store check, the boundary check.
    - Seed changes: column-major outside order, no group key, reversed inside seeds.
    - The review's three survivors, now red: the stream drawn with an offset (`[s:]`), the key shifted when `s > 0`,
      and in-block seeds 2 and 3 swapped.
- **T16**: the two dispatch rows above.
- **T18** (`tests/test_import_graph.py`): the graph is built with `ast` on module files. Every import statement
  counts, nested ones included. `from t3toolbox import corewise` is an edge to `corewise.py`; implicit
  parent-package imports are not edges.
  - Pinned: the graph reachable from `t3toolbox/backend/` is acyclic; `backend/optimizers.py` imports exactly
    `common`, `regularization`, `corewise`; `padded_frames` (and, once it exists, `uniform_padded_frames`) import
    only their §3.2 / §3.7 sets; the cycle finder finds a planted cycle.
  - The reviewer planted a lazy cycle in a scratch copy, and the test caught it. Today the whole package graph is
    acyclic too.
- **T19**: above, plus the `d = 1` length-1 TT array (round 2).

## Judgment calls

| # | issue | decision | why |
|---|---|---|---|
| 1 | The brief puts HISTORICAL in `backend/contractions.py`. | Edited `tests/test_contractions_interpreter.py`, where the table lives (the plan's §3.1(4) and §8 say so). | `contractions.py` has no table. |
| 2 | Round 1: a committed npz with a bitwise comparison. | **Superseded by the session's ruling** (no binary fixtures): the S2 comparison is a one-time check (380/380 bitwise), and T5 pins the platform-independent facts, including the Tucker stream against an independent QR. | A test reading a gitignored file fails on a clean checkout; a bitwise comparison of QR output with a stored file would also be platform-flaky. |
| 3 | The brief asks the zero tensor to equal `fv_zero_point_frame` bitwise; the plan says Tucker ≤ 1e-14. | Bitwise, all four stores. | Both sides run the same sign-fixed numpy QR on byte-identical seeds in the same process (numpy's QR copies its input into one Fortran-order buffer either way), so the equality is exact on every platform. |
| 4 | The brief allows importing `_haar_rows` / `_identity_prefix` from `fv_operations`; the plan (§3.2, r1 change 29) says no private import. | No private import: every Tucker completion, the zero tensor's included, goes through `linalg.pinned_completion`; the TT/down unit vectors are placed by `_coordinate_seed_order`. | With no pinned rows `pinned_completion` is the same QR as `_haar_rows` and reproduces it bitwise. S2's special branch for `s = 0` is gone with no change in any output. |
| 5 | §3.2 lists `contractions` among the imports. | Not imported in N1. | The hint-free start has no contraction; the import would be unused. N2 adds it with the step. T18's allowed set includes it. |
| 6 | §3.3: the old frame's (down, right) widths must equal those its (up, left) widths imply. | Every core's full shape is compared with the shapes `compute_orthogonal_representation_ranks(shape, up widths, left widths)` gives (`_old_frame_stores`); the first mismatch is named. | One integer comparison covers the (down, right) rule and also inconsistent neighbours, a stray stack axis, a boundary bond ≠ 1, an up width above `N`. |
| 7 | `tt_ranks` boundary ranks ≠ 1. | **Round 2, per the session's ruling:** a structural `ValueError`, after the length check (so a length error keeps its message). | §3.3: "with boundary bonds 1 (structural)". |
| 8 | No template, so where does the dtype come from? | From the old frame, core by core (constants are cast to the core's dtype); the cores are first passed through `xnp.asarray` so a mixed numpy/jax frame comes out all jax. The empty frame is the default float. | §3.3: "dtype and backend" are read off the old frame. |
| 9 | Sharing labels to groups. | `sharing.validate_sharing` (singletons included) rather than `canonical_groups`. | The per-group loop needs every group; `canonical_groups` collapses the unshared case to `()`. Same module, same validation. |
| 10 | `oversampled_ranks` with one sequence and one stacked array; lengths on the array path. | The sequence path needs both to be sequences; otherwise both go through `np.asarray`. Integer dtype (`TypeError`), the lengths (`ValueError`, the sequence path's messages, **checked before the boundary splice**, round 2), and one shared stack (`ValueError`). | §5.4 specifies sequences and arrays, not the mix. Before round 2 a length-1 TT array at d = 1 was silently lengthened by the splice. |
| 11 | §5.1's `VariationsData` alias and the "(D-6)" label. | **Round 2:** the alias is removed (N2 adds it with the step that uses it), and the label is dropped from the comment. | The session's rulings: no plan labels in shipped code; an alias only if referenced. |
| 12 | Where the constants live in `__all__`. | `ALIGN_*` are in `padded_frames.__all__`; the alias is not. | Other modules will import the constants (as `geometry.DEFAULT_CURVATURE_DAMPING`); the alias follows `fv_operations`' unexported `Variations`. |

## Round 2 (review `N1_review.md`, REVISE)

| finding | fix |
|---|---|
| MAJOR 1: T5 loaded the gitignored npz | `TestFrozenReference`, `REFERENCE`, `qr_computed` and the npz are gone, and `tests/data/` is removed. The test docstring is rewritten. The dev script is now the live one-time check: it writes nothing and prints 380/380 bitwise. It was renamed `N1_check_s2_equivalence.py`. |
| MAJOR 2: the Tucker completion was unpinned at a nonzero old factor | `test_tucker_completion_is_the_keyed_stream_completed` checks against an independent numpy QR, unshared and shared, two seeds. Both named mutations are now red. |
| MINOR 3: `tt_ranks` boundary | `ValueError` plus an isolated test for each end. |
| MINOR 4, 5: alias; plan label | Removed. |
| MINOR 6: stale S1 dispatch comment | Reworded: the padded start's in-block completion runs that path under jit. |
| MINOR 7: the array path's lengths | Checked before the splice; the `d = 1` case is in T19. |
| MINOR 8: the script and handoff treated the npz as a fixture | Both rewritten; no `git add -f` / `.gitignore` advice remains. |
| NIT 9: `k_in = 2` | Two cases added (`left[1]`, `right[2]`). The swapped-seeds mutation is now red. |
| NIT 10: N3's planned npz (for the session) | Not N1. |

## In the plan: wrong or ambiguous

- §8 row N1 and the brief: the brief's "`backend/contractions.py`" for HISTORICAL (judgment call 1); the plan is
  right.
- §6 T5 ("vs S2 outputs frozen ... `tests/data/padded_frame_s2_reference.npz`"), T4 and Appendix A assume binary
  fixtures in `tests/data/`. That contradicts `.gitignore` and the session's ruling, and N3's cycle case needs
  another route (review NIT 10).
- §3.2 vs the brief on private imports (call 4); neither is needed.
- §3.3's layout rule names only (down, right); the check is naturally over all four stores (call 6).

## Full suite

`PYTHONPATH=$PWD python -m pytest tests -q -n auto` (env `t3toolbox`: py3.11, numpy 2.x, jax 0.10.2), the complete
`tests/` tree on the round-2 working tree, with no `tests/data/`: **1,217 passed, 4 skipped, 47,834 subtests passed,
0 failed** in 20 min 43 s on 8 workers.
- The one warning is the existing `adam`-on-MANIFOLD `RuntimeWarning` from the oracle sweep.
- Round 1 gave 1,216 passed / 47,856 subtests: round 2 drops the npz test and adds the Tucker-reference and
  boundary tests.
- The module doctests of `padded_frames`, `ranks`, `fv_operations` and `linalg` pass (`pytest --doctest-modules`,
  19 passed).
- Not run: the full `--doctest-modules t3toolbox` and the docs doctests (no docs changed).
