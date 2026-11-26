SIGN OFF

# Review of K0's build, round 1

Reviewer 2 (Claude), 2026-11-25. The working tree as handed over: 32 files changed, uncommitted, `git diff` checksum
`aa1603ea…` at the start and at the end of this review (nothing changed under me). Read against
`K0_DESIGN_NOTE.md`, `k0/prose_edits.md`, `K_ARC_DECISIONS.md` 8 to 16 and my plan, `k0/REVIEWER2_PLAN.md`. The
handover was read last. Tools and logs:
`/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/reviewer2/`. Every run
was in a scratch copy, with `python -B`.

**Verdict.** The deletion is the approved note's and nothing more, and the kept code is untouched where it should be.
The re-based tests have oracles independent of the engine, and each fails for a real bug. The suite's count of 1,882
holds for the final tree. No blocking or should-fix finding. I recommend applying the three minor ones before the
commit; they are a few lines, and I can check them by diff.

## 1. The deletion is the note's and nothing more

- **References to the 315 names** (my `refs.py`, independent of `k0_inventory.py`, over `t3toolbox/`, `tests/`,
  `examples/` and the pages): **0 code hits resolved to a deleted object, 0 unresolved, 0 doctest hits, 0
  kind-templates**. In the library, not even prose is left. Every hit that remains is explained:
  - the strings are the 15 that the implementer's allowlist holds (the o1 labels; `test_jets`' absence tests; the new
    absence test; "PROBE" in an example's heading);
  - the `LOCAL` hits are kept names of the same spelling (`optimizers._HESSIAN_MODES`, a loop variable `_fg`, a
    subtest label `PROBE=`);
  - the comments: `test_contractions_interpreter.py:237`, `:241` are history, left by ruling; `test_dispatch.py:2540`
    is minor finding m3;
  - the pages are P19's (81 prose lines, 12 fenced lines that are not doctests).

  The implementer's `k0_inventory.py --check`, rerun by me, agrees: 0 references, 15 allowed strings. The calls
  with a computed name that could pass silently (`hasattr`, `getattr` with a default, `getattr` on one of the eight
  modules), 28 of them, read by hand: none reaches a deleted name.
- **The mechanical cut against the build** (`sim.py` on `HEAD`, then `unitdiff.py` over the library). Beyond the cut,
  every difference is one of the note's §1.3 items:
  - the seven module docstrings and `probing.__all__`'s comments;
  - three function texts, each exactly its edit of `prose_edits.md`: `entries.t3_entries_ambient_transpose`'s comment,
    `utv_sampling._gauge_masks_over_Knew`'s one word, `common.readonly_mask_copies`'s last line;
  - the imports of Decision 3's two lists, plus the handover's `contractions` in `apply` and `entries`.

  Nothing else in the library changed. `git diff --stat` names only the seven modules, `common.py`, the deleted file,
  `api_reference.rst` and the README.
- **No removed import is still used**, inside functions the tests might not run included. An `ast` check of every
  free name of the eight edited modules against the module's globals found none unresolved. Control: removing
  `probing`'s `contractions` import is flagged at the one use, inside a function.
- **The comment blocks between statements** (`unitdiff.py --comments`, `HEAD` against the build), against note §1.3:
  - every range on the "deleted whole" list is gone, at its line numbers;
  - the rewritten ones carry their new text;
  - `probing.py:274-276` is kept;
  - the banner `probing.py:1236-1238` heads `t3_probe_corewise_hessian`, and the 28 lines under it are gone;
  - no other block is gone (the class-level comments of the deleted classes went with them).
- **Imports** (`imports.py`): 59 modules, with jax and with jax unimportable, no failure.
- **`api_reference.rst`**: 82 roles, 0 unresolved (`roles.py`; 9 unresolved on `HEAD`). The README line is the O-26
  line, as approved.

## 2. Kept code untouched; the prose word for word

- **Every kept unit** of the seven modules and `common.py` (signature, aligned shape comments, docstring, body,
  diagrams) is identical to `HEAD`'s except the three edits above (`unitdiff.py`, by exact text).
- **`prose_check.py`**: all 15 After-texts of `prose_edits.md` are present EXACT, relative indentation and banners
  included. All 15 Before-texts are absent. The `__all__` comments and `utv_sampling.py:86`
  ("(probe, apply, entries)") were checked by diff.
- **`backend/fitting.py`'s module docstring**, round 3's text, checked against the code:
  - `Π` is named as `geometry.project`;
  - "the minibatch scale inside `M`" holds: `_SCALED` holds `g` and `JtJp`, not `Jp`;
  - a numerical check on a weighted probe batch with scale 2.7 gives the oracle's `Jp` equal to `tv_probe`'s
    (relative 0.0), its `g` equal to `tv_probe_transpose` of `γω²r` (1.4e-15), and its `JtJp` equal to that of
    `γω²·tv_probe` (2.7e-16).
- **Nothing describing deleted code is left** in the seven modules. A grep for "polymorphic", "kernel", "wrapper",
  "mask-once", "from_sweep", "old", "P1x/P20", "precompute", "_derivatives", "chunk", "SamplingKind", "sum_over_probes"
  and `K1` to `K7` finds only the three approved texts: `ut3_entries_corewise_hessian`'s "kernel", the "(P14)"
  banner, and "the old order" of arguments. **Nothing anticipates K1 to K7.**

## 3. The tests

- **Deleted tests**: 98, each on the note's lists:
  - 48 in the five files deleted whole (9, 5, 12, 5, 17);
  - in `test_curvature`, 8 + 10 + 3, by class;
  - in `test_uniform_curvature`, 2 + 3 + 3 + 2 + 3 + 1;
  - in `test_uniform_fitting`, 4 + 1 + 4 + 1 + 1;
  - one each in `test_contractions_sharding`, `test_dispatch`, `test_least_squares_oracle` and
    `test_sampling_definitions`.

  Two renames and one added test. The changed test functions are the note's §3, plus the handover's departures.
- **The re-based oracles, read in full.** None calls the engine on the oracle side.
  - The `test_tucker_tensor_train` forwards: `dense_*`, inline factorials.
  - The entries jets off the grid: the subset-sum definition, with `np.take` and `np.tensordot`.
  - `test_residual_weights`: a central difference of the engine's gradient minus the Gauss-Newton term, an identity
    between different engine variables, as the note says.
  - The jvp rows: `jax.jvp`.
  - The dtype test: the dtype.
  - The hygiene test: the ragged face per element, and the masks derived from the ragged shapes.
  - Remark, as in my plan: `affine_jet` and `index_to_one_hot` are shared by the oracle and the subject
    (`_derivative_data.py:43-44`). This is harmless for an adjoint identity at a given jet, and both are pinned on
    their own.
- **Mutations, rerun by me on the built tests**. Each in a fresh copy, checked to differ from the tree only by its
  mutation, and deleted after:

| Mutation | Test | Outcome |
|---|---|---|
| `H_tilde_dot_step` ×(1+1e-3) | `test_residual_weights` | both probe subtests fail (4e-4, 6e-4 against 1e-6) |
| `H_tilde_step` ×(1+1e-3) | the same | all four fail |
| `H_tilde_dot_step` ×(1+1e-6) | `TestCurvatureJvpOracle` | probe, probe at a line, uniform probe at a line fail |
| `H_tilde_step` ×(1+1e-6) | the same | apply, entries, apply at a line, at the one-hot line, and the probe rows fail |
| `H_readout_step` ×(1+1e-6) | the same | apply, entries, apply at a line, at the one-hot line fail |
| `V_top_step` ×(1+1e-6) | the same | passes: on both sides of the identity, the note's stated survivor (pinned by `test_sampling_apply.TestByDefinition`) |
| `start_zeros` without `dtype` | `TestJetDtypeUnderX64` | 21 of 30 fail (every op whose rows start from zeros, numpy and jax, eager and jit) |
| `H_tilde_step` ×(1+1e-3) (probe's `g`) | `test_derivative_methods` | fails (9.2e-4 against 1e-5) |
| `H_readout_step` ×(1+1e-3) (apply's `g`) | `test_corewise_derivative_transpose_nontrivial_boundary` | apply and entries fail |
| `dense_apply`'s slot 1 ×(1+1e-6) | `test_the_jets_of_entries_off_the_grid` | 12 of 12 fail |
| the negative control without the `_UNIFORM_MASKS` patch | `TestTransposeOutputHygiene` | 264 of 264 controls fail ("AssertionError not raised") |
| without the `_mask_variable` patch | the same | 264 of 264 fail |
| without the `ufv_apply_variations_masks` patch | the same | the 168 tangent controls fail; the 96 corewise ones pass (they have no such re-mask) |
| `sampling_derivatives.py` put back | `test_the_old_sampling_kernels_are_gone` | fails |

  So each masking layer alone zeroes the padding, and the control removes exactly the layers there are. No patch is
  idle where it applies, and the control cannot pass vacuously.
- **Unchanged tests reached through an edited helper** (`helper_users.py`: same file, class-scoped, or an imported
  helper module): 20 of them.
  - 5 in `test_curvature` (through `_plain_ops`, `_plain_problem`, `_plain_blocksum`);
  - `TestTheEngineShardsOverW` (through `setUpClass`, `_check`);
  - 12 through `_derivative_data`'s two transposes;
  - the two o1 sweeps.

  Every one has the same outcome and subtest count in `baseline_v.log.gz` and `k0_suite_v.log.gz`. The o1 sweeps,
  for example, have 6,084 and 6,706 subtests in both. The helpers' edits are the dropped tuple slots and the unread
  `chunk_size`. **`tests/oracle_sweeps/` is not in the diff.**
- **The floor of CI**: the 22 changed Python files parse with `ast.parse(..., feature_version=(3, 9))`; a `match`
  control is rejected. No numpy-2-only call is in the added lines. The moved `_check`, which uses `jax.tree`, is
  pre-existing code that ran on the floor before.

## 4. The tally, and the count 1,882

- **`tally2.py`** on `baseline_v.log.gz` and `k0_suite_v.log.gz` (my parser stops at the report sections, which echo
  captured subtest lines): the totals are pytest's own, 1,881 passed, 1 failed, 5 skipped, 1 xfailed, 90,134
  subtests. The differences:
  - 98 deleted;
  - 2 renames, equal records (1 passed; 2 subtests);
  - 1 added;
  - 4 edited tests with new counts: the hygiene pair 336 → 168 and 192 → 96, the dtype test 22 → 30, the sharding
    sweep 930 → 228;
  - 3 tests unchanged in text whose subtests are one per library item: the call sites 286 → 113, and the backend
    modules 48 → 47 twice.

  With the renames paired, that is the handover's 106, each accounted for. Every other node is equal, including the
  15 edited tests whose counts did not move.
- **The log is of the final tree but one line.** The library was last written at 18:34:21, and the run began at
  about 18:34:40 (its end at 19:09:03 less 2,063.66 s). The only file written after it is
  `tests/test_contractions_sharding.py`, the floor fix. **I ran that file on the final tree: 4 passed, 476
  subtests.** So 1,882 passed holds, and I judge a second full run unnecessary. My own runs of the final tree also
  pass: the eight re-based tests and classes (10 passed, 328 subtests).

## 5. The departures (handover §4)

**The third floor** (`test_contractions_sharding.py:137`, `checked > 150`). **It is the same kind as the two ruled
ones; 150 is sensible.**
- `checked` counts 2 cases per group of each scanned string, and the loop skips nothing. So it is a deterministic
  function of the vocabulary: a guard that the sweep ran, not a tolerance on what it checks.
- Every scanned string has a group (`_scan_vocabulary` keeps only those), so `checked ≥ 2 × len(vocab) ≥ 122`
  whenever the vocabulary floor holds. Its own job is to catch a loop body that stops counting.
- 150 is 66% of today's 228, as 60 is 71% of the 84 strings. At the vocabulary floor, with today's 2.7 cases per
  string, the sweep would give about 165.

**A fourth floor** I searched for every count assertion against a literal in `tests/`. The ones that scan the library,
measured on the built tree with the test modules' own functions:

| Floor | Value at K0 | Note |
|---|---|---|
| vocabulary `> 60` | 84 | |
| call sites `> 90` | 113 | the closest |
| subscript literals `> 100` | 208 | |
| jet literals with the case table `>= 29` | 38 | |
| the sharding sweep `> 150` | 228 | |
| backend modules `> 30` (two tests) | 47 | |

None trips at K0. The others (`test_jet_contract`'s case-table counts, `test_least_squares_oracle`'s variants) do not
depend on the amount of library code. **For K5's note**: `call sites > 90` at 113 is the floor K5 is likeliest to
reach, since it replaces the per-mode loops by single operations.

**The other departures, each checked:**
- `setUpClass` and `_check` are moved verbatim (a line-by-line `diff` against the deleted class: identical).
- The two `contractions` imports were read only by the dead scan bodies.
- `_derivative_data`'s `chunk_size` was unread, and no caller passes it (the reached tests are unchanged).
- The "Six facts" docstring now matches its six bullets.

## 6. The gate

Rerun by me on the final tree:
- imports both ways;
- **the module doctests: 320 passed** (325 collected on `HEAD`: the note's five fewer);
- **the page doctests: 47 pages, 0 failures**;
- the 3.9 syntax;
- the sharding file and the re-based tests above.

**Not rerun**, and why:
- **Sphinx**: K0 changes no page but `api_reference.rst`, whose roles all resolve, and `-W` cannot see roles anyway.
- **The examples**: no example names a deleted name, and the kept library code is textually identical.
- **The slow tier**: `test_uniform_curvature.py`'s edits are the table and the three sweep lines, checked by text.
  The handover's slow run (52 passed) and its pre-deletion one (45 passed) are on record, and its `_SLOW` tests read
  only kept code.

I take the handover's records of these three.

## 7. Findings

**m1. Minor. The two new floor comments state a count that is not the tree's.**
- Place: `tests/test_contractions_interpreter.py:551-552` and `tests/test_contractions_sharding.py:102-103`: "86
  subscripts at K0".
- Evidence: the test modules' own scans on the built tree give 84; both `scan_vocabulary` and the sharding file's
  `_scan_vocabulary` say so. The note's 86 was measured on a simulated tree that still held the two dead scan bodies
  of Decision 2.
- A comment that states a measured count is content.
- Resolve: "84".

**m2. Minor. A new docstring line of 170 columns.**
- Place: `tests/backend/test_uniform_curvature.py`, the module docstring's line
  "``ut3_{probe,apply,entries}_corewise_hessian``; the uniform geometries' exact members ...". It is the only added
  line over 120 columns in the diff; the long line in `test_contractions_sharding.py`'s class docstring is pre-existing
  text.
- Resolve: rewrap to the house width.

**m3. Minor. A comment in an unchanged test still describes the deleted kernels.**
- Place: `tests/test_dispatch.py:2538-2540`, in `test_jit_uniform_derivatives`: "the d-prefixed JET contractions + the
  scan sweeps, the binomial tensor + masks stay host constants, build_input_jets vectorizes over d (no unroll),
  pack/unpack slice with static shapes".
- Evidence: the test now jits the frontend's merged methods through the engine. `build_input_jets` is one of the 315
  names (`refs.py`, COMMENT), and no binomial tensor is contracted anywhere. Neither inventory lists comments in
  tests, so the build had no reason to see it. It has been stale since P16, but K0 corrected test prose of the same
  kind (`test_jets`' docstring).
- Resolve: one line saying what is jitted (the merged methods at a line through the engine, `K`-stacked tangents),
  or leave it and put it on P19's list by name.

Not findings:
- `TestTheEngineShardsOverW` has no `skipUnless(jax)`; it never had one (the handover's note).
- The handover's "378 insertions" is 381 by `git diff --stat` today.
- The new absence test's failure message prints the whole module map (correct, only noisy).

## 8. What I ran

All in scratch copies with `python -B`, one heavy run at a time:
- the static tools: `refs.py`, `sim.py` + `unitdiff.py`, `prose_check.py`, `helper_users.py`, `tally2.py`, the
  free-name check, `roles.py`, `doctest_count.py`, the floor measurements, the 3.9 parse;
- the oracle check of the `fitting` docstring;
- the mutations of §3, with `-n 2`;
- the final-tree runs: the re-based tests, `test_contractions_sharding.py`, the module doctests and the page doctests;
- `k0_inventory.py --check`.

No test was run in the repository, the tree is as I found it apart from this file, and nothing is left running.
