NOT YET

# Review of `K0_DESIGN_NOTE.md`, round 1

Reviewer: Claude, 2026-11-25. Tree `b5125f76` (library and tests as at `3375d213`); the note as it stands in
`implementation/`, with `k0/`. My pre-notes, written before I saw the note: `k0/REVIEWER_PRENOTES.md`. Scripts and logs
of this review: `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/reviewer/k0/`.

**The verdict in one paragraph.** The note is careful and right on nearly every point I could check. My own
inventory, built independently, deletes the same 313 names it does; the two it adds are the two dead scan bodies it
decides to delete. On a simulated K0 library, today's tests fail exactly where the note says they will: in the 11
files it edits or deletes, and in the 6 tests it names elsewhere. Within those 11 files my static scans agree with
its per-test lists (§1.2). Every risk in my pre-notes is handled, one of them differently and better. The re-bases I
tested by mutation catch the mutants they should. NOT YET is for three should-fix findings, none of them about the
deletion itself:
- S1: one mutation in the evidence cannot fail;
- S2: the rewritten library prose is left to the build;
- S3: the new permanent test pins export lists that K1 and K5 change.

## 1. What I ran

### 1.1 My own inventory against the note's

`scan.py` is my `ast` scan, written before the note and sharing no code with `k0_inventory.py`. It reads every
module-level statement, imports excluded as references, and closes over private orphans transitively. Its deleted
set: the M1 lists, `sampling_derivatives` whole, and the closure. It gives **313 names**. `k0_names.json` holds
**315**. The difference is exactly `_t3_apply_step` and `_t3_entries_step`, which my scan lists as "no reference at
all" and the note deletes by Decision 2. Nothing in either list is missing from the other. The private helpers the note
keeps are the ones my scan finds referenced by kept statements, `_BATCH_FIELDS` and the oracle's `_take_*` helpers
among them.

### 1.2 The affected tests, by a run

`mysim.py` is my own simulated K0 library: my deleted set cut by `ast` line ranges, `sampling_derivatives.py` removed,
the dead import lines and the `__all__` entries dropped. I ran today's tests on it (`-n 2`, 14 min,
`sim_run.log`):

- **Imports:** 58 modules import with jax and with jax unimportable (`import_all.py`).
- **Collection:** exactly the 11 files the note edits or deletes fail to import: `test_curvature`,
  `test_from_sweep_reuse`, `test_uniform_curvature`, `test_uniform_fitting` (its class tables), the bridge,
  `test_chunk_size_estimator`, `test_dispatch`, `test_jet_recurrence`, `test_probe_derivatives`,
  `test_tucker_tensor_train`, `test_uniform_probing`.
- **In every other file:** 1,518 passed, 3 skipped, 1 xfailed, and **6 failed**. Each of the six is one the note
  names: the two scan floors (`TestVocabularyCompleteness`, `TestInterpreterAnyAxisSharding`, which runs on this
  machine), `TestJetKernelsShardOverW`, `TestTheBridgeToTheOldKinds`, `TestAgainstTheOldKernels`, and
  `test_the_jets_of_entries_off_the_grid` (re-based). No failure the note does not account for.

Inside the 11 files that fail at import, the run cannot tell test from test. There I compared the note's lists with
my static scans: the per-test references, the module- and class-level tables (`tables_scan.py`), the `%s` templates
(`tests_scan.py`). Every unit my scans flag is in the note's §3, deleted, edited or re-based, with a reason. I also
rebuilt the 98 deleted tests from the note's own lists against `baseline_v.log.gz`: **98 nodes, all PASSED** there
(per file: test_curvature 21, test_probe_derivatives 17, test_uniform_curvature 14, chunk estimator 12,
test_uniform_fitting 11, bridge 9, from_sweep_reuse 5, jet_recurrence 5, and one each in contractions_sharding,
dispatch, least_squares_oracle and sampling_definitions).

### 1.3 Experiments and mutations

| What | Result |
|---|---|
| Their E1 and E2, rerun | as the note says: the engine without its exit mask leaves zero padding and the same real block; without the direction masks, the real block of all three kinds' `Kp` is unchanged (0.0) |
| The jvp re-base (my `jvp_rebase.py`, written before the note) | ≤ 3.4e-16 for probe and apply at `order=(2,)` and uniform probe with junk in the direction padding; agrees with E3 |
| Mutant: `H_tilde_dot_step` × (1 + 1e-6) against the re-based jvp rows | caught on the probe rows (~7e-7 vs 1e-12) and on uniform probe; apply unaffected (another step), so mutation (iv) needs a step per kind (§6) |
| Mutant: `start_zeros` without `dtype` (the 2026.2.0 bug class) against the note's re-based `TestJetDtypeUnderX64` op list | promotes `g`, `Kp`, `JtJp` to float64, on numpy and on jax (eager and jit): the re-based list (`tv_probe_transpose`, `t3_probe_corewise_hessian`, `tv_*_transpose`) contains outputs that read `start_zeros`, so the re-based test will catch it |
| The same mutant against the surviving dtype tests (`-k "float32 or dtype or x64 or precision"` over 19 engine, sampling and optimizer test files, `TestJetDtypeUnderX64` included) | **caught by a surviving test**, `test_sampling_stacks.TestTheDtypeMatrix`: two subtests of `test_every_output_in_the_inputs_dtype` (numpy float32, jax float32) and `test_mixed_inputs`. The other dtype tests (26) pass. **My pre-note was wrong**: it said nothing but `TestResultDtype` (`z`, `f` only) would see this bug class after K0. P10's dtype matrix sees it, on numpy and jax. So the re-based `TestJetDtypeUnderX64` is a second guard (x64 on, jit, uniform supercores), not the only one. I still agree with re-basing it: it is cheap and it covers jit and the uniform layout. Dropping it would also be defensible |
| The coverage claim, tested: the same `H_tilde_dot_step` mutant in a scratch library, against tests that survive K0 untouched | caught by `test_sampling_variables.TestByDefinition` (42 subtests, against the dense reference) and by the kept `TestPlainKindsTwoHole.test_against_the_block_sum_oracle` (19). Together with the dtype mutant above, both properties I broke stay pinned after K0 by an independent oracle |
| Mutation (i) of the note, square dropped in `test_residual_weights` | **survives** (1 passed): S1 |
| Doctests of the seven modules against the imports the note removes | none reads a removed module global (an `ast` check of every kept doctest's free names) |
| Scan floors on my simulated tree | `scan_vocabulary` 86, literal sites 115: the note's numbers |
| A parametrization of `TestKpAtTTSlack` over the three kinds | does not run as is (the harness builds probe data): open point 4 |

No test was run on the repository itself; every run was in a scratch copy with `python -B`.

## 2. My pre-notes against the note

| Pre-note point | The note |
|---|---|
| 1a `_plain_ops` mixes engine and old kernels (`test_curvature.py:889-906`) | handled: trimmed to `(two-hole, one-hole)`, the engine tests kept (§3.2) |
| 1a `_PLAIN_UNIFORM`, module-level (`test_uniform_curvature.py:637-642`) | handled: keeps the two-hole; `test_per_element_vs_ragged` kept, its three sweep lines go (§3.3) |
| 1a module-level `sd` imports fail five large files | handled, file by file (§3) |
| 1a `TestCurvatureJvpOracle`: re-base, not delete | handled as I proposed (§3.5); E3 matches my own run (≤ 3.4e-16) |
| 1a `TestJetDtypeUnderX64`: a real bug class, unguarded if deleted | handled: re-based on the engine's faces (Decision 6), mutation (vi) in the evidence. My premise was wrong, though: `test_sampling_stacks.TestTheDtypeMatrix` already catches the mutant (§1.3). The note's §4(d) "re-based here, nothing lost" is right either way |
| 1b `test_tucker_tensor_train` forward: `dd.forward` would be circular | handled better than "not circular": the dense forward (`dense_<kind>` + `taylor_to_derivatives`), E5 |
| 1b `test_probe_derivatives.py:38` is vacuous | handled: the file goes whole, with that reason |
| 1b `test_apply_tt_assembly_is_the_three_term_product_rule` | handled: deleted, the block-sum and Richardson tests named as its cover |
| 1c `o1_*` are labels | handled (§0.2, §3.7): kept, allowlisted |
| 1c `_curvature_oracles.py`: delete the twin family, no re-base | handled (§3.2) |
| 1c `_jet_trs_twins.py` must stay (test_dispatch imports it) | handled **differently**: the file goes and its one surviving function moves into `test_dispatch`. Right: the rest of the file serves deleted tests only. I withdraw "must stay" |
| 1c `api_reference.rst:130`, the README line (O-26) | handled (Decisions 4, 7) |
| 2 second-order orphans (5) | handled: the closure finds all of them, and more (23 in `probing`) |
| 2 `_t3_apply_step`, `_t3_entries_step` dead | handled: deleted (Decision 2) |
| 2 `_entry_xis`, `_onehot_vectors` used by the oracles | handled: they go with the twin family that uses them |
| 2 names built at run time, module and class tables | handled: the inventory now reads both (§1.1); my run confirms the list (§1.2) |
| 2 false positives of a name gate (`_HESSIAN_MODES`, `_init_jet`, absence tests, labels) | handled: names resolved to their module; a short allowlist |
| 2 Sphinx `-W` blind to dangling references | handled: stated (§2.2), library prose corrected in K0 (Decision 10), pages listed for P19. Not mentioned: `api_reference.rst:75-80` already names six factories P16 deleted (finding M2) |
| 2 every module imported with and without jax | handled (§1.4, §6): 58 modules, both ways. My own simulated tree agrees |
| 3 the bridge: no loss | the note agrees (§4(a)) |
| 3 the direction-mask negative control only partly replaced | handled with a measurement I did not have: E2 shows the real-block property does not hold for the engine in general position (I reran E2: relative change 0.0 for all three kinds). Open point 4 |
| 3 `TestTransposeOutputHygiene`: keep (a), (c), (d); (b) against ragged | handled exactly so; non-vacuity replaced (open point 2) |

Nothing on my list is missed.

## 3. Findings

**S1. Should fix. Mutation (i) of §6 cannot fail.** Place: §6, "Mutations", (i): "`test_residual_weights`: drop the
square on the probe weight". The test checks an identity, `hess(w2 r) == d/ds J^T(w2 r) - J^T(w2 J p)`. That identity
holds for any diagonal weight, so dropping the square in the test's `w2` changes both sides alike. Evidence: in a
scratch copy, with `_apply_weight(..., 2)` changed to `..., 1)` in the test, `test_residual_weights` passes (1 passed,
4 subtests). The test still has value: it pins the engine's `Kp` against a finite difference of the engine's
transpose, which is an identity between different variables. But this mutation proves nothing. Resolve: mutate the
library instead, for example scale one term of the probe kind's `Kp` (I did this to `H_tilde_dot_step`, ×(1+1e-6),
and the re-based jvp rows caught it at ~7e-7 against the 1e-12 bar). Or mutate one side only, `got = hess(w2(r) ...)`
against a `curv_fd` taken at another power. And say of each mutation what it is expected to do.

**S2. Should fix. The rewritten library prose is left to the build.** Place: §1.3, "Module docstrings" and
"Comments and docstrings of kept code"; Decision 10. The note says exactly which lines are **deleted** (the comment
blocks with their line ranges). That is reviewable now, and I agree with the list. But for the **rewrites** it gives
only an intent:
- `fitting.py:1-23` "becomes a short docstring of the oracle and its batch";
- `uniform_fitting.py:1-25`;
- the docstrings of `probing`, `apply`, `entries`, `ut3_sampling` and `utv_sampling` are "cut to what stays";
- `ut3_sampling.py:63-69`, `fitting.py:154-160` and `fitting.py:745-749` are "cut to what stays";
- `entries.py:115`, `common.py:526` and `utv_sampling.py:451`.

"Comments are content". These docstrings are the rendered API reference, and K1 rewrites `fitting`'s right after. A
reviewer of the build would then judge new text against nothing agreed. Resolve: an appendix with the new text of
each rewritten docstring or comment, about a dozen and most a sentence or two. The deletions stay as line ranges. Two
cautions for that text:
- `fitting`'s new module docstring should describe only what is there at K0 (the oracle and the batch as they are),
  since K1 moves the kind into the batch.
- The cut of `probing`'s docstring removes the sentence that names it "Reference module for the signature style".
  `docs/contributor/signature_style.md` and `reading_signatures.md` quote `compute_mu`, which goes. The note flags
  those pages for P19 (§2.2), but `probing.py` stays the named reference module. Its remaining functions still carry
  the house signature form, so the sentence can stay true. Say whether it stays.

**S3. Should fix. The new permanent test pins export lists that K1 and K5 change** (§3.8, Decision 5). "`backend.fitting`
and `backend.uniform_fitting` export only the kept names" lists, today, `uniform_fitting`'s `pack_sample` and
`pack_data`:
- K1 deletes `pack_data`, K5 deletes `pack_sample`;
- K1 adds the batch's methods, and K2 moves `pack_vectors`/`unpack_vectors` out of `ut3_operations`.

So the test is edited in at least two later slices for reasons unrelated to what it pins. Its patterns (`compute_*`,
`*_derivatives*`, `*precompute*` over the five sampling modules' `__all__`) also forbid names nobody has proposed yet,
a naming rule hidden in a test. The deletion itself is pinned once, by the gate's inventory (`--check`), and an absent
module cannot come back without a commit that adds a file. Resolve (my preference): keep only the
`find_spec('t3toolbox.backend.sampling_derivatives') is None` assertion, or drop the test. If the export lists stay,
write them as "no deleted name is exported" against the frozen `k0_names.json`, not as an exact list of what remains.
(This changes the expected count by 0 or -1.)

**M1. Minor. `taylor_to_derivatives` in the re-based `test_tucker_tensor_train` forwards** (§3.7). It is shared with
the subject (`_derivative_data._scale`), but harmlessly: the adjoint identity holds for any diagonal factor `s`
(`<J^T(s r), dc>` against `d<r, s J c>`), so a shared error cancels and the subject is still pinned against the dense
forward. Two reasons to write the factorials inline instead (`np.array([math.factorial(t) ...])`, three lines):
- K6 renames `taylor_to_derivatives` to `probe_/apply_taylor_to_derivatives`, so as written the test is edited again
  in K6;
- the oracle then names no library function but `dense_*`.

**M2. Minor, outside the slice: `docs/api_reference.rst:75-80` names the six model factories P16 deleted**
(`entries_model`, ..., `probe_derivatives_model`). None exists today (`hasattr(t3toolbox.fitting, ...)`: none), and
Sphinx is silent because `nitpicky` is off. The note fixes `:130` of the same file under Decision 4 ("an index must not
name a missing" module). The same reasoning covers `:75-80`, but the note does not list it. Either fix it in K0 beside
`:130`, as the same kind of edit, or put it on P19's list explicitly. For the orchestrator: it is a P16 leftover, so
by `BRIEF.md` "write it down, do not fix" unless you rule it into K0.

**M3. Minor. Test comments that name deleted code.** The note corrects `test_contractions_interpreter.py:299`; `:237`
and `:241` of the same table ("the probe-derivative two-hole (sampling_derivatives)", "`_adj_tilde_tangent_step`")
describe frozen history rows of `HISTORICAL`, as `:299` does. Either leave all three as history, which is what the
table is, or correct all three; not one of the three.

**M4. Minor. Non-vacuity, said exactly** (§3.6). E1b measures that the output *changes* when every mask is removed.
The clause it replaces was about (c), that zero padding is not trivially true. Make the negative control assert that
**(c) fails** (padding not zero) without the masks, as `TestKpAtTTSlack` does with `'the padding is not zero'`. Then it
pins the clause it stands for, not only (d).

## 4. The four open points

1. **The scan floors, 100 → 60. Agree.** I measured them on my own simulated tree: `scan_vocabulary` 86, literal
   call sites 115, against 209 and 286 today. The floor's job is to catch a scan that silently finds nothing (a moved
   `LIBRARY_ROOT`, a parse that skips files). It is not a tolerance on what is checked: every string still found is
   still checked against `HISTORICAL`. So lowering it weakens no test of the library. Two remarks:
   - K5 replaces per-mode loops by single array operations and may remove more literals. 60 leaves room; say in the
     comment that the number is a sanity floor, with the count at K0.
   - An alternative that never drifts: assert that a few named literals of kept steps are in the scan (for example
     `sampling_steps.xi_step`'s and `mu_step`'s strings), beside or instead of a count. Either is fine with me.
2. **The non-vacuity clause replaced by a masks-removed negative control. Agree**, with M4. E1 is right: I reran it,
   and the engine's exit mask is not needed on these inputs (padding 0.0, real block bitwise), so no "bare seam with
   junk" exists to show non-vacuity against. A clause that cannot be satisfied must be replaced, not dropped, and a
   negative control on the masks is the replacement that keeps the test honest.
3. **`sum_over_probes` in the test helper, left to P17. Agree.** It is a keyword of a test-only helper, not a deleted
   library name, and P17 rewrites most of its callers (the old-call-form tests). Renaming 60 lines now, in files that
   P17 rewrites, is churn. One condition: put it on P17's list by name (the grep that P20 had is split between K0,
   P17 and P19, plan §7 K0; this word moves from K0's part to P17's), so that P17's final grep is not surprised.
4. **The per-kind direction-mask control deleted, its apply and entries extension left to K5. Agree**, with a
   condition on K5. The property it pinned does not hold for the engine in general position (E2, reproduced: 0.0). At
   TT slack it does, and it is pinned for probe only. I tried the obvious parametrization of `TestKpAtTTSlack` over
   the three kinds (`ttslack_kinds.py`). It does not run as is: the harness's `check()` builds probe-shaped data and
   a probe sample, so apply and entries need their own arguments. It is real work, not a one-line parametrization.
   K5 is the slice that changes the engine's masking (to selection), and that is exactly when this control matters.
   So: **K5's design note adds the apply and entries cases first, against the engine as it then is, before changing
   the masking.** Please put that sentence in the K5 row of the plan or the arc's decisions, so it is not lost.

## 5. The decisions (§7)

| # | Decision | My view |
|---|---|---|
| 1 | The closure decides the private helpers | Agree; my independent closure gives the same set |
| 2 | `_t3_apply_step`, `_t3_entries_step` go | Agree: no reference outside `dev/`, the scan bodies of the pre-engine functions |
| 3 | Unused imports go, including those already unused | Agree. The risk was a module doctest leaning on a module global. I checked every kept doctest of the seven modules against the imports the note removes: none uses one |
| 4 | `api_reference.rst:130` | Agree; and see M2 for `:75-80` |
| 5 | A permanent pin of the deletion | Disagree with its present form: S3 |
| 6 | Re-base where the property is the engine's | Agree; this is the right rule, and the note applies it consistently |
| 7 | The README line (O-26) in K0 | Agree |
| 8 | Two renames | Agree; `tally.py` maps them |
| 9 | `_jet_trs_twins.py` goes, `binomial_combine_tensor` moves | Agree (see §2) |
| 10 | Library prose in K0, pages in P19 | Agree with the split; the rewrites need their text in the note (S2) |

## 6. Order, gate, evidence, counts

- **Order.** Tests first, then the library bottom-up, then the two text files. Sound for a slice that cannot be built
  in runnable parts: every re-based test runs on today's library before the deletion, so a failure there is the test's,
  not the deletion's. I would add one step between 1 and 2: run the edited test files on the untouched library. The
  note says this ("where they must pass"); make it a recorded step with its log, since it is the only point where a
  re-base can be told apart from a deletion side effect.
- **The gate.** The full gate at `-n 2`, the slow tier serially (it includes `test_uniform_curvature.py`, which K0
  edits), `--check`, imports both ways, the per-test tally, mutations. Enough for this size, with S1 fixed. Also:
  - run the slow tier on the edited `test_uniform_curvature.py` before the deletion, too (its two `_SLOW` tests do not
    run in the always-on gate, so a broken edit there shows only in the slow tier);
  - give the jvp rows' mutation (iv) a stated expected outcome. My `H_tilde_dot_step` ×(1+1e-6) gives ~7e-7 against
    1e-12 on the probe rows and nothing on apply, which uses another step. So (iv) should name a step on each kind's
    path, or one per kind.
- **The counts.** 1,979 - 98 + 1 = 1,882. Derived soundly: I rebuilt the 98 from the note's own lists against the
  baseline log, and all 98 nodes are PASSED there. None of the 5 skipped or the xfail is among them, and the two
  `_SLOW` skips stay. With S3 resolved by dropping the new test, the count is 1,881. The subtest arithmetic is right in
  form; each edited test's new count must be stated in the handover, as the note says.
