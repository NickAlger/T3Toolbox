# K0 build review: what it must establish (the second reviewer's plan)

Claude (reviewer 2), 2026-11-25, written during the orientation, before the handover and before reading the built
tree. Starts from `K_REVIEWER_CARRYOVER.md` ("What to check in K0's build review") and adds what a fresh reading of
`K0_DESIGN_NOTE.md`, `k0/prose_edits.md` and the plan's K0 row suggests. Every check is mine to run or read; where a
tool decides it, the tool is named. Tools (written by me, each tested on `HEAD` with a planted positive and negative
case): `/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a03b046-9177-4c43-923a-b31595edb27e/scratchpad/reviewer2/`.

## Tools

| Tool | What it shows | Tested on HEAD |
|---|---|---|
| `refs.py ROOT --docs` | every occurrence of the 315 frozen names (and the module name) in `t3toolbox/`, `tests/`, `examples/`, the pages: code hits resolved by `ast` to the owning module (RESOLVED / OTHER-MODULE / LOCAL / UNRESOLVED), doctest, string, docstring, comment; run-time names (`getattr`/`hasattr` with a computed name on an owner module or with a silent default; `%`, `format` and f-string templates read with each placeholder as a kind) | finds the two run-time cases of the note (`test_uniform_curvature.py:727-728`, `test_uniform_probing.py:913-914`); on my simulated K0 library: 0 code and 0 doctest hits, the prose residue = exactly the texts the note rewrites |
| `sim.py SRC OUT` | the mechanical cut alone (frozen names, their imports, their `__all__` lines) | asserts every frozen name is a top-level binding of its module; 59 modules import |
| `unitdiff.py OLD NEW [--diff] [--comments]` | every unit (module docstring, top-level or class-level statement, method; decorators, comments and blank lines inside included) identical / changed / deleted / added, with diffs; comment blocks between units kept / gone / new | a shifted signature column, a reworded comment block and a changed diagram line in a copy: all three caught |
| `prose_check.py ROOT` | each "After" block of `prose_edits.md` present EXACT (relative indentation kept) in its file; each "Before" absent | HEAD: every before present, every after absent; a hand-applied copy: EXACT; one space of indentation off: CONTENT, flagged |
| `imports.py ROOT [nojax]` | every module imported, jax present or hidden; asserts the package imported is ROOT's | 60 modules (HEAD), 59 (simulated); a planted import of the deleted module and an unconditional `import jax` both fail loudly |
| `roles.py PAGE` | every `:py:` role resolved by import and `getattr` (Sphinx `-W` is blind here) | `api_reference.rst`: the note's 9 on HEAD, 10 on the simulated tree |
| `tally2.py OLDLOG NEWLOG OLDROOT NEWROOT` | per node outcome and subtest counts, each difference labelled by the static status of the test's own text; a difference on an unchanged test is flagged | baseline against itself: 1,979 / 5 / 1 / 93,034, pytest's own totals; a doctored log: the removed node, the CHANGED test and the unchanged one each reported in their group |
| `helper_users.py OLD NEW` | unchanged tests reached through a changed or deleted helper (same file, class-scoped, or an imported helper module), transitively | 9 s on the whole suite |
| `doctest_count.py ROOT` | docstrings with examples (`DocTestFinder`) | 308 on HEAD, 303 on the simulated tree: the note's "5 fewer" |
| `mut.py copy / step / patch / same` | scratch copies; a step scaled by `1 + eps` (rebinding the module attribute the engine looks up); exact single-occurrence patches; a check that nothing else differs from the source tree | the `H_tilde_dot_step` mutant changes the engine's probe `Kp` on a copy and not on HEAD |

## What the review must establish

**A. The deletion is complete and is only the deletion.**

1. `refs.py` on the built tree: no RESOLVED, UNRESOLVED or DOCTEST hit; each string, docstring and comment hit
   explained (the allowlisted labels of `o1_*`, the absence tests of `test_jets`, P19's pages, P17's
   `sum_over_probes`); the run-time list read by hand. `k0_inventory.py --check` gives 0 (the implementer's, run
   beside mine; the two must agree).
2. `sim.py` on `HEAD`, then `unitdiff.py sim BUILT` over the library: every difference beyond the mechanical cut is
   one of note §1.3's items (a prose text of `prose_edits.md`, a comment range of the "deleted whole" list, an
   import of Decision 3's two lists). Nothing else changed, in particular no kept function's text (its signature
   columns, comments, diagrams). `git diff --stat` of the library, docs and examples names only the seven modules,
   `common.py`, `api_reference.rst` and the deleted file.
3. The comment ledger of `unitdiff.py --comments` against note §1.3: each block on the "deleted whole" list gone, at
   its range; each of the five section headers kept or rewritten; no other block gone.
4. `prose_check.py`: every After EXACT, every Before ABSENT; `fitting`'s docstring carries r3's M1 wording.
5. `imports.py` both ways; `roles.py` on `api_reference.rst`: 0 unresolved; the README line (O-26).

**B. The tests: what went, what changed, and that the changes are the note's.**

6. `unitdiff.py HEAD BUILT tests/`: the deleted, changed and added test functions and helpers equal note §3's lists,
   file by file (98 deleted, 2 renamed, 1 added; the edited and re-based ones by name).
7. `helper_users.py`: every unchanged test reached through a changed helper is explained. Already visible on the
   mid-build tree: 5 in `test_curvature` (`_plain_ops`, `_plain_problem`), about 14 through `_derivative_data`'s
   `tangent_transpose` and `corewise_transpose` (the note drops their unused `chunk_size`: confirm nothing else
   changed and no caller passes it), `TestTheEngineShardsOverW` (its `setUpClass`, `_check`), and the two o1 sweeps,
   whose files `tests/oracle_sweeps/README.md` asks to keep byte-close to the archive: confirm the o1 files are not
   edited and what reaches them is only that parameter.
8. Each re-based oracle traced to its imports: no engine function and no deleted code on the oracle side (dense
   reference, jax AD, inline factorials, `itertools` subsets). Remark, not a finding: the `test_tucker_tensor_train`
   forward also calls `affine_jet` and `t3_to_dense`, which the subject shares (`_derivative_data.py:44`); harmless
   for an adjoint identity at a given jet, but the note's "names no library function but `dense_*`" is then not
   literal.
9. The negative control of `TestTransposeOutputHygiene` patches `_UNIFORM_MASKS`, `_mask_variable` and
   `ufv_masking.ufv_apply_variations_masks` (the three layers) and asserts `'padding not zero'`; with one patch
   removed in a copy it must fail (`AssertionError not raised`), so it cannot pass vacuously.
10. The two floors at 60 with the comment giving the count at K0; the count re-measured by the test module's own scan
    on the built tree (86 expected).
11. The floor of CI: every changed test file parses with `ast.parse(..., feature_version=(3, 9))`; no numpy-2-only or
    jax-0.10-only call in the re-based tests (read by hand); `subTest` labels builtins (the conftest enforces it at
    run time).
12. `test_import_graph.py`'s docstring names the new test; the test asserts absence by `find_spec` and the module map
    only.

**C. The run.**

13. Mine, when the orchestrator gives me the turn: `tally2.py baseline_v.log.gz <the build's -v log>`. Expected 1,882
    passed, 5 skipped, 1 xfailed; every node difference DELETED, renamed, ADDED or CHANGED with its count in the
    handover; no difference on an unchanged test without a reason from 7. The subtests: 93,034 less the 1,767 of the
    deleted tests, plus or minus each edited test's stated delta, summed to the total the log gives.
14. The gate's steps read from its logs (suite, module doctests with their count reconciled with
    `doctest_count.py`'s 308 to 303, quickstart, pages, examples, Sphinx `-W`), the slow tier serially including the
    edited `test_uniform_curvature.py`, and the recorded run of the edited tests on the untouched library
    (`k0/logs/pre_deletion_edited_tests_v.log`): every re-base passed there.
15. Mutations on the **built** tests, in `mut.py` copies, each with its expected outcome (note §6 and the carry-over):
    `H_tilde_dot_step` and `H_tilde_step` x(1+1e-3) on `test_residual_weights`; x(1+1e-6) on the jvp rows (each row
    caught by a step on its path; `V_top_step` a stated survivor); `start_zeros` without `dtype` on the re-based
    dtype test; a dropped subset term in the entries-jets oracle; the dense forward without `t!`; the hygiene control
    with masks (passes) and without (raises). Each copy checked with `mut.py same` before the run.

## Of the approved design, read fresh

Nothing I would reopen. Points to settle in the build review rather than findings now:

- **The tests reached through `_derivative_data`** (item 7) are not in the note's per-file lists, which count only
  `test_curvature`'s seven as "kept unchanged through an edited helper". The note's §3.7 row covers the change
  (`chunk_size` dropped if no caller passes it); the handover should list the tests it reaches.
- **The doctest count.** The note says 320 docstrings with examples stay; `DocTestFinder` gives 308 today and 303
  after the cut. The difference of 5 agrees; the absolute numbers differ by method. The gate's doctest step count is
  the one to state.
- **The line counts** of note §1.2 for `apply` and `entries` (702 → 467, 573 → 424) do not include Decision 2's two
  dead scan bodies; with them the cut gives about 457 and 406. Immaterial.
- The `affine_jet` remark of item 8.
