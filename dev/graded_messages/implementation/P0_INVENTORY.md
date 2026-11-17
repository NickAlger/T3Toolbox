# P0: the inventory

Slice P0 of `PART1_PLAN.md` (section 5.3). A working note, ephemeral. Made on 2026-11-16 on the branch
`graded-messages`, on the unchanged tree (the library and `tests/` as at `4d38df7`); regenerated after the first
review of P0 (the root `README.md` added, the star patterns in backticks listed).

Everything below the line "Generated tables" is the output of

    python3 dev/graded_messages/implementation/inventory.py

and a later slice re-runs it to see what is left. The script's docstring says what it searches, what each kind
means, and what it can and cannot see; read it before relying on a count. For the lines behind a count:

    python3 dev/graded_messages/implementation/inventory.py --lines --name tv_probe --kind call
    python3 dev/graded_messages/implementation/inventory.py --lines --group method --file tests/test_dispatch.py
    python3 dev/graded_messages/implementation/inventory.py --count --group factory --kind call --by file,context

The names are frozen in `inventory_names.json` (written at P0 with `--freeze`), so the inventory keeps its
meaning after a slice has removed a module. Do not run `--freeze` again.

## The names

| Group | Count | From |
| --- | --- | --- |
| sampling names | 166 | the `__all__` lists of `backend/probing` (34), `apply` (10), `entries` (10), `ut3_sampling` (25), `utv_sampling` (30), `sampling_derivatives` (57). `dense_probe` is one of them |
| frontend sampling methods | 51, 21 distinct names | `TuckerTensorTrain` 15, `T3Tangent` 12, `UniformTuckerTensorTrain` 12, `UT3Tangent` 12: the methods named `probe*`, `apply*`, `entries*` (`UniformTuckerTensorTrain.apply_masks` is not one) |
| model factories | 6 | `t3toolbox/fitting.py`: `apply_model`, `entries_model`, `probe_model`, `apply_derivatives_model`, `entries_derivatives_model`, `probe_derivatives_model`; all six re-exported from `t3toolbox/__init__.py` (import and `__all__`) |
| other | 7 | `Problem`, `least_squares_problem`, `_canonical_weight`, `_check_ragged_sample`, `chunk_size`, `sum_over_probes`, `ztildes` |

Searched: 189 files of `t3toolbox/`, `tests/`, `docs/`, `examples/`, `CLAUDE.md` and the root `README.md`. 7,390
occurrences; every one of the 200 distinct names occurs somewhere.

- **The root `README.md` is searched although the plan's list for P0 does not name it.** It holds the old call
  forms (`x.probe(ww)`, `x.apply(ww)`, `x.entries(...)`, and at line 87
  `t3t.newton_cg(t3t.MANIFOLD, 'apply', ww, b, x0, max_newton=30)`, the optimizer call form that P16 and P17
  change), no CI step runs it, and no slice named it: a gap in the plan's file list, found by the reviewer. It was
  added by Claude's decision; the plan will name it in P17.
- **`CHANGELOG.md` and `docs/release_notes.md` are left out on purpose.** They record the history of the releases
  and name the removed names deliberately; the final search of P20 excludes them (section 10 of the plan). Their
  counts are in a table of their own at the end (159 and 9 occurrences), so that nobody takes them for forgotten.

## How to read the counts

- A **call** is one call site in the syntax tree, in code or in a doctest example. This is the plan's unit ("a
  call counts once per call site").
- The method names are not typed: `x.probe(ww)` counts under `probe` whatever `x` is. The rows of `probe`,
  `apply`, `entries` (and of `Problem`) are dominated, outside the `call` columns, by the backend modules of the
  same name, by English words, and by the **old kind strings**: `'probe'`, `'apply_derivatives'` and the other
  kind names are spelled like the methods, so of the 677 occurrences in the `string` column of the method names, 616 are a
  string that is exactly `'probe'`, `'apply'`, `'entries'` or the same with `_derivatives` (kind strings passed to
  the optimizers and the display, and keys of tables by kind), and most of the rest are kind names inside longer
  strings. They go in P16 and P17, not P13.
- The names held **by reference** (a table of functions, a lambda's body is a call, a bare `obj.method` is not) are
  in the `ref` column. For the sampling names there are 55, all in tests; a search by call name misses them. For
  the frontend transposes there are 3 (`tests/test_uniform_probing.py` lines 600 to 602).
- No file imports one of the names under another name (`from m import name as other`). Module aliases
  (`import t3toolbox.backend.probing as t3p`) are everywhere, and are resolved in the `binding` field of
  `--lines`.
- The **star patterns** (`*_from_sweep`, `compute_*`, in backticks or not) are listed in a table of their own,
  with the number of names each matches; they are not counted for any name. Section 5.1 asks each slice's
  inventory for them.
- A local helper with a library name is a `def` with binding `local` (10 of them: `dense_probe` in four test
  files and three examples, `dense_apply_derivatives` in three examples).

## Checked against the plan

Every count and line number that the plan states about these names, recounted with the script. "Agrees" means
the same number, by the plan's own unit.

| Plan, where | Statement | Here |
| --- | --- | --- |
| 2.4 Removed | 166 public names: 34, 10, 10, 25, 30, 57 | agrees |
| 2.4 Removed | the split changed 30 / kept 7 / removed 129, per module (27, 4, 4, 15, 24, 55), by the patterns the table gives | agrees |
| 2.4 Removed | `backend/fitting.py` 20 names; `backend/uniform_fitting.py` 12 | agrees (their `__all__`) |
| 2.5 | 15, 12, 12, 12 sampling methods; 24 go | agrees |
| 2.5, P13 | no call passes `vecs=` | agrees (nor `variation=`, `frame_data=`, `core_pair=`, `ztildes=`, P12) |
| 1 | `sampling_derivatives.py` calls `xmap` sixteen times | agrees |
| 1 | "the library's two chunked uniform assemblies (`_wchunked_sum`)" | **differs in the count**: `_wchunked_sum` is called from four functions (`assemble_tucker_variation_jets`, `assemble_tt_variation_jets`, `assemble_tucker_tangent_variation_jets`, `assemble_tt_tangent_variation_jets`; lines 1538, 1776, 2826, 2948). Two if the Tucker and TT halves are counted as one assembly each |
| P2 | 12 calls of the library's `dense_probe`, all positional: `backend/probing.py` doctests 8, `manifold.py` 1, `uniform_manifold.py` 1, `tests/test_manifold.py` 524, `tests/test_tucker_tensor_train.py` 3166 | agrees |
| P2 | four test files and three examples define a local `dense_probe` | agrees |
| P12 | the table of calls of the 30 changed names (calls, in tests, in doctests), every row | agrees, every cell |
| P12 | the lines of the six `point_forward` calls (`backend/fitting.py` 538, 565, 604; `uniform_fitting.py` 143, 172, 207) | agrees |
| P12 | the references held in tables: `tests/backend/test_uniform_curvature.py` 595 to 608 and its calls at 646, 647, 656, 673, 676; `tests/test_dispatch.py` 619 to 626; `tests/backend/test_curvature.py` 889 to 896; `tests/test_uniform_probing.py` 832 to 836, 906 to 910 | agrees |
| P12 | the four callers of the backend ambient transposes | agrees (3 in `tucker_tensor_train.py`, 1 in `backend/entries.py`) |
| P13 | 91 calls of the frontend transposes and corewise transposes, by name (23, 14, 10, 12, 17, 15) and by file | agrees, every number |
| P13 | the ambient transposes: 16 calls in tests, nine with `sum_over_probes=`, three doctests | agrees |
| P13 | `tests/test_uniform_probing.py` 52 calls of removed methods; the 16 test files listed | agrees |
| P13 | `tests/test_jax_absent.py` lines 66 and 79 (and line 71, section 10) | agrees |
| P13 | `t3toolbox/optimizers.py` and `t3toolbox/fitting.py` doctests: two calls each of a removed method | agrees |
| P13 | page doctests: `fitting_and_optimization.md` five calls, `numerical_contracts.md` two | agrees |
| P13 | links to `docs/chunking.md`: `fitting_and_optimization.md` three, `glossary.md` three, `optimizers.py` three, `manifold.py` one, `sampling_derivatives.py` four | **differs for the glossary**: it has two links (lines 398 and 406); the third mention (line 769) is the file name in backticks, not a link. The others agree. (In the two pages they are markdown links, `[...](chunking.md#...)`, not `:doc:` roles; in the docstrings they are `:doc:`) |
| P13 | six examples call a removed method; five import `sampling_derivatives` as `pd`; `pd.dense_apply_derivatives` once in three (helpers at lines 116, 110, 74, each called twice), `pd.dense_probe_derivatives` once in two; seven examples define their own `dense_apply` or `dense_probe` | agrees |
| P13 | `tests/test_probe_derivatives.py`: nine calls of the three `dense_*_derivatives` | agrees |
| P16 | `tests/test_fitting.py`: three calls of `_canonical_weight` | agrees |
| P16 | `t3toolbox/fitting.py` doctests: six factory calls | **differs: seven** (lines 244, 270, 285, 770, 779, 839, 931; line 779 is the call whose traceback the example shows) |
| P16 | `t3toolbox/optimizers.py` doctests: four calls of `least_squares_problem` | agrees |
| P16 | page doctests that call a factory: `fitting_and_optimization.md` lines 147, 182, 424, 430; `numerical_contracts.md` line 94 | agrees |
| P16 | the five examples that call a factory, and which | agrees |
| 10 | the 74 removed names outside `sampling_derivatives` | agrees |

Not checked here, because they are about names outside P0's list: P15's calls of a backend geometry's `frame`,
P16's sixteen optimizer calls in the doctests of `t3toolbox/optimizers.py`, and section 10's 14 prototype scripts
(`dev/` is not searched).

Three disagreements, all small, none changes what a slice does: the fourth `_wchunked_sum` caller pair
(wording), the glossary's two links, and the seventh factory call in the doctests of `t3toolbox/fitting.py`.

## Generated tables

Files searched: 189. Occurrences: 7390, of 200 names (0 of the frozen names occur nowhere).

Columns: `def` definitions; `import`; `call` call sites in code; `kw` keyword arguments and parameters; `ref` other uses in code (tables, arguments, assignments); `dt-call`, `dt-other` the same inside doctest examples (docstrings and pages; `dt-other` includes names inside the string literals of an example); `string` in strings that are not docstrings (`__all__` entries, `getattr` names, labels, scripts in strings); `text` comments, docstring prose, page prose, fenced code without prompts, brace patterns.

### By group and area

| group | area | def | import | call | kw | ref | dt-call | dt-other | string | text | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sampling | t3toolbox/backend | 166 | 6 | 303 | 0 | 0 | 37 | 0 | 170 | 452 | 1134 |
| sampling | t3toolbox | 0 | 0 | 77 | 0 | 0 | 2 | 0 | 0 | 11 | 90 |
| sampling | tests | 4 | 0 | 412 | 0 | 55 | 0 | 0 | 37 | 33 | 541 |
| sampling | docs | 0 | 0 | 0 | 0 | 0 | 7 | 2 | 0 | 83 | 92 |
| sampling | examples | 6 | 0 | 14 | 0 | 0 | 0 | 0 | 0 | 3 | 23 |
| sampling | CLAUDE.md | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| **sampling** | **all** | 176 | 6 | 806 | 0 | 55 | 46 | 2 | 207 | 586 | 1884 |
| method | t3toolbox/backend | 0 | 14 | 0 | 0 | 20 | 1 | 11 | 34 | 651 | 731 |
| method | t3toolbox | 51 | 8 | 0 | 0 | 10 | 82 | 38 | 40 | 479 | 708 |
| method | tests | 0 | 27 | 426 | 2 | 36 | 0 | 0 | 569 | 398 | 1458 |
| method | docs | 0 | 0 | 0 | 0 | 0 | 26 | 28 | 0 | 627 | 681 |
| method | examples | 0 | 0 | 32 | 0 | 0 | 0 | 0 | 34 | 171 | 237 |
| method | CLAUDE.md | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 26 |
| method | README.md | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 12 |
| **method** | **all** | 51 | 49 | 458 | 2 | 66 | 109 | 77 | 677 | 2364 | 3853 |
| factory | t3toolbox | 6 | 6 | 0 | 0 | 0 | 7 | 0 | 18 | 15 | 52 |
| factory | tests | 0 | 0 | 100 | 0 | 28 | 0 | 0 | 2 | 8 | 138 |
| factory | docs | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 18 | 23 |
| factory | examples | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 7 | 12 |
| **factory** | **all** | 6 | 6 | 105 | 0 | 28 | 12 | 0 | 20 | 48 | 225 |
| other | t3toolbox/backend | 2 | 0 | 3 | 162 | 168 | 3 | 7 | 17 | 118 | 480 |
| other | t3toolbox | 3 | 0 | 16 | 83 | 49 | 4 | 14 | 2 | 60 | 231 |
| other | tests | 0 | 0 | 115 | 302 | 34 | 0 | 0 | 20 | 81 | 552 |
| other | docs | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 136 | 142 |
| other | examples | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 19 | 21 |
| other | CLAUDE.md | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| **other** | **all** | 5 | 0 | 134 | 549 | 251 | 7 | 27 | 39 | 416 | 1428 |

### The 166 sampling names, by name

| name | module | def | import | call | kw | ref | dt-call | dt-other | string | text | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `t3_probe` | probing | 1 | 0 | 6 | 0 | 0 | 3 | 0 | 2 | 14 | 26 |
| `compute_xi` | probing | 1 | 2 | 41 | 0 | 0 | 0 | 0 | 1 | 9 | 54 |
| `compute_mu` | probing | 1 | 2 | 12 | 0 | 0 | 0 | 0 | 1 | 5 | 21 |
| `compute_nu` | probing | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 1 | 2 | 10 |
| `compute_eta` | probing | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 1 | 3 | 10 |
| `assemble_z` | probing | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 4 | 7 |
| `tv_probe` | probing | 1 | 0 | 5 | 0 | 0 | 4 | 0 | 2 | 21 | 33 |
| `compute_dxi` | probing | 1 | 1 | 9 | 0 | 0 | 0 | 0 | 1 | 8 | 20 |
| `compute_sigma` | probing | 1 | 0 | 9 | 0 | 0 | 0 | 0 | 1 | 12 | 23 |
| `compute_tau` | probing | 1 | 0 | 7 | 0 | 0 | 0 | 0 | 1 | 9 | 18 |
| `compute_deta` | probing | 1 | 0 | 7 | 0 | 0 | 0 | 0 | 1 | 13 | 22 |
| `assemble_tangent_z` | probing | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 7 | 11 |
| `compute_sigma_hat` | probing | 1 | 1 | 3 | 0 | 0 | 0 | 0 | 1 | 2 | 8 |
| `tv_precompute_probe_frame_sweep` | probing | 1 | 0 | 3 | 0 | 2 | 0 | 0 | 1 | 10 | 17 |
| `tv_probe_jacobian_from_sweep` | probing | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 3 | 8 |
| `tv_probe_transpose_from_sweep` | probing | 1 | 0 | 3 | 0 | 1 | 0 | 0 | 1 | 3 | 9 |
| `compute_deta_tilde` | probing | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 3 | 9 |
| `compute_tau_tilde` | probing | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 1 | 7 | 14 |
| `compute_sigma_tilde` | probing | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 3 | 9 |
| `compute_dxi_tilde` | probing | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 3 | 8 |
| `assemble_tucker_variations` | probing | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `assemble_tt_variations` | probing | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 5 | 9 |
| `tv_probe_transpose` | probing | 1 | 0 | 5 | 0 | 0 | 2 | 0 | 1 | 9 | 18 |
| `t3_probe_ambient_transpose` | probing | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 2 | 5 |
| `t3_probe_corewise_transpose` | probing | 1 | 0 | 5 | 0 | 0 | 1 | 0 | 1 | 4 | 12 |
| `compute_deta_tilde_tangent` | probing | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 3 | 8 |
| `compute_tau_tilde_tangent` | probing | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 3 | 9 |
| `compute_sigma_tilde_tangent` | probing | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 1 | 6 |
| `compute_dxi_tilde_tangent` | probing | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 2 | 6 |
| `assemble_tucker_tangent_variations` | probing | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 2 | 5 |
| `assemble_tt_tangent_variations` | probing | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 4 | 8 |
| `t3_probe_corewise_hessian` | probing | 1 | 0 | 2 | 0 | 3 | 2 | 0 | 1 | 6 | 15 |
| `t3_probe_corewise_hessian_from_sweep` | probing | 1 | 0 | 3 | 0 | 2 | 0 | 0 | 2 | 2 | 10 |
| `dense_probe` | probing | 8 | 0 | 33 | 0 | 0 | 10 | 0 | 3 | 3 | 57 |
| `t3_apply` | apply | 1 | 0 | 9 | 0 | 0 | 0 | 0 | 2 | 7 | 19 |
| `t3_apply_ambient_transpose` | apply | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `tv_apply` | apply | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 2 | 15 | 24 |
| `tv_apply_transpose` | apply | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 1 | 6 | 14 |
| `tv_precompute_apply_frame_sweep` | apply | 1 | 0 | 6 | 0 | 2 | 0 | 0 | 1 | 12 | 22 |
| `tv_apply_jacobian_from_sweep` | apply | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 2 | 8 |
| `tv_apply_transpose_from_sweep` | apply | 1 | 0 | 5 | 0 | 1 | 0 | 0 | 1 | 3 | 11 |
| `t3_apply_corewise_transpose` | apply | 1 | 0 | 6 | 0 | 0 | 1 | 0 | 1 | 4 | 13 |
| `t3_apply_corewise_hessian` | apply | 1 | 0 | 4 | 0 | 3 | 1 | 0 | 1 | 8 | 18 |
| `t3_apply_corewise_hessian_from_sweep` | apply | 1 | 0 | 3 | 0 | 2 | 0 | 0 | 2 | 1 | 9 |
| `t3_entries` | entries | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 2 | 6 | 12 |
| `t3_entries_ambient_transpose` | entries | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 4 |
| `tv_entries` | entries | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 2 | 8 | 17 |
| `tv_entries_transpose` | entries | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 1 | 3 | 11 |
| `tv_precompute_entries_frame_sweep` | entries | 1 | 0 | 5 | 0 | 2 | 0 | 0 | 1 | 4 | 13 |
| `tv_entries_jacobian_from_sweep` | entries | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 2 | 7 |
| `tv_entries_transpose_from_sweep` | entries | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 4 | 11 |
| `t3_entries_corewise_transpose` | entries | 1 | 0 | 5 | 0 | 0 | 1 | 0 | 1 | 2 | 10 |
| `t3_entries_corewise_hessian` | entries | 1 | 0 | 3 | 0 | 3 | 1 | 0 | 1 | 7 | 16 |
| `t3_entries_corewise_hessian_from_sweep` | entries | 1 | 0 | 3 | 0 | 2 | 0 | 0 | 2 | 1 | 9 |
| `ut3_entries` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `ut3_apply` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `ut3_probe` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 4 | 8 |
| `ut3_full_sum` | ut3_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 2 | 5 |
| `ut3_apply_corewise_transpose` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `ut3_entries_corewise_transpose` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `ut3_probe_corewise_transpose` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `ut3_probe_derivatives` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| `ut3_apply_derivatives` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| `ut3_entries_derivatives` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| `ut3_apply_corewise_derivatives_transpose` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| `ut3_entries_corewise_derivatives_transpose` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| `ut3_probe_corewise_derivatives_transpose` | ut3_sampling | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 1 | 6 |
| `ut3_probe_corewise_derivatives_hessian` | ut3_sampling | 1 | 0 | 5 | 0 | 0 | 1 | 0 | 1 | 3 | 11 |
| `ut3_probe_corewise_derivatives_hessian_from_sweep` | ut3_sampling | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 2 | 8 |
| `ut3_probe_corewise_hessian` | ut3_sampling | 1 | 0 | 0 | 0 | 2 | 1 | 0 | 1 | 4 | 9 |
| `ut3_probe_corewise_hessian_from_sweep` | ut3_sampling | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 2 | 6 |
| `ut3_apply_corewise_hessian` | ut3_sampling | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 4 | 8 |
| `ut3_apply_corewise_hessian_from_sweep` | ut3_sampling | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 5 |
| `ut3_entries_corewise_hessian` | ut3_sampling | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 2 | 6 |
| `ut3_entries_corewise_hessian_from_sweep` | ut3_sampling | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 5 |
| `ut3_apply_corewise_derivatives_hessian` | ut3_sampling | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 4 | 9 |
| `ut3_apply_corewise_derivatives_hessian_from_sweep` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| `ut3_entries_corewise_derivatives_hessian` | ut3_sampling | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 2 | 6 |
| `ut3_entries_corewise_derivatives_hessian_from_sweep` | ut3_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| `utv_probe` | utv_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 2 | 6 |
| `utv_apply` | utv_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 1 | 5 |
| `utv_entries` | utv_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 4 |
| `utv_probe_transpose` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `utv_apply_transpose` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| `utv_entries_transpose` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `utv_precompute_apply_frame_sweep` | utv_sampling | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 3 | 10 |
| `utv_precompute_probe_frame_sweep` | utv_sampling | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 3 | 10 |
| `utv_precompute_entries_frame_sweep` | utv_sampling | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 3 | 10 |
| `utv_apply_jacobian_from_sweep` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `utv_probe_jacobian_from_sweep` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `utv_entries_jacobian_from_sweep` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `utv_apply_transpose_from_sweep` | utv_sampling | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 0 | 7 |
| `utv_probe_transpose_from_sweep` | utv_sampling | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 0 | 7 |
| `utv_entries_transpose_from_sweep` | utv_sampling | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 0 | 7 |
| `utv_precompute_apply_frame_sweep_jets` | utv_sampling | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 3 | 9 |
| `utv_precompute_probe_frame_sweep_jets` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `utv_precompute_entries_frame_sweep_jets` | utv_sampling | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 3 | 8 |
| `utv_apply_jacobian_derivatives_from_sweep` | utv_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 3 |
| `utv_probe_jacobian_derivatives_from_sweep` | utv_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 3 |
| `utv_entries_jacobian_derivatives_from_sweep` | utv_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 3 |
| `utv_apply_transpose_derivatives_from_sweep` | utv_sampling | 1 | 0 | 3 | 0 | 1 | 0 | 0 | 1 | 0 | 6 |
| `utv_probe_transpose_derivatives_from_sweep` | utv_sampling | 1 | 0 | 3 | 0 | 1 | 0 | 0 | 1 | 0 | 6 |
| `utv_entries_transpose_derivatives_from_sweep` | utv_sampling | 1 | 0 | 3 | 0 | 1 | 0 | 0 | 1 | 0 | 6 |
| `utv_probe_derivatives` | utv_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 4 |
| `utv_apply_derivatives` | utv_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 4 |
| `utv_entries_derivatives` | utv_sampling | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 4 |
| `utv_probe_derivatives_transpose` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `utv_apply_derivatives_transpose` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `utv_entries_derivatives_transpose` | utv_sampling | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 4 |
| `check_perturbation_vectors` | sampling_derivatives | 1 | 0 | 16 | 0 | 0 | 0 | 0 | 1 | 0 | 18 |
| `check_perturbation_index` | sampling_derivatives | 1 | 0 | 8 | 0 | 0 | 0 | 0 | 1 | 0 | 10 |
| `t3_probe_derivatives` | sampling_derivatives | 1 | 0 | 32 | 0 | 0 | 1 | 0 | 3 | 5 | 42 |
| `build_input_jets` | sampling_derivatives | 1 | 0 | 24 | 0 | 0 | 0 | 0 | 1 | 4 | 30 |
| `compute_mu_jets` | sampling_derivatives | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 1 | 7 | 14 |
| `compute_nu_jets` | sampling_derivatives | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 2 | 8 |
| `compute_eta_jets` | sampling_derivatives | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 8 | 14 |
| `assemble_z_jets` | sampling_derivatives | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 3 | 6 |
| `tv_precompute_apply_frame_sweep_jets` | sampling_derivatives | 1 | 0 | 9 | 0 | 1 | 0 | 0 | 1 | 7 | 19 |
| `tv_precompute_entries_frame_sweep_jets` | sampling_derivatives | 1 | 0 | 7 | 0 | 1 | 0 | 0 | 1 | 5 | 15 |
| `tv_precompute_probe_frame_sweep_jets` | sampling_derivatives | 1 | 0 | 7 | 0 | 0 | 0 | 0 | 1 | 5 | 14 |
| `tv_probe_derivatives` | sampling_derivatives | 1 | 0 | 13 | 0 | 0 | 0 | 0 | 3 | 7 | 24 |
| `tv_probe_jacobian_derivatives_from_sweep` | sampling_derivatives | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 1 | 7 |
| `compute_sigma_jets` | sampling_derivatives | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 1 | 4 | 11 |
| `compute_tau_jets` | sampling_derivatives | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 1 | 6 |
| `compute_deta_jets` | sampling_derivatives | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 4 | 10 |
| `assemble_tangent_z_jets` | sampling_derivatives | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 2 | 5 |
| `t3_apply_derivatives` | sampling_derivatives | 1 | 0 | 11 | 0 | 0 | 0 | 0 | 2 | 5 | 19 |
| `tv_apply_derivatives` | sampling_derivatives | 1 | 0 | 9 | 0 | 0 | 0 | 0 | 3 | 5 | 18 |
| `tv_apply_jacobian_derivatives_from_sweep` | sampling_derivatives | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 2 | 8 |
| `t3_entries_derivatives` | sampling_derivatives | 1 | 0 | 9 | 0 | 0 | 0 | 0 | 2 | 4 | 16 |
| `tv_entries_derivatives` | sampling_derivatives | 1 | 0 | 7 | 0 | 0 | 0 | 0 | 2 | 3 | 13 |
| `tv_entries_jacobian_derivatives_from_sweep` | sampling_derivatives | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 1 | 7 |
| `tv_probe_derivatives_transpose` | sampling_derivatives | 1 | 0 | 17 | 0 | 0 | 0 | 0 | 3 | 9 | 30 |
| `tv_probe_transpose_derivatives_from_sweep` | sampling_derivatives | 1 | 0 | 5 | 0 | 1 | 0 | 0 | 1 | 1 | 9 |
| `estimate_chunk_size` | sampling_derivatives | 1 | 0 | 18 | 0 | 1 | 3 | 1 | 8 | 19 | 51 |
| `max_chunk_size_within` | sampling_derivatives | 1 | 0 | 8 | 0 | 0 | 4 | 1 | 2 | 11 | 27 |
| `compute_deta_tilde_jets` | sampling_derivatives | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 2 | 8 |
| `compute_tau_tilde_jets` | sampling_derivatives | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 5 | 11 |
| `compute_sigma_tilde_jets` | sampling_derivatives | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 4 | 9 |
| `compute_dxi_tilde_jets` | sampling_derivatives | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 2 | 7 |
| `assemble_tucker_variation_jets` | sampling_derivatives | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 2 | 7 |
| `assemble_tt_variation_jets` | sampling_derivatives | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 1 | 3 | 10 |
| `tv_apply_derivatives_transpose` | sampling_derivatives | 1 | 0 | 9 | 0 | 0 | 0 | 0 | 2 | 4 | 16 |
| `tv_apply_transpose_derivatives_from_sweep` | sampling_derivatives | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 2 | 9 |
| `tv_entries_derivatives_transpose` | sampling_derivatives | 1 | 0 | 8 | 0 | 0 | 0 | 0 | 2 | 4 | 15 |
| `tv_entries_transpose_derivatives_from_sweep` | sampling_derivatives | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 1 | 1 | 8 |
| `compute_sigma_hat_jets` | sampling_derivatives | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 1 | 5 | 11 |
| `t3_probe_corewise_derivatives_transpose` | sampling_derivatives | 1 | 0 | 10 | 0 | 0 | 1 | 0 | 1 | 2 | 15 |
| `t3_apply_corewise_derivatives_transpose` | sampling_derivatives | 1 | 0 | 7 | 0 | 0 | 1 | 0 | 1 | 2 | 12 |
| `t3_entries_corewise_derivatives_transpose` | sampling_derivatives | 1 | 0 | 7 | 0 | 0 | 1 | 0 | 1 | 2 | 12 |
| `t3_probe_corewise_derivatives_hessian` | sampling_derivatives | 1 | 0 | 24 | 0 | 0 | 3 | 0 | 1 | 13 | 42 |
| `t3_probe_corewise_derivatives_hessian_from_sweep` | sampling_derivatives | 1 | 0 | 5 | 0 | 1 | 0 | 0 | 2 | 2 | 11 |
| `compute_deta_tilde_tangent_jets` | sampling_derivatives | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `compute_tau_tilde_tangent_jets` | sampling_derivatives | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 5 | 9 |
| `compute_sigma_tilde_tangent_jets` | sampling_derivatives | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `compute_dxi_tilde_tangent_jets` | sampling_derivatives | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `assemble_tucker_tangent_variation_jets` | sampling_derivatives | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `assemble_tt_tangent_variation_jets` | sampling_derivatives | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 4 | 8 |
| `t3_apply_corewise_derivatives_hessian` | sampling_derivatives | 1 | 0 | 6 | 0 | 1 | 2 | 0 | 1 | 6 | 17 |
| `t3_apply_corewise_derivatives_hessian_from_sweep` | sampling_derivatives | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 2 | 2 | 10 |
| `t3_entries_corewise_derivatives_hessian` | sampling_derivatives | 1 | 0 | 5 | 0 | 1 | 1 | 0 | 1 | 6 | 15 |
| `t3_entries_corewise_derivatives_hessian_from_sweep` | sampling_derivatives | 1 | 0 | 4 | 0 | 1 | 0 | 0 | 2 | 0 | 8 |
| `compute_sigma_hat_tangent_jets` | sampling_derivatives | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 3 | 7 |
| `dense_probe_derivatives` | sampling_derivatives | 1 | 0 | 7 | 0 | 0 | 0 | 0 | 1 | 4 | 13 |
| `dense_apply_derivatives` | sampling_derivatives | 4 | 0 | 12 | 0 | 0 | 0 | 0 | 1 | 6 | 23 |
| `dense_entries_derivatives` | sampling_derivatives | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 2 | 6 |

### The frontend sampling methods, by name

One row per distinct name; the classes that define it are in the second column (T = `TuckerTensorTrain`, G = `T3Tangent`, UT = `UniformTuckerTensorTrain`, UG = `UT3Tangent`). The receiver of a method call is not typed, so a row counts every attribute of that name; for `probe`, `apply`, `entries` the `import` and `text` columns are mostly the backend modules of the same name and English words (marked \*).

| name | defined in | def | import | call | kw | ref | dt-call | dt-other | string | text | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `entries` (\*) | T, G, UT, UG | 4 | 23 | 43 | 2 | 34 | 12 | 8 | 105 | 489 | 720 |
| `apply` (\*) | T, G, UT, UG | 4 | 26 | 79 | 0 | 26 | 24 | 24 | 161 | 650 | 994 |
| `probe` (\*) | T, G, UT, UG | 4 | 0 | 69 | 0 | 0 | 24 | 29 | 191 | 957 | 1274 |
| `probe_ambient_transpose` | T | 1 | 0 | 5 | 0 | 0 | 1 | 0 | 1 | 6 | 14 |
| `probe_corewise_transpose` | T, UT | 2 | 0 | 12 | 0 | 0 | 0 | 0 | 1 | 7 | 22 |
| `apply_ambient_transpose` | T | 1 | 0 | 5 | 0 | 0 | 1 | 0 | 1 | 17 | 25 |
| `entries_ambient_transpose` | T | 1 | 0 | 6 | 0 | 0 | 1 | 0 | 1 | 8 | 17 |
| `apply_corewise_transpose` | T, UT | 2 | 0 | 14 | 0 | 0 | 3 | 0 | 1 | 19 | 39 |
| `entries_corewise_transpose` | T, UT | 2 | 0 | 15 | 0 | 0 | 0 | 0 | 1 | 6 | 24 |
| `probe_derivatives` | T, G, UT, UG | 4 | 0 | 59 | 0 | 0 | 18 | 15 | 118 | 71 | 285 |
| `apply_derivatives` | T, G, UT, UG | 4 | 0 | 31 | 0 | 0 | 5 | 1 | 47 | 35 | 123 |
| `entries_derivatives` | T, G, UT, UG | 4 | 0 | 26 | 0 | 0 | 4 | 0 | 48 | 21 | 103 |
| `probe_corewise_derivatives_transpose` | T, UT | 2 | 0 | 13 | 0 | 0 | 0 | 0 | 0 | 6 | 21 |
| `apply_corewise_derivatives_transpose` | T, UT | 2 | 0 | 9 | 0 | 0 | 2 | 0 | 0 | 8 | 21 |
| `entries_corewise_derivatives_transpose` | T, UT | 2 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 1 | 12 |
| `probe_transpose` | G, UG | 2 | 0 | 15 | 0 | 1 | 8 | 0 | 0 | 18 | 44 |
| `apply_transpose` | G, UG | 2 | 0 | 13 | 0 | 1 | 1 | 0 | 0 | 18 | 35 |
| `entries_transpose` | G, UG | 2 | 0 | 10 | 0 | 1 | 0 | 0 | 0 | 8 | 21 |
| `probe_derivatives_transpose` | G, UG | 2 | 0 | 9 | 0 | 1 | 5 | 0 | 1 | 9 | 27 |
| `apply_derivatives_transpose` | G, UG | 2 | 0 | 9 | 0 | 1 | 0 | 0 | 0 | 7 | 19 |
| `entries_derivatives_transpose` | G, UG | 2 | 0 | 7 | 0 | 1 | 0 | 0 | 0 | 3 | 13 |

### The six model factories, by name

| name | defined in | def | import | call | kw | ref | dt-call | dt-other | string | text | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `apply_model` |  | 1 | 1 | 31 | 0 | 5 | 5 | 0 | 3 | 13 | 59 |
| `entries_model` |  | 1 | 1 | 5 | 0 | 5 | 0 | 0 | 3 | 8 | 23 |
| `probe_model` |  | 1 | 1 | 15 | 0 | 6 | 3 | 0 | 3 | 11 | 40 |
| `apply_derivatives_model` |  | 1 | 1 | 6 | 0 | 4 | 1 | 0 | 3 | 5 | 21 |
| `entries_derivatives_model` |  | 1 | 1 | 3 | 0 | 4 | 0 | 0 | 3 | 3 | 15 |
| `probe_derivatives_model` |  | 1 | 1 | 45 | 0 | 4 | 3 | 0 | 5 | 8 | 67 |

### The other names, by name

| name | defined in | def | import | call | kw | ref | dt-call | dt-other | string | text | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Problem` (\*) |  | 1 | 0 | 2 | 0 | 19 | 0 | 0 | 17 | 118 | 157 |
| `least_squares_problem` |  | 2 | 0 | 118 | 0 | 2 | 7 | 0 | 14 | 32 | 175 |
| `_canonical_weight` |  | 1 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 2 | 11 |
| `_check_ragged_sample` |  | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| `chunk_size` |  | 0 | 0 | 0 | 145 | 60 | 0 | 3 | 3 | 91 | 302 |
| `sum_over_probes` |  | 0 | 0 | 0 | 360 | 95 | 0 | 24 | 5 | 139 | 623 |
| `ztildes` |  | 0 | 0 | 0 | 44 | 75 | 0 | 0 | 0 | 34 | 153 |

### By file

Every file with at least one occurrence. `s`, `m`, `f`, `o`: occurrences of the sampling names, the method names, the factories, the other names; then the same columns as above over all four.

| file | s | m | f | o | def | import | call | kw | ref | dt-call | dt-other | string | text | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `t3toolbox/backend/apply.py` | 82 | 50 | 0 | 19 | 10 | 4 | 17 | 6 | 6 | 2 | 3 | 10 | 93 | 151 |
| `t3toolbox/backend/common.py` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `t3toolbox/backend/contractions.py` | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 4 |
| `t3toolbox/backend/entries.py` | 60 | 43 | 0 | 13 | 10 | 4 | 9 | 6 | 5 | 2 | 2 | 10 | 68 | 116 |
| `t3toolbox/backend/fitting.py` | 30 | 71 | 0 | 14 | 0 | 2 | 30 | 8 | 3 | 0 | 0 | 9 | 63 | 115 |
| `t3toolbox/backend/fv_operations.py` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `t3toolbox/backend/linalg.py` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `t3toolbox/backend/optimizer_display.py` | 0 | 7 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 9 |
| `t3toolbox/backend/optimizers.py` | 2 | 11 | 0 | 57 | 2 | 0 | 2 | 0 | 11 | 5 | 1 | 16 | 33 | 70 |
| `t3toolbox/backend/probing.py` | 277 | 104 | 0 | 86 | 34 | 0 | 64 | 23 | 37 | 19 | 5 | 37 | 248 | 467 |
| `t3toolbox/backend/ranks.py` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `t3toolbox/backend/regularization.py` | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `t3toolbox/backend/sampling_derivatives.py` | 390 | 188 | 0 | 165 | 57 | 2 | 98 | 51 | 69 | 6 | 3 | 61 | 396 | 743 |
| `t3toolbox/backend/sharing.py` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `t3toolbox/backend/t3_conversions.py` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `t3toolbox/backend/t3_svd.py` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `t3toolbox/backend/ufv_masking.py` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `t3toolbox/backend/ufv_operations.py` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `t3toolbox/backend/uniform_fitting.py` | 36 | 63 | 0 | 25 | 0 | 0 | 31 | 12 | 5 | 0 | 3 | 23 | 50 | 124 |
| `t3toolbox/backend/ut3_masking.py` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `t3toolbox/backend/ut3_sampling.py` | 126 | 67 | 0 | 44 | 25 | 4 | 25 | 24 | 24 | 7 | 0 | 25 | 103 | 237 |
| `t3toolbox/backend/ut3_svd.py` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `t3toolbox/backend/utv_sampling.py` | 131 | 102 | 0 | 52 | 30 | 4 | 30 | 32 | 28 | 0 | 0 | 30 | 131 | 285 |
| `t3toolbox/__init__.py` | 0 | 0 | 12 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 12 |
| `t3toolbox/fitting.py` | 2 | 93 | 40 | 15 | 8 | 0 | 10 | 0 | 0 | 9 | 1 | 38 | 84 | 150 |
| `t3toolbox/frame_variations_format.py` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `t3toolbox/manifold.py` | 19 | 141 | 0 | 47 | 12 | 4 | 18 | 16 | 13 | 22 | 9 | 0 | 113 | 207 |
| `t3toolbox/optimizers.py` | 7 | 81 | 0 | 55 | 1 | 0 | 7 | 12 | 9 | 12 | 16 | 15 | 71 | 143 |
| `t3toolbox/safety.py` | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| `t3toolbox/shared_geometry.py` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `t3toolbox/tucker_tensor_train.py` | 21 | 193 | 0 | 46 | 15 | 4 | 21 | 23 | 19 | 23 | 10 | 1 | 144 | 260 |
| `t3toolbox/uniform_frame_variations_format.py` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `t3toolbox/uniform_manifold.py` | 19 | 97 | 0 | 36 | 12 | 0 | 18 | 16 | 9 | 16 | 8 | 0 | 73 | 152 |
| `t3toolbox/uniform_tucker_tensor_train.py` | 22 | 92 | 0 | 32 | 12 | 0 | 19 | 16 | 9 | 13 | 8 | 0 | 69 | 146 |
| `tests/_jet_trs_twins.py` | 0 | 8 | 0 | 4 | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 8 | 12 |
| `tests/backend/_curvature_oracles.py` | 22 | 23 | 0 | 10 | 0 | 1 | 12 | 5 | 7 | 0 | 0 | 0 | 30 | 55 |
| `tests/backend/_uniform_padded_frame_testing.py` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `tests/backend/test_curvature.py` | 113 | 150 | 0 | 44 | 0 | 2 | 101 | 22 | 13 | 0 | 0 | 68 | 101 | 307 |
| `tests/backend/test_from_sweep_reuse.py` | 48 | 24 | 0 | 21 | 0 | 4 | 48 | 21 | 14 | 0 | 0 | 6 | 0 | 93 |
| `tests/backend/test_linalg.py` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `tests/backend/test_optimizer_display.py` | 0 | 10 | 0 | 5 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 8 | 15 |
| `tests/backend/test_optimizers.py` | 22 | 43 | 6 | 69 | 1 | 0 | 83 | 3 | 12 | 0 | 0 | 18 | 23 | 140 |
| `tests/backend/test_padded_start.py` | 0 | 15 | 0 | 7 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 8 | 5 | 22 |
| `tests/backend/test_ranks.py` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `tests/backend/test_uniform_curvature.py` | 43 | 88 | 0 | 43 | 0 | 2 | 30 | 26 | 22 | 0 | 0 | 58 | 36 | 174 |
| `tests/backend/test_uniform_fitting.py` | 19 | 75 | 0 | 17 | 0 | 3 | 17 | 7 | 13 | 0 | 0 | 53 | 18 | 111 |
| `tests/backend/test_uniform_padded_frame.py` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `tests/backend/test_zero_reference.py` | 0 | 19 | 0 | 3 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 14 | 0 | 22 |
| `tests/oracle_sweeps/o1_common.py` | 3 | 2 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 2 | 5 |
| `tests/oracle_sweeps/o1_ragged.py` | 14 | 64 | 0 | 15 | 0 | 0 | 42 | 15 | 0 | 0 | 0 | 36 | 0 | 93 |
| `tests/oracle_sweeps/o1_uniform.py` | 10 | 62 | 0 | 19 | 0 | 0 | 51 | 19 | 0 | 0 | 0 | 21 | 0 | 91 |
| `tests/oracle_sweeps/oracle.py` | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 8 |
| `tests/oracle_sweeps/sweep_exact.py` | 0 | 14 | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 14 | 0 | 20 |
| `tests/oracle_sweeps/sweep_models.py` | 0 | 33 | 6 | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 27 | 0 | 39 |
| `tests/oracle_sweeps/sweep_optimizers.py` | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 9 |
| `tests/test_aligned_padded_frame_frontend.py` | 0 | 8 | 0 | 1 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 5 | 0 | 9 |
| `tests/test_chunk_size_estimator.py` | 34 | 27 | 0 | 12 | 0 | 0 | 29 | 7 | 1 | 0 | 0 | 18 | 18 | 73 |
| `tests/test_contractions_interpreter.py` | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 |
| `tests/test_contractions_sharding.py` | 2 | 2 | 0 | 6 | 0 | 0 | 2 | 5 | 0 | 0 | 0 | 1 | 2 | 10 |
| `tests/test_dispatch.py` | 78 | 128 | 18 | 68 | 0 | 5 | 137 | 54 | 21 | 0 | 0 | 31 | 44 | 292 |
| `tests/test_fitting.py` | 5 | 120 | 88 | 11 | 1 | 0 | 90 | 3 | 25 | 0 | 0 | 59 | 46 | 224 |
| `tests/test_jax_absent.py` | 0 | 5 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 1 | 8 |
| `tests/test_jet_recurrence.py` | 28 | 8 | 0 | 13 | 0 | 0 | 28 | 8 | 0 | 0 | 0 | 7 | 6 | 49 |
| `tests/test_manifold.py` | 6 | 61 | 1 | 9 | 0 | 4 | 34 | 8 | 4 | 0 | 0 | 7 | 20 | 77 |
| `tests/test_optimizers_frontend.py` | 8 | 131 | 1 | 70 | 1 | 0 | 36 | 15 | 5 | 0 | 0 | 110 | 43 | 210 |
| `tests/test_oracle_sweep.py` | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 4 |
| `tests/test_probe_derivatives.py` | 52 | 34 | 0 | 25 | 0 | 4 | 56 | 16 | 5 | 0 | 0 | 14 | 16 | 111 |
| `tests/test_sharing.py` | 0 | 22 | 10 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 11 | 1 | 32 |
| `tests/test_tucker_tensor_train.py` | 4 | 78 | 0 | 9 | 0 | 0 | 50 | 9 | 3 | 0 | 0 | 2 | 27 | 91 |
| `tests/test_uniform_manifold.py` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `tests/test_uniform_probing.py` | 30 | 160 | 0 | 70 | 0 | 2 | 131 | 59 | 6 | 0 | 0 | 15 | 47 | 260 |
| `tests/test_uniform_tucker_tensor_train.py` | 0 | 8 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| `docs/api_reference.rst` | 0 | 2 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 10 |
| `docs/batching_and_stacking.md` | 1 | 63 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 74 | 74 |
| `docs/chunking.md` | 21 | 12 | 0 | 44 | 0 | 0 | 0 | 0 | 0 | 10 | 8 | 0 | 59 | 77 |
| `docs/conf.py` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `docs/contributor/ambient_derivative_transpose_note.md` | 4 | 27 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 39 | 39 |
| `docs/contributor/batching_internals.md` | 2 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 8 |
| `docs/contributor/contractions_internals.md` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `docs/contributor/curvature_internals.md` | 4 | 25 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 32 |
| `docs/contributor/deferred_and_rejected.md` | 1 | 20 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 | 25 |
| `docs/contributor/doctest_style.md` | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `docs/contributor/fitting_internals.md` | 0 | 9 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 14 |
| `docs/contributor/numerical_contract_catalog.md` | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 9 |
| `docs/contributor/padded_frame_internals.md` | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `docs/contributor/parameters_not_closures.md` | 0 | 5 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 7 |
| `docs/contributor/precompute_and_caching.md` | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `docs/contributor/refactoring_methodology.md` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `docs/contributor/scan_body_principles.md` | 1 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 7 |
| `docs/contributor/sharing_internals.md` | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `docs/contributor/signature_style.md` | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| `docs/contributor/uniform_internals.md` | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| `docs/contributor/uniform_polymorphism.md` | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| `docs/contributor/uniform_pytree_composition.md` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `docs/contributor/uniform_rank_masks_rationale.md` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `docs/contributor/verification_records.md` | 1 | 29 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 31 | 31 |
| `docs/contributor/weighted_internals.md` | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4 |
| `docs/entries_apply_probe.md` | 17 | 111 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 135 | 135 |
| `docs/exact_hessian_two_hole.tex` | 12 | 26 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 42 | 42 |
| `docs/fitting_and_optimization.md` | 0 | 92 | 13 | 16 | 0 | 0 | 0 | 0 | 0 | 16 | 21 | 0 | 84 | 121 |
| `docs/getting_started.rst` | 0 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 4 | 0 | 9 | 21 |
| `docs/glossary.md` | 7 | 42 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 61 | 61 |
| `docs/grouped_contractions.md` | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 3 | 4 |
| `docs/index.rst` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `docs/naming_conventions.md` | 17 | 18 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 37 | 37 |
| `docs/numerical_contracts.md` | 0 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 6 | 9 |
| `docs/pad_safe_svd.tex` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| `docs/rank_continuation.md` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `docs/reading_signatures.md` | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `docs/shared_t3_math.tex` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 |
| `docs/sharing.md` | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 3 |
| `docs/symmetric_probe_derivatives.tex` | 0 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 10 |
| `docs/transposes.md` | 1 | 64 | 0 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 78 | 78 |
| `docs/ttm_t3m_ht_note.tex` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `docs/uniform_equivalence_contract.md` | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 |
| `docs/uniform_supercore_layout.md` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `docs/user_guide.rst` | 0 | 29 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 35 | 35 |
| `examples/fit_hilbert_from_apply_derivatives.py` | 5 | 22 | 1 | 4 | 1 | 0 | 5 | 1 | 0 | 0 | 0 | 2 | 23 | 32 |
| `examples/fit_hilbert_from_apply_derivatives_flat.py` | 5 | 20 | 0 | 5 | 1 | 0 | 5 | 1 | 0 | 0 | 0 | 2 | 21 | 30 |
| `examples/fit_hilbert_from_apply_derivatives_topt.py` | 4 | 14 | 0 | 1 | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 4 | 10 | 19 |
| `examples/fit_hilbert_from_entries_lbfgs.py` | 0 | 23 | 3 | 1 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 4 | 18 | 27 |
| `examples/fit_hilbert_from_probes_adam.py` | 2 | 32 | 3 | 1 | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 2 | 29 | 38 |
| `examples/fit_hilbert_from_probes_optax.py` | 2 | 10 | 1 | 1 | 1 | 0 | 6 | 0 | 0 | 0 | 0 | 1 | 6 | 14 |
| `examples/fit_hilbert_regularized.py` | 0 | 7 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 6 | 8 |
| `examples/fit_hilbert_tensor_newton_cg.py` | 0 | 20 | 2 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 19 | 23 |
| `examples/fit_hilbert_uniform_newton_cg.py` | 0 | 11 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 3 | 7 | 12 |
| `examples/fit_hilbert_uniform_probe_derivatives_newton_cg.py` | 2 | 23 | 0 | 1 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 3 | 20 | 26 |
| `examples/fit_per_mode_weight_probes.py` | 2 | 15 | 1 | 1 | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 3 | 12 | 19 |
| `examples/fit_probe_display.py` | 0 | 20 | 0 | 1 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 4 | 13 | 21 |
| `examples/fit_shared_factors_jetted_probes.py` | 1 | 10 | 0 | 1 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 8 | 12 |
| `examples/fit_varied_rank_tensor_newton_cg.py` | 0 | 10 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 8 | 12 |
| `CLAUDE.md` | 4 | 26 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 32 |
| `README.md` | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 12 |

### Names reached through an import alias

None: no file imports one of the names under another name.

### Star patterns in prose that match at least one name

Not counted for any name above. Pattern, the number of names it matches, and how often it is written, by area.

| pattern | names matched | written | where |
| --- | --- | --- | --- |
| `ut3_*` | 25 | 10 | CLAUDE.md 2, docs 5, t3toolbox 2, t3toolbox/backend 1 |
| `*_from_sweep` | 36 | 9 | docs 6, tests 3 |
| `*_derivatives` | 18 | 8 | docs 6, t3toolbox 1, t3toolbox/backend 1 |
| `*_model` | 6 | 4 | CLAUDE.md 1, docs 3 |
| `*_corewise_transpose` | 9 | 3 | docs 1, t3toolbox/backend 2 |
| `*_jacobian_derivatives_from_sweep` | 6 | 2 | t3toolbox/backend 2 |
| `*_transpose_derivatives_from_sweep` | 6 | 2 | t3toolbox/backend 2 |
| `*_corewise_derivatives_transpose` | 9 | 2 | docs 1, t3toolbox/backend 1 |
| `ut3_*_corewise_transpose` | 3 | 2 | t3toolbox/backend 1, tests 1 |
| `utv_*` | 30 | 2 | CLAUDE.md 1, tests 1 |
| `utv_*_transpose` | 6 | 2 | docs 1, tests 1 |
| `utv_*_transpose_from_sweep` | 3 | 2 | docs 1, tests 1 |
| `*_ambient_transpose` | 6 | 2 | docs 2 |
| `*_transpose` | 42 | 2 | docs 2 |
| `compute_*_jets` | 16 | 1 | t3toolbox/backend 1 |
| `assemble_*` | 12 | 1 | t3toolbox/backend 1 |
| `t3_*_corewise_transpose` | 3 | 1 | t3toolbox/backend 1 |
| `t3_*_derivatives` | 3 | 1 | t3toolbox/backend 1 |
| `*_transpose_from_sweep` | 6 | 1 | t3toolbox/backend 1 |
| `tv_precompute_*_frame_sweep` | 3 | 1 | t3toolbox/backend 1 |
| `tv_*_jacobian_from_sweep` | 3 | 1 | t3toolbox/backend 1 |
| `tv_*_transpose_from_sweep` | 3 | 1 | t3toolbox/backend 1 |
| `tv_precompute_*_frame_sweep_jets` | 3 | 1 | t3toolbox/backend 1 |
| `utv_*_derivatives_transpose` | 3 | 1 | tests 1 |
| `ut3_*_corewise_derivatives_transpose` | 3 | 1 | tests 1 |
| `tv_precompute_*` | 6 | 1 | tests 1 |
| `probe_*` | 8 | 1 | docs 1 |
| `*_derivatives_transpose` | 18 | 1 | docs 1 |
| `*_tilde` | 4 | 1 | docs 1 |
| `t3_*_corewise_derivatives_hessian` | 3 | 1 | docs 1 |
| `*_derivatives_model` | 3 | 1 | docs 1 |
| `*_tangent` | 4 | 1 | docs 1 |
| `max_*` | 1 | 1 | docs 1 |
| `ut3_*_corewise_*transpose` | 6 | 1 | docs 1 |

### Not in the inventory: the two files that name the removed names on purpose

`CHANGELOG.md` and `docs/release_notes.md` record the history of the releases, so they keep the old names, and the final search of P20 leaves them out (PART1_PLAN.md section 10). They are not counted anywhere above. Their occurrences, for the record:

| file | group | distinct names | lines | occurrences |
| --- | --- | --- | --- | --- |
| `CHANGELOG.md` | sampling | 50 | 37 | 81 |
| `CHANGELOG.md` | method | 7 | 33 | 46 |
| `CHANGELOG.md` | factory | 2 | 2 | 2 |
| `CHANGELOG.md` | other | 4 | 27 | 30 |
| `CHANGELOG.md` | **all** | 63 | 91 | 159 |
| `docs/release_notes.md` | sampling | 3 | 2 | 3 |
| `docs/release_notes.md` | method | 0 | 0 | 0 |
| `docs/release_notes.md` | factory | 1 | 1 | 1 |
| `docs/release_notes.md` | other | 2 | 5 | 5 |
| `docs/release_notes.md` | **all** | 6 | 8 | 9 |

### Notes from the scan

- docs/contributor/doctest_style.md:1: doctest examples not parsable as a whole; kinds guessed from the text
