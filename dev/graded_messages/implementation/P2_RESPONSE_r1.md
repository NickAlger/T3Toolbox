# P2: response to review round 1

To `P2_REVIEW_r1.md` (verdict: not yet, two should-fix findings, six minor) and to Claude's decisions on the
points the reviewer routed onward (finding 6, and the two decisions (a) and (b)). 2026-11-17. Everything is
accepted; nothing is refused. The handover is brought up to date.

## Should fix

### 1. The guard of the module's independence could be walked around. Accepted.

`tests/test_sampling_definitions.py`. The guard is now a function, `independence_violations(source)`, which
returns what a source reaches of the library beyond what it may. The source may:

- `from t3toolbox.backend.common import *`, and use of it only `NDArray`, `jax_available`, `get_backend`,
  `tree_contains_jax`, `tree_to_jax`;
- import the three sampling modules under their own names (`import t3toolbox.backend.probing as probing`, and so
  for `apply`, `entries`), and use each name only as the value of its one allowed attribute
  (`probing.dense_probe`, `apply.dense_apply`, `entries.dense_entries`). Every `ast.Name` with one of those three
  ids that is not the value of such an attribute is a violation, so the module cannot be passed to `getattr`,
  renamed, or looked into;
- import, outside the library, only `math`, `typing`, `dataclasses`, `numpy`, `jax`;
- call `get_backend` only as `xnp, _, _ = get_backend(...)` (it also hands out a map and a scan).

It refuses any other import of the library (a name imported *from* a module, another module, an allowed module
under another name or under none, a relative import), and the names `getattr`, `vars`, `globals`, `locals`,
`eval`, `exec`, `compile`, `__import__`, `importlib`, `sys`, `__dict__`, `__getattribute__`, `modules`,
`t3toolbox`, beside `contract`, `jet_contract`, `contractions`, `safety`, `xscan`, `xmap` as before. (The module
itself had one `getattr`, for jax's x64 context; that went with finding 6.)

Two tests use it:
- `test_the_module_calls_nothing_of_the_library_but_the_dense_functions`: no violation in the module.
- `test_the_guard_of_independence_sees_every_way_around_it` (new): the module's own source with one line added,
  for 23 lines, each a different road to library code, and each is seen. The reviewer's two are among them.

**The reviewer's two mutants**, applied to the module itself and run against the test file (bytecode off):

    from t3toolbox.backend.probing import t3_probe as _old_kernel   (and used)
        FAILED TestStructure::test_the_module_calls_nothing_of_the_library_but_the_dense_functions   1 failed, 55 passed
    _old = getattr(probing, 't3_' + 'probe')
        FAILED TestStructure::test_the_module_calls_nothing_of_the_library_but_the_dense_functions   1 failed, 55 passed

Both passed before; both fail now.

### 2. `mu_tilde`'s docstring was false for apply and entries. Accepted, and every such statement gone through.

Each statement below was checked by a script before it was written, and the script is now a test class,
`TestApplyAndEntries` (4 tests, all three problems with apply and entries, orders 0, 1, 3):

| What the docstrings now say | Checked by |
| --- | --- |
| apply, entries: `xi_tilde`, `mu_tilde`, `xi_tilde_dot`, `mu_tilde_dot` are zero (exactly, with the group `T`) | `test_they_are_zero_or_the_seed_times_the_untilded_variable` |
| apply, entries: `nu_tilde`, `nu_L_tilde`, `eta_tilde`, `eta_D_tilde` are the seed times `nu`, `nu_L`, `eta`, `eta_D`; `nu_L_tilde_dot`, `eta_D_tilde_dot` are the seed times the raw `nu_L_dot`, `eta_D_dot` (products of polynomials in `s`) | the same |
| apply, entries: `V_tilde` is the one term `V_term(eta_tilde, w)`; `H_tilde` the one term with `nu_tilde`; `V_tilde_dot` the one term with `eta_D_tilde_dot`; `H_tilde_dot` the three terms with the tilde on `nu_L` | `test_their_blocks_have_only_the_terms_with_the_seed_on_the_right` |
| `mu_tilde` multiplies `nu_L_dot`, and `xi_tilde` multiplies `eta_D_dot`, **for the probe kind only**. For apply and entries those pairings are zero, and what multiplies `nu_L_dot` and `eta_D_dot` is the seed times `mu` and the seed times `xi`, which have no names | `test_two_of_the_four_pairings_are_of_the_probe_kind_only` (the pairing is exactly 0.0; the seed times `mu`, `xi` reproduces the seed paired with `Jp`), with `TestPairings` for the probe kind |
| `nu_tilde` multiplies `mu_R_dot`, and `eta_tilde` multiplies `xi_U_dot`, for every kind | `TestPairings` (already there) |
| every tilde-dot variable is the derivative of its tilde variable along `(D_dot, L_dot)` with the seed held, for every kind | `test_a_tilde_dot_variable_is_the_derivative_of_its_tilde_with_the_seed_held` (a fourth-order difference, `1e-8`) |
| the recurrences of the tilde and tilde-dot grades hold for every kind, with the terms that are zero for apply and entries named | `TestRecurrences` (already there, for the three kinds) |

Changed docstrings: `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `nu_L_tilde`, `eta_D_tilde`, `xi_tilde_dot`,
`mu_tilde_dot`, `nu_L_tilde_dot`, `eta_D_tilde_dot`, `H_tilde`, `H_tilde_dot`; and the module docstring, whose
`_tilde` and both-marks bullets now say what each variable is for apply and entries, and which has a new paragraph,
"The multipliers", saying which two pairings hold for every kind and which two for the probe kind only. Each
sentence either holds for the three kinds or names its kind. (`V_tilde` and `V_tilde_dot` already said that their
second term is the probe kind's.)

For Claude, as the reviewer says: section 8 of `SAMPLE_VARIABLES.md` should say the same when P19 writes the page,
and P9's identity tests cannot assume the two probe-only pairings.

## Minor

### 3. The interpolation lost digits with the order. Accepted: switched to the roots of unity.

**The choice: the better-conditioned scheme, because it was a small and clear change.** Three places changed, and
the code is not harder to read:
- `_setting`: the nodes are the `m`-th roots of unity, `exp(2 pi i k / m)`, `m = d + order + 1`, in place of the
  Chebyshev nodes; the inputs at the nodes are then complex. A helper `_complex` makes host constants into a
  complex array of the right precision (complex64 beside float32) and array module.
- `_taylor`: the coefficients are the discrete Fourier transform of the values divided by `m`, real part
  (`fft(values)[:n_slots].real / m`), in place of the product with an inverse Vandermonde matrix. Its docstring
  derives this in three lines.
- `_at_nodes`: the powers of the nodes that evaluate the seed's polynomial are complex.

Measured against exact arithmetic (the recurrences in explicit loops), `d = 4`, `mu` and `nu_tilde` at every bond,
probe and apply, worst relative error:

| Order | Before (Chebyshev, per the review) | Now |
| --- | --- | --- |
| 3 | 1e-14 | 2e-15 |
| 8 | 2e-12 | 5e-15 |
| 16 | 2e-8 | 1e-14 |
| 32 | not measured | 1e-14 |

Coefficients that are zero in truth come out at 2e-16 relative at every order (5e-7 at order 16 before). The
module docstring's "Jets" paragraph says how the coefficients are found and what accuracy to expect ("about
`1e-14` at order 3 and at order 32 alike"), and that the module is still for small problems because the cost
grows with the order. One thing given up, and said under "What is not here": complex cores (the real part is
taken; the library's tensors are real).

### 4. `dense_entries` at order 0 indexes. Accepted.

`backend/entries.py`: at order 0, `T[..., index[0], ..., index[d-1]]` and a `moveaxis` from `C + W` to `W + C`;
no arithmetic. With jets it goes through the unit vectors as before. The docstring says both, and that with jets a
non-finite entry anywhere spoils every sample. `T` is passed through `xnp.asarray` first, so a host numpy tensor
indexed by a traced index works under `jit` (the dispatch test has that case). Tests: with one `inf` and one `nan`
at positions next to an indexed one, the entries come back exactly as stored; `W = ()`.

### 5. A negative `order` raises. Accepted.

`probing._check_pp_and_order(pp, order, name)`: `order < 0` raises, then the check of `pp` against `order` that
was there. `_dense_jet` calls it, and so does `dense_entries` (which no longer reaches `_dense_jet` at order 0).
All three functions, with `pp` and without it: tested. The three docstrings' "Raises" say so.

### 6. `Kp`'s route. Done as Claude decided: read off the arrays.

`_Kp`: finite differences when every array it reads is numpy (the stores `D`, `L`, the sample, the seed, the
direction); automatic differentiation when one of them is a jax array. So a numpy result no longer depends on
whether jax is installed. `_Kp_by_jax` lost its x64 context and its conversion back to numpy (and the module its
`getattr` on jax): it computes in jax, in the precision jax is set to. Docstrings of `Kp` and of the module say
so.
- `test_jax_against_finite_differences` is kept: jax arrays, made and differentiated inside a double-precision
  context that the *test* opens (no setting of the process changes), against the finite differences of numpy
  arrays, `1e-10`, for the three kinds, orders 0, 1, 3, the five problems.
- `test_the_route_of_Kp_is_read_off_the_arrays` (in place of `test_which_of_the_two_Kp_takes`): with numpy arrays
  `Kp`, `V_tilde_dot` and `H_tilde_dot` equal the finite differences exactly and `_Kp_by_jax`, patched to raise, is
  not reached; with jax arrays, and with only the direction a jax array, the result is jax and the finite
  differences, patched to raise, are not reached.

### 7. `W = ()`, one site, two sites. Accepted.

Two more problems in `PROBLEMS`: `two sites, W = ()` (d = 2, no sample axis) and `one site` (d = 1, tails on both
ends of unequal size). Every test that runs over the problems now runs over five: the outputs against dense
tensors, the five recurrence tests, the readouts, the pairings, `TestApplyAndEntries`, jax against finite
differences. All pass with no change to the code.

### 8. Form. Accepted.

- **No `optimize=` on the jax path.** A helper `_einsum`: numpy gets `optimize='greedy'`, jax gets no argument.
  (Measured first that jax orders a 19-operand string in no time, so nothing is lost.)
- **The return comment is in the arguments' column** on all 37 functions (the three that return a pair of tuples
  have their two lines there).
- **Ragged only**: "What is not here" says that the module is for the ragged layout only and that a uniform
  `.data` must not be passed, with the reason (four entries, read as four stores, no error).
- Einsum strings built in code, and `i` before `order`: noted by the reviewer as not needing a change; none made.

## Claude's decisions

- **3(a), the zeros for apply and entries: kept**, and documented as finding 2 has it.
- **3(b), the four-store frame and the manifold names: kept.** Nothing changed.

## Testing the changes of this round

Eight more mutations, of the code this round changed, each run against the test file with bytecode off, each
caught: one node too few (aliasing); the transform not divided by `m`; the seed's polynomial at the conjugate
nodes; numpy arrays sent to jax for `Kp`; the route read off the frame only; `dense_entries` at order 0 returning
`C + W`; `dense_entries` at order 0 by unit vectors again (caught by the non-finite case); a negative order
accepted.

## Not accepted

Nothing.

## The gate

The whole gate, detached, two workers, on the final tree of this round, with the bytecode cache removed first:
it started at 09:13:40, and every file of the slice was last written before that (the latest at 09:07:23); none
changed afterwards (only this note and the handover).

    dev/graded_messages/implementation/run_gate.sh --detach -n 2 -l /tmp/gate_P2_r2
    dev/graded_messages/implementation/run_gate.sh wait /tmp/gate_P2_r2

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1501 passed, 5 skipped, 1 xfailed, 1 warning, 65157 subtests passed    937.7      3637       4256
    doctest-modules     PASS   282 passed                                                     24.5      1208       6529
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 1.3       226       7408
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               10.5       467       7227
    examples            PASS   14 examples, 0 failed                                           188       370       7352
    sphinx              PASS   0 warning/error line(s)                                       188.9       702       6888
    logs: /tmp/gate_P2_r2

No failure. Five more tests than the round-1 gate (1,496 passed): the four of `TestApplyAndEntries` and the
self-test of the independence guard. The skips, the expected failure and the warning are the baseline's.

Before it, with a clean cache and bytecode off: the slice's test files (`tests/test_sampling_definitions.py`,
`tests/test_jax_absent.py`, `tests/test_import_graph.py`, `tests/test_manifold.py`,
`tests/test_tucker_tensor_train.py`, and of `tests/test_dispatch.py` the dense test and every class but
`TestDispatch`): 231 passed, 28,632 subtests. The module doctests of the six touched modules: 74 passed.
