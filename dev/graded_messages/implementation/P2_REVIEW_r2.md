SIGN OFF

# P2 review, round 2

Slice P2 of `PART1_PLAN.md`, after the implementer's `P2_RESPONSE_r1.md`. Reviewed 2026-11-17 on
`graded-messages` at `3fcfbe7`, working tree as the implementer left it (last write to a slice file 09:07:23).

**In one paragraph.** Every finding of round 1 is resolved, each checked by my own run and not from the response.
The changed interpolation (roots of unity and a Fourier transform) is right: my independent recomputation of all
37 names agrees with the changed module to 1.4e-14 at orders 0 to 5 and to 5e-15 at order 16, and the plain and
dot variables to 7e-16 at order 32. All 24 real mutations of this round's code were caught, and my two mutants of
round 1 now fail. No blocking or should-fix finding remains. Four minor findings (9 to 12) are listed; none needs
another round.

## The findings of round 1

| # | Finding | State | How I checked |
| --- | --- | --- | --- |
| 1 | The independence guard could be walked around | Resolved | My two mutants, applied to the module in a scratch copy: both now fail the test file (`mut3.log`, R01, R02). Eight ordinary roads of my own (an unaliased import, `from t3toolbox.backend import probing as p2`, a star import from `probing`, `probing._dense_jet`, `import inspect`, `dc.sys.modules[...]`, the name `contract`, `getattr`) are all seen. Three contrived ones are not: finding 9 |
| 2 | `mu_tilde`'s docstring false for apply and entries | Resolved | Read every changed sentence. Each either names its kind or holds for all three. `/tmp/p2rev/pair_apply.py` again: the pairing of `mu_tilde` with `nu_L_dot` is 0.0 for apply, `seed * mu` gives the seed paired with `Jp` (30469.0086 both). My recomputation for apply and entries (zeros; the seed times the untilded variable; the blocks with only the right-side terms) agrees with the module. `TestApplyAndEntries` fails when a statement is made false (mutant A01: `eta_D_tilde_dot` of apply without the seed factor; and N12 of round 1, `mu_tilde = seed * mu`) |
| 3 | The interpolation lost digits | Resolved, by the better scheme | Section below |
| 4 | `dense_entries` at order 0 not indexing | Resolved | `/tmp/p2rev/dense.py`: equal to indexing with `==` for `W = (2, 2)`, `C = (2, 3)`; one `inf` elsewhere now spoils 0 of 24 entries; jets unchanged (2e-14 against my polynomial arithmetic and against the old `dense_*_derivatives / t!`). Mutants: `C + W` returned, `W` axes reversed, numpy on a tracer: caught |
| 5 | A negative `order` accepted | Resolved | `order = -1` raises in the three functions, with and without `pp`; the mutant is caught; `dense_entries` without the check is caught |
| 6 | `Kp`'s route by what is installed | Resolved as Claude decided | Section below |
| 7 | No `W = ()`, one site, two sites in the tests | Resolved | Two more problems in `PROBLEMS`; the file runs 56 tests, 4,107 subtests |
| 8 | Form | Resolved | `_einsum` omits `optimize` for jax (mutant F01 caught); the return comments are in the column; the docstring says ragged only and why |

## The changed interpolation

**What it is.** `_setting` takes the `m = d + order + 1` roots of unity `exp(2 pi i k / m)` as nodes; the inputs
at the nodes are complex. `_taylor` returns `fft(values, axis=0)[:n_slots].real / m`. `_at_nodes` evaluates the
seed's polynomial `sum_t w_tilde[t] s_k^t` with complex powers. The convention is consistent: nodes with `+` in
the exponent, numpy's `fft` with `-`, so slot `t` of the transform divided by `m` is the coefficient of `s^t`.
Nothing else changed; the dense functions do not use it (they still expand over the subsets of modes).

**My recomputation against the changed module** (`/tmp/p2rev/indep.py` through `r2_run.py`; the page's
recurrences in exact polynomial arithmetic, outputs from my own dense tensors, Jacobian and double sum; nothing
of the implementer's):
- Orders 0, 1, 2, 4, 5; three kinds; d = 1, 2, 3, 4; tails with `rL_d != rR_d`; `W` of 0, 1 and 2 axes; both dot
  forms and both directions given; the seed from `y` and from a given `w_tilde`; every index; shapes exact.
  All 37 names agree. Worst relative difference 1.4e-14 (`mu_tilde_dot`); it was 8e-14 with the old nodes.
- Order 16, all 37 names, three kinds, d = 1, 2, 3, both seeds: worst 5e-15 (`r2_16.log`).
- d = 4, probe, `mu`, `eta`, `mu_R_dot`, the raw `eta_D_dot` at orders 3, 8, 16, 32: 3e-16 to 7e-16
  (`misc2.py`). The response claims 1e-14; it is better than claimed.
- Coefficients that are zero in truth: exactly 0 at order 3, at most 7e-14 against a scale of 190 at orders 8
  to 32 (4e-16 relative). The claim holds.
- The hand-written einsums, the old-library comparison and the dense-function check of round 1, rerun: unchanged
  (4e-15, 7e-15, 2e-14).

**The seed's polynomial, from first principles** (`slot.py`, rerun on the changed module): with no turn and no
complex number in my code, the sensitivity of `sum over modes of <weight^2 r, z>` to slot `m` of the other side
of a bond equals `mu_tilde[i][order - m]` and `nu_tilde[i][order - m]` at every bond and slot (5e-16 relative).
So the seed is evaluated at the nodes themselves, not at their conjugates, and the mutant with conjugate nodes
in the seed fails.

**What complex nodes do.**
- *Dtype.* Real in, real out: float64 in gives float64, float32 in gives float32 for all 37 names (worst
  difference to float64 8e-7), with complex64 inside. No complex array comes out of any function.
- *jax.* jax arrays in (x64 off) give real jax arrays for all 37 names, probe and apply, within 3e-6 of numpy.
- *The real part.* It is taken in one place, the last line of `_taylor`. What is dropped is rounding: 1e-13
  against 190 at order 16. It cannot hide a wrong degree (aliasing lands in the real part: the mutant with one
  node too few fails), a wrong sign of the exponent, conjugate nodes, a rotated or scaled set of nodes, or a
  real part taken early: all of those fail (below). It does hide complex cores: finding 10.
- *Mutations of the interpolation* (scratch copy, no bytecode; `mutate3.py`, `mut3.log`), all caught: one node
  too few; the transform not divided by `m`; conjugate nodes; the inverse transform; the real part of the nodes
  in the inputs; the seed's polynomial at conjugate nodes; the real part of the seed's powers; the modulus in
  place of the real part; nodes rotated by half a step (which scales slot `t` by `cos(pi t / m)`); nodes in
  single precision beside float64 (an error of 8e-9, caught at 1e-10); nodes on a circle of radius 1.001; the
  seed's powers decreasing. One node too many passes, as it must (it is still exact). A mutant that adds 1e-6
  of the dropped imaginary part to the result passes, because that part is rounding: it is not a real mutant.

## `Kp` by the two routes

- Both against my exact double sum over pairs of cores (`kp2.py`; d = 2 and 3 with tails, three kinds, orders
  0, 2, 4): finite differences 2.7e-15; automatic differentiation on jax arrays in double precision 1.8e-15.
- numpy in never touches jax: with the module's `jax` replaced by an object that raises on any use, all 37
  names for the three kinds at order 2 run and return numpy. `tests/test_jax_absent.py` passes too.
- jax in gives jax arrays, also when only the direction is a jax array. The process's x64 setting is unchanged.
- float32 jax arrays: `Kp` comes back float32, within 6e-7 of the float64 result. That is the accuracy of the
  inputs, and the docstrings of `Kp` and of `_Kp_by_jax` say "in the precision jax is set to". Enough.
- Mutants: numpy sent to jax, the route read off the frame only, jax sent to finite differences: all caught.

## New findings (all minor)

**9. The guard still does not see three contrived roads.** `independence_violations` passes
`probing.dense_probe.__globals__['t3_probe']`, `__builtins__['__imp' + 'ort__'](...)`, and a call of `_` after
`xnp, _, _ = get_backend(...)` (the second `_` is the scan). None is something an edit would do by accident, and
the 23 ways of the self-test and my eight ordinary ones are seen. If wanted: add `__globals__`, `__builtins__`,
`__code__` to the forbidden names, and refuse `_` as a name that is read.

**10. Complex cores are silently made real at order 1 and above.** With `D + 0j` the functions return float64
with no error; at order 0 they return complex. The module docstring lists "Complex arrays" under "What is not
here", which is enough for a library whose tensors are real; one more clause ("their imaginary part is dropped
without an error") would make it exact.

**11. Nothing tests the size of the dropped imaginary part.** It is rounding today (above), and every error I
could put into the interpolation showed in the real part. A line in a test that the imaginary part of the
transform is below 1e-10 of the real part would turn "which is rounding" from a comment into a checked fact.

**12. Mixed precision.** float64 cores with a float32 sample give a float64 array that is right to 9e-8 (the
nodes follow the sample's precision). It was the same before this round, and the error is that of the float32
inputs. For P10, which decides the dtype rule of the engine: the reference does not follow "the lowest
precision wins" in its dtype, only in its accuracy.

## The gate

The times agree with the response: the last write to a slice file is 09:07:23 (`tests/test_sampling_definitions.py`),
the gate's `PID` file is 09:13:40 and its `DONE` 09:36:11; only the response, the handover and Claude's notes in
`SAMPLE_VARIABLES.md` and `PART1_PLAN.md` (text under `dev/`, no code) are newer. `/tmp/gate_P2_r2/results.tsv`
has every step PASS: suite 1,501 passed, 5 skipped, 1 xfailed, 65,157 subtests; 282 module doctests; quickstart;
48 pages; 14 examples; sphinx with 0 warnings. I saw no reason to run the whole gate again. I ran, in the tree,
with no bytecode written and no pytest cache:
- `tests/test_sampling_definitions.py`, `tests/test_jax_absent.py`, `tests/test_import_graph.py`,
  `tests/test_tucker_tensor_train.py` and `test_jit_dense_sampling_functions`: 145 passed, 28,376 subtests;
- the probe tests of `tests/test_manifold.py`: 2 passed;
- the module doctests of `sampling_definitions`, `probing`, `apply`, `entries`, `manifold`, `uniform_manifold`:
  74 passed.
And the test file alone in the scratch copy: 56 passed, 4,107 subtests.

## Claude's notes in the planning documents

Read; both say what the review found. `SAMPLE_VARIABLES.md` section 8 now says for which kind each pairing
holds, and section 9 has the table of the names of a corewise call, marked as awaiting the maintainer before P8.
`PART1_PLAN.md` has the corrected sentence on `Kp`'s route in two places. Nothing in the slice depends on the
table.

## State of the tree

Unchanged apart from this file (`git status` before and after my runs is the same; no worktree; nothing left
running). My scripts and logs are in `/tmp/p2rev/`; the scratch copy of the tree there is removed.
