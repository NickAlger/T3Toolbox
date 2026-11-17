NOT YET

# P4, review round 1

Reviewer, 2026-11-17, on the working tree of `graded-messages` (library files last changed 13:47:34). One
should-fix finding (1); everything else is minor or a record for Claude and the maintainer. The mathematics is
right, the accuracy step is right, the tiers are respected, and every real mutation but one was caught.

## Findings

**1. Should fix. A norm that overflows makes an inconsistent frame read 0.0 (pass).**
`backend/fv_operations.py`, `_t3_left_orthogonal_norm` (`sqrt(sum(last * last))`), used for every numerator and
denominator.
- Evidence (`/tmp/p4rev/d_overflow.py`): a float32 frame of norm `1e20` whose `R[0]` is off by a relative `1e-3`:
  `fv_frame_consistency_residual = 0.0`, `T3Frame.is_consistent() = True`; the centers' residual with a wrong
  `Vc[1]` is `0.0` too. The denominator's square overflows to `inf` while the numerator's does not, and
  `num / inf = 0`. float32 overflows from norm `1.8e19` (representable up to `3.4e38`); float64 from `1.3e154`.
  Same on jax and under `jit`.
- For a numpy float32 frame this is a regression: the code at `HEAD` returned `1.0e-3` for the same frame (its
  dense route ran in float64). For jax float32 and for float64 the old code overflowed in the same way.
- The docstring says such input gives "`nan` or `inf`", "which fails every threshold". It gives 0.
- Resolve: a norm that does not overflow (divide by the largest magnitude first, with a `where` for zero, which
  works under `jit`), or `nan` wherever a norm is not finite; a test with a float32 frame of norm `1e20`,
  consistent and not; the docstring corrected.

**2. Minor. The `1` of `max(1, ||T(D, R)||)` in the frame residual is pinned only at the zero frame.**
Mutation `max(1.0, den)` to `max(1e-3, den)` in `fv_frame_consistency_residual` passes every test (the centers'
twin of this mutation is caught, by the `d = 1` structure). Resolve: one hand-made frame of norm below 1 in
`test_it_is_the_definition_on_hand_made_frames`.

**3. Minor. The test of the guard hangs instead of failing when the guard is broken.**
With the zeroing removed (or `all` changed to `any`), `test_cores_that_are_not_finite_give_nan_and_no_error` does
not return: numpy's SVD on an `inf` never comes back and ignores `SIGALRM` (confirmed, `/tmp/p4rev/hang.py`). In
CI that is a stalled worker, not a red test. Resolve: run the `inf` cases in a subprocess with a time limit, or
accept and say so.

**4. Minor. Cost is understated in the docstrings.**
- "Three orthogonalizations of core-sized arrays": the third is at doubled Tucker and TT widths. Measured
  (`d = 8`, `N = 2000`, `n = 60`, `r = 40`): building the frame 0.15 s, one left orthogonalization 0.06 s, the
  frame residual 0.90 to 0.98 s; at `d = 6`, `N = 500`, `n = 100`, `r = 80`: 0.77 s against 6.5 s. Fine for a
  check made once, but P16 should know it costs six to eight frame constructions.
- The centers' residual on the same two frames: 13.6 s and 75 s. The docstring does say `d**2`.
- "whatever the mode sizes" (three docstrings): the cost is linear in `N`, not independent of it.
- Since both terms share `D`, down-orthogonalizing once and taking the difference of the two TTs (bonds doubled,
  Tucker edge not) gives the same accuracy (`1.5e-14` at norm `4e14`, identical on a hand-made frame to 15 digits)
  in 0.42 s where the built route takes 0.97 s (`/tmp/p4rev/l_alt.py`). Optional.

**5. Minor. `nan` or `inf` in the padding of a uniform frame gives `nan`,** while the twins' docstrings say
"garbage in the padding does not reach the result". The cause is the library's multiplicative masking
(`nan * 0`), which the handover notes; the docstrings should say "finite garbage".

**6. Minor. The `LinAlgError` branch differs by backend and can mislabel.** On numpy an overflow in one stack
element gives `nan` for the whole stack (documented); on jax only that element. Any other `LinAlgError` (an SVD
that does not converge on finite input) also becomes `nan`, which a frontend would report as "inconsistent". No
change needed; record it.

**7. Minor, not verified here (no numpy 1.x in the workspace).** For an unstacked float32 numpy frame,
`num / xnp.maximum(1.0, den)` and `xnp.where(finite_C, ..., xnp.nan)` mix a float32 scalar with a Python float;
under numpy 1.x scalar promotion that is float64. No test covers it (the float32 tests use the stack `(3,)`), so
the floor leg would not fail, but the dtype promise would not hold there. The old code had the same `maximum`.

**8. Minor. The wall-clock assertions (`< 0.5 s`) add risk and prove nothing more.** With both cores busy
(`run_gate.sh -n 2` running) I measured at most 0.004 s (frame) and 0.052 s (centers), so a tenfold margin; not
likely to flake. But the tests already prove "no dense tensor" by returning at all (`1e24` entries cannot be
allocated). Keep, or drop the clock.

**9. Minor. Duplication forced by the import graph.** `_t3_core_direct_sum` is the block of `t3_linalg.t3_add`,
and `_t3_left_orthogonal_norm` is the square root of `geometry.t3_left_orthogonal_norm_sq`; neither module can be
imported from `fv_operations`. The plan names `t3_operations` for the difference, but the addition lives in
`t3_linalg`, so the route as the plan words it did not exist. Nothing to change in the slice; a note for the
plan.

## For Claude and the maintainer (no change asked of the implementer)

**10. The accuracy step.** Reproduced with my own code: the literal route gives `1.2e-9` at norm `4e6`, `1.0e-6`
at `4e9`, `0.35` at `4e14` (largest of 20 seeds); the built route at most `1.6e-14` at every norm from `4e-6` to
`4e14`, the centers at most `1.8e-14`. float32: `2e-7` to `3e-7` on numpy, `8e-7` to `1.4e-6` on jax, the same
under `jit`, with and without tails. Rank-deficient cores, cores scaled by `1e150` and `1e-150` in neighbours, a
stack whose elements have norms `1e-12` to `1e12` with one zero element: all at rounding level. I found no case
where the new route is worse, apart from finding 1, which the literal route shares. It is within the plan (the
same routine, applied once more per term) and is the right fix.

**11. The centers' consistency.** The statement is what `SAMPLE_VARIABLES.md` section 1 and the plan's 2.4 say,
and it is exactly the tensor behind "`z[i]` is `eta[i]` through `Vc[i]`". Verified densely (my own einsum, not
`t3_to_dense`): 216 hand-made cases to `5e-15`; library frames and centers with `squash_tails` true and false.
What the decisions do not fix, and should be recorded as decided by someone:
- the denominator `max(1, ||T(D, L)||)`: below norm 1 the residual is absolute, so a tensor of norm `1e-6` with
  centers wrong by `1e-3` passes `rtol = 1e-9`. The frame residual has the same property, kept from before;
- the maximum over sites and families; a family that is `None` is skipped; `(None, None)` raises `ValueError`
  (structural, consistent with section 7.1);
- row 4.1's note calls the check "core-sized" and guesses it "needs an orthogonal frame". What was built needs no
  orthogonal frame and costs `d**2` core operations (finding 4). `R[0]` is not read by the definition.

**12. `UT3Frame.is_consistent` changed meaning** from `<= rtol * ||T||` to `<= rtol * max(1, ||T||)`: looser
below norm 1. It also became more exact (the old subtraction read `3.9e-10` on a consistent frame of norm `4e6`,
the new residual `4e-15`). Public behaviour; no caller in the library, the examples, the pages or T3Polynomial,
only tests. The plan says it moves to the twin, so it follows from the plan, but the maintainer was not told in
those words; it belongs in the log and, at P19, in the changelog. `T3Frame.is_consistent` keeps its formula (its
old docstring said `rtol * ||right||`, the code always had the `max`).

**13. `(x - y).norm()`.** Confirmed: `3.9e-10` for one tensor in two gauges at norm `3.7e6`. Outside the slice.
Worth recording for the maintainer, with `_t3_difference_norm` as the remedy and a public name in `t3_linalg` as
the question. The other private helpers are rightly private.

**14. For P16.** `safety.effective_rtol` gives `1e-9` for a numpy frame whatever its dtype, so a float32 numpy
frame fails (`2.5e-7`); it fails `is_orthogonal` today for the same reason (`1.2e-7`), so nothing new. The
enforcement must pass `effective_rtol(frame.data)` and not rely on the default `rtol`.

**15. The uniform twins by delegation.** Sound: zero padding changes no tensor, and an SVD of a zero-padded
matrix reproduces it whatever the completion vectors are. Verified on frames I built with ranks drawn per stack
element, tails, garbage of size `1e3` in every padded slot, prefix masks and gappy masks: 792 comparisons with
the ragged functions on the real cores and with my dense definition, `5e-15`; garbage against clean padding bit
for bit; a mask widened by one slot changes the answer; a traced mask is refused. Under `jit` the steps are
unrolled: compile time for the centers 0.7 s (`d = 4`), 1.7 s (`d = 8`), 3.8 s (`d = 12`), growing like `d**2`;
the frame residual 0.4 to 0.7 s. The equivalence contract names "no loop unrolling" as a purpose of the uniform
layer, so this is a departure in style, acceptable for a checker that the frontend never calls under a trace.

## Confirmed

- Nothing under `t3toolbox/backend/` imports `safety` or reads a mode. The only new `raise` is `(None, None)`.
  No enforcement added, no geometry member, `geometry.py` and `safety.py` untouched.
- The guard: `nan`, `inf`, `-inf` in every store and both centers, at every site, in one stack element, on numpy,
  eager jax and under `jit`: that element is `nan`, the others keep their value, 54 cases, none unexpected. It
  uses `where`, no branch on values. It costs 0.009 s on a frame whose residual takes 0.9 s.
- The frame classes: only the four hunks of the diff. The cached property adds no leaf, leaves the treedef and
  its hash unchanged, is not carried by `dataclasses.replace` or `tree_map`, and a jitted function of the frame
  compiles once across frames.
- Names and signatures as sections 2.4, 2.5 and 11.3. `t3_linalg` is not imported.

## Mutations (58, in a scratch copy, `python -B`)

Caught: `R` against `R`; the denominator from the other store; first or last site skipped; either tail not
summed; first site, last site, mean or minimum in place of the maximum; normalization dropped (both); `max(1, .)`
dropped (both); sum in place of difference; `L` and `R` swapped around a center; `Vc`, `Hc` or `U` ignored; the
base point through `R`; the norm taken over the stack; **the pre-orthogonalization removed** (frame and
centers); frame, `D` alone, `U` alone, `Vc`, `Hc` unmasked; variation masks unshifted; `nan` not set; the `try`
removed; its dtype; the guard reading one family; `n_stack` wrong; the dtype of the direct sum; both properties
uncached; the cache bypassed; `rtol` ignored (both); `L` and `R` swapped in the twin; `np` on a tracer in three
places; `import jax` in the module.

Not caught: finding 2. The guard removed is caught only as a hang (finding 3). Two more were equivalent mutants.

## What I ran

- My own scripts, in `/tmp/p4rev` (`a_math`, `b_acc`, `c_edge`, `d_overflow`, `e_uniform`, `f_guard`,
  `h_front`, `j_compile`, `l_alt`, `t_timing`).
- `run_gate.sh --detach -n 2 -l /tmp/gate_P4_review quick` on the new file, the two frame test files,
  `test_import_graph.py`, `test_jax_absent.py`, `tests/backend/test_fv_operations.py`, `test_dispatch.py`, on the
  final tree: 254 passed, 1,105 subtests, 591 s.
- `pytest --doctest-modules` on the four touched modules: 35 passed.
- The gate: file times agree with the logs (tests last changed 13:08, the two backend files at 13:47:34, between
  `/tmp/gate_P4` ending 13:47:12 and `/tmp/gate_P4_b` starting 13:47:39). The rendered pages of the two sphinx
  builds differ in exactly the two summary lines. I did not rerun the whole gate and see no reason to.
- The tree is as I found it apart from this file; the scratch copies are removed.
