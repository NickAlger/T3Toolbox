SIGN OFF

# P8 review, round 1: Engine D (the raw dots, the tilde-dot grade, `Kp`, the corewise point)

Reviewer, 2026-11-20, on the working tree of `graded-messages` (the implementer's uncommitted changes on `2aee3c1d`).
No blocking and no should-fix finding; one minor, and one note on memory.

`Kp` agrees with a dense second derivative I computed myself from its definition, with tails and jets in two
variables. The corewise point agrees with the manifold call at `(D, L)`, and runs every step at most `d` times. Every
structural refusal of the design is in place.

## My checks (`p8_check.py`)

The problem: `d = 3`, with tails at both ends where the `L` and `R` end bonds differ, at `()`, `(2,)` with mixed input
sizes `(2,)`, `(3,)`, `(1,)`, and the jet matrix's mixed modes at `(1, 2)`; a nonuniform weight; `K = (2,)`.

1. **`Kp` against a dense second derivative, independent of the engine and the reference.**
   - *The objective:* `Φ(x) = Σ_i Σ_k ω_ik² <r_ik, z_ik(x)>`, with the residual held, `z` by my roots-of-unity oracle
     on `TuckerTensorTrain(D, L).to_dense()`.
   - *The method:* for each entry `e` of every `D[j]` and `L[j]`, the `ε¹` coefficient of
     `Φ(x + ε ẋ + e) - Φ(x + ε ẋ)`. `Φ` is linear in that one entry, and a polynomial of degree at most `2d` in `ε`
     (it has `2d` cores), so it is interpolated exactly on `2d + 1` points.
   - *The result:* every entry of `Kp = (V_tilde_dot, H_tilde_dot)`, for each `k`, agrees to 8.8e-15. This checks the
     raw dots, the tilde-dot grade, its short node and the nine `H` readouts end to end.
2. **The corewise call at `(D, L)` against the manifold call:** `mu`, `nu = nu_L`, `eta = eta_D`, `z`, `Kp`,
   `nu_dot = nu_L_dot` (raw), and `eta_tilde = eta_D_tilde`. All agree to rounding.
3. **Corewise `JtJp` against `g`** with `Jp_k` held as `r`, per `k`. They agree, which also shows that the corewise
   twins are the renamed path's.
4. **Each sweep once.** On a corewise call wanting `('z', 'g')` and `('Kp',)`, with every step function of
   `sampling_steps` counted, every step called more than once is called exactly `d` times. In particular `nu_step` and
   `eta_step` are `d` each in `('z', 'g')`: one `nu` sweep, through `G`.
5. **The refusals,** each a `ValueError`:
   - a `K` of 3 on `D_dot` beside a `K` of 2 on `V`, the operand named;
   - a basis name (`nu_L`) on a corewise call (O-13);
   - `variations` on a corewise call, with the message "pass it as `core_directions`".

## The bridge to the old `corewise_hessian`

The conversion is the one I derived in the design review. The old kernel is the Hessian of
`Σ_t <c_t, t! z_t(x)>`, and with `c_t = ω_t² r_t / t!` forward that is `Σ_t ω_t² <r_t, z_t(x)>`, the engine's
objective with the seed held. `test_Kp_against_the_old_corewise_hessian` asserts it at `K = ()` and `(2,)`, and it
passes.

## The departures

1. **`_jp_twins` takes the call type** and skips rows needing the other call's. Right. Without it the corewise-only
   routes of `V_tilde` and `H_tilde` would grow twins on the manifold graph that no manifold call can take.
2. **`HeldVariables` reordered,** manifold names then the corewise extras, 40 slots. Fine: slots are read by name.
3. **The `K` message names the operand `K` was read from** (`V[0]` or `D_dot[0]`). Right: either can be first, and the
   message tells the user which one the others are compared with.
4. **The mutant note on "`needs` not rewritten".** Correctly handled. It was equivalent because no surviving row needed
   `variations` (P7's `JtJp` gets its need from `_jp_twins`, after the rename). The new import-time test adds a
   surviving row needing `variations` and asserts it comes out needing `core_directions`, so the rewrite is now
   pinned for P9 to P11. That is the right response to an equivalent mutant: a test of the property that will matter
   later, not a weakened claim.

## Minor finding

**1. The import-time invariants of `_corewise` are `assert` statements.** The rename never lands on a surviving name,
and the corewise twin path is complete. Under `python -O` both are stripped. The graph is static and the tests run
without `-O`, so nothing is wrong today, but these are structural guards meant to catch a future row. *Resolve:*
`if ...: raise RuntimeError(...)` (or `ValueError`) with the same message. Two lines.

## Note on memory (not a finding)

The gate's suite peaked at 13.4 GB (P7: 10.4 GB). I measured P8's own test files in one process: the four engine
files, the new corewise file, the bridge, the import graph, the new dispatch test and the module doctests. 74 passed,
in 56 s, with a **maximum resident set of 0.68 GB**. So the rise is not one of P8's tests on its own. It is either the
three workers together, or a test elsewhere landing beside another by xdist's distribution. Worth a look before P10's
stack matrix adds work, as the handover says. A run of the suite at `-n 3` under `/usr/bin/time -v`, per worker, would
show which.

## What I ran (single process)

- `p8_check.py`: all OK, worst 8.8e-15.
- **Current env:** the files above with `--durations` and `/usr/bin/time -v`. 74 passed, 5,335 subtests. The largest is
  the corewise by-definition test, at 12 s.
- **Floor env (`tttt`):** the four engine files, the corewise file and the bridge, 63 passed in 143 s.

The working tree is as I found it, apart from this file. Nothing of mine is left running.
