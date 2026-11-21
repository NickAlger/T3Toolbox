SIGN OFF: both round-1 findings are fixed and tested, the nit and the record are done, and the full gate is green.

# P12a review, round 2

Reviewer, 2026-11-20, on `P12a_RESPONSE_r1.md` and the updated tree (uncommitted, on top of `e2623c75`). Everything
was run single-process. The mutant copies have been deleted, and the working tree is untouched apart from this file.

## Finding 1, the core sizes of the directions: fixed

**The check.** `_check_stacks` now compares the trailing core axes of `V`, `H`, `Vc`, `D_dot` and `L_dot` with the
sizes the stores give, by tuple equality:
- on a corewise point it uses the point's own sizes;
- on uniform, the supercores' padded sizes;
- a count of cores other than `d` raises too.

**My two silent cases** now raise a `ValueError` that names the operand:
- `L_dot[1]` at `rL = 1` against 2 (`t3_apply_corewise_hessian`);
- `V[0]` at `nU = 1` against 3 (`tv_apply`).

A slice shorter than the expected axes also fails the comparison, so it cannot pass by accident.

**Mutant.** My mutant switching the comparison off is killed in 62 subtests of `TestTheCoreSizesOfTheDirections`.
That class covers each operand alone, every core axis cut to size 1, manifold and corewise, ragged and uniform.

**Where it sits.** It is inside `_check_stacks`, so it is removable with that function, pending O-18. That matches the
ruling that it is the same structural rule as the stack checks. If O-18 removes the stack checks, this goes with them
by design.

## Finding 2, the transposes with a residual carrying `K`: fixed

`TestAResidualWithAStack` covers all six transposes at `K = (2,)` and `(2, 3)`, summed and unsummed. Per element it
checks the call against that element's residual, plus the element's adjoint identity.

**Mutants.** My three surviving mutants (each `t3_*_corewise_transpose` rerouted to `g` from a held `r`) are now
killed by `test_sampling_functions.py` alone.

## The nit and the record

- The handover's claim about the guard's coverage is reworded.
- P20's deletion of the plain direction-mask control, which now runs on the old kernel, is recorded. P11's
  `TestKpAtTTSlack` is the engine's own control.

## Runs

- `p12a_check.py`, seed 2, current env: every dense check passes (worst 3.7e-14), and both silent cases raise.
- **Mutants** against `test_sampling_functions.py`: the core-size check off and the three corewise reroutes, all
  killed.
- **On the floor:** `test_sampling_functions`, `test_sampling_uniform` and `test_sampling_corewise`, 38 passed.
- **The coordinator's report:** the full gate passes (1,816).
  - The lowest MemAvailable was 2.6 GB, down from 4.6 GB. That is still above the floor the gate has run at, but worth
    watching as the suite grows.
