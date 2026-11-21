NOT YET: two things for round 1. The coordinator's ruled check of the directions' trailing shapes, and tests of the three corewise transposes with a residual carrying `K` (three surviving mutants). The build itself is correct.

# P12a review, round 1: the ragged backend sampling functions on the engine

Reviewer, 2026-11-20. This reviews the uncommitted P12a on top of `e2623c75`, against the plan's P12 paragraph, my
`P12_DESIGN_REVIEW.md` and the coordinator's rulings. Everything was run single-process, one run at a time. The
mutant copies have been deleted, and the working tree is untouched apart from this file.

## 1. What I ran

| Check | Result |
| --- | --- |
| The 11 edited test files (functions, bridge, graph, corewise, variables, curvature, uniform curvature, uniform probing, sweep reuse, probe derivatives, manifold), current env | 339 passed, 2 skipped, 8,320 subtests |
| On the floor: functions, bridge, corewise | 30 passed |
| My `p12a_check.py`, dense ground truth only (details below), seeds 0 and 1, both envs | all pass; worst 1.4e-13 (5.6e-13 on the floor) |
| The ruled gap, reproduced | `t3_apply_corewise_hessian` with `L_dot[1]` at `rL = 1` against 2, and `tv_apply` with `V[0]` at `nU = 1` against 3: **both return with no error** |
| 11 mutants of my own (section 3) | 8 killed, 3 missed |
| Plan sections 11.3 and 11.4 against `HEAD` | identical; the only difference in that range is 11.5's heading, "proposed" becoming "approved" |

**What `p12a_check.py` checks, against dense ground truth only:**
1. `tv_probe` and `tv_apply` against the dense tangent through my roots-of-unity oracle, at `()`, `(2,)` and `(1, 1)`.
2. `tv_*_transpose` with a residual carrying `K = (2,)`: per `k`, `<J v, c_k> = <v, J^T c_k>`, with `J v` from the
   dense tangent.
3. `t3_*_corewise_transpose` with that residual: per `k`, the pairing equals `d/de <c_k, z(x + e x_dot)>`, by exact
   polynomial interpolation (degree at most `2d`).
4. `t3_*_corewise_hessian` with `K = (2,)` directions: `<y, H x>` against the polarization of `phi''`, interpolated
   exactly.
5. `sum_over_samples=False` sums to the default.
6. The entries tangent and its transpose with `K`.

## 2. The settled design, point by point

**The held-`Jp` route.** Correct:
- the second `JtJp` row needs `held:Jp`, and that fact is in `present` only for a held `Jp`;
- P8's message is unchanged (the filter keeps `held:Jp` out of it);
- `K` is read after the jets and `W`, before probe's mode edge;
- a direction's `K` wins, and a disagreeing held `K` raises.

The contract is stated in the module docstring, in `HeldVariables` and in the plan's P12 paragraph. The weight
wording ("`J^T` of `weight^2 c`") is right: the transposes pass none.

The engine tests are as asked:
- against `g` at `K = ()`;
- per element at `K`;
- the structural step count;
- P8's message.

My mutants on the route all die: the agreement check off, `K` read without skipping `W`, the route without its fact.

**The 15 functions.** Each is one engine call, in N16's argument order, with the sum flag. They are right by my dense
checks.

**The ragged mode-size check (ruling 5).** It is right and isolated; my mutant removing it dies in 6 subtests. The
handover's table of 21 malformed inputs shows nothing lost, and three silent broadcasts closed.

**The bridge guard.** `old_side()` patches the module attribute that the functions look up, and `TestTheGuard` shows
it bites on `t3_probe`, `t3_apply`, `t3_entries` and `ApplyKind.point_forward`. The order-0 `z` goes through the
derivative kernels at order 0, inside the guard.

*Nit:* the handover says the guard wraps "every old call of the file", but some old calls run outside it, for example
`t3_apply_derivatives(...)` at line 325 and `kind.forward(...)` at line 333. They are derivative kernels and do not
reach the engine today, but either wrap them or reword the claim.

**Bitwise to rounding.** I checked each change against the code paths. Each is a genuine change of path, from the
engine against the old kernel:

| Test | Where | Old against new |
| --- | --- | --- |
| `test_curvature` | the sweep reuse | the one-shot name is now the engine, the from-sweep tail the old kernel |
| `test_curvature` | order 0 against plain | the plain name is now the engine; 1e-14 becomes 1e-13 |
| `test_curvature` | the kind against the two-hole | the kind's tail is old, the reference the engine |
| `test_uniform_probing` | the hygiene of the three plain corewise transposes | the twin (engine) against the old kernel at the masked point's corewise frame; 1e-12 relative on the real block |

In the hygiene test the exact assertions stay exact: padding `== 0`, `array_equal` on the garbage, the shapes. The
derivative rows of that test keep `tobytes`. That is right.

One consequence to record for P20. In `test_uniform_curvature`, the plain two-hole's negative control (the direction
mask matters) now runs on the **old** `*_from_sweep` kernel, which P20 deletes, so P20 must delete the control with it.
The engine's own permanent control of the directions' masks is P11's `TestKpAtTTSlack`, so no coverage is lost then.
A line in the plan's P20 list would make sure.

**The uniform twins' bodies.** They go through the engine on the full `.data`, with the frame's masks, as ruled. The
signatures and the mirror are P12b's.

## 3. Mutants (11, mine)

**Killed:**
- the `K` agreement check off;
- `K` read without skipping `W`;
- the ragged `N` check off;
- the `held:Jp` route without its fact;
- `tv_probe_transpose`, `tv_apply_transpose` and `tv_entries_transpose` each computed as `g` from a held `r` (correct at
  `K = ()`, wrong with `K`). These are killed, but only by `test_manifold`, through `T3Tangent`'s static transposes.

**Missed:** `t3_probe_corewise_transpose`, `t3_apply_corewise_transpose` and `t3_entries_corewise_transpose`, each
computed as `g` from a held `r`. No test in the 14 files I ran (all the edited ones, plus `test_tucker_tensor_train`)
gives a corewise transpose a residual with `K`. The implementation is right (my check 3 above), but the capability is
untested, so a plausible regression back to the `g` route would pass.

## 4. Findings

1. **(Ruled by the coordinator; must fix.) Check the trailing (core) shapes of the directions against the stores.**
   - Reproduced:
     - `L_dot[1]` with `rL = 1` against 2 returns a wrong `t3_apply_corewise_hessian`, silently;
     - `V[0]` with `nU = 1` against 3 returns a wrong `tv_apply`, silently.
   - The check: `V[i]`, `H[i]`, `D_dot[i]` and `L_dot[i]` (and `Vc[i]`) have exactly the trailing shapes the stores
     give:
     - `V`, `Vc`: `(nU_i, N_i)`;
     - `H`: `(rL_i, nD_i, rR_{i+1})`;
     - `D_dot`: `D[i]`'s;
     - `L_dot`: `L[i]`'s;
     - on a corewise call, the point's own.

     Use tuple equality and a `ValueError` naming the operand, as for the stacks. On uniform, compare against the
     supercores' trailing shapes.
   - Test each operand alone (size 1 against more), on a manifold frame and a corewise point, on both layouts where
     P12b's twins take them.
2. **(Should fix.) Test the six transposes with a residual carrying `K`, in `test_sampling_functions.py`.**
   - The test: per element, the call with `K` equals the call on that element's residual, with `K = (2,)` and
     `K = (2, 3)`, and with `sum_over_samples=False`.
   - Give each transpose its own adjoint identity per `k`.
   - This kills the three surviving mutants, and puts the `tv` cases in the functions' own file rather than only in
     `test_manifold`.
3. **(Nit.)** The bridge guard's coverage claim (section 2).
4. **(Record.)** P20 deletes `test_uniform_curvature`'s plain direction-mask control, which now runs on the old
   kernel; the engine's control is P11's.
5. **(Process.)** The full gate was not re-run after the rename-table test changed. Run it once with round 1's changes.

## 5. Round 2

I will re-run the ruled check's tests, my three missed mutants, `p12a_check.py` and the edited files on both envs.
