# P12a: the ragged backend sampling functions on the engine — handover

The first commit of slice P12 of `PART1_PLAN.md` (the split ruled 2026-11-20, the plan's P12 "Design, settled before
the build"), built to `P12_DESIGN_NOTE.md`, `P12_DESIGN_REVIEW.md` and the coordinator's rulings 0 to 5. Written by
the implementer, 2026-11-20, against `graded-messages` at `e2623c75`. Nothing is committed.

## 1. What changed

**The plan** (`PART1_PLAN.md`, asked for): sections 11.3 and 11.4 restored verbatim from `HEAD` (checked by diff: only
the tidy's intended removals remain); the rulings written into P12 as one paragraph, before P13.

**The engine** (`backend/sampling_variables.py`):
- **A second `JtJp` route** (ruling 1, option (a)): `_jp_twins` emits the `JtJp` row twice, the second needing the fact
  `_HELD_JP = 'held:Jp'` in place of the direction. `present` gains `'held:Jp'` only when `Jp` is held. The
  missing-argument message leaves that fact out, so P8's error ("'JtJp' needs the argument variations, which was not
  given") is unchanged for a call with neither.
- **`K` off a held `Jp`**: `_read_K` returns `None` with no direction; `_read_setting(..., held_Jp)` reads `K` off the
  held `Jp` (between the jets and `W`, and `C`, and probe's mode edge). A direction's `K` wins; a held `Jp` whose `K`
  differs raises ("the held Jp has the stack of directions K = ..., where the directions give K = ...").
- **The core sizes of the directions** (round 1): `_check_stacks` checks the trailing core axes of `V`, `H`, `Vc`,
  `D_dot` and `L_dot` against the stores by tuple equality (a `ValueError` naming the operand; uniform: the padded
  sizes). This closes the silent case of section 2.
- **The ragged mode size** (ruling 5): the ragged branch of `_check_stacks` checks `w[i].shape[-1] == N_i` read off
  `D[i]`, beside P11's packed check. A size-1 mode axis no longer broadcasts.
- **Docstrings**: the probe table and apply's own table gain the second `JtJp` row (`held:Jp`); the prose and
  `HeldVariables` state the contract: holding any vector `c` of `Jp`'s type as `Jp`, with no direction, gives
  `J^T (weight^2 c)`.

**The ragged functions** (`backend/probing.py`, `apply.py`, `entries.py`): the 15 functions are one engine call each,
with the merged signatures (object first, then the sample, the residual, the directions; `order=()`, none for entries;
`sum_over_samples=True` on the transposes):
- `t3_*` want `z` on the point;
- `tv_*` want `Jp`;
- `tv_*_transpose` and `t3_*_corewise_transpose` want `JtJp` with the residual held as `Jp`;
- `t3_*_corewise_hessian` want `Kp` with the residual held as `r`.

Each has a doctest with real output. The docstrings of the transposes and the Hessians point repeated calls at one
frame to the engine with `held=`. The three ambient transposes take `(sample, [shape,] residual,
sum_over_samples=False)`. The old kernels, sweeps and `*_from_sweep` functions are untouched.

**The uniform twins, bodies only** (`ut3_sampling.py`, `utv_sampling.py`; signatures and mirroring move in P12b).
The bodies called the ragged functions with masked bare supercores, which broke, so they now call the engine on the
full uniform data:
- `ut3_{probe,apply,entries,full_sum}` want `z`;
- the corewise transposes want `JtJp` (held `Jp`), the Hessians `Kp` (held `r`);
- `utv_{probe,apply,entries}` want `Jp`, with `variations_data[:2]` masked by the engine with **the frame's masks**
  (ruling 3; the module docstring says so);
- `utv_*_transpose` want `JtJp`, then attach and re-mask the gauge masks as before.

**The library call sites**:
- the plain kinds' `point_forward` (`backend/fitting.py`; probe's arguments swapped);
- `tucker_tensor_train.py` (probe, the corewise transposes, the three ambient static methods);
- `manifold.py` (`tv_*` and their transposes).

The frontend keeps `sum_over_probes=False` and passes `sum_over_samples=sum_over_probes` until P13.

**Tests**:
- **New: `tests/test_sampling_functions.py`**, numpy only:
  - the held-`Jp` route against `g` from a held `r` at `K = ()`, against the per-element `g` at `K = (2,)`, on the
    manifold frame and the corewise point, for all three kinds;
  - a direction wins, and a mismatched `K` raises;
  - neither direction nor held `Jp` gives P8's message;
  - **the step count, structural**: the closure from a held `Jp` runs exactly `g`'s steps, each once, and no row that
    needs a direction;
  - the ragged size-1 mode axis raises (forward and transpose);
  - `t3_*` and `tv_*` against `dense_*`, at `order = ()`, `(2,)` and `(1, 1)`;
  - the adjoint identity for the tangent and corewise transposes;
  - the corewise transposes as gradients, and the Hessians as second derivatives along a stack of directions, by
    central differences (automatic differentiation is P12b's);
  - `sum_over_samples=False` keeps `W` and sums to the default;
  - the ambient transposes by definition, with a default that does not sum;
  - the signatures (entries has no `order`).
- **The bridge** (ruling 2):
  - the order-0 rows call old code only: `z` through the order-0 derivative kernels (`t3_*_derivatives(..., 0)[0]`), and
    `Jp`, `g`, `JtJp`, `Kp` through the old plain kinds' `forward`, `transpose` and `corewise_hessian` (the
    `*_from_sweep` functions);
  - never `point_forward` of a plain kind;
  - `old_side()` patches `sampling_variables` to raise while the old side runs. It wraps every old call of the plain
    kinds and the order-0 `z`, and the derivative kinds' calls in `TestTheDirection` and `TestTheCurvature`; some
    derivative-kernel calls run outside it (for example `t3_apply_derivatives` and `kind.forward` in the apply rows at
    `n >= 1`), which do not reach the engine today;
  - `TestTheGuard` shows it bites on `t3_probe`, `t3_apply`, `t3_entries` and `ApplyKind.point_forward`.
- **Graph tests**:
  - the twin test takes `JtJp`'s first route and checks the second is the same row needing only `held:Jp`;
  - `JtJp` joins the names with two routes (a `held_for` helper gives `_closure` the held `Jp` the fact implies);
  - `held:Jp` is an allowed need.
- **The call sites moved** to the new orders:
  - `test_dispatch.py` (the two tables);
  - `test_manifold.py`, `test_from_sweep_reuse.py`, `test_probe_derivatives.py`;
  - `test_curvature.py` (the helper, three call sites);
  - `test_uniform_curvature.py` (the by-reference table, adapted by a lambda; the direction-mask control now runs the
    old `*_from_sweep` kernel, since the ragged name now masks);
  - `test_uniform_probing.py`: the bare seam of the plain corewise transposes is the old `tv_*_transpose_from_sweep`
    at the masked point's corewise frame.
- **Bitwise to rounding**: where a test compared the ragged name with an old kernel bitwise, the two are now
  different code (the engine against the old kernel). It now compares them to rounding: 1e-13 in `test_curvature`
  (the sweep-reuse test, order 0 against plain, the kind against the two-hole), and 1e-12 relative on the real block
  in `test_uniform_probing`'s hygiene for the three plain corewise transposes. The padding and garbage assertions stay
  exact.

## 2. Today's checks, kept (ruling 5)

The 15 functions at `HEAD` raise nothing of their own; their checks were numpy's (`einsum` label sizes, indexing). I
compared old and new on 21 malformed inputs (`scratchpad/p12/checks/razor.py`, against `HEAD`'s modules):

| Input | `HEAD` | P12a |
| --- | --- | --- |
| `ww` with `d - 1` or `d + 1` vectors | `IndexError` | `IndexError` (same) |
| `ww[i]` with the wrong `N_i` | `einsum` broadcast error | the engine's `N_i` message |
| `ww[i]` with `N_i = 1` | **silent broadcast** | raises (new check) |
| `ww[i]` with a different `W` | `einsum` label error | the engine's `W` message |
| `ww[i]` with `W` of size 1 | **silent broadcast** | raises |
| a residual with the wrong `N_i`, `W` or `d` | `einsum` error or `IndexError` | the held-`Jp` shape message |
| a residual with `N_i = 1` or `W = 1` | **silent broadcast** | raises |
| `index` with `d - 1` rows | **silent** | raises (P11's check) |
| `index` out of range, negative or float | `IndexError` | `IndexError` (same) |

Nothing is lost.

**One silent case was common to `HEAD` and the first build:** a variation or core direction with a size-1 core axis
where the stores give more (for example `L_dot[1]` with `rL = 1` against 2) broadcast silently in `tv_apply` and
`t3_apply_corewise_hessian`. The coordinator ruled the check, and round 1 added it (section 1;
`P12a_RESPONSE_r1.md`): the core axes of `V`, `H`, `Vc`, `D_dot` and `L_dot` are now checked against the stores. The
point's own cores (a `G[1]` of rank 1 against 2 still broadcasts in `t3_apply`) are the frontend's `validate()`, as
before.

## 3. Verified by script, before the tests

- `JtJp` from a held `Jp` equals `g` from a held `r` for probe, apply and entries, on manifold and corewise calls:
  worst 1.8e-16.
- The per-element `K = (2,)` case matches.
- With neither a held `Jp` nor a direction, the message is P8's.

## 4. Mutation pass (scratch copy, deleted)

17 mutants, all killed:
- the engine: the held route removed; `held:Jp` not in `present`; `K` not read off the held `Jp`; no `K` agreement
  check; no ragged `N` check; the missing-names filter removed;
- the ragged functions: a transpose's default `False`; an ambient default `True`; `t3_probe` dropping `order`; a
  corewise transpose ignoring its flag; a Hessian holding `Jp` for `r`;
- the uniform twins: the probe corewise transpose not packing; `ut3_probe` not unpacking; a `utv` transpose dropping
  its flag; `ut3_full_sum`'s ones changed;
- the frontend sum flag inverted;
- the bridge guard switched off.

## 5. Environments and gate

The affected files pass on both environments:
- `t3toolbox` (py3.11 / numpy 2.4.6 / jax 0.10.2);
- the floor `tttt` (py3.9 / numpy 1.22 / jax 0.4.30): 120 tests in the engine, bridge, graph, functions, sweep-reuse,
  probe-derivatives and uniform-probing files;
- the module doctests of every changed module.

Gate: see section 7.

## 6. For P20

`test_uniform_curvature`'s plain two-hole negative control (`test_the_direction_mask_is_load_bearing` of
`TestUniformPlainKindsTwoHole`) now runs on the old `*_corewise_hessian_from_sweep` kernel, which P20 deletes: P20
deletes this control with it. The engine's own control of the directions' masks is P11's `TestKpAtTTSlack`, so no
coverage is lost.

## 6a. Left for P12b

- the uniform twins' merged signatures and sample mirroring, and their call sites (`uniform_manifold.py`,
  `uniform_tucker_tensor_train.py`, `uniform_fitting.py`);
- the `t3_*_corewise_tangent` functions and their twins;
- the corewise Hessians' `sum_over_samples`;
- the automatic-differentiation tests;
- `test_dispatch` rows for the twins.

The docs pages that name the functions in prose are P19's.

## 7. Gate

The full gate ran at `-n 2` (logs in `scratchpad/gate_P12a`).
- **suite**: 1 failed, 1813 passed, 5 skipped, 1 xfailed, 90,079 subtests, in 2003 s; peak 7.5 GB; **lowest
  MemAvailable 4586 MB**.
- **doctest-modules**: PASS, 324.
- **quickstart**: PASS.
- **pages**: PASS, 48.
- **examples**: PASS, 14.
- **sphinx**: PASS, 0 warnings.

The one failure was `tests/test_sampling_corewise.py::TestTheDerivedGraph::test_the_rename_table`. It pinned the
corewise graph's `JtJp` to a single route, so I updated it to the two routes (`core_directions`, then `held:Jp`). The
file was then re-run on its own: 8 passed, 2160 subtests, on `t3toolbox` through the gate's `quick` step, and on
`tttt`. The full gate was not re-run after this one-assertion test change.

**Round 1's full gate** (`-n 2`, with round 1's changes and the rename-table fix in; logs in `scratchpad/gate_P12a_r1`):
**all PASS.**
- suite: 1816 passed, 5 skipped, 1 xfailed, 90,237 subtests, in 2109 s; peak 7.9 GB; **lowest MemAvailable 2556 MB**;
- doctest-modules 324, quickstart, pages 48, examples 14, sphinx with 0 warnings.
