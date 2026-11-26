# P7: Engine C, the dot grade of the frame form — handover

Slice P7 of `PART1_PLAN.md`, built to the settled design: `P7_DESIGN_NOTE.md`, `P7_DESIGN_REVIEW.md`, the plan's P7
and the orchestrator's rulings of 2026-11-20. Written by the implementer, 2026-11-20, against `graded-messages` at
`3cae91e5` (P6 committed). Nothing is committed.

## 1. What changed

**`t3toolbox/backend/sampling_steps.py`.**
- **The dot steps**, written to serve both forms of the direction (the frame's now, the raw in P8):
  - `xi_dot_step`: `'JWo,KCio->JWKCi'`, the leaf's jet shape.
  - `mu_dot_step`, `nu_dot_step` and `eta_dot_step`: three terms each. The carry, or the stored core next to the
    direction, has `K`; so does `H` (on `KCaib`), and so does the core over the direction's leaf.
  - `Jp_step`: two `z_term`s.
- **The whole steps:** `identity_step` (a route over a private node) and `Jp_norm_sq_step` (`<Jp, omega^2 Jp>`,
  keeping `K + C`, no half).
- **The start values** take the stacks after `W` as one argument, `KC` (`K + C` for a row that carries `K`).
- Every string is among `STRINGS`.

**`t3toolbox/backend/sampling_variables.py`.**
- **The rows of the frame form,** all `carries_K=True`, the first five needing `{'variations'}`:
  - `xi_U_dot`;
  - `mu_R_dot` (left sweep through `R`, zeros);
  - `_frame.nu_L_dot` (right sweep through `L`, zeros) and its public route `nu_L_dot` (`identity_step`);
  - `_frame.eta_D_dot` and its route `eta_D_dot`;
  - `Jp`, which reads `_frame.eta_D_dot`;
  - `Jp_norm_sq`.
- **The adjoint seeded by `Jp` (ruling 1).**
  - `_JP_PATH` names the residual's rows on the path to the blocks.
  - `_twin(name) = '_Jp.' + name.lstrip('_')`.
  - `_jp_twins(graph)` derives each twin mechanically: the same step, start and needs, `carries_K=True`, reads on the
    path renamed, `r` read as `Jp`. `JtJp` is the twin of `g`. Nothing is written by hand but the docstring table.
- **Helpers.** `_base(name)` strips a private prefix and the leading underscore. `_letter(name)` is the side.
  `_table_name(name)` gives the `_SHAPES` entry: its own, its base's, or `'_' + base`. `_at_sites`, `_count` and
  `_shape` read through them.
- **`K`** (ruling 3): `_read_K` reads it off `V[0]` (the axes before the frame's `C` and the core's two). Every `V[i]`
  and `H[i]` must have exactly `K + C` plus its core's axes; otherwise a `ValueError` names the operand, its shape,
  `K` and `C`.
- **The setting** gains `K` and `with_K`, the names whose rows carry `K`. The shapes put `K` after `W` for those, on
  the blocks in front, and `Jp_norm_sq` is `K + C`. The call's record has `K = len(K)`.
- **The inputs** gain `V` and `H` (`_ARGUMENTS`), counted for the dtype when read.
- **`HeldVariables`** gains `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`, `Jp`, `Jp_norm_sq` and `JtJp` (25 slots).
  A held `Jp` must have the call's `K`, through the table.
- **The docstring:**
  - the graph table with the 17 new rows (twins included);
  - a paragraph on the dot grade, `K` and the route rows;
  - a paragraph on the twins;
  - the shape table's new rows;
  - the parameter `variations`.

**Tests.**
- **`tests/test_sampling_variables.py`** (now 17 tests):
  - the 7 new names by definition, with `variations` given in every call (`K = ()`);
  - `TestTheDirectionStack`:
    - every dot name with `K = (2,)`, element by element against the reference (which has no `K`), on the jet matrix
      and three problems;
    - the `K` check: a size-1 `K` on `H[1]`, no `K` on `V[2]`, one `K` axis more on `H[0]`, a `C` on `V[1]` (the
      message, and the operand named);
    - a held `Jp` without `K`.
  - The shape test covers `Jp_norm_sq` and `JtJp`. The jit test runs with `K = (2,)`.
- **`tests/test_sampling_identities.py`** (now 8):
  - `TestTheDirection`, per element of `K = (2,)` on the jet matrix and three problems, to rounding:
    - `<Jp, omega^2 r> = <p, g>`;
    - `<p, JtJp> = Jp_norm_sq`;
    - `JtJp` equals `g` with `Jp` held as `r`, with no data.
  - A held public `nu_L_dot`/`eta_D_dot` (doubled) never feeds `Jp`: bit for bit unchanged, and the public name is
    returned as held. This pins the frame-node design behaviourally.
  - The held-recomputed test runs over all 25 slots with `K = (2,)`.
- **`tests/test_sampling_graph.py`** (now 19):
  - the graph table with the twins;
  - the twin test: every row on `_JP_PATH` and `g` has its twin by the rule; no `_Jp.eta_tilde`, no `_Jp.g`;
  - the `carries_K` set;
  - `_base` and `_letter`;
  - the bond rule (a route over a sweep counts as a bond);
  - the starts of the new sweeps;
  - hand-derived step counts for 7 wants with directions (`('Jp',)` reads no `nu_L`/`eta_D`; `('g', 'JtJp')` runs
    each tilde step twice, `nu_step` 2d, `eta_step` 2d);
  - `variations` missing for `Jp`, `JtJp`, `nu_L_dot`, `Jp_norm_sq`;
  - private names refused.
- **`tests/test_held_variables.py`:** 25 slots; `_Jp.mu_tilde` and `Kp` are unknown names.
- **`tests/bridge/test_old_kernels_bridge.py`:** `TestTheDirection`, `ProbeDerivativesKind.forward` (= `t!` × new
  `Jp`) and `.transpose` given `omega^2 Jp / t!` forward (= new `JtJp`), at `n` = 1 to 3, `K = ()` and `(2,)`, three
  problems. The docstring's list of left-out kernels is updated: the frame dots are compared through `Jp` and `JtJp`.
- **`tests/test_dispatch.py`:** `test_jit_sampling_variables_with_a_stack_of_directions`, `('Jp', 'JtJp',
  'Jp_norm_sq')` at `K = (2,)`.
- **`tests/test_jax_absent.py`:** `Jp` and `JtJp` with `K = (2,)`.

**Verified by script before the tests asserted it:**
- the 7 new names against the reference, `K = ()` and `(2,)`, element by element, on the jet matrix and three
  problems: worst 9.5e-16;
- the bridge: forward and transpose, `K = ()` and `(2,)`, `n` = 1 to 3: worst 4.8e-16.

**Mutation pass** (scratch copy, deleted). 17 mutants, all killed:
- **The reviewer's five:** a dot term dropped (each of the four steps, one mutant each); `H` without its `K` check; the
  frame node replaced by the public route in `Jp`; a twin reading the residual's tilde; `K` read from `H`.
  - "The frame node replaced by the public route in `Jp`" was at first killed only by the structural docstring test.
    The held test above now kills it by behaviour.
- **The others:**
  - `mu_R_dot` sweeping `L`;
  - the frame `eta_D_dot` reading `nu_L`;
  - the twins without `K`;
  - the `K` check reading only the count;
  - `Jp_norm_sq`'s weight not squared;
  - a dot start of ones;
  - the seed of `JtJp` reading `r`;
  - `xi_U_dot` given the full order;
  - `_base` keeping the prefix.

**Single-process runs:** the changed files with the module doctests, `test_import_graph` and
`test_contractions_interpreter` pass on both environments: 85 tests (5,448 subtests) current, 85 on the floor.

**`subTest` labels:** builtins (the guard ran).

## 2. The gate

The whole gate at `-n 3` (`--detach`, then `run_gate.sh wait`), with nothing else running. The environment was python
3.11.15, numpy 2.4.6 and jax 0.10.2. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1722 passed, 5 skipped, 1 xfailed, 1 warning, 72953 subtests passed   1260.5     10433       1903
    doctest-modules     PASS   314 passed                                                     55.5      1683      10392
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.6       259      11812
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.9       360      11723
    examples            PASS   14 examples, 0 failed                                           385       636      11445
    sphinx              PASS   0 warning/error line(s)                                      1017.0       633      11314

**No failure.** Against P6 (1,715):
- The suite has 7 more tests:
  - the direction stack, 2;
  - the identities of the direction, 1;
  - the held public dot, 1;
  - the twin test, 1;
  - the bridge's `Jp` and `JtJp`, 1;
  - the jit with `K`, 1.
- The suite's peak memory is 10.4 GB, with 1.9 GB available at its lowest.

No process, lock or file of the gate is left in the repository.

## 3. Departures and choices

1. **The dot steps are named by grade** (`xi_dot_step`, `mu_dot_step`, `nu_dot_step`, `eta_dot_step`), as in
   `jet_steps_check.py`, and serve both forms. P8's raw rows reuse them with `L`, `L`, `L_dot` (and `D_dot`).
2. **`Jp_norm_sq` has `carries_K=True`.** Its step takes the record only for the counts of `J` and `W`; the flag
   puts `K` in its shape.
3. **The `K` check also catches a wrong core rank** (the total count of axes), which is how "a wrong count" is
   detected when `K + C` alone could still be a prefix.
4. **The `JtJp` route has no `needs` of its own.** It reaches `variations` through `Jp`, and the closure's message
   names the chain (`JtJp -> _Jp.V_tilde -> ... -> Jp`). That chain shows private names.
5. **The bridge compares the frame dots through `Jp` and `JtJp`**, not through `compute_sigma_jets` and its kin.
   Their input is a derivative-normalized layout of their own, and the outputs are the documented interface. The
   docstring says so.
