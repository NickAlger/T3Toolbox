# P8: Engine D, the raw dot grade, the tilde-dot grade, `Kp` and the corewise point — handover

Slice P8 of `PART1_PLAN.md`, built to the settled design: `P8_DESIGN_NOTE.md`, `P8_DESIGN_REVIEW.md`, the plan's P8
and the orchestrator's rulings of 2026-11-20. Written by the implementer, 2026-11-20, against `graded-messages` at
`2aee3c1d` (P7 committed). Nothing is committed.

## 1. What changed

**`t3toolbox/backend/sampling_steps.py`**, a new section "The tilde-dot grade (5.4)":
- `xi_tilde_dot_step`: `'JWCo,KCio->JWKCi'` (no `K` on the seed `w_tilde`; `K` comes from `D_dot`).
- `mu_tilde_dot_step` and `nu_tilde_dot_step`: six terms each (the carry with `K`; `xi_tilde_dot`, `xi_dot`; `G_dot`
  over `xi_tilde`/`xi` with the carry's dot or tilde).
- `eta_tilde_dot_step`: six terms.
- `eta_tilde_dot_top_step`: the short node's six readouts per slot, among them the new literal
  `'JWCa,KCaib,JWCb->WKCi'` (`G_dot` between `mu_tilde` and `nu`).
- `V_tilde_dot_step`: the leaf readout plus `'JWKCi,JWCo->KCio'` (`eta_dot` against `w_tilde`).
- `H_tilde_dot_step`: nine readouts.
- Every string is among `STRINGS` (`tests/_jet_contract_cases.py` gains the new literal: 32 strings; the counts in
  `test_jet_contract.py` and `test_contractions_sharding.py` follow).
- The raw dots reuse P7's `xi_dot_step`, `mu_dot_step`, `nu_dot_step`, `eta_dot_step` and `Jp_step` with `L`, `L`,
  `L_dot` and `D_dot` in the frame's places.

**`t3toolbox/backend/sampling_variables.py`.**
- **The raw dot rows** (ruling 1), needing `{'core_directions'}`: `xi_dot`, `mu_dot`, `_raw.nu_L_dot`,
  `_raw.eta_D_dot`.
- **The O-12 routes.** `nu_L_dot` and `eta_D_dot` have two route rows each, raw first (needs `core_directions`),
  frame second (needs `variations`): with both directions given, the public dots are the raw form. `Jp` has the
  frame route first, then the raw route (`_raw.eta_D_dot`, `D`, `eta_D`, `D_dot`): with both given, `Jp` is the
  frame's.
- **The tilde-dot rows:** `xi_tilde_dot`, `mu_tilde_dot`, `nu_L_tilde_dot`, `eta_D_tilde_dot`, the short node
  `_eta_D_tilde_dot_top`, and the blocks `V_tilde_dot`, `H_tilde_dot`; `Kp` is `g_step` over them. All need
  `core_directions`.
- **The two short nodes of the tilde grade** used by the corewise routes: `nu_L_tilde`, `eta_D_tilde` and
  `_eta_D_tilde_top`.
- **`JtJp` is unreachable with only `core_directions` on a manifold call** (ruling 1): its twin path reads the frame's
  tilde blocks. The docstring's dot-grade paragraph says why.
- **One `K`** (ruling 2). `_read_K(variations, core_directions, C)` reads `K` off `V[0]` (or `D_dot[0]` when
  `variations` is absent) and checks every `V[i]`, `H[i]`, `D_dot[i]` and `L_dot[i]` against `K + C`; a mismatch
  names the operand, its shape, the operand `K` was read from, `K` and `C`.
- **The corewise point** (rulings 3 and 4). A 2-store frame `(B, G)` is a corewise call: `D = B`, `L = G`, present
  gains `'_corewise'` (a manifold call gains `'_manifold'`).
  - The corewise graph is derived from the probe rows: drop rows reading `U`, `R`, `V`, `H`, `Vc`, a `_frame.*` node
    or a dropped name (transitively), or needing `'_manifold'`; then rename names, reads AND needs by
    `_COREWISE_RENAME` (`nu_L` -> `nu`, `eta_D` -> `eta`, their tilde, dot and tilde-dot forms, the `_raw.*` and
    short nodes, `variations` -> `core_directions`); then the twins, through `_jp_twins(renamed, path,
    'core_directions', '_corewise')`, with the path itself renamed.
  - Two import-time assertions: a rename never lands on a surviving name ("the corewise rename of %r lands on the
    surviving %r"); every name on the twin path survives. A loop at import reaches every public corewise name.
  - `_jp_twins` takes the call type and skips rows needing the other call's, so the corewise routes have no twins on
    the manifold graph.
  - On a corewise call `variations` raises: "a corewise point has one form of a direction, the raw one: pass it as
    core_directions, not variations". One `if`, easy to change after P11.
  - `_GRAPHS` is keyed by `(kind, call)`: 56 rows on the manifold graph, 41 on the corewise one.
- **`_DIRECTION_MASKS`** (ruling 5), one table for P11: `V`, `H`, `Vc` -> the variation masks; `D_dot` -> `D`'s,
  `L_dot` -> `L`'s.
- **The inputs** gain `D_dot` and `L_dot` (`_ARGUMENTS`), counted for the dtype when read.
- **`HeldVariables`**: 40 slots, the public manifold names in graph order, then the corewise extras (`nu_dot`,
  `eta_dot`, `nu_tilde_dot`, `eta_tilde_dot`); `Kp` and the new dots among them.
- **`_SHAPES`** gains the new nodes; block sizes `VD` and `HL` for `V_tilde_dot` and `H_tilde_dot`.
- **The docstring:** the graph table regenerated from the data (twins included); a corewise paragraph and the rename
  table (`| On a manifold frame | On a corewise point |`); the dot-grade paragraph; the shape table's new rows; the
  parameters `frame` (or `(B, G)`) and `core_directions`.

**Tests.**
- **`tests/test_sampling_variables.py`:** the new names by definition (both directions passed); the reference
  helper takes either direction; `Kp`, `V_tilde_dot`, `H_tilde_dot` by element of `K`; the `K` check for
  `D_dot`/`L_dot` (a mismatch on each, no `K` on `D_dot[0]`); the shape test.
- **`tests/test_sampling_graph.py`:** the slots with the corewise extras; the `carries_K` set; the starts; the bond
  rule (excluding `Kp`); the shape table's `VD`/`HL`; the earlier-row test over `z`, `nu_L_dot`, `eta_D_dot`, `Jp` and
  the corewise routes `V_tilde`, `H_tilde`; `test_the_direction_operands_and_their_masks`; the missing-direction
  messages.
- **`tests/test_held_variables.py`:** 40 slots; `Hp` is the unknown name.
- **`tests/test_sampling_identities.py`:** `TestTheTwoFormsOfADirection`: along the lift (the undamped lift of a gauged
  `(V, H)`), the raw `nu_L_dot`, `eta_D_dot` and `Jp` equal the frame's, to rounding, at order `()` and `(3,)`; with
  both given (O-12, on directions where the forms differ), the public dots are the raw form, bit for bit, and `Jp` the
  frame's.
  The held-recomputed test runs over the manifold public names.
- **`tests/test_sampling_corewise.py`** (new, 8 tests): every corewise name against the reference by its Section-9
  definition; `JtJp` equal to `g` with `Jp` held as `r` (ruling 3); the corewise call equal to the manifold call at
  `(D, L)` (`nu` = `nu_L`, `eta` = `eta_D`, `mu`, `z`, `Kp`, `nu_dot` = `nu_L_dot`, `eta_tilde` = `eta_D_tilde`); the derived graph against the rename table (no row reads `U`, `R`, `V`, `H`, `Vc` or a
  frame node, needs `variations` or `'_manifold'`, or keeps a renamed name; the twins are the renamed path's); the
  collision assertion fires on a colliding row, and every public corewise name is reached again; a basis name is unknown and `variations` raises; step counts for `('z', 'g')` and `('Kp',)` (ruling 6);
  a held `nu` with the corewise sizes.
- **`tests/bridge/test_old_kernels_bridge.py`:** `test_Kp_against_the_old_corewise_hessian`, the old kernel given
  forward `omega^2 r / t!`, `K = ()` and `(2,)` (ruling 6).
- **`tests/test_dispatch.py`:** `test_jit_sampling_variables_curvature_and_the_corewise_point`.
- **`tests/test_jax_absent.py`:** `Kp` on a manifold call; `nu` and `g` on a corewise call.
- **`Kp` is checked against the reference module's `Kp`, exactly** (ruling 6), not by a difference of `g`.

**Verified by script before the tests asserted it:**
- the manifold names, raw and tilde-dot, against the reference on the jet matrix, `K = ()` and `(2,)`: worst 1.6e-15;
- the corewise names against the reference: worst 1.6e-15;
- the bridge, `Kp` against the old `corewise_hessian`: worst 6.5e-16.

**Single-process runs:** the changed files pass on both environments: 143 passed, 1 xfailed (22,877 subtests)
current; 143 passed, 1 xfailed on the floor. (One test gained a check after the mutation pass, below; the gate ran it.)

**`subTest` labels:** builtins (the guard ran).

**Mutation pass** (scratch copy, deleted). 20 non-equivalent mutants, all killed:
- **The reviewer's five:**
  - one of the nine `H_tilde_dot` terms dropped;
  - `xi_tilde_dot` with `K` on the seed;
  - the short node's third literal (`G_dot` between `mu_tilde` and `nu`) dropped;
  - the rename table missing `nu_L_tilde`;
  - `needs` not rewritten. This one first **survived**, as an equivalent mutant: no row that survives the drop needs
    `variations` today (every such row reads a frame node). `test_the_import_time_assertions` now adds a later row
    needing `variations` that survives (a route over `z`) and asserts it comes out needing `core_directions`; it
    kills the mutant.
- **The others:**
  - `mu_tilde_dot` without its `G_dot`-over-`xi_tilde` term; `nu_tilde_dot` without its `nu_dot` term;
    `V_tilde_dot` without its `eta_dot` term; the short node without its `nu_tilde_dot` term;
  - the raw route of `nu_L_dot` after the frame's;
  - `V_tilde_dot` reading the public `eta_D_dot` (the frame's, when both are given);
  - the raw route of `Jp` reading `eta` (the `U` basis);
  - the `K` check skipping `L_dot`;
  - a corewise call accepting `variations`;
  - the corewise drop keeping rows that read `R`; the corewise `V_tilde` route reading `_eta_tilde_top` (both killed
    at import, by the closure's reachability loop);
  - the corewise twin path not renamed (killed at import, by the path assertion), missing `mu_tilde`, missing
    `H_tilde`.
- **One equivalent mutant, not counted:** the corewise twins built from `_JP_PATH` (the manifold names). After the
  rename the corewise path is spelled the same (`nu_L_tilde` -> `nu_tilde`, `_eta_D_tilde_top` -> `_eta_tilde_top`),
  which is the coincidence the review named; the three path mutants above are the non-equivalent forms, and the
  corewise `JtJp`-against-`g` test kills two of them.

## 2. The gate

The whole gate at `-n 3` (`--detach`, then `run_gate.sh wait`), with nothing else running. The environment was python
3.11.15, numpy 2.4.6 and jax 0.10.2. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1734 passed, 5 skipped, 1 xfailed, 1 warning, 76616 subtests passed   1258.4     13428       1940
    doctest-modules     PASS   314 passed                                                     46.9      2060       9994
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.6       272      11775
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.2       367      11703
    examples            PASS   14 examples, 0 failed                                           383       638      11363
    sphinx              PASS   0 warning/error line(s)                                      1016.3       630      11297

**No failure.** Against P7 (1,722), the suite has 12 more tests: the new `test_sampling_corewise.py` (8), the
direction-masks test, the two forms of a direction, the bridge's `Kp`, and the jit of the curvature and the corewise
point.

**Memory:** the suite's peak rose from 10.4 GB to **13.4 GB**, while its lowest available memory stayed at 1.9 GB
(as at P7). The time is unchanged (1,258 s against 1,260 s). I have not traced which test raised the peak; the
curvature and corewise jit test and the jet-matrix sweeps of `Kp` are the likely candidates. Worth watching at P9.

No process, lock or file of the gate is left in the repository.

## 3. Departures and choices

1. **`_jp_twins` takes the call type** and skips rows needing the other call's; without it the corewise-only routes
   grew twins on the manifold graph (the docstring table went from 58 to 56 rows).
2. **The corewise routes for `V_tilde` and `H_tilde` sit before the manifold routes** and need `'_corewise'`; the
   manifold routes need nothing new. The call type is a `present` token, not a field.
3. **`HeldVariables` is reordered**: the public manifold names in graph order, then the corewise extras. A held slot
   is still read by name.
4. **The `K` check's message names the operand `K` was read from** (`V[0]` or `D_dot[0]`), since either can be
   first.
5. **The new literal is added to `STRINGS`**, and the counts of the contract and sharding tests follow (32 strings;
   `eta_top` has 3 readouts).
