# P6: Engine B, the tilde grade — handover

Slice P6 of `PART1_PLAN.md`, built to the settled design: `P6_DESIGN_NOTE.md`, `P6_DESIGN_REVIEW.md`, the plan's P6
as amended, and the orchestrator's rulings of 2026-11-20. Written by the implementer, 2026-11-20, against
`graded-messages` at `9a94b887` (P5 committed). Nothing is committed.

## 1. What changed

**`t3toolbox/backend/sampling_steps.py`.**
- **The tilde steps**, each term one literal `jet_contract` call:
  - `xi_tilde_step`: `'JWKCo,Cio->JWKCi'`.
  - `mu_tilde_step`: two terms, `'JWKCa,Caib,JWCi->JWKCb'` and `'JWCa,Caib,JWKCi->JWKCb'`.
  - `nu_tilde_step`: two terms, `'Caib,JWCi,JWKCb->JWKCa'` and `'Caib,JWKCi,JWCb->JWKCa'`.
  - `eta_tilde_step` (the full one): two terms, `'JWKCa,Caib,JWCb->JWKCi'` and `'JWCa,Caib,JWKCb->JWKCi'`.
- **`eta_tilde_top_step`, the short form.** It holds the slots `eta_tilde[c - a + 1 : c + 1]` on every jet axis,
  with `a = min(J_in_i, F)` read off `w[i]`'s static shape.
  - Each slot is two readouts at the lowered corner `c - a + 1 + j`: `'JWKCa,Caib,JWCb->WKCi'` and
    `'JWCa,Caib,JWKCb->WKCi'`, which keep `W`.
  - The slots are stacked over the box of `a`.
- **`V_tilde_step`** reads the short node. `V_term(top, w)` is a readout at the corner `a - 1`, the leaf's jet shape,
  string `'JWKCi,JWo->KCio'`. `V_term(eta, w_tilde)` is a readout at the corner, `'JWCi,JWKCo->KCio'`.
- **`H_tilde_step`:** three readouts `'..->KCaib'`, the tilde on `mu`, `xi` or `nu`.
- **The whole steps** `w_tilde_step` (`jets.jet_to_dual_jet('probe', ...)`) and `g_step`, which pairs the blocks.
- **`start_zeros`.**
- **`K` in the strings.** Every tilde string carries `K` on its tilde operands. With the record's `K = 0` they are the
  plain strings, so P7's `Jp`-seeded rows reuse the steps.
- **The import graph:** `sampling_steps` now imports `jets`. The graph stays acyclic.

**`t3toolbox/backend/sampling_variables.py`.**
- **The row type** gains a sixth field, `start` (ruling 3): `'ones'` or `'zeros'` for a sweep, `None` for the other
  drivers. `_run` takes `start_<value>` from `sampling_steps`; an undeclared start raises.
- **The probe graph** gains 9 rows:
  - `w_tilde`;
  - `xi_tilde`;
  - `mu_tilde` and `nu_tilde`, which start at zeros;
  - `eta_tilde`, the full one, run only when wanted;
  - the private node `_eta_tilde_top`;
  - `V_tilde`, ONE row, reading `_eta_tilde_top`;
  - `H_tilde`;
  - `g`.

  Every row has `carries_K=False`.
- **Private nodes.** A leading underscore marks one: it is never wanted, never a slot. `_public(graph)` lists the
  public variables, and the engine accepts `_public + ('w',)` in `want`.
- **`_SHAPES` grows.** The tilde variables are full jets. `_eta_tilde_top` has the jet sizes `leaf`, like a leaf. The
  blocks have `'none'`, with the core's shape (`V`: `(nU_i, N_i)`; `H`: `(rL_i, nD_i, rR_(i+1))`), and no `W`.
- **The held check is general.** `_expected(name, setting)` gives the form of a whole variable: a shape (`f`,
  `r_block_sumsq`), a tuple of shapes over the index, or the pair for `g`. `_check_form` checks a held value against
  it recursively, with the path in the message (`the held g[1][0] has shape ...`).
- **`HeldVariables`** has one slot per public computed variable (ruling 4): the P5 eight, plus `f`, `r_block_sumsq`,
  the seven tilde names and `g`. Each has its shape comment.
- **The module docstring:**
  - the graph table with a `Start` column;
  - the private-node rule;
  - a paragraph on the short form;
  - the shape table with the new rows;
  - the slots rule.

**Tests.**
- **`tests/_jet_contract_cases.py`:** `STRINGS` gains the family `eta_top`, with the two lowered-corner readout
  strings (the reviewer's point 1). The pinned counts follow:
  - `test_jet_contract.py`: 31 strings, 11 readouts, `eta_top: 2`, the two-forms count 27, and the P1-count table
    row `('eta_top', 'full')`, 1;
  - `test_contractions_sharding.py`: `(31 + 12 + 5 + 3) * 4`.
- **`tests/_generic_twin.py`** (new):
  - The prototype's `gmul` over the grades `''` and `'z'`, with a truncated product over multi-indices (any jet
    shape), Taylor input jets lifted to the box, and the seed weighted, squared and flipped by itself.
  - `PointedCore` per frame. The value is taken at `(D, L, L, L)`; the adjoint on the manifold frame.
  - `V_tilde` is read from the FULL `eta_tilde` (ruling 7).
  - It imports `numpy` and `itertools` only, which an AST test pins.
- **`tests/test_sampling_variables.py`:**
  - `NAMES` gains the 8 new names, checked by definition over the jet matrix, three problems, nonuniform weights.
  - `TestAgainstTheGenericTwin`: the twin against the engine on two problems and five jet cases, the mixed-modes
    case among them. Also the AST independence test.
  - `TestTheShortForm`: the private node, recorded through a hook on its step, equals the slice of the full
    `eta_tilde` to `1e-13`, on every jet case and problem. `g` is bit-identical with and without `eta_tilde` wanted.
  - The shape test covers the blocks and `g`.
- **`tests/test_sampling_identities.py`:**
  - `TestGradient`: `<g, (V, H)>` equals the central difference of `f` along the direction, every jet case,
    nonuniform weight. It is exact to rounding, since `z` moves along the reference's `Jp`, linearly.
  - The held-recomputed test runs over every slot.
- **`tests/test_sampling_graph.py`:**
  - the graph table with `Start`;
  - the shape table with the `none` rows;
  - the slots are the public computed variables;
  - every sweep's declared start, and the zero starts in the arrays;
  - hand-derived step counts for 9 tilde wants, among them `('g',)` (two `eta_step`, `nu_step` 2d, ...) and
    `('g', 'eta_tilde')` (one `eta_tilde_step` more);
  - a private node is not wantable;
  - held `f`, `r_block_sumsq` and `g` of wrong forms.
- **`tests/test_held_variables.py`:** the 18 slots; unknown names `Jp` and `_eta_tilde_top`.
- **`tests/bridge/test_old_kernels_bridge.py`** (new, temporary; deleted in P20), at `n` = 0 to 3 on the three
  problems:
  - `z`, `mu`, `nu`, `eta` (old = `t!` × new);
  - `g` through the forward seed, against both `tv_probe_derivatives_transpose` and `ProbeDerivativesKind.transpose`;
  - the adjoint variables (`deta_tilde`, `tau_tilde`, `sigma_tilde` at slot `r` = new at `n - r` / `r!`;
    `dxi_tilde`'s two rows = `eta_tilde` at `n - u`);
  - at `n = 0`, `t3_probe` and `tv_probe_transpose`.

  The docstring lists every old kernel left out, with its slice (the dots: P7; the two-hole: P8; `sigma_hat`: P9;
  the dense functions: already in `test_sampling_definitions`).
- **`tests/test_dispatch.py`:** `test_jit_sampling_variables` jits `('f', 'g')`.
- **`tests/test_jax_absent.py`:** the engine script wants `g` too.

**Verified by script before the tests asserted it** (scratch scripts, not kept):
- every new name against the reference, worst 1.0e-15;
- the bridge conversions for `z`, `g`, `mu`, `nu` and `eta`, worst 3.6e-16;
- the old adjoint variables. Five candidate conversions were tried, and only `new[n - r] / r!` holds (1e-16); the
  others are at 0.3 or worse.

**Mutation pass** (scratch copy, deleted). 17 mutants, all killed. Among them are the reviewer's four:
- the slot box short at the low end;
- the lowered corner off by one;
- the seed not flipped;
- `nu_tilde` starting at ones.

And these:
- a term dropped from `mu_tilde`, from the top slot, from `V_tilde`, from `H_tilde`, or from the full `eta_tilde`;
- `nu_tilde` reading `nu_L`;
- `V_tilde`'s top read at the full corner, or reading `eta_D`;
- `xi_tilde` reading the input;
- `g`'s blocks swapped;
- `start_zeros` giving ones;
- the held check of `g` taking `V` shapes for `H`;
- private names wantable.

**Single-process runs:** the files above, with the module doctests and `test_import_graph`, pass on both
environments: 61 tests (1,435 subtests) on the current env, 61 on the floor.

**`subTest` labels:** builtins throughout (the guard ran).

## 2. The gate

The whole gate at `-n 3` (`--detach`, then `run_gate.sh wait`), with nothing else running. The environment was python
3.11.15, numpy 2.4.6 and jax 0.10.2. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1715 passed, 5 skipped, 1 xfailed, 1 warning, 72188 subtests passed   1625.1      8889       3378
    doctest-modules     PASS   314 passed                                                     45.1      1512      10478
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.6       284      11710
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.2       366      11637
    examples            PASS   14 examples, 0 failed                                           381       639      11367
    sphinx              PASS   0 warning/error line(s)                                      1013.1       636      11128

**No failure.** Against P5's committed run (1,706):
- The suite has 9 more tests:
  - the twin, 2;
  - the short form, 2;
  - the gradient, 1;
  - the starts, 1;
  - the bridge, 3.
- The module doctests are the same 314.
- The suite's peak memory is 8.9 GB, with 3.4 GB available at its lowest.

No process, lock or file of the gate is left in the repository.

## 3. Departures and choices

1. **The row type has six fields** (`start` added, ruling 3). P9's start by node name (apply's `nu_tilde[d]`) is not
   built: an undeclared start raises.
2. **`w_tilde_step` names the kind `'probe'`.** The steps take no kind; P9's apply and entries rows will have their
   own seed step.
3. **The new strings are written with `K`** (`'..->WKCi'`), per the reviewer's point 1. The `K`-free readout
   `'JWCa,Caib,JWCb->WCi'` named in the plan is not a literal of the library, so it is not in `STRINGS`: guard (ii)
   lists what the library writes.
4. **The twin computes `z` at `(D, L, L, L)`**: the corewise form of the base point, as the prototype did, since a
   manifold frame's Tucker cores have no variant for the mode edge.
5. **The bridge's `n = 0` row compares with `probing`'s plain functions.** `sampling_derivatives` has no order 0.
