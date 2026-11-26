# P9a: Engine E, the apply kind on both call types — handover

Slice P9a of `PART1_PLAN.md` (P9 split in two by the ruling of 2026-11-20), built to `P9_DESIGN_NOTE.md`,
`P9_DESIGN_REVIEW.md` and the rulings at the end of the plan's P9. Written by the implementer, 2026-11-20, against
`graded-messages` at `c0bae86b` (P8 committed). Nothing is committed.

## 1. What changed

**`t3toolbox/backend/sampling_steps.py`.**
- **`nu_step` and `eta_step` carry `K` on the right side** (`'Caib,JWCi,JWKCb->JWKCa'`, `'JWCa,Caib,JWKCb->JWKCi'`),
  for the twin `_Jp.nu_tilde`. Both strings were already in `STRINGS` (the tilde steps' terms); with `K = 0` they are
  the old contractions. The two plain literals leave the source; no contract or sharding count changed.
- **New site steps** for apply's adjoint, each a readout of one term:
  - `eta_top_step(mu, G, nu, w)`: the short form of `eta_step` (one readout per slot, at the lowered corner);
  - `eta_dot_top_step(... as eta_dot_step ..., w)`: the short form of `eta_dot_step` (three readouts per slot);
  - `V_top_step(top, w)`: the block from the top slots, at the leaf's corner;
  - `H_readout_step(mu, xi, nu)`: the corner of `H_term(mu, xi, nu_tilde)`.
  - No new literal: every string is among `STRINGS`.
- **The whole steps take `kind`** (a static string), all of them (`identity_step`, `g_step` too). A private
  `_modes(kind, value, weight, xnp)` gives the arrays of a value of the type of `z`, each with its weight row and its
  number of mode-edge axes, so `r_step`, `f_step`, `r_block_sumsq_step`, `Jp_norm_sq_step` serve probe and apply;
  `w_tilde_step` calls `jets.jet_to_dual_jet(kind, ...)`.
- **`close_step(bonds)`**, a whole step: bond `d` of a left side summed over its last axis (the closing ones). It
  makes apply's `z` (of `mu`, `rL_d`) and `Jp` (of `mu_R_dot`, `rR_d`; or of the raw `mu_dot`).
- **`start_node(node, size, xnp)`**: a start that names a node, the node times the closing ones of the end bond.
- The module docstring: the whole steps, `start_node`, a paragraph on the apply kind.

**`t3toolbox/backend/sampling_variables.py`.**
- **The rows in blocks** (ruling: shared plain and dot blocks): `_PLAIN_ROWS`, `_RESIDUAL_ROWS` (`r`, `f`,
  `r_block_sumsq`, `w_tilde`), `_DOT_ROWS` (both forms, the O-12 routes), `_DOT_SUM_ROWS` (`Jp_norm_sq`), and each
  kind's own: `_PROBE_Z_ROWS`, `_PROBE_ADJOINT_ROWS`, `_PROBE_JP_ROWS`, `_PROBE_CURVATURE_ROWS`; `_APPLY_Z_ROWS`,
  `_APPLY_ADJOINT_ROWS`, `_APPLY_JP_ROWS`, `_APPLY_CURVATURE_ROWS`. `_PROBE_ROWS` is the concatenation, **row for row
  the P8 graph** (checked by script against `HEAD`: both probe graphs equal).
- **Apply's rows** (the docstring's table "Apply's own row", generated from the data):
  - `z`: `close_step` of `mu`; `Jp`: `close_step` of `mu_R_dot` (needs `variations`) or `mu_dot` (needs
    `core_directions`);
  - `nu_tilde`, `nu_L_tilde`: `nu_step` from the start node `w_tilde`; `eta_tilde`, `eta_D_tilde`: `eta_step`;
    the short nodes `_eta_tilde_top`, `_eta_D_tilde_top`: `eta_top_step`;
  - `V_tilde`: `V_top_step` (corewise route over `_eta_D_tilde_top` first); `H_tilde`: `H_readout_step` (corewise
    route over `nu_L_tilde` first); `g`;
  - `nu_L_tilde_dot`: `nu_dot_step` over `nu_L_tilde`, from zeros; `eta_D_tilde_dot`: `eta_dot_step` over the tilde
    forms; `_eta_D_tilde_dot_top`: `eta_dot_top_step`; `V_tilde_dot`: `V_top_step`; `H_tilde_dot`: `H_tilde_step`
    with `(mu_dot, mu, xi_dot, xi, nu_L_tilde_dot, nu_L_tilde)`; `Kp`.
- **`_deps(rule)`** = reads plus a start node. Used by the closure, the dtype count, the corewise drop, and (through
  the start field) the twin and corewise renames. `_STARTS` moved up beside the row constants.
- **`_JP_PATHS`**, one table keyed `(kind, call)`, each path in its graph's own names; `_JP_PATH` and
  `_JP_PATH_COREWISE` are gone. `_jp_twins` renames a start node on the path (`'w_tilde'` -> `'_Jp.w_tilde'`).
- **`_corewise(rows, kind)`** drops by `_deps`, renames the start too, and reads the kind's path; new
  **`_manifold(rows, kind)`** checks the path and adds the twins. Both invariants raise `RuntimeError` (ruling 2).
  `_GRAPHS` is built by one loop over `_ROWS = {'probe': ..., 'apply': ...}`: four graphs (probe 56 / 41 rows, apply
  49 / 35). `_PROBE_GRAPH`, `_PROBE_COREWISE_GRAPH` stay as aliases.
- **The zero names** (ruling 1): `_ZERO_NAMES = {'apply': {xi_tilde, mu_tilde, xi_tilde_dot, mu_tilde_dot}}` and
  `_ZERO_REASON`. They are not on apply's graphs (an import-time check raises if a graph makes one). Asking for one
  raises: "unknown variable 'mu_tilde' for the apply kind on a manifold call ('mu_tilde' is zero for the apply kind:
  the seed sits on the ones that close bond d, which a left side or a leaf never reaches); the names are ...". To
  switch to zero rows later: delete the entry, add four site/sweep rows of zeros.
- **`_names(kind, call)`**, the private reader of a call's names (the public `_public(graph) + ('w',)`); the engine's
  unknown-name check uses it. No public reader (ruling 1).
- **Shapes by kind**: `_KIND_SHAPES['apply']` gives `z`, `r`, `w_tilde`, `Jp` the index `'one'` (`J + W + [K] + C`, no
  mode edge); `_entry(name, kind)` reads it; `r_block_sumsq` is `F` for apply. `_Setting` has the `kind`.
- **The import-time reachability loop** runs over every kind's corewise graph.
- **`_run`**: whole steps get `kind=setting.kind`; a sweep whose start is not `'ones'`/`'zeros'` starts from
  `start_node(values[start], size)`.
- **Docstrings**: the `start` field and `_deps`; a section "The apply kind" with its table; the shapes paragraph; the
  closing sentence (probe and apply built); the engine's `kind`, `y`, `weight` (apply's shapes), `Vc` ("only the probe
  kind's `z` reads it; given to apply it is read by nothing and accepted", ruling 3) and the return; two doctest
  examples (apply's shapes, `z` = `mu[d]` summed; the zero-name error), outputs from a real run.

**Tests.**
- **`tests/test_sampling_apply.py`** (new, 13 tests):
  - by definition, every apply name on a manifold frame and on a corewise point (through section 9), `K = ()` and
    element by element `K = (2,)`, the jet matrix, three problems;
  - **the names comparison** (ruling 1): per kind and call, the engine's names against `sdef.__all__`: probe manifold
    equal; apply manifold equal but the four zero names; corewise, the section-9 spellings but the names in the `R`
    and `U` bases (`nu`, `eta`, `nu_tilde`, `eta_tilde`, `xi_U_dot`, `mu_R_dot`), and for apply the zero names; and
    `_ZERO_NAMES` exactly;
  - a zero name raises with its message, on both call types;
  - `nu_tilde` and `nu_L_tilde` against the jet product of the seed and `nu`, `nu_L`; the start is the seed times
    the closing ones;
  - **tails** (ruling 4): on `rL_d != rR_d`, `z = mu[d].sum(-1)`, `Jp = mu_R_dot[d].sum(-1)` (the `R` end bond),
    the raw `Jp = mu_dot[d].sum(-1)`, `Jp` against the reference;
  - the identities per element of `K = (2,)` on both call types: `<Jp, omega^2 r> = <p, g>`, `<p, JtJp> =
    Jp_norm_sq`, `JtJp` = `g` with `Jp` held as `r`;
  - the corewise call at `(D, L)` against the manifold call;
  - **the start node** (ruling 4): `_deps`, the twin's start `_Jp.w_tilde`, the corewise `nu_tilde` (the renamed
    `nu_L_tilde`, reading `L`) starting from `w_tilde`, the closure reaching the seed through the start; a held
    float32 seed is the start and makes the result float32;
  - a held `z` has the apply shape (probe's tuple form and a wrong shape raise);
  - step counts for `('z',)`, `('z', 'g')` on both call types (no plain `nu`/`eta` sweep), `('Kp',)`, `('JtJp',)`;
  - a `Vc` is read by nothing.
- **`tests/test_sampling_graph.py`**: the probe twin and `carries_K` tests read `_JP_PATHS`; `CountingSteps` patches
  the steps of every graph; new: apply's docstring table row for row; every graph made of the blocks, its deps made
  by it or a store or an argument, every sweep's start a named value or a node it makes; one table of the paths (keys,
  names, twins, a twin's start node; the `RuntimeError` of an incomplete path); apply's starts; apply's routes; the
  zero names are not made; the direction-masks test reads every graph's deps. The `NotImplementedError` test is now
  for entries only.
- **`tests/test_sampling_corewise.py`**: `_JP_PATHS['probe', '_corewise']`, `_corewise(rows, 'probe')`.
- **`tests/bridge/test_old_kernels_bridge.py`**: `TestTheApplyKind`, at `n` = 0 (`t3_apply`, `tv_apply`,
  `tv_apply_transpose` for `g` and `JtJp`, `t3_apply_corewise_hessian`, `ApplyKind.transpose`) and `n` = 1 to 3
  (`t3_apply_derivatives`, `ApplyDerivativesKind` point_forward, forward, transpose for `g` and `JtJp`,
  corewise_hessian), `K = ()` and `(2,)`, three problems; the docstring's conversions.
- **`tests/test_dispatch.py`**: `test_jit_sampling_variables_the_apply_kind` (manifold `z`, `f`, `g`, `Kp`, `JtJp`,
  `Jp_norm_sq`; corewise `g`, `Kp`; tails at both ends).
- **`tests/test_jax_absent.py`**: apply's `z`, `Kp`, `JtJp`.

**Verified by script before the tests asserted it:**
- every apply name against the reference, manifold, `K = ()` and `(2,)`, the jet matrix, three problems: worst
  7.8e-15 (`JtJp`); corewise: worst 5.9e-15;
- `nu_tilde = w_tilde * nu` (jet product): worst 3.2e-16; the reference's four zero names are exactly zero;
- the bridge (`z`, `g`, `Jp`, `JtJp`, `Kp`, both kinds, `n` = 0 to 3, both `K`): worst 3.1e-15.

**Single-process runs:** the changed and related files (the sampling tests, the bridge, the jet-contract and
sharding tests, jax-absent, the import graph, the contractions interpreter) with the two modules' doctests pass on
both environments: 167 passed, 1 xfailed (27,602 subtests) current; 167 passed, 1 xfailed on the floor (after
`test_sampling_variables`' call of the private `_read_setting` gained the kind). `test_sampling_apply.py` was re-run on
the floor after the mutation pass added its check: 13 passed.

**`subTest` labels:** builtins (the guard ran; the graph keys are passed as `repr`).

**Mutation pass** (scratch copy, deleted). 20 mutants, all killed:
- **The reviewer's:**
  - `close_step` of `Jp` using the `L` end bond (on the tails problem, the first `rL_d = 2` of the `rR_d = 4` entries);
  - a wrong seed start: `r` (neither weighted nor turned), `'ones'`, and the seed read turned the wrong way.
- **The others:** `close_step` squeezing; the twin's start not renamed; `_deps` ignoring the start (killed at import,
  by the reachability loop); the dtype count without the start node; `H_tilde_dot` with the tilde and the tilde-dot
  swapped; `eta_dot_top_step` without its `L_dot` term; `eta_top_step` at the corner, not lowered; `V_top_step` at the
  full corner; apply's `r` not subtracting `y`; `Jp_norm_sq`'s weight not squared; apply's `z` shape entry missing;
  `nu_step` without `K` on the carry; the zero-name message dropped; the raw `Jp` route reading the frame dot (killed
  at import).
- **Two first survived, as equivalent mutants:** the corewise drop ignoring start nodes, and the corewise rename not
  carrying the start. Apply's start node `w_tilde` is neither dropped nor renamed on a corewise point, so both are
  no-ops today. `test_deps_and_the_renames_carry_the_start` now adds two later rows (a sweep starting from the
  `U`-basis `eta`, which must be dropped; one starting from `nu_L_tilde`, whose start must be renamed `nu_tilde`);
  both mutants are then killed.

## 2. The gate

The whole gate at `-n 3` (`--detach`, then `run_gate.sh wait`), with nothing else running. The environment was python
3.11.15, numpy 2.4.6 and jax 0.10.2. The summary, copied:

    step                result counts                                                      seconds   peak_MB  minavl_MB
    suite               PASS   1755 passed, 5 skipped, 1 xfailed, 1 warning, 80852 subtests passed   1367.1     10821       1560
    doctest-modules     PASS   314 passed                                                     47.8      1465      10588
    doctest-quickstart  PASS   1 file(s), 0 with failures, 0 failed example(s)                 2.7       283      11771
    doctest-pages       PASS   48 file(s), 0 with failures, 0 failed example(s)               14.4       361      11699
    examples            PASS   14 examples, 0 failed                                           378       637      11458
    sphinx              PASS   0 warning/error line(s)                                      1021.3       627      11333

**No failure.** Against P8 (1,734), the suite has 21 more tests: `test_sampling_apply.py` (13), the graph tests (6),
the bridge's apply kind (1), the jit of the apply kind (1). The module doctests count the same (314): the two new
examples are in the engine's one docstring.

**Memory** (both figures, as asked): the peak RSS summed over the workers is 10.8 GB (13.4 GB at P8; a sum that counts
shared pages more than once); **the lowest MemAvailable is 1.56 GB**, down from 1.9 GB at P7 and P8, and just above the
1.5 GB line. The suite took 1,367 s (1,258 s at P8): `test_sampling_apply.py`'s by-definition sweeps, against the
dense reference on every name, are the likely cause of both. Worth watching at P9b, which adds entries the same way.

No process, lock or file of the gate is left in the repository.

## 3. Departures and choices

1. **Every whole step takes `kind`**, not only the five the note named: `_run` passes it to every whole step, which
   keeps one calling convention.
2. **The corewise path is spelled in the corewise graph's own names** in `_JP_PATHS` (the review's "one table keyed
   like `_GRAPHS`"). For probe and apply it is then spelled as the manifold path, by design now, and the import-time
   check (every path name is made by its graph) plus the collision check guard it.
3. **The start node is not a read**; `_deps` adds it. The corewise rename and drop of a start node are equivalent
   today (`w_tilde` is neither renamed nor dropped); they are written and tested by construction for P9b and later
   kinds.
4. **`_manifold(rows, kind)`** is new, the manifold counterpart of `_corewise`, so the path check is the same for both.
5. **The apply `Jp` raw route** is `close_step` of `mu_dot`: the raw direction's output needs only the left dot sweep.
