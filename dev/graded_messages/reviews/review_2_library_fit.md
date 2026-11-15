# Review 2: how the graded-messages proposal fits T3Toolbox

Reviewed against T3Toolbox `3dc15d9` and the handoff package as found on 2026-11-14. Nothing in either tree was modified. Paths are relative to the repository root unless they start with a handoff file name.

## Overall judgement

The algebra and the description of what the library computes today are accurate: all fifteen check scripts reproduce their recorded output at `3dc15d9`, and the counts and name tables I sampled are right. The fit problems are in the layers above the algebra. The proposal's containers and generated subscripts cut against three things the library treats as settled: plain-tuple backend data, literal grouped strings as the tested specification, and a CG solve that is compiled whole with the Hessian product inside it. The execution-plan layer also cannot yet express today's behaviour as its default, which the maintainer's "no loss of performance" requirement needs. Much of the outer-block and policy benefit is reachable from hooks the library already has, without the kernel rewrite, and several of the proposed reuse savings were already measured and deliberately not built.

## Findings

### Conflicts with library rules

**F1. Generated subscripts remove the strings that three test suites and the style rules treat as the specification.** Severity: high. Verified.
- Proposal: `PROPOSAL.md` "Layer 1: operands know their stacks"; `API_IMPACT.md` adapter 8; `README.md` "Things learned" (sharding test only).
- Library: `tests/test_contractions_interpreter.py` (`scan_vocabulary` l.414, `scan_literal_call_sites` l.428, `HISTORICAL` l.74 with 293 entries, `TestVocabularyCompleteness`, `TestVocabularyOracle`, `TestCallSiteConsistency`); `tests/test_contractions_sharding.py` (`_scan_vocabulary`, the `> 100` assertion); `docs/contributor/batching_internals.md` "The interpreter era"; `docs/contributor/contractions_internals.md` §9; `docs/contributor/signature_style.md` (the `contract` call-site exception).
- Evidence: the loop oracle, the frozen supplement table, the call-site check and the sharding sweep all take their vocabulary from an AST scan of string literals. The scan regex requires an uppercase group letter, so rule strings such as `'a,aib,i->b'` are invisible to it. As strings move from literals to assembled, the completeness test keeps passing while covering less; only the `> 100` count would eventually fail. The contributor docs call the self-maintaining scan "load-bearing" because four hand inventories were found wrong. The proposal lists an adapter for the sharding test only.
- Suggestion: decide this before step 2. Either keep grouped strings literal at the term-list layer (a term table of literal strings is still one table per rule), or have the generator expose a pure enumerator that all three scans union with the AST result, with every enumerated string pinned in `HISTORICAL`. Add a test that the enumerated set equals what the drivers actually emit.

**F2. Graded dicts, operand-signature objects and a dict-returning `evaluate` conflict with the backend's plain-data rule.** Severity: high. Verified. Challenges a decided point in part (the shape of `want`/`have` at the public backend surface, not the dependency-graph idea).
- Proposal: `PROPOSAL.md` "Components are keyed by grade", layers 1 and 3; `API_IMPACT.md` adapters 4 and 5; `EVALUATOR.md` "One entry point".
- Library: `CLAUDE.md` backend razor; `docs/contributor/parameters_not_closures.md` "Rejected: structured backend data (design B)"; `dev/HANDOFF.md` ruling of 2026-11-02 (an arc was paused because a helper returned "a mutable dict of shapes (opaque, untyped)"; backend inputs and outputs are "immutable objects composed of basic types"); `docs/contributor/deferred_and_rejected.md` "A kwarg-unified derivative interface" (rejected for return-rank polymorphism); `backend/common.py` `partition_static` l.710.
- Evidence: `evaluate(..., want=...)` returns a dict whose contents depend on an argument, the pattern the ledger records as a locked rejection. `partition_static` walks tuples and lists only; I ran it on a sweep containing a dict and the dict came back as one dynamic leaf with its shape ints and mask inside, so a uniform `LocalModel` holding such a container would hand static structure to jax.
- Suggestion: specify graded messages as fixed-slot tuples (plain, `t`, `z`, `tz`, with `None` for absent) and keep named whole-operation functions with fixed return shapes on top of the evaluator. If `want` stays open-ended, confine it to one internal function and give the public surface an enumerated set of operations.

**F3. The executor has to live inside a `lax.while_loop`, and the plan has no route into it.** Severity: high. Verified. Touches the decided point "the plan is passed to every call" without asking to reverse it.
- Proposal: `EXECUTION_PLANS.md` "The plan and the drivers" (`executor='host_loop'` default; "the sample loop is an interchangeable executor"); `EVALUATOR.md` "An operation is a per-block kernel plus the sample loop".
- Library: `backend/optimizers.py` `_cg_body` l.1177 (`Hd = lm.hvp(d)`), `_cg_core` l.1223 (`xwhile`), `_cg_solve` l.1237 (jitted whole, key = the model's pytree); `docs/contributor/scan_body_principles.md` "Where jitting the caller IS the right remedy"; `dev/sharded_fitting_plan.md`; `docs/chunking.md` "Sharding over `W`".
- Evidence: under `use_jit` every Gauss-Newton and exact product runs inside the compiled CG. A host loop there unrolls at trace time, cannot stream, and cannot write per-sample outputs into host arrays. `hvp(p)` takes one argument, so a plan must ride as a static argument of `_cg_core` or as value-hashed aux, which is where `chunk_size` lives today. The sharding plan in `dev/` puts one `shard_map` around the whole step with `psum` at `sumsq` and `transpose`, and `docs/chunking.md` records that a dynamic slice on a sharded axis under plain `jit` all-gathers `W`. The library therefore composes sharding across devices with a chunk scan within a device; a single `executor` enum makes them alternatives.
- Suggestion: state the plan's carrier through the jitted CG. Make the executor a composition of levels (across devices, in graph, on host). Restrict the host level to operations the Newton loop already runs on the host (objective, gradient, model build) unless CG itself moves to the host. Reconcile with decisions 1 to 3 of `dev/sharded_fitting_plan.md`.

**F4. The proposed names collide with registered letters and with the glossary's naming rule.** Severity: medium. Verified.
- Proposal: `PROPOSAL.md` "Edge variables", "Taylor jets", name table; `EVALUATOR.md`; `WORKED_EXAMPLE.md`.
- Library: `docs/naming_conventions.md` "Index letters"; `docs/glossary.md` (entries and symbol table); `docs/contributor/curvature_internals.md` "The T/S transforms"; `dev/glossary_naming_decisions.md` (naming rule); `dev/HANDOFF.md` ruling of 2026-11-11 (no general naming rules; an enumerated token map and a do-not-touch list).
- Evidence:
  - `_t` collides with `t`, the reserved order-axis letter in strings and shape comments; `_z` with the probe output; the alternative `_r` with TT ranks and the residual.
  - The gauge matrix `S` (`nu_L = S nu_R`) is the library's `T` (`ν_c = Tν`); the library's `S` is the other transform.
  - `phi` collides with `Φ`, the parametrization map. `p` for the highest order collides with `pp`; the rule is that order "has no letter" and is spelled `order`.
  - "Center cores" are the glossary's **carriers**; "centers" already means `H_i = L_i T_i`. "Messages" already names the padded-frame `Ψ`.
  - Rule `RIGHT` builds `mu` from the `L` store, which the code calls the "leftward" variables.
  - `evaluate` exists as `fitting.LocalModel.evaluate(p)` (the model value); `sweep` is the noun for the precomputed frame sweep in every kind signature.
  - `evaluate`, `sitewise` and `fill` lack the family prefix that public backend functions carry.
  - The glossary's rule makes the public API spelling canonical, so with adapters (case 1) `sigma`, `tau` and `*_tilde` stay canonical and the new names are a second vocabulary.
- Suggestion: turn the name table into an enumerated token map checked against the glossary's symbol table, use "carrier", and choose grade markers outside the index-letter registry.

**F5. Binary association as a general rule, and `Plan.path`, go against the recorded contraction policy.** Severity: medium. Verified in the docs; not measured.
- Proposal: `PROPOSAL.md` "Requirements on the drivers" ("a three-factor rule is evaluated as two binary graded products"); `EXECUTION_PLANS.md` `path`.
- Library: `docs/contributor/fitting_internals.md` "numpy einsum: force BLAS-eligible pairwise paths; jax: one big einsum"; `docs/contributor/contractions_internals.md` §6; `docs/contributor/deferred_and_rejected.md` (numpy two-operand entry: "numpy is not the performance path").
- Evidence: the measured policy is one einsum per contraction on jax, with hand factoring only where it was measured to win (the factored assemblies). The motivation for `path` in the package is an eager-numpy timing.
- Suggestion: make "one einsum per term on jax" the rule-level default and keep factoring as a per-rule declaration. Drop `path` from the first `Plan`, or scope it to numpy.

**F6. Memoizing step bodies on the whole plan invites unbounded specializations.** Severity: medium. Not verified by running.
- Proposal: `EXECUTION_PLANS.md` ("a plan is hashable, so it can be part of the key").
- Library: `docs/contributor/scan_body_principles.md` principle 4 ("keys should be structural"; a key that varies with the data "grows the cache and the compile count without bound").
- Evidence: `fill` derives block sizes from a memory budget, so they vary between runs. The existing precedent, `_wchunked_sum_step(chunk_fn, cs, w_axes)`, keys on one resolved integer per fit.
- Suggestion: key each factory on the fields its driver reads, and quantize sizes produced by policies.

**F7. Frame consistency as a safe-mode precondition does not match the contract catalog.** Severity: medium. Verified.
- Proposal: `PROPOSAL.md` "Frame consistency", proposal items 1 and 3.
- Library: `docs/contributor/numerical_contract_catalog.md` (precondition against caveat; frontend-only enforcement; probing is precondition-free); `backend/geometry.py` `base_point` l.304, 530, 763, 1026; `fv_base_point_tangent` l.118; `t3_left_orthogonal_norm_sq` l.103; `frame_variations_format.py` `to_t3` l.721.
- Evidence: once one family is authoritative nothing reads the other carrier, so by the catalog's own test consistency is a caveat, and caveats are not enforced. The backend already reads `(D, L)` everywhere I looked (four `base_point` methods, the regularizer's base-point tangent, the point norm, the kinds' `corewise_hessian`). The readers of `(D, R)` that I found are `T3Frame.to_t3` and what calls it (`to_dense`, `allclose`, `orthogonalize`). I did not search the uniform frontend for others.
- Suggestion: record `(D, L)` as authoritative and change or document `to_t3`. Add a precondition only if plain terminals or objective-through-frame are adopted, and then as a frontend check on a cached residual with a catalog amendment.

### Uncovered or broken features

**F8. Today's behaviour is not one `Plan` value.** Severity: high. Verified.
- Proposal: `EXECUTION_PLANS.md` "Design constraints" (the default plan must reproduce today's choices), `Plan` fields, policy `current`.
- Library: `backend/probing.py` l.211, 302, 523, 1069, 1155 (on uniform the plain kernels vectorize every per-site step over sites, combines and assemblies included); `backend/sampling_derivatives.py` l.533, 889, 1257 (jet lifts vectorized) against l.504, 853, 1440, 1620 (jet combines and assemblies mapped over sites); `backend/fitting.py` l.698 (`chunk_size` is a kind field, default 100); `t3toolbox/optimizers.py` `_resolve_chunk_size` l.159 (frontend only; uniform `probe_derivatives` only); `docs/contributor/parameters_not_closures.md` "Options are not parameters" (lists `chunk_size` among kind parameters).
- Evidence: site handling differs per kernel and between the plain and jet families today, chunking exists for one kind on one layout, and the chunk size is part of the kind's identity. `Plan.sites` is one value, `Plan.assembly=None` means no inner block, and `chunk_size` would have a second home.
- Suggestion: define `current` first and concretely, as "each rule keeps its present choice unless the plan overrides it", and say whether `chunk_size` remains a kind field or moves.

**F9. The per-sample residual has more readers than the proposal lists.** Severity: medium. Verified. Touches a decided point (residuals on request) without asking to reverse it.
- Proposal: `EVALUATOR.md` "Decisions", "What can be kept".
- Library: `backend/optimizers.py` `LocalModel.residual` (a pytree leaf, l.2329), `misfit` l.264, `_curvature_block` l.315, `_resolution_eps(lm.residual)` l.2232; `backend/optimizer_display.py` l.467 (`block_sumsq(info.lm.residual)`), l.471 (validation).
- Evidence: the verbose display's per-(mode, order) table and every exact product read the stored residual. `block_sumsq` is a third reduction over `W` beside `sumsq` and `transpose`.
- Suggestion: add the unweighted block sums as a summed component, and state the default `want` for verbose and exact runs.

**F10. The sample loop needs a per-kind slicing hook that does not exist for kept components.** Severity: medium. Verified.
- Proposal: `EVALUATOR.md` "What to keep" ("the sample loop slices them per block exactly as it slices the data").
- Library: `backend/fitting.py` `_flat_gather` l.153 and the six `_take_*` functions; `w_axes`, `n_measurements`; `backend/uniform_fitting.py` `_ptake_*`; `backend/utv_sampling.py` l.112 to 125 (uniform sweep layout); `assemble_tt_variation_jets` docstring.
- Evidence: `W` sits at a different axis per kind and layout (leading; after the order axis; after the site axis on uniform; `(d,) + W` for an index). `W` may have several axes; `take` flattens them by reshape and chunking today falls back to dense for multi-axis `W`. The uniform sweep is `(masked frame, packed sample, [packed pp,] shape, components)`, so what is kept is more than letter-and-grade components.
- Suggestion: specify a reshape-free `W`-slice hook beside `take`, covering sample, data, residual weight and each kept component on both layouts.

**F11. A `gn_product` hook and `want`/`have` enlarge an extension surface that is under a standing open question.** Severity: medium. Verified.
- Proposal: `EVALUATOR.md` "Two kinds of fusion"; `EXECUTION_PLANS.md` "Parts can land without the refactor".
- Library: `dev/OPEN_QUESTION_extension_surface.md`; `docs/contributor/deferred_and_rejected.md` "A `LocalContext` bundle"; `t3toolbox/fitting.py` `gn_hessian` l.434; `backend/optimizers.py` `hvp` l.297 (two branches) and `gn_quadratic` l.291.
- Evidence: the composition forward-then-transpose is written in three places, and the exact branch adds the curvature block before one projection. The fused form needs an orthogonal frame and a gauged tangent, so it does not apply to the corewise frame `(B, G, G, G)`.
- Suggestion: treat the hook as part of the kind-protocol rethink the ledger already defers, and list all call sites.

### Wrong or outdated claims about the library

**F12. Tail ranks above 1 already work in the current kernels.** Severity: medium. Verified by running.
- Proposal: `README.md` requirements; `PROPOSAL.md` "Not checked" and step 4 ("including tails larger than 1").
- Library: `backend/probing.py` `compute_mu` l.247 (carry of ones sized by the first bond); `backend/fv_conversions.py` `t3_orthogonal_representations(squash_tails=True)`; `backend/padded_frames.py` `_rank_stores` ("a frame's boundary bonds are 1").
- Evidence: with `tt_ranks = (2, 3, 4, 3)`, `t3_probe` and `t3_apply` match dense to 1e-15. On a frame built with `squash_tails=False` (tails 2 and 3), `tv_probe` matches dense to 5e-16 and `tv_probe_transpose` satisfies the adjoint identity to 3e-16. Default frames have tails of 1, and the padded-frame code refuses anything else.
- Suggestion: restate the requirement as "must not regress", and add these two cases to the prototype's checks, which currently use boundary size 1.

**F13. The rename pass is finished; the pending work is a docs pass.** Severity: low. Verified.
- Proposal: `README.md` "How to resume" ("held plans for a rename pass at that time").
- Library: `dev/HANDOFF.md` ("THE STORE RENAME ARC IS COMPLETE: R1-R6 COMMITTED"), before `89ac203f`.
- Evidence: the name table already uses post-rename names (`H_tildes`, `V_tildes`), so it is not stale. What remains in flight is F16.

**F14. Small inaccuracies.** Severity: low. Verified.
- The Gauss-Newton product the optimizer runs is `backend/optimizers.py::LocalModel.hvp`, not the frontend `gn_hessian` named in `README.md` and `EVALUATOR.md`.
- `base_point` is a method of four geometry classes and a protocol member, not a module function.
- The uniform sweep wrapper is a 4-tuple for plain kinds and a 5-tuple for jets (`API_IMPACT.md` adapter 4 says 4).
- `chunk_size='auto'` is resolved in the frontend, not by the backend optimizers (`EXECUTION_PLANS.md` "Automatic sizing").
- "Sequential map for combines and assemblies; vectorized for lifts" (`EXECUTION_PLANS.md`, the decisions table) describes the jet kernels. The plain kernels vectorize combines and assemblies over sites on uniform (`backend/probing.py` l.302, 1155).
- By AST the four files hold 194 `contract` calls, 21 of them with a non-literal first argument; 117 distinct literal strings is right.
- The exact-equality survey (`API_IMPACT.md`) omits `tests/backend/test_uniform_curvature.py` (36), `tests/test_sharing.py` (24) and `tests/test_optimizers_frontend.py` (10). The list of tests naming sweep-split functions omits `tests/backend/test_uniform_fitting.py`.
- "Centers described as padding": the glossary already says carriers "hold the base point".

**F15. The exact-mode reuse is described as uncounted; the library has counted it.** Severity: low. Verified in the docs.
- Proposal: `EVALUATOR.md` "What to keep" ("a larger part, not yet counted").
- Library: `docs/contributor/curvature_internals.md` "The T/S transforms"; `docs/contributor/deferred_and_rejected.md` "Exact-Newton refinements".
- Evidence: storing the residual-dependent state is recorded as about +60% of the jet sweep resident for about 10 to 15% of the two-hole, and the `nu` re-sweep as 1 to 3% of the two-hole. See F19.

### Work in flight

**F16. The 2026.5.0 docs pass is about to rewrite the docstrings of the kernels the proposal replaces.** Severity: medium. Verified.
- Proposal: `PROPOSAL.md` order of work, step 1.
- Library: `dev/glossary_naming_decisions.md` items E22 and E8 (13 work rows in `backend/probing.py`, 33 in `backend/sampling_derivatives.py`); `dev/HANDOFF.md` (switch slices W2 to W5, the timing final stop, the docs pass, then the release commit).
- Evidence: E22 exists because "leftward" and "rightward" do not say whether they mean side or travel direction; its replacements ("the left edge variables mu (accumulated left to right)", "adjoint right-side variation edge variables tau_tilde") fix a vocabulary that step 1 would then restate.
- Suggestion: settle the side-and-direction wording once, before the docs pass reaches those rows, even if no code changes.

**F17. Several backlog items change the proposal's premises.** Severity: medium. Verified in `dev/` and the ledger; effects not tested.
- float32 end to end (`dev/HANDOFF.md` backlog): a ready patch touches the plain scan inits, which are the proposal's boundary vectors, with the rule that each init takes the dtype of all its scan's operands.
- Data-shaped residual weight (`dev/data_shaped_residual_weight_plan.md`, deferred): the weight becomes a traced per-sample leaf, a further per-sample input for the sample loop.
- Sharded fitting (F3).
- The toolbox reference paper (`dev/paper_scope.md`): `docs/symmetric_probe_derivatives.tex` is a core source and is written in the binomial form, so the Taylor decision should precede it.
- The codebase-wide style audit announced in the 2026-11-02 handoff entry ("Nick's own read first; breaking release").
- The leftover-chunk overshoot, which bears on padding the last block.

### Migration path

**F18. Steps 2 and 3 can and should be shown bitwise-neutral.** Severity: medium. Not verified by running.
- Proposal: `README.md` "Things learned" ("a port that reorders contractions cannot offer that"); `PROPOSAL.md` steps 2 and 3.
- Library: `dev/store_rename_battery.py`; the commit messages of `ca13b37` and `b263a1d` (bitwise batteries of 30k and 26k arrays); `docs/contributor/contractions_internals.md` §6 (the numpy path is keyed on the grouped string).
- Evidence: a term-list port that emits the same grouped strings in the same term order uses the same numpy paths and the same single einsums on jax. The tolerance argument is needed only where numbers must move: closure by contraction with a ones vector (step 4) and the `t!` change of variables (step 5).
- Suggestion: order the slices so every structural change is gated by the battery, and put each numerical change in its own slice with its tolerance argument.

**F19. What pins behaviour, what must change, and what the adapter list lacks.** Severity: medium. Verified by reading the tests.
- Pins that hold through the port:
  - `tests/backend/test_from_sweep_reuse.py` (the twelve hooks, with `W`, `K` and `C` stacks);
  - `tests/test_jet_recurrence.py` with `tests/_jet_trs_twins.py`;
  - the block-sum, jvp and dense oracles of the exact Hessian;
  - the trace-count assertions in `tests/test_dispatch.py`;
  - the garbage-in-padding equalities in `tests/backend/test_uniform_curvature.py`, which are structural and should stay exact;
  - "exact gradient equals the Gauss-Newton gradient bit for bit" and `_gauss_newton_view`.
- Must change: the AST-scan suites (F1); tests that index the sweep tuple (`tests/test_dispatch.py` l.1179, `tests/backend/test_uniform_curvature.py` l.581); the closed-form floors `_jet_floor_bytes` and `_two_hole_jet_floor_bytes`.
- Missing from the nine adapters: `partition_static` or a pytree registration for the native container; a `W`-slice for kept components (F10); block sums (F9); dispatch rows for each new public backend function, host loops included (a recorded rule in `dev/HANDOFF.md`); a run on the jax 0.4.30 / numpy 1.26 floor.
- Step 1's convention note belongs in `dev/` until the code matches, since `docs/` pages are doctested and describe shipped names. The pairing test and the cheap checker are sound independent slices.

### Already in the library

**F20. The outer sample block and the policies do not need the drivers.** Severity: medium. Verified by reading; not prototyped.
- Proposal: `EXECUTION_PLANS.md` "Relation to the graded-messages refactor" ("the plan needs the drivers").
- Library: `SamplingKind.take`; `Problem.local_model(x, sample, data, frame=)` and `Problem.objective(x, sample, data)`; `_wchunked_sum` and `_wchunked_concat` (`backend/sampling_derivatives.py` l.1698, the add-or-concatenate reducer); `estimate_chunk_size(n_shards=)`; `max_chunk_size_within(target_bytes=)`; `_resolve_chunk_size`.
- Evidence: a per-block kernel over an explicit `(sample, data)` already exists at the kind level, with the same summed-or-kept reducer split the mockup uses. `max_chunk_size_within` is the `fill` policy for the inner block and `estimate_chunk_size` is `balance`. Only `assembly`, `sites`, `orders` and `path` have to reach inside kernels.
- Suggestion: consider landing `samples` and the policy interface as a kind-level wrapper first. It gives users the control the maintainer asked for, tests the `Plan` design under jit and sharding, and leaves the kernels alone.

**F21. The three "savings that need no memory" were examined, and two need a channel the geometry protocol lacks.** Severity: medium. Verified in the docs and code.
- Proposal: `EVALUATOR.md` "Frames and reuse", "What this means for the plan"; `scripts/reuse_check.py`.
- Library: `docs/contributor/precompute_and_caching.md` ("retract→frame fusion: recorded, not built"; "retaining `HH`/`S` on the frame: examined and left alone"; the checklist: a new state channel "is a design decision"); `docs/contributor/curvature_internals.md` (the transforms, with recompute as the house low-memory default); `Geometry.frame` in `backend/optimizers.py` l.126.
- Evidence: reading the probe value off the frame needs the Tucker centers. `geom.frame(x)` returns only the frame, and `Problem.local_model(frame=...)` accepts an aligned padded frame with no centers. I confirmed the identity itself at a default frame (1e-15). Evaluating trial points through their frames is the fusion the audit declined. The fused paths would also merge the `sweep`, `residual` and `trial_objective` fields of the public `PhaseSeconds` record.
- Suggestion: present these as reopening recorded decisions, with the recorded magnitudes, not as free.

**F22. Smaller overlaps.** Severity: low. Verified.
- `contract` already serves every stack configuration from one string (groups may be empty), so operand signatures must justify themselves by what they add beyond that.
- The two-hole already calls the Gauss-Newton tangent kernels with `U = L = R = G` (`backend/probing.py`, `t3_probe_corewise_hessian_from_sweep`).
- `compute_nu` is already `compute_mu` on reversed inputs.
- Memoized step factories and a cached `lax.map` adapter exist (`backend/common.py` l.289).

## Coverage table

| Feature | Status | Reason |
| --- | --- | --- |
| Uniform (supercore) layout | needs extension | Not prototyped; site-axis choices are per kernel (F8); bond arrays of length `d+1` on supercores untested |
| Padded frames and masks | just works for consistency; needs extension for reuse | `fv_padded_frame` and the zero-point frame are consistent (verified; the aligned loop was not run); they carry no centers (F21); mask-once wrappers unaffected |
| `W`, `K`, `C` stacks; `sum_over_probes` | needs extension | Prototype has one `W` axis; multi-axis `W`, `K` on residuals, shape-`C` objectives not shown (F10) |
| Residual weights | needs extension | Seed scaling composes on paper; untested with fusion; the data-shaped weight is pending (F17) |
| Regularizer | just works | Added in `LocalModel`, above the evaluator, provided the evaluator returns raw `J^T r` |
| Minibatching and `take` | needs extension | `take` covers sample and data only (F10) |
| `w_axes` / `n_measurements` / `sumsq` | needs extension | `block_sumsq` is a third reduction (F9) |
| `SamplingKind` and custom kinds | needs extension | Six operations preserved by adapters; `gn_product` and `want`/`have` widen an open surface (F11) |
| `LocalModel` and `Problem` | conflicts | Sweep and residual are pytree leaves; a dict container breaks `partition_static` (F2); `hvp(p)` has no plan argument (F3) |
| Exact Hessian | needs extension | `tz` intermediates not compared array by array; the kept-state trade is recorded as deferred (F15) |
| Tied lift under sharing | not assessed | Proposal states it is unchecked; the geometry stays outside, which matches the library |
| `hessian='switch'` | needs extension | The Gauss-Newton view must keep a Gauss-Newton treedef; kept sets differ by phase |
| Newton timing | conflicts if objective-through-frame is adopted | `PhaseSeconds` fields `sweep`, `residual`, `trial_objective` are public (F21) |
| Rank continuation | just works | Between solves; one plan and one set of compiles per level |
| Sharding | conflicts | Executor enum against `shard_map` around the step (F3) |
| Compile-count tests | needs extension | Hold if the model treedef is stable and keys are structural (F6) |
| Bitwise tests | needs extension | Structural ones survive; a battery can gate steps 2 and 3 (F18, F19) |
| Ambient (CP) transposes | just works | No sweep; outside the framework |
| Safe mode | needs extension | New precondition needs a catalog change (F7); validity of `have` undesigned |
| Chunk-size estimators | needs extension | Slope measurement carries over; closed-form floors depend on the kept set |
| Eager jax | not assessed | Depends on memoization of generated bodies (F6) |

## What holds up

- All fifteen scripts in `run_all.sh` run at `3dc15d9`; with floating-point literals masked, the output is identical to `expected_output_3dc15d9.txt`.
- `__all__` counts: 166 across the six sampling modules, per module 34/10/10/57/25/30, and the 46/48/72 split.
- Every current name in the correspondence table exists in the four modules.
- What is kept and recomputed today: one `kind.precompute` per model (`_local_model_phases`), full sweeps for probe and lean for apply and entries, the residual through `point_forward`, and `nu`, `deta_tilde`, `tau_tilde`, `sigma_tilde` recomputed on every exact product.
- Chunking: only the probe-derivative transpose and its two-hole, effective on uniform with jax; `DEFAULT_CHUNK_SIZE = 100`; `_chunk_slope` reads XLA's memory analysis.
- The contraction split is numpy against jax.
- `T3Frame.to_t3` reads `(D, R)` and the geometry reads `(D, L)`.
- Sharing lives in the geometry; the sampling kernels and kinds contain no sharing code.
- Library-built frames are consistent, including the zero-point frame (both carriers zero) and `fv_padded_frame` (3e-16).
- The entries transposes build one-hot vectors.
- The exact-equality counts 37, 17 and 9 for the three files named.

## Questions for the maintainer

1. Must graded messages be fixed-slot tuples in the backend, or is a keyed container acceptable there?
2. With CG compiled whole, where should the plan ride: a static argument, or a value-hashed field as `chunk_size` is today? Does the `shard_map`-around-the-step design in `dev/sharded_fitting_plan.md` still stand?
3. Should `chunk_size` remain a field of the kind once a plan exists?
4. Do `sigma`, `tau` and `*_tilde` stay the canonical public names (the glossary's rule), with the graded names internal?
5. Should the E22 rows of the docs pass wait for one decision on side-and-direction wording?
6. Is `(D, L)` the authoritative base point, and is consistency then a caveat?
7. Would a kind-level sample-block wrapper with the policy interface be wanted ahead of the kernel refactor?
8. Are frames with tails above 1 (`squash_tails=False`) a configuration to keep supported?
9. Should the Taylor-or-derivative decision be taken before the toolbox reference paper is written?
