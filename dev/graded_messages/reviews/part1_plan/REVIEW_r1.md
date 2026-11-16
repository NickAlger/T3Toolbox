NOT YET

# Review of `PART1_PLAN.md`, draft r1

Reviewer: Claude, 2026-11-16. Branch `graded-messages` at `9f42cab`. The library and `tests/` are unchanged since
`4d38df7`, the commit the plan names, so its statements about the code are judged at that state.

## What I checked, and how

- **Read:** the brief; `CLAUDE.md`; `TARGET_DECISIONS.md` in full (the row tables, "The two parts" with its six
  small items, and every note, the 2026-11-16 amendments included); `SAMPLE_VARIABLES.md`; the README's status
  list and "Things learned"; the plan in full.
- **Rows checked against the plan:** 0.1 to 0.8; 1.1 to 1.3, with the geometry boundary; 2.1 to 2.9; 3.1 to 3.7;
  4.1 to 4.5; 5.1 to 5.6; 6.1 to 6.7; 7.1 to 7.8; 8.1 to 8.5, with the note "The oracle is a description and the
  data are an argument" and the three parts of 8.3; 9.1 to 9.7; small items 1 to 6. For 9.1 I checked only that
  the plan contradicts nothing in `SAMPLE_VARIABLES.md` section 10.
- **Counts recounted** with an `ast` reading of each `__all__` and each class body. All of the plan's counts are
  right:
  - the six sampling modules hold 166 names (`probing` 34, `apply` 10, `entries` 10, `ut3_sampling` 25,
    `utv_sampling` 30, `sampling_derivatives` 57). Of these 30 change, 7 stay and 129 go; 74 of the 129 are
    outside `sampling_derivatives`.
  - `backend/fitting.py` has 20 names (9 classes, 3 singletons, 4 factories, 4 `sumsq` helpers).
    `backend/uniform_fitting.py` has 12, of which 9 go.
  - Frontend sampling methods: 15 + 12 + 12 + 12 = 51 (`apply_masks` on the uniform T3 excluded), and 24 of them
    go.
  - 6 of the 14 examples call a `*_derivatives` name; 41 strings in `term_stacks_check.py` `STRINGS`; 22 step
    functions in `jet_steps_check.py`; 6 frontend model factories.
  - 8 T3Polynomial scripts call `least_squares_problem`.
  - The five old `LocalModel` fields, the build laps, the `optimizers.py` import pin and `safety.py`'s imports
    are all as stated.
- **Code read for claims:**
  - backend: `geometry.py` (`frame`, `base_point`, `retract`, the uniform corewise frame); `fv_conversions.py`
    and `ufv_conversions.py`; `fv_operations.fv_frame_consistency_residual`; `contractions.contract` (its
    handling of `len_<G>`); `optimizers.py` (`Problem`, `_local_model_phases`, `_prepare_jit_inputs`,
    `_objective_resolution`, the pytree registration); `common.ValueHashedFields` and `partition_static`;
    `uniform_fitting.py`; `optimizer_display.py`;
  - frontend: `fitting.py`, `optimizers.py`, `shared_geometry.py`, `__init__.py`;
  - tests: `test_import_graph.py`, `test_contractions_sharding.py`;
  - the CI workflow.
- **Searches:** every caller of the 15 changed ragged names; every user of the frontend `*_derivatives` methods
  in `t3toolbox/`, `tests/`, `docs/` and `examples/`; every user of `Problem`, the kinds and the factories in
  `tests/`; every doc page that names old kernels or the old fitting API, with the doctests among them; the
  backend import reachability for P4.
- **Consumer:** `t3polynomial/fitting.py` (`_newton_fit`, `fit_rank_continuation*`, `_level_newton_kwargs`);
  `scripts/b01_backend_bench.py`; the T3Toolbox imports of the package; `dev/t3toolbox-upstream-notes.md`
  (searched only). I ran none of its code.
- **Ran:** `scripts/review/term_stacks_check.py`. It passes: 2760 calls, 41 strings.

## Findings

### 1. Blocking. The dry port cannot work as written (section 9, lines 912 to 938; section 10 item 3, line 948)

- **What is wrong.** The port is "a drop-in replacement for `_newton_fit`". It is to run "two levels of
  continuation", and then T3Polynomial's benchmark is to run "unchanged" on the new library. But
  `fit_rank_continuation` and `fit_rank_continuation_adaptive` call `X.probe_derivatives(...)` themselves, outside
  `_newton_fit`, to score each level. P12 removes that method.
- **Evidence.**
  - `t3polynomial/fitting.py:384-385` and `:597-598`: `pv_tr = X.probe_derivatives(ww_tr, pp_tr, jet_order)`.
  - `scripts/b01_backend_bench.py:136-142` drives `fit_rank_continuation`.
- **What would resolve it.** Say exactly what the script replaces besides `_newton_fit`, in a way that does not
  change T3Polynomial. For example, also patch the scoring calls with a derivative-form forward built from the
  merged `probe(ww, pp, order)` and `derivatives_from_taylor`. Then state that the benchmark runs with both
  patches installed.
- **Also to check.** The port must keep the `stats` keys the consumer reads: `newton_steps`, `newton_accepted`,
  `stop_reason`, `history`, `diagnostics` (`fitting.py:391-395`). O-17 would change `stats['timing']`.

### 2. Blocking. The suite is not green between P14 and P15 (P14 lines 687 to 719; P15 lines 721 to 725; table 5.4 line 743)

- **What is wrong.** P14 replaces the backend `LocalModel` and removes `Problem`. The frontend `fitting.py` stays
  on the old kinds until P15, but it depends on both.
- **Evidence.**
  - The frontend `LocalModel._blm` builds the OLD backend model:
    `bopt.LocalModel(self._bgeom, self.kind, self.sample, self.frame.data, self.sweep, self.residual, self._n_w, ...)`
    (`t3toolbox/fitting.py:339`).
  - `_with_curvature` calls `bopt._require_exact_capable(model.kind, geom)` (`:591`).
  - So every exact-mode frontend model breaks at P14, with `tests/test_fitting.py` and the frontend doctests.
  - P14 also does not list two tests that use the old API it removes:
    - `tests/test_jax_absent.py:71` (`bopt.least_squares_problem`) and `:80` (`probe_derivatives_model`);
    - `tests/test_aligned_padded_frame_frontend.py:152` (`topt.least_squares_problem`).
- **What would resolve it.** Do one of these:
  - keep the old backend model and `_require_exact_capable` under a private name until P15;
  - merge P15 into P14;
  - move the frontend model onto the new backend model before the switch.

  In every case, complete P14's test list.

### 3. Blocking. A sharding property that holds today is dropped without a word (section 7.6, lines 846 to 853; section 6, line 780; P18, line 959)

- **What is wrong.** `tests/test_contractions_sharding.py` has two classes. The plan accounts only for the AST
  sweep. The second class, `TestJetKernelsShardOverW` (lines 136 to 211), pins that the jetted transpose and the
  exact Hessian's two-hole, run under `shard_map` with `W` sharded, compile with zero all-gathers and match the
  unsharded result.
- **Evidence.** That class imports `t3toolbox.backend.sampling_derivatives` and calls
  `sd.tv_probe_derivatives_transpose` and `sd.t3_probe_corewise_derivatives_hessian`. P18 deletes that module. The
  plan's table says only "Extended in P1" for this file. So at P18 either the gate fails on an import, or the only
  end-to-end test of sharding over `W` is deleted ad hoc.
- **Why it matters.** The brief says Part 1 "must not break what holds today" on sharding. Today this is what
  holds, beyond the per-string sweep.
- **What would resolve it.** Port the property to `sampling_variables`: `g` and `Kp` at order 0 and order 3,
  ragged and uniform, `W` sharded, zero all-gathers, equal to the unsharded result. Do it in P10 or P11. Then say
  in P18 that the old class goes. The note in that class about the varying-axes check (`check_vma` and
  `check_rep` off) applies to the engine's scan carries too.

### 4. Blocking. A silent decision in the backend corewise Hessian (section 2.4, line 372; N16, line 446)

- **What is wrong.** The sketch is
  `t3_probe_corewise_hessian(core_directions, residual, ww, x, pp=None, order=0, weight=None)`. It differs from
  today in three ways:
  - it moves `core_directions` to the front;
  - it renames the seed `residual`;
  - it adds a `weight`.

  The transposes beside it take a residual and no weight.
- **Evidence.**
  - Today the order is `(ztildes, ww, core_pair, core_directions)`, and `ztildes` is documented as "residuals
    (omega^2 applied)", a seed (`backend/probing.py:1745-1755`). N16 justifies the sketches by "Keeps today's
    leading arguments", which this one does not.
  - 3.6: "The bare transpose takes a seed and no weight."
  - 2.4: the frontend transposes take a residual and apply the turn.
  - No row says whether the backend function behind `Kp` takes a residual and a weight, or a seed.
- **What would resolve it.** Give the corewise Hessian the same convention as the transposes and keep today's
  argument order. Or list the question as an open point.

### 5. Should fix. P12's inventory is incomplete, and `docs/chunking.md` fails CI from P12 (P12 lines 666 to 675; section 6, line 773; section 8, line 900; P18, line 959)

- **What is wrong.** P12 removes the frontend `*_derivatives` methods. These callers are not in its list:
  - tests: `tests/backend/test_optimizers.py`, `test_padded_start.py`, `test_uniform_fitting.py`,
    `test_zero_reference.py`; `tests/oracle_sweeps/o1_ragged.py`, `o1_uniform.py`, `sweep_models.py`;
    `tests/test_fitting.py`, `test_jax_absent.py`, `test_dispatch.py`, `test_chunk_size_estimator.py`;
    `tests/test_uniform_probing.py`, with 51 calls (it is listed under P11 only);
  - module doctests: `t3toolbox/optimizers.py:279` and `:805`; `t3toolbox/fitting.py:928-930`;
  - doc pages: `docs/chunking.md:54-56` and `:119`; `docs/numerical_contracts.md`;
    `docs/fitting_and_optimization.md`.
- **A wrong classification.** `tests/test_chunk_size_estimator.py` is classed as "test old kernels only", but at
  lines 288 to 302 it calls the frontend `newton_cg` with `'probe_derivatives'`, `chunk_size` and `probe_derivatives`.
- **The chunking page.** Section 8 and P18 remove `docs/chunking.md` in P17 or P18. Its doctests call
  `T3Tangent.probe_derivatives_transpose(..., chunk_size=...)`, so CI is red from P12 on.
- **What would resolve it.** Give P12 and P14 the full lists. Move the removal of `docs/chunking.md` and its
  toctree entry to P12, or cut its doctests there.

### 6. Should fix. The "one commit" P14 can be split (line 687: "no part can be green alone")

- **What is wrong.** Part 1 of P14 can be green on its own: `frame(x)` returning `(frame, centers)`. The old path
  adapts with one line in `_local_model_phases` (`frame, _ = geom.frame(x)`), plus the frontend's two calls.
- **What cannot move earlier.** The corewise frame becoming `(B, G)` cannot precede the switch, because the old
  kinds read a four-store frame.
- **The size of the rest.** The remaining switch is the largest commit of the arc:
  - about 15 test files;
  - 13 examples;
  - `docs/fitting_and_optimization.md`, with 115 doctest prompts;
  - `getting_started.rst`.
- **What would resolve it.** Split off the geometry change. Then either justify why the backend optimizers cannot
  land beside the old ones before the frontend switches, or split that too.

### 7. Should fix. The uniform tests are weaker than the library's rule, and the masks for `Kp` are not stated (P9 lines 635 to 641; P10 lines 643 to 651; section 7.2 lines 802 to 810; section 2.3 step 2)

- **The stack matrix.** The plan's matrix is `W` empty, `(3,)` or `(2, 2)`; `K` `()` or `(2,)`; `C` `()` or
  `(2,)`. The library's matrix (`docs/contributor/testing_strategy.md:92-99`, `_CONFIGS`) also has:
  - multi-axis `C=(2,3)` and `K=(2,3)`;
  - forced-larger padding;
  - a varying-rank-across-`C` stack.

  The same note (lines 88 to 90) asks for a mutation check that removes the mask and must turn the suite red.
- **The masks for the core directions.** The plan does not say which masks `core_directions` are masked with, or
  which masks `Kp` is re-masked with. CLAUDE.md records a past bug: "a uniform two-hole masks θ̂ with the FRAME's
  masks (never the sweep's or the kind's own, which is a superset at TT slack)".
- **What would resolve it.**
  - Use the full `_CONFIGS` matrix for the engine and the uniform twins.
  - Add the mutation check.
  - State that `(D_dot, L_dot)` and `Kp` take the frame's `D` and `L` masks.
  - Test it on a frame with TT slack.

### 8. Should fix. A claim about the consumer is not true of its code (section 1, lines 41 to 42)

- **What is wrong.** The plan says "T3Polynomial runs with chunking off today (`t3polynomial/fitting.py`, and
  `scripts/b01_backend_bench.py` passes `chunk_size=None`)". The second half is true. The first is not.
- **Evidence.**
  - With `newton_kwargs=None`, `_level_newton_kwargs` returns `{}` (`t3polynomial/fitting.py:104-118`), and
    `_newton_fit` passes no `chunk_size` (`:225`, `:236`).
  - The library default is `chunk_size='auto'` (`t3toolbox/optimizers.py:213`, `:351`), so uniform fits through
    the fitting module chunk today.
  - The consumer's upstream notes say the same (line 253).
  - The decisions repeat the error ("The cost").
- **What would resolve it.** Correct the sentence. Note for O-1 and O-2 that the gate compares chunking-off runs,
  which is not the fitting module's default path.

### 9. Should fix. A conflict inside "The two parts" is adopted without being raised (section 1, line 29; section 2.3 step 6; section 7.2)

- **What is wrong.** The row for 5.3 fixes map-type operations "at all sites at once on the uniform one", and calls
  that "as the library does today". That is true of the plain kernels. It is not true of the jet kernels: on the
  uniform layout they map one site at a time through `xmap`, which is `jax.lax.map`.
- **Evidence.** `backend/sampling_derivatives.py:504`, `1259`, `1440`, `1504`, `1620`, `2763`, `2792`, `2903`; the
  library table in the 5.2 notes ("a per-site map whose step has a product of two full jets ... `jax.lax.map`, one
  site at a time").
- **Why it matters.** Vectorizing those maps over the sites multiplies the jetted intermediates by `d` on the
  uniform layout. That is a second step back in memory, beside the chunking that "What Part 1 removes" names.
- **What would resolve it.** List it as an open point: does Part 1 vectorize the jetted maps, or keep `lax.map`
  as today? Name its memory cost in section 1.

### 10. Should fix. The table of names is incomplete and inconsistent (section 3, lines 423 to 454)

- **Modules missing.** The brief asks for "the modules things live in". These placements have no row:
  - `LeastSquaresOracle`, `LeastSquaresBatch` and `HeldVariables` in `backend/fitting.py`;
  - the `Oracle` protocol, `local_model` and `point_objective` in `backend/optimizers.py`;
  - `jet_contract` and `GroupLengths` in `backend/contractions.py`.
- **The converter grammar.** N4 cites "the converter grammar `a_to_b`", but N6 and N7 propose
  `taylor_from_derivatives` and `taylor_weight_from_derivative_weight`. The library's free converters are
  `a_to_b` (`t3_to_dense`, `tv_to_t3`, `fv_to_t3`; `docs/naming_conventions.md`, rows of the table and section
  "Converters encode the operand type").
- **The frontend keyword.** The frontend methods' new keyword in place of `sum_over_probes` (section 2.5) has no
  row; N3 names only the engine's.
- **History.** N12's "`evaluate` was rejected as generic for the engine" is history.
- **What would resolve it.** Add the rows. Make N4 and N6 to N8 follow one grammar, or say why they differ.

### 11. Should fix. Signature sketches are not in the house style (section 2.4, lines 289, 293 to 297, 360, 367 to 373; section 2.5)

- **What is wrong.** The brief asks for "a signature sketch in the house style for the new and changed ones". These
  are one-liners without the shape-comment column:
  - the changed backend functions;
  - `dual_jet_to_jet`;
  - the five preparation functions;
  - `point_objective`.

  The merged frontend methods and the frontend `newton_cg` have no sketch at all.
- **What would resolve it.** Write them one argument per line with the shape contracts
  (`docs/contributor/signature_style.md`), at least for `t3_probe`, `tv_probe_transpose`, the corewise Hessian,
  one frontend method and the frontend `newton_cg`.

### 12. Should fix. `dual_jet_to_jet` "raises where a weight is zero" decides something open (line 289)

- **What is wrong.** The 3.6 note records the zero-weight case as "noted by Claude, not answered". Raising on a
  data value is a numerical check. It cannot run under a trace, and the plan does not put it under safe mode.
- **What would resolve it.** List it as an open point: raise in safe mode, return `inf` or `nan`, or document it
  as the caller's business.

### 13. Should fix. The documentation inventory is incomplete, and two requirements of 9.7 and 9.5 are taken or missed silently (section 8, lines 871 to 908)

- **Pages that name old kernels or the old fitting API and are not in the rewrite list:**
  - `docs/reading_signatures.md`, a user page whose exemplar is `compute_mu`;
  - `docs/contributor/signature_style.md`, the same exemplar; CLAUDE.md also names `backend/probing.py` as the
    reference module of the signature style;
  - `docs/contributor/uniform_polymorphism.md`, whose loop table cites `compute_mu`, `compute_eta_jets` and others;
  - `docs/contributor/uniform_internals.md`, which says the kinds re-mask;
  - `docs/contributor/padded_frame_internals.md`, which cites `least_squares_problem` and `Problem.local_model`;
  - `docs/contributor/sharing_internals.md`, which cites `Problem.local_model`;
  - `docs/contributor/ambient_derivative_transpose_note.md`.
- **The package root.** `t3toolbox/__init__.py` re-exports the six factories that P15 removes. It is not in P15.
- **9.7.** "How the diagrams are drawn is open (suggested: generated from the declared graph ...)". The plan adopts
  the suggestion (line 882) without listing it, and does not say what the script outputs or when it runs.
- **9.5.** "What it asks of the documents is that each diagram shows its arrows." Not stated.
- **What would resolve it.** Complete the list. Add the open point on the diagrams. State the arrow rule for the
  diagrams.

### 14. Should fix. 8.4's stored record and the trust-region fallback are not carried out (section 2.4, line 383; P14, lines 703 to 704)

- **What is wrong.** 8.4 says nothing is reduced over modes or orders "in the display or in the stored record",
  and the restatement records "the stored record keeps the full arrays for training and validation at every
  iteration". The plan says only what the display reads. It says nothing of `stats['history']` or
  `stats['diagnostics']`, and has no test for them.
- **The fallback.** 8.4 also decides the floor `eps |f|` for an oracle without `f_resolution`. The plan does not
  state it or test it.
- **What would resolve it.** Say what the record stores, with a test. Add the fallback with a test on a user
  oracle that lacks `f_resolution`.

### 15. Should fix. After P11 the bridge compares the engine with itself (P5, lines 600 to 602; section 6, line 765)

- **What is wrong.** The bridge compares the new outputs with `t3_probe` and the other changed names. From P11 on
  those names call the engine, so those rows compare the engine with itself.
- **Is it what 9.3 means?** 9.3 calls the old library "a bridge used once during the port". The plan makes the
  bridge a CI test that runs from P5 to P17.
- **What would resolve it.** Bridge only against code that is not rerouted (`sampling_derivatives`, the old
  kinds), or freeze reference arrays at P5. State the reading of "used once", or ask.

### 16. Minor. P4 as written creates an import cycle, and changes a formula silently (P4, lines 576 to 584)

- **The cycle.** P4 computes the residual in `fv_operations` "as the norm of a T3 difference". If that uses
  `t3_linalg.t3_norm`, the backend graph gains a cycle:
  `fv_operations -> t3_linalg -> ranks -> sharing -> fv_operations` (`ranks.py:16`, `sharing.py:34`). That breaks
  `test_the_backend_graph_is_acyclic`. `t3_operations` and `t3_orthogonalization` do not reach `fv_operations`,
  and `geometry.point_norm_sq` already shows the route through them.
- **The formula.** Today the denominator is `max(1, ||right||)` (`fv_operations.py:241-253`). The plan writes
  `/ ||T(D, R)||`.
- **The test.** "Assert on the largest intermediate" is not something a test can read. Use a shape whose dense
  form cannot be allocated.

### 17. Minor. `jet_contract` cannot pass the whole record to `contract` (lines 269 to 280; section 2.1: `contract` "unchanged")

`contract` raises when a `len_<G>` names a group that is not in the string (`contractions.py:214-218`). Every
string lacks some of `S`, `K` or the order groups. State that `jet_contract` passes `contract` only the lengths of
the groups its string names.

### 18. Minor. The grep of P18 will match by design (lines 962 to 964)

`dev/graded_messages/`, `dev/HANDOFF.md` and `CHANGELOG.md` all contain the removed names. Exclude `dev/` and
`CHANGELOG.md` from the grep, not only `dev/archive/`.

### 19. Minor. "Nothing squashes" is ambiguous (section 4, lines 475 to 476)

The geometry's `frame` calls `t3_orthogonal_representations` with its default `squash_tails=True`
(`fv_conversions.py:135`, `geometry.py:293`), and P14 touches that function. 4.4's note leaves this default "not
spoken of". Say that the engine squashes nothing and that frame construction keeps its default.

### 20. Minor. Small statements to make exact

- **P7 (line 620).** "`nu` equals the manifold call's `nu_L` at the same point" holds only when the corewise point
  is the frame's `(D, L)`. The raw `x` differs from it by a gauge.
- **P13 (line 682).** The unbiasedness of a drawn `f` is tested statistically. On a tiny `n`, enumerate every
  subset and test the expectation exactly.
- **The weight check.** Section 7.5 (line 839) has the frontend validate the weight against `x`, while O-18 and
  section 2.3 step 1 put the check in the engine. Make them agree.
- **Section 5.1 (line 490).** "Nothing old changes" in P1 to P10, but P4 changes `fv_frame_consistency_residual`
  and `T3Frame.is_consistent`.
- **Section 2.2 (line 102).** "Two stores or four" does not tell the cases apart on the uniform layout: the uniform
  corewise point `(B_sc, G_sc, shape, masks)` has four elements, like a ragged manifold frame. State the rule as
  "layout first, then the count" (uniform: 4 or 6).
- **The draw.** 8.3 part 2 says that MC-SGD or Adam on a user's oracle without a draw raises. The plan does not
  state it.
- **Tilde variables (P2).** The reference module must return them turned (2.4); say so.
- **The minibatch scale.** `minibatch_scale` multiplies `r_block_sumsq` (line 344). That is a choice no row makes.
  The plan also does not say whether `r` and `Jp` on a drawn batch are scaled. This is a question.
- **The protocol's `held`.** The `Oracle` protocol types `held` as `HeldVariables`, a least-squares record. 8.3
  says "what goes into `held` is the oracle's business". Fold this into O-8.
- **P5.** P5 holds the graph, the forward, the tilde grade, held variables, the structural checks, safe mode, the
  bridge and the twin. Splitting it in two (forward; then tilde and `g`) would make each review smaller.
- **Paths (P0).** `scripts/run_all.sh` means `dev/graded_messages/scripts/run_all.sh`.

### On the open points

- O-1 to O-3, O-5 to O-11 and O-13 to O-17 are genuinely open and stated so that the maintainer can answer.
- O-4 is half decided: 4.1 and the 8.3 note decide that `frame(x)` returns the centers. Only the form, and the
  frontend `COREWISE` question, are open. Say so.
- O-12 is open, and is put well.
- O-18 is a placement question. The shape rule itself is decided (2.5).
- **Missing from the list:** finding 4 (the corewise Hessian's arguments), finding 9 (the jetted maps on the
  uniform layout), finding 12 (a zero weight), finding 13 (the diagrams), and the minibatch scale on
  `r_block_sumsq` (finding 20).

## What is sound (keep it)

- **Counts.** Every count in the plan is correct at this commit.
- **Scope.** The scope table and the Part-2 deferrals follow "The two parts" exactly:
  - no `plan` argument anywhere;
  - trial policy a only;
  - the scan form only;
  - CG and the stochastic step compiled, as today;
  - `take` traceable.

  Every seam for Part 2 is additive.
- **The engine's call.** The steps of a call are right: structure always; layout and call type from the argument;
  dtype absorbing with one cast; the safe-mode center check only when `Vc` is given; the closure from the declared
  graph; the seed `r = z - y`, then the weight, then the turn; the sums kept on `C` and on `K + C`; masks on entry
  and re-masks on exit.
- **The data of the fitting layer** follow 8.1 to 8.4 and the oracle note: the batch, the oracle with two static
  fields, `held`, the fields of `LocalModel`.
- **Order of the slices.** Additive first: `jet_contract` and the reference module before the engine, and the
  engine finished on both layouts before any surface switches.
- **The safety import.** Finding that the backend has never imported `safety`, and raising it as O-6 with a
  no-cycle argument, is correct.
- **Statics and leaves.** The table in section 7.7 matches how the library registers its pytrees today
  (`partition_static`, `ValueHashedFields`).
- **The razor.** A backend-only fit as a doctest is the right test.
- **The dry port's weight.** Broadcasting a per-slot weight to the exact shape is the right reading of 2.5.
- **The table of names** is marked as awaiting the maintainer and is not treated as decided elsewhere.
- **Form.** The plan is written plainly and carries almost no history.
