# Graded-messages refactor of the T3Toolbox sampling layer: working folder

> **Moved into the repository on 2026-11-15.** This folder was a standalone handoff package until then; the
> maintainer moved the work here once T3Toolbox 2026.5.0 was released: "let's move our development there. We can
> put the current handoff content in a folder in dev/, including prototyping. When we change the library code
> itself, we can do it on a new branch, then merge it in when it is done." This folder is now the working copy:
> documents, decisions and prototypes. Nothing in it is library code, and nothing in the library has been changed
> yet. **Where it lives:** `main` has the snapshot of 2026-11-15 (18:04). From then on the work, this folder
> included, continues on the branch `graded-messages`, because the refactor is major and severely breaking; it
> is merged when it is done. The scripts read the library and default to this repository
> (`cd scripts && ./run_all.sh && ./review/run_review_checks.sh`). Where the text below says "package", read
> "folder".

Start here. This package holds a design proposal for refactoring the sampling layer of
[T3Toolbox](https://github.com/NickAlger/T3Toolbox), the evidence behind it, and what is needed to resume the
work later. It was produced in a design conversation on 2026-11-13/14, in two parts: the refactor itself
(`PROPOSAL.md`), and a framework for performance choices built on it: how a call is executed
(`EXECUTION_PLANS.md`) and what it computes, keeps and hands back (`EVALUATOR.md`). Both were then reviewed by two
reviewer agents, and the maintainer went through their reports on the evening of 2026-11-14. The design as it
stands after that review is in `DESIGN_AFTER_REVIEW.md`, which changes a good deal of the two performance
documents and some of the proposal.

## Status

- **Decided by the maintainer (Nick Alger):** this is the right approach and the library should eventually be
  refactored into this form.
- **Not started, on purpose.** Other features are being added to T3Toolbox, and this refactor would interrupt
  that work. Do not begin until the maintainer says so.
- **Reviewed on 2026-11-14, and paused.** Two reviewer reports (`reviews/`) were walked through with the
  maintainer, who took ten groups of decisions (`REVIEW_DECISIONS.md`). He then stopped the session: "I feel like
  we are cutting corners here for expediency. Let's slow things down and plan everything right." The decisions
  stand as recorded; the rest of the planning is to be done slowly. `DESIGN_AFTER_REVIEW.md` section 10 lists what
  most needs a second look, what was not discussed, and what is undecided.
- **Redirected on 2026-11-15.** The maintainer found the cause of the pause: "we are relying on preserving old
  behavior as a crutch, instead of doing the hard work of deciding what stays and what goes from first principles
  reasoning", and "we are focusing too much on the sequencing of phases of the refactor ... before we have fully
  decided on the target". This is a breaking change and that is accepted. Compatibility is not a reason for any
  design choice, and the target is decided before any staging. `TARGET_DECISIONS.md` is the working agenda: every
  decision of the target in dependency order, with the earlier decisions marked as decided, to confirm, or
  reopened. First results: Taylor coefficients replace derivatives everywhere, with derivative mode as one adapter
  at the outer level of fitting; order 0 is the one-coefficient case of the same functions, with the order axis
  an optional axis `T` that is absent at order 0; input jets are always a point and one direction; and the plain
  and jetted sampling functions are one function, `probe(ww, pp=None, order=0)`, with three fitting kinds that
  take an `order` in place of six (`REVIEW_DECISIONS.md` entries 12 to 56).
- **Paused on 2026-11-15 at 05:53, to resume the next day.** Decided that morning: one function for the plain and
  the jetted computation (6.6), three kinds with an `order` (2.8), how the direction `pp` is passed (part of 6.2),
  and the convention for arrows on the edges (9.5). Held: what a tilde jet is (2.4); the material for it is
  `TYPE_CATALOG.md`. Parked for another arc: weights on internal edges as inner products. The section "Where this
  stands" at the top of `TARGET_DECISIONS.md` has the full state and the nearest open questions. The maintainer on
  the session: going slowly "has really clarified things", and it was good "to see everything come together in a
  coherent mathematical framework."
- **Resumed on 2026-11-15 at 15:34.** The 35 rows then remaining were put into seven groups (the section "What
  remains, in groups" of `TARGET_DECISIONS.md`), and group A, jets and the misfit, was begun. Decided: what a
  tilde jet is (2.4); the misfit, its default weight of ones and the weight's shape (2.5); standalone preparation
  functions with opt-in keywords (6.7). `T3POLYNOMIAL_SCALING_SURVEY.md` is the consumer's record behind 2.5.
- **The earlier documents are not rewritten.** `EXECUTION_PLANS.md`, `EVALUATOR.md` and `scripts/plan_mockup.py`
  describe the design before the review, and `PROPOSAL.md`, `API_IMPACT.md` and `WORKED_EXAMPLE.md` are affected in
  places. Each has a status note at its top saying what no longer holds. Where they disagree with
  `DESIGN_AFTER_REVIEW.md`, that document wins.
- **A new requirement governs the end state.** Straight after the refactor the maintainer will profile the
  downstream consumer T3Polynomial and tune the new options. A consumer must be able to use every addition
  without rewriting the optimizers and without changing T3Toolbox.
- **Nothing in T3Toolbox or T3Polynomial was modified.** Everything here is documents and standalone scripts.
- **The algebra is verified numerically on small problems; performance is not.** See "What is established" below.

## Contents

| File | What it is |
| --- | --- |
| `README.md` | This file: status, how to resume, context that is not in the proposal |
| `TYPE_CATALOG.md` | (Also the "Type catalog" tab of the Claude doc.) The types of all the sampling variables (jet or dual jet, vector or covector), where the dual pairings and the inner products are, and where the Riesz maps are. Claude's analysis for target decision 2.4; not a decision |
| `T3POLYNOMIAL_SCALING_SURVEY.md` | (Also the "Scaling survey" tab of the Claude doc.) How T3Polynomial scales, weights and normalizes today, the history of what was tried, and observations for target decision 2.5. Evidence, not a decision. T3Polynomial is private for now; the maintainer allowed this account of it here (2026-11-15) |
| `TARGET_DECISIONS.md` | **The working agenda since 2026-11-15**: the decisions that define the target, in dependency order, as questions with a status each. No answers proposed, nothing about sequencing |
| `DESIGN_AFTER_REVIEW.md` | The design as it stood on the evening of 2026-11-14, by topic: names, the explicit functions, `sampling_variables`, the `SamplePlan`, staging and wiring, the consumer requirement, evidence, open items, and the outcome of every review finding |
| `REVIEW_DECISIONS.md` | The same decisions in the order they were taken on 2026-11-14, with the evidence for each |
| `reviews/review_1_design.md` | Reviewer 1: internal inconsistencies, problems and simplifications (findings D1 to D23) |
| `reviews/review_2_library_fit.md` | Reviewer 2: how the proposal fits T3Toolbox (findings F1 to F22, with a coverage table) |
| `reviews/review_1_scratch/` | Reviewer 1's scratch scripts, cited as evidence in its findings. Not run by any runner |
| `PROPOSAL.md` | The design: network, naming, building blocks, frame, Taylor jets, code layers, name correspondence, consistency, performance, evidence, risks, order of work |
| `API_IMPACT.md` | What changes or breaks in the public API: with adapters, and with redundant things removed |
| `WORKED_EXAMPLE.md` | One fit step (objective, gradient, Gauss-Newton Hessian product) written with the building blocks |
| `EXECUTION_PLANS.md` | **Before the review.** How a call is executed: the immutable `Plan` (blocking, scan or vectorize, contraction order, executor), keeping against blocking, and the policies that choose a plan |
| `EVALUATOR.md` | **Before the review.** What a call computes: components named by letter and grade, the dependency graph of every case (entries, apply, probe; value, gradient, Gauss-Newton, exact; jets), `want` and `have`, what can be kept, and the mockup |
| `network_diagram.png` | The sampled T3 network with bond numbering and the messages around one site |
| `dependency_pattern.png` | The two factors of every dependency graph: the site graph of letters and the lattice of grades |
| `dependency_graph.png` | The dependency graph of the per-sample components, grouped by grade and coloured by what each changes with |
| `dependency_graph.dot` | The mockup's graph for the probe kind with every edge, in Graphviz DOT, written by `scripts/plan_mockup.py --dot`. It has no exact term, unlike the picture |
| `scripts/graded_messages_prototype.py` | The generated-terms layer (layer 3), unoptimized, checked against 34 library outputs |
| `scripts/probe_fit_step_example.py` | The explicit term-list layer (layer 2), runnable, checked against the library |
| `scripts/correspondence_check.py` | Current edge-variable jets against proposed ones, array by array |
| `scripts/pairing_check.py` | The pairing identity on every edge |
| `scripts/f32_experiment.py` | float32 accuracy of derivative-normalized against Taylor-normalized jets |
| `scripts/extra_core_experiment.py` | What the frame's center cores (`L[d-1]`, `R[0]`) actually touch |
| `scripts/index_check.py`, `scripts/index_check2.py` | Which bond each current bond variable lives on |
| `scripts/frame_consistency_check.py` | Library frames are consistent; a sample-based check; the discarded final carry |
| `scripts/frame_consistency_nondense_check.py` | A non-dense deterministic consistency check |
| `scripts/plan_mockup.py` | **Before the review.** The mockup of the interface as it was decided then: a `Plan`, a declared dependency graph, and one entry point `evaluate(want=..., have=...)`. Objective, gradient and Gauss-Newton product under five combinations of plan and keep set, checked against the library, with measured memory |
| `scripts/gn_fusion_check.py` | The algebraic fusion of the Gauss-Newton product for probes (no ambient-sized intermediate) |
| `scripts/component_graphs.py` | The dependency graph of every case, derived by tracing the graded program of the prototype: which components each case needs, what each reads, and through which cores |
| `scripts/sharing_check.py` | How the evaluator meets the library's sharing (one Tucker factor at several modes): it runs unchanged, and sharing is a core-sized step after the sample loop |
| `scripts/reuse_check.py` | Which edge variables can be reused between operations: the gauge of each sweep, the bond gauge matrices, and the probe output read off the frame's components |
| `scripts/run_all.sh` | Runs every correctness check against a T3Toolbox checkout. Needs only numpy |
| `scripts/expected_output_3dc15d9.txt` | Recorded output of `run_all.sh` at commit `3dc15d9` (all fifteen scripts) |
| `scripts/expected_output_89ac203f.txt` | Recorded output of the first ten scripts at commit `89ac203f`. Identical to the first ten sections of the newer record |
| `scripts/order_numpy.py`, `scripts/eta_axes.py`, `scripts/sweep_blocks.py` | Timing experiments: association order in numpy; scan or vectorize over orders and sites; a sweep in sample blocks. The last two need jax |
| `scripts/parked/` | A parked idea with its prototype and check: `xi` once per group when a sharing group's modes also receive the same sample vector. Not run by `run_all.sh`; see its own `README.md` |
| `scripts/run_timings.sh` | Runs the three timing experiments. Indicative only |
| `scripts/review/jet_steps_check.py` | **The explicit form with the decided names**: probe derivatives (jets) with residual weights, against the library's own `ProbeDerivativesKind` at orders 0, 2 and 4 |
| `scripts/review/unbundled_examples.py` | The explicit form at order 0 (earlier names): one function per variable, explicit arguments |
| `scripts/review/kind_readout_check.py` | Apply and entries on the probe kind's step functions |
| `scripts/review/tilde_adjoint_check.py` | In what sense the tilde variables are adjoints: multipliers of their edge partners, and variations of the variable named |
| `scripts/review/slot_tuple_examples.py`, `scripts/review/slot_tuple_check.py` | The fixed-slot tuple form that was prototyped and declined; kept as a record. The second needs jax |
| `scripts/review/pad_reshape_check.py` | Padding and reshaping the samples, with the library's kernels |
| `scripts/review/levels_sketch.py` | Nested scans over axes of the sample stack, on the mockup |
| `scripts/review/levels_jax_check.py` | The same on the library's kernels under jax: jit, nesting, the uniform layout, jets, masks, `shard_map`, a compiled CG loop. Needs jax |
| `scripts/review/order0_assessment.py` | The assessment for target decision 2.7: order 0 as the one-coefficient case of the same jet functions, with an order group that is empty at order 0. Needs jax |
| `scripts/review/merged_frontend_prototype.py` | A prototype of the signature for target decision 6.6: one frontend `probe` for the plain and the jetted computation, with the optional order axis `T` |
| `scripts/review/tilde_jet_pairing_check.py` | For target decision 2.4: a tilde jet is a jet, paired with a jet by the top coefficient of their product. Checks the invariance of that pairing, that it makes each product its own transpose, and that the flip alone fails in derivative coordinates. numpy only |
| `scripts/review/type_catalog_check.py` | Checks `TYPE_CATALOG.md`: every variable transforms as its catalogued type under arbitrary changes of basis of the network spaces and under `s -> c s`; the one inner product is at the seed. numpy only |
| `scripts/review/arrow_type_check.py` | The maintainer's arrow picture as a type checker: each leg carries its edge and the direction of the edge's arrow, and every contraction and sum of the explicit program is checked. Also shows what is rejected. numpy only |
| `scripts/review/gauge_arrow_check.py` | Checks on the library's own orthogonalization methods that the arrows of the frame cores are their orthogonality conditions: `M M^T = I` along the arrows |
| `scripts/review/tilde_multiplier_check.py` | With jets: each tilde variable is the multiplier of the dot of its partner, under the top-coefficient pairing; the slot-by-slot pairing is wrong. numpy only |
| `scripts/review/bad_scaling_example.py` | Prototype of the example required by 6.7: badly scaled data, a bad fit as given, a good fit after the standalone preparation functions. Two cases, few samples and many. Uses the library's Newton solver; add `--numbers` to print each error and time |
| `scripts/review/adapter_doctest_draft.py` | Draft of the doctest required by 2.3: what the derivative-mode keyword does to the data and the weight, in three cases. The three conversions are prototyped so that it runs. numpy only |
| `scripts/review/order_scan_check.py` | For 2.9: the product of two full jets in three forms: a loop with slices, the library's present scheme in Taylor form (a scan with the other jet carried and shifted one slot per step), and a scan over a table of the needed pairs of orders. For each: correctness against the explicit sum, the size of the traced program, and a count of multiplications that separates the expensive contraction from the cheap one. The two forms of the target are written in the library's pattern (`get_backend`, `tree_contains_jax`, `xscan`, `contract`) and offered as one function with a setting (`product_of_full_jets`); the setting is checked on the four backends (numpy or jax, ragged or uniform) in float64 and float32. Needs jax and the path of a T3Toolbox checkout |
| `scripts/review/subtree_gauge_check.py` | For the naming of the variables (type catalog, section 14): an edge variable is the contraction of the subnetwork on one side of an edge, in a basis. Checks that the frame's dot variables and the base point's dots along the lifted direction are one tangent vector, and that the lift's factors join the twins in the two bases. numpy only; needs the path of a T3Toolbox checkout |
| `scripts/review/run_review_checks.sh` | Runs the twenty review scripts. The four jax ones are skipped without jax |
| `scripts/review/expected_review_output_3dc15d9.txt` | Recorded output of `run_review_checks.sh` at commit `3dc15d9`, with jax 0.12 |
| `scripts/timings_2core_cpu_3dc15d9.txt` | Recorded output of `run_timings.sh` on a 2-core CPU. Not a basis for defaults |

The proposal also exists as a Claude doc with nine tabs: the main proposal, "Target decisions" (the content of
`TARGET_DECISIONS.md`), "Type catalog" (`TYPE_CATALOG.md`), "Variable diagrams" (a prototype with no file of its own; its drawings are generated by `scripts/variable_diagrams_gen.py`), "Scaling survey" (`T3POLYNOMIAL_SCALING_SURVEY.md`), "Design after review" (`DESIGN_AFTER_REVIEW.md`), API
impact, Worked example, Execution plans and Evaluator, the older tabs with the same status notes as the files
here: <https://claude.ai/artifact/PvTCvDZBKaPKqRTnntH42s>. The decisions log and the
review reports are only in this package. The markdown files here are the durable copy.

## Pinned state

- **T3Toolbox commit for the proposal, API survey and worked example:** `89ac203f512d213ec45da89eaaf2d9d5e0e3ec25`
  (2026-11-13). All names, line counts and measurements in those three documents refer to this commit.
- **T3Toolbox commit for the execution plans:** `3dc15d9646b17c089829f20b183e1e69d7972017` (2026-11-14). The six
  commits in between added Newton timing records and `hessian='switch'` and did not touch `entries.py`, `apply.py`,
  `probing.py`, `sampling_derivatives.py` or `utv_sampling.py`. All ten original checks give identical output at both.
- **T3Toolbox 2026.5.0, verified on 2026-11-15 (17:10).** The release (tag `v2026.5.0` on `4463231f`; the branch
  head read was `309e637`, 17 commits after `3dc15d9`) changes the optimizers (the Gauss-Newton to exact switch,
  phase timing), shape comments and documents. Both check suites give the recorded output against it, apart from
  the one known last-digit difference. The recorded-output files keep `3dc15d9` in their names.
  `scripts/review/bad_scaling_example.py` was written and recorded against 2026.5.0. The library's own
  `dev/HANDOFF.md` at that head names this refactor as its next arc.
- **Environment the checks ran in:** Python 3.13.18, numpy 2.5.3, 2-core CPU. No jax for the first ten checks;
  jax 0.12.0.1 was installed for the execution-plan work (the correctness checks do not need it).
- **The review of 2026-11-14** used the same T3Toolbox commit, `3dc15d9`, and the same environment. The scan
  experiments used jax 0.12.0.1 on four virtual CPU devices. T3Polynomial was read at its commit `dbda0de`
  (2026-11-14); the repository is private, and nothing in it was changed or copied here beyond the few lines
  quoted in `DESIGN_AFTER_REVIEW.md` section 7.
- **Read during the review, in part:** `backend/optimizers.py`, `backend/optimizer_display.py`,
  `backend/fitting.py`, `backend/common.py`, `t3toolbox/optimizers.py`, `tests/test_contractions_interpreter.py`,
  `docs/glossary.md`, `docs/naming_conventions.md`, `docs/contributor/precompute_and_caching.md`,
  `docs/contributor/curvature_internals.md`, `docs/contributor/deferred_and_rejected.md`, `dev/HANDOFF.md`,
  `dev/OPEN_QUESTION_extension_surface.md`, `CHANGELOG.md`. The reviewers read more; see their reports.
- **Files read in full:** `backend/entries.py`, `backend/apply.py`. **Read nearly in full, docstrings trimmed:**
  `backend/probing.py`. **Read in large part:**
  `backend/sampling_derivatives.py` (all kernel families; the chunk-size estimators and some drivers only skimmed).
  **Read in part:** `backend/utv_sampling.py`, `backend/fitting.py`, `backend/contractions.py`,
  `frame_variations_format.py`, `backend/geometry.py`, `docs/frame_variations.md`,
  `docs/contributor/scan_body_principles.md`. **Not read:** `backend/ut3_sampling.py` beyond its function list,
  `backend/padded_frames.py`, `backend/uniform_padded_frames.py`.

## How to resume

1. **Re-run the checks against the current library before anything else.**
   ```
   cd scripts && ./run_all.sh /path/to/T3Toolbox
   cd review && ./run_review_checks.sh /path/to/T3Toolbox
   ```
   Compare with `expected_output_3dc15d9.txt` and `review/expected_review_output_3dc15d9.txt`. The scripts call
   backend functions by their names at the pinned commits. If the library has moved, some will fail on imports or
   signatures. Fix the scripts first; a failure there is drift, not a flaw in the design. To reproduce the
   recorded numbers exactly, check out the pinned commit. Do not run `run_timings.sh` as a check; it is an
   experiment, and its recorded numbers are from a small machine.
2. **Read** `TARGET_DECISIONS.md` first (its section "Where this stands" is the state at the last pause), then
   `TYPE_CATALOG.md`, `DESIGN_AFTER_REVIEW.md` and `REVIEW_DECISIONS.md` (its entries 12 and 13 are the
   redirection, 19 to 29 the morning of 2026-11-15). Then `PROPOSAL.md`, `WORKED_EXAMPLE.md` and
   `API_IMPACT.md`, minding the status note at the top of each. Read `EVALUATOR.md` and `EXECUTION_PLANS.md` last
   and as history: their evidence and measurements still stand, their interface does not.
3. **Read the library's own records before proposing anything.** The reviewers found that several ideas here had
   already been measured, decided or named there. At least: `docs/glossary.md`, `docs/naming_conventions.md`,
   `docs/contributor/precompute_and_caching.md`, `docs/contributor/curvature_internals.md`,
   `docs/contributor/deferred_and_rejected.md`, `docs/contributor/parameters_not_closures.md`,
   `docs/contributor/scan_body_principles.md`, `dev/HANDOFF.md` and `dev/OPEN_QUESTION_extension_surface.md`.
4. **Settle the target with the maintainer, slowly, by going through `TARGET_DECISIONS.md` in order.** Reason from
   first principles; compatibility with today's behaviour is not a criterion, and nothing about staging is
   decided until the target is. The earlier list of open items is `DESIGN_AFTER_REVIEW.md` section 10: first the decisions
   that were taken fastest, then finding F17, then the undecided points, then an acceptance list for the consumer
   requirement. The maintainer asked for this pace explicitly.
5. **Then rewrite** `EXECUTION_PLANS.md`, `EVALUATOR.md`, the affected parts of the other three documents, and
   `scripts/plan_mockup.py` to the settled design. `scripts/review/jet_steps_check.py` is the starting point for
   the explicit functions and the names.
6. **Re-survey the API** if much time has passed. The counts in `API_IMPACT.md` are from text searches at the
   pinned commit.
7. **Only then follow the order of work** at the end of `PROPOSAL.md`, revised for the two stages of
   `DESIGN_AFTER_REVIEW.md` section 6.
8. **Follow the repository's own rules.** Read `CLAUDE.md` and `dev/README.md` in the T3Toolbox repo. Plans and
   handoffs live in `dev/`; the repo asks for incremental, reviewable slices; scan and map bodies must follow
   `docs/contributor/scan_body_principles.md`; signatures follow `docs/contributor/signature_style.md`.

A prompt to hand to Claude Code, from the root of a T3Toolbox checkout with this folder available:

> Read `README.md`, `DESIGN_AFTER_REVIEW.md` and `REVIEW_DECISIONS.md` in the graded-messages handoff folder, then
> its other documents, minding the status note at the top of each. Read `CLAUDE.md` and `dev/README.md` in this
> repository, and the library records listed in step 3 of the handoff's "How to resume". Then run
> `scripts/run_all.sh` and `scripts/review/run_review_checks.sh` against this checkout and report which checks
> still pass and which have drifted since commit `3dc15d9`. Do not change library code. The review was paused so
> that the remaining planning could be done carefully: list the open items of `DESIGN_AFTER_REVIEW.md` section 10
> for my decision, one at a time, and do not treat the decisions flagged there for a second look as final.

A natural home in the repository is a folder such as `dev/graded_messages/`; that is the maintainer's call.

## What is established

Verified numerically against the library at the pinned commit (details and scripts in the Evidence section of
`PROPOSAL.md`):

- Every sampling kernel is one of seven site rules at one of four grades. The prototype reproduces 34 library
  outputs (value, Jacobian, tangent transpose, corewise Hessian term; entries, apply, probe; orders 0 and 3) to
  1.2e-15.
- The current edge variables map to the proposed ones by the two conversion rules F and R, array by array.
- Taylor coefficients with reversed residual jets make the adjoint recurrences identical to the forward ones.
- float32 rounding error is the same in both jet normalizations; the Taylor form has more overflow headroom.
- In the current code, left-swept bond arrays sit on bond `i` and right-swept ones on bond `i+1`.
- The frame's center cores (`L[d-1]`, `R[0]`) do not affect any tangent operation's value, but must be finite.
- Library-built frames are consistent, and a non-dense consistency check works to a floor of about 1e-12.
- One fit step written with the two drivers and explicit term lists matches the library to 7e-16.
- The same fit step driven by a `Plan` and `evaluate(want=..., have=...)` gives the library's objective, gradient,
  Gauss-Newton product and per-sample residual to 4e-15 under five combinations of plan and keep set. The gradient
  is unchanged under all 512 subsets of the nine keepable components handed in.
- The dependency graph of every case is the product of the site graph of letters and the lattice of grades: a
  component reads the letters upstream of it at every grade at or below its own. This was derived by tracing the
  graded program that reproduces the library, for entries, apply and probe, for the value, `J p`, `J^T r` and the
  exact term. Entries has the same graph as apply, and jets add an order axis but no components and no reads.
- The frame's `nu` and the base point's `nu` differ by a bond-sized gauge matrix (also at the residual grade), and
  the probe output can be read off the frame's components and the point's center cores, to 2e-15.
- For probes with an orthogonal frame and a gauged tangent, the Gauss-Newton product can be formed with no
  ambient-sized intermediate; it matches the through-ambient route to 5e-16.

Corrections to the list above, from the review:

- The rule "a component reads the letters upstream of it at every grade at or below its own" is an upper bound,
  not the exact rule, and the drawn graph of the exact term disagrees with the reuse table (finding D4).
- Reading the probe output off the frame through the center cores is a true identity, but the route is parked:
  the value is computed through the base point's own cores (`DESIGN_AFTER_REVIEW.md` section 4.4).
- Tail sizes above 1 do work: in the library's current kernels, and in the prototype after three small edits
  (`reviews/review_1_scratch/proto_tails.py`, 34 comparisons to 2.3e-15). That check is not yet in `run_all.sh`.

Established in the review (scripts in `scripts/review/`; details in `DESIGN_AFTER_REVIEW.md` section 9):

- With one function per variable and explicit arguments, probe derivatives with residual weights match the
  library's own `ProbeDerivativesKind` at orders 0, 2 and 4, for `z`, `f`, `g`, `Jp`, `JtJp`, `Kp` and the two
  sums, to 4e-16. Jets change only the seven term functions.
- Apply and entries use the probe kind's step functions; only the gather for entries is not shared.
- Each tilde variable is the multiplier of its edge partner's constraint, and the variation of the variable it is
  named after.
- Padded samples are inert for apply and probe by multilinearity; entries need a mask.
- Scans over axes of the sample stack work on the library's kernels under jit, nested, on the uniform layout,
  with jets, under `shard_map`, and inside a compiled CG loop. Scanning a sharded axis under plain jit gathers.

Not verified: speed and memory of any generic kernel; `K` and `C` stacks; kept `W`; the explicit form on the
uniform layout or under jit; the stage-1 adapter; jax beyond the scan experiments; GPU; the `tz` intermediates
array by array. For execution plans: the `orders`, `sites` and
`executor` fields of `Plan` are declared in the mockup but not exercised; nothing was run on a GPU, sharded, or at
scale; the algebraic fusion is unchecked with weights, jets or a tangent stack.

## Requirements and preferences stated by the maintainer

- **No loss of performance.** The current kernels were derived by hand for efficiency, including memory at large
  sample counts. The refactor must match them.
- **Tail edges of size greater than 1 must work naturally**, so that uniformly shaped cores need no special cases.
  The review found this already holds in the current kernels, so the requirement is that the refactor must not
  regress it.
- **Simplicity, mathematical elegance and elimination of special cases** were the stated design criteria, with the
  instruction not to be biased by the existing code.
- **Edge-aligned indices were the original intent.** The maintainer meant bond-variable indices to match their
  edges. In the current code `nus[i]` sits on bond `i+1`.
- **The extra frame cores were kept deliberately** so that every family has `d` cores, for code reuse and
  off-by-one safety. All library-controlled functions produce consistent centers. Nothing guards against a user
  constructing an inconsistent frame.
- **Fit data is normally normalized before the fit**, so that different derivative orders are similar in scale.
  Any `t!` conversion must therefore compose with existing weights, not add a normalization.
- **The explicit term-list layer was judged "very clean."** It is the right first target.

Stated for the execution-plan work:

- **A framework for performance choices, not one locked approach.** The library must serve users with different
  problems and setups, not only the maintainer's examples.
- **Be very cautious about leaning on timings.** The machine and problem scale available in the design session do
  not reflect the large GPU problems the library targets.
- **Do not make performance decisions from small-scale tests that could limit scaling.** The eventual target is far
  larger problems, sharded across large numbers of GPUs.
- **Tolerance tests for numerical operations, bitwise tests for structural things.** Bitwise checks on numerical
  operations are probably drift. Rounding-level deviations from a refactor, such as reordered contractions, are
  allowed as long as the new algorithm stays numerically stable.
- **Keep it simple and avoid state as much as possible.** A design principle, not a hard rule.
- **"Fill the available memory" is not the preferred sizing rule**, but was to be thought through; see the Policies
  section of `EXECUTION_PLANS.md`.
- **A lean, not a decision: prefer larger outer blocks over more caching.** The maintainer sees arguments for both.
- **Decided: the plan is passed to every call.** Amended in the review: `LocalModel` carries the plan, with a
  default; `sampling_variables` still takes it explicitly.
- **Decided: the kept components are a separate argument** (`have`). Refined in the review: held variables are
  keyword arguments, one per variable.
- **Decided: per-sample residuals are returned only when the user asks.** Returning them must be supported. They can
  be very large, and sometimes only the objective is wanted (no gradient, no residual), so the default is not to
  return them. The maintainer's running theme: make such things user-controllable with an option.
- **Decided: it is a dependency graph and should be treated as one.** Each call says what it wants and what it has.
  `want` and `have` are call arguments and `Plan.keep` is dropped.
- **Decided: the residual option is folded into `want`** (wanting the component `r`); there is no separate
  `return_residual` flag.
- **Parked: `xi` once per group for shared sample vectors.** Shared sample vectors are a common, though not
  universal, use case. The maintainer judged exploiting them low priority and not worth it at this time; it may
  or may not be revisited. Recorded under "Parked ideas" in `EVALUATOR.md`.
- **The purpose is a framework that gives users control over performance options**, such as what to recompute,
  which they tune for their own problems. Stated more than once, together with the caution about small-scale
  CPU timings.

Stated in the review of 2026-11-14 (the decisions themselves are in `DESIGN_AFTER_REVIEW.md`):

- **The end state must serve a consumer.** Straight after the refactor the maintainer will profile T3Polynomial
  and tune the new options, so every addition must be usable without rewriting the optimizers or changing
  T3Toolbox.
- **Do not cut corners for expediency.** The maintainer paused the review for this reason. Plan everything right.
- **Code must be readable statically.** The backend takes and returns immutable objects made of basic types. A
  dict is out, and so is a tuple whose `None` slots have to be inspected: unbundle and make the caller supply
  each array as its own argument. "This is not for the compiler; it is for the reader of the code."
- **Names must not be generic.** `evaluate` and `compute_variables` were rejected for not saying which variables.
  The cross-cutting word is "sampling".
- **`Hp` must never mean Gauss-Newton.** Confusion between the two Hessians is the concern; the regularizer is not
  (regularization is usually handled by rank truncation).
- **Tuples of axis indices, not counts of leading axes,** for the scanned sample axes: the user of this feature
  can handle the generality, and such generality tends to pay off later.
- **`chunk_size` in the sampling kind was an uncomfortable compromise** (conceptually it is a solver parameter).
  It is to go as part of this proposal; when is not important.
- **Threading the plan through about 25 optimizer functions is too much.**
- **Help the user choose what to hold:** an exhaustive table of the ideal held variables for each wanted variable,
  in the docstring.
- **Reservation on record:** the 0/1 mask for padded entries "gives me pause"; accepted for now.

## Open questions for the maintainer

**After the review, the current list of open items is `DESIGN_AFTER_REVIEW.md` section 10.** Of the questions
below: 1 is decided (`_dot` and `_tilde`); 3 is decided (`(D, L)` is authoritative; `to_t3` is noted); 5 leans to
converting at the boundary, which is what the jet check does, but was not formally decided; 8 and 9 are replaced
by the new list. 2, 4, 6 and 7 are still open.


1. **Grade suffix.** `_z` uses the same letter as the probe output `z`. `_r` (residual) is an alternative.
2. **Bond storage.** True length `d+1` arrays, or `d` core inputs plus one terminal behind a bond view.
3. **Authoritative family for the base point.** `T3Frame.to_t3()` returns `(D, R)`; `base_point(frame)` in
   `backend/geometry.py` returns `(D, L)`. One should be chosen.
4. **Default boundary vector on the padded uniform path.** Ones (relies on masked cores) or the tail's rank mask.
5. **Public jet convention.** Keep derivative-normalized inputs and outputs and convert at the boundary, or expose
   Taylor coefficients.
6. **Adapters or removal.** `API_IMPACT.md` describes both. With adapters nothing public changes; removal deletes
   about 94 of 166 public backend sampling names.
7. **Which Hessian first.** The worked example covers the Gauss-Newton product. The exact Hessian needs the `tz`
   grade at the corewise frame plus the lift.
8. **Execution plans.** Six further questions are listed at the end of `EXECUTION_PLANS.md`: the priority for
   spending memory, whether a timed policy is wanted, which executors come first, reduction order at scale,
   measurements on target hardware, and the numpy contraction path.
9. **Evaluator.** Seven are listed at the end of `EVALUATOR.md`: generating the graph from the site rules instead
   of declaring it per case, alternative rules in the graph, names for copies of a component, how frontend
   objects pass `plan`, `want` and `have` down, validity of `have`, compilation per pair of `want` and `have`,
   and the `K` and `C` stacks.

## Things learned that are easy to get wrong

- **`is_consistent()` already exists** on `T3Frame` and `UT3Frame`. The ragged one is dense
  (`fv_operations.fv_frame_consistency_residual`); the uniform one already uses a difference norm. The proposal is
  to make the ragged one cheap and to wire it as a safe-mode precondition, not to add a new method.
- **A manifold frame has no plain probe output and no grade-`t` core gradient.** The prototype hit both: the
  plain value needs the point's own cores, and every rule call needs an explicit set of wanted grades.
- **In a manifold frame the grade decides the bond space.** `mu` and `mu_z` live on `rL`; `mu_t` lives on `rR`.
  Components must be keyed by grade, not stacked on an axis.
- **The sharding test finds subscript strings by scanning source literals** (`tests/test_contractions_sharding.py`,
  which asserts it finds more than 100). Generated strings would be invisible to it.
- **Some tests assert exact equality** (37 in `tests/backend/test_curvature.py`, 17 in `tests/test_fitting.py`,
  9 in `tests/test_uniform_probing.py`), including a bitwise containment of entries in apply at one-hot vectors.
  A port changes results at rounding level. Per the maintainer's testing policy these are a cleanup list: sort them
  into structural (keep exact) and numerical (move to a tolerance). Separately, the repository's recent commits use
  bitwise comparison to show that a new off-by-default option leaves existing behaviour unchanged; a port that
  reorders contractions cannot offer that, and the review should expect a tolerance argument instead.
- **The library already keeps the frame sweep.** `kind.precompute(frame, sample)` runs once per local model through
  the orthogonal frame, and Gauss-Newton products reuse it. What is recomputed today is the forward through the
  point's own cores (for the residual, and at trial points) and, on every exact-Hessian product, `nu` through `L`
  and the residual-grade state. So keeping the frame sweep and the residual is today's behaviour, not a new option.
- **Reuse depends on the gauge of the sweep.** Components swept through the point's own cores are not the frame's.
  Through the base point `(D, L)`, `xi` and `mu` are; `nu` differs by a bond-sized gauge matrix.
- **The contraction split is numpy against jax, not eager against jit.** jax always gets one einsum; numpy gets a
  fixed pairwise path computed from the subscript string alone, so it is the same at every size.
- **Timings on a small, busy machine do not rank close alternatives.** Two runs of the same scripts differed by up
  to about 70% in individual cells, and the fastest variant in the small cases changed between runs.
- **Apply and entries need far fewer components than probe.** They read the output at the right boundary, so the
  forward needs only `xi` and `mu`, and the scalar residual enters at the boundary, so the adjoint needs only
  `nu_z` and `eta_z`. For them `nu_z` is the residual times the plain `nu`.
- **"Sharing" is the library's word for one Tucker factor used at several modes** (`docs/sharing.md`). These
  documents use it only in that sense, and say "reuse" for a component used by more than one operation or frame.
  Sharing lives in the geometry wrapper, on cores; the evaluator works per sample and needs nothing new for it.
- **Declare the dependency graph; do not discover it.** An evaluator that computed components on demand held
  every one until the end of its block. With each component's reads declared as data, the evaluator knows what is
  needed before computing anything and drops each component at its last read.
- **In a host loop, a block's arrays must really die with the block.** The mockup's first demand-driven evaluator
  was a self-referencing closure, a reference cycle, so each block's arrays lived until the next garbage
  collection. Blocked peaks were about double and not affine in the block size, and a budget policy overshot by 6%.
  (An earlier draft blamed probe size; that was wrong.) Test that memory is affine in the block size and that
  nothing is held after a call.
- **Probe derivatives vanish above order `d-1`.** High-order tests need many modes.
- **Generated scan bodies must be memoized on a hashable specification**, or eager jax recompiles every call.
- **The carry marker of a sweep should be a named singleton** (an enum member), not `object()`, `None` or a string.
- **`dxis` means two things today**: `V` applied to `w` in the tangent code, and `B` applied to `p` in
  `t3_probe_derivatives`.

## Where things are in the current code (pinned commit)

| Thing | Location |
| --- | --- |
| Plain kernels (`compute_xi`, `compute_mu`, `compute_sigma`, the tilde family, the two-hole "tangent" family) | `t3toolbox/backend/probing.py` |
| The three-term tangent step used by all sweeps | `_sigma_step` in `probing.py` |
| Adjoint-state apply and entries transposes (`sigma_hat`) | `_apply_transpose_adjoint` in `backend/apply.py`; `compute_sigma_hat` in `probing.py` |
| Jet kernels, binomial shifts, order scans, chunked assemblies, chunk-size estimators | `t3toolbox/backend/sampling_derivatives.py` |
| The adjoint of the affine pushthrough | `_adj_prop` in `sampling_derivatives.py` |
| Consumers of the kernels | `SamplingKind` classes in `backend/fitting.py` and `backend/uniform_fitting.py` |
| Grouped einsum, and the fixed numpy path (`_pairwise_path`) | `backend/contractions.py` (`contract`) |
| Chunk-size default and estimators (`DEFAULT_CHUNK_SIZE`, `_chunk_slope`, `estimate_chunk_size`, `max_chunk_size_within`) | `backend/sampling_derivatives.py`; described in `docs/chunking.md` |
| The Gauss-Newton product as forward then transpose | `LocalModel.hvp` in `t3toolbox/backend/optimizers.py` is what the optimizer runs (also `gn_quadratic` there, and the frontend `gn_hessian` in `t3toolbox/fitting.py`) |
| Per-phase timing records (`PhaseSeconds`, `NewtonTiming`), at `3dc15d9` | `backend/optimizers.py` |
| Frame classes, `to_t3`, `is_consistent`, `is_orthogonal` | `t3toolbox/frame_variations_format.py`, `uniform_frame_variations_format.py` |
| The geometry's base point | `base_point` in `backend/geometry.py` |
| Frame construction and gauge projection used by the worked example | `fv_conversions.t3_orthogonal_representations`, `tv_operations.tv_orthogonal_gauge_projection` |
| The maintainer's rationale for `d` frame cores | `docs/frame_variations.md` |
