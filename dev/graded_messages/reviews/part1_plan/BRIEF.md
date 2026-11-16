# Brief for the implementation plan of Part 1

Written by Claude on 2026-11-16 for two agents: a writer, who writes `dev/graded_messages/PART1_PLAN.md`, and a
reviewer, who reviews it. They iterate until the reviewer agrees. The maintainer then reads the plan.

## What is asked for

The maintainer's words (2026-11-16, 16:45): "a focused implementation plan document for part 1, which includes
not just what we want to do, but also how we will do it. The plan will including more details about what will be
done, and less details about the historical path we took to get to decisions and dead ends we tried."

So the plan says **what will be built and how**, for someone who implements it slice by slice. It does not retell
how the decisions were reached, and it does not list alternatives that were not taken. Where a reader needs the
reason for a rule, one sentence and the row number are enough.

## Where things are

- The repository is `/home/claude/t3toolbox`, on the branch `graded-messages`. Its `CLAUDE.md` is the project
  guide and its rules apply to the plan (the razor between backend and frontend, the signature style, scan bodies,
  the dispatch rules, the tests, the doctests, "no auto-formatter").
- The design work is in `dev/graded_messages/`. Read, in this order:
  1. `README.md`, the status list at the top.
  2. `TARGET_DECISIONS.md`: the section "The two parts" (the scope, and the six small items settled after it),
     then the row tables of sections 0 to 9. **The rows are authoritative.** The notes under each table give the
     detail of a row; where notes of different dates disagree, the latest wins, and a note that says a row was
     amended or reworded tells you which text stands. The file is long; read the notes of a row when you plan it.
  3. `SAMPLE_VARIABLES.md`: the reference for every variable's name, definition, recurrence and shape.
  4. `TYPE_CATALOG.md`, for the types and arrows.
  5. The prototypes in `scripts/review/`, above all `term_stacks_check.py` (`jet_contract` and the literal
     strings of 5.4), `jet_steps_check.py` (the step functions, by grade), `sample_variables_check.py` (the
     definitions as dense contractions), `order_scan_check.py` (the scan over orders written through `xscan`),
     `adapter_doctest_draft.py` (the derivative-data functions), `bad_scaling_example.py`. They establish the
     algebra and the names. They are **not** the form of the code: the library's own modules and contributor
     notes are.
  6. `API_IMPACT.md`, a survey of the public names the refactor touches. Its counts are from text searches at an
     earlier commit; recount what you rely on.
  7. `DESIGN_AFTER_REVIEW.md` and the other documents are older and carry status notes. Use them for material,
     never against a row of `TARGET_DECISIONS.md`.
- The library's own records, which the plan must follow: `docs/contributor/refactoring_methodology.md`,
  `testing_strategy.md`, `signature_style.md`, `scan_body_principles.md`, `uniform_polymorphism.md`,
  `uniform_internals.md`, `uniform_pytree_composition.md`, `precompute_and_caching.md`,
  `parameters_not_closures.md`, `doctest_style.md`, `naming_rules.md`, `fitting_internals.md`,
  `curvature_internals.md`, `sharing_internals.md`, `numerical_contract_catalog.md`; and in `docs/`:
  `naming_conventions.md`, `batching_and_stacking.md`, `grouped_contractions.md`, `numerical_contracts.md`,
  `uniform_equivalence_contract.md`, `entries_apply_probe.md`, `fitting_and_optimization.md`, `glossary.md`.
- The code the plan changes: `t3toolbox/backend/` (`probing.py`, `apply.py`, `entries.py`,
  `sampling_derivatives.py`, `ut3_sampling.py` and its relatives, `fitting.py`, `optimizers.py`,
  `optimizer_display.py`, `geometry.py`, `contractions.py`, `common.py`, `regularization.py`, the
  `fv_*`/`ufv_*` conversion modules), the frontend (`fitting.py`, `optimizers.py`, `tucker_tensor_train.py`,
  `frame_variations_format.py`, `manifold.py`, `shared_geometry.py` and the `uniform_*` files), `tests/`,
  `examples/`, `docs/`.
- Earlier implementation plans of this repository are in `dev/archive/` (for example
  `riemannian_newton_curvature_impl_plan_2026-10-29_shipped.md`, `aligned_padded_frame_impl_plan_2026-11-09_shipped.md`,
  `gn_newton_switch_plan_2026-11-13_shipped.md`). They show the house form of a plan in slices with gates.
  `dev/HANDOFF.md`, section "The next arc", has the process the maintainer planned for this arc.
- The consumer is T3Polynomial, a private repository cloned at `/home/claude/t3polynomial`. Read it; do not
  change it. Its `t3polynomial/fitting.py` and `dev/t3toolbox-upstream-notes.md` show how it calls the library.

## Scope

Part 1 is **what is computed**: the variables and their names, the conventions, and how the layers talk. It
carries out every computation the way the library does it today. Part 2, **how it is carried out** (the plan
object, chunked scans over samples, the choice of sites scanned or vectorized, the contraction order, the loop
form over the orders, what is held and compiled by default, the trial policies b and c, sizing, sharding,
several MC-SGD steps in one program), is **out of this plan**. The section "The two parts" lists what Part 1
fixes "as today" wherever a row is open, and which decided rows have options that wait for Part 2.

Settled for Part 1 and easy to miss, all from 2026-11-16:

- The oracle is a description with no arrays; the data are a batch passed as an argument:
  `newton_cg(geom, oracle, batch, x0, regularizer=None, ...)`. There is no `Problem` at the seam.
- The oracle's one operation takes the batch, `want` and `held`, and returns the values and `held`.
- Derivative data are prepared outside by standalone functions; no fitting function has a keyword for them, nor
  for the equalizing weight.
- Each place a term is used has its own literal contraction string at the call site, through `jet_contract`,
  and every call passes one record of the axis counts of `S`, `T`, `W`, `K` and `C` (5.4).
- A corewise call passes the point's cores `(B, G)` where a frame `(D, U, L, R)` would go.
- Frame consistency is checked in safe mode only: when a `LocalModel` is built, and in the engine when centers
  are given.
- Correct is defined by the definitions (9.3), the definitions are shipped as a public reference module, and
  the variables are documented in four layers, with the dependency structure as a table and as diagrams (9.7).
- Part 1 is not released on its own. All work stays on the branch `graded-messages` until Part 2 is done too.
- The old kernels are deleted as the **last** step of Part 1, behind a gate: the tests by definition and the
  identities pass, and the consumer's benchmark is run with chunking off, new against old. The maintainer runs
  that on CPU and on a GPU pod.
- A dry port of the consumer is part of "done": a script in `dev/graded_messages/` that makes T3Polynomial's
  main fit call through the new interface.

## What the plan must contain

1. **Scope**, in half a page: what Part 1 delivers, what it leaves to Part 2, and the seams it leaves for Part 2
   (where the plan's arguments and fields will be added later without changing Part 1's interface).
2. **The end state.** The modules that exist when Part 1 is done, and what each holds. The public surface, backend
   and frontend: every function and class that is new, changed or removed, with a signature sketch in the house
   style for the new and changed ones. What a frame, a batch, an oracle, a local model and `held` are as data.
3. **A table of names.** Several names are placeholders: the engine's keywords `direction`, `core_direction` and
   `sum_samples`; the helper that turns a jet into a dual jet and its inverse; the data preparation functions;
   the functions for derivative data; the batch record and its fields (the minibatch scale and the drawn indices
   among them); the oracle classes and the name of the operation; the record of axis counts; `jet_contract`; the
   reference module; the modules things live in. Propose one name for each, following `docs/naming_conventions.md`
   and `docs/contributor/naming_rules.md` (names must not be generic), and mark the table as awaiting the
   maintainer. Do not treat your proposals as decided anywhere else in the plan: refer to the table.
4. **The work in slices.** An ordered list of slices, each small enough to review. For each: what it changes
   (files and functions), how, what it adds to the tests, what must pass before the next slice starts, and what
   it deletes. Say how the suite stays green while old and new code stand side by side, and when each old piece
   stops being called. Follow `docs/contributor/refactoring_methodology.md`.
5. **The tests** (9.3): the reference module and the tests by definition; the identities; the structural tests
   (the table of variables against the declared graph, shapes, the dtype rule, uniform masks); the uniform
   layer's tests (equivalence with ragged, exact masks, garbage in the padding); the dispatch tests under jit;
   the doctests. Which existing tests are rewritten, which go, and when. The bridge to the old kernels during the
   port, and where it is removed.
6. **The areas that have been tricky in the past.** The maintainer named seven. Give each its own short section
   saying concretely what Part 1 does about it and how that is tested: safe mode against unsafe mode; the ragged
   against the uniform layout; the dispatch between numpy and jax; shared Tucker factors; the razor between
   backend and frontend; sharding (Part 1 promises nothing new, and must not break what holds today); compiling
   with jit (compile keys, pytrees, value-hashed statics, scan bodies, what is traced and what is static).
7. **The documentation** (9.7): the engine's docstring table, the page in `docs/`, the example file, the `.tex`
   note, the reference module, the dependency table and its diagrams; the existing pages that have to be
   rewritten or removed; the glossary, the naming pages, `CLAUDE.md`, the release notes and the CHANGELOG.
8. **The consumer:** the dry port.
9. **The last step:** deleting the old kernels, and the gate.
10. **Open points for the maintainer.** Anything you find that the decisions do not settle. List it; do not
    settle it yourself. A plan that silently decides an open design question is wrong.

## Rules for both agents

- Check every statement about the library against the code. Name files and functions. Recount anything you rely
  on.
- Write plainly: short sentences, tables where they help, no history, no dead ends.
- Do not change any file of the library, of T3Polynomial, or any other file of `dev/graded_messages/`. The writer
  writes `PART1_PLAN.md`; the reviewer writes `reviews/part1_plan/REVIEW_r<n>.md`. Do not commit.
- Do not put anyone's email address in any file.
