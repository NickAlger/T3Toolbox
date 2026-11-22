# Target decisions

> **Amended for the generalized jets (2026-11-19).** On 2026-11-17 the maintainer decided one jet group `J` of any
> number of axes in place of `T` and `A`; `order` a tuple of maximum orders (the jet shape); inputs that are jets, with
> no `pp`; the entries kind of order 0 only; and he answered O-36 to O-47 (the notes of 2026-11-17 near the end of this
> file; entries 116 to 122 of `REVIEW_DECISIONS.md`; `PART1_PLAN.md`). He signed off the changed plan on 2026-11-19
> (the note "The changed plan signed off"), and the rows and notes the change reaches were amended that day, each with
> an amendment marked "Amended 2026-11-19" beside its old text: rows 2.1, 2.3 to 2.9, 3.4, 5.4, 6.2, 6.6 and 9.4; the
> notes on N16, O-10, O-27 and O-28, the group `A`, and jets in several variables as a future feature; and "Carried to
> Part 2". Where an older text still speaks of `T`, `A`, an order axis, an integer `order` or `pp`, the amendment and
> the plan win.

The decisions that define the target design of the sampling layer, in the order they depend on each other. This
list asks questions. It proposes no answers. It was drafted on 2026-11-15 at the maintainer's request, after he
found that the review of 2026-11-14 had leaned on preserving old behaviour and on the sequencing of the refactor.

Nothing here is about how the refactor is carried out. Migration, staging and what a consumer has to change are
planned only after the target is settled.

## Where this stands (pause of 2026-11-16, 04:46)

The maintainer paused the session at 04:46 on 2026-11-16, to pick up the next day.

- **Counts.** Of the 56 rows in sections 1 to 9: 41 decided, 3 guidance, 1 parked, 11 remaining (7 open,
  3 reopened, 1 to confirm). The section "What remains, in groups" lists them.
- **Decided in the session of 2026-11-15, 15:34 to 2026-11-16, 04:46:**
  - Group A, jets and the misfit, finished: 2.4, 2.5, 2.3, 2.9, 3.6 (the weight inside the engine), 3.7 (what
    happens to the sample stack on each output), 3.3 (which adjoint sweep the tilde names refer to).
  - The notation, 9.6 with 3.1: four letters and a basis. `SAMPLE_VARIABLES.md` is the reference; the folder and
    the Claude doc were brought into line by an editor and a reviewer agent (`reviews/notation_2026-11-15/`).
  - Group B, the frame and the value, finished: 4.1 (four stores, optional centers `Vc`, `Hc`), 4.2 (frame
    consistency as a precondition, never through a dense tensor), 4.3 (three trial-point policies), 4.4
    (arbitrary tails), 3.2 (`nu_L` stays), 4.5 (projection, lift and Hessian assembly stay in the geometry).
  - Group C, what a kind is and the public surface, finished: 3.4 (a kind is closed; the open extension point is
    in the fitting layer), 6.5, 6.2 (the signature sketch of `sampling_variables`), 6.4, 6.3, 8.2 (the `Oracle`),
    9.2.
  - Group D in part: 5.2 (one set of step functions for both layouts), 5.6 (the absorbing dtype rule).
  - Group F in part: 8.1 (what `LocalModel` is), with the name `Jp_norm_sq`.
- **Held for one sitting on performance hooks and defaults, at the maintainer's wish:** 5.3 (sites scanned or
  vectorized); 5.5 (the contraction order: it is in the plan; its granularity, the form of the field and the
  helpers are open, and a reading for jets is proposed and not yet confirmed); all of group E (7.2 and 7.3 the
  defaults and where the plan lives, 7.5 inert padded samples, 7.6 automatic sizing with the names of the sizing
  and pad-and-reshape helpers, 7.7 sharding and device placement).
- **Owed by Claude:** a prototype for 5.4, the terms written generically over several sample axes, `K`, `C` and
  the order group, with `K` on a different operand in each term. Not started.
- **The next questions, one at a time:** 8.3 (the optimizers' contract with a model: how a minibatch is given,
  how `held` passes between calls of an oracle, what is compiled as one program); 8.4 (what the display and the
  diagnostics read, including from a user's oracle); 9.3 (the reference when old behaviour is not it).
- **Open details carried forward** (none of them a row):
  - the keyword names `direction`, `core_direction` and `sum_samples` in the sketch of 6.2;
  - the names and modules of the jet-to-dual-jet helper, its inverse, and the data preparation functions;
  - the name of the trial-policy option and of its three values (4.3);
  - which functions require frame consistency (4.2);
  - how the engine is told that a frame is corewise, so that it does not sweep twice (`SAMPLE_VARIABLES.md`,
    section 11);
  - the damping case for the dots that `Kp` reads, and whether an index subscript follows the basis in formulas;
  - which inputs count for the dtype rule and where the casting is done (5.6).
- **For the branch, when code is written:** `is_consistent` and `fv_frame_consistency_residual` rewritten without
  dense tensors; every function the refactor touches supports arbitrary tails; the library's own open question
  `dev/OPEN_QUESTION_extension_surface.md` given its resolution; the paper-to-code notation map rewritten;
  `docs/contributor/precompute_and_caching.md` amended where it says the carrying cost of a held object is not
  memory (note to 8.3).
- **The Claude doc** is in step with this folder at this pause.
- **Since the pause** (resumed on 2026-11-16 at 13:03): 8.3, the optimizers' contract, was decided in three parts
  (13:51, 14:07, 14:30), and 8.4 at 15:12, with derivative mode moved out of the fitting layer (2.2 and 2.3
  amended). At 15:44 the oracle became a description with the data as an argument,
  `newton_cg(geom, oracle, batch, x0, regularizer=None, ...)`, which rewords 8.1 to 8.4; and the equalizing
  weight became an explicit call with no keyword (6.7 amended). 43 rows are decided and 9 remain. The next
  question was 9.3, decided at 15:58: 44 rows are decided and 8 remain, all of them in groups D and E (5.3,
  5.4, 5.5, 7.2, 7.3, 7.5, 7.6, 7.7). At 16:03 a row was added and decided, 9.7, how the sampling variables are
  documented; and at 16:26, 5.4, the terms over the stacks: 46 of 57 rows are decided and 7 remain (5.3, 5.5,
  7.2, 7.3, 7.5, 7.6, 7.7). At 16:45 the refactor was split into two parts (section "The two parts"): Part 1,
  what is computed, goes forward; Part 2, how it is carried out, holds the 7 open rows and comes later.
  `DESIGN_AFTER_REVIEW.md` and `dev/HANDOFF.md` do not yet carry the decisions made since the pause.

## Where this stood at the earlier pause (2026-11-15, 05:53)

- **Decided on 2026-11-15:** Taylor coefficients replace derivatives, with derivative mode as one adapter at the
  outer level of fitting (2.1, 2.2). Input jets are a point and one direction (2.6). Order 0 is the one-coefficient
  case of the same functions, and the order axis is an optional axis `T` (2.7, 9.4). Three kinds with an `order`
  in place of six (2.8). One function for the plain and the jetted computation, `probe(ww, pp=None, order=0)`
  (6.6). How the direction is passed: the fitting functions take a bundled sample, the frontend methods take `pp`
  as its own argument, and `order` decides, never the type (part of 6.2). The convention for arrows on the edges,
  and the words for their direction (9.5).
- **Since the pause:** 2.4 was decided on the afternoon of 2026-11-15 (see its row), and 9.6, the notation of the
  sampling variables, at 23:23 (`SAMPLE_VARIABLES.md`). What follows describes the state at the pause.
- **Held by the maintainer at the pause:** what a tilde jet is (2.4). His lean is the residual in forward order at the API
  seam. The material is `TYPE_CATALOG.md`: the types of every variable, his picture of arrows on the edges, and a
  type checker for every contraction. In its terms the remaining choice is between turning the order leg once at
  the seed (reversed inside, one product for every grade) and giving the order leg its second orientation (forward
  inside, a correlation beside the convolution).
- **Parked for another arc:** weights on the internal edges as inner products, and orthogonality with respect to
  them.
- **Nearest open questions:** 2.4 above; 2.5 and 2.3, for which the catalog says the misfit is a choice of inner
  product on data jets and derivative mode is another such choice; 2.9; the rest of 6.2; 6.3; 3.4.

## The two parts (decided 2026-11-16, 16:45)

The maintainer (16:35): "The refactor has gotten big enough to the point where I am having trouble holding all of
it in my mind at once. Therefore, I would like to try to find way to cleanly split it into two parts. One of which
we can move forward on without much more planning, and the other which comes later." On the split below (16:45):
"Ok, let's do it."

- **Part 1: what is computed.** The variables and their names, the conventions, and how the layers talk. It
  carries out every computation the way the library does today.
- **Part 2: how it is carried out.** The plan and every performance option. It changes no number.
- **The test for any item:** does it change a result, a name or an interface (Part 1), or only time and memory
  (Part 2)? The line is the maintainer's own, from 2.9: a performance option "changes how the computation is
  carried out and never what is computed".

| | Part 1 | Part 2 |
| --- | --- | --- |
| Sampling | the engine `sampling_variables`; the variables of `SAMPLE_VARIABLES.md`; `jet_contract` and the literal strings; Taylor coefficients; order 0 as the same functions; the weight and the data as arguments; frames with centers; arbitrary tails; both layouts | scans over chunks of samples; the choice of sites scanned or vectorized; the contraction order; the loop form over the orders |
| Fitting | the oracle and the batch; `LocalModel`; the optimizers' seam; the display; derivative data and weights prepared outside | what is held by default; which phases are compiled; the trial policies b and c; which draws the library ships |
| Around it | the reference module; the page in `docs/`, the example and the note; the tests by definition and the identities | sizing helpers; sharding and device placement; padded samples for entries; MC-SGD over several steps |
| Rows | every decided row, with the exceptions below | 5.3, 5.5, 7.2, 7.3, 7.5, 7.6, 7.7 |

**Decided rows whose options wait for Part 2** (the row stands; Part 1 builds only the form the library has today):
2.9, the user's choice of the form over the orders (Part 1 sets it itself: the scan on the uniform layout, the loop on the ragged one, as today; corrected 2026-11-16, 19:35); 4.3, the trial policies b and c (Part 1 has policy a,
with the frame's centers of 4.1 in place); 7.1 and 7.4, the plan and chunk sizes as array shapes (Part 1 has no
plan); 8.3, which phases are compiled and several steps of MC-SGD in one program (Part 1 compiles the CG solve and
the stochastic step, as today, and makes `take` traceable).

**What Part 1 fixes where a row is open,** each as the library does it today:

| Open row | Part 1 |
| --- | --- |
| 5.3 sites | sweeps scanned; map-type operations in a loop on the ragged layout and one site at a time through `xmap` on the uniform one (the maintainer, 2026-11-16, 20:28, for Part 1 only, "must be revisited in Part 2"; this line first said "at all sites at once", which was wrong against today's jetted kernels) |
| 5.5 the contraction order | `contract`'s present default |
| 7.2 the defaults | the model holds the plain sweep variables and the residual |
| 7.3 where the plan lives | there is no plan, and no signature has the argument |
| 7.5, 7.6, 7.7 | no chunking, so no padding, no sizing and no promise about sharding |

- **Part 2 adds to Part 1's interface and does not change it:** the plan arrives as optional arguments and fields.
- **What keeps the two honest:** by 9.3 the result does not depend on the plan. Part 1 is the reference that every
  option of Part 2 is tested against.
- **The cost.** Part 1 has no chunking of samples, and the frontend's `chunk_size` goes. Today's library chunks
  two assemblies on the uniform layout with jax. So Part 1 alone is a step back in memory for large jetted fits
  until Part 2. T3Polynomial's scripts run with chunking off today, the GPU benchmark included: they pass
  `chunk_size=None`. (Corrected on 2026-11-16, 17:39, after the plan's review: this said that the fitting module
  itself defaults to `chunk_size=None`. It does not. With no extra options it passes none, so the library's
  `'auto'` applies; its docstring recommends passing `None`.)
- **A correction to the 5.3 line of the table above** (2026-11-16, 17:39, found by the plan's reviewer and
  checked by Claude). "At all sites at once on the uniform one" is how today's plain kernels do it. Today's
  jetted kernels do not: on the uniform layout with jax they map one site at a time (`get_backend` gives
  `jax_map`, which is `jax.lax.map`; `sampling_derivatives.py` calls it 16 times). Since Part 1 computes order 0
  and jets through one path, "as today" does not pick one. All sites at once makes the jetted intermediates `d`
  times larger; one site at a time is the plain kernels slower. **Open, for the maintainer** (the plan's O-19).
- **Next** (the maintainer, 16:45): the small planning items Part 1 still needs from him; then "a focused
  implementation plan document for part 1, which includes not just what we want to do, but also how we will do
  it", with "more details about what will be done, and less details about the historical path we took to get to
  decisions and dead ends we tried"; written by one Opus agent, reviewed by another, the two iterating until they
  agree; then looked over together.

**The small items Part 1 needed, settled** (2026-11-16, 16:50 to 16:55). Six, put by Claude with a proposal each.

1. **How the engine is told that a call is corewise.** On the frame `(B, G, G, G)` it cannot see that `nu` and
   `nu_L` are one array and would sweep twice. Decided ("Agree"): a corewise call passes the point's cores
   `(B, G)` where a frame `(D, U, L, R)` would go. The structure of the argument says which, and it stays visible
   under `jit`, where a test of object identity would not. The corewise geometry's `frame(x)` returns the point
   itself; "`f` at raw cores" is the same call; on such a call the variables have no basis in their names.
2. **Where frame consistency is checked** (4.2). Decided: in safe mode only, once when a `LocalModel` is built,
   and in the engine when centers are given. Otherwise the engine does not check; someone who calls it with four
   hand-made stores vouches for them. The maintainer: "Agree if you mean safe mode only. Unsafe mode should not
   check." **Reworded on 2026-11-16 at 19:30 (O-5, O-6 below):** nothing is checked in the engine or anywhere in
   the backend, which has no safe mode. The frontend checks, and only a frame the caller supplied, when the model
   is built.
3. **The old kernels are deleted as the last step of Part 1,** behind a gate: the tests by definition and the
   identities pass, and the consumer's benchmark is run with chunking off, new against old. The maintainer: "Yes,
   last step. We will run on CPU, and I will also start a runpod for GPU at that time." **Changed on 2026-11-16
   at 19:11 and 19:30 (O-1, O-2 below):** the benchmark moves to Part 2 and compares with the released 2026.5.0;
   the deletion is gated on the tests alone.
4. **Part 1 is not released on its own.** The maintainer: "No, both parts are needed for the release. Work
   remains on the separate branch until both parts are complete."
5. **A port of the consumer is part of "done"** (ground rule 0.4): the plan includes a dry port, a script in this
   folder that makes T3Polynomial's main fit call through the new interface. The maintainer: "Agree. You should
   have access to edit T3Polynomial through github connectors. If not I can set that up." (Until then Claude had
   been told to read that repository and not edit it.)
6. **Names.** Several are placeholders: three keywords of the engine, the helpers, the batch record, the oracle
   classes, the reference module, the modules things live in. Decided: the plan's writer proposes one complete
   table of names, and they are settled when the plan is looked over.

**What the writer and the reviewer of the plan are to keep in mind,** from the maintainer, as things that "have
been tricky in the past": safe mode against unsafe mode; the ragged against the uniform layout; the dispatch
between numpy and jax; shared Tucker factors; the razor between backend and frontend; sharding; compiling with
jit.

**The names of Part 1, and the form of the merged sampling functions** (2026-11-16, 18:13 and 18:18). The plan's
table of names (`PART1_PLAN.md`, section 3) is settled: the maintainer agreed with every row but one at 18:13, and
of the oracle's operation he said at 18:18 "query is better than evaluate in my opinion" (the notes of 8.3 wrote
`evaluate` as a placeholder). The one row held back was N16, the form of the merged sampling functions, which the
plan had proposed to keep as today "so that only the calls listed move". His concern: "that it should be thought
through and a first principles decision should be made". Today's forms disagree with each other (`t3_probe(ww, x)`
against `t3_apply(x, vecs)`; `tv_probe(ww, variation, frame)` against `utv_probe(ww, frame_data,
variations_data)`), and no note of the library records that as deliberate. Decided ("I like your proposal, let's
go with it"):

- **One rule: the object the function is evaluated at comes first, then the sample, then what is applied.** It
  is the order the engine and the oracle already have (`sampling_variables(kind, frame, sample, ...)`,
  `oracle.query(frame, batch, ...)`).
- **The keywords follow:** `pp=None, order=0`, then `sum_over_samples=True` where there is a sum. `pp` stays a
  keyword, as 6.6 has it, so a call at order 0 does not pass `None`. **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** the keywords are `order=()`
  and then `sum_over_samples=True`; there is no `pp`, and the entries functions have no `order`. The sketches below,
  with `pp=None, order=0`, are superseded; `PART1_PLAN.md` section 2.4 has the forms.
- **One name for one thing:** `x` for a Tucker tensor train's cores, `frame`, `variations`, `ww` (or `index`),
  `residual`, `core_directions`, on both layouts.

```python
t3_probe(x, ww, pp=None, order=0)
t3_apply(x, ww, pp=None, order=0)
t3_entries(x, index, pp=None, order=0)
tv_probe(frame, variations, ww, pp=None, order=0)
tv_probe_transpose(frame, ww, residual, pp=None, order=0, sum_over_samples=True)
t3_probe_corewise_transpose(x, ww, residual, pp=None, order=0, sum_over_samples=True)
t3_probe_corewise_hessian(x, ww, residual, core_directions, pp=None, order=0)
```

- The uniform twins take the same lists. The frontend's instance methods follow the rule with `self` as the
  object (`x.probe_corewise_transpose(ww, residual, ...)`). The frontend's tangent transposes stay static methods
  with the frame first (`T3Tangent.probe_transpose(frame, ww, residual, ...)`): their input is a residual and
  their output a tangent, so they are constructors, like `T3Tangent.zeros(frame)`.
- Every caller of these functions changes; the slice that merges them already visits every caller.
- **The three ambient transposes** (decided 2026-11-16, 18:35: "take the rule's order and names, default
  sum_over_samples to False"). They keep their own code and are not merged with anything (6.3); they have no
  object to be evaluated at and no `pp` or `order`:
  `t3_probe_ambient_transpose(ww, residual, sum_over_samples=False)`,
  `t3_apply_ambient_transpose(ww, residual, sum_over_samples=False)`,
  `t3_entries_ambient_transpose(index, shape, residual, sum_over_samples=False)`, and the frontend's static
  methods likewise. Their default is the opposite of the other transposes' (3.7) on purpose: for them the sum
  folds `W` into the CP rank, which is another operation than summing blocks.
- Not settled by this: whether `dense_probe(vectors, T)` takes the rule too (`dense_probe(T, ww)`).

**The plan's open points, as the maintainer answers them** (from 2026-11-16, 19:11). The points are stated in
`PART1_PLAN.md`, section 11.

- **O-1, O-2 (the benchmark gate):** "let's move this gate to the performance work in part 2." So Part 1 has no
  benchmark gate. This changes small item 3 above, which put the deletion of the old kernels behind it.
- **O-3 (the engine on the uniform layout):** as proposed. The engine takes the uniform frame or point `.data`,
  masks once on entry and re-masks core-shaped outputs on exit; the `ut3_`/`utv_` twins stay as thin wrappers.
- **O-4 (the geometry's `frame(x)`):** as proposed. `(frame, centers)` with `centers = (Vc, Hc)` or `None`, on
  the backend protocol only; the frontend geometries keep returning frame objects; the frontend `COREWISE` keeps
  its `T3Frame` `(B, G, G, G)` and passes `(B, G)` to the backend.
- **O-7 (optional outputs of an oracle):** as proposed, a class attribute `optional_outputs`.
- **O-9 (the frontend fitting surface):** as proposed. One `fitting.local_model(geometry, oracle, batch, x,
  regularizer=None, hessian=..., curvature_damping=None)`; the six model factories, `least_squares_problem` and
  `LeastSquaresSetup` go.
- **O-11 (the dtype rule, 5.6):** the inputs that count are "all numerical (floating point) arrays whose entries
  could affect the numerical values of the output". So an integer index does not count, and neither does an
  argument that nothing wanted reads.
- **O-12 (`nu_L_dot`, `eta_D_dot` with both directions):** as proposed, the raw form, the one `Kp` reads.
- **O-13 (spellings on a corewise call):** as proposed, the names with a basis are not accepted.
- **O-1, O-2, confirmed** (19:30): the old kernels are deleted as the last step of Part 1 on the tests alone, and
  Part 2's benchmark compares the branch with the released 2026.5.0. ("Yes, delete and use 2026.5.0 branch for
  comparison in Part 2.")
- **O-6 (safety), decided 19:30 as Claude proposed after the maintainer's objection.** The backend has checkers
  only: functions that return a residual and never raise and never read a mode (`fv_frame_consistency_residual`,
  made free of dense tensors, and a centers' residual beside it, with uniform twins). The engine, the backend
  `LocalModel` and the backend optimizers check nothing, as `backend/optimizers.py` already says of itself.
  Enforcement is the frontend's, in safe mode, on concrete arguments, where a frame comes in from the user,
  through a cached `T3Frame.consistency_residual` like today's `orthogonality_residual`. `backend/optimizers.py`
  keeps its three pinned imports. **This rewords small item 2 above:** nothing is checked "in the engine".
- **O-5 (what frame consistency checks), decided 19:30 as proposed.** A frame the caller supplied (a
  `start_frame`, or a frame given to `fitting.local_model`): `L` against `R` through `D`, once, when the model is
  built. A frame the library built in the same call: nothing. Centers: not checked in the frontend, since by O-4
  no frontend user supplies them; the backend checker exists for users of raw data and for the tests. `U` on a
  supplied frame is not checked, as today.
- **O-8 (the form of `held`), decided 19:35.** The maintainer (19:30): "We are dancing around the fact that the
  natural datatype for held variables really is a dictionary. I'll allow a dictionary or dictionary-like object
  here. What about a held variables object, which has slots for all the allowable held variables only. It can be
  used like a dictionary, but will refuse if someone tries to put in an invalid key." Decided as Claude then
  wrote it out:
  - `HeldVariables` is a frozen record with one slot per variable that some other variable reads (taken from the
    declared graph, and a test checks that the two agree), each `None` when not held, each with its shape comment.
  - It reads like a dictionary: `held['mu']`, `'mu' in held`, `held.keys()` (the filled slots), iteration, `len`.
    An unknown name raises, at construction and at lookup.
  - Adding returns a new object (`held.with_values(mu=mu)`); the object is never changed in place (0.7; a model
    never changes after it is built; a traced object must not be mutated).
  - The engine takes it directly, `sampling_variables(..., held=None)`, in place of `**held` keywords, and the
    oracle's `query(..., held=None)` passes it through.
  - No shape check in the object: the shapes depend on the frame, the sample and the order, and the engine checks
    every held variable it is handed at every call (6.5).
  - A numpy-only user gets an ordinary object; it is registered with jax only where jax is importable.
  - It lives beside the engine, in `backend/sampling_variables.py` (the plan's N26 had `backend/fitting.py`).
  - The oracle protocol still types `held` as any object, the oracle's own business; `HeldVariables` is the
    least-squares oracle's.
- **O-10 (how `jet_contract` learns the layout), decided 19:35: it does not.** The call form of 5.4 already has
  the form over the orders as an argument, `jet_contract(subscripts, *operands, lens, over_orders)`, with the two
  values of 2.9: `'scan'` (one jet carried and shifted; a real `lax.scan` on jax, a Python loop on numpy) and
  `'loop'` (a loop with slices). `jet_contract` reads whether its operands are jax, as every function does, and
  knows nothing of the layout. In Part 1 the engine sets the value: `'scan'` on the uniform layout and `'loop'` on
  the ragged one, which is how the library runs today. In Part 2 it comes from the plan, with no change of
  signature. `GroupLengths` keeps the five fields of 5.4. **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** four fields, `J`, `W`, `K`, `C`, and the
  call takes the output's jet shape: `jet_contract(subscripts, *operands, lens, order, over_orders)`. Both forms are built and tested in Part 1.
  **This corrects the section "The two parts":** "Part 1 has the scan" for 2.9 was wrong against the library,
  whose ragged layout runs the orders in a Python loop. On O-6 the maintainer set the frame: "Backend importing from frontend
  is not acceptable. Perhaps we can put the checking functionality in the backend somewhere, then the frontend
  safety module can import it. There is no such thing as safe mode in the backend." On O-8: "I am concerned the
  proposed pytree approach will not work for the numpy-only user."

- **O-14 (the two estimates read off the data), decided 19:54:** "compute inline, do not ship at this time.
  Sometime in the future after this refactor is complete, we will do some more in-depth investigations on
  scaling. That will be the appropriate time to add such things."
- **O-15 (`docs/symmetric_probe_derivatives.tex` and the paper-to-code map), decided 19:54:** the document is
  archived. The map is updated with the new variables where appropriate: "the new variables are different, but
  most of them have some kind of analog in the paper. We explain the differences." It moves out of `CLAUDE.md`
  into a file in `docs/`, "so that it is de-emphasized".
- **O-16 (frontend access to the preparation functions), decided 19:54:** "Both backend and frontend. Frontend
  user must be able to fit their tensors without digging through the backend to find the helpers."
- **O-17 (the timing laps), decided 19:54:** as proposed, `oracle` in place of `sweep` and `residual`.
- **The two users** (the maintainer, 19:54, on O-18 and the points like it): "there are two different
  hypothetical users: the frontend user and the backend user. They each must be able to do all the important
  operations with tools in their own layer. The backend is purely functional on basic types (with limited
  exceptions, such as the held variables object we just discussed), and the frontend is OO-based. The frontend is
  a wrapper over the backend. Any code that contains nontrivial logic should be in a backend function which is
  called from the corresponding frontend function. This is to be understood *within reason*; exceptions can be
  made in limited circumstances."

- **Checking in the two tiers** (the maintainer, 20:04, after Claude proposed that the engine make every
  structural check so that both users get the same errors): "The backend is explicitly looser with checking
  requirements. We make no checking guarantees about backend functions (some functions have them, others don't,
  that's fine). However, we must provide the *tools* to the backend user to do checks themselves if they so
  desire. There is no safe mode in the backend. If the user passes a bad input to a backend function, that is on
  them. They should have checked it beforehand with one of the checkers we provide. For example, the backend
  functions that require orthogonal frames will happily compute a wrong answer if the user provides a
  non-orthogonal one, but the backend also provides a function that checks if a frame is orthogonal." So the
  requirement on the backend is that a checker exists for each thing the frontend checks, not that backend
  functions check. Claude's proposal is withdrawn.

- **What follows from the two rules, decided 20:15** ("Yes, this is the right reading", and the answers to
  Claude's items 3 to 6):
  - **O-18 (the weight's shape).** A backend checker for the weight's exact shape, beside the two validators of
    the sample; the frontend calls it where a weight comes in; the engine promises nothing about it. The
    frontend's `_canonical_weight` goes (2.5 removed the rules it implements).
  - **One checker for a whole batch,** a method of the least-squares oracle (`oracle.check_batch(batch, shape)`,
    name open): the form of the sample, its shapes against the mode sizes, the shape of `y`, the shape of the
    weight. Built from the small checkers, which stay public. The frontend optimizers and `fitting.local_model`
    call it once where a batch enters.
  - **What the engine checks itself, and nothing else:** what it needs in order to run (an unknown name in
    `want`, a wanted variable whose argument is missing); the shapes of held variables (6.5); the ranks of
    operands in `jet_contract` (5.4).
  - **The checks of the sample** stay as the plan had them: backend validators that the frontend methods call.
  - **The preparation functions and the jet helpers** are reachable from `t3toolbox.fitting` under the same
    names; the one that acts on a tensor is also a method, `x.unscale_coordinates(scales)`.
  - **The oracle and the batch** are one class each for both tiers, reachable from `t3toolbox.fitting`. One
    backend `pack_batch` packs a batch for the uniform layout, and the frontend calls it. A frontend-only fit
    from derivative data, importing nothing from `t3toolbox.backend`, is a test beside the backend-only one.
  - **The engine has no frontend entry.** "I would leave the engine to the backend. If a use cases arises, we can
    add the frontend version at that time."
  - **Writing an oracle is backend-level work,** a stated exception to the rule of the two users. The frontend
    optimizers take an oracle object, so they can drive a user-written one.
  - The reference module of definitions is backend only.

- **Answers of 20:28.**
  - **O-19 (map-type operations on the uniform layout):** "It does not matter what is done here for Part 1
    because we will address this in detail in Part 2. Let's do the xmap style one (less memory) temporarily. This
    must be revisited in Part 2." So Part 1 maps one site at a time through `xmap` on the uniform layout, for
    order 0 and for jets alike; the strings have no site group `S` in Part 1. **Carried to Part 2 with row 5.3.**
  - **O-20 (`dual_jet_to_jet` where a weight is zero):** "Raise in safe mode, divide by zero in unsafe mode,
    divide by zero in backend." So the backend function divides, and the frontend's function of the same name
    makes the check in safe mode before calling it.
  - **O-23 (the dependency diagrams):** as proposed: generated from the declared graph, committed, and tested
    against a fresh run.
  - **O-24 (the switch slice):** "Split. Two commits." For one slice the frontend optimizers accept the old call
    form and translate it; the next slice moves the examples and pages and deletes the translation.
  - **O-25 (the bridge):** the first reading: a temporary test file that compares with code never rerouted onto
    the engine, run in CI from P6 and deleted in P19.
  - **O-26 (the folder's prototype scripts):** as proposed: they stay as records that run against `v2026.5.0`,
    and the folder's README says so.

- **Answers of 20:35, which complete the list.**
  - **O-21 (what the minibatch scale multiplies).** The maintainer's view: "The minibatch problem is viewed as a
    new problem with its own objective function, gradient, and other variables. The problem is scaled so that the
    minibatch problem is a stochastic estimator for the full problem. All quantities in the minibatch problem are
    scaled accordingly, so they provide a little self consistent universe." Recorded as Claude then put it: the
    scale belongs to the batch's inner product on data space (`<a, b>` is `scale * <a, omega^2 b>`), which is
    what making it a field of its own and not folding it into the weight already meant. So it multiplies `f`,
    `g`, `JtJp`, `Jp_norm_sq`, `Kp` and `f_resolution`, and nothing else: `r` and `Jp` are the plain vectors at
    the drawn samples, and `r_block_sumsq` and `y_block_sumsq` are plain sums of squares (their ratio, which the
    display shows, would not change anyway). Every identity holds: `f = 1/2 <r, r>`, `<Jp, r> = <p, g>`,
    `Jp_norm_sq = <Jp, Jp>`.
  - **O-22 (the backend corewise Hessian).** It exists, and it takes what the transposes take: a vector `c` of
    the type of `z`, no weight, turned inside. The two are then one definition: for the scalar function
    `phi(x) = <c, z(x)>` with the plain inner product, the corewise transpose is the gradient of `phi` and the
    corewise Hessian is its second derivative applied to a direction. The curvature term of a fit is the case
    `c = omega^2 r`. It is tested against automatic differentiation of `<c, t3_probe(x, ww)>`. It stays although
    it has no frontend method (6.3 read strictly would drop it): it names a capability a backend user would not
    guess from the engine.
  - **O-27 and O-28 (jets of a dense tensor; `dense_probe` under N16).** Three backend functions, beside their
    kinds: `dense_probe(T, ww, pp=None, order=0)`, `dense_apply(T, ww, pp=None, order=0)`,
    `dense_entries(T, index, pp=None, order=0)`. The reference module calls them for `z`, so the code for the jets
    of a dense tensor exists once; the examples and tests use them for ground truth. Not wrapped in the frontend,
    which has no class for a dense array. This amends 6.3, which kept `dense_probe` "as it is". **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):**
    `dense_probe(T, ww, order=())` and `dense_apply(T, ww, order=())` take input jets; `dense_entries(T, index)` has
    no jets (P4b).
  - **Every open point of the plan is now answered.** The plan's body is brought in line in one pass and checked
    by the reviewer.

- **The points the revision raised, and the names added, decided 20:59** ("Agree with all O-29 -- O-35. Names
  approved."). As Claude had answered each:
  - **O-29.** `check_batch` is an optional member of the oracle protocol: the frontend calls it when the oracle
    has it and checks nothing about the batch otherwise.
  - **O-30.** The protocol says `held` is any object, and that in a compiled program it must be something jax can
    carry: arrays, tuples, `None` or a registered record.
  - **O-31.** The geometry's consistency member is dropped (N21 is withdrawn from the table of names). A backend
    user has `fv_frame_consistency_residual`.
  - **O-32.** Settled by the amendments to rows 4.2 and 8.3.
  - **O-33.** The new page takes the notation paragraph of `CLAUDE.md` and the probing map of
    `docs/entries_apply_probe.md`; other pages link to it.
  - **O-34.** `UniformTuckerTensorTrain` gets `unscale_coordinates` too, over a uniform backend twin.
  - **O-35.** One scale per coordinate: `scales` is `len=d`, `elm_shape=(Ni,)`, as the prototype has it.
  - **Names:** `over_orders='scan'` and `'loop'`; `dense_apply`, `dense_entries`; `check_sample`, `check_data`,
    `check_weight`; `LeastSquaresOracle.check_batch(batch, shape)`; `pack_batch(kind, order, batch, N)`;
    `HeldVariables.with_values`; `T3Frame.consistency_residual`, `UT3Frame.consistency_residual`;
    `fv_centers_consistency_residual`, `ufv_frame_consistency_residual`, `ufv_centers_consistency_residual`; the
    keyword `frame=None` on `fitting.local_model`; `docs/paper_to_code.md`.
  - **Nothing in the plan is open now.** Its writer folds these in with the six minor items of the reviewer's
    last check, and the maintainer then reviews the plan for sign-off.

**How Part 1 is implemented** (the maintainer, 2026-11-16, 20:51). "Once the plan is complete I would like to
review it. When I sign off, we can begin implementing. You will orchestrate two new agents: the implementer and
the reviewer. Both implementer and reviewer are Fable 5.2 on medium effort. The implementer writes a slice, then
the reviewer critiques it, and they go back and forth until the reviewer signs off. Then you look over it. It it
is good, they can continue and start the next slice. If there are major issues along the way, please consult with
me. If there are minor issues, I trust you to make good decisions." Nothing is implemented before his sign-off.
**Amended 2026-11-19:** from P4a on, the two agents are Opus 5.5 on high effort, and the same pair is kept from
slice to slice; see the note "The changed plan signed off" of that date, below.

- **Memory when running the gate** (the maintainer, 21:02): "one thing to beware of is running sphinx and the
  tests at the same time, especially if the tests are being run in parallel. On my laptop (13GB and 8 cores) we
  have run into OOM issues before." The workspace the agents run in has 2 cores and 7 GB. So: the steps of the
  gate run one after another, never two at once; the implementer and the reviewer never run tests at the same
  time; the number of test workers and the peak memory are measured in the baseline slice before a full parallel
  run is trusted.

**The plan of Part 1 is approved** (the maintainer, 2026-11-16, 21:16: "Plan approved. Go ahead."). The plan is
`PART1_PLAN.md` at commit `9d070e1`: 21 slices, P0 to P20. The implementation follows
`implementation/BRIEF.md`; each slice leaves a handover and its reviews in `implementation/`.

**Groups in contraction strings: the site group removed, and a rule** (the maintainer, 2026-11-17, 01:59 to
02:23). In the review of P1 the maintainer asked what the group `S` was. It was the site axis, the library's
leading `d` of a uniform supercore, written as a group of zero or one axes so that one string could serve a single
site and all sites at once. Claude had put it into the 5.4 prototype and row on 2026-11-16 without raising it as a
question; the library writes that axis as a lowercase `d`, always present in a string for the uniform layout.

- **The rule:** "I am worried that the introduction of the optional T has given license to invent new optional
  groups for convenience. This is not the case. New groups must be weighed carefully and only introduced with good
  reason and after deliberation."
- **On `S`:** "I do not like the S group for site, because it invites over-abstraction, treating ragged and uniform
  the same in circumstances where they should be thought of differently. A uniform supercore is a different kind
  if thing than a single ragged core, and reusing the same operation for both is a code smell." Decided: "S
  should go." Row 5.4 is amended; `GroupLengths` has the fields `T`, `W`, `K`, `C`.
- **Row 5.2 stands:** "Sweep polymorphism (Row 5.2) is intended and stays because the step in a sweep is doing the
  same thing either way: operating with a single core."
- **For Part 2 (row 5.3):** if map-type operations are done at all sites at once on the uniform layout, that is
  uniform code of its own, written with the `d` axis, not the shared step with an optional group.
- **Superseded (2026-11-17, 13:31 to 14:44, by "One jet group, of any number of axes" below; row 5.4 amended
  2026-11-19): `A` is merged with `T` into the one jet group `J`.** Kept as the record of what was decided at 02:31.
  The rule on new groups above stands.
- **The group `A` is kept** (decided 2026-11-17, 02:31, after deliberation: "Ok this is convincing. Let's keep A.
  I see how it has meaning as a jet axis in the same way as T."). The maintainer put three tests to it, and
  Claude's answers are the record of why it stays:
  - *Is its absence a natural special case, or is it one code path over two different objects?* An affine
    variable is a jet: `w + s p`, whose coefficients beyond the first two are zero. `A` holds those two. At order 0
    the jet is its constant term, one slot, squeezed, exactly as `T = ()` arises. So `A` and `T` are one group,
    the order axis, in two storage formats: `order+1` slots, or two. Where `S` treated two different things
    alike, `A` tells two formats of one thing apart.
  - *What would removing it cost?* Every replacement reopens a decided row: storing affine variables as full jets
    makes every sweep step a product of two full jets (`order+1` contractions for two; it reverses 2.9, and the
    library stores `xi` with two slots today); writing `T` for both makes `jet_contract` tell the formats apart by
    the length of an axis; handling the affine case in each step brings back the branch on order 0 that 2.7
    removed. 16 of the 41 strings carry `A`.
  - *Does it mean something with more than one axis?* Yes: with several jet variables `A` has one axis of two
    slots per variable, as many axes as `T`; it holds the polynomials of degree at most one in each variable.
    (For `w + s p + t q` that stores one known zero, the `s t` slot.)
  - **The definition, for the record:** `A` is the order group in the two-slot format of a variable of degree at
    most one. It always has as many axes as `T`, and it is absent exactly when `T` is. The record of axis counts
    has no field for it; it takes `T`'s.
  - **`T` and `A` are not ordinary groups of `contract`.** The same letter on two operands means that their jets
    multiply as truncated polynomials, not that their axes line up. They are the two jet axes; `C`, `W`, `K` are
    the grouped stack axes. The maintainer: "It's too bad english does not have a third form of letter aside from
    lowercase and uppercase which we could use to distinguish grouped axes from jet axes." What tells them apart
    is that the two letters are reserved and that the string is handed to `jet_contract`; the documentation says
    so in one table of the letters (P19).

**Superseded (2026-11-17, 13:31 to 14:44) by the notes "One jet group, of any number of axes" and "The public
interface is generalized from the start", below: jets in several variables are built in Part 1 (slices P4a to P4d).
The ledger entry planned here is replaced by what `PART1_PLAN.md` section 8 lists.**

**A future feature: jets in several variables** (the maintainer, 2026-11-17, 02:35: "Also worth considering
support for multidimensional jets as a future feature. I don't know if anyone will ever use it, but it would
complete the picture nicely."). Not part of this arc, and nothing is designed. What the present design already
says about it: the order group `T` would have one axis per jet variable (`x + s u + t v` has two), `A` one axis of
two slots per variable, and a product is the truncated product variable by variable. What is single-variable
today: `order` is one integer, the sample has one set of perturbation vectors `pp`, `GroupLengths` accepts `T` of
0 or 1 only, and `jet_contract` requires one order axis leading its term. It goes into the library's ledger of
deferred items (`docs/contributor/deferred_and_rejected.md`) with the documentation slice.

**A `kind` argument where a function is three functions in one** (the maintainer, 2026-11-17, 12:02, in the
review of P3). The dual-jet helpers and the derivative conversions had no `kind` and told the probe type from the
others by whether they were given a tuple; a packed uniform probe array passed whole was then misread, silently.
The maintainer: "A kind argument is acceptable and sometimes even desirable for functions whose output differs
based on kind. In the examples listed it makes sense because the code function is really three mathematical
functions bundled onto one, so it makes sense the user may have to say which one they want." So:

- `jet_to_dual_jet(kind, order, jet, weight)`, `dual_jet_to_jet(kind, order, dual_jet, weight)`,
  `derivatives_to_taylor(kind, order, jets)`, `taylor_to_derivatives(kind, order, jets)`. The argument order is
  the one the other functions of the two modules have. This changes the signatures of the names N4 to N7 of the
  plan's table. `derivative_weight_to_taylor_weight(weight, order)` is unchanged.
- The type is decided by `kind`, never by the type of the object. For `'probe'` the function goes over the modes,
  so a packed uniform array is taken mode by mode and comes out right, as a tuple; no code for the uniform layout
  is added.
- **The names of a corewise call** (`SAMPLE_VARIABLES.md`, section 9; `nu_dot`, `eta_dot`, `nu_tilde_dot`,
  `eta_tilde_dot` among them) are confirmed: "those are fine."
- Also decided by Claude in this slice, as minor: a public backend checker `check_coordinate_scales(scales,
  shape)`, by the rule that the backend offers the checkers; and the frontend's `fitting.rescale_coordinates` is
  its own function that calls it, so that a frontend user gets the error where the wrong `scales` are passed.

**One jet group, of any number of axes** (the maintainer, 2026-11-17, 13:31 and 13:49). Decided in principle;
the slices are to be planned at the pause after P4 and carried out before P5.

- **The idea, in his words:** "The generalization is to allow jet axis groups of arbitrary size, of which T and A
  are special cases which have len=0 or len=1, and A has further restrictions. We would lift these restrictions,
  allowing multidimensional jets of arbitrary order, and the jet contraction would correspond to truncated
  multiplication of polynomials, which can now be multidimensional. The user would have to tell the jet
  contractions engine what jet shape they want as output (maximum order of the output polynomial in each
  direction) ... This seems to be the natural mathematical structure behind what we are doing here." Efficient
  algorithms for several variables belong to the performance phase, behind the seam of the jet contraction; "we
  could start out with algorithms that work for our desired cases, with less efficient algorithms for the more
  general case, then figure out those later."
- **Claude's analysis, which he accepted** ("The box truncation is exactly what I mean. I agree with your
  reasoning that an explicit output shape is required and that is OK."):
  - A jet in `m` variables with maximum order `n_j` in direction `j` is an element of the polynomial ring in
    `s_1 ... s_m` cut off at `s_j^(n_j+1)`: a tensor product of one-variable rings. The jet group has `m` axes of
    sizes `n_j + 1`; order 0 is `m = 0`. An affine variable is a jet with smaller sizes (two slots per axis), so
    `A` is no longer a group of its own: there is one jet group, and each operand carries its own sizes.
  - **The truncation is a box, not total degree.** The dual jets rest on a property of the ring: pairing a jet
    with a dual jet is the top coefficient of their product (row 2.4). The box keeps it (the top coefficient is
    the corner `s_1^(n_1) ... s_m^(n_m)`, and the turn is a flip along every jet axis). Truncation by total degree
    does not, for `m >= 2`. Data given by total degree fit in the box with zero weight on the slots not measured.
  - **What an operand's shape means:** a polynomial whose higher coefficients are zero. A full jet is a
    truncation and is valid only up to its own order, so the output's shape must not exceed the shape of any
    operand that is itself a truncation.
  - **The output shape is explicit, with no default.** The maximum of the inputs is wrong for two exact affine
    operands; the sum of the degrees is wrong for truncated ones. The engine always knows it: it is the order of
    the computation.
  - **Outside the jet contraction:** `order` becomes a jet shape; the sample has one set of perturbation vectors
    per variable (or, more generally, the inputs are a jet); each variable has its own jet shape (the full one,
    or the input's); the data and the weight take the box; the dual jet turns every jet axis; the derivative
    conversions use the product of the factorials; strings that differ only in `A` against `T` merge;
    `GroupLengths.T` is `m`; the display, the reference module and the dense functions get `m` order axes.
  - **A cost named:** with one jet letter a string no longer shows which operand is affine.
  - **Staging proposed by Claude, and not taken** (see the next entry): the jet contraction for general `m` from
    the start, by a simple correct algorithm; the engine and the public interface single-variable in Part 1,
    written so that widening later touches only the items above.
- **How it is done** (13:49): "I would like to do this as a new slice (or multiple slices) at the pause, inbetween
  P4 and P5. We would do a plan phase with writer and critic Opus agents to create the new slices and update the
  plan document before implementing."
- This supersedes, for the jet contraction, the entry "A future feature: jets in several variables" above, and it
  changes row 5.4 (the groups `T` and `A`), the 41 strings and P1's `jet_contract`, which is rebuilt.

**The public interface is generalized from the start** (the maintainer, 2026-11-17, 13:58). Decided: the scope.
Open with him: the structure below, and the letter.

- **Decided, in his words:** "I would like to generalize it from the start. I am concerned that if we do not do it
  right in the first place, downstream code and future features may start to assume the current structure, and we
  will have to break them later if we generalize. This release is already mega-breaking, so this is the time to do
  it." So Claude's staging (the public interface single-variable in Part 1) is not taken. What Part 1 owes for
  several variables is the right interface and correct results, tested against the definitions; speed for several
  variables is Part 2.
- **The letter.** He proposed `J` for jet, noting that it meets the `J` of the Jacobian and that `T` and `A` have
  downsides of their own ("they are likely variable names for tensors in a user's code"), and asked for a
  recommendation. **Claude recommends `J`**: `T` already meets the dense tensor argument `T` in the very
  signatures where the shapes are written (the dense functions of P2), and reads as transpose and as tangent;
  the Jacobian's `J` lives in the names of variables (`Jp`, `JtJp`, `Jp_norm_sq`) and the group's would live in
  contraction strings and shape comments, which are different places; `J` is free among the symbols of the
  stores and cores; the other free letters are worse (`O` reads as zero, `P` sits beside `pp`, `S` was just
  removed and names the variable). Not yet confirmed by him; the working choice for the plan phase.
- **The structure Claude proposed to him (14:00), not yet confirmed:**
  1. *One concept, the jet shape:* a tuple of maximum orders, one per jet variable, `()` for no jets, in place of
     the integer `order` everywhere. An integer stays as shorthand for one variable and `0` for none. The number
     of jet variables is `m = len(order)`, and every array that carries jets has exactly `m` leading jet axes.
     Row 6.2 keeps its sense: `order` decides how the leading axes are read, never the type of the object.
  2. *The inputs are jets:* the sample's vectors carry the jet axes themselves, with sizes of their own
     (`ww[i]` of shape `J_in + W + (N_i,)`), and `pp` leaves every signature. The line `w + s p` is the input jet
     of two slots, built by a helper. Inputs along a curve, or affine in several variables, need nothing new. The
     affine case stops being a concept: an input of two slots makes `xi` a jet of two slots and the general
     product handles it.
  3. *Everything of the type of `z`* (data, residual, weight, block sums) has the jet axes where the one order
     axis is today. The dual jet turns every jet axis; the derivative conversions use the product of the
     factorials.
- **Costs and open points named to him:** the most common call gets one step longer
  (`probe(ww, pp, order=3)` becomes a call through the helper), which reverses the form row 6.6 decided, so it is
  his to confirm; the entries kind, whose sample is an integer index and cannot carry jet axes; the display of a
  table with several order axes; the rework of P1 to P3 and the rewriting of every signature with `pp` or
  `order` in P5 to P20.
- **Confirmed since** (14:20, 14:39): one `ww` plus `order`, with a helper for inputs of the `(ww, pp)` type
  ("Definitely I prefer having one ww plus order as the API"); and an input may have any number of slots along a
  jet axis, so its higher coefficients are free ("Let's allow it (differing p1, p2) per your proposal"). `order`
  cannot be inferred from the input: the shape does not say how many leading axes are jet axes, and the order of
  the output is not the degree of the input. The cost of a product follows the input's own size (one step per
  slot), so a line costs what it costs today.
- **The letter is `J`** (the maintainer, 14:44). `T` and `A` leave the contraction strings, and `GroupLengths.T`
  becomes `GroupLengths.J`.
- **The entries kind is of order 0 only** (the maintainer, 14:44, on his own question of 14:39: "For entries, the
  setting is different: we are fitting a tensor to data, and it is not clear what the function should be"). The
  reason: an index is a discrete argument, and a tensor on a grid has no derivative until it is extended off the
  grid. The extension is a choice the tensor does not contain, and once it is made it is a curve of input vectors
  starting at `e_i`, which is apply. The `e_index + s P` of the released `entries_derivatives` is one such choice
  (the multilinear extension); a nodal basis and finite differences are others, and none is the derivative of an
  entry. What entries saves over apply at order 0 (an integer sample, a slice in place of a product, a scatter in
  the transpose) is mostly lost once any slot is a dense vector. So: the entries functions take an index and no
  `order`; an oracle of the entries kind has no jet shape; a helper turns an index into one-hot vectors, and a
  user who wants the old behaviour calls apply with the input jet `(one_hot(index), p)`, which also serves a
  tensor with some discrete modes and some continuous ones. This removes a released feature
  (`entries_derivatives` and its tangent forms; no example and nothing found in the consumer uses it), for the
  upgrade notes. It amends every row that gives entries an `order` or a `pp`, among them the ambient transpose
  `t3_entries_ambient_transpose(index, shape, residual, sum_over_samples=False)`, which stands as written.
- The brief of the plan phase (`reviews/jet_group_plan/BRIEF.md`, item 3) carries this.

**The open points of the jet-group plan, as the maintainer answers them** (2026-11-17, 16:21; the points are in
`PART1_PLAN.md` section 11.2, the plan change agreed by the critic in three rounds, `reviews/jet_group_plan/`).

- **O-36:** "Agree. Order is a tuple only." `order=(3,)`; `order=()` for no jets; an integer raises. This withdraws
  the integer shorthand Claude had proposed at 14:00 (an integer `0` would have meant `()` while every other `n`
  meant `(n,)`).
- **O-37:** the two helpers are exported from the top level `t3toolbox` as well as from `t3toolbox.fitting`.
- **O-38:** for two or more jet variables, Part 1 runs a loop under both `over_orders` settings, so the compiled
  program grows with the order there until Part 2.
- **O-39:** withdrawn before it reached him: rows 2.1 and 2.2 already decide that the slots of an input jet are
  Taylor coefficients.
- **O-40:** `rescale_direction` takes one length (the default form), or one per jet variable.
- **O-41:** `index_to_one_hot` returns the default float dtype, with no `dtype` argument.
- **O-42, the display for two or more jet variables:** the columns are the multi-indices in the order "you would
  get if [you] unfolded the tensor into a matrix" (row-major); no columns by total degree. And, his addition: "the
  table is likely to be too wide in all but the smallest cases. Therefore we should add an option for the user to
  turn off table display while retaining the rest of the display diagnostics." The name of that option is to be
  proposed to him.
- **O-43:** "The ambient transposes stay without jets. However, there is no plan to add jetting to ambient
  transposes in the future." So it is not a deferred item: the ledger does not list it as work to come.
- **O-44, the default of `order`:** "Go with recommendation (b)": no default on `LeastSquaresOracle`; `()` stays the
  default on the sampling functions and methods, whose docstrings show the trap (an input jet read with
  `order=()` has its jet axis taken for a sample axis) as a wrong-against-right example.
- **O-46, the entries oracle:** "accept order=() for entries." One class, one rule for every kind: `order` is
  required, and for `'entries'` it must be `()`. No sentinel default.
- **O-47:** the dual-jet and derivative functions accept `'entries'` at `()` and return the data unchanged.
- **O-45, the jet axes on the uniform layout** (16:26, after the question was put again): "Agree with proposal.
  Every slot along a jet axis is data." His picture of uniform data stands for the padding of the mode sizes (the
  user supplies a raw array; nonzeros in the padded places have no effect, because the cores are masked; helpers
  pad ragged data). The slots of a jet axis are another thing: one packed array gives every mode the same jet
  sizes, so a mode whose input has fewer slots gets more, and no mask of the cores covers them. Those slots are
  coefficients. A zero there, written by the packing helper, is the coefficient; a nonzero is another valid input.
  No mask on the jet axes and no checker. The uniform equivalence contract gets a sentence saying so.
- **The names** (16:26): "Agree with proposed names." So, from section 11.4 of the plan: `GroupLengths.J`; `order`
  as the argument of `jet_contract` for the output's jet shape; `affine_jet(ww, *directions)`;
  `index_to_one_hot(index, shape)`, both in `backend/jets.py`; the marker `PART 2 (row 5.4, jets in several
  variables)`; `F` for the full jet sizes in shape comments; the terms "jet shape" (the tuple of maximum orders named
  `order`) and "jet sizes" (the sizes of an array's jet axes); `len_J=0` on `pack_vectors` and `pack_if_ragged`.
  Claude reads the same answer as covering `error_table=True`, the keyword proposed to him in the same message for
  turning the display's table off (beside `verbose` on `newton_cg`, and on `make_newton_display`); to be confirmed
  when he reviews the plan. **Confirmed 2026-11-19**, on all three places (the frontend `newton_cg`,
  `make_newton_display`, `format_newton_iter`).
- **What happens next** (16:26): "Once the plan is complete, please commit it, update the handoff, and pause before
  implementation. ... We will resume in a couple days ... Also, this will give me time to review the updated plan
  more carefully." So the plan change is complete as a text but not signed off as a whole; P4a does not start
  until he says so. **Signed off 2026-11-19** (the next note).

**The changed plan signed off; how the implementation continues** (the maintainer, 2026-11-19, in Claude Code on
his laptop; his message came before 10:56, the first clock reading after it).

- **Sign-off:** "Part 1 of the plan is confirmed as a whole. But we may encounter issues along the way which require
  adjusting the plan as we go." So `PART1_PLAN.md` at `48a1af4` is the plan, P4a may start, and a change found
  necessary while building is made in the plan and recorded, not worked around.
- **Three planned pauses**, at each of which the work stops for a check-in and a discussion of what happened and
  what arose: after the jet sub-arc P4a to P4d; after the engine sub-arc P5 to P11; after P15, just before the
  switch.
- **Decisions on the way:** "For small issues or things which are clear, please make the decisions and log them.
  If there are big unforeseen issues or situations where the right decision is uncertain, check in and we can
  talk through it." And: "Please think from first principles about what we are trying to do here, and use
  judgement rather than blindly following rules. Please go slow and do things right rather than cutting corners
  for expediency." (The rule of 2026-11-16 stands beside it: a departure from a decided row, a public name or the
  order of the slices is raised with him.)
- **The agents:** two Opus 5.5 agents on high effort, an implementer and a reviewer, **kept from slice to slice**
  ("Keep the same agents going so they have more context"). The implementer writes a slice, the reviewer reviews
  it, they iterate until the reviewer signs off and send it to Claude, who commits it or sends it back; then the
  same pair starts the next slice. This replaces "two new agents, Fable 5.2 on medium effort" of 2026-11-16, 20:51.
- **Required reading** for Claude and every agent: "pick a handful of methods from the flagship class
  TuckerTensorTrain in tucker_tensor_train.py, read those methods and the traced functions that they call in the
  backend. This can give you a sense of the style and conventions of the library."
- **CI on the branch:** the workflows run only on pushes and pull requests to `main`, so nothing pushed to
  `graded-messages` had run in CI, and P1 to P4 had never run on the floor (py3.9, numpy 1.x, jax 0.4.30). Asked to
  choose, he chose a draft pull request: #26, `graded-messages` into `main`, opened 2026-11-19, marked not to be
  merged before Parts 1 and 2 are done. Both test legs, the examples and the docs build now run on every push.
- **N42 confirmed:** `error_table=True`, on the frontend `newton_cg` beside `verbose`, on `make_newton_display` and
  on `format_newton_iter`.
- **The amendments owed for the jets were made the same day** (`reviews/jet_group_plan/ACCOUNT.md` section 7):
  rows 2.1, 2.3 to 2.9, 3.4, 5.4, 6.2, 6.6 and 9.4; the notes on N16, O-10, O-27 and O-28, the group `A`, and jets in
  several variables as a future feature; the section "Carried to Part 2" (items 1, 4 to 6); and sections 4, 6 and 7
  of `SAMPLE_VARIABLES.md`. Each is an amendment beside the old text, which is kept.

**The first planned pause, after P4a to P4d: the maintainer's answers** (2026-11-20; the report of 2026-11-19, 19:30,
`implementation/LOG.md`).

- **Claude's minor decisions of the sub-arc, reviewed:** agreed with (1) the carrier counted after clipping, a
  string with no `J` raising at every `m`, an input term of `J` alone raising; (3) `index_to_one_hot` following numpy
  indexing (a numpy index out of range raises, a jax one gives zeros); (4) `rescale_direction` raising for entries.
  On (2), that a plain integer `order` raises, he asked what it guards against; Claude answered (it is his O-36):
  `0` is ambiguous between `()` and `(0,)`, which differ in rank, so any integer shorthand makes a natural call
  mean something other than its author meant; every ported call site passes an integer with the old meaning, and a
  misread jet shape shifts which axes are read as jet axes (row 6.2: never inferred), a silent misreading where a
  raise is loud; one canonical form keys the caches. A shorthand can be added later compatibly; removing one could
  not. **Confirmed** (2026-11-20): "having plain integers order raise per O-36 is correct and makes sense."
- **The contraction order of an einsum of many operands is Part 2's** ("This is an issue for the performance part
  of the refactor, Part 2 ... For now, what you have done is good"): item 7 of "Carried to Part 2" below.
- **A `conftest.py` guard that every `subTest` label is a builtin: approved.**
- **The test cut** (the four-site half of `TestJetsTwoWays`): the tests stay; the suggestion, its reasoning, its
  consequences and the measured savings are recorded in `dev/HANDOFF.md` (backlog, "A test cut held in reserve").
- **The same pair of agents continues** ("Please keep the same agent pair if you can"), on P5. "Good work so far.
  Please continue on P5."

**"The same value" means equal to within rounding** (the maintainer, 2026-11-20, during P6's design: "Not changing
means not changing to within numerical rounding. We do not require bitwise identical results."). Two algorithms that
give the same result to within rounding are both acceptable candidates, provided they are roughly equally
numerically stable: changing the order of a contraction or of a sum is acceptable; removing an orthogonalization is
typically not, because it changes stability. Bitwise equivalence is given low weight when options are weighed;
simplicity and performance decide. This is how rows 6.4 ("wanting a variable never changes a value") and 2.3 step 3
are read, and every comparison of two routes is a tolerance test (row 0.8).

**The second planned pause, after P5 to P11: the maintainer's answers** (2026-11-20, evening).

- **`sum_over_samples`** (discussed at his request): aligned with his intent. The flag touches only the outputs shaped
  like cores (`g`, `JtJp`, `Kp`); `z`, `r`, `Jp` and the edge variables are always per sample; `f`, `r_block_sumsq`,
  `Jp_norm_sq` always summed. `W` keeps its place, multi-axis, in `W + K + C`, the site axis first on uniform. **The
  oracle does not ask for unsummed blocks:** wait for a use case; his lean even then is to add hooks another way
  than through the oracle, decided on the facts at that time. The low-rank structure of per-sample blocks (and its
  link to the ambient transposes) is filed in `dev/HANDOFF.md`'s backlog.
- **Corewise is a first-class feature** ("a feature people expect, and it should work well rather than be an
  afterthought"). Approved: a corewise forward derivative (P12), frontend methods for it and for the corewise Hessian
  (P13, reversing "the corewise Hessian has no frontend method", on the rule that each user can do everything in
  their own layer), a tested bridge to external optimizers (P17/P19); the check of `t3_corewise_frame` in P20, **with a
  check-in with him between P19 and P20**. A writer and a critic update the plan; the names go to him.
- **Re-gauging for corewise optimization:** agreed with Claude's recommendation (a re-gauging option with a period and
  two modes, `'orthogonalize'` optionally followed by spreading the norm evenly, and `'balance'`, a scalar per core that
  carries optimizer state exactly; defaults chosen on real problems). For after the refactor: `dev/HANDOFF.md`'s backlog.
- **After Parts 1 and 2, before the release, the backlog is considered**; a Part 3 may be made from it.
- **The zero tilde names on apply and entries** (`xi_tilde`, `mu_tilde`, `xi_tilde_dot`, `mu_tilde_dot`): unknown names,
  their message saying they are zero there ("I'm ambivalent ... let's go with your recommendation").
- **A public reader of a call's names:** not now; maybe later if a need arises.
- **O-49: a frontend corewise direction is the pair of core tuples only, never a `TuckerTensorTrain`** ("Although the
  arrays are the same, the meaning is different."). **O-50: the helper `point_value_and_gradient(geom, oracle, batch,
  x_cores, regularizer=None)` is adopted**, with its frontend twin (P17). **O-48: `_corewise_tangent`** (the maintainer, after
  a signature prototype: `_jacobian` could be read as the map with free core-direction indices; the function probes a
  tangent vector of the corewise geometry). `t3_<kind>_corewise_tangent`, `ut3_...`, frontend `<kind>_corewise_tangent`;
  it fits the documented `_tangent` stem (a directional derivative) and makes the trio tangent / transpose / hessian.
  P19 gives the naming doc's `_tangent` entry the new functions as examples (its old ones go in P20).
- **The corewise call takes the direction as `core_directions` only; `variations` raises** (P8; "agree, keep
  core_directions only for corewise and refuse variations"). The least-squares oracle passes the direction as
  `core_directions` on a corewise frame, told from the frame's structure.
- **The structural stack checks** (P7's `K`, P10's `W` and `C`, P11's index rows): kept ("Yes, keep the checks.
  Strongly agree."). O-18's "the engine checks what it needs to run, and nothing more" is read to include: when the
  engine reads a group's size from one operand, it checks that every other operand carrying the group agrees, since
  a size-1 axis would otherwise broadcast into a silent wrong answer.

**Old positional calls to the probe transposes: documented, not guarded** (the maintainer, 2026-11-21, on finding A of
P13's design review). `x.probe_corewise_transpose(ztildes, ww)` and `probe_ambient_transpose(ztildes, ww)` in today's
order would silently swap the residual and the sample (same shapes for probe). Claude had recommended making
`residual` keyword-only; the maintainer declined: it compromises the design for (anti-)compatibility, while the
refactor does things right from first principles and accepts breaking old code. Instead: doctests showing the right
call (the adjoint relation with the forward at the intended sample holds to tight tolerance) and the wrong one (it
does not), and the CHANGELOG upgrade note. No safe-mode adjoint check: besides its cost, it cannot detect the swap (a
swapped call is a valid transpose of a different problem, whose own adjoint relation holds).

**The hold list: `computed_names`, not `keep=`** (the maintainer, 2026-11-21, at the third planned pause). P14 had
given the engine a `keep=` keyword returning, besides `want`, a record of the hold-list names the call computed
anyway. Dropped: it makes the engine's return depend on a keyword and its contents on the routing, so the output is
not predictable from the inputs. `keep=` and `want`+`held` have the same computational reach (a route is chosen from
the call's facts, never from `want`, so wanting a name the call computes anyway adds no work); its only value is not
having to know the graph. That value is kept, without the cost, by a pure query `computed_names(kind, frame, want,
**arguments)`: the names a call with these arguments would compute, read from the declared graph by the engine's own
cached closure. The engine returns exactly `want` and reuses `held`, nothing else. The least-squares oracle passes the
same arguments to both, adds to `want` the hold-list names `computed_names` says the call computes anyway, splits them
off, and builds its `HeldVariables` itself. A one-call helper on top of it (a split return of the free extras) was
considered and is not built; if oracle writers ask for it later, it is a helper on `computed_names`, never an engine
keyword. The maintainer's reading of the concept: `want` is need-to-have (changes the computation), the hold list is
nice-to-have (returned only when free).

**`frame_form` kept; `frame_shape` added** (the maintainer, 2026-11-21, at the third planned pause). The engine and the
oracle keep inferring the call type from the frame's structure (plan 2.2), through the one public reader
`frame_form(frame) -> (layout, call)`, two strings. Beside it, a small public `frame_shape(frame)`: the mode sizes for
any of the four forms (the uniform ones read the static `shape` element by position, which is as easy to get wrong as
the count of four). Built in P14b.

**The notes carried to P16, settled** (the maintainer, 2026-11-21, at the third planned pause).
1. **`held` is valid for one frame and one batch.** No runtime guard (the record cannot carry the identity of its
   batch or frame under jit); the guard is the documented contract plus a P16 test that a mutant carrying `held`
   across minibatch draws (MC-SGD, Adam, minibatched Newton) fails.
2. **`f_resolution` and the regularizer.** The oracle's `f_resolution` covers the misfit only; P16's optimizer adds
   `eps |rho|` to reproduce today's floor (`objective_floor`, the switch's rounding gate). **The regularizer is not
   scaled** under minibatching (corrected 2026-11-21 by Claude, from the P16 reviewer's orientation; an earlier wording
   here, "the library's minibatch rule", was wrong). Under O-21 a drawn batch's misfit is scaled by `n/b` so that the
   minibatch objective is an unbiased estimator of the full one, `(n/b) misfit_b + rho`; the regularizer is a term of
   the full objective, not a sum over samples, so it stays as it is. Today's rule (data unscaled, `lambda` scaled by
   `b/n`) gives the same objective times `b/n`: the MC-SGD Cauchy step is unchanged, Adam differs only through its
   `eps`, and reported objective values change by that factor. P17's `point_value_and_gradient` follows the same rule.
3. **The aligned padded frame returns `(frame, centers)`,** like `frame(x)`. This reverses Claude's P15 ruling
   (frame only). The padded frame keeps the old point's frame bitwise in its leading slots and the old point has zero
   coordinates in the new slots, so the padded point's centers are the old point's centers zero-padded (checked:
   consistency residual 5.7e-15 at the aligned frame of `padded_start`'s docstring example); at the cone start
   (old point = the zero tensor) they are zeros. `aligned_padded_frame` already computes `frame(x_old)` and keeps its
   centers. Ragged and uniform. Every frame the optimizer meets then carries its centers; a P16 test asserts the
   consistency residual at a padded start, with and without alignment passes.
4. **`pack_sample(kind, order, sample, N)` and `pack_data(kind, order, y, N)`** replace the old-kind-name forms in P16.

**`UniformTuckerTensorTrain.from_t3` raises for a width below a rank** (the maintainer, 2026-11-21; found by the P16
implementer, confirmed on released `main`). Today `from_t3(x, n=...)` and `r=...` below the largest Tucker or TT rank
silently truncate by slicing (8.8% and 62% relative error in Claude's check), while `N` below a mode size already
raises. Decided: `from_t3` is a pure change of representation (the uniform equivalence contract) and raises a
`ValueError` for `n` or `r` below a rank, as for `N`, with a message pointing to `TuckerTensorTrain.resize` (truncation
by slicing, a deliberate operation the library keeps) or `t3svd`. The maintainer's view: truncation when "padding" is a
valid ask, but it belongs in `resize`, not in a conversion. At the backend conversion too, so raw-`.data` users get
it; `from_t3weights` checked for the same gap; P15's uniform frame-centers test (its shared case) fixed to use widths at
least the ranks; a CHANGELOG line. A small slice after P16, on `graded-messages`.

**P16's D1 and D7** (the maintainer, 2026-11-21). **D1:** a required `Geometry` member `point_data(x_cores)`: the
point's full `.data` from its bare cores (ragged: the cores unchanged; uniform: `(B_sc, G_sc, shape, masks)`, the
geometry's own shape and point masks), so the backend optimizers can sample a trial point without importing T3
classes; used by `point_objective` and P17's `point_value_and_gradient`. **D7:** the top level exports
`LeastSquaresOracle`, `LeastSquaresBatch` and `local_model` in place of the six model factories.

**The data metric: decisions 1 and 2 of the sampling audit** (the maintainer, 2026-11-21, at the pause after P16;
`SAMPLING_AUDIT.md`). **1.** The transposes' (and the second-order transposes') data argument is an element of `Y`
(under the maintainer's typing `J: X -> Y'`, `J^T: Y -> X'`, `W: Y' -> Y`), paired with `z` slot by slot, with no
weight inside the objective-free maps; it is renamed from `residual` and documented so; a first-class public
function applies `W` (`Y' -> Y`) in the fitting layer, the weight being the objective's. **2.** `W` is an elementwise
diagonal weight broadcastable to the shape of the data `y` (per sample, per output entry, per jet slot); the current
block forms (per mode and order) are broadcast cases. A full operator (a covariance) is out of scope. Decisions 3
(corewise as point methods or as tangent operations of `COREWISE`) and 4 (the names: the Hessians, `jet_to_dual_jet`)
are still with the maintainer. A sub-arc plans and builds 1 and 2 before P17; the maintainer reads the plan first.

## Carried to Part 2 from the implementation of Part 1

**The measured slowdown of Part 1, and its cause** (2026-11-21, at the third planned pause; the maintainer asked
whether it is architectural). It is not. Against the old derivative kernels (numpy): at the example's size the engine
is about 2x slower from per-call Python overhead; at scale its forwards are 2-6x slower, growing with the jet size and
the rank, because `jet_contract`'s step `cda,aib,di->cdb` contracts the carried jet with the core first (the row 5.5
order, `TestOrderInsideAStep`). With only that order fixed (the cheapest first pair) the engine is 2-3x FASTER than the
old kernels on every operation at every large size measured, apply and probe. Part 2 starts from these numbers and
scripts: `dev/graded_messages/implementation/bench/` (they need the old kernels, deleted in P20; run them before it).
The maintainer's criterion: overhead on microbenchmarks is acceptable; a limit at scale is not.

Kept here so that nothing found while building Part 1 is forgotten when Part 2 is planned. Each item names its
row and where it is marked in the code. A search for `PART 2 (row` in `t3toolbox/` and `tests/` finds the markers.

1. **The contraction order inside one step of `jet_contract`** (row 5.5). Found on 2026-11-17 while P1 was in
   review; the maintainer had asked whether the scan was repeating the extra contractions of the first prototype.
   - A step of a product is one three-operand `contract` call: the carried jet with all its slots, the operands
     with no order group, one coefficient of the partner. The cheap order contracts the coefficient with the
     no-order operands first (the expensive contraction, once) and then against all slots of the carried jet. That
     is how the library at 2026.5.0 writes it by hand, as two pairwise contractions.
   - On jax, `jnp.einsum`'s default optimizer picks that order. On numpy, `contract` forces `_pairwise_path`,
     which pairs the operands that share the most indices; here that is a three-way tie, the first pair wins, and
     the carried jet is contracted with the core first: the expensive contraction on every slot at every step.
     Measured on a representative step (`W = 2000`, TT ranks 24, Tucker rank 12, order 4): 7.2e7 against 2.0e7
     flops, a factor that grows with the order. It holds for 16 of the 20 product strings, affine ones in the
     sweeps among them; in the four `nu` strings with an affine leaf the core and the coefficient happen to be
     the first pair, which is the cheap order (the implementer's count, P1). jax's path depends on the sizes,
     so on the later steps of the loop form, where the carried slice is short, it can pair differently. The
     readout step folds its two jets first on both backends.
   - The numbers computed are the same. Only numpy is slower, and Part 1 claims nothing about speed.
   - **Decided by the maintainer (01:51): deferred to Part 2**, "clearly marked so we don't forget". It is
     contained in `jet_contract` (or in `contract`'s choice of path) and needs no change to any call: the fix is
     either internal, or the per-term order of 5.5 arriving as one more optional argument. `over_orders` is not
     the handle for it: that keyword says how a full partner's coefficients are taken, not in what order one
     step's operands are contracted.
   - **Markers:** a comment `PART 2 (row 5.5)` at the product step in `backend/contractions.py`; in
     `tests/test_jet_contract.py` a test marked as an expected failure, which asserts that numpy does not
     contract the carried jet with the core first. It is reported in every run, and it turns into a failure the
     day the order is fixed, so it has to be dealt with then.
   - **Restated for the one jet group (2026-11-19; P4a):** the 41 strings become 29 and the product strings 12
     (`mu`, `nu`, `eta`, 4 each). The expected failure is restated on them with the library's operand shapes, 20
     cases as in P1. The plan's recount (`PART1_PLAN.md`, P4a): at orders 2 and above 16 of the 20 contract the
     carried jet with the core first, as in P1; at order 1, where a two-slot leaf ties with a full jet of two slots
     and the leaf of the `nu` strings becomes the carrier, 20 of 20. P4a's implementer confirms the recount.
2. **Map-type operations on the uniform layout** (row 5.3). Part 1 maps one site at a time through `xmap` (the
   answer to O-19, "must be revisited in Part 2"). There is no site group: if Part 2 does these at all sites at
   once, it is uniform code of its own with the `d` axis (the maintainer, 2026-11-17, 02:23).
3. **The benchmark** (the answers to O-1 and O-2). Part 1 has none. Part 2 compares the branch with the released
   2026.5.0, on CPU and on a GPU pod, and measures the order-0 condition of 2.7.
4. **A scan over several jet axes** (rows 2.9 and 5.4; O-38; added 2026-11-19). For `m >= 2` jet variables
   `jet_contract` loops over the partners' multi-indices under both `over_orders` settings, so the compiled
   program grows with the box there, and row 2.9's one program size is given up for `m >= 2` until Part 2. The
   faster algorithms (the order in which a product's multi-indices are taken, a scan over more than one jet
   axis, transforms) live inside `jet_contract`, behind its signature. **Marker:** `PART 2 (row 5.4, jets in several
   variables)` in `backend/contractions.py`, at the place where `m >= 2` takes the loop (P4a).
5. **The zero slots of an input affine in several variables** (added 2026-11-19; Claude to the maintainer,
   2026-11-17, 14:20). `affine_jet(ww, p_1, ..., p_m)` has a box of `2^m` slots holding `m + 1` nonzero
   coefficients, and the simple algorithm of Part 1 visits every slot of a partner. A sparse form of such an
   input, if it matters, is Part 2's.
6. **Data measured to a total degree** (row 2.5; entry 116; added 2026-11-19). Part 1 has the box only: such data
   are stored in the box with weight zero on the slots not measured. A cheaper storage, if wanted, is Part 2's.
7. **The contraction order of an einsum of many operands** (rows 5.5 and 2.9; the maintainer, 2026-11-20; added
   2026-11-20). `jnp.einsum`'s default `optimize` differs between jax generations: `'auto'` on jax 0.10, `'optimal'`
   on jax 0.4.30 (CI's floor), an exhaustive path search exponential in the number of operands. The reference module
   contracts a whole network in one einsum (`sampling_definitions._einsum`), which hung CI's floor leg; Part 1 passes
   `optimize='greedy'` there (P4b), and the library's two- and three-operand contractions keep the house convention
   of no `optimize` on jax. The general question, how the order of a many-operand contraction is chosen and whether
   the house convention needs a caveat in `CLAUDE.md`, is Part 2's. **Marker:** the docstring of
   `sampling_definitions._einsum`.

## What remains, in groups (2026-11-15, 15:34)

Of the 57 rows in sections 1 to 9 (9.7 was added on 2026-11-16), 46 are decided, 3 are guidance and 1 is parked. The other 7 are below, in
seven groups that cut across the sections. The grouping and the suggested order are Claude's, made at the
maintainer's request; they decide nothing.

| Group | Rows | Open | Reopened | To confirm | Held |
| --- | --- | --- | --- | --- | --- |
| A. Jets and the misfit | 0 | | | | |
| B. The frame and the value | 0 | | | | |
| C. What a kind is, and the public surface | 0 | | | | |
| D. One program for every layout and stack | 2 | 1 | 1 | | |
| E. Plans, memory and scale | 5 | 3 | 2 | | |
| F. The fitting layer above | 0 | | | | |
| G. What defines correct | 0 | | | | |
| Total | 7 | 4 | 3 | | |

- **A. Jets and the misfit.** (Decided on the afternoon of 2026-11-15: 2.4, what a tilde jet is; 2.5, the misfit;
  2.3, the derivative-mode adapter; and, at 19:32, 2.9, how a term is carried out over the orders.)
  (And on 2026-11-16: at 01:04, 3.6, the weight inside the engine; at 01:32, 3.7, what happens to the sample
  stack on each output; at 01:36, 3.3, which adjoint sweep a tilde name refers to.) Group A is finished.
- **B. The frame and the value.** Finished on 2026-11-16: at 01:52, 4.1, what a frame is, with the centers' part
  of 4.2; at 01:59, 4.3, the route to the value at a trial point; at 02:07, the rest of 4.2, frame consistency;
  at 02:12, 4.4, tails and boundaries; at 02:16, 3.2, `nu_L` stays, and 4.5, the projection, the lift and the
  Hessian assembly stay in the geometry.
- **C. What a kind is, and the public surface.** Finished on 2026-11-16 (3.1 had been decided with the notation
  on 2026-11-15 at 23:23): at 02:22, 3.4, a kind is closed in the sampling layer and the open extension point is
  in the fitting layer; at 02:31, 6.5, a held variable is trusted and only its shape is checked; at 02:34, 6.2,
  the arguments of `sampling_variables`, up to three names; at 02:37, 6.4, the table of variables written out and
  tested against the declared graph; at 02:40, 6.3, the public functions of the sampling layer; at 02:45 and
  02:52, 8.2, the oracle; at 02:56, 9.2, the kind strings and the unit name `'sweep'`, with the sizing function
  names moved to 7.6.
- **D. One program for every layout and stack.** (Decided on 2026-11-16: at 03:11, 5.2, one set of step
  functions for both layouts; at 03:32, 5.6, the dtype rule; at 16:26, 5.4, a literal contraction string at
  every call site, through `jet_contract`.) 5.5 the contraction order inside a term (open: it goes into the
  plan; its granularity, the form of the field and the helpers are held for the performance discussion of
  group E; how an order reads with jets is proposed); 5.3 sites scanned or vectorized (reopened; with the
  performance discussion). The maintainer gave rules for how the functions are written on 2026-11-15, 19:50
  (notes to section 5).
- **E. Plans, memory and scale.** 7.3 where the plan lives (reopened); 7.2 the defaults (reopened; with them,
  from 8.3, which phases are compiled by default and which minibatch draws the library ships); 7.5 inert
  padded samples for entries (open); 7.6 automatic sizing, with the names of the sizing functions and of the pad-and-reshape helpers (open); 7.7
  sharding and device placement (open). The maintainer wants the sizing names taken up "in the broader context
  of performance optimization hooks and defaults" (02:56).
- **F. The fitting layer above.** Finished on 2026-11-16: at 04:46, 8.1, what `LocalModel` is; from 13:51 to
  14:30, 8.3, the optimizers' contract, in three parts; at 15:12, 8.4, what the display and the diagnostics
  read, with derivative mode moved out of the fitting layer (2.2 and 2.3 amended); at 15:44, the oracle as a
  description with the data as an argument, which rewords 8.1 to 8.4.
- **G. What defines correct.** Finished on 2026-11-16 at 15:58: 9.3, correct is defined by the definitions.

**How the groups depend on each other.** A stands alone and continues the morning's work. B feeds C: 6.2 and 6.4
wait on 4.1 and 4.5. C feeds F: 8.1 and 8.3 need the public surface. D and E are each mostly self-contained; E's
7.3 is tied to F's 8.1. G bears on how every other answer is checked.


**Outside the list:** 8.5 shared sample vectors (parked); weights on internal edges as inner products (parked for
another arc); review finding F17 (never discussed; listed under "After the target"); forward-only probing as a
setting of the probe kind (left off this arc on 2026-11-16, 02:28); a library-wide audit of functions that
require squashed tails (left for later on 2026-11-16, 02:16).

## How to read the status column

| Status | Meaning |
| --- | --- |
| Decided | The maintainer decided this on principle |
| To confirm | Decided on 2026-11-14 for a reason of principle, but quickly. Confirm or reopen |
| Reopened | Decided on 2026-11-14, but the stated reason was compatibility, cost of change, or "as the library does today". That reason no longer counts |
| Open | Not decided |
| Assessed | An assessment was asked for and is done (see the notes). The decision is still to be taken |
| Parked | Set aside by the maintainer |
| Guidance | The maintainer answered with principles to guide judgement, not hard rules |

"Decision n" refers to `REVIEW_DECISIONS.md`. Sections 1, 2 and 4 are the roots: most of the rest depends on them.

## 0. Ground rules

These are the maintainer's, and they are the criteria for every decision below.

| # | Rule | Status |
| --- | --- | --- |
| 0.1 | Compatibility with today's behaviour is not a criterion. This is a breaking change with a wide blast radius, and that is accepted | Decided |
| 0.2 | The target is decided before the sequencing | Decided |
| 0.3 | Design as if writing the library today, knowing what is known now | Decided |
| 0.4 | The end state must let a consumer (T3Polynomial first) use every addition without rewriting the optimizers and without changing T3Toolbox | Decided |
| 0.5 | Code is readable statically: immutable data of basic types, no containers whose contents must be inspected, each array its own argument | Decided |
| 0.6 | The framework gives users control of performance choices. The target scale is very large problems sharded across many GPUs. Small-scale CPU timings do not decide anything | Decided |
| 0.7 | Keep it simple, and avoid state | Decided |
| 0.8 | Tolerance tests for numerical operations, exact tests for structural things | Decided |

## 1. Scope

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 1.1 | Which layers does the target redesign: the sampling kernels only, or also the fitting layer (sampling kinds, `Problem`, `LocalModel`), the geometry interface (frame, projection, lift), and the optimizers' contract with a model? | Guidance | |
| 1.2 | Which things are fixed inputs to the design, not open to change: the ragged and uniform layouts, the numpy and jax backends, the `W`, `K` and `C` stacks, the frame's four core families, Tucker-factor sharing as a geometry wrapper? | Guidance | |
| 1.3 | How do the criteria rank when they conflict: clarity for the reader, control of memory, throughput at large scale, latency on small problems, compile time, float32 accuracy, size of the public surface? | Guidance | |

The maintainer's answers (2026-11-15). In his words they are "not hard answers but more principles to help guide
judgement".

- **1.1.** Redesign may be considered for everything downstream of sampling: the optimization and fitting layers,
  and the frontend and backend sampling functions. Things unrelated to sampling are off limits, such as the
  T3-SVD, orthogonalization, or adding two T3s. The data structures of the flagship basic types are off limits:
  T3s, frames, tangent vectors, and their uniform twins.
- **1.2.** All of these are fixed inputs: the ragged and uniform layouts, the numpy and jax backends, the `W`, `K`
  and `C` stacks, the frame's four core families, and Tucker-factor sharing. The one near-exception is the way
  sharing is handled. Sharing as a feature must be preserved in full. Another way of handling it may be
  considered only if it loses no functionality and no performance.
- **1.3.** There is no hard ranking. When criteria conflict, each case gets thoughtful consideration.

What this settles elsewhere in the list:

- **3.2 and 4.1.** The frame keeps its four core families and its data structure. So a different frame convention
  is not an option, and a frame cannot be given the center cores. What remains of 4.1 is which family is
  authoritative for the base point, and whether anything travels beside the frame.
- **5.2 and 5.4.** Both layouts and all three stacks must be served.
- **6.3.** The existing public sampling functions, frontend and backend, are in scope, so none of their names has
  to remain.
- **Section 8.** The fitting layer and the optimizers are in scope.
- **8.5.** Sharing stays as it is unless an alternative loses nothing.

The geometry boundary (the maintainer, 2026-11-15):

- **Off limits:** basic frame construction, projection, lift and retraction.
- **On the table in principle:** the geometry singleton pattern that organizes those functions for optimization
  and fitting. He is not sure it can be improved. Simpler ways he tried ran into chains of mutual contradiction:
  some forced recompiling too often, some did not respect the frontend and backend split, some could not support
  numpy and jax seamlessly.
- **To preserve, even if the geometry changes:** what concerns a specific tangent vector lives in the tangent
  vector class, and what concerns the geometry in general lives in the geometry.

What that settles: "off limits" means the signature and behaviour of the existing functions. New functions may be
added beside them. So fusing retraction with frame construction is allowed as an additive feature, and every
route in 4.3 stays open. For 4.5 nothing is proposed for the projection or the lift themselves; the question
there is only which layer calls them. Any change to the organization has to pass the three tests his earlier
attempts failed.

## 2. Jets

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 2.1 | Taylor coefficients replace derivatives everywhere, and jet edge variables are first-class, like the plain ones **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** where a later text says "the order axis", read "the jet axes": an array that carries jets has `m = len(order)` leading jet axes, one per jet variable. | Decided | |
| 2.2 | Taylor coefficients are the only form in the sampling, fitting and optimization layers. Derivative mode is not a mode of the library: a user with derivative data prepares it before fitting, with standalone conversions (amended on 2026-11-16; it had been an adapter at the outer level of fitting) | Decided | 2.1 |
| 2.3 | The conversions for derivative data are standalone functions and nothing else: derivatives to Taylor coefficients (slot `t` divided by `t!`), the reverse, a derivative-mode weight to a Taylor weight (slot `t` multiplied by `t!`), and one function that prepares the data, the validation data and the weight together and returns the weight `t!` when none is given. No fitting function, oracle, display or sampling method has a keyword for it, and nothing is converted on the way out. The functions are documented with a doctest (amended on 2026-11-16: the opt-in keyword on the fitting functions is removed) **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** with the jet shape `order = (n_1, ..., n_m)` the conversions divide (multiply) slot `k` by `k_1! ... k_m!`, and the preparing function returns `k_1! ... k_m!` per slot when no weight is given. The entries kind has no derivative data: at `order = ()` these functions return its data and weight unchanged (O-47). | Decided | 2.2, 6.7 |
| 2.4 | A tilde jet is an adjoint variable: a Lagrange multiplier, a dual object. The data and the residual are of the type of `z`, in forward order. The misfit is the weighted sum of squares slot by slot. The dualization is one step at the seed: the weights, then the turn of the order leg. Tilde variables are stored turned, the ones a user holds included. `w_tilde` names the seed **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** the turn is a flip of every jet axis. The pairing of a jet with a dual jet is the corner coefficient `s_1^n_1 ... s_m^n_m` of their product (what the older texts call the top coefficient). The truncation is the box `F = (n_1 + 1, ..., n_m + 1)`, never total degree (entry 116). | Decided | 2.1 |
| 2.5 | The misfit is `1/2 sum omega^2 r^2` over modes, orders and samples, on Taylor coefficient residuals. With no weight given the weight is ones; that default may change in a later release if a more robust method that "just works" is found. The weight has the shape `(d,) + T` for probe and `T` for apply and entries, with no rule for a bare vector **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** the weight is `(d,) + F` for probe, `F` for apply and `()` for entries, `F` the full jet sizes. Data measured only up to a total degree are given in the box, with weight zero on the slots not measured (entry 116). | Decided | 2.1, 2.2 |
| 2.6 | Is an input jet always affine (`w + s p`), or can it be general? This decides whether the two-term banded form is part of the structure **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** an input jet is general (entries 118, 119): `ww[i]` has jet axes of sizes of its own, `J_in_i + W + (N_i,)`, its slots Taylor coefficients, and there is no `pp`. The line `w + s p` is the common case, built by `affine_jet(ww, pp)`. The two-term banded form is no part of the structure: it is a partner of two slots. | Decided | |
| 2.7 | Is order 0 the one-coefficient case of the same functions, with no order axis ever created? **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** order 0 is `order = ()`, `m = 0`, and no jet axis is ever created. `order` is a tuple only; an integer raises (O-36). | Decided | 2.1, 2.6 |
| 2.8 | There are three sampling kinds, `probe`, `apply` and `entries`, each with an `order` (0 for the plain computation), in place of six. The fitting layer has the same shape as the frontend methods of 6.6 **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** probe and apply each take a jet shape `order` (a tuple; `()` is the plain computation). The entries kind has no jets (entry 120): its functions take an `index` and no `order`, and its least-squares oracle takes `order = ()`, required as for every kind (O-44, O-46). | Decided | 2.7, 6.6 |
| 2.9 | A term over the orders is the truncated product of its operands' jets. One operand carries the order axis `T` through the contraction; the other is taken one coefficient at a time: two contractions when it is affine. No dense order-by-order form. Where both operands are full jets there are two forms, chosen by the user through the plan (7.1): a scan over one operand's coefficients with the other jet carried and shifted one slot per step, which is the default; and a loop with slices. The block readouts compute only the terms whose orders sum to the top **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** the jet group is `J`, of `m` axes, in place of `T` and `A`; "affine" reads "a partner with fewer slots than the box". Which operand is carried and which are taken one coefficient at a time is read off the operands' static jet sizes. A readout computes the corner coefficient. The two forms differ only where a full partner is taken one coefficient at a time at `m = 1`; for `m >= 2` both loop in Part 1 (O-38), and a scan over several jet axes is Part 2's. | Decided | 2.4, 2.6, 7.1 |

Notes:

- **2.7, the maintainer's conditions.** Do it only if order 0 loses no performance. His first jet implementation
  tried this and could not: there were extra contractions. That was with the dense `trs` tensors. The order axis
  must not be created and then squeezed; the preferred form never creates it when it is absent, perhaps through
  the contraction engine as a missing group.
- **2.7, one fact.** `contract` already defines an uppercase letter as a group of zero or more axes, and an empty
  group vanishes. So an absent order group vanishes in a term that is linear in one jet. The products of two jets
  are where the assessment has to look.
- **2.6, decided (2026-11-15).** The jets are Taylor series in `s` of functions of the form `f(w + s p)`. The input
  is always affine: a point `w` and one direction `p`. Several directions, `f(w + s p + t q)`, are not supported.
  A curved path in the one variable, `w + s p + s^2 q + ...`, is not wanted either. So an input never carries an
  order axis, and the two-term form for a product with an input jet is part of the structure.
- **2.7, assessment** (2026-11-15, `scripts/review/order0_assessment.py`).
  - **In the explicit Taylor form, order 0 is exactly the plain program.** Checked for the probe kind on ragged
    cores with one sample axis, for `z`, `Jp` and `g`. The same functions, called with order 0, make the same 124
    contractions on the same shapes in the same order as an implementation that knows nothing about jets. No
    array has an order axis. The results are bitwise equal on numpy, and under jax the two traced programs are
    identical character for character. At orders 2 and 4 the same functions match the library's derivative kind
    to 4e-16.
  - **How.** The order is a group in each contraction string, empty at order 0; this is the engine's missing
    group, with `len_T` passed at the two terms where the operand ranks cannot pin it. Each product term carries
    the order group on one operand and takes one coefficient of the other. The loop over those coefficients, and
    the start values, are four small helpers that branch on the static `order`.
  - **Costs.** `order` becomes a static argument of every step function, or is bound once. The branch in the
    helpers is on a static integer, not on which data is present. The input jet is `w` alone at order 0 and
    `(w, p)` above, so the form of the sample depends on the order.
  - **Today's library, for comparison.** Order 0 through the derivative kind is not the plain program. It carries
    an order axis of length 1, makes one more contraction per mode in `z` and `Jp` and none more in `g`, and has
    2.5 to 3.7 times as many primitives in the traced program (concatenations, multiplications, additions). `p`
    is not read.
  - **Not assessed.** Orders above 0 were checked for correctness only: the loops over coefficients are unrolled
    and unoptimized (304 and 452 contractions at orders 2 and 4), so this says nothing about the banded and
    scanned forms of 2.9. Not covered: the uniform layout, the `K` and `C` stacks, several sample axes, apply
    and entries, the tilde-dot grade, and whether one contraction per term is itself the best plain program.
- **2.7, decided in structure (2026-11-15).** The maintainer supports making the order a grouped axis that is
  either empty or has one entry, so that structurally it acts like the batching and stacking axes. He asks for
  care with the terminology, because a batching axis means a bunch of the same thing bundled together and has
  fewer restrictions; see 9.4. The cases the assessment did not cover still have to be shown to behave the same
  way.
- **2.9.** Decision 4 dropped the option for orders, because for two full jets "vectorized over orders" is the
  dense `trs` form the library removed for its memory use. That is a reason of principle, but the measurements
  behind it were made in the derivative normalization.
- **2.8, decided** (2026-11-15, 04:11). After 6.6, Claude noted that the frontend now has one function with an
  `order` while the fitting layer still has six kinds. The maintainer: "Yes, let's make the fitting layer
  consistent." So the `*Derivatives` kinds and the kind strings `'probe_derivatives'`, `'apply_derivatives'` and
  `'entries_derivatives'` go; a kind is one of three, and the order is a separate integer. In the consumer,
  T3Polynomial's call `newton_cg(geometry, 'probe_derivatives', (ww, pp), ..., order=jet_order)` names the kind
  `'probe'`.
- **2.8, what it leaves open.** What a kind is and whether a user can define one (3.4). How the fitting functions
  take the sample: today it is `ww` for the plain kinds and the pair `(ww, pp)` for the derivative kinds, so with
  one kind the sample's contents would depend on the order. That belongs to 6.2 and 8.1.
- **2.4, the maintainer's suggestion** (2026-11-15, 04:26): "I have not figured that out yet. Maybe some insight
  can be gained from the theory of convolutions of formal polynomials."
- **2.4, Claude's assessment: a candidate definition** (`scripts/review/tilde_jet_pairing_check.py`). Not decided.
  - A jet of order `n` is an element of the ring `R = R[s] / (s^(n+1))`, and every term of the program is the
    product of `R`, the truncated convolution.
  - `R` has a pairing made from its own product: `beta(a, b) = [s^n](a b) = sum_t a[t] b[n-t]`, the top
    coefficient of the product. It is nondegenerate, and it is invariant: `beta(a b, c) = beta(a, b c)`. (`R` is
    a Frobenius algebra, and this is its form.)
  - So, with jets and tilde jets paired by `beta`, multiplication by a jet is its own transpose. The adjoint of
    every term is the same product as the term.
  - **The candidate definition.** A tilde jet is a jet. It acts on a plain jet by the top coefficient of their
    product. Slot `t` of a tilde jet is the sensitivity to the coefficient `t` orders below the top:
    `a_tilde[t] = d f / d a[n-t]`.
  - **The two storage orders in this language.** A residual in natural order, `c[t] = d f / d z[t]`, is an element
    of the dual of `R`: a polynomial in `1/s`. `R` acts on it by multiplying and keeping the non-positive powers,
    which is a correlation; that is the "second kind of product". Multiplying by `s^n` turns it into an ordinary
    polynomial, the reciprocal polynomial `s^n c(1/s)`, which is `c` reversed. The reversal is the isomorphism
    between `R` and its dual.
  - **Where the pairing shows.** Only at the two ends. At the seed, the weighted residual is reversed once. At
    the readouts, a core does not depend on `s`, so its gradient block is the top coefficient of a product; that
    is `top2` and `top3` in `order0_assessment.py`. At order 0 the reversal is the identity and the top
    coefficient is the only coefficient.
  - **The same thing as a sweep direction.** The forward program is causal in the order: slot `t` reads slots
    `<= t`. Its adjoint runs the other way, as the adjoint of a left-to-right sweep runs right to left. Stored
    reversed, it is causal again: slot 0 of a tilde jet reads only the top residual.
  - **A side result for 2.1.** In derivative coordinates the product is a binomial convolution, and the flip
    alone is not its transpose; it needs the weights `t!(n-t)!`. Taylor coefficients are the coordinates in which
    the transpose is a plain flip.
  - **Checked on numbers** at `n = 4`: the invariance of `beta` (1e-16); flip times the transposed product matrix
    times flip equals the product matrix (exactly); a chain of two products against finite differences (2e-10);
    the readout for a constant factor (2e-11); the failure of the plain flip in derivative coordinates (relative
    difference 0.6) and its repair by `t!(n-t)!` (exactly). `jet_steps_check.py` had already matched the library
    with tilde jets stored this way.
  - **What the definition does not settle.** A tilde jet means something only relative to `n`. The public
    residual of the transposes pairs slot by slot with the output (6.6); whether the tilde variables a user can
    hold or ask for are in the reversed form is part of the decision.
- **2.4, held by the maintainer** (2026-11-15, 04:44). "Let's hold off on deciding this for now. It hinges on
  whether we can come up with a more natural meaning where the reversed order follows naturally. My lean is to
  keep the residual in forward order at the API seam." 2.4 stays Assessed.
- **2.4, a second reading offered by Claude, for when he returns to it. Not decided.** A tilde jet is an ordinary
  Taylor jet in `s`: the Taylor coefficients of the plain tilde variable, when the plain transpose is run at the
  point `w + s p` for each `s`. Nothing in it is reversed. The residual enters through one polynomial,
  `c(s) = sum_t c[t] s^(n-t)`, and the gradient is the coefficient of `s^n` of the result. The reason is one line:
  `z[t] = [s^t] z(s) = [s^n] (s^(n-t) z(s))`, so `sum_t c[t] z[t] = [s^n] (c(s) z(s))`. The factor `s^(n-t)` lifts
  the order-`t` output to the top order, where one coefficient reads all of them. Under this reading the residual
  is in forward order at the API seam, as the maintainer leans; the seed step builds `c(s)` from it; and the
  reversed array is nothing but the coefficient list of `c(s)`. Checked in part 6 of
  `tilde_jet_pairing_check.py`: the plain adjoint evaluated at 13 values of `s`, with no jets anywhere, has the
  tilde jet as its Taylor coefficients (4e-13), and the gradient is its coefficient of `s^n` (2e-11).
- **2.4, the maintainer's question** (2026-11-15, 04:51): "Is it a jet or some kind of dual functional on jets?
  Maybe that is where the reversal is coming from." Claude's answer, not a decision: it is a dual functional, and
  yes, that is where the reversal comes from.
  - A tilde variable is a linear functional on jets, an element of the dual space `R*`. That is the object with
    a meaning; the two arrays are two ways of writing it down.
  - **Written in the dual basis** (forward order): slot `t` is the functional's value on `s^t`, which is
    `d f / d z[t]`. The ring acts on functionals by `(a . phi)(z) = phi(a z)`. In these coordinates that action is
    the correlation, and `s` acts as a shift down: `s . delta_t = delta_(t-1)`, `s . delta_0 = 0`. The dual has
    degrees `-n .. 0`: polynomials in `1/s`.
  - **Written by a representing jet** (reversed order): `R*` is isomorphic to `R` as a module over `R`, and the
    isomorphism is fixed by choosing the generator `[s^n]` of `R*`: every functional is `z -> [s^n](c z)` for one
    jet `c`. The dual basis element `delta_t` is represented by `s^(n-t)`. That is the reversal: a shift of
    degree by `n`, carrying degrees `-n .. 0` onto `0 .. n`. In these coordinates the action is the product.
  - **Why order 0 hides it.** At order 0 the dual of the vector space `V` is identified with `V` by the dot
    product, silently. At order `n` the dual is `V* (x) R*`, and `R*` needs its own identification with `R`. The
    dot product on coefficients is one, but it is not compatible with multiplication. The compatible one is the
    top-coefficient form, whose matrix is the flip.
  - **The second reading above is this representation**, not something separate: the polynomial `c(s)` is the
    representing jet of the residual functional. Saying "nothing is reversed" there moved the reversal into
    `c(s)`; it did not remove it.
- **2.4, the type catalog** (2026-11-15, 04:57). The maintainer: "Let's catalog the types for all edge variables
  (jet vs dual jet), where are inner products vs dual pairings, and where any Riesz maps are occurring." It is in
  `TYPE_CATALOG.md`, checked by `scripts/review/type_catalog_check.py`. Its findings, none of them a decision:
  - The letter fixes the network factor (`w`, `xi`, `mu` vectors; `eta`, `nu`, `gamma` (now `nu_L`), `z` covectors)
    in every grade. The grade fixes the order factor: plain and dot are jets, tilde and tilde-dot are dual jets.
  - The multiplier of a variable lives in the dual of that variable's space. In the network the dual is the
    partner's own space, so the partner rule absorbs it. In the order both partners are jets, so nothing absorbs
    it. That leftover dual is the reversal.
  - Inside the sweeps the order index is never contracted: plain and dot terms are the ring product, tilde terms
    are the action of the ring on its dual. It is contracted in three places: the block readouts (a dual pairing),
    the misfit and its sums (an inner product), and the seed (the Riesz map of that inner product).
  - There is one Riesz map in the sampling layer: residual to seed, `omega^2 r`. No inner product of a bond or
    Tucker-rank space is used anywhere in it. The flip is not a Riesz map: its form is not positive.
  - For 2.5: the misfit is a choice of inner product on data jets. For 2.3: derivative mode is another such
    choice and needs nothing below the seed step.
- **2.4, the maintainer's arrow picture** (2026-11-15, 05:20). Arrows on each edge, contracted into a core only
  when they point towards it; on internal edges an object is "neither plain nor dual"; plain and dual are definite
  only at the exterior, where `w` points in and `z` and `r` point out; `r` "must be converted to a plain object by
  a Riesz map. For the unjetted version, this means applying the weights. For the jetted version, this also means
  reversing the jets." His full words and Claude's reading are in section 8 of `TYPE_CATALOG.md`. In short: the
  order index is one more leg whose core is the ring product; a forward-order residual points out on it and the
  reversal turns it round; the weights are the misfit's part of the map and the reversal is the network's part;
  and the choice in 2.4 is the choice between turning the arrow once at the seed and giving the order leg its
  second orientation (the correlation), as a bond has `nu_term` beside `mu_term`. 2.4 stays held.
- **2.4, arrows on the cores** (2026-11-15, 05:26). The maintainer: give the frame cores orientations too, `L` with
  left-to-right arrows on `a` and `b` and an up arrow on `i`, `R` with right-to-left arrows; "Then every
  contraction can be type checked." Done in `scripts/review/arrow_type_check.py` and written up in section 9 of
  `TYPE_CATALOG.md`. The whole explicit program type checks (424 contractions, 176 sums). The plain and tilde frame
  variables flow with the arrows; `gamma`, `z` and the dot variables flow against them (now: `nu_L`, `z`, and the
  frame's dots `mu_R_dot`, `xi_U_dot`, `nu_L_dot`, `eta_D_dot`; in the new names, what flows against the arrows is
  what carries a basis in its name, and the outputs `z` and `Jp`); each block fits its hole. The checker rejects
  the residual put back as it is, and the weighted residual with the order not reversed. The arrows coincide with
  the frame's orthogonality conditions. A naming hazard is noted there: the library's names say where a store sits,
  and its arrow points the other way. Not a decision.
- **2.4, the convention for the arrows** (2026-11-15, 05:39). Agreed by the maintainer: the typing rule is the
  two ends of one arrow ("I agree with the first rule change, this is right"). His lean for the one convention:
  an arrow entering a core marks a dual index ("contracting puts a vector 'in'"), an arrow leaving marks a plain
  one. Claude agrees: an arrow shows the way a vector travels. It is the convention of index notation and of
  tensors as maps (TensorKit.jl); TeNPy labels the other way round, so it must be stated wherever arrows are
  used. With it every frame core, as a map along its arrows, satisfies `M M^T = I`, checked on the library in
  `scripts/review/gauge_arrow_check.py`. Section 10 of `TYPE_CATALOG.md`. The convention is not yet recorded as
  decided.
- **Parked for another arc** (2026-11-15, 05:39): the maintainer's remark that the diagonal weight matrices
  between the cores of a T3 are (pseudo-)inner products on the edges, with hidden identities on a plain T3, and
  that orthogonality for weighted networks should then be taken with respect to the weighted inner products.
  Outside this refactor (scope guidance 1.1). Notes in section 10 of `TYPE_CATALOG.md`.
- **2.4, resumed** (2026-11-15, 15:49). The maintainer: "I think we have a clear picture now. It is an adjoint
  variable with respect to a Lagrangian, and the reversal of order within the jet comes in because it is a dual
  object which points the opposite way on its edge. Are there any more subtleties here I'm not seeing?" Claude
  raised six points, none of which changes the picture; they are in section 11 of `TYPE_CATALOG.md`. The two that
  touch the decision: on the manifold frame a tilde variable is the multiplier of the dot of its partner (checked
  with jets in `scripts/review/tilde_multiplier_check.py`), and the forward-order functional at the seam and the
  turned seed inside are two arrays, so `w_tilde` has to name one of them. Awaiting his confirmation of the three
  parts of the decision: forward order at the seam; turned once at the seed and stored that way inside,
  including the tilde variables a user holds; `w_tilde` names the turned seed.
- **2.4, the maintainer's statement of it** (2026-11-15, 16:04). "My thinking is that the residual is the same type
  of object as the output z. So, if the function that takes w's and pushes them through the network is f, then
  the residual is r = z_true - f(w). We can expect the user to supply z_true in the same order as if it had been
  generated by an f-like function. The dualization comes from the misfit function ||r||^2 = <r,r> (norm which is
  weighted on base vector and has the appropriate inner product on jet involving the reversal). Does this comport
  with your suggestions, and if not, what are the differences?"
- **2.4, Claude's answer: it comports, with three precisions.** Awaiting his reply.
  - **Where the reversal is.** The misfit's inner product itself has no reversal: it is the weighted sum of
    squares slot by slot, `sum_t omega[t]^2 r[t]^2`. It cannot have one, because the reversed form
    `[s^n](r r)` is not positive (at order 1 it is `2 r[0] r[1]`). The reversal is in the map from `r` to the
    seed: the functional `<r, .>` written as a jet that can enter the network, `flip(omega^2 r)`. The two are
    tied by `sum_t omega[t]^2 a[t] b[t] = [s^n](flip(omega^2 a) b)`.
  - **The sign.** The library and the prototypes use `r = f(w) - z_true`. With the other sign the gradient
    changes sign.
  - **Two seams.** In fitting, the user supplies the data and the weight, both in forward order and of the type
    of `z`; the dual object never crosses the seam. In the frontend transposes, the argument is an array of the
    type of `z` in forward order, and the function applies the plain slot-by-slot inner product, which is the
    identity on the array, then the turn. Under the maintainer's view that argument is a residual, not a tilde
    object, and `w_tilde` is what the dualization produces.
- **2.4, decided** (2026-11-15, 16:10). On Claude's three precisions the maintainer answered: "1. Agree. 2. No
  preference of z-f(w) vs f(w)-z. Keep the libraries sign convention, whatever it is. If the library is
  inconsistent, you can use the convention from your prototype. 3. Ok." What is fixed:
  - The data and the residual are of the type of `z`: jets in forward order. The user supplies the data in the
    order in which the function would have produced it.
  - The misfit's inner product is slot by slot, `sum_t omega[t]^2 r[t]^2`, with no reversal in it.
  - The dualization is one step at the seed: the weights, then the turn. Every tilde variable is stored turned,
    including those a user holds. Its pairing with a jet is the top coefficient of the product.
  - `w_tilde` names the seed. The argument of the frontend transposes is a residual, of the type of `z`, in
    forward order; the function applies the plain slot-by-slot inner product (the identity on the array) and the
    turn. Its present name `ztildes` is then a misnomer; renaming it is left to the naming questions.
  - **The sign** is the library's, `r = S(x) - data`. Checked: the library is consistent (`backend/fitting.py`,
    `backend/optimizers.py`), and the prototypes use the same sign.
  - This also settles the first half of 2.5, what the weight multiplies. The default weight is still open there.
- **2.5, opened** (2026-11-15, 16:10 to 16:19). After 2.4 only two parts remain: the default weight, and the shape
  of the weight under one kind. Claude recommended ones on the Taylor coefficients, and a weight of shape
  `(d,) + T` for probe and `T` for apply and entries, with no rule for a bare vector. The maintainer did not
  answer these. He set out three ideas that should guide the choice: the aim is to approximate a function with
  a polynomial; the importance of each term depends on a length scale, for which 1.0 is the natural choice; and,
  empirically, equalizing the influence of each term helps the fit a lot. He asked for the consumer's record
  first.
- **2.5, the survey** (`T3POLYNOMIAL_SCALING_SURVEY.md`). T3Polynomial today: whitened coordinates, unit probes,
  centering, an input scale `kappa` (automatic, an estimate of the Taylor radius), and block-equalizing weights
  `1 / (norm of the block)` per mode and order, with exact zeros for dead blocks. The history from per-order
  weights (July) through block weights, the selection score, centering and `kappa` (September), the deferred
  per-sample weight, and the negative result of whitening the directions. Claude's observations there, none a
  decision: under block-equalizing weights derivatives and Taylor coefficients give the same loss; the length
  scale is already an input, the length of the direction `p`; `kappa` acts on conditioning and is not a weight;
  equalizing is a policy computed from data, for which the library supplies the mechanism; a zero weight must
  remove a block exactly; a weight per sample has been wanted and is deferred in T3Toolbox
  (`dev/data_shaped_residual_weight_reviews.md`).
- **2.5, the maintainer's lean on the architecture** (2026-11-15, 16:33). "The big picture idea is that we want to
  give hooks to allow the user to play with different approaches for their problem, similar to our approach for
  performance options. It is not just a question of default, but also a question of code architecture. My lean
  would be to provide standalone functions which prepare the data and create weights, which the user can do on
  their own before passing the data to our fitting functions. We can also create nicety kwarg options which do it
  for the user within the fitting, by calling the preparation functions from within. For the default, I am torn
  between do nothing, and do everything." For nothing: the fit is likely to sit inside a rank continuation loop,
  and the data should be prepared once before it. For everything: the fit "just works" for a user who is not an
  expert, where raw data is "likely to have the fit immediately fail badly, when some simple scaling and
  equalization could have made it work."
- **2.5, Claude's analysis of that lean.** Awaiting his choice; nothing decided.
  - **Three classes of preparation.** (a) Those that change only the loss: a weight, the conversion between
    derivatives and Taylor coefficients, the length of the direction `p`. Nothing has to be undone afterwards. (b) Those
    that change the unknowns: a rescaling of a mode's input coordinates, as `kappa` does to the points. The
    starting point and the fitted tensor are then in other coordinates, and the function comes with an inverse.
    (c) Those that belong to the problem: centering, the homogenization coordinate, the pivot from measured jets
    to targets. These stay in the consumer.
  - **A rule for what may be done inside a fit:** only class (a), because it does not change the meaning of the
    starting point or of the tensor returned. Classes (b) and (c) stay standalone.
  - **For the misfit, "everything" is one thing.** Under equalizing weights the conversion and the length of the
    direction cancel. So the default is a choice between ones and the equalizing weight.
  - **A layered default, suggested.** The backend does nothing: the weight is what it is given, and none means
    ones. The frontend fitting functions equalize when no weight is given. An explicit weight always wins, so a
    rank continuation loop prepares once and passes it, as T3Polynomial does today. The library already has this
    split: the frontend owns the weight contract (`fitting._canonical_weight`) and the backend is lenient.
  - **The hazard of equalizing by default:** a block of data that is zero or nearly zero. T3Polynomial's record
    is that a floor turned a dead block into a weight of 1e12. The standalone function needs a stated rule.
  - **2.3 has the same shape.** The derivative-mode adapter of 2.2 is a standalone conversion with a keyword
    that calls it.
- **2.5, the default weight, decided** (2026-11-15, 16:49). The maintainer: "Let's go with do-nothing (ones) by
  default, with an asterisk that the default is subject to change in a later release if and when we figure out a
  more robust 'just works' method." So with no weight given the misfit is the plain sum of squares of the Taylor
  coefficient residuals, at both the frontend and the backend. The documentation of the default has to carry
  that asterisk. The layered default Claude suggested is not taken.
- **2.5, what is still open.** The shape of the weight under one kind (Claude's recommendation: `(d,) + T` for
  probe, `T` for apply and entries, no rule for a bare vector). 2.5 stays open until that is answered.
- **Carried to other rows, not decided.** The maintainer's lean, with Claude's agreement: standalone functions
  that prepare data and create weights, and keywords in the fitting functions that call them (now opt-in, since
  the default does nothing). Which functions exist is 6.3; the keywords are part of 6.2 and 8.x. Claude's
  suggested rule, that a keyword inside a fit does only loss-only preparation, and the rule the equalizing
  function needs for a zero or nearly zero block, are also unconfirmed.
- **2.5, decided** (2026-11-15, 17:00). The maintainer: "Agree with architecture and keyword-inside-fit rule, and
  shape weight proposal." With the default of 16:49 this closes 2.5. The weight is `(d,) + T` for probe (so `(d,)`
  at order 0 and `(d, order+1)` above) and `T` for apply and entries; a bare vector has no special meaning, and
  the user broadcasts a per-order or per-mode weight themselves. Claude's question whether a mode dimension of 1
  is still accepted and broadcast was not answered; the simpler reading, the exact shape only, is taken, to be
  corrected if wrong. The architecture and the rule for keywords are recorded as 6.7.
- **2.3, a proposal and an example** (2026-11-15, 17:04 to 18:32). Claude's proposal for the adapter, in the
  terms of 6.7: standalone conversions between Taylor coefficients and derivatives (slot `t` times or divided by
  `t!`), for anything of the type of `z`; a separate conversion for the weight, which goes the opposite way
  because it is a dual object; one opt-in keyword on the fitting functions that converts the data, the validation
  data and the weight on the way in and nothing on the way out; no such keyword on the sampling methods; the
  backend Taylor only. The maintainer asked to see a minimal example of it in use; it was shown in the
  conversation. A point the example raises: with the keyword and no weight, the weight has to be `t!` per order,
  so that the objective is the plain sum of squares of the derivative residuals. Checked on numbers: the
  conversions reproduce the derivative-mode objective exactly, with a weight and without. Not decided.
- **2.3, amended on 2026-11-16 (15:12): the keyword is removed and a fourth standalone function is added; see
  the note "Derivative mode moved out of the fitting layer" in section 8.**
- **2.3, decided** (2026-11-15, 18:37). The maintainer, on the form of the adapter and its example: "Yes, this is
  good." On the keyword changing what "no weight" means: "yes, but let's make sure to document clearly. We can
  have a doctest example illustrating this behavior, showing what happens to the data in each case".
  - Three standalone conversions: derivatives to Taylor coefficients (slot `t` divided by `t!`), the reverse, and
    a derivative-mode weight to a Taylor weight (slot `t` multiplied by `t!`).
  - One opt-in keyword on the fitting functions that applies them to the data, the validation data and the
    weight. With the keyword and no weight, the weight is `t!` per order.
  - Required: clear documentation of that, with a doctest that shows what happens to the data in each case. A
    draft that runs is `scripts/review/adapter_doctest_draft.py` (three cases: Taylor data; derivative data with
    a weight; derivative data with no weight).
  - Not decided: the names of the functions and the spelling of the keyword (`jet_form='derivatives'` in the
    draft), and what the display shows in derivative mode (8.4).
- **2.9, decided** (2026-11-15, 19:08). The maintainer: "I confirm 2.9 as stated. However, I want to implement with
  a scan rather than a loop. Although the 4th order polynomial is my target, there can be other users with
  different problems who want higher order polynomials; for example to get higher accuracy on smaller problems."
  - Confirmed as stated: every term over the orders is the truncated product; one operand carries `T` through the
    contraction and the other is taken one coefficient at a time; no dense order-by-order form; no user option;
    the block readouts compute only the terms whose orders sum to the top.
  - His addition: where the second operand is a full jet, its coefficients are taken by a scan, not an unrolled
    loop, so that the compiled program does not grow with the order. Claude had proposed leaving that to group D.
  - Where a scan is needed: the products of two full jets, which are in the tilde and tilde-dot steps and their
    readouts. A product with an affine operand (`xi`, `w`, `xi_dot` (now `xi_U_dot`; the raw `xi_dot` is affine
    too)) is two contractions and needs none. At order 0 there is no order axis and no scan (2.7).
  - What a scan costs (`scripts/review/order_scan_check.py`). The body must have the same shapes at every
    iteration, so the truncation cannot be done by slicing; it is done by a shift along the order axis and a
    mask. The scan matches the explicit sum to rounding at orders 1 to 16, under jit. The traced program has the
    same size at every order (77 lines), where the unrolled loop grows from 56 lines at order 1 to 491 at order
    16. The mask throws work away: the scan does `(order+1)^2` contractions of one pair of orders where the
    triangle has `(order+1)(order+2)/2`, a factor 1.67 at order 4 that tends to 2.
- **2.9, the scan reconsidered** (2026-11-15, 19:14). The maintainer, on seeing that the scan does up to twice the
  contractions: "Oh hold on. We need to think more about the scan. Performance is important. Can you show me
  pseudocode of representative contractions, loop vs scan, showing how the scan doubles the contractions?" The
  parts of 2.9 confirmed as stated stand. How a full operand is taken is open again, so the row is marked to
  confirm.
- **2.9, how the library does it today** (2026-11-15, 19:22). The maintainer: "How does this compare with how the
  code currently implements the analogous operation for convolution of jets?" Read in
  `backend/sampling_derivatives.py` at 2026.5.0 (the note "The jet kernels" and `_eta_jets_order_step`,
  `_adj_src_order_step`, `_mu_jets_step`, `_adj_prop`):
  - **An affine operand:** the two shifted copies of the full jet are stacked on a pair axis and contracted
    against the two coefficients in ONE contraction. No scan.
  - **Two full jets:** a scan over the order `r` of one factor. Each step contracts that one order slice with the
    core, then contracts the result against ALL slots of the other jet, which rides the carry and is shifted one
    slot per step. The truncation comes from the zeros the shift brings in. No traced shift, no mask, no dynamic
    slice. The binomial weights come from the same shift; for Taylor coefficients they are gone.
  - **It is a real scan only on the uniform layout** (`jax.lax.scan`); on the ragged layout it is a Python loop,
    which unrolls under jit.
  - **The reason the code gives is memory**, not the size of the program: only one order slice of the spatial
    product is live at a time. Its docstring records a measured 14 to 28 times smaller peak than the vectorized
    dense form, constant in the order, and says: "The order loop being a real scan is what forces this -- an
    unrolled loop does not."
- **2.9, a correction to Claude's comparison.** The first table counted "pairs of orders" as if every contraction
  cost the same. Two contractions are involved. The expensive one is the core with one coefficient
  (`W ra n rb` multiplications); the cheap one pairs that product with one coefficient of the other jet
  (`W ra rb`, a factor `n` fewer). Counted properly (`scripts/review/order_scan_check.py`):

  | Form | Multiplications | Traced program |
  | --- | --- | --- |
  | Loop over `r`, with slices | `P E + P(P+1)/2 c` | grows with the order: 143 lines at order 4, 491 at order 16 |
  | Scan over `r` with the other jet carried and shifted one slot per step (the library's scheme, in Taylor form) | `P E + P^2 c` | 34 lines at every order |
  | Scan over a table of the needed pairs | about `P(P+1)/2 (E + c)` | 74 lines at every order |

  So the scan costs more than the loop only in the cheap contraction: 6 percent in all at order 4 with bonds of
  64 and a mode rank of 32, 20 percent at order 16; 29 percent at order 4 when every rank is 4. "The scan doubles
  the contractions" was true of the cheap ones only. The pair table repeats the EXPENSIVE contraction for every
  pair and costs 2 to 7 times the loop: Claude's recommendation of it is withdrawn. The scan with a shift and a
  mask that Claude wrote first is the library's scheme done less well (a traced shift and a mask where a carried
  shift needs neither). All three forms give the explicit sum under jit. Nothing was timed.
- **2.9, decided: two forms, chosen through the plan** (2026-11-15, 19:32). The maintainer: "Let's have two
  options, controllable by the user through the Plan abstraction. The analog of the libraries current scheme, and
  a loop with slices. The scan is the default. In the plentiful memory regime, I can imagine jax's jit compiler
  may find optimizations which increase speed,if it can see the whole loop."
  - **The scan (the default).** The library's present scheme in Taylor form: a scan over the coefficients of one
    operand; each step contracts that one coefficient with the core and then with every slot of the other jet,
    which rides the carry and is shifted one slot per step. One coefficient of the expensive product is live at
    a time, the compiled program has one size at every order, and it does the square of the cheap contractions
    where the triangle would do.
  - **The loop with slices.** A Python loop over the same coefficients, with the truncation done by slicing. It
    does only the triangle, the compiler sees the whole loop, the compiled program grows with the order, and
    nothing is promised about peak memory (the library's own note on its scan: "an unrolled loop does not" bound
    it).
  - This replaces "No user option" in the row. "No dense order-by-order form" stands: neither option is that
    form. It also supersedes decision 4 of 2026-11-14 ("no option for orders; drop `Plan.orders`") and section
    4.7 of `DESIGN_AFTER_REVIEW.md`: the option that decision removed was scan against the dense form, and the
    dense form stays out. The `SamplePlan` of that document's section 5 gains one field. It is one more of the performance options the framework hands to the user, and like the others it
    changes how the computation is carried out and never what is computed.
  - **Claude's readings, not said by the maintainer:**
    - The setting is one field of the plan and applies to every product of two full jets in the call, not term
      by term or site by site. The order axis is not an axis of the sample stack, so it is a setting of its own
      and not one more entry in the plan's tuple of scans (see 7.1).
    - ~~With the default, the scan is a real scan on every layout.~~ **Wrong; corrected by the maintainer at
      19:50** (next note). The scan is written through `xscan`, so it is a real scan with jax on the uniform
      layout and a Python loop on numpy or on the ragged layout, as in the library today.
    - The two settings agree to rounding, not bit for bit, so the test that they agree is a tolerance test.
    - A product with an affine operand, and order 0, are the same under both settings: neither has a loop over
      the orders.
    - The block readouts take only the pairs of orders that sum to the top, so there the two settings do the
      same contractions and differ only in the form of the program.
  - Not decided: the name of the setting and of its two values (`over_orders='scan'` and `'loop'` in the
    prototype are placeholders).
  - Checked (`scripts/review/order_scan_check.py`, part 3): the two settings as one function agree to rounding
    under jit at orders 1 to 16, in float64 and in float32, and keep the dtype.
- **2.9, how the scan is written** (2026-11-15, 19:50). The maintainer, correcting Claude's reading above:
  "The implementation is through the xscan abstraction, which dispatches to a jax scan if jax is used and the
  layer is uniform, and a loop if pure numpy or ragged. In general, things should respect the numpy vs jax dispatch
  rule, and scan type operations should be polymorphic in ragged vs uniform. This may have reasonable exceptions
  for performance optimizations and issues related to masking; though for uniform sampling the pattern is to mask
  first, then compute edge variables with the assumption that masking was already done. For map-type operations,
  uniform and ragged may have different behavior for performance and memory considerations; this is one of the
  things we are helping the user control with the Plan abstraction in this refactor arc. Dispatch at the lowest
  level within reason, use jax if any input is jax and numpy otherwise, write code generically with xscan and
  similar, which are obtained from the get_backend function and may also use the tree_contains_jax function.
  Uniform vs nonuniform is deduced from the input types. Look at other functions in the library for examples, for
  example things in the existing backend/probing.py."
  - For 2.9: the `'scan'` setting is the carried shift written with `xscan`. With jax on the uniform layout it
    is `jax.lax.scan`; on numpy or on the ragged layout it is a Python loop over the same body. The `'loop'`
    setting is the loop with slices on every backend.
  - The general rules in this statement are recorded under section 5, where they bear on 5.2, 5.3 and 5.6.
  - The prototype is rewritten in that pattern (`scripts/review/order_scan_check.py`): `get_backend`,
    `tree_contains_jax`, `xscan`, a closure-free body with the core on the carry, and the library's `contract`.
    The layout is deduced by the caller and handed down as `is_uniform`, as `_eta_jets_core` does, because one
    core's arrays look the same in both layouts. Checked on all four backends, in float64 and float32: both
    settings give the explicit sum, numpy in gives numpy out, jax in gives jax out, and the dtype is kept.
  - **What follows, Claude's observation.** Where `xscan` is a Python loop, the `'scan'` setting has neither of
    the scan's two properties: under jit on the ragged layout its traced program grows with the order (143 lines
    at order 4, 479 at order 16, against 34 at every order on the uniform layout), and nothing bounds the peak
    memory. It still does the square of the cheap contractions, where `'loop'` does the triangle (138 and 474
    lines). So on the ragged layout and on numpy, `'loop'` does less work and gives up nothing. The user can
    choose it through the plan. Whether the default should depend on the layout is a question for 7.2 (the
    default plan); as decided, the default is `'scan'`.
  - The line counts quoted in the earlier notes (143 and 491 for the loop) were of a first draft written in
    jax only. Written generically the loop has 138 and 474.

## 3. Variables and kinds

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 3.1 | The primitives of a variable's name are a letter for the side contracted (`mu`, `nu`, `xi`, `eta`), the basis of the cut edge where it is not the usual one, and a grade (plain, dot, tilde, tilde-dot) | Decided | 9.6 |
| 3.2 | `nu_L` (`gamma` until 2026-11-15, 23:23) stays, as an ordinary variable. It is computed only when `Kp` is wanted, or when the value goes through `D`. No frame convention removes it | Decided | 4.1 |
| 3.3 | The tilde names always mean the adjoint sweep of the residual. The adjoint sweep of `Jp` has no names in a call that also holds the residual's; its variables are reached by a call with `Jp` given as the residual | Decided | 6.5 |
| 3.4 | In the sampling layer a kind is closed: a small description (what goes in at a mode edge, what comes out, the order), not a class a user subclasses. The open extension point is in the fitting layer: a protocol for an objective, which the built-in objectives implement by calling the engine **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** for the entries kind the order is `()` (O-46). | Decided | 1.1 |
| 3.5 | The output is `z` for every kind | Decided | |
| 3.6 | The engine takes the data `y` and an optional `weight` (ones by default). The weight acts on `f`, `g`, `JtJp` and `Kp` alike, through one public helper that turns a jet into a dual jet. The bare transpose takes a seed and no weight. The engine has no other notion of a loss | Decided | 1.1 |
| 3.7 | What happens to the sample stack `W` on each output. Kept: `z`, `r`, `Jp` and every edge variable. Summed by default, kept on request: the outputs shaped like cores, `g`, `JtJp`, `Kp`. Always summed: `f`, `r_block_sumsq`, `Jp_norm_sq` | Decided | 3.6 |

Notes:

- **3.1, decided with the notation** (2026-11-15, 23:23; see 9.6). The question was whether letters and grades
  are the right primitives. The notation the maintainer locked in has three: the letter, the basis and the
  grade. That this closes 3.1 is Claude's reading of his decision, to be corrected if wrong.
- **3.2, decided** (2026-11-16, 02:16). The maintainer: "3.2: Agree with proposal." With `Vc` the value no
  longer reads `nu_L`. `Kp` still does: the curvature term is taken at the base point in its own cores, so the
  right of every bond has to be in the `L` basis there, and with slack ranks `nu_L` cannot be recovered from
  `nu`. So `nu_L` stays, a variable like every other, computed only when `Kp` is wanted or when the value goes
  through `D`.
- **3.2, what the notation changes.** The variable is now `nu_L`: the right of a bond in the `L` basis. The
  letter `gamma` is retired. It can be wanted and held like every variable (confirmed at 21:57). What stays to
  confirm is the second half of the question, which waits on 4.1: whether a different frame convention removes
  the need for a second basis.
- **3.2.** Decided on 2026-11-14 as public. It exists because the frame's right variable is swept through `R`
  while the value and the curvature term need it through `L`.
- **3.3.** Decided on 2026-11-14: only the copies seeded by the residual, because those depend on the point and
  the data only.
- **3.4, decided** (2026-11-16, 02:22). The maintainer: "Agree with split. I have been wanting to do something
  like this for some time."
  - **What a kind is after the decisions so far.** The three kinds share every sweep. They differ in what goes
    in at a mode edge (a vector for probe and apply, an index for entries) and in what comes out (one output per
    mode with that mode left open, for probe; one number, for apply and entries). The weight and the data are
    arguments of the engine (3.6), the chunk size left the kind (7.8), and the order is a number (2.8).
  - **The library's own open question** (`dev/OPEN_QUESTION_extension_surface.md`, opened 2026-08-21) gives the
    purpose of the open surface in the maintainer's words: "the point of the open surface is to allow the user
    to use our machinery to fit tensors with their custom objective function." Its four wanted extensions land
    as follows: several data sources in one fit, above the engine (one call per source, the results add);
    weights that change every iteration, solved by 3.6 (the weight is an argument, not part of an identity that
    is compiled in); a general linear functional with its own forward and transpose, above the engine (it has
    no sweeps to share); a new sampling operator (forward-only probing, contraction with another T3), inside
    the engine as new step functions.
  - **Decided: the split by layer.** In the sampling layer a kind is closed, and a new sampling operator is an
    addition to the library. The open extension point is in the fitting layer: a protocol for an objective that
    says what the optimizers need from it. What the protocol contains is 8.2 and 8.3.
  - **Forward-only probing, left off this arc** (02:28). Claude raised it as the probe kind with only some
    modes left open: a field of the kind's description, "all modes" by default, which lets the engine skip about
    half the sweeps where a zero weight could not (a weight is data; a set of open modes is structure). Not
    prototyped. The maintainer: "Leave it off this arc. Nice to have not need to have." So the probe kind of
    this arc has every mode open and no such field.
  - **For when the refactor lands:** this answers the library's standing open question, which by the
    repository's rule is not archived until it is resolved. That file is to be given its resolution then.
- **3.4, the earlier record.** Decision 8 left the `SamplingKind` protocol unchanged so that user-written kinds keep working. That is a
  compatibility reason. This question is also the library's standing open question on its extension surface,
  which the target can answer instead of deferring.
- **3.6, decided** (2026-11-16, 00:56 to 01:04).
  - **Why anything is inside.** The gradient is the transpose applied to a seed, and the seed is made from the
    output: `r = z - y`, then the weight. For `f` and `g` in one pass over the samples that step has to happen
    inside the pass, between the forward and the adjoint sweep of one chunk. Otherwise the plain variables are
    swept twice, or held for every sample.
  - **The maintainer's structure** (00:56): "we create a publicly accessible helper function which converts a
    dual jet to a jet using weights and jet reversal (and another function which does the inverse), then we pass
    the weights to the engine as needed, and inside the engine it calls the helper function. The helper function
    can also be used by the user for their own purposes."
  - **The weight on `f`** (01:04). Claude had said that `f` does not need the weight inside, since the block sums
    of squares could be weighted outside. The maintainer: "There needs to be consistency between f and its
    derivatives. Therefore, if g, JtJp, and Kp are weighted, then f must be weighted. Therefore I think we should
    add the weight to f. If the user supplies the same weight to all these functions, then the results they get
    are derivatives of each other without extra work. It should be optional on all of these, defaulting to ones.
    The transpose does not need weight. Really, JtJp is "JtWJP" where W is the Riesz operator (inverse mass
    matrix), and Jt is the unweighted transpose. We keep the name JtJp, because the weight is seen as optional."
  - **So:** with `omega` the weight and `omega^2` its Riesz operator, `f = 1/2 <r, omega^2 r>`,
    `g = Jt omega^2 r`, `JtJp = Jt omega^2 J p`, and `Kp` uses the seed `omega^2 r`, with one optional `weight`
    of the shape of 2.5. The bare transpose takes a seed and no weight. The operator is applied in one place, the
    public helper. (The operator is not written `W` here: `W` is the sample stack. The maintainer pointed out
    the clash at 01:26.) The regularizer, the derivative-mode adapter (2.3) and the data preparation (6.7) stay above
    the engine. Another loss is built above it from `z` and the bare transpose, at the cost of a second pass.
  - **Details noted by Claude, not answered:** the inverse helper divides by the weight squared, so it does not
    exist where a weight is zero (6.7 lets a weight be zero to mask a block); the helper works in Taylor
    coefficients, the `t!` factors being the adapter's; at order 0 it is the multiplication alone. The names of
    the two helpers are open (6.3).
  - **Considered and not taken:** a loss function passed by the caller and applied inside the pass.
- **3.3, decided** (2026-11-16, 01:36). The maintainer: "Proposal confirmed."
  - **The question, as put to him.** In one Newton step the adjoint sweep runs for two inputs. The residual's:
    the seed is made from `r`; it gives `g`, and `Kp` reads its variables; it changes only when the point or the
    data change. The direction's: the seed is made from `Jp`; it gives `JtJp`; it changes every CG iteration.
    When one call asks for `JtJp` and `Kp` together, both sweeps run in that pass, and a name like `mu_tilde`
    can mean only one of them.
  - **Decided.** The tilde names always mean the residual's sweep: those are the ones worth asking for and
    holding. The direction's sweep has no names in that call; its variables live inside `JtJp`. Nothing is
    unreachable: a call with `Jp` given as the residual has them under the ordinary names. That last part rests
    on 6.5 (who vouches for a held variable), which is open.
  - **Not taken:** a second set of names for the direction's sweep.
  - This is the decision of 2026-11-14 confirmed, with the route for the direction's variables added. The row
    used to read "Which seeded copies of the tilde variables are public?"; "public" has since come to mean every
    variable (6.4), so the row is stated as what a name refers to.
- **3.7, decided** (2026-11-16, 01:32). Put first as "which reductions over the samples are sampling
  variables", the question confused the maintainer (01:26). Put as a table of what happens to `W` on each
  output, he confirmed it: "I understand now. Confirmed. The table with columns "What happens to W" and
  "Outputs" in your message is what I want."

  | What happens to `W` | Outputs |
  | --- | --- |
  | Kept | `z`, `r`, `Jp`, and every edge variable |
  | Summed by default, kept on request | The outputs shaped like cores: `g`, `JtJp`, `Kp` |
  | Always summed | `f`, `r_block_sumsq`, `Jp_norm_sq` |

  - The third row holds numbers, not arrays over the samples. They are formed inside the pass, because under a
    chunked plan `r` and `Jp` are dropped chunk by chunk.
  - `f = 1/2 <r, omega^2 r>` is the objective. `r_block_sumsq` is the sum of squares of `r` for each mode and
    order, unweighted, in the shape of the weight; the display's table of relative errors reads it.
    `Jp_norm_sq = <Jp, omega^2 Jp>` is the Gauss-Newton quadratic: the same number as `p` paired with `JtJp`, with
    no adjoint sweep.
  - **Said with the question and not spoken of in his answer** (Claude's, to be corrected if wrong): `f` keeps
    `C`, `Jp_norm_sq` keeps `K + C`, and `r_block_sumsq` keeps `C` (the library's `block_sumsq` sums `C` away
    today); and an unsummed `W` on a transpose becomes the tangent stack of the result in the order
    `W + K + C`, as in the library today.
  - The block sums of squares of the data are not sampling variables: they depend on the data alone (6.7).
- **The sample stack on the outputs** (the maintainer, 2026-11-16, 01:26; it bears on 3.7, 5.4, 6.2 and 8.1).
  - **The residual.** "The residual is f(x)-y or z-y, where y is the data, and z=f(x) is the model prediction of
    the sampling map. The data y will typically be stacked over the W axis group ..., while the model prediction
    z can be further batched over the base point batch axis group C, because the T3 itself can have a C batch.
    Then, in the formula z-y, the data y is broadcast over C."
  - **The transposes sum the samples by default.** "When it comes to the transposes which output cores, the
    W-axes from the sample data gets incorporated into their C or K stacks (forget the details of which), unless
    the option is given to sum the samples, in which case the W stack is summed. This is important since stacking
    the W results at the front (as simple jax autodiff approaches do) would put it before the site index which
    must come first, and necessitate a transpose and hence memory reordering. Right now, the default is to *not*
    sum the W stack on transposes. For this refactor I would like to change the default so that it does sum the
    stack, but keep the option to not sum it." (In the library today the flag is `sum_over_probes=False`, and an
    unsummed `W` becomes the tangent stack `K` of the result, in the order `W + K + C`:
    `docs/batching_and_stacking.md` section 11.)
  - **Which layer supports the stacks.** "The optimization and fitting layers can assume that there is no C or K
    stacks, but the sampling layer must support C and K stacking."
- **3.6, the earlier record.** Decided on 2026-11-14 as inside, because computing `f` and `g` in one pass needs the residual formed
  inside the chunk scan. A reviewer argued for outside.

## 4. The frame and the value

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 4.1 | A frame is the four stores `(D, U, L, R)`, and `(D, L)` is the base point. The Tucker center cores may be supplied beside it. With them the value is read off the frame, `z[i]` being `eta[i]` through the center; without them it goes through `D` | Decided | 1.1, 1.2 |
| 4.2 | Frame consistency (the four stores describe one tensor; given centers agree with it) is a precondition, checked in safe mode at the moment a function that requires it is called, and never otherwise. No runtime safety check goes through a dense tensor **Amended 2026-11-16, 19:30 (the plan's O-5, O-6):** the check is the frontend's; the backend has checkers and no safe mode; what is checked is a frame the caller supplied, `L` against `R` through `D`, once when the model is built; centers are not checked in the frontend, since no frontend user supplies them | Decided | 4.1 |
| 4.3 | At an accepted point the value is read off the frame with `Vc` (4.1). At a trial point the optimizers offer three policies: the trial's raw cores; a frame at every trial, with its plain variables held for the accepted one (the default); a frame at every trial with the gradient computed with the value | Decided | 4.1 |
| 4.4 | Frames support arbitrary tails, and nothing checks for them. The boundary vector is ones, sized by the end bond of the cores passed. A function that requires squashed tails is to be looked at with suspicion | Decided | 4.1 |
| 4.5 | The gauge projection, the lift and the assembly of the exact Hessian product live in the geometry and are called by the fitting layer. The sampling layer never knows the geometry: it returns blocks before projection and takes the lifted direction `(D_dot, L_dot)` for `Kp` | Decided | 1.1 |

Notes:

- **4.1, the first half of 4.3, and the centers' part of 4.2, decided** (2026-11-16, 01:49 to 01:52).
  - **The two routes to the value at a point that has a frame.** Through the base point's cores:
    `z[i] = z_term(eta_D[i], D[i])`, which costs one more right sweep (`nu_L`) and one more trunk per site
    (`eta_D`) beyond what `g` needs. Read off the frame: `z[i] = z_term(eta[i], center[i])`, with the Tucker
    center core of site `i`, which has the shape of `V[i]`; no extra sweep. The identity is checked
    (`scripts/reuse_check.py`, 1.7e-15).
  - **The maintainer** (01:49): "Can we make both options available? If the user supplies the Tucker centers, we
    use them. If not, we go through D. In the optimization routines that we control, we would supply the Tucker
    centers. Are there any downsides of supporting both? Also, currently I believe the orthogonalization creates
    the TT and the Tucker centers, can you check this? With the planned retract-and-orthogonalize function, we
    should make sure it can return the centers too."
  - **Checked in the library.** `t3_orthogonal_representations` (`backend/fv_conversions.py`) returns the frame
    and, as a second result, the TT centers (from the right-to-left sweep) and the Tucker centers (from the
    up-orthogonalization of those). The manifold geometry's `frame` keeps the first result only
    (`frame, _ = t3_orthogonal_representations(x_cores)`). So the centers cost nothing new.
  - **Downsides named by Claude.** A supplied center can be wrong or stale, and then `f` is silently wrong;
    two routes to one number have to be tested against each other; wrong shapes are a structural error. Only
    the probe kind gains: for apply and entries the value through `D` is `mu` at the last bond, which the left
    sweep already gives. `Kp` reads `nu_L` and its family on either route.
  - **Decided** (01:52: "Agree with proposal for 4.1 and first half of 4.3. Also, agree that wrong centers
    should error in safe mode."):
    - the engine takes optional Tucker centers; with them `z[i]` is `eta[i]` through the center, without them
      it goes through `D`;
    - the frame construction and the retract-and-orthogonalize function return the centers;
    - the optimizers the library controls supply them;
    - centers that do not agree with `(D, L)` are an error in safe mode. (The check is core-sized. Claude
      believes it needs an orthogonal frame; not written.)
  - **The name** (01:54). The maintainer: plain "tucker_centers" is unambiguous but long. Claude suggested
    `Vc` for the Tucker centers and `Hc` for the TT centers: each has the shape of the direction core it sits in
    place of, `V[i]` and `H[i]`, and `c` is for center. The maintainer: "Agree with Vc, Hc." The engine's
    keyword is `Vc=`. Checked on the library's own frame and centers, with jets
    (`scripts/review/sample_variables_check.py`, part E): the value with `Vc`, with `Hc`, and through `D` agree.
  - **Not decided by this:** what a trial point in a line search does (the second half of 4.3); whether the
    four stores' agreement among themselves is also a precondition (the rest of 4.2).
  - This reverses decision 7 of 2026-11-14 in part: reading the value off the frame is no longer parked. Its
    three objections were the missing guard (now the safe-mode check), the missing channel in the geometry
    protocol (now provided), and the library's caching audit (a fact about the present structure).
- **4.2, decided** (2026-11-16, 01:52 for the centers, 02:07 for the four stores).
  - **What agreement means.** A frame describes one tensor several times: `D` over `L`, `D` over `R`, and at each
    site `L` cores to the left, `U`, and `R` cores to the right. A frame the library builds agrees by
    construction; one assembled by hand need not. If the stores disagree, the value comes from one tensor and
    `Jp`, `g`, `JtJp` go through the cores of another, so `g` is not the gradient of `f`, silently.
  - **The library today.** `T3Frame.is_consistent` is a diagnostic that nothing calls automatically. It compares
    the `L` and the `R` reconstruction only, not `U`, and it forms both dense tensors
    (`fv_frame_consistency_residual`); its docstring calls it expensive. Decision 7 of 2026-11-14 left it a
    diagnostic.
  - **The maintainer** (02:07): "This is a precondition checked in safe mode at the moment that a function that
    requires it is called. Runtime safety checks must not go through dense; this will probably blow out the
    users memory. We can check ||A-B|| using the existing non-dense norm machinery. The is_consistent helper
    should be updated to avoid dense as well. Do not call the check when it is not needed; the user may do
    whatever they want with the extra core; if they are not using functions that depend on consistency, then we
    let them be."
  - **So:** a precondition in safe mode, skipped in unsafe mode and under jit like the others; checked only
    where a function needs it; computed through the cores, as the norm of the difference of two tensors in T3
    form, never through a dense tensor. `is_consistent` and `fv_frame_consistency_residual` are to be rewritten
    the same way. With 01:52: centers that are given and do not agree with `(D, L)` are the same kind of error.
  - **Open:** which functions require it. Claude's reading, to be settled with the table of variables (6.4): the
    fitting layer's local model always; `g` when its residual comes from the `z` of the same frame; `z` with
    `Vc` (the centers' agreement); and not an edge variable asked for on its own, nor `z` through `D`.
- **4.3, the trial point, decided** (2026-11-16, 01:59). The maintainer: "Add all three as options, default to
  (b)".
  - The engine does not care: a trial's value is `f` on whatever frame it is given. The policy belongs to the
    optimizers. With F one forward pass over the samples (the plain sweeps and `z`), A one adjoint pass, and `t`
    trials in an outer iteration:

    | Policy | At each trial | On acceptance | Sample work | Needs |
    | --- | --- | --- | --- | --- |
    | a. Raw cores at the trial (today) | F through the trial's own cores | Orthogonalize, then F and A at the new frame | `t F + F + A` | Nothing |
    | b. A frame at every trial (the default) | Retract-and-orthogonalize, then F through its frame with `Vc` | A only: the trial's plain variables are the frame's | `t F + A` | The plain variables of one trial held until it is accepted; `t` orthogonalizations |
    | c. A frame at every trial, the gradient with it | Retract-and-orthogonalize, then F and A together | Nothing | `t (F + A)` | `t` orthogonalizations; nothing held |

  - b saves one forward pass per outer iteration over a, for core-sized work per rejected trial and the memory
    of one trial's plain variables. Under a plan that holds nothing, b saves nothing; c is the one that works
    with nothing held, equal to b when the first trial is accepted and worse than a from the third trial on.
    That A costs about as much as F is Claude's estimate, not a measurement.
  - "Carrying `r` from an accepted trial" (decision 7, point 3) is overtaken: with `Vc`, forming `r` again at
    the frame costs one `z_term` per site.
  - Open: the name of the option and of its three values.
- **4.1 to 4.3.** Decision 7 chose the value through the base point's own cores and parked reading it off the
  frame. Two of the reasons were that the geometry protocol has no channel for the center cores and that the
  library's caching audit had declined it. Those are facts about the present structure. The counts of sample
  sweeps per outer iteration (`t + 2` today, `t + 1`, or `t` with a frame per trial) do not depend on them.
- **4.4, decided** (2026-11-16, 02:12). The maintainer: "Frames must support arbitrary tails. Allow and do not
  check. Functions which require squashed tails should be looked at with suspicion."
  - **Background.** A TT whose end bonds are larger than 1 (tails) stands for the tensor with those two end
    indices summed, that is, contracted with a vector of ones. Squashing sums the end index into the end core.
  - **Put with the question, and not objected to:** the boundary vector is ones, sized by the end bond of
    whatever cores are passed, so tails work everywhere in the engine with nothing special-cased (a requirement
    with a test); and each sweep's start value follows from its grade: ones for plain (with jets, ones at order
    0 and zero above), zeros for dot, tilde-dot and the probe kind's tilde, and the seed times the ones vector
    at the right end for the tilde of apply and entries.
  - **Noted by Claude with the choice:** on a frame with tails, variations of an end core that sum to zero over
    the tail index change nothing, so the tangent coordinates are not unique there. That is the caller's
    business.
  - **Not spoken of:** that the frames the library builds are squashed by default (`squash_tails=True` today).
  - **The reach of his last sentence** (02:16). Claude had noted a library-wide audit of the functions that
    squash or assume squashed tails. The maintainer: "This refactor arc is already too big to add a broader
    squash tails drift audit on top. Keep the squash tails discipline to the functions we touch here and we can
    deal with other ones at a later time." So: every function this refactor writes or touches supports arbitrary
    tails; the rest of the library is left for later.
- **4.4, the earlier record.** Adopted on 2026-11-14 "as the library does it".
- **4.3, after the geometry boundary.** A new function that retracts and builds the frame together may be added.
  The existing retraction and frame construction keep their signatures and behaviour.
- **4.5, decided** (2026-11-16, 02:16). The maintainer: "4.5: Agree with proposal." The projection, the lift
  and the assembly of the exact Hessian product stay in the geometry and are called by the fitting layer. The
  engine returns `g` and `JtJp` as blocks before any projection, takes the direction as `(V, H)` for `Jp` and,
  already lifted, as `(D_dot, L_dot)` for `Kp`. This is what lets tied Tucker factors stay a wrapper around a
  geometry.
- **4.5, what is and is not proposed.** Nothing is proposed for the projection or the lift. The suggestion to
  confirm is that they stay where they are: the geometry owns them and the fitting layer calls them, so the
  sampling layer returns the blocks before projection, takes the already-lifted direction for the curvature term,
  and never needs to know the geometry. That is also why Tucker-factor sharing can stay in a geometry wrapper.

## 5. Functions and layouts

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 5.1 | One function per variable with explicit arguments, seven terms, no graded containers | Decided | |
| 5.2 | One set of step functions for the ragged and the uniform layout. The layout is detected from the structure of the inputs, as low as reasonably possible. Sweeps go through `xscan` on both layouts. Masks never reach a step: the uniform cores are masked once on entry. Shapes are read from the operands | Decided | 1.2 |
| 5.3 | For each variable, are sites scanned or vectorized, what is the default, and which variables admit which? | Reopened | 5.2 |
| 5.4 | The terms are generic over the stack groups through the groups of the contraction engine: a variable is `[S] + T + W + [K] + C + (size,)`, with `T` and `A` the optional order axes of a full and of an affine jet. Each place a term is used has its own literal contraction string, with `K` on the operand that carries the direction, at the call site in the step function. The call is `jet_contract`: `contract` with the order groups multiplying as truncated polynomials, and exactly `contract` at order 0. Every call passes one record of the axis counts of `S`, `T`, `W`, `K` and `C`, so that an operand of the wrong rank raises. The seven terms are seven families of strings **Amended 2026-11-17, 02:23: the site group `S` is removed.** A variable is `T + W + [K] + C + (size,)` at one site, with a leading `(d,)` on a uniform supercore, written as the library writes it; the record of axis counts has four fields, `T`, `W`, `K`, `C`. The maintainer: a uniform supercore is a different kind of thing than a single ragged core, and reusing one operation for both is a code smell; new groups are introduced only with good reason and after deliberation **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** the groups are `J`, `W`, `K`, `C`. `J` is the one jet group, of any number of axes, each operand with its own sizes along them; `A` is removed, its definition of 2026-11-17, 02:31 superseded (an affine variable is a jet of two slots per axis). The record is `GroupLengths(J, W, K, C)`; the output's jet shape is an argument `order` of every call, `jet_contract(subscripts, *operands, lens, order, over_orders)`. The 41 strings become 29. `J` is not a new group: it is the merger of `T` and `A`. | Decided | 2.7, 2.9 |
| 5.5 | The contraction order inside a term is in the plan: the user can specify it and tune it on their problem, and the library provides helpers that choose good orders. Held for the performance discussion (group E): the granularity (per term function, or per place a term is used), the form of the plan's field, and the helpers. Proposed, to be confirmed: how an order reads where the operands are jets | Open | 7.6 |
| 5.6 | The dtype rule is absorbing: jax takes precedence over numpy, and lower precision over higher. The engine creates no array with a default dtype | Decided | |

Notes:

- **5.3.** Decision 4 made the site choice per variable, which stands on an accounting argument. Its default,
  "what each kernel does today", is a reason from history.
- **5.5, reversed** (2026-11-16, 03:32). Claude proposed confirming "no option". The maintainer: "This should be
  in the plan. The user must be able to specify the contraction ordering and tune it on their problem. We should
  also provide helper functions to choose good contraction orderings. What is uncertain is how the contraction
  ordering is specified by the user and passed down through the contractions engine. I do not know the degree to
  which the contractions engine supports einsum paths, or how this interacts with grouped indices. Talk more about
  this."
  - **What the engine does today** (read in `backend/contractions.py`). `contract` solves the group lengths,
    expands each group into single letters, and runs one einsum on the operands as given. On numpy it forces a
    greedy pairwise path, computed on the grouped string and cached, because numpy's own optimizer runs a
    multi-operand contraction with tied flop counts as one loop that skips BLAS. On jax it passes no path, and
    jax's own optimizer chooses. `contract` has no argument for a path.
  - **Checked by running** (jax 0.12, numpy 2.5): `np.einsum` and `jnp.einsum` both accept an explicit path.
    The three orders of `mu_term` give equal values on both. In jax each path lowers to a different pair of
    `dot_general` operations with a different intermediate: `(mu, G)` first keeps an array over the Tucker
    edge, the right bond and the samples; `(G, xi)` first keeps one over the samples and both bonds; `(mu, xi)`
    first is an outer product over the samples, the Tucker edge and the left bond.
  - **Groups do not interact with a path.** A path names operands by position, not by letter, so the same path
    serves the grouped string and its expansion. The library already computes its numpy path on the grouped
    string for that reason.
  - **Open:** the form of the plan's field, its granularity (per term function, or per place a term is used),
    and what the helpers optimize (flops, or the largest intermediate).
- **5.5, granularity and helpers** (2026-11-16, 03:59). Asked for the granularity, what a helper optimizes, and how
  an order reads with jets, the maintainer: "Granularity: per term. Helper: unsure. Needs a design phase. We can do
  several. Flops is the standard (e.g., numpy's optimize_path), but I do not like it because flops often don't
  reflect real world performance. Limiting memory of intermediates is a good one. I think the ideal is to actually
  do some test contractions and measure them. This relates to the "measure" mode we discussed before, for
  automatic performance tuning. Jets: What is the question here?"
  - **Granularity: not decided; moved to the performance discussion** (04:08). Claude read "per term" as per term
    function (one order for each of `mu_term`, `nu_term`, `eta_term`, `H_term`, taken by every use of that
    function in the call) and asked whether one order for every place a term is used was meant. The maintainer:
    "I'm a bit unclear on granularity actually. Let's move that decision to the performance phase". So neither
    reading is decided, and the granularity is taken up with group E.
  - **Helpers: not decided; a design phase of their own.** There can be several. Flop counts are the standard but
    he does not trust them to reflect real performance. A bound on the memory of the intermediates is a good
    criterion. The ideal is to run test contractions and measure them, as one of the candidates of the `measure`
    policy (`EXECUTION_PLANS.md`, "Policies": time the few discrete candidates on one small block and pick the
    fastest that fits). So the helpers belong with 7.6 and the performance discussion of group E.
  - **A fact found while checking** (`scripts/review/contraction_order_check.py`, part 1). The two backends spell
    an explicit path differently: numpy takes a list that begins with `'einsum_path'`, and jax (through
    `opt_einsum`) takes the bare list of pairs and rejects the prefix. So the plan holds a plain tuple of pairs
    of operand positions, and `contract` spells it for the backend it dispatches to.
  - **The question about jets, restated.** An order says which two of a term's three operands are contracted
    first. Decision 2.9 already says part of that for a product of two full jets: one jet is taken one
    coefficient at a time and contracted with the core, and the result is paired with every slot of the other
    jet. So the form fixes "the core first", and what is left free is which of the two jets goes with the core.
    The question was whether that is the same setting as the order in the plan, and what an order that pairs
    the two jets first means above order 0.
  - **Claude's proposal, to be confirmed.** One reading covers every case: an order ranks the pairwise products,
    and a pairwise product of two operands that both carry the order axis is their truncated product, carried
    out as 2.9 says (the scan or the loop). Then:
    - the operand paired with the core first is the one taken one coefficient at a time, so "which jet is
      scanned" is not a setting of its own;
    - an affine operand, order 0 and the block readouts need no rule of their own;
    - the order that pairs the two jets first is defined too. It forms their truncated product, an array over
      the orders, the samples and both bonds, and then contracts it with the core. It gives the same values and
      does the expensive contraction the same number of times, but nothing in it is one order slice at a time,
      so the memory property of the scan holds only for the two orders that take the core first.
    Checked on small arrays (the same script, part 2): the three orders of `eta_term(mu, G, nu)` with two full
    jets, read this way, each give the explicit sum.
- **5.5, the earlier record.** Decision 4 left association out, on the library's own measurement that one
  contraction per term is right on jax.
- **5.6, decided** (2026-11-16, 03:32). Claude proposed that the engine never create an array with a default
  dtype, every array it makes taking the result dtype of the operands of its sweep, and that mixed inputs
  follow the usual promotion. The maintainer: "Hmm. This is ok. The one thing I'm thinking about is what dtype to
  choose if the user supplies arrays of mixed dtypes, which is something the library is supposed to support. In
  this case, I think we can use the absorbing rule where jax takes precedence over numpy and low precision takes
  precedence over high prececision. The jax priority is the numpy vs jax dispatch rule. Lower precision taking
  priority is because the output will not be more precise than its inputs."
  - **So:** with mixed inputs, the result is jax if any input is jax, and has the lowest precision among the
    floating inputs. This is the opposite of numpy's promotion, which goes to the highest. The engine creates no
    array with a default dtype; constants are Python scalars.
  - **The test:** for every output, on all four backends, float32 in gives float32 out, and a mix of float32
    and float64 gives float32.
  - Open detail: which inputs count (the cores, the sample, the data, the weight, the directions, the held
    variables; not an integer index), and where the casting is done.
- **5.4, the prototype** (2026-11-16, 16:05 to 16:50; `scripts/review/term_stacks_check.py`). Not decided.
  - **The shapes it has to serve** (`SAMPLE_VARIABLES.md`, section 7): a variable is `[S] + T + W + [K] + C +
    (size,)`. `S` is the site group, `(d,)` when a map-type operation is done at all sites at once on the
    uniform layout (5.2), else empty. `T` is the optional order axis, `A` the same for an affine variable. `W`
    is any number of sample axes. `K` is the stack of directions, on one operand of a term, a different one
    from term to term. `C` is on every core and every variable, and not on the input `w`.
  - **One contraction string per term serves every placement of the direction.** For `mu_term`:
    `'S T W X C a, S Y C a i b, S W Z C i -> S T W X Y Z C b'`. `X`, `Y`, `Z` are the direction stack as
    carried by the first, second and third operand. Their lengths are not given: each is solved from the rank of
    its own operand, the other groups being known from four static integers of the call (how many axes `S`,
    `T`, `W` and `C` have). So a call site is the formula and nothing else: `mu_term(mu_dot, R, xi, lens) +
    mu_term(mu, H, xi, lens) + mu_term(mu, U, xi_dot, lens)`. A term checks that at most one operand carries a
    direction stack, and raises otherwise. No reshape is made; the library's `contract` does all of it.
  - **The order.** Over the orders a term is the truncated product of its operands' jets (2.9). One operand
    carries the order axis through the contraction as the riding group `T`; the other is taken one coefficient
    at a time. One helper does that for an affine partner (two contractions) and for a full one (`'loop'` or
    `'scan'`); at order 0 it is one contraction and no order axis exists anywhere. `xi_term` and `z_term` have
    no product over the orders: the axis rides. The block readouts `V_term` and `H_term` contract the order axis
    itself (the top coefficient of the product) and sum the samples.
  - **Checked.** Against a reference that uses neither `contract` nor the helpers (every operand expanded to all
    the stacks, each stack flattened to one axis, the truncated products as explicit loops over `np.einsum`):
    all seven terms, the direction on every operand that can carry it and on none, an affine and a full partner,
    both forms over the orders, at orders 0, 1 and 3, with `S` empty or `(2,)`, `W` empty, `(3,)` or `(2, 2)`,
    `K` `(2,)` or `(2, 2)`, `C` empty or `(2,)`: 3912 calls, largest relative error 9.5e-16. A perturbation of
    1e-6 in a term is caught. A dotted step at order 0 makes 3 contractions for its 3 terms. The one string
    gives the result of the four literal strings of the library's style, to the last bit. Under `jax.jit`, in
    float32, three of the terms agree with numpy to rounding and keep the dtype.
  - **Found on the way.** An input `w` (`A + W + (N,)`, no `C`) and a seed `w_tilde` (`T + W + [K] + C + (N,)`)
    cannot be told apart by rank when `C` is empty and there is no `K`. `xi_term` and `V_term` take either. The
    prototype tells them apart by the length of the order axis (2 for an input), which is safe because at
    order 1 the two formulas coincide. A first version without that test gave a wrong `V_term`, and the
    reference caught it. In the engine the caller knows which it has, so two named functions or a flag would be
    more explicit than a test on the shape.
  - **The alternative, the library's present style:** a literal string for each place the direction can sit,
    chosen at the call site (`mu_term(mu, H, xi, direction_on=1)`). It is stricter: a wrong rank raises inside
    `contract`. It needs about three times the strings, and every call site states a placement that its
    operands' names already say. With the one string, an operand of a wrong rank is read as carrying the
    direction; the guard catches two such operands, and the engine's check of held shapes (6.5) catches a
    wrong one coming from outside.
  - **Not covered:** the scan written through `xscan` with a module-level body (the prototype's `'scan'` is the
    same carried shift as a Python loop; `order_scan_check.py` has the `xscan` form for `W` and `C`); the plan's
    contraction order (5.5); masks; the readouts with the samples kept (3.7), which is one more output string
    each; how the four static integers reach a scan body (a memoized factory keyed on them, by the library's
    rule).
  - **To decide:** the one string per term with the direction found from the ranks, or a literal string per
    placement; and whether `xi_term` and `V_term` are told what they are given.

- **5.4, decided** (2026-11-16, 16:24 to 16:26, the prototype rewritten by 16:45). After choosing a literal string
  for each placement, the maintainer was asked two things and answered both with "Yes, this is good".
  - **The strings stand at the call sites in the step functions, through `jet_contract`.** A table inside a term
    function that takes the direction's place as an argument would put the string somewhere else again, against
    his reason for the choice. `jet_contract` is `contract` with one more rule: the order groups `T` (a full
    jet) and `A` (an affine variable) multiply as truncated polynomials. One operand with an order group, and
    the output with it: the axis rides, one contraction. Two operands with one, and `T` in the output: the
    truncated product of 2.9, two contractions for an affine partner, `'loop'` or `'scan'` for a full one. No
    order group in the output: the top coefficient of the product, which is what the block readouts are. At
    order 0 both groups are empty and it is exactly one `contract` call.

    ```python
    def mu_dot_step(mu_dot, mu, xi, xi_dot, R, U, H, len_T, len_C, over_orders):
        t1 = jet_contract('TWKCa,Caib,AWCi->TWKCb', mu_dot, R, xi,     len_T=len_T, len_C=len_C, over_orders=over_orders)
        t2 = jet_contract('TWCa,KCaib,AWCi->TWKCb', mu,     H, xi,     len_T=len_T, len_C=len_C, over_orders=over_orders)
        t3 = jet_contract('TWCa,Caib,AWKCi->TWKCb', mu,     U, xi_dot, len_T=len_T, len_C=len_C, over_orders=over_orders)
        return t1 + t2 + t3
    ```

  - **Every call passes both `len_T` and `len_C`,** and not only the one it strictly needs, so that an operand
    of the wrong rank raises.
  - **What follows.** The seven terms are seven families of strings, not seven functions; this is how the
    "seven terms" of 5.1 are to be read. Every operand's whole shape is in its string, so the reader sees that
    `xi` is affine in a term (`A`) and full in another (`T`). `xi_term` and `V_term` need no flag: a string for
    an input has `A` and no `C` on that operand, a string for a seed has `T` and `C`. The site group `S` is on
    the strings of the map-type operations (xi, z, eta and the readouts) and not on those of the sweeps.
  - **The prototype in this form** (`scripts/review/term_stacks_check.py`, rewritten). The 41 strings of the
    seven families are written out as they would stand at their call sites, and each is checked against the
    reference that uses neither `contract` nor `jet_contract`: orders 0, 1 and 3; `S` empty or `(2,)`; `W`
    empty, `(3,)` or `(2, 2)`; `K` `(2,)` or `(2, 2)`; `C` empty or `(2,)`; both forms over the orders where two
    full jets meet: 2760 calls, largest relative error 4.1e-16, and a perturbation of 1e-6 in any family is
    caught. `mu_dot_step` makes 3 contractions at order 0 and 6 at order 3. `V_tilde_step` runs at all sites at
    once. Under `jax.jit` in float32, a product with an affine leaf, a product of two full jets and a block
    readout agree with numpy and keep the dtype.
  - **Strictness, as measured.** With `len_T` and `len_C` passed, a stray axis on `mu` raises wherever it is
    put. One case is not closed by those two: where `W` and `K` stand together on the only operand that has
    them (a seed that carries `K`, in `xi_term`), a stray axis is read as one more axis of `W` or of `K`. It
    raises once `len_W` and `len_K` are passed as well.
  - **Decided: one record of all the lengths at every call** (16:35). Claude suggested passing the axis counts of
    `S`, `T`, `W`, `K` and `C`, five static integers of the engine's call, as one record `lens` in place of the
    two keywords: `jet_contract('TWCa,KCaib,AWCi->TWKCb', mu, H, xi, lens, over_orders)`. It closes the case
    above and shortens every call; a call no longer shows which lengths it uses. The maintainer: "I'm
    ambivalent regarding lens. Let's go with it." The prototype still has the two keywords; it is to be changed
    with the rest when the code is written.
  - **Not covered:** the scan written through `xscan` with a module-level body; the plan's contraction order
    (5.5); masks; the readouts with the samples kept (3.7), which is one more string each.
- **5.4, the form of the strings: decided, option A** (2026-11-16, 16:18 to 16:24). The maintainer asked to see
  an example call to `contract` for each option, on a contraction the code really uses. Shown on the dotted left
  step (`_sigma_step` in `backend/probing.py`), with every call run and the two options equal bit for bit:
  option A, `contract('TWCa,KCaib,WCi->TWKCb', mu, H, xi, len_C=len_C)` and its two siblings with `K` on the
  carry and on the leaf; option B, the one string `'T W X C a, Y C a i b, W Z C i -> T W X Y Z C b'` for all
  three, with `len_T`, `len_W` and `len_C`.
  - **The maintainer** (16:24): "Option A. The reason is actually so the code is more self-documenting. The
    reader of B has a harder time figuring out what contractions are being done, because they need to infer Y."
  - **So:** a literal contraction string for each place the direction sits, as the library writes them today,
    with the order group added. The one-string form of the prototype is not taken; the prototype stands as the
    check that the terms are generic over the stacks, and is to be rewritten in the decided form.
  - **A correction made by Claude before the decision.** It had said the literal form is stricter. Tested with
    a stray axis of length 1 on `mu`: with the lengths as minimal as the library passes them today
    (`len_C` alone on the middle term) a stray axis in front of `W` is absorbed into `T` and accepted; with
    both `len_T` and `len_C` given at every call, every stray axis raises. Option B accepts a stray axis
    between `W` and `C` as a direction stack. So option A is strict only when every call passes both lengths.
  - **Open:** whether every call passes `len_T` and `len_C` for that reason; and where the literal strings
    stand: at the call sites in the step functions, or in a table inside a term function that takes the
    direction's place as an argument.

- **5.4, how the library does it today** (asked by the maintainer at 03:32; read in `backend/probing.py`). Each
  term of a step is its own `contract` call with its own string, and the string places `K` on the operand that
  carries it. In `_sigma_step`, the dotted left sweep: `'WKCa,Caib,WCi->WKCb'` (`K` on the carry),
  `'WCa,KCaib,WCi->WKCb'` with `len_C` given (`K` on the variation core, where the string alone cannot pin the
  split), and `'WCa,Caib,WKCi->WKCb'` (`K` on the dotted leaf). Each reduces to the two-group result when `K` is
  empty. So the library has no term functions: a term is a literal string at the place it is used.
- **5.1, step functions and the basis** (the maintainer, 2026-11-16, 00:03 and 00:23). Asked whether a step
  function is named by letter and grade alone, with the basis following from the cores it is given, he first
  said: "I would prefer to have separate step functions. It is ok to have lots of small functions." Shown that
  the two would be one recurrence with different arguments, he said: "Oh, I see. There is no need for extra step
  functions if they literally do the exact same things with different arguments." So a step function is named by
  letter and grade only and serves both bases: `nu_step` given `L` cores returns `nu_L`, and `mu_dot_step` given
  `(R, U, H)` returns `mu_R_dot`. The prototype's 22 step functions stand.
- **5.2, decided** (2026-11-16, 03:11). Claude put the rules of 19:50 (below) to the maintainer as four points:
  one set of step functions for both layouts, the layout deduced from the input types; a sweep along the chain
  written once through `xscan`; masks never reaching a step, the uniform cores being masked once on entry;
  shapes read from the operands, and a static value that no operand carries reaching a scan body through a step
  variant and never a closure. His answer:
  "Yes, out the outer level each step function takes in inputs. If the behavior depends on whether the cores
  are ragged or uniform, this is detected from the structure of the inputs themselves, and then they are
  dispatched accordingly. The detection is done as low as reasonably possible. Generally sweeps (e.g., through
  TT cores) work the through the xscan abstraction whether ragged or uniform. For operations which are applied
  simultaneously to all sites (edge variables can be computed all at once; one does not depend on a previous
  one, like Tucker core edge variable contractions and assemblies), the ragged continues through xscan (which
  becomes a loop under the hood), while uniform operates chunked, through the new Plan concept. The chunk sizes
  are decided outside: the user reshapes W batches and tells the engine which batches to chunk over for which
  operations through the Plan. We provide helper functions to the user to reshape and pad their arrays to
  desired chunk sizes. The user can achieve an unchunked scan by reshaping so the effective chunk size is 1."
  - **What this adds beyond 5.2.** For the operations done at all sites at once (the Tucker contractions and
    the assemblies): on ragged cores, a loop over the sites; on uniform cores, all sites at once, chunked over
    the sample axes as the plan says, per operation. Chunk sizes are array shapes the user sets by reshaping,
    with helpers for padding and reshaping (7.4), and a chunk size of 1 gives a scan sample by sample. This
    bears on 5.3 and on group E, which the maintainer wants discussed together as the performance hooks and
    defaults.
- **Rules for how the functions are written** (the maintainer, 2026-11-15, 19:50, said while settling 2.9):
  "The implementation is through the xscan abstraction, which dispatches to a jax scan if jax is used and the
  layer is uniform, and a loop if pure numpy or ragged. In general, things should respect the numpy vs jax dispatch
  rule, and scan type operations should be polymorphic in ragged vs uniform. This may have reasonable exceptions
  for performance optimizations and issues related to masking; though for uniform sampling the pattern is to mask
  first, then compute edge variables with the assumption that masking was already done. For map-type operations,
  uniform and ragged may have different behavior for performance and memory considerations; this is one of the
  things we are helping the user control with the Plan abstraction in this refactor arc. Dispatch at the lowest
  level within reason, use jax if any input is jax and numpy otherwise, write code generically with xscan and
  similar, which are obtained from the get_backend function and may also use the tree_contains_jax function.
  Uniform vs nonuniform is deduced from the input types. Look at other functions in the library for examples, for
  example things in the existing backend/probing.py."
  Read as rules, by Claude:
  - **numpy or jax** is inferred from the inputs: jax if any input is a jax array, numpy otherwise
    (`tree_contains_jax`). The dispatch is made at the lowest level, within reason. No `use_jax` argument.
  - **Ragged or uniform** is deduced from the input types.
  - **Scan-type operations** are written once, generically, with `xscan` and its relatives from `get_backend`,
    and are polymorphic in the layout. Exceptions are allowed for performance and for masking.
  - **Masking on the uniform layout:** mask first; the edge variables are then computed on the assumption that
    the masking has been done.
  - **Map-type operations** may behave differently on the two layouts, for performance and memory. That
    difference is one of the things the plan lets the user control.
  - The existing `backend/probing.py` is the example to follow.
  These bear on 5.2 (one set of step functions), 5.3 (sites scanned or vectorized) and 5.6 (the dtype rule).
  The rows are not marked decided by this; they are to be taken up in group D with these rules as given.
- **The same rules as the library has them** (read on 2026-11-15, 20:00, at the maintainer's request: "the
  prototype is just that; the way the code is written there may be what we are going for, or it may be something
  that was done for expediency"). Read in full: `backend/probing.py`, `backend/sampling_derivatives.py`,
  `backend/common.py` (the scans and maps), `backend/ut3_sampling.py` (its head), and the notes they cite. The
  rules above are already written down in the library: `docs/contributor/uniform_polymorphism.md` (which loops
  are shared between the layouts, and masking once) and `docs/contributor/scan_body_principles.md` (how a scan or
  map body is written). Those two notes, not this folder's prototypes, are the reference for the form of the code.

  | Loop | Ragged layout | Uniform layout | Example in the library |
  | --- | --- | --- | --- |
  | A sweep along the chain | `xscan`: a Python loop | `xscan`: `jax.lax.scan` with jax | `compute_mu`, `compute_sigma`, `_adj_sweep_scanned` |
  | A per-site map in the plain kernels, and every lift | `xmap`: a Python loop | no map: one contraction with a leading `d` | `compute_eta`, `assemble_z`, `assemble_z_jets` |
  | A per-site map whose step has a product of two full jets | `xmap`: a Python loop | `xmap`: `jax.lax.map`, one site at a time | `compute_eta_jets`, `compute_deta_jets`, `compute_dxi_tilde_jets`, the TT assembly |
  | The scan over the orders inside one site | a Python loop | `jax.lax.scan` with jax | `_eta_jets_core(data, is_uniform)` |
  | Chunks of samples | none: the unchunked assembly | dynamic slices inside `jax.lax.scan`, with jax, in two assemblies only | `_wchunked_sum` |

  - **Dispatch.** Every function begins `use_jax = tree_contains_jax((its inputs))`, `is_uniform = is_ndarray(a
    core argument)`, `xnp, xmap, xscan = get_backend(is_uniform, use_jax)`. A body re-derives `xnp` from all of
    its own arguments.
  - **Masks.** The kernels have no mask logic. The wrappers in `backend/ut3_sampling.py` mask the supercores
    once, pack the vectors, call the shared kernel, and re-mask a core-shaped output. This is the library's
    present answer to the second half of 5.2.
  - **Bodies.** Module-level functions that close over nothing; statics are read off shapes; operands that do
    not change ride the carry. A static that no operand carries reaches a body through two step variants
    (`_eta_jets_step_uniform`, `_eta_jets_step_ragged`) or a memoized factory keyed on hashable scalars
    (`_tt_variation_jets_step(n_probe, sum_over_probes, is_uniform)`). A setting of the plan would reach a body
    the same way, which is why the plan has to be hashable.
  - **Dtype.** A start value is built in `xnp.result_type` of the sweep's operands in the jet kernels and the
    two-hole; the plain sweeps (`compute_mu`, `compute_sigma`, `compute_tau_tilde`) use the default dtype.

  What the reading bears on. Claude's observations, none a decision:
  - **5.3.** The site option is a choice only where `xmap` is a real map: on the uniform layout with jax. The
    library today has both behaviours there: vectorized over the sites in the plain kernels and the lifts,
    one site at a time where a step has a product of two full jets.
  - **2.9 with 5.3.** On the uniform layout the setting for the orders and the setting for the sites combine.
    The lever the library's own record names for the small-problem latency of the scanned kernels
    (`docs/contributor/deferred_and_rejected.md`) is one of the combinations: one scan over the orders whose
    body is vectorized over the sites. `scripts/eta_axes.py` timed four combinations for the eta combine, but
    its "vectorized over orders" is the dense form, not the loop with slices.
  - **A question for group E.** By the rule as stated, a scan is real only with jax on the uniform layout. Is a
    scan over chunks of samples (7.1) real on ragged cores with jax? The sample axis has the same length at
    every mode on both layouts. Decision 1 of 2026-11-14 kept a host loop "only for numpy and for streamed
    data", and `levels_jax_check.py` ran chunk scans on ragged cores under jit. The library's present chunking
    engages on the uniform layout with jax only.
  - **What the explicit prototypes are.** `jet_steps_check.py` and its relatives are ragged, numpy, with Python
    loops and one `W` axis. They establish the algebra and the names. They are not the form of the code.
  - **One difference that matters for the target.** The prototype forms every tilde variable at every order and
    takes the top coefficient of full blocks at the end. The library forms only what is read: `dxi_tilde` (the
    target's `eta_tilde`) in its two read rows only, because it is paired with the affine input `(w, p)` and
    nothing else, and each gradient block directly, summed over the orders. The row of 2.9 covers the block
    readouts. The two rows of `eta_tilde` are not recorded anywhere, and they bear on 3.3 and 6.4: if
    `eta_tilde` is public, the question is whether it is those two rows or all of them.

## 6. The interface

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 6.1 | `sampling_variables` is a public backend function. It returns a tuple aligned with `want`, and held variables are keyword arguments | Decided | |
| 6.2 | The arguments of `sampling_variables`: `kind`, `frame`, `sample`, `want`, then `order`, `plan`, `y`, `weight`, the direction `(V, H)`, the lifted direction `(D_dot, L_dot)`, `Vc`, whether `g`, `JtJp`, `Kp` are summed over the samples, and the held variables by name. Decided up to the names of three keywords **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** `order` is a tuple, the jet shape; the sample is `ww` (input jets) or `index`; there is no `pp`. | Decided | 2.8, 3.4, 4.5 |
| 6.3 | The public functions of the sampling layer: the engine; the jet-to-dual-jet helper and its inverse; the data preparation functions; and one function behind each frontend method, calling the engine. Every per-variable kernel, every `*_derivatives` function and the sweep functions go. The step and term functions are private | Decided | 1.1 |
| 6.4 | Every variable of a kind's graph, at every grade, can be wanted and held. The table of variables (name, shape, what it reads, which arguments it needs) is written out in the docstring and the documentation, the engine declares its graph as data, and a test checks that the two agree exactly | Decided | 3.1 to 3.4 |
| 6.5 | The engine trusts a held variable and checks only its shape, the order included. Whoever holds it vouches for it. The caller may give the engine whatever they want, modified variables included | Decided | |
| 6.6 | The plain and the jetted sampling function are one function: `probe(ww, pp=None, order=0)`, returning `T + W + C + (Ni,)`. This holds for the frontend methods of the flagship classes and for the backend function behind each, which uses the engine **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** entry 118 reverses the `pp` keyword: `probe(ww, order=())`, returning `F + W + C + (Ni,)`, with `ww` input jets; the call that was `probe(ww, pp, order=3)` is `probe(affine_jet(ww, pp), order=(3,))`. The entries functions have no `order` (entry 120). | Decided | 2.6, 2.7 |
| 6.7 | Preparing the data and creating weights is done by standalone functions that the user calls before fitting, when the batch is made. The fitting functions have no keyword for it: not for derivative data (2.3) and not for the equalizing weight (amended on 2026-11-16; opt-in keywords had been allowed). Should such a keyword be added later, it may do only preparation that changes the loss alone; anything that changes the unknowns or belongs to the problem stays standalone | Decided | 2.5 |

Notes:

- **6.2, decided up to three names** (2026-11-16, 02:34). The maintainer, on the sketch below: "Perfect. sketch
  looks good". It is assembled from the decisions taken piece by piece; it is a sketch and not final spelling.

  ```python
  def sampling_variables(
          kind,                 # 'probe' | 'apply' | 'entries'
          frame,                # (D, U, L, R)
          sample,               # ww if order == 0 else (ww, pp); the index for entries
          want,                 # names; the return is a tuple aligned with it
          order=0,              # highest Taylor order.  T=(order+1,) if order else ()
          plan=None,            # how the pass is carried out
          y=None,               # the data; needed for r and what reads it
          weight=None,          # ones if None; acts on f, g, JtJp, Kp, Jp_norm_sq
          direction=None,       # (V, H); needed for the dot grade
          core_direction=None,  # (D_dot, L_dot); needed for Kp
          Vc=None,              # Tucker centers; the value is read off the frame if given
          sum_samples=True,     # on g, JtJp, Kp
          **held,               # held variables, by name
  ):
  ```

  - Where each line comes from: `want`, the return and the held variables, 6.1 and 6.5; the bundled `sample`
    and `order` deciding, never the type, the part of 6.2 decided on 2026-11-15 (04:22) and 6.6; `y` and
    `weight`, 3.6; `Vc`, 4.1; `sum_samples`, 3.7; the two directions, 4.5.
  - **Open inside it:** the names `direction`, `core_direction` and `sum_samples`, which are Claude's
    placeholders (the library's flag today is `sum_over_probes`); `kind` as a string (9.2); `plan` as an
    argument here (7.3).
  - A missing argument is a structural error: `Kp` wanted without the lifted direction, or `r` wanted without
    `y` and without a held `r`.
- **6.4, the table, decided** (2026-11-16, 02:37). The maintainer: "Let's go with the second. Table written out
  and tested against the declared graph." (That every variable can be wanted and held was confirmed on
  2026-11-15 at 21:57.)
  - `sampling_variables` returns a tuple aligned with `want`, so no signature carries the shapes: the table is
    the contract.
  - **Not taken:** a docstring built from the declared graph at import time. It cannot drift, but the table is
    then not in the source.
  - **Taken:** the table written by hand, the graph declared as plain data (a tuple of records with a name, a
    step function, the names it reads and the arguments it needs), and an exact test that the two agree row for
    row. This keeps the code statically readable and is an exact test of a structural thing, the two ground
    rules that apply.
  - **Two more tests the table makes possible** (Claude's, put with the question): asking for one variable
    computes only what the table says it reads; and the short form of a step equals a slice of the full one
    (wanting never changes a value).
- **6.5, decided** (2026-11-16, 02:31). The maintainer: "Agree with proposal. Check shapes, trust values. Also,
  I want to leave it open to the user to apply the algorithm to whatever they want. There may be situations where
  applying the sweeping algorithms to modified variables could be useful."
  - A held variable is right only for the point, the samples, the data and the order it was made with, and
    the engine could check its values only by computing them again. So a held variable is an input with a
    meaning, like the frame or the sample. Shapes are checked always, and a wrong number of coefficients for
    the order is a structural error. There is no numerical check, in safe mode or otherwise.
  - In the fitting layer the local model that computed the variables holds them with its frame, sample and
    data, so a user of that layer cannot mix them up. Someone calling the engine directly vouches for what they
    pass, and by the maintainer's second sentence that freedom is intended: held variables are also the way to
    run the sweeps on variables the caller has modified.
  - Not taken: a spot check in safe mode that recomputes a held variable for one sample.
- **What "public" means for a variable, under discussion** (2026-11-15, 21:35). Not decided. It bears on 3.3,
  6.3, 6.4, 6.5 and 3.4.
  - **The maintainer's take.** "In the current codebase, there are handcoded functions that implement the
    computation of every edge variable, in addition to functions that do higher level operations like probing T3s
    and tangent vectors, or applying the transpose of the probing map. In the new codebase, the computation of
    edge variables is done through the sample variable engine, where you ask for what variables you want and
    supply what variables you have. In this sense, there is no need for individualized functions for computing
    each edge variable, since the engine does it better and more generally. This is my take; how do you see it?"
  - **Claude's answer: agreed, and it gives the word one meaning.** A variable is public when the engine accepts
    its name, in `want` and as a held keyword. It is not public by having a function of its own. The public
    functions of the sampling backend are then the engine and the operations of 6.6 that call it. The step and
    term functions are the engine's private implementation.
  - **Reasons, from the design and not from today's code.** (1) A function per variable makes the caller walk
    the dependency graph, which is the engine's knowledge. (2) Functions composed by the caller cannot follow the
    plan: each returns its variable for every sample, and the plan's memory bound comes from forming a chain
    inside one chunk and dropping it. So the composable route and the fast route would be different routes.
    (3) The count: today 46 of the 111 public names in `probing`, `sampling_derivatives`, `apply` and `entries`
    are per-variable functions (`compute_*`, `assemble_*`, `build_*`), plain and jetted. (4) The library's razor
    for the backend is met by the engine: a user of raw `.data` finds one function and asks it for `mu`.
  - **What a name promises, proposed as the test for a name being public.** A meaning that does not depend on the
    route the engine took (so a held value may come from anywhere); one shape; and a use outside: it is an output,
    or it is read by an output and is worth holding from one call to the next. By this test the residual-seeded
    tilde variables that `Kp` reads are public, and `eta_tilde` is not (only `g` reads it, once), which leaves the
    engine free to form its two read rows only. The tilde variables seeded by `Jp` and the dot variables of `Kp`
    stay internal, as in the review.
  - **Costs and conditions.** The shape contract of each variable has no signature to sit on, because the return
    is a tuple aligned with `want`: the table of variables in the docstring (6.4) becomes the contract and has to
    be generated and tested. Asking for one variable must compile to the program a hand-written chain would give.
    The engine has not been prototyped; only the step functions have.
  - **What it moves to 3.4.** With no function per variable, a user cannot assemble a new quantity by hand. If
    users may define kinds, the terms and steps are building blocks for the author of a kind, which is a second
    meaning of public, for a second audience.
  - **A consequence for 6.5, noted.** If `r` can be held, the transpose applied to any array of the type of `z`
    is `want=('g',)` with that array held as `r`. A held variable is then an input with a meaning, and its
    validity is the caller's.
- **What "public" means, the maintainer's two objections** (2026-11-15, 21:44). Not decided.
  - **His words.** "First, the edge variables like eta_tilde are themselves valid mathematical quantities with a
    definition (jetted adjoint with respect to a top level order) that user may want. This particular example
    seems unlikely, but I can easily see situations where the user would want, say, mu. And I don't have a clear
    sense of where to draw the line, which suggests that we provide all of them. Second, the shape contract has
    some nuance when it comes to jets, particularly since we are planning to unify the jetted and unjetted
    variables, so that the unjetted variable is just the jet of order zero. Perhaps this is fine given our
    definition of the optional T index group."
  - **Claude's answer to the first: the third part of the test is withdrawn.** "A use outside" drew a line that
    has no principle behind it. Every variable of a kind's graph, at every grade, can be wanted and held. What a
    name promises is then two things: a meaning that does not depend on the route, and one shape. Whether a
    variable is worth holding is guidance in the table, not a condition for having a name.
    - **A rule this needs:** wanting a variable never changes any value; not wanting it lets the engine skip
      work. `eta_tilde` is the case in point: wanted, it is formed at every order; not wanted, only the two
      slots that `g` reads are formed. Those are two forms of one step, and the short one has to equal a slice
      of the full one, by test.
    - **The seeds (3.3).** A tilde variable is linear in its seed. If `r` can be held, one name per tilde
      variable is enough: the copy seeded by `Jp` is the same name in a call with `Jp` held as `r`. Then "which
      seeded copies are public" has the answer "all, under one name". This rests on 6.5.
    - **Still without a name:** the `eta`-like variable through `L` that `z` reads, and the dot variables of `Kp`,
      which belong to the lifted direction. To be named or given a route at 6.4. Not checked. (Named at 23:23:
      `eta_D`; `mu_dot`, `xi_dot`, `nu_L_dot`, `eta_D_dot`, and `eta_D_tilde_dot`.)
  - **Claude's answer to the second: the optional group `T` carries it, with three things the table has to
    say.** Every variable has the shape `[d] + T + W + [K] + C + (leg,)`, with `T = (order+1,) if order else ()`,
    so at order 0 it is today's plain variable, and the rank is declared by `order`, never read off an array.
    - **Affine variables do not have `T`.** The input `w + s p`, `xi` and `xi_dot` (`xi_dot`: now `xi_U_dot`; the
      raw `xi_dot` too) have two coefficients at every order above 0 (catalog, point 6; 2.6; 2.9). Their order
      group is a second optional group, two slots or absent. The library stores `xi_jets` this way. The prototype
      pads them to `T` with zeros, which is expediency. Claude's reading; not said by the maintainer.
    - **The tilde grades have the same `T` with the order leg turned** (2.4), so the table needs a column for
      the orientation; the shape does not show it.
    - **A plain jet cut at a lower order is the jet of that order; a tilde jet is not** (catalog, point 4).
      This bears on what a held variable is valid for (6.5), not on the shape.
- **Public, two points confirmed** (2026-11-15, 21:57). The maintainer: "Confirmed on both counts."
  - Every variable of a kind's graph, at every grade, can be wanted and held. Wanting a variable never changes
    any value; not wanting it lets the engine skip work.
  - The affine variables (the input `w + s p`, `xi`, `xi_dot`; now `w`, `xi`, `xi_dot`, `xi_U_dot`:
    `SAMPLE_VARIABLES.md` section 7) carry a two-slot group of their own in place of `T`: two slots at every order
    above 0, absent at order 0.
  - This settles the first half of 6.4 (which variables are public: all of them) and the test of 21:35 is
    reduced to its first two parts. The rest of 6.4 (the table generated from the graph), 3.3 (one name per
    tilde variable, the seed being whatever `r` is) and 6.5 are not confirmed by this.
- **The variables that still had no name, and the naming that followed** (2026-11-15, 21:57 to 22:41).
  Deprecated: superseded on 2026-11-15 at 23:23 by target decision 9.6. The note that stood here is in
  `NOTATION_HISTORY.md`, split by topic:
  - 21:57: Claude listed six variables without a name (the maintainer's words of 21:57 and 21:59 with it); all
    were named at 23:23. `NOTATION_HISTORY.md` section 3.2.
  - 22:11: `zeta` for the down variable through the base point's cores; it became `eta_D` at 23:23.
    `NOTATION_HISTORY.md` section 4.
  - 22:11: options A and B for the dots of `xi` and `mu` along the raw core direction; neither was taken: the
    dots are `xi_dot`/`mu_dot` (raw form) and `xi_U_dot`/`mu_R_dot` (frame's form). `NOTATION_HISTORY.md`
    section 5.
  - 22:24 to 22:41: the naming was rethought from first principles; the result is 9.6 (`SAMPLE_VARIABLES.md`
    section 2). `NOTATION_HISTORY.md` section 6.
- **6.3, decided** (2026-11-16, 02:40). The maintainer: "Confirmed. Exactly what I intend." The list, for the
  probe kind; apply and entries follow the same pattern.

  | | Functions | Why |
  | --- | --- | --- |
  | New | `sampling_variables` | The engine (6.1) |
  | New | The helper that turns a jet into a dual jet, and its inverse | 3.6 |
  | New | The five data preparation functions | 6.7 |
  | Stay, each now calling the engine, with the merged signature of 6.6 | `t3_probe`, `tv_probe`, `tv_probe_transpose`, `t3_probe_corewise_transpose`, `t3_probe_corewise_hessian` | One backend function behind each frontend method |
  | Stay as they are | `t3_probe_ambient_transpose`, `dense_probe` | They do not sweep over a frame |
  | Go | Every `compute_*` and `assemble_*` | The engine returns the variable by name |
  | Go | Every `*_derivatives` function | Merged into the plain ones by 6.6 |
  | Go | `tv_precompute_probe_frame_sweep` and the `*_from_sweep` functions | A "sweep" is now a set of held variables |
  | Private | The step and term functions | They are the engine's implementation |

  - A user who wants to run the sweeps on their own arrays does it through held variables (6.5), not by
    calling a step.
  - The old kernels are not kept as adapters: that was for compatibility.
  - **Not settled by this:** which module each function lives in, the names of the new helpers, and the uniform
    twins (group D).
- **6.3, the earlier record.** The review kept the old kernels as public adapters and the new names as a second vocabulary. That was
  compatibility. With one vocabulary, the near miss between the new `xi_tilde` and today's `dxi_tilde` exists
  only during migration.
- **6.6.** Raised by the maintainer on 2026-11-15. It was a goal of his when he first tried to merge the jetted
  and plain functions. His condition: consider it again if it can be done cleanly; if it needs complicated
  hackery or is awkward, do not force it. The issue he met: how to tell whether an axis is part of the batch or is
  the derivative-order axis.
- **6.6, the library's record.** `deferred_and_rejected.md` lists "a kwarg-unified derivative interface"
  (`probe(..., order=)`) as a locked rejection, for four reasons: the output rank would depend on whether a
  keyword was passed; that fights the shape-comment-as-contract style; it is worse for the transposes, whose
  residual input rank becomes context-dependent; and a named `*_derivatives` method documents an advanced feature
  that would otherwise be buried.
- **6.6, a precedent in the library.** The batching convention already has groups whose rank varies and is never
  read off an array alone: the `K` and `C` split of a tangent is "recovered, not stored", from the pairing of the
  variations with the frame.
- **6.6, where the order axis is.** Today the derivative methods return `(order+1,) + W + K + C + ...`, order
  outermost, and on the uniform packed layout `(d,) + (order+1,) + W + C + (N,)`, order just after the site axis.
  The plain methods return the same shapes without that axis. The idea under discussion for 6.6 moves nothing:
  the order axis stays exactly there when the order is above 0 and is absent at order 0, as in the plain methods
  today.
- **6.6, a prototype of the signature** (`scripts/review/merged_frontend_prototype.py`): `probe(ww, pp=None, order=0)`
  returning `T + W + C + (Ni,)`, and the corewise transpose with the same three arguments. `pp` is given exactly
  when `order >= 1`. The order is declared, never read off an array; in the transpose it is also recovered by
  pairing the residual with the sample, and the two are checked against each other. The bodies wrap the library's
  existing methods, so the shapes are real: order 0 returns exactly `x.probe(ww)`, and order 3 times `t!` equals
  `x.probe_derivatives`.
- **6.6, decided** (2026-11-15, 04:04). The maintainer, on the prototype: "Perfect. This is good for a frontend
  method of the flagship classes, and a backend function which uses the engine under the hood." What this fixes:
  the order is declared by `order` and the dispatch is on it, not on the type of an input; `pp` is given exactly
  when `order >= 1`; the return carries the optional axis `T`, outermost; the values are Taylor coefficients; the
  transposes take the same `pp` and `order`, and their residual carries `T`. The `*_derivatives` method names go.
  The entry "a kwarg-unified derivative interface" in the library's `deferred_and_rejected.md` is reversed at the
  target, and that file has to say so when the refactor is done.
- **6.6, Claude's reading of the reach, to be corrected if wrong.** The prototype showed `probe` and its corewise
  transpose on `TuckerTensorTrain`. The same pattern is taken to cover `apply` and `entries`, the transposes, the
  `T3Tangent` methods, and the uniform twins. Which backend functions exist beside `sampling_variables` is
  question 6.3, still open; 6.6 fixes only that each one has the merged signature.
- **6.2, the sample and the direction, decided** (2026-11-15, 04:22). 6.2 as a whole stays open. The maintainer
  asked whether the sample can be of type `Union[ww_type, Tuple[ww_type, pp_type]]` with a dispatch on it. The
  answer he agreed to: that is the type, but the dispatch is on `order`, never on the type. Two reasons. A pair
  says the call is jetted but not how many coefficients, so `order` is an argument either way. And the type alone
  is ambiguous: a ragged `ww` with `d = 2` is a 2-tuple of arrays, and so is a jetted sample `(ww, pp)` on the
  uniform layout. With `order` deciding, the function unpacks `ww, pp = sample` when `order >= 1` and takes
  `ww = sample` otherwise, and inspects nothing.
  - **Fitting functions take the bundled sample**: `sample,  # ww if order == 0 else (ww, pp)`, with `index` in
    place of `ww` for entries. The sample is already opaque there and passes through many functions; a separate
    `pp` would have to follow it everywhere, with a second one beside `val_sample`.
  - **Frontend sampling methods take `pp` as its own argument**, as in 6.6.
  - `sampling_variables` is kind-generic and takes the bundled sample; the backend function behind each
    frontend method takes `ww` and `pp` separately, like its method. (Claude's reading; approved by the
    maintainer at 04:26.)
- **6.6, the wording of the comment.** The maintainer: the comment on `order` should relate it to `T`, and
  `T=(order+1)` is wrong at order 0. The form now in the prototype is a conditional written as Python, on the
  `order` line, with the return line using `T` like any stack letter:
  `order: int = 0,  # highest Taylor order.  T=(order+1,) if order else ()`. Approved at 04:26.
- **6.7, amended on 2026-11-16 (15:12 and 15:44): the fitting functions have no preparation keywords; see the
  note "The oracle is a description and the data are an argument" in section 8.**
- **6.7, decided** (2026-11-15, 16:33 to 17:00). The maintainer's lean of 16:33, confirmed at 17:00 together with
  Claude's rule for what a keyword may do. The three classes of preparation are under 2.5. With the default of
  2.5 doing nothing, every such keyword is opt-in. Which standalone functions exist is part of 6.3.
- **6.7, the equalizing weight, under discussion** (17:00). The maintainer: the rule for a zero or nearly zero
  block "is an open research question. For a default, let's think of something simple. What do you think about a
  filtered inverse like 1/max(x, floor), where floor is a user-specified parameter that defaults to something
  like 1e-6? What I am concerned about is that for input data in a bad length scale, the norms are expected to
  vary hugely across derivative order, often way more than 1e-6, and this is expected and not a sign of numerical
  zeros. So, perhaps the guidance is that one should adjust the length scales first, before equalizing". Claude's
  answer, awaiting his reply: yes to the filtered inverse, with the floor relative to the largest block,
  `1 / max(x, floor * max(x))`, because an absolute floor depends on the units of the data and equalizing is
  meant not to. No special case for a block that is exactly zero; the user sets a weight to zero to mask a block.
  His guidance holds in a precise form: rescaling the direction `p` changes nothing under equalization except which blocks
  reach the floor, so doing it first is what lets the floor separate true zeros from small legitimate blocks; the
  rescaling that repairs a badly conditioned fit is the other one, of a mode's input coordinates. Suggested: one
  function for the block norms and one for the filtered inverse, since the norms are also the diagnostic of the
  length scale.
- **6.7, the equalizing weight, decided** (2026-11-15, 17:04). The maintainer: "Yes, agree with broadcasting
  decision (exact shape only), relative floor, and two-function split."
  - Two standalone functions. One returns the norm of each block of the data, in the shape of the weight. The
    other is the filtered inverse of those norms, `1 / max(x, floor * max(x))`, with `floor` set by the user and
    1e-6 by default. The floor is relative to the largest block.
  - A block that is exactly zero has no special case: it gets the floor weight. To mask a block the user sets
    its weight to zero in the array returned. (Part of Claude's proposal; the maintainer agreed to the relative
    floor without naming this point.)
  - The norms are also the diagnostic of the length scale. The guidance is to adjust the length scales first and
    equalize after.
  - The weight's shape is exact: a mode dimension of 1 is not broadcast (confirmed).
- **6.7, a required example** (17:04). The maintainer: "let's make sure to provide an example with badly scaled
  data, which shows the bad fit as given, and then a good fit after we process the data with these functions."
  A deliverable of the refactor. Not prototyped yet.
- **6.7, a correction by Claude to its own class (b).** For the probe kind, rescaling a mode's input coordinates
  is not loss-neutral: the data of that mode live in the same coordinates, so they are rescaled too, by the
  inverse, and the components of that mode's residual are reweighted. T3Polynomial's record says the same of
  `kappa` on its input modes. The class still "changes the unknowns and comes with an inverse"; for probe it also
  changes the loss. The example has to state this.
- **6.7, the example prototyped** (2026-11-15, 17:10; `scripts/review/bad_scaling_example.py`, against T3Toolbox
  2026.5.0). A tensor with 4 modes fitted from probe jets of order 3 by the library's Newton solver, from the zero
  tensor, with default settings and exact data. The data are badly scaled in two ways: the coordinates of the
  sample vectors span four decades, and the directions `p` are 100 times too short. The error is that of the fitted
  tensor applied to fresh samples in the original coordinates.

  | Preparation | 8 samples | 200 samples |
  | --- | --- | --- |
  | As given | bad (87) | bad (1.8) |
  | Equalizing weight only | bad (17) | bad (1.9) |
  | Coordinates only | bad (3.3) | good (9e-11) |
  | Coordinates, then the direction | good (1e-09) | not run |
  | Coordinates, then equalizing weight | good (2e-10) | good (4e-12) |
  | Coordinates and direction, then equalizing weight | good (3e-10) | not run |

  The numbers in brackets are from one run and are not reproducible to the digit; the script's checked output
  says only good or bad. With 8 samples the same pattern held for four seeds.
- **6.7, what the prototype shows.** Claude's reading.
  - The required example exists in prototype: bad as given, good after preparation.
  - **The equalizing weight alone does not rescue this fit.** What decides it is the rescaling of the
    coordinates, which changes the unknowns. The equalizing weight is needed as well when the higher orders are
    needed to determine the tensor (8 samples), and not when the order-0 data determine it alone (200 samples).
  - Rescaling the direction and equalizing are interchangeable here, as predicted: either works with the
    coordinates, and doing both gives the result of equalizing alone.
  - The bad fits also took longer: 14 to 20 seconds at 200 samples, against under 1 second for the good ones.
  - **The example needs more functions than the two decided.** Besides the block norms and the filtered inverse
    it uses a rescaling of the direction `p`, a rescaling of a mode's coordinates with its inverse on the fitted tensor,
    and two estimates read off the data (a length for the directions from the block norms, a coordinate scale from the
    samples). Which of these the library provides is part of 6.3, not decided.
  - Limits: exact data, one problem, the ragged layout, numpy, and today's derivative data. For probe the
    coordinate rescaling also changes the loss, so its effect here is not conditioning alone.
- **6.7, a word withdrawn** (2026-11-15, 17:47). Claude had written "the step" for the direction `p` of the input
  jet `w + s p`, and "rescaling the step" for multiplying `p` by a length and slot `t` of the data by that length
  to the power `t`. The maintainer asked: "What is the step rescaling? Is this the newton step length or something
  else?" It is not the Newton step. The word also collides with the step functions of 5.1. It is replaced by
  "direction" throughout, and the prototype's function is `rescale_direction`.
- **6.7, two more functions decided** (2026-11-15, 18:32). The maintainer: "Coordinate rescaling and direction
  rescaling become library functions." So the standalone preparation functions decided so far are five: the block
  norms; the filtered inverse of the norms; the rescaling of the direction `p` (loss-only); the rescaling of a
  mode's coordinates, which changes the unknowns and, for probe, the loss; and its inverse on the fitted tensor.
  Not decided: the two estimates the prototype reads off the data (a length for the directions from the block
  norms, a coordinate scale from the samples), the names, and where the functions live (6.3).

## 7. Memory and execution

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 7.1 | A plan is an ordered tuple of scans over axes of the sample stack, each with a `hold`, plus a top-level `hold`. Since 2.9 it also carries how a product of two full jets is taken over the orders | Decided | |
| 7.2 | What are the defaults: the default plan, and what a model holds between calls? | Reopened | 7.3 |
| 7.3 | Where does the plan live: in the model, or as a solver parameter passed to each operation? | Reopened | 1.1 |
| 7.4 | Chunk sizes are array shapes: the user pads and reshapes, with helpers | Decided | |
| 7.5 | How are padded samples made inert for entries? | Open | 7.4 |
| 7.6 | What is automatic sizing, given that confirming a size by compiling is costly? | Open | 7.1 |
| 7.7 | What does the target promise about sharding, and who owns device placement? | Open | 7.1 |
| 7.8 | `chunk_size` leaves the sampling kind | Decided | |

Notes:

- **7.1, one more setting** (2026-11-15, 19:32). By 2.9 the plan also says how a product of two full jets is
  taken over the orders: a scan with a carried shift (the default) or a loop with slices. The plan stays what it
  was: settings that change how a computation is carried out and never what it computes. What the default plan
  is otherwise is still 7.2.
- **7.2.** The default plan was defined as "today's behaviour". The lean default for what the model holds was the
  maintainer's choice between two options framed against today's behaviour.
- **7.3.** Decision 3 put the plan in `LocalModel` because threading it through about 25 optimizer functions was
  "too much". That is a cost-of-change reason. The maintainer has also said that a chunk size is conceptually a
  solver parameter, not part of the local model.
- **7.5.** The 0/1 mask was accepted with the maintainer's reservation on record.

## 8. The fitting layer

By 1.1 this section is in scope.

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 8.1 | `LocalModel` is the geometry, the oracle, the batch and the regularizer put together at one frame, and the holder of the local information at that point. Its own `gn_quadratic` and `hvp` are built on the fused `Jp_norm_sq` and `JtJp`. `residual` and `jacobian(p)` stay on it, as optional outputs of an oracle. It stores only what the plan's `hold` keeps; the rest is reachable through `model.oracle`, `model.batch`, `model.frame` and `model.held` | Decided | 1.1, 6.1, 8.2 |
| 8.2 | The fitting layer's open extension point is an oracle (`Oracle`): a frozen description with no arrays, and one operation. At a frame and for a batch it returns the wanted ones of `f`, `g`, `JtJp`, `Jp_norm_sq` (required) and `Kp`, `Jp`, `r` (optional), the blocks before projection, in the engine's vocabulary. The data are an argument: a batch of arrays. The built-in oracle is least squares: it holds the kind and the order, its batch is the sample, `y` and the weight, and its operation is one call of `sampling_variables` | Decided | 3.4 |
| 8.3 | The optimizers' contract: `newton_cg(geom, oracle, batch, x0, regularizer=None, ...)`. An optimizer touches a `Geometry`, an `Oracle`, a `Regularizer`, a batch it never looks inside, and the `LocalModel` built from them; there is no `Problem` at the seam. The oracle's one operation takes the batch, `want` and `held` and returns the values and `held`; a model is built once and never changes. A minibatch is a batch, scaled so that the misfit on it estimates the full one; the default draw keeps the drawn indices in it. The units of compilation are five phases (build, gradient, CG solve, trial, stochastic step), with the outer loop on the host; a compiled program's preconditions are checked on the host in safe mode. Which phases are compiled by default is left to 7.2 **Amended 2026-11-16, 19:30 (the plan's O-6):** "checked on the host" means by the frontend, before it calls the backend; the backend optimizers check nothing | Decided | 1.1, 8.1, 8.2, 7.2 |
| 8.4 | The display and the diagnostics read from the oracle and a batch. `r_block_sumsq` (at the model's frame) and `y_block_sumsq` (independent of the frame, computed once by the display) are optional oracle outputs in the shape of the weight; nothing is reduced over modes or orders, in the display or in the stored record. A validation set is a batch given to the same oracle. The trust-region floor is the optional output `f_resolution`, and `eps |f|` without it. An oracle without block sums gets the header line and no table. Nothing is converted for the display | Decided | 3.7, 8.2 |
| 8.5 | Shared sample vectors: in the target or not? | Parked | |

Notes:

- **The oracle is a description and the data are an argument** (2026-11-16, 15:26 to 15:44). This rewords 8.1 to
  8.4 and amends 6.7. **The notes that follow in this section were written before it:** where they say that the
  oracle holds the sample, the data or the weight, that a minibatch or a validation set "is an oracle", or that
  `Problem` bundles the geometry, the oracle and the regularizer, read them as reworded here.
  - **How it arose.** Asked whether the equalizing weight should be a keyword, the maintainer asked for a
    prototype signature (15:26): "Can you show me a prototype function signature for the newton_cg optimization
    function? I am getting confused about what is getting passed in at the seam, vs what is created inside."
    Claude showed the oracle made by the caller with the data in it, `Problem(geom, oracle, regularizer)`, and
    `newton_cg(problem, x0, ...)`, and asked which shape the frontend should have. The maintainer (15:32): "I am
    concerned about how the data is bundled with other things at such a high level and so early in the program."
    Claude put the concern back as: the oracle binds a description (kind, order) and arrays (sample, data,
    weight) into one object, so that new data force a new oracle, and it binds again what the engine takes as
    arguments on purpose (3.6); and offered the split. The maintainer (15:38): "Yes, this is the concern. The API:
    newton_cg(geom, oracle, batch, x0, regularizer=None, ...) is ideal if we can make it work without
    substantially complicating the rest of the code or forcing unwanted performance tradeoffs. Can you trace
    through the implications of oracle-as-description and data as argument, and see how that changes the plan?"
    After the trace (15:44): "Confirmed both".
  - **The split.** Every argument of the engine has one of three homes. Static: the kind and the order (and the
    plan, if 7.3 puts it there) are the oracle, a hashable description with no arrays. Arrays of the data: the
    sample, `y` and the weight are the batch, arrays only. Per call: the frame, the centers, `want`, the
    directions and `held` are arguments of the operation.

    ```python
    newton_cg(geom, oracle, batch, x0, regularizer=None, ...)

    values, held = oracle.evaluate(frame, batch, want, Vc=None, direction=None,
                                   core_direction=None, held=None)
    ```

    The least-squares oracle is the engine with its static arguments fixed. This is today's split between
    `kind` and `(sample, data)`, with the kind generalized to an oracle.
  - **What changes in each decision.**

    | Decision | As recorded before | With the split |
    | --- | --- | --- |
    | 8.2, the oracle | holds kind, order, sample, data, weight | holds the kind and the order only |
    | 8.1, the model | reaches the data as `model.oracle.sample` | holds the batch as a leaf: `model.batch` |
    | 8.3, the contract | an optimizer touches geometry, oracle, regularizer, model | the same, and a batch it never looks inside |
    | 8.3, `Problem` | bundles geometry, oracle, regularizer | not at the seam; `local_model(geom, oracle, batch, x, ...)` is a function |
    | 8.3, a minibatch | "is an oracle" | is a batch; `draw(rng)` returns one |
    | 8.3, the scale and the drawn indices | fields of the minibatch oracle | fields of the drawn batch |
    | 8.3, `n_measurements` and `take` | methods of the least-squares oracle | the same, taking the batch: `take(batch, indices)` |
    | 8.3, the compile key | the statics and the shapes | the oracle is all static, the batch is all leaves |
    | 8.4, a validation set | "is an oracle" | is a batch: `make_newton_display(oracle, batch, val_batch=None)` |
    | 8.4, `y_block_sumsq` | computed when the oracle is made | computed once by the display, from the batch |
    | 8.4, `f_resolution` | an optional output | the same; the norm of `y` comes out of the build pass, which reads `y` anyway |

    Unchanged: `want` and `held`, a model that never changes, the trial policies, the five phases, the
    safe-mode rule, the traceable `take`.
  - **Performance, as traced.** Nothing the bound oracle could cache is lost: packing and padding the sample is
    done once, when the batch is made, by the helpers of 5.2; arrays move to the device once at the optimizer's
    entry, as today; the reductions over the data ride in a pass that reads `y` anyway or are done once by the
    display; trial models hold a reference to the same batch; and sharding is helped, the arrays being
    top-level arguments of every compiled program. One cost: the display's one pass over `y` for the data's
    block sums. Traced on paper, not run.
  - **What it complicates.** One more argument on the optimizers, the operation, `local_model`, the display and
    the draw helpers. The batch needs a stated form for least squares: `sample`, `y`, `weight`, and for a drawn
    batch `scale` and `indices`; suggested, a small record of arrays with defaults, so that the sample and `y`
    alone are enough. The plan and the batch must agree (the plan names axes of the sample stack, the user
    reshapes the arrays); a mismatch is a structural error at the first call; for the performance discussion.
    `mc_sgd`'s argument `batch`, the size, becomes `batch_size`. A user's oracle with no data of its own takes
    `None` as its batch.
  - **Settled on the way.** Weights that change every iteration need nothing: pass a new batch. The oracle has
    no arrays, so one object serves the frontend and the backend, and the frontend's signature is the one above.
    The equalizing weight has no keyword (next sub-bullet).
  - **The equalizing weight: an explicit call, no keyword** (amends 6.7; the second thing confirmed at 15:44).
    The maintainer had asked (15:12) whether equalizing is done once at the outset or causes branching deeper.
    Claude's answer (15:20): once; the equalizing weight is a function of the data alone, its result is an
    ordinary weight, and no layer below can tell it happened; unlike derivative mode it transforms nothing, so
    nothing is in another form. With the split there is no place where an oracle is made with the data in hand,
    so a keyword would have to sit on every optimizer. Decided: the user writes the weight when the batch is
    made, `weight = filtered_inverse(block_norms(kind, order, y))` (placeholder names), and "no weight means
    ones" holds everywhere. With derivative data out as well (2.3), none of the preparation keywords that were
    discussed remains in the fitting functions; the rule of 6.7 for what such a keyword may do stands for any
    later one, such as in the `fit(...)` facade.
  - **Open details:** the name and the fields of the batch record; whether a record of the three static things
    remains inside the library; the name `batch` beside `batch_size`.

- **8.1, decided** (2026-11-16, 04:46). Asked for the name and whether he confirmed the revised proposal as a
  whole, the maintainer: "1. Go with Jp_norm_sq. 2. confirmed."
  - **What `LocalModel` is.** The quadratic model of the objective at one frame, which the optimizers query; and,
    in the maintainer's words, "a holder for local information at a point, which may be used by the user for their
    own optimizers or their own other purposes." It is the geometry, the oracle and the regularizer put together.
  - **What it holds:** `geom`; `oracle`; the frame, with the centers `(Vc, Hc)` when the geometry returned them;
    `held`, what the oracle asked to have kept at this frame, which the model passes back and never looks into;
    the regularizer; the geometry's companion; the curvature state. `kind`, `sample` and `n_w` are no longer
    fields of the model: the least-squares oracle has the kind, the order, the sample, the data and the weight
    as its own fields, reachable as `model.oracle`.
  - **What each method asks the oracle for:** `misfit` and `objective`, `f`; `gradient`, `g`, then the
    projection; `gn_quadratic(p)`, `Jp_norm_sq` at the projected `p`; `hvp(p)`, `JtJp` at the projected `p`, and
    in exact mode `JtJp` and `Kp` in one request so that they can share the dotted sweep (corrected on
    2026-11-16, 13:35: what they share is `nu_L_dot` and `eta_D_dot`, two of the sixteen variables such a
    product computes, and only without curvature damping; note to 8.3), with the geometry's
    lift, tangential curvature and lift transpose around `Kp`; `retract(p)`, nothing. The regularizer adds its
    terms as it does today.
  - **The model's own methods do not form arrays over the samples.** `gn_quadratic` and `hvp` no longer call
    `jacobian`. MC-SGD's Cauchy step stays `|g|^2 / gn_quadratic(g)`.
  - **`residual` and `jacobian(p)` stay on the model.** `jacobian(p)` asks the oracle for `Jp` at the projected
    `p`; `residual` asks it for `r`, read from the held variables if `r` is held and computed otherwise. Neither
    is a field that is always stored: the plan's `hold` decides what is kept.
  - **This amends 8.2.** An oracle's outputs are `f`, `g`, `JtJp` and `Jp_norm_sq`, required, and `Kp`, `Jp` and
    `r`, optional. An oracle without `Kp` works with the Gauss-Newton Hessian only; one without `Jp` or `r` makes
    `jacobian` or `residual` raise, and the rest works.
  - **The name `Jp_norm_sq`** replaces `Jp_sumsq` (the name in every record before 04:46): the squared norm of
    `Jp` in the weighted inner product, `<Jp, omega^2 Jp>`. The rule behind it, Claude's and put with the
    question: `sumsq` means a plain sum of squares (`r_block_sumsq` is unweighted and keeps its name); `norm_sq`
    means the squared norm in the weighted inner product, as in the library's `point_norm_sq`.
  - **Two tests required of the target** (Claude's, put with the answer on the weights and not objected to), both
    tolerance tests with a random weight: `p` paired with `JtJp` equals `Jp_norm_sq`; and `g` is the derivative
    of `f` by finite differences.
  - **Left open:** what the least-squares oracle keeps in `held` and where the plan lives (7.2, 7.3, the
    performance discussion; a field of the least-squares oracle is a third possible home for the plan, noted
    and not proposed); the form in which the oracle's one operation returns and takes back `held`; how a
    minibatch is given, with `Problem` holding the geometry, the oracle and the regularizer (8.3); what the
    display reads (8.4).
- **8.1, the discussion** (2026-11-16, 04:08 to 04:41; decided at 04:46, the note above).
  - **Claude's first proposal** (04:08): `LocalModel` is the geometry, the oracle and the regularizer put together
    at one frame. It holds `geom`, `oracle`, the frame with its centers, an opaque `held`, the regularizer, the
    geometry's companion and the curvature state. `kind`, `sample`, `n_w`, `sweep` and `residual` stop being
    fields, and `jacobian(p)` leaves the model.
  - **The maintainer** (04:24): "I am concerned about getting rid of things from LocalModel. It serves as a holder
    for local information at a point, which may be used by the user for their own optimizers or their own other
    purposes. Having access to the residual can be useful for many purposes. Another concern is MC-SGD: we need
    Jacobian vector products for setting the Cauchy step length. That said, the local model contains the oracle
    within it, so these things would still remain accessible. Can you explain more the reasoning behind your
    proposal?"
  - **The reasoning, as Claude then put it, with the maintainer's answers of 04:41:**
    1. The model's own methods are not to be built on arrays over the samples: `gn_quadratic` and `hvp` ask for
       the fused quantities (`<Jp, omega^2 Jp>` and `JtJp`), formed chunk by chunk, and do not call `jacobian`.
       MC-SGD's Cauchy step is `|g|^2 / gn_quadratic(g)` today and stays that. His answer raised three points
       about the fused quadratic, taken below.
    2. `residual` and the sweep are not always-stored fields, because the plan's `hold` decides what is kept;
       `residual` becomes something the model returns on request, read from the held variables if `r` is held
       and computed otherwise. "Agree."
    3. "A general oracle has no sample space" was too strong and is withdrawn: an oracle that gives `JtJp` has a
       `J`. What is true is that the model cannot know the shape of `Jp` and `r` for a user's oracle, and it does
       not need to. "Ok understood".
  - **The revised proposal** (04:24, Claude's): `residual` and `jacobian(p)` stay on `LocalModel`; both are
    optional outputs of an oracle, like `Kp` (the least-squares oracle supplies them; without them those two
    raise and the rest works); `gn_quadratic` and `hvp` use the fused quantities; `model.oracle`, `model.frame`
    and `model.held` keep the rest of the local information reachable. This would amend 8.2: `f`, `g`, `JtJp`
    and the weighted quadratic required; `Kp`, `Jp`, `r` optional. To be confirmed as a whole.
  - **The fused quadratic, his three points** (04:41): "Does Jp_sumsq incorporate weights properly? Since we are
    leaning heavily on weights now, this must work. Also, I do not like the name Jp_sumsq since the desired
    quantity is ||Jp||=<Jp,Jp> under weighted norm/inner product. Another thing I am unsure of is how the jetting
    is supposed to interact with the norm/inner product. I think we just "sumsq with weights" on the non-jet axes
    and sumsq regular on the jet axes, though I'm not sure."
    - **Weights.** By 3.6 and 3.7 the quantity is `<Jp, omega^2 Jp>`, the same number as `p` paired with `JtJp`.
      Checked (`scripts/review/jet_steps_check.py`, the line added on 2026-11-16): with a random weight for every
      mode and order, at orders 0, 2 and 4, the two agree to rounding, and each agrees with the library's own
      weighted kind. Not covered there: apply and entries, the `K` and `C` stacks, the uniform layout.
    - **The name.** Decided at 04:46 (the note above). Claude's suggestion: `Jp_norm_sq`, with `sumsq` kept for the plain sum of squares
      (`r_block_sumsq`, which is unweighted) and `norm_sq` meaning the squared norm in the weighted inner
      product; the library has `point_norm_sq`. The other candidate: `pJtJp`.
    - **Jets and the inner product.** By 2.5 the weight has one entry for each mode and order (`(d,) + T` for
      probe, `T` for apply and entries), so the order axis is weighted too, one weight per order; the samples
      and the free mode index are summed plainly. The inner product is diagonal: order `t` with order `t`, no
      term between different orders. The jet structure enters only in the pairing of a jet with a dual jet
      (order `t` with order `top - t`, 2.4), which the adjoint sweep uses inside; the helper of 3.6 (the weight,
      then the turn of the order leg) is what takes one to the other.

- **8.3, part 1, decided: the contract, and how `held` passes** (2026-11-16, 13:03 to 13:51). Claude split 8.3
  into three parts: the contract itself, with how `held` passes; how a minibatch is given; what is compiled as
  one program. This note is the first.
  - **The proposal** (13:03), which is what was decided:
    - An optimizer touches a `Geometry`, an `Oracle`, a `Regularizer`, and the `LocalModel` built from them.
      `Problem` bundles the first three. No kind, sample, data, sweep or residual appears in an optimizer.
    - The oracle's one operation takes `held` and returns it. A sketch; the names are not final:

      ```python
      values, held = oracle.evaluate(
          frame,                # (D, U, L, R)
          want,                 # names among f, g, JtJp, Jp_norm_sq, Kp, Jp, r
          Vc=None,              # centers, when the geometry returned them
          direction=None,       # (V, H): for JtJp, Jp_norm_sq, Jp
          core_direction=None,  # (D_dot, L_dot): for Kp
          held=None,            # what an earlier call at this frame returned
      )
      ```

    - What goes into `held` is the oracle's business; for the least-squares oracle it is the named variables
      its plan says to hold, so `model.held` can be inspected. A call never computes a variable only to hold
      it: it keeps what it computed anyway and its plan lists.
    - A model is built once and never changes. Building is one oracle call at a frame, wanting `f`, and `g` too
      in exact mode, as today; the model keeps those and `held`. Every later method passes `held` in and keeps
      nothing new, so inside CG the model is fixed loop state, as today. Keeping more means building a new
      model at the same frame, given the old one's `held`.
    - Trial points, following 4.3. Policy a: `f` at the trial's raw cores, nothing held; if accepted,
      orthogonalize and build a model. Policy b, the default: retract-and-orthogonalize, then build a model
      wanting `f`; if accepted, that model is the next model and its gradient is one adjoint pass. Policy c:
      the same, the model wanting `f` and `g`.
    - `problem.objective(x)` on raw cores stays, for policy a and for the full-batch stop checks of MC-SGD and
      Adam. It is the oracle's `f` at the corewise frame `(B, G, G, G)`, so it depends on the open detail of
      how the engine is told that a frame is corewise.
    - The geometry's `frame(x)` and its retract-and-orthogonalize function both return the frame with its
      centers (already implied by 4.1).
  - **The maintainer's first question** (13:22): "As structured, it seems like the oracle is a thin wrapper over
    the sample variables engine, whose purpose is to close over the data. Is it doing anything else? Second, it
    continues the pattern started in the sampling variables engine of having program path depend on magic words
    (f, g, JtJp, Jp_norm_sq, Kp, Jp, r) which the user has to read documentation to find, and which are harder
    though not impossible to analyze statically. For the engine, we have weighed the pros and cons of this
    approach and deemed it worthwhile because of the complexity. Here I'm wondering if the value proposition is
    still there. The other approach I see is to add named methods, such as oracle.f, oracle.g, and so on. Can
    you walk me through the reasoning for designing it as proposed, then assess alternatives and list pros and
    cons?"
  - **Claude's answer** (13:35).
    - The least-squares oracle is a thin wrapper, by the library's rule that a backend class binds parameters
      and names a role. Its purpose is to be the seam: the optimizers stop knowing about sampling; its statics
      are the compile key and its arrays are leaves; and it is where a user puts something that is not one
      engine call.
    - Of the reasons for `want`, only fusion is real, and it buys two things. `f` with `g` saves one forward
      pass per accepted point, and only when the plan holds nothing. `JtJp` with `Kp` share `nu_L_dot` and
      `eta_D_dot`, two of the sixteen variables an exact Hessian product computes, and only without curvature
      damping (`SAMPLE_VARIABLES.md`, section 5.2, "One dot"); the library does not fuse them today. This
      corrects the phrase "share the dotted sweep" in the note to 8.1.
    - The alternatives put: A, one operation with `want` (any combination in one pass; against it, magic
      words, string dispatch in every user's oracle, an optional output found missing at run time, harder
      static reading, unlike the library's other protocols); B, named methods `f`, `g`, `JtJp`, `Jp_norm_sq`,
      `Kp`, `Jp`, `r` with `precompute(frame)` for what is held, the pattern of `Geometry.precompute` and
      `geom_aux` (against it, outputs share work only through `held`); C, B with named fused methods. Claude
      leaned to B.
  - **The maintainer's second question** (13:41): "It seems like the problem is that we are trying very hard to
    avoid state (a worthy goal) for an operation that is fundamentally about state (holding variables for
    performance reasons). If we allow state to enter the oracle, do things simplify, and if so are there any
    risks? If we are unwilling to let go of the performance gains from holding, then the state has to be held
    somewhere, and we should make a principled decision about where. I want you to think through the design of
    the oracle and things that use it, with a new eye on where state *should* be held. Then think of potential
    problems that may arise, e.g., if the optimization is done in parallel over many gpus sharded"
  - **Claude's answer** (13:48), from the library's own record `docs/contributor/precompute_and_caching.md`
    (the audit of 2026-08-20).
    - State inside the oracle would simplify the call sites. Its risks are those the record names for a hidden
      cache on a data object: staleness ("same point" is a numerical question, and with policy b two points are
      alive at once); tracing (CG is one compiled program); compile keys (the oracle is value-hashed). The
      record admits hidden state only when it is correctness-neutral, harmless when lost and never traced; held
      variables are none of the three.
    - Where the state goes, by the record's rule (an explicit value, at the widest scope whose invalidation
      is structural, through a channel the loop already threads): the sample, data, weight, kind, order and
      plan are fields of the oracle and die when the problem changes; the held variables, the frame, the
      centers, the geometry's companion, `f` and `g` are leaves of `LocalModel` and die when the iterate
      moves; the dots and the tilde variables of `Jp` are not kept; the current model, the trial model, the
      radius, the moments and the rng are the optimizer loop's variables. The oracle is stateless, `LocalModel`
      is the holder for a point, and the optimizer's host loop is the only place where anything is replaced.
      `held=None` recomputes, so a standalone call is always correct. No object is moved to a new point, so
      nothing can be stale.
    - At scale, six points. (1) Memory is now a carrying cost, which the record says it is not: the plan's
      `hold` decides what is carried, and under policy b a trial's held variables and the current point's are
      alive together. (2) Sharded held variables must be arguments of the compiled programs; reached through a
      closure or a hidden field they would be expected to become constants of the program, not sharded, with a
      recompile at every point (expected, not run). (3) Sums over the samples cross devices, so `g`, `JtJp`
      and `f` agree across device counts only to rounding, and a near tie in a line search can go differently
      (bears on tests and on 9.3). (4) Every host decision is a synchronization of all devices: a few per
      Newton iteration are fine, one per MC-SGD step may not be (part 3). (5) Minibatch draws must not move
      samples between devices (part 2). (6) With several hosts every process runs the same loop, and explicit
      values keep them in step.
  - **The decision** (13:51): "Having thought through all of this, I have a greater appreciation for your
    original design with oracle.evaluate. Let’s go with that". So alternative A, as proposed at 13:03: one
    operation with `want`, `held` going in and coming out. The placement of state above is Claude's and was
    put before the decision; under A it is unchanged, since `held` is still an explicit value kept by the
    model.
  - **Carried forward under A,** from the list of what counts against it: how an oracle says which outputs it
    supplies, so that a missing optional one (`Kp` in exact mode) is found before any work, as the library
    checks `corewise_hessian` today; and that `want` is a literal at every call site in the library.

- **8.3, part 2, decided: a minibatch is an oracle** (2026-11-16, 13:55 to 14:07).
  - **Today.** `draw(rng)` returns `(sample_B, data_B)` and the optimizer passes them to
    `problem.local_model(x, sample_B, data_B)`. The default draw uses the kind's `n_measurements` and `take`.
    The optimizer scales the regularizer by `batch/n` with `batch` the nominal size, so a custom draw of another
    size gets the wrong factor, as the docstring of `_minibatch_step_problem` says.
  - **Decided** (the maintainer, 14:07: "1. Agree. Settled."): `draw(rng)` returns an oracle, and the step
    builds its model from the geometry, that oracle and the regularizer. The optimizer never sees a sample or
    data. The default draw is for the least-squares oracle, a uniform random subset as today; `n_measurements`
    and `take` become methods of the least-squares oracle and are not part of the oracle protocol. A user's
    oracle comes with its own draw, and MC-SGD or Adam on a user's oracle without a draw raises. Every
    minibatch oracle has the same statics and shapes, so the step compiles once. The full-batch stop check
    uses the full oracle. `rng` stays a host generator passed by the caller.
  - **The regularizer's fraction: decided, the third of three options** (14:07: "2. Your lean."). The options
    were: the nominal `batch`, as today; the draw returning the fraction beside the oracle; or the minibatch
    oracle scaled so that it estimates the full one. Taken: a draw returns an oracle whose `f` is an unbiased
    estimate of the full `f` (its `f`, `g` and the rest scaled by `n/batch`), and the regularizer is not
    scaled. The optimizer then has no batch logic: no `batch` in the step, no scaled regularizer, no count of
    measurements. The scale is a field of its own on the least-squares oracle, not folded into the weight,
    because the weight means the inner product on the data space. Its name is open.
    - Put with the question: MC-SGD takes the same step either way (scaling the data term by `c` in place of
      the regularizer by `1/c` scales the gradient by `c` and the Cauchy length by `1/c`); Adam is the same up
      to its `eps`, and the size of its gradient no longer depends on the batch size. Argued, not run.
  - **Is the minibatch data still accessible?** The maintainer (14:05): "will the minibatch data itself still be
    accessible under this architecture? I can imagine future optimization methods which want to use the
    minibatch data for,e.g., building regularizers or preconditioners, or steplengths, or something else."
    Yes: the drawn `sample`, `data` and `weight` are fields of the minibatch oracle; the model built on it has
    `model.oracle`, `model.residual` and `model.jacobian(p)`, and with `model.frame` and `model.held` the engine
    gives any sampling variable on the minibatch; and the oracle enters a compiled step as an argument with its
    arrays as leaves. The access is through the concrete oracle and not through the protocol, which says
    nothing of samples or data: a method that needs the minibatch data is written for oracles that have data.
  - **The drawn indices are kept** (14:07: "Also do the suggested addition (keep drawn indices on minibatch
    oracle)."). The default draw records on the minibatch oracle which samples it drew; variance-reduction
    methods need to know which, not only their values. The field's name is open.
  - **Sharding.** A draw of random indices across all samples moves data between devices. Since a draw is any
    function returning an oracle, a shard-local or a block draw fits the contract unchanged. Which draws the
    library ships is left to the performance discussion.

- **8.3, part 3, decided: what is compiled as one program** (2026-11-16, 14:10 to 14:30). With it 8.3 is decided.
  - **Today, under `use_jit`.** MC-SGD and Adam compile one step per program (`_mc_sgd_step`: gradient, Cauchy
    length, retraction), and a host loop calls it with `draw(rng)` and does the full-batch check every
    `check_every` steps. `newton_cg` compiles only the CG solve; the model build, the gradient and the trial
    objective run eagerly. `gradient_descent` has no `use_jit`.
  - **Decided: the phases are the units, and the outer loop is on the host.** Five phases, each a module-level
    function of pytrees:

    | Phase | Takes | Returns |
    | --- | --- | --- |
    | build | problem, frame and centers | the model (`f`, `held`) |
    | gradient | model | the projected gradient |
    | CG solve | model, right-hand side, tolerance, cap, radius | the step and the loop state, as today |
    | trial | model, step | policy a: `f`; policies b and c: the model at the trial |
    | stochastic step | problem with the minibatch oracle, point, moments | the next point, as today |

    Everything sized by the samples is inside one of them. The host keeps the decisions: accept or reject,
    stopping, callbacks, display, timing, history. `use_jit=False` runs the same functions eagerly, on numpy or
    jax, as today. The compile key is the statics and the shapes: the geometry, the oracle's kind, order and
    plan, the regularizer, the Hessian mode, the literal `want`. Not in the key: the weight, the minibatch and
    its scale, the tolerance, cap and radius, the learning rate. A rank change recompiles, as today.
  - **Left to the performance discussion:** which of the phases are compiled by default. Claude first argued
    that the build, the gradient and the trial have to be compiled because an eager scan recompiles on every
    call. The maintainer (14:24): "For the scans, are the scan bodies closures? I think we did a lot of work to
    make the scan bodies module level functions in the current codebase for this reason. Can you check?" He was
    right: `docs/contributor/scan_body_principles.md` records the problem (measured 2026-08-20, 19 compilations
    per Newton iteration) and the fix, and every `xscan`, `xmap` and `xwhile` body found in `backend/` is a
    module-level function or comes from a memoized factory. So an eager scan compiles once per shape, and
    compiling those phases is a choice. What remains for it: a phase that runs across devices with its sums
    reduced over them has to be one function; and eagerly every scan is its own small program with the rest
    dispatched operation by operation. Against it: more and larger programs to compile at each rank level,
    which the consumer's notes already complain of.
  - **Decided: safe mode.** Checks are skipped inside a trace. Today the build is eager, so its preconditions
    are checked there even under `use_jit`. Rule: in safe mode a compiled program's preconditions are checked on
    the host, on its concrete arguments, before it is called.
  - **MC-SGD compiled over several steps: wanted later, not in this arc.** The maintainer (14:24): "I thought we
    already supported compiling the whole MCSGD loop, in which case this is a regression in performance compared
    to what we already have. Though maybe this is in another experimental codebase not this one and I’m getting
    it mixed up; can you check how MCSGD is currently compiled here? Eventually we want to support compiling
    multiple steps of MCSGD." Checked: T3Toolbox compiles one step per program, so there is no regression;
    T3Polynomial has no compiled MC-SGD loop either (its `ot.py` compiles whole Newton and Sinkhorn loops for
    optimal transport, and its upstream notes record "Fully jitting the Newton loop: evaluated, deferred").
    Recorded as a requirement on the contract: nothing decided now may block it. Two properties keep it an
    addition: the step is a pure module-level function of the problem, the point and the minibatch oracle, so
    it can be a scan body as it stands; and the default draw goes through indices, so the host can draw the
    index sets of many steps at once and one program scans the step over them, with the host `rng` kept. The
    natural size is `check_every` steps per program. A user's draw would need the same split, a host part
    giving arrays and a traceable part giving the oracle.
  - **Decided now, for that:** `take(indices)` on the least-squares oracle can be traced with a static batch
    size.
  - **Not proposed:** the whole Newton iteration as one program.
  - The maintainer, on the three points (the phases with the default left to the performance discussion; the
    safe-mode rule; traceable `take` now and multi-step MC-SGD later), at 14:30: "Confirmed".

- **8.4, the first part** (2026-11-16, 14:32 to 14:57; the derivative-mode point was settled at 15:12, next note).
  - **The earlier decision** (2026-11-14, decision 9): the display gets a summed variable `r_block_sumsq`, so it
    needs no per-sample residual; the data's block sums and weighted norm stay `kind.block_sumsq(data)` and
    `kind.sumsq(data)`; the trust-region floor reads only the residual's dtype. That was written against kinds.
  - **Claude's restatement with an oracle in place** (14:32). Train table, numerators: `r_block_sumsq`, an
    optional oracle output at the model's frame, using `held`. Denominators: `y_block_sumsq`, an optional oracle
    output that does not depend on the frame; the least-squares oracle computes it once when it is made.
    Validation table: a validation oracle, asked for `r_block_sumsq` at the model's frame. Unweighted objective:
    half the sum of the block sums, as today. Trust-region floor: `f_resolution`, an optional oracle output.
    Callbacks: `info.x_cores` and `info.lm` as today, `info.lm` carrying the oracle, the frame and `held`.
  - **The maintainer on the table** (14:43): "Regarding the data block sum display. If I recall correctly,
    previously we displayed differently for the unjetted and jetted variants, because the jetted one had the
    additional derivative order axis. Here the unjetted versions are subsumed as jetted problems of zeroth order,
    so one display has to work for both. What is important is that the display show the full table of errors,
    for each derivative order and for each tensor index. The display and stored diagnostics should not reduce
    this information over any axis."
    - Checked in the library: today the block sums are always a 2-D array `(n_mode, n_order)` with a size-1 axis
      where a kind has no modes or no orders, and one formatter, `_format_table`, chooses among four layouts by
      that shape. So the two displays already come from one function.
    - **The rule, put to him at 14:50 and not objected to:** `r_block_sumsq` and `y_block_sumsq` have exactly the
      shape of the weight (probe: `(d,)` at order 0 and `(d, order+1)` above; apply and entries: a scalar at
      order 0 and `(order+1,)` above). An absent axis takes the place of today's size-1 axis and the same four
      layouts follow. Nothing is reduced over modes or orders: the only sums are over the samples and over the
      entries of each mode's output vector; the display prints every entry; the stored record keeps the full
      arrays for training and validation at every iteration; the unweighted objective is an extra number and
      replaces nothing. Claude read "for each tensor index" as "for each mode" and said so.
  - **Decided** (14:57: "1.,2.,3. are confirmed."):
    1. A validation set is an oracle, as a minibatch is. `make_newton_display` takes `val_oracle` in place of
       `val_sample` and `val_data`.
    2. The trust-region floor is the oracle's statement: an optional output `f_resolution`. The formula
       `eps (|y| |r| + |f|)` is the cancellation error of a least-squares misfit and belongs in the
       least-squares oracle. An oracle without it gets `eps |f|`, with `eps` from the dtype of `f`.
    3. An oracle without block sums gets the display's header line and no table, where today the display raises.
  - **Open: point 4, derivative mode.** Claude proposed converting the unweighted objective back to the form the
    user gave the data in. That went against 2.3 as decided ("Nothing is converted on the way out"). The
    maintainer (14:57): "For 4, I am considering the architecture regarding taylor mode vs derivative mode data.
    In my conceptualization, derivative mode data is prepared for taylor mode optimization outside the optimizer
    using helper functions that we provide, which scale the data and create weights. I believe we also decided
    to add a helper option that can do it for them in the optimizer, but this is a nicety and not central. The
    optimizer remains focused on Taylor mode. Now here, it seems like the structure is mixing. Instead of Taylor
    mode being what the optimizer does, with an adapter to prepare derivative mode data and generate its weights,
    now both are becoming first class citizens, with all the complications that entails. Can you look through our
    previous work on this? I am considering simply moving the derivative mode stuff completely outside, and
    forcing the user to prepare their derivative mode data before optimizing it. Please think through the pros
    and cons of this."

- **Derivative mode moved out of the fitting layer; 8.4 decided** (2026-11-16, 15:05 to 15:12). This amends 2.2
  and 2.3.
  - **What had been decided** (2026-11-15): derivative mode as one adapter at the outer level of fitting (2.2);
    three standalone conversions and one opt-in keyword on the fitting functions that calls them on the data, the
    validation data and the weight, with nothing converted on the way out and, under the keyword, "no weight"
    meaning `t!` per order (2.3); preparation by standalone functions, with opt-in keywords in the fitting
    functions for preparation that changes the loss alone (6.7).
  - **Claude's reading of the mixing** (15:05). Its own point 4 caused it and is withdrawn. But the keyword
    leaks further than the display: every fitting entry point has to take it; a validation set is now an oracle
    the user builds, so the oracle's constructor would need it too; "no weight" gets two meanings; a weight that
    changes between iterations is converted only the first time; and what the library reports is in Taylor form
    while the user's arrays are not. The keyword does nothing that the standalone calls do not do.
  - **For moving it out:** one form and no branch in the fitting layer, with "no weight" always ones; every number
    the library reports is in the form of the arrays the user passed in, so the display question disappears;
    nothing reaches the oracle, the validation oracle, the display or the stored diagnostics; the conversion is
    one visible place in the user's script; fewer signatures, tests and documents. **Against:** the user writes
    the conversion; the weight trap (a user who converts the data and passes no weight gets ones on Taylor
    coefficients, which is not ones on derivatives: the weight must be `t!`); what they want back in derivative
    form they convert themselves. **Neither way:** derivative data read as Taylor coefficients is a silently
    wrong fit whether the user forgets a keyword or forgets a call.
  - **Decided** (the maintainer, 15:12): "Ok, let's move it out and add the standalone conversion function like
    you suggest." So:
    - Taylor coefficients are the only form in the sampling, fitting and optimization layers. No fitting
      function, oracle, display or sampling method has a keyword for derivative data.
    - The standalone functions are four: derivatives to Taylor coefficients (slot `t` divided by `t!`); the
      reverse; a derivative-mode weight to a Taylor weight (slot `t` multiplied by `t!`); and one function that
      prepares the data, the validation data and the weight together and returns the weight `t!` when none is
      given, which answers the weight trap in one line.
    - The doctest required by 2.3 documents these functions. The draft is rewritten:
      `scripts/review/adapter_doctest_draft.py` (three cases, the validation data, and the objective reproduced).
      The names are placeholders. For probe the weight has a mode axis as well, which the draft, written for
      apply, does not show.
    - If a keyword is ever wanted again, the `fit(...)` facade planned for later is the layer for it (Claude's
      remark, not a decision).
  - **So 8.4, point 4:** nothing is converted for the display. Everything it prints is in the form of the arrays
    the user passed in. With points 1 to 3 and the rule that nothing is reduced over modes or orders, 8.4 is
    decided.
  - **Not included: the other preparation keywords of 6.7.** The maintainer (15:12): "Regarding 6.7 (equalization
    weights). I see these differently. For derivative vs Taylor, it is about adapting a different problem
    (derivative) to our setting (Taylor). For equalization, I see that more as a tunable parameter for solving
    the Taylor problem better. Now, strictly speaking the optimization problem itself does change for
    equalization too. But I see it as the same in the broader sense that we are trying to fit a polynomial to
    Taylor data. Now, that said, some of added commplexity downsides may still be present for 6.7 and I would
    like to think through these. If we equalize weights, is that something which is done once at the outset of
    the optimization, then never used again, or does the fact that weights have equalized cause more branching
    behavior deeper in the optimization?" Under discussion; see the notes to 6.7.

- **8.2, decided, the name apart** (2026-11-16, 02:45). The maintainer: "I agree with this data structure.
  However, I would like to find another word than objective. When the user hears "objective", they may think
  just "f", and not all of the other things that come bundled with it."
  - **What the optimizers need.** Read from `backend/optimizers.py`: a local model asks its kind for a forward,
    a transpose, sums of squares and the curvature term, and wraps each in the geometry's projection and lift.
    With the geometry's parts left in the geometry (4.5), what remains is a list of quantities at a frame: `f`
    (the value, at trial and accepted points), `g` (the gradient before projection), `JtJp` (for a direction
    `(V, H)`), `Jp_norm_sq` (the Gauss-Newton quadratic), `Kp` (for a lifted direction `(D_dot, L_dot)`, exact
    Hessian only), and `r_block_sumsq` for the display. That is the engine's own output vocabulary.
  - **Decided: the data structure.** A frozen record with one operation: at this frame, give these of `f`, `g`,
    `JtJp`, `Jp_norm_sq`, `Kp`, with the directions passed when needed; blocks before any projection, in the
    engine's shapes. Its parameters are fields, so it has value identity. The built-in one is least squares: it
    holds the kind, the sample, the data, the weight and the order, and implements the operation with one call
    of `sampling_variables`. A user's implements the same operation however it likes: several data sources add
    the results of their parts; a general linear functional never calls the engine; a different loss calls the
    engine for `z` and the bare transpose. One that cannot supply `Kp` still works with the Gauss-Newton
    Hessian. The regularizer stays its own protocol.
  - **The name: `Oracle`** (02:52). Claude's suggestion: in optimization an oracle is the thing an algorithm
    queries at a point for the value, the gradient and Hessian products, so the word leads a user to expect
    several things from it; against it, it is jargon. The alternatives offered were `Misfit` (the library's
    word for the data term, pairing with `Regularizer`, but still sounding like the one number), `DataSource`
    and `Evaluator`. The maintainer: "Oracle is ok, let's go with it. I see your point about jargon, but I have
    not thought of anything better". The built-in one is the least-squares oracle.
  - **Open,** for group F: how held
    variables pass between calls of the operation (8.1), what is compiled as one program (8.3), and what the
    display reads from a user's oracle (8.4).

- **8.5.** Parked by the maintainer on 2026-11-14 as low priority. T3Polynomial's main fit has the same sample
  vectors at every input mode.

## 9. Names and verification

| # | Question | Status | Depends on |
| --- | --- | --- | --- |
| 9.1 | The names settled on 2026-11-14 | Decided | |
| 9.2 | The kind is one of the three strings `'probe'`, `'apply'`, `'entries'`, everywhere. `'sweep'` as a unit name goes: a `hold` lists variable names. The names of the sizing functions are settled with 7.6 | Decided | 3.4, 6.3, 7.6 |
| 9.3 | Correct is defined by the definitions, checked on small problems without any of the sweeps: every edge variable against its contracted subnetwork by a dense einsum, `z` against the dense tensor, jets against the Taylor expansion of the same contraction, `Jp` against the dense tangent tensor, the transposes and the tilde variables against the dense Jacobian, `Kp` against the second derivative of the dense objective. The identities are tests in their own right. The generic twin is a second check. Numerical comparisons are to a tolerance, structural ones exact. The old library is a bridge used once during the port. These checks also serve as documentation | Decided | |
| 9.4 | What is the order axis called as a group, so that it is not taken for a batching axis, and what letter does it have in contraction strings? **Amended 2026-11-19 (the generalized jets; the notes of 2026-11-17 and entries 116 to 122):** the jet axes are the one group `J` (entry 120), in place of `T` and `A`. | Decided | 2.7 |
| 9.5 | The convention for arrows on the edges. An arrow shows the way a vector travels: where it leaves an array the index is plain, where it enters the index is dual. A contraction joins the two ends of one arrow. Every arrow of a frame points away from its store, towards the hole. Arrows are described by the way they point in the diagram: left-to-right, right-to-left, bottom-to-top, top-to-bottom | Decided | |
| 9.6 | The notation for the sampling variables: four letters and a basis. `mu`, `nu`, `xi`, `eta` name the four sides of a cut edge that can be contracted; the store of the basis follows the letter where the basis is not the usual one (`nu_L`, `eta_D`, `mu_R`, `xi_U`); then the grade. The reference is `SAMPLE_VARIABLES.md` | Decided | |
| 9.7 | The sampling variables are documented in layers, each kind of fact with one source and every other place generated from it or tested against it. The engine's docstring is the index (the table of 6.4). One page in `docs/`, whose code CI runs, is the home: the naming scheme; an entry per variable with its diagram, definition, recurrence, shape and a doctest; the identities; the dependency structure, as a table and as diagrams; how to use them. One example file is the tour. A `.tex` note is the derivation. The definitions are shipped as a public reference module: one slow, dense function per variable, called by the tests, the page, the example and users | Decided | 6.4, 9.3 |

Notes:

- **9.7, added and decided** (2026-11-16, 15:58 to 16:03). The maintainer opened it with his agreement to 9.3:
  "Let's think more broadly about documenting these variables. As I see it, we have: The definitions; The tensor
  diagrams; The identities; Code that tests the definitions and identities; There may be more I'm not
  remembering. I am wondering how we can organize all of this coherently, so that the user can easily find and
  understand the variables when they need them." His ideas, "not mutually exclusive": a monolithic and
  exhaustive document, organized in separate sections or by variable; the engine's docstring; a `.tex` and
  `.pdf` mathematical paper; one or more example `.py` files that show the definitions being checked. "The goal
  is to help the user of the code understand what they are, what properties they satisfy, and how to use them."
  - **The readers, as Claude put them:** someone holding a name, who asks what it is, what shape, what it costs
    to keep; someone with a purpose (a custom oracle, an optimizer of their own), who asks which variable they
    need and what they can rely on; someone who must trust or extend it, who asks why and where it is checked.
  - **The rule:** each kind of fact has one source, and every other place is generated from it or tested against
    it. Which variables exist, their shapes and what reads what: the engine's declared graph (6.4), with the
    docstring table and the page tested against it. What a variable is: its definition as code, a dense
    contraction, kept honest by the definition tests of 9.3. The picture: the diagram specs, with a test that
    every variable has a diagram. What it satisfies: the identities, stated once, with the identity tests of
    9.3. Why: the mathematical note, whose notation is the code's names.
  - **The four layers** (the maintainer, 16:03: "1. Yes, this is exactly right."):
    1. The engine's docstring is the index: the table of 6.4 and a pointer to the page. Nothing longer.
    2. One page in `docs/` is the home. It grows out of `SAMPLE_VARIABLES.md`, and CI runs its code as it does
       for every `docs/*.md` page. It opens with the naming scheme in half a page (four letters, a basis, two
       marks), so that a reader can decode any name; then a section per grade (plain, dot, tilde, tilde-dot,
       outputs) with an entry per variable: diagram, definition, recurrence, shape, and a doctest that checks
       it against its definition; then the identities, each with a doctest; then how to use them (holding, a
       custom oracle, modified variables). Entries are by variable, because the reader arrives with a name.
    3. One example file is the tour: a small problem, a handful of variables computed and checked, the
       identities, a held variable reused. A selected subset; the exhaustive checks are the tests.
    4. A `.tex` note is the derivation: why the recurrences equal the subnetworks, the jets, the adjoint
       variables, the arrows. It would become a section of the toolbox paper.
  - **The definitions are shipped as a public reference module** (16:03: "2. Yes, good idea. Let's ship the
    definitions as a public reference module."). One slow, dense function per variable that computes it from
    its definition. The tests, the page's doctests, the example and a user checking a modification of their own
    all call the same code. The library ships `dense_probe` in the same spirit. Its name and place are open.
  - **The dependency structure is documented too** (16:03: "On top of these, we also need to document the
    dependency structure: The table of dependencies, and the diagrams showing it visually"). Its source is the
    declared graph of 6.4. `scripts/component_graphs.py` already derives, for every case, what each component
    reads and which cores it goes through, from the generic program; it is the check on the declared reads.
    How the diagrams are drawn is open (suggested: generated from the declared graph, as one layered picture
    per grade and one per output showing everything that output needs).
  - **Also belonging on the page,** from Claude's list: sizes, since what a variable costs to hold decides the
    `hold` list; recipes for use; and, once, in the release notes, the map from old names to new.
  - **Not examined:** drawing a variable's picture and building its dense contraction from one spec, so that
    the picture cannot disagree with the definition. The diagram specs already say which core sits at which
    site.
- **9.3, decided** (2026-11-16, 15:50 to 15:58). The maintainer: "I agree wholeheartedly with the proposal about
  how the definitions are checked, and that the identities listed get tests. On top of that, the definition
  checks and identity tests can also serve as a form of documentation, illustrating to the user what the things
  being computed actually are, in code."
  - **The reference for each output,** on small problems, without any of the sweeps: every plain and dot edge
    variable, its definition as a contracted subnetwork (`SAMPLE_VARIABLES.md`) by a hand-written dense einsum;
    `z`, the dense tensor contracted with the inputs; jets of any of these, the Taylor expansion in `s` of the
    same dense contraction, by interpolation in `s`, which is exact for a polynomial; `Jp`, the same contraction
    applied to the dense tangent tensor; `g`, `JtJp` and the tilde variables, the dense Jacobian (the forward
    applied to a basis), transposed; `Kp`, the second derivative of the dense objective, by automatic
    differentiation where jax is present and by finite differences otherwise; `f`, `r`, the block sums and
    `Jp_norm_sq`, their formulas on the outputs above. This extends the library's rule that dense ground truth
    is the bedrock. `scripts/review/sample_variables_check.py`, parts B and C, does it for 14 variables.
  - **The identities, tests in their own right,** each with a random weight: the adjoint identity between `Jp`
    and `g`; `p` paired with `JtJp` equals `Jp_norm_sq`; `g` is the derivative of `f`; a held variable gives the
    result of a recomputed one; the result does not depend on the plan, the chunking, the contraction order, the
    layout or the backend; order 0 equals the first slot of a higher order.
  - **The generic twin** stays as a second check: it derives every term mechanically, so it catches a missing
    term, and it shares the sweep structure, so it cannot catch an error in that.
  - **Tolerances.** Numerical things to a tolerance scaled to the dtype, never bit for bit, comparisons across
    plans, contraction orders and device counts included. Structural things exact: shapes, the table of
    variables against the declared graph, the dtype rule, uniform masks.
  - **The old library is a bridge, used once.** During the port new outputs may be compared with the old
    kernels through the stated conversions. It is not a permanent test, and a disagreement is settled by the
    definitions.
  - **Optimizers** are judged by outcomes on problems with known answers and by the model's internal
    consistency, not by the old iteration counts or stop reasons.
  - **At scale,** correctness rests on the invariances that link a large run to a small verified one; the
    identities are cheap at any size and can be run on the real problem.
  - **Raised by the maintainer with his agreement:** how all of this (the definitions, the diagrams, the
    identities, the code that tests them) is organized as documentation for the user. Under discussion.
- **9.2, decided** (2026-11-16, 02:56). The maintainer: "Confirmed. We will need to discuss 3 more later in the
  broader context of performance optimization hooks and defaults".
  - **1. The kind strings.** On 2026-11-14 the engine's three strings stood beside the frontend's six, and the
    two spellings "need reconciling". The six went with the `*_derivatives` kinds (2.8), and a kind is a closed
    description (3.4). The three strings `'probe'`, `'apply'`, `'entries'` are the kind, everywhere.
  - **2. `'sweep'` as a unit name in `hold`.** It named a kind's whole frame sweep as one unit, because a
    user-written kind would have only that unit. There are no user-written kinds, every variable has a name
    (6.4), and a "sweep" is a set of held variables (6.3). The unit name goes; a `hold` lists variable names, so
    the old default is `('xi', 'mu', 'nu', 'eta')`.
  - **3. The sizing function names** (`estimate_chunk_size`, `max_chunk_size_within`, and the helpers `chunk`
    and `unchunk`, which were to be methods of a kind). Not settled here: moved to 7.6, to be discussed with the
    performance hooks and defaults as a whole.
- **Names and the T4S paper** (the maintainer, 2026-11-15, 21:59). "Trying to keep consistency with the t4s paper's
  notation through this refactor is an impossible task. We do not need to attempt it." So a name is chosen for the
  target alone; `sigma`, `tau` and the other letters the refactor frees may be given new meanings. (The notation of
  9.6 uses neither.) The library's notation map from the paper to the code will need rewriting when the names land.
- **9.6, decided** (2026-11-15, 23:23). The maintainer, on the seven drawings in formula style: "This is excellent
  and further confirms that this notation is the right one. I want to lock in this new notation." At 23:02 he
  had called the scheme "the best so far", with "great potential for confusion", and asked for the drawings to
  meet that.
  - **What is decided.** A variable is named by the side of the cut edge that is contracted (`mu` the left of a
    bond, `nu` the right of a bond, `xi` the leaf below a Tucker edge, `eta` the trunk above it), by the basis
    of the cut edge where it is not the letter's usual one (`L` for `mu`, `R` for `nu`, `D` for `xi`, `U` for
    `eta`), and by its grade. In formulas and drawings: the Greek letter, a tilde or a dot above it, the basis
    as a subscript.
  - **What it replaces.** `gamma` is `nu_L` and `zeta` is `eta_D` (both letters are retired; `zeta` had been
    decided at 22:11). The frame's `mu_dot` and `xi_dot` are `mu_R_dot` and `xi_U_dot`, and the bare `mu_dot`
    and `xi_dot` now mean the base point's. The frame's `nu_dot` and `eta_dot` are `nu_L_dot` and `eta_D_dot`,
    the same names as the base point's, because they are the same arrays. The six variables that had no name
    (notes to section 6) all have one.
  - **The reference** is `SAMPLE_VARIABLES.md`: the taxonomy, every definition as a recurrence and as a
    subnetwork, the types and shapes, and the map from the old names and from the library's. Checked by
    `scripts/review/sample_variables_check.py`. The earlier names and schemes are in `NOTATION_HISTORY.md`.
  - **Not decided by this:** see section 11 of `SAMPLE_VARIABLES.md`.
- **9.4.** The maintainer's concern: a batching axis means a bunch of the same thing bundled together, and it has
  fewer restrictions. The order axis is neither. The words the library already has: the contraction engine's
  "group" (an uppercase letter standing for zero or more axes), which is structural and says nothing about
  batching; "batch block" and "stack" for `C`, `W` and `K`; and "order axis" or "jet axis", lowercase `t`, for the
  one axis.
- **9.4, the maintainer's suggestion (2026-11-15): "optional axis".** The order axis is an optional axis: absent at
  order 0, present with length `order + 1` above. The phrase is not used anywhere in the library today. Still
  open: its letter in contraction strings. The engine needs an uppercase letter for anything that can be absent;
  today's strings use only `C`, `W` and `K` in uppercase, and write the order axis as lowercase `t`.
- **9.4, decided (2026-11-15).** The term is "optional axis", and the optional order axis is written `T` in
  contraction strings. Lowercase `t` stays the letter for the order axis where it is always present.

- **9.5, decided** (2026-11-15, 05:39 to 05:47). The rule (two ends of one arrow) was agreed at 05:39. The
  meaning of the arrows was the maintainer's lean ("inward pointing arrows towards a core should mean dual
  indices ... outward facing arrows from a core should mean plain"), Claude agreed, and he confirmed: "All good
  except the name suggestion." The full account is in sections 8 to 10 of `TYPE_CATALOG.md`.
- **9.5, the direction words.** Claude had suggested keeping "left", "right", "up" and "down" for the names of
  the stores only, and describing arrows relative to the hole. Rejected by the maintainer: "Describing
  directionality with respect to diagrams (ascii in the documentation here and tikz in papers) is immensely
  valuable to humans, even if there is the risk of misinterpreting or switching it (as I have already done). It's
  a risk that is worth it in my opinion." So both uses stand: a name says where a store sits, an arrow is
  described by the way it points in the diagram, and the left cores carry left-to-right arrows. The risk is
  accepted. What it asks of the documents is that each diagram shows its arrows.
- **9.5, the wording for the direction of an arrow** (2026-11-15, 05:49). The maintainer: "For directionality, we
  can use 'left-to-right' and 'right-to-left' rather than rightward/leftward to avoid confusion with stores." So
  an arrow along a bond is "left-to-right" or "right-to-left", and "left" and "right" alone stay with the stores.
  For the vertical direction, "bottom-to-top" and "top-to-bottom": Claude's proposal, confirmed by the maintainer
  at 05:50. "Up" and "down" alone stay with the stores.
- **9.5, the arrows of the base point** (2026-11-15, 15:49 to 16:10). The maintainer's convention: a gauge
  convention used unless otherwise stated, bottom-to-top and left-to-right, so that every edge points towards the
  last TT core; with the last `L` core and the first `R` core as the known exception to gauge correctness. It is
  the `D` and `L` rows of the frame, and the checker already uses it. Asked to confirm it with 2.4 he said "I
  think so". Claude's wording of the exception, to be corrected if wrong: those two cores are typed like the
  others, and what fails for them is `M M^T = I`, since they carry the norm (checked). Section 11 of
  `TYPE_CATALOG.md`.

## After the target

Not in this list, on purpose: the order of work, staging, adapters for migration, what T3Polynomial has to change,
the reference paper, and the library backlog items of finding F17.
