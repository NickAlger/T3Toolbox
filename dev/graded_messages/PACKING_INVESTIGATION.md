# Packing and unpacking at the uniform sampling seam — working document

> **Superseded as the statement of decisions by `PACKING_REFACTOR.md` (2026-11-24).** This document stays as the
> historical record (the evidence, the measurements, the arguments and the rulings as they were made).

_Started 2026-11-23 (Nick + Claude), on the branch `graded-messages` at `43314164` (the M arc complete, P17 next). A
living document: findings and decisions are added as the investigation goes. Nothing here is decided unless a
section says so._

_**A scouting report** (Nick, 2026-11-24). This session works out **what** to do, not how. The slices and a detailed
plan come later, through their own process; the prototypes in §10 show what is meant, they are not a build plan.
The M arc's decisions are open to revision here (§9.6)._

## 1. The issue

The uniform sampling functions (`backend/ut3_sampling.py`, `backend/utv_sampling.py`, the `UniformTuckerTensorTrain`
and `UT3Tangent` methods, and on this branch the engine `sampling_variables` itself) accept the mode vectors either
**packed** (one `(d,) + ... + (N,)` array, zero padding after each mode's real width `N_i`) or **ragged** (a `len=d`
sequence of real-width arrays). A ragged input is packed inside the function; a probe-type output is unpacked again
when the input was ragged ("the packedness mirror"). This is a union type on the arguments, which is not the library's
usual style: functions normally take one type, and conveniences such as packing are helpers the user calls if they
want them.

The questions: is this working correctly; is it the right design; and if automatic packing stays, what principle makes
its behaviour predictable?

**Scope (Nick, 2026-11-24):** every sampling function, and everything directly downstream of them. The fitting layer
that works with probes (the sampling kinds, the oracle and batch, data preparation, the data metric, the display) is
in scope; something like `t3svd` is not.

## 2. Principles

**Agreed (Nick, 2026-11-24):** (the principles are labelled G1, G2, G3; they were P1, P2, P3 until 2026-11-24, renamed so as not to collide with the arc's slice names, P0 to P20.)

- **G1. Automatic packing raises if the unpacked vectors do not match the uniform object's real shape.** Ragged
  vectors whose widths differ from the masked shape `(N_0, ..., N_{d-1})` are almost certainly a bug in the caller's
  code, not a request to be padded. Packing at the input/output seam is safer than packing internal edge variables,
  because the external mode padding is a prefix (real entries first) while the real slots of internal rank edges
  can be scattered; but silently zero-padding a wrong-width vector still turns a bug into a wrong answer (finding F1).

- **G2. Liberal in, canonical out** (agreed, Nick, 2026-11-24; §9.8). On the uniform layer, the padding of every
  input (cores and data alike) is junk: anything finite, and every operation neutralizes it (masks on entry by
  default). The padding of every output of a sampling operation is exactly zero. Core-producing operations outside
  sampling (`t3svd`, orthogonalization) keep "no promise", as now. **Every packer returns a canonical object too**
  (O5: the data packer refuses mixed edges rather than put a constant on the padding). The refinements from the
  case-by-case review: G2's "liberal in" applies to the functions that hold the masks; a raw-array primitive states
  zero padding as its precondition instead (§10.8, C2). On every padded contraction the engine zeroes the core side;
  data-space objects that no core touches are masked themselves (§10.9, O1 item 4).

- **G3. One mathematical operation per function, one return type** (agreed, Nick, 2026-11-24; §10.5 position (B)).
  Functions that perform different mathematical operations have different names; they are not bundled behind a
  `kind` argument. A function has a single return type, not `typ.Any`. The engine and the oracle were deliberate
  relaxations (generic machinery); since then the style has been creeping into the library, and G3 rolls it back
  elsewhere. Inventory: §10.6.

**Requirement (Nick):** whatever is decided needs a well-thought-out principle behind it that makes the behaviour
predictable.

**The range of principles (Nick, 2026-11-24):**

- **One extreme: no automatic packing or unpacking.** Every function takes one form; the user packs and unpacks with
  the helpers if they want to.
- **The other extreme: an absorbing rule, like the existing dispatch.** As jax absorbs numpy and the less precise
  float absorbs the more precise, *packed absorbs unpacked on the uniform layout, and unpacked absorbs packed on the
  ragged layout*, done only as needed at the lowest level of the code. The conversion goes one way:
  `{packed, unpacked} x ragged -> unpacked`, `{packed, unpacked} x uniform -> packed`. Converting the outputs back to
  the input's form (the mirror) needs the converting operation to remember the input's form and unpack after the
  computation.
- **The question (Nick): where is the seam?** A small packer/unpacker at the start and end of every sampling function?
  If not, where does it make sense to draw the line? Analysis in §8.

## 3. History of the automatic packing

| Date | Commit | What happened |
|---|---|---|
| 2026-06-15 | `409352b3` | Uniform slice 4: `ut3_sampling` first written. Always packs: it takes ragged vectors only, packs them, unpacks `probe`'s output. Packed input not accepted. |
| 2026-06-21 | `36c56d71` | **Decision "packing-strict"** (`dev/archive/uniform_fix_plan.md`): "Sampling is strict; packing lives at the boundary … a mismatch ERRORS (structural house-rule; no silent auto-pack) … The `ut3_sampling` 'bug' = packing done inside the op — move it out." Reason given: performance (pack once at the boundary, like mask-once). **Never implemented.** |
| 2026-07-06 | `1c2c111a` | **Slice U3.5, the "packedness-mirror pivot"**, recorded as "design change, agreed with Nick". `is_packed` and `pack_if_ragged` added; the user-facing ops infer packedness and mirror it; the optimizer's `*_from_sweep` seam packed-only. Reasons, from `docs/uniform_equivalence_contract.md` §"The vector I/O boundary — packedness mirror": the same "infer, don't flag" rule as numpy/jax and ragged/uniform; ragged in/out makes the uniform layer a drop-in "faster ragged" (`op_uniform(ragged) == op_ragged` as written); packed input passes through untouched, which met the June performance concern; the helpers stay public for backend users. The conversation itself is not preserved. |
| 2026-07-12 | `f19ccc09`, handoff | The July handoff: "The packedness-mirror convention (U3.5) — don't 'normalize' it to a flag." The same week the June "strict" wording was promoted into `docs/contributor/uniform_polymorphism.md:44-47`, **after** the pivot (finding F5). |
| 2026-11-10 | `99006763` | Glossary scouting flags the contradiction (item 29 of `dev/archive/glossary_docs_pass_2026-11-15/glossary_candidates_B_backend.md`). Not resolved. |
| 2026-11-17 | slice P3's review | A tuple-vs-array silent misread in the jets helpers (§5). Nick's ruling: the type is decided by `kind`, never by the object. |
| 2026-11-19 → 21 | `PART1_PLAN.md`, P11, P12b, P13 | The plan carries the mirror forward "as today". The engine (P11) packs a ragged sample itself; the uniform twins (P12b) and the frontend methods (P13) keep the mirror. |

## 4. What the code does today

Four layers, each with its own rule:

| Layer | Sample `ww` / index | Probe's `y`, `weight` | Held variables | Outputs | Checks the ragged widths? |
|---|---|---|---|---|---|
| Frontend `UniformTuckerTensorTrain` / `UT3Tangent` methods | ragged or packed | — | — | mirror the sample (by delegation) | **yes**, `_check_sample` → `jets.check_sample` (ragged only) |
| `ut3_*` / `utv_*` twins | ragged or packed (`pack_if_ragged`) | — | packed by the twin before the engine call | probe-type outputs mirrored (`unpack_vectors` if the input was ragged); blocks, scalars and sweeps always packed | **no** |
| The engine `sampling_variables` | ragged or packed (`pack_if_ragged`, `sampling_variables.py:1870`); index a single array | ragged or packed, mixing allowed (`_pack_data_space`, `:1872-1873`) | **packed only** (`_check_held`) | **always packed**; never unpacks | **no** (its docstring: "It does not check the mode sizes of the sample against the frame's") |
| Older jet kernels (`sampling_derivatives`, the `*_from_sweep` functions) | whatever the twin hands them (always packed) | — | — | packed | cannot: the twin hands them `ut3_apply_masks(data)`, the two supercores only, no shape |

On the **ragged layout** the engine is strict the other way: a packed array is read as a sequence of `d` rows (valid
when every `N_i` is equal), and unequal widths raise ("w[0] has shape (6,), where N_0 = 4").

**What the engine knows on the uniform layout.** It takes the full `.data`, `(B, G, shape, masks)` or
`(D, U, L, R, shape, masks)`, reads `shape = frame[-2]` (`sampling_variables.py:1865`) and **masks every input itself
on entry**, re-masking the outputs on exit. It uses `shape` already for the entries index (`_uniform_index`). So the
engine could check the ragged widths; it does not.

**Concrete examples** (shape `(4, 5, 6)`, `N = 6`; script
`/tmp/claude-1000/-home-nick-repos-T3Toolbox/d6c0341d-52ea-4136-9f37-7350e7c6e6cf/scratchpad/packing_examples.py`):

```
ut3_probe, ragged ww                -> [(4,), (5,), (6,)]      # the twin unpacks: mirror
ut3_probe, packed ww                -> array (3, 6)
engine, ragged ww, want (w, z)      -> w: array (3, 6), z: array (3, 6)   # the engine never unpacks
engine, packed ww + ragged y        -> r: array (3, 6), same numbers as all-ragged or all-packed
engine, held Jp ragged              -> ValueError: the held Jp has shape tuple; expected (3, 6)
jets.jet_to_dual_jet('probe', packed)  -> [(3, 6), (3, 6), (3, 6)]   # a tuple of N-wide slices (F7)
```

## 5. The tuple-vs-array silent misread (slice P3, 2026-11-17): resolved

Found in slice P3's review (`implementation/P3_REVIEW_r1.md:165`). `jet_to_dual_jet` and friends told the probe kind from
the others by whether they were given a tuple, so a packed probe array passed whole was read as the apply type:
modes reversed and orders not turned, no error; `derivatives_to_taylor` scaled the mode axis by `t!` when
`d == order + 1`. Nick's ruling (`P3_RESPONSE_r1.md`, Addendum 2; `TARGET_DECISIONS.md` ~l.498): the four functions
take `kind` first, and "the type is decided by `kind`, never by the type of the object". `jets._arrays(kind, value)`
walks the first axis for probe, so a list, a tuple or a packed array are all read right. Regression test:
`test_kind_decides_the_type_and_a_packed_probe_array_is_taken_mode_by_mode` (`tests/test_jets.py:248`).

The remaining tuple tests in `jets.py` were checked: `data_metric` (no `kind`) decides by `isinstance(r, tuple)`,
which is safe because it is elementwise, and a ragged weight with a packed `r` raises naming `pack_batch` (M1 design
review, decision 2); `blocks_to_weight` and `_mode_weights` take `kind` and only use the test to tell ragged from
packed within probe. Only `jets.py` and the engine were checked for this.

## 6. Audit of the primary paths (2026-11-23)

Nick's questions, for the intended forms only (mixed inputs are a later question):

1. Uniform function, all inputs packed: is every output packed?
2. Uniform function, all inputs packed: does it ever unpack in the middle and repack? (A slice at a time inside a scan
   over the slices does not count.) The worry: uniform operations redirected to ragged ones and back.
3. Ragged function, ragged inputs: is every output ragged?
4. Ragged function, ragged inputs: does it ever pack in the middle and unpack later?

**Method.** A pytest plugin (read-only instrumentation, nothing in the repo changed) wrapped every public sampling
function (the ragged and uniform backend modules, the engine, the four frontend classes, the fitting kinds' methods)
and every converter (`pack_vectors`, `unpack_vectors`, `_pack_data_space`, `t3_to_ut3`, `ut3_to_t3`, the frame and
variation conversions, `fitting._arrays_of`, `pack_sample` / `pack_data` / `pack_batch`). For every call it recorded
the form of the inputs and outputs, and for every converter call the stack of sampling calls it happened inside. Run
over 29 test files: 794 tests, 58,043 subtests, all passing; 218 distinct functions seen. Since the tests mostly call
the uniform functions with ragged vectors, the plugin also re-called each such function with the vectors packed
("shadow" calls) and compared the result with the ragged one (real part equal, padding exactly zero). The last
uncovered functions (`ut3_entries_derivatives`, `utv_{probe,apply,entries}_derivatives`) were called directly, and the
code was grepped for inline width-slicing, stacking and padding that the wrappers cannot see.

**Results.**

| Question | Answer |
|---|---|
| 1 | **Yes**, every uniform function. Every shadow call agreed with the ragged result. (The plugin's flags of "a per-mode tuple" in packed outputs were all `d = 2` coincidences: the `(Tucker block, TT block)` pair, or a sweep record, whose length equals `d`; checked directly that no leaf has a real width.) |
| 2 | **No.** No pack, unpack or layout conversion inside any all-packed uniform call; the only converter seen is `_pack_data_space` given an array, which returns at once. No width-slicing in the sampling code. On the uniform path the engine masks once on entry, then runs its site steps through `xmap` and its sweeps through `xscan` over the leading `d` of the padded supercores (`lax.map` / `lax.scan` on jax; on numpy, `numpy_map` / `numpy_scan` loop over full-width slices, the excluded case). The older jet kernels dispatch the same way. |
| 3 | **Yes.** Per-mode outputs are tuples of real-width arrays (also at `d = 1` and equal widths), never a packed array. |
| 4 | **No.** No conversion of any kind inside any ragged call. |

**Limits.** The evidence covers the inputs the test suite builds (plus their packed twins); a conversion done inline
rather than through a named converter is caught only by the grep and the output checks. The fitting layer was covered
only where the tests exercise it through these functions; it has not been audited as such (F3).

**Tools.** `dev/graded_messages/packing_audit/`: `sampling_audit_plugin.py` (the plugin; its docstring has the usage
and the limits) and `analyze.py` (reads its JSON: `table`, `shadows`, `convs`). The test files of the final run:
`test_sampling_{functions,uniform,apply,entries,corewise,stacks,typed,identities,variables}.py`,
`test_uniform_probing.py`, `test_probe_derivatives.py`, `test_frontend_sampling.py`, `test_uniform_tucker_tensor_train.py`,
`test_uniform_manifold.py`, `test_manifold.py`, `test_tucker_tensor_train.py`, `test_engine_data_space.py`,
`test_held_variables.py`, `test_transposes_typed.py`, `test_fitting.py`, `test_oracle_data_space.py`,
`test_least_squares_oracle.py`, `test_block_sums.py`, `test_data_metric.py`, `test_oracle_sweep.py`, and in
`tests/backend/`: `test_from_sweep_reuse.py`, `test_uniform_fitting.py`, `test_curvature.py`, `test_uniform_curvature.py`
(about 10 minutes at `-n 2`).

## 7. Findings and open items

### F1. A wrong-width ragged vector is silently zero-padded on the uniform backend (a bug; G1 applies)

```
frontend u.probe            width 3 for N_0 = 4: ValueError: ww[0] has the mode size 3; expected N_0 = 4
backend ut3_probe           width 3 for N_0 = 4: accepted silently -> [(4,), (5,), (6,)]   (a wrong answer)
engine sampling_variables   width 3 for N_0 = 4: accepted silently -> (3, 6)
ut3_apply (engine)                               accepted silently
ut3_probe_derivatives (older kernel), bad pp     accepted silently
ut3_apply_derivatives (older kernel), bad ww     accepted silently
```

(Scripts `width_check.py`, `width_check2.py` in the scratchpad.) The ragged layout raises for the same mistake. Every
place that packs today has the shape at hand: the engine (`frame[-2]`), and each twin (`data[2]` / `frame[4]`), even
where the kernel it calls does not. The smallest fix in line with G1: `pack_if_ragged` takes the shape tuple instead
of `N` and raises on a width mismatch (`_pack_data_space` likewise for an edge that is neither 1 nor `N_i`). Not
decided; depends on §2.

### F2. Per-mode Python loops over packed arrays (Nick: "looping over slices unnecessarily is highly suspect"; to review case by case)

None of these unpacks to real widths (the slices are full width `N`), but each loops over the `d` modes in Python and
restacks, where one operation on the packed array may do. Inventory (grep, 2026-11-24; not yet reviewed):

| Site | What it does |
|---|---|
| `sampling_steps.py:915` `_modes`, used at `:958` (`_weighted_sum`), `:1024` (`w_tilde_step`), `:1098` (block sums) | the data-space steps: per mode `data_metric`, flip, sum; restacked for probe (`:1028`, `:1099`) |
| `sampling_steps.py:862-863` | the entries kind's one-hot vectors on the uniform layout: built per mode, then stacked |
| `sampling_steps.py:1063` | `r_step`: ragged only (the packed path is one subtraction at `:1062`) — fine |
| `jets.py:229` `_arrays`, `:317` `_mode_weights`, and their loops at `:448`, `:529`, `:732`, `:1159`, `:1533` | `jet_to_dual_jet`, `dual_jet_to_jet`, `data_metric` (ragged branch), `prepare_derivative_data`, the checkers: per mode; a packed probe array comes back as a tuple (F7) |
| `jets.py:1275-1279` | `affine_jet`: per mode (input construction) |
| `fitting.py:840` `_arrays_of`, used by `_y_times_metric_y` (`:870-893`), restack at `:936` | **unpacks a packed probe `y` / weight to real widths**, per mode (F3) |
| `data_preparation.py:195`, `:718` | per-mode block norms of `y`; per-mode scales |
| `sampling_definitions.py:276`, `:664`, `:683`, `:851-852`, `:3259` | the dense definitions (reference twins); per mode by design? To confirm they are test oracles only |

### F3. The fitting layer is in scope and not yet audited

Known so far: `fitting._y_times_metric_y` (`backend/fitting.py:870`) unpacks a packed probe `y` and weight to the real
widths through `_arrays_of` (`:840-855`, which calls `unpack_vectors`) and computes per mode: a uniform-to-ragged
detour in the data-norm computation `<y, M y>`. The converter wrapper did not see it because it runs outside any
sampling call. To audit: `backend/fitting.py` (the kinds, `LeastSquaresOracle`, `LeastSquaresBatch`, the data
sums), `backend/uniform_fitting.py` (`pack_sample`, `pack_data`, `pack_batch`, the uniform kinds),
`backend/data_preparation.py`, `backend/jets.py` (the data metric, dual jets, derivative data), the display
(`backend/optimizer_display.py`), and their frontends (`t3toolbox/fitting.py`, `t3toolbox/optimizers.py`). Same four
questions, plus: where does each function pack, and does it check the shape (G1)?

### F4. The engine masks every input on entry and every output on exit, on every call

Nick (2026-11-24): the always-mask approach is safe and useful, but when several sampling operations are chained
through held variables (rather than asking for everything at once), the masking is redone at every call. That costs
when the number of samples is small compared with the cores, or when the chain is long. Proposal to consider: **an
engine option that turns the automatic masking off.** (The earlier code did not mask inside because it was too much
work to write; the engine has done that work, so that reason no longer applies.)

Facts to settle before designing it:

- What exactly is masked per call: every floating input by the object it is defined at (`_UNIFORM_MASKS`: the stores,
  `V`, `H`, `Vc`, `D_dot`, `L_dot`, `w`, probe's `y` and weight), every held variable by its table, and every output
  but a passed `w` on exit (`_mask_variable`) (`sampling_variables.py:1910-1935`).
- Its cost: one elementwise multiply per array; not measured. Under jit the masks are constants and the multiply
  may fuse into its consumer; to measure, eager and jit, against the sweeps' cost.
- What unmasked inputs would require: zero padding in every input (garbage in the padding is real: the S0c fix found
  padding up to 3e12 in transpose outputs). Is that a precondition checked in safe mode, a structural contract, or
  the caller's responsibility? How does it fit the house rule (structural always errors; numerical preconditions
  checked in safe mode)?
- Which other functions rely on the engine's masking (the twins mask nothing themselves now; the older kernels
  receive `ut3_apply_masks(data)` from the twins).

### F5. The documentation contradicts itself

`docs/contributor/uniform_polymorphism.md:44-47` states the June "strict … never a silent auto-pack" rule;
`docs/uniform_equivalence_contract.md` §"The vector I/O boundary" and `ut3_operations.is_packed` state the mirror.
Flagged 2026-11-10 (glossary item 29), unresolved. To fix once the principle is decided.

### F6. Smaller items

- The engine docstring says a wanted `w` "is returned as given"; on the uniform layout a ragged `w` comes back packed.
- The frontend methods annotate `ww: Sequence[NDArray]` but accept a packed array (the comment says "or packed"); the
  backend's `Vectors = Union[...]` is accurate.
- `is_packed` is `not isinstance(v, (list, tuple))`: a generator counts as packed (`TypeError: 'generator' object is
  not subscriptable`), and so does an object array (a clear "must have the frame's d and N" error). No silent wrong
  answer found.

### F7. The jets helpers return a third form for a packed probe array

`jet_to_dual_jet('probe', order, packed, w)` returns a tuple of `d` arrays each `N` wide: neither ragged (`N_i`) nor
packed. Values are right; passed back to a uniform twin it is re-padded to `N` (a no-op). Under G1 a strict width check
would reject it as ragged input of the wrong widths, so this interacts with F1.

## 8. Where the seam goes (Claude's analysis, 2026-11-24; not decided)

### 8.1 What a seam needs

A function that converts vectors needs:

1. **the shape** `(N_0, ..., N_{d-1})`, to check the vectors (G1) and to pack or unpack them;
2. **the layout of the computation** (ragged or uniform), to know which way to convert;
3. **for a mirror only:** a memory of the input's form until the output exists.

(1) and (2) are available exactly where the object that defines the layout is present with its data: the point,
the frame, the tangent. Below that level (the steps, the contractions, the jets helpers, the data metric) no function
has the shape, so under G1 no function below it can convert. **The lowest level a conversion can live is the lowest
function that holds the object.**

### 8.2 Where that level is

- **The engine `sampling_variables`.** It receives the uniform `.data` with its shape and masks, or the ragged cores
  (`N_i = D[i].shape[-1]`). Every sampling function on the new path goes through it: the `t3_*` / `tv_*` functions,
  the `ut3_*` / `utv_*` twins (P12), the frontend methods (P13), the kinds (P16).
- **The older jet wrappers in `ut3_sampling` / `utv_sampling`, until P20.** They hold `data[2]`; their kernels in
  `sampling_derivatives` do not. `sampling_derivatives` and the old jet functions are deleted in P20
  (`PART1_PLAN.md:139`, `:2239`), after which the engine is the only kernel path.
- **The explicit packers downstream**: `uniform_fitting.pack_sample` / `pack_data` / `pack_batch`, and the scales in
  `data_preparation` (`:641`). They take `N` today, not the shape.
- **Functions without the object** (the jets helpers, `data_metric`, the block sums, the steps) have no shape. By §8.1
  they never convert.

### 8.3 The options

| | Seam | Output form | Where G1 is checked | Conversion code | Signatures |
|---|---|---|---|---|---|
| (a) No automatic packing (one extreme) | the caller | the object's layout | the engine must still detect a wrong form to raise, so it checks there | none; errors name the helper | one type |
| (b) Every public function, mirroring (roughly today) | each twin and frontend method, plus the engine | the input's form, for probe-type outputs only | in every twin | 52 pack/unpack/`is_packed` lines in the twins (27 in `ut3_sampling`, 25 in `utv_sampling`), plus the engine | union |
| (c) The frontend only | the frontend methods | either rule | the frontend (it has `self.shape`) | the frontend; the backend strict | one type in the backend |
| (d) The lowest level holding the object, absorbing (the other extreme) | the engine's entry (and the older jet wrappers until P20) | the object's layout, every output | the engine, once per layout | one site per layout; everything above passes through | union, with a rule |

Observations:

- **(a) and (d) put the check in the same place, and differ by one line.** To raise on the wrong form, (a) has to
  look at the vectors where the shape is known, which is the engine's entry. (d) converts at that point instead of
  raising. So the location of the seam is not really in question between them; the question is *raise or convert*.
- **The mirror is what forces a seam in every function.** Only a function that sees both the input and the final
  output can mirror, which is every public function. It also has no single rule: `ut3_probe_corewise_transpose` takes
  two vector arguments (`ww`, `c`), so which one is mirrored? The corewise tangent mirrors `ww`; blocks and sweeps are
  never mirrored. And it repeats work in chains: a caller who stays ragged on a uniform object pays pack, compute,
  unpack at every call, and packs again at the next (the same objection as the repeated masking in F4).
- **Under (d) a chain stays in one form.** Every output on the uniform layout is packed, so a chain of calls (held
  variables, `Jp` fed to a transpose, a residual fed back) converts once, at the first call, and never again.
- **(d) is the library's existing pattern.** numpy/jax is inferred at the lowest level from the arrays and absorbed
  (CLAUDE.md: "infer it from the input array types at the lowest level — do NOT thread `use_jax` params around").
  A function in the middle neither checks nor converts; its callee does.
- **(c) gives the two tiers different rules for the same operation.** A backend user and a frontend user see different
  output forms from the same call.
- **The cost under (d) is a conversion inside the compiled program** when a caller passes ragged vectors into a
  jitted loop, like passing numpy arrays to a jitted jax function. Packing once in advance (`pack_batch`, which the
  fitting path already does) avoids it.

### 8.4 Recommendation: (d), with four rules

- **R1 (where).** Vectors are converted only by a function that holds the object defining the layout (point, frame
  or tangent, with its shape). In the sampling layer that is the engine's entry, plus the older jet wrappers until
  P20. Nothing above it converts again; nothing below it converts at all.
- **R2 (which way).** Absorbing: the object's layout wins, for every vector input and every output. Uniform gives
  packed; ragged gives ragged. No mirror.
- **R3 (G1).** The conversion checks the vectors against the object's shape and raises on a mismatch.
- **R4 (functions without the object).** They never convert. They act on the form they are given and return that
  form; mixed forms raise, naming the packer (as `data_metric` does today). This also settles F7: a packed probe array
  comes back packed, not as a tuple of padded slices. That in turn asks for native packed implementations, which is F2.

Consequences: the `ut3_*` / `utv_*` twins and the frontend methods lose their pack/unpack lines. A ragged input on a
uniform object returns packed (breaking against today's mirror). A user who wants ragged results calls
`unpack_vectors(z, x.shape)`, or a frontend helper if one is wanted. The frontend's `_check_sample` width check becomes
redundant (R3 does it). The equivalence contract becomes `unpack(op_uniform(v)) == op_ragged(v)` for either form of `v`.

**Why (d) over (a):** the same site and the same check. The only difference is converting instead of raising, and
that conversion follows the dispatch rule the library already uses. **When (a) would be better:** if single-typed
backend signatures matter more than the convenience. Its cost is explicit pack/unpack calls at interactive call
sites and in the tests; nothing else in the design changes. **Why not (b):** the mirror forces a seam in every
function, has no single rule when a function takes several vectors, and converts repeatedly in chains.

### 8.5 Open sub-questions under (d)

- **What G1 means on the ragged side.** "Unpacked absorbs packed": the engine slices a packed `(d,) + ... + (N,)` array
  to each `N_i`. Must `N` equal `max N_i` (the canonical packing of this object), or only be at least `max N_i` (a
  uniform object padded wider, its padding dropped as don't-care)? Today the ragged engine reads a packed array as `d`
  rows and raises unless every row has width `N_i`.
- **The data-space objects** (`y`, the weight): the same seam (`_pack_data_space` at the engine's entry). G1 for them is
  "each axis 1 or full; the edge 1 or `N_i`".
- **Held variables**: absorbed too, or kept packed-only as now?
- **The fitting packers** take `N`; under R3 they take the shape.
- **`want=('w',)`** returns the sample in the layout's form (what happens in practice today; F6's docstring).

## 9. Option (a): Nick's leaning, and its blast radius (2026-11-24)

**Nick's leaning (not decided):** (a), no automatic packing or unpacking, with a possible exception (also not
decided): a packed array accepted on the ragged layout as a sequence of `d` vectors of the same length. His reasons:

1. The most likely mixed situation in practice is a user packing and unpacking in a loop in their uniform
   optimization code. That is almost certainly suboptimal; they should keep things packed throughout. Forcing an
   explicit pack brings it to their attention: a list comprehension deep in a subfunction is easily hidden, a direct
   call to a packing function is not.
2. The absorbing rule is a whole framework built to support something we do not want users to do.

Before deciding: what is the blast radius of refusing unpacked vectors on uniform and packed vectors on ragged, and
what are the downsides and unintended consequences?

### 9.1 Method

`packing_audit/strict_forms_plugin.py` records, without raising, every place where (a) would refuse an input:

- **uniform side:** ragged vectors reaching `pack_if_ragged` or `_pack_data_space`, plus every `pack_batch` call. The
  analysis separates **automatic** conversions (the library converting on the caller's behalf) from **explicit** ones
  (the caller calling `pack_batch` / `pack_sample` / `pack_data` / `pack_if_ragged` directly, which (a) keeps);
- **ragged side:** a packed array given where the ragged layout takes a per-mode sequence: the engine's sample,
  probe's `y` / weight / held variables, and the per-mode arguments of the ragged public functions. Each is flagged by
  whether its trailing width equals every `N_i`, which sizes the candidate exception.

For every event it records the library caller and the test, doctest or example line that triggered it. It ran over
the full suite (1,979 passed; 4 failures, all caused by the recorder wrapping names the frontend re-exports and a test
checks by identity), the module doctests (325 passed), the doc pages (10 passed) and the two uniform examples (both
pass). `strict_analyze.py <dir> auto` reproduces the counts of §9.2 (`events`, `sites`, `users` for the rest).

### 9.2 Uniform side: refusing ragged vectors

**Tests: 135 distinct tests (of about 1,980) rely on an automatic conversion.**

| Path | Tests |
|---|---|
| sampling functions (the engine, the `ut3_*` / `utv_*` twins, the frontend methods) given ragged vectors | 73 |
| the uniform kinds or the backend oracle called with a ragged sample (backend fitting users) | 57 |
| the fitting frontend (`fitting.prepare_batch`: `newton_cg`, `adam`, `mc_sgd`, `local_model`) given a ragged batch | 49 |
| `ut3_unscale_coordinates` given ragged scales | 4 |

(A test can be in several rows.) By file: `test_uniform_probing.py` 32, `test_optimizers_frontend.py` 22,
`test_dispatch.py` 16, `test_fitting.py` 12, `backend/test_uniform_fitting.py` 12, `backend/test_uniform_curvature.py`
6, `test_engine_data_space.py` 6, `test_oracle_sweep.py` 4, then 13 files with 1 to 3 each. A further 101 tests
already pack explicitly (`pack_batch` called directly) and are unaffected.

**Module doctests: 13.** `UniformTuckerTensorTrain.probe`, `.apply`, `.apply_corewise_transpose`,
`.unscale_coordinates`; `UT3Tangent.probe`, `.apply`, `.probe_transpose`; `ut3_probe_corewise_hessian`,
`ut3_apply_corewise_derivatives_hessian`, `ut3_probe_corewise_derivatives_hessian`;
`data_preparation.ut3_unscale_coordinates`; `fitting.LocalModel`; `optimizers.newton_cg`.

**Doc pages: 1** (`sharing.md`, a uniform fit given a ragged sample).

**Examples: both uniform examples** (`fit_hilbert_uniform_newton_cg.py`, `fit_hilbert_uniform_probe_derivatives_newton_cg.py`),
through the fitting frontend, once per `newton_cg` call (one per rank-continuation round: 6 and 4 calls). No ragged
example is affected: every example builds its sample as a list.

**Library code that converts automatically today:**

- the engine: `sampling_variables.py:1870` (sample), `:1872-1873` (probe's `y`, weight);
- the twins: 52 lines of `pack_if_ragged` / `unpack_vectors` / `is_packed` (27 in `ut3_sampling`, 25 in `utv_sampling`);
- the frontend: `UniformTuckerTensorTrain._check_sample` (`uniform_tucker_tensor_train.py:53-61`) branches on the form;
- the fitting frontend: `fitting.prepare_batch` and `_is_packed` (`t3toolbox/fitting.py:582-609`); the optimizers'
  `_frontend_draw` (`t3toolbox/optimizers.py:117-122`), which packs **every batch a user's minibatch draw returns, at
  every step**; the validation batch (`optimizers.py:841`);
- `data_preparation.ut3_unscale_coordinates` (`backend/data_preparation.py:641`) and its frontend method;
- prose: the "mirrors" / packedness wording in the docstrings of `ut3_sampling` (7), `utv_sampling` (6),
  `uniform_tucker_tensor_train` (3), `ut3_operations` (3), `backend/fitting.py` (3), `sampling_steps` (2),
  `uniform_manifold` (1); the pages `uniform_equivalence_contract.md` (§"The vector I/O boundary"), `glossary.md`
  ("packed vectors … packedness mirror"), `naming_conventions.md`; the uniform example's docstring ("the only
  changes are the two marked lines").

### 9.3 Ragged side: refusing packed arrays

**Nothing measured breaks.** No test, doctest, doc page or example passes a packed sample, residual or held variable
to a ragged function. The one padded weight given on the ragged layout is in
`test_engine_data_space.py::TestTheCheck::test_other_forms_raise`, which already expects an error.

The candidate exception (a packed array read as `d` same-length vectors) has **no measured user** either. It is what
the ragged engine does today: it reads an array as `d` rows and raises unless each row's width is `N_i`, so it works
only when every `N_i` is equal. Whether to keep it is a question of ergonomics for equal-size tensors (for example
`rng.standard_normal((d, M, n))` for an `n x n x n` tensor), not of compatibility.

### 9.4 Downsides and unintended consequences

- **D1. The rule has to be about a padded mode edge, not about tuples versus arrays.** The probe kind's weight as one
  mode-leading array with an edge of 1 (the output of `jets.blocks_to_weight`; the per-(mode, order) ω matrix of
  `docs/fitting_and_optimization.md` §4.6) is given on the ragged layout in **180 tests and two doc pages**
  (`fitting_and_optimization.md`, `numerical_contracts.md`), through the oracle's `query`. It has no `N` axis, so it is
  the same object on both layouts, neither packed nor ragged. A rule "probe-type objects are tuples on the ragged
  layout" would break all of these and the documented feature. The same holds for a `y` with an edge of 1. The
  refusal must target an array whose trailing axis is a padded mode axis.
- **D2. G1 moves into the explicit packers, which take `N` today.** `pack_vectors(ww, N)`,
  `pack_sample(kind, order, sample, N)`, `pack_data(..., N)`, `pack_batch(..., N)`. Under (a), F1's silent
  zero-padding of a wrong-width vector does not disappear; it moves into the user's own pack call. The packers have to
  take the shape (or the object) and check.
- **D3. Uniform fitting batches lose their structural check unless it moves.** `LeastSquaresOracle.check_batch` reads
  only the ragged form; today the frontend runs it and then packs, and it already skips a batch that arrives packed.
  Under (a) every uniform batch arrives packed, so nothing would check it. The natural home is the packer: it sees the
  ragged form and, with D2, the shape.
- **D4. The library has one hidden per-step pack, and one harmless once-per-call pack.** `_frontend_draw` packs a
  user's minibatch draw at every step: exactly the pattern of Nick's first reason, and (a) removes it (the user packs
  once and draws from the packed batch with `take`). But the optimizers' entry (`prepare_batch`) packs once per call,
  outside any loop the user writes (a continuation loop packs once per round, negligible next to a solve). (a) removes
  that convenience too, though it is not the bad pattern.
- **D5. Frontend users need a public, object-bound packing helper.** Packing for fitting lives only in the backend
  (`uniform_fitting.pack_sample` / `pack_data` / `pack_batch`, taking `kind`, `order`, `N`). Two details are easy to
  get wrong by hand: `len_P=len(order)` for input jets whose jet sizes differ by mode, and the data-space rules (an edge
  of 1 kept, broadcast axes joined), whose general function `_pack_data_space` is private. A helper bound to the
  uniform object (something like `x.pack(ww, order=...)`, `x.unpack(z)`, and a frontend batch packer) would know `N`,
  the shape (G1) and `len_P`.
- **D6. The uniform layer stops being a drop-in for interactive use.** The uniform example's "only two changes"
  becomes three. Every probe output on the uniform layer is packed, and reading it per mode takes `unpack_vectors`.
- **D7. The jets helpers need their own rule.** `jet_to_dual_jet`, `dual_jet_to_jet`, `derivatives_to_taylor`,
  `taylor_to_derivatives` and `prepare_derivative_data` have no object and so no layout. By slice P3's ruling they take a
  packed probe array mode by mode, and they return a tuple of padded slices (F7). "Uniform refuses ragged, ragged
  refuses packed" does not cover them. Form-preserving (packed in, packed out; §8.4 R4) looks like the consistent rule.
- **D8. `pack_if_ragged` and `is_packed` lose their purpose.** Both are public in `ut3_operations` and exist for the
  mirror; 29 tests call `pack_if_ragged` directly. Removing them is a breaking change for any backend user who calls them.
- **D9. The error messages carry the design.** A refusal should name the form found, the form needed and the exact call
  that fixes it. The engine knows `N`, the shape and the jet shape, so it can print that call.

### 9.5 D1 against the data-metric design (2026-11-24)

The M arc's rule (`DATA_METRIC_PLAN.md` §2, decided 2026-11-22, built in M2) already has the concepts D1 needs.

- **Every data-space object has the type of `z`, each axis 1 or `z`'s size** (the slots `F`, the samples `W`, the
  stack `C`, the edge; never `K`). The edge is `e ∈ {1, N_i}` (§9.0's table). An edge of 1 means "the same along
  the edge". The weight of the D1 example, `blocks_to_weight`'s output `(3, 1, 1)`, is simply the full form of
  "one number per mode", with size 1 on `W` and on the edge.
- **For probe, the rule allows "a tuple or list, or one array whose first axis is the mode", on either layout.** On
  the ragged layout a mode-leading array must give each mode its own size. So it works with an edge of 1, or with a
  full edge when every `N_i` is equal (checked: `(5, 5, 5)` with a `(3, 10, 5)` `y` is accepted; `(4, 5, 6)` with a
  `(3, 10, 6)` `y` raises "y (mode 0) has shape (10, 6) … whose array there has shape (10, 4)").
  **This is Nick's candidate exception, already decided for data-space objects** (it is 6.7's mode-count point, kept by
  §2). The engine reads a sample `ww` the same way today. So the open question is not whether to add the exception,
  but whether the sample follows the same rule as the data.
- **So D1 dissolves when (a) is stated in these terms.** "Packed" is a property of the edge: an edge-1 object belongs to
  neither layout. A wording of (a) consistent with §2:
  - ragged: a per-mode object is a tuple or list of `d` arrays, or one mode-leading array read along its first axis,
    each array with its mode's size (an edge of 1 for a data-space object, or every `N_i` equal);
  - uniform: one mode-leading array, edge `N` (packed), or edge 1 for a data-space object; a tuple is refused, naming
    `pack_data` / `pack_vectors`;
  - nowhere a conversion: reading an array along its first axis is not packing or unpacking.
  `blocks_to_weight`'s output is then legal on both layouts unchanged.

Where the M arc already met the other downsides:

- **D3 is a documented limit of the M arc** (§3: a packed weight whose mode axis has size 1 broadcasts unchecked; "the
  checkers take the ragged form only, and `pack_batch` builds packed weights from checked ragged ones"). Under (a)
  every uniform batch is packed by the user, so the limit moves from a corner case onto the main path.
- **The data-space packing rule is non-trivial** (§4, "Uniform packing": join `F + W + C` across the modes, keep an edge
  of 1, never zero-pad an edge of 1, "it would put the value on entry 0 only"). Under (a) users apply it themselves
  through `pack_data`, which is D5's case for a public, object-bound helper.
- **F3's unpacking is in the plan by design** (§4: "A packed probe `y` is unpacked (`fitting.py:928-929`), and so is a
  packed `ω` with an `N` edge", for `<y, M y>` in `f_resolution`). It is a recorded decision, to revisit under F2/F3.
- **One inconsistency inside the design:** `data_metric` keeps its input's form (a tuple gives a tuple, a packed array
  an array; a packed `r` with a tuple weight raises, M1 decision 2), which is §8.4's R4. But `jet_to_dual_jet`, routed
  through `data_metric` by §3, returns a tuple of padded slices for a packed probe array (checked: `(3, 10, 6)` gives
  `[(10, 6)] * 3`). Making the converters keep their input's form (D7, F7) would match `data_metric`.

### 9.6 The eagle's-eye view (Nick, 2026-11-24)

**Ground rules (Nick).** The M arc's decisions are not binding: M was just built and its review had only begun. The
arc is a deliberate breaking change toward a clean mathematical design, aiming at simplicity, performance and user
control over computational trade-offs. Part 1 is structure; Part 2 is performance. Reason from first principles,
not from fitting the existing code.

**The view.** Every sampling function and subfunction takes inputs in its layer's form: uniform takes packed; ragged
takes a sequence of `d` arrays, or an array that is blindly unpacked. Then the whole of sampling is internally
consistent. The only possible failure is a user preparing inputs wrongly outside sampling, and preparing them by the
stated rules is the user's responsibility.

**Claude's assessment: this is right, and most of §9.4 dissolves under it.** Re-sorted:

- **Not design problems.** D1 came from defining "packed" too narrowly (as "has a padded edge"). With the definitions
  below, `blocks_to_weight`'s `(3, 1, 1)` is a valid packed object, and, being an array, is blindly unpacked into
  three `(1, 1)` arrays on ragged: valid on both, nothing breaks. D6 (the uniform layer is not a drop-in) is the
  design: one opts into a representation by preparing inputs in it. D8 (`pack_if_ragged`, `is_packed`) is migration.
- **Consequences to carry out.** D3: each layer's checkers check that layer's form (today they read only the ragged
  form). D4: the fitting frontend takes a batch in `x0`'s layer, and the hidden per-step pack of a minibatch draw goes
  with it. D9: the refusals name the fix.
- **About the helpers, outside sampling.** D2, D5: the packing helpers should be hard to misuse (take the shape or the
  object, so G1 is checked where packing happens).
- **The one place the view needs a word.** D7: functions with no object (`data_metric`, the jets converters) have no
  layer. The rule that fits the view is "the form out is the form in" (`data_metric` does this; the converters do not,
  F7).

**What remains to define precisely:**

1. **Packed**, for any per-mode object: one array, the mode axis leading with exactly `d` (never broadcast over
   modes), every other axis by the broadcast rule (1 or full), the edge 1 or `N`.
2. **Blindly unpacked**, on ragged: `value[i]` along the first axis, with no slicing. So an array is accepted exactly
   when each row already has its mode's shape (equal widths, or an edge of 1), and the ragged checks raise otherwise.
   Slicing to `N_i` would be a hidden unpack that discards data silently: rejected.
3. **The hazard no check inside sampling can catch:** real data in the padding of a packed input (packed with a wrong
   `N_i`, or garbage). A packed array's `d` and `N` can be checked; its padding content cannot be told from junk.
   Mask-on-entry zeros it silently. Under the view this is the user's preparation, so G1 belongs in the packing
   helper. If masking becomes optional (F4), "the padding is zero" is the stated precondition of the unmasked mode:
   the same principle.

**What the view changes in the M arc's design:** §2 of `DATA_METRIC_PLAN.md` accepts a tuple on the uniform layer
(through the engine's automatic packing); under the view only the packed form is accepted there. And F3's internal
unpacking of a packed `y` in `f_resolution` is a uniform function leaving its layer, which the view excludes; it can
compute `<y, M y>` on the packed array directly (its padding is zero, or masked by the shape it already holds).

**On Part 2.** Row 5.3 of `TARGET_DECISIONS.md` makes the uniform layer's one-site-at-a-time `xmap` a Part 1 choice,
"must be revisited in Part 2". The data-space loops over the modes (F2: `_modes`, the converters) are not marked as
Part 2 items that Claude found. They follow from per-mode tuple handling; under the view, on the uniform layer they
become single elementwise operations on the packed array.

### 9.7 The packing helpers and the checkers (Nick's proposal, 2026-11-24)

**The proposal.** Packing does no checking: the user asks for their vectors to be packed, and they are packed as
asked. Passing a whole uniform T3 to a packer is heavy-handed: one may not exist while the data are prepared, and
building one only to check against it is circular (the shape would be read off the data to build it). Passing a shape
is overkill for the same reason, since the user can see their arrays' shapes. Instead, the backend provides checker
functions for the consistency of data with a uniform T3 and of data with a uniform tangent; in the frontend they become
methods of the classes. The packer itself is a standalone function or a staticmethod, since it is not tied to an
instance.

**Claude's assessment: agreed, with four points.**

1. **Packing erases what G1 checks.** A packed array does not carry the real widths (zeros in the padding do not
   encode them; `pack_vectors`'s own docstring says the fill is never used to infer shape). So G1 (each width equals
   `N_i`) can be checked only on the ragged form, before packing. On packed data a checker can verify the structure
   (`d`, `N`, the broadcast rule, the stacks), which the sampling functions check anyway. It can also verify that the
   padding is zero: that catches data wider than `N_i` (spilled into the padding), but not narrower (a missing real
   entry looks like a zero pad), and it reads the data, so it cannot run under jit. So the checkers should accept the
   ragged form, the workflow is check, then pack, then sample, and **for uniform users the strongest check becomes
   opt-in.** Today the fitting frontend runs it automatically before packing. That is the trade-off to accept
   knowingly.
2. **Unpacking is not symmetric with packing.** Packing needs nothing beyond the arrays. Unpacking needs the widths,
   which the packed array no longer has, so the unpacker takes the shape (backend) and is naturally an instance method
   in the frontend (`x.unpack(z)` reads `x.shape`).
3. **"Pack as asked" is not well defined in one case: a data-space object whose modes have mixed edges** (some 1,
   some `N_i`; a per-entry weight on one mode and a constant on another). To make one array, the edge-1 modes must be
   widened. With zero padding they would need their own `N_i`, which the packer does not have. Today `_pack_data_space`
   broadcasts them to `N`, putting the constant on the padding too (checked: `[1 2 3 4 0 0]` and `[7 7 7 7 7 7]` for
   edges 4 and 1 at `N = 6`), and relies on the engine's masking. That collides with an unmasked mode (F4), whose
   precondition is zero padding. Options: (a) the packer refuses mixed edges, and the user widens the edge-1 modes
   first; (b) widen to `N` and require masking for `y` and `ω` always; (c) a shape argument for this case only.
   Claude leans to (a): the packer stays checkless and total on every other input, and never produces padding that is
   not zero.
4. **One checker may serve both objects.** In the backend a point's or frame's `.data` is a light tuple (unlike asking
   the packer for it), and what the check needs from either is the same: the shape, `N`, and the stack `C`.
   `sampling_variables.frame_shape(frame)` already reads the shape off all four forms (ragged or uniform, frame or
   point). The existing ragged checkers, `jets.check_sample` / `check_data` / `check_weight(kind, order, …, shape,
   stack_shape)`, are already the logic. So the backend checker could be one function taking any of the four forms (or
   the shape and stack directly), and the frontend methods `x.check_…(…)` on the four classes delegate to it. Two
   separate checkers (point, tangent) would duplicate the same logic.

### 9.8 Padding as junk for the data too? (Nick's question, 2026-11-24)

**The question.** For cores, the uniform layer's rule is that padding is junk: anything finite may be there, and an
operation makes no promise about what it writes there (a uniform `t3svd` putting 9's in the padding would be fine).
Does it simplify things to make the same rule for the data passed to sampling functions (the sample, `y`, the
weight): the uniform sampling operations do not care what is in their inputs' padding, and promise nothing about their
outputs' padding?

**Claude's assessment: yes for inputs; keep a guarantee for outputs.**

- **Inputs: yes, and it is already how the engine behaves.** The engine masks every input on entry, and P12b's tests
  put garbage in every padded slot. Stating it as the rule simplifies the helpers. The packer never needs a shape. "Pack as
  asked" is well defined for every input, including mixed edges: widening an edge-1 mode to `N` puts its constant on
  the padding, which is allowed junk, so §9.7's point 3 and its options go away. The fill is anything finite (zeros by
  convenience). One rule then covers everything uniform. The price is that each operation must neutralize its inputs'
  padding. Junk enters only through a contraction over a padded axis, so it is enough to zero one factor of each such
  contraction. On the mode edge that is either the data or the Tucker cores' `N` columns, whichever is cheaper; this
  is relevant to F4. Lost: the "padding is zero" check on packed data, which never covered narrower data anyway (§9.7,
  point 1).
- **Outputs: keep the guarantee that the padding is zero.**
  - *Cores and outputs are different.* With junk allowed, a packed object is an equivalence class (the padding is
    quotiented out). Cores are consumed only by library operations, all defined on the classes (they mask), so cores
    never need a canonical representative. Sampling outputs (`z`, `r`, `Jp` in `Y*`, the transposes' blocks in `X*`)
    are consumed by users and by generic code: numpy reductions, the CG state, optax, external optimizers. All of these use the
    array's own inner product. That inner product is right on the classes only for the zero-padded representative:
    with zero padding, packing is an isometric embedding of the ragged space into the array space.
  - *This already happened once.* 2026.3.0 re-masked every uniform transpose output because the adjoint sweep wrote up
    to ~3e12 into the padding and every consumer, the CG state included, "had to remember to mask on entry"
    (`CHANGELOG.md`, "Uniform transposes return clean padding"; `docs/contributor/uniform_internals.md`).
  - *Junk is not stable under iteration.* "Anything but nan" cannot be promised for outputs that are accumulated: junk
    added into an iterate can grow to `inf`, and then `inf · 0 = nan`. The simplest bound is zero.
  - *It is cheap.* One multiply by a host mask, at most the size of the output, always smaller than the work that made
    it: measured 0.08–1.1 % eager, 0 under jit (`docs/contributor/curvature_internals.md`).
  - *It makes F4 safe in chains.* If outputs are canonical, everything a chain feeds back (held variables, `Jp` into a
    transpose, a residual) is already clean. Only external inputs need cleaning, once (`x.apply_masks()`, the packer's
    zero fill). An option to skip the entry masking then has a precondition, zero padding on the inputs, that the
    library's own outputs always meet. That answers F4's concern about repeated masking along held chains. In safe mode
    the precondition can be checked (it reads data, so it is skipped under jit: the house rule for numerical
    preconditions).
- **The rule this gives: liberal in, canonical out.** Inputs, cores and data alike: padding is junk (finite), and every
  operation masks on entry by default. Outputs of sampling: padding is zero. Later (F4): an option that skips the entry
  masking, with zero padding on the inputs as its precondition. Core-producing operations outside sampling (`t3svd`,
  orthogonalization) keep "no promise", as now.

**What (a) gains, in the same terms:** the engine's uniform entry needs only a check (it has the shape); the 52 mirror
lines, `_is_packed`, `_frontend_draw` and the form branch of `_check_sample` go; every sampling signature has one type
per argument; F5 is settled in favour of the June wording; and the one hidden per-step pack in the library (D4)
disappears.

## 10. A preliminary plan for packing and unpacking (Claude, 2026-11-24; for discussion)

### 10.1 Principles

1. **Each layer takes and returns its own form (the layer rule; Nick's view, §9.6).** A uniform sampling function
   takes per-mode objects packed and returns them packed. A ragged one takes a sequence of `d` arrays, or one array
   read along its first axis, and returns sequences. No sampling function converts between the forms, mirrors its
   input's form, or infers a form from an object's type. A per-mode object in the wrong form raises, naming the
   helper that converts it.
2. **Liberal in, canonical out (G2).** Input padding is junk (finite); output padding is zero.
3. **Packing is a change of representation, not a check.** A packer takes the arrays (and what it needs to place their
   axes) and packs them as asked: no shape, no object, no validation beyond what packing itself needs. Unpacking
   needs the real widths, which a packed array has lost, so the unpacker takes the shape.
4. **Checking is separate and opt-in.** A checker compares data with an object: a point or a frame, ragged or uniform.
   Data in the ragged form are checked fully, widths included (G1). Data in the packed form are checked
   structurally only (`d`, `N`, the broadcast rule, the stacks), since packing erased the widths. The structural checks
   of a layer's own form stay inside the sampling functions, always on.
5. **Functions without an object keep their input's form** (`data_metric`, the jets converters): packed in, packed
   out; a sequence in, a sequence out; mixed forms raise.
6. **The forms, defined once, for every per-mode object** (the sample and the data-space objects alike):
   - *packed*: one array, the mode axis leading with exactly `d` (never broadcast over the modes); every other axis
     one size for all modes; the edge (last axis) `N`, or 1 for a data-space object constant along it;
   - *ragged*: a sequence (tuple or list) of `d` arrays, each with its own mode's shape. **A single array is refused**
     (O2, ruled 2026-11-24: the axis-order coincidence), the entries index included.
   - **Only the edge has padding.** A jet axis brought to the largest `F_in_i` is filled with zero *coefficients*, which
     are data, not junk. Joining a data-space object's `F`, `W` and `C` axes across the modes is broadcasting, not
     padding.

### 10.2 Functions and methods

Shapes and types in the library's grammar (`docs/contributor/signature_style.md`). The homes and names are proposals
(10.4, O3).

**Backend: the packers and unpackers** (`backend/ut3_operations.py`, beside today's `pack_vectors`):

```python
def pack_vectors(                                  # changed: never checks; also takes one array read along its first axis
        vectors:  typ.Sequence[NDArray],           # P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        N:        typ.Optional[int] = None,        # the packed width; None = max(Ni)
        len_P:    int = 0,                         # the number of leading jet axes (sizes may differ by mode)
) -> NDArray:                                      # P_in_i⊗V_i packed, shape=(d,)+F_in+W+(N,): jet axes zero-filled
                                                   #   (coefficients), the edge zero-padded


def unpack_vectors(                                # unchanged
        packed:  NDArray,                          # shape=(d,)+...+(N,)
        shape:   typ.Sequence[int],                # the real widths (N_0, ..., N_(d-1))
) -> typ.Tuple[NDArray, ...]:                      # len=d, elm_shape=...+(Ni,)


def pack_data(                                     # moved from uniform_fitting, with today's _pack_data_space as its body
        kind:   str,                               # 'probe' | 'apply' | 'entries'
        value:  typ.Any,                           # Y* or Y (or omega), of the type of z: probe len=d, elm_shape=F+W+C+(e_i,),
                                                   #   e_i in {1, Ni}; apply F+W+C; entries W+C
        N:      typ.Optional[int] = None,          # the packed width; None = the largest full edge
) -> typ.Any:                                      # probe: shape=(d,)+F+W+C+(e,), each axis the join over the modes, e = 1
                                                   #   if every e_i is 1, else N; apply, entries: value unchanged


def unpack_data(                                   # new
        kind:   str,                               # 'probe' | 'apply' | 'entries'
        value:  typ.Any,                           # of the type of z, packed: probe shape=(d,)+F+W+C+(e,); apply F+W+C; ...
        shape:  typ.Sequence[int],                 # the real widths (N_0, ..., N_(d-1))
) -> typ.Any:                                      # probe: len=d, elm_shape=F+W+C+(e_i,), an edge of 1 kept; else unchanged
```

`pack_batch` stays in `backend/uniform_fitting.py`, a composite of the two packers:

```python
def pack_batch(                                    # changed: never checks; N optional
        kind:   str,                               # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],               # the jet shape (len_P = len(order) for the sample)
        batch:  'bfit.LeastSquaresBatch',          # the sample, y and the weight, ragged
        N:      typ.Optional[int] = None,          # the packed width; None = max(Ni) of the sample
) -> 'bfit.LeastSquaresBatch':                     # the sample, y and the weight packed to N
```

**Backend: the checkers** (`backend/jets.py`, generalizing today's three). *Superseded by §10.9's O3 addendum: the
checkers are split along the types (`check_vectors`, `check_index`, `check_probe_data`, …), take the ragged form only,
and are frontend methods of the object checked against.* One set serves every object: what a check
needs from an object is its shape, its `N` on the uniform layer, and its stack `C`, which
`sampling_variables.frame_shape` and the frame stack already read off all four forms.

```python
def check_sample(                                  # changed: an object instead of a shape; either form of the sample
        kind:    str,                              # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],              # the jet shape; entries: ()
        sample:  typ.Any,                          # P_in_i⊗V_i, ragged (len=d, elm_shape=F_in_i+W+(Ni,)) or packed
                                                   #   ((d,)+F_in+W+(N,)) | index^d ((d,)+W, int)
        frame:   typ.Tuple,                        # the object: a point or a frame, ragged or uniform (any of the four forms)
) -> None:                                         # raises a ValueError naming the mismatch; ragged: the widths too (G1)


def check_data(                                    # changed: an object instead of stack_shape; serves y, omega, r, c
        kind:    str,                              # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],              # the jet shape
        sample:  typ.Any,                          # P_in_i⊗V_i | index^d, a good sample (W is read off it)
        value:   typ.Optional[typ.Any],            # of the type of z, ragged or packed (each axis 1 or z's size), or None
        frame:   typ.Tuple,                        # the object (any of the four forms): the shape, N and the stack C
) -> None:                                         # raises a ValueError naming the mismatch
```

`check_weight` folds into `check_data` (one rule for every data-space object). `LeastSquaresOracle.check_batch(batch,
frame)` replaces `(batch, shape, stack_shape)`.

**Backend: the sampling functions.**

- **The engine** `sampling_variables(...)`. On the uniform layer `sample`, `y`, `weight` and held variables must be
  packed; a sequence raises, naming `pack_vectors` / `pack_data`. On the ragged layer they are sequences, or arrays
  read along the first axis. The `'w'` echo is masked like every other output (canonical out; today it returns the
  input with its padding). One new keyword, for F4, whose timing is open (O1):

  ```python
          mask_inputs:  bool = True,               # False: every uniform input has zero padding (a precondition,
                                                   #   checked in safe mode); skips the masking on entry
  ```

- **The uniform twins** (`ut3_*`, `utv_*`): the `Vectors` union goes; per-mode arguments are `NDArray`, packed; probe-type
  returns are packed. The 52 lines of `pack_if_ragged` / `unpack_vectors` / `is_packed` go. Each twin is a call of the
  engine.
- **The ragged functions** (`t3_*`, `tv_*`): unchanged in behaviour (they already take sequences or arrays read along
  the first axis); their comments state the ragged form once, as 10.1 item 6 defines it.
- **The jets converters** (`jet_to_dual_jet`, `dual_jet_to_jet`, `derivatives_to_taylor`, `taylor_to_derivatives`,
  `prepare_derivative_data`): keep their input's form (F7). For a packed probe array that is one elementwise
  operation along the jet axes at offset 1, instead of a loop over the modes returning a tuple.
- **`data_preparation.ut3_unscale_coordinates`**: `scales` packed, `shape=(d, N)` (today automatically packed).
- **`fitting._y_times_metric_y`** (F3): computes `<y, M y>` on the packed array directly (its padding masked by the
  shape it already holds), instead of unpacking per mode.

**Frontend.**

```python
class UniformTuckerTensorTrain:                    # and UT3Tangent; the checkers also on TuckerTensorTrain, T3Tangent
    @staticmethod
    def pack_vectors(
            vectors:  Sequence[NDArray],           # P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
            N:        Optional[int] = None,        # pass self.N to match a point padded wider than max(Ni)
            len_P:    int = 0,                     # the number of leading jet axes
    ) -> NDArray:                                  # P_in_i⊗V_i packed, shape=(d,)+F_in+W+(N,)

    @staticmethod
    def pack_data(kind, value, N=None): ...        # as the backend pack_data

    def unpack_data(                               # an instance method: it needs self.shape
            self,
            kind:   str,                           # 'probe' | 'apply' | 'entries'
            value:  NDArray,                       # of the type of z, packed: probe shape=(d,)+F+W+C+(e,)
    ) -> Any:                                      # probe: len=d, elm_shape=F+W+C+(e_i,); else unchanged

    def check_sample(
            self,
            kind:    str,                          # 'probe' | 'apply' | 'entries'
            sample:  Any,                          # P_in_i⊗V_i, ragged or packed | index^d
            order:   Tuple[int, ...] = (),         # the jet shape
    ) -> None:                                     # raises a ValueError naming the mismatch

    def check_data(self, kind, sample, value, order=()): ...   # likewise, for y, omega, r, c
```

`t3toolbox.fitting` gains a frontend batch packer (the oracle supplies `kind` and `order`):

```python
def pack_batch(
        oracle:  LeastSquaresOracle,               # its kind and order
        batch:   LeastSquaresBatch,                # ragged
        N:       Optional[int] = None,             # pass x0.N; None = max(Ni) of the sample
) -> LeastSquaresBatch:                            # packed
```

- **`fitting.prepare_batch`** and the optimizers: for a uniform `x`, the batch must already be packed; it is
  checked structurally in its packed form, and a ragged batch raises, naming `fitting.pack_batch`. `_is_packed` and
  `optimizers._frontend_draw` go: a minibatch draw returns batches in `x0`'s layer (pack once, draw with `take`).
- **The uniform frontend sampling methods** lose the mirror and `_check_sample`'s branch on the form.

**Removed:** `ut3_operations.is_packed`, `pack_if_ragged`, `_pack_data_space` (folded into `pack_data`);
`fitting._is_packed`, `fitting._arrays_of`'s unpacking; `optimizers._frontend_draw`; the `Vectors` alias.

**Documentation:** `uniform_equivalence_contract.md`'s vector-I/O section rewritten (the contract becomes
`unpack_data(op_uniform(pack(v)), shape) == op_ragged(v)` on the real parts); `uniform_polymorphism.md:44-47` made
true as written (F5); the glossary's "packed vectors" entry; the two uniform examples gain their explicit packing (P17
switches the examples anyway); an upgrade note.

### 10.3 Pros and cons

**For:**

- One rule, stated in two lines, with no exceptions inside sampling: predictable.
- No form is inferred from an object's type anywhere in sampling. This is slice P3's ruling ("the type is decided by
  `kind`, never by the type of the object") carried to its end.
- No hidden conversions. The one hidden per-step pack in the library (`_frontend_draw`) goes, and so does F1's
  silent zero-padding inside sampling.
- Less code: the 52 mirror lines, `is_packed`, `pack_if_ragged`, `_is_packed`, `_frontend_draw`, and the form
  branches. One type per argument in every sampling signature.
- The packers are total and shape-free; the mixed-edge question disappears under G2.
- Canonical outputs keep chains clean and make F4's unmasked mode safe on everything the library produces.

**Against:**

- Uniform users pack explicitly: one more line in the uniform examples (the "only two changes" story becomes three),
  and probe outputs read per mode need `unpack_data`.
- **G1 becomes opt-in on the uniform layer.** Today the fitting frontend checks a ragged batch's widths before it
  packs. Afterwards a packed batch can be checked only structurally, and the width check runs only if the user calls
  a checker before packing.
- Breaking: 135 tests, 13 module doctests, `sharing.md`, the two uniform examples (§9.2), and any backend user
  calling `pack_if_ragged` or `is_packed`.

### 10.4 Outstanding issues

- **O1. The masking option (F4).** Its name, and whether it belongs in Part 1 or Part 2 (a performance option "changes
  how the computation is carried out", the Part 2 line). Also: which side of a padded contraction to mask (the data or
  the Tucker cores' `N` columns), by measurement; and whether masking the outputs should also be optional.
- **O2. The ragged layer's one-array form for the sample** (Nick's undecided exception). It is already the rule for
  data-space objects and the engine's behaviour for the sample; keeping it makes the two consistent.
- **O3. Homes and names.** Packers in `ut3_operations` or a new `backend/packing.py`; `pack_vectors` / `pack_data`
  (typed by what is packed: the sample's `V_i` or the data space) or one `pack(kind, …)`; the frontend packers as
  staticmethods on `UniformTuckerTensorTrain` or as module functions. `pack_data`'s `order` is unused today (checked:
  its body reads only `kind` and `N`), so it can go.
- **O4. Is G1 opt-in acceptable for uniform fitting?** The only way to keep the width check automatic is for something
  to see the ragged batch and the model together. That is either a packer that checks (excluded by principle 3) or an
  optimizer that takes a ragged batch (automatic packing, excluded by principle 1). So under the plan the check is
  opt-in by construction. Claude's lean: accept it, and document check → pack → sample on the packers and the uniform
  pages.
- **O5. Mixed edges and canonical inputs.** `pack_data` widens an edge-1 mode to `N` with its constant, junk on the
  padding (allowed by G2). Its output is canonical except in that case, so the unmasked mode's safe-mode check would
  reject it. Either document it, or let `pack_data` zero the padding when it has the widths (it does not, by
  principle 3).
- **O6. Public functions that return junk padding today:** the bare `utv_*_transpose_from_sweep` seam halves (their
  callers re-mask). Under G2 a public function returns canonical output; they go in P20 with the old kernels, or
  re-mask until then.
- **O7. The remaining findings:** F2 (the per-mode loops, case by case), F3 (above), F5 and F6 (follow from the plan).
- **O8. Sequencing:** deferred to the later planning process (Nick: this session decides what, not how). Two facts for
  it: P17 switches the examples and pages anyway; the old jet wrappers go in P20.

### 10.5 Splitting the data packers by kind (Nick's proposal, 2026-11-24)

**The proposal.** `pack_data` becomes `pack_probe_data`, `pack_apply_data`, `pack_entries_data` (and likewise for
unpacking), so that each signature states its input and output types; shared machinery lives in private
subfunctions.

**What the split shows.** Only the probe kind has anything to pack. Apply's data-space objects are `P` (shape
`F+W+C`) and entries' are `ℝ` (`W+C`): no mode axis, so their packed and ragged forms are the same array.
`pack_apply_data`, `pack_entries_data` and their unpackers are identities. That is not an argument against the split:
the split makes the fact visible, where `pack_data(kind, …)` hid it.

**The sample side, by the same logic.** The sample's type does not depend on the kind: probe and apply both take
`P_in_i⊗V_i` (input jets, `d` of them), and entries' `index^d` is the same array on both layers. So the sample
packer stays one function, `pack_vectors`, with no kind. Today's `uniform_fitting.pack_sample(kind, …)`, whose only
kind logic is "entries: unchanged", dissolves.

**Downstream, directly.**

- `pack_batch` (one library caller of `pack_data` today) keeps the single kind switch. It packs a kind-generic container
  (`LeastSquaresBatch`), so it is genuinely three operations in one: the case where slice P3's ruling found a `kind`
  argument "acceptable and sometimes even desirable". It dispatches to the split functions, and calls
  `pack_vectors` for the sample of probe and apply.
- The frontend batch packer `fitting.pack_batch(oracle, batch, N)` reads the kind from the oracle, unchanged.
- Users mostly unpack probe outputs (`z`, `r`, `Jp`): `unpack_probe_data(z, shape)`, and in the frontend the instance
  method `x.unpack_probe_data(z)`.
- The equivalence contract becomes explicit per kind: `unpack_probe_data(ut3_probe(x, pack_vectors(ww)), shape) ==
  t3_probe(x, ww)`; `ut3_apply(x, pack_vectors(ww)) == t3_apply(x, ww)`.
- Renames: 9 lines in the tests, 6 in the docs and examples (`pack_data(`), plus the glossary and the frontend
  staticmethods. jit is unaffected (the kind was static either way).

**The identity functions: keep or omit?** Kept, a pipeline that is generic over the kind (a user's own loop, or
`pack_batch`'s table) calls `pack_{kind}_data` with no special case, and the identity's docstring answers "does apply
data need packing?" Omitted, there are fewer functions, none of which suggests a step that does nothing, and the
apply and entries signatures state "the same on both layers". Claude's slight lean: keep them; they cost nothing and
make the kind-generic recipe uniform.

**The broader question: is this a one-off or a principle?** Thirteen public backend helpers take `kind` first and
return different types by kind: `jets.blocks_to_weight`, `jet_to_dual_jet`, `dual_jet_to_jet`,
`derivatives_to_taylor`, `taylor_to_derivatives`, `derivative_weight_to_taylor_weight`, `prepare_derivative_data`,
`check_sample`, `check_data`, `check_weight`, and `data_preparation.data_block_norms`, `rescale_direction`,
`rescale_coordinates`. (The engine, `computed_names` and the dense definitions in `sampling_definitions` take `kind`
too, as the generic machinery.) The sampling functions themselves are already split by kind (`t3_probe`, `t3_apply`,
…). Three positions:

- **(A) A one-off for packing.** The packers match the sampling functions; the other helpers keep `kind` (slice P3's
  ruling), their types stated in the comment after a colon, as M5 does.
- **(B) A principle for every helper a user calls with kind-specific data in hand.** Split by kind where the types
  differ; keep `kind` only in kind-generic machinery (the engine, the oracle, the batch, `pack_batch`). This reverses
  slice P3's ruling for most of the thirteen and roughly triples them. The jet helpers have no entries version (entries
  takes no jets), which trims it somewhat.
- **(C) No split anywhere:** `kind` arguments throughout, with type clarity from the M5 comments alone.

The dividing line in (B), machinery versus typed user-facing functions, is the one the sampling layer already has (the
engine takes `kind`; `t3_probe` does not). Claude's view: (B) is the consistent end state, but it is a separate,
larger decision. (A) is safe to do now and does not prejudge it, since the packers would be split under (B) anyway.

### 10.6 G3 across the library: an inventory (2026-11-24)

An AST scan of every public function and method, backend and frontend, for a `kind` parameter or a return annotated
`typ.Any` or `Union`: 108 hits (13 with `kind` only, 52 with such a return only, 43 with both). The scan script was a
one-off in the session scratchpad. By cause:

| Group | Functions | Under G3 |
|---|---|---|
| **A. A `kind` argument selecting different operations** | `jets`: `jet_to_dual_jet`, `dual_jet_to_jet`, `derivatives_to_taylor`, `taylor_to_derivatives`, `derivative_weight_to_taylor_weight`, `prepare_derivative_data`, `blocks_to_weight`, `check_sample`, `check_data`, `check_weight` (10). `data_preparation`: `data_block_norms`, `rescale_direction`, `rescale_coordinates` (3; the last also in the frontend `fitting`). `uniform_fitting`: `pack_sample`, `pack_data` (2, the packing plan) | split, along the types (below) |
| **A'. The reference definitions** | `sampling_definitions`: 37 functions, one per engine variable, keyed by `kind` (backend only, the dense twins of the engine's rows) | open: the engine's exception (they mirror its table), or split |
| **B. Generic machinery** | the engine `sampling_variables` and `computed_names`; `LeastSquaresOracle.query`, `Oracle.query` (a return aligned with `want`); `pack_batch` (a kind-generic container); `geometry.*.precompute` (the protocol's opaque aux); `LocalModel.jacobian`, `optimizers.padded_start`, `uniform_minimal` | the sanctioned relaxation, kept confined to machinery |
| **C. Same operation, the return following the input's representation** | `common.xcat`, `xappend`, `xprepend`; `tt_reverse`, `tt_squash_tails`, `tt_left_orthogonalize`, `tt_right_orthogonalize`, `t3_absorb_tucker_into_tt`; `geometry.*.aligned_padded_frame`; `jets.affine_jet`, `data_metric`, `LeastSquaresBatch.data_metric` | a reading of "one return type" to settle (below) |
| **C'. A `Union` for another reason** (stacked or not, a flag) | `ranks` (3), `ut3_conversions` (2), `ufv_conversions` (3), `fv_conversions` (2) | to classify one by one |
| **D. Going anyway** | `probing.compute_*` / `assemble_*` (22) and the old kinds (P20, `PART1_PLAN.md:2238-2240`); `ut3_probe_derivatives` (the old jet wrapper, and the mirror) | none |

**Split along the types, not the kind labels.** G3 asks for one function per mathematical operation, and two kinds
sometimes share an operation. Examples from the packing plan:

- the sample of probe and of apply is the same type (`P_in_i⊗V_i`, `d` input jets): one packer (`pack_vectors`) and one
  checker for both. The entries index is another type, with its own checker and nothing to pack;
- the data-space objects differ by kind (probe `⊕ P⊗V_i*`, apply `P`, entries `ℝ`): `pack_probe_data` and
  `check_probe_data`, and so on.

The same reading applies to group A. For example the dual-jet converters differ between probe (per mode, the jet axes
after the mode axis) and apply (one array), and entries has no jets, so they become two functions each, not three.

**A reading to settle for group C: "one return type" as "the return type is fixed by the parameter types".**

- A **string** choosing the return type (group A) violates it.
- A return that **follows the input's representation** (a ragged tuple in, a ragged tuple out; a uniform array in, a
  uniform array out) is the same operation over two representations. Under the layer rule the input's form
  determines the output's, so it is parametric polymorphism and can be annotated as such, with a `TypeVar` in place of
  `Union` or `Any` (`R = TypeVar('R', Tuple[NDArray, ...], NDArray)`, then `def tt_reverse(cores: R) -> R`). This
  keeps the library's ragged/uniform polymorphism ("the ragged name is the polymorphic name") and states its type
  precisely.
- A return aligned with a **value** (the engine's and the oracle's `want`) is the sanctioned relaxation of group B.

**Size.** Group A is 15 public functions. Split along the types they become roughly 35 (the jet helpers have no
entries version). Group C, kept, changes annotations only.

**Nick's rulings (2026-11-24):**

- **A: split** (along the types).
- **A': left as they are.** An exception is granted for the reference definitions in `sampling_definitions`.
- **B: intentional and sanctioned.**
- **C: parametric polymorphism is the idea.** A `TypeVar` annotation is ideal if it works. Checked so far: at runtime it
  does. With `NDArray` (`Union[np.ndarray, jax.Array]` when jax is present),
  `Cores = TypeVar('Cores', Tuple[NDArray, ...], NDArray)` and `def f(cores: Cores) -> Cores` give
  `get_type_hints(f) == {'cores': ~Cores, 'return': ~Cores}`. The docs build is not yet checked: the
  `sphinx_autodoc_typehints` extension is not in the local env. The library's one existing `TypeVar`
  (`common.CarryType`, unconstrained) already builds clean under `-W` in CI.
- **The rest (C', and any member of C that is not a pure representation-following function): case by case.** Go
  through the examples one at a time and let the principles emerge from them, rather than prescribing rules ahead of
  time.

### 10.7 The C' cases, one at a time (2026-11-24)

The ten fall under three causes: the five `ut3_*_to_t3*` conversions (example 1); the three `ranks` functions; and
`fv_to_t3` / `t3_orthogonal_representations`, which follow the input's layer (group C proper, written as an explicit
`Union`), the second with a uniform-only argument.

**Example 1: `ut3_to_t3`**, standing for `ut3weights_to_t3weights`, `ut3frame_to_t3frame`,
`ut3variations_to_t3variations` and `ut3frameweights_to_t3frameweights` (the same cause).

- *What it returns.* An unstacked input gives one ragged T3, `(tucker_cores, tt_cores)`. A stacked input (`C ≠ ()`)
  gives a nested tuple shaped like `C` whose leaves are ragged T3s. A uniform stack may have different ranks per
  element, and a ragged T3 with a stack shares its ranks across the stack, so one stacked ragged object cannot hold it.
  The frontend `UniformTuckerTensorTrain.to_t3()` (and `to_t3weights()`) mirrors this: a `TuckerTensorTrain`, or a
  tuple tree of them.
- *Why it is a `Union`.* The return type depends on `len(C)`, a property of the input's shape, not on its Python type.
- *It is two operations.* The stacked branch is "unstack, then convert each": `ut3_unstack` already exists (a stacked
  uniform T3 to a tree of unstacked ones). The function bundles the conversion of one object with the map of that
  conversion over a tree.
- *The two forms are structurally ambiguous.* Run on `(4, 5, 6)`: unstacked gives `tuple(tuple of 3 arrays, tuple of 3
  arrays)`; stacked at `C = (2,)` gives `tuple(T3, T3)`, also a pair of tuples. A consumer can tell them apart only by
  knowing `len(C)`. This is the kind of read-by-structure that slice P3's ruling of 2026-11-17 removed from the jets helpers.
- *Callers.* In the library, only the frontend `to_t3` (`uniform_tucker_tensor_train.py:254`) and the function's own
  stacked loop (`ut3_conversions.py:174`).

Options:

- **(a) Split by operation.** `ut3_to_t3` converts one unstacked object (one type in, one out) and raises on a stack,
  naming the alternative. The stacked conversion is the composition `ut3_unstack`, then `ut3_to_t3` on each leaf. It
  is provided as its own named function (say `ut3_to_t3_tree`), because mapping over a nested tree of a given depth
  is the fiddly step the backend razor's corollary says not to leave to the user. The frontend splits the same way
  (`to_t3()` for one; `unstack()` then convert, or a `to_t3_tree()`).
- **(b) Keep one function, typed as a recursive type:** "a tree of depth `len(C)` of ragged T3s", whose depth-0 tree is
  the leaf. That is one type in name, but the structural ambiguity stays for any consumer that does not track the depth.
- (An object ndarray of shape `C` would make the shape explicit, but it is not a jax pytree and is foreign to the
  library: not pursued.)

Claude's recommendation was (a): the two branches are different operations, and (a) removes the ambiguity rather
than documenting it.

**Ruling (Nick, 2026-11-24): (b).** The intention is a recursive tree; the question was how to type it in Python.

- **How (tested on 3.9.19 and 3.11.15; the library's floor is 3.9, so no `type` statement):**

  ```python
  T3Data = typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]   # (tucker_cores, tt_cores)
  T3Tree = typ.Union[T3Data, typ.Tuple['T3Tree', ...]]                   # a leaf, or a tuple of subtrees

  def ut3_to_t3(x: UT3Data) -> T3Tree: ...
  ```

  `get_type_hints` resolves one level and leaves the inner `ForwardRef('T3Tree')` (Python's recursion guard). A
  generic `Tree = Union[Leaf, Tuple['Tree[Leaf]', ...]]` reads well, but at runtime `Tree[T3Data]` substitutes only the
  outer level (the inner level still says `~Leaf`). Static checkers handle recursive generic aliases, but the project
  runs none. So: **one concrete alias per leaf type**, five one-liners (the T3, the T3 weights, the frame, the
  variations, the frame weights).
- **Docs rendering: unverified.** Autodoc may print the expanded `Union[..., Tuple[ForwardRef('T3Tree'), ...]]` rather
  than the name. The typehints extension is not in the local env. The signature comment carries the contract anyway:
  `# T3Tree: a tree shaped like C (depth len(C)); a leaf is (tucker_cores, tt_cores)`.
- **The ambiguity is handled by the depth.** A leaf is itself a pair of tuples, so `len(C)` is part of the contract, and
  consumers use the depth-aware tree helpers of `stacking.py`, as the library already does.
- **A principle that emerged: a recursive container type is one type.** The `Union` in a tree alias is the type's
  definition (a leaf or a node), a sum type. It is not a choice between return types made by a switch, so G3 is met.

**Example 2: the `ranks` functions** `compute_minimal_ranks`, `oversampled_ranks`,
`compute_orthogonal_representation_ranks`.

- *What they do.* Integer rank recurrences on structure: the minimal ranks; the ranks plus `p`, made minimal; the four
  store widths `(nD, nU, rL, rR)` of the orthogonal representation.
- *Two representations of a rank vector.* An **int tuple**, for one T3: static, hashable structure, which the ragged
  layer uses as shapes, as jit static arguments and as dataclass fields. Or a **HOST int array** `(d,)+C` /
  `(d+1,)+C`, for a stack whose elements may have different ranks: what a uniform stack's masks give. The return comes
  in the input's representation. Callers: the ragged side (the `TuckerTensorTrain` frontend, rank continuation,
  `padded_frames`) passes tuples; the uniform side (`ut3_svd` on mask sums, `uniform_padded_frames`, the uniform
  frontend) passes arrays.
- *It is one operation.* `compute_minimal_ranks` converts the input to an array, runs one recurrence elementwise over
  the stack, and converts back to the input's container. The tuple is the `C = ()` case. By its cause this is group C
  (parametric polymorphism), filed under C' only because its annotation spells out the `Union`.
- *Typing: one constrained `TypeVar`, shared by both rank parameters and the returns.*

  ```python
  Ranks = typ.TypeVar('Ranks', typ.Tuple[int, ...], NDArray)   # one rank vector (int tuple), or a stack of them (HOST int)

  def compute_minimal_ranks(
          shape:         typ.Sequence[int],          # len=d
          tucker_ranks:  Ranks,                      # len=d; or HOST int, shape=(d,)+C
          tt_ranks:      Ranks,                      # len=d+1; or HOST int, shape=(d+1,)+C
          sharing:       typ.Optional[typ.Sequence] = None,
  ) -> typ.Tuple[
          Ranks,                                     # minimal tucker ranks, the input's representation
          Ranks,                                     # minimal tt ranks
  ]:
  ```

  A constrained `TypeVar` used by several parameters resolves to one constraint per call, so a checker reads "both
  rank vectors in one representation" off the signature. One narrowing: today's inputs are annotated `Sequence[int]` (a
  list is accepted) while the outputs are tuples. With the `TypeVar` the annotation says tuple; the runtime may still
  accept a list.
- *Mixed inputs today* (one tuple, one array) are not a clean error. `compute_minimal_ranks` fails inside numpy
  ("TypeError: only 0-dimensional arrays can be converted to Python scalars", or "setting an array element with a
  sequence … inhomogeneous"), because it decides the form by `tucker_ranks` alone. `oversampled_ranks` raises a clear
  structural error ("tucker_ranks (3,) and tt_ranks (4, 2) must share one stack C"). The other two should do the same.
- *An aside, in the same functions.* `compute_minimal_ranks` and `compute_orthogonal_representation_ranks` take
  `use_jax: bool = False`. No caller in the library or the tests passes `True`. Ranks are static structure, host numpy
  by the uniform masks rule; with jax they would become device arrays, tracers under jit. The house rule keeps
  `use_jax` only for pure constructors with no array input. So it looks dead and against the grain.

Claude's recommendation: keep one function each (parametric, under the ruling for C); annotate with the shared
`TypeVar`; make mixed forms a clear structural error in all three; remove `use_jax`.

**Ruling (Nick, 2026-11-24): agreed on all four.** Nick asked whether `use_jax` has a reason or is "a random dinosaur
that survived the meteor". Its history says dinosaur:

1. **2026-04-30, the research-code era.** Added to `compute_minimal_ranks` (`75a07d89`, "making uniform masking work
   with stacks properly") and `compute_orthogonal_representation_ranks` (`e7caf156`, the same day). Masks were then
   built with the array backend, so a jax-backed uniform T3 needed jax rank arrays to build its masks: the original
   reason.
2. **Mid-June.** `compute_raw_sweep_ranks` (`0d32e9c0`, for `ut3_svd`) copied the flag. The frontend `minimal_ranks`
   passed `use_jax` through (`e7ce8bfc`).
3. **2026-06-18, the meteor.** The host-mask refactor (uniform slice 7, `8dc39cc8`) made masks always host numpy
   ("supercores → `xnp`; masks → `np`"). The reason was gone: jax rank arrays would leak tracers into `aux_data`.
4. **2026-08-22/23, the review worked around it instead of removing it.** R8-9 ("`minimal_ranks` returns jax arrays on a
   jax-backed UT3") was fixed by pinning the call to `use_jax=False` (`ccab4ed0`). S1a (`18d31413`) made
   `compute_orthogonal_representation_ranks` host numpy throughout, its docstring now reading "`use_jax` is accepted for
   compatibility and ignored".
5. **Today.** `compute_orthogonal_representation_ranks` ignores it. `compute_minimal_ranks` and
   `compute_raw_sweep_ranks` still honour it, so `True` would still produce the jax rank arrays the host-numpy rule
   forbids. No caller passes `True`. The newest sibling, `oversampled_ranks`, never had it ("HOST: ranks are static
   structure").

So: remove it from all three (`compute_raw_sweep_ranks` too, though not in C'), and make ranks host numpy throughout.

**Example 3: `fv_conversions.t3_orthogonal_representations` and `fv_conversions.fv_to_t3`**, filed under C' because
their annotations spell out a ragged/uniform `Union`.

*3a. `t3_orthogonal_representations(x, already_left_orthogonal, squash_tails, uniform_masks=None)`* (the T3 to its
orthogonal frame and variations: the T4S Algorithm 11 sweep). It has three paths:

1. **ragged**: the ragged orthogonalizations;
2. **uniform without `uniform_masks`**: the plain-SVD sweep, the one the S1b fix (2026-08-23) found loses tangent
   directions at numerically rank-deficient points. It has **no caller**: the only uniform caller,
   `ufv_conversions.ut3_orthogonal_representations`, always passes the masks, and `geometry.py:323` is the ragged
   geometry;
3. **uniform with `uniform_masks`**: the pad-safe sweep, with its own mask recurrences (`fv_conversions.py:197-215`),
   masked SVDs and completion zeroing at every step. It returns bare supercores; the wrapper then builds the frame and
   variation masks.

So this is not one operation over two representations. It is two algorithms that share a skeleton (the sweep order
and the polymorphic `tt_left/right_orthogonalize`), with a uniform-only argument, plus a dead path that is the unsafe
one. The library's own polymorphism lens already says so ("orthogonalization returning a T3-like object →
uniform-specific: masks entangle with the result's structure; inside a scalar read-out → polymorphic",
`dev/archive/uniform_fix_plan.md`), and S1b proved it.

Recommendation: `t3_orthogonal_representations` takes the ragged form only (one type in, one out). The pad-safe sweep
becomes the body of `ufv_conversions.ut3_orthogonal_representations`, already its only uniform caller, which returns
complete uniform frame and variation `.data`, masks included. The unmasked uniform path is deleted. The shared steps
stay shared through subfunctions (`tt_left_orthogonalize` / `tt_right_orthogonalize` with `pad_masks`, the
`uniform_orth` functions).

*3b. `fv_to_t3(index, frame, variations)`* (term `index` of a tangent's sum: the frame with core `i` replaced by its
Tucker or TT variation).

- The code is parametric (the same substitution on tuples or on supercores), but the uniform branch returns two bare
  supercores, with no shape and no masks: not a uniform T3. It runs only when the padded widths agree. Checked on a
  real uniform frame: it returns `[(3, 3, 6), (3, 3, 3, 3)]` because every padded width there is 3. The module records
  why the general uniform op was dropped: `uniform_frame_variations_format.py:1182`, "`ufv_to_ut3` … was dropped: the
  left/right subchains become differently-shaped supercores glued by the variation, with no clean single uniform
  supercore op".
- No library caller passes uniform data: `tv_operations.tv_to_t3`, the frontend and the tests are all ragged.

Recommendation: `fv_to_t3` takes the ragged form only. A uniform twin comes only if a need arises, and then returns a
complete uniform `.data` with masks (the substituted T3's TT edge masks are the left masks before site `i` and the
right masks after it).

A side note on `index: (bool, int)`: the bool picks a Tucker or a TT term. It indexes the `2d` terms of the tangent's
sum (the same output type, the same operation of substituting one core), so it reads as data, not as a switch between
operations. Claude would keep it; Nick's call under G3.

**A principle emerging from 3a and 3b: parametric polymorphism needs parallel outputs.** One function serves both
layers only when it does the same operation and returns a complete object of the input's layer on each. A uniform
branch that needs an extra input (3a's masks), or returns an incomplete object (3b's maskless supercores), is a
different function.

**Ruling (Nick, 2026-11-24): agreed on both** (3a split by layer, the unmasked uniform path deleted; 3b ragged only).
"The index is data, not a switch between operations." Nick: "This is cleaning up old messes I had forgotten about."

**What the C' cases taught** (the principles that emerged, for G3's reading of "one return type"):

1. **A recursive container type is one type** (example 1). A tree of a given leaf type is a sum type by definition;
   one concrete alias per leaf type, with the depth (`len(C)`) in the contract.
2. **The same operation over two representations is parametric** (example 2). The return follows the input's
   representation; it is annotated with one constrained `TypeVar` shared by the co-varying parameters and the returns,
   and mixed representations are a structural error.
3. **Parametric polymorphism needs parallel outputs** (example 3). A branch with an extra input, or with an incomplete
   output, is a different function, and is split.
4. **An index is data, a switch is not** (example 3b). An argument that selects which of a family of like terms to
   compute (same output type, same operation) is data. An argument that selects between different operations or
   different output types is a switch, and G3 splits it.
5. **Dead or unsafe branches found on the way go** (the unmasked uniform orthogonalization; `use_jax` in the ranks).

### 10.8 The group C members, one at a time, against §10.7's principles (2026-11-24)

**C1: `common.xcat`, `xappend`, `xprepend`.**

- *What they are.* The backend's representation primitives, in the family of `xnp`, `xmap` and `xscan`: concatenate,
  append, prepend along the per-site axis, for a tuple of per-site arrays (ragged) or one array with a leading `d`
  (uniform; `jnp` or `np` by the inputs). Callers: `t3_orthogonalization` (substituting one site's cores, `:384-421`)
  and the engine's bond assembly (`sampling_variables._with_end`, `:1329`).
- *Against the principles.* The same operation over two representations of a sequence of `d` per-site arrays; the
  return follows the input (principle 2); each output is complete in the input's representation (principle 3); no
  switch (principle 4). Mixed forms already raise a clear `TypeError` both ways (checked: "expected an ndarray to
  append, got tuple"; "expected two sequences, got tuple and ndarray").
- *Typing.* One `TypeVar` for "a sequence of per-site arrays, ragged or stacked", which the `tt_*` operations (C2)
  could share:

  ```python
  PerSite = typ.TypeVar('PerSite', typ.Tuple[NDArray, ...], NDArray)   # len=d per-site arrays, or (d,)+... stacked

  def xcat(x: PerSite, y: PerSite) -> PerSite: ...
  def xappend(S: PerSite, x: NDArray) -> PerSite: ...      # x: one site's array (a slice of a stacked S)
  def xprepend(x: NDArray, S: PerSite) -> PerSite: ...
  ```

- *One quirk.* With an empty sequence on either side, `xcat` returns the other argument unchanged, so a list comes
  back as a list (checked: `xcat((), [a, b])` is a `list`), while every other sequence path returns a tuple. Under the
  annotation the sequence branch always returns a tuple: `tuple(y)`, `tuple(x)`.

Recommendation: keep (the model case of principle 2); annotate with the shared `TypeVar`; make the empty branches
return tuples. The `TypeVar`'s name is open (`PerSite` is a placeholder).

**Ruling (Nick, 2026-11-24): agreed.**

**C2: `tt_operations.tt_reverse`, `tt_squash_tails`.** (The two orthogonalizations, `tt_left_orthogonalize` /
`tt_right_orthogonalize`, are C3: they have issues of their own.)

- *What they do.* `tt_reverse` reverses a TT chain (the site order, and the two bond axes of each core).
  `tt_squash_tails` collapses the leading and trailing bonds to one without changing the tensor: ragged cores shrink
  to boundary rank 1, while a uniform supercore keeps its shape, with the bond summed into slot 0 and the rest
  zeroed. Each layer represents a rank its own way (ragged by shape, uniform by mask). Both are widely used on both
  layers (`tt_squash_tails`: 24 calls in 9 modules, the uniform ones in `ut3_linalg`, `ut3_svd`, `ut3_operations`,
  `ut3_conversions`).
- *Against the principles.* The same operation over two representations of the cores; the output is complete in the
  input's representation as raw cores (principles 2 and 3); no switch. Annotated with C1's `PerSite` `TypeVar`.
- *What it shows about G2 (liberal in, canonical out).* These are **raw-array primitives**: they take cores, not
  masks, so they cannot neutralize junk. The uniform squash sums the boundary bond over every slot, padding included,
  so on the uniform layer its precondition is zero padding. Updating the rank masks (the boundary rank becomes 1;
  a reversed chain's masks are reversed) is likewise the caller's business. The object-level function does both:
  `ut3_operations.ut3_squash_tails` masks on entry ("any nonzero padding … would be summed into the real slot and change
  the represented tensor", the fix of `439069b4`), calls the primitive, and updates the boundary masks.

Recommendation: keep both (parametric); annotate with `PerSite`; state the raw-primitive precondition in their
docstrings.

**A principle emerging from C2: G2 applies to functions that hold the masks.** "Liberal in" is a promise only an
operation that receives an object with its masks can keep. A raw-array primitive (cores without masks) states "zero
padding" as its precondition; the object-level function above it masks on entry and keeps the masks consistent with
the result.

**Ruling (Nick, 2026-11-24): agreed.**

**C3: `tt_orthogonalization.tt_left_orthogonalize`, `tt_right_orthogonalize`** (the second is the first on the reversed
chain). The left-to-right SVD sweep of a TT. 33 library call sites, ragged and uniform. Two separate issues.

*Issue 1: `return_variation_cores: bool` switches the return type.* `False` returns the orthogonal cores; `True` returns
`(cores, variation cores)`. By principle 4 that is a switch (it changes the output type), and G3 asks for one return
type. The variation cores (the running non-orthogonal core at each site, the TT variations of the orthogonal
representation) are a by-product the sweep computes anyway: the scan collects them as its second output (`HH0`) on
every call. So the flag chooses only how much of the result to hand back. 11 library calls pass `True`. The frontend
mirrors the switch: `TuckerTensorTrain.left_orthogonalize_tt_cores(return_variation_cores=False)` and
`right_orthogonalize_tt_cores` are annotated `-> 'TuckerTensorTrain'`, but return `(TuckerTensorTrain, variation
cores)` when the flag is set, so the annotation is wrong in that case.

Options: (i) always return both (one operation, one return type: everything it computes; a caller that wants only the
cores writes `L, _ = …`; under jit the unused output is eliminated, and eagerly it costs one concatenation beside
`d-1` SVDs); (ii) two names, one per return. Claude recommends (i) for the backend. For the frontend, (i) makes
`x.left_orthogonalize_tt_cores()` return a pair. Whether a user-facing method should instead return the T3 alone, with
a second method for the pair, is Nick's call.

*Issue 2: `pad_masks` is a uniform-only input.* With it, each step's SVD goes through `pad_safe_svd`: the σ≈0
completion columns stay off the padded slots, the guarantee a frame needs (S1b). Without it, the plain SVD runs. Unlike
example 3a, **the plain uniform path is not dead.** `ut3svd_supercores` takes it when called on bare supercores without
the masks (`content=None`, `ut3_svd.py:373-376`). And it is the right path for an exact factorization inside a read-out,
where the padding provably contracts to zero (the June "exactness argument"; `t3_linalg`, run on masked uniform cores
by `ut3_linalg`). By principle 3 (an extra input on one branch) and C2's refinement (only a function holding the
masks can guarantee anything about the padding), the pad-safe sweep is a different function. The step functions are
already split (`_tt_left_step`, `_tt_left_step_pad_safe`).

Recommendation: `tt_left_orthogonalize` / `tt_right_orthogonalize` stay parametric (plain SVD steps, ragged cores or a
raw supercore; zero padding as the uniform precondition, per C2). The pad-safe sweep becomes its own uniform function,
in `ut3_orthogonalization` beside `_tt_left_sweep_pad_masks`, which builds its masks. The small scan skeleton is shared
through a private helper. (Whether the pad-safe function takes today's precomputed per-step masks, or the rank masks
and builds the per-step ones itself, is a "how" for later.)

**Rulings (Nick, 2026-11-24).** *Issue 1:* (i), always return both, **on the frontend and the backend.* *Issue 2:*
agreed, **with a caution: making this sweep work with the padding was very tricky.** For whoever builds it:

- **The split is a move, not a rewrite.** The pad-safe path's numbers should come out **bitwise identical** before and
  after (the computation does not change, only which function holds it). The sweep is deterministic: "same input →
  bitwise same frame", pinned by a test. So a bitwise before/after battery, as the store-rename arc used, is the gate.
- **What pins the behaviour today:**
  - `tests/test_uniform_frame_variations_format.py::TestPadSafeFrame` (review S1b: before the fix, a zero-padded `resize`
    warm start lost tangent directions in 60/60 trials; the uniform frame is gauge-equivalent to ragged, not
    bit-identical);
  - `tests/backend/test_linalg.py::TestPadSafeSvd` and `::TestPadSafePinnedCompletion`;
  - pad-safe cases in `tests/test_dispatch.py` (jit) and `tests/backend/test_uniform_fitting.py`.
- **The reasoning lives in** `docs/pad_safe_svd.tex` (the problem, the sketch-project algorithm, the alternatives
  measured and rejected), `docs/uniform_equivalence_contract.md` ("Gauge-carrying operations") and
  `docs/contributor/uniform_svd_prefix_orthogonalization.md`. The review record is in
  `dev/archive/review_2026-08-22/`.

**C4: `t3_operations.t3_absorb_tucker_into_tt(tucker_cores, tt_cores)`.** It absorbs each Tucker factor into its TT core,
replacing the `n` leg with the physical `N` leg: `big[a,o,b] = Σ_n tt[a,n,b] tucker[n,o]`. It is the opening step of
`t3_to_dense` and of the inner-product/norm zipper. Callers: `t3_conversions.t3_to_dense` (ragged) and `ut3_linalg`
(uniform, on masked cores).

- *Against the principles.* The same operation on both layers (one batched einsum over a supercore, or one einsum per
  core of a tuple); the output is complete in the input's representation as raw cores; no switch. Parametric, as C2.
  Annotated with `PerSite`, shared by both inputs and the return. A raw-array primitive, so on the uniform layer its
  precondition is zero padding (the sum runs over the padded `n` slots); its uniform caller masks first (C2's
  principle).
- *Mixed forms: not a clean error, and once silently wrong.* It decides the layer by `tucker_cores` alone. A uniform
  Tucker with a ragged TT fails inside numpy ("setting an array element with a sequence … inhomogeneous"), and a ragged
  Tucker with a uniform TT also fails inside numpy ("operands could not be broadcast together"). But when no padding
  is needed (shape `(4, 4, 4)`, ranks `(2, 2, 2)`, `(1, 2, 2, 1)`), **the ragged-Tucker, uniform-TT call is accepted
  and returns a tuple of cores `[(2, 4, 2), (2, 4, 2), (2, 4, 2)]`**, with the uniform padded boundary rank 2 where a
  ragged T3 has 1: a malformed ragged object, silently.

Recommendation: keep (parametric); annotate with `PerSite`; make mixed forms a structural error (both arguments in one
representation, per principle 2); state the zero-padding precondition.

**Ruling (Nick, 2026-11-24): agreed.**

**C5: `ManifoldGeometryOps.aligned_padded_frame` and `UniformManifoldGeometryOps.aligned_padded_frame`** (the
geometry protocol's member behind rank continuation's aligned padded frame).

- *Not a layer question.* The two layers already have their own methods, on their own classes. The `Union` comes from
  **`timing: bool`, which changes the arity of the return**: `(frame, centers, movements)`, or with `timing=True` a
  fourth element, the seconds per pass. That is a switch (principle 4), the same shape as C3's
  `return_variation_cores`. Unlike there, timing is not free: it adds a synchronized start and end to each pass, so
  "always compute it" is not an option.
- *The fixed form already exists one level up.* `optimizers._aligned_padded_frame`, the member's only caller,
  normalizes it to a fixed four-element return with `seconds: Optional[Tuple[float, ...]]` ("None unless timing and the
  member reports them"). The uniform member already returns `None` there under `use_jit`.
- *Secondary: the same quantity in two forms.* The movements are a tuple of per-pass arrays on ragged, and a NaN-padded
  `(max_passes,)+C` array on uniform (the static shape jit needs). Through the protocol the return type therefore
  depends on the layer ("as the member returns them", `padded_start`). The NaN-padded array fits both layers; the
  tuple fits only ragged.

Recommendation: the member always returns the four elements, `seconds` `None` unless `timing=True` (one return type; the
flag then sets a value, not the arity); `padded_start` keeps returning the first three. Secondary, lighter: the ragged
movements adopt the NaN-padded array, so the protocol member has one return type on both layers.

**Ruling on the first recommendation (Nick, 2026-11-24): agreed** (`seconds=None` when `timing=False`).

**The movements, explained (Nick asked; `docs/aligned_padded_frame.md`, `backend/padded_frames.py`).** In each pass of
the fixed-point iteration, the hint is projected onto the tangent space of the current frame. Then, at every
directed edge of every store, the new slots are filled with the leading directions of that edge's projected-hint
content, orthogonalized against the pinned old block (`_edge`: `P⁺ = pinned_leading_columns(Abar, Y, k1)`). The
previous pass's completion `g_old`, in the new coordinates, is compared with it: **the edge's movement is
`‖g_old − P⁺P⁺ᵀ g_old‖_F`**, the part of the old completion outside the new span. For orthonormal columns that is the
root-sum-square of the sines of the principal angles; it is 0 exactly when the span did not turn. **The pass's movement
is the maximum over all directed edges** (`padded_frames.py:704`). It drives the stop (both movements of a two-colour
pair below `tol`) and is reported as a diagnostic. One pass gives one movement per base point: shape `C` (0-d when
unstacked). The record over the passes is a variable-length history of that scalar: on ragged, a tuple of the passes
run; on uniform, `(max_passes,)+C`, NaN after the last pass (jit's static shape), which "cannot tell a NaN movement from
a pass that never ran".

Options for the secondary point:

- (a) leave each layer its natural form (the movements are a diagnostic);
- (b) the NaN-padded array on both (one type; ragged inherits the NaN/never-ran ambiguity);
- (c) (b) plus the number of passes run, a count shared across the stack, since the stop takes the maximum over it.
  One type, no ambiguity on either layer.

Claude's lean: (c), or (a) if the movements count as purely diagnostic.

**Ruling on the secondary point (Nick, 2026-11-24): (a). "It's a diagnostic."** Each layer keeps its natural form.

**A principle emerging from C5: a diagnostic may take its layer's natural form.** One return type is for the
operation's mathematical results. A diagnostic history riding along (here a variable-length record of a scalar) may be
held however its layer holds such things best.

**C6: `jets.affine_jet(ww, *directions)`.** It builds input jets, `w + s_1 p_1 + … + s_m p_m` per mode (slot `0` is `w`,
slot `e_j` is `p_j`, the rest zero). Used across the library (the frontend optimizers' translation; the
`entries_derivatives` recipe `affine_jet(index_to_one_hot(index, shape), pp)`), 89 test lines and 17 in the docs and
examples.

- *Against the principles.* The same operation on ragged vectors (a tuple of `d` arrays, giving a tuple of input jets)
  or packed ones (`(d,)+W+(N,)`, giving `(d,)+(2,)*m+W+(N,)`). The return follows the input, and the output is complete
  in its form (principles 2 and 3); no switch. It is the exemplar of §10.1's item 5 ("functions without an object
  keep their input's form"): it has no T3, so no layer of its own. It is elementwise, so with zero-padded input the
  output's padding is zero too.
- *Mixed forms* already raise a clear `ValueError` both ways ("direction 0 must be packed like ww, an array of shape …";
  "direction 0 must be a sequence of 3 arrays, as ww is").
- *Typing:* the same pair of representations as C1, so `PerSite`: `affine_jet(ww: PerSite, *directions: PerSite) ->
  PerSite`.

Recommendation: keep; annotate with `PerSite`. Nothing else to change.

**Ruling (Nick, 2026-11-24): agreed.**

**C7: `jets.data_metric(r, weight, minibatch_scale)` and `LeastSquaresBatch.data_metric(r)`** (the data metric
`M: Y* -> Y`, `γ ω² r` elementwise; the batch method supplies its own weight and scale). Called by the engine's steps,
the transposes' docs and examples, the oracle and the frontend (about 34 lines in 9 modules).

- *Against the principles.* One operation, elementwise, the same for every kind. So being kind-free (the M arc's
  choice, "no `kind` or `order`") is what G3 asks: the types differ by kind, but the operation does not. That differs
  from packing, where only probe has work to do. The return follows `r` (a tuple for a tuple, an array for an array):
  parametric (principle 2), annotated with a `TypeVar` over the same pair as `PerSite` (a tuple of arrays, or an array;
  for apply and entries the array has no site axis, so the shared `TypeVar` wants a neutral name, O3). The weight
  accepts the ragged layer's two forms for a tuple `r` (a tuple, or one mode-leading array read along its first axis:
  §10.1 item 6). Those are input forms, not a return union. A packed `r` with a tuple weight already raises, naming
  `pack_batch` (M1 decision 2).
- *Padding: transparent.* Elementwise, so a real entry never meets padding, and the output's padding is `γ ω²` times
  `r`'s padding. When `r` is canonical (zero padding, as every sampling output is under G2), so is `M r`. No
  precondition is needed.
- *The one real question: the M arc's documented limit* (`DATA_METRIC_PLAN.md` §3). On a packed probe `r`, a weight whose
  mode axis has size 1 broadcasts over the modes, unchecked (checked: a `(1, 1, 1)` weight on a `(3, 10, 6)` `r` gives
  the same value on every mode). The packed form's definition says the mode axis is exactly `d`, never broadcast
  (§10.1 item 6, 6.7's point). `data_metric` cannot check it: it does not know the kind, and for apply a size-1 leading
  axis (a jet axis) is legitimate. Options: (i) accept it; the structural rule is the checker's (`check_probe_data`),
  opt-in for packed data under the plan, so it belongs to O4's trade-off; (ii) split `data_metric` by type, so the
  probe version can check its mode axis: three copies of one elementwise operation, split by a structural check only;
  (iii) a `kind` argument, which G3 excludes. The broadcast case is also benign in meaning ("the same weight for every
  mode"), so the risk is a looser check, not a wrong number.

Recommendation: keep one function (and the method), kind-free; annotate with the `TypeVar`; (i) for the limit.

**Ruling (Nick, 2026-11-24): agreed, (i).** Nick asked for a doctest that shows the function used incorrectly and
giving an incorrect result. **Added** to `data_metric`'s docstring (`backend/jets.py`, uncommitted): a weight per
entry built for mode 0 alone (`wrong = right[:1]`, a mode axis of 1), against a packed probe `r` of `d = 2`
(`Ni = 3, 4`, `N = 4`). Mode 1 comes back weighted by mode 0's entries, `[[1.0, 4.0, 9.0, 0.0]]` (its last real entry
even dropped), where the right weight gives `[[1.0, 1.0, 1.0, 1.0]]`, with no error. The checker then catches it on the
ragged form (`check_weight` raises "… exactly d = 2 arrays, one per mode … a mode axis of 1 is not broadcast"). It
passes on both numpy generations (py3.11/numpy 2, and the py3.9/numpy 1.22 floor env). **Made outside the review
process:** Nick allowed it once, since it is small and he will verify it. Every other change goes through the process
(proposal, critic, implementation, review, gates); drafts and prototypes go in `dev/`. When the checkers are renamed
(§10.2), this doctest's `check_weight` call changes with them.

**Group C, summed up.** Every member stays one function, parametric over its representations, annotated with a shared
constrained `TypeVar` (`PerSite`; the name is open, and C7 wants a neutral one, O3). The changes found on the way:

| Member | Change |
|---|---|
| C1 `xcat` / `xappend` / `xprepend` | the empty-sequence branches return tuples |
| C2 `tt_reverse`, `tt_squash_tails` | state the zero-padding precondition (raw primitives) |
| C3 `tt_left/right_orthogonalize` | always return both (frontend and backend); the pad-safe sweep becomes its own uniform function, a bitwise-checked move |
| C4 `t3_absorb_tucker_into_tt` | mixed forms a structural error (one was silently malformed); state the precondition |
| C5 `aligned_padded_frame` | always four elements, `seconds=None` unless timing; the movements stay layer-natural (a diagnostic) |
| C6 `affine_jet` | annotation only |
| C7 `data_metric` | annotation only; the mode-axis broadcast limit stays with the checkers (O4) |

Principles that emerged here, beside §10.7's: **G2 applies to functions that hold the masks**, and a raw-array
primitive states zero padding as its precondition (C2); **a diagnostic may take its layer's natural form** (C5).

### 10.9 The open issues of §10.4, one at a time (2026-11-24)

**O1. The masking option** (F4: Nick, "an option to the engine which allows the user to turn off the automatic
masking").

*The cost, by count rather than by laptop benchmark.* Per uniform call the engine masks every input once: the stores
or the point's cores (about `d·C·(n·N + r²·n)` entries for a corewise point; a frame's four stores alike), the sample,
`y`, the weight and the held variables. The forward sweep's site contractions and bond sweeps carry the sample stack:
about `d·W·C·(n·N + r²·n)`, times the jet slots. **So masking the stores costs about `1/W` of the sweep, at every
call.** That is negligible for hundreds of samples, comparable to the sweep for a handful, and paid again by every call
of a held chain: the regime Nick described. Under jit, XLA may fuse the multiply into its consumer, but nothing
guarantees it. A regime-dependent trade-off, so a documented option; a measurement would set defaults, not decide
whether the option exists.

*The sub-questions, with Claude's recommendations:*

1. **What it means.** One engine keyword: the engine is the only function that holds the masks (C2's principle). The
   default masks. Off, the caller guarantees canonical inputs (zero padding), a precondition. Every library output is
   canonical (G2), so a chain of library calls meets it by construction. External inputs are made canonical once:
   `x.apply_masks()` for cores, the packers' zero fill for data.
2. **Its safe-mode check costs about what the masking costs.** It reads every padded entry, so in safe mode the option
   saves nothing. It pays off in unsafe mode and under jit, where checks are skipped. Recommendation: accept that. It
   is the house rule (safe mode verifies, unsafe and jit run fast), and the check is what makes turning masking off
   safe to try.
3. **Output masking stays always on.** It is G2's guarantee. It costs at most the output's size, never more than the
   work that made it (measured 0.08–1.1 % eager, 0 under jit). It is also what makes item 1's chains safe. An option to
   skip it would undo both.
4. **Which side of a padded contraction to neutralize: Part 2.** G2 needs only one factor of each padded contraction
   zeroed. On the mode edge that is the sample (`d·W·N`) or the Tucker cores' columns (`d·C·n·N`): the cheaper side
   depends on `W` against `C·n`. On a rank edge it is the TT core, or every bond variable the sweep makes (`W`-sized at
   each step). A choice per contraction, by regime: a plan, Part 2's kind of decision.
5. **Timing: Part 2, by the arc's own line.** "Part 2: how it is carried out. The plan and every performance option.
   It changes no number" (`TARGET_DECISIONS.md`). With its precondition met, the option changes no number. Part 1
   delivers what makes it safe later: canonical outputs everywhere (G2) and the precondition, defined. The name is
   Part 2's.

**Rulings (Nick, 2026-11-24):**

1. **Agreed: one engine keyword; the default masks.**
2. **Agreed:** the option saves significant cost only when safe mode is off.
3. **Agreed: output masking stays on.**
4. **One consistent choice, not an optimization: zero the cores.** "The user who wants to maximize performance will turn
   off the masking with the engine keyword and do the zeroing themselves as they wish." Consequence (Claude's reading):
   every padded *contraction* has a core on one side, so zeroing the cores neutralizes the sample's padding (the mode
   edge) and every bond variable's, held ones included (the rank edges). The engine then need not mask the sample, as
   it does today. The data-space objects that no core touches (`y` and the weight in `r = z − y`, then `f` and the block
   sums; a held `r` or `Jp`) are combined elementwise and summed over the edge. No core can neutralize them, so they
   are still masked themselves, as today. That is their only option, not an optimization.
5. **Not a question for now.** "We are still deciding what to do, not when to do it."

**O2. The ragged layer's one-array form for the sample** (Nick's candidate exception, undecided: a packed-shape array
accepted on the ragged layer, read as `d` vectors of the same length).

- *Where it stands.* For data-space objects it is already the rule (M2, `DATA_METRIC_PLAN.md` §2: "a tuple or list, or
  one array whose first axis is the mode"). For the sample it is the engine's behaviour (an array is read as `d` rows,
  each of which must have its mode's shape). §10.1 item 6 defines the ragged form this way for every per-mode object.
- *Who relies on it.* For the sample, nobody measured (§9.3: no test, doctest, page or example passes an array sample
  to a ragged function). For data-space objects it is load-bearing: `blocks_to_weight` returns one mode-leading array
  on either layer, and the per-mode ω matrix reaches the ragged engine that way (D1: 180 tests, two pages).
- *Is it automatic packing?* No. Nothing is padded or sliced. A numpy array is already a sequence of its rows; reading
  it along its first axis is Python's own meaning. Refusing it on the ragged layer would take an explicit type check
  against duck typing.
- *Its risk.* An array whose leading axis is not the modes, but has length `d` by coincidence (a `(W, d, n)` array with
  `W = d`), is misread silently. That is the general "axes in another order, sizes coinciding" limit the M arc
  documents for every shape check. A tuple cannot make this mistake. Unequal widths are caught: each row must have its
  own mode's shape.
- *Under G3.* The ragged functions' inputs stay `Sequence[NDArray]` (an array is a sequence of its rows), and their
  returns are always tuples. One return type holds.

Options: (a) keep it for both, one definition of the ragged form (§10.1 item 6); (b) keep it for data-space objects
only, and require a sequence for the sample; (c) remove it for both (data-space objects too: `blocks_to_weight` would
return a tuple on the ragged layer, and D1's 180 tests change). Claude's recommendation: (a). One definition of a
form, Python's own sequence meaning, no padding involved; the coincidence risk is the documented general limit.

**Nick's leaning (2026-11-24): (c), as a general rule for the ragged layer.** "This is a real risk." Every per-mode
input on the ragged layer is a sequence of arrays; a single array of stacked slices is refused, and a user who has one
unstacks it first. He asked for the blast radius.

**The blast radius, measured** (`packing_audit/ragged_sequences_plugin.py`, recording without raising every array
given where the rule requires a sequence: the engine's sample, `y`, weight and held data-space variables; every
per-mode argument of the ragged and dense public functions; `data_metric` with a tuple `r`; the `jets` and
`data_preparation` helpers holding a ragged object beside an array; `blocks_to_weight` returning an array for a ragged
`y`; and, apart, the entries index). Run over the full suite (1,979 passed), the module doctests (325), the doc pages
(10) and the five examples that pass weights (all pass):

- **Samples: nothing breaks.** No array sample, and no array per-mode argument of any ragged or dense public
  function, anywhere.
- **Everything else is one object: the probe weight.** 275 test units are affected (265 tests, 4 module doctests, 2
  pages, 4 examples). **262 of them reach it through `blocks_to_weight`**, which returns one mode-leading array
  (edge 1) for a ragged `y`. Its callers: the shared fit harness `tests/_fits.py` (144 tests),
  `test_sampling_definitions.py` (57), the old-form translation `optimizers._old_fit` (`:196`, `:216-217`; deleted in
  P17 anyway), and 15 more test files directly. From there the array flows into the engine (180 tests), `check_weight`
  (58), `prepare_derivative_data` (65), `data_metric` (9) and the dual-jet converters (10).
- **The other 13 build a mode-leading array by hand**, and most of them test that very form or a refusal: for example
  `test_probe_ragged_with_a_tuple_and_with_an_array_weight`, `test_check_data_passes_size_1_axes_and_a_mode_leading_array`,
  `test_the_mode_count_of_a_ragged_r_is_exact`, and the `raises` tests of the checkers and converters. They would be
  rewritten to test the rule.

**So the radius is small at its source.** If `blocks_to_weight` returns the weight **in `y`'s form** (a tuple for a
ragged `y`, a packed array for a packed `y`: representation-following, as §10.7's principle 2 says it should), the 262
pass through unchanged, except the few that assert its probe output's array shape (its own tests and doctests, and
`data_metric`'s doctest that prints `(2, 3, 1, 1)`). Under G3 it is split by type anyway (§10.6, group A).

**The code side.** Six places read a ragged probe object along its first axis, and change to tuples only:
`jets._arrays` (an array then means packed, kept packed: F7), `jets._mode_weights`, `data_metric`'s ragged branch
(`jets.py:745`), the engine's data-space check (`sampling_variables.py:1199`), `sampling_steps._modes` / `r_step`, and
the reference twins in `sampling_definitions`. The backend docstrings promising the one-array form: `jets` 13,
`sampling_variables` 4, `sampling_definitions` 3, `fitting` 2, `sampling_steps` 1. No user-facing page states it.
`DATA_METRIC_PLAN.md` §2's rule changes for the ragged layer ("a tuple or list" only).

**The entries index: is it covered?** `index^d` is one integer array `(d,)+W`, the same on both layers: a stacked
array of per-mode slices, but with no widths, hence no ragged or packed form. The same axis-order coincidence can hit
it (a `(W, d)` array with `W = d`). Covering it would change 135 tests, 9 module doctests and 1 page, and every user's
entries call. Claude's lean: exempt it. It is one array by definition on both layers (the type `index^d`), with no
ragged/packed confusion to prevent; the coincidence risk is shared with every array argument. Nick's call.

**Ruling on O2 (Nick, 2026-11-24): (c), a general rule for the ragged layer, the entries index included.** "Entries is
not exempt because of the same axis-order coincidence ambiguity." On the ragged layer every per-mode input is a
sequence of `d` arrays: the sample, the data-space objects (`y`, the weight, residuals, held data-space variables)
and the entries index (`d` integer arrays, each of shape `W`). A single array of stacked slices is refused; the user
unstacks it first (`tuple(index)`, `list(weights)`). §10.1 item 6's ragged form loses "or one array read along its
first axis".

Consequences recorded for the plan:

- `blocks_to_weight` (probe) returns the weight in `y`'s form: a tuple for a ragged `y`, a packed array for a packed
  `y`. That removes 262 of the 275 affected units at their source (above).
- The six places that read a ragged probe object along its first axis take tuples only; on the uniform layer an array
  means packed, and the object-free helpers keep it packed (F7).
- **The index differs between the layers now:** a tuple of `d` integer arrays on ragged, one `(d,)+W` array on uniform
  (packed, as today; an index has no edge, so "packing" it is `np.stack`, and unpacking is `tuple(...)`, trivial and
  obvious, so no helper). This matches numpy's own multi-index convention: `A[tuple(idx)]`, `np.nonzero`,
  `np.ravel_multi_index` all use a tuple of per-axis integer arrays.
- The index's radius: 135 tests, 9 module doctests, 1 page (measured), plus `examples/fit_hilbert_from_entries_lbfgs.py`
  (which slices a `(d, M)` index, `index_all[:, :N_TRAIN]`) and the code that reads the index as one array:
  `jets.check_sample`'s entries branch, `jets.index_to_one_hot`, the engine's entries handling, the entries
  functions, the frontend `entries` methods and the batch `take` for the entries kind.
- The data-metric plan's §2 rule changes on the ragged layer to "a tuple or list" only.

**O3. Homes and names.**

1. **Home.** Today the packers are scattered: `pack_vectors`, `unpack_vectors`, `is_packed`, `pack_if_ragged` and
   `_pack_data_space` in `ut3_operations`; `pack_sample`, `pack_data` and `pack_batch` in `uniform_fitting`. The naming
   conventions file the latter group as "fitting infrastructure, not family data ops" (`docs/naming_conventions.md:220`).
   Under the plan, packing is the user's own step at the layer boundary, for any uniform sampling call. The module map's
   conversions column already holds the T3's layer conversions (`ut3_conversions`: `t3_to_ut3`, `ut3_to_t3`), and
   packing is their analog for per-mode objects. Recommendation: the packers in `ut3_conversions` (no new module; the
   map unchanged); `pack_batch` stays in `uniform_fitting`, since it packs a fitting container.
2. **Names, after the G3 split.** Sample: `pack_vectors` / `unpack_vectors` (probe and apply share the type). Data:
   `pack_probe_data` / `unpack_probe_data`; apply and entries are identities. O2 adds the index: a tuple on ragged, an
   array on uniform, so `pack_index` (`np.stack`) / `unpack_index` (`tuple`). Recommendation: the full per-kind set,
   identities included (one sample packer and one data packer per kind), so kind-generic code (`pack_batch`, a user's
   own loop) needs no special case.
3. **`pack_data`'s `order`** is unused: it goes.
4. **The frontend.** The packers need nothing from an instance: module-level functions, not methods. The unpackers need
   the shape: instance methods on `UniformTuckerTensorTrain` and `UT3Tangent` (`x.unpack_probe_data(z)`), plus
   `fitting.pack_batch(oracle, batch, N)`.
5. **The `TypeVar`.** `PerSite` is a placeholder, and C7's `data_metric` wants a name without sites. Recommendation: two
   names over the same constraints, each saying what it holds: `PerSite` (site- and mode-indexed objects: cores, sample
   vectors) and `DataSpace` (`data_metric`), in `common.py` beside `NDArray`.

**Ruling on O3 (Nick, 2026-11-24): agreed on all five.** And an addition:

**O3 addendum: the checkers** (Nick: "include the checkers, is a set of vectors being packed consistent with a target
uniform object, as backend functions and frontend methods of the object being checked against").

- **They take the ragged form: the vectors about to be packed.** It is the only form that still carries the widths
  (§9.7 point 1), so the check G1 asks for (each width equals the target's `N_i`) can only happen there. It also gives
  each checker one input type. Packed data are checked structurally (`d`, `N`, the broadcast rule) by the sampling
  functions themselves, always on.
- **Split along the types (G3), like the packers.** The target is any object's `.data`: a point or a frame, ragged or
  uniform (`sampling_variables.frame_shape` already reads the shape off all four forms). It supplies the shape, `N`
  and the stack `C`. A data-space object is checked against `z`'s type, read from the sample and the target.

  ```python
  def check_vectors(                                 # the sample of probe and apply
          vectors:  typ.Sequence[NDArray],           # P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
          order:    typ.Tuple[int, ...],             # the jet shape (len(order) leading jet axes, F_in_i <= F)
          target:   typ.Tuple,                       # the object: a point or a frame, ragged or uniform
  ) -> None:                                         # ValueError: d, each width against the target's Ni, the jets, W

  def check_index(                                   # the sample of entries
          index:   typ.Sequence[NDArray],            # index^d, len=d, elm_shape=W, int
          target:  typ.Tuple,
  ) -> None:                                         # ValueError: d, one W for all; IndexError: out of [-Ni, Ni) (numpy)

  def check_probe_data(                              # y, the weight, a residual: of the type of z
          value:   typ.Sequence[NDArray],            # len=d, elm_shape=F+W+C+(e_i,), each axis 1 or z's size, e_i in {1, Ni}
          sample:  typ.Sequence[NDArray],            # P_in_i⊗V_i: W is read off it
          order:   typ.Tuple[int, ...],
          target:  typ.Tuple,                        # C and the Ni
  ) -> None:

  # check_apply_data(value, sample, order, target) and check_entries_data(value, index, target): one array each,
  # the same on both layers (F+W+C; W+C), checked by the broadcast rule.
  ```

- **Frontend: methods of the object checked against**, on the four classes (for the uniform ones, the check before
  packing): `x.check_vectors(vectors, order=())`, `x.check_index(index)`, `x.check_probe_data(value, sample,
  order=())`, `x.check_apply_data(...)`, `x.check_entries_data(...)`. A tangent checks against its frame.
- **They replace** `jets.check_sample`, `check_data` and `check_weight` (which take `shape` and `stack_shape`), and
  `LeastSquaresOracle.check_batch(batch, shape, stack_shape)` becomes `check_batch(batch, target)`, composed of them.
- **Home (Nick, 2026-11-24): `ut3_conversions`**, beside the packers: check, then pack, in one place.
- The workflow they serve, to document on the packers and the uniform pages (O4): **check the ragged data against
  the target, then pack, then sample.**

**O4. The width check is opt-in on the uniform layer.** By construction: to check widths (G1), something must see the
ragged data and the target together, which is either a packer that checks (excluded: packing is a change of
representation) or an optimizer that takes ragged batches (excluded: the layer rule). With the checkers on the target
it is one call before packing (`ux.check_vectors(ww, order)`, `ux.check_probe_data(y, ww, order)`). What stays
automatic: the structural checks of packed data (`d`, `N`, the broadcast rule, `C`) in the sampling functions and the
fitting frontend. What can slip through without the check: data packed at the wrong widths (narrower than `N_i`: a real
entry read as a zero pad; wider: entries spilled into the padding and masked away).

**Ruling on O4 (Nick, 2026-11-24): agreed.** Opt-in, with the workflow (check the ragged data against the target, then
pack, then sample) documented on the packers' docstrings, on the uniform pages, and in the two uniform examples (which
gain their explicit packing line anyway, and can show the check beside it).

**O5. Mixed edges and canonical inputs.** The data packer's three cases (probe weights, `N = 6`, real widths 4 and 5):

```
every edge full -> (2, 1, 6)  mode 1: [1, 2, 3, 4, 5, 0]        zero-padded: canonical
every edge 1    -> (2, 1, 1)  mode 1: [3]                        no padding at all: canonical
mixed edges     -> (2, 1, 6)  mode 1: [3, 3, 3, 3, 3, 3]         the constant on mode 1's padding slot too
```

To make one array of mixed edges, the edge-1 modes must be widened. With zero padding they would need their own
`N_i`, which the packer does not have (it takes no shape, by the O3/§9.7 ruling). So it widens them to `N`, putting
their constant on the padding. Under G2 that is allowed junk, and with the default masking it is harmless: the engine
masks the data-space objects on entry (O1, item 4). It matters only with masking off (O1), whose precondition is zero
padding: the safe-mode check rejects it, and in unsafe mode or under jit the padding enters the sums (`f`, the block
sums) and the result is wrong. Mixed edges are the rare case: a weight per entry on some modes and a constant on
others. The common forms are canonical: `blocks_to_weight`'s weights are all edge 1, and a weight per entry is all
full.

Options:

- (a) Document it: the packer's output is canonical except for mixed edges; with masking off, widen the edge-1 modes
  to their own `N_i` before packing.
- (b) **The packer refuses mixed edges**, naming the fix (broadcast the edge-1 modes to their `N_i` first). Like
  `pack_vectors` refusing modes whose `W` differs, this is a requirement on the input's own form, not a check
  against a target. It needs no shape.
- (c) A shape argument for this case only (excluded: the packers take no shape).

Claude's recommendation: (b). It gives the packers a one-line contract that matches G2's "canonical out": **every
packer returns a canonical object.** Then everything the library produces (the packers' output, every sampling
output) meets the unmasked mode's precondition, and only a packed array built by hand needs the user's care.

**Ruling on O5 (Nick, 2026-11-24): (b), the packer refuses mixed edges**, naming the fix. With it, every packer returns
a canonical object (added to G2 in §2).

**O6. Public functions that return junk padding.** Known: the engine-based twins return canonical output (P12b's
tests put garbage in every padded slot and assert exact zero padding on the outputs); every uniform transpose is
re-masked (2026.3.0). The one deliberate exception is the bare `utv_*_transpose_from_sweep` seam halves (the raw sweep's
padding, about 3e12 before that fix; their callers re-mask), on the old path that P20 deletes. The question is a "what":
Claude's recommendation, **no public sampling function is exempt from canonical output**; one that cannot meet it
masks or is not public.

**Nick's concern (2026-11-24): erosion.** "Over time, new code may start taking for granted that the padded regions
are zero, because in all of their examples the arrays are fed in from library functions which make this guarantee.
Then, slowly and without realizing it, the code no longer supports junk-in." Today's defence is a convention:
`docs/contributor/testing_strategy.md` §2 ("a correct op must give the identical result on a garbage-padded input as
on a clean one"), with garbage helpers in twelve test files (`corrupt`, `packed_with_garbage`). Nothing enumerates the
public functions to make sure each has such a test (`test_dispatch` lists its cases by hand).

**The guards (recorded at Nick's request, 2026-11-24):**

1. **Junk-in lives in few places.** After P20 every sampling function goes through the engine, which masks on entry
   by default, so a new sampling function inherits junk-in. Erosion can happen only where masking is decided: the
   engine's entry and the few object-level uniform functions that hold masks.
2. **Relying on zero padding is legal only when stated.** Code may take zero padding for granted only under an explicit
   precondition: a raw-array primitive whose docstring says so (§10.8, C2), or the engine's `mask_inputs=False` (O1).
   A function that holds masks and states no such precondition must handle junk.
3. **Hostile tests by default.** The test fixtures that build uniform objects and packed data put garbage in the
   padding by default, with an explicit opt-out for the tests of canonical output itself. The suite then exercises
   junk-in unless a test says otherwise. Test helpers only; nothing in the library changes.
4. **A coverage meta-test.** One test enumerates every public uniform function through `__all__` and requires a
   registered garbage case for each (the same call with garbage-padded and with clean inputs; the outputs bitwise
   equal). A new public function without one fails the suite, unless it is exempted with a reason (a raw primitive
   with its precondition stated). New code meets junk-in at birth.

Rejected: a test mode in which the library's own outputs carry junk. It would contradict the canonical-out guarantee
that callers may rely on (the unmasked chains of O1).

**The prototype of guard 4** (`packing_audit/junk_in_plugin.py`, `junk_in_analyze.py`). For each public uniform
sampling function a test calls (`ut3_sampling`, `utv_sampling`, the engine), it calls it twice more: with every input it
can clean made canonical, and with finite garbage (1e3) in that padding, and compares the outputs bitwise. Run over the
sampling test set (560 tests passed, at most 100 checks per function per worker):

- **56 functions checked, 3,089 checks, 3,084 bitwise equal.** Every engine-based twin, every transpose and Hessian
  (the `*_from_sweep` ones included), the jets, `ut3_full_sum`, the engine itself: junk-in holds.
- **5 differences, all in the output's padding, one each:** the five `utv_precompute_*_frame_sweep(_jets)`. Checked
  directly: the sweep record **stores the packed sample as given**, its junk padding included, while the sweep
  variables it computes are equal and the forward computed from either sweep is bitwise equal. So no number is
  wrong, but a public function returns non-canonical output (G2). They are on the old path (P20). It shows in one test
  only (`test_theta_hat_masking_unit`), the only one that passes these functions an already-packed sample; the others
  pass ragged samples, which are packed with zeros inside.
- **Not covered:** 6 functions called with nothing the prototype could corrupt (the `utv_*_transpose(_derivatives)_from_sweep`
  seam halves take a sweep and a residual only: the known O6 exception), and 3 never called by these tests
  (`utv_{probe,apply,entries}_derivatives`). A meta-test would require a registered case for each: exactly guard 4's
  point.
- **A false positive on the way, and its lesson.** The first run flagged `ut3_{apply,entries}_corewise_derivatives_hessian_from_sweep`
  with "real" differences (up to 1.7e5). The cause was the prototype itself: it treated any argument named `c` as
  packed per mode whenever its first axis had length `d` and its last `N`. An apply residual `(order+1,)+W+C` met that by
  coincidence (`order+1 = d`, last axis `= N`), so the prototype "padded" real data. Read by kind instead (residuals are
  per mode for probe only), both functions came out equal in every check. It is O2's axis-order coincidence, in the
  audit's own code: a check must read a form by kind, never by shape.

**Nick's second concern (2026-11-24): mask threading.** "It may force awkward code architecture, where masks are
required to be passed to functions to do the output zeroing, and this forces us to split functions that could be
otherwise polymorphic. This could be a huge chain where masks need to be passed down through a tower of function calls
just to be used once at the low levels. I'm not sure if this is actually a problem; perhaps masks should be required in
such situations."

Claude's analysis:

- **Sampling needs no threading: the sandwich.** The engine masks every input once on entry, runs steps that "see no
  mask" (polymorphic raw primitives whose precondition, zero padding, the entry mask establishes; §10.8 C2), and masks
  every output once on exit. Junk the middle creates in a padding (the 2026.3.0 adjoint sweep's 3e12 came from the
  squash's adjoint broadcasting a gradient into padded boundary slots) meets, in a multilinear computation, only
  the zeroed padded entries of the masked cores. So it never reaches a real entry, and one mask at the exit removes it.
  That was the 2026.3.0 fix: one exit mask, the real block bit-identical. The masks stay at the boundary; the middle
  stays one polymorphic code path. Sampling operations are multilinear in this sense (sweeps, transposes, Hessians).
- **Where masks must go down: factorizations whose result is kept.** An SVD or QR in an orthogonalization chooses
  directions; on the uniform layer it can put a completion direction in padded slots, where the mask then erases a
  real direction (S1b: a frame that looked orthonormal but had lost tangent directions, 60 of 60 trials). No mask
  after the fact repairs it. The padded structure is an input of that algorithm, so the masks must reach each SVD
  (the pad-safe sweep's per-step masks). That is the split of 3a and C3: the mathematics', not the architecture's.
  Inside a scalar read-out even a factorization needs none (the June exactness argument).
- **The rule proposed:** masks at the boundary (the function holding the object masks inputs on entry and outputs on
  exit: G2's "liberal in, canonical out" is that function's contract); no masks in the middle (raw polymorphic
  primitives take none: "zero padding in" is their precondition, and where it holds they keep zero padding); masks
  inside only for factorizations whose result is kept, and there required (the uniform-specific case, split from the
  ragged one).
- **When the worry would come true:** only if every public raw primitive had to be liberal-in and canonical-out on its
  own. Without masks it can do neither. So G2 is the contract of the mask-holders, as C2 began to say.

**The call stacks, concretely (Nick asked, 2026-11-24).**

*Stack A today, the uniform frame: masks reach every SVD.* `ut3_orthogonal_representations` (frontend,
`uniform_frame_variations_format.py:1174`) → `ufv_conversions.ut3_orthogonal_representations` (`:60`, holds the masks,
masks on entry `:101`) → `fv_conversions.t3_orthogonal_representations(…, uniform_masks=…)` (`:123`, **polymorphic,
uniform-only keyword**; builds the per-step masks, `:197-215`) → `down_orthogonalize_tucker_supercores(…, row_mask,
col_mask, out_mask)` (`ut3_orthogonalization.py:125`) → `linalg.pad_safe_svd` (`:148`); and
`tt_orthogonalization.tt_left_orthogonalize(…, pad_masks=…)` (`:58`, **polymorphic, uniform-only keyword**) →
`xscan(_tt_left_step_pad_safe, …)` (`:89-91`) → `linalg.pad_safe_svd` (`:52`); the right sweep the same, reversed; and
`up_orthogonalize_tt_supercores` (`:182`) → `pad_safe_svd` (`:204`). Five levels, through two polymorphic functions'
uniform-only keywords: the tower, here justified (a kept frame; S1b).

*Stack A after 3a and C3.* `ufv_conversions.ut3_orthogonal_representations` holds the masks, builds the per-step masks
(moved from `fv_conversions`), and calls the uniform-only `down_orthogonalize_tucker_supercores`, the new pad-safe left
and right sweeps (C3) and `up_orthogonalize_tt_supercores`, each to `pad_safe_svd`.
`fv_conversions.t3_orthogonal_representations` is ragged only, and `tt_left/right_orthogonalize` are plain and
polymorphic: **no polymorphic function carries a mask.** The masks still reach every SVD, as the mathematics requires.
The strict output rule changes nothing more here: the pad-safe sweep zeroes its completion columns (measured: 0.0 in
the output's padding).

*Stack B: where the strict rule would force masks in.* The plain polymorphic sweep on a uniform supercore is reached
only through entry points that take bare supercores: `ut3svd_supercores(…, content=None)` (`ut3_svd.py:373-376`,
"exported-but-uncalled" in the library; a test calls it) and a backend user calling `tt_left_orthogonalize(supercore)`.
No library-internal path uses it (every library path that orthogonalizes uniform cores passes the masks). **Measured: on
canonical input (zero padding) the plain sweep returns entries up to 1.0 in the output's padding** (the SVD's orthonormal
completion columns), where the pad-safe sweep returns 0.0. Under a strict "every public function returns zero padding"
rule it cannot comply without masks. So either (i) the plain sweep becomes ragged only, every uniform caller uses
the pad-safe function and passes masks, `ut3svd_supercores`' `content` becomes required, and a bare supercore without
masks can no longer be orthogonalized; or (ii) it takes masks after all, a uniform-only keyword on a polymorphic
function, the awkwardness Nick named. Under the refined rule (G2 binds the mask-holders; a raw primitive states "zero
padding in" and may return junk in the padding) it stays polymorphic and mask-free, with its junk output documented.
**The decision is narrow:** whether entry points that take bare supercores may return junk padding (refined) or must
take masks or go (strict).

**Nick's proposal (2026-11-24): a private core, two typed public functions.** "The value of the polymorphism is to
allow complicated shared logic to be written once and used for both types. Perhaps we can still support this by
breaking the operational core into a non-public subfunction which is called from two functions: one uniform with
masks, and another nonuniform without."

Claude's assessment: it resolves the tension, and it is the general form of what C3 and 3a decided.
`tt_left_orthogonalize` already has the shape: one scan skeleton, two step functions (`_tt_left_step`,
`_tt_left_step_pad_safe`), and only its public surface merges them behind an optional `pad_masks`. A private core may
take internal plumbing that varies by layer (the step, the per-step masks): it makes no public contract. Checked: the
polymorphic group C primitives keep zero padding on canonical input (`tt_reverse`, `tt_squash_tails`,
`t3_absorb_tucker_into_tt` measured, 0.0 in the output's padding; `xcat`, `affine_jet`, `data_metric` trivially, being
concatenation and elementwise). The factorizations are the only operations that make junk from clean input
(`tt_left_orthogonalize`, plain: entries up to 1.0). The public surface then has three kinds:

1. **Mask-holders** (the full `.data`, masks inside): liberal in, canonical out. The engine, the `ut3_*` / `utv_*`
   twins, `ut3_orthogonal_representations`. With the masks inside the object's data they need no mask argument: the
   tower appears only where masks were stripped off, in functions that take bare supercores.
2. **Zero-padding-preserving polymorphic primitives** (group C): public, polymorphic, mask-free; "zero padding in"
   as their precondition, "zero padding out" as their guarantee.
3. **Junk-creating operations** (factorizations whose result is kept): the shared logic in a private polymorphic core;
   two public functions of one type each, ragged (no masks) and uniform (masks required, canonical out).

What changes: `tt_left/right_orthogonalize` (a private sweep core; a ragged public function and the uniform pad-safe
one); `t3_orthogonal_representations` / `ut3_orthogonal_representations` (the sweep-order core private);
`down_orthogonalize_tucker_supercores` and `up_orthogonalize_tt_supercores` (uniform-only already, masks optional
today, plain SVD without them: masks required); `ut3svd_supercores` (`content` required).

With it, one statement holds for the whole public surface: **every public function returns zero padding whenever its
stated preconditions hold** (mask-holders state none; raw primitives, zero padding in). No public polymorphic function
carries a mask. Masks go deep only inside private cores and uniform-specific functions, where the mathematics needs
them. Guard 4's meta-test then checks both halves: junk-in for the mask-holders, canonical in and canonical out for the
primitives.

**Ruling (Nick, 2026-11-24): agreed.** This settles O6 and the mask-threading question: no public function is exempt
from returning zero padding under its stated preconditions; the three kinds of public function above; masks deep
inside only in private cores and uniform-specific functions; the guards recorded above.

### 10.10 The fitting-layer audit (F3; 2026-11-24)

*Scope and method.* The new fitting path: the oracle and the batch (`backend/fitting.py`), the packers
(`uniform_fitting.py`), `data_preparation.py`, the optimizers and the display, and the two frontends. A static scan
for form inference (tuple-or-array tests), conversions (pack, unpack, stack), per-mode loops and `kind` switches,
then a dynamic check of the oracle (`scratchpad/oracle_check.py`: unpack calls counted, and junk-in for every output).
The old kinds (`SamplingKind` and its subclasses, the six uniform kinds and their factories, `sumsq_over_probes`,
`block_sumsq_over_probes`) branch on the form too, but P20 deletes them ("The six uniform kinds and their factories
go"): not audited further.

*Findings.*

- **F3.1. The oracle's two local data sums leave the uniform layer.** `y_block_sumsq` and the `<y, M y>` of
  `f_resolution` are computed outside the engine (`_y_block_metric_y`, `_y_metric_y` → `_y_times_metric_y` →
  `_arrays_of`), which **unpacks a packed `y` and weight to the real widths** (`unpack_vectors`,
  `backend/fitting.py:855`), loops over the modes, and restacks (`:936`). Measured: **4 `unpack_vectors` calls in one
  uniform query** asking for both; 0 in the ragged twin. They run once per fit (the display's denominators,
  `optimizer_display.py:631-634`) and on a trust-region rejection or tracked row (`optimizers.py:1857-1858`): a hidden
  conversion inside the optimizer's loop, if an occasional one. Junk-in holds today (`f`, `r_block_sumsq`,
  `y_block_sumsq`, `f_resolution` and `g` bitwise equal with a garbage-padded and a clean packed batch): the engine
  masks, and the unpacking slices the padding away. **Under the rulings:** the oracle holds the frame, so it is a
  mask-holder. It computes the two sums on the packed product directly, masked by the shape's prefix mask on the edge
  (a data-space object no core touches is masked itself: O1 item 4). One operation, no unpack, no loop over the modes,
  junk-in by the mask.
- **F3.2. The batch reads its layer off its sample's container.** `_w_axes`, `_W`, `n_measurements`, `take` and
  `_take_data_space` branch on `isinstance(…, (tuple, list))`. After O2 this is exact: a tuple is the ragged form by
  definition, an array the packed one. The batch's methods are then representation-following (§10.7 principle 2),
  and the batch stays minimal (derive, don't store). **But O2 makes the ragged entries index a tuple**, and these
  methods assume the index is one `(d,)+W` array (`if self.kind == 'entries': return 1, np.ndim(batch.sample) - 1`);
  they need the ragged-index branch. A question for Nick: accept "the container is the layer" inside the batch, or
  make the batch's layer explicit (a field, or two batch types)? Claude's lean: accept, since after O2 it is a
  definition, not a guess.
- **F3.3. The automatic packing sites, all already ruled:** the frontend `prepare_batch` / `_is_packed` (a ragged batch
  on a uniform point raises, naming `fitting.pack_batch`; the packed form is checked structurally);
  `optimizers._frontend_draw` (removed); `data_preparation.ut3_unscale_coordinates` (the scales packed);
  `uniform_fitting.pack_sample` / `pack_data` (to `ut3_conversions`, split by kind, mixed edges refused).
- **F3.4. The `kind` switches** (group A, ruled split): `data_block_norms`, `rescale_direction`, `rescale_coordinates`
  (and the frontend's), and `check_batch` through the new checkers (O3 addendum).
- **F3.5. `data_block_norms` (probe)** loops over the modes and stacks (`data_preparation.py:195`). On a packed `y` it
  reduces over the padded edge with no masks, so its precondition is zero padding (C2's raw-primitive rule), and the
  loop can be one reduction over the packed array (F2).
- **F3.6. Smaller:** `optimizers._aligned_padded_frame` inspects the member's signature for `timing` (`:1074-1081`),
  which goes with C5's ruling (the member always returns four elements); `fitting._frontend_frame` repeats
  `frame_form`'s layer test by hand (`:507`); `check_coordinate_scales` takes the ragged form only, as the checkers
  will.
- **F3.7. The ragged path converts nothing** (0 unpack calls in a ragged query; the strict-forms run of §9.2 found no
  conversion inside any ragged call).

*For Nick:* F3.1's recommendation (the masked sum on the packed array), and F3.2's question (the container as the
layer inside the batch).

## 11. What is still to scout (updated 2026-11-24, for Nick to order)

**Done:** §10.4's O1 to O6 (ruled, §10.9); the case-by-case review of groups C' and C (§10.7, §10.8). The masking-cost
measurement (F4) is no longer needed: O1 was decided by the count (`1/W` of the sweep per call), with the option
defined and the side to zero fixed (the cores).

**Still to scout:**

1. ~~**The fitting layer (F3)**~~ (done, §10.10). Was: in scope (Nick): the four questions and the instrumentation, over `backend/fitting.py`
   (the oracle, the batch, `take`, the data sums), `uniform_fitting.py`, `data_preparation.py`, the display, and their
   frontends. Known so far: `_y_times_metric_y` unpacks a packed `y` to the real widths and loops over the modes.
2. **The per-mode loops (F2)**, case by case (Nick: "highly suspect"): the 13 sites of the F2 table. Under the layer
   rule a packed object is one array, so on the uniform layer most become one elementwise operation.
3. **The P3 split of group A in detail** (ruled: split along the types): the names and signatures of the ~35 functions
   the 15 `jets` / `data_preparation` helpers become, and of the checkers (O3 addendum).
4. **The docs build with a constrained `TypeVar`** (§10.6): verified at runtime only; the Sphinx typehints extension
   is not in the local env.

**Follow from the rulings; to confirm in the consolidation, not to scout:** F5 (`uniform_polymorphism.md`'s strict
wording becomes true; the equivalence contract's vector-I/O section is rewritten); F6 (the engine's `'w'` echo is
masked like every output; the frontend annotations become single-typed; `is_packed` goes); F7 (the jets converters
keep their input's form, as `data_metric` does: §10.1 item 5); the engine keyword's name (O1, open, not urgent).

**Consolidation.** The document grew by accretion: §10.2's prototypes are partly superseded by §10.5 to §10.9 (the
homes, the kind-split names, the checkers, the ragged-sequence rule, the mixed-edge refusal, the three kinds of public
function). Before the planning process, one section should state every decision in one place: the "what" that the
plan then sequences.

**For Nick:** the `jets.py` doctest added outside the process (§10.8, C7), to verify.

**Deferred by Nick's instruction:** sequencing and slices (O8).
