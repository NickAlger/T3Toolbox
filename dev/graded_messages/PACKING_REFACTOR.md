# The packing refactor: decisions (consolidated 2026-11-24)

The decisions of the packing investigation in one place: **what** to do, for the planning process that will sequence
it (the slices and the order are that process's, not this document's). The history, the evidence, the measurements
and every argument are in `PACKING_INVESTIGATION.md` (kept for reference through the refactor; section numbers below
point there). Nothing here is built: every change goes through the review process (proposal, critic,
implementation, review, gates).

**Scope** (Nick): every sampling function and everything directly downstream of them (the fitting layer, the data
preparation, the optimizers' use of the oracle, the display, the frontends); not, for example, `t3svd`. The M arc's
decisions are open to revision where these decisions say so.

## 1. Principles

- **G1. Widths.** Data whose widths differ from the target's shape is a user's bug, never a request to pad. Packing
  erases the widths, so the check runs on the ragged form, by the checkers (§2.2), before packing: opt-in on the
  uniform layer (O4).
- **G2. Liberal in, canonical out.** On the uniform layer input padding is junk (finite) and every output's padding
  is exactly zero. Refined (§9.8, §10.8 C2, §10.9 O1, O5, O6):
  - "liberal in" is the contract of the functions that **hold the masks** (they mask on entry);
  - a raw-array primitive states **zero padding in** as its precondition, and keeps zero padding where it can;
  - **every packer returns a canonical object**;
  - on every padded contraction the engine zeroes the **core** side; data-space objects that no core touches (`y`, the
    weight, a held `r` or `Jp`) are masked themselves;
  - **every public function returns zero padding whenever its stated preconditions hold.** No exceptions.
- **G3. One mathematical operation per function, one return type** (§10.6 to §10.8):
  - functions that do different mathematics have different names; split **along the types**, not the kind labels (probe
    and apply share the sample's type; the data-space objects differ by kind);
  - a **switch** (an argument choosing the operation or the output type) is split; an **index** (choosing one of a
    family of like terms, the same output type) is data;
  - the same operation over two representations is **parametric**: one function, annotated with a constrained
    `TypeVar` shared by the co-varying parameters and the returns; mixed representations are a structural error;
  - parametric needs **parallel outputs**: a branch with an extra input or an incomplete output is a different
    function;
  - a **recursive container type is one type** (a tree alias; the depth in the contract);
  - a **diagnostic** may take its layer's natural form;
  - sanctioned exceptions, generic machinery: the engine, the oracle, the batch, `pack_batch`, the geometry protocol;
    the reference definitions of `sampling_definitions` are exempt.
- **The layer rule** (§9.6). Each layer takes and returns its own form. No sampling function converts between the
  forms, mirrors its input's form, or infers a form from an object's type; a wrong form raises, naming the helper that
  converts it.
- **The forms**, for every per-mode object (the sample, the data-space objects, the entries index):
  - *packed*: one array, the mode axis leading with exactly `d` (never broadcast over the modes); every other axis one
    size for all modes (data-space axes by the broadcast rule); the edge `N`, or 1 for a data-space object constant
    along it;
  - *ragged*: a sequence (tuple or list) of `d` arrays, each with its own mode's shape. **A single array is refused,
    the entries index included** (O2: the axis-order coincidence). On the ragged layer the index is `d` integer arrays
    of shape `W`, numpy's own multi-index convention;
  - only the edge has padding: jet axes brought to the largest size are zero *coefficients* (data); joining data-space
    axes across the modes is broadcasting.
- **Three kinds of public function** (O6):
  1. **mask-holders** (the full `.data`, masks inside: the engine, the twins, `ut3_orthogonal_representations`):
     liberal in, canonical out; they need no mask argument;
  2. **zero-padding-preserving polymorphic primitives** (group C): mask-free; zero padding in, zero padding out;
  3. **junk-creating operations** (factorizations whose result is kept): the shared logic in a **private polymorphic
     core**; two public functions of one type each, ragged (no masks) and uniform (masks required, canonical out).
  Masks go deep only inside private cores and uniform-specific functions, where the mathematics needs them (S1b).
- **Functions without an object** (`data_metric`, the jets converters, `affine_jet`) keep their input's form; mixed
  forms raise.
- **Guards against erosion of junk-in** (O6): junk-in lives in few places (the engine's entry, the mask-holders);
  relying on zero padding is legal only when stated (a raw primitive's precondition, or the engine's unmasked mode);
  test fixtures put garbage in the padding by default; a coverage meta-test requires a junk-in case for every public
  uniform function (prototype: `packing_audit/junk_in_plugin.py`). Lesson from the prototype: a check reads a form by
  kind, never by shape.

## 2. The public surface

### 2.1 Packers and unpackers (home: `backend/ut3_conversions.py`; O3, §10.5)

- `pack_vectors(vectors, N=None, len_P=0)` / `unpack_vectors(packed, shape)`: the sample of probe and apply.
- `pack_index(index)` (`np.stack`) / `unpack_index(index)` (`tuple`): the entries sample.
- `pack_probe_data(value, N=None)` / `unpack_probe_data(value, shape)`; `pack_apply_data`, `pack_entries_data` and their
  unpackers are identities (one sample packer and one data packer per kind). `pack_data`'s `order` goes.
- The packers take no shape and check nothing beyond their input's own form: modes with different non-edge axes, or
  **mixed edges** (some 1, some full: O5), are refused, naming the fix.
- Frontend: the packers as module-level functions; the unpackers as instance methods of `UniformTuckerTensorTrain` and
  `UT3Tangent` (they need the shape); `fitting.pack_batch(oracle, batch, N)`. The backend `pack_batch` stays in
  `uniform_fitting` (a composite over a kind-generic container).

### 2.2 Checkers (home: `ut3_conversions`; O3 addendum)

- Ragged form only (the only form with widths): `check_vectors(vectors, order, target)`, `check_index(index, target)`,
  `check_probe_data(value, sample, order, target)`, `check_apply_data`, `check_entries_data`. The target is any
  object's `.data` (a point or a frame, ragged or uniform): its shape, `N` and stack `C`.
- Frontend: methods of the object checked against, on the four classes (a tangent checks against its frame).
- They replace `jets.check_sample`, `check_data`, `check_weight`; `LeastSquaresOracle.check_batch(batch, target)` is
  composed of them.
- The workflow, documented on the packers, the uniform pages and the uniform examples: **check the ragged data against
  the target, then pack, then sample.** Packed data are checked structurally (`d`, `N`, the broadcast rule, `C`) by the
  sampling functions, always.

### 2.3 The engine (`sampling_variables`; O1, O2)

- Uniform: the sample, `y`, the weight and the held variables must be packed; a sequence raises, naming the packer.
  Ragged: sequences only.
- Masking: zero the cores on padded contractions; mask the data-space objects no core touches; no longer mask the
  sample; mask every output, the `'w'` echo included.
- **One keyword to skip the input masking**: default masks; off, the caller guarantees canonical inputs (a
  precondition checked in safe mode, so it saves cost only in unsafe mode or under jit). Output masking stays on.
  Name open.

### 2.4 The uniform twins and the frontends

No mirror. The `Vectors` union goes; per-mode arguments are packed `NDArray`s, probe-type returns packed. The 52
`pack_if_ragged` / `unpack_vectors` / `is_packed` lines in the twins go, as does `_check_sample`'s branch on the form.
Annotations single-typed.

### 2.5 The fitting layer (§10.10)

- The frontend `prepare_batch` takes a batch in `x`'s layer: a ragged batch on a uniform point raises, naming
  `fitting.pack_batch`; the packed form is checked structurally. `optimizers._frontend_draw` goes (a draw returns
  batches in `x0`'s layer: pack once, draw with `take`).
- `ut3_unscale_coordinates` takes packed scales.
- `data_block_norms` (probe): zero padding in as its precondition; one reduction over the packed array, not a loop.
- **Pending Nick's ruling:** F3.1, the oracle's two local sums (`y_block_sumsq`, the `<y, M y>` of `f_resolution`)
  computed on the packed product masked by the shape's prefix mask (no unpack, no loop); F3.2, the batch keeps
  reading its layer off its sample's container (exact after O2), and gains the ragged-index branch.

### 2.6 The `kind`-switched helpers (G3 group A; §10.6)

The 15 helpers of `jets` and `data_preparation` split along the types (about 35 functions; the jet helpers have no
entries version). `blocks_to_weight` returns the weight **in `y`'s form** (a tuple for a ragged `y`, packed for a
packed `y`), which removes 262 of the 275 test units the ragged rule touches. `data_metric` stays one kind-free
function (C7); its mode-axis broadcast limit stays with the checkers (a doctest shows it; §10.8 C7).

### 2.7 Group C: one function each, parametric (§10.8)

`TypeVar`s in `common.py`: `PerSite` (site- and mode-indexed objects) and `DataSpace` (`data_metric`), the names
final in the plan.

- **C1** `xcat`, `xappend`, `xprepend`: the empty-sequence branches return tuples.
- **C2** `tt_reverse`, `tt_squash_tails`: state the zero-padding precondition.
- **C3** `tt_left/right_orthogonalize`: always return the cores and the variation cores (frontend
  `left/right_orthogonalize_tt_cores` too). The pad-safe sweep becomes its own uniform function over a private sweep
  core: **a move, not a rewrite**, checked bitwise before and after (the sweep is deterministic); the tests that pin it:
  `TestPadSafeFrame`, `TestPadSafeSvd`, `TestPadSafePinnedCompletion`; the reasoning: `docs/pad_safe_svd.tex`.
- **C4** `t3_absorb_tucker_into_tt`: mixed forms a structural error (one silently malformed case found); precondition.
- **C5** `aligned_padded_frame`: always four elements, `seconds=None` unless `timing`; the movements stay
  layer-natural (a diagnostic).
- **C6** `affine_jet`, **C7** `data_metric`: annotations only.

### 2.8 The C' cases (§10.7)

- `ut3_to_t3` and its four siblings return one **recursive tree type** (one concrete alias per leaf, 3.9-compatible
  forward reference; the depth `len(C)` in the contract).
- `compute_minimal_ranks`, `oversampled_ranks`, `compute_orthogonal_representation_ranks`: a shared `TypeVar`
  `Ranks`; mixed forms a clear structural error; `use_jax` removed (also from `compute_raw_sweep_ranks`): ranks are
  host numpy.
- `t3_orthogonal_representations`: ragged only; the pad-safe sweep becomes the body of
  `ufv_conversions.ut3_orthogonal_representations`; the unmasked uniform path is deleted.
- `fv_to_t3`: ragged only (a uniform twin only if needed, returning a complete `.data` with masks). Its `index` is
  data.
- `down_orthogonalize_tucker_supercores`, `up_orthogonalize_tt_supercores`: masks required.
  `ut3svd_supercores`: `content` required.

### 2.9 Removed

`is_packed`, `pack_if_ragged`, `_pack_data_space` (folded into the data packers), `fitting._is_packed`,
`_arrays_of`'s unpacking, `optimizers._frontend_draw`, the `Vectors` alias, `use_jax` in the ranks, the dead uniform
branches above, `_aligned_padded_frame`'s signature introspection for `timing`.

## 3. Documentation to change

The equivalence contract's vector-I/O section (the contract becomes `unpack(op_uniform(pack(v))) == op_ragged(v)`);
`docs/contributor/uniform_polymorphism.md:44-47` (its strict wording becomes true, F5); the glossary's "packed
vectors"; `naming_conventions.md` (the packers are layer conversions, not fitting infrastructure); the data-metric
plan's §2 rule (ragged: a tuple or list only; uniform: packed only); `testing_strategy.md` (hostile fixtures, the
meta-test); the backend docstrings promising the one-array form (about 23 lines); the uniform examples (explicit
packing, with the check); an upgrade note (breaking: uniform takes and returns packed; ragged refuses arrays, the index
included; `blocks_to_weight`'s form; the packers and checkers renamed and moved).

## 4. Blast radius, measured (for the planning; §9, §10.9)

- Refusing ragged input on uniform: 135 tests, 13 module doctests, `sharing.md`, the two uniform examples.
- Refusing arrays on ragged: 275 test units, 262 of them through `blocks_to_weight` (fixed at the source); the entries
  index: 135 tests, 9 module doctests, 1 page, `examples/fit_hilbert_from_entries_lbfgs.py`.
- Junk-in today: 3,084 of 3,089 checks bitwise equal over 56 public uniform functions; the 5 others are the old
  `utv_precompute_*_frame_sweep(_jets)` storing a packed sample as given (no wrong number; P20 deletes them).

## 5. Still open

- Nick's rulings on F3.1 and F3.2 (§2.5).
- **F2, the per-mode loops, case by case** (Nick: "highly suspect"): `sampling_steps._modes` and its users
  (`_weighted_sum`, `w_tilde_step`, the block sums), the entries one-hot builder, the jets helpers' loops,
  `data_block_norms`, the dense definitions. Under the layer rule most become one operation on the packed array.
- The names and signatures of the group A split and of the checkers.
- The `TypeVar` names, and the docs build with a constrained `TypeVar` (verified at runtime only).
- The engine keyword's name.
- The `jets.py` doctest added outside the process (§10.8 C7): for Nick to verify.
- Sequencing: deferred to the planning process. Facts for it: P17 switches the examples and pages; P20 deletes the old
  kernels and kinds.

## 6. Tools (`dev/graded_messages/packing_audit/`)

`sampling_audit_plugin.py` + `analyze.py` (forms in and out, conversions inside calls, packed re-runs);
`strict_forms_plugin.py` + `strict_analyze.py` + `strict_run_script.py` (where refusing the other form breaks);
`ragged_sequences_plugin.py` (arrays where the ragged layer requires sequences); `junk_in_plugin.py` +
`junk_in_analyze.py` (the prototype of the coverage meta-test). Each docstring has its usage and limits.
