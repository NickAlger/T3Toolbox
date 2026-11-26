# The packing sub-arc: rulings of the planning session (2026-11-24)

A running record of what the maintainer and Claude settle while turning `PACKING_REFACTOR.md` into a plan. The plan
document is written from `PACKING_REFACTOR.md` **as amended by this file**; where the two disagree, this file wins.
The M arc's rulings from the same session are in `implementation/M_ARC_DECISIONS.md` (its last section).

## 1. Corrections to the consolidated document

**1a. The reach of G2 (ruled).** `PACKING_REFACTOR.md` states G2 two ways: "every output of a sampling operation"
(the investigation's §2) and "every public function … No exceptions" (its §1, from the O6 ruling), while the junk-in
audit measured only the 56 functions of the sampling modules. Ruling: **the sub-arc's scope is sampling and what is
downstream of it.** "Every public function returns zero padding under its stated preconditions" is the target for
that surface, and the coverage meta-test enumerates that surface, not every uniform `__all__`. An audit of the rest of
the uniform library (`ut3_linalg`, `ut3_svd`, uniform `+` and `*`, `resize`, the uniform retractions, the weights)
with the same plugin, and any fixes it finds, **join the backlog considered after Part 2 (a possible Part 3)**:
`dev/HANDOFF.md`, "Before the release, after Parts 1 and 2: consider this backlog".

**1b. G3's criterion for splitting a `kind`-switched function (ruled).** Not "split along the types": `data_metric`'s
types differ by kind exactly as the converters' do, and it stays one function. The criterion: **a function is split
when its computation depends on the kind (on where the mode axis or the jet axes sit), not when only its types do.**
`data_metric` is elementwise and needs no axis; `derivatives_to_taylor` must know where the jet axes are (leading in
an apply array and in each array of a ragged probe tuple, at offset 1 in a packed probe array), and an apply array and
a packed probe array are both arrays. The shape of the split that follows: **one private array-level core per
operation, taking the offset of the jet axes; `apply_X` calls it at offset 0; `probe_X` maps it over a tuple, or calls
it at offset 1 on a packed array** (the "private core, typed public functions" pattern of the O6 ruling).

**1c. The layer rule, restated (ruled; the maintainer's two principles and one definition).** The consolidated
sentence "no sampling function … mirrors its input's form, or infers a form from an object's type" contradicts
"functions without an object keep their input's form". The maintainer's principles: "First, the sampling functions
disallow mixing and matching of ragged with padded or uniform with unpadded. Second, if there is ambiguity, we force
the user to specify what they want rather than guessing." Stated so that each is checkable:

- **Definition.** For a per-mode object the container is the form: a tuple or a list of `d` arrays is ragged; one
  array is packed. Nothing else is read from the container.
- **No mixing.** A function with a T3 object requires the form of its object's layer and raises on the other, naming
  the packer. A function without an object requires all its per-mode arguments in one form and returns that form.
- **No guessing.** What the container cannot determine (the kind; where the jet axes sit) is said by the function's
  name or by an explicit argument, never inferred.

This supersedes, by name: slice P3's ruling "the type is decided by `kind`, never by the type of the object"
(`TARGET_DECISIONS.md`, "A `kind` argument where a function is three functions in one", 2026-11-17): after the split
the function's name decides the kind, and the container decides the form. **F3.2 is accepted on this basis** (the
batch tells ragged from packed by its sample's container, which is the definition, and takes its kind from the
oracle); it gains the ragged-index branch.

**1d. `data_metric` is not "annotations only" (ruled).** Under the sequences-only rule a ragged `r` pairs with a tuple
or list weight only; the branch pairing a tuple `r` with one mode-leading array (`jets.py:740-745`) goes. The
signature becomes `data_metric(r: DataSpace, weight: Optional[DataSpace], minibatch_scale) -> DataSpace`, the shared
`TypeVar` being the pairing rule.

## 2. The engine's status (ruled)

**The engine `sampling_variables` is exempt from G3 and from nothing else.** `kind`, the call type (a frame or a
corewise point), the layer and `want` are its data. It is not exempt from no mixing (it is the lowest function holding
the object, so it enforces the rule for all of sampling), from no guessing, or from G2 (it implements it). What it
reads from structure: the layer, from the frame's own container (the definition of 1c applied to the object); frame
or point, from the number of stores (decided earlier; unambiguous once the layer is known); the stacks `W`, `K`, `C`
from the operands' ranks given the explicit `order` (the library's convention; the documented blind spot of an axis
moved between `W` and `K` on every operand stays); the route, from which optional arguments are present (a held
variable replaces its own computation).

**The exemption does not leak:** every other public sampling function is a typed face, one engine call with `kind`,
the call type and a literal `want` fixed. The other sanctioned generic pieces stay as listed in `PACKING_REFACTOR.md`
(the oracle, the batch, `pack_batch`, the geometry protocol).

**The option to skip the input masking moves to Part 2** (it is an execution option: it changes no number when its
precondition holds). It may become part of the Plan object rather than a keyword: "Not certain, let's deal with it
when we get there." This sub-arc defines the precondition (canonical inputs) and delivers canonical outputs.
`PACKING_REFACTOR.md` §2.3's keyword is not built here.

## 3. The checkers, the backend module, and the frontend's packed object (ruled)

**3a. The checkers take plain structural arguments, not a target.** `check_vectors(vectors, order, shape)`,
`check_index(index, shape)`, `check_probe_data(value, sample, order, stack_shape=())`, `check_apply_data(value, sample,
order, stack_shape=())`, `check_entries_data(value, index, stack_shape=())`. A sample checker needs the mode sizes only;
a data checker reads `W` and the `N_i` off the already-checked sample and needs only the stack `C`. A backend user
holding a frame reads them with `frame_shape` and the newly public `frame_stack_shape` (M-arc ruling 22).
`LeastSquaresOracle.check_batch(batch, shape, stack_shape=())` keeps M2's signature. Reasons: the ambient transposes
have no object; the ragged frontend (about 30 call sites) passes `self.shape`; data may be checked before a tensor exists.

**3b. A new backend module, `backend/packing.py`** (the maintainer chose the name). It holds: the one reader of the
definition of 1c (is a container ragged or packed), so every refusal has one wording and a test can pin it; the five
checkers; the packers and unpackers. Not in it: `pack_batch` (stays in `uniform_fitting`), the oracle's `check_batch`,
the frame readers. It imports only `common` and the jet-shape reader of `contractions`. **No identity packers for
apply and entries data** (they have no mode axis and no forms; `pack_batch` branches); **one `unpack_vectors(packed,
shape)`** serves the sample and probe data (the same slicing).

**3c. The frontend: `PackedVectors`, a frozen dataclass** (in the uniform T3 module), the packed array with the widths
it was packed from: `array` (a leaf) and the static `shape`. "Padding does not encode widths", the reason a uniform T3
carries `shape` beside its supercores. Arrived at in steps: packing bound to the tensor was rejected by the maintainer
(users start with data and build the tensor in the fit; a tensor's ranks and values are irrelevant to packing; the
mode sizes are the problem's, read off the data); the widths on `LeastSquaresBatch` was rejected as too specific (a
custom loss; a fitted model being used); so the widths go on the per-mode object itself.

- **On the uniform frontend every per-mode object is a `PackedVectors`:** the sample of probe and apply, probe-type
  outputs (`z`, `Jp`, a model's residual), probe-type data (`y`, a weight, `c`). Apply and entries data, and the
  entries index, stay bare arrays.
- **Its constructors are the packers** (no free packing functions in the frontend), and **packing to a user-given
  `N` larger than the largest width is permitted** (optional argument). **It unpacks itself** (`z.unpack()`), so no
  shape can be passed wrong.
- **Each uniform frontend method** checks the shape (two static tuples: automatic, and valid under jit), passes
  `.array` to the backend, and wraps a probe-type result with its own shape. So G1 is automatic in the frontend again
  (it supersedes O4's opt-in for frontend users), and **the frontend has no public `check_*` methods** (ragged methods
  check as today; the backend keeps the checkers as tools).
- **No arithmetic on the class:** compute on `.array`, re-wrap with `dataclasses.replace(z, array=new)`.
- **The numerical helpers stay on plain data** (`data_metric`, the jet converters, `affine_jet`: tuples and arrays in
  both tiers, never a `PackedVectors`): prepare ragged, pack last, unpack at the end.
- **Padding is not validated at construction** (liberal in); every `PackedVectors` the library returns is canonical.
- `fitting.pack_batch(oracle, batch, N=None)` checks `y` and the weight against the sample's widths (the input's own
  consistency, no tensor needed), then packs; the sample's `shape` stands for the batch.
- The backend is unchanged: bare arrays and the pure functions of `backend/packing.py`.

## 3d. The weights, redesigned (ruled: "this design is better. Let's do it.")

**The diagnosis.** The maintainer: the weights are "a constant sticking point; it keeps coming up as an edge case in
open decisions, here and in previous sessions. This suggests that perhaps the design of the weights is not right."
One rule caused it: the weight as one array "of the type of `z`" in which each axis is independently 1 or full (the M
arc's Q5 and Q8). It met every seam differently: packing (an edge of 1 against `N_i`: the mixed-edge refusal, the
separate data-packing rule, the `shape` of an edge-1 `PackedVectors`); masking (only when the edge is `N`); `take` (a
weight with no `W` axis: M-arc decision 18); the placement of `K`; the mode axis (`blocks_to_weight` exists only to
turn a table into a tuple of arrays with trailing size-1 axes); and everywhere else in the library a size-1 axis that
broadcasts is a silent error.

**The design. Two named weights, each with exactly one legal shape; they multiply:**
`M = gamma * block_weight**2 * full_weight**2`.

| | `block_weight` | `full_weight` (name open) |
|---|---|---|
| shape | `(d,)+F` probe, `F` apply, `()` entries | exactly `z`'s shape |
| the same object on both layers | yes (no edge) | no: a tuple, or packed with full widths |
| packed, masked | never | like `y` |
| drawn by `take` | never (constant over the samples) | like `y` |
| applied | left-aligned on the leading axes | elementwise |

The measurement space is a direct sum of blocks, one per (mode, jet slot), which the library already treats as first
class (the display's table, `data_block_norms`, `equalizing_weight`, the `k!` of derivative coordinates, 2026.5.0's
`omega[mode, order]` matrix). `block_weight` is that table; `full_weight` is the weight per measured number, for
anything finer. No size-1 axes: the field's name says which weight it is (no guessing). Equalization is
`block_weight = equalizing_weight(data_block_norms(y))`, passed straight in; derivative data multiply the table by `k!`.

- **`y` and `full_weight` have exactly `z`'s shape, except that a stack axis `C` may be 1** (data shared by a stack).
  The "edge-1 `y`" of M-arc decision 13(b) no longer exists.
- **Removed:** `blocks_to_weight`; the data-packing rule, the mixed-edge refusal (O5) and `_pack_data_space` (`y` and
  a full weight pack exactly as vectors do: **one packer**; a `PackedVectors` always holds true widths, which is the
  answer to "what `shape` does an edge-1 weight hold"); the "edge is 1" branch of the engine's masking; the case
  decision 18 guards; the "mode axis of 1 broadcasts unchecked" limit.
- `data_metric(r, block_weight, full_weight, minibatch_scale)` stays one kind-free function (left alignment needs no
  kind).
- **`sample_weight` (shape `W`): not yet.** Until then a weight per sample is given as a full weight (for the probe
  kind at `y`'s memory; for apply and entries `z` is per sample already). The way to extend: a named factor with one
  fixed shape when a structured case earns it; the full weight is the catch-all.
- **This reverses, by name,** the M arc's Q5 and Q8 for weights (`TARGET_DECISIONS.md`, "every data-space object has
  the type of `z`, axis by axis either 1 or `z`'s size … The full form only, no block form") and
  `DATA_METRIC_PLAN.md` §2 and §4's "Uniform packing"; it reworks parts of M2 (the engine's weight handling, `take`,
  `data_metric`, the checkers) and the display's reading of the weight's shape.

## 3e. Inside the engine: the per-mode loops (F2) and F3.1 (ruled)

- **G3 inside the engine: nothing to do** (exempt; its steps are private and may take `kind` as a static).
- **The rule for F2: on the uniform layer a per-mode object is one array, so anything that is a map over the modes
  is one array operation with the mode axis leading; a Python loop over the `d` slices of a packed array does not
  appear in library code.** The ragged layer loops over the modes. (Row 5.2's "uniform: no map, one contraction with a
  leading `d`"; the same shape as 1b: one private array-level core taking the offset of the leading axes, mapped over a
  tuple or called once on a packed array.) Sites: the engine's data-space steps (`_modes` in `_weighted_sum`,
  `w_tilde_step`, the block sums); the entries kind's one-hot vectors on uniform; the jet converters on a packed probe
  array (F7: packed in, packed out); `data_block_norms` (probe); the oracle's two data sums. Exempt:
  `sampling_definitions` (the dense reference, ragged only).
- **F3.1 accepted:** `<y, M y>` and `y_block_sumsq` are a product and a reduction on the packed array (trivial now
  that `y` has full widths and exactly `z`'s shape); the oracle, which holds the frame, masks `y`'s padding by the
  shape's prefix mask first. `_arrays_of`'s unpacking goes.
- **Not in this:** the engine's per-site steps on uniform, one site at a time through `xmap`: a memory trade-off
  assigned to Part 2 (O-19).
- **The gate that pins the rule:** the uniform engine's compiled program has the same size at two values of `d` (a
  Python loop over the modes unrolls under jit; a scan or one array operation does not). To be established in the
  slice's design note: not yet checked that it holds today on the per-site steps.

## 3f. Scope and sequencing (ruled: "Agreed on all 3 points")

**The old kernels are deleted first** (today's P20, moved to the front of the sub-arc). `PART1_PLAN.md` §5.4: every
old kernel and old kind lost its last library caller at P16; they live on for the bridge tests only, hold the mirror,
the old packers and the old weight handling, and are the five junk-in failures of the investigation. Deleted first,
the layer rule holds for the whole sampling surface with no exemption list. The check-in on `t3_corewise_frame`
stays after P19 (a frontend helper, not a kernel). The slice begins with an inventory that nothing in the library,
the examples or the pages still reaches the old code (the plan's table is the evidence so far, not a fresh measurement).

**The scope line ("sampling and what is downstream of it") applied to the C and C' rulings.** In the sub-arc: the
`TypeVar`s and the annotations of `xcat` / `xappend` / `xprepend`, `affine_jet`, `data_metric`; `aligned_padded_frame`
always returning four elements. **To the backlog after Part 2 (a possible Part 3), the rulings standing and only the
timing moved:** `tt_left/right_orthogonalize` always returning both, and the pad-safe sweep's move;
`t3_orthogonal_representations` ragged only; `fv_to_t3` ragged only; masks required on the supercore
orthogonalizations, `content` on `ut3svd_supercores`; the `ranks` functions (`TypeVar`, mixed forms, `use_jax`
removed); `tt_reverse`, `tt_squash_tails`, `t3_absorb_tucker_into_tt`; the `ut3_to_t3` tree types. With them, the audit
of the rest of the uniform library for G2 (1a).

**The slices, letter K, before P17** (so the examples and pages are rewritten once):

| Slice | Content | Size |
|---|---|---|
| K0 | Delete the old kernels, the old kinds and the bridge (today's P20, moved) | medium, mechanical |
| K1 | The weights: `block_weight` and `full_weight`; `y` at exact shape | large (like M2) |
| K2 | `backend/packing.py`: the form reader, packers, unpackers, checkers; `frame_stack_shape` public | medium, additive |
| K3 | The ragged layer takes sequences only (the sample, the data, the entries index) | medium, many tests |
| K4 | The uniform backend takes packed only: no conversion, G2's masking, the data-space steps as single array operations, the program-size gate | large |
| K5 | The frontend: `PackedVectors`, the uniform methods, `fitting.pack_batch`, no automatic packing in the optimizers | large |
| K6 | The split of the `kind`-switched helpers, with the remaining per-mode loops | medium to large |
| K7 | The guards: hostile fixtures by default, the junk-in coverage meta-test | small |

The weights first (they remove the data-packing rule, so K2 to K5 never port it); the module additive, before
anything is refused; ragged before uniform; the guards last, on the final surface. The translation of the old call
form stays alive until P17, adapted in K1 (the old `omega` matrix is `block_weight`). Pages with doctests are fixed in
the slice that breaks them; the narrative documentation waits for P19.

**The process for the plan (the maintainer, 2026-11-24):** Claude drafts it and is the writer; the critic is one Opus
agent at extra-high effort, told to familiarize itself with the library first, to try hard to find flaws, logical
inconsistencies and unforeseen problems, and to think from first principles, not from rules or convention; the writer
has the final say this time, without brushing the critic's concerns aside; then the maintainer and Claude review the
result together.

## 4. The entries index (ruled: the O2 ruling confirmed, with three details)

On the ragged layer the index is a tuple or a list of `d` integer arrays, each of shape `W`; on the uniform layer one
`(d,)+W` array; `pack_index` is `np.stack`, `unpack_index` is `tuple`. The reason recorded with the confirmation, beside
the axis coincidence (which the tuple reduces and does not remove: `tuple(points)` with `M = d` points passes): **the
tuple is numpy's own multi-index** (`np.nonzero`, `np.unravel_index`, `np.ravel_multi_index`; `dense[tuple(idx)]` is the
entries, `dense[idx]` with a `(d,)+W` array indexes along axis 0), so `x.entries(idx)` and `x.to_dense()[idx]` take the
same object and `x.entries(np.nonzero(mask))` works; the definition of 1c keeps no exception; and every kind has the
one uniform workflow (check, pack, sample). Rejected: one array on both layers (an exception, no numpy meaning); a tuple
on both layers (a hidden stack at every uniform call).

- **Lists are accepted** (`x.entries([9, 4, 7])` is the entry at `(9, 4, 7)`), as for every per-mode argument, although
  numpy reads a list as a fancy index; only a single array is refused, the message naming `tuple(index)`.
- **No broadcasting between the `d` index arrays:** one shape `W` exactly, as the stack checks require everywhere.
- **The uniform frontend refuses a tuple** (`ux.entries((9, 4, 7))` raises, naming `pack_index`).

## 5. Names and small placements (ruled)

- The second weight is **`full_weight`**.
- **`PackedVectors.pack(vectors, N=None, order=())`** (a classmethod; one constructor, since there is one packing
  rule; the frontend's argument is `order`, the backend's `len_P = len(order)`) and **`z.unpack()`**. The class is
  exported at the package's top level, as every other uniform class is.
- **The split helpers are named kind first:** `probe_derivatives_to_taylor` / `apply_derivatives_to_taylor`,
  `probe_data_block_norms` / `apply_data_block_norms` / `entries_data_block_norms`, and so on; the plan lists every name
  in one table.
- **The `from_t3` slice** (raise for a width below a rank; decided earlier) stays its own small slice after K7.
- Decided by Claude in the draft, flagged for the review: the `k!` of derivative coordinates goes into
  `block_weight` only (`prepare_derivative_data` with no weight returns the `k!` table); `index_to_one_hot` keeps its
  input's form (a packed index gives a packed array at a stated `N`); `LeastSquaresBatch`'s field order and the
  engine's two weight arguments; the list of what the plan supersedes by name; the consequences for P17, the dry port
  and P19; the slice where CI first builds the docs with a constrained `TypeVar`, and the fallback.
- Cost accepted by the maintainer: two or three critic rounds (Opus, extra-high effort, about 1 to 2M tokens in all),
  and one bounded Sonnet agent for the plan's measurements.

## 6. After the critic (2026-11-25): where the plan departs from these notes

`PACKING_PLAN.md` (revision 3) was reviewed by a critic in two rounds (`reviews/packing_plan/`); the critic signed
off. The plan departs from the rulings above in the places its §13 lists, **all pending the maintainer's review**:

- 3c: a batch holds plain arrays and carries its own static `widths`; `PackedVectors` is the type of the uniform
  frontend's sampling methods, not of a batch's fields (a batch of wrappers does not run with one batch class for both
  tiers). The maintainer's words in the session, which 3c above records too strongly as a rejection: "I like the idea
  of having the uniform batch remember its shape in the frontend with a frozen dataclass. The name LeastSquaresBatch
  seems too specific though. The user may be fitting with their own loss function which is not least squares. Or the
  fit may be done and now they are calling these functions to use the fitted model."
- 3c: the field is `widths`, not `shape` (numpy reads `.shape`).
- 1b, 1d, 3d: `data_metric` is split by kind and the batch's method moves to the oracle (a block weight cannot be
  checked without the kind); the criterion gains "or the structural check of its arguments"; in the public metric the
  full weight keeps size-1 axes, the mode axis exact; `DataSpace` is dropped.
- F7 of the investigation, and 3c's "helpers on plain data": the probe helpers are ragged only, except
  `probe_data_metric` and `affine_jet`.
- G2: the data-space inputs and the held variables are masked by selection (their padding may hold `inf` or `nan`).
- 3f: the frontend slice (K4) comes before the uniform backend's (K5).
- Put to the maintainer, not adopted: whether the sample axes `W` of a full weight may be 1 (the plan's §13.7).

## 7. The maintainer's review of the plan's §13 (2026-11-25, in progress)

- **§13.2, the field's name: `mode_sizes`** (ruled), on `PackedVectors`, on the batch, and as `unpack_vectors`'
  argument. "shape" is ambiguous beside stacking and is the array protocol's word; "widths" does not say whose;
  `tensor_shape` names a tensor that may not exist yet (data first); "mode sizes" is the library's own phrase for
  this tuple (120 uses in `t3toolbox/`; `frame_shape` returns "the mode sizes").
- **§13.4, the probe helpers ragged only: accepted.** "This also encourages the desired pipeline which is for a user
  to prepare the data and make it uniform once at the outset, then keep it that way during their optimization." A
  packed branch can be added later if a need shows.
- **§13.6, the slice order: accepted.** "In general I think you have a better sense of the proper order of operations
  here and I don't want to micromanage."
- **§13.5, masking by selection: accepted in principle,** pending a check for hidden issues (he recalls trouble, in
  the research code, with integer tuples of "hot" indices per edge).
- **§13.1 and §13.3 (the batch's `mode_sizes`; the metric split and `oracle.data_metric`): acceptable, but "maybe we
  can deal with these by reconsidering the batch design."** Under discussion.
- **§13.7 (may `W` be 1 on a full weight): to be discussed after 1 to 6.**
- **§13.5, ruled: masking by selection, as the engine's one mechanism for everything it masks, the sample
  included** (so the sample is masked again, by selection, where the plan's revision 3 leaves it unmasked). The
  documented contract: the padding of data may hold anything; the padding of cores must be finite, because the rest of
  the uniform library multiplies. Checked for hidden issues (2026-11-25): what the maintainer recalled was the
  representation of the real slots (integer tuples of hot indices against boolean masks:
  `docs/contributor/uniform_rank_masks_rationale.md`, Part II), not how a boolean mask is applied; `where(mask, x, 0)`
  keeps the shape, needs no gather, and would work with traced masks as the multiply does. Measured on both
  environments: same order of cost eager and under jit; the gradient through it is exactly zero on the padding with
  `inf` there. Cares for the implementation: the zero has the array's dtype (on numpy 2 a `np.float64(0)` promotes a
  float32 array); values are compared, not bytes (signed zeros); `where`, never boolean indexing. Precedent in the
  library: `ut3_operations.py:410`, `ufv_operations.py:314`.
- **B1, ruled: `kind` and `order` move from the oracle into the batch** (static fields). The batch knows what kind
  of data it holds, so `data_metric(r)`, `take(indices)`, `n_measurements`, `check(shape, stack_shape)` and `pack(N)`
  are methods of the batch; the oracle keeps the objective's rule (`query`). This reverses, by name, "the oracle is
  all static, the batch is all leaves" (`TARGET_DECISIONS.md`, 8.3; the jit argument holds, since the static fields
  are constant across draws) and replaces the plan's `LeastSquaresOracle.data_metric(batch, r)`.
- **B2 (a frontend batch of `PackedVectors`, distinct from the backend's raw batch): undecided;** to be taken up
  after the other decisions.
- **§13.7, ruled (the maintainer "leaning towards" it): a full weight may have size 1 on the sample axes `W`, as on
  the stack axes `C`; `y` stays exact on `W`** (only its `C` axes may be 1). Stated as a principle: **a size of 1 is
  allowed exactly on stack axes** (`W`, `C`: collections of independent instances, where it means "the same for every
  instance"), **never on structural axes** (the mode, the jet slots, the edge: the parts of one measurement, where
  every troublesome seam was). `y` is the stated exception on `W`: a `y` that is the same for every sample is not a
  dataset, and refusing it catches one sample's data passed against many. The maintainer's view of the uses: "In the
  typical scenario, the weighting is independent of the W and C stacks. I can envision some weighting ideas that use
  the W stack (e.g., reduce the impact of outlier samples). The C-stack is not currently supported within the
  optimizers … I don't see C-dependent weighting happening any time soon, if ever." So the typical full weight is
  `F + (1,)*len(W) + (1,)*len(C) + (N_i,)` per mode, at no cost in `W`; `take`'s "every `W` axis is 1: no gather"
  branch stays. Not covered: a weight per sample and slot that is constant along the edge (it needs an edge of 1; for
  the probe kind it is a full weight at `y`'s memory).
- **The plan's §12 (Claude's ten decisions in the draft): no objection** (the maintainer, 2026-11-25). Item 5 of that
  list ("the sample is no longer masked") is overtaken by the ruling on §13.5 above; with B1 the batch's fields read
  `kind, order, sample, y, block_weight, full_weight, minibatch_scale, drawn_indices, mode_sizes`.
- **B2, ruled: design F** ("Let's go with F"), after seeing both designs as code: **the frontend has its own batch
  class, whose per-mode members are `PackedVectors`**, and hands the backend its `.data` (a backend batch of plain
  arrays), as every frontend class does. Against the plan's revision 3 (one shared class with a static field): in the
  first thing a user does after a fit (the model used at the batch's sample points) the shared class needs
  `PackedVectors(batch.sample, batch.mode_sizes)`, and a packed probe object has two types (`batch.y` an array,
  `model.residual` a `PackedVectors`); under F every packed per-mode object of the frontend is a `PackedVectors`, and a
  batch built by hand is checked at construction (its members state their mode sizes and must agree). Costs accepted:
  a second batch class with thin methods; the frontend hands over `.data` of each drawn batch; it reverses "one class
  each for both tiers" for the batch (the oracle stays one class). The backend batch then needs no `mode_sizes` field.

## 8. The plan after the maintainer's review (2026-11-25)

`PACKING_PLAN.md` is at revision 5, signed off by the critic after four rounds (`reviews/packing_plan/REVIEW_r1.md`
to `REVIEW_r4.md`, with the writer's responses; `DIFF_rev4_to_rev5.txt`). Revision 4 carried the rulings of §7;
revision 5 specified the seams the critic found in them (who still reads the oracle's kind; one kind and one order
per fit; the public metric obeying the stack-axis principle; the modes of a probe object agreeing; the two batch
classes of one name; the constructor checks moving with the fields; K1's part 0). **Pending the maintainer:** the
plan's §14 (nine decisions Claude made in writing revisions 4 and 5, the first being the frontend batch's name), and
his sign-off of the plan as a whole. Then: the dated note in `TARGET_DECISIONS.md`, and K0's design note.

## 9. Signed off (2026-11-25)

The maintainer: "The decisions in 14 look fine, including the naming collision with LeastSquaresBatch. I sign off on
the design." The dated note is in `TARGET_DECISIONS.md` ("The packing sub-arc: the rulings of 2026-11-24 and
2026-11-25"). The build is orchestrated from `implementation/K_ORCHESTRATOR_HANDOFF.md`.
