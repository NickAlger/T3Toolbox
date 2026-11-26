# The packing sub-arc: slices K0 to K7, between the M arc and P17

Status: revision 5 (2026-11-25), **signed off by the critic** (`reviews/packing_plan/REVIEW_r4.md`) **and by the
maintainer** ("The decisions in 14 look fine, including the naming collision with LeastSquaresBatch. I sign off on the
design."). The critic signed off on revision 3 after two rounds
(`reviews/packing_plan/REVIEW_r1.md`, `RESPONSE_r1.md`, `REVIEW_r2.md`, `RESPONSE_r2.md`). The maintainer then reviewed
it with Claude and ruled on every open item (§13; `PACKING_PLAN_NOTES.md` §7); this revision carries those rulings,
four of which change the design (the batch, the masking, the full weight's sample axes, a field's name). The critic
reviewed that revision too (`REVIEW_r3.md`: no ruling challenged, six seams left unspecified; `RESPONSE_r3.md`), and
revision 5 specifies them. **Nothing here is built.**
Implements: `PACKING_REFACTOR.md` (the consolidated decisions of the packing investigation) **as amended by**
`PACKING_PLAN_NOTES.md` (the rulings of the planning session). Where those two disagree the notes win. The notes' §7 is the
maintainer's review of this plan; §14 lists what Claude decided in writing revision 4.
Evidence: `PACKING_INVESTIGATION.md` (measurements; tools in `packing_audit/`); `packing_audit/K_MEASUREMENTS.md` (the
counts of this plan, each with its method); the critic's experiments E1 to E7 (named in `REVIEW_r1.md` §1 and `REVIEW_r2.md` §1).
`file:line` refers to `3375d213` (branch `graded-messages`).
Out of scope: everything upstream of sampling (frames, orthogonalization, the T3-SVD, the rank functions: §10); the
option to skip the engine's input masking (Part 2); the sampling audit's decisions 3 and 4 (pending with the
maintainer); a `sample_weight`; a non-diagonal data metric.

**In brief.**

- **Two principles and one definition** govern how per-mode objects cross the ragged/uniform seam. *Definition:* the
  container is the form (a tuple or a list of `d` arrays is ragged; one array is packed). *No mixing:* a function with
  a T3 object requires the form of its object's layer; a function without one requires all its per-mode arguments in
  one form and returns that form. *No guessing:* what the container cannot determine is said by the function's name
  or an explicit argument.
- **No sampling function packs or unpacks.** The mirror, `is_packed`, `pack_if_ragged` and the automatic packing of the
  fitting frontend go. On the ragged layer every per-mode input is a sequence of `d` arrays, the entries index
  included.
- **The weight becomes two named weights**, `block_weight` (one number per block: `(d,)+F`, `F` or `()`) and
  `full_weight` (of `z`'s shape; a size of 1 only on a stack axis), which multiply. `y` has exactly `z`'s shape. This removes `blocks_to_weight`, the
  data-packing rule and every edge-of-1 special case. The public metric is split by kind, because a block weight
  cannot be checked without it.
- **Backend:** one new module, `backend/packing.py` (the form reader, the packers and unpackers, the checkers), pure
  functions on plain data.
- **Frontend:** a frozen dataclass `PackedVectors` (the packed array with the widths it was packed from) is every
  per-mode object of the uniform frontend, in its sampling methods and in its batch. The batch knows its kind. The
  check of the mode sizes is automatic; the frontend has no `check_*` methods.
- **G2, liberal in, canonical out,** for sampling and what is downstream of it; **G3, one operation per function, one
  return type,** with the engine its one exemption.
- **Eight slices, K0 to K7, before P17.** K0 deletes the old kernels (today's P20, moved to the front). The frontend
  (K4) switches before the uniform backend refuses anything (K5).

---

## 1. The principles

### 1.1 The definition, and the two principles

The maintainer (2026-11-24): "First, the sampling functions disallow mixing and matching of ragged with padded or
uniform with unpadded. Second, if there is ambiguity, we force the user to specify what they want rather than
guessing." Stated so that each can be checked:

- **Definition.** For a per-mode object the container is the form: a tuple or a list of `d` arrays is **ragged**; one
  array is **packed**. Nothing else is read from the container.
- **No mixing.** A function with a T3 object (a point, a frame, a tangent) requires the form of its object's layer and
  raises on the other, naming what converts it. A function without an object requires all its per-mode arguments in
  one form and returns that form; mixed forms raise.
- **No guessing.** What the container cannot determine (the kind; where the jet axes sit) is said by the function's
  name or by an explicit argument, never inferred.

**The per-mode objects** are: the sample of probe and apply (the input jets `ww`); the entries index; the probe kind's
objects of the type of `z` (`z`, `y`, `full_weight`, `r`, `Jp`, `c`); the coordinate scales of
`unscale_coordinates`. Apply's and entries' objects of the type of `z` have no mode axis: one array, the same on both
layers, with no forms. The directions (`variations`, `core_directions`), the centers `Vc` and the held variables are
per-site objects of the frame's own layer: the engine requires them in it (§3.2).

**The forms:**

| | ragged | packed |
|---|---|---|
| input jets `ww` | `len=d`, `elm_shape=F_in_i+W+(Ni,)` | `(d,)+F_in+W+(N,)`: jet axes zero-filled to the largest `F_in_i`, the edge zero-padded |
| entries index | `len=d`, `elm_shape=W`, int | `(d,)+W`, int |
| probe objects of the type of `z` | `len=d`, `elm_shape=F+W+[K]+C+(Ni,)` | `(d,)+F+W+[K]+C+(N,)`, the edge zero-padded |

Only the edge has padding. `N >= max(Ni)`: a uniform T3 may be padded wider than its largest mode
(`UniformTuckerTensorTrain.from_t3(x, N=...)`, `uniform_tucker_tensor_train.py:232`), so every packer takes an optional
`N`.

**What packing does not keep.** The widths `N_i` (§1.2). And, for input jets whose jet sizes differ by mode, the
per-mode sizes `F_in_i`: the zeros written on a jet axis are coefficients, so the packed jet is the same polynomial, but
`unpack(pack(ww))` has the largest jet sizes, not `ww`'s. The equivalence contract is therefore
`unpack(op_uniform(pack(v))) == op_ragged(v)` **up to zero coefficients on the jet axes of the leaves** (`xi`, `xi_dot`,
`xi_U_dot`, the `'w'` echo): equal as polynomials.

**This supersedes, by name:** slice P3's ruling "the type is decided by `kind`, never by the type of the object"
(`TARGET_DECISIONS.md:498-511`): the function's name now decides the kind, and the container decides the form. And the
sentence of `PACKING_REFACTOR.md` §1 "no sampling function … mirrors its input's form, or infers a form from an
object's type", which contradicts its own "functions without an object keep their input's form".

### 1.2 G1: widths

Data whose widths differ from the target's mode sizes are a user's bug, never a request to pad. The widths are the
one thing packing destroys that cannot be told apart from valid data afterwards (a missing entry looks like a zero
pad). So they are carried, not re-derived:

- **Frontend:** automatic. A `PackedVectors` carries the mode sizes it was packed from, a frontend batch is made of
  them (§4), and every uniform frontend entry compares them with its object's `shape`.
- **Backend:** the backend gives checkers, not guarantees (`TARGET_DECISIONS.md:318-327`). The checkers of
  `backend/packing.py` take the ragged form, the only one whose arrays still have widths; the workflow is check,
  pack, sample.

What survives packing is checked on the packed form by the engine, on every call: `d`, `N` against the frame's, the
sample stack `W`, the stacks `K` and `C`, and the shapes of `y` and the weights. The jet sizes of a packed sample are
not checked beyond their count (as on the ragged layer today, `sampling_variables.py:1710-1711`); the engine gains one
check, that every jet axis has at least one slot.

### 1.3 G2: liberal in, canonical out

On the uniform layer the padding of an input is junk, and the padding of every output is exactly zero. Scope
(ruled): **sampling and what is downstream of it.** "Every public function returns zero padding whenever its stated
preconditions hold" is the target for that surface.

- "Liberal in" is the contract of the functions that **hold the masks** (the engine, the twins, the oracle on a
  uniform frame, `ut3_unscale_coordinates`): they neutralize junk.
- **Which junk.** The engine masks every input it reads **by selection** (`where(mask, x, 0)`): the cores, the
  directions, the sample, the data and the held variables alike. One mechanism, which never does arithmetic on the
  junk (ruled, 2026-11-25). So inside sampling an input's padding may hold anything, `inf` and `nan` included. The
  contract a user is given is narrower for cores: **the padding of data may hold anything; the padding of cores must
  be finite**, because the rest of the uniform library masks by a product (`0 * inf` is `nan`). The reason data get
  the stronger promise: users compute them on packed arrays, and the commonest computation, a relative weight
  `1 / abs(y)`, writes `inf` into the padding (the critic's E4: `f = nan` today).
- A function on raw arrays with no masks states **zero padding in** as its precondition and keeps zero padding
  (`probe_data_metric`, `affine_jet`).
- Every packer returns a canonical object.

### 1.4 G3: one mathematical operation per function, one return type

- A **switch** (an argument choosing the operation or the output type) is split into functions; an **index**
  (choosing one of a family of like terms) is data.
- **The criterion for a `kind` argument: a function is split when its computation, or the structural check of its
  arguments, depends on the kind** (on where the mode axis or the jet axes sit). The first half is ruled; the second
  was added in the review (§13.3): a function that can compute without the kind but cannot check its arguments
  without it turns a wrong argument into a silent misread on a coincidence of sizes.
- The same operation over the two forms is **parametric**: one function, annotated with a constrained `TypeVar` shared
  by the co-varying parameters and the return. Mixed forms are a structural error. A branch with an extra input is a
  different function.
- **The shape of a split, where the kinds differ only in where the axes sit:** one private array-level core taking
  the offset of the leading axes; the apply function calls it at offset 0; the probe function maps it over a tuple.
  Where the mathematics differs by kind (`rescale_coordinates`: probe data are divided by the scales, apply data are
  not), the two functions are simply two functions.

### 1.5 The engine's status

**`sampling_variables` is exempt from G3 and from nothing else** (ruled); `computed_names`, the query that shares its
signature, with it. `kind`, the call type (a frame or a corewise point), the layer and `want` are its data. It
enforces no mixing for all of sampling (it is the lowest function that holds the object) and implements G2. What it
reads from structure: the layer, from the frame's own container (`frame_form`, `sampling_variables.py:1643`); frame or
point, from the number of stores; the stacks `W`, `K`, `C`, from the operands' ranks given the explicit `order`; the
route, from which optional arguments are present.

The exemption does not leak: every other public sampling function is a **typed face**, one engine call with `kind`,
the call type, **the layer** and a literal `want` fixed. Fixing the layer is new: today `probing.t3_probe` runs on a
uniform point and returns the packed form (the critic's E6), because the engine reads the layer. Each ragged function
(`t3_*`, `tv_*`) and each uniform twin (`ut3_*`, `utv_*`) checks its object's layer (`frame_form(x)[0]`) and raises
on the other, naming its twin. The other sanctioned generic pieces: the oracle, the batch, `pack_batch`, the geometry
protocol.

### 1.6 The per-mode loops

**On the uniform layer a per-mode object is one array, so anything that is a map over the modes is one array
operation with the mode axis leading; a Python loop over the `d` slices of a packed array does not appear in library
code** (ruled). The ragged layer loops over the modes, and so do the conversions between the forms (`unpack_vectors`,
`unpack_index`, `PackedVectors.unpack`), which is what they are. The rule covers reading shapes: on the uniform layout the engine
reads its sizes from the supercores' own shapes, not by iterating over their slices
(`sampling_variables.py:1020-1022`, `:1070-1091`, `:1094-1169` do today). The dense reference `sampling_definitions`
is exempt (ragged only, by definition). Not covered: the engine's per-site steps on uniform, one site at a time
through `xmap`, a memory trade-off that belongs to Part 2 (O-19, `TARGET_DECISIONS.md:354-357`).

---

## 2. The weights

### 2.1 Why they change

The M arc gave every data-space object "the type of `z`, axis by axis either 1 or `z`'s size" (Q5 and Q8,
`TARGET_DECISIONS.md:869-873`; `DATA_METRIC_PLAN.md` §2). For the weight, that one rule met every seam differently:

- packing: an edge of 1 has no padding, an edge of `N_i` has; hence the mixed-edge refusal, the separate packing rule
  of `_pack_data_space` (`ut3_operations.py:211`: join the axes, never zero-pad an edge of 1), and no answer to "what
  widths does a packed edge-1 weight carry";
- masking: only when the edge is `N` (`sampling_variables.py:1914`);
- `take`: a weight with no `W` axis had to be recognized (M-arc decision 18);
- the mode axis: `blocks_to_weight` (`jets.py:767`) exists only to turn a table of numbers into arrays with trailing
  size-1 axes;
- everywhere else in the library a size-1 axis that broadcasts is a silent error; here it was the meaning.

The maintainer (2026-11-24): the weights are "a constant sticking point … This suggests that perhaps the design of
the weights is not right."

### 2.2 The design

The measurement space is a direct sum of **blocks**, one per mode and jet slot (probe), per jet slot (apply), or one
(entries). The library already treats the block as first class: the display's error table, `data_block_norms`,
`equalizing_weight`, the `k!` between derivative and Taylor coordinates, and the `omega[mode, order]` matrix released
in 2026.5.0.

**Two named weights. They multiply:**

    M = gamma * block_weight**2 * full_weight**2        (gamma the minibatch scale; None is ones, or 1)

| | `block_weight` | `full_weight` |
|---|---|---|
| what it is | one number per block | one number per measured number |
| shape, as data of a fit | probe `(d,)+F`; apply `F`; entries `()`: exact | `z`'s shape; a stack axis (`W` or `C`) may be 1 |
| the same object on both layers | yes: it has no edge | no: a tuple, or packed (probe); one array (apply, entries) |
| packed, masked | never | like `y` |
| drawn by `take` | never (constant over the samples) | like `y`; not at all if its `W` axes are 1 |
| applied | left-aligned on the leading axes | elementwise |

- **Left-aligned** means: the block weight's axes are the leading axes of what it weighs: `(d,)+F` of a packed probe
  `r`, `F` of an apply `r`; against a tuple `r`, row `i` (of shape `F`) against `r[i]`. `K` sits after `W`, behind
  those axes, so a block weight needs no placement against a stack of directions.
- **A size of 1 is allowed exactly on stack axes, never on structural ones** (ruled, 2026-11-25). The stack axes
  (`W`, the samples; `C`, the base points) are collections of independent instances, and a size of 1 there means "the
  same for every instance", as broadcasting over a batch does everywhere in the library. The structural axes (the
  mode, the jet slots, the edge) are the parts of one measurement; every seam of §2.1 was on one of them. So a
  **full weight** has `z`'s shape, with each `W` and `C` axis 1 or full; the typical one, independent of the samples
  and the stack (the maintainer's "typical scenario"), is `F + (1,)*len(W) + (1,)*len(C) + (N_i,)` per mode, at no
  cost in `W`. **`y` is exact on `W`** (only its `C` axes may be 1): a `y` that is the same for every sample is not a
  dataset, and refusing it catches one sample's data passed against many. Never `K`. The "edge-1 `y`" that M-arc
  decision 13(b) had to mask no longer exists. The engine and the checkers, which know the stacks, enforce this.
- **The modes of a probe object of the type of `z` (`y`, a full weight, `r`, `Jp`, `c`) agree in everything but the
  edge:** one pattern of size-1 stack axes for all `d` arrays of a full weight (and of `y`'s `C`). (Not the sample:
  the jet sizes of input jets may differ by mode, which is what the packer's `len_P` is for.) The one packer needs it (it stacks the modes), and one rule on both
  layers is simpler than a mix that is legal ragged and cannot be packed (the critic's E8a). Checked by the engine,
  the probe data checker, `pack_batch` and the frontend batch; the refusal names `np.broadcast_to`.
- **Equalization** is `block_weight = equalizing_weight(probe_data_block_norms(order, y))`, passed straight in.
- **Derivative coordinates:** `omega_t = k! omega_d` per slot, so the `k!` goes into the block weight only, once, also
  when a full weight was given in derivative coordinates (`(k! b_d) f_d`). With no weight, the Taylor block weight is
  the `k!` table ("default means weights of 1.0 in the natural coordinates of the object",
  `TARGET_DECISIONS.md:889-891`). A caution for the docs: the two weights multiply, so a user who computes a relative
  full weight on the Taylor data and also passes the returned `k!` table has both.
- **Not yet: `sample_weight`** (shape `W`). The way to extend is a named factor with one fixed shape when a
  structured case earns it; the full weight is the catch-all. The case that would earn it is the one use of `W` the
  maintainer envisions, a weight that reduces the impact of outlier samples: it varies along `W` and is constant along
  the slots and the edge, so on the probe kind it is a full weight at `y`'s memory until then.

**What still costs `y`'s memory** on the probe kind: a weight that varies along the samples and is constant along a
structural axis (the outlier weight above; the weight per sample and slot that T3Polynomial wants and reaches today by
scaling its directions, `T3POLYNOMIAL_JET_USE.md:140-153`). T3Polynomial's weight per coordinate of a mode, constant
over the samples, is free under the rule above. For apply and entries `z` is per sample already and nothing is lost.
The dry port (P18) measures the first.

### 2.3 The public metric

**`data_metric` is split by kind** (changed in the review, §13.3). With a block weight it is no longer elementwise:
the weight is applied on leading axes, and whether a given array is a legal block weight depends on the kind. A
kind-free function can only test that the block weight's shape is a prefix of `r`'s, and a per-slot vector `(n+1,)`
against a probe `r` with `d = n+1` then passes and is applied per mode (the critic's E3b; today the call raises by
rank; T3Polynomial meets exactly this coincidence).

```python
def probe_data_metric(
        order:            typ.Tuple[int, ...],       # the jet shape; F = tuple(n+1 for n in order)
        r:                PerSite,                   # Y*: len=d, elm_shape=F+W+[K]+C+(Ni,); or packed (d,)+F+W+[K]+C+(N,)
        block_weight:     typ.Optional[NDArray],     # shape=(d,)+F exactly; None = ones
        full_weight:      typ.Optional[PerSite],     # in r's form: r's rank, each axis 1 or r's size; None = ones
        minibatch_scale:  typ.Optional[typ.Any],     # gamma: a number or a 0-d array; None = 1
) -> PerSite:                                        # M r: Y, in r's form

def apply_data_metric(order, r, block_weight, full_weight, minibatch_scale) -> NDArray   # r: F+W+[K]+C; block: F
def entries_data_metric(r, block_weight, full_weight, minibatch_scale) -> NDArray        # r: W+[K]+C; block: ()
```

- Every argument is required (`None` written out is ones, or a scale of 1), as in the M arc.
- **The block weight is checked exactly** (the kind and `order` give its shape; `d` is read off `r`).
- **The full weight obeys the principle of §2.2 as far as the function can see: the structural axes exact, the
  axes between them 1 or `r`'s size.** `probe_data_metric` knows the mode (first; a tuple of exactly `d` arrays, or a
  packed first axis of exactly `d`), the slots `F` (from `order`) and the edge (last): those are exact. Of the axes
  between them (`W`, `K`, `C`) it is not told which is which, so each may be 1 or `r`'s size; that also admits the
  size-1 `K` axes the engine inserts with `_with_K` (`sampling_steps.py:926`, which stays, for the full weight).
  `apply_data_metric`: `F` exact, the rest 1 or full. This closes the limit `data_metric`'s doctest shows today
  (`jets.py:678-690`: a packed weight with a mode axis of 1 broadcast over the modes), and the metric and the batch
  agree except for what only the batch can know: `y`'s exactness on `W`, and never `K` in a batch.
- `probe_data_metric` is parametric over the two forms (it runs inside uniform loops); a packed `r` with a tuple
  full weight raises. `entries_data_metric` is `apply_data_metric` at `order=()`, kept as a face without an `order`
  (the entries kind has data and no jets; a user holding entries data should not have to know that).
- **`LeastSquaresBatch.data_metric(r)`** stays the batch's method, the call that cannot forget the scale of a drawn
  batch: the batch has its kind and its order (§4.3), so it calls the right one of the three.

### 2.4 What else changes in the code

- **The engine** (`sampling_variables.py:1678`): the argument `weight` becomes `block_weight` then `full_weight`, in
  `weight`'s place after `y`; `computed_names` follows (its arguments are an engine call's). The structural check is
  the rule of §2.2: `y` exact but for `C` axes of 1; the full weight exact but for `W` and `C` axes of 1; the modes
  of a probe object agreeing; the block weight exact. On the uniform
  layer `y` and the full weight are masked, always (they always have the edge `N`); the block weight never.
- **`LeastSquaresBatch`** (`backend/fitting.py:758`): its fields become `kind, order, sample, y, block_weight,
  full_weight, minibatch_scale, drawn_indices`, the first two static (§4.3). Every construction in the library and the
  tests is rewritten, with keywords after `y`, so that no old positional call lands its scale in a weight. **Its
  constructor makes the checks the oracle's makes today** (`backend/fitting.py:1057-1066`; O-44, O-46: the kind one of
  three, the order a tuple of nonnegative integers, the entries kind's order `()`), on the static fields only, so they
  are safe when jax rebuilds the object, and it normalizes `order` to Python ints. An old positional
  `LeastSquaresBatch(ww, y)` then fails at once, on its kind, and K1's design note decides whether the fields after `y` become keyword-only (which
  `dataclass(kw_only=True)` cannot do on the Python 3.9 floor).
- **`take`** (`backend/fitting.py:1190`, a method of the batch from K1): the block weight passes through; the full
  weight is gathered like `y`, or kept as it is when its `W` axes are 1 (today's branch). M-arc decision 18's raise
  (an object with no `W` axes at all) stays.
- **The oracle's two data sums** (`backend/fitting.py:870-936`): `<y, M y>` and `y_block_sumsq` as one product and
  one reduction on the array itself (§6); `y` and the full weight are masked by selection first; a `y` whose `C` axes
  are 1 is counted over `z`'s shape, as M-arc 13(c) decided. `_arrays_of` (`:840`) goes.
- **Removed in K1:** `blocks_to_weight`; `_pack_data_space` and `pack_data`, its one caller (`pack_batch` packs a
  probe `y` and full weight with `pack_vectors`); `check_weight` (a full weight is checked as data, by `check_data`
  until K3 and the new checkers after; a block weight's shape by the engine and by the batch's `check`).
  `jets.check_data` learns the exact shape of `y` in K1 and is replaced in K3: one
  function edited twice, so that the checker is never looser than the engine.
- **The display** (`backend/optimizer_display.py`): its table has the block shape, as today. It reads the jet shape
  off the batch (`optimizer_display.py:622` reads `getattr(oracle, 'order', None)` today, which on an oracle without
  fields would be `None` and print positional labels, silently: the critic's E8b), and `make_newton_display` refuses a
  validation batch whose kind or order differs from the batch's (their tables would set two quantities side by side).

**This reverses, by name:** Q5 and Q8 of the data-metric plan for weights and for `y`'s size-1 axes
(`TARGET_DECISIONS.md:869-873`; `DATA_METRIC_PLAN.md` §2, and §4's "Uniform packing"); Q1 and Q3's kind-free
`data_metric(r, weight, minibatch_scale)`; "the oracle is all static, the batch is all leaves" (8.3,
`TARGET_DECISIONS.md:2517`); M-arc decisions 7, 10, 13(d). It keeps: the typing
`J: X -> Y*`, `J^T: Y -> X*`, `M: Y* -> Y`; the transposes taking `c` with no weight inside; Taylor coordinates
everywhere; the weighted block sums; 13(a) to (c) in substance.

---

## 3. The backend

### 3.1 `backend/packing.py` (new)

Pure functions on plain data; imports `common` and the jet-shape reader of `contractions` only, so the engine, `jets`,
`data_preparation`, `fitting` and both frontends can import it, and no ragged module imports a uniform one. Its
module docstring states the forms (§1.1), the workflow, and what the packers guarantee.

```python
def form_of(                                       # the definition of 1.1, in one place (name open)
        value:  typ.Any,                           # a per-mode object
) -> str:                                          # 'ragged' (a tuple or a list) | 'packed' (an array); else TypeError

# --- packers and unpackers: no shape, no check beyond the input's own form; canonical out ---
def pack_vectors(
        vectors:  typ.Sequence[NDArray],           # len=d, elm_shape=F_in_i+lead+(Ni,); lead agrees across the modes
        N:        typ.Optional[int] = None,        # the packed width; None = max(Ni); N < max(Ni) raises
        len_P:    int = 0,                         # the number of leading jet axes (sizes may differ by mode)
) -> NDArray:                                      # (d,)+F_in+lead+(N,): jet axes zero-filled, the edge zero-padded

def unpack_vectors(
        packed:  NDArray,                          # (d,)+lead+(N,)
        mode_sizes:  typ.Sequence[int],            # the real sizes (N_0, ..., N_(d-1))
) -> typ.Tuple[NDArray, ...]:                      # len=d, elm_shape=lead+(Ni,)

def pack_index(index: typ.Sequence[NDArray]) -> NDArray:           # len=d, elm_shape=W, int -> (d,)+W
def unpack_index(index: NDArray) -> typ.Tuple[NDArray, ...]:       # (d,)+W -> len=d, elm_shape=W

# --- checkers: the ragged form against plain structural arguments ---
def check_vectors(vectors, order, shape) -> None          # the sample of probe and apply: d, each width, the jets, W
def check_index(index, shape) -> None                     # the sample of entries: d, one W, integer dtype
def check_probe_data(value, sample, order, stack_shape=()) -> None    # y, a full weight, a residual
def check_apply_data(value, sample, order, stack_shape=()) -> None
def check_entries_data(value, index, stack_shape=()) -> None
```

- **One packer** serves the sample and every probe object of the type of `z`: by the rule of §2.2 the arrays of
  such an object agree across the modes in everything but the edge, and a sample's in everything but the edge and
  its `len_P` jet axes. There are no identity packers for apply and entries data. **One
  unpacker** likewise.
- **The checkers take plain arguments, not a target** (ruled): a sample checker needs the mode sizes only; a data
  checker reads `W` and the `N_i` off the already-checked sample and needs only the stack `C`. A backend user holding
  a frame reads them with `frame_shape` (`sampling_variables.py:1556`) and **`frame_stack_shape`, made public** (today
  the private `_stack_of_frame`, `backend/fitting.py:858`; M-arc ruling 22). The ambient transposes have no object at
  all and check against the shape read off the vectors.
- They replace `jets.check_sample` and `check_data` (`jets.py:1424`, `:1568`) **in K3**, with the change of form
  (until then the ragged layer still takes a `(d,)+W` index array, which the new checkers refuse); `check_weight`
  (`:1657`) goes in K1, with the weight it checked.
  `LeastSquaresOracle.check_batch(batch, shape, stack_shape=())` (`backend/fitting.py:1137`) becomes the batch's
  `check(shape, stack_shape=())` (§4.3), composed of them, plus the exact shape of the block weight.
- **Moved here:** `pack_vectors`, `unpack_vectors` (from `ut3_operations.py:128`, `:179`). **Deleted, each with its
  last caller:** `_pack_data_space` and `pack_data` in K1 (`ut3_operations.py:211`, `uniform_fitting.py:398`);
  `is_packed`, `pack_if_ragged` and `pack_sample` in K5 (`ut3_operations.py:191`, `:200`, `uniform_fitting.py:375`).
- **`pack_batch(batch, N=None)`** (`uniform_fitting.py:431`; the kind and the order are the batch's) stays, a
  composite over a kind-generic container: `pack_vectors` for the sample of probe and apply and for a probe `y` and
  full weight, `pack_index` for the entries index, nothing for the rest. It refuses a batch whose members disagree in
  their mode sizes (the sample, `y` and the full weight of each mode: a requirement on the input's own form, like the
  agreement of `W`).

### 3.2 The engine

- **Uniform frame:** the sample, a probe `y`, a probe full weight, the directions, `Vc` and every held variable must
  be in the uniform form (one array each; the entries index one `(d,)+W` array). A sequence raises, naming the packer.
  The lines that pack today (`sampling_variables.py:1868-1873`) go.
- **Ragged frame:** sequences only, the entries index included. A single array raises, naming `unpack_vectors` or
  `tuple(index)`.
- **Masking (G2).** On entry, by selection (§1.3), everything the engine reads: the cores and the directions (the
  maintainer: "zero the cores … one consistent choice, not an optimization"), the sample, `y`, the full weight and
  the held variables. One helper (`_masked`, `sampling_variables.py:1442`), one behaviour, for every input the
  engine reads. (The start values of the sweeps are not inputs: the engine makes them, ones on the real slots of
  their end bond, by a product, `:1368-1370`; unchanged.)
  - *The sample.* The critic removed its mask in today's engine and ran 1,536 calls with finite junk in its padding
    (every public name, both kinds with jets and both call types, `K`, `Vc`, both values of `sum_over_samples`): no
    real entry, held output or scalar differed by a bit (E1). So for finite junk the zeroed cores suffice. The mask is
    kept, by selection, so that the sample has the promise the other data have; it costs one select of the sample's
    size, about `1/n` of the contraction that reads it.
  - *The held variables* need theirs: the apply kind closes its end bond by a sum that no core touches (`close_step`).
  - *On exit:* every output, **the `'w'` echo included** (today it is returned as given,
    `sampling_variables.py:1932-1935`).
  - *Checked for hidden costs (2026-11-25; the notes' §7):* `where` keeps the shape and needs no gather, so nothing of
    the old trouble with index lists applies, and it would work with traced masks as the product does; the cost is of
    the same order eager and under jit; the gradient through it is exactly zero on the padding. Cares for the
    implementation: the zero has the array's dtype (on numpy 2 a `np.float64(0)` promotes a float32 array); `where`,
    never boolean indexing; comparisons with the tree before are on values, not bytes (signed zeros).
- **Not built here:** the option to skip the entry masking. It moves to Part 2 (an execution option; it may become
  part of the Plan object). This sub-arc delivers what makes it safe: canonical outputs everywhere.

### 3.3 The uniform twins, and the object-free helpers

- **`ut3_sampling`, `utv_sampling`:** the `Vectors` union (`ut3_sampling.py:71`) goes; a per-mode argument is one
  packed `NDArray`; a probe-type return is packed. Each twin checks its object's layer (§1.5) and is one engine call
  (today `ut3_probe`, `ut3_probe_corewise_tangent` unpack on the way out, `ut3_sampling.py:107-109`, `:229-232`; the
  transposes pack `c`, `:250`).
- **Parametric over the two forms** (they are needed on packed arrays, and they keep zero padding): `affine_jet`
  (as today) and `probe_data_metric`.
- **Ragged only, refusing a single array and naming `unpack_vectors`** (changed in the review, §13.4): every other
  probe helper of `jets` and `data_preparation` (§5); and `index_to_one_hot(index, shape)`, which takes the index as a
  tuple from K3 and names `tuple(index)` in its refusal. The helpers renamed in K1 refuse an array from K1, the
  others from K6. They prepare data, which is done ragged and packed last. Two of them divide by a per-mode quantity (`probe_rescale_coordinates` by the
  scales, `probe_dual_jet_to_jet` by the squared weights), which on a packed array is `0 / 0` in the padding and
  cannot be guarded without the widths (the critic's E3a: `f = nan`).
- **`ut3_unscale_coordinates`** (`data_preparation.py:566`) takes packed scales only, masked by selection with the
  shape it holds.
- **One `TypeVar` in `common.py`, `PerSite`** (a tuple of per-site or per-mode arrays, or one array with that axis
  leading). Annotated with it in this sub-arc: `xcat`, `xappend`, `xprepend` (whose empty-sequence branches return
  tuples), `affine_jet`, `probe_data_metric`. The `DataSpace` of the notes is not needed once the metric is split.

---

## 4. The frontend

### 4.1 `PackedVectors`

"Padding does not encode widths", the reason a uniform T3 carries `shape` beside its supercores. So the uniform
frontend's per-mode object is the array together with the widths it was packed from:

```python
@dataclass(frozen=True, eq=False)                  # in t3toolbox/uniform_tucker_tensor_train.py; exported at the top level
class PackedVectors(common.ExplicitEquality):
    array:   NDArray                               # (d,)+lead+(N,), ndim >= 2; a jax pytree leaf
    mode_sizes:  Tuple[int, ...]                   # the sizes it was packed from, (N_0, ..., N_(d-1)); static (aux)

    @classmethod
    def pack(
            cls,
            vectors:  Sequence[NDArray],           # len=d, elm_shape=F_in_i+lead+(Ni,)
            N:        Optional[int] = None,        # the packed width; None = max(Ni)
            order:    Tuple[int, ...] = (),        # the jet shape: len(order) leading jet axes may differ by mode
    ) -> 'PackedVectors':

    def unpack(self) -> Tuple[NDArray, ...]:       # len=d, elm_shape=lead+(Ni,)
```

How it was arrived at (the planning session): packing bound to the tensor was rejected (users start with data and
build the tensor in the fit; the mode sizes are the problem's, read off the data; a tensor's ranks and values are
irrelevant to packing); so the widths travel with the packed object.

- **The field is `mode_sizes`** (ruled, 2026-11-25). Not `shape`: numpy reads an object's `.shape` attribute as its
  array shape (the critic's E5 got a silently wrong sample count from exactly this), and beside stacking the word is
  ambiguous. Not `tensor_shape`: the tensor may not exist yet. "Mode sizes" is the library's own phrase for this
  tuple (`frame_shape` returns "the mode sizes"). Normalized to a tuple of Python ints at construction.
- **A `PackedVectors` is every per-mode object of the uniform frontend:** the sample of probe and apply; probe-type
  outputs (`z`, `Jp`, a local model's residual); `c` of the probe transposes and Hessians; a batch's sample and its
  probe `y` and full weight (§4.3); the scales of `unscale_coordinates`. Apply and entries data, the entries index
  and a block weight are bare arrays.
- **Each uniform frontend method** raises a `TypeError` for anything but a `PackedVectors` where one is due (naming
  `PackedVectors.pack`), raises a `ValueError` unless its `mode_sizes` equal the object's `shape` and its `N` the
  object's `N`, passes `.array` to the backend, and wraps a probe-type result with its own `shape`. Two static tuples are
  compared, so the check runs under jit.
- **No arithmetic on the class.** Compute on `.array`; re-wrap with `dataclasses.replace(z, array=new)`, which keeps
  the mode sizes.
- **The numerical helpers stay on plain data** in both tiers.
- **Padding is not validated at construction** (liberal in, §1.3). Every `PackedVectors` the library
  returns is canonical.
- **Structural validation at construction:** `array.ndim >= 2`; `array.shape[0] == len(mode_sizes)`; `0 <= Ni <= N`. As
  for the library's other classes it runs when jax rebuilds the object, so `jax.vmap` over a `PackedVectors` is not
  supported (its first axis is the modes; the sample stack `W` is the batch), as for a stacked uniform T3.

**A custom loss, end to end** (the workflow the two rules above give):

```python
ww_p = PackedVectors.pack(ww)                          # once
y_p  = PackedVectors.pack(y)
z    = ux.probe(ww_p)                                  # PackedVectors; mode sizes checked against ux.shape
c    = dataclasses.replace(z, array=loss_gradient(z.array, y_p.array))    # any array computation
g    = ux.probe_corewise_transpose(ww_p, c)            # the gradient blocks
```

### 4.2 What goes

- **No public `check_*` methods** on any of the four classes: the ragged methods check their sample on every call, as
  today (through `packing.check_vectors` and `check_index`); the uniform methods compare the mode sizes.
- `UniformTuckerTensorTrain._check_sample` and its branch on the form (`uniform_tucker_tensor_train.py:53-61`);
  `tucker_tensor_train._as_index` (`tucker_tensor_train.py:63`), which turns `(1, 2, 3)` and `[9, 4, 7]` into one
  array: the ragged methods stop calling it in K3 (a ragged index is a sequence, each member converted with
  `asarray` by the engine, so a list of integers and a list of lists both work), and the function is deleted in K4
  with its last caller, the uniform frontend (`uniform_tucker_tensor_train.py:420`).
- **Refusals:** the ragged frontend given a `PackedVectors` or a single array; the uniform frontend given a tuple or a
  bare array. `ux.entries((9, 4, 7))` raises, naming `pack_index`.

### 4.3 Fitting: the batch

**Ruled in the maintainer's review (2026-11-25), replacing revision 3's design** (a batch class shared by both tiers,
with a static field for its mode sizes). Two changes.

**The batch knows its kind** (both tiers). `kind` and `order` move from the oracle into the batch, as static fields:
the kind is a property of the measurements, not of the loss. The operations that need them are then the batch's own
methods, where today they are the oracle's and take a batch: `data_metric(r)`, `take(indices)`, `n_measurements`,
`check(shape, stack_shape=())` (and, on the frontend class, `pack(N=None)`; below). `LeastSquaresOracle` keeps the objective's rule, `query`, and has no
fields. A pytree's static fields are part of the compilation key and are constant across the draws of a fit, so the
reason for the old split holds.

**The frontend has its own batch class, whose per-mode members are `PackedVectors`** (chosen by the maintainer over
the shared class after seeing both as code). The frontend's `LeastSquaresBatch` holds frontend objects and hands the
backend its `.data`, a backend `LeastSquaresBatch` of plain arrays, as every frontend class does (and as the frontend
`LocalModel` wraps the backend's of the same name). This reverses "the oracle and the batch are one class each for
both tiers" (`TARGET_DECISIONS.md:344-346`) for the batch; the oracle stays one class.

```python
ragged = LeastSquaresBatch('probe', (), ww, y, block_weight=bw)     # ww, y: lists of d arrays
batch  = ragged.pack()                                              # once; N=None is the largest mode size

x, stats = t3.newton_cg(t3.UNIFORM_MANIFOLD, LeastSquaresOracle(), batch, x0)
sub    = batch.take(np.arange(64))                                  # a drawn batch, of the same class; what a draw returns

z = x.probe(batch.sample)                                           # batch.sample is a PackedVectors
g = x.probe_corewise_transpose(batch.sample,
        batch.data_metric(dataclasses.replace(z, array=z.array - batch.y.array)))
```

- **Its members.** On the ragged layer, tuples, as today. Packed: a `PackedVectors` for the sample of probe and apply
  and for a probe `y` and full weight; bare arrays for apply and entries data, the entries index and the block weight.
- **Validated at construction** (structural, always, as the library's other frontend classes are): the kind and the
  order (the backend constructor's checks, §2.4); the per-mode members in one form; packed members agreeing in their
  mode sizes and in `N`. So a packed batch built by hand is checked: each member states its sizes
  (`PackedVectors(y_arr, (4, 5, 6))`), and they must agree.
- **A registered pytree:** its members are the children, the kind and the order static. Validation runs when jax
  rebuilds it, as for the library's other classes.
- **Its methods are thin, over the backend's on `.data`:** `pack(N=None)` refuses ragged members that disagree in
  their mode sizes and returns the packed batch (no tensor is needed; on a batch that is already packed it raises);
  `take(indices)`; `data_metric(r)`, for the probe kind a `PackedVectors` in and out, whose mode sizes it compares
  with the batch's; `check(shape, stack_shape=())`; and the property `n_measurements`.
- **In the backend** the batch class has `check`, `take`, `data_metric` and `n_measurements`; packing is the function
  `uniform_fitting.pack_batch(batch, N=None)` (`uniform_fitting` imports `backend/fitting`, not the reverse), which the
  frontend's `pack` calls.
- **One kind and one order per fit.** Each batch is read under its own, so the frontend entries refuse a validation
  batch whose kind or order differs from the batch's, and the hand-over of a drawn batch compares them too (two
  tuples per draw). In the backend the display refuses (§2.4); the optimizers do not look.
- **A frontend entry takes the frontend batch.** Given the backend's (a packed one built by hand would otherwise reach
  the engine with no check of its mode sizes), it raises, and since the two classes share a name the message and each
  class's `repr` name the module (`t3toolbox.fitting.LeastSquaresBatch`, not
  `t3toolbox.backend.fitting.LeastSquaresBatch`) and say how to build the one from the other. A user's draw that
  returns the backend's is refused the same way. **The refusal starts in K4:** until then the frontend's name is the
  backend class itself. On the ragged layer the two hold the same tuples, so there it is type discipline, not a
  numerical guard.
- **`prepare_batch`** (`t3toolbox/fitting.py:589`), where a batch enters the optimizers and `local_model`: for a
  uniform point the batch must be packed (a ragged one raises, naming `pack`), its mode sizes must equal the point's
  `shape` and its `N` the point's; the backend receives `.data`. The rest (ranks, `W`, the jets, `C`) the engine
  checks on the packed form. `_is_packed` (`:582`) goes.
- **`optimizers._frontend_draw`** (`t3toolbox/optimizers.py:117-122`) no longer packs: it hands over the `.data` of
  each batch a user's draw returns (an attribute read; the draw returns batches in `x0`'s layer).
- **The frontend `LocalModel`'s** residual and Jacobian product of the probe kind on a uniform point are
  `PackedVectors`. Its `batch` is the frontend batch, derived on request from the backend model's batch and the
  frame's shape, as everything else it shows is derived: the class still holds `(geometry, backend)` and its pytree
  does not change.
- **User-written oracles, and the `Oracle` protocol** (`backend/optimizers.py:181-231`). A frontend
  `LeastSquaresBatch` is handed to the backend as its `.data` whatever the oracle, so an oracle that reuses the
  library's batch receives the backend class and reads the kind and the order off it (writing an oracle is
  backend-level work, `TARGET_DECISIONS.md:349`). A batch of any other type passes through unread, as today. Two
  contracts of the protocol change, by name: its optional members `check_batch`, `n_measurements` and `take` (O-29)
  are the batch's, and "its data are its batch, which the optimizers pass through unread" gains one exception, **the
  default draw** (`flat_draw`, `_require_draw`, `backend/optimizers.py:625-654`), which calls `take` and
  `n_measurements` on the batch. An oracle with its own batch type gives its batch those two, or passes `draw=`.
- **The backend batch** holds arrays and the two static fields, nothing about mode sizes: the backend gives
  checkers, and its `pack_batch` refuses members that disagree (§3.1).
- **`aligned_padded_frame`** (the geometry protocol's member) always returns four elements, `seconds=None` unless
  `timing`; `optimizers._aligned_padded_frame`'s signature inspection goes.

### 4.4 The translation of the 2026.5.0 call form, slice by slice

It lives until P17 (`t3toolbox/optimizers.py:132-229`), and the old-form tests pin it (`TestTheTranslation`, 19
pre-P16 numbers). It is the one frontend entry that packs for the user, since the old form had no packed batch.

| Slice | What the translation gains |
|---|---|
| K1 | the old `omega` matrix is passed as `block_weight` (today through `blocks_to_weight`, `:196`, `:216`); `prepare_derivative_data` under its new names |
| K1 | it builds its batch with the kind and the order it reads off the old call, and calls `n_measurements` on the batch (`t3toolbox/optimizers.py:249`) |
| K3 | an old entries sample, a `(d,)+W` array, is turned into the tuple (`tuple(index)`) before `index_to_one_hot` and the engine |
| K4 | for a uniform `x0` it packs the batch itself with `batch.pack(N=x0.N)` (the default is the largest mode size), the old form's validation data likewise (`val_sample`, `val_data`, `:257-259`), and wraps an old draw (`_old_draw`) to pack each batch it returns (what `_frontend_draw` does today) |

It calls none of the helpers K6 renames.

---

## 5. The split of the `kind`-switched helpers

Named kind first (ruled). Today's signatures, verbatim: `packing_audit/K_MEASUREMENTS.md`, M2(d).

| Today | After | Slice | Why |
|---|---|---|---|
| `data_metric(r, weight, minibatch_scale)` | `probe_data_metric`, `apply_data_metric`, `entries_data_metric`; `LeastSquaresBatch.data_metric(r)` stays, calling the one of its kind | K1 | §2.3 |
| `LeastSquaresOracle(kind, order)` with `check_batch`, `take`, `n_measurements` | `LeastSquaresOracle()`; `LeastSquaresBatch(kind, order, ...)` with `check`, `take`, `n_measurements` | K1 | §4.3 |
| `jet_to_dual_jet(kind, order, jet, weight)` | `probe_jet_to_dual_jet(order, jet, block_weight, full_weight)`, `apply_jet_to_dual_jet(...)` | K1 | the flip acts on the jet axes |
| `dual_jet_to_jet(kind, order, dual_jet, weight)` | `probe_dual_jet_to_jet`, `apply_dual_jet_to_jet` | K1 | likewise |
| `derivative_weight_to_taylor_weight(kind, order, weight)` | `derivative_block_weight_to_taylor(order, block_weight)` | K1 | one function: the `k!` multiplies the table's trailing `F` axes, the same for `(d,)+F` and `F` |
| `prepare_derivative_data(kind, order, y, weight, validation)` | `prepare_probe_derivative_data(order, y, block_weight=None, validation=None)`, `prepare_apply_derivative_data` | K1 | a full weight is not converted (§2.2) |
| `blocks_to_weight(kind, order, blocks, y)` | removed | K1 | §2 |
| `check_sample`, `check_data` | the five checkers of §3.1 (added in K2, replacing in K3) | K2, K3 | |
| `check_weight` | removed | K1 | §2.4 |
| `pack_sample`, `pack_data` | `pack_vectors`, `pack_index` | K1 (`pack_data`), K5 (`pack_sample`) | |
| `derivatives_to_taylor(kind, order, jets)` | `probe_derivatives_to_taylor(order, jets)`, `apply_derivatives_to_taylor` | K6 | the `k!` acts on the jet axes |
| `taylor_to_derivatives(kind, order, jets)` | `probe_taylor_to_derivatives`, `apply_taylor_to_derivatives` | K6 | likewise |
| `data_block_norms(kind, order, y)` (`data_preparation.py:130`) | `probe_data_block_norms(order, y)`, `apply_data_block_norms(order, y)`, `entries_data_block_norms(y)` | K6 | which axes are reduced |
| `rescale_direction(kind, order, sample, y, length)` (`data_preparation.py:279`; probe and apply only) | `probe_rescale_direction`, `apply_rescale_direction` | K6 | where the slots of `y` sit |
| `rescale_coordinates(kind, order, sample, y, scales)` (`data_preparation.py:391`; probe and apply only), and the frontend's `fitting.rescale_coordinates` (`t3toolbox/fitting.py:110`) | `probe_rescale_coordinates`, `apply_rescale_coordinates`, in both tiers | K6 | different mathematics: probe data are divided by the scales, apply data are unchanged |
| `equalizing_weight(block_norms, floor)`, `affine_jet` | one function each, as today | | their computation and their checks do not depend on the kind |
| `index_to_one_hot(index, shape)`, the index a `(d,)+W` array | the same name, the index a tuple; ragged only | K3 | §3.3 |

- **The weight-carrying helpers change once, in K1** (their arguments and their names together); K6 has the rest.
  What is still touched twice: `jets.check_data` (§2.4); `jets.py`'s module doctests (`:67-152`) and
  `tests/test_jets.py`'s `TestDerivativeData`, which use K1's functions and K6's side by side.
- **Forms:** `probe_data_metric` and `affine_jet` are parametric; every other probe helper is ragged only (§3.3). This
  settles the investigation's F7 (today a packed probe array comes back as a tuple of padded slices) by refusal.
- **Entries.** The jet functions have no entries version (the entries kind has no jets). Where the entries kind has
  data (`entries_data_metric`, `entries_data_block_norms`), it has a face without an `order`, which is the apply
  function at `order=()`.
- Whether `jet_to_dual_jet` and `dual_jet_to_jet` stay public in the backend is the audit's decision 4, pending; they
  are split as they are.
- **The count.** Sixteen `kind`-dependent public helpers today (`data_metric`, kind-free now, and the fifteen that take
  `kind`: ten in `jets`, three in `data_preparation`, two in `uniform_fitting`) become twenty-eight functions: the
  three metrics; eighteen from the other rows of the split; five checkers; two packers (with their unpackers).
  `blocks_to_weight` and `check_weight` have no successor.

---

## 6. What the rules remove (the per-mode loops, §1.6)

| Site | Today | After |
|---|---|---|
| The engine's data-space steps: `_modes` in `_weighted_sum`, `w_tilde_step`, the block sums (`sampling_steps.py:915-1099`) | a loop over the modes, then `stack` | one application of the metric, one flip, one reduction on the packed array |
| The entries kind's one-hot vectors on uniform (`w_one_hot_step`, `sampling_steps.py:847`) | built per mode, stacked | one comparison of the index against `arange(N)` |
| The engine's shape reads on uniform (`sampling_variables.py:1020-1022`, `:1070-1091`, `:1094-1169`) | a loop over the slices of each supercore | the supercores' own shapes |
| The oracle's two data sums (`backend/fitting.py:840-936`) | unpack to real widths, loop, restack | one product, one reduction |

The helpers of `jets` and `data_preparation` loop over the modes of a tuple, as the ragged layer must; their packed
branches, where the loops over slices were (`jets._arrays`, `jets.py:225`), go with §3.3.

**The gate that pins it.** The instruction count of the **optimized** program
(`jax.jit(f).lower(*args).compile().as_text()`, on CPU) is the same at `d = 3` and `d = 5`, for the uniform engine and
for the oracle's query, over a fixed list of wanted names that covers every row above. The measure matters: at the
level of the traced program the count grows with `d` on every path today, because of the shape reads, which XLA then
removes; the optimized program is what a Python loop over the modes actually enlarges. The critic measured it (E2c).
Today the count is constant in `d` on the paths that have no data-space step (63, 412 and 576 instructions on three
such paths), so the per-site steps pass; it grows for the probe kind's `f` (438, 452, 491 at `d = 3, 5, 8`) and
`f, g` (1414, 1436, 1487), which are the data-space loops of the first row. `d = 1` and `2` are poor test points (XLA
simplifies a loop of one trip away).

---

## 7. The slices

One row per slice; the detail of each (exact signatures, the tests and their oracles, the expected-red lists, the
batteries) is its design note, reviewed before the build, as in the M arc. Process: `implementation/BRIEF.md`,
`ORCHESTRATION.md`; large slices in reviewed parts on a local side branch, squashed (a side branch may be wrong between
parts if the squash is right: the maintainer, 2026-11-24).

| Slice | What changes | Tested against | Size |
|---|---|---|---|
| **K0** | Delete the old kernels, the old kinds and the bridge (today's P20) | the suite that remains, its lost oracles re-based | medium |
| **K1** | The weights: `block_weight`, `full_weight`; `y` at exact shape; the metric split by kind; the kind and the order into the batch, with its methods; the oracle's sums; the weight-carrying helpers under their new names | the dense reference, changed first; a battery against the tree before K1 | large |
| **K2** | `backend/packing.py` (additive); `frame_stack_shape` public; `PerSite` | its own tests | medium |
| **K3** | The ragged layer takes sequences only (the sample, the data, the entries index); the ragged functions refuse a uniform object; the new checkers replace the old | the same numbers as before K3 on converted inputs | medium |
| **K4** | The frontend: `PackedVectors`; the uniform frontend methods; the frontend batch; `prepare_batch`; no automatic packing | the backend through `.array`; the refusals | large |
| **K5** | The uniform backend takes packed only: the engine, the twins (no mirror; the layer fixed); G2's masking; the data-space steps and the shape reads as single operations; the program-size gate | ragged, per element; junk in every padding; exact zero padding out | large |
| **K6** | The remaining `kind`-switched helpers, ragged only; `aligned_padded_frame` | each new function against the function it replaces | small to medium |
| **K7** | The guards: hostile fixtures by default; the junk-in coverage meta-test | itself | small |

**Order.** K0 first, so that no rule needs an exemption list and no doomed code is edited. The weights before the
forms, because they remove the data-packing rule: K2 to K5 never port it. The module is additive and lands before
anything is refused. Ragged before uniform: the checkers and the entries index are ragged-side. **The frontend before
the uniform backend:** in K4 the frontend starts handing the backend packed arrays, which the backend already
accepts; only then does K5 make the backend refuse the ragged form, and what it breaks are backend tests. (In the
first draft the two were the other way round, and K4 would have turned every uniform frontend sampling method red:
they do not pack, they pass a ragged sample to the twin, `uniform_tucker_tensor_train.py:484-485` and its siblings.)
The guards last, on the final surface.

### K0: the deletion

What P20 deletes (`PART1_PLAN.md` §10, `:2609-2660`): `backend/sampling_derivatives.py`; the removed names of
`probing`, `apply`, `entries`, `ut3_sampling`, `utv_sampling`; the old kinds and their factories in
`backend/fitting.py` and `backend/uniform_fitting.py`; the tests of the old kernels and the bridge. The plan's table
(`PART1_PLAN.md:2234-2252`) has every old piece losing its last library caller at P16.

- **The inventory, measured at `3375d213`** (`K_MEASUREMENTS.md`, M1; an `ast` scan with `implementation/inventory.py`'s
  name lists): 74 removed names in the five sampling modules, 33 in `backend/fitting.py`, 17 in
  `backend/uniform_fitting.py`, and `sampling_derivatives.py` whole (3,534 lines). From library code that stays, the
  only code references are import lines (`apply.py:23`, `entries.py:26`; `import sampling_derivatives` at
  `backend/fitting.py:34`, `ut3_sampling.py:26`, `utv_sampling.py:35`) and two private helpers of `apply.py`
  (`_apply_transpose_adjoint`, `:617`; `_apply_corewise_hessian_adjoint`, `:663`) whose only callers are functions
  that are deleted (`apply.py:313`, `:427`; `entries.py:265`, `:375`). **The examples: no reference. The pages: 16
  pages, 70 lines, prose and four fenced lines that are not doctests** (P19's).
- **The orphaned private helpers go too, but not by M1(a2)'s list as it stands:** that scan reads top-level
  definitions only, and misses a reference inside a module-level `if`. It lists `_BATCH_FIELDS` as unreferenced, which
  `LeastSquaresBatch`'s pytree registration uses (`backend/fitting.py:809-815`). K0 re-runs it reading every
  module-level statement, and its gate imports every module with and without jax.
- **The tests:** 21 test files refer to the old code. Four wholly (`tests/bridge/test_old_kernels_bridge.py`,
  `tests/backend/test_from_sweep_reuse.py`, `tests/test_chunk_size_estimator.py`, `tests/test_jet_recurrence.py`), and
  `tests/test_probe_derivatives.py` in 16 of its 17 tests. The rest in part, counting the tests that reach the old
  code through a helper as well as directly: `tests/backend/test_curvature.py` 29 of 96,
  `tests/backend/test_uniform_fitting.py` 11 of 30, `tests/backend/test_uniform_curvature.py` 10 of 56,
  `tests/test_dispatch.py` 11 of 108; 16 test files import `sampling_derivatives`, some at module level, where the
  import fails the whole file. **Where a test that stays uses an old kernel as its oracle**
  (`tests/backend/_curvature_oracles.py`, `tests/oracle_sweeps/o1_*.py`), K0's design note re-bases it on an oracle
  that does not share code with the engine (the dense reference, a difference quotient, jax AD) or deletes it with a
  stated reason: no test is left comparing the engine with itself.
- **Not in K0:** the check-in on `t3_corewise_frame` and `ut3_corewise_frame` stays after P19 (a frontend helper, not
  a kernel); the translation of the old call form stays until P17.
- **Gate:** the full gate, the slow tier serially, both CI legs; and **no code reference** to a deleted name in
  `t3toolbox/`, `tests/` or `examples/` (the inventory, re-run on the slice's tree). Not P20's grep as written
  (`PART1_PLAN.md:2657-2660`), which cannot pass yet: the pages' prose mentions are P19's, and `chunk_size` in the
  translation's old signatures goes with it in P17 (`sum_over_probes` is in deleted code only, and goes here). P17 and
  P19 each end with the part of that grep that is theirs.

### K1: the weights

One change, like M2, built in reviewed parts on a side branch. **Part 0 is the move of the kind and the order into
the batch, with its methods**: a refactor with a bitwise battery (no number changes), reviewed apart from the
numerical parts. Its sites beside `backend/fitting.py`: the display (`optimizer_display.py:622`); the default draw
(`flat_draw`, `_require_draw`, `backend/optimizers.py:625-654`); the `Oracle` protocol's text (`:181-231`); the
translation (`t3toolbox/optimizers.py:249`); the frontend's `_is_packed` and `prepare_batch` (`t3toolbox/fitting.py:585`,
`:608`) and the parameter comments of `mc_sgd` and `adam` (`t3toolbox/optimizers.py:331`, `:354`); about 104
`LeastSquaresOracle(` and 76 `LeastSquaresBatch(` constructions in the tests, the examples, the pages and the
doctests. Then, bottom-up: (1) the dense reference
`sampling_definitions` and the test oracles read the two weights (so their errors cannot correlate with the
engine's); (2) the engine and its steps; (3) `jets`: the three metrics, the dual-jet pair, the derivative functions
under their new names, `blocks_to_weight` removed; (4) the backend fitting layer (the batch's fields, with the kind
and the order, and its methods; the oracle's sums; `pack_batch`); (5) the frontend, the translation (§4.4), the
display's tests, the examples and pages with doctests that pass weights.

- **What breaks** (`K_MEASUREMENTS.md`, M2; `ast` call counts at `3375d213`, to be re-run after K0):
  `blocks_to_weight`, 3 library calls (`jets.py:1178`; the translation, `t3toolbox/optimizers.py:196`, `:216`), 48
  in the tests, 7 library doctests, 2 page doctests; `data_metric`, 6 library calls, 41 in the tests, 8 library
  doctests; the keyword `weight=`, 206 times in 29 test files (some in old-form optimizer calls, which the translation
  keeps working); `LeastSquaresBatch(`, 76 calls in 18 test files; the M arc's tests of size-1 axes (they become
  tests of the refusal). The shared harness `tests/_fits.py` already speaks blocks (M-arc decision 18), so most fits
  change in one place.
- **Battery:** the numbers of the tree before K1, with each old block weight passed as `block_weight` and each old
  full-form weight as `full_weight`: bitwise where the arithmetic is unchanged, to rounding where the product is
  regrouped (`block**2 * full**2` against one `omega**2`). Optimizer runs are compared on what they promise (the
  loss, the iteration count, the stop reason, the represented tensor), not on a gauge.
- **Why one slice:** an intermediate tree with the engine on two weights and the oracle on one would run and be
  wrong on every weighted fit.

### K2: `backend/packing.py`

Additive: the module of §3.1 (nothing calls it yet but its tests and the engine's refusals' wording);
`frame_stack_shape` beside `frame_shape`; `PerSite` in `common.py` with the annotations of §3.3.

- **The docs build with a constrained `TypeVar` is first exercised here** (CI's Sphinx job; the extension is not in
  the local environment). If it cannot be made to build clean under `-W`, the fallback is the `Union` alias in the
  annotation with the `TypeVar`'s contract in the signature comment, decided in this slice.
- **Tests:** the new checkers against the rejection tables of `tests/test_jets.py` (the old checkers stay until K3);
  `form_of` on every container the library meets (a jax array and a numpy array are packed, whatever their rank: a
  0-d array falls to the structural checks, since nothing but the container is read; a generator, a Python or numpy
  scalar and a `PackedVectors` raise); the packers' refusals (`N` too small, members that
  disagree).

### K3: the ragged layer takes sequences only

The engine's ragged path, the ragged sampling functions (which now refuse a uniform object, §1.5), the ragged
frontend methods, the batch's readers (`_w_axes`, `_W`, `take`: `backend/fitting.py:1166-1190`), `pack_batch`'s
`pack_index` branch (from K3 a ragged index is a tuple, and a uniform entries fit must hand the engine an array) and
the dense reference take: the sample as `d` arrays; a probe object of the type of `z` as `d` arrays; the entries index
as `d` integer arrays of one shape `W` (no broadcasting between them). A single array is refused. A list is accepted
wherever a tuple is. The old checkers are replaced by the new ones here; the ragged frontend's `entries` methods stop
calling `_as_index` (§4.2); `index_to_one_hot` takes the tuple.

- **What breaks** (the investigation's measurements, to be re-run after K1): the entries index, 135 tests, 9 module
  doctests, one page, `examples/fit_hilbert_from_entries_lbfgs.py`; nothing passes an array sample today.
- **The reason for the tuple**, beside the axis coincidence it reduces: it is numpy's own multi-index
  (`np.nonzero`, `np.unravel_index`; `dense[tuple(idx)]`), so `x.entries(idx)` and `x.to_dense()[idx]` take the same
  object.

### K4: the frontend

`PackedVectors` (§4.1); the sampling methods of `UniformTuckerTensorTrain` and `UT3Tangent`; the frontend
`LocalModel`; the frontend batch (§4.3), `prepare_batch`, `_frontend_draw` reduced to a hand-over; the removal of
`_is_packed`, `_check_sample`, `_as_index`; `unscale_coordinates`; the translation (§4.4).

- **What breaks** (the investigation's measurements): 135 tests, 13 module doctests and `docs/sharing.md` reach an
  automatic conversion today; the frontend ones among them are converted here (the backend ones in K5). The two
  uniform examples call the optimizers in the 2026.5.0 form, so the translation keeps them running; their packing line
  is P17's.
- **Tests:** every uniform frontend method against its backend twin on `.array`; a `PackedVectors` of the wrong
  widths, of the wrong `N`, a tuple, a bare array, each refused; a batch whose members disagree refused at
  construction, and one of the wrong mode sizes where it enters; `take`, `n_measurements` and `data_metric` on a
  packed batch; the pytree round trip under jit with the
  mode sizes static (a second object of the same sizes does not recompile); the custom-loss walk of §4.1; a uniform fit
  that names no backend module.
- **For K4's design note to measure:** the tests that use the frontend's batch name with backend calls
  (`tests/test_least_squares_oracle.py`, `tests/_fits.py`, `tests/oracle_sweeps/*`), which work by duck typing on the
  ragged layer and need `.data` on the packed one; the other direction, backend batches handed to frontend entries
  (`tests/_fits.py:232`, the shared harness, into `fitting.local_model`; `tests/test_dispatch.py` and
  `tests/test_jax_absent.py`; the translation, which imports the backend class at `t3toolbox/optimizers.py:57` and
  from K4 builds the frontend one); and the short cross-references to `LeastSquaresBatch` in the
  docs, which a second class of that name may make ambiguous under `-W` (`docs/conf.py:144-178` resolves a name within
  its own module).

### K5: the uniform backend takes packed only

The engine's uniform path (§3.2), the twins (§3.3), `ut3_unscale_coordinates` (packed scales; it masks its output with
the masks it holds, where today the rank padding passes through, `data_preparation.py:588-592`), the data-space steps
and the shape reads as single array operations (§6), the gate on the program size. `is_packed`, `pack_if_ragged`, `pack_sample`,
`pack_data` are deleted here.

- **What goes** (`K_MEASUREMENTS.md`, M3, M4; before K0): `pack_if_ragged`, 45 library call sites, 21 of them in
  `ut3_sampling.py`; 93 reads of a container's type in the sampling and fitting modules (47 `isinstance(x, (tuple,
  list))`, 2 `isinstance(x, tuple)`, 36 `is_ndarray`, 8 `is_packed`), each of which becomes a call of the form
  reader, a refusal, or nothing.
- **Tests:** the per-element comparison with ragged on packed inputs (the data-space steps differ between the layers,
  so ragged is a real oracle for what K5 changes); junk in every padded slot of every input: finite (1e3), and `inf` and `nan`, for the cores,
  the directions, the sample, `y`, the full weight and each held variable; exact zero in every output's padding, the `'w'` echo included; against the tree before K5, **equal values, not
  equal bytes**: selection and the multiply differ in the sign of a zero (the critic's E7: 104 wanted names on clean
  inputs, every value equal, four differing only in signed zeros); the refusals, each naming its fix: a
  sequence where an array is due, for the sample, the data, the directions, `Vc` and the held variables; a ragged
  object given to a uniform twin.

### K6: the remaining helpers

The rows of §5 marked K6, each ragged only; the frontend's `fitting.rescale_coordinates`; `aligned_padded_frame`.

- **Tests:** each new function against the function it replaces on the same ragged inputs (bitwise: the computation
  moves, it does not change); a single array refused, naming `unpack_vectors`.

### K7: the guards

Recorded at the maintainer's request (his concern: erosion, "new code may start taking for granted that the padded
regions are zero").

- **Hostile fixtures by default:** the test helpers that build uniform objects and packed data put garbage in the
  padding, with an explicit opt-out for the tests of canonical output.
- **A coverage meta-test** enumerates the public uniform sampling surface (`ut3_sampling.__all__`,
  `utv_sampling.__all__`, the engine, the uniform oracle route, `ut3_unscale_coordinates`) and requires a registered
  case for each: the same call with garbage-padded and with clean inputs, the outputs bitwise equal and their padding
  zero. A raw primitive is registered with its stated precondition instead. Prototype:
  `packing_audit/junk_in_plugin.py` (its lesson: read a form by kind, never by shape).

---

## 8. Gates, for every slice

- The full gate (`implementation/run_gate.sh`, at `-n 2`): the suite, the module doctests, the quickstart, the pages,
  the examples, Sphinx with `-W`. The slow tier serially for K0, K1, K4 and K5 (they touch the optimizers' path).
  Both CI legs read after every push.
- **Expected-red lists are recorded by a run**, never predicted, for every part that leaves the side branch red.
- **Bitwise where the computation only moves, rounding where it is regrouped,** and never a tolerance loosened to
  make a knife-edge pass (change the run). Bitwise agreement of optimizer paths is welcome, not required (the
  maintainer, 2026-11-24): where it is not available the comparison is on gauge-invariant quantities.
- **Comments are content.** A scripted edit of comments is checked for ASCII diagrams and for the aligned signature
  columns before it is called mechanical; when in doubt, the slice gets its design review.
- Mutation passes by the implementer and by the reviewer, in scratch copies.

---

## 9. What this changes in the other documents and slices

- **`PART1_PLAN.md`:** its P20 becomes K0 (the check-in on `t3_corewise_frame` stays after P19; the grep of its gate
  is split between K0, P17 and P19); its text on the mirror (§2.4 "They mirror the form of the sample as today";
  P11's "mirroring the sample's form is P12's twins'"; §2.5 "the batch is ragged, and the frontend packs it for a
  uniform `x0`") is superseded by §1 and §4.3; the weight's rows by §2.
- **`DATA_METRIC_PLAN.md`:** §2's rule, §3's `data_metric` and §4's "Uniform packing" are superseded by §2 here.
- **`TARGET_DECISIONS.md`:** a dated note records the rulings of 2026-11-24 and points here and to the notes.
- **P17** (examples and pages): its list gains the packing line of every uniform example and the two weights of
  every weighted one. `point_value_and_gradient` takes the batch of §4.3 (the frontend's in the frontend, its `.data` in the backend).
- **P18** (the dry port): T3Polynomial's per-order weights are a block weight; its per-component weight a full weight with
  sample axes of 1 (§2.2); its weight per sample and slot a full weight, whose cost the port measures; it packs with
  the batch's `pack`.
- **P19** (documentation): the equivalence contract's vector-I/O section (the contract of §1.1, with its "up to zero
  coefficients"); `docs/contributor/uniform_polymorphism.md:44-47`, whose strict wording becomes true; the glossary's
  "packed vectors"; `naming_conventions.md` (the packers; the kind-first names; `PackedVectors`);
  `testing_strategy.md` (hostile fixtures, the meta-test); `fitting_and_optimization.md` §4.6 (the two weights, and the
  derivative-data recipe with its caution, §2.2); the upgrade note.
- **The `from_t3` slice** (raise for a width below a rank) stays its own small slice after K7.

---

## 10. Deferred, with the rulings standing

**To Part 2:** the option to skip the engine's input masking; all sites at once on the uniform layout (O-19).

**To the backlog considered after Part 2 (a possible Part 3):** an audit of the rest of the uniform library for G2,
with the plugin of `packing_audit/`; and the rulings of the investigation that are upstream of sampling:
`tt_left/right_orthogonalize` always returning both, and the pad-safe sweep's move (a move, not a rewrite, checked
bitwise); `t3_orthogonal_representations` ragged only; `fv_to_t3` ragged only; masks required on the supercore
orthogonalizations, `content` on `ut3svd_supercores`; the `ranks` functions (`TypeVar`, mixed forms, `use_jax`
removed); the annotations and preconditions of `tt_reverse`, `tt_squash_tails`, `t3_absorb_tucker_into_tt`; the
`ut3_to_t3` tree types.

---

## 11. The decisions

| Decision | By | Recorded |
|---|---|---|
| The definition and the two principles; P3's ruling superseded | the maintainer | notes 1c |
| G2's reach: sampling and downstream; the rest to the backlog | the maintainer | notes 1a |
| G3's criterion for a `kind` argument (computation); the private-core shape | the maintainer | notes 1b |
| The engine exempt from G3 only; the masking option to Part 2 | the maintainer | notes 2 |
| The entries index a tuple on ragged; lists accepted; no broadcasting; the uniform frontend refuses a tuple | the maintainer | notes 4 |
| The checkers on plain arguments; `backend/packing.py`; no identity packers; one unpacker | the maintainer | notes 3a, 3b |
| `PackedVectors`, in and out; no arithmetic; helpers on plain data; no frontend `check_*` | the maintainer | notes 3c |
| The two weights; `y` at exact shape; no `sample_weight` yet | the maintainer | notes 3d |
| The rule for the per-mode loops; F3.1; the program-size gate | the maintainer | notes 3e |
| The old kernels deleted first; the scope line for the cleanups; the slices before P17 | the maintainer | notes 3f |
| `full_weight`; `PackedVectors.pack` / `unpack`; the kind-first names; `from_t3` its own slice | the maintainer | notes 5 |
| `frame_stack_shape` public | the maintainer | `M_ARC_DECISIONS.md`, ruling 22 |
| The review of this plan: `mode_sizes`; selection for everything the engine masks; the kind and the order in the batch; the frontend batch of `PackedVectors`; a full weight's `W` axes may be 1, `y` exact on `W`; the metric split; the probe helpers ragged only; the slice order; §12 | the maintainer | notes 7 |

---

## 12. Decided by Claude in the draft: accepted by the maintainer (2026-11-25), item 5 overtaken

1. **The `k!` of derivative coordinates goes into the block weight only**; `prepare_*_derivative_data` has no full
   weight argument (§2.2, §5). The critic checked the algebra.
2. **`derivative_block_weight_to_taylor` is one kind-free function** (§5).
3. **No public checker for the block weight:** its exact shape is checked by the engine, by the metric functions and
   by the batch's `check`.
4. **`index_to_one_hot` is ragged only** (§3.3): the notes say it "keeps its input's form"; a packed result needs an
   `N` the ragged one does not, so by G3 it would be a second function, and nothing needs it.
5. *(Overtaken: the sample is masked, by selection, §3.2.)* The held variables are masked, as today, which departs
   from the investigation's reading that zeroed cores neutralize every held variable.
6. **The translation of the old call form keeps packing for a uniform `x0` until P17** (§4.4).
7. **The `TypeVar` fallback** (§7, K2).
8. **`form_of`** as the name of the form reader (§3.1).
9. **The order of the batch's fields and of the engine's two weight arguments** (§2.4).
10. **The entries kind keeps faces without an `order`** where it has data (§5), although each is the apply function
    at `order=()`.

## 13. The maintainer's review (2026-11-25): what was ruled

Revision 3 put seven changes made with the critic to the maintainer. His rulings (the notes' §7 has his words):

1. **The batch.** Revision 3's shared class with a static field was acceptable, and then replaced: **the kind and the
   order move into the batch, and the frontend has its own batch class of `PackedVectors`** (§4.3).
2. **The field's name is `mode_sizes`** (§4.1).
3. **The metric split by kind: accepted** (§2.3). With the kind in the batch, the batch's `data_metric(r)` stays.
4. **The probe helpers ragged only: accepted** (§3.3): "this also encourages the desired pipeline", to prepare the
   data and make it uniform once at the outset.
5. **Masking by selection: accepted, as the engine's one mechanism for everything it masks, the sample included**
   (§1.3, §3.2), after a check for hidden issues.
6. **The slice order: accepted** (§7).
7. **A full weight's sample axes may be 1; `y` stays exact on `W`** (§2.2), stated as: a size of 1 only on stack axes.

## 14. Decided by Claude in writing revisions 4 and 5: accepted by the maintainer (2026-11-25)

1. **The frontend batch has the backend's name, `LeastSquaresBatch`,** as the frontend `LocalModel` has the backend's
   (§4.3). The critic notes the precedent is weaker here: users build batches in both tiers, and one name for two
   types makes `isinstance`, messages and cross-references ambiguous. It would still keep the name, since design F
   was chosen as code with this name in it, and make every message and `repr` name the module; the plan does that.
   The alternative is a name of its own for one of the two.
2. **The backend batch has no field for the mode sizes** (§4.3): under the frontend batch nothing in the library
   would read it.
3. **`_frontend_draw` stays as a hand-over of `.data`** (§4.3), where revision 3 removed it.
4. **The frontend batch validates its members at construction** (§4.3), so what is left where a batch meets a point
   is the comparison of its mode sizes and `N` with the point's, and the engine's checks.
5. **`LeastSquaresOracle` keeps its name and has no fields** (§4.3).
6. **The batch's methods move in K1** (with the kind and the order), on the backend class; the frontend class is K4's.
7. **`n_measurements` is a property of the batch; `pack` on a packed batch raises; in the backend packing stays the
   function `pack_batch`, not a method** (§4.3).
8. **A frontend entry refuses the backend's batch** (§4.3), and a validation or drawn batch of another kind or
   order.
9. **Part 0 of K1 is the move of the kind and the order,** with its own bitwise battery (§7).
