# Part 1 of the graded-messages refactor: implementation plan

**Status.** The implementation plan of Part 1. Approved by the maintainer on 2026-11-16 at 21:16; slices P0 to P4
are built. **The change of 2026-11-17 for the generalized jet group (next paragraph) is agreed by the critic; its
open points were answered and its names approved by the maintainer on 2026-11-17 (16:21 and 16:26; section 11.1,
section 11.4).** **The maintainer signed off the plan as a whole on 2026-11-19** ("Part 1 of the plan is confirmed as
a whole. But we may encounter issues along the way which require adjusting the plan as we go."; `TARGET_DECISIONS.md`,
the note "The changed plan signed off"); P4a starts. A change found necessary while building is made here and
recorded. The plan was first written against the branch
`graded-messages` at `a9c31b2` (the library and `tests/` unchanged since `4d38df7`); every statement about the
library was checked against the code at that commit, and counts were recounted there.

**Changed on 2026-11-17 for the generalized jet group** (entries 116 to 120 of `REVIEW_DECISIONS.md`;
`TARGET_DECISIONS.md`, the notes "One jet group, of any number of axes" and "The public interface is generalized
from the start"). Four slices, P4a to P4d, are inserted between P4 and P5; they rebuild what P1 to P3 built where
the change reaches it, and every later section is brought in line. Written against `ae3811d` (the library and
`tests/` as P4 left them at `5991725`). The account of the change, with every choice it makes, is
`reviews/jet_group_plan/ACCOUNT.md`. The maintainer's answers to its open points, O-36 to O-47, are in section 11.1.

**Two terms, defined once and used everywhere below.** The **jet shape** is the tuple of maximum orders, one per
jet variable, `order = (n_1, ..., n_m)`, as the record uses the term (entries 116, 117, 120: "one jet shape in place
of the integer order"); the argument is named `order` (entry 118). `order = ()` (`m = 0`) is the computation without
jets, which earlier versions of this plan called order 0. The **jet sizes** of an array are the sizes of its `m`
leading jet axes; a full jet has the jet sizes `F = (n_1 + 1, ..., n_m + 1)`. In shape comments `F` is the full jet
sizes, `J` the jet axes of an array in general and `J_in_i` those of the input `ww[i]` (N39). A jet is a polynomial
in `s_1 ... s_m` truncated to the box `F`; an array's jet sizes mean that its coefficients beyond them are zero. Every
array that carries jets has exactly `m` leading jet axes. The inputs are jets: `ww[i]` is `J_in_i + W + (N_i,)`,
sizes of its own, and there is no `pp`. The entries kind has no jets. `order` is a tuple only; an integer raises
(O-36). It has the default `()` on the sampling functions and methods, and no default on `LeastSquaresOracle`, where
it is required for every kind, `()` for entries (O-44, O-46).

**How to read it.** The rows of `TARGET_DECISIONS.md` are the authority, with the maintainer's answers to this
plan's open points (the note "The plan's open points, as the maintainer answers them" and the notes after it,
2026-11-16). A rule is given here with its row number (for example "2.9") and, where a reader needs it, one
sentence of reason. `SAMPLE_VARIABLES.md` is the reference for every variable: its name, definition, recurrence
and shape. The prototypes in `scripts/review/` establish the algebra; the library's contributor notes
(`docs/contributor/`) fix the form of the code.

---

## 1. Scope

**Part 1 delivers what is computed.** The sampling variables of `SAMPLE_VARIABLES.md` under their decided names;
one engine, `sampling_variables`, that computes any of them for the three kinds, in Taylor coefficients, for jets
in any number of variables and of any order (the probe and apply kinds; the entries kind has no jets), with
`order = ()` as the same functions; `jet_contract` over one jet group `J` of any number of axes, and a literal
contraction string at every call site; inputs that are jets, with a helper for the line `w + s p` and one for
one-hot vectors;
the weight and the data as arguments; frames with optional centers; arbitrary tails; both layouts. Above it: the
oracle and the batch, `LocalModel`, the optimizers' new seam, the display, and standalone functions that prepare
data and weights. Around it: the reference module of definitions and the dense functions it calls, the checkers,
the tests by definition and the identities, the documentation of the variables, and a dry port of T3Polynomial.

**For jets in several variables Part 1 owes the right interface and correct results**, tested against the
definitions; their speed is Part 2's (entry 117). It keeps today's cost in the cases the library uses: one jet
variable; a full jet with a full partner or a partner of two slots; the block readouts (section 2.4, "The
algorithm in Part 1").

**Part 1 carries out every computation the way the library does it today.** Where a row is open it fixes the
following ("The two parts", and the answer to O-19):

| Open row | What Part 1 does |
| --- | --- |
| 5.3 sites | Sweeps go through `xscan`. Map-type operations go one site at a time: a loop over the sites on the ragged layout, `xmap` on the uniform one, at `order = ()` and for jets alike. There is no site group: a step sees one site's arrays on both layouts |
| 5.5 contraction order | `contract`'s present default: one einsum on jax, the cached greedy pairwise path on numpy |
| 7.2 defaults | The model holds the plain sweep variables and the residual |
| 7.3 where the plan lives | There is no plan, and no signature has a `plan` argument |
| 7.5, 7.6, 7.7 | No chunking of samples, so no padding of samples, no sizing helpers, no promise about sharding |

Decided rows whose options wait for Part 2: 2.9 (Part 1 sets the form over the orders itself: `'scan'` on the
uniform layout, `'loop'` on the ragged one, as the library runs today; both forms are built), 4.3 (Part 1 has
trial policy a, with the centers of 4.1 in place), 7.1 and 7.4 (no plan), 8.3 (Part 1 compiles the CG solve and
the stochastic step, as today, and makes `take` traceable).

**What Part 1 gives up until Part 2.**
- *Memory.* The frontend `chunk_size` and the library's chunked uniform assemblies (`_wchunked_sum` in
  `backend/sampling_derivatives.py`, called from four functions: the Tucker and the TT assembly of the transpose and
  of the tangent) go. Until Part 2, a large jetted fit on the uniform layout with jax uses more
  memory than today. T3Polynomial's fitting module does chunk today: with `newton_kwargs=None` it passes no
  `chunk_size` (`t3polynomial/fitting.py`, `_level_newton_kwargs` and `_newton_fit`), so the library default
  `chunk_size='auto'` (`t3toolbox/optimizers.py`) applies on its uniform backends.
- *Time at `order = ()` on the uniform layout.* Today's plain kernels run the map-type operations at all sites at once
  (one contraction with a leading `d`: `compute_eta`, `assemble_z`); Part 1 maps them one site at a time, as
  today's jetted kernels do (`get_backend(True, True)` makes `xmap` `jax.lax.map`; `sampling_derivatives.py` calls
  it sixteen times). The plain ones may be slower than today until Part 2.
- *No benchmark in Part 1.* The old kernels are deleted on the tests alone (section 10). Part 2's performance work
  benchmarks the branch against the released 2026.5.0.

**The seams left for Part 2.** Part 2 adds; it does not change Part 1's interface.

| Seam | Where Part 2 adds |
| --- | --- |
| The plan | An optional `plan=None` keyword on `sampling_variables`, and wherever 7.3 puts it (the oracle, the model or the solver). No Part 1 signature has to change |
| The form over the orders (2.9) | `jet_contract` already takes `over_orders`; Part 1's engine sets it from the layout, Part 2 from the plan. For `m >= 2` both values run the loop in Part 1 (O-38) |
| The sites (5.3) | The map-type operations at all sites at once on the uniform layout, if Part 2 wants them: uniform code of its own, written with the library's `d` axis, not the shared step with an optional group (the maintainer, 2026-11-17: the site group `S` is removed). To be revisited in Part 2 (O-19) |
| What is held (7.2) | The least-squares oracle's fixed hold list becomes a field of the plan; `held` already flows through every call |
| Chunking (7.4) | Chunks are array shapes of the batch; the engine already accepts any number of sample axes `W` |
| Trial policies b and c (4.3) | An optimizer option; the geometry already returns centers with the frame |
| Several MC-SGD steps in one program (8.3) | The stochastic step is a module-level function of pytrees, and `take` is traceable with a static size |
| Contraction order (5.5) | Every term is a literal string at its call site, so a per-term order has one place to attach. Known cost until then: on numpy the default order inside one step of a product does the expensive contraction on every slot (`TARGET_DECISIONS.md`, "Carried to Part 2", item 1; marked in the code and by an expected-failure test) |
| The benchmark | Part 2 compares the branch with the released 2026.5.0; the dry port (section 9) lets T3Polynomial's benchmark run on the branch |
| Jets in several variables (5.4, 2.9) | Fast algorithms for `m >= 2` live inside `jet_contract`, behind its signature: the order in which a product's multi-indices are taken, a scan over more than one jet axis, transforms. Part 1 has a simple correct loop over the multi-indices of the operands taken one coefficient at a time, and under `'scan'` it scans only where `m = 1` (section 2.4). No call site changes when Part 2 replaces it |
| Total-degree data (2.5) | Part 1 has the box only. Data measured up to a total degree fit the box with weight zero on the slots not measured; a cheaper storage of them, if wanted, is Part 2's |

Part 1 is not released on its own. All work stays on `graded-messages` until Part 2 is done.

---

## 2. The end state

### 2.1 Modules

| Module | Holds at the end of Part 1 | Status |
| --- | --- | --- |
| `backend/contractions.py` | `contract` (unchanged); `jet_contract` over one jet group `J` of any number of axes; the record of axis counts `GroupLengths(J, W, K, C)` | changed |
| `backend/sampling_variables.py` | The engine `sampling_variables`; `HeldVariables`; the declared graph, one tuple of records per kind; the shape table | new |
| `backend/sampling_steps.py` | The 22 step functions and the start values. Every name private, `__all__ = []` | new |
| `backend/sampling_definitions.py` | The reference module: one slow, dense function per variable and per output (9.7). Backend only | new |
| `backend/jets.py` | `jet_to_dual_jet` and `dual_jet_to_jet` (3.6); the four functions for derivative data (2.3); the two helpers that make input jets, `affine_jet` and `index_to_one_hot` (names approved, section 11.4); the checkers of a sample, its data and its weight, `check_sample`, `check_data`, `check_weight`. The two validators of `pp` that P3 copied here go in P4d | new |
| `backend/data_preparation.py` | The five preparation functions (6.7), and `ut3_unscale_coordinates`, the uniform twin of the last (O-34) | new |
| `backend/probing.py`, `apply.py`, `entries.py` | One backend function behind each frontend method, each calling the engine, with the merged signature (6.3, 6.6); the ambient transposes in the form of N16, their own code otherwise as it is; the dense functions beside their kinds: `dense_probe(T, ww, order=())` in its new form, `dense_apply(T, ww, order=())` and `dense_entries(T, index)` new | rewritten |
| `backend/ut3_sampling.py`, `utv_sampling.py` | The uniform twins of the functions above: check packing, call the engine; `ut3_full_sum` as it is | rewritten |
| `backend/sampling_derivatives.py` | Deleted in the last slice | removed |
| `backend/fitting.py` | `LeastSquaresOracle` (with `check_batch`), `LeastSquaresBatch`; the default draw's `take` and `n_measurements` | rewritten |
| `backend/uniform_fitting.py` | `pack_sample`, `pack_data` (taking `(kind, order)`), `pack_batch`, `uniform_minimal`. The six uniform kinds and their factories go | rewritten |
| `backend/optimizers.py` | The `Geometry` and `Oracle` protocols; `LocalModel`; `local_model`; `point_objective`; the five phases; the four optimizers; `flat_draw`. `Problem`, `least_squares_problem` go | rewritten |
| `backend/optimizer_display.py` | The display, reading from an oracle and a batch | changed |
| `backend/geometry.py` | `frame(x)` returns the frame with its centers; the corewise geometries' frame is the point | changed |
| `backend/fv_operations.py`, `backend/ufv_operations.py` | The checkers of a frame: `fv_frame_consistency_residual` without dense tensors, the centers' residual beside it, and the uniform twins of both (4.2; O-5, O-6) | changed |
| `t3toolbox/tucker_tensor_train.py`, `uniform_tucker_tensor_train.py`, `manifold.py`, `uniform_manifold.py` | Merged sampling methods (6.6); `unscale_coordinates` on `TuckerTensorTrain` and `UniformTuckerTensorTrain` | changed |
| `t3toolbox/frame_variations_format.py`, `uniform_frame_variations_format.py` | The cached `consistency_residual` of `T3Frame` and `UT3Frame`, and `is_consistent` on it, without dense tensors | changed |
| `t3toolbox/fitting.py` | The frontend `local_model` over the backend one; the oracle and the batch, the preparation functions and the jet helpers reachable under their backend names; its own `dual_jet_to_jet` | rewritten |
| `t3toolbox/optimizers.py` | The four optimizers with `(geometry, oracle, batch, x0, ...)` | rewritten |
| `t3toolbox/shared_geometry.py` | Unchanged in behaviour; its `frame` follows the backend change only where it calls the backend | checked |

The import rules stay: the backend graph is acyclic, the backend never imports `t3toolbox.safety` ("There is no safe
mode in the backend", the answers of 19:30 and 20:04), and `backend/optimizers.py` stays T3-free (`tests/test_import_graph.py` pins its
imports to `common`, `regularization`, `corewise`).

### 2.2 The data

**A frame**, as the engine takes it, is one of four structures. The structure says which call it is, and it stays
visible under `jit` (small item 1 of "The two parts").

| Call | Ragged layout | Uniform layout |
| --- | --- | --- |
| Manifold | `(D, U, L, R)`: four tuples of `d` arrays (`SAMPLE_VARIABLES.md` section 1) | `(D, U, L, R, shape, masks)`: today's uniform frame `.data` |
| Corewise | `(B, G)`: the point's two tuples of cores | `(B_sc, G_sc, shape, masks)`: today's `UniformTuckerTensorTrain.data` |

The rule is layout first, then the count. Ragged or uniform is read from `is_ndarray` on the first element, as the
library does today (`docs/contributor/uniform_polymorphism.md`): a tuple of cores is ragged, an array is uniform.
Then the count of elements: on ragged, two (corewise) or four (manifold); on uniform, four (corewise) or six
(manifold). On a corewise call the variables have no basis in their names (`SAMPLE_VARIABLES.md` section 9), and
the names with a basis are not accepted (O-13).

**Centers** `(Vc, Hc)`: `Vc[i]` has the shape of `V[i]`, `Hc[i]` that of `H[i]`. They are the second result of
`fv_conversions.t3_orthogonal_representations` (its "variations"), which the geometry's `frame` discards today
(`backend/geometry.py`, `ManifoldGeometryOps.frame`). The engine takes `Vc` only (4.1).

**The jet shape** `order` is a tuple of maximum orders, one per jet variable (entries 116, 117, 120): `m =
len(order)` jet variables, `order = ()` for none. The full jet sizes are `F = tuple(n + 1 for n in order)`. Every
array that carries jets has exactly `m` leading jet axes, and `order` decides how the leading axes are read,
never the type of the object (6.2). A jet is a polynomial in `s_1 ... s_m` truncated to a box; an array's jet sizes
mean that its coefficients beyond them are zero, so an axis of size 1 is a variable the array does not depend on.
Every slot of a jet axis is a coefficient: there is no don't-care slot on a jet axis (section 7.2). The entries kind
has no jets.

**The slots of an input jet are Taylor coefficients**: `ww[i]` at the multi-index `k` is the coefficient of `s^k`.
This follows from rows 2.1 and 2.2 ("Taylor coefficients replace derivatives everywhere"; "the only form in the
sampling, fitting and optimization layers"); a user whose curve is given by its derivatives converts it with
`derivatives_to_taylor` first. For a line it makes no difference.

**A sample.** For probe and apply, `ww` alone, at every order: `ww[i]` is `J_in_i + W + (N_i,)`, the input jet of
mode `i`, with `m` leading jet axes of sizes of its own (at `m = 0` it is `W + (N_i,)`). The line `w + s p` is the
input jet of two slots, `affine_jet(ww, pp)` (section 2.4); a curve has more slots; `w + s_1 p_1 + s_2 p_2` is
`affine_jet(ww, pp_1, pp_2)`, of jet sizes `(2, 2)` with a zero at `(1, 1)`. The modes may have different jet
sizes (a mode of size 1 along every axis does not move: a discrete mode beside continuous ones). `order` is
never inferred from the sample (entry 119): the count of jet axes is not in the shape, since `W` has any number of
axes, and row 6.2 ("`order` decides, never the type") is why the array cannot say it. So a forgotten `order` is not
seen by any check of shapes (O-44). For entries, `index`, `(d,) + W`, integer, with no jets (entry 120). There is no
`pp` anywhere. On the uniform layout the vectors are packed once, when the batch is packed (`pack_sample`, called by
`pack_batch`): the modes' jet axes are brought to their largest size per axis by writing zero coefficients, which
are data, not padding (section 7.2; O-45). The engine checks the packed sample's shapes against one another
(`(d,) + J + W + (N,)`, `len(order)` jet axes) as it needs to run, and nothing about its values.

**The data `y`** is of the type of `z`, in forward order (2.4): probe `y[i]` is `F + W + (N_i,)`; apply `y` is
`F + W`; entries `y` is `W`. It broadcasts over `C`. Data measured only up to a total degree are given in the box,
with weight zero on the slots not measured (entry 116).

**The weight** has exactly the shape `(d,) + F` for probe, `F` for apply and `()` for entries; no rule for a bare
vector, no broadcast of a mode dimension of 1 (2.5, 6.7). `None` means ones. The engine promises nothing about a
weight of another shape; the backend checker `check_weight` tests it, and the frontend calls it where a weight
comes in (O-18).

**A batch** is the data of a least-squares fit, arrays only (8.x note "The oracle is a description"):

```python
@dataclass(frozen=True, eq=False)
class LeastSquaresBatch:                 # a registered pytree: every field a leaf (None allowed)
    sample:          typ.Any             # ww (probe, apply: input jets, elm_shape=J_in_i+W+(Ni,)) | index (entries)
    y:               typ.Any             # the data: probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W
    weight:          typ.Optional[NDArray] = None   # probe (d,)+F; apply F; entries (); None = ones
    minibatch_scale: typ.Optional[float]   = None   # n/batch_size on a drawn batch; None = 1 (a full batch)
    drawn_indices:   typ.Optional[NDArray] = None   # int, (batch_size,): which samples a draw took
```

A user's oracle with no data of its own takes `None` as its batch. The batch and the least-squares oracle are one
class each for both tiers, reachable from `t3toolbox.fitting`.

**An oracle** is a frozen description with no arrays and one operation (8.2, 8.3). Its fields are statics, so it
is value-hashed (`common.ValueHashedFields`) and rides as `aux_data`. The least-squares oracle has two fields,
`kind` and `order` (the jet shape, a tuple, required for every kind with no default, O-44; for the entries kind it
must be `()`, and any other value raises when the oracle is built, O-46: `LeastSquaresOracle('entries', order=())`).

**`held`** is what an oracle keeps at one frame; what goes into it is the oracle's business (8.3), and the
optimizers pass it back unread. The `Oracle` protocol types it as any object (O-8); in a compiled program it must
be something jax can carry: arrays, tuples, `None` or a registered record (O-30). The least-squares oracle's is a
`HeldVariables` (O-8):

- a frozen record with one slot per PUBLIC COMPUTED variable of the declared graphs, taken from them (a test checks
  that the two agree): row 6.4, every variable can be held; the input `w` is an argument, not a slot, and private
  nodes are never slots (amended 2026-11-20, P5 and P6's design reviews; it first said "per variable that some other
  variable reads", which would drop `eta_tilde` once `V_tilde` reads only the short form); each slot is `None` when not held, and carries its shape comment;
- it reads like a dictionary: `held['mu']`, `'mu' in held`, `held.keys()` (the filled slots), iteration, `len`;
  an unknown name raises, at construction and at lookup;
- `held.with_values(mu=mu)` returns a new record; a record is never changed in place;
- it checks no shape: the engine checks every held variable it is handed, at every call (6.5);
- it is an ordinary object for a numpy-only user, and is registered as a jax pytree only where jax is importable
  (the filled slots are the leaves, the set of filled slots is the static structure);
- the engine takes it as `held=`, and the oracle's `query(..., held=...)` passes it through.

**A local model** is the geometry, the oracle, the batch and the regularizer at one frame, and the holder of the
local information there (8.1). It never changes after it is built.

| Field | Kind | What |
| --- | --- | --- |
| `geom` | aux | the backend geometry (value-hashed, as today) |
| `oracle` | aux | the oracle (value-hashed) |
| `regularizer` | aux | as today |
| `batch` | leaves | the batch the model was built on |
| `frame` | leaves | the frame (`partition_static` keeps uniform shapes and masks static, as today) |
| `centers` | leaves | `(Vc, Hc)` when the geometry returned them, else `None` |
| `held` | leaves | what the oracle returned at the build (for the least-squares oracle, the filled slots of `HeldVariables`) |
| `f` | leaf | the misfit at the build |
| `geom_aux` | leaves | `geom.precompute(frame)`, as today |
| `curvature` | leaves | the `CurvatureState`, exact mode only, as today |

`sweep`, `residual`, `sample`, `kind` and `n_w` stop being fields (today's backend `LocalModel` in
`backend/optimizers.py` has all five). The residual stays reachable as `model.residual`, an optional oracle output
read from `held`.

### 2.3 The engine

```python
def sampling_variables(
        kind:              str,                              # 'probe' | 'apply' | 'entries'
        frame:             typ.Tuple,                        # a manifold frame or a corewise point: section 2.2
        sample:            typ.Any,                          # ww (input jets, J_in_i+W+(Ni,)) | index (entries)
        want:              typ.Tuple[str, ...],              # variable names (the table); the return is aligned

        order:             typ.Tuple[int, ...] = (),         # the jet shape (n_1..n_m); F=(n_1+1,..). entries: () only
        y:                 typ.Optional[typ.Any] = None,     # the data; needed for r and what reads it
        weight:            typ.Optional[NDArray] = None,     # probe (d,)+F, apply F, entries (); None = ones
        variations:        typ.Optional[typ.Tuple] = None,   # (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        core_directions:   typ.Optional[typ.Tuple] = None,   # (D_dot, L_dot) at (D, L), shapes of D, L with K+C
        Vc:                typ.Optional[typ.Sequence[NDArray]] = None,   # len=d, elm_shape=C+(nUi,Ni)
        sum_over_samples:  bool = True,                      # on g, JtJp, Kp; False: W joins the stack, W+K+C
        held:              typ.Optional['HeldVariables'] = None,   # held variables; shapes checked, values trusted
) -> typ.Tuple[typ.Any, ...]:                                # aligned with want; shapes in the table of variables
```

The three placeholder keywords of 6.2 are named `variations`, `core_directions`, `sum_over_samples`
(table of names, rows N1 to N3). The `plan` of the 6.2 sketch is absent in Part 1 (7.3).

**The engine gives no guarantees beyond what it needs to run.** The backend gives checkers, not guarantees (the
answers to O-6 and O-18, and "Checking in the two tiers", 20:04). It does not import `safety`, has no safe mode, and checks nothing numerical: a caller who passes four
inconsistent stores, wrong centers or a weight of a wrong shape gets a wrong answer or an error from deep inside,
and has the checkers to test those inputs first (section 7.1).

**What a call does, in order.**

1. **What it needs in order to run, always.** An unknown kind or an unknown name in `want` raises (the graph cannot
   be read otherwise); a wanted variable whose argument is missing raises (`Kp` without `core_directions`; `r`
   without `y` and without a held `r`); an entries call with `order != ()` raises (its sample cannot carry jets);
   every held variable it is handed has the shape of the table, its declared jet sizes included (6.5). The ranks of the operands are checked in `jet_contract` (5.4). Nothing else is checked here: not
   the form of the sample, not its sizes against the frame's, not the shape of `y`, not the shape of the weight
   (those are the checkers of `jets`, section 2.4).
2. **Layout and call type** from the structure of `frame` (section 2.2). On the uniform layout the inputs are masked
   once, here, and nothing below sees a mask (5.2). Every mask comes from the object the operation is defined at,
   the frame, never from a superset of it (`docs/contributor/testing_strategy.md`; CLAUDE.md on the two-hole): the
   frame's stores with their own masks; `variations` and `Vc` with the frame's variation masks
   (`ufv_masking.ufv_variation_masks`, as the uniform geometry does today); `core_directions` `(D_dot, L_dot)` with
   the masks of the frame's `D` and `L` stores, not the point's (which are a superset at TT slack).
3. **The closure.** From `want` and the filled slots of `held`, the declared graph gives the set of variables to
   compute, in dependency order. Not wanting a variable lets a step take its short form (`eta_tilde` in its two
   read rows when only `g` reads it); wanting it never changes a value (6.4).
4. **Backend and dtype.** `use_jax = tree_contains_jax(...)` over every array input. The result dtype follows the
   absorbing rule (5.6): jax over numpy, the lowest floating precision among the counted inputs. The counted inputs
   are "all numerical (floating point) arrays whose entries could affect the numerical values of the output"
   (O-11): of the frame, `Vc`, the sample's vectors, `y`, the weight, the two directions and the held variables,
   those the closure of step 3 reads; an integer `index` does not count, and neither does an argument nothing
   wanted reads. The counted inputs are cast once to the result dtype; every start value is built in it; constants
   are Python scalars. **Amended 2026-11-20 (P5, review round 1, ruled by Claude from 5.6 and O-11):** a held
   variable counts when a row reads it OR it is wanted (a wanted held variable is an output, so its entries affect
   it), and is cast once like any counted input; so every output of a call but the sample `w` (returned as given) is
   in the one result dtype. A held variable neither read nor wanted is not counted and not returned. `use_jax` still
   reads every array input, held ones included (P10's dtype matrix states that case).
5. **The sweeps and the maps.** A sweep along the chain goes through `xscan` with a module-level body
   (`docs/contributor/scan_body_principles.md`). A map-type operation (the `xi`, `eta`, `z` steps and the block
   readouts) goes one site at a time: a loop over the sites on the ragged layout, `xmap` on the uniform one, at
   every order (O-19). Every term is one `jet_contract` call with the output's jet shape from the table of declared
   jet sizes below; `over_orders='loop'` on the ragged layout and `'scan'` on the uniform one (O-10). Boundary vectors are ones sized by the end bond of the cores
   passed; start values follow the grade (4.4). The engine squashes no tail.
6. **The seed.** `r = z - y` (the library's sign, 2.4); the seed `w_tilde = jet_to_dual_jet(kind, order, r, weight)`: the
   weights, then the turn of every jet axis (a flip of each). `JtJp` seeds from `Jp` the same way. A held `r` is used as it is.
7. **The sums.** `f = 1/2 <r, omega^2 r>`, `r_block_sumsq` (unweighted, the shape of the weight),
   `Jp_norm_sq = <Jp, omega^2 Jp>` are formed inside the call; `f` keeps `C`, `Jp_norm_sq` keeps `K + C`
   (3.7).
8. **The return**, a tuple aligned with `want`. On the uniform layout the core-shaped outputs (`g`, `JtJp`, `Kp`, the
   blocks) are re-masked on exit with the masks the engine took on entry (`Kp`'s blocks with the frame's `D` and
   `L` masks, the shapes of `D_dot` and `L_dot`), so every caller, the oracle included,
   sees clean padding (the schedule of `docs/contributor/uniform_internals.md`: mask on entry, re-mask the output of
   every transpose).

**The names the engine accepts** (6.4: every variable of a kind's graph, at every grade):

| Group | Names |
| --- | --- |
| Inputs and seed | `w`, `w_tilde` |
| Plain | `xi`, `mu`, `nu`, `eta`, `nu_L`, `eta_D` |
| Dot, frame form | `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot` |
| Dot, raw form | `xi_dot`, `mu_dot` (and `nu_L_dot`, `eta_D_dot`, the same arrays) |
| Tilde | `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `nu_L_tilde`, `eta_D_tilde` |
| Tilde-dot | `xi_tilde_dot`, `mu_tilde_dot`, `nu_L_tilde_dot`, `eta_D_tilde_dot` |
| Outputs | `z`, `r`, `Jp`, `g`, `JtJp`, `Kp`, `V_tilde`, `H_tilde`, `V_tilde_dot`, `H_tilde_dot` |
| Sums | `f`, `r_block_sumsq`, `Jp_norm_sq` |

**The declared jet sizes of each variable** (entries 116 and 117: each variable of the engine has a declared jet
shape, the input's or the full one; written here as the sizes of its jet axes). With `F` the full jet sizes and `J_in_i` the jet sizes of `ww[i]`:

| Variables | Jet sizes at index `i` | Why |
| --- | --- | --- |
| `w[i]` | `J_in_i` as given | the input |
| `xi[i]`, `xi_dot[i]`, `xi_U_dot[i]` | `min(J_in_i, F)` axis by axis | a Tucker core on the input: exact, of the input's degree; slots beyond `F` never reach an output, so they are dropped here |
| every other variable: `mu`, `nu`, `eta` in every basis and grade, the seed `w_tilde`, every tilde and tilde-dot variable (`xi_tilde` and `xi_tilde_dot` included), `z`, `r`, `Jp` | `F` | a product truncated to the box, or a dual jet, which is always full (2.4) |
| `r_block_sumsq`, `y_block_sumsq` | the shape of the weight: `(d,) + F`, `F`, or `()` for entries | a sum over the samples, per mode and slot |
| the blocks, `f`, `Jp_norm_sq` | none | read at the corner coefficient, or summed over every slot |

The engine passes each `jet_contract` call the jet shape of its output from this table: `order` itself for a
variable of jet sizes `F`, `tuple(min(a, b) - 1 for a, b in zip(J_in_i, F))` for a leaf. No call asks for an output
box larger than that of an operand that is a truncation (entry 116): the only operands that are not truncated are
the inputs and the leaves, whose jets are exact. The entries kind has no jet axes. With a line input and one
variable this is today's `T` and `A` exactly: the leaves have two slots, everything else `order + 1`.

`nu_L_dot` and `eta_D_dot` are one name each for both dot forms (`SAMPLE_VARIABLES.md` 5.2, "One dot"). In a call
that has both `variations` and `core_directions`, the two directions differ in general, so the engine forms the
frame form for `Jp`/`JtJp` internally, and the public name is the raw form, at the base point, the one `Kp` reads
(O-12). The tilde names always mean the residual's adjoint sweep (3.3).

**The declared graph** is plain data: per kind, a tuple of records `(name, step, reads, needs)` (6.4). It lives in
`sampling_variables.py` beside the engine and `HeldVariables`; the docstring's table of variables is written by
hand and a test checks the two agree row for row, and another that the slots of `HeldVariables` are exactly the
variables some other variable reads.

**What the engine does not know.** The geometry (projection, lift, Hessian assembly stay there, 4.5), the
regularizer, any loss other than the weighted sum of squares (3.6), derivative data (2.2), and plans. It has no
frontend entry ("I would leave the engine to the backend", the answers of 20:15).

### 2.4 The public surface, backend

**New.**

| Function or class | Module | Signature sketch |
| --- | --- | --- |
| `sampling_variables`, `HeldVariables` | `sampling_variables` | sections 2.3 and 2.2 |
| `jet_contract`, `GroupLengths` | `contractions` | below |
| `jet_to_dual_jet`, `dual_jet_to_jet` | `jets` | below |
| `derivatives_to_taylor`, `taylor_to_derivatives`, `derivative_weight_to_taylor_weight`, `prepare_derivative_data` | `jets` | below; the bodies as in `scripts/review/adapter_doctest_draft.py`, with the probe kind's mode axis |
| `check_sample`, `check_data`, `check_weight`, beside `check_perturbation_vectors` and `check_perturbation_index` | `jets` | below |
| `data_block_norms`, `equalizing_weight`, `rescale_direction`, `rescale_coordinates`, `t3_unscale_coordinates`, `ut3_unscale_coordinates` | `data_preparation` | below |
| `dense_apply`, `dense_entries` (and `dense_probe` in its new form) | `apply`, `entries` (`probing`) | below |
| one function per variable and output | `sampling_definitions` | below |
| `fv_centers_consistency_residual`, `ufv_frame_consistency_residual`, `ufv_centers_consistency_residual` | `fv_operations`, `ufv_operations` | below |
| `Oracle` (protocol) | `optimizers` | below |
| `LeastSquaresOracle` (with `check_batch`), `LeastSquaresBatch` | `fitting` | below and section 2.2 |
| `pack_batch` | `uniform_fitting` | below |
| `local_model`, `point_objective` | `optimizers` | below |

```python
@dataclass(frozen=True)
class GroupLengths:                 # the axis counts of one call's groups (5.4); hashable, a static
    J:  int                         # jet axes: m = len(order), any count >= 0
    W:  int                         # sample axes
    K:  int                         # direction-stack axes
    C:  int                         # base-point-stack axes

def jet_contract(
        subscripts:   str,                    # a literal grouped string over J, W, K, C; J leads its term
        *operands:    NDArray,                # one per input term; each with its own sizes along J
        lens:         GroupLengths,           # every call passes the whole record, so a wrong rank raises
        order:        typ.Tuple[int, ...],    # the output's jet shape, len=lens.J; no default (entry 116)
        over_orders:  str,                    # 'scan' | 'loop': how a jet is taken one coefficient at a time
) -> NDArray:                                 # J=tuple(n+1 for n in order) leading if the output term has J
```

**What `jet_contract` computes: one behaviour.** Each operand is a polynomial in `s_1 ... s_m` (`m = lens.J`): an
operand whose term has `J` has its coefficients on its `m` leading axes, of sizes of its own, and is zero beyond
them; an operand whose term has no `J` is a constant. The result is their product, contracted as `contract`
contracts the string with `J` deleted. If the output term has `J`, the result is that product's coefficients on
the box `F = tuple(n + 1 for n in order)`: the truncated product. If it has no `J`, the result is the one
coefficient at the corner `s_1^(n_1) ... s_m^(n_m)`: a block readout, since the pairing of a jet with a dual jet
stored turned is the corner coefficient of their product (2.4). What were three behaviours are cases of it: a
string with one jet operand and `J` in the output is a ride (one `contract`, the operand sliced to the box, the
result padded with zeros where the box is larger); two jet operands and `J` in the output is a product; no `J` in
the output is a readout. At `m = 0` (`lens.J == 0`, `order == ()`) the call is exactly one `contract` call on the
string as written (2.7).

The easy cases to get wrong, stated:
- *An operand with fewer slots than the box* along an axis is zero beyond them (entry 116): nothing is padded; its
  coefficients are taken where they exist.
- *An operand with more slots than the box*: its slots beyond the box never reach the result (in a product the
  index of a contribution is the sum of the operands' indices) and are not read.
- *A jet axis of size 1* is a variable the operand does not depend on. It is an axis like any other and counts in
  the rank; it is not an absent axis.
- *An operand with no `J` in its term* is a constant: at `m >= 1` it has no jet axes at all.
- *The output's box larger than every operand's* is allowed and well defined (zeros where no product reaches);
  the engine never asks for it.
- *No jet operand*: a string in which no operand has `J` raises, at every `m`, `m = 0` included. Under the one
  behaviour such a string with `J` in the output would be a constant in slot `(0, ..., 0)`, and a readout zero
  unless `order` is all zeros; neither occurs in the library, and either is more likely a mistake in a string than
  an intent. **Changed 2026-11-19, at the start of P4a** (Claude, on a point the implementer and the reviewer raised
  independently): this said "at `m >= 1`", with every string plain `contract` at `m = 0`, which would make a
  string's validity depend on the order, against P1's rule that a call site is wrong at every order or at none. A
  string with no `J` is a `contract` string. (A string with `J` in the output and no jet operand is also rejected by
  `contract`'s parser.)
- *The truncation is the caller's.* `jet_contract` treats every operand as an exact polynomial; it cannot tell a
  full jet truncated at order 1 (two slots) from an exact line (two slots). The rule of entry 116, that the output
  box must not exceed the box of an operand that is itself a truncation, is kept by the engine through its table
  of declared jet sizes (section 2.3), not checked here.

**What `jet_contract` checks**, structurally, at every call (5.4; "What the engine checks itself" of the answers
of 20:15): the letters of the string are among `J`, `W`, `K`, `C` and lowercase; `J` leads every term that has it; at least
one operand has `J`, at every `m`;
the record is a `GroupLengths`; `len(order) == lens.J`, each entry a nonnegative integer; every operand has the
rank its term has under `lens` (a stray or a missing axis raises, wherever it is); every jet axis of every operand
has at least one slot; `over_orders` is `'scan'` or `'loop'`. Nothing numerical. The products of the strings may
have any number of jet operands; the library's strings have one, two or three.

**`GroupLengths`** keeps its role: one record per computation, passed whole to every call, hashable, a key of the
memoized scan bodies. **Amended 2026-11-20 (P5's design review):** one record per NODE of the engine's graph, each
row of the graph declaring which it uses: the record of the call, or the same record with `K = 0` for a node that
carries no direction. A step whose terms carry `K` on its tilde operands then serves the residual's adjoint sweep
(`K = 0`) and the `Jp`-seeded one (`K` of the call) with one literal string; a stray or missing `K` axis still raises,
node by node. Every call still passes one whole record (row 5.4). Its fields are `J`, `W`, `K`, `C`; `J` is the number of jet axes, any nonnegative integer.
The output's jet shape is not a field of it: it differs between the calls of one computation (a leaf has the
input's jet sizes, a sweep variable the full one), so it is an argument of each call. The jet sizes of the
operands are not in it either: they are read off the arrays, whose shapes are static under `jit`.

**The algorithm in Part 1** (entry 117 and the brief's item 6: a simple correct algorithm for every case, and
today's cost in the cases the library uses). Which operand is carried and which are taken one coefficient at a
time is read off the operands' static shapes, never off the string:
- *Product* (`J` in the output; two or more jet operands). The carrier is the jet operand with the most slots (the
  product of its jet sizes, each operand counted after it is clipped to the box, since slots beyond the box are
  never read; made explicit 2026-11-19, at the start of P4a), the first in the string among equals; it keeps `J` through the contraction. Every
  other jet operand is taken one multi-index at a time, `J` deleted from its term: for each combination of their
  multi-indices `r` that lies inside the box, one `contract` of the carrier, sliced to the part of the box above
  `r`, with those coefficients, added into the result at offset `r`. Its cost is one contraction per multi-index of
  the partners, so a partner of two slots costs two, as today, and a full partner `order + 1`.
- *Readout* (no `J` in the output). The two jet operands with the most slots (counted after clipping to the box of the corner, `order + 1` on
  each axis; first in the string among equals) form the pair, folded in one
  `contract` call along all their jet axes, the second flipped on every jet axis and both sliced to the ranges
  that meet at the corner (`sum_k a[k] b[c - k]`); every other jet operand is taken one multi-index at a time, with
  the corner reduced by its index. One jet operand alone gives its corner coefficient, one slice. Its cost is one
  contraction per multi-index of the operands beyond the pair, as today.
- *`over_orders`* names how an operand taken one coefficient at a time is taken when it is a full partner: one
  with as many slots as the box along its jet axis, once it is sliced to the box (a partner with more slots than
  the box is sliced first, since slots beyond the box are not read, and is then a full partner). `'scan'`: the carrier rides the carry and is shifted one slot
  per step, through `xscan` (a real `lax.scan` on jax, a Python loop on numpy), so the compiled program has one size
  at every order; the readout with a third full operand scans it likewise. `'loop'`: a Python loop with slices. A
  partner with fewer slots than the box (a leaf, an input) is looped under both forms, as today's affine partner
  is. **For `m >= 2` both forms run the loop in Part 1** (the seam of section 1, and O-38): a scan over more than
  one jet axis is Part 2's.
- With one jet variable, a line input and a full order this makes exactly the contractions of P1's `jet_contract`
  on the corresponding `T`/`A` strings: one for a ride, two for a product with a leaf, `order + 1` for two full
  jets; a readout makes no more than before (the `V` strings with an input fold the trunk and the input in one call,
  where P1 made two). P4a pins the counts by counting `contract` calls.
- *Not chosen:* the product taken along each jet axis in turn (the ring is a tensor product). For a contraction it
  needs an intermediate that carries the remaining jet axes of both operands at once, which costs more than the loop
  over the partner's multi-indices and saves nothing in the cases the library uses.

`jet_contract` knows nothing of the layout; it reads whether its operands are jax, as every function does. The
caller says which form: in Part 1 the engine passes `'scan'` on the uniform layout and `'loop'` on the ragged one,
which is how the library runs today. The two forms agree to rounding, not bit for bit. Nothing is reshaped, and
only jet axes are sliced, so every sub-axis of `W`, `K`, `C` shards free, as with `contract`.

**The strings in Part 1.** The 41 strings of P1 (`tests/_jet_contract_cases.py`, `STRINGS`, seven families) are
written with `J` in place of both `T` and `A`. Strings that differed only in `A` against `T` become one: 41 become
29, by family `xi` 5, `z` 3, `mu` 4 (from 8), `nu` 4 (from 8), `eta` 4, `V` 5, `H` 4 (from 8). Of the 16 strings
that carried `A`, 12 merge with a string that carried `T` (the four of `mu`, of `nu` and of `H` with an affine leaf);
4 stay distinct (two of `xi` and two of `V`, where the input carries no `C` and the seed does). Derived by
replacing the letters in P1's table and removing repeats (account, section 4). A string no longer shows which
operand is a leaf (a cost named in entry 116); the comment at the call site says it. `GroupLengths` has four
fields, `J`, `W`, `K`, `C`, and `T`, `A` and `S` are not letters a string may name.

The helpers that make input jets (entries 118 to 120; names approved, section 11.4), in `backend/jets.py`:

```python
def affine_jet(
        ww:           typ.Sequence[NDArray],           # len=d, elm_shape=W+(Ni,): the point
        *directions:  typ.Sequence[NDArray],           # m of them, each len=d, elm_shape=W+(Ni,)
) -> typ.Tuple[NDArray, ...]:                          # len=d, elm_shape=(2,)*m+W+(Ni,): w + s_1 p_1 + ... + s_m p_m
                                                       #   slot 0 is w, slot e_j is p_j, every other slot zero

def index_to_one_hot(
        index:  NDArray,                               # int, shape=(d,)+W
        shape:  typ.Sequence[int],                     # the mode sizes N_i, len=d
) -> typ.Tuple[NDArray, ...]:                          # len=d, elm_shape=W+(Ni,): e_index, in the default float dtype
```

`affine_jet(ww, pp)` is the input jet of the line `w + s p`, of jet sizes `(2,)` (jet shape `(1,)`): the call that was
`probe(ww, pp, order=3)` is `probe(affine_jet(ww, pp), order=(3,))`. With no direction it returns `ww` unchanged
(`m = 0`). It checks nothing beyond what it needs to stack (each direction has the shape of `ww`, structurally).
It returns the form it is given: for a tuple or a list, a tuple, one array per mode; for packed vectors, one array
`(d,) + W + (N,)` (with directions packed alike), one packed array `(d,) + (2,) * m + W + (N,)`. The rule of entry 115
(a tuple, mode by mode) is for the functions whose result depends on `kind`; `affine_jet` has no kind and acts on
each mode alike, so it keeps the packed form, and the uniform twins, which mirror the form of their sample, then
return a packed output as today. `index_to_one_hot` returns a tuple; a user of the uniform layout packs it, or the
twins pack it on entry.
`index_to_one_hot` makes apply's sample at a grid point: what `entries_derivatives(index, pp)` computed is
`apply(affine_jet(index_to_one_hot(index, shape), pp), order=(n,))`. A user with a curve, or with several directions
of their own, builds the input jet directly; nothing else is needed. The dtype of the one-hot vectors is the
library's default float (the dtype rule of 5.6 then lowers it to the cores' where they are float32); it has no
`dtype` argument (O-41).

The input's slots are Taylor coefficients, like every other jet of the library (rows 2.1 and 2.2): `ww[i]` at the
multi-index `k` is the coefficient of `s^k`. A user whose curve is given by derivatives converts it with
`derivatives_to_taylor` first.

The functions of the dual jet and of derivative data take `kind` first and `order` after it (the maintainer,
2026-11-17, 12:02: such a function is three mathematical functions in one, so the caller says which). The type of
the data is decided by `kind`, never by the type of the object. Below, `F = tuple(n + 1 for n in order)`; the
entries kind has `order = ()` only, and its jets have no jet axes.

```python
def jet_to_dual_jet(
        kind:    str,                      # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],      # the jet shape; entries: ()
        jet:     typ.Any,                  # of the type of z: probe len=d, elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C; entries W+[K]+C
        weight:  typ.Optional[NDArray],    # probe (d,)+F; apply F; entries (); None = ones
) -> typ.Any:                              # the dual jet, same shapes: omega^2 * jet, then every jet axis flipped

def dual_jet_to_jet(
        kind:      str,
        order:     typ.Tuple[int, ...],
        dual_jet:  typ.Any,                # same shapes as jet_to_dual_jet's result
        weight:    typ.Optional[NDArray],  # probe (d,)+F; apply F; entries (); None = ones
) -> typ.Any:                              # the jet: every jet axis flipped back, then divided by omega^2
                                           #   (where a weight is zero it divides by zero: no check, O-20)

def derivatives_to_taylor(
        kind:  str,
        order: typ.Tuple[int, ...],
        jets:  typ.Any,                    # derivatives: probe len=d, elm_shape=F+W+(Ni,); apply F+W
) -> typ.Any:                              # Taylor coefficients, same shapes: slot k divided by k_1! ... k_m!

def taylor_to_derivatives(kind, order, jets)    # the reverse: slot k multiplied by k_1! ... k_m!

def derivative_weight_to_taylor_weight(
        weight:  NDArray,                  # a weight on derivative residuals: probe (d,)+F, apply F
        order:   typ.Tuple[int, ...],
) -> NDArray:                              # same shape: slot k multiplied by k_1! ... k_m!

def prepare_derivative_data(
        kind:        str,                      # 'probe' | 'apply' | 'entries' (entries: order () only)
        order:       typ.Tuple[int, ...],
        y:           typ.Any,                  # derivatives, of the type of z
        weight:      typ.Optional[NDArray] = None,   # on derivative residuals; None = ones on the derivatives
        validation:  typ.Optional[typ.Any] = None,   # validation derivatives, or None
) -> typ.Tuple[
        typ.Any,                               # y in Taylor coefficients
        NDArray,                               # the Taylor weight: k_1! ... k_m! per slot when weight is None
        typ.Optional[typ.Any],                 # validation in Taylor coefficients, or None
]:
```

Why the turn is a flip of every axis and the factor a product of factorials: the ring is the tensor product of
one-variable rings, so the pairing `sum_k omega_k^2 r_k a_k` is the corner coefficient of `a` times the weighted `r`
flipped on every jet axis (checked numerically, account section 3), and the mixed derivative
`d^k / ds^k` of `s^k` is `k_1! ... k_m!`. The functions of the entries kind accept `order = ()` only and do
nothing to the jet axes (there are none); for it the functions of derivative data return the data and the weight as
they are (a weight of ones when none is given).

The checkers of a sample, its data and its weight. Each raises a `ValueError` naming what is wrong and returns
nothing; shapes are static, so each is safe under `jit`. They are checkers, not guarantees: no backend function
calls them, the frontend does (section 2.5), and `LeastSquaresOracle.check_batch` is built from them.

```python
def check_sample(
        kind:    str,                      # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],      # the jet shape; entries: ()
        sample:  typ.Any,                  # ww (probe, apply) | index (entries)
        shape:   typ.Sequence[int],        # the mode sizes N_i, len=d
) -> None:                                 # d arrays J_in_i+W+(Ni,), each with len(order) jet axes of size >= 1 and
                                           #   one W for all; or index (d,)+W, integer

def check_data(kind: str, order: typ.Tuple[int, ...], sample: typ.Any, y: typ.Any) -> None:
                                           # y is F+W+(Ni,) per mode (probe), F+W (apply), W (entries)

def check_weight(kind: str, order: typ.Tuple[int, ...], weight: typ.Optional[NDArray], d: int) -> None:
                                           # None, or exactly (d,)+F (probe), F (apply), () (entries)
```

A shape cannot say how many of an input's leading axes are jet axes: `check_sample` reads the first `len(order)`
axes of every `ww[i]` as jet axes and the axes between them and the last as `W`, and requires one `W` for all modes.
A sample of the wrong form for the order is caught when the modes' `W` then disagree, or the count of axes is too
small; an input of jet sizes `(2,)` passed with `order = ()` and a `W` of one more axis is not distinguishable from
a two-axis sample stack, and the docstring says so (the cost of never inferring `order`, entry 119).
`check_perturbation_vectors` and `check_perturbation_index`, which P3 copied into `jets`, go from it in P4d: there is
no `pp`. The originals stay in `sampling_derivatives`, called by the old frontend methods until P13, and go in
P20. The logic of `check_sample` and `check_data` replaces today's frontend `fitting._check_ragged_sample`, which
goes with the factories.

```python
def data_block_norms(
        kind:   str,
        order:  typ.Tuple[int, ...],
        y:      typ.Any,                   # probe len=d, elm_shape=F+W+(Ni,); apply F+W; entries W
) -> NDArray:                              # the shape of the weight: probe (d,)+F; apply F; entries ()

def equalizing_weight(
        block_norms:  NDArray,             # from data_block_norms
        floor:        float = 1e-6,        # relative to the largest block
) -> NDArray:                              # same shape: 1 / max(x, floor * max(x))

def rescale_direction(
        kind:    str,                      # 'probe' | 'apply'
        order:   typ.Tuple[int, ...],      # at least one jet variable
        sample:  typ.Any,                  # ww, input jets: J_in_i+W+(Ni,)
        y:       typ.Any,                  # of the type of z
        length:  typ.Any,                  # a number, or one per jet variable (len=m): slot k of the input jets
                                           #   and of y times length_1**k_1 ... length_m**k_m   (loss-only)
) -> typ.Tuple[
        typ.Any,                           # the sample
        typ.Any,                           # y
]:

def rescale_coordinates(
        kind:    str,
        order:   typ.Tuple[int, ...],
        sample:  typ.Any,                  # ww, input jets; probe and apply only
        y:       typ.Any,                  # of the type of z
        scales:  typ.Sequence[NDArray],    # len=d, elm_shape=(Ni,): coordinate j of mode i times scales[i][j]
) -> typ.Tuple[
        typ.Any,                           # the sample: every slot of every input jet scaled
        typ.Any,                           # y (for probe, mode i's data divided by scales[i] too: 6.7 note)
]:

def t3_unscale_coordinates(
        x:       typ.Tuple,                # (tucker_cores, tt_cores) fitted in rescaled coordinates
        scales:  typ.Sequence[NDArray],    # as given to rescale_coordinates
) -> typ.Tuple:                            # the same tensor in the original coordinates

def ut3_unscale_coordinates(x, scales)    # the uniform twin: x the uniform cores' .data; the same scales
```

`rescale_direction` in several variables: a new variable `s_j = length_j s'_j` multiplies the coefficient of
`s^k` by `length_1^k_1 ... length_m^k_m`, in the input and in the output alike, so the result is the data of the
same tensor at the new sample. With a line input it is today's rule (the direction times `length`, slot `t` of `y`
times `length^t`). A single number, the default form, is the same length for every variable; one number per jet
variable is also accepted (O-40). It changes the loss only. `t3_unscale_coordinates`, its twin, `equalizing_weight` and
`check_coordinate_scales` are unchanged.

`scales` is one scale per coordinate, `len=d`, `elm_shape=(Ni,)` (O-35), as the prototype
`scripts/review/bad_scaling_example.py` has it: its problem has coordinate scales spread over four decades within a
mode, which one number per mode cannot undo.

The two estimates the prototype reads off the data (a length for the directions, a coordinate scale) are not
shipped: the example computes them inline (O-14: "compute inline, do not ship at this time").

The dense functions, beside their kinds, in the form of N16 (O-27, O-28). They are the dense ground truth of the
library: the reference module calls them for `z`, and the examples and the tests use them. The jets of a dense
tensor at input jets are exact: the tensor is contracted with the input jets one mode at a time, each a truncated
product of polynomials written as explicit loops over the multi-indices of the input's slots, with `numpy` or
`jax.numpy` and never `jet_contract` (so that the ground truth shares no code with what it checks). They keep today's
`dense_probe` stacks (`W` on the sample, `C` on `T`). The frontend does not wrap them: it has no class for a dense
array.

```python
def dense_probe(
        T:      NDArray,                              # shape=C+(N0,...,N(d-1))
        ww:     typ.Sequence[NDArray],                # len=d, elm_shape=J_in_i+W+(Ni,): input jets
        order:  typ.Tuple[int, ...] = (),             # the jet shape; F=tuple(n+1 for n in order)
) -> typ.Tuple[NDArray, ...]:                         # z. len=d, elm_shape=F+W+C+(Ni,)

def dense_apply(T, ww, order=()) -> NDArray:          # F+W+C
def dense_entries(T, index) -> NDArray:               # index (d,)+W, integer; W+C. No jets (entry 120)
```

```python
def mu(kind: str, frame, sample, i: int, order=(), *, variations=None, core_directions=None,
       w_tilde=None) -> NDArray:           # sampling_definitions: mu[i] by its definition, dense, no sweep
```

One such function per name of the table, all with the same arguments, each built as the dense contraction of
its subnetwork (`SAMPLE_VARIABLES.md`, the column "As a subnetwork"). Jets come from interpolation in `s`, which is
exact for a polynomial (9.3): in several variables, evaluation on the tensor grid of roots of unity, `M_j` nodes for
variable `j`, and a discrete Fourier transform along each node axis. Every variable is a polynomial in `s_j` of
degree at most `d * (a_j - 1) + n_j`, with `a_j` the largest input size along axis `j` clipped to `n_j + 1`, so `M_j
= d * (a_j - 1) + n_j + 1` nodes are exact (the inputs are clipped to the box first; their higher slots never reach
a variable). With a line input that is P2's `d + order + 1`. `z` comes from the dense tensor through `dense_probe`,
`dense_apply` or `dense_entries`; `Jp` from the dense tangent tensor; `g`, `JtJp` and the tilde variables from the
dense Jacobian; `Kp` from the second derivative of the dense objective (by the house dispatch rule, inferred from
the arrays: finite differences, which are exact for this polynomial, on numpy arrays; jax's second derivative on jax
arrays). Each variable is returned with its declared jet sizes (section 2.3). Ragged layout only; small problems
only (the node grid has `M_1 ... M_m` points). The module is backend only.

The checkers of a frame (4.2; O-5, O-6). Each returns a residual and never raises and never reads a mode; the
frontend compares it with a tolerance in safe mode (section 2.5).
- `fv_frame_consistency_residual(frame)`: today's signature and formula, `||T(D, L) - T(D, R)|| /
  max(1, ||T(D, R)||)`, through the cores (P4).
- `fv_centers_consistency_residual(frame, centers)`: at each site, the tensor with `Vc[i]` under
  `U[i]` (and with `Hc[i]` over `D[i]`), `L` cores to the left and `R` cores to the right, against `T(D, L)`. No
  frontend function calls it (O-4: no frontend user supplies centers); it is there for raw-data users and the
  tests.
- `ufv_frame_consistency_residual` and `ufv_centers_consistency_residual`, the uniform twins, in
  `backend/ufv_operations.py` beside `ufv_frame_orthogonality_residual`.

```python
class Oracle(typ.Protocol):                # the fitting layer's extension point (3.4, 8.2)
    optional_outputs: typ.ClassVar[typ.Tuple[str, ...]]   # which of the optional outputs it supplies (O-7)

    def query(
            self,
            frame:            typ.Tuple,                      # a manifold frame or a corewise point
            batch:            typ.Any,                        # opaque to the optimizers; None if it has no data
            want:             typ.Tuple[str, ...],            # required: f, g, JtJp, Jp_norm_sq; optional: Kp, Jp, r,
                                                              #   r_block_sumsq, y_block_sumsq, f_resolution
            Vc:               typ.Optional[typ.Sequence[NDArray]] = None,
            variations:       typ.Optional[typ.Tuple] = None, # (V, H): JtJp, Jp_norm_sq, Jp
            core_directions:  typ.Optional[typ.Tuple] = None, # (D_dot, L_dot): Kp
            held:             typ.Any = None,                 # what an earlier call at this frame returned (O-8)
    ) -> typ.Tuple[
        typ.Tuple[typ.Any, ...],           # values, aligned with want; blocks before projection
        typ.Any,                           # held: any object; in a compiled program one jax can carry (O-30)
    ]: ...

    # optional: check_batch(batch, shape) -> None. The frontend calls it where a batch enters when the oracle has
    # it, and checks nothing about the batch otherwise (O-29).

@dataclass(frozen=True, eq=False)
class LeastSquaresOracle(ValueHashedFields):
    kind:   str                            # 'probe' | 'apply' | 'entries'
    order:  typ.Tuple[int, ...]            # the jet shape; required, no default (O-44); entries: () only (O-46)
    # query(...): one call of sampling_variables. Its batch is a LeastSquaresBatch, its held a HeldVariables.
    # check_batch(batch, shape) -> None: the form of the sample, its sizes against shape, y, the weight (O-18)
    # n_measurements(batch) -> int;  take(batch, indices) -> LeastSquaresBatch (traceable, static len(indices))
```

On a corewise frame (two cores, told from its structure, section 2.2) the least-squares oracle passes the direction
it is given as `core_directions`, the engine refusing `variations` there (P8; ruled 2026-11-20). The least-squares
oracle's `query` asks the engine for `want` plus the names of its hold list that the declared
graph says the call computes anyway, and returns those as `held`; a call never computes a variable only to hold
it (8.3). Part 1's hold list is fixed: `xi`, `mu`, `nu`, `eta`, `r` (7.2 "as today"). `y_block_sumsq` needs no
frame (the display asks once). `f_resolution` is `eps (|omega y| |omega r| + |f|)`, the formula of
`_objective_resolution` today, with `|omega y|` from the same pass. `n_measurements` and `take` are methods of the
least-squares oracle, not of the protocol (8.3 part 2); `check_batch` is an optional member of the protocol (O-29),
which the least-squares oracle has. `check_batch(batch, shape)`, with `shape`
the mode sizes, calls `check_sample`, `check_data` and `check_weight`; it is a checker like them: `query` does not
call it.

**The minibatch scale** (O-21) belongs to the batch's inner product on data space, `<a, b> = scale * <a, omega^2
b>`: a drawn batch is a problem of its own, scaled so that it is a stochastic estimator of the full one. So the
least-squares oracle multiplies `f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp` and `f_resolution` by `minibatch_scale`, and
nothing else: `r` and `Jp` are the plain vectors at the drawn samples, and `r_block_sumsq` and `y_block_sumsq` are
plain sums of squares. The identities hold on a drawn batch: `f = 1/2 <r, r>`, `<Jp, r> = <p, g>`,
`Jp_norm_sq = <Jp, Jp>`.

```python
def pack_batch(                            # backend/uniform_fitting.py
        kind:   str,
        order:  typ.Tuple[int, ...],
        batch:  LeastSquaresBatch,         # ragged
        N:      int,                       # the packed width
) -> LeastSquaresBatch:                    # the sample's vectors and the probe data packed (pack_sample, pack_data)
```

```python
def local_model(
        geom,                                 # a backend Geometry
        oracle,                               # an Oracle
        batch,                                # its batch
        x_cores,                              # the point
        regularizer=None,
        hessian:            str = 'gauss_newton',          # or 'exact'
        curvature_damping:  typ.Optional[float] = None,
        frame:              typ.Any = None,                # a given frame of x_cores (padded start), else geom.frame
) -> LocalModel:                              # checks nothing: a given frame is the caller's to vouch for

def point_objective(
        geom,                                 # a backend Geometry
        oracle,                               # an Oracle
        batch,                                # its batch
        x_cores:      typ.Tuple,              # the raw cores (B, G), ragged or the uniform supercore pair
        regularizer=None,
) -> NDArray:                                 # f (+ rho(x)), shape C: a corewise query of 'f' at x_cores
```


**Changed.** Each keeps its name and calls the engine, with the merged signature of 6.6 in the form N16 decides
(the note "The names of Part 1, and the form of the merged sampling functions" in "The two parts"). One rule: the
object the function is evaluated at comes first (the cores `x`, or the `frame`, with the `variations` after it for
the tangent forward), then the sample (`ww` or `index`), then what is applied (the `residual`, then the
`core_directions`); then the keyword `order=()` (no `pp`: the sample is input jets; the entries functions have no `order`, entry 120); then `sum_over_samples=True` where there is a sum. One
name for one thing, on both layouts: `x` for a Tucker tensor train's cores (today `x`, `data` or `core_pair`),
`frame` (today `frame` or `frame_data`), `variations` (today `variation` or `variations_data`), `ww` (today `ww` or
`vecs`) or `index`, `residual` (today `ztildes` or `c`), `core_directions`. The sum flag is `sum_over_samples=True`
in place of `sum_over_probes=False` (3.7 note; N3, N25).

| Function, today | Becomes |
| --- | --- |
| `t3_probe(ww, x)`, `ut3_probe(ww, data)` | `t3_probe(x, ww, order=())`, `ut3_probe` likewise |
| `t3_apply(x, vecs)`, `ut3_apply(data, vecs)` | `t3_apply(x, ww, order=())`, `ut3_apply` likewise |
| `t3_entries(x, index)`, `ut3_entries(data, index)` | `t3_entries(x, index)`, `ut3_entries` likewise: no `order` (entry 120) |
| `tv_probe(ww, variation, frame)`, `utv_probe(ww, frame_data, variations_data)` | `tv_probe(frame, variations, ww, order=())`, `utv_probe` likewise; apply likewise; `tv_entries(frame, variations, index)` |
| `tv_probe_transpose(ztildes, ww, frame, sum_over_probes=False)`, `utv_probe_transpose(ztildes, ww, frame_data, ...)` | `tv_probe_transpose(frame, ww, residual, order=(), sum_over_samples=True)`, `utv_probe_transpose` likewise; apply (`c`) likewise; `tv_entries_transpose(frame, index, residual, sum_over_samples=True)` |
| `t3_probe_corewise_transpose(ztildes, ww, core_pair, sum_over_probes=False)`, `ut3_probe_corewise_transpose(ztildes, ww, data, ...)` | `t3_probe_corewise_transpose(x, ww, residual, order=(), sum_over_samples=True)`, the twin likewise; apply likewise; entries without `order` |
| `t3_probe_corewise_hessian(ztildes, ww, core_pair, core_directions)`, `ut3_probe_corewise_hessian(ztildes, ww, data, core_directions)` | `t3_probe_corewise_hessian(x, ww, residual, core_directions, order=())`, the twin likewise; apply likewise; entries without `order` |

The transposes take a residual of the type of `z` and no weight and turn it inside: by 6.6 the backend function
behind a frontend method has the method's signature, and by 2.4 and 3.6 the frontend transposes take a residual and
no weight. The corewise Hessian takes the same (O-22): a vector `c` of the type of `z`, no weight, turned inside.
The two are then one definition: for the scalar function `phi(x) = <c, z(x)>` with the plain inner product, the
corewise transpose is the gradient of `phi` at the cores `x`, and the corewise Hessian is its second derivative
applied to `core_directions`. The curvature term of a fit is the case `c = omega^2 r`. The corewise Hessian has no
frontend method and stays: it names a capability a backend user would not guess from the engine. The three ambient transposes keep their own code (6.3)
and take the rule's order and names, with the sum flag off by default (the maintainer, 2026-11-16, 18:35):
`t3_probe_ambient_transpose(ww, residual, sum_over_samples=False)`,
`t3_apply_ambient_transpose(ww, residual, sum_over_samples=False)`,
`t3_entries_ambient_transpose(index, shape, residual, sum_over_samples=False)`. They have no object to be evaluated
at, and no `order`; their `ww` is plain vectors (the ambient transposes stay without jets, and no jets are planned for them; O-43). Their
default differs from the other transposes' on purpose: for them the sum folds `W` into the CP rank. `dense_probe`
takes the rule too, and `order` (above).

```python
def t3_probe(
        x:      typ.Tuple,                            # (tucker_cores, tt_cores), elm_shapes C+(nDi,Ni), C+(ri,nDi,r(i+1))
        ww:     typ.Sequence[NDArray],                # len=d, elm_shape=J_in_i+W+(Ni,): input jets
        order:  typ.Tuple[int, ...] = (),             # the jet shape; F=tuple(n+1 for n in order)
) -> typ.Tuple[NDArray, ...]:                         # z. len=d, elm_shape=F+W+C+(Ni,)

def tv_probe(
        frame:       typ.Tuple,                       # (D, U, L, R)
        variations:  typ.Tuple,                       # (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        ww:          typ.Sequence[NDArray],           # len=d, elm_shape=J_in_i+W+(Ni,)
        order:       typ.Tuple[int, ...] = (),
) -> typ.Tuple[NDArray, ...]:                         # Jp. len=d, elm_shape=F+W+K+C+(Ni,)

def tv_probe_transpose(
        frame:             typ.Tuple,                 # (D, U, L, R)
        ww:                typ.Sequence[NDArray],     # len=d, elm_shape=J_in_i+W+(Ni,)
        residual:          typ.Sequence[NDArray],     # of the type of z: len=d, elm_shape=F+W+[K]+C+(Ni,)
        order:             typ.Tuple[int, ...] = (),
        sum_over_samples:  bool = True,               # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                      # V blocks. len=d, elm_shape=[W]+[K]+C+(nUi,Ni)
        typ.Tuple[NDArray, ...],                      # H blocks. len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))
]:

def t3_probe_corewise_hessian(
        x:                typ.Tuple,                  # (B, G): the point whose cores the directions vary
        ww:               typ.Sequence[NDArray],      # len=d, elm_shape=J_in_i+W+(Ni,)
        residual:         typ.Sequence[NDArray],      # c, of the type of z: len=d, elm_shape=F+W+C+(Ni,); no weight
        core_directions:  typ.Tuple,                  # (B_dot, G_dot), shapes of B, G with K+C
        order:            typ.Tuple[int, ...] = (),
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                      # Tucker block. len=d, elm_shape=K+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                      # TT block.     len=d, elm_shape=K+C+(ri,nDi,r(i+1))
]:
```

`t3_probe_corewise_transpose(x, ww, residual, order=(), sum_over_samples=True)` is the transpose on a
corewise call and follows the layout above. The uniform twins (`ut3_probe`, `utv_probe`, `utv_probe_transpose`,
`ut3_probe_corewise_transpose`, `ut3_probe_corewise_hessian`, and apply and entries) take the same lists, with `x`
the uniform cores' `.data` and `frame` the uniform frame's; they check packing and call the engine, which masks
once on entry and re-masks core-shaped outputs on exit (O-3). They mirror the form of the sample as today
(`backend/ut3_sampling.py`, `ut3_probe`): a ragged `ww` (`len=d`, `J_in_i + W + (Ni,)`) gives a ragged output, a
packed `(d,) + J + W + (N,)` gives a packed output. To pack input jets they call `ut3_operations.pack_if_ragged`
with `len_J=len(order)`; `pack_vectors` and `pack_if_ragged` (public in `ut3_operations`) gain the keyword
`len_J=0`, the number of leading jet axes brought to their largest size across the modes with zeros, every other
axis still required to agree (P11; N41). Every positional call of a function
whose order changes moves, and every call by keyword of a renamed argument breaks; section 5.3, P12, lists them.

In `backend/optimizers.py`: `LocalModel` (section 2.2),
`newton_cg(geom, oracle, batch, x0, regularizer=None, ...)` (8.3), and the other three with the same head and, as
N30 has it, `regularizer=None` after their positional arguments, today's keywords following in today's order:
`mc_sgd(geom, oracle, batch, x0, rng, batch_size, regularizer=None, draw=None, ...)`, `adam(...)` likewise,
`gradient_descent(geom, oracle, batch, x0, regularizer=None, ...)`, `flat_draw(oracle, batch, batch_size)` returning `draw(rng) -> batch`. The
`Geometry` protocol: `frame(x)` returns `(frame, centers)` with `centers = (Vc, Hc)` or `None`, on the backend
protocol only (O-4); the corewise geometries' frame is the point `(B, G)`. The geometry has no consistency member
(O-31): a backend user checks a frame with `fv_frame_consistency_residual` or its uniform twin. In
`optimizer_display.py`:
`make_newton_display(oracle, batch, val_batch=None, ..., error_table=True)` and
`format_newton_iter(info, train_err, val_err=None, ..., error_table=True)`; `relative_errors` and the table take
arrays of the shape of the weight. For two or more jet variables the table has one column per multi-index, in
row-major order (the order of unfolding the array of the weight's jet axes into columns), and no other layout
(O-42). With `error_table=False` each iteration prints its header lines and no table; the records keep
`train_err` and `val_err` in full either way. Today the table is printed by `format_newton_iter`
(`backend/optimizer_display.py`), which the callback of `make_newton_display` calls; the backend `newton_cg` has no
`verbose` and prints through its `callback=` only; the frontend `newton_cg` builds that callback when `verbose=True`
(`t3toolbox/optimizers.py`, lines 906 to 911). `gradient_descent`, `mc_sgd` and `adam` have no `verbose` and no
table, in either tier. So `error_table` is added exactly at those three places: the frontend `newton_cg`, beside
`verbose`, and `make_newton_display` and `format_newton_iter` in the backend. In `uniform_fitting.py`: `pack_sample(kind, order, sample, N)`, `pack_data(kind, order, y, N)` and `pack_batch`
(above). The backend `LocalModel`, `local_model` and the optimizers check nothing numerical (O-6), as
`backend/optimizers.py` already says of itself.

**Removed.** Counted from the `__all__` lists of the six sampling modules, 166 public names today:

| Module | Today | Changed (call the engine) | Kept as they are | Removed |
| --- | --- | --- | --- | --- |
| `probing` | 34 | 5 | 2, in the form of N16, with their own code: `t3_probe_ambient_transpose`; `dense_probe`, which also gains `order` and takes input jets | 27: every `compute_*`, `assemble_*`, `tv_precompute_probe_frame_sweep`, the three `*_from_sweep` |
| `apply` | 10 | 5 | 1 (`t3_apply_ambient_transpose`, its arguments by N16) | 4: the sweep and `*_from_sweep` functions |
| `entries` | 10 | 5 | 1 (`t3_entries_ambient_transpose`, its arguments by N16) | 4: likewise |
| `ut3_sampling` | 25 | 9 | 1 (`ut3_full_sum`) | 15: every `*_derivatives*` and `*_from_sweep` |
| `utv_sampling` | 30 | 6 | 0 | 24: the sweeps, `*_from_sweep`, every `*_derivatives*` |
| `sampling_derivatives` | 57 | 0 | 0 (P3 copied `check_perturbation_vectors` and `check_perturbation_index` into `jets`; P4d deletes the copies, there being no `pp`) | 57, the module with them (the chunk estimators and `dense_*_derivatives` included; the dense jets are `dense_probe` and `dense_apply` with input jets and `order`) |
| **Total** | 166 | 30 | 5 | 131 |

New in these modules: `dense_apply` in `apply`, `dense_entries` in `entries`.

Also removed: in `backend/fitting.py` all 20 names (the nine kind classes, the three singletons, the four kind
factories, the four `sumsq` helpers); in `backend/uniform_fitting.py` 9 of 12 (the six uniform kinds,
`uniform_sampling_kind`, `uniform_derivatives_kind`, `uniform_least_squares_problem`); in
`backend/optimizers.py` `Problem`, `least_squares_problem`, `_minibatch_step_problem`; in `t3toolbox/fitting.py`
the private `_canonical_weight` and `_check_ragged_sample` (O-18); in
`backend/regularization.py` the private `_ScaledRegularizer` (defined there, imported by `optimizers.py`).

### 2.5 The public surface, frontend

| Class | Sampling methods today | After | Removed |
| --- | --- | --- | --- |
| `TuckerTensorTrain` | 15 | 9: `entries`, `apply`, `probe` and their `*_corewise_transpose`, probe and apply with `(…, order=())`, entries with no `order`; the three ambient transposes with their arguments by N16 | the 3 `*_derivatives`, the 3 `*_corewise_derivatives_transpose` |
| `T3Tangent` | 12 | 6: `probe`, `apply`, `entries` and their `*_transpose`, merged | 6 |
| `UniformTuckerTensorTrain` | 12 | 6 | 6 |
| `UT3Tangent` | 12 | 6 | 6 |

Each merged method returns `F + W + [K] + C + ...` with the jet axes outermost, Taylor coefficients (6.6). The methods follow
the rule of N16 (section 2.4) with `self` as the object: the sample, then what is applied, then `order=()` for
probe and apply (6.6: the transposes take the same `order`; entries has none), then `sum_over_samples=True` where
there is a sum. The
tangent transposes stay static methods with the frame first: their input is a residual and their output a tangent,
so they are constructors, like `T3Tangent.zeros(frame)` (N16). Besides the order, three things change:

- the residual argument (`ztildes` or `c` today) is named `residual`, and `TuckerTensorTrain.apply`'s `vecs` is
  named `ww` (N16, N19);
- `sum_over_probes=False` becomes `sum_over_samples=True` (3.7 note: the transposes sum by default; the name is N25);
- the six `*_derivatives` methods of each class go (6.6).

Today's forms, as defined in `t3toolbox/tucker_tensor_train.py`, `manifold.py`, `uniform_tucker_tensor_train.py`
and `uniform_manifold.py`, and what each becomes:

| Class | Method, today | Becomes |
| --- | --- | --- |
| `TuckerTensorTrain`, `UniformTuckerTensorTrain` | `probe(ww)` | `probe(ww, order=())`; `ww` input jets |
| | `apply(vecs)` | `apply(ww, order=())` |
| | `entries(index)` | `entries(index)`, unchanged in form |
| | `probe_corewise_transpose(ztildes, ww, sum_over_probes=False)` | `probe_corewise_transpose(ww, residual, order=(), sum_over_samples=True)` |
| | `apply_corewise_transpose(c, ww, sum_over_probes=False)` | `apply_corewise_transpose(ww, residual, order=(), sum_over_samples=True)` |
| | `entries_corewise_transpose(c, index, sum_over_probes=False)` | `entries_corewise_transpose(index, residual, sum_over_samples=True)` |
| `TuckerTensorTrain` only, static | `probe_ambient_transpose(ztildes, ww, sum_over_probes=False)`, `apply_ambient_transpose(c, ww, ...)`, `entries_ambient_transpose(c, index, shape, ...)` | `probe_ambient_transpose(ww, residual, sum_over_samples=False)`, `apply_ambient_transpose(ww, residual, sum_over_samples=False)`, `entries_ambient_transpose(index, shape, residual, sum_over_samples=False)`; still static, the default still off |
| `T3Tangent`, `UT3Tangent` | `probe(ww)`, `apply(ww)`, `entries(index)` | probe and apply with `order=()`; `entries(index)` unchanged in form |
| static | `probe_transpose(ztildes, ww, frame, sum_over_probes=False)` | `probe_transpose(frame, ww, residual, order=(), sum_over_samples=True)` |
| static | `apply_transpose(c, ww, frame, ...)`, `entries_transpose(c, index, frame, ...)` | `apply_transpose(frame, ww, residual, order=(), ...)`, `entries_transpose(frame, index, residual, sum_over_samples=True)` |

The sketches, one instance method, one static transpose, one corewise transpose; the others follow the same rule:

```python
    def probe(                                        # TuckerTensorTrain (instance method)
            self,
            ww:     typ.Sequence[NDArray],                    # len=d, elm_shape=J_in_i+W+(Ni,): input jets
            order:  typ.Tuple[int, ...] = (),                 # the jet shape; F=tuple(n+1 for n in order)
    ) -> typ.Tuple[NDArray, ...]:                             # len=d, elm_shape=F+W+C+(Ni,)

    @staticmethod
    def probe_transpose(                              # T3Tangent (static method, a constructor)
            frame:             'bvf.T3Frame',
            ww:                typ.Sequence[NDArray],         # len=d, elm_shape=J_in_i+W+(Ni,)
            residual:          typ.Sequence[NDArray],         # len=d, elm_shape=F+W+[K]+C+(Ni,)
            order:             typ.Tuple[int, ...] = (),
            sum_over_samples:  bool = True,
    ) -> 'T3Tangent':                                         # at frame; tangent stack [W]+[K], frame stack C

    def probe_corewise_transpose(                     # TuckerTensorTrain (instance method)
            self,
            ww:                typ.Sequence[NDArray],         # len=d, elm_shape=J_in_i+W+(Ni,)
            residual:          typ.Sequence[NDArray],         # len=d, elm_shape=F+W+C+(Ni,)
            order:             typ.Tuple[int, ...] = (),
            sum_over_samples:  bool = True,
    ) -> typ.Tuple[
            typ.Tuple[NDArray, ...],                          # Tucker-core gradients, shapes of self.tucker_cores (with [W])
            typ.Tuple[NDArray, ...],                          # TT-core gradients, shapes of self.tt_cores (with [W])
    ]:
```

Every call of a transpose or a corewise transpose moves its arguments; a call of `apply` with `vecs=` breaks
(none was found). P13 lists them.

The merged methods check the sample structurally through the backend checker `check_sample(kind, order, ww,
self.shape)` (today they check the sample's vectors against `pp`; there is no `pp`).

`TuckerTensorTrain.unscale_coordinates(scales)` returns the tensor in the original coordinates; it calls
`t3_unscale_coordinates` (O-16 and the answers of 20:15). `UniformTuckerTensorTrain.unscale_coordinates(scales)`
calls `ut3_unscale_coordinates` (O-34).

**The frames' checkers.** `T3Frame.consistency_residual` and `UT3Frame.consistency_residual` are
cached properties, like today's `orthogonality_residual`, calling `fv_frame_consistency_residual` and its uniform
twin; `is_consistent(rtol)` compares it with `rtol`. A frame used in an inner loop is contracted once.

**Fitting and optimization (frontend).** The four optimizers take `(geometry, oracle, batch, x0, ...)` (8.3), with
`val_batch` in place of `val_sample`/`val_data` (8.4), and without `kind`, `sample`, `data`, `order`, `weight`
(8.3, 2.3, 6.7) and `chunk_size` (Part 1 drops it). `mc_sgd` and `adam` take `batch_size` in place of `batch`
(note to 8.1 to 8.4). `regularizer=None` follows the positional arguments: 8.3 writes it so for `newton_cg`, and
N30 the same for the other three; every other keyword keeps today's order. The oracle is the same object in the frontend and the backend; the batch is
ragged, and the frontend packs it for a uniform `x0`. The keywords not shown are today's:

```python
def newton_cg(
        geometry,                                 # MANIFOLD / COREWISE / UNIFORM_* / a SharedGeometry over one
        oracle,                                   # e.g. LeastSquaresOracle('probe', order=(3,))
        batch,                                    # e.g. LeastSquaresBatch(affine_jet(ww, pp), y, weight)
        x0:           Point,                      # ragged or uniform: picks the representation
        regularizer:  typ.Any = None,             # e.g. IdentityRegularizer(lam)
        verbose:      bool = False,
        error_table:  bool = True,                # with verbose: False prints each row's header lines, no table
        val_batch:    typ.Any = None,             # a batch for the same oracle: the validation column
        callback:     typ.Optional[typ.Callable] = None,
        use_jit:      bool = False,               # moves x0 and the batch's arrays to jax, as today
        ...,                                      # g0_reference ... switch_track: today's keywords, unchanged
) -> typ.Tuple[
        Point,                                    # x_opt, in x0's representation
        dict,                                     # stats
]:
```

What the frontend optimizers do before calling the backend one, where a batch or a frame comes in from the
user:
- `oracle.check_batch(batch, x0.shape)` on the batch and on `val_batch`, once, when the oracle has it (O-18,
  O-29); nothing about the batch otherwise;
- the batch packed with `pack_batch` for a uniform `x0`;
- in safe mode, on a `start_frame` the caller supplied: `L` against `R` through `D`, once, through the cached
  `consistency_residual`, beside today's checks of the same frame (`is_orthogonal`, that it represents `x0`).
  A frame the library builds in the same call is not checked; neither are centers or `U` (O-5).

The six model factories of `t3toolbox/fitting.py` (and their re-exports from `t3toolbox/__init__.py`),
`least_squares_problem` and `LeastSquaresSetup` go (O-9). In their place, one frontend local model:

```python
def local_model(                                  # t3toolbox/fitting.py
        geometry,                                 # MANIFOLD / COREWISE / UNIFORM_* / a SharedGeometry over one
        oracle,                                   # e.g. LeastSquaresOracle('probe', order=(3,))
        batch,                                    # e.g. LeastSquaresBatch(affine_jet(ww, pp), y, weight)
        x:                  Point,
        regularizer:        typ.Any = None,
        hessian:            str = 'gauss_newton',  # or 'exact'
        curvature_damping:  typ.Optional[float] = None,
        frame:              typ.Any = None,        # a T3Frame / UT3Frame of x the caller supplies (O-5), or None
) -> LocalModel:                                  # tangents in and out, the same-frame guard
```

It makes the checks of the optimizers above (the batch, and in safe mode a given `frame`, once) and calls the
backend `local_model`.

**What else `t3toolbox.fitting` holds** (O-16 and the answers of 20:15: "Frontend user must be able to fit their
tensors without digging through the backend to find the helpers"):
- `LeastSquaresOracle` and `LeastSquaresBatch`, the backend classes, reachable here (one class each for both
  tiers);
- the preparation functions and the jet helpers under their backend names: `derivatives_to_taylor`,
  `taylor_to_derivatives`, `derivative_weight_to_taylor_weight`, `prepare_derivative_data`, `jet_to_dual_jet`,
  `data_block_norms`, `equalizing_weight`, `rescale_direction`, `t3_unscale_coordinates`, and the two helpers that
  make input jets, `affine_jet` and `index_to_one_hot` (the backend objects; also exported from the top level
  `t3toolbox`, O-37);
- its own `dual_jet_to_jet`, a small function: in safe mode it raises where the divisor, the squared weight in
  the precision of the computation, is zero, then calls the backend one, which divides (O-20);
- its own `rescale_coordinates`, a small function: it calls the backend checker `check_coordinate_scales` with the
  mode sizes read off the sample, then the backend function, so that wrong `scales` raise where they are passed
  (a structural check: in both safety modes and under `jit`). The checker is not re-exported.

**What the frontend does not have.** An entry to the engine; the reference module and the dense functions; a way to
write an oracle without the backend (writing an oracle is backend-level work, a stated exception to the rule of
the two users; the frontend optimizers take any oracle object, so they drive a user-written one). Section 7.5.


---

## 3. Table of names (settled)

**Settled by the maintainer on 2026-11-16, at 18:13 and 18:18** (`TARGET_DECISIONS.md`, the note "The names of
Part 1, and the form of the merged sampling functions"): every row as written, `query` for the oracle's operation
(N12), and N16 as the rule that note decides. N26 is amended by the answer to O-8: `HeldVariables` lives in
`backend/sampling_variables.py`, beside the engine. N21 is withdrawn by the answer to O-31 (the geometry has no
consistency member). The names added since were approved on 2026-11-16 at 20:59; section 11.3 records them. The
names follow `docs/naming_conventions.md` (family prefixes for functions of a representation; none for
representation-agnostic infrastructure) and `docs/contributor/naming_rules.md` (clarity wins; no generic names).

**Amended for the generalized jet group (2026-11-17, entries 117 to 120).** N16's keywords are `order=()` in place
of `pp=None, order=0`: the sample is input jets and there is no `pp` (entry 118); the entries functions have no
`order` (entry 120). N14's record has the fields `J`, `W`, `K`, `C`. The new names this change needs were approved on 2026-11-17,
in section 11.4.

| # | Placeholder | Name | Why |
| --- | --- | --- | --- |
| N1 | engine keyword `direction` | `variations` | The library's word for `(V, H)` (`T3Variations`) |
| N2 | engine keyword `core_direction` | `core_directions` | The library's word for `(D_dot, L_dot)` (the first argument of `corewise_hessian` today) |
| N3 | engine keyword `sum_samples` | `sum_over_samples` | Replaces `sum_over_probes`; matches `sumsq_over_samples` |
| N4 | the jet-to-dual-jet helper / its inverse | `jet_to_dual_jet` / `dual_jet_to_jet` | The library's grammar for free converters, `a_to_b` (`t3_to_dense`, `tv_to_t3`) |
| N5 | module of N4 and the derivative-data functions | `backend/jets.py` | Both are about the form of a jet |
| N6 | derivatives to Taylor / the reverse | `derivatives_to_taylor` / `taylor_to_derivatives` | The same `a_to_b` grammar as N4 (the draft spells it the other way round) |
| N7 | derivative weight to Taylor weight | `derivative_weight_to_taylor_weight` | As N6 |
| N8 | the one function that prepares data, validation data and weight | `prepare_derivative_data` | As in the draft |
| N9 | the five preparation functions | `data_block_norms`, `equalizing_weight`, `rescale_direction`, `rescale_coordinates`, `t3_unscale_coordinates` | The last acts on a T3, so it has the prefix |
| N10 | module of N9 | `backend/data_preparation.py` | |
| N11 | the batch record; the minibatch scale; the drawn indices | `LeastSquaresBatch`; `minibatch_scale`; `drawn_indices` | It is the least-squares oracle's batch, not a general one |
| N12 | the oracle protocol; the built-in oracle; the operation | `Oracle`; `LeastSquaresOracle`; `query` | "Query an oracle" is the optimization idiom |
| N13 | the record of held variables | `HeldVariables` | |
| N14 | the record of axis counts | `GroupLengths` | `contract` already spells them `len_<G>` |
| N15 | `jet_contract` | `jet_contract` | Already the decided spelling in 5.4; a placeholder only in the brief |
| N16 | the form of the merged functions, backend and frontend | One rule: the object the function is evaluated at first (`x`, or `frame` then `variations` for a tangent), then the sample (`ww` or `index`), then what is applied (`residual`, then `core_directions`); then the keyword `order=()` (no `pp`: the sample is input jets; the entries functions have no `order`, entry 120); then `sum_over_samples=True` where there is a sum. One name for one thing on both layouts: `x`, `frame`, `variations`, `ww` or `index`, `residual`, `core_directions`. The frontend's instance methods follow the rule with `self` as the object; the tangent transposes stay static methods with the frame first. The ambient transposes take the rule's order and names and keep the sum flag off by default (decided 18:35). Sections 2.4 and 2.5 give every form | Decided by the maintainer, 2026-11-16 (18:13 and 18:18), `TARGET_DECISIONS.md`, the note "The names of Part 1, and the form of the merged sampling functions": the order the engine and the oracle already have |
| N17 | the reference module; its functions | `backend/sampling_definitions.py`; one function named exactly as the variable (`sampling_definitions.mu`) | The module name says what they are; the function name is the variable |
| N18 | the engine's module; the steps' module | `backend/sampling_variables.py`; `backend/sampling_steps.py` | The module is named after its function, as `regularization` is |
| N19 | the residual argument of the transposes and corewise Hessians (`ztildes` or `c` today) | `residual` | 2.4 calls `ztildes` a misnomer and leaves the name to the naming questions |
| N20 | the raw-cores objective of policy a | `point_objective` | "f at raw cores" |
| N21 | the frame-consistency member of a geometry | Withdrawn (O-31, 20:59): the geometry has no such member; a backend user calls `fv_frame_consistency_residual` | |
| N22 | the page in `docs/` | `docs/sampling_variables.md` | |
| N23 | the example file | `examples/tour_sampling_variables.py` | |
| N24 | the `.tex` note | `docs/sampling_variables.tex` | |
| N25 | the frontend transposes' keyword in place of `sum_over_probes` | `sum_over_samples` | The engine's (N3) |
| N26 | the modules of the fitting records | `LeastSquaresOracle`, `LeastSquaresBatch` in `backend/fitting.py`; `HeldVariables` in `backend/sampling_variables.py` (O-8); the `Oracle` protocol, `LocalModel`, `local_model`, `point_objective` in `backend/optimizers.py` | The protocol beside `Geometry`, so `optimizers.py` stays T3-free; the least-squares pieces where the kinds are today; the held variables beside the graph they come from |
| N27 | the module of N14 and N15 | `backend/contractions.py` | `jet_contract` is `contract` with one more rule |
| N28 | the module of the uniform twins | `backend/ut3_sampling.py`, `backend/utv_sampling.py`, as today | Grouped by object type, as the naming page records |
| N29 | the script that draws the dependency diagrams | `docs/sampling_diagrams.py` | Beside the page it draws for |
| N30 | the place of `regularizer` in `gradient_descent`, `mc_sgd` and `adam`, backend and frontend | `regularizer=None` right after the positional arguments, as 8.3 writes it for `newton_cg`; in `mc_sgd` and `adam` it then comes before `draw` (today it comes after `use_jit` in the frontend, and the backend has none) | By analogy with 8.3, which fixes it for `newton_cg` only |

---

## 4. How the code is written

These are the library's rules, applied to this arc. Every slice is held to them; the reviewer checks them first.

- **One set of step functions for both layouts** (5.2). A step is named by letter and grade and serves both bases
  (5.1 note): `nu_step` given `L` cores returns `nu_L`. The 22 steps of `scripts/review/jet_steps_check.py` stand.
- **Every term is a literal string at its call site, through `jet_contract`**, with `K` on the operand that carries
  it, the record `GroupLengths` passed whole, the output's jet shape from the table of declared jet sizes (section
  2.3) and the form over the orders named (5.4, 2.9). The 29 strings of the seven families are listed in
  `tests/_jet_contract_cases.py` (`STRINGS`, rewritten in P4a); `J` is the one jet group, and no string carries a
  site group (section 2.4).
- **Dispatch** at the lowest level: `use_jax = tree_contains_jax(...)` over all inputs, `is_uniform` from the
  structure, `xnp, xmap, xscan = get_backend(is_uniform, use_jax)`. No `use_jax` argument. Uniform masks are host
  numpy and use `np`, never `xnp` (`CLAUDE.md`).
- **Scan and map bodies** are module-level and close over nothing; a static no operand carries reaches a body
  through a factory memoized on hashable scalars (`GroupLengths`, the jet shape and the jet sizes, the kind, the
  grade), never arrays
  (`docs/contributor/scan_body_principles.md`, rule 4).
- **Masks**: the uniform inputs are masked once on the engine's entry, each with the frame's own masks; steps never
  see a mask; core-shaped outputs are re-masked on its exit (`docs/contributor/uniform_internals.md`).
- **Tails**: the boundary vector is ones sized by the end bond of the cores passed; the engine squashes no tail;
  every function this arc writes or touches supports arbitrary tails (4.4). Frame construction
  (`t3_orthogonal_representations`, with its default `squash_tails=True`, called by the geometry's `frame`) keeps
  its default: 4.4 leaves that default unspoken, and 1.1 puts frame construction off limits.
- **Signatures** in the house style (`docs/contributor/signature_style.md`); body locals carry their axis layout in
  the suffix (`mu_JWCa`; `J` the jet axes, as in the strings).
- **The razor and the two users** (section 7.5): anything a raw-`.data` user would get wrong is in the backend; the
  frontend validates its objects, calls the backend's checkers, and delegates.
- **Checkers, not guarantees** (section 7.1): no backend module imports `safety`; a backend function checks only what
  it needs to run; the frontend enforces, through backend checkers.
- **The jet shape is a tuple everywhere** (`order = ()` for none); an integer raises (O-36).
- **Doctests** teach, are seeded, print structure, and are run and pasted (`docs/contributor/doctest_style.md`).
- **No auto-formatter.**

---

## 5. The work in slices

### 5.1 How the suite stays green

- **Additive first.** Slices P1 to P11 add new modules beside the old ones. Two old things change in them, each with
  its callers in the same slice: `dense_probe` takes the order of N16 and gains `pp` and `order` (P2), then input
  jets in place of `pp` (P4b), and P4's consistency residual goes through the cores, keeping its signature
  (`fv_frame_consistency_residual`, `T3Frame.is_consistent`). P4a to P4d rebuild what P1 to P3 added, which nothing
  old calls, with its tests and its few callers in the same slice. So the suite stays green by construction.
- **Then switch, one surface at a time.** P12, P13, P15, P16 and P17 each move one surface (P14 is additive). Each
  switching slice rewrites the tests, doctests, documentation pages and examples that call the surface it changes,
  in the same commit, because CI runs all of them (`.github/workflows/tests.yaml`: the suite, the module doctests,
  `getting_started.rst`, every `docs/*.md` and `docs/contributor/*.md` page except `doctest_style.md`, and every
  `examples/*.py`). The one exception is the switch of the optimizers, split in two (O-24): P16 changes the
  library and the tests, and the frontend optimizers translate the old call form for one slice; P17 moves the
  examples and the pages and deletes the translation.
- **The old kernels stay until the last slice.** After a surface switches, its old kernels are still in the tree
  and still tested by their own tests. They are deleted in P20 on the tests alone (O-1, O-2).
- **The bridge** (9.3: "the old library is a bridge used once during the port") is a temporary test file,
  `tests/bridge/test_old_kernels_bridge.py`, that compares new outputs with old code through the stated conversions
  (`t!` per slot, the turn, the sum flag), run in CI from P6 and deleted in P20 (O-25). It compares only with code
  that is never rerouted onto the engine: the functions of `sampling_derivatives` (at order 0 too), the old
  `compute_*` and `assemble_*` kernels, and the old kinds, except, from P12 on, the plain kinds' `point_forward` and
  the `Problem.objective` built on it (they call `t3_probe`, `t3_apply`, `t3_entries`, which P12 routes onto the
  engine: `backend/fitting.py`, the three `point_forward` methods).
- **Each slice's gate**: the full suite, `pytest --doctest-modules t3toolbox`, the page doctests, the examples, and
  `sphinx -W`, run unpiped on a clean tree (`docs/contributor/refactoring_methodology.md`, "Gates").
- **Per slice**: an inventory first (every inbound reference to what the slice changes, both import forms,
  `getattr` strings, slash-glob prose); exact-equality assertions in every test file the slice rewrites are sorted
  into structural (exact) and numerical (tolerance) (0.8).

### 5.2 Overview

The names are settled: the table of section 3, the names added after it (section 11.3), and the names of the
change of 2026-11-17 (section 11.4, approved that day at 16:26). Nothing is open. The order is the order of work; each slice starts when the one before it is committed
(section 12).

| Slice | Content | Size |
| --- | --- | --- |
| P0 | Baseline: the folder's check suites on the branch head; the inventory | none in the library |
| P1 | `jet_contract` with both forms over the orders, and `GroupLengths` | small |
| P2 | The reference module `sampling_definitions`; `dense_probe` in its new form, `dense_apply`, `dense_entries` | medium |
| P3 | `jets` (with the checkers) and `data_preparation`, with the uniform `ut3_unscale_coordinates`; their frontend reach | small |
| P4 | The checkers of a frame, without dense tensors; the frames' cached `consistency_residual` | small |
| P4a | `jet_contract` over one jet group `J` of any number of axes, the output's jet shape an argument; `GroupLengths(J, W, K, C)`; the 29 strings; its tests rebuilt | medium |
| P4b | The input jets and the dense functions: `affine_jet`, `index_to_one_hot`; `dense_probe(T, ww, order=())`, `dense_apply(T, ww, order=())`, `dense_entries(T, index)` | medium |
| P4c | The reference module in several variables, with input jets; entries of order `()` only | medium |
| P4d | `jets` and `data_preparation` in the jet shape: the dual jet flips every jet axis, factorials of multi-indices, the checkers of input jets, the rescalings; the validators of `pp` removed; their frontend reach | medium |
| P5 | Engine A: the declared graph, what the engine needs to run, `HeldVariables`; probe, ragged, the plain grade (`z` both routes, `r`, `f`, `r_block_sumsq`) | medium |
| P6 | Engine B: the tilde grade and `g`; the bridge; the generic twin | medium |
| P7 | Engine C: the dot grade of the frame form (`Jp`, `Jp_norm_sq`, `JtJp`) | medium |
| P8 | Engine D: the raw dots and the tilde-dot grade (`Kp`), the `nu_L` family, the corewise call | medium |
| P9 | Engine E: apply and entries | medium |
| P10 | Engine F: the stacks `W`, `K`, `C`; `sum_over_samples=False`; the dtype rule | medium |
| P11 | Engine G: the uniform layout; sharding over `W` | medium |
| P12 | The backend sampling functions on the engine (merged signatures); uniform twins; the corewise Hessian against automatic differentiation | large |
| P13 | The frontend sampling methods merged; the examples' dense ground truth; `docs/chunking.md` removed | large |
| P14 | The oracle and the batch (additive): `query`, `check_batch`, `pack_batch`, the minibatch scale; reachable from `t3toolbox.fitting` | medium |
| P15 | The geometry's `frame(x)` returns the centers | small |
| P16 | The switch, first part: the corewise frame, `LocalModel`, `local_model`, the optimizers, the display, the frontend optimizers (with the translation of the old call form) and the frontend local model; the tests | largest |
| P17 | The switch, second part: the examples and the pages in the new form; the translation deleted | medium |
| P18 | The dry port of T3Polynomial | small |
| P19 | Documentation | large |
| P20 | The deletion of the old kernels | medium |

### 5.3 The slices

**P0. Baseline.** Run `dev/graded_messages/scripts/run_all.sh` and `dev/graded_messages/scripts/review/run_review_checks.sh`
against the branch head and record drift. Write the inventory: every reference to the 166 sampling names, the 51
frontend sampling methods, `dense_probe`, the six model factories, `Problem`, `least_squares_problem`,
`_canonical_weight`, `_check_ragged_sample`, `chunk_size`, `sum_over_probes` and `ztildes`, in `t3toolbox/` (the
package `__init__.py` included), `tests/`, `docs/`, `examples/`, `CLAUDE.md`. The inventory is a working note in
`dev/` (ephemeral). Gate: none.

**P1. `jet_contract`.** (Built and committed, `3fcfbe7`, with the groups `T` and `A`; rebuilt over the one jet group `J`
in P4a. The text below is the slice as it was carried out.) In `backend/contractions.py`: `GroupLengths` and `jet_contract` (section 2.4), with both
forms over the orders. `'scan'` is the carried shift through `xscan`, with a module-level body from a factory
memoized on `(subscripts, lens)`; `'loop'` is a Python loop over the pairs of orders. The prototype's
`jet_contract` (`term_stacks_check.py`) is the algebra and `scripts/review/order_scan_check.py` the check of the two
forms; their Python-loop scan is not the code. It passes `contract` only the lengths of the groups each derived
string names.
- Tests (`tests/test_jet_contract.py`): every one of the 41 strings of P1's `tests/_jet_contract_cases.py` as built
  (P4a replaces them with the 29 of section 2.4),
  against a reference that uses neither `contract` nor `jet_contract` (the prototype's `reference`, moved into the
  test), at orders 0, 1, 3, with `W`, `K`, `C` empty and not, under both forms; the two forms agree to tolerance on
  every product of two full jets, at orders 1 to 16, in float64 and float32; a perturbation is caught; a stray axis
  on any operand raises; at order 0 exactly one `contract` call under either form; an unknown `over_orders` raises;
  dtype kept in float32; numpy in numpy out, jax in jax out.
- `tests/test_contractions_sharding.py` (class `TestInterpreterAnyAxisSharding`) and
  `tests/test_contractions_interpreter.py` scan source literals for grouped strings; the new strings at call sites
  are literals, so they are seen. Add a test that every string `jet_contract` derives (a literal with `T` or `A`
  deleted) is a valid `contract` string, and add `jet_contract` on the product strings to the sharding sweep, under
  both forms (shard each `W` sub-axis, compile on 4 virtual devices, zero all-gathers).
- `tests/test_dispatch.py`: jit `jet_contract` on each of the three cases, under both forms; `'scan'` on jax
  compiles to one `lax.scan`.
- Gate: the full gate. Deletes nothing.

**P2. The reference module and the dense functions.** (Built and committed, `a38d1a3`, with `pp` and an integer
`order`; the dense functions are rebuilt in P4b and the reference module in P4c. The text below is the slice as it
was carried out.) `backend/sampling_definitions.py`: one dense function per
name of section 2.3 and per output, for the three kinds, ragged, from the column "As a subnetwork" of
`SAMPLE_VARIABLES.md`; jets by interpolation in `s`; `z` through the dense functions; `Kp` by finite differences on numpy arrays and by jax on jax arrays (the route inferred from the arrays, as everywhere). Tilde and tilde-dot variables are returned turned, as the engine stores them (2.4).
It uses no sweep and no engine. The starting material is parts B and C of
`scripts/review/sample_variables_check.py`. The dense functions of section 2.4 (O-27, O-28): `dense_probe` in
`backend/probing.py` takes `(T, ww, pp=None, order=0)`, `dense_apply` in `backend/apply.py` and `dense_entries` in
`backend/entries.py` are new, each in its module's `__all__`.

What breaks: every call of the library's `dense_probe`, which takes its arguments in the other order. An
alias-aware search of `t3toolbox/`, `tests/`, `docs/`, `examples/` and every doctest finds 12, all positional:
- doctests: `backend/probing.py` (eight: three in `t3_probe`'s, two in `tv_probe`'s, three in `dense_probe`'s own),
  `t3toolbox/manifold.py` (one, `T3Tangent.probe`), `t3toolbox/uniform_manifold.py` (one, `UT3Tangent.probe`);
- tests: `tests/test_manifold.py` (line 524, as `t3p.dense_probe`), `tests/test_tucker_tensor_train.py` (line 3166, as
  `probing.dense_probe`).
No page calls it. Four test files and three examples define a local helper of the same name, already in the new
order (`tests/test_fitting.py`, `tests/backend/test_optimizers.py`, `tests/test_optimizers_frontend.py`,
`tests/oracle_sweeps/o1_common.py`, used by `o1_ragged.py` and `o1_uniform.py`; `examples/fit_hilbert_from_probes_adam.py`,
`fit_hilbert_from_probes_optax.py`, `fit_per_mode_weight_probes.py`); they are not the library's and do not change.
- Tests (`tests/test_sampling_definitions.py`), so that the reference itself is pinned: `z` against
  `t3_to_dense` contracted with the inputs; jets against the Taylor expansion computed two ways (interpolation at
  two node sets); `Jp` against the dense tangent tensor (`tv_to_t3`, then dense); `g` against the transposed dense
  Jacobian; the subnetwork pairings of section 8 of `SAMPLE_VARIABLES.md` at every cut edge; `Kp` jax against finite
  differences where jax is present. The dense functions: at order 0 `dense_probe` gives today's values, `dense_apply`
  equals the probe contracted with the last vector, `dense_entries` equals indexing `T`; their jets equal
  interpolation in `s` of order-0 calls; `W` and `C` stacks. Every function has a doctest.
- `tests/test_dispatch.py`: the three dense functions under jit.
- Gate: the full gate.

**P3. Jets, the checkers and data preparation.** (Built and committed, `625d3ba`, with `pp` and an integer `order`;
rebuilt in P4d. The text below is the slice as it was carried out.) `backend/jets.py` (N4 to N8; the two validators of `pp`, copied,
the old ones deleted in P20; `check_sample`, `check_data`, `check_weight`) and `backend/data_preparation.py` (N9),
with `scales` one per coordinate (O-35), and `ut3_unscale_coordinates` (O-34). Their frontend reach (O-16, the
answers of 20:15):
`t3toolbox/fitting.py` makes the preparation functions and the jet helpers reachable under their backend names,
with its own `dual_jet_to_jet` (O-20); `unscale_coordinates(scales)` on `TuckerTensorTrain` and
`UniformTuckerTensorTrain`.
- Tests: `dual_jet_to_jet(jet_to_dual_jet(x))` equals `x` to tolerance for a nonzero weight; the backend
  `dual_jet_to_jet` with a zero weight divides by zero and raises nothing; the frontend one raises in safe mode,
  divides in `safety.unsafe()` and under `jax.jit`, and the numbers are the same where both run; the pairing
  identity `sum_t omega[t]^2 a[t] b[t] = [s^n](jet_to_dual_jet(a) b)`; the derivative functions reproduce the
  derivative-mode objective (the draft's case 2); `prepare_derivative_data` with no weight returns `t!`, for apply
  and for probe with its mode axis; `equalizing_weight` on a zero block gives the floor weight; the shape of
  `data_block_norms` is the shape of the weight at orders 0 and above; `rescale_coordinates` then
  `t3_unscale_coordinates` gives back the same probe values in the original coordinates, and the method equals the
  function; `ut3_unscale_coordinates` equals the ragged one on real parts, with garbage in the padding; each checker passes a good input and raises on each wrong form (a wrong count of vectors, a wrong mode
  size, `pp` unlike `ww`, a wrong `y`, a weight of the old shapes: a bare vector, a mode dimension of 1).
- Doctests: the doctest of 2.3, from `scripts/review/adapter_doctest_draft.py`, extended with the probe kind.
- `tests/test_dispatch.py`, `tests/test_jax_absent.py`: the new modules. `tests/test_import_graph.py`: no backend
  module imports `t3toolbox.safety` (true today; pinned from here).
- Gate: the full gate.

**P4. The checkers of a frame** (4.2; O-5, O-6). `fv_operations.fv_frame_consistency_residual` keeps its
signature and its formula, `||T(D, L) - T(D, R)|| / max(1, ||T(D, R)||)`, and computes both norms through the
cores, never densely. The route is the one `geometry.point_norm_sq` takes today: `t3_operations` (the difference)
and `t3_orthogonalization` (a left orthogonalization, then the last core's norm). `t3_linalg` is not used: it
imports `ranks`, which imports `sharing`, which imports `fv_operations`, so it would make a cycle that
`tests/test_import_graph.py::test_the_backend_graph_is_acyclic` rejects. Beside it, `fv_centers_consistency_residual`;
in `backend/ufv_operations.py`, the uniform twins of both. Each returns a residual and never raises. In the
frontend, `T3Frame.consistency_residual` and `UT3Frame.consistency_residual`, cached, and `is_consistent` on them
(`UT3Frame.is_consistent` goes through `UniformTuckerTensorTrain` arithmetic today and moves to the twin).
- Tests: a library-built frame passes at about 1e-12; a perturbed store fails; wrong centers fail; ragged and
  uniform agree; a frame of mode sizes whose dense tensor could not be allocated (for example `d = 12`, `N = 100`)
  is checked in well under a second; the cached property is computed once.
- `tests/test_frame_variations_format.py`, `tests/test_uniform_frame_variations_format.py`: the `is_consistent`
  cases.
- Gate: the full gate.

**P4a. `jet_contract` over one jet group** (entries 116, 120; section 2.4). In `backend/contractions.py`:
- `GroupLengths(J, W, K, C)`: the field `T` is renamed `J` and its bound (`T` of 0 or 1) is dropped: `J` is any
  nonnegative integer.
- `jet_contract(subscripts, *operands, lens, order, over_orders)`: the letters of a string are `J`, `W`, `K`, `C`
  (`T`, `A` and `S` raise); the one behaviour of section 2.4 (the truncated product on the box of `order`, or its
  corner when the output has no `J`); the checks of section 2.4. The reading of a string (which terms carry `J`,
  whether the output does) is cached on the string; the choice of the carrier, the pair and the operands taken one
  coefficient at a time is made from the operands' jet sizes, which are static, and cached on `(string, jet sizes,
  order)`. The scan body stays module-level, from a factory memoized on hashable statics only (the string, the
  record, `order`, the jet sizes, the positions). Under `'scan'` it scans where `m = 1` and the partner has as many
  slots as the box; everywhere else (a smaller partner; `m >= 2`) it loops, under both forms.
- The comments and the module docstring say `J` for `T` and `A`. The note `PART 2 (row 5.5)` stays at the product
  step. A second note, `PART 2 (row 5.4, jets in several variables)`, marks the place where `m >= 2` takes the loop
  under `'scan'`, and the item is added to "Carried to Part 2" of `TARGET_DECISIONS.md` (by Claude; account,
  section 7).
- `tests/_jet_contract_cases.py`: `STRINGS` holds the 29 strings of section 2.4; `reference` is rewritten as an
  independent truncated product: each operand's slots enumerated by explicit loops over its multi-indices, a fixed
  lowercase `np.einsum` per combination, accumulated at the sum of the multi-indices where it lies in the box (or at
  the corner); it uses neither `contract` nor `jet_contract`. `n_contractions` follows the rule of section 2.4.
- Tests (`tests/test_jet_contract.py`, rebuilt; its 32 tests' subjects kept):
  - every one of the 29 strings against the reference at `m = 0`; at `m = 1` with the orders `(1,)`, `(3,)` and the
    library's operand shapes (full jets of `F`, a leaf or an input of two slots) and with random sizes per operand
    (below, at and above the box, and 1); at `m = 2` with the orders `(1, 2)` and `(2, 0)` and the same kinds of
    shapes; `m = 3` on a few strings; `W`, `K`, `C` empty and not; both forms;
  - at `m = 1`, on strings whose cores are scalars, against `numpy.polynomial.polynomial.polymul` truncated, an
    arithmetic that shares nothing with the reference;
  - the pairing that the dual jets rest on: the corner of `a` times the weighted `b` flipped on every jet axis equals
    `sum_k omega_k^2 a_k b_k`, at `m = 1` and `m = 2`;
  - the cases of section 2.4: slots beyond the box are not read (changing them changes nothing); an operand with
    fewer slots equals the same operand padded with zeros; a jet axis of size 1; an output box larger than every
    operand's;
  - structure: `m = 0` is exactly one `contract` call on the string as written, under either form; at `m = 1` with
    the library's shapes the number of `contract` calls is P1's (one for a ride, two for a product with a two-slot
    partner, `order + 1` for two full jets) and no more for a readout; no contraction names a pair of orders;
  - the two forms agree to tolerance on every product of two full jets at `m = 1`, orders 1 to 16, in float64 and
    float32; at `m = 2` they give the same arrays;
  - strictness: a stray or a missing axis on any operand raises; a wrong record; `len(order) != lens.J`; a negative
    or non-integer entry of `order`; a jet axis of size 0; `T`, `A` or `S` in a string; `J` not leading its term; an
    unknown `over_orders`; `lens` not a `GroupLengths`; a count of operands that does not match the string;
  - float32 stays float32; numpy in numpy out; the operands are not changed;
  - `GroupLengths`: the fields `J`, `W`, `K`, `C` in order; equality and hash by value; frozen; `J` any nonnegative
    integer; an equal record gets the same scan body;
  - the expected failure of `TestOrderInsideAStep` (row 5.5) kept, restated on the 12 product strings with the
    library's operand shapes: a two-slot leaf and a full leaf in the 8 `mu` and `nu` strings, two full jets in the
    4 `eta` strings, 20 cases as in P1. At orders of 2 and above the carrier is the one P1 chose, and the writer's
    recount with `contractions._pairwise_path` gives P1's result: 16 of the 20 pair the carried jet with the core
    first; the 4 `nu` strings with a two-slot leaf pair the core with the coefficient. At order 1 a leaf and a full
    jet both have two slots, and in the `nu` strings the leaf (first in the string) becomes the carrier: those 4 then
    pair the carried jet with the core too (20 of 20). The tests pin the counts at order 4 as P1 did (`ORDER = 4`), and state the
    order-1 case in the docstring; the implementer checks the recount and puts it in the handover.
- `tests/test_contractions_interpreter.py` (`TestJetContractDerivedStrings`, three records): every string that
  `jet_contract` hands `contract` (the literal with `J` deleted from the operands taken one coefficient at a time)
  is a valid `contract` string, for the 29 strings and every literal call site; its `HISTORICAL` table follows.
- `tests/test_contractions_sharding.py`: the product strings under both forms at `m = 1`, and one product at
  `m = 2`, each `W` sub-axis sharded, zero all-gathers.
- `tests/test_dispatch.py`: jit `jet_contract` on a ride, a product with a leaf, a product of two full jets and a
  readout, under both forms, at `m = 0`, `1` and `2`. P1's `test_jet_contract_scan_is_one_lax_scan` is kept with its
  three assertions, restated: at `m = 1` under `'scan'` the program of a product of two full jets and of a readout
  with a third full jet has the same number of lines at orders `(2,)` and `(8,)`, and under `'loop'` it grows; a ride
  and a readout with no third full jet have no scan, and a product with a two-slot leaf has no scan at orders above
  1 and one scan of one step at order `(1,)` (the leaf is then as large as the box); the eager scan body is one
  stable object. At `m = 2` neither form has a scan, and the program grows with the box (O-38).
- What breaks: tests only. `jet_contract` and `GroupLengths` have no caller in the library (searched
  `t3toolbox/`: only `backend/contractions.py`). The calls in tests (an `ast` count of calls of the two
  names): `tests/test_jet_contract.py` 55, `tests/test_dispatch.py` 4, `tests/test_contractions_interpreter.py` 2,
  `tests/test_contractions_sharding.py` 2; `tests/_jet_contract_cases.py` names them in its docstring only; `tests/test_sampling_definitions.py`
  names `jet_contract` only in its list of forbidden names, which stays.
- Gate: the full gate.

**P4b. The input jets and the dense functions** (entries 118 to 120; section 2.4). In `backend/jets.py`: `affine_jet`
and `index_to_one_hot` (N35 to N37), and the private reading of `order` that every function taking it uses (a tuple
of nonnegative integers; an integer raises, O-36). The dense functions:
`dense_probe(T, ww, order=())` and `dense_apply(T, ww, order=())` take input jets of any jet sizes, clipped to the box;
`dense_entries(T, index)` loses `pp` and `order`. Their shared private `_dense_jet` (`backend/probing.py`) contracts
the tensor with the input jets one mode at a time, a truncated product of polynomials by explicit loops over the
input's multi-indices, never through `jet_contract`. The docstrings say what a jet of a dense tensor is in several
variables, with one doctest each of a line (`affine_jet(ww, pp)`), of two directions, and of an input of three
slots.

What breaks: every call of a dense function that passes `pp` (an order-0 call, `dense_probe(T, ww)`, is unchanged;
`t3toolbox/manifold.py` line 735, `t3toolbox/uniform_manifold.py` line 575, `tests/test_manifold.py` line 524 and
`tests/test_tucker_tensor_train.py` line 3166 are such). The calls with `pp`, from a search of `t3toolbox/`,
`tests/`, `docs/`, `examples/` and the doctests:
- the dense functions' own doctests in `backend/probing.py`, `apply.py`, `entries.py`;
- `backend/data_preparation.py`: four, in its module doctest (two) and in `rescale_direction`'s (two). They become
  `dense_probe(T, affine_jet(ww, pp), (order,))`; the rest of those doctests changes in P4d;
- `backend/sampling_definitions.py`: three, in the definition of `z` (`_dense_sample`, lines 572 to 581). Until P4c
  the module builds the two-slot input jet itself, `xnp.stack([w, p])` per mode, one line, and passes it with
  `(order,)`; for entries at an order above 0 it calls `apply.dense_apply` on that input jet made from the unit
  vectors it already builds (`_setting`, line 233), which is what the old call computed. It never imports `jets`:
  its independence test (`tests/test_sampling_definitions.py`, `independence_violations`, lines 1064 to 1133)
  allows only `probing`, `apply`, `entries` and their dense functions, and `*` from `common`, and is unchanged;
- tests: `tests/test_jax_absent.py` (lines 133 and 134, three calls), `tests/test_data_preparation.py` (line 156),
  `tests/test_jets.py` (lines 446 and 449), `tests/test_sampling_definitions.py` (its calls with `pp`; the slice's
  inventory lists them).
No page and no example calls a library dense function with `pp`; the examples' local helpers of the same names are
not the library's.
- Tests (in `tests/test_sampling_definitions.py`, beside the dense functions' tests of P2, which are restated):
  at `order = ()` today's values (`dense_apply` is the probe contracted with the last vector, `dense_entries`
  indexes `T`); the jets equal interpolation in `s` of `order = ()` calls along the input jet evaluated at roots of
  unity: a line at `m = 1`, two directions at `m = 2` (a two-dimensional grid), a curve of three slots; modes with
  different input jet sizes, a mode of size 1 on every jet axis, and the case of the jet matrix with modes `(3, 2)`,
  `(1, 1)`, `(2, 3)` at `(1, 2)`; input slots beyond the box change nothing; `W` and
  `C` stacks. `affine_jet`: the shapes, slot 0 the point and slot `e_j` the `j`-th direction, zeros elsewhere, no
  direction returns `ww`; on packed vectors it returns the packed array, equal to stacking its result on the
  ragged vectors. `index_to_one_hot`: the rows of the identity. The recipe of entry 120:
  `dense_apply(T, affine_jet(index_to_one_hot(index, shape), pp), (n,))` equals
  `sampling_derivatives.dense_entries_derivatives` divided by `t!` (the old module stays until P20).
- `tests/test_dispatch.py`: the three dense functions and the two helpers under jit. `tests/test_jax_absent.py`:
  the helpers without jax.
- Gate: the full gate.

**P4c. The reference module in several variables** (section 2.4, "One such function per name"). In
`backend/sampling_definitions.py`: `order` a tuple; the sample `ww` (input jets) or `index`; an entries call with
`order != ()` raises; the nodes a grid of roots of unity, `M_j` per variable, and a discrete Fourier transform along
each node axis; the inputs clipped to the box; every function returns its variable with the declared jet sizes of
section 2.3 (the leaves with the input's, clipped); the seed `w_tilde` turned on every jet axis. The module reads
`order` with a private function of its own and builds its own inputs (unit vectors, input jets); it calls nothing
of `jets`, by its independence test. The module docstring
is rewritten for jets in several variables (the paragraph on `A` goes; the inputs are jets; the entries kind has
none).
- Tests (`tests/test_sampling_definitions.py`; 47 of its 56 tests build a sample, pass `order` or name entries, and
  the slice's inventory lists them): every name at `order = ()`, `(1,)` and `(3,)` with a line input, at `(1, 2)`
  with `affine_jet(ww, p1, p2)` and with modes of input jet sizes `(3, 2)`, `(1, 1)`, `(2, 3)`, and at `(2,)` with an
  input of three slots and with modes of different input jet sizes; the jets against the Taylor expansion computed on two node grids; `Jp`, `g` and `Kp` against the dense
  tangent tensor, the dense Jacobian and the second derivative, at `m = 2` too; the pairings of section 8 of
  `SAMPLE_VARIABLES.md` at every cut edge, at `m = 2`; with one variable and a line input, `z` and `g` of the probe
  kind against the old `sampling_derivatives` functions through the conversions (`t!` per slot, the turn); the
  entries kind at `order = ()` equals apply at `index_to_one_hot`, and an entries call with an order raises.
- The independence test of the module (its forbidden names) is kept unchanged, and passes.
- What breaks: nothing outside the module's tests: the library has no caller of `sampling_definitions` (searched
  `t3toolbox/`, `docs/`, `examples/`; `tests/test_jax_absent.py` imports it and is updated).
- Gate: the full gate.

**P4d. `jets` and `data_preparation` in the jet shape** (entries 116 to 120; section 2.4). In `backend/jets.py`:
`jet_to_dual_jet` and `dual_jet_to_jet` weigh, then flip every jet axis; `derivatives_to_taylor`,
`taylor_to_derivatives`, `derivative_weight_to_taylor_weight` and `prepare_derivative_data` use `k_1! ... k_m!`;
`check_sample`, `check_data` and `check_weight` take the jet shape and input jets, and the entries kind with
`order = ()` only; `check_perturbation_vectors` and `check_perturbation_index` are deleted from `jets` (their
`__all__` loses two names; the originals stay in `sampling_derivatives`, which the frontend's old methods import
until P13). The module docstring and its doctests are rewritten (a jet in several variables, the turn as a flip of
every axis; the entries kind has no jets). In `backend/data_preparation.py`: `data_block_norms`, `rescale_direction`
and `rescale_coordinates` take the jet shape and input jets (section 2.4); the module doctest uses `affine_jet`. In
`t3toolbox/fitting.py` and `t3toolbox/__init__.py`: `affine_jet` and `index_to_one_hot` reachable under their backend
names from both (O-37); `t3toolbox.fitting`'s own
`dual_jet_to_jet` and `rescale_coordinates` follow; the module docstring.
- Tests (`tests/test_jets.py`, 39 of 42 tests touched; `tests/test_data_preparation.py`, 15 of 29: the tests whose
  source names `pp`, `order` or `entries`, by an `ast` walk; the slice's inventory lists them): the pairing identity `sum_k omega_k^2 a_k b_k = [corner](a * jet_to_dual_jet(b))` at `m = 1`
  and `m = 2`, against a product computed in the test; the round trip of the dual jet, on a packed probe array too;
  the derivative conversions against mixed partial derivatives of a known polynomial in two variables;
  `prepare_derivative_data` with no weight returns `k_1! ... k_m!` per slot, for apply and for probe with its mode
  axis; the entries kind with `order = ()` returns its data unchanged, and with an order raises; each checker passes
  a good input and raises on each wrong form (too few axes for the jet shape, modes whose `W` disagree once the jet
  axes are read, a jet axis of size 0, data not of the full box, a weight of the old shapes: a bare vector, a mode
  dimension of 1, a weight of `(d, n + 1)` given for `m = 2`); `rescale_direction` gives the data of the same tensor
  at the new sample (`dense_probe`, `dense_apply`) at `m = 1` and `m = 2`, with one length and with one per
  variable; `rescale_coordinates` then `t3_unscale_coordinates` gives the same probe values at an input jet; the
  frontend's `dual_jet_to_jet` raises in safe mode at `m = 2` as at `m = 1`.
- `tests/test_dispatch.py` (45 calls of the functions of `jets` and `data_preparation`, an `ast` count) and
  `tests/test_jax_absent.py` (16 calls, inside its subprocess script, counted by a regular expression) by the new
  signatures; `tests/test_import_graph.py` unchanged.
- What breaks: the callers of the changed functions, which are only these modules, `t3toolbox/fitting.py` and the
  tests above (searched `t3toolbox/`, `docs/`, `examples/`: no page or example calls them yet). The frontend methods
  `unscale_coordinates` are unchanged.
- Gate: the full gate. After P4d nothing in P1 to P3's code takes `pp` or reads an integer order.

**P5. Engine A.** `backend/sampling_steps.py` with the steps of the plain grade; `backend/sampling_variables.py` with
the probe kind's declared graph, `HeldVariables` and `sampling_variables` for `w`, `xi`, `mu`, `nu`, `eta`, `nu_L`,
`eta_D`, `z` (through `D`, and with `Vc` read off the frame), `r`, `f`, `r_block_sumsq`. Ragged layout, one sample
axis, no `K` or `C` yet, numpy and jax. What the engine needs to run (section 2.3 step 1), and nothing more.
- Tests by definition (`tests/test_sampling_variables.py`): every name against `sampling_definitions`, with a
  random weight, at the jet shapes of the matrix of section 6 ("The jet matrix"): `()`, `(1,)` and `(3,)` with a line
  input, `(1, 2)` with `affine_jet(ww, p1, p2)`, `(2,)` with an input of three slots and with modes of different
  input jet sizes, and `(1, 2)` with modes of input jet sizes `(3, 2)`, `(1, 1)`, `(2, 3)`.
- The declared jet sizes (section 2.3): every variable the engine returns has the jet sizes of the table, the
  leaves the input's clipped to `F`; every `jet_contract` call of a step is given the output's jet shape of the
  table (a test hook on the private step module records the calls).
- Identities (`tests/test_sampling_identities.py`): `order = ()` equals the slot `(0, ..., 0)` of any jet shape; a
  jet shape `(n_1, n_2)` restricted to `(n_1, 0)` equals the one-variable call along the first direction; a held variable gives
  the result of a recomputed one; `z` through `D` equals `z` with `Vc`.
- Structural tests (`tests/test_sampling_graph.py`): the docstring table equals the declared graph row for row,
  exactly; the slots of `HeldVariables` are exactly the public computed variables of the graphs (section 2.2, amended
  2026-11-20: the input `w` is an argument, not a slot; private nodes are not slots); asking for one variable
  computes only what the graph says it reads (count step calls with a test hook on the private step module); an
  unknown name in `want` raises; every missing argument raises; a held array of a wrong shape raises.
- `HeldVariables` (`tests/test_held_variables.py`): reads like a dictionary; an unknown name raises at construction
  and at lookup; `with_values` returns a new record and leaves the old one unchanged; the pytree round trip where
  jax is present; imported and used with no jax (`tests/test_jax_absent.py`).
- No check of the engine depends on the safety mode: the numbers and the errors are the same in safe mode, in
  `safety.unsafe()` and under `jax.jit`.
- `tests/test_dispatch.py`: jit `sampling_variables` for `('f',)`; a stray `np.` on a tracer raises.
- Gate: the full gate.

**P6. Engine B.** The tilde steps; `w_tilde`, `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `V_tilde`, `H_tilde`,
`g`.
- **Design, settled before the build (2026-11-20; `implementation/P6_DESIGN_NOTE.md`, `P6_DESIGN_REVIEW.md`):** the
  short form of `eta_tilde` is a private node holding only the slots `V_tilde` reads at the corner (the top
  `a = min(J_in, F)` slots per jet axis, each a readout at a lowered corner; one new readout string,
  `'JWCa,Caib,JWCb->WCi'`, added to `STRINGS`), and **`V_tilde` always reads it**, by one row; the full `eta_tilde` has
  its own row, run only when it is wanted. Chosen for simplicity (one row for `V_tilde`, one path to test, nothing
  about `want` in a call's facts), at a negligible cost: `2 prod(a)` extra corner readouts per site, only when
  `eta_tilde` and `g` are both wanted. (The alternative, two routes for `V_tilde` chosen by whether `eta_tilde` is
  wanted, was equally acceptable numerically: values equal to rounding are the same value, the maintainer's rule of
  2026-11-20 in `TARGET_DECISIONS.md`.) A test checks that the private node equals the slice of the full `eta_tilde`,
  to rounding. **Each row declares its own start value** (`'ones'`, `'zeros'`, or the
  name of a node: apply's `nu_tilde[d]` starts at the seed, P9), not chosen by grade. Every tilde row has
  `carries_K=False`; P7's `Jp`-seeded rows are private twins named by a fixed rule, seeded through the same
  `jet_to_dual_jet`, and a test checks every public tilde row has its twin. The bridge lists each old tilde kernel it
  leaves out, with the reason; the generic twin runs the mixed-modes case and computes `V_tilde` from the full
  `eta_tilde`, an independent check of the short form.
- Tests by definition for the new names. Identities, each with a random weight: `g` is the derivative of `f` by
  finite differences; the short form of `eta_tilde` equals a slice of the full one.
- The bridge (`tests/bridge/test_old_kernels_bridge.py`, temporary, section 5.1): `z` and `g` against
  `sampling_derivatives.t3_probe_derivatives`, the old `compute_*` kernels and `ProbeDerivativesKind.transpose`,
  the new side called at `affine_jet(ww, pp)` and `(order,)`; the dense functions against `dense_*_derivatives`
  divided by `t!` (the entries one against `dense_apply` at `affine_jet(index_to_one_hot(index, shape), pp)`). The
  bridge has one variable only: the old code has no other. **The conversions, as derived in P4c** (added
  2026-11-19; `TestAgainstTheOldKernels` of `tests/test_sampling_definitions.py`, checked by the reviewer from the
  definitions): the old `z` is `t!` times the new one, slot by slot; the old transpose computes
  `sum_t t! J_t^T c_t` and the new `g` is `sum_t J_t^T omega_t^2 r_t`, so the old transpose given
  `c_t = omega_t^2 r_t / t!`, in FORWARD order, equals the new `g`. Not turned: the turn is inside the new `g`. A
  nonuniform weight makes the `t!`, the square and the turn each visible.
- The generic twin (`tests/_generic_twin.py`): `scripts/graded_messages_prototype.py` brought to the decided names,
  as a second check of the written-out terms (9.3).
- `tests/test_dispatch.py`: jit `sampling_variables` for `('f', 'g')`.
- Gate: the full gate.

**P7. Engine C.** The dot steps of the frame form; `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`, `Jp`,
`Jp_norm_sq`, `JtJp` (the adjoint sweep seeded from `Jp`, its variables unnamed, 3.3).
- **Design, settled before the build (2026-11-20; `implementation/P7_DESIGN_NOTE.md`, `P7_DESIGN_REVIEW.md`):** the
  dot rows carry `K` and start at zeros; `JtJp`'s adjoint sweep is private twins of P6's rows on its path, named
  `'_Jp.' + name.lstrip('_')`, derived mechanically (same step and start, `carries_K=True`, the seed reading `Jp`), with
  a `_base(name)` helper for a twin's letter, and `JtJp` the twin of `g`; no twin of the full `eta_tilde` or of `g`.
  **The two names with both dot forms (O-12):** private nodes `_frame.nu_L_dot` and `_frame.eta_D_dot`, which `Jp`
  reads; the public `nu_L_dot` and `eta_D_dot` are route rows over them needing `variations`, and P8 adds the raw
  route in front needing `core_directions`, so the precedence rule gives O-12 and no public name moves between
  slices. **One structural check at entry, ruled by Claude:** every `V[i]` and `H[i]` has the `K + C` prefix the
  engine reads from `V[0]` (a size-1 `K` on `H` would otherwise broadcast silently in the contraction and give a
  wrong `Jp`); read as part of what the engine needs to run (O-18: it reads the size of `K` from one operand, and
  checks the others that carry `K` agree), and `CLAUDE.md`'s rule that structural problems always raise. P10 applies
  the same rule to `W` and `C` (flagged for the maintainer at the pause after P11).
- Tests by definition for the new names. Identities: the adjoint identity between `Jp` and `g`; `p` paired with
  `JtJp` equals `Jp_norm_sq`; `JtJp` equals `g` computed with `Jp` held as `r`. Bridge against
  `ProbeDerivativesKind.forward` and `.transpose`.
- Gate: the full gate.

**P8. Engine D.** The raw dots and the tilde-dot steps; `xi_dot`, `mu_dot`, `nu_L_tilde`, `eta_D_tilde`, the four
tilde-dot variables, `V_tilde_dot`, `H_tilde_dot`, `Kp` from `core_directions`. The corewise call `(B, G)`: one
basis per edge, each sweep made once (small item 1); the names with a basis are not accepted on it (O-13). In a call
with both directions the public `nu_L_dot` and `eta_D_dot` are the raw form (O-12).
- Tests by definition, `Kp` included. Identities: on a corewise call at the frame's base point `(D, L)` (not at a
  raw `x`, which differs from it by a gauge), `nu` equals the manifold call's `nu_L`, and `z` agrees with the
  manifold call's `z`; along the lift of `(V, H)` with no damping, the raw and frame forms of `nu_L_dot` and
  `eta_D_dot` are the same arrays (`subtree_gauge_check.py`). Bridge against `corewise_hessian` of the old
  derivative kinds.
- A count test: a corewise call wanting `('z', 'g')` runs each sweep once. A corewise call wanting `nu_L` raises.
- **Design, settled before the build (2026-11-20; `implementation/P8_DESIGN_NOTE.md`, `P8_DESIGN_REVIEW.md`):** the raw
  dots (`xi_dot`, `mu_dot`, private `_raw.nu_L_dot`/`_raw.eta_D_dot`) reuse P7's dot steps; `nu_L_tilde`,
  `eta_D_tilde` reuse P6's, with a short node `_eta_D_tilde_top`; the tilde-dot grade adds `xi_tilde_dot`, the six-term
  `mu_tilde_dot` and `nu_L_tilde_dot`, the full `eta_D_tilde_dot` (run only when wanted) and a short node read by
  `V_tilde_dot`; `H_tilde_dot` is nine readouts; `Kp` pairs the blocks. **O-12:** raw route rows in front of P7's
  frame routes, needing `core_directions`; `Jp`/`JtJp` read the frame nodes, `Kp` the raw ones; a raw route for `Jp`
  after the frame route (for the corewise call); `JtJp` unreachable with only `core_directions` on a manifold call,
  the docstring saying why. **One shared `K`** for `V`, `H`, `D_dot`, `L_dot` (the curvature pairs element `k` of a
  direction with element `k` of its lift); a mismatch raises, the structural check of P7 extended to the four. **The
  corewise call:** its graph derived at import from the manifold one (drop the rows reading `U`, `R`, `V`, `H`, `Vc` or
  a frame node; one rename table over names, `reads` AND `needs`, `'variations'` to `'core_directions'`; then the
  twins, derived through the rename), with two import-time assertions (the rename never lands on a name a surviving
  row uses; every public corewise name is reachable) and a test of corewise `JtJp` against `g` with `Jp` held as `r`;
  "each sweep once" and O-13 follow by construction. **On a corewise call the direction argument is
  `core_directions`, and `variations` raises** (ruled by Claude, option (a) of the review; to be shown the maintainer
  at the pause after P11): at the frame `(B, G, G, G)` the frame form of a direction is its raw form, so accepting both
  would run two identical families of raw dots when both are given (the exact Hessian on a corewise geometry, whose
  lift is the identity); the least-squares oracle passes the direction as `core_directions` on a corewise frame,
  telling the call type from the frame's structure (section 2.2), not from the geometry (section 2.4, P14). **One
  table of the direction operands for P11's masks** (`V`, `H` to the variation masks; `D_dot`, `L_dot` to the
  stores' masks; `Vc`). **Tests:** `Kp` against the reference (exact), not a central difference of `g` (exact only to
  degree 2, and `g` has degree `d - 1`); a step-count test for a corewise `('Kp',)`; the bridge to the old
  `corewise_hessian` given forward `omega^2 r / t!` (derived: the old Hessian is of `sum_t t! <c_t, z_t>`).
- Gate: the full gate.

**P9. Engine E.** Apply and entries: their graphs (the forward needs only `xi` and `mu`; the adjoint only `nu_tilde`
and `eta_tilde`; the README's "Things learned"), the apply kind with jets (the scalar output `F + W`, the weight of
shape `F`), and the entries kind at `order = ()` only (entry 120): the gather, the output `W`, the weight `()`; an
entries call with an order raises (section 2.3 step 1). The engine's graph of the entries kind has no jets anywhere.
- Tests by definition for both kinds (apply on the jet matrix of section 6, entries at `()`); the identities of P5
  to P8 for both; entries against apply at `index_to_one_hot(index, shape)` to tolerance (not bitwise; 0.8). Bridge
  against `t3_apply_derivatives` and its kind at `affine_jet(ww, pp)`; `t3_entries` and the plain entries kind at
  order 0; `t3_entries_derivatives` against apply at `affine_jet(index_to_one_hot(index, shape), pp)`, which is the
  migration of entry 120.
- Gate: the full gate.

- **Design, settled before the build (2026-11-20; `implementation/P9_DESIGN_NOTE.md`, `P9_DESIGN_REVIEW.md`):** for
  apply and entries the seed sits on the ones closing bond `d`, so the right-side tilde variables are the plain
  recurrences started from the seed (a row's start may name a node; `_deps(rule)` = its reads plus its start node,
  used by the closure, the last-read drop, the dtype count and both renames), and `xi_tilde`, `mu_tilde`,
  `xi_tilde_dot`, `mu_tilde_dot` are zero. The rows split into a plain and a dot block shared by every kind, plus
  each kind's own; apply built, entries derived from apply by a declared table of steps (a gather for the leaves, a
  **scatter-add** for `V`: `W x n` memory, not the one-hot product's `W x N`; written with a static "keep `W`" option for
  P10; duplicate indices tested); corewise by the same derivation per kind; `_GRAPHS` and the twin paths keyed
  `(kind, call)`; apply's `z` and `Jp` by a `close_step` that SUMS over the end bond (with tails `z` closes with `rL_d`
  ones and `Jp` with `rR_d`); `nu_step`/`eta_step` carry `K` on the carry; the whole steps take `kind` as a static;
  import-time invariants raise, never assert; a `Vc` given to apply or entries is read by nothing and accepted, as any
  unread argument (the docstring says so). **The zero tilde names are left off the apply and entries graphs** (an
  unknown name, its message saying it is zero there; zeros rows would need a false shape read or a new driver),
  **pending the maintainer at the pause after P11**, with the proposal of a public reader of a call's names (a private
  one meanwhile, used by the test that compares the engine's names with the reference module's per kind).
  **P9 is split in two commits** (in place; the order and scope unchanged): **P9a** apply on both call types with
  `_deps`, the starts, the paths, `close_step` and its bridge; **P9b** entries. **`'w'` means its definition** (ruled
  in P9b): on entries it is the one-hot vectors `e_index`, a row reading `index` and `D` (through `index_to_one_hot`,
  cast once, built only when wanted), on probe and apply the sample as given; `w` is not a slot of `HeldVariables` (it
  is the input, a stated exception to "every public computed variable"). For P11: on a uniform supercore `'w'` of
  entries is one-hot of the padded length, its real columns as ragged and its padding zero.

**P10. Engine F.** Any number of axes in `W`; the direction stack `K` on `variations` and `core_directions`; the
base-point stack `C` on every core; `sum_over_samples=False` with the result's stack in the order `W + K + C`; `f`
keeping `C`, `Jp_norm_sq` keeping `K + C`; the dtype rule over the inputs the call reads (O-11).
**`sum_over_samples=False` needs readout strings with `W` in the output** (the `V` and `H` families, about nine),
which are not among the 29 of section 2.4: they are added to `STRINGS` of `tests/_jet_contract_cases.py`, with their
library shapes (their `K` forms too, for `JtJp` and `Kp`; P7's design review), and the step chooses between the summed
and the unsummed literal by a static flag, as the library
does today with two bodies (`_tucker_variation_jets_step_summed` and `_unsummed`). Found by the reviewer in P5's
design review (2026-11-20); no new group, row 3.7 decides it.
- Tests: `W` empty, `(3,)`, `(2, 2)`, crossed with the library's stack matrix (`_CONFIGS`,
  `docs/contributor/testing_strategy.md` "The stack matrix": no stack, `C`, `K`, `K + C`, multi-axis `C = (2, 3)`
  and `K = (2, 3)`), against the reference applied element by element; the dtype rule for every output on the four
  backends (float32 in, float32 out; float32 mixed with float64 gives float32; jax mixed with numpy gives jax); an
  input nothing wanted reads does not change the dtype (a float64 `y` on a call wanting only `z` in float32 gives
  float32); an integer `index` does not count; the one-hot vectors of `index_to_one_hot` are an input like any other
  float array. The stack matrix runs at `order = ()`, `(2,)` and `(1, 1)` for probe and apply; entries at `()` only.
  The held variables in the dtype rule (section 2.3 step 4, as amended in P5): one read or wanted counts and is cast;
  one neither read nor wanted does not change the dtype, but a jax one still makes the call jax (`use_jax` reads every
  array input).
- Gate: the full gate.

- **Design, settled before the build (2026-11-20; `implementation/P10_DESIGN_NOTE.md`, `P10_DESIGN_REVIEW.md`):**
  most of P10 already holds in the engine (each stack read once, the steps seeing only `GroupLengths`; nothing
  reshapes) and is proved by tests. **Six** unsummed readout literals, not "about nine" (three `V`, three `H`; `K` is a
  group, so its placements are the same strings); a readout step takes a static `keep_W` and picks between its two
  literals in place, the flag passed only to block rows (read off the shape table) and joining the memo key; entries'
  scatter wired the same way; `f`, `r_block_sumsq`, `Jp_norm_sq` ignore `sum_over_samples`. **A removable structural
  check** `_check_stacks` (exact tuple equality, never broadcast-compatibility): `C` as the prefix of every store core
  and of `Vc`, `W` on every `w[i]`; `y`, the weight and the jet sizes stay with the `jets` checkers; it extends P7's
  `K` check and is **pending the maintainer under O-18** (the pause after P11). **The tests depart from the letter
  above, to keep the suite affordable** (the full product against the reference would cost 2 to 20 minutes): the
  stack matrix against the UNSTACKED engine element by element (itself reference-verified), on a covering design
  (every config x `W` pair once per kind, orders cyclic), with multi-axis `W` compared against the engine with `W`
  FLATTENED to one axis (a same-`W` comparison would let a multi-axis-`W` bug cancel), one maximal case per kind
  (`K = (2, 3)`, `C = (2, 3)`, `W = (2, 2)`, order `(1, 1)`), a held stacked variable in one case, the unsummed tier on a
  multi-axis `K` too, and a thin reference anchor; the dtype matrix (four backends) with a read held variable of
  another precision and entries' one-hot `'w'`, x64 by the context manager.

**P11. Engine G: the uniform layout, and sharding over `W`.** The uniform frame and point structures; mask once on
entry, each input with the frame's own masks (section 2.3 step 2); re-mask on exit (O-3); the map-type operations
one site at a time through `xmap` (O-19); the sweeps through `xscan`, a real `jax.lax.scan` with jax; the terms with
`over_orders='scan'`, nested inside (a real scan over a full partner where `m = 1`; the loop where `m >= 2`, section
2.4). The packed sample is `(d,) + J + W + (N,)`: each mode's input jet brought to the largest size per jet axis by
zero coefficients, written by `ut3_operations.pack_vectors` through `pack_if_ragged` (which the uniform twins and
`pack_sample` call). Today `pack_vectors` pads the mode axis only, takes the stack shape of the first vector, and
raises when the modes disagree in `W` (the `xnp.pad` or the `xnp.stack` at the end fails; `backend/ut3_operations.py`,
lines 128 to 152). P11 gives it and `pack_if_ragged` a keyword `len_J=0`, the number of leading jet axes: those
`len_J` axes are brought to their largest size across the modes with zeros; every other axis but the last must still
agree across the modes, and a mismatch raises as today. With `len_J=0` the function is today's, so its 55 calls in
the library and 36 in the tests (an `ast` count of calls of the two names) are unchanged. It takes the count, not the jet shape: what it needs
to know is how many leading axes may differ, and the target sizes are the largest of the modes' own sizes, which
are not `F` (an input may have more slots than the box, or fewer); the count is what `contract` takes for a group
too (`len_W=`, `len_C=`). The engine (P11), the uniform twins (P12) and `pack_sample` (P14) pass
`len_J=len(order)`; `pack_data` needs no jet padding (the data have the full jet sizes `F` in every mode) and passes
nothing. Those zeros are coefficients, not padding (section 7.2): the packed input is the same
polynomial as the ragged one, so every site of the scan has one shape and the result is the ragged result. The
leaves then have the packed jet sizes, clipped to `F`, and equal the ragged leaves, zero-extended.
- Tests: equivalence with ragged on real parts (`to_uniform -> op -> to_ragged == op_ragged`) for every name, over
  the full `_CONFIGS` matrix of the uniform tests: forced-larger padding and a stack whose ranks vary across `C`
  included; garbage in the masked slots of the inputs (the padding of the cores and of the mode axis `N`, and
  nothing on a jet axis) gives the same real parts; a nonzero written into a jet slot that packing filled with zero
  changes the result exactly as the definition says (the reference called with that slot added to the ragged input
  jet gives the uniform result); arbitrary tails on both layouts; exact masks of every
  core-shaped output, derived non-circularly from the frame's masks; at `order = ()`, `(2,)` with a line input,
  `(1, 1)`, and on the case of the jet matrix with modes of different input jet sizes (packed with zero
  coefficients: `(1, 2)` with modes `(3, 2)`, `(1, 1)`, `(2, 3)`, and `(2,)`); entries at `()` only. Under jit with jax at `m = 1`, the
  compiled program of a call has the same size at orders `(2,)` and `(6,)` (the scan over the orders, 2.9).
- `pack_vectors` and `pack_if_ragged` with `len_J` (`tests/test_uniform_probing.py` beside the packing tests, or a
  new file): modes of unequal jet sizes are packed with zeros, and unpacking the ragged sizes gives the inputs back;
  modes that disagree in `W` raise at every `len_J`, as today at `len_J=0`; with `len_J=0` the result equals
  today's on the existing cases; a packed array is returned as it is.
- What breaks: nothing. `len_J` is a new keyword with the default of today's behaviour; no existing call changes.
- `Kp` on a frame with TT slack (its masks a strict subset of its point's): `core_directions` and the blocks take
  the frame's `D` and `L` masks; garbage in the padding of `core_directions` leaves the real blocks unchanged; and a
  mutation check (the mask on `core_directions` removed) turns this test red
  (`docs/contributor/testing_strategy.md`, the two-hole note).
- Sharding (`tests/test_contractions_sharding.py`, a new class beside `TestJetKernelsShardOverW`): `g` and `Kp` from
  `sampling_variables` at `order = ()`, `(3,)` and `(1, 1)`, ragged and uniform, under `shard_map` with `W` sharded and the cores
  replicated: zero all-gathers in the compiled program, and the result equal to the unsharded one. The existing
  class's note on the varying-axes check (`check_vma` / `check_rep` off) applies to the engine's scan carries too.
- `tests/test_dispatch.py`: jit the engine on uniform data; an eager call twice adds no compilation (the scan
  bodies' identity is stable).
- Gate: the full gate. This closes the additive phase.

**P12. The backend sampling functions on the engine.** The 30 changed names
of section 2.4 call the engine with the merged signatures, in the order of N16; `sum_over_samples=True` by default;
the arguments renamed. The uniform twins check packing and call the engine, which masks and re-masks (O-3). The
corewise Hessians take a vector of the type of `z`, no weight, turned inside, as the transposes do (O-22). The old kinds
keep calling the old `*_derivatives`, sweep and `*_from_sweep` functions, which stay; the plain kinds'
`point_forward` now reaches the engine through `t3_probe`, `t3_apply` and `t3_entries` (`backend/fitting.py` lines
538, 565, 604; `backend/uniform_fitting.py` lines 143, 172, 207). Of these six calls, the two of probe move their
arguments (`probing.t3_probe(ww, x_cores)` becomes `t3_probe(x_cores, ww)`, and `ut3_probe` likewise); the calls
of apply and entries already have the object first.

What breaks, from an alias-aware search of `t3toolbox/`, `tests/`, `docs/`, `examples/` and every doctest for the
30 names. Every positional call of a function whose order changes breaks, and so does every call that passes
`sum_over_probes=`; a call that relies on the default of a transpose changes value. `t3_apply`, `t3_entries`,
`ut3_apply` and `ut3_entries` keep their positions (only `vecs` and `data` are renamed, and no caller passes them by
keyword). No caller passes any renamed argument by keyword (`vecs=`, `variation=`, `frame_data=`, `data=`,
`core_pair=`, `ztildes=`, `c=`). The calls, by function (backend and uniform twins together):

| Function | Calls | In tests | In doctests |
| --- | --- | --- | --- |
| `t3_probe`; `ut3_probe` | 9; 2 | 3; 0 | 3; 0 |
| `tv_probe`, `tv_apply`, `tv_entries` | 9, 6, 6 | 3, 4, 4 | 4, 0, 0 |
| `utv_probe`, `utv_apply`, `utv_entries` | 1 each | 0 | 0 |
| `tv_probe_transpose`, `tv_apply_transpose`, `tv_entries_transpose` | 7, 6, 6 | 2, 3, 3 | 2, 0, 0 |
| `utv_*_transpose` | 2 each | 1 each | 0 |
| `t3_probe_corewise_transpose`, apply, entries | 6, 7, 6 | 3, 4, 3 | 1 each |
| `ut3_*_corewise_transpose` | 2 each | 1 each | 0 |
| `t3_probe_corewise_hessian`, apply, entries | 4, 5, 4, and by reference | 1, 3, 2, and by reference | 2, 1, 1 |
| `ut3_probe_corewise_hessian`, apply, entries | 1, 0, 0, and by reference | by reference | 1, 0, 0 |
| unchanged positions: `t3_apply`, `t3_entries`, `ut3_apply`, `ut3_entries` | 9, 3, 2, 2 | 5, 0, 0, 0 | 0 |

A call counts once per call site, in `t3toolbox/` (the uniform twins' bodies and the frontend included), `tests/`
and the doctests; "by reference" is `tests/backend/test_uniform_curvature.py`, which holds the six corewise Hessians
in a table and a helper (lines 595 to 608) and calls them as `hess(z, s, theta, D)` and
`ragged_hess(...)` (lines 646, 647, 656, 673, 676); the `*_from_sweep` functions it also holds stay as they are.
Two more sites hold these functions by reference, and a search by call name misses them too:
`tests/test_dispatch.py` (lines 619 to 626, the corewise Hessians in a table) and `tests/backend/test_curvature.py`
(lines 889 to 896, a helper that returns them, with the corewise transposes inside lambdas). In P13,
`tests/test_uniform_probing.py` (lines 600 to 605) holds the three `UT3Tangent` transposes in a list and calls them
as `op(*args, v.frame, ...)`, which has to be rewritten to put the frame first. All three files are already on the
slices' lists.

Where they are:
- the library: the uniform twins' bodies (`backend/ut3_sampling.py`, `backend/utv_sampling.py`), which call the
  ragged functions and are rewritten here; the two `point_forward` calls above; the frontend call sites, which move
  their arguments to the new order and change to `sum_over_samples=`, keeping their own `sum_over_probes=False`
  default until P13: `t3toolbox/manifold.py` (`tv_{probe,apply,entries}` and their transposes),
  `t3toolbox/tucker_tensor_train.py` (`t3_{probe,apply,entries}` and the corewise transposes),
  `t3toolbox/uniform_manifold.py` (`utv_*` and their transposes), `t3toolbox/uniform_tucker_tensor_train.py`
  (`ut3_*` and the corewise transposes); the three backend ambient transposes take their new order and names
  here, and their four callers follow (the three static methods of `t3toolbox/tucker_tensor_train.py`, which keep
  their own form until P13, and `backend/entries.py`, where the entries one calls the apply one);
- tests: `tests/backend/test_curvature.py` (the corewise transposes and Hessians), `tests/backend/test_from_sweep_reuse.py`
  (the three `tv_*` and `tv_*_transpose`, two places each; it stays until P20 and is updated here),
  `tests/test_dispatch.py` (the three corewise transposes and Hessians), `tests/test_manifold.py` (`tv_apply`,
  `tv_entries` and their transposes), `tests/test_probe_derivatives.py` (`t3_probe`, `tv_probe`, `tv_apply`,
  `tv_entries`), `tests/test_uniform_probing.py` (the `utv_*_transpose`, `ut3_*_corewise_transpose` with the sum flag
  positional at lines 832 to 836 and 906 to 910, and the ragged corewise transposes),
  `tests/backend/test_uniform_curvature.py` (the six corewise Hessians, by reference: the call sites through its
  tables move), `tests/backend/test_uniform_fitting.py` (`t3_apply`,
  unchanged), `tests/oracle_sweeps/o1_ragged.py`, `o1_uniform.py` (labels only);
- doctests: only those of the defining modules (`backend/probing.py`, `apply.py`, `entries.py`, `ut3_sampling.py`),
  rewritten with them. No page doctest and no example calls a changed name; pages that name them in prose
  (`docs/batching_and_stacking.md`, `docs/entries_apply_probe.md`, `docs/transposes.md`, `docs/glossary.md`,
  `docs/naming_conventions.md`, `docs/contributor/signature_style.md`, `CLAUDE.md`) are rewritten in P19;
- the prototype scripts `dev/graded_messages/scripts/extra_core_experiment.py` and `graded_messages_prototype.py`
  call the old forms; they are records that run against `v2026.5.0`, not run by CI (O-26).
- Tests added: the uniform twins' exact output masks and garbage-padded inputs (garbage in the masked slots only;
  section 7.2), through their public signatures; the twins mirror the form of the sample, ragged in ragged out and
  packed `(d,) + J + W + (N,)` in packed out, at `order = (2,)` and `(1, 1)`;
  the default of every transpose sums the samples; the corewise transpose equals the gradient, and the corewise
  Hessian applied to a direction equals the second derivative along it, of `phi(x) = <c, t3_probe(x, ww, order)>`
  (and of apply, and of entries at `order = ()`) by automatic differentiation in jax, at `order = ()`, `(2,)` and
  `(1, 1)`, skipped where jax is absent (O-22); the ambient transposes' default does not sum; an entries function
  has no `order` argument.
- `tests/test_dispatch.py`: the rows for these names, by the new signatures.
- Gate: the full gate. From here `probing.compute_*` and the plain `assemble_*` are called only by the old kinds,
  their tests and the bridge.

**P13. The frontend sampling methods.** The four classes get the merged methods of
section 2.5; the 24 methods go; the frontend transposes and corewise transposes take their arguments in the order of
N16 (the residual after the sample, the tangent transposes' frame first), `residual` in place of `ztildes` or `c`,
and `sum_over_samples=True` in place of `sum_over_probes=False`; `TuckerTensorTrain.apply` and its uniform twin take
`ww` in place of `vecs`; the validators come from `jets`. Where a caller still fits with the old call form of the frontend optimizers (the tests until P16, the examples and
pages until P17), it makes the old derivative data from
the merged method as `taylor_to_derivatives('probe', (order,), x.probe(affine_jet(ww, pp), (order,)))`; old
entries-derivative data, as `taylor_to_derivatives('apply', (order,), x.apply(affine_jet(index_to_one_hot(index,
x.shape), pp), (order,)))`.

What breaks, from a search of `t3toolbox/`, `tests/`, `docs/`, `examples/` and every doctest (the calls, scripts
inside strings included, of the 24 removed methods; every call of the frontend transposes and corewise transposes,
each of which breaks on the new order; and every link to `docs/chunking.md`):
- removed methods, in tests: `tests/test_tucker_tensor_train.py`, `tests/test_manifold.py`,
  `tests/test_uniform_probing.py` (52 calls), `tests/test_probe_derivatives.py` (folded into the frontend sampling
  tests), `tests/test_fitting.py`, `tests/test_dispatch.py`, `tests/test_optimizers_frontend.py`,
  `tests/test_jax_absent.py` (in its subprocess script, lines 66 and 79), `tests/test_chunk_size_estimator.py`,
  `tests/backend/test_optimizers.py`, `tests/backend/test_padded_start.py`, `tests/backend/test_uniform_fitting.py`,
  `tests/backend/test_zero_reference.py`, `tests/oracle_sweeps/o1_ragged.py`, `o1_uniform.py`, `sweep_models.py`;
- the frontend transposes and corewise transposes, 91 calls (all positional in their first arguments, so all move):
  `probe_transpose` 23, `apply_transpose` 14, `entries_transpose` 10, `probe_corewise_transpose` 12,
  `apply_corewise_transpose` 17, `entries_corewise_transpose` 15. By file: `tests/test_uniform_probing.py` 24,
  `tests/test_dispatch.py` 18, `tests/test_tucker_tensor_train.py` 13, `tests/oracle_sweeps/o1_uniform.py` 10,
  `tests/test_manifold.py` 6, `tests/oracle_sweeps/o1_ragged.py` 6, `tests/test_fitting.py` 2; the doctests of
  `t3toolbox/manifold.py` 7, `uniform_manifold.py` 2, `uniform_tucker_tensor_train.py` 2, `tucker_tensor_train.py`
  1. No page and no example calls one. The frontend's three ambient transposes move too: 16 calls in tests
  (`tests/test_tucker_tensor_train.py`, `tests/test_dispatch.py`, `tests/oracle_sweeps/o1_ragged.py`), nine of them
  with `sum_over_probes=` by keyword, and three doctests in `t3toolbox/tucker_tensor_train.py`; their default does
  not change, so no call changes its result;
  `.probe`, `.apply` and `.entries` keep their positions, and no caller passes `vecs=`;
- `tests/test_chunk_size_estimator.py::test_newton_cg_exact_across_chunk_size` keeps testing the frontend
  `chunk_size`, which lives until P16, with its data made as above; it goes in P16;
- module doctests: the methods' own in `t3toolbox/tucker_tensor_train.py`, `uniform_tucker_tensor_train.py`,
  `manifold.py`, `uniform_manifold.py` (the transposes' examples pass `sum_over_probes`); `t3toolbox/optimizers.py`
  and `t3toolbox/fitting.py` (two calls each of a removed method);
- page doctests: `docs/fitting_and_optimization.md` (five calls), `docs/numerical_contracts.md` (two);
- `docs/chunking.md`: removed here with its toctree entry in `docs/design_notes.rst`, because its doctests call
  `probe_derivatives_transpose(..., chunk_size=...)`. The `:doc:` links to it go in the same commit, or `sphinx -W`
  fails: `docs/fitting_and_optimization.md` (three), `docs/glossary.md` (two links, and one mention of the file name), and the docstrings of
  `t3toolbox/optimizers.py` (three), `t3toolbox/manifold.py` (one) and `backend/sampling_derivatives.py` (four,
  though the module stays until P20); plain mentions in `CLAUDE.md` and other docstrings are fixed in P19;
- examples: the six that call a removed method (`fit_hilbert_from_apply_derivatives.py`, `_flat.py`, `_topt.py`,
  `fit_hilbert_uniform_probe_derivatives_newton_cg.py`, `fit_probe_display.py`,
  `fit_shared_factors_jetted_probes.py`). No example calls a frontend transpose. Five of them (all but
  `fit_probe_display.py`) also import `sampling_derivatives` as `pd` for their dense ground truth: five calls of the
  library's functions, each inside a local helper that loops over the samples. `pd.dense_apply_derivatives` is
  called once in each of `fit_hilbert_from_apply_derivatives.py`, `_flat.py` and `_topt.py`, inside a local helper
  that is itself named `dense_apply_derivatives` (lines 116, 110 and 74; called twice in each), which is one of the
  removed names P20's gate greps for. `pd.dense_probe_derivatives` is called once in each of
  `fit_hilbert_uniform_probe_derivatives_newton_cg.py` (helper `dense_probe_derivative_jets`) and
  `fit_shared_factors_jetted_probes.py` (helper `dense_jets`). The helpers are replaced here by `dense_apply` and
  `dense_probe` at `affine_jet(ww, pp)` and `(order,)` (P4b; O-27), which take the stack `W` themselves, so the loops over the samples
  go too, and P20 can delete the module. Where an example still fits derivative data with the old frontend
  optimizers (until P17), it takes `taylor_to_derivatives('apply', (order,), dense_apply(T, affine_jet(ww, pp), (order,)))`. (Seven
  examples define their own `dense_apply(A, ww)` or `dense_probe(A, ww)` helpers, local and of no removed name;
  they may stay.)
  `tests/test_probe_derivatives.py` (nine calls of the three `dense_*_derivatives`) is folded into the frontend
  sampling tests here, on the dense functions.
- Pages that name the removed methods in prose only (`docs/entries_apply_probe.md`, `docs/naming_conventions.md`,
  `docs/transposes.md`) are rewritten in P19.
- Tests added: the doctests on the merged methods (`order=()` returns today's shapes; `order=(2,)` at
  `affine_jet(ww, pp)` adds one jet axis outermost; `order=(1, 1)` at `affine_jet(ww, p, q)` adds two); the merged
  methods raise through `check_sample` on a sample of the wrong form; `entries` has no `order`; on `probe` and
  `apply`, the doctest of a forgotten `order` (wrong against right, O-44).
- Gate: the full gate.

**P14. The oracle and the batch** (additive). `LeastSquaresOracle` and `LeastSquaresBatch` in `backend/fitting.py`
beside the old kinds; the `Oracle` protocol as documentation in `backend/optimizers.py`, with `optional_outputs`
(O-7), `check_batch` as an optional member (O-29), and `held` typed as any object that, in a compiled program, jax
can carry (O-8, O-30); `query` passing `held` (a `HeldVariables`) to the engine;
`n_measurements`, `take`, `check_batch`, `y_block_sumsq`, `f_resolution`, the minibatch scale on `f`, `g`, `JtJp`,
`Jp_norm_sq`, `Kp` and `f_resolution` only (O-21); `pack_batch` in `backend/uniform_fitting.py` (the input jets
brought to one set of jet sizes across the modes, as in P11). Both classes reachable from `t3toolbox.fitting`. The oracle's
`order` is a tuple, required for every kind with no default (O-44): a plain fit writes
`LeastSquaresOracle('apply', order=())`, an entries fit `LeastSquaresOracle('entries', order=())` (O-46), and
`LeastSquaresOracle('entries', order=(1,))` raises at construction.
- Tests: each output of `query` against the engine and against the reference; `held` holds exactly the names of the
  hold list that the call computed (none computed only to be held); `take` under `jax.jit` with a static size;
  `minibatch_scale` makes the drawn `f` an unbiased estimate of the full `f`, tested exactly on a small `n` by
  enumerating every subset of the batch size; on a drawn batch, `r`, `Jp`, `r_block_sumsq` and `y_block_sumsq` are
  unscaled and the identities `f = 1/2 <r, r>`, `<Jp, r> = <p, g>`, `Jp_norm_sq = <Jp, Jp>` hold in the batch's
  inner product; `check_batch` passes a good batch and raises on each wrong part; `pack_batch` then the uniform
  query equals the ragged query on real parts; the pytree round trip of the batch; the oracle's hash and equality
  by value; bridge against the old kinds through the conversions (one variable, the sample `affine_jet(ww, pp)`);
  the outputs at `order = (1, 1)` against the reference; `order` is required (O-44): `LeastSquaresOracle('probe')`
  raises `TypeError`, for every kind; `LeastSquaresOracle('entries', order=(1,))` raises (O-46); the docstring
  shows the trap of an input jet read with `order=()` as a wrong-against-right example.
- Gate: the full gate.

**P15. The geometry returns the centers.** The backend `frame(x)` returns `(frame, centers)` on the four backend
geometries (the manifold ones keep the second result of `t3_orthogonal_representations`, which they compute today
and discard; the corewise ones return `None` as centers, and still the four-store frame `(B, G, G, G)` until P16).
The geometry gets no consistency member (O-31). The frontend geometries (`manifold.MANIFOLD`, `COREWISE`, the
uniform singletons, `shared_geometry.SharedGeometry`) do
not call the backend `frame` and do not change (O-4).

What breaks, from a search for every call of a backend geometry's `frame` in `t3toolbox/`, `tests/`, `docs/`,
`examples/` and the doctests (calls on the frontend geometries, which keep their `T3Frame` result, excluded); each
reads `frame, _ = geom.frame(x)` from here:
- the library: `backend/optimizers.py` (`_local_model_phases`), `backend/geometry.py` (the aligned padded frame's
  `old = self.frame(x_old)`, and the uniform manifold geometry's `point_norm_sq`, which calls
  `self.point_tangent(self.frame(x_sc))`). No module or page doctest calls a backend `frame`;
- tests: `tests/backend/test_geometry.py` (five calls), `tests/backend/test_curvature.py` (four),
  `tests/backend/test_uniform_curvature.py` (two), `tests/test_dispatch.py` (four),
  `tests/backend/test_optimizers.py` (two), `tests/backend/test_padded_start.py` (seven, one in a test subclass's `aligned_padded_frame`),
  `tests/backend/test_uniform_fitting.py` (one, `_compare_ops`), `tests/backend/test_zero_reference.py` (two, and the
  wrapper geometry whose `frame` delegates to the base one).
- Tests added: the centers equal the second result of `t3_orthogonal_representations`, and
  `fv_centers_consistency_residual` of the returned frame and centers is at rounding level.
- Gate: the full gate.

**P16. The switch, first part: the library and the tests** (O-24: "Split. Two commits."). One commit, reviewed in
four parts. The backend optimizers keep their names (`newton_cg` and the rest) and change their arguments, so the
frontend optimizers and the frontend local model change with them. For this slice only, the frontend optimizers
also accept the old call form `(geometry, kind, sample, data, x0, order=, weight=, ...)` and translate it; P17
deletes the translation.
1. *Geometry.* The corewise frame is the point `(B, G)` on the ragged and uniform corewise geometries (`frame`,
   `base_point`, `retract`, `point_tangent`, the shared corewise post-pass).
2. *Backend model and optimizers.* `LocalModel` (section 2.2); `local_model` builds the frame and makes one `query`
   wanting `f` (and `g` in exact mode); it checks nothing (O-6). `gradient` asks for `g` with `held`; `gn_quadratic`
   for `Jp_norm_sq`; `hvp` for `JtJp`, and in exact mode `JtJp` and `Kp` in one request; `jacobian(p)` for `Jp`;
   `residual` for `r`. Exact mode checks, before any work, that the oracle lists `Kp` in `optional_outputs` (a
   structural check, O-7). The five phases as module-level functions of pytrees; `use_jit` compiles the CG solve and
   the stochastic step, as today. Trial points use policy a: `point_objective` at the trial's raw cores. The
   minibatch is a batch from `draw(rng)`; the regularizer is not scaled; MC-SGD and Adam on an oracle without a
   draw raise (8.3 part 2). The zero reference, the padded start, the cone reference and the switch mode go through
   `query`. `Problem`, `least_squares_problem` and `_minibatch_step_problem` go, with the import of
   `_ScaledRegularizer` and its definition in `backend/regularization.py` (its mention in
   `docs/contributor/fitting_internals.md` is P19's), and so does `uniform_fitting.uniform_least_squares_problem`
   with its doctest: it is annotated `-> bopt.Problem` in a module without `from __future__ import annotations`, so
   it would fail the import of `uniform_fitting` once `Problem` is gone.
3. *Display and record.* The display reads `r_block_sumsq` at the model's frame through `held`, `y_block_sumsq`
   once, and `f_resolution`; without `f_resolution` the trust-region floor is `eps |f|` with `eps` from the dtype of
   `f` (8.4); an oracle without block sums gets the header line and no table. The stored record (the display's
   records, which the frontend returns as `stats['diagnostics']`) keeps `train_err` and `val_err` at every
   iteration in the full shape of the weight, `(d,) + F` or `F`, nothing reduced over modes or orders (8.4). The
   printed table is today's for `m <= 1`; for `m >= 2` it has one column per multi-index, in row-major order
   (O-42). `error_table=True` is added to the frontend `newton_cg` (beside `verbose`), to `make_newton_display` and to
   `format_newton_iter`: `False` prints each row's header lines and no table, and changes no record. The timing laps: `oracle`
   in place of `sweep` and `residual` (O-17), so `stats['timing']` has the keys `frame`, `oracle`, `companion`,
   `gradient`, `curvature`.
4. *Frontend.* `t3toolbox/optimizers.py` takes `(geometry, oracle, batch, x0, ...)`; it maps the geometry to the
   backend one, reduces `x0` to minimal ranks, calls `oracle.check_batch` on the batch and on `val_batch` when the
   oracle has it (O-18, O-29),
   packs them with `pack_batch` for a uniform `x0`, checks a supplied `start_frame`'s consistency in safe mode
   (O-5), and rewraps the result; `least_squares_problem`, `LeastSquaresSetup`, `_resolve_chunk_size`, `_check_kind`
   and the frontend `chunk_size` go. `t3toolbox/fitting.py` gets the frontend `local_model` of section 2.5 over the
   backend `LocalModel`, with tangents in and out, the same-frame guard and the same checks; the six factories go,
   with their re-exports from `t3toolbox/__init__.py`, and with them `_check_ragged_sample`. The translation (one
   private function of `t3toolbox/optimizers.py`): a string where the oracle goes marks the old form; the old
   positional arguments are then read as the old signature names them (`geometry, kind, sample, data, x0`, and for
   `mc_sgd` and `adam` `rng, batch`), as in `topt.adam(t3m.COREWISE, 'probe', ww, pdata, x0c, rng, 20, max_iter=50)`
   (`docs/fitting_and_optimization.md`), and rebuilt into the new form; the kind string gives the oracle; the old weight goes through `_canonical_weight` (kept until P17 for this) and is
   broadcast to the exact shape; the old plain kinds (`'probe'`, `'apply'`, `'entries'`, no `pp`) give the jet shape
   `()`; the old derivative kinds at order `n` give `(n,)`, `(0,)` included, because their data carry an order axis
   of length 1 at order 0 (`sampling_derivatives.t3_probe_derivatives(ww, pp, x, 0)` returns `(1,) + W + (Ni,)` per
   mode; row 2.7), and their sample `(ww, pp)` becomes `affine_jet(ww, pp)`; the old kind `'entries_derivatives'` becomes the oracle `('apply', (n,))` at
   `affine_jet(index_to_one_hot(index, x0.shape), pp)`; derivative data, validation data and weight go through
   `prepare_derivative_data`;
   `val_sample`/`val_data` become `val_batch`; `batch` (an int, for `mc_sgd` and `adam`) becomes `batch_size`; an
   old custom `draw` returning `(sample, data)` is wrapped to return a batch; `chunk_size` is accepted and dropped.

What breaks, from a search of `t3toolbox/`, `tests/`, `docs/`, `examples/` and every doctest for `Problem`,
`least_squares_problem` (both layers), `LeastSquaresSetup`, the six factories, the four optimizers' calls,
`flat_draw`, `make_newton_display`, `local_model`, `_resolve_chunk_size`, `_canonical_weight`,
`_check_ragged_sample`, `_require_exact_capable`, `_minibatch_step_problem`, the kind strings with `_derivatives`,
and the backend corewise geometries. What the translation keeps working moves in P17; everything else moves here:
- tests rewritten: `tests/backend/test_optimizers.py`, `tests/backend/test_optimizer_display.py`,
  `tests/backend/test_geometry.py`, `tests/backend/test_zero_reference.py`, `tests/backend/test_padded_start.py`,
  `tests/backend/test_uniform_fitting.py`, the model parts of `tests/backend/test_curvature.py` and
  `tests/backend/test_uniform_curvature.py`, `tests/test_fitting.py` (its three calls of `_canonical_weight` among
  them), `tests/test_optimizers_frontend.py`, `tests/test_manifold.py` (its `fitting.apply_model` call),
  `tests/test_sharing.py`'s fitting cases, `tests/test_oracle_sweep.py` with `tests/oracle_sweeps/sweep_models.py`,
  `sweep_exact.py`, `sweep_optimizers.py` and `oracle.py` (the kind strings), `tests/test_jax_absent.py` (its
  `bopt.least_squares_problem`, `probe_derivatives_model` and `newton_cg` script),
  `tests/test_aligned_padded_frame_frontend.py` (`topt.least_squares_problem`), `tests/test_chunk_size_estimator.py`
  (`test_resolver`, which calls `_resolve_chunk_size`, and `test_newton_cg_exact_across_chunk_size` are removed), and
  the fitting rows of `tests/test_dispatch.py`. A case that fits from entries derivatives moves to the apply kind at
  `affine_jet(index_to_one_hot(index, shape), pp)` (entry 120). Every file rewritten here also drops its uses of the old kinds, so
  that after P16 only the old kernels' own tests and the bridge reference them;
- module doctests: `backend/optimizers.py`, `backend/uniform_fitting.py`, `backend/optimizer_display.py` (its
  `make_newton_display` example, and the `PhaseSeconds` example, whose laps change), `t3toolbox/fitting.py` (seven
  factory calls, one of them shown raising), `t3toolbox/optimizers.py` (sixteen optimizer calls, four of `least_squares_problem`, one of
  `Problem.local_model`);
- page doctests that call a factory, which the translation does not cover: `docs/fitting_and_optimization.md`
  (four: lines 147, 182, 424, 430) and `docs/numerical_contracts.md` (one, line 94);
- examples that call a factory: `fit_hilbert_from_entries_lbfgs.py` (`entries_model`),
  `fit_hilbert_from_probes_adam.py` and `fit_hilbert_from_probes_optax.py` (`probe_model`),
  `fit_hilbert_tensor_newton_cg.py` and `fit_varied_rank_tensor_newton_cg.py` (`apply_model`): they move to
  `fitting.local_model` with a `LeastSquaresOracle` (its `order` given, `()` for these plain fits) and a
  `LeastSquaresBatch`.
- Pages that name the removed objects in prose only (`docs/api_reference.rst`, `docs/user_guide.rst`,
  `docs/glossary.md`, `docs/naming_conventions.md`, and the contributor pages of section 8) are rewritten in P19
  (the Sphinx configuration is not nitpicky, so a stale `:py:` reference does not fail `sphinx -W`; a stale `:doc:`
  link does).
- Tests added: optimizers judged by outcomes on problems with known answers and by the model's consistency
  (`p` paired with `hvp(p)` equals `gn_quadratic(p)` in Gauss-Newton mode), not by old iteration counts (9.3); a
  model rebuilt each step is one cache key; a new weight or minibatch does not recompile; `want` is a literal; the
  stored record's shapes at `order = ()` for each kind and at `(2,)` and `(1, 1)` for probe and apply, with and
  without a validation batch; the display's table at `(1, 1)`, its columns the multi-indices in row-major order
  (O-42); with `verbose=True, error_table=False` each row prints its header lines and no table, and the stored
  records (`stats['diagnostics']`) are equal to those of a run with the table; the floor `eps |f|`
  on a user oracle without `f_resolution`; a user oracle without a draw raises in `mc_sgd`; a supplied `start_frame`
  with a perturbed store raises in safe mode and runs in `safety.unsafe()`; a library-built frame is not checked; a
  bad batch raises in the frontend before any work; the backend optimizers run on the same bad inputs without a
  check. The translation (temporary, deleted in P17): each old call form gives the result of the new one.
- The frontend-only fit (the answers of 20:15): a test that fits a tensor from derivative data importing nothing from
  `t3toolbox.backend` (`t3toolbox.fitting` for the oracle, the batch, `affine_jet` and `prepare_derivative_data`;
  `t3toolbox.optimizers.newton_cg`), beside the backend-only fit of section 7.5.
- The slow tier (`T3TOOLBOX_SLOW_TESTS=1`, serially) is part of this slice's gate.
- From here nothing in the library calls the old kinds, `Problem`, or the old sampling kernels except their own
  tests, the bridge, and the translation through the new form.

**P17. The switch, second part: the examples and the pages.** The examples and the page doctests that call the
frontend optimizers in the old form move to `(geometry, oracle, batch, x0, ...)`; then the translation, its test
and `_canonical_weight` are deleted.
- examples: `fit_hilbert_from_apply_derivatives_topt.py` (`mc_sgd`, with a custom draw),
  `fit_hilbert_regularized.py`, `fit_hilbert_uniform_newton_cg.py`, `fit_hilbert_uniform_probe_derivatives_newton_cg.py`,
  `fit_per_mode_weight_probes.py`, `fit_probe_display.py` (with `val_sample`, `val_data`),
  `fit_shared_factors_jetted_probes.py`; the examples that fit derivative data use `affine_jet` and
  `prepare_derivative_data`, and
  those of P13 that took `taylor_to_derivatives` of the dense jets take the Taylor jets directly;
- page doctests: `docs/fitting_and_optimization.md` (the optimizer calls, one with a custom `draw`),
  `docs/sharing.md` (two), `docs/getting_started.rst` (two);
- the comments of `examples/fit_hilbert_from_apply_derivatives.py` that name `fitting.apply_derivatives_model`;
- the root `README.md`: its fitting example calls `t3t.newton_cg(t3t.MANIFOLD, 'apply', ww, b, x0, ...)`. No CI step
  runs it, so it is rewritten here by hand and the new snippet is run once to check it (found by P0's review; the
  inventory searches the root `README.md` from P0 on).
- Gate: the full gate, the slow tier. A search for the old call form (a kind string in the second position of a
  frontend optimizer, `val_sample=`, `val_data=`, `chunk_size=`) in `t3toolbox/`, `tests/`, `docs/`, `examples/`
  and the root `README.md` finds none.

**P18. The dry port.** Section 9.

**P19. Documentation.** Section 8. Two commits: the new material (the page, the example, the note, the dependency
diagrams and their script, the reference module's docs, the page of the paper-to-code map), then the rewrites,
removals and moves of existing pages.

**P20. The deletion.** Section 10.

### 5.4 When each old piece stops being called

| Old piece | Last caller in the library | Stops at | Deleted at |
| --- | --- | --- | --- |
| `probing.compute_*`, `assemble_*`, `compute_sigma_hat` (plain) | the old plain kinds | P16 | P20 |
| `sampling_derivatives` (all) | the old derivative kinds; the frontend `*_derivatives` methods | P13 (frontend), P16 (kinds) | P20 |
| `*_precompute_*_frame_sweep*`, `*_from_sweep` | the old kinds | P16 | P20 |
| `ut3_sampling` and `utv_sampling` removed names | the uniform kinds; the uniform frontend `*_derivatives` | P13, P16 | P20 |
| The old kinds (`backend/fitting.py`, `backend/uniform_fitting.py`) | `Problem`; the frontend factories | P16 | P20 |
| `Problem`, `least_squares_problem` (both layers), `LeastSquaresSetup`, the six frontend factories | the optimizers; users | P16 | P16 |
| The old call form of the frontend optimizers, `_canonical_weight` | the translation; the examples and pages | P17 | P17 |
| The frontend `*_derivatives` methods | users | P13 | P13 |
| `chunk_size` (frontend), `docs/chunking.md` | the frontend optimizers | P13 (the page), P16 (the keyword) | P13, P16 |
| `estimate_chunk_size`, `max_chunk_size_within`, `DEFAULT_CHUNK_SIZE` | the frontend optimizers | P16 | P20 |
| `TestJetKernelsShardOverW` | itself (it imports `sampling_derivatives`) | replaced by P11's class | P20 |
| The bridge test | itself | P20 | P20 |
| The copies of `check_perturbation_vectors`, `check_perturbation_index` in `jets` (P3) | nothing (there is no `pp`) | P4d | P4d |

`Problem`, the factories, the old call form and the frontend methods are removed when they switch because they are
interfaces, not kernels.

---

## 6. The tests

| Layer | What | Where | When |
| --- | --- | --- | --- |
| `jet_contract` | the 29 strings against an independent truncated product at `m = 0, 1, 2, 3`; `numpy.polynomial` at `m = 1`; the box pairing; the cases of section 2.4; contraction counts; strictness | `tests/test_jet_contract.py` | P1, rebuilt in P4a |
| Reference | `sampling_definitions` pinned against the dense tensor, the dense tangent tensor, the dense Jacobian and the Hessian | `tests/test_sampling_definitions.py` | P2, P4c |
| By definition (9.3) | every name, every kind, the jet matrix below (entries at `()` only), a random weight, against the reference | `tests/test_sampling_variables.py` | P5 to P11 |
| Identities (9.3) | the adjoint identity of `Jp` and `g`; `p` with `JtJp` equals `Jp_norm_sq`; `g` is the derivative of `f`; held equals recomputed; independence of the layout and the backend; `order = ()` equals the slot `(0, ..., 0)`; a second variable of order 0 equals one variable; `Vc` route equals `D` route; short forms are slices | `tests/test_sampling_identities.py` | P5 to P11 |
| Generic twin | the written-out terms against terms derived mechanically | `tests/_generic_twin.py`, used by the identity tests | P6 |
| Structural | the table equals the declared graph; the slots of `HeldVariables` equal the variables read by others; one wanted variable computes only its reads; unknown names and missing arguments raise; held shapes; the dtype rule; uniform output masks | `tests/test_sampling_graph.py`, the uniform tests | P5, P10, P11, P12 |
| Checkers | each checker passes a good input and catches each wrong one; no backend function calls one; the frontend enforces through them, in safe mode where numerical | `tests/test_jets.py`, the frame tests, the fitting tests | P3, P4, P4d, P14, P16 |
| Reference | the dense functions against interpolation and today's values, at input jets of every kind of the jet matrix; the corewise transpose and Hessian against automatic differentiation of `<c, z(x)>` | `tests/test_sampling_definitions.py`, the curvature tests | P2, P4b, P12 |
| Uniform | equivalence with ragged on real parts over the full `_CONFIGS` matrix; exact output masks; garbage-padded inputs; tails; the two-hole masks on a TT-slack frame, with a mutation check | `tests/test_sampling_variables.py` (uniform cases), `tests/test_uniform_probing.py` | P11, P12 |
| Sharding | the AST sweep over every literal string; `jet_contract`'s product strings; `g` and `Kp` under `shard_map` with `W` sharded, zero all-gathers, equal to unsharded | `tests/test_contractions_sharding.py` | P1, P11 |
| Dispatch | jit each new public function; numpy in numpy out, jax in jax out; eager calls compile once | `tests/test_dispatch.py` | every slice |
| Doctests | every new public function; the page; the 2.3 doctest; the gotcha of a forgotten `order` (O-44) | modules, `docs/` | P2, P3, P4a to P4d, P5, P19 |
| Fitting | outcomes on known answers; model consistency; compile keys; the stored record's shapes; the `eps |f|` floor; a draw-less oracle in MC-SGD; the minibatch scale's identities; a frontend-only fit from derivative data | the rewritten fitting tests | P14, P16 |
| Bridge (temporary) | new outputs against old code that is never rerouted (`sampling_derivatives`, the old kernels, the old kinds), through the stated conversions | `tests/bridge/test_old_kernels_bridge.py` | P6 to P14; deleted in P20 |
| Translation (temporary) | each old call form of the frontend optimizers gives the result of the new one | the frontend optimizer tests | P16; deleted in P17 |

**The jet matrix.** Every test by definition, every identity and the uniform equivalence run over the same jet
shapes and inputs, so that several variables are tested wherever one is: `order = ()`; `(1,)` and `(3,)` with a line
input (`affine_jet(ww, pp)`, today's case); `(1, 2)` with `affine_jet(ww, p1, p2)`; `(2,)` with an input of three
slots (a curve) and with modes of different input jet sizes (one of them of size 1); and **`(1, 2)` with `d = 3`
modes of input jet sizes `(3, 2)`, `(1, 1)` and `(2, 3)`**: a curve in one variable and a line in the other (the
curve that reaches the outputs is the `(2, 3)` mode's, along `s_2`; the third slot of the `(3, 2)` mode is clipped), a
discrete mode (one-hot, one slot) beside continuous ones (entry 120), and an input larger than the box `F = (2, 3)` on
the first axis only, so the leaves' clipping `min(J_in_i, F)` acts on one axis. Only `m = 2` can show a per-axis bug
(the clipping, the packing per axis), so the last case runs wherever the matrix does, packed on the uniform layout.
The entries kind runs at `()` only. Order 16 at `m = 1` where the scan is pinned (one program size). The sizes stay small: the reference evaluates on a grid of `M_1 ... M_m` nodes.

**Tolerances.** Numerical comparisons are to a tolerance scaled to the dtype; structural ones are exact (0.8, 9.3).

**Existing tests, sorted.** The slice lists of section 5.3 are the full record; this is the summary.

| File | Fate |
| --- | --- |
| `tests/test_jet_recurrence.py`, `tests/_jet_trs_twins.py` | Test old kernels only; deleted in P20 |
| `tests/backend/test_from_sweep_reuse.py` | Its calls updated in P12; deleted in P20 |
| `tests/test_chunk_size_estimator.py` | Its `newton_cg` case fed through the merged `probe` in P13; that case and `test_resolver` removed in P16; its estimator cases deleted in P20 |
| `tests/test_contractions_sharding.py` | Extended in P1, P4a and P11; its class `TestJetKernelsShardOverW` (imports `sampling_derivatives`) deleted in P20, replaced by P11's |
| `tests/test_probe_derivatives.py` | Folded into the frontend sampling tests in P13, on the dense functions |
| `tests/test_uniform_probing.py`, `tests/test_manifold.py` | `test_manifold.py`'s `dense_probe` call in P2; their calls of the changed backend names in P12, of the frontend methods in P13; `test_manifold.py`'s `fitting.apply_model` call in P16 |
| `tests/test_tucker_tensor_train.py` | Its `dense_probe` call in P2; its frontend sampling calls in P13 |
| `tests/test_uniform_manifold.py`, `tests/test_uniform_tucker_tensor_train.py` | No call of a changed or removed sampling name (searched); unchanged unless a slice's search finds one |
| `tests/backend/test_curvature.py`, `tests/backend/test_uniform_curvature.py`, `tests/backend/_curvature_oracles.py` | The changed corewise transposes in P12; the backend `frame` in P15; the model parts in P16; their comparisons with the old two-hole and jet functions removed in P20; `_curvature_oracles.py`'s dense oracles kept as test helpers |
| `tests/test_fitting.py`, `tests/backend/test_uniform_fitting.py` | Their removed-method calls in P13; the backend `frame` in P15 (`test_uniform_fitting.py`); rewritten in P16 |
| `tests/backend/test_optimizers.py`, `test_padded_start.py`, `test_zero_reference.py` | Removed-method calls in P13; the backend `frame` in P15; rewritten in P16 |
| `tests/backend/test_geometry.py` | The backend `frame` in P15; rewritten in P16 |
| `tests/backend/test_optimizer_display.py`, `tests/test_optimizers_frontend.py`, `tests/test_sharing.py` (fitting), `tests/test_oracle_sweep.py`, `tests/oracle_sweeps/`, `tests/test_jax_absent.py`, `tests/test_aligned_padded_frame_frontend.py` | Removed-method calls in P13 where they have them; rewritten in P16 |
| `tests/test_dispatch.py` | Rows added from P1; the rows of P1 to P3 rewritten in P4a to P4d; changed calls in P12, P13, P15; fitting rows in P16; old rows removed in P20 |
| `tests/test_contractions_interpreter.py` | Extended in P1; its jet records and derived strings rewritten in P4a |
| `tests/test_import_graph.py`, `tests/test_jax_absent.py` | New modules registered as they land; no backend module imports `safety` (P3); `test_jax_absent.py`'s calls of the dense functions, `jets` and `sampling_definitions` in P4b to P4d |
| `tests/test_jet_contract.py`, `tests/_jet_contract_cases.py` | Written in P1; rebuilt in P4a (29 strings, the one jet group) |
| `tests/test_sampling_definitions.py` | Written in P2; its dense tests in P4b, its definitions in P4c |
| `tests/test_jets.py`, `tests/test_data_preparation.py` | Written in P3; the dense calls in P4b; rewritten for the jet shape in P4d |
| `tests/test_frame_variations_format.py`, `tests/test_uniform_frame_variations_format.py` | `is_consistent` cases in P4 |

---

## 7. The areas that have been tricky in the past

### 7.1 Safe mode against unsafe mode

- **The rule** (the answers of 19:30 and 20:04): the backend gives checkers, not guarantees. "There is no safe mode
  in the backend." The backend never imports `t3toolbox.safety` (pinned in P3). For each thing the frontend checks,
  the backend has a checker a raw-data user can call: a residual function that never raises and reads no mode, or a
  structural checker that raises. Backend functions do not call them: "If the user passes a bad input to a backend
  function, that is on them."
- **The backend's checkers in Part 1.** Of a frame: `fv_frame_consistency_residual`, the centers' residual, their
  uniform twins (P4), beside today's `fv_frame_orthogonality_residual` and its twin. Of a sample, its data and its
  weight: `check_sample`, `check_data`, `check_weight` (P3, in the jet shape from P4d); for a whole batch,
  `LeastSquaresOracle.check_batch` (P14).
- **What the backend still checks, structurally, because it must to run:** the engine's unknown names, missing
  arguments and held shapes (section 2.3 step 1), the operand ranks in `jet_contract` (5.4), and exact mode's
  `optional_outputs` (P16). Nothing numerical.
- **What the frontend enforces.** Structural, always: the sample through `check_sample` in the merged methods; the batch through
  `check_batch` where a batch enters the optimizers or `fitting.local_model`; the same-frame guard. Numerical, in
  safe mode only, skipped in `safety.unsafe()` and under a jax trace: a frame the caller supplied (`start_frame`, or
  the `frame` of `fitting.local_model`), `L` against `R` through `D`, once when the model is built, through the
  cached `consistency_residual` (O-5); a zero weight in the frontend `dual_jet_to_jet` (O-20); and today's checks of
  a supplied `start_frame` (orthonormal; represents `x0`). A frame the library built in the same call is not
  checked; centers and `U` are not checked in the frontend.
- **The compiled phases** (rows 4.2 and 8.3 as amended on 2026-11-16, 19:30; O-32). "Checked on the host" means by
  the frontend, before it calls the backend optimizer; the backend optimizers check nothing before a compiled
  phase.
- **No dense tensor** in any runtime check (4.2): the residuals of P4 go through the cores.
- **Tests.** For every frontend check: it raises in safe mode; it is silent in `safety.unsafe()` and under
  `jax.jit`; the numbers are identical in both modes; the check forms no ambient-sized array. For every backend
  checker: it returns the residual (or raises, for a structural one) when called, and no backend function calls it
  (the backend run on a bad input gives a result, not an error).

### 7.2 The ragged against the uniform layout

- One set of step functions; the layout from the structure of the inputs (layout first, then the count); sweeps
  through `xscan`; map-type operations one site at a time, a loop on ragged and `xmap` on uniform (O-19); every
  `jet_contract` with `'loop'` on ragged and `'scan'` on uniform (O-10), which scans only where `m = 1` (section
  2.4).
- **There is no don't-care slot on a jet axis** (entry 116: an array's jet sizes mean zeros beyond them). Every slot
  of a jet axis is a coefficient. When modes of unequal jet sizes are packed into one array, the zeros written by
  `ut3_operations.pack_vectors` with its new keyword `len_J` (P11; through `pack_if_ragged`, which the twins of P12
  and `pack_sample` and `pack_batch` of P14 call with `len_J=len(order)`; a mismatch in `W` still raises) are the coefficients the
  ragged input has there: data, not padding. So the jet axes need no mask, and the uniform equivalence contract
  (`docs/uniform_equivalence_contract.md`) gains one sentence: the don't-care slots are those of the masks (rank
  and mode padding); the slots of a jet axis are data (O-45). Nothing checks them, and there is no checker: a
  packed input with a nonzero there is another valid input jet, so a checker would have nothing to tell wrong from
  right. A user who packs by hand gets the polynomial they packed; the docstrings of the twins and of `pack_batch`
  say so. The leaves on the uniform layout have the packed jet sizes, clipped to `F`, and agree with ragged on the
  slots ragged has and are zero beyond them.
- Mask once on entry, each input with the frame's own masks (`core_directions` with the frame's `D` and `L` masks,
  never the point's, a superset at TT slack); steps mask-free; core-shaped outputs re-masked on exit by the engine.
  The uniform frame `.data` carries host-numpy masks that `partition_static` keeps out of the trace, as today.
- **Tests** (`docs/contributor/testing_strategy.md`): equivalence with ragged on real parts for every name over the
  full `_CONFIGS` matrix (forced-larger padding, multi-axis `C` and `K`, ranks varying across `C`); exact output
  masks derived non-circularly; garbage-padded inputs; tails on both layouts; `Kp` on a TT-slack frame, with a
  mutation check that removes the mask on `core_directions` and must turn the test red (P11).

### 7.3 The dispatch between numpy and jax

- `tree_contains_jax` over every array input, at the lowest level; no `use_jax` argument; uniform masks with `np`.
- Every scan body re-derives `xnp` from all its own arguments (`scan_body_principles.md` rule 3).
- The dtype rule is absorbing (5.6): jax over numpy, lower precision over higher, which is the opposite of numpy's
  promotion; the engine casts once on entry and creates no array with a default dtype (today's plain sweeps do; the note to 5.6).
- **Tests.** `test_dispatch.py` jits every new public function (a stray `np.` on a tracer raises); jax in gives jax
  out; the dtype matrix of P10; `test_jax_absent.py` imports every new module without jax; the optimizers'
  `use_jit=True` auto-conversion moves the batch to jax (today it moves `sample` and `data`,
  `_prepare_jit_inputs`); `HeldVariables` and the checkers import and run without jax.
- `index_to_one_hot` returns numpy arrays for a numpy index and jax arrays for a jax index, in the default floating
  dtype; it is then an input like any other (the dtype rule of 5.6).

### 7.4 Shared Tucker factors

- Sampling does not know about sharing (4.5): the engine takes the frame and returns blocks before projection; the
  shared geometry's projection, retraction and tied lift stay in the geometry wrapper (1.2).
- What changes for sharing is in the geometry only: the corewise frame becomes the point, so the shared corewise
  `retract` (which reads `base_point(frame)` today) and the uniform twin follow; `frame(x)` returns centers.
- `Kp` reaches the tied lift through the oracle (`core_directions` is the lifted direction), as
  `corewise_hessian` does today.
- **Tests.** `tests/test_sharing.py`'s fitting cases ported in P16; the shared exact local model under jit
  (`test_jit_shared_*` in `test_dispatch.py`); a shared corewise retraction preserves tying; the example
  `examples/fit_shared_factors_jetted_probes.py` rewritten and run by CI.

### 7.5 The razor between backend and frontend

- **The rule of the two users** (the maintainer, 2026-11-16, 19:54, `TARGET_DECISIONS.md` "The two users"): the
  frontend user and the backend user "each must be able to do all the important operations with tools in their own
  layer". Nontrivial logic lives in a backend function that the frontend calls, within reason.
- **The backend user** has the engine, `HeldVariables`, the oracle and the batch, `pack_batch`, `local_model`, the
  optimizers, the display, the helpers of `jets` and `data_preparation`, the checkers (section 7.1), the dense
  functions and the reference module.
- **The frontend user** has the merged sampling methods, `unscale_coordinates` on both tensor classes, and
  `t3toolbox.fitting` and `t3toolbox.optimizers`: the frontend `local_model`, the four optimizers, the oracle and the
  batch, the preparation functions and the jet helpers under their backend names, and its own `dual_jet_to_jet`
  (section 2.5). A frontend user fits from derivative data without importing the backend.
- **Where the logic lives.** Every check the frontend makes calls a backend checker (`check_batch`, the frame
  residuals); packing is the backend's `pack_batch`; the old frontend logic that a backend user would have had to
  rewrite (`fitting._canonical_weight`, `fitting._check_ragged_sample`) goes, into `check_weight`, `check_sample` and
  `check_data`. The frontend `dual_jet_to_jet` adds only the safe-mode check before calling the backend one.
- **The stated exceptions.** Writing an oracle is backend-level work: the frontend optimizers take any oracle object,
  so they drive a user-written one, but there is no frontend way to write one. The engine has no frontend entry
  ("If a use case arises, we can add the frontend version at that time"). The reference module is backend only.
  The dense functions are not wrapped: the frontend has no class for a dense array.
- Corollary: a backend user who gets `g` from the engine gets blocks before projection; the geometry's `project` is
  the one documented follow-up, and `local_model` does it for them.
- **Tests.** A backend-only fit in a doctest: build the batch, the oracle, call `newton_cg` from
  `backend.optimizers`, read the result, with no frontend object. A frontend-only fit from derivative data, a test
  that imports nothing from `t3toolbox.backend` (P16).

### 7.6 Sharding

- Part 1 promises nothing new about sharding (7.7 is open). It must not break what holds today: every sub-axis of
  every group is shardable with no all-gather, because `contract` never reshapes.
- `jet_contract` reshapes nothing either: both forms slice jet axes only, never `W`, at any `m`.
- Every string at a call site is a literal, so the AST scan of `tests/test_contractions_sharding.py` sees it.
- The arrays of a batch are top-level arguments of the compiled phases, not closures (8.3 note).
- What holds today beyond the per-string sweep: the jetted transpose and the two-hole, run under `shard_map` with
  `W` sharded, compile with no all-gather and match the unsharded result (`tests/test_contractions_sharding.py`,
  class `TestJetKernelsShardOverW`). P11 pins the same property on the engine; the old class goes with
  `sampling_derivatives` in P20.
- **Tests.** The sharding sweep on the new literals and on `jet_contract`'s product strings (P1); `g` and `Kp` from
  the engine at `order = ()`, `(3,)` and `(1, 1)`, ragged and uniform, under `shard_map` with `W` sharded (P11).

### 7.7 Compiling with jit

- **Statics and leaves.** Static (`aux_data`, value-hashed): the geometry, the oracle (`kind`, and `order`, a tuple
  of Python integers), the regularizer, `want` (a literal tuple at every library call site), the filled slots of
  `HeldVariables`, uniform masks and shapes, `GroupLengths`, `over_orders`, the jet shape passed to each
  `jet_contract` call. The input jets' jet sizes are array shapes: a sample of other jet sizes recompiles, as a
  sample of another `W` does today. Leaves: the batch (the weight, the minibatch, its scale, its
  indices), the frame, the centers, the held arrays, `f`, the curvature state.
- So a new weight, a new minibatch and a new frame do not recompile; a rank change does, as today.
- **Scan bodies** are module-level; statics reach them through factories memoized on hashable scalars, never
  arrays.
- **What is compiled in Part 1**: with `use_jit=True`, the CG solve and the stochastic step, as today. The five phases
  are module-level functions of pytrees so that Part 2 can compile more.
- **Tests.** A model rebuilt at each step is one cache key; a draw with the same shapes compiles once; a new weight
  does not recompile; the CG solve compiles once per fit; an eager scan compiles once per shape.

---

## 8. The documentation

**New** (9.7: each kind of fact has one source; every other place is generated from it or tested against it).

| Layer | What | Source of truth, and its test |
| --- | --- | --- |
| 1. The engine's docstring | The table of variables (name, shape, what it reads, which arguments it needs) and a pointer to the page | The declared graph; the exact test of P5 |
| 2. The page `docs/sampling_variables.md` | The naming scheme in half a page; a section per grade with an entry per variable (diagram, definition, recurrence, shape, a doctest against `sampling_definitions`); the identities, each with a doctest; the dependency structure; sizes; how to use them (holding, a custom oracle, modified variables) | Grows out of `SAMPLE_VARIABLES.md`; CI runs its doctests |
| 3. The example `examples/tour_sampling_variables.py` | A small problem, a few variables computed and checked, the identities, a held variable reused | CI runs it |
| 4. The note `docs/sampling_variables.tex` (+pdf) | Why the recurrences equal the subnetworks; jets; the adjoint variables; the arrows | Its notation is the code's names |
| The reference module | `sampling_definitions`, one doctest per function | The tests of P2 |
| The dependency table and diagrams | The table of reads, and the diagrams that show it | Their source is the declared graph (9.7). A script (N29) generates, from the graph, one layered picture per grade and one per output, as SVG files committed beside the page, run by hand when the graph changes; a test checks that every name has a diagram and that the committed files match a fresh run (O-23). Every diagram shows its arrows (9.5) |
| The bad-scaling example | `examples/fit_badly_scaled_data.py`: bad as given, good after the preparation functions (6.7); it computes the two estimates (a length for the directions, a coordinate scale) inline (O-14); a frontend example, with the functions from `t3toolbox.fitting` | From `scripts/review/bad_scaling_example.py`; CI runs it |
| The paper-to-code map | `docs/paper_to_code.md`: T4S's letters and variables against the code's, updated for the new variables, with the differences explained ("the new variables are different, but most of them have some kind of analog in the paper. We explain the differences", O-15). It takes the notation map out of `CLAUDE.md`, which keeps a one-line pointer, and the probing map of `docs/entries_apply_probe.md` section 8, which links to it (O-33) | Its letters are checked against the arXiv version, as the citations in the tree are |
| The derivative-data doctest | In `backend/jets.py` (2.3) | From `scripts/review/adapter_doctest_draft.py` |

**Rewritten.** `docs/entries_apply_probe.md` (its table of the transposes' call forms, section 4) and
`docs/transposes.md` (its table and headings: the call forms in N16's order; sum by default; `residual`),
`docs/fitting_and_optimization.md` (the oracle, the batch, the new signatures; the jet shape and input jets, with
`affine_jet` for the line and `index_to_one_hot` for derivatives at grid points; the weight's exact shape `(d,) + F`
or `F`; ones by default with the asterisk of 2.5; derivative data prepared outside; data given by total degree, in
the box with weight zero; the display's `error_table=False`, which keeps the header lines and drops the table, and
the table's columns for several jet variables), `docs/batching_and_stacking.md` (the jet group `J` beside the three stacks, and the sample
stack on the outputs), `docs/grouped_contractions.md` (`jet_contract`, with one table of the letters that says which
are grouped stack axes, `C`, `W`, `K`, and which is the jet group, `J`, whose meaning is multiplication of
polynomials truncated to a box and not alignment of axes; the output's jet shape; the corner coefficient of a
readout; why the truncation is a box and not a total degree), `docs/frame_variations.md`
(centers), `docs/numerical_contracts.md` and `docs/contributor/numerical_contract_catalog.md` (the backend's
checkers, not guarantees; what the frontend enforces, and when), `docs/sharing.md`, `docs/rank_continuation.md` and `docs/aligned_padded_frame.md` where they call
optimizers, `docs/uniform_equivalence_contract.md` (and its sentence of O-45: the slots of a jet axis are data, never
don't-care), `docs/contributor/fitting_internals.md`,
`docs/contributor/curvature_internals.md`, `docs/contributor/scan_body_principles.md` (its examples name old kernels),
`docs/contributor/batching_internals.md`, `docs/contributor/parameters_not_closures.md` (kinds),
`docs/contributor/precompute_and_caching.md` (where it says the carrying cost of a held object is not memory, and the
scope ladder's "per-problem" row), `docs/contributor/verification_records.md`, `docs/api_reference.rst`,
`docs/user_guide.rst`, `docs/getting_started.rst`, `docs/design_notes.rst`, `docs/exact_hessian_two_hole.tex` (the
names); and the pages whose exemplars or citations name old kernels or the old fitting API:
`docs/reading_signatures.md` and `docs/contributor/signature_style.md` (their exemplar is `compute_mu`; CLAUDE.md names
`backend/probing.py` as the signature style's reference module, and a new exemplar is chosen among the engine's
public functions), `docs/contributor/uniform_polymorphism.md` (its loop table cites `compute_mu`,
`compute_eta_jets`), `docs/contributor/uniform_internals.md` (it says the kinds re-mask),
`docs/contributor/padded_frame_internals.md` (`least_squares_problem`, `Problem.local_model`),
`docs/contributor/sharing_internals.md` (`Problem.local_model`), `docs/contributor/ambient_derivative_transpose_note.md`.
The examples that fit derivative data use `prepare_derivative_data`. Pages whose doctests call a changed function
are rewritten in the slice that changes it (section 5); P19 rewrites the prose.

**Removed or moved.** `docs/chunking.md`, in P13 (Part 2 writes the sizing page).
`docs/symmetric_probe_derivatives.tex` is archived: moved to `dev/archive/` in P19 (O-15). Nothing in `docs/`,
`t3toolbox/`, `tests/` or `examples/` links to it (searched); `dev/paper_scope.md` names it, and is not changed.

**Records.** `docs/glossary.md` (the variables, the oracle, the batch, held, centers; the old names removed);
`docs/naming_conventions.md` (the sampling modules, the new modules, the corewise-call rule) and
`docs/contributor/naming_rules.md`; `docs/contributor/deferred_and_rejected.md` (the "kwarg-unified derivative
interface" entry reversed, 6.6 note; jets in several variables are built, so the entry the earlier plan foresaw for
them as a future feature becomes three records: the fast algorithms for `m >= 2` deferred to Part 2 behind
`jet_contract`; truncation by total degree rejected, because the dual jets need the box (row 2.4); the entries kind
with jets removed, with the reason of entry 120 and the migration to apply at one-hot vectors); `dev/OPEN_QUESTION_extension_surface.md` given its resolution (3.4) and
archived; `CLAUDE.md` (the paper-to-code map replaced by a pointer to its page; the sampling section; the
safe-mode paragraph, whose "Frontend-only for now (a backend mirror for raw-`.data` users is deferred)" becomes the
backend's checkers; the shipped surface; the module list);
`docs/release_notes.md` with the old-to-new name map (`SAMPLE_VARIABLES.md` section 10), and the upgrade of a jet
call: the released `probe_derivatives(ww, pp, 3)` becomes `probe(affine_jet(ww, pp), order=(3,))`, in Taylor
coefficients (`taylor_to_derivatives` gives the derivatives back), and `entries_derivatives(index, pp, n)` and its tangent forms, which are
removed, become apply at `affine_jet(index_to_one_hot(index, shape), pp)`; `CHANGELOG.md`.

---

## 9. The consumer: the dry port

**What it is.** A script `dev/graded_messages/scripts/dry_port_t3polynomial.py` that makes T3Polynomial's main fit
call through the new interface, without changing a file of T3Polynomial or of the library (small item 5). It
works by replacing, at run time and only inside its own process, the two places where T3Polynomial meets the
removed API.

**Where T3Polynomial meets the library today** (`t3polynomial/fitting.py`):

| Place | Today | Removed in |
| --- | --- | --- |
| `_newton_fit` | `topt.newton_cg(geometry, 'probe_derivatives', (ww, pp), data, X0, order=jet_order, weight=weight, **kwargs)` | P17 (in P16 the translation still takes the old form) |
| `fit_rank_continuation` and `fit_rank_continuation_adaptive`, scoring each level | `X.probe_derivatives(ww, pp, jet_order)`, then the consumer's own `weighted_misfit` and `forward_summed_reverse_misfit` | P13 |
| `polynomial_predict` | `T.probe(ww)[-1]` | unchanged (`order=()`, plain vectors; the sample is `probe`'s only positional argument under N16 too) |

T3Polynomial calls no transpose, corewise transpose or backend sampling function of the library (searched), so
the order of N16 does not reach it.

Its data are derivative jets (slot `j` is the `j`-th `s`-derivative, `t3polynomial/jets.py`); its weight is `(J+1,)`
per slot or `(degree+1, J+1)` per mode and slot; its options include `val_sample`, `val_data`, `padded_from`,
`g0_reference`, `hessian`, `cg_forcing_power`, and in some callers `chunk_size=None`. It reads the `stats` keys
`newton_steps`, `newton_accepted`, `stop_reason`, `history`, `diagnostics`, and in the adaptive scheme
`start_frame`, `start_movements`, `cone_movements`.

**The two replacements.**

1. `_newton_fit` is replaced by a function with the same signature that:
   - converts with `prepare_derivative_data(...)`: the data, the validation data and the weight to Taylor form, a
     per-slot weight broadcast to `(d, J+1)` first, because the weight's shape is exact (2.5);
   - builds `batch = LeastSquaresBatch(affine_jet(ww, pp), y, weight)`, `val_batch` likewise, and
     `oracle = LeastSquaresOracle('probe', order=(J,))`; the conversions take `(J,)` too
     (`prepare_derivative_data('probe', (J,), ...)`);
   - calls `newton_cg(geometry, oracle, batch, X0, val_batch=val_batch, **kwargs)`, with `chunk_size` dropped from
     the options and `val_sample`, `val_data` turned into `val_batch`;
   - keeps T3Polynomial's own handling around the call (the uniform and jax conversions of `X0` and
     `padded_from`, the warm-start warning) as it is.
2. The scoring call `X.probe_derivatives(ww, pp, jet_order)` is given back to `TuckerTensorTrain`, in the script's
   process only, as `taylor_to_derivatives('probe', (jet_order,), X.probe(affine_jet(ww, pp), order=(jet_order,)))`,
   so that the consumer's
   misfits keep their derivative form.

The script checks that every `stats` key T3Polynomial reads is present, with the meaning it has today; `timing`
is not read by T3Polynomial (its keys change, O-17).

**What it checks.** On T3Polynomial's toy problem (a known polynomial, `t3polynomial/toy.py`), two levels of
`fit_rank_continuation` and of `fit_rank_continuation_adaptive`, on each backend (`'ragged'`, `'uniform'`,
`'uniform_jax'`): the fit reaches the known answer to the validation error the release reaches; the display's table
appears; every `newton_cg` option T3Polynomial passes is accepted. The script's output is recorded in the folder
like the other checks.

**What the consumer's calls become** (entry 120 asks for them; read from `/home/claude/t3polynomial`, not run).
T3Polynomial uses one jet variable and a line input everywhere: its sample is `(ww, pp)` and its order the integer
`jet_order`. Its library calls become, after P17:

| Today (`t3polynomial/fitting.py` and its scripts) | After |
| --- | --- |
| `topt.newton_cg(geometry, 'probe_derivatives', (ww, pp), data, X0, order=J, weight=weight, ...)` (`t3polynomial/fitting.py`, `_newton_fit`, lines 225 and 236) | `y, w, val = prepare_derivative_data('probe', (J,), data, weight if weight is None or weight.ndim == 2 else np.tile(weight, (d, 1)), val_data)`; `newton_cg(geometry, LeastSquaresOracle('probe', order=(J,)), LeastSquaresBatch(affine_jet(ww, pp), y, w), X0, val_batch=..., ...)` |
| `X.probe_derivatives(ww, pp, J)` (scoring: `t3polynomial/fitting.py` lines 384, 385, 597, 598; and the scripts `d05`, `d06`, `d09`, `d11`, `e01`, `g05`, `n01`, `o01`, `s00b`, `s05`, `x02`, `x06`, `x07`) | `taylor_to_derivatives('probe', (J,), X.probe(affine_jet(ww, pp), order=(J,)))` |
| `topt.least_squares_problem(t3m.MANIFOLD, 'probe_derivatives', ...)` (the scripts `d07`, `d08`, `d11`, `g06`, `g06b`, `g06c`, `h01`), and the backend `least_squares_problem` with `bfit.probe_derivatives_kind` (`g01`) | `fitting.local_model(t3m.MANIFOLD, oracle, batch, X)` with the oracle and the batch above; the backend `local_model` with the same oracle and batch |
| `tfit.probe_derivatives_model(...)` (the scripts `x06`, `x07`) | `fitting.local_model(...)`, as the row above |
| `T.probe(ww)` | unchanged |

`(J,)` is right for every `J`, `0` included: the old derivative kind at order 0 has one slot. An integer is not
accepted for `order` (O-36). Nothing of T3Polynomial depends
on the entries kind. The dry port covers the first two rows; the scripts are not part of it.

**What it is for.** A port of the consumer is part of "done" (small item 5). With both replacements installed,
T3Polynomial's own benchmark runs unchanged on the branch, which is what Part 2's benchmark against the released
2026.5.0 needs (O-1, O-2). Part 1 runs no benchmark.

---

## 10. The last step: the deletion

The old kernels are deleted as the last step of Part 1, on the tests alone (O-1, O-2: "Yes, delete and use 2026.5.0
branch for comparison in Part 2"). No benchmark gates it.

**Before the deletion**, all of:

1. The tests by definition and the identities pass, on both CI generations, and the slow tier passes serially.
2. The full suite on the jax-0.4.30 / numpy-1.26 floor, as at a release.

**The deletion**, one commit:

- `backend/sampling_derivatives.py`, whole.
- In `backend/probing.py`, `apply.py`, `entries.py`, `ut3_sampling.py`, `utv_sampling.py`: the 74 removed names of
  section 2.4 that are not in `sampling_derivatives`.
- The old kinds and their factories (`backend/fitting.py`, `backend/uniform_fitting.py`).
- The tests of old kernels (section 6), the bridge test, the class `TestJetKernelsShardOverW`.
- Every remaining reference to what P20 deletes: the module `sampling_derivatives`, the removed names of the five
  other sampling modules, and the old kinds of `backend/fitting.py` and `backend/uniform_fitting.py`. The search reads
  imports under any alias (`import t3toolbox.backend.sampling_derivatives as pd`, `from t3toolbox.backend import
  fitting as bfit`) and then every attribute of that alias, in code and in every module and page doctest; a grep
  adds string references (`mock.patch.object(sd, 'estimate_chunk_size')`). At this commit it finds, outside the
  defining modules:

  | Where | What | Removed in |
  | --- | --- | --- |
  | `t3toolbox/tucker_tensor_train.py`, `manifold.py`, `uniform_tucker_tensor_train.py`, `uniform_manifold.py` | the `*_derivatives` backend functions and the `pp` validators | P13 |
  | `t3toolbox/fitting.py`, `t3toolbox/optimizers.py` (with its lazy import of `sampling_derivatives` in `_bfit_pd`), `backend/optimizers.py` (a doctest calls `sd.t3_probe_derivatives`) | the old kinds, `uniform_sampling_kind`, `uniform_least_squares_problem`, the chunk estimator | P16 |
  | `examples/fit_hilbert_from_apply_derivatives.py`, `_flat.py`, `_topt.py`, `fit_hilbert_uniform_probe_derivatives_newton_cg.py`, `fit_shared_factors_jetted_probes.py` | `pd.dense_apply_derivatives` or `pd.dense_probe_derivatives`, for the dense ground truth | P13, to `dense_apply` and `dense_probe` |
  | `tests/test_probe_derivatives.py`, `test_tucker_tensor_train.py`, `test_uniform_probing.py` | the `*_derivatives` functions as comparisons | P13 |
  | `tests/test_jax_absent.py` (in its subprocess script, a string: `bfit.probe_derivatives_kind(order)`, line 71) | an old kind | P16 |
  | `tests/test_fitting.py`, `tests/test_optimizers_frontend.py` (also by `mock.patch` strings), `tests/backend/test_optimizers.py`, `test_optimizer_display.py`, `test_padded_start.py`, `test_zero_reference.py`, `test_uniform_fitting.py` | the old kinds, the `*_derivatives` data makers, the chunk estimator | P16 |
  | `tests/backend/test_curvature.py`, `_curvature_oracles.py`, `test_uniform_curvature.py` | comparisons with the old two-holes, sweeps and jet functions; the old kinds | the kinds in P16; the rest here |
  | `tests/test_dispatch.py` | rows for old functions and old kinds | the fitting rows in P16; the rest here |
  | `tests/test_jet_recurrence.py`, `tests/_jet_trs_twins.py`, `tests/backend/test_from_sweep_reuse.py`, `tests/test_chunk_size_estimator.py`, `tests/test_contractions_sharding.py` (`TestJetKernelsShardOverW`) | old kernels only | here |
  | `docs/chunking.md` | the chunk estimators | P13 |
  | the pages and the note of section 8 | names in prose | P19 |
  | `dev/graded_messages/scripts/` (14 scripts, among them `graded_messages_prototype.py`, `correspondence_check.py`, `review/levels_jax_check.py`, `review/sample_variables_check.py`) | old kernels and old kinds, as the reference the prototypes were checked against | not changed: they stay as records that run against `v2026.5.0`, and this commit says so in the folder's README (O-26) |

  `CLAUDE.md` names none of them in code. The grep below confirms none is left in the library, the tests, the docs
  and the examples.
- `__all__` lists, `docs/api_reference.rst`, the import-graph pins, the inventory of P0 re-run to zero references.
- `dev/graded_messages/README.md`: a line saying the prototype scripts run against `v2026.5.0` (O-26).

Its own gate: the full gate, the slow tier, `sphinx -W`, and a grep over `t3toolbox/`, `tests/`, `docs/` and
`examples/` (not `dev/`, `CHANGELOG.md` or `docs/release_notes.md`, which name the removed names on purpose) that finds
none of the 131 removed names (129 before the jet group: the two validators of `pp` no longer move to `jets`), nor
`chunk_size`, `sum_over_probes`, `ztildes`, or a keyword `pp=` of a sampling function.

---

## 11. The open points

### 11.1 How they were answered

The maintainer answered O-1 to O-28 on 2026-11-16, 18:35 to 20:35, and O-29 to O-35 at 20:59 (`TARGET_DECISIONS.md`,
the note "The plan's open points, as the maintainer answers them" and the notes after it). He answered O-36 to O-47,
the points of the change for the generalized jet group, on 2026-11-17 at 16:21 and O-45 at 16:26
(`TARGET_DECISIONS.md`, the note "The open points of the jet-group plan, as the maintainer answers them"; log
entries 121 and 122). The plan states each answer where it uses it.

| Point | Answer |
| --- | --- |
| O-1 | No benchmark gate in Part 1; the benchmark moves to Part 2 (section 10) |
| O-2 | As O-1: the old kernels are deleted on the tests alone; Part 2 compares with the released 2026.5.0 |
| O-3 | The engine takes the uniform `.data`, masks once on entry, re-masks on exit; the twins are thin wrappers (2.4, P11, P12) |
| O-4 | `(frame, centers)` on the backend protocol only; the frontend geometries and `COREWISE`'s `T3Frame` unchanged (2.4, P15) |
| O-5 | A frame the caller supplied: `L` against `R` through `D`, once, at the build, in the frontend; nothing else (2.5, 7.1) |
| O-6 | No safe mode in the backend: checkers only; the frontend enforces through a cached `consistency_residual` (2.3, 7.1, P4) |
| O-7 | `optional_outputs` (2.4, P14, P16) |
| O-8 | `HeldVariables`, a dictionary-like frozen record in `backend/sampling_variables.py`; the engine takes `held=` (2.2, P5) |
| O-9 | One frontend `fitting.local_model`; the factories, `least_squares_problem`, `LeastSquaresSetup` go (2.5, P16) |
| O-10 | `jet_contract` takes `over_orders`; the engine passes `'scan'` on uniform, `'loop'` on ragged; both built (2.4, P1) |
| O-11 | Every floating-point input whose entries could affect the output counts (2.3 step 4, P10) |
| O-12 | The raw form (2.3, P8) |
| O-13 | Not accepted (2.2, P8) |
| O-14 | Not shipped; computed inline in the example (2.4, 8) |
| O-15 | The `.tex` note archived; the paper-to-code map updated and moved to a page in `docs/` (8) |
| O-16 | Both tiers: reachable from `t3toolbox.fitting` (2.5, P3) |
| O-17 | The lap `oracle` (P16) |
| O-18 | A backend checker for the weight; one checker for a batch; the engine checks only what it needs to run (2.3, 2.4, 7.1) |
| O-19 | One site at a time through `xmap` on uniform, for every order; to be revisited in Part 2 (1, 2.3, P11) |
| O-20 | The backend divides; the frontend function raises in safe mode (2.4, 2.5, P3) |
| O-21 | The scale belongs to the batch's inner product: `f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp`, `f_resolution` (2.4, P14) |
| O-22 | A vector of the type of `z`, no weight, turned inside; tested against automatic differentiation (2.4, P12) |
| O-23 | Diagrams generated from the declared graph, committed, tested against a fresh run (8) |
| O-24 | Split in two commits, with a translation of the old call form for one slice (P16, P17) |
| O-25 | A temporary test file, deleted in the last slice (5.1) |
| O-26 | The prototype scripts stay as records run against `v2026.5.0` (10) |
| O-27 | `dense_probe`, `dense_apply`, `dense_entries` with `pp` and `order`, backend, used by the reference module, the examples and the tests (2.4, P2, P13). Amended by entries 118 to 120: input jets and the jet shape, no `pp`; `dense_entries` has no jets (P4b) |
| O-28 | The ambient transposes in N16's order and names, the sum off by default; `dense_probe` as O-27 (2.4) |
| O-29 | `check_batch` is an optional member of the oracle protocol; the frontend calls it when the oracle has it, and checks nothing about the batch otherwise (2.4, 2.5, P14, P16) |
| O-30 | `held` is any object; in a compiled program, something jax can carry: arrays, tuples, `None` or a registered record (2.2, 2.4) |
| O-31 | The geometry has no consistency member; N21 withdrawn; a backend user calls `fv_frame_consistency_residual` (2.4, 3, P15) |
| O-32 | Settled by the amendments of 19:30 to rows 4.2 and 8.3 (7.1) |
| O-33 | The page takes the notation paragraph of `CLAUDE.md` and the probing map of `docs/entries_apply_probe.md`; other pages link to it (8) |
| O-34 | `UniformTuckerTensorTrain.unscale_coordinates`, over `ut3_unscale_coordinates` (2.4, 2.5, P3) |
| O-35 | One scale per coordinate: `scales` is `len=d`, `elm_shape=(Ni,)` (2.4, P3) |
| O-36 | An integer for `order`: "Agree. Order is a tuple only." `order=(3,)`, `order=()` for no jets; an integer raises (2026-11-17; head, 2.2, 4) |
| O-37 | `affine_jet` and `index_to_one_hot` exported from the top level `t3toolbox` as well as from `t3toolbox.fitting` (2026-11-17; 2.5, P4d) |
| O-38 | For two or more jet variables Part 1 runs the loop under both `over_orders` settings; the compiled program grows with the order there until Part 2 (2026-11-17; 1, 2.4, P4a) |
| O-39 | Withdrawn before it was put: rows 2.1 and 2.2 decide that the slots of an input jet are Taylor coefficients (2.2) |
| O-40 | `rescale_direction` takes one length (the default form) or one per jet variable (2026-11-17; 2.4, P4d) |
| O-41 | `index_to_one_hot` returns the default float dtype; no `dtype` argument (2026-11-17; 2.4, P4b) |
| O-42 | The display for two or more jet variables: one column per multi-index, in the order of unfolding into a matrix (row-major), no columns by total degree; and, his addition, an option that turns the table off and keeps the rest of the display: `error_table=True` on the frontend `newton_cg` beside `verbose`, on `make_newton_display` and on `format_newton_iter` (2026-11-17; 2.4, 2.5, P16, 8) |
| O-43 | "The ambient transposes stay without jets. However, there is no plan to add jetting to ambient transposes in the future." Not a deferred item (2026-11-17; 2.4) |
| O-44 | "Go with recommendation (b)": no default of `order` on `LeastSquaresOracle`; `()` stays the default on the sampling functions and methods, whose docstrings show the trap as a wrong-against-right example (2026-11-17; head, 2.2, 2.4, P13, P14) |
| O-45 | "Agree with proposal. Every slot along a jet axis is data." No mask on the jet axes, no checker; one sentence in the uniform equivalence contract (2026-11-17, 16:26; 2.2, 7.2, P11, 8) |
| O-46 | "accept order=() for entries": one class, `order` required for every kind, `()` for entries: `LeastSquaresOracle('entries', order=())`. No sentinel (2026-11-17; 2.2, 2.4, P14) |
| O-47 | The dual-jet and derivative functions accept `'entries'` at `()` and return the data unchanged (2026-11-17; 2.4, P4d) |

### 11.2 Open points

None. Every point is answered (11.1), and every name approved (sections 3, 11.3, 11.4).

### 11.3 Names added after the table was settled

Approved by the maintainer on 2026-11-16 at 20:59, with the uniform twin of O-34 named by the library's grammar.

| Name | What | Where | Slice |
| --- | --- | --- | --- |
| `over_orders='scan'`, `'loop'` | The values of `jet_contract`'s form over the orders; passed at every call, with no default | `backend/contractions.py` | P1 |
| `dense_apply(T, ww, pp=None, order=0)`, `dense_entries(T, index, pp=None, order=0)`; `dense_probe(T, ww, pp=None, order=0)` | The dense functions. From P4b: `dense_apply(T, ww, order=())`, `dense_entries(T, index)`, `dense_probe(T, ww, order=())` | `backend/apply.py`, `entries.py`, `probing.py` | P2, P4b |
| `check_sample(kind, order, sample, shape)`, `check_data(kind, order, sample, y)`, `check_weight(kind, order, weight, d)` | The checkers of a sample, its data and its weight; from P4d `order` is the jet shape and the sample input jets | `backend/jets.py` | P3, P4d |
| `ut3_unscale_coordinates(x, scales)`; `UniformTuckerTensorTrain.unscale_coordinates(scales)` | The uniform twin of `t3_unscale_coordinates` and its method (O-34) | `backend/data_preparation.py`; `uniform_tucker_tensor_train.py` | P3 |
| `fv_centers_consistency_residual`, `ufv_frame_consistency_residual`, `ufv_centers_consistency_residual` | The centers' residual and the uniform twins of both frame residuals | `backend/fv_operations.py`, `backend/ufv_operations.py` | P4 |
| `T3Frame.consistency_residual`, `UT3Frame.consistency_residual` | The cached residual the frontend checks | `frame_variations_format.py`, `uniform_frame_variations_format.py` | P4 |
| `HeldVariables.with_values(**values)` | A new record with the given slots filled | `backend/sampling_variables.py` | P5 |
| `check_coordinate_scales(scales, shape)` | The checker of the coordinate scales (their form only, never a value). Added in P3 by Claude, by the rule that the backend offers the checkers and after the pattern of the three above; the two `unscale_coordinates` methods and the frontend's own `fitting.rescale_coordinates` call it | `backend/data_preparation.py` | P3 |
| `LeastSquaresOracle.check_batch(batch, shape)` | The checker of a whole batch; an optional member of the protocol | `backend/fitting.py` | P14 |
| `pack_batch(kind, order, batch, N)` | The packing of a batch for the uniform layout (`order` the jet shape; input jets brought to one set of jet sizes with zero coefficients) | `backend/uniform_fitting.py` | P14 |
| `fitting.local_model(..., frame=None)` | The keyword through which a caller supplies a frame | `t3toolbox/fitting.py` | P16 |
| `docs/paper_to_code.md` | The page of the paper-to-code map | `docs/` | P19 |

### 11.4 Names of the generalized jet group

Approved by the maintainer on 2026-11-17 at 16:26 ("Agree with proposed names"; entries 117 and 120 had left "the
names of the two helpers" to the plan phase). N31 and N34 record names decided earlier. N42 was proposed at 16:21 and
is taken as approved with the others at 16:26; he confirmed it, on all three places, on 2026-11-19. Settled names that only
change their arguments (N16's keywords, `GroupLengths`) are in section 3's amendment.

| # | What | Name | Why | Alternatives considered |
| --- | --- | --- | --- | --- |
| N31 | The jet group of a contraction string | `J` | Decided (entry 120) | |
| N32 | The field of `GroupLengths` for it | `J` | The record's fields are the letters of the groups (`W`, `K`, `C`) | `m`, `n_jet` |
| N33 | The output's jet shape, an argument of `jet_contract` | `order` | One concept and one name across the library: the same tuple of maximum orders the engine, the oracle and the sampling functions take; the box is `F = tuple(n + 1 for n in order)` | `jet_sizes` in slot counts (the array's sizes, `n + 1`; but then the caller converts, off by one); `jet_shape` (the record's term for the same tuple; but every other function calls it `order`); `out_order` |
| N34 | The argument the user gives for the jets of a call | `order`, a tuple of maximum orders | Decided as the name (entry 118: "one ww plus order"); the tuple is entry 117's jet shape | |
| N35 | The helper that makes the input jet of a line (or of a point and `m` directions) | `affine_jet(ww, *directions)` | It is what it makes: the input jet of degree at most one in each variable, `w + s_1 p_1 + ... + s_m p_m`; `jet` as in `jet_to_dual_jet` | `line_jet(ww, pp)` (one direction only, simplest for the common case); `input_jet(ww, *pp)`; `lines_to_jets` (the `a_to_b` grammar, but it is not a conversion of one object) |
| N36 | The helper that turns an index into one-hot vectors | `index_to_one_hot(index, shape)` | The library's `a_to_b` grammar for a free converter (`t3_to_dense`, `jet_to_dual_jet`); `index` is the entries kind's word | `one_hot(index, shape)`; `index_to_vectors` |
| N37 | Their module | `backend/jets.py` | Both make the input jets a sampling function takes, beside the other helpers of jets | `backend/entries.py` for `index_to_one_hot` (it is about the entries kind; but it is used with apply) |
| N38 | The Part 2 marker for jets in several variables | `PART 2 (row 5.4, jets in several variables)` | The form of the existing marker, `PART 2 (row 5.5)` | |
| N39 | The full jet sizes in shape comments | `F` (`F = tuple(n + 1 for n in order)`); `J` for the jet axes of an array in general, `J_in_i` for an input's | `J` is the group's letter; `F` says "full". `T` is the dense tensor argument of the dense functions and leaves the comments | `J` alone, with the size in prose |
| N40 | The two terms of prose and docstrings | **jet shape** for the tuple of maximum orders (`order`), as the record uses it (entries 116, 117, 120); **jet sizes** for the sizes of an array's jet axes (`n + 1` for a full jet) | The record's term kept for the concept; a second, plainly different word for the array sizes, so that neither has two senses | "jet order" for the tuple (the first draft's term: it says what the entries are, maximum orders) with "jet shape" for the sizes; the record would then need amending |
| N41 | The keyword of `pack_vectors` and `pack_if_ragged` for the number of leading jet axes | `len_J=0` | The count is what `contract` takes for a group (`len_W=`, `len_C=`), and what the function needs: how many leading axes may differ across the modes; `0` keeps today's behaviour | `n_jet_axes=0`; `jet_axes=0` |
| N42 | The keyword that turns the display's error table off and keeps the rest of the display (O-42) | `error_table=True`, on the frontend `newton_cg` beside `verbose`, on `make_newton_display` and on `format_newton_iter` | It names what it turns off; `True` keeps today's display. Proposed 16:21, taken as approved with the other names at 16:26; confirmed at his review, 2026-11-19 | `show_table`, `table` |

---

## 12. How this plan is carried out

The maintainer set the process on 2026-11-16 at 20:51 (`TARGET_DECISIONS.md`, "How Part 1 is implemented").

- **Sign-off first.** The maintainer reviews this plan and signs it off. Nothing is implemented before.
- **One slice at a time.** Two new agents, an implementer and a reviewer, take the slices in the order of 5.2. The
  implementer writes the slice; the reviewer critiques it; they go back and forth until the reviewer signs off on
  the slice.
- **Then Claude.** Claude looks over the signed-off slice and commits it on `graded-messages`. The next slice starts
  only after that commit.
- **What a slice hands over:** the code; its tests; the results of its gate (section 5.1), run unpiped on a clean
  tree; and a short note of anything that departed from this plan, with the reason.
- **The agents:** both Fable 5.2 on medium effort (the maintainer, 20:51), for P0 to P4. **From P4a (the maintainer,
  2026-11-19):** both Opus 5.5 on high effort, and the same implementer and reviewer are kept from slice to slice, so
  that they carry their context; the next slice starts after Claude's commit, with the same pair. Required reading for
  both, and for Claude: a handful of methods of `TuckerTensorTrain` (`t3toolbox/tucker_tensor_train.py`) and the
  backend functions they call, for the style and the conventions of the library.
- **Planned pauses (the maintainer, 2026-11-19):** after P4d (the jet sub-arc P4a to P4d), after P11 (the engine
  sub-arc P5 to P11) and after P15 (before the switch). At each the work stops for a check-in. Between them, small or
  clear issues are decided by Claude and logged; big unforeseen ones, or ones where the right decision is uncertain,
  go to the maintainer at once. Think from first principles and use judgement; go slowly and do it right.
- **CI:** a draft pull request, #26 (`graded-messages` into `main`, opened 2026-11-19, not to be merged before Parts 1
  and 2 are done), so that both CI test legs, the examples and the docs build run on every push to the branch.
- **Memory.** The gate's steps run one after another, never two at once, and the implementer and the reviewer
  never run tests at the same time (the maintainer, 21:02: running sphinx and a parallel test run together has
  run a 13 GB machine out of memory). The workspace the agents run in has 2 cores and 7 GB. P0 measures the suite's
  peak memory and time with one worker and with two before a parallel run is relied on.
- **Issues.** Major issues go to the maintainer. A departure from a decided row, from a public name, or from the
  slice order is a major issue. Minor issues Claude decides, and records the decision in the slice's note.
