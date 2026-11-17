# Part 1 of the graded-messages refactor: implementation plan

**Status.** The implementation plan of Part 1, complete: every open point is answered and every name approved.
Agreed by the reviewer; awaiting the maintainer's sign-off. 2026-11-16. Written against the branch
`graded-messages` at `a9c31b2` (the library and `tests/` unchanged since `4d38df7`); every statement about the
library was checked against the code at that commit, and counts were recounted there.

**How to read it.** The rows of `TARGET_DECISIONS.md` are the authority, with the maintainer's answers to this
plan's open points (the note "The plan's open points, as the maintainer answers them" and the notes after it,
2026-11-16). A rule is given here with its row number (for example "2.9") and, where a reader needs it, one
sentence of reason. `SAMPLE_VARIABLES.md` is the reference for every variable: its name, definition, recurrence
and shape. The prototypes in `scripts/review/` establish the algebra; the library's contributor notes
(`docs/contributor/`) fix the form of the code.

---

## 1. Scope

**Part 1 delivers what is computed.** The sampling variables of `SAMPLE_VARIABLES.md` under their decided names;
one engine, `sampling_variables`, that computes any of them for the three kinds, in Taylor coefficients, at any
order, with order 0 as the same functions; `jet_contract` and a literal contraction string at every call site;
the weight and the data as arguments; frames with optional centers; arbitrary tails; both layouts. Above it: the
oracle and the batch, `LocalModel`, the optimizers' new seam, the display, and standalone functions that prepare
data and weights. Around it: the reference module of definitions and the dense functions it calls, the checkers,
the tests by definition and the identities, the documentation of the variables, and a dry port of T3Polynomial.

**Part 1 carries out every computation the way the library does it today.** Where a row is open it fixes the
following ("The two parts", and the answer to O-19):

| Open row | What Part 1 does |
| --- | --- |
| 5.3 sites | Sweeps go through `xscan`. Map-type operations go one site at a time: a loop over the sites on the ragged layout, `xmap` on the uniform one, for order 0 and for jets alike. There is no site group: a step sees one site's arrays on both layouts |
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
- *Time at order 0 on the uniform layout.* Today's plain kernels run the map-type operations at all sites at once
  (one contraction with a leading `d`: `compute_eta`, `assemble_z`); Part 1 maps them one site at a time, as
  today's jetted kernels do (`get_backend(True, True)` makes `xmap` `jax.lax.map`; `sampling_derivatives.py` calls
  it sixteen times). The plain ones may be slower than today until Part 2.
- *No benchmark in Part 1.* The old kernels are deleted on the tests alone (section 10). Part 2's performance work
  benchmarks the branch against the released 2026.5.0.

**The seams left for Part 2.** Part 2 adds; it does not change Part 1's interface.

| Seam | Where Part 2 adds |
| --- | --- |
| The plan | An optional `plan=None` keyword on `sampling_variables`, and wherever 7.3 puts it (the oracle, the model or the solver). No Part 1 signature has to change |
| The form over the orders (2.9) | `jet_contract` already takes `over_orders`; Part 1's engine sets it from the layout, Part 2 from the plan |
| The sites (5.3) | The map-type operations at all sites at once on the uniform layout, if Part 2 wants them: uniform code of its own, written with the library's `d` axis, not the shared step with an optional group (the maintainer, 2026-11-17: the site group `S` is removed). To be revisited in Part 2 (O-19) |
| What is held (7.2) | The least-squares oracle's fixed hold list becomes a field of the plan; `held` already flows through every call |
| Chunking (7.4) | Chunks are array shapes of the batch; the engine already accepts any number of sample axes `W` |
| Trial policies b and c (4.3) | An optimizer option; the geometry already returns centers with the frame |
| Several MC-SGD steps in one program (8.3) | The stochastic step is a module-level function of pytrees, and `take` is traceable with a static size |
| Contraction order (5.5) | Every term is a literal string at its call site, so a per-term order has one place to attach. Known cost until then: on numpy the default order inside one step of a product does the expensive contraction on every slot (`TARGET_DECISIONS.md`, "Carried to Part 2", item 1; marked in the code and by an expected-failure test) |
| The benchmark | Part 2 compares the branch with the released 2026.5.0; the dry port (section 9) lets T3Polynomial's benchmark run on the branch |

Part 1 is not released on its own. All work stays on `graded-messages` until Part 2 is done.

---

## 2. The end state

### 2.1 Modules

| Module | Holds at the end of Part 1 | Status |
| --- | --- | --- |
| `backend/contractions.py` | `contract` (unchanged); `jet_contract`; the record of axis counts `GroupLengths` | changed |
| `backend/sampling_variables.py` | The engine `sampling_variables`; `HeldVariables`; the declared graph, one tuple of records per kind; the shape table | new |
| `backend/sampling_steps.py` | The 22 step functions and the start values. Every name private, `__all__ = []` | new |
| `backend/sampling_definitions.py` | The reference module: one slow, dense function per variable and per output (9.7). Backend only | new |
| `backend/jets.py` | `jet_to_dual_jet` and `dual_jet_to_jet` (3.6); the four functions for derivative data (2.3); the checkers of a sample, its data and its weight: the two validators of `pp`, moved from `sampling_derivatives`, and three new ones | new |
| `backend/data_preparation.py` | The five preparation functions (6.7), and `ut3_unscale_coordinates`, the uniform twin of the last (O-34) | new |
| `backend/probing.py`, `apply.py`, `entries.py` | One backend function behind each frontend method, each calling the engine, with the merged signature (6.3, 6.6); the ambient transposes in the form of N16, their own code otherwise as it is; the dense functions beside their kinds: `dense_probe` in its new form, `dense_apply` and `dense_entries` new | rewritten |
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

**A sample.** `ww` (probe, apply) or `index` (entries) when `order == 0`; `(ww, pp)` or `(index, pp)` above.
`order` decides, never the type (6.2). `ww[i]` is `W + (N_i,)`; `index` is `(d,) + W`, integer. On the uniform
layout the vectors are packed once, when the batch is packed (`pack_batch`), and the engine is strict about it.

**The data `y`** is of the type of `z`, in forward order (2.4): probe `y[i]` is `T + W + (N_i,)`; apply and entries
`y` is `T + W`. It broadcasts over `C`.

**The weight** has exactly the shape `(d,) + T` for probe and `T` for apply and entries; no rule for a bare vector,
no broadcast of a mode dimension of 1 (2.5, 6.7). `None` means ones. The engine promises nothing about a weight of
another shape; the backend checker `check_weight` tests it, and the frontend calls it where a weight comes in
(O-18).

**A batch** is the data of a least-squares fit, arrays only (8.x note "The oracle is a description"):

```python
@dataclass(frozen=True, eq=False)
class LeastSquaresBatch:                 # a registered pytree: every field a leaf (None allowed)
    sample:          typ.Any             # ww | index | (ww, pp) | (index, pp), as the oracle's order says
    y:               typ.Any             # the data: probe len=d, elm_shape=T+W+(Ni,); apply/entries T+W
    weight:          typ.Optional[NDArray] = None   # probe (d,)+T; apply/entries T; None = ones
    minibatch_scale: typ.Optional[float]   = None   # n/batch_size on a drawn batch; None = 1 (a full batch)
    drawn_indices:   typ.Optional[NDArray] = None   # int, (batch_size,): which samples a draw took
```

A user's oracle with no data of its own takes `None` as its batch. The batch and the least-squares oracle are one
class each for both tiers, reachable from `t3toolbox.fitting`.

**An oracle** is a frozen description with no arrays and one operation (8.2, 8.3). Its fields are statics, so it
is value-hashed (`common.ValueHashedFields`) and rides as `aux_data`. The least-squares oracle has two fields,
`kind` and `order`.

**`held`** is what an oracle keeps at one frame; what goes into it is the oracle's business (8.3), and the
optimizers pass it back unread. The `Oracle` protocol types it as any object (O-8); in a compiled program it must
be something jax can carry: arrays, tuples, `None` or a registered record (O-30). The least-squares oracle's is a
`HeldVariables` (O-8):

- a frozen record with one slot per variable that some other variable reads, taken from the declared graph (a test
  checks that the two agree); each slot is `None` when not held, and carries its shape comment;
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
        sample:            typ.Any,                          # ww | index if order == 0, else (ww, pp) | (index, pp)
        want:              typ.Tuple[str, ...],              # variable names (the table); the return is aligned

        order:             int = 0,                          # highest Taylor order.  T=(order+1,) if order else ()
        y:                 typ.Optional[typ.Any] = None,     # the data; needed for r and what reads it
        weight:            typ.Optional[NDArray] = None,     # probe (d,)+T, apply/entries T; None = ones
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
   without `y` and without a held `r`); every held variable it is handed has the shape of the table, the order
   included (6.5). The ranks of the operands are checked in `jet_contract` (5.4). Nothing else is checked here: not
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
   are Python scalars.
5. **The sweeps and the maps.** A sweep along the chain goes through `xscan` with a module-level body
   (`docs/contributor/scan_body_principles.md`). A map-type operation (the `xi`, `eta`, `z` steps and the block
   readouts) goes one site at a time: a loop over the sites on the ragged layout, `xmap` on the uniform one, at
   every order (O-19). A product of two full jets is taken over the orders with `over_orders='loop'` on the ragged
   layout and `'scan'` on the uniform one (O-10). Boundary vectors are ones sized by the end bond of the cores
   passed; start values follow the grade (4.4). The engine squashes no tail.
6. **The seed.** `r = z - y` (the library's sign, 2.4); the seed `w_tilde = jet_to_dual_jet(r, weight)`: the
   weights, then the turn of the order leg. `JtJp` seeds from `Jp` the same way. A held `r` is used as it is.
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
    T:  int                         # 1 if order else 0; also the count of the affine group A
    W:  int                         # sample axes
    K:  int                         # direction-stack axes
    C:  int                         # base-point-stack axes

def jet_contract(
        subscripts:   str,           # a literal grouped string; T a full jet's order group, A an affine one's
        *operands:    NDArray,
        lens:         GroupLengths,  # every call passes the whole record, so a wrong rank raises
        over_orders:  str,           # 'scan' | 'loop': how two full jets are multiplied (2.9); no default
) -> NDArray:                        # contract(), with T and A multiplying as truncated polynomials
```

`jet_contract` follows the rule of 5.4: one operand with an order group and the output with it, the axis rides
(one `contract`); two operands with one and `T` in the output, the truncated product of 2.9 (two contractions for
an affine partner; for two full jets, the form `over_orders` names); no order group in the output, the top
coefficient of the product. At order 0 it is exactly one `contract` call. `contract` raises on a `len_<G>` whose
group is not in its string (`contractions._solve_group_lengths`), so `jet_contract` passes on only the lengths of
the groups that each string it hands to `contract` names; the strictness comes from those being all of them.

The two forms over the orders (2.9; O-10), both built and tested in Part 1:
- `'scan'`: one jet carried and shifted one slot per step, the other contracted against it, through the `xscan` of
  `get_backend(True, use_jax)`: a real `lax.scan` when the operands are jax, a Python loop on numpy. Its body is
  module-level, from a factory memoized on `(subscripts, lens)`.
- `'loop'`: a Python loop over the pairs of orders, with slices.

`jet_contract` knows nothing of the layout; it reads whether its operands are jax, as every function does. The
caller says which form: in Part 1 the engine passes `'scan'` on the uniform layout and `'loop'` on the ragged one,
which is how the library runs today. A product with an affine operand, order 0, and a block readout
whose jets are all contracted in one call are the same under both forms. A readout that takes a further full jet
one coefficient at a time (the four `H` strings with a full leaf) follows the form: under `'scan'` it is a real
scan with the carried shift, so the compiled program has one size at every order (2.9, the notes of 19:08 and
19:32); under `'loop'` it is slices over the triangle. The two forms agree to rounding, not bit for bit.

**The strings in Part 1.** The 41 strings of the seven families of `scripts/review/term_stacks_check.py`
(`STRINGS`). The 25 of the map-type families (`xi` 5, `z` 3, `eta` 4, `V` 5, `H` 8) are written there with the
site group `S`, which the maintainer removed on 2026-11-17; they run one site at a time, so they stand at their
call sites with `S` deleted, and the 16 sweep strings (`mu`, `nu`) as they are: 41 literal strings. `GroupLengths`
has four fields, `T`, `W`, `K`, `C`, and `S` is not a letter a string may name.

```python
def jet_to_dual_jet(
        jet:     typ.Any,                  # of the type of z: probe len=d, elm_shape=T+W+[K]+C+(Ni,); apply/entries T+W+[K]+C
        weight:  typ.Optional[NDArray],    # probe (d,)+T; apply/entries T; None = ones
        order:   int,
) -> typ.Any:                              # the dual jet, same shapes: omega^2 * jet, then the order leg turned

def dual_jet_to_jet(
        dual_jet:  typ.Any,                # same shapes as jet_to_dual_jet's result
        weight:    typ.Optional[NDArray],  # probe (d,)+T; apply/entries T; None = ones
        order:     int,
) -> typ.Any:                              # the jet: the order leg turned back, then divided by omega^2
                                           #   (where a weight is zero it divides by zero: no check, O-20)

def derivatives_to_taylor(
        jets:  typ.Any,                    # derivatives: probe len=d, elm_shape=T+W+(Ni,); apply/entries T+W
        order: int,
) -> typ.Any:                              # Taylor coefficients, same shapes: slot t divided by t!

def taylor_to_derivatives(jets, order)    # the reverse: slot t multiplied by t!; same arguments and shapes

def derivative_weight_to_taylor_weight(
        weight:  NDArray,                  # a weight on derivative residuals: probe (d,)+T, apply/entries T
        order:   int,
) -> NDArray:                              # same shape: slot t multiplied by t!

def prepare_derivative_data(
        kind:        str,                      # 'probe' | 'apply' | 'entries'
        order:       int,                      # >= 1
        y:           typ.Any,                  # derivatives, of the type of z
        weight:      typ.Optional[NDArray] = None,   # on derivative residuals; None = ones on the derivatives
        validation:  typ.Optional[typ.Any] = None,   # validation derivatives, or None
) -> typ.Tuple[
        typ.Any,                               # y in Taylor coefficients
        NDArray,                               # the Taylor weight: t! per slot when weight is None
        typ.Optional[typ.Any],                 # validation in Taylor coefficients, or None
]:
```

The checkers of a sample, its data and its weight. Each raises a `ValueError` naming what is wrong and returns
nothing; shapes are static, so each is safe under `jit`. They are checkers, not guarantees: no backend function
calls them, the frontend does (section 2.5), and `LeastSquaresOracle.check_batch` is built from them.

```python
def check_sample(
        kind:    str,                      # 'probe' | 'apply' | 'entries'
        order:   int,
        sample:  typ.Any,                  # ww | index | (ww, pp) | (index, pp), as order says
        shape:   typ.Sequence[int],        # the mode sizes N_i, len=d
) -> None:                                 # the form for the order; d vectors of W+(Ni,), or index (d,)+W; pp as ww

def check_data(kind: str, order: int, sample: typ.Any, y: typ.Any) -> None:
                                           # y is T+W+(Ni,) per mode (probe) or T+W (apply, entries)

def check_weight(kind: str, order: int, weight: typ.Optional[NDArray], d: int) -> None:
                                           # None, or exactly (d,)+T (probe) or T (apply, entries)
```

`check_perturbation_vectors(ww, pp)` and `check_perturbation_index(index, pp, shape=None)` move from
`sampling_derivatives` unchanged; `check_sample` calls them. The logic of `check_sample` and `check_data` is today's
frontend `fitting._check_ragged_sample`, which goes with the factories.

```python
def data_block_norms(
        kind:   str,
        order:  int,
        y:      typ.Any,                   # probe len=d, elm_shape=T+W+(Ni,); apply/entries T+W
) -> NDArray:                              # the shape of the weight: probe (d,)+T; apply/entries T

def equalizing_weight(
        block_norms:  NDArray,             # from data_block_norms
        floor:        float = 1e-6,        # relative to the largest block
) -> NDArray:                              # same shape: 1 / max(x, floor * max(x))

def rescale_direction(
        kind:    str,
        order:   int,
        sample:  typ.Any,                  # (ww, pp) | (index, pp)
        y:       typ.Any,                  # of the type of z
        length:  float,                    # pp *= length; slot t of y *= length**t   (loss-only)
) -> typ.Tuple[
        typ.Any,                           # the sample
        typ.Any,                           # y
]:

def rescale_coordinates(
        kind:    str,
        order:   int,
        sample:  typ.Any,                  # ww | (ww, pp); probe and apply only
        y:       typ.Any,                  # of the type of z
        scales:  typ.Sequence[NDArray],    # len=d, elm_shape=(Ni,): coordinate j of mode i times scales[i][j]
) -> typ.Tuple[
        typ.Any,                           # the sample
        typ.Any,                           # y (for probe, mode i's data divided by scales[i] too: 6.7 note)
]:

def t3_unscale_coordinates(
        x:       typ.Tuple,                # (tucker_cores, tt_cores) fitted in rescaled coordinates
        scales:  typ.Sequence[NDArray],    # as given to rescale_coordinates
) -> typ.Tuple:                            # the same tensor in the original coordinates

def ut3_unscale_coordinates(x, scales)    # the uniform twin: x the uniform cores' .data; the same scales
```

`scales` is one scale per coordinate, `len=d`, `elm_shape=(Ni,)` (O-35), as the prototype
`scripts/review/bad_scaling_example.py` has it: its problem has coordinate scales spread over four decades within a
mode, which one number per mode cannot undo.

The two estimates the prototype reads off the data (a length for the directions, a coordinate scale) are not
shipped: the example computes them inline (O-14: "compute inline, do not ship at this time").

The dense functions, beside their kinds, in the form of N16 (O-27, O-28). They are the dense ground truth of the
library: the reference module calls them for `z`, and the examples and the tests use them. Jets come from the
multilinear expansion over which modes take `p`, which is exact (the method of today's `dense_*_derivatives`,
without the `t!`). They keep today's `dense_probe` stacks (`W` on the sample, `C` on `T`). The frontend does not wrap
them: it has no class for a dense array.

```python
def dense_probe(
        T:      NDArray,                              # shape=C+(N0,...,N(d-1))
        ww:     typ.Sequence[NDArray],                # len=d, elm_shape=W+(Ni,)
        pp:     typ.Optional[typ.Sequence[NDArray]] = None,   # len=d, elm_shape=W+(Ni,); given iff order >= 1
        order:  int = 0,
) -> typ.Tuple[NDArray, ...]:                         # z. len=d, elm_shape=T+W+C+(Ni,)

def dense_apply(T, ww, pp=None, order=0) -> NDArray:      # T+W+C
def dense_entries(T, index, pp=None, order=0) -> NDArray: # index (d,)+W, integer; T+W+C
```

```python
def mu(kind: str, frame, sample, i: int, order: int = 0, *, variations=None, core_directions=None,
       w_tilde=None) -> NDArray:           # sampling_definitions: mu[i] by its definition, dense, no sweep
```

One such function per name of the table, all with the same arguments, each built as the dense contraction of
its subnetwork (`SAMPLE_VARIABLES.md`, the column "As a subnetwork"). Jets come from interpolation in `s`, which is
exact for a polynomial (9.3); `z` comes from the dense tensor through `dense_probe`, `dense_apply` or
`dense_entries`; `Jp` from the dense tangent tensor; `g`, `JtJp` and the tilde variables from the dense Jacobian;
`Kp` from the second derivative of the dense objective (jax where present, finite differences otherwise). Ragged
layout only; small problems only. The module is backend only.

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
    order:  int = 0
    # query(...): one call of sampling_variables. Its batch is a LeastSquaresBatch, its held a HeldVariables.
    # check_batch(batch, shape) -> None: the form of the sample, its sizes against shape, y, the weight (O-18)
    # n_measurements(batch) -> int;  take(batch, indices) -> LeastSquaresBatch (traceable, static len(indices))
```

The least-squares oracle's `query` asks the engine for `want` plus the names of its hold list that the declared
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
        order:  int,
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
`core_directions`); then the keywords `pp=None, order=0`; then `sum_over_samples=True` where there is a sum. One
name for one thing, on both layouts: `x` for a Tucker tensor train's cores (today `x`, `data` or `core_pair`),
`frame` (today `frame` or `frame_data`), `variations` (today `variation` or `variations_data`), `ww` (today `ww` or
`vecs`) or `index`, `residual` (today `ztildes` or `c`), `core_directions`. The sum flag is `sum_over_samples=True`
in place of `sum_over_probes=False` (3.7 note; N3, N25).

| Function, today | Becomes |
| --- | --- |
| `t3_probe(ww, x)`, `ut3_probe(ww, data)` | `t3_probe(x, ww, pp=None, order=0)`, `ut3_probe` likewise |
| `t3_apply(x, vecs)`, `ut3_apply(data, vecs)` | `t3_apply(x, ww, pp=None, order=0)`, `ut3_apply` likewise |
| `t3_entries(x, index)`, `ut3_entries(data, index)` | `t3_entries(x, index, pp=None, order=0)`, `ut3_entries` likewise |
| `tv_probe(ww, variation, frame)`, `utv_probe(ww, frame_data, variations_data)` | `tv_probe(frame, variations, ww, pp=None, order=0)`, `utv_probe` likewise; apply and entries (`index`) likewise |
| `tv_probe_transpose(ztildes, ww, frame, sum_over_probes=False)`, `utv_probe_transpose(ztildes, ww, frame_data, ...)` | `tv_probe_transpose(frame, ww, residual, pp=None, order=0, sum_over_samples=True)`, `utv_probe_transpose` likewise; apply (`c`) and entries (`c`, `index`) likewise |
| `t3_probe_corewise_transpose(ztildes, ww, core_pair, sum_over_probes=False)`, `ut3_probe_corewise_transpose(ztildes, ww, data, ...)` | `t3_probe_corewise_transpose(x, ww, residual, pp=None, order=0, sum_over_samples=True)`, the twin likewise; apply and entries likewise |
| `t3_probe_corewise_hessian(ztildes, ww, core_pair, core_directions)`, `ut3_probe_corewise_hessian(ztildes, ww, data, core_directions)` | `t3_probe_corewise_hessian(x, ww, residual, core_directions, pp=None, order=0)`, the twin likewise; apply and entries likewise |

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
at, and no `pp` or `order`. Their default differs from the other transposes' on purpose: for them the sum folds `W`
into the CP rank. `dense_probe` takes the rule too, and `pp` and `order` (above).

```python
def t3_probe(
        x:      typ.Tuple,                            # (tucker_cores, tt_cores), elm_shapes C+(nDi,Ni), C+(ri,nDi,r(i+1))
        ww:     typ.Sequence[NDArray],                # len=d, elm_shape=W+(Ni,)
        pp:     typ.Optional[typ.Sequence[NDArray]] = None,   # len=d, elm_shape=W+(Ni,); given iff order >= 1
        order:  int = 0,                              # highest Taylor order.  T=(order+1,) if order else ()
) -> typ.Tuple[NDArray, ...]:                         # z. len=d, elm_shape=T+W+C+(Ni,)

def tv_probe(
        frame:       typ.Tuple,                       # (D, U, L, R)
        variations:  typ.Tuple,                       # (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        ww:          typ.Sequence[NDArray],           # len=d, elm_shape=W+(Ni,)
        pp:          typ.Optional[typ.Sequence[NDArray]] = None,
        order:       int = 0,
) -> typ.Tuple[NDArray, ...]:                         # Jp. len=d, elm_shape=T+W+K+C+(Ni,)

def tv_probe_transpose(
        frame:             typ.Tuple,                 # (D, U, L, R)
        ww:                typ.Sequence[NDArray],     # len=d, elm_shape=W+(Ni,)
        residual:          typ.Sequence[NDArray],     # of the type of z: len=d, elm_shape=T+W+[K]+C+(Ni,)
        pp:                typ.Optional[typ.Sequence[NDArray]] = None,
        order:             int = 0,
        sum_over_samples:  bool = True,               # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                      # V blocks. len=d, elm_shape=[W]+[K]+C+(nUi,Ni)
        typ.Tuple[NDArray, ...],                      # H blocks. len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))
]:

def t3_probe_corewise_hessian(
        x:                typ.Tuple,                  # (B, G): the point whose cores the directions vary
        ww:               typ.Sequence[NDArray],      # len=d, elm_shape=W+(Ni,)
        residual:         typ.Sequence[NDArray],      # c, of the type of z: len=d, elm_shape=T+W+C+(Ni,); no weight
        core_directions:  typ.Tuple,                  # (B_dot, G_dot), shapes of B, G with K+C
        pp:               typ.Optional[typ.Sequence[NDArray]] = None,
        order:            int = 0,
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                      # Tucker block. len=d, elm_shape=K+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                      # TT block.     len=d, elm_shape=K+C+(ri,nDi,r(i+1))
]:
```

`t3_probe_corewise_transpose(x, ww, residual, pp=None, order=0, sum_over_samples=True)` is the transpose on a
corewise call and follows the layout above. The uniform twins (`ut3_probe`, `utv_probe`, `utv_probe_transpose`,
`ut3_probe_corewise_transpose`, `ut3_probe_corewise_hessian`, and apply and entries) take the same lists, with `x`
the uniform cores' `.data` and `frame` the uniform frame's; they check packing and call the engine, which masks
once on entry and re-masks core-shaped outputs on exit (O-3). Every positional call of a function
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
`make_newton_display(oracle, batch, val_batch=None, ...)`; `relative_errors` and the table take arrays of the shape
of the weight. In `uniform_fitting.py`: `pack_sample(kind, order, sample, N)`, `pack_data(kind, order, y, N)` and `pack_batch`
(above). The backend `LocalModel`, `local_model` and the optimizers check nothing numerical (O-6), as
`backend/optimizers.py` already says of itself.

**Removed.** Counted from the `__all__` lists of the six sampling modules, 166 public names today:

| Module | Today | Changed (call the engine) | Kept as they are | Removed |
| --- | --- | --- | --- | --- |
| `probing` | 34 | 5 | 2, in the form of N16, with their own code: `t3_probe_ambient_transpose`; `dense_probe`, which also gains `pp` and `order` | 27: every `compute_*`, `assemble_*`, `tv_precompute_probe_frame_sweep`, the three `*_from_sweep` |
| `apply` | 10 | 5 | 1 (`t3_apply_ambient_transpose`, its arguments by N16) | 4: the sweep and `*_from_sweep` functions |
| `entries` | 10 | 5 | 1 (`t3_entries_ambient_transpose`, its arguments by N16) | 4: likewise |
| `ut3_sampling` | 25 | 9 | 1 (`ut3_full_sum`) | 15: every `*_derivatives*` and `*_from_sweep` |
| `utv_sampling` | 30 | 6 | 0 | 24: the sweeps, `*_from_sweep`, every `*_derivatives*` |
| `sampling_derivatives` | 57 | 0 | 2 moved to `jets` (`check_perturbation_vectors`, `check_perturbation_index`) | 55, the module with them (the chunk estimators and `dense_*_derivatives` included; the dense jets are `dense_probe`, `dense_apply`, `dense_entries` with `pp` and `order`) |
| **Total** | 166 | 30 | 7 | 129 |

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
| `TuckerTensorTrain` | 15 | 9: `entries`, `apply`, `probe` and their `*_corewise_transpose` with `(…, pp=None, order=0)`; the three ambient transposes with their arguments by N16 | the 3 `*_derivatives`, the 3 `*_corewise_derivatives_transpose` |
| `T3Tangent` | 12 | 6: `probe`, `apply`, `entries` and their `*_transpose`, merged | 6 |
| `UniformTuckerTensorTrain` | 12 | 6 | 6 |
| `UT3Tangent` | 12 | 6 | 6 |

Each merged method returns `T + W + [K] + C + ...` with `T` outermost, Taylor coefficients (6.6). The methods follow
the rule of N16 (section 2.4) with `self` as the object: the sample, then what is applied, then `pp=None, order=0`
(6.6: "the transposes take the same `pp` and `order`"), then `sum_over_samples=True` where there is a sum. The
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
| `TuckerTensorTrain`, `UniformTuckerTensorTrain` | `probe(ww)` | `probe(ww, pp=None, order=0)` |
| | `apply(vecs)` | `apply(ww, pp=None, order=0)` |
| | `entries(index)` | `entries(index, pp=None, order=0)` |
| | `probe_corewise_transpose(ztildes, ww, sum_over_probes=False)` | `probe_corewise_transpose(ww, residual, pp=None, order=0, sum_over_samples=True)` |
| | `apply_corewise_transpose(c, ww, sum_over_probes=False)` | `apply_corewise_transpose(ww, residual, pp=None, order=0, sum_over_samples=True)` |
| | `entries_corewise_transpose(c, index, sum_over_probes=False)` | `entries_corewise_transpose(index, residual, pp=None, order=0, sum_over_samples=True)` |
| `TuckerTensorTrain` only, static | `probe_ambient_transpose(ztildes, ww, sum_over_probes=False)`, `apply_ambient_transpose(c, ww, ...)`, `entries_ambient_transpose(c, index, shape, ...)` | `probe_ambient_transpose(ww, residual, sum_over_samples=False)`, `apply_ambient_transpose(ww, residual, sum_over_samples=False)`, `entries_ambient_transpose(index, shape, residual, sum_over_samples=False)`; still static, the default still off |
| `T3Tangent`, `UT3Tangent` | `probe(ww)`, `apply(ww)`, `entries(index)` | the same with `pp=None, order=0` |
| static | `probe_transpose(ztildes, ww, frame, sum_over_probes=False)` | `probe_transpose(frame, ww, residual, pp=None, order=0, sum_over_samples=True)` |
| static | `apply_transpose(c, ww, frame, ...)`, `entries_transpose(c, index, frame, ...)` | `apply_transpose(frame, ww, residual, ...)`, `entries_transpose(frame, index, residual, ...)`, the same keywords |

The sketches, one instance method, one static transpose, one corewise transpose; the others follow the same rule:

```python
    def probe(                                        # TuckerTensorTrain (instance method)
            self,
            ww:     typ.Sequence[NDArray],                    # len=d, elm_shape=W+(Ni,)
            pp:     typ.Optional[typ.Sequence[NDArray]] = None,   # len=d, elm_shape=W+(Ni,); given iff order >= 1
            order:  int = 0,                                  # highest Taylor order.  T=(order+1,) if order else ()
    ) -> typ.Tuple[NDArray, ...]:                             # len=d, elm_shape=T+W+C+(Ni,)

    @staticmethod
    def probe_transpose(                              # T3Tangent (static method, a constructor)
            frame:             'bvf.T3Frame',
            ww:                typ.Sequence[NDArray],         # len=d, elm_shape=W+(Ni,)
            residual:          typ.Sequence[NDArray],         # len=d, elm_shape=T+W+[K]+C+(Ni,)
            pp:                typ.Optional[typ.Sequence[NDArray]] = None,
            order:             int = 0,
            sum_over_samples:  bool = True,
    ) -> 'T3Tangent':                                         # at frame; tangent stack [W]+[K], frame stack C

    def probe_corewise_transpose(                     # TuckerTensorTrain (instance method)
            self,
            ww:                typ.Sequence[NDArray],         # len=d, elm_shape=W+(Ni,)
            residual:          typ.Sequence[NDArray],         # len=d, elm_shape=T+W+C+(Ni,)
            pp:                typ.Optional[typ.Sequence[NDArray]] = None,
            order:             int = 0,
            sum_over_samples:  bool = True,
    ) -> typ.Tuple[
            typ.Tuple[NDArray, ...],                          # Tucker-core gradients, shapes of self.tucker_cores (with [W])
            typ.Tuple[NDArray, ...],                          # TT-core gradients, shapes of self.tt_cores (with [W])
    ]:
```

Every call of a transpose or a corewise transpose moves its arguments; a call of `apply` with `vecs=` breaks
(none was found). P13 lists them.

The merged methods check what they check today, structurally, through the backend checkers: the sample's vectors
against `pp` (`check_perturbation_vectors`, `check_perturbation_index`, from `jets`).

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
        oracle,                                   # e.g. LeastSquaresOracle('probe', order=3)
        batch,                                    # e.g. LeastSquaresBatch((ww, pp), y, weight)
        x0:           Point,                      # ragged or uniform: picks the representation
        regularizer:  typ.Any = None,             # e.g. IdentityRegularizer(lam)
        verbose:      bool = False,
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
        oracle,                                   # e.g. LeastSquaresOracle('probe', order=3)
        batch,                                    # e.g. LeastSquaresBatch((ww, pp), y, weight)
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
  `data_block_norms`, `equalizing_weight`, `rescale_direction`, `rescale_coordinates`, `t3_unscale_coordinates`;
- its own `dual_jet_to_jet`, a small function: in safe mode it raises where a weight is zero, then calls the
  backend one, which divides (O-20).

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
| N16 | the form of the merged functions, backend and frontend | One rule: the object the function is evaluated at first (`x`, or `frame` then `variations` for a tangent), then the sample (`ww` or `index`), then what is applied (`residual`, then `core_directions`); then the keywords `pp=None, order=0`; then `sum_over_samples=True` where there is a sum. One name for one thing on both layouts: `x`, `frame`, `variations`, `ww` or `index`, `residual`, `core_directions`. The frontend's instance methods follow the rule with `self` as the object; the tangent transposes stay static methods with the frame first. The ambient transposes take the rule's order and names and keep the sum flag off by default (decided 18:35). Sections 2.4 and 2.5 give every form | Decided by the maintainer, 2026-11-16 (18:13 and 18:18), `TARGET_DECISIONS.md`, the note "The names of Part 1, and the form of the merged sampling functions": the order the engine and the oracle already have |
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
  it, the record `GroupLengths` passed whole and the form over the orders named (5.4, 2.9). The 41 strings of the
  seven families are listed in `scripts/review/term_stacks_check.py` (`STRINGS`); none carries a site
  group (section 2.4).
- **Dispatch** at the lowest level: `use_jax = tree_contains_jax(...)` over all inputs, `is_uniform` from the
  structure, `xnp, xmap, xscan = get_backend(is_uniform, use_jax)`. No `use_jax` argument. Uniform masks are host
  numpy and use `np`, never `xnp` (`CLAUDE.md`).
- **Scan and map bodies** are module-level and close over nothing; a static no operand carries reaches a body
  through a factory memoized on hashable scalars (`GroupLengths`, the kind, the grade), never arrays
  (`docs/contributor/scan_body_principles.md`, rule 4).
- **Masks**: the uniform inputs are masked once on the engine's entry, each with the frame's own masks; steps never
  see a mask; core-shaped outputs are re-masked on its exit (`docs/contributor/uniform_internals.md`).
- **Tails**: the boundary vector is ones sized by the end bond of the cores passed; the engine squashes no tail;
  every function this arc writes or touches supports arbitrary tails (4.4). Frame construction
  (`t3_orthogonal_representations`, with its default `squash_tails=True`, called by the geometry's `frame`) keeps
  its default: 4.4 leaves that default unspoken, and 1.1 puts frame construction off limits.
- **Signatures** in the house style (`docs/contributor/signature_style.md`); body locals carry their axis layout in
  the suffix (`mu_TWCa`).
- **The razor and the two users** (section 7.5): anything a raw-`.data` user would get wrong is in the backend; the
  frontend validates its objects, calls the backend's checkers, and delegates.
- **Checkers, not guarantees** (section 7.1): no backend module imports `safety`; a backend function checks only what
  it needs to run; the frontend enforces, through backend checkers.
- **Doctests** teach, are seeded, print structure, and are run and pasted (`docs/contributor/doctest_style.md`).
- **No auto-formatter.**

---

## 5. The work in slices

### 5.1 How the suite stays green

- **Additive first.** Slices P1 to P11 add new modules beside the old ones. Two old things change in them, each with
  its callers in the same slice: `dense_probe` takes the order of N16 and gains `pp` and `order` (P2), and P4's
  consistency residual goes through the cores, keeping its signature (`fv_frame_consistency_residual`,
  `T3Frame.is_consistent`). So the suite stays green by construction.
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

The names are settled: the table of section 3, and the names added after it (section 11.3). Nothing is open, so no
slice waits on an answer. The order is the order of work; each slice starts when the one before it is committed
(section 12).

| Slice | Content | Size |
| --- | --- | --- |
| P0 | Baseline: the folder's check suites on the branch head; the inventory | none in the library |
| P1 | `jet_contract` with both forms over the orders, and `GroupLengths` | small |
| P2 | The reference module `sampling_definitions`; `dense_probe` in its new form, `dense_apply`, `dense_entries` | medium |
| P3 | `jets` (with the checkers) and `data_preparation`, with the uniform `ut3_unscale_coordinates`; their frontend reach | small |
| P4 | The checkers of a frame, without dense tensors; the frames' cached `consistency_residual` | small |
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

**P1. `jet_contract`.** In `backend/contractions.py`: `GroupLengths` and `jet_contract` (section 2.4), with both
forms over the orders. `'scan'` is the carried shift through `xscan`, with a module-level body from a factory
memoized on `(subscripts, lens)`; `'loop'` is a Python loop over the pairs of orders. The prototype's
`jet_contract` (`term_stacks_check.py`) is the algebra and `scripts/review/order_scan_check.py` the check of the two
forms; their Python-loop scan is not the code. It passes `contract` only the lengths of the groups each derived
string names.
- Tests (`tests/test_jet_contract.py`): every one of the 41 strings, as Part 1 writes them (section 2.4),
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

**P2. The reference module and the dense functions.** `backend/sampling_definitions.py`: one dense function per
name of section 2.3 and per output, for the three kinds, ragged, from the column "As a subnetwork" of
`SAMPLE_VARIABLES.md`; jets by interpolation in `s`; `z` through the dense functions; `Kp` by jax where present,
finite differences otherwise. Tilde and tilde-dot variables are returned turned, as the engine stores them (2.4).
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

**P3. Jets, the checkers and data preparation.** `backend/jets.py` (N4 to N8; the two validators of `pp`, copied,
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

**P5. Engine A.** `backend/sampling_steps.py` with the steps of the plain grade; `backend/sampling_variables.py` with
the probe kind's declared graph, `HeldVariables` and `sampling_variables` for `w`, `xi`, `mu`, `nu`, `eta`, `nu_L`,
`eta_D`, `z` (through `D`, and with `Vc` read off the frame), `r`, `f`, `r_block_sumsq`. Ragged layout, one sample
axis, no `K` or `C` yet, numpy and jax. What the engine needs to run (section 2.3 step 1), and nothing more.
- Tests by definition (`tests/test_sampling_variables.py`): every name against `sampling_definitions`, at orders 0,
  1, 3, with a random weight.
- Identities (`tests/test_sampling_identities.py`): order 0 equals slot 0 of a higher order; a held variable gives
  the result of a recomputed one; `z` through `D` equals `z` with `Vc`.
- Structural tests (`tests/test_sampling_graph.py`): the docstring table equals the declared graph row for row,
  exactly; the slots of `HeldVariables` are exactly the variables some other variable reads; asking for one variable
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
- Tests by definition for the new names. Identities, each with a random weight: `g` is the derivative of `f` by
  finite differences; the short form of `eta_tilde` equals a slice of the full one.
- The bridge (`tests/bridge/test_old_kernels_bridge.py`, temporary, section 5.1): `z` and `g` against
  `sampling_derivatives.t3_probe_derivatives`, the old `compute_*` kernels and `ProbeDerivativesKind.transpose`;
  the dense functions of P2 against `dense_*_derivatives` divided by `t!`.
- The generic twin (`tests/_generic_twin.py`): `scripts/graded_messages_prototype.py` brought to the decided names,
  as a second check of the written-out terms (9.3).
- `tests/test_dispatch.py`: jit `sampling_variables` for `('f', 'g')`.
- Gate: the full gate.

**P7. Engine C.** The dot steps of the frame form; `xi_U_dot`, `mu_R_dot`, `nu_L_dot`, `eta_D_dot`, `Jp`,
`Jp_norm_sq`, `JtJp` (the adjoint sweep seeded from `Jp`, its variables unnamed, 3.3).
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
- Gate: the full gate.

**P9. Engine E.** Apply and entries: their graphs (the forward needs only `xi` and `mu`; the adjoint only `nu_tilde`
and `eta_tilde`; the README's "Things learned"), the entries gather, the scalar output and the weight of shape `T`.
- Tests by definition for both kinds; the identities of P5 to P8 for both; entries against apply at one-hot
  vectors to tolerance (not bitwise; 0.8). Bridge against `t3_apply_derivatives`, `t3_entries_derivatives` and
  their kinds.
- Gate: the full gate.

**P10. Engine F.** Any number of axes in `W`; the direction stack `K` on `variations` and `core_directions`; the
base-point stack `C` on every core; `sum_over_samples=False` with the result's stack in the order `W + K + C`; `f`
keeping `C`, `Jp_norm_sq` keeping `K + C`; the dtype rule over the inputs the call reads (O-11).
- Tests: `W` empty, `(3,)`, `(2, 2)`, crossed with the library's stack matrix (`_CONFIGS`,
  `docs/contributor/testing_strategy.md` "The stack matrix": no stack, `C`, `K`, `K + C`, multi-axis `C = (2, 3)`
  and `K = (2, 3)`), against the reference applied element by element; the dtype rule for every output on the four
  backends (float32 in, float32 out; float32 mixed with float64 gives float32; jax mixed with numpy gives jax); an
  input nothing wanted reads does not change the dtype (a float64 `y` on a call wanting only `z` in float32 gives
  float32); an integer `index` does not count.
- Gate: the full gate.

**P11. Engine G: the uniform layout, and sharding over `W`.** The uniform frame and point structures; mask once on
entry, each input with the frame's own masks (section 2.3 step 2); re-mask on exit (O-3); the map-type operations
one site at a time through `xmap` (O-19); the sweeps through `xscan`, a real `jax.lax.scan` with jax; the products
of two full jets with `over_orders='scan'`, nested inside.
- Tests: equivalence with ragged on real parts (`to_uniform -> op -> to_ragged == op_ragged`) for every name, over
  the full `_CONFIGS` matrix of the uniform tests: forced-larger padding and a stack whose ranks vary across `C`
  included; garbage-padded inputs give the same real parts; arbitrary tails on both layouts; exact masks of every
  core-shaped output, derived non-circularly from the frame's masks.
- `Kp` on a frame with TT slack (its masks a strict subset of its point's): `core_directions` and the blocks take
  the frame's `D` and `L` masks; garbage in the padding of `core_directions` leaves the real blocks unchanged; and a
  mutation check (the mask on `core_directions` removed) turns this test red
  (`docs/contributor/testing_strategy.md`, the two-hole note).
- Sharding (`tests/test_contractions_sharding.py`, a new class beside `TestJetKernelsShardOverW`): `g` and `Kp` from
  `sampling_variables` at order 0 and order 3, ragged and uniform, under `shard_map` with `W` sharded and the cores
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
- Tests added: the uniform twins' exact output masks and garbage-padded inputs, through their public signatures;
  the default of every transpose sums the samples; the corewise transpose equals the gradient, and the corewise
  Hessian applied to a direction equals the second derivative along it, of `phi(x) = <c, t3_probe(x, ww, pp,
  order)>` (and of apply and entries) by automatic differentiation in jax, at orders 0 and 2, skipped where jax is
  absent (O-22); the ambient transposes' default does not sum.
- `tests/test_dispatch.py`: the rows for these names, by the new signatures.
- Gate: the full gate. From here `probing.compute_*` and the plain `assemble_*` are called only by the old kinds,
  their tests and the bridge.

**P13. The frontend sampling methods.** The four classes get the merged methods of
section 2.5; the 24 methods go; the frontend transposes and corewise transposes take their arguments in the order of
N16 (the residual after the sample, the tangent transposes' frame first), `residual` in place of `ztildes` or `c`,
and `sum_over_samples=True` in place of `sum_over_probes=False`; `TuckerTensorTrain.apply` and its uniform twin take
`ww` in place of `vecs`; the validators come from `jets`. Where a caller still fits with the old call form of the frontend optimizers (the tests until P16, the examples and
pages until P17), it makes the old derivative data from
the merged method as `taylor_to_derivatives(x.probe(ww, pp, order), order)`.

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
  `dense_probe` with `pp` and `order` (P2; O-27), which take the stack `W` themselves, so the loops over the samples
  go too, and P20 can delete the module. Where an example still fits derivative data with the old frontend
  optimizers (until P17), it takes `taylor_to_derivatives(dense_apply(T, ww, pp, order), order)`. (Seven
  examples define their own `dense_apply(A, ww)` or `dense_probe(A, ww)` helpers, local and of no removed name;
  they may stay.)
  `tests/test_probe_derivatives.py` (nine calls of the three `dense_*_derivatives`) is folded into the frontend
  sampling tests here, on the dense functions.
- Pages that name the removed methods in prose only (`docs/entries_apply_probe.md`, `docs/naming_conventions.md`,
  `docs/transposes.md`) are rewritten in P19.
- Tests added: the doctests on the merged methods (`order=0` returns today's shapes; `order=2` adds `T` outermost).
- Gate: the full gate.

**P14. The oracle and the batch** (additive). `LeastSquaresOracle` and `LeastSquaresBatch` in `backend/fitting.py`
beside the old kinds; the `Oracle` protocol as documentation in `backend/optimizers.py`, with `optional_outputs`
(O-7), `check_batch` as an optional member (O-29), and `held` typed as any object that, in a compiled program, jax
can carry (O-8, O-30); `query` passing `held` (a `HeldVariables`) to the engine;
`n_measurements`, `take`, `check_batch`, `y_block_sumsq`, `f_resolution`, the minibatch scale on `f`, `g`, `JtJp`,
`Jp_norm_sq`, `Kp` and `f_resolution` only (O-21); `pack_batch` in `backend/uniform_fitting.py`. Both classes
reachable from `t3toolbox.fitting`.
- Tests: each output of `query` against the engine and against the reference; `held` holds exactly the names of the
  hold list that the call computed (none computed only to be held); `take` under `jax.jit` with a static size;
  `minibatch_scale` makes the drawn `f` an unbiased estimate of the full `f`, tested exactly on a small `n` by
  enumerating every subset of the batch size; on a drawn batch, `r`, `Jp`, `r_block_sumsq` and `y_block_sumsq` are
  unscaled and the identities `f = 1/2 <r, r>`, `<Jp, r> = <p, g>`, `Jp_norm_sq = <Jp, Jp>` hold in the batch's
  inner product; `check_batch` passes a good batch and raises on each wrong part; `pack_batch` then the uniform
  query equals the ragged query on real parts; the pytree round trip of the batch; the oracle's hash and equality
  by value; bridge against the old kinds through the conversions.
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
   iteration in the full shape of the weight, nothing reduced over modes or orders (8.4). The timing laps: `oracle`
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
   broadcast to the exact shape; derivative data, validation data and weight go through `prepare_derivative_data`;
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
  the fitting rows of `tests/test_dispatch.py`. Every file rewritten here also drops its uses of the old kinds, so
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
  `fitting.local_model` with a `LeastSquaresOracle` and a `LeastSquaresBatch`.
- Pages that name the removed objects in prose only (`docs/api_reference.rst`, `docs/user_guide.rst`,
  `docs/glossary.md`, `docs/naming_conventions.md`, and the contributor pages of section 8) are rewritten in P19
  (the Sphinx configuration is not nitpicky, so a stale `:py:` reference does not fail `sphinx -W`; a stale `:doc:`
  link does).
- Tests added: optimizers judged by outcomes on problems with known answers and by the model's consistency
  (`p` paired with `hvp(p)` equals `gn_quadratic(p)` in Gauss-Newton mode), not by old iteration counts (9.3); a
  model rebuilt each step is one cache key; a new weight or minibatch does not recompile; `want` is a literal; the
  stored record's shapes at orders 0 and 2 for each kind, with and without a validation batch; the floor `eps |f|`
  on a user oracle without `f_resolution`; a user oracle without a draw raises in `mc_sgd`; a supplied `start_frame`
  with a perturbed store raises in safe mode and runs in `safety.unsafe()`; a library-built frame is not checked; a
  bad batch raises in the frontend before any work; the backend optimizers run on the same bad inputs without a
  check. The translation (temporary, deleted in P17): each old call form gives the result of the new one.
- The frontend-only fit (the answers of 20:15): a test that fits a tensor from derivative data importing nothing from
  `t3toolbox.backend` (`t3toolbox.fitting` for the oracle, the batch and `prepare_derivative_data`;
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
  `fit_shared_factors_jetted_probes.py`; the examples that fit derivative data use `prepare_derivative_data`, and
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

`Problem`, the factories, the old call form and the frontend methods are removed when they switch because they are
interfaces, not kernels.

---

## 6. The tests

| Layer | What | Where | When |
| --- | --- | --- | --- |
| Reference | `sampling_definitions` pinned against the dense tensor, the dense tangent tensor, the dense Jacobian and the Hessian | `tests/test_sampling_definitions.py` | P2 |
| By definition (9.3) | every name, every kind, orders 0, 1, 3, a random weight, against the reference | `tests/test_sampling_variables.py` | P5 to P11 |
| Identities (9.3) | the adjoint identity of `Jp` and `g`; `p` with `JtJp` equals `Jp_norm_sq`; `g` is the derivative of `f`; held equals recomputed; independence of the layout and the backend; order 0 equals slot 0; `Vc` route equals `D` route; short forms are slices | `tests/test_sampling_identities.py` | P5 to P11 |
| Generic twin | the written-out terms against terms derived mechanically | `tests/_generic_twin.py`, used by the identity tests | P6 |
| Structural | the table equals the declared graph; the slots of `HeldVariables` equal the variables read by others; one wanted variable computes only its reads; unknown names and missing arguments raise; held shapes; the dtype rule; uniform output masks | `tests/test_sampling_graph.py`, the uniform tests | P5, P10, P11, P12 |
| Checkers | each checker passes a good input and catches each wrong one; no backend function calls one; the frontend enforces through them, in safe mode where numerical | `tests/test_jets.py`, the frame tests, the fitting tests | P3, P4, P14, P16 |
| Reference | the dense functions against interpolation and today's values; the corewise transpose and Hessian against automatic differentiation of `<c, z(x)>` | `tests/test_sampling_definitions.py`, the curvature tests | P2, P12 |
| Uniform | equivalence with ragged on real parts over the full `_CONFIGS` matrix; exact output masks; garbage-padded inputs; tails; the two-hole masks on a TT-slack frame, with a mutation check | `tests/test_sampling_variables.py` (uniform cases), `tests/test_uniform_probing.py` | P11, P12 |
| Sharding | the AST sweep over every literal string; `jet_contract`'s product strings; `g` and `Kp` under `shard_map` with `W` sharded, zero all-gathers, equal to unsharded | `tests/test_contractions_sharding.py` | P1, P11 |
| Dispatch | jit each new public function; numpy in numpy out, jax in jax out; eager calls compile once | `tests/test_dispatch.py` | every slice |
| Doctests | every new public function; the page; the 2.3 doctest | modules, `docs/` | P2, P3, P5, P19 |
| Fitting | outcomes on known answers; model consistency; compile keys; the stored record's shapes; the `eps |f|` floor; a draw-less oracle in MC-SGD; the minibatch scale's identities; a frontend-only fit from derivative data | the rewritten fitting tests | P14, P16 |
| Bridge (temporary) | new outputs against old code that is never rerouted (`sampling_derivatives`, the old kernels, the old kinds), through the stated conversions | `tests/bridge/test_old_kernels_bridge.py` | P6 to P14; deleted in P20 |
| Translation (temporary) | each old call form of the frontend optimizers gives the result of the new one | the frontend optimizer tests | P16; deleted in P17 |

**Tolerances.** Numerical comparisons are to a tolerance scaled to the dtype; structural ones are exact (0.8, 9.3).

**Existing tests, sorted.** The slice lists of section 5.3 are the full record; this is the summary.

| File | Fate |
| --- | --- |
| `tests/test_jet_recurrence.py`, `tests/_jet_trs_twins.py` | Test old kernels only; deleted in P20 |
| `tests/backend/test_from_sweep_reuse.py` | Its calls updated in P12; deleted in P20 |
| `tests/test_chunk_size_estimator.py` | Its `newton_cg` case fed through the merged `probe` in P13; that case and `test_resolver` removed in P16; its estimator cases deleted in P20 |
| `tests/test_contractions_sharding.py` | Extended in P1 and P11; its class `TestJetKernelsShardOverW` (imports `sampling_derivatives`) deleted in P20, replaced by P11's |
| `tests/test_probe_derivatives.py` | Folded into the frontend sampling tests in P13, on the dense functions |
| `tests/test_uniform_probing.py`, `tests/test_manifold.py` | `test_manifold.py`'s `dense_probe` call in P2; their calls of the changed backend names in P12, of the frontend methods in P13; `test_manifold.py`'s `fitting.apply_model` call in P16 |
| `tests/test_tucker_tensor_train.py` | Its `dense_probe` call in P2; its frontend sampling calls in P13 |
| `tests/test_uniform_manifold.py`, `tests/test_uniform_tucker_tensor_train.py` | No call of a changed or removed sampling name (searched); unchanged unless a slice's search finds one |
| `tests/backend/test_curvature.py`, `tests/backend/test_uniform_curvature.py`, `tests/backend/_curvature_oracles.py` | The changed corewise transposes in P12; the backend `frame` in P15; the model parts in P16; their comparisons with the old two-hole and jet functions removed in P20; `_curvature_oracles.py`'s dense oracles kept as test helpers |
| `tests/test_fitting.py`, `tests/backend/test_uniform_fitting.py` | Their removed-method calls in P13; the backend `frame` in P15 (`test_uniform_fitting.py`); rewritten in P16 |
| `tests/backend/test_optimizers.py`, `test_padded_start.py`, `test_zero_reference.py` | Removed-method calls in P13; the backend `frame` in P15; rewritten in P16 |
| `tests/backend/test_geometry.py` | The backend `frame` in P15; rewritten in P16 |
| `tests/backend/test_optimizer_display.py`, `tests/test_optimizers_frontend.py`, `tests/test_sharing.py` (fitting), `tests/test_oracle_sweep.py`, `tests/oracle_sweeps/`, `tests/test_jax_absent.py`, `tests/test_aligned_padded_frame_frontend.py` | Removed-method calls in P13 where they have them; rewritten in P16 |
| `tests/test_dispatch.py` | Rows added from P1; changed calls in P12, P13, P15; fitting rows in P16; old rows removed in P20 |
| `tests/test_contractions_interpreter.py` | Extended in P1 |
| `tests/test_import_graph.py`, `tests/test_jax_absent.py` | New modules registered as they land; no backend module imports `safety` (P3) |
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
  weight: `check_sample`, `check_data`, `check_weight` and the two validators of `pp` (P3); for a whole batch,
  `LeastSquaresOracle.check_batch` (P14).
- **What the backend still checks, structurally, because it must to run:** the engine's unknown names, missing
  arguments and held shapes (section 2.3 step 1), the operand ranks in `jet_contract` (5.4), and exact mode's
  `optional_outputs` (P16). Nothing numerical.
- **What the frontend enforces.** Structural, always: the sample against `pp` in the merged methods; the batch through
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
  through `xscan`; map-type operations one site at a time, a loop on ragged and `xmap` on uniform (O-19); products
  of two full jets with `'loop'` on ragged and `'scan'` on uniform (O-10).
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
- `jet_contract` reshapes nothing either: both forms over the orders slice the order axis, never `W`.
- Every string at a call site is a literal, so the AST scan of `tests/test_contractions_sharding.py` sees it.
- The arrays of a batch are top-level arguments of the compiled phases, not closures (8.3 note).
- What holds today beyond the per-string sweep: the jetted transpose and the two-hole, run under `shard_map` with
  `W` sharded, compile with no all-gather and match the unsharded result (`tests/test_contractions_sharding.py`,
  class `TestJetKernelsShardOverW`). P11 pins the same property on the engine; the old class goes with
  `sampling_derivatives` in P20.
- **Tests.** The sharding sweep on the new literals and on `jet_contract`'s product strings (P1); `g` and `Kp` from
  the engine at orders 0 and 3, ragged and uniform, under `shard_map` with `W` sharded (P11).

### 7.7 Compiling with jit

- **Statics and leaves.** Static (`aux_data`, value-hashed): the geometry, the oracle (`kind`, `order`), the
  regularizer, `want` (a literal tuple at every library call site), the filled slots of `HeldVariables`, uniform
  masks and shapes, `GroupLengths`, `over_orders`. Leaves: the batch (the weight, the minibatch, its scale, its
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
`docs/fitting_and_optimization.md` (the oracle, the batch, the new signatures; the weight's exact shape; ones by
default with the asterisk of 2.5; derivative data prepared outside), `docs/batching_and_stacking.md` (`T` and `A`,
the sample stack on the outputs), `docs/grouped_contractions.md` (`jet_contract`, with one table of the letters that says which are grouped stack axes, `C`, `W`, `K`, and which are the two jet axes, `T` and `A`, whose meaning is multiplication of jets and not alignment of axes; and the definition of `A` as the two-slot format of the order group), `docs/frame_variations.md`
(centers), `docs/numerical_contracts.md` and `docs/contributor/numerical_contract_catalog.md` (the backend's
checkers, not guarantees; what the frontend enforces, and when), `docs/sharing.md`, `docs/rank_continuation.md` and `docs/aligned_padded_frame.md` where they call
optimizers, `docs/uniform_equivalence_contract.md`, `docs/contributor/fitting_internals.md`,
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
interface" entry reversed, 6.6 note; a new entry for jets in several variables, a future feature the maintainer named on 2026-11-17, with what the design implies for it: `T` and `A` with one axis per jet variable); `dev/OPEN_QUESTION_extension_surface.md` given its resolution (3.4) and
archived; `CLAUDE.md` (the paper-to-code map replaced by a pointer to its page; the sampling section; the
safe-mode paragraph, whose "Frontend-only for now (a backend mirror for raw-`.data` users is deferred)" becomes the
backend's checkers; the shipped surface; the module list);
`docs/release_notes.md` with the old-to-new name map (`SAMPLE_VARIABLES.md` section 10); `CHANGELOG.md`.

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
| `polynomial_predict` | `T.probe(ww)[-1]` | unchanged (`order=0`; the sample is `probe`'s only positional argument under N16 too) |

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
   - builds `batch = LeastSquaresBatch((ww, pp), y, weight)`, `val_batch` likewise, and
     `oracle = LeastSquaresOracle('probe', order=J)`;
   - calls `newton_cg(geometry, oracle, batch, X0, val_batch=val_batch, **kwargs)`, with `chunk_size` dropped from
     the options and `val_sample`, `val_data` turned into `val_batch`;
   - keeps T3Polynomial's own handling around the call (the uniform and jax conversions of `X0` and
     `padded_from`, the warm-start warning) as it is.
2. The scoring call `X.probe_derivatives(ww, pp, jet_order)` is given back to `TuckerTensorTrain`, in the script's
   process only, as `taylor_to_derivatives(X.probe(ww, pp, order=jet_order), jet_order)`, so that the consumer's
   misfits keep their derivative form.

The script checks that every `stats` key T3Polynomial reads is present, with the meaning it has today; `timing`
is not read by T3Polynomial (its keys change, O-17).

**What it checks.** On T3Polynomial's toy problem (a known polynomial, `t3polynomial/toy.py`), two levels of
`fit_rank_continuation` and of `fit_rank_continuation_adaptive`, on each backend (`'ragged'`, `'uniform'`,
`'uniform_jax'`): the fit reaches the known answer to the validation error the release reaches; the display's table
appears; every `newton_cg` option T3Polynomial passes is accepted. The script's output is recorded in the folder
like the other checks.

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
none of the 129 removed names, nor `chunk_size`, `sum_over_probes` or `ztildes`.

---

## 11. The open points

### 11.1 How they were answered

The maintainer answered O-1 to O-28 on 2026-11-16, 18:35 to 20:35, and O-29 to O-35 at 20:59 (`TARGET_DECISIONS.md`,
the note "The plan's open points, as the maintainer answers them" and the notes after it). The plan states each
answer where it uses it.

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
| O-27 | `dense_probe`, `dense_apply`, `dense_entries` with `pp` and `order`, backend, used by the reference module, the examples and the tests (2.4, P2, P13) |
| O-28 | The ambient transposes in N16's order and names, the sum off by default; `dense_probe` as O-27 (2.4) |
| O-29 | `check_batch` is an optional member of the oracle protocol; the frontend calls it when the oracle has it, and checks nothing about the batch otherwise (2.4, 2.5, P14, P16) |
| O-30 | `held` is any object; in a compiled program, something jax can carry: arrays, tuples, `None` or a registered record (2.2, 2.4) |
| O-31 | The geometry has no consistency member; N21 withdrawn; a backend user calls `fv_frame_consistency_residual` (2.4, 3, P15) |
| O-32 | Settled by the amendments of 19:30 to rows 4.2 and 8.3 (7.1) |
| O-33 | The page takes the notation paragraph of `CLAUDE.md` and the probing map of `docs/entries_apply_probe.md`; other pages link to it (8) |
| O-34 | `UniformTuckerTensorTrain.unscale_coordinates`, over `ut3_unscale_coordinates` (2.4, 2.5, P3) |
| O-35 | One scale per coordinate: `scales` is `len=d`, `elm_shape=(Ni,)` (2.4, P3) |

### 11.2 Open points

None. Every point is answered, and every name approved.

### 11.3 Names added after the table was settled

Approved by the maintainer on 2026-11-16 at 20:59, with the uniform twin of O-34 named by the library's grammar.

| Name | What | Where | Slice |
| --- | --- | --- | --- |
| `over_orders='scan'`, `'loop'` | The values of `jet_contract`'s form over the orders; passed at every call, with no default | `backend/contractions.py` | P1 |
| `dense_apply(T, ww, pp=None, order=0)`, `dense_entries(T, index, pp=None, order=0)`; `dense_probe(T, ww, pp=None, order=0)` | The dense functions | `backend/apply.py`, `entries.py`, `probing.py` | P2 |
| `check_sample(kind, order, sample, shape)`, `check_data(kind, order, sample, y)`, `check_weight(kind, order, weight, d)` | The checkers of a sample, its data and its weight | `backend/jets.py` | P3 |
| `ut3_unscale_coordinates(x, scales)`; `UniformTuckerTensorTrain.unscale_coordinates(scales)` | The uniform twin of `t3_unscale_coordinates` and its method (O-34) | `backend/data_preparation.py`; `uniform_tucker_tensor_train.py` | P3 |
| `fv_centers_consistency_residual`, `ufv_frame_consistency_residual`, `ufv_centers_consistency_residual` | The centers' residual and the uniform twins of both frame residuals | `backend/fv_operations.py`, `backend/ufv_operations.py` | P4 |
| `T3Frame.consistency_residual`, `UT3Frame.consistency_residual` | The cached residual the frontend checks | `frame_variations_format.py`, `uniform_frame_variations_format.py` | P4 |
| `HeldVariables.with_values(**values)` | A new record with the given slots filled | `backend/sampling_variables.py` | P5 |
| `LeastSquaresOracle.check_batch(batch, shape)` | The checker of a whole batch; an optional member of the protocol | `backend/fitting.py` | P14 |
| `pack_batch(kind, order, batch, N)` | The packing of a batch for the uniform layout | `backend/uniform_fitting.py` | P14 |
| `fitting.local_model(..., frame=None)` | The keyword through which a caller supplies a frame | `t3toolbox/fitting.py` | P16 |
| `docs/paper_to_code.md` | The page of the paper-to-code map | `docs/` | P19 |

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
- **The agents:** both Fable 5.2 on medium effort (the maintainer, 20:51).
- **Memory.** The gate's steps run one after another, never two at once, and the implementer and the reviewer
  never run tests at the same time (the maintainer, 21:02: running sphinx and a parallel test run together has
  run a 13 GB machine out of memory). The workspace the agents run in has 2 cores and 7 GB. P0 measures the suite's
  peak memory and time with one worker and with two before a parallel run is relied on.
- **Issues.** Major issues go to the maintainer. A departure from a decided row, from a public name, or from the
  slice order is a major issue. Minor issues Claude decides, and records the decision in the slice's note.
