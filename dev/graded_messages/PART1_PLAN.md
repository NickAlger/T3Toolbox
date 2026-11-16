# Part 1 of the graded-messages refactor: implementation plan

**Status.** Draft r2, 2026-11-16, for review. Written against the branch `graded-messages` at `7df35fe` (the library
and `tests/` unchanged since `4d38df7`). The names
in this plan are proposals until the maintainer settles the table of names (section 3). Every statement about the
library was checked against the code at that commit; counts were recounted there.

**How to read it.** The rows of `TARGET_DECISIONS.md` are the authority. A rule is given here with its row number
(for example "2.9") and, where a reader needs it, one sentence of reason. `SAMPLE_VARIABLES.md` is the reference
for every variable: its name, definition, recurrence and shape. The prototypes in `scripts/review/` establish the
algebra; the library's contributor notes (`docs/contributor/`) fix the form of the code.

---

## 1. Scope

**Part 1 delivers what is computed.** The sampling variables of `SAMPLE_VARIABLES.md` under their decided names;
one engine, `sampling_variables`, that computes any of them for the three kinds, in Taylor coefficients, at any
order, with order 0 as the same functions; `jet_contract` and a literal contraction string at every call site;
the weight and the data as arguments; frames with optional centers; arbitrary tails; both layouts. Above it: the
oracle and the batch, `LocalModel`, the optimizers' new seam, the display, and standalone functions that prepare
data and weights. Around it: the reference module of definitions, the tests by definition and the identities,
the documentation of the variables, and a dry port of T3Polynomial.

**Part 1 carries out every computation the way the library does it today.** Where a row is open it fixes the
following ("The two parts"):

| Open row | What Part 1 does |
| --- | --- |
| 5.3 sites | Sweeps go through `xscan`. Map-type operations loop over the sites on the ragged layout and run at all sites at once on the uniform one. Today's jet kernels do not do the latter (open point O-19) |
| 5.5 contraction order | `contract`'s present default: one einsum on jax, the cached greedy pairwise path on numpy |
| 7.2 defaults | The model holds the plain sweep variables and the residual |
| 7.3 where the plan lives | There is no plan, and no signature has a `plan` argument |
| 7.5, 7.6, 7.7 | No chunking of samples, so no padding of samples, no sizing helpers, no promise about sharding |

Decided rows whose options wait for Part 2: 2.9 (Part 1 has the scan form only), 4.3 (Part 1 has trial policy a,
with the centers of 4.1 in place), 7.1 and 7.4 (no plan), 8.3 (Part 1 compiles the CG solve and the stochastic
step, as today, and makes `take` traceable).

**What Part 1 removes that Part 2 gives back.** The frontend `chunk_size` and the library's two chunked uniform
assemblies (`_wchunked_sum` in `backend/sampling_derivatives.py`) go. Until Part 2, a large jetted fit on the
uniform layout with jax uses more memory than today. T3Polynomial's fitting module does chunk today: with
`newton_kwargs=None` it passes no `chunk_size` (`t3polynomial/fitting.py`, `_level_newton_kwargs` and
`_newton_fit`), so the library default `chunk_size='auto'` (`t3toolbox/optimizers.py`) applies on its uniform
backends. Its benchmark `scripts/b01_backend_bench.py` passes `chunk_size=None`, and so do some of its scripts.

**A second possible step back in memory.** If the jetted map-type operations run at all sites at once on the
uniform layout (the reading of 5.3 in "The two parts"), their intermediates are `d` times larger than today's,
which maps them one site at a time with `jax.lax.map` (open point O-19).

**The seams left for Part 2.** Part 2 adds; it does not change Part 1's interface.

| Seam | Where Part 2 adds |
| --- | --- |
| The plan | An optional `plan=None` keyword on `sampling_variables`, and wherever 7.3 puts it (the oracle, the model or the solver). No Part 1 signature has to change |
| The form over the orders (2.9) | An optional setting reaching `jet_contract`; Part 1's form is the scan |
| What is held (7.2) | The least-squares oracle's fixed hold list becomes a field of the plan; `held` already flows through every call |
| Chunking (7.4) | Chunks are array shapes of the batch; the engine already accepts any number of sample axes `W` |
| Trial policies b and c (4.3) | An optimizer option; the geometry already returns centers with the frame |
| Several MC-SGD steps in one program (8.3) | The stochastic step is a module-level function of pytrees, and `take` is traceable with a static size |
| Contraction order (5.5) | Every term is a literal string at its call site, so a per-term order has one place to attach |

Part 1 is not released on its own. All work stays on `graded-messages` until Part 2 is done.

---

## 2. The end state

### 2.1 Modules

| Module | Holds at the end of Part 1 | Status |
| --- | --- | --- |
| `backend/contractions.py` | `contract` (unchanged); `jet_contract`; the record of axis counts `GroupLengths` | changed |
| `backend/sampling_variables.py` | The engine `sampling_variables`; the declared graph, one tuple of records per kind; the shape table | new |
| `backend/sampling_steps.py` | The 22 step functions and the start values. Every name private, `__all__ = []` | new |
| `backend/sampling_definitions.py` | The reference module: one slow, dense function per variable and per output (9.7) | new |
| `backend/jets.py` | `jet_to_dual_jet` and `dual_jet_to_jet` (3.6); the four functions for derivative data (2.3); the two validators of `pp`, moved from `sampling_derivatives` | new |
| `backend/data_preparation.py` | The five preparation functions (6.7) | new |
| `backend/probing.py`, `apply.py`, `entries.py` | One backend function behind each frontend method, each calling the engine, with the merged signature (6.3, 6.6); the ambient transposes and `dense_probe` as they are | rewritten |
| `backend/ut3_sampling.py`, `utv_sampling.py` | The uniform twins of the functions above: check packing, call the engine (open point O-3); `ut3_full_sum` as it is | rewritten |
| `backend/sampling_derivatives.py` | Deleted in the last slice | removed |
| `backend/fitting.py` | `LeastSquaresOracle`, `LeastSquaresBatch`, `HeldVariables`; the default draw's `take` and `n_measurements` | rewritten |
| `backend/uniform_fitting.py` | `pack_sample`, `pack_data` (taking `(kind, order)`), `uniform_minimal`. The six uniform kinds and their factories go | rewritten |
| `backend/optimizers.py` | The `Geometry` and `Oracle` protocols; `LocalModel`; `local_model`; `point_objective`; the five phases; the four optimizers; `flat_draw`. `Problem`, `least_squares_problem` go | rewritten |
| `backend/optimizer_display.py` | The display, reading from an oracle and a batch | changed |
| `backend/geometry.py` | `frame(x)` returns the frame with its centers; the corewise geometries' frame is the point; a frame-consistency member | changed |
| `backend/fv_operations.py` | `fv_frame_consistency_residual` without dense tensors, and the centers' residual (4.2); the uniform twins beside it if O-5 asks for one | changed |
| `t3toolbox/tucker_tensor_train.py`, `uniform_tucker_tensor_train.py`, `manifold.py`, `uniform_manifold.py` | Merged sampling methods (6.6) | changed |
| `t3toolbox/frame_variations_format.py`, `uniform_frame_variations_format.py` | `is_consistent` without dense tensors | changed |
| `t3toolbox/fitting.py` | The frontend local model over the backend one (open point O-9) | rewritten |
| `t3toolbox/optimizers.py` | The four optimizers with `(geometry, oracle, batch, x0, ...)` | rewritten |
| `t3toolbox/shared_geometry.py` | Unchanged in behaviour; its `frame` follows the backend change only where it calls the backend | checked |

The import rules stay: the backend graph is acyclic, and `backend/optimizers.py` stays T3-free
(`tests/test_import_graph.py` pins its imports to `common`, `regularization`, `corewise`; open point O-6 asks to
add `safety`).

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
(manifold). On a corewise call
the variables have no basis in their names (`SAMPLE_VARIABLES.md` section 9). The uniform half of this table is
a proposal (open point O-3).

**Centers** `(Vc, Hc)`: `Vc[i]` has the shape of `V[i]`, `Hc[i]` that of `H[i]`. They are the second result of
`fv_conversions.t3_orthogonal_representations` (its "variations"), which the geometry's `frame` discards today
(`backend/geometry.py`, `ManifoldGeometryOps.frame`). The engine takes `Vc` only (4.1).

**A sample.** `ww` (probe, apply) or `index` (entries) when `order == 0`; `(ww, pp)` or `(index, pp)` above.
`order` decides, never the type (6.2). `ww[i]` is `W + (N_i,)`; `index` is `(d,) + W`, integer. On the uniform
layout the vectors are packed once, when the batch is made (`pack_sample`), and the engine is strict about it.

**The data `y`** is of the type of `z`, in forward order (2.4): probe `y[i]` is `T + W + (N_i,)`; apply and entries
`y` is `T + W`. It broadcasts over `C`.

**The weight** has exactly the shape `(d,) + T` for probe and `T` for apply and entries; no rule for a bare vector,
no broadcast of a mode dimension of 1 (2.5, 6.7). `None` means ones.

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

A user's oracle with no data of its own takes `None` as its batch.

**An oracle** is a frozen description with no arrays and one operation (8.2, 8.3). Its fields are statics, so it
is value-hashed (`common.ValueHashedFields`) and rides as `aux_data`. The least-squares oracle has two fields,
`kind` and `order`.

**`held`** is what an oracle keeps at one frame; what goes into it is the oracle's business (8.3), and the
optimizers pass it back unread. For the least-squares oracle it is a set of named sampling variables, passed to the
engine as keyword arguments. Its form, and how the protocol types it, are open (O-8); the plan assumes, for the
least-squares oracle, a small frozen record `HeldVariables` registered as a pytree: the names are static
`aux_data`, the arrays are leaves.

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
| `held` | leaves (names aux) | what the oracle returned at the build |
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
        **held:            NDArray,                          # held variables by name; shapes checked, values trusted
) -> typ.Tuple[typ.Any, ...]:                                # aligned with want; shapes in the table of variables
```

The three placeholder keywords of 6.2 are proposed here as `variations`, `core_directions`, `sum_over_samples`
(table of names, rows N1 to N3). The `plan` of the 6.2 sketch is absent in Part 1 (7.3).

**What a call does, in order.**

1. **Structure, always.** The kind string, `order`, the form of `sample`, every name in `want` and in `held` is in
   the kind's graph, every argument a wanted name needs is present (a missing one raises: `Kp` without
   `core_directions`; `r` without `y` and without a held `r`), held shapes match the table with the order
   included (6.5), the weight has its exact shape. Structural errors raise in both safety modes.
2. **Layout and call type** from the structure of `frame` (section 2.2). On the uniform layout the inputs are masked
   once, here, and nothing below sees a mask (5.2). Every mask comes from the object the operation is defined at,
   the frame, never from a superset of it (`docs/contributor/testing_strategy.md`; CLAUDE.md on the two-hole): the
   frame's stores with their own masks; `variations` and `Vc` with the frame's variation masks
   (`ufv_masking.ufv_variation_masks`, as the uniform geometry does today); `core_directions` `(D_dot, L_dot)` with
   the masks of the frame's `D` and `L` stores, not the point's (which are a superset at TT slack).
3. **Backend and dtype.** `use_jax = tree_contains_jax(...)` over every array input. The result dtype follows the
   absorbing rule (5.6): jax over numpy, the lowest floating precision among the counted inputs. The inputs are
   cast once to it; every start value is built in it; constants are Python scalars. (Which inputs count is open,
   O-11; the proposal is the frame, `Vc`, the sample, `y`, the weight, the two directions and the held variables,
   not an integer index.)
4. **Safe mode.** When `Vc` is given and `safety.checks_active(...)`, the centers' agreement with `(D, L)` is
   checked through the cores (4.2; small item 2). Nothing else numerical is checked.
5. **The closure.** From `want` and the held names, the declared graph gives the set of variables to compute,
   in dependency order. Not wanting a variable lets a step take its short form (`eta_tilde` in its two read rows
   when only `g` reads it); wanting it never changes a value (6.4).
6. **The sweeps and the maps.** A sweep along the chain goes through `xscan` with a module-level body
   (`docs/contributor/scan_body_principles.md`). A map-type operation (the `xi`, `eta`, `z` steps and the block
   readouts) is a loop over the sites on the ragged layout and one call with the site group `S` on the uniform
   layout, as "The two parts" fixes for 5.3 (for the jetted ones see O-19). Boundary vectors are ones sized by the
   end bond of the cores passed; start values follow the grade (4.4). The engine squashes no tail.
7. **The seed.** `r = z - y` (the library's sign, 2.4); the seed `w_tilde = jet_to_dual_jet(r, weight)`: the
   weights, then the turn of the order leg. `JtJp` seeds from `Jp` the same way. A held `r` is used as it is.
8. **The sums.** `f = 1/2 <r, omega^2 r>`, `r_block_sumsq` (unweighted, the shape of the weight),
   `Jp_norm_sq = <Jp, omega^2 Jp>` are formed inside the call; `f` keeps `C`, `Jp_norm_sq` keeps `K + C`
   (3.7).
9. **The return**, a tuple aligned with `want`. On the uniform layout the core-shaped outputs (`g`, `JtJp`, `Kp`, the
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
frame form for `Jp`/`JtJp` and the raw form for `Kp` internally, and the public name refers to the one for `Kp`
(the raw form, at the base point). That is a reading of "One dot"; open point O-12 asks the maintainer to confirm
it. The tilde names always mean the residual's adjoint sweep (3.3).

**The declared graph** is plain data: per kind, a tuple of records `(name, step, reads, needs)` (6.4). It lives in
`sampling_variables.py` beside the engine; the docstring's table of variables is written by hand and a test checks
the two agree row for row.

**What the engine does not know.** The geometry (projection, lift, Hessian assembly stay there, 4.5), the
regularizer, any loss other than the weighted sum of squares (3.6), derivative data (2.2), and plans.

### 2.4 The public surface, backend

**New.**

| Function or class | Module | Signature sketch |
| --- | --- | --- |
| `sampling_variables` | `sampling_variables` | section 2.3 |
| `jet_contract`, `GroupLengths` | `contractions` | below |
| `jet_to_dual_jet`, `dual_jet_to_jet` | `jets` | below |
| `derivatives_to_taylor`, `taylor_to_derivatives`, `derivative_weight_to_taylor_weight`, `prepare_derivative_data` | `jets` | below; the bodies as in `scripts/review/adapter_doctest_draft.py`, with the probe kind's mode axis |
| `data_block_norms`, `equalizing_weight`, `rescale_direction`, `rescale_coordinates`, `t3_unscale_coordinates` | `data_preparation` | below |
| one function per variable and output | `sampling_definitions` | below |
| `Oracle` (protocol) | `optimizers` | below |
| `LeastSquaresOracle`, `LeastSquaresBatch`, `HeldVariables` | `fitting` | below and section 2.2 |
| `local_model`, `point_objective` | `optimizers` | below |

```python
@dataclass(frozen=True)
class GroupLengths:                 # the axis counts of one call's groups (5.4); hashable, a static
    S:  int                         # 1 on a uniform map-type operation, else 0
    T:  int                         # 1 if order else 0; also the count of the affine group A
    W:  int                         # sample axes
    K:  int                         # direction-stack axes
    C:  int                         # base-point-stack axes

def jet_contract(
        subscripts:  str,           # a literal grouped string; T a full jet's order group, A an affine one's
        *operands:   NDArray,
        lens:        GroupLengths,  # every call passes the whole record, so a wrong rank raises
) -> NDArray:                       # contract(), with T and A multiplying as truncated polynomials
```

`jet_contract` follows the rule of 5.4: one operand with an order group and the output with it, the axis rides
(one `contract`); two operands with one and `T` in the output, the truncated product of 2.9 (two contractions for
an affine partner; for two full jets the scan with the other jet carried and shifted one slot per step, through
`xscan`); no order group in the output, the top coefficient of the product. At order 0 it is exactly one
`contract` call. `contract` raises on a `len_<G>` whose group is not in its string
(`contractions._solve_group_lengths`), so `jet_contract` passes on only the lengths of the groups that each string
it hands to `contract` names; the strictness comes from those being all of them. How the scan learns the layout is
open (O-10; the proposal is a sixth field of `GroupLengths`).

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
                                           #   (what it does where a weight is zero is open, O-20)

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
        scales:  NDArray,                  # shape (d,): mode i's input coordinates divided by scales[i]
) -> typ.Tuple[
        typ.Any,                           # the sample
        typ.Any,                           # y (for probe, mode i's data rescaled too: 6.7 note)
]:

def t3_unscale_coordinates(
        x:       typ.Tuple,                # (tucker_cores, tt_cores) fitted in rescaled coordinates
        scales:  NDArray,                  # shape (d,), as given to rescale_coordinates
) -> typ.Tuple:                            # the same tensor in the original coordinates
```

The two estimates the prototype reads off the data (a length for the directions, a coordinate scale) are not
decided (6.7 note of 18:32); this plan does not ship them (open point O-14).

```python
def mu(kind: str, frame, sample, i: int, order: int = 0, *, variations=None, core_directions=None,
       w_tilde=None) -> NDArray:           # sampling_definitions: mu[i] by its definition, dense, no sweep
```

One such function per name of the table, all with the same arguments, each built as the dense contraction of
its subnetwork (`SAMPLE_VARIABLES.md`, the column "As a subnetwork"). Jets come from interpolation in `s`, which is
exact for a polynomial (9.3). Outputs: `z` from the dense tensor; `Jp` from the dense tangent tensor; `g`,
`JtJp` and the tilde variables from the dense Jacobian; `Kp` from the second derivative of the dense objective
(jax where present, finite differences otherwise). Ragged layout only; small problems only.

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
        typ.Any,                           # held: what to keep at this frame; a pytree the optimizers never read
    ]: ...

@dataclass(frozen=True, eq=False)
class LeastSquaresOracle(ValueHashedFields):
    kind:   str                            # 'probe' | 'apply' | 'entries'
    order:  int = 0
    # query(...): one call of sampling_variables. Its batch is a LeastSquaresBatch.
    # n_measurements(batch) -> int;  take(batch, indices) -> LeastSquaresBatch (traceable, static len(indices))
```

The least-squares oracle's `query` asks the engine for `want` plus the names of its hold list that the declared
graph says the call computes anyway, and returns those as `held`; a call never computes a variable only to hold
it (8.3). Part 1's hold list is fixed: `xi`, `mu`, `nu`, `eta`, `r` (7.2 "as today"). On a drawn batch it
multiplies `f`, `g`, `JtJp`, `Jp_norm_sq` and `Kp` by `minibatch_scale`, so that `f` estimates the full one (8.3
part 2). Whether `r_block_sumsq`, `r` and `Jp` are scaled too is open (O-21). `y_block_sumsq` needs no
frame (the display asks once). `f_resolution` is `eps (|omega y| |omega r| + |f|)`, the formula of
`_objective_resolution` today, with `|omega y|` from the same pass. `n_measurements` and `take` are methods of
the least-squares oracle, not of the protocol (8.3 part 2).

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
) -> LocalModel:

def point_objective(
        geom,                                 # a backend Geometry
        oracle,                               # an Oracle
        batch,                                # its batch
        x_cores:      typ.Tuple,              # the raw cores (B, G), ragged or the uniform supercore pair
        regularizer=None,
) -> NDArray:                                 # f (+ rho(x)), shape C: a corewise query of 'f' at x_cores
```

**Changed.** Each keeps its name and calls the engine; the merged signature of 6.6. Sketches for probe; apply and
entries follow (an `index` in place of `ww` for entries, a scalar output for both). Today's leading arguments are
kept and `pp`, `order` added as keywords; `sum_over_samples=True` replaces `sum_over_probes=False` (3.7); the
residual argument is named `residual` (2.4). The transposes and the corewise Hessian take a residual of the type of
`z` and no weight, and turn it inside (2.4, 3.6); see O-22.

```python
def t3_probe(
        ww:     typ.Sequence[NDArray],                # len=d, elm_shape=W+(Ni,)
        x:      typ.Tuple,                            # (tucker_cores, tt_cores), elm_shapes C+(nDi,Ni), C+(ri,nDi,r(i+1))
        pp:     typ.Optional[typ.Sequence[NDArray]] = None,   # len=d, elm_shape=W+(Ni,); given iff order >= 1
        order:  int = 0,                              # highest Taylor order.  T=(order+1,) if order else ()
) -> typ.Tuple[NDArray, ...]:                         # z. len=d, elm_shape=T+W+C+(Ni,)

def tv_probe_transpose(
        residual:          typ.Sequence[NDArray],     # of the type of z: len=d, elm_shape=T+W+[K]+C+(Ni,)
        ww:                typ.Sequence[NDArray],     # len=d, elm_shape=W+(Ni,)
        frame:             typ.Tuple,                 # (D, U, L, R)
        pp:                typ.Optional[typ.Sequence[NDArray]] = None,
        order:             int = 0,
        sum_over_samples:  bool = True,               # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                      # V blocks. len=d, elm_shape=[W]+[K]+C+(nUi,Ni)
        typ.Tuple[NDArray, ...],                      # H blocks. len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))
]:

def t3_probe_corewise_hessian(
        residual:         typ.Sequence[NDArray],      # of the type of z: len=d, elm_shape=T+W+C+(Ni,)
        ww:               typ.Sequence[NDArray],      # len=d, elm_shape=W+(Ni,)
        x:                typ.Tuple,                  # (B, G): the point whose cores the directions vary
        core_directions:  typ.Tuple,                  # (B_dot, G_dot), shapes of B, G with K+C
        pp:               typ.Optional[typ.Sequence[NDArray]] = None,
        order:            int = 0,
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                      # Tucker block. len=d, elm_shape=K+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                      # TT block.     len=d, elm_shape=K+C+(ri,nDi,r(i+1))
]:
```

`tv_probe(ww, frame, variations, pp=None, order=0)` returns `Jp`, `T+W+K+C+(Ni,)`, and
`t3_probe_corewise_transpose(residual, ww, x, pp=None, order=0, sum_over_samples=True)` is the transpose on a
corewise call; both follow the layout above. The argument order is a proposal (names table, N16). The uniform twins (`ut3_probe`, `utv_probe`,
`utv_probe_transpose`, `ut3_probe_corewise_transpose`, `ut3_probe_corewise_hessian`, and apply and entries) take
the same arguments (open point O-3). In `backend/optimizers.py`: `LocalModel` (section 2.2),
`newton_cg(geom, oracle, batch, x0, regularizer=None, ...)`, `mc_sgd(geom, oracle, batch, x0, rng, batch_size,
draw=None, regularizer=None, ...)`, `adam(...)` likewise, `gradient_descent(geom, oracle, batch, x0,
regularizer=None, ...)`, `flat_draw(oracle, batch, batch_size)` returning `draw(rng) -> batch`. The
`Geometry` protocol: `frame(x)` returns `(frame, centers)`, the corewise geometries' frame is the point `(B, G)`,
and a consistency member is added (open point O-4 for the form). In `optimizer_display.py`:
`make_newton_display(oracle, batch, val_batch=None, ...)`; `relative_errors` and the table take arrays of the shape
of the weight. In `uniform_fitting.py`: `pack_sample(kind, order, sample, N)` and `pack_data(kind, order, y, N)`.

**Removed.** Counted from the `__all__` lists of the six sampling modules, 166 public names today:

| Module | Today | Changed (call the engine) | Kept as they are | Removed |
| --- | --- | --- | --- | --- |
| `probing` | 34 | 5 | 2 (`t3_probe_ambient_transpose`, `dense_probe`) | 27: every `compute_*`, `assemble_*`, `tv_precompute_probe_frame_sweep`, the three `*_from_sweep` |
| `apply` | 10 | 5 | 1 (`t3_apply_ambient_transpose`) | 4: the sweep and `*_from_sweep` functions |
| `entries` | 10 | 5 | 1 (`t3_entries_ambient_transpose`) | 4: likewise |
| `ut3_sampling` | 25 | 9 | 1 (`ut3_full_sum`) | 15: every `*_derivatives*` and `*_from_sweep` |
| `utv_sampling` | 30 | 6 | 0 | 24: the sweeps, `*_from_sweep`, every `*_derivatives*` |
| `sampling_derivatives` | 57 | 0 | 2 moved to `jets` (`check_perturbation_vectors`, `check_perturbation_index`) | 55, the module with them (the chunk estimators and `dense_*_derivatives` included) |
| **Total** | 166 | 30 | 7 | 129 |

Also removed: in `backend/fitting.py` all 20 names (the nine kind classes, the three singletons, the four kind
factories, the four `sumsq` helpers); in `backend/uniform_fitting.py` 9 of 12 (the six uniform kinds,
`uniform_sampling_kind`, `uniform_derivatives_kind`, `uniform_least_squares_problem`); in
`backend/optimizers.py` `Problem`, `least_squares_problem`, `_minibatch_step_problem`, `_ScaledRegularizer`.

### 2.5 The public surface, frontend

| Class | Sampling methods today | After | Removed |
| --- | --- | --- | --- |
| `TuckerTensorTrain` | 15 | 9: `entries`, `apply`, `probe` and their `*_corewise_transpose` with `(…, pp=None, order=0)`; the three ambient transposes as they are | the 3 `*_derivatives`, the 3 `*_corewise_derivatives_transpose` |
| `T3Tangent` | 12 | 6: `probe`, `apply`, `entries` and their `*_transpose`, merged | 6 |
| `UniformTuckerTensorTrain` | 12 | 6 | 6 |
| `UT3Tangent` | 12 | 6 | 6 |

Each merged method returns `T + W + [K] + C + ...` with `T` outermost, Taylor coefficients (6.6). The transposes
take a `residual` of the type of `z` and sum the samples by default; their keyword is the engine's
`sum_over_samples` (names table, N25). The sketch for one method and one transpose; the others follow:

```python
    def probe(                                        # TuckerTensorTrain
            self,
            ww:     typ.Sequence[NDArray],                    # len=d, elm_shape=W+(Ni,)
            pp:     typ.Optional[typ.Sequence[NDArray]] = None,   # len=d, elm_shape=W+(Ni,); given iff order >= 1
            order:  int = 0,                                  # highest Taylor order.  T=(order+1,) if order else ()
    ) -> typ.Tuple[NDArray, ...]:                             # len=d, elm_shape=T+W+C+(Ni,)

    def probe_transpose(                              # T3Tangent
            self,
            residual:          typ.Sequence[NDArray],         # len=d, elm_shape=T+W+[K]+C+(Ni,)
            ww:                typ.Sequence[NDArray],         # len=d, elm_shape=W+(Ni,)
            pp:                typ.Optional[typ.Sequence[NDArray]] = None,
            order:             int = 0,
            sum_over_samples:  bool = True,
    ) -> 'T3Tangent':                                         # at self.frame; stack [W]+[K]+C
```

Fitting and optimization (frontend). The four optimizers take `(geometry, oracle, batch, x0, ...)`, with `val_batch`
in place of `val_sample`/`val_data`, and without `kind`, `sample`, `data`, `order`, `weight`, `chunk_size`.
`mc_sgd` and `adam` take `batch_size` (8.x note). The oracle is the same object in the frontend and the backend;
the batch is ragged, and the frontend packs it for a uniform `x0`. The keywords not shown are today's:

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

The six model factories of `t3toolbox/fitting.py` (and their re-exports from `t3toolbox/__init__.py`),
`least_squares_problem` and `LeastSquaresSetup` go; what replaces them is open (O-9).

---

## 3. Table of names (awaiting the maintainer)

**Every row is a proposal.** Nothing else in this plan treats them as decided. Proposals follow
`docs/naming_conventions.md` (family prefixes for functions of a representation; none for
representation-agnostic infrastructure) and `docs/contributor/naming_rules.md` (clarity wins; no generic names).

| # | Placeholder | Proposed | Why |
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
| N16 | the argument order of the merged backend functions | `t3_probe(ww, x, pp=None, order=0)` and the sketches of section 2.4 | Keeps today's leading arguments; `pp` and `order` are added as keywords |
| N17 | the reference module; its functions | `backend/sampling_definitions.py`; one function named exactly as the variable (`sampling_definitions.mu`) | The module name says what they are; the function name is the variable |
| N18 | the engine's module; the steps' module | `backend/sampling_variables.py`; `backend/sampling_steps.py` | The module is named after its function, as `regularization` is |
| N19 | the residual argument of the transposes (`ztildes` today) | `residual` | 2.4 calls `ztildes` a misnomer |
| N20 | the raw-cores objective of policy a | `point_objective` | "f at raw cores" |
| N21 | the frame-consistency member of a geometry | `frame_consistency_residual(frame, centers)` | Pairs with `fv_frame_consistency_residual` |
| N22 | the page in `docs/` | `docs/sampling_variables.md` | |
| N23 | the example file | `examples/tour_sampling_variables.py` | |
| N24 | the `.tex` note | `docs/sampling_variables.tex` | |
| N25 | the frontend transposes' keyword in place of `sum_over_probes` | `sum_over_samples` | The engine's (N3) |
| N26 | the modules of the fitting records | `LeastSquaresOracle`, `LeastSquaresBatch`, `HeldVariables` in `backend/fitting.py`; the `Oracle` protocol, `LocalModel`, `local_model`, `point_objective` in `backend/optimizers.py` | The protocol beside `Geometry`, so `optimizers.py` stays T3-free; the least-squares pieces where the kinds are today |
| N27 | the module of N14 and N15 | `backend/contractions.py` | `jet_contract` is `contract` with one more rule |
| N28 | the module of the uniform twins | `backend/ut3_sampling.py`, `backend/utv_sampling.py`, as today | Grouped by object type, as the naming page records |
| N29 | the script that draws the dependency diagrams | `docs/sampling_diagrams.py` | Beside the page it draws for (O-23) |

---

## 4. How the code is written

These are the library's rules, applied to this arc. Every slice is held to them; the reviewer checks them first.

- **One set of step functions for both layouts** (5.2). A step is named by letter and grade and serves both bases
  (5.1 note): `nu_step` given `L` cores returns `nu_L`. The 22 steps of `scripts/review/jet_steps_check.py` stand.
- **Every term is a literal string at its call site, through `jet_contract`**, with `K` on the operand that carries
  it and the record `GroupLengths` passed whole (5.4). The 41 strings of the seven families are listed in
  `scripts/review/term_stacks_check.py` (`STRINGS`). The site group `S` appears on the map-type strings only.
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
- **The razor**: anything a raw-`.data` user would get wrong is in the backend; the frontend validates its objects
  and delegates.
- **Doctests** teach, are seeded, print structure, and are run and pasted (`docs/contributor/doctest_style.md`).
- **No auto-formatter.**

---

## 5. The work in slices

### 5.1 How the suite stays green

- **Additive first.** Slices P1 to P11 add new modules beside the old ones. The one old thing they change is P4's
  consistency residual, which keeps its signature (`fv_frame_consistency_residual`, `T3Frame.is_consistent`).
  So the suite stays green by construction.
- **Then switch, one surface at a time.** P12 to P16 move one surface each onto the engine. Each switching slice
  rewrites the tests, doctests, documentation pages and examples that call the surface it changes, in the same
  commit, because CI runs all of them (`.github/workflows/tests.yaml`: the suite, the module doctests,
  `getting_started.rst`, every `docs/*.md` and `docs/contributor/*.md` page except `doctest_style.md`, and every
  `examples/*.py`).
- **The old kernels stay until the last slice** (small item 3). After a surface switches, its old kernels are still
  in the tree and still tested by their own tests.
- **The bridge** (9.3: "the old library is a bridge used once during the port") is read here as a temporary test
  file, `tests/bridge/test_old_kernels_bridge.py`, that compares new outputs with old code through the stated
  conversions (`t!` per slot, the turn, the sum flag). It compares only with code that is never rerouted onto the
  engine: the functions of `sampling_derivatives` (at order 0 too), the old `compute_*` and `assemble_*` kernels,
  and the old kinds. It is deleted with them in P19. Whether "used once" meant a one-off script instead is O-25.
- **Each slice's gate**: the full suite, `pytest --doctest-modules t3toolbox`, the page doctests, the examples, and
  `sphinx -W`, run unpiped on a clean tree (`docs/contributor/refactoring_methodology.md`, "Gates").
- **Per slice**: an inventory first (every inbound reference to what the slice changes, both import forms,
  `getattr` strings, slash-glob prose); exact-equality assertions in every test file the slice rewrites are sorted
  into structural (exact) and numerical (tolerance) (0.8).

### 5.2 Overview

| Slice | Content | Size |
| --- | --- | --- |
| P0 | Baseline: the folder's check suites on the branch head; the inventory | none in the library |
| P1 | `jet_contract` and `GroupLengths` | small |
| P2 | The reference module `sampling_definitions` | medium |
| P3 | `jets` and `data_preparation` | small |
| P4 | Frame consistency without dense tensors | small |
| P5 | Engine A: the declared graph, the structural checks, held variables; probe, ragged, the plain grade (`z` both routes, `r`, `f`, `r_block_sumsq`); the center check | medium |
| P6 | Engine B: the tilde grade and `g`; the bridge; the generic twin | medium |
| P7 | Engine C: the dot grade of the frame form (`Jp`, `Jp_norm_sq`, `JtJp`) | medium |
| P8 | Engine D: the raw dots and the tilde-dot grade (`Kp`), the `nu_L` family, the corewise call | medium |
| P9 | Engine E: apply and entries | medium |
| P10 | Engine F: the stacks `W`, `K`, `C`; `sum_over_samples=False`; the dtype rule | medium |
| P11 | Engine G: the uniform layout; sharding over `W` | medium |
| P12 | The backend sampling functions on the engine (merged signatures); uniform twins | large |
| P13 | The frontend sampling methods merged; `docs/chunking.md` removed | large |
| P14 | The oracle, the batch and `HeldVariables` (additive) | medium |
| P15 | The geometry's `frame(x)` returns the centers, and the consistency member | small |
| P16 | The switch: the corewise frame, `LocalModel`, `local_model`, the optimizers, the display, the frontend optimizers and the frontend local model | largest |
| P17 | The dry port of T3Polynomial | small |
| P18 | Documentation | large |
| P19 | The gate, then the deletion of the old kernels | medium |

### 5.3 The slices

**P0. Baseline.** Run `dev/graded_messages/scripts/run_all.sh` and `dev/graded_messages/scripts/review/run_review_checks.sh`
against the branch head and record drift. Write the inventory: every reference to the 166 sampling names, the 51
frontend sampling methods, the six model factories, `Problem`, `least_squares_problem`, `chunk_size`,
`sum_over_probes` and `ztildes`, in `t3toolbox/` (the package `__init__.py` included), `tests/`, `docs/`,
`examples/`, `CLAUDE.md`. The inventory is a working note in `dev/` (ephemeral). Gate: none.

**P1. `jet_contract`.** In `backend/contractions.py`: `GroupLengths` and `jet_contract` (section 2.4). The product of
two full jets is the carried shift through `xscan`, with a module-level body from a factory memoized on
`(subscripts, lens, layout)`. The prototype's `jet_contract` (`term_stacks_check.py`) is the algebra; its loop
form and its Python-loop scan are not the code. It passes `contract` only the lengths of the groups each derived
string names.
- Tests (`tests/test_jet_contract.py`): every string of the 41 against a reference that uses neither `contract` nor
  `jet_contract` (the prototype's `reference`, moved into the test), at orders 0, 1, 3, with `S`, `W`, `K`, `C`
  empty and not; a perturbation is caught; a stray axis on any operand raises; at order 0 exactly one `contract`
  call; dtype kept in float32; numpy in numpy out, jax in jax out.
- `tests/test_contractions_sharding.py` (class `TestInterpreterAnyAxisSharding`) and
  `tests/test_contractions_interpreter.py` scan source literals for grouped strings; the new strings at call sites
  are literals, so they are seen. Add a test that every string `jet_contract` derives (a literal with `T` or `A`
  deleted) is a valid `contract` string, and add `jet_contract` on the product strings to the sharding sweep (shard
  each `W` sub-axis, compile on 4 virtual devices, zero all-gathers).
- `tests/test_dispatch.py`: jit `jet_contract` on each of the three cases.
- Gate: the full gate. Deletes nothing.

**P2. The reference module.** `backend/sampling_definitions.py`: one dense function per name of section 2.3 and per
output, for the three kinds, ragged, from the column "As a subnetwork" of `SAMPLE_VARIABLES.md`; jets by
interpolation in `s`; `Kp` by jax where present, finite differences otherwise. Tilde and tilde-dot variables are
returned turned, as the engine stores them (2.4). It uses no sweep and no engine. The starting material is parts B
and C of `scripts/review/sample_variables_check.py`.
- Tests (`tests/test_sampling_definitions.py`), so that the reference itself is pinned: `z` against
  `t3_to_dense` contracted with the inputs and against `dense_probe`; jets against the Taylor expansion computed
  two ways (interpolation at two node sets); `Jp` against the dense tangent tensor (`tv_to_t3`, then dense);
  `g` against the transposed dense Jacobian; the subnetwork pairings of section 8 of `SAMPLE_VARIABLES.md` at every
  cut edge; `Kp` jax against finite differences where jax is present. Every function has a doctest.
- Gate: the full gate. Deletes nothing.

**P3. Jets and data preparation.** `backend/jets.py` (N4 to N8, and the two validators of `pp`, copied; the old ones
are deleted in P19) and `backend/data_preparation.py` (N9).
- Tests: `dual_jet_to_jet(jet_to_dual_jet(x))` equals `x` to tolerance for a nonzero weight; the pairing identity
  `sum_t omega[t]^2 a[t] b[t] = [s^n](jet_to_dual_jet(a) b)`; the derivative functions reproduce the
  derivative-mode objective (the draft's case 2); `prepare_derivative_data` with no weight returns `t!`, for apply
  and for probe with its mode axis; `equalizing_weight` on a zero block gives the floor weight; the shape of
  `data_block_norms` is the shape of the weight at orders 0 and above; `rescale_coordinates` then
  `t3_unscale_coordinates` gives back the same probe values in the original coordinates.
- Doctests: the doctest of 2.3, from `scripts/review/adapter_doctest_draft.py`, extended with the probe kind.
- `tests/test_dispatch.py`, `tests/test_jax_absent.py`: the new modules.
- Gate: the full gate.

**P4. Frame consistency through the cores** (4.2). `fv_operations.fv_frame_consistency_residual` keeps its
signature and its formula, `||T(D, L) - T(D, R)|| / max(1, ||T(D, R)||)`, and computes both norms through the
cores, never densely. The route is the one `geometry.point_norm_sq` takes today: `t3_operations` (the difference)
and `t3_orthogonalization` (a left orthogonalization, then the last core's norm). `t3_linalg` is not used: it
imports `ranks`, which imports `sharing`, which imports `fv_operations`, so it would make a cycle that
`tests/test_import_graph.py::test_the_backend_graph_is_acyclic` rejects. `T3Frame.is_consistent` is unchanged
around it; `UT3Frame.is_consistent` already goes through the cores. Add the centers' residual beside it: at each
site, the tensor with `Vc[i]` under `U[i]` (and with `Hc[i]` over `D[i]`), `L` cores to the left and `R` cores to
the right, against `T(D, L)`. Which of these the checks require is open (O-5).
- Tests: a library-built frame passes at about 1e-12; a perturbed store fails; wrong centers fail; ragged and
  uniform agree; a frame of mode sizes whose dense tensor could not be allocated (for example `d = 12`, `N = 100`)
  is checked in well under a second.
- Gate: the full gate.

**P5. Engine A.** `backend/sampling_steps.py` with the steps of the plain grade; `backend/sampling_variables.py` with
the probe kind's declared graph and `sampling_variables` for `w`, `xi`, `mu`, `nu`, `eta`, `nu_L`, `eta_D`, `z`
(through `D`, and with `Vc` read off the frame), `r`, `f`, `r_block_sumsq`. Ragged layout, one sample axis, no `K`
or `C` yet, numpy and jax. Held variables and the structural checks of section 2.3 step 1. The safe-mode center
check from P4.
- Tests by definition (`tests/test_sampling_variables.py`): every name against `sampling_definitions`, at orders 0,
  1, 3, with a random weight.
- Identities (`tests/test_sampling_identities.py`): order 0 equals slot 0 of a higher order; a held variable gives
  the result of a recomputed one; `z` through `D` equals `z` with `Vc`.
- Structural tests (`tests/test_sampling_graph.py`): the docstring table equals the declared graph row for row,
  exactly; asking for one variable computes only what the graph says it reads (count step calls with a test hook
  on the private step module); every missing argument raises; a held array of a wrong shape raises; a weight of a
  wrong shape raises.
- Safe mode: wrong centers raise in safe mode, pass in `safety.unsafe()` and under `jax.jit`, and the numbers are
  the same in both modes.
- `tests/test_dispatch.py`: jit `sampling_variables` for `('f',)`; a stray `np.` on a tracer raises.
- Gate: the full gate.

**P6. Engine B.** The tilde steps; `w_tilde`, `xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `V_tilde`, `H_tilde`,
`g`.
- Tests by definition for the new names. Identities, each with a random weight: `g` is the derivative of `f` by
  finite differences; the short form of `eta_tilde` equals a slice of the full one.
- The bridge (`tests/bridge/test_old_kernels_bridge.py`, temporary, section 5.1): `z` and `g` against
  `sampling_derivatives.t3_probe_derivatives`, the old `compute_*` kernels and `ProbeDerivativesKind.transpose`.
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
basis per edge, each sweep made once (small item 1).
- Tests by definition, `Kp` included. Identities: on a corewise call at the frame's base point `(D, L)` (not at a
  raw `x`, which differs from it by a gauge), `nu` equals the manifold call's `nu_L`, and `z` agrees with the
  manifold call's `z`; along the lift of `(V, H)` with no damping, the raw and frame forms of `nu_L_dot` and
  `eta_D_dot` are the same arrays (`subtree_gauge_check.py`). Bridge against `corewise_hessian` of the old
  derivative kinds.
- A count test: a corewise call wanting `('z', 'g')` runs each sweep once.
- Gate: the full gate.

**P9. Engine E.** Apply and entries: their graphs (the forward needs only `xi` and `mu`; the adjoint only `nu_tilde`
and `eta_tilde`; the README's "Things learned"), the entries gather, the scalar output and the weight of shape `T`.
- Tests by definition for both kinds; the identities of P5 to P8 for both; entries against apply at one-hot
  vectors to tolerance (not bitwise; 0.8). Bridge against `t3_apply_derivatives`, `t3_entries_derivatives` and
  their kinds.
- Gate: the full gate.

**P10. Engine F.** Any number of axes in `W`; the direction stack `K` on `variations` and `core_directions`; the
base-point stack `C` on every core; `sum_over_samples=False` with the result's stack in the order `W + K + C`; `f`
keeping `C`, `Jp_norm_sq` keeping `K + C`; the dtype rule.
- Tests: `W` empty, `(3,)`, `(2, 2)`, crossed with the library's stack matrix (`_CONFIGS`,
  `docs/contributor/testing_strategy.md` "The stack matrix": no stack, `C`, `K`, `K + C`, multi-axis `C = (2, 3)`
  and `K = (2, 3)`), against the reference applied element by element; the dtype rule for every output on the four
  backends (float32 in, float32 out; float32 mixed with float64 gives float32; jax mixed with numpy gives jax).
- Gate: the full gate.

**P11. Engine G: the uniform layout, and sharding over `W`.** The uniform frame and point structures; mask once on
entry, each input with the frame's own masks (section 2.3 step 2); re-mask on exit; the site group `S` on the
map-type strings (O-19 for the jetted ones); the sweeps through `xscan`, a real `jax.lax.scan` with jax; the order
scan nested inside.
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

**P12. The backend sampling functions on the engine.** The 30 changed names of section 2.4 call the engine with the
merged signatures; `sum_over_samples=True` by default; the residual argument renamed. The uniform twins check
packing and call the engine, which masks and re-masks. The frontend call sites change mechanically to the new
backend signatures and keep their present behaviour (their own `sum_over_probes=False` default is passed through)
until P13. The old kinds keep calling the old `*_derivatives`, sweep and `*_from_sweep` functions, which stay.
- Tests: the tests of the 30 names are rewritten (the ragged and uniform calls in `tests/test_uniform_probing.py`,
  the sampling parts of `tests/test_tucker_tensor_train.py`, `tests/test_manifold.py`,
  `tests/test_uniform_manifold.py`, `tests/backend/test_curvature.py` for the corewise Hessians); the uniform twins'
  exact output masks and garbage-padded inputs, through their public signatures.
- `tests/test_dispatch.py`: the rows for these names, by the new signatures.
- Gate: the full gate. From here `probing.compute_*` and the plain `assemble_*` are called only by the old kinds,
  their tests and the bridge.

**P13. The frontend sampling methods.** The four classes get the merged methods of section 2.5; the 24 methods go;
the validators come from `jets`. Every caller of a removed method is rewritten in this slice. Where the caller
still fits with the old frontend optimizers (until P16), it makes the old derivative data from the merged method as
`taylor_to_derivatives(x.probe(ww, pp, order))`. The callers, from a search of the tree at this commit:
- tests: `tests/test_tucker_tensor_train.py`, `tests/test_uniform_tucker_tensor_train.py`, `tests/test_manifold.py`,
  `tests/test_uniform_manifold.py`, `tests/test_uniform_probing.py` (51 calls), `tests/test_probe_derivatives.py`
  (folded into the frontend sampling tests), `tests/test_fitting.py`, `tests/test_jax_absent.py`,
  `tests/test_dispatch.py`, `tests/test_optimizers_frontend.py`, `tests/test_oracle_sweep.py`,
  `tests/backend/test_optimizers.py`, `tests/backend/test_padded_start.py`, `tests/backend/test_uniform_fitting.py`,
  `tests/backend/test_zero_reference.py`, `tests/backend/test_optimizer_display.py`,
  `tests/oracle_sweeps/o1_ragged.py`, `o1_uniform.py`, `sweep_models.py`, `sweep_exact.py`, `sweep_optimizers.py`,
  `oracle.py`;
- `tests/test_chunk_size_estimator.py`: its cases that call the frontend `newton_cg` with `chunk_size` and
  `probe_derivatives` (`test_newton_cg_exact_across_chunk_size` and its kin) are removed here, as Part 1 drops
  `chunk_size`; its estimator cases stay until P19;
- module doctests: `t3toolbox/optimizers.py` (`least_squares_problem`'s and `newton_cg`'s examples),
  `t3toolbox/fitting.py` (the derivative factories' examples), and the methods' own;
- pages: `docs/entries_apply_probe.md`, `docs/numerical_contracts.md`, `docs/fitting_and_optimization.md`;
- `docs/chunking.md`: removed here with its toctree entry in `docs/design_notes.rst` and every link to it, because
  its doctests call `probe_derivatives_transpose(..., chunk_size=...)` and Part 1 drops the frontend `chunk_size`;
- the six examples that call a `*_derivatives` name.
- Tests: the doctests on the merged methods (`order=0` returns today's shapes; `order=2` adds `T` outermost).
- Gate: the full gate.

**P14. The oracle and the batch** (additive). `LeastSquaresOracle`, `LeastSquaresBatch`, `HeldVariables` in
`backend/fitting.py` beside the old kinds; the `Oracle` protocol as documentation in `backend/optimizers.py`;
`n_measurements`, `take`, `y_block_sumsq`, `f_resolution`, the minibatch scale.
- Tests: each output of `query` against the engine and against the reference; `held` holds exactly the names of the
  hold list that the call computed (none computed only to be held); `take` under `jax.jit` with a static size;
  `minibatch_scale` makes the drawn `f` an unbiased estimate of the full `f`, tested exactly on a small `n` by
  enumerating every subset of the batch size; the pytree round trip of the batch and of `HeldVariables`; the
  oracle's hash and equality by value; bridge against the old kinds through the conversions.
- Gate: the full gate.

**P15. The geometry returns the centers.** `frame(x)` returns `(frame, centers)` on the four backend geometries
(the manifold ones keep the second result of `t3_orthogonal_representations`, which they compute today and discard;
the corewise ones return `None` as centers, and still the four-store frame `(B, G, G, G)` until P16); the
consistency member (N21). The old path adapts in place: `_local_model_phases`, the zero reference, the padded start
and the frontend `t3toolbox/fitting.py` read `frame, _ = geom.frame(x)`. `shared_geometry.SharedGeometry` follows
where it calls the backend. The form of the result is open in part (O-4).
- Tests: `tests/backend/test_geometry.py` for the new result; the centers equal the second result of
  `t3_orthogonal_representations`; the consistency member on library-built and on perturbed frames.
- Gate: the full gate.

**P16. The switch.** One commit, reviewed in four parts. It cannot be split further without transitional code: the
backend optimizers keep their names (`newton_cg` and the rest) and change their arguments, the frontend optimizers
and the frontend local model call them, and CI runs every example and doctest page. Keeping the old backend model
alive beside the new one for the frontend model does not work either, because the corewise frame changes in this
slice and the old kinds read a four-store frame. A split with a temporary translation of the old frontend call form
is O-24. O-9 (the frontend fitting surface) must be answered before this slice.
1. *Geometry.* The corewise frame is the point `(B, G)` on the ragged and uniform corewise geometries (`frame`,
   `base_point`, `retract`, `point_tangent`, the shared corewise post-pass).
2. *Backend model and optimizers.* `LocalModel` (section 2.2); `local_model` builds the frame, checks consistency
   in safe mode once, and makes one `query` wanting `f` (and `g` in exact mode); `gradient` asks for `g` with
   `held`; `gn_quadratic` for `Jp_norm_sq`; `hvp` for `JtJp`, and in exact mode `JtJp` and `Kp` in one request;
   `jacobian(p)` for `Jp`; `residual` for `r`. Exact mode checks, before any work, that the oracle supplies `Kp`
   (O-7). The five phases as module-level functions of pytrees; `use_jit` compiles the CG solve and the stochastic
   step, as today; a compiled program's preconditions are checked on the host first (8.3). Trial points use policy
   a: `point_objective` at the trial's raw cores. The minibatch is a batch from `draw(rng)`; the regularizer is not
   scaled; MC-SGD and Adam on an oracle without a draw raise (8.3 part 2). The zero reference, the padded start, the
   cone reference and the switch mode go through `query`.
3. *Display and record.* The display reads `r_block_sumsq` at the model's frame through `held`, `y_block_sumsq`
   once, and `f_resolution`; without `f_resolution` the trust-region floor is `eps |f|` with `eps` from the dtype of
   `f` (8.4); an oracle without block sums gets the header line and no table. The stored record (the display's
   records, which the frontend returns as `stats['diagnostics']`) keeps `train_err` and `val_err` at every
   iteration in the full shape of the weight, nothing reduced over modes or orders (8.4).
4. *Frontend.* `t3toolbox/optimizers.py` takes `(geometry, oracle, batch, x0, ...)`; it maps the geometry to the
   backend one, reduces `x0` to minimal ranks and packs the batch for a uniform `x0` (as `least_squares_problem` does
   today), and rewraps the result. `t3toolbox/fitting.py` is rebuilt over the backend `LocalModel`, with tangents in
   and out and the same-frame guard, in the form O-9 settles; the six factories go, with their re-exports from
   `t3toolbox/__init__.py`. Every example and doctest page that calls an optimizer or a factory is rewritten.
- Tests rewritten: `tests/backend/test_optimizers.py`, `tests/backend/test_optimizer_display.py`,
  `tests/backend/test_geometry.py`, `tests/backend/test_zero_reference.py`, `tests/backend/test_padded_start.py`,
  `tests/backend/test_uniform_fitting.py`, the model parts of `tests/backend/test_curvature.py` and
  `tests/backend/test_uniform_curvature.py`, `tests/test_fitting.py`, `tests/test_optimizers_frontend.py`,
  `tests/test_sharing.py`'s fitting cases, `tests/test_oracle_sweep.py` and `tests/oracle_sweeps/`,
  `tests/test_jax_absent.py` (its `bopt.least_squares_problem` and `probe_derivatives_model` script),
  `tests/test_aligned_padded_frame_frontend.py` (`topt.least_squares_problem`), and the fitting rows of
  `tests/test_dispatch.py` (a model rebuilt each step is one cache key; a new weight or minibatch does not
  recompile; `want` is a literal). Optimizers are judged by outcomes on problems with known answers and by the
  model's consistency (`p` paired with `hvp(p)` equals `gn_quadratic(p)` in Gauss-Newton mode), not by old
  iteration counts (9.3). New: the stored record's shapes at orders 0 and 2 for each kind, with and without a
  validation batch; the floor `eps |f|` on a user oracle without `f_resolution`; a user oracle without a draw
  raises in `mc_sgd`.
- Pages and examples: the 13 examples that call an optimizer or a factory; `docs/fitting_and_optimization.md`,
  `docs/getting_started.rst`, `docs/sharing.md`, `docs/numerical_contracts.md`, `docs/rank_continuation.md`,
  `docs/aligned_padded_frame.md` where their doctests call them.
- The slow tier (`T3TOOLBOX_SLOW_TESTS=1`, serially) is part of this slice's gate.
- From here nothing in the library calls the old kinds, `Problem`, or the old sampling kernels except their own
  tests and the bridge.

**P17. The dry port.** Section 9.

**P18. Documentation.** Section 8. Two commits: the new material (the page, the example, the note, the dependency
diagrams, the reference module's docs), then the rewrites and removals of existing pages.

**P19. The gate, then the deletion.** Section 10.

### 5.4 When each old piece stops being called

| Old piece | Last caller in the library | Stops at | Deleted at |
| --- | --- | --- | --- |
| `probing.compute_*`, `assemble_*`, `compute_sigma_hat` (plain) | the old plain kinds | P16 | P19 |
| `sampling_derivatives` (all) | the old derivative kinds; the frontend `*_derivatives` methods | P13 (frontend), P16 (kinds) | P19 |
| `*_precompute_*_frame_sweep*`, `*_from_sweep` | the old kinds | P16 | P19 |
| `ut3_sampling` and `utv_sampling` removed names | the uniform kinds; the uniform frontend `*_derivatives` | P13, P16 | P19 |
| The old kinds (`backend/fitting.py`, `backend/uniform_fitting.py`) | `Problem`; the frontend factories | P16 | P19 |
| `Problem`, `least_squares_problem` (both layers), `LeastSquaresSetup`, the six frontend factories | the optimizers; users | P16 | P16 |
| The frontend `*_derivatives` methods | users | P13 | P13 |
| `chunk_size` (frontend), `docs/chunking.md` | the frontend optimizers | P13 (the page), P16 (the keyword) | P13, P16 |
| `estimate_chunk_size`, `max_chunk_size_within`, `DEFAULT_CHUNK_SIZE` | the frontend optimizers | P16 | P19 |
| `TestJetKernelsShardOverW` | itself (it imports `sampling_derivatives`) | replaced by P11's class | P19 |
| The bridge test | itself | P19 | P19 |

`Problem`, the factories and the frontend methods are removed when they switch because they are interfaces, not
kernels.

---

## 6. The tests

| Layer | What | Where | When |
| --- | --- | --- | --- |
| Reference | `sampling_definitions` pinned against the dense tensor, the dense tangent tensor, the dense Jacobian and the Hessian | `tests/test_sampling_definitions.py` | P2 |
| By definition (9.3) | every name, every kind, orders 0, 1, 3, a random weight, against the reference | `tests/test_sampling_variables.py` | P5 to P11 |
| Identities (9.3) | the adjoint identity of `Jp` and `g`; `p` with `JtJp` equals `Jp_norm_sq`; `g` is the derivative of `f`; held equals recomputed; independence of the layout and the backend; order 0 equals slot 0; `Vc` route equals `D` route; short forms are slices | `tests/test_sampling_identities.py` | P5 to P11 |
| Generic twin | the written-out terms against terms derived mechanically | `tests/_generic_twin.py`, used by the identity tests | P6 |
| Structural | the table equals the declared graph; one wanted variable computes only its reads; missing arguments raise; held shapes; weight shapes; the dtype rule; uniform output masks | `tests/test_sampling_graph.py`, the uniform tests | P5, P10, P11, P12 |
| Uniform | equivalence with ragged on real parts over the full `_CONFIGS` matrix; exact output masks; garbage-padded inputs; tails; the two-hole masks on a TT-slack frame, with a mutation check | `tests/test_sampling_variables.py` (uniform cases), `tests/test_uniform_probing.py` | P11, P12 |
| Sharding | the AST sweep over every literal string; `jet_contract`'s product strings; `g` and `Kp` under `shard_map` with `W` sharded, zero all-gathers, equal to unsharded | `tests/test_contractions_sharding.py` | P1, P11 |
| Dispatch | jit each new public function; numpy in numpy out, jax in jax out; eager calls compile once | `tests/test_dispatch.py` | every slice |
| Doctests | every new public function; the page; the 2.3 doctest | modules, `docs/` | P2, P3, P5, P18 |
| Fitting | outcomes on known answers; model consistency; compile keys; the stored record's shapes; the `eps |f|` floor; a draw-less oracle in MC-SGD | the rewritten fitting tests | P14, P16 |
| Bridge (temporary) | new outputs against old code that is never rerouted (`sampling_derivatives`, the old kernels, the old kinds), through the stated conversions | `tests/bridge/test_old_kernels_bridge.py` | P6 to P14; deleted in P19 |

**Tolerances.** Numerical comparisons are to a tolerance scaled to the dtype; structural ones are exact (0.8, 9.3).

**Existing tests, sorted.**

| File | Fate |
| --- | --- |
| `tests/test_jet_recurrence.py`, `tests/_jet_trs_twins.py`, `tests/backend/test_from_sweep_reuse.py` | Test old kernels only; deleted in P19 |
| `tests/test_chunk_size_estimator.py` | Its frontend `newton_cg` cases (with `chunk_size` and `probe_derivatives`) removed in P13; its estimator cases deleted in P19 |
| `tests/test_contractions_sharding.py` | Extended in P1 and P11; its class `TestJetKernelsShardOverW` (imports `sampling_derivatives`) deleted in P19, replaced by P11's |
| `tests/test_probe_derivatives.py` | Folded into the frontend sampling tests in P13 |
| `tests/test_uniform_probing.py`; the sampling parts of `test_tucker_tensor_train.py`, `test_manifold.py`, `test_uniform_manifold.py`, `test_uniform_tucker_tensor_train.py` | Rewritten in P12 and P13 |
| `tests/backend/test_curvature.py`, `tests/backend/_curvature_oracles.py`, `tests/backend/test_uniform_curvature.py` | The sampling half rewritten in P12, the model half in P16; `_curvature_oracles.py`'s dense oracles kept as test helpers |
| `tests/test_fitting.py`, `tests/backend/test_uniform_fitting.py` | Their `*_derivatives` calls in P13; rewritten in P16 |
| `tests/backend/test_optimizers.py`, `test_optimizer_display.py`, `test_geometry.py`, `test_zero_reference.py`, `test_padded_start.py`, `tests/test_optimizers_frontend.py`, `tests/test_sharing.py` (fitting), `tests/test_oracle_sweep.py`, `tests/oracle_sweeps/`, `tests/test_jax_absent.py`, `tests/test_aligned_padded_frame_frontend.py` | Their `*_derivatives` calls in P13; rewritten in P16 (`test_geometry.py` also in P15) |
| `tests/test_dispatch.py` | Rows added from P1; old rows replaced as each surface switches |
| `tests/test_contractions_interpreter.py` | Extended in P1 |
| `tests/test_import_graph.py`, `tests/test_jax_absent.py` | New modules registered as they land |
| `tests/test_frame_variations_format.py`, `tests/test_uniform_frame_variations_format.py` | `is_consistent` cases in P4 |

---

## 7. The areas that have been tricky in the past

### 7.1 Safe mode against unsafe mode

- **The checks in Part 1.** Numerical, safe mode only, skipped in `safety.unsafe()` and under a jax trace: the
  centers' agreement in the engine when `Vc` is given; frame consistency once when a `LocalModel` is built (small
  item 2); the preconditions of a compiled program, checked on the host before it is called (8.3). Structural,
  always: everything in step 1 of section 2.3, and the frontend's same-frame guard.
- **No dense tensor** in any runtime check (4.2): the residuals of P4 go through the cores.
- **Where the safe-mode gate lives.** `t3toolbox/safety.py` is a frontend module and the backend has never imported
  it (CLAUDE.md: "Frontend-only for now"). The engine and the backend `LocalModel` now need it. `safety.py` imports
  only `backend.common`, so importing it from the backend makes no cycle, as `backend/optimizers.py` already imports
  the frontend `corewise`. This is open point O-6.
- **Tests.** For every check: it raises in safe mode; it is silent in `safety.unsafe()` and under `jax.jit`; the
  numbers are identical in both modes; the check forms no ambient-sized array.

### 7.2 The ragged against the uniform layout

- One set of step functions; the layout from the structure of the inputs (layout first, then the count); sweeps
  through `xscan`; map-type operations a loop on ragged and all sites at once (the group `S`) on uniform, as "The
  two parts" fixes 5.3. For the jetted maps that is not today's behaviour (O-19).
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
  `_prepare_jit_inputs`).

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

- Everything a raw-`.data` user needs is backend: the engine, the oracle and the batch, `local_model`, the
  optimizers, the display, the helpers of `jets` and `data_preparation`, the reference module.
- The frontend validates what only it has: the shapes of `ww` and `pp` against its `x` (the validators from
  `jets`), and the same-frame guard. The weight's exact shape is checked in the engine, where both tiers meet
  (section 2.3 step 1; its placement is O-18). The frontend holds no logic a backend user would have to rewrite.
- Corollary: a backend user who gets `g` from the engine gets blocks before projection; the geometry's `project` is
  the one documented follow-up, and `local_model` does it for them.
- **Test.** A backend-only fit in a doctest: build the batch, the oracle, call `newton_cg` from
  `backend.optimizers`, read the result, with no frontend object.

### 7.6 Sharding

- Part 1 promises nothing new about sharding (7.7 is open). It must not break what holds today: every sub-axis of
  every group is shardable with no all-gather, because `contract` never reshapes.
- `jet_contract` reshapes nothing either: the order scan slices the order axis, never `W`.
- Every string at a call site is a literal, so the AST scan of `tests/test_contractions_sharding.py` sees it.
- The arrays of a batch are top-level arguments of the compiled phases, not closures (8.3 note).
- What holds today beyond the per-string sweep: the jetted transpose and the two-hole, run under `shard_map` with
  `W` sharded, compile with no all-gather and match the unsharded result (`tests/test_contractions_sharding.py`,
  class `TestJetKernelsShardOverW`). P11 pins the same property on the engine; the old class goes with
  `sampling_derivatives` in P19.
- **Tests.** The sharding sweep on the new literals and on `jet_contract`'s product strings (P1); `g` and `Kp` from
  the engine at orders 0 and 3, ragged and uniform, under `shard_map` with `W` sharded (P11).

### 7.7 Compiling with jit

- **Statics and leaves.** Static (`aux_data`, value-hashed): the geometry, the oracle (`kind`, `order`), the
  regularizer, `want` (a literal tuple at every library call site), the names of `HeldVariables`, uniform masks and
  shapes, `GroupLengths`. Leaves: the batch (the weight, the minibatch, its scale, its indices), the frame, the
  centers, the held arrays, `f`, the curvature state.
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
| The dependency table and diagrams | The table of reads, and the diagrams that show it | Their source is the declared graph (9.7). How the diagrams are drawn is open (O-23); the proposal is a script (N29) that generates, from the graph, one layered picture per grade and one per output, as SVG files committed beside the page, run by hand when the graph changes, with a test that every name has a diagram and that the committed files match a fresh run. Every diagram shows its arrows (9.5) |
| The bad-scaling example | `examples/fit_badly_scaled_data.py`: bad as given, good after the preparation functions (6.7) | From `scripts/review/bad_scaling_example.py`; CI runs it |
| The derivative-data doctest | In `backend/jets.py` (2.3) | From `scripts/review/adapter_doctest_draft.py` |

**Rewritten.** `docs/entries_apply_probe.md`, `docs/transposes.md` (sum by default; `residual`),
`docs/fitting_and_optimization.md` (the oracle, the batch, the new signatures; the weight's exact shape; ones by
default with the asterisk of 2.5; derivative data prepared outside), `docs/batching_and_stacking.md` (`T` and `A`,
the sample stack on the outputs), `docs/grouped_contractions.md` (`jet_contract`), `docs/frame_variations.md`
(centers), `docs/numerical_contracts.md` and `docs/contributor/numerical_contract_catalog.md` (the new
preconditions), `docs/sharing.md`, `docs/rank_continuation.md` and `docs/aligned_padded_frame.md` where they call
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
are rewritten in the slice that changes it (section 5); P18 rewrites the prose.

**Removed.** `docs/chunking.md`, in P13 (Part 2 writes the sizing page); `docs/symmetric_probe_derivatives.tex` if its
content moves into the new note (open point O-15).

**Records.** `docs/glossary.md` (the variables, the oracle, the batch, held, centers; the old names removed);
`docs/naming_conventions.md` (the sampling modules, the new modules, the corewise-call rule) and
`docs/contributor/naming_rules.md`; `docs/contributor/deferred_and_rejected.md` (the "kwarg-unified derivative
interface" entry reversed, 6.6 note); `dev/OPEN_QUESTION_extension_surface.md` given its resolution (3.4) and
archived; `CLAUDE.md` (the paper-to-code map, the sampling section, the shipped surface, the module list);
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
| `_newton_fit` | `topt.newton_cg(geometry, 'probe_derivatives', (ww, pp), data, X0, order=jet_order, weight=weight, **kwargs)` | P16 |
| `fit_rank_continuation` and `fit_rank_continuation_adaptive`, scoring each level | `X.probe_derivatives(ww, pp, jet_order)`, then the consumer's own `weighted_misfit` and `forward_summed_reverse_misfit` | P13 |
| `polynomial_predict` | `T.probe(ww)[-1]` | unchanged (`order=0`) |

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
is not read by T3Polynomial (O-17 changes its keys).

**What it checks.** On T3Polynomial's toy problem (a known polynomial, `t3polynomial/toy.py`), two levels of
`fit_rank_continuation` and of `fit_rank_continuation_adaptive`, on each backend (`'ragged'`, `'uniform'`,
`'uniform_jax'`): the fit reaches the known answer to the validation error the release reaches; the display's table
appears; every `newton_cg` option T3Polynomial passes is accepted. The script's output is recorded in the folder
like the other checks.

**What it is for at the gate.** With both replacements installed, T3Polynomial's own benchmark runs unchanged on
the new library (section 10).

---

## 10. The last step: the gate, then the deletion

**The gate** (small item 3). All of:

1. The tests by definition and the identities pass, on both CI generations, and the slow tier passes serially.
2. The full suite on the jax-0.4.30 / numpy-1.26 floor, as at a release.
3. The consumer's benchmark, chunking off, new against old: the old is the release `v2026.5.0` with T3Polynomial
   as it is; the new is the branch with the dry port's two replacements. Run by the maintainer on CPU and on a GPU
   pod. The candidate is `scripts/b01_backend_bench.py`: it drives `fit_rank_continuation_adaptive` at the Darcy
   sizes on four arms and passes `chunk_size=None` on all of them. Chunking off is not the path T3Polynomial's
   fitting module takes by default (it inherits `chunk_size='auto'`, section 1). Which benchmark, and what result
   passes, are open (O-1, O-2).

**The deletion**, one commit after the gate:

- `backend/sampling_derivatives.py`, whole.
- In `backend/probing.py`, `apply.py`, `entries.py`, `ut3_sampling.py`, `utv_sampling.py`: the 74 removed names of
  section 2.4 that are not in `sampling_derivatives`.
- The old kinds and their factories (`backend/fitting.py`, `backend/uniform_fitting.py`).
- The tests of old kernels (section 6), the bridge test, the class `TestJetKernelsShardOverW`.
- `__all__` lists, `docs/api_reference.rst`, the import-graph pins, the inventory of P0 re-run to zero references.

Its own gate: the full gate, the slow tier, `sphinx -W`, and a grep over `t3toolbox/`, `tests/`, `docs/` and
`examples/` (not `dev/`, `CHANGELOG.md` or `docs/release_notes.md`, which name the removed names on purpose) that finds
none of the 129 removed names, nor `chunk_size`; `sum_over_probes` and `ztildes` remain only on the three ambient
transposes, which stay as they are (6.3).

---

## 11. Open points for the maintainer

None of these is settled by this plan. Where the plan needs an answer to proceed, it states a proposal and marks
it.

- **O-1. Which benchmark is the gate.** The decisions say "the consumer's benchmark". Proposed:
  `t3polynomial/scripts/b01_backend_bench.py`. It passes `chunk_size=None`; T3Polynomial's fitting module by default
  does not, so it chunks on its uniform backends today (`chunk_size='auto'`). The gate thus compares chunking-off
  runs, not the module's default path. ("The cost" in "The two parts" says the fitting module defaults to
  `chunk_size=None`; the code does not bear that out: `_level_newton_kwargs` returns `{}` for `newton_kwargs=None`,
  and `_newton_fit` passes no `chunk_size`.)
- **O-2. What result passes the gate.** Time and peak memory, new against old: equal within a tolerance, or no worse
  than a stated factor? And the order-0 condition of 2.7 ("only if order 0 loses no performance"): measured how?
- **O-3. The uniform surface of the engine and the twins.** Proposed: the engine takes the uniform frame `.data`
  (with shape and masks) or the uniform point `.data`, masks once on entry and re-masks core-shaped outputs on exit;
  the `ut3_`/`utv_` twins stay as thin wrappers (check packing, call). 6.3 left the uniform twins to group D, and 5.2 does not say
  whose entry "masked once on entry" is.
- **O-4. The form of the geometry's `frame` with centers.** That `frame(x)` returns the centers is decided (4.1 and
  the 8.3 note); its form is not. Proposed: `(frame, centers)` with `centers = (Vc, Hc)` or `None`, on the backend
  protocol only. The frontend geometries keep
  returning a `T3Frame` / `UT3Frame` (the frame classes are off limits, 1.1). Also: does the frontend `COREWISE`
  keep its `T3Frame` `(B, G, G, G)` for corewise tangents, with the frontend passing `(B, G)` to the backend?
  Proposed: yes.
- **O-5. What frame consistency checks.** `L` against `R` through `D`, always; `U` can be compared only through the
  centers; given centers against `(D, L)`. Which of these does the `LocalModel` build check when the geometry
  returned centers, and which when a frame was given (`start_frame`)?
- **O-6. The backend imports `safety`.** The engine's center check and the model's consistency check need the
  safe-mode gate in the backend. Proposed: the backend imports `t3toolbox.safety`, and the pinned imports of
  `backend/optimizers.py` gain it (or the check sits in the geometry member and the oracle, so that
  `optimizers.py` keeps its three imports).
- **O-7. How an oracle says which optional outputs it supplies** (carried forward under 8.3's alternative A).
  Proposed: a class attribute `optional_outputs`, read before any work, as `corewise_hessian`'s marker is read
  today.
- **O-8. The form of `held`, and how the protocol types it.** The engine takes held variables as keywords; the
  oracle takes and returns `held`. A dict is out (0.5), and "what goes into `held` is the oracle's business" (8.3).
  Proposed: the protocol types `held` as any pytree the optimizers pass back unread; the least-squares oracle's is
  `HeldVariables`, a frozen pytree with static names and array leaves, expanded to the engine's keywords.
- **O-9. The frontend fitting surface.** Needed before P16. The six model factories and `least_squares_problem` /
  `LeastSquaresSetup` have no row; the factories take a residual, which the new model no longer does. Proposed: one frontend `fitting.local_model(geometry, oracle, batch, x, regularizer=None,
  hessian=..., curvature_damping=None)` over the backend `LocalModel`, with tangents in and out; the factories,
  `least_squares_problem` and `LeastSquaresSetup` removed. Eight of T3Polynomial's scripts call `least_squares_problem`
  (among them `scripts/d07_curvature_dispersion.py`, `d08_profile_ops.py`, `d11_profile_ops.py`, `g06_aligned_frame.py`).
- **O-10. How `jet_contract` learns the layout.** A product of two full jets is a real scan only with jax on the
  uniform layout, and one core's arrays look the same on both layouts. Proposed: a sixth field of `GroupLengths`;
  5.4 names five.
- **O-11. Which inputs count for the dtype rule, and where the cast is done** (5.6 open detail). Proposed in
  section 2.3 step 3.
- **O-12. The public meaning of `nu_L_dot` and `eta_D_dot` in a call with both directions.** Proposed: the raw
  form (the one `Kp` reads); the frame form used for `Jp` is internal in such a call. Also the case of curvature
  damping above zero (`SAMPLE_VARIABLES.md` section 11).
- **O-13. Spellings on a corewise call.** Whether such a call also accepts the names with a basis (`nu_L` for its
  `nu`). Proposed: no; one name per variable on each call type (`SAMPLE_VARIABLES.md` section 11).
- **O-14. The two estimates read off the data** (a length for the directions, a coordinate scale). Not shipped by
  this plan; the bad-scaling example computes them inline. Ship them?
- **O-15. The fate of `docs/symmetric_probe_derivatives.tex`** and the numbering of the paper-to-code map in
  `CLAUDE.md`, which the new names touch.
- **O-16. Frontend access to the preparation functions.** They are backend functions on arrays. Re-export them from a
  frontend module, or document the backend import?
- **O-17. The timing laps of `newton_cg(timing=True)`.** Today's build laps are `frame`, `sweep`, `companion`,
  `residual`, `gradient`, `curvature`. With one oracle call, `sweep` and `residual` become one lap. Proposed:
  `oracle`. This changes `stats['timing']` keys.
- **O-18. Where the weight's shape check lives.** The rule is decided (2.5: the exact shape). Its placement is not:
  today the frontend owns the weight contract (`fitting._canonical_weight`, which accepts bare vectors and
  broadcasts). Proposed: in the engine, so both tiers get it; `_canonical_weight` goes.
- **O-19. The jetted map-type operations on the uniform layout.** "The two parts" fixes 5.3 for Part 1 as map-type
  operations "at all sites at once on the uniform one", as the library does today. That is true of today's plain
  kernels (`compute_eta`, `assemble_z`: one contraction with a leading `d`). It is not true of today's jet kernels:
  they map one site at a time through `xmap`, which `get_backend(True, True)` makes `jax.lax.map`
  (`backend/common.py`, `get_backend`; `backend/sampling_derivatives.py` lines 504, 1259, 1440, 1504, 1620 and
  others). Vectorizing those maps over the sites makes their intermediates `d` times larger on the uniform layout,
  beside the memory that removing the chunked assemblies costs. Does Part 1 vectorize the jetted maps, or keep
  `jax.lax.map` for them as today? The plan follows "The two parts" until this is answered.
- **O-20. `dual_jet_to_jet` where a weight is zero.** It divides by the weight squared (3.6 note, not answered).
  Raise in safe mode (a numerical check, skipped under a trace); return `inf`/`nan`; or document it as the caller's
  business?
- **O-21. What the minibatch scale multiplies.** 8.3 part 2 scales the drawn oracle's `f`, `g` "and the rest" by
  `n/batch` so that `f` estimates the full one. Do `r_block_sumsq` (and so the display's table on a drawn batch),
  `r` and `Jp` get the scale too? The plan scales `f`, `g`, `JtJp`, `Jp_norm_sq`, `Kp` and nothing else until
  answered.
- **O-22. The seed convention of the backend transposes and of the corewise Hessian.** 2.4 settles that the frontend
  transposes take a residual of the type of `z` and turn it; 3.6 that the bare transpose takes no weight. Today's
  backend corewise Hessian takes `ztildes`, documented as "residuals (omega^2 applied)" (`backend/probing.py`,
  `t3_probe_corewise_hessian`), with the order `(ztildes, ww, core_pair, core_directions)`. Proposed: every backend
  function behind a transpose and the corewise Hessian take a `residual`, no weight, turn it inside, and keep today's
  argument order (section 2.4). The alternative is a seed already turned.
- **O-23. How the dependency diagrams are drawn** (9.7: "How the diagrams are drawn is open"). Proposed in section 8:
  generated from the declared graph by a script, one layered picture per grade and one per output, committed as
  files, with a test that they match a fresh run. Each shows its arrows (9.5).
- **O-24. Splitting P16.** P16 is the largest commit of the arc. It can be split only with transitional code: the
  frontend optimizers accept the old call form `(geometry, kind, sample, data, x0, order=, weight=)` for one slice
  and translate it (with `prepare_derivative_data` for the derivative kinds), so that the examples and pages move in
  a following slice that also deletes the translation. One commit (no transitional code), or two?
- **O-25. The reading of "the old library is a bridge used once during the port"** (9.3). The plan reads it as a
  temporary test file compared against code never rerouted onto the engine, run in CI from P6 and deleted in P19.
  The other reading is a one-off script run once per surface, never in CI.
