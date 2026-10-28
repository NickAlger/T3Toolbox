# Release notes

The full changelog is reproduced below, newest first; the format follows
[Keep a Changelog](https://keepachangelog.com/) and versions are `YYYY.MINOR.PATCH` (a calendar year,
then a minor number that increments with each release of that year). Start with the upgrade notes if
you are moving an existing codebase forward.

## Upgrading from 2026.2.0

Most code runs unchanged; the exact Hessian is opt-in (`newton_cg(..., hessian='exact')`, Gauss-Newton stays
the default). What can break, roughly by how likely you are to meet it:

**The default Newton reference moved** (manifold geometries). `newton_cg`'s relative stop and CG forcing
term, and `gradient_descent`'s stop, are now relative to the gradient norm at the **zero tensor of `x0`'s
ranks** (`g0_reference=None`, resolved to `'zero'`), not the warm-start `‖g(x0)‖`. Every manifold run stops
at a slightly different point -- an O(1) factor in the reference, zero starts included. To keep the
2026.2.0 behaviour pass `g0_reference='warm_start'`; the explicit `g0norm_newton` / `g0norm_cg` overrides
still win (and must now be `> 0`: a non-positive value raises `ValueError`). Corewise geometries keep the
warm start.

**Delete your reference bookkeeping.** A continuation loop that carried `stats['history'][0]['gnorm']` from
one level into the next level's `g0norm_newton` can drop it: the default reference is already per level
and at the level's own scale. If a loop passed `g0norm_newton=None` *meaning* "use the warm start", that now
gets the zero reference -- say `g0_reference='warm_start'` instead. A **cold start** at zero
(`TuckerTensorTrain.zeros(...)` as `x0`) remains available at every level, in either Hessian mode; its
default reference is the same as the warm start's at those ranks.

**`stats['newton']` is gone; read `stats['newton_steps']`** (the same count: CG solves). Beside it are
`stats['newton_accepted']` (steps taken) and `stats['stop_reason']`.

**`fitting.GaussNewtonModel` is now `fitting.LocalModel`** (and `t3toolbox.LocalModel` at the root).
It is renamed, not aliased, so the old name raises `AttributeError` / `ImportError`; rename it where you
spell it. The six `*_model` factories, the model's methods and its fields are unchanged, so code that
only calls the factories needs no change. The new name mirrors the backend class it wraps,
`backend.optimizers.LocalModel`.

**`optimizers._setup` is gone; use `optimizers.least_squares_problem`.** It takes the same arguments
(`geometry, kind, sample, data, x0, order, weight, regularizer, chunk_size, batch`, plus `hessian`) and
returns a named `LeastSquaresSetup` instead of a 3-tuple, so read its fields rather than unpacking:

```python
setup = optimizers.least_squares_problem(geometry, kind, sample, data, x0, order=order, weight=weight)
lm = setup.problem.local_model(setup.x0_cores)       # was: problem, init, rewrap = optimizers._setup(...)
```

`setup.problem.geom` is the backend geometry that the private `optimizers._geometry_ops(...)` used to
give you; `_geometry_ops` stays private.

**`optimizers.newton_cg` takes explicit keywords only.** Every valid call is unchanged; a misspelt option
now raises `TypeError` at the call instead of failing later inside the backend.

**Custom kinds and geometries keep working for Gauss-Newton.** To use them with `hessian='exact'`, a
`SamplingKind` adds the sixth operation `corewise_hessian` (the derivative of its `transpose` along raw
core directions), and a geometry the four exact-mode members `curvature_precompute`, `lift`,
`lift_transpose` and `tangential_curvature` (the `Geometry` protocol in `backend/optimizers.py`); without
them an exact model raises a structural error naming what is missing. A geometry **may** also add
`zero_point_frame` (optional, looked up duck-typed): without it the default reference is the warm start, as
in 2026.2.0.

**Backend callers of the jet kernels** (`backend.sampling_derivatives`): the `trs` arguments are gone (the
order is read off the jets, or passed as `order`), `compute_dxi_tilde_jets` returns only its two read rows,
and `binomial_combine_tensor` and the dense `*_trs` kernels are removed. Code that only calls the
transposes, the forwards or the optimizers is unaffected.

**Uniform transposes return exact zeros in their padding.** Results that respect "padding is don't-care"
are bit-identical; only code that read the padded slots sees a difference.

## Upgrading from 2026.1.0

The optimization layer's axis objects became classes. Everything here raises rather than failing
silently, and the constructor spellings you already use mostly still work.

**Geometries moved and are constructed, not fetched.** `backend.optimizers.MANIFOLD_OPS` becomes
`backend.geometry.ManifoldGeometryOps()`, `COREWISE_OPS` becomes `CorewiseGeometryOps()`, and
`shared_geometry_ops(base, groups)` becomes `base.with_sharing(sharing, shape)` (ragged; the uniform twins take
`with_sharing(sharing)` -- they carry their shape already). On the uniform side,
`uniform_geometry_ops(name, x0_data, sharing)` becomes
`UniformManifoldGeometryOps.from_point(x0_data, sharing)` (or the `Corewise` twin). The **frontend**
geometry singletons -- `MANIFOLD`, `COREWISE`, `UNIFORM_MANIFOLD`, `UNIFORM_COREWISE`,
`shared_manifold(...)` -- are untouched, so ordinary fitting code needs no change.

**A custom sampling kind is a subclass now.** Building one by copying an existing kind and swapping a
function -- `dataclasses.replace(APPLY, forward=...)` -- no longer type-checks. This is deliberate: that
construction copied the kind's identity along with it, so the variant claimed to *be* `APPLY` and `jit`
handed back `APPLY`'s compiled program. Subclass `ApplyKind` (or the relevant base) and override the
operations you need; keep your parameters as dataclass **fields** and you get correct jit cache
behaviour for free.

**A custom backend geometry needs three more members.** The 2026.0.0 notes below show a geometry
object with `frame` / `project` / `retract` / `inner` / `precompute = None`. The optimizers now also call
`stack_shape(x_cores)` (returns the frame stack `C`, used by the stacked-point guard) and
`base_point(frame)` (the point the frame is attached to, used by the regularizer), and `precompute` must
be a **callable** returning `None` rather than `None` itself. Adding the three reproduces the old
behaviour exactly. The full surface is the `Geometry` protocol in `backend/optimizers.py`.

**The six uniform kind builders are gone.** `uniform_apply_kind(x0_data)` and its five siblings become
`UniformApplyKind.from_point(x0_data)` and theirs; the `uniform_sampling_kind(name, x0_data, weight)` /
`uniform_derivatives_kind(name, x0_data, order, weight, chunk_size)` dispatchers are unchanged and are
the easier route. The four **ragged** constructors (`probe_kind`, the three `*_derivatives_kind`) keep
both their positional and their `weight=` keyword spellings.

**`fitting.UniformGaussNewtonModel` is gone.** One `GaussNewtonModel` (renamed `LocalModel` in 2026.3.0,
above) now serves both representations,
and the `*_model` factories still dispatch on `x`, so `fitting.apply_model(UNIFORM_MANIFOLD, ux, ...)`
returns what it always did -- just of the merged type. If you were checking
`isinstance(m, UniformGaussNewtonModel)`, check the frame instead:
`isinstance(m.frame, t3toolbox.UT3Frame)`.

**Stacked points raise in the optimizers.** If you were passing a stacked `x0` to `newton_cg`,
`gradient_descent`, `mc_sgd` or `adam`, you were getting a `TypeError` from inside the loop; you now get
a `NotImplementedError` at the entry that says what to do. Fit the stack elements separately, or build
the local model and drive your own loop -- that path supports stacked points and returns a shape-`C`
objective. A regularizer on a stacked point raises for a separate reason (see the changelog).

## Upgrading from 2026.0.0

Three changes in this release can break existing code. All three are mechanical, and each raises a
clear error rather than failing silently.

**The named contraction functions are gone.** `backend.contractions` used to export ~104 functions
whose names spelled their own subscripts (`WCa_Caib_WCi_to_WCb` and relatives). They are replaced by
one interpreter, `contract`, and the migration is a direct rewrite — the function name *is* the
subscripts string:

```python
from t3toolbox.backend.contractions import contract

# before:  WCa_Caib_WCi_to_WCb(mu, G, xi, n_probe, n_frame)
# after:
contract('WCa,Caib,WCi->WCb', mu, G, xi, len_W=n_probe, len_C=n_frame)
```

A trailing `n_probe` / `n_frame` argument becomes the keyword `len_W=` / `len_C=`, and you only need
to supply one when the subscripts alone cannot pin the split — the error message names exactly which.
The results are numerically identical (each named function was checked against its `contract` call
before removal). See [`grouped_contractions.md`](grouped_contractions.md).

**A custom `GeometryOps` must accept `aux`.** The backend geometry protocol gained an optional
`precompute(frame)` slot, and `project` / `retract` now take a third argument. If you implement your
own geometry, accept and ignore it:

```python
def project(frame, variations, aux=None): ...
def retract(frame, variations, aux=None): ...
precompute = None          # or a callable frame -> whatever project/retract need
```

The slot exists so a geometry with expensive per-frame setup pays for it once per local model instead
of once per Hessian matvec; the built-in geometries pass `None`. Rationale:
[`contributor/precompute_and_caching.md`](contributor/precompute_and_caching.md).

**A custom `Regularizer` must accept `aux`.** Same shape of change, for the same reason:
`gradient(geom, frame, aux=None)`, `hessian(geom, frame, p, aux=None)` and
`quadratic(geom, frame, p, aux=None)` now receive the geometry aux, so a regularizer composed with a
shared-factor geometry reuses the frame companion instead of rebuilding it per matvec. Accept and
ignore the parameter if you do not need it.

**One behavior change worth knowing about, which is not a signature break:** `use_jit=True` now moves
your inputs onto jax rather than silently running eager when they are numpy. Runs that previously
reported "jit" timings while executing eagerly will now genuinely compile, return **jax-backed**
results (float32 unless you enable `jax_enable_x64`), and raise if jax is not installed. If you were
relying on the old behavior you were not getting jit; see
[`fitting_and_optimization.md`](fitting_and_optimization.md) §4.5.

```{include} ../CHANGELOG.md
:start-line: 5
```
