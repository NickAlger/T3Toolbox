# Release notes

The full changelog is reproduced below, newest first; the format follows
[Keep a Changelog](https://keepachangelog.com/) and versions are `YYYY.MINOR.PATCH` (a calendar year,
then a minor number that increments with each release of that year). Start with the upgrade notes if
you are moving an existing codebase forward.

## Upgrading from 2026.1.0

The optimization layer's axis objects became classes. Everything here raises rather than failing
silently, and the constructor spellings you already use mostly still work.

**Geometries moved and are constructed, not fetched.** `backend.optimizers.MANIFOLD_OPS` becomes
`backend.geometry.ManifoldGeometryOps()`, `COREWISE_OPS` becomes `CorewiseGeometryOps()`, and
`shared_geometry_ops(base, groups)` becomes `base.with_sharing(sharing, shape)`. On the uniform side,
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

**`fitting.UniformGaussNewtonModel` is gone.** One `GaussNewtonModel` now serves both representations,
and the `*_model` factories still dispatch on `x`, so `fitting.apply_model(UNIFORM_MANIFOLD, ux, ...)`
returns what it always did -- just of the merged type. If you were checking
`isinstance(m, UniformGaussNewtonModel)`, check the frame instead:
`isinstance(m.frame, ubv.UT3Frame)`.

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
