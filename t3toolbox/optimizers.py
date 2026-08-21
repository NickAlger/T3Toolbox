# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Frontend adapter for the geometry-agnostic optimizers.

A thin wrapper over :py:mod:`t3toolbox.backend.optimizers` (where the algorithms live -- check-free, so a
raw-``.data`` user can call them directly). This layer (1) maps the frontend geometry singletons and the
sampling-kind name to the backend ``GeometryOps`` / ``SamplingKind``, (2) calls the backend optimizer on the
raw cores, and (3) re-wraps the result. Design: ``dev/archive/optimizers_plan.md``.

**Two representations, inferred from** ``x0`` (the library-wide ragged/uniform dispatch; ``_setup``):

* a **ragged** ``TuckerTensorTrain`` x0 with a ragged geometry (``manifold.MANIFOLD`` / ``COREWISE``) --
  the optimizer runs on the raw cores; the frame is re-orthogonalized internally each step.
* a **uniform** ``UniformTuckerTensorTrain`` x0 with a uniform geometry
  (``uniform_manifold.UNIFORM_MANIFOLD`` / ``UNIFORM_COREWISE``) -- the optimizer runs on the packed
  supercore pair (``lax.scan`` over the mode axis; the speed path). The uniform path requires a
  **minimal-rank frame**; it calls :py:func:`~t3toolbox.backend.uniform_fitting.uniform_minimal`
  transparently so a frontend user never meets that requirement.

The geometry must match ``x0``'s representation (a uniform x0 with a ragged geometry, or vice versa, is a
structural error). The result is returned in the same representation as ``x0``.

**``use_jit=True``** (an explicit kwarg on ``mc_sgd`` / ``adam`` / ``newton_cg``) opts into jax: the
optimizer **auto-converts** ``x0`` / ``sample`` / ``data`` onto jax and jit-compiles, so the result comes
back **jax-backed** in jax's default float32 (enable jax x64 for float64); it **raises** if jax is not
installed rather than silently running eager. See :py:func:`t3toolbox.backend.optimizers.newton_cg`.

    >>> # x_opt, stats = optimizers.gradient_descent(MANIFOLD, 'probe', ww, data, x0)           # ragged
    >>> # x_opt, stats = optimizers.newton_cg(UNIFORM_MANIFOLD, 'probe', ww, data, ux0)         # uniform
"""
import typing as typ

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.shared_geometry as sg
import t3toolbox.fitting as _fitting   # for _canonical_weight (the shared frontend weight contract; no cycle)
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.optimizer_display as bdisp
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf

# Regularizers live in the backend (check-free; a raw-.data user constructs them directly) and are
# re-exported here for frontend convenience -- neither user is privileged (dev/regularization_design.md §5a).
from t3toolbox.backend.regularization import Regularizer, IdentityRegularizer

__all__ = [
    'gradient_descent',
    'mc_sgd',
    'adam',
    'newton_cg',
    'Regularizer',
    'IdentityRegularizer',
]

_KIND = {'apply': bfit.APPLY, 'entries': bfit.ENTRIES, 'probe': bfit.PROBE}
_DERIV_KIND = {'apply_derivatives':   bfit.apply_derivatives_kind,
               'entries_derivatives': bfit.entries_derivatives_kind,
               'probe_derivatives':   bfit.probe_derivatives_kind}

# A frontend point in either representation. The four optimizers infer ragged (TuckerTensorTrain) vs
# uniform (UniformTuckerTensorTrain) from x0's type, and require a matching geometry singleton -- see _setup.
Point = typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain]


def _geometry_ops(geometry, shape=None):
    """Map a **ragged** frontend geometry (a singleton, or a :py:class:`SharedGeometry` over one)
    to its backend geometry (check-free; :py:mod:`t3toolbox.backend.geometry`). A shared wrapper needs
    ``shape`` (the mode sizes) to canonicalize its partition."""
    if geometry is t3m.MANIFOLD:
        return bgeo.ManifoldGeometryOps()
    if geometry is t3m.COREWISE:
        return bgeo.CorewiseGeometryOps()
    if isinstance(geometry, sg.SharedGeometry):
        if geometry.is_uniform:
            raise ValueError("a ragged TuckerTensorTrain x0 requires a SharedGeometry over a RAGGED "
                             "base (manifold.MANIFOLD / manifold.COREWISE); this one wraps a uniform "
                             "base -- pass a UniformTuckerTensorTrain x0 instead")
        if shape is None:
            raise ValueError("a SharedGeometry needs the point's shape to canonicalize its "
                             "sharing partition (internal: pass shape=x0.shape)")
        base = bgeo.ManifoldGeometryOps() if geometry.base is t3m.MANIFOLD else bgeo.CorewiseGeometryOps()
        return base.with_sharing(geometry.sharing, shape)
    raise ValueError(f"unknown geometry {geometry!r}; expected manifold.MANIFOLD / manifold.COREWISE "
                     f"(or a shared_geometry.SharedGeometry over one, or the uniform singletons "
                     f"uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE with a "
                     f"UniformTuckerTensorTrain x0)")


def _uniform_geometry_name(geometry) -> typ.Tuple[
    str,                          # backend geometry name: 'manifold' | 'corewise'
    typ.Optional[typ.Tuple],      # the sharing labels (None = unshared)
]:
    """Map a **uniform** frontend geometry (a singleton, or a :py:class:`SharedGeometry` over one)
    to the backend geometry name plus its sharing labels (``None`` for the plain singletons)."""
    if geometry is ut3m.UNIFORM_MANIFOLD:
        return 'manifold', None
    if geometry is ut3m.UNIFORM_COREWISE:
        return 'corewise', None
    if isinstance(geometry, sg.SharedGeometry) and geometry.is_uniform:
        name = 'manifold' if geometry.base is ut3m.UNIFORM_MANIFOLD else 'corewise'
        return name, geometry.sharing
    raise ValueError(f"a UniformTuckerTensorTrain x0 requires a uniform geometry "
                     f"(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE, or a "
                     f"shared_geometry.SharedGeometry over one), got {geometry!r}")


def _check_kind(kind: str, order: typ.Optional[int]) -> None:
    """Validate the sampling-kind name (shared across representations); derivative kinds need ``order``."""
    if kind not in _KIND and kind not in _DERIV_KIND:
        raise ValueError(f"unknown sampling kind {kind!r}; expected one of "
                         f"{sorted(_KIND) + sorted(_DERIV_KIND)}")
    if kind in _DERIV_KIND and order is None:
        raise ValueError(f"derivative kind {kind!r} requires order=")


def _n_modes(kind: str, sample: typ.Any) -> int:
    """Number of tensor modes ``d`` from the kind's sample (``ww`` list / ``index`` array; a derivative
    sample is the pair ``(ww/index, pp)``) -- needed to validate a residual weight's mode dimension."""
    s = sample[0] if kind.endswith('_derivatives') else sample
    return s.shape[0] if kind.startswith('entries') else len(s)


def _resolve_chunk_size(chunk_size, kind, x0, sample, order, batch=None):
    """Resolve ``chunk_size='auto'`` to a memory-balanced value for a uniform **probe_derivatives** fit
    (:py:func:`~t3toolbox.backend.sampling_derivatives.estimate_chunk_size`, measured eagerly from the
    supercore shapes); ints / ``None`` pass through unchanged. Only probe_derivatives has a chunkable
    ``𝒥ᵀ`` assembly, so every other kind (and the ragged path, which frees eagerly) resolves to ``None``
    -- a no-op. For minibatch optimizers the transpose sees ``batch`` probes, not the full ``|W|``. See
    :doc:`/chunking`."""
    if chunk_size != 'auto':
        return chunk_size
    if kind != 'probe_derivatives' or not isinstance(x0, ut3.UniformTuckerTensorTrain):
        return None
    try:
        import jax  # noqa: F401  -- the estimator measures peak scratch via a compile
    except ImportError:
        return 100
    import numpy as np
    tsc, qsc = np.asarray(x0.tucker_supercore), np.asarray(x0.tt_supercore)  # (d,nU,N) ; (d,r,nU,r)
    d, nU, r = tsc.shape[0], int(tsc.shape[-2]), int(qsc.shape[-1])
    full_W = int(np.prod(np.asarray(sample[0][0]).shape[:-1]))
    w = min(int(batch), full_W) if batch else full_W
    return bfit_pd().estimate_chunk_size(tuple(x0.shape), (nU,) * d, (r,) * (d + 1), order, w, dtype=tsc.dtype)


def bfit_pd():
    """Lazy import of the sampling-derivatives backend (only needed to resolve chunk_size='auto')."""
    import t3toolbox.backend.sampling_derivatives as pd
    return pd


def _setup(
        geometry,           # ragged (t3m.MANIFOLD/COREWISE) or uniform (ut3m.UNIFORM_MANIFOLD/COREWISE) singleton
        kind:   str,        # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample,             # ww / index (regular) or (ww, pp) / (index, pp) (derivatives)
        data,               # observed S(x_true) (+ noise)
        x0,                 # TuckerTensorTrain (ragged) or UniformTuckerTensorTrain (uniform)
        order:  typ.Optional[int] = None,  # derivative kinds only: highest order (required)
        weight: typ.Optional[typ.Any] = None,  # residual weight ω: per-mode (probe) / ω[mode,order] (derivatives)
        regularizer: typ.Any = None,       # optional backend.regularization.Regularizer (ragged or uniform)
        chunk_size: typ.Any = 'auto',      # probe_derivatives 𝒥ᵀ memory chunk; 'auto' -> estimate_chunk_size (docs/chunking.md)
        batch:  typ.Optional[int] = None,  # minibatch size (mc_sgd/adam): the W the transpose sees, for 'auto'
) -> typ.Tuple[
        bopt.Problem,       # the fixed-rank least-squares problem
        typ.Any,            # initial optimizer state: x0.data (ragged) or the bare supercore pair (uniform)
        typ.Callable,       # rewrap: backend result cores -> the frontend tensor of the same representation
]:
    """Build the backend ``(problem, initial state, rewrap)`` for either representation.

    The representation is **inferred from** ``x0``'s type (like the library-wide ragged/uniform dispatch);
    the geometry must match -- a ``UniformTuckerTensorTrain`` x0 pairs with a uniform geometry singleton, a
    ``TuckerTensorTrain`` x0 with a ragged one -- else a structural error. The uniform path reduces ``x0`` to
    minimal ranks (:py:func:`~t3toolbox.backend.uniform_fitting.uniform_minimal`) transparently, so a
    frontend user never meets the minimal-rank requirement, and rewraps the optimizer's bare supercore pair
    with the frame's held ``shape`` + ``masks``.
    """
    _check_kind(kind, order)
    if weight is not None and kind in ('apply', 'entries'):      # plain apply/entries: no axis to weight
        raise ValueError(f"the plain '{kind}' kind takes no residual weight (no mode or order axis); "
                         "per-mode weighting is defined for probe, per-order for the derivative kinds.")
    wm = _fitting._canonical_weight(weight, kind, _n_modes(kind, sample), order or 0)   # 2-D ω[m,o] or None

    if isinstance(x0, ut3.UniformTuckerTensorTrain):
        geom_name, sharing_spec = _uniform_geometry_name(geometry)
        x0m = uf.uniform_minimal(x0, sharing=sharing_spec)   # transparent SHARED-minimal reduction (no-op if minimal)
        cs = _resolve_chunk_size(chunk_size, kind, x0m, sample, order, batch)   # 'auto' -> balanced (probe only)
        problem = uf.uniform_least_squares_problem(geom_name, kind, x0m, sample, data, order, wm, regularizer,
                                                   chunk_size=cs, sharing=sharing_spec)
        init = (x0m.tucker_supercore, x0m.tt_supercore)  # optimizer state = the bare supercore pair
        return problem, init, lambda sc: ut3.UniformTuckerTensorTrain(sc[0], sc[1], x0m.shape, x0m.masks)

    if isinstance(x0, t3.TuckerTensorTrain):
        if kind in _KIND:                                # plain kinds: only probe is weightable (per-mode)
            bk = bfit.probe_kind(wm) if kind == 'probe' and wm is not None else _KIND[kind]
        elif kind == 'probe_derivatives':                # the only derivative kind with a chunkable 𝒥ᵀ assembly
            cs = _resolve_chunk_size(chunk_size, kind, x0, sample, order, batch)   # ragged -> None (no-op)
            bk = _DERIV_KIND[kind](order, wm, chunk_size=cs)
        else:
            bk = _DERIV_KIND[kind](order, wm)
        problem = bopt.least_squares_problem(_geometry_ops(geometry, x0.shape), bk, sample, data,
                                             regularizer=regularizer)
        return problem, x0.data, lambda cores: t3.TuckerTensorTrain(*cores)

    raise TypeError(f"x0 must be a TuckerTensorTrain or UniformTuckerTensorTrain, got {type(x0).__name__}")


# Derivative kinds (kind='*_derivatives') need `order` and a paired `(ww, pp)` / `(index, pp)` sample;
# everything else is identical. `order`/`weight` build the kind (`weight` = a residual weight ω:
# per-mode for probe, per-order for derivatives, the full ω[mode,order] matrix for probe_derivatives;
# apply/entries take none). `draw` (mc_sgd / adam) is the custom minibatch draw (None = the flat default).
def gradient_descent(
        geometry,                       # ragged or uniform geometry singleton (must match x0's representation)
        kind:     str,                  # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample:   typ.Any,              # ww / index, or (ww, pp) / (index, pp) for derivatives
        data:     typ.Any,             # observed values to fit
        x0:       Point,                # initial point (any cores; the geometry orthogonalizes internally)
        order:    typ.Optional[int] = None,  # derivative kinds: highest order (required)
        weight:   typ.Optional[typ.Any] = None,  # residual weight ω: per-mode (probe) / ω[mode,order] (derivatives)
        regularizer: typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
        chunk_size: typ.Any = 'auto',   # probe_derivatives 𝒥ᵀ memory chunk; 'auto' -> estimate_chunk_size (docs/chunking.md)
        **kwargs,                       # forwarded to backend.optimizers.gradient_descent (n_iter, gtol_rel, ...)
) -> typ.Tuple[Point, dict]:            # (x_opt, stats)
    """Fit ``x`` to ``data`` by steepest descent (Cauchy step + Armijo line search) on ``geometry``.

    Accepts a ragged ``TuckerTensorTrain`` (with ``manifold.MANIFOLD`` / ``COREWISE``) or a uniform
    ``UniformTuckerTensorTrain`` (with ``uniform_manifold.UNIFORM_MANIFOLD`` / ``UNIFORM_COREWISE``); the
    representation is inferred from ``x0`` and returned in kind. Pass ``regularizer`` (e.g.
    ``optimizers.IdentityRegularizer(λ)``) to add ``ρ(x)`` to the objective (either representation). See
    :py:func:`t3toolbox.backend.optimizers.gradient_descent`."""
    problem, init, rewrap = _setup(geometry, kind, sample, data, x0, order, weight, regularizer, chunk_size=chunk_size)
    x_cores, stats = bopt.gradient_descent(problem, init, **kwargs)
    return rewrap(x_cores), stats


def mc_sgd(
        geometry,                       # ragged/uniform MANIFOLD (intended) / COREWISE (must match x0)
        kind:     str,                  # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample:   typ.Any,              # ww / index, or (ww, pp) / (index, pp) for derivatives
        data:     typ.Any,             # observed values to fit
        x0:       Point,                # initial point
        rng,                            # np.random.Generator -- passed to the draw each step
        batch:    int,                  # measurements per minibatch (default flat draw; ignored if draw given)
        order:    typ.Optional[int] = None,
        weight:   typ.Optional[typ.Any] = None,   # residual weight ω: per-mode (probe) / ω[mode,order] (derivatives)
        draw:     typ.Optional[typ.Callable]        = None,  # custom draw(rng)->(sample_B,data_B); None = flat
        use_jit:  bool = False,         # jit the per-step kernel: auto-converts x0/sample/data to jax -> a jax-backed float32 result; raises if jax absent
        regularizer: typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ); scaled by batch/n per step
        chunk_size: typ.Any = 'auto',   # probe_derivatives 𝒥ᵀ memory chunk; 'auto' -> estimate_chunk_size (docs/chunking.md)
        **kwargs,                       # forwarded to backend.optimizers.mc_sgd (max_iter, check_every, ...)
) -> typ.Tuple[Point, dict]:
    """Manifold Cauchy SGD -- minibatched, tuning-free Cauchy step. Ragged or uniform ``x0`` (see
    :py:func:`gradient_descent`). A ``regularizer`` is scaled by ``batch/n`` per step so
    ``λ`` matches the full-batch optimizers. See :py:func:`t3toolbox.backend.optimizers.mc_sgd`."""
    problem, init, rewrap = _setup(geometry, kind, sample, data, x0, order, weight, regularizer, chunk_size=chunk_size, batch=batch)
    x_cores, stats = bopt.mc_sgd(problem, init, rng, batch, draw=draw, use_jit=use_jit, **kwargs)
    return rewrap(x_cores), stats


def adam(
        geometry,                       # ragged/uniform COREWISE (intended) / MANIFOLD (must match x0)
        kind:     str,                  # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample:   typ.Any,              # ww / index, or (ww, pp) / (index, pp) for derivatives
        data:     typ.Any,             # observed values to fit
        x0:       Point,                # initial point
        rng,                            # np.random.Generator -- passed to the draw each step
        batch:    int,                  # measurements per minibatch (default flat draw; ignored if draw given)
        order:    typ.Optional[int] = None,
        weight:   typ.Optional[typ.Any] = None,   # residual weight ω: per-mode (probe) / ω[mode,order] (derivatives)
        draw:     typ.Optional[typ.Callable]        = None,
        use_jit:  bool = False,         # jit the per-step kernel: auto-converts x0/sample/data to jax -> a jax-backed float32 result; raises if jax absent
        regularizer: typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ); scaled by batch/n per step
        chunk_size: typ.Any = 'auto',   # probe_derivatives 𝒥ᵀ memory chunk; 'auto' -> estimate_chunk_size (docs/chunking.md)
        **kwargs,                       # forwarded to backend.optimizers.adam (lr, max_iter, ...)
) -> typ.Tuple[Point, dict]:
    """Adam over the cores -- the dependency-free first-order method for the corewise geometry. Ragged or
    uniform ``x0`` (see :py:func:`gradient_descent`). A ``regularizer`` is scaled by
    ``batch/n`` per step so ``λ`` matches the full-batch optimizers. See
    :py:func:`t3toolbox.backend.optimizers.adam`."""
    problem, init, rewrap = _setup(geometry, kind, sample, data, x0, order, weight, regularizer, chunk_size=chunk_size, batch=batch)
    x_cores, stats = bopt.adam(problem, init, rng, batch, draw=draw, use_jit=use_jit, **kwargs)
    return rewrap(x_cores), stats


def newton_cg(
        geometry,                       # ragged/uniform MANIFOLD (intended) / COREWISE (must match x0)
        kind:     str,                  # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample:   typ.Any,              # ww / index, or (ww, pp) / (index, pp) for derivatives
        data:     typ.Any,             # observed values to fit
        x0:       Point,                # initial point (zero is fine on the manifold)
        order:    typ.Optional[int] = None,
        weight:   typ.Optional[typ.Any] = None,   # residual weight ω: per-mode (probe) / ω[mode,order] (derivatives)
        verbose:  bool = False,                   # print per-iteration diagnostics (a relative-error table + more)
        val_sample: typ.Any = None,               # optional validation sample (same layout as `sample`) -> a train|val table
        val_data:   typ.Any = None,               # optional validation data (both given adds the val column)
        callback:   typ.Optional[typ.Callable] = None,  # custom callback(NewtonInfo) each iter (overrides `verbose`)
        use_jit:    bool = False,               # jit the inner CG: auto-converts x0/sample/data to jax -> a jax-backed float32 result; raises if jax absent
        regularizer: typ.Any = None,            # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
        chunk_size: typ.Any = 'auto',   # probe_derivatives 𝒥ᵀ memory chunk; 'auto' -> estimate_chunk_size (docs/chunking.md)
        **kwargs,                       # forwarded to backend.optimizers.newton_cg (max_newton, gtol_rel, g0norm_newton, ...)
) -> typ.Tuple[Point, dict]:
    """Inexact Riemannian Newton-CG with an Armijo line search -- the manifold workhorse. Ragged or uniform
    ``x0`` (see :py:func:`gradient_descent`). See :py:func:`t3toolbox.backend.optimizers.newton_cg`.

    In a **warm-start continuation loop** the initial gradient norm ‖g0‖ is misleadingly small, which
    over-tightens the Newton stop and slackens CG; pass ``g0norm_newton`` / ``g0norm_cg`` (forwarded via
    ``**kwargs``) to override the reference norm the two relative stopping tests use, and
    ``cg_forcing_power`` to trade more CG iterations per Newton step for fewer Newton steps (all detailed
    on the backend).

    ``verbose=True`` prints a per-iteration diagnostic block (objective / gradient, CG stats, line search,
    and the per-``(mode, order)`` relative-error table); pass ``val_sample`` / ``val_data`` to add a
    validation column. With a ``regularizer`` attached the objective is shown split as ``obj = misfit +
    reg`` (the ``½‖ω⊙r‖²`` data misfit vs ``ρ(x)``); both parts are on every record as ``misfit`` /
    ``regularization`` (the latter ``None`` when unregularized), in ``stats['history']`` (always) and
    ``stats['diagnostics']`` (when verbose). This is a
    thin convenience over the **backend** display -- a raw-``.data`` user builds the same callback with
    :py:func:`t3toolbox.backend.optimizer_display.make_newton_display` and passes it as ``callback=`` to
    :py:func:`t3toolbox.backend.optimizers.newton_cg`. A custom ``callback`` overrides ``verbose``. Works on
    both the ragged and uniform layers (the uniform ``block_sumsq`` reduces the packed residual directly;
    validation data is packed automatically).

    For a large-``|W|`` uniform ``probe_derivatives`` fit, ``chunk_size='auto'`` (the default) sizes the
    ``𝒥ᵀ`` gradient assembly's memory automatically (via
    :py:func:`~t3toolbox.backend.sampling_derivatives.estimate_chunk_size`); pass an ``int`` / ``None`` to
    override. Other kinds and the ragged layer ignore it. See :doc:`/chunking`.

    Examples
    --------
    Recover a low-rank tensor from noiseless ``apply`` measurements on the **uniform** layer -- a uniform
    ``x0`` (with ``UNIFORM_MANIFOLD``) returns a ``UniformTuckerTensorTrain`` (a zero start is fine on the
    manifold; unit-norm probe rows keep the least-squares well-conditioned):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> import t3toolbox.optimizers as optimizers
    >>> np.random.seed(0)
    >>> A = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(120, N) for N in (6, 7, 8)]
    >>> ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]   # unit-norm rows
    >>> b = A.apply(ww)
    >>> x0 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.zeros((6, 7, 8), (2, 2, 2), (1, 2, 2, 1)))
    >>> x_opt, stats = optimizers.newton_cg(ut3m.UNIFORM_MANIFOLD, 'apply', ww, b, x0, max_newton=30)
    >>> type(x_opt).__name__                              # returned in the same (uniform) representation
    'UniformTuckerTensorTrain'
    >>> bool(np.linalg.norm(x_opt.to_dense() - A.to_dense()) / np.linalg.norm(A.to_dense()) < 1e-6)
    True

    For a **ragged** fit pass ``t3m.MANIFOLD`` and a ``TuckerTensorTrain`` ``x0`` instead; the call is
    otherwise identical.

    **Precision — jit runs in jax's default float32.** ``use_jit=True`` opts into jax world, whose default
    dtype is float32. Fit a small exact-rank tensor three ways; the giveaway throughout is the returned
    core's dtype. First set up the problem:

    >>> import t3toolbox.manifold as t3m
    >>> np.random.seed(0)
    >>> A = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(60, N) for N in (4, 5, 6)]
    >>> ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
    >>> b = A.probe(ww)
    >>> x0 = t3.TuckerTensorTrain.zeros((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
    >>> def rel_err(x): return float(np.linalg.norm(np.asarray(x.to_dense()) - A.to_dense()) / np.linalg.norm(A.to_dense()))

    (1) The numpy path is float64 and recovers to near machine precision (~1e-10):

    >>> x_np, _ = optimizers.newton_cg(t3m.MANIFOLD, 'probe', ww, b, x0, max_newton=20)
    >>> str(x_np.data[0][0].dtype), bool(rel_err(x_np) < 1e-8)
    ('float64', True)

    (2) ``use_jit=True`` runs the SAME code in jax's default float32 and stalls ~1000x coarser (~1e-7):

    >>> x_jit, _ = optimizers.newton_cg(t3m.MANIFOLD, 'probe', ww, b, x0, max_newton=20, use_jit=True)
    >>> str(x_jit.data[0][0].dtype), bool(rel_err(x_jit) > 1e-8)
    ('float32', True)

    (3) Enabling jax x64 restores float64 under jit -- full accuracy again (~1e-10). ``jax_enable_x64`` is
    a **global, process-wide** flag, so restore it right after the fit; the check reads values captured
    while it was on:

    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>> x_x64, _ = optimizers.newton_cg(t3m.MANIFOLD, 'probe', ww, b, x0, max_newton=20, use_jit=True)
    >>> dtype_x64, ok_x64 = str(x_x64.data[0][0].dtype), bool(rel_err(x_x64) < 1e-8)
    >>> jax.config.update("jax_enable_x64", False)          # restore the default before asserting (no leak)
    >>> dtype_x64, ok_x64
    ('float64', True)
    """
    problem, init, rewrap = _setup(geometry, kind, sample, data, x0, order, weight, regularizer, chunk_size=chunk_size)
    records = None
    if callback is None and verbose:
        vs, vd = val_sample, val_data
        if isinstance(x0, ut3.UniformTuckerTensorTrain) and val_data is not None:
            vs = uf.pack_sample(kind, val_sample, x0.N)   # pack validation to the packed kind's width
            vd = uf.pack_data(kind, val_data, x0.N)
        callback, records = bdisp.make_newton_display(problem, val_sample=vs, val_data=vd)
    x_cores, stats = bopt.newton_cg(problem, init, callback=callback, use_jit=use_jit, **kwargs)
    if records is not None:
        stats['diagnostics'] = records
    return rewrap(x_cores), stats
