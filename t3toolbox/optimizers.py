# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Frontend adapter for the geometry-agnostic optimizers.

A thin wrapper over :py:mod:`t3toolbox.backend.optimizers` (where the algorithms live -- check-free, so a
raw-``.data`` user can call them directly). This layer (1) maps the frontend geometry singletons and the
sampling-kind name to the backend geometry / ``SamplingKind``, (2) calls the backend optimizer on the
raw cores, and (3) re-wraps the result. Design: ``docs/contributor/fitting_internals.md`` §"Backend-first".

**Two representations, inferred from** ``x0`` (the library-wide ragged/uniform dispatch;
:py:func:`least_squares_problem`):

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
back **jax-backed** in jax's default float32 (enable jax x64 for float64); if jax is not installed it
runs eager on numpy with a one-time warning (the jax-absent policy, ``backend.common.jax_or_warn``). See :py:func:`t3toolbox.backend.optimizers.newton_cg`.

    >>> # x_opt, stats = optimizers.gradient_descent(MANIFOLD, 'probe', ww, data, x0)           # ragged
    >>> # x_opt, stats = optimizers.newton_cg(UNIFORM_MANIFOLD, 'probe', ww, data, ux0)         # uniform

The four optimizers build their backend problem with :py:func:`least_squares_problem`, which is public: a
script that wants the backend :py:class:`~t3toolbox.backend.optimizers.Problem` (to build local models, or to
call a backend optimizer directly) builds it from the same frontend vocabulary.
"""
import dataclasses as dc
import typing as typ
import warnings

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
# re-exported here for frontend convenience -- neither user is privileged (``docs/contributor/fitting_internals.md`` §Regularization).
from t3toolbox.backend.regularization import Regularizer, IdentityRegularizer

__all__ = [
    'gradient_descent',
    'mc_sgd',
    'adam',
    'newton_cg',
    'least_squares_problem',
    'LeastSquaresSetup',
    'Regularizer',
    'IdentityRegularizer',
]

_KIND = {'apply': bfit.APPLY, 'entries': bfit.ENTRIES, 'probe': bfit.PROBE}
_DERIV_KIND = {'apply_derivatives':   bfit.apply_derivatives_kind,
               'entries_derivatives': bfit.entries_derivatives_kind,
               'probe_derivatives':   bfit.probe_derivatives_kind}

# A frontend point in either representation. The four optimizers infer ragged (TuckerTensorTrain) vs
# uniform (UniformTuckerTensorTrain) from x0's type, and require a matching geometry singleton -- see
# least_squares_problem.
Point = typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain]


def _geometry_ops(geometry, shape=None):
    """Map a **ragged** frontend geometry (a singleton, or a :py:class:`SharedGeometry` over one)
    to its backend geometry (check-free; :py:mod:`t3toolbox.backend.geometry`). A shared wrapper needs
    ``shape`` (the mode sizes) to canonicalize its partition."""
    if isinstance(geometry, sg.SharedGeometry):
        if geometry.is_uniform:
            raise ValueError("a ragged TuckerTensorTrain x0 requires a SharedGeometry over a RAGGED "
                             "base (manifold.MANIFOLD / manifold.COREWISE); this one wraps a uniform "
                             "base -- pass a UniformTuckerTensorTrain x0 instead")
        if shape is None:
            raise ValueError("a SharedGeometry needs the point's shape to canonicalize its "
                             "sharing partition (internal: pass shape=x0.shape)")
    ops = _fitting.ragged_backend_geometry(geometry, shape)
    if ops is not None:
        return ops
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


def _is_manifold_geometry(geometry) -> bool:
    """Whether ``geometry`` is (a shared wrapper over) a manifold geometry, ragged or uniform."""
    if isinstance(geometry, sg.SharedGeometry):
        return geometry._is_manifold_kind
    return geometry is t3m.MANIFOLD or geometry is ut3m.UNIFORM_MANIFOLD


def _ragged_minimal(x0, sharing):
    """Reduce a ragged manifold start to minimal ranks (the SAME tensor) -- the mirror of
    :py:func:`~t3toolbox.backend.uniform_fitting.uniform_minimal`. A non-minimal ``x0`` is legal, but the
    first retraction drops the redundant rank anyway, and an optimizer that allocates state at ``x0``'s
    core shapes (adam's moments) then broke at step 2. With ``sharing`` the SHARED-minimal notion applies.
    No-op when already minimal, so a minimal start is untouched (same gauge, same trajectory)."""
    minimal = t3.TuckerTensorTrain.get_minimal_ranks(x0.shape, x0.tucker_ranks, x0.tt_ranks, sharing=sharing)
    if (tuple(x0.tucker_ranks), tuple(x0.tt_ranks)) == (tuple(minimal[0]), tuple(minimal[1])):
        return x0
    y, _, _ = x0.t3svd(sharing=sharing)                              # lossless, left-orthogonal
    return y.rank_adjustment_sweep('right_to_left', sharing=sharing)  # minimal (the t3svd output is left-orthogonal)


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


def _resolve_chunk_size(chunk_size, kind, x0, sample, order, batch=None, hessian='gauss_newton'):
    """Resolve ``chunk_size='auto'`` to a memory-balanced value for a uniform **probe_derivatives** fit
    (:py:func:`~t3toolbox.backend.sampling_derivatives.estimate_chunk_size`, measured eagerly from the
    supercore shapes); ints / ``None`` pass through unchanged. Only probe_derivatives has a chunkable
    ``𝒥ᵀ`` assembly, so every other kind (and the ragged path, which frees eagerly) resolves to ``None``
    -- a no-op. For minibatch optimizers the transpose sees ``batch`` probes, not the full ``|W|``.
    ``hessian`` is the solver's mode, passed straight to the estimator: with ``'exact'`` the same chunk also
    drives the exact Hessian's two-hole assemblies, so it is sized for those too. See :doc:`/chunking`."""
    if chunk_size != 'auto':
        return chunk_size
    if kind != 'probe_derivatives' or not isinstance(x0, ut3.UniformTuckerTensorTrain):
        return None
    try:
        import jax  # noqa: F401  -- the estimator measures peak scratch via a compile
    except ImportError:
        return _bfit_pd().DEFAULT_CHUNK_SIZE
    import math
    tsc, qsc = x0.tucker_supercore, x0.tt_supercore  # (d,nU,N) ; (d,r,nU,r) -- shapes/dtype only, no host pull
    d, nU, r = int(tsc.shape[0]), int(tsc.shape[-2]), int(qsc.shape[-1])
    full_W = math.prod(sample[0][0].shape[:-1])
    w = min(int(batch), full_W) if batch else full_W
    return _bfit_pd().estimate_chunk_size(tuple(x0.shape), (nU,) * d, (r,) * (d + 1), order, w, dtype=tsc.dtype,
                                          hessian=hessian)


def _bfit_pd():
    """Lazy import of the sampling-derivatives backend (only needed to resolve chunk_size='auto')."""
    import t3toolbox.backend.sampling_derivatives as pd
    return pd


@dc.dataclass(frozen=True, eq=False)   # eq=False: identity eq/hash (array data + a callable; value-eq is ill-posed)
class LeastSquaresSetup:
    """What :py:func:`~t3toolbox.optimizers.least_squares_problem` returns: the backend problem, the backend starting cores, and
    the map from backend cores back to a frontend tensor.

    A named result, not a bare tuple, so that the frontend constructor's three outputs are not mistaken
    for the backend :py:func:`t3toolbox.backend.optimizers.least_squares_problem`'s single ``Problem``."""
    problem:  bopt.Problem    # the backend problem: .local_model(x_cores) / .objective(x_cores) / .geom
    x0_cores: typ.Any         # x0 as backend cores, maybe rank-reduced: x0.data (ragged) / the supercore pair (uniform)
    rewrap:   typ.Callable    # backend cores -> the frontend tensor, in x0's representation


def least_squares_problem(
        geometry,               # MANIFOLD / COREWISE / UNIFORM_* / a SharedGeometry over one
        kind:        str,       # 'apply' / 'entries' / 'probe' (+ '_derivatives'), case-insensitive
        sample:      typ.Any,   # ww / index, or (ww, pp) / (index, pp) for derivatives
        data:        typ.Any,   # observed S(x_true) (+ noise)
        x0:          Point,     # ragged or uniform: picks the representation

        order:       typ.Optional[int] = None,      # derivative kinds only: highest order (required)
        weight:      typ.Optional[typ.Any] = None,  # residual weight ω: per-mode (probe) / ω[mode,order]
        regularizer: typ.Any = None,                # optional, e.g. optimizers.IdentityRegularizer(λ)
        chunk_size:  typ.Any = 'auto',              # probe_derivatives 𝒥ᵀ memory chunk (docs/chunking.md)
        batch:       typ.Optional[int] = None,      # minibatch W (mc_sgd / adam), for chunk_size='auto'
        hessian:     str = 'gauss_newton',          # the solve's mode, for chunk_size='auto': or 'exact'
) -> LeastSquaresSetup:  # .problem, .x0_cores, .rewrap
    """Build the backend least-squares problem from the frontend vocabulary -- the constructor the four
    optimizers use, public for scripts that drive the backend directly.

    The frontend twin of :py:func:`t3toolbox.backend.optimizers.least_squares_problem`: that one takes
    backend objects (a backend geometry, a ``SamplingKind``) and returns a bare ``Problem``; this one takes
    what the frontend optimizers take -- a geometry singleton (or a ``SharedGeometry``), the kind's name,
    ``order`` / ``weight`` -- and returns a :py:class:`LeastSquaresSetup` ``(problem, x0_cores, rewrap)``.
    ``problem`` builds local models and evaluates the objective at backend cores, and can be handed with
    ``x0_cores`` to a backend optimizer (:py:func:`t3toolbox.backend.optimizers.newton_cg` and the
    others); ``rewrap`` turns the backend result back into a frontend tensor.

    The representation is **inferred from** ``x0``'s type (like the library-wide ragged/uniform dispatch);
    the geometry must match -- a ``UniformTuckerTensorTrain`` x0 pairs with a uniform geometry, a
    ``TuckerTensorTrain`` x0 with a ragged one -- else a structural error. ``x0_cores`` may have smaller
    ranks than ``x0`` (the same tensor): on a manifold geometry a ragged ``x0`` is reduced to minimal ranks,
    and a uniform ``x0`` always is (:py:func:`~t3toolbox.backend.uniform_fitting.uniform_minimal`; the
    uniform path needs a minimal-rank frame), so a frontend user never meets that requirement. A uniform
    ``rewrap`` carries the reduced point's ``shape`` + ``masks``. ``chunk_size`` and ``batch`` are as in
    the optimizers (:doc:`/chunking`).

    ``hessian`` (``'gauss_newton'`` or ``'exact'``, case-insensitive) is the mode the problem will be
    solved in. It does not fix the mode -- that is the local model's and the solver's own ``hessian``
    argument -- it sizes ``chunk_size='auto'``: the chunk is a field of the sampling kind, fixed here, and
    with the exact Hessian the same chunk also drives its two-hole assemblies, so it is sized for those
    too. Pass the solver's value (:py:func:`newton_cg` does).

    Examples
    --------
    Build the problem of a ragged ``probe`` fit, take the backend local model at the start, and read off
    its gradient -- raw ``(tucker_variations, tt_variations)`` tuples, shaped like the frame's cores:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.optimizers as optimizers
    >>> np.random.seed(0)
    >>> A = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(30, N) for N in (4, 5, 6)]
    >>> x0 = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
    >>> setup = optimizers.least_squares_problem(t3m.MANIFOLD, 'probe', ww, A.probe(ww), x0)
    >>> type(setup.problem).__name__
    'Problem'
    >>> g_tucker, g_tt = setup.problem.local_model(setup.x0_cores).gradient
    >>> [V.shape for V in g_tucker], [H.shape for H in g_tt]
    ([(2, 4), (2, 5), (2, 6)], [(1, 2, 2), (2, 2, 2), (2, 2, 1)])

    The same ``problem`` drives a backend optimizer, and ``rewrap`` returns its result as a frontend tensor:

    >>> import t3toolbox.backend.optimizers as bopt
    >>> x_cores, stats = bopt.newton_cg(setup.problem, setup.x0_cores, max_newton=20)
    >>> x_opt = setup.rewrap(x_cores)
    >>> type(x_opt).__name__, stats['stop_reason']
    ('TuckerTensorTrain', 'gradient')
    >>> bool(np.linalg.norm(x_opt.to_dense() - A.to_dense()) < 1e-6 * np.linalg.norm(A.to_dense()))
    True

    For exact Newton pass the mode to both -- here on ``probe_derivatives`` (the solver's trust region is the
    exact Hessian's default step control):

    >>> pp = [np.random.randn(30, N) for N in (4, 5, 6)]
    >>> esetup = optimizers.least_squares_problem(t3m.MANIFOLD, 'probe_derivatives', (ww, pp),
    ...                                           A.probe_derivatives(ww, pp, 1), x0, order=1, hessian='exact')
    >>> x_cores, stats = bopt.newton_cg(esetup.problem, esetup.x0_cores, max_newton=40, hessian='exact')
    >>> stats['history'][0]['step_control'], stats['stop_reason']
    ('trust_region', 'gradient')

    A uniform ``x0`` (with a uniform geometry) gives the packed problem; ``x0_cores`` is then the bare
    supercore pair:

    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> usetup = optimizers.least_squares_problem(ut3m.UNIFORM_MANIFOLD, 'probe', ww, A.probe(ww),
    ...                                           ut3.UniformTuckerTensorTrain.from_t3(x0))
    >>> [sc.shape for sc in usetup.x0_cores], type(usetup.rewrap(usetup.x0_cores)).__name__
    ([(3, 2, 6), (3, 2, 2, 2)], 'UniformTuckerTensorTrain')

    A geometry of the other representation is a structural error:

    >>> ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)
    >>> optimizers.least_squares_problem(t3m.MANIFOLD, 'probe', ww, A.probe(ww), ux0)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: a UniformTuckerTensorTrain x0 requires a uniform geometry
    """
    kind = kind.lower()                                           # 'Probe', 'PROBE_DERIVATIVES', ... accepted
    _check_kind(kind, order)
    hessian = bopt._check_hessian(hessian, 'least_squares_problem')   # structural, before any work
    if weight is not None and kind in ('apply', 'entries'):      # plain apply/entries: no axis to weight
        raise ValueError(f"the plain '{kind}' kind takes no residual weight (no mode or order axis); "
                         "per-mode weighting is defined for probe, per-order for the derivative kinds.")
    wm = _fitting._canonical_weight(weight, kind, _n_modes(kind, sample), order or 0)   # 2-D ω[m,o] or None

    if isinstance(x0, ut3.UniformTuckerTensorTrain):
        geom_name, sharing_spec = _uniform_geometry_name(geometry)
        x0m = uf.uniform_minimal(x0, sharing=sharing_spec)   # transparent SHARED-minimal reduction (no-op if minimal)
        cs = _resolve_chunk_size(chunk_size, kind, x0m, sample, order, batch, hessian)   # 'auto' -> balanced
        problem = uf.uniform_least_squares_problem(geom_name, kind, x0m, sample, data, order, wm, regularizer,
                                                   chunk_size=cs, sharing=sharing_spec)
        init = (x0m.tucker_supercore, x0m.tt_supercore)  # optimizer state = the bare supercore pair
        return LeastSquaresSetup(problem, init,
                                 lambda sc: ut3.UniformTuckerTensorTrain(sc[0], sc[1], x0m.shape, x0m.masks))

    if isinstance(x0, t3.TuckerTensorTrain):
        if _is_manifold_geometry(geometry):
            x0 = _ragged_minimal(x0, getattr(geometry, 'sharing', None))   # mirror of uniform_minimal (no-op if minimal)
        if kind in _KIND:                                # plain kinds: only probe is weightable (per-mode)
            bk = bfit.probe_kind(wm) if kind == 'probe' and wm is not None else _KIND[kind]
        elif kind == 'probe_derivatives':                # the only derivative kind with a chunkable 𝒥ᵀ assembly
            cs = _resolve_chunk_size(chunk_size, kind, x0, sample, order, batch, hessian)   # ragged -> None (no-op)
            bk = _DERIV_KIND[kind](order, wm, chunk_size=cs)
        else:
            bk = _DERIV_KIND[kind](order, wm)
        problem = bopt.least_squares_problem(_geometry_ops(geometry, x0.shape), bk, sample, data,
                                             regularizer=regularizer)
        return LeastSquaresSetup(problem, x0.data, lambda cores: t3.TuckerTensorTrain(*cores))

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
        g0_reference: typ.Optional[str] = None,   # the stop's ‖g0‖: None = the geometry's own ('zero' on manifolds); 'zero' | 'warm_start'
        **kwargs,                       # forwarded to backend.optimizers.gradient_descent (n_iter, gtol_rel, ...)
) -> typ.Tuple[Point, dict]:            # (x_opt, stats)
    """Fit ``x`` to ``data`` by steepest descent (Cauchy step + Armijo line search) on ``geometry``.

    Accepts a ragged ``TuckerTensorTrain`` (with ``manifold.MANIFOLD`` / ``COREWISE``) or a uniform
    ``UniformTuckerTensorTrain`` (with ``uniform_manifold.UNIFORM_MANIFOLD`` / ``UNIFORM_COREWISE``); the
    representation is inferred from ``x0`` and returned in kind. Pass ``regularizer`` (e.g.
    ``optimizers.IdentityRegularizer(λ)``) to add ``ρ(x)`` to the objective (either representation).
    **Eager only**: there is no ``use_jit`` here (that kwarg lives on :py:func:`mc_sgd` / :py:func:`adam` /
    :py:func:`newton_cg`); this is the reference/pedagogical optimizer, ``newton_cg`` the performance path.
    ``on_line_search_failure`` ('stop' default / 'accept') controls Armijo-exhaustion behavior as in
    :py:func:`newton_cg`, and ``g0_reference`` picks the reference ``‖g0‖`` of the relative stop as there
    (by default the zero tensor of ``x0``'s ranks on a manifold geometry). See
    :py:func:`t3toolbox.backend.optimizers.gradient_descent`."""
    setup = least_squares_problem(geometry, kind, sample, data, x0, order, weight, regularizer, chunk_size=chunk_size)
    x_cores, stats = bopt.gradient_descent(setup.problem, setup.x0_cores, g0_reference=g0_reference, **kwargs)
    return setup.rewrap(x_cores), stats


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
        use_jit:  bool = False,         # jit the per-step kernel: auto-converts x0/sample/data to jax -> a jax-backed float32 result; warns + runs eager if jax absent
        regularizer: typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ); scaled by batch/n per step
        chunk_size: typ.Any = 'auto',   # probe_derivatives 𝒥ᵀ memory chunk; 'auto' -> estimate_chunk_size (docs/chunking.md)
        **kwargs,                       # forwarded to backend.optimizers.mc_sgd (max_iter, check_every, ...)
) -> typ.Tuple[Point, dict]:
    """Manifold Cauchy SGD -- minibatched, tuning-free Cauchy step. Ragged or uniform ``x0`` (see
    :py:func:`gradient_descent`). A ``regularizer`` is scaled by ``batch/n`` per step so
    ``λ`` matches the full-batch optimizers. See :py:func:`t3toolbox.backend.optimizers.mc_sgd`."""
    setup = least_squares_problem(geometry, kind, sample, data, x0, order, weight, regularizer, chunk_size=chunk_size,
                                  batch=batch)
    x_cores, stats = bopt.mc_sgd(setup.problem, setup.x0_cores, rng, batch, draw=draw, use_jit=use_jit, **kwargs)
    return setup.rewrap(x_cores), stats


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
        use_jit:  bool = False,         # jit the per-step kernel: auto-converts x0/sample/data to jax -> a jax-backed float32 result; warns + runs eager if jax absent
        regularizer: typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ); scaled by batch/n per step
        chunk_size: typ.Any = 'auto',   # probe_derivatives 𝒥ᵀ memory chunk; 'auto' -> estimate_chunk_size (docs/chunking.md)
        **kwargs,                       # forwarded to backend.optimizers.adam (lr, max_iter, ...)
) -> typ.Tuple[Point, dict]:
    """Adam over the cores -- the dependency-free first-order method for the corewise geometry. Ragged or
    uniform ``x0`` (see :py:func:`gradient_descent`). A ``regularizer`` is scaled by
    ``batch/n`` per step so ``λ`` matches the full-batch optimizers. See
    :py:func:`t3toolbox.backend.optimizers.adam`."""
    if _is_manifold_geometry(geometry):
        warnings.warn("adam on a MANIFOLD geometry: Adam's per-coordinate moments live in the core "
                      "coordinates, which are gauge-dependent, so the iterates depend on the backend / "
                      "representation (measured ~1e-2 between numpy, jax and uniform). It runs, but COREWISE "
                      "is the intended geometry for adam; mc_sgd / newton_cg are the gauge-invariant "
                      "manifold optimizers.", RuntimeWarning, stacklevel=2)
    setup = least_squares_problem(geometry, kind, sample, data, x0, order, weight, regularizer, chunk_size=chunk_size,
                                  batch=batch)
    x_cores, stats = bopt.adam(setup.problem, setup.x0_cores, rng, batch, draw=draw, use_jit=use_jit, **kwargs)
    return setup.rewrap(x_cores), stats


def newton_cg(
        geometry,                   # ragged/uniform MANIFOLD (intended) / COREWISE (must match x0)
        kind:             str,      # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample:           typ.Any,  # ww / index, or (ww, pp) / (index, pp) for derivatives
        data:             typ.Any,  # observed values to fit
        x0:               Point,    # initial point (zero is fine on the manifold)

        order:            typ.Optional[int] = None,      # derivative kinds: highest order (required)
        weight:           typ.Optional[typ.Any] = None,  # residual weight ω: per-mode (probe) / ω[mode,order] (derivatives)

        verbose:          bool = False,    # print per-iteration diagnostics (a relative-error table + more)
        val_sample:       typ.Any = None,  # optional validation sample (same layout as `sample`) -> a train|val table
        val_data:         typ.Any = None,  # optional validation data (both given adds the val column)

        callback:         typ.Optional[typ.Callable] = None,  # custom callback(NewtonInfo) each iter (overrides `verbose`)

        use_jit:          bool = False,      # jit the inner CG: auto-converts x0/sample/data to jax -> a jax-backed float32 result; warns + runs eager if jax absent
        regularizer:      typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
        chunk_size:       typ.Any = 'auto',  # probe_derivatives 𝒥ᵀ memory chunk; 'auto' -> estimate_chunk_size (docs/chunking.md)

        g0_reference:     typ.Optional[str]   = None,  # which ‖g0‖: None = the geometry's own ('zero' on manifolds, 'warm_start' on corewise); 'zero' | 'warm_start'
        g0norm_newton:    typ.Optional[float] = None,  # explicit ‖g0‖ for the Newton stop (wins; > 0); also feeds CG unless g0norm_cg is given
        g0norm_cg:        typ.Optional[float] = None,  # explicit ‖g0‖ for the CG forcing term (wins for CG; > 0); default = the Newton reference

        cg_forcing_power: float = 0.5,   # η = min(0.5, (‖g‖/‖g0‖)**power); larger => tighter CG, fewer Newton steps
        max_newton:       int   = 30,    # Newton-iteration cap (rejected trust-region trials included)
        gtol_rel:         float = 1e-8,  # stop when ‖g‖ <= gtol_rel * ‖g0‖
        cg_maxiter:       int   = 200,   # inner CG iteration cap per Newton step
        c_armijo:         float = 1e-4,  # Armijo sufficient-decrease constant (line search only)
        on_line_search_failure: str = 'stop',  # Armijo exhausted: 'stop' = reject the step and terminate; 'accept' = take the last trial

        hessian:          str = 'gauss_newton',               # or 'exact' (case-insensitive): the local model's Hessian
        curvature_damping: typ.Optional[float] = None,        # exact only: lift damping, relative to ‖X‖; None = the default
        step_control:     typ.Optional[str]   = None,         # None = 'line_search' (GN) / 'trust_region' (exact)
        tr_radius0:       typ.Optional[float] = None,         # trust region only: Δ0 > 0; None = the GN Cauchy length
) -> typ.Tuple[Point, dict]:
    """Inexact Riemannian Newton-CG with a line search or a trust region -- the manifold workhorse. Ragged or
    uniform ``x0`` (see :py:func:`gradient_descent`). See :py:func:`t3toolbox.backend.optimizers.newton_cg`.

    ``hessian`` picks the local model's Hessian: ``'gauss_newton'`` (the default) or ``'exact'``, the exact
    Riemannian Hessian -- the Gauss-Newton term plus the curvature term the Gauss-Newton model drops, which
    gives the superlinear Newton tail on a nonzero-residual fit (q-quadratic with ``cg_forcing_power=1.0``)
    at a few times the cost of a Gauss-Newton Hessian-apply. Its default step control is a trust region
    (``step_control='trust_region'``: Steihaug-Toint CG, one trial retraction per step, accepted on the
    model's predicted decrease), the Gauss-Newton one the Armijo line search; either combines with either.
    ``tr_radius0`` is the initial trust radius and ``curvature_damping`` the exact Hessian's lift damping
    (relative to ``‖X‖``; ``None`` = the geometry's default); each raises where it would have no effect. The
    exact Hessian is implemented for every sampling kind, on every geometry (``MANIFOLD`` / ``COREWISE``, their
    uniform twins, and the shared wrappers of all four). ``hessian`` also sizes ``chunk_size='auto'``
    (:py:func:`least_squares_problem`). Details,
    stop reasons and the trust-region rules: the backend :py:func:`t3toolbox.backend.optimizers.newton_cg`.

    Both stopping tests are relative to a reference ‖g0‖. By default (``g0_reference=None``) a manifold
    geometry uses the gradient norm at the **zero tensor of x0's ranks** -- the right scale after a
    warm start, whose own initial gradient is deflated -- and a corewise geometry the warm-start ‖g(x0)‖;
    ``g0_reference='warm_start'`` keeps the 2026.2.0 behaviour. The explicit overrides ``g0norm_newton`` /
    ``g0norm_cg`` win, and ``cg_forcing_power`` trades more CG iterations per Newton step for fewer Newton
    steps (all detailed on the backend; ``docs/fitting_and_optimization.md`` §5).

    The solver options -- ``max_newton``, ``gtol_rel``, ``cg_maxiter``, ``c_armijo``,
    ``on_line_search_failure``, the references above, ``cg_forcing_power``, ``hessian``,
    ``curvature_damping``, ``step_control`` and ``tr_radius0`` -- are the backend's, with the backend's
    defaults, passed straight through. They are explicit keywords (there is no ``**kwargs`` pass-through),
    so a misspelt option raises ``TypeError`` at the call, before the problem is built.

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

    **The exact Hessian.** On a noisy ``probe_derivatives`` fit warm-started near the solution, the exact
    Hessian (with its default trust region, and ``cg_forcing_power=1.0`` for the q-quadratic tail) reaches
    the gradient test in a fraction of the steps Gauss-Newton takes (linear there, it ends here on its line
    search, short of this tight ``gtol_rel``):

    >>> import t3toolbox.manifold as t3m
    >>> np.random.seed(0)
    >>> shape, rng = (5, 5, 5), np.random.default_rng(0)
    >>> ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
    >>> target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    >>> data = [z + 0.01 * rng.standard_normal(z.shape) for z in target.probe_derivatives(ww, pp, 1)]
    >>> x0, _ = optimizers.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), data,
    ...                              t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)), order=1,
    ...                              max_newton=5)
    >>> opts = dict(order=1, max_newton=40, g0_reference='warm_start', gtol_rel=1e-10)
    >>> _, gn = optimizers.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), data, x0, **opts)
    >>> _, ex = optimizers.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), data, x0, hessian='exact',
    ...                              cg_forcing_power=1.0, **opts)
    >>> ex['history'][0]['step_control'], ex['stop_reason']
    ('trust_region', 'gradient')
    >>> gn['newton_steps'] > 2 * ex['newton_steps']
    True

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
    bopt._check_hessian(hessian, 'newton_cg')            # structural, named for the caller, before the setup work
    setup = least_squares_problem(geometry, kind, sample, data, x0, order, weight, regularizer, chunk_size=chunk_size,
                                  hessian=hessian)
    records = None
    if (val_sample is None) != (val_data is None):
        raise ValueError("newton_cg: pass val_sample and val_data together, or neither (val_sample=%s, "
                         "val_data=%s)" % ('None' if val_sample is None else 'given',
                                            'None' if val_data is None else 'given'))
    if callback is None and verbose:
        vs, vd = val_sample, val_data
        if isinstance(x0, ut3.UniformTuckerTensorTrain) and val_data is not None:
            vs = uf.pack_sample(kind, val_sample, x0.N)   # pack validation to the packed kind's width
            vd = uf.pack_data(kind, val_data, x0.N)
        callback, records = bdisp.make_newton_display(setup.problem, val_sample=vs, val_data=vd)
    x_cores, stats = bopt.newton_cg(setup.problem, setup.x0_cores,
                                    max_newton=max_newton, gtol_rel=gtol_rel, cg_maxiter=cg_maxiter,
                                    c_armijo=c_armijo, on_line_search_failure=on_line_search_failure,
                                    g0_reference=g0_reference, g0norm_newton=g0norm_newton, g0norm_cg=g0norm_cg,
                                    cg_forcing_power=cg_forcing_power, use_jit=use_jit, callback=callback,
                                    hessian=hessian, curvature_damping=curvature_damping,
                                    step_control=step_control, tr_radius0=tr_radius0)
    if records is not None:
        stats['diagnostics'] = records
    return setup.rewrap(x_cores), stats
