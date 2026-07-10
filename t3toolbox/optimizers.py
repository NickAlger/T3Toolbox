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
  **minimal-rank base**; it calls :py:func:`~t3toolbox.backend.uniform_fitting.uniform_minimal`
  transparently so a frontend user never meets that requirement.

The geometry must match ``x0``'s representation (a uniform x0 with a ragged geometry, or vice versa, is a
structural error). The result is returned in the same representation as ``x0``.

    >>> # x_opt, stats = optimizers.gradient_descent(MANIFOLD, 'probe', ww, data, x0)           # ragged
    >>> # x_opt, stats = optimizers.newton_cg(UNIFORM_MANIFOLD, 'probe', ww, data, ux0)         # uniform
"""
import typing as typ

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf

__all__ = [
    'gradient_descent',
    'mc_sgd',
    'adam',
    'newton_cg',
]

_KIND = {'apply': bfit.APPLY, 'entries': bfit.ENTRIES, 'probe': bfit.PROBE}
_DERIV_KIND = {'apply_derivatives':   bfit.apply_derivatives_kind,
               'entries_derivatives': bfit.entries_derivatives_kind,
               'probe_derivatives':   bfit.probe_derivatives_kind}

# A frontend point in either representation. The four optimizers infer ragged (TuckerTensorTrain) vs
# uniform (UniformTuckerTensorTrain) from x0's type, and require a matching geometry singleton -- see _setup.
Point = typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain]


def _geometry_ops(geometry) -> bopt.GeometryOps:
    """Map a **ragged** frontend geometry singleton to its backend ``GeometryOps`` (check-free)."""
    if geometry is t3m.MANIFOLD:
        return bopt.MANIFOLD
    if geometry is t3m.COREWISE:
        return bopt.COREWISE
    raise ValueError(f"unknown geometry {geometry!r}; expected manifold.MANIFOLD / manifold.COREWISE "
                     f"(or the uniform singletons uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE, "
                     f"with a UniformTuckerTensorTrain x0)")


def _uniform_geometry_name(geometry) -> str:
    """Map a **uniform** frontend geometry singleton to the backend geometry name (``'manifold'`` / ``'corewise'``)."""
    if geometry is ut3m.UNIFORM_MANIFOLD:
        return 'manifold'
    if geometry is ut3m.UNIFORM_COREWISE:
        return 'corewise'
    raise ValueError(f"a UniformTuckerTensorTrain x0 requires a uniform geometry "
                     f"(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE), got {geometry!r}")


def _check_kind(kind: str, order: typ.Optional[int]) -> None:
    """Validate the sampling-kind name (shared across representations); derivative kinds need ``order``."""
    if kind not in _KIND and kind not in _DERIV_KIND:
        raise ValueError(f"unknown sampling kind {kind!r}; expected one of "
                         f"{sorted(_KIND) + sorted(_DERIV_KIND)}")
    if kind in _DERIV_KIND and order is None:
        raise ValueError(f"derivative kind {kind!r} requires order=")


def _setup(
        geometry,           # ragged (t3m.MANIFOLD/COREWISE) or uniform (ut3m.UNIFORM_MANIFOLD/COREWISE) singleton
        kind:   str,        # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample,             # ww / index (regular) or (ww, pp) / (index, pp) (derivatives)
        data,               # observed S(x_true) (+ noise)
        x0,                 # TuckerTensorTrain (ragged) or UniformTuckerTensorTrain (uniform)
        order:  typ.Optional[int]                 = None,  # derivative kinds only: highest order (required)
        weight: typ.Optional[typ.Sequence[float]] = None,  # derivative kinds only: per-order residual weight ω
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
    with the base's held ``shape`` + ``masks``.
    """
    _check_kind(kind, order)

    if isinstance(x0, ut3.UniformTuckerTensorTrain):
        geom_name = _uniform_geometry_name(geometry)
        x0m = uf.uniform_minimal(x0)                     # transparent minimal-rank reduction (no-op if minimal)
        problem = uf.uniform_least_squares_problem(geom_name, kind, x0m, sample, data, order, weight)
        init = (x0m.tucker_supercore, x0m.tt_supercore)  # optimizer state = the bare supercore pair
        return problem, init, lambda sc: ut3.UniformTuckerTensorTrain(sc[0], sc[1], x0m.shape, x0m.masks)

    if isinstance(x0, t3.TuckerTensorTrain):
        bk = _KIND[kind] if kind in _KIND else _DERIV_KIND[kind](order, weight)
        problem = bopt.least_squares_problem(_geometry_ops(geometry), bk, sample, data)
        return problem, x0.data, lambda cores: t3.TuckerTensorTrain(*cores)

    raise TypeError(f"x0 must be a TuckerTensorTrain or UniformTuckerTensorTrain, got {type(x0).__name__}")


# Derivative kinds (kind='*_derivatives') need `order` (+ optional per-order weight ω) and a paired
# `(ww, pp)` / `(index, pp)` sample; everything else is identical. `order`/`weight` build the kind;
# `draw` (mc_sgd / adam) is the custom minibatch draw (None = the flat default).
def gradient_descent(
        geometry,                       # ragged or uniform geometry singleton (must match x0's representation)
        kind:     str,                  # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample:   typ.Any,              # ww / index, or (ww, pp) / (index, pp) for derivatives
        data:     typ.Any,             # observed values to fit
        x0:       Point,                # initial point (any cores; the geometry orthogonalizes internally)
        order:    typ.Optional[int]                 = None,  # derivative kinds: highest order (required)
        weight:   typ.Optional[typ.Sequence[float]] = None,  # derivative kinds: per-order residual weight ω
        **kwargs,                       # forwarded to backend.optimizers.gradient_descent (n_iter, gtol_rel, ...)
) -> typ.Tuple[Point, dict]:            # (x_opt, stats)
    """Fit ``x`` to ``data`` by steepest descent (Cauchy step + Armijo line search) on ``geometry``.

    Accepts a ragged ``TuckerTensorTrain`` (with ``manifold.MANIFOLD`` / ``COREWISE``) or a uniform
    ``UniformTuckerTensorTrain`` (with ``uniform_manifold.UNIFORM_MANIFOLD`` / ``UNIFORM_COREWISE``); the
    representation is inferred from ``x0`` and returned in kind. See
    :py:func:`t3toolbox.backend.optimizers.gradient_descent`."""
    problem, init, rewrap = _setup(geometry, kind, sample, data, x0, order, weight)
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
        order:    typ.Optional[int]                 = None,
        weight:   typ.Optional[typ.Sequence[float]] = None,
        draw:     typ.Optional[typ.Callable]        = None,  # custom draw(rng)->(sample_B,data_B); None = flat
        **kwargs,                       # forwarded to backend.optimizers.mc_sgd
) -> typ.Tuple[Point, dict]:
    """Manifold Cauchy SGD -- minibatched, tuning-free Cauchy step. Ragged or uniform ``x0`` (see
    :py:func:`gradient_descent`). See :py:func:`t3toolbox.backend.optimizers.mc_sgd`."""
    problem, init, rewrap = _setup(geometry, kind, sample, data, x0, order, weight)
    x_cores, stats = bopt.mc_sgd(problem, init, rng, batch, draw=draw, **kwargs)
    return rewrap(x_cores), stats


def adam(
        geometry,                       # ragged/uniform COREWISE (intended) / MANIFOLD (must match x0)
        kind:     str,                  # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample:   typ.Any,              # ww / index, or (ww, pp) / (index, pp) for derivatives
        data:     typ.Any,             # observed values to fit
        x0:       Point,                # initial point
        rng,                            # np.random.Generator -- passed to the draw each step
        batch:    int,                  # measurements per minibatch (default flat draw; ignored if draw given)
        order:    typ.Optional[int]                 = None,
        weight:   typ.Optional[typ.Sequence[float]] = None,
        draw:     typ.Optional[typ.Callable]        = None,
        **kwargs,                       # forwarded to backend.optimizers.adam (lr, max_iter, ...)
) -> typ.Tuple[Point, dict]:
    """Adam over the cores -- the dependency-free first-order method for the corewise geometry. Ragged or
    uniform ``x0`` (see :py:func:`gradient_descent`). See :py:func:`t3toolbox.backend.optimizers.adam`."""
    problem, init, rewrap = _setup(geometry, kind, sample, data, x0, order, weight)
    x_cores, stats = bopt.adam(problem, init, rng, batch, draw=draw, **kwargs)
    return rewrap(x_cores), stats


def newton_cg(
        geometry,                       # ragged/uniform MANIFOLD (intended) / COREWISE (must match x0)
        kind:     str,                  # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        sample:   typ.Any,              # ww / index, or (ww, pp) / (index, pp) for derivatives
        data:     typ.Any,             # observed values to fit
        x0:       Point,                # initial point (zero is fine on the manifold)
        order:    typ.Optional[int]                 = None,
        weight:   typ.Optional[typ.Sequence[float]] = None,
        **kwargs,                       # forwarded to backend.optimizers.newton_cg (max_newton, use_jit, ...)
) -> typ.Tuple[Point, dict]:
    """Inexact Riemannian Newton-CG with an Armijo line search -- the manifold workhorse. Ragged or uniform
    ``x0`` (see :py:func:`gradient_descent`). See :py:func:`t3toolbox.backend.optimizers.newton_cg`.

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
    """
    problem, init, rewrap = _setup(geometry, kind, sample, data, x0, order, weight)
    x_cores, stats = bopt.newton_cg(problem, init, **kwargs)
    return rewrap(x_cores), stats
