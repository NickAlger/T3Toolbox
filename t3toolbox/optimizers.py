# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Frontend adapter for the geometry-agnostic optimizers.

A thin wrapper over :py:mod:`t3toolbox.backend.optimizers` (where the algorithms live -- check-free, so a
raw-``.data`` user can call them directly). This layer (1) **validates the input once** (structural always;
numerical preconditions in safe mode -- here the input is just a well-formed ``TuckerTensorTrain``, since
the geometry re-builds/orthogonalizes the frame internally each step), (2) maps the frontend geometry
singletons (``manifold.MANIFOLD`` / ``manifold.COREWISE``) and the sampling-kind name to the backend
``GeometryOps`` / ``SamplingKind``, (3) calls the backend optimizer on the raw cores, and (4) re-wraps the
result as a ``TuckerTensorTrain``. Design: ``docs/optimizers_plan.md``.

    >>> # x_opt, stats = optimizers.gradient_descent(MANIFOLD, 'probe', ww, data, x0)
"""
import typing as typ

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.fitting as bfit

__all__ = [
    'gradient_descent',
    'mc_sgd',
    'adam',
    'newton_cg',
]

_KIND = {'apply': bfit.APPLY, 'entries': bfit.ENTRIES, 'probe': bfit.PROBE}


def _geometry_ops(geometry) -> bopt.GeometryOps:
    """Map a frontend geometry singleton to its backend ``GeometryOps`` (check-free)."""
    if geometry is t3m.MANIFOLD:
        return bopt.MANIFOLD
    if geometry is t3m.COREWISE:
        return bopt.COREWISE
    raise ValueError(f"unknown geometry {geometry!r}; expected manifold.MANIFOLD or manifold.COREWISE")


def _problem(
        geometry,           # t3m.MANIFOLD / t3m.COREWISE
        kind:   str,        # 'apply' / 'entries' / 'probe'
        sample,             # ww (apply/probe) or index (entries)
        data,               # observed S(x_true) (+ noise)
) -> bopt.Problem:
    if kind not in _KIND:
        raise ValueError(f"unknown sampling kind {kind!r}; expected one of {sorted(_KIND)}")
    return bopt.least_squares_problem(_geometry_ops(geometry), _KIND[kind], sample, data)


def gradient_descent(
        geometry,                       # t3m.MANIFOLD / t3m.COREWISE
        kind:     str,                  # 'apply' / 'entries' / 'probe'
        sample:   typ.Any,              # ww (apply/probe) or index (entries)
        data:     typ.Any,             # observed values to fit
        x0:       t3.TuckerTensorTrain, # initial point (any cores; the geometry orthogonalizes internally)
        **kwargs,                       # forwarded to backend.optimizers.gradient_descent (n_iter, gtol_rel, ...)
) -> typ.Tuple[t3.TuckerTensorTrain, dict]:   # (x_opt, stats)
    """Fit ``x`` to ``data`` by steepest descent (Cauchy step + Armijo line search) on ``geometry``.
    See :py:func:`t3toolbox.backend.optimizers.gradient_descent`."""
    problem = _problem(geometry, kind, sample, data)
    x_cores, stats = bopt.gradient_descent(problem, x0.data, **kwargs)
    return t3.TuckerTensorTrain(*x_cores), stats


def mc_sgd(
        geometry,                       # t3m.MANIFOLD (intended) / t3m.COREWISE
        kind:     str,                  # 'apply' / 'entries' / 'probe'
        sample:   typ.Any,              # ww or index
        data:     typ.Any,             # observed values to fit
        x0:       t3.TuckerTensorTrain, # initial point
        rng,                            # np.random.Generator -- minibatch draws
        batch:    int,                  # samples per minibatch
        **kwargs,                       # forwarded to backend.optimizers.mc_sgd
) -> typ.Tuple[t3.TuckerTensorTrain, dict]:
    """Manifold Cauchy SGD -- minibatched, tuning-free Cauchy step.
    See :py:func:`t3toolbox.backend.optimizers.mc_sgd`."""
    problem = _problem(geometry, kind, sample, data)
    x_cores, stats = bopt.mc_sgd(problem, x0.data, rng, batch, **kwargs)
    return t3.TuckerTensorTrain(*x_cores), stats


def adam(
        geometry,                       # t3m.COREWISE (intended) / t3m.MANIFOLD
        kind:     str,                  # 'apply' / 'entries' / 'probe'
        sample:   typ.Any,              # ww or index
        data:     typ.Any,             # observed values to fit
        x0:       t3.TuckerTensorTrain, # initial point
        rng,                            # np.random.Generator -- minibatch draws
        batch:    int,                  # samples per minibatch
        **kwargs,                       # forwarded to backend.optimizers.adam (lr, max_iter, ...)
) -> typ.Tuple[t3.TuckerTensorTrain, dict]:
    """Adam over the cores -- the dependency-free first-order method for the corewise geometry.
    See :py:func:`t3toolbox.backend.optimizers.adam`."""
    problem = _problem(geometry, kind, sample, data)
    x_cores, stats = bopt.adam(problem, x0.data, rng, batch, **kwargs)
    return t3.TuckerTensorTrain(*x_cores), stats


def newton_cg(
        geometry,                       # t3m.MANIFOLD (intended) / t3m.COREWISE
        kind:     str,                  # 'apply' / 'entries' / 'probe'
        sample:   typ.Any,              # ww or index
        data:     typ.Any,             # observed values to fit
        x0:       t3.TuckerTensorTrain, # initial point (zero is fine on the manifold)
        **kwargs,                       # forwarded to backend.optimizers.newton_cg (max_newton, use_jit, ...)
) -> typ.Tuple[t3.TuckerTensorTrain, dict]:
    """Inexact Riemannian Newton-CG with an Armijo line search -- the manifold workhorse.
    See :py:func:`t3toolbox.backend.optimizers.newton_cg`."""
    problem = _problem(geometry, kind, sample, data)
    x_cores, stats = bopt.newton_cg(problem, x0.data, **kwargs)
    return t3.TuckerTensorTrain(*x_cores), stats
