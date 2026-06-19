# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
'''Frontend for the least-squares fitting layer: the :py:class:`GaussNewtonModel`.

A thin frozen-dataclass frontend over ``backend/fitting.py``. A ``GaussNewtonModel`` is the local
Gauss-Newton model of a least-squares objective at a fixed base point (an orthonormal ``T3Basis``): it
exposes the objective value ``c``, the Riemannian gradient ``g``, the Gauss-Newton Hessian action
``H p = JᵀJ p``, and the quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p``.

The expensive base sweep (the base-and-data edge variables, ``probing.precompute_apply_base_sweep``) is a
``@cached_property``: computed once on first use and reused across every ``gn_hessian`` / ``evaluate`` of
an inner solve, then garbage-collected when the model leaves scope. This is the whole point -- in an inner
CG the base is fixed, so the sweep is computed once, not once per matrix-vector product.

Gauge handling lives entirely in the backend (``J = 𝒥∘Π``, ``Jᵀ = Π∘𝒥ᵀ``): every method returns a gauged
tangent and accepts any-gauge inputs. See ``docs/fitting_plan.md``.
'''

from __future__ import annotations

import functools as ft
import typing as typ
from dataclasses import dataclass

import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.probing as probing
import t3toolbox.backend.fitting as fb
from t3toolbox.backend.common import *

__all__ = ['GaussNewtonModel']


@dataclass(frozen=True)
class GaussNewtonModel:
    '''The local Gauss-Newton model of an all-modes ``apply`` least-squares objective at ``base``.

    Holds the base (an orthonormal frame), the sample vectors ``ww``, and the data residual
    ``r = F(base) − y``. The base sweep, objective value, and gradient are cached on first use.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.fitting as fitting
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> base, _ = bvf.t3_orthogonal_representations(x)
    >>> ww = [np.random.randn(15, N) for N in (6, 7, 8)]    # 15 samples, each a (w_1, w_2, w_3)
    >>> r = np.random.randn(15)                             # the data residual at the base
    >>> model = fitting.GaussNewtonModel(base, ww, r)

    The objective value ``c = ½‖r‖²`` and the gradient ``g = Π 𝒥ᵀ r`` (a gauged tangent):

    >>> print(round(float(model.objective_value), 6))
    8.295196
    >>> model.gradient.is_gauged()
    True

    The Gauss-Newton Hessian action ``H p`` stays on the (gauged) tangent space:

    >>> p = t3m.T3Tangent.randn(base)                       # a gauged trial step
    >>> hp = model.gn_hessian(p)
    >>> hp.is_gauged()
    True

    ``evaluate(p)`` is the quadratic model value -- consistent with assembling it from the cached
    ``c`` / ``g`` and the Hessian action (and costs only one forward apply, not a Hessian apply):

    >>> m_eval = float(model.evaluate(p))
    >>> m_built = float(model.objective_value + model.gradient.inner(p) + 0.5 * p.inner(hp))
    >>> bool(np.allclose(m_eval, m_built))
    True

    A trial step at a *different* base is a structural error (the model is tied to its base):

    >>> other, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1)))
    >>> model.gn_hessian(t3m.T3Tangent.randn(other))   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: trial tangent must live at the model's base (same T3Basis object)
    '''

    base:     bvf.T3Basis              # the orthonormal frame -- the linearization point
    ww:       typ.Sequence[NDArray]    # sample vectors, len=d, elm_shape=W+(Ni,)
    residual: NDArray                  # r = F(base) − y, shape W+C

    @ft.cached_property
    def _base_sweep(self) -> typ.Tuple:  # (xis, mus, nus, etas) -- computed ONCE, reused
        return probing.precompute_apply_base_sweep(self.base.data, self.ww)

    @ft.cached_property
    def objective_value(self) -> NDArray:  # c = ½‖r‖², shape C
        '''The least-squares objective at the base, ``c = ½‖r‖²`` (the model constant ``m(0)``).'''
        n_w = self.ww[0].ndim - 1                      # sample-stack (W) axes; keep the base stack C
        return 0.5 * (self.residual ** 2).sum(axis=tuple(range(n_w)))

    @ft.cached_property
    def gradient(self) -> t3m.T3Tangent:  # g = Π 𝒥ᵀ r
        '''The Riemannian gradient ``g = Π 𝒥ᵀ r`` (the Gauss-Newton ``Jᵀr``), a gauged tangent at base.'''
        dU_dG = fb.compute_gradient(self.residual, self.ww, self.base.data, self._base_sweep)
        return t3m.T3Tangent(self.base, bvf.T3Variations(*dU_dG))

    def gn_hessian(self, p: t3m.T3Tangent) -> t3m.T3Tangent:
        '''The Gauss-Newton Hessian action ``H p = Π 𝒥ᵀ 𝒥 Π p`` (``H = JᵀJ``), a gauged tangent at base.'''
        self._check_at_base(p)
        dU_dG = fb.apply_gn_hessian(p.variations.data, self.ww, self.base.data, self._base_sweep)
        return t3m.T3Tangent(self.base, bvf.T3Variations(*dU_dG))

    def evaluate(self, p: t3m.T3Tangent) -> NDArray:  # m(p), shape C
        '''The quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p`` (one forward apply; reuses ``c``, ``g``).'''
        self._check_at_base(p)
        return fb.quadratic_model_value(
            p.variations.data, self.ww, self.base.data, self._base_sweep,
            self.gradient.variations.data, self.objective_value)

    def _check_at_base(self, p: t3m.T3Tangent) -> None:
        if p.basis is not self.base:
            raise ValueError('trial tangent must live at the model\'s base (same T3Basis object)')
