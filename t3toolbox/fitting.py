# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
'''Frontend for the least-squares fitting layer: one geometry-generic Gauss-Newton model.

A :py:class:`GaussNewtonModel` is the local Gauss-Newton model of a least-squares objective
``½‖F(X) − y‖²`` at a fixed point, linearized through a **geometry**: it exposes the objective value
``c = ½‖r‖²``, the gradient ``g``, the Gauss-Newton Hessian action ``H p = JᵀJ p``, and the
quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p``. The model is generic over two independent choices:

  * the **geometry** -- :py:data:`~t3toolbox.manifold.MANIFOLD` (optimize on the fixed-rank manifold;
    the gauge projection ``Π`` makes ``g`` / ``H`` Riemannian) or
    :py:data:`~t3toolbox.manifold.COREWISE` (optimize the raw cores; no ``Π``). The geometry supplies the
    frame ``base = geometry.base(X)`` and the projection ``Π = geometry.project``; the forward is
    ``J = 𝒥∘Π`` and the gradient is ``Jᵀr = Π∘𝒥ᵀr``. **Manifold ⟺ Π, corewise ⟺ no Π is structural**
    (bundled in the geometry), never a flag -- mixing the two silently corrupts the result.
  * the **sampling kind** -- bound by the factory :py:func:`apply_model` / :py:func:`entries_model` /
    :py:func:`probe_model` (the bare ``𝒥`` / ``𝒥ᵀ`` from :py:mod:`t3toolbox.backend.probing`, bundled in
    :py:mod:`t3toolbox.backend.fitting`).

Every input and output is a :py:class:`~t3toolbox.manifold.T3Tangent` at ``model.base`` -- including the
**corewise** case, where the tangent lives at the non-orthonormal frame ``(U,G,G,G)`` (a core
perturbation; see :py:class:`~t3toolbox.manifold.CorewiseGeometry`). Build trial steps at ``model.base``
(e.g. ``geometry.randn(model.base)``); a step at a different base is a structural error (the same
``T3Basis``-object guard as :py:class:`~t3toolbox.manifold.T3Tangent`).

The expensive base sweep (the base-and-data edge variables) is a ``@cached_property``: computed once on
first use and reused across every ``gn_hessian`` / ``evaluate`` of an inner solve. This is the whole
point -- in an inner CG the base is fixed, so the sweep is computed once, not once per matrix-vector
product. See :py:mod:`t3toolbox.backend.fitting` and ``docs/geometry_refactor_plan.md``.

Jitting an optimizer
--------------------
A ``GaussNewtonModel`` is **not** a registered jax pytree -- by design. It is a scope-local operator
built fresh at each outer step, and the two efficient ways to ``jit`` an optimizer never pass the model
(or its base) *across* a jit boundary, so they never trigger the per-base recompile that base-carrying
objects incur:

1. **Whole-step jit (per-step methods -- Cauchy / gradient descent / a fixed-step loop).** ``jit`` a
   step function whose only argument is the point ``X`` (a ``TuckerTensorTrain``, already a pytree);
   build the base, model, and gradient *inside*. The base is recomputed from the traced ``X`` each
   step, so it is traced (not a constant) and the function **compiles once and is reused across all
   steps** even though the base changes every step::

       @jax.jit
       def step(X):
           r     = X.apply(ww) - b
           model = fitting.apply_model(t3m.MANIFOLD, X, ww, r)
           g     = model.gradient
           alpha = g.inner(g) / model.gn_quadratic(g)     # cheap Cauchy step (one forward)
           return t3m.MANIFOLD.retract((-alpha) * g)
       X = x0
       for _ in range(n_steps):
           X = step(X)                                    # compiled once, reused

2. **Inner-solve jit (Newton-CG).** Build the model eagerly at the outer point, *close it over*, and
   ``jit`` only the inner matrix-vector product ``lambda p: model.gn_hessian(p)``. The base is then a
   closed-over constant -- fixed within the inner CG -- so the matvec compiles once per outer step and
   is reused across all CG iterations; the trial tangents ``p`` cross the boundary cheaply (``T3Tangent``
   carries its base as aux_data, identity-preserved). The per-outer-step recompile amortizes over the
   inner iterations. *(If the inner CG is very short, prefer pattern 1; see the basis-as-aux REVISIT
   note in ``manifold.py``.)*

The geometry singletons (``t3m.MANIFOLD`` / ``COREWISE``) are registered as zero-leaf pytrees, so they
may be closed over or passed as ordinary args freely.
'''

from __future__ import annotations

import functools as ft
import typing as typ
from dataclasses import dataclass

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.fitting as fb
from t3toolbox.backend.common import *

__all__ = ['GaussNewtonModel', 'apply_model', 'entries_model', 'probe_model']


def _require_at_base(base: bvf.T3Basis, p: t3m.T3Tangent) -> None:
    '''Structural guard: a trial tangent must live at the model's base -- the SAME ``T3Basis`` object
    (identity, not value equality, like ``T3Tangent``'s same-tangent-space guard).'''
    if p.basis is not base:
        raise ValueError("trial tangent must live at the model's base (same T3Basis object)")


@dataclass(frozen=True)
class GaussNewtonModel:
    '''The local Gauss-Newton model of a least-squares objective at ``base``, generic over the geometry.

    Built by a sampling-kind factory (:py:func:`apply_model` / :py:func:`entries_model` /
    :py:func:`probe_model`), not directly. Holds the geometry, the frame ``base = geometry.base(X)``, the
    sampling ``kind``, the measurement ``sample`` (the probe/apply vectors ``ww`` or the integer grid
    ``index``), and the data residual ``r = F(X) − y``. The base sweep, objective value, and gradient are
    cached on first use.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.fitting as fitting
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(15, N) for N in (6, 7, 8)]    # 15 samples, each a (w_1, w_2, w_3)
    >>> r = np.random.randn(15)                             # the data residual at x

    On the **manifold** geometry the gradient ``g = Π 𝒥ᵀ r`` is a gauged tangent and ``H p`` stays on the
    gauged tangent space:

    >>> model = fitting.apply_model(t3m.MANIFOLD, x, ww, r)
    >>> print(round(float(model.objective_value), 6))
    8.295196
    >>> model.gradient.is_gauged()
    True
    >>> p = t3m.MANIFOLD.randn(model.base)                  # a gauged trial step at the model's base
    >>> model.gn_hessian(p).is_gauged()
    True

    ``evaluate(p)`` is the quadratic-model value, consistent with assembling it from the cached ``c`` /
    ``g`` and the Hessian action (and costs only one forward apply, not a Hessian apply):

    >>> hp = model.gn_hessian(p)
    >>> m_built = float(model.objective_value + model.gradient.inner(p) + 0.5 * p.inner(hp))
    >>> bool(np.allclose(float(model.evaluate(p)), m_built))
    True

    The Gauss-Newton quadratic form ``pᵀHp = ‖Jp‖²`` comes from one forward sweep (no ``H p`` assembly) --
    the cheap step-length denominator for Cauchy / line search (``alpha = g.inner(g) / gn_quadratic(g)``):

    >>> bool(np.allclose(float(model.gn_quadratic(p)), float(p.inner(model.gn_hessian(p)))))
    True

    The **corewise** geometry is the same call with a different geometry: the gradient is a tangent at the
    raw ``(U,G,G,G)`` frame (a core perturbation), with **no** gauge projection:

    >>> cmodel = fitting.apply_model(t3m.COREWISE, x, ww, r)
    >>> cmodel.gradient.is_gauged()
    False
    >>> cp = t3m.COREWISE.randn(cmodel.base)
    >>> bool(np.allclose(float(cmodel.evaluate(cp)),
    ...                  float(cmodel.objective_value + cmodel.gradient.inner(cp)
    ...                        + 0.5 * cp.inner(cmodel.gn_hessian(cp)))))
    True

    A trial step at a *different* base is a structural error (the model is tied to its base):

    >>> other = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> model.gn_hessian(t3m.MANIFOLD.randn(t3m.MANIFOLD.base(other)))   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: trial tangent must live at the model's base (same T3Basis object)
    '''

    geometry: typ.Any              # the geometry (MANIFOLD / COREWISE): supplies base & project (Π)
    base:     bvf.T3Basis          # = geometry.base(X) -- the linearization frame
    kind:     fb.SamplingKind      # the sampling kind (APPLY / ENTRIES / PROBE)
    sample:   typ.Any              # ww (apply / probe; len=d, elm_shape=W+(Ni,)) or index (entries; (d,)+W)
    residual: typ.Any              # r = F(X) − y; shape W+C (apply / entries) or len=d, W+C+(Ni,) (probe)

    @ft.cached_property
    def _n_w(self) -> int:  # number of leading sample-stack (W) axes
        return self.kind.w_axes(self.sample)

    @ft.cached_property
    def _base_sweep(self) -> typ.Tuple:  # (xis, mus, nus, etas) on geometry.base(X) -- computed ONCE
        return self.kind.precompute(self.base.data, self.sample)

    @ft.cached_property
    def objective_value(self) -> NDArray:  # c = ½‖r‖², shape C
        '''The least-squares objective at the base, ``c = ½‖r‖²`` (the model constant ``m(0)``).'''
        return 0.5 * self.kind.sumsq(self.residual, self._n_w)

    @ft.cached_property
    def gradient(self) -> t3m.T3Tangent:  # g = Π 𝒥ᵀ r, a tangent at base
        '''The gradient ``g = geometry.project(𝒥ᵀ r)`` (the Gauss-Newton ``Jᵀr``), a tangent at base.

        On the manifold geometry this is the gauged Riemannian gradient; on the corewise geometry it is
        the raw core gradient ``𝒥ᵀr`` (a tangent at ``(U,G,G,G)``, no ``Π``).'''
        dU_dG = self.kind.transpose(self.residual, self.sample, self.base.data, self._base_sweep)
        return self.geometry.project(t3m.T3Tangent(self.base, bvf.T3Variations(*dU_dG)))

    def jacobian(
            self,
            p:  t3m.T3Tangent,
    ) -> NDArray:  # J p = 𝒥(Π p); apply/entries: shape W+C; probe: len=d, elm_shape=W+C+(Ni,)
        '''The linearized forward ``J p = 𝒥(Π p)`` (the Gauss-Newton Jacobian-vector product).

        ONE forward sweep -- no transpose ``𝒥ᵀ``, no gauge re-projection of the output, no tangent
        assembly. The general forward primitive: it gives the predicted residual ``r + J p`` (for
        trust-region / line-search predicted reduction) and, via :py:meth:`gn_quadratic`, the Gauss-Newton
        quadratic form. The result lives in the sample space (a scalar per sample for apply / entries; one
        vector per mode for probe).'''
        _require_at_base(self.base, p)
        Pp = self.geometry.project(p)
        return self.kind.forward(Pp.variations.data, self.sample, self.base.data, self._base_sweep)

    def gn_quadratic(
            self,
            p:  t3m.T3Tangent,
    ) -> NDArray:  # pᵀ H p = ‖J p‖², shape C
        '''The Gauss-Newton quadratic form ``pᵀ H p = ‖J p‖²`` -- ONE forward sweep, NOT a Hessian apply.

        The cheap denominator for Cauchy / line-search step lengths: ``alpha = g.inner(g) /
        model.gn_quadratic(g)``. Because ``H = JᵀJ``, ``pᵀHp = (Jp)ᵀ(Jp) = ‖Jp‖²``, so this needs only the
        forward :py:meth:`jacobian` -- it avoids the transpose ``𝒥ᵀ`` and the ``H p`` tangent
        materialization that the equivalent ``p.inner(self.gn_hessian(p))`` would incur.'''
        return self.kind.sumsq(self.jacobian(p), self._n_w)

    def gn_hessian(self, p: t3m.T3Tangent) -> t3m.T3Tangent:
        '''The Gauss-Newton Hessian action ``H p = Π 𝒥ᵀ 𝒥 Π p`` (``H = JᵀJ``, *not* the full Hessian).

        Projects ``p`` (so the caller need not), applies the bare forward then transpose, and projects the
        result -- a tangent at base. Symmetric. (Corewise: ``Π`` is the identity, so ``H p = 𝒥ᵀ 𝒥 p``,
        which is gauge-singular -- fine for first-order methods, needs regularization for Newton.) For the
        *scalar* quadratic form ``pᵀHp`` alone, prefer the cheaper :py:meth:`gn_quadratic`.'''
        _require_at_base(self.base, p)
        Pp = self.geometry.project(p)
        z = self.kind.forward(Pp.variations.data, self.sample, self.base.data, self._base_sweep)
        dU_dG = self.kind.transpose(z, self.sample, self.base.data, self._base_sweep)
        return self.geometry.project(t3m.T3Tangent(self.base, bvf.T3Variations(*dU_dG)))

    def evaluate(self, p: t3m.T3Tangent) -> NDArray:  # m(p), shape C
        '''The quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p`` with ``H = JᵀJ``, reusing the cached
        ``c`` and ``g`` and one **forward** apply: the quadratic term needs only ``½ pᵀ H p = ½‖𝒥 Π p‖²``,
        not a Hessian apply. ``p`` is projected here, shared by the linear term ``⟨g, Πp⟩`` and the
        quadratic. Equals ``½‖r + 𝒥 Π p‖²`` exactly (the objective is quadratic in the ambient tensor).'''
        _require_at_base(self.base, p)
        Pp = self.geometry.project(p)
        Jp = self.kind.forward(Pp.variations.data, self.sample, self.base.data, self._base_sweep)
        return self.objective_value + self.gradient.inner(Pp) + 0.5 * self.kind.sumsq(Jp, self._n_w)


def apply_model(
        geometry,                            # MANIFOLD / COREWISE (the geometry to linearize through)
        x:          t3.TuckerTensorTrain,    # the current point (the linearization)
        ww:         typ.Sequence[NDArray],   # sample vectors, len=d, elm_shape=W+(Ni,)
        residual:   NDArray,                 # r = apply(x) − y, shape W+C
) -> GaussNewtonModel:
    '''The Gauss-Newton model of an all-modes ``apply`` least-squares objective at ``x``, on ``geometry``.'''
    return GaussNewtonModel(geometry, geometry.base(x), fb.APPLY, ww, residual)


def entries_model(
        geometry,                            # MANIFOLD / COREWISE
        x:          t3.TuckerTensorTrain,    # the current point
        index:      NDArray,                 # int, shape=(d,)+W -- the grid points
        residual:   NDArray,                 # r = entries(x) − y, shape W+C
) -> GaussNewtonModel:
    '''The Gauss-Newton model of an all-modes ``entries`` least-squares objective at ``x``, on ``geometry``.

    Identical to :py:func:`apply_model` but the measurements are tensor **entries** at integer grid points
    ``index`` (shape ``(d,)+W``) rather than applies against probe vectors.'''
    return GaussNewtonModel(geometry, geometry.base(x), fb.ENTRIES, index, residual)


def probe_model(
        geometry,                            # MANIFOLD / COREWISE
        x:          t3.TuckerTensorTrain,    # the current point
        ww:         typ.Sequence[NDArray],   # probe vectors, len=d, elm_shape=W+(Ni,)
        residual:   typ.Sequence[NDArray],   # r = probe(x) − y, len=d, elm_shape=W+C+(Ni,)
) -> GaussNewtonModel:
    '''The Gauss-Newton model of a ``probe`` least-squares objective at ``x``, on ``geometry``.

    Like :py:func:`apply_model` but the measurements are **probes** -- vector-valued (one free mode each),
    so ``residual`` is a sequence of ``d`` arrays (``elm_shape = W+C+(Ni,)``).'''
    return GaussNewtonModel(geometry, geometry.base(x), fb.PROBE, ww, residual)
