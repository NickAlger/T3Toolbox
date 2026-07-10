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
(e.g. ``geometry.randn(model.base)``); a step at a different frame raises the numerical same-frame guard
(skipped under ``safety.unsafe()`` / a jax trace), like :py:class:`~t3toolbox.manifold.T3Tangent`.

The base sweep (the base-and-data edge variables) is computed once by the factory and stored as the
``sweep`` field, reused across every ``gradient`` / ``gn_hessian`` / ``evaluate`` -- in an inner CG the
base is fixed, so the sweep is computed once, not once per matrix-vector product. See
:py:mod:`t3toolbox.backend.fitting` and ``dev/archive/geometry_refactor_plan.md``.

Jitting an optimizer
--------------------
``GaussNewtonModel`` **is** a registered jax pytree (the data -- ``base``, ``sweep``, ``sample``,
``residual`` -- are leaves; ``geometry`` / ``kind`` are static aux). Crucially the base flows as a *leaf*,
not aux (the same is true of :py:class:`~t3toolbox.manifold.T3Tangent`'s frame), so a model or tangent
that **crosses a jit boundary does NOT recompile when the base changes** -- the per-base recompile that
frame-as-aux used to force is gone. So you can jit the frontend matvec directly:

1. **Inner-solve jit (Newton-CG).** Jit the matvec with the model *and* the tangent as arguments; it
   compiles **once for the whole solve** (the model's base/sweep change as data, not as a recompile
   trigger), reused across every outer step and CG iteration::

       Hmatvec = jax.jit(lambda model, p: model.gn_hessian(p))
       # per outer step:  model = fitting.apply_model(geom, X, ww, r)
       # inner CG:         Hmatvec(model, p_k)

2. **Whole-step jit (Cauchy / gradient descent / a fixed-step loop).** Jit a step function whose only
   argument is the point ``X`` and build the model *inside*; compiles once, reused across steps::

       @jax.jit
       def step(X):
           r     = X.apply(ww) - b
           model = fitting.apply_model(t3m.MANIFOLD, X, ww, r)
           g     = model.gradient
           alpha = g.corewise_inner(g) / model.gn_quadratic(g)     # cheap Cauchy step (one forward)
           return t3m.MANIFOLD.retract((-alpha) * g)

Under a trace the numerical same-frame guards skip (you cannot branch on a tracer; jit is unsafe mode), so
a matvec output combines with the eager CG iterates freely. The geometry singletons (``t3m.MANIFOLD`` /
``COREWISE``) are zero-leaf pytrees -- close over or pass as args freely.
'''

from __future__ import annotations

import functools as ft
import typing as typ
from dataclasses import dataclass

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.safety as safety
import t3toolbox.backend.fitting as fb
import t3toolbox.backend.uniform_fitting as ufit
from t3toolbox.backend.common import *

__all__ = ['GaussNewtonModel', 'UniformGaussNewtonModel',
           'apply_model', 'entries_model', 'probe_model',
           'apply_derivatives_model', 'entries_derivatives_model', 'probe_derivatives_model']


def _require_at_base(base: bvf.T3Frame, p: t3m.T3Tangent) -> None:
    '''Same-frame guard: a trial tangent must live at the model's base. The ``is`` fast-path, else a
    NUMERICAL frame compare (safe mode, eager-only; skips under ``safety.unsafe()`` / a jax trace), like
    ``T3Tangent``'s same-tangent-space guard.'''
    if not (p.frame is base or safety.frames_equal_or_skip(base.data, p.frame.data)):
        raise ValueError("trial tangent must live at the model's base (it is at a different frame); "
                         "run inside safety.unsafe() to skip this numerical check")


def _require_at_base_uniform(base: ubv.UT3Frame, p: ut3m.UT3Tangent) -> None:
    '''Uniform twin of :py:func:`_require_at_base`: the frame compare is on the four supercores
    (``base.data[:4]`` -- the full ``.data`` carries the int-tuple ``shape`` that safety's array compare
    cannot take), mirroring ``UT3Tangent._check_same_tangent_space``.'''
    if not (p.frame is base or safety.frames_equal_or_skip(base.data[:4], p.frame.data[:4])):
        raise ValueError("trial tangent must live at the model's base (it is at a different frame); "
                         "run inside safety.unsafe() to skip this numerical check")


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
    >>> print(model.gradient.is_gauged())
    True
    >>> p = t3m.MANIFOLD.randn(model.base)                  # a gauged trial step at the model's base
    >>> print(model.gn_hessian(p).is_gauged())
    True

    ``evaluate(p)`` is the quadratic-model value, consistent with assembling it from the cached ``c`` /
    ``g`` and the Hessian action (and costs only one forward apply, not a Hessian apply):

    >>> hp = model.gn_hessian(p)
    >>> m_built = float(model.objective_value + model.gradient.corewise_inner(p) + 0.5 * p.corewise_inner(hp))
    >>> bool(np.allclose(float(model.evaluate(p)), m_built))
    True

    The Gauss-Newton quadratic form ``pᵀHp = ‖Jp‖²`` comes from one forward sweep (no ``H p`` assembly) --
    the cheap step-length denominator for Cauchy / line search (``alpha = g.corewise_inner(g) / gn_quadratic(g)``):

    >>> bool(np.allclose(float(model.gn_quadratic(p)), float(p.corewise_inner(model.gn_hessian(p)))))
    True

    The **corewise** geometry is the same call with a different geometry: the gradient is a tangent at the
    raw ``(U,G,G,G)`` frame (a core perturbation), with **no** gauge projection:

    >>> cmodel = fitting.apply_model(t3m.COREWISE, x, ww, r)
    >>> print(cmodel.gradient.is_gauged())
    False
    >>> cp = t3m.COREWISE.randn(cmodel.base)
    >>> bool(np.allclose(float(cmodel.evaluate(cp)),
    ...                  float(cmodel.objective_value + cmodel.gradient.corewise_inner(cp)
    ...                        + 0.5 * cp.corewise_inner(cmodel.gn_hessian(cp)))))
    True

    A trial step at a *different* base is a structural error (the model is tied to its base):

    >>> other = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> model.gn_hessian(t3m.MANIFOLD.randn(t3m.MANIFOLD.base(other)))   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: trial tangent must live at the model's base (it is at a different frame)
    '''

    geometry: typ.Any              # the geometry (MANIFOLD / COREWISE): supplies base & project (Π)
    base:     bvf.T3Frame          # = geometry.base(X) -- the linearization frame
    kind:     fb.SamplingKind      # the sampling kind (APPLY / ENTRIES / PROBE)
    sample:   typ.Any              # ww (apply / probe; len=d, elm_shape=W+(Ni,)) or index (entries; (d,)+W)
    residual: typ.Any              # r = F(X) − y; shape W+C (apply / entries) or len=d, W+C+(Ni,) (probe)
    sweep:    typ.Any              # = kind.precompute(base.data, sample); a FIELD (jax leaf) so it is
                                   # carried across a jit boundary and reused, not recomputed per matvec

    @ft.cached_property
    def _n_w(self) -> int:  # number of leading sample-stack (W) axes
        return self.kind.w_axes(self.sample)

    @ft.cached_property
    def objective_value(self) -> NDArray:  # c = ½‖r‖², shape C
        '''The least-squares objective at the base, ``c = ½‖r‖²`` (the model constant ``m(0)``).'''
        return 0.5 * self.kind.sumsq(self.residual, self._n_w)

    @ft.cached_property
    def gradient(self) -> t3m.T3Tangent:  # g = Π 𝒥ᵀ r, a tangent at base
        '''The gradient ``g = geometry.project(𝒥ᵀ r)`` (the Gauss-Newton ``Jᵀr``), a tangent at base.

        On the manifold geometry this is the gauged Riemannian gradient; on the corewise geometry it is
        the raw core gradient ``𝒥ᵀr`` (a tangent at ``(U,G,G,G)``, no ``Π``).'''
        dU_dG = self.kind.transpose(self.residual, self.sample, self.base.data, self.sweep)
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
        return self.kind.forward(Pp.variations.data, self.sample, self.base.data, self.sweep)

    def gn_quadratic(
            self,
            p:  t3m.T3Tangent,
    ) -> NDArray:  # pᵀ H p = ‖J p‖², shape C
        '''The Gauss-Newton quadratic form ``pᵀ H p = ‖J p‖²`` -- ONE forward sweep, NOT a Hessian apply.

        The cheap denominator for Cauchy / line-search step lengths: ``alpha = g.corewise_inner(g) /
        model.gn_quadratic(g)``. Because ``H = JᵀJ``, ``pᵀHp = (Jp)ᵀ(Jp) = ‖Jp‖²``, so this needs only the
        forward :py:meth:`jacobian` -- it avoids the transpose ``𝒥ᵀ`` and the ``H p`` tangent
        materialization that the equivalent ``p.corewise_inner(self.gn_hessian(p))`` would incur.'''
        return self.kind.sumsq(self.jacobian(p), self._n_w)

    def gn_hessian(self, p: t3m.T3Tangent) -> t3m.T3Tangent:
        '''The Gauss-Newton Hessian action ``H p = Π 𝒥ᵀ 𝒥 Π p`` (``H = JᵀJ``, *not* the full Hessian).

        Projects ``p`` (so the caller need not), applies the bare forward then transpose, and projects the
        result -- a tangent at base. Symmetric. (Corewise: ``Π`` is the identity, so ``H p = 𝒥ᵀ 𝒥 p``,
        which is gauge-singular -- fine for first-order methods, needs regularization for Newton.) For the
        *scalar* quadratic form ``pᵀHp`` alone, prefer the cheaper :py:meth:`gn_quadratic`.'''
        _require_at_base(self.base, p)
        Pp = self.geometry.project(p)
        z = self.kind.forward(Pp.variations.data, self.sample, self.base.data, self.sweep)
        dU_dG = self.kind.transpose(z, self.sample, self.base.data, self.sweep)
        return self.geometry.project(t3m.T3Tangent(self.base, bvf.T3Variations(*dU_dG)))

    def evaluate(self, p: t3m.T3Tangent) -> NDArray:  # m(p), shape C
        '''The quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p`` with ``H = JᵀJ``, reusing the cached
        ``c`` and ``g`` and one **forward** apply: the quadratic term needs only ``½ pᵀ H p = ½‖𝒥 Π p‖²``,
        not a Hessian apply. ``p`` is projected here, shared by the linear term ``⟨g, Πp⟩`` and the
        quadratic. Equals ``½‖r + 𝒥 Π p‖²`` exactly (the objective is quadratic in the ambient tensor).'''
        _require_at_base(self.base, p)
        Pp = self.geometry.project(p)
        Jp = self.kind.forward(Pp.variations.data, self.sample, self.base.data, self.sweep)
        return self.objective_value + self.gradient.corewise_inner(Pp) + 0.5 * self.kind.sumsq(Jp, self._n_w)


@dataclass(frozen=True)
class UniformGaussNewtonModel:
    '''The uniform-layer twin of :py:class:`GaussNewtonModel`: the local Gauss-Newton model at ``base``,
    surfacing :py:class:`~t3toolbox.uniform_manifold.UT3Tangent` gradients / Hessian actions so a frontend
    user can **roll their own uniform optimizer** (e.g. manifold L-BFGS) with the same ergonomics as the
    ragged model -- ``UNIFORM_MANIFOLD.inner`` / ``.retract`` / ``.transport`` supply the rest. Built by the
    same factories (:py:func:`apply_model` &c.), which dispatch on ``x``'s type; not directly.

    Internally the sampling ``kind`` runs on the **packed** supercore pair (the speed path); the
    ``UT3Tangent`` boundary conversion (bare variation pair ⟷ gauged tangent) is the only frontend work.

    **Jit (compile-once).** The packed kind is a per-problem closure over ``x``'s rank masks, so -- unlike
    the ragged model's stateless singleton kind -- it cannot go in the jax ``aux_data`` directly (a fresh
    closure each rebuild would recompile). Instead the aux is the **value-hashed** ``(kind_name, x0_masks,
    order, weight)`` and the kind is **rebuilt lazily** from it (the kind builders use only ``(shape,
    masks)``, not the supercores). So a model crossing a jit boundary keeps a stable, value-based aux and
    ``jit(lambda m, p: m.gn_hessian(p))`` compiles **once** across rebuilt models of the same rank (base /
    sweep / sample / residual flow as leaves). See ``docs/uniform_backend_jit_recipe.md``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> import t3toolbox.fitting as fitting
    >>> np.random.seed(0)
    >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1)))
    >>> ww = [np.random.randn(15, N) for N in (6, 7, 8)]
    >>> r = np.random.randn(15)                            # residual r = apply(x) − y

    The gradient is a gauged ``UT3Tangent`` and the Gauss-Newton quadratic form ``pᵀHp = ‖J p‖²`` agrees
    with the Hessian action:

    >>> model = fitting.apply_model(ut3m.UNIFORM_MANIFOLD, x, ww, r)
    >>> print(bool(model.gradient.is_gauged().all()))
    True
    >>> p = ut3m.UNIFORM_MANIFOLD.randn(model.base)
    >>> bool(np.allclose(float(model.gn_quadratic(p)), float(p.corewise_inner(model.gn_hessian(p)))))
    True
    '''
    geometry:  typ.Any               # UNIFORM_MANIFOLD / UNIFORM_COREWISE
    base:      ubv.UT3Frame          # = geometry.base(x); the linearization frame (a jax-leaf pytree)
    kind_name: str                   # 'apply'/'entries'/'probe' (+'_derivatives') -- rebuilds the packed kind
    x0_masks:  ut3.UT3Masks          # x0's plain rank masks (value-hashed aux) -> rebuilds the kind
    order:     typ.Optional[int]                          # derivative kinds only (None for a plain kind)
    weight:    typ.Optional[typ.Tuple[float, ...]]        # derivative kinds only: per-order weight ω (hashable)
    sample:    typ.Any               # PACKED ww (apply/probe) / index (entries) / (ww, pp) (derivatives)
    residual:  typ.Any               # PACKED r = S(x) − y
    sweep:     typ.Any               # = kind.precompute(base.data, sample); a leaf (carried across a jit boundary)

    @ft.cached_property
    def kind(self) -> fb.SamplingKind:  # the packed uniform sampling kind, rebuilt from the value-hashed aux
        x0_data = (None, None, self.base.shape, self.x0_masks.data)   # the kind uses only (shape, masks)
        if self.order is None:
            return ufit.uniform_sampling_kind(self.kind_name, x0_data)
        return ufit.uniform_derivatives_kind(self.kind_name, x0_data, self.order, self.weight)

    @ft.cached_property
    def _n_w(self) -> int:  # number of leading sample-stack (W) axes
        return self.kind.w_axes(self.sample)

    @ft.cached_property
    def objective_value(self) -> NDArray:  # c = ½‖r‖², shape C
        '''The least-squares objective at the base, ``c = ½‖r‖²``.'''
        return 0.5 * self.kind.sumsq(self.residual, self._n_w)

    def _wrap(self, bare) -> ut3m.UT3Tangent:  # gauge-project a bare variation pair -> a UT3Tangent at base
        var = ubv.UT3Variations(bare[0], bare[1], self.base.shape,
                                ubv.UT3Variations._variation_masks_of(self.base))
        return self.geometry.project(ut3m.UT3Tangent(self.base, var))

    @ft.cached_property
    def gradient(self) -> ut3m.UT3Tangent:  # g = Π 𝒥ᵀ r, a UT3Tangent at base
        '''The gradient ``g = geometry.project(𝒥ᵀ r)`` -- gauged on the manifold, raw on corewise.'''
        return self._wrap(self.kind.transpose(self.residual, self.sample, self.base.data, self.sweep))

    def jacobian(self, p: ut3m.UT3Tangent) -> NDArray:  # J p = 𝒥(Π p)
        '''The linearized forward ``J p = 𝒥(Π p)`` -- ONE forward sweep (the predicted residual ``r + J p``).'''
        _require_at_base_uniform(self.base, p)
        Pp = self.geometry.project(p)
        return self.kind.forward(Pp.variations.supercores, self.sample, self.base.data, self.sweep)

    def gn_quadratic(self, p: ut3m.UT3Tangent) -> NDArray:  # pᵀHp = ‖J p‖², shape C
        '''The Gauss-Newton quadratic form ``pᵀHp = ‖J p‖²`` -- ONE forward sweep (the Cauchy / line-search
        step-length denominator), not a Hessian apply.'''
        return self.kind.sumsq(self.jacobian(p), self._n_w)

    def gn_hessian(self, p: ut3m.UT3Tangent) -> ut3m.UT3Tangent:  # H p = Π 𝒥ᵀ 𝒥 Π p
        '''The Gauss-Newton Hessian action ``H p = Π 𝒥ᵀ 𝒥 Π p`` (``H = JᵀJ``), a UT3Tangent at base.'''
        _require_at_base_uniform(self.base, p)
        Pp = self.geometry.project(p)
        z = self.kind.forward(Pp.variations.supercores, self.sample, self.base.data, self.sweep)
        return self._wrap(self.kind.transpose(z, self.sample, self.base.data, self.sweep))

    def evaluate(self, p: ut3m.UT3Tangent) -> NDArray:  # m(p), shape C
        '''The quadratic-model value ``m(p) = c + gᵀp + ½ pᵀHp``, reusing ``c`` / ``g`` and ONE forward apply.'''
        _require_at_base_uniform(self.base, p)
        Pp = self.geometry.project(p)
        Jp = self.kind.forward(Pp.variations.supercores, self.sample, self.base.data, self.sweep)
        return self.objective_value + self.gradient.corewise_inner(Pp) + 0.5 * self.kind.sumsq(Jp, self._n_w)


def _ragged_base(geometry, x: t3.TuckerTensorTrain) -> bvf.T3Frame:
    '''Validate + build the frame for a ragged model: a ragged ``x`` needs a ragged geometry singleton.'''
    if geometry is not t3m.MANIFOLD and geometry is not t3m.COREWISE:
        raise ValueError("a ragged TuckerTensorTrain requires a ragged geometry (manifold.MANIFOLD / "
                         "manifold.COREWISE); for a UniformTuckerTensorTrain use the uniform geometries "
                         "(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE).")
    return geometry.base(x)


def _uniform_model(
        geometry,                            # UNIFORM_MANIFOLD / UNIFORM_COREWISE (required for a uniform x)
        x:          ut3.UniformTuckerTensorTrain,
        kind_name:  str,                     # 'apply'/'entries'/'probe' (+'_derivatives')
        sample:     typ.Any,                 # ragged or packed (packed once here, mirror-tolerant)
        residual:   typ.Any,                 # ragged or packed r = S(x) − y
        order:      typ.Optional[int]                 = None,
        weight:     typ.Optional[typ.Sequence[float]] = None,
) -> UniformGaussNewtonModel:
    '''Assemble a :py:class:`UniformGaussNewtonModel`: build the frame, pack the loop-invariant sample +
    residual once (:py:func:`~t3toolbox.backend.uniform_fitting.pack_sample` / ``pack_data``), precompute
    the base sweep, and store the value-hashed aux to rebuild the packed kind under jit.'''
    if geometry is not ut3m.UNIFORM_MANIFOLD and geometry is not ut3m.UNIFORM_COREWISE:
        raise ValueError("a UniformTuckerTensorTrain requires a uniform geometry "
                         "(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE).")
    N = x.N
    base = geometry.base(x)                              # UT3Frame
    weight_t = tuple(weight) if weight is not None else None   # hashable aux (jit)
    packed_sample = ufit.pack_sample(kind_name, sample, N)
    packed_residual = ufit.pack_data(kind_name, residual, N)
    x0_data = (None, None, x.shape, x.masks.data)        # the kind builders use only (shape, masks)
    kind = (ufit.uniform_sampling_kind(kind_name, x0_data) if order is None
            else ufit.uniform_derivatives_kind(kind_name, x0_data, order, weight_t))
    sweep = kind.precompute(base.data, packed_sample)
    return UniformGaussNewtonModel(geometry, base, kind_name, x.masks, order, weight_t,
                                   packed_sample, packed_residual, sweep)


def apply_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:         typ.Sequence[NDArray],   # sample vectors, len=d, elm_shape=W+(Ni,)
        residual:   NDArray,                 # r = apply(x) − y, shape W+C
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The Gauss-Newton model of an all-modes ``apply`` least-squares objective at ``x``, on ``geometry``.

    Accepts a ragged ``TuckerTensorTrain`` (-> :py:class:`GaussNewtonModel`) or a uniform
    ``UniformTuckerTensorTrain`` (-> :py:class:`UniformGaussNewtonModel`); the representation is inferred
    from ``x`` and the geometry must match.'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'apply', ww, residual)
    base = _ragged_base(geometry, x)
    return GaussNewtonModel(geometry, base, fb.APPLY, ww, residual, fb.APPLY.precompute(base.data, ww))


def entries_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        index:      NDArray,                 # int, shape=(d,)+W -- the grid points
        residual:   NDArray,                 # r = entries(x) − y, shape W+C
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The Gauss-Newton model of an all-modes ``entries`` least-squares objective at ``x``, on ``geometry``.

    Identical to :py:func:`apply_model` but the measurements are tensor **entries** at integer grid points
    ``index`` (shape ``(d,)+W``) rather than applies against probe vectors.'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'entries', index, residual)
    base = _ragged_base(geometry, x)
    return GaussNewtonModel(geometry, base, fb.ENTRIES, index, residual, fb.ENTRIES.precompute(base.data, index))


def probe_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:         typ.Sequence[NDArray],   # probe vectors, len=d, elm_shape=W+(Ni,)
        residual:   typ.Sequence[NDArray],   # r = probe(x) − y, len=d, elm_shape=W+C+(Ni,)
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The Gauss-Newton model of a ``probe`` least-squares objective at ``x``, on ``geometry``.

    Like :py:func:`apply_model` but the measurements are **probes** -- vector-valued (one free mode each),
    so ``residual`` is a sequence of ``d`` arrays (``elm_shape = W+C+(Ni,)``).'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'probe', ww, residual)
    base = _ragged_base(geometry, x)
    return GaussNewtonModel(geometry, base, fb.PROBE, ww, residual, fb.PROBE.precompute(base.data, ww))


# --------------------------------------------------------------------------------------------------
# Derivative sampling models (the symmetric directional-derivative jets of apply/entries/probe). Same
# GaussNewtonModel, a parameterized derivative kind (order + the per-order residual weight ω): the
# measurement is a jet (a leading order axis), the sample is the paired (ww/index, pp), and `residual`
# is RAW (r = S(x) − y); ω weights the objective ½‖ω⊙r‖² inside the kind. See dev/archive/derivative_fitting_plan.md.
# --------------------------------------------------------------------------------------------------
def apply_derivatives_model(
        geometry,                            # MANIFOLD / COREWISE
        x:          t3.TuckerTensorTrain,    # the current point
        ww:         typ.Sequence[NDArray],   # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:      int,                     # highest derivative order
        residual:   NDArray,                 # RAW r = apply_derivatives(x) − y, shape (order+1)+W+C
        weight:     typ.Optional[typ.Sequence[float]] = None,  # per-order residual weight ω, (order+1,); None=1
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The Gauss-Newton model of an ``apply``-**derivatives** least-squares objective at ``x``: the
    symmetric directional derivatives (orders ``0..order``) of the all-modes apply, in direction ``P``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.fitting as fitting
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(15, N) for N in (6, 7, 8)]    # 15 samples
    >>> pp = [np.random.randn(15, N) for N in (6, 7, 8)]    # one direction P per sample
    >>> r = np.random.randn(4, 15)                          # RAW residual jet, (order+1, W) for order 3

    A per-order weight ``ω`` balances the wildly-different-magnitude orders -- it weights the objective
    ``½‖ω⊙r‖²`` inside the kind; the gradient stays a gauged Riemannian tangent, and the Gauss-Newton
    quadratic form ``pᵀHp = ‖J p‖²`` agrees with the Hessian action:

    >>> model = fitting.apply_derivatives_model(t3m.MANIFOLD, x, ww, pp, 3, r, weight=[1.0, 0.5, 0.3, 0.2])
    >>> print(model.gradient.is_gauged())
    True
    >>> p = t3m.MANIFOLD.randn(model.base)
    >>> bool(np.allclose(float(model.gn_quadratic(p)), float(p.corewise_inner(model.gn_hessian(p)))))
    True
    '''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'apply_derivatives', (ww, pp), residual, order, weight)
    kind = fb.apply_derivatives_kind(order, weight)
    base = _ragged_base(geometry, x)
    return GaussNewtonModel(geometry, base, kind, (ww, pp), residual, kind.precompute(base.data, (ww, pp)))


def entries_derivatives_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        index:      NDArray,                 # int, shape=(d,)+W -- the grid points
        pp:         typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:      int,
        residual:   NDArray,                 # RAW r = entries_derivatives(x) − y, shape (order+1)+W+C
        weight:     typ.Optional[typ.Sequence[float]] = None,
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The ``entries``-derivatives Gauss-Newton model -- like :py:func:`apply_derivatives_model` at integer
    grid points ``index``.'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'entries_derivatives', (index, pp), residual, order, weight)
    kind = fb.entries_derivatives_kind(order, weight)
    base = _ragged_base(geometry, x)
    return GaussNewtonModel(geometry, base, kind, (index, pp), residual, kind.precompute(base.data, (index, pp)))


def probe_derivatives_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:         typ.Sequence[NDArray],   # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:      int,
        residual:   typ.Sequence[NDArray],   # RAW r = probe_derivatives(x) − y, len=d, elm_shape=(order+1)+W+C+(Ni,)
        weight:     typ.Optional[typ.Sequence[float]] = None,
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The ``probe``-derivatives Gauss-Newton model -- vector-valued (one free mode per probe), so
    ``residual`` is a sequence of ``d`` jets.'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'probe_derivatives', (ww, pp), residual, order, weight)
    kind = fb.probe_derivatives_kind(order, weight)
    base = _ragged_base(geometry, x)
    return GaussNewtonModel(geometry, base, kind, (ww, pp), residual, kind.precompute(base.data, (ww, pp)))


if has_jax:
    import jax

    # Register GaussNewtonModel as a jax pytree: the data (base, sweep, sample, residual) are LEAVES, the
    # statics (geometry, kind) are aux_data. Because T3Tangent's frame is now a leaf too (see manifold.py),
    # nothing carries a base as aux_data, so a model crossing a jit boundary does NOT recompile when the
    # base changes -- `jit(lambda model, p: model.gn_hessian(p))(model, p)` compiles once and reuses across
    # outer steps. The sweep is a stored field (a leaf) so it is carried/reused, not recomputed inside the
    # trace. The same-base guard is the numerical same-frame check (skips under the trace).
    jax.tree_util.register_pytree_node(
        GaussNewtonModel,
        lambda m: ((m.base, m.sweep, m.sample, m.residual), (m.geometry, m.kind)),
        lambda aux, children: GaussNewtonModel(aux[0], children[0], aux[1], children[2], children[3], children[1]),
    )

    # UniformGaussNewtonModel: the data (base -- itself a leaf pytree -- sweep, sample, residual) are LEAVES;
    # the aux is VALUE-HASHED (geometry singleton, kind_name, x0's rank masks, order, weight) so a rebuilt
    # model of the same rank is the SAME jit cache key -> `jit(lambda m, p: m.gn_hessian(p))` compiles once
    # across outer steps (the packed kind is rebuilt lazily from this aux; it can't be aux itself -- a fresh
    # closure would recompile). See the class docstring + docs/uniform_backend_jit_recipe.md.
    jax.tree_util.register_pytree_node(
        UniformGaussNewtonModel,
        lambda m: ((m.base, m.sweep, m.sample, m.residual),
                   (m.geometry, m.kind_name, m.x0_masks, m.order, m.weight)),
        lambda aux, ch: UniformGaussNewtonModel(aux[0], ch[0], aux[1], aux[2], aux[3], aux[4],
                                                ch[2], ch[3], ch[1]),
    )
