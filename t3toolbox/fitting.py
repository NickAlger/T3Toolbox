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
    frame ``frame = geometry.frame(X)`` and the projection ``Π = geometry.project``; the forward is
    ``J = 𝒥∘Π`` and the gradient is ``Jᵀr = Π∘𝒥ᵀr``. **Manifold ⟺ Π, corewise ⟺ no Π is structural**
    (bundled in the geometry), never a flag -- mixing the two silently corrupts the result.
  * the **sampling kind** -- bound by the factory :py:func:`apply_model` / :py:func:`entries_model` /
    :py:func:`probe_model` (the bare ``𝒥`` / ``𝒥ᵀ`` from :py:mod:`t3toolbox.backend.probing`, bundled in
    :py:mod:`t3toolbox.backend.fitting`).

Every input and output is a :py:class:`~t3toolbox.manifold.T3Tangent` at ``model.frame`` -- including the
**corewise** case, where the tangent lives at the non-orthonormal frame ``(U,G,G,G)`` (a core
perturbation; see :py:class:`~t3toolbox.manifold.CorewiseGeometry`). Build trial steps at ``model.frame``
(e.g. ``geometry.randn(model.frame)``); a step at a different frame raises the numerical same-frame guard
(skipped under ``safety.unsafe()`` / a jax trace), like :py:class:`~t3toolbox.manifold.T3Tangent`.

The frame sweep (the frame-and-data edge variables) is computed once by the factory and stored as the
``sweep`` field, reused across every ``gradient`` / ``gn_hessian`` / ``evaluate`` -- in an inner CG the
frame is fixed, so the sweep is computed once, not once per matrix-vector product. See
:py:mod:`t3toolbox.backend.fitting` and ``dev/archive/geometry_refactor_plan.md``.

Jitting an optimizer
--------------------
``GaussNewtonModel`` **is** a registered jax pytree (the data -- ``frame``, ``sweep``, ``sample``,
``residual`` -- are leaves; ``geometry`` / ``kind`` are static aux). Crucially the frame flows as a *leaf*,
not aux (the same is true of :py:class:`~t3toolbox.manifold.T3Tangent`'s frame), so a model or tangent
that **crosses a jit boundary does NOT recompile when the frame changes** -- the per-frame recompile that
frame-as-aux used to force is gone. So you can jit the frontend matvec directly:

1. **Inner-solve jit (Newton-CG).** Jit the matvec with the model *and* the tangent as arguments; it
   compiles **once for the whole solve** (the model's frame/sweep change as data, not as a recompile
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

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.safety as safety
import t3toolbox.backend.fitting as fb
import t3toolbox.backend.optimizers as bopt   # the backend GeometryOps a Regularizer leans on (no cycle: backend never imports fitting)
import t3toolbox.backend.uniform_fitting as ufit
from t3toolbox.backend.common import *

__all__ = ['GaussNewtonModel', 'UniformGaussNewtonModel',
           'apply_model', 'entries_model', 'probe_model',
           'apply_derivatives_model', 'entries_derivatives_model', 'probe_derivatives_model']


def _require_at_frame(frame: bvf.T3Frame, p: t3m.T3Tangent) -> None:
    '''Same-frame guard: a trial tangent must live at the model's frame. The ``is`` fast-path, else a
    NUMERICAL frame compare (safe mode, eager-only; skips under ``safety.unsafe()`` / a jax trace), like
    ``T3Tangent``'s same-tangent-space guard.'''
    if not (p.frame is frame or safety.frames_equal_or_skip(frame.data, p.frame.data)):
        raise ValueError("trial tangent must live at the model's frame (it is at a different frame); "
                         "run inside safety.unsafe() to skip this numerical check")


def _require_at_frame_uniform(frame: ubv.UT3Frame, p: ut3m.UT3Tangent) -> None:
    '''Uniform twin of :py:func:`_require_at_frame`: the frame compare is on the four supercores
    (``frame.data[:4]`` -- the full ``.data`` carries the int-tuple ``shape`` that safety's array compare
    cannot take), mirroring ``UT3Tangent._check_same_tangent_space``.'''
    if not (p.frame is frame or safety.frames_equal_or_skip(frame.data[:4], p.frame.data[:4])):
        raise ValueError("trial tangent must live at the model's frame (it is at a different frame); "
                         "run inside safety.unsafe() to skip this numerical check")


def _canonical_weight(
        weight:    typ.Optional[typ.Any],  # None, or array / (nested) sequence -- the raw ω input
        kind_name: str,                    # 'probe' / 'probe_derivatives' / 'apply_derivatives' / ...
        d:         int,                    # number of modes
        order:     int,                    # highest derivative order (0 for a plain kind)
) -> typ.Optional[NDArray]:                # None, or the canonical 2-D ω[m,o], m in {1,d}, o in {1,order+1}
    '''Validate + canonicalize a residual weight to the 2-D matrix ``ω[mode, order]`` (structural errors,
    both modes -- shape is not a numerical property). Enforces the frontend contracts the backend leaves
    lenient: **plain probe** takes a 1-D per-mode weight ``(d,)`` (a 2-D ``(d, 1)`` is rejected -- it has no
    order axis, and accepting it would break forward compat if a less-important axis is ever added);
    **apply/entries** have no mode axis, so their weight is **order-only** (mode dim must be 1 -- mode
    weighting is probe-only). A bare 1-D vector binds to each model's innermost/most-important axis: order
    for the derivative kinds (the backward-compatible rule), mode for plain probe.'''
    if weight is None:
        return None
    w = np.asarray(weight, dtype=float)
    plain_probe = (kind_name == 'probe')            # plain probe: 1-D per-mode, no order axis
    if plain_probe:
        if w.ndim != 1:
            raise ValueError("plain probe takes a 1-D per-mode residual weight of shape (d,); it has no "
                             "order axis (got shape %s). For per-(mode, order) weighting use "
                             "probe_derivatives." % (w.shape,))
        wm = w[:, None]                             # (d, 1) -- per-mode
    elif w.ndim == 1:
        wm = w[None, :]                             # bare vector -> per-order row (1, order+1)
    elif w.ndim == 2:
        wm = w
    else:
        raise ValueError("residual weight must be 1-D or 2-D (ω[mode, order]); got shape %s" % (w.shape,))
    m, o = wm.shape
    if o not in (1, order + 1):
        raise ValueError("residual weight's order dimension must be 1 or order+1=%d; got %d (shape %s)"
                         % (order + 1, o, wm.shape))
    if m not in (1, d):
        raise ValueError("residual weight's mode dimension must be 1 or d=%d; got %d (shape %s)"
                         % (d, m, wm.shape))
    if 'probe' not in kind_name and m > 1:          # apply/entries: no mode axis (mode weighting is probe-only)
        raise ValueError("apply / entries contract every mode into a scalar -- they have no mode axis, so a "
                         "per-mode weight (mode dim %d > 1) is undefined. Use an order-only weight "
                         "(order+1,); per-mode weighting is defined only for probe." % m)
    return wm


def _hashable_weight(
        wm:  typ.Optional[NDArray],   # canonical 2-D ω[m,o] from _canonical_weight, or None
) -> typ.Optional[typ.Tuple[typ.Tuple[float, ...], ...]]:  # a nested tuple (a stable, hashable jit-aux key)
    '''The value-hashed aux form of a weight matrix (a tuple of row tuples), so a rebuilt
    :py:class:`UniformGaussNewtonModel` of the same weight is the SAME jit cache key.'''
    return None if wm is None else tuple(tuple(float(v) for v in row) for row in wm)


@dataclass(frozen=True)
class GaussNewtonModel:
    '''The local Gauss-Newton model of a least-squares objective at ``frame``, generic over the geometry.

    Built by a sampling-kind factory (:py:func:`apply_model` / :py:func:`entries_model` /
    :py:func:`probe_model`), not directly. Holds the geometry, the frame ``frame = geometry.frame(X)``, the
    sampling ``kind``, the measurement ``sample`` (the probe/apply vectors ``ww`` or the integer grid
    ``index``), and the data residual ``r = F(X) − y``. The frame sweep, objective value, and gradient are
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
    >>> p = t3m.MANIFOLD.randn(model.frame)                  # a gauged trial step at the model's frame
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
    >>> cp = t3m.COREWISE.randn(cmodel.frame)
    >>> bool(np.allclose(float(cmodel.evaluate(cp)),
    ...                  float(cmodel.objective_value + cmodel.gradient.corewise_inner(cp)
    ...                        + 0.5 * cp.corewise_inner(cmodel.gn_hessian(cp)))))
    True

    A trial step at a *different* frame is a structural error (the model is tied to its frame):

    >>> other = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> model.gn_hessian(t3m.MANIFOLD.randn(t3m.MANIFOLD.frame(other)))   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: trial tangent must live at the model's frame (it is at a different frame)
    '''

    geometry: typ.Any              # the geometry (MANIFOLD / COREWISE): supplies frame & project (Π)
    frame:     bvf.T3Frame          # = geometry.frame(X) -- the linearization frame
    kind:     fb.SamplingKind      # the sampling kind (APPLY / ENTRIES / PROBE)
    sample:   typ.Any              # ww (apply / probe; len=d, elm_shape=W+(Ni,)) or index (entries; (d,)+W)
    residual: typ.Any              # r = F(X) − y; shape W+C (apply / entries) or len=d, W+C+(Ni,) (probe)
    sweep:    typ.Any              # = kind.precompute(frame.data, sample); a FIELD (jax leaf) so it is
                                   # carried across a jit boundary and reused, not recomputed per matvec

    regularizer: typ.Any = None    # optional backend.regularization.Regularizer; ρ folded into obj/grad/hessian/quadratic/evaluate

    @property
    def _bgeom(self):              # the backend GeometryOps for this geometry (the regularizer's primitives)
        return _backend_geometry_ops(self.geometry)

    def _reg_tangent(self, raw) -> t3m.T3Tangent:   # wrap a backend raw (tucker_var, tt_var) as a T3Tangent at frame
        return t3m.T3Tangent(self.frame, bvf.T3Variations(*raw))

    @ft.cached_property
    def _n_w(self) -> int:  # number of leading sample-stack (W) axes
        return self.kind.w_axes(self.sample)

    @ft.cached_property
    def objective_value(self) -> NDArray:  # c = ½‖r‖² (+ ρ(X)), shape C
        '''The least-squares objective at the frame, ``c = ½‖r‖²`` (the model constant ``m(0)``); plus
        ``ρ(X)`` when a ``regularizer`` is set (``X`` = the frame's tensor ``(U, P)``).'''
        c = 0.5 * self.kind.sumsq(self.residual, self._n_w)
        if self.regularizer is not None:
            c = c + self.regularizer.value(self._bgeom, (self.frame.data[0], self.frame.data[2]))
        return c

    @ft.cached_property
    def gradient(self) -> t3m.T3Tangent:  # g = Π 𝒥ᵀ r (+ g_R), a tangent at frame
        '''The gradient ``g = geometry.project(𝒥ᵀ r)`` (the Gauss-Newton ``Jᵀr``), a tangent at frame,
        plus the regularizer gradient ``g_R`` when set.

        On the manifold geometry this is the gauged Riemannian gradient; on the corewise geometry it is
        the raw core gradient ``𝒥ᵀr`` (a tangent at ``(U,G,G,G)``, no ``Π``).'''
        dU_dG = self.kind.transpose(self.residual, self.sample, self.frame.data, self.sweep)
        g = self.geometry.project(t3m.T3Tangent(self.frame, bvf.T3Variations(*dU_dG)))
        if self.regularizer is not None:
            g = g + self._reg_tangent(self.regularizer.gradient(self._bgeom, self.frame.data))
        return g

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
        _require_at_frame(self.frame, p)
        Pp = self.geometry.project(p)
        return self.kind.forward(Pp.variations.data, self.sample, self.frame.data, self.sweep)

    def gn_quadratic(
            self,
            p:  t3m.T3Tangent,
    ) -> NDArray:  # pᵀ H p = ‖J p‖², shape C
        '''The Gauss-Newton quadratic form ``pᵀ H p = ‖J p‖²`` -- ONE forward sweep, NOT a Hessian apply.

        The cheap denominator for Cauchy / line-search step lengths: ``alpha = g.corewise_inner(g) /
        model.gn_quadratic(g)``. Because ``H = JᵀJ``, ``pᵀHp = (Jp)ᵀ(Jp) = ‖Jp‖²``, so this needs only the
        forward :py:meth:`jacobian` -- it avoids the transpose ``𝒥ᵀ`` and the ``H p`` tangent
        materialization that the equivalent ``p.corewise_inner(self.gn_hessian(p))`` would incur. When a
        ``regularizer`` is set this adds its ``⟨p, H_R p⟩`` term, consistent with :py:meth:`gn_hessian`.'''
        q = self.kind.sumsq(self.jacobian(p), self._n_w)
        if self.regularizer is not None:
            q = q + self.regularizer.quadratic(self._bgeom, self.frame.data, p.variations.data)
        return q

    def gn_hessian(self, p: t3m.T3Tangent) -> t3m.T3Tangent:
        '''The Gauss-Newton Hessian action ``H p = Π 𝒥ᵀ 𝒥 Π p`` (``H = JᵀJ``, *not* the full Hessian),
        plus the regularizer term ``H_R p`` when set.

        Projects ``p`` (so the caller need not), applies the bare forward then transpose, and projects the
        result -- a tangent at frame. Symmetric. (Corewise: ``Π`` is the identity, so ``H p = 𝒥ᵀ 𝒥 p``,
        which is gauge-singular -- fine for first-order methods, needs regularization for Newton.) For the
        *scalar* quadratic form ``pᵀHp`` alone, prefer the cheaper :py:meth:`gn_quadratic`.'''
        _require_at_frame(self.frame, p)
        Pp = self.geometry.project(p)
        z = self.kind.forward(Pp.variations.data, self.sample, self.frame.data, self.sweep)
        dU_dG = self.kind.transpose(z, self.sample, self.frame.data, self.sweep)
        Hp = self.geometry.project(t3m.T3Tangent(self.frame, bvf.T3Variations(*dU_dG)))
        if self.regularizer is not None:
            Hp = Hp + self._reg_tangent(self.regularizer.hessian(self._bgeom, self.frame.data, p.variations.data))
        return Hp

    def evaluate(self, p: t3m.T3Tangent) -> NDArray:  # m(p), shape C
        '''The quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p`` with ``H = JᵀJ``, reusing the cached
        ``c`` and ``g`` and one **forward** apply: the quadratic term needs only ``½ pᵀ H p = ½‖𝒥 Π p‖²``,
        not a Hessian apply. ``p`` is projected here, shared by the linear term ``⟨g, Πp⟩`` and the
        quadratic. Equals ``½‖r + 𝒥 Π p‖²`` exactly (the objective is quadratic in the ambient tensor).
        With a ``regularizer``, ``c`` and ``g`` already carry ``ρ`` / ``g_R``, so only the quadratic
        ``½⟨p, H_R p⟩`` is added here.'''
        _require_at_frame(self.frame, p)
        Pp = self.geometry.project(p)
        Jp = self.kind.forward(Pp.variations.data, self.sample, self.frame.data, self.sweep)
        m = self.objective_value + self.gradient.corewise_inner(Pp) + 0.5 * self.kind.sumsq(Jp, self._n_w)
        if self.regularizer is not None:
            m = m + 0.5 * self.regularizer.quadratic(self._bgeom, self.frame.data, p.variations.data)
        return m


@dataclass(frozen=True)
class UniformGaussNewtonModel:
    '''The uniform-layer twin of :py:class:`GaussNewtonModel`: the local Gauss-Newton model at ``frame``,
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
    ``jit(lambda m, p: m.gn_hessian(p))`` compiles **once** across rebuilt models of the same rank (frame /
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
    >>> p = ut3m.UNIFORM_MANIFOLD.randn(model.frame)
    >>> bool(np.allclose(float(model.gn_quadratic(p)), float(p.corewise_inner(model.gn_hessian(p)))))
    True
    '''
    geometry:  typ.Any               # UNIFORM_MANIFOLD / UNIFORM_COREWISE
    frame:      ubv.UT3Frame          # = geometry.frame(x); the linearization frame (a jax-leaf pytree)
    kind_name: str                   # 'apply'/'entries'/'probe' (+'_derivatives') -- rebuilds the packed kind
    x0_masks:  ut3.UT3Masks          # x0's plain rank masks (value-hashed aux) -> rebuilds the kind
    order:     typ.Optional[int]                          # derivative kinds only (None for a plain kind)
    weight:    typ.Optional[typ.Tuple[typ.Tuple[float, ...], ...]]  # residual weight ω[mode,order], nested tuple (hashable)
    sample:    typ.Any               # PACKED ww (apply/probe) / index (entries) / (ww, pp) (derivatives)
    residual:  typ.Any               # PACKED r = S(x) − y
    sweep:     typ.Any               # = kind.precompute(frame.data, sample); a leaf (carried across a jit boundary)

    @ft.cached_property
    def kind(self) -> fb.SamplingKind:  # the packed uniform sampling kind, rebuilt from the value-hashed aux
        x0_data = (None, None, self.frame.shape, self.x0_masks.data)   # the kind uses only (shape, masks)
        if self.order is None:                                        # plain kinds: only probe is weightable
            return ufit.uniform_sampling_kind(self.kind_name, x0_data, self.weight)
        return ufit.uniform_derivatives_kind(self.kind_name, x0_data, self.order, self.weight)

    @ft.cached_property
    def _n_w(self) -> int:  # number of leading sample-stack (W) axes
        return self.kind.w_axes(self.sample)

    @ft.cached_property
    def objective_value(self) -> NDArray:  # c = ½‖r‖², shape C
        '''The least-squares objective at the frame, ``c = ½‖r‖²``.'''
        return 0.5 * self.kind.sumsq(self.residual, self._n_w)

    def _wrap(self, bare) -> ut3m.UT3Tangent:  # gauge-project a bare variation pair -> a UT3Tangent at frame
        var = ubv.UT3Variations(bare[0], bare[1], self.frame.shape,
                                ubv.UT3Variations._variation_masks_of(self.frame))
        return self.geometry.project(ut3m.UT3Tangent(self.frame, var))

    @ft.cached_property
    def gradient(self) -> ut3m.UT3Tangent:  # g = Π 𝒥ᵀ r, a UT3Tangent at frame
        '''The gradient ``g = geometry.project(𝒥ᵀ r)`` -- gauged on the manifold, raw on corewise.'''
        return self._wrap(self.kind.transpose(self.residual, self.sample, self.frame.data, self.sweep))

    def jacobian(self, p: ut3m.UT3Tangent) -> NDArray:  # J p = 𝒥(Π p)
        '''The linearized forward ``J p = 𝒥(Π p)`` -- ONE forward sweep (the predicted residual ``r + J p``).'''
        _require_at_frame_uniform(self.frame, p)
        Pp = self.geometry.project(p)
        return self.kind.forward(Pp.variations.supercores, self.sample, self.frame.data, self.sweep)

    def gn_quadratic(self, p: ut3m.UT3Tangent) -> NDArray:  # pᵀHp = ‖J p‖², shape C
        '''The Gauss-Newton quadratic form ``pᵀHp = ‖J p‖²`` -- ONE forward sweep (the Cauchy / line-search
        step-length denominator), not a Hessian apply.'''
        return self.kind.sumsq(self.jacobian(p), self._n_w)

    def gn_hessian(self, p: ut3m.UT3Tangent) -> ut3m.UT3Tangent:  # H p = Π 𝒥ᵀ 𝒥 Π p
        '''The Gauss-Newton Hessian action ``H p = Π 𝒥ᵀ 𝒥 Π p`` (``H = JᵀJ``), a UT3Tangent at frame.'''
        _require_at_frame_uniform(self.frame, p)
        Pp = self.geometry.project(p)
        z = self.kind.forward(Pp.variations.supercores, self.sample, self.frame.data, self.sweep)
        return self._wrap(self.kind.transpose(z, self.sample, self.frame.data, self.sweep))

    def evaluate(self, p: ut3m.UT3Tangent) -> NDArray:  # m(p), shape C
        '''The quadratic-model value ``m(p) = c + gᵀp + ½ pᵀHp``, reusing ``c`` / ``g`` and ONE forward apply.'''
        _require_at_frame_uniform(self.frame, p)
        Pp = self.geometry.project(p)
        Jp = self.kind.forward(Pp.variations.supercores, self.sample, self.frame.data, self.sweep)
        return self.objective_value + self.gradient.corewise_inner(Pp) + 0.5 * self.kind.sumsq(Jp, self._n_w)


def _ragged_frame(geometry, x: t3.TuckerTensorTrain) -> bvf.T3Frame:
    '''Validate + build the frame for a ragged model: a ragged ``x`` needs a ragged geometry singleton.'''
    if geometry is not t3m.MANIFOLD and geometry is not t3m.COREWISE:
        raise ValueError("a ragged TuckerTensorTrain requires a ragged geometry (manifold.MANIFOLD / "
                         "manifold.COREWISE); for a UniformTuckerTensorTrain use the uniform geometries "
                         "(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE).")
    return geometry.frame(x)


def _backend_geometry_ops(geometry):
    '''Map a ragged frontend geometry singleton to its backend ``GeometryOps`` -- the regularizer lives in
    the backend and leans on ``point_norm_sq`` / ``point_tangent`` / ``project`` / ``inner``, so the
    frontend model delegates to it on raw ``.data`` (dev/regularization_design.md §5a).'''
    if geometry is t3m.MANIFOLD:
        return bopt.MANIFOLD_OPS
    if geometry is t3m.COREWISE:
        return bopt.COREWISE_OPS
    raise ValueError("regularization requires a ragged geometry (manifold.MANIFOLD / COREWISE).")


def _reject_uniform_regularizer(regularizer) -> None:
    '''Uniform-layer regularization is a later slice (S3); reject it clearly on the uniform model path.'''
    if regularizer is not None:
        raise NotImplementedError("regularization is not yet supported on the uniform layer; use a ragged "
                                  "TuckerTensorTrain x (uniform support is planned).")


def _uniform_model(
        geometry,                            # UNIFORM_MANIFOLD / UNIFORM_COREWISE (required for a uniform x)
        x:          ut3.UniformTuckerTensorTrain,
        kind_name:  str,                     # 'apply'/'entries'/'probe' (+'_derivatives')
        sample:     typ.Any,                 # ragged or packed (packed once here, mirror-tolerant)
        residual:   typ.Any,                 # ragged or packed r = S(x) − y
        order:      typ.Optional[int] = None,
        weight:     typ.Optional[NDArray] = None,   # canonical 2-D ω[m,o] (from _canonical_weight), or None
) -> UniformGaussNewtonModel:
    '''Assemble a :py:class:`UniformGaussNewtonModel`: build the frame, pack the loop-invariant sample +
    residual once (:py:func:`~t3toolbox.backend.uniform_fitting.pack_sample` / ``pack_data``), precompute
    the frame sweep, and store the value-hashed aux (the ``ω`` matrix as a nested tuple) to rebuild the
    packed kind under jit.'''
    if geometry is not ut3m.UNIFORM_MANIFOLD and geometry is not ut3m.UNIFORM_COREWISE:
        raise ValueError("a UniformTuckerTensorTrain requires a uniform geometry "
                         "(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE).")
    N = x.N
    frame = geometry.frame(x)                              # UT3Frame
    weight_t = _hashable_weight(weight)                   # nested-tuple ω[m,o] (hashable jit aux)
    packed_sample = ufit.pack_sample(kind_name, sample, N)
    packed_residual = ufit.pack_data(kind_name, residual, N)
    x0_data = (None, None, x.shape, x.masks.data)        # the kind builders use only (shape, masks)
    kind = (ufit.uniform_sampling_kind(kind_name, x0_data, weight_t) if order is None
            else ufit.uniform_derivatives_kind(kind_name, x0_data, order, weight_t))
    sweep = kind.precompute(frame.data, packed_sample)
    return UniformGaussNewtonModel(geometry, frame, kind_name, x.masks, order, weight_t,
                                   packed_sample, packed_residual, sweep)


def apply_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:         typ.Sequence[NDArray],   # sample vectors, len=d, elm_shape=W+(Ni,)
        residual:   NDArray,                 # r = apply(x) − y, shape W+C
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ) (ragged only)
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The Gauss-Newton model of an all-modes ``apply`` least-squares objective at ``x``, on ``geometry``.

    Accepts a ragged ``TuckerTensorTrain`` (-> :py:class:`GaussNewtonModel`) or a uniform
    ``UniformTuckerTensorTrain`` (-> :py:class:`UniformGaussNewtonModel`); the representation is inferred
    from ``x`` and the geometry must match. ``regularizer`` adds ``ρ(x)`` to the model (ragged only).'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        _reject_uniform_regularizer(regularizer)
        return _uniform_model(geometry, x, 'apply', ww, residual)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, fb.APPLY, ww, residual, fb.APPLY.precompute(frame.data, ww),
                            regularizer)


def entries_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        index:      NDArray,                 # int, shape=(d,)+W -- the grid points
        residual:   NDArray,                 # r = entries(x) − y, shape W+C
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ) (ragged only)
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The Gauss-Newton model of an all-modes ``entries`` least-squares objective at ``x``, on ``geometry``.

    Identical to :py:func:`apply_model` but the measurements are tensor **entries** at integer grid points
    ``index`` (shape ``(d,)+W``) rather than applies against probe vectors.'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        _reject_uniform_regularizer(regularizer)
        return _uniform_model(geometry, x, 'entries', index, residual)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, fb.ENTRIES, index, residual,
                            fb.ENTRIES.precompute(frame.data, index), regularizer)


def probe_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:         typ.Sequence[NDArray],   # probe vectors, len=d, elm_shape=W+(Ni,)
        residual:   typ.Sequence[NDArray],   # r = probe(x) − y, len=d, elm_shape=W+C+(Ni,)
        weight:     typ.Optional[typ.Any] = None,   # per-mode residual weight ω, 1-D (d,); None = 1 (unweighted)
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ) (ragged only)
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The Gauss-Newton model of a ``probe`` least-squares objective at ``x``, on ``geometry``.

    Like :py:func:`apply_model` but the measurements are **probes** -- vector-valued (one free mode each),
    so ``residual`` is a sequence of ``d`` arrays (``elm_shape = W+C+(Ni,)``). Optionally **per-mode**
    weighted: the objective is ``½ Σ_i ‖ω_i r_i‖²`` over the ``d`` per-mode probe residuals, so a 1-D
    weight ``ω`` of shape ``(d,)`` up- or down-weights each mode's data (e.g. inverse-scale / inverse-noise
    balancing). Plain probe has no order axis, so ``weight`` is a **1-D** per-mode vector (a 2-D ``(d, 1)``
    is rejected); for per-order weighting use :py:func:`probe_derivatives_model`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.fitting as fitting
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(15, N) for N in (6, 7, 8)]
    >>> r = [np.random.randn(15, N) for N in (6, 7, 8)]     # per-mode probe residual (a list of d)

    A per-mode weight down-weights mode 0 and up-weights mode 2 in the objective ``½ Σ_i ‖ω_i r_i‖²``:

    >>> model = fitting.probe_model(t3m.MANIFOLD, x, ww, r, weight=[0.5, 1.0, 2.0])
    >>> print(model.gradient.is_gauged())
    True
    >>> p = t3m.MANIFOLD.randn(model.frame)
    >>> bool(np.allclose(float(model.gn_quadratic(p)), float(p.corewise_inner(model.gn_hessian(p)))))
    True

    A 2-D weight is rejected -- plain probe has no order axis:

    >>> fitting.probe_model(t3m.MANIFOLD, x, ww, r, weight=[[0.5], [1.0], [2.0]])   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: plain probe takes a 1-D per-mode residual weight of shape (d,)
    '''
    wm = _canonical_weight(weight, 'probe', len(ww), 0)
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        _reject_uniform_regularizer(regularizer)
        return _uniform_model(geometry, x, 'probe', ww, residual, weight=wm)
    frame = _ragged_frame(geometry, x)
    kind = fb.probe_kind(wm)
    return GaussNewtonModel(geometry, frame, kind, ww, residual, kind.precompute(frame.data, ww), regularizer)


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
        weight:     typ.Optional[typ.Any] = None,  # ORDER-only residual weight ω, (order+1,); None = 1
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ) (ragged only)
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The Gauss-Newton model of an ``apply``-**derivatives** least-squares objective at ``x``: the
    symmetric directional derivatives (orders ``0..order``) of the all-modes apply, in direction ``P``.

    The all-modes apply contracts every mode into a scalar, so ``weight`` is **order-only** (a per-mode
    weight is a structural error -- mode weighting is defined only for probe; see
    :py:func:`probe_derivatives_model`).

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
    >>> p = t3m.MANIFOLD.randn(model.frame)
    >>> bool(np.allclose(float(model.gn_quadratic(p)), float(p.corewise_inner(model.gn_hessian(p)))))
    True
    '''
    wm = _canonical_weight(weight, 'apply_derivatives', len(ww), order)
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        _reject_uniform_regularizer(regularizer)
        return _uniform_model(geometry, x, 'apply_derivatives', (ww, pp), residual, order, wm)
    kind = fb.apply_derivatives_kind(order, wm)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, kind, (ww, pp), residual,
                            kind.precompute(frame.data, (ww, pp)), regularizer)


def entries_derivatives_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        index:      NDArray,                 # int, shape=(d,)+W -- the grid points
        pp:         typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:      int,
        residual:   NDArray,                 # RAW r = entries_derivatives(x) − y, shape (order+1)+W+C
        weight:     typ.Optional[typ.Any] = None,  # ORDER-only residual weight ω, (order+1,); None = 1
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ) (ragged only)
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The ``entries``-derivatives Gauss-Newton model -- like :py:func:`apply_derivatives_model` at integer
    grid points ``index``. Order-only ``weight`` (no mode axis -- mode weighting is probe-only).'''
    wm = _canonical_weight(weight, 'entries_derivatives', index.shape[0], order)
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        _reject_uniform_regularizer(regularizer)
        return _uniform_model(geometry, x, 'entries_derivatives', (index, pp), residual, order, wm)
    kind = fb.entries_derivatives_kind(order, wm)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, kind, (index, pp), residual,
                            kind.precompute(frame.data, (index, pp)), regularizer)


def probe_derivatives_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:         typ.Sequence[NDArray],   # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:      int,
        residual:   typ.Sequence[NDArray],   # RAW r = probe_derivatives(x) − y, len=d, elm_shape=(order+1)+W+C+(Ni,)
        weight:     typ.Optional[typ.Any] = None,  # residual weight ω[mode,order], (d,order+1) broadcast; None = 1
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ) (ragged only)
) -> typ.Union[GaussNewtonModel, UniformGaussNewtonModel]:
    '''The ``probe``-derivatives Gauss-Newton model -- vector-valued (one free mode per probe), so
    ``residual`` is a sequence of ``d`` jets. Probe has both a mode and an order axis, so ``weight`` is the
    full ``ω[mode, order]`` matrix ``(d, order+1)``: a bare row ``(order+1,)`` = per-order (broadcast over
    modes), a column ``(d, 1)`` = per-mode (broadcast over orders), a matrix = both. The objective is
    ``½ Σ_i ‖ω_i ⊙ r_i‖²`` over the ``d`` per-mode residual jets.'''
    wm = _canonical_weight(weight, 'probe_derivatives', len(ww), order)
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        _reject_uniform_regularizer(regularizer)
        return _uniform_model(geometry, x, 'probe_derivatives', (ww, pp), residual, order, wm)
    kind = fb.probe_derivatives_kind(order, wm)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, kind, (ww, pp), residual,
                            kind.precompute(frame.data, (ww, pp)), regularizer)


if jax_available:
    import jax

    # Register GaussNewtonModel as a jax pytree: the data (frame, sweep, sample, residual) are LEAVES, the
    # statics (geometry, kind) are aux_data. Because T3Tangent's frame is now a leaf too (see manifold.py),
    # nothing carries a frame as aux_data, so a model crossing a jit boundary does NOT recompile when the
    # frame changes -- `jit(lambda model, p: model.gn_hessian(p))(model, p)` compiles once and reuses across
    # outer steps. The sweep is a stored field (a leaf) so it is carried/reused, not recomputed inside the
    # trace. The same-frame guard is the numerical same-frame check (skips under the trace).
    jax.tree_util.register_pytree_node(
        GaussNewtonModel,
        lambda m: ((m.frame, m.sweep, m.sample, m.residual), (m.geometry, m.kind, m.regularizer)),
        lambda aux, children: GaussNewtonModel(aux[0], children[0], aux[1], children[2], children[3],
                                               children[1], aux[2]),
    )

    # UniformGaussNewtonModel: the data (frame -- itself a leaf pytree -- sweep, sample, residual) are LEAVES;
    # the aux is VALUE-HASHED (geometry singleton, kind_name, x0's rank masks, order, weight) so a rebuilt
    # model of the same rank is the SAME jit cache key -> `jit(lambda m, p: m.gn_hessian(p))` compiles once
    # across outer steps (the packed kind is rebuilt lazily from this aux; it can't be aux itself -- a fresh
    # closure would recompile). See the class docstring + docs/uniform_backend_jit_recipe.md.
    jax.tree_util.register_pytree_node(
        UniformGaussNewtonModel,
        lambda m: ((m.frame, m.sweep, m.sample, m.residual),
                   (m.geometry, m.kind_name, m.x0_masks, m.order, m.weight)),
        lambda aux, ch: UniformGaussNewtonModel(aux[0], ch[0], aux[1], aux[2], aux[3], aux[4],
                                                ch[2], ch[3], ch[1]),
    )
