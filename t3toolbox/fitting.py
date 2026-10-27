# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
'''Frontend for the least-squares fitting layer: one geometry-generic local model.

A :py:class:`~t3toolbox.fitting.LocalModel` is the local model of a least-squares objective
``½‖F(X) − y‖²`` at a fixed point, linearized through a **geometry**: it exposes the objective value
``c = ½‖r‖²``, the gradient ``g``, the Gauss-Newton Hessian action ``H p = JᵀJ p``, and the
quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p``. Built with ``hessian='exact'`` its
:py:meth:`~LocalModel.hessian` is the **exact** Riemannian Hessian instead (the Gauss-Newton term plus the
curvature term). It is the frontend twin of the backend
:py:class:`~t3toolbox.backend.optimizers.LocalModel` (same name, one level up). The model is generic over
two independent choices:

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
:py:mod:`t3toolbox.backend.fitting` and ``docs/fitting_and_optimization.md`` §4.3.

Jitting an optimizer
--------------------
``LocalModel`` **is** a registered jax pytree (the data -- ``frame``, ``sweep``, ``sample``,
``residual``, and an exact model's ``curvature`` -- are leaves; ``geometry`` / ``kind`` are static aux).
Crucially the frame flows as a *leaf*,
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

import dataclasses as dc
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
import t3toolbox.shared_geometry as sg
import t3toolbox.safety as safety
import t3toolbox.backend.fitting as fb
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.optimizers as bopt         # exact mode: its checks, CurvatureState, the backend LocalModel
import t3toolbox.backend.regularization as breg    # the backend geometry a Regularizer leans on (no cycle: backend never imports fitting)
import t3toolbox.backend.uniform_fitting as ufit
from t3toolbox.backend.common import *

__all__ = ['LocalModel',
           'apply_model', 'entries_model', 'probe_model',
           'apply_derivatives_model', 'entries_derivatives_model', 'probe_derivatives_model']


# --------------------------------------------------------------------------------------------------
# Layer dispatch for the model. ONE LocalModel serves both representations; these three helpers
# are everything that differs, and each is a question about the FRAME's layer -- so they dispatch on the
# frame type, the library's usual "infer from the input" rule. (The geometry could answer instead, but a
# SharedGeometry wraps either layer and would have to forward all three; the frame always knows.)
# --------------------------------------------------------------------------------------------------
def _is_uniform_frame(frame) -> bool:
    return isinstance(frame, ubv.UT3Frame)


def _require_at_frame(frame, p) -> None:
    '''Same-frame guard: a trial tangent must live at the model's frame. The ``is`` fast-path, else a
    NUMERICAL frame compare (safe mode, eager-only; skips under ``safety.unsafe()`` / a jax trace), like
    ``T3Tangent``'s same-tangent-space guard. On the uniform layer the compare is on the four supercores
    (``frame.data[:4]``) -- the full ``.data`` carries the int-tuple ``shape`` that safety's array compare
    cannot take -- mirroring ``UT3Tangent._check_same_tangent_space``.'''
    a, b = (frame.data[:4], p.frame.data[:4]) if _is_uniform_frame(frame) else (frame.data, p.frame.data)
    if p.frame is not frame:
        # structural first, always: different core shapes are different tangent spaces whatever the values
        # (the numerical compare below is skipped under unsafe()/jit and must not be the only guard)
        sa = tuple(tuple(tuple(c.shape) for c in fam) if isinstance(fam, (list, tuple)) else tuple(fam.shape) for fam in a)
        sb = tuple(tuple(tuple(c.shape) for c in fam) if isinstance(fam, (list, tuple)) else tuple(fam.shape) for fam in b)
        if sa != sb:
            raise ValueError("trial tangent must live at the model's frame: its frame has different core "
                             "shapes (%s vs %s)" % (sb, sa))
    if not (p.frame is frame or safety.frames_equal_or_skip(a, b)):
        raise ValueError("trial tangent must live at the model's frame (it is at a different frame); "
                         "run inside safety.unsafe() to skip this numerical check")


def _tangent_at(frame, raw):
    '''Wrap raw backend variation coordinates as the frontend tangent of this frame's layer.

    Ragged: a ``(tucker_variations, tt_variations)`` core-tuple pair -> :py:class:`~t3toolbox.manifold.T3Tangent`.
    Uniform: a bare supercore pair -> :py:class:`~t3toolbox.uniform_manifold.UT3Tangent`, carrying the
    frame's gauge-shifted variation masks -- broadcast over a tangent stack ``K`` when the pair carries one
    (``K + C`` coordinates at a ``C`` frame, as a ``K``-stacked trial tangent's actions return).'''
    if _is_uniform_frame(frame):
        masks = ubv.UT3Variations._variation_masks_of(frame)
        K = tuple(raw[0].shape[1:raw[0].ndim - 2 - len(frame.stack_shape)])
        if K:
            masks = ut3m._broadcast_variation_masks_over_K(masks, K)
        return ut3m.UT3Tangent(frame, ubv.UT3Variations(raw[0], raw[1], frame.shape, masks))
    return t3m.T3Tangent(frame, bvf.T3Variations(*raw))


def _coordinates_of(p):
    '''The raw backend variation coordinates of a frontend tangent -- what the sampling kind consumes.'''
    return p.variations.supercores if isinstance(p, ut3m.UT3Tangent) else p.variations.data


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
    if 1 < m < d:
        # m > d is allowed: the extra rows are ignored (a continuation scheme can carry one per-mode weight
        # across stages that add modes; the backend truncates at apply time). Fewer rows than modes is a
        # structural error.
        raise ValueError("residual weight's mode dimension must be 1 or >= d=%d (extra rows are ignored); "
                         "got %d (shape %s)" % (d, m, wm.shape))
    if 'probe' not in kind_name and m > 1:          # apply/entries: no mode axis (mode weighting is probe-only)
        raise ValueError("apply / entries contract every mode into a scalar -- they have no mode axis, so a "
                         "per-mode weight (mode dim %d > 1) is undefined. Use an order-only weight "
                         "(order+1,); per-mode weighting is defined only for probe." % m)
    return wm


@dataclass(frozen=True)
class LocalModel:
    '''The local model of a least-squares objective at ``frame``, generic over the geometry: the objective
    value, the gradient, and the curvature actions there (the Gauss-Newton ``H = JᵀJ``: :py:meth:`gn_hessian`
    / :py:meth:`gn_quadratic`). The frontend twin of the backend
    :py:class:`~t3toolbox.backend.optimizers.LocalModel`, with ``T3Tangent`` / ``UT3Tangent`` in and out.

    Built by a sampling-kind factory (:py:func:`apply_model` / :py:func:`entries_model` /
    :py:func:`probe_model`), not directly. Holds the geometry, the frame ``frame = geometry.frame(X)``, the
    sampling ``kind``, the measurement ``sample`` (the probe/apply vectors ``ww`` or the integer grid
    ``index``), and the data residual ``r = F(X) − y``. The frame sweep, objective value, and gradient are
    cached on first use.

    **Two modes**, chosen by the factory's ``hessian`` argument and derived rather than stored
    (:py:attr:`is_exact` is ``curvature is not None``):

    * **Gauss-Newton** (``hessian='gauss_newton'``, the default): :py:meth:`hessian` / :py:meth:`quadratic`
      are :py:meth:`gn_hessian` / :py:meth:`gn_quadratic`.
    * **Exact** (``hessian='exact'``): :py:meth:`hessian` is the exact Riemannian Hessian -- the Gauss-Newton
      term plus the curvature term :py:meth:`curvature_hessian` (``Π Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π``; see
      :py:meth:`t3toolbox.backend.optimizers.Problem.local_model`), and :py:meth:`quadratic` likewise. The
      model's one misfit transpose is done at construction (``curvature``), so :py:attr:`gradient` is the
      Gauss-Newton gradient bit for bit. Only ``probe_derivatives`` implements it so far, ragged and uniform;
      the other kinds raise a structural error at construction. On a shared manifold geometry, ragged or
      uniform, the curvature goes through the tied lift (one factor direction per sharing group).

    ``gradient``, ``jacobian``, ``gn_hessian``, ``gn_quadratic``, ``objective_value`` and :py:meth:`retract`
    mean the same in both modes.

    **One class, both representations.** The factories dispatch on ``x``: a ragged
    ``TuckerTensorTrain`` gives a model on a ``T3Frame``, whose gradient and Hessian actions are of type
    ``T3Tangent``; a ``UniformTuckerTensorTrain`` gives one on a ``UT3Frame``, running on the packed
    supercores with ``UT3Tangent`` at the boundary. Everything between -- the objective, the gradient,
    the Gauss-Newton actions, the regularizer folding -- is the same code.

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

    The **uniform** layer is the same call with a uniform ``x`` (and a uniform geometry): the gradient
    comes back as a gauged ``UT3Tangent``, and ``pᵀHp = ‖J p‖²`` agrees with the Hessian action there too:

    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    >>> umodel = fitting.apply_model(ut3m.UNIFORM_MANIFOLD, ux, ww, r)
    >>> type(umodel).__name__, type(umodel.gradient).__name__
    ('LocalModel', 'UT3Tangent')
    >>> print(bool(umodel.gradient.is_gauged().all()))
    True
    >>> up = ut3m.UNIFORM_MANIFOLD.randn(umodel.frame)
    >>> bool(np.allclose(float(umodel.gn_quadratic(up)), float(up.corewise_inner(umodel.gn_hessian(up)))))
    True

    A trial step at a *different* frame is a structural error (the model is tied to its frame):

    >>> other = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> model.gn_hessian(t3m.MANIFOLD.randn(t3m.MANIFOLD.frame(other)))   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: trial tangent must live at the model's frame (it is at a different frame)
    '''

    geometry: typ.Any              # the geometry (MANIFOLD / COREWISE): supplies frame & project (Π)
    frame:    typ.Any              # = geometry.frame(X): a T3Frame (ragged) or UT3Frame (uniform)
    kind:     fb.SamplingKind      # the sampling kind; a uniform kind carries this model's fixed rank
    sample:   typ.Any              # ww / index (uniform: PACKED); the derivative kinds pair it with pp
    residual: typ.Any              # r = F(X) − y; shape W+C (apply / entries) or len=d, W+C+(Ni,) (probe)
    sweep:    typ.Any              # = kind.precompute(frame.data, sample); a FIELD (jax leaf) so it is
                                   # carried across a jit boundary and reused, not recomputed per matvec

    regularizer: typ.Any = None    # optional backend.regularization.Regularizer; ρ folded into obj/grad/hessian/quadratic/evaluate
    geometry_aux: typ.Any = None   # per-frame geometry companion (e.g. the SF-T3 SharedFrameData); a jax LEAF
    curvature:    typ.Any = None   # exact mode: the backend CurvatureState; a jax LEAF (None: Gauss-Newton)


    def __post_init__(self):
        if self.regularizer is not None:
            breg.require_unstacked_for_regularizer(self.frame.stack_shape, 'LocalModel')

    @property
    def is_exact(self) -> bool:
        '''Whether this model's Hessian is the exact one (built with ``hessian='exact'``) rather than
        Gauss-Newton. Derived, never stored: ``curvature is not None``.'''
        return self.curvature is not None

    @property
    def _has_curvature_term(self) -> bool:      # exact AND the term does not vanish at this frame (d = 1 manifold)
        return self.curvature is not None and self.curvature.geometry_data is not None

    def _require_exact(self, who: str) -> None:
        if self.curvature is None:
            raise ValueError("LocalModel.%s needs an exact-mode model (a *_model factory with hessian='exact'); "
                             "this one is Gauss-Newton" % who)

    @ft.cached_property
    def _blm(self):
        """This model as the backend :py:class:`~t3toolbox.backend.optimizers.LocalModel` -- the same frame,
        sweep, sample, residual, companions and curvature, on raw ``.data``. The curvature actions delegate
        to it, so the frontend runs the backend's own code for them."""
        return bopt.LocalModel(self._bgeom, self.kind, self.sample, self.frame.data, self.sweep, self.residual,
                               self._n_w, self.regularizer, self.geometry_aux, self.curvature)

    def _project(self, v):                       # Pi, with the once-per-model geometry companion
        if self.geometry_aux is not None:
            return self.geometry.project(v, shared_data=self.geometry_aux)
        return self.geometry.project(v)

    def _wrap(self, raw):                        # raw backend coordinates -> a gauged tangent at frame
        return self._project(_tangent_at(self.frame, raw))

    def _reg_tangent(self, raw):                 # raw backend coordinates -> a tangent at frame (ungauged)
        return _tangent_at(self.frame, raw)

    @ft.cached_property
    def _bgeom(self):
        """The backend geometry for this model -- the primitives the regularizer and the exact mode lean on.

        On the uniform layer it is built at this model's fixed rank, whose ``(shape, masks)`` the sampling
        kind already carries as fields; that is what retired the model's own ``x0_masks`` shadow field. The
        manifold one is built from the frame already in hand
        (:py:meth:`~t3toolbox.backend.geometry.UniformManifoldGeometryOps.from_frame`: its variation masks
        are the frame's own), not re-orthogonalized from the base point."""
        if not _is_uniform_frame(self.frame):
            return _backend_geometry_ops(self.geometry, self.frame.shape)
        if isinstance(self.geometry, sg.SharedGeometry):
            base, sharing_spec = self.geometry.base, self.geometry.sharing
        else:
            base, sharing_spec = self.geometry, None
        if base is ut3m.UNIFORM_MANIFOLD:
            return bgeo.UniformManifoldGeometryOps.from_frame(self.frame.data, self.kind.masks, sharing_spec)
        x0_data = (self.frame.data[0], self.frame.data[2], self.kind.shape, self.kind.masks)
        return bgeo.UniformCorewiseGeometryOps.from_point(x0_data, sharing_spec)   # no orthogonalization there

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
    def gradient(self):  # g = Π 𝒥ᵀ r (+ g_R), a tangent at frame
        '''The gradient ``g = geometry.project(𝒥ᵀ r)`` (the Gauss-Newton ``Jᵀr``), a tangent at frame,
        plus the regularizer gradient ``g_R`` when set.

        On the manifold geometry this is the gauged Riemannian gradient; on the corewise geometry it is
        the raw core gradient ``𝒥ᵀr`` (a tangent at ``(U,G,G,G)``, no ``Π``). The same in both modes: an
        exact model did its one misfit transpose at construction and reads it here, bit for bit.'''
        if self.curvature is not None:           # exact: the misfit gradient, already computed (by _wrap)
            g = _tangent_at(self.frame, self.curvature.misfit_gradient)
        else:
            g = self._wrap(self.kind.transpose(self.residual, self.sample, self.frame.data, self.sweep))
        if self.regularizer is not None:
            g = g + self._reg_tangent(self.regularizer.gradient(self._bgeom, self.frame.data,
                                                                aux=self.geometry_aux))
        return g

    def jacobian(
            self,
            p,      # a trial tangent at this model's frame
    ) -> NDArray:  # J p = 𝒥(Π p); apply/entries: shape W+C; probe: len=d, elm_shape=W+C+(Ni,)
        '''The linearized forward ``J p = 𝒥(Π p)`` (the Gauss-Newton Jacobian-vector product).

        ONE forward sweep -- no transpose ``𝒥ᵀ``, no gauge re-projection of the output, no tangent
        assembly. The general forward primitive: it gives the predicted residual ``r + J p`` (for
        trust-region / line-search predicted reduction) and, via :py:meth:`gn_quadratic`, the Gauss-Newton
        quadratic form. The result lives in the sample space (a scalar per sample for apply / entries; one
        vector per mode for probe).'''
        _require_at_frame(self.frame, p)
        return self.kind.forward(_coordinates_of(self._project(p)), self.sample, self.frame.data, self.sweep)

    def gn_quadratic(
            self,
            p,      # a trial tangent at this model's frame
    ) -> NDArray:  # pᵀ H p = ‖J p‖², shape C
        '''The Gauss-Newton quadratic form ``pᵀ H p = ‖J p‖²`` -- ONE forward sweep, NOT a Hessian apply.

        The cheap denominator for Cauchy / line-search step lengths: ``alpha = g.corewise_inner(g) /
        model.gn_quadratic(g)``. Because ``H = JᵀJ``, ``pᵀHp = (Jp)ᵀ(Jp) = ‖Jp‖²``, so this needs only the
        forward :py:meth:`jacobian` -- it avoids the transpose ``𝒥ᵀ`` and the ``H p`` tangent
        materialization that the equivalent ``p.corewise_inner(self.gn_hessian(p))`` would incur. When a
        ``regularizer`` is set this adds its ``⟨p, H_R p⟩`` term, consistent with :py:meth:`gn_hessian`.'''
        q = self.kind.sumsq(self.jacobian(p), self._n_w)
        if self.regularizer is not None:
            q = q + self.regularizer.quadratic(self._bgeom, self.frame.data, _coordinates_of(p),
                                               aux=self.geometry_aux)
        return q

    def gn_hessian(self, p):
        '''The Gauss-Newton Hessian action ``H p = Π 𝒥ᵀ 𝒥 Π p`` (``H = JᵀJ``, *not* the full Hessian),
        plus the regularizer term ``H_R p`` when set.

        Projects ``p`` (so the caller need not), applies the bare forward then transpose, and projects the
        result -- a tangent at frame. Symmetric. (Corewise: ``Π`` is the identity, so ``H p = 𝒥ᵀ 𝒥 p``,
        which is gauge-singular -- fine for first-order methods, needs regularization for Newton.) For the
        *scalar* quadratic form ``pᵀHp`` alone, prefer the cheaper :py:meth:`gn_quadratic`.'''
        _require_at_frame(self.frame, p)
        z = self.kind.forward(_coordinates_of(self._project(p)), self.sample, self.frame.data, self.sweep)
        Hp = self._wrap(self.kind.transpose(z, self.sample, self.frame.data, self.sweep))
        if self.regularizer is not None:
            Hp = Hp + self._reg_tangent(self.regularizer.hessian(self._bgeom, self.frame.data,
                                                                 _coordinates_of(p), aux=self.geometry_aux))
        return Hp

    def hessian(self, p):
        '''The Hessian action of this model's mode, a tangent at frame: :py:meth:`gn_hessian` on a
        Gauss-Newton model; on an exact one (``hessian='exact'``) the exact Riemannian Hessian
        ``gn_hessian(p) + curvature_hessian(p)``. Symmetric in both modes; the exact one is indefinite away
        from a minimizer.'''
        Hp = self.gn_hessian(p)
        if self._has_curvature_term:
            Hp = Hp + self.curvature_hessian(p)
        return Hp

    def quadratic(self, p) -> NDArray:  # pᵀ H p of this mode, shape K+C
        '''The quadratic form ``pᵀ H p`` of this model's mode: :py:meth:`gn_quadratic` on a Gauss-Newton
        model; ``gn_quadratic(p) + curvature_quadratic(p)`` on an exact one -- the objective's second
        derivative along the retraction (the T3-SVD retraction is second order).'''
        q = self.gn_quadratic(p)
        if self._has_curvature_term:
            q = q + self.curvature_quadratic(p)
        return q

    def curvature_hessian(self, p):
        '''The exact Hessian's curvature term alone, ``Π Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π p`` -- a gauged tangent
        at frame (zero where the term vanishes, the manifold at ``d = 1``). Exact models only: on a
        Gauss-Newton one it raises (structural). The backend's own code
        (:py:meth:`t3toolbox.backend.optimizers.LocalModel.curvature_hvp`) on this model's data.'''
        self._require_exact('curvature_hessian')
        _require_at_frame(self.frame, p)
        return _tangent_at(self.frame, self._blm.curvature_hvp(_coordinates_of(p)))

    def curvature_quadratic(self, p) -> NDArray:  # ⟨Πp, Lᵀ K_N L Πp⟩, shape K+C
        '''The curvature term's quadratic form ``⟨Π p, Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π p⟩`` (the part of
        :py:meth:`quadratic` that Gauss-Newton drops); it can be negative. Exact models only: on a
        Gauss-Newton one it raises (structural).'''
        self._require_exact('curvature_quadratic')
        _require_at_frame(self.frame, p)
        return self._blm.curvature_quadratic(_coordinates_of(p))

    def evaluate(self, p) -> NDArray:  # m(p), shape C
        '''The quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p`` of this model's mode, reusing the cached
        ``c`` and ``g``. Gauss-Newton (``H = JᵀJ``): one **forward** apply -- the quadratic term needs only
        ``½ pᵀ H p = ½‖𝒥 Π p‖²``, not a Hessian apply -- and the value equals ``½‖r + 𝒥 Π p‖²`` exactly
        (the objective is quadratic in the ambient tensor). Exact: plus ``½`` :py:meth:`curvature_quadratic`.
        ``p`` is projected here, shared by the linear term ``⟨g, Πp⟩`` and the quadratic. With a
        ``regularizer``, ``c`` and ``g`` already carry ``ρ`` / ``g_R``, so only the quadratic
        ``½⟨p, H_R p⟩`` is added here.'''
        _require_at_frame(self.frame, p)
        Pp = self._project(p)
        Jp = self.kind.forward(_coordinates_of(Pp), self.sample, self.frame.data, self.sweep)
        m = self.objective_value + self.gradient.corewise_inner(Pp) + 0.5 * self.kind.sumsq(Jp, self._n_w)
        if self.regularizer is not None:
            m = m + 0.5 * self.regularizer.quadratic(self._bgeom, self.frame.data, _coordinates_of(p),
                                                     aux=self.geometry_aux)
        if self._has_curvature_term:
            m = m + 0.5 * self.curvature_quadratic(p)
        return m

    def retract(self, p):
        '''The point reached by the step ``p`` from this model's frame: ``geometry.retract(p)``, with the
        once-per-model companion on a shared manifold geometry -- a ``TuckerTensorTrain`` (ragged) or a
        ``UniformTuckerTensorTrain`` (uniform). The same retraction in both modes; on the manifold it is the
        T3-SVD retraction, second order on a minimal, numerically full-rank frame, which is what makes the
        exact model's :py:meth:`quadratic` the objective's second derivative along it. The frontend twin of
        :py:meth:`t3toolbox.backend.optimizers.LocalModel.retract`; ``p`` is retracted as given (not
        projected), and the geometry's safe-mode preconditions apply.'''
        _require_at_frame(self.frame, p)
        if self.geometry_aux is not None:
            return self.geometry.retract(p, shared_data=self.geometry_aux)
        return self.geometry.retract(p)


def _ragged_frame(geometry, x: t3.TuckerTensorTrain) -> bvf.T3Frame:
    '''Validate + build the frame for a ragged model: a ragged ``x`` needs a ragged geometry
    (a singleton, or a :py:class:`~t3toolbox.shared_geometry.SharedGeometry` over one).'''
    if (geometry is not t3m.MANIFOLD and geometry is not t3m.COREWISE
            and not (isinstance(geometry, sg.SharedGeometry) and not geometry.is_uniform)):
        raise ValueError("a ragged TuckerTensorTrain requires a ragged geometry (manifold.MANIFOLD / "
                         "manifold.COREWISE, or a shared_geometry.SharedGeometry over one); for a "
                         "UniformTuckerTensorTrain use the uniform geometries "
                         "(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE, or a SharedGeometry "
                         "over one).")
    return geometry.frame(x)


def _ragged_geometry_aux(geometry, frame):
    '''The once-per-model geometry companion (SharedGeometry's precompute hook; None otherwise).'''
    if isinstance(geometry, sg.SharedGeometry):
        return geometry.precompute(frame)
    return None


def ragged_backend_geometry(geometry, shape=None):
    '''Map a ragged frontend geometry to its backend geometry (:py:mod:`t3toolbox.backend.geometry`),
    or ``None`` if ``geometry`` is not one.

    Returning ``None`` rather than raising is deliberate: the two callers -- the regularizer path here and
    the optimizer adapter in :py:mod:`t3toolbox.optimizers` -- want the same mapping but different error
    messages, and duplicating the mapping to get them is how the two copies drifted. A shared wrapper
    needs ``shape`` to canonicalize its partition.'''
    if geometry is t3m.MANIFOLD:
        return bgeo.ManifoldGeometryOps()
    if geometry is t3m.COREWISE:
        return bgeo.CorewiseGeometryOps()
    if isinstance(geometry, sg.SharedGeometry) and not geometry.is_uniform and shape is not None:
        base = bgeo.ManifoldGeometryOps() if geometry.base is t3m.MANIFOLD else bgeo.CorewiseGeometryOps()
        return base.with_sharing(geometry.sharing, shape)
    return None


def _backend_geometry_ops(geometry, shape=None):
    '''The ragged backend geometry for the regularizer, which lives in the backend and leans on
    ``point_norm_sq`` / ``point_tangent`` / ``project`` / ``inner`` (``docs/contributor/fitting_internals.md`` §Regularization).'''
    ops = ragged_backend_geometry(geometry, shape)
    if ops is None:
        raise ValueError("regularization requires a ragged geometry (manifold.MANIFOLD / COREWISE, "
                         "or a SharedGeometry over one).")
    return ops


def _with_curvature(
        model:              LocalModel,           # the Gauss-Newton model a factory just built
        hessian:            str,                  # 'gauss_newton' | 'exact' (case-insensitive)
        curvature_damping:  typ.Optional[float],  # exact only: relative to ‖X‖_HS; None = the geometry's default
        who:                str,                  # the factory, for the messages
) -> LocalModel:                                  # model itself (Gauss-Newton), or its exact-mode twin
    '''The factories' one exact-mode step: ``hessian='gauss_newton'`` returns ``model`` unchanged;
    ``'exact'`` returns it with the ``curvature`` child -- the backend
    :py:class:`~t3toolbox.backend.optimizers.CurvatureState` of the model's ONE misfit transpose (the
    Gauss-Newton gradient's own expression, so :py:attr:`LocalModel.gradient` is bit-identical) and the
    backend geometry's per-frame curvature data at ``curvature_damping``.

    The rules are the backend's (:py:meth:`t3toolbox.backend.optimizers.Problem.local_model`), with its
    checks and messages: an unknown mode raises; a ``curvature_damping`` with Gauss-Newton raises (nothing
    would read it), as it does on a corewise geometry (no lift to damp); a kind without ``corewise_hessian``
    or a geometry without the exact-mode members raises structurally, naming what is missing -- which is
    how the uniform kinds' interim stub surfaces (every uniform kind but ``probe_derivatives``'s).'''
    exact = bopt._check_hessian(hessian, who) == 'exact'
    if curvature_damping is not None and not exact:
        raise ValueError("%s: curvature_damping is only meaningful with hessian='exact'" % who)
    if not exact:
        return model
    geom = model._bgeom
    bopt._require_exact_capable(model.kind, geom)
    misfit_gradient = _coordinates_of(model._wrap(model.kind.transpose(model.residual, model.sample,
                                                                       model.frame.data, model.sweep)))
    geometry_data = geom.curvature_precompute(model.frame.data, misfit_gradient, model.geometry_aux,
                                              curvature_damping)
    return dc.replace(model, curvature=bopt.CurvatureState(misfit_gradient, geometry_data))


def _uniform_model(
        geometry,                                   # UNIFORM_MANIFOLD / UNIFORM_COREWISE (required for a uniform x)
        x:                  ut3.UniformTuckerTensorTrain,
        kind_name:          str,                    # 'apply'/'entries'/'probe' (+'_derivatives')
        sample:             typ.Any,                # ragged or packed (packed once here, mirror-tolerant)
        residual:           typ.Any,                # ragged or packed r = S(x) − y
        order:              typ.Optional[int] = None,
        weight:             typ.Optional[NDArray] = None,   # canonical 2-D ω[m,o] (from _canonical_weight), or None
        regularizer:        typ.Any = None,                 # optional backend.regularization.Regularizer
        hessian:            str = 'gauss_newton',           # or 'exact' (see _with_curvature)
        curvature_damping:  typ.Optional[float] = None,     # exact only; None = the geometry's default
) -> LocalModel:
    '''Assemble the uniform :py:class:`~t3toolbox.fitting.LocalModel`: build the frame, pack the
    loop-invariant sample + residual once (:py:func:`~t3toolbox.backend.uniform_fitting.pack_sample` /
    ``pack_data``), and precompute the frame sweep.

    The packed kind is stored directly. It used to be reconstructed on demand from four shadow fields
    (``kind_name`` / ``x0_masks`` / ``order`` / ``weight``) because a kind built out of closures could not
    be compared by value and would have recompiled as jit ``aux_data``; kinds are value-typed classes now,
    so the kind itself is the aux.'''
    if (geometry is not ut3m.UNIFORM_MANIFOLD and geometry is not ut3m.UNIFORM_COREWISE
            and not (isinstance(geometry, sg.SharedGeometry) and geometry.is_uniform)):
        raise ValueError("a UniformTuckerTensorTrain requires a uniform geometry "
                         "(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE, or a "
                         "shared_geometry.SharedGeometry over one).")
    N = x.N
    frame = geometry.frame(x)                              # UT3Frame (a SharedGeometry checks tied factors)
    geometry_aux = (geometry.precompute(frame)         # the once-per-model SF-T3 companion (or None)
                    if isinstance(geometry, sg.SharedGeometry) else None)
    packed_sample = ufit.pack_sample(kind_name, sample, N)
    packed_residual = ufit.pack_data(kind_name, residual, N)
    x0_data = (None, None, x.shape, x.masks.data)        # the kind builders use only (shape, masks)
    kind = (ufit.uniform_sampling_kind(kind_name, x0_data, weight) if order is None
            else ufit.uniform_derivatives_kind(kind_name, x0_data, order, weight))
    sweep = kind.precompute(frame.data, packed_sample)
    return _with_curvature(LocalModel(geometry, frame, kind, packed_sample, packed_residual, sweep,
                                      regularizer, geometry_aux),
                           hessian, curvature_damping, kind_name + '_model')


def _check_ragged_sample(x, kind_name, sample, residual, order=0):
    """Structural validation of a ragged factory's (sample, residual) against the point ``x`` -- named
    errors at construction instead of an einsum / index error deep inside the first gradient
    (the 2026-08-22 review, C13). ``W`` is read off the sample; ``C`` is the point's stack."""
    d, C = x.d, tuple(x.stack_shape)
    deriv = kind_name.endswith('_derivatives')
    base = kind_name[:-len('_derivatives')] if deriv else kind_name
    main, pp = (sample if deriv else (sample, None))
    if base == 'entries':
        index = np.asarray(main)
        if index.ndim < 1 or index.shape[0] != d:
            raise ValueError('%s_model: index must have shape (d,)+W with d=%d; got %s' % (kind_name, d, index.shape))
        W = tuple(index.shape[1:])
    else:
        if len(main) != d:
            raise ValueError('%s_model: expected %d sample vectors (one per mode), got %d' % (kind_name, d, len(main)))
        W = tuple(np.shape(main[0])[:-1])
        for i, w in enumerate(main):
            if np.shape(w)[-1] != x.shape[i] or tuple(np.shape(w)[:-1]) != W:
                raise ValueError('%s_model: sample vector %d has shape %s, expected W+(N_%d,) = %s'
                                 % (kind_name, i, np.shape(w), i, W + (x.shape[i],)))
    if pp is not None:
        if len(pp) != d:
            raise ValueError('%s_model: expected %d perturbation vectors, got %d' % (kind_name, d, len(pp)))
        for i, p in enumerate(pp):
            if np.shape(p)[-1] != x.shape[i] or tuple(np.shape(p)[:-1]) != W:
                raise ValueError('%s_model: perturbation vector %d has shape %s, expected W+(N_%d,) = %s'
                                 % (kind_name, i, np.shape(p), i, W + (x.shape[i],)))
    lead = (order + 1,) if deriv else ()
    if base == 'probe':
        if len(residual) != d:
            raise ValueError('%s_model: the probe residual must be a len=d list (d=%d), got len %d'
                             % (kind_name, d, len(residual)))
        for i, r in enumerate(residual):
            if tuple(np.shape(r)) != lead + W + C + (x.shape[i],):
                raise ValueError('%s_model: residual[%d] has shape %s, expected %s (= %sW+C+(N_%d,))'
                                 % (kind_name, i, np.shape(r), lead + W + C + (x.shape[i],),
                                    '(order+1,)+' if deriv else '', i))
    elif tuple(np.shape(residual)) != lead + W + C:
        raise ValueError('%s_model: residual has shape %s, expected %s (= %sW+C)'
                         % (kind_name, np.shape(residual), lead + W + C, '(order+1,)+' if deriv else ''))


def apply_model(
        geometry,                                    # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:                  typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:                 typ.Sequence[NDArray],   # sample vectors, len=d, elm_shape=W+(Ni,)
        residual:           NDArray,                 # r = apply(x) − y, shape W+C
        regularizer:        typ.Any = None,          # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)

        hessian:            str = 'gauss_newton',            # or 'exact' (case-insensitive): the model's mode
        curvature_damping:  typ.Optional[float] = None,      # exact only: the lift damping; None = the geometry's
) -> LocalModel:
    '''The local model of an all-modes ``apply`` least-squares objective at ``x``, on ``geometry``.

    Accepts a ragged ``TuckerTensorTrain`` (-> :py:class:`~t3toolbox.fitting.LocalModel`) or a uniform
    ``UniformTuckerTensorTrain``; the representation is inferred
    from ``x`` and the geometry must match. ``regularizer`` adds ``ρ(x)`` to the model (ragged or uniform).
    ``hessian`` / ``curvature_damping`` choose the model's mode (:py:class:`LocalModel`); the exact
    Hessian needs the kind's ``corewise_hessian``, which ``apply`` does not implement yet, so
    ``hessian='exact'`` raises a structural ``ValueError`` here (see :py:func:`probe_derivatives_model`).'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'apply', ww, residual, regularizer=regularizer,
                              hessian=hessian, curvature_damping=curvature_damping)
    _check_ragged_sample(x, 'apply', ww, residual)
    frame = _ragged_frame(geometry, x)
    return _with_curvature(LocalModel(geometry, frame, fb.APPLY, ww, residual, fb.APPLY.precompute(frame.data, ww),
                                      regularizer, _ragged_geometry_aux(geometry, frame)),
                           hessian, curvature_damping, 'apply_model')


def entries_model(
        geometry,                                    # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:                  typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        index:              NDArray,                 # int, shape=(d,)+W -- the grid points
        residual:           NDArray,                 # r = entries(x) − y, shape W+C
        regularizer:        typ.Any = None,          # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)

        hessian:            str = 'gauss_newton',            # or 'exact' (case-insensitive): the model's mode
        curvature_damping:  typ.Optional[float] = None,      # exact only: the lift damping; None = the geometry's
) -> LocalModel:
    '''The local model of an all-modes ``entries`` least-squares objective at ``x``, on ``geometry``.

    Identical to :py:func:`apply_model` but the measurements are tensor **entries** at integer grid points
    ``index`` (shape ``(d,)+W``) rather than applies against probe vectors. (``hessian='exact'`` is not
    implemented for ``entries`` yet: a structural ``ValueError``.)'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'entries', index, residual, regularizer=regularizer,
                              hessian=hessian, curvature_damping=curvature_damping)
    _check_ragged_sample(x, 'entries', index, residual)
    frame = _ragged_frame(geometry, x)
    return _with_curvature(LocalModel(geometry, frame, fb.ENTRIES, index, residual,
                                      fb.ENTRIES.precompute(frame.data, index), regularizer,
                                      _ragged_geometry_aux(geometry, frame)),
                           hessian, curvature_damping, 'entries_model')


def probe_model(
        geometry,                                    # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:                  typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:                 typ.Sequence[NDArray],   # probe vectors, len=d, elm_shape=W+(Ni,)
        residual:           typ.Sequence[NDArray],   # r = probe(x) − y, len=d, elm_shape=W+C+(Ni,)
        weight:             typ.Optional[typ.Any] = None,    # per-mode residual weight ω, 1-D (d,); None = 1 (unweighted)
        regularizer:        typ.Any = None,          # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)

        hessian:            str = 'gauss_newton',            # or 'exact' (case-insensitive): the model's mode
        curvature_damping:  typ.Optional[float] = None,      # exact only: the lift damping; None = the geometry's
) -> LocalModel:
    '''The local model of a ``probe`` least-squares objective at ``x``, on ``geometry``.

    Like :py:func:`apply_model` but the measurements are **probes** -- vector-valued (one free mode each),
    so ``residual`` is a sequence of ``d`` arrays (``elm_shape = W+C+(Ni,)``). Optionally **per-mode**
    weighted: the objective is ``½ Σ_i ‖ω_i r_i‖²`` over the ``d`` per-mode probe residuals, so a 1-D
    weight ``ω`` of shape ``(d,)`` up- or down-weights each mode's data (e.g. inverse-scale / inverse-noise
    balancing). Plain probe has no order axis, so ``weight`` is a **1-D** per-mode vector (a 2-D ``(d, 1)``
    is rejected); for per-order weighting use :py:func:`probe_derivatives_model`. (``hessian='exact'`` is
    not implemented for plain ``probe`` yet: a structural ``ValueError``; see :py:func:`apply_model`.)

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
        return _uniform_model(geometry, x, 'probe', ww, residual, weight=wm, regularizer=regularizer,
                              hessian=hessian, curvature_damping=curvature_damping)
    _check_ragged_sample(x, 'probe', ww, residual)
    frame = _ragged_frame(geometry, x)
    kind = fb.probe_kind(wm)
    return _with_curvature(LocalModel(geometry, frame, kind, ww, residual, kind.precompute(frame.data, ww),
                                      regularizer, _ragged_geometry_aux(geometry, frame)),
                           hessian, curvature_damping, 'probe_model')


# --------------------------------------------------------------------------------------------------
# Derivative sampling models (the symmetric directional-derivative jets of apply/entries/probe). Same
# LocalModel, a parameterized derivative kind (order + the per-order residual weight ω): the
# measurement is a jet (a leading order axis), the sample is the paired (ww/index, pp), and `residual`
# is RAW (r = S(x) − y); ω weights the objective ½‖ω⊙r‖² inside the kind. See docs/fitting_and_optimization.md §4.6.
# --------------------------------------------------------------------------------------------------
def apply_derivatives_model(
        geometry,                                    # MANIFOLD / COREWISE
        x:                  t3.TuckerTensorTrain,    # the current point
        ww:                 typ.Sequence[NDArray],   # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:                 typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:              int,                     # highest derivative order
        residual:           NDArray,                 # RAW r = apply_derivatives(x) − y, shape (order+1)+W+C
        weight:             typ.Optional[typ.Any] = None,    # ORDER-only residual weight ω, (order+1,); None = 1
        regularizer:        typ.Any = None,          # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)

        hessian:            str = 'gauss_newton',            # or 'exact' (case-insensitive): the model's mode
        curvature_damping:  typ.Optional[float] = None,      # exact only: the lift damping; None = the geometry's
) -> LocalModel:
    '''The local model of an ``apply``-**derivatives** least-squares objective at ``x``: the
    symmetric directional derivatives (orders ``0..order``) of the all-modes apply, in direction ``P``.

    The all-modes apply contracts every mode into a scalar, so ``weight`` is **order-only** (a per-mode
    weight is a structural error -- mode weighting is defined only for probe; see
    :py:func:`probe_derivatives_model`). (``hessian='exact'`` is not implemented for
    ``apply_derivatives`` yet: a structural ``ValueError``.)

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
        return _uniform_model(geometry, x, 'apply_derivatives', (ww, pp), residual, order, wm, regularizer=regularizer,
                              hessian=hessian, curvature_damping=curvature_damping)
    _check_ragged_sample(x, 'apply_derivatives', (ww, pp), residual, order)
    kind = fb.apply_derivatives_kind(order, wm)
    frame = _ragged_frame(geometry, x)
    return _with_curvature(LocalModel(geometry, frame, kind, (ww, pp), residual,
                                      kind.precompute(frame.data, (ww, pp)), regularizer,
                                      _ragged_geometry_aux(geometry, frame)),
                           hessian, curvature_damping, 'apply_derivatives_model')


def entries_derivatives_model(
        geometry,                                    # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:                  typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        index:              NDArray,                 # int, shape=(d,)+W -- the grid points
        pp:                 typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:              int,
        residual:           NDArray,                 # RAW r = entries_derivatives(x) − y, shape (order+1)+W+C
        weight:             typ.Optional[typ.Any] = None,    # ORDER-only residual weight ω, (order+1,); None = 1
        regularizer:        typ.Any = None,          # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)

        hessian:            str = 'gauss_newton',            # or 'exact' (case-insensitive): the model's mode
        curvature_damping:  typ.Optional[float] = None,      # exact only: the lift damping; None = the geometry's
) -> LocalModel:
    '''The ``entries``-derivatives local model -- like :py:func:`apply_derivatives_model` at integer
    grid points ``index``. Order-only ``weight`` (no mode axis -- mode weighting is probe-only).
    (``hessian='exact'`` is not implemented for ``entries_derivatives`` yet: a structural ``ValueError``.)'''
    wm = _canonical_weight(weight, 'entries_derivatives', index.shape[0], order)
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'entries_derivatives', (index, pp), residual, order, wm,
                              regularizer=regularizer, hessian=hessian, curvature_damping=curvature_damping)
    _check_ragged_sample(x, 'entries_derivatives', (index, pp), residual, order)
    kind = fb.entries_derivatives_kind(order, wm)
    frame = _ragged_frame(geometry, x)
    return _with_curvature(LocalModel(geometry, frame, kind, (index, pp), residual,
                                      kind.precompute(frame.data, (index, pp)), regularizer,
                                      _ragged_geometry_aux(geometry, frame)),
                           hessian, curvature_damping, 'entries_derivatives_model')


def probe_derivatives_model(
        geometry,                                    # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:                  typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:                 typ.Sequence[NDArray],   # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:                 typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:              int,
        residual:           typ.Sequence[NDArray],   # RAW r = probe_derivatives(x) − y, len=d, elm_shape=(order+1)+W+C+(Ni,)
        weight:             typ.Optional[typ.Any] = None,    # residual weight ω[mode,order], (d,order+1) broadcast; None = 1
        regularizer:        typ.Any = None,          # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)

        hessian:            str = 'gauss_newton',            # or 'exact' (case-insensitive): the model's mode
        curvature_damping:  typ.Optional[float] = None,      # exact only: the lift damping; None = the geometry's
) -> LocalModel:
    '''The ``probe``-derivatives local model -- vector-valued (one free mode per probe), so
    ``residual`` is a sequence of ``d`` jets. Probe has both a mode and an order axis, so ``weight`` is the
    full ``ω[mode, order]`` matrix ``(d, order+1)``: a bare row ``(order+1,)`` = per-order (broadcast over
    modes), a column ``(d, 1)`` = per-mode (broadcast over orders), a matrix = both. The objective is
    ``½ Σ_i ‖ω_i ⊙ r_i‖²`` over the ``d`` per-mode residual jets.

    ``hessian='exact'`` builds the exact-Hessian model (:py:class:`LocalModel`): :py:meth:`LocalModel.hessian`
    adds the curvature term to the Gauss-Newton one. ``curvature_damping`` (exact only) is the damping of
    the lift from frame coordinates to core variations, relative to ``‖X‖_HS`` -- ``None`` for the
    geometry's default (``backend.geometry.DEFAULT_CURVATURE_DAMPING``); a value with ``hessian='gauss_newton'``
    or on a corewise geometry raises, as nothing would read it. The ragged manifold and corewise geometries,
    the uniform ones (a uniform ``x``; per stack element the ragged model's) and the shared wrappers of all
    four support it (on a shared manifold the curvature goes through the tied lift).

    Examples
    --------
    On a noisy fit the exact model's quadratic form is the objective's second derivative along the
    retraction, where the Gauss-Newton one misses the curvature term:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.fitting as fitting
    >>> np.random.seed(0)
    >>> shape, ranks = (4, 5, 6), ((2, 2, 2), (1, 2, 2, 1))
    >>> ww, pp = [np.random.randn(20, N) for N in shape], [np.random.randn(20, N) for N in shape]
    >>> y = t3.TuckerTensorTrain.randn(shape, *ranks).probe_derivatives(ww, pp, 1)   # data from another tensor
    >>> def model_at(x, hessian='gauss_newton'):
    ...     r = [z - yi for z, yi in zip(x.probe_derivatives(ww, pp, 1), y)]
    ...     return fitting.probe_derivatives_model(t3m.MANIFOLD, x, ww, pp, 1, r, hessian=hessian)
    >>> x = t3.TuckerTensorTrain.randn(shape, *ranks)
    >>> gn, exact = model_at(x), model_at(x, hessian='exact')
    >>> gn.is_exact, exact.is_exact
    (False, True)
    >>> p = exact.gradient * (1.0 / float(t3m.MANIFOLD.norm(exact.gradient)))          # a unit tangent
    >>> f = lambda t: float(model_at(exact.retract(p * t)).objective_value)
    >>> fd = (f(1e-3) - 2 * f(0.0) + f(-1e-3)) / 1e-6                                   # d²/dt² f(retract(t p))
    >>> bool(np.isclose(float(exact.quadratic(p)), fd, rtol=1e-4))
    True
    >>> bool(np.isclose(float(gn.quadratic(p)), fd, rtol=1e-2))
    False

    ``hessian(p)`` is the matching action (``⟨p, H p⟩ == quadratic(p)``), the Gauss-Newton one plus the
    curvature term:

    >>> bool(np.isclose(float(p.corewise_inner(exact.hessian(p))), float(exact.quadratic(p))))
    True
    >>> bool(np.allclose(float(p.corewise_inner(exact.hessian(p) - exact.gn_hessian(p))),
    ...                  float(exact.curvature_quadratic(p))))
    True

    The curvature term exists only on an exact model:

    >>> gn.curvature_quadratic(p)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: LocalModel.curvature_quadratic needs an exact-mode model
    '''
    wm = _canonical_weight(weight, 'probe_derivatives', len(ww), order)
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'probe_derivatives', (ww, pp), residual, order, wm,
                              regularizer=regularizer, hessian=hessian, curvature_damping=curvature_damping)
    _check_ragged_sample(x, 'probe_derivatives', (ww, pp), residual, order)
    kind = fb.probe_derivatives_kind(order, wm)
    frame = _ragged_frame(geometry, x)
    return _with_curvature(LocalModel(geometry, frame, kind, (ww, pp), residual,
                                      kind.precompute(frame.data, (ww, pp)), regularizer,
                                      _ragged_geometry_aux(geometry, frame)),
                           hessian, curvature_damping, 'probe_derivatives_model')


if jax_available:
    import jax

    # Register LocalModel as a jax pytree: the data (frame, sweep, sample, residual) are LEAVES, the
    # statics (geometry, kind) are aux_data. Because T3Tangent's frame is now a leaf too (see manifold.py),
    # nothing carries a frame as aux_data, so a model crossing a jit boundary does NOT recompile when the
    # frame changes -- `jit(lambda model, p: model.gn_hessian(p))(model, p)` compiles once and reuses across
    # outer steps. The sweep is a stored field (a leaf) so it is carried/reused, not recomputed inside the
    # trace. The same-frame guard is the numerical same-frame check (skips under the trace).
    # ONE registration serves both representations: the model is layer-polymorphic, and a T3Frame and a
    # UT3Frame are each already a registered pytree that keeps its own masks in ITS aux.
    # The kind sits in the aux, so it must be VALUE-comparable. It is: kinds are frozen dataclasses whose
    # parameters are fields (backend/fitting.py), so a kind rebuilt per model is the same cache key. The
    # uniform kind also carries this model's fixed rank, which is why the model needs no mask field of its
    # own. See test_dispatch.test_jit_ragged_local_model_parameterized_kind.
    # The children are partitioned like the backend LocalModel's (`partition_static`): a uniform sweep
    # carries the mode sizes as Python ints, which stay static structure in the aux instead of being traced.
    # An exact model's `curvature` (a backend CurvatureState, itself a registered pytree) rides as one more
    # child: its arrays are leaves, and its PRESENCE is tree structure, so the mode is static under the
    # treedef with no mode aux -- a new frame or damping changes leaf values only.
    def _flatten_local_model(m):
        dyn, skel = partition_static((m.frame, m.sweep, m.sample, m.residual, m.geometry_aux, m.curvature))
        return dyn, (m.geometry, m.kind, m.regularizer, skel)

    def _unflatten_local_model(aux, children):
        frame, sweep, sample, residual, geometry_aux, curvature = rebuild_static(children, aux[3])
        return LocalModel(aux[0], frame, aux[1], sample, residual, sweep, aux[2], geometry_aux, curvature)

    jax.tree_util.register_pytree_node(LocalModel, _flatten_local_model, _unflatten_local_model)
