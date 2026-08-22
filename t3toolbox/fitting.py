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
import t3toolbox.shared_geometry as sg
import t3toolbox.safety as safety
import t3toolbox.backend.fitting as fb
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.regularization as breg    # the backend geometry a Regularizer leans on (no cycle: backend never imports fitting)
import t3toolbox.backend.uniform_fitting as ufit
from t3toolbox.backend.common import *

__all__ = ['GaussNewtonModel',
           'apply_model', 'entries_model', 'probe_model',
           'apply_derivatives_model', 'entries_derivatives_model', 'probe_derivatives_model']


# --------------------------------------------------------------------------------------------------
# Layer dispatch for the model. ONE GaussNewtonModel serves both representations; these three helpers
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
    if not (p.frame is frame or safety.frames_equal_or_skip(a, b)):
        raise ValueError("trial tangent must live at the model's frame (it is at a different frame); "
                         "run inside safety.unsafe() to skip this numerical check")


def _tangent_at(frame, raw):
    '''Wrap raw backend variation coordinates as the frontend tangent of this frame's layer.

    Ragged: a ``(tucker_variations, tt_variations)`` core-tuple pair -> :py:class:`~t3toolbox.manifold.T3Tangent`.
    Uniform: a bare supercore pair -> :py:class:`~t3toolbox.uniform_manifold.UT3Tangent`, carrying the
    frame's gauge-shifted variation masks.'''
    if _is_uniform_frame(frame):
        return ut3m.UT3Tangent(frame, ubv.UT3Variations(raw[0], raw[1], frame.shape,
                                                        ubv.UT3Variations._variation_masks_of(frame)))
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
    if m not in (1, d):
        raise ValueError("residual weight's mode dimension must be 1 or d=%d; got %d (shape %s)"
                         % (d, m, wm.shape))
    if 'probe' not in kind_name and m > 1:          # apply/entries: no mode axis (mode weighting is probe-only)
        raise ValueError("apply / entries contract every mode into a scalar -- they have no mode axis, so a "
                         "per-mode weight (mode dim %d > 1) is undefined. Use an order-only weight "
                         "(order+1,); per-mode weighting is defined only for probe." % m)
    return wm


@dataclass(frozen=True)
class GaussNewtonModel:
    '''The local Gauss-Newton model of a least-squares objective at ``frame``, generic over the geometry.

    Built by a sampling-kind factory (:py:func:`apply_model` / :py:func:`entries_model` /
    :py:func:`probe_model`), not directly. Holds the geometry, the frame ``frame = geometry.frame(X)``, the
    sampling ``kind``, the measurement ``sample`` (the probe/apply vectors ``ww`` or the integer grid
    ``index``), and the data residual ``r = F(X) − y``. The frame sweep, objective value, and gradient are
    cached on first use.

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
    ('GaussNewtonModel', 'UT3Tangent')
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


    def __post_init__(self):
        if self.regularizer is not None:
            breg.require_unstacked_for_regularizer(self.frame.stack_shape, 'GaussNewtonModel')

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
        """The backend geometry for this model -- the primitives the regularizer leans on.

        On the uniform layer it is built at this model's fixed rank, whose ``(shape, masks)`` the sampling
        kind already carries as fields; that is what retired the model's own ``x0_masks`` shadow field."""
        if not _is_uniform_frame(self.frame):
            return _backend_geometry_ops(self.geometry, self.frame.shape)
        if isinstance(self.geometry, sg.SharedGeometry):
            base, sharing_spec = self.geometry.base, self.geometry.sharing
        else:
            base, sharing_spec = self.geometry, None
        x0_data = (self.frame.data[0], self.frame.data[2], self.kind.shape, self.kind.masks)
        return (bgeo.UniformManifoldGeometryOps.from_point(x0_data, sharing_spec)
                if base is ut3m.UNIFORM_MANIFOLD
                else bgeo.UniformCorewiseGeometryOps.from_point(x0_data, sharing_spec))

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
        the raw core gradient ``𝒥ᵀr`` (a tangent at ``(U,G,G,G)``, no ``Π``).'''
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

    def evaluate(self, p) -> NDArray:  # m(p), shape C
        '''The quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p`` with ``H = JᵀJ``, reusing the cached
        ``c`` and ``g`` and one **forward** apply: the quadratic term needs only ``½ pᵀ H p = ½‖𝒥 Π p‖²``,
        not a Hessian apply. ``p`` is projected here, shared by the linear term ``⟨g, Πp⟩`` and the
        quadratic. Equals ``½‖r + 𝒥 Π p‖²`` exactly (the objective is quadratic in the ambient tensor).
        With a ``regularizer``, ``c`` and ``g`` already carry ``ρ`` / ``g_R``, so only the quadratic
        ``½⟨p, H_R p⟩`` is added here.'''
        _require_at_frame(self.frame, p)
        Pp = self._project(p)
        Jp = self.kind.forward(_coordinates_of(Pp), self.sample, self.frame.data, self.sweep)
        m = self.objective_value + self.gradient.corewise_inner(Pp) + 0.5 * self.kind.sumsq(Jp, self._n_w)
        if self.regularizer is not None:
            m = m + 0.5 * self.regularizer.quadratic(self._bgeom, self.frame.data, _coordinates_of(p),
                                                     aux=self.geometry_aux)
        return m


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
    ``point_norm_sq`` / ``point_tangent`` / ``project`` / ``inner`` (dev/archive/regularization_design.md §5a).'''
    ops = ragged_backend_geometry(geometry, shape)
    if ops is None:
        raise ValueError("regularization requires a ragged geometry (manifold.MANIFOLD / COREWISE, "
                         "or a SharedGeometry over one).")
    return ops


def _uniform_model(
        geometry,                            # UNIFORM_MANIFOLD / UNIFORM_COREWISE (required for a uniform x)
        x:          ut3.UniformTuckerTensorTrain,
        kind_name:  str,                     # 'apply'/'entries'/'probe' (+'_derivatives')
        sample:     typ.Any,                 # ragged or packed (packed once here, mirror-tolerant)
        residual:   typ.Any,                 # ragged or packed r = S(x) − y
        order:      typ.Optional[int] = None,
        weight:     typ.Optional[NDArray] = None,   # canonical 2-D ω[m,o] (from _canonical_weight), or None
        regularizer: typ.Any = None,                # optional backend.regularization.Regularizer
) -> GaussNewtonModel:
    '''Assemble the uniform :py:class:`GaussNewtonModel`: build the frame, pack the loop-invariant sample
    + residual once (:py:func:`~t3toolbox.backend.uniform_fitting.pack_sample` / ``pack_data``), and
    precompute the frame sweep.

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
    return GaussNewtonModel(geometry, frame, kind, packed_sample, packed_residual, sweep,
                            regularizer, geometry_aux)


def apply_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:         typ.Sequence[NDArray],   # sample vectors, len=d, elm_shape=W+(Ni,)
        residual:   NDArray,                 # r = apply(x) − y, shape W+C
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
) -> GaussNewtonModel:
    '''The Gauss-Newton model of an all-modes ``apply`` least-squares objective at ``x``, on ``geometry``.

    Accepts a ragged ``TuckerTensorTrain`` (-> :py:class:`GaussNewtonModel`) or a uniform
    ``UniformTuckerTensorTrain``; the representation is inferred
    from ``x`` and the geometry must match. ``regularizer`` adds ``ρ(x)`` to the model (ragged or uniform).'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'apply', ww, residual, regularizer=regularizer)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, fb.APPLY, ww, residual, fb.APPLY.precompute(frame.data, ww),
                            regularizer, _ragged_geometry_aux(geometry, frame))


def entries_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        index:      NDArray,                 # int, shape=(d,)+W -- the grid points
        residual:   NDArray,                 # r = entries(x) − y, shape W+C
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
) -> GaussNewtonModel:
    '''The Gauss-Newton model of an all-modes ``entries`` least-squares objective at ``x``, on ``geometry``.

    Identical to :py:func:`apply_model` but the measurements are tensor **entries** at integer grid points
    ``index`` (shape ``(d,)+W``) rather than applies against probe vectors.'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'entries', index, residual, regularizer=regularizer)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, fb.ENTRIES, index, residual,
                            fb.ENTRIES.precompute(frame.data, index), regularizer,
                            _ragged_geometry_aux(geometry, frame))


def probe_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:         typ.Sequence[NDArray],   # probe vectors, len=d, elm_shape=W+(Ni,)
        residual:   typ.Sequence[NDArray],   # r = probe(x) − y, len=d, elm_shape=W+C+(Ni,)
        weight:     typ.Optional[typ.Any] = None,   # per-mode residual weight ω, 1-D (d,); None = 1 (unweighted)
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
) -> GaussNewtonModel:
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
        return _uniform_model(geometry, x, 'probe', ww, residual, weight=wm, regularizer=regularizer)
    frame = _ragged_frame(geometry, x)
    kind = fb.probe_kind(wm)
    return GaussNewtonModel(geometry, frame, kind, ww, residual, kind.precompute(frame.data, ww),
                            regularizer, _ragged_geometry_aux(geometry, frame))


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
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
) -> GaussNewtonModel:
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
        return _uniform_model(geometry, x, 'apply_derivatives', (ww, pp), residual, order, wm, regularizer=regularizer)
    kind = fb.apply_derivatives_kind(order, wm)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, kind, (ww, pp), residual,
                            kind.precompute(frame.data, (ww, pp)), regularizer,
                            _ragged_geometry_aux(geometry, frame))


def entries_derivatives_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        index:      NDArray,                 # int, shape=(d,)+W -- the grid points
        pp:         typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:      int,
        residual:   NDArray,                 # RAW r = entries_derivatives(x) − y, shape (order+1)+W+C
        weight:     typ.Optional[typ.Any] = None,  # ORDER-only residual weight ω, (order+1,); None = 1
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
) -> GaussNewtonModel:
    '''The ``entries``-derivatives Gauss-Newton model -- like :py:func:`apply_derivatives_model` at integer
    grid points ``index``. Order-only ``weight`` (no mode axis -- mode weighting is probe-only).'''
    wm = _canonical_weight(weight, 'entries_derivatives', index.shape[0], order)
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'entries_derivatives', (index, pp), residual, order, wm, regularizer=regularizer)
    kind = fb.entries_derivatives_kind(order, wm)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, kind, (index, pp), residual,
                            kind.precompute(frame.data, (index, pp)), regularizer,
                            _ragged_geometry_aux(geometry, frame))


def probe_derivatives_model(
        geometry,                            # MANIFOLD / COREWISE (or the UNIFORM_* twin for a uniform x)
        x:          typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],   # the current point
        ww:         typ.Sequence[NDArray],   # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:      int,
        residual:   typ.Sequence[NDArray],   # RAW r = probe_derivatives(x) − y, len=d, elm_shape=(order+1)+W+C+(Ni,)
        weight:     typ.Optional[typ.Any] = None,  # residual weight ω[mode,order], (d,order+1) broadcast; None = 1
        regularizer: typ.Any = None,         # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
) -> GaussNewtonModel:
    '''The ``probe``-derivatives Gauss-Newton model -- vector-valued (one free mode per probe), so
    ``residual`` is a sequence of ``d`` jets. Probe has both a mode and an order axis, so ``weight`` is the
    full ``ω[mode, order]`` matrix ``(d, order+1)``: a bare row ``(order+1,)`` = per-order (broadcast over
    modes), a column ``(d, 1)`` = per-mode (broadcast over orders), a matrix = both. The objective is
    ``½ Σ_i ‖ω_i ⊙ r_i‖²`` over the ``d`` per-mode residual jets.'''
    wm = _canonical_weight(weight, 'probe_derivatives', len(ww), order)
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return _uniform_model(geometry, x, 'probe_derivatives', (ww, pp), residual, order, wm, regularizer=regularizer)
    kind = fb.probe_derivatives_kind(order, wm)
    frame = _ragged_frame(geometry, x)
    return GaussNewtonModel(geometry, frame, kind, (ww, pp), residual,
                            kind.precompute(frame.data, (ww, pp)), regularizer,
                            _ragged_geometry_aux(geometry, frame))


if jax_available:
    import jax

    # Register GaussNewtonModel as a jax pytree: the data (frame, sweep, sample, residual) are LEAVES, the
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
    # own. See test_dispatch.test_jit_ragged_gauss_newton_model_parameterized_kind.
    jax.tree_util.register_pytree_node(
        GaussNewtonModel,
        lambda m: ((m.frame, m.sweep, m.sample, m.residual, m.geometry_aux),
                   (m.geometry, m.kind, m.regularizer)),
        lambda aux, children: GaussNewtonModel(aux[0], children[0], aux[1], children[2], children[3],
                                               children[1], aux[2], children[4]),
    )
