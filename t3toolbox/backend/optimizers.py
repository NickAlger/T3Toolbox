# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Backend-first optimization algorithms for fixed-rank T3 least-squares fitting.

These operate on **raw cores / tangent tuples** via backend functions only -- no frontend imports
(no ``TuckerTensorTrain`` / ``T3Tangent`` / ``fitting.LocalModel`` / geometry classes). A raw-``.data``
user can call them directly; the frontend ``optimizers.py`` is a thin adapter that validates the input
once and calls the same backend functions (design: ``docs/contributor/fitting_internals.md`` §"Backend-first").

Because the numerical safety preconditions live only in the frontend, this layer is **check-free** of
them (structural shape guards inside the backend functions remain, and are jit-safe). So ``jit`` needs no
``unsafe()`` wrapping; the ``use_jit`` machinery is a thin jax-only layer on top of the ordinary
numpy/jax dispatch (the per-step kernels jit via ``_jitted``; the inner CG runs as a
``lax.while_loop`` -- eager-vs-jit agreement is covered in ``tests/backend/test_optimizers.py``).

An optimizer touches four things (graded messages, P16), and knows nothing of what is fitted:

* ``Geometry``    -- the *where you optimize* axis: frame / project / retract / inner on raw data.
  This module declares only the **protocol** it needs; the T3 implementations live in
  :py:mod:`t3toolbox.backend.geometry`, so nothing here depends on the T3 representation.
* ``Oracle``      -- *what is fitted*: one operation, ``query``, which answers the values an optimizer asks for at a
  frame (``f``, ``g``, ``JtJp``, ``Jp_norm_sq``, and the optional ones). The built-in one is
  :py:class:`~t3toolbox.backend.fitting.LeastSquaresOracle`. Its data are a **batch** the optimizers pass through
  unread (a :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` for the built-in oracle).
* ``Regularizer`` -- an optional term ``rho(x)`` of the objective
  (:py:mod:`t3toolbox.backend.regularization`).
* ``LocalModel``  -- the geometry, the oracle, the batch and the regularizer at one frame, built by
  :py:func:`local_model` (``.gradient`` / ``.objective`` / ``.hvp`` / ``.gn_quadratic`` / ``.retract``; Gauss-Newton
  by default, the exact Riemannian Hessian with ``hessian='exact'``), the backend twin of the frontend
  ``fitting.LocalModel``.

The value of the objective at a point with no model is :py:func:`point_objective` (a trial point, a full-batch
check). Tangent vectors are raw ``(tucker_var, tt_var)`` tuples; vector arithmetic is the ``corewise`` ops.
"""
import dataclasses as dc
import functools as ft
import inspect
import math
import numbers
import time
import typing as typ
import warnings

import numpy as np

from t3toolbox.backend.common import *
from t3toolbox.backend.regularization import Regularizer, IdentityRegularizer   # re-exported for backend users
from t3toolbox.backend.regularization import require_unstacked_for_regularizer
import t3toolbox.corewise as cw

__all__ = [
    'LocalModel',
    'CurvatureState',
    'NewtonInfo',
    'NewtonTiming',
    'PhaseSeconds',
    'Regularizer',
    'IdentityRegularizer',
    'local_model',
    'point_objective',
    'flat_draw',
    'gradient_descent',
    'mc_sgd',
    'adam',
    'newton_cg',
    'padded_start',
    'Oracle',
]

Tangent = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]   # (tucker_variations, tt_variations)
Cores = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]     # (tucker_cores, tt_cores) -- a point


class Geometry(typ.Protocol):
    """What an optimizer needs of a geometry -- declared here, by the consumer, so this module stays
    independent of any particular tensor representation.

    The T3 implementations are in :py:mod:`t3toolbox.backend.geometry`
    (``ManifoldGeometryOps`` / ``CorewiseGeometryOps`` + their uniform twins). Anything supplying these
    methods works: the optimizers only ever build tangents through ``project`` / ``retract`` /
    ``point_tangent`` and combine them with the layer-agnostic ``corewise`` tree arithmetic.

    Implementations should hash and compare **by value** (their parameters, not their identity) --
    a geometry rides as jax ``aux_data``, so identity semantics recompile on every rebuild. See
    :py:class:`~t3toolbox.backend.common.ValueHashedFields`.

    ``frame(x_cores)`` returns the pair ``(frame, centers)`` (graded messages, P15): the linearization frame, and its
    centers ``(Vc, Hc)`` (on the uniform layout the two supercores), from the same sweep: at every site the center in
    the frame's hole is the point. They are not the point as a tangent (read so, ``(frame, centers)`` is ``2d`` times
    the point; the point as a tangent is ``point_tangent(frame)``). A corewise geometry's frame is the point itself,
    as the engine reads a corewise call (ragged ``(B, G)``, uniform ``(B_sc, G_sc, shape, masks)``), and its centers
    are ``None``. An oracle may read the probe ``z`` through the centers (``Vc``) instead of
    the base point's sweep.

    ``point_data(x_cores)`` returns the point as an oracle reads it, a corewise frame: on the ragged layout the cores
    themselves, on the uniform layout ``(B_sc, G_sc, shape, masks)`` with the geometry's own shape and point masks (the
    bare supercore pair does not carry them). On a corewise geometry it is ``frame(x_cores)[0]``. It is how the
    optimizers ask for the objective at a point with no local model (:py:func:`point_objective`: a trial point, a
    full-batch check), so the point is never sampled through a frame of it.

    **Optional member** (looked up duck-typed, ``getattr(geom, 'zero_point_frame', None)``; a geometry
    without it keeps working):

    * ``zero_point_frame(frame)`` -- the canonical frame at the zero tensor with ``frame``'s structure, or
      ``None`` if this geometry has none. It is where ``newton_cg`` / ``gradient_descent`` evaluate the
      default Newton reference ``‖g(0)‖`` (``g0_reference``). The manifold geometries define one; the
      corewise geometries return ``None`` (the zero cores are a critical point there); a geometry without
      the member, or one returning ``None``, gets the warm-start reference by default.

    **The start-frame members** -- optional, looked up duck-typed, needed only by the features that use them (an
    absent member raises a ``ValueError`` naming it); the manifold geometries define them, the corewise ones do not:

    * ``with_frame(frame)`` -- this geometry at ``frame``'s structure (the uniform geometry's variation masks are
      the frame's own; the ragged geometry returns itself).
    * ``require_frame_structure(x_cores, frame)`` -- raises ``ValueError`` unless ``frame`` has the structure of
      this geometry's frame at ``x_cores``, from integers and held masks only (never orthogonalizes); needed by
      ``local_model(frame=)``.
    * ``aligned_padded_frame(x0, x_old, project, hint, oversampling, oversampled_passes, max_passes, tol,
      alternate, seed, use_jit)`` -- ``(frame, centers, movements)``: the padded frame of ``x0`` from ``x_old``
      (``None``: the zero tensor) aligned with ``hint`` through ``project(frame, hint)``, its centers as ``frame(x)``
      returns them (the old point's zero-padded; zeros from the zero tensor; ``None`` if the geometry has none), and
      the movements; needed by ``padded_start``, ``newton_cg(padded_from=)`` and ``g0_reference='cone'``. One more
      keyword is optional, ``timing=False``: with ``timing=True`` the member returns a fourth element, the seconds
      per pass (``None`` when it cannot read them,
      as under the uniform whole-loop jit). The optimizers pass ``timing`` only to a member whose signature names it
      (a ``**kwargs`` catch-all does not count), so a member without it keeps working: ``newton_cg(timing=True)``,
      the only caller that asks, then reports ``alignment_passes`` and ``reference_passes`` as ``None``.

    **The exact-mode members** -- needed **only** for ``hessian='exact'``.
    ``local_model(hessian='exact')`` checks that all four exist (a structural ``ValueError`` naming
    the missing ones), so a Gauss-Newton-only custom geometry keeps working:

    * ``curvature_precompute(frame, misfit_gradient, aux, curvature_damping)`` -- the per-frame curvature
      data, built once per local model from the model's one misfit gradient ``Pi J^T omega^2 r``, this frame's
      ``aux`` and the solver's ``curvature_damping`` (a relative float, or ``None`` for the geometry's own
      default; passed through opaquely); returns ``None`` iff the curvature term vanishes at this frame.
      The returned data may carry two read-only diagnostics, ``sigma_min`` (the smallest lift singular
      value) and ``eps`` (the absolute damping used), which ``newton_cg`` reports as ``lift_sigma_min`` /
      ``lift_eps`` (looked up duck-typed; absent means ``None``).
    * ``lift(frame, variations, curvature_data)`` -- tangent coordinates -> raw core variations of
      ``base_point(frame)`` (stack ``K + C``).
    * ``lift_transpose(frame, core_variations, curvature_data)`` -- its adjoint, raw -> UNGAUGED coordinates.
    * ``tangential_curvature(frame, core_directions, curvature_data)`` -- the raw ``K_{grad f} theta_hat`` term
      subtracted from the sampling curvature, or ``None`` when there is none (a corewise geometry)."""

    def frame(self, x_cores):                          # x_cores -> (the linearization frame, its centers or None)
        ...

    def base_point(self, frame):                       # frame -> the point X the frame is attached to
        ...

    def stack_shape(self, x_cores):                    # x_cores -> the frame stack C
        ...

    def precompute(self, frame):                       # frame -> per-frame companion (None if none)
        ...

    def project(self, frame, variations, aux=None):    # gauge Pi (identity for corewise)
        ...

    def retract(self, frame, variations, aux=None):    # chart step -> new x_cores
        ...

    def inner(self, a, b):                             # coordinate <.,.> on tangents; PER-ELEMENT over
        ...                                            #   leading stacks (0-d scalar unstacked)

    def point_norm_sq(self, x_cores):                  # ‖X‖² in the coordinate metric (regularizer);
        ...                                            #   per-element over the stack C

    def point_tangent(self, frame):                    # X as a gauged tangent v_X (regularizer)
        ...

    def point_data(self, x_cores):                     # x_cores -> the point as an oracle reads it (a corewise frame)
        ...


# --------------------------------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------------------------------
class Oracle(typ.Protocol):
    """What an optimizer asks of a fit (graded messages, Part 1; the fitting layer's extension point): one
    operation, ``query``, at a frame. Declared here, by the consumer, as documentation. The built-in implementation is
    :py:class:`~t3toolbox.backend.fitting.LeastSquaresOracle`.

    An oracle is a frozen description with no arrays (it rides as jax ``aux_data``; value-hashed); its data are its
    ``batch``, which the optimizers pass through unread (``None`` for an oracle with no data). ``query`` returns the
    values of ``want``, aligned with it, and ``held``: anything the oracle wants back at the same frame and batch
    (O-8), passed back unread by the optimizers; in a compiled program it must be something jax can carry (arrays,
    tuples, ``None`` or a registered record, O-30). The required outputs are ``f``, ``g``, ``JtJp`` and
    ``Jp_norm_sq``; ``optional_outputs`` names which of ``Kp``, ``Jp``, ``r``, ``r_block_sumsq``, ``y_block_sumsq``
    and ``f_resolution`` it supplies (O-7). The blocks are returned before the geometry's projection. The two block
    sums are what the display divides, per block (:py:func:`~t3toolbox.backend.optimizer_display.relative_errors`):
    the least-squares oracle's are ``<r, M r>`` and ``<y, M y>``, weighted and scaled like ``f``, so a user oracle
    that returns plain sums of squares gets plain ratios.

    **What the optimizers rely on.** ``held`` is valid for one frame and one batch: the optimizers hand a model's
    ``held`` back only to queries at that model's frame on that model's batch, and never across a draw, a trial point,
    a new model, a validation batch or an alignment pass (nothing in the record says which frame or batch it is
    for). ``f`` and the blocks are per element of the frame stack ``C``. A drawn batch's ``f`` and its blocks
    estimate the full batch's (the least-squares oracle's ``minibatch_scale``, O-21); the optimizers add the
    regularizer unscaled. ``Kp`` is needed by ``hessian='exact'`` and ``'switch'``, ``Jp`` by
    :py:meth:`LocalModel.jacobian` and ``r`` by :py:attr:`LocalModel.residual` (each checked structurally where it
    is needed). Without ``f_resolution`` the trust region's floor is ``eps |f|``, ``eps`` read off ``f``'s dtype.

    **Optional members** ``check_batch(batch, shape, stack_shape=()) -> None`` (O-29): the structural checker of a
    batch against the mode sizes ``shape`` and the model's stack of base points ``stack_shape`` (``C``; ``()`` for an
    unstacked model), which the frontend calls, always with the model's ``stack_shape``, where a batch enters when the
    oracle has one, and nothing about the batch otherwise; and
    ``n_measurements(batch)`` with ``take(batch, indices)``, the default draw of ``mc_sgd`` and ``adam``
    (:py:func:`flat_draw`), which without them need a ``draw``.
    """
    optional_outputs: typ.ClassVar[typ.Tuple[str, ...]]

    def query(
            self,
            frame:            typ.Tuple,                      # a manifold frame or a corewise point
            batch:            typ.Any,                        # opaque to the optimizers; None if it has no data
            want:             typ.Tuple[str, ...],            # required: f, g, JtJp, Jp_norm_sq; optional: Kp, Jp, r,
                                                              #   r_block_sumsq, y_block_sumsq, f_resolution
            Vc:               typ.Optional[typ.Sequence[NDArray]] = None,
            variations:       typ.Optional[typ.Tuple] = None, # (V, H): JtJp, Jp_norm_sq, Jp
            core_directions:  typ.Optional[typ.Tuple] = None, # (D_dot, L_dot): Kp
            held:             typ.Any = None,                 # what an earlier query at this frame and batch returned
    ) -> typ.Tuple[
            typ.Tuple[typ.Any, ...],                          # values, aligned with want; blocks before projection
            typ.Any,                                          # held: any object jax can carry in a compiled program
    ]:
        ...


_HESSIAN_MODES = ('gauss_newton', 'exact')                # a local model's modes
_NEWTON_HESSIAN_MODES = _HESSIAN_MODES + ('switch',)       # newton_cg's: 'switch' is a solver mode, not a model's
_EXACT_GEOMETRY_MEMBERS = ('curvature_precompute', 'lift', 'lift_transpose', 'tangential_curvature')


def _check_hessian(
        hessian:  str,                                     # one of `modes` (case-insensitive)
        who:      str,                                     # the caller, for the message
        modes:    typ.Tuple[str, ...] = _HESSIAN_MODES,    # the caller's allowed modes
) -> str:                                                  # the lower-cased mode
    """Validate a ``hessian`` mode selector (structural: anything else raises, a ``bool`` included -- the
    keyword is the solver's string mode everywhere). ``modes`` are the caller's: a local model's two by default,
    ``newton_cg``'s three (``_NEWTON_HESSIAN_MODES``)."""
    if not isinstance(hessian, str) or hessian.lower() not in modes:
        raise ValueError("%s: hessian must be %s or %r (case-insensitive); got %r"
                         % (who, ', '.join(map(repr, modes[:-1])), modes[-1], hessian))
    return hessian.lower()


def _require_output(
        oracle,                   # an Oracle
        name:     str,            # an optional output: 'Kp', 'Jp', 'r', ...
        needed:   str,            # what needs it, for the message
) -> None:
    """Structural: the oracle lists ``name`` in its ``optional_outputs`` (O-7), else a ``ValueError`` before any work."""
    if name not in tuple(getattr(oracle, 'optional_outputs', ())):
        raise ValueError("%s does not supply %r (it is not in its optional_outputs), which %s needs"
                         % (type(oracle).__name__, name, needed))


def _require_exact_capable(
        oracle,                   # the fit's Oracle
        geom,                     # the fit's Geometry
        hessian:  str = 'exact',  # the mode that needs them, for the message: 'exact' | 'switch'
) -> None:
    """The structural checks of ``hessian='exact'`` (and ``'switch'``), run before any work: the oracle supplies
    ``Kp`` (it lists it in ``optional_outputs``, O-7) and the geometry has the four exact-mode members."""
    _require_output(oracle, 'Kp', "hessian=%r (it still works with hessian='gauss_newton')" % hessian)
    missing = [name for name in _EXACT_GEOMETRY_MEMBERS if not callable(getattr(geom, name, None))]
    if missing:
        raise ValueError("%s does not implement %s, which hessian=%r needs (the exact-mode geometry "
                         "members: %s); it still works with hessian='gauss_newton'"
                         % (type(geom).__name__, ', '.join(missing), hessian, ', '.join(_EXACT_GEOMETRY_MEMBERS)))


@dc.dataclass(frozen=True, eq=False)   # eq=False -> identity hash/eq: array fields (a leaf container)
class CurvatureState:
    """The exact-mode companion of a :py:class:`~t3toolbox.backend.optimizers.LocalModel` (``hessian='exact'``):
    a pytree LEAF beside ``held`` / ``geom_aux``, built once per model by :py:func:`local_model`.

    ``misfit_gradient`` is the misfit's Riemannian gradient ``Pi J^T omega^2 r`` -- the projection of the build's
    one ``g``, read by both :py:attr:`~t3toolbox.backend.optimizers.LocalModel.gradient` and the geometry's curvature
    data, so exact mode costs no second transpose. ``geometry_data`` is ``geom.curvature_precompute(...)``;
    ``None`` means the curvature term vanishes at this frame (the manifold at ``d = 1``), and ``hvp`` is then
    the Gauss-Newton one."""
    misfit_gradient:  typ.Any   # Pi J^T omega^2 r (no regularizer): the projection of the build's g
    geometry_data:    typ.Any   # geom.curvature_precompute(frame, misfit_gradient, aux, curvature_damping)


@dc.dataclass(frozen=True)
class LocalModel:
    """The local model at one frame: the geometry, the oracle, the batch and the regularizer put together, and the
    holder of what the oracle computed there -- the backend twin of the frontend
    :py:class:`t3toolbox.fitting.LocalModel` (same name, one level down). Built by :py:func:`local_model`, never
    changed after: its ``f`` and ``held`` are the build's, and every method below is one oracle query at the frame on
    the batch, with ``held`` passed back and a literal ``want``.

    **Two modes**, derived rather than stored (:py:attr:`is_exact` is ``curvature is not None``):

    * **Gauss-Newton** (``hessian='gauss_newton'``, the default): ``hvp(p) = Pi J^T omega^2 J Pi p (+ H_R p)``, the
      oracle's ``JtJp`` at ``Pi p``, projected.
    * **Exact** (``hessian='exact'``): the exact Riemannian Hessian,

      ``hvp(p) = Pi [J^T omega^2 J Pi p + L^T (K_{A^T omega^2 r} - K_{grad f}) L Pi p] (+ H_R p)``,

      with ``L`` the geometry's lift from coordinates to raw core variations, ``K_{A^T omega^2 r}`` the oracle's
      ``Kp`` along the lifted direction (the second derivative of ``<omega^2 r, z>`` at the fixed residual; one query
      with ``JtJp``) and ``K_{grad f}`` the geometry's tangential curvature (Absil, Mahony & Trumpf (2013), "An
      extrinsic look at the Riemannian Hessian": the Weingarten term as ``<N, d^2 Phi>`` with ``N`` the normal part
      of ``A^T omega^2 r``). On a corewise geometry it is the plain Hessian of ``f o Phi`` (no lift, no tangential
      term).

    ``gn_quadratic`` (the oracle's ``Jp_norm_sq``) / ``jacobian`` / ``residual`` / ``retract`` mean the same in both
    modes (``gn_quadratic`` is always ``‖ω J Pi p‖² (+ <p, H_R p>)``); ``quadratic`` is the mode's model
    ``<p, H p>``. On a drawn batch every misfit value is the oracle's scaled one (an estimate of the full batch's),
    the regularizer's terms unscaled."""
    geom:     typ.Any        # a Geometry (backend.geometry.*GeometryOps); aux_data, so it must hash by VALUE
    oracle:   typ.Any        # an Oracle (backend.fitting.LeastSquaresOracle); aux_data, value-hashed
    batch:    typ.Any        # the oracle's data (a LeastSquaresBatch); leaves; held belongs to it
    frame:    typ.Tuple      # the frame: (D, U, L, R), a corewise point (B, G), or their uniform .data
    centers:  typ.Any        # (Vc, Hc) when the geometry returned them (or the caller gave them), else None
    held:     typ.Any        # what the build's query returned to hand back here (a HeldVariables): this frame, this batch
    f:        typ.Any        # the misfit at the build, shape C (the oracle's f; scaled on a drawn batch)

    regularizer: typ.Any = None   # optional backend.regularization.Regularizer; ρ folded into obj/grad/gn_quadratic/hvp
    geom_aux:    typ.Any = None   # = geom.precompute(frame) (per-frame companion, e.g. SF-T3); a leaf
    curvature:   typ.Any = None   # CurvatureState in exact mode, else None; a leaf

    @property
    def _Vc(self):                                       # the centers' Vc, what the oracle may read z through
        return None if self.centers is None else self.centers[0]

    @property
    def is_exact(self) -> bool:                          # exact mode (hessian='exact'): derived, never stored
        return self.curvature is not None

    @property
    def _has_curvature_term(self) -> bool:               # exact AND the term does not vanish at this frame
        return self.curvature is not None and self.curvature.geometry_data is not None

    def _require_exact(self, who: str) -> None:
        if self.curvature is None:
            raise ValueError("LocalModel.%s needs an exact-mode model (local_model(..., hessian='exact')); "
                             "this one is Gauss-Newton" % who)

    @property
    def misfit(self):                                    # ½‖ω⊙r‖²  (the weighted data misfit alone): the build's f
        return self.f

    @property
    def regularization(self):                            # ρ(X) at X = geom.base_point(frame); 0 if unregularized
        if self.regularizer is None:
            return 0.0
        return self.regularizer.value(self.geom, self.geom.base_point(self.frame))

    @property
    def objective(self):                                 # c = ½‖ω⊙r‖²  (+ ρ(X) if regularized)
        return self.misfit + self.regularization

    @property
    def residual(self):                                  # r = z(X) - y, of the type of z (the oracle's optional r)
        _require_output(self.oracle, 'r', 'LocalModel.residual')
        (r,), _ = self.oracle.query(self.frame, self.batch, ('r',), Vc=self._Vc, held=self.held)
        return r

    @property
    def gradient(self) -> Tangent:                       # g = Π 𝒥ᵀ ω² r  (+ g_R)
        if self.curvature is not None:                   # exact: the build's g, projected once already
            g = self.curvature.misfit_gradient
        else:
            (g,), _ = self.oracle.query(self.frame, self.batch, ('g',), Vc=self._Vc, held=self.held)
            g = self.geom.project(self.frame, g, aux=self.geom_aux)
        if self.regularizer is not None:
            g = cw.corewise_add(g, self.regularizer.gradient(self.geom, self.frame, aux=self.geom_aux))
        return g

    def jacobian(self, p: Tangent):                      # 𝒥 Π p, of the type of z with K (the oracle's optional Jp)
        _require_output(self.oracle, 'Jp', 'LocalModel.jacobian')
        pp = self.geom.project(self.frame, p, aux=self.geom_aux)
        (Jp,), _ = self.oracle.query(self.frame, self.batch, ('Jp',), Vc=self._Vc, variations=pp, held=self.held)
        return Jp

    def gn_quadratic(self, p: Tangent):                  # ‖ω 𝒥 Π p‖²  (+ ⟨p, H_R p⟩) -- the Cauchy denominator; K+C
        pp = self.geom.project(self.frame, p, aux=self.geom_aux)
        (q,), _ = self.oracle.query(self.frame, self.batch, ('Jp_norm_sq',), Vc=self._Vc, variations=pp,
                                    held=self.held)
        if self.regularizer is not None:
            q = q + self.regularizer.quadratic(self.geom, self.frame, p, aux=self.geom_aux)
        return q

    def hvp(self, p: Tangent) -> Tangent:                # GN: H p = Π 𝒥ᵀ ω² 𝒥 Π p  (+ H_R p)
        pp = self.geom.project(self.frame, p, aux=self.geom_aux)
        if not self._has_curvature_term:                 # exact: Π [𝒥ᵀω²𝒥 Πp + Lᵀ K_N L Πp]  (+ H_R p)
            (JtJp,), _ = self.oracle.query(self.frame, self.batch, ('JtJp',), Vc=self._Vc, variations=pp,
                                           held=self.held)
            Hp = self.geom.project(self.frame, JtJp, aux=self.geom_aux)
        else:
            theta_hat = self.geom.lift(self.frame, pp, self.curvature.geometry_data)   # raw core directions
            (JtJp, Kp), _ = self.oracle.query(self.frame, self.batch, ('JtJp', 'Kp'), Vc=self._Vc, variations=pp,
                                              core_directions=theta_hat, held=self.held)
            Hp = self.geom.project(self.frame, cw.corewise_add(JtJp, self._curvature_from(theta_hat, Kp)),
                                   aux=self.geom_aux)
        if self.regularizer is not None:
            Hp = cw.corewise_add(Hp, self.regularizer.hessian(self.geom, self.frame, p, aux=self.geom_aux))
        return Hp

    def _curvature_from(                                 # Lᵀ (K_{Aᵀω²r} − K_{grad f}) θ̂ -- UNGAUGED, K+C
            self,
            theta_hat:  Tangent,                         # the lifted direction L Π p (raw core directions)
            Kp:         Tangent,                         # the oracle's Kp along it: K_{Aᵀω²r} θ̂ (ω² inside)
    ) -> Tangent:
        geom, data = self.geom, self.curvature.geometry_data
        K_grad = geom.tangential_curvature(self.frame, theta_hat, data)           # K_{grad f} θ̂, or None
        K = Kp if K_grad is None else cw.corewise_sub(Kp, K_grad)
        return geom.lift_transpose(self.frame, K, data)

    def _curvature_block(self, pp: Tangent) -> Tangent:  # Lᵀ (K_{Aᵀω²r} − K_{grad f}) L pp -- UNGAUGED, K+C
        theta_hat = self.geom.lift(self.frame, pp, self.curvature.geometry_data)
        (Kp,), _ = self.oracle.query(self.frame, self.batch, ('Kp',), Vc=self._Vc, core_directions=theta_hat,
                                     held=self.held)
        return self._curvature_from(theta_hat, Kp)

    def curvature_hvp(self, p: Tangent) -> Tangent:      # Π Lᵀ K_N L Π p -- the exact Hessian's curvature term alone
        self._require_exact('curvature_hvp')
        pp = self.geom.project(self.frame, p, aux=self.geom_aux)
        if self.curvature.geometry_data is None:         # the term vanishes at this frame
            return cw.corewise_zeros_like(pp)
        return self.geom.project(self.frame, self._curvature_block(pp), aux=self.geom_aux)

    def quadratic(self, p: Tangent):                     # <p, H p> of this mode: gn_quadratic (+ curvature_quadratic)
        q = self.gn_quadratic(p)
        if self._has_curvature_term:
            q = q + self.curvature_quadratic(p)
        return q

    def curvature_quadratic(self, p: Tangent):           # <Π p, Lᵀ K_N L Π p>, per K+C element
        self._require_exact('curvature_quadratic')
        pp = self.geom.project(self.frame, p, aux=self.geom_aux)
        if self.curvature.geometry_data is None:         # the term vanishes: an exact 0 of the stack's shape
            return self.geom.inner(pp, cw.corewise_zeros_like(pp))
        return self.geom.inner(pp, self._curvature_block(pp))

    def retract(self, p: Tangent) -> Cores:              # chart step from this frame -> new x_cores
        return self.geom.retract(self.frame, p, aux=self.geom_aux)


def local_model(
        geom:               typ.Any,                       # a Geometry (backend.geometry.*GeometryOps)
        oracle:             typ.Any,                       # an Oracle (backend.fitting.LeastSquaresOracle)
        batch:              typ.Any,                       # its batch (a LeastSquaresBatch; None for an oracle with no data)
        x_cores:            Cores,                         # the point to linearize at (uniform: the bare supercore pair)
        regularizer:        typ.Any = None,                # optional backend.regularization.Regularizer
        hessian:            str = 'gauss_newton',          # or 'exact' (case-insensitive): the model's Hessian
        curvature_damping:  typ.Optional[float] = None,    # exact only: the lift damping; None -> the geometry's
        frame:              typ.Any = None,                # a frame of x_cores to linearize at; None -> geom.frame(x_cores)
        centers:            typ.Any = None,                # the centers of `frame` (only with frame=); None: none
) -> LocalModel:
    """The local model at ``x_cores``: the frame (and its centers) from ``geom.frame``, then **one** oracle query
    wanting ``f`` (and ``g`` in exact mode), with the centers' ``Vc``. The model keeps the build's ``f`` and the
    ``held`` the query returned, and every later method of the model is one query at this frame on this batch with that
    ``held`` (:py:class:`LocalModel`).

    It checks only what is structural, before any work: the ``hessian`` mode; ``hessian='exact'`` needs an oracle that
    supplies ``Kp`` (``'Kp'`` in ``oracle.optional_outputs``) and a geometry with the four exact-mode members (see
    :py:class:`Geometry`); ``curvature_damping`` with Gauss-Newton raises, where nothing would read it; a regularizer
    on a stacked point raises (:py:func:`~t3toolbox.backend.regularization.require_unstacked_for_regularizer`);
    ``centers`` without ``frame`` raises; a given frame's structure is checked by the geometry's
    ``require_frame_structure`` (integers and held masks only; a geometry without that member raises). Nothing
    numerical (O-6): **a given frame and its centers are the caller's to vouch for**. A frame of another point gives
    the model of that other point, and wrong centers give a wrong ``f`` and wrong values read off the probe ``z``, with
    no error: check them with :py:func:`~t3toolbox.backend.fv_operations.fv_frame_consistency_residual` and
    :py:func:`~t3toolbox.backend.fv_operations.fv_centers_consistency_residual` (uniform:
    :py:func:`~t3toolbox.backend.ufv_operations.ufv_frame_consistency_residual`,
    :py:func:`~t3toolbox.backend.ufv_operations.ufv_centers_consistency_residual`). A frame given without centers is
    right, read through the base point (the probe kind then holds less: each Hessian product recomputes its right
    sweep).

    Examples
    --------
    A backend-only fit's model, from the oracle and the batch (no frontend object), and its gradient check:

    >>> import numpy as np
    >>> import t3toolbox.backend.optimizers as bopt
    >>> import t3toolbox.corewise as cw
    >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> from t3toolbox.backend.geometry import ManifoldGeometryOps
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.probing import t3_probe
    >>> np.random.seed(0)
    >>> shape, ranks = (4, 5, 6), ((2, 2, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(20, N) for N in shape]
    >>> oracle = LeastSquaresOracle('probe', order=())
    >>> batch = LeastSquaresBatch(ww, t3_probe(t3_corewise_randn(shape, *ranks), ww))
    >>> geom = ManifoldGeometryOps()
    >>> x = t3_corewise_randn(shape, *ranks)
    >>> model = bopt.local_model(geom, oracle, batch, x)
    >>> print(model.held.keys(), bool(np.isclose(model.objective, bopt.point_objective(geom, oracle, batch, x))))
    ('xi', 'mu', 'nu', 'eta', 'r') True
    >>> g = model.gradient
    >>> print(bool(np.isclose(geom.inner(g, model.hvp(g)), model.gn_quadratic(g))))     # <g, H g> = ‖J g‖²
    True

    The exact model's quadratic form is the objective's second derivative along the retraction (the T3-SVD retraction
    is second order on a minimal, numerically full-rank frame, as here), where the Gauss-Newton one misses the
    curvature term:

    >>> exact = bopt.local_model(geom, oracle, batch, x, hessian='exact')
    >>> p = cw.corewise_scale(g, 1.0 / float(geom.inner(g, g)) ** 0.5)
    >>> f = lambda t: float(bopt.point_objective(geom, oracle, batch, exact.retract(cw.corewise_scale(p, t))))
    >>> fd = (f(1e-3) - 2 * f(0.0) + f(-1e-3)) / 1e-6
    >>> bool(abs(float(exact.quadratic(p)) - fd) < 1e-4 * abs(fd)), bool(abs(float(model.quadratic(p)) - fd) < 1e-2 * abs(fd))
    (True, False)
    """
    return _local_model_phases(geom, oracle, batch, x_cores, regularizer, hessian, curvature_damping, frame, centers,
                               None)


def _local_model_phases(
        geom:               typ.Any,                       # as local_model
        oracle:             typ.Any,
        batch:              typ.Any,
        x_cores:            Cores,
        regularizer:        typ.Any,
        hessian:            str,                           # 'gauss_newton' | 'exact' (case-insensitive)
        curvature_damping:  typ.Optional[float],
        frame:              typ.Any,                       # a frame of x_cores, or None -> geom.frame(x_cores)
        centers:            typ.Any,                       # its centers (with frame only), or None
        clock:              typ.Optional['_PhaseClock'],   # None: no synchronization, no clock reading
) -> LocalModel:
    """:py:func:`local_model`'s body (the build phase), its parts lapped on ``clock`` in run order: ``frame``,
    ``oracle`` (the one query), ``companion`` and, in exact mode, ``gradient`` (the projection of the build's ``g``)
    then ``curvature``. ``local_model`` calls it with ``clock=None``: the same operations in the same order, nothing
    else; ``newton_cg(timing=True)`` calls it with its clock."""
    exact = _check_hessian(hessian, 'local_model') == 'exact'
    if curvature_damping is not None and not exact:
        raise ValueError("local_model: curvature_damping is only meaningful with hessian='exact'")
    if exact:
        _require_exact_capable(oracle, geom)
    if centers is not None and frame is None:
        raise ValueError('local_model: centers= needs the frame= they belong to (geom.frame(x) returns its own)')
    if regularizer is not None:
        require_unstacked_for_regularizer(geom.stack_shape(x_cores), 'local_model')
    _mark(clock)
    if frame is None:
        frame, centers = geom.frame(x_cores)
    else:
        require = getattr(geom, 'require_frame_structure', None)
        if require is None:
            raise ValueError('local_model: frame= needs a geometry with a require_frame_structure member; '
                             '%s has none' % type(geom).__name__)
        require(x_cores, frame)
    _lap(clock, 'frame', (frame, centers))
    Vc = None if centers is None else centers[0]
    if exact:
        (f, g), held = oracle.query(frame, batch, ('f', 'g'), Vc=Vc)
    else:
        (f,), held = oracle.query(frame, batch, ('f',), Vc=Vc)
    _lap(clock, 'oracle', (f, held))
    geom_aux = geom.precompute(frame)             # None when the geometry has no companion
    _lap(clock, 'companion', geom_aux)
    curvature = None
    if exact:                                     # the build's g, projected once (LocalModel.gradient reads it)
        misfit_gradient = geom.project(frame, g, aux=geom_aux)
        _lap(clock, 'gradient', misfit_gradient)
        curvature = CurvatureState(misfit_gradient, geom.curvature_precompute(
            frame, misfit_gradient, geom_aux, curvature_damping))
        _lap(clock, 'curvature', curvature)
    return LocalModel(geom, oracle, batch, frame, centers, held, f, regularizer, geom_aux, curvature)


def point_objective(
        geom:         typ.Any,                 # a Geometry
        oracle:       typ.Any,                 # an Oracle
        batch:        typ.Any,                 # its batch
        x_cores:      Cores,                   # the raw cores (B, G), ragged or the uniform supercore pair
        regularizer:  typ.Any = None,          # optional backend.regularization.Regularizer
) -> NDArray:                                  # f (+ rho(x)), shape C
    """The objective at a point with no local model: the oracle's ``f`` at the point read as a corewise frame
    (``geom.point_data(x_cores)``: one query wanting ``f``, nothing held), plus the regularizer's ``rho(x)`` read off
    the cores (``geom.point_norm_sq`` for :py:class:`~t3toolbox.backend.regularization.IdentityRegularizer`; exact for
    any point). The trial points of the line search and the trust region (policy a) and the full-batch checks of
    ``mc_sgd`` and ``adam`` evaluate it. On a drawn batch ``f`` is the oracle's scaled one and ``rho`` is not scaled.
    A regularizer on a stacked point raises (structural).

    >>> import numpy as np
    >>> import t3toolbox.backend.optimizers as bopt
    >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> from t3toolbox.backend.geometry import CorewiseGeometryOps
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.apply import t3_apply
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(10, N) for N in (4, 5, 6)]
    >>> y = np.random.randn(10)
    >>> f = bopt.point_objective(CorewiseGeometryOps(), LeastSquaresOracle('apply', order=()),
    ...                          LeastSquaresBatch(ww, y), x)
    >>> print(bool(np.isclose(f, 0.5 * np.sum((t3_apply(x, ww) - y) ** 2))))
    True
    """
    f, rho = _point_objective_parts(geom, oracle, batch, x_cores, regularizer)
    return f if rho is None else f + rho


def _point_objective_parts(geom, oracle, batch, x_cores, regularizer=None):
    """:py:func:`point_objective`'s two terms, ``(f, rho(x))`` (``rho`` ``None`` without a regularizer): the scored
    misfit and regularization that ``newton_cg`` carries and reports (D9)."""
    if regularizer is not None:
        require_unstacked_for_regularizer(geom.stack_shape(x_cores), 'point_objective')
    (f,), _ = oracle.query(geom.point_data(x_cores), batch, ('f',))
    return f, (None if regularizer is None else regularizer.value(geom, x_cores))


def _scored(geom, oracle, batch, x_cores, regularizer):
    """The point-route objective of ``x_cores`` as host floats ``(misfit, regularization, objective)``
    (``regularization`` ``None`` without a regularizer; ``objective = misfit + regularization``): what the step
    acceptance of ``newton_cg`` compares and carries (D9)."""
    f, rho = _point_objective_parts(geom, oracle, batch, x_cores, regularizer)
    misfit, reg = float(f), (None if rho is None else float(rho))
    return misfit, reg, misfit + (reg or 0.0)


def flat_draw(
        oracle:      typ.Any,       # an oracle with n_measurements(batch) and take(batch, indices)
        batch:       typ.Any,       # the full batch
        batch_size:  int,           # samples per minibatch
) -> typ.Callable:                  # draw(rng) -> a drawn batch
    """The **default** minibatch draw: a uniform random subset of ``min(batch_size, n)`` of the ``n`` samples
    (``oracle.n_measurements(batch)``), flat across the whole sample stack ``W``, as ``oracle.take(batch, indices)``
    returns it: for the least-squares oracle a batch with ``minibatch_scale = n / b`` and the ``drawn_indices``. Returns
    ``draw(rng) -> batch``, which calls ``rng.choice(n, size=min(batch_size, n), replace=False)`` once per draw. A user
    may pass **any** ``draw`` instead (slice ``W``, importance-sample, ...), returning a batch of the oracle. Host numpy
    indices; on a jax batch the gather runs on device (the optimizer never compiles the draw, only the step)."""
    n = oracle.n_measurements(batch)
    size = min(batch_size, n)

    def draw(rng):
        return oracle.take(batch, rng.choice(n, size=size, replace=False))
    return draw


def _require_draw(
        oracle:  typ.Any,
        draw:    typ.Optional[typ.Callable],   # the user's draw, or None (the default flat_draw)
        who:     str,
) -> None:
    """Structural, at entry: without a ``draw`` the oracle needs ``take(batch, indices)`` and ``n_measurements(batch)``,
    the default draw (:py:func:`flat_draw`); else a ``ValueError``."""
    if draw is None and not (callable(getattr(oracle, 'take', None))
                             and callable(getattr(oracle, 'n_measurements', None))):
        raise ValueError('%s: %s has no take(batch, indices) and n_measurements(batch), the default draw; pass '
                         'draw=, a function draw(rng) returning a batch of the oracle' % (who, type(oracle).__name__))


# --------------------------------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------------------------------
def _require_unstacked(
        geom,             # the fit's geometry (supplies stack_shape)
        x0,               # the initial point
        who:      str,    # the optimizer name, for the message
) -> None:
    """Structural guard: **optimizing a STACKED point is not supported.**

    A stacked point carries a batch of base points ``C``, so every objective the optimizers consume is
    an array of shape ``C``, one value per element -- but their convergence tests, Armijo line searches
    and plateau detection all reduce it with a Python ``float()``. Left alone the failure is a
    ``TypeError`` (or a broadcast error) from deep inside the loop, and for ``adam`` only once its loss
    logging first fires, tens of iterations in. Raise at entry instead.

    Fit the stack elements separately (``backend.stacking.unstack``), or build the local model directly
    -- :py:func:`local_model` and the frontend ``fitting.LocalModel`` DO support a stacked point
    and return a shape-``C`` objective, so rolling your own loop over a stacked fit works. It is only
    the library optimizers' scalar reductions that do not. (A **regularizer** on a stacked point is a
    separate matter and raises on that path too --
    :py:func:`~t3toolbox.backend.regularization.require_unstacked_for_regularizer`.) A possible future
    feature.

    Structural (a shape question), so it raises in both safety modes and is jit-safe."""
    stack = tuple(geom.stack_shape(x0))
    if stack:
        raise NotImplementedError(
            "%s: optimizing a STACKED point (stack C = %r) is not supported. The objective is an array "
            "of shape C -- one value per stack element -- and the optimizers' convergence tests and line "
            "searches reduce it to a scalar. Fit the elements separately (backend.stacking.unstack), or "
            "drive your own loop from local_model / fitting.LocalModel, which do support a "
            "stacked point." % (who, stack))


def _array_leaves(
        tree:  typ.Any,          # a pytree: nested tuples / lists, None, registered records (a batch), arrays
) -> typ.List[typ.Any]:          # its numpy and jax array leaves
    """The array leaves of any pytree: through jax's own flattening when jax is importable (so a registered record,
    such as a :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch`, is walked), else through tuples and lists."""
    if jax_available:
        import jax
        leaves = jax.tree_util.tree_leaves(tree)
    else:
        leaves = []
        stack = [tree]
        while stack:
            node = stack.pop()
            if isinstance(node, (tuple, list)):
                stack.extend(node)
            elif node is not None:
                leaves.append(node)
    return [leaf for leaf in leaves if is_ndarray(leaf)]


def _prepare_jit_inputs(use_jit, x0, batch):
    """When ``use_jit``, move the optimizer inputs onto jax so the jit path actually engages -- the
    **auto-convert** contract: asking for jit is opting into "jax world" (device residency + jax's default
    float32), so we ``jnp.asarray`` ``x0`` and every array of the ``batch`` (any pytree: its sample, data and weight)
    rather than silently dropping the request. The geometry, the oracle and the regularizer (and any uniform masks
    they hold) are left untouched -- masks stay host-numpy per the uniform jit recipe. Returns ``(x0, batch)``
    unchanged when ``use_jit`` is False.

    When ``use_jit`` but jax is unavailable: runs eager on numpy with a one-time warning (the library-wide
    jax-absent policy, ``common.jax_or_warn`` -- a script developed without jax and deployed with it must
    run unchanged on both machines). Loud, not silent: the 2026.1.0 silent eager drop is what this replaces."""
    if not use_jit:
        return x0, batch
    if not jax_or_warn('use_jit=True'):
        return x0, batch
    import jax
    import jax.numpy as jnp

    def converted(tree):                     # an already-jax tree is returned AS IS: the same object (a caller that
        leaves = jax.tree_util.tree_leaves(tree)   # converted first, e.g. to hand a display this very batch, keeps it)
        if all(is_jax_ndarray(a) or not is_ndarray(a) for a in leaves):
            return tree
        return jax.tree_util.tree_map(lambda a: jnp.asarray(a) if is_ndarray(a) else a, tree)
    return converted(x0), converted(batch)


def _jit_engaged(use_jit, x0, batch) -> bool:
    """Whether the jit path actually engages: asked for, jax present, ``x0`` jax and every array of the batch jax (a
    batch with no arrays, an oracle with no data, is no obstacle), so the minibatch gather and the kernel run on device.
    Under ``use_jit`` the inputs are guaranteed jax by :py:func:`_prepare_jit_inputs`; this stays a check because a
    raw-``.data`` caller may reach these functions directly."""
    return bool(use_jit and jax_available and tree_contains_jax(x0)
                and all(is_jax_ndarray(leaf) for leaf in _array_leaves(batch)))


@ft.lru_cache(maxsize=None)
def _jitted(fn, static_argnums=()):
    """``jax.jit(fn)``, memoized on the (module-level, hence stable) function object.

    The jit wrapper must itself be a stable object: ``jax.jit`` caches on the wrapper, so building a
    fresh one per call -- which is what the previous ``_maybe_jit(step, ...)`` did with a per-call
    closure -- discarded the cache every time and recompiled once per optimizer invocation. Memoizing a
    module-level function instead compiles once per *shape signature*, process-wide.

    jax is imported here rather than at module scope: this module stays importable without jax
    (``docs/fitting_and_optimization.md`` §4.5)."""
    import jax
    return jax.jit(fn, static_argnums=static_argnums)


# The phases (TARGET_DECISIONS 8.3): build (local_model), gradient (_model_gradient), CG solve (_cg_core), trial
# (point_objective) and the stochastic steps below -- module-level functions of pytrees and statics, so a phase can
# be compiled as one program; with use_jit the CG solve and the stochastic step are, as before P16.
def _model_gradient(
        lm:  LocalModel,
) -> Tangent:                # the model's gradient, Π 𝒥ᵀ ω² r (+ g_R)
    """The gradient phase: :py:attr:`LocalModel.gradient` (one query wanting ``g`` with the model's ``held``)."""
    return lm.gradient


def _mc_sgd_step(
        geom,                # the Geometry (static)
        oracle,              # the Oracle (static)
        regularizer,         # the Regularizer or None (static)
        cores:     Tangent,  # the current point
        batch_B,             # the drawn batch (its misfit scaled by the oracle; the regularizer is not)
) -> Tangent:                # the retracted next point
    """One Manifold Cauchy SGD step on the drawn batch: a model built afresh there (nothing held across draws), its
    gradient, the tuning-free Cauchy length ``α = ‖g‖²/‖𝒥g‖²``, retract. The step objective is ``s misfit_B + rho``
    (``s`` the batch's minibatch scale), an estimate of the full objective; the Cauchy step is invariant under scaling
    the objective, so ``s`` does not change it (a regularizer weighs ``rho`` against the misfit's estimate).

    Module-level and closure-free so the jit wrapper memoized on it is stable -- see :py:func:`_jitted`."""
    lm = local_model(geom, oracle, batch_B, cores, regularizer)
    g = lm.gradient
    gg = geom.inner(g, g)
    xnp, _, _ = get_backend(False, tree_contains_jax(g))
    alpha = gg / xnp.maximum(lm.gn_quadratic(g), 1e-30)
    return lm.retract(cw.corewise_scale(g, -alpha))


def _adam_step(
        geom,                # the Geometry (static)
        oracle,              # the Oracle (static)
        regularizer,         # the Regularizer or None (static)
        cores:     Tangent,  # the current point
        m:         Tangent,  # first-moment EMA, a tree matching the cores
        v:         Tangent,  # second-moment EMA
        batch_B,             # the drawn batch
        lr_t,                # the scheduled learning rate for this step
        t,                   # 1-based step index (bias correction)
        b1,                  # beta_1
        b2,                  # beta_2
        eps,                 # denominator floor
) -> typ.Tuple[Tangent, Tangent, Tangent]:   # (next point, m, v)
    """One Adam step over the cores, elementwise (so this IS per-core Adam), on a model built afresh at the drawn batch.
    Module-level and closure-free; ``lr_t`` / ``t`` flow in as arguments so the schedule forces no recompile."""
    lm = local_model(geom, oracle, batch_B, cores, regularizer)
    g = lm.gradient
    m = cw.corewise_map(lambda mi, gi: b1 * mi + (1.0 - b1) * gi, m, g)
    v = cw.corewise_map(lambda vi, gi: b2 * vi + (1.0 - b2) * gi * gi, v, g)
    bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t            # bias corrections
    xnp, _, _ = get_backend(False, tree_contains_jax(g))
    update = cw.corewise_map(lambda mi, vi: lr_t * (mi / bc1) / (xnp.sqrt(vi / bc2) + eps), m, v)
    return lm.retract(cw.corewise_scale(update, -1.0)), m, v


# --------------------------------------------------------------------------------------------------
# The reference gradient norm ‖g0‖ of the relative stopping tests (g0_reference; the zero reference)
# --------------------------------------------------------------------------------------------------
_G0_REFERENCES = ('zero', 'warm_start', 'cone')


def _check_g0_reference(
        g0_reference:  typ.Optional[str],   # None | 'zero' | 'warm_start' | 'cone' (case-insensitive)
        who:           str,                 # the optimizer name, for the message
) -> typ.Optional[str]:                     # the lower-cased selector, or None
    """Validate the ``g0_reference`` selector (structural: an unknown string raises in both modes)."""
    if g0_reference is None:
        return None
    ref = str(g0_reference).lower()
    if ref not in _G0_REFERENCES:
        raise ValueError("%s: g0_reference must be None (the geometry's own reference), 'zero', "
                         "'warm_start' or 'cone'; got %r" % (who, g0_reference))
    return ref


def _check_positive_reference(
        name:   str,                        # 'g0norm_newton' / 'g0norm_cg'
        value:  typ.Optional[float],        # the explicit override, or None
) -> None:
    """A reference gradient norm override must be ``> 0``: the stop and the forcing term divide by it
    (``g0norm_*=0.0`` used to crash with ``ZeroDivisionError`` in the forcing term)."""
    if value is not None and not float(value) > 0.0:
        raise ValueError("%s must be > 0 (a reference gradient norm the relative tests divide by); got %r"
                         % (name, value))


def _zero_reference_gnorm(
        geom:     typ.Any,     # the fit's geometry
        oracle:   typ.Any,     # the fit's oracle
        batch:    typ.Any,     # the full batch
        frame0:   typ.Any,     # geom.zero_point_frame(frame): the canonical frame at the zero tensor
) -> float:                    # ‖Π 𝒥ᵀ ω² r(0)‖ at frame0 -- the zero reference
    """The gradient norm at the zero tensor of the frame's ranks, evaluated at the canonical zero-point frame.

    One query wanting ``g`` at ``frame0``, then the projection: the cost of one Newton gradient. The zero-point frame's
    base point is the zero tensor exactly (its last left core is zero), so the oracle's residual there is ``z(0) - y``
    (``-y`` for the least-squares oracle). A regularizer is not included: its gradient vanishes at the zero tensor, so
    the reference is the data-gradient scale alone."""
    (g,), _ = oracle.query(frame0, batch, ('g',))
    g = geom.project(frame0, g, aux=geom.precompute(frame0))
    return float(geom.inner(g, g)) ** 0.5


def _resolve_g0_reference(
        geom:          typ.Any,             # the fit's geometry
        oracle:        typ.Any,             # the fit's oracle
        batch:         typ.Any,             # the full batch
        frame:         typ.Any,             # iteration 0's local-model frame (its structure is copied)
        g0_reference:  typ.Optional[str],   # the validated selector (_check_g0_reference)
        compute:       bool,                # False when g0norm_newton is given: both references are fixed
        who:           str,                 # the optimizer name, for the messages
        hint:          str,                 # what to pass instead, for the explicit-zero ValueError
) -> typ.Tuple[
    typ.Optional[str],                      # 'zero' | 'warm_start', or None (nothing to resolve: compute=False)
    typ.Optional[float],                    # the zero reference when resolved to 'zero' and computed, else None
]:
    """Resolve which reference ``‖g0‖`` the relative tests use (``docs/fitting_and_optimization.md`` §5).

    ``None`` means the geometry's own: ``'zero'`` when it has a zero-point frame (the manifold
    geometries), else ``'warm_start'`` (the corewise geometries, and a custom geometry without the
    optional ``zero_point_frame`` member). An explicit ``'zero'`` with no zero-point frame is a structural
    error. A vanishing zero reference is never silently replaced: by default it falls back to the warm
    start with a ``RuntimeWarning``; an explicit ``'zero'`` raises."""
    zpf = getattr(geom, 'zero_point_frame', None)
    if g0_reference == 'warm_start':
        return 'warm_start', None
    if g0_reference is None and not compute:
        return None, None                   # the overrides fix both references: nothing to resolve
    explicit = g0_reference == 'zero'
    if zpf is None:
        if explicit:
            raise ValueError("%s: g0_reference='zero' needs a geometry with a zero_point_frame member; %s "
                             "has none (pass g0_reference='warm_start', or give the geometry a "
                             "zero_point_frame)" % (who, type(geom).__name__))
        return 'warm_start', None
    frame0 = zpf(frame)
    if frame0 is None:
        if explicit:
            raise ValueError("%s: g0_reference='zero' has no meaning on %s, which defines no zero-point frame "
                             "(on a corewise geometry the zero cores are a critical point, so the reference "
                             "would be exactly 0); pass g0_reference='warm_start' or None"
                             % (who, type(geom).__name__))
        return 'warm_start', None
    if not compute:
        return 'zero', None
    zero_ref = _zero_reference_gnorm(geom, oracle, batch, frame0)
    if zero_ref > 0.0:
        return 'zero', zero_ref
    if explicit:
        raise ValueError("%s: the zero-tensor reference is 0 at these ranks; pass %s" % (who, hint))
    warnings.warn("%s: the zero-tensor reference vanishes at these ranks (the data gradient at the zero "
                  "tensor is 0, e.g. all-zero data); using the warm-start reference" % (who,),
                  RuntimeWarning, stacklevel=3)
    return 'warm_start', None


# --------------------------------------------------------------------------------------------------
# The padded start: an aligned frame at a zero-padded rank increase (start_frame / padded_from) and the
# 'cone' reference (the gradient norm at the zero tensor, at the frame aligned with the data gradient there).
# The layer-specific work is the geometry's (its aligned_padded_frame member); this module imports nothing new.
# --------------------------------------------------------------------------------------------------
_ALIGNMENT_OPTIONS = ('oversampling', 'oversampled_passes', 'max_passes', 'tol', 'alternate', 'seed')


@ft.lru_cache(maxsize=None)     # unbounded: one entry per (geometry, oracle), i.e. one per rank level -- negligible
def _alignment_projector(
        geom:    typ.Any,     # the fit's geometry (value-hashed)
        oracle:  typ.Any,     # the fit's oracle (value-hashed)
) -> typ.Callable:            # project(frame, batch) -> Π 𝒥ᵀ ω² r at frame (ragged variations / uniform bare pair)
    """The alignment loops' projector of the misfit gradient: at each pass's ``frame``, one query wanting ``g`` on the
    batch (the loop's ``hint``), projected by the geometry **at that frame's structure** (``geom.with_frame(frame)``:
    the loop's frames change width, and on the uniform layout the projection's masks are the frame's own). Nothing is
    held across passes: each pass has its own frame. Every frame of the loop has the loop's base point (``x0``, or the
    zero tensor at the cone), so the residual each pass reads is the one the alignment is for.

    A factory memoized on its two value-hashed statics (``scan_body_principles.md``, rule 4): the projector is a stable
    function object per ``(geom, oracle)``, a static argument of the uniform whole-loop jit, and the batch rides as
    the loop's pytree hint."""
    def project(frame, batch):
        frame_geom = geom.with_frame(frame)
        (g,), _ = oracle.query(frame, batch, ('g',))
        return frame_geom.project(frame, g, aux=frame_geom.precompute(frame))
    return project


def _alignment_options(
        options:  typ.Dict[str, typ.Any],   # the six alignment options by name (None = unset)
        prefix:   str,                      # '' (padded_start), 'padded_' or 'cone_' (the optimizers' keywords)
        who:      str,
) -> typ.Dict[str, typ.Any]:                # the set options only
    """The set options; zero passes with an option they would not read is a structural error."""
    given = {k: v for k, v in options.items() if v is not None}
    if given.get('max_passes') == 0:
        unread = [prefix + k for k in ('oversampling', 'oversampled_passes', 'tol', 'alternate') if k in given]
        if unread:
            raise ValueError('%s: %smax_passes=0 (the hint-free padded start) does not read %s; remove it, or set a '
                             'positive pass count' % (who, prefix, ', '.join(unread)))
    return given


def padded_start(
        geom:                typ.Any,                   # the fit's Geometry (a manifold one: it needs aligned_padded_frame)
        oracle:              typ.Any,                   # the fit's Oracle
        batch:               typ.Any,                   # its batch (uniform: packed)
        x0:                  Cores,                     # the zero-padded point at the target ranks (minimal); uniform: bare pair
        x_old:               typ.Optional[typ.Any],     # the old point at minimal ranks (ragged cores / uniform full .data, rank 0 allowed); None = the zero tensor (the cone)

        oversampling:        typ.Optional[int]   = None,   # None = the member's default (ALIGN_OVERSAMPLING = 5); forwarded only when set
        oversampled_passes:  typ.Optional[int]   = None,   # None = 2
        max_passes:          typ.Optional[int]   = None,   # None = 10
        tol:                 typ.Optional[float] = None,   # None = 0.05
        alternate:           typ.Optional[bool]  = None,   # None = True
        seed:                typ.Optional[int]   = None,   # None = ZERO_FRAME_SEED
        use_jit:             bool = False,                 # forwarded to the loop; newton_cg passes its own use_jit
) -> typ.Tuple[
    typ.Any,   # the aligned frame of x0 (ragged frame / uniform frame .data at the level's widths)
    typ.Any,   # its centers (Vc, Hc): the old point's zero-padded (zeros from the zero tensor)
    typ.Any,   # movements, as the member returns them (ragged tuple; uniform NaN-padded array)
]:
    """The frame of ``x0`` aligned with the misfit gradient, and its centers: the start of a Newton run after a rank
    increase (the aligned padded frame with the misfit gradient as its hint; user guide :doc:`/aligned_padded_frame`).

    ``x0`` is the zero-padded point at the target ranks and ``x_old`` the old point (``None``: the zero tensor, the
    ``'cone'`` reference's frame). The alignment runs on the geometry's ``aligned_padded_frame`` member; each pass
    projects the oracle's ``g`` on ``batch`` at its own frame (every frame of the loop has ``x0`` as its base point, the
    zero tensor at the cone). Unset options take the member's defaults; ``max_passes=0`` returns the hint-free padded
    start, and an option it would not read then raises. The centers are the old point's zero-padded, so
    ``local_model(..., frame=frame, centers=centers)`` reads the probe ``z`` through them as at ``geom.frame(x)``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.frame_variations_format as bvf
    >>> import t3toolbox.backend.optimizers as bopt
    >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> from t3toolbox.backend.geometry import ManifoldGeometryOps
    >>> np.random.seed(0)
    >>> shape = (5, 6, 5)
    >>> ww = [np.random.randn(60, n) for n in shape]
    >>> batch = LeastSquaresBatch(ww, t3.TuckerTensorTrain.randn(shape, (3, 4, 3), (1, 3, 3, 1)).probe(ww))
    >>> geom, oracle = ManifoldGeometryOps(), LeastSquaresOracle('probe', order=())
    >>> x_old = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
    >>> x0 = x_old.resize(shape, (3, 4, 3), (1, 3, 3, 1))                 # zero-padded to the target ranks
    >>> frame, centers, movements = bopt.padded_start(geom, oracle, batch, x0.data, x_old.data)
    >>> bvf.T3Frame(*frame).down_ranks, len(movements) <= 10               # rewrap
    ((3, 4, 3), True)

    Newton can start from it (``newton_cg(start_frame=...)``; ``padded_from=x_old.data`` does both steps):

    >>> lm = bopt.local_model(geom, oracle, batch, x0.data, frame=frame, centers=centers)
    >>> print(bool(np.isclose(lm.objective, bopt.point_objective(geom, oracle, batch, x0.data))))
    True
    """
    _alignment_member(geom)                             # a geometry without the member raises before the options
    options = _alignment_options(dict(oversampling=oversampling, oversampled_passes=oversampled_passes,
                                      max_passes=max_passes, tol=tol, alternate=alternate, seed=seed),
                                 '', 'padded_start')
    return _padded_start(geom, oracle, batch, x0, x_old, options, use_jit, False)[:3]


def _alignment_member(
        geom:  typ.Any,      # the fit's geometry
) -> typ.Callable:           # its aligned_padded_frame member
    """The geometry's ``aligned_padded_frame`` member; a geometry without one raises (a padded start needs it)."""
    member = getattr(geom, 'aligned_padded_frame', None)
    if member is None:
        raise ValueError('padded_start: the geometry %s has no aligned_padded_frame member (a padded start needs a '
                         'manifold geometry)' % type(geom).__name__)
    return member


def _padded_start(
        geom:     typ.Any,
        oracle:   typ.Any,
        batch:    typ.Any,
        x0:       Cores,                     # as padded_start
        x_old:    typ.Optional[typ.Any],     # as padded_start
        options:  typ.Dict[str, typ.Any],    # the set alignment options (unprefixed, validated)
        use_jit:  bool,
        timing:   bool,                      # True: ask the member for its seconds per pass (_aligned_padded_frame)
) -> typ.Tuple[
    typ.Any,                                 # the aligned frame of x0
    typ.Any,                                 # its centers
    typ.Any,                                 # movements, as the member returns them
    typ.Optional[typ.Tuple[float, ...]],     # the seconds per pass; None unless timing and the member reports them
]:
    """:py:func:`padded_start` after its option validation, with ``timing`` (``newton_cg(padded_from=)``'s call)."""
    return _aligned_padded_frame(_alignment_member(geom), x0, x_old, _alignment_projector(geom, oracle), batch,
                                 options, use_jit, timing)


def _aligned_padded_frame(
        member:   typ.Callable,              # a geometry's aligned_padded_frame
        x0:       Cores,
        x_old:    typ.Optional[typ.Any],
        project:  typ.Callable,              # _alignment_projector(geom, oracle)
        hint:     typ.Any,                   # the batch: the projector's second argument
        options:  typ.Dict[str, typ.Any],    # the set alignment options (unprefixed)
        use_jit:  bool,
        timing:   bool,
) -> typ.Tuple[
    typ.Any,                                 # the aligned frame of x0
    typ.Any,                                 # its centers (None if the member returns none)
    typ.Any,                                 # movements, as the member returns them
    typ.Optional[typ.Tuple[float, ...]],     # the member's seconds per pass, or None
]:
    """The optimizers' call of the member. With ``timing``, ``timing=True`` is passed only when the member's
    signature names it: a ``**kwargs`` catch-all is no evidence (it may forward to a loop without the keyword), and a
    diagnostic must not break a working run. Otherwise the call is the untimed one and the seconds are ``None``."""
    if timing:
        try:
            parameter = inspect.signature(member).parameters.get('timing')
        except (TypeError, ValueError):      # no signature to read
            parameter = None
        if parameter is not None and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            return member(x0, x_old, project, hint, use_jit=use_jit, timing=True, **options)
    frame, centers, movements = member(x0, x_old, project, hint, use_jit=use_jit, **options)
    return frame, centers, movements, None


def _cone_options(
        options:  typ.Dict[str, typ.Any],   # the six cone_* options by name (None = unset)
        engaged:  bool,                     # the cone reference is computed (g0_reference='cone', no g0norm_newton)
        who:      str,
) -> typ.Dict[str, typ.Any]:                # the set options only
    """The cone_* options, checked: set only when the cone is computed, and read by the pass count."""
    given = sorted(k for k, v in options.items() if v is not None)
    if given and not engaged:
        raise ValueError("%s: %s set, but the cone reference is not computed (it needs g0_reference='cone' and no "
                         "g0norm_newton)" % (who, ', '.join('cone_' + k for k in given)))
    return _alignment_options(options, 'cone_', who)


def _cone_reference(
        geom:     typ.Any,
        oracle:   typ.Any,
        batch:    typ.Any,                   # the full batch
        x0:       Cores,                     # the start point (its structure: the zero tensor at its ranks)
        options:  typ.Dict[str, typ.Any],    # the set cone_* options (unprefixed)
        use_jit:  bool,
        who:      str,
        hint:     str,                       # what to pass instead, for the vanishing-reference ValueError
        timing:   bool,                      # True: ask the member for its seconds per pass (_aligned_padded_frame)
) -> typ.Tuple[
    float,                                   # ‖Π 𝒥ᵀ ω² r(0)‖ at the aligned frame of the zero tensor
    typ.Tuple[float, ...],                   # the alignment's movements, one per pass
    typ.Optional[typ.Tuple[float, ...]],     # the seconds per pass; None unless timing and the member reports them
]:
    """The 'cone' reference: the misfit gradient norm at the zero tensor, at the frame aligned with it."""
    member = getattr(geom, 'aligned_padded_frame', None)
    if member is None:
        raise ValueError("%s: g0_reference='cone' needs a geometry with an aligned_padded_frame member; %s has none "
                         "(pass %s)" % (who, type(geom).__name__, hint))
    project = _alignment_projector(geom, oracle)
    frame, _centers, movements, pass_seconds = _aligned_padded_frame(member, x0, None, project, batch, options,
                                                                     use_jit, timing)
    g = project(frame, batch)                       # the frames of the cone's loop are at the zero tensor
    value = float(geom.with_frame(frame).inner(g, g)) ** 0.5
    if not value > 0.0:
        raise ValueError("%s: the cone reference is 0 at these ranks; pass %s" % (who, hint))
    return value, _movement_record(movements), pass_seconds


def _movement_record(
        movements:  typ.Any,   # the member's movements: a tuple of 0-d arrays (ragged) or a NaN-padded array (uniform)
) -> typ.Tuple[float, ...]:    # one float per pass (the uniform record's NaN tail dropped)
    """The alignment's movements as plain floats."""
    values = [float(m) for m in movements]
    if not isinstance(movements, tuple):
        while values and math.isnan(values[-1]):
            values.pop()
    return tuple(values)


def gradient_descent(
        geom:     typ.Any,    # the Geometry
        oracle:   typ.Any,    # the Oracle
        batch:    typ.Any,    # its batch
        x0:       Tangent,    # initial cores (B, G); uniform: the bare supercore pair
        regularizer: typ.Any = None,   # optional backend.regularization.Regularizer
        n_iter:   int   = 100,
        gtol_rel: float = 1e-8,   # stop when ‖g‖ <= gtol_rel * ‖g_0‖
        c_armijo: float = 1e-4,   # Armijo sufficient-decrease constant
        on_line_search_failure: str = 'stop',  # Armijo exhausted (50 halvings): 'stop' = reject + terminate; 'accept' = take the last trial and continue
        g0_reference: typ.Optional[str] = None,  # ‖g_0‖ for the stop: None = the geometry's own ('zero' on manifolds); 'zero' | 'warm_start' | 'cone'
        *,
        cone_oversampling:        typ.Optional[int]   = None,   # None = 5    (g0_reference='cone' only, else a structural error)
        cone_oversampled_passes:  typ.Optional[int]   = None,   # None = 2
        cone_max_passes:          typ.Optional[int]   = None,   # None = 10
        cone_tol:                 typ.Optional[float] = None,   # None = 0.05
        cone_alternate:           typ.Optional[bool]  = None,   # None = True
        cone_seed:                typ.Optional[int]   = None,   # None = ZERO_FRAME_SEED
) -> typ.Tuple[Tangent, dict]:    # (x_cores, stats)
    """Steepest descent with a Cauchy initial step and an **Armijo backtracking line search**. The step
    length starts at the Cauchy value ``α = ‖g‖² / ‖𝒥g‖²`` (the 1D minimizer of the local GN quadratic
    along ``−g``) and backtracks (``α ← α/2``) until ``f(retract(−α g)) ≤ f − c·α‖g‖²`` -- so it descends
    on any geometry, including the additive corewise chart where a bare Cauchy step overshoots the
    high-degree objective. Exercises the whole backend-first stack (``gradient`` / ``gn_quadratic`` /
    ``objective`` / ``retract``; the trials through :py:func:`point_objective`). **Eager only, deliberately** -- the
    reference implementation of the backend-first stack; ``use_jit`` lives on ``mc_sgd`` / ``adam`` / ``newton_cg``.

    ``on_line_search_failure``: as in :py:func:`newton_cg` -- exhausting all 50 halvings means the
    objective's floor; ``'stop'`` (default) rejects the step and terminates, ``'accept'`` takes the
    last trial and keeps going (the deliberate escape mode). ``stats['line_search_failed']`` reports
    whether an exhaustion occurred.

    ``g0_reference``: the reference ``‖g_0‖`` of the relative stop, resolved exactly as in
    :py:func:`newton_cg` -- by default the gradient norm at the zero tensor of ``x0``'s ranks on a manifold
    geometry, the warm-start ``‖g(x0)‖`` on a corewise one; ``'cone'`` and its ``cone_*`` options as in
    :py:func:`newton_cg` (the alignment runs eagerly here)."""
    _require_unstacked(geom, x0, 'gradient_descent')
    olsf = str(on_line_search_failure).lower()
    if olsf not in ('stop', 'accept'):
        raise ValueError("on_line_search_failure must be 'stop' or 'accept'; got %r" % (on_line_search_failure,))
    ref_selector = _check_g0_reference(g0_reference, 'gradient_descent')
    cone_options = _cone_options(dict(oversampling=cone_oversampling, oversampled_passes=cone_oversampled_passes,
                                      max_passes=cone_max_passes, tol=cone_tol, alternate=cone_alternate,
                                      seed=cone_seed), ref_selector == 'cone', 'gradient_descent')
    extra = {}
    x = x0
    losses = []
    any_ls_failed = False
    g0norm = None
    f = None            # D9: the point-route objective of exactly the current x (the start's, then the accepted trial's)
    for it in range(n_iter):
        lm = local_model(geom, oracle, batch, x, regularizer)
        if f is None:
            f = float(point_objective(geom, oracle, batch, x, regularizer))
        losses.append(f)
        g = _model_gradient(lm)
        gg = float(geom.inner(g, g))                        # ‖g‖²
        if g0norm is None and ref_selector == 'cone':       # iteration 0: the cone reference (once)
            g0norm, extra['cone_movements'], _ = _cone_reference(geom, oracle, batch, x0, cone_options, False,
                                                                 'gradient_descent', "g0_reference='warm_start'",
                                                                 False)
        if g0norm is None:                                  # iteration 0: resolve the reference (once)
            ref, zero_ref = _resolve_g0_reference(geom, oracle, batch, lm.frame, ref_selector, True,
                                                  'gradient_descent', "g0_reference='warm_start'")
            g0norm = zero_ref if ref == 'zero' else (gg ** 0.5 if gg > 0 else 1.0)
        if gg ** 0.5 <= gtol_rel * g0norm:                  # converged
            break
        alpha = gg / max(float(lm.gn_quadratic(g)), 1e-30 * gg)   # Cauchy initial step
        x_trial = x
        ls_failed = True
        for _bt in range(50):                               # Armijo backtracking
            x_trial = lm.retract(cw.corewise_scale(g, -alpha))
            f_trial = float(point_objective(geom, oracle, batch, x_trial, regularizer))
            if f_trial <= f - c_armijo * alpha * gg:
                ls_failed = False
                break
            if _bt < 49:
                alpha *= 0.5        # halve BETWEEN trials only
        if ls_failed:
            any_ls_failed = True
            if olsf == 'stop':
                break                                        # reject the step: x unchanged
        x, f = x_trial, f_trial                              # 'accept' after exhaustion: the last trial's value
    return x, {'losses': losses, 'n_iter': len(losses), 'line_search_failed': any_ls_failed, **extra}


def mc_sgd(
        geom:        typ.Any,    # the Geometry
        oracle:      typ.Any,    # the Oracle
        batch:       typ.Any,    # its full batch
        x0:          Tangent,    # initial cores (B, G); uniform: the bare supercore pair
        rng,                     # np.random.Generator -- passed to the draw each step
        batch_size:  int,        # samples per minibatch (the default flat draw's; unread with a draw)
        regularizer: typ.Any = None,   # optional backend.regularization.Regularizer (unscaled on a draw)
        draw:        typ.Optional[typ.Callable] = None,  # custom draw(rng) -> a batch; None = flat_draw(oracle, batch, batch_size)
        max_iter:    int   = 3000,
        check_every: int   = 25,     # iterations between full-batch loss checks (the absolute-iteration window)
        smooth_tau:  float = 2.0,    # EMA timescale of the loss check, in checks
        plateau_lag: int   = 4,      # stop when the smoothed loss rises over this many checks
        use_jit:     bool  = False,  # jit the per-step kernel (gradient + Cauchy step + retract) when jax
) -> typ.Tuple[Tangent, dict]:       # (x_cores, stats)
    """Manifold Cauchy SGD (T4S §5.3.2): minibatch + the tuning-free Cauchy step
    ``α = ‖g‖² / ‖𝒥g‖²`` -- no learning rate. Intended for the **manifold** geometry (its bounded
    retraction makes the raw Cauchy step stable; on the additive corewise chart use ``adam``). Each step draws a batch
    (``draw(rng)``; by default :py:func:`flat_draw`, which needs an oracle with ``take``, else a ``ValueError`` at
    entry) and builds a model there afresh: the step objective is the drawn batch's misfit, scaled by the oracle to
    estimate the full one, plus the **unscaled** regularizer. Stops on an exponentially-smoothed **full-batch** loss
    (:py:func:`point_objective`) with an **absolute-iteration window** (``plateau_lag * check_every`` iterations) --
    decoupled from batch size (the epoch-based window made small batches fragile; findings in a separate research repo,
    maintainer-local). ``stats['losses']`` records the **smoothed** check values (the stop signal), not the raw
    full-batch losses. ``use_jit`` jits the per-step kernel (the host loop draws minibatches; the full-batch stop check
    stays on the host), auto-converting numpy inputs to jax first (:py:func:`_prepare_jit_inputs`) -- so the
    **default** draw then slices the on-device batch (a custom ``draw`` should return jax batches to stay on device)."""
    _require_unstacked(geom, x0, 'mc_sgd')
    _require_draw(oracle, draw, 'mc_sgd')
    x0, batch = _prepare_jit_inputs(use_jit, x0, batch)
    draw = draw if draw is not None else flat_draw(oracle, batch, batch_size)
    a_smooth = 1.0 - math.exp(-1.0 / smooth_tau)

    step = _jitted(_mc_sgd_step, (0, 1, 2)) if _jit_engaged(use_jit, x0, batch) else _mc_sgd_step

    x = x0
    s_hist = []
    n_iter = 0
    for k in range(max_iter):
        n_iter = k + 1
        x = step(geom, oracle, regularizer, x, draw(rng))
        if n_iter % check_every == 0:                       # full-batch loss check (the stop signal, host)
            L = float(point_objective(geom, oracle, batch, x, regularizer))
            s = L if not s_hist else a_smooth * L + (1.0 - a_smooth) * s_hist[-1]
            s_hist.append(s)
            if len(s_hist) > plateau_lag and (s_hist[-1] - s_hist[-1 - plateau_lag]) > 0.0:
                break
    return x, {'losses': s_hist, 'n_iter': n_iter}


def adam(
        geom:     typ.Any,    # the Geometry
        oracle:   typ.Any,    # the Oracle
        batch:    typ.Any,    # its full batch
        x0:       Tangent,    # initial cores (B, G); uniform: the bare supercore pair
        rng,                  # np.random.Generator -- passed to the draw each step
        batch_size:  int,     # samples per minibatch (the default flat draw's; unread with a draw)
        regularizer: typ.Any = None,   # optional backend.regularization.Regularizer (unscaled on a draw)
        draw:     typ.Optional[typ.Callable] = None,  # custom draw(rng) -> a batch; None = flat_draw(oracle, batch, batch_size)
        lr:       float = 1e-2,
        max_iter: int   = 2000,
        betas:    typ.Tuple[float, float] = (0.9, 0.999),
        eps:      float = 1e-8,
        cosine:   bool  = True,   # cosine-decay the learning rate over max_iter (helps it settle)
        use_jit:  bool  = False,  # jit the per-step kernel (gradient + Adam update + retract) when jax
) -> typ.Tuple[Tangent, dict]:    # (x_cores, stats)
    """Adam over the cores -- the dependency-free first-order method for the **corewise** geometry. The
    moments ``m`` / ``v`` are trees matching the cores (a ``corewise_map`` per step); elementwise, so this
    is exactly per-core Adam. (``lr`` is a real hyperparameter -- Adam is not tuning-free, unlike the Cauchy
    step.) On corewise the step is additive (``lm.retract`` = ``cores += step``). Each step builds a model afresh at a
    drawn batch, as :py:func:`mc_sgd` does (the regularizer unscaled; an oracle without ``take`` needs a ``draw``).
    ``use_jit`` jits the per-step kernel (``lr_t``/``t`` flow in as traced args, so the schedule does not force a
    recompile), auto-converting numpy inputs to jax first (:py:func:`_prepare_jit_inputs`). ``stats['losses']`` is
    the full-batch objective (:py:func:`point_objective`) every 50 steps."""
    _require_unstacked(geom, x0, 'adam')
    _require_draw(oracle, draw, 'adam')
    x0, batch = _prepare_jit_inputs(use_jit, x0, batch)
    b1, b2 = betas

    step = _jitted(_adam_step, (0, 1, 2)) if _jit_engaged(use_jit, x0, batch) else _adam_step

    draw = draw if draw is not None else flat_draw(oracle, batch, batch_size)
    cores = x0
    m = cw.corewise_zeros_like(cores)
    v = cw.corewise_zeros_like(cores)
    losses = []
    for k in range(max_iter):
        batch_B = draw(rng)
        lr_t = lr * (0.5 * (1.0 + math.cos(math.pi * k / max_iter)) if cosine else 1.0)
        cores, m, v = step(geom, oracle, regularizer, cores, m, v, batch_B, lr_t, k + 1, b1, b2, eps)
        if (k + 1) % 50 == 0:
            losses.append(float(point_objective(geom, oracle, batch, cores, regularizer)))
    return cores, {'losses': losses, 'n_iter': max_iter}


# --------------------------------------------------------------------------------------------------
# Conjugate gradients. The loop bodies are MODULE-LEVEL and closure-free, and every value they need --
# including the tolerance and the iteration cap -- rides in the loop state.
#
# This is a correctness requirement, not a tidiness one. `lax.while_loop` caches its traced body on the
# body's IDENTITY, so a stable body that READS a Python value which changed since the last call gets the
# cached jaxpr with the OLD value, silently. `tol` is recomputed every Newton iteration from the forcing
# term, so a body closing over it would solve to a stale tolerance
# (`docs/contributor/scan_body_principles.md`, principle 5). Carrying it makes that unrepresentable.
#
# The local model rides in the state too -- loop-invariant, passed through untouched. XLA treats a
# loop-carried invariant like a constant, so this costs nothing (measured in the tier-2 chunking work).
# --------------------------------------------------------------------------------------------------
def _cg_cond(
        state:  typ.Tuple,   # (p, r, d, rs, i, ok, local_model, tol2, maxiter, radius, boundary)
) -> NDArray:                # 0-d bool: keep iterating
    """Continue while the residual is above tolerance, the cap is unreached, curvature stayed positive, and
    the iterate has not stopped on the trust-region boundary."""
    _p, _r, _d, rs, i, ok, _lm, tol2, maxiter, _radius, boundary = state
    return (rs > tol2) & (i < maxiter) & ok & ~boundary


def _cg_body(
        state:  typ.Tuple,   # (p, r, d, rs, i, ok, local_model, tol2, maxiter, radius, boundary)
) -> typ.Tuple:              # the same, advanced one CG iteration
    """One Steihaug-Toint CG iteration (Steihaug 1983; Nocedal & Wright, *Numerical Optimization*, Alg. 7.2).
    **Backend-agnostic and branch-free** -- every branch is an ``xnp.where``, so the same body drives the eager
    Python loop and ``lax.while_loop``.

    * Inside the trust region it is a plain CG step.
    * If the step would leave the region (``‖p + αd‖ > Δ``), or ``dᵀHd ≤ 0`` with a **finite** radius, it
      steps to the boundary instead -- ``p + τd`` with ``τ ≥ 0`` and ``‖p + τd‖ = Δ`` -- and sets
      ``boundary``, which stops the loop.
    * ``dᵀHd ≤ 0`` with ``radius = inf`` (no trust region) takes a zero step and clears ``ok``, which stops
      the loop: truncated CG, exactly the pre-trust-region behaviour (a gauge direction of the singular
      corewise ``H``). With ``radius = inf`` the selected arithmetic is the plain CG body's, bit for bit.

    No ``inf`` enters any arithmetic: the boundary test compares a finite norm against the radius, and the
    boundary length ``τ`` is computed from a radius that is ``where``-replaced by 0 wherever it is unused."""
    p, r, d, rs, i, ok, lm, tol2, maxiter, radius, boundary = state
    xnp, _, _ = get_backend(False, tree_contains_jax(r))
    inner = lm.geom.inner
    Hd = lm.hvp(d)
    dHd = inner(d, Hd)
    pos = dHd > 0.0
    alpha = xnp.where(pos, rs / xnp.where(pos, dHd, 1.0), 0.0)        # 0 step on nonpositive curvature
    pp, pd, dd = inner(p, p), inner(p, d), inner(d, d)
    outside = xnp.sqrt(xnp.maximum(pp + 2.0 * alpha * pd + alpha * alpha * dd, 0.0)) > radius  # ‖p+αd‖ > Δ
    to_boundary = (pos & outside) | (~pos & xnp.isfinite(radius))
    rad = xnp.where(to_boundary, radius, 0.0)                          # finite wherever it is read
    dd_safe = xnp.where(dd > 0.0, dd, 1.0)
    tau = (-pd + xnp.sqrt(xnp.maximum(pd * pd + dd * (rad * rad - pp), 0.0))) / dd_safe   # ‖p + τd‖ = Δ, τ ≥ 0
    step = xnp.where(to_boundary, tau, alpha)
    p = cw.corewise_add(p, cw.corewise_scale(d, step))
    r = cw.corewise_sub(r, cw.corewise_scale(Hd, step))
    rs_new = inner(r, r)
    beta = xnp.where(pos, rs_new / rs, 0.0)
    d = cw.corewise_add(r, cw.corewise_scale(d, beta))
    return (p, r, d, rs_new, i + 1, pos, lm, tol2, maxiter, radius, boundary | to_boundary)


def _cg_core(
        local_model,            # the LocalModel supplying H (`hvp`) and the coordinate inner product
        rhs:      Tangent,      # the right-hand side (= −g)
        tol:      typ.Any,      # stop at ‖H p − rhs‖ ≤ tol
        maxiter:  typ.Any,      # iteration cap
        use_jit:  bool,         # STATIC: selects lax.while_loop vs the eager Python loop
        radius:   typ.Any,      # the trust-region radius Δ (‖p‖ ≤ Δ); inf = none (plain truncated CG)
) -> typ.Tuple[
    Tangent,                    # p: the (approximate) solution, ‖p‖ ≤ radius
    typ.Any,                    # i: CG iterations run
    typ.Any,                    # rs: the residual² ‖H p − rhs‖² at p
    typ.Any,                    # ok: the last curvature dᵀHd was positive (False: stopped on nonpositive curvature)
    typ.Any,                    # boundary: stopped on the trust-region boundary ‖p‖ = Δ
]:
    """Solve ``H p = rhs`` by Steihaug-Toint conjugate gradients to residual ``‖r‖ ≤ tol`` inside the trust
    region ``‖p‖ ≤ radius`` (``H`` symmetric; indefinite allowed with a finite radius).

    Returns the solution plus the loop state the caller needs for diagnostics. The outcome is derived from
    ``(rs, i, ok, boundary)``: on the boundary (``boundary``), converged (``rs ≤ tol²``), truncated on
    nonpositive curvature (``not ok``, only with ``radius = inf``), or the iteration cap. With
    ``radius = inf`` it is the plain truncated CG, bit for bit on eager numpy."""
    xnp, _, _ = get_backend(False, tree_contains_jax(rhs))
    rs0 = local_model.geom.inner(rhs, rhs)
    tol2 = tol * tol
    radius = xnp.asarray(radius, dtype=xnp.result_type(rs0))           # the iterate's dtype (a float32 solve stays float32)
    state0 = (cw.corewise_zeros_like(rhs), rhs, rhs, rs0, 0, rs0 > tol2, local_model, tol2, maxiter,
              radius, xnp.asarray(rs0 < 0.0))                          # boundary0: False, as a 0-d bool array
    p, _r, _d, rs, i, ok, _lm, _tol2, _maxiter, _radius, boundary = xwhile(_cg_cond, _cg_body, state0, use_jit)
    return p, i, rs, ok, boundary


def _cg_solve(local_model, rhs, tol, maxiter, use_jit, radius=math.inf):
    """Dispatch CG to the compiled or the eager path -- the SAME core either way.

    Under jit the whole solve is one compiled function of ``(local_model, rhs, tol, maxiter, radius)``, so
    the cache key is jax's own: the model's pytree structure (its geometry and kind ride as
    **value-hashed** aux) plus the argument shapes. ``tol``, ``maxiter`` and ``radius`` are traced
    arguments, so a Newton loop that tightens the forcing term (or moves the trust region) every iteration
    reuses one compiled program instead of recompiling -- and cannot go stale. Returns
    ``(p, i, rs, ok, boundary)`` (:py:func:`_cg_core`)."""
    if use_jit and jax_available and tree_contains_jax(rhs):
        return _jitted(_cg_core, (4,))(local_model, rhs, tol, maxiter, True, radius)
    return _cg_core(local_model, rhs, tol, maxiter, False, radius)


# --------------------------------------------------------------------------------------------------
# Phase timing, newton_cg(timing=True): the records, the compile listener and the clock.
#
# Each history row's wall_time window is split into laps, each from a mark (or the previous lap's end) to a
# synchronized end (common.tree_block_until_ready on the lap's outputs: jax dispatch is asynchronous); the rest of the
# window is its remainder. The clock reads nothing and synchronizes nothing when timing is off: the loop passes
# clock=None, and _mark / _lap return at once.
# --------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class PhaseSeconds:
    """One ``newton_cg`` history row's window (:py:attr:`NewtonTiming.window`) split by phase, in seconds -- or, as
    :py:attr:`NewtonTiming.compile_seconds`, jax's compile wait inside each phase. ``None``: the phase did not run on
    this row. The fields are in run order; with ``remainder`` they tile the window.

    * ``previous_callback`` -- the previous row's records, its ``callback`` and its history append (so a heavy
      display appears on the next row); ``None`` on row 0.
    * ``frame``, ``oracle``, ``companion``, ``curvature`` -- the local model's build (:py:func:`local_model`):
      ``geom.frame`` (or the start frame's structure check), the build's one oracle query (wanting ``f``, and ``g`` in
      exact mode), ``geom.precompute``, and in exact mode ``geom.curvature_precompute`` (with ``hessian='switch'``
      every build is exact-mode, the Gauss-Newton phase's included). ``None`` on a row that reuses the model (after a
      rejected trust-region trial); ``curvature`` also in Gauss-Newton mode.
    * ``gradient`` -- the misfit gradient's query and projection ``Π 𝒥ᵀ ω² r`` (in exact mode, and with
      ``hessian='switch'``, the projection of the build's ``g``, inside the build and counted here), the misfit and
      regularization values, the gradient, ``‖g‖`` and the lift diagnostics; ``None`` as ``frame``.
    * ``reference`` -- resolving the reference ``‖g0‖`` (``g0_reference``; the ``'cone'`` loop included): row 0 only.
    * ``cg`` -- the CG solve (under ``use_jit``, one compiled call); ``forms`` -- the first trust-region radius, the
      slope, the descent fallback, ``pᵀHp`` and ``‖p‖``; ``trial_retraction`` / ``trial_objective`` -- every trial's
      retraction / objective. All four ``None`` on the converged row.
    * ``tracking`` -- ``hessian='switch'``: the tracking product ``⟨p, Lᵀ K_N L p⟩`` of the point's exact build, on a
      tracked Gauss-Newton-phase row (:py:class:`NewtonInfo`'s ``switch_pKp``); ``None`` on every other row.
    * ``remainder`` -- the window minus the phases above (the forcing term, ``‖x‖``, the trust region's floor and the
      switch's rounding gate, which query the oracle's ``f_resolution`` on a rejection or a tracked row, Python
      bookkeeping); never ``None``, ``>= 0`` up to rounding."""
    previous_callback:  typ.Optional[float]      # the previous row's record, callback and history append; None on row 0
    frame:              typ.Optional[float]      # frame .. gradient: None on a row that reuses the model
    oracle:             typ.Optional[float]      # the build's one query (O-17: in place of sweep and residual)
    companion:          typ.Optional[float]
    curvature:          typ.Optional[float]      # also None in Gauss-Newton mode
    gradient:           typ.Optional[float]      # the misfit gradient (wherever it runs) .. the lift diagnostics
    reference:          typ.Optional[float]      # row 0 only
    cg:                 typ.Optional[float]      # cg .. trial_objective: None on the converged row
    forms:              typ.Optional[float]
    trial_retraction:   typ.Optional[float]      # every trial's retraction
    trial_objective:    typ.Optional[float]      # every trial's objective
    tracking:           typ.Optional[float]      # hessian='switch': a tracked row's tracking product, after the trials
    remainder:          float                    # window minus the phases above (>= 0 up to rounding)


@dc.dataclass(frozen=True)
class NewtonTiming:
    """:py:attr:`NewtonInfo.timing`: one history row's times, set by ``newton_cg(timing=True)`` (else ``None``).

    ``window`` is the row's ``wall_time`` window, the same float on a step row; the converged row, which has no
    ``wall_time``, gets one too. ``seconds`` splits it by phase (:py:class:`PhaseSeconds`); ``compile_seconds`` gives
    jax's compile wait inside each phase (``None`` on a numpy run, and on a jax whose ``jax.monitoring`` cannot
    unregister a listener). ``trials`` counts the retractions tried (``None`` on the converged row) and
    ``accepted_retraction`` is the accepted trial's share of ``trial_retraction`` (``None`` when none was accepted).

    Row 0 also carries the run's one-offs (``None`` on later rows): ``setup`` (the entry to the loop, the alignment
    excluded), ``alignment`` (``padded_from``'s padded start) and their compile twins; ``alignment_passes`` and
    ``reference_passes``, the seconds per pass of the alignment and of the ``'cone'`` reference's loop, breakdowns
    of ``alignment`` and of ``seconds.reference`` (``None`` when the geometry member cannot report them, e.g. under
    the uniform whole-loop jit)."""
    window:               float                                  # this row's window (== wall_time on a step row)
    seconds:              PhaseSeconds
    compile_seconds:      typ.Optional[PhaseSeconds]             # jax's compile wait per phase; None off jax / without unregister
    trials:               typ.Optional[int]                      # retraction trials; None on the converged row
    accepted_retraction:  typ.Optional[float]                    # the accepted trial's share of trial_retraction; None if none accepted

    setup:                typ.Optional[float]                 = None   # row 0 only, from here down
    setup_compile:        typ.Optional[float]                 = None
    alignment:            typ.Optional[float]                 = None   # padded_from only
    alignment_compile:    typ.Optional[float]                 = None
    alignment_passes:     typ.Optional[typ.Tuple[float, ...]] = None   # per pass; None under the uniform whole-loop jit
    reference_passes:     typ.Optional[typ.Tuple[float, ...]] = None   # 'cone' only


class _CompileSeconds:
    """The run's ``jax.monitoring`` duration listener: the running total of jax's compile wait. A host-side mutable
    accumulator, not a parameter-holding backend class (``parameters_not_closures.md`` does not apply): it never rides
    as aux and never enters a trace.

    It counts jax's three compile events, which do not nest (the trace event fires only at a clean trace state; a
    persistent-cache retrieval sits inside the backend event). The listener is process-global: compiles in other
    threads, or in a timed run nested inside a callback, are counted too."""
    EVENTS = ('/jax/core/compile/jaxpr_trace_duration',
              '/jax/core/compile/jaxpr_to_mlir_module_duration',
              '/jax/core/compile/backend_compile_duration')

    def __init__(self) -> None:
        self.total = 0.0                                                # compile seconds so far

    def __call__(self, event: str, duration: float, **kwargs) -> None:
        if event in self.EVENTS:
            self.total += duration

    @classmethod
    def register(
            cls,
            x0:  Cores,                           # the run's start point, after _prepare_jit_inputs
    ) -> typ.Optional['_CompileSeconds']:         # registered, or None: a numpy run, or a jax that cannot unregister it
        """A registered listener on a jax run whose ``jax.monitoring`` can unregister it (jax 0.4.30 cannot), else
        ``None``. The caller unregisters it in a ``finally``."""
        if not (jax_available and tree_contains_jax(x0)):
            return None
        import jax.monitoring as monitoring
        if not hasattr(monitoring, 'unregister_event_duration_listener'):
            return None
        listener = cls()
        monitoring.register_event_duration_secs_listener(listener)
        return listener

    def unregister(self) -> None:
        import jax.monitoring as monitoring
        monitoring.unregister_event_duration_listener(self)


class _PhaseClock:
    """The host clock of ``newton_cg(timing=True)``. A host-side mutable accumulator, like :py:class:`_CompileSeconds`.

    A *stretch* runs from one :py:meth:`close` reading to the next: the setup before the loop, then each history row's
    window, then the last row's callback. Its laps are disjoint sub-intervals, each from a :py:meth:`mark` (or the
    previous lap's end) to a synchronized end, summed by phase; the rest of the stretch is its ``remainder``. With a
    listener every lap also records the listener's change: the compile wait inside it."""

    def __init__(
            self,
            listener:  typ.Optional[_CompileSeconds],   # None: no compile column
            t:         float,                           # the run's first reading: the setup stretch starts here
    ) -> None:
        self.listener = listener
        self._laps = {}                                 # phase -> (seconds, compile seconds) in this stretch
        self._start = self._t = t                       # the stretch's start; the current lap's start
        self._start_c = self._c = self._compile()

    def _compile(self) -> float:
        return self.listener.total if self.listener is not None else 0.0

    def mark(self) -> None:
        """Start a lap now; the time since the last lap's end is no phase's (it is the stretch's remainder)."""
        self._t, self._c = time.perf_counter(), self._compile()

    def lap(
            self,
            phase:    str,               # a PhaseSeconds field, or a one-off ('alignment')
            outputs:  typ.Any = None,    # waited on first (a lap ending in float(...) needs none)
    ) -> float:                          # this lap's seconds
        """End the lap at a synchronized reading, add it (and its compile wait) to ``phase``, and start the next."""
        tree_block_until_ready(outputs)
        t, c = time.perf_counter(), self._compile()
        lap, compile_lap = t - self._t, c - self._c
        seconds, compile_seconds = self._laps.get(phase, (0.0, 0.0))
        self._laps[phase] = (seconds + lap, compile_seconds + compile_lap)
        self._t, self._c = t, c
        return lap

    def close(
            self,
            t:  float,                   # the stretch's end: the loop's own clock reading
    ) -> typ.Tuple[
        typ.Dict[str, float],                  # seconds by phase, plus 'remainder'
        typ.Optional[typ.Dict[str, float]],    # the compile wait likewise; None without a listener
    ]:
        """End the stretch at ``t`` (the next one starts there) and return its laps and remainder."""
        c = self._compile()
        seconds = {phase: s for phase, (s, _) in self._laps.items()}
        seconds['remainder'] = (t - self._start) - sum(seconds.values())
        compile_seconds = None
        if self.listener is not None:
            compile_seconds = {phase: k for phase, (_, k) in self._laps.items()}
            compile_seconds['remainder'] = (c - self._start_c) - sum(compile_seconds.values())
        self._laps = {}
        self._start = self._t = t
        self._start_c = self._c = c
        return seconds, compile_seconds

    def row(
            self,
            t:                    float,                    # the row's t_now: its window ends here
            trials:               typ.Optional[int],
            accepted_retraction:  typ.Optional[float],
            **one_offs,                                     # row 0: setup .. reference_passes (NewtonTiming's fields)
    ) -> NewtonTiming:
        """Close the row's window and return its record."""
        window = t - self._start                            # the loop's own wall_time: the same two readings
        seconds, compile_seconds = self.close(t)
        return NewtonTiming(window, _phase_seconds(seconds),
                            None if compile_seconds is None else _phase_seconds(compile_seconds),
                            trials, accepted_retraction, **one_offs)


def _phase_seconds(
        laps:  typ.Dict[str, float],     # a closed window's laps by phase, plus 'remainder'
) -> PhaseSeconds:                       # the phases that did not run are None
    """A window's laps as a record (an unknown phase name is a ``TypeError``)."""
    return PhaseSeconds(**{**dict.fromkeys(f.name for f in dc.fields(PhaseSeconds)), **laps})


def _mark(clock: typ.Optional[_PhaseClock]) -> None:
    """``clock.mark()``, or nothing without a clock."""
    if clock is not None:
        clock.mark()


def _lap(
        clock:    typ.Optional[_PhaseClock],
        phase:    str,
        outputs:  typ.Any = None,
) -> typ.Optional[float]:                # the lap's seconds; None without a clock
    """``clock.lap(phase, outputs)``, or nothing (no synchronization) without a clock."""
    return None if clock is None else clock.lap(phase, outputs)


def _timing_stats(
        rows:      typ.Sequence[NewtonTiming],   # the history rows' records, in order
        history:   typ.Sequence[dict],           # the history rows (their cg_iters)
        one_offs:  typ.Dict[str, typ.Any],       # setup .. alignment_passes, reference_passes (as row 0 carries them)
        total:     float,                        # the run's entry to the loop's exit

        last:      typ.Tuple[typ.Dict[str, float], typ.Optional[typ.Dict[str, float]]],   # the last callback's stretch (close)
) -> dict:                                       # stats['timing']
    """``stats['timing']``: the run's totals, by phase over the rows and one-off, and the rows' records as dicts. The
    pass tuples are breakdowns of ``alignment`` and of row 0's ``reference``, never added."""
    names = [f.name for f in dc.fields(PhaseSeconds)]
    total_of = lambda field: {name: sum(getattr(getattr(row, field), name) or 0.0 for row in rows) for name in names}
    listened = last[1] is not None
    compile_seconds = None
    if listened:
        compile_seconds = dict(total_of('compile_seconds'), setup=one_offs['setup_compile'],
                               alignment=one_offs['alignment_compile'], last_callback=last[1]['remainder'])
    iterations = sum(row['cg_iters'] or 0 for row in history)
    cg = sum(row.seconds.cg - (row.compile_seconds.cg if listened else 0.0)
             for row in rows if row.seconds.cg is not None)
    return {'total': total, 'setup': one_offs['setup'], 'alignment': one_offs['alignment'],
            'alignment_passes': one_offs['alignment_passes'], 'reference_passes': one_offs['reference_passes'],
            'last_callback': last[0]['remainder'], 'phases': total_of('seconds'), 'compile': compile_seconds,
            'cg_per_iteration': cg / iterations if iterations > 0 else None,
            'rows': [dc.asdict(row) for row in rows]}


@dc.dataclass(frozen=True)
class NewtonInfo:
    """Per-iteration diagnostics passed to a ``newton_cg`` ``callback`` -- everything one Newton step
    produces, so a custom callback (or the display in :py:mod:`t3toolbox.backend.optimizer_display`) can
    report anything without the loop anticipating it. ``x_cores`` (the point *before* the step) and ``lm``
    (its :py:class:`LocalModel` -- oracle / batch / frame / held) are carried so a callback can compute
    per-block errors or evaluate a validation forward. The step-related fields are ``None`` on the final
    **converged** line (no CG / line search ran). The scalar subset (all but ``x_cores`` / ``lm`` / ``timing``) is
    what lands in ``stats['history']``.

    Five strings (2026.3.0): ``newton_cg`` sets the first four on every row it emits, so a history consumer can
    filter by them, and ``stop_reason`` on its last row. All default to ``None``, so a hand-built row (a display
    test) needs none of them:

    * ``g0_reference`` -- which **Newton-stop** reference ``g0norm`` is: ``'zero'`` (the gradient norm at
      the zero tensor of these ranks, at the canonical zero-point frame), ``'warm_start'`` (the initial
      ``‖g‖``), ``'cone'`` (the gradient norm at the zero tensor on the aligned frame), or ``'override'``
      (``g0norm_newton`` was given).
    * ``cg_reference`` -- the CG forcing term's reference: ``'override'`` iff ``g0norm_cg`` was given, else
      equal to ``g0_reference`` (CG inherits the Newton reference).
    * ``step_control`` -- the step-acceptance mode: ``'line_search'`` (Armijo backtracking along the
      retraction) or ``'trust_region'`` (one trial at ``α = 1``, accepted on the ratio ``ρ``; Steihaug CG).
    * ``hessian`` -- the row's model: ``'gauss_newton'`` or ``'exact'`` (the model that solved the step; on the
      converged row, the model built there). The run's mode on every row, except under ``hessian='switch'`` (below).
    * ``stop_reason`` -- set on the run's **last** row only (``None`` before it): ``'gradient'`` (the
      relative stop fired: the converged row), ``'max_newton'`` (the iteration cap), ``'line_search'``
      (Armijo exhausted under ``on_line_search_failure='stop'``), ``'objective_floor'`` (a rejected
      trust-region step whose predicted change is below the objective's floating-point resolution -- the
      attainable accuracy is reached; **not a failure**) or ``'trust_region_floor'`` (rejections drove the
      radius below ``1e-12`` of its initial value); the same value as ``stats['stop_reason']``.

    **Mode-dependent fields.** A line-search row has ``ls_steps`` / ``ls_failed`` set and ``tr_*`` ``None``; a
    trust-region row has ``tr_radius`` (the radius this trial was bounded by) and ``tr_accepted`` set,
    ``ls_steps = ls_failed = None`` and ``alpha = 1.0``. A rejected trust-region row keeps ``x_cores``
    (the next row starts from the same point, with the same local model). ``pKp`` / ``lift_sigma_min`` /
    ``lift_eps`` are set only on a row whose ``hessian`` is ``'exact'`` (the last two only on a geometry whose
    curvature data carries them, i.e. the manifold one at ``d > 1``). The CG outcome is exactly one of
    ``cg_converged``, ``cg_boundary``, ``cg_truncated`` or (none of them) the iteration cap.

    **The switch** (``newton_cg(hessian='switch')``). A row's ``hessian`` is ``'gauss_newton'`` in the Gauss-Newton
    phase and ``'exact'`` from the switch on. On a Gauss-Newton-phase row ``lm`` is the Gauss-Newton view of the
    point's exact-mode build (the model that solved the step, the same object across a rejected trial; its
    ``gradient`` recomputes the misfit transpose), so ``pKp`` and the lift diagnostics are ``None`` there, as on a
    Gauss-Newton run's rows. Five fields, ``None`` outside switch mode and history fields only in switch mode (the
    display prints them as a Gauss-Newton-phase row's ``sw`` segment:
    :py:func:`~t3toolbox.backend.optimizer_display.format_newton_iter`):

    * ``switch_e_gn`` -- every Gauss-Newton-phase row with a step: ``|Δf − m_GN(s)|``, ``m_GN(s)`` the row's predicted
      change (``α·slope + ½α²·pHp``; ``α = 1`` in the trust region).
    * ``switch_pKp``, ``switch_e_n`` -- the *tracked* rows (the countable ones, or every Gauss-Newton-phase step row
      under ``switch_track='all'``): the curvature term ``⟨p, Lᵀ K_N L p⟩`` of the exact build (not in ``pHp``) and
      ``|Δf − m_N(s)|`` with ``m_N(s) = m_GN(s) + ½α²·switch_pKp``.
    * ``switch_count`` -- every Gauss-Newton-phase row: the count of consecutive hits after this row (0: a reset); the
      converged row's is the count carried to it.
    * ``switch_fired`` -- every switch-mode row: ``True`` on the row whose count reached ``switch_k``, else ``False``.

    ``timing`` is a :py:class:`NewtonTiming` with ``newton_cg(timing=True)``, else ``None``; like ``x_cores`` and
    ``lm`` it is not a history field (``stats['timing']['rows']`` holds the records as dicts). The display prints it
    as the row's ``time`` line (:py:func:`~t3toolbox.backend.optimizer_display.format_newton_iter`)."""
    iteration:    int              # Newton iteration index (0-based)
    objective:    float            # total objective ½‖ω⊙r‖² + ρ(X) at x (= misfit + regularization), point_objective's
                                   # value (the one the acceptance compares; lm.objective is the frame route's, = to rounding)
    gnorm:        float            # ‖g‖ (Riemannian gradient norm, geom.inner)
    g0norm:       float            # ‖g₀‖ Newton-stop reference in effect (its kind: g0_reference)
    converged:    bool             # gnorm <= gtol_rel*g0norm -- this is the final line (step fields None)
    x_cores:      typ.Any = None                       # the point BEFORE the step (for a val forward)
    lm:           typ.Any = None                       # the LocalModel at x (oracle/batch/frame/held)
    misfit:       typ.Optional[float] = None           # ½‖ω⊙r‖² data-misfit part of the objective
    regularization: typ.Optional[float] = None         # ρ(X) regularization part (None when no regularizer is attached)
    forcing_eta:  typ.Optional[float] = None           # inexact-Newton forcing term η
    cg_tol:       typ.Optional[float] = None           # CG stop tolerance = η·‖g‖
    cg_iters:     typ.Optional[int]   = None           # CG iterations run
    cg_maxiter:   typ.Optional[int]   = None           # CG iteration cap (for the "iters/maxiter" display)
    cg_resid:     typ.Optional[float] = None           # achieved ‖H p + g‖ (CG residual)
    cg_converged: typ.Optional[bool]  = None           # CG hit its tolerance inside the region (not boundary / truncated / maxiter)
    cg_truncated: typ.Optional[bool]  = None           # CG stopped on nonpositive curvature with no trust region (gauge-singular H)
    ls_steps:     typ.Optional[int]   = None           # Armijo backtracks (0 = full step; 40 = exhausted); None in trust-region mode
    ls_failed:    typ.Optional[bool]  = None           # Armijo exhausted all 40 halvings ('stop' rejected the step; 'accept' took it); None in TR mode
    alpha:        typ.Optional[float] = None           # step length of the LAST TRIAL (= the step taken, unless ls_failed under 'stop'); 1.0 in TR mode
    slope:        typ.Optional[float] = None           # gᵀp (the directional derivative along the step)
    pHp:          typ.Optional[float] = None           # pᵀHp of the mode's model (GN: ‖𝒥p‖² (+ reg); exact: + pKp)
    delta_f:      typ.Optional[float] = None           # actual objective change f_new − f
    rho:          typ.Optional[float] = None           # actual / predicted reduction (model trust)
    step_rel:     typ.Optional[float] = None           # ‖αp‖ / ‖x‖ (coordinate norms; relative step size)
    wall_time:    typ.Optional[float] = None           # seconds spent in this iteration (host clock)
    g0_reference: typ.Optional[str]   = None           # the Newton-stop reference: 'zero' | 'warm_start' | 'cone' | 'override'
    cg_reference: typ.Optional[str]   = None           # the CG-forcing reference: 'override' iff g0norm_cg, else = g0_reference
    step_control: typ.Optional[str]   = None           # 'line_search' | 'trust_region'
    stop_reason:  typ.Optional[str]   = None           # last row only: 'gradient' | 'max_newton' | 'line_search' | 'objective_floor' | 'trust_region_floor'
    hessian:      typ.Optional[str]   = None           # this row's model: 'gauss_newton' | 'exact'
    tr_radius:    typ.Optional[float] = None           # trust-region radius Δ this trial was bounded by (TR mode)
    tr_accepted:  typ.Optional[bool]  = None           # the trial was accepted (ρ > 0.1); False keeps x (TR mode)
    cg_boundary:  typ.Optional[bool]  = None           # CG stopped on the trust-region boundary ‖p‖ = Δ (Steihaug)
    pKp:          typ.Optional[float] = None           # ⟨p, Lᵀ K_N L p⟩: the curvature term's part of pHp (exact rows)
    lift_sigma_min: typ.Optional[float] = None         # the smallest lift singular value at this frame (exact rows)
    lift_eps:     typ.Optional[float] = None           # the absolute lift damping ε at this frame (exact rows)
    timing:       typ.Optional[NewtonTiming] = None    # newton_cg(timing=True) only: this row's times; not a history field
    switch_pKp:   typ.Optional[float] = None           # hessian='switch', tracked rows: ⟨p, Lᵀ K_N L p⟩ (not in pHp)
    switch_e_gn:  typ.Optional[float] = None           # hessian='switch', GN-phase rows with a step: |Δf − m_GN(s)|
    switch_e_n:   typ.Optional[float] = None           # hessian='switch', tracked rows: |Δf − m_N(s)|
    switch_count: typ.Optional[int]   = None           # hessian='switch', GN-phase rows: the hit count after this row
    switch_fired: typ.Optional[bool]  = None           # hessian='switch', every row: True on the row that switched


_SWITCH_FIELDS = ('switch_pKp', 'switch_e_gn', 'switch_e_n', 'switch_count', 'switch_fired')   # history: switch mode only
_NEWTON_SCALAR_FIELDS = tuple(f.name for f in dc.fields(NewtonInfo)
                              if f.name not in ('x_cores', 'lm', 'timing') + _SWITCH_FIELDS)


def _newton_scalar_record(info: NewtonInfo) -> dict:
    """The scalar subset of a :py:class:`NewtonInfo` (drops ``x_cores`` / ``lm`` / ``timing``) -- one ``history``
    row. The five ``switch_*`` fields follow, in switch mode only (``switch_fired`` is a ``bool`` on every row of a
    ``hessian='switch'`` run and ``None`` otherwise), so the rows of the other modes keep their keys."""
    record = {k: getattr(info, k) for k in _NEWTON_SCALAR_FIELDS}
    if info.switch_fired is not None:
        record.update((k, getattr(info, k)) for k in _SWITCH_FIELDS)
    return record


# --------------------------------------------------------------------------------------------------
# Newton-CG: step control (line search / trust region) and its constants
# --------------------------------------------------------------------------------------------------
_STEP_CONTROLS = ('line_search', 'trust_region')

# The trust-region policy (Nocedal & Wright, Alg. 4.1, with Steihaug CG). Module constants, not options.
_TR_ACCEPT = 0.1               # accept the trial iff ρ > this
_TR_SHRINK_BELOW = 0.25        # ρ below this, or non-finite: Δ ← _TR_SHRINK·‖p‖
_TR_SHRINK = 0.25
_TR_EXPAND_ABOVE = 0.75        # ρ above this on a boundary step: Δ ← min(_TR_EXPAND·Δ, Δ_max)
_TR_EXPAND = 2.0
_TR_RADIUS_MAX_REL = 1e4       # Δ_max = this·Δ0
_TR_RESOLUTION_FACTOR = 10     # objective_floor iff |pred| ≤ this·ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)  (rejection only)
_TR_RADIUS_FLOOR_REL = 1e-12   # trust_region_floor iff Δ < this·Δ0                         (rejection only)


def _check_step_control(
        step_control:  typ.Optional[str],   # None | 'line_search' | 'trust_region' (case-insensitive)
        hessian:       str,                 # the validated mode: sets the default
) -> str:                                   # the lower-cased step control
    """Validate ``step_control``; ``None`` is the mode's default -- ``'line_search'`` with the Gauss-Newton
    Hessian (positive semidefinite: Armijo along the retraction suffices), ``'trust_region'`` with the
    exact one (indefinite away from a minimizer: the region bounds the step) and with ``'switch'``, which ends in it."""
    if step_control is None:
        return 'trust_region' if hessian in ('exact', 'switch') else 'line_search'
    if not isinstance(step_control, str) or step_control.lower() not in _STEP_CONTROLS:
        raise ValueError("newton_cg: step_control must be None, 'line_search' or 'trust_region' "
                         "(case-insensitive); got %r" % (step_control,))
    return step_control.lower()


def _floating_dtypes(
        tree:  typ.Any,              # an array, or nested tuples / lists of numpy or jax arrays
) -> typ.Iterator[np.dtype]:         # the dtype of each floating leaf, in tree order
    """The dtypes of a tree's floating leaves (nested tuples / lists of arrays), read off the leaves only."""
    if isinstance(tree, typ.Sequence) and not isinstance(tree, (str, bytes)):
        for t in tree:
            yield from _floating_dtypes(t)
        return
    dtype = getattr(tree, 'dtype', None)
    if dtype is not None and np.issubdtype(dtype, np.inexact):
        yield dtype


def _resolution_eps(
        value:  typ.Any,             # the model's f (an array), or any tree of arrays; numpy or jax
) -> float:                          # np.finfo(dtype).eps, maximised over the floating leaves
    """The machine epsilon the objective is computed in -- host-side and **dtype-only** (no array is read, so no device
    sync): the largest ``np.finfo(dtype).eps`` over the floating leaves of ``value`` (the optimizers pass the model's
    ``f``), so a float32 run is judged at float32's resolution and a mixed tree at its coarsest. ``float64``'s eps if
    there is no floating leaf."""
    eps = [float(np.finfo(dtype).eps) for dtype in _floating_dtypes(value)]
    return max(eps) if eps else float(np.finfo(np.float64).eps)


def _objective_resolution(
        lm:              'LocalModel',         # the model at the point
        regularization:  typ.Optional[float],  # rho(X) at the point; None when unregularized
) -> float:                                    # f_resolution + eps |rho|, or eps (|f| + |rho|)
    """The objective's floating-point resolution at the model's point, the scale below which a change of the objective
    cannot be measured. The trust region's ``objective_floor`` reads it (times ``_TR_RESOLUTION_FACTOR``), and so does
    the switch's rounding gate (times ``_SWITCH_ROUNDING`` on top).

    The misfit's part is the oracle's optional ``f_resolution`` (for least squares the cancellation error of
    ``1/2 |S(x) - y|^2``, ``eps (|omega y| |omega r| + |f|)``, scaled on a drawn batch), queried here at the model's
    frame with its ``held``; the regularizer adds ``eps |rho|`` (the oracle's covers the misfit only; together they are
    the 2026.5.0 floor ``eps (|omega y| |omega r| + |f + rho|)``). An oracle without ``f_resolution``: ``eps |f|`` and
    ``eps |rho|``. ``eps`` is read off the dtype of the model's ``f`` (TARGET_DECISIONS 8.4). The resolution is a scale
    estimate, so either route's ``|r|`` serves: the frame route's here sizes the point-route values the acceptance
    compares (D9) as well."""
    eps = _resolution_eps(lm.f)
    rho = 0.0 if regularization is None else abs(float(regularization))
    if 'f_resolution' in tuple(getattr(lm.oracle, 'optional_outputs', ())):
        (f_resolution,), _ = lm.oracle.query(lm.frame, lm.batch, ('f_resolution',), Vc=lm._Vc, held=lm.held)
        return float(f_resolution) + eps * rho
    return eps * (abs(float(lm.f)) + rho)


_TINY = float(np.finfo(np.float64).tiny)   # the smallest positive normal float (== sys.float_info.min)


def _initial_radius(
        lm:     LocalModel,          # iteration 0's local model
        g:      Tangent,             # its gradient
        gnorm:  float,               # ‖g‖ > 0
) -> float:                          # Δ0 > 0
    """The default initial trust-region radius: the length of the Gauss-Newton Cauchy step,
    ``‖g‖³ / ⟨g, H_GN g⟩`` (the minimizer of the GN model along ``−g``), whatever the Hessian; ``‖g‖`` when
    that is not a positive finite number (a vanishing or non-finite ``⟨g, H_GN g⟩``)."""
    q = float(lm.gn_quadratic(g))
    radius0 = (gnorm * gnorm * gnorm) / max(q, _TINY)     # products, not **: inf, never OverflowError
    return radius0 if (math.isfinite(radius0) and radius0 > 0.0) else gnorm


def _lift_diagnostics(
        lm: LocalModel,              # an exact- or GN-mode model
) -> typ.Tuple[
    typ.Optional[float],             # lift_sigma_min, or None
    typ.Optional[float],             # lift_eps, or None
]:
    """The exact-mode lift diagnostics, read duck-typed off the geometry's curvature data (its optional
    ``sigma_min`` / ``eps`` attributes; :py:class:`Geometry`). ``None`` in Gauss-Newton mode, where the data
    has none (corewise), or where the curvature term vanishes (``geometry_data is None``)."""
    data = lm.curvature.geometry_data if lm.curvature is not None else None
    sigma_min, eps = getattr(data, 'sigma_min', None), getattr(data, 'eps', None)
    return (None if sigma_min is None else float(sigma_min)), (None if eps is None else float(eps))


# --------------------------------------------------------------------------------------------------
# Newton-CG: the Gauss-Newton-to-exact switch (hessian='switch'). Start with Gauss-Newton; after each tracked step
# compare how well each model predicted the actual change, and switch to the exact Hessian for good once the exact
# model was switch_tau times better on switch_k consecutive countable steps.
# --------------------------------------------------------------------------------------------------
_SWITCH_TAU = 0.25                      # switch_tau=None: e_N < this·e_GN is a hit
_SWITCH_K = 2                           # switch_k=None: this many consecutive hits switch
_SWITCH_CG_FORCING_POWER = 1.0          # switch_cg_forcing_power=None: the exact phase's (q-quadratic tail)
_SWITCH_TRACKS = ('countable', 'all')   # switch_track's values; None = the first
_SWITCH_ROUNDING = 100.0                # the rounding gate: floor = this·_TR_RESOLUTION_FACTOR·_objective_resolution


def _check_switch_options(
        hessian:                  str,                    # the validated mode (_check_hessian)
        switch_tau:               typ.Optional[float],    # None = _SWITCH_TAU; a number in [0, 1)
        switch_k:                 typ.Optional[int],      # None = _SWITCH_K; an int >= 0, not a bool
        switch_cg_forcing_power:  typ.Optional[float],    # None = _SWITCH_CG_FORCING_POWER; unchecked, as cg_forcing_power
        switch_track:             typ.Optional[str],      # None = 'countable'; 'countable' | 'all' (case-insensitive)
) -> typ.Optional[typ.Tuple[
    float,                                                # tau
    int,                                                  # k
    float,                                                # the forcing power from the switch on
    typ.Optional[str],                                    # the tracked rows: 'countable' | 'all'; None iff k = 0
]]:                                                       # None outside switch mode
    """``newton_cg``'s ``switch_*`` options, checked (structural, before any work) and resolved to their values.

    Each option set with ``hessian`` other than ``'switch'`` raises, as does one the run would not read: ``switch_tau``
    or ``switch_track`` with ``switch_k=0`` (exact from iteration 0, nothing is tracked), ``switch_k`` or
    ``switch_cg_forcing_power`` with ``switch_tau=0`` (the count never moves: the run tracks and never switches)."""
    given = sorted(name for name, value in (('switch_tau', switch_tau), ('switch_k', switch_k),
                                            ('switch_cg_forcing_power', switch_cg_forcing_power),
                                            ('switch_track', switch_track)) if value is not None)
    if hessian != 'switch':
        if given:
            raise ValueError("newton_cg: %s set without hessian='switch'" % ', '.join(given))
        return None
    if switch_tau is not None and (isinstance(switch_tau, bool) or not isinstance(switch_tau, numbers.Real)
                                   or not 0.0 <= switch_tau < 1.0):
        raise ValueError("newton_cg: switch_tau must be a number in [0, 1); got %r" % (switch_tau,))
    if switch_k is not None and (isinstance(switch_k, bool) or not isinstance(switch_k, numbers.Integral)
                                 or switch_k < 0):
        raise ValueError("newton_cg: switch_k must be an int >= 0 (not a bool); got %r" % (switch_k,))
    if switch_track is not None and (not isinstance(switch_track, str) or switch_track.lower() not in _SWITCH_TRACKS):
        raise ValueError("newton_cg: switch_track must be None, 'countable' or 'all' (case-insensitive); got %r"
                         % (switch_track,))
    if switch_k == 0 and (switch_tau is not None or switch_track is not None):
        raise ValueError("newton_cg: %s set with switch_k=0, which tracks nothing (the run is exact from iteration 0)"
                         % ', '.join(name for name in given if name in ('switch_tau', 'switch_track')))
    if switch_tau == 0 and (switch_k is not None or switch_cg_forcing_power is not None):
        raise ValueError("newton_cg: %s set with switch_tau=0, which never switches (the run tracks only)"
                         % ', '.join(name for name in given if name in ('switch_cg_forcing_power', 'switch_k')))
    tau = _SWITCH_TAU if switch_tau is None else float(switch_tau)
    k = _SWITCH_K if switch_k is None else int(switch_k)
    power = _SWITCH_CG_FORCING_POWER if switch_cg_forcing_power is None else switch_cg_forcing_power
    track = None if k == 0 else (_SWITCH_TRACKS[0] if switch_track is None else switch_track.lower())
    return tau, k, power, track


def _switch_sample(
        delta_f:    float,           # f(trial) − f
        predicted:  float,           # m_GN(s): α·slope + ½α²·pHp (line search), slope + ½·pHp (trust region)
        curvature:  float,           # m_N(s) − m_GN(s) = ½α²·switch_pKp
        floor:      float,           # the rounding floor, >= 0
        tau:        float,           # switch_tau
) -> typ.Tuple[
    float,                           # e_gn = |Δf − m_GN(s)|
    float,                           # e_n  = |Δf − m_N(s)|
    bool,                            # informative: both finite, and not both <= floor
    bool,                            # hit: informative and e_n < tau·e_gn
]:
    """One tracked step of the switch: how far each quadratic model's predicted change missed the actual one.

    Near a minimizer with a nonzero residual, along a second-order retraction, ``e_gn = ½|⟨s, K_N s⟩| + O(‖s‖³)`` while
    ``e_n = O(‖s‖³)``, so ``e_n / e_gn → 0``; in the noisy transient the ratio is about 1 or more. Below the objective's
    resolution neither error can be measured: a sample whose two errors are both at or below ``floor`` is
    uninformative, never a hit (so is a non-finite one). ``e_n`` at the floor with ``e_gn`` above it is a hit."""
    e_gn = abs(delta_f - predicted)
    e_n = abs(delta_f - (predicted + curvature))
    informative = math.isfinite(e_gn) and math.isfinite(e_n) and max(e_gn, e_n) > floor
    return e_gn, e_n, informative, informative and e_n < tau * e_gn


def _gauss_newton_view(
        local_model:  LocalModel,    # an exact-mode model (local_model(..., hessian='exact'))
) -> LocalModel:                     # the same linearization in Gauss-Newton mode
    """The Gauss-Newton model of an exact-mode model's linearization: ``curvature=None``, every other field shared
    (frame, centers, held, f, companion, regularizer), so its ``hvp`` / ``quadratic`` / ``gn_quadratic`` / ``retract``
    are a Gauss-Newton model's bit for bit, gauge and regularizer included. Its ``gradient`` redoes the misfit
    gradient's query; ``newton_cg`` reads the exact model's, which is the same value. Its pytree structure is a
    Gauss-Newton build's when the two builds hold the same slots, as at every frame the library builds (which carries
    its centers): there under ``use_jit`` CG reuses the Gauss-Newton program. At a user's start frame without centers
    the exact build of the probe kind holds more (its ``g`` computed the right sweeps), so one more program compiles."""
    return dc.replace(local_model, curvature=None)


def newton_cg(
        geom:                   typ.Any,  # the Geometry
        oracle:                 typ.Any,  # the Oracle
        batch:                  typ.Any,  # its batch (a LeastSquaresBatch; uniform: packed, pack_batch)
        x0:                     Tangent,  # initial cores (B, G); uniform: the bare supercore pair
        regularizer:            typ.Any = None,  # optional backend.regularization.Regularizer

        max_newton:             int   = 30,    # Newton iterations (history rows) -- rejected trust-region trials included
        gtol_rel:               float = 1e-8,  # stop when ‖g‖ <= gtol_rel * ‖g0‖  (‖g0‖: g0norm_newton, else the g0_reference)
        cg_maxiter:             int   = 200,
        c_armijo:               float = 1e-4,  # Armijo constant (line search only)
        on_line_search_failure: str = 'stop',  # Armijo exhausted (40 halvings): 'stop' = reject the step and terminate; 'accept' = take the last trial and continue

        hessian:                str   = 'gauss_newton',      # or 'exact' | 'switch' (case-insensitive): the Hessian; 'switch' = GN, then exact
        curvature_damping:      typ.Optional[float] = None,  # exact / switch only: the lift damping, relative to ‖X‖; None = the geometry's default
        step_control:           typ.Optional[str]   = None,  # None = 'line_search' (GN) / 'trust_region' (exact, switch); either works with either
        tr_radius0:             typ.Optional[float] = None,  # trust region only: Δ0 > 0; None = the GN Cauchy length ‖g‖³/⟨g, H_GN g⟩

        g0_reference:           typ.Optional[str]   = None,  # which ‖g0‖: None = the geometry's own ('zero' on manifolds, 'warm_start' on corewise); 'zero' | 'warm_start' | 'cone'
        g0norm_newton:          typ.Optional[float] = None,  # explicit ‖g0‖ for the Newton stop (wins; > 0). Also feeds CG unless g0norm_cg set.
        g0norm_cg:              typ.Optional[float] = None,  # explicit ‖g0‖ for the CG forcing term (wins for CG; > 0); default = the Newton reference

        cg_forcing_power:       float = 0.5,    # η = min(0.5, (‖g‖/‖g0‖)**power); larger => tighter CG, fewer Newton steps
        use_jit:                bool  = False,  # jit the inner CG (lax.while_loop) when the inputs are jax; else eager

        callback:               typ.Optional[typ.Callable] = None,  # callback(NewtonInfo) each iteration (host-side; e.g. a display)
        *,
        start_frame:                typ.Any = None,             # a frame of x0 for iteration 0 (structurally checked)
        padded_from:                typ.Any = None,             # the old point: start_frame = padded_start(geom, oracle, batch, x0, padded_from, padded_*)

        padded_oversampling:        typ.Optional[int]   = None, # None = 5    (padded_from only, else a structural error)
        padded_oversampled_passes:  typ.Optional[int]   = None, # None = 2
        padded_max_passes:          typ.Optional[int]   = None, # None = 10
        padded_tol:                 typ.Optional[float] = None, # None = 0.05
        padded_alternate:           typ.Optional[bool]  = None, # None = True
        padded_seed:                typ.Optional[int]   = None, # None = ZERO_FRAME_SEED

        cone_oversampling:          typ.Optional[int]   = None, # None = 5    (g0_reference='cone' without g0norm_newton only, else a structural error)
        cone_oversampled_passes:    typ.Optional[int]   = None, # None = 2
        cone_max_passes:            typ.Optional[int]   = None, # None = 10
        cone_tol:                   typ.Optional[float] = None, # None = 0.05
        cone_alternate:             typ.Optional[bool]  = None, # None = True
        cone_seed:                  typ.Optional[int]   = None, # None = ZERO_FRAME_SEED

        timing:                     bool = False,               # NewtonInfo.timing per row, stats['timing'] per run

        switch_tau:                 typ.Optional[float] = None, # None = 0.25; in [0, 1); 0 = never switch (hessian='switch' only, else a structural error)
        switch_k:                   typ.Optional[int]   = None, # None = 2; an int >= 0; 0 = exact from iteration 0
        switch_cg_forcing_power:    typ.Optional[float] = None, # None = 1.0: the forcing power from the switch on
        switch_track:               typ.Optional[str]   = None, # None = 'countable'; 'countable' | 'all': the rows that track
) -> typ.Tuple[Tangent, dict]:      # (x_cores, stats)
    """Inexact Riemannian Newton-CG -- the manifold workhorse. Each Newton step builds the local model once,
    solves ``H p = −g`` by (Steihaug) CG to an inexact forcing-term tolerance (the inner loop -- jit-able via
    :py:func:`_cg_solve`), then controls the step by a line search or a trust region. ``use_jit`` jits only
    the inner CG (the outer loop, step control, and convergence test stay on the host), auto-converting
    numpy inputs to jax first (:py:func:`_prepare_jit_inputs`) -- so a ``use_jit=True`` call on numpy data
    returns a jax-backed result (jax's default float32 unless x64 is enabled); if jax is not installed it
    runs eager with a warning.

    **The Hessian** (``hessian``). ``'gauss_newton'`` (default): ``H = Π 𝒥ᵀω²𝒥 Π`` (+ the regularizer's),
    positive semidefinite -- the gauged ``H`` is positive-definite on the manifold, and CG truncates on the
    gauge-singular corewise ``H``. ``'exact'``: the exact Riemannian Hessian, which adds the curvature term
    ``Π Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π`` (:py:func:`local_model`; it needs an oracle supplying ``Kp`` and a geometry
    with the four exact-mode members). Gauss-Newton converges linearly at
    a rate set by that missing term (``|1 − ρ|`` measures it); the exact Hessian gives the superlinear Newton
    tail -- q-quadratic with ``cg_forcing_power=1.0``, order about 1.5 at the default ``0.5``. It costs a
    few times a Gauss-Newton Hessian-apply. ``curvature_damping`` (exact and switch only; a value with Gauss-Newton
    raises) is the lift's damping relative to ``‖X‖``, ``None`` for the geometry's own default. ``'switch'`` starts
    with Gauss-Newton and switches to the exact Hessian within the run (below).

    **Switching from Gauss-Newton to the exact Hessian** (``hessian='switch'``; it needs what ``'exact'`` needs, checked
    at the call). Each point of the Gauss-Newton phase is linearized once in exact mode (one misfit transpose, as in
    either pure mode, plus the geometry's curvature data); CG, the step control and the row use its Gauss-Newton view,
    so the steps are a Gauss-Newton run's. After a step, ``e_GN = |Δf − m_GN(s)|`` (``m_GN(s)`` the row's predicted
    change) and, on a *countable* row, ``e_N = |Δf − m_N(s)|`` with ``m_N(s) = m_GN(s) + ½α²⟨s, Π Lᵀ K_N L Π s⟩`` (one
    curvature product). A countable row with ``e_N < switch_tau·e_GN`` (default ``0.25``) is a hit; ``switch_k``
    (default ``2``) consecutive hits switch to the exact Hessian for good, from the next point on: **the exact model
    must have predicted the objective at least four times better, on two clean steps in a row**. A row is countable
    iff it took the full model step (an accepted trust-region trial; an Armijo step with ``α = 1``), so a rejected
    trial, a backtrack, the ``−g`` fallback and a forced ``'accept'`` reset the count, as do a miss and a sample whose
    two errors are both at the objective's rounding level, ``1000·ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)`` (the trust region's floor
    times 100; under float32 only large early steps clear it -- prefer jax x64); a CG step stopped on the region's
    boundary, or truncated, counts like any other. ``switch_track='all'`` runs the product on every Gauss-Newton step
    row (reported, never counted: the switch fires at the same row). The default step control is the trust region,
    whose radius carries across the switch; ``cg_forcing_power`` is the Gauss-Newton phase's forcing power and
    ``switch_cg_forcing_power`` (default ``1.0``) the exact phase's. ``switch_tau=0`` tracks and never switches (the
    run is the Gauss-Newton run with the same step control, bit for bit, plus the tracking); ``switch_k=0`` is exact
    from iteration 0 (the exact run with ``cg_forcing_power=switch_cg_forcing_power``). Each call starts in
    Gauss-Newton: nothing carries across calls, so a continuation level after a rank increase starts there too. It may
    never fire, safely: the curvature term vanishes (the manifold at ``d = 1``), the residual is zero, or the retraction
    is only first order at the point (near small singular values, a zero-padded start). Each row records the model that
    solved it (``hessian``) and the ``switch_*`` fields (:py:class:`NewtonInfo`); ``stats`` gains three keys (below). A
    ``switch_*`` option without ``hessian='switch'`` raises, as does one the run would not read (``switch_tau`` or
    ``switch_track`` with ``switch_k=0``; ``switch_k`` or ``switch_cg_forcing_power`` with ``switch_tau=0``).
    The user guide, with a worked run: ``docs/fitting_and_optimization.md`` §4.14.

    **Step control** (``step_control``; the default follows the Hessian, and either combines with either):

    * ``'line_search'`` (the Gauss-Newton default): plain truncated CG, then Armijo backtracking along
      ``retract(α p)``.
    * ``'trust_region'`` (the exact and switch default -- the exact ``H`` is indefinite away from a minimizer):
      Steihaug-Toint CG inside ``‖p‖ ≤ Δ`` (it steps to the boundary on nonpositive curvature or when it
      would leave the region), then **one** trial ``retract(p)``, accepted iff the ratio of the actual to the
      model's predicted change ``ρ = Δf / (gᵀp + ½ pᵀHp)`` exceeds ``0.1``. ``Δ ← ¼‖p‖`` when ``ρ < 0.25`` (or
      not finite); ``Δ ← min(2Δ, 10⁴Δ0)`` when ``ρ > 0.75`` on a boundary step. ``Δ0 = tr_radius0``, by
      default the Gauss-Newton Cauchy length ``‖g‖³ / ⟨g, H_GN g⟩``. A rejected trial keeps ``x`` -- and the
      local model, which is rebuilt only after an accepted step -- emits its own history row
      (``tr_accepted=False``) and counts toward ``max_newton``. Two **floors**, checked only after a rejection
      (an accepted step never stops the run): ``'objective_floor'`` when the predicted change is below the
      objective's floating-point resolution, ``|pred| ≤ 10·(f_resolution + ε|ρ|)`` with ``f_resolution`` the oracle's
      (for least squares ``ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)``; ``ε|f|`` for an oracle without it) and ``ε`` the machine epsilon
      of ``f``'s dtype -- the attainable accuracy is reached, which is **not a failure** (``converged``
      stays ``False`` because the gradient test did not fire); else ``'trust_region_floor'`` when ``Δ`` has
      fallen below ``10⁻¹²·Δ0``. ``tr_radius0`` with ``'line_search'`` raises, as does
      ``on_line_search_failure='accept'`` with ``'trust_region'`` (neither would have an effect).

    **The reference gradient norm ‖g0‖** (``g0_reference`` / ``g0norm_newton`` / ``g0norm_cg`` /
    ``cg_forcing_power``). Both stopping tests are *relative* to a reference ‖g0‖: the Newton stop is
    ``‖g‖ ≤ gtol_rel·‖g0‖`` and the CG forcing term is ``η = min(0.5, (‖g‖/‖g0‖)**cg_forcing_power)``.
    Which reference, resolved once at iteration 0 (``docs/fitting_and_optimization.md`` §5):

    * ``g0_reference=None`` (default) -- **the geometry's own**: ``'zero'`` on a manifold geometry (one
      with a ``zero_point_frame``), ``'warm_start'`` on a corewise one (the zero cores are a critical point
      there, so the zero reference would be exactly 0) and on a custom geometry without the member.
    * ``'zero'`` -- the gradient norm at the **zero tensor of x0's ranks**, evaluated at the canonical
      zero-point frame (:py:func:`~t3toolbox.backend.fv_operations.fv_zero_point_frame`): one extra query
      wanting ``g`` on the batch. A deterministic data-gradient scale at these ranks, fixed by
      convention and good to an O(1) factor -- comparable across rank-continuation levels and identical on
      the ragged and uniform layers. It is the right reference after a **warm start**, whose initial
      gradient is deflated (the guess is already near the solution). An explicit ``'zero'`` on a geometry
      with no zero-point frame raises ``ValueError``.
    * ``'warm_start'`` -- the initial gradient norm ``‖g(x0)‖`` (the 2026.2.0 default; ``1.0`` if it is 0,
      when x0 is stationary and the stop fires at once).
    * ``'cone'`` -- the gradient norm at the zero tensor of x0's ranks, at the frame there ALIGNED with the
      data gradient (:py:func:`padded_start` from the zero tensor; the ``cone_*`` options, ``use_jit`` as
      below). ``'zero'`` evaluates it at a fixed, data-independent frame, which can miss most of the gradient;
      ``cone_max_passes=0`` evaluates it at the zero tensor's hint-free start (random orthonormal cores, a
      different frame from ``'zero'``'s). ``stats['cone_movements']`` records the passes.

    **The start frame** (``start_frame`` / ``padded_from``). Iteration 0 linearizes at ``geom.frame(x0)`` by
    default. ``start_frame`` gives a frame of ``x0`` instead (structurally checked by the geometry's
    ``require_frame_structure``); ``padded_from=x_old`` builds it: ``x0`` is ``x_old`` zero-padded to the target
    ranks, whose own frame is arbitrary on the new slots, and :py:func:`padded_start` aligns the new slots with the
    misfit gradient (the ``padded_*`` options; ``padded_max_passes=0`` is the hint-free padded frame). Both
    need a manifold geometry. ``use_jit`` governs the alignment as well as CG. With either keyword,
    ``stats['start_frame']`` is ``'given'`` or ``'padded'`` and ``stats['start_movements']`` the passes. An
    option of a group set without its group engaged (``padded_*`` without ``padded_from``, ``cone_*`` without
    ``'cone'`` or with ``g0norm_newton``) raises, as does zero passes with an option they would not read.

    A **vanishing** zero reference (the data gradient at the zero tensor is 0, e.g. all-zero data) is never
    silently replaced by a constant: by default it falls back to the warm start with a ``RuntimeWarning``;
    an explicit ``'zero'`` raises ``ValueError``. The explicit overrides then win, with a chained fallback:
    ``g0norm_newton`` sets the Newton reference (and CG inherits it unless ``g0norm_cg`` is also given --
    so with ``g0norm_newton`` given the zero reference is not computed at all); ``g0norm_cg`` alone sets
    only the CG reference. A non-positive override raises ``ValueError``. Every history row records the
    kinds (``g0_reference``, ``cg_reference``; ``'override'`` for an explicit value).
    ``cg_forcing_power`` (default ``0.5``, the conventional Eisenstat-Walker value, in both modes) tunes CG
    effort per Newton step: since ``‖g‖/‖g0‖ < 1`` near the solution, a **larger** power (``0.75``, ``1.0``)
    tightens CG -> more CG iterations but fewer Newton steps -- worth it on the manifold when the retraction
    is expensive relative to a Hessian-apply, and ``1.0`` is what makes the exact Hessian's tail
    q-quadratic. The ``min(0.5, …)`` cap on ``η`` is retained regardless.

    **Line-search exhaustion** (``on_line_search_failure``). If all 40 Armijo halvings fail, the
    objective's floor has been reached (typically deep in a continuation run). The default ``'stop'``
    rejects the step and terminates -- the conventional reading, and it stops the ~2×40 objective
    evaluations per stagnant iteration. ``'accept'`` instead takes the last (smallest) trial step and
    keeps iterating even though it may climb -- a deliberate escape mode: on some landscapes the small
    non-descent step bumps the iterate out of a stagnant region, after which convergence resumes.
    Either way the iteration's :py:class:`NewtonInfo` row carries ``ls_failed=True`` and ``alpha`` is
    the step of the last trial actually evaluated (``2^-39``), so ``delta_f`` / ``rho`` / ``step_rel``
    describe a step that was genuinely tried.

    ``callback``, if given, is called with a :py:class:`NewtonInfo` each iteration (including the final
    converged line) -- the hook for a live diagnostic display; it runs **host-side** (it reads concrete
    values), so it composes with ``use_jit`` (only the inner CG jits) but not with a hypothetical
    fully-jitted outer loop. Ready-made displays: :py:func:`t3toolbox.backend.optimizer_display.make_newton_display`.

    ``stats`` is ``{'losses', 'newton_steps', 'newton_accepted', 'stop_reason', 'history'}``:
    ``newton_steps`` counts the CG solves (the history rows that took a step), ``newton_accepted`` the steps
    taken (``newton_steps`` minus the rejections: an Armijo-exhausted one under ``'stop'``, or the rejected
    trust-region trials), ``stop_reason`` is ``'gradient'`` / ``'max_newton'`` / ``'line_search'`` /
    ``'objective_floor'`` / ``'trust_region_floor'`` (see :py:class:`NewtonInfo`), and ``history`` is one
    :py:func:`_newton_scalar_record` per iteration (``losses[k] == history[k]['objective']``). The reported objective,
    misfit and regularization are :py:func:`point_objective`'s at the row's point, the values the step acceptance
    compares: the start's is one query, each later one the accepted trial's, carried, so on an accepted row the
    next row's objective is the trial's value itself and ``objective + delta_f`` equals it to the rounding of
    ``delta_f``. (The 2026.2.0
    ``stats['newton']`` is removed; it is ``newton_steps``.) ``hessian='switch'`` adds ``stats['hessian'] = 'switch'``,
    ``stats['switch_track']`` (``'countable'`` or ``'all'``; ``None`` with ``switch_k=0``) and ``stats['switched_at']``,
    the index of the first row whose model is exact (``len(history)`` when the switch fired on the last row; ``0``
    with ``switch_k=0``; ``None`` if it never fired). ``timing=True`` adds ``stats['timing']`` (below).

    **Timing** (``timing=True``; it changes no number). Each row's ``wall_time`` window is split by phase into
    :py:attr:`NewtonInfo.timing` (a :py:class:`NewtonTiming`; :py:class:`PhaseSeconds` lists the phases), with jax's
    compile wait per phase on a jax run (``None`` on numpy, and on a jax whose ``jax.monitoring`` cannot unregister a
    listener). ``stats['timing']`` is a plain dict: ``total``, ``setup``, ``alignment``, ``alignment_passes``,
    ``reference_passes``, ``last_callback`` (the last row's callback, outside every window), ``phases`` (each phase
    summed over the rows), ``compile`` (the same for the compile wait, with ``setup``, ``alignment`` and
    ``last_callback``; ``None`` as above), ``cg_per_iteration`` (CG's seconds less its compile wait, per CG iteration)
    and ``rows`` (each row's record as a dict). ``total == setup + (alignment or 0) + Σ window + last_callback``.
    Every phase ends by waiting on its outputs (jax dispatch is asynchronous), so a timed jax run is serialized and
    slower than an untimed one; work dispatched without waiting (a callback's, or ``-g`` before CG) is charged to the
    next phase that waits for it. The compile listener is process-global: compiles in other threads, or in a timed
    run nested inside a callback, are counted too. The geometry member ``aligned_padded_frame`` is
    asked for its seconds per pass only when its signature names ``timing`` (see :py:class:`Geometry`). With
    :py:func:`~t3toolbox.backend.optimizer_display.make_newton_display` as the callback, each row prints a ``time``
    line and the run ends with a closing table. What each phase covers, what the synchronization costs and how the
    compile column reads under ``use_jit`` and eager jax: the user guide, ``docs/fitting_and_optimization.md`` §4.13.

    Examples
    --------
    A backend-only fit: the geometry, the oracle and its batch, and ``newton_cg``. The exact Hessian with its default
    trust region, against Gauss-Newton with its default line search, on a noisy fit of probes along lines (the jet
    shape ``(1,)``) warm-started near the solution:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.optimizers as bopt
    >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> from t3toolbox.backend.geometry import ManifoldGeometryOps
    >>> from t3toolbox.backend.jets import affine_jet
    >>> from t3toolbox.backend.probing import t3_probe
    >>> np.random.seed(0)
    >>> shape, rng = (5, 5, 5), np.random.default_rng(0)
    >>> ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
    >>> oracle = LeastSquaresOracle('probe', order=(1,))
    >>> target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1)).data
    >>> y = [z + 0.01 * rng.standard_normal(z.shape) for z in t3_probe(target, affine_jet(ww, pp), (1,))]
    >>> geom, batch = ManifoldGeometryOps(), LeastSquaresBatch(affine_jet(ww, pp), y)
    >>> x0, _ = bopt.newton_cg(geom, oracle, batch, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data,
    ...                        max_newton=5)
    >>> _, gn = bopt.newton_cg(geom, oracle, batch, x0, max_newton=40, g0_reference='warm_start')
    >>> _, ex = bopt.newton_cg(geom, oracle, batch, x0, max_newton=40, g0_reference='warm_start',
    ...                        hessian='exact', cg_forcing_power=1.0)
    >>> ex['history'][0]['step_control'], ex['stop_reason'] in ('gradient', 'objective_floor')
    ('trust_region', True)
    >>> gn['newton_steps'] > 2 * ex['newton_steps']     # Gauss-Newton is linear here; exact has the Newton tail
    True
    """
    t_entry = time.perf_counter() if timing else None                   # timing: the run's first reading
    _require_unstacked(geom, x0, 'newton_cg')
    olsf = str(on_line_search_failure).lower()
    if olsf not in ('stop', 'accept'):
        raise ValueError("on_line_search_failure must be 'stop' or 'accept'; got %r" % (on_line_search_failure,))
    hessian = _check_hessian(hessian, 'newton_cg', _NEWTON_HESSIAN_MODES)
    switch = _check_switch_options(hessian, switch_tau, switch_k, switch_cg_forcing_power, switch_track)
    if hessian != 'gauss_newton':                                       # exact / switch: Kp and the members, at the call
        _require_exact_capable(oracle, geom, hessian)
    exact = hessian == 'exact'
    model_hessian = 'gauss_newton' if hessian == 'gauss_newton' else 'exact'   # 'switch' builds exact-mode models
    if curvature_damping is not None and model_hessian != 'exact':
        raise ValueError("newton_cg: curvature_damping is only meaningful with hessian='exact' or 'switch'")
    step_control = _check_step_control(step_control, hessian)
    trust_region = step_control == 'trust_region'
    if tr_radius0 is not None:
        if not trust_region:
            raise ValueError("newton_cg: tr_radius0 is only meaningful with step_control='trust_region'")
        if not (math.isfinite(float(tr_radius0)) and float(tr_radius0) > 0.0):
            raise ValueError("newton_cg: tr_radius0 must be a finite number > 0; got %r" % (tr_radius0,))
    if trust_region and olsf == 'accept':
        raise ValueError("newton_cg: on_line_search_failure='accept' has no effect with "
                         "step_control='trust_region' (there is no line search)")
    ref_selector = _check_g0_reference(g0_reference, 'newton_cg')
    _check_positive_reference('g0norm_newton', g0norm_newton)
    _check_positive_reference('g0norm_cg', g0norm_cg)
    if start_frame is not None and padded_from is not None:
        raise ValueError('newton_cg: pass start_frame or padded_from, not both (padded_from builds the start frame)')
    padded = dict(oversampling=padded_oversampling, oversampled_passes=padded_oversampled_passes,
                  max_passes=padded_max_passes, tol=padded_tol, alternate=padded_alternate, seed=padded_seed)
    unread = sorted('padded_' + k for k, v in padded.items() if v is not None)
    if unread and padded_from is None:
        raise ValueError('newton_cg: %s set without padded_from' % ', '.join(unread))
    padded_options = _alignment_options(padded, 'padded_', 'newton_cg')
    cone = ref_selector == 'cone' and g0norm_newton is None             # the cone reference is computed
    cone_options = _cone_options(dict(oversampling=cone_oversampling, oversampled_passes=cone_oversampled_passes,
                                      max_passes=cone_max_passes, tol=cone_tol, alternate=cone_alternate,
                                      seed=cone_seed), cone, 'newton_cg')
    x0, batch = _prepare_jit_inputs(use_jit, x0, batch)                 # auto-convert to jax when jitting
    clock = None                                                        # timing: the phase clock
    if timing:
        tree_block_until_ready((x0, batch))
        clock = _PhaseClock(_CompileSeconds.register(x0), t_entry)      # the compile listener, unregistered below
    try:
        extra = {}                                                      # stats keys of the new keywords only
        one_offs = dict(alignment_passes=None, reference_passes=None)   # timing: row 0's run-level fields
        start_centers = None                                            # a given start frame has none (D3)
        if padded_from is not None:
            _mark(clock)
            start_frame, start_centers, movements, one_offs['alignment_passes'] = _padded_start(
                geom, oracle, batch, x0, padded_from, padded_options, use_jit, timing)
            _lap(clock, 'alignment', (start_frame, start_centers, movements))
            extra.update(start_frame='padded', start_movements=_movement_record(movements))
        elif start_frame is not None:
            extra['start_frame'] = 'given'
        if switch is not None:
            tau, k, switch_power, track = switch
        switched = switch is not None and k == 0                        # switch mode: the exact phase (k = 0: from 0)
        switched_at = 0 if switched else None
        count = 0                                                       # switch mode: the consecutive hits so far
        x = x0
        lm = None                                                       # rebuilt only after an accepted step
        # D9: the step acceptance compares like with like. `scored` is the point-route objective (point_objective's
        # misfit, regularization and sum) of EXACTLY the current x: the start's (one query, in the reference lap),
        # then the accepted trial's, carried. A rejected trial keeps x and it; the exact rebuild of the switch is at
        # the same x. INVARIANT: any future step that changes x between iterations (e.g. a re-gauging) must
        # recompute it. The model's frame-route f stays LocalModel.misfit (the floor's dtype, the display).
        scored = None
        lm_track = None                                                 # switch mode, GN phase: the exact build
        newton_ref = cg_ref = None                                      # resolved once, at iteration 0
        g0_ref_kind = cg_ref_kind = None
        radius = radius0 = radius_max = None                            # trust region: set at the first step
        losses, newton_steps, newton_accepted, history, timings = [], 0, 0, [], []
        stop_reason = 'max_newton'
        t_prev = time.perf_counter()
        if clock is not None:                                           # the setup ends here
            seconds, compile_seconds = clock.close(t_prev)
            one_offs.update(setup=seconds['remainder'], alignment=seconds.get('alignment'),
                            setup_compile=None if compile_seconds is None else compile_seconds['remainder'],
                            alignment_compile=None if compile_seconds is None else compile_seconds.get('alignment'))
        for it in range(max_newton):
            if it > 0:
                _lap(clock, 'previous_callback')
            if lm is None:                                              # a new point: build its model there (fresh held)
                lm = _local_model_phases(geom, oracle, batch, x, regularizer, model_hessian, curvature_damping,
                                         start_frame, start_centers, clock)
                if switch is not None and not switched:                 # the GN phase: the exact build tracks, and
                    lm_track, lm = lm, _gauss_newton_view(lm)           #   CG, the step and the row use its GN view
                start_frame = start_centers = None                      # iteration 0 only
                _mark(clock)
                g = _model_gradient(lm if lm_track is None else lm_track)   # GN view: the exact build's g, read
                gnorm = float(geom.inner(g, g)) ** 0.5
                lift_sigma_min, lift_eps = _lift_diagnostics(lm)
                _lap(clock, 'gradient')
            if newton_ref is None:                                       # iteration 0: resolve the references
                _mark(clock)
                scored = _scored(geom, oracle, batch, x, regularizer)    # the start's point-route objective (D9)
                if ref_selector == 'cone':
                    ref, zero_ref = 'cone', None                         # no zero reference: the cone's (below)
                else:
                    ref, zero_ref = _resolve_g0_reference(geom, oracle, batch, lm.frame, ref_selector,
                                                          g0norm_newton is None, 'newton_cg',
                                                          "g0_reference='warm_start' or g0norm_newton")
                if g0norm_newton is not None:
                    newton_ref, g0_ref_kind = g0norm_newton, 'override'
                elif ref == 'zero':
                    newton_ref, g0_ref_kind = zero_ref, 'zero'
                elif ref == 'cone':
                    newton_ref, extra['cone_movements'], one_offs['reference_passes'] = _cone_reference(
                        geom, oracle, batch, x0, cone_options, use_jit, 'newton_cg',
                        "g0_reference='warm_start' or g0norm_newton",
                        timing)
                    g0_ref_kind = 'cone'
                else:
                    newton_ref, g0_ref_kind = (gnorm if gnorm > 0 else 1.0), 'warm_start'  # the initial ‖g‖
                if g0norm_cg is not None:
                    cg_ref, cg_ref_kind = g0norm_cg, 'override'
                else:
                    cg_ref, cg_ref_kind = newton_ref, g0_ref_kind        # CG inherits the Newton reference
                _lap(clock, 'reference')
            misfit, reg_val, f = scored                                  # None regularization: no regularizer attached
            losses.append(f)
            row_hessian = hessian if switch is None else ('exact' if switched else 'gauss_newton')   # this row's model
            run_fields = dict(g0_reference=g0_ref_kind, cg_reference=cg_ref_kind, step_control=step_control,
                              hessian=row_hessian, lift_sigma_min=lift_sigma_min, lift_eps=lift_eps)
            if gnorm <= gtol_rel * newton_ref:                           # converged -- final line, no step
                stop_reason = 'gradient'
                row_timing = None
                if clock is not None:                                    # its window (it has no wall_time)
                    row_timing = clock.row(time.perf_counter(), None, None, **(one_offs if it == 0 else {}))
                    timings.append(row_timing)
                switch_fields = {} if switch is None else dict(switch_count=None if switched else count,
                                                               switch_fired=False)
                info = NewtonInfo(iteration=it, objective=f, gnorm=gnorm, g0norm=newton_ref, converged=True,
                                  x_cores=x, lm=lm, misfit=misfit, regularization=reg_val,
                                  stop_reason=stop_reason, timing=row_timing, **run_fields, **switch_fields)
                if callback is not None:
                    callback(info)
                history.append(_newton_scalar_record(info))
                break
            newton_steps += 1
            power = switch_power if switched else cg_forcing_power      # switch mode: the exact phase's power
            eta = min(0.5, (gnorm / cg_ref) ** power)                    # inexact-Newton (Eisenstat-Walker) forcing term
            cg_tol = eta * gnorm
            neg_g = cw.corewise_scale(g, -1.0)
            if trust_region and radius is None:                          # the first step: the initial radius
                _mark(clock)
                radius0 = float(tr_radius0) if tr_radius0 is not None else _initial_radius(lm, g, gnorm)
                radius, radius_max = radius0, _TR_RADIUS_MAX_REL * radius0
                _lap(clock, 'forms')
            tr_radius = radius if trust_region else None                 # the radius THIS trial is bounded by
            _mark(clock)
            p, cg_i, cg_rs, cg_ok, cg_hit = _cg_solve(lm, neg_g, cg_tol, cg_maxiter, use_jit,
                                                      radius if trust_region else math.inf)
            _lap(clock, 'cg', (p, cg_i, cg_rs, cg_ok, cg_hit))
            cg_iters, cg_rs, cg_ok, cg_boundary = int(cg_i), float(cg_rs), bool(cg_ok), bool(cg_hit)
            cg_converged = (cg_rs <= cg_tol * cg_tol) and not cg_boundary
            cg_truncated = (not cg_converged) and (not cg_ok) and (not cg_boundary)
            on_boundary = cg_boundary                                    # the step was limited by Δ
            _mark(clock)
            slope = float(geom.inner(g, p))
            fallback = (not math.isfinite(slope)) or slope >= 0.0
            if fallback:                                                 # ensure a descent direction
                if trust_region:                                         #   ... inside the region
                    scale = min(1.0, radius / gnorm)
                    p, slope, on_boundary = cw.corewise_scale(g, -scale), -scale * gnorm * gnorm, scale < 1.0
                else:
                    p, slope = neg_g, -gnorm * gnorm
            if exact or switched:                                        # pᵀHp = GN part + curvature part (once each)
                pKp = lm.curvature_quadratic(p)
                pHp = float(lm.gn_quadratic(p) + pKp)                    # == lm.quadratic(p)
                pKp = float(pKp)
            else:
                pHp, pKp = float(lm.quadratic(p)), None                  # pᵀHp = ‖𝒥p‖² (+ reg) (for ρ)
            p_norm = float(geom.inner(p, p)) ** 0.5
            _lap(clock, 'forms', p)
            stop_now = False
            if not trust_region:                                         # ---- Armijo line search
                alpha, ls_steps, f_new = 1.0, 40, f
                x_trial = x
                ls_failed = True
                for bt in range(40):                                     # Armijo backtracking
                    _mark(clock)
                    x_trial = lm.retract(cw.corewise_scale(p, alpha))
                    retraction = _lap(clock, 'trial_retraction', x_trial)
                    trial_scored = _scored(geom, oracle, batch, x_trial, regularizer)            # policy a
                    f_new = trial_scored[2]
                    _lap(clock, 'trial_objective')
                    if f_new <= f + c_armijo * alpha * slope:
                        ls_steps, ls_failed = bt, False
                        break
                    if bt < 39:
                        alpha *= 0.5    # halve BETWEEN trials only: alpha stays the last trial's step
                trials = bt + 1
                delta_f = f_new - f
                predicted = alpha * slope + 0.5 * alpha * alpha * pHp    # the model's change along αp
                rho = (delta_f / predicted) if predicted != 0.0 else float('nan')
                accepted = not (ls_failed and olsf == 'stop')            # 'stop' rejects: x unchanged
                if not accepted:
                    stop_reason, stop_now = 'line_search', True
                tr_accepted = None
            else:                                                        # ---- trust region: one trial at α = 1
                alpha, ls_steps, ls_failed = 1.0, None, None
                _mark(clock)
                x_trial = lm.retract(p)
                retraction = _lap(clock, 'trial_retraction', x_trial)
                trial_scored = _scored(geom, oracle, batch, x_trial, regularizer)                # policy a
                f_new = trial_scored[2]
                _lap(clock, 'trial_objective')
                trials = 1
                delta_f = f_new - f
                predicted = slope + 0.5 * pHp                            # the model's change along p (< 0 expected)
                rho = (delta_f / predicted) if predicted < 0.0 else float('nan')
                accepted = tr_accepted = math.isfinite(rho) and rho > _TR_ACCEPT
                if (not math.isfinite(rho)) or rho < _TR_SHRINK_BELOW:
                    radius = _TR_SHRINK * p_norm
                elif rho > _TR_EXPAND_ABOVE and on_boundary:
                    radius = min(_TR_EXPAND * radius, radius_max)
                if not accepted:                                         # the floors: on a rejection ONLY
                    resolution = _objective_resolution(lm, reg_val)
                    if abs(predicted) <= _TR_RESOLUTION_FACTOR * resolution:
                        stop_reason, stop_now = 'objective_floor', True  # the model change is below f's resolution
                    elif radius < _TR_RADIUS_FLOOR_REL * radius0:
                        stop_reason, stop_now = 'trust_region_floor', True
            if switch is None:                                           # switch mode: this row's switch_* fields
                switch_fields = {}
            elif switched:                                               # the exact phase tracks nothing
                switch_fields = dict(switch_fired=False)
            else:                                                        # the GN phase: track, count, maybe switch
                countable = accepted and not fallback and (tr_accepted if trust_region
                                                           else (ls_steps == 0 and not ls_failed))
                e_gn, e_n, pKp_track, hit = abs(delta_f - predicted), None, None, False
                if countable or track == 'all':                          # the tracked rows: one curvature product
                    _mark(clock)
                    pKp_track = float(lm_track.curvature_quadratic(p))
                    _lap(clock, 'tracking')
                    floor = _SWITCH_ROUNDING * _TR_RESOLUTION_FACTOR * _objective_resolution(lm, reg_val)
                    e_gn, e_n, _, hit = _switch_sample(delta_f, predicted, 0.5 * alpha * alpha * pKp_track, floor, tau)
                count = count + 1 if (countable and hit) else 0          # a non-countable row resets, whatever e_N
                if count >= k:                                           # for good: the next point is built exact
                    switched, switched_at = True, it + 1
                switch_fields = dict(switch_pKp=pKp_track, switch_e_gn=e_gn, switch_e_n=e_n, switch_count=count,
                                     switch_fired=switched)
            x_norm = float(cw.corewise_norm(x))                         # coordinate norm (ragged-exact; uniform: D6)
            step_rel = (alpha * p_norm / x_norm) if x_norm > 0 else float('nan')
            last_row = stop_now or it == max_newton - 1
            t_now = time.perf_counter()
            row_timing = None
            if clock is not None:
                row_timing = clock.row(t_now, trials, retraction if accepted else None,
                                       **(one_offs if it == 0 else {}))
                timings.append(row_timing)
            info = NewtonInfo(iteration=it, objective=f, gnorm=gnorm, g0norm=newton_ref, converged=False,
                              x_cores=x, lm=lm, misfit=misfit, regularization=reg_val,
                              forcing_eta=eta, cg_tol=cg_tol, cg_iters=cg_iters,
                              cg_maxiter=cg_maxiter,
                              cg_resid=cg_rs ** 0.5, cg_converged=cg_converged, cg_truncated=cg_truncated,
                              ls_steps=ls_steps, ls_failed=ls_failed, alpha=alpha, slope=slope, pHp=pHp,
                              delta_f=delta_f, rho=rho, step_rel=step_rel, wall_time=t_now - t_prev,
                              stop_reason=stop_reason if last_row else None, tr_radius=tr_radius,
                              tr_accepted=tr_accepted, cg_boundary=cg_boundary, pKp=pKp, timing=row_timing,
                              **run_fields, **switch_fields)
            t_prev = t_now
            if callback is not None:
                callback(info)
            history.append(_newton_scalar_record(info))
            if stop_now:
                break
            if accepted:
                newton_accepted += 1
                x, lm, lm_track = x_trial, None, None                    # the next row linearizes at the new point
                scored = trial_scored                                    # ... and carries its scored value (D9)
        if switch is not None:
            extra.update(hessian='switch', switch_track=track, switched_at=switched_at)
        if clock is not None:                                            # the last row's callback ends here
            t_exit = time.perf_counter()
            extra['timing'] = _timing_stats(timings, history, one_offs, t_exit - t_entry, clock.close(t_exit))
    finally:
        if clock is not None and clock.listener is not None:
            clock.listener.unregister()
    return x, {'losses': losses, 'newton_steps': newton_steps, 'newton_accepted': newton_accepted,
               'stop_reason': stop_reason, 'history': history, **extra}


if jax_available:
    import jax

    # Register LocalModel as a jax pytree. The DATA are leaves (the batch, the frame, the centers, held, f, the
    # per-frame geometry companion, the curvature state); the STATICS are aux (geometry, oracle, regularizer).
    #
    # This is what makes the jit cache key value-based: geometry and oracle are frozen dataclasses whose parameters
    # are FIELDS, so a model rebuilt at each Newton iteration -- which the outer loop does, every iteration, by
    # construction -- hashes and compares EQUAL to the previous one, and `jit(_cg_core)` compiles once per fit. The
    # regularizers are frozen dataclasses of floats, hence hashable by value already. Which slots `held` fills is its
    # own pytree structure (HeldVariables), so it is static too.
    # The data tuples are partitioned rather than handed over whole: a backend frame is
    # `(supercores..., shape, masks)`, so flattening it naively would TRACE the masks and the shape ints -- which
    # `require_concrete_masks` rejects outright, and which would break the host-integer shape arithmetic the uniform
    # layer runs on. `partition_static` keeps them in the aux, exactly as the frontend `UT3Frame` keeps its masks in
    # aux. A registered record among the leaves (the batch, held) is flattened by jax itself.
    #
    # The exact-mode companion rides in the same tuple: its arrays are leaves, and its PRESENCE (curvature is
    # None or not; geometry_data None or not) is tree structure, so the mode is static under the treedef --
    # there is no mode aux to carry.
    def _flatten_local_model(m):
        dyn, skel = partition_static((m.batch, m.frame, m.centers, m.held, m.f, m.geom_aux, m.curvature))
        return dyn, (m.geom, m.oracle, m.regularizer, skel)

    def _unflatten_local_model(aux, children):
        batch, frame, centers, held, f, geom_aux, curvature = rebuild_static(children, aux[3])
        return LocalModel(aux[0], aux[1], batch, frame, centers, held, f, aux[2], geom_aux, curvature)

    def _flatten_curvature_state(c):
        return partition_static((c.misfit_gradient, c.geometry_data))

    def _unflatten_curvature_state(skeleton, children):
        return CurvatureState(*rebuild_static(children, skeleton))

    jax.tree_util.register_pytree_node(LocalModel, _flatten_local_model, _unflatten_local_model)
    jax.tree_util.register_pytree_node(CurvatureState, _flatten_curvature_state, _unflatten_curvature_state)
