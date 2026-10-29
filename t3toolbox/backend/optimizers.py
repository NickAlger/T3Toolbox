# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Backend-first optimization algorithms for fixed-rank T3 least-squares fitting.

These operate on **raw cores / tangent tuples** via backend functions only -- no frontend imports
(no ``TuckerTensorTrain`` / ``T3Tangent`` / ``fitting.LocalModel`` / geometry classes). A raw-``.data``
user can call them directly; the frontend ``optimizers.py`` is a thin adapter that validates the input
once and assembles the ``Problem`` oracle from the same backend functions (design:
``docs/contributor/fitting_internals.md`` §"Backend-first").

Because the numerical safety preconditions live only in the frontend, this layer is **check-free** of
them (structural shape guards inside the backend functions remain, and are jit-safe). So ``jit`` needs no
``unsafe()`` wrapping; the ``use_jit`` machinery is a thin jax-only layer on top of the ordinary
numpy/jax dispatch (the per-step kernels jit via ``_jitted``; the inner CG runs as a
``lax.while_loop`` -- eager-vs-jit agreement is covered in ``tests/backend/test_optimizers.py``).

The oracle:

* ``Problem``     -- (geometry ops, sampling kind, sample, data); builds per-point ``LocalModel``
  objects and retracts.
* ``LocalModel``  -- the local model linearized at a point (``.gradient`` / ``.objective`` /
  ``.hvp`` / ``.gn_quadratic`` / ``.retract``; Gauss-Newton by default, the exact Riemannian Hessian with
  ``Problem.local_model(..., hessian='exact')``), the backend twin of the frontend ``fitting.LocalModel``.
* ``Geometry``    -- the *where you optimize* axis: frame / project / retract / inner on raw data.
  This module declares only the **protocol** it needs; the T3 implementations live in
  :py:mod:`t3toolbox.backend.geometry`, so nothing here depends on the T3 representation.

Tangent vectors are raw ``(tucker_var, tt_var)`` tuples; vector arithmetic is the ``corewise`` ops.
"""
import dataclasses as dc
import functools as ft
import math
import time
import typing as typ
import warnings

import numpy as np

from t3toolbox.backend.common import *
from t3toolbox.backend.regularization import Regularizer, IdentityRegularizer   # re-exported for backend users
from t3toolbox.backend.regularization import _ScaledRegularizer                 # internal: minibatch reg scaling
from t3toolbox.backend.regularization import require_unstacked_for_regularizer
import t3toolbox.corewise as cw

__all__ = [
    'Problem',
    'LocalModel',
    'CurvatureState',
    'NewtonInfo',
    'Regularizer',
    'IdentityRegularizer',
    'least_squares_problem',
    'flat_draw',
    'gradient_descent',
    'mc_sgd',
    'adam',
    'newton_cg',
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

    **Optional member** (looked up duck-typed, ``getattr(geom, 'zero_point_frame', None)``; a geometry
    without it keeps working):

    * ``zero_point_frame(frame)`` -- the canonical frame at the zero tensor with ``frame``'s structure, or
      ``None`` if this geometry has none. It is where ``newton_cg`` / ``gradient_descent`` evaluate the
      default Newton reference ``‖g(0)‖`` (``g0_reference``). The manifold geometries define one; the
      corewise geometries return ``None`` (the zero cores are a critical point there); a geometry without
      the member, or one returning ``None``, gets the warm-start reference by default.

    **The exact-mode members** -- needed **only** for ``hessian='exact'``.
    ``Problem.local_model(hessian='exact')`` checks that all four exist (a structural ``ValueError`` naming
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

    def frame(self, x_cores):                          # x_cores -> the linearization frame
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



# --------------------------------------------------------------------------------------------------
# The problem oracle
# --------------------------------------------------------------------------------------------------
_HESSIAN_MODES = ('gauss_newton', 'exact')
_EXACT_GEOMETRY_MEMBERS = ('curvature_precompute', 'lift', 'lift_transpose', 'tangential_curvature')


def _check_hessian(
        hessian:  str,          # 'gauss_newton' | 'exact' (case-insensitive)
        who:      str,          # the caller, for the message
) -> str:                       # the lower-cased mode
    """Validate a ``hessian`` mode selector (structural: anything else raises, a ``bool`` included -- the
    keyword is the solver's string mode everywhere, as in ``sampling_derivatives.estimate_chunk_size``)."""
    if not isinstance(hessian, str) or hessian.lower() not in _HESSIAN_MODES:
        raise ValueError("%s: hessian must be 'gauss_newton' or 'exact' (case-insensitive); got %r"
                         % (who, hessian))
    return hessian.lower()


def _require_exact_capable(
        kind,       # the problem's SamplingKind
        geom,       # the problem's Geometry
) -> None:
    """The structural checks of ``hessian='exact'``, run before any work: the kind implements the sixth
    operation ``corewise_hessian`` and the geometry has the four exact-mode members.

    "Implements" means the method the kind would run is not an unimplemented stub -- one carrying the stdlib
    :py:func:`abc.abstractmethod` marker, as the ``SamplingKind`` base method (or a stub a subclass re-declares) does.
    Reading the marker keeps this module free of T3-specific imports; for a kind that does not re-declare a
    marked stub it agrees exactly with ``type(kind).corewise_hessian is SamplingKind.corewise_hessian``."""
    method = getattr(type(kind), 'corewise_hessian', None)
    if method is None or getattr(method, '__isabstractmethod__', False):
        raise ValueError("%s does not implement corewise_hessian(), which hessian='exact' needs; it still "
                         "works with hessian='gauss_newton'" % type(kind).__name__)
    missing = [name for name in _EXACT_GEOMETRY_MEMBERS if not callable(getattr(geom, name, None))]
    if missing:
        raise ValueError("%s does not implement %s, which hessian='exact' needs (the exact-mode geometry "
                         "members: %s); it still works with hessian='gauss_newton'"
                         % (type(geom).__name__, ', '.join(missing), ', '.join(_EXACT_GEOMETRY_MEMBERS)))


@dc.dataclass(frozen=True, eq=False)   # eq=False -> identity hash/eq: array fields (a leaf container)
class CurvatureState:
    """The exact-mode companion of a :py:class:`~t3toolbox.backend.optimizers.LocalModel` (``hessian='exact'``):
    a pytree LEAF beside ``sweep`` / ``geom_aux``, built once per model by
    :py:meth:`~t3toolbox.backend.optimizers.Problem.local_model`.

    ``misfit_gradient`` is the misfit's Riemannian gradient ``Pi J^T omega^2 r`` -- computed ONCE per model
    and read by both :py:attr:`~t3toolbox.backend.optimizers.LocalModel.gradient` and the geometry's curvature
    data, so exact mode costs no second transpose. ``geometry_data`` is ``geom.curvature_precompute(...)``;
    ``None`` means the curvature term vanishes at this frame (the manifold at ``d = 1``), and ``hvp`` is then
    the Gauss-Newton one."""
    misfit_gradient:  typ.Any   # Pi J^T omega^2 r (no regularizer): the model's one misfit transpose
    geometry_data:    typ.Any   # geom.curvature_precompute(frame, misfit_gradient, aux, curvature_damping)


@dc.dataclass(frozen=True)
class LocalModel:
    """The local model linearized at a point -- the backend twin of the frontend
    :py:class:`t3toolbox.fitting.LocalModel` (same name, one level down).
    Built by ``Problem.local_model``; the frame sweep is computed once and reused by every method below.

    **Two modes**, derived rather than stored (:py:attr:`is_exact` is ``curvature is not None``):

    * **Gauss-Newton** (``hessian='gauss_newton'``, the default): ``hvp(p) = Pi J^T omega^2 J Pi p (+ H_R p)``.
    * **Exact** (``hessian='exact'``): the exact Riemannian Hessian,

      ``hvp(p) = Pi [J^T omega^2 J Pi p + L^T (K_{A^T omega^2 r} - K_{grad f}) L Pi p] (+ H_R p)``,

      with ``L`` the geometry's lift from coordinates to raw core variations, ``K_{A^T omega^2 r}`` the
      kind's :py:meth:`~t3toolbox.backend.fitting.SamplingKind.corewise_hessian` (the derivative of the
      corewise transpose at the fixed residual) and ``K_{grad f}`` the geometry's tangential curvature
      (Absil, Mahony & Trumpf (2013), "An extrinsic look at the Riemannian Hessian": the Weingarten term as
      ``<N, D^2 Phi>`` with ``N`` the normal part of ``A^T omega^2 r``). On a corewise geometry it is the
      plain Hessian of ``f o Phi`` (no lift, no tangential term).

    ``gn_quadratic`` / ``jacobian`` / ``retract`` mean the same in both modes (``gn_quadratic`` is always
    ``‖ω J Pi p‖² (+ <p, H_R p>)``); ``quadratic`` is the mode's model ``<p, H p>``."""
    geom:     typ.Any        # a Geometry (backend.geometry.*GeometryOps); aux_data, so it must hash by VALUE
    kind:     typ.Any        # backend fitting.SamplingKind (forward / transpose / sumsq / w_axes)
    sample:   typ.Any        # ww (probe/apply) or index (entries)
    frame:     typ.Tuple      # (U, O, P, Q)
    sweep:    typ.Tuple      # precomputed frame sweep (reused across gradient/jacobian/hvp)
    residual: typ.Any        # r = S(x) - data
    n_w:      int            # number of leading sample-stack axes

    regularizer: typ.Any = None   # optional backend.regularization.Regularizer; ρ folded into obj/grad/gn_quadratic/hvp
    geom_aux:    typ.Any = None   # = geom.precompute(frame) (per-frame companion, e.g. SF-T3); a leaf, like sweep
    curvature:   typ.Any = None   # CurvatureState in exact mode, else None; a leaf, like sweep

    @property
    def is_exact(self) -> bool:                          # exact mode (hessian='exact'): derived, never stored
        return self.curvature is not None

    @property
    def _has_curvature_term(self) -> bool:               # exact AND the term does not vanish at this frame
        return self.curvature is not None and self.curvature.geometry_data is not None

    def _require_exact(self, who: str) -> None:
        if self.curvature is None:
            raise ValueError("LocalModel.%s needs an exact-mode model (Problem.local_model(..., "
                             "hessian='exact')); this one is Gauss-Newton" % who)

    @property
    def misfit(self):                                    # ½‖ω⊙r‖²  (the weighted data misfit alone)
        return 0.5 * self.kind.sumsq(self.residual, self.n_w)

    @property
    def regularization(self):                            # ρ(X) at X = geom.base_point(frame); 0 if unregularized
        if self.regularizer is None:
            return 0.0
        return self.regularizer.value(self.geom, self.geom.base_point(self.frame))

    @property
    def objective(self):                                 # c = ½‖ω⊙r‖²  (+ ρ(X) if regularized)
        return self.misfit + self.regularization

    @property
    def gradient(self) -> Tangent:                       # g = Π 𝒥ᵀ r  (+ g_R)
        if self.curvature is not None:                   # exact: the model's one misfit transpose, already done
            g = self.curvature.misfit_gradient           #   (by the same expression -- bit-identical to GN's)
        else:
            g = self.geom.project(self.frame, self.kind.transpose(self.residual, self.sample, self.frame,
                                                                  self.sweep), aux=self.geom_aux)
        if self.regularizer is not None:
            g = cw.corewise_add(g, self.regularizer.gradient(self.geom, self.frame, aux=self.geom_aux))
        return g

    def jacobian(self, p: Tangent):                      # 𝒥 Π p
        return self.kind.forward(self.geom.project(self.frame, p, aux=self.geom_aux),
                                 self.sample, self.frame, self.sweep)

    def gn_quadratic(self, p: Tangent):                  # ‖𝒥 Π p‖²  (+ ⟨p, H_R p⟩) -- the Cauchy denominator
        q = self.kind.sumsq(self.jacobian(p), self.n_w)
        if self.regularizer is not None:
            q = q + self.regularizer.quadratic(self.geom, self.frame, p, aux=self.geom_aux)
        return q

    def hvp(self, p: Tangent) -> Tangent:                # GN: H p = Π 𝒥ᵀ 𝒥 Π p  (+ H_R p)
        if not self._has_curvature_term:                 # exact: Π [𝒥ᵀ𝒥 Πp + Lᵀ K_N L Πp]  (+ H_R p)
            z = self.jacobian(p)
            Hp = self.geom.project(self.frame, self.kind.transpose(z, self.sample, self.frame, self.sweep),
                                   aux=self.geom_aux)
        else:
            pp = self.geom.project(self.frame, p, aux=self.geom_aux)
            z = self.kind.forward(pp, self.sample, self.frame, self.sweep)
            y = self._curvature_block(pp)
            Hp = self.geom.project(self.frame, cw.corewise_add(
                self.kind.transpose(z, self.sample, self.frame, self.sweep), y), aux=self.geom_aux)
        if self.regularizer is not None:
            Hp = cw.corewise_add(Hp, self.regularizer.hessian(self.geom, self.frame, p, aux=self.geom_aux))
        return Hp

    def _curvature_block(self, pp: Tangent) -> Tangent:  # Lᵀ (K_{Aᵀω²r} − K_{grad f}) L pp -- UNGAUGED, K+C
        geom, data = self.geom, self.curvature.geometry_data
        theta_hat = geom.lift(self.frame, pp, data)                               # raw core directions
        K = self.kind.corewise_hessian(theta_hat, self.residual, self.sample, self.frame, self.sweep,
                                       geom.base_point(self.frame))               # K_{Aᵀω²r} θ̂ (ω² inside)
        K_grad = geom.tangential_curvature(self.frame, theta_hat, data)           # K_{grad f} θ̂, or None
        if K_grad is not None:
            K = cw.corewise_sub(K, K_grad)
        return geom.lift_transpose(self.frame, K, data)

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


@dc.dataclass(frozen=True)
class Problem:
    """A fixed-rank least-squares fitting problem ``min_x ½‖S(x) - data‖²`` for a sampling op ``S`` on
    geometry ``geom``. **Layout-agnostic**: it holds the operator (``kind``), the geometry, and the FULL
    ``(sample, data)``. ``local_model`` / ``objective`` linearize / evaluate at a point on the full data,
    or on an explicitly-passed minibatch ``(sample, data)`` (e.g. from a ``draw``). The ``Problem`` itself
    owns **no** minibatch-layout logic -- where the sample stack ``W`` lives, how to slice it -- that is
    the ``kind``'s (``kind.take`` for the default flat draw) or the user's ``draw``."""
    geom:   typ.Any       # a Geometry (see the Geometry protocol above)
    kind:   typ.Any       # backend fitting.SamplingKind
    sample: typ.Any       # the FULL sample (ww / index / (ww,pp) / (index,pp))
    data:   typ.Any       # the FULL observed data S(x_true) (+ noise)

    regularizer: typ.Any = None   # optional backend.regularization.Regularizer; ρ folded into local_model + objective

    def _sample_and_data(self, sample, data, where: str):
        """The full ``(sample, data)`` when neither is given, the explicit pair when both are; exactly one
        is a structural error (``data=`` alone used to be silently dropped in favour of the training data)."""
        if sample is None and data is None:
            return self.sample, self.data
        if sample is None or data is None:
            raise ValueError('%s: pass `sample` and `data` together (an explicit minibatch), or neither '
                             '(the full training data); got sample=%s, data=%s'
                             % (where, 'None' if sample is None else 'given', 'None' if data is None else 'given'))
        return sample, data

    def local_model(
            self,
            x_cores:            Cores,                   # the point to linearize at
            sample:             typ.Any = None,          # an explicit minibatch sample (with data), or None
            data:               typ.Any = None,          # its observed data (with sample), or None
            hessian:            str = 'gauss_newton',    # or 'exact' (case-insensitive): the model's Hessian
            curvature_damping:  typ.Optional[float] = None,  # exact only: the lift damping; None -> the geometry's
    ) -> LocalModel:
        """Linearize at ``x_cores`` on the full data (``sample=None``) or an explicit minibatch
        (``sample`` AND ``data`` -- passing one without the other is a structural error).

        ``hessian='exact'`` builds the exact-mode model (:py:class:`~t3toolbox.backend.optimizers.LocalModel`): it
        checks, structurally and before any work, that the kind implements ``corewise_hessian`` and the geometry
        the four exact-mode members (see :py:class:`~t3toolbox.backend.optimizers.Geometry`), then does the
        model's ONE misfit transpose and the geometry's per-frame curvature data. ``curvature_damping``
        (relative to ``‖X‖_HS``) is handed to the geometry unread; ``None`` means the geometry's own default,
        and a value with ``hessian='gauss_newton'`` -- where nothing would read it -- raises.

        Examples
        --------
        The exact model's quadratic form is the objective's second derivative along the retraction (the T3-SVD
        retraction is second order on a minimal, numerically full-rank frame, as here), where the Gauss-Newton one
        misses the curvature term:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> import t3toolbox.backend.fitting as bfit
        >>> import t3toolbox.backend.geometry as bgeo
        >>> import t3toolbox.backend.optimizers as bopt
        >>> import t3toolbox.backend.sampling_derivatives as sd
        >>> np.random.seed(0)
        >>> shape, ranks = (4, 5, 6), ((2, 2, 2), (1, 2, 2, 1))
        >>> ww = [np.random.randn(20, N) for N in shape]
        >>> pp = [np.random.randn(20, N) for N in shape]
        >>> data = sd.t3_probe_derivatives(ww, pp, t3.TuckerTensorTrain.randn(shape, *ranks).data, 1)
        >>> problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.ProbeDerivativesKind(order=1),
        ...                                      (ww, pp), data)
        >>> x = t3.TuckerTensorTrain.randn(shape, *ranks).data
        >>> gn, exact = problem.local_model(x), problem.local_model(x, hessian='exact')
        >>> gn.is_exact, exact.is_exact
        (False, True)
        >>> g = exact.gradient                                         # the Gauss-Newton gradient, bit for bit
        >>> p = cw.corewise_scale(g, 1.0 / float(problem.geom.inner(g, g)) ** 0.5)
        >>> f = lambda t: float(problem.objective(exact.retract(cw.corewise_scale(p, t))))
        >>> fd = (f(1e-3) - 2 * f(0.0) + f(-1e-3)) / 1e-6
        >>> bool(abs(float(exact.quadratic(p)) - fd) < 1e-4 * abs(fd))
        True
        >>> bool(abs(float(gn.quadratic(p)) - fd) < 1e-2 * abs(fd))
        False
        """
        exact = _check_hessian(hessian, 'Problem.local_model') == 'exact'
        if curvature_damping is not None and not exact:
            raise ValueError("Problem.local_model: curvature_damping is only meaningful with hessian='exact'")
        if exact:
            _require_exact_capable(self.kind, self.geom)
        sample, data = self._sample_and_data(sample, data, 'Problem.local_model')
        if self.regularizer is not None:
            require_unstacked_for_regularizer(self.geom.stack_shape(x_cores), 'Problem.local_model')
        frame = self.geom.frame(x_cores)
        sweep = self.kind.precompute(frame, sample)
        geom_aux = self.geom.precompute(frame)        # None when the geometry has no companion
        residual = cw.corewise_sub(self.kind.point_forward(x_cores, sample), data)
        curvature = None
        if exact:                                     # the one misfit transpose (LocalModel.gradient reads it)
            misfit_gradient = self.geom.project(frame, self.kind.transpose(residual, sample, frame, sweep),
                                                aux=geom_aux)
            curvature = CurvatureState(misfit_gradient, self.geom.curvature_precompute(
                frame, misfit_gradient, geom_aux, curvature_damping))
        return LocalModel(self.geom, self.kind, sample, frame, sweep, residual, self.kind.w_axes(sample),
                          self.regularizer, geom_aux, curvature)

    def objective(self, x_cores: Tangent, sample=None, data=None):
        """``½‖S(x)-data‖²`` (+ ``ρ(x)`` if regularized) on the full data (``sample=None``) or an explicit
        minibatch; no frame sweep (cheap -- for the line search / the full-batch stop signal). The reg term
        reads ``x_cores`` directly through ``geom.point_norm_sq`` (exact for any point;
        ``docs/contributor/fitting_internals.md`` §"The base point as a tangent"). ``sample`` and ``data`` go together: one without the
        other is a structural error (the data used to be silently ignored when only it was passed)."""
        sample, data = self._sample_and_data(sample, data, 'Problem.objective')
        if self.regularizer is not None:
            require_unstacked_for_regularizer(self.geom.stack_shape(x_cores), 'Problem.objective')
        residual = cw.corewise_sub(self.kind.point_forward(x_cores, sample), data)
        c = 0.5 * self.kind.sumsq(residual, self.kind.w_axes(sample))
        if self.regularizer is not None:
            c = c + self.regularizer.value(self.geom, x_cores)
        return c


def least_squares_problem(
        geom:        typ.Any,       # a Geometry (backend.geometry.{Manifold,Corewise}GeometryOps, or uniform)
        kind:        typ.Any,       # backend fitting.{APPLY,ENTRIES,PROBE} or a derivative kind
        sample:      typ.Any,       # ww / index / (ww,pp) / (index,pp)
        data:        typ.Any,       # observed values
        regularizer: typ.Any = None,  # optional backend.regularization.Regularizer (e.g. IdentityRegularizer(λ))
) -> Problem:
    """Assemble a least-squares ``Problem`` from a geometry, a sampling kind, the sample, and the observed
    data, plus an optional ``regularizer`` (adds ``ρ(x)`` to the objective, folded into the local GN model).
    (The frontend adapter calls this with the same backend objects a raw-data user would.)"""
    return Problem(geom, kind, sample, data, regularizer)


def flat_draw(
        problem: Problem,
        batch:   int,        # measurements per minibatch
) -> typ.Callable:           # draw(rng) -> (sample_B, data_B)
    """The **default** minibatch draw: a uniform random subset of ``batch`` measurements across the whole
    (flattened) sample stack ``W`` -- the robust default. Returns a ``draw(rng) -> (sample_B, data_B)``
    over ``problem``'s full data, via the kind's ``take``. A user may pass **any** ``draw`` instead (slice
    X / P / order, importance-sample, ...). Host numpy by default; pass jax data + a jax draw to keep the
    minibatch on device (the optimizer never compiles the draw -- only the per-step kernel)."""
    n = problem.kind.n_measurements(problem.sample)

    def draw(rng):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        return problem.kind.take(problem.sample, problem.data, idx)
    return draw


def _minibatch_step_problem(
        problem: Problem,
        batch:   int,        # nominal minibatch size (the ``batch`` arg; = the default flat draw's size)
) -> Problem:
    """The problem the **stochastic** per-step kernel linearizes against: identical to ``problem`` but with
    the regularizer scaled by ``batch/n``. The minibatch data gradient is a ``batch/n`` estimate of the
    full-data gradient, so scaling the (deterministic) regularizer to match keeps ``λ``'s meaning the same
    as in the full-batch optimizers (``docs/contributor/fitting_internals.md`` §"Stochastic scaling"). No-op when unregularized. The
    **full-batch** stop/loss keeps the full-strength ``problem.regularizer``. The factor is ``batch/n`` for
    the NOMINAL ``batch`` -- ``batch`` is not ignored when a custom ``draw`` is given: a draw of another size
    gets a mismatched factor, so pass ``batch`` equal to the size your ``draw`` produces."""
    if problem.regularizer is None:
        return problem
    n = problem.kind.n_measurements(problem.sample)
    return dc.replace(problem, regularizer=_ScaledRegularizer(problem.regularizer, min(batch, n) / n))


# --------------------------------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------------------------------
def _require_unstacked(
        geom,             # the problem's geometry (supplies stack_shape)
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
    -- :py:meth:`Problem.local_model` and the frontend ``fitting.LocalModel`` DO support a stacked point
    and return a shape-``C`` objective, so rolling your own loop over a stacked problem works. It is only
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
            "drive your own loop from Problem.local_model / fitting.LocalModel, which do support a "
            "stacked point." % (who, stack))


def _prepare_jit_inputs(use_jit, x0, problem):
    """When ``use_jit``, move the optimizer inputs onto jax so the jit path actually engages -- the
    **auto-convert** contract: asking for jit is opting into "jax world" (device residency + jax's default
    float32), so we ``jnp.asarray`` ``x0`` and the problem's ``sample`` / ``data`` rather than silently
    dropping the request. ``geom`` / ``kind`` (and any uniform masks / residual weight they hold) are left
    untouched -- masks stay host-numpy per the uniform jit recipe, and a numpy weight folds in as a
    constant. Returns ``(x0, problem)`` unchanged when ``use_jit`` is False.

    When ``use_jit`` but jax is unavailable: runs eager on numpy with a one-time warning (the library-wide
    jax-absent policy, ``common._jax_or_warn`` -- a script developed without jax and deployed with it must
    run unchanged on both machines). Loud, not silent: the 2026.1.0 silent eager drop is what this replaces."""
    if not use_jit:
        return x0, problem
    if not jax_or_warn('use_jit=True'):
        return x0, problem
    return (tree_to_jax(x0),
            dc.replace(problem, sample=tree_to_jax(problem.sample), data=tree_to_jax(problem.data)))


def _jit_engaged(use_jit, x0, problem) -> bool:
    """Whether the jit path actually engages: asked for, jax present, and the inputs (x0, sample, data)
    are **all** jax so the minibatch gather + kernel run on device. Under ``use_jit`` the inputs are
    guaranteed jax by :py:func:`_prepare_jit_inputs`; this stays a check because a raw-``.data`` caller
    may reach these functions directly."""
    return bool(use_jit and jax_available and tree_contains_jax(x0)
                and tree_contains_jax(problem.sample) and tree_contains_jax(problem.data))


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


def _mc_sgd_step(
        problem,             # the (reg-scaled) minibatch Problem
        cores:     Tangent,  # the current point
        sample_B,            # the drawn minibatch sample
        data_B,              # its observed data
) -> Tangent:                # the retracted next point
    """One Manifold Cauchy SGD step: gradient, the tuning-free Cauchy length ``α = ‖g‖²/‖𝒥g‖²``, retract.

    Module-level and closure-free so the jit wrapper memoized on it is stable -- see :py:func:`_jitted`."""
    lm = problem.local_model(cores, sample_B, data_B)
    g = lm.gradient
    gg = problem.geom.inner(g, g)
    xnp, _, _ = get_backend(False, tree_contains_jax(g))
    alpha = gg / xnp.maximum(lm.gn_quadratic(g), 1e-30)
    return lm.retract(cw.corewise_scale(g, -alpha))


def _adam_step(
        problem,             # the (reg-scaled) minibatch Problem
        cores:     Tangent,  # the current point
        m:         Tangent,  # first-moment EMA, a tree matching the cores
        v:         Tangent,  # second-moment EMA
        sample_B,            # the drawn minibatch sample
        data_B,              # its observed data
        lr_t,                # the scheduled learning rate for this step
        t,                   # 1-based step index (bias correction)
        b1,                  # beta_1
        b2,                  # beta_2
        eps,                 # denominator floor
) -> typ.Tuple[Tangent, Tangent, Tangent]:   # (next point, m, v)
    """One Adam step over the cores, elementwise (so this IS per-core Adam). Module-level and
    closure-free; ``lr_t`` / ``t`` flow in as arguments so the schedule forces no recompile."""
    lm = problem.local_model(cores, sample_B, data_B)
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
_G0_REFERENCES = ('zero', 'warm_start')


def _check_g0_reference(
        g0_reference:  typ.Optional[str],   # None | 'zero' | 'warm_start' (case-insensitive)
        who:           str,                 # the optimizer name, for the message
) -> typ.Optional[str]:                     # the lower-cased selector, or None
    """Validate the ``g0_reference`` selector (structural: an unknown string raises in both modes)."""
    if g0_reference is None:
        return None
    ref = str(g0_reference).lower()
    if ref not in _G0_REFERENCES:
        raise ValueError("%s: g0_reference must be None (the geometry's own reference), 'zero' or "
                         "'warm_start'; got %r" % (who, g0_reference))
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
        problem:  'Problem',   # the level's own Problem (its kind carries the resolved chunk_size)
        frame0:   typ.Any,     # geom.zero_point_frame(frame): the canonical frame at the zero tensor
        x0:       Cores,       # the start point (only its STRUCTURE is used: the zero tensor at its ranks)
) -> float:                    # ‖Π 𝒥ᵀ r(0)‖ at frame0 -- the zero reference
    """The gradient norm at the zero tensor of ``x0``'s ranks, evaluated at the canonical zero-point frame.

    One frame sweep plus one transpose -- the cost of one Newton gradient -- on the level's own problem.
    The residual is ``point_forward(0) - data`` (``Problem.local_model``'s formula, so a custom kind with
    ``S(0) != 0`` is handled; for the built-in kinds it is ``-data``). A regularizer is not included: its
    gradient vanishes at the zero tensor, so the reference is the data-gradient scale alone."""
    geom, kind, sample = problem.geom, problem.kind, problem.sample
    sweep0 = kind.precompute(frame0, sample)
    r0 = cw.corewise_sub(kind.point_forward(cw.corewise_zeros_like(x0), sample), problem.data)
    g = geom.project(frame0, kind.transpose(r0, sample, frame0, sweep0), aux=geom.precompute(frame0))
    return float(geom.inner(g, g)) ** 0.5


def _resolve_g0_reference(
        problem:       'Problem',           # the level's own Problem
        frame:         typ.Any,             # iteration 0's local-model frame (its structure is copied)
        x0:            Cores,               # the start point
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
    zpf = getattr(problem.geom, 'zero_point_frame', None)
    if g0_reference == 'warm_start':
        return 'warm_start', None
    if g0_reference is None and not compute:
        return None, None                   # the overrides fix both references: nothing to resolve
    explicit = g0_reference == 'zero'
    if zpf is None:
        if explicit:
            raise ValueError("%s: g0_reference='zero' needs a geometry with a zero_point_frame member; %s "
                             "has none (pass g0_reference='warm_start', or give the geometry a "
                             "zero_point_frame)" % (who, type(problem.geom).__name__))
        return 'warm_start', None
    frame0 = zpf(frame)
    if frame0 is None:
        if explicit:
            raise ValueError("%s: g0_reference='zero' has no meaning on %s, which defines no zero-point frame "
                             "(on a corewise geometry the zero cores are a critical point, so the reference "
                             "would be exactly 0); pass g0_reference='warm_start' or None"
                             % (who, type(problem.geom).__name__))
        return 'warm_start', None
    if not compute:
        return 'zero', None
    zero_ref = _zero_reference_gnorm(problem, frame0, x0)
    if zero_ref > 0.0:
        return 'zero', zero_ref
    if explicit:
        raise ValueError("%s: the zero-tensor reference is 0 at these ranks; pass %s" % (who, hint))
    warnings.warn("%s: the zero-tensor reference vanishes at these ranks (the data gradient at the zero "
                  "tensor is 0, e.g. all-zero data); using the warm-start reference" % (who,),
                  RuntimeWarning, stacklevel=3)
    return 'warm_start', None


def gradient_descent(
        problem:  Problem,    # the fixed-rank least-squares problem
        x0:       Tangent,    # initial cores (U, G)
        n_iter:   int   = 100,
        gtol_rel: float = 1e-8,   # stop when ‖g‖ <= gtol_rel * ‖g_0‖
        c_armijo: float = 1e-4,   # Armijo sufficient-decrease constant
        on_line_search_failure: str = 'stop',  # Armijo exhausted (50 halvings): 'stop' = reject + terminate; 'accept' = take the last trial and continue
        g0_reference: typ.Optional[str] = None,  # ‖g_0‖ for the stop: None = the geometry's own ('zero' on manifolds); 'zero' | 'warm_start'
) -> typ.Tuple[Tangent, dict]:    # (x_cores, stats)
    """Steepest descent with a Cauchy initial step and an **Armijo backtracking line search**. The step
    length starts at the Cauchy value ``α = ‖g‖² / ‖𝒥g‖²`` (the 1D minimizer of the local GN quadratic
    along ``−g``) and backtracks (``α ← α/2``) until ``f(retract(−α g)) ≤ f − c·α‖g‖²`` -- so it descends
    on any geometry, including the additive corewise chart where a bare Cauchy step overshoots the
    high-degree objective. Exercises the whole backend-first stack (``gradient`` / ``gn_quadratic`` /
    ``objective`` / ``retract``). **Eager only, deliberately** -- the reference implementation of the
    backend-first stack; ``use_jit`` lives on ``mc_sgd`` / ``adam`` / ``newton_cg``.

    ``on_line_search_failure``: as in :py:func:`newton_cg` -- exhausting all 50 halvings means the
    objective's floor; ``'stop'`` (default) rejects the step and terminates, ``'accept'`` takes the
    last trial and keeps going (the deliberate escape mode). ``stats['line_search_failed']`` reports
    whether an exhaustion occurred.

    ``g0_reference``: the reference ``‖g_0‖`` of the relative stop, resolved exactly as in
    :py:func:`newton_cg` -- by default the gradient norm at the zero tensor of ``x0``'s ranks on a manifold
    geometry, the warm-start ``‖g(x0)‖`` on a corewise one."""
    _require_unstacked(problem.geom, x0, 'gradient_descent')
    olsf = str(on_line_search_failure).lower()
    if olsf not in ('stop', 'accept'):
        raise ValueError("on_line_search_failure must be 'stop' or 'accept'; got %r" % (on_line_search_failure,))
    ref_selector = _check_g0_reference(g0_reference, 'gradient_descent')
    x = x0
    losses = []
    any_ls_failed = False
    g0norm = None
    for it in range(n_iter):
        lm = problem.local_model(x)
        f = float(lm.objective)
        losses.append(f)
        g = lm.gradient
        gg = float(problem.geom.inner(g, g))                # ‖g‖²
        if g0norm is None:                                  # iteration 0: resolve the reference (once)
            ref, zero_ref = _resolve_g0_reference(problem, lm.frame, x0, ref_selector, True, 'gradient_descent',
                                                  "g0_reference='warm_start'")
            g0norm = zero_ref if ref == 'zero' else (gg ** 0.5 if gg > 0 else 1.0)
        if gg ** 0.5 <= gtol_rel * g0norm:                  # converged
            break
        alpha = gg / max(float(lm.gn_quadratic(g)), 1e-30 * gg)   # Cauchy initial step
        x_trial = x
        ls_failed = True
        for _bt in range(50):                               # Armijo backtracking
            x_trial = lm.retract(cw.corewise_scale(g, -alpha))
            if float(problem.objective(x_trial)) <= f - c_armijo * alpha * gg:
                ls_failed = False
                break
            if _bt < 49:
                alpha *= 0.5        # halve BETWEEN trials only (review R7-7)
        if ls_failed:
            any_ls_failed = True
            if olsf == 'stop':
                break                                        # reject the step: x unchanged
        x = x_trial
    return x, {'losses': losses, 'n_iter': len(losses), 'line_search_failed': any_ls_failed}


def mc_sgd(
        problem:     Problem,    # the fixed-rank least-squares problem
        x0:          Tangent,    # initial cores (U, G)
        rng,                     # np.random.Generator -- passed to the draw each step
        batch:       int,        # measurements per minibatch (for the default flat draw; ignored if draw given)
        draw:        typ.Optional[typ.Callable] = None,  # custom draw(rng)->(sample_B,data_B); None = flat default
        max_iter:    int   = 3000,
        check_every: int   = 25,   # iterations between full-batch loss checks (the absolute-iteration window)
        smooth_tau:  float = 2.0,  # EMA timescale of the loss check, in checks
        plateau_lag: int   = 4,    # stop when the smoothed loss rises over this many checks
        use_jit:     bool  = False,  # jit the per-step kernel (gradient + Cauchy step + retract) when jax
) -> typ.Tuple[Tangent, dict]:     # (x_cores, stats)
    """Manifold Cauchy SGD (T4S §5.3.2): minibatch + the tuning-free Cauchy step
    ``α = ‖g‖² / ‖𝒥g‖²`` -- no learning rate. Intended for the **manifold** geometry (its bounded
    retraction makes the raw Cauchy step stable; on the additive corewise chart use ``adam``). Stops on an
    exponentially-smoothed **full-batch** loss with an **absolute-iteration window** (``plateau_lag *
    check_every`` iterations) -- decoupled from batch size (the epoch-based window made small batches
    fragile; findings in a separate research repo, maintainer-local). ``stats['losses']`` records the
    **smoothed** check values (the stop signal), not the raw full-batch losses. ``use_jit`` jits the per-step kernel (the host loop
    draws minibatches; the full-batch stop check stays on the host), auto-converting numpy inputs to jax
    first (:py:func:`_prepare_jit_inputs`) -- so the **default** ``flat_draw`` then slices the on-device
    ``sample`` / ``data`` (a custom ``draw`` should return jax minibatches to stay on device)."""
    _require_unstacked(problem.geom, x0, 'mc_sgd')
    x0, problem = _prepare_jit_inputs(use_jit, x0, problem)
    draw = draw if draw is not None else flat_draw(problem, batch)
    step_problem = _minibatch_step_problem(problem, batch)  # reg scaled by batch/n; full reg kept for the stop
    a_smooth = 1.0 - math.exp(-1.0 / smooth_tau)

    step = _jitted(_mc_sgd_step) if _jit_engaged(use_jit, x0, problem) else _mc_sgd_step

    x = x0
    s_hist = []
    n_iter = 0
    for k in range(max_iter):
        n_iter = k + 1
        x = step(step_problem, x, *draw(rng))
        if n_iter % check_every == 0:                       # full-batch loss check (the stop signal, host)
            L = float(problem.objective(x))
            s = L if not s_hist else a_smooth * L + (1.0 - a_smooth) * s_hist[-1]
            s_hist.append(s)
            if len(s_hist) > plateau_lag and (s_hist[-1] - s_hist[-1 - plateau_lag]) > 0.0:
                break
    return x, {'losses': s_hist, 'n_iter': n_iter}


def adam(
        problem:  Problem,    # the fixed-rank least-squares problem
        x0:       Tangent,    # initial cores (U, G)
        rng,                  # np.random.Generator -- passed to the draw each step
        batch:    int,        # measurements per minibatch (for the default flat draw; ignored if draw given)
        draw:     typ.Optional[typ.Callable] = None,  # custom draw(rng)->(sample_B,data_B); None = flat default
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
    step.) On corewise the step is additive (``lm.retract`` = ``cores += step``). ``use_jit`` jits the
    per-step kernel (``lr_t``/``t`` flow in as traced args, so the schedule does not force a recompile),
    auto-converting numpy inputs to jax first (:py:func:`_prepare_jit_inputs`)."""
    _require_unstacked(problem.geom, x0, 'adam')
    x0, problem = _prepare_jit_inputs(use_jit, x0, problem)
    step_problem = _minibatch_step_problem(problem, batch)  # reg scaled by batch/n; full reg kept for the stop
    b1, b2 = betas

    step = _jitted(_adam_step) if _jit_engaged(use_jit, x0, problem) else _adam_step

    draw = draw if draw is not None else flat_draw(problem, batch)
    cores = x0
    m = cw.corewise_zeros_like(cores)
    v = cw.corewise_zeros_like(cores)
    losses = []
    for k in range(max_iter):
        sample_B, data_B = draw(rng)
        lr_t = lr * (0.5 * (1.0 + math.cos(math.pi * k / max_iter)) if cosine else 1.0)
        cores, m, v = step(step_problem, cores, m, v, sample_B, data_B, lr_t, k + 1, b1, b2, eps)
        if (k + 1) % 50 == 0:
            losses.append(float(problem.objective(cores)))
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


@dc.dataclass(frozen=True)
class NewtonInfo:
    """Per-iteration diagnostics passed to a ``newton_cg`` ``callback`` -- everything one Newton step
    produces, so a custom callback (or the display in :py:mod:`t3toolbox.backend.optimizer_display`) can
    report anything without the loop anticipating it. ``x_cores`` (the point *before* the step) and ``lm``
    (its :py:class:`LocalModel` -- residual / sample / frame) are carried so a callback can compute
    per-block errors or evaluate a validation forward. The step-related fields are ``None`` on the final
    **converged** line (no CG / line search ran). The scalar subset (all but ``x_cores`` / ``lm``) is what
    lands in ``stats['history']``.

    Five run-level strings (2026.3.0): ``newton_cg`` sets the first four on every row it emits, so a
    history consumer can filter by them, and ``stop_reason`` on its last row. All default to ``None``, so a
    hand-built row (a display test) needs none of them:

    * ``g0_reference`` -- which **Newton-stop** reference ``g0norm`` is: ``'zero'`` (the gradient norm at
      the zero tensor of these ranks, at the canonical zero-point frame), ``'warm_start'`` (the initial
      ``‖g‖``), or ``'override'`` (``g0norm_newton`` was given).
    * ``cg_reference`` -- the CG forcing term's reference: ``'override'`` iff ``g0norm_cg`` was given, else
      equal to ``g0_reference`` (CG inherits the Newton reference).
    * ``step_control`` -- the step-acceptance mode: ``'line_search'`` (Armijo backtracking along the
      retraction) or ``'trust_region'`` (one trial at ``α = 1``, accepted on the ratio ``ρ``; Steihaug CG).
    * ``hessian`` -- the local model's Hessian: ``'gauss_newton'`` or ``'exact'``.
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
    ``lift_eps`` are set only with ``hessian='exact'`` (the last two only on a geometry whose curvature data
    carries them, i.e. the manifold one at ``d > 1``). The CG outcome is exactly one of ``cg_converged``,
    ``cg_boundary``, ``cg_truncated`` or (none of them) the iteration cap."""
    iteration:    int              # Newton iteration index (0-based)
    objective:    float            # total objective ½‖ω⊙r‖² + ρ(X) at x (= misfit + regularization; = lm.objective)
    gnorm:        float            # ‖g‖ (Riemannian gradient norm, geom.inner)
    g0norm:       float            # ‖g₀‖ Newton-stop reference in effect (its kind: g0_reference)
    converged:    bool             # gnorm <= gtol_rel*g0norm -- this is the final line (step fields None)
    x_cores:      typ.Any = None                       # the point BEFORE the step (for a val forward)
    lm:           typ.Any = None                       # the LocalModel at x (residual/sample/frame)
    misfit:         typ.Optional[float] = None         # ½‖ω⊙r‖² data-misfit part of the objective
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
    g0_reference: typ.Optional[str]   = None           # the Newton-stop reference: 'zero' | 'warm_start' | 'override'
    cg_reference: typ.Optional[str]   = None           # the CG-forcing reference: 'override' iff g0norm_cg, else = g0_reference
    step_control: typ.Optional[str]   = None           # 'line_search' | 'trust_region'
    stop_reason:  typ.Optional[str]   = None           # last row only: 'gradient' | 'max_newton' | 'line_search' | 'objective_floor' | 'trust_region_floor'
    hessian:      typ.Optional[str]   = None           # 'gauss_newton' | 'exact'
    tr_radius:    typ.Optional[float] = None           # trust-region radius Δ this trial was bounded by (TR mode)
    tr_accepted:  typ.Optional[bool]  = None           # the trial was accepted (ρ > 0.1); False keeps x (TR mode)
    cg_boundary:  typ.Optional[bool]  = None           # CG stopped on the trust-region boundary ‖p‖ = Δ (Steihaug)
    pKp:          typ.Optional[float] = None           # ⟨p, Lᵀ K_N L p⟩: the curvature term's part of pHp (exact mode)
    lift_sigma_min: typ.Optional[float] = None         # the smallest lift singular value at this frame (exact mode)
    lift_eps:     typ.Optional[float] = None           # the absolute lift damping ε at this frame (exact mode)


_NEWTON_SCALAR_FIELDS = tuple(f.name for f in dc.fields(NewtonInfo) if f.name not in ('x_cores', 'lm'))


def _newton_scalar_record(info: NewtonInfo) -> dict:
    """The scalar subset of a :py:class:`NewtonInfo` (drops ``x_cores`` / ``lm``) -- one ``history`` row."""
    return {k: getattr(info, k) for k in _NEWTON_SCALAR_FIELDS}


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
    exact one (indefinite away from a minimizer: the region bounds the step)."""
    if step_control is None:
        return 'trust_region' if hessian == 'exact' else 'line_search'
    if not isinstance(step_control, str) or step_control.lower() not in _STEP_CONTROLS:
        raise ValueError("newton_cg: step_control must be None, 'line_search' or 'trust_region' "
                         "(case-insensitive); got %r" % (step_control,))
    return step_control.lower()


def _floating_dtypes(
        tree:  typ.Any,              # a residual tree: nested tuples / lists of numpy or jax arrays
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
        residual:  typ.Any,          # the local model's residual tree (numpy or jax leaves)
) -> float:                          # np.finfo(dtype).eps, maximised over the floating leaves
    """The machine epsilon of the residual -- host-side and **dtype-only** (no array is read, so no device
    sync): the largest ``np.finfo(dtype).eps`` over the residual's floating leaves, so a mixed float32 /
    float64 tree is judged at float32's resolution. ``float64``'s eps if there is no floating leaf."""
    eps = [float(np.finfo(dtype).eps) for dtype in _floating_dtypes(residual)]
    return max(eps) if eps else float(np.finfo(np.float64).eps)


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


def newton_cg(
        problem:                Problem,  # the fixed-rank least-squares problem
        x0:                     Tangent,  # initial cores (U, G)

        max_newton:             int   = 30,    # Newton iterations (history rows) -- rejected trust-region trials included
        gtol_rel:               float = 1e-8,  # stop when ‖g‖ <= gtol_rel * ‖g0‖  (‖g0‖: g0norm_newton, else the g0_reference)
        cg_maxiter:             int   = 200,
        c_armijo:               float = 1e-4,  # Armijo constant (line search only)
        on_line_search_failure: str = 'stop',  # Armijo exhausted (40 halvings): 'stop' = reject the step and terminate; 'accept' = take the last trial and continue

        hessian:                str   = 'gauss_newton',      # or 'exact' (case-insensitive): the local model's Hessian
        curvature_damping:      typ.Optional[float] = None,  # exact only: the lift damping, relative to ‖X‖; None = the geometry's default
        step_control:           typ.Optional[str]   = None,  # None = 'line_search' (GN) / 'trust_region' (exact); either works with either
        tr_radius0:             typ.Optional[float] = None,  # trust region only: Δ0 > 0; None = the GN Cauchy length ‖g‖³/⟨g, H_GN g⟩

        g0_reference:           typ.Optional[str]   = None,  # which ‖g0‖: None = the geometry's own ('zero' on manifolds, 'warm_start' on corewise); 'zero' | 'warm_start'
        g0norm_newton:          typ.Optional[float] = None,  # explicit ‖g0‖ for the Newton stop (wins; > 0). Also feeds CG unless g0norm_cg set.
        g0norm_cg:              typ.Optional[float] = None,  # explicit ‖g0‖ for the CG forcing term (wins for CG; > 0); default = the Newton reference

        cg_forcing_power:       float = 0.5,    # η = min(0.5, (‖g‖/‖g0‖)**power); larger => tighter CG, fewer Newton steps
        use_jit:                bool  = False,  # jit the inner CG (lax.while_loop) when the inputs are jax; else eager

        callback:               typ.Optional[typ.Callable] = None,  # callback(NewtonInfo) each iteration (host-side; e.g. a display)
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
    ``Π Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π`` (:py:meth:`Problem.local_model`; it needs a kind implementing
    ``corewise_hessian`` and a geometry with the four exact-mode members). Gauss-Newton converges linearly at
    a rate set by that missing term (``|1 − ρ|`` measures it); the exact Hessian gives the superlinear Newton
    tail -- q-quadratic with ``cg_forcing_power=1.0``, order about 1.5 at the default ``0.5``. It costs a
    few times a Gauss-Newton Hessian-apply. ``curvature_damping`` (exact only; a value with Gauss-Newton
    raises) is the lift's damping relative to ``‖X‖``, ``None`` for the geometry's own default.

    **Step control** (``step_control``; the default follows the Hessian, and either combines with either):

    * ``'line_search'`` (the Gauss-Newton default): plain truncated CG, then Armijo backtracking along
      ``retract(α p)``.
    * ``'trust_region'`` (the exact default -- the exact ``H`` is indefinite away from a minimizer):
      Steihaug-Toint CG inside ``‖p‖ ≤ Δ`` (it steps to the boundary on nonpositive curvature or when it
      would leave the region), then **one** trial ``retract(p)``, accepted iff the ratio of the actual to the
      model's predicted change ``ρ = Δf / (gᵀp + ½ pᵀHp)`` exceeds ``0.1``. ``Δ ← ¼‖p‖`` when ``ρ < 0.25`` (or
      not finite); ``Δ ← min(2Δ, 10⁴Δ0)`` when ``ρ > 0.75`` on a boundary step. ``Δ0 = tr_radius0``, by
      default the Gauss-Newton Cauchy length ``‖g‖³ / ⟨g, H_GN g⟩``. A rejected trial keeps ``x`` -- and the
      local model, which is rebuilt only after an accepted step -- emits its own history row
      (``tr_accepted=False``) and counts toward ``max_newton``. Two **floors**, checked only after a rejection
      (an accepted step never stops the run): ``'objective_floor'`` when the predicted change is below the
      objective's floating-point resolution, ``|pred| ≤ 10·ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)`` with ``ε`` the residual
      dtype's machine epsilon -- the attainable accuracy is reached, which is **not a failure** (``converged``
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
      zero-point frame (:py:func:`~t3toolbox.backend.fv_operations.fv_zero_point_frame`): one extra sweep
      and transpose on this problem. A deterministic data-gradient scale at these ranks, fixed by
      convention and good to an O(1) factor -- comparable across rank-continuation levels and identical on
      the ragged and uniform layers. It is the right reference after a **warm start**, whose initial
      gradient is deflated (the guess is already near the solution). An explicit ``'zero'`` on a geometry
      with no zero-point frame raises ``ValueError``.
    * ``'warm_start'`` -- the initial gradient norm ``‖g(x0)‖`` (the 2026.2.0 default; ``1.0`` if it is 0,
      when x0 is stationary and the stop fires at once).

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
    converged line) -- the hook for a live diagnostic display; it runs **host-side** (it reads the concrete
    residual), so it composes with ``use_jit`` (only the inner CG jits) but not with a hypothetical
    fully-jitted outer loop. Ready-made displays: :py:func:`t3toolbox.backend.optimizer_display.make_newton_display`.

    ``stats`` is ``{'losses', 'newton_steps', 'newton_accepted', 'stop_reason', 'history'}``:
    ``newton_steps`` counts the CG solves (the history rows that took a step), ``newton_accepted`` the steps
    taken (``newton_steps`` minus the rejections: an Armijo-exhausted one under ``'stop'``, or the rejected
    trust-region trials), ``stop_reason`` is ``'gradient'`` / ``'max_newton'`` / ``'line_search'`` /
    ``'objective_floor'`` / ``'trust_region_floor'`` (see :py:class:`NewtonInfo`), and ``history`` is one
    :py:func:`_newton_scalar_record` per iteration (``losses[k] == history[k]['objective']``). (The 2026.2.0
    ``stats['newton']`` is removed; it is ``newton_steps``.)

    Examples
    --------
    The exact Hessian with its default trust region, against Gauss-Newton with its default line search, on
    a noisy probe-derivative fit warm-started near the solution:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.fitting as bfit
    >>> import t3toolbox.backend.geometry as bgeo
    >>> import t3toolbox.backend.optimizers as bopt
    >>> np.random.seed(0)
    >>> shape, rng = (5, 5, 5), np.random.default_rng(0)
    >>> ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
    >>> kind = bfit.ProbeDerivativesKind(order=1)
    >>> target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1)).data
    >>> data = [z + 0.01 * rng.standard_normal(z.shape) for z in kind.point_forward(target, (ww, pp))]
    >>> problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), kind, (ww, pp), data)
    >>> x0, _ = bopt.newton_cg(problem, t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)).data,
    ...                        max_newton=5)
    >>> _, gn = bopt.newton_cg(problem, x0, max_newton=40, g0_reference='warm_start')
    >>> _, ex = bopt.newton_cg(problem, x0, max_newton=40, g0_reference='warm_start',
    ...                        hessian='exact', cg_forcing_power=1.0)
    >>> ex['history'][0]['step_control'], ex['stop_reason'] in ('gradient', 'objective_floor')
    ('trust_region', True)
    >>> gn['newton_steps'] > 2 * ex['newton_steps']     # Gauss-Newton is linear here; exact has the Newton tail
    True
    """
    _require_unstacked(problem.geom, x0, 'newton_cg')
    olsf = str(on_line_search_failure).lower()
    if olsf not in ('stop', 'accept'):
        raise ValueError("on_line_search_failure must be 'stop' or 'accept'; got %r" % (on_line_search_failure,))
    hessian = _check_hessian(hessian, 'newton_cg')
    exact = hessian == 'exact'
    if curvature_damping is not None and not exact:
        raise ValueError("newton_cg: curvature_damping is only meaningful with hessian='exact'")
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
    x0, problem = _prepare_jit_inputs(use_jit, x0, problem)             # auto-convert to jax when jitting
    model_options = dict(hessian='exact', curvature_damping=curvature_damping) if exact else {}
    x = x0
    lm = None                                                           # rebuilt only after an accepted step
    newton_ref = cg_ref = None                                          # resolved once, at iteration 0
    g0_ref_kind = cg_ref_kind = None
    radius = radius0 = radius_max = None                                # trust region: set at the first step
    data_norm = None                                                    # ‖ω⊙y‖: once, at the first rejection
    losses, newton_steps, newton_accepted, history = [], 0, 0, []
    stop_reason = 'max_newton'
    t_prev = time.perf_counter()
    for it in range(max_newton):
        if lm is None:                                                  # a new point: linearize there
            lm = problem.local_model(x, **model_options)
            misfit = float(lm.misfit)
            reg_val = float(lm.regularization) if lm.regularizer is not None else None   # None: no regularizer attached
            f = misfit + (reg_val or 0.0)
            g = lm.gradient
            gnorm = float(problem.geom.inner(g, g)) ** 0.5
            lift_sigma_min, lift_eps = _lift_diagnostics(lm)
        losses.append(f)
        if newton_ref is None:                                           # iteration 0: resolve the references
            ref, zero_ref = _resolve_g0_reference(problem, lm.frame, x0, ref_selector, g0norm_newton is None,
                                                  'newton_cg', "g0_reference='warm_start' or g0norm_newton")
            if g0norm_newton is not None:
                newton_ref, g0_ref_kind = g0norm_newton, 'override'
            elif ref == 'zero':
                newton_ref, g0_ref_kind = zero_ref, 'zero'
            else:
                newton_ref, g0_ref_kind = (gnorm if gnorm > 0 else 1.0), 'warm_start'   # the initial ‖g‖
            if g0norm_cg is not None:
                cg_ref, cg_ref_kind = g0norm_cg, 'override'
            else:
                cg_ref, cg_ref_kind = newton_ref, g0_ref_kind            # CG inherits the Newton reference
        run_fields = dict(g0_reference=g0_ref_kind, cg_reference=cg_ref_kind, step_control=step_control,
                          hessian=hessian, lift_sigma_min=lift_sigma_min, lift_eps=lift_eps)
        if gnorm <= gtol_rel * newton_ref:                               # converged -- final line, no step
            stop_reason = 'gradient'
            info = NewtonInfo(iteration=it, objective=f, gnorm=gnorm, g0norm=newton_ref, converged=True,
                              x_cores=x, lm=lm, misfit=misfit, regularization=reg_val,
                              stop_reason=stop_reason, **run_fields)
            if callback is not None:
                callback(info)
            history.append(_newton_scalar_record(info))
            break
        newton_steps += 1
        eta = min(0.5, (gnorm / cg_ref) ** cg_forcing_power)             # inexact-Newton (Eisenstat-Walker) forcing term
        cg_tol = eta * gnorm
        neg_g = cw.corewise_scale(g, -1.0)
        if trust_region and radius is None:                              # the first step: the initial radius
            radius0 = float(tr_radius0) if tr_radius0 is not None else _initial_radius(lm, g, gnorm)
            radius, radius_max = radius0, _TR_RADIUS_MAX_REL * radius0
        tr_radius = radius if trust_region else None                     # the radius THIS trial is bounded by
        p, cg_i, cg_rs, cg_ok, cg_hit = _cg_solve(lm, neg_g, cg_tol, cg_maxiter, use_jit,
                                                  radius if trust_region else math.inf)
        cg_iters, cg_rs, cg_ok, cg_boundary = int(cg_i), float(cg_rs), bool(cg_ok), bool(cg_hit)
        cg_converged = (cg_rs <= cg_tol * cg_tol) and not cg_boundary
        cg_truncated = (not cg_converged) and (not cg_ok) and (not cg_boundary)
        on_boundary = cg_boundary                                        # the step was limited by Δ
        slope = float(problem.geom.inner(g, p))
        if (not math.isfinite(slope)) or slope >= 0.0:                   # ensure a descent direction
            if trust_region:                                             #   ... inside the region
                scale = min(1.0, radius / gnorm)
                p, slope, on_boundary = cw.corewise_scale(g, -scale), -scale * gnorm * gnorm, scale < 1.0
            else:
                p, slope = neg_g, -gnorm * gnorm
        if exact:                                                        # pᵀHp = GN part + curvature part (once each)
            pKp = lm.curvature_quadratic(p)
            pHp = float(lm.gn_quadratic(p) + pKp)                        # == lm.quadratic(p)
            pKp = float(pKp)
        else:
            pHp, pKp = float(lm.quadratic(p)), None                      # pᵀHp = ‖𝒥p‖² (+ reg) (for ρ)
        p_norm = float(problem.geom.inner(p, p)) ** 0.5
        stop_now = False
        if not trust_region:                                             # ---- Armijo line search
            alpha, ls_steps, f_new = 1.0, 40, f
            x_trial = x
            ls_failed = True
            for bt in range(40):                                         # Armijo backtracking
                x_trial = lm.retract(cw.corewise_scale(p, alpha))
                f_new = float(problem.objective(x_trial))
                if f_new <= f + c_armijo * alpha * slope:
                    ls_steps, ls_failed = bt, False
                    break
                if bt < 39:
                    alpha *= 0.5        # halve BETWEEN trials only: alpha stays the last trial's step (review R7-7)
            delta_f = f_new - f
            predicted = alpha * slope + 0.5 * alpha * alpha * pHp        # the model's change along αp
            rho = (delta_f / predicted) if predicted != 0.0 else float('nan')
            accepted = not (ls_failed and olsf == 'stop')                # 'stop' rejects: x unchanged (review R7-7)
            if not accepted:
                stop_reason, stop_now = 'line_search', True
            tr_accepted = None
        else:                                                            # ---- trust region: one trial at α = 1
            alpha, ls_steps, ls_failed = 1.0, None, None
            x_trial = lm.retract(p)
            f_new = float(problem.objective(x_trial))
            delta_f = f_new - f
            predicted = slope + 0.5 * pHp                                # the model's change along p (< 0 expected)
            rho = (delta_f / predicted) if predicted < 0.0 else float('nan')
            accepted = tr_accepted = math.isfinite(rho) and rho > _TR_ACCEPT
            if (not math.isfinite(rho)) or rho < _TR_SHRINK_BELOW:
                radius = _TR_SHRINK * p_norm
            elif rho > _TR_EXPAND_ABOVE and on_boundary:
                radius = min(_TR_EXPAND * radius, radius_max)
            if not accepted:                                             # the floors: on a rejection ONLY
                if data_norm is None:
                    data_norm = float(problem.kind.sumsq(problem.data, lm.n_w)) ** 0.5
                resolution = _resolution_eps(lm.residual) * (data_norm * math.sqrt(2.0 * misfit) + abs(f))
                if abs(predicted) <= _TR_RESOLUTION_FACTOR * resolution:
                    stop_reason, stop_now = 'objective_floor', True      # the model change is below f's resolution
                elif radius < _TR_RADIUS_FLOOR_REL * radius0:
                    stop_reason, stop_now = 'trust_region_floor', True
        x_norm = float(cw.corewise_norm(x))                             # coordinate norm (ragged-exact; uniform: D6)
        step_rel = (alpha * p_norm / x_norm) if x_norm > 0 else float('nan')
        last_row = stop_now or it == max_newton - 1
        t_now = time.perf_counter()
        info = NewtonInfo(iteration=it, objective=f, gnorm=gnorm, g0norm=newton_ref, converged=False,
                          x_cores=x, lm=lm, misfit=misfit, regularization=reg_val,
                          forcing_eta=eta, cg_tol=cg_tol, cg_iters=cg_iters,
                          cg_maxiter=cg_maxiter,
                          cg_resid=cg_rs ** 0.5, cg_converged=cg_converged, cg_truncated=cg_truncated,
                          ls_steps=ls_steps, ls_failed=ls_failed, alpha=alpha, slope=slope, pHp=pHp,
                          delta_f=delta_f, rho=rho, step_rel=step_rel, wall_time=t_now - t_prev,
                          stop_reason=stop_reason if last_row else None, tr_radius=tr_radius,
                          tr_accepted=tr_accepted, cg_boundary=cg_boundary, pKp=pKp, **run_fields)
        t_prev = t_now
        if callback is not None:
            callback(info)
        history.append(_newton_scalar_record(info))
        if stop_now:
            break
        if accepted:
            newton_accepted += 1
            x, lm = x_trial, None                                        # the next row linearizes at the new point
    return x, {'losses': losses, 'newton_steps': newton_steps, 'newton_accepted': newton_accepted,
               'stop_reason': stop_reason, 'history': history}


if jax_available:
    import jax

    # Register Problem and LocalModel as jax pytrees. The DATA are leaves (sample / data / frame / sweep /
    # residual / the per-frame geometry companion); the STATICS are aux (geometry, kind, regularizer, and
    # the sample-stack axis count derived from the sample's shape).
    #
    # This is what makes the jit cache key value-based, and it is the whole point of slices 1-2: geometry
    # and kind are frozen dataclasses whose parameters are FIELDS, so a model rebuilt at each Newton
    # iteration -- which the outer loop does, every iteration, by construction -- hashes and compares
    # EQUAL to the previous one. Under the old record-of-closures they were fresh objects every time, so
    # `jit(_cg_core)` would have recompiled per iteration and there would have been no point compiling it
    # at all. The regularizers are frozen dataclasses of floats, hence hashable by value already.
    # The data tuples are partitioned rather than handed over whole: a backend frame is
    # `(supercores..., shape, masks)` and a frame sweep carries a shape too, so flattening them naively
    # would TRACE the masks and the shape ints -- which `require_concrete_masks` rejects outright, and
    # which would break the host-integer shape arithmetic the uniform layer runs on. `partition_static`
    # keeps them in the aux, exactly as the frontend `UT3Frame` keeps its masks in aux.
    def _flatten_problem(p):
        dyn, skel = partition_static((p.sample, p.data))
        return dyn, (p.geom, p.kind, p.regularizer, skel)

    def _unflatten_problem(aux, children):
        sample, data = rebuild_static(children, aux[3])
        return Problem(aux[0], aux[1], sample, data, aux[2])

    # The exact-mode companion rides in the same tuple: its arrays are leaves, and its PRESENCE (curvature is
    # None or not; geometry_data None or not) is tree structure, so the mode is static under the treedef --
    # there is no mode aux to carry.
    def _flatten_local_model(m):
        dyn, skel = partition_static((m.sample, m.frame, m.sweep, m.residual, m.geom_aux, m.curvature))
        return dyn, (m.geom, m.kind, m.n_w, m.regularizer, skel)

    def _unflatten_local_model(aux, children):
        sample, frame, sweep, residual, geom_aux, curvature = rebuild_static(children, aux[4])
        return LocalModel(aux[0], aux[1], sample, frame, sweep, residual, aux[2], aux[3], geom_aux,
                          curvature=curvature)

    def _flatten_curvature_state(c):
        return partition_static((c.misfit_gradient, c.geometry_data))

    def _unflatten_curvature_state(skeleton, children):
        return CurvatureState(*rebuild_static(children, skeleton))

    jax.tree_util.register_pytree_node(Problem, _flatten_problem, _unflatten_problem)
    jax.tree_util.register_pytree_node(LocalModel, _flatten_local_model, _unflatten_local_model)
    jax.tree_util.register_pytree_node(CurvatureState, _flatten_curvature_state, _unflatten_curvature_state)
