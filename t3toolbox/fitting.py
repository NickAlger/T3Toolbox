# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/T3Toolbox
'''Frontend for the least-squares fitting layer: one geometry-generic local model.

:py:func:`local_model` builds the local model of an **oracle** (the objective's description, e.g.
:py:class:`LeastSquaresOracle` for ``½‖ω⊙(S(X) − y)‖²``) on a **batch** of data (e.g.
:py:class:`LeastSquaresBatch`) at a point ``X``, linearized through a **geometry**: a :py:class:`LocalModel`
exposing the objective value ``c``, the gradient ``g``, the Gauss-Newton Hessian action ``H p = JᵀJ p`` and the
quadratic-model value ``m(p) = c + gᵀp + ½ pᵀ H p``. Built with ``hessian='exact'`` its
:py:meth:`~LocalModel.hessian` is the **exact** Riemannian Hessian instead (the Gauss-Newton term plus the
curvature term). It wraps the backend :py:class:`~t3toolbox.backend.optimizers.LocalModel` (same name, one level
down) with tangents in and out. The geometry is one of

  * :py:data:`~t3toolbox.manifold.MANIFOLD` (optimize on the fixed-rank manifold; the gauge projection ``Π``
    makes ``g`` / ``H`` Riemannian) or :py:data:`~t3toolbox.manifold.COREWISE` (optimize the raw cores; no
    ``Π``), their uniform twins, or a shared wrapper over one. **Manifold ⟺ Π, corewise ⟺ no Π is structural**
    (bundled in the geometry), never a flag.

Every input and output is a :py:class:`~t3toolbox.manifold.T3Tangent` (``UT3Tangent``) at ``model.frame`` --
including the **corewise** case, where the tangent lives at the non-orthonormal frame ``(B,G,G,G)`` (a core
perturbation; see :py:class:`~t3toolbox.manifold.CorewiseGeometry`). Build trial steps at ``model.frame``
(e.g. ``geometry.randn(model.frame)``); a step at a different frame raises the numerical same-frame guard
(skipped under ``safety.unsafe()`` / a jax trace), like :py:class:`~t3toolbox.manifold.T3Tangent`.

The model's build makes one oracle query and keeps what it computed (``held``); each action is one more query
at the same frame on the same batch with it. ``LocalModel`` is a registered jax pytree (the backend model is
its child, the geometry its aux), so a jitted matvec ``jax.jit(lambda model, p: model.gn_hessian(p))`` compiles
once across frames. See ``docs/fitting_and_optimization.md``.

Preparing data, and the jet helpers
-----------------------------------
The functions a user calls before fitting are reachable here under their backend names, so that a fit needs no
import from ``t3toolbox.backend``:

  * derivative data to the library's Taylor coefficients: :py:func:`derivatives_to_taylor`,
    :py:func:`taylor_to_derivatives`, :py:func:`derivative_weight_to_taylor_weight`, and
    :py:func:`prepare_derivative_data`, which does the data, the validation data and the weight in one call
    (:py:mod:`t3toolbox.backend.jets`);
  * badly scaled data: :py:func:`data_block_norms`, :py:func:`equalizing_weight`, :py:func:`rescale_direction`
    and :py:func:`t3_unscale_coordinates` (:py:mod:`t3toolbox.backend.data_preparation`), and this module's own
    :py:func:`rescale_coordinates`, which checks the form of the scales (one per coordinate) and then calls the
    backend function of the same name, which does not; a tensor fitted in rescaled coordinates also has the
    method ``x.unscale_coordinates(scales)``, which makes the same check;
  * the dual jet: :py:func:`jet_to_dual_jet`, and this module's own :py:func:`dual_jet_to_jet`, which raises in
    safe mode where the squared weight is zero (the backend function of the same name divides by zero);
  * the input jets a sampling function with a jet shape takes: :py:func:`affine_jet` (the line ``w + s p``, or a
    point moved along several directions) and :py:func:`index_to_one_hot` (the one-hot vectors of a grid point),
    also reachable from ``t3toolbox`` itself.
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
import t3toolbox.backend.regularization as breg     # the backend geometry a Regularizer leans on (no cycle: backend never imports fitting)
import t3toolbox.backend.uniform_fitting as ufit
import t3toolbox.backend.jets as backend_jets
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.ufv_conversions as ufv_conversions
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch   # one class each, both tiers
from t3toolbox.backend.jets import (                # reachable here under their backend names
    derivatives_to_taylor,
    taylor_to_derivatives,
    derivative_weight_to_taylor_weight,
    prepare_derivative_data,
    jet_to_dual_jet,
    affine_jet,
    index_to_one_hot,
)
import t3toolbox.backend.data_preparation as backend_preparation
from t3toolbox.backend.data_preparation import (    # reachable here under their backend names
    data_block_norms,
    equalizing_weight,
    rescale_direction,
    t3_unscale_coordinates,
)
from t3toolbox.backend.common import *

__all__ = ['LocalModel', 'local_model',
           'LeastSquaresOracle', 'LeastSquaresBatch',
           # derivative data and the jet helpers (the backend functions, and this module's own dual_jet_to_jet)
           'derivatives_to_taylor', 'taylor_to_derivatives', 'derivative_weight_to_taylor_weight',
           'prepare_derivative_data', 'jet_to_dual_jet', 'dual_jet_to_jet',
           # the input jets (the backend functions)
           'affine_jet', 'index_to_one_hot',
           # preparing the data of a fit (the backend functions, and this module's own rescale_coordinates)
           'data_block_norms', 'equalizing_weight', 'rescale_direction', 'rescale_coordinates',
           't3_unscale_coordinates']


def rescale_coordinates(
        kind:    str,                      # 'probe' | 'apply'
        order:   typ.Tuple[int, ...],      # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        sample:  typ.Any,                  # ww, the input jets: len=d, elm_shape=J_in_i+W+(Ni,)
        y:       typ.Any,                  # of the type of z: probe len=d, elm_shape=F+W+(Ni,); apply F+W
        scales:  typ.Sequence[NDArray],    # len=d, elm_shape=(Ni,): coordinate j of mode i times scales[i][j]
) -> typ.Tuple[
        typ.Any,                           # the sample: every slot of every input jet times its mode's scales
        typ.Any,                           # y: for probe, mode i's data divided by scales[i]; for apply, as given
]:
    """Rescale the coordinates of each mode: the inputs times ``scales``, the probe data divided by them.

    The frontend's function over :py:func:`t3toolbox.backend.data_preparation.rescale_coordinates`, with the
    same arguments and the same result (its docstring has the mathematics and an example): coordinate ``j`` of
    mode ``i`` of every slot of every input jet is multiplied by ``scales[i][j]``, and for the probe kind the data of
    mode ``i`` are divided by ``scales[i]``. This changes the unknowns: after the fit,
    ``x.unscale_coordinates(scales)`` gives the fitted tensor in the original coordinates.

    **What this function adds is the check of ``scales``**, which the backend function does not make: one
    vector of ``Ni`` scales per mode, with the mode sizes read off the sample (the last axis of each ``ww[i]``).
    Scales of a wrong count, of a wrong length for a mode (a length of 1 included) or that are not vectors
    raise a ``ValueError`` here, through
    :py:func:`t3toolbox.backend.data_preparation.check_coordinate_scales`, where the backend function would
    silently drop a mode or broadcast. The check is structural, so it is made always: in safe and in unsafe
    mode, and under ``jax.jit`` (shapes are static). No value is checked: a zero scale passes.

    Parameters
    ----------
    kind: str
        ``'probe'`` or ``'apply'`` (``'entries'`` has no coordinates to rescale, and raises).
    order: tuple of int
        The jet shape (``()`` for no jets).
    sample
        The input jets ``ww``: ``len=d``, ``elm_shape=J_in_i+W+(Ni,)``.
    y: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        The data. probe: ``len=d``, ``elm_shape=F+W+(Ni,)``. apply: ``shape=F+W``.
    scales: Sequence[NDArray]
        One positive scale per coordinate: ``len=d``, ``elm_shape=(Ni,)``.

    Returns
    -------
    sample
        The rescaled input jets, a tuple.
    y: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        The data for the rescaled sample.

    Raises
    ------
    ValueError
        If ``scales`` is not one vector of ``Ni`` scales per mode; if ``kind`` is not ``'probe'`` or ``'apply'``.
    TypeError
        If ``order`` is not a tuple of integers.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.fitting as fitting
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(7, N) for N in x.shape]
    >>> y = x.apply(ww)
    >>> scales = [10.0 ** np.random.randn(N) for N in x.shape]          # one scale per coordinate
    >>> ww_rescaled, y_rescaled = fitting.rescale_coordinates('apply', (), ww, y, scales)
    >>> print(np.allclose(ww_rescaled[1], ww[1] * scales[1]), y_rescaled is y)
    True True

    One number per mode is not the form, and is refused here (the backend function would broadcast it):

    >>> fitting.rescale_coordinates('apply', (), ww, y, [np.ones(1) for N in x.shape])   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: scales[0] has shape (1,); expected (N_0,) = (4,): one scale per coordinate of mode 0
    """
    backend_jets._kind_and_order(kind, order, 'rescale_coordinates')   # the kind and the jet shape first, as dual_jet_to_jet
    if kind in ('probe', 'apply'):                      # any other kind: the backend function raises
        backend_preparation.check_coordinate_scales(scales, tuple(np.shape(w_JWo)[-1] for w_JWo in sample))
    return backend_preparation.rescale_coordinates(kind, order, sample, y, scales)


def dual_jet_to_jet(
        kind:      str,                    # 'probe' | 'apply' | 'entries': which type the dual jet is of
        order:     typ.Tuple[int, ...],    # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...). entries: ()
        dual_jet:  typ.Any,                # of the type of z: probe len=d, elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C; entries W+[K]+C
        weight:    typ.Optional[NDArray],  # probe (d,)+F; apply F; entries (); None = ones
) -> typ.Any:                              # the jet: every jet axis flipped back, then divided by weight**2. probe: a tuple
    """Make a dual jet into a jet: the inverse of :py:func:`jet_to_dual_jet`, checked in safe mode.

    Every jet axis is flipped back and each slot is divided by ``weight**2``:
    ``jet[k] = dual_jet[order - k] / weight[k]**2``. Where a weight is zero the dual jet keeps nothing of the
    jet, and there is nothing to give back.

    **In safe mode** (the default) a zero divisor raises a ``ValueError``. The check is on the divisor as it is
    computed, ``weight**2`` in the precision of the computation (the lower floating precision of the dual jet
    and the weight), and it is exact, with no tolerance: a small weight is a legitimate weight and passes, but
    a weight whose square underflows to zero in that precision (``1e-30`` in float32, or a float64 ``1e-30``
    against a float32 dual jet) is refused like a zero. Inside ``safety.unsafe()`` and under a jax trace
    (``jit``, ``grad``, ``vmap``: a tracer cannot be branched on) the check is skipped and the division is
    made, giving ``inf`` or ``nan`` in those slots; everywhere else the numbers are the same in both modes.
    The backend function, :py:func:`t3toolbox.backend.jets.dual_jet_to_jet`, never checks.

    ``kind`` says which type the dual jet is of: for ``'probe'`` one array per mode, taken along the first axis
    (a tuple, a list, or an array packed for the uniform layout), and a tuple is returned; for ``'apply'`` and
    ``'entries'`` one array, its leading axes the jet axes (see
    :py:func:`t3toolbox.backend.jets.jet_to_dual_jet`).

    Parameters
    ----------
    kind: str
        ``'probe'``, ``'apply'`` or ``'entries'``.
    order: tuple of int
        The jet shape. ``()`` for no jets, and the only one the entries kind takes.
    dual_jet: tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        probe: ``len=d``, ``elm_shape=F+W+[K]+C+(Ni,)``. apply: ``shape=F+W+[K]+C``. entries: ``shape=W+[K]+C``.
    weight: NDArray or None
        probe: ``shape=(d,)+F``. apply: ``shape=F``. entries: ``shape=()``. ``None`` is ones, the library's default
        weight (a default that may change in a later release). The argument is required: pass ``None`` for
        no weight.

    Returns
    -------
    tuple of NDArray or NDArray
        Of the type of ``z`` (a tuple of ``d`` arrays for probe, one array otherwise).
        The jet, with the shapes of ``dual_jet``. probe: a tuple, one array per mode.

    Raises
    ------
    ValueError
        In safe mode, if an entry of ``weight**2``, as computed, is zero. If ``kind`` is not one of the three
        kinds, or the entries kind is given a jet shape but ``()``.
    TypeError
        If ``order`` is not a tuple of integers.

    See Also
    --------
    jet_to_dual_jet

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.fitting as fitting
    >>> import t3toolbox.safety as safety
    >>> np.random.seed(0)
    >>> r = np.random.randn(3, 4)                           # apply at (2,): F + W = (3, 4)
    >>> weight = np.array([1.0, 2.0, 10.0])
    >>> dual = fitting.jet_to_dual_jet('apply', (2,), r, weight)
    >>> print(bool(np.allclose(fitting.dual_jet_to_jet('apply', (2,), dual, weight), r)))
    True

    A zero weight raises in safe mode, and divides by zero in unsafe mode:

    >>> zero_weight = np.array([1.0, 0.0, 10.0])
    >>> fitting.dual_jet_to_jet('apply', (2,), dual, zero_weight)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: dual_jet_to_jet: the squared weight is zero in 1 entry, and a dual jet cannot be divided by zero (the jet is lost there). Run inside safety.unsafe() to divide anyway; the result is inf or nan in those slots
    >>> with safety.unsafe():
    ...     back = fitting.dual_jet_to_jet('apply', (2,), dual, zero_weight)
    >>> print(np.isfinite(back).all(axis=1).tolist())       # slot 1 is lost
    [True, False, True]
    """
    backend_jets._kind_and_order(kind, order, 'dual_jet_to_jet')     # the kind and the jet shape, before the check
    if weight is not None and safety.checks_active(dual_jet, weight):
        xnp, _, _ = get_backend(False, tree_contains_jax((dual_jet, weight)))
        divisors = backend_jets._squared_weights(kind, dual_jet, weight, xnp)  # weight**2, as the backend divides by it
        n_zero = sum(int(np.sum(np.asarray(divisor) == 0)) for divisor in divisors)
        safety.require(n_zero == 0,
                       'dual_jet_to_jet: the squared weight is zero in %d entr%s, and a dual jet cannot be divided '
                       'by zero (the jet is lost there). Run inside safety.unsafe() to divide anyway; the result '
                       'is inf or nan in those slots' % (n_zero, 'y' if n_zero == 1 else 'ies'))
    return backend_jets.dual_jet_to_jet(kind, order, dual_jet, weight)


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


@dataclass(frozen=True)
class LocalModel:
    '''The local model of a least-squares objective at a point, generic over the geometry: the objective value,
    the gradient, and the curvature actions there, with ``T3Tangent`` / ``UT3Tangent`` in and out. The frontend twin
    of the backend :py:class:`~t3toolbox.backend.optimizers.LocalModel`, which it holds (D6): ``geometry`` is the
    frontend geometry, ``backend`` the backend model (its oracle, batch, frame, centers and ``held``), and
    :py:attr:`frame` the frontend frame the tangents live at, derived from the backend one (a ``T3Frame`` or
    ``UT3Frame``; on a corewise geometry the ``(B, G, G, G)`` frame of the point). Build it with
    :py:func:`local_model`, not directly.

    **Two modes**, chosen by :py:func:`local_model`'s ``hessian`` and derived rather than stored
    (:py:attr:`is_exact`):

    * **Gauss-Newton** (``hessian='gauss_newton'``, the default): :py:meth:`hessian` / :py:meth:`quadratic`
      are :py:meth:`gn_hessian` / :py:meth:`gn_quadratic`.
    * **Exact** (``hessian='exact'``): :py:meth:`hessian` is the exact Riemannian Hessian -- the Gauss-Newton
      term plus the curvature term :py:meth:`curvature_hessian` (``Π Lᵀ(K_{Aᵀω²r} − K_{grad f}) L Π``), and
      :py:meth:`quadratic` likewise.

    ``gradient``, ``jacobian``, ``residual``, ``gn_hessian``, ``gn_quadratic``, ``objective_value`` and
    :py:meth:`retract` mean the same in both modes. Each action is one query of the oracle at the model's frame, on
    its batch, with the ``held`` of its build (:py:class:`~t3toolbox.backend.optimizers.LocalModel`). A trial step
    must be a tangent at :py:attr:`frame` (the same-frame guard, numerical, skipped under ``safety.unsafe()`` and a
    jax trace).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.fitting as fitting
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(15, N) for N in (6, 7, 8)]    # 15 samples, each a (w_1, w_2, w_3)
    >>> y = x.apply(ww) - np.random.randn(15)                # data whose residual at x is known
    >>> oracle = fitting.LeastSquaresOracle('apply', order=())
    >>> batch = fitting.LeastSquaresBatch(ww, y)

    On the **manifold** geometry the gradient ``g = Π 𝒥ᵀ r`` is a gauged tangent and ``H p`` stays on the
    gauged tangent space:

    >>> model = fitting.local_model(t3m.MANIFOLD, oracle, batch, x)
    >>> print(round(float(model.objective_value), 6))
    8.295196
    >>> print(model.gradient.is_gauged())
    True
    >>> p = t3m.MANIFOLD.randn(model.frame)                  # a gauged trial step at the model's frame
    >>> print(model.gn_hessian(p).is_gauged())
    True

    ``evaluate(p)`` is the quadratic-model value, consistent with assembling it from ``c``, ``g`` and the Hessian
    action:

    >>> hp = model.gn_hessian(p)
    >>> m_built = float(model.objective_value + model.gradient.corewise_inner(p) + 0.5 * p.corewise_inner(hp))
    >>> bool(np.allclose(float(model.evaluate(p)), m_built))
    True

    The Gauss-Newton quadratic form ``pᵀHp = ‖Jp‖²`` is one forward (no ``H p`` assembly):

    >>> bool(np.allclose(float(model.gn_quadratic(p)), float(p.corewise_inner(model.gn_hessian(p)))))
    True

    The **corewise** geometry is the same call with a different geometry: the gradient is a tangent at the
    raw ``(B,G,G,G)`` frame (a core perturbation), with **no** gauge projection:

    >>> cmodel = fitting.local_model(t3m.COREWISE, oracle, batch, x)
    >>> print(cmodel.gradient.is_gauged())
    False
    >>> cp = t3m.COREWISE.randn(cmodel.frame)
    >>> bool(np.allclose(float(cmodel.evaluate(cp)),
    ...                  float(cmodel.objective_value + cmodel.gradient.corewise_inner(cp)
    ...                        + 0.5 * cp.corewise_inner(cmodel.gn_hessian(cp)))))
    True

    The **uniform** layer is the same call with a uniform ``x`` (and a uniform geometry); the batch is packed for it:

    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    >>> umodel = fitting.local_model(ut3m.UNIFORM_MANIFOLD, oracle, batch, ux)
    >>> type(umodel).__name__, type(umodel.gradient).__name__
    ('LocalModel', 'UT3Tangent')
    >>> up = ut3m.UNIFORM_MANIFOLD.randn(umodel.frame)
    >>> bool(np.allclose(float(umodel.gn_quadratic(up)), float(up.corewise_inner(umodel.gn_hessian(up)))))
    True

    A trial step at a *different* frame raises (the model is tied to its frame):

    >>> other = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 3, 2), (1, 2, 2, 1))
    >>> model.gn_hessian(t3m.MANIFOLD.randn(t3m.MANIFOLD.frame(other)))   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: trial tangent must live at the model's frame (it is at a different frame)
    '''

    geometry:  typ.Any             # the frontend geometry (MANIFOLD / COREWISE / UNIFORM_* / a SharedGeometry)
    backend:   bopt.LocalModel     # the backend model: oracle, batch, frame, centers, held, f, ...

    # ------------------------------------------------------------------ what the model holds
    @property
    def oracle(self):
        '''The oracle (read-only; the backend model's).'''
        return self.backend.oracle

    @property
    def batch(self):
        '''The batch (read-only; packed on the uniform layer).'''
        return self.backend.batch

    @property
    def held(self):
        '''What the build's query holds for the model's later queries (this frame, this batch).'''
        return self.backend.held

    @ft.cached_property
    def frame(self):
        '''The frontend frame the tangents live at, derived from the backend frame: ``T3Frame(D, U, L, R)`` /
        ``UT3Frame``; on a corewise geometry the ``(B, G, G, G)`` frame of the point.'''
        return _frontend_frame(self.backend.frame)

    @property
    def is_exact(self) -> bool:
        '''Whether this model's Hessian is the exact one (``hessian='exact'``) rather than Gauss-Newton.'''
        return self.backend.is_exact

    def _require_exact(self, who: str) -> None:
        if not self.is_exact:
            raise ValueError("LocalModel.%s needs an exact-mode model (local_model(..., hessian='exact')); "
                             "this one is Gauss-Newton" % who)

    def _coordinates(self, p):                   # a trial tangent's raw coordinates, after the same-frame guard
        _require_at_frame(self.frame, p)
        return _coordinates_of(p)

    def _tangent(self, raw):                     # raw backend coordinates -> a tangent at frame
        return _tangent_at(self.frame, raw)

    # ------------------------------------------------------------------ the values
    @ft.cached_property
    def objective_value(self) -> NDArray:        # c = ½‖ω⊙r‖² (+ ρ(X)), shape C
        return self.backend.objective

    @property
    def residual(self):                          # r = z(X) - y, of the type of z (the oracle's optional output)
        return self.backend.residual

    @ft.cached_property
    def gradient(self):                          # g = Π 𝒥ᵀ ω² r (+ g_R), a tangent at frame
        return self._tangent(self.backend.gradient)

    def jacobian(
            self,
            p,      # a trial tangent at this model's frame
    ) -> typ.Any:   # J p = 𝒥(Π p), of the type of z with p's stack K
        return self.backend.jacobian(self._coordinates(p))

    def gn_quadratic(
            self,
            p,      # a trial tangent at this model's frame
    ) -> NDArray:   # pᵀ H p = ‖ω J p‖² (+ ⟨p, H_R p⟩), shape K+C
        return self.backend.gn_quadratic(self._coordinates(p))

    def gn_hessian(self, p):                     # H p = Π 𝒥ᵀ ω² 𝒥 Π p (+ H_R p): Gauss-Newton in either mode
        raw = self._coordinates(p)
        model = bopt._gauss_newton_view(self.backend) if self.is_exact else self.backend
        return self._tangent(model.hvp(raw))

    def hessian(self, p):                        # the mode's Hessian: gn_hessian, or the exact one
        return self._tangent(self.backend.hvp(self._coordinates(p)))

    def quadratic(self, p) -> NDArray:           # pᵀ H p of this mode, shape K+C
        return self.backend.quadratic(self._coordinates(p))

    def curvature_hessian(self, p):              # Π Lᵀ K_N L Π p, the exact Hessian's curvature term
        self._require_exact('curvature_hessian')
        return self._tangent(self.backend.curvature_hvp(self._coordinates(p)))

    def curvature_quadratic(self, p) -> NDArray:  # ⟨Πp, Lᵀ K_N L Πp⟩, shape K+C
        self._require_exact('curvature_quadratic')
        return self.backend.curvature_quadratic(self._coordinates(p))

    def evaluate(self, p) -> NDArray:            # m(p) = c + ⟨g, Π p⟩ + ½ pᵀ H p, shape K+C
        raw = self._coordinates(p)
        b = self.backend
        Pp = self._tangent(b.geom.project(b.frame, raw, aux=b.geom_aux))
        return self.objective_value + self.gradient.corewise_inner(Pp) + 0.5 * b.quadratic(raw)

    def retract(self, p):                        # the geometry's retraction from this frame: a frontend point
        _require_at_frame(self.frame, p)
        if isinstance(self.geometry, sg.SharedGeometry):        # the companion the build computed (geom.precompute)
            return self.geometry.retract(p, shared_data=self.backend.geom_aux)
        return self.geometry.retract(p)


def _frontend_frame(frame_data):
    '''The frontend frame of a backend frame: a manifold frame ``(D, U, L, R)`` or its uniform 6-tuple, as is; a
    corewise point ``(B, G)`` (or ``(B_sc, G_sc, shape, masks)``) as the ``(B, G, G, G)`` frame of the point.'''
    import t3toolbox.backend.sampling_variables as sv
    layout, call = sv.frame_form(frame_data)
    uniform = not isinstance(frame_data[0], (tuple, list))
    if call == 'corewise':
        if uniform:
            down, up, left, right, shape, masks = ufv_conversions.ut3_corewise_frame(frame_data)
            return ubv.UT3Frame(down, up, left, right, shape, ubv.UT3FrameMasks(*masks))
        return bvf.T3Frame(*fv_conversions.t3_corewise_frame(frame_data))
    if uniform:
        return ubv.UT3Frame(*frame_data[:5], ubv.UT3FrameMasks(*frame_data[5]))
    return bvf.T3Frame(*frame_data)


def _require_layer(geometry, x) -> None:
    '''A ragged point needs a ragged geometry, a uniform point a uniform one (structural).'''
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        if not (geometry is ut3m.UNIFORM_MANIFOLD or geometry is ut3m.UNIFORM_COREWISE
                or (isinstance(geometry, sg.SharedGeometry) and geometry.is_uniform)):
            raise ValueError("a UniformTuckerTensorTrain requires a uniform geometry "
                             "(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE, or a "
                             "shared_geometry.SharedGeometry over one).")
    elif isinstance(x, t3.TuckerTensorTrain):
        if not (geometry is t3m.MANIFOLD or geometry is t3m.COREWISE
                or (isinstance(geometry, sg.SharedGeometry) and not geometry.is_uniform)):
            raise ValueError("a ragged TuckerTensorTrain requires a ragged geometry (manifold.MANIFOLD / "
                             "manifold.COREWISE, or a shared_geometry.SharedGeometry over one); for a "
                             "UniformTuckerTensorTrain use the uniform geometries "
                             "(uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE, or a SharedGeometry "
                             "over one).")
    else:
        raise TypeError("x must be a TuckerTensorTrain or a UniformTuckerTensorTrain, got %s" % type(x).__name__)


def _is_manifold(geometry) -> bool:
    '''Whether ``geometry`` is (a shared wrapper over) a manifold geometry, ragged or uniform.'''
    if isinstance(geometry, sg.SharedGeometry):
        return geometry._is_manifold_kind
    return geometry is t3m.MANIFOLD or geometry is ut3m.UNIFORM_MANIFOLD


def _sharing_of(geometry):
    return geometry.sharing if isinstance(geometry, sg.SharedGeometry) else None


def backend_geometry(
        geometry,                    # a frontend geometry
        x,                           # the point (ragged or uniform): its shape, and on the uniform layer its masks
        frame_data:  typ.Any = None, # uniform manifold: the frame of x, for the geometry's masks (from_frame)
):                                   # the backend geometry (backend.geometry.*GeometryOps)
    '''The backend geometry of a frontend geometry at ``x`` (private to the frontend; the optimizers and
    :py:func:`local_model` share it). On the uniform layer the masks are ``x``'s: ``from_point`` for a corewise
    geometry; for a manifold one ``from_frame`` when ``frame_data`` is given (a local model at a point that need not
    be minimal, D5's path), else ``from_point`` (which requires a minimal point, D5).'''
    _require_layer(geometry, x)
    sharing = _sharing_of(geometry)
    if isinstance(x, t3.TuckerTensorTrain):
        return ragged_backend_geometry(geometry, x.shape)
    if _is_manifold(geometry):
        if frame_data is not None:
            return bgeo.UniformManifoldGeometryOps.from_frame(frame_data, x.masks.data, sharing)
        return bgeo.UniformManifoldGeometryOps.from_point(x.data, sharing)
    return bgeo.UniformCorewiseGeometryOps.from_point(x.data, sharing)


def _tied(geometry, x):
    '''``x`` with its Tucker factors tied by the geometry's sharing (a shared geometry's frame describes a point on
    the shared set; :py:meth:`~t3toolbox.shared_geometry.SharedGeometry.frame`); ``x`` itself otherwise.'''
    sharing = _sharing_of(geometry)
    if sharing is None:
        return x
    geometry.groups(x.shape)                                         # structural validation
    import t3toolbox.backend.sharing as backend_sharing
    if isinstance(x, ut3.UniformTuckerTensorTrain):
        return ut3m._ut3_from_data(backend_sharing.ut3_tie_tucker_factors(x.data, sharing))
    return t3.TuckerTensorTrain(*backend_sharing.t3_tie_tucker_factors(x.data, sharing))


def _is_packed(oracle, batch) -> bool:
    '''Whether a least-squares batch is already packed for the uniform layer (its sample a supercore, not a list).
    Asked only for a uniform point: a ragged point's batch is always checked.'''
    return (getattr(oracle, 'kind', None) in ('probe', 'apply') and isinstance(batch, LeastSquaresBatch)
            and not isinstance(batch.sample, (tuple, list)))


def prepare_batch(
        oracle,
        batch,
        x,                           # the point: its shape, and on the uniform layer its width N
        check:  bool = True,         # False: pack only (a draw's batch, made by the caller's draw every step)
):                                   # the batch the backend takes (packed on the uniform layer)
    '''The frontend's batch handling, the one place for it (private to the frontend; the optimizers and
    :py:func:`local_model` share it): ``oracle.check_batch(batch, x.shape)`` when the oracle has it (structural,
    before any work; O-18), then, for a uniform ``x``, a ``LeastSquaresBatch`` packed with
    :py:func:`~t3toolbox.backend.uniform_fitting.pack_batch`. A batch already packed for a uniform ``x`` is neither
    checked (the checker reads the ragged form) nor packed again; another oracle's batch passes unchanged.'''
    if batch is None:
        return None
    uniform = isinstance(x, ut3.UniformTuckerTensorTrain)
    packed = uniform and _is_packed(oracle, batch)
    if check and hasattr(oracle, 'check_batch') and not packed:
        oracle.check_batch(batch, tuple(x.shape))
    if uniform and isinstance(batch, LeastSquaresBatch) and not packed:
        return ufit.pack_batch(oracle.kind, oracle.order, batch, x.N)
    return batch


def check_given_frame(
        geometry,                    # a manifold geometry (a corewise one has no frame of its own: the point)
        frame,                       # a T3Frame / UT3Frame the caller supplied
        x,                           # the point it must be a frame of
        who:  str,
) -> None:
    '''Safe mode (O-5): a frame the caller supplies is checked once, numerically: that it is orthonormal and
    consistent (``orthogonality_residual`` and ``consistency_residual`` within ``safety.effective_rtol``, so a
    float32 frame passes), that it represents ``x``, and on a shared geometry that its down cores are tied under the
    geometry's sharing. Skipped in unsafe mode and under a jax trace. Its type is checked always (structural).'''
    uniform = isinstance(x, ut3.UniformTuckerTensorTrain)
    frame_type = ubv.UT3Frame if uniform else bvf.T3Frame
    if not isinstance(frame, frame_type):
        raise TypeError('%s: the frame must be a %s for a %s, got %s'
                        % (who, frame_type.__name__, type(x).__name__, type(frame).__name__))
    data = frame.data
    arrays = data[:4] if uniform else data[0]
    if not safety.checks_active(*arrays):
        return
    if uniform:
        base = ut3.UniformTuckerTensorTrain(data[0], data[2], tuple(data[4]), ut3.UT3Masks(data[5][0], data[5][2]))
    else:
        base = t3.TuckerTensorTrain(data[0], data[2])
    rtol = safety.effective_rtol(*arrays)
    safety.require(bool(np.all(frame.is_orthogonal(atol=rtol))), '%s: the frame is not orthonormal' % who)
    safety.require(bool(np.all(np.asarray(frame.consistency_residual) <= rtol)),
                   '%s: the frame is not consistent (its left and right cores do not describe one point through '
                   'its down cores; consistency_residual above %.1e)' % (who, rtol))
    sharing = _sharing_of(geometry)
    if sharing is not None:
        safety.require(bool(np.all(base.has_shared_tucker_factors(sharing, rtol=rtol))),
                       '%s: the frame\'s down cores are not tied under the geometry\'s sharing %r (a shared '
                       'geometry\'s frame is the frame of a tied point)' % (who, tuple(sharing)))
    safety.require(bool(np.all(base.allclose(x))), '%s: the frame does not represent the point' % who)


def local_model(
        geometry,                                  # MANIFOLD / COREWISE / UNIFORM_* / a SharedGeometry over one
        oracle,                                    # e.g. LeastSquaresOracle('probe', order=(3,))
        batch,                                     # e.g. LeastSquaresBatch(affine_jet(ww, pp), y, weight)
        x,                                         # the point: a TuckerTensorTrain or a UniformTuckerTensorTrain
        regularizer:        typ.Any = None,        # e.g. IdentityRegularizer(lam)
        hessian:            str = 'gauss_newton',  # or 'exact'
        curvature_damping:  typ.Optional[float] = None,   # exact only; None = the geometry's default
        frame:              typ.Any = None,        # manifold only: a T3Frame / UT3Frame of x the caller supplies (O-5)
) -> LocalModel:                                   # tangents in and out, the same-frame guard
    '''The local model of ``oracle`` on ``batch`` at ``x`` through ``geometry`` (:py:class:`LocalModel`).

    It checks the batch (``oracle.check_batch``, when the oracle has it), packs it for a uniform ``x``, and in safe
    mode checks a ``frame`` the caller gives (:py:func:`check_given_frame`: orthonormal, consistent, tied on a shared
    geometry, representing ``x``), then calls the
    backend :py:func:`~t3toolbox.backend.optimizers.local_model`. The model keeps the user's point: no reduction to
    minimal ranks (the tangents the user builds live at its frame). A shared geometry ties ``x``'s factors first, as
    its ``frame`` does. A uniform manifold model is built from the frame (``from_frame``), so a non-minimal ``x``
    works (D5). ``frame=`` is for a manifold geometry; on a corewise one the frame is the point itself, and a given
    frame raises (structural).

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.fitting as fitting
    >>> np.random.seed(1)
    >>> target, x = (t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)) for _ in range(2))
    >>> ww, pp = ([np.random.randn(30, N) for N in (4, 5, 6)] for _ in range(2))
    >>> jet = fitting.affine_jet(ww, pp)                       # the lines ww + s pp
    >>> oracle = fitting.LeastSquaresOracle('probe', order=(1,))
    >>> batch = fitting.LeastSquaresBatch(jet, target.probe(jet, (1,)))
    >>> model = fitting.local_model(t3m.MANIFOLD, oracle, batch, x, hessian='exact')
    >>> p = t3m.MANIFOLD.randn(model.frame)
    >>> bool(np.isclose(float(model.quadratic(p)), float(model.gn_quadratic(p) + model.curvature_quadratic(p))))
    True

    A bad batch raises before any work:

    >>> fitting.local_model(t3m.MANIFOLD, oracle, fitting.LeastSquaresBatch(ww, batch.y), x)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ...
    '''
    _require_layer(geometry, x)
    uniform, manifold = isinstance(x, ut3.UniformTuckerTensorTrain), _is_manifold(geometry)
    if frame is not None and not manifold:
        raise ValueError('local_model: frame= needs a manifold geometry (MANIFOLD / UNIFORM_MANIFOLD, or a shared '
                         'wrapper over one); on a corewise geometry the frame is the point itself: pass frame=None')
    if frame is not None:
        check_given_frame(geometry, frame, x, 'local_model')
    else:
        x = _tied(geometry, x)
    bbatch = prepare_batch(oracle, batch, x)
    frame_data = centers = None
    if frame is not None:                                                # the caller's frame, without centers
        frame_data = frame.data
    elif uniform and manifold:                                           # D5's path: the geometry from the frame
        frame_data, variations = ufv_conversions.ut3_orthogonal_representations(x.data)
        centers = (variations[0], variations[1])
    geom = backend_geometry(geometry, x, frame_data if uniform and manifold else None)
    bmodel = bopt.local_model(geom, oracle, bbatch, (x.data[0], x.data[1]), regularizer, hessian, curvature_damping,
                              frame=frame_data, centers=centers)
    return LocalModel(geometry, bmodel)


if jax_available:
    import jax

    # Register LocalModel as a jax pytree: the backend model is the one child (a registered pytree: its data are
    # leaves, its geometry, oracle and regularizer aux, value-hashed), the frontend geometry the aux. The frontend
    # frame is derived from the backend frame, so a model crossing a jit boundary does not recompile when the
    # frame changes.
    jax.tree_util.register_pytree_node(
        LocalModel,
        lambda m: ((m.backend,), m.geometry),
        lambda geometry, children: LocalModel(geometry, children[0]))
