# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Frontend adapter for the geometry-agnostic optimizers.

A thin wrapper over :py:mod:`t3toolbox.backend.optimizers` (where the algorithms live -- check-free, so a
raw-``.data`` user can call them directly). Each optimizer takes ``(geometry, oracle, batch, x0, ...)``: the
frontend geometry, the objective's **oracle** (e.g. :py:class:`~t3toolbox.fitting.LeastSquaresOracle`), its **batch**
of data (e.g. :py:class:`~t3toolbox.fitting.LeastSquaresBatch`) and the start. This layer (1) checks the batch
(``batch.check`` on a ragged batch, when the batch has it; structural, before any work), (2) maps the
geometry to the backend one and reduces ``x0`` to minimal ranks (a manifold geometry; always on the uniform layer),
(3) packs the batch for a uniform ``x0``, (4) calls the backend optimizer on the raw cores, and (5) re-wraps the
result. Design: ``docs/contributor/fitting_internals.md`` §"Backend-first".

**Two representations, inferred from** ``x0`` (the library-wide ragged/uniform dispatch):

* a **ragged** ``TuckerTensorTrain`` x0 with a ragged geometry (``manifold.MANIFOLD`` / ``COREWISE``) --
  the optimizer runs on the raw cores; the frame is re-orthogonalized internally each step.
* a **uniform** ``UniformTuckerTensorTrain`` x0 with a uniform geometry
  (``uniform_manifold.UNIFORM_MANIFOLD`` / ``UNIFORM_COREWISE``) -- the optimizer runs on the packed
  supercore pair (``lax.scan`` over the mode axis; the speed path). The uniform path requires a
  **minimal-rank** start; it calls :py:func:`~t3toolbox.backend.uniform_fitting.uniform_minimal`
  transparently so a frontend user never meets that requirement.

The geometry must match ``x0``'s representation (a uniform x0 with a ragged geometry, or vice versa, is a
structural error). The result is returned in the same representation as ``x0``.

**``use_jit=True``** (an explicit kwarg on ``mc_sgd`` / ``adam`` / ``newton_cg``) opts into jax: the
optimizer **auto-converts** ``x0`` and the batch onto jax and jit-compiles, so the result comes
back **jax-backed** in jax's default float32 (enable jax x64 for float64); if jax is not installed it
runs eager on numpy with a one-time warning (the jax-absent policy, ``backend.common.jax_or_warn``). See
:py:func:`t3toolbox.backend.optimizers.newton_cg`.

    >>> # x_opt, stats = optimizers.newton_cg(MANIFOLD, oracle, batch, x0)       # ragged
    >>> # x_opt, stats = optimizers.newton_cg(UNIFORM_MANIFOLD, oracle, batch, ux0)  # uniform

**The old call form** ``(geometry, kind, sample, data, x0, order=, weight=, ...)`` (2026.5.0) is still accepted for
this release and translated to the new one (a string where the oracle goes marks it); it is removed in the next.
"""
import dataclasses as dc
import functools as ft
import inspect
import typing as typ
import warnings

import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.fitting as _fitting
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.optimizer_display as bdisp
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.jets as jets
import t3toolbox.safety as safety
import t3toolbox.backend.common as common
from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch

# Regularizers live in the backend (check-free; a raw-.data user constructs them directly) and are
# re-exported here for frontend convenience -- neither user is privileged (``docs/contributor/fitting_internals.md`` §Regularization).
from t3toolbox.backend.regularization import Regularizer, IdentityRegularizer

__all__ = [
    'gradient_descent',
    'mc_sgd',
    'adam',
    'newton_cg',
    'Regularizer',
    'IdentityRegularizer',
]

# A frontend point in either representation. The four optimizers infer ragged (TuckerTensorTrain) vs
# uniform (UniformTuckerTensorTrain) from x0's type, and require a matching geometry.
Point = typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain]


def _ragged_minimal(x0, sharing):
    """Reduce a ragged manifold start to minimal ranks (the SAME tensor) -- the mirror of
    :py:func:`~t3toolbox.backend.uniform_fitting.uniform_minimal`. A non-minimal ``x0`` is legal, but the
    first retraction drops the redundant rank anyway, and an optimizer that allocates state at ``x0``'s
    core shapes (adam's moments) then broke at step 2. With ``sharing`` the SHARED-minimal notion applies.
    No-op when already minimal, so a minimal start is untouched (same gauge, same trajectory)."""
    minimal = t3.TuckerTensorTrain.get_minimal_ranks(x0.shape, x0.tucker_ranks, x0.tt_ranks, sharing=sharing)
    if (tuple(x0.tucker_ranks), tuple(x0.tt_ranks)) == (tuple(minimal[0]), tuple(minimal[1])):
        return x0
    y, _, _ = x0.t3svd(sharing=sharing)                               # lossless, left-orthogonal
    return y.rank_adjustment_sweep('right_to_left', sharing=sharing)  # minimal (the t3svd output is left-orthogonal)


@dc.dataclass(frozen=True, eq=False)
class _Setup:
    geom:      typ.Any       # the backend geometry
    x0:        Point         # x0 reduced (the same tensor), frontend
    x0_cores:  typ.Any       # its backend cores: .data (ragged) / the bare supercore pair (uniform)
    batch:     typ.Any       # the batch the backend takes (packed on the uniform layer)

    def rewrap(self, cores) -> Point:                # backend cores -> the frontend point, in x0's representation
        if isinstance(self.x0, ut3.UniformTuckerTensorTrain):
            return ut3.UniformTuckerTensorTrain(cores[0], cores[1], self.x0.shape, self.x0.masks)
        return t3.TuckerTensorTrain(*cores)


def _setup(geometry, batch, x0: Point) -> _Setup:
    """The frontend's work before a backend optimizer: the layer check, the batch check (structural, before any
    work), x0's reduction to minimal ranks, the backend geometry, the batch packed for a uniform x0 (the reduction
    keeps the packed width N, so packing at the given x0 is packing at the reduced one)."""
    _fitting._require_layer(geometry, x0)
    bbatch = _fitting.prepare_batch(batch, x0)
    sharing = _fitting._sharing_of(geometry)
    if isinstance(x0, ut3.UniformTuckerTensorTrain):
        x0 = uf.uniform_minimal(x0, sharing=sharing)          # transparent SHARED-minimal reduction (no-op if minimal)
    elif _fitting._is_manifold(geometry):
        x0 = _ragged_minimal(x0, sharing)                     # mirror of uniform_minimal (no-op if minimal)
    return _Setup(_fitting.backend_geometry(geometry, x0), x0, (x0.data[0], x0.data[1]), bbatch)


def _same_kind_and_order(
        batch,                       # the full batch: a LeastSquaresBatch, or a user's type (getattr, None if absent)
        other,                       # a drawn or validation batch, likewise
        who:   str,                  # for the message
) -> None:
    """One kind and one jet shape per fit: each batch is read under its own, so a second batch of another kind or
    order would fit another objective with no error. Structural (two static tuples); a ``ValueError`` naming both."""
    full = (getattr(batch, 'kind', None), getattr(batch, 'order', None))
    given = (getattr(other, 'kind', None), getattr(other, 'order', None))
    if given != full:
        raise ValueError('%s is of kind %r and order %r, the batch of kind %r and order %r: a fit has one kind and one '
                         'jet shape' % ((who,) + given + full))


def _frontend_draw(draw: typ.Optional[typ.Callable], batch, x0: Point) -> typ.Optional[typ.Callable]:
    """Each batch a user's draw returns is compared with the full batch, its kind and its jet shape (structural, two
    tuples per draw: a draw of another kind or order would fit another objective), and for a uniform x0 packed (not
    re-checked each step)."""
    if draw is None:
        return draw
    uniform = isinstance(x0, ut3.UniformTuckerTensorTrain)

    def checked(rng):
        drawn = draw(rng)
        _same_kind_and_order(batch, drawn, 'the batch a draw returned')
        return _fitting.prepare_batch(drawn, x0, check=False) if uniform else drawn
    return checked


# --------------------------------------------------------------------------------------------------
# The old call form (2026.5.0), translated (P16; removed in P17)
# --------------------------------------------------------------------------------------------------
_OLD_PLAIN = ('probe', 'apply', 'entries')
_OLD_DERIVATIVE = ('probe_derivatives', 'apply_derivatives', 'entries_derivatives')


def _old_gradient_descent(geometry, kind, sample, data, x0, order=None, weight=None, regularizer=None,
                          chunk_size='auto', g0_reference=None, **kwargs):
    pass


def _old_mc_sgd(geometry, kind, sample, data, x0, rng, batch, order=None, weight=None, draw=None, use_jit=False,
                regularizer=None, chunk_size='auto', **kwargs):
    pass


def _old_newton_cg(geometry, kind, sample, data, x0, order=None, weight=None, verbose=False, val_sample=None,
                   val_data=None, callback=None, use_jit=False, regularizer=None, chunk_size='auto',
                   g0_reference=None, g0norm_newton=None, g0norm_cg=None, cg_forcing_power=0.5, max_newton=30,
                   gtol_rel=1e-8, cg_maxiter=200, c_armijo=1e-4, on_line_search_failure='stop',
                   hessian='gauss_newton', curvature_damping=None, step_control=None, tr_radius0=None, **keywords):
    pass


_OLD_SIGNATURES = {'gradient_descent': inspect.signature(_old_gradient_descent),
                   'mc_sgd': inspect.signature(_old_mc_sgd),
                   'adam': inspect.signature(_old_mc_sgd),
                   'newton_cg': inspect.signature(_old_newton_cg)}


def _shared_by_the_stack(
        kind:         str,                   # the new kind: 'probe' | 'apply' (the data's type)
        data:         typ.Any,               # the old data, with no stack C: probe d arrays, else one array
        stack_shape:  typ.Tuple[int, ...],   # C, the point's stack
) -> typ.Any:                                # the data with len(C) axes of 1 in the place of C
    """The old data were one dataset broadcast over a stacked point; in the new form a dataset shared by the stack has
    size-1 ``C`` axes, before the probe kind's mode edge and last for the others."""
    if not stack_shape:
        return data
    ones = (1,) * len(stack_shape)
    if kind == 'probe':
        return [np.reshape(a, np.shape(a)[:-1] + ones + np.shape(a)[-1:]) for a in data]
    return np.reshape(data, np.shape(data) + ones)


def _old_fit(
        kind:         str,                 # the old kind name ('probe', 'apply_derivatives', ...), case-insensitive
        sample:       typ.Any,             # ww / index, or (ww, pp) / (index, pp)
        data:         typ.Any,             # the old data (derivatives for a derivative kind)
        shape:        typ.Tuple[int, ...], # the mode sizes
        order:        typ.Optional[int],   # derivative kinds: the highest order
        weight:       typ.Any,             # the old residual weight, or None
        who:          str,
        stack_shape:  typ.Tuple[int, ...] = (),   # the point's stack C (the old data are shared by it)
) -> typ.Tuple[LeastSquaresOracle, LeastSquaresBatch]:
    """The oracle and the batch of an old-form fit. Plain kinds: jet shape ``()``; the derivative kinds at order
    ``n``: ``(n,)`` (``(0,)`` at order 0: the old data carry an order axis of length 1), the sample ``(ww, pp)`` the
    lines ``affine_jet(ww, pp)`` (``entries_derivatives``: the apply kind at ``affine_jet(index_to_one_hot(index,
    shape), pp)``), the derivative data and weight through ``prepare_derivative_data`` (the old objective exactly).
    The old weight goes through the old canonicalization (``_canonical_weight``), is broadcast to one value per block
    (probe ``(d,)`` or ``(d, n+1)``, else ``(n+1,)``), and is placed on the data by ``blocks_to_weight``, a weight of
    the type of ``z``. The old data, one dataset for a stacked point, gain size-1 ``C`` axes (shared by the stack)."""
    kind = str(kind).lower()
    d = len(shape)
    if kind in _OLD_PLAIN:
        if weight is not None and kind in ('apply', 'entries'):
            raise ValueError(f"the plain '{kind}' kind takes no residual weight (no mode or order axis); "
                             "per-mode weighting is defined for probe, per-order for the derivative kinds.")
        wm = _fitting._canonical_weight(weight, kind, d, 0)       # (1, 1) or (m >= d, 1): one row broadcasts
        data = _shared_by_the_stack(kind, data, stack_shape)
        w = None if wm is None else jets.blocks_to_weight(kind, (), np.ascontiguousarray(
            np.broadcast_to(wm[:d, 0], (d,))), data)
        return LeastSquaresOracle(), LeastSquaresBatch(kind, (), sample, data, weight=w)
    if kind not in _OLD_DERIVATIVE:
        raise ValueError(f"unknown sampling kind {kind!r}; expected one of "
                         f"{sorted(_OLD_PLAIN) + sorted(_OLD_DERIVATIVE)}")
    if order is None:
        raise ValueError(f"derivative kind {kind!r} requires order=")
    n = int(order)
    base = kind[:-len('_derivatives')]
    wm = _fitting._canonical_weight(weight, kind, d, n)
    w = None
    if wm is not None:
        w = (np.broadcast_to(wm[:d] if wm.shape[0] >= d else wm, (d, n + 1)) if base == 'probe'
             else np.broadcast_to(wm[0], (n + 1,)))
        w = np.ascontiguousarray(w)
    main, pp = sample
    if base == 'entries':
        base, main = 'apply', jets.index_to_one_hot(main, shape)
    data = _shared_by_the_stack(base, data, stack_shape)
    w = None if w is None else jets.blocks_to_weight(base, (n,), w, data)          # on the derivatives, then converted
    y, w, _ = jets.prepare_derivative_data(base, (n,), data, w)
    return LeastSquaresOracle(), LeastSquaresBatch(base, (n,), jets.affine_jet(main, pp), y, weight=w)


def _old_draw(draw, kind, shape, order, weight, scale, who, stack_shape=()):
    """An old custom draw ``draw(rng) -> (sample_B, data_B)`` as a new one returning a batch: the full per-draw
    conversion (the jets, the data and the weight), with ``minibatch_scale = n / min(batch, n)``, the 2026.5.0
    nominal factor."""
    def new_draw(rng):
        sample_b, data_b = draw(rng)
        _, b = _old_fit(kind, sample_b, data_b, shape, order, weight, who, stack_shape)
        return dc.replace(b, minibatch_scale=scale)
    return new_draw


def _translate(name, args, kwargs):
    """The old call form's arguments, bound to the old signature, as the new form's ``(args, kwargs)``."""
    bound = _OLD_SIGNATURES[name].bind(*args, **kwargs)
    bound.apply_defaults()
    a = dict(bound.arguments)
    extra = dict(a.pop('kwargs', None) or a.pop('keywords', None) or {})
    geometry, x0 = a['geometry'], a['x0']
    shape = tuple(x0.shape)
    stack = tuple(x0.stack_shape)
    oracle, batch = _old_fit(a['kind'], a['sample'], a['data'], shape, a['order'], a['weight'], name, stack)
    new_kw = dict(regularizer=a['regularizer'])
    if name == 'gradient_descent':
        new_kw.update(g0_reference=a['g0_reference'])
        return (geometry, oracle, batch, x0), {**new_kw, **extra}
    if name in ('mc_sgd', 'adam'):
        draw = a['draw']
        if draw is not None:
            n = batch.n_measurements
            draw = _old_draw(draw, a['kind'], shape, a['order'], a['weight'], n / min(int(a['batch']), n), name,
                             stack)
        new_kw.update(draw=draw, use_jit=a['use_jit'])
        return (geometry, oracle, batch, x0, a['rng'], a['batch']), {**new_kw, **extra}
    if (a['val_sample'] is None) != (a['val_data'] is None):
        raise ValueError("newton_cg: pass val_sample and val_data together, or neither (val_sample=%s, val_data=%s)"
                         % ('None' if a['val_sample'] is None else 'given', 'None' if a['val_data'] is None else 'given'))
    val_batch = None
    if a['val_sample'] is not None:
        _, val_batch = _old_fit(a['kind'], a['val_sample'], a['val_data'], shape, a['order'], None, name, stack)
    for key in ('verbose', 'callback', 'use_jit', 'g0_reference', 'g0norm_newton', 'g0norm_cg', 'cg_forcing_power',
                'max_newton', 'gtol_rel', 'cg_maxiter', 'c_armijo', 'on_line_search_failure', 'hessian',
                'curvature_damping', 'step_control', 'tr_radius0'):
        new_kw[key] = a[key]
    new_kw['val_batch'] = val_batch
    return (geometry, oracle, batch, x0), {**new_kw, **extra}


def _translates(new_fn):
    """The old call form of an optimizer, translated (P16, for this release only; P17 deletes it). A string where the
    oracle goes, or ``kind=``, marks it; the old arguments are then read as the old signature names them, rebuilt into
    the new form (:py:func:`_translate`), and the new function called. ``chunk_size`` is accepted and dropped."""
    @ft.wraps(new_fn)
    def wrapper(*args, **kwargs):
        if (len(args) > 1 and isinstance(args[1], str)) or 'kind' in kwargs:
            args, kwargs = _translate(new_fn.__name__, args, kwargs)
        return new_fn(*args, **kwargs)
    return wrapper


# --------------------------------------------------------------------------------------------------
# The optimizers
# --------------------------------------------------------------------------------------------------
@_translates
def gradient_descent(
        geometry,                       # ragged or uniform geometry (must match x0's representation)
        oracle,                         # the objective, e.g. LeastSquaresOracle()
        batch,                          # its data, e.g. LeastSquaresBatch('probe', (), ww, y)
        x0:       Point,                # initial point (any cores; the geometry orthogonalizes internally)
        regularizer: typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)
        g0_reference: typ.Optional[str] = None,   # the stop's ‖g0‖: None = the geometry's own ('zero' on manifolds); 'zero' | 'warm_start' | 'cone'
        **kwargs,                       # forwarded to backend.optimizers.gradient_descent (n_iter, gtol_rel, ...)
) -> typ.Tuple[Point, dict]:            # (x_opt, stats)
    """Fit by steepest descent (Cauchy step + Armijo line search) on ``geometry``: minimize ``oracle``'s objective on
    ``batch`` (plus ``regularizer``'s ``ρ(x)``).

    Accepts a ragged ``TuckerTensorTrain`` (with ``manifold.MANIFOLD`` / ``COREWISE``) or a uniform
    ``UniformTuckerTensorTrain`` (with ``uniform_manifold.UNIFORM_MANIFOLD`` / ``UNIFORM_COREWISE``); the
    representation is inferred from ``x0`` and returned in kind. **Eager only**: there is no ``use_jit`` here (that
    kwarg lives on :py:func:`mc_sgd` / :py:func:`adam` / :py:func:`newton_cg`); this is the reference/pedagogical
    optimizer, ``newton_cg`` the performance path. ``on_line_search_failure`` ('stop' default / 'accept') controls
    Armijo-exhaustion behavior as in :py:func:`newton_cg`, and ``g0_reference`` picks the reference ``‖g0‖`` of the
    relative stop as there (by default the zero tensor of ``x0``'s ranks on a manifold geometry).
    ``g0_reference='cone'`` and its six ``cone_*`` options (``cone_oversampling``, ``cone_oversampled_passes``,
    ``cone_max_passes``, ``cone_tol``, ``cone_alternate``, ``cone_seed``) reach the backend through ``**kwargs``;
    they are as in :py:func:`newton_cg`, and the alignment runs eagerly here. See
    :py:func:`t3toolbox.backend.optimizers.gradient_descent`.

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.optimizers as optimizers
    >>> from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> np.random.seed(0)
    >>> A = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(60, N) for N in (4, 5, 6)]
    >>> x0 = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
    >>> x, stats = optimizers.gradient_descent(t3m.MANIFOLD, LeastSquaresOracle(),
    ...                                        LeastSquaresBatch('apply', (), ww, A.apply(ww)), x0, n_iter=20)
    >>> stats['losses'][-1] < 0.1 * stats['losses'][0]
    True
    """
    s = _setup(geometry, batch, x0)
    x_cores, stats = bopt.gradient_descent(s.geom, oracle, s.batch, s.x0_cores, regularizer,
                                           g0_reference=g0_reference, **kwargs)
    return s.rewrap(x_cores), stats


@_translates
def mc_sgd(
        geometry,                       # ragged/uniform MANIFOLD (intended) / COREWISE (must match x0)
        oracle,                         # the objective
        batch,                          # its full data (with n_measurements / take for the default draw)
        x0:       Point,                # initial point
        rng,                            # np.random.Generator -- passed to the draw each step
        batch_size: int,                # measurements per minibatch (the default flat draw's; unread with a draw)
        regularizer: typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ); not scaled
        draw:     typ.Optional[typ.Callable] = None,  # custom draw(rng) -> a batch; None = the flat draw
        use_jit:  bool = False,         # jit the per-step kernel: auto-converts x0 and the batch to jax -> a jax-backed float32 result; warns + runs eager if jax absent
        **kwargs,                       # forwarded to backend.optimizers.mc_sgd (max_iter, check_every, ...)
) -> typ.Tuple[Point, dict]:
    """Manifold Cauchy SGD -- minibatched, tuning-free Cauchy step. Ragged or uniform ``x0`` (see
    :py:func:`gradient_descent`). Each step builds a model afresh on a drawn batch: the oracle scales the misfit to
    estimate the full one, and the ``regularizer`` is not scaled. A custom ``draw(rng)`` returns a batch of the full
    batch's kind and jet shape, checked on each draw (another raises a ``ValueError``); ragged, it is packed for a
    uniform ``x0``. See :py:func:`t3toolbox.backend.optimizers.mc_sgd`."""
    s = _setup(geometry, batch, x0)
    x_cores, stats = bopt.mc_sgd(s.geom, oracle, s.batch, s.x0_cores, rng, batch_size, regularizer,
                                 draw=_frontend_draw(draw, batch, s.x0), use_jit=use_jit, **kwargs)
    return s.rewrap(x_cores), stats


@_translates
def adam(
        geometry,                       # ragged/uniform COREWISE (intended) / MANIFOLD (must match x0)
        oracle,                         # the objective
        batch,                          # its full data (with n_measurements / take for the default draw)
        x0:       Point,                # initial point
        rng,                            # np.random.Generator -- passed to the draw each step
        batch_size: int,                # measurements per minibatch (the default flat draw's; unread with a draw)
        regularizer: typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ); not scaled
        draw:     typ.Optional[typ.Callable] = None,  # custom draw(rng) -> a batch; None = the flat draw
        use_jit:  bool = False,         # jit the per-step kernel: auto-converts x0 and the batch to jax -> a jax-backed float32 result; warns + runs eager if jax absent
        **kwargs,                       # forwarded to backend.optimizers.adam (lr, max_iter, ...)
) -> typ.Tuple[Point, dict]:
    """Adam over the cores -- the dependency-free first-order method for the corewise geometry. Ragged or
    uniform ``x0`` (see :py:func:`gradient_descent`); the minibatch rule as in :py:func:`mc_sgd`. See
    :py:func:`t3toolbox.backend.optimizers.adam`."""
    if _fitting._is_manifold(geometry):
        warnings.warn("adam on a MANIFOLD geometry: Adam's per-coordinate moments live in the core "
                      "coordinates, which are gauge-dependent, so the iterates depend on the backend / "
                      "representation (measured ~1e-2 between numpy, jax and uniform). It runs, but COREWISE "
                      "is the intended geometry for adam; mc_sgd / newton_cg are the gauge-invariant "
                      "manifold optimizers.", RuntimeWarning, stacklevel=3)
    s = _setup(geometry, batch, x0)
    x_cores, stats = bopt.adam(s.geom, oracle, s.batch, s.x0_cores, rng, batch_size, regularizer,
                               draw=_frontend_draw(draw, batch, s.x0), use_jit=use_jit, **kwargs)
    return s.rewrap(x_cores), stats


def _padded_start_keywords(
        geometry,                              # the frontend geometry
        s:              _Setup,                # the backend setup (x0 reduced)
        x0:             Point,                 # the frontend start (as given)
        start_frame:    typ.Any,               # a T3Frame / UT3Frame, or None
        padded_from:    typ.Any,               # the old point, or None
        to_jax:         bool,
        g0_reference:   typ.Optional[str],
        options:        typ.Dict[str, typ.Any],   # the twelve padded_* / cone_* options by keyword (None = unset)
) -> typ.Dict[str, typ.Any]:                   # the backend keywords: only the set ones
    out = {k: v for k, v in options.items() if v is not None}
    cone = g0_reference is not None and str(g0_reference).lower() == 'cone'
    if (start_frame is not None or padded_from is not None or cone) and not _fitting._is_manifold(geometry):
        raise ValueError('newton_cg: start_frame, padded_from and g0_reference=\'cone\' need a manifold geometry '
                         '(MANIFOLD / UNIFORM_MANIFOLD, or a shared wrapper over one)')
    uniform = isinstance(x0, ut3.UniformTuckerTensorTrain)
    sharing = _fitting._sharing_of(geometry)
    if start_frame is not None:
        _fitting.check_given_frame(geometry, start_frame, s.x0, 'newton_cg: start_frame')
        data = start_frame.data
        if to_jax:
            data = (tuple(common.to_jax(a) for a in data[:4]) + tuple(data[4:]) if uniform
                    else common.tree_to_jax(data))
        out['start_frame'] = data
    if padded_from is not None:
        out['padded_from'] = _old_point_data(padded_from, x0, sharing)
    return out


def _old_point_data(
        x_old:    Point,                        # the frontend old point
        x0:       Point,                        # the frontend start (the zero-padded point)
        sharing:  typ.Optional[typ.Sequence],   # the geometry's sharing labels
) -> typ.Any:                                   # the backend data of x_old (reduced; a structurally zero x_old as is)
    """padded_from, checked against x0 (layer, shape, stack, N) and reduced to minimal ranks."""
    uniform = isinstance(x0, ut3.UniformTuckerTensorTrain)
    kind = ut3.UniformTuckerTensorTrain if uniform else t3.TuckerTensorTrain
    if not isinstance(x_old, kind):
        conversion = ('UniformTuckerTensorTrain.from_t3(padded_from, N=x0.N)' if uniform else 'padded_from.to_t3()')
        raise TypeError('newton_cg: padded_from must be a %s like x0, got %s (convert it: %s)'
                        % (kind.__name__, type(x_old).__name__, conversion))
    if tuple(x_old.shape) != tuple(x0.shape) or tuple(x_old.stack_shape) != tuple(x0.stack_shape):
        raise ValueError('newton_cg: padded_from has shape %r and stack %r; x0 has %r and %r'
                         % (tuple(x_old.shape), tuple(x_old.stack_shape), tuple(x0.shape), tuple(x0.stack_shape)))
    if uniform and x_old.N != x0.N:
        raise ValueError('newton_cg: padded_from is padded to N=%d, x0 to N=%d; repack it with '
                         'UniformTuckerTensorTrain.from_t3(padded_from.to_t3(), N=%d)' % (x_old.N, x0.N, x0.N))
    tucker, tt = np.asarray(x_old.tucker_ranks), np.asarray(x_old.tt_ranks)
    if np.any(tucker == 0) or np.any(tt[1:-1] == 0):                      # the zero tensor: no reduction (it breaks
        return x_old.data                                                 #   at rank 0); the members read it as zero
    arrays = x_old.supercores if uniform else x_old.data[0]
    if safety.checks_active(*arrays):
        if sharing is not None:
            safety.require(bool(np.all(x_old.has_shared_tucker_factors(sharing))),
                           'newton_cg: padded_from does not have tied Tucker factors under the geometry\'s sharing')
        safety.require(bool(np.all(x0.allclose(x_old))),
                       'newton_cg: x0 is not padded_from zero-padded (they represent different tensors)')
    return (uf.uniform_minimal(x_old, sharing) if uniform else _ragged_minimal(x_old, sharing)).data


@_translates
def newton_cg(
        geometry,                   # ragged/uniform MANIFOLD (intended) / COREWISE (must match x0)
        oracle,                     # the objective, e.g. LeastSquaresOracle()
        batch,                      # its data, e.g. LeastSquaresBatch('probe', (1,), affine_jet(ww, pp), y)
        x0:                     Point,    # initial point (zero is fine on the manifold)

        regularizer:            typ.Any = None,    # optional regularizer, e.g. optimizers.IdentityRegularizer(λ)

        verbose:                bool = False,    # print per-iteration diagnostics (a relative-error table + more)
        error_table:            bool = True,     # with verbose: False prints each row's header lines, no table
        val_batch:              typ.Any = None,  # optional validation batch, batch's kind and order -> a train|val table

        callback:               typ.Optional[typ.Callable] = None,  # custom callback(NewtonInfo) each iter (overrides `verbose`)

        use_jit:                bool = False,      # jit the inner CG: auto-converts x0 and the batch to jax -> a jax-backed float32 result; warns + runs eager if jax absent

        g0_reference:           typ.Optional[str]   = None,  # which ‖g0‖: None = the geometry's own ('zero' on manifolds, 'warm_start' on corewise); 'zero' | 'warm_start' | 'cone'
        g0norm_newton:          typ.Optional[float] = None,  # explicit ‖g0‖ for the Newton stop (wins; > 0); also feeds CG unless g0norm_cg is given
        g0norm_cg:              typ.Optional[float] = None,  # explicit ‖g0‖ for the CG forcing term (wins for CG; > 0); default = the Newton reference

        cg_forcing_power:       float = 0.5,   # η = min(0.5, (‖g‖/‖g0‖)**power); larger => tighter CG, fewer Newton steps
        max_newton:             int   = 30,    # Newton-iteration cap (rejected trust-region trials included)
        gtol_rel:               float = 1e-8,  # stop when ‖g‖ <= gtol_rel * ‖g0‖
        cg_maxiter:             int   = 200,   # inner CG iteration cap per Newton step
        c_armijo:               float = 1e-4,  # Armijo sufficient-decrease constant (line search only)
        on_line_search_failure: str = 'stop',  # Armijo exhausted: 'stop' = reject the step and terminate; 'accept' = take the last trial

        hessian:                str = 'gauss_newton',        # or 'exact' | 'switch' (GN, then exact); case-insensitive
        curvature_damping:      typ.Optional[float] = None,  # exact, switch: lift damping, relative to ‖X‖; None = the default
        step_control:           typ.Optional[str]   = None,  # None = 'line_search' (GN) / 'trust_region' (exact, switch)
        tr_radius0:             typ.Optional[float] = None,  # trust region only: Δ0 > 0; None = the GN Cauchy length
        *,
        start_frame:                typ.Optional[typ.Any] = None,   # a T3Frame / UT3Frame of x0 (after its reduction) for iteration 0
        padded_from:                typ.Optional[Point] = None,   # the old point; rank 0 = the zero tensor

        padded_oversampling:        typ.Optional[int]   = None,   # None = 5; padded_from only
        padded_oversampled_passes:  typ.Optional[int]   = None,   # None = 2
        padded_max_passes:          typ.Optional[int]   = None,   # None = 10
        padded_tol:                 typ.Optional[float] = None,   # None = 0.05
        padded_alternate:           typ.Optional[bool]  = None,   # None = True
        padded_seed:                typ.Optional[int]   = None,   # None = ZERO_FRAME_SEED

        cone_oversampling:          typ.Optional[int]   = None,   # None = 5; g0_reference='cone' without g0norm_newton only
        cone_oversampled_passes:    typ.Optional[int]   = None,   # None = 2
        cone_max_passes:            typ.Optional[int]   = None,   # None = 10
        cone_tol:                   typ.Optional[float] = None,   # None = 0.05
        cone_alternate:             typ.Optional[bool]  = None,   # None = True
        cone_seed:                  typ.Optional[int]   = None,   # None = ZERO_FRAME_SEED

        timing:                     bool = False,                 # True: stats['timing'], the seconds by phase (changes no number)

        switch_tau:                 typ.Optional[float] = None,   # None = 0.25; in [0, 1); 0 = never switch; 'switch' only
        switch_k:                   typ.Optional[int]   = None,   # None = 2; an int >= 0; 0 = exact from iteration 0
        switch_cg_forcing_power:    typ.Optional[float] = None,   # None = 1.0: the forcing power from the switch on
        switch_track:               typ.Optional[str]   = None,   # None = 'countable'; or 'all': the rows that track
) -> typ.Tuple[Point, dict]:
    """Inexact Riemannian Newton-CG with a line search or a trust region -- the manifold workhorse. Ragged or
    uniform ``x0`` (see :py:func:`gradient_descent`). See :py:func:`t3toolbox.backend.optimizers.newton_cg`.

    ``hessian`` picks the local model's Hessian: ``'gauss_newton'`` (the default) or ``'exact'``, the exact
    Riemannian Hessian -- the Gauss-Newton term plus the curvature term the Gauss-Newton model drops, which
    gives the superlinear Newton tail on a nonzero-residual fit (q-quadratic with ``cg_forcing_power=1.0``)
    at a few times the cost of a Gauss-Newton Hessian-apply. Its default step control is a trust region
    (``step_control='trust_region'``: Steihaug-Toint CG, one trial retraction per step, accepted on the
    model's predicted decrease), the Gauss-Newton one the Armijo line search; either combines with either.
    ``'switch'`` starts with Gauss-Newton and switches to the exact Hessian inside the run, once the exact model
    predicts the objective clearly better (below; its default step control is the trust region too).
    ``tr_radius0`` is the initial trust radius and ``curvature_damping`` the exact Hessian's lift damping
    (relative to ``‖X‖``; ``None`` = the geometry's default); each raises where it would have no effect. The
    exact Hessian is implemented for every sampling kind, on every geometry (``MANIFOLD`` / ``COREWISE``, their
    uniform twins, and the shared wrappers of all four); it needs an oracle that supplies ``Kp`` (the
    least-squares oracle does). Details,
    stop reasons and the trust-region rules: the backend :py:func:`t3toolbox.backend.optimizers.newton_cg`.

    Both stopping tests are relative to a reference ‖g0‖. By default (``g0_reference=None``) a manifold
    geometry uses the gradient norm at the **zero tensor of x0's ranks** -- the right scale after a
    warm start, whose own initial gradient is deflated -- and a corewise geometry the warm-start ‖g(x0)‖;
    ``g0_reference='warm_start'`` keeps the 2026.2.0 behaviour. The explicit overrides ``g0norm_newton`` /
    ``g0norm_cg`` win, and ``cg_forcing_power`` trades more CG iterations per Newton step for fewer Newton
    steps (all detailed on the backend; ``docs/fitting_and_optimization.md`` §5).

    The solver options -- ``max_newton``, ``gtol_rel``, ``cg_maxiter``, ``c_armijo``,
    ``on_line_search_failure``, the references above, ``cg_forcing_power``, ``hessian``,
    ``curvature_damping``, ``step_control`` and ``tr_radius0`` -- are the backend's, with the backend's
    defaults, passed straight through. They are explicit keywords (there is no ``**kwargs`` pass-through),
    so a misspelt option raises ``TypeError`` at the call, before the problem is built. The ``switch_*`` options
    (below) are the backend's too, checked at the call before the problem is built and forwarded only when set.

    ``verbose=True`` prints a per-iteration diagnostic block (objective / gradient, CG stats, line search,
    and the per-block relative-error table ``‖ω⊙r_b‖ / ‖ω⊙y_b‖``, in the block shape); pass ``val_batch`` (a batch of
    the same kind and jet shape as ``batch``; another raises a ``ValueError`` at entry) to add a validation column, and
    ``error_table=False`` to print the header lines only (the records are unchanged). Each table is under its own
    batch's weight: give ``val_batch`` a weight of the same form as ``batch``'s, or the training and validation cells
    are different quantities. With a ``regularizer`` attached the objective is shown split as ``obj = misfit + reg``
    (the ``½‖ω⊙r‖²`` data misfit vs ``ρ(x)``); both parts are on every record as ``misfit`` / ``regularization`` (the
    latter ``None`` when unregularized), in ``stats['history']`` (always) and ``stats['diagnostics']`` (when verbose).
    This is a thin convenience over the **backend** display -- a raw-``.data`` user builds the same callback with
    :py:func:`t3toolbox.backend.optimizer_display.make_newton_display` and passes it as ``callback=`` to
    :py:func:`t3toolbox.backend.optimizers.newton_cg`; this one hands the display the very batch the run uses (under
    ``use_jit`` the converted one), so each row reads the residual the model's build computed. A custom ``callback``
    overrides ``verbose``. Works on both the ragged and uniform layers (``batch`` and ``val_batch`` are packed for a
    uniform ``x0``).

    **After a rank increase** (``padded_from`` / ``start_frame`` / ``g0_reference='cone'``). Rank continuation
    restarts Newton from ``x0`` = the previous level's ``x_old`` zero-padded to the new ranks. The orthonormal frame of
    such a point is arbitrary on the new slots, so the first Gauss-Newton step can miss the directions the fit needs.
    ``padded_from=x_old`` builds iteration 0's frame by aligning the new slots with the misfit gradient
    (:py:func:`t3toolbox.backend.optimizers.padded_start`; the ``padded_*`` options: ``padded_max_passes=0`` is the
    hint-free padded frame). ``x_old`` must be ``x0``'s layer, shape, stack and padded ``N``; it is reduced to minimal
    ranks first, and a structurally zero ``x_old`` (any Tucker or interior TT rank 0) is the zero tensor. In safe
    mode ``x0`` must equal ``x_old`` (the zero padding) and, with sharing, ``x_old``'s factors must be tied.
    ``start_frame`` gives the frame directly (a ``T3Frame`` / ``UT3Frame`` of ``x0`` after its reduction; checked
    structurally, and in safe mode orthonormal, consistent, tied on a shared geometry and representing ``x0``).
    ``g0_reference='cone'`` takes the reference
    ``‖g0‖`` at the zero tensor at the frame aligned with the data gradient there (the ``cone_*`` options;
    ``cone_max_passes=0`` evaluates it at the zero tensor's hint-free start). ``use_jit`` governs the alignment as well as CG. These need a
    manifold geometry. Every set option is forwarded and unset ones are not, so without these keywords the run and
    its ``stats`` are exactly as before; with them ``stats`` gains ``'start_frame'``, ``'start_movements'`` and
    ``'cone_movements'``.

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.optimizers as optimizers
    >>> from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> np.random.seed(0)
    >>> shape = (5, 6, 5)
    >>> ww = [np.random.randn(80, n) for n in shape]
    >>> data = t3.TuckerTensorTrain.randn(shape, (3, 4, 3), (1, 3, 3, 1)).probe(ww)
    >>> oracle, batch = LeastSquaresOracle(), LeastSquaresBatch('probe', (), ww, data)
    >>> x_old, _ = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch,
    ...                                 t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)), max_newton=5)
    >>> x0 = x_old.resize(shape, (3, 4, 3), (1, 3, 3, 1))                  # the next level, zero-padded
    >>> x, stats = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=5, padded_from=x_old)
    >>> stats['start_frame'], len(stats['start_movements']) <= 10
    ('padded', True)

    **Switching from Gauss-Newton to the exact Hessian** (``hessian='switch'``). Gauss-Newton converges linearly on
    a nonzero-residual fit, at a rate set by the curvature term it drops; the exact Hessian gives the superlinear tail
    but costs a few Gauss-Newton Hessian-applies per apply and is indefinite far from a minimizer. ``'switch'`` makes
    the choice inside one call: it starts with Gauss-Newton, measures after each step whether the exact model would
    have predicted the actual change of the objective better, and switches to the exact Hessian for good once it
    clearly would have. With ``s`` the step taken (``αp`` on the line search, ``p`` in the trust region), ``Δf`` the
    objective's actual change, ``m_GN(s)`` the Gauss-Newton model's predicted change (``α·slope + ½α²·pHp`` in the
    row's fields) and ``⟨s, K s⟩`` the curvature term (the exact Hessian's quadratic form minus Gauss-Newton's;
    :py:func:`t3toolbox.backend.optimizers.newton_cg` has it in full)::

        m_N(s) = m_GN(s) + ½⟨s, K s⟩                      the exact model's predicted change: one curvature product
        e_GN   = |Δf − m_GN(s)|,   e_N = |Δf − m_N(s)|    how far each model missed
        hit    : e_N < switch_tau · e_GN                  on a countable step (below)
        switch : switch_k consecutive hits                for good, from the next point on

    With the defaults (``switch_tau=0.25``, ``switch_k=2``): **the exact model must have predicted the objective at
    least four times better than Gauss-Newton, on two clean steps in a row.** Near a minimizer ``e_GN`` is the
    dropped curvature term, second order in the step, while ``e_N`` is third order, so ``e_N / e_GN`` falls toward 0;
    in the transient far from it the ratio is about 1 or more, so a threshold well below 1, required twice running,
    rarely fires early.

    *What counts, what resets.* A step is countable iff the full model step was taken: an accepted trust-region
    trial, or an Armijo step with ``α = 1``. Everything else resets the count to 0: a miss, a rejected trust-region
    trial, a line-search backtrack, the ``−g`` fallback (CG's step was not a descent direction), a step forced by
    ``on_line_search_failure='accept'``, and a step whose two errors are both at the objective's rounding level. A CG
    step stopped on the trust region's boundary, truncated or at its iteration cap counts like any other (both models
    are evaluated at the step actually taken). The rounding gate: ``Δf`` is a difference of two objective values, so
    below ``1000·ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)`` (``ε`` the residual dtype's machine epsilon: the trust region's
    ``objective_floor`` scale times 100) neither error can be measured. In jax's default float32 (``use_jit=True``
    without x64) only large early steps clear it and the switch rarely fires: enable jax x64 with ``'switch'``.

    *The options*, each with ``hessian='switch'`` only (elsewhere a structural error):

    * ``switch_tau`` (``None`` = ``0.25``; in ``[0, 1)``) -- the hit threshold. ``0`` tracks and never switches: the
      run is the Gauss-Newton run with the same step control, bit for bit, plus the tracking.
    * ``switch_k`` (``None`` = ``2``; an int ``>= 0``) -- the consecutive hits that switch. ``0`` is exact from
      iteration 0: the exact run with ``cg_forcing_power=switch_cg_forcing_power``.
    * ``switch_cg_forcing_power`` (``None`` = ``1.0``) -- the CG forcing power from the switch on (``1.0`` gives the
      exact Hessian's q-quadratic tail); ``cg_forcing_power`` (default ``0.5``) stays the Gauss-Newton phase's.
    * ``switch_track`` (``None`` = ``'countable'``; or ``'all'``, case-insensitive) -- the steps the curvature
      product runs on: the countable ones (any other step resets the count whatever ``e_N`` is), or every
      Gauss-Newton-phase step, which reports ``e_N`` there too, at one product each, and switches at the same row.

    An option the run would not read raises too: ``switch_tau`` or ``switch_track`` with ``switch_k=0`` (nothing is
    tracked), ``switch_k`` or ``switch_cg_forcing_power`` with ``switch_tau=0`` (nothing switches). The default step
    control is the trust region, whose radius carries across the switch; the line search works too.
    ``curvature_damping`` applies to every exact build, the tracking ones included. Each call starts in Gauss-Newton
    with a count of 0 -- nothing carries across calls, so each level of a rank continuation decides afresh (after a
    rank increase the run may have left the basin). The switch may never fire, at no cost beyond the tracking: the
    curvature term vanishes (the manifold at ``d = 1``), the residual is zero, or the retraction is only first order
    at the point (near small singular values; at a zero-padded start).

    *What it costs.* Each Gauss-Newton-phase point is linearized once, in exact mode: one misfit transpose, as in
    either pure mode, plus the geometry's curvature data. CG, the step control and the row use its Gauss-Newton view,
    so the steps are a Gauss-Newton run's; each tracked step adds one curvature product, about two or three
    Gauss-Newton Hessian-applies. Under ``use_jit`` CG compiles at most two programs per call: the Gauss-Newton one
    (which a Gauss-Newton run at the same structure has already compiled) and the exact one.

    *What it reports.* Each history row's ``hessian`` is the model that solved its step (``'gauss_newton'`` before the
    switch, ``'exact'`` from it on), and every row of a switch run carries five more fields
    (:py:class:`t3toolbox.backend.optimizers.NewtonInfo`): ``switch_e_gn`` (each Gauss-Newton-phase step),
    ``switch_pKp`` and ``switch_e_n`` (each tracked step: the curvature product ``⟨p, K p⟩`` and ``e_N``),
    ``switch_count`` (the count after the row, in the Gauss-Newton phase) and ``switch_fired`` (``True`` on the row
    whose count reached ``switch_k``). ``stats`` gains ``'hessian'`` (``'switch'``), ``'switch_track'`` (the resolved
    value; ``None`` with ``switch_k=0``) and ``'switched_at'``, the index of the first row whose model is exact
    (``len(history)`` when the switch fired on the last row; ``0`` with ``switch_k=0``; ``None`` if it never fired).
    With ``verbose=True`` each Gauss-Newton-phase step row's header carries ``sw`` and the ratio ``e_N / e_GN``
    (``—`` where no product ran), then ``×`` the count, the switching row adds ``⇒ exact`` (all before a last row's
    stop reason), and the exact rows after it show their ``κ σ ε`` cells::

        iter  1 | obj 2.537e+05 ... | Δf -2.03e+02 ρ 0.85 | 0.03s | sw 3.0e-03 ×2 ⇒ exact
        iter  2 | obj 2.535e+05 ... | Δf -9.75e+01 ρ 1.00 | 0.05s | κ -3.5e-01 σ 5.8e+01 ε 1.3e-04

    The rule, the options, the cost, when to use it and a worked run: :doc:`/fitting_and_optimization` §4.14.

    **Timing a run** (``timing=True``). Where does a Newton run's time go -- CG or the model build, the retraction or
    the line search's extra trials, the alignment of a padded start, the compiler? ``timing=True`` splits each history
    row's ``wall_time`` window into phases and adds ``stats['timing']``. It changes no number: ``x`` and every other
    ``stats`` field are bitwise the untimed run's, the ``wall_time`` clock readings aside. A row's phases, in the
    order the ``time`` line shows them (``None`` when one did not run on that row;
    :py:class:`t3toolbox.backend.optimizers.PhaseSeconds`):

    * ``frame``, ``sweep``, ``companion``, ``residual``, ``curvature`` -- the local model's build: the orthonormal frame
      at the point, the sampling kind's sweep at it, the geometry's per-frame data, the residual, and with
      ``hessian='exact'`` or ``'switch'`` the curvature data. ``None`` on a row that reuses the model (after a rejected
      trust-region trial).
    * ``gradient`` -- the misfit transpose ``Π 𝒥ᵀ r`` (in every mode) and what reads it: the objective's parts, the
      gradient and its norm.
    * ``reference`` -- row 0 only: the reference ``‖g0‖`` of ``g0_reference``, the ``'cone'`` alignment included.
    * ``cg`` -- the CG solve (under ``use_jit``, one compiled call). ``forms`` -- the step's scalars (the slope,
      ``pᵀHp``, ``‖p‖``, the first trust radius).
    * ``trial_retraction``, ``trial_objective`` -- every trial's retraction and objective (``trials`` counts them).
    * ``tracking`` -- ``hessian='switch'``: a tracked Gauss-Newton-phase row's curvature product (``switch_pKp``);
      ``None`` on every other row.
    * ``previous_callback`` -- the previous row's ``callback`` (a display) and bookkeeping, so a heavy display appears
      a row late; ``remainder`` -- what no phase covers.

    ``stats['timing']`` carries the run's ``total``, its one-offs (``setup``; ``alignment`` and ``alignment_passes``
    with ``padded_from``; ``reference_passes`` with ``'cone'``; ``last_callback``), each phase summed over the rows
    (``phases``), the compile wait likewise (``compile``), CG's seconds per iteration (``cg_per_iteration``) and every
    row's record (``rows``). ``total`` is the setup, the alignment, the rows' windows and the last callback, end to
    end. With ``verbose=True`` each row also prints a ``time`` line, and the run ends with a closing table; a custom
    ``callback`` reads the row's record as ``NewtonInfo.timing``.

    *What it costs.* Every phase ends by waiting for its results, because jax dispatches asynchronously: without the
    wait, one phase's work would be charged to whichever later phase first needed it. On numpy the waits are free.
    On jax they serialize the run (the host no longer queues the next phase while the device works), so a timed jax
    run is slower, and its times describe the serialized run. On the maintainer's laptop CPU, timing on against off
    differed by -4 % to +2 % on numpy, eager jax and ``use_jit``, ragged and uniform, at d = 3 and d = 6: within the
    run-to-run noise. On an A100 GPU the same cases and a downstream level differed by -1.4 % to +3.1 %, also within
    the noise.

    *The compile column.* On a jax run each phase also carries jax's compile wait (``compile_seconds``: tracing,
    lowering and backend compilation, from ``jax.monitoring``). With ``use_jit=True`` CG is one program, compiled on
    its first call (row 0's ``cg``), and a second run at the same structure compiles nothing; the eager phases around
    CG compile their operations the first time they run, and the uniform alignment's whole-loop program is compiled
    inside ``alignment``. Eager jax compiles every operation of a new structure on first use. On the laptop, the
    example below run on jax (x64) took 36 s cold under ``use_jit``, 31 s of it compiling (16 s in the alignment,
    2.9 s CG's program), and 3.5 s at the same structure again; eager, 71 s cold (54 s compiling) and 14 s again. The
    column is ``None`` on numpy and on a jax whose ``jax.monitoring`` cannot unregister a listener (jax 0.4.30). The
    listener is process-global: compiles in other threads, or in a timed run inside a callback, are counted too.

    *What is not timed.* This function's own setup -- checking and packing the batch, reducing ``x0`` and ``padded_from`` to
    minimal ranks, building the display -- runs before the backend's clock starts, so ``stats['timing']`` describes
    the backend call, the same for a raw-``.data`` user. The phases, the costs and the display in full:
    :doc:`/fitting_and_optimization` §4.13.

    Examples
    --------
    Recover a low-rank tensor from noiseless ``apply`` measurements on the **uniform** layer -- a uniform
    ``x0`` (with ``UNIFORM_MANIFOLD``) returns a ``UniformTuckerTensorTrain`` (a zero start is fine on the
    manifold; unit-norm probe rows keep the least-squares well-conditioned):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> import t3toolbox.optimizers as optimizers
    >>> from t3toolbox.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> np.random.seed(0)
    >>> A = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(120, N) for N in (6, 7, 8)]
    >>> ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]   # unit-norm rows
    >>> b = A.apply(ww)
    >>> x0 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.zeros((6, 7, 8), (2, 2, 2), (1, 2, 2, 1)))
    >>> x_opt, stats = optimizers.newton_cg(ut3m.UNIFORM_MANIFOLD, LeastSquaresOracle(),
    ...                                     LeastSquaresBatch('apply', (), ww, b), x0, max_newton=30)
    >>> type(x_opt).__name__                              # returned in the same (uniform) representation
    'UniformTuckerTensorTrain'
    >>> bool(np.linalg.norm(x_opt.to_dense() - A.to_dense()) / np.linalg.norm(A.to_dense()) < 1e-6)
    True

    For a **ragged** fit pass ``t3m.MANIFOLD`` and a ``TuckerTensorTrain`` ``x0`` instead; the call is
    otherwise identical.

    **The exact Hessian.** On a noisy ``probe_derivatives`` fit warm-started near the solution, the exact
    Hessian (with its default trust region, and ``cg_forcing_power=1.0`` for the q-quadratic tail) converges
    in a fraction of the steps Gauss-Newton takes (Gauss-Newton is linear there, about halving ``‖g‖`` per
    step; where its tail ends -- the gradient test or a failed line search -- depends on the BLAS's rounding):

    >>> import t3toolbox.manifold as t3m
    >>> np.random.seed(0)
    >>> shape, rng = (5, 5, 5), np.random.default_rng(0)
    >>> ww, pp = ([rng.standard_normal((40, n)) for n in shape] for _ in range(2))
    >>> target = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
    >>> from t3toolbox.fitting import affine_jet
    >>> jet = affine_jet(ww, pp)                                   # the lines ww + s pp
    >>> data = [z + 0.01 * rng.standard_normal(z.shape) for z in target.probe(jet, (1,))]   # noisy, along lines
    >>> oracle, batch = LeastSquaresOracle(), LeastSquaresBatch('probe', (1,), jet, data)
    >>> x0, _ = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch,
    ...                              t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1)), max_newton=5)
    >>> opts = dict(max_newton=40, g0_reference='warm_start')
    >>> _, gn = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x0, **opts)
    >>> _, ex = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x0, hessian='exact',
    ...                              cg_forcing_power=1.0, **opts)
    >>> ex['history'][0]['step_control'], ex['stop_reason'] in ('gradient', 'objective_floor')
    ('trust_region', True)
    >>> gn['newton_steps'] > 2 * ex['newton_steps']
    True

    **Switching from Gauss-Newton to the exact Hessian.** The same problem with ``hessian='switch'``: Gauss-Newton
    steps until the exact model has twice running predicted the objective four times better, then the exact Hessian,
    in a fraction of Gauss-Newton's steps:

    >>> _, sw = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x0, hessian='switch', **opts)
    >>> sw['hessian'], sw['switch_track'], sw['stop_reason'] in ('gradient', 'objective_floor')
    ('switch', 'countable', True)
    >>> j = sw['switched_at']                                   # the first row whose model is exact
    >>> fired = sw['history'][j - 1]                            # the row that switched
    >>> [row['hessian'] for row in sw['history'][j - 1:j + 1]], fired['switch_fired'], fired['switch_count']
    (['gauss_newton', 'exact'], True, 2)
    >>> fired['switch_e_n'] < 0.25 * fired['switch_e_gn']
    True
    >>> sw['newton_steps'] < gn['newton_steps'] / 2             # gn: Gauss-Newton with its line search
    True

    **Timing a run.** The same problem one level up: the rank-2 fit ``x0`` zero-padded to rank 3, restarted from the
    aligned frame (``padded_from``) with the ``'cone'`` reference. ``timing=True`` adds ``stats['timing']``; the
    phases tile each row's window, and the run is the untimed run:

    >>> x_new = x0.resize(shape, (3, 3, 3), (1, 3, 3, 1))                      # the next level, zero-padded
    >>> level = dict(max_newton=40, hessian='exact', cg_forcing_power=1.0, padded_from=x0,
    ...              g0_reference='cone')
    >>> _, timed = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x_new, timing=True,
    ...                                 **level)
    >>> T = timed['timing']
    >>> rows = T['rows']                                                        # one record per history row
    >>> len(rows) == len(timed['history'])
    True
    >>> all(abs(sum(s for s in row['seconds'].values() if s is not None) - row['window']) < 1e-9
    ...     for row in rows)                                                     # the phases tile each window
    True
    >>> rows[0]['seconds']['reference'] is not None, rows[1]['seconds']['reference']     # row 0 only
    (True, None)
    >>> T['compile'] is None, len(T['alignment_passes']) == len(timed['start_movements'])   # numpy: no compiler
    (True, True)
    >>> _, plain = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x_new, **level)
    >>> 'timing' in plain, plain['losses'] == timed['losses']                   # the same run, untimed
    (False, True)

    **Precision — jit runs in jax's default float32.** ``use_jit=True`` opts into jax world, whose default
    dtype is float32. Fit a small exact-rank tensor three ways; the giveaway throughout is the returned
    core's dtype. First set up the problem:

    >>> import t3toolbox.manifold as t3m
    >>> np.random.seed(0)
    >>> A = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(60, N) for N in (4, 5, 6)]
    >>> ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
    >>> b = A.probe(ww)
    >>> oracle, batch = LeastSquaresOracle(), LeastSquaresBatch('probe', (), ww, b)
    >>> x0 = t3.TuckerTensorTrain.zeros((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
    >>> def rel_err(x): return float(np.linalg.norm(np.asarray(x.to_dense()) - A.to_dense()) / np.linalg.norm(A.to_dense()))

    (1) The numpy path is float64 and recovers to near machine precision (~1e-10):

    >>> x_np, _ = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=20)
    >>> str(x_np.data[0][0].dtype), bool(rel_err(x_np) < 1e-8)
    ('float64', True)

    (2) ``use_jit=True`` runs the SAME code in jax's default float32 and stalls ~1000x coarser (~1e-7):

    >>> x_jit, _ = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=20, use_jit=True)
    >>> str(x_jit.data[0][0].dtype), bool(rel_err(x_jit) > 1e-8)
    ('float32', True)

    (3) Enabling jax x64 restores float64 under jit -- full accuracy again (~1e-10). ``jax_enable_x64`` is
    a **global, process-wide** flag, so restore it right after the fit; the check reads values captured
    while it was on:

    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>> x_x64, _ = optimizers.newton_cg(t3m.MANIFOLD, oracle, batch, x0, max_newton=20, use_jit=True)
    >>> dtype_x64, ok_x64 = str(x_x64.data[0][0].dtype), bool(rel_err(x_x64) < 1e-8)
    >>> jax.config.update("jax_enable_x64", False)          # restore the default before asserting (no leak)
    >>> dtype_x64, ok_x64
    ('float64', True)
    """
    mode = bopt._check_hessian(hessian, 'newton_cg', bopt._NEWTON_HESSIAN_MODES)   # structural, before the setup work
    bopt._check_switch_options(mode, switch_tau, switch_k, switch_cg_forcing_power, switch_track)   # likewise
    if not isinstance(timing, (bool, np.bool_)):
        raise TypeError('newton_cg: timing must be a bool, got %r' % (timing,))
    if val_batch is not None:                        # checked (structural, before any work) and packed for a uniform x0
        _same_kind_and_order(batch, val_batch, 'newton_cg: the validation batch')
        _fitting._require_layer(geometry, x0)
        val_batch = _fitting.prepare_batch(val_batch, x0)
    s = _setup(geometry, batch, x0)
    x0_cores, run_batch = s.x0_cores, s.batch
    to_jax = bool(use_jit) and common.jax_available
    if to_jax:                                       # convert here, so the display holds the very batch the run uses
        x0_cores, run_batch = bopt._prepare_jit_inputs(True, x0_cores, run_batch)
    records = None
    if callback is None and verbose:
        callback, records = bdisp.make_newton_display(oracle, run_batch, val_batch, error_table=error_table)
    seam = _padded_start_keywords(geometry, s, x0, start_frame, padded_from, to_jax, g0_reference, dict(
        padded_oversampling=padded_oversampling, padded_oversampled_passes=padded_oversampled_passes,
        padded_max_passes=padded_max_passes, padded_tol=padded_tol, padded_alternate=padded_alternate,
        padded_seed=padded_seed, cone_oversampling=cone_oversampling, cone_oversampled_passes=cone_oversampled_passes,
        cone_max_passes=cone_max_passes, cone_tol=cone_tol, cone_alternate=cone_alternate, cone_seed=cone_seed))
    timed = {'timing': True} if timing else {}           # forwarded only when set, like the seam's options
    switching = {name: value for name, value in (('switch_tau', switch_tau), ('switch_k', switch_k),
                                                 ('switch_cg_forcing_power', switch_cg_forcing_power),
                                                 ('switch_track', switch_track)) if value is not None}   # likewise
    x_cores, stats = bopt.newton_cg(s.geom, oracle, run_batch, x0_cores, regularizer,
                                    max_newton=max_newton, gtol_rel=gtol_rel, cg_maxiter=cg_maxiter,
                                    c_armijo=c_armijo, on_line_search_failure=on_line_search_failure,
                                    g0_reference=g0_reference, g0norm_newton=g0norm_newton, g0norm_cg=g0norm_cg,
                                    cg_forcing_power=cg_forcing_power, use_jit=use_jit, callback=callback,
                                    hessian=hessian, curvature_damping=curvature_damping,
                                    step_control=step_control, tr_radius0=tr_radius0, **seam, **timed, **switching)
    if records is not None:
        stats['diagnostics'] = records
    return s.rewrap(x_cores), stats
