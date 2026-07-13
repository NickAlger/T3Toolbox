'''Sampling-kind primitives for the geometry-generic Gauss-Newton fitting model.

The Gauss-Newton model ``m(p) = c + gᵀp + ½ pᵀ(JᵀJ)p`` of a least-squares objective factors over two
*independent* choices, and the fitting layer is generic in both:

  * the sampling **kind** (apply / entries / probe) -- the bare single-sample Jacobian ``𝒥`` / its
    transpose ``𝒥ᵀ`` (from :py:mod:`t3toolbox.backend.probing`), plus the kind-specific ``‖·‖²``
    reduction over the sample stack; and
  * the **geometry** (manifold / corewise) -- the gauge projection ``Π`` (``geometry.project``), with
    the Riemannian forward ``J = 𝒥∘Π`` and gradient ``Jᵀr = Π∘𝒥ᵀr``.

This module supplies the **kind** half: a :py:class:`SamplingKind` descriptor bundling the bare probing
primitives + the reduction per kind (:py:data:`APPLY` / :py:data:`ENTRIES` / :py:data:`PROBE`). The
**geometry** half lives in the geometry (:py:data:`t3toolbox.manifold.MANIFOLD` /
:py:data:`~t3toolbox.manifold.COREWISE`); the frontend :py:class:`t3toolbox.fitting.GaussNewtonModel`
composes the two. The §6.3 corewise substitution ``(U,O,P,Q) -> (U,G,G,G)`` is now nothing but
``CorewiseGeometry.frame`` -- there is no separate corewise backend.

``probe`` is vector-valued (one free mode per probe), so its residual / forward output is a sequence of
``d`` arrays and its reduction sums the free mode too; apply / entries are the scalar all-modes special
case. The objective constant ``c = ½‖r‖²`` is just ``½ · sumsq(r)`` (the same reduction as the model's
quadratic term ``½‖𝒥Πp‖²``), so the kind needs no separate objective function.
'''

import math
import typing as typ
from dataclasses import dataclass

import numpy as np

import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as pd
from t3toolbox.backend import apply as bapply
from t3toolbox.backend import entries as bentries
from t3toolbox.backend.common import *

__all__ = [
    'SamplingKind',
    'APPLY',
    'ENTRIES',
    'PROBE',
    'probe_kind',
    'apply_derivatives_kind',
    'entries_derivatives_kind',
    'probe_derivatives_kind',
    'sumsq_over_samples',
    'sumsq_over_probes',
    'block_sumsq_over_samples',
    'block_sumsq_over_probes',
]


def sumsq_over_samples(
        out:    NDArray,    # scalar-output forward/residual, shape W+C (apply / entries)
        n_w:    int,        # number of leading sample-stack (W) axes
) -> NDArray:               # sum of squares over W, keeping the frame stack C
    '''The ``‖·‖²`` reduction for the scalar-output (apply / entries) kinds: sum ``out**2`` over the
    leading ``n_w`` sample axes, keeping the frame stack ``C``. Used for both the objective ``c = ½‖r‖²``
    and the model's quadratic term ``½‖𝒥 Π p‖²``.'''
    use_jax = is_jax_ndarray(out)
    xnp, _, _ = get_backend(False, use_jax)
    return xnp.sum(out ** 2, axis=tuple(range(n_w)))


def sumsq_over_probes(
        zz:     typ.Union[typ.Sequence[NDArray], NDArray],  # ragged len=d (elm W+C+(Ni,)) OR packed (d,)+W+C+(N,)
        n_w:    int,                                         # number of leading sample-stack (W) axes
) -> NDArray:                           # sum of squares over W and the free mode, summed over d, keep C
    '''The ``‖·‖²`` reduction for the vector-output ``probe`` kind: sum over the leading ``n_w`` sample
    axes and the trailing free mode, keeping the frame stack ``C``, summed over the ``d`` probes. **Mirrors**
    ``zz``'s packedness: a ragged ``len=d`` sequence loops over ``d`` (each ``z_i`` is ``W+C+(Ni,)``); a
    packed ``(d,)+W+C+(N,)`` array sums ``d`` + ``W`` + the padded mode ``N`` in one op (the free-mode
    padding is a zeroed prefix, so it contributes nothing -- the packed inner-loop path).'''
    if not isinstance(zz, (list, tuple)):               # packed (d,)+W+C+(N,): d (axis 0) + W + N, keep C
        use_jax = is_jax_ndarray(zz)
        xnp, _, _ = get_backend(False, use_jax)
        axes = (0,) + tuple(range(1, 1 + n_w)) + (zz.ndim - 1,)
        return xnp.sum(zz ** 2, axis=axes)
    use_jax = is_jax_ndarray(zz[0])
    xnp, _, _ = get_backend(False, use_jax)
    total = None
    for z in zz:
        axes = tuple(range(n_w)) + (z.ndim - 1,)        # W (leading) + the free mode Ni (last); keep C
        s = xnp.sum(z ** 2, axis=axes)
        total = s if total is None else total + s
    return total


# --------------------------------------------------------------------------------------------------
# Per-(mode, order) block reductions -- the `sumsq_*` siblings that KEEP the mode/order axes instead of
# collapsing them, for the Newton-CG diagnostic error table (docs/fitting..§?, dev/newton_display_plan.md).
# They return a 2-D (n_mode, n_order) matrix mirroring the kind's ω[mode, order] shape, and are
# **UNWEIGHTED** (raw ‖r_ij‖², so the table is the honest data-norm relative error and ½·Σ block == the
# unweighted objective). apply/entries have no mode axis (n_mode = 1); plain kinds have no order (n_order = 1).
# --------------------------------------------------------------------------------------------------
def block_sumsq_over_samples(
        out:        NDArray,    # scalar-output residual: (order+1)+W+C (has_order) or W+C
        n_w:        int,        # number of leading W axes (unused -- W + C are summed wholesale)
        has_order:  bool,       # True for the derivative kinds (a leading order axis at axis 0)
) -> NDArray:                   # (1, n_order) -- per-order sum of squares (no mode axis: apply/entries)
    '''Per-``(mode, order)`` sum-of-squares for the scalar-output apply/entries kinds -> a 2-D
    ``(1, n_order)`` matrix (no mode axis -- they contract every mode). Keeps only a leading order axis,
    sums the rest (``W`` + ``C``); **UNWEIGHTED** (raw ``‖r_·j‖²``).'''
    use_jax = is_jax_ndarray(out)
    xnp, _, _ = get_backend(False, use_jax)
    if has_order:
        return xnp.sum(out ** 2, axis=tuple(range(1, out.ndim))).reshape(1, -1)   # keep order (axis 0)
    return xnp.sum(out ** 2).reshape(1, 1)


def block_sumsq_over_probes(
        zz:         typ.Sequence[NDArray],  # vector-output residual, len=d, elm (order+1)+W+C+(Ni,) or W+C+(Ni,)
        n_w:        int,                    # number of leading W axes (unused; see block_sumsq_over_samples)
        has_order:  bool,
) -> NDArray:                               # (d, n_order) -- per-(mode, order) sum of squares
    '''Per-``(mode, order)`` sum-of-squares for the vector-output probe kinds -> a 2-D ``(d, n_order)``
    matrix (``d`` probe modes). Keeps only a leading order axis per list element, sums the rest (``W`` +
    ``C`` + the free mode); **UNWEIGHTED**. (The packed uniform probe -- a ``(d,)+…`` array, not a list --
    gets its own override; D6 of dev/newton_display_plan.md.)'''
    use_jax = is_jax_ndarray(zz[0])
    xnp, _, _ = get_backend(False, use_jax)
    rows = [xnp.sum(z ** 2, axis=tuple(range(1, z.ndim))) if has_order else xnp.sum(z ** 2).reshape(1)
            for z in zz]
    return xnp.stack(rows, axis=0)


# --------------------------------------------------------------------------------------------------
# Default-draw layout helpers (the minimal layout the kind exposes so the optimizer can build its
# DEFAULT flat minibatch draw -- a random subset across the whole sample stack W. A USER-supplied draw
# bypasses these entirely.) `n_measurements` = the flat W size; `take` gathers a flat W subset out of
# (sample, data) -- both kind-specific, since W sits in a different place per kind (and the derivative
# data has a leading order axis).
# --------------------------------------------------------------------------------------------------
def _flat_gather(
        arr:     NDArray,    # an array carrying the sample stack W
        w_start: int,        # axis where W begins
        n_w:     int,        # number of W axes (flattened to one, then gathered)
        idx:     NDArray,    # int -- the drawn flat positions over the combined W axis
) -> NDArray:                # arr with its W axes replaced by one axis of len(idx)
    '''Flatten ``arr``'s ``n_w`` W-axes (from ``w_start``) into one and gather ``idx`` along it.'''
    use_jax = is_jax_ndarray(arr)
    xnp, _, _ = get_backend(False, use_jax)
    sh = arr.shape
    return xnp.take(arr.reshape(sh[:w_start] + (-1,) + sh[w_start + n_w:]), idx, axis=w_start)


def _prod_w(arr, w_start, n_w):
    return math.prod(arr.shape[w_start:w_start + n_w])

# `take` per kind: gather a flat W subset. Regular: sample is ww / index, data is W-leading (apply/
# entries scalar, probe list). Derivative: sample is (ww/index, pp), data has a leading order axis (W
# at axis 1). All flatten W to a single axis of size len(idx).
def _take_apply(ww, data, idx):
    n_w = ww[0].ndim - 1
    return [_flat_gather(w, 0, n_w, idx) for w in ww], _flat_gather(data, 0, n_w, idx)

def _take_probe(ww, data, idx):
    n_w = ww[0].ndim - 1
    return [_flat_gather(w, 0, n_w, idx) for w in ww], [_flat_gather(d, 0, n_w, idx) for d in data]

def _take_entries(index, data, idx):
    n_w = index.ndim - 1
    return _flat_gather(index, 1, n_w, idx), _flat_gather(data, 0, n_w, idx)

def _take_deriv_apply(sample, data, idx):
    ww, pp = sample; n_w = ww[0].ndim - 1
    return ([_flat_gather(w, 0, n_w, idx) for w in ww], [_flat_gather(p, 0, n_w, idx) for p in pp]), \
        _flat_gather(data, 1, n_w, idx)

def _take_deriv_entries(sample, data, idx):
    index, pp = sample; n_w = index.ndim - 1
    return (_flat_gather(index, 1, n_w, idx), [_flat_gather(p, 0, n_w, idx) for p in pp]), \
        _flat_gather(data, 1, n_w, idx)

def _take_deriv_probe(sample, data, idx):
    ww, pp = sample; n_w = ww[0].ndim - 1
    return ([_flat_gather(w, 0, n_w, idx) for w in ww], [_flat_gather(p, 0, n_w, idx) for p in pp]), \
        [_flat_gather(d, 1, n_w, idx) for d in data]


def _weight_matrix(
        weight:  typ.Optional[typ.Any],   # None, or array / (nested) sequence -- the raw ω input
        order:   int,                      # highest derivative order (0 for the plain kinds)
        bare:    str,                      # a 1-D input binds to -- 'order' (row) or 'mode' (column)
) -> typ.Optional[NDArray]:                # None, or the canonical 2-D ω[m,o], m in {1,d}, o in {1,order+1}
    '''Normalize a raw residual-weight input to the canonical 2-D matrix ``ω[mode, order]`` (host numpy).

    The residual weight enters the objective ``½‖ω ⊙ r‖²`` only (``sumsq`` scales by ``ω``, ``transpose`` --
    the gradient ``𝒥ᵀ ω²r`` -- by ``ω²``; ``forward`` / ``point_forward`` / ``data`` stay raw). ``ω`` is a
    ``(mode, order)`` matrix: numpy right-alignment makes **order the innermost/most-important axis**, so a
    bare 1-D input binds to each model's innermost axis -- ``bare='order'`` for the derivative kinds (a
    vector is per-order, broadcast over modes -- the backward-compatible rule) and ``bare='mode'`` for plain
    probe (a vector is per-mode; plain probe has no order axis). A 2-D array passes through idempotently (so
    the uniform layer can re-feed a normalized matrix). Validates the order dim ``o in {1, order+1}`` here
    (``order`` is known); the mode dim ``m in {1, d}`` is validated in the frontend (which knows ``d``). The
    frontend also enforces the plain-probe **1-D** contract (rejecting a 2-D ``(d, 1)``); this helper stays
    lenient so the internal re-feed works. Weights are host numpy (static structure, like the uniform masks
    and ``ω`` before it) -- they fold into the compiled program as device constants on the jax path.'''
    if weight is None:
        return None
    w = np.asarray(weight, dtype=float)
    if w.ndim == 1:
        w = w[None, :] if bare == 'order' else w[:, None]
    elif w.ndim != 2:
        raise ValueError("residual weight must be 1-D or 2-D (ω[mode, order]); got %d-D of shape %s"
                         % (w.ndim, w.shape))
    o = w.shape[1]
    if o not in (1, order + 1):
        raise ValueError("residual weight's order dimension must be 1 or order+1=%d; got %d (shape %s)"
                         % (order + 1, o, w.shape))
    return w


def _make_weight(
        w2d:         typ.Optional[NDArray],  # canonical 2-D ω[m,o] from _weight_matrix, or None (ω = 1)
        order_axis:  typ.Optional[int] = 0,  # order axis in an ARRAY x (None => no order axis; o must be 1)
        mode_axis:   typ.Optional[int] = None,  # mode axis in an ARRAY x (None => no mode axis; m must be 1)
) -> typ.Callable:                           # apply_w(x, power) = x · ω**power, broadcast over its axes
    '''Return ``apply_w(x, power) = x · ω**power`` for the canonical weight matrix ``ω = w2d`` (or identity
    if ``None``), covering the three residual/output layouts:

      * **ragged probe** (``x`` a list of ``d`` order-leading arrays): element ``i`` is scaled by the
        order-vector ``ω[i if m>1 else 0]`` along its order axis (axis 0 per element). Plain probe has
        ``o = 1`` (a per-mode scalar); the derivative probe has ``o = order+1``. ``order_axis`` / ``mode_axis``
        are ignored here (order is always axis 0 per element, mode is the list index).
      * **apply / entries array** (no mode axis): ``mode_axis=None`` -> ``m`` must be 1 (else the structural
        "no mode axis" error -- mode weighting is probe-only); the order-vector is placed at ``order_axis``.
      * **packed uniform probe array** (``(d,)+…`` for plain, ``(d,)+(order+1,)+…`` for derivatives):
        ``ω``'s mode axis at ``mode_axis`` (axis 0), order axis at ``order_axis`` (axis 1, or ``None`` for
        the order-free plain probe), 1s elsewhere.'''
    if w2d is None:
        return lambda x, power: x
    m, o = w2d.shape
    def apply_w(x, power):
        wp = w2d ** power
        if isinstance(x, (list, tuple)):                 # ragged probe: a list of d order-leading arrays
            return [xi * wp[i if m > 1 else 0].reshape((o,) + (1,) * (xi.ndim - 1))
                    for i, xi in enumerate(x)]
        if mode_axis is None and m > 1:                  # apply/entries have no mode axis (probe-only)
            raise ValueError("this sampling kind has no mode axis (apply / entries); a per-mode residual "
                             "weight (mode dim %d > 1) is only defined for probe" % m)
        shp = [1] * x.ndim                               # place ω's non-unit axes; 1s broadcast elsewhere
        if mode_axis is not None:
            shp[mode_axis] = m
        if order_axis is not None:
            shp[order_axis] = o
        return x * wp.reshape(tuple(shp))
    return apply_w


@dataclass(frozen=True)
class SamplingKind:
    '''A sampling kind's bare primitives, bundled so the GN model is generic over the kind.

    Holds the kind-specific functions the geometry-generic Gauss-Newton model needs -- the bare ``𝒥`` /
    ``𝒥ᵀ`` (no gauge ``Π``: that is the geometry's), the ``‖·‖²`` reduction, and the sample-stack axis
    count. The model binds one of :py:data:`APPLY` / :py:data:`ENTRIES` / :py:data:`PROBE`; the geometry
    supplies ``Π`` around them. ``sample`` is the kind's measurement spec: the probe/apply vectors ``ww``
    (apply / probe) or the integer grid ``index`` (entries).
    '''
    name:           str           # 'apply' / 'entries' / 'probe' (+ '_derivatives')
    precompute:     typ.Callable   # (frame_data, sample)                        -> frame_sweep
    forward:        typ.Callable   # (variations_data, sample, frame_data, sweep)-> 𝒥 v  (the bare forward)
    transpose:      typ.Callable   # (residual, sample, frame_data, sweep)       -> 𝒥ᵀ r (summed over W; raw dU,dG)
    sumsq:          typ.Callable   # (forward_out, n_w)                         -> ‖forward_out‖² (over W [+order,Ni])
    w_axes:         typ.Callable   # (sample)                                   -> n_w (leading sample-stack axes)
    point_forward:  typ.Callable   # (x_cores, sample)                          -> S(x) (the POINT op, for the residual)
    n_measurements: typ.Callable   # (sample)                                   -> int (flat |W|, for the default draw)
    take:           typ.Callable   # (sample, data, idx)                        -> (sample_B, data_B) (flat W subset)
    block_sumsq:    typ.Optional[typ.Callable] = None   # (out, n_w) -> (n_mode, n_order) per-block ‖·‖² (UNWEIGHTED; for the diagnostic table)


APPLY = SamplingKind(
    name='apply',
    precompute=lambda frame, ww: bapply.tv_precompute_apply_frame_sweep(frame, ww),
    forward=lambda v, ww, frame, bs: bapply.tv_apply_jacobian_from_sweep(v, ww, frame, bs),
    transpose=lambda r, ww, frame, bs: bapply.tv_apply_transpose_from_sweep(r, ww, frame, bs, sum_over_probes=True),
    sumsq=sumsq_over_samples,
    w_axes=lambda ww: ww[0].ndim - 1,
    point_forward=lambda x_cores, ww: bapply.t3_apply(x_cores, ww),
    n_measurements=lambda ww: _prod_w(ww[0], 0, ww[0].ndim - 1),
    take=_take_apply,
    block_sumsq=lambda out, n_w: block_sumsq_over_samples(out, n_w, has_order=False),
)

ENTRIES = SamplingKind(
    name='entries',
    precompute=lambda frame, index: bentries.tv_precompute_entries_frame_sweep(frame, index),
    forward=lambda v, index, frame, bs: bentries.tv_entries_jacobian_from_sweep(v, index, frame, bs),
    transpose=lambda r, index, frame, bs: bentries.tv_entries_transpose_from_sweep(r, index, frame, bs, sum_over_probes=True),
    sumsq=sumsq_over_samples,
    w_axes=lambda index: index.ndim - 1,
    point_forward=lambda x_cores, index: bentries.t3_entries(x_cores, index),
    n_measurements=lambda index: _prod_w(index, 1, index.ndim - 1),
    take=_take_entries,
    block_sumsq=lambda out, n_w: block_sumsq_over_samples(out, n_w, has_order=False),
)

def probe_kind(
        weight: typ.Optional[typ.Any] = None,  # per-mode residual weight ω, (d,) / (d,1); None = 1 (unweighted)
) -> SamplingKind:                             # the vector-valued `probe` kind (optionally per-mode weighted)
    '''The **probe** sampling kind (vector-valued: one free mode per probe), optionally **per-mode**
    weighted. Mode weighting is the order-0 special case of the same residual-weight machinery as the
    derivative kinds: the objective is ``½ Σ_i ‖ω_i z_i‖²`` over the ``d`` per-mode probe residuals ``z_i``,
    so ``ω`` (a per-mode scalar) enters ``sumsq`` (×ω) and ``transpose`` (×ω²) only. ``weight=None`` is the
    plain unweighted probe (``PROBE``). Plain probe has no order axis, so the weight is a 1-D ``(d,)``
    per-mode vector -- the frontend enforces that (rejecting a 2-D ``(d, 1)``; see
    :py:func:`t3toolbox.fitting.probe_model`).'''
    aw = _make_weight(_weight_matrix(weight, 0, 'mode'))   # ragged probe list; per-mode scalar (o = 1)
    return SamplingKind(
        name='probe',
        precompute=lambda frame, ww: probing.tv_precompute_probe_frame_sweep(frame, ww),
        forward=lambda v, ww, frame, bs: probing.tv_probe_jacobian_from_sweep(v, ww, frame, bs),
        transpose=lambda r, ww, frame, bs: probing.tv_probe_transpose_from_sweep(aw(r, 2), ww, frame, bs, sum_over_probes=True),
        sumsq=lambda out, n_w: sumsq_over_probes(aw(out, 1), n_w),
        w_axes=lambda ww: ww[0].ndim - 1,
        point_forward=lambda x_cores, ww: probing.t3_probe(ww, x_cores),
        n_measurements=lambda ww: _prod_w(ww[0], 0, ww[0].ndim - 1),
        take=_take_probe,
        block_sumsq=lambda out, n_w: block_sumsq_over_probes(out, n_w, has_order=False),
    )


PROBE = probe_kind()   # the plain unweighted probe kind (a module singleton, as APPLY / ENTRIES)


# --------------------------------------------------------------------------------------------------
# Derivative sampling kinds (the symmetric directional-derivative jets of apply/entries/probe). The
# operator is parameterized by `order` (highest derivative order) + an optional residual weight `weight`
# (ω[mode, order], a matrix -- `_weight_matrix`/`_make_weight`). `sample` is the paired `(ww, pp)` /
# `(index, pp)`; the data + outputs gain a leading order axis, so `sumsq`/`w_axes` count it via `n_w + 1`.
# ω enters only `sumsq` (×ω) and `transpose` (×ω²); `forward`/`point_forward` are raw (the user passes RAW
# data + ω). apply/entries contract every mode into a scalar -- no mode axis -- so they take an ORDER-ONLY
# weight (a per-mode weight is a structural error, caught in `_make_weight`); only probe is mode-weightable
# (`(d, order+1)`). See dev/archive/derivative_fitting_plan.md §5 and dev/per_mode_weighting_plan.md.
# --------------------------------------------------------------------------------------------------
def apply_derivatives_kind(
        order:  int,                                # highest derivative order
        weight: typ.Optional[typ.Any] = None,       # ORDER-only residual weight ω, (order+1,); None = 1
) -> SamplingKind:                                  # sample = (ww, pp); data = (order+1)+W
    '''The **apply-derivatives** sampling kind (operator only): symmetric directional derivatives of the
    all-modes apply, orders ``0..order``, in direction ``P``. ``sample = (ww, pp)``. All-modes apply has no
    mode axis, so ``weight`` is **order-only** (a per-mode weight raises -- mode weighting is probe-only).'''
    aw = _make_weight(_weight_matrix(weight, order, 'order'))
    return SamplingKind(
        name='apply_derivatives',
        precompute=lambda frame, s: pd.tv_precompute_apply_frame_sweep_jets(frame, s[0], s[1], order),
        forward=lambda v, s, frame, bs: pd.tv_apply_jacobian_derivatives_from_sweep(v, s[0], s[1], frame, bs, order),
        transpose=lambda r, s, frame, bs: pd.tv_apply_transpose_derivatives_from_sweep(
            aw(r, 2), s[0], s[1], frame, bs, order, sum_over_probes=True),
        sumsq=lambda out, n_w: sumsq_over_samples(aw(out, 1), n_w + 1),
        w_axes=lambda s: s[0][0].ndim - 1,
        point_forward=lambda x_cores, s: pd.t3_apply_derivatives(s[0], s[1], x_cores, order),
        n_measurements=lambda s: _prod_w(s[0][0], 0, s[0][0].ndim - 1),
        take=_take_deriv_apply,
        block_sumsq=lambda out, n_w: block_sumsq_over_samples(out, n_w, has_order=True),
    )


def entries_derivatives_kind(
        order:  int,
        weight: typ.Optional[typ.Any] = None,       # ORDER-only residual weight ω, (order+1,); None = 1
) -> SamplingKind:                                  # sample = (index, pp); data = (order+1)+W
    '''The **entries-derivatives** sampling kind: like :py:func:`apply_derivatives_kind` but at integer
    grid points. ``sample = (index, pp)``. Order-only ``weight`` (no mode axis -- mode weighting is
    probe-only).'''
    aw = _make_weight(_weight_matrix(weight, order, 'order'))
    return SamplingKind(
        name='entries_derivatives',
        precompute=lambda frame, s: pd.tv_precompute_entries_frame_sweep_jets(frame, s[0], s[1], order),
        forward=lambda v, s, frame, bs: pd.tv_entries_jacobian_derivatives_from_sweep(v, s[0], s[1], frame, bs, order),
        transpose=lambda r, s, frame, bs: pd.tv_entries_transpose_derivatives_from_sweep(
            aw(r, 2), s[0], s[1], frame, bs, order, sum_over_probes=True),
        sumsq=lambda out, n_w: sumsq_over_samples(aw(out, 1), n_w + 1),
        w_axes=lambda s: s[0].ndim - 1,
        point_forward=lambda x_cores, s: pd.t3_entries_derivatives(s[0], s[1], x_cores, order),
        n_measurements=lambda s: _prod_w(s[0], 1, s[0].ndim - 1),
        take=_take_deriv_entries,
        block_sumsq=lambda out, n_w: block_sumsq_over_samples(out, n_w, has_order=True),
    )


def probe_derivatives_kind(
        order:  int,
        weight: typ.Optional[typ.Any] = None,       # residual weight ω[mode,order], (d,order+1) broadcast; None = 1
) -> SamplingKind:                                  # sample = (ww, pp); data = list of d, (order+1)+W+(Ni,)
    '''The **probe-derivatives** sampling kind: vector-valued (one free mode per probe), so the residual
    / output is a list of ``d`` arrays. ``sample = (ww, pp)``. Probe has both a mode and an order axis, so
    ``weight`` is the full ``ω[mode, order]`` matrix ``(d, order+1)`` (a row ``(order+1,)`` = per-order, a
    column ``(d, 1)`` = per-mode, a matrix = both).'''
    aw = _make_weight(_weight_matrix(weight, order, 'order'))
    return SamplingKind(
        name='probe_derivatives',
        precompute=lambda frame, s: pd.tv_precompute_probe_frame_sweep_jets(frame, s[0], s[1], order),
        forward=lambda v, s, frame, bs: pd.tv_probe_jacobian_derivatives_from_sweep(v, s[0], s[1], frame, bs, order),
        transpose=lambda r, s, frame, bs: pd.tv_probe_transpose_derivatives_from_sweep(
            aw(r, 2), s[0], s[1], frame, bs, order, sum_over_probes=True),
        sumsq=lambda out, n_w: sumsq_over_probes(aw(out, 1), n_w + 1),
        w_axes=lambda s: s[0][0].ndim - 1,
        point_forward=lambda x_cores, s: pd.t3_probe_derivatives(s[0], s[1], x_cores, order),
        n_measurements=lambda s: _prod_w(s[0][0], 0, s[0][0].ndim - 1),
        take=_take_deriv_probe,
        block_sumsq=lambda out, n_w: block_sumsq_over_probes(out, n_w, has_order=True),
    )
