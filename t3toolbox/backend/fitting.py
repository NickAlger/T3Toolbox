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
``CorewiseGeometry.base`` -- there is no separate corewise backend.

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
import t3toolbox.backend.probe_derivatives as pd
from t3toolbox.backend import apply as bapply
from t3toolbox.backend import entries as bentries
from t3toolbox.backend.common import *

__all__ = [
    'SamplingKind',
    'APPLY',
    'ENTRIES',
    'PROBE',
    'apply_derivatives_kind',
    'entries_derivatives_kind',
    'probe_derivatives_kind',
    'sumsq_over_samples',
    'sumsq_over_probes',
]


def sumsq_over_samples(
        out:    NDArray,    # scalar-output forward/residual, shape W+C (apply / entries)
        n_w:    int,        # number of leading sample-stack (W) axes
) -> NDArray:               # sum of squares over W, keeping the base stack C
    '''The ``‖·‖²`` reduction for the scalar-output (apply / entries) kinds: sum ``out**2`` over the
    leading ``n_w`` sample axes, keeping the base stack ``C``. Used for both the objective ``c = ½‖r‖²``
    and the model's quadratic term ``½‖𝒥 Π p‖²``.'''
    use_jax = is_jax_ndarray(out)
    xnp, _, _ = get_backend(False, use_jax)
    return xnp.sum(out ** 2, axis=tuple(range(n_w)))


def sumsq_over_probes(
        zz:     typ.Union[typ.Sequence[NDArray], NDArray],  # ragged len=d (elm W+C+(Ni,)) OR packed (d,)+W+C+(N,)
        n_w:    int,                                         # number of leading sample-stack (W) axes
) -> NDArray:                           # sum of squares over W and the free mode, summed over d, keep C
    '''The ``‖·‖²`` reduction for the vector-output ``probe`` kind: sum over the leading ``n_w`` sample
    axes and the trailing free mode, keeping the base stack ``C``, summed over the ``d`` probes. **Mirrors**
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


def _make_order_weight(weight, order, order_axis=0):
    '''The per-order residual weight ``ω``: returns ``apply(x, power) = x · ω**power`` broadcast over the
    order axis (``x`` an array for apply/entries, a list of ``d`` arrays for probe). ``weight=None`` is
    ``ω=1`` (identity). ``ω`` enters the objective ``½‖ω⊙r‖²`` only -- so ``sumsq`` scales by ``ω`` and
    ``transpose`` (the gradient ``𝒥ᵀ ω²r``) by ``ω²``; ``forward`` / ``point_forward`` stay raw.

    ``order_axis`` is the position of the order axis in an *array* ``x`` (default 0 -- order leads, as for
    apply/entries and the per-element ragged-probe arrays). The **packed** probe-derivative output is
    ``(d,)+(order+1,)+…`` -- order at axis 1 after the mode index ``d`` -- so its kind builds this with
    ``order_axis=1``. (The ``list`` branch is always the ragged probe, order-leading per element.)'''
    if weight is None:
        return lambda x, power: x
    w = np.asarray(weight, dtype=float)                  # (order+1,)
    def apply_w(x, power):
        wp = w ** power
        if isinstance(x, (list, tuple)):                 # ragged probe: a list of d order-leading arrays
            return [xi * wp.reshape((order + 1,) + (1,) * (xi.ndim - 1)) for xi in x]
        shp = [1] * x.ndim                               # broadcast ω over the order axis of the array
        shp[order_axis] = order + 1
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
    precompute:     typ.Callable   # (base_data, sample)                        -> base_sweep
    forward:        typ.Callable   # (variations_data, sample, base_data, sweep)-> 𝒥 v  (the bare forward)
    transpose:      typ.Callable   # (residual, sample, base_data, sweep)       -> 𝒥ᵀ r (summed over W; raw dU,dG)
    sumsq:          typ.Callable   # (forward_out, n_w)                         -> ‖forward_out‖² (over W [+order,Ni])
    w_axes:         typ.Callable   # (sample)                                   -> n_w (leading sample-stack axes)
    point_forward:  typ.Callable   # (x_cores, sample)                          -> S(x) (the POINT op, for the residual)
    n_measurements: typ.Callable   # (sample)                                   -> int (flat |W|, for the default draw)
    take:           typ.Callable   # (sample, data, idx)                        -> (sample_B, data_B) (flat W subset)


APPLY = SamplingKind(
    name='apply',
    precompute=lambda base, ww: probing.precompute_apply_base_sweep(base, ww),
    forward=lambda v, ww, base, bs: probing.apply_jacobian_from_sweep(v, ww, base, bs),
    transpose=lambda r, ww, base, bs: probing.apply_transpose_from_sweep(r, ww, base, bs, sum_over_probes=True),
    sumsq=sumsq_over_samples,
    w_axes=lambda ww: ww[0].ndim - 1,
    point_forward=lambda x_cores, ww: bapply.tucker_tensor_train_apply(x_cores, ww),
    n_measurements=lambda ww: _prod_w(ww[0], 0, ww[0].ndim - 1),
    take=_take_apply,
)

ENTRIES = SamplingKind(
    name='entries',
    precompute=lambda base, index: probing.precompute_entries_base_sweep(base, index),
    forward=lambda v, index, base, bs: probing.entries_jacobian_from_sweep(v, index, base, bs),
    transpose=lambda r, index, base, bs: probing.entries_transpose_from_sweep(r, index, base, bs, sum_over_probes=True),
    sumsq=sumsq_over_samples,
    w_axes=lambda index: index.ndim - 1,
    point_forward=lambda x_cores, index: bentries.tucker_tensor_train_entries(x_cores, index),
    n_measurements=lambda index: _prod_w(index, 1, index.ndim - 1),
    take=_take_entries,
)

PROBE = SamplingKind(
    name='probe',
    precompute=lambda base, ww: probing.precompute_probe_base_sweep(base, ww),
    forward=lambda v, ww, base, bs: probing.probe_jacobian_from_sweep(v, ww, base, bs),
    transpose=lambda r, ww, base, bs: probing.probe_transpose_from_sweep(r, ww, base, bs, sum_over_probes=True),
    sumsq=sumsq_over_probes,
    w_axes=lambda ww: ww[0].ndim - 1,
    point_forward=lambda x_cores, ww: probing.probe_t3(ww, x_cores),
    n_measurements=lambda ww: _prod_w(ww[0], 0, ww[0].ndim - 1),
    take=_take_probe,
)


# --------------------------------------------------------------------------------------------------
# Derivative sampling kinds (the symmetric directional-derivative jets of apply/entries/probe). The
# operator is parameterized by `order` (highest derivative order) + an optional per-order residual
# weight `weight` (ω). `sample` is the paired `(ww, pp)` / `(index, pp)`; the data + outputs gain a
# leading order axis, so `sumsq`/`w_axes` count it via `n_w + 1`. ω enters only `sumsq` (×ω) and
# `transpose` (×ω²); `forward`/`point_forward` are raw (the user passes RAW data + ω). See
# dev/archive/derivative_fitting_plan.md §5.
# --------------------------------------------------------------------------------------------------
def apply_derivatives_kind(
        order:  int,                                # highest derivative order
        weight: typ.Optional[typ.Sequence[float]] = None,  # per-order residual weight ω, (order+1,); None = 1
) -> SamplingKind:                                  # sample = (ww, pp); data = (order+1)+W
    '''The **apply-derivatives** sampling kind (operator only): symmetric directional derivatives of the
    all-modes apply, orders ``0..order``, in direction ``P``. ``sample = (ww, pp)``.'''
    aw = _make_order_weight(weight, order)
    return SamplingKind(
        name='apply_derivatives',
        precompute=lambda base, s: pd.precompute_apply_base_sweep_jets(base, s[0], s[1], order),
        forward=lambda v, s, base, bs: pd.apply_jacobian_derivatives_from_sweep(v, s[0], s[1], base, bs, order),
        transpose=lambda r, s, base, bs: pd.apply_transpose_derivatives_from_sweep(
            aw(r, 2), s[0], s[1], base, bs, order, sum_over_probes=True),
        sumsq=lambda out, n_w: sumsq_over_samples(aw(out, 1), n_w + 1),
        w_axes=lambda s: s[0][0].ndim - 1,
        point_forward=lambda x_cores, s: pd.apply_derivatives_t3(s[0], s[1], x_cores, order),
        n_measurements=lambda s: _prod_w(s[0][0], 0, s[0][0].ndim - 1),
        take=_take_deriv_apply,
    )


def entries_derivatives_kind(
        order:  int,
        weight: typ.Optional[typ.Sequence[float]] = None,
) -> SamplingKind:                                  # sample = (index, pp); data = (order+1)+W
    '''The **entries-derivatives** sampling kind: like :py:func:`apply_derivatives_kind` but at integer
    grid points. ``sample = (index, pp)``.'''
    aw = _make_order_weight(weight, order)
    return SamplingKind(
        name='entries_derivatives',
        precompute=lambda base, s: pd.precompute_entries_base_sweep_jets(base, s[0], s[1], order),
        forward=lambda v, s, base, bs: pd.entries_jacobian_derivatives_from_sweep(v, s[0], s[1], base, bs, order),
        transpose=lambda r, s, base, bs: pd.entries_transpose_derivatives_from_sweep(
            aw(r, 2), s[0], s[1], base, bs, order, sum_over_probes=True),
        sumsq=lambda out, n_w: sumsq_over_samples(aw(out, 1), n_w + 1),
        w_axes=lambda s: s[0].ndim - 1,
        point_forward=lambda x_cores, s: pd.entries_derivatives_t3(s[0], s[1], x_cores, order),
        n_measurements=lambda s: _prod_w(s[0], 1, s[0].ndim - 1),
        take=_take_deriv_entries,
    )


def probe_derivatives_kind(
        order:  int,
        weight: typ.Optional[typ.Sequence[float]] = None,
) -> SamplingKind:                                  # sample = (ww, pp); data = list of d, (order+1)+W+(Ni,)
    '''The **probe-derivatives** sampling kind: vector-valued (one free mode per probe), so the residual
    / output is a list of ``d`` arrays. ``sample = (ww, pp)``.'''
    aw = _make_order_weight(weight, order)
    return SamplingKind(
        name='probe_derivatives',
        precompute=lambda base, s: pd.precompute_probe_base_sweep_jets(base, s[0], s[1], order),
        forward=lambda v, s, base, bs: pd.probe_jacobian_derivatives_from_sweep(v, s[0], s[1], base, bs, order),
        transpose=lambda r, s, base, bs: pd.probe_transpose_derivatives_from_sweep(
            aw(r, 2), s[0], s[1], base, bs, order, sum_over_probes=True),
        sumsq=lambda out, n_w: sumsq_over_probes(aw(out, 1), n_w + 1),
        w_axes=lambda s: s[0][0].ndim - 1,
        point_forward=lambda x_cores, s: pd.probe_derivatives_t3(s[0], s[1], x_cores, order),
        n_measurements=lambda s: _prod_w(s[0][0], 0, s[0][0].ndim - 1),
        take=_take_deriv_probe,
    )
