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

import typing as typ
from dataclasses import dataclass

import t3toolbox.backend.probing as probing
from t3toolbox.backend.common import *

__all__ = [
    'SamplingKind',
    'APPLY',
    'ENTRIES',
    'PROBE',
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
        zz:     typ.Sequence[NDArray],  # probe forward/residual, len=d, elm_shape=W+C+(Ni,)
        n_w:    int,                    # number of leading sample-stack (W) axes
) -> NDArray:                           # sum of squares over W and the free mode Ni, summed over d, keep C
    '''The ``‖·‖²`` reduction for the vector-output ``probe`` kind: each ``z_i`` is ``W+C+(Ni,)``; sum
    ``z_i**2`` over the leading ``n_w`` sample axes and the trailing free mode ``Ni``, keeping the base
    stack ``C``, summed over the ``d`` probes.'''
    use_jax = is_jax_ndarray(zz[0])
    xnp, _, _ = get_backend(False, use_jax)
    total = None
    for z in zz:
        axes = tuple(range(n_w)) + (z.ndim - 1,)        # W (leading) + the free mode Ni (last); keep C
        s = xnp.sum(z ** 2, axis=axes)
        total = s if total is None else total + s
    return total


@dataclass(frozen=True)
class SamplingKind:
    '''A sampling kind's bare primitives, bundled so the GN model is generic over the kind.

    Holds the kind-specific functions the geometry-generic Gauss-Newton model needs -- the bare ``𝒥`` /
    ``𝒥ᵀ`` (no gauge ``Π``: that is the geometry's), the ``‖·‖²`` reduction, and the sample-stack axis
    count. The model binds one of :py:data:`APPLY` / :py:data:`ENTRIES` / :py:data:`PROBE`; the geometry
    supplies ``Π`` around them. ``sample`` is the kind's measurement spec: the probe/apply vectors ``ww``
    (apply / probe) or the integer grid ``index`` (entries).
    '''
    name:       str           # 'apply' / 'entries' / 'probe'
    precompute: typ.Callable   # (base_data, sample)                      -> base_sweep
    forward:    typ.Callable   # (variations_data, sample, base_data, sweep) -> 𝒥 v  (the bare forward)
    transpose:  typ.Callable   # (residual, sample, base_data, sweep)     -> 𝒥ᵀ r  (summed over W; raw dU,dG)
    sumsq:      typ.Callable   # (forward_out, n_w)                        -> ‖forward_out‖²  (over W [+ Ni])
    w_axes:     typ.Callable   # (sample)                                 -> n_w  (leading sample-stack axes)


APPLY = SamplingKind(
    name='apply',
    precompute=lambda base, ww: probing.precompute_base_sweep(base, ww),
    forward=lambda v, ww, base, bs: probing.apply_jacobian_from_sweep(v, ww, base, bs),
    transpose=lambda r, ww, base, bs: probing.apply_transpose_from_sweep(r, ww, bs, sum_over_probes=True),
    sumsq=sumsq_over_samples,
    w_axes=lambda ww: ww[0].ndim - 1,
)

ENTRIES = SamplingKind(
    name='entries',
    precompute=lambda base, index: probing.precompute_entries_base_sweep(base, index),
    forward=lambda v, index, base, bs: probing.entries_jacobian_from_sweep(v, index, base, bs),
    transpose=lambda r, index, base, bs: probing.entries_transpose_from_sweep(r, index, base, bs, sum_over_probes=True),
    sumsq=sumsq_over_samples,
    w_axes=lambda index: index.ndim - 1,
)

PROBE = SamplingKind(
    name='probe',
    precompute=lambda base, ww: probing.precompute_base_sweep(base, ww),
    forward=lambda v, ww, base, bs: probing.probe_jacobian_from_sweep(v, ww, base, bs),
    transpose=lambda r, ww, base, bs: probing.probe_transpose_from_sweep(r, ww, base, bs, sum_over_probes=True),
    sumsq=sumsq_over_probes,
    w_axes=lambda ww: ww[0].ndim - 1,
)
