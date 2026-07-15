# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Ragged <-> uniform <-> dense conversions of plain T3 data.

``t3_to_ut3``/``ut3_to_t3``/``ut3_to_dense`` -- the seam the uniform equivalence contract is
stated through: ``to_uniform -> op -> to_ragged == op_ragged`` on real parts, garbage
don't-care (``docs/uniform_equivalence_contract.md``). ``t3weights_to_ut3weights`` /
``ut3weights_to_t3weights`` are the same seam for the weighted layer, and are how every uniform
weight op is tested against its ragged twin.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.t3_operations as t3_ops
import t3toolbox.backend.ut3_masking as ut3_masking
from t3toolbox.backend.common import *

__all__ = [
    't3_to_ut3',
    'ut3_to_t3',
    'ut3_to_dense',
    't3weights_to_ut3weights',
    'ut3weights_to_t3weights',
]


def t3_to_ut3(
        x: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_cores, len=d, elm_shape=stack_shape+(ni, Ni)
            typ.Sequence[NDArray],  # tt_cores,     len=d, elm_shape=stack_shape+(ri, ni, r(i+1))
        ],
        N: int = None,              # padded mode dim   (default max(Ni)); pass to force a larger pad
        n: int = None,              # padded Tucker rank (default max(tucker_ranks))
        r: int = None,              # padded TT rank    (default max(tt_ranks))
        squash_tails: bool = True,
) -> typ.Tuple[
    NDArray,                          # tucker_supercore, shape=(d,)+stack_shape+(n,N)
    NDArray,                          # tt_supercore,     shape=(d,)+stack_shape+(r,n,r)
    typ.Tuple[int, ...],              # shape = (N0,...,N(d-1)), static int tuple
    typ.Tuple[NDArray, NDArray],      # (tucker_edge_mask, tt_edge_mask), HOST bool, static
]:
    """Convert a (ragged) TuckerTensorTrain core pair to uniform supercores + shape + masks (nested .data).

    Pads each core to common sizes ``(n, N)`` / ``(r, n, r)``, stacks the ``d`` cores onto a leading
    axis, and records the real extent as prefix masks. ``use_jax`` is inferred from the input cores for
    the SUPERCORES; the masks are always numpy (host) structure (``docs/contributor/uniform_pytree_composition.md``).
    """
    use_jax = tree_contains_jax(x)
    xnp, _, _ = get_backend(False, use_jax)

    if squash_tails:
        x = (x[0], tt_operations.tt_squash_tails(x[1]))

    tucker_cores, tt_cores = x
    stack_shape = tucker_cores[0].shape[:-2]
    ones_stack = np.ones(stack_shape, dtype=int)  # np: ranks are mask (host) metadata, not supercore data

    shape        = tuple(B.shape[-1] for B in tucker_cores)
    tucker_ranks = np.stack([B.shape[-2] * ones_stack for B in tucker_cores])                  # (d,)   + stack
    tt_ranks     = np.stack([G.shape[-3] * ones_stack for G in tt_cores]
                            + [tt_cores[-1].shape[-1] * ones_stack])                            # (d+1,) + stack

    d = len(shape)
    N = max(shape)                if N is None else N
    n = int(np.max(tucker_ranks)) if n is None else n
    r = int(np.max(tt_ranks))     if r is None else r

    padded_shape        = (N,) * d
    padded_tucker_ranks = (n,) * d
    padded_tt_ranks     = (r,) * (d + 1)

    padded_tucker_cores = t3_ops.tucker_change_core_shapes(tucker_cores, padded_shape, padded_tucker_ranks)
    padded_tt_cores     = tt_operations.tt_change_core_shapes(tt_cores, padded_tucker_ranks, padded_tt_ranks)

    tucker_supercore = xnp.stack(padded_tucker_cores)
    tt_supercore     = xnp.stack(padded_tt_cores)

    masks = ut3_masking.ut3_make_masks(tucker_ranks, tt_ranks, n, r)
    return tucker_supercore, tt_supercore, shape, masks


def ut3_to_dense(
        x: typ.Tuple[
            NDArray,                          # tucker_supercore
            NDArray,                          # tt_supercore
            typ.Tuple[int, ...],              # shape, static int tuple
            typ.Tuple[NDArray, NDArray],      # (tucker_edge_mask, tt_edge_mask), HOST bool, static
        ],
) -> NDArray:  # shape = stack_shape + (N0,...,N(d-1))
    """Form the dense tensor from a uniform Tucker tensor train: mask the supercores, chain-contract
    (shared with the ragged path), then static-prefix-slice the padded physical axes to the real shape.

    The real ``shape`` is the static int tuple ``x[2]``; the padded physical axes are sliced to it.

    Jitting this functionally (no frontend): close over the host masks as constants and trace only the
    supercores. Tracing the whole ``.data`` instead would put the masks among the traced args, which the
    guard rejects (``docs/contributor/uniform_pytree_composition.md``):

    >>> import numpy as np, jax
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.backend.ut3_conversions as ut3_conversions
    >>> np.random.seed(0)
    >>> ux = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))).to_jax()
    >>> tk, tt, shape, masks = ux.data                          # shape ints + HOST bool masks, static
    >>> dense = jax.jit(lambda a, b: ut3_conversions.ut3_to_dense((a, b, shape, masks)))(tk, tt)  # RIGHT
    >>> bool(np.allclose(dense, ux.to_dense()))
    True
    >>> jax.jit(ut3_conversions.ut3_to_dense)(ux.data)  # WRONG: masks are traced args   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: uniform masks must be concrete host (numpy) arrays, ...
    """
    masked_tucker, masked_tt = ut3_masking.ut3_apply_masks(x)
    T = t3_conversions.t3_to_dense_chain(masked_tucker, masked_tt)   # stack + (N,)*d (padded)
    shape = x[2]                                             # static int tuple (N0,...,N(d-1))
    sl = (Ellipsis,) + tuple(slice(0, Ni) for Ni in shape)
    return T[sl]


def ut3_to_t3(
        x: typ.Tuple[
            NDArray,                          # tucker_supercore
            NDArray,                          # tt_supercore
            typ.Tuple[int, ...],              # shape, static int tuple
            typ.Tuple[NDArray, NDArray],      # (tucker_edge_mask, tt_edge_mask)
        ],
) -> typ.Union[
    typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]],  # (tucker_cores, tt_cores), if unstacked
    typ.Tuple,                                                     # else a nested tree (shape stack_shape) of those
]:
    """Convert uniform supercores + masks back to ragged TuckerTensorTrain core pairs.

    Unstacked: returns one ``(tucker_cores, tt_cores)``. Stacked: returns a nested tuple (shaped like
    ``stack_shape``) of such pairs -- a *tree*, since a varying-rank stack has no single stacked
    ``TuckerTensorTrain`` (``docs/uniform_ranks_and_varieties.md``). The real sub-blocks are extracted by
    ``argwhere`` (handles gappy edge masks; ``docs/uniform_masks_vs_ranks.md``).
    """
    tucker_supercore, tt_supercore, shape, (tucker_masks, tt_masks) = x
    require_concrete_masks(tucker_masks, tt_masks)  # host masks: argwhere is np

    stack_shape = tucker_supercore.shape[1:-2]

    if not stack_shape:  # unstacked -> one ragged T3
        # np: the masks are host structure, so the real-index extraction runs on the host. The physical
        # `shape` is a contiguous prefix, so it slices [:Ni] (no argwhere); only the rank masks scatter.
        tucker_inds = [np.argwhere(em).reshape(-1) for em in list(tucker_masks)]
        tt_inds     = [np.argwhere(em).reshape(-1) for em in list(tt_masks)]

        tucker_cores = tuple(
            B[ii, :][:, :Ni]
            for ii, Ni, B in zip(tucker_inds, shape, list(tucker_supercore))
        )
        tt_cores = tuple(
            G[ii, :, :][:, aa, :][:, :, jj]
            for ii, aa, jj, G in zip(tt_inds[:-1], tucker_inds, tt_inds[1:], list(tt_supercore))
        )
        return tucker_cores, tt_cores

    all_t3s = []
    for ii in range(tucker_supercore.shape[1]):
        xi = (
            tucker_supercore[:, ii],
            tt_supercore[:, ii],
            shape,
            (tucker_masks[:, ii], tt_masks[:, ii]),
        )
        all_t3s.append(ut3_to_t3(xi))
    return tuple(all_t3s)


def t3weights_to_ut3weights(
        weights: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_weights, len=d,   elm_shape=stack_shape+(ni,)
            typ.Sequence[NDArray],  # tt_weights,     len=d+1, elm_shape=stack_shape+(ri,)
        ],
        n: int = None,              # padded Tucker rank (default max(tucker rank)); pass to force a larger pad
        r: int = None,              # padded TT rank     (default max(tt rank))
) -> typ.Tuple[
    NDArray,                      # tucker_weight_supercore, shape=(d,)  +stack_shape+(n,)
    NDArray,                      # tt_weight_supercore,     shape=(d+1,)+stack_shape+(r,)
    typ.Tuple[NDArray, NDArray],  # (tucker_edge_mask, tt_edge_mask), HOST bool, static
]:
    """Pack a ragged ``T3Weights`` core pair into uniform weight supercores + masks (the ``.data`` layout).

    The weight twin of :py:func:`t3_to_ut3`, and much simpler: a weight is one vector per edge, so there
    is nothing to pad but the last axis and no physical ``shape`` at all (weights live only on internal
    edges). Each vector is zero-padded to the common width and the real extent recorded as a prefix mask.

    The **zero** pad matters and is not a free choice: masking downstream works by multiplication, so the
    fill must stay finite (``docs/uniform_equivalence_contract.md``), and zero is the layer's canonical
    clean padding. It is emphatically *not* an identity for weighting -- absorbing a zero-padded weight
    zeroes the padding of the object it weights, which is exactly right (that padding is don't-care), but
    it is why ``ut3_reciprocal_weights`` cannot naively divide (``1/0 = inf``).

    Pass ``n``/``r`` to match the padding of the object these weights will pair with; the defaults pad
    tightly to the weights' own max rank. ``use_jax`` is inferred for the SUPERCORES; the masks are always
    numpy (host) structure.
    """
    use_jax = tree_contains_jax(weights)
    xnp, _, _ = get_backend(False, use_jax)

    tucker_weights, tt_weights = weights
    d = len(tucker_weights)
    stack_shape = tucker_weights[0].shape[:-1]
    ones_stack = np.ones(stack_shape, dtype=int)  # np: ranks are (host) structure, not supercore data

    tucker_ranks = np.stack([w.shape[-1] * ones_stack for w in tucker_weights])  # (d,)   + stack
    tt_ranks     = np.stack([w.shape[-1] * ones_stack for w in tt_weights])      # (d+1,) + stack

    n = int(np.max(tucker_ranks)) if n is None else n
    r = int(np.max(tt_ranks))     if r is None else r

    def pad_to(ws, width):
        out = []
        for w in ws:
            pad = ((0, 0),) * len(stack_shape) + ((0, width - w.shape[-1]),)
            out.append(xnp.pad(w, pad))
        return xnp.stack(out)

    tucker_weight_supercore = pad_to(tucker_weights, n)
    tt_weight_supercore     = pad_to(tt_weights, r)

    # prefix_mask, not the masking layer: weighting and masking each call the shared primitive.
    masks = (prefix_mask(tucker_ranks, n), prefix_mask(tt_ranks, r))
    return tucker_weight_supercore, tt_weight_supercore, masks


def ut3weights_to_t3weights(
        weights: typ.Tuple[
            NDArray,                      # tucker_weight_supercore, (d,)  +stack_shape+(n,)
            NDArray,                      # tt_weight_supercore,     (d+1,)+stack_shape+(r,)
            typ.Tuple[NDArray, NDArray],  # (tucker_edge_mask, tt_edge_mask)
        ],
) -> typ.Union[
    typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]],  # (tucker_weights, tt_weights), if unstacked
    typ.Tuple,                                                     # else a nested tree (shaped stack_shape)
]:
    """Convert uniform weight supercores + masks back to ragged ``T3Weights`` core pairs.

    The weight twin of :py:func:`ut3_to_t3`, and it inherits that function's two honest awkwardnesses:
    the real slots are gathered by ``argwhere`` (an edge mask may be **gappy** after concat/Kronecker --
    ``docs/uniform_masks_vs_ranks.md`` -- so this is a gather, not a slice), and a **stacked** weight
    returns a *tree* of ragged weights rather than one stacked weight, since a varying-rank stack has no
    single ragged representation (``docs/uniform_ranks_and_varieties.md``).
    """
    tucker_weight_supercore, tt_weight_supercore, (tucker_mask, tt_mask) = weights
    require_concrete_masks(tucker_mask, tt_mask)  # host masks: argwhere is np

    stack_shape = tucker_weight_supercore.shape[1:-1]

    if not stack_shape:  # unstacked -> one ragged weight pair
        # np: the masks are host structure, so the real-index extraction runs on the host.
        tucker_out = tuple(w[np.argwhere(m).reshape(-1)]
                           for m, w in zip(list(tucker_mask), list(tucker_weight_supercore)))
        tt_out = tuple(w[np.argwhere(m).reshape(-1)]
                       for m, w in zip(list(tt_mask), list(tt_weight_supercore)))
        return tucker_out, tt_out

    return tuple(
        ut3weights_to_t3weights((
            tucker_weight_supercore[:, ii],
            tt_weight_supercore[:, ii],
            (tucker_mask[:, ii], tt_mask[:, ii]),
        ))
        for ii in range(tucker_weight_supercore.shape[1])
    )
