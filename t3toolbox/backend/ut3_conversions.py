# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.t3_operations as t3_ops
import t3toolbox.backend.ut3_masking as ut3_masking
from t3toolbox.backend.common import *

__all__ = [
    't3_to_ut3',
    'ut3_to_t3',
    'ut3_to_dense',
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
    NDArray,                                # tucker_supercore, shape=(d,)+stack_shape+(n,N)
    NDArray,                                # tt_supercore,     shape=(d,)+stack_shape+(r,n,r)
    typ.Tuple[NDArray, NDArray, NDArray],   # masks = (shape_mask, tucker_edge_mask, tt_edge_mask), HOST bool, static
]:
    """Convert a (ragged) TuckerTensorTrain core pair to uniform supercores + masks (nested .data layout).

    Pads each core to common sizes ``(n, N)`` / ``(r, n, r)``, stacks the ``d`` cores onto a leading
    axis, and records the real extent as prefix masks. ``use_jax`` is inferred from the input cores for
    the SUPERCORES; the masks are always numpy (host) structure (``docs/uniform_pytree_composition.md``).
    """
    use_jax = tree_contains_jax(x)
    xnp, _, _ = get_backend(False, use_jax)

    if squash_tails:
        x = (x[0], t3_ops.squash_tt_tails(x[1]))

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

    padded_tucker_cores = t3_ops.change_tucker_core_shapes(tucker_cores, padded_shape, padded_tucker_ranks)
    padded_tt_cores     = t3_ops.change_tt_core_shapes(tt_cores, padded_tucker_ranks, padded_tt_ranks)

    tucker_supercore = xnp.stack(padded_tucker_cores)
    tt_supercore     = xnp.stack(padded_tt_cores)

    masks = ut3_masking.make_uniform_masks(shape, tucker_ranks, tt_ranks, N, n, r)
    return tucker_supercore, tt_supercore, masks


def ut3_to_dense(
        x: typ.Tuple[
            NDArray,                                # tucker_supercore
            NDArray,                                # tt_supercore
            typ.Tuple[NDArray, NDArray, NDArray],   # masks, HOST bool, static
        ],
) -> NDArray:  # shape = stack_shape + (N0,...,N(d-1))
    """Form the dense tensor from a uniform Tucker tensor train: mask the supercores, chain-contract
    (shared with the ragged path), then static-prefix-slice the padded physical axes to the real shape.

    ``int(m.sum())`` pulls the real shape as host ints -- correct only because the masks are numpy (host);
    a traced mask would raise ``ConcretizationTypeError`` (``apply_masks_to_cores`` guards it first).

    Jitting this functionally (no frontend): close over the host masks as constants and trace only the
    supercores. Tracing the whole ``.data`` instead would put the masks among the traced args, which the
    guard rejects (``docs/uniform_pytree_composition.md``):

    >>> import numpy as np, jax
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.backend.ut3_conversions as ut3_conversions
    >>> np.random.seed(0)
    >>> ux = ut3.t3_to_ut3(t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))).to_jax()
    >>> tk, tt, masks = ux.data                                 # masks: HOST bool, static
    >>> dense = jax.jit(lambda a, b: ut3_conversions.ut3_to_dense((a, b, masks)))(tk, tt)  # RIGHT
    >>> bool(np.allclose(dense, ux.to_dense()))
    True
    >>> jax.jit(ut3_conversions.ut3_to_dense)(ux.data)  # WRONG: masks are traced args   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: uniform masks must be concrete host (numpy) arrays, ...
    """
    masked_tucker, masked_tt = ut3_masking.apply_masks_to_cores(x)
    T = t3_ops.t3_to_dense_chain(masked_tucker, masked_tt)   # stack + (N,)*d (padded)
    shape = [int(m.sum()) for m in x[2][0]]                  # x[2][0] = shape_mask (HOST bool), shape=(d, N)
    sl = (Ellipsis,) + tuple(slice(0, Ni) for Ni in shape)
    return T[sl]


def ut3_to_t3(
        x: typ.Tuple[
            NDArray,                                # tucker_supercore
            NDArray,                                # tt_supercore
            typ.Tuple[NDArray, NDArray, NDArray],   # masks = (shape_mask, tucker_edge_mask, tt_edge_mask)
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
    tucker_supercore, tt_supercore, (shape_masks, tucker_masks, tt_masks) = x
    ut3_masking.require_concrete_masks(shape_masks, tucker_masks, tt_masks)  # host masks: argwhere is np

    stack_shape = tucker_supercore.shape[1:-2]

    if not stack_shape:  # unstacked -> one ragged T3
        # np: the masks are host structure, so the real-index extraction runs on the host.
        shape_inds  = [np.argwhere(em).reshape(-1) for em in list(shape_masks)]
        tucker_inds = [np.argwhere(em).reshape(-1) for em in list(tucker_masks)]
        tt_inds     = [np.argwhere(em).reshape(-1) for em in list(tt_masks)]

        tucker_cores = tuple(
            B[ii, :][:, jj]
            for ii, jj, B in zip(tucker_inds, shape_inds, list(tucker_supercore))
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
            (shape_masks, tucker_masks[:, ii], tt_masks[:, ii]),
        )
        all_t3s.append(ut3_to_t3(xi))
    return tuple(all_t3s)
