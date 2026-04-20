# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

from t3toolbox.common import *

__all__ = [
    'contract_edge_vectors_into_t3',
    'reverse_edge_vectors',
    'concatenate_edge_vectors',
    'wt3_squash_tails',
]


def contract_edge_vectors_into_t3(
        x0: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
        edge_vectors: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_vectors, len=d, elm_shape=stack_shape+(ni,)
            typ.Sequence[NDArray],  # tt_vectors, len=d+1, elm_shape=stack_shape+(ri,)
        ],
        use_jax: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:
    """Contract each edge weight into a neighboring backend.
    """
    xnp, xmap, xscan = get_backend(False, use_jax)

    #
    tucker_cores0, tt_cores0 = x0
    tucker_weights, tt_weights = edge_vectors

    tucker_cores = []
    for tw, B in zip(tucker_weights, tucker_cores0):
        wB = xnp.einsum('...i,...io->...io', tw, B)
        tucker_cores.append(wB)

    tt_cores = []
    for lw, G in zip(tt_weights[:-2], tt_cores0[:-1]):
        wG = xnp.einsum('...i,...iaj->...iaj', lw, G)
        tt_cores.append(wG)

    Gf = xnp.einsum('...i,...iaj,...j->...iaj', tt_weights[-2], tt_cores0[-1], tt_weights[-1])
    tt_cores.append(Gf)

    return tuple(tucker_cores), tuple(tt_cores)


def reverse_edge_vectors(
        edge_vectors: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_weights, len=d, elm_shape=stack_shape+(ni,)
            typ.Sequence[NDArray],  # tt_weights, len=d+1, elm_shape=stack_shape+(ri,)
        ],
) -> typ.Tuple[
    typ.Sequence[NDArray],  # reversed_tucker_weights,
    typ.Sequence[NDArray],  # reversed_tt_weights,
]:
    return edge_vectors[0][::-1], edge_vectors[1][::-1]


def concatenate_edge_vectors(
        edge_vectors_A: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_weights, len=d, elm_shape=stack_shape+(ni,)
            typ.Sequence[NDArray],  # tt_weights, len=d+1, elm_shape=stack_shape+(ri,)
        ],
        edge_vectors_B: typ.Tuple[
            typ.Sequence[NDArray],  # tucker_weights, len=d, elm_shape=stack_shape+(ni,)
            typ.Sequence[NDArray],  # tt_weights, len=d+1, elm_shape=stack_shape+(ri,)
        ],
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Sequence[NDArray],  # reversed_tucker_weights,
    typ.Sequence[NDArray],  # reversed_tt_weights,
]:
    xnp, _, _ = get_backend(False, use_jax)

    tucker_A, tt_A = edge_vectors_A
    tucker_B, tt_B = edge_vectors_B

    tucker_AB = tuple([xnp.concatenate([vA, vB], axis=-1) for vA, vB in zip(tucker_A, tucker_B)])
    tt_AB = tuple([xnp.concatenate([vA, vB], axis=-1) for vA, vB in zip(tt_A, tt_B)])

    return tucker_AB, tt_AB


def wt3_squash_tails(
        x, # weighted Tucker tensor train
        use_jax: bool = False,
):
    """Reduce the first and last dimensions of the first and last tt cores to 1.
    """
    xnp, _, _ = get_backend(False, use_jax=use_jax)

    x0, w = x
    tucker_cores, tt_cores = x0
    tucker_weights, tt_weights = w

    stack_shape = tucker_weights[0].shape[:-1]

    first_G = xnp.einsum('...aib,...a->...aib', tt_cores[0], tt_weights[0])
    first_G = first_G.sum(axis=-3, keepdims=True)
    first_wtt = xnp.ones(stack_shape + (1,))

    mid_G = tt_cores[1:-1]
    mid_wtt = tt_weights[1:-1]

    last_G = xnp.einsum('...aib,...b->...aib', tt_cores[-1], tt_weights[-1])
    last_G = last_G.sum(axis=-1, keepdims=True)
    last_wtt = xnp.ones(stack_shape + (1,))

    tt_cores = (first_G,) + mid_G + (last_G,)
    tt_weights = (first_wtt,) + mid_wtt + (last_wtt,)

    x0 = (tucker_cores, tt_cores)
    w = (tucker_weights, tt_weights)
    return (x0, w)

