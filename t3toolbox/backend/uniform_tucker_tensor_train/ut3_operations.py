# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.uniform_tucker_tensor_train.ut3_masking as ut3_masking
from t3toolbox.backend.common import *

__all__ = [
    'reverse_utt',
    'uniform_squash_tt_tails',
    'ut3_unstack',
    'ut3_stack',
]


def reverse_utt(
        tt_cores: typ.Union[typ.Sequence[NDArray], NDArray]
) -> typ.Union[typ.Sequence[NDArray], NDArray]:
    """Reverse a uniform tensor train (no Tucker).
    """
    return tt_cores[::-1].swapaxes(-3, -1)


def uniform_squash_tt_tails(
        tt_supercore: NDArray,
        use_jax: bool = False,
) -> NDArray: # new_tt_supercore
    """Squash tails of uniform tensor train.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.tucker_tensor_train.uniform.operations as uniform_operations
    >>> tt_supercore = np.random.randn(4, 2,3, 5,6,5)
    >>> new_tt_supercore = uniform_operations.uniform_squash_tt_tails(tt_supercore)
    >>> print(np.linalg.norm(np.sum(tt_supercore[0], axis=-3) - new_tt_supercore[0, :,:, 0,:,:]))
    0.0
    >>> print(np.linalg.norm(new_tt_supercore[0, :,:, 1:,:,:]))
    0.0
    >>> print(np.linalg.norm(np.sum(tt_supercore[-1], axis=-1) - new_tt_supercore[-1, :,:, :,:,0]))
    0.0
    >>> print(np.linalg.norm(new_tt_supercore[-1, :,:, :,:,1:]))
    0.0
    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    d = tt_supercore.shape[0]
    stack_shape = tt_supercore.shape[1:-3]
    n = tt_supercore.shape[-2]
    r = tt_supercore.shape[-1]

    G0 = tt_supercore[:1] # shape=(1,)+stack_shape+(r,n,r)
    new_G0_flat = xnp.sum(G0, axis=-3, keepdims=True) # shape=(1,)+stack_shape+(1,n,r)
    Z0_flat = xnp.zeros((1,)+stack_shape+(r-1,n,r))
    new_G0 = xnp.concatenate([new_G0_flat, Z0_flat], axis=-3) # shape=(1,)+stack_shape+(r,n,r)

    GG_mid = tt_supercore[1:-1] # shape=(d-2,)+stack_shape+(r,n,r)

    Gf = tt_supercore[-1:] # shape=(1,)+stack_shape+(r,n,r)
    new_Gf_flat = xnp.sum(Gf, axis=-1, keepdims=True) # shape=(1,)+stack_shape+(r,n,1)
    Zf_flat = xnp.zeros((1,)+stack_shape+(r,n,r-1))
    new_Gf = xnp.concatenate([new_Gf_flat, Zf_flat], axis=-1)  # shape=(1,)+stack_shape+(r,n,r)

    new_tt_supercore = xnp.concatenate([new_G0, GG_mid, new_Gf], axis=0)
    return new_tt_supercore


def pack_vectors(
        unpacked_vectors = typ.Sequence[NDArray], # len=d, ith_elm.shape=stack_shape+(Ni,)
        N: int = None,
        use_jax: bool = False,
) -> NDArray: # packed_vectors, shape=(d,)+stack_shape+(N,), where N=max(N0,...,N(d-1))
    """Use zero-padding to pack several vectors with ragged shapes into one tensor with an extra dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.tucker_tensor_train.uniform.uniform_t3_operations as ut3_ops
    >>> vv = [np.random.randn(2,3, 6), np.random.randn(2,3, 4), np.random.randn(2,3, 5)]
    >>> packed_vv = ut3_ops.pack_vectors(vv)
    >>> print(packed_vv.shape)
    (3, 2, 3, 6)
    >>> print(np.linalg.norm(vv[1] - packed_vv[1,:,:,:4]))
    0.0
    >>> print(np.linalg.norm(packed_vv[1,:,:,4:]))
    0.0
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    if not unpacked_vectors:
        return xnp.array(())

    stack_shape = unpacked_vectors[0].shape[:-1]

    if N is None:
        N = max([v.shape[-1] for v in unpacked_vectors])

    padded_vectors_list = []
    for v in unpacked_vectors:
        pad = ((0,0),)*len(stack_shape) + ((0, N - v.shape[-1]),)

        padded_v = xnp.pad(v, pad)
        padded_vectors_list.append(padded_v)

    packed_vectors = xnp.stack(padded_vectors_list)
    return packed_vectors


def unpack_vectors(
        packed_vectors: NDArray, # shape=(d,)+stack_shape+Ni
        unpacking_shape: typ.Sequence[int], # (N0,...,N(d-1))
) -> typ.Sequence[NDArray]:
    """Unpacks stacked vectors of size N,...,N into tuple of vectors of size N0,...,N(d-1).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.tucker_tensor_train.uniform.uniform_t3_operations as ut3_ops
    >>> vv = [np.random.randn(2,3, 6), np.random.randn(2,3, 4), np.random.randn(2,3, 5)]
    >>> packed_vv = ut3_ops.pack_vectors(vv)
    >>> vv2 = ut3_ops.unpack_vectors(packed_vv, [v.shape[-1] for v in vv])
    >>> for v, v2 in zip(vv, vv2): print(np.linalg.norm(v - v2))
    0.0
    0.0
    0.0
    """
    return tuple([
        packed_vectors[ii, ..., :unpacking_shape[ii]]
        for ii in range(len(unpacking_shape))
    ])


def ut3_unstack(
        x: typ.Tuple[
            NDArray,  # tucker_supercore
            NDArray,  # tt_supercore
            NDArray,  # shape_mask
            NDArray,  # tucker_edge_mask
            NDArray,  # tt_edge_mask
        ],
):
    """Unstacks this uniform Tucker tensor train into an array-like tree
    of nested Tuples containing uniform Tucker tensor trains as leaf nodes.
    """
    (tucker_supercore, tt_supercore, shape_mask, tucker_edge_mask, tt_edge_mask) = x
    stack_shape = tucker_supercore.shape[1:-2]

    stacked = (tucker_supercore, tt_supercore, tucker_edge_mask, tt_edge_mask) # no shape_mask
    axes = tuple(range(1, 1+len(stack_shape)))

    unstacked = stacking.unstack(stacked, axes=axes)

    def _func(x):
        if is_ndarray(x[0]):
            tk, tt, mtk, mtt = x
            return (tk, tt, shape_mask, mtk, mtt)

        return tuple(_func(xi) for xi in x)

    return _func(unstacked)


def ut3_stack(
        xx,
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray,  # tucker_supercore
    NDArray,  # tt_supercore
    NDArray,  # shape_mask
    NDArray,  # tucker_edge_mask
    NDArray,  # tt_edge_mask
]:
    """Stacks array-like nested tree of uniform Tucker tensor trains into one uniform Tucker tensor train.
    """
    tk0 = stacking.get_first_leaf(xx)
    stack_shape = tk0.shape[1:-2]
    axes = tuple(range(1, 1 + len(stack_shape)))

    return stacking.stack(xx, axes, use_jax=use_jax)



# def ut3_sum_stack(
#     x: typ.Tuple[
#         NDArray,  # tucker_supercore
#         NDArray,  # tt_supercore
#         NDArray,  # shape_mask
#         NDArray,  # tucker_edge_mask
#         NDArray,  # tt_edge_mask
#     ],
#         use_jax: bool = False,
# ) -> typ.Tuple[
#     NDArray,  # summed_tucker_supercore
#     NDArray,  # summed_tt_supercore
#     NDArray,  # summed_shape_mask
#     NDArray,  # summed_tucker_edge_mask
#     NDArray,  # summed_tt_edge_mask
# ]:
#     x = ut3_masking.apply_masks_to_cores(x, use_jax=use_jax)
#     tk, tt, sm, tkm, ttm = x
#     stack_shape = tk.shape[1:-2]
#     axes = tuple(range(1, 1 + len(stack_shape)))
#
#     summed_tk = tk.sum(axis=axes)
#     summed_tt = tt.sum(axis=axes)
#
#


