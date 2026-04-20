# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tucker_tensor_train.ragged_t3_operations as t3_ops
from t3toolbox.backend.common import *

__all__ = [
    'reverse_utt',
    # 'absorb_edge_weights_into_ut3',
    'uniform_squash_tt_tails',
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


def make_uniform_masks(
        shape:          typ.Tuple[int,...],
        tucker_ranks:   typ.Tuple[int,...],
        tt_ranks:       typ.Tuple[int,...],
        stack_shape:    typ.Tuple[int,...],
        N: int,
        n: int,
        r: int,
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # shape_mask, dtype=bool, shape=(d,N)
    NDArray, # tucker_edge_masks, dtype=bool, shape=(d,)+stack_shape+(n,)
    NDArray, # tt_edge_masks, dtype=bool, shape=(d,)+stack_shape+(r,)
]:
    xnp, xmap, xscan = get_backend(False, use_jax)


    shape_masks = xnp.stack([
        xnp.concatenate([
            xnp.ones((Ni,), dtype=bool),
            xnp.zeros((N-Ni,), dtype=bool),
        ], axis=-1,
        )
        for Ni in shape
    ])

    tucker_masks = xnp.stack([
        xnp.concatenate([
            xnp.ones(stack_shape+(ni,), dtype=bool),
            xnp.zeros(stack_shape+(n-ni,), dtype=bool)
        ], axis=-1,
        )
        for ni in tucker_ranks
    ])

    tt_masks = xnp.stack([
        xnp.concatenate([
            xnp.ones(stack_shape+(ri,), dtype=bool),
            xnp.zeros(stack_shape+(r-ri,), dtype=bool)
        ], axis=-1,
        )
        for ri in tt_ranks
    ])

    return shape_masks, tucker_masks, tt_masks


def t3_to_ut3(
        x: typ.Tuple[
            typ.Tuple[NDArray,...], # tt_cores
            typ.Tuple[NDArray,...], # tucker_cores
        ],
        squash_tails: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # tucker_supercore
    NDArray, # tt_supercore
    NDArray, # shape_mask
    NDArray, # tucker_edge_mask
    NDArray, # tt_edge_mask
]:
    """Convert TuckerTensorTrain to UniformTuckerTensorTrain.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    if squash_tails:
        x = (x[0], t3_ops.squash_tt_tails(x[1], use_jax=use_jax))

    tucker_cores, tt_cores = x

    shape = tuple([B.shape[-1] for B in tucker_cores])
    tucker_ranks = tuple([B.shape[-2] for B in tucker_cores])
    tt_ranks = tuple([G.shape[-3] for G in tt_cores]) + (tt_cores[-1].shape[-1],)
    stack_shape = tucker_cores[0].shape[:-2]

    d = len(shape)
    N = max(shape)
    n = max(tucker_ranks)
    r = max(tt_ranks)

    padded_shape = (N,)*d
    padded_tucker_ranks = (n,)*d
    padded_tt_ranks = (r,)*(d+1)

    padded_tucker_cores = t3_ops.change_tucker_core_shapes(
        tucker_cores, padded_shape, padded_tucker_ranks, use_jax=use_jax,
    )
    padded_tt_cores = t3_ops.change_tt_core_shapes(
        tt_cores, padded_tucker_ranks, padded_tt_ranks, use_jax=use_jax,
    )

    tucker_supercore = xnp.stack(padded_tucker_cores)
    tt_supercore = xnp.stack(padded_tt_cores)

    shape_masks, tucker_masks, tt_masks = make_uniform_masks(
        shape, tucker_ranks, tt_ranks, stack_shape, N, n, r,
    )

    return tucker_supercore, tt_supercore, shape_masks, tucker_masks, tt_masks


def ut3_to_t3(
        x: typ.Tuple[
            NDArray, # tucker_supercore
            NDArray, # tt_supercore
            NDArray, # shape_mask
            NDArray, # tucker_edge_mask
            NDArray, # tt_edge_mask
        ],
        stack_t3s: bool = False,
        use_jax: bool = False,
) -> typ.Union[
    typ.Tuple, #
    typ.Tuple[
        typ.Tuple[NDArray,...], # stacked_tt_cores
        typ.Tuple[NDArray,...], # stacked_tucker_cores
    ],
]:
    '''Convert UniformTuckerTensorTrain to TuckerTensorTrain.

    If uniform T3 is stacked, either:
        - return an array-line nesting of tuples containing the T3s (stack_t3s=False),
        - or one stacked T3 (stack_t3s=True)

    Can only return a stacked T3 if the stacked UT3s all have the same structure.
    '''
    xnp, _, _ = get_backend(True, use_jax)

    #
    tucker_supercore, tt_supercore, shape_masks, tucker_masks, tt_masks = x
    stack_shape = tucker_supercore[0].shape[:-2]

    if not stack_shape: # not stacked
        shape_inds  = [xnp.argwhere(em).reshape(-1) for em in list(shape_masks)]
        tucker_inds = [xnp.argwhere(em).reshape(-1) for em in list(tucker_masks)]
        tt_inds     = [xnp.argwhere(em).reshape(-1) for em in list(tt_masks)]

        tucker_cores = tuple([
            B[ii,:][:,jj]
            for ii, jj, B
            in zip(tucker_inds, shape_inds, list(tucker_supercore))
        ])
        tt_cores = tuple([
            G[ii, :, :][:,aa,:][:, :, jj]
            for ii, aa, jj, G
            in zip(tt_inds[:-1], tucker_inds, tt_inds[1:], list(tt_supercore))
        ])
        return tucker_cores, tt_cores

    all_T3s = []
    for ii in range(tucker_supercore.shape[1]):
        xi = (
            tucker_supercore[:, ii],
            tt_supercore[:, ii],
            shape_masks,
            tucker_masks[:, ii],
            tt_masks[:, ii],
        )
        ith_t3 = ut3_to_t3(xi, use_jax=use_jax)
        all_T3s.append(ith_t3)

    all_T3s = tuple(all_T3s)

    if stack_t3s:
        all_T3s = t3_ops.t3_stack(all_T3s, use_jax=use_jax)

    return all_T3s


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


def apply_masks_to_cores(
        x: typ.Tuple[
            NDArray,  # tucker_supercore
            NDArray,  # tt_supercore
            NDArray,  # shape_mask
            NDArray,  # tucker_edge_mask
            NDArray,  # tt_edge_mask
        ],
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # masked_tucker_supercore
    NDArray, # masked_tt_supercore
]:
    """Applies masking to supercores, replacing unmasked regions with zeros.
    """
    xnp,_,_ = get_backend(True, use_jax)

    tucker_supercore, tt_supercore, shape_mask, tucker_edge_mask, tt_edge_mask = x

    masked_tucker_supercore = xnp.einsum(
        'd...nN,d...n,dN->d...nN',
        tucker_supercore, tucker_edge_mask, shape_mask,
    )
    masked_tt_supercore = xnp.einsum(
        'd...lnr,d...l,d...n,d...r->d...lnr',
        tt_supercore, tt_edge_mask[:-1], tucker_edge_mask, tt_edge_mask[1:],
    )
    return masked_tucker_supercore, masked_tt_supercore
