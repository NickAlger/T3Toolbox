# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.stacking as stacking
from t3toolbox.backend.common import *

__all__ = [
    'reverse_utt',
    'uniform_squash_tt_tails',
    'pack_vectors',
    'unpack_vectors',
    'ut3_unstack',
    'ut3_stack',
]

# A uniform-T3 leaf in nested .data layout: (tucker_supercore, tt_supercore, (shape_mask, tucker_mask, tt_mask)).
_UT3_LEAF_STRUCTURE = (None, None, (None, None, None))


def reverse_utt(
        tt_supercore: NDArray,  # shape=(d,)+stack_shape+(r,n,r)
) -> NDArray:                   # reversed,  shape=(d,)+stack_shape+(r,n,r)
    """Reverse a uniform tensor train: reverse the mode order and swap the two bond axes of each core.
    """
    return tt_supercore[::-1].swapaxes(-3, -1)


def uniform_squash_tt_tails(
        tt_supercore: NDArray,  # shape=(d,)+stack_shape+(r,n,r)
) -> NDArray:                   # shape=(d,)+stack_shape+(r,n,r), leading/trailing bond summed into slot 0
    """Make the leading bond of the first TT core and the trailing bond of the last collapse to one,
    by summing them into slot 0 (and zeroing the rest), so the represented tensor is unchanged.
    """
    use_jax = is_jax_ndarray(tt_supercore)
    xnp, _, _ = get_backend(True, use_jax)

    stack_shape = tt_supercore.shape[1:-3]
    n = tt_supercore.shape[-2]
    r = tt_supercore.shape[-1]

    G0 = tt_supercore[:1]                                              # (1,)+stack+(r,n,r)
    new_G0 = xnp.concatenate([
        xnp.sum(G0, axis=-3, keepdims=True),                          # (1,)+stack+(1,n,r)
        xnp.zeros((1,) + stack_shape + (r - 1, n, r)),
    ], axis=-3)

    GG_mid = tt_supercore[1:-1]

    Gf = tt_supercore[-1:]                                            # (1,)+stack+(r,n,r)
    new_Gf = xnp.concatenate([
        xnp.sum(Gf, axis=-1, keepdims=True),                          # (1,)+stack+(r,n,1)
        xnp.zeros((1,) + stack_shape + (r, n, r - 1)),
    ], axis=-1)

    return xnp.concatenate([new_G0, GG_mid, new_Gf], axis=0)


def pack_vectors(
        unpacked_vectors: typ.Sequence[NDArray],  # len=d, ith elm_shape=stack_shape+(Ni,)
        N: int = None,                            # padded length (default max(Ni))
) -> NDArray:                                     # packed, shape=(d,)+stack_shape+(N,)
    """Zero-pad and stack a sequence of (ragged-length) vectors into one supercore-shaped tensor.
    """
    if not unpacked_vectors:
        return np.array(())
    use_jax = tree_contains_jax(unpacked_vectors)
    xnp, _, _ = get_backend(False, use_jax)

    stack_shape = unpacked_vectors[0].shape[:-1]
    if N is None:
        N = max(v.shape[-1] for v in unpacked_vectors)

    padded = []
    for v in unpacked_vectors:
        pad = ((0, 0),) * len(stack_shape) + ((0, N - v.shape[-1]),)
        padded.append(xnp.pad(v, pad))
    return xnp.stack(padded)


def unpack_vectors(
        packed_vectors:  NDArray,            # shape=(d,)+stack_shape+(N,)
        unpacking_shape: typ.Sequence[int],  # (N0,...,N(d-1))
) -> typ.Tuple[NDArray, ...]:                # len=d, ith elm_shape=stack_shape+(Ni,)
    """Slice a packed supercore-shaped tensor back into a tuple of (ragged-length) vectors.
    """
    return tuple(
        packed_vectors[ii, ..., :unpacking_shape[ii]]
        for ii in range(len(unpacking_shape))
    )


def ut3_unstack(
        x: typ.Tuple[
            NDArray,                                # tucker_supercore
            NDArray,                                # tt_supercore
            typ.Tuple[NDArray, NDArray, NDArray],   # (shape_mask, tucker_edge_mask, tt_edge_mask)
        ],
):  # -> nested tuple (shaped like stack_shape) of unstacked uniform-T3 .data leaves
    """Unstack a uniform Tucker tensor train into an array-like tree of unstacked ones.

    The stack lives at axes ``1 .. len(stack_shape)`` (axis 0 is the mode index ``d``). The supercores
    and the rank masks unstack along it; ``shape_mask`` is shared and replicated onto every leaf.
    """
    tucker_supercore, tt_supercore, (shape_mask, tucker_edge_mask, tt_edge_mask) = x
    stack_shape = tucker_supercore.shape[1:-2]
    axes = tuple(range(1, 1 + len(stack_shape)))

    tree = stacking.unstack((tucker_supercore, tt_supercore, tucker_edge_mask, tt_edge_mask), axes=axes)

    return stacking.apply_func_to_leaf_subtrees(
        tree,
        lambda leaf: (leaf[0], leaf[1], (shape_mask, leaf[2], leaf[3])),
        (None, None, None, None),
    )


def ut3_stack(
        xx,  # nested tuple (shaped like stack_shape) of unstacked uniform-T3 .data leaves
) -> typ.Tuple[
    NDArray,                                # tucker_supercore
    NDArray,                                # tt_supercore
    typ.Tuple[NDArray, NDArray, NDArray],   # (shape_mask, tucker_edge_mask, tt_edge_mask)
]:
    """Stack an array-like tree of uniform Tucker tensor trains into one.

    Inverse of :py:func:`ut3_unstack`: stacks the supercores and rank masks onto axes
    ``1 .. num_levels`` (after the mode index), keeping the shared ``shape_mask`` unstacked.
    """
    shape_mask = stacking.get_first_leaf(
        stacking.apply_func_to_leaf_subtrees(xx, lambda leaf: leaf[2][0], _UT3_LEAF_STRUCTURE)
    )

    flat_tree = stacking.apply_func_to_leaf_subtrees(
        xx,
        lambda leaf: (leaf[0], leaf[1], leaf[2][1], leaf[2][2]),
        _UT3_LEAF_STRUCTURE,
    )

    num_levels = tree_depth_of_tree_over_leaf(flat_tree)
    axes = tuple(range(1, 1 + num_levels))

    tucker_supercore, tt_supercore, tucker_edge_mask, tt_edge_mask = stacking.stack(flat_tree, axes)
    return tucker_supercore, tt_supercore, (shape_mask, tucker_edge_mask, tt_edge_mask)


def tree_depth_of_tree_over_leaf(
        flat_tree,  # tree whose leaves are flat tuples (tk, tt, tucker_mask, tt_mask) of arrays
) -> int:           # number of stacking levels (tree nesting depth above the leaf tuple)
    """Number of stack levels in a tree of 4-array leaves = total nesting depth minus the 1 leaf level."""
    return stacking.tree_depth(flat_tree) - 1
