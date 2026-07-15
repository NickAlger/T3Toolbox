# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Structural operations on uniform supercore data.

``ut3_squash_tails``/``ut3_reverse``, stack/unstack + leaf structure, and the packing seam
(``pack_vectors``/``unpack_vectors``/``is_packed``/``pack_if_ragged``) behind the
packedness-mirror convention (user-facing ops mirror the input's packedness).
"""
import numpy as np
import typing as typ

import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ut3_masking as ut3_masking
from t3toolbox.backend.common import *
from t3toolbox.backend.tt_operations import tt_reverse, tt_squash_tails

__all__ = [
    'ut3_squash_tails',
    'ut3_reverse',
    'pack_vectors',
    'unpack_vectors',
    'is_packed',
    'pack_if_ragged',
    'ut3_unstack',
    'ut3_stack',
    'ut3_leaf_structure',
]

# A uniform-T3 .data tuple: (tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask)).
# `shape` is a static int tuple (N0,...,N(d-1)); the two rank masks are HOST bool, static structure
# (numpy, never traced); the supercores are xnp data.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]


def ut3_leaf_structure(d: int):  # leaf-structure template for stacking.apply_func_to_leaf_subtrees
    """Template marking one uniform-T3 ``.data`` leaf for the tree machinery in ``stacking.py``.

    The leaf is ``(tucker_supercore, tt_supercore, shape, (tucker_mask, tt_mask))``. The ``shape``
    int tuple has ``d`` int leaves, so the template must encode its length: a bare ``None`` there
    fails to match (an int tuple is a ``Sequence``, unlike the ndarray leaves, so the walker would
    recurse into it)."""
    return (None, None, (None,) * d, (None, None))


def _first_data_leaf(xx):  # drill to the first .data leaf without recursing into the int-tuple `shape`
    # A .data leaf has an ndarray (tucker_supercore) at [0]; a nesting node has a subtree (tuple) there.
    # (get_first_leaf can't be used here: it would drill into `shape`, which is itself a Sequence.)
    while not is_ndarray(xx[0]):
        xx = xx[0]
    return xx


def ut3_squash_tails(data: UT3Data) -> UT3Data:
    """Sum the leading/trailing TT bonds down to rank 1 (preserves the tensor), updating those edge
    masks to rank 1. Operates on the full .data tuple."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)

    tk, tt, shape, (tkm, ttm) = data
    require_concrete_masks(tkm, ttm)  # masks are host, not traced
    new_tt = tt_squash_tails(tt)
    r = tt.shape[-1]
    stack = tt.shape[1:-3]
    # np (host): the rank-1 boundary masks are static structure, not supercore data. Intentional.
    rank1 = np.broadcast_to(prefix_mask(1, r), stack + (r,))                   # [True, False, ...]
    new_ttm = np.concatenate([rank1[None], ttm[1:-1], rank1[None]], axis=0)
    return tk, new_tt, shape, (tkm, new_ttm)


def ut3_reverse(data: UT3Data) -> UT3Data:
    """Reverse the mode order (supercores, shape, and masks). Operates on the full .data tuple."""
    tk, tt, shape, (tkm, ttm) = data
    return tk[::-1], tt_reverse(tt), shape[::-1], (tkm[::-1], ttm[::-1])


def pack_vectors(
        unpacked_vectors: typ.Sequence[NDArray],  # len=d, ith elm_shape=stack_shape+(Ni,)
        N: int = None,                            # padded length (default max(Ni))
) -> NDArray:                                     # packed, shape=(d,)+stack_shape+(N,)
    """Zero-pad and stack a sequence of (ragged-length) vectors into one supercore-shaped tensor.

    The pad fill is zeros, and must stay FINITE: masking works by multiplication, and
    ``0 * nan = nan`` -- a ``nan``/``inf`` fill would poison masked reductions downstream
    (``docs/uniform_equivalence_contract.md``). Shape information always travels alongside the
    packed array; the fill is never used to infer shape.
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


def is_packed(
        vectors,  # a packed supercore array (single ndarray) OR a ragged len=d sequence of per-mode vectors
) -> bool:        # True if already packed (one array), False if a ragged list/tuple of d per-mode arrays
    """Whether mode ``vectors`` are packed (a single supercore-shaped ndarray) or ragged (a ``len=d``
    sequence of per-mode arrays of differing widths). The uniform sampling ops infer packedness from this
    and **mirror** it -- packed in ``->`` packed out, ragged in ``->`` ragged out."""
    return not isinstance(vectors, (list, tuple))


def pack_if_ragged(
        vectors,        # packed array (returned as-is) OR ragged len=d sequence, ith elm_shape=stack+(Ni,)
        N:  int = None,  # padded length (default max Ni); ignored when ``vectors`` is already packed
) -> NDArray:            # packed, shape=(d,)+stack_shape+(N,)
    """Pack ``vectors`` iff ragged (a ``len=d`` sequence); an already-packed array is returned unchanged.
    The input side of the sampling-op packedness mirror (:py:func:`is_packed`)."""
    return vectors if is_packed(vectors) else pack_vectors(vectors, N)


def ut3_unstack(
        x: typ.Tuple[
            NDArray,                          # tucker_supercore
            NDArray,                          # tt_supercore
            typ.Tuple[int, ...],              # shape
            typ.Tuple[NDArray, NDArray],      # (tucker_edge_mask, tt_edge_mask)
        ],
):  # -> nested tuple (shaped like stack_shape) of unstacked uniform-T3 .data leaves
    """Unstack a uniform Tucker tensor train into an array-like tree of unstacked ones.

    The stack lives at axes ``1 .. len(stack_shape)`` (axis 0 is the mode index ``d``). The supercores
    and the rank masks unstack along it; ``shape`` is shared and replicated onto every leaf (the
    ndarray-only ``(tk, tt, tkm, ttm)`` go through the tree machinery; ``shape`` is woven in after).
    """
    tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask) = x
    stack_shape = tucker_supercore.shape[1:-2]
    axes = tuple(range(1, 1 + len(stack_shape)))

    tree = stacking.unstack((tucker_supercore, tt_supercore, tucker_edge_mask, tt_edge_mask), axes=axes)

    return stacking.apply_func_to_leaf_subtrees(
        tree,
        lambda leaf: (leaf[0], leaf[1], shape, (leaf[2], leaf[3])),
        (None, None, None, None),
    )


def ut3_stack(
        xx,  # nested tuple (shaped like stack_shape) of unstacked uniform-T3 .data leaves
) -> typ.Tuple[
    NDArray,                          # tucker_supercore
    NDArray,                          # tt_supercore
    typ.Tuple[int, ...],              # shape
    typ.Tuple[NDArray, NDArray],      # (tucker_edge_mask, tt_edge_mask)
]:
    """Stack an array-like tree of uniform Tucker tensor trains into one.

    Inverse of :py:func:`ut3_unstack`: stacks the supercores and rank masks onto axes
    ``1 .. num_levels`` (after the mode index), keeping the shared ``shape`` unstacked. Only the four
    ndarray components go through ``stacking.stack``; ``shape`` (a ``Sequence`` the walker would recurse
    into) is read once from the first leaf and re-attached.
    """
    first = _first_data_leaf(xx)        # shape is shared across the stack -> read once (manual drill)
    shape = first[2]
    d = first[0].shape[0]

    # Stack the supercores and the masks via SEPARATE stacking.stack calls. stacking.stack infers ONE
    # backend per call (tree_contains_jax over the whole tree), so a mixed (jax supercore + host mask) call
    # would promote the masks to jax -- breaking the masks-are-host-numpy invariant. The mask-only call has
    # no jax inputs, so the masks stay host numpy; the supercores follow xnp as usual.
    sc_tree   = stacking.apply_func_to_leaf_subtrees(
        xx, lambda leaf: (leaf[0], leaf[1]), ut3_leaf_structure(d))
    mask_tree = stacking.apply_func_to_leaf_subtrees(
        xx, lambda leaf: (leaf[3][0], leaf[3][1]), ut3_leaf_structure(d))

    num_levels = tree_depth_of_tree_over_leaf(sc_tree)
    axes = tuple(range(1, 1 + num_levels))

    tucker_supercore, tt_supercore = stacking.stack(sc_tree, axes)
    tucker_edge_mask, tt_edge_mask = stacking.stack(mask_tree, axes)
    return tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask)


def tree_depth_of_tree_over_leaf(
        flat_tree,  # tree whose leaves are flat tuples (tk, tt, tucker_mask, tt_mask) of arrays
) -> int:           # number of stacking levels (tree nesting depth above the leaf tuple)
    """Number of stack levels in a tree of 4-array leaves = total nesting depth minus the 1 leaf level."""
    return stacking.tree_depth(flat_tree) - 1
