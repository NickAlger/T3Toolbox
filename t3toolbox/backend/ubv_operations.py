# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Stateless stack/unstack for the uniform basis-variations layer (UT3Basis / UT3Variations).

Mirror of ``ut3_operations.ut3_unstack`` / ``ut3_stack`` for the bv ``.data`` layout
``(*supercores, shape, masks)``, generic over the supercore count (4 for a frame, 2 for variations;
both carry the SAME 4-tuple of rank masks). The stack lives at axes ``1 .. len(stack_shape)`` (axis 0 is
the mode index ``d``, NOT a stack axis); the supercores and all four masks unstack along it, while the
int-tuple ``shape`` is shared and woven onto every leaf. The masks' differing leading axis (``d`` vs
``d+1``) is irrelevant -- only axes ``1..k`` are sliced.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import *

__all__ = [
    'ubv_leaf_structure',
    'ubv_unstack',
    'ubv_stack',
    'ubv_variations_sum_stack',
    'ubv_reverse_basis',
    'ubv_reverse_variations',
]

N_MASKS = 4  # both UT3Basis and UT3Variations hold four edge masks


def ubv_leaf_structure(
        d:            int,  # number of modes (length of the int-tuple `shape`)
        n_supercores: int,  # 4 for a UT3Basis .data leaf, 2 for a UT3Variations one
):  # -> leaf-structure template for stacking.apply_func_to_leaf_subtrees
    """Template marking one bv ``.data`` leaf ``(*supercores, shape, masks)`` for the tree machinery.

    The int-tuple ``shape`` has ``d`` int leaves, so the template must spell its length out: a bare
    ``None`` there fails to match (an int tuple is a ``Sequence``, unlike the ndarray leaves, so the
    walker would recurse into it). Mirrors ``ut3_operations.ut3_leaf_structure``.
    """
    return (None,) * n_supercores + ((None,) * d,) + ((None,) * N_MASKS,)


def _first_data_leaf(xx):  # drill to the first .data leaf without recursing into the int-tuple `shape`
    # A .data leaf has an ndarray (the first supercore) at [0]; a nesting node has a subtree (tuple) there.
    # (get_first_leaf can't be used: it would drill into `shape`, itself a Sequence.)
    while not is_ndarray(xx[0]):
        xx = xx[0]
    return xx


def ubv_unstack(
        data:         typ.Tuple,  # (*supercores, shape, masks): n_supercores arrays, int-tuple, 4-tuple of masks
        n_supercores: int,        # 4 (frame) or 2 (variations)
):  # -> nested tuple (shaped like stack_shape) of unstacked bv .data leaves
    """Unstack a (stacked) bv ``.data`` tuple into an array-like tree of unstacked ones.

    The supercores and the four rank masks unstack along the stack axes ``1 .. len(stack_shape)``;
    ``shape`` is shared and replicated onto every leaf. Mirrors :py:func:`ut3_operations.ut3_unstack`.
    """
    supercores = tuple(data[:n_supercores])
    shape      = data[n_supercores]
    masks      = tuple(data[n_supercores + 1])

    stack_shape = supercores[0].shape[1:-2]   # first supercore is the tucker one: (d,)+stack+(n, N)
    axes = tuple(range(1, 1 + len(stack_shape)))

    tree = stacking.unstack(supercores + masks, axes=axes)

    return stacking.apply_func_to_leaf_subtrees(
        tree,
        lambda leaf: tuple(leaf[:n_supercores]) + (shape, tuple(leaf[n_supercores:])),
        (None,) * (n_supercores + len(masks)),
    )


def ubv_stack(
        xx,                       # nested tuple (shaped like stack_shape) of unstacked bv .data leaves
        n_supercores: int,        # 4 (frame) or 2 (variations)
) -> typ.Tuple:                   # one stacked bv .data tuple (*supercores, shape, masks)
    """Stack an array-like tree of bv ``.data`` leaves into one. Inverse of :py:func:`ubv_unstack`.

    Stacks the supercores and rank masks onto axes ``1 .. num_levels`` (after the mode index), keeping the
    shared ``shape`` unstacked (read once from the first leaf -- a ``Sequence`` the walker would recurse
    into). Mirrors :py:func:`ut3_operations.ut3_stack`.
    """
    first = _first_data_leaf(xx)        # shape is shared across the stack -> read once (manual drill)
    shape = first[n_supercores]
    d     = first[0].shape[0]
    template = ubv_leaf_structure(d, n_supercores)

    # Stack the supercores and the masks via SEPARATE stacking.stack calls. stacking.stack infers ONE
    # backend per call (tree_contains_jax over the whole tree), so a mixed (jax supercore + host mask) call
    # would promote the masks to jax -- breaking the masks-are-host-numpy invariant. The mask-only call has
    # no jax inputs, so the masks stay host numpy; the supercores follow xnp as usual.
    sc_tree   = stacking.apply_func_to_leaf_subtrees(xx, lambda leaf: tuple(leaf[:n_supercores]), template)
    mask_tree = stacking.apply_func_to_leaf_subtrees(xx, lambda leaf: tuple(leaf[n_supercores + 1]), template)

    num_levels = stacking.tree_depth(sc_tree) - 1   # minus the 1 flat-tuple leaf level
    axes = tuple(range(1, 1 + num_levels))

    supercores = stacking.stack(sc_tree, axes)      # supercores: xnp-inferred (jax-aware)
    masks      = stacking.stack(mask_tree, axes)    # masks: all host -> stay host numpy
    return tuple(supercores) + (shape, tuple(masks))


def ubv_variations_sum_stack(
        data,             # UT3Variations .data: (tkv, ttv, shape, (4 masks))
        axis: typ.Optional[int] = None,  # stack axis to sum (None = whole stack); 0-based within the stack
):  # -> summed .data (the summed stack axes removed)
    """Sum a UT3Variations over stack axes (a batch of tangents -> their sum; corewise == tangent sum by
    linearity). The supercores sum via ``xnp``; the masks **OR** over the same axes (host ``np``) -- the
    union of real slots, a no-op for a same-mask stack (``docs/uniform_masks_vs_ranks.md``). The stack
    lives at axes ``1 ..`` (after the leading mode index ``d``), shared by supercores and masks."""
    tkv, ttv, shape, masks = data
    n_stack = tkv.ndim - 3   # (d,)+stack+(nD, N)
    use_jax = tree_contains_jax((tkv, ttv))
    xnp, _, _ = get_backend(True, use_jax)

    stack_axes = tuple(range(1, 1 + n_stack)) if axis is None else (1 + axis,)
    new_tkv = xnp.sum(tkv, axis=stack_axes)
    new_ttv = xnp.sum(ttv, axis=stack_axes)
    new_masks = tuple(np.any(m, axis=stack_axes) for m in masks)   # host np: OR the real slots over the stack
    return new_tkv, new_ttv, shape, new_masks


def ubv_reverse_basis(data):  # UT3Basis .data -> reversed UT3Basis .data
    """Reverse the mode order of a UT3Basis ``.data``. The left/right supercores **and** their masks
    **swap roles** (reversing a left-orthogonal chain yields a right-orthogonal one) and reverse; up/down
    reverse (down with a bond swap, via :py:func:`ut3_operations.reverse_utt`). The redundant L/R store
    makes this exact -- no re-orthogonalization. Inverse of itself."""
    up_sc, down_sc, left_sc, right_sc, shape, (um, dm, lm, rm) = data
    rev = ut3_operations.reverse_utt
    return (
        up_sc[::-1],
        rev(down_sc),
        rev(right_sc),                              # old right -> new left
        rev(left_sc),                               # old left  -> new right
        tuple(shape[::-1]),
        (um[::-1], dm[::-1], rm[::-1], lm[::-1]),    # up/down reverse; left/right masks swap + reverse
    )


def ubv_reverse_variations(data):  # UT3Variations .data -> reversed UT3Variations .data
    """Reverse the mode order of a UT3Variations ``.data``: the tucker-variation supercore reverses; the
    tt-variation supercore reverses with a bond swap (:py:func:`ut3_operations.reverse_utt`). The per-slot
    left/right masks swap + reverse (a variation occupies one TT slot). Inverse of itself."""
    tkv, ttv, shape, (vup, vdown, vleft, vright) = data
    return (
        tkv[::-1],
        ut3_operations.reverse_utt(ttv),
        tuple(shape[::-1]),
        (vup[::-1], vdown[::-1], vright[::-1], vleft[::-1]),   # up/down reverse; left/right swap + reverse
    )
