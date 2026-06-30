# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Stateless tangent-stack reshuffles for the uniform tangent layer (UT3Tangent), uniform-fix 3b-1b.

The uniform mirror of the ragged ``tangent_operations`` stack/unstack helpers, for the ``.data`` layout
``(*supercores, shape, masks)`` with a leading mode index ``d`` and the stack at axes ``1 ..``. A uniform
tangent's variations carry the full ``K + C`` stack (tangent stack ``K`` outermost, base stack ``C``
inner); the basis carries only ``C``. These functions split that stack into a Python tree of per-element
objects and recombine -- a tree<->array conversion, NOT an axis permutation of the supercores. They are
the backend the UT3Tangent ``unstack_tangents`` / ``unstack_basis`` / ``stack_tangents`` / ``stack_basis``
/ ``sum_tangents`` methods delegate to.

Varying ranks across the base ``C`` stack are first-class here (the rank-sweep use case): the per-element
masks may differ, and ``stack_base_stack`` stacks frames of different ranks into one batch (the masks just
ride along). Uniform rank is required only across ``K`` (one shared base = one tangent space), which
``stack_tangent_stack`` gets for free (the leaves share a frame). See
``docs/uniform_ranks_and_varieties.md``.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ubv_operations as ubv_operations
from t3toolbox.backend.common import *

__all__ = [
    'unstack_tangent_stack',
    'stack_tangent_stack',
    'unstack_base_stack',
    'stack_base_stack',
    'sum_tangent_stack',
]


def _tangent_stack_split(
        basis_data,       # (up, down, left, right, shape, masks), each supercore stack = C
        variations_data,  # (tkv, ttv, shape, masks),             each supercore stack = K + C
) -> typ.Tuple[int, int]:  # (|K|, |C|)
    """Recover the tangent / base stack split (|K|, |C|) from a (basis, variations) ``.data`` pair.

    The frame supercores carry only the base stack ``C`` (up: ``(d,)+C+(nU, N)``); the variation supercores
    carry the full ``K + C`` (tucker variation: ``(d,)+K+C+(nD, N)``). So ``|C|`` comes from a frame core
    and ``|K|`` is the remainder.
    """
    n_base = basis_data[0].ndim - 3       # |C|   (up supercore: (d,) + C + (nU, N))
    n_full = variations_data[0].ndim - 3  # |K+C| (tucker variation: (d,) + K + C + (nD, N))
    return n_full - n_base, n_base


def _pair_leaves(basis_tree, variations_tree, n_base):  # mirror tangent_operations._pair_base_leaves
    """Pair a basis-data tree and a variations-data tree (same ``C``-shaped outer structure, ``n_base``
    levels deep) leaf-by-leaf into one tree of ``(basis_data, variations_data)`` pairs. NOT a
    :py:func:`stacking.tree_zip`: the data-tuple leaves are themselves sequences (and carry the int-tuple
    ``shape``), so a generic zip would recurse into them -- we stop at the known base depth ``n_base``."""
    if n_base == 0:
        return (basis_tree, variations_tree)  # both are single .data tuples -> one pair
    return tuple(_pair_leaves(b, v, n_base - 1) for b, v in zip(basis_tree, variations_tree))


def _unpair_leaves(paired_tree, n_base):  # inverse of _pair_leaves
    """Split a ``C``-shaped tree of ``(basis_data, variations_data)`` pairs back into
    ``(basis_tree, variations_tree)``."""
    if n_base == 0:
        return paired_tree  # already a single (basis_data, variations_data) pair
    split = [_unpair_leaves(p, n_base - 1) for p in paired_tree]
    return tuple(s[0] for s in split), tuple(s[1] for s in split)


def _depth_to_pair(paired_tree) -> int:  # |C|: nesting levels above a (basis_data, variations_data) leaf
    """Count the nesting depth of a ``C``-shaped tree of ``(basis_data, variations_data)`` pairs without
    recursing into the data tuples (whose int-tuple ``shape`` would fool :py:func:`stacking.tree_depth`).
    A leaf pair is reached when ``node[0][0]`` -- the first frame supercore of ``basis_data`` -- is an
    ndarray; until then ``node[0]`` is a subtree."""
    depth, node = 0, paired_tree
    while not is_ndarray(node[0][0]):
        node, depth = node[0], depth + 1
    return depth


def unstack_tangent_stack(
        basis_data,       # frame .data,      supercore stack = C
        variations_data,  # variations .data, supercore stack = K + C
):  # -> array-like tree (shape K) of variations .data tuples (each stack = C)
    """Peel the tangent stack ``K`` off the variations, returning a ``K``-shaped tree of variation-``.data``.

    The base is shared across ``K``, so the frame is untouched (the caller pairs the same base with every
    leaf). Inverse of :py:func:`stack_tangent_stack`."""
    n_tangent, _ = _tangent_stack_split(basis_data, variations_data)
    return ubv_operations.ubv_unstack_axes(variations_data, 2, range(1, 1 + n_tangent))


def stack_tangent_stack(
        variations_tree,  # array-like tree (shape K) of variations .data tuples (each stack = C)
):  # -> variations .data tuple (stack = K + C)
    """Stack a ``K``-shaped tree of variation-``.data`` over the tangent stack ``K`` (outermost; ``C`` stays
    inner). Inverse of :py:func:`unstack_tangent_stack`. The leaves share one frame, so ranks are uniform
    across ``K`` (the masks, constant along ``K``, just replicate)."""
    return ubv_operations.ubv_stack(variations_tree, 2)


def unstack_base_stack(
        basis_data,       # frame .data,      supercore stack = C
        variations_data,  # variations .data, supercore stack = K + C
):  # -> array-like tree (shape C) of (basis_data, variations_data) pairs
    """Peel the base stack ``C`` off both the frame and the variations, returning a ``C``-shaped tree whose
    leaves are ``(basis_data, variations_data)`` pairs -- one single-base-point tangent per leaf.

    Each frame-``.data`` leaf has stack ``()`` (a single base); each variations-``.data`` leaf has stack
    ``K``. The base stack is the *inner* part of the ``K + C`` variation stack, so it is peeled from the
    interior axes ``1+|K| .. 1+|K|+|C|`` of the variation supercores; the frame's whole stack is ``C``. The
    leaves are paired for you (a plain :py:func:`stacking.tree_zip` would recurse into the data tuples).
    Inverse of :py:func:`stack_base_stack`."""
    n_tangent, n_base = _tangent_stack_split(basis_data, variations_data)
    basis_tree = ubv_operations.ubv_unstack(basis_data, 4)  # frame stack is all C
    variations_tree = ubv_operations.ubv_unstack_axes(variations_data, 2,
                                                      range(1 + n_tangent, 1 + n_tangent + n_base))
    return _pair_leaves(basis_tree, variations_tree, n_base)


def stack_base_stack(
        paired_tree,  # array-like tree (shape C) of (basis_data, variations_data) pairs
):  # -> (basis_data [stack C], variations_data [stack K + C])
    """Stack a ``C``-shaped tree of ``(basis_data, variations_data)`` pairs over the base stack ``C``.

    The base stack is placed *innermost* (the variation stack becomes ``K + C``), matching the base-inner
    convention. Frames of DIFFERENT ranks stack into one batch (varying-``C`` -- the per-element masks just
    ride along); the shared requirement is only matching padded dims and tangent stack ``K``. Takes exactly
    the layout :py:func:`unstack_base_stack` produces (its inverse)."""
    n_base = _depth_to_pair(paired_tree)
    basis_tree, variations_tree = _unpair_leaves(paired_tree, n_base)
    basis_data = ubv_operations.ubv_stack(basis_tree, 4)                 # C at axes 1.. (frame stack = C)
    n_tangent = ubv_operations._first_data_leaf(variations_tree)[0].ndim - 3  # |K| of a variations leaf
    variations_data = ubv_operations.ubv_stack_axes(variations_tree, 2, axes_start=1 + n_tangent)  # C after K
    return basis_data, variations_data


def sum_tangent_stack(
        variations_data,         # variations .data, supercore stack = K + C
        n_tangent:   int,        # |K|
        axis:        typ.Optional[int] = None,  # 0-based index WITHIN K (None = the whole tangent stack)
):  # -> variations .data with the summed K axes removed (stack = C, or K-with-one-axis-removed)
    """Sum the variations over the tangent stack ``K`` (a batch of tangents at one base -> their sum;
    corewise == the tangent sum, by linearity). The base stack ``C`` is preserved.

    The supercores sum via ``xnp``; the masks **OR** over the same axes (host ``np``). Because a ``K`` stack
    shares one base, its masks are constant along ``K``, so the OR is a no-op (the summed tangent carries
    the base's gauge masks) -- but it is the correct reduction in general. ``axis`` indexes within ``K``."""
    tkv, ttv, shape, masks = variations_data
    xnp, _, _ = get_backend(True, tree_contains_jax((tkv, ttv)))

    k_axes = tuple(range(1, 1 + n_tangent)) if axis is None else (1 + axis,)
    new_tkv = xnp.sum(tkv, axis=k_axes)
    new_ttv = xnp.sum(ttv, axis=k_axes)
    new_masks = tuple(np.any(m, axis=k_axes) for m in masks)   # host np: OR the real slots over K
    return new_tkv, new_ttv, shape, new_masks
