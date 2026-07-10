# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Stateless stack/unstack for the uniform frame-variations layer (UT3Frame / UT3Variations).

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
import t3toolbox.backend.ufv_masking as ufv_masking
from t3toolbox.backend.common import *

__all__ = [
    'ufv_leaf_structure',
    'ufv_unstack',
    'ufv_stack',
    'ufv_unstack_axes',
    'ufv_stack_axes',
    'ufv_variations_sum_stack',
    'ufv_reverse_frame',
    'ufv_reverse_variations',
    'ufv_save',
    'ufv_load',
    'ufv_frame_orthogonality_residual',
]

N_MASKS = 4  # both UT3Frame and UT3Variations hold four edge masks


def ufv_leaf_structure(
        d:            int,  # number of modes (length of the int-tuple `shape`)
        n_supercores: int,  # 4 for a UT3Frame .data leaf, 2 for a UT3Variations one
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


def ufv_unstack_axes(
        data:         typ.Tuple,        # (*supercores, shape, masks): n_supercores arrays, int-tuple, 4-tuple of masks
        n_supercores: int,              # 4 (frame) or 2 (variations)
        axes:         typ.Sequence[int],  # the array axes to peel into tree levels (a CONTIGUOUS run within the stack)
):  # -> nested tuple (shaped like the peeled axes) of bv .data leaves carrying the remaining stack
    """Unstack a bv ``.data`` tuple along the GIVEN array axes (a sub-run of the stack), leaving the rest.

    Generalizes :py:func:`ufv_unstack` (which peels the whole stack) to a contiguous sub-run -- the
    primitive the tangent layer needs to split a ``K + C`` variation stack into just its ``K`` part or just
    its ``C`` part. The supercores and all four rank masks slice along ``axes`` (they share those stack-axis
    positions; the masks' leading ``d`` vs ``d+1`` is irrelevant since only ``axes`` are touched); ``shape``
    is shared and replicated onto every leaf.
    """
    supercores = tuple(data[:n_supercores])
    shape      = data[n_supercores]
    masks      = tuple(data[n_supercores + 1])

    tree = stacking.unstack(supercores + masks, axes=tuple(axes))

    return stacking.apply_func_to_leaf_subtrees(
        tree,
        lambda leaf: tuple(leaf[:n_supercores]) + (shape, tuple(leaf[n_supercores:])),
        (None,) * (n_supercores + len(masks)),
    )


def ufv_stack_axes(
        xx,                       # nested tuple (shaped like the stacked sub-run) of bv .data leaves
        n_supercores: int,        # 4 (frame) or 2 (variations)
        axes_start:   int,        # array axis the OUTERMOST new stack level lands on (others follow contiguously)
) -> typ.Tuple:                   # one stacked bv .data tuple (*supercores, shape, masks)
    """Stack an array-like tree of bv ``.data`` leaves onto a CONTIGUOUS run of array axes starting at
    ``axes_start``. Inverse of :py:func:`ufv_unstack_axes`.

    Generalizes :py:func:`ufv_stack` (which stacks onto axes ``1 ..``) so the tangent layer can slot a new
    stack run at the right place -- e.g. the base stack ``C`` *after* an existing tangent stack ``K`` (at
    ``axes_start = 1 + |K|``), keeping ``C`` inner. ``shape`` is read once (shared) and supercores/masks are
    stacked in SEPARATE :py:func:`stacking.stack` calls so the host-numpy masks are not promoted to jax.
    """
    first = _first_data_leaf(xx)        # shape is shared across the stack -> read once (manual drill)
    shape = first[n_supercores]
    d     = first[0].shape[0]
    template = ufv_leaf_structure(d, n_supercores)

    sc_tree   = stacking.apply_func_to_leaf_subtrees(xx, lambda leaf: tuple(leaf[:n_supercores]), template)
    mask_tree = stacking.apply_func_to_leaf_subtrees(xx, lambda leaf: tuple(leaf[n_supercores + 1]), template)

    num_levels = stacking.tree_depth(sc_tree) - 1   # minus the 1 flat-tuple leaf level
    axes = tuple(range(axes_start, axes_start + num_levels))

    supercores = stacking.stack(sc_tree, axes)      # supercores: xnp-inferred (jax-aware)
    masks      = stacking.stack(mask_tree, axes)    # masks: all host -> stay host numpy
    return tuple(supercores) + (shape, tuple(masks))


def ufv_unstack(
        data:         typ.Tuple,  # (*supercores, shape, masks): n_supercores arrays, int-tuple, 4-tuple of masks
        n_supercores: int,        # 4 (frame) or 2 (variations)
):  # -> nested tuple (shaped like stack_shape) of unstacked bv .data leaves
    """Unstack a (stacked) bv ``.data`` tuple into an array-like tree of unstacked ones (the WHOLE stack).

    The supercores and the four rank masks unstack along the stack axes ``1 .. len(stack_shape)``;
    ``shape`` is shared and replicated onto every leaf. Mirrors :py:func:`ut3_operations.ut3_unstack`;
    thin wrapper over :py:func:`ufv_unstack_axes` over the full stack.
    """
    stack_shape = data[0].shape[1:-2]   # first supercore is the tucker one: (d,)+stack+(n, N)
    return ufv_unstack_axes(data, n_supercores, tuple(range(1, 1 + len(stack_shape))))


def ufv_stack(
        xx,                       # nested tuple (shaped like stack_shape) of unstacked bv .data leaves
        n_supercores: int,        # 4 (frame) or 2 (variations)
) -> typ.Tuple:                   # one stacked bv .data tuple (*supercores, shape, masks)
    """Stack an array-like tree of bv ``.data`` leaves into one (the WHOLE stack at axes ``1 ..``). Inverse
    of :py:func:`ufv_unstack`; thin wrapper over :py:func:`ufv_stack_axes` with ``axes_start = 1``."""
    return ufv_stack_axes(xx, n_supercores, axes_start=1)


def ufv_variations_sum_stack(
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


def ufv_reverse_frame(data):  # UT3Frame .data -> reversed UT3Frame .data
    """Reverse the mode order of a UT3Frame ``.data``. The left/right supercores **and** their masks
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


def ufv_reverse_variations(data):  # UT3Variations .data -> reversed UT3Variations .data
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


def ufv_save(file, data) -> None:  # data: (*supercores, shape, masks) for a UT3Frame or UT3Variations
    """Save a bv ``.data`` tuple to a ``.npz`` (3 families: the supercores, the rank masks, the ``shape``
    ints). Generic over the supercore count; mirrors :py:func:`ut3_constructors.ut3_save`. ``np.savez``
    keeps the boolean mask dtype, so :py:func:`ufv_load` recovers host bool masks."""
    *supercores, shape, masks = data
    save_core_families(file, (tuple(supercores), tuple(masks), (np.asarray(shape, dtype=int),)))


def ufv_load(file, use_jax: bool = False):  # -> (*supercores, shape, masks)
    """Load a bv ``.data`` tuple saved by :py:func:`ufv_save`. The supercores follow ``use_jax``; the
    masks always come back **numpy (host) bool** (a jax mask is a tracer under jit). The caller wraps the
    returned tuple into the OO class (the supercore count is fixed per class)."""
    supercores, masks, shape_family = load_core_families(file)
    if use_jax:
        supercores = tuple(to_jax(s) for s in supercores)
    masks = tuple(np.asarray(m, dtype=bool) for m in masks)
    shape = tuple(int(x) for x in shape_family[0])
    return tuple(supercores) + (shape, masks)


def ufv_frame_orthogonality_residual(data):  # UT3Frame .data -> max orthogonality deviation, per stack element
    """Max deviation of the four masked frame supercores from orthonormality, **per stack element** (shape
    ``stack_shape``). Each masked supercore slice IS a hypothetical ragged core (mask the padding to zero);
    require its Gram == ``diag(outgoing_mask)`` -- the masked-Gram pattern, over the four senses (up ``U``,
    down/outer ``O``, left ``L``, right ``R``), with the correct outgoing mask per sense (left core ``i`` ->
    ``frame_left_mask[i+1]``). The uniform analog of
    :py:func:`orthogonal_representations.frame_orthogonality_residual`; the per-element oracle is the ragged
    one via ``to_t3frame``. The boundary left/right cores are remainders and are not checked (so left
    checks cores ``0..d-2``, right checks ``1..d-1``)."""
    up_sc, down_sc, left_sc, right_sc, shape, (um, dm, lm, rm) = data
    mup, mdown, mleft, mright = ufv_masking.apply_frame_masks(data)   # zero the padding (mask-once)
    use_jax = tree_contains_jax((up_sc, down_sc, left_sc, right_sc))
    xnp, _, _ = get_backend(True, use_jax)
    d = up_sc.shape[0]
    nU, nD, rL, rR = mup.shape[-2], mdown.shape[-2], mleft.shape[-1], mright.shape[-1]

    def dev(G, mask, n):  # max over the leading core axis + the two gram axes -> keep stack
        return xnp.max(xnp.abs(G - xnp.eye(n) * mask[..., None, :]), axis=(0, -2, -1))

    devs = [
        dev(xnp.einsum('...io,...jo->...ij', mup, mup), um, nU),          # up: rows orthonormal over mode
        dev(xnp.einsum('...iaj,...ibj->...ab', mdown, mdown), dm, nD),    # down/outer: mode orthonormal over bonds
    ]
    if d > 1:   # interior left/right cores; the boundary core of each is the unchecked remainder
        devs.append(dev(xnp.einsum('...iaj,...iak->...jk', mleft[:-1], mleft[:-1]), lm[1:-1], rL))   # outgoing edges 1..d-1
        devs.append(dev(xnp.einsum('...iaj,...kaj->...ik', mright[1:], mright[1:]), rm[1:-1], rR))   # incoming edges 1..d-1
    return xnp.max(xnp.stack(devs), axis=0)   # max over the four senses, keep stack_shape
