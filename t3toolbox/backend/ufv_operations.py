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

import t3toolbox.backend.tt_operations as tt_operations
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
    'ufv_frame_reverse',
    'ufv_variations_reverse',
    'ufv_save',
    'ufv_load',
    'ufv_frame_orthogonality_residual',
    'ufv_absorb_weights',
    'ufv_weights_consistent',
    'ufv_weights_from_ut3_weights',
    'ufv_reciprocal_weights',
    'ufv_sqrt_weights',
    'ufv_concatenate_weights',
    'ufv_kronecker_weights',
]

# A uniform FRAME-WEIGHTS .data tuple: (up, down, left, right, (4 variation edge masks)).
# The four supercores are each (d,)+C+(size,) and the masks match them. NO `shape` (weights have no
# physical legs) and NO K: a frame weight is FRAME-LIKE -- one metric per base point, carrying the frame
# stack C, broadcast over the variations' K+C at absorb (docs/contributor/weighted_internals.md).
UT3FrameWeightsData = typ.Tuple[NDArray, NDArray, NDArray, NDArray,
                                typ.Tuple[NDArray, NDArray, NDArray, NDArray]]

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
    stack run at the right place -- e.g. the frame stack ``C`` *after* an existing tangent stack ``K`` (at
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


def ufv_frame_reverse(data):  # UT3Frame .data -> reversed UT3Frame .data
    """Reverse the mode order of a UT3Frame ``.data``. The left/right supercores **and** their masks
    **swap roles** (reversing a left-orthogonal chain yields a right-orthogonal one) and reverse; up/down
    reverse (down with a bond swap, via :py:func:`tt_operations.tt_reverse`). The redundant L/R store
    makes this exact -- no re-orthogonalization. Inverse of itself."""
    up_sc, down_sc, left_sc, right_sc, shape, (um, dm, lm, rm) = data
    rev = tt_operations.tt_reverse
    return (
        up_sc[::-1],
        rev(down_sc),
        rev(right_sc),                              # old right -> new left
        rev(left_sc),                               # old left  -> new right
        tuple(shape[::-1]),
        (um[::-1], dm[::-1], rm[::-1], lm[::-1]),    # up/down reverse; left/right masks swap + reverse
    )


def ufv_variations_reverse(data):  # UT3Variations .data -> reversed UT3Variations .data
    """Reverse the mode order of a UT3Variations ``.data``: the tucker-variation supercore reverses; the
    tt-variation supercore reverses with a bond swap (:py:func:`tt_operations.tt_reverse`). The per-slot
    left/right masks swap + reverse (a variation occupies one TT slot). Inverse of itself."""
    tkv, ttv, shape, (vup, vdown, vleft, vright) = data
    return (
        tkv[::-1],
        tt_operations.tt_reverse(ttv),
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
    :py:func:`fv_operations.fv_frame_orthogonality_residual`; the per-element oracle is the ragged
    one via ``to_t3frame``. The boundary left/right cores are remainders and are not checked (so left
    checks cores ``0..d-2``, right checks ``1..d-1``)."""
    up_sc, down_sc, left_sc, right_sc, shape, (um, dm, lm, rm) = data
    mup, mdown, mleft, mright = ufv_masking.ufv_apply_frame_masks(data)   # zero the padding (mask-once)
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


def ufv_absorb_weights(
        variations: typ.Tuple,       # UT3Variations .data: (tkv, ttv, shape, masks), stack = K + C
        weights:    UT3FrameWeightsData,  # (up, down, left, right, masks), stack = C
) -> typ.Tuple:                      # UT3Variations .data: weighted, same shape and masks
    """Absorb the four-family metric weights into the VARIATION supercores -- the uniform twin of
    ``fv_absorb_weights`` (the tangent metric on coordinates: ``down`` -> V's ``nD`` leg;
    ``up``/``left``/``right`` -> H's ``nU``/``rL``/``rR`` legs). The frame is left orthonormal and
    untouched, so this is O(ranks) and does not disturb the tangent space.

    **The weight is frame-like (stack ``C``) and the variations are ``K + C``, and the broadcast is
    free**: the ellipsis in ``'d...i,d...io->d...io'`` right-aligns, so a ``C``-stacked weight lifts over
    the tangent stack ``K`` at no cost. That works *only because* ``C`` is innermost -- the library-wide
    frame-inner convention (``docs/batching_and_stacking.md``). One metric, shared by the ``K`` tangent
    vectors at that frame.

    **No entry masking**, for the same reason as ``ut3_absorb_weights``: this is a pointwise scale along
    each edge axis, not a reduction, so garbage never mixes into a real slot (garbage-transparent). The
    reduction that follows (``utv_weighted_norm``/``_inner``) masks its own input.

    **Precondition (structural, NOT enforced here):** the weight's masks, broadcast over ``K``, must equal
    the variations' masks -- ``ufv_weights_consistent``. Uniform padding hides a mismatch that ragged
    would raise as a shape error; the frontend enforces it.
    """
    xnp, _, _ = get_backend(True, tree_contains_jax((variations[:2], weights[:4])))
    tucker_variations, tt_variations, shape, masks = variations
    up, down, left, right, _ = weights

    weighted_tucker = xnp.einsum('d...i,d...io->d...io', down, tucker_variations)
    weighted_tt = xnp.einsum('d...aib,d...a,d...i,d...b->d...aib', tt_variations, left, up, right)
    return weighted_tucker, weighted_tt, shape, masks


def ufv_weights_consistent(
        variations: typ.Tuple,            # UT3Variations .data: (tkv, ttv, shape, masks), stack = K + C
        weights:    UT3FrameWeightsData,  # (up, down, left, right, masks), stack = C
) -> bool:                                # True iff `weights` can be absorbed into `variations`
    """True iff the four weight families fit ``variations`` (non-raising): padded widths match, the weight
    stack is the **trailing part** of the variation stack, and the weight's masks -- broadcast constant
    over the excess ``K`` -- equal the variations' masks.

    Two checks ragged does not need, both because uniform pads:

    - **The trailing-stack rule** is ragged's too (a weight is frame-like: it carries ``C``, the variations
      carry ``K + C``; ``fv_weights_consistent``), but here it must be spelled out against padded shapes.
    - **Mask equality** is uniform-only. Ragged catches a rank mismatch as an einsum shape error; uniform
      pads both to the common ``(nU, nD, rL, rR)``, so a mismatch is invisible to the shapes and would
      silently corrupt -- a weight whose mask calls slot ``i`` padding carries a canonical zero there, so
      absorbing it **zeroes a real variation slot**. The mask comparison mirrors ``check_ufv_pair``'s
      (the variation masks are constant along ``K``).
    """
    tucker_variations, tt_variations, _, variation_masks = variations
    up, down, left, right, weight_masks = weights

    d = tucker_variations.shape[0]
    var_stack = tucker_variations.shape[1:-2]
    weight_stack = up.shape[1:-1]
    n_K = len(var_stack) - len(weight_stack)
    if n_K < 0 or var_stack[n_K:] != weight_stack:   # C must be the trailing part of K + C
        return False

    nD = tucker_variations.shape[-2]
    rL, nU, rR = tt_variations.shape[-3], tt_variations.shape[-2], tt_variations.shape[-1]
    for arr, size in ((up, nU), (down, nD), (left, rL), (right, rR)):
        if tuple(arr.shape) != (d,) + weight_stack + (size,):
            return False

    # np (host): masks are static structure. Reshape each C-mask to insert n_K size-1 axes after the
    # leading mode axis, broadcast up to K+C, and compare -- the check_ufv_pair pattern.
    for a, b in zip(weight_masks, variation_masks):
        if tuple(a.shape) != tuple(b.shape[:1]) + weight_stack + tuple(b.shape[-1:]):
            return False
        a_bcast = np.broadcast_to(a.reshape(a.shape[:1] + (1,) * n_K + a.shape[1:]), b.shape)
        if not np.array_equal(a_bcast, b):
            return False
    return True


def ufv_weights_from_ut3_weights(
        weights: typ.Tuple[NDArray, NDArray, typ.Tuple[NDArray, NDArray]],  # UT3Weights .data
) -> UT3FrameWeightsData:  # (up, down, left, right, masks) -- a tangent metric
    """Build uniform frame weights (a tangent metric) from uniform base-point edge weights -- the twin of
    ``fv_weights_from_t3_weights``, and the same slicing, applied to supercores **and** masks:

    ``up = down = tucker``; ``left = tt[:-1]``, ``right = tt[1:]``. The TT slicing encodes the ``H_i`` bond
    convention (``H_i``'s left bond is TT bond ``i``, its right bond is bond ``i+1``), which turns the
    ``d+1`` bond supercore into the two ``d``-length families -- non-obvious, hence a named function.

    The result pairs with a **minimal-rank** tangent (where the complement rank ``nD`` equals the Tucker
    rank ``nU``, as for ``ut3svd`` output). A non-minimal tangent has ``nD < nU`` and would mismatch the
    ``down`` family at use -- caught by ``ufv_weights_consistent`` rather than silently absorbed.
    """
    tucker_weight_supercore, tt_weight_supercore, (tucker_mask, tt_mask) = weights
    up = down = tucker_weight_supercore
    left, right = tt_weight_supercore[:-1], tt_weight_supercore[1:]
    masks = (tucker_mask, tucker_mask, tt_mask[:-1], tt_mask[1:])   # np: host structure, sliced not rebuilt
    return up, down, left, right, masks


def _ufv_map_real_weights(
        weights: UT3FrameWeightsData,  # (up, down, left, right, masks)
        fn,                            # (xnp, w) -> w, applied elementwise to the REAL slots only
) -> UT3FrameWeightsData:              # fn on the real slots; padding forced to a canonical, finite 0
    """Apply ``fn`` to the real slots of all four families, forcing the padding to a finite ``0``. The
    frame-weight twin of ``_ut3_map_real_weights``; see it for why the double-``where`` is required (``fn``
    may be undefined or non-differentiable at the padding's canonical zero, and a ``nan`` from a dead
    branch still propagates through the gradient)."""
    xnp, _, _ = get_backend(True, tree_contains_jax(weights[:4]))
    masks = weights[4]
    require_concrete_masks(*masks)  # masks are host, not traced

    def go(w, m):
        neutral = xnp.where(m, w, 1.0)              # padding (canonical 0 OR garbage) -> 1: fn safe, grad finite
        return xnp.where(m, fn(xnp, neutral), 0.0)  # real slots: fn(w). padding: the canonical finite 0

    return tuple(go(w, m) for w, m in zip(weights[:4], masks)) + (masks,)


def ufv_reciprocal_weights(weights: UT3FrameWeightsData) -> UT3FrameWeightsData:
    """Elementwise ``1/w`` on the real slots of all four families (masks unchanged); the padding stays a
    canonical, **finite** zero rather than becoming ``inf``. See ``ut3_reciprocal_weights`` -- this is the
    Grasedyck-Kramer path (``from_ut3weights(from_ut3svd(x)).reciprocal()``), so the guard is on the
    headline route. Real-slot zeros are deliberately NOT guarded."""
    return _ufv_map_real_weights(weights, lambda xnp, w: 1.0 / w)


def ufv_sqrt_weights(weights: UT3FrameWeightsData) -> UT3FrameWeightsData:
    """Elementwise ``sqrt`` on the real slots of all four families; the padding stays a canonical, finite
    zero (masks unchanged)."""
    return _ufv_map_real_weights(weights, lambda xnp, w: xnp.sqrt(w))


def ufv_concatenate_weights(
        weights_A: UT3FrameWeightsData,
        weights_B: UT3FrameWeightsData,
) -> UT3FrameWeightsData:  # per-family concatenation: padded widths add, masks concatenate
    """Per-edge concatenation of two frame-weight 4-tuples (the ``+`` combine; ranks add). Supercores and
    masks concatenate on the last axis -- the same operation applied twice, because concatenation commutes
    with elementwise multiply. Output masks may go **gappy** (expected; ``docs/uniform_masks_vs_ranks.md``)."""
    xnp, _, _ = get_backend(True, tree_contains_jax((weights_A[:4], weights_B[:4])))
    require_concrete_masks(*weights_A[4], *weights_B[4])
    families = tuple(xnp.concatenate([a, b], axis=-1) for a, b in zip(weights_A[:4], weights_B[:4]))
    # np (host): masks are static structure.
    masks = tuple(np.concatenate([a, b], axis=-1) for a, b in zip(weights_A[4], weights_B[4]))
    return families + (masks,)


def ufv_kronecker_weights(
        weights_A: UT3FrameWeightsData,
        weights_B: UT3FrameWeightsData,
) -> UT3FrameWeightsData:  # per-family Kronecker: padded widths multiply, masks Kronecker
    """Per-edge Kronecker product of two frame-weight 4-tuples (the Hadamard combine; ranks multiply).

    Kronecker the weights, Kronecker the masks -- one operation applied twice, since the Kronecker product
    commutes with elementwise multiply (see ``ut3_kronecker_weights`` for the argument). A **last-axis
    outer product broadcasting the shared prefix**, A-major -- NOT ``np.kron``. Output masks are strided
    (gappy), which is correct."""
    xnp, _, _ = get_backend(True, tree_contains_jax((weights_A[:4], weights_B[:4])))
    require_concrete_masks(*weights_A[4], *weights_B[4])

    def kron_last(a, b, np_module):  # (...,pA),(...,pB) -> (...,pA*pB), A-major, shared prefix broadcast
        prefix = np_module.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        return (a[..., :, None] * b[..., None, :]).reshape(prefix + (a.shape[-1] * b.shape[-1],))

    families = tuple(kron_last(a, b, xnp) for a, b in zip(weights_A[:4], weights_B[:4]))
    masks = tuple(kron_last(a, b, np) for a, b in zip(weights_A[4], weights_B[4]))
    return families + (masks,)
