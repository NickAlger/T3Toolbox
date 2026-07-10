# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Stateless tangent-stack reshuffles for the uniform tangent layer (UT3Tangent), uniform-fix 3b-1b.

The uniform mirror of the ragged ``tangent_operations`` stack/unstack helpers, for the ``.data`` layout
``(*supercores, shape, masks)`` with a leading mode index ``d`` and the stack at axes ``1 ..``. A uniform
tangent's variations carry the full ``K + C`` stack (tangent stack ``K`` outermost, frame stack ``C``
inner); the frame carries only ``C``. These functions split that stack into a Python tree of per-element
objects and recombine -- a tree<->array conversion, NOT an axis permutation of the supercores. They are
the backend the UT3Tangent ``unstack_tangents`` / ``unstack_frame`` / ``stack_tangents`` / ``stack_frame``
/ ``sum_tangents`` methods delegate to.

Varying ranks across the frame ``C`` stack are first-class here (the rank-sweep use case): the per-element
masks may differ, and ``stack_frame_stack`` stacks frames of different ranks into one batch (the masks just
ride along). Uniform rank is required only across ``K`` (one shared frame = one tangent space), which
``stack_tangent_stack`` gets for free (the leaves share a frame). See
``docs/uniform_ranks_and_varieties.md``.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ufv_operations as ufv_operations
import t3toolbox.backend.ufv_masking as ufv_masking
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ut3_svd as ut3_svd
import t3toolbox.backend.tangent_operations as tangent_operations
from t3toolbox.backend.common import *

__all__ = [
    'tangent_to_ut3',
    'retract',
    'corewise_retract',
    'ufv_corewise_inner',
    'orthogonal_gauge_projection',
    'oblique_gauge_projection',
    'gauge_residual',
    'project_ut3_onto_tangent_space',
    'unstack_tangent_stack',
    'stack_tangent_stack',
    'unstack_frame_stack',
    'stack_frame_stack',
    'sum_tangent_stack',
]


def _tangent_stack_split(
        frame_data,       # (up, down, left, right, shape, masks), each supercore stack = C
        variations_data,  # (tkv, ttv, shape, masks),             each supercore stack = K + C
) -> typ.Tuple[int, int]:  # (|K|, |C|)
    """Recover the tangent / frame stack split (|K|, |C|) from a (frame, variations) ``.data`` pair.

    The frame supercores carry only the frame stack ``C`` (up: ``(d,)+C+(nU, N)``); the variation supercores
    carry the full ``K + C`` (tucker variation: ``(d,)+K+C+(nD, N)``). So ``|C|`` comes from a frame core
    and ``|K|`` is the remainder.
    """
    n_frame = frame_data[0].ndim - 3       # |C|   (up supercore: (d,) + C + (nU, N))
    n_full = variations_data[0].ndim - 3  # |K+C| (tucker variation: (d,) + K + C + (nD, N))
    return n_full - n_frame, n_frame


def _pair_leaves(frame_tree, variations_tree, n_frame):  # mirror tangent_operations._pair_frame_leaves
    """Pair a frame-data tree and a variations-data tree (same ``C``-shaped outer structure, ``n_frame``
    levels deep) leaf-by-leaf into one tree of ``(frame_data, variations_data)`` pairs. NOT a
    :py:func:`stacking.tree_zip`: the data-tuple leaves are themselves sequences (and carry the int-tuple
    ``shape``), so a generic zip would recurse into them -- we stop at the known frame depth ``n_frame``."""
    if n_frame == 0:
        return (frame_tree, variations_tree)  # both are single .data tuples -> one pair
    return tuple(_pair_leaves(b, v, n_frame - 1) for b, v in zip(frame_tree, variations_tree))


def _unpair_leaves(paired_tree, n_frame):  # inverse of _pair_leaves
    """Split a ``C``-shaped tree of ``(frame_data, variations_data)`` pairs back into
    ``(frame_tree, variations_tree)``."""
    if n_frame == 0:
        return paired_tree  # already a single (frame_data, variations_data) pair
    split = [_unpair_leaves(p, n_frame - 1) for p in paired_tree]
    return tuple(s[0] for s in split), tuple(s[1] for s in split)


def _depth_to_pair(paired_tree) -> int:  # |C|: nesting levels above a (frame_data, variations_data) leaf
    """Count the nesting depth of a ``C``-shaped tree of ``(frame_data, variations_data)`` pairs without
    recursing into the data tuples (whose int-tuple ``shape`` would fool :py:func:`stacking.tree_depth`).
    A leaf pair is reached when ``node[0][0]`` -- the first frame supercore of ``frame_data`` -- is an
    ndarray; until then ``node[0]`` is a subtree."""
    depth, node = 0, paired_tree
    while not is_ndarray(node[0][0]):
        node, depth = node[0], depth + 1
    return depth


def unstack_tangent_stack(
        frame_data,       # frame .data,      supercore stack = C
        variations_data,  # variations .data, supercore stack = K + C
):  # -> array-like tree (shape K) of variations .data tuples (each stack = C)
    """Peel the tangent stack ``K`` off the variations, returning a ``K``-shaped tree of variation-``.data``.

    The frame is shared across ``K``, so the frame is untouched (the caller pairs the same frame with every
    leaf). Inverse of :py:func:`stack_tangent_stack`."""
    n_tangent, _ = _tangent_stack_split(frame_data, variations_data)
    return ufv_operations.ufv_unstack_axes(variations_data, 2, range(1, 1 + n_tangent))


def stack_tangent_stack(
        variations_tree,  # array-like tree (shape K) of variations .data tuples (each stack = C)
):  # -> variations .data tuple (stack = K + C)
    """Stack a ``K``-shaped tree of variation-``.data`` over the tangent stack ``K`` (outermost; ``C`` stays
    inner). Inverse of :py:func:`unstack_tangent_stack`. The leaves share one frame, so ranks are uniform
    across ``K`` (the masks, constant along ``K``, just replicate)."""
    return ufv_operations.ufv_stack(variations_tree, 2)


def unstack_frame_stack(
        frame_data,       # frame .data,      supercore stack = C
        variations_data,  # variations .data, supercore stack = K + C
):  # -> array-like tree (shape C) of (frame_data, variations_data) pairs
    """Peel the frame stack ``C`` off both the frame and the variations, returning a ``C``-shaped tree whose
    leaves are ``(frame_data, variations_data)`` pairs -- one single-frame-point tangent per leaf.

    Each frame-``.data`` leaf has stack ``()`` (a single frame); each variations-``.data`` leaf has stack
    ``K``. The frame stack is the *inner* part of the ``K + C`` variation stack, so it is peeled from the
    interior axes ``1+|K| .. 1+|K|+|C|`` of the variation supercores; the frame's whole stack is ``C``. The
    leaves are paired for you (a plain :py:func:`stacking.tree_zip` would recurse into the data tuples).
    Inverse of :py:func:`stack_frame_stack`."""
    n_tangent, n_frame = _tangent_stack_split(frame_data, variations_data)
    frame_tree = ufv_operations.ufv_unstack(frame_data, 4)  # frame stack is all C
    variations_tree = ufv_operations.ufv_unstack_axes(variations_data, 2,
                                                      range(1 + n_tangent, 1 + n_tangent + n_frame))
    return _pair_leaves(frame_tree, variations_tree, n_frame)


def stack_frame_stack(
        paired_tree,  # array-like tree (shape C) of (frame_data, variations_data) pairs
):  # -> (frame_data [stack C], variations_data [stack K + C])
    """Stack a ``C``-shaped tree of ``(frame_data, variations_data)`` pairs over the frame stack ``C``.

    The frame stack is placed *innermost* (the variation stack becomes ``K + C``), matching the base-inner
    convention. Frames of DIFFERENT ranks stack into one batch (varying-``C`` -- the per-element masks just
    ride along); the shared requirement is only matching padded dims and tangent stack ``K``. Takes exactly
    the layout :py:func:`unstack_frame_stack` produces (its inverse)."""
    n_frame = _depth_to_pair(paired_tree)
    frame_tree, variations_tree = _unpair_leaves(paired_tree, n_frame)
    frame_data = ufv_operations.ufv_stack(frame_tree, 4)                 # C at axes 1.. (frame stack = C)
    n_tangent = ufv_operations._first_data_leaf(variations_tree)[0].ndim - 3  # |K| of a variations leaf
    variations_data = ufv_operations.ufv_stack_axes(variations_tree, 2, axes_start=1 + n_tangent)  # C after K
    return frame_data, variations_data


def sum_tangent_stack(
        variations_data,         # variations .data, supercore stack = K + C
        n_tangent:   int,        # |K|
        axis:        typ.Optional[int] = None,  # 0-based index WITHIN K (None = the whole tangent stack)
):  # -> variations .data with the summed K axes removed (stack = C, or K-with-one-axis-removed)
    """Sum the variations over the tangent stack ``K`` (a batch of tangents at one frame -> their sum;
    corewise == the tangent sum, by linearity). The frame stack ``C`` is preserved.

    The supercores sum via ``xnp``; the masks **OR** over the same axes (host ``np``). Because a ``K`` stack
    shares one frame, its masks are constant along ``K``, so the OR is a no-op (the summed tangent carries
    the frame's gauge masks) -- but it is the correct reduction in general. ``axis`` indexes within ``K``."""
    tkv, ttv, shape, masks = variations_data
    xnp, _, _ = get_backend(True, tree_contains_jax((tkv, ttv)))

    k_axes = tuple(range(1, 1 + n_tangent)) if axis is None else (1 + axis,)
    new_tkv = xnp.sum(tkv, axis=k_axes)
    new_ttv = xnp.sum(ttv, axis=k_axes)
    new_masks = tuple(np.any(m, axis=k_axes) for m in masks)   # host np: OR the real slots over K
    return new_tkv, new_ttv, shape, new_masks


def tangent_to_ut3(
        frame_data,       # UT3Frame .data:      (up, down, left, right, shape, (4 masks)),  supercore stack = C
        variations_data,  # UT3Variations .data: (tkv, ttv, shape, (4 masks)),               supercore stack = K + C
        include_shift: bool = False,  # False: tangent vector v. True: base point + v.
):  # -> doubled-rank UniformTuckerTensorTrain .data: (tucker_supercore, tt_supercore, shape, (tucker_mask, tt_mask))
    """Doubled-rank uniform Tucker tensor train representing a uniform frame-variations tangent vector.

    The uniform mirror of :py:func:`tangent_operations.tangent_to_t3` (equations (50)-(53) / Figure 20,
    Appendix A.3.1 of Alger et al. 2026). The Tucker supercore becomes ``[U ; V]`` (concat along the
    Tucker-rank axis); the TT supercore is the block-bidiagonal embedding, uniform-padded to bonds
    ``rL+rR`` for every core with the **base-inner ``[R, L]`` bond order** (mirroring the ragged build).
    The doubled rank masks are concatenations of the existing masks (the **#1 trap**: the appended boundary
    slots are FULL ``ones`` -- the supercore is zero there, so to_dense's mask-then-contract is unaffected):
    ``tucker_mask = concat([up, down])``; ``tt_mask = concat([right_ext, left_ext])`` with
    ``left_ext = [var_left, ones]`` and ``right_ext = [ones, var_right]``.

    Stack-aware: the variation supercores carry ``K + C``; the frame supercores (stack ``C``) are broadcast
    up to ``K + C`` (mirror ragged ``bcast``), and the masks (host numpy, carrying ``K + C`` already from
    the variations) are concatenated on the host. With ``include_shift=True`` the base point is folded into
    the last core (``base point + v``)."""
    up_sc, down_sc, left_sc, right_sc, shape, _base_masks = frame_data
    tkv, ttv, _shape_v, var_masks = variations_data
    var_up_mask, var_down_mask, var_left_mask, var_right_mask = var_masks

    use_jax = tree_contains_jax((up_sc, down_sc, left_sc, right_sc, tkv, ttv))
    xnp, _, _ = get_backend(True, use_jax)

    d  = up_sc.shape[0]
    nU = up_sc.shape[-2]; N = up_sc.shape[-1]; nD = down_sc.shape[-2]
    rL = left_sc.shape[-1]; rR = right_sc.shape[-1]
    frame_stack = up_sc.shape[1:-2]                 # C
    ss = tkv.shape[1:-2]                           # K + C (the variation/output stack)
    n_K = len(ss) - len(frame_stack)

    def bcast(sc):  # frame supercore (d,)+C+(core) -> (d,)+K+C+(core): insert |K| size-1 axes after d
        return xnp.broadcast_to(sc.reshape(sc.shape[:1] + (1,) * n_K + sc.shape[1:]),
                                sc.shape[:1] + tuple(ss[:n_K]) + sc.shape[1:])

    U = bcast(up_sc); O = bcast(down_sc); L = bcast(left_sc); R = bcast(right_sc)
    Z = lambda nd, a, b, c: xnp.zeros((nd,) + ss + (a, b, c))   # nd cores of a (a, b, c) zero block

    # ---- Tucker supercore: [U ; V] along the Tucker-rank axis (-2). Fully vectorized over d. ----
    tucker_supercore = xnp.concatenate([U, tkv], axis=-2)       # (d,)+ss+(nU+nD, N)

    # ---- TT supercore: first / mid / last cores, [R, L] bond order, padded to (rR+rL, nU+nD, rR+rL). ----
    def first_core():     # left bond is the boundary (the L-block); pad the R-block (rR rows) with zeros
        Gtop = xnp.concatenate([ttv[:1], L[:1]], axis=-1)                       # (rL, nU, rR+rL)
        Gbot = xnp.concatenate([O[:1], Z(1, rL, nD, rL)], axis=-1)             # (rL, nD, rR+rL)
        G = xnp.concatenate([Gtop, Gbot], axis=-2)                            # (rL, nU+nD, rR+rL)
        return xnp.concatenate([Z(1, rR, nU + nD, rR + rL), G], axis=-3)       # (rR+rL, nU+nD, rR+rL)

    def mid_cores():      # the full block-bidiagonal core (ragged-middle), vectorized over [1:-1]
        nd = d - 2
        Gtop = xnp.concatenate([
            xnp.concatenate([R[1:-1], Z(nd, rR, nU, rL)], axis=-1),            # [R, Z001] -> (rR, nU, rR+rL)
            xnp.concatenate([ttv[1:-1], L[1:-1]], axis=-1),                    # [dU, L]   -> (rL, nU, rR+rL)
        ], axis=-3)                                                            # (rR+rL, nU, rR+rL)
        Gbot = xnp.concatenate([
            xnp.concatenate([Z(nd, rR, nD, rR), Z(nd, rR, nD, rL)], axis=-1),  # (rR, nD, rR+rL)
            xnp.concatenate([O[1:-1], Z(nd, rL, nD, rL)], axis=-1),            # [O, Z111] -> (rL, nD, rR+rL)
        ], axis=-3)                                                            # (rR+rL, nD, rR+rL)
        return xnp.concatenate([Gtop, Gbot], axis=-2)                         # (rR+rL, nU+nD, rR+rL)

    def last_core():      # right bond is the boundary (the R-block); pad the L-block (rL cols) with zeros
        dG = ttv[-1:]
        if include_shift:  # fold the base point's last core in: add its rank-1 right boundary (col 0) to dU
            Lf_boundary = xnp.concatenate(
                [L[-1:][..., :1], Z(1, rL, nU, rR - 1)], axis=-1)              # (rL, nU, rR), boundary in col 0
            dG = dG + Lf_boundary
        Gtop = xnp.concatenate([R[-1:], dG], axis=-3)                          # [R, dU] -> (rR+rL, nU, rR)
        Gbot = xnp.concatenate([Z(1, rR, nD, rR), O[-1:]], axis=-3)            # (rR+rL, nD, rR)
        G = xnp.concatenate([Gtop, Gbot], axis=-2)                            # (rR+rL, nU+nD, rR)
        return xnp.concatenate([G, Z(1, rR + rL, nU + nD, rL)], axis=-1)       # (rR+rL, nU+nD, rR+rL)

    if d < 2:
        # d==1 is degenerate: the single core's two boundaries don't double into a square (rR+rL) bond,
        # so it does not fit the square-bond uniform tt_supercore (r, n, r) when rL != rR. A single-mode T3
        # is a plain Tucker decomposition; support it separately if a real use case appears.
        raise NotImplementedError(
            'tangent_to_ut3 requires d >= 2 (the d == 1 single-mode case does not fit the square-bond '
            'uniform doubled-rank format).')
    blocks = [first_core()] + ([mid_cores()] if d > 2 else []) + [last_core()]
    tt_supercore = xnp.concatenate(blocks, axis=0)                             # (d,)+ss+(rR+rL, nU+nD, rR+rL)

    # ---- Doubled rank masks (HOST numpy; carry K+C from the variation masks). The order respects the
    # block structure (eqs 50-53): Tucker axis [U | dU] = [up | down]; the TT bond is [Q | P] = [R | L], so
    # the right-chain mask sits in the Q block and the left-chain mask in the P block -- the unique
    # concatenation where each diagonal mask multiplies the core block it belongs to. At the two boundary
    # bonds the global "1" sits in the CONTENT block (P at bond 0, Q at bond d); the opposite (free) block
    # has no core, hence rank 0 -- appended as zeros (not phantom ones), so the doubled ranks stay honest. ----
    tucker_mask = np.concatenate([var_up_mask, var_down_mask], axis=-1)        # (d,)+ss+(nU+nD,)  [U | dU]
    left_ext  = np.concatenate([var_left_mask,  np.zeros((1,) + ss + (rL,), bool)], axis=0)  # (d+1,)+ss+(rL,): P-part
    right_ext = np.concatenate([np.zeros((1,) + ss + (rR,), bool), var_right_mask], axis=0)  # (d+1,)+ss+(rR,): Q-part
    tt_mask = np.concatenate([right_ext, left_ext], axis=-1)                   # (d+1,)+ss+(rR+rL,)  [Q | P] order

    return tucker_supercore, tt_supercore, shape, (tucker_mask, tt_mask)


def retract(
        frame_data,       # UT3Frame .data:      supercore stack = C
        variations_data,  # UT3Variations .data: supercore stack = K + C
):  # -> retracted UniformTuckerTensorTrain .data (at the BASE point's ranks; stack = K + C)
    """Retract a uniform frame-variations tangent vector onto the fixed-rank manifold.

    Forms the shifted doubled-rank embedding ``base point + v`` (:py:func:`tangent_to_ut3` with
    ``include_shift=True``) and truncates it back to the **base point's own ranks** -- the Tucker ``up``
    ranks and ``left`` TT ranks read off the frame masks -- via the mask-truncated uniform T3-SVD. The output
    is a UT3 at the frame padded dims (``ut3svd`` truncates by max rank to a fixed shape, so no extra slice
    is needed), one retracted point per stack element. The uniform mirror of
    :py:func:`tangent_operations.retract` (the implicit T3-SVD / Algorithm 10, Alger et al. 2026).

    **Varying ranks across ``C``** work for free: the per-``C`` frame ranks are the per-element truncation
    targets. **The ``K`` (tangent) stack:** the frame ranks have stack ``C`` while the shifted UT3 has stack
    ``K + C``, so the frame ranks are broadcast over ``K`` (the ``K`` tangents share the frame, hence the same
    truncation targets)."""
    doubled = tangent_to_ut3(frame_data, variations_data, include_shift=True)   # .data, stack K + C
    ss = doubled[0].shape[1:-2]                       # K + C (the shifted UT3 stack)
    C = frame_data[0].shape[1:-2]                     # C (the frame stack)
    n_K = len(ss) - len(C)

    up_mask, _down_mask, frame_left_mask, _frame_right_mask = frame_data[5]
    up_ranks   = up_mask.sum(axis=-1)                 # (d,)   + C, HOST int
    left_ranks = frame_left_mask.sum(axis=-1)         # (d+1,) + C, HOST int

    def bcast_over_K(ranks):  # (L,)+C -> (L,)+K+C: the K tangents share the base point's truncation targets
        return np.broadcast_to(ranks.reshape(ranks.shape[:1] + (1,) * n_K + ranks.shape[1:]),
                               ranks.shape[:1] + ss)

    new_data, _ss_tucker, _ss_tt = ut3_svd.ut3svd(
        doubled, max_tucker_ranks=bcast_over_K(up_ranks), max_tt_ranks=bcast_over_K(left_ranks))
    return new_data


def corewise_retract(
        frame_data,       # UT3Frame .data: the (U, G, G, G) corewise frame, supercore stack = C
        variations_data,  # UT3Variations .data: free core perturbations (dU, dG), stack = K + C
):  # -> retracted UniformTuckerTensorTrain .data (at the base point's ranks; stack = K + C)
    """Additive (corewise) retraction: ``cores += variations``.

    The uniform mirror of the additive retraction on the corewise frame ``(U, G, G, G)`` (Section 6.3,
    Alger et al. 2026 -- the ``(P, Q, O) -> G`` substitution): recovers the point ``(U, G)`` from the frame
    (``up_tucker_supercore`` and ``left_tt_supercore``, which the corewise frame sets to the single core
    ``G``) and adds the variation supercores, giving a uniform Tucker tensor train at the base point's own
    ranks. Mirrors ``CorewiseGeometry.retract`` / ``corewise.corewise_add`` -- but the uniform supercores are
    ``d``-leading (stack interior), so a ``K`` tangent stack cannot be added by plain numpy broadcasting (it
    would misalign ``d`` with ``K``): the base point (stack ``C``) is broadcast up to ``K + C`` by inserting
    ``n_K`` size-1 axes *after* the leading mode axis -- the ``K`` perturbations share one base point. The
    result masks are the frame plain-UT3 masks (``up_mask``, ``frame_left_mask``) broadcast over ``K``.
    """
    up_sc = frame_data[0]       # (d,)   + C + (nU, N)        -- the point's Tucker core U
    G_sc  = frame_data[2]       # (d,)   + C + (rL, nU, rR)   -- the point's TT core G (left == right == G)
    shape = frame_data[4]
    up_mask, _down_mask, frame_left_mask, _frame_right_mask = frame_data[5]

    dU, dG = variations_data[0], variations_data[1]   # (d,) + K + C + (nD, N), (d,) + K + C + (rL, nU, rR)
    C  = up_sc.shape[1:-2]      # frame stack C
    ss = dU.shape[1:-2]         # K + C (the variation stack)
    n_K = len(ss) - len(C)

    def bcast_sc(sc):    # (d,)+C+core -> (d,)+(1,)*n_K+C+core: insert K size-1 axes after the leading mode axis
        return sc.reshape(sc.shape[:1] + (1,) * n_K + sc.shape[1:])

    def bcast_mask(m):   # (L,)+C+(rank,) -> (L,)+K+C+(rank,); host numpy (masks are np)
        return np.broadcast_to(m.reshape(m.shape[:1] + (1,) * n_K + m.shape[1:]),
                               m.shape[:1] + ss + m.shape[-1:])

    new_tk = bcast_sc(up_sc) + dU
    new_tt = bcast_sc(G_sc) + dG
    return (new_tk, new_tt, shape, (bcast_mask(up_mask), bcast_mask(frame_left_mask)))


def ufv_corewise_inner(
        variations_a:  typ.Tuple,  # UT3Variations .data: (tkv, ttv, shape, masks), supercore stack = K + C
        variations_b:  typ.Tuple,  # UT3Variations .data: same structure as variations_a
        n_stack:       int,        # number of leading stack axes (K + C) to keep; 0 -> a single scalar
) -> NDArray:                      # coordinate inner product, shape = stack_shape[:n_stack] (scalar if n_stack==0)
    """The raw coordinate (corewise) inner product of two uniform tangents' variations -- mask-applied and
    stack-keeping; **not** the Hilbert-Schmidt metric.

    The raw-tuple backend twin of :py:meth:`~t3toolbox.uniform_manifold.UT3Tangent.corewise_inner` (which
    delegates here). Masks both variation supercores once (``apply_variations_masks`` -- so the garbage
    padding is zeroed, never summed into the dot), then sums the elementwise product over the leading mode
    index ``d`` and the trailing core axes, **keeping the first** ``n_stack`` **stack axes** (one dot per
    stacked tangent). Pass ``n_stack = len(stack_shape)`` to keep the whole ``K + C`` stack, or
    ``n_stack = 0`` to collapse to a single scalar -- the unstacked optimizer's coordinate ``⟨·,·⟩``, the
    check-free twin the geometries' ``inner`` binds (it equals Hilbert-Schmidt only on an orthonormal,
    gauged frame). Masking makes it robust to garbage padding, so the reduction needs no clean-padding
    precondition.
    """
    use_jax = tree_contains_jax((variations_a[0], variations_a[1], variations_b[0], variations_b[1]))
    xnp, _, _ = get_backend(True, use_jax)
    masked_a = ufv_masking.apply_variations_masks(variations_a)   # (masked_tkv, masked_ttv), stack = K + C
    masked_b = ufv_masking.apply_variations_masks(variations_b)
    total = 0.0
    for sa, sb in zip(masked_a, masked_b):
        total = total + xnp.sum(sa * sb, axis=(0,) + tuple(range(1 + n_stack, sa.ndim)))
    return total


def orthogonal_gauge_projection(
        frame_data,       # UT3Frame .data
        variations_data,  # UT3Variations .data
):  # -> gauged variations .data (same masks; the tangent VECTOR changes)
    """Orthogonally project the variations onto the gauge-satisfying subspace (the uniform mirror of
    :py:func:`tangent_operations.orthogonal_gauge_projection`). Removes the component of each Tucker
    variation parallel to its up-core ``U`` and of each left-interior TT variation parallel to its
    left-core ``P`` -- the gauge conditions (48)-(49), Appendix A.3 of Alger et al. (2026). The
    represented tangent vector CHANGES (orthogonal, not oblique).

    Mask-once up front (the frame and variation padding zeroed), then mask-agnostic einsums vectorized over
    the leading mode axis ``d`` (a per-core *map*); the last TT variation is left unchanged (the ``[:-1]``
    boundary). The output carries the variation's masks unchanged (gauge preserves the rank structure)."""
    up_sc, _down, left_sc, _right = ufv_masking.apply_frame_masks(frame_data)
    tkv, ttv = ufv_masking.apply_variations_masks(variations_data)
    _tkv0, _ttv0, shape, masks = variations_data
    xnp, _, _ = get_backend(True, tree_contains_jax((up_sc, left_sc, tkv, ttv)))

    # TT variations: remove the P-parallel component (vectorized over d); keep the last core unchanged.
    gram = xnp.einsum('d...iaj,d...iak->d...jk', left_sc, ttv)                 # (P^L)^T dG^L, (d,)+stack+(rL, rR)
    parallel = xnp.einsum('d...iaj,d...jk->d...iak', left_sc, gram)
    new_ttv = xnp.concatenate([(ttv - parallel)[:-1], ttv[-1:]], axis=0)       # last TT variation untouched

    # Tucker variations: remove the U-parallel component (all d cores).
    gram_tk = xnp.einsum('d...jo,d...ko->d...jk', tkv, up_sc)                  # dB U^T, (d,)+stack+(nD, nU)
    new_tkv = tkv - xnp.einsum('d...jk,d...ko->d...jo', gram_tk, up_sc)

    return new_tkv, new_ttv, shape, masks


def oblique_gauge_projection(
        frame_data,       # UT3Frame .data
        variations_data,  # UT3Variations .data
):  # -> gauged variations .data (same masks; the tangent VECTOR is PRESERVED)
    """Project the variations onto the gauge-satisfying subspace while PRESERVING the tangent vector (the
    uniform mirror of :py:func:`tangent_operations.oblique_gauge_projection`). The Tucker perturbation is
    made perpendicular to ``U`` (compensating through the down core ``O``), then the TT variations are made
    left-perpendicular (compensating through the right core ``Q``).

    The Tucker step is independent per core -> vectorized over ``d``. The TT step carries a correction
    forward to the next core (a left-to-right sweep), implemented as an ``xscan`` over ``d`` -- NOT a Python
    loop: an unrolled loop bakes ``d`` copies of the step into the jit graph (compile time superlinear in
    the tensor order ``d``), while a scan compiles the body once. Mask-once up front. Enforces the gauge
    conditions (48)-(49), Appendix A.3 of Alger et al. (2026)."""
    up_sc, down_sc, left_sc, right_sc = ufv_masking.apply_frame_masks(frame_data)
    tkv, ttv = ufv_masking.apply_variations_masks(variations_data)
    _tkv0, _ttv0, shape, masks = variations_data
    xnp, _, xscan = get_backend(True, tree_contains_jax((up_sc, down_sc, left_sc, right_sc, tkv, ttv)))

    # Make Tucker variations perpendicular to U; compensate through the down core O (vectorized over d).
    X = xnp.einsum('d...jo,d...io->d...ji', tkv, up_sc)                        # dB U^T, (d,)+stack+(nD, nU)
    new_tkv = tkv - xnp.einsum('d...ji,d...io->d...jo', X, up_sc)
    ttv = ttv + xnp.einsum('d...aib,d...ij->d...ajb', down_sc, X)              # O-compensation into the TT vars

    # Make TT variations left-perpendicular; compensate through the right core Q. Left-to-right SCAN: the
    # carry is the previous core's X = (P^L)^T dG^L (shape stack+(rL, rR)); at core ii it first applies the
    # incoming correction X_prev @ Q_ii, then projects. The last core is left un-projected (its emitted
    # dG -- the corrected-but-not-projected core -- is used instead of its projection).
    ss = tkv.shape[1:-2]                                                       # variation stack K+C
    rL, rR = left_sc.shape[-1], right_sc.shape[-1]

    def _tt_step(X_prev, core):
        L, R, dG_in = core                                                    # slices: L,R (stack C), dG (stack K+C)
        dG = dG_in + xnp.einsum('...jk,...kbl->...jbl', X_prev, R)            # incoming correction X_prev @ Q_ii
        Xi = xnp.einsum('...iaj,...iak->...jk', L, dG)                         # (P^L)^T dG^L, stack+(rL, rR)
        proj = dG - xnp.einsum('...iaj,...jk->...iak', L, Xi)
        return Xi, (proj, dG)

    _, (proj_stack, dG_stack) = xscan(_tt_step, xnp.zeros(ss + (rL, rR)), (left_sc, right_sc, ttv))
    new_ttv = xnp.concatenate([proj_stack[:-1], dG_stack[-1:]], axis=0)        # last core: un-projected

    return new_tkv, new_ttv, shape, masks


def gauge_residual(
        frame_data,       # UT3Frame .data
        variations_data,  # UT3Variations .data
) -> NDArray:  # shape = variation stack (K+C); per stack element (scalar/0-d when unstacked)
    """Max violation of the gauge conditions for a uniform tangent vector, **per stack element** (the
    uniform mirror of :py:func:`tangent_operations.gauge_residual`).

    Each Tucker variation must be orthogonal to its up-core (``U^T dU = 0``) and each left-interior TT
    variation orthogonal to its left-core (``(P^L)^T dG^L = 0``). Mask-once, then the gauge grams vectorized
    over ``d``, max-abs reduced over the gram axes (keeping the stack), then the max over all checks (the
    last TT core is excluded -- the ``[:-1]`` boundary). A caller thresholds it (``<= atol``)."""
    up_sc, _down, left_sc, _right = ufv_masking.apply_frame_masks(frame_data)
    tkv, ttv = ufv_masking.apply_variations_masks(variations_data)
    xnp, _, _ = get_backend(True, tree_contains_jax((up_sc, left_sc, tkv, ttv)))

    g_tk = xnp.einsum('d...ia,d...ja->d...ij', up_sc, tkv)                     # U^T dU,        (d,)+stack+(nU, nD)
    dev_tk = xnp.max(xnp.abs(g_tk), axis=(-2, -1))                             # (d,)+stack
    g_tt = xnp.einsum('d...abi,d...abj->d...ij', left_sc, ttv)                 # (P^L)^T dG^L,  (d,)+stack+(rL, rR)
    dev_tt = xnp.max(xnp.abs(g_tt), axis=(-2, -1))                             # (d,)+stack

    devs = xnp.concatenate([dev_tk, dev_tt[:-1]], axis=0)                      # d Tucker + (d-1) interior TT
    return xnp.max(devs, axis=0)                                              # max over checks, keep the stack


def project_ut3_onto_tangent_space(
        frame_data,  # UT3Frame .data (an orthogonal frame), supercore stack = C
        x_data,      # UniformTuckerTensorTrain .data to project, supercore stack = C
):  # -> gauged variations .data (the orthogonal projection of x onto the tangent space at the frame)
    """Orthogonal projection of a uniform Tucker tensor train onto the tangent space at an orthogonal frame
    (the uniform mirror of :py:func:`tangent_operations.project_t3_onto_tangent_space`). Returns gauged
    variations representing the projection of ``x`` *directly* onto the tangent space (the linear subspace;
    it does NOT subtract the base point). The frame must be orthogonal (minimal rank not required).

    Mask-once up front, then: re-express ``x``'s TT cores in the frame's up-Tucker basis (a per-core map
    vectorized over ``d``); accumulate the left/right TT environments with the polymorphic
    :py:func:`tangent_operations.tt_zipper_*` (uniform ``xscan``, mask-free since the operands are masked);
    contract each environment into the ungauged Tucker/TT variations (a ``d``-axis map); then gauge."""
    up_sc, down_sc, left_sc, right_sc = ufv_masking.apply_frame_masks(frame_data)
    other_tk, other_tt = ut3_masking.apply_masks_to_cores(x_data)
    shape = frame_data[4]
    up_mask, down_mask, frame_left_mask, frame_right_mask = frame_data[5]
    xnp, _, _ = get_backend(True, tree_contains_jax((up_sc, down_sc, left_sc, right_sc, other_tk, other_tt)))

    # Re-express x's TT cores in the frame's up-Tucker basis (vectorized over d).
    BU1 = xnp.einsum('d...iz,d...xz->d...ix', other_tk, up_sc)                 # B_x U^T, (d,)+stack+(n_x, nU)
    other_tt2 = xnp.einsum('d...aib,d...ix->d...axb', other_tt, BU1)           # (d,)+stack+(rA, nU, rB)

    # Accumulate the left/right TT environments (polymorphic zippers -> tuples; stack into supercores).
    zl = xnp.stack(tangent_operations.tt_zipper_left_to_right(other_tt2[:-1], left_sc[:-1]), axis=0)
    zr = xnp.stack(tangent_operations.tt_zipper_right_to_left(other_tt2[1:], right_sc[1:]), axis=0)

    # Contract the environments into the ungauged variations (vectorized over d).
    env = xnp.einsum('d...ax,d...aib,d...by->d...xiy', zl, other_tt, zr)       # (d,)+stack+(rL, n_x, rR)
    BU = xnp.einsum('d...io,d...jo->d...ij', other_tk, up_sc)                  # B_x U^T, (d,)+stack+(n_x, nU)
    dG = xnp.einsum('d...xiy,d...ij->d...xjy', env, BU)                        # tt variation, (d,)+stack+(rL, nU, rR)
    M  = xnp.einsum('d...xiy,d...xjy->d...ij', env, down_sc)                   # (d,)+stack+(n_x, nD)
    dB = xnp.einsum('d...ij,d...io->d...jo', M, other_tk)                      # tucker variation, (d,)+stack+(nD, N)

    # Gauge the ungauged variations (they carry the frame's gauge-shifted masks).
    gauge_masks = (up_mask, down_mask, frame_left_mask[:-1], frame_right_mask[1:])
    return orthogonal_gauge_projection(frame_data, (dB, dG, shape, gauge_masks))
