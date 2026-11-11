# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Mask construction/application for uniform (frame, variations) data.

Masks are static structure and ALWAYS host numpy (``np``, never ``xnp``) -- intentional, required
for jit; do not "fix" it (``docs/uniform_masks_vs_ranks.md``, ``docs/contributor/uniform_pytree_composition.md``).
The two ``ufv_apply_*_masks`` functions are the frame/variation **mask chokepoints** and guard the
contract with ``require_concrete_masks``, mirroring ``ut3_apply_masks`` on the plain layer.
"""
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'ufv_make_frame_masks',
    'ufv_variation_masks',
    'ufv_variation_masks_over_stack',
    'ufv_apply_frame_masks',
    'ufv_apply_variations_masks',
    'ufv_frame_widths',
]


def ufv_variation_masks(
        frame_masks:  typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (up, down, frame_left, frame_right); HOST bool
) -> typ.Tuple[NDArray, NDArray, NDArray, NDArray]:                   # (up, down, left[:-1], right[1:])
    """The variation masks of a frame: its rank masks with the **gauge shift** applied.

    A variation occupies one TT slot, not a boundary edge, so it takes the frame's left masks without
    the last and the right masks without the first. The up/down masks pass through. This one line is
    the whole convention, and it was previously open-coded at four sites; call it rather than slicing
    by hand. (The frontend twin is ``UT3Variations._variation_masks_of``.)"""
    down_mask, up_mask, frame_left_mask, frame_right_mask = frame_masks
    return (down_mask, up_mask, frame_left_mask[:-1], frame_right_mask[1:])


def ufv_frame_widths(
        N:  int,   # the point's padded mode size
        n:  int,   # its padded Tucker width
        r:  int,   # its padded TT width
) -> typ.Tuple[
    int,   # WD = min(n, N)
    int,   # WU = min(n, N, r*r)
    int,   # WL = r
    int,   # WR = r
]:
    """The padded widths ``(nD, nU, rL, rR)`` of the frame ``ut3_orthogonal_representations`` gives a point padded to
    ``(N, n, r)``.

    Examples
    --------
    >>> import t3toolbox.backend.ufv_masking as ufv_masking
    >>> ufv_masking.ufv_frame_widths(8, 5, 3)
    (5, 5, 3, 3)
    >>> ufv_masking.ufv_frame_widths(4, 6, 2)      # the Tucker width capped by N, the down width by r*r
    (4, 4, 2, 2)
    """
    return min(n, N), min(n, N, r * r), r, r


def ufv_make_frame_masks(
        down_ranks:    NDArray,   # HOST int, (d,)   + stack_shape
        up_ranks:  NDArray,   # HOST int, (d,)   + stack_shape
        left_ranks:  NDArray,   # HOST int, (d+1,) + stack_shape
        right_ranks: NDArray,   # HOST int, (d+1,) + stack_shape
        nD: int,
        nU: int,
        rL: int,
        rR: int,
) -> typ.Tuple[
    NDArray,  # down_mask,    HOST bool, (d,)   + stack_shape + (nD,)
    NDArray,  # up_mask,  HOST bool, (d,)   + stack_shape + (nU,)
    NDArray,  # left_mask,  HOST bool, (d+1,) + stack_shape + (rL,)
    NDArray,  # right_mask, HOST bool, (d+1,) + stack_shape + (rR,)
]:
    """Build the prefix RANK edge masks for a uniform frame. The physical ``shape`` is a separate
    int tuple (not a mask), so this returns only the four rank masks. HOST numpy (masks are static
    structure -- ``np``, not ``xnp``; see ``docs/contributor/uniform_pytree_composition.md``)."""
    down_mask    = prefix_mask(down_ranks, nD)
    up_mask  = prefix_mask(up_ranks, nU)
    left_mask  = prefix_mask(left_ranks, rL)
    right_mask = prefix_mask(right_ranks, rR)
    return down_mask, up_mask, left_mask, right_mask


def ufv_apply_frame_masks(
        data: typ.Tuple[
            NDArray,             # down_tucker_supercore,  (d,)+stack_shape+(nD, N)
            NDArray,             # up_tt_supercore,    (d,)+stack_shape+(rL, nU, rR)
            NDArray,             # left_tt_supercore,    (d,)+stack_shape+(rL, nD, rL)
            NDArray,             # right_tt_supercore,   (d,)+stack_shape+(rR, nD, rR)
            typ.Sequence[int],   # shape = (N0,...,N(d-1)), static int tuple
            typ.Tuple[
                NDArray,  # down_mask,          dtype=bool, (d,)  +stack_shape+(nD,)
                NDArray,  # up_mask,        dtype=bool, (d,)  +stack_shape+(nU,)
                NDArray,  # frame_left_mask,  dtype=bool, (d+1,)+stack_shape+(rL,)
                NDArray,  # frame_right_mask, dtype=bool, (d+1,)+stack_shape+(rR,)
            ],
        ],
) -> typ.Tuple[
    NDArray,  # masked_down_tucker_supercore
    NDArray,  # masked_up_tt_supercore
    NDArray,  # masked_left_tt_supercore
    NDArray,  # masked_right_tt_supercore
]:
    """Zero the padded ("garbage") regions of the frame supercores via the edge masks. The physical
    ``shape_mask`` is reconstructed on the host from the static ``shape`` ints (``np``, never ``jnp`` --
    a traced mask breaks the layer; see ``docs/contributor/uniform_pytree_composition.md``)."""
    (down_tucker_supercore, up_tt_supercore, left_tt_supercore, right_tt_supercore,
     shape, (down_mask, up_mask, frame_left_mask, frame_right_mask)) = data
    require_concrete_masks(down_mask, up_mask, frame_left_mask, frame_right_mask)  # host, not traced

    d = down_tucker_supercore.shape[0]
    ss = down_tucker_supercore.shape[1:-2]
    nD = down_tucker_supercore.shape[-2]
    N = down_tucker_supercore.shape[-1]
    rL = up_tt_supercore.shape[-3]
    nU = up_tt_supercore.shape[-2]
    rR = up_tt_supercore.shape[-1]

    shape_mask = prefix_mask(shape, N)  # (d, N) HOST bool, reconstructed from the static shape ints

    SM_k = shape_mask.reshape(           (d,) + (1,)*len(ss) + (1,)  + (N,))
    DM_k = down_mask.reshape(              (d,) + ss           + (nD,) + (1,))
    DM_t = down_mask.reshape(              (d,) + ss           + (1,)  + (nD,) + (1,))
    UM_t = up_mask.reshape(            (d,) + ss           + (1,)  + (nU,) + (1,))
    LM_l = frame_left_mask[:-1].reshape( (d,) + ss           + (rL,) + (1,)  + (1,))
    LM_r = frame_left_mask[1:].reshape(  (d,) + ss           + (1,)  + (1,)  + (rL,))
    RM_l = frame_right_mask[:-1].reshape((d,) + ss           + (rR,) + (1,)  + (1,))
    RM_r = frame_right_mask[1:].reshape( (d,) + ss           + (1,)  + (1,)  + (rR,))

    masked_down_tucker_supercore = down_tucker_supercore * (SM_k * DM_k)
    masked_up_tt_supercore   = up_tt_supercore   * (LM_l * UM_t * RM_r)
    masked_left_tt_supercore   = left_tt_supercore   * (LM_l * DM_t * LM_r)
    masked_right_tt_supercore  = right_tt_supercore  * (RM_l * DM_t * RM_r)

    return (
        masked_down_tucker_supercore, masked_up_tt_supercore,
        masked_left_tt_supercore, masked_right_tt_supercore
    )


def ufv_apply_variations_masks(
        data: typ.Tuple[
            NDArray,             # tucker_variations_supercore, (d,)+stack_shape+(nU, N)
            NDArray,             # tt_variations_supercore,     (d,)+stack_shape+(rL, nD, rR)
            typ.Sequence[int],   # shape = (N0,...,N(d-1)), static int tuple
            typ.Tuple[
                NDArray,  # variations_down_mask,    dtype=bool, (d,)+stack_shape+(nD,)
                NDArray,  # variations_up_mask,  dtype=bool, (d,)+stack_shape+(nU,)
                NDArray,  # variations_left_mask,  dtype=bool, (d,)+stack_shape+(rL,)
                NDArray,  # variations_right_mask, dtype=bool, (d,)+stack_shape+(rR,)
            ],
        ],
) -> typ.Tuple[
    NDArray,  # masked_tucker_variations_supercore
    NDArray,  # masked_tt_variations_supercore
]:
    """Zero the padded ("garbage") regions of the variation supercores via the edge masks. ``shape_mask``
    is reconstructed on the host from the static ``shape`` ints (``np``, never ``jnp``)."""
    (tucker_variations_supercore, tt_variations_supercore,
     shape, (down_mask, up_mask, variations_left_mask, variations_right_mask)) = data
    require_concrete_masks(down_mask, up_mask, variations_left_mask, variations_right_mask)  # host, not traced

    d = tucker_variations_supercore.shape[0]
    ss = tucker_variations_supercore.shape[1:-2]
    nU = tucker_variations_supercore.shape[-2]
    N = tucker_variations_supercore.shape[-1]
    rL = tt_variations_supercore.shape[-3]
    nD = tt_variations_supercore.shape[-2]
    rR = tt_variations_supercore.shape[-1]

    shape_mask = prefix_mask(shape, N)  # (d, N) HOST bool, reconstructed from the static shape ints

    # tucker_variations (d,)+ss+(nU, N): mask nU by variations_up, N by shape.
    SM   = shape_mask.reshape(            (d,) + (1,)*len(ss) + (1,)  + (N,))
    UM_k = up_mask.reshape(             (d,) + ss           + (nU,) + (1,))
    # tt_variations (d,)+ss+(rL, nD, rR): mask rL/nD/rR by left/up/right.
    LM_l = variations_left_mask.reshape(  (d,) + ss           + (rL,) + (1,)  + (1,))
    DM_t = down_mask.reshape(               (d,) + ss           + (1,)  + (nD,) + (1,))
    RM_r = variations_right_mask.reshape( (d,) + ss           + (1,)  + (1,)  + (rR,))

    masked_tucker_variations_supercore = tucker_variations_supercore * (SM * UM_k)
    masked_tt_variations_supercore     = tt_variations_supercore     * (LM_l * DM_t * RM_r)

    return masked_tucker_variations_supercore, masked_tt_variations_supercore





def ufv_variation_masks_over_stack(
        frame_masks,        # the frame's (up, down, frame_left, frame_right) masks, each (len,)+C+(size,)
        frame_stack,        # C, the frame stack
        out_stack,          # K + C, the stack of the variations the masks are for (K leading, possibly empty)
):                          # -> the 4 variation masks, each (d,)+K+C+(size,), HOST numpy
    """The frame's gauge-shifted variation masks (:py:func:`ufv_variation_masks`) broadcast (constant) over
    an extra leading tangent stack ``K`` -- the masks a ``K``-stack of tangents at one ``C``-stacked frame
    carries. Every op that produces variations with a stack wider than the frame's needs this: the
    sampling transposes (``K_new`` = the kept probe/tangent stack), the ambient projection of a
    ``K``-stacked gradient, hence ``transport`` of a ``K``-stacked tangent (the last two used to hand a
    ``C``-stack mask to ``K+C`` variations and fail in a reshape, review C3). Host numpy: static structure."""
    return _masks_over_stack(ufv_variation_masks(frame_masks), frame_stack, out_stack)


def _masks_over_stack(
        masks,              # a family of masks, each (len,)+C+(size,), HOST numpy
        frame_stack,        # C, the stack the masks carry
        out_stack,          # K + C, the stack of the arrays the masks are for (K leading, possibly empty)
):                          # -> the same family, each (len,)+K+C+(size,), HOST numpy (a read-only broadcast view)
    """Broadcast a mask family (constant) over an extra leading tangent stack ``K`` -- the shared half of
    :py:func:`ufv_variation_masks_over_stack`, for masks that are already gauge-shifted (a geometry's held
    variation masks) or that need no shift (a point's ``(up, left)`` pair). ``K = ()`` returns views of the
    same values. Host numpy: static structure."""
    K = tuple(out_stack[:len(out_stack) - len(frame_stack)])
    if tuple(out_stack[len(K):]) != tuple(frame_stack):
        raise ValueError('variation stack %s does not end with the frame stack %s' % (tuple(out_stack), tuple(frame_stack)))

    def b(m):  # (len,)+C+(size,) -> (len,)+K+C+(size,)
        return np.broadcast_to(m.reshape(m.shape[:1] + (1,) * len(K) + m.shape[1:]), m.shape[:1] + K + m.shape[1:])

    return tuple(b(m) for m in masks)
