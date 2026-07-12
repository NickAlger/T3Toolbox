# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Mask construction/application for uniform (frame, variations) data.

Masks are static structure and ALWAYS host numpy (``np``, never ``xnp``) -- intentional, required
for jit; do not "fix" it (``docs/uniform_masks_vs_ranks.md``, ``docs/contributor/uniform_pytree_composition.md``).
"""
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'ufv_make_frame_masks',
    'ufv_apply_frame_masks',
    'ufv_apply_variations_masks',
]


def ufv_make_frame_masks(
        up_ranks:    NDArray,   # HOST int, (d,)   + stack_shape
        down_ranks:  NDArray,   # HOST int, (d,)   + stack_shape
        left_ranks:  NDArray,   # HOST int, (d+1,) + stack_shape
        right_ranks: NDArray,   # HOST int, (d+1,) + stack_shape
        nU: int,
        nD: int,
        rL: int,
        rR: int,
) -> typ.Tuple[
    NDArray,  # up_mask,    HOST bool, (d,)   + stack_shape + (nU,)
    NDArray,  # down_mask,  HOST bool, (d,)   + stack_shape + (nD,)
    NDArray,  # left_mask,  HOST bool, (d+1,) + stack_shape + (rL,)
    NDArray,  # right_mask, HOST bool, (d+1,) + stack_shape + (rR,)
]:
    """Build the prefix RANK edge masks for a uniform frame. The physical ``shape`` is a separate
    int tuple (not a mask), so this returns only the four rank masks. HOST numpy (masks are static
    structure -- ``np``, not ``xnp``; see ``docs/contributor/uniform_pytree_composition.md``)."""
    up_mask    = np.arange(nU) < np.asarray(up_ranks)[..., None]
    down_mask  = np.arange(nD) < np.asarray(down_ranks)[..., None]
    left_mask  = np.arange(rL) < np.asarray(left_ranks)[..., None]
    right_mask = np.arange(rR) < np.asarray(right_ranks)[..., None]
    return up_mask, down_mask, left_mask, right_mask


def ufv_apply_frame_masks(
        data: typ.Tuple[
            NDArray,             # up_tucker_supercore,  (d,)+stack_shape+(nU, N)
            NDArray,             # down_tt_supercore,    (d,)+stack_shape+(rL, nD, rR)
            NDArray,             # left_tt_supercore,    (d,)+stack_shape+(rL, nU, rL)
            NDArray,             # right_tt_supercore,   (d,)+stack_shape+(rR, nU, rR)
            typ.Sequence[int],   # shape = (N0,...,N(d-1)), static int tuple
            typ.Tuple[
                NDArray,  # up_mask,          dtype=bool, (d,)  +stack_shape+(nU,)
                NDArray,  # down_mask,        dtype=bool, (d,)  +stack_shape+(nD,)
                NDArray,  # frame_left_mask,  dtype=bool, (d+1,)+stack_shape+(rL,)
                NDArray,  # frame_right_mask, dtype=bool, (d+1,)+stack_shape+(rR,)
            ],
        ],
) -> typ.Tuple[
    NDArray,  # masked_up_tucker_supercore
    NDArray,  # masked_down_tt_supercore
    NDArray,  # masked_left_tt_supercore
    NDArray,  # masked_right_tt_supercore
]:
    """Zero the padded ("garbage") regions of the frame supercores via the edge masks. The physical
    ``shape_mask`` is reconstructed on the host from the static ``shape`` ints (``np``, never ``jnp`` --
    a traced mask breaks the layer; see ``docs/contributor/uniform_pytree_composition.md``)."""
    (up_tucker_supercore, down_tt_supercore, left_tt_supercore, right_tt_supercore,
     shape, (up_mask, down_mask, frame_left_mask, frame_right_mask)) = data

    d = up_tucker_supercore.shape[0]
    ss = up_tucker_supercore.shape[1:-2]
    nU = up_tucker_supercore.shape[-2]
    N = up_tucker_supercore.shape[-1]
    rL = down_tt_supercore.shape[-3]
    nD = down_tt_supercore.shape[-2]
    rR = down_tt_supercore.shape[-1]

    shape_mask = np.arange(N) < np.asarray(shape)[:, None]  # (d, N) HOST bool, reconstructed from ints

    SM_k = shape_mask.reshape(           (d,) + (1,)*len(ss) + (1,)  + (N,))
    UM_k = up_mask.reshape(              (d,) + ss           + (nU,) + (1,))
    UM_t = up_mask.reshape(              (d,) + ss           + (1,)  + (nU,) + (1,))
    DM_t = down_mask.reshape(            (d,) + ss           + (1,)  + (nD,) + (1,))
    LM_l = frame_left_mask[:-1].reshape( (d,) + ss           + (rL,) + (1,)  + (1,))
    LM_r = frame_left_mask[1:].reshape(  (d,) + ss           + (1,)  + (1,)  + (rL,))
    RM_l = frame_right_mask[:-1].reshape((d,) + ss           + (rR,) + (1,)  + (1,))
    RM_r = frame_right_mask[1:].reshape( (d,) + ss           + (1,)  + (1,)  + (rR,))

    masked_up_tucker_supercore = up_tucker_supercore * (SM_k * UM_k)
    masked_down_tt_supercore   = down_tt_supercore   * (LM_l * DM_t * RM_r)
    masked_left_tt_supercore   = left_tt_supercore   * (LM_l * UM_t * LM_r)
    masked_right_tt_supercore  = right_tt_supercore  * (RM_l * UM_t * RM_r)

    return (
        masked_up_tucker_supercore, masked_down_tt_supercore,
        masked_left_tt_supercore, masked_right_tt_supercore
    )


def ufv_apply_variations_masks(
        data: typ.Tuple[
            NDArray,             # tucker_variations_supercore, (d,)+stack_shape+(nD, N)
            NDArray,             # tt_variations_supercore,     (d,)+stack_shape+(rL, nU, rR)
            typ.Sequence[int],   # shape = (N0,...,N(d-1)), static int tuple
            typ.Tuple[
                NDArray,  # variations_up_mask,    dtype=bool, (d,)+stack_shape+(nU,)
                NDArray,  # variations_down_mask,  dtype=bool, (d,)+stack_shape+(nD,)
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
     shape, (up_mask, down_mask, variations_left_mask, variations_right_mask)) = data

    d = tucker_variations_supercore.shape[0]
    ss = tucker_variations_supercore.shape[1:-2]
    nD = tucker_variations_supercore.shape[-2]
    N = tucker_variations_supercore.shape[-1]
    rL = tt_variations_supercore.shape[-3]
    nU = tt_variations_supercore.shape[-2]
    rR = tt_variations_supercore.shape[-1]

    shape_mask = np.arange(N) < np.asarray(shape)[:, None]  # (d, N) HOST bool, reconstructed from ints

    # tucker_variations (d,)+ss+(nD, N): mask nD by variations_down, N by shape.
    SM   = shape_mask.reshape(            (d,) + (1,)*len(ss) + (1,)  + (N,))
    DM_k = down_mask.reshape(             (d,) + ss           + (nD,) + (1,))
    # tt_variations (d,)+ss+(rL, nU, rR): mask rL/nU/rR by left/up/right.
    LM_l = variations_left_mask.reshape(  (d,) + ss           + (rL,) + (1,)  + (1,))
    UM_t = up_mask.reshape(               (d,) + ss           + (1,)  + (nU,) + (1,))
    RM_r = variations_right_mask.reshape( (d,) + ss           + (1,)  + (1,)  + (rR,))

    masked_tucker_variations_supercore = tucker_variations_supercore * (SM * DM_k)
    masked_tt_variations_supercore     = tt_variations_supercore     * (LM_l * UM_t * RM_r)

    return masked_tucker_variations_supercore, masked_tt_variations_supercore



