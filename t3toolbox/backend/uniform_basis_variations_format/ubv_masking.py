# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'apply_basis_masks',
    'apply_variations_masks',
]


def make_basis_masks(
        up_ranks: NDArray,      # dtype=int, shape=(d,)+stack_shape
        down_ranks: NDArray,    # dtype=int, shape=(d,)+stack_shape
        left_ranks,             # dtype=int, shape=(d+1,)+stack_shape
        right_ranks,            # dtype=int, shape=(d+1,)+stack_shape
        n: int,
        r: int,
) -> typ.Tuple[
    NDArray, # up_mask, dtype=bool, shape=(d,)+stack_shape+(n,)
    NDArray, # down_mask, dtype=bool, shape=(d,)+stack_shape+(n,)
    NDArray, # left_mask, dtype=bool, shape=(d+1,)+stack_shape+(r,)
    NDArray, # right_mask, dtype=bool, shape=(d+1,)+stack_shape+(r,)
]:
    pass


def apply_basis_masks(
        up_tucker_supercore:    NDArray,  # B_dxo B_dyo   = I_dxy, shape = (d,)+stack_shape+(nU, N)
        down_tt_supercore:      NDArray,  # R_dixj R_diyj = I_dxy  shape = (d,)+stack_shape+(rL, nD, rR)
        left_tt_supercore:      NDArray,  # P_diax P_diay = I_dxy, shape = (d,)+stack_shape+(rL, nU, rL)
        right_tt_supercore:     NDArray,  # Q_dxaj Q_dyaj = I_dxy  shape = (d,)+stack_shape+(rR, nU, rR)
        shape_mask:             NDArray,  # dtype=bool, (d,N)
        up_mask:                NDArray,  # dtype=bool, shape=(d,)+stack_shape+nU
        down_mask:              NDArray,  # dtype=bool, shape=(d,)+stack_shape+nD
        basis_left_mask:        NDArray,  # dtype=bool, shape=(d+1,)+stack_shape+rL
        basis_right_mask:       NDArray,  # dtype=bool, shape=(d+1,)+stack_shape+rR
) -> typ.Tuple[
    NDArray,  # masked_up_tucker_supercore
    NDArray,  # masked_down_tt_supercore
    NDArray,  # masked_left_tt_supercore
    NDArray,  # masked_right_tt_supercore
]:
    d = up_tucker_supercore.shape[0]
    ss = up_tucker_supercore.shape[1:-2]
    nU = up_tucker_supercore.shape[-2]
    N = up_tucker_supercore.shape[-1]
    rL = down_tt_supercore.shape[-3]
    nD = down_tt_supercore.shape[-2]
    rR = down_tt_supercore.shape[-1]

    SM_k = shape_mask.reshape(           (d,) + (1,)*len(ss) + (1,)  + (N,))
    UM_k = up_mask.reshape(              (d,) + ss           + (nU,) + (1,))
    UM_t = up_mask.reshape(              (d,) + ss           + (1,)  + (nU,) + (1,))
    DM_k = down_mask.reshape(            (d,) + ss           + (1,)  + (nD,) + (1,)) # not used
    DM_t = down_mask.reshape(            (d,) + ss           + (1,)  + (nD,) + (1,))
    LM_l = basis_left_mask[:-1].reshape( (d,) + ss           + (rL,) + (1,)  + (1,))
    LM_r = basis_left_mask[1:].reshape(  (d,) + ss           + (1,)  + (1,)  + (rL,))
    RM_l = basis_right_mask[:-1].reshape((d,) + ss           + (rR,) + (1,)  + (1,))
    RM_r = basis_right_mask[1:].reshape( (d,) + ss           + (1,)  + (1,)  + (rR,))

    masked_up_tucker_supercore = up_tucker_supercore * (SM_k * UM_k)
    masked_down_tt_supercore   = down_tt_supercore   * (LM_l * DM_t * RM_r)
    masked_left_tt_supercore   = left_tt_supercore   * (LM_l * UM_t * LM_r)
    masked_right_tt_supercore  = right_tt_supercore  * (RM_l * UM_t * RM_r)

    return (
        masked_up_tucker_supercore, masked_down_tt_supercore,
        masked_left_tt_supercore, masked_right_tt_supercore
    )


def apply_variations_masks(
        tucker_variations_supercore: NDArray,  # shape = (d,)+stack_shape+(nD, N)
        tt_variations_supercore:     NDArray,  # shape = (d,)+stack_shape+(rL, nU, rR)
        shape_mask:             NDArray,  # dtype=bool, (d,N)
        up_mask:                NDArray,  # dtype=bool, shape=(d,)+stack_shape+(nU,)
        down_mask:              NDArray,  # dtype=bool, shape=(d,)+stack_shape+(nD,)
        variations_left_mask:   NDArray,  # dtype=bool, shape=(d,)+stack_shape+(rL,)
        variations_right_mask:  NDArray,  # dtype=bool, shape=(d,)+stack_shape+(rR,)
) -> typ.Tuple[
    NDArray,  # masked_tucker_variations_supercore
    NDArray,  # masked_tt_variations_supercore
]:
    d = tucker_variations_supercore.shape[0]
    ss = tucker_variations_supercore.shape[1:-2]
    nD = tucker_variations_supercore.shape[-2]
    N = tucker_variations_supercore.shape[-1]
    rL = tt_variations_supercore.shape[-3]
    nU = tt_variations_supercore.shape[-2]
    rR = tt_variations_supercore.shape[-1]

    SM_k = shape_mask.reshape(            (d,) + (1,)*len(ss) + (1,)  + (N,))
    UM_k = up_mask.reshape(               (d,) + ss           + (nU,) + (1,)) # not used
    UM_t = up_mask.reshape(               (d,) + ss           + (1,)  + (nU,) + (1,))
    DM_k = down_mask.reshape(             (d,) + ss           + (1,)  + (nD,) + (1,))
    DM_t = down_mask.reshape(             (d,) + ss           + (1,)  + (nD,) + (1,)) # not used
    LM_l = variations_left_mask.reshape(  (d,) + ss           + (rL,) + (1,)  + (1,))
    RM_r = variations_right_mask.reshape( (d,) + ss           + (1,)  + (1,)  + (rR,))

    masked_tucker_variations_supercore = tucker_variations_supercore * (SM_k * DM_k)
    masked_tt_variations_supercore     = tt_variations_supercore     * (LM_l * UM_t * RM_r)

    return masked_tucker_variations_supercore, masked_tt_variations_supercore



