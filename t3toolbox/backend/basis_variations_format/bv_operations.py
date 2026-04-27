# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *
import t3toolbox.backend.stacking as stacking

__all__ = [
    't3b_unstack',
    't3b_stack',
]


def t3b_unstack(
        x: typ.Tuple[
            typ.Tuple[NDArray, ...],  # up_tucker_cores
            typ.Tuple[NDArray, ...],  # down_tucker_cores
            typ.Tuple[NDArray, ...],  # left_tt_cores
            typ.Tuple[NDArray, ...],  # right_tucker_cores
        ],
):
    """Unstack stacked T3Basis into an array-like tree.
    """
    num_stacking_axes = len(x[0][0].shape[:-2])
    axes = tuple(range(num_stacking_axes))
    return stacking.unstack(x, axes=axes)


def t3b_stack(
        xx, # Array-like tree of bases
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # up_tucker_cores
    typ.Tuple[NDArray, ...],  # down_tucker_cores
    typ.Tuple[NDArray, ...],  # left_tt_cores
    typ.Tuple[NDArray, ...],  # right_tucker_cores
]:
    """Stack array-like tree of T3 bases into a single T3 basis
    """
    xnp,_,_ = get_backend(False, use_jax)

    num_stacking_axes = stacking.tree_depth(xx) - 2
    stacking_axes = tuple(range(num_stacking_axes))
    return stacking.stack(xx, axes=stacking_axes)


