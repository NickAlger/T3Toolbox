# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import jax.numpy as jnp
import t3toolbox.uniform as ut3

__all__ = [
    'UniformTuckerTensorTrainCores',
    'UniformTuckerTensorTrainMasks',
    'check_ut3',
    'get_padded_structure',
    'get_original_structure',
    'unpack',
    'apply_masks',
    't3_to_ut3',
    'ut3_to_t3',
    'ut3_to_dense',
    'are_ut3_ranks_minimal',
    'ut3_entry',
    # Linear algebra operations:
    'ut3_add',
    'ut3_scale',
    'ut3_neg',
    'ut3_sub',
]


UniformTuckerTensorTrainCores   = ut3.UniformTuckerTensorTrainCores
UniformTuckerTensorTrainMasks   = ut3.UniformEdgeWeights
check_ut3                       = ut3.check_ut3
get_padded_structure            = ut3.get_padded_structure
get_original_structure          = ut3.get_original_structure
are_ut3_ranks_minimal           = ut3.are_ut3_ranks_minimal

unpack          = ft.partial(ut3.unpack, xnp=jnp)
apply_masks     = ft.partial(ut3.apply_masks, xnp=jnp)
t3_to_ut3       = ft.partial(ut3.t3_to_ut3, xnp=jnp)
ut3_to_t3       = ft.partial(ut3.ut3_to_t3, xnp=jnp)
ut3_to_dense    = ft.partial(ut3.ut3_to_dense, xnp=jnp)
ut3_entry       = ft.partial(ut3.ut3_entry, xnp=jnp)
ut3_add         = ft.partial(ut3.ut3_add, xnp=jnp)
ut3_scale       = ft.partial(ut3.ut3_scale, xnp=jnp)
ut3_neg         = ft.partial(ut3.ut3_neg, xnp=jnp)
ut3_sub         = ft.partial(ut3.ut3_sub, xnp=jnp)
