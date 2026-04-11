# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import numpy as np
import jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3

__all__ = [
    # Tucker tensor train
    'TuckerTensorTrain',
    'T3Structure',
    'get_structure',
    't3_apply',
    't3_entry',
    't3_to_dense',
    'squash_tails',
    'reverse_t3',
    'check_t3',
    't3_zeros',
    't3_corewise_randn',
    'compute_minimal_ranks',
    'are_t3_ranks_minimal',
    'pad_t3',
    't3_save',
    't3_load',
    # Linear algebra
    't3_add',
    't3_scale',
    't3_neg',
    't3_sub',
    't3_dot_t3',
    't3_norm',
]

def randn(*args):
    return jnp.array(np.random.randn(*args)) # should convert this to pure jax

TuckerTensorTrain       = t3.TuckerTensorTrain
T3Structure             = t3.T3Structure
get_structure           = t3.get_structure
reverse_t3              = t3.reverse_t3
check_t3                = t3.check_t3
compute_minimal_ranks   = t3.compute_minimal_ranks
are_t3_ranks_minimal    = t3.are_t3_ranks_minimal
t3_save                 = t3.t3_save
t3_load                 = t3.t3_load
t3_scale                = t3.t3_scale
t3_neg                  = t3.t3_neg

t3_apply            = ft.partial(t3.t3_apply, xnp=jnp)
t3_entry            = ft.partial(t3.t3_entry, xnp=jnp)
t3_to_dense         = ft.partial(t3.t3_to_dense, xnp=jnp)
squash_tails        = ft.partial(t3.squash_tails, xnp=jnp)
t3_zeros            = ft.partial(t3.t3_zeros, xnp=jnp)
t3_corewise_randn   = ft.partial(t3.t3_corewise_randn, randn=randn)
pad_t3              = ft.partial(t3.change_structure, xnp=jnp)
t3_add              = ft.partial(t3.t3_add, xnp=jnp)
t3_sub              = ft.partial(t3.t3_sub, xnp=jnp)
t3_dot_t3           = ft.partial(t3.t3_inner_product_t3, xnp=jnp)
t3_norm             = ft.partial(t3.t3_norm, xnp=jnp)
