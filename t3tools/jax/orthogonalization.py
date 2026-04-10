# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import jax.numpy as jnp
import t3tools.orthogonalization as orth

__all__ = [
    'up_svd_ith_basis_core',
    'left_svd_ith_tt_core',
    'right_svd_ith_tt_core',
    'up_svd_ith_tt_core',
    'down_svd_ith_tt_core',
    'orthogonalize_relative_to_ith_basis_core',
    'orthogonalize_relative_to_ith_tt_core',
    'orthogonal_representations',
]

up_svd_ith_basis_core                       = ft.partial(orth.up_svd_ith_basis_core, xnp=jnp)
left_svd_ith_tt_core                        = ft.partial(orth.left_svd_ith_tt_core, xnp=jnp)
right_svd_ith_tt_core                       = ft.partial(orth.right_svd_ith_tt_core, xnp=jnp)
up_svd_ith_tt_core                          = ft.partial(orth.up_svd_ith_tt_core, xnp=jnp)
down_svd_ith_tt_core                        = ft.partial(orth.down_svd_ith_tt_core, xnp=jnp)
orthogonalize_relative_to_ith_basis_core    = ft.partial(orth.orthogonalize_relative_to_ith_basis_core, xnp=jnp)
orthogonalize_relative_to_ith_tt_core       = ft.partial(orth.orthogonalize_relative_to_ith_tt_core, xnp=jnp)
orthogonal_representations                  = ft.partial(orth.orthogonal_representations, xnp=jnp)

