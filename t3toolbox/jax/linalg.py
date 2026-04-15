# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import jax.numpy as jnp
import t3toolbox.util_linalg as linalg

__all__ = [
    'truncated_svd',
    'left_svd_3tensor',
    'right_svd_3tensor',
    'outer_svd_3tensor',
]

truncated_svd       = ft.partial(linalg.truncated_svd, xnp=jnp)
left_svd_3tensor    = ft.partial(linalg.left_svd_3tensor, xnp=jnp)
right_svd_3tensor   = ft.partial(linalg.right_svd_3tensor, xnp=jnp)
outer_svd_3tensor   = ft.partial(linalg.outer_svd_3tensor, xnp=jnp)
