# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import jax.numpy as jnp
import t3tools.corewise as cw

__all__ = [
    'NDArrayTree',
    'corewise_add',
    'corewise_sub',
    'corewise_scale',
    'corewise_neg',
    'corewise_dot',
    'corewise_norm',
]

NDArrayTree     = cw.NDArrayTree
corewise_add    = cw.corewise_add
corewise_sub    = cw.corewise_sub
corewise_scale  = cw.corewise_scale
corewise_neg    = cw.corewise_neg

corewise_dot    = ft.partial(cw.corewise_dot, xnp=jnp)
corewise_norm   = ft.partial(cw.corewise_norm, xnp=jnp)
