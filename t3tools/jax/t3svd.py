# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import jax.numpy as jnp
import t3tools.t3svd as t3svd

__all__ = [
    't3_svd',
    'tucker_svd_dense',
    'tt_svd_dense',
    't3_svd_dense',
]

t3_svd              = ft.partial(t3svd.t3_svd, xnp=jnp)
tucker_svd_dense    = ft.partial(t3svd.tucker_svd_dense, xnp=jnp)
tt_svd_dense        = ft.partial(t3svd.tt_svd_dense, xnp=jnp)
t3_svd_dense        = ft.partial(t3svd.t3_svd_dense, xnp=jnp)
