# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import jax.numpy as jnp
import t3tools.base_variation_format as bvf

__all__ = [
    'T3Base',
    'T3Variation',
    'check_t3base',
    'check_t3variation',
    'hole_shapes',
    'check_fit',
    'ith_bv_to_t3',
]

T3Base              = bvf.T3Base
T3Variation         = bvf.T3Variation
check_t3base        = bvf.check_t3base
check_t3variation   = bvf.check_t3variation
hole_shapes         = bvf.hole_shapes
check_fit           = bvf.check_fit

ith_bv_to_t3 = ft.partial(bvf.ith_bv_to_t3, xnp=jnp)
