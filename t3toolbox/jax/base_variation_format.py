# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import jax.numpy as jnp
import t3toolbox.base_variation_format as bvf

__all__ = [
    'T3Base',
    'T3Variation',
    'base_hole_shapes',
    'ith_bv_to_t3',
]

T3Base              = bvf.T3Base
T3Variation         = bvf.T3Variation
base_hole_shapes         = bvf.get_base_hole_shapes
ith_bv_to_t3        = bvf.ith_bv_to_t3
