# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import numpy as np
import jax.numpy as jnp
import t3toolbox.manifold as t3m

__all__ = [
    # Tangent vectors
    'manifold_dim',
    'tangent_to_dense',
    'tangent_to_t3',
    'tangent_zeros',
    'tangent_randn',
    # Projection and retraction
    'orthogonal_gauge_projection',
    'oblique_gauge_projection',
    'project_t3_onto_tangent_space',
    'retract',
]

manifold_dim = t3m.manifold_dim

def randn(*args):
    return jnp.array(np.random.randn(*args)) # should convert this to pure jax

tangent_to_dense                = ft.partial(t3m.tangent_to_dense, xnp=jnp)
tangent_to_t3                   = ft.partial(t3m.tangent_to_t3, xnp=jnp)
tangent_zeros                   = ft.partial(t3m.tangent_zeros, xnp=jnp)
tangent_randn                   = ft.partial(t3m.tangent_randn, randn=randn)
orthogonal_gauge_projection     = ft.partial(t3m.orthogonal_gauge_projection, use_jax=True)
oblique_gauge_projection        = ft.partial(t3m.oblique_gauge_projection, xnp=jnp)
project_t3_onto_tangent_space   = ft.partial(t3m.project_t3_onto_tangent_space, xnp=jnp)
retract                         = ft.partial(t3m.retract, xnp=jnp)
