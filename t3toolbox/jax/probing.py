# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import functools as ft
import jax.numpy as jnp
import t3toolbox.probing as t3p

__all__ = [
    # Probe a dense tensor
    'probe_dense',
    # Probe a Tucker tensor train
    'probe_t3',
    'compute_xis',
    'compute_mus',
    'compute_nus',
    'compute_etas',
    'assemble_probes',
    # Probe a tangent vector
    'probe_tangent',
    'compute_dxis',
    'compute_sigmas',
    'compute_taus',
    'compute_detas',
    'assemble_tangent_probes',
    # Transpose of map from tangent vector to probes
    'compute_deta_tildes',
    'compute_tau_tildes',
    'compute_sigma_tildes',
    'compute_dxi_tildes',
    'assemble_basis_variations',
    'assemble_tt_variations',
    'probe_tangent_transpose',
]

probe_dense                 = ft.partial(t3p.probe_dense, xnp=jnp)
probe_t3                    = ft.partial(t3p.probe_t3, xnp=jnp)
compute_xis                 = ft.partial(t3p.compute_weighted_xis, xnp=jnp)
compute_mus                 = ft.partial(t3p.compute_weighted_mus, xnp=jnp)
compute_nus                 = ft.partial(t3p.compute_weighted_nus, xnp=jnp)
compute_etas                = ft.partial(t3p.compute_weighted_etas, xnp=jnp)
assemble_probes             = ft.partial(t3p.assemble_weighted_zs, xnp=jnp)
probe_tangent               = ft.partial(t3p.probe_tangent, xnp=jnp)
compute_dxis                = ft.partial(t3p.compute_dxis, xnp=jnp)
compute_sigmas              = ft.partial(t3p.compute_weighted_sigmas, xnp=jnp)
compute_taus                = ft.partial(t3p.compute_weighted_taus, xnp=jnp)
compute_detas               = ft.partial(t3p.compute_weighted_detas, xnp=jnp)
assemble_tangent_probes     = ft.partial(t3p.assemble_weighted_tangent_zs, xnp=jnp)
compute_deta_tildes         = ft.partial(t3p.compute_weighted_deta_tildes, xnp=jnp)
compute_tau_tildes          = ft.partial(t3p.compute_weighted_tau_tildes, xnp=jnp)
compute_sigma_tildes        = ft.partial(t3p.compute_weighted_sigma_tildes, xnp=jnp)
compute_dxi_tildes          = ft.partial(t3p.compute_weighted_dxi_tildes, xnp=jnp)
assemble_basis_variations   = ft.partial(t3p.assemble_tucker_variations, xnp=jnp)
assemble_tt_variations      = ft.partial(t3p.assemble_tt_variations, xnp=jnp)
probe_tangent_transpose     = ft.partial(t3p.probe_tangent_transpose, xnp=jnp)
