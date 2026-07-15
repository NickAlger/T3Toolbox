# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The plain-ut3 mask layer: build, validate, apply.

``ut3_make_masks`` and ``ut3_apply_masks``. (The jit tracer guard ``require_concrete_masks`` is
infrastructure and lives in ``common``, beside the ``ValueHashedMasks`` mixin -- the two halves of the
uniform mask-representation contract, shared by every uniform object's masks.)
Masks are boolean prefix vectors, static structure, and ALWAYS host numpy (``np``, never
``xnp``) -- do not "fix" that (``docs/uniform_masks_vs_ranks.md``,
``docs/contributor/uniform_rank_masks_rationale.md``).
"""
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'ut3_make_masks',
    'ut3_apply_masks',
]


def ut3_make_masks(
        tucker_ranks:   NDArray,            # HOST int, shape=(d,)   + stack_shape
        tt_ranks:       NDArray,            # HOST int, shape=(d+1,) + stack_shape
        n:              int,                # padded Tucker rank,     n >= max(tucker_ranks)
        r:              int,                # padded TT rank,         r >= max(tt_ranks)
) -> typ.Tuple[
    NDArray,  # tucker_edge_mask, HOST bool, static, shape=(d,)   + stack_shape + (n,)
    NDArray,  # tt_edge_mask,     HOST bool, static, shape=(d+1,) + stack_shape + (r,)
]:
    """Build the prefix RANK edge masks for a uniform Tucker tensor train.

    Slot ``j`` of an edge is marked real iff ``j < rank`` -- the canonical prefix form
    (``docs/uniform_masks_vs_ranks.md``). The rank masks carry the stack, so each stack element may
    declare its own ranks (the variety -- see ``docs/uniform_ranks_and_varieties.md``). The physical
    ``shape`` is a separate static int tuple (not a mask), held alongside these in ``.data`` slot 2.

    Masks are STRUCTURE, so this builder emits **numpy (host)** arrays via ``np`` regardless of whether
    the supercores are jax -- the ``np.*`` here is intentional and jit-required (a jax mask becomes a
    tracer under jit and breaks the layer). See ``docs/contributor/uniform_pytree_composition.md``.
    """
    # prefix_mask is np (host), not xnp: masks are static structure -- a jax mask is a tracer under jit.
    tucker_edge_mask = prefix_mask(tucker_ranks, n)   # (d,)   + stack_shape + (n,)
    tt_edge_mask     = prefix_mask(tt_ranks, r)       # (d+1,) + stack_shape + (r,)

    return tucker_edge_mask, tt_edge_mask


def ut3_apply_masks(
        x: typ.Tuple[
            NDArray,             # tucker_supercore, shape=(d,)+stack_shape+(n,N)
            NDArray,             # tt_supercore,     shape=(d,)+stack_shape+(r,n,r)
            typ.Sequence[int],   # shape = (N0,...,N(d-1)), static int tuple
            typ.Tuple[
                NDArray,  # tucker_edge_mask, HOST bool, static
                NDArray,  # tt_edge_mask,     HOST bool, static
            ],
        ],
) -> typ.Tuple[
    NDArray,  # masked_tucker_supercore, shape=(d,)+stack_shape+(n,N)
    NDArray,  # masked_tt_supercore,     shape=(d,)+stack_shape+(r,n,r)
]:
    """Zero the padded ("garbage") regions of the supercores by multiplying through the edge masks.

    The mask chokepoint: every mask-using op masks on entry, so the guard here covers the whole layer.
    ``xnp.einsum`` on the supercore with the numpy mask as a constant operand is fine (jax promotes it).
    The physical ``shape_mask`` is reconstructed on the host from the static ``shape`` ints (``np``, never
    ``jnp`` -- a traced mask breaks the layer; see ``docs/contributor/uniform_pytree_composition.md``).
    """
    tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask) = x
    require_concrete_masks(tucker_edge_mask, tt_edge_mask)  # masks must be host, not traced
    use_jax = tree_contains_jax((tucker_supercore, tt_supercore))
    xnp, _, _ = get_backend(True, use_jax)

    N = tucker_supercore.shape[-1]
    shape_mask = prefix_mask(shape, N)  # (d, N) HOST bool, reconstructed from the static shape ints

    masked_tucker_supercore = xnp.einsum(
        'd...nN,d...n,dN->d...nN',
        tucker_supercore, tucker_edge_mask, shape_mask,
    )
    masked_tt_supercore = xnp.einsum(
        'd...lnr,d...l,d...n,d...r->d...lnr',
        tt_supercore, tt_edge_mask[:-1], tucker_edge_mask, tt_edge_mask[1:],
    )
    return masked_tucker_supercore, masked_tt_supercore
