# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *
import t3toolbox.backend.common as common

if common.jax_available:
    import jax  # only for the tracer-detection guard below (gated by common.jax_available)

__all__ = [
    'ut3_make_masks',
    'require_concrete_masks',
    'ut3_apply_masks',
]


def require_concrete_masks(
        *masks: NDArray,  # HOST bool, static -- the uniform structure masks (must NOT be traced)
) -> None:
    """Guard the uniform-mask contract: masks are concrete host (numpy) arrays, never jax tracers.

    Under jit any ``jnp`` op on a mask returns a tracer, which breaks the layer two ways: host-int
    shape/rank extraction (``int(mask.sum())``) raises ``ConcretizationTypeError``, and recomputed masks
    leak as tracers into the (identity-hashed, never-inspected) output ``aux_data`` -- silently invalid.
    So a traced mask here means the masks were passed *among* the traced jit args; the fix is functional,
    not numerical (raise, per the structural-vs-numerical philosophy). See
    ``docs/uniform_pytree_composition.md``.
    """
    if not common.jax_available:
        return
    for m in masks:
        if isinstance(m, jax.core.Tracer):
            raise ValueError(
                'uniform masks must be concrete host (numpy) arrays, but a traced mask was seen -- you '
                'likely jitted a backend function with the masks among the traced args. Close over the '
                'masks as constants and trace only the supercores (the masks are static structure). '
                'See docs/uniform_pytree_composition.md.')


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
    tracer under jit and breaks the layer). See ``docs/uniform_pytree_composition.md``.
    """
    # np (host), not xnp: masks are static structure -- a jax mask is a tracer under jit. Intentional.
    tucker_ranks = np.asarray(tucker_ranks) # (d,)   + stack_shape
    tt_ranks     = np.asarray(tt_ranks)     # (d+1,) + stack_shape

    tucker_edge_mask = np.arange(n) < tucker_ranks[..., None]   # (d,)   + stack_shape + (n,)
    tt_edge_mask     = np.arange(r) < tt_ranks[..., None]       # (d+1,) + stack_shape + (r,)

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
    ``jnp`` -- a traced mask breaks the layer; see ``docs/uniform_pytree_composition.md``).
    """
    tucker_supercore, tt_supercore, shape, (tucker_edge_mask, tt_edge_mask) = x
    require_concrete_masks(tucker_edge_mask, tt_edge_mask)  # masks must be host, not traced
    use_jax = tree_contains_jax((tucker_supercore, tt_supercore))
    xnp, _, _ = get_backend(True, use_jax)

    N = tucker_supercore.shape[-1]
    shape_mask = np.arange(N) < np.asarray(shape)[:, None]  # (d, N) HOST bool, reconstructed from ints

    masked_tucker_supercore = xnp.einsum(
        'd...nN,d...n,dN->d...nN',
        tucker_supercore, tucker_edge_mask, shape_mask,
    )
    masked_tt_supercore = xnp.einsum(
        'd...lnr,d...l,d...n,d...r->d...lnr',
        tt_supercore, tt_edge_mask[:-1], tucker_edge_mask, tt_edge_mask[1:],
    )
    return masked_tucker_supercore, masked_tt_supercore
