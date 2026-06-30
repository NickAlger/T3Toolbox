# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform tangent sampling (UT3Tangent probe / apply / entries), uniform-fix 3b-6b.

The uniform mirror of the ragged ``T3Tangent`` sampling frontend, sharing the polymorphic
``probing.{probe,apply,entries}_tangent``. The boundary work is the same as the plain layer's
``ut3_sampling`` (the precedent), lifted to the basis-variations pair:

1. **mask-once** the basis (4) and variation (2) supercores -- so the padded rank slots are zero and the
   core algorithm runs mask-free (the padded bonds contract to zero);
2. **pack** the probe vectors into one supercore-shaped tensor (``apply``/``probe``; ``entries`` slices
   fibers, so its integer ``index`` needs no packing);
3. call the polymorphic backend (``is_uniform`` is inferred from the masked supercores being bare ndarrays);
4. **unpack** the ``probe`` output back to ragged-width vectors (``apply``/``entries`` return a scalar --
   nothing to unpack).

These are the bare Jacobian ``𝒥`` (no gauge projector ``Π``); the transpose ``𝒥ᵀ`` lands in 3b-6c.
"""
import t3toolbox.backend.probing as probing
import t3toolbox.backend.ubv_masking as ubv_masking
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import *

__all__ = [
    'ut3tangent_probe',
    'ut3tangent_apply',
    'ut3tangent_entries',
]


def _mask_once(
        basis_data,       # UT3Basis .data:      (up, down, left, right, shape, masks)
        variations_data,  # UT3Variations .data: (tkv, ttv, shape, masks)
):  # -> (masked variation supercores (tkv, ttv), masked basis supercores (up, down, left, right))
    """Apply the rank masks once up front; return the masked supercores in probing's operand order
    (variation, base) -- the padded slots are now zero, so the shared probing algorithm runs mask-free."""
    return (ubv_masking.apply_variations_masks(variations_data),
            ubv_masking.apply_basis_masks(basis_data))


def ut3tangent_probe(
        ww,               # probe vectors, len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data
        variations_data,  # UT3Variations .data
):  # -> len=d tuple, ith elm_shape=W+K+C+(Ni,)  (the d probes, one free mode each)
    """Probe a uniform tangent vector (the bare ``𝒥``): mask-once, pack ``ww``, share
    ``probing.probe_tangent``, unpack the d probes back to ragged widths."""
    mv, mb = _mask_once(basis_data, variations_data)
    packed = ut3_operations.pack_vectors(ww, mb[0].shape[-1])      # zero-pad to the supercore mode dim N
    zz = probing.probe_tangent(packed, mv, mb)                     # packed; shape=(d,)+W+K+C+(N,)
    return ut3_operations.unpack_vectors(zz, basis_data[4])        # basis_data[4] = the static shape tuple


def ut3tangent_apply(
        ww,               # apply vectors, len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data
        variations_data,  # UT3Variations .data
):  # -> scalar apply, shape=W+K+C
    """Apply a uniform tangent vector in all modes (the bare ``𝒥``; a scalar per stack element)."""
    mv, mb = _mask_once(basis_data, variations_data)
    packed = ut3_operations.pack_vectors(ww, mb[0].shape[-1])
    return probing.apply_tangent(packed, mv, mb)


def ut3tangent_entries(
        index,            # int array, shape=(d,)+W -- the (stack W of) multi-indices
        basis_data,       # UT3Basis .data
        variations_data,  # UT3Variations .data
):  # -> scalar entries, shape=W+K+C
    """Entries of a uniform tangent vector at ``index`` (the bare ``𝒥``; fiber slicing, no packing -- the
    real indices ``index_i < Ni`` never reach the mode padding)."""
    mv, mb = _mask_once(basis_data, variations_data)
    return probing.entries_tangent(index, mv, mb)
