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
import numpy as np

import t3toolbox.backend.probing as probing
import t3toolbox.backend.probe_derivatives as probe_derivatives
import t3toolbox.backend.ubv_masking as ubv_masking
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import *

__all__ = [
    'ut3tangent_probe',
    'ut3tangent_apply',
    'ut3tangent_entries',
    'ut3tangent_probe_transpose',
    'ut3tangent_apply_transpose',
    'ut3tangent_entries_transpose',
    'ut3tangent_probe_derivatives',
    'ut3tangent_apply_derivatives',
    'ut3tangent_entries_derivatives',
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


# ----------------------------------------------------------------- derivative sampling (jets 𝒥; 3b-6'b)
# The symmetric-directional-derivative twins of ut3tangent_probe / apply / entries (the forward Riemannian
# Jacobian derivatives). Same boundary work: mask-once basis+variations, pack ww AND pp (entries slices
# fibers -> only pp packed), share probe_derivatives.*_tangent_derivatives, unpack the probe output (which
# carries a leading derivative-order axis; the middle axis rides through unpack_vectors' `...`). Output
# order 0 is the ordinary tangent sample. These are the bare 𝒥 (no gauge projector Π).

def ut3tangent_probe_derivatives(
        ww,               # probe vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp,               # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data
        variations_data,  # UT3Variations .data
        order,            # highest derivative order
):  # -> len=d tuple, ith elm_shape=(order+1,)+W+K+C+(Ni,)
    """Symmetric probe derivatives of a uniform tangent vector (the bare ``𝒥``): mask-once, pack ``ww``/``pp``,
    share ``probe_derivatives.probe_tangent_derivatives``, unpack the d probe jets."""
    mv, mb = _mask_once(basis_data, variations_data)
    N = mb[0].shape[-1]
    packed_ww = ut3_operations.pack_vectors(ww, N)
    packed_pp = ut3_operations.pack_vectors(pp, N)
    zz = probe_derivatives.probe_tangent_derivatives(packed_ww, packed_pp, mv, mb, order)
    return ut3_operations.unpack_vectors(zz, basis_data[4])


def ut3tangent_apply_derivatives(
        ww,               # apply vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp,               # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data
        variations_data,  # UT3Variations .data
        order,            # highest derivative order
):  # -> scalar jets, shape=(order+1,)+W+K+C
    """Symmetric all-modes apply derivatives of a uniform tangent vector (the bare ``𝒥``; a scalar jet per
    stack element)."""
    mv, mb = _mask_once(basis_data, variations_data)
    N = mb[0].shape[-1]
    packed_ww = ut3_operations.pack_vectors(ww, N)
    packed_pp = ut3_operations.pack_vectors(pp, N)
    return probe_derivatives.apply_tangent_derivatives(packed_ww, packed_pp, mv, mb, order)


def ut3tangent_entries_derivatives(
        index,            # int array, shape=(d,)+W -- the multi-indices
        pp,               # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data
        variations_data,  # UT3Variations .data
        order,            # highest derivative order
):  # -> scalar jets, shape=(order+1,)+W+K+C
    """Symmetric entry derivatives of a uniform tangent vector at ``index`` (the bare ``𝒥``; fiber slicing,
    so only ``pp`` is packed)."""
    mv, mb = _mask_once(basis_data, variations_data)
    packed_pp = ut3_operations.pack_vectors(pp, mb[0].shape[-1])
    return probe_derivatives.entries_tangent_derivatives(index, packed_pp, mv, mb, order)


# ----------------------------------------------------------------- the transpose 𝒥ᵀ (probe; 3b-6c)
def _gauge_masks_over_Knew(
        basis_data,      # UT3Basis .data
        out_supercore,   # a transpose-output variation supercore (d,)+K_new+C+(...), pins the stack
):  # -> the 4 variation rank masks, each (d,)+K_new+C+(size,)
    """The result tangent's variation masks: the basis's gauge-shifted variation masks
    ``(up, down, basis_left[:-1], basis_right[1:])`` broadcast (constant) over the new tangent stack
    ``K_new`` -- the leading stack of the transpose output minus the base stack ``C``. ``sum_over_probes``
    determines ``K_new`` (``W+K`` kept / ``K`` summed); we read it off the output supercore rather than
    re-deriving it. Masks are host numpy (static aux), so this stays on ``np``."""
    up_mask, down_mask, basis_left_mask, basis_right_mask = basis_data[5]
    gauge = (up_mask, down_mask, basis_left_mask[:-1], basis_right_mask[1:])  # length-d variation masks, stack C
    C = basis_data[0].shape[1:-2]                 # base stack C (up supercore is (d,)+C+(nU,N))
    out_stack = out_supercore.shape[1:-2]         # K_new + C
    K_new = out_stack[:len(out_stack) - len(C)]

    def b(m):  # (d,)+C+(size,) -> (d,)+K_new+C+(size,)
        return np.broadcast_to(m.reshape(m.shape[:1] + (1,) * len(K_new) + m.shape[1:]),
                               m.shape[:1] + K_new + m.shape[1:])

    return tuple(b(m) for m in gauge)


def ut3tangent_probe_transpose(
        ztildes,          # probe residuals, len=d, ith elm_shape=W+K+C+(Ni,)
        ww,               # probe vectors,   len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data (an orthogonal frame), supercore stack = C
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the probe map to residuals (the bare adjoint; no gauge projector).

    Mask-once the basis, pack both the residuals and the probe vectors, share
    ``probing.probe_tangent_transpose`` (which routes through the 3b-6a d-prefixed WKC contractions), and
    attach the result variation masks: the basis's gauge masks broadcast over the new tangent stack
    ``K_new`` (``W+K`` if ``sum_over_probes=False``, ``K`` if ``True``). The bare ``𝒥ᵀ``."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    N = mb[0].shape[-1]
    packed_z = ut3_operations.pack_vectors(ztildes, N)
    packed_ww = ut3_operations.pack_vectors(ww, N)
    dU_tilde, dG_tilde = probing.probe_tangent_transpose(
        packed_z, packed_ww, mb, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(basis_data, dU_tilde)
    return (dU_tilde, dG_tilde, basis_data[4], masks)


def ut3tangent_apply_transpose(
        c,                # residual, shape=W+K+C (a scalar per stack element)
        ww,               # apply vectors, len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data (orthogonal frame), stack=C
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the all-modes apply (the bare adjoint; the adjoint-state method).
    Mask-once the basis, pack ``ww``, share ``probing.apply_tangent_transpose``, attach the gauge masks
    over the new tangent stack ``K_new``."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    packed_ww = ut3_operations.pack_vectors(ww, mb[0].shape[-1])
    dU_tilde, dG_tilde = probing.apply_tangent_transpose(
        c, packed_ww, mb, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(basis_data, dU_tilde)
    return (dU_tilde, dG_tilde, basis_data[4], masks)


def ut3tangent_entries_transpose(
        c,                # residual, shape=W+K+C
        index,            # int array, shape=(d,)+W -- the indices c weights
        basis_data,       # UT3Basis .data (orthogonal frame), stack=C
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the all-modes entries (scatter ``c`` at ``index``). Like
    :py:func:`ut3tangent_apply_transpose` with the apply vectors replaced by the one-hot ``e_index`` (built
    packed inside ``probing._onehot_vectors``), so ``index`` needs no packing."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    dU_tilde, dG_tilde = probing.entries_tangent_transpose(
        c, index, mb, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(basis_data, dU_tilde)
    return (dU_tilde, dG_tilde, basis_data[4], masks)
