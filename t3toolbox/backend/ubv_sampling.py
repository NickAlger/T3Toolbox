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
    'ut3tangent_precompute_apply_base_sweep',
    'ut3tangent_precompute_probe_base_sweep',
    'ut3tangent_precompute_entries_base_sweep',
    'ut3tangent_apply_jacobian_from_sweep',
    'ut3tangent_probe_jacobian_from_sweep',
    'ut3tangent_entries_jacobian_from_sweep',
    'ut3tangent_apply_transpose_from_sweep',
    'ut3tangent_probe_transpose_from_sweep',
    'ut3tangent_entries_transpose_from_sweep',
    'ut3tangent_precompute_apply_base_sweep_jets',
    'ut3tangent_precompute_probe_base_sweep_jets',
    'ut3tangent_precompute_entries_base_sweep_jets',
    'ut3tangent_apply_jacobian_derivatives_from_sweep',
    'ut3tangent_probe_jacobian_derivatives_from_sweep',
    'ut3tangent_entries_jacobian_derivatives_from_sweep',
    'ut3tangent_apply_transpose_derivatives_from_sweep',
    'ut3tangent_probe_transpose_derivatives_from_sweep',
    'ut3tangent_entries_transpose_derivatives_from_sweep',
    'ut3tangent_probe_derivatives',
    'ut3tangent_apply_derivatives',
    'ut3tangent_entries_derivatives',
    'ut3tangent_probe_derivatives_transpose',
    'ut3tangent_apply_derivatives_transpose',
    'ut3tangent_entries_derivatives_transpose',
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
):  # -> MIRRORS ww: ragged -> len=d tuple (W+K+C+(Ni,)); packed -> (d,)+W+K+C+(N,)
    """Probe a uniform tangent vector (the bare ``𝒥``): mask-once, share ``probing.probe_tangent``.
    **Mirrors** ``ww``'s packedness (ragged -> the d probes sliced to real widths; packed -> the packed array)."""
    mv, mb = _mask_once(basis_data, variations_data)
    ragged_in = not ut3_operations.is_packed(ww)
    packed = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])      # zero-pad to the supercore mode dim N
    zz = probing.probe_tangent(packed, mv, mb)                     # packed; shape=(d,)+W+K+C+(N,)
    return ut3_operations.unpack_vectors(zz, basis_data[4]) if ragged_in else zz


def ut3tangent_apply(
        ww,               # apply vectors, len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data
        variations_data,  # UT3Variations .data
):  # -> scalar apply, shape=W+K+C
    """Apply a uniform tangent vector in all modes (the bare ``𝒥``; a scalar per stack element)."""
    mv, mb = _mask_once(basis_data, variations_data)
    packed = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])
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


# ----------------------------------------------------------------- the split precompute -> from_sweep seam
# The SamplingKind hooks for the uniform Gauss-Newton fitting model (optimizers-on-uniform U3), the uniform
# twins of probing.{precompute_*_base_sweep, *_jacobian_from_sweep, *_transpose_from_sweep}. The base sweep
# is the expensive, W-scaled part (the environment tensors); precomputed ONCE per base and reused across
# every J / Jᵀ of an inner solve (e.g. Newton-CG). Here the uniform sweep additionally carries the
# boundary-processed operands -- the mask-once basis (a bare-supercore 4-tuple) + the packed probe vectors
# (or the raw index) + the static shape -- so the from_sweep half re-masks / re-packs NOTHING per matvec:
#
#     uniform_sweep = (masked_base_4tuple, packed_ww_or_index, shape, probing_base_sweep)
#
# The from_sweep halves mask-once their variation / residual operands and call the polymorphic probing
# from_sweep functions. These are the bare 𝒥 / 𝒥ᵀ (no gauge projector Π); the geometry supplies Π.

def ut3tangent_precompute_apply_base_sweep(
        basis_data,  # UT3Basis .data (an orthogonal frame), supercore stack = C
        ww,          # apply vectors, len=d, ith elm_shape=W+(Ni,)
):  # -> uniform apply base sweep: (masked_base, packed_ww, shape, (xis, mus))
    """The all-modes **apply** base sweep for the uniform tangent Jacobian: mask-once the basis, pack ``ww``
    to the supercore mode width ``N``, and precompute the lean ``(xis, mus)`` via the polymorphic
    :py:func:`probing.precompute_apply_base_sweep`. The reuse hook for the uniform fitting inner solve."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    packed_ww = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])
    return (mb, packed_ww, basis_data[4], probing.precompute_apply_base_sweep(mb, packed_ww))


def ut3tangent_precompute_probe_base_sweep(
        basis_data,  # UT3Basis .data (orthogonal frame)
        ww,          # probe vectors, len=d, ith elm_shape=W+(Ni,)
):  # -> uniform probe base sweep: (masked_base, packed_ww, shape, (xis, mus, nus, etas))
    """The **probe** base sweep (full ``(xis, mus, nus, etas)``): mask-once, pack ``ww``, share
    :py:func:`probing.precompute_probe_base_sweep`."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    packed_ww = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])
    return (mb, packed_ww, basis_data[4], probing.precompute_probe_base_sweep(mb, packed_ww))


def ut3tangent_precompute_entries_base_sweep(
        basis_data,  # UT3Basis .data (orthogonal frame)
        index,       # int array, shape=(d,)+W -- the grid points (no packing; the real block is the prefix)
):  # -> uniform entries base sweep: (masked_base, index, shape, (xis, mus))
    """The all-modes **entries** base sweep (lean ``(xis, mus)``): mask-once, keep ``index`` (fiber slicing
    needs no packing), share :py:func:`probing.precompute_entries_base_sweep`."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    return (mb, index, basis_data[4], probing.precompute_entries_base_sweep(mb, index))


def ut3tangent_apply_jacobian_from_sweep(
        variations_data,  # UT3Variations .data (the tangent direction); stack = K + C
        base_sweep,       # = ut3tangent_precompute_apply_base_sweep(...)
):  # -> scalar apply, shape=W+K+C
    """Forward all-modes apply of a uniform tangent reusing the base sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`probing.apply_jacobian_from_sweep`."""
    mv = ubv_masking.apply_variations_masks(variations_data)
    mb, packed_ww, _shape, psweep = base_sweep
    return probing.apply_jacobian_from_sweep(mv, packed_ww, mb, psweep)


def ut3tangent_probe_jacobian_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        base_sweep,       # = ut3tangent_precompute_probe_base_sweep(...)
):  # -> PACKED probes, shape=(d,)+W+K+C+(N,)  (the inner-loop path is packed; no unpack)
    """Forward probe of a uniform tangent reusing the base sweep (the bare ``𝒥``): mask-once the variation,
    share :py:func:`probing.probe_jacobian_from_sweep`. **Packed** output (the split-seam is the optimizer's
    inner-loop path, which stays packed; use :py:func:`ut3tangent_probe` or ``unpack_vectors`` for ragged)."""
    mv = ubv_masking.apply_variations_masks(variations_data)
    mb, packed_ww, shape, psweep = base_sweep
    return probing.probe_jacobian_from_sweep(mv, packed_ww, mb, psweep)


def ut3tangent_entries_jacobian_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        base_sweep,       # = ut3tangent_precompute_entries_base_sweep(...)
):  # -> scalar entries, shape=W+K+C
    """Forward all-modes entries of a uniform tangent reusing the base sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`probing.entries_jacobian_from_sweep`."""
    mv = ubv_masking.apply_variations_masks(variations_data)
    mb, index, _shape, psweep = base_sweep
    return probing.entries_jacobian_from_sweep(mv, index, mb, psweep)


def ut3tangent_apply_transpose_from_sweep(
        residual,     # apply residual, shape=W+K+C (a scalar per stack element)
        base_sweep,   # = ut3tangent_precompute_apply_base_sweep(...)
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the all-modes apply reusing the base sweep (the bare adjoint; no gauge ``Π`` --
    the geometry projects). Shares :py:func:`probing.apply_transpose_from_sweep`; returns the bare gradient
    supercores ``(dU_tilde, dG_tilde)`` (the caller's ``project`` attaches the variation masks)."""
    mb, packed_ww, _shape, psweep = base_sweep
    return probing.apply_transpose_from_sweep(residual, packed_ww, mb, psweep, sum_over_probes=sum_over_probes)


def ut3tangent_probe_transpose_from_sweep(
        ztildes,      # probe residuals, len=d, ith elm_shape=W+K+C+(Ni,)
        base_sweep,   # = ut3tangent_precompute_probe_base_sweep(...)
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the probe reusing the base sweep (the bare adjoint): pack the residuals to ``N``,
    share :py:func:`probing.probe_transpose_from_sweep`. Returns the bare gradient supercores."""
    mb, packed_ww, _shape, psweep = base_sweep
    packed_z = ut3_operations.pack_if_ragged(ztildes, mb[0].shape[-1])
    return probing.probe_transpose_from_sweep(packed_z, packed_ww, mb, psweep, sum_over_probes=sum_over_probes)


def ut3tangent_entries_transpose_from_sweep(
        residual,     # entries residual, shape=W+K+C
        base_sweep,   # = ut3tangent_precompute_entries_base_sweep(...)
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the all-modes entries reusing the base sweep (scatter ``residual`` at the sweep's
    ``index``). Shares :py:func:`probing.entries_transpose_from_sweep`; returns the bare gradient supercores."""
    mb, index, _shape, psweep = base_sweep
    return probing.entries_transpose_from_sweep(residual, index, mb, psweep, sum_over_probes=sum_over_probes)


# ----------------------------------------------------------------- the split seam, DERIVATIVE (jet) twins
# The jet-ified split-seam hooks for the uniform derivative Gauss-Newton fitting model (optimizers-on-
# uniform U3'), the uniform twins of probe_derivatives.{precompute_*_base_sweep_jets,
# *_jacobian_derivatives_from_sweep, *_transpose_derivatives_from_sweep}. Same recipe as the plain seam:
# the uniform sweep carries the mask-once base + the packed X (`ww`) AND perturbation (`pp`) vectors +
# the static shape + the jet base sweep, so from_sweep re-masks/re-packs nothing per matvec:
#
#     uniform_jet_sweep = (masked_base_4tuple, packed_ww_or_index, packed_pp, shape, probing_jet_sweep)
#
# `order` (highest derivative order) threads through; the per-order residual weight omega lives in the
# geometry-agnostic SamplingKind (sumsq / transpose), not here. The bare 𝒥 / 𝒥ᵀ (no gauge projector Π).

def ut3tangent_precompute_apply_base_sweep_jets(
        basis_data,  # UT3Basis .data (an orthogonal frame)
        ww,          # apply vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp,          # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        order,       # highest derivative order
):  # -> uniform apply jet sweep: (masked_base, packed_ww, packed_pp, shape, (xi_jets, mu_jets))
    """The **apply**-derivative base sweep: mask-once the basis, pack ``ww`` and ``pp`` to ``N``, share
    :py:func:`probe_derivatives.precompute_apply_base_sweep_jets`."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    N = mb[0].shape[-1]
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    return (mb, packed_ww, packed_pp, basis_data[4],
            probe_derivatives.precompute_apply_base_sweep_jets(mb, packed_ww, packed_pp, order))


def ut3tangent_precompute_probe_base_sweep_jets(
        basis_data,  # UT3Basis .data (orthogonal frame)
        ww,          # probe vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp,          # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        order,       # highest derivative order
):  # -> uniform probe jet sweep: (masked_base, packed_ww, packed_pp, shape, (xi, mu, nu, eta jets))
    """The **probe**-derivative base sweep (full jets): mask-once, pack ``ww``/``pp``, share
    :py:func:`probe_derivatives.precompute_probe_base_sweep_jets`."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    N = mb[0].shape[-1]
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    return (mb, packed_ww, packed_pp, basis_data[4],
            probe_derivatives.precompute_probe_base_sweep_jets(mb, packed_ww, packed_pp, order))


def ut3tangent_precompute_entries_base_sweep_jets(
        basis_data,  # UT3Basis .data (orthogonal frame)
        index,       # int array, shape=(d,)+W -- the grid points (no packing)
        pp,          # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        order,       # highest derivative order
):  # -> uniform entries jet sweep: (masked_base, index, packed_pp, shape, (xi_jets, mu_jets))
    """The **entries**-derivative base sweep: mask-once, pack only ``pp`` (fiber slicing needs no ``index``
    packing), share :py:func:`probe_derivatives.precompute_entries_base_sweep_jets`."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    packed_pp = ut3_operations.pack_if_ragged(pp, mb[0].shape[-1])
    return (mb, index, packed_pp, basis_data[4],
            probe_derivatives.precompute_entries_base_sweep_jets(mb, index, packed_pp, order))


def ut3tangent_apply_jacobian_derivatives_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        base_sweep,       # = ut3tangent_precompute_apply_base_sweep_jets(...)
        order,            # highest derivative order
):  # -> scalar jets, shape=(order+1,)+W+K+C
    """Forward apply-derivatives of a uniform tangent reusing the jet sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`probe_derivatives.apply_jacobian_derivatives_from_sweep`."""
    mv = ubv_masking.apply_variations_masks(variations_data)
    mb, packed_ww, packed_pp, _shape, psweep = base_sweep
    return probe_derivatives.apply_jacobian_derivatives_from_sweep(mv, packed_ww, packed_pp, mb, psweep, order)


def ut3tangent_probe_jacobian_derivatives_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        base_sweep,       # = ut3tangent_precompute_probe_base_sweep_jets(...)
        order,            # highest derivative order
):  # -> PACKED probe jets, shape=(d,)+(order+1,)+W+K+C+(N,)  (packed inner-loop path; no unpack)
    """Forward probe-derivatives of a uniform tangent reusing the jet sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`probe_derivatives.probe_jacobian_derivatives_from_sweep`. **Packed** output
    (the split-seam inner-loop path stays packed)."""
    mv = ubv_masking.apply_variations_masks(variations_data)
    mb, packed_ww, packed_pp, shape, psweep = base_sweep
    return probe_derivatives.probe_jacobian_derivatives_from_sweep(mv, packed_ww, packed_pp, mb, psweep, order)


def ut3tangent_entries_jacobian_derivatives_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        base_sweep,       # = ut3tangent_precompute_entries_base_sweep_jets(...)
        order,            # highest derivative order
):  # -> scalar jets, shape=(order+1,)+W+K+C
    """Forward entries-derivatives of a uniform tangent reusing the jet sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`probe_derivatives.entries_jacobian_derivatives_from_sweep`."""
    mv = ubv_masking.apply_variations_masks(variations_data)
    mb, index, packed_pp, _shape, psweep = base_sweep
    return probe_derivatives.entries_jacobian_derivatives_from_sweep(mv, index, packed_pp, mb, psweep, order)


def ut3tangent_apply_transpose_derivatives_from_sweep(
        residual,     # apply residual jet (scalar), shape=(order+1,)+W+K+C
        base_sweep,   # = ut3tangent_precompute_apply_base_sweep_jets(...)
        order,        # highest derivative order
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the apply-derivatives reusing the jet sweep (the bare adjoint; sums the order
    axis in assembly). Shares :py:func:`probe_derivatives.apply_transpose_derivatives_from_sweep`."""
    mb, packed_ww, packed_pp, _shape, psweep = base_sweep
    return probe_derivatives.apply_transpose_derivatives_from_sweep(
        residual, packed_ww, packed_pp, mb, psweep, order, sum_over_probes=sum_over_probes)


def ut3tangent_probe_transpose_derivatives_from_sweep(
        ztildes,      # probe residual jets, len=d, ith elm_shape=(order+1,)+W+K+C+(Ni,)
        base_sweep,   # = ut3tangent_precompute_probe_base_sweep_jets(...)
        order,        # highest derivative order
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the probe-derivatives reusing the jet sweep (the bare adjoint): pack the residual
    jets to ``N``, share :py:func:`probe_derivatives.probe_transpose_derivatives_from_sweep`."""
    mb, packed_ww, packed_pp, _shape, psweep = base_sweep
    packed_z = ut3_operations.pack_if_ragged(ztildes, mb[0].shape[-1])
    return probe_derivatives.probe_transpose_derivatives_from_sweep(
        packed_z, packed_ww, packed_pp, mb, psweep, order, sum_over_probes=sum_over_probes)


def ut3tangent_entries_transpose_derivatives_from_sweep(
        residual,     # entries residual jet (scalar), shape=(order+1,)+W+K+C
        base_sweep,   # = ut3tangent_precompute_entries_base_sweep_jets(...)
        order,        # highest derivative order
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the entries-derivatives reusing the jet sweep (scatter at the sweep's ``index``).
    Shares :py:func:`probe_derivatives.entries_transpose_derivatives_from_sweep`."""
    mb, index, packed_pp, _shape, psweep = base_sweep
    return probe_derivatives.entries_transpose_derivatives_from_sweep(
        residual, index, packed_pp, mb, psweep, order, sum_over_probes=sum_over_probes)


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
):  # -> MIRRORS ww: ragged -> len=d tuple ((order+1,)+W+K+C+(Ni,)); packed -> (d,)+(order+1,)+W+K+C+(N,)
    """Symmetric probe derivatives of a uniform tangent vector (the bare ``𝒥``): mask-once, share
    ``probe_derivatives.probe_tangent_derivatives``. **Mirrors** ``ww``'s packedness."""
    mv, mb = _mask_once(basis_data, variations_data)
    N = mb[0].shape[-1]
    ragged_in = not ut3_operations.is_packed(ww)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    zz = probe_derivatives.probe_tangent_derivatives(packed_ww, packed_pp, mv, mb, order)
    return ut3_operations.unpack_vectors(zz, basis_data[4]) if ragged_in else zz


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
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
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
    packed_pp = ut3_operations.pack_if_ragged(pp, mb[0].shape[-1])
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
    packed_z = ut3_operations.pack_if_ragged(ztildes, N)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
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
    packed_ww = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])
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


# ----------------------------------------------------------------- the transpose 𝒥ᵀ derivatives (jets; 3b-6'c)
# The jet-ified twins of the transpose wrappers above: back-project residual JETS (which carry the leading
# order axis) into a SINGLE variation gradient (the transpose sums the order axis in the assembly, so the
# output has no order axis -- structurally identical to the plain transpose). Mask-once, pack ww AND pp
# (entries: pp only), share probe_derivatives.*_tangent_derivatives_transpose, attach the gauge masks over
# the new tangent stack K_new (the same _gauge_masks_over_Knew as the plain transpose). The bare 𝒥ᵀ.

def ut3tangent_probe_derivatives_transpose(
        ztildes,          # probe residual jets, len=d, ith elm_shape=(order+1,)+W+K+C+(Ni,)
        ww,               # probe vectors,   len=d, ith elm_shape=W+(Ni,)
        pp,               # perturbation P,  len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data (an orthogonal frame), supercore stack = C
        order,            # highest derivative order
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the probe-derivative map to residual jets (the bare adjoint). Mask-once,
    pack the residual jets + ``ww`` + ``pp``, share ``probe_derivatives.probe_tangent_derivatives_transpose``,
    attach the gauge masks over the new tangent stack ``K_new``."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    N = mb[0].shape[-1]
    packed_z = ut3_operations.pack_if_ragged(ztildes, N)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    dU_tilde, dG_tilde = probe_derivatives.probe_tangent_derivatives_transpose(
        packed_z, packed_ww, packed_pp, mb, order, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(basis_data, dU_tilde)
    return (dU_tilde, dG_tilde, basis_data[4], masks)


def ut3tangent_apply_derivatives_transpose(
        c,                # residual jet (scalar), shape=(order+1,)+W+K+C
        ww,               # apply vectors,  len=d, ith elm_shape=W+(Ni,)
        pp,               # perturbation P, len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data (orthogonal frame), stack=C
        order,            # highest derivative order
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the all-modes apply-derivative (the bare adjoint-state method)."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    N = mb[0].shape[-1]
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    dU_tilde, dG_tilde = probe_derivatives.apply_tangent_derivatives_transpose(
        c, packed_ww, packed_pp, mb, order, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(basis_data, dU_tilde)
    return (dU_tilde, dG_tilde, basis_data[4], masks)


def ut3tangent_entries_derivatives_transpose(
        c,                # residual jet (scalar), shape=(order+1,)+W+K+C
        index,            # int array, shape=(d,)+W
        pp,               # perturbation P, len=d, ith elm_shape=W+(Ni,)
        basis_data,       # UT3Basis .data (orthogonal frame), stack=C
        order,            # highest derivative order
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the all-modes entry-derivative (scatter ``c`` at ``index``; ``pp`` packed,
    ``index`` unpacked -- the one-hots are built packed inside ``probing._onehot_vectors``)."""
    mb = ubv_masking.apply_basis_masks(basis_data)
    packed_pp = ut3_operations.pack_if_ragged(pp, mb[0].shape[-1])
    dU_tilde, dG_tilde = probe_derivatives.entries_tangent_derivatives_transpose(
        c, index, packed_pp, mb, order, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(basis_data, dU_tilde)
    return (dU_tilde, dG_tilde, basis_data[4], masks)
