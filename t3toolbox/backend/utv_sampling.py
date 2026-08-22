# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform tangent sampling (UT3Tangent probe / apply / entries).

The uniform mirror of the ragged ``T3Tangent`` sampling frontend, sharing the polymorphic
``{probing.tv_probe, apply.tv_apply, entries.tv_entries}``. The boundary work is the same as the plain layer's
``ut3_sampling`` (the precedent), lifted to the frame-variations pair:

1. **mask-once** the frame (4) and variation (2) supercores -- so the padded rank slots are zero and the
   core algorithm runs mask-free (the padded bonds contract to zero);
2. **pack** the probe vectors into one supercore-shaped tensor (``apply``/``probe``; ``entries`` slices
   fibers, so its integer ``index`` needs no packing);
3. call the polymorphic backend (``is_uniform`` is inferred from the masked supercores being bare ndarrays);
4. **unpack** the ``probe`` output back to ragged-width vectors (``apply``/``entries`` return a scalar --
   nothing to unpack).

These are the bare Jacobian ``𝒥`` (no gauge projector ``Π``); the transpose ``𝒥ᵀ`` is the last section of this module.
"""
import numpy as np

import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.sampling_derivatives as sampling_derivatives
import t3toolbox.backend.ufv_masking as ufv_masking
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import *

__all__ = [
    'utv_probe',
    'utv_apply',
    'utv_entries',
    'utv_probe_transpose',
    'utv_apply_transpose',
    'utv_entries_transpose',
    'utv_precompute_apply_frame_sweep',
    'utv_precompute_probe_frame_sweep',
    'utv_precompute_entries_frame_sweep',
    'utv_apply_jacobian_from_sweep',
    'utv_probe_jacobian_from_sweep',
    'utv_entries_jacobian_from_sweep',
    'utv_apply_transpose_from_sweep',
    'utv_probe_transpose_from_sweep',
    'utv_entries_transpose_from_sweep',
    'utv_precompute_apply_frame_sweep_jets',
    'utv_precompute_probe_frame_sweep_jets',
    'utv_precompute_entries_frame_sweep_jets',
    'utv_apply_jacobian_derivatives_from_sweep',
    'utv_probe_jacobian_derivatives_from_sweep',
    'utv_entries_jacobian_derivatives_from_sweep',
    'utv_apply_transpose_derivatives_from_sweep',
    'utv_probe_transpose_derivatives_from_sweep',
    'utv_entries_transpose_derivatives_from_sweep',
    'utv_probe_derivatives',
    'utv_apply_derivatives',
    'utv_entries_derivatives',
    'utv_probe_derivatives_transpose',
    'utv_apply_derivatives_transpose',
    'utv_entries_derivatives_transpose',
]


def _mask_once(
        frame_data,       # UT3Frame .data:      (up, down, left, right, shape, masks)
        variations_data,  # UT3Variations .data: (tkv, ttv, shape, masks)
):  # -> (masked variation supercores (tkv, ttv), masked frame supercores (up, down, left, right))
    """Apply the rank masks once up front; return the masked supercores in probing's operand order
    (variation, frame) -- the padded slots are now zero, so the shared probing algorithm runs mask-free."""
    return (ufv_masking.ufv_apply_variations_masks(variations_data),
            ufv_masking.ufv_apply_frame_masks(frame_data))


def utv_probe(
        ww,               # probe vectors, len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data
        variations_data,  # UT3Variations .data
):  # -> MIRRORS ww: ragged -> len=d tuple (W+K+C+(Ni,)); packed -> (d,)+W+K+C+(N,)
    """Probe a uniform tangent vector (the bare ``𝒥``): mask-once, share ``probing.tv_probe``.
    **Mirrors** ``ww``'s packedness (ragged -> the d probes sliced to real widths; packed -> the packed array)."""
    mv, mb = _mask_once(frame_data, variations_data)
    ragged_in = not ut3_operations.is_packed(ww)
    packed = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])      # zero-pad to the supercore mode dim N
    zz = probing.tv_probe(packed, mv, mb)                     # packed; shape=(d,)+W+K+C+(N,)
    return ut3_operations.unpack_vectors(zz, frame_data[4]) if ragged_in else zz


def utv_apply(
        ww,               # apply vectors, len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data
        variations_data,  # UT3Variations .data
):  # -> scalar apply, shape=W+K+C
    """Apply a uniform tangent vector in all modes (the bare ``𝒥``; a scalar per stack element)."""
    mv, mb = _mask_once(frame_data, variations_data)
    packed = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])
    return apply.tv_apply(packed, mv, mb)


def utv_entries(
        index,            # int array, shape=(d,)+W -- the (stack W of) multi-indices
        frame_data,       # UT3Frame .data
        variations_data,  # UT3Variations .data
):  # -> scalar entries, shape=W+K+C
    """Entries of a uniform tangent vector at ``index`` (the bare ``𝒥``; fiber slicing, no packing -- the
    real indices ``index_i < Ni`` never reach the mode padding)."""
    mv, mb = _mask_once(frame_data, variations_data)
    return entries.tv_entries(index, mv, mb)


# ----------------------------------------------------------------- the split precompute -> from_sweep seam
# The SamplingKind hooks for the uniform Gauss-Newton fitting model (optimizers-on-uniform U3), the uniform
# twins of the ragged tv sweep machinery (tv_precompute_*_frame_sweep, tv_*_jacobian_from_sweep, tv_*_transpose_from_sweep in probing/apply/entries). The frame sweep
# is the expensive, W-scaled part (the environment tensors); precomputed ONCE per frame and reused across
# every J / Jᵀ of an inner solve (e.g. Newton-CG). Here the uniform sweep additionally carries the
# boundary-processed operands -- the mask-once frame (a bare-supercore 4-tuple) + the packed probe vectors
# (or the raw index) + the static shape -- so the from_sweep half re-masks / re-packs NOTHING per matvec:
#
#     uniform_sweep = (masked_frame_4tuple, packed_ww_or_index, shape, probing_frame_sweep)
#
# The from_sweep halves mask-once their variation / residual operands and call the polymorphic probing
# from_sweep functions. These are the bare 𝒥 / 𝒥ᵀ (no gauge projector Π); the geometry supplies Π.

def utv_precompute_apply_frame_sweep(
        frame_data,  # UT3Frame .data (an orthogonal frame), supercore stack = C
        ww,          # apply vectors, len=d, ith elm_shape=W+(Ni,)
):  # -> uniform apply frame sweep: (masked_frame, packed_ww, shape, (xis, mus))
    """The all-modes **apply** frame sweep for the uniform tangent Jacobian: mask-once the frame, pack ``ww``
    to the supercore mode width ``N``, and precompute the lean ``(xis, mus)`` via the polymorphic
    :py:func:`apply.tv_precompute_apply_frame_sweep`. The reuse hook for the uniform fitting inner solve."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    packed_ww = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])
    return (mb, packed_ww, frame_data[4], apply.tv_precompute_apply_frame_sweep(mb, packed_ww))


def utv_precompute_probe_frame_sweep(
        frame_data,  # UT3Frame .data (orthogonal frame)
        ww,          # probe vectors, len=d, ith elm_shape=W+(Ni,)
):  # -> uniform probe frame sweep: (masked_frame, packed_ww, shape, (xis, mus, nus, etas))
    """The **probe** frame sweep (full ``(xis, mus, nus, etas)``): mask-once, pack ``ww``, share
    :py:func:`probing.tv_precompute_probe_frame_sweep`."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    packed_ww = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])
    return (mb, packed_ww, frame_data[4], probing.tv_precompute_probe_frame_sweep(mb, packed_ww))


def utv_precompute_entries_frame_sweep(
        frame_data,  # UT3Frame .data (orthogonal frame)
        index,       # int array, shape=(d,)+W -- the grid points (no packing; the real block is the prefix)
):  # -> uniform entries frame sweep: (masked_frame, index, shape, (xis, mus))
    """The all-modes **entries** frame sweep (lean ``(xis, mus)``): mask-once, keep ``index`` (fiber slicing
    needs no packing), share :py:func:`entries.tv_precompute_entries_frame_sweep`."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    return (mb, index, frame_data[4], entries.tv_precompute_entries_frame_sweep(mb, index))


def utv_apply_jacobian_from_sweep(
        variations_data,  # UT3Variations .data (the tangent direction); stack = K + C
        frame_sweep,       # = utv_precompute_apply_frame_sweep(...)
):  # -> scalar apply, shape=W+K+C
    """Forward all-modes apply of a uniform tangent reusing the frame sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`apply.tv_apply_jacobian_from_sweep`."""
    mv = ufv_masking.ufv_apply_variations_masks(variations_data)
    mb, packed_ww, _shape, psweep = frame_sweep
    return apply.tv_apply_jacobian_from_sweep(mv, packed_ww, mb, psweep)


def utv_probe_jacobian_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        frame_sweep,       # = utv_precompute_probe_frame_sweep(...)
):  # -> PACKED probes, shape=(d,)+W+K+C+(N,)  (the inner-loop path is packed; no unpack)
    """Forward probe of a uniform tangent reusing the frame sweep (the bare ``𝒥``): mask-once the variation,
    share :py:func:`probing.tv_probe_jacobian_from_sweep`. **Packed** output (the split-seam is the optimizer's
    inner-loop path, which stays packed; use :py:func:`utv_probe` or ``unpack_vectors`` for ragged)."""
    mv = ufv_masking.ufv_apply_variations_masks(variations_data)
    mb, packed_ww, shape, psweep = frame_sweep
    return probing.tv_probe_jacobian_from_sweep(mv, packed_ww, mb, psweep)


def utv_entries_jacobian_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        frame_sweep,       # = utv_precompute_entries_frame_sweep(...)
):  # -> scalar entries, shape=W+K+C
    """Forward all-modes entries of a uniform tangent reusing the frame sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`entries.tv_entries_jacobian_from_sweep`."""
    mv = ufv_masking.ufv_apply_variations_masks(variations_data)
    mb, index, _shape, psweep = frame_sweep
    return entries.tv_entries_jacobian_from_sweep(mv, index, mb, psweep)


def utv_apply_transpose_from_sweep(
        residual,     # apply residual, shape=W+K+C (a scalar per stack element)
        frame_sweep,   # = utv_precompute_apply_frame_sweep(...)
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the all-modes apply reusing the frame sweep (the bare adjoint; no gauge ``Π`` --
    the geometry projects). Shares :py:func:`apply.tv_apply_transpose_from_sweep`; returns the bare gradient
    supercores ``(dU_tilde, dG_tilde)`` (the caller's ``project`` attaches the variation masks)."""
    mb, packed_ww, _shape, psweep = frame_sweep
    return apply.tv_apply_transpose_from_sweep(residual, packed_ww, mb, psweep, sum_over_probes=sum_over_probes)


def utv_probe_transpose_from_sweep(
        ztildes,      # probe residuals, len=d, ith elm_shape=W+K+C+(Ni,)
        frame_sweep,   # = utv_precompute_probe_frame_sweep(...)
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the probe reusing the frame sweep (the bare adjoint): pack the residuals to ``N``,
    share :py:func:`probing.tv_probe_transpose_from_sweep`. Returns the bare gradient supercores."""
    mb, packed_ww, _shape, psweep = frame_sweep
    packed_z = ut3_operations.pack_if_ragged(ztildes, mb[0].shape[-1])
    return probing.tv_probe_transpose_from_sweep(packed_z, packed_ww, mb, psweep, sum_over_probes=sum_over_probes)


def utv_entries_transpose_from_sweep(
        residual,     # entries residual, shape=W+K+C
        frame_sweep,   # = utv_precompute_entries_frame_sweep(...)
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the all-modes entries reusing the frame sweep (scatter ``residual`` at the sweep's
    ``index``). Shares :py:func:`entries.tv_entries_transpose_from_sweep`; returns the bare gradient supercores."""
    mb, index, _shape, psweep = frame_sweep
    return entries.tv_entries_transpose_from_sweep(residual, index, mb, psweep, sum_over_probes=sum_over_probes)


# ----------------------------------------------------------------- the split seam, DERIVATIVE (jet) twins
# The jet-ified split-seam hooks for the uniform derivative Gauss-Newton fitting model (optimizers-on-
# uniform U3'), the uniform twins of the ragged jet sweep machinery (sampling_derivatives: tv_precompute_*_frame_sweep_jets,
# *_jacobian_derivatives_from_sweep, *_transpose_derivatives_from_sweep). Same recipe as the plain seam:
# the uniform sweep carries the mask-once frame + the packed X (`ww`) AND perturbation (`pp`) vectors +
# the static shape + the jet frame sweep, so from_sweep re-masks/re-packs nothing per matvec:
#
#     uniform_jet_sweep = (masked_frame_4tuple, packed_ww_or_index, packed_pp, shape, probing_jet_sweep)
#
# `order` (highest derivative order) threads through; the per-order residual weight omega lives in the
# geometry-agnostic SamplingKind (sumsq / transpose), not here. The bare 𝒥 / 𝒥ᵀ (no gauge projector Π).

def utv_precompute_apply_frame_sweep_jets(
        frame_data,  # UT3Frame .data (an orthogonal frame)
        ww,          # apply vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp,          # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        order,       # highest derivative order
):  # -> uniform apply jet sweep: (masked_frame, packed_ww, packed_pp, shape, (xi_jets, mu_jets))
    """The **apply**-derivative frame sweep: mask-once the frame, pack ``ww`` and ``pp`` to ``N``, share
    :py:func:`sampling_derivatives.tv_precompute_apply_frame_sweep_jets`."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    N = mb[0].shape[-1]
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    return (mb, packed_ww, packed_pp, frame_data[4],
            sampling_derivatives.tv_precompute_apply_frame_sweep_jets(mb, packed_ww, packed_pp, order))


def utv_precompute_probe_frame_sweep_jets(
        frame_data,  # UT3Frame .data (orthogonal frame)
        ww,          # probe vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp,          # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        order,       # highest derivative order
):  # -> uniform probe jet sweep: (masked_frame, packed_ww, packed_pp, shape, (xi, mu, nu, eta jets))
    """The **probe**-derivative frame sweep (full jets): mask-once, pack ``ww``/``pp``, share
    :py:func:`sampling_derivatives.tv_precompute_probe_frame_sweep_jets`."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    N = mb[0].shape[-1]
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    return (mb, packed_ww, packed_pp, frame_data[4],
            sampling_derivatives.tv_precompute_probe_frame_sweep_jets(mb, packed_ww, packed_pp, order))


def utv_precompute_entries_frame_sweep_jets(
        frame_data,  # UT3Frame .data (orthogonal frame)
        index,       # int array, shape=(d,)+W -- the grid points (no packing)
        pp,          # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        order,       # highest derivative order
):  # -> uniform entries jet sweep: (masked_frame, index, packed_pp, shape, (xi_jets, mu_jets))
    """The **entries**-derivative frame sweep: mask-once, pack only ``pp`` (fiber slicing needs no ``index``
    packing), share :py:func:`sampling_derivatives.tv_precompute_entries_frame_sweep_jets`."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    packed_pp = ut3_operations.pack_if_ragged(pp, mb[0].shape[-1])
    return (mb, index, packed_pp, frame_data[4],
            sampling_derivatives.tv_precompute_entries_frame_sweep_jets(mb, index, packed_pp, order))


def utv_apply_jacobian_derivatives_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        frame_sweep,       # = utv_precompute_apply_frame_sweep_jets(...)
        order,            # highest derivative order
):  # -> scalar jets, shape=(order+1,)+W+K+C
    """Forward apply-derivatives of a uniform tangent reusing the jet sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`sampling_derivatives.tv_apply_jacobian_derivatives_from_sweep`."""
    mv = ufv_masking.ufv_apply_variations_masks(variations_data)
    mb, packed_ww, packed_pp, _shape, psweep = frame_sweep
    return sampling_derivatives.tv_apply_jacobian_derivatives_from_sweep(mv, packed_ww, packed_pp, mb, psweep, order)


def utv_probe_jacobian_derivatives_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        frame_sweep,       # = utv_precompute_probe_frame_sweep_jets(...)
        order,            # highest derivative order
):  # -> PACKED probe jets, shape=(d,)+(order+1,)+W+K+C+(N,)  (packed inner-loop path; no unpack)
    """Forward probe-derivatives of a uniform tangent reusing the jet sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`sampling_derivatives.tv_probe_jacobian_derivatives_from_sweep`. **Packed** output
    (the split-seam inner-loop path stays packed)."""
    mv = ufv_masking.ufv_apply_variations_masks(variations_data)
    mb, packed_ww, packed_pp, shape, psweep = frame_sweep
    return sampling_derivatives.tv_probe_jacobian_derivatives_from_sweep(mv, packed_ww, packed_pp, mb, psweep, order)


def utv_entries_jacobian_derivatives_from_sweep(
        variations_data,  # UT3Variations .data; stack = K + C
        frame_sweep,       # = utv_precompute_entries_frame_sweep_jets(...)
        order,            # highest derivative order
):  # -> scalar jets, shape=(order+1,)+W+K+C
    """Forward entries-derivatives of a uniform tangent reusing the jet sweep (the bare ``𝒥``): mask-once the
    variation, share :py:func:`sampling_derivatives.tv_entries_jacobian_derivatives_from_sweep`."""
    mv = ufv_masking.ufv_apply_variations_masks(variations_data)
    mb, index, packed_pp, _shape, psweep = frame_sweep
    return sampling_derivatives.tv_entries_jacobian_derivatives_from_sweep(mv, index, packed_pp, mb, psweep, order)


def utv_apply_transpose_derivatives_from_sweep(
        residual,     # apply residual jet (scalar), shape=(order+1,)+W+K+C
        frame_sweep,   # = utv_precompute_apply_frame_sweep_jets(...)
        order,        # highest derivative order
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the apply-derivatives reusing the jet sweep (the bare adjoint; sums the order
    axis in assembly). Shares :py:func:`sampling_derivatives.tv_apply_transpose_derivatives_from_sweep`."""
    mb, packed_ww, packed_pp, _shape, psweep = frame_sweep
    return sampling_derivatives.tv_apply_transpose_derivatives_from_sweep(
        residual, packed_ww, packed_pp, mb, psweep, order, sum_over_probes=sum_over_probes)


def utv_probe_transpose_derivatives_from_sweep(
        ztildes,      # probe residual jets, len=d, ith elm_shape=(order+1,)+W+K+C+(Ni,)
        frame_sweep,   # = utv_precompute_probe_frame_sweep_jets(...)
        order,        # highest derivative order
        sum_over_probes=True,
        chunk_size=100,   # W-chunk size for the gradient assembly; None -> dense. docs/chunking.md
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the probe-derivatives reusing the jet sweep (the bare adjoint): pack the residual
    jets to ``N``, share :py:func:`sampling_derivatives.tv_probe_transpose_derivatives_from_sweep`."""
    mb, packed_ww, packed_pp, _shape, psweep = frame_sweep
    packed_z = ut3_operations.pack_if_ragged(ztildes, mb[0].shape[-1])
    return sampling_derivatives.tv_probe_transpose_derivatives_from_sweep(
        packed_z, packed_ww, packed_pp, mb, psweep, order, sum_over_probes=sum_over_probes,
        chunk_size=chunk_size)


def utv_entries_transpose_derivatives_from_sweep(
        residual,     # entries residual jet (scalar), shape=(order+1,)+W+K+C
        frame_sweep,   # = utv_precompute_entries_frame_sweep_jets(...)
        order,        # highest derivative order
        sum_over_probes=True,
):  # -> bare variation supercore pair (dU_tilde, dG_tilde); stack K_new + C
    """Transpose ``𝒥ᵀ`` of the entries-derivatives reusing the jet sweep (scatter at the sweep's ``index``).
    Shares :py:func:`sampling_derivatives.tv_entries_transpose_derivatives_from_sweep`."""
    mb, index, packed_pp, _shape, psweep = frame_sweep
    return sampling_derivatives.tv_entries_transpose_derivatives_from_sweep(
        residual, index, packed_pp, mb, psweep, order, sum_over_probes=sum_over_probes)


# ----------------------------------------------------------------- derivative sampling (jets 𝒥; 3b-6'b)
# The symmetric-directional-derivative twins of utv_probe / apply / entries (the forward Riemannian
# Jacobian derivatives). Same boundary work: mask-once frame+variations, pack ww AND pp (entries slices
# fibers -> only pp packed), share sampling_derivatives.*_tangent_derivatives, unpack the probe output (which
# carries a leading derivative-order axis; the middle axis rides through unpack_vectors' `...`). Output
# order 0 is the ordinary tangent sample. These are the bare 𝒥 (no gauge projector Π).

def utv_probe_derivatives(
        ww,               # probe vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp,               # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data
        variations_data,  # UT3Variations .data
        order,            # highest derivative order
):  # -> MIRRORS ww: ragged -> len=d tuple ((order+1,)+W+K+C+(Ni,)); packed -> (d,)+(order+1,)+W+K+C+(N,)
    """Symmetric probe derivatives of a uniform tangent vector (the bare ``𝒥``): mask-once, share
    ``sampling_derivatives.tv_probe_derivatives``. **Mirrors** ``ww``'s packedness."""
    mv, mb = _mask_once(frame_data, variations_data)
    N = mb[0].shape[-1]
    ragged_in = not ut3_operations.is_packed(ww)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    zz = sampling_derivatives.tv_probe_derivatives(packed_ww, packed_pp, mv, mb, order)
    return ut3_operations.unpack_vectors(zz, frame_data[4]) if ragged_in else zz


def utv_apply_derivatives(
        ww,               # apply vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp,               # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data
        variations_data,  # UT3Variations .data
        order,            # highest derivative order
):  # -> scalar jets, shape=(order+1,)+W+K+C
    """Symmetric all-modes apply derivatives of a uniform tangent vector (the bare ``𝒥``; a scalar jet per
    stack element)."""
    mv, mb = _mask_once(frame_data, variations_data)
    N = mb[0].shape[-1]
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    return sampling_derivatives.tv_apply_derivatives(packed_ww, packed_pp, mv, mb, order)


def utv_entries_derivatives(
        index,            # int array, shape=(d,)+W -- the multi-indices
        pp,               # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data
        variations_data,  # UT3Variations .data
        order,            # highest derivative order
):  # -> scalar jets, shape=(order+1,)+W+K+C
    """Symmetric entry derivatives of a uniform tangent vector at ``index`` (the bare ``𝒥``; fiber slicing,
    so only ``pp`` is packed)."""
    mv, mb = _mask_once(frame_data, variations_data)
    packed_pp = ut3_operations.pack_if_ragged(pp, mb[0].shape[-1])
    return sampling_derivatives.tv_entries_derivatives(index, packed_pp, mv, mb, order)


# ----------------------------------------------------------------- the transpose 𝒥ᵀ (probe; 3b-6c)
def _gauge_masks_over_Knew(
        frame_data,      # UT3Frame .data
        out_supercore,   # a transpose-output variation supercore (d,)+K_new+C+(...), pins the stack
):  # -> the 4 variation rank masks, each (d,)+K_new+C+(size,)
    """The result tangent's variation masks: the frame's gauge-shifted variation masks
    ``(up, down, frame_left[:-1], frame_right[1:])`` broadcast (constant) over the new tangent stack
    ``K_new`` -- the leading stack of the transpose output minus the frame stack ``C``. ``sum_over_probes``
    determines ``K_new`` (``W+K`` kept / ``K`` summed); we read it off the output supercore rather than
    re-deriving it. Masks are host numpy (static aux), so this stays on ``np``."""
    gauge = ufv_masking.ufv_variation_masks(frame_data[5])   # length-d variation masks, stack C
    C = frame_data[0].shape[1:-2]                 # frame stack C (up supercore is (d,)+C+(nU,N))
    out_stack = out_supercore.shape[1:-2]         # K_new + C
    K_new = out_stack[:len(out_stack) - len(C)]

    def b(m):  # (d,)+C+(size,) -> (d,)+K_new+C+(size,)
        return np.broadcast_to(m.reshape(m.shape[:1] + (1,) * len(K_new) + m.shape[1:]),
                               m.shape[:1] + K_new + m.shape[1:])

    return tuple(b(m) for m in gauge)


def utv_probe_transpose(
        ztildes,          # probe residuals, len=d, ith elm_shape=W+K+C+(Ni,)
        ww,               # probe vectors,   len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data (an orthogonal frame), supercore stack = C
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the probe map to residuals (the bare adjoint; no gauge projector).

    Mask-once the frame, pack both the residuals and the probe vectors, share
    ``probing.tv_probe_transpose`` (which routes through the 3b-6a d-prefixed WKC contractions), and
    attach the result variation masks: the frame's gauge masks broadcast over the new tangent stack
    ``K_new`` (``W+K`` if ``sum_over_probes=False``, ``K`` if ``True``). The bare ``𝒥ᵀ``."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    N = mb[0].shape[-1]
    packed_z = ut3_operations.pack_if_ragged(ztildes, N)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    dU_tilde, dG_tilde = probing.tv_probe_transpose(
        packed_z, packed_ww, mb, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(frame_data, dU_tilde)
    return (dU_tilde, dG_tilde, frame_data[4], masks)


def utv_apply_transpose(
        c,                # residual, shape=W+K+C (a scalar per stack element)
        ww,               # apply vectors, len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data (orthogonal frame), stack=C
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the all-modes apply (the bare adjoint; the adjoint-state method).
    Mask-once the frame, pack ``ww``, share ``apply.tv_apply_transpose``, attach the gauge masks
    over the new tangent stack ``K_new``."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    packed_ww = ut3_operations.pack_if_ragged(ww, mb[0].shape[-1])
    dU_tilde, dG_tilde = apply.tv_apply_transpose(
        c, packed_ww, mb, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(frame_data, dU_tilde)
    return (dU_tilde, dG_tilde, frame_data[4], masks)


def utv_entries_transpose(
        c,                # residual, shape=W+K+C
        index,            # int array, shape=(d,)+W -- the indices c weights
        frame_data,       # UT3Frame .data (orthogonal frame), stack=C
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the all-modes entries (scatter ``c`` at ``index``). Like
    :py:func:`utv_apply_transpose` with the apply vectors replaced by the one-hot ``e_index`` (built
    packed inside ``entries._onehot_vectors``), so ``index`` needs no packing."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    dU_tilde, dG_tilde = entries.tv_entries_transpose(
        c, index, mb, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(frame_data, dU_tilde)
    return (dU_tilde, dG_tilde, frame_data[4], masks)


# ----------------------------------------------------------------- the transpose 𝒥ᵀ derivatives (jets; 3b-6'c)
# The jet-ified twins of the transpose wrappers above: back-project residual JETS (which carry the leading
# order axis) into a SINGLE variation gradient (the transpose sums the order axis in the assembly, so the
# output has no order axis -- structurally identical to the plain transpose). Mask-once, pack ww AND pp
# (entries: pp only), share sampling_derivatives.*_tangent_derivatives_transpose, attach the gauge masks over
# the new tangent stack K_new (the same _gauge_masks_over_Knew as the plain transpose). The bare 𝒥ᵀ.

def utv_probe_derivatives_transpose(
        ztildes,          # probe residual jets, len=d, ith elm_shape=(order+1,)+W+K+C+(Ni,)
        ww,               # probe vectors,   len=d, ith elm_shape=W+(Ni,)
        pp,               # perturbation P,  len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data (an orthogonal frame), supercore stack = C
        order,            # highest derivative order
        sum_over_probes=False,
        chunk_size=100,   # W-chunk size for the gradient assembly; None -> dense. docs/chunking.md
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the probe-derivative map to residual jets (the bare adjoint). Mask-once,
    pack the residual jets + ``ww`` + ``pp``, share ``sampling_derivatives.tv_probe_derivatives_transpose``,
    attach the gauge masks over the new tangent stack ``K_new``."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    N = mb[0].shape[-1]
    packed_z = ut3_operations.pack_if_ragged(ztildes, N)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    dU_tilde, dG_tilde = sampling_derivatives.tv_probe_derivatives_transpose(
        packed_z, packed_ww, packed_pp, mb, order, sum_over_probes=sum_over_probes, chunk_size=chunk_size)
    masks = _gauge_masks_over_Knew(frame_data, dU_tilde)
    return (dU_tilde, dG_tilde, frame_data[4], masks)


def utv_apply_derivatives_transpose(
        c,                # residual jet (scalar), shape=(order+1,)+W+K+C
        ww,               # apply vectors,  len=d, ith elm_shape=W+(Ni,)
        pp,               # perturbation P, len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data (orthogonal frame), stack=C
        order,            # highest derivative order
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the all-modes apply-derivative (the bare adjoint-state method)."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    N = mb[0].shape[-1]
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    dU_tilde, dG_tilde = sampling_derivatives.tv_apply_derivatives_transpose(
        c, packed_ww, packed_pp, mb, order, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(frame_data, dU_tilde)
    return (dU_tilde, dG_tilde, frame_data[4], masks)


def utv_entries_derivatives_transpose(
        c,                # residual jet (scalar), shape=(order+1,)+W+K+C
        index,            # int array, shape=(d,)+W
        pp,               # perturbation P, len=d, ith elm_shape=W+(Ni,)
        frame_data,       # UT3Frame .data (orthogonal frame), stack=C
        order,            # highest derivative order
        sum_over_probes=False,
):  # -> UT3Variations .data: (dU_tilde, dG_tilde, shape, masks); stack K_new + C
    """Apply the transpose ``𝒥ᵀ`` of the all-modes entry-derivative (scatter ``c`` at ``index``; ``pp`` packed,
    ``index`` unpacked -- the one-hots are built packed inside ``entries._onehot_vectors``)."""
    mb = ufv_masking.ufv_apply_frame_masks(frame_data)
    packed_pp = ut3_operations.pack_if_ragged(pp, mb[0].shape[-1])
    dU_tilde, dG_tilde = sampling_derivatives.tv_entries_derivatives_transpose(
        c, index, packed_pp, mb, order, sum_over_probes=sum_over_probes)
    masks = _gauge_masks_over_Knew(frame_data, dU_tilde)
    return (dU_tilde, dG_tilde, frame_data[4], masks)
