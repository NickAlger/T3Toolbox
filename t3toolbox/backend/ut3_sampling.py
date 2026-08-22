# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform plain-layer sampling wrappers: probe/apply/entries on ut3 data (+ transposes, jets).

Grouped by OBJECT type, deliberately asymmetric with the ragged by-sampling-type modules: each
function here is a thin wrapper -- mask once, pack, delegate to the shared polymorphic ragged
machinery -- so it lives with the object it wraps. The algorithm story is in
``probing``/``apply``/``entries``/``sampling_derivatives`` (``docs/naming_conventions.md``).
"""
import numpy as np
import typing as typ

import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as sampling_derivatives
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import *

__all__ = [
    'ut3_entries',
    'ut3_apply',
    'ut3_probe',
    'ut3_full_sum',
    'ut3_apply_corewise_transpose',
    'ut3_entries_corewise_transpose',
    'ut3_probe_corewise_transpose',
    'ut3_probe_derivatives',
    'ut3_apply_derivatives',
    'ut3_entries_derivatives',
    'ut3_apply_corewise_derivatives_transpose',
    'ut3_entries_corewise_derivatives_transpose',
    'ut3_probe_corewise_derivatives_transpose',
]

# All re-mask first (so the padded "garbage" contributes nothing), then call the SHARED, already-
# polymorphic sampling backends (entries/apply/probing) on the masked supercores. Physical vectors are
# zero-padded to N (pack) and probe results sliced back to the real shape (unpack); index entries need
# no packing (the real block is the prefix, so an index in [0,Ni) hits the right slot).

# .data[2] is the static int-tuple shape; .data[3] = (tucker_edge_mask, tt_edge_mask) are HOST bool,
# static structure (numpy, never traced); the supercores are xnp.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]


def ut3_entries(
        data:  UT3Data,
        index: NDArray,  # dtype=int, shape=(d,)+idx_stack
) -> NDArray:            # shape=idx_stack+stack_shape
    """Compute entries of a uniform Tucker tensor train (shares ``entries.t3_entries``)."""
    masked = ut3_masking.ut3_apply_masks(data)
    return entries.t3_entries(masked, index)


def ut3_apply(
        data: UT3Data,
        vecs: typ.Sequence[NDArray],  # len=d, ith elm_shape=vec_stack+(Ni,)
) -> NDArray:                         # shape=vec_stack+stack_shape
    """Contract a uniform Tucker tensor train with vectors in all modes (shares
    ``apply.t3_apply``). Vectors are zero-padded to ``N``."""
    masked = ut3_masking.ut3_apply_masks(data)
    packed = ut3_operations.pack_if_ragged(vecs, masked[0].shape[-1])
    return apply.t3_apply(masked, packed)


def ut3_probe(
        ww:   typ.Sequence[NDArray],  # len=d, ith elm_shape=W+(Ni,)
        data: UT3Data,
) -> typ.Union[typ.Tuple[NDArray, ...], NDArray]:  # MIRRORS ww: ragged -> len=d tuple; packed -> (d,)+W+(N,)
    """Probe a uniform Tucker tensor train (contract all-but-one mode, for each mode; shares
    ``probing.t3_probe``). **Mirrors** ``ww``'s packedness: a ragged ``ww`` returns a ``len=d`` tuple sliced
    back to real widths; a packed ``ww`` returns the packed ``(d,)+W+(N,)`` array (zero-padded prefix)."""
    masked = ut3_masking.ut3_apply_masks(data)            # guards: masks must be host
    ragged_in = not ut3_operations.is_packed(ww)
    packed = ut3_operations.pack_if_ragged(ww, masked[0].shape[-1])
    zz = probing.t3_probe(packed, masked)                       # packed (TRACED under jit), shape=(d,)+W+(N)
    # MIRROR: ragged ww -> slice to real widths (a static bound from data[2], jit-safe); packed ww ->
    # return the packed array unchanged.
    return ut3_operations.unpack_vectors(zz, data[2]) if ragged_in else zz


def ut3_full_sum(
        data: UT3Data,
) -> NDArray:  # shape=stack_shape (scalar if unstacked) -- sum over ALL physical modes
    """Sum the represented dense tensor over all physical modes: ``apply`` with all-ones vectors (the
    masked supercore's padding is zero, so ones over ``N`` sum only the real ``Ni``)."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)
    masked = ut3_masking.ut3_apply_masks(data)
    d, N = masked[0].shape[0], masked[0].shape[-1]
    return apply.t3_apply(masked, xnp.ones((d, N)))


# ----------------------------------------------------------------- corewise (non-manifold) transposes (3b-6c)
# Gradient of a sampling op w.r.t. the frame's own supercores (the Section 6.3 (P,Q,O)->G substitution, via
# the now-polymorphic probing.*_corewise_transpose -- the tangent transpose at the frame (U, G, G, G)). For
# a uniform core-wise optimizer (Adam, L-BFGS). Mask-once + pack at the boundary; return the RAW gradient
# supercores (dU, dG) -- the uniform mirror of the ragged TuckerTensorTrain.*_corewise_transpose raw-tuple
# return. The padding of the result is NOT guaranteed clean: the boundary-bond squash is a sum over the whole
# bond, so its gradient broadcasts into the padded boundary-bond slots too. That is don't-care by the
# equivalence contract (every consumer masks on entry -- ut3_squash_tails included, since 2026-08-22), and
# the masks, not the values, govern rank.

def ut3_apply_corewise_transpose(
        c:    NDArray,                # residual, shape=W+K+C
        ww:   typ.Sequence[NDArray],  # apply vectors, len=d, ith elm_shape=W+(Ni,)
        data: UT3Data,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_apply`: gradient w.r.t. the frame supercores. ``ww`` packed to N."""
    masked = ut3_masking.ut3_apply_masks(data)
    packed = ut3_operations.pack_if_ragged(ww, masked[0].shape[-1])
    return apply.t3_apply_corewise_transpose(c, packed, masked, sum_over_probes=sum_over_probes)


def ut3_entries_corewise_transpose(
        c:     NDArray,    # residual, shape=W+K+C
        index: NDArray,    # int, shape=(d,)+W (the indices c weights)
        data:  UT3Data,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_entries`: gradient w.r.t. the frame supercores (index unpacked --
    the one-hot vectors are built packed inside ``entries._onehot_vectors``)."""
    masked = ut3_masking.ut3_apply_masks(data)
    return entries.t3_entries_corewise_transpose(c, index, masked, sum_over_probes=sum_over_probes)


def ut3_probe_corewise_transpose(
        ztildes: typ.Sequence[NDArray],  # probe residuals, len=d, ith elm_shape=W+K+C+(Ni,)
        ww:      typ.Sequence[NDArray],  # probe vectors,   len=d, ith elm_shape=W+(Ni,)
        data:    UT3Data,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_probe`: gradient w.r.t. the frame supercores. ``ztildes`` and
    ``ww`` packed to N."""
    masked = ut3_masking.ut3_apply_masks(data)
    N = masked[0].shape[-1]
    packed_z = ut3_operations.pack_if_ragged(ztildes, N)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    return probing.t3_probe_corewise_transpose(packed_z, packed_ww, masked, sum_over_probes=sum_over_probes)


# ----------------------------------------------------------------- derivative sampling (jets; 3b-6'b)
# The symmetric-directional-derivative twins of ut3_probe / ut3_apply / ut3_entries: mask-once, pack the
# probe vectors ww AND the perturbation direction pp (entries slices fibers, so only pp is packed), share
# the polymorphic sampling_derivatives.t3_*_derivatives, and unpack the probe output (which now carries a
# leading derivative-order axis -- the middle axis rides through unpack_vectors' `...`). Output order 0 is
# the ordinary (non-derivative) sample.

def ut3_probe_derivatives(
        ww:    typ.Sequence[NDArray],  # probe vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp:    typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:  UT3Data,
        order: int,                    # highest derivative order
) -> typ.Union[typ.Tuple[NDArray, ...], NDArray]:  # MIRRORS ww: ragged -> len=d tuple; packed -> (d,)+(order+1,)+W+(N,)
    """Symmetric probe derivatives of a uniform Tucker tensor train (shares
    ``sampling_derivatives.t3_probe_derivatives``). **Mirrors** ``ww``'s packedness (ragged -> ``len=d`` tuple of
    real widths; packed -> the packed ``(d,)+(order+1,)+W+(N,)`` array)."""
    masked = ut3_masking.ut3_apply_masks(data)
    N = masked[0].shape[-1]
    ragged_in = not ut3_operations.is_packed(ww)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    zz = sampling_derivatives.t3_probe_derivatives(packed_ww, packed_pp, masked, order)  # (d,)+(order+1,)+W+(N,)
    return ut3_operations.unpack_vectors(zz, data[2]) if ragged_in else zz


def ut3_apply_derivatives(
        ww:    typ.Sequence[NDArray],  # apply vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp:    typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:  UT3Data,
        order: int,                    # highest derivative order
) -> NDArray:                          # shape=(order+1,)+W+stack_shape
    """Symmetric all-modes apply derivatives of a uniform Tucker tensor train (shares
    ``sampling_derivatives.t3_apply_derivatives``; a scalar jet per stack element). ``ww``/``pp`` packed to ``N``."""
    masked = ut3_masking.ut3_apply_masks(data)
    N = masked[0].shape[-1]
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    return sampling_derivatives.t3_apply_derivatives(packed_ww, packed_pp, masked, order)


def ut3_entries_derivatives(
        index: NDArray,                # int, shape=(d,)+W -- the grid points
        pp:    typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:  UT3Data,
        order: int,                    # highest derivative order
) -> NDArray:                          # shape=(order+1,)+W+stack_shape
    """Symmetric entry derivatives of a uniform Tucker tensor train at ``index`` (shares
    ``sampling_derivatives.t3_entries_derivatives``; the up-index jet slices Tucker fibers, so only ``pp`` is
    packed -- ``index`` in ``[0,Ni)`` hits the real prefix)."""
    masked = ut3_masking.ut3_apply_masks(data)
    packed_pp = ut3_operations.pack_if_ragged(pp, masked[0].shape[-1])
    return sampling_derivatives.t3_entries_derivatives(index, packed_pp, masked, order)


# ----------------------------------------------------------- corewise derivative transposes (jets; 3b-6'c)
# The jet-ified twins of the corewise transposes above: gradient of a plain-T3 derivative sampling op w.r.t.
# the frame's own supercores (the §6.3 (P,Q,O)->G substitution, via the now-polymorphic
# sampling_derivatives.*_corewise_derivatives_transpose). Mask-once + pack ww/pp (entries: pp only) at the
# boundary; return the RAW gradient supercores (dU, dG), clean-padded. For a uniform core-wise optimizer.

def ut3_apply_corewise_derivatives_transpose(
        c:     NDArray,                # residual jet (scalar), shape=(order+1,)+W+K+C
        ww:    typ.Sequence[NDArray],  # apply vectors,          len=d, ith elm_shape=W+(Ni,)
        pp:    typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:  UT3Data,
        order: int,                    # highest derivative order
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_apply_derivatives`: gradient w.r.t. the frame supercores. ``ww``/``pp`` packed to N."""
    masked = ut3_masking.ut3_apply_masks(data)
    N = masked[0].shape[-1]
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    return sampling_derivatives.t3_apply_corewise_derivatives_transpose(
        c, packed_ww, packed_pp, masked, order, sum_over_probes=sum_over_probes)


def ut3_entries_corewise_derivatives_transpose(
        c:     NDArray,                # residual jet (scalar), shape=(order+1,)+W+K+C
        index: NDArray,                # int, shape=(d,)+W
        pp:    typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:  UT3Data,
        order: int,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_entries_derivatives`: gradient w.r.t. the frame supercores
    (``pp`` packed, ``index`` unpacked)."""
    masked = ut3_masking.ut3_apply_masks(data)
    packed_pp = ut3_operations.pack_if_ragged(pp, masked[0].shape[-1])
    return sampling_derivatives.t3_entries_corewise_derivatives_transpose(
        c, index, packed_pp, masked, order, sum_over_probes=sum_over_probes)


def ut3_probe_corewise_derivatives_transpose(
        ztildes: typ.Sequence[NDArray],  # probe residual jets, len=d, ith elm_shape=(order+1,)+W+C+(Ni,)
        ww:      typ.Sequence[NDArray],  # probe vectors,          len=d, ith elm_shape=W+(Ni,)
        pp:      typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:    UT3Data,
        order:   int,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_probe_derivatives`: gradient w.r.t. the frame supercores.
    ``ztildes``/``ww``/``pp`` packed to N."""
    masked = ut3_masking.ut3_apply_masks(data)
    N = masked[0].shape[-1]
    packed_z = ut3_operations.pack_if_ragged(ztildes, N)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    return sampling_derivatives.t3_probe_corewise_derivatives_transpose(
        packed_z, packed_ww, packed_pp, masked, order, sum_over_probes=sum_over_probes)
