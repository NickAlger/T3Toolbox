# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.probing as probing
import t3toolbox.backend.probe_derivatives as probe_derivatives
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
    """Compute entries of a uniform Tucker tensor train (shares ``entries.tucker_tensor_train_entries``)."""
    masked = ut3_masking.apply_masks_to_cores(data)
    return entries.tucker_tensor_train_entries(masked, index)


def ut3_apply(
        data: UT3Data,
        vecs: typ.Sequence[NDArray],  # len=d, ith elm_shape=vec_stack+(Ni,)
) -> NDArray:                         # shape=vec_stack+stack_shape
    """Contract a uniform Tucker tensor train with vectors in all modes (shares
    ``apply.tucker_tensor_train_apply``). Vectors are zero-padded to ``N``."""
    masked = ut3_masking.apply_masks_to_cores(data)
    packed = ut3_operations.pack_vectors(vecs, masked[0].shape[-1])
    return apply.tucker_tensor_train_apply(masked, packed)


def ut3_probe(
        ww:   typ.Sequence[NDArray],  # len=d, ith elm_shape=W+(Ni,)
        data: UT3Data,
) -> typ.Tuple[NDArray, ...]:         # len=d, ith elm_shape=W+(Ni,)
    """Probe a uniform Tucker tensor train (contract all-but-one mode, for each mode; shares
    ``probing.probe_t3``). Vectors padded to ``N``; results sliced back to the real shape."""
    masked = ut3_masking.apply_masks_to_cores(data)            # guards: masks must be host
    packed = ut3_operations.pack_vectors(ww, masked[0].shape[-1])
    zz = probing.probe_t3(packed, masked)                       # packed (TRACED under jit), shape=(d,)+W+(N)
    # data[2] is the static int-tuple shape -> the unpack slices the TRACED zz with a static bound
    # (jit-safe; no np.asarray on zz).
    return ut3_operations.unpack_vectors(zz, data[2])


def ut3_full_sum(
        data: UT3Data,
) -> NDArray:  # shape=stack_shape (scalar if unstacked) -- sum over ALL physical modes
    """Sum the represented dense tensor over all physical modes: ``apply`` with all-ones vectors (the
    masked supercore's padding is zero, so ones over ``N`` sum only the real ``Ni``)."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)
    masked = ut3_masking.apply_masks_to_cores(data)
    d, N = masked[0].shape[0], masked[0].shape[-1]
    return apply.tucker_tensor_train_apply(masked, xnp.ones((d, N)))


# ----------------------------------------------------------------- corewise (non-manifold) transposes (3b-6c)
# Gradient of a sampling op w.r.t. the base's own supercores (the Section 6.3 (P,Q,O)->G substitution, via
# the now-polymorphic probing.*_corewise_transpose -- the tangent transpose at the frame (U, G, G, G)). For
# a uniform core-wise optimizer (Adam, L-BFGS). Mask-once + pack at the boundary; return the RAW gradient
# supercores (dU, dG), clean-padded (the masked base zeros the padding, so the gradient never grows rank
# into it) -- the uniform mirror of the ragged TuckerTensorTrain.*_corewise_transpose raw-tuple return.

def ut3_apply_corewise_transpose(
        c:    NDArray,                # residual, shape=W+K+C
        ww:   typ.Sequence[NDArray],  # apply vectors, len=d, ith elm_shape=W+(Ni,)
        data: UT3Data,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_apply`: gradient w.r.t. the base supercores. ``ww`` packed to N."""
    masked = ut3_masking.apply_masks_to_cores(data)
    packed = ut3_operations.pack_vectors(ww, masked[0].shape[-1])
    return probing.apply_corewise_transpose(c, packed, masked, sum_over_probes=sum_over_probes)


def ut3_entries_corewise_transpose(
        c:     NDArray,    # residual, shape=W+K+C
        index: NDArray,    # int, shape=(d,)+W (the indices c weights)
        data:  UT3Data,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_entries`: gradient w.r.t. the base supercores (index unpacked --
    the one-hot vectors are built packed inside ``probing._onehot_vectors``)."""
    masked = ut3_masking.apply_masks_to_cores(data)
    return probing.entries_corewise_transpose(c, index, masked, sum_over_probes=sum_over_probes)


def ut3_probe_corewise_transpose(
        ztildes: typ.Sequence[NDArray],  # probe residuals, len=d, ith elm_shape=W+K+C+(Ni,)
        ww:      typ.Sequence[NDArray],  # probe vectors,   len=d, ith elm_shape=W+(Ni,)
        data:    UT3Data,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_probe`: gradient w.r.t. the base supercores. ``ztildes`` and
    ``ww`` packed to N."""
    masked = ut3_masking.apply_masks_to_cores(data)
    N = masked[0].shape[-1]
    packed_z = ut3_operations.pack_vectors(ztildes, N)
    packed_ww = ut3_operations.pack_vectors(ww, N)
    return probing.probe_corewise_transpose(packed_z, packed_ww, masked, sum_over_probes=sum_over_probes)


# ----------------------------------------------------------------- derivative sampling (jets; 3b-6'b)
# The symmetric-directional-derivative twins of ut3_probe / ut3_apply / ut3_entries: mask-once, pack the
# probe vectors ww AND the perturbation direction pp (entries slices fibers, so only pp is packed), share
# the polymorphic probe_derivatives.*_derivatives_t3, and unpack the probe output (which now carries a
# leading derivative-order axis -- the middle axis rides through unpack_vectors' `...`). Output order 0 is
# the ordinary (non-derivative) sample.

def ut3_probe_derivatives(
        ww:    typ.Sequence[NDArray],  # probe vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp:    typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:  UT3Data,
        order: int,                    # highest derivative order
) -> typ.Tuple[NDArray, ...]:          # len=d, ith elm_shape=(order+1,)+W+(Ni,)
    """Symmetric probe derivatives of a uniform Tucker tensor train (shares
    ``probe_derivatives.probe_derivatives_t3``). ``ww``/``pp`` packed to ``N``; results sliced back."""
    masked = ut3_masking.apply_masks_to_cores(data)
    N = masked[0].shape[-1]
    packed_ww = ut3_operations.pack_vectors(ww, N)
    packed_pp = ut3_operations.pack_vectors(pp, N)
    zz = probe_derivatives.probe_derivatives_t3(packed_ww, packed_pp, masked, order)  # (d,)+(order+1,)+W+(N,)
    return ut3_operations.unpack_vectors(zz, data[2])


def ut3_apply_derivatives(
        ww:    typ.Sequence[NDArray],  # apply vectors X,        len=d, ith elm_shape=W+(Ni,)
        pp:    typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:  UT3Data,
        order: int,                    # highest derivative order
) -> NDArray:                          # shape=(order+1,)+W+stack_shape
    """Symmetric all-modes apply derivatives of a uniform Tucker tensor train (shares
    ``probe_derivatives.apply_derivatives_t3``; a scalar jet per stack element). ``ww``/``pp`` packed to ``N``."""
    masked = ut3_masking.apply_masks_to_cores(data)
    N = masked[0].shape[-1]
    packed_ww = ut3_operations.pack_vectors(ww, N)
    packed_pp = ut3_operations.pack_vectors(pp, N)
    return probe_derivatives.apply_derivatives_t3(packed_ww, packed_pp, masked, order)


def ut3_entries_derivatives(
        index: NDArray,                # int, shape=(d,)+W -- the grid points
        pp:    typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:  UT3Data,
        order: int,                    # highest derivative order
) -> NDArray:                          # shape=(order+1,)+W+stack_shape
    """Symmetric entry derivatives of a uniform Tucker tensor train at ``index`` (shares
    ``probe_derivatives.entries_derivatives_t3``; the up-index jet slices Tucker fibers, so only ``pp`` is
    packed -- ``index`` in ``[0,Ni)`` hits the real prefix)."""
    masked = ut3_masking.apply_masks_to_cores(data)
    packed_pp = ut3_operations.pack_vectors(pp, masked[0].shape[-1])
    return probe_derivatives.entries_derivatives_t3(index, packed_pp, masked, order)
