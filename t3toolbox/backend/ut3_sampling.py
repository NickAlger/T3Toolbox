# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform plain-layer sampling wrappers: probe/apply/entries on ut3 data (+ transposes, jets).

Grouped by OBJECT type, deliberately asymmetric with the ragged by-sampling-type modules: each
function here is a thin wrapper -- mask once, pack, delegate to the shared polymorphic ragged
machinery (the corewise transposes and the two-holes then re-mask their output) -- so it
lives with the object it wraps. The algorithm story is in
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
    'ut3_probe_corewise_derivatives_hessian',
    'ut3_probe_corewise_derivatives_hessian_from_sweep',
    'ut3_probe_corewise_hessian',
    'ut3_probe_corewise_hessian_from_sweep',
    'ut3_apply_corewise_hessian',
    'ut3_apply_corewise_hessian_from_sweep',
    'ut3_entries_corewise_hessian',
    'ut3_entries_corewise_hessian_from_sweep',
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


# ----------------------------------------------------------------- corewise (non-manifold) transposes
# Gradient of a sampling op w.r.t. the frame's own supercores (the Section 6.3 (P,Q,O)->G substitution, via
# the now-polymorphic probing.*_corewise_transpose -- the tangent transpose at the frame (U, G, G, G)). For
# a uniform core-wise optimizer (Adam, L-BFGS). Mask-once + pack at the boundary; return the RAW gradient
# supercores (dU, dG) -- the uniform mirror of the ragged TuckerTensorTrain.*_corewise_transpose raw-tuple
# return -- RE-MASKED on exit with the same point masks that masked the input. The raw padding is not clean
# (the boundary-bond squash is a sum over the whole bond, so its gradient broadcasts into the padded
# boundary-bond slots too); the re-mask zeroes it, so a consumer gets exact zeros there. The equivalence
# contract still declares the padding don't-care (every consumer masks on entry -- ut3_squash_tails
# included, since 2026-08-22), and the masks, not the values, govern rank.

def _remasked_core_gradient(
        grads:  typ.Tuple[NDArray, NDArray],  # raw (tucker grad, tt grad) supercores, stack [W+]K+C
        data:   UT3Data,                      # the point; its C-stacked masks broadcast over [W+]K
) -> typ.Tuple[NDArray, NDArray]:             # the same pair, padding exactly zero
    """Re-mask a corewise-transpose output with the point's own masks (:py:func:`ut3_masking.ut3_apply_masks`;
    its ``'...'`` einsums broadcast the ``C``-stacked masks over a kept ``W`` / tangent ``K`` stack). One
    multiply by a host mask: the real block is unchanged bit for bit."""
    return ut3_masking.ut3_apply_masks((grads[0], grads[1], data[2], data[3]))


def ut3_apply_corewise_transpose(
        c:    NDArray,                # residual, shape=W+K+C
        ww:   typ.Sequence[NDArray],  # apply vectors, len=d, ith elm_shape=W+(Ni,)
        data: UT3Data,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_apply`: gradient w.r.t. the frame supercores. ``ww`` packed to N."""
    masked = ut3_masking.ut3_apply_masks(data)
    packed = ut3_operations.pack_if_ragged(ww, masked[0].shape[-1])
    grads = apply.t3_apply_corewise_transpose(c, packed, masked, sum_over_probes=sum_over_probes)
    return _remasked_core_gradient(grads, data)


def ut3_entries_corewise_transpose(
        c:     NDArray,    # residual, shape=W+K+C
        index: NDArray,    # int, shape=(d,)+W (the indices c weights)
        data:  UT3Data,
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_entries`: gradient w.r.t. the frame supercores (index unpacked --
    the one-hot vectors are built packed inside ``entries._onehot_vectors``)."""
    masked = ut3_masking.ut3_apply_masks(data)
    grads = entries.t3_entries_corewise_transpose(c, index, masked, sum_over_probes=sum_over_probes)
    return _remasked_core_gradient(grads, data)


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
    grads = probing.t3_probe_corewise_transpose(packed_z, packed_ww, masked, sum_over_probes=sum_over_probes)
    return _remasked_core_gradient(grads, data)


# ----------------------------------------------------------------- derivative sampling (jets)
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


# ----------------------------------------------------------- corewise derivative transposes (jets)
# The jet-ified twins of the corewise transposes above: gradient of a plain-T3 derivative sampling op w.r.t.
# the frame's own supercores (the §6.3 (P,Q,O)->G substitution, via the now-polymorphic
# sampling_derivatives.*_corewise_derivatives_transpose). Mask-once + pack ww/pp (entries: pp only) at the
# boundary; return the RAW gradient supercores (dU, dG), re-masked on exit (clean padding, as above). For a
# uniform core-wise optimizer.

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
    grads = sampling_derivatives.t3_apply_corewise_derivatives_transpose(
        c, packed_ww, packed_pp, masked, order, sum_over_probes=sum_over_probes)
    return _remasked_core_gradient(grads, data)


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
    grads = sampling_derivatives.t3_entries_corewise_derivatives_transpose(
        c, index, packed_pp, masked, order, sum_over_probes=sum_over_probes)
    return _remasked_core_gradient(grads, data)


def ut3_probe_corewise_derivatives_transpose(
        ztildes: typ.Sequence[NDArray],  # probe residual jets, len=d, ith elm_shape=(order+1,)+W+C+(Ni,)
        ww:      typ.Sequence[NDArray],  # probe vectors,          len=d, ith elm_shape=W+(Ni,)
        pp:      typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:    UT3Data,
        order:   int,
        sum_over_probes: bool = False,
        chunk_size: typ.Optional[int] = sampling_derivatives.DEFAULT_CHUNK_SIZE,  # W-chunk size; None -> dense. docs/chunking.md
) -> typ.Tuple[NDArray, NDArray]:     # (tucker-core grad supercore, tt-core grad supercore)
    """Corewise transpose of :py:func:`ut3_probe_derivatives`: gradient w.r.t. the frame supercores.
    ``ztildes``/``ww``/``pp`` packed to N."""
    masked = ut3_masking.ut3_apply_masks(data)
    N = masked[0].shape[-1]
    packed_z = ut3_operations.pack_if_ragged(ztildes, N)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    grads = sampling_derivatives.t3_probe_corewise_derivatives_transpose(
        packed_z, packed_ww, packed_pp, masked, order, sum_over_probes=sum_over_probes, chunk_size=chunk_size)
    return _remasked_core_gradient(grads, data)


# ----------------------------------------------------------- corewise two-hole: the probe-derivative Hessian
# The derivative of ut3_probe_corewise_derivatives_transpose along raw core directions (the exact Riemannian
# Hessian's sampling term). The kernel is mask-free by contract and a second derivative multiplies a
# direction's padded slot into real slots, so the wrapper masks theta with ITS masks and theta_hat with the
# SAME masks (a K + C direction broadcasts against the C masks), packs, delegates to the polymorphic kernel,
# and re-masks the output (exactly-zero padding, as the one-hole transposes above).

def ut3_probe_corewise_derivatives_hessian(
        ztildes:          typ.Sequence[NDArray],   # residual jets (omega^2 applied), len=d, elm_shape=(order+1,)+W+C+(Ni,)
        ww:               typ.Sequence[NDArray],   # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:               typ.Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        data:             UT3Data,                 # the point theta, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],  # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        order:            int,                     # highest derivative order
        chunk_size:       typ.Optional[int] = sampling_derivatives.DEFAULT_CHUNK_SIZE,  # W-chunk size; None -> dense
) -> typ.Tuple[NDArray, NDArray]:                  # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """The uniform twin of :py:func:`~t3toolbox.backend.sampling_derivatives.t3_probe_corewise_derivatives_hessian`:
    the core-coordinate Hessian of ``theta -> <z~, S(Phi(theta))>`` applied to ``theta_hat``, for ``S`` the
    probe-derivative jets, on supercores.

    ``theta`` is masked with its masks, ``theta_hat`` with the SAME masks, ``ztildes`` / ``ww`` / ``pp`` packed to
    ``N``; the polymorphic kernel runs on the supercores (its ``chunk_size`` W-chunks the assemblies), and the
    output is re-masked with ``theta``'s masks. Per stack element the real block is the ragged two-hole of the real
    blocks, whatever the padding of ``theta``, ``theta_hat``, the residual or the sample holds.

    Examples
    --------
    With garbage in every padded slot of the direction, the real block is the ragged two-hole's and the padding
    is exactly zero; the bare kernel on the unmasked direction is not (the mask is load-bearing):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.backend.sampling_derivatives as sd
    >>> import t3toolbox.backend.ut3_masking as ut3m
    >>> import t3toolbox.backend.ut3_sampling as ut3s
    >>> np.random.seed(0)
    >>> rng = np.random.default_rng(0)
    >>> shape, order, W = (4, 5, 3), 2, 6
    >>> theta, direction = (t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)) for _ in range(2))
    >>> ww, pp = ([rng.standard_normal((W, N)) for N in shape] for _ in range(2))
    >>> zt = [rng.standard_normal((order + 1, W, N)) for N in shape]
    >>> u = ut3.UniformTuckerTensorTrain.from_t3(theta, N=6, n=4, r=4).data
    >>> ud = ut3.UniformTuckerTensorTrain.from_t3(direction, N=6, n=4, r=4).data
    >>> pad = [1.0 - m for m in ut3m.ut3_apply_masks((np.ones_like(ud[0]), np.ones_like(ud[1]), ud[2], ud[3]))]
    >>> dirty = (ud[0] + 1e3 * pad[0], ud[1] + 1e3 * pad[1])               # garbage in the direction's padding
    >>> KU, KG = ut3s.ut3_probe_corewise_derivatives_hessian(zt, ww, pp, u, dirty, order)
    >>> ref = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta.data, direction.data, order)
    >>> all(np.allclose(KG[i][:G.shape[0], :G.shape[1], :G.shape[2]], G) for i, G in enumerate(ref[1]))
    True
    >>> float(np.abs(KG * pad[1]).max()), float(np.abs(KU * pad[0]).max())   # the padding: exactly zero
    (0.0, 0.0)
    >>> import t3toolbox.backend.ut3_operations as ut3o
    >>> pk = lambda v: ut3o.pack_if_ragged(v, 6)
    >>> bare = sd.t3_probe_corewise_derivatives_hessian(pk(zt), pk(ww), pk(pp), ut3m.ut3_apply_masks(u), dirty, order)
    >>> all(np.allclose(bare[1][i][:G.shape[0], :G.shape[1], :G.shape[2]], G) for i, G in enumerate(ref[1]))
    False
    """
    masked = ut3_masking.ut3_apply_masks(data)
    directions = ut3_masking.ut3_apply_masks((core_directions[0], core_directions[1], data[2], data[3]))
    N = masked[0].shape[-1]
    packed_z = ut3_operations.pack_if_ragged(ztildes, N)
    packed_ww = ut3_operations.pack_if_ragged(ww, N)
    packed_pp = ut3_operations.pack_if_ragged(pp, N)
    grads = sampling_derivatives.t3_probe_corewise_derivatives_hessian(
        packed_z, packed_ww, packed_pp, masked, directions, order, chunk_size=chunk_size)
    return _remasked_core_gradient(grads, data)


def ut3_probe_corewise_derivatives_hessian_from_sweep(
        ztildes:          typ.Union[typ.Sequence[NDArray], NDArray],  # residual jets (omega^2 applied); ragged, or
                                                   #   packed (d,)+(order+1,)+W+C+(N,)
        data:             UT3Data,                 # the point theta = the sweep frame's base point, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],  # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        order:            int,                     # highest derivative order
        frame_sweep:      typ.Tuple,               # a uniform jet sweep at a frame whose base point is theta
        chunk_size:       typ.Optional[int] = sampling_derivatives.DEFAULT_CHUNK_SIZE,  # W-chunk size; None -> dense
) -> typ.Tuple[NDArray, NDArray]:                  # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """:py:func:`ut3_probe_corewise_derivatives_hessian` reusing a uniform jet sweep's packed sample and its
    ``xi`` / ``mu`` jets -- the reuse hook for a uniform local model, the twin of
    :py:func:`~t3toolbox.backend.sampling_derivatives.t3_probe_corewise_derivatives_hessian_from_sweep`.

    ``frame_sweep`` is :py:func:`~t3toolbox.backend.utv_sampling.utv_precompute_probe_frame_sweep_jets` at the
    manifold frame ``(U, O, P, Q)`` of ``theta = (U, P)`` (or the apply sweep at the corewise frame): its masked
    frame's ``U`` and ``P`` are ``theta`` masked with the frame's ``(up, left)`` masks, so ``data`` must carry those
    masks (a uniform kind reads them off the frame it is handed; at TT slack the point's own masks are a strict
    superset). ``theta`` and ``theta_hat`` are masked with ``data``'s masks, and the output is re-masked."""
    _mb, packed_ww, packed_pp, _shape, psweep = frame_sweep
    masked = ut3_masking.ut3_apply_masks(data)
    directions = ut3_masking.ut3_apply_masks((core_directions[0], core_directions[1], data[2], data[3]))
    packed_z = ut3_operations.pack_if_ragged(ztildes, masked[0].shape[-1])
    grads = sampling_derivatives.t3_probe_corewise_derivatives_hessian_from_sweep(
        packed_z, packed_ww, packed_pp, masked, directions, order, psweep, chunk_size=chunk_size)
    return _remasked_core_gradient(grads, data)


# ----------------------------------------------------------- corewise two-holes: the plain kinds' Hessians
# The same wrapper, for the plain probe / apply / entries two-holes (the derivatives of ut3_*_corewise_transpose
# along raw core directions): theta masked with its masks, theta_hat with the SAME masks, the residual and the
# vectors packed (entries: the integer index is not packed -- the one-hots are built packed inside the kernel),
# the polymorphic kernel run on the supercores, the output re-masked.

def _masked_point_and_directions(
        data:             UT3Data,                       # the point theta, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores, (d,)+K+C+(...)
) -> typ.Tuple[
    typ.Tuple[NDArray, NDArray],                         # theta's supercores, masked with its masks
    typ.Tuple[NDArray, NDArray],                         # theta_hat's supercores, masked with THETA's masks
]:
    """The two masks every uniform two-hole needs: ``theta`` with its own masks and ``theta_hat`` with the SAME
    masks (a ``K + C`` direction broadcasts against the ``C`` masks). The kernel downstream is mask-free, and a
    second derivative multiplies a direction's padded slot into real slots -- so the second mask is load-bearing."""
    masked = ut3_masking.ut3_apply_masks(data)
    directions = ut3_masking.ut3_apply_masks((core_directions[0], core_directions[1], data[2], data[3]))
    return masked, directions


def ut3_probe_corewise_hessian(
        ztildes:          typ.Union[typ.Sequence[NDArray], NDArray],  # residuals (omega^2 applied), len=d, elm_shape=W+C+(Ni,); or packed
        ww:               typ.Union[typ.Sequence[NDArray], NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,); or packed

        data:             UT3Data,                       # the point theta, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """The uniform twin of :py:func:`~t3toolbox.backend.probing.t3_probe_corewise_hessian`: the core-coordinate
    Hessian of ``theta -> <z~, probe(Phi(theta))>`` applied to ``theta_hat``, on supercores. ``theta`` is masked with
    its masks, ``theta_hat`` with the SAME masks, ``ztildes`` / ``ww`` packed to ``N``, and the output re-masked; per
    stack element the real block is the ragged two-hole of the real blocks, whatever the padding holds.

    Examples
    --------
    With garbage in every padded slot of the direction, the real block is the ragged two-hole's and the padding is
    exactly zero:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.backend.probing as t3p
    >>> import t3toolbox.backend.ut3_masking as ut3m
    >>> import t3toolbox.backend.ut3_sampling as ut3s
    >>> np.random.seed(0)
    >>> rng = np.random.default_rng(0)
    >>> shape, W = (4, 5, 3), 6
    >>> theta, direction = (t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)) for _ in range(2))
    >>> ww, zt = ([rng.standard_normal((W, N)) for N in shape] for _ in range(2))
    >>> u = ut3.UniformTuckerTensorTrain.from_t3(theta, N=6, n=4, r=4).data
    >>> ud = ut3.UniformTuckerTensorTrain.from_t3(direction, N=6, n=4, r=4).data
    >>> pad = [1.0 - m for m in ut3m.ut3_apply_masks((np.ones_like(ud[0]), np.ones_like(ud[1]), ud[2], ud[3]))]
    >>> dirty = (ud[0] + 1e3 * pad[0], ud[1] + 1e3 * pad[1])               # garbage in the direction's padding
    >>> KU, KG = ut3s.ut3_probe_corewise_hessian(zt, ww, u, dirty)
    >>> ref = t3p.t3_probe_corewise_hessian(zt, ww, theta.data, direction.data)
    >>> all(np.allclose(KG[i][:G.shape[0], :G.shape[1], :G.shape[2]], G) for i, G in enumerate(ref[1]))
    True
    >>> float(np.abs(KG * pad[1]).max()), float(np.abs(KU * pad[0]).max())   # the padding: exactly zero
    (0.0, 0.0)
    """
    masked, directions = _masked_point_and_directions(data, core_directions)
    N = masked[0].shape[-1]
    grads = probing.t3_probe_corewise_hessian(ut3_operations.pack_if_ragged(ztildes, N),
                                              ut3_operations.pack_if_ragged(ww, N), masked, directions)
    return _remasked_core_gradient(grads, data)


def ut3_probe_corewise_hessian_from_sweep(
        ztildes:          typ.Union[typ.Sequence[NDArray], NDArray],  # residuals (omega^2 applied); ragged, or packed (d,)+W+C+(N,)

        data:             UT3Data,                       # the point theta = the sweep frame's base point, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        frame_sweep:      typ.Tuple,                     # a uniform probe (or apply) sweep at a frame whose base point is theta
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """:py:func:`ut3_probe_corewise_hessian` reusing a uniform sweep's packed vectors and its ``xi`` / ``mu`` -- the
    reuse hook for a uniform local model, the twin of
    :py:func:`~t3toolbox.backend.probing.t3_probe_corewise_hessian_from_sweep`. ``frame_sweep`` is
    :py:func:`~t3toolbox.backend.utv_sampling.utv_precompute_probe_frame_sweep` at the manifold frame ``(U, O, P, Q)``
    of ``theta = (U, P)`` (or a sweep at the corewise frame), so ``data`` must carry that frame's ``(up, left)`` masks
    (a uniform kind reads them off the frame it is handed); ``theta`` and ``theta_hat`` are masked with them and the
    output is re-masked."""
    _mb, packed_ww, _shape, psweep = frame_sweep
    masked, directions = _masked_point_and_directions(data, core_directions)
    packed_z = ut3_operations.pack_if_ragged(ztildes, masked[0].shape[-1])
    grads = probing.t3_probe_corewise_hessian_from_sweep(packed_z, packed_ww, masked, directions, psweep)
    return _remasked_core_gradient(grads, data)


def ut3_apply_corewise_hessian(
        c:                NDArray,                       # residual (omega^2 applied), shape=W+C

        ww:               typ.Union[typ.Sequence[NDArray], NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,); or packed

        data:             UT3Data,                       # the point theta, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """The uniform twin of :py:func:`~t3toolbox.backend.apply.t3_apply_corewise_hessian` (``theta`` and ``theta_hat``
    masked with ``theta``'s masks, ``ww`` packed, the output re-masked), as :py:func:`ut3_probe_corewise_hessian`."""
    masked, directions = _masked_point_and_directions(data, core_directions)
    grads = apply.t3_apply_corewise_hessian(c, ut3_operations.pack_if_ragged(ww, masked[0].shape[-1]), masked,
                                            directions)
    return _remasked_core_gradient(grads, data)


def ut3_apply_corewise_hessian_from_sweep(
        c:                NDArray,                       # residual (omega^2 applied), shape=W+C
        data:             UT3Data,                       # the point theta = the sweep frame's base point, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        frame_sweep:      typ.Tuple,                     # utv_precompute_apply_frame_sweep at a frame whose base point is theta
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """:py:func:`ut3_apply_corewise_hessian` reusing a uniform apply sweep's packed vectors and lean ``(xi, mu)``
    (``data`` carries the sweep frame's ``(up, left)`` masks, as :py:func:`ut3_probe_corewise_hessian_from_sweep`)."""
    _mb, packed_ww, _shape, psweep = frame_sweep
    masked, directions = _masked_point_and_directions(data, core_directions)
    grads = apply.t3_apply_corewise_hessian_from_sweep(c, packed_ww, masked, directions, psweep)
    return _remasked_core_gradient(grads, data)


def ut3_entries_corewise_hessian(
        c:                NDArray,                       # residual (omega^2 applied), shape=W+C
        index:            NDArray,                       # int, shape=(d,)+W
        data:             UT3Data,                       # the point theta, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """The uniform twin of :py:func:`~t3toolbox.backend.entries.t3_entries_corewise_hessian` (``index`` unpacked -- the
    one-hot vectors are built packed inside the kernel), as :py:func:`ut3_apply_corewise_hessian`."""
    masked, directions = _masked_point_and_directions(data, core_directions)
    grads = entries.t3_entries_corewise_hessian(c, index, masked, directions)
    return _remasked_core_gradient(grads, data)


def ut3_entries_corewise_hessian_from_sweep(
        c:                NDArray,                       # residual (omega^2 applied), shape=W+C
        data:             UT3Data,                       # the point theta = the sweep frame's base point, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        frame_sweep:      typ.Tuple,                     # utv_precompute_entries_frame_sweep at a frame whose base point is theta
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """:py:func:`ut3_entries_corewise_hessian` reusing a uniform entries sweep's index and lean (fiber-sliced)
    ``(xi, mu)`` (``data`` carries the sweep frame's ``(up, left)`` masks)."""
    _mb, index, _shape, psweep = frame_sweep
    masked, directions = _masked_point_and_directions(data, core_directions)
    grads = entries.t3_entries_corewise_hessian_from_sweep(c, index, masked, directions, psweep)
    return _remasked_core_gradient(grads, data)
