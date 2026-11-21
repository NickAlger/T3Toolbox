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

The twins of the ragged sampling functions (``ut3_probe``, ``ut3_apply``, ``ut3_entries`` and their
``*_corewise_tangent``, ``*_corewise_transpose``, ``*_corewise_hessian``; graded messages P12) take the ragged
functions' signatures, with ``x`` the uniform point's ``.data``, and are one call of the engine
(:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`), which packs a ragged sample, masks every
operand on entry and every output on exit. A probe output mirrors the sample's form: ragged in, ragged out; packed
in, packed out. The jet wrappers (``*_derivatives*``) still share ``sampling_derivatives``.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as sampling_derivatives
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.sampling_variables as sampling_variables
from t3toolbox.backend.common import *

__all__ = [
    'ut3_entries',
    'ut3_apply',
    'ut3_probe',
    'ut3_full_sum',
    'ut3_apply_corewise_tangent',
    'ut3_apply_corewise_transpose',
    'ut3_entries_corewise_tangent',
    'ut3_entries_corewise_transpose',
    'ut3_probe_corewise_tangent',
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
    'ut3_apply_corewise_derivatives_hessian',
    'ut3_apply_corewise_derivatives_hessian_from_sweep',
    'ut3_entries_corewise_derivatives_hessian',
    'ut3_entries_corewise_derivatives_hessian_from_sweep',
]

# All re-mask first (so the padded "garbage" contributes nothing), then call the SHARED, already-
# polymorphic sampling backends (entries/apply/probing) on the masked supercores. Physical vectors are
# zero-padded to N (pack) and probe results sliced back to the real shape (unpack); index entries need
# no packing (the real block is the prefix, so an index in [0,Ni) hits the right slot).

# .data[2] is the static int-tuple shape; .data[3] = (tucker_edge_mask, tt_edge_mask) are HOST bool,
# static structure (numpy, never traced); the supercores are xnp.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]
Vectors = typ.Union[typ.Sequence[NDArray], NDArray]   # one vector per mode: ragged, len=d; or packed, (d,)+...+(N,)


def ut3_entries(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        index:             NDArray,                      # int, shape=(d,)+W: the grid points
) -> NDArray:                               # z. shape=W+C
    """Entries of a uniform Tucker tensor train at ``index``: the uniform twin of
    :py:func:`~t3toolbox.backend.entries.t3_entries`, one call of the engine on ``x`` wanting ``z``."""
    (z,) = sampling_variables.sampling_variables('entries', x, index, ('z',))
    return z


def ut3_apply(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: len=d, J_in_i+W+(Ni,); or packed (d,)+J+W+(N,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # z. shape=F+W+C
    """Contract a uniform Tucker tensor train with the input jets in all modes: the uniform twin of
    :py:func:`~t3toolbox.backend.apply.t3_apply`, one call of the engine on ``x`` wanting ``z`` (ragged or packed
    ``ww``; the output is one array per sample either way)."""
    (z,) = sampling_variables.sampling_variables('apply', x, ww, ('z',), order=order)
    return z


def ut3_probe(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: len=d, J_in_i+W+(Ni,); or packed (d,)+J+W+(N,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> Vectors:                               # z, MIRRORS ww: len=d, F+W+C+(Ni,); or packed
    """Probe a uniform Tucker tensor train (all modes but one contracted, for each mode in turn): the uniform twin of
    :py:func:`~t3toolbox.backend.probing.t3_probe`, one call of the engine on ``x`` wanting ``z``. **Mirrors** the
    sample's form: a ragged ``ww`` gives a ``len=d`` tuple sliced to the real widths, a packed ``ww`` the packed
    array (zero padding). The engine packs a ragged ``ww`` (``len_J=len(order)``) and masks on entry and exit."""
    ragged_in = not ut3_operations.is_packed(ww)
    (z,) = sampling_variables.sampling_variables('probe', x, ww, ('z',), order=order)
    return ut3_operations.unpack_vectors(z, x[2]) if ragged_in else z


def ut3_full_sum(
        data: UT3Data,
) -> NDArray:  # shape=stack_shape (scalar if unstacked) -- sum over ALL physical modes
    """Sum the represented dense tensor over all physical modes: ``apply`` with all-ones vectors (the
    masked supercore's padding is zero, so ones over ``N`` sum only the real ``Ni``)."""
    use_jax = tree_contains_jax(data[:2])
    xnp, _, _ = get_backend(True, use_jax)
    d, N = data[0].shape[0], data[0].shape[-1]
    (z,) = sampling_variables.sampling_variables('apply', data, xnp.ones((d, N)), ('z',))   # the mode padding masked
    return z


# ----------------------------------------------------------------- corewise (non-manifold) transposes
# Gradient of a sampling op w.r.t. the frame's own supercores (the Section 6.3 (U,L,R)->G substitution, via
# the now-polymorphic probing.*_corewise_transpose -- the tangent transpose at the frame (B, G, G, G)). For
# a uniform core-wise optimizer (Adam, L-BFGS). Mask-once + pack at the boundary; return the RAW gradient
# supercores (dB, dG) -- the uniform mirror of the ragged TuckerTensorTrain.*_corewise_transpose raw-tuple
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


def ut3_apply_corewise_tangent(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: len=d, J_in_i+W+(Ni,); or packed (d,)+J+W+(N,)
        core_directions:   typ.Tuple[NDArray, NDArray],  # (B_dot, G_dot): supercores (d,)+K+C+(...), the point's sizes
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # J x_dot. shape=F+W+K+C
    """The corewise forward derivative on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.apply.t3_apply_corewise_tangent`, ``J x_dot`` along ``core_directions``. One call
    of the engine on ``x`` wanting ``Jp`` (its raw route); the engine masks the directions with the point's masks,
    so their padding is never read."""
    (Jp,) = sampling_variables.sampling_variables('apply', x, ww, ('Jp',), order=order,
                                                   core_directions=core_directions)
    return Jp


def ut3_apply_corewise_transpose(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: len=d, J_in_i+W+(Ni,); or packed (d,)+J+W+(N,)
        residual:          NDArray,                      # of the type of z: shape=F+W+[K]+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block, (d,)+[W]+[K]+C+(n,N)
        NDArray,                                         # TT block,     (d,)+[W]+[K]+C+(r,n,r)
]:
    """The corewise transpose on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.apply.t3_apply_corewise_transpose`, the gradient of ``<residual, z(x)>`` in the
    cores. The engine's ``JtJp`` with the residual held as ``Jp``; the output is masked by the point's masks."""
    held = sampling_variables.HeldVariables(Jp=residual)
    (blocks,) = sampling_variables.sampling_variables('apply', x, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def ut3_entries_corewise_tangent(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        index:             NDArray,                      # int, shape=(d,)+W: the grid points
        core_directions:   typ.Tuple[NDArray, NDArray],  # (B_dot, G_dot): supercores (d,)+K+C+(...), the point's sizes
) -> NDArray:                               # J x_dot. shape=W+K+C
    """The corewise forward derivative on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.entries.t3_entries_corewise_tangent`, ``J x_dot`` along ``core_directions``. One
    call of the engine on ``x`` wanting ``Jp`` (its raw route); the engine masks the directions with the point's
    masks, so their padding is never read."""
    (Jp,) = sampling_variables.sampling_variables('entries', x, index, ('Jp',),
                                                   core_directions=core_directions)
    return Jp


def ut3_entries_corewise_transpose(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        index:             NDArray,                      # int, shape=(d,)+W: the grid points
        residual:          NDArray,                      # of the type of z: shape=W+[K]+C
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block, (d,)+[W]+[K]+C+(n,N)
        NDArray,                                         # TT block,     (d,)+[W]+[K]+C+(r,n,r)
]:
    """The corewise transpose on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.entries.t3_entries_corewise_transpose`, the gradient of ``<residual, z(x)>`` in the
    cores. The engine's ``JtJp`` with the residual held as ``Jp``; the output is masked by the point's masks."""
    held = sampling_variables.HeldVariables(Jp=residual)
    (blocks,) = sampling_variables.sampling_variables('entries', x, index, ('JtJp',), held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def ut3_probe_corewise_tangent(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: len=d, J_in_i+W+(Ni,); or packed (d,)+J+W+(N,)
        core_directions:   typ.Tuple[NDArray, NDArray],  # (B_dot, G_dot): supercores (d,)+K+C+(...), the point's sizes
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> Vectors:                               # J x_dot, MIRRORS ww: len=d, F+W+K+C+(Ni,); or packed
    """The corewise forward derivative on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.probing.t3_probe_corewise_tangent`, ``J x_dot`` along ``core_directions``. One call
    of the engine on ``x`` wanting ``Jp`` (its raw route); the engine masks the directions with the point's masks,
    so their padding is never read. **Mirrors** the sample's form, as :py:func:`ut3_probe`."""
    ragged_in = not ut3_operations.is_packed(ww)
    (Jp,) = sampling_variables.sampling_variables('probe', x, ww, ('Jp',), order=order,
                                                   core_directions=core_directions)
    return ut3_operations.unpack_vectors(Jp, x[2]) if ragged_in else Jp


def ut3_probe_corewise_transpose(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: len=d, J_in_i+W+(Ni,); or packed (d,)+J+W+(N,)
        residual:          Vectors,                      # of the type of z: len=d, F+W+[K]+C+(Ni,); or packed
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block, (d,)+[W]+[K]+C+(n,N)
        NDArray,                                         # TT block,     (d,)+[W]+[K]+C+(r,n,r)
]:
    """The corewise transpose on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.probing.t3_probe_corewise_transpose`, the gradient of ``<residual, z(x)>`` in the
    cores. The engine's ``JtJp`` with the residual held as ``Jp`` (a ragged residual packed here); the output is
    masked by the point's masks."""
    held = sampling_variables.HeldVariables(Jp=ut3_operations.pack_if_ragged(residual, x[0].shape[-1]))
    (blocks,) = sampling_variables.sampling_variables('probe', x, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


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
    ``sampling_derivatives.t3_entries_derivatives``; the upward jet slices Tucker fibers, so only ``pp`` is
    packed -- ``index`` in ``[0,Ni)`` hits the real prefix)."""
    masked = ut3_masking.ut3_apply_masks(data)
    packed_pp = ut3_operations.pack_if_ragged(pp, masked[0].shape[-1])
    return sampling_derivatives.t3_entries_derivatives(index, packed_pp, masked, order)


# ----------------------------------------------------------- corewise derivative transposes (jets)
# The jet-ified twins of the corewise transposes above: gradient of a plain-T3 derivative sampling op w.r.t.
# the frame's own supercores (the §6.3 (U,L,R)->G substitution, via the now-polymorphic
# sampling_derivatives.*_corewise_derivatives_transpose). Mask-once + pack ww/pp (entries: pp only) at the
# boundary; return the RAW gradient supercores (dB, dG), re-masked on exit (clean padding, as above). For a
# uniform core-wise optimizer.

def ut3_apply_corewise_derivatives_transpose(
        c:     NDArray,                # residual jet (scalar), shape=(order+1,)+W+K+C
        ww:    typ.Sequence[NDArray],  # apply vectors,          len=d, ith elm_shape=W+(Ni,)
        pp:    typ.Sequence[NDArray],  # perturbation vectors P, len=d, ith elm_shape=W+(Ni,)
        data:  UT3Data,
        order: int,                    # highest derivative order
        sum_over_probes: bool = False,
) -> typ.Tuple[NDArray, NDArray]:      # (tucker-core grad supercore, tt-core grad supercore)
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
    >>> KB, KG = ut3s.ut3_probe_corewise_derivatives_hessian(zt, ww, pp, u, dirty, order)
    >>> ref = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta.data, direction.data, order)
    >>> all(np.allclose(KG[i][:G.shape[0], :G.shape[1], :G.shape[2]], G) for i, G in enumerate(ref[1]))
    True
    >>> float(np.abs(KG * pad[1]).max()), float(np.abs(KB * pad[0]).max())   # the padding: exactly zero
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
    manifold frame ``(D, U, L, R)`` of ``theta = (D, L)`` (or the apply sweep at the corewise frame): its masked
    frame's ``D`` and ``L`` are ``theta`` masked with the frame's ``(down, left)`` masks, so ``data`` must carry those
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
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: len=d, J_in_i+W+(Ni,); or packed (d,)+J+W+(N,)
        residual:          Vectors,                      # c, of the type of z: len=d, F+W+C+(Ni,); or packed
        core_directions:   typ.Tuple[NDArray, NDArray],  # (B_dot, G_dot): supercores (d,)+K+C+(...), the point's sizes
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block, (d,)+[W]+K+C+(n,N)
        NDArray,                                         # TT block,     (d,)+[W]+K+C+(r,n,r)
]:
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
    >>> KB, KG = ut3s.ut3_probe_corewise_hessian(u, ww, zt, dirty)
    >>> ref = t3p.t3_probe_corewise_hessian(theta.data, ww, zt, direction.data)
    >>> all(np.allclose(KG[i][:G.shape[0], :G.shape[1], :G.shape[2]], G) for i, G in enumerate(ref[1]))
    True
    >>> float(np.abs(KG * pad[1]).max()), float(np.abs(KB * pad[0]).max())   # the padding: exactly zero
    (0.0, 0.0)
    """
    held = sampling_variables.HeldVariables(r=ut3_operations.pack_if_ragged(residual, x[0].shape[-1]))
    (blocks,) = sampling_variables.sampling_variables('probe', x, ww, ('Kp',), order=order, held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def ut3_probe_corewise_hessian_from_sweep(
        ztildes:          typ.Union[typ.Sequence[NDArray], NDArray],  # residuals (omega^2 applied); ragged, or packed (d,)+W+C+(N,)

        data:             UT3Data,                       # the point theta = the sweep frame's base point, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        frame_sweep:      typ.Tuple,                     # a uniform probe (or apply) sweep at a frame whose base point is theta
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """:py:func:`ut3_probe_corewise_hessian` reusing a uniform sweep's packed vectors and its ``xi`` / ``mu`` -- the
    reuse hook for a uniform local model, the twin of
    :py:func:`~t3toolbox.backend.probing.t3_probe_corewise_hessian_from_sweep`. ``frame_sweep`` is
    :py:func:`~t3toolbox.backend.utv_sampling.utv_precompute_probe_frame_sweep` at the manifold frame ``(D, U, L, R)``
    of ``theta = (D, L)`` (or a sweep at the corewise frame), so ``data`` must carry that frame's ``(down, left)`` masks
    (a uniform kind reads them off the frame it is handed); ``theta`` and ``theta_hat`` are masked with them and the
    output is re-masked."""
    _mb, packed_ww, _shape, psweep = frame_sweep
    masked, directions = _masked_point_and_directions(data, core_directions)
    packed_z = ut3_operations.pack_if_ragged(ztildes, masked[0].shape[-1])
    grads = probing.t3_probe_corewise_hessian_from_sweep(packed_z, packed_ww, masked, directions, psweep)
    return _remasked_core_gradient(grads, data)


def ut3_apply_corewise_hessian(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: len=d, J_in_i+W+(Ni,); or packed (d,)+J+W+(N,)
        residual:          NDArray,                      # c, of the type of z: shape=F+W+C; no weight
        core_directions:   typ.Tuple[NDArray, NDArray],  # (B_dot, G_dot): supercores (d,)+K+C+(...), the point's sizes
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block, (d,)+[W]+K+C+(n,N)
        NDArray,                                         # TT block,     (d,)+[W]+K+C+(r,n,r)
]:
    """The uniform twin of :py:func:`~t3toolbox.backend.apply.t3_apply_corewise_hessian` (``theta`` and ``theta_hat``
    masked with ``theta``'s masks, ``ww`` packed, the output re-masked), as :py:func:`ut3_probe_corewise_hessian`."""
    held = sampling_variables.HeldVariables(r=residual)
    (blocks,) = sampling_variables.sampling_variables('apply', x, ww, ('Kp',), order=order, held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def ut3_apply_corewise_hessian_from_sweep(
        c:                NDArray,                       # residual (omega^2 applied), shape=W+C
        data:             UT3Data,                       # the point theta = the sweep frame's base point, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        frame_sweep:      typ.Tuple,                     # utv_precompute_apply_frame_sweep at a frame whose base point is theta
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """:py:func:`ut3_apply_corewise_hessian` reusing a uniform apply sweep's packed vectors and lean ``(xi, mu)``
    (``data`` carries the sweep frame's ``(down, left)`` masks, as :py:func:`ut3_probe_corewise_hessian_from_sweep`)."""
    _mb, packed_ww, _shape, psweep = frame_sweep
    masked, directions = _masked_point_and_directions(data, core_directions)
    grads = apply.t3_apply_corewise_hessian_from_sweep(c, packed_ww, masked, directions, psweep)
    return _remasked_core_gradient(grads, data)


def ut3_entries_corewise_hessian(
        x:                 UT3Data,                      # the point's .data: (B, G, shape, masks), stack C
        index:             NDArray,                      # int, shape=(d,)+W: the grid points
        residual:          NDArray,                      # c, of the type of z: shape=W+C; no weight
        core_directions:   typ.Tuple[NDArray, NDArray],  # (B_dot, G_dot): supercores (d,)+K+C+(...), the point's sizes
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block, (d,)+[W]+K+C+(n,N)
        NDArray,                                         # TT block,     (d,)+[W]+K+C+(r,n,r)
]:
    """The uniform twin of :py:func:`~t3toolbox.backend.entries.t3_entries_corewise_hessian` (``index`` unpacked -- the
    one-hot vectors are built packed inside the kernel), as :py:func:`ut3_apply_corewise_hessian`."""
    held = sampling_variables.HeldVariables(r=residual)
    (blocks,) = sampling_variables.sampling_variables('entries', x, index, ('Kp',), held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def ut3_entries_corewise_hessian_from_sweep(
        c:                NDArray,                       # residual (omega^2 applied), shape=W+C
        data:             UT3Data,                       # the point theta = the sweep frame's base point, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        frame_sweep:      typ.Tuple,                     # utv_precompute_entries_frame_sweep at a frame whose base point is theta
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """:py:func:`ut3_entries_corewise_hessian` reusing a uniform entries sweep's index and lean (fiber-sliced)
    ``(xi, mu)`` (``data`` carries the sweep frame's ``(down, left)`` masks)."""
    _mb, index, _shape, psweep = frame_sweep
    masked, directions = _masked_point_and_directions(data, core_directions)
    grads = entries.t3_entries_corewise_hessian_from_sweep(c, index, masked, directions, psweep)
    return _remasked_core_gradient(grads, data)


# ----------------------------------------------- corewise two-holes: the apply / entries derivative Hessians
# The same wrapper, for the apply- and entries-derivative two-holes (the derivatives of
# ut3_{apply,entries}_corewise_derivatives_transpose along raw core directions): theta masked with its masks, theta_hat
# with the SAME masks, the vectors packed (entries: only pp -- the integer index is not packed, the one-hots are built
# packed inside the kernel), the polymorphic kernel run on the supercores, the output re-masked.

def ut3_apply_corewise_derivatives_hessian(
        c:                NDArray,                       # residual jet (omega^2 applied), shape=(order+1,)+W+C

        ww:               typ.Union[typ.Sequence[NDArray], NDArray],  # apply vectors X, len=d, elm_shape=W+(Ni,); or packed
        pp:               typ.Union[typ.Sequence[NDArray], NDArray],  # perturbation vectors P, likewise

        data:             UT3Data,                       # the point theta, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        order:            int,                           # highest derivative order
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """The uniform twin of
    :py:func:`~t3toolbox.backend.sampling_derivatives.t3_apply_corewise_derivatives_hessian`: the core-coordinate
    Hessian of ``theta -> <c, S(Phi(theta))>`` applied to ``theta_hat``, for ``S`` the apply-derivative jets, on
    supercores. ``theta`` is masked with its masks, ``theta_hat`` with the SAME masks, ``ww`` / ``pp`` packed to ``N``,
    and the output re-masked; per stack element the real block is the ragged two-hole of the real blocks, whatever
    the padding holds.

    Examples
    --------
    With garbage in every padded slot of the direction, the real block is the ragged two-hole's and the padding is
    exactly zero:

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
    >>> c = rng.standard_normal((order + 1, W))
    >>> u = ut3.UniformTuckerTensorTrain.from_t3(theta, N=6, n=4, r=4).data
    >>> ud = ut3.UniformTuckerTensorTrain.from_t3(direction, N=6, n=4, r=4).data
    >>> pad = [1.0 - m for m in ut3m.ut3_apply_masks((np.ones_like(ud[0]), np.ones_like(ud[1]), ud[2], ud[3]))]
    >>> dirty = (ud[0] + 1e3 * pad[0], ud[1] + 1e3 * pad[1])               # garbage in the direction's padding
    >>> KB, KG = ut3s.ut3_apply_corewise_derivatives_hessian(c, ww, pp, u, dirty, order)
    >>> ref = sd.t3_apply_corewise_derivatives_hessian(c, ww, pp, theta.data, direction.data, order)
    >>> all(np.allclose(KG[i][:G.shape[0], :G.shape[1], :G.shape[2]], G) for i, G in enumerate(ref[1]))
    True
    >>> float(np.abs(KG * pad[1]).max()), float(np.abs(KB * pad[0]).max())   # the padding: exactly zero
    (0.0, 0.0)
    """
    masked, directions = _masked_point_and_directions(data, core_directions)
    N = masked[0].shape[-1]
    grads = sampling_derivatives.t3_apply_corewise_derivatives_hessian(
        c, ut3_operations.pack_if_ragged(ww, N), ut3_operations.pack_if_ragged(pp, N), masked, directions, order)
    return _remasked_core_gradient(grads, data)


def ut3_apply_corewise_derivatives_hessian_from_sweep(
        c:                NDArray,                       # residual jet (omega^2 applied), shape=(order+1,)+W+C
        data:             UT3Data,                       # the point theta = the sweep frame's base point, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        order:            int,                           # highest derivative order
        frame_sweep:      typ.Tuple,                     # utv_precompute_apply_frame_sweep_jets at a frame whose base point is theta
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """:py:func:`ut3_apply_corewise_derivatives_hessian` reusing a uniform apply jet sweep's packed ``ww`` / ``pp`` and
    its lean ``(xi, mu)`` jets -- the reuse hook for a uniform local model, the twin of
    :py:func:`~t3toolbox.backend.sampling_derivatives.t3_apply_corewise_derivatives_hessian_from_sweep`. ``data``
    carries the sweep frame's ``(down, left)`` masks (a uniform kind reads them off the frame it is handed; at TT slack
    the point's own masks are a strict superset), as :py:func:`ut3_probe_corewise_derivatives_hessian_from_sweep`."""
    _mb, packed_ww, packed_pp, _shape, psweep = frame_sweep
    masked, directions = _masked_point_and_directions(data, core_directions)
    grads = sampling_derivatives.t3_apply_corewise_derivatives_hessian_from_sweep(
        c, packed_ww, packed_pp, masked, directions, order, psweep)
    return _remasked_core_gradient(grads, data)


def ut3_entries_corewise_derivatives_hessian(
        c:                NDArray,                       # residual jet (omega^2 applied), shape=(order+1,)+W+C
        index:            NDArray,                       # int, shape=(d,)+W

        pp:               typ.Union[typ.Sequence[NDArray], NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,); or packed

        data:             UT3Data,                       # the point theta, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        order:            int,                           # highest derivative order
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """The uniform twin of
    :py:func:`~t3toolbox.backend.sampling_derivatives.t3_entries_corewise_derivatives_hessian` (``index`` unpacked --
    the one-hot vectors are built packed inside the kernel; ``pp`` packed), as
    :py:func:`ut3_apply_corewise_derivatives_hessian`."""
    masked, directions = _masked_point_and_directions(data, core_directions)
    grads = sampling_derivatives.t3_entries_corewise_derivatives_hessian(
        c, index, ut3_operations.pack_if_ragged(pp, masked[0].shape[-1]), masked, directions, order)
    return _remasked_core_gradient(grads, data)


def ut3_entries_corewise_derivatives_hessian_from_sweep(
        c:                NDArray,                       # residual jet (omega^2 applied), shape=(order+1,)+W+C
        data:             UT3Data,                       # the point theta = the sweep frame's base point, stack C
        core_directions:  typ.Tuple[NDArray, NDArray],   # bare raw direction supercores theta_hat, (d,)+K+C+(...)
        order:            int,                           # highest derivative order
        frame_sweep:      typ.Tuple,                     # utv_precompute_entries_frame_sweep_jets at a frame whose base point is theta
) -> typ.Tuple[NDArray, NDArray]:                        # (Tucker, TT) blocks of K theta_hat, (d,)+K+C+(...), W summed
    """:py:func:`ut3_entries_corewise_derivatives_hessian` reusing a uniform entries jet sweep's index, packed ``pp``
    and lean (fiber-sliced) ``(xi, mu)`` jets (``data`` carries the sweep frame's ``(down, left)`` masks)."""
    _mb, index, packed_pp, _shape, psweep = frame_sweep
    masked, directions = _masked_point_and_directions(data, core_directions)
    grads = sampling_derivatives.t3_entries_corewise_derivatives_hessian_from_sweep(
        c, index, packed_pp, masked, directions, order, psweep)
    return _remasked_core_gradient(grads, data)
