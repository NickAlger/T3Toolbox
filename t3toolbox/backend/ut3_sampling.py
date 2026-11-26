# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform plain-layer sampling: probe/apply/entries on ut3 data, their corewise derivative, transpose and
Hessian, and the full sum.

Grouped by OBJECT type, deliberately asymmetric with the ragged by-sampling-type modules
(``docs/naming_conventions.md``). The twins of the ragged sampling functions (``ut3_probe``, ``ut3_apply``,
``ut3_entries`` and their ``*_corewise_tangent``, ``*_corewise_transpose``, ``*_corewise_hessian``) take the
ragged functions' signatures, with ``x`` the uniform point's ``.data``, and are one call of the engine
(:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`), which packs a ragged sample, masks every
operand on entry and every output on exit. A probe output mirrors the sample's form: ragged in, ragged out;
packed in, packed out.
"""
import typing as typ

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
    'ut3_probe_corewise_hessian',
    'ut3_apply_corewise_hessian',
    'ut3_entries_corewise_hessian',
]

# .data[2] is the static int-tuple shape; .data[3] = (tucker_edge_mask, tt_edge_mask) are HOST bool,
# static structure (numpy, never traced); the supercores are xnp.
UT3Data = typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]]
Vectors = typ.Union[typ.Sequence[NDArray], NDArray]   # one vector per mode: ragged, len=d; or packed, (d,)+...+(N,)


def ut3_entries(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
) -> NDArray:                               # z: ℝ, shape=W+C
    """Entries of a uniform Tucker tensor train at ``index``: the uniform twin of
    :py:func:`~t3toolbox.backend.entries.t3_entries`, one call of the engine on ``x`` wanting ``z``."""
    (z,) = sampling_variables.sampling_variables('entries', x, index, ('z',))
    return z


def ut3_apply(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # z: P, shape=F+W+C
    """Contract a uniform Tucker tensor train with the input jets in all modes: the uniform twin of
    :py:func:`~t3toolbox.backend.apply.t3_apply`, one call of the engine on ``x`` wanting ``z`` (ragged or packed
    ``ww``; the output is one array per sample either way)."""
    (z,) = sampling_variables.sampling_variables('apply', x, ww, ('z',), order=order)
    return z


def ut3_probe(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> Vectors:                               # z: P⊗V_i*, MIRRORS ww: len=d, F+W+C+(Ni,); or packed
    """Probe a uniform Tucker tensor train (all modes but one contracted, for each mode in turn): the uniform twin of
    :py:func:`~t3toolbox.backend.probing.t3_probe`, one call of the engine on ``x`` wanting ``z``. **Mirrors** the
    sample's form: a ragged ``ww`` gives a ``len=d`` tuple sliced to the real widths, a packed ``ww`` the packed
    array (zero padding). The engine packs a ragged ``ww`` (``len_P=len(order)``) and masks on entry and exit."""
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


# ----------------------------------------------------------------- corewise (non-manifold) tangents and transposes
# The forward derivative of a sampling op along raw directions in the point's own supercores, and its gradient
# w.r.t. them (for a uniform core-wise optimizer: Adam, L-BFGS). The transposes return the RAW gradient
# supercores (dB, dG) -- the uniform mirror of the ragged TuckerTensorTrain.*_corewise_transpose raw-tuple
# return -- with their padding exactly zero: the engine masks every input on entry and every output on exit.
# The equivalence contract still declares the padding don't-care (every consumer masks on entry --
# ut3_squash_tails included, since 2026-08-22), and the masks, not the values, govern rank.


def ut3_apply_corewise_tangent(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        core_directions:   typ.Tuple[NDArray, NDArray],  # X, (B_dot, G_dot): supercores (d,)+K+C+(...), the point's
                                                         #   sizes
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # J x_dot: P, shape=F+W+K+C
    """The corewise forward derivative on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.apply.t3_apply_corewise_tangent`, ``J x_dot`` along ``core_directions``. One call
    of the engine on ``x`` wanting ``Jp`` (its raw route); the engine masks the directions with the point's masks,
    so their padding is never read."""
    (Jp,) = sampling_variables.sampling_variables('apply', x, ww, ('Jp',), order=order,
                                                   core_directions=core_directions)
    return Jp


def ut3_apply_corewise_transpose(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        c:                 NDArray,                      # P*, shape=F+W+[K]+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block: X*, (d,)+[W]+[K]+C+(n,N)
        NDArray,                                         # TT block:     X*, (d,)+[W]+[K]+C+(r,n,r)
]:
    """The corewise transpose on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.apply.t3_apply_corewise_transpose`, the gradient of ``<c, z(x)>`` in the cores (``c``
    an element of ``Y``, with no metric, as there). The engine's ``JtJp`` with ``c`` held as ``Jp``; the output is
    masked by the point's masks."""
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('apply', x, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def ut3_entries_corewise_tangent(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
        core_directions:   typ.Tuple[NDArray, NDArray],  # X, (B_dot, G_dot): supercores (d,)+K+C+(...), the point's
                                                         #   sizes
) -> NDArray:                               # J x_dot: ℝ, shape=W+K+C
    """The corewise forward derivative on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.entries.t3_entries_corewise_tangent`, ``J x_dot`` along ``core_directions``. One
    call of the engine on ``x`` wanting ``Jp`` (its raw route); the engine masks the directions with the point's
    masks, so their padding is never read."""
    (Jp,) = sampling_variables.sampling_variables('entries', x, index, ('Jp',),
                                                   core_directions=core_directions)
    return Jp


def ut3_entries_corewise_transpose(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
        c:                 NDArray,                      # ℝ, shape=W+[K]+C
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block: X*, (d,)+[W]+[K]+C+(n,N)
        NDArray,                                         # TT block:     X*, (d,)+[W]+[K]+C+(r,n,r)
]:
    """The corewise transpose on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.entries.t3_entries_corewise_transpose`, the gradient of ``<c, z(x)>`` in the cores
    (``c`` an element of ``Y``, with no metric, as there). The engine's ``JtJp`` with ``c`` held as ``Jp``; the output
    is masked by the point's masks."""
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('entries', x, index, ('JtJp',), held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def ut3_probe_corewise_tangent(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        core_directions:   typ.Tuple[NDArray, NDArray],  # X, (B_dot, G_dot): supercores (d,)+K+C+(...), the point's
                                                         #   sizes
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> Vectors:                               # J x_dot: P⊗V_i*, MIRRORS ww: len=d, F+W+K+C+(Ni,); or packed
    """The corewise forward derivative on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.probing.t3_probe_corewise_tangent`, ``J x_dot`` along ``core_directions``. One call
    of the engine on ``x`` wanting ``Jp`` (its raw route); the engine masks the directions with the point's masks,
    so their padding is never read. **Mirrors** the sample's form, as :py:func:`ut3_probe`."""
    ragged_in = not ut3_operations.is_packed(ww)
    (Jp,) = sampling_variables.sampling_variables('probe', x, ww, ('Jp',), order=order,
                                                   core_directions=core_directions)
    return ut3_operations.unpack_vectors(Jp, x[2]) if ragged_in else Jp


def ut3_probe_corewise_transpose(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        c:                 Vectors,                      # P*⊗V_i, len=d, F+W+[K]+C+(Ni,); or packed
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block: X*, (d,)+[W]+[K]+C+(n,N)
        NDArray,                                         # TT block:     X*, (d,)+[W]+[K]+C+(r,n,r)
]:
    """The corewise transpose on supercores: the uniform twin of
    :py:func:`~t3toolbox.backend.probing.t3_probe_corewise_transpose`, the gradient of ``<c, z(x)>`` in the cores (``c``
    an element of ``Y``, with no metric, as there). The engine's ``JtJp`` with ``c`` held as ``Jp`` (a ragged ``c``
    packed here); the output is masked by the point's masks."""
    held = sampling_variables.HeldVariables(Jp=ut3_operations.pack_if_ragged(c, x[0].shape[-1]))
    (blocks,) = sampling_variables.sampling_variables('probe', x, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


# ----------------------------------------------------------- corewise two-holes: the plain kinds' Hessians
# The derivatives of ut3_*_corewise_transpose along raw core directions, one engine call each (Kp): theta and
# theta_hat masked with theta's masks on entry, the output masked on exit.


def ut3_probe_corewise_hessian(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        c:                 Vectors,                      # P*⊗V_i, len=d, F+W+C+(Ni,); or packed
        core_directions:   typ.Tuple[NDArray, NDArray],  # X, (B_dot, G_dot): supercores (d,)+K+C+(...), the point's
                                                         #   sizes
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block: X*, (d,)+[W]+K+C+(n,N)
        NDArray,                                         # TT block:     X*, (d,)+[W]+K+C+(r,n,r)
]:
    """The uniform twin of :py:func:`~t3toolbox.backend.probing.t3_probe_corewise_hessian`: the core-coordinate Hessian
    of ``theta -> <c, probe(Phi(theta))>`` applied to ``theta_hat``, on supercores (``c`` an element of ``Y``, with no
    metric, as there). ``theta`` is masked with its masks, ``theta_hat`` with the SAME masks, ``c`` / ``ww`` packed to
    ``N``, and the output re-masked; per stack element the real block is the ragged two-hole of the real blocks,
    whatever the padding holds.

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
    held = sampling_variables.HeldVariables(r=ut3_operations.pack_if_ragged(c, x[0].shape[-1]))
    (blocks,) = sampling_variables.sampling_variables('probe', x, ww, ('Kp',), order=order, held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def ut3_apply_corewise_hessian(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        c:                 NDArray,                      # P*, shape=F+W+C
        core_directions:   typ.Tuple[NDArray, NDArray],  # X, (B_dot, G_dot): supercores (d,)+K+C+(...), the point's
                                                         #   sizes
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block: X*, (d,)+[W]+K+C+(n,N)
        NDArray,                                         # TT block:     X*, (d,)+[W]+K+C+(r,n,r)
]:
    """The uniform twin of :py:func:`~t3toolbox.backend.apply.t3_apply_corewise_hessian` (``theta`` and ``theta_hat``
    masked with ``theta``'s masks, ``ww`` packed, the output re-masked), as :py:func:`ut3_probe_corewise_hessian`."""
    held = sampling_variables.HeldVariables(r=c)
    (blocks,) = sampling_variables.sampling_variables('apply', x, ww, ('Kp',), order=order, held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def ut3_entries_corewise_hessian(
        x:                 UT3Data,                      # the point: X, its .data (B, G, shape, masks), stack C
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
        c:                 NDArray,                      # ℝ, shape=W+C
        core_directions:   typ.Tuple[NDArray, NDArray],  # X, (B_dot, G_dot): supercores (d,)+K+C+(...), the point's
                                                         #   sizes
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        NDArray,                                         # Tucker block: X*, (d,)+[W]+K+C+(n,N)
        NDArray,                                         # TT block:     X*, (d,)+[W]+K+C+(r,n,r)
]:
    """The uniform twin of :py:func:`~t3toolbox.backend.entries.t3_entries_corewise_hessian` (``index`` unpacked -- the
    one-hot vectors are built packed inside the kernel), as :py:func:`ut3_apply_corewise_hessian`."""
    held = sampling_variables.HeldVariables(r=c)
    (blocks,) = sampling_variables.sampling_variables('entries', x, index, ('Kp',), held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks

