# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform tangent sampling (UT3Tangent probe / apply / entries).

The uniform mirror of the ragged ``T3Tangent`` sampling frontend. The functions ``utv_{probe,apply,entries}`` and
their transposes take the ragged ``tv_*`` signatures and are one call of the engine
(:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`) on the frame's six elements: the engine masks
the frame and **the variations with the frame's masks** (the variations' own masks are not read; a uniform
tangent's variation masks are the frame's gauge masks), packs the sample, and masks every output; ``utv_probe``
mirrors the sample's form.

These are the bare Jacobian ``𝒥`` (no gauge projector ``Π``) and its transpose ``𝒥ᵀ``. The transposes attach the
result tangent's variation masks (the frame's gauge masks over the new tangent stack) and **re-mask their
output** with them, so the padded slots come back as exact zeros (one multiply by a host mask; the real block is
unchanged bit for bit).
"""
import typing as typ

import t3toolbox.backend.ufv_masking as ufv_masking
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.sampling_variables as sampling_variables
from t3toolbox.backend.common import *

Vectors = typ.Union[typ.Sequence[NDArray], NDArray]   # one vector per mode: ragged, len=d; or packed, (d,)+...+(N,)

__all__ = [
    'utv_probe',
    'utv_apply',
    'utv_entries',
    'utv_probe_transpose',
    'utv_apply_transpose',
    'utv_entries_transpose',
]


def utv_probe(
        frame:             typ.Tuple,                    # the base point: UT3Frame .data (D, U, L, R, shape, masks),
                                                         #   stack C
        variations:        typ.Tuple,                    # X, UT3Variations .data: (V, H) read, the frame's masks
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> Vectors:                               # Jp: P⊗V_i*, MIRRORS ww: len=d, F+W+K+C+(Ni,); or packed
    """The uniform twin of :py:func:`~t3toolbox.backend.probing.tv_probe` (the bare ``J``, no gauge projector): one
    call of the engine on the frame wanting ``Jp``. Only ``variations[:2]`` is read, masked by the FRAME's masks (a
    uniform tangent's variation masks are derived from its frame's; their own are not read). **Mirrors** the
    sample's form: ragged ``ww`` in, ``len=d`` tuple out; packed in, packed out."""
    ragged_in = not ut3_operations.is_packed(ww)
    (Jp,) = sampling_variables.sampling_variables('probe', frame, ww, ('Jp',), order=order,
                                                   variations=tuple(variations[:2]))
    return ut3_operations.unpack_vectors(Jp, frame[4]) if ragged_in else Jp


def utv_apply(
        frame:             typ.Tuple,                    # the base point: UT3Frame .data (D, U, L, R, shape, masks),
                                                         #   stack C
        variations:        typ.Tuple,                    # X, UT3Variations .data: (V, H) read, the frame's masks
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> NDArray:                               # Jp: P, shape=F+W+K+C
    """The uniform twin of :py:func:`~t3toolbox.backend.apply.tv_apply` (the bare ``J``, no gauge projector): one
    call of the engine on the frame wanting ``Jp``. Only ``variations[:2]`` is read, masked by the FRAME's masks (a
    uniform tangent's variation masks are derived from its frame's; their own are not read)."""
    (Jp,) = sampling_variables.sampling_variables('apply', frame, ww, ('Jp',), order=order,
                                                   variations=tuple(variations[:2]))
    return Jp


def utv_entries(
        frame:             typ.Tuple,                    # the base point: UT3Frame .data (D, U, L, R, shape, masks),
                                                         #   stack C
        variations:        typ.Tuple,                    # X, UT3Variations .data: (V, H) read, the frame's masks
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
) -> NDArray:                               # Jp: ℝ, shape=W+K+C
    """The uniform twin of :py:func:`~t3toolbox.backend.entries.tv_entries` (the bare ``J``, no gauge projector):
    one call of the engine on the frame wanting ``Jp``. Only ``variations[:2]`` is read, masked by the FRAME's masks
    (a uniform tangent's variation masks are derived from its frame's; their own are not read)."""
    (Jp,) = sampling_variables.sampling_variables('entries', frame, index, ('Jp',),
                                                   variations=tuple(variations[:2]))
    return Jp


# ----------------------------------------------------------------- the transpose 𝒥ᵀ (probe, apply, entries)
def _gauge_masks_over_Knew(
        frame_data,      # UT3Frame .data
        out_supercore,   # a transpose-output variation supercore (d,)+K_new+C+(...), pins the stack
):  # -> the 4 variation rank masks, each (d,)+K_new+C+(size,)
    """The result tangent's variation masks: the frame's gauge-shifted variation masks
    ``(down, up, frame_left[:-1], frame_right[1:])`` broadcast (constant) over the new tangent stack
    ``K_new`` -- the leading stack of the transpose output minus the frame stack ``C``. ``sum_over_samples``
    determines ``K_new`` (``W+K`` kept / ``K`` summed); we read it off the output supercore rather than
    re-deriving it. Masks are host numpy (static aux), so this stays on ``np``."""
    return ufv_masking.ufv_variation_masks_over_stack(frame_data[5], frame_data[0].shape[1:-2],
                                                      out_supercore.shape[1:-2])


def _remasked_transpose_output(
        frame_data,   # UT3Frame .data
        V_tilde,      # bare transpose-output tucker variation supercore, (d,)+K_new+C+(nU, N)
        H_tilde,      # bare transpose-output tt variation supercore,     (d,)+K_new+C+(rL, nD, rR)
):  # -> UT3Variations .data: (V_tilde, H_tilde, shape, masks), padding exactly zero
    """Attach the result tangent's variation masks (:py:func:`_gauge_masks_over_Knew`) AND re-mask the
    supercores with them, so the padded slots come back as exact zeros rather than the raw sweep's garbage
    (the boundary-bond squash, for one, broadcasts into the padded boundary slots). One multiply by a host
    mask: the real block is unchanged bit for bit, and under jit the masks are folded constants."""
    masks = _gauge_masks_over_Knew(frame_data, V_tilde)
    V_tilde, H_tilde = ufv_masking.ufv_apply_variations_masks((V_tilde, H_tilde, frame_data[4], masks))
    return (V_tilde, H_tilde, frame_data[4], masks)


def utv_probe_transpose(
        frame:             typ.Tuple,                    # the base point: UT3Frame .data (D, U, L, R, shape, masks),
                                                         #   stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        c:                 Vectors,                      # P*⊗V_i, len=d, F+W+[K]+C+(Ni,); or packed
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
):  # -> X*: UT3Variations .data (V_tilde, H_tilde, shape, masks); stack K_new + C
    """The uniform twin of :py:func:`~t3toolbox.backend.probing.tv_probe_transpose` (the bare ``J^T``, no gauge
    projector; ``c`` an element of ``Y``, with no metric, as there): the engine's ``JtJp`` with ``c`` held as ``Jp`` (a
    ragged ``c`` packed here), then the result variation masks attached, the frame's gauge masks broadcast over the new
    tangent stack ``K_new`` (``W+K`` if ``sum_over_samples=False``, else ``K``), and the output re-masked with them
    (exact zero padding)."""
    held = sampling_variables.HeldVariables(Jp=ut3_operations.pack_if_ragged(c, frame[0].shape[-1]))
    ((V_tilde, H_tilde),) = sampling_variables.sampling_variables('probe', frame, ww, ('JtJp',), order=order, held=held,
                                                                   sum_over_samples=sum_over_samples)
    return _remasked_transpose_output(frame, V_tilde, H_tilde)


def utv_apply_transpose(
        frame:             typ.Tuple,                    # the base point: UT3Frame .data (D, U, L, R, shape, masks),
                                                         #   stack C
        ww:                Vectors,                      # input jets: P_in_i⊗V_i, len=d, F_in_i+W+(Ni,); or packed
                                                         #   (d,)+F_in+W+(N,)
        c:                 NDArray,                      # P*, shape=F+W+[K]+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
):  # -> X*: UT3Variations .data (V_tilde, H_tilde, shape, masks); stack K_new + C
    """The uniform twin of :py:func:`~t3toolbox.backend.apply.tv_apply_transpose` (the bare ``J^T``, no gauge projector;
    ``c`` an element of ``Y``, with no metric, as there): the engine's ``JtJp`` with ``c`` held as ``Jp``, then the
    result variation masks attached, the frame's gauge masks broadcast over the new tangent stack ``K_new`` (``W+K`` if
    ``sum_over_samples=False``, else ``K``), and the output re-masked with them (exact zero padding)."""
    held = sampling_variables.HeldVariables(Jp=c)
    ((V_tilde, H_tilde),) = sampling_variables.sampling_variables('apply', frame, ww, ('JtJp',), order=order, held=held,
                                                                   sum_over_samples=sum_over_samples)
    return _remasked_transpose_output(frame, V_tilde, H_tilde)


def utv_entries_transpose(
        frame:             typ.Tuple,                    # the base point: UT3Frame .data (D, U, L, R, shape, masks),
                                                         #   stack C
        index:             NDArray,                      # index^d, shape=(d,)+W, int: the grid points
        c:                 NDArray,                      # ℝ, shape=W+[K]+C
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
):  # -> X*: UT3Variations .data (V_tilde, H_tilde, shape, masks); stack K_new + C
    """The uniform twin of :py:func:`~t3toolbox.backend.entries.tv_entries_transpose` (the bare ``J^T``, no gauge
    projector; ``c`` an element of ``Y``, with no metric, as there): the engine's ``JtJp`` with ``c`` held as ``Jp``,
    then the result variation masks attached, the frame's gauge masks broadcast over the new tangent stack ``K_new``
    (``W+K`` if ``sum_over_samples=False``, else ``K``), and the output re-masked with them (exact zero padding)."""
    held = sampling_variables.HeldVariables(Jp=c)
    ((V_tilde, H_tilde),) = sampling_variables.sampling_variables('entries', frame, index, ('JtJp',), held=held,
                                                                   sum_over_samples=sum_over_samples)
    return _remasked_transpose_output(frame, V_tilde, H_tilde)

