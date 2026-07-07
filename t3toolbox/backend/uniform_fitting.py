# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Uniform-layer fitting seams: the ``GeometryOps`` factories (and, later, the ``SamplingKind`` builders)
that let the geometry-generic optimizers (:py:mod:`t3toolbox.backend.optimizers`) run on **uniform
supercores**. The uniform twin of :py:mod:`t3toolbox.backend.fitting` -- optimizers-on-uniform slice U2.

The optimizers are written against three pluggable seams (``GeometryOps``, ``SamplingKind``,
``corewise``) and never mention a layer; supplying uniform implementations of the seams runs the SAME
algorithm bodies on uniform data. This module supplies the **geometry** half.

The mask-holding recipe (``docs/uniform_backend_jit_recipe.md``). A uniform ``.data`` tuple carries host-
numpy **masks** that cannot be jit-traced. So -- unlike the stateless ragged ``MANIFOLD`` / ``COREWISE``
singletons -- the uniform geometry is a **factory** that captures the loop-invariant masks (fixed at
fixed rank) and closes over them; the optimizer traces only the supercores. The convention throughout:

  * the optimizer's **point** ``x`` is a bare supercore pair ``(tucker_sc, tt_sc)`` (masks closed over);
  * a **tangent** is a bare variation supercore pair ``(tucker_var_sc, tt_var_sc)``;
  * a **base** is the full frame ``.data`` (``base`` returns it, ``project`` / ``retract`` consume it).

Each closure re-attaches the held ``shape`` + masks at the boundary (building the full ``.data`` tuples
the ``ubv_tangent_operations`` primitives expect) and strips them back to a bare pair on output, so
``corewise.*`` and ``GeometryOps.inner`` see only supercores. Under jit the re-derived frame masks
constant-fold to device constants (the "1 compile" behaviour of the recipe).
"""
import dataclasses as dc
import typing as typ

from t3toolbox.backend import optimizers as bopt
from t3toolbox.backend import fitting as bfit
from t3toolbox.backend import ubv_conversions
from t3toolbox.backend import ubv_tangent_operations as ubv_tops
from t3toolbox.backend import ubv_sampling
from t3toolbox.backend import ut3_sampling
from t3toolbox.backend.common import *

__all__ = [
    'uniform_manifold_ops',
    'uniform_corewise_ops',
    'uniform_geometry_ops',
    'uniform_apply_kind',
    'uniform_entries_kind',
    'uniform_probe_kind',
    'uniform_sampling_kind',
]


def uniform_manifold_ops(
        x0_data:  typ.Tuple,   # UniformTuckerTensorTrain.data = (tk_sc, tt_sc, shape, (tucker_mask, tt_mask))
) -> bopt.GeometryOps:         # the uniform manifold geometry ops (masks of x0's fixed rank closed over)
    """The uniform **manifold** ``GeometryOps`` at ``x0``'s fixed rank -- the raw-supercore twin of
    :py:data:`t3toolbox.uniform_manifold.UNIFORM_MANIFOLD`.

    ``base`` = the orthonormal frame (:py:func:`ubv_conversions.ut3_orthogonal_representations`); ``project``
    = the gauge projection ``Pi``; ``retract`` = the manifold (doubled-rank, mask-truncated) retraction;
    ``inner`` = the masked coordinate dot (:py:func:`ubv_tangent_operations.ubv_corewise_inner`, the
    check-free twin of ``UNIFORM_MANIFOLD.inner`` -- equal to Hilbert-Schmidt on this orthonormal, gauged
    frame). All masks are loop-invariant at ``x0``'s rank and closed over here.
    """
    tucker_sc, _tt_sc, shape, base_masks = x0_data
    n_stack = tucker_sc.ndim - 3     # |C| base stack (0 for a single tensor); tucker sc = (d,) + C + (nU, N)
    _frame, var_data = ubv_conversions.ut3_orthogonal_representations(x0_data)
    var_masks = var_data[3]          # (up, down, left[:-1], right[1:]) -- fixed at this rank; for `inner`

    def base(x_sc):                                          # (tk_sc, tt_sc) -> orthonormal frame .data
        return ubv_conversions.ut3_orthogonal_representations(
            (x_sc[0], x_sc[1], shape, base_masks))[0]

    def project(basis_data, var_sc):                        # gauge Pi; bare variation pair in and out
        gauged = ubv_tops.orthogonal_gauge_projection(
            basis_data, (var_sc[0], var_sc[1], shape, var_masks))
        return (gauged[0], gauged[1])

    def retract(basis_data, var_sc):                        # manifold retraction -> bare point pair
        new_x = ubv_tops.retract(basis_data, (var_sc[0], var_sc[1], shape, var_masks))
        return (new_x[0], new_x[1])

    def inner(a_sc, b_sc):                                  # masked coordinate <.,.> over the bare pairs
        return ubv_tops.ubv_corewise_inner(
            (a_sc[0], a_sc[1], shape, var_masks), (b_sc[0], b_sc[1], shape, var_masks), n_stack)

    return bopt.GeometryOps(base=base, project=project, retract=retract, inner=inner)


def uniform_corewise_ops(
        x0_data:  typ.Tuple,   # UniformTuckerTensorTrain.data = (tk_sc, tt_sc, shape, (tucker_mask, tt_mask))
) -> bopt.GeometryOps:         # the uniform corewise geometry ops (masks of x0's fixed rank closed over)
    """The uniform **corewise** ``GeometryOps`` at ``x0``'s fixed rank -- the raw-supercore twin of
    :py:data:`t3toolbox.uniform_manifold.UNIFORM_COREWISE`.

    ``base`` = the ``(U, G, G, G)`` non-orthonormal frame (Section 6.3 ``(P, Q, O) -> G``); ``project`` =
    the identity (no gauge on the core space); ``retract`` = the additive retraction (``cores += var``);
    ``inner`` = the masked coordinate dot (the Euclidean metric here). Masks are loop-invariant and closed
    over.
    """
    tucker_sc, _tt_sc, shape, (tucker_mask, tt_mask) = x0_data
    n_stack = tucker_sc.ndim - 3
    var_masks = (tucker_mask, tucker_mask, tt_mask[:-1], tt_mask[1:])   # _variation_masks_of the (U,G,G,G) frame

    def base(x_sc):                                         # (tk_sc, tt_sc) -> the (U, G, G, G) frame .data
        return (x_sc[0], x_sc[1], x_sc[1], x_sc[1], shape,
                (tucker_mask, tucker_mask, tt_mask, tt_mask))

    def project(basis_data, var_sc):                       # identity (Euclidean core space, no gauge)
        return var_sc

    def retract(basis_data, var_sc):                       # additive: cores += var -> bare point pair
        new_x = ubv_tops.corewise_retract(basis_data, (var_sc[0], var_sc[1], shape, var_masks))
        return (new_x[0], new_x[1])

    def inner(a_sc, b_sc):
        return ubv_tops.ubv_corewise_inner(
            (a_sc[0], a_sc[1], shape, var_masks), (b_sc[0], b_sc[1], shape, var_masks), n_stack)

    return bopt.GeometryOps(base=base, project=project, retract=retract, inner=inner)


def uniform_geometry_ops(
        kind:     str,        # 'manifold' or 'corewise'
        x0_data:  typ.Tuple,  # UniformTuckerTensorTrain.data at the fixed rank to optimize over
) -> bopt.GeometryOps:
    """Dispatch to :py:func:`uniform_manifold_ops` / :py:func:`uniform_corewise_ops` by name."""
    if kind == 'manifold':
        return uniform_manifold_ops(x0_data)
    if kind == 'corewise':
        return uniform_corewise_ops(x0_data)
    raise ValueError(f"unknown uniform geometry kind {kind!r}; expected 'manifold' or 'corewise'")


# --------------------------------------------------------------------------------------------------
# SamplingKind builders -- the uniform twins of backend.fitting.{APPLY,ENTRIES,PROBE}. Only the four
# layer-specific fields (precompute / forward / transpose / point_forward) differ; the layer-agnostic
# reductions and default-draw layout (sumsq / w_axes / n_measurements / take) are reused verbatim from
# the ragged kind via `dataclasses.replace`. `forward` derives the variation masks from the frame it is
# handed (the `(up, down, left[:-1], right[1:])` gauge shift -- valid for BOTH the orthonormal manifold
# frame and the (U,G,G,G) corewise frame), so the kind is geometry-agnostic; only `point_forward` (the
# S(x) op on the plain-UT3 point) closes over the plain-UT3 shape + edge masks.
# --------------------------------------------------------------------------------------------------
def _var_masks_from_base(base_data):
    """The variation masks of a frame: the basis's gauge-shifted rank masks
    ``(up, down, basis_left[:-1], basis_right[1:])`` (mirrors ``UT3Variations._variation_masks_of``)."""
    up_mask, down_mask, basis_left_mask, basis_right_mask = base_data[5]
    return (up_mask, down_mask, basis_left_mask[:-1], basis_right_mask[1:])


def uniform_apply_kind(
        x0_data:  typ.Tuple,   # UniformTuckerTensorTrain.data = (tk_sc, tt_sc, shape, (tucker_mask, tt_mask))
) -> bfit.SamplingKind:        # the uniform all-modes `apply` sampling kind at x0's fixed rank
    """The uniform **apply** ``SamplingKind`` -- the twin of :py:data:`t3toolbox.backend.fitting.APPLY`."""
    _tk, _tt, shape, base_masks = x0_data
    return dc.replace(
        bfit.APPLY,
        precompute=lambda base_data, ww: ubv_sampling.ut3tangent_precompute_apply_base_sweep(base_data, ww),
        forward=lambda v_sc, ww, base_data, sweep: ubv_sampling.ut3tangent_apply_jacobian_from_sweep(
            (v_sc[0], v_sc[1], base_data[4], _var_masks_from_base(base_data)), sweep),
        transpose=lambda r, ww, base_data, sweep: ubv_sampling.ut3tangent_apply_transpose_from_sweep(
            r, sweep, sum_over_probes=True),
        point_forward=lambda x_sc, ww: ut3_sampling.ut3_apply((x_sc[0], x_sc[1], shape, base_masks), ww),
    )


def uniform_entries_kind(
        x0_data:  typ.Tuple,
) -> bfit.SamplingKind:        # the uniform all-modes `entries` sampling kind
    """The uniform **entries** ``SamplingKind`` -- the twin of :py:data:`t3toolbox.backend.fitting.ENTRIES`."""
    _tk, _tt, shape, base_masks = x0_data
    return dc.replace(
        bfit.ENTRIES,
        precompute=lambda base_data, index: ubv_sampling.ut3tangent_precompute_entries_base_sweep(base_data, index),
        forward=lambda v_sc, index, base_data, sweep: ubv_sampling.ut3tangent_entries_jacobian_from_sweep(
            (v_sc[0], v_sc[1], base_data[4], _var_masks_from_base(base_data)), sweep),
        transpose=lambda r, index, base_data, sweep: ubv_sampling.ut3tangent_entries_transpose_from_sweep(
            r, sweep, sum_over_probes=True),
        point_forward=lambda x_sc, index: ut3_sampling.ut3_entries((x_sc[0], x_sc[1], shape, base_masks), index),
    )


def uniform_probe_kind(
        x0_data:  typ.Tuple,
) -> bfit.SamplingKind:        # the uniform vector-valued `probe` sampling kind
    """The uniform **probe** ``SamplingKind`` -- the twin of :py:data:`t3toolbox.backend.fitting.PROBE`."""
    _tk, _tt, shape, base_masks = x0_data
    return dc.replace(
        bfit.PROBE,
        precompute=lambda base_data, ww: ubv_sampling.ut3tangent_precompute_probe_base_sweep(base_data, ww),
        forward=lambda v_sc, ww, base_data, sweep: ubv_sampling.ut3tangent_probe_jacobian_from_sweep(
            (v_sc[0], v_sc[1], base_data[4], _var_masks_from_base(base_data)), sweep),
        transpose=lambda r, ww, base_data, sweep: ubv_sampling.ut3tangent_probe_transpose_from_sweep(
            r, sweep, sum_over_probes=True),
        point_forward=lambda x_sc, ww: ut3_sampling.ut3_probe(ww, (x_sc[0], x_sc[1], shape, base_masks)),
    )


_SAMPLING_KIND = {'apply': uniform_apply_kind, 'entries': uniform_entries_kind, 'probe': uniform_probe_kind}


def uniform_sampling_kind(
        name:     str,        # 'apply' / 'entries' / 'probe'
        x0_data:  typ.Tuple,  # UniformTuckerTensorTrain.data at the fixed rank
) -> bfit.SamplingKind:
    """Dispatch to the uniform :py:func:`uniform_apply_kind` / ``entries`` / ``probe`` by name."""
    if name not in _SAMPLING_KIND:
        raise ValueError(f"unknown uniform sampling kind {name!r}; expected one of {sorted(_SAMPLING_KIND)}")
    return _SAMPLING_KIND[name](x0_data)
