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
from t3toolbox.backend import ut3_operations
from t3toolbox.backend.common import *

__all__ = [
    'uniform_manifold_ops',
    'uniform_corewise_ops',
    'uniform_geometry_ops',
    'uniform_apply_kind',
    'uniform_entries_kind',
    'uniform_probe_kind',
    'uniform_sampling_kind',
    'uniform_apply_derivatives_kind',
    'uniform_entries_derivatives_kind',
    'uniform_probe_derivatives_kind',
    'uniform_derivatives_kind',
    'uniform_least_squares_problem',
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


# --------------------------------------------------------------------------------------------------
# Derivative (jet) SamplingKind builders -- the uniform twins of backend.fitting.{apply,entries,probe}_
# derivatives_kind. The sample is the paired `(ww, pp)` / `(index, pp)`; the per-order residual weight
# omega enters only sumsq (inherited from the ragged kind) and transpose (re-applied here as aw(r, 2), the
# omega**2 residual weight of the gradient J^T omega^2 r). forward / point_forward stay RAW (the user
# passes raw data + omega). `forward` derives the variation masks from the frame (geometry-agnostic).
# --------------------------------------------------------------------------------------------------
def uniform_apply_derivatives_kind(
        x0_data:  typ.Tuple,                             # UniformTuckerTensorTrain.data at the fixed rank
        order:    int,                                   # highest derivative order
        weight:   typ.Optional[typ.Sequence[float]] = None,  # per-order residual weight omega, (order+1,); None=1
) -> bfit.SamplingKind:                                  # sample = (ww, pp)
    """The uniform **apply-derivatives** ``SamplingKind`` -- the twin of
    :py:func:`t3toolbox.backend.fitting.apply_derivatives_kind`."""
    _tk, _tt, shape, base_masks = x0_data
    aw = bfit._make_order_weight(weight, order)
    return dc.replace(
        bfit.apply_derivatives_kind(order, weight),
        precompute=lambda base_data, s: ubv_sampling.ut3tangent_precompute_apply_base_sweep_jets(
            base_data, s[0], s[1], order),
        forward=lambda v_sc, s, base_data, sweep: ubv_sampling.ut3tangent_apply_jacobian_derivatives_from_sweep(
            (v_sc[0], v_sc[1], base_data[4], _var_masks_from_base(base_data)), sweep, order),
        transpose=lambda r, s, base_data, sweep: ubv_sampling.ut3tangent_apply_transpose_derivatives_from_sweep(
            aw(r, 2), sweep, order, sum_over_probes=True),
        point_forward=lambda x_sc, s: ut3_sampling.ut3_apply_derivatives(
            s[0], s[1], (x_sc[0], x_sc[1], shape, base_masks), order),
    )


def uniform_entries_derivatives_kind(
        x0_data:  typ.Tuple,
        order:    int,
        weight:   typ.Optional[typ.Sequence[float]] = None,
) -> bfit.SamplingKind:                                  # sample = (index, pp)
    """The uniform **entries-derivatives** ``SamplingKind`` -- the twin of
    :py:func:`t3toolbox.backend.fitting.entries_derivatives_kind`."""
    _tk, _tt, shape, base_masks = x0_data
    aw = bfit._make_order_weight(weight, order)
    return dc.replace(
        bfit.entries_derivatives_kind(order, weight),
        precompute=lambda base_data, s: ubv_sampling.ut3tangent_precompute_entries_base_sweep_jets(
            base_data, s[0], s[1], order),
        forward=lambda v_sc, s, base_data, sweep: ubv_sampling.ut3tangent_entries_jacobian_derivatives_from_sweep(
            (v_sc[0], v_sc[1], base_data[4], _var_masks_from_base(base_data)), sweep, order),
        transpose=lambda r, s, base_data, sweep: ubv_sampling.ut3tangent_entries_transpose_derivatives_from_sweep(
            aw(r, 2), sweep, order, sum_over_probes=True),
        point_forward=lambda x_sc, s: ut3_sampling.ut3_entries_derivatives(
            s[0], s[1], (x_sc[0], x_sc[1], shape, base_masks), order),
    )


def uniform_probe_derivatives_kind(
        x0_data:  typ.Tuple,
        order:    int,
        weight:   typ.Optional[typ.Sequence[float]] = None,
) -> bfit.SamplingKind:                                  # sample = (ww, pp)
    """The uniform **probe-derivatives** ``SamplingKind`` -- the twin of
    :py:func:`t3toolbox.backend.fitting.probe_derivatives_kind`.

    The forward / residual are the **packed** probe-derivative jets ``(d,)+(order+1,)+W+C+(N,)`` (order at
    axis 1, after the mode index ``d``), so the per-order weight ``ω`` is built with ``order_axis=1`` and
    ``sumsq`` / ``transpose`` are overridden to weight the correct axis (the inherited order-leading ``aw``
    would broadcast ``ω`` over ``d``)."""
    _tk, _tt, shape, base_masks = x0_data
    aw = bfit._make_order_weight(weight, order, order_axis=1)   # packed probe: order axis is 1 (after d)
    return dc.replace(
        bfit.probe_derivatives_kind(order, weight),
        precompute=lambda base_data, s: ubv_sampling.ut3tangent_precompute_probe_base_sweep_jets(
            base_data, s[0], s[1], order),
        forward=lambda v_sc, s, base_data, sweep: ubv_sampling.ut3tangent_probe_jacobian_derivatives_from_sweep(
            (v_sc[0], v_sc[1], base_data[4], _var_masks_from_base(base_data)), sweep, order),
        transpose=lambda r, s, base_data, sweep: ubv_sampling.ut3tangent_probe_transpose_derivatives_from_sweep(
            aw(r, 2), sweep, order, sum_over_probes=True),
        sumsq=lambda out, n_w: bfit.sumsq_over_probes(aw(out, 1), n_w + 1),
        point_forward=lambda x_sc, s: ut3_sampling.ut3_probe_derivatives(
            s[0], s[1], (x_sc[0], x_sc[1], shape, base_masks), order),
    )


_DERIV_SAMPLING_KIND = {'apply_derivatives':   uniform_apply_derivatives_kind,
                        'entries_derivatives': uniform_entries_derivatives_kind,
                        'probe_derivatives':   uniform_probe_derivatives_kind}


def uniform_derivatives_kind(
        name:     str,        # 'apply_derivatives' / 'entries_derivatives' / 'probe_derivatives'
        x0_data:  typ.Tuple,  # UniformTuckerTensorTrain.data at the fixed rank
        order:    int,
        weight:   typ.Optional[typ.Sequence[float]] = None,
) -> bfit.SamplingKind:
    """Dispatch to the uniform derivative sampling kind by name."""
    if name not in _DERIV_SAMPLING_KIND:
        raise ValueError(f"unknown uniform derivative kind {name!r}; expected one of "
                         f"{sorted(_DERIV_SAMPLING_KIND)}")
    return _DERIV_SAMPLING_KIND[name](x0_data, order, weight)


# --------------------------------------------------------------------------------------------------
# The least-squares Problem factory. Packs the LOOP-INVARIANT sample + data ONCE (the probe vectors /
# perturbations to the supercore mode width N; the probe observed data d-list -> a packed (d,)+...+(N,)
# array), so the reused backend Problem/LocalModel/optimizers run FULLY PACKED -- no per-matvec pack/unpack
# (the whole point of the uniform layer). apply/entries scalar data + the entries integer index need no
# packing. The optimizer's state is the bare supercore pair (x0.data[0], x0.data[1]); the masks of x0's
# fixed rank are captured by the geometry / kind factories (docs/uniform_backend_jit_recipe.md).
# --------------------------------------------------------------------------------------------------
def _pack_sample(name, sample, N):
    """Pack the loop-invariant mode-vectors of ``sample`` once (mirror-tolerant: packed input is kept)."""
    pk = ut3_operations.pack_if_ragged
    if name in ('apply', 'probe'):
        return pk(sample, N)                                  # ww
    if name == 'entries':
        return sample                                         # integer index -- no packing
    if name in ('apply_derivatives', 'probe_derivatives'):
        ww, pp = sample
        return (pk(ww, N), pk(pp, N))
    if name == 'entries_derivatives':
        index, pp = sample
        return (index, pk(pp, N))
    raise ValueError(f"unknown uniform sampling kind {name!r}")


def _pack_data(name, data, N):
    """Pack the observed data once: probe kinds -> a packed ``(d,)+...+(N,)`` array; apply/entries -> the
    scalar data unchanged (mirror-tolerant)."""
    if name in ('probe', 'probe_derivatives'):
        return ut3_operations.pack_if_ragged(data, N)
    return data


def uniform_least_squares_problem(
        geometry:  str,        # 'manifold' / 'corewise'
        kind_name: str,        # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        x0:        typ.Any,    # UniformTuckerTensorTrain -- the fixed rank to optimize over (masks captured)
        sample:    typ.Any,    # ww / index / (ww, pp) / (index, pp) -- ragged or packed (packed once here)
        data:      typ.Any,    # observed S(x_true): scalar array (apply/entries) or a d-list/packed (probe)
        order:     typ.Optional[int]                 = None,  # derivative kinds only (required)
        weight:    typ.Optional[typ.Sequence[float]] = None,  # derivative kinds only: per-order weight ω
) -> bopt.Problem:
    """Assemble a fully-packed uniform least-squares :py:class:`~t3toolbox.backend.optimizers.Problem`.

    Builds the uniform geometry (:py:func:`uniform_geometry_ops`) + sampling kind
    (:py:func:`uniform_sampling_kind` / :py:func:`uniform_derivatives_kind`) at ``x0``'s fixed rank, packs
    the loop-invariant ``sample`` + ``data`` once, and returns the reused backend ``Problem``. The optimizer
    then runs on the bare supercore pair ``(x0.data[0], x0.data[1])`` -- e.g.
    ``backend.optimizers.newton_cg(problem, (x0.data[0], x0.data[1]))``."""
    x0_data = x0.data
    N = x0_data[0].shape[-1]
    geom = uniform_geometry_ops(geometry, x0_data)
    kind = (uniform_sampling_kind(kind_name, x0_data) if kind_name in ('apply', 'entries', 'probe')
            else uniform_derivatives_kind(kind_name, x0_data, order, weight))
    return bopt.least_squares_problem(geom, kind, _pack_sample(kind_name, sample, N), _pack_data(kind_name, data, N))
