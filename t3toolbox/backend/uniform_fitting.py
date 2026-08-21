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
  * a **frame** is the full frame ``.data`` (``frame`` returns it, ``project`` / ``retract`` consume it).

Each closure re-attaches the held ``shape`` + masks at the boundary (building the full ``.data`` tuples
the ``utv_operations`` primitives expect) and strips them back to a bare pair on output, so
``corewise.*`` and ``GeometryOps.inner`` see only supercores. Under jit the re-derived frame masks
constant-fold to device constants (the "1 compile" behaviour of the recipe).
"""
import dataclasses as dc
import typing as typ

import numpy as np

from t3toolbox.backend import optimizers as bopt
from t3toolbox.backend import fitting as bfit
from t3toolbox.backend import geometry as geometry_module
from t3toolbox.backend import ufv_conversions
from t3toolbox.backend import ufv_masking
from t3toolbox.backend import utv_operations as utv_ops
from t3toolbox.backend import utv_sampling
from t3toolbox.backend import ut3_sampling
from t3toolbox.backend import ut3_operations
from t3toolbox.backend import ranks
from t3toolbox.backend import sharing as sharing_module
from t3toolbox.backend.common import *

__all__ = [
    'uniform_apply_kind',
    'uniform_entries_kind',
    'uniform_probe_kind',
    'uniform_sampling_kind',
    'uniform_apply_derivatives_kind',
    'uniform_entries_derivatives_kind',
    'uniform_probe_derivatives_kind',
    'uniform_derivatives_kind',
    'uniform_minimal',
    'uniform_least_squares_problem',
    'pack_sample',
    'pack_data',
]


# --------------------------------------------------------------------------------------------------
# SamplingKind builders -- the uniform twins of backend.fitting.{APPLY,ENTRIES,PROBE}. Only the four
# layer-specific fields (precompute / forward / transpose / point_forward) differ; the layer-agnostic
# reductions and default-draw layout (sumsq / w_axes / n_measurements / take) are reused verbatim from
# the ragged kind via `dataclasses.replace`. `forward` derives the variation masks from the frame it is
# handed (the `(up, down, left[:-1], right[1:])` gauge shift -- valid for BOTH the orthonormal manifold
# frame and the (U,G,G,G) corewise frame), so the kind is geometry-agnostic; only `point_forward` (the
# S(x) op on the plain-UT3 point) closes over the plain-UT3 shape + edge masks.
# --------------------------------------------------------------------------------------------------
def _var_masks_from_frame(frame_data):
    """The variation masks of a frame -- the gauge shift, applied to the frame's own rank masks."""
    return ufv_masking.ufv_variation_masks(frame_data[5])


def uniform_apply_kind(
        x0_data:  typ.Tuple,   # UniformTuckerTensorTrain.data = (tk_sc, tt_sc, shape, (tucker_mask, tt_mask))
) -> bfit.SamplingKind:        # the uniform all-modes `apply` sampling kind at x0's fixed rank
    """The uniform **apply** ``SamplingKind`` -- the twin of :py:data:`t3toolbox.backend.fitting.APPLY`."""
    _tk, _tt, shape, base_masks = x0_data
    return dc.replace(
        bfit.APPLY,
        precompute=lambda frame_data, ww: utv_sampling.utv_precompute_apply_frame_sweep(frame_data, ww),
        forward=lambda v_sc, ww, frame_data, sweep: utv_sampling.utv_apply_jacobian_from_sweep(
            (v_sc[0], v_sc[1], frame_data[4], _var_masks_from_frame(frame_data)), sweep),
        transpose=lambda r, ww, frame_data, sweep: utv_sampling.utv_apply_transpose_from_sweep(
            r, sweep, sum_over_probes=True),
        point_forward=lambda x_sc, ww: ut3_sampling.ut3_apply((x_sc[0], x_sc[1], shape, base_masks), ww),
        take=_ptake_apply,
    )


def uniform_entries_kind(
        x0_data:  typ.Tuple,
) -> bfit.SamplingKind:        # the uniform all-modes `entries` sampling kind
    """The uniform **entries** ``SamplingKind`` -- the twin of :py:data:`t3toolbox.backend.fitting.ENTRIES`."""
    _tk, _tt, shape, base_masks = x0_data
    return dc.replace(
        bfit.ENTRIES,
        precompute=lambda frame_data, index: utv_sampling.utv_precompute_entries_frame_sweep(frame_data, index),
        forward=lambda v_sc, index, frame_data, sweep: utv_sampling.utv_entries_jacobian_from_sweep(
            (v_sc[0], v_sc[1], frame_data[4], _var_masks_from_frame(frame_data)), sweep),
        transpose=lambda r, index, frame_data, sweep: utv_sampling.utv_entries_transpose_from_sweep(
            r, sweep, sum_over_probes=True),
        point_forward=lambda x_sc, index: ut3_sampling.ut3_entries((x_sc[0], x_sc[1], shape, base_masks), index),
        take=_ptake_entries,
    )


def uniform_probe_kind(
        x0_data:  typ.Tuple,
        weight:   typ.Optional[typ.Any] = None,  # per-mode residual weight ω, (d,) / (d,1); None = 1
) -> bfit.SamplingKind:        # the uniform vector-valued `probe` sampling kind (optionally per-mode weighted)
    """The uniform **probe** ``SamplingKind`` -- the twin of :py:func:`t3toolbox.backend.fitting.probe_kind`.

    The forward / residual are the **packed** probe output ``(d,)+W+C+(N,)`` (mode index ``d`` at axis 0, no
    order axis), so the per-mode weight ``ω`` is built with ``mode_axis=0`` (no order axis) and ``sumsq`` /
    ``transpose`` are overridden to weight it (``ω`` enters ``sumsq`` ×ω, ``transpose`` ×ω²)."""
    _tk, _tt, shape, base_masks = x0_data
    aw = bfit._make_weight(bfit._weight_matrix(weight, 0, 'mode'), order_axis=None, mode_axis=0)
    return dc.replace(
        bfit.PROBE,
        precompute=lambda frame_data, ww: utv_sampling.utv_precompute_probe_frame_sweep(frame_data, ww),
        forward=lambda v_sc, ww, frame_data, sweep: utv_sampling.utv_probe_jacobian_from_sweep(
            (v_sc[0], v_sc[1], frame_data[4], _var_masks_from_frame(frame_data)), sweep),
        transpose=lambda r, ww, frame_data, sweep: utv_sampling.utv_probe_transpose_from_sweep(
            aw(r, 2), sweep, sum_over_probes=True),
        sumsq=lambda out, n_w: bfit.sumsq_over_probes(aw(out, 1), n_w),
        point_forward=lambda x_sc, ww: ut3_sampling.ut3_probe(ww, (x_sc[0], x_sc[1], shape, base_masks)),
        take=_ptake_probe,
    )


_SAMPLING_KIND = {'apply': uniform_apply_kind, 'entries': uniform_entries_kind, 'probe': uniform_probe_kind}


def uniform_sampling_kind(
        name:     str,        # 'apply' / 'entries' / 'probe'
        x0_data:  typ.Tuple,  # UniformTuckerTensorTrain.data at the fixed rank
        weight:   typ.Optional[typ.Any] = None,  # per-mode weight ω (probe only); apply/entries take none
) -> bfit.SamplingKind:
    """Dispatch to the uniform :py:func:`uniform_apply_kind` / ``entries`` / ``probe`` by name. Only the
    vector-valued **probe** kind is weightable (per-mode ``ω``); plain apply/entries have no mode axis and
    take no weight (a non-``None`` ``weight`` for them is a structural error)."""
    if name == 'probe':
        return uniform_probe_kind(x0_data, weight)
    if name in ('apply', 'entries'):
        if weight is not None:
            raise ValueError(f"the plain '{name}' kind takes no residual weight (no mode or order axis); "
                             "per-mode weighting is defined for probe, per-order for the derivative kinds.")
        return _SAMPLING_KIND[name](x0_data)
    raise ValueError(f"unknown uniform sampling kind {name!r}; expected one of {sorted(_SAMPLING_KIND)}")


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
    aw = bfit._make_weight(bfit._weight_matrix(weight, order, 'order'), order_axis=0, mode_axis=None)
    return dc.replace(
        bfit.apply_derivatives_kind(order, weight),
        precompute=lambda frame_data, s: utv_sampling.utv_precompute_apply_frame_sweep_jets(
            frame_data, s[0], s[1], order),
        forward=lambda v_sc, s, frame_data, sweep: utv_sampling.utv_apply_jacobian_derivatives_from_sweep(
            (v_sc[0], v_sc[1], frame_data[4], _var_masks_from_frame(frame_data)), sweep, order),
        transpose=lambda r, s, frame_data, sweep: utv_sampling.utv_apply_transpose_derivatives_from_sweep(
            aw(r, 2), sweep, order, sum_over_probes=True),
        point_forward=lambda x_sc, s: ut3_sampling.ut3_apply_derivatives(
            s[0], s[1], (x_sc[0], x_sc[1], shape, base_masks), order),
        take=_ptake_deriv_apply,
    )


def uniform_entries_derivatives_kind(
        x0_data:  typ.Tuple,
        order:    int,
        weight:   typ.Optional[typ.Sequence[float]] = None,
) -> bfit.SamplingKind:                                  # sample = (index, pp)
    """The uniform **entries-derivatives** ``SamplingKind`` -- the twin of
    :py:func:`t3toolbox.backend.fitting.entries_derivatives_kind`."""
    _tk, _tt, shape, base_masks = x0_data
    aw = bfit._make_weight(bfit._weight_matrix(weight, order, 'order'), order_axis=0, mode_axis=None)
    return dc.replace(
        bfit.entries_derivatives_kind(order, weight),
        precompute=lambda frame_data, s: utv_sampling.utv_precompute_entries_frame_sweep_jets(
            frame_data, s[0], s[1], order),
        forward=lambda v_sc, s, frame_data, sweep: utv_sampling.utv_entries_jacobian_derivatives_from_sweep(
            (v_sc[0], v_sc[1], frame_data[4], _var_masks_from_frame(frame_data)), sweep, order),
        transpose=lambda r, s, frame_data, sweep: utv_sampling.utv_entries_transpose_derivatives_from_sweep(
            aw(r, 2), sweep, order, sum_over_probes=True),
        point_forward=lambda x_sc, s: ut3_sampling.ut3_entries_derivatives(
            s[0], s[1], (x_sc[0], x_sc[1], shape, base_masks), order),
        take=_ptake_deriv_entries,
    )


def uniform_probe_derivatives_kind(
        x0_data:    typ.Tuple,
        order:      int,
        weight:     typ.Optional[typ.Sequence[float]] = None,
        chunk_size: typ.Optional[int] = 100,                 # W-chunk size for the 𝒥ᵀ assembly (docs/chunking.md)
) -> bfit.SamplingKind:                                  # sample = (ww, pp)
    """The uniform **probe-derivatives** ``SamplingKind`` -- the twin of
    :py:func:`t3toolbox.backend.fitting.probe_derivatives_kind`.

    The forward / residual are the **packed** probe-derivative jets ``(d,)+(order+1,)+W+C+(N,)`` (order at
    axis 1, after the mode index ``d``), so the per-order weight ``ω`` is built with ``order_axis=1`` and
    ``sumsq`` / ``transpose`` are overridden to weight the correct axis (the inherited order-leading ``aw``
    would broadcast ``ω`` over ``d``)."""
    _tk, _tt, shape, base_masks = x0_data
    aw = bfit._make_weight(bfit._weight_matrix(weight, order, 'order'), order_axis=1, mode_axis=0)  # packed probe: mode axis 0, order axis 1
    return dc.replace(
        bfit.probe_derivatives_kind(order, weight),
        precompute=lambda frame_data, s: utv_sampling.utv_precompute_probe_frame_sweep_jets(
            frame_data, s[0], s[1], order),
        forward=lambda v_sc, s, frame_data, sweep: utv_sampling.utv_probe_jacobian_derivatives_from_sweep(
            (v_sc[0], v_sc[1], frame_data[4], _var_masks_from_frame(frame_data)), sweep, order),
        transpose=lambda r, s, frame_data, sweep: utv_sampling.utv_probe_transpose_derivatives_from_sweep(
            aw(r, 2), sweep, order, sum_over_probes=True, chunk_size=chunk_size),
        sumsq=lambda out, n_w: bfit.sumsq_over_probes(aw(out, 1), n_w + 1),
        point_forward=lambda x_sc, s: ut3_sampling.ut3_probe_derivatives(
            s[0], s[1], (x_sc[0], x_sc[1], shape, base_masks), order),
        take=_ptake_deriv_probe,
    )


_DERIV_SAMPLING_KIND = {'apply_derivatives':   uniform_apply_derivatives_kind,
                        'entries_derivatives': uniform_entries_derivatives_kind,
                        'probe_derivatives':   uniform_probe_derivatives_kind}


def uniform_derivatives_kind(
        name:       str,        # 'apply_derivatives' / 'entries_derivatives' / 'probe_derivatives'
        x0_data:    typ.Tuple,  # UniformTuckerTensorTrain.data at the fixed rank
        order:      int,
        weight:     typ.Optional[typ.Sequence[float]] = None,
        chunk_size: typ.Optional[int] = 100,   # probe_derivatives only (its 𝒥ᵀ assembly chunks); ignored otherwise
) -> bfit.SamplingKind:
    """Dispatch to the uniform derivative sampling kind by name."""
    if name not in _DERIV_SAMPLING_KIND:
        raise ValueError(f"unknown uniform derivative kind {name!r}; expected one of "
                         f"{sorted(_DERIV_SAMPLING_KIND)}")
    if name == 'probe_derivatives':            # the only derivative kind with a chunkable assembly
        return _DERIV_SAMPLING_KIND[name](x0_data, order, weight, chunk_size=chunk_size)
    return _DERIV_SAMPLING_KIND[name](x0_data, order, weight)


# --------------------------------------------------------------------------------------------------
# The least-squares Problem factory. Packs the LOOP-INVARIANT sample + data ONCE (the probe vectors /
# perturbations to the supercore mode width N; the probe observed data d-list -> a packed (d,)+...+(N,)
# array), so the reused backend Problem/LocalModel/optimizers run FULLY PACKED -- no per-matvec pack/unpack
# (the whole point of the uniform layer). apply/entries scalar data + the entries integer index need no
# packing. The optimizer's state is the bare supercore pair (x0.data[0], x0.data[1]); the masks of x0's
# fixed rank are captured by the geometry / kind factories (docs/uniform_backend_jit_recipe.md).
# --------------------------------------------------------------------------------------------------
def pack_sample(name, sample, N):
    """Pack the loop-invariant mode-vectors of ``sample`` once (mirror-tolerant: packed input is kept).

    A boundary helper (the uniform kinds run on packed vectors of width ``N``): dispatches
    :py:func:`~t3toolbox.backend.ut3_operations.pack_if_ragged` per sampling-kind (``ww`` for apply/probe,
    both ``ww``/``pp`` for the derivative kinds; the integer ``index`` is never packed). Used by
    :py:func:`uniform_least_squares_problem` and the frontend :py:mod:`t3toolbox.fitting` uniform models."""
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


def pack_data(name, data, N):
    """Pack the observed data (or a residual of the same shape) once: probe kinds -> a packed
    ``(d,)+...+(N,)`` array; apply/entries -> the scalar data unchanged (mirror-tolerant)."""
    if name in ('probe', 'probe_derivatives'):
        return ut3_operations.pack_if_ragged(data, N)
    return data


# Packed-aware minibatch `take` (the SamplingKind default-draw hook), so mc_sgd/adam keep minibatches
# PACKED (a single `(d,)+W'+…` gather via bfit._flat_gather) instead of the ragged `take` iterating the
# packed sample back into a d-list. W sits at axis 1 of the d-leading packed sample/ww/pp/probe-data (axis
# 2 for a probe-derivative jet's data, after d + order); apply/entries scalar data has W at axis 0 (axis 1
# after a leading order axis). `w_axes` / `n_measurements` need no override -- they index ww[0], which drops
# the d axis to the same W-leading shape as the ragged sample.
_fg = bfit._flat_gather

def _ptake_apply(ww, data, idx):
    n_w = ww.ndim - 2                                          # ww packed (d,)+W+(N,)
    return _fg(ww, 1, n_w, idx), _fg(data, 0, n_w, idx)        # data (W,)+C

def _ptake_probe(ww, data, idx):
    n_w = ww.ndim - 2
    return _fg(ww, 1, n_w, idx), _fg(data, 1, n_w, idx)        # data packed (d,)+W+C+(N,)

def _ptake_entries(index, data, idx):
    n_w = index.ndim - 1                                       # index (d,)+W
    return _fg(index, 1, n_w, idx), _fg(data, 0, n_w, idx)

def _ptake_deriv_apply(sample, data, idx):
    ww, pp = sample; n_w = ww.ndim - 2
    return (_fg(ww, 1, n_w, idx), _fg(pp, 1, n_w, idx)), _fg(data, 1, n_w, idx)   # data (order+1,)+W+C

def _ptake_deriv_entries(sample, data, idx):
    index, pp = sample; n_w = index.ndim - 1
    return (_fg(index, 1, n_w, idx), _fg(pp, 1, n_w, idx)), _fg(data, 1, n_w, idx)

def _ptake_deriv_probe(sample, data, idx):
    ww, pp = sample; n_w = ww.ndim - 2
    return (_fg(ww, 1, n_w, idx), _fg(pp, 1, n_w, idx)), _fg(data, 2, n_w, idx)   # data (d,)+(order+1,)+W+C+(N,)


def uniform_minimal(
        x0:  typ.Any,   # UniformTuckerTensorTrain
        sharing: typ.Optional[typ.Sequence] = None,  # len=d, static; one hashable group label per mode (None = unshared)
) -> typ.Any:           # the same tensor with structurally-minimal ranks (x0 itself if already minimal)
    """Reduce ``x0`` to its **structurally-minimal ranks** -- the SAME tensor, with any unrealizable nominal
    rank dropped (e.g. a TT bond rank exceeding what the Tucker ranks can realize). A no-op (returns ``x0``
    unchanged) when it is already minimal, which is the common case.

    **Uniform fitting requires a minimal frame** (:py:func:`uniform_least_squares_problem`). The reason is
    structural: from a *non*-minimal frame the manifold retraction truncates to the realizable (minimal)
    rank, which no longer matches the fixed masks the optimizer holds loop-invariant -- so the next step's
    masking desyncs and crashes. The ragged layer tolerates non-minimal ranks (per-core shapes adapt); the
    uniform layer cannot (its masks are fixed), so it must start minimal and stay minimal (from a minimal
    frame the retraction provably preserves the ranks). Reduction: ``t3svd`` (-> left-orthogonal) then a
    ``'right_to_left'`` :py:meth:`~t3toolbox.uniform_tucker_tensor_train.UniformTuckerTensorTrain.rank_adjustment_sweep`
    (-> minimal, right-orthogonal). Same-tensor, done once at setup (eager).

    With ``sharing``, minimality is the SHARED notion and the reduction is the grouped one -- REQUIRED
    for a shared start: the per-mode reduction can clip a group rank the group ceiling admits (untying
    the group), and even at shared-minimal ranks its per-mode SVDs rotate each factor independently
    (untying the values). The grouped path keeps the factors tied and the group rank shared."""
    tucker_ranks = np.asarray(x0.tucker_ranks)
    tt_ranks = np.asarray(x0.tt_ranks)
    min_tucker, min_tt = ranks.compute_minimal_ranks(x0.shape, tucker_ranks, tt_ranks, sharing=sharing)
    if bool(np.all(tucker_ranks == np.asarray(min_tucker)) and np.all(tt_ranks == np.asarray(min_tt))):
        return x0
    left_orthogonal, _ss_tk, _ss_tt = x0.t3svd(sharing=sharing)
    return left_orthogonal.rank_adjustment_sweep('right_to_left', sharing=sharing)


def uniform_least_squares_problem(
        geometry:  str,        # 'manifold' / 'corewise'
        kind_name: str,        # 'apply' / 'entries' / 'probe' (+ '_derivatives')
        x0:        typ.Any,    # UniformTuckerTensorTrain -- MINIMAL-rank frame (see uniform_minimal); masks captured
        sample:    typ.Any,    # ww / index / (ww, pp) / (index, pp) -- ragged or packed (packed once here)
        data:      typ.Any,    # observed S(x_true): scalar array (apply/entries) or a d-list/packed (probe)
        order:     typ.Optional[int] = None,  # derivative kinds only (required)
        weight:    typ.Optional[typ.Any] = None,  # residual weight ω: per-mode (probe) or ω[mode,order] (derivatives)
        regularizer: typ.Any = None,          # optional backend.regularization.Regularizer (e.g. IdentityRegularizer(λ))
        chunk_size: typ.Optional[int] = 100,  # probe_derivatives only: W-chunk size for 𝒥ᵀ (docs/chunking.md)
        sharing:    typ.Optional[typ.Sequence] = None,  # len=d, static; group labels (None = unshared)
) -> bopt.Problem:
    """Assemble a fully-packed uniform least-squares :py:class:`~t3toolbox.backend.optimizers.Problem`.

    Builds the uniform geometry (:py:class:`~t3toolbox.backend.geometry.UniformManifoldGeometryOps` /
    :py:class:`~t3toolbox.backend.geometry.UniformCorewiseGeometryOps`) + sampling kind
    (:py:func:`uniform_sampling_kind` / :py:func:`uniform_derivatives_kind`) at ``x0``'s fixed rank, packs
    the loop-invariant ``sample`` + ``data`` once, and returns the reused backend ``Problem``. The optimizer
    then runs on the bare supercore pair ``(x0.data[0], x0.data[1])`` -- e.g.
    ``backend.optimizers.newton_cg(problem, (x0.data[0], x0.data[1]))``.

    **``x0`` must have minimal ranks** -- call :py:func:`uniform_minimal` first if it might not. A
    non-minimal nominal rank is unrealizable and would desync the retraction from the held masks
    mid-optimization; this is checked (structurally, cheap) and rejected up front rather than crashing
    later.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.backend.optimizers as bopt
    >>> import t3toolbox.backend.uniform_fitting as uf
    >>> from t3toolbox.backend import apply as bapply
    >>> np.random.seed(0)
    >>> ww = [np.random.randn(20, n) for n in (6, 6, 6)]
    >>> data = np.random.randn(20)

    A **non-minimal** frame -- here TT bond rank 3 is unrealizable for a 2x2x2 central Tucker core (its TT
    bonds are at most 2) -- is rejected up front with a clear error:

    >>> x0 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((6, 6, 6), (2, 2, 2), (1, 3, 3, 1)))
    >>> uf.uniform_least_squares_problem('manifold', 'apply', x0, ww, data)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: uniform_least_squares_problem requires a minimal-rank frame x0 ...

    :py:func:`uniform_minimal` reduces it to minimal ranks (the SAME tensor), and then it works:

    >>> x0m = uf.uniform_minimal(x0)
    >>> print(bool(np.allclose(x0m.to_dense(), x0.to_dense())))   # same tensor, minimal ranks
    True
    >>> prob = uf.uniform_least_squares_problem('manifold', 'apply', x0m, ww, data)
    >>> x_opt, stats = bopt.gradient_descent(prob, (x0m.data[0], x0m.data[1]), n_iter=5)
    >>> print(bool(stats['losses'][-1] < stats['losses'][0]))     # it descends
    True
    """
    min_tucker, min_tt = ranks.compute_minimal_ranks(x0.shape, np.asarray(x0.tucker_ranks),
                                                     np.asarray(x0.tt_ranks), sharing=sharing)
    if not bool(np.all(np.asarray(x0.tucker_ranks) == np.asarray(min_tucker))
                and np.all(np.asarray(x0.tt_ranks) == np.asarray(min_tt))):
        raise ValueError(
            "uniform_least_squares_problem requires a minimal-rank frame x0 (a non-minimal nominal rank is "
            "unrealizable and would desync the retraction from the fixed masks mid-optimization; with "
            "sharing, minimality is the SHARED notion -- the group ceiling). Reduce it first: "
            "x0 = uniform_minimal(x0" + (", sharing=...)" if sharing is not None else ")") + ".")
    x0_data = x0.data
    N = x0_data[0].shape[-1]
    geom = (geometry_module.UniformManifoldGeometryOps.from_point(x0_data, sharing) if geometry == 'manifold'
            else geometry_module.UniformCorewiseGeometryOps.from_point(x0_data, sharing))
    kind = (uniform_sampling_kind(kind_name, x0_data, weight) if kind_name in ('apply', 'entries', 'probe')
            else uniform_derivatives_kind(kind_name, x0_data, order, weight, chunk_size=chunk_size))
    return bopt.least_squares_problem(geom, kind, pack_sample(kind_name, sample, N), pack_data(kind_name, data, N),
                                      regularizer=regularizer)
