# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Uniform-layer fitting: the packed ``SamplingKind`` classes and the least-squares ``Problem``.

The uniform twins of :py:mod:`t3toolbox.backend.fitting`'s kinds -- subclasses that override the five
layer-specific operations and, for the probe kinds, where ``omega``'s axes sit in the PACKED output.
Each carries the fixed rank it was built at (``shape`` + the plain-UT3 ``masks``) as FIELDS, so a
rebuilt kind of the same rank is the same jax cache key
(:py:mod:`t3toolbox.backend.geometry` follows the same rule; the reasoning is
``docs/contributor/parameters_not_closures.md``).

:py:func:`uniform_least_squares_problem` packs the loop-invariant sample + data ONCE and returns the
shared backend ``Problem``, so the optimizers run fully packed -- no per-matvec pack/unpack. The
**geometry** half lives in :py:mod:`t3toolbox.backend.geometry`.

**Protocol note (review H2-9, ruled keep + document):** the kinds' ``forward`` / ``transpose`` keep
the ragged Kind protocol's ``(v, sample, frame_data, sweep)`` signature even though the packed
``sweep`` already carries the sample and the frame -- the extra arguments are deliberately unread.
Protocol uniformity is the point: generic code (the shared ``LocalModel``) calls every kind the same
way, ragged or uniform.
"""
import dataclasses as dc
import typing as typ

import numpy as np

from t3toolbox.backend import optimizers as bopt
from t3toolbox.backend.common import readonly_mask_copies
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
    'UniformApplyKind',
    'UniformEntriesKind',
    'UniformProbeKind',
    'UniformApplyDerivativesKind',
    'UniformEntriesDerivativesKind',
    'UniformProbeDerivativesKind',
    'uniform_sampling_kind',
    'uniform_derivatives_kind',
    'uniform_minimal',
    'uniform_least_squares_problem',
    'pack_sample',
    'pack_data',
]


# --------------------------------------------------------------------------------------------------
# The uniform sampling kinds -- subclasses of the ragged ones (backend.fitting) that override exactly
# the five layer-specific operations (precompute / forward / transpose / point_forward / take) and, for
# the probe kinds, where omega's axes sit in the PACKED output. The layer-agnostic half -- the
# reductions, the sample layout, the residual-weight machinery, and the value identity -- is inherited.
#
# Each carries the fixed rank it is built at (`shape` + the plain-UT3 `masks`) as FIELDS, so a rebuilt
# kind of the same rank is the same jax cache key (the same rule as backend.geometry). `forward` derives
# the variation masks from the frame it is handed, so the kinds stay geometry-agnostic; only
# `point_forward` (the S(x) op on the plain-UT3 point) needs the held shape + masks.
# --------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True, eq=False)
class _UniformKind:
    """The fixed rank a uniform kind is built at. Defaults so the field ordering works alongside the
    inherited defaulted parameters (Python 3.9 has no ``kw_only``); use :py:meth:`from_point`."""

    shape:  typ.Tuple[int, ...] = ()   # the mode sizes
    masks:  typ.Tuple = ()             # plain-UT3 (tucker_edge_mask, tt_edge_mask); HOST numpy

    @classmethod
    def from_point(cls, x0_data, **parameters):
        """The kind at ``x0``'s fixed rank; ``parameters`` are the kind's own (order / weight / chunk)."""
        _tk_sc, _tt_sc, shape, base_masks = x0_data
        return cls(shape=tuple(shape), masks=readonly_mask_copies(base_masks), **parameters)

    @property
    def _point(self):
        """A plain-UT3 ``.data`` shim for the point operation, given the bare supercore pair."""
        return lambda x_sc: (x_sc[0], x_sc[1], self.shape, self.masks)

    @staticmethod
    def _variations(v_sc, frame_data):
        """A bare variation supercore pair as variation ``.data``, masked by the frame's gauge shift."""
        return (v_sc[0], v_sc[1], frame_data[4], ufv_masking.ufv_variation_masks(frame_data[5]))


@dc.dataclass(frozen=True, eq=False)
class UniformApplyKind(_UniformKind, bfit.ApplyKind):
    """The uniform twin of :py:data:`~t3toolbox.backend.fitting.APPLY`."""

    def precompute(self, frame_data, ww):
        return utv_sampling.utv_precompute_apply_frame_sweep(frame_data, ww)

    def forward(self, v_sc, ww, frame_data, sweep):
        return utv_sampling.utv_apply_jacobian_from_sweep(self._variations(v_sc, frame_data), sweep)

    def transpose(self, r, ww, frame_data, sweep):
        return utv_sampling.utv_apply_transpose_from_sweep(r, sweep, sum_over_probes=True)

    def point_forward(self, x_sc, ww):
        return ut3_sampling.ut3_apply(self._point(x_sc), ww)

    def take(self, sample, data, idx):
        return _ptake_apply(sample, data, idx)


@dc.dataclass(frozen=True, eq=False)
class UniformEntriesKind(_UniformKind, bfit.EntriesKind):
    """The uniform twin of :py:data:`~t3toolbox.backend.fitting.ENTRIES`."""

    def precompute(self, frame_data, index):
        return utv_sampling.utv_precompute_entries_frame_sweep(frame_data, index)

    def forward(self, v_sc, index, frame_data, sweep):
        return utv_sampling.utv_entries_jacobian_from_sweep(self._variations(v_sc, frame_data), sweep)

    def transpose(self, r, index, frame_data, sweep):
        return utv_sampling.utv_entries_transpose_from_sweep(r, sweep, sum_over_probes=True)

    def point_forward(self, x_sc, index):
        return ut3_sampling.ut3_entries(self._point(x_sc), index)

    def take(self, sample, data, idx):
        return _ptake_entries(sample, data, idx)


@dc.dataclass(frozen=True, eq=False)
class UniformProbeKind(_UniformKind, bfit.ProbeKind):
    """The uniform twin of :py:class:`~t3toolbox.backend.fitting.ProbeKind`.

    The forward / residual are the **packed** probe output ``(d,)+W+C+(N,)`` -- mode index ``d`` at axis
    0, no order axis -- so ``omega``'s mode axis is 0 and it has no order axis, unlike the ragged kind
    whose per-mode weight indexes a list."""

    _order_axis = None
    _mode_axis = 0

    def precompute(self, frame_data, ww):
        return utv_sampling.utv_precompute_probe_frame_sweep(frame_data, ww)

    def forward(self, v_sc, ww, frame_data, sweep):
        return utv_sampling.utv_probe_jacobian_from_sweep(self._variations(v_sc, frame_data), sweep)

    def transpose(self, r, ww, frame_data, sweep):
        return utv_sampling.utv_probe_transpose_from_sweep(self._apply_weight(r, 2), sweep,
                                                           sum_over_probes=True)

    def point_forward(self, x_sc, ww):
        return ut3_sampling.ut3_probe(ww, self._point(x_sc))

    def take(self, sample, data, idx):
        return _ptake_probe(sample, data, idx)


@dc.dataclass(frozen=True, eq=False)
class UniformApplyDerivativesKind(_UniformKind, bfit.ApplyDerivativesKind):
    """The uniform twin of :py:class:`~t3toolbox.backend.fitting.ApplyDerivativesKind`."""

    def precompute(self, frame_data, s):
        return utv_sampling.utv_precompute_apply_frame_sweep_jets(frame_data, s[0], s[1], self.order)

    def forward(self, v_sc, s, frame_data, sweep):
        return utv_sampling.utv_apply_jacobian_derivatives_from_sweep(
            self._variations(v_sc, frame_data), sweep, self.order)

    def transpose(self, r, s, frame_data, sweep):
        return utv_sampling.utv_apply_transpose_derivatives_from_sweep(
            self._apply_weight(r, 2), sweep, self.order, sum_over_probes=True)

    def point_forward(self, x_sc, s):
        return ut3_sampling.ut3_apply_derivatives(s[0], s[1], self._point(x_sc), self.order)

    def take(self, sample, data, idx):
        return _ptake_deriv_apply(sample, data, idx)


@dc.dataclass(frozen=True, eq=False)
class UniformEntriesDerivativesKind(_UniformKind, bfit.EntriesDerivativesKind):
    """The uniform twin of :py:class:`~t3toolbox.backend.fitting.EntriesDerivativesKind`."""

    def precompute(self, frame_data, s):
        return utv_sampling.utv_precompute_entries_frame_sweep_jets(frame_data, s[0], s[1], self.order)

    def forward(self, v_sc, s, frame_data, sweep):
        return utv_sampling.utv_entries_jacobian_derivatives_from_sweep(
            self._variations(v_sc, frame_data), sweep, self.order)

    def transpose(self, r, s, frame_data, sweep):
        return utv_sampling.utv_entries_transpose_derivatives_from_sweep(
            self._apply_weight(r, 2), sweep, self.order, sum_over_probes=True)

    def point_forward(self, x_sc, s):
        return ut3_sampling.ut3_entries_derivatives(s[0], s[1], self._point(x_sc), self.order)

    def take(self, sample, data, idx):
        return _ptake_deriv_entries(sample, data, idx)


@dc.dataclass(frozen=True, eq=False)
class UniformProbeDerivativesKind(_UniformKind, bfit.ProbeDerivativesKind):
    """The uniform twin of :py:class:`~t3toolbox.backend.fitting.ProbeDerivativesKind`.

    The forward / residual are the packed jets ``(d,)+(order+1,)+W+C+(N,)`` -- order at axis 1, after the
    mode index ``d`` -- so ``omega`` sits at ``mode_axis=0, order_axis=1``. The ragged kind's
    order-leading placement would broadcast ``omega`` over ``d`` here."""

    _order_axis = 1
    _mode_axis = 0

    def precompute(self, frame_data, s):
        return utv_sampling.utv_precompute_probe_frame_sweep_jets(frame_data, s[0], s[1], self.order)

    def forward(self, v_sc, s, frame_data, sweep):
        return utv_sampling.utv_probe_jacobian_derivatives_from_sweep(
            self._variations(v_sc, frame_data), sweep, self.order)

    def transpose(self, r, s, frame_data, sweep):
        return utv_sampling.utv_probe_transpose_derivatives_from_sweep(
            self._apply_weight(r, 2), sweep, self.order, sum_over_probes=True,
            chunk_size=self.chunk_size)

    def point_forward(self, x_sc, s):
        return ut3_sampling.ut3_probe_derivatives(s[0], s[1], self._point(x_sc), self.order)

    def take(self, sample, data, idx):
        return _ptake_deriv_probe(sample, data, idx)


_SAMPLING_KIND = {'apply': UniformApplyKind, 'entries': UniformEntriesKind, 'probe': UniformProbeKind}
_DERIV_SAMPLING_KIND = {'apply_derivatives':   UniformApplyDerivativesKind,
                        'entries_derivatives': UniformEntriesDerivativesKind,
                        'probe_derivatives':   UniformProbeDerivativesKind}


def uniform_sampling_kind(
        name:     str,        # 'apply' / 'entries' / 'probe'
        x0_data:  typ.Tuple,  # UniformTuckerTensorTrain.data at the fixed rank
        weight:   typ.Optional[typ.Any] = None,  # per-mode weight omega (probe only); apply/entries take none
) -> bfit.SamplingKind:
    """Build the uniform plain sampling kind by name, at ``x0``'s fixed rank. Only the vector-valued
    **probe** kind is weightable (per-mode ``omega``); plain apply/entries have no mode axis and take no
    weight (a non-``None`` ``weight`` for them is a structural error)."""
    if name == 'probe':
        return UniformProbeKind.from_point(x0_data, weight=weight)
    if name in ('apply', 'entries'):
        if weight is not None:
            raise ValueError(f"the plain '{name}' kind takes no residual weight (no mode or order axis); "
                             "per-mode weighting is defined for probe, per-order for the derivative kinds.")
        return _SAMPLING_KIND[name].from_point(x0_data)
    raise ValueError(f"unknown uniform sampling kind {name!r}; expected one of {sorted(_SAMPLING_KIND)}")


def uniform_derivatives_kind(
        name:       str,        # 'apply_derivatives' / 'entries_derivatives' / 'probe_derivatives'
        x0_data:    typ.Tuple,  # UniformTuckerTensorTrain.data at the fixed rank
        order:      int,
        weight:     typ.Optional[typ.Sequence[float]] = None,
        chunk_size: typ.Optional[int] = 100,   # probe_derivatives only; ignored otherwise
) -> bfit.SamplingKind:
    """Build the uniform derivative sampling kind by name, at ``x0``'s fixed rank."""
    if name not in _DERIV_SAMPLING_KIND:
        raise ValueError(f"unknown uniform derivative kind {name!r}; expected one of "
                         f"{sorted(_DERIV_SAMPLING_KIND)}")
    cls = _DERIV_SAMPLING_KIND[name]
    if name == 'probe_derivatives':            # the only derivative kind with a chunkable assembly
        return cls.from_point(x0_data, order=order, weight=weight, chunk_size=chunk_size)
    return cls.from_point(x0_data, order=order, weight=weight)


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
    geometry, kind_name = geometry.lower(), kind_name.lower()      # 'Manifold', 'PROBE', ... accepted
    if geometry not in ('manifold', 'corewise'):
        raise ValueError("geometry must be 'manifold' or 'corewise'; got %r (a typo used to build the "
                         "corewise geometry silently)" % (geometry,))
    if kind_name not in _SAMPLING_KIND and kind_name not in _DERIV_SAMPLING_KIND:
        raise ValueError("unknown kind_name %r; expected one of %s"
                         % (kind_name, sorted(_SAMPLING_KIND) + sorted(_DERIV_SAMPLING_KIND)))
    if kind_name in _DERIV_SAMPLING_KIND and order is None:
        raise ValueError("derivative kind %r requires order= (the Problem used to build and fail on first use)"
                         % (kind_name,))
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
