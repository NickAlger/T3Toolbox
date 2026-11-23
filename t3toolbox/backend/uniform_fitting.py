# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Uniform-layer fitting: packing a least-squares batch for the uniform layout, the minimal-rank reduction of a
uniform start, and the old packed ``SamplingKind`` classes.

:py:func:`pack_batch` (with :py:func:`pack_sample` and :py:func:`pack_data`) packs the loop-invariant sample and data
of a :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` ONCE, so the optimizers run fully packed -- no
per-matvec pack/unpack: ``newton_cg(geom, oracle, pack_batch(kind, order, batch, N), (x_sc[0], x_sc[1]))`` with a
uniform geometry (:py:mod:`t3toolbox.backend.geometry`, ``from_point``). :py:func:`uniform_minimal` reduces a
uniform start to minimal ranks, which the uniform manifold geometry requires.

The uniform twins of :py:mod:`t3toolbox.backend.fitting`'s old kinds -- subclasses that override the
layer-specific operations and, for the probe kinds, where ``omega``'s axes sit in the PACKED output -- are no longer
called by the optimizers (graded messages, P16) and go with the old kinds (P20).
Each carries the fixed rank it was built at (``shape`` + the plain-UT3 ``masks``) as FIELDS, so a
rebuilt kind of the same rank is the same jax cache key
(:py:mod:`t3toolbox.backend.geometry` follows the same rule; the reasoning is
``docs/contributor/parameters_not_closures.md``).

**Protocol note (ruled: keep and document):** the kinds' ``forward`` / ``transpose`` keep
the ragged Kind protocol's ``(v, sample, frame_data, sweep)`` signature even though the packed
``sweep`` already carries the sample and the frame -- the extra arguments are deliberately unread.
Protocol uniformity is the point: generic code (the shared ``LocalModel``) calls every kind the same
way, ragged or uniform.
"""
import dataclasses as dc
import typing as typ

import numpy as np

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
    'pack_sample',
    'pack_data',
    'pack_batch',
]


# --------------------------------------------------------------------------------------------------
# The uniform sampling kinds -- subclasses of the ragged ones (backend.fitting) that override the six
# layer-specific operations (precompute / forward / transpose / point_forward / take / corewise_hessian)
# and, for the probe kinds, where omega's axes sit in the PACKED output. The layer-agnostic half -- the
# reductions, the sample layout, the residual-weight machinery, and the value identity -- is inherited.
#
# Each carries the fixed rank it is built at (`shape` + the plain-UT3 `masks`) as FIELDS, so a rebuilt
# kind of the same rank is the same jax cache key (the same rule as backend.geometry). `forward` derives
# the variation masks from the frame it is handed, and `transpose` re-masks its output with the same masks
# (broadcast over the residual's tangent stack), so the kinds stay geometry-agnostic; only
# `point_forward` (the S(x) op on the plain-UT3 point) needs the held shape + masks.
#
# `corewise_hessian` (the sixth, hessian='exact' only) is overridden by EVERY uniform kind, on its own class:
# `_UniformLayerMixin` precedes the ragged kind in the MRO but defines no corewise_hessian, so a uniform kind that
# did not override it would inherit the ragged (mask-free) method and run it on padded supercores -- a second
# derivative multiplies a direction's padded slot into real slots. Each override masks theta and theta_hat with
# the masks of the FRAME it is handed (frame_data[5]: down and left) and re-masks its output.
# --------------------------------------------------------------------------------------------------
def _frame_theta(
        frame_data:  typ.Tuple,                       # the uniform frame the model's sweep was built at
        base_point:  typ.Tuple[NDArray, NDArray],     # its base point's bare supercores (down, left)
) -> typ.Tuple:                                       # the base point as plain-UT3 .data, with the FRAME's masks
    """The base point as a plain-UT3 ``.data`` carrying the masks of the FRAME it was read off
    (``frame_data[5]``: down and left) -- what every uniform ``corewise_hessian`` masks ``theta`` and ``theta_hat``
    with. Not the sweep's (bare supercores) and not the kind's ``self.masks`` (the point's: a strict superset of the
    frame's at TT slack, which would let the frame's padded completions into ``theta``)."""
    return (base_point[0], base_point[1], tuple(frame_data[4]), (frame_data[5][0], frame_data[5][2]))


@dc.dataclass(frozen=True, eq=False)
class _UniformLayerMixin:
    """The uniform layer's half of a uniform kind: the fixed rank it is built at (``shape`` + the plain-UT3
    ``masks``, as fields) and the packed-layout helpers (the point shim, the frame-derived variation masks, the
    output re-mask). Defaults so the field ordering works alongside the inherited defaulted parameters (Python 3.9
    has no ``kw_only``); use :py:meth:`from_point`."""

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
        """A bare variation supercore pair as variation ``.data``, masked by the frame's gauge shift broadcast
        over the variation's tangent stack ``K`` (``K + C`` variations at a ``C`` frame -- a stack of trial
        tangents, as the exact Hessian's matvecs carry), as :py:meth:`_remask` does for the output."""
        masks = ufv_masking.ufv_variation_masks_over_stack(
            frame_data[5], frame_data[0].shape[1:-2], v_sc[0].shape[1:-2])
        return (v_sc[0], v_sc[1], frame_data[4], masks)

    @staticmethod
    def _remask(v_sc, frame_data):
        """Re-mask a ``transpose`` output (a bare variation supercore pair, stack ``K+C``) with the frame's
        gauge-shifted variation masks broadcast over the tangent stack ``K``, so its padding is exactly
        zero instead of the raw seam's garbage. One multiply by a host mask: the real block is unchanged
        bit for bit, and under jit the masks are folded constants."""
        masks = ufv_masking.ufv_variation_masks_over_stack(
            frame_data[5], frame_data[0].shape[1:-2], v_sc[0].shape[1:-2])
        return ufv_masking.ufv_apply_variations_masks((v_sc[0], v_sc[1], frame_data[4], masks))


@dc.dataclass(frozen=True, eq=False)
class UniformApplyKind(_UniformLayerMixin, bfit.ApplyKind):
    """The uniform twin of :py:data:`~t3toolbox.backend.fitting.APPLY`."""

    def precompute(self, frame_data, ww):
        return utv_sampling.utv_precompute_apply_frame_sweep(frame_data, ww)

    def forward(self, v_sc, ww, frame_data, sweep):
        return utv_sampling.utv_apply_jacobian_from_sweep(self._variations(v_sc, frame_data), sweep)

    def transpose(self, r, ww, frame_data, sweep):
        v_sc = utv_sampling.utv_apply_transpose_from_sweep(r, sweep, sum_over_probes=True)
        return self._remask(v_sc, frame_data)

    def point_forward(self, x_sc, ww):
        return ut3_sampling.ut3_apply(self._point(x_sc), ww)

    def take(self, sample, data, idx):
        return _ptake_apply(sample, data, idx)

    def corewise_hessian(self, core_directions, r, ww, frame_data, sweep, base_point):
        """The masked uniform two-hole (:py:func:`~t3toolbox.backend.ut3_sampling.ut3_apply_corewise_hessian_from_sweep`),
        reusing the model sweep's packed vectors and lean ``(xi, mu)``. ``theta`` and ``theta_hat`` are masked with the
        FRAME's ``(down, left)`` masks (``frame_data[5]``, not ``self.masks``: the point's, a strict superset at TT
        slack), and the output is re-masked with them -- as :py:meth:`UniformProbeDerivativesKind.corewise_hessian`."""
        return ut3_sampling.ut3_apply_corewise_hessian_from_sweep(
            r, _frame_theta(frame_data, base_point), core_directions, sweep)


@dc.dataclass(frozen=True, eq=False)
class UniformEntriesKind(_UniformLayerMixin, bfit.EntriesKind):
    """The uniform twin of :py:data:`~t3toolbox.backend.fitting.ENTRIES`."""

    def precompute(self, frame_data, index):
        return utv_sampling.utv_precompute_entries_frame_sweep(frame_data, index)

    def forward(self, v_sc, index, frame_data, sweep):
        return utv_sampling.utv_entries_jacobian_from_sweep(self._variations(v_sc, frame_data), sweep)

    def transpose(self, r, index, frame_data, sweep):
        v_sc = utv_sampling.utv_entries_transpose_from_sweep(r, sweep, sum_over_probes=True)
        return self._remask(v_sc, frame_data)

    def point_forward(self, x_sc, index):
        return ut3_sampling.ut3_entries(self._point(x_sc), index)

    def take(self, sample, data, idx):
        return _ptake_entries(sample, data, idx)

    def corewise_hessian(self, core_directions, r, index, frame_data, sweep, base_point):
        """The masked uniform two-hole (:py:func:`~t3toolbox.backend.ut3_sampling.ut3_entries_corewise_hessian_from_sweep`),
        masked with the FRAME's ``(down, left)`` masks -- as :py:meth:`UniformApplyKind.corewise_hessian`."""
        return ut3_sampling.ut3_entries_corewise_hessian_from_sweep(
            r, _frame_theta(frame_data, base_point), core_directions, sweep)


@dc.dataclass(frozen=True, eq=False)
class UniformProbeKind(_UniformLayerMixin, bfit.ProbeKind):
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
        v_sc = utv_sampling.utv_probe_transpose_from_sweep(self._apply_weight(r, 2), sweep,
                                                           sum_over_probes=True)
        return self._remask(v_sc, frame_data)

    def point_forward(self, x_sc, ww):
        return ut3_sampling.ut3_probe(self._point(x_sc), ww)

    def take(self, sample, data, idx):
        return _ptake_probe(sample, data, idx)

    def corewise_hessian(self, core_directions, r, ww, frame_data, sweep, base_point):
        """The masked uniform two-hole at ``omega^2 r``
        (:py:func:`~t3toolbox.backend.ut3_sampling.ut3_probe_corewise_hessian_from_sweep`), reusing the model sweep's
        packed vectors and ``xi`` / ``mu``, masked with the FRAME's ``(down, left)`` masks -- as
        :py:meth:`UniformApplyKind.corewise_hessian`."""
        return ut3_sampling.ut3_probe_corewise_hessian_from_sweep(
            self._apply_weight(r, 2), _frame_theta(frame_data, base_point), core_directions, sweep)


@dc.dataclass(frozen=True, eq=False)
class UniformApplyDerivativesKind(_UniformLayerMixin, bfit.ApplyDerivativesKind):
    """The uniform twin of :py:class:`~t3toolbox.backend.fitting.ApplyDerivativesKind`."""

    def precompute(self, frame_data, s):
        return utv_sampling.utv_precompute_apply_frame_sweep_jets(frame_data, s[0], s[1], self.order)

    def forward(self, v_sc, s, frame_data, sweep):
        return utv_sampling.utv_apply_jacobian_derivatives_from_sweep(
            self._variations(v_sc, frame_data), sweep, self.order)

    def transpose(self, r, s, frame_data, sweep):
        v_sc = utv_sampling.utv_apply_transpose_derivatives_from_sweep(
            self._apply_weight(r, 2), sweep, self.order, sum_over_probes=True)
        return self._remask(v_sc, frame_data)

    def point_forward(self, x_sc, s):
        return ut3_sampling.ut3_apply_derivatives(s[0], s[1], self._point(x_sc), self.order)

    def take(self, sample, data, idx):
        return _ptake_deriv_apply(sample, data, idx)

    def corewise_hessian(self, core_directions, r, s, frame_data, sweep, base_point):
        """The masked uniform two-hole at ``omega^2 r``
        (:py:func:`~t3toolbox.backend.ut3_sampling.ut3_apply_corewise_derivatives_hessian_from_sweep`), reusing the
        model sweep's packed sample and lean ``(xi, mu)`` jets, masked with the FRAME's ``(down, left)`` masks -- as
        :py:meth:`UniformProbeDerivativesKind.corewise_hessian`."""
        return ut3_sampling.ut3_apply_corewise_derivatives_hessian_from_sweep(
            self._apply_weight(r, 2), _frame_theta(frame_data, base_point), core_directions, self.order, sweep)


@dc.dataclass(frozen=True, eq=False)
class UniformEntriesDerivativesKind(_UniformLayerMixin, bfit.EntriesDerivativesKind):
    """The uniform twin of :py:class:`~t3toolbox.backend.fitting.EntriesDerivativesKind`."""

    def precompute(self, frame_data, s):
        return utv_sampling.utv_precompute_entries_frame_sweep_jets(frame_data, s[0], s[1], self.order)

    def forward(self, v_sc, s, frame_data, sweep):
        return utv_sampling.utv_entries_jacobian_derivatives_from_sweep(
            self._variations(v_sc, frame_data), sweep, self.order)

    def transpose(self, r, s, frame_data, sweep):
        v_sc = utv_sampling.utv_entries_transpose_derivatives_from_sweep(
            self._apply_weight(r, 2), sweep, self.order, sum_over_probes=True)
        return self._remask(v_sc, frame_data)

    def point_forward(self, x_sc, s):
        return ut3_sampling.ut3_entries_derivatives(s[0], s[1], self._point(x_sc), self.order)

    def take(self, sample, data, idx):
        return _ptake_deriv_entries(sample, data, idx)

    def corewise_hessian(self, core_directions, r, s, frame_data, sweep, base_point):
        """The masked uniform two-hole at ``omega^2 r``
        (:py:func:`~t3toolbox.backend.ut3_sampling.ut3_entries_corewise_derivatives_hessian_from_sweep`), masked with
        the FRAME's ``(down, left)`` masks -- as :py:meth:`UniformApplyDerivativesKind.corewise_hessian`."""
        return ut3_sampling.ut3_entries_corewise_derivatives_hessian_from_sweep(
            self._apply_weight(r, 2), _frame_theta(frame_data, base_point), core_directions, self.order, sweep)


@dc.dataclass(frozen=True, eq=False)
class UniformProbeDerivativesKind(_UniformLayerMixin, bfit.ProbeDerivativesKind):
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
        v_sc = utv_sampling.utv_probe_transpose_derivatives_from_sweep(
            self._apply_weight(r, 2), sweep, self.order, sum_over_probes=True,
            chunk_size=self.chunk_size)
        return self._remask(v_sc, frame_data)

    def point_forward(self, x_sc, s):
        return ut3_sampling.ut3_probe_derivatives(s[0], s[1], self._point(x_sc), self.order)

    def take(self, sample, data, idx):
        return _ptake_deriv_probe(sample, data, idx)

    def corewise_hessian(self, core_directions, r, s, frame_data, sweep, base_point):
        """The masked uniform two-hole at ``omega^2 r``
        (:py:func:`~t3toolbox.backend.ut3_sampling.ut3_probe_corewise_derivatives_hessian_from_sweep`),
        reusing the model sweep's packed sample and ``xi`` / ``mu`` jets. ``theta`` (the base
        point's bare supercores) and ``theta_hat`` are masked with the FRAME's ``(down, left)`` masks -- read off
        ``frame_data``, the frame this model's sweep was built at, not ``self.masks`` (the point's, a strict
        superset at TT slack) and not the sweep (bare supercores) -- and the output is re-masked with them."""
        return ut3_sampling.ut3_probe_corewise_derivatives_hessian_from_sweep(
            self._apply_weight(r, 2), _frame_theta(frame_data, base_point), core_directions, self.order, sweep,
            chunk_size=self.chunk_size)


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
# Packing a batch. Packs the LOOP-INVARIANT sample + data ONCE (the input jets to the supercore mode width N; the
# probe observed data d-list -> a packed (d,)+F+W+(N,) array), so the backend LocalModel/optimizers run FULLY PACKED
# -- no per-matvec pack/unpack (the whole point of the uniform layer). apply/entries scalar data + the entries
# integer index need no packing. The optimizer's state is the bare supercore pair (x0.data[0], x0.data[1]); the masks
# of x0's fixed rank are held by the geometry (docs/uniform_backend_jit_recipe.md).
# --------------------------------------------------------------------------------------------------
def pack_sample(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],       # the jet shape: the number of leading jet axes of each input jet
        sample:  typ.Any,                   # P_in_i⊗V_i (the input jets ww, len=d, elm_shape=F_in_i+W+(Ni,)) | index^d
                                            #   ((d,)+W, int)
        N:       int,                       # the packed mode width
) -> typ.Any:                               # P_in_i⊗V_i packed (d,)+F_in+W+(N,) | index^d unchanged
    """Pack the loop-invariant sample once for the uniform layout: the input jets with
    :py:func:`~t3toolbox.backend.ut3_operations.pack_if_ragged` and ``len_P=len(order)`` (each jet axis brought to its
    largest size across the modes by zero coefficients, which are data: an input jet is zero beyond its slots, O-45;
    the modes padded to ``N`` with zeros). The entries ``index`` is never packed. A packed sample is kept as it is.

    >>> import numpy as np
    >>> from t3toolbox.backend.uniform_fitting import pack_sample
    >>> ww = [np.ones((2, 3, 4)), np.ones((1, 3, 5))]                 # jet sizes 2 and 1, W = (3,)
    >>> print(pack_sample('apply', (1,), ww, 6).shape)
    (2, 2, 3, 6)
    """
    if kind == 'entries':
        return sample
    return ut3_operations.pack_if_ragged(sample, N, len_P=len(order))


def pack_data(
        kind:   str,                        # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],        # the jet shape (the probe data's leading F axes ride along)
        y:      typ.Any,                    # Y* or omega, of the type of z (the data, a weight, a residual):
                                            #   probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C; each axis
                                            #   1 or z's size
        N:      int,                        # the packed mode width
) -> typ.Any:                               # probe: (d,)+F+W+C+(e,), e = 1 or N; apply, entries: y unchanged
    """Pack a data-space object once for the uniform layout: the data ``y``, a weight or a residual, each of the type
    of ``z`` (every axis 1 or ``z``'s size). The probe kind's ``d`` arrays become one ``(d,) + F + W + C + (e,)``
    array: each axis of ``F + W + C`` is the join across the modes (full if any mode has it full; the others are
    broadcast); the edge ``e`` stays 1 if every mode's is 1, else it is ``N``, a full edge padded with zeros and an
    edge of 1 broadcast to ``N`` (never padded, which would put its value on entry 0 only). The engine masks the
    packed object on entry, so its padding is never data. The arrays of apply and entries have no mode edge and are
    returned unchanged, and a packed object is kept as it is.

    >>> import numpy as np
    >>> from t3toolbox.backend.uniform_fitting import pack_data
    >>> y = [np.ones((3, 2, 4)), np.ones((3, 2, 5))]                  # F + W + (Ni,), F = (3,), W = (2,)
    >>> print(pack_data('probe', (2,), y, 6).shape)
    (2, 3, 2, 6)
    >>> per_slot = [np.ones((3, 1, 1)), np.ones((3, 1, 1))]           # a weight per slot: W and the edge of 1
    >>> print(pack_data('probe', (2,), per_slot, 6).shape)            # the edge stays 1
    (2, 3, 1, 1)
    >>> mixed = [np.ones((3, 1, 4)), 2.0 * np.ones((1, 2, 1))]        # an edge of 1 is broadcast, never padded
    >>> print(pack_data('probe', (2,), mixed, 6)[1, 0, 0].tolist())
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    """
    if kind == 'probe':
        return ut3_operations._pack_data_space(y, N)
    return y


def pack_batch(
        kind:   str,                                # 'probe' | 'apply' | 'entries'
        order:  typ.Tuple[int, ...],                # the jet shape
        batch:  'bfit.LeastSquaresBatch',           # ragged (or already packed: kept)
        N:      int,                                # the packed mode width
) -> 'bfit.LeastSquaresBatch':                      # the sample's vectors and the probe data packed to N
    """Pack a :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` once for the uniform layout: the input jets
    with ``pack_if_ragged(sample, N, len_P=len(order))``, which brings each jet axis to its largest size across the
    modes by writing zero coefficients (data, not padding: an input jet is zero beyond its slots, O-45), and the
    probe data ``y`` and weight with :py:func:`pack_data`, the one rule for a data-space object (each to
    ``(d,)+F+W+C+(e,)``, an edge of 1 kept). The entries ``index`` and the data and weight of apply and entries are
    not packed; the scale and the drawn indices pass through. A packed sample, ``y`` or weight is kept as it is.

    >>> import numpy as np
    >>> from t3toolbox.backend.fitting import LeastSquaresBatch
    >>> from t3toolbox.backend.uniform_fitting import pack_batch
    >>> ww = [np.ones((2, 3, 4)), np.ones((1, 3, 5))]                 # jet sizes 2 and 1, W = (3,)
    >>> y = [np.zeros((3, 3, 4)), np.zeros((3, 3, 5))]                # F = (3,): order (2,)
    >>> packed = pack_batch('probe', (2,), LeastSquaresBatch(ww, y), 6)
    >>> print(packed.sample.shape, packed.y.shape)
    (2, 2, 3, 6) (2, 3, 3, 6)
    """
    y = batch.y if batch.y is None else pack_data(kind, order, batch.y, N)
    weight = batch.weight if batch.weight is None else pack_data(kind, order, batch.weight, N)
    return dc.replace(batch, sample=pack_sample(kind, order, batch.sample, N), y=y, weight=weight)

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

    **Uniform fitting requires a minimal frame**
    (:py:meth:`~t3toolbox.backend.geometry.UniformManifoldGeometryOps.from_point` raises otherwise). The reason is
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
