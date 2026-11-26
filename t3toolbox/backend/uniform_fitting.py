# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Uniform-layer fitting: packing a least-squares batch for the uniform layout, and the minimal-rank reduction of
a uniform start.

:py:func:`pack_batch` (with :py:func:`pack_sample` and :py:func:`pack_data`) packs the loop-invariant sample and data
of a :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` ONCE, so the optimizers run fully packed -- no
per-matvec pack/unpack: ``newton_cg(geom, oracle, pack_batch(batch, N), (x_sc[0], x_sc[1]))`` with a uniform
geometry (:py:mod:`t3toolbox.backend.geometry`, ``from_point``). :py:func:`uniform_minimal` reduces a
uniform start to minimal ranks, which the uniform manifold geometry requires.
"""
import dataclasses as dc
import typing as typ

import numpy as np

from t3toolbox.backend import fitting as bfit
from t3toolbox.backend import ut3_operations
from t3toolbox.backend import ranks
from t3toolbox.backend.common import *

__all__ = [
    'uniform_minimal',
    'pack_sample',
    'pack_data',
    'pack_batch',
]


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
        batch:  'bfit.LeastSquaresBatch',           # ragged (or already packed: kept)
        N:      typ.Optional[int] = None,           # the packed mode width; None = the largest mode size of the sample
) -> 'bfit.LeastSquaresBatch':                      # the sample's vectors and the probe data packed to N
    """Pack a :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` once for the uniform layout, by its kind and
    jet shape: the input jets with ``pack_if_ragged(sample, N, len_P=len(order))``, which brings each jet axis to its
    largest size across the modes by writing zero coefficients (data, not padding: an input jet is zero beyond its
    slots, O-45), and the probe data ``y`` and weight with :py:func:`pack_data`, the one rule for a data-space object
    (each to ``(d,)+F+W+C+(e,)``, an edge of 1 kept). The entries ``index`` and the data and weight of apply and entries
    are not packed; the scale and the drawn indices pass through. A packed sample, ``y`` or weight is kept as it is.
    ``N=None`` is the largest mode size of a ragged sample (an entries batch is not padded, and a packed sample keeps
    its own width).

    >>> import numpy as np
    >>> from t3toolbox.backend.fitting import LeastSquaresBatch
    >>> from t3toolbox.backend.uniform_fitting import pack_batch
    >>> ww = [np.ones((2, 3, 4)), np.ones((1, 3, 5))]                 # jet sizes 2 and 1, W = (3,)
    >>> y = [np.zeros((3, 3, 4)), np.zeros((3, 3, 5))]                # F = (3,): order (2,)
    >>> packed = pack_batch(LeastSquaresBatch('probe', (2,), ww, y), 6)
    >>> print(packed.sample.shape, packed.y.shape)
    (2, 2, 3, 6) (2, 3, 3, 6)
    """
    kind, order = batch.kind, batch.order
    if N is None and kind != 'entries':
        N = (max(int(np.shape(w)[-1]) for w in batch.sample) if isinstance(batch.sample, (tuple, list))
             else int(np.shape(batch.sample)[-1]))
    y = batch.y if batch.y is None else pack_data(kind, order, batch.y, N)
    weight = batch.weight if batch.weight is None else pack_data(kind, order, batch.weight, N)
    return dc.replace(batch, sample=pack_sample(kind, order, batch.sample, N), y=y, weight=weight)


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
