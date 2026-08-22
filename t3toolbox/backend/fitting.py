'''Sampling-kind primitives for the geometry-generic Gauss-Newton fitting model.

The Gauss-Newton model ``m(p) = c + gᵀp + ½ pᵀ(JᵀJ)p`` of a least-squares objective factors over two
*independent* choices, and the fitting layer is generic in both:

  * the sampling **kind** (apply / entries / probe) -- the bare single-sample Jacobian ``𝒥`` / its
    transpose ``𝒥ᵀ`` (from :py:mod:`t3toolbox.backend.probing`), plus the kind-specific ``‖·‖²``
    reduction over the sample stack; and
  * the **geometry** (manifold / corewise) -- the gauge projection ``Π`` (``geometry.project``), with
    the Riemannian forward ``J = 𝒥∘Π`` and gradient ``Jᵀr = Π∘𝒥ᵀr``.

This module supplies the **kind** half: a :py:class:`SamplingKind` descriptor bundling the bare probing
primitives + the reduction per kind (:py:data:`APPLY` / :py:data:`ENTRIES` / :py:data:`PROBE`). The
**geometry** half lives in the geometry (:py:data:`t3toolbox.manifold.MANIFOLD` /
:py:data:`~t3toolbox.manifold.COREWISE`); the frontend :py:class:`t3toolbox.fitting.GaussNewtonModel`
composes the two. The §6.3 corewise substitution ``(U,O,P,Q) -> (U,G,G,G)`` is now nothing but
``CorewiseGeometry.frame`` -- there is no separate corewise backend.

``probe`` is vector-valued (one free mode per probe), so its residual / forward output is a sequence of
``d`` arrays and its reduction sums the free mode too; apply / entries are the scalar all-modes special
case. The objective constant ``c = ½‖r‖²`` is just ``½ · sumsq(r)`` (the same reduction as the model's
quadratic term ``½‖𝒥Πp‖²``), so the kind needs no separate objective function.
'''

import functools as ft
import math
import typing as typ
from dataclasses import dataclass

import numpy as np

import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as pd
from t3toolbox.backend import apply as bapply
from t3toolbox.backend import entries as bentries
from t3toolbox.backend.common import *

__all__ = [
    'SamplingKind',
    'ScalarOutputKind',
    'ProbeOutputKind',
    'ApplyKind',
    'EntriesKind',
    'ProbeKind',
    'ApplyDerivativesKind',
    'EntriesDerivativesKind',
    'ProbeDerivativesKind',
    'APPLY',
    'ENTRIES',
    'PROBE',
    'probe_kind',
    'apply_derivatives_kind',
    'entries_derivatives_kind',
    'probe_derivatives_kind',
    'sumsq_over_samples',
    'sumsq_over_probes',
    'block_sumsq_over_samples',
    'block_sumsq_over_probes',
]


def sumsq_over_samples(
        out:    NDArray,    # scalar-output forward/residual, shape W+C (apply / entries)
        n_w:    int,        # number of leading sample-stack (W) axes
) -> NDArray:               # sum of squares over W, keeping the frame stack C
    '''The ``‖·‖²`` reduction for the scalar-output (apply / entries) kinds: sum ``out**2`` over the
    leading ``n_w`` sample axes, keeping the frame stack ``C``. Used for both the objective ``c = ½‖r‖²``
    and the model's quadratic term ``½‖𝒥 Π p‖²``.'''
    use_jax = is_jax_ndarray(out)
    xnp, _, _ = get_backend(False, use_jax)
    return xnp.sum(out ** 2, axis=tuple(range(n_w)))


def sumsq_over_probes(
        zz:     typ.Union[typ.Sequence[NDArray], NDArray],  # ragged len=d (elm W+C+(Ni,)) OR packed (d,)+W+C+(N,)
        n_w:    int,                                         # number of leading sample-stack (W) axes
) -> NDArray:                           # sum of squares over W and the free mode, summed over d, keep C
    '''The ``‖·‖²`` reduction for the vector-output ``probe`` kind: sum over the leading ``n_w`` sample
    axes and the trailing free mode, keeping the frame stack ``C``, summed over the ``d`` probes. **Mirrors**
    ``zz``'s packedness: a ragged ``len=d`` sequence loops over ``d`` (each ``z_i`` is ``W+C+(Ni,)``); a
    packed ``(d,)+W+C+(N,)`` array sums ``d`` + ``W`` + the padded mode ``N`` in one op (the free-mode
    padding is a zeroed prefix, so it contributes nothing -- the packed inner-loop path).'''
    if not isinstance(zz, (list, tuple)):               # packed (d,)+W+C+(N,): d (axis 0) + W + N, keep C
        use_jax = is_jax_ndarray(zz)
        xnp, _, _ = get_backend(False, use_jax)
        axes = (0,) + tuple(range(1, 1 + n_w)) + (zz.ndim - 1,)
        return xnp.sum(zz ** 2, axis=axes)
    use_jax = is_jax_ndarray(zz[0])
    xnp, _, _ = get_backend(False, use_jax)
    total = None
    for z in zz:
        axes = tuple(range(n_w)) + (z.ndim - 1,)        # W (leading) + the free mode Ni (last); keep C
        s = xnp.sum(z ** 2, axis=axes)
        total = s if total is None else total + s
    return total


# --------------------------------------------------------------------------------------------------
# Per-(mode, order) block reductions -- the `sumsq_*` siblings that KEEP the mode/order axes instead of
# collapsing them, for the Newton-CG diagnostic error table (docs/fitting..§?, dev/newton_display_plan.md).
# They return a 2-D (n_mode, n_order) matrix mirroring the kind's ω[mode, order] shape, and are
# **UNWEIGHTED** (raw ‖r_ij‖², so the table is the honest data-norm relative error and ½·Σ block == the
# unweighted objective). apply/entries have no mode axis (n_mode = 1); plain kinds have no order (n_order = 1).
# --------------------------------------------------------------------------------------------------
def block_sumsq_over_samples(
        out:        NDArray,    # scalar-output residual: (order+1)+W+C (has_order) or W+C
        n_w:        int,        # number of leading W axes (unused -- W + C are summed wholesale)
        has_order:  bool,       # True for the derivative kinds (a leading order axis at axis 0)
) -> NDArray:                   # (1, n_order) -- per-order sum of squares (no mode axis: apply/entries)
    '''Per-``(mode, order)`` sum-of-squares for the scalar-output apply/entries kinds -> a 2-D
    ``(1, n_order)`` matrix (no mode axis -- they contract every mode). Keeps only a leading order axis,
    sums the rest (``W`` + ``C``); **UNWEIGHTED** (raw ``‖r_·j‖²``).'''
    use_jax = is_jax_ndarray(out)
    xnp, _, _ = get_backend(False, use_jax)
    if has_order:
        return xnp.sum(out ** 2, axis=tuple(range(1, out.ndim))).reshape(1, -1)   # keep order (axis 0)
    return xnp.sum(out ** 2).reshape(1, 1)


def block_sumsq_over_probes(
        zz:         typ.Union[typ.Sequence[NDArray], NDArray],  # ragged len=d (elm [order]+W+C+(Ni,)) OR packed (d,)+[order]+W+C+(N,)
        n_w:        int,                    # number of leading W axes (unused; see block_sumsq_over_samples)
        has_order:  bool,
) -> NDArray:                               # (d, n_order) -- per-(mode, order) sum of squares
    '''Per-``(mode, order)`` sum-of-squares for the vector-output probe kinds -> a 2-D ``(d, n_order)``
    matrix (``d`` probe modes). Keeps the mode axis (and a leading order axis, if any) and sums the rest
    (``W`` + ``C`` + the free mode); **UNWEIGHTED**. **Mirrors** ``zz``'s packedness (like
    :py:func:`sumsq_over_probes`): a ragged ``len=d`` list stacks per-mode reductions; a **packed uniform**
    ``(d,)+[order]+W+C+(N,)`` array keeps the leading ``d`` (and order) axes and sums the rest -- the padded
    free mode ``N`` is a zeroed prefix (like ``sumsq``), so it contributes nothing. So the uniform probe
    kinds inherit this verbatim via ``dc.replace`` (no override needed).'''
    if not isinstance(zz, (list, tuple)):               # packed (d,)+[order]+W+C+(N,)
        use_jax = is_jax_ndarray(zz)
        xnp, _, _ = get_backend(False, use_jax)
        if has_order:                                   # keep axis 0 (d) + axis 1 (order); sum the rest
            return xnp.sum(zz ** 2, axis=tuple(range(2, zz.ndim)))
        return xnp.sum(zz ** 2, axis=tuple(range(1, zz.ndim))).reshape(-1, 1)   # keep d only
    use_jax = is_jax_ndarray(zz[0])
    xnp, _, _ = get_backend(False, use_jax)
    rows = [xnp.sum(z ** 2, axis=tuple(range(1, z.ndim))) if has_order else xnp.sum(z ** 2).reshape(1)
            for z in zz]
    return xnp.stack(rows, axis=0)


# --------------------------------------------------------------------------------------------------
# Default-draw layout helpers (the minimal layout the kind exposes so the optimizer can build its
# DEFAULT flat minibatch draw -- a random subset across the whole sample stack W. A USER-supplied draw
# bypasses these entirely.) `n_measurements` = the flat W size; `take` gathers a flat W subset out of
# (sample, data) -- both kind-specific, since W sits in a different place per kind (and the derivative
# data has a leading order axis).
# --------------------------------------------------------------------------------------------------
def _flat_gather(
        arr:     NDArray,    # an array carrying the sample stack W
        w_start: int,        # axis where W begins
        n_w:     int,        # number of W axes (flattened to one, then gathered)
        idx:     NDArray,    # int -- the drawn flat positions over the combined W axis
) -> NDArray:                # arr with its W axes replaced by one axis of len(idx)
    '''Flatten ``arr``'s ``n_w`` W-axes (from ``w_start``) into one and gather ``idx`` along it.'''
    use_jax = is_jax_ndarray(arr)
    xnp, _, _ = get_backend(False, use_jax)
    sh = arr.shape
    return xnp.take(arr.reshape(sh[:w_start] + (-1,) + sh[w_start + n_w:]), idx, axis=w_start)


def _prod_w(arr, w_start, n_w):
    return math.prod(arr.shape[w_start:w_start + n_w])

# `take` per kind: gather a flat W subset. Regular: sample is ww / index, data is W-leading (apply/
# entries scalar, probe list). Derivative: sample is (ww/index, pp), data has a leading order axis (W
# at axis 1). All flatten W to a single axis of size len(idx).
def _take_apply(ww, data, idx):
    n_w = ww[0].ndim - 1
    return [_flat_gather(w, 0, n_w, idx) for w in ww], _flat_gather(data, 0, n_w, idx)

def _take_probe(ww, data, idx):
    n_w = ww[0].ndim - 1
    return [_flat_gather(w, 0, n_w, idx) for w in ww], [_flat_gather(d, 0, n_w, idx) for d in data]

def _take_entries(index, data, idx):
    n_w = index.ndim - 1
    return _flat_gather(index, 1, n_w, idx), _flat_gather(data, 0, n_w, idx)

def _take_deriv_apply(sample, data, idx):
    ww, pp = sample; n_w = ww[0].ndim - 1
    return ([_flat_gather(w, 0, n_w, idx) for w in ww], [_flat_gather(p, 0, n_w, idx) for p in pp]), \
        _flat_gather(data, 1, n_w, idx)

def _take_deriv_entries(sample, data, idx):
    index, pp = sample; n_w = index.ndim - 1
    return (_flat_gather(index, 1, n_w, idx), [_flat_gather(p, 0, n_w, idx) for p in pp]), \
        _flat_gather(data, 1, n_w, idx)

def _take_deriv_probe(sample, data, idx):
    ww, pp = sample; n_w = ww[0].ndim - 1
    return ([_flat_gather(w, 0, n_w, idx) for w in ww], [_flat_gather(p, 0, n_w, idx) for p in pp]), \
        [_flat_gather(d, 1, n_w, idx) for d in data]


def _weight_matrix(
        weight:  typ.Optional[typ.Any],   # None, or array / (nested) sequence -- the raw ω input
        order:   int,                      # highest derivative order (0 for the plain kinds)
        bare:    str,                      # a 1-D input binds to -- 'order' (row) or 'mode' (column)
) -> typ.Optional[NDArray]:                # None, or the canonical 2-D ω[m,o], m in {1,d}, o in {1,order+1}
    '''Normalize a raw residual-weight input to the canonical 2-D matrix ``ω[mode, order]`` (host numpy).

    The residual weight enters the objective ``½‖ω ⊙ r‖²`` only (``sumsq`` scales by ``ω``, ``transpose`` --
    the gradient ``𝒥ᵀ ω²r`` -- by ``ω²``; ``forward`` / ``point_forward`` / ``data`` stay raw). ``ω`` is a
    ``(mode, order)`` matrix: numpy right-alignment makes **order the innermost/most-important axis**, so a
    bare 1-D input binds to each model's innermost axis -- ``bare='order'`` for the derivative kinds (a
    vector is per-order, broadcast over modes -- the backward-compatible rule) and ``bare='mode'`` for plain
    probe (a vector is per-mode; plain probe has no order axis). A 2-D array passes through idempotently (so
    the uniform layer can re-feed a normalized matrix). Validates the order dim ``o in {1, order+1}`` here
    (``order`` is known); the mode dim ``m in {1, d}`` is validated in the frontend (which knows ``d``). The
    frontend also enforces the plain-probe **1-D** contract (rejecting a 2-D ``(d, 1)``); this helper stays
    lenient so the internal re-feed works. Weights are host numpy (static structure, like the uniform masks
    and ``ω`` before it) -- they fold into the compiled program as device constants on the jax path.'''
    if weight is None:
        return None
    w = np.array(weight, dtype=float)     # COPY, never a view: the canonical matrix is part of the
    #                                      kind's VALUE IDENTITY and hence of the jit cache key, so a
    #                                      caller mutating their own array in place (the natural way
    #                                      to write a weight sweep) would otherwise desync the key
    #                                      from the compiled program -- a silent wrong answer.
    if w.ndim == 1:
        w = w[None, :] if bare == 'order' else w[:, None]
    elif w.ndim != 2:
        raise ValueError("residual weight must be 1-D or 2-D (ω[mode, order]); got %d-D of shape %s"
                         % (w.ndim, w.shape))
    w.setflags(write=False)               # and frozen, so the kind cannot be mutated through its own field
    o = w.shape[1]
    if o not in (1, order + 1):
        raise ValueError("residual weight's order dimension must be 1 or order+1=%d; got %d (shape %s)"
                         % (order + 1, o, w.shape))
    return w


def _make_weight(
        w2d:         typ.Optional[NDArray],  # canonical 2-D ω[m,o] from _weight_matrix, or None (ω = 1)
        order_axis:  typ.Optional[int] = 0,  # order axis in an ARRAY x (None => no order axis; o must be 1)
        mode_axis:   typ.Optional[int] = None,  # mode axis in an ARRAY x (None => no mode axis; m must be 1)
) -> typ.Callable:                           # apply_w(x, power) = x · ω**power, broadcast over its axes
    '''Return ``apply_w(x, power) = x · ω**power`` for the canonical weight matrix ``ω = w2d`` (or identity
    if ``None``), covering the three residual/output layouts:

      * **ragged probe** (``x`` a list of ``d`` order-leading arrays): element ``i`` is scaled by the
        order-vector ``ω[i if m>1 else 0]`` along its order axis (axis 0 per element). Plain probe has
        ``o = 1`` (a per-mode scalar); the derivative probe has ``o = order+1``. ``order_axis`` / ``mode_axis``
        are ignored here (order is always axis 0 per element, mode is the list index).
      * **apply / entries array** (no mode axis): ``mode_axis=None`` -> ``m`` must be 1 (else the structural
        "no mode axis" error -- mode weighting is probe-only); the order-vector is placed at ``order_axis``.
      * **packed uniform probe array** (``(d,)+…`` for plain, ``(d,)+(order+1,)+…`` for derivatives):
        ``ω``'s mode axis at ``mode_axis`` (axis 0), order axis at ``order_axis`` (axis 1, or ``None`` for
        the order-free plain probe), 1s elsewhere.'''
    if w2d is None:
        return lambda x, power: x
    m, o = w2d.shape
    def apply_w(x, power):
        wp = w2d ** power
        if isinstance(x, (list, tuple)):                 # ragged probe: a list of d order-leading arrays
            return [xi * wp[i if m > 1 else 0].reshape((o,) + (1,) * (xi.ndim - 1))
                    for i, xi in enumerate(x)]
        if mode_axis is None and m > 1:                  # apply/entries have no mode axis (probe-only)
            raise ValueError("this sampling kind has no mode axis (apply / entries); a per-mode residual "
                             "weight (mode dim %d > 1) is only defined for probe" % m)
        shp = [1] * x.ndim                               # place ω's non-unit axes; 1s broadcast elsewhere
        if mode_axis is not None:
            shp[mode_axis] = m
        if order_axis is not None:
            shp[order_axis] = o
        return x * wp.reshape(tuple(shp))
    return apply_w


# --------------------------------------------------------------------------------------------------
# The sampling kinds
#
# Parameters are FIELDS, not closures (the same rule as backend.geometry): a kind rides as jax
# ``aux_data``, so its hash/eq are part of the compilation cache key, and a kind rebuilt from the same
# parameters must be the SAME key or every rebuilt model recompiles. Value-based hash/eq comes from
# `common.ValueHashedFields` over the fields themselves, so there is nothing to keep in sync -- this
# replaces the hand-maintained `identity` tuple, which was correct only while someone remembered to
# extend it and fell back to object identity for a user-built kind that omitted it.
#
# What varies factors along three axes, and the class structure follows them rather than the six names:
#   * output shape  -- scalar (apply/entries) vs a per-mode vector (probe): the `sumsq` / `block_sumsq`
#                      reductions, supplied by ScalarOutputKind / ProbeOutputKind.
#   * derivative or not -- `has_order`, which adds a leading order axis to the data and one to `n_w`.
#   * layer         -- ragged here, uniform in `uniform_fitting`, which overrides exactly five methods
#                      (precompute / forward / transpose / point_forward / take) and the weight axes.
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)   # eq=False + ValueHashedFields: value hash/eq over the fields
class SamplingKind(ValueHashedFields):
    """A sampling kind: the bare ``J`` / ``J^T`` for one measurement operator, plus its reductions.

    The *what you measure* axis of a fit. Holds the kind-specific operations the geometry-generic
    Gauss-Newton model needs -- the bare Jacobian and its transpose (no gauge ``Pi``: that is the
    geometry's), the ``|.|^2`` reduction over the sample stack, and the point operation ``S(x)`` used to
    form the residual -- plus the minimal layout the default minibatch draw needs.

    Built-in kinds: :py:data:`APPLY`, :py:data:`ENTRIES`, :py:data:`PROBE` (module singletons) and the
    parameterized :py:class:`ProbeKind`, :py:class:`ApplyDerivativesKind`,
    :py:class:`EntriesDerivativesKind`, :py:class:`ProbeDerivativesKind`. The uniform twins are in
    :py:mod:`t3toolbox.backend.uniform_fitting`.

    **Writing your own.** Subclass and supply the five operation methods; you get value identity for free
    provided your parameters are dataclass fields (that is the point -- a kind whose parameters hide in
    closures cannot be compared, and silently recompiles).
    """

    name:              typ.ClassVar[str] = 'sampling'
    has_order:         typ.ClassVar[bool] = False   # True for the derivative kinds (leading order axis)
    has_block_sumsq:   typ.ClassVar[bool] = False   # set True by whoever IMPLEMENTS block_sumsq below;
    #                                                a kind that does not gets the friendly guard in
    #                                                optimizer_display rather than a bare NotImplementedError

    # ---- the five operations a layer supplies -------------------------------------------------
    def precompute(self, frame, sample):
        """The reusable frame sweep at ``frame`` for ``sample`` (computed once per local model)."""
        raise NotImplementedError(
            "%s does not implement precompute(): the reusable frame sweep. Subclass a concrete kind (ApplyKind, ProbeKind, "
            "the *DerivativesKind) to inherit it, or ScalarOutputKind / ProbeOutputKind if you "
            "are writing a new operator from scratch." % type(self).__name__)

    def forward(self, variations, sample, frame, sweep):
        """The bare Jacobian ``J v`` (no gauge)."""
        raise NotImplementedError(
            "%s does not implement forward(): the bare Jacobian J v. Subclass a concrete kind (ApplyKind, ProbeKind, "
            "the *DerivativesKind) to inherit it, or ScalarOutputKind / ProbeOutputKind if you "
            "are writing a new operator from scratch." % type(self).__name__)

    def transpose(self, residual, sample, frame, sweep):
        """The bare transpose ``J^T r``, summed over the sample stack ``W``; raw ``(dU, dG)``."""
        raise NotImplementedError(
            "%s does not implement transpose(): the bare transpose J^T r. Subclass a concrete kind (ApplyKind, ProbeKind, "
            "the *DerivativesKind) to inherit it, or ScalarOutputKind / ProbeOutputKind if you "
            "are writing a new operator from scratch." % type(self).__name__)

    def point_forward(self, x_cores, sample):
        """The point operation ``S(x)`` -- what the residual is measured against."""
        raise NotImplementedError(
            "%s does not implement point_forward(): the point operation S(x). Subclass a concrete kind (ApplyKind, ProbeKind, "
            "the *DerivativesKind) to inherit it, or ScalarOutputKind / ProbeOutputKind if you "
            "are writing a new operator from scratch." % type(self).__name__)

    def take(self, sample, data, idx):
        """Gather a flat ``W`` subset of ``(sample, data)`` -- the default minibatch draw's layout hook."""
        raise NotImplementedError(
            "%s does not implement take(): the minibatch gather. Subclass a concrete kind (ApplyKind, ProbeKind, "
            "the *DerivativesKind) to inherit it, or ScalarOutputKind / ProbeOutputKind if you "
            "are writing a new operator from scratch." % type(self).__name__)

    # ---- sample layout ------------------------------------------------------------------------
    def w_axes(self, sample):
        """The number of leading sample-stack (``W``) axes."""
        raise NotImplementedError(
            "%s does not implement w_axes(): the number of leading sample-stack axes. Subclass a concrete kind (ApplyKind, ProbeKind, "
            "the *DerivativesKind) to inherit it, or ScalarOutputKind / ProbeOutputKind if you "
            "are writing a new operator from scratch." % type(self).__name__)

    def n_measurements(self, sample):
        """The flat ``|W|`` measurement count, for the default draw."""
        raise NotImplementedError(
            "%s does not implement n_measurements(): the flat |W| measurement count. Subclass a concrete kind (ApplyKind, ProbeKind, "
            "the *DerivativesKind) to inherit it, or ScalarOutputKind / ProbeOutputKind if you "
            "are writing a new operator from scratch." % type(self).__name__)

    # ---- reductions (supplied by the output-shape bases below) --------------------------------
    def sumsq(self, out, n_w):
        """``|out|^2`` over ``W`` (and the order / free-mode axes), keeping the frame stack ``C``."""
        raise NotImplementedError(
            "%s does not implement sumsq(): the |.|^2 reduction. Subclass a concrete kind (ApplyKind, ProbeKind, "
            "the *DerivativesKind) to inherit it, or ScalarOutputKind / ProbeOutputKind if you "
            "are writing a new operator from scratch." % type(self).__name__)

    def block_sumsq(self, out, n_w):
        """Per-``(mode, order)`` ``|.|^2`` -> a 2-D matrix; UNWEIGHTED (the honest diagnostic table).

        Optional: only the Newton-CG relative-error display needs it. A kind that implements it must also
        set ``has_block_sumsq = True`` so the display knows (the two output-shape bases below do)."""
        raise NotImplementedError(
            "%s does not implement block_sumsq -- it is optional, and only the Newton-CG relative-error "
            "display needs it. Implement it and set has_block_sumsq = True." % type(self).__name__)

    # ---- the residual weight -------------------------------------------------------------------
    _bare_binds_to:  typ.ClassVar[str] = 'order'          # what a 1-D weight means (see _weight_matrix)
    _order_axis:     typ.ClassVar[typ.Optional[int]] = 0  # omega's order axis in an ARRAY output
    _mode_axis:      typ.ClassVar[typ.Optional[int]] = None   # omega's mode axis in an ARRAY output


    @ft.cached_property
    def _apply_weight(self):
        """``apply_w(x, power) = x * omega**power`` for this kind's output layout.

        A **weightable** kind declares a ``weight`` FIELD holding the canonical 2-D ``omega[mode, order]``
        (:py:class:`ProbeKind`, the three ``*DerivativesKind``). The rest have no such field and no
        weight, which is what the default here means -- it is asked for rather than declared on this base
        because declaring it would place it first in every subclass's constructor signature."""
        return _make_weight(getattr(self, 'weight', None), self._order_axis, self._mode_axis)


@dataclass(frozen=True, eq=False)
class ScalarOutputKind(SamplingKind):
    """Kinds whose output is a scalar per measurement (apply / entries): every mode is contracted, so
    there is no mode axis and a per-mode weight is a structural error."""

    has_block_sumsq = True                      # implemented just below

    def sumsq(self, out, n_w):
        return sumsq_over_samples(self._apply_weight(out, 1), n_w + (1 if self.has_order else 0))

    def block_sumsq(self, out, n_w):
        return block_sumsq_over_samples(out, n_w, has_order=self.has_order)


@dataclass(frozen=True, eq=False)
class ProbeOutputKind(SamplingKind):
    """Kinds whose output is a vector per measurement per mode (probe): one free mode each, so the
    reduction also sums the free mode and the kind carries a mode axis to weight."""

    has_block_sumsq = True                      # implemented just below

    def sumsq(self, out, n_w):
        return sumsq_over_probes(self._apply_weight(out, 1), n_w + (1 if self.has_order else 0))

    def block_sumsq(self, out, n_w):
        return block_sumsq_over_probes(out, n_w, has_order=self.has_order)


# --------------------------------------------------------------------------------------------------
# Sample-layout mixins -- where W lives differs by sample type, not by kind
# --------------------------------------------------------------------------------------------------
class _WwSample:
    """``sample = ww`` (a list of d probe/apply vector stacks)."""
    def w_axes(self, sample):
        return sample[0].ndim - 1

    def n_measurements(self, sample):
        return _prod_w(sample[0], 0, sample[0].ndim - 1)


class _IndexSample:
    """``sample = index`` (an integer grid, mode index leading)."""
    def w_axes(self, sample):
        return sample.ndim - 1

    def n_measurements(self, sample):
        return _prod_w(sample, 1, sample.ndim - 1)


class _WwPpSample:
    """``sample = (ww, pp)`` -- the derivative pairing."""
    def w_axes(self, sample):
        return sample[0][0].ndim - 1

    def n_measurements(self, sample):
        return _prod_w(sample[0][0], 0, sample[0][0].ndim - 1)


class _IndexPpSample:
    """``sample = (index, pp)`` -- the derivative pairing at grid points."""
    def w_axes(self, sample):
        return sample[0].ndim - 1

    def n_measurements(self, sample):
        return _prod_w(sample[0], 1, sample[0].ndim - 1)


# --------------------------------------------------------------------------------------------------
# The six ragged kinds
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ApplyKind(_WwSample, ScalarOutputKind):
    """The all-modes ``apply``: one scalar per measurement. Unweighted (no mode or order axis)."""

    name = 'apply'

    def precompute(self, frame, ww):
        return bapply.tv_precompute_apply_frame_sweep(frame, ww)

    def forward(self, v, ww, frame, sweep):
        return bapply.tv_apply_jacobian_from_sweep(v, ww, frame, sweep)

    def transpose(self, r, ww, frame, sweep):
        return bapply.tv_apply_transpose_from_sweep(r, ww, frame, sweep, sum_over_probes=True)

    def point_forward(self, x_cores, ww):
        return bapply.t3_apply(x_cores, ww)

    def take(self, sample, data, idx):
        return _take_apply(sample, data, idx)


@dataclass(frozen=True, eq=False)
class EntriesKind(_IndexSample, ScalarOutputKind):
    """The all-modes ``entries``: one scalar per multi-index. Unweighted."""

    name = 'entries'

    def precompute(self, frame, index):
        return bentries.tv_precompute_entries_frame_sweep(frame, index)

    def forward(self, v, index, frame, sweep):
        return bentries.tv_entries_jacobian_from_sweep(v, index, frame, sweep)

    def transpose(self, r, index, frame, sweep):
        return bentries.tv_entries_transpose_from_sweep(r, index, frame, sweep, sum_over_probes=True)

    def point_forward(self, x_cores, index):
        return bentries.t3_entries(x_cores, index)

    def take(self, sample, data, idx):
        return _take_entries(sample, data, idx)


@dataclass(frozen=True, eq=False)
class ProbeKind(_WwSample, ProbeOutputKind):
    """The vector-valued ``probe`` (one free mode per measurement), optionally **per-mode** weighted.

    Mode weighting is the order-0 case of the same residual-weight machinery as the derivative kinds:
    the objective is ``1/2 sum_i |omega_i z_i|^2``, so ``omega`` enters ``sumsq`` (x omega) and
    ``transpose`` (x omega^2) only. Plain probe has no order axis, so the weight is a 1-D ``(d,)``
    per-mode vector."""

    name = 'probe'
    _bare_binds_to = 'mode'

    weight:  typ.Optional[typ.Any] = None   # per-mode omega, (d,) / (d,1); None = unweighted

    def __post_init__(self):
        # canonicalize once, so equal-but-differently-spelled weights are the SAME cache key
        object.__setattr__(self, 'weight', _weight_matrix(self.weight, 0, self._bare_binds_to))

    def precompute(self, frame, ww):
        return probing.tv_precompute_probe_frame_sweep(frame, ww)

    def forward(self, v, ww, frame, sweep):
        return probing.tv_probe_jacobian_from_sweep(v, ww, frame, sweep)

    def transpose(self, r, ww, frame, sweep):
        return probing.tv_probe_transpose_from_sweep(self._apply_weight(r, 2), ww, frame, sweep,
                                                     sum_over_probes=True)

    def point_forward(self, x_cores, ww):
        return probing.t3_probe(ww, x_cores)

    def take(self, sample, data, idx):
        return _take_probe(sample, data, idx)


@dataclass(frozen=True, eq=False)
class _DerivativesKind(SamplingKind):
    """Shared parameters of the symmetric directional-derivative (jet) kinds: the highest ``order`` and
    an optional residual weight. The data and outputs gain a leading order axis, so the reductions count
    it via ``n_w + 1``; ``omega`` enters ``sumsq`` (x omega) and ``transpose`` (x omega^2) only, leaving
    ``forward`` / ``point_forward`` raw (the user passes RAW data + omega)."""

    has_order = True

    order:   int = 0                        # highest derivative order
    weight:  typ.Optional[typ.Any] = None   # omega; see the concrete kinds for its shape

    def __post_init__(self):
        object.__setattr__(self, 'weight', _weight_matrix(self.weight, self.order, self._bare_binds_to))


@dataclass(frozen=True, eq=False)
class ApplyDerivativesKind(_WwPpSample, ScalarOutputKind, _DerivativesKind):
    """Symmetric directional derivatives of the all-modes apply, orders ``0..order``, in direction ``P``.
    ``sample = (ww, pp)``. All-modes apply has no mode axis, so ``weight`` is **order-only**."""

    name = 'apply_derivatives'

    def precompute(self, frame, s):
        return pd.tv_precompute_apply_frame_sweep_jets(frame, s[0], s[1], self.order)

    def forward(self, v, s, frame, sweep):
        return pd.tv_apply_jacobian_derivatives_from_sweep(v, s[0], s[1], frame, sweep, self.order)

    def transpose(self, r, s, frame, sweep):
        return pd.tv_apply_transpose_derivatives_from_sweep(
            self._apply_weight(r, 2), s[0], s[1], frame, sweep, self.order, sum_over_probes=True)

    def point_forward(self, x_cores, s):
        return pd.t3_apply_derivatives(s[0], s[1], x_cores, self.order)

    def take(self, sample, data, idx):
        return _take_deriv_apply(sample, data, idx)


@dataclass(frozen=True, eq=False)
class EntriesDerivativesKind(_IndexPpSample, ScalarOutputKind, _DerivativesKind):
    """:py:class:`ApplyDerivativesKind` at integer grid points. ``sample = (index, pp)``. Order-only weight."""

    name = 'entries_derivatives'

    def precompute(self, frame, s):
        return pd.tv_precompute_entries_frame_sweep_jets(frame, s[0], s[1], self.order)

    def forward(self, v, s, frame, sweep):
        return pd.tv_entries_jacobian_derivatives_from_sweep(v, s[0], s[1], frame, sweep, self.order)

    def transpose(self, r, s, frame, sweep):
        return pd.tv_entries_transpose_derivatives_from_sweep(
            self._apply_weight(r, 2), s[0], s[1], frame, sweep, self.order, sum_over_probes=True)

    def point_forward(self, x_cores, s):
        return pd.t3_entries_derivatives(s[0], s[1], x_cores, self.order)

    def take(self, sample, data, idx):
        return _take_deriv_entries(sample, data, idx)


@dataclass(frozen=True, eq=False)
class ProbeDerivativesKind(_WwPpSample, ProbeOutputKind, _DerivativesKind):
    """Vector-valued derivative probing: the residual is a list of ``d`` order-leading arrays.
    ``sample = (ww, pp)``. Probe has both a mode and an order axis, so ``weight`` is the full
    ``omega[mode, order]`` matrix (a row = per-order, a column = per-mode, a matrix = both)."""

    name = 'probe_derivatives'

    chunk_size:  typ.Optional[int] = 100   # W-chunk size for the J^T assembly (docs/chunking.md)

    def precompute(self, frame, s):
        return pd.tv_precompute_probe_frame_sweep_jets(frame, s[0], s[1], self.order)

    def forward(self, v, s, frame, sweep):
        return pd.tv_probe_jacobian_derivatives_from_sweep(v, s[0], s[1], frame, sweep, self.order)

    def transpose(self, r, s, frame, sweep):
        return pd.tv_probe_transpose_derivatives_from_sweep(
            self._apply_weight(r, 2), s[0], s[1], frame, sweep, self.order,
            sum_over_probes=True, chunk_size=self.chunk_size)

    def point_forward(self, x_cores, s):
        return pd.t3_probe_derivatives(s[0], s[1], x_cores, self.order)

    def take(self, sample, data, idx):
        return _take_deriv_probe(sample, data, idx)


APPLY = ApplyKind()      # the plain module singletons
ENTRIES = EntriesKind()
PROBE = ProbeKind()

# Constructor aliases -- the pre-class spelling, kept because it reads better at a call site
# (`probe_derivatives_kind(order, weight)`) and because it is the documented public surface.
probe_kind = ProbeKind
apply_derivatives_kind = ApplyDerivativesKind
entries_derivatives_kind = EntriesDerivativesKind
probe_derivatives_kind = ProbeDerivativesKind
