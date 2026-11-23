'''Sampling-kind primitives for the geometry-generic Gauss-Newton fitting model.

The Gauss-Newton model ``m(p) = c + gᵀp + ½ pᵀ(JᵀJ)p`` of a least-squares objective factors over two
*independent* choices, and the fitting layer is generic in both:

  * the sampling **kind** (apply / entries / probe) -- the bare single-sample Jacobian ``𝒥`` / its
    transpose ``𝒥ᵀ`` (from :py:mod:`t3toolbox.backend.probing`), plus the kind-specific ``‖·‖²``
    reduction over the probe stack; and
  * the **geometry** (manifold / corewise) -- the gauge projection ``Π`` (``geometry.project``), with
    the Riemannian forward ``J = 𝒥∘Π`` and gradient ``Jᵀr = Π∘𝒥ᵀr``.

This module supplies the **kind** half: a :py:class:`SamplingKind` descriptor bundling the bare probing
primitives + the reduction per kind (:py:data:`APPLY` / :py:data:`ENTRIES` / :py:data:`PROBE`). The
**geometry** half lives in the geometry (:py:data:`t3toolbox.manifold.MANIFOLD` /
:py:data:`~t3toolbox.manifold.COREWISE`); the frontend :py:class:`t3toolbox.fitting.LocalModel`
composes the two. The §6.3 corewise substitution ``(D,U,L,R) -> (B,G,G,G)`` is now nothing but
``CorewiseGeometry.frame`` -- there is no separate corewise backend.

``probe`` is vector-valued (one free mode per probe), so its residual / forward output is a sequence of
``d`` arrays and its reduction sums the free mode too; apply / entries are the scalar all-modes special
case. The objective constant ``c = ½‖r‖²`` is just ``½ · sumsq(r)`` (the same reduction as the model's
quadratic term ``½‖𝒥Πp‖²``), so the kind needs no separate objective function.
'''

import abc
import functools as ft
import math
import typing as typ
from dataclasses import dataclass

import numpy as np

import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as pd
from t3toolbox.backend import apply as bapply
from t3toolbox.backend import entries as bentries
import t3toolbox.backend.sampling_variables as sampling_variables
import t3toolbox.backend.jets as jets
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import *

if jax_available:
    import jax

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
    'LeastSquaresOracle',
    'LeastSquaresBatch',
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
        zz:     typ.Union[typ.Sequence[NDArray], NDArray],   # ragged len=d (elm W+C+(Ni,)) OR packed (d,)+W+C+(N,)
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
# collapsing them, for the Newton-CG diagnostic error table (docs/fitting_and_optimization.md §2.1, the `verbose=` display).
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
# DEFAULT flat minibatch draw -- a random subset across the whole probe stack W. A USER-supplied draw
# bypasses these entirely.) `n_measurements` = the flat W size; `take` gathers a flat W subset out of
# (sample, data) -- both kind-specific, since W sits in a different place per kind (and the derivative
# data has a leading order axis).
# --------------------------------------------------------------------------------------------------
def _flat_gather(
        arr:     NDArray,    # an array carrying the probe stack W
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
        weight:  typ.Optional[typ.Any],    # None, or array / (nested) sequence -- the raw ω input
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

    def check_modes(d):
        # A per-mode weight with MORE rows than modes is truncated to the first d (intended: a continuation
        # scheme can carry one weight across rank/mode stages); FEWER rows (but not the broadcast 1) is a
        # structural error here, not an IndexError deep inside the reduction.
        if 1 < m < d:
            raise ValueError("residual weight's mode dimension must be 1 or >= d=%d (extra rows are ignored); "
                             "got %d (shape %s)" % (d, m, w2d.shape))

    def apply_w(x, power):
        wp = w2d ** power
        if isinstance(x, (list, tuple)):                 # ragged probe: a list of d order-leading arrays
            check_modes(len(x))
            return [xi * wp[i if m > 1 else 0].reshape((o,) + (1,) * (xi.ndim - 1))
                    for i, xi in enumerate(x)]
        if mode_axis is None and m > 1:                  # apply/entries have no mode axis (probe-only)
            raise ValueError("this sampling kind has no mode axis (apply / entries); a per-mode residual "
                             "weight (mode dim %d > 1) is only defined for probe" % m)
        shp = [1] * x.ndim                               # place ω's non-unit axes; 1s broadcast elsewhere
        if mode_axis is not None:
            d = x.shape[mode_axis]
            check_modes(d)
            if m > d:
                wp = wp[:d]                              # packed layout cannot broadcast the extra rows
            shp[mode_axis] = min(m, d)
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
#   * layer         -- ragged here, uniform in `uniform_fitting`, which overrides the layer-specific
#                      operations (precompute / forward / transpose / point_forward / take, and the
#                      masked corewise_hessian) and the weight axes.
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)   # eq=False + ValueHashedFields: value hash/eq over the fields
class SamplingKind(ValueHashedFields):
    """A sampling kind: the bare ``J`` / ``J^T`` for one measurement operator, plus its reductions.

    The *what you measure* axis of a fit. Holds the kind-specific operations the geometry-generic
    Gauss-Newton model needs -- the bare Jacobian and its transpose (no gauge ``Pi``: that is the
    geometry's), the ``|.|^2`` reduction over the probe stack, and the point operation ``S(x)`` used to
    form the residual -- plus the minimal layout the default minibatch draw needs.

    Built-in kinds: :py:data:`APPLY`, :py:data:`ENTRIES`, :py:data:`PROBE` (module singletons) and the
    parameterized :py:class:`ProbeKind`, :py:class:`ApplyDerivativesKind`,
    :py:class:`EntriesDerivativesKind`, :py:class:`ProbeDerivativesKind`. The uniform twins are in
    :py:mod:`t3toolbox.backend.uniform_fitting`.

    **Writing your own.** Subclass and supply the six operation methods; you get value identity for free
    provided your parameters are dataclass fields (that is the point -- a kind whose parameters hide in
    closures cannot be compared, and silently recompiles). The sixth, :py:meth:`corewise_hessian` (the
    derivative of ``transpose``'s corewise form), is needed only for the exact Hessian
    (``hessian='exact'``): a kind without it keeps working with the default ``hessian='gauss_newton'``,
    and ``Problem.local_model(hessian='exact')`` rejects it up front with a structural error.
    """

    name:              typ.ClassVar[str] = 'sampling'
    has_order:         typ.ClassVar[bool] = False   # True for the derivative kinds (leading order axis)
    has_block_sumsq:   typ.ClassVar[bool] = False   # set True by whoever IMPLEMENTS block_sumsq below;
    #                                                a kind that does not gets the friendly guard in
    #                                                optimizer_display rather than a bare NotImplementedError

    # ---- the six operations a layer supplies --------------------------------------------------
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
        """The bare transpose ``J^T r``, summed over the probe stack ``W``; raw ``(V, H)``."""
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

    @abc.abstractmethod     # the stdlib "unimplemented" marker: Problem.local_model(hessian='exact') reads it
    def corewise_hessian(
            self,
            core_directions,   # theta_hat = (Bdot, Gdot) at base_point; elm_shape=K+C+(...) (raw core shapes)
            residual,          # RAW r = S(x) - data, as transpose receives it; omega^2 is applied HERE
            sample,            # as forward / transpose take it
            frame,             # this model's frame -- the slot forward / transpose take
            sweep,             # this model's precompute(frame, sample); its xi / mu half is what is reused
            base_point,        # theta = geom.base_point(frame): the raw cores the directions vary
    ):                         # -> raw (dB, dG), stack K+C, probe stack W summed; MASK-FREE by contract
        """The derivative of :py:meth:`transpose`'s **corewise** form along a core direction.

        ``K_r theta_hat = d/ds grad_theta <omega^2 r, S(Phi(theta + s theta_hat))> |_{s=0}`` at
        ``theta = base_point`` -- the core-coordinate Hessian of ``theta -> <omega^2 r, S(Phi(theta))>``
        applied to ``theta_hat``, with the residual ``r`` held fixed (raw, weighted here exactly as
        ``transpose`` weights it). It is the sampling half of the exact Hessian's curvature term; the
        geometry lifts a tangent to ``theta_hat`` and maps the result back.

        Needed **only** for ``hessian='exact'``. Raw ``(dB, dG)`` shaped like the directions (``K + C``),
        ``W`` summed. Mask-free by contract: on the uniform layer the kind's own override masks ``theta``
        and ``theta_hat`` with the frame's masks and re-masks the output.

        The base method is marked with :py:func:`abc.abstractmethod` (the class is not an ``ABC``, so the
        marker changes nothing about instantiation): ``Problem.local_model(hessian='exact')`` reads the marker
        to reject a kind that does not implement it before any work, while the Gauss-Newton path never
        consults it."""
        raise NotImplementedError(
            "%s does not implement corewise_hessian(): the directional derivative of the corewise transpose, "
            "which hessian='exact' needs. Subclass a concrete kind (ApplyKind, ProbeKind, the *DerivativesKind) "
            "to inherit it, or implement it for a new operator; a kind without it still works with "
            "hessian='gauss_newton'." % type(self).__name__)

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

    def corewise_hessian(self, core_directions, r, ww, frame, sweep, base_point):
        # the lean sweep's xi / mu are base_point's own (xi through D, mu through L at the manifold frame
        # (D, U, L, R); through B and G at the corewise frame) -- unweighted, so the residual is r itself
        return bapply.t3_apply_corewise_hessian_from_sweep(r, ww, base_point, core_directions, sweep)


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

    def corewise_hessian(self, core_directions, r, index, frame, sweep, base_point):
        # as ApplyKind's: the lean (fiber-sliced) sweep is base_point's own
        return bentries.t3_entries_corewise_hessian_from_sweep(r, index, base_point, core_directions, sweep)


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
        return probing.t3_probe(x_cores, ww)

    def take(self, sample, data, idx):
        return _take_probe(sample, data, idx)

    def corewise_hessian(self, core_directions, r, ww, frame, sweep, base_point):
        # the probe sweep's xi / mu are base_point's own (its nu belongs to the manifold frame's R and is re-swept
        # through G by the kernel); omega^2 r, exactly as transpose weights it
        return probing.t3_probe_corewise_hessian_from_sweep(self._apply_weight(r, 2), ww, base_point,
                                                            core_directions, sweep)


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

    def corewise_hessian(self, core_directions, r, s, frame, sweep, base_point):
        # the lean sweep's xi / mu jets are base_point's own (xi through D, mu through L at the manifold frame
        # (D, U, L, R); through B and G at the corewise frame); omega^2 r, exactly as transpose weights it
        return pd.t3_apply_corewise_derivatives_hessian_from_sweep(
            self._apply_weight(r, 2), s[0], s[1], base_point, core_directions, self.order, sweep)


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

    def corewise_hessian(self, core_directions, r, s, frame, sweep, base_point):
        # as ApplyDerivativesKind's: the lean (fiber-sliced) jet sweep is base_point's own
        return pd.t3_entries_corewise_derivatives_hessian_from_sweep(
            self._apply_weight(r, 2), s[0], s[1], base_point, core_directions, self.order, sweep)


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

    def corewise_hessian(self, core_directions, r, s, frame, sweep, base_point):
        # the sweep's xi / mu are base_point's own: the manifold frame (D, U, L, R) sweeps xi through D and mu
        # through L, and the corewise frame (B, G, G, G) through B and G -- base_point is (D, L) / (B, G)
        return pd.t3_probe_corewise_derivatives_hessian_from_sweep(
            self._apply_weight(r, 2), s[0], s[1], base_point, core_directions, self.order, sweep,
            chunk_size=self.chunk_size)


APPLY = ApplyKind()      # the plain module singletons
ENTRIES = EntriesKind()
PROBE = ProbeKind()

# Constructor aliases -- the pre-class spelling, kept because it reads better at a call site
# (`probe_derivatives_kind(order, weight)`) and because it is the documented public surface.
probe_kind = ProbeKind
apply_derivatives_kind = ApplyDerivativesKind
entries_derivatives_kind = EntriesDerivativesKind
probe_derivatives_kind = ProbeDerivativesKind


#####################################################################
#########    The least-squares oracle and its batch (P14)    ########
#####################################################################
# The fitting layer's new extension point (graded messages, Part 1): an oracle answers a query at a frame through
# the engine, its batch holds the data. Beside the old kinds above, which the optimizers use until P16.

_KINDS = ('probe', 'apply', 'entries')
_HOLD_LIST = ('xi', 'mu', 'nu', 'eta', 'r')            # Part 1's fixed hold list (7.2, "as today")
_SCALED = ('f', 'g', 'JtJp', 'Jp_norm_sq', 'Kp', 'r_block_sumsq')   # the minibatch scale's reach (O-21), with the
                                                                      #   oracle's own y_block_sumsq and f_resolution


@dataclass(frozen=True, eq=False)
class LeastSquaresBatch:
    """The data of a least-squares fit, arrays only: a registered pytree, every field a leaf (``None`` allowed).

    ``sample`` is the input jets ``ww`` (probe, apply: ``len=d``, ``elm_shape=F_in_i+W+(Ni,)``, or packed
    ``(d,)+F_in+W+(N,)`` on the uniform layout, :py:func:`~t3toolbox.backend.uniform_fitting.pack_batch`), or the
    integer ``index`` ``(d,)+W`` (entries). ``y`` and ``weight`` are **of the type of** ``z``, with the model's stack
    of base points ``C``: probe ``F+W+C+(Ni,)`` per mode (a tuple or a list of ``d`` arrays, or one array whose first
    axis is the mode; packed ``(d,)+F+W+C+(e,)`` on the uniform layout), apply ``F+W+C``, entries ``W+C``; each axis 1
    or that size, a size of 1 meaning the same along the axis (data shared by a stack have size-1 ``C`` axes, a
    weight per slot size-1 ``W``, ``C`` and edge axes). :py:func:`~t3toolbox.backend.jets.blocks_to_weight` makes a
    weight from one value per block. ``weight=None`` is ones. ``minibatch_scale`` is ``n / batch_size`` on a drawn
    batch (:py:meth:`LeastSquaresOracle.take`), ``None`` (one) on a full batch; ``drawn_indices`` the flat sample
    indices a draw took, relative to the batch it was taken from. Checked by
    :py:meth:`LeastSquaresOracle.check_batch`, never by the oracle's ``query`` (the engine checks the rule of ``y``
    and the weight on entry).
    """
    sample:           typ.Any                                    # P_in_i⊗V_i (the input jets ww) | index^d (entries)
    y:                typ.Any = None                             # Y*: probe len=d, F+W+C+(Ni,); apply F+W+C; entries W+C
    weight:           typ.Any = None                             # omega, of the type of z likewise; None = ones
    minibatch_scale:  typ.Optional[typ.Any] = None               # gamma: n / batch_size on a drawn batch; None = 1
    drawn_indices:    typ.Optional[NDArray] = None               # int, (batch_size,): the draw, flat over W

    def __eq__(self, other):
        raise TypeError('LeastSquaresBatch: == is not defined on array-carrying records; compare the fields you mean')

    __hash__ = None

    def data_metric(
            self,
            r:  typ.Any,        # Y*, of the type of z (no K), on this batch's samples
    ) -> typ.Any:               # M r: Y, minibatch_scale * weight**2 * r elementwise, r's structure
        """The data metric of this batch: :py:func:`~t3toolbox.backend.jets.data_metric` with the batch's own weight
        and minibatch scale, ``data_metric(r, self.weight, self.minibatch_scale)``. The call that cannot forget the
        scale of a drawn batch.

        The weight must be of the type of ``z`` (each axis 1 or ``z``'s size), the form
        :py:func:`~t3toolbox.backend.jets.blocks_to_weight` makes. A batch whose weight is one value per block,
        ``(d,) + F``, ``F`` or ``()``, is not of that type, and raises here by rank, except where the two forms
        coincide (an apply or entries batch with no sample or stack axes).

        >>> import numpy as np
        >>> from t3toolbox.backend.fitting import LeastSquaresBatch
        >>> from t3toolbox.backend.jets import blocks_to_weight
        >>> y = np.ones((3, 4))                                             # apply: F + W, F = (3,), W = (4,)
        >>> batch = LeastSquaresBatch(None, y, blocks_to_weight('apply', (2,), np.array([1.0, 2.0, 3.0]), y), 0.5)
        >>> print(batch.data_metric(y)[:, 0].tolist())                     # 0.5 * weight**2 * r
        [0.5, 2.0, 4.5]
        """
        return jets.data_metric(r, self.weight, self.minibatch_scale)


_BATCH_FIELDS = ('sample', 'y', 'weight', 'minibatch_scale', 'drawn_indices')

if jax_available:
    jax.tree_util.register_pytree_node(
        LeastSquaresBatch,
        lambda b: (tuple(getattr(b, name) for name in _BATCH_FIELDS), None),
        lambda aux, children: LeastSquaresBatch(*children))


def _same_leaves(
        a:  typ.Any,                        # a direction: a pair of tuples of arrays, or a pair of supercores
        b:  typ.Any,
) -> bool:                                  # the same object, or the same arrays leaf by leaf (identity)
    '''Whether two directions are the one direction: the same object, or containers of the very same arrays (a rebuilt
    tuple of the same arrays is the same direction). Never compares values.'''
    if a is b:
        return True
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same_leaves(p, q) for p, q in zip(a, b))
    return False


def _scale_tree(
        value:  typ.Any,                    # an array, or nested tuples of arrays (the blocks)
        scale:  typ.Any,                    # a float or a 0-d array (traced under jit)
) -> typ.Any:
    if isinstance(value, (tuple, list)):
        return tuple(_scale_tree(item, scale) for item in value)
    return value * scale


def _arrays_of(
        kind:   str,                        # 'probe' | 'apply' | 'entries'
        value:  typ.Any,                    # of the type of z: probe a tuple or a list of d arrays, or one array whose
                                            #   first axis is the mode (packed or not); else one array; None
        shape:  typ.Tuple[int, ...],        # the mode sizes N_i (a packed probe object is sliced to them)
) -> typ.List[typ.Any]:                     # the arrays, one per mode (probe) or one; [None] * count for None
    '''The arrays of a data-space object, unpadded: a packed probe object (``(d,) + F + W + C + (e,)``, the edge ``e``
    1 or the padded ``N``) sliced to each mode's ``N_i`` (an edge of 1 stays 1).'''
    count = len(shape) if kind == 'probe' else 1
    if value is None:
        return [None] * count
    if kind != 'probe':
        return [value]
    if isinstance(value, (tuple, list)):
        return list(value)
    return list(ut3_operations.unpack_vectors(value, shape))


def _stack_of_frame(
        frame:  typ.Tuple,                  # any of the four forms (sampling_variables.frame_form)
) -> typ.Tuple[int, ...]:                   # C, the stack of base points
    '''The stack of base points ``C`` of a frame or a corewise point: the leading axes of the first Tucker core
    (``D[0]`` or ``B[0]``) on the ragged layout, the axes after ``d`` of the Tucker supercore on the uniform one.
    The data and the weight cannot give it: shared by the stack, they have size-1 ``C`` axes.'''
    layout, _ = sampling_variables.frame_form(frame)
    if layout == 'uniform':
        return tuple(int(n) for n in np.shape(frame[0])[1:-2])
    return tuple(int(n) for n in np.shape(frame[0][0])[:-2])


def _y_times_metric_y(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],       # the jet shape
        y:       typ.Any,                   # Y*, the data, of the type of z: probe len=d (or (d,)+..., packed or not),
                                            #   elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C; each axis 1 or z's size
        weight:  typ.Any,                   # omega, of the type of z likewise; None = ones
        W:       typ.Tuple[int, ...],       # the sample stack, read off the sample
        C:       typ.Tuple[int, ...],       # the stack of base points
        shape:   typ.Tuple[int, ...],       # the mode sizes N_i
) -> typ.List[NDArray]:                     # y * M y at gamma = 1, entrywise at z's shape: per mode (probe), or one
    '''``y * data_metric(y, omega, None)`` at ``z``'s shape, one array per mode for probe (one otherwise): ``y`` is
    broadcast to ``z``'s shape first, so an axis where ``y`` has size 1 and ``z`` does not counts every entry of ``z``
    along it, as ``f`` and ``r_block_sumsq`` do. The two data sums below reduce it.'''
    F = tuple(n + 1 for n in order)
    xnp, _, _ = get_backend(False, tree_contains_jax((y, weight)))
    if kind == 'probe':
        targets = [F + W + C + (N,) for N in shape]
    else:
        targets = [(F + W + C) if kind == 'apply' else (W + C)]
    products = []
    for y_i, omega_i, z_shape in zip(_arrays_of(kind, y, shape), _arrays_of(kind, weight, shape), targets):
        y_full = xnp.broadcast_to(xnp.asarray(y_i), z_shape)
        products.append(y_full * jets.data_metric(y_full, omega_i, None))
    return products


def _y_metric_y(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],       # the jet shape
        y:       typ.Any,                   # Y*, the data, of the type of z (_y_times_metric_y)
        weight:  typ.Any,                   # omega, of the type of z likewise; None = ones
        W:       typ.Tuple[int, ...],       # the sample stack, read off the sample
        C:       typ.Tuple[int, ...],       # the stack of base points (the shape of f)
        shape:   typ.Tuple[int, ...],       # the mode sizes N_i
) -> NDArray:                               # <y, M y> at gamma = 1 over z's shape, keeping C. shape=C
    '''``<y, M y>`` with ``M`` the data metric at ``gamma = 1``, summed over every entry of ``z`` (the modes, the slots
    of the box ``F``, the samples ``W`` and the mode edge), one sum per element of ``C``, like ``f``. ``y`` is
    broadcast to ``z``'s shape first: an axis where ``y`` has size 1 and ``z`` does not counts every entry of ``z``
    along it, as ``f`` does (the floor bounds the rounding of ``f``, which sums every sample). The metric is
    :py:func:`~t3toolbox.backend.jets.data_metric`, the library's one ``omega**2``.'''
    m = len(order)
    xnp, _, _ = get_backend(False, tree_contains_jax((y, weight)))
    total = None
    for product in _y_times_metric_y(kind, order, y, weight, W, C, shape):
        summed_axes = tuple(range(m + len(W))) + ((product.ndim - 1,) if kind == 'probe' else ())
        summed = xnp.sum(product, axis=summed_axes)
        total = summed if total is None else total + summed
    return total


def _y_block_metric_y(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],       # the jet shape
        y:       typ.Any,                   # Y*, the data, of the type of z (_y_times_metric_y)
        weight:  typ.Any,                   # omega, of the type of z likewise; None = ones
        W:       typ.Tuple[int, ...],       # the sample stack, read off the sample
        C:       typ.Tuple[int, ...],       # the stack of base points
        shape:   typ.Tuple[int, ...],       # the mode sizes N_i
) -> NDArray:                               # <y, M y> at gamma = 1 per block: the block shape (d,)+F | F | ()
    '''``<y, M y>`` per block, with ``M`` the data metric at ``gamma = 1``, over ``z``'s shape: per mode and slot for
    probe, per slot for apply, one number for entries; summed over the samples, the stack ``C`` and the mode edge. The
    data's half of ``r_block_sumsq``: the same blocks, the same metric, the same entries counted.'''
    m = len(order)
    xnp, _, _ = get_backend(False, tree_contains_jax((y, weight)))
    sums = [xnp.sum(product, axis=tuple(range(m, product.ndim)))
            for product in _y_times_metric_y(kind, order, y, weight, W, C, shape)]
    return xnp.stack(sums) if kind == 'probe' else sums[0]


def _take_W(
        array:    NDArray,                  # one array of a data-space object, its W axes at w_start
        w_start:  int,                      # m (an apply or entries array, a ragged probe mode); 1 + m (mode-leading)
        W:        typ.Tuple[int, ...],      # the batch's sample stack
        indices:  NDArray,                  # int, (b,): flat positions over W
        what:     str,                      # 'the data y' or 'the weight', for the message
) -> NDArray:                               # W replaced by one axis: (1,) if every W axis was 1, else (b,)
    '''One array of ``y`` or the weight, drawn: its ``W`` axes (each 1 or ``W``'s size) kept as one axis of 1 when
    every one is 1, else broadcast to ``W``, flattened and gathered at ``indices``.'''
    n_w = len(W)
    shape = tuple(np.shape(array))
    if len(shape) < w_start + n_w or any(a not in (1, b) for a, b in zip(shape[w_start:w_start + n_w], W)):
        raise ValueError('LeastSquaresOracle.take: %s has shape %s, whose axes %d to %d are not the sample stack W = '
                         '%s (each axis 1 or its size): it is not of the type of z'
                         % (what, shape, w_start, w_start + n_w - 1, W))
    xnp, _, _ = get_backend(False, is_jax_ndarray(array))
    if all(a == 1 for a in shape[w_start:w_start + n_w]):
        return xnp.reshape(array, shape[:w_start] + (1,) + shape[w_start + n_w:])
    full = xnp.broadcast_to(array, shape[:w_start] + W + shape[w_start + n_w:])
    return _flat_gather(full, w_start, n_w, indices)


def _take_data_space(
        kind:     str,                      # 'probe' | 'apply' | 'entries'
        m:        int,                      # the number of jet axes
        W:        typ.Tuple[int, ...],      # the batch's sample stack
        value:    typ.Any,                  # y or the weight, of the type of z: probe a tuple or a list of d arrays, or
                                            #   one mode-leading array (packed or not); else one array; None
        indices:  NDArray,                  # int, (b,): flat positions over W
        what:     str,                      # 'the data y' or 'the weight', for the message
) -> typ.Any:                               # value drawn, its form kept; None for None
    '''``y`` or the weight of a drawn batch (:py:meth:`LeastSquaresOracle.take`): its ``W`` axes start at ``m`` in an
    apply or entries array and in each array of a ragged probe object, and at ``1 + m`` in a mode-leading probe array.'''
    if value is None:
        return None
    if kind == 'probe' and isinstance(value, (tuple, list)):
        return [_take_W(array, m, W, indices, what) for array in value]
    return _take_W(value, (1 + m) if kind == 'probe' else m, W, indices, what)


@dataclass(frozen=True, eq=False)
class LeastSquaresOracle(ValueHashedFields):
    """The least-squares oracle: ``f = 1/2 <M r, r>`` with ``r = z(x) - y`` and ``M`` the batch's data metric
    (:py:meth:`LeastSquaresBatch.data_metric`, ``minibatch_scale * weight**2`` elementwise), and what an optimizer asks
    of it at a frame, through one call of the engine.

    It is a frozen description with no arrays (its fields are statics, value-hashed: it rides as jax ``aux_data``);
    the data are its batch, a :py:class:`LeastSquaresBatch`. ``order`` is the jet shape of the sample and the data, a
    tuple, **required for every kind** (O-44): ``LeastSquaresOracle('apply', order=())`` for a plain fit,
    ``('probe', order=(2,))`` along a line; the entries kind has no jets and takes ``order=()`` only (O-46).

    :py:meth:`query` answers ``want`` at a frame (a manifold frame, or a corewise point ``(B, G)``, told from its
    structure by :py:func:`~t3toolbox.backend.sampling_variables.frame_form`). On a corewise point a direction has one
    form, the raw one: the oracle passes ``core_directions`` when it is given, else the ``variations`` it is given, as
    ``core_directions``; given both, they must be the same direction (the same object, or the same arrays), else a
    ``ValueError``. The **minibatch scale** of the batch (O-21) is part of ``M``: it multiplies ``f``, ``g``,
    ``JtJp``, ``Jp_norm_sq``, ``Kp``, ``f_resolution`` and the two block sums, and nothing else: ``r`` and ``Jp`` are
    the plain vectors at the drawn samples. ``f_resolution`` is ``eps * (sqrt(<y, M y>) * sqrt(2 f) + |f|)``, ``eps``
    of ``f``'s dtype: a bound on the rounding of ``f``, per element of ``C`` like ``f``, with ``<y, M y>`` summed over
    every entry of ``z`` (``y`` broadcast to ``z``'s shape, so a ``y`` shared by the samples counts once per sample, as
    ``f`` does). It needs the batch's ``y`` even when ``f`` comes from a held ``r`` (a ``ValueError`` otherwise).

    **The block sums** ``r_block_sumsq`` and ``y_block_sumsq`` are ``<r, M r>`` and ``<y, M y>`` per block, the block
    shape ``(d,) + F`` (probe: a mode at a slot), ``F`` (apply: a slot) or ``()`` (entries), each summed over the
    samples, the stack ``C`` and the mode edge, over ``z``'s shape (``y`` broadcast to it, as for ``f_resolution``).
    Both carry the scale, so their ratio, ``‖omega⊙r_b‖² / ‖omega⊙y_b‖²``, does not depend on it; with no weight they
    are sums of squares. On a drawn batch they estimate the full batch's without bias, as ``f`` does; a ``y`` and a
    weight that are the same for every sample (``W`` axes of 1, kept as one axis of 1 by :py:meth:`take`) give the full
    batch's ``y_block_sumsq`` exactly. Each batch's sums use that batch's own weight: a validation batch with another
    weight (or none) gives the ratio under that weight.

    **The outputs, typed** (the glossary's Symbols): ``r`` and ``Jp`` are in ``Y*``, of the type of ``z``; ``g``,
    ``JtJp`` and ``Kp`` in ``X*``, the blocks before the geometry's projection; ``f``, ``Jp_norm_sq`` and
    ``f_resolution`` are numbers, ``ℝ`` per element of ``C`` (``K + C`` for ``Jp_norm_sq``); the two block sums
    ``ℝ`` per block.

    **What it holds.** ``query`` returns, with its values, a
    :py:class:`~t3toolbox.backend.sampling_variables.HeldVariables`: the ``held`` it was given plus each of ``xi``,
    ``mu``, ``nu``, ``eta``, ``r`` that the call computed anyway (never one computed only to be held). Hand it back
    to a later query **at the same frame and the same batch**: a held ``r`` belongs to its batch's ``y``, and the
    sweeps to its sample. The record carries no identity of either.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> from t3toolbox.backend.jets import affine_jet
    >>> np.random.seed(0)
    >>> x, target = (t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1)) for _ in range(2))
    >>> frame = t3m.MANIFOLD.frame(x).data
    >>> ww, pp = ([np.random.randn(7, N) for N in x.shape] for _ in range(2))          # W = (7,)
    >>> oracle = LeastSquaresOracle('probe', order=(1,))
    >>> batch = LeastSquaresBatch(affine_jet(ww, pp), target.probe(affine_jet(ww, pp), (1,)))
    >>> (f, g), held = oracle.query(frame, batch, ('f', 'g'))
    >>> print(f.shape, [b.shape for b in g[0]], held.keys())
    () [(2, 4), (3, 5), (2, 6)] ('xi', 'mu', 'nu', 'eta', 'r')

    The trap: input jets read with ``order=()`` pass every check of shapes (the jet axis is read as one more sample
    axis), and fit a different problem -- the probes at ``ww`` and at ``pp`` separately, not along the line:

    >>> plain = LeastSquaresOracle('probe', order=())
    >>> (f_wrong,), _ = plain.query(frame, LeastSquaresBatch(affine_jet(ww, pp), x.probe(affine_jet(ww, pp))), ('f',))
    >>> (f_right,), _ = oracle.query(frame, LeastSquaresBatch(affine_jet(ww, pp), x.probe(affine_jet(ww, pp), (1,))),
    ...                              ('f',))
    >>> print(float(f_wrong) < 1e-20, float(f_right) < 1e-20)                 # each fits its own data exactly
    True True
    >>> (f_mixed,), _ = oracle.query(frame, LeastSquaresBatch(affine_jet(ww, pp), x.probe(affine_jet(ww, pp))), ('f',))
    >>> print(float(f_mixed) > 1.0)                    # the slots of the line are not the probes at ww and pp
    True
    """
    kind:   str                                         # 'probe' | 'apply' | 'entries'
    order:  typ.Tuple[int, ...]                         # the jet shape; required (O-44); entries: () only (O-46)

    optional_outputs: typ.ClassVar[typ.Tuple[str, ...]] = (
        'Kp', 'Jp', 'r', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution')

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError('LeastSquaresOracle: unknown kind %r; expected one of %s' % (self.kind, _KINDS))
        if not isinstance(self.order, tuple) or not all(isinstance(n, (int, np.integer)) and not isinstance(n, bool)
                                                         and n >= 0 for n in self.order):
            raise TypeError('LeastSquaresOracle: order is the jet shape, a tuple of nonnegative ints (() for no jets); '
                            'got %r' % (self.order,))
        if self.kind == 'entries' and self.order != ():
            raise ValueError("LeastSquaresOracle: the entries kind has no jets: its order is (), got %r "
                             "(a derivative of entries is the apply kind at the one-hot jet)" % (self.order,))

    # ------------------------------------------------------------------ the operation
    def query(
            self,
            frame:            typ.Tuple,                      # a manifold frame or a corewise point (frame_form)
            batch:            LeastSquaresBatch,              # its data
            want:             typ.Sequence[str],              # f, g, JtJp, Jp_norm_sq; optional: optional_outputs
            Vc:               typ.Optional[typ.Sequence[NDArray]] = None,
            variations:       typ.Optional[typ.Tuple] = None, # X, (V, H): JtJp, Jp_norm_sq, Jp; raw on a point
            core_directions:  typ.Optional[typ.Tuple] = None, # X, (D_dot, L_dot): Kp
            held:             typ.Any = None,                 # a HeldVariables from an earlier query here, or None
    ) -> typ.Tuple[
            typ.Tuple[typ.Any, ...],                          # values, aligned with want: r, Jp Y*; g, JtJp, Kp X*
                                                              #   (before the projection); f, Jp_norm_sq, f_resolution
                                                              #   ℝ; r_block_sumsq, y_block_sumsq ℝ per block
            typ.Any,                                          # held: a HeldVariables
    ]:
        '''The values ``want`` at ``frame`` on ``batch``, and what to hold for a later query here (the class doc).'''
        want = tuple(want)
        _, call = sampling_variables.frame_form(frame)
        if call == 'corewise' and variations is not None:
            if core_directions is None:
                core_directions = variations
            elif not _same_leaves(variations, core_directions):
                raise ValueError('LeastSquaresOracle.query: a corewise point has one form of a direction, and both '
                                 'variations and core_directions were given, not the same direction (pass one, or '
                                 'the same arrays as both)')
            variations = None
        local = ('y_block_sumsq', 'f_resolution')
        engine_want = tuple(name for name in want if name not in local)
        if 'f_resolution' in want and 'f' not in engine_want:
            engine_want = engine_want + ('f',)
        values = {}
        held_out = held if held is not None else sampling_variables.HeldVariables()
        if engine_want:
            # one set of arguments for the query and the call: the hold-list names the call computes anyway are
            # added to want and split off (none is computed only to be held; TARGET_DECISIONS, the hold list)
            args = dict(order=self.order, y=batch.y, weight=batch.weight, variations=variations,
                        core_directions=core_directions, Vc=Vc, held=held)
            free = sampling_variables.computed_names(self.kind, frame, batch.sample, engine_want, **args)
            extras = tuple(name for name in _HOLD_LIST if name in free and name not in engine_want)
            outs = sampling_variables.sampling_variables(self.kind, frame, batch.sample, engine_want + extras, **args)
            values = dict(zip(engine_want + extras, outs))
            kept = {name: values[name] for name in _HOLD_LIST if name in free}
            held_out = held_out.with_values(**kept) if kept else held_out
        if 'y_block_sumsq' in want:                   # the data's half of the block sums: <y, M y> per block
            if batch.y is None:
                raise ValueError('LeastSquaresOracle.query: y_block_sumsq is <y, M y> per block and needs the '
                                 'batch\'s data y, which is None')
            values['y_block_sumsq'] = _y_block_metric_y(self.kind, self.order, batch.y, batch.weight, self._W(batch),
                                                        _stack_of_frame(frame), sampling_variables.frame_shape(frame))
        if 'f_resolution' in want:                    # from the unscaled batch values; the scale once, below
            if batch.y is None:                         # f can come from a held r; <y, M y> cannot
                raise ValueError('LeastSquaresOracle.query: f_resolution is eps (sqrt(<y, M y>) sqrt(2 f) + |f|) and '
                                 'needs the batch\'s data y, which is None (f alone can come from a held r; the '
                                 'resolution cannot)')
            f = values['f']
            xnp, _, _ = get_backend(False, tree_contains_jax((f, batch.y, batch.weight)))
            y_norm_sq = _y_metric_y(self.kind, self.order, batch.y, batch.weight, self._W(batch),
                                    _stack_of_frame(frame), sampling_variables.frame_shape(frame))
            eps = float(np.finfo(np.dtype(f.dtype)).eps)
            values['f_resolution'] = eps * (xnp.sqrt(y_norm_sq) * xnp.sqrt(2.0 * f) + xnp.abs(f))
        scale = batch.minibatch_scale
        if scale is not None:
            for name in _SCALED + ('y_block_sumsq', 'f_resolution'):
                if name in values and name in want:
                    values[name] = _scale_tree(values[name], scale)
        return tuple(values[name] for name in want), held_out

    # ------------------------------------------------------------------ the batch
    def check_batch(
            self,
            batch:        LeastSquaresBatch,                  # ragged (check before packing)
            shape:        typ.Sequence[int],                  # the mode sizes (N0, ..., N(d-1))
            stack_shape:  typ.Tuple[int, ...] = (),           # C, the model's stack of base points; () unstacked
    ) -> None:
        '''Check a batch structurally, on its ragged form: the sample against the jet shape and ``shape``
        (:py:func:`~t3toolbox.backend.jets.check_sample`), the data and the weight against the sample and the model's
        stack ``stack_shape`` (``check_data``, ``check_weight``: each of the type of ``z``, every axis 1 or ``z``'s
        size); raise a ``ValueError`` naming what is wrong. Data of a stacked model checked without its
        ``stack_shape`` fail by rank. A checker: :py:meth:`query` does not call it.

        >>> import numpy as np
        >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
        >>> oracle = LeastSquaresOracle('apply', order=())
        >>> ww = [np.ones((10, N)) for N in (4, 5, 6)]                            # W = (10,)
        >>> shared = LeastSquaresBatch(ww, np.zeros((10, 1)))                     # one y for a stack C = (2,)
        >>> oracle.check_batch(shared, (4, 5, 6), stack_shape=(2,))
        >>> oracle.check_batch(LeastSquaresBatch(ww, np.zeros((10, 3))), (4, 5, 6), stack_shape=(2,))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: the data y of the apply kind has shape (10, 3); it is of the type of z, ...
        '''
        jets.check_sample(self.kind, self.order, batch.sample, shape)
        if batch.y is not None:
            jets.check_data(self.kind, self.order, batch.sample, batch.y, stack_shape)
        jets.check_weight(self.kind, self.order, batch.sample, batch.weight, stack_shape)

    def _w_axes(self, batch):
        '''(where W starts in the sample, the number of W axes): after the jets (and the packed d), or after the
        mode axis of an index.'''
        m = len(self.order)
        if self.kind == 'entries':
            return 1, np.ndim(batch.sample) - 1
        if isinstance(batch.sample, (tuple, list)):
            return m, np.ndim(batch.sample[0]) - m - 1
        return 1 + m, np.ndim(batch.sample) - m - 2

    def _W(self, batch) -> typ.Tuple[int, ...]:
        '''The sample stack ``W``, read off the sample.'''
        start, n_w = self._w_axes(batch)
        lead = batch.sample if self.kind == 'entries' or not isinstance(batch.sample, (tuple, list)) \
            else batch.sample[0]
        return tuple(int(n) for n in np.shape(lead)[start:start + n_w])

    def n_measurements(
            self,
            batch:  LeastSquaresBatch,
    ) -> int:                                                 # |W|, the number of samples
        '''The number of samples of a batch, ``prod(W)``: the population a draw takes from.'''
        return math.prod(self._W(batch))

    def take(
            self,
            batch:    LeastSquaresBatch,
            indices:  NDArray,                                # int, (b,): flat positions over W; len(indices) static
    ) -> LeastSquaresBatch:                                   # the drawn batch, W flattened to (b,)
        '''A draw: the samples at ``indices`` (flat over ``W``, which becomes one axis of ``b = len(indices)``), with
        ``minibatch_scale = n / b`` (times the batch's own scale) and ``drawn_indices = indices``, relative to
        ``batch``. The scale makes the drawn ``f`` (and every scaled output) an unbiased estimate of the full one under
        a uniform draw of ``b`` of the ``n`` samples. Traceable, ``b`` static. For a plain subset with no scale, such
        as a fixed validation slice, drop it: ``dataclasses.replace(sub, minibatch_scale=None)``.

        The data ``y`` and the weight are of the type of ``z``, each axis 1 or ``z``'s size, and are drawn alike: an
        object whose every ``W`` axis is 1 (the same for every sample) keeps one axis of 1 in their place, with no
        gather; any other is broadcast to the batch's ``W``, flattened and gathered at ``indices``. So a weight per
        sample is drawn with its samples, and a weight per slot stays one per slot.

        >>> import dataclasses as dc
        >>> import numpy as np
        >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
        >>> oracle = LeastSquaresOracle('apply', order=())
        >>> batch = LeastSquaresBatch([np.ones((10, N)) for N in (4, 5, 6)], np.zeros(10))
        >>> sub = oracle.take(batch, np.array([0, 3]))
        >>> print(sub.y.shape, sub.minibatch_scale, sub.drawn_indices)
        (2,) 5.0 [0 3]
        >>> print(dc.replace(sub, minibatch_scale=None).minibatch_scale)          # a plain subset
        None

        A weight per sample is drawn with the samples; one constant over the samples keeps its axis of 1:

        >>> per_sample = dc.replace(batch, weight=np.arange(10.0))
        >>> print(oracle.take(per_sample, np.array([0, 3])).weight)
        [0. 3.]
        >>> print(oracle.take(dc.replace(batch, weight=np.full((1,), 2.0)), np.array([0, 3])).weight)
        [2.]
        '''
        n = self.n_measurements(batch)
        start, n_w = self._w_axes(batch)
        W = self._W(batch)
        m = len(self.order)
        if self.kind == 'entries':
            sample = _flat_gather(batch.sample, 1, n_w, indices)
        elif isinstance(batch.sample, (tuple, list)):
            sample = [_flat_gather(w, m, n_w, indices) for w in batch.sample]
        else:
            sample = _flat_gather(batch.sample, 1 + m, n_w, indices)
        y = _take_data_space(self.kind, m, W, batch.y, indices, 'the data y')
        weight = _take_data_space(self.kind, m, W, batch.weight, indices, 'the weight')
        scale = n / len(indices)
        if batch.minibatch_scale is not None:
            scale = batch.minibatch_scale * scale
        return LeastSquaresBatch(sample, y, weight, scale, indices)
