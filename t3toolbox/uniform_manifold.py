# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform tangent + manifold layer (the uniform-fix 1.0 centerpiece, increment 3b).

Mirrors the ragged :py:mod:`t3toolbox.manifold` on the uniform (stacked-supercore + boolean-mask)
representation. Holds:

- :py:class:`UT3Tangent` -- the structural bundle ``(UT3Frame, UT3Variations)`` (the ``K``/``C`` stack
  inferred from the pair), the vector-space ops, the raw coordinate inner/norm, the (delegating) validity
  checkers, the constructors, ``reverse``, the doubled-rank ``to_ut3``/``to_dense``, ``gauge_residual`` /
  ``is_gauged``, the cross-layer converters, and the stack/unstack tree conversions;
- :py:class:`UniformManifoldGeometry` / :py:class:`UniformCorewiseGeometry` (3b-5) -- the stateless
  geometry bundles (``frame`` / ``randn`` / ``project`` / ``inner`` / ``norm`` / ``retract`` ...) mirroring
  the ragged ``MANIFOLD`` / ``COREWISE``, behind the per-element safe-mode preconditions, with the module
  singletons :py:data:`UNIFORM_MANIFOLD` / :py:data:`UNIFORM_COREWISE`.

Deferred to later 3b slices: ``probe`` / ``apply`` / ``entries`` (+ derivatives) and the ``WKC``
contractions (3b-6).
"""
from __future__ import annotations

import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.safety as safety
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ufv_operations as ufv_operations
import t3toolbox.backend.ufv_tangent_operations as ufv_tangent_operations
import t3toolbox.backend.ufv_sampling as ufv_sampling
import t3toolbox.backend.probe_derivatives as probe_derivatives
from t3toolbox.backend.common import *

__all__ = [
    'UT3Tangent',
    'UniformManifoldGeometry',
    'UniformCorewiseGeometry',
    'UNIFORM_MANIFOLD',
    'UNIFORM_COREWISE',
]


def _broadcast_variation_masks_over_K(
        masks:  ubv.UT3VariationsMasks,  # gauge-shifted variation masks, each (d,)  +C+(size,)
        K:      typ.Tuple[int, ...],     # tangent stack to prepend (outer)
) -> ubv.UT3VariationsMasks:             # masks broadcast constant along K, each (d,)+K+C+(size,)
    """Broadcast variation rank masks over an outer tangent stack ``K`` (constant along it).

    A ``K``-stacked zero tangent is a bundle of ``K`` tangent vectors at one frame, so each tangent
    carries the *same* gauge masks -- the frame's masks replicated (constant) along ``K``. Masks are host
    numpy (static aux), so this stays on ``np`` (see CLAUDE.md: supercores -> ``xnp``, masks -> ``np``).
    """
    K = tuple(K)

    def b(m):  # insert |K| size-1 axes after the leading mode (d) axis, then broadcast to (d,)+K+C+(size,)
        return np.broadcast_to(m.reshape(m.shape[:1] + (1,) * len(K) + m.shape[1:]),
                               m.shape[:1] + K + m.shape[1:])

    return ubv.UT3VariationsMasks(b(masks.variations_up_mask), b(masks.variations_down_mask),
                                  b(masks.variations_left_mask), b(masks.variations_right_mask))


def _ut3frame_from_data(bd) -> ubv.UT3Frame:        # bd = (up, down, left, right, shape, masks_tuple)
    return ubv.UT3Frame(bd[0], bd[1], bd[2], bd[3], bd[4], ubv.UT3FrameMasks(*bd[5]))


def _ut3variations_from_data(vd) -> ubv.UT3Variations:  # vd = (tkv, ttv, shape, masks_tuple)
    return ubv.UT3Variations(vd[0], vd[1], vd[2], ubv.UT3VariationsMasks(*vd[3]))


def _flatten_tangents(tree) -> typ.List['UT3Tangent']:
    """Flatten an array-like tree of UT3Tangents (nested tuples) into a flat list of leaves."""
    if isinstance(tree, UT3Tangent):
        return [tree]
    out = []
    for sub in tree:
        out.extend(_flatten_tangents(sub))
    return out


@dataclass(frozen=True)
class UT3Tangent:
    """Tangent vector to the uniform manifold of fixed-rank Tucker tensor trains (uniform analog of
    :py:class:`~t3toolbox.manifold.T3Tangent`).

    A ``UT3Tangent`` bundles a :py:class:`~t3toolbox.uniform_frame_variations_format.UT3Frame` (the frame
    at the base point where the tangent space is attached) with a
    :py:class:`~t3toolbox.uniform_frame_variations_format.UT3Variations` (the tangent direction in that
    frame). The ``K``/``C`` stack split (extra tangent stack ``K`` over the shared frame stack ``C``) is
    **inferred** from the pair -- the variation stack is ``K + C`` and the frame stack is ``C`` -- never
    stored (the split-agnostic stacking of increment 2c).

    Like the ragged class, the metric lives on the *geometry*, not here; this exposes only the **raw
    coordinate** :py:meth:`corewise_inner` / :py:meth:`corewise_norm` (no Hilbert-Schmidt claim), computed
    on the **real (masked)** content of the variations.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_frame_variations_format as ubv
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
    >>> frame, variations = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
    >>> v = ut3m.UT3Tangent(frame, variations)
    >>> print(v.shape, v.stack_shape)
    (10, 11, 12) ()
    >>> print(bool(v.is_orthogonal()))   # frame from ut3_orthogonal_representations is orthogonal
    True
    >>> w = 2.0 * v - v                  # linear algebra stays in the same tangent space
    >>> print(bool(w.allclose(v)))       # (2v - v) == v
    True
    """
    frame:      ubv.UT3Frame
    variations: ubv.UT3Variations

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate this tangent: both components well-formed and a compatible (frame, variations) pair.

        Runs ``frame.validate()`` + ``variations.validate()`` + the bv-pair compatibility check
        (:py:func:`~t3toolbox.uniform_frame_variations_format.check_ufv_pair`, which permits an extra
        tangent stack ``K``). Structural only (shapes / ranks / masks; masks are host numpy), so it is
        safe in ``__post_init__`` and under jit/pytree tracing.
        """
        self.frame.validate()
        self.variations.validate()
        ubv.check_ufv_pair(self.frame, self.variations)

    def __repr__(self) -> str:
        return (f"UT3Tangent(shape={self.shape}, tucker_ranks={self.frame.up_ranks}, "
                f"tt_ranks={self.frame.left_ranks}, tangent_stack={self.tangent_stack_shape}, "
                f"frame_stack={self.frame_stack_shape})")

    # ------------------------------------------------------------- structure (K/C inferred from the pair)
    @ft.cached_property
    def d(self) -> int:
        return self.frame.d

    @ft.cached_property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.frame.shape

    @ft.cached_property
    def frame_stack_shape(self) -> typ.Tuple[int, ...]:
        """Frame stack ``C``: the batch of base points, shared with the frame (``frame.stack_shape``)."""
        return self.frame.stack_shape

    @ft.cached_property
    def tangent_stack_shape(self) -> typ.Tuple[int, ...]:
        """Tangent stack ``K``: the extra *outer* batch of tangent vectors sharing this frame.

        The part of the variation stack that exceeds the frame stack (often empty). The variation
        supercores are stacked ``(d,) + K + C + (core,)`` -- extra axes outermost, frame stack inner.
        """
        full = self.variations.stack_shape
        return full[:len(full) - len(self.frame_stack_shape)]

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int, ...]:
        """Full stack ``K + C`` (``tangent_stack_shape + frame_stack_shape``), outer-to-inner."""
        return self.variations.stack_shape

    @ft.cached_property
    def structure(self):
        """The frame's per-element structure ``(shape, up_ranks, down_ranks, left_ranks, right_ranks,
        stack_shape)`` (ranks are arrays over the frame stack ``C``)."""
        return self.frame.structure

    @ft.cached_property
    def data(self) -> typ.Tuple[ubv.UT3Frame, ubv.UT3Variations]:
        return self.frame, self.variations

    def to_jax(self) -> 'UT3Tangent':
        """Copy with frame and variation supercores converted to jax arrays (masks stay host numpy)."""
        return UT3Tangent(self.frame.to_jax(), self.variations.to_jax())

    def to_numpy(self) -> 'UT3Tangent':
        """Copy with frame and variation supercores converted to numpy arrays."""
        return UT3Tangent(self.frame.to_numpy(), self.variations.to_numpy())

    def copy(self) -> 'UT3Tangent':
        """Deep copy (copies the frame and variation supercores)."""
        return UT3Tangent(self.frame.copy(), self.variations.copy())

    @ft.cached_property
    def contains_jax(self) -> bool:
        """True if any frame or variation supercore is a jax array."""
        return self.frame.contains_jax or self.variations.contains_jax

    # ------------------------------------------------------------- linear algebra (fixed-rank vector space)
    def _check_same_tangent_space(self, other: 'UT3Tangent') -> None:
        # STRUCTURAL (always): matching stack (K+C) and rank masks (the uniform stratum / variety point).
        # Masks are host numpy, so these are cheap and valid under jit.
        if self.stack_shape != other.stack_shape:
            raise ValueError(
                'Tangent vectors have different stack shapes; elementwise linear algebra requires '
                'matching stacks (same tangent stack K over the shared frame stack C).\n'
                + str(self.stack_shape) + ' = self.stack_shape != other.stack_shape = ' + str(other.stack_shape))
        if self.variations.masks != other.variations.masks:
            raise ValueError(
                'Tangent vectors have different rank masks (different strata of the determinantal '
                'variety, i.e. different tangent spaces).')
        # NUMERICAL (skippable): are these the same frame? The `is` fast-path keeps the common eager case
        # O(1); the value compare (frame supercores only -- data[:4]; the full .data carries the int-tuple
        # `shape`, which safety's array compare cannot take) runs only when the objects differ, e.g. a jit
        # round-trip reconstructs a value-equal frame. Skips under safety.unsafe() and under a jax trace.
        if not (self.frame is other.frame
                or safety.frames_equal_or_skip(self.frame.data[:4], other.frame.data[:4])):
            raise ValueError(
                'Tangent vectors are in different tangent spaces (their frames are not the same frame).\n'
                'Linear algebra between tangent vectors requires the same frame; run inside '
                'safety.unsafe() to skip this numerical check.')

    def __add__(self, other: 'UT3Tangent') -> 'UT3Tangent':
        """Add tangent vectors. Requires the same frame + matching stack/masks (same tangent space)."""
        self._check_same_tangent_space(other)
        return UT3Tangent(self.frame, self.variations + other.variations)

    def __sub__(self, other: 'UT3Tangent') -> 'UT3Tangent':
        """Subtract tangent vectors. Requires the same frame + matching stack/masks."""
        self._check_same_tangent_space(other)
        return UT3Tangent(self.frame, self.variations - other.variations)

    def __mul__(self, scalar) -> 'UT3Tangent':
        """Scale a tangent vector by a scalar (the base point is unchanged)."""
        return UT3Tangent(self.frame, self.variations * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> 'UT3Tangent':
        return UT3Tangent(self.frame, -self.variations)

    def corewise_inner(self, other: 'UT3Tangent') -> NDArray:  # shape = stack_shape (K+C); scalar unstacked
        """The raw corewise (coordinate) dot of two tangents' variations -- **not** the HS inner product.

        Computed on the **real (masked)** content, vectorized over the stack: returns an array of shape
        :py:attr:`stack_shape` (``K + C``), one dot per stacked tangent (a scalar when unstacked). The
        same-tangent-space precondition is checked. It equals Hilbert-Schmidt only on an orthonormal,
        gauged frame; for that semantic use the (forthcoming) manifold geometry's ``inner``.
        """
        self._check_same_tangent_space(other)
        return ufv_tangent_operations.ufv_corewise_inner(
            self.variations.data, other.variations.data, len(self.stack_shape))

    def corewise_norm(self) -> NDArray:  # shape = stack_shape (K+C); scalar unstacked
        """The raw corewise (coordinate) norm of the variations (masked content) -- **not** the HS norm.

        Vectorized over the stack: an array of shape :py:attr:`stack_shape` (``K + C``), one norm per
        stacked tangent (scalar when unstacked)."""
        xnp, _, _ = get_backend(True, self.contains_jax)
        return xnp.sqrt(self.corewise_inner(self))

    def normalized(self) -> 'UT3Tangent':
        """Unit-norm rescaling ``self / self.corewise_norm()``, vectorized over the stack (each stacked
        tangent scaled by its own norm; the base point unchanged)."""
        inv = 1.0 / self.corewise_norm()                     # shape = stack_shape (K+C)
        n_stack = len(self.stack_shape)

        def scale(sc):  # align inv (shape = stack) to the supercore stack axes 1 .. 1+n_stack
            s = inv.reshape((1,) + self.stack_shape + (1,) * (sc.ndim - 1 - n_stack))
            return sc * s

        tkv, ttv = self.variations.supercores
        scaled = ubv.UT3Variations(scale(tkv), scale(ttv), self.variations.shape, self.variations.masks)
        return UT3Tangent(self.frame, scaled)

    def allclose(
            self,
            other:  'UT3Tangent',  # compared at the SAME base point (corewise, like __sub__)
            rtol:   float = 1e-9,
            atol:   float = 0.0,
    ) -> NDArray:  # bool array, shape = stack_shape (K+C); scalar when unstacked
        """``True`` (per stack element) if ``other`` is the same tangent vector as ``self`` at the same frame.

        Checks ``||self - other|| <= atol + rtol * ||other||`` via :py:meth:`corewise_norm`, **per stacked
        element** (reduce with ``.all()`` for a single verdict). Assumes a shared frame (compares corewise on
        the variations, like :py:meth:`__sub__`); for tangents at different bases, compare dense."""
        dn = (self - other).corewise_norm()
        rn = other.corewise_norm()
        return dn <= atol + rtol * rn

    # ------------------------------------------------------------- validity checkers (delegate to UT3Frame)
    @ft.cached_property
    def minimal_ranks(self):
        """Structural minimal ranks of this tangent's base point, **per frame-stack element** (the ranks
        vary across the stack). See :py:meth:`UT3Frame.minimal_ranks`."""
        return self.frame.minimal_ranks

    @ft.cached_property
    def tangent_space_dimension(self):  # int (unstacked) or int array of shape = frame stack C
        """Dimension of the tangent space at this base point (= the fixed-rank manifold dimension),
        **per frame-stack element** ``C`` (the ``K`` vectors share the frame, hence the same tangent space).

        An ``int`` when unstacked (``C == ()``), else an int array of shape ``C``. Computed from the
        structurally-minimal ranks (gauge already quotiented); see
        :py:func:`~t3toolbox.backend.ranks.compute_manifold_dim`. Masks are host numpy, so the ranks --
        and this dimension -- are host quantities (a small loop over the frame stack, computed once)."""
        up = np.asarray(self.frame.up_ranks)        # (d,)   + C
        left = np.asarray(self.frame.left_ranks)    # (d+1,) + C
        C = self.frame_stack_shape

        def dim_at(idx):  # idx: an index tuple into C
            sel = (slice(None),) + idx
            tucker_ranks = tuple(int(n) for n in up[sel])
            tt_ranks = tuple(int(r) for r in left[sel])
            return ranks.compute_manifold_dim(self.shape, tucker_ranks, tt_ranks)

        if C == ():
            return dim_at(())
        out = np.empty(C, dtype=int)
        for idx in np.ndindex(*C):
            out[idx] = dim_at(idx)
        return out

    @ft.cached_property
    def has_minimal_ranks(self) -> NDArray:  # bool array, shape = frame stack C (scalar unstacked)
        """True (per frame-stack element) if this tangent's frame has **structurally** minimal ranks. See
        :py:meth:`UT3Frame.has_minimal_ranks`. Minimal rank is *not* a correctness precondition for the
        tangent ops (see the contract catalog); for the numerical check see
        :py:meth:`has_numerically_minimal_ranks`. Not enforced at construction."""
        return self.frame.has_minimal_ranks

    def has_numerically_minimal_ranks(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = frame stack C
        """True (per frame-stack element) if this tangent's frame is **numerically** minimal. See
        :py:meth:`UT3Frame.has_numerically_minimal_ranks` (orthogonal + structurally-minimal, no SVD)."""
        return self.frame.has_numerically_minimal_ranks(atol=atol)

    def is_orthogonal(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = frame stack C (scalar unstacked)
        """True (per frame-stack element) if this tangent's frame is orthogonal. See
        :py:meth:`UT3Frame.is_orthogonal`. Reduce with ``.all()`` for a single verdict."""
        return self.frame.is_orthogonal(atol=atol)

    @ft.cached_property
    def gauge_residual(self) -> NDArray:  # shape = variation stack K+C (scalar/0-d when unstacked); cached
        """Max absolute gauge-condition violation, **per stack element** (shape = variation stack ``K+C``;
        atol-independent; **cached**). The expensive part of :py:meth:`is_gauged` -- a fixed tangent reused
        across an inner loop (e.g. the safe-mode GAUGE precondition of the manifold inner product) is
        contracted once. See :py:func:`~t3toolbox.backend.ufv_tangent_operations.gauge_residual`."""
        return ufv_tangent_operations.gauge_residual(self.frame.data, self.variations.data)

    def is_gauged(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = variation stack K+C (scalar unstacked)
        """True (per stack element) if the variations are gauged w.r.t. the frame.

        Gauge conditions (needed for the manifold Hilbert-Schmidt inner product / norm to equal the
        coordinate ones; not enforced at construction): ``U_i^T dU_i = 0`` (all i) and
        ``(P_i^L)^T dG_i^L = 0`` (i = 0..d-2) -- conditions (48)-(49), Appendix A.3 of Alger et al. (2026).
        **Per-stack-element bool array** (reduce with ``.all()`` for a single verdict)."""
        return self.gauge_residual <= atol

    # ------------------------------------------------------------- conversions
    def to_ut3(
            self,
            include_shift:  bool = False,  # False: tangent vector v. True: base point + v.
    ) -> ut3.UniformTuckerTensorTrain:  # doubled-rank uniform Tucker tensor train
        """Doubled-rank :py:class:`UniformTuckerTensorTrain` representation of this tangent vector.

        The Tucker and TT ranks are (roughly) doubled (masks double via concatenation). With
        ``include_shift=True`` the result represents ``base point + v`` (used by retraction). The uniform
        mirror of :py:meth:`T3Tangent.to_t3`; see
        :py:func:`~t3toolbox.backend.ufv_tangent_operations.tangent_to_ut3` (Appendix A.3.1)."""
        tk, tt, shape, (tucker_mask, tt_mask) = ufv_tangent_operations.tangent_to_ut3(
            self.frame.data, self.variations.data, include_shift=include_shift)
        return ut3.UniformTuckerTensorTrain(tk, tt, shape, ut3.UT3Masks(tucker_mask, tt_mask))

    def to_dense(
            self,
            include_shift:  bool = False,  # False: tangent vector v. True: base point + v.
    ) -> NDArray:  # dense tangent vector, shape = stack_shape (K+C) + (N0,...,N(d-1))
        """Form the dense tensor represented by this tangent vector (``= to_ut3(...).to_dense()``).

        The tangent vector is the sum of the 2d single-core-replacement terms; with ``include_shift=True``
        the base point is added. Stack-aware (the result is stacked ``K + C``). Inspection/tests only."""
        return self.to_ut3(include_shift).to_dense()

    def to_t3tangent(self):  # -> T3Tangent (unstacked) or a nested tree (shaped K+C) of unstacked T3Tangents
        """Convert to a ragged :py:class:`~t3toolbox.manifold.T3Tangent` (the cross-layer converter; uniform
        analog of pairing :py:meth:`UT3Frame.to_t3frame` with :py:meth:`UT3Variations.to_t3variations`).

        Unstacked: one :py:class:`T3Tangent`. Stacked: a nested tree (shaped ``K + C``) of *unstacked*
        :py:class:`T3Tangent` s -- a single ragged stacked tangent cannot carry varying-``C`` ranks, so the
        truthful answer is one ragged tangent per element."""
        if not self.stack_shape:
            return t3m.T3Tangent(self.frame.to_t3frame(), self.variations.to_t3variations())
        sub = self.unstack_tangents() if self.tangent_stack_shape else self.unstack_frame()
        return stacking.apply_func_to_leaf_subtrees(sub, lambda leaf: leaf.to_t3tangent(), None)

    @staticmethod
    def from_t3tangent(
            tangent:  't3m.T3Tangent',
            N:   typ.Optional[int] = None,   # padded mode dim   (default max(shape))
            nU:  typ.Optional[int] = None,   # padded up rank    (default max(up_ranks))
            nD:  typ.Optional[int] = None,   # padded down rank  (default max(down_ranks))
            rL:  typ.Optional[int] = None,   # padded left rank  (default max(left_ranks))
            rR:  typ.Optional[int] = None,   # padded right rank (default max(right_ranks))
    ) -> 'UT3Tangent':
        """Pack a ragged :py:class:`~t3toolbox.manifold.T3Tangent` into a uniform one (inverse of
        :py:meth:`to_t3tangent` on a single tangent). The frame and variations are padded to the **same**
        dims (derived from the frame if not given) so the masks come out consistent."""
        b = tangent.frame
        N  = max(b.shape)            if N  is None else N
        nU = int(max(b.up_ranks))    if nU is None else nU
        nD = int(max(b.down_ranks))  if nD is None else nD
        rL = int(max(b.left_ranks))  if rL is None else rL
        rR = int(max(b.right_ranks)) if rR is None else rR
        pad = dict(N=N, nU=nU, nD=nD, rL=rL, rR=rR)
        return UT3Tangent(ubv.UT3Frame.from_t3frame(b, **pad),
                          ubv.UT3Variations.from_t3variations(tangent.variations, **pad))

    def reverse(self) -> 'UT3Tangent':
        """Reverse the mode order of this tangent (reverses both the frame and the variations).

        Will commute with :py:meth:`to_dense` (the dense tangent's mode axes are reversed) once the
        doubled-rank conversion lands -- lets you reverse a T3 and its derived tangent without
        recomputing the orthogonal representation."""
        return UT3Tangent(self.frame.reverse(), self.variations.reverse())

    # ------------------------------------------------------------- sampling (the bare Jacobian 𝒥; 3b-6b)
    def probe(
            self,
            ww:  typ.Sequence[NDArray],  # probe vectors, len=d, ith elm_shape=W+(Ni,)
    ) -> typ.Tuple[NDArray, ...]:        # d probes, ith elm_shape=W+K+C+(Ni,)
        """Probe this tangent vector: the single-sample Riemannian Jacobian ``𝒥`` (contract all-but-one
        mode, for each mode). Uniform mirror of :py:meth:`~t3toolbox.manifold.T3Tangent.probe`; ``ww`` is
        packed at the boundary and the ``d`` probes come back as ragged-width vectors, stacked ``W+K+C``.
        The bare ``𝒥`` (no gauge projector ``Π``).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> import t3toolbox.uniform_manifold as ut3m
        >>> import t3toolbox.backend.probing as t3p
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(x))
        >>> ww = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))   # probe stack W=(2,)
        >>> zz = v.probe(ww)
        >>> print(zz[0].shape)                                     # W + K + C + (N0,) = (2,) + () + () + (10,)
        (2, 10)
        >>> print(bool(max(float(np.linalg.norm(a - b))
        ...                for a, b in zip(zz, t3p.probe_dense(ww, v.to_dense()))) < 1e-9))   # dense reference
        True
        """
        return ufv_sampling.ut3tangent_probe(ww, self.frame.data, self.variations.data)

    def apply(
            self,
            ww:  typ.Sequence[NDArray],  # apply vectors, len=d, ith elm_shape=W+(Ni,)
    ) -> NDArray:                        # scalar apply, shape=W+K+C
        """Apply this tangent vector in all modes (the all-modes special case of :py:meth:`probe`; a scalar
        per stack element, stacked ``W+K+C``). The bare ``𝒥``. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.apply`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> import t3toolbox.uniform_manifold as ut3m
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(x))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> print(bool(abs(float(v.apply(ww)) - float(np.einsum('ijk,i,j,k->', v.to_dense(), *ww))) < 1e-9))
        True
        """
        return ufv_sampling.ut3tangent_apply(ww, self.frame.data, self.variations.data)

    def entries(
            self,
            index:  NDArray,  # int, shape=(d,)+W (a stack W of multi-indices)
    ) -> NDArray:             # scalar entries, shape=W+K+C
        """Entries of the dense tangent at ``index`` (= :py:meth:`apply` with unit vectors, by fiber
        slicing). The bare ``𝒥``. Uniform mirror of :py:meth:`~t3toolbox.manifold.T3Tangent.entries`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> import t3toolbox.uniform_manifold as ut3m
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(x))
        >>> print(bool(abs(float(v.entries((3, 5, 7))) - float(v.to_dense()[3, 5, 7])) < 1e-9))
        True
        """
        return ufv_sampling.ut3tangent_entries(index, self.frame.data, self.variations.data)

    # --------------------------------------------------------------- derivative sampling (jets 𝒥; 3b-6'b)
    def probe_derivatives(
            self,
            ww:     typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                    # highest derivative order
    ) -> typ.Tuple[NDArray, ...]:           # len=d, ith elm_shape=(order+1,)+W+K+C+(Ni,)
        """Symmetric directional derivatives of :py:meth:`probe`, in one repeated direction ``P`` (``pp``):
        the forward Riemannian Jacobian derivatives ``y_i^(t) = d^t/ds^t [probe(X + s P)]_i|_0`` for
        ``t=0..order`` (order 0 is :py:meth:`probe`; the bare ``𝒥``, no gauge ``Π``). ``ww``/``pp`` share the
        sample stack ``W``; stacks ``order + W + K + C``. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.probe_derivatives`.

        No *numerical* precondition (gauge-invariant, any frame). **Structural precondition** (hard error):
        ``P`` (``pp``) shares the sample stack ``W`` and mode dims of ``X`` (``ww``).

        See Also
        --------
        probe
        apply_derivatives
        probe_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> import t3toolbox.uniform_manifold as ut3m
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(x))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> zj = v.probe_derivatives(ww, pp, 3)
        >>> print([z.shape for z in zj])                                    # (order+1,) + (Ni,)
        [(4, 10), (4, 11), (4, 12)]
        >>> print([bool(np.allclose(z[0], z0)) for z, z0 in zip(zj, v.probe(ww))])   # order 0 == probe
        [True, True, True]

        ``P`` must match ``X``'s sample stack and mode dims (structural, raises):

        >>> v.probe_derivatives(ww, (np.random.randn(10), np.random.randn(11), np.random.randn(99)), 3)
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError
        """
        probe_derivatives.check_perturbation_vectors(ww, pp)
        return ufv_sampling.ut3tangent_probe_derivatives(ww, pp, self.frame.data, self.variations.data, order)

    def apply_derivatives(
            self,
            ww:     typ.Sequence[NDArray],  # apply vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                    # highest derivative order
    ) -> NDArray:                           # scalar jets, shape=(order+1,)+W+K+C
        """Symmetric all-modes apply derivatives (derivative twin of :py:meth:`apply`; the bare ``𝒥``, a
        scalar jet per stack element). Structural precondition as in :py:meth:`probe_derivatives`. Uniform
        mirror of :py:meth:`~t3toolbox.manifold.T3Tangent.apply_derivatives`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> import t3toolbox.uniform_manifold as ut3m
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(x))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> yj = v.apply_derivatives(ww, pp, 3)
        >>> print(yj.shape, bool(np.allclose(yj[0], v.apply(ww))))          # (order+1,); order 0 == apply
        (4,) True
        """
        probe_derivatives.check_perturbation_vectors(ww, pp)
        return ufv_sampling.ut3tangent_apply_derivatives(ww, pp, self.frame.data, self.variations.data, order)

    def entries_derivatives(
            self,
            index:  NDArray,                # int, shape=(d,)+W -- the grid points
            pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                    # highest derivative order
    ) -> NDArray:                           # scalar jets, shape=(order+1,)+W+K+C
        """Symmetric entry derivatives at ``index`` in direction ``P`` (derivative twin of :py:meth:`entries`;
        the bare ``𝒥``). ``index`` and ``pp`` share ``W``. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.entries_derivatives`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> import t3toolbox.uniform_manifold as ut3m
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(x))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> yj = v.entries_derivatives(np.array([3, 5, 7]), pp, 3)
        >>> print(yj.shape, bool(np.allclose(yj[0], v.entries((3, 5, 7)))))  # (order+1,); order 0 == entries
        (4,) True
        """
        probe_derivatives.check_perturbation_index(index, pp, self.shape)
        return ufv_sampling.ut3tangent_entries_derivatives(index, pp, self.frame.data, self.variations.data, order)

    @staticmethod
    def probe_transpose(
            ztildes,                   # probe residuals, len=d, ith elm_shape=W+K+C+(Ni,)
            ww,                        # probe vectors,   len=d, ith elm_shape=W+(Ni,)
            frame:  ubv.UT3Frame,      # the orthogonal frame the tangent attaches at
            sum_over_probes:  bool = False,   # True: sum the probe stack W (Gauss-Newton 𝒥ᵀr); False: keep it
    ) -> 'UT3Tangent':  # tangent stack W+K (sum_over_probes=False) or K (True); frame stack C
        """Apply the transpose ``𝒥ᵀ`` of the probe map to residuals; returns a :py:class:`UT3Tangent` at
        ``frame``. The adjoint of :py:meth:`probe`. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.probe_transpose`: the residuals live in the forward probe
        space (``W+K+C``); ``sum_over_probes=False`` keeps the probe stack ``W`` as the result's tangent
        stack (``W+K``), ``True`` sums it (``K``, the Gauss-Newton ``𝒥ᵀr``). The bare ``𝒥ᵀ`` (no gauge
        projector ``Π``).

        See Also
        --------
        probe
        apply_transpose
        probe_derivatives_transpose

        Examples
        --------
        The defining property -- the adjoint identity ``<r, 𝒥v> = <𝒥ᵀr, v>`` (summed over probes):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> import t3toolbox.uniform_manifold as ut3m
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(x))
        >>> ww = [np.random.randn(2, N) for N in (10, 11, 12)]              # probe stack W=(2,)
        >>> zz = v.probe(ww)
        >>> r = [np.random.randn(*np.asarray(z).shape) for z in zz]
        >>> JTr = ut3m.UT3Tangent.probe_transpose(r, ww, v.frame, sum_over_probes=True)
        >>> lhs = sum(float(np.sum(ri * np.asarray(zi))) for ri, zi in zip(r, zz))
        >>> print(bool(abs(lhs - float(JTr.corewise_inner(v))) < 1e-9))
        True
        >>> keep = ut3m.UT3Tangent.probe_transpose(r, ww, v.frame, sum_over_probes=False)
        >>> print(keep.tangent_stack_shape, JTr.tangent_stack_shape)        # keep W (=W+K) vs sum W (=K)
        (2,) ()
        """
        vd = ufv_sampling.ut3tangent_probe_transpose(
            ztildes, ww, frame.data, sum_over_probes=sum_over_probes)
        return UT3Tangent(frame, _ut3variations_from_data(vd))

    @staticmethod
    def apply_transpose(
            c,                         # residual, shape=W+K+C (a scalar per stack element)
            ww,                        # apply vectors, len=d, ith elm_shape=W+(Ni,)
            frame:  ubv.UT3Frame,
            sum_over_probes:  bool = False,
    ) -> 'UT3Tangent':  # tangent stack W+K (sum_over_probes=False) or K (True); frame stack C
        """Apply the transpose ``𝒥ᵀ`` of :py:meth:`apply` -- back-project a residual ``c`` into a tangent
        at ``frame``. The adjoint of :py:meth:`apply`. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.apply_transpose`; ``sum_over_probes`` as in
        :py:meth:`probe_transpose`. The bare ``𝒥ᵀ`` (no gauge projector ``Π``)."""
        vd = ufv_sampling.ut3tangent_apply_transpose(
            c, ww, frame.data, sum_over_probes=sum_over_probes)
        return UT3Tangent(frame, _ut3variations_from_data(vd))

    @staticmethod
    def entries_transpose(
            c,                         # residual, shape=W+K+C
            index,                     # int, shape=(d,)+W (the indices whose entries c weights)
            frame:  ubv.UT3Frame,
            sum_over_probes:  bool = False,
    ) -> 'UT3Tangent':  # tangent stack W+K (sum_over_probes=False) or K (True); frame stack C
        """Apply the transpose ``𝒥ᵀ`` of :py:meth:`entries` -- scatter ``c`` at ``index`` into a tangent at
        ``frame``. The adjoint of :py:meth:`entries` (= :py:meth:`apply_transpose` with one-hot vectors).
        Uniform mirror of :py:meth:`~t3toolbox.manifold.T3Tangent.entries_transpose`. The bare ``𝒥ᵀ``."""
        vd = ufv_sampling.ut3tangent_entries_transpose(
            c, index, frame.data, sum_over_probes=sum_over_probes)
        return UT3Tangent(frame, _ut3variations_from_data(vd))

    # --------------------------------------------------------- derivative transpose 𝒥ᵀ (jets; 3b-6'c)
    @staticmethod
    def probe_derivatives_transpose(
            ztildes,                   # probe residual jets, len=d, ith elm_shape=(order+1,)+W+K+C+(Ni,)
            ww,                        # probe vectors,   len=d, ith elm_shape=W+(Ni,)
            pp,                        # perturbation P,  len=d, ith elm_shape=W+(Ni,)
            frame:  ubv.UT3Frame,      # the orthogonal frame the tangent attaches at
            order:  int,               # highest derivative order
            sum_over_probes:  bool = False,
    ) -> 'UT3Tangent':  # tangent stack W+K (sum_over_probes=False) or K (True); frame stack C
        """Transpose ``𝒥ᵀ`` of :py:meth:`probe_derivatives`: back-project residual jets into a
        :py:class:`UT3Tangent` at ``frame``. The residual jets live in the forward derivative-probe space
        (``(order+1)+W+K+C+(Ni,)``); the transpose sums the order axis, so the result is a single tangent
        (no order axis). The tangent batch ``K`` rides through, ``sum_over_probes`` sums (``True``,
        Gauss-Newton ``𝒥ᵀr``) or keeps (``False``) the sample stack ``W``. Bare ``𝒥ᵀ``. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.probe_derivatives_transpose`.

        See Also
        --------
        probe_derivatives
        probe_transpose
        apply_derivatives_transpose

        Examples
        --------
        The adjoint identity ``<r, 𝒥v> = <𝒥ᵀr, v>`` per order (the measurement dot sums the order axis too):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> import t3toolbox.uniform_manifold as ut3m
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)))
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(x))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> Jv = v.probe_derivatives(ww, pp, 2)
        >>> r = [np.random.randn(*np.asarray(z).shape) for z in Jv]
        >>> JTr = ut3m.UT3Tangent.probe_derivatives_transpose(r, ww, pp, v.frame, 2, sum_over_probes=True)
        >>> lhs = sum(float(np.sum(ri * np.asarray(zi))) for ri, zi in zip(r, Jv))
        >>> print(bool(abs(lhs - float(JTr.corewise_inner(v))) < 1e-9))
        True
        """
        probe_derivatives.check_perturbation_vectors(ww, pp)
        vd = ufv_sampling.ut3tangent_probe_derivatives_transpose(
            ztildes, ww, pp, frame.data, order, sum_over_probes=sum_over_probes)
        return UT3Tangent(frame, _ut3variations_from_data(vd))

    @staticmethod
    def apply_derivatives_transpose(
            c,                         # residual jet (scalar), shape=(order+1,)+W+K+C
            ww,                        # apply vectors, len=d, ith elm_shape=W+(Ni,)
            pp,                        # perturbation P, len=d, ith elm_shape=W+(Ni,)
            frame:  ubv.UT3Frame,
            order:  int,
            sum_over_probes:  bool = False,
    ) -> 'UT3Tangent':
        """Transpose ``𝒥ᵀ`` of :py:meth:`apply_derivatives` (the adjoint-state apply-derivative transpose).
        ``sum_over_probes`` as in :py:meth:`probe_derivatives_transpose`. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.apply_derivatives_transpose`."""
        probe_derivatives.check_perturbation_vectors(ww, pp)
        vd = ufv_sampling.ut3tangent_apply_derivatives_transpose(
            c, ww, pp, frame.data, order, sum_over_probes=sum_over_probes)
        return UT3Tangent(frame, _ut3variations_from_data(vd))

    @staticmethod
    def entries_derivatives_transpose(
            c,                         # residual jet (scalar), shape=(order+1,)+W+K+C
            index,                     # int, shape=(d,)+W
            pp,                        # perturbation P, len=d, ith elm_shape=W+(Ni,)
            frame:  ubv.UT3Frame,
            order:  int,
            sum_over_probes:  bool = False,
    ) -> 'UT3Tangent':
        """Transpose ``𝒥ᵀ`` of :py:meth:`entries_derivatives`: scatter residual jets ``c`` at ``index`` into a
        tangent (= :py:meth:`apply_derivatives_transpose` with one-hot vectors). Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.entries_derivatives_transpose`."""
        probe_derivatives.check_perturbation_index(index, pp, frame.shape)
        vd = ufv_sampling.ut3tangent_entries_derivatives_transpose(
            c, index, pp, frame.data, order, sum_over_probes=sum_over_probes)
        return UT3Tangent(frame, _ut3variations_from_data(vd))

    def sum_tangents(self, axis=None) -> 'UT3Tangent':
        """Sum over the tangent stack ``K`` (a batch of tangents at the shared frame) into one tangent.

        Corewise (= the tensor sum, by linearity); the frame stack ``C`` is preserved. ``axis`` indexes
        within ``K`` (default: the whole tangent stack). The masks OR over the summed axes -- a no-op for
        a single-frame ``K`` stack (constant masks), but the correct reduction in general."""
        vd = ufv_tangent_operations.sum_tangent_stack(
            self.variations.data, len(self.tangent_stack_shape), axis)
        return UT3Tangent(self.frame, _ut3variations_from_data(vd))

    # ------------------------------------------------------------- constructors
    @staticmethod
    def zeros(
            frame:        ubv.UT3Frame,
            stack_shape:  typ.Tuple[int, ...] = (),  # extra tangent stack K (a batch of tangents)
    ) -> 'UT3Tangent':
        """Zero tangent vector at a given frame (numpy/jax matching the frame).

        ``stack_shape`` is the extra *outer* tangent stack ``K`` (a batch of tangents sharing this frame);
        the variation supercores are stacked ``(d,) + K + C + (core,)``, and the variation masks are the
        frame's gauge masks replicated (constant) along ``K``. Default ``K = ()``."""
        K = tuple(stack_shape)
        gauge_masks = ubv.UT3Variations._variation_masks_of(frame)         # gauge-shifted masks, stack C
        masks = _broadcast_variation_masks_over_K(gauge_masks, K)          # -> stack K + C
        variations = ubv.UT3Variations.zeros(
            frame.uniform_variation_shapes, frame.shape, stack_shape=K + frame.stack_shape,
            masks=masks, use_jax=frame.contains_jax)
        return UT3Tangent(frame, variations)

    @staticmethod
    def unit(
            frame:  ubv.UT3Frame,
            index:  typ.Tuple[bool, int, typ.Sequence[int]],  # (use_tt_coordinate, i, within_index)
    ) -> 'UT3Tangent':
        """Canonical unit tangent at ``frame``: variations zero except a single core entry.

        ``index = (use_tt_coordinate, i, within_index)`` (see :py:meth:`UT3Variations.unit`). These units
        are the standard basis of the variation supercores -- an overcomplete, non-ambient-orthogonal
        generating set of the tangent space, not an orthonormal basis (gauge it yourself if needed)."""
        gauge_masks = ubv.UT3Variations._variation_masks_of(frame)         # gauge masks at C (no K)
        variations = ubv.UT3Variations.unit(
            frame.uniform_variation_shapes, frame.shape, index,
            stack_shape=frame.stack_shape, masks=gauge_masks, use_jax=frame.contains_jax)
        return UT3Tangent(frame, variations)

    @staticmethod
    def zeros_like(tangent: 'UT3Tangent') -> 'UT3Tangent':
        """Zero tangent at ``tangent``'s frame, with ``tangent``'s tangent stack ``K``."""
        return UT3Tangent.zeros(tangent.frame, stack_shape=tangent.tangent_stack_shape)

    # ------------------------------------------------------------- stack/unstack (tree <-> stacked tangent)
    def unstack_tangents(self):
        """Unstack over the tangent stack ``K``: a ``K``-shaped tree of tangents sharing this frame.

        Decomposes the batch of tangent *directions*. Each leaf is a :py:class:`UT3Tangent` with
        ``tangent_stack_shape == ()`` and this tangent's ``frame_stack_shape`` -- and, because the frame is
        shared across ``K``, every leaf holds the **same** :py:class:`UT3Frame` object, so the leaves live
        in one tangent space. Inverse of :py:meth:`stack_tangents`."""
        variations_tree = ufv_tangent_operations.unstack_tangent_stack(self.frame.data, self.variations.data)
        return stacking.apply_func_to_leaf_subtrees(
            variations_tree,
            lambda vd: UT3Tangent(self.frame, _ut3variations_from_data(vd)),   # SAME frame object (shared)
            ufv_operations.ufv_leaf_structure(self.d, 2))

    def unstack_frame(self):
        """Unstack over the frame stack ``C``: a ``C``-shaped tree of single-frame-point tangents.

        Decomposes over frame *points*. Each leaf is a :py:class:`UT3Tangent` with ``frame_stack_shape == ()``
        and this tangent's ``tangent_stack_shape``; the leaves sit at **different** base points (different
        tangent spaces, not mutually linear-algebra compatible) and may have **different ranks** (the
        varying-``C`` rank-sweep case). Inverse of :py:meth:`stack_frame`."""
        paired_tree = ufv_tangent_operations.unstack_frame_stack(self.frame.data, self.variations.data)
        leaf_structure = (ufv_operations.ufv_leaf_structure(self.d, 4),    # a frame_data leaf
                          ufv_operations.ufv_leaf_structure(self.d, 2))    # a variations_data leaf
        return stacking.apply_func_to_leaf_subtrees(
            paired_tree,
            lambda bv: UT3Tangent(_ut3frame_from_data(bv[0]), _ut3variations_from_data(bv[1])),
            leaf_structure)

    @staticmethod
    def stack_tangents(tree) -> 'UT3Tangent':
        """Stack a ``K``-shaped tree of tangents (sharing one frame) into a tangent-stacked UT3Tangent.

        Inverse of :py:meth:`unstack_tangents`. Requires every leaf to be at the **same frame** (the
        numerical same-frame check, as in :py:meth:`__add__`): the tangents being stacked must live in the
        same tangent space. The first leaf's frame is reused; the variations stack over the new outer
        tangent stack ``K``."""
        leaves = _flatten_tangents(tree)
        frame = leaves[0].frame
        for t in leaves[1:]:
            if not (t.frame is frame or safety.frames_equal_or_skip(t.frame.data[:4], frame.data[:4])):
                raise ValueError(
                    'stack_tangents requires every tangent to be at the same frame -- they must live in the '
                    'same tangent space. To stack tangents at *different* base points, use stack_frame. '
                    '(Run inside safety.unsafe() to skip this numerical check.)')
        variations_tree = stacking.apply_func_to_leaf_subtrees(tree, lambda t: t.variations.data, None)
        vd = ufv_tangent_operations.stack_tangent_stack(variations_tree)
        return UT3Tangent(frame, _ut3variations_from_data(vd))

    @staticmethod
    def stack_frame(tree) -> 'UT3Tangent':
        """Stack a ``C``-shaped tree of single-frame-point tangents into a frame-stacked UT3Tangent.

        Inverse of :py:meth:`unstack_frame`. The leaves sit at **different** base points, so no shared-frame
        identity is required; they must share the padded dims ``(d, N, nU, nD, rL, rR)``, the real mode
        ``shape``, and the tangent stack ``K`` -- but the **ranks/masks MAY differ across the frame stack
        ``C``** (varying-rank stacks are supported). The bases stack over the inner frame stack ``C`` (so the
        variation stack becomes ``K + C``)."""
        leaves = _flatten_tangents(tree)
        key = lambda t: (t.frame.uniform_structure[:6], t.tangent_stack_shape, t.shape)
        k0 = key(leaves[0])
        for t in leaves[1:]:
            if key(t) != k0:
                raise ValueError(
                    'stack_frame requires all tangents to share the padded dims (d, N, nU, nD, rL, rR), the '
                    'mode shape, and the tangent stack K (only the base point -- and its ranks/masks -- may '
                    'differ across the frame stack C).')
        paired_tree = stacking.apply_func_to_leaf_subtrees(
            tree, lambda t: (t.frame.data, t.variations.data), None)
        frame_data, variations_data = ufv_tangent_operations.stack_frame_stack(paired_tree)
        return UT3Tangent(_ut3frame_from_data(frame_data), _ut3variations_from_data(variations_data))


if has_jax:
    import jax

    # Register UT3Tangent as a jax pytree with the frame as a LEAF: both the frame and the variations are
    # children (no aux_data at this level -- each carries its own static masks aux internally). Mirrors the
    # ragged T3Tangent: the frame flows as ordinary traced data, so a tangent crossing a jit boundary does
    # NOT recompile when the frame changes; the same-tangent-space guard is the NUMERICAL same-frame check
    # (safety.frames_equal_or_skip, safe-mode + eager-only), which survives a jit round-trip and skips
    # under a trace. (The masks ride inside UT3Frame / UT3Variations as value-hashed aux.)
    jax.tree_util.register_pytree_node(
        UT3Tangent,
        lambda x: ((x.frame, x.variations), None),
        lambda aux, children: UT3Tangent(children[0], children[1]),
    )


# ============================================================================================== geometries
# The uniform mirror of t3m.ManifoldGeometry / CorewiseGeometry (manifold.py). Stateless bundles of the
# chart-level choices (frame / gauge projection / retraction + metric); the point lives in the caller. Each
# method delegates to the 3b backend (ufv_tangent_operations) and the 2c UT3Frame / UT3Variations, behind
# the per-element safe-mode preconditions (.all() over the per-stack-element checkers, from 2c-G). See the
# ragged docstrings for the math (Section 6 + Appendix A.3 of Alger et al. 2026, arXiv:2603.21141).

def _ut3_from_data(data) -> ut3.UniformTuckerTensorTrain:  # data = (tk_sc, tt_sc, shape, (tucker_mask, tt_mask))
    return ut3.UniformTuckerTensorTrain(data[0], data[1], data[2], ut3.UT3Masks(*data[3]))


def _require_orthogonal_frame(frame: ubv.UT3Frame, who: str) -> None:
    """Safe-mode ORTH precondition (uniform mirror of :py:func:`t3toolbox.manifold._require_orthogonal_frame`).

    ``who`` (a manifold geometry op) requires an orthonormal frame **on every frame-stack element**; in safe
    mode a non-orthogonal element raises. Skipped under ``safety.unsafe()`` / a jax trace. The gating helpers
    are fed the four supercores (``frame.data[:4]``) -- not the full ``.data``, whose trailing ``shape`` int
    tuple and mask holder carry no float content -- and the orthogonality check itself is mask-aware (it
    tests the real, masked content; the contract is ``to_t3frame(frame).is_orthogonal()``). ORTH (not
    minimal rank) is the only numerical precondition for the manifold projections / retraction (see
    ``docs/numerical_contract_catalog.md``)."""
    if safety.checks_active(frame.data[:4]):
        atol = safety.effective_rtol(frame.data[:4])
        safety.require(
            frame.is_orthogonal(atol=atol).all(),     # per-element check -> require ALL frame-stack elements orthogonal
            '{} requires an orthogonal frame (the manifold geometry needs an orthonormal frame to be the '
            'Hilbert-Schmidt-orthogonal projection). Build the frame with UniformManifoldGeometry.frame / '
            'UT3Frame.random_orthogonal, or run in unsafe mode (safety.unsafe()).'.format(who))


def _randn_variations_at(
        frame:  ubv.UT3Frame,
        K:      typ.Tuple[int, ...],   # extra tangent stack K (a batch of directions)
) -> ubv.UT3Variations:  # raw N(0,1) variations at frame (ungauged), gauge masks broadcast over K
    """Raw i.i.d. ``N(0, 1)`` variations at ``frame`` with the frame's gauge-shifted rank masks replicated
    (constant) along the tangent stack ``K`` -- the natural (corewise) Gaussian on the variation supercores,
    before any gauge projection. Mirrors :py:meth:`UT3Tangent.zeros`'s mask derivation with a randn fill (the
    masks are the rank structure, identical for both geometries; only whether you gauge afterwards differs)."""
    K = tuple(K)
    gauge_masks = ubv.UT3Variations._variation_masks_of(frame)        # gauge-shifted masks, stack C
    masks = _broadcast_variation_masks_over_K(gauge_masks, K)         # -> stack K + C
    return ubv.UT3Variations.randn(
        frame.uniform_variation_shapes, frame.shape, stack_shape=K + frame.stack_shape,
        masks=masks, use_jax=frame.contains_jax)


class UniformManifoldGeometry:
    """The Riemannian geometry of the uniform fixed-rank Tucker tensor train manifold (uniform mirror of
    :py:class:`~t3toolbox.manifold.ManifoldGeometry`).

    A stateless bundle of the chart-level choices that distinguish the manifold from the over-parametrized
    corewise geometry -- the orthonormal frame (:py:meth:`frame`), the gauge projection ``Pi``
    (:py:meth:`project`), and the manifold retraction (:py:meth:`retract`) -- plus the ambient projection and
    transport. Tangents live in ``T_xM`` with an orthonormal, gauged frame; the metric is Hilbert-Schmidt.
    All methods are **per-element-mask-aware** (varying ranks across the frame stack ``C`` are supported) and
    carry the per-element safe-mode preconditions (orthogonal frame, gauged variations; ``.all()`` over the
    stack). Use the module singleton :py:data:`UNIFORM_MANIFOLD`. The corewise counterpart is
    :py:class:`UniformCorewiseGeometry`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> np.random.seed(0)
    >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1)))
    >>> frame = ut3m.UNIFORM_MANIFOLD.frame(x)            # orthonormal frame at x
    >>> print(bool(frame.is_orthogonal().all()))
    True
    >>> v = ut3m.UNIFORM_MANIFOLD.randn(frame)           # a standard Gaussian on T_xM (gauged)
    >>> print(bool(v.is_gauged().all()))
    True
    >>> y = ut3m.UNIFORM_MANIFOLD.retract(ut3m.UT3Tangent.zeros(frame))   # retract the zero tangent == the base point
    >>> print(bool(np.allclose(y.to_dense(), x.to_dense())))
    True
    """

    def frame(
            self,
            x:  ut3.UniformTuckerTensorTrain,
    ) -> ubv.UT3Frame:  # orthonormal frame at x
        """The orthonormal frame at ``x`` (the frame part of ``ut3_orthogonal_representations``)."""
        return ubv.UT3Frame.from_ut3(x)

    def randn(
            self,
            frame:        ubv.UT3Frame,
            stack_shape:  typ.Tuple[int, ...] = (),   # extra tangent stack K (a batch of tangents)
    ) -> UT3Tangent:  # gauged random tangent at frame
        """Random tangent at ``frame``: a standard Gaussian on the tangent space ``T_xM``.

        Raw i.i.d. ``N(0, 1)`` variation supercores, then the gauge projection :py:meth:`project` (``Pi``).
        For an orthogonal, minimal-rank ``frame`` this is the standard Gaussian on ``T_xM``. ``stack_shape``
        is the extra outer tangent stack ``K`` (default ``()``). Inherits :py:meth:`project`'s safe-mode ORTH
        precondition (a non-orthogonal ``frame`` raises; skipped under ``safety.unsafe()`` / a jax trace)."""
        return self.project(UT3Tangent(frame, _randn_variations_at(frame, stack_shape)))

    def random_orthogonal(
            self,
            shape:                  typ.Sequence[int],         # (N0,...,N(d-1))
            tucker_ranks,                                      # int | len-d seq | (d,)+C array (the variety)
            tt_ranks,                                          # int | len-(d+1) seq | (d+1,)+C array
            stack_shape:            typ.Tuple[int, ...] = (),  # frame stack C (random base points)
            tangent_stack_shape:    typ.Tuple[int, ...] = (),  # tangent stack K
            use_jax:                bool = False,
    ) -> UT3Tangent:  # gauged random tangent at a random orthogonal base point
        """A gauged random tangent at a random orthogonal base point (random direction, random frame)."""
        frame = ubv.UT3Frame.random_orthogonal(shape, tucker_ranks, tt_ranks,
                                              stack_shape=stack_shape, use_jax=use_jax)
        return self.randn(frame, stack_shape=tangent_stack_shape)

    def randn_like(
            self,
            tangent:    UT3Tangent,   # reuse its frame + tangent stack K
    ) -> UT3Tangent:  # gauged random tangent at tangent's frame
        """A gauged random tangent at ``tangent``'s frame, with ``tangent``'s tangent stack ``K``."""
        return self.randn(tangent.frame, stack_shape=tangent.tangent_stack_shape)

    def project(
            self,
            v:  UT3Tangent,
    ) -> UT3Tangent:  # gauged tangent at v's frame (a DIFFERENT vector)
        """The gauge projection ``Pi``: raw cotangent -> Riemannian gradient (orthogonal gauge).

        Orthogonally projects ``v``'s variations onto the gauged tangent space (conditions (48)-(49),
        Appendix A.3). Represents a DIFFERENT tangent vector than ``v`` -- the map that turns a bare adjoint
        ``J^T r`` into the Riemannian gradient. For the vector-preserving fix see :py:meth:`project_oblique`.
        **Safe mode** requires ``v``'s frame orthogonal (raises otherwise; skipped under ``safety.unsafe()`` /
        a jax trace)."""
        _require_orthogonal_frame(v.frame, 'UniformManifoldGeometry.project')
        vd = ufv_tangent_operations.orthogonal_gauge_projection(v.frame.data, v.variations.data)
        return UT3Tangent(v.frame, _ut3variations_from_data(vd))

    def project_oblique(
            self,
            v:  UT3Tangent,
    ) -> UT3Tangent:  # gauged tangent at v's frame (the SAME vector)
        """Gauge ``v``'s variations while preserving the represented tangent vector (oblique projection).

        Returns a tangent at the same frame representing the SAME vector as ``v`` but gauged, so that on an
        orthogonal minimal-rank frame :py:meth:`inner` / :py:meth:`norm` give the true Hilbert-Schmidt values.
        **Safe mode** requires ``v``'s frame orthogonal (raises otherwise); skipped under ``safety.unsafe()``
        / a jax trace."""
        _require_orthogonal_frame(v.frame, 'UniformManifoldGeometry.project_oblique')
        vd = ufv_tangent_operations.oblique_gauge_projection(v.frame.data, v.variations.data)
        return UT3Tangent(v.frame, _ut3variations_from_data(vd))

    def inner(
            self,
            t1: UT3Tangent,
            t2: UT3Tangent,
    ) -> NDArray:  # the Hilbert-Schmidt inner product, shape = stack_shape (K + C)
        """The **Hilbert-Schmidt** inner product of two tangents -- the Riemannian metric on the manifold.

        The corewise (coordinate) dot, which equals HS on this geometry's orthonormal, gauged frame
        (vectorized over the stack -> shape ``K + C``). In **safe mode** it checks the preconditions for that
        equality: the two tangents share a frame, the frame is orthogonal, and **both** variations are gauged
        (per stack element, reduced with ``.all()``; minimal rank is a documented caveat -- see
        ``docs/numerical_contract_catalog.md``). For the raw coordinate dot with no HS claim and no
        orthogonal/gauge check, use :py:meth:`UT3Tangent.corewise_inner`."""
        t1._check_same_tangent_space(t2)
        if safety.checks_active(t1.frame.data[:4], t1.variations.data[:2], t2.variations.data[:2]):
            atol = safety.effective_rtol(t1.frame.data[:4], t1.variations.data[:2], t2.variations.data[:2])
            safety.require(t1.frame.is_orthogonal(atol=atol).all(),
                           'UniformManifoldGeometry.inner is the Hilbert-Schmidt metric and requires an '
                           'orthogonal frame. Use UT3Tangent.corewise_inner for the raw coordinate dot, or '
                           'run in unsafe mode (safety.unsafe()).')
            safety.require(t1.is_gauged(atol=atol).all() and t2.is_gauged(atol=atol).all(),
                           'UniformManifoldGeometry.inner requires both tangents gauged. Gauge them via '
                           'UniformManifoldGeometry.project / project_oblique, use UT3Tangent.corewise_inner, '
                           'or run in unsafe mode.')
        return t1.corewise_inner(t2)

    def norm(
            self,
            t:  UT3Tangent,
    ) -> NDArray:  # the Hilbert-Schmidt norm, shape = stack_shape (K + C)
        """The **Hilbert-Schmidt** norm of a tangent. Safe mode checks the frame orthogonal + variations
        gauged, per stack element (the preconditions for the coordinate norm to equal HS; minimal rank a
        documented caveat). For the raw coordinate norm use :py:meth:`UT3Tangent.corewise_norm`."""
        if safety.checks_active(t.frame.data[:4], t.variations.data[:2]):
            atol = safety.effective_rtol(t.frame.data[:4], t.variations.data[:2])
            safety.require(t.frame.is_orthogonal(atol=atol).all(),
                           'UniformManifoldGeometry.norm is the Hilbert-Schmidt metric and requires an '
                           'orthogonal frame. Use UT3Tangent.corewise_norm, or run in unsafe mode.')
            safety.require(t.is_gauged(atol=atol).all(),
                           'UniformManifoldGeometry.norm requires gauged variations. Gauge via '
                           'UniformManifoldGeometry.project / project_oblique, use UT3Tangent.corewise_norm, '
                           'or run in unsafe mode.')
        return t.corewise_norm()

    def retract(
            self,
            p:  UT3Tangent,   # step (a tangent at the current point's frame)
    ) -> ut3.UniformTuckerTensorTrain:  # retracted point on M, at p's frame ranks
        """Retract the step ``p`` to the manifold: shifted doubled-rank embedding, truncated to frame ranks.

        Forms ``base point + p`` and truncates back to ``p``'s frame ranks via the implicit uniform T3-SVD
        (Algorithm 10). The current point is carried by ``p.frame``, so no separate point argument is needed.
        **Safe mode** requires ``p``'s frame orthogonal (raises otherwise; skipped under ``safety.unsafe()`` /
        a jax trace). ORTH only -- retract is gauge-invariant; minimal rank is a documented caveat."""
        _require_orthogonal_frame(p.frame, 'UniformManifoldGeometry.retract')
        return _ut3_from_data(ufv_tangent_operations.retract(p.frame.data, p.variations.data))

    def project_ambient(
            self,
            frame:  ubv.UT3Frame,                     # orthogonal base point of the tangent space
            grad:   ut3.UniformTuckerTensorTrain,     # ambient gradient as a uniform Tucker tensor train
    ) -> UT3Tangent:  # the Riemannian gradient (gauged projection of grad) at frame
        """Project an ambient gradient onto ``T_xM`` -- the Riemannian gradient.

        ``grad`` is the Euclidean/ambient gradient as a :py:class:`UniformTuckerTensorTrain`. Returns the
        gauged tangent ``P_T(grad)`` (the residual ``grad - P_T(grad)`` is orthogonal to the tangent space).
        Requires an **orthogonal** ``frame`` (minimal rank not required); enforced in safe mode (skipped under
        ``safety.unsafe()`` / a jax trace).

        Unlike the ragged :py:meth:`~t3toolbox.manifold.ManifoldGeometry.project_ambient`, the uniform layer
        has **no native dense-array path** (the uniform layer is a performance layer; working dense here
        defeats its purpose). For a dense gradient, go via the ragged geometry and the cross-layer converters
        (:py:meth:`UT3Frame.to_t3frame` -> ``MANIFOLD.project_ambient`` -> :py:meth:`UT3Tangent.from_t3tangent`)."""
        if not isinstance(grad, ut3.UniformTuckerTensorTrain):
            raise TypeError(
                'UniformManifoldGeometry.project_ambient accepts a UniformTuckerTensorTrain gradient only; '
                'the uniform layer has no native dense-array projection. For a dense gradient, project via '
                'the ragged ManifoldGeometry.project_ambient and the cross-layer converters '
                '(UT3Frame.to_t3frame / UT3Tangent.from_t3tangent). Got %r.' % (type(grad).__name__,))
        _require_orthogonal_frame(frame, 'UniformManifoldGeometry.project_ambient')
        vd = ufv_tangent_operations.project_ut3_onto_tangent_space(frame.data, grad.data)
        return UT3Tangent(frame, _ut3variations_from_data(vd))

    def transport(
            self,
            v:          UT3Tangent,
            new_frame:  ubv.UT3Frame,
    ) -> UT3Tangent:  # v transported to the tangent space at new_frame
        """Projective vector transport of ``v`` to the tangent space at ``new_frame``.

        Re-projects ``v`` (as an ambient tensor via its doubled-rank :py:meth:`UT3Tangent.to_ut3`)
        orthogonally onto the tangent space at ``new_frame`` -- the cheap, standard choice for fixed-rank
        Riemannian optimization (not parallel transport). Inherits :py:meth:`project_ambient`'s safe-mode ORTH
        precondition on ``new_frame``."""
        return self.project_ambient(new_frame, v.to_ut3())


class UniformCorewiseGeometry:
    """The Euclidean geometry of the uniform core parameter space (uniform mirror of
    :py:class:`~t3toolbox.manifold.CorewiseGeometry`).

    Optimization happens *on* the raw supercores: tangents are perturbations of the cores ``(U, G, G, G)``
    (the non-orthonormal frame whose down/left/right cores are all the TT supercore ``G`` -- the Section 6.3
    ``(P, Q, O) -> G`` substitution), the metric is the plain Euclidean (corewise) inner product, the
    "projection" is the identity (no gauge), and the retraction is vector addition in the supercores. Use the
    module singleton :py:data:`UNIFORM_COREWISE`. The manifold counterpart is
    :py:class:`UniformManifoldGeometry`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> np.random.seed(0)
    >>> x = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1)))
    >>> frame = ut3m.UNIFORM_COREWISE.frame(x)            # the (U, G, G, G) frame
    >>> v = ut3m.UNIFORM_COREWISE.randn(frame)           # raw randn cores (no gauge)
    >>> print(bool(v.is_gauged().all()))
    False
    >>> y = ut3m.UNIFORM_COREWISE.retract(ut3m.UT3Tangent.zeros(frame))   # additive: cores += step (zero -> the point)
    >>> print(bool(np.allclose(y.to_dense(), x.to_dense())))
    True
    """

    def frame(
            self,
            x:  ut3.UniformTuckerTensorTrain,
    ) -> ubv.UT3Frame:  # the (U, G, G, G) non-orthonormal frame at x
        """The core-parameter frame at ``x``: ``(U, G, G, G)`` (down/left/right all the TT supercore ``G``).

        The mask analog of the cores: ``up = down = tucker_edge_mask`` (the U-G Tucker edges) and
        ``left = right = tt_edge_mask`` (the G-G TT bonds, full length ``d+1``) -- the Section 6.3
        substitution ``(P, Q, O) -> G`` gives all three non-up slots the single core ``G``'s rank structure,
        with no boundary slicing (the ``[:-1]`` / ``[1:]`` gauge shift is a *variations*, not a frame,
        thing)."""
        tk_sc, tt_sc, shape, (tucker_mask, tt_mask) = x.data
        masks = ubv.UT3FrameMasks(tucker_mask, tucker_mask, tt_mask, tt_mask)
        return ubv.UT3Frame(tk_sc, tt_sc, tt_sc, tt_sc, shape, masks)

    def randn(
            self,
            frame:        ubv.UT3Frame,
            stack_shape:  typ.Tuple[int, ...] = (),   # extra tangent stack K
    ) -> UT3Tangent:  # raw random tangent at frame (ungauged)
        """Random tangent at ``frame``: raw i.i.d. ``N(0, 1)`` variation supercores (the natural corewise
        Gaussian; no gauge projection). ``stack_shape`` is the extra outer tangent stack ``K``."""
        return UT3Tangent(frame, _randn_variations_at(frame, stack_shape))

    def randn_like(
            self,
            tangent:    UT3Tangent,
    ) -> UT3Tangent:  # raw random tangent at tangent's frame
        """A raw random tangent at ``tangent``'s frame, with ``tangent``'s tangent stack ``K``."""
        return self.randn(tangent.frame, stack_shape=tangent.tangent_stack_shape)

    def project(
            self,
            v:  UT3Tangent,
    ) -> UT3Tangent:  # v unchanged (no gauge on the core parameter space)
        """The identity: the core parameter space is Euclidean, with no gauge projection."""
        return v

    def inner(
            self,
            t1: UT3Tangent,
            t2: UT3Tangent,
    ) -> NDArray:  # the Euclidean (coordinate) inner product, shape = stack_shape (K + C)
        """The **Euclidean** (coordinate) inner product of two tangents on the core parameter space.

        The corewise dot of the variations -- *no* orthogonal/gauge requirement (the ``(U, G, G, G)`` frame is
        non-orthonormal by design). Safe mode checks only that the two tangents share a frame. This is exactly
        :py:meth:`UT3Tangent.corewise_inner`."""
        return t1.corewise_inner(t2)

    def norm(
            self,
            t:  UT3Tangent,
    ) -> NDArray:  # the Euclidean (coordinate) norm, shape = stack_shape (K + C)
        """The **Euclidean** (coordinate) norm of a tangent (= :py:meth:`UT3Tangent.corewise_norm`); no
        precondition."""
        return t.corewise_norm()

    def retract(
            self,
            p:  UT3Tangent,   # step (a corewise tangent at frame = (U, G, G, G))
    ) -> ut3.UniformTuckerTensorTrain:  # additive retraction: cores += p
        """Additive retraction: add the variation supercores to the point's cores (``cores += p``).

        Recovers the point ``(U, G)`` from ``p.frame`` (which :py:meth:`frame` built as ``(U, G, G, G)``) and
        adds the variations. ``p`` must be a corewise tangent (a frame from :py:meth:`frame`). See
        :py:func:`~t3toolbox.backend.ufv_tangent_operations.corewise_retract`."""
        return _ut3_from_data(ufv_tangent_operations.corewise_retract(p.frame.data, p.variations.data))


UNIFORM_MANIFOLD = UniformManifoldGeometry()   # the uniform fixed-rank Riemannian geometry (gauge Pi, manifold retraction)
UNIFORM_COREWISE = UniformCorewiseGeometry()   # the uniform core-parameter Euclidean geometry (no gauge, additive retraction)
