# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform tangent + manifold layer (the uniform-fix 1.0 centerpiece, increment 3b).

Mirrors the ragged :py:mod:`t3toolbox.manifold` on the uniform (stacked-supercore + boolean-mask)
representation. Holds:

- :py:class:`UT3Tangent` -- the structural bundle ``(UT3Basis, UT3Variations)`` (the ``K``/``C`` stack
  inferred from the pair), the vector-space ops, the raw coordinate inner/norm, the (delegating) validity
  checkers, the constructors, ``reverse``, the doubled-rank ``to_ut3``/``to_dense``, ``gauge_residual`` /
  ``is_gauged``, the cross-layer converters, and the stack/unstack tree conversions;
- :py:class:`UniformManifoldGeometry` / :py:class:`UniformCorewiseGeometry` (3b-5) -- the stateless
  geometry bundles (``base`` / ``randn`` / ``project`` / ``inner`` / ``norm`` / ``retract`` ...) mirroring
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

import t3toolbox.uniform_basis_variations_format as ubv
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.basis_variations_format as bvf
import t3toolbox.safety as safety
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ubv_operations as ubv_operations
import t3toolbox.backend.ubv_tangent_operations as ubv_tangent_operations
import t3toolbox.backend.ubv_sampling as ubv_sampling
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

    A ``K``-stacked zero tangent is a bundle of ``K`` tangent vectors at one base, so each tangent
    carries the *same* gauge masks -- the base's masks replicated (constant) along ``K``. Masks are host
    numpy (static aux), so this stays on ``np`` (see CLAUDE.md: supercores -> ``xnp``, masks -> ``np``).
    """
    K = tuple(K)

    def b(m):  # insert |K| size-1 axes after the leading mode (d) axis, then broadcast to (d,)+K+C+(size,)
        return np.broadcast_to(m.reshape(m.shape[:1] + (1,) * len(K) + m.shape[1:]),
                               m.shape[:1] + K + m.shape[1:])

    return ubv.UT3VariationsMasks(b(masks.variations_up_mask), b(masks.variations_down_mask),
                                  b(masks.variations_left_mask), b(masks.variations_right_mask))


def _ut3basis_from_data(bd) -> ubv.UT3Basis:        # bd = (up, down, left, right, shape, masks_tuple)
    return ubv.UT3Basis(bd[0], bd[1], bd[2], bd[3], bd[4], ubv.UT3BasisMasks(*bd[5]))


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


def _variations_stack_dot(
        va:  ubv.UT3Variations,  # mask-applied variations (real content only)
        vb:  ubv.UT3Variations,  # mask-applied variations (same structure)
) -> NDArray:                    # shape = stack_shape (K+C); scalar (0-d) when unstacked
    """Coordinate dot of two (mask-applied) variations, reduced over the non-stack axes (leading mode
    index ``d`` + the core axes), keeping the stack -- one dot per stacked tangent. Uniform-direct (the
    supercores are ``d``-leading), mirroring :py:meth:`UT3Variations.allclose`'s reduction."""
    n_stack = len(va.stack_shape)
    xnp, _, _ = get_backend(True, va.contains_jax or vb.contains_jax)
    total = 0.0
    for sa, sb in zip(va.supercores, vb.supercores):
        total = total + xnp.sum(sa * sb, axis=(0,) + tuple(range(1 + n_stack, sa.ndim)))
    return total


@dataclass(frozen=True)
class UT3Tangent:
    """Tangent vector to the uniform manifold of fixed-rank Tucker tensor trains (uniform analog of
    :py:class:`~t3toolbox.manifold.T3Tangent`).

    A ``UT3Tangent`` bundles a :py:class:`~t3toolbox.uniform_basis_variations_format.UT3Basis` (the frame
    at the base point where the tangent space is attached) with a
    :py:class:`~t3toolbox.uniform_basis_variations_format.UT3Variations` (the tangent direction in that
    frame). The ``K``/``C`` stack split (extra tangent stack ``K`` over the shared base stack ``C``) is
    **inferred** from the pair -- the variation stack is ``K + C`` and the basis stack is ``C`` -- never
    stored (the split-agnostic stacking of increment 2c).

    Like the ragged class, the metric lives on the *geometry*, not here; this exposes only the **raw
    coordinate** :py:meth:`corewise_inner` / :py:meth:`corewise_norm` (no Hilbert-Schmidt claim), computed
    on the **real (masked)** content of the variations.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_basis_variations_format as ubv
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
    >>> base, variations = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
    >>> v = ut3m.UT3Tangent(base, variations)
    >>> print(v.shape, v.stack_shape)
    (10, 11, 12) ()
    >>> print(bool(v.is_orthogonal()))   # base from ut3_orthogonal_representations is orthogonal
    True
    >>> w = 2.0 * v - v                  # linear algebra stays in the same tangent space
    >>> print(bool(w.allclose(v)))       # (2v - v) == v
    True
    """
    basis:      ubv.UT3Basis
    variations: ubv.UT3Variations

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate this tangent: both components well-formed and a compatible (basis, variations) pair.

        Runs ``basis.validate()`` + ``variations.validate()`` + the bv-pair compatibility check
        (:py:func:`~t3toolbox.uniform_basis_variations_format.check_ubv_pair`, which permits an extra
        tangent stack ``K``). Structural only (shapes / ranks / masks; masks are host numpy), so it is
        safe in ``__post_init__`` and under jit/pytree tracing.
        """
        self.basis.validate()
        self.variations.validate()
        ubv.check_ubv_pair(self.basis, self.variations)

    def __repr__(self) -> str:
        return (f"UT3Tangent(shape={self.shape}, tucker_ranks={self.basis.up_ranks}, "
                f"tt_ranks={self.basis.left_ranks}, tangent_stack={self.tangent_stack_shape}, "
                f"base_stack={self.base_stack_shape})")

    # ------------------------------------------------------------- structure (K/C inferred from the pair)
    @ft.cached_property
    def d(self) -> int:
        return self.basis.d

    @ft.cached_property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.basis.shape

    @ft.cached_property
    def base_stack_shape(self) -> typ.Tuple[int, ...]:
        """Base stack ``C``: the batch of base points, shared with the basis (``basis.stack_shape``)."""
        return self.basis.stack_shape

    @ft.cached_property
    def tangent_stack_shape(self) -> typ.Tuple[int, ...]:
        """Tangent stack ``K``: the extra *outer* batch of tangent vectors sharing this base.

        The part of the variation stack that exceeds the base stack (often empty). The variation
        supercores are stacked ``(d,) + K + C + (core,)`` -- extra axes outermost, base stack inner.
        """
        full = self.variations.stack_shape
        return full[:len(full) - len(self.base_stack_shape)]

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int, ...]:
        """Full stack ``K + C`` (``tangent_stack_shape + base_stack_shape``), outer-to-inner."""
        return self.variations.stack_shape

    @ft.cached_property
    def structure(self):
        """The basis's per-element structure ``(shape, up_ranks, down_ranks, left_ranks, right_ranks,
        stack_shape)`` (ranks are arrays over the base stack ``C``)."""
        return self.basis.structure

    @ft.cached_property
    def data(self) -> typ.Tuple[ubv.UT3Basis, ubv.UT3Variations]:
        return self.basis, self.variations

    def to_jax(self) -> 'UT3Tangent':
        """Copy with basis and variation supercores converted to jax arrays (masks stay host numpy)."""
        return UT3Tangent(self.basis.to_jax(), self.variations.to_jax())

    def to_numpy(self) -> 'UT3Tangent':
        """Copy with basis and variation supercores converted to numpy arrays."""
        return UT3Tangent(self.basis.to_numpy(), self.variations.to_numpy())

    def copy(self) -> 'UT3Tangent':
        """Deep copy (copies the basis and variation supercores)."""
        return UT3Tangent(self.basis.copy(), self.variations.copy())

    @ft.cached_property
    def contains_jax(self) -> bool:
        """True if any basis or variation supercore is a jax array."""
        return self.basis.contains_jax or self.variations.contains_jax

    # ------------------------------------------------------------- linear algebra (fixed-rank vector space)
    def _check_same_tangent_space(self, other: 'UT3Tangent') -> None:
        # STRUCTURAL (always): matching stack (K+C) and rank masks (the uniform stratum / variety point).
        # Masks are host numpy, so these are cheap and valid under jit.
        if self.stack_shape != other.stack_shape:
            raise ValueError(
                'Tangent vectors have different stack shapes; elementwise linear algebra requires '
                'matching stacks (same tangent stack K over the shared base stack C).\n'
                + str(self.stack_shape) + ' = self.stack_shape != other.stack_shape = ' + str(other.stack_shape))
        if self.variations.masks != other.variations.masks:
            raise ValueError(
                'Tangent vectors have different rank masks (different strata of the determinantal '
                'variety, i.e. different tangent spaces).')
        # NUMERICAL (skippable): are these the same frame? The `is` fast-path keeps the common eager case
        # O(1); the value compare (frame supercores only -- data[:4]; the full .data carries the int-tuple
        # `shape`, which safety's array compare cannot take) runs only when the objects differ, e.g. a jit
        # round-trip reconstructs a value-equal frame. Skips under safety.unsafe() and under a jax trace.
        if not (self.basis is other.basis
                or safety.frames_equal_or_skip(self.basis.data[:4], other.basis.data[:4])):
            raise ValueError(
                'Tangent vectors are in different tangent spaces (their frames are not the same frame).\n'
                'Linear algebra between tangent vectors requires the same frame; run inside '
                'safety.unsafe() to skip this numerical check.')

    def __add__(self, other: 'UT3Tangent') -> 'UT3Tangent':
        """Add tangent vectors. Requires the same frame + matching stack/masks (same tangent space)."""
        self._check_same_tangent_space(other)
        return UT3Tangent(self.basis, self.variations + other.variations)

    def __sub__(self, other: 'UT3Tangent') -> 'UT3Tangent':
        """Subtract tangent vectors. Requires the same frame + matching stack/masks."""
        self._check_same_tangent_space(other)
        return UT3Tangent(self.basis, self.variations - other.variations)

    def __mul__(self, scalar) -> 'UT3Tangent':
        """Scale a tangent vector by a scalar (the base point is unchanged)."""
        return UT3Tangent(self.basis, self.variations * scalar)

    __rmul__ = __mul__

    def __neg__(self) -> 'UT3Tangent':
        return UT3Tangent(self.basis, -self.variations)

    def corewise_inner(self, other: 'UT3Tangent') -> NDArray:  # shape = stack_shape (K+C); scalar unstacked
        """The raw corewise (coordinate) dot of two tangents' variations -- **not** the HS inner product.

        Computed on the **real (masked)** content, vectorized over the stack: returns an array of shape
        :py:attr:`stack_shape` (``K + C``), one dot per stacked tangent (a scalar when unstacked). The
        same-tangent-space precondition is checked. It equals Hilbert-Schmidt only on an orthonormal,
        gauged frame; for that semantic use the (forthcoming) manifold geometry's ``inner``.
        """
        self._check_same_tangent_space(other)
        return _variations_stack_dot(self.variations.apply_masks(), other.variations.apply_masks())

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
        return UT3Tangent(self.basis, scaled)

    def allclose(
            self,
            other:  'UT3Tangent',  # compared at the SAME base point (corewise, like __sub__)
            rtol:   float = 1e-9,
            atol:   float = 0.0,
    ) -> NDArray:  # bool array, shape = stack_shape (K+C); scalar when unstacked
        """``True`` (per stack element) if ``other`` is the same tangent vector as ``self`` at the same base.

        Checks ``||self - other|| <= atol + rtol * ||other||`` via :py:meth:`corewise_norm`, **per stacked
        element** (reduce with ``.all()`` for a single verdict). Assumes a shared base (compares corewise on
        the variations, like :py:meth:`__sub__`); for tangents at different bases, compare dense."""
        dn = (self - other).corewise_norm()
        rn = other.corewise_norm()
        return dn <= atol + rtol * rn

    # ------------------------------------------------------------- validity checkers (delegate to UT3Basis)
    @ft.cached_property
    def minimal_ranks(self):
        """Structural minimal ranks of this tangent's base point, **per base-stack element** (the ranks
        vary across the stack). See :py:meth:`UT3Basis.minimal_ranks`."""
        return self.basis.minimal_ranks

    @ft.cached_property
    def tangent_space_dimension(self):  # int (unstacked) or int array of shape = base stack C
        """Dimension of the tangent space at this base point (= the fixed-rank manifold dimension),
        **per base-stack element** ``C`` (the ``K`` vectors share the base, hence the same tangent space).

        An ``int`` when unstacked (``C == ()``), else an int array of shape ``C``. Computed from the
        structurally-minimal ranks (gauge already quotiented); see
        :py:func:`~t3toolbox.backend.ranks.compute_manifold_dim`. Masks are host numpy, so the ranks --
        and this dimension -- are host quantities (a small loop over the base stack, computed once)."""
        up = np.asarray(self.basis.up_ranks)        # (d,)   + C
        left = np.asarray(self.basis.left_ranks)    # (d+1,) + C
        C = self.base_stack_shape

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
    def has_minimal_ranks(self) -> NDArray:  # bool array, shape = base stack C (scalar unstacked)
        """True (per base-stack element) if this tangent's basis has **structurally** minimal ranks. See
        :py:meth:`UT3Basis.has_minimal_ranks`. Minimal rank is *not* a correctness precondition for the
        tangent ops (see the contract catalog); for the numerical check see
        :py:meth:`has_numerically_minimal_ranks`. Not enforced at construction."""
        return self.basis.has_minimal_ranks

    def has_numerically_minimal_ranks(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = base stack C
        """True (per base-stack element) if this tangent's basis is **numerically** minimal. See
        :py:meth:`UT3Basis.has_numerically_minimal_ranks` (orthogonal + structurally-minimal, no SVD)."""
        return self.basis.has_numerically_minimal_ranks(atol=atol)

    def is_orthogonal(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = base stack C (scalar unstacked)
        """True (per base-stack element) if this tangent's basis is orthogonal. See
        :py:meth:`UT3Basis.is_orthogonal`. Reduce with ``.all()`` for a single verdict."""
        return self.basis.is_orthogonal(atol=atol)

    @ft.cached_property
    def gauge_residual(self) -> NDArray:  # shape = variation stack K+C (scalar/0-d when unstacked); cached
        """Max absolute gauge-condition violation, **per stack element** (shape = variation stack ``K+C``;
        atol-independent; **cached**). The expensive part of :py:meth:`is_gauged` -- a fixed tangent reused
        across an inner loop (e.g. the safe-mode GAUGE precondition of the manifold inner product) is
        contracted once. See :py:func:`~t3toolbox.backend.ubv_tangent_operations.gauge_residual`."""
        return ubv_tangent_operations.gauge_residual(self.basis.data, self.variations.data)

    def is_gauged(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = variation stack K+C (scalar unstacked)
        """True (per stack element) if the variations are gauged w.r.t. the basis.

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
        :py:func:`~t3toolbox.backend.ubv_tangent_operations.tangent_to_ut3` (Appendix A.3.1)."""
        tk, tt, shape, (tucker_mask, tt_mask) = ubv_tangent_operations.tangent_to_ut3(
            self.basis.data, self.variations.data, include_shift=include_shift)
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
        analog of pairing :py:meth:`UT3Basis.to_t3basis` with :py:meth:`UT3Variations.to_t3variations`).

        Unstacked: one :py:class:`T3Tangent`. Stacked: a nested tree (shaped ``K + C``) of *unstacked*
        :py:class:`T3Tangent` s -- a single ragged stacked tangent cannot carry varying-``C`` ranks, so the
        truthful answer is one ragged tangent per element."""
        if not self.stack_shape:
            return t3m.T3Tangent(self.basis.to_t3basis(), self.variations.to_t3variations())
        sub = self.unstack_tangents() if self.tangent_stack_shape else self.unstack_basis()
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
        :py:meth:`to_t3tangent` on a single tangent). The basis and variations are padded to the **same**
        dims (derived from the basis if not given) so the masks come out consistent."""
        b = tangent.basis
        N  = max(b.shape)            if N  is None else N
        nU = int(max(b.up_ranks))    if nU is None else nU
        nD = int(max(b.down_ranks))  if nD is None else nD
        rL = int(max(b.left_ranks))  if rL is None else rL
        rR = int(max(b.right_ranks)) if rR is None else rR
        pad = dict(N=N, nU=nU, nD=nD, rL=rL, rR=rR)
        return UT3Tangent(ubv.UT3Basis.from_t3basis(b, **pad),
                          ubv.UT3Variations.from_t3variations(tangent.variations, **pad))

    def reverse(self) -> 'UT3Tangent':
        """Reverse the mode order of this tangent (reverses both the basis and the variations).

        Will commute with :py:meth:`to_dense` (the dense tangent's mode axes are reversed) once the
        doubled-rank conversion lands -- lets you reverse a T3 and its derived tangent without
        recomputing the orthogonal representation."""
        return UT3Tangent(self.basis.reverse(), self.variations.reverse())

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
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.base(x))
        >>> ww = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))   # probe stack W=(2,)
        >>> zz = v.probe(ww)
        >>> print(zz[0].shape)                                     # W + K + C + (N0,) = (2,) + () + () + (10,)
        (2, 10)
        >>> print(bool(max(float(np.linalg.norm(a - b))
        ...                for a, b in zip(zz, t3p.probe_dense(ww, v.to_dense()))) < 1e-9))   # dense reference
        True
        """
        return ubv_sampling.ut3tangent_probe(ww, self.basis.data, self.variations.data)

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
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.base(x))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> print(bool(abs(float(v.apply(ww)) - float(np.einsum('ijk,i,j,k->', v.to_dense(), *ww))) < 1e-9))
        True
        """
        return ubv_sampling.ut3tangent_apply(ww, self.basis.data, self.variations.data)

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
        >>> v = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.base(x))
        >>> print(bool(abs(float(v.entries((3, 5, 7))) - float(v.to_dense()[3, 5, 7])) < 1e-9))
        True
        """
        return ubv_sampling.ut3tangent_entries(index, self.basis.data, self.variations.data)

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
        :py:meth:`~t3toolbox.manifold.T3Tangent.probe_derivatives`."""
        probe_derivatives.check_perturbation_vectors(ww, pp)
        return ubv_sampling.ut3tangent_probe_derivatives(ww, pp, self.basis.data, self.variations.data, order)

    def apply_derivatives(
            self,
            ww:     typ.Sequence[NDArray],  # apply vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                    # highest derivative order
    ) -> NDArray:                           # scalar jets, shape=(order+1,)+W+K+C
        """Symmetric all-modes apply derivatives (derivative twin of :py:meth:`apply`; the bare ``𝒥``, a
        scalar jet per stack element). Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.apply_derivatives`."""
        probe_derivatives.check_perturbation_vectors(ww, pp)
        return ubv_sampling.ut3tangent_apply_derivatives(ww, pp, self.basis.data, self.variations.data, order)

    def entries_derivatives(
            self,
            index:  NDArray,                # int, shape=(d,)+W -- the grid points
            pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                    # highest derivative order
    ) -> NDArray:                           # scalar jets, shape=(order+1,)+W+K+C
        """Symmetric entry derivatives at ``index`` in direction ``P`` (derivative twin of :py:meth:`entries`;
        the bare ``𝒥``). ``index`` and ``pp`` share ``W``. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.entries_derivatives`."""
        probe_derivatives.check_perturbation_index(index, pp, self.shape)
        return ubv_sampling.ut3tangent_entries_derivatives(index, pp, self.basis.data, self.variations.data, order)

    @staticmethod
    def probe_transpose(
            ztildes,                   # probe residuals, len=d, ith elm_shape=W+K+C+(Ni,)
            ww,                        # probe vectors,   len=d, ith elm_shape=W+(Ni,)
            basis:  ubv.UT3Basis,      # the orthogonal frame the tangent attaches at
            sum_over_probes:  bool = False,   # True: sum the probe stack W (Gauss-Newton 𝒥ᵀr); False: keep it
    ) -> 'UT3Tangent':  # tangent stack W+K (sum_over_probes=False) or K (True); base stack C
        """Apply the transpose ``𝒥ᵀ`` of the probe map to residuals; returns a :py:class:`UT3Tangent` at
        ``basis``. The adjoint of :py:meth:`probe`. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.probe_transpose`: the residuals live in the forward probe
        space (``W+K+C``); ``sum_over_probes=False`` keeps the probe stack ``W`` as the result's tangent
        stack (``W+K``), ``True`` sums it (``K``, the Gauss-Newton ``𝒥ᵀr``). The bare ``𝒥ᵀ`` (no gauge
        projector ``Π``)."""
        vd = ubv_sampling.ut3tangent_probe_transpose(
            ztildes, ww, basis.data, sum_over_probes=sum_over_probes)
        return UT3Tangent(basis, _ut3variations_from_data(vd))

    @staticmethod
    def apply_transpose(
            c,                         # residual, shape=W+K+C (a scalar per stack element)
            ww,                        # apply vectors, len=d, ith elm_shape=W+(Ni,)
            basis:  ubv.UT3Basis,
            sum_over_probes:  bool = False,
    ) -> 'UT3Tangent':  # tangent stack W+K (sum_over_probes=False) or K (True); base stack C
        """Apply the transpose ``𝒥ᵀ`` of :py:meth:`apply` -- back-project a residual ``c`` into a tangent
        at ``basis``. The adjoint of :py:meth:`apply`. Uniform mirror of
        :py:meth:`~t3toolbox.manifold.T3Tangent.apply_transpose`; ``sum_over_probes`` as in
        :py:meth:`probe_transpose`. The bare ``𝒥ᵀ`` (no gauge projector ``Π``)."""
        vd = ubv_sampling.ut3tangent_apply_transpose(
            c, ww, basis.data, sum_over_probes=sum_over_probes)
        return UT3Tangent(basis, _ut3variations_from_data(vd))

    @staticmethod
    def entries_transpose(
            c,                         # residual, shape=W+K+C
            index,                     # int, shape=(d,)+W (the indices whose entries c weights)
            basis:  ubv.UT3Basis,
            sum_over_probes:  bool = False,
    ) -> 'UT3Tangent':  # tangent stack W+K (sum_over_probes=False) or K (True); base stack C
        """Apply the transpose ``𝒥ᵀ`` of :py:meth:`entries` -- scatter ``c`` at ``index`` into a tangent at
        ``basis``. The adjoint of :py:meth:`entries` (= :py:meth:`apply_transpose` with one-hot vectors).
        Uniform mirror of :py:meth:`~t3toolbox.manifold.T3Tangent.entries_transpose`. The bare ``𝒥ᵀ``."""
        vd = ubv_sampling.ut3tangent_entries_transpose(
            c, index, basis.data, sum_over_probes=sum_over_probes)
        return UT3Tangent(basis, _ut3variations_from_data(vd))

    def sum_tangents(self, axis=None) -> 'UT3Tangent':
        """Sum over the tangent stack ``K`` (a batch of tangents at the shared base) into one tangent.

        Corewise (= the tensor sum, by linearity); the base stack ``C`` is preserved. ``axis`` indexes
        within ``K`` (default: the whole tangent stack). The masks OR over the summed axes -- a no-op for
        a single-base ``K`` stack (constant masks), but the correct reduction in general."""
        vd = ubv_tangent_operations.sum_tangent_stack(
            self.variations.data, len(self.tangent_stack_shape), axis)
        return UT3Tangent(self.basis, _ut3variations_from_data(vd))

    # ------------------------------------------------------------- constructors
    @staticmethod
    def zeros(
            basis:        ubv.UT3Basis,
            stack_shape:  typ.Tuple[int, ...] = (),  # extra tangent stack K (a batch of tangents)
    ) -> 'UT3Tangent':
        """Zero tangent vector at a given basis (numpy/jax matching the basis).

        ``stack_shape`` is the extra *outer* tangent stack ``K`` (a batch of tangents sharing this base);
        the variation supercores are stacked ``(d,) + K + C + (core,)``, and the variation masks are the
        base's gauge masks replicated (constant) along ``K``. Default ``K = ()``."""
        K = tuple(stack_shape)
        gauge_masks = ubv.UT3Variations._variation_masks_of(basis)         # gauge-shifted masks, stack C
        masks = _broadcast_variation_masks_over_K(gauge_masks, K)          # -> stack K + C
        variations = ubv.UT3Variations.zeros(
            basis.uniform_variation_shapes, basis.shape, stack_shape=K + basis.stack_shape,
            masks=masks, use_jax=basis.contains_jax)
        return UT3Tangent(basis, variations)

    @staticmethod
    def unit(
            basis:  ubv.UT3Basis,
            index:  typ.Tuple[bool, int, typ.Sequence[int]],  # (use_tt_coordinate, i, within_index)
    ) -> 'UT3Tangent':
        """Canonical unit tangent at ``basis``: variations zero except a single core entry.

        ``index = (use_tt_coordinate, i, within_index)`` (see :py:meth:`UT3Variations.unit`). These units
        are the standard basis of the variation supercores -- an overcomplete, non-ambient-orthogonal
        generating set of the tangent space, not an orthonormal basis (gauge it yourself if needed)."""
        gauge_masks = ubv.UT3Variations._variation_masks_of(basis)         # gauge masks at C (no K)
        variations = ubv.UT3Variations.unit(
            basis.uniform_variation_shapes, basis.shape, index,
            stack_shape=basis.stack_shape, masks=gauge_masks, use_jax=basis.contains_jax)
        return UT3Tangent(basis, variations)

    @staticmethod
    def zeros_like(tangent: 'UT3Tangent') -> 'UT3Tangent':
        """Zero tangent at ``tangent``'s base, with ``tangent``'s tangent stack ``K``."""
        return UT3Tangent.zeros(tangent.basis, stack_shape=tangent.tangent_stack_shape)

    # ------------------------------------------------------------- stack/unstack (tree <-> stacked tangent)
    def unstack_tangents(self):
        """Unstack over the tangent stack ``K``: a ``K``-shaped tree of tangents sharing this base.

        Decomposes the batch of tangent *directions*. Each leaf is a :py:class:`UT3Tangent` with
        ``tangent_stack_shape == ()`` and this tangent's ``base_stack_shape`` -- and, because the base is
        shared across ``K``, every leaf holds the **same** :py:class:`UT3Basis` object, so the leaves live
        in one tangent space. Inverse of :py:meth:`stack_tangents`."""
        variations_tree = ubv_tangent_operations.unstack_tangent_stack(self.basis.data, self.variations.data)
        return stacking.apply_func_to_leaf_subtrees(
            variations_tree,
            lambda vd: UT3Tangent(self.basis, _ut3variations_from_data(vd)),   # SAME basis object (shared)
            ubv_operations.ubv_leaf_structure(self.d, 2))

    def unstack_basis(self):
        """Unstack over the base stack ``C``: a ``C``-shaped tree of single-base-point tangents.

        Decomposes over base *points*. Each leaf is a :py:class:`UT3Tangent` with ``base_stack_shape == ()``
        and this tangent's ``tangent_stack_shape``; the leaves sit at **different** base points (different
        tangent spaces, not mutually linear-algebra compatible) and may have **different ranks** (the
        varying-``C`` rank-sweep case). Inverse of :py:meth:`stack_basis`."""
        paired_tree = ubv_tangent_operations.unstack_base_stack(self.basis.data, self.variations.data)
        leaf_structure = (ubv_operations.ubv_leaf_structure(self.d, 4),    # a basis_data leaf
                          ubv_operations.ubv_leaf_structure(self.d, 2))    # a variations_data leaf
        return stacking.apply_func_to_leaf_subtrees(
            paired_tree,
            lambda bv: UT3Tangent(_ut3basis_from_data(bv[0]), _ut3variations_from_data(bv[1])),
            leaf_structure)

    @staticmethod
    def stack_tangents(tree) -> 'UT3Tangent':
        """Stack a ``K``-shaped tree of tangents (sharing one base) into a tangent-stacked UT3Tangent.

        Inverse of :py:meth:`unstack_tangents`. Requires every leaf to be at the **same frame** (the
        numerical same-frame check, as in :py:meth:`__add__`): the tangents being stacked must live in the
        same tangent space. The first leaf's base is reused; the variations stack over the new outer
        tangent stack ``K``."""
        leaves = _flatten_tangents(tree)
        base = leaves[0].basis
        for t in leaves[1:]:
            if not (t.basis is base or safety.frames_equal_or_skip(t.basis.data[:4], base.data[:4])):
                raise ValueError(
                    'stack_tangents requires every tangent to be at the same frame -- they must live in the '
                    'same tangent space. To stack tangents at *different* base points, use stack_basis. '
                    '(Run inside safety.unsafe() to skip this numerical check.)')
        variations_tree = stacking.apply_func_to_leaf_subtrees(tree, lambda t: t.variations.data, None)
        vd = ubv_tangent_operations.stack_tangent_stack(variations_tree)
        return UT3Tangent(base, _ut3variations_from_data(vd))

    @staticmethod
    def stack_basis(tree) -> 'UT3Tangent':
        """Stack a ``C``-shaped tree of single-base-point tangents into a base-stacked UT3Tangent.

        Inverse of :py:meth:`unstack_basis`. The leaves sit at **different** base points, so no shared-base
        identity is required; they must share the padded dims ``(d, N, nU, nD, rL, rR)``, the real mode
        ``shape``, and the tangent stack ``K`` -- but the **ranks/masks MAY differ across the base stack
        ``C``** (varying-rank stacks are supported). The bases stack over the inner base stack ``C`` (so the
        variation stack becomes ``K + C``)."""
        leaves = _flatten_tangents(tree)
        key = lambda t: (t.basis.uniform_structure[:6], t.tangent_stack_shape, t.shape)
        k0 = key(leaves[0])
        for t in leaves[1:]:
            if key(t) != k0:
                raise ValueError(
                    'stack_basis requires all tangents to share the padded dims (d, N, nU, nD, rL, rR), the '
                    'mode shape, and the tangent stack K (only the base point -- and its ranks/masks -- may '
                    'differ across the base stack C).')
        paired_tree = stacking.apply_func_to_leaf_subtrees(
            tree, lambda t: (t.basis.data, t.variations.data), None)
        basis_data, variations_data = ubv_tangent_operations.stack_base_stack(paired_tree)
        return UT3Tangent(_ut3basis_from_data(basis_data), _ut3variations_from_data(variations_data))


if has_jax:
    import jax

    # Register UT3Tangent as a jax pytree with the basis as a LEAF: both the basis and the variations are
    # children (no aux_data at this level -- each carries its own static masks aux internally). Mirrors the
    # ragged T3Tangent: the base flows as ordinary traced data, so a tangent crossing a jit boundary does
    # NOT recompile when the base changes; the same-tangent-space guard is the NUMERICAL same-frame check
    # (safety.frames_equal_or_skip, safe-mode + eager-only), which survives a jit round-trip and skips
    # under a trace. (The masks ride inside UT3Basis / UT3Variations as value-hashed aux.)
    jax.tree_util.register_pytree_node(
        UT3Tangent,
        lambda x: ((x.basis, x.variations), None),
        lambda aux, children: UT3Tangent(children[0], children[1]),
    )


# ============================================================================================== geometries
# The uniform mirror of t3m.ManifoldGeometry / CorewiseGeometry (manifold.py). Stateless bundles of the
# chart-level choices (frame / gauge projection / retraction + metric); the point lives in the caller. Each
# method delegates to the 3b backend (ubv_tangent_operations) and the 2c UT3Basis / UT3Variations, behind
# the per-element safe-mode preconditions (.all() over the per-stack-element checkers, from 2c-G). See the
# ragged docstrings for the math (Section 6 + Appendix A.3 of Alger et al. 2026, arXiv:2603.21141).

def _ut3_from_data(data) -> ut3.UniformTuckerTensorTrain:  # data = (tk_sc, tt_sc, shape, (tucker_mask, tt_mask))
    return ut3.UniformTuckerTensorTrain(data[0], data[1], data[2], ut3.UT3Masks(*data[3]))


def _require_orthogonal_frame(basis: ubv.UT3Basis, who: str) -> None:
    """Safe-mode ORTH precondition (uniform mirror of :py:func:`t3toolbox.manifold._require_orthogonal_frame`).

    ``who`` (a manifold geometry op) requires an orthonormal frame **on every base-stack element**; in safe
    mode a non-orthogonal element raises. Skipped under ``safety.unsafe()`` / a jax trace. The gating helpers
    are fed the four supercores (``basis.data[:4]``) -- not the full ``.data``, whose trailing ``shape`` int
    tuple and mask holder carry no float content -- and the orthogonality check itself is mask-aware (it
    tests the real, masked content; the contract is ``to_t3basis(basis).is_orthogonal()``). ORTH (not
    minimal rank) is the only numerical precondition for the manifold projections / retraction (see
    ``docs/numerical_contract_catalog.md``)."""
    if safety.checks_active(basis.data[:4]):
        atol = safety.effective_rtol(basis.data[:4])
        safety.require(
            basis.is_orthogonal(atol=atol).all(),     # per-element check -> require ALL base-stack elements orthogonal
            '{} requires an orthogonal frame (the manifold geometry needs an orthonormal base to be the '
            'Hilbert-Schmidt-orthogonal projection). Build the base with UniformManifoldGeometry.base / '
            'UT3Basis.random_orthogonal, or run in unsafe mode (safety.unsafe()).'.format(who))


def _randn_variations_at(
        basis:  ubv.UT3Basis,
        K:      typ.Tuple[int, ...],   # extra tangent stack K (a batch of directions)
) -> ubv.UT3Variations:  # raw N(0,1) variations at basis (ungauged), gauge masks broadcast over K
    """Raw i.i.d. ``N(0, 1)`` variations at ``basis`` with the base's gauge-shifted rank masks replicated
    (constant) along the tangent stack ``K`` -- the natural (corewise) Gaussian on the variation supercores,
    before any gauge projection. Mirrors :py:meth:`UT3Tangent.zeros`'s mask derivation with a randn fill (the
    masks are the rank structure, identical for both geometries; only whether you gauge afterwards differs)."""
    K = tuple(K)
    gauge_masks = ubv.UT3Variations._variation_masks_of(basis)        # gauge-shifted masks, stack C
    masks = _broadcast_variation_masks_over_K(gauge_masks, K)         # -> stack K + C
    return ubv.UT3Variations.randn(
        basis.uniform_variation_shapes, basis.shape, stack_shape=K + basis.stack_shape,
        masks=masks, use_jax=basis.contains_jax)


class UniformManifoldGeometry:
    """The Riemannian geometry of the uniform fixed-rank Tucker tensor train manifold (uniform mirror of
    :py:class:`~t3toolbox.manifold.ManifoldGeometry`).

    A stateless bundle of the chart-level choices that distinguish the manifold from the over-parametrized
    corewise geometry -- the orthonormal frame (:py:meth:`base`), the gauge projection ``Pi``
    (:py:meth:`project`), and the manifold retraction (:py:meth:`retract`) -- plus the ambient projection and
    transport. Tangents live in ``T_xM`` with an orthonormal, gauged frame; the metric is Hilbert-Schmidt.
    All methods are **per-element-mask-aware** (varying ranks across the base stack ``C`` are supported) and
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
    >>> base = ut3m.UNIFORM_MANIFOLD.base(x)            # orthonormal frame at x
    >>> print(bool(base.is_orthogonal().all()))
    True
    >>> v = ut3m.UNIFORM_MANIFOLD.randn(base)           # a standard Gaussian on T_xM (gauged)
    >>> print(bool(v.is_gauged().all()))
    True
    >>> y = ut3m.UNIFORM_MANIFOLD.retract(ut3m.UT3Tangent.zeros(base))   # retract the zero tangent == the base point
    >>> print(bool(np.allclose(y.to_dense(), x.to_dense())))
    True
    """

    def base(
            self,
            x:  ut3.UniformTuckerTensorTrain,
    ) -> ubv.UT3Basis:  # orthonormal frame at x
        """The orthonormal frame at ``x`` (the frame part of ``ut3_orthogonal_representations``)."""
        return ubv.UT3Basis.from_ut3(x)

    def randn(
            self,
            basis:        ubv.UT3Basis,
            stack_shape:  typ.Tuple[int, ...] = (),   # extra tangent stack K (a batch of tangents)
    ) -> UT3Tangent:  # gauged random tangent at basis
        """Random tangent at ``basis``: a standard Gaussian on the tangent space ``T_xM``.

        Raw i.i.d. ``N(0, 1)`` variation supercores, then the gauge projection :py:meth:`project` (``Pi``).
        For an orthogonal, minimal-rank ``basis`` this is the standard Gaussian on ``T_xM``. ``stack_shape``
        is the extra outer tangent stack ``K`` (default ``()``). Inherits :py:meth:`project`'s safe-mode ORTH
        precondition (a non-orthogonal ``basis`` raises; skipped under ``safety.unsafe()`` / a jax trace)."""
        return self.project(UT3Tangent(basis, _randn_variations_at(basis, stack_shape)))

    def random_orthogonal(
            self,
            shape:                  typ.Sequence[int],         # (N0,...,N(d-1))
            tucker_ranks,                                      # int | len-d seq | (d,)+C array (the variety)
            tt_ranks,                                          # int | len-(d+1) seq | (d+1,)+C array
            stack_shape:            typ.Tuple[int, ...] = (),  # base stack C (random base points)
            tangent_stack_shape:    typ.Tuple[int, ...] = (),  # tangent stack K
            use_jax:                bool = False,
    ) -> UT3Tangent:  # gauged random tangent at a random orthogonal base point
        """A gauged random tangent at a random orthogonal base point (random direction, random base)."""
        base = ubv.UT3Basis.random_orthogonal(shape, tucker_ranks, tt_ranks,
                                              stack_shape=stack_shape, use_jax=use_jax)
        return self.randn(base, stack_shape=tangent_stack_shape)

    def randn_like(
            self,
            tangent:    UT3Tangent,   # reuse its base + tangent stack K
    ) -> UT3Tangent:  # gauged random tangent at tangent's base
        """A gauged random tangent at ``tangent``'s base, with ``tangent``'s tangent stack ``K``."""
        return self.randn(tangent.basis, stack_shape=tangent.tangent_stack_shape)

    def project(
            self,
            v:  UT3Tangent,
    ) -> UT3Tangent:  # gauged tangent at v's basis (a DIFFERENT vector)
        """The gauge projection ``Pi``: raw cotangent -> Riemannian gradient (orthogonal gauge).

        Orthogonally projects ``v``'s variations onto the gauged tangent space (conditions (48)-(49),
        Appendix A.3). Represents a DIFFERENT tangent vector than ``v`` -- the map that turns a bare adjoint
        ``J^T r`` into the Riemannian gradient. For the vector-preserving fix see :py:meth:`project_oblique`.
        **Safe mode** requires ``v``'s frame orthogonal (raises otherwise; skipped under ``safety.unsafe()`` /
        a jax trace)."""
        _require_orthogonal_frame(v.basis, 'UniformManifoldGeometry.project')
        vd = ubv_tangent_operations.orthogonal_gauge_projection(v.basis.data, v.variations.data)
        return UT3Tangent(v.basis, _ut3variations_from_data(vd))

    def project_oblique(
            self,
            v:  UT3Tangent,
    ) -> UT3Tangent:  # gauged tangent at v's basis (the SAME vector)
        """Gauge ``v``'s variations while preserving the represented tangent vector (oblique projection).

        Returns a tangent at the same basis representing the SAME vector as ``v`` but gauged, so that on an
        orthogonal minimal-rank basis :py:meth:`inner` / :py:meth:`norm` give the true Hilbert-Schmidt values.
        **Safe mode** requires ``v``'s frame orthogonal (raises otherwise); skipped under ``safety.unsafe()``
        / a jax trace."""
        _require_orthogonal_frame(v.basis, 'UniformManifoldGeometry.project_oblique')
        vd = ubv_tangent_operations.oblique_gauge_projection(v.basis.data, v.variations.data)
        return UT3Tangent(v.basis, _ut3variations_from_data(vd))

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
        if safety.checks_active(t1.basis.data[:4], t1.variations.data[:2], t2.variations.data[:2]):
            atol = safety.effective_rtol(t1.basis.data[:4], t1.variations.data[:2], t2.variations.data[:2])
            safety.require(t1.basis.is_orthogonal(atol=atol).all(),
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
        if safety.checks_active(t.basis.data[:4], t.variations.data[:2]):
            atol = safety.effective_rtol(t.basis.data[:4], t.variations.data[:2])
            safety.require(t.basis.is_orthogonal(atol=atol).all(),
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
    ) -> ut3.UniformTuckerTensorTrain:  # retracted point on M, at p's base ranks
        """Retract the step ``p`` to the manifold: shifted doubled-rank embedding, truncated to base ranks.

        Forms ``base point + p`` and truncates back to ``p``'s base ranks via the implicit uniform T3-SVD
        (Algorithm 10). The current point is carried by ``p.basis``, so no separate point argument is needed.
        **Safe mode** requires ``p``'s frame orthogonal (raises otherwise; skipped under ``safety.unsafe()`` /
        a jax trace). ORTH only -- retract is gauge-invariant; minimal rank is a documented caveat."""
        _require_orthogonal_frame(p.basis, 'UniformManifoldGeometry.retract')
        return _ut3_from_data(ubv_tangent_operations.retract(p.basis.data, p.variations.data))

    def project_ambient(
            self,
            basis:  ubv.UT3Basis,                     # orthogonal base point of the tangent space
            grad:   ut3.UniformTuckerTensorTrain,     # ambient gradient as a uniform Tucker tensor train
    ) -> UT3Tangent:  # the Riemannian gradient (gauged projection of grad) at basis
        """Project an ambient gradient onto ``T_xM`` -- the Riemannian gradient.

        ``grad`` is the Euclidean/ambient gradient as a :py:class:`UniformTuckerTensorTrain`. Returns the
        gauged tangent ``P_T(grad)`` (the residual ``grad - P_T(grad)`` is orthogonal to the tangent space).
        Requires an **orthogonal** ``basis`` (minimal rank not required); enforced in safe mode (skipped under
        ``safety.unsafe()`` / a jax trace).

        Unlike the ragged :py:meth:`~t3toolbox.manifold.ManifoldGeometry.project_ambient`, the uniform layer
        has **no native dense-array path** (the uniform layer is a performance layer; working dense here
        defeats its purpose). For a dense gradient, go via the ragged geometry and the cross-layer converters
        (:py:meth:`UT3Basis.to_t3basis` -> ``MANIFOLD.project_ambient`` -> :py:meth:`UT3Tangent.from_t3tangent`)."""
        if not isinstance(grad, ut3.UniformTuckerTensorTrain):
            raise TypeError(
                'UniformManifoldGeometry.project_ambient accepts a UniformTuckerTensorTrain gradient only; '
                'the uniform layer has no native dense-array projection. For a dense gradient, project via '
                'the ragged ManifoldGeometry.project_ambient and the cross-layer converters '
                '(UT3Basis.to_t3basis / UT3Tangent.from_t3tangent). Got %r.' % (type(grad).__name__,))
        _require_orthogonal_frame(basis, 'UniformManifoldGeometry.project_ambient')
        vd = ubv_tangent_operations.project_ut3_onto_tangent_space(basis.data, grad.data)
        return UT3Tangent(basis, _ut3variations_from_data(vd))

    def transport(
            self,
            v:          UT3Tangent,
            new_basis:  ubv.UT3Basis,
    ) -> UT3Tangent:  # v transported to the tangent space at new_basis
        """Projective vector transport of ``v`` to the tangent space at ``new_basis``.

        Re-projects ``v`` (as an ambient tensor via its doubled-rank :py:meth:`UT3Tangent.to_ut3`)
        orthogonally onto the tangent space at ``new_basis`` -- the cheap, standard choice for fixed-rank
        Riemannian optimization (not parallel transport). Inherits :py:meth:`project_ambient`'s safe-mode ORTH
        precondition on ``new_basis``."""
        return self.project_ambient(new_basis, v.to_ut3())


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
    >>> base = ut3m.UNIFORM_COREWISE.base(x)            # the (U, G, G, G) frame
    >>> v = ut3m.UNIFORM_COREWISE.randn(base)           # raw randn cores (no gauge)
    >>> print(bool(v.is_gauged().all()))
    False
    >>> y = ut3m.UNIFORM_COREWISE.retract(ut3m.UT3Tangent.zeros(base))   # additive: cores += step (zero -> the point)
    >>> print(bool(np.allclose(y.to_dense(), x.to_dense())))
    True
    """

    def base(
            self,
            x:  ut3.UniformTuckerTensorTrain,
    ) -> ubv.UT3Basis:  # the (U, G, G, G) non-orthonormal frame at x
        """The core-parameter frame at ``x``: ``(U, G, G, G)`` (down/left/right all the TT supercore ``G``).

        The mask analog of the cores: ``up = down = tucker_edge_mask`` (the U-G Tucker edges) and
        ``left = right = tt_edge_mask`` (the G-G TT bonds, full length ``d+1``) -- the Section 6.3
        substitution ``(P, Q, O) -> G`` gives all three non-up slots the single core ``G``'s rank structure,
        with no boundary slicing (the ``[:-1]`` / ``[1:]`` gauge shift is a *variations*, not a basis,
        thing)."""
        tk_sc, tt_sc, shape, (tucker_mask, tt_mask) = x.data
        masks = ubv.UT3BasisMasks(tucker_mask, tucker_mask, tt_mask, tt_mask)
        return ubv.UT3Basis(tk_sc, tt_sc, tt_sc, tt_sc, shape, masks)

    def randn(
            self,
            basis:        ubv.UT3Basis,
            stack_shape:  typ.Tuple[int, ...] = (),   # extra tangent stack K
    ) -> UT3Tangent:  # raw random tangent at basis (ungauged)
        """Random tangent at ``basis``: raw i.i.d. ``N(0, 1)`` variation supercores (the natural corewise
        Gaussian; no gauge projection). ``stack_shape`` is the extra outer tangent stack ``K``."""
        return UT3Tangent(basis, _randn_variations_at(basis, stack_shape))

    def randn_like(
            self,
            tangent:    UT3Tangent,
    ) -> UT3Tangent:  # raw random tangent at tangent's base
        """A raw random tangent at ``tangent``'s base, with ``tangent``'s tangent stack ``K``."""
        return self.randn(tangent.basis, stack_shape=tangent.tangent_stack_shape)

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
            p:  UT3Tangent,   # step (a corewise tangent at base = (U, G, G, G))
    ) -> ut3.UniformTuckerTensorTrain:  # additive retraction: cores += p
        """Additive retraction: add the variation supercores to the point's cores (``cores += p``).

        Recovers the point ``(U, G)`` from ``p.basis`` (which :py:meth:`base` built as ``(U, G, G, G)``) and
        adds the variations. ``p`` must be a corewise tangent (a basis from :py:meth:`base`). See
        :py:func:`~t3toolbox.backend.ubv_tangent_operations.corewise_retract`."""
        return _ut3_from_data(ubv_tangent_operations.corewise_retract(p.basis.data, p.variations.data))


UNIFORM_MANIFOLD = UniformManifoldGeometry()   # the uniform fixed-rank Riemannian geometry (gauge Pi, manifold retraction)
UNIFORM_COREWISE = UniformCorewiseGeometry()   # the uniform core-parameter Euclidean geometry (no gauge, additive retraction)
