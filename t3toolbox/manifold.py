# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
from __future__ import annotations

import math
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.corewise as cw
import t3toolbox.safety as safety
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.tangent_operations as tangent_operations
import t3toolbox.backend.probing as probing
import t3toolbox.backend.probe_derivatives as probe_derivatives
import t3toolbox.backend.ranks as ranks
from t3toolbox.backend.common import *

__all__ = [
    'T3Tangent',
    'ManifoldGeometry',
    'CorewiseGeometry',
    'MANIFOLD',
    'COREWISE',
    'manifold_dim',
]


def manifold_dim(
        s,  # structure: (shape, tucker_ranks, tt_ranks) = ((N0,...), (n0,...), (1,r1,...,1))
) -> int:  # dimension of the fixed-rank T3 manifold
    """Get the dimension of the fixed rank T3 manifold with a given structure.

    The fixed-rank Tucker tensor train manifold M_{n,r} is described in Appendix A.3 of Alger et al.
    (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> s = ((15,16,13), (9,10,8), (2,7,6,3))
    >>> mdim = t3m.manifold_dim(s)
    >>> print(mdim)
    578

    In the following more detailed example, we verify that the manifold dim
    is correct by generating an excessive number of random dense tangent vectors
    and performing an SVD on them. The number of nonzero singular values is the
    dimension of the tangent space, which is the dimension of the manifold.

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.basis_variations_format as bvf
    >>> s = ((5, 6, 3), (5, 3, 2), (2, 2, 4, 1))
    >>> mdim = t3m.manifold_dim(s)
    >>> print(mdim)
    29
    >>> base, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(*s))
    >>> tucker_shapes, tt_shapes = base.variation_shapes
    >>> n_entries = sum(int(np.prod(sh)) for sh in tucker_shapes) + sum(int(np.prod(sh)) for sh in tt_shapes)
    >>> dense_vv = np.stack([t3m.MANIFOLD.randn(base).to_dense().reshape(-1)
    ...                      for _ in range(n_entries)])
    >>> ss = np.linalg.svd(dense_vv, compute_uv=False)
    >>> print(int(np.sum(ss > 1e-9 * ss[0])))   # number of nonzero singular values == manifold_dim
    29
    """
    return ranks.compute_manifold_dim(s[0], s[1], s[2])


@dataclass(frozen=True)
class T3Tangent:
    """Tangent vector to the manifold of fixed-rank Tucker tensor trains.

    A ``T3Tangent`` bundles a :py:class:`~t3toolbox.basis_variations_format.T3Basis` (the frame at
    the base point where the tangent space is attached) with a
    :py:class:`~t3toolbox.basis_variations_format.T3Variations` (the tangent direction in that
    frame). Bundling them makes "which tangent space" a checkable property: linear algebra between
    two tangent vectors is only defined when they live in the same tangent space, which here means
    they hold the **same** ``T3Basis`` object (identity, not merely numerically-equal cores).

    The metric lives on the *geometry*, not here: :py:meth:`ManifoldGeometry.inner` / ``norm`` are the
    Hilbert-Schmidt inner product / norm (which check the orthogonal-frame + gauged preconditions in safe
    mode), and :py:meth:`CorewiseGeometry.inner` / ``norm`` are the Euclidean ones. This class exposes only
    the **raw coordinate** :py:meth:`corewise_inner` / :py:meth:`corewise_norm` (equal to HS only on an
    orthonormal, gauged, minimal-rank frame -- see :py:meth:`is_orthogonal` / :py:meth:`is_gauged` and the
    contract catalog), with no HS claim.

    A tangent vector is the sum of 2d single-core variation terms -- equation (47), Appendix A.3, of
    Alger, Christierson, Chen & Ghattas (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
    >>> base, variations = bvf.t3_orthogonal_representations(x)
    >>> v = t3m.T3Tangent(base, variations)
    >>> print(v.shape, v.stack_shape)
    (10, 11, 12) ()
    >>> print(v.is_orthogonal())   # base from t3_orthogonal_representations is orthogonal
    True
    >>> print(v.is_gauged())       # ...but those variations are not gauged
    False
    >>> w = 2.0 * v - v            # linear algebra stays in the same tangent space
    >>> print(np.linalg.norm(w.to_dense() - v.to_dense()))   # (2v - v) == v
    0.0
    """
    basis:      bvf.T3Basis
    variations: bvf.T3Variations

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate this tangent: both components well-formed and a compatible (basis, variations) pair.

        Runs ``basis.validate()`` + ``variations.validate()`` + the bv-pair compatibility check
        (:py:func:`~t3toolbox.basis_variations_format.check_bv_pair`). Structural only (shapes/ranks), so
        it is safe to run in ``__post_init__`` (which it is, on every construction) and under jit/pytree
        tracing.
        """
        self.basis.validate()
        self.variations.validate()
        bvf.check_bv_pair(self.basis, self.variations)

    def __repr__(self) -> str:
        return (f"T3Tangent(shape={self.shape}, tucker_ranks={self.basis.up_ranks}, "
                f"tt_ranks={self.basis.left_ranks}, tangent_stack={self.tangent_stack_shape}, "
                f"base_stack={self.base_stack_shape})")

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

        This is the part of the variation stack that exceeds the base stack (often empty). The
        variation cores are stacked as ``K + C + (core,)`` -- extra axes outermost, base stack inner.
        """
        full = self.variations.stack_shape
        return full[:len(full) - len(self.base_stack_shape)]

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int, ...]:
        """Full stack ``K + C`` (``tangent_stack_shape + base_stack_shape``), outer-to-inner."""
        return self.variations.stack_shape

    @ft.cached_property
    def structure(self):
        return self.basis.structure

    @ft.cached_property
    def data(self) -> typ.Tuple[bvf.T3Basis, bvf.T3Variations]:
        return self.basis, self.variations

    def to_jax(self) -> 'T3Tangent':
        """Copy with basis and variation cores converted to jax arrays."""
        return T3Tangent(self.basis.to_jax(), self.variations.to_jax())

    def to_numpy(self) -> 'T3Tangent':
        """Copy with basis and variation cores converted to numpy arrays."""
        return T3Tangent(self.basis.to_numpy(), self.variations.to_numpy())

    def copy(self) -> 'T3Tangent':
        """Deep copy (copies the basis and variation cores)."""
        return T3Tangent(self.basis.copy(), self.variations.copy())

    @ft.cached_property
    def contains_jax(self) -> bool:
        """True if any basis or variation core is a jax array."""
        return self.basis.contains_jax or self.variations.contains_jax

    @ft.cached_property
    def size(self) -> int:
        """Number of elements of the represented dense tangent vector (``prod(shape)``)."""
        return math.prod(self.shape)

    @ft.cached_property
    def data_size(self) -> int:
        """Number of stored core entries (size on disk): basis + variations."""
        return self.basis.data_size + self.variations.data_size

    ############################################
    ##########    Conversions    ###############
    ############################################

    def to_dense(
            self,
            include_shift:  bool = False,  # False: tangent vector v. True: base point + v.
    ) -> NDArray:  # shape=stack_shape+(N0,...,N(d-1))
        """Form the dense tensor represented by this tangent vector.

        The tangent vector is the sum of the 2d single-core-replacement terms (one per Tucker hole
        and one per TT hole). With ``include_shift=True``, the base point is added (base point + v).
        """
        return tangent_operations.tangent_to_dense(
            self.basis.data, self.variations.data, include_shift=include_shift,
        )

    def to_t3(
            self,
            include_shift:  bool = False,  # False: tangent vector v. True: base point + v.
    ) -> t3.TuckerTensorTrain:  # doubled-rank Tucker tensor train
        """Doubled-rank :py:class:`TuckerTensorTrain` representation of this tangent vector.

        The Tucker and TT ranks are (roughly) doubled. With ``include_shift=True`` the result
        represents ``base point + v`` (the standard shifted embedding used by :py:meth:`retract`).

        This is the doubled-rank representation of Appendix A.3.1 (equations (50)-(53) and Figure 20)
        in Alger et al. (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).
        """
        cores = tangent_operations.tangent_to_t3(
            self.basis.data, self.variations.data, include_shift=include_shift,
        )
        return t3.TuckerTensorTrain(*cores)

    def to_vector(self) -> NDArray:
        """Flatten this tangent's variation degrees of freedom to a 1D vector (the basis is the fixed
        point and is *not* included). The optimization interface (pairs with :py:meth:`from_vector`)."""
        return self.variations.to_vector()

    @staticmethod
    def from_vector(
            flat:                   NDArray,                   # 1D vector of variation DOFs (from to_vector)
            basis:                  bvf.T3Basis,
            tangent_stack_shape:    typ.Tuple[int, ...] = (),  # tangent stack K (default ())
    ) -> 'T3Tangent':
        """Inverse of :py:meth:`to_vector`: rebuild the tangent at ``basis`` from a 1D DOF vector.

        ``tangent_stack_shape`` is the tangent stack ``K`` (default ``()``); the variations are rebuilt
        with stack ``K + basis.stack_shape``.
        """
        variations = bvf.T3Variations.from_vector(
            flat, basis.variation_shapes, stack_shape=tuple(tangent_stack_shape) + basis.stack_shape)
        return T3Tangent(basis, variations)

    def save(self, file) -> None:
        """Save the basis + variation cores to a ``.npz`` file (load with :py:meth:`load`)."""
        families = self.basis.data + self.variations.data   # 4 basis families + 2 variation families
        save_core_families(file, families)

    @staticmethod
    def load(file, use_jax: bool = False) -> 'T3Tangent':
        """Load a tangent saved by :py:meth:`save`."""
        f = load_core_families(file)
        t = T3Tangent(bvf.T3Basis(f[0], f[1], f[2], f[3]), bvf.T3Variations(f[4], f[5]))
        return t.to_jax() if use_jax else t

    def reverse(self) -> 'T3Tangent':
        """Reverse the mode order of this tangent (reverses both the basis and the variations).

        Commutes with :py:meth:`to_dense` (the dense tangent's mode axes are reversed). Lets you reverse
        a T3 and its derived tangent without recomputing the orthogonal representation."""
        return T3Tangent(self.basis.reverse(), self.variations.reverse())

    def sum_tangents(self, axis=None) -> 'T3Tangent':
        """Sum over the tangent stack ``K`` (a batch of tangents at the shared base) into one tangent.

        Corewise (= the tensor sum, by linearity); the base stack ``C`` is preserved. ``axis`` indexes
        within ``K`` (default: the whole tangent stack).
        """
        summed = cw.corewise_stack_sum(self.variations.data, axis, len(self.tangent_stack_shape))
        return T3Tangent(self.basis, bvf.T3Variations(*summed))

    @staticmethod
    def zeros(
            basis:          bvf.T3Basis,
            stack_shape:    typ.Tuple[int, ...] = (),  # extra tangent stack K (a batch of tangents)
    ) -> 'T3Tangent':
        """Zero tangent vector at a given basis (numpy/jax matching the basis).

        ``stack_shape`` is the extra *outer* tangent stack ``K`` (a batch of tangents sharing this
        base); the variation cores are stacked as ``K + C + (core,)``. Default ``K=()``.
        """
        use_jax = tree_contains_jax(basis.data)       # match the basis's array type
        full_stack = stack_shape + basis.stack_shape  # K + C
        variations = bvf.T3Variations.zeros(basis.variation_shapes, full_stack, use_jax)
        return T3Tangent(basis, variations)

    @staticmethod
    def unit(
            basis:  bvf.T3Basis,
            index,              # (use_tt_coordinate, i, within_index); see T3Variations.unit
    ) -> 'T3Tangent':
        """Canonical unit tangent at ``basis``: variations zero except a single core entry.

        ``index = (use_tt_coordinate, i, within_index)`` (see :py:meth:`T3Variations.unit`). These units
        are the standard basis of the variation cores -- an overcomplete, non-ambient-orthogonal
        generating set of the tangent space, not an orthonormal basis (gauge it yourself if needed).
        """
        variations = bvf.T3Variations.unit(basis.variation_shapes, index,
                                            stack_shape=basis.stack_shape, use_jax=basis.contains_jax)
        return T3Tangent(basis, variations)

    @staticmethod
    def zeros_like(tangent: 'T3Tangent') -> 'T3Tangent':
        """Zero tangent at ``tangent``'s base, with ``tangent``'s tangent stack ``K``."""
        return T3Tangent.zeros(tangent.basis, stack_shape=tangent.tangent_stack_shape)

    ############################################
    ##########    Linear algebra    ############
    ############################################

    def _check_same_tangent_space(self, other: 'T3Tangent') -> None:
        if self.basis is not other.basis:
            raise ValueError(
                'Tangent vectors are in different tangent spaces.\n'
                'Linear algebra between tangent vectors requires the *same* T3Basis object '
                '(object identity, not merely numerically-equal cores).'
            )
        if self.stack_shape != other.stack_shape:
            raise ValueError(
                'Tangent vectors have different stack shapes; elementwise linear algebra requires '
                'matching stacks (same tangent stack K over the shared base stack C).\n'
                + str(self.stack_shape) + ' = self.stack_shape != other.stack_shape = ' + str(other.stack_shape)
            )

    def __add__(self, other: 'T3Tangent') -> 'T3Tangent':
        """Add tangent vectors. Requires both to share the same T3Basis object."""
        self._check_same_tangent_space(other)
        return T3Tangent(self.basis, bvf.T3Variations(*cw.corewise_add(self.variations.data, other.variations.data)))

    def __sub__(self, other: 'T3Tangent') -> 'T3Tangent':
        """Subtract tangent vectors. Requires both to share the same T3Basis object."""
        self._check_same_tangent_space(other)
        return T3Tangent(self.basis, bvf.T3Variations(*cw.corewise_sub(self.variations.data, other.variations.data)))

    def __mul__(self, scalar) -> 'T3Tangent':
        """Scale a tangent vector by a scalar."""
        return T3Tangent(self.basis, bvf.T3Variations(*cw.corewise_scale(self.variations.data, scalar)))

    __rmul__ = __mul__

    def __neg__(self) -> 'T3Tangent':
        return self * (-1.0)

    def corewise_inner(self, other: 'T3Tangent'):
        """The raw corewise (coordinate) dot of two tangents' variations -- **not** the HS inner product.

        Vectorized over the stack: returns an array of shape :py:attr:`stack_shape` (``K + C``), one dot
        per stacked tangent (a scalar when unstacked). The same-frame precondition is checked.

        This is the *coordinate* inner product; it equals Hilbert-Schmidt only on an orthonormal, gauged,
        (minimal-rank) frame. For that semantic -- with the orthogonal/gauge preconditions checked -- use
        :py:meth:`ManifoldGeometry.inner` (or :py:meth:`CorewiseGeometry.inner` for the Euclidean metric).
        The gauged identity ``<v, v'>_HS = sum_i <dU_i, dU_i'> + sum_i <dG_i, dG_i'>`` is Appendix A.3 of
        Alger et al. (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).
        """
        self._check_same_tangent_space(other)
        return cw.corewise_stack_dot(
            self.variations.data, other.variations.data, len(self.stack_shape),
        )

    def corewise_norm(self):
        """The raw corewise (coordinate) norm of the variations -- **not** the HS norm.

        Vectorized over the stack: returns an array of shape :py:attr:`stack_shape` (``K + C``), one norm
        per stacked tangent (a scalar when unstacked). Equals the Hilbert-Schmidt norm only on an
        orthonormal, gauged frame; for that semantic use :py:meth:`ManifoldGeometry.norm`.
        """
        return cw.corewise_stack_norm(self.variations.data, len(self.stack_shape))

    def normalized(self) -> 'T3Tangent':
        """Unit-norm rescaling ``self / self.corewise_norm()``, vectorized over the stack.

        Scales the variations so the result has :py:meth:`corewise_norm` 1. Each stacked tangent is scaled
        by its own norm; the base point is unchanged.
        """
        variations = bvf.T3Variations(*cw.corewise_stack_scale(self.variations.data, 1.0 / self.corewise_norm()))
        return T3Tangent(self.basis, variations)

    def allclose(
            self,
            other:  'T3Tangent',  # compared at the SAME base point (corewise, like __sub__)
            rtol:   float = 1e-9,
            atol:   float = 0.0,
    ) -> bool:
        """``True`` if ``other`` is the same tangent vector as ``self`` at the same base point.

        Checks ``||self - other|| <= atol + rtol * ||other||`` via :py:meth:`corewise_norm` (per stacked
        element, all of which must pass). Assumes a shared base point (compares corewise on the
        variations, like :py:meth:`__sub__`); for tangents at different bases compare the dense forms.
        """
        dn = (self - other).corewise_norm()
        rn = other.corewise_norm()
        return bool((dn <= atol + rtol * rn).all())

    ############################################
    ##########    Validity checkers    #########
    ############################################

    @ft.cached_property
    def minimal_ranks(self):
        """Structural minimal ranks of this tangent's base point. See :py:attr:`T3Basis.minimal_ranks`."""
        return self.basis.minimal_ranks

    @ft.cached_property
    def tangent_space_dimension(self) -> int:
        """Dimension of the tangent space at this base point (= the fixed-rank manifold dimension).

        Computed from the structurally-minimal ranks (gauge already quotiented), so it equals the true
        tangent-space dimension for a minimal-rank base point. See :py:func:`manifold_dim`.
        """
        return manifold_dim((self.shape, self.basis.up_ranks, self.basis.left_ranks))

    @ft.cached_property
    def has_minimal_ranks(self) -> bool:
        """True if this tangent's basis has **structurally** minimal ranks. See
        :py:attr:`T3Basis.has_minimal_ranks`. Minimal rank is *not* a correctness precondition for the
        verified tangent ops (see :py:meth:`T3Basis.has_minimal_ranks` / the contract catalog); for the
        numerical check see :py:meth:`has_numerically_minimal_ranks`. Not enforced at construction.
        """
        return self.basis.has_minimal_ranks

    def has_numerically_minimal_ranks(self, atol: float = 1e-9) -> bool:
        """True if this tangent's basis is **numerically** minimal. See
        :py:meth:`T3Basis.has_numerically_minimal_ranks` (orthogonal + structurally-minimal, no SVD)."""
        return self.basis.has_numerically_minimal_ranks(atol=atol)

    def is_orthogonal(self, atol: float = 1e-9) -> bool:
        """True if this tangent's basis is orthogonal. See :py:meth:`T3Basis.is_orthogonal`."""
        return self.basis.is_orthogonal(atol=atol)

    def is_gauged(self, atol: float = 1e-9) -> bool:
        """True if the variations are gauged with respect to the basis.

        Gauge conditions (needed for :py:meth:`ManifoldGeometry.inner` / :py:meth:`ManifoldGeometry.norm`
        to equal the Hilbert-Schmidt values; not enforced at construction):
            - ``einsum('...ia,...ja->...ij', U_i, V_i) = 0`` for all i (Tucker variations ⟂ U).
            - ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` for i = 0..d-2 (TT variations ⟂ L).

        These are the gauge conditions (48)-(49), Appendix A.3, of Alger et al. (2026),
        "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).
        """
        return bool(tangent_operations.gauge_residual(self.basis.data, self.variations.data) <= atol)

    ############################################
    ##########    Probing    ###################
    ############################################

    def probe(
            self,
            ww:         typ.Sequence[NDArray],  # probing vectors, len=d, elm_shape=W+(Ni,)
    ) -> typ.Sequence[NDArray]:                 # probes, len=d, elm_shape=W+K+C+(Ni,)
        """Probe this tangent vector: apply the single-sample least-squares Jacobian J^(s).

        Contracts the tangent vector with the probing vectors ``ww`` in all-but-one index, for each
        index -- the tangent analogue of :py:meth:`.TuckerTensorTrain.probe`. The probes are stacked
        ``W + K + C`` (probe stack ``W`` from ``ww`` outermost, tangent stack ``K`` next, base stack
        ``C`` innermost). ``K`` is empty unless this is a tangent-stacked (K-stacked) T3Tangent, in
        which case ``J^(s)`` is applied to each of the ``K`` tangent vectors sharing the base.

        This is the bare ``J^(s)`` (no gauge projector ``Pi``); for the Riemannian ``J = J^(s) o Pi``
        compose a gauge projection (e.g. :py:meth:`ManifoldGeometry.project`) yourself.

        See Section 6.2.2 (Algorithms 6-7) of Alger et al. (2026), "Tucker Tensor Train Taylor
        Series" (arXiv:2603.21141).

        See Also
        --------
        probe_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.manifold as t3m
        >>> import t3toolbox.backend.probing as t3p
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> base, variations = bvf.t3_orthogonal_representations(x)
        >>> v = t3m.T3Tangent(base, variations)
        >>> ww = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
        >>> zz = v.probe(ww)
        >>> print(zz[0].shape)             # W + C + (N0,) = (2,) + () + (10,)
        (2, 10)
        >>> zz2 = t3p.probe_dense(ww, v.to_dense())   # dense reference
        >>> print(bool(max(float(np.linalg.norm(a - b)) for a, b in zip(zz, zz2)) < 1e-9))
        True

        A tangent-stacked (K-stacked) tangent probes each of its ``K`` vectors, output ``W + K + C``:

        >>> vb = t3m.COREWISE.randn(base, stack_shape=(3,))
        >>> zzb = vb.probe(ww)
        >>> print(zzb[0].shape)            # W + K + C + (N0,) = (2,) + (3,) + () + (10,)
        (2, 3, 10)
        """
        # probing's base order is exactly T3Basis.data = (up, down, left, right) -- no reorder.
        # numpy/jax dispatch is inferred from the input array types inside probing.
        return probing.probe_tangent(ww, self.variations.data, self.basis.data)

    @staticmethod
    def probe_transpose(
            ztildes:            typ.Sequence[NDArray],  # probe residuals, len=d, elm_shape=W+K+C+(Ni,)
            ww:                 typ.Sequence[NDArray],  # probing vectors, len=d, elm_shape=W+(Ni,)
            basis:              bvf.T3Basis,
            sum_over_probes:    bool = False,           # True: sum the probe stack W (Gauss-Newton J^T r)
    ) -> 'T3Tangent':
        """Apply the transpose ``(J^(s))^T`` of the probe map to residuals; returns a T3Tangent at ``basis``.

        The adjoint of :py:meth:`probe`. The residuals ``ztildes`` live in the forward probe space,
        ``elm_shape = W + K + C + (Ni,)`` (probe stack ``W`` outer, optional tangent batch ``K``, base
        stack ``C`` inner -- the output space of a ``K``-stacked :py:meth:`probe`; ``K`` is empty in
        the common case). The tangent batch ``K`` is always carried to the result's tangent stack; the
        probe stack ``W`` is summed or kept per ``sum_over_probes``:

        - ``sum_over_probes=False`` (default): each probe residual becomes one tangent -- the result's
          tangent stack is ``W + K`` (base stack ``C``).
        - ``sum_over_probes=True``: the probe stack is summed -- the result's tangent stack is ``K``
          (base stack ``C``) -- the usual Gauss-Newton ``J^T r`` (a single tangent when ``K = ()``).

        ``False`` is the **primary** transpose (``W`` a passthrough stack); ``True`` is the derived
        contraction ``sum_over_probes=True == Σ_W sum_over_probes=False``. See *Batching & stacking* §11
        (``docs/batching_and_stacking.md``) for which mode to use and why.

        Bare ``(J^(s))^T`` (no gauge projector). See Section 6.2.3 (Algorithm 8) of Alger et al. (2026).

        See Also
        --------
        probe

        Examples
        --------
        Adjoint identity ``<z, J v> = <J^T z, v>`` (sum over probes):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.manifold as t3m
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> base, _ = bvf.t3_orthogonal_representations(x)
        >>> v = t3m.MANIFOLD.randn(base)
        >>> ww = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
        >>> z = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
        >>> Jv = v.probe(ww)
        >>> JTz = t3m.T3Tangent.probe_transpose(z, ww, base, sum_over_probes=True)
        >>> lhs = float(np.sum([np.sum(a * b) for a, b in zip(z, Jv)]))
        >>> print(bool(abs(lhs - float(JTz.corewise_inner(v))) < 1e-9))
        True

        Without summing, the result is a tangent-stacked T3Tangent (V = the probe stack):

        >>> JTz_batch = t3m.T3Tangent.probe_transpose(z, ww, base)  # sum_over_probes=False
        >>> print(JTz_batch.tangent_stack_shape, JTz_batch.base_stack_shape)
        (2,) ()

        With ``K``-stacked residuals (``W + K + C``), the tangent batch ``K`` is carried through:

        >>> zb = tuple(np.random.randn(2, 3, N) for N in (10, 11, 12))  # W=(2,), K=(3,), C=()
        >>> print(t3m.T3Tangent.probe_transpose(zb, ww, base, sum_over_probes=True).tangent_stack_shape)
        (3,)
        >>> print(t3m.T3Tangent.probe_transpose(zb, ww, base).tangent_stack_shape)  # sum=False -> W + K
        (2, 3)

        ``sum_over_probes=True`` is exactly the probe-stack (``W``) sum of the ``False`` result:

        >>> kept   = t3m.T3Tangent.probe_transpose(z, ww, base)                        # W stays a stack
        >>> summed = t3m.T3Tangent.probe_transpose(z, ww, base, sum_over_probes=True)   # W summed
        >>> dU_sum = tuple(np.sum(c, axis=0) for c in kept.variations.tucker_variations)  # sum the W axis
        >>> err = max(float(np.linalg.norm(a - b))
        ...           for a, b in zip(dU_sum, summed.variations.tucker_variations))
        >>> print(bool(err < 1e-9))
        True
        """
        # probing's base order is exactly T3Basis.data = (up, down, left, right) -- no reorder.
        # numpy/jax dispatch is inferred from the input array types inside probing.
        dU_tildes, dG_tildes = probing.probe_tangent_transpose(
            ztildes, ww, basis.data, sum_over_probes=sum_over_probes,
        )
        return T3Tangent(basis, bvf.T3Variations(dU_tildes, dG_tildes))

    def apply(
            self,
            ww:         typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
    ) -> NDArray:                               # apply(v, ww), one scalar per stack element; shape=W+K+C
        """Apply this tangent vector in all modes: contract the dense tangent with ``ww`` everywhere.

        The all-modes special case of :py:meth:`probe` (probing leaves one index free; this contracts
        them all). It is the tangent analogue of :py:meth:`.TuckerTensorTrain.apply`, and is cheaper
        than probing -- a single left-to-right sweep, no right/central sweeps, no per-mode assembly.
        The result is stacked ``W + K + C`` (apply-vector stack ``W`` outer, tangent stack ``K``, base
        stack ``C`` inner); a plain scalar when there are no stacks.

        See Section 6.2.2 (Algorithms 6-7) of Alger et al. (2026), "Tucker Tensor Train Taylor
        Series" (arXiv:2603.21141).

        See Also
        --------
        entries
        probe

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.manifold as t3m
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> base, variations = bvf.t3_orthogonal_representations(x)
        >>> v = t3m.T3Tangent(base, variations)
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> a = v.apply(ww)
        >>> a_dense = np.einsum('ijk,i,j,k->', v.to_dense(), *ww)   # dense reference
        >>> print(bool(abs(float(a) - float(a_dense)) < 1e-9))
        True
        """
        # base order is exactly T3Basis.data = (up, down, left, right) -- no reorder.
        # numpy/jax dispatch is inferred from the input array types inside probing.
        return probing.apply_tangent(ww, self.variations.data, self.basis.data)

    def entries(
            self,
            index:      NDArray,  # int, shape=(d,)+W (a stack W of multi-indices)
    ) -> NDArray:                 # entries of the dense tangent at ``index``; shape=W+K+C
        """Extract entries of the dense tangent at ``index`` (without forming the dense tangent).

        The tangent analogue of :py:meth:`.TuckerTensorTrain.entries`, and the all-modes special case
        of :py:meth:`probe` with unit vectors -- computed by **slicing** Tucker-core fibers (no
        contraction, no ``N`` factor), then the same left-to-right sweep as :py:meth:`apply`. ``index``
        is ``(d,) + W``: ``index[m]`` holds the (stack ``W`` of) indices into mode ``m``. Result
        stacked ``W + K + C``.

        See Also
        --------
        apply
        probe

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.manifold as t3m
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> base, variations = bvf.t3_orthogonal_representations(x)
        >>> v = t3m.T3Tangent(base, variations)
        >>> idx = (3, 5, 7)
        >>> print(bool(abs(float(v.entries(idx)) - float(v.to_dense()[idx])) < 1e-9))
        True
        """
        return probing.entries_tangent(index, self.variations.data, self.basis.data)

    @staticmethod
    def apply_transpose(
            c:                  NDArray,                # residual, shape=W+C
            ww:                 typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
            basis:              bvf.T3Basis,
            sum_over_probes:    bool = False,           # True: sum the apply stack W (Gauss-Newton apply^T c)
    ) -> 'T3Tangent':
        """Apply the transpose ``apply^T`` of :py:meth:`apply`: back-project a residual ``c`` into a tangent.

        The adjoint of the (linear-in-the-variation) all-modes :py:meth:`apply`. With
        ``sum_over_probes=False`` (default) the apply-vector stack ``W`` becomes the result's tangent
        stack (one tangent per apply-set); with ``sum_over_probes=True`` ``W`` is summed -- the usual
        Gauss-Newton ``apply^T c`` back-projection (a single tangent when ``W = ()``). Needs only the
        base sweep + a single-term scatter assembly (cheaper than a general :py:meth:`probe_transpose`).

        ``False`` is the primary transpose; ``sum_over_probes=True == Σ_W sum_over_probes=False``. See
        *Batching & stacking* §11 (``docs/batching_and_stacking.md``) for which mode to use and why.

        Examples
        --------
        Adjoint identity ``<apply^T c, v> == c * apply(v)`` (no stacks):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.manifold as t3m
        >>> import t3toolbox.corewise as cw
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> base, _ = bvf.t3_orthogonal_representations(x)
        >>> v = t3m.COREWISE.randn(base)
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> ATc = t3m.T3Tangent.apply_transpose(np.asarray(1.7), ww, base, sum_over_probes=True)
        >>> lhs = float(cw.corewise_dot(ATc.variations.data, v.variations.data))
        >>> print(bool(abs(lhs - 1.7 * float(v.apply(ww))) < 1e-9))
        True
        """
        # base order is exactly T3Basis.data = (up, down, left, right) -- no reorder.
        dU, dG = probing.apply_tangent_transpose(c, ww, basis.data, sum_over_probes=sum_over_probes)
        return T3Tangent(basis, bvf.T3Variations(dU, dG))

    @staticmethod
    def entries_transpose(
            c:                  NDArray,                # residual, shape=W+C
            index:              NDArray,                # int, shape=(d,)+W
            basis:              bvf.T3Basis,
            sum_over_probes:    bool = False,           # True: sum the apply stack W (Gauss-Newton entries^T c)
    ) -> 'T3Tangent':
        """Apply the transpose ``entries^T`` of :py:meth:`entries`: scatter ``c`` at ``index`` into a tangent.

        The adjoint of :py:meth:`entries` -- identical to :py:meth:`apply_transpose` with the up-index
        ``ξ̂`` from fiber slicing and unit apply-vectors ``e_{index}``. ``sum_over_probes`` as in
        :py:meth:`apply_transpose` (see *Batching & stacking* §11).

        See Also
        --------
        entries
        apply_transpose
        """
        dU, dG = probing.entries_tangent_transpose(c, index, basis.data, sum_over_probes=sum_over_probes)
        return T3Tangent(basis, bvf.T3Variations(dU, dG))

    ###############################################
    ##########    Symmetric derivatives    ########
    ###############################################

    def probe_derivatives(
            self,
            ww:     typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                    # highest derivative order
    ) -> typ.Sequence[NDArray]:             # len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
        """Symmetric directional derivatives of probing this tangent vector, in one repeated direction.

        Returns, for each mode ``i``, the stack ``y_i^(t) = d^t/ds^t [probe(X + s P)]_i|_0`` for
        ``t=0..order`` -- the derivative analogue of :py:meth:`probe`, obtained by perturbing every probe
        vector in the same direction ``P``. Index ``0`` is the ordinary :py:meth:`probe`. Stacks ``order
        + W + K + C`` (order outermost; sample stack ``W``, tangent stack ``K``, base stack ``C``). The
        points ``X`` (``ww``) and the perturbations ``P`` (``pp``) must share the sample stack ``W``.

        See ``docs/symmetric_probe_derivatives.tex`` and ``docs/derivatives_mirror_plan.md``.

        See Also
        --------
        probe
        apply_derivatives
        probe_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.manifold as t3m
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> base, variations = bvf.t3_orthogonal_representations(x)
        >>> v = t3m.T3Tangent(base, variations)
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> zj = v.probe_derivatives(ww, pp, 3)
        >>> print([z.shape for z in zj])           # (order+1,) + (Ni,)
        [(4, 10), (4, 11), (4, 12)]
        >>> print([bool(np.allclose(z[0], z0)) for z, z0 in zip(zj, v.probe(ww))])  # order 0 == probe
        [True, True, True]
        """
        probe_derivatives.check_perturbation_vectors(ww, pp)
        return probe_derivatives.probe_tangent_derivatives(ww, pp, self.variations.data, self.basis.data, order)

    def apply_derivatives(
            self,
            ww:     typ.Sequence[NDArray],  # apply vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                    # highest derivative order
    ) -> NDArray:                           # shape=(order+1,)+W+K+C
        """Symmetric directional derivatives of applying this tangent vector in all modes.

        The all-modes analogue of :py:meth:`probe_derivatives` (and the derivative analogue of
        :py:meth:`apply`): ``y^(t) = d^t/ds^t apply(X + s P)|_0`` for ``t=0..order`` (a scalar per
        stack element). Stacks ``order + W + K + C``. ``X`` and ``P`` must share the sample stack ``W``.

        See Also
        --------
        apply
        probe_derivatives
        apply_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.manifold as t3m
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> base, variations = bvf.t3_orthogonal_representations(x)
        >>> v = t3m.T3Tangent(base, variations)
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> yj = v.apply_derivatives(ww, pp, 3)
        >>> print(yj.shape)                        # (order+1,) -- one scalar per order
        (4,)
        >>> print(bool(np.allclose(yj[0], v.apply(ww))))     # order 0 == apply
        True
        """
        probe_derivatives.check_perturbation_vectors(ww, pp)
        return probe_derivatives.apply_tangent_derivatives(ww, pp, self.variations.data, self.basis.data, order)

    def entries_derivatives(
            self,
            index:  NDArray,                # int, shape=(d,)+W -- grid points
            pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                    # highest derivative order
    ) -> NDArray:                           # shape=(order+1,)+W+K+C
        """Symmetric directional derivatives of this tangent's entries at ``index``, in direction ``P``.

        The Taylor data of the tangent's multilinear extension at grid corner ``index``, in direction
        ``P``: ``y^(t) = d^t/ds^t apply(e_{index} + s P)|_0`` for ``t=0..order``. Index ``0`` is the
        ordinary :py:meth:`entries`. Stacks ``order + W + K + C``. ``index`` and ``P`` share ``W``.

        See Also
        --------
        entries
        apply_derivatives
        entries_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.manifold as t3m
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> base, variations = bvf.t3_orthogonal_representations(x)
        >>> v = t3m.T3Tangent(base, variations)
        >>> index = np.array([3, 5, 7])
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> yj = v.entries_derivatives(index, pp, 3)
        >>> print(yj.shape)
        (4,)
        >>> print(bool(np.allclose(yj[0], v.entries(index))))   # order 0 == entries
        True
        """
        probe_derivatives.check_perturbation_index(index, pp, self.shape)
        return probe_derivatives.entries_tangent_derivatives(index, pp, self.variations.data, self.basis.data, order)

    @staticmethod
    def probe_derivatives_transpose(
            ztildes:            typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
            ww:                 typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
            pp:                 typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            basis:              bvf.T3Basis,
            order:              int,                    # highest derivative order
            sum_over_probes:    bool = False,           # True: sum the sample stack W (Gauss-Newton J^T r)
    ) -> 'T3Tangent':
        """Transpose ``(J^(s))^T`` of :py:meth:`probe_derivatives`; returns a :py:class:`T3Tangent` at ``basis``.

        The adjoint of :py:meth:`probe_derivatives`. Residual jets ``ztildes`` live in its output space
        (``(order+1)+W+K+C+(Ni,)``); the tangent batch ``K`` rides through, the sample stack ``W`` is
        summed (``sum_over_probes=True``, the Gauss-Newton ``J^T r``) or kept (``False``, ``W`` becomes
        the tangent stack). Bare ``(J^(s))^T`` (no gauge projector).

        See Also
        --------
        probe_derivatives
        probe_transpose

        Examples
        --------
        Adjoint identity ``<r, J v> = <J^T r, v>`` (sum over probes):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.manifold as t3m
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> base, _ = bvf.t3_orthogonal_representations(x)
        >>> v = t3m.COREWISE.randn(base)
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> pp = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> Jv = v.probe_derivatives(ww, pp, 2)
        >>> r = [np.random.randn(*z.shape) for z in Jv]
        >>> JTr = t3m.T3Tangent.probe_derivatives_transpose(r, ww, pp, base, 2, sum_over_probes=True)
        >>> lhs = sum(float(np.sum(ri * zi)) for ri, zi in zip(r, Jv))
        >>> print(bool(abs(lhs - float(JTr.corewise_inner(v))) < 1e-9))
        True
        """
        probe_derivatives.check_perturbation_vectors(ww, pp)
        dU, dG = probe_derivatives.probe_tangent_derivatives_transpose(
            ztildes, ww, pp, basis.data, order, sum_over_probes=sum_over_probes)
        return T3Tangent(basis, bvf.T3Variations(dU, dG))

    @staticmethod
    def apply_derivatives_transpose(
            c:                  NDArray,                # residual jet (scalar), shape=(order+1,)+W+K+C
            ww:                 typ.Sequence[NDArray],  # apply vectors X,        len=d, elm_shape=W+(Ni,)
            pp:                 typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            basis:              bvf.T3Basis,
            order:              int,                    # highest derivative order
            sum_over_probes:    bool = False,
    ) -> 'T3Tangent':
        """Transpose of :py:meth:`apply_derivatives`: back-project residual jets ``c`` into a tangent.

        The adjoint-state apply-derivative transpose (the scalar residual jet seeds one sweep; about half
        a :py:meth:`probe_derivatives_transpose`). ``sum_over_probes`` as in :py:meth:`probe_derivatives_transpose`.

        See Also
        --------
        apply_derivatives
        apply_transpose
        """
        probe_derivatives.check_perturbation_vectors(ww, pp)
        dU, dG = probe_derivatives.apply_tangent_derivatives_transpose(
            c, ww, pp, basis.data, order, sum_over_probes=sum_over_probes)
        return T3Tangent(basis, bvf.T3Variations(dU, dG))

    @staticmethod
    def entries_derivatives_transpose(
            c:                  NDArray,                # residual jet (scalar), shape=(order+1,)+W+K+C
            index:              NDArray,                # int, shape=(d,)+W
            pp:                 typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            basis:              bvf.T3Basis,
            order:              int,                    # highest derivative order
            sum_over_probes:    bool = False,
    ) -> 'T3Tangent':
        """Transpose of :py:meth:`entries_derivatives`: scatter residual jets ``c`` at ``index`` into a tangent.

        Identical to :py:meth:`apply_derivatives_transpose` with the base up-index from fiber slicing and
        the ambient ``w_jet`` from the unit vectors ``e_{index}`` (so the Tucker-variation gradient
        scatters onto the indexed rows).

        See Also
        --------
        entries_derivatives
        apply_derivatives_transpose
        """
        probe_derivatives.check_perturbation_index(index, pp, basis.shape)
        dU, dG = probe_derivatives.entries_tangent_derivatives_transpose(
            c, index, pp, basis.data, order, sum_over_probes=sum_over_probes)
        return T3Tangent(basis, bvf.T3Variations(dU, dG))

    ############################################
    ##########    Stacking    ##################
    ############################################

    def unstack_tangents(self):
        """Unstack over the tangent stack ``K``: a ``K``-shaped tree of tangents sharing this base.

        Decomposes the batch of tangent *directions* ("for each vector within the basis"). Each leaf
        is a :py:class:`T3Tangent` with ``tangent_stack_shape == ()`` and ``base_stack_shape`` equal
        to this tangent's -- and, because the base point is shared across ``K``, every leaf holds the
        **same** :py:class:`T3Basis` object, so the leaves live in one tangent space (linear algebra
        between them is defined). Inverse of :py:meth:`stack_tangents`.
        """
        variations_tree = tangent_operations.unstack_tangent_stack(self.basis.data, self.variations.data)
        leaf_structure = ((None,) * self.d, (None,) * self.d)  # a single T3Variations.data
        return stacking.apply_func_to_leaf_subtrees(
            variations_tree,
            lambda vd: T3Tangent(self.basis, bvf.T3Variations(*vd)),  # SAME basis object (shared)
            leaf_structure,
        )

    def unstack_basis(self):
        """Unstack over the base stack ``C``: a ``C``-shaped tree of single-base-point tangents.

        Decomposes over base *points* ("for each basis"). Each leaf is a :py:class:`T3Tangent` with
        ``base_stack_shape == ()`` and ``tangent_stack_shape`` equal to this tangent's; the leaves
        sit at **different** base points (different tangent spaces, so they are not mutually
        linear-algebra compatible). Inverse of :py:meth:`stack_basis`.
        """
        paired_tree = tangent_operations.unstack_base_stack(self.basis.data, self.variations.data)
        leaf_structure = (((None,) * self.d,) * 4,            # one (basis_data,
                          ((None,) * self.d, (None,) * self.d))  #      variations_data) pair
        return stacking.apply_func_to_leaf_subtrees(
            paired_tree,
            lambda bv: T3Tangent(bvf.T3Basis(*bv[0]), bvf.T3Variations(*bv[1])),
            leaf_structure,
        )

    @staticmethod
    def stack_tangents(tree) -> 'T3Tangent':
        """Stack a ``K``-shaped tree of tangents (sharing one base) into a tangent-stacked T3Tangent.

        Inverse of :py:meth:`unstack_tangents`. Requires every leaf to hold the **same**
        :py:class:`T3Basis` object (object identity, as in :py:meth:`inner` / :py:meth:`__add__`):
        the tangents being stacked must live in the same tangent space. The shared base is reused and
        the variations are stacked over the new outer tangent stack ``K``.
        """
        leaves = _flatten_tangents(tree)
        base = leaves[0].basis
        for t in leaves[1:]:
            if t.basis is not base:
                raise ValueError(
                    'stack_tangents requires every tangent to share the same T3Basis object (object '
                    'identity, not merely numerically-equal cores) -- they must live in the same '
                    'tangent space. To stack tangents at *different* base points, use stack_basis.'
                )
        variations_tree = stacking.apply_func_to_leaf_subtrees(tree, lambda t: t.variations.data, None)
        variations_data = tangent_operations.stack_tangent_stack(variations_tree)
        return T3Tangent(base, bvf.T3Variations(*variations_data))

    @staticmethod
    def stack_basis(tree) -> 'T3Tangent':
        """Stack a ``C``-shaped tree of single-base-point tangents into a base-stacked T3Tangent.

        Inverse of :py:meth:`unstack_basis`. The leaves sit at **different** base points (distinct
        bases), so no shared-base identity is required; they must share the same structure and the
        same tangent stack ``K``. The bases are stacked over the base stack ``C``, which is placed
        innermost so the variation stack becomes ``K + C``.
        """
        leaves = _flatten_tangents(tree)
        v0 = leaves[0]
        for t in leaves[1:]:
            if t.structure != v0.structure or t.tangent_stack_shape != v0.tangent_stack_shape:
                raise ValueError(
                    'stack_basis requires all tangents to share the same structure and tangent '
                    'stack K (only the base point may differ across the base stack C).'
                )
        paired_tree = stacking.apply_func_to_leaf_subtrees(
            tree, lambda t: (t.basis.data, t.variations.data), None)
        basis_data, variations_data = tangent_operations.stack_base_stack(paired_tree)
        return T3Tangent(bvf.T3Basis(*basis_data), bvf.T3Variations(*variations_data))


############################################
##########    Geometry    ##################
############################################


class ManifoldGeometry:
    """The Riemannian geometry of the fixed-rank Tucker tensor train manifold ``M``.

    Optimization happens *on* ``M``: tangents live in ``T_xM`` with an orthonormal, gauged frame, the
    metric is Hilbert-Schmidt, and one moves by the manifold retraction (truncate the shifted
    doubled-rank embedding back to the base ranks). A geometry is a *stateless* bundle of the three
    chart-level choices that distinguish this from the over-parametrized corewise geometry -- the
    frame (:py:meth:`base`), the gauge projection ``Pi`` (:py:meth:`project`), and the retraction
    (:py:meth:`retract`) -- plus the manifold-only ambient projection and transport. The point lives
    in the caller (or the fitting model), not the geometry; use the module singleton :py:data:`MANIFOLD`.

    See Section 6 and Appendix A.3 of Alger et al. (2026), "Tucker Tensor Train Taylor Series"
    (arXiv:2603.21141). The corewise counterpart is :py:class:`CorewiseGeometry`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
    >>> base = t3m.MANIFOLD.base(x)            # orthonormal frame at x
    >>> print(base.is_orthogonal())
    True
    >>> v = t3m.MANIFOLD.randn(base)           # a standard Gaussian on T_xM (gauged)
    >>> print(v.is_gauged())
    True
    >>> y = t3m.MANIFOLD.retract(t3m.T3Tangent.zeros(base))   # retract the zero tangent == the base point
    >>> print(bool(np.allclose(y.to_dense(), x.to_dense())))
    True
    """

    def base(
            self,
            x:  t3.TuckerTensorTrain,
    ) -> bvf.T3Basis:  # orthonormal frame (U, O, P, Q) at x
        """The orthonormal frame at ``x`` (the left/right/outer-orthogonal representation)."""
        base, _ = bvf.t3_orthogonal_representations(x)
        return base

    def randn(
            self,
            basis:          bvf.T3Basis,
            stack_shape:    typ.Tuple[int, ...] = (),  # extra tangent stack K (a batch of tangents)
    ) -> T3Tangent:  # gauged random tangent at basis
        """Random tangent at ``basis``: a standard Gaussian on the tangent space ``T_xM``.

        Raw i.i.d. N(0, 1) variation cores, then the gauge projection :py:meth:`project` (``Pi``). For
        an orthogonal, minimal-rank ``basis`` this is exactly the standard Gaussian on ``T_xM``; for a
        non-orthogonal basis it is merely gauged. ``stack_shape`` is the extra outer tangent stack
        ``K`` (default ``()``).
        """
        use_jax = tree_contains_jax(basis.data)
        full_stack = tuple(stack_shape) + basis.stack_shape  # K + C
        v = T3Tangent(basis, bvf.T3Variations.randn(basis.variation_shapes, full_stack, use_jax))
        return self.project(v)

    def random_orthogonal(
            self,
            shape:                  typ.Sequence[int],         # (N0,...,N(d-1))
            tucker_ranks:           typ.Sequence[int],         # (n0,...,n(d-1))
            tt_ranks:               typ.Sequence[int],         # (1,r1,...,r(d-1),1)
            stack_shape:            typ.Tuple[int, ...] = (),  # base stack C (random base points)
            tangent_stack_shape:    typ.Tuple[int, ...] = (),  # tangent stack K
            use_jax:                bool = False,
    ) -> T3Tangent:  # gauged random tangent at a random orthogonal base point
        """A gauged random tangent at a random orthogonal base point (random direction, random base)."""
        base = bvf.T3Basis.random_orthogonal(shape, tucker_ranks, tt_ranks,
                                             stack_shape=stack_shape, use_jax=use_jax)
        return self.randn(base, stack_shape=tangent_stack_shape)

    def randn_like(
            self,
            tangent:    T3Tangent,  # reuse its base + tangent stack K
    ) -> T3Tangent:  # gauged random tangent at tangent's base
        """A gauged random tangent at ``tangent``'s base, with ``tangent``'s tangent stack ``K``."""
        return self.randn(tangent.basis, stack_shape=tangent.tangent_stack_shape)

    def project(
            self,
            v:  T3Tangent,
    ) -> T3Tangent:  # gauged tangent at v's basis (a DIFFERENT vector)
        """The gauge projection ``Pi``: raw cotangent -> Riemannian gradient (orthogonal gauge).

        Orthogonally projects ``v``'s variations onto the gauged tangent space (the gauge conditions
        (48)-(49), Appendix A.3). Represents a DIFFERENT tangent vector than ``v``. This is the map
        that turns a bare adjoint ``J^T r`` into the Riemannian gradient. For the vector-preserving
        gauge fix see :py:meth:`project_oblique`.
        """
        new_variations = tangent_operations.orthogonal_gauge_projection(v.basis.data, v.variations.data)
        return T3Tangent(v.basis, bvf.T3Variations(*new_variations))

    def project_oblique(
            self,
            v:  T3Tangent,
    ) -> T3Tangent:  # gauged tangent at v's basis (the SAME vector)
        """Gauge ``v``'s variations while preserving the represented tangent vector (oblique projection).

        Returns a tangent at the same basis representing the SAME vector as ``v`` but gauged, so that on
        an orthogonal minimal-rank basis :py:meth:`inner` / :py:meth:`norm` give the true Hilbert-Schmidt
        values.
        """
        new_variations = tangent_operations.oblique_gauge_projection(v.basis.data, v.variations.data)
        return T3Tangent(v.basis, bvf.T3Variations(*new_variations))

    def inner(
            self,
            t1: T3Tangent,
            t2: T3Tangent,
    ) -> NDArray:  # the Hilbert-Schmidt inner product, shape = stack_shape (K + C)
        """The **Hilbert-Schmidt** inner product of two tangents -- the Riemannian metric on ``M``.

        Computes the corewise (coordinate) dot, which equals HS on this geometry's orthonormal, gauged
        frame. In **safe mode** it checks the preconditions for that equality: the two tangents share a
        frame, the frame is orthogonal, and **both** variations are gauged (minimal rank is a documented
        caveat -- see ``docs/numerical_contract_catalog.md``). For the raw coordinate dot with no HS claim
        and no orthogonal/gauge check, use :py:meth:`T3Tangent.corewise_inner`.
        """
        t1._check_same_tangent_space(t2)
        if safety.checks_active(t1.basis.data, t1.variations.data, t2.variations.data):
            atol = safety.effective_rtol(t1.basis.data, t1.variations.data, t2.variations.data)
            safety.require(t1.basis.is_orthogonal(atol=atol),
                           'ManifoldGeometry.inner is the Hilbert-Schmidt metric and requires an '
                           'orthogonal frame. Use T3Tangent.corewise_inner for the raw coordinate dot, '
                           'or run in unsafe mode (safety.unsafe()).')
            safety.require(t1.is_gauged(atol=atol) and t2.is_gauged(atol=atol),
                           'ManifoldGeometry.inner requires both tangents gauged. Gauge them via '
                           'ManifoldGeometry.project / project_oblique, use T3Tangent.corewise_inner, '
                           'or run in unsafe mode.')
        return cw.corewise_stack_dot(t1.variations.data, t2.variations.data, len(t1.stack_shape))

    def norm(
            self,
            t:  T3Tangent,
    ) -> NDArray:  # the Hilbert-Schmidt norm, shape = stack_shape (K + C)
        """The **Hilbert-Schmidt** norm of a tangent. Safe mode checks the frame orthogonal + variations
        gauged (the preconditions for the coordinate norm to equal HS; minimal rank a documented caveat).
        For the raw coordinate norm use :py:meth:`T3Tangent.corewise_norm`."""
        if safety.checks_active(t.basis.data, t.variations.data):
            atol = safety.effective_rtol(t.basis.data, t.variations.data)
            safety.require(t.basis.is_orthogonal(atol=atol),
                           'ManifoldGeometry.norm is the Hilbert-Schmidt metric and requires an '
                           'orthogonal frame. Use T3Tangent.corewise_norm, or run in unsafe mode.')
            safety.require(t.is_gauged(atol=atol),
                           'ManifoldGeometry.norm requires gauged variations. Gauge via '
                           'ManifoldGeometry.project / project_oblique, use T3Tangent.corewise_norm, '
                           'or run in unsafe mode.')
        return cw.corewise_stack_norm(t.variations.data, len(t.stack_shape))

    def retract(
            self,
            p:  T3Tangent,  # step (a tangent at the current point's frame)
    ) -> t3.TuckerTensorTrain:  # retracted point on M, at p's base ranks
        """Retract the step ``p`` to the manifold: shifted doubled-rank embedding, truncated to base ranks.

        Forms ``base point + p`` and truncates back to ``p``'s base ranks via the implicit T3-SVD
        (Algorithm 10). The current point is carried by ``p.basis`` (its orthonormal frame), so no
        separate point argument is needed.
        """
        cores = tangent_operations.retract(p.basis.data, p.variations.data)
        return t3.TuckerTensorTrain(*cores)

    def project_ambient(
            self,
            basis:  bvf.T3Basis,                               # orthogonal base point of the tangent space
            grad:   typ.Union[t3.TuckerTensorTrain, NDArray],  # ambient gradient: a T3 or a dense array
            method: str = 'contraction',                       # dense only: 'contraction' (no SVD) or 't3svd'
    ) -> T3Tangent:  # the Riemannian gradient (gauged projection of grad) at basis
        """Project an ambient gradient onto ``T_xM`` -- the Riemannian gradient.

        ``grad`` is the Euclidean/ambient gradient, either a :py:class:`TuckerTensorTrain` or a dense
        array (leading axes beyond the ``d`` modes are a stack). Returns the (gauged) tangent
        ``P_T(grad)``; the residual ``grad - P_T(grad)`` is orthogonal to the tangent space. Requires
        an **orthogonal** ``basis`` (minimal rank is *not* required). For a dense ``grad``, ``method``
        selects the algorithm (same projection): ``'contraction'`` (default, contract against the
        frames, no SVD) or ``'t3svd'`` (exact T3-SVD then project; expensive). This is the
        tangent-space projection of Section 6 / Appendix A.3.
        """
        if isinstance(grad, t3.TuckerTensorTrain):
            variations = tangent_operations.project_t3_onto_tangent_space(basis.data, grad.data)
            return T3Tangent(basis, bvf.T3Variations(*variations))
        if method == 'contraction':
            variations = tangent_operations.project_dense_onto_tangent_space(basis.data, grad)
            return T3Tangent(basis, bvf.T3Variations(*variations))
        elif method == 't3svd':
            d = len(basis.shape)
            stack_shape = tuple(grad.shape[:grad.ndim - d])
            x, _, _ = t3.TuckerTensorTrain.t3svd_dense(grad, stack_shape=stack_shape)
            return self.project_ambient(basis, x)
        else:
            raise ValueError(
                "project_ambient: method must be 'contraction' or 't3svd', got %r" % (method,))

    def transport(
            self,
            v:          T3Tangent,
            new_basis:  bvf.T3Basis,
    ) -> T3Tangent:  # v transported to the tangent space at new_basis
        """Projective vector transport of ``v`` to the tangent space at ``new_basis``.

        Re-projects ``v`` (as an ambient tensor via :py:meth:`T3Tangent.to_t3`) orthogonally onto the
        tangent space at ``new_basis``. The cheap, standard choice for fixed-rank Riemannian
        optimization -- not parallel transport.
        """
        return self.project_ambient(new_basis, v.to_t3())


class CorewiseGeometry:
    """The Euclidean geometry of the core parameter space ``P`` (the over-parametrized cover of ``M``).

    Optimization happens *on* the raw cores: tangents are perturbations of the cores ``(U, G, G, G)``
    (the non-orthonormal frame whose down/left/right cores are all the TT cores ``G``), the metric is
    the plain Euclidean (corewise) inner product, the "projection" is the identity (no gauge), and the
    retraction is vector addition in the cores. The corewise gradient is a genuine
    :py:class:`T3Tangent` at this frame -- not a raw tuple. Gauge directions lie in the kernel of the
    pushforward, so the Gauss-Newton Hessian is gauge-singular (fine for Adam / L-BFGS, needs
    regularization for Newton). Use the module singleton :py:data:`COREWISE`. The §6.3 substitution
    ``(O, P, Q) -> G`` is exactly this change of frame; the manifold counterpart is
    :py:class:`ManifoldGeometry`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
    >>> base = t3m.COREWISE.base(x)            # the (U, G, G, G) frame
    >>> v = t3m.COREWISE.randn(base)           # raw randn cores (no gauge)
    >>> print(v.is_gauged())
    False
    >>> y = t3m.COREWISE.retract(v)            # additive: cores += v (a multilinear curve on M)
    >>> U, G = x.data
    >>> dU, dG = v.variations.tucker_variations, v.variations.tt_variations
    >>> ref = t3.TuckerTensorTrain(tuple(u + du for u, du in zip(U, dU)),
    ...                            tuple(g + dg for g, dg in zip(G, dG)))
    >>> print(bool(np.allclose(y.to_dense(), ref.to_dense())))
    True
    """

    def base(
            self,
            x:  t3.TuckerTensorTrain,
    ) -> bvf.T3Basis:  # the (U, G, G, G) non-orthonormal frame at x
        """The core-parameter frame at ``x``: ``(U, G, G, G)`` (down/left/right all the TT cores ``G``)."""
        tucker_cores, tt_cores = x.data
        return bvf.T3Basis(tucker_cores, tt_cores, tt_cores, tt_cores)

    def randn(
            self,
            basis:          bvf.T3Basis,
            stack_shape:    typ.Tuple[int, ...] = (),  # extra tangent stack K
    ) -> T3Tangent:  # raw random tangent at basis (ungauged)
        """Random tangent at ``basis``: raw i.i.d. N(0, 1) variation cores (the natural corewise Gaussian)."""
        use_jax = tree_contains_jax(basis.data)
        full_stack = tuple(stack_shape) + basis.stack_shape  # K + C
        return T3Tangent(basis, bvf.T3Variations.randn(basis.variation_shapes, full_stack, use_jax))

    def randn_like(
            self,
            tangent:    T3Tangent,
    ) -> T3Tangent:  # raw random tangent at tangent's base
        """A raw random tangent at ``tangent``'s base, with ``tangent``'s tangent stack ``K``."""
        return self.randn(tangent.basis, stack_shape=tangent.tangent_stack_shape)

    def project(
            self,
            v:  T3Tangent,
    ) -> T3Tangent:  # v unchanged (no gauge on the core parameter space)
        """The identity: the core parameter space is Euclidean, with no gauge projection."""
        return v

    def inner(
            self,
            t1: T3Tangent,
            t2: T3Tangent,
    ) -> NDArray:  # the Euclidean (coordinate) inner product, shape = stack_shape (K + C)
        """The **Euclidean** (coordinate) inner product of two tangents on the core parameter space.

        The corewise dot of the variations -- *no* orthogonal/gauge requirement (the ``(U,G,G,G)`` frame
        is non-orthonormal by design). Safe mode checks only that the two tangents share a frame. This is
        exactly :py:meth:`T3Tangent.corewise_inner`."""
        return t1.corewise_inner(t2)

    def norm(
            self,
            t:  T3Tangent,
    ) -> NDArray:  # the Euclidean (coordinate) norm, shape = stack_shape (K + C)
        """The **Euclidean** (coordinate) norm of a tangent (= :py:meth:`T3Tangent.corewise_norm`); no
        precondition."""
        return t.corewise_norm()

    def retract(
            self,
            p:  T3Tangent,  # step (a corewise tangent at base = (U, G, G, G))
    ) -> t3.TuckerTensorTrain:  # additive retraction: cores += p
        """Additive retraction: add the variation cores to the point's cores (``cores += p``).

        Recovers the point ``(U, G)`` from ``p.basis`` (which :py:meth:`base` built as ``(U, G, G, G)``)
        and adds the variations. ``p`` must be a corewise tangent (a basis from :py:meth:`base`).
        """
        x_data = (p.basis.up_tucker_cores, p.basis.left_tt_cores)  # (U, G) from the (U, G, G, G) frame
        new = cw.corewise_add(x_data, p.variations.data)
        return t3.TuckerTensorTrain(*new)


MANIFOLD = ManifoldGeometry()    # the fixed-rank Riemannian geometry (gauge Pi, manifold retraction)
COREWISE = CorewiseGeometry()    # the core parameter-space Euclidean geometry (no gauge, additive retraction)


# Note: exp / log / geodesic (the true Riemannian exponential, logarithm, and geodesics) are
# intentionally omitted -- fixed-rank manifolds have no closed-form geodesics. The library uses the
# retraction (:py:meth:`ManifoldGeometry.retract`) and projective transport
# (:py:meth:`ManifoldGeometry.transport`) as the practical substitutes.


def _flatten_tangents(tree) -> typ.List['T3Tangent']:
    """Flatten an array-like tree of T3Tangents (nested tuples) into a flat list of leaves."""
    if isinstance(tree, T3Tangent):
        return [tree]
    out = []
    for sub in tree:
        out.extend(_flatten_tangents(sub))
    return out


if has_jax:
    import jax

    # Register T3Tangent as a jax pytree with the BASIS as aux_data: the fixed frame is static, only
    # the variations (the moving tangent vector) are differentiable leaves -- matching the manifold
    # picture and what one optimizes/vmaps. Because aux_data preserves object identity through
    # flatten/unflatten, the same-tangent-space guard (`self.basis is other.basis`) keeps working
    # under jit (two tangents built from the same T3Basis object stay identical). The basis is then a
    # jit compile-time constant: hold the basis object stable to keep cache hits (a new base point
    # recompiles). To differentiate w.r.t. the basis, use the backend functions on the raw cores.
    #
    # REVISIT (basis-as-aux vs recompile cost): this choice was made on conceptual grounds -- the basis
    # is the "fixed" frame, the tangent varies within it; we do not want autodiff to differentiate
    # through the base; we want the same-base identity guard to deny linear algebra across frames. The
    # PRACTICAL cost was under-weighed: a new base recompiles, so a Newton-CG that jits at the
    # outer-step granularity (base changes every step) recompiles every step, which can dominate when
    # the inner CG is short. The two jit patterns sidestep it (see fitting.py "Jitting an optimizer":
    # build the base inside the jit for a per-step loop; close the model over for an amortized inner
    # matvec), but if neither fits a use case, weigh offering a base-as-leaf tangent variant (no
    # recompile, identity guard then eager-only). Decision deferred; the patterns cover today's needs.
    jax.tree_util.register_pytree_node(
        T3Tangent,
        lambda x: ((x.variations,), x.basis),
        lambda basis, children: T3Tangent(basis, children[0]),
    )

    # Register the stateless geometry singletons as ZERO-LEAF pytrees (no array leaves, aux=None): they
    # carry no data, so they pass through jit/vmap transparently as ordinary args and reconstruct to an
    # equivalent stateless instance (all instances of a geometry are interchangeable). This is purely
    # for ergonomics -- geometries are normally closed over -- but it removes the "cannot interpret
    # ManifoldGeometry as an abstract value" footgun. The GaussNewtonModel is deliberately NOT
    # registered: it is a scope-local, per-outer-step operator that the jit patterns never cross the
    # boundary with (built inside, or closed over); see fitting.py "Jitting an optimizer".
    jax.tree_util.register_pytree_node(
        ManifoldGeometry, lambda g: ((), None), lambda aux, children: ManifoldGeometry())
    jax.tree_util.register_pytree_node(
        CorewiseGeometry, lambda g: ((), None), lambda aux, children: CorewiseGeometry())

