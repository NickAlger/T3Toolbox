# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
from __future__ import annotations

import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.corewise as cw
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.tangent_operations as tangent_operations
import t3toolbox.backend.probing as probing
from t3toolbox.backend.common import *

__all__ = [
    'T3Tangent',
    'manifold_dim',
]


def manifold_dim(
        s,
) -> int:
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
    >>> dense_vv = np.stack([t3m.T3Tangent.randn(base, apply_gauge_projection=False).to_dense().reshape(-1)
    ...                      for _ in range(n_entries)])
    >>> ss = np.linalg.svd(dense_vv, compute_uv=False)
    >>> print(int(np.sum(ss > 1e-9 * ss[0])))   # number of nonzero singular values == manifold_dim
    29
    """
    shape = s[0]
    min_tucker_ranks, min_tt_ranks = t3.TuckerTensorTrain.get_minimal_ranks(s[0], s[1], s[2])

    num_cores = len(shape)
    assert(len(min_tucker_ranks) == num_cores)
    assert(len(min_tt_ranks) == num_cores+1)
    manifold_dim: int = 0
    for ii in range(num_cores):
        n = min_tucker_ranks[ii]
        rL = min_tt_ranks[ii]
        rR = min_tt_ranks[ii+1]
        if ii == num_cores-1:
            manifold_dim += rL * n * rR
        else:
            manifold_dim += (rL * n - rR) * rR

    for ii in range(num_cores):
        n = min_tucker_ranks[ii]
        N = shape[ii]
        manifold_dim += (N - n) * n

    return manifold_dim


@dataclass(frozen=True)
class T3Tangent:
    """Tangent vector to the manifold of fixed-rank Tucker tensor trains.

    A ``T3Tangent`` bundles a :py:class:`~t3toolbox.basis_variations_format.T3Basis` (the frame at
    the base point where the tangent space is attached) with a
    :py:class:`~t3toolbox.basis_variations_format.T3Variations` (the tangent direction in that
    frame). Bundling them makes "which tangent space" a checkable property: linear algebra between
    two tangent vectors is only defined when they live in the same tangent space, which here means
    they hold the **same** ``T3Basis`` object (identity, not merely numerically-equal cores).

    Validity caveats (NOT enforced):
        - :py:meth:`inner` and :py:meth:`norm` (and faithful corewise linear algebra) equal the
          Hilbert-Schmidt values only when the basis is **orthogonal** and the variations are
          **gauged**. These are not checked at construction. Use :py:meth:`is_orthogonal` and
          :py:meth:`is_gauged` to check, and see each operation's docstring for the failure mode.

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
        bvf.check_bv_pair(self.basis, self.variations)

    @ft.cached_property
    def d(self) -> int:
        return self.basis.d

    @ft.cached_property
    def shape(self) -> typ.Tuple[int, ...]:
        return self.basis.shape

    @ft.cached_property
    def base_stack_shape(self) -> typ.Tuple[int, ...]:
        """Base stack ``G``: the batch of base points, shared with the basis (``basis.stack_shape``)."""
        return self.basis.stack_shape

    @ft.cached_property
    def tangent_stack_shape(self) -> typ.Tuple[int, ...]:
        """Tangent stack ``V``: the extra *outer* batch of tangent vectors sharing this base.

        This is the part of the variation stack that exceeds the base stack (often empty). The
        variation cores are stacked as ``V + G + (core,)`` -- extra axes outermost, base stack inner.
        """
        full = self.variations.stack_shape
        return full[:len(full) - len(self.base_stack_shape)]

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int, ...]:
        """Full stack ``V + G`` (``tangent_stack_shape + base_stack_shape``), outer-to-inner."""
        return self.variations.stack_shape

    @ft.cached_property
    def structure(self):
        return self.basis.structure

    @ft.cached_property
    def data(self) -> typ.Tuple[bvf.T3Basis, bvf.T3Variations]:
        return self.basis, self.variations

    ############################################
    ##########    Conversions    ###############
    ############################################

    def to_dense(
            self,
            include_shift:  bool = False,  # False: tangent vector v. True: base point + v.
            use_jax:        bool = False,
    ) -> NDArray:  # shape=stack_shape+(N0,...,N(d-1))
        """Form the dense tensor represented by this tangent vector.

        The tangent vector is the sum of the 2d single-core-replacement terms (one per Tucker hole
        and one per TT hole). With ``include_shift=True``, the base point is added (base point + v).
        """
        return tangent_operations.tangent_to_dense(
            self.basis.data, self.variations.data, include_shift=include_shift, use_jax=use_jax,
        )

    def to_t3(
            self,
            include_shift:  bool = False,  # False: tangent vector v. True: base point + v.
            use_jax:        bool = False,
    ) -> t3.TuckerTensorTrain:  # doubled-rank Tucker tensor train
        """Doubled-rank :py:class:`TuckerTensorTrain` representation of this tangent vector.

        The Tucker and TT ranks are (roughly) doubled. With ``include_shift=True`` the result
        represents ``base point + v`` (the standard shifted embedding used by :py:meth:`retract`).

        This is the doubled-rank representation of Appendix A.3.1 (equations (50)-(53) and Figure 20)
        in Alger et al. (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).
        """
        cores = tangent_operations.tangent_to_t3(
            self.basis.data, self.variations.data, include_shift=include_shift, use_jax=use_jax,
        )
        return t3.TuckerTensorTrain(*cores)

    def retract(
            self,
            use_jax: bool = False,
    ) -> t3.TuckerTensorTrain:  # retracted Tucker tensor train (on the manifold)
        """Retract the tangent vector to the fixed-rank manifold.

        Forms the shifted doubled-rank embedding (base point + v) and truncates it back to the base
        ranks via T3-SVD, yielding a point on the manifold of the base point's ranks.

        The truncation is the implicit T3-SVD (Algorithm 10) of Alger et al. (2026),
        "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).
        """
        shifted = self.to_t3(include_shift=True, use_jax=use_jax)
        retracted_x, _, _ = shifted.t3svd(
            max_tucker_ranks=self.basis.up_ranks, max_tt_ranks=self.basis.left_ranks,
        )
        return retracted_x

    @staticmethod
    def zeros(
            basis:          bvf.T3Basis,
            stack_shape:    typ.Tuple[int, ...] = (),  # extra tangent stack V (a batch of tangents)
            use_jax:        bool = False,
    ) -> 'T3Tangent':
        """Zero tangent vector at a given basis.

        ``stack_shape`` is the extra *outer* tangent stack ``V`` (a batch of tangents sharing this
        base); the variation cores are stacked as ``V + G + (core,)``. Default ``V=()``.
        """
        xnp, _, _ = get_backend(False, use_jax)

        full_stack = stack_shape + basis.stack_shape  # V + G
        tucker_hole_shapes, tt_hole_shapes = basis.variation_shapes
        tucker_variations = tuple(xnp.zeros(full_stack + s) for s in tucker_hole_shapes)
        tt_variations = tuple(xnp.zeros(full_stack + s) for s in tt_hole_shapes)
        return T3Tangent(basis, bvf.T3Variations(tucker_variations, tt_variations))

    @staticmethod
    def randn(
            basis:                  bvf.T3Basis,
            stack_shape:            typ.Tuple[int, ...] = (),  # extra tangent stack V (a batch of tangents)
            apply_gauge_projection: bool = True,
            use_jax:                bool = False,
    ) -> 'T3Tangent':
        """Random tangent vector at a given basis.

        ``stack_shape`` is the extra *outer* tangent stack ``V`` (a batch of tangents sharing this
        base); the variation cores are stacked as ``V + G + (core,)``. Default ``V=()``.

        With ``apply_gauge_projection=True`` (default) the variations are gauged (via orthogonal
        projection); for an orthogonal, minimal-rank basis this makes the tangent vector a standard
        Gaussian on the tangent space. With ``apply_gauge_projection=False`` the variations are raw
        i.i.d. N(0, 1) cores (ungauged).
        """
        full_stack = stack_shape + basis.stack_shape  # V + G
        tucker_hole_shapes, tt_hole_shapes = basis.variation_shapes
        tucker_variations = tuple(randn(*(full_stack + s), use_jax=use_jax) for s in tucker_hole_shapes)
        tt_variations = tuple(randn(*(full_stack + s), use_jax=use_jax) for s in tt_hole_shapes)

        v = T3Tangent(basis, bvf.T3Variations(tucker_variations, tt_variations))
        if apply_gauge_projection:
            v = v.orthogonal_gauge_projection(use_jax=use_jax)
        return v

    @staticmethod
    def project(
            x:          t3.TuckerTensorTrain,
            basis:      bvf.T3Basis,
            use_jax:    bool = False,
    ) -> 'T3Tangent':
        """Orthogonal projection of a TuckerTensorTrain onto the tangent space at ``basis``.

        Returns the (gauged) tangent vector representing the orthogonal projection of
        ``x - (base point)`` onto the tangent space. Requires an orthogonal, minimal-rank ``basis``.
        """
        variations = tangent_operations.project_t3_onto_tangent_space(basis.data, x.data, use_jax=use_jax)
        return T3Tangent(basis, bvf.T3Variations(*variations))

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
                'matching stacks (same tangent stack V over the shared base stack G).\n'
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

    def inner(self, other: 'T3Tangent', use_jax: bool = False):
        """Inner product of two tangent vectors (corewise dot of the variations).

        Vectorized over the stack: returns an array of shape :py:attr:`stack_shape` (``V + G``), one
        inner product per stacked tangent (a scalar when unstacked). Requires the same T3Basis object
        and matching stacks.

        .. warning::
            This equals the Hilbert-Schmidt inner product of the represented tangent vectors only
            when the basis is orthogonal and BOTH variations are gauged (see :py:meth:`is_gauged`).
            Otherwise it is merely the corewise dot of the variation cores.

        The gauged identity ``<v, v'>_HS = sum_i <dU_i, dU_i'> + sum_i <dG_i, dG_i'>`` is given in
        Appendix A.3 of Alger et al. (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).
        """
        self._check_same_tangent_space(other)
        return cw.corewise_stack_dot(
            self.variations.data, other.variations.data, len(self.stack_shape), use_jax=use_jax,
        )

    def norm(self, use_jax: bool = False):
        """Norm of the tangent vector (corewise norm of the variations).

        Vectorized over the stack: returns an array of shape :py:attr:`stack_shape` (``V + G``), one
        norm per stacked tangent (a scalar when unstacked).

        .. warning::
            This equals the Hilbert-Schmidt norm only when the basis is orthogonal and the
            variations are gauged (see :py:meth:`is_gauged`).
        """
        xnp, _, _ = get_backend(False, use_jax)
        return xnp.sqrt(xnp.abs(self.inner(self, use_jax=use_jax)))

    ############################################
    ##########    Validity checkers    #########
    ############################################

    @ft.cached_property
    def has_minimal_ranks(self) -> bool:
        """True if this tangent's basis has minimal ranks. See :py:attr:`T3Basis.has_minimal_ranks`.

        .. note::
            Some tangent-space operations are only correct when the basis has minimal ranks (which
            exactly is TBD; flagged for later). Not enforced at construction.
        """
        return self.basis.has_minimal_ranks

    def is_orthogonal(self, atol: float = 1e-9) -> bool:
        """True if this tangent's basis is orthogonal. See :py:meth:`T3Basis.is_orthogonal`."""
        return self.basis.is_orthogonal(atol=atol)

    def is_gauged(self, atol: float = 1e-9) -> bool:
        """True if the variations are gauged with respect to the basis.

        Gauge conditions (needed for :py:meth:`inner`/:py:meth:`norm` to equal the Hilbert-Schmidt
        values; not enforced at construction):
            - ``einsum('...ia,...ja->...ij', U_i, V_i) = 0`` for all i (Tucker variations ⟂ U).
            - ``einsum('...abi,...abj->...ij', L_i, H_i) = 0`` for i = 0..d-2 (TT variations ⟂ L).

        These are the gauge conditions (48)-(49), Appendix A.3, of Alger et al. (2026),
        "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).
        """
        resid = 0.0
        for U, V in zip(self.basis.up_tucker_cores, self.variations.tucker_variations):
            g = np.einsum('...ia,...ja->...ij', np.asarray(U), np.asarray(V))
            resid = max(resid, float(np.max(np.abs(g))))
        for L, H in zip(self.basis.left_tt_cores[:-1], self.variations.tt_variations[:-1]):
            g = np.einsum('...abi,...abj->...ij', np.asarray(L), np.asarray(H))
            resid = max(resid, float(np.max(np.abs(g))))
        return resid <= atol

    ############################################
    ##########    Gauge projections    #########
    ############################################

    def orthogonal_gauge_projection(self, use_jax: bool = False) -> 'T3Tangent':
        """Gauge the variations via orthogonal projection (changes the tangent vector).

        Returns a tangent vector at the same basis whose variations satisfy the gauge conditions.
        Because it is an orthogonal projection of the variations, it represents a DIFFERENT tangent
        vector than ``self``. For the gauge-preserving variant see :py:meth:`oblique_gauge_projection`.
        """
        new_variations = tangent_operations.orthogonal_gauge_projection(
            self.basis.data, self.variations.data, use_jax=use_jax,
        )
        return T3Tangent(self.basis, bvf.T3Variations(*new_variations))

    def oblique_gauge_projection(self, use_jax: bool = False) -> 'T3Tangent':
        """Gauge the variations while preserving the represented tangent vector.

        Returns a tangent vector at the same basis representing the SAME vector as ``self`` but with
        gauged variations. When the basis is orthogonal with minimal ranks, corewise linear algebra
        on the gauged variations then faithfully matches the Hilbert-Schmidt operations, so
        :py:meth:`inner` / :py:meth:`norm` give the true HS values.
        """
        new_variations = tangent_operations.oblique_gauge_projection(
            self.basis.data, self.variations.data, use_jax=use_jax,
        )
        return T3Tangent(self.basis, bvf.T3Variations(*new_variations))

    ############################################
    ##########    Probing    ###################
    ############################################

    def probe(
            self,
            ww:         typ.Sequence[NDArray],  # probing vectors, len=d, elm_shape=F+(Ni,)
    ) -> typ.Sequence[NDArray]:                 # probes, len=d, elm_shape=F+G+(Ni,)
        """Probe this tangent vector: apply the single-sample least-squares Jacobian J^(s).

        Contracts the tangent vector with the probing vectors ``ww`` in all-but-one index, for each
        index -- the tangent analogue of :py:meth:`.TuckerTensorTrain.probe`. The probes are stacked
        ``F + G`` (probe stack ``F`` from ``ww`` outermost, base stack ``G`` innermost).

        This is the bare ``J^(s)`` (no gauge projector ``Pi``); for the Riemannian ``J = J^(s) o Pi``
        compose a gauge projection (e.g. :py:meth:`orthogonal_gauge_projection`) yourself.

        Requires ``tangent_stack_shape == ()`` (probing a *batch* of tangents needs 3-block
        contractions, not yet implemented).

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
        >>> print(zz[0].shape)             # F + G + (N0,) = (2,) + () + (10,)
        (2, 10)
        >>> zz2 = t3p.probe_dense(ww, v.to_dense())   # dense reference
        >>> print([float(np.linalg.norm(a - b)) for a, b in zip(zz, zz2)])
        [1.9485689247039e-12, 4.4498813137605194e-12, 3.528192267475046e-12]
        """
        if self.tangent_stack_shape != ():
            raise ValueError(
                'probe() does not support a tangent-stacked T3Tangent (tangent_stack_shape != ()).\n'
                'Probing a batch of tangent vectors needs 3-block contractions (not yet implemented).'
            )
        # probing's base order is exactly T3Basis.data = (up, down, left, right) -- no reorder.
        # numpy/jax dispatch is inferred from the input array types inside probing.
        return probing.probe_tangent(ww, self.variations.data, self.basis.data)

    @staticmethod
    def probe_transpose(
            ztildes:            typ.Sequence[NDArray],  # probe residuals, len=d, elm_shape=F+G+(Ni,)
            ww:                 typ.Sequence[NDArray],  # probing vectors, len=d, elm_shape=F+(Ni,)
            basis:              bvf.T3Basis,
            sum_over_probes:    bool = False,
    ) -> 'T3Tangent':
        """Apply the transpose ``(J^(s))^T`` of the probe map to residuals; returns a T3Tangent at ``basis``.

        The adjoint of :py:meth:`probe`. With ``sum_over_probes=False`` (default) the result is a
        *batch* of tangents -- a T3Tangent whose tangent stack ``V`` is the probe stack ``F`` (one
        tangent per probe residual; the ``F + G`` output is already in canonical ``V + G`` order, so
        no reorder is needed). With ``sum_over_probes=True`` the probe stack is summed and the result
        is a single tangent (``V = ()``) at ``basis`` -- the usual Gauss-Newton ``J^T r``.

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
        >>> v = t3m.T3Tangent.randn(base)
        >>> ww = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
        >>> z = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
        >>> Jv = v.probe(ww)
        >>> JTz = t3m.T3Tangent.probe_transpose(z, ww, base, sum_over_probes=True)
        >>> lhs = float(np.sum([np.sum(a * b) for a, b in zip(z, Jv)]))
        >>> print(bool(abs(lhs - float(JTz.inner(v))) < 1e-9))
        True

        Without summing, the result is a tangent-stacked T3Tangent (V = the probe stack):

        >>> JTz_batch = t3m.T3Tangent.probe_transpose(z, ww, base)  # sum_over_probes=False
        >>> print(JTz_batch.tangent_stack_shape, JTz_batch.base_stack_shape)
        (2,) ()
        """
        # probing's base order is exactly T3Basis.data = (up, down, left, right) -- no reorder.
        # numpy/jax dispatch is inferred from the input array types inside probing.
        dU_tildes, dG_tildes = probing.probe_tangent_transpose(
            ztildes, ww, basis.data, sum_over_probes=sum_over_probes,
        )
        return T3Tangent(basis, bvf.T3Variations(dU_tildes, dG_tildes))

    ############################################
    ##########    Stacking    ##################
    ############################################

    def unstack(self):
        """Unstack into an array-like tree of T3Tangents (tree shape = stack_shape)."""
        basis_tree = self.basis.unstack()
        variations_tree = self.variations.unstack()
        paired = stacking.tree_zip(basis_tree, variations_tree)
        return stacking.apply_func_to_leaf_subtrees(paired, lambda bv: T3Tangent(*bv), (None, None))

    @staticmethod
    def stack(xx) -> 'T3Tangent':
        """Stack an array-like tree of T3Tangents into one stacked T3Tangent."""
        basis_tree = stacking.apply_func_to_leaf_subtrees(xx, lambda t: t.basis, None)
        variations_tree = stacking.apply_func_to_leaf_subtrees(xx, lambda t: t.variations, None)
        return T3Tangent(bvf.T3Basis.stack(basis_tree), bvf.T3Variations.stack(variations_tree))

