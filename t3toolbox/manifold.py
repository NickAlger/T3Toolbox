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
from t3toolbox.backend.common import *

__all__ = [
    'T3Tangent',
    'manifold_dim',
]

# NOTE: the module-level tangent_* / *_gauge_projection / project_t3_onto_tangent_space /
# retract functions below are the pre-refactor implementations, pending port into T3Tangent
# methods + backend/tangent_operations.py (with stacking). They are not part of the public API.


####################################################################
##################    Tangent vectors backend  ##################
####################################################################

def manifold_dim(
        s,
) -> int:
    """Get the dimension of the fixed rank T3 manifold with a given structure.

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
    >>> import t3toolbox.basis_coordinates_format as bvf
    >>> import t3toolbox.orthogonalization as orth
    >>> s = ((5,6,3), (5,3,2), (2,2,4,1))
    >>> mdim = t3m.manifold_dim(s)
    >>> print(mdim)
    29
    >>> p = t3.t3_corewise_randn(s)
    >>> base, _ = orth.orthogonal_representations(p)
    >>> tucker_shapes, tt_shapes = bvf.get_base_hole_shapes(base)
    >>> num_tucker_entries = np.sum([np.prod(shape) for shape in tucker_shapes])
    >>> num_tt_entries = np.sum([np.prod(shape) for shape in tt_shapes])
    >>> num_core_entries = num_tucker_entries + num_tt_entries
    >>> print(num_core_entries)
    80
    >>> vv = [t3m.tangent_randn(base, apply_gauge_projection=False) for _ in range(num_core_entries)]
    >>> dense_vv = np.stack([t3m.tangent_to_dense(v, base) for v in vv])
    >>> _, ss, _ = np.linalg.svd(dense_vv.reshape((num_core_entries,-1)), full_matrices=False)
    >>> print(ss[mdim-1]) # last nonzero singular value
    2.8197268462367813
    >>> print(ss[mdim]) # first zero singular value
    1.1933078683104488e-14
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
    def stack_shape(self) -> typ.Tuple[int, ...]:
        return self.basis.stack_shape

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

    @staticmethod
    def zeros(
            basis:      bvf.T3Basis,
            use_jax:    bool = False,
    ) -> 'T3Tangent':
        """Zero tangent vector at a given basis."""
        xnp, _, _ = get_backend(False, use_jax)

        ss = basis.stack_shape
        tucker_hole_shapes, tt_hole_shapes = basis.variation_shapes
        tucker_variations = tuple(xnp.zeros(ss + s) for s in tucker_hole_shapes)
        tt_variations = tuple(xnp.zeros(ss + s) for s in tt_hole_shapes)
        return T3Tangent(basis, bvf.T3Variations(tucker_variations, tt_variations))

    @staticmethod
    def randn(
            basis:                  bvf.T3Basis,
            apply_gauge_projection: bool = True,
            use_jax:                bool = False,
    ) -> 'T3Tangent':
        """Random tangent vector at a given basis.

        With ``apply_gauge_projection=True`` (default) the variations are gauged (via orthogonal
        projection); for an orthogonal, minimal-rank basis this makes the tangent vector a standard
        Gaussian on the tangent space. With ``apply_gauge_projection=False`` the variations are raw
        i.i.d. N(0, 1) cores (ungauged).
        """
        ss = basis.stack_shape
        tucker_hole_shapes, tt_hole_shapes = basis.variation_shapes
        tucker_variations = tuple(randn(*(ss + s), use_jax=use_jax) for s in tucker_hole_shapes)
        tt_variations = tuple(randn(*(ss + s), use_jax=use_jax) for s in tt_hole_shapes)

        v = T3Tangent(basis, bvf.T3Variations(tucker_variations, tt_variations))
        if apply_gauge_projection:
            v = v.orthogonal_gauge_projection(use_jax=use_jax)
        return v

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

        .. warning::
            This equals the Hilbert-Schmidt inner product of the represented tangent vectors only
            when the basis is orthogonal and BOTH variations are gauged (see :py:meth:`is_gauged`).
            Otherwise it is merely the corewise dot of the variation cores. Requires the same
            T3Basis object.
        """
        self._check_same_tangent_space(other)
        return cw.corewise_dot(self.variations.data, other.variations.data, use_jax=use_jax)

    def norm(self, use_jax: bool = False):
        """Norm of the tangent vector (corewise norm of the variations).

        .. warning::
            This equals the Hilbert-Schmidt norm only when the basis is orthogonal and the
            variations are gauged (see :py:meth:`is_gauged`).
        """
        return cw.corewise_norm(self.variations.data, use_jax=use_jax)

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


def tangent_to_dense(
        variation: bvf.T3Variation,
        base: bvf.T3Base,
        include_shift: bool = False, # False: V. True: P+V. P=base point, V=tangent vector. Must supply "rep"
        use_jax: bool=False,
) -> NDArray:
    """Convert Tangent vector to Tucker tensor train manifold into dense tensor.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.basis_coordinates_format as bvf
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> v_dense = t3m.tangent_to_dense(variation, base) # Convert tangent to dense
    >>> ((U0,U1,U2), (L0,L1,L2), (R0,R1,R2), (O0,O1,O2)) = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = variation
    >>> s1 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,H0,R1,R2)
    >>> s2 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,L0,H1,R2)
    >>> s3 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,U2,L0,L1,H2)
    >>> s4 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', V0,U1,U2,O0,R1,R2)
    >>> s5 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,V1,U2,L0,O1,R2)
    >>> s6 = np.einsum('ai,bj,ck,xay,ybz,zcw->ijk', U0,U1,V2,L0,L1,O2)
    >>> v_dense2 = s1 + s2 + s3 + s4 + s5 + s6
    >>> print(np.linalg.norm(v_dense - v_dense2))
    1.2760924630140578e-14
    >>> p_plus_v_dense = t3m.tangent_to_dense(variation, base, include_shift=True) # Convert shifted tangent, p+v, to dense
    >>> p_plus_v_dense2 =  t3.t3_to_dense(p) + v_dense
    >>> print(np.linalg.norm(p_plus_v_dense - p_plus_v_dense2))
    1.2677102046134292e-12
    """
    is_uniform = not isinstance(variation[0], typ.Sequence)
    xnp, _, _ = get_backend(is_uniform, use_jax)

    #
    num_cores = len(variation[0])
    tucker_terms = [bvf.ith_bv_to_t3(ii, False, base, variation) for ii in range(num_cores)]
    tt_terms     = [bvf.ith_bv_to_t3(ii, True, base, variation) for ii in range(num_cores)]
    terms = tucker_terms + tt_terms
    V = t3.t3_to_dense(terms[0])
    for t in terms[1:]:
        V = V + t3.t3_to_dense(t, use_jax=use_jax)

    if include_shift:
        tucker_cores, left_tt_cores, _, _ = base
        P = t3.t3_to_dense((tucker_cores, left_tt_cores))
        X = P + V
    else:
        X = V

    return X


def tangent_to_t3(
        variation: bvf.T3Variation,
        base: bvf.T3Base,
        include_shift: bool = False,  # False: v. True: p+v. p=base point, v=tangent vector
        use_jax: bool=False,
) -> t3.TuckerTensorTrain:
    '''Rank 2r Tucker tensor train representation of tangent vector.

    Without shift, we use the formula::

        v(x,y,z,w) = ([dU1(B x) L1(B x)]) ([R2(B y)        0]) ([R3(B z)        0]) ([R4(B w) ])
                     (                  ) ([dU2(B y) L2(B y)]) ([dU3(B z) L3(B z)]) ([dU4(B w)])
                     (         +        ) (         +        ) (        +         ) (    +     )
                     ([O1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
                     (                  ) ([O2(dB y)       0]) ([O3(dB z)       0]) ([O4(dB w)])

    With shift is same as unshifted, except last backend modified as follows::

        [R4(B w) ]                  [R4(B w)           ]
        [dU4(B w)]                  [L4(B w) + dU4(B w)]
            +             ->            +
        [0       ]                  [0                 ]
        [O4(dB w)]                  [O4(dB w)          ]

    NOTE: can modify other cores with their shifts instead, if desired

    Parameters
    ----------
    variation: T3Variation,
        Variation representing the tangent vector
    base: T3Base,
        Representation of the base point at which the tangent space attaches to the manifold.
    include_shift: bool
        If False, return tangent vector v only. If True, shift tangent vector so it is attached at the base point, p+v.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    TuckerTensorTrain
        Tucker tensor train representation of tangent vector, which has doubled ranks

    See Also
    --------
    T3Base
    T3Variation
    TuckerTensorTrain

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> v_t3 = t3m.tangent_to_t3(variation, base) # tangent vector only (attached at zero)
    >>> v_dense = t3.t3_to_dense(v_t3)
    >>> v_dense2 = t3m.tangent_to_dense(variation, base)
    >>> print(np.linalg.norm(v_dense - v_dense2))
    2.678565538404836e-15
    >>> p_plus_v_t3 = t3m.tangent_to_t3(variation, base, include_shift=True) # shifted tangent vector (include attachment at base point)
    >>> p_plus_v_dense = t3.t3_to_dense(p_plus_v_t3)
    >>> p_plus_v_dense2 = v_dense2 + t3.t3_to_dense(p)
    >>> print(np.linalg.norm(p_plus_v_dense - p_plus_v_dense2))
    1.2102169224182523e-12
    '''
    is_uniform = not isinstance(variation[0], typ.Sequence)
    xnp, _, _ = get_backend(is_uniform, use_jax)

    #
    tucker_vars, tt_vars = variation
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base

    num_cores = len(tucker_cores)

    x_tucker_cores = []
    for B, dB in zip(tucker_cores, tucker_vars):
        B2 = xnp.concatenate([B, dB], axis=0)
        x_tucker_cores.append(B2)

    x_tt_cores = []

    dU = tt_vars[0]
    O = outer_tt_cores[0]
    L = left_tt_cores[0]
    Z = xnp.zeros((O.shape[0], O.shape[1], L.shape[2]))
    G_top = xnp.concatenate([dU, L], axis=2)
    G_bot = xnp.concatenate([O, Z], axis=2)
    G = xnp.concatenate([G_top, G_bot], axis=1)
    x_tt_cores.append(G)

    for ii in range(1, num_cores-1):
        L = left_tt_cores[ii]
        R = right_tt_cores[ii]
        O = outer_tt_cores[ii]
        dU = tt_vars[ii]
        Z001 = xnp.zeros((R.shape[0], dU.shape[1], L.shape[2]))
        Z100 = xnp.zeros((R.shape[0], O.shape[1], R.shape[2]))
        Z101 = xnp.zeros((R.shape[0], O.shape[1], L.shape[2])) #Z001
        Z111 = xnp.zeros((L.shape[0], O.shape[1], L.shape[2])) #jnp.zeros(L.shape)
        G_top = xnp.concatenate([
            xnp.concatenate([R, Z001], axis=2),
            xnp.concatenate([dU, L], axis=2)
        ], axis=0)
        G_bot = xnp.concatenate([
            xnp.concatenate([Z100, Z101], axis=2),
            xnp.concatenate([O, Z111], axis=2)
        ], axis=0)
        G = xnp.concatenate([G_top, G_bot], axis=1)
        x_tt_cores.append(G)

    dU = tt_vars[-1]
    R = right_tt_cores[-1]
    O = outer_tt_cores[-1]
    Z = xnp.zeros((R.shape[0], O.shape[1], R.shape[2]))
    if include_shift:
        Lf = left_tt_cores[-1]
        G_top = xnp.concatenate([R, Lf + dU], axis=0)
    else:
        G_top = xnp.concatenate([R, dU], axis=0)
    G_bot = xnp.concatenate([Z, O], axis=0)
    G = xnp.concatenate([G_top, G_bot], axis=1)
    x_tt_cores.append(G)

    return tuple(x_tucker_cores), tuple(x_tt_cores)


def tangent_zeros(
        base: bvf.T3Base, # orthogonal base
        use_jax: bool=False,
) -> bvf.T3Variation:
    """Construct the zero vector in a Tucker tensor train tangent space.

    Parameters
    ----------
    base: T3Base
        Representations of base point on manifold where tangent space is attached
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Variation
        Variation representing the zero vector in the tangent space

    See Also
    --------
    t3tangent_randn

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> z = t3m.tangent_zeros(base)
    >>> print(np.linalg.norm(t3m.tangent_to_dense(z, base)))
    0.0
    """
    is_uniform = not isinstance(base[0], typ.Sequence)
    xnp, _, _ = get_backend(is_uniform, use_jax)

    #
    var_tucker_shapes, var_tt_shapes = bvf.get_base_hole_shapes(base)

    tucker_vars = tuple([xnp.zeros(s) for s in var_tucker_shapes])
    tt_vars = tuple([xnp.zeros(s) for s in var_tt_shapes])

    zero_variation = (tucker_vars, tt_vars)
    return zero_variation


def tangent_randn(
        base:   typ.Union[bvf.T3Base,               ut3.UniformT3Base], # orthogonal base
        masks:  typ.Union[ut3.UniformEdgeWeights,   ut3.UniformBVEdgeWeights] = (None, None, None),
        apply_gauge_projection: bool = True,
        randn: typ.Callable[..., NDArray] = np.random.randn,
        use_jax: bool = False,
) -> typ.Union[bvf.T3Variation, ut3.UniformT3Variation]:
    """Draw a random T3Variation.

    Parameters
    ----------
    orthogonal_base: T3Base
        Representations of base point on manifold where tangent space is attached.
    randn: typ.Callable[[..., NDArray]
        Function for creating random arrays. Arguments are a sequence of ints defining the shape of the array.
        Default: np.random.randn (numpy)

    Returns
    -------
    T3Tangent
        Random tangent vector. If base is orthogonal, ranks are minimal, and gauge projection is applied,
        then the random tangent vector is distributed according to
        a standard multivariate distribution on the tangent space.
    use_jax: bool
        If True, return jax arrays, if False return numpy. Should update this to use pure jax, rather than converting numpy->jax.
    apply_gauge_projection: bool
        Default: True. If False, gauge projection is not applied and vector is not i.i.d. N(0,1) on the tangent space

    See Also
    --------
    t3tangent_zeros

    Examples
    --------

    Apply Gauge projection (default):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, vars0 = orth.orthogonal_representations(p)
    >>> v = t3m.tangent_randn(base) # Random tangent vector, gauged.

    Don't apply Gauge projection:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, vars0 = orth.orthogonal_representations(p)
    >>> v = t3m.tangent_randn(base, apply_gauge_projection=False) # Random tangent vector, ungauged
    """
    is_uniform = not isinstance(base[0], typ.Sequence)

    if is_uniform:
        var_tucker_shape, var_tt_shape = ut3.get_uniform_base_hole_shapes(base)
        var_tucker_supercore = randn(*var_tucker_shape, use_jax=use_jax)
        var_tt_supercore = randn(*var_tt_shape, use_jax=use_jax)

        variation = (var_tucker_supercore, var_tt_supercore)
        if masks is not None:
            variation = ut3.apply_masks_to_variation(variation, masks, use_jax=use_jax)

        if apply_gauge_projection:
            variation = orthogonal_gauge_projection(variation, base, use_jax=use_jax)
    else:
        var_tucker_shapes, var_tt_shapes = bvf.get_base_hole_shapes(base)

        tucker_vars0 = tuple([randn(*s) for s in var_tucker_shapes])
        tt_vars0 = tuple([randn(*s) for s in var_tt_shapes])

        variation = (tucker_vars0, tt_vars0)
        if apply_gauge_projection:
            variation = orthogonal_gauge_projection(variation, base, use_jax=use_jax)

    return variation


####################################################################
#################    Projection and retraction   ###################
####################################################################

def orthogonal_gauge_projection(
        variation:          typ.Union[bvf.T3Variation,  ut3.UniformT3Variation],
        orthogonal_base:    typ.Union[bvf.T3Base,       ut3.UniformT3Base],
        use_jax: bool = False,
) -> typ.Union[bvf.T3Variation, ut3.UniformT3Variation]:
    """Makes tangent variation gauged via orthogonal projection. Changes tangent vector.

    Gauge condition:
        - All variation Tucker cores Vi are orthogonal to the corresponding base Tucker cores Ui:
            Ui @ Vi.T = 0    for    i=1,...,d
        - All but the last variation TT-cores H are left-perpendicular to the corresponding base left TT-cores L:
            einsum('iaj,iak->jk', Hi, Li) = 0    for    i=1,...,d-1

    Parameters
    ----------
    variation: T3Variation,
        The variation which will become gauged.
    orthogonal_base: T3Base,
        The base representations. Must be orthogonal for the operation to work properly.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Variation
        Projected variation satisfying Gauge condition.
        Represents different tangent vector than original variation.

    See Also
    --------
    T3Base
    T3Variation
    t3_oblique_gauge_projection

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.common as common
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.corewise as cw
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base, apply_gauge_projection=False)
    >>> proj_variation = t3m.orthogonal_gauge_projection(variation, base) # Make gauged via orthogonal projection
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = proj_variation
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for tucker backend 1
    3.512073125137391e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-backend 1
    1.5807940730805242e-15
    >>> v_minus_p_dot_p = cw.corewise_dot(cw.corewise_sub(variation, proj_variation), proj_variation)
    >>> print(v_minus_p_dot_p) # Projection is orthogonal w.r.t. corewise dot
    -4.995303314442243e-18

    Uniform example:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.common as common
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.uniform as ut3
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, dummy_var = orth.orthogonal_representations(p)
    >>> _, uniform_base, masks = ut3.bv_to_ubv(dummy_var, base)
    >>> uniform_var = t3m.tangent_randn(uniform_base, masks=masks, apply_gauge_projection=False)
    >>> proj_var = t3m.orthogonal_gauge_projection(uniform_var, uniform_base)
    >>> UU, LL, RR, OO = uniform_base
    >>> proj_tucker_var, proj_tt_var = proj_var
    >>> print(np.linalg.norm(np.einsum('dio,djo->dij', proj_tucker_var, UU)))
    6.860678066865219e-15
    >>> print(np.linalg.norm(np.einsum('diaj,diak->djk', proj_tt_var[:-1], LL[:-1]))) # first var tt cores are left-orthogonal to base
    2.0607190172353126e-15
    >>> ip = cw.corewise_dot(cw.corewise_sub(uniform_var, proj_var), proj_var)
    >>> print(ip) # Projection is orthogonal w.r.t. corewise dot
    4.496403249731884e-14
    """
    is_uniform = not isinstance(orthogonal_base[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    tucker_vars, tt_vars = variation

    if is_uniform:
        first_dV2 = tt_vars[:-1] - xnp.einsum(
            'diaj,djk->diak',
            left_tt_cores[:-1],
            xnp.einsum('diaj,diak->djk', left_tt_cores[:-1], tt_vars[:-1])
        )
        last_dV2 = tt_vars[-1]
        new_tt_variations = xnp.concatenate([first_dV2, last_dV2.reshape((1,)+last_dV2.shape)], axis=0)

        new_tucker_variations = tucker_vars - xnp.einsum(
            'dio,dij->djo',
            tucker_cores,
            xnp.einsum('dio,djo->dij', tucker_cores, tucker_vars)
        )

    else:
        new_tt_variations = []
        for dV, P in zip(tt_vars[:-1], left_tt_cores[:-1]):
            dV2 = dV - xnp.einsum('iaj,jk->iak', P, xnp.einsum('iaj,iak->jk', P, dV))
            new_tt_variations.append(dV2)
        new_tt_variations.append(tt_vars[-1])

        new_tucker_variations = []
        for dB, B in zip(tucker_vars, tucker_cores):
            dB2 = dB - (dB @ B.T) @ B
            new_tucker_variations.append(dB2)

        new_tucker_variations = tuple(new_tucker_variations)
        new_tt_variations = tuple(new_tt_variations)

    return new_tucker_variations, new_tt_variations


def oblique_gauge_projection(
        variation: bvf.T3Variation,
        orthogonal_base: bvf.T3Base,
        use_jax: bool = False,
) -> bvf.T3Variation:
    """Makes variations left-perpendicular while preserving tangent vector.

    Straightforward generalization of the method from:
        Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider.
        "On manifolds of tensors of fixed TT-rank." Numerische Mathematik 120.4 (2012): 701-731.

    Parameters
    ----------
    variation: T3Variation,
        The variation that we wish to make gauged
    orthogonal_base: T3Base,
        Orthogonal representations of the base point on the manifold.
        If non-orthogonal, this method doesn't work properly.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Variation
        Projected variation satisfying Gauge condition.
        Represents the same tangent vector as the original variation.

    See Also
    --------
    T3Base
    T3Variation
    orthogonal_gauge_projection

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> proj_variation = t3m.oblique_gauge_projection(variation, base) # Make gauged via oblique projection
    >>> v_dense = t3m.tangent_to_dense(variation, base)
    >>> proj_v_dense = t3m.tangent_to_dense(proj_variation, base)
    >>> print(np.linalg.norm(v_dense - proj_v_dense)) # Zero since projection preserves represented tangent vector
    3.4398319441148304e-15
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = proj_variation
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for Tucker backend 1
    2.931519226677228e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-backend 1
    6.99005312491287e-16

    With minimal ranks, orthogonal bases, and gauged variations, the corewise dot product faithfully represents
    the Hilbert-Schmidt inner product on the ambient space:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.common
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.corewise as cw
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> u = t3m.tangent_randn(base, apply_gauge_projection=False)
    >>> v = t3m.tangent_randn(base, apply_gauge_projection=False)
    >>> bad_u_inner_v = cw.corewise_dot(u, v) # u and v are ungauged, so this will not give the right answer
    >>> u_dense = t3m.tangent_to_dense(u, base)
    >>> v_dense = t3m.tangent_to_dense(v, base)
    >>> u_inner_v_true = np.sum(u_dense * v_dense)
    >>> print(np.abs(bad_u_inner_v - u_inner_v_true)) # error nonzero because we didn't respect gauge
    6.21838915941413
    >>> u_gauged = t3m.oblique_gauge_projection(u, base) # make them gauged and try again
    >>> v_gauged = t3m.oblique_gauge_projection(v, base)
    >>> u_inner_v = cw.corewise_dot(u_gauged, v_gauged)
    >>> print(np.abs(u_inner_v - u_inner_v_true)) # Now the error is numerical zero
    0.0
    """
    is_uniform = not isinstance(variation[0], typ.Sequence)
    xnp, _, _ = get_backend(is_uniform, use_jax)

    #
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    tucker_vars, tt_vars = variation
    num_cores = len(tucker_cores)

    tt_vars = list(tt_vars)
    tucker_vars = list(tucker_vars)

    # Make Tucker variations left-perpendicular
    for ii in range(num_cores):
        B_io = tucker_cores[ii]
        dB_jo = tucker_vars[ii]
        O_aib = outer_tt_cores[ii]
        dG_ajb = tt_vars[ii]

        X_ji = dB_jo @ B_io.T
        dB_parallel_jo = X_ji @ B_io
        dB2_jo = dB_jo - dB_parallel_jo # dB_perp
        dG2_ajb = dG_ajb + xnp.einsum('aib,ij->ajb', O_aib, X_ji)

        tt_vars[ii] = dG2_ajb
        tucker_vars[ii] = dB2_jo

    # Make tt cores left-perpendicular
    for ii in range(num_cores-1):
        dG1 = tt_vars[ii]
        dG2 = tt_vars[ii+1]

        L1 = left_tt_cores[ii]
        R2 = right_tt_cores[ii+1]
        X = xnp.einsum('iaj,iak->jk', L1, dG1)
        new_dV1 = dG1 - xnp.einsum('iaj,jk->iak', L1, X)
        new_dV2 = dG2 + xnp.einsum('jk,kbl->jbl', X, R2)

        tt_vars[ii] = new_dV1
        tt_vars[ii+1] = new_dV2

    return tuple(tucker_vars), tuple(tt_vars)


def tt_zipper_left_to_right(
        coresA: typ.Union[typ.Sequence[NDArray], NDArray],
        coresB: typ.Union[typ.Sequence[NDArray], NDArray],
        use_jax: bool = False,
) -> typ.Union[typ.Tuple[NDArray, ...], NDArray]:  # zipper_matrices. len=d+1
    is_uniform = not isinstance(coresA, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    def _func(Z, x):
        GA, GB = x
        Z_next = xnp.einsum('ij,iak,jal->kl', Z, GA, GB)
        return Z_next, (Z,)

    # Z0 = xnp.array([[1.0]])
    Z0 = xnp.ones((coresA[0].shape[0], coresB[0].shape[0]))
    Zf, (ZZ_first,) = xscan(_func, Z0, (coresA, coresB))

    if is_uniform:
        zipper_matrices = xnp.concatenate([ZZ_first, Zf.reshape((1,)+Zf.shape)], axis=0)
    else:
        zipper_matrices = tuple(ZZ_first) + (Zf,)

    return zipper_matrices


def tt_zipper_right_to_left(
        coresA: typ.Union[typ.Sequence[NDArray], NDArray],
        coresB: typ.Union[typ.Sequence[NDArray], NDArray],
        use_jax: bool = False,
) -> typ.Union[typ.Sequence[NDArray], NDArray]:  # zipper_matrices. len=d+1
    return tt_zipper_left_to_right(t3.reverse_tt(coresA), t3.reverse_tt(coresB), use_jax=use_jax)[::-1]


def project_t3_onto_tangent_space(
        x:                  typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain], # Tucker tensor train to be projected
        orthogonal_base:    typ.Union[bvf.T3Base,           ut3.UniformT3Base], # Orthogonal representations of base point
        use_jax: bool = False,
) -> bvf.T3Variation:
    """Projects TuckerTensorTrain onto tangent space to the manifold of fixed rank TuckerTensorTrains.

    Parameters
    ----------
    x: t3.TuckerTensorTrain
        TuckerTensorTrain to project
    orthogonal_base: T3Base
        Minimal rank orthogonal representations of base point on manifold where tangent space is attached
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Variation
        Gauged variation representing the orthogonal projection of x onto the tangent space.

    See Also
    --------
    T3Base
    oblique_gauge_projection
    orthogonal_gauge_projection

    ADD UNIFORM EXAMPLE/TEST

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (5,3,2,4)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> x = t3.t3_corewise_randn(((14,15,16), (7,4,8), (3,5,4,2)))
    >>> proj_x = t3m.project_t3_onto_tangent_space(x, base) # Project x onto tangent space
    >>> P = t3.t3_to_dense(p)
    >>> X = t3.t3_to_dense(x)
    >>> proj_X = t3m.tangent_to_dense(proj_x, base)
    >>> print(np.sum((X - proj_X) * (proj_X - P)) / np.sum(X)) # Check that x was projected orthogonally
    -2.7295025395842007e-13

    Uniform example: DOESNT WORK YET

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.uniform_manifold as utm
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.t3svd as t3svd
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (5,3,2,4)))
    # >>> p = t3.t3_corewise_randn(((15,15,15), (5,5,5), (1,3,3,1)))
    >>> p, _, _ = t3svd.t3svd(p)
    >>> base, dummy_var = orth.orthogonal_representations(p)
    >>> x = t3.t3_corewise_randn(((14,15,16), (7,4,8), (3,5,4,2)))
    # >>> x = t3.t3_corewise_randn(((15,15,15), (5,5,5), (1,3,3,1)))
import t3toolbox.backend.uniform_tucker_tensor_train.ut3_conversions    >>> x, _, _ = t3svd.t3svd(x)
    >>> proj_x = t3m.project_t3_onto_tangent_space(x, base) # Project x onto tangent space
    >>> dense_proj_x = t3m.tangent_to_dense(proj_x, base)
    >>> _, uniform_base, bv_mask = ut3.bv_to_ubv(dummy_var, base)
    >>> uniform_x, x_mask = t3toolbox.backend.uniform_tucker_tensor_train.ut3_conversions.t3_to_ut3(x)
    >>> uniform_proj_x = t3m.project_t3_onto_tangent_space(uniform_x, uniform_base) # Project x onto tangent space
    >>> dense_uniform_proj_x = utm.uniform_tangent_to_dense(uniform_proj_x, uniform_base, bv_mask)
    >>> print(np.linalg.norm(dense_uniform_proj_x - dense_proj_x))
    """
    is_uniform = not isinstance(orthogonal_base[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        x = ut3.uniform_squash_tails(x)
    else:
        x = t3.squash_tails(x)

    up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    other_tucker_cores, other_tt_cores = x

    if is_uniform:
    # if False:
        other_tt_cores2 = xnp.einsum(
            'daib,dix->daxb',
            other_tt_cores,
            xnp.einsum('diz,dxz->dix', other_tucker_cores, up_tucker_cores)
        )
    else:
        def _func1(args):
            G_other, B_other, U = args
            # G_other2 = xnp.einsum('aib,ix->axb', G_other, B_other @ U.T)
            G_other2 = xnp.einsum(
                'aib,ix->axb',
                G_other,
                xnp.einsum('iz,xz->ix', B_other, U)
            )
            return (G_other2,)

        (other_tt_cores2,) = xmap(_func1, (other_tt_cores, other_tucker_cores, up_tucker_cores))

    zipper_left2right = tt_zipper_left_to_right(other_tt_cores2[:-1], left_tt_cores[:-1], use_jax=use_jax)
    zipper_right2left = tt_zipper_right_to_left(other_tt_cores2[1:], right_tt_cores[1:], use_jax=use_jax)

    if is_uniform:
    # if False:
        ZL_ax, ZR_by, G_aib, B_io, O_xjy, U_jo = (
            zipper_left2right, zipper_right2left,
            other_tt_cores, other_tucker_cores,
            outer_tt_cores, up_tucker_cores,
        )
        X_xiy = xnp.einsum('dax,daib,dby->dxiy', ZL_ax, G_aib, ZR_by)
        dG_xjy = xnp.einsum(
            'dxiy,dij->dxjy',
            X_xiy,
            xnp.einsum('dio,djo->dij', B_io, U_jo)
        )
        M_ij = xnp.einsum('dxiy,dxjy->dij', X_xiy, O_xjy)
        dB_jo = xnp.einsum('dij,dio->djo', M_ij, B_io)
        ungauged_tt_variations = dG_xjy
        ungauged_tucker_variations = dB_jo
    else:
        def _func2(args):
            ZL_ax, ZR_by, G_aib, B_io, O_xjy, U_jo = args
            X_xiy = xnp.einsum('ax,aib,by->xiy', ZL_ax, G_aib, ZR_by)
            # dG_xjy = xnp.einsum('xiy,ij->xjy', X_xiy, B_io @ U_jo.T)
            dG_xjy = xnp.einsum(
                'xiy,ij->xjy',
                X_xiy,
                xnp.einsum('io,jo->ij', B_io, U_jo)
            )
            M_ij = xnp.einsum('xiy,xjy->ij', X_xiy, O_xjy)
            dB_jo = xnp.einsum('ij,io->jo', M_ij, B_io)
            return dG_xjy, dB_jo

        ungauged_tt_variations, ungauged_tucker_variations = xmap(
            _func2,
            (zipper_left2right, zipper_right2left,
             other_tt_cores, other_tucker_cores,
             outer_tt_cores, up_tucker_cores)
        )

    ungauged_u = (ungauged_tucker_variations, ungauged_tt_variations)
    gauged_u = orthogonal_gauge_projection(ungauged_u, orthogonal_base)
    return gauged_u


def retract(
        variation: bvf.T3Variation,
        base: bvf.T3Base,
        use_jax: bool = False,
) -> t3.TuckerTensorTrain: # retracted Tucker tensor train
    """Retract Tucker tensor train tangent vector to manifold.

    Parameters
    ----------
    variation: T3Variation,
        Variation representing the tangent vector we wish to retract to the manifold
    base: T3Base,
        Representation of the base point on the manifold where the tangent space is attached.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    t3.TuckerTensorTrain
        Retraction of tangent vector onto the manifold.

    See Also
    --------
    T3Base
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.common
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.corewise as cw
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base) # Random tangent vector
    >>> ret_v = t3m.retract(variation, base) # Retract tangent vector to manifold
    >>> ret_V = t3.t3_to_dense(ret_v)
    >>> V = t3m.tangent_to_dense(variation, base, include_shift=True)
    >>> print(np.linalg.norm(ret_V - V)) # vector changes
    0.14335564543255402
    >>> v2 = cw.corewise_scale(variation, 1e-2) # make the tangent vector shorter for smaller retraction
    >>> ret_v2 = t3m.retract(v2, base)
    >>> ret_V2 = t3.t3_to_dense(ret_v2)
    >>> V2 = t3m.tangent_to_dense(v2, base, include_shift=True)
    >>> print(np.linalg.norm(ret_V2 - V2)) # vector changes
    4.9488133126395654e-05
    """
    tucker_cores, left_tt_cores, _, _ = base
    _, base_tucker_ranks, base_tt_ranks = t3.get_structure((tucker_cores, left_tt_cores))

    x_t3 = tangent_to_t3(variation, base, include_shift=True, use_jax=use_jax)
    retracted_x_t3, _, _ = t3svd.t3svd(
        x_t3,
        max_tt_ranks = base_tt_ranks,
        max_tucker_ranks = base_tucker_ranks,
        use_jax=use_jax,
    )
    return retracted_x_t3

