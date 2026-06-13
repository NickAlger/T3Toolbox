# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.bv_conversions as bv_conversions
import t3toolbox.backend.t3_operations as t3_operations
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.orthogonal_representations as orth_reps
from t3toolbox.backend.common import *


__all__ = [
    'T3Basis',
    'T3Variations',
    'bv_to_t3',
    't3_orthogonal_representations',
]


@dataclass(frozen=True, eq=False)  # eq=False -> identity __hash__/__eq__, so a T3Basis can be
class T3Basis:                     # jax aux_data (it holds arrays; value hash/eq is impossible).
    """Basis for basis-variations representation of TuckerTensorTrains

    Often, one works with TuckerTensorTrains of the following forms::

        1--(H0)--R1---R2---1    1---L0--(H1)--R2---1    1---L0---L1--(H2)--1
            |    |    |             |    |    |             |    |    |
            U0   U1   U2            U0   U1   U2            U0   U1   U2
            |    |    |             |    |    |             |    |    |

        1---D0---R1---R2---1    1---L0---D1---R2---1    1---L0---L1---D2---1
            |    |    |             |    |    |             |    |    |
           (V0)  U1   U2            U0  (V1)  U2            U0   U1  (V2)
            |    |    |             |    |    |             |    |    |

    In each of these, there is a special "variation" core, indicated by parentheses (X), surrounded by "basis" cores.

    The components of T3Basis are the "basis cores":
        - up_tucker_cores   = (U0, ..., U(d-1)), elm_shape=(nUi, Ni)
        - down_tt_cores     = (D0, ..., D(d-1)), elm_shape=(rLi, nDi, rR(i+1))
        - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))
        - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))

    The components of T3Variations are the "variation cores":
        - tucker_variations = (V0, ..., V(d-1)), elm_shape=(nDi, Ni)
        - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, nUi, rRi)

    Note that Ld and R0 are not used in these diagrams.

    The edge ranks are shown in the following diagrams::

           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ----- (H1) ----- ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nU0     | nU1                | nU(d-1)
                 |         |                    |
                 U0        U1                   Ud
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |

    and::

           rL0       rL1       rR2      rR(d-1)         rRd
        1 ------ L0 ------ D1 ------ ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nU0     | nO1                | nU(d-1)
                 |         |                    |
                 U0       (V1)                   Ud
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |


    A tangent vector can be written as the sum of all the tensor diagrams above.
    In this case, the basis cores are representations of the point where the
    tangent space attaches to the manifold, and the variation cores define the
    tangent vector with respect to the basis.

    Often, it is desirable for the base cores to be **orthogonal** as follows:
        - up_tucker_cores   = (U0,...,U(d-1)), orthogonal:       U_ia U_ja = delta_ij
        - down_tt_cores     = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij
        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij
        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal  R_ibc R_jbc = delta_ij

    Often, it is desirable for the variations to satisfy the following **Gauge conditions**:
        - U_ia V_ja = 0    (all V)
        - L_abi H_abj = 0  (all but the last H)

    If these conditions are satisfied, then one can do "dumb" corewise linear algebra
    (add, scale, dot product, etc) with the variations, and those core faithfully correspond
    to linear algebra with the N1 x ... x Nd tangent vectors represented by the variations.

    The orthogonal representations (45)-(46) and gauge conditions (48)-(49) are defined in Appendix
    A.3 of Alger, Christierson, Chen & Ghattas (2026), "Tucker Tensor Train Taylor Series"
    (arXiv:2603.21141).

    See Also
    --------
    T3Variations
    check_t3_base
    orthogonal_representations
    oblique_gauge_projection

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.basis_variations_format as bvf
    >>> ss = (2,3)
    >>> up_tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
    >>> down_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
    >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3,12,5)))
    >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
    >>> basis = bvf.T3Basis(up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
    >>> print(basis.structure)
    ((14, 15, 16), (10, 11, 12), (1, 2, 3, 5), (2, 4, 5, 1), (9, 8, 7), (2, 3))
    >>> print(basis.variation_shapes)
    (((9, 14), (8, 15), (7, 16)), ((1, 10, 4), (2, 11, 5), (3, 12, 1)))
    """
    up_tucker_cores:    typ.Tuple[NDArray,...]  # len=d. B_xo B_yo   = I_xy, Bi.shape = stack_shape+(nUi, Ni)
    down_tt_cores:      typ.Tuple[NDArray,...]  # len=d. R_ixj R_iyj = I_xy  Ri.shape = stack_shape+(rLi, nDi, rR(i+1))
    left_tt_cores:      typ.Tuple[NDArray,...]  # len=d. P_iax P_iay = I_xy, Pi.shape = stack_shape+(rLi, nUi, rL(i+1))
    right_tt_cores:     typ.Tuple[NDArray,...]  # len=d. Q_xaj Q_yaj = I_xy  Qi.shape = stack_shape+(rRi, nUi, rR(i+1))

    @ft.cached_property
    def d(self) -> int:
        return len(self.up_tucker_cores)

    @ft.cached_property
    def shape(self) -> typ.Tuple[int,...]:
        return tuple([U.shape[-1] for U in self.up_tucker_cores])

    @ft.cached_property
    def up_ranks(self) -> typ.Tuple[int,...]:
        return tuple([U.shape[-2] for U in self.up_tucker_cores])

    @ft.cached_property
    def down_ranks(self) -> typ.Tuple[int,...]:
        return tuple([G.shape[-2] for G in self.down_tt_cores])

    @ft.cached_property
    def left_ranks(self) -> typ.Tuple[int,...]:
        return tuple([G.shape[-3] for G in self.left_tt_cores]) + (self.left_tt_cores[-1].shape[-1],)

    @ft.cached_property
    def right_ranks(self) -> typ.Tuple[int, ...]:
        return tuple([G.shape[-3] for G in self.right_tt_cores]) + (self.right_tt_cores[-1].shape[-1],)

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:   # C (base/core stack): the batch of base points, on every core
        """The base/core stack ``C`` -- a batch of base points, shared by every core."""
        return self.up_tucker_cores[0].shape[:-2]

    @ft.cached_property
    def structure(self) -> typ.Tuple[
        typ.Tuple[int, ...], # shape
        typ.Tuple[int, ...], # up_tucker_ranks
        typ.Tuple[int, ...], # down_ranks
        typ.Tuple[int, ...], # left_ranks
        typ.Tuple[int, ...], # right_ranks
        typ.Tuple[int, ...], # stack_shape
    ]:
        return (
            self.shape,
            self.up_ranks, self.down_ranks,
            self.left_ranks, self.right_ranks,
            self.stack_shape,
        )

    @ft.cached_property
    def variation_shapes(
            self,
    ) -> typ.Tuple[
        typ.Tuple[typ.Tuple[int, ...], ...],  # tucker_variation_shapes. len=d. elm_len=2
        typ.Tuple[typ.Tuple[int, ...], ...],  # tt_variation_shapes. len=d. elm_len=3
    ]:
        '''T3Variations shapes that fit with this T3Basis.

        Shapes of the "holes" in the following tensor diagrams::

            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 U0     U1     U2     U3
                 |      |      |      |

            1 -- L0 -- L1 -- O2 -- R3 -- 1
                 |     |     |     |
                 U0    U1    ( )   U3
                 |     |     |     |

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.basis_variations_format as bvf
        >>> ss = (2,3) # not included in variation_shapes.
        >>> up_tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
        >>> down_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
        >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3,12,5)))
        >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
        >>> basis = bvf.T3Basis(up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
        >>> print(basis.variation_shapes)
        (((9, 14), (8, 15), (7, 16)), ((1, 10, 4), (2, 11, 5), (3, 12, 1)))
        '''
        tucker_variation_shapes = tuple([(nD, N) for nD, N in zip(self.down_ranks, self.shape)])
        tt_variation_shapes = tuple([
            (rL, nU, rR) for rL, nU, rR
            in zip(self.left_ranks[:-1], self.up_ranks, self.right_ranks[1:])])

        return tucker_variation_shapes, tt_variation_shapes

    @ft.cached_property
    def data(self) -> typ.Tuple[
        typ.Tuple[NDArray,...],  # up_tucker_cores
        typ.Tuple[NDArray, ...], # down_tt_cores
        typ.Tuple[NDArray,...],  # left_tt_cores
        typ.Tuple[NDArray,...],  # right_tt_cores
    ]:
        return self.up_tucker_cores, self.down_tt_cores, self.left_tt_cores, self.right_tt_cores

    def to_jax(self) -> 'T3Basis':
        """Copy with all basis cores converted to jax arrays."""
        return T3Basis(*[tuple(to_jax(c) for c in fam) for fam in self.data])

    def to_numpy(self) -> 'T3Basis':
        """Copy with all basis cores converted to numpy arrays."""
        return T3Basis(*[tuple(to_numpy(c) for c in fam) for fam in self.data])

    def copy(self) -> 'T3Basis':
        """Deep copy (copies every basis core)."""
        return T3Basis(*[tuple(c.copy() for c in fam) for fam in self.data])

    @ft.cached_property
    def contains_jax(self) -> bool:
        """True if any basis core is a jax array."""
        return tree_contains_jax(self.data)

    @ft.cached_property
    def size(self) -> int:
        """Number of elements of the represented (base-point) dense tensor (``prod(shape)``)."""
        return int(np.prod(self.shape))

    @ft.cached_property
    def data_size(self) -> int:
        """Number of stored core entries (size on disk)."""
        return sum(int(c.size) for fam in self.data for c in fam)

    def __repr__(self) -> str:
        ss = f", stack_shape={self.stack_shape}" if self.stack_shape else ""
        return (f"T3Basis(shape={self.shape}, up_ranks={self.up_ranks}, "
                f"left_ranks={self.left_ranks}{ss})")

    def is_orthogonal(self, atol: float = 1e-9) -> bool:
        '''True if the basis cores are orthogonal in their respective senses.

        Checks (each stacked block; max absolute deviation from identity <= atol):
            - up_tucker U_i (all i):    ``einsum('...io,...jo->...ij', U, U) = I``
            - down/outer D_i (all i):   ``einsum('...iaj,...ibj->...ab', D, D) = I``
            - left L_i (i = 0..d-2):    ``einsum('...iaj,...iak->...jk', L, L) = I``
            - right R_i (i = 1..d-1):   ``einsum('...iaj,...kaj->...ik', R, R) = I``

        The last left core and the first right core are the (non-orthogonal) boundary remainders
        and are not checked. This is a non-enforcing convenience checker; ``T3Basis`` does not
        require orthogonality at construction.

        Orthogonal cores (left/right/outer/Tucker) are defined in Appendix A.1 of Alger et al.
        (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
        >>> base, _ = bvf.t3_orthogonal_representations(x)
        >>> print(base.is_orthogonal())
        True
        '''
        UU, DD, LL, RR = self.data
        d = self.d

        def _dev(gram, n):
            return float(np.max(np.abs(np.asarray(gram) - np.eye(n))))

        resid = 0.0
        for ii in range(d):
            U = np.asarray(UU[ii])
            D = np.asarray(DD[ii])
            resid = max(resid, _dev(np.einsum('...io,...jo->...ij', U, U), U.shape[-2]))
            resid = max(resid, _dev(np.einsum('...iaj,...ibj->...ab', D, D), D.shape[-2]))
        for ii in range(d - 1):
            L = np.asarray(LL[ii])
            resid = max(resid, _dev(np.einsum('...iaj,...iak->...jk', L, L), L.shape[-1]))
        for ii in range(1, d):
            R = np.asarray(RR[ii])
            resid = max(resid, _dev(np.einsum('...iaj,...kaj->...ik', R, R), R.shape[-3]))
        return resid <= atol

    @ft.cached_property
    def minimal_ranks(self) -> typ.Tuple[typ.Tuple[int, ...], typ.Tuple[int, ...]]:
        """Structural minimal ranks ``(min_tucker_ranks, min_tt_ranks)`` for this basis's shape/ranks.

        **Structural**: computed from the rank tuples + shape, not the numerical core values (see
        :py:meth:`TuckerTensorTrain.get_minimal_ranks`).
        """
        return t3.TuckerTensorTrain.get_minimal_ranks(self.shape, self.up_ranks, self.left_ranks)

    @ft.cached_property
    def has_minimal_ranks(self) -> bool:
        '''True if the basis has minimal ranks.

        Minimal-rank basis means:
            - ``left_ranks == right_ranks``,
            - ``up_ranks == down_ranks``, and
            - those ranks (tucker_ranks=up_ranks, tt_ranks=left_ranks) are minimal for a regular
              Tucker tensor train of this shape (see :py:meth:`TuckerTensorTrain.get_minimal_ranks`).

        .. note::
            Some tangent-space operations are only correct when the basis has minimal ranks;
            exactly which is still to be determined (flagged for later consideration). This is a
            non-enforcing checker; ``T3Basis`` does not require minimal ranks at construction.

        Minimal (non-degenerate) ranks and their connection to matricizations and matrix unfoldings
        are discussed in Appendix A.2 of Alger et al. (2026), "Tucker Tensor Train Taylor Series"
        (arXiv:2603.21141).

        Examples
        --------
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.basis_variations_format as bvf
        >>> x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))  # minimal ranks
        >>> base, _ = bvf.t3_orthogonal_representations(x)
        >>> print(base.has_minimal_ranks)
        True
        >>> x2 = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))  # Tucker rank 4 > 1*3
        >>> base2, _ = bvf.t3_orthogonal_representations(x2)
        >>> print(base2.has_minimal_ranks)
        False
        '''
        if tuple(self.left_ranks) != tuple(self.right_ranks):
            return False
        if tuple(self.up_ranks) != tuple(self.down_ranks):
            return False
        min_tucker_ranks, min_tt_ranks = self.minimal_ranks
        return (tuple(map(int, min_tucker_ranks)) == tuple(map(int, self.up_ranks))
                and tuple(map(int, min_tt_ranks)) == tuple(map(int, self.left_ranks)))

    def validate(self) -> None:
        '''Check rank and shape consistency of Tucker tensor train basis (`T3Basis`).

        Parameters
        ----------
        x : T3Basis

        Raises
        ------
        ValueError
            Error raised if the cores of the T3Basis have inconsistent shapes.

        See Also
        --------
        T3Basis
        T3Variations
        '''
        UU, DD, LL, RR = self.data

        d = len(UU)
        if not (len(LL) == d and len(RR) == d and len(DD) == d):
            raise ValueError(
                'Inconsistent T3Basis.\n'
                + 'All core sequences must have length d=' + str(d) + '.\n'
                + 'len(UU)=' + str(len(UU))
                + ', len(DD)=' + str(len(DD))
                + ', len(LL)=' + str(len(LL))
                + ', len(RR)=' + str(len(RR))

            )

        for ii, U in enumerate(UU):
            if len(U.shape) < 2:
                raise ValueError(
                    'Inconsistent T3Basis.\n'
                    + 'tucker_cores[' + str(ii) + '] is not a (stacked) matrix. shape=' + str(U.shape)
                )

        for name, CC in zip(["left_tt", "right_tt", "outer_tt"], [LL, RR, DD]):
            for ii, C in enumerate(CC):
                if len(C.shape) < 3:
                    raise ValueError(
                        'Inconsistent T3Basis.\n'
                        + name + '_cores[' + str(ii) + '] is not a (stacked) 3-tensor. '
                        + 'shape=' + str(C.shape)
                    )

        up_stack_shapes     = tuple([B.shape[:-2] for B in self.up_tucker_cores])
        left_stack_shapes   = tuple([G.shape[:-3] for G in self.left_tt_cores])
        right_stack_shapes  = tuple([G.shape[:-3] for G in self.right_tt_cores])
        down_stack_shapes   = tuple([G.shape[:-3] for G in self.down_tt_cores])

        if not (
                up_stack_shapes
                == down_stack_shapes
                == left_stack_shapes
                == right_stack_shapes
                == (self.stack_shape,)*self.d
        ):
            raise ValueError(
                'Inconsistent T3Basis.\n'
                + str(up_stack_shapes) + ' = up_stack_shapes.\n'
                + str(down_stack_shapes) + ' = down_stack_shapes.\n'
                + str(left_stack_shapes) + ' = left_stack_shapes.\n'
                + str(right_stack_shapes) + ' = right_stack_shapes.'
            )

        rLl = tuple([int(LL[0].shape[-3])] + [int(L.shape[-1]) for L in LL])
        rLr = tuple([int(L.shape[-3]) for L in LL] + [int(LL[-1].shape[-1])])
        if rLl != rLr:
            raise ValueError(
                'Inconsistent T3Basis.\n'
                + str(rLl) + ' = rL_left != rL_right = ' + str(rLr)
            )

        rRl = tuple([int(RR[0].shape[-3])] + [int(R.shape[-1]) for R in RR])
        rRr = tuple([int(R.shape[-3]) for R in RR] + [int(RR[-1].shape[-1])])
        if rRl != rRr:
            raise ValueError(
                'Inconsistent T3Basis.\n'
                + str(rRl) + ' = rR_left != rR_right = ' + str(rRr)
            )

        for ii in range(d):
            U, L, R, D = UU[ii], LL[ii], RR[ii], DD[ii]

            if not (U.shape[-2] == L.shape[-2] == R.shape[-2]):
                raise ValueError(
                    'Inconsistent T3Basis.\n'
                    + 'Tucker rank mismatch at index ' + str(ii)
                    + ': U.shape[-2]=' + str(U.shape[0])
                    + ', L.shape[-2]=' + str(L.shape[1])
                    + ', R.shape[-2]=' + str(R.shape[1])
                )

            if D.shape[-3] != L.shape[-3]:
                raise ValueError(
                    'Inconsistent T3Basis.\n'
                    + 'Down TT core left rank mismatch at index' + str(ii)
                    + ': D.shape[-3]=' + str(D.shape[-3])
                    + '!= L.shape[-3]=' + str(L.shape[-3])
                )

            if D.shape[-1] != R.shape[-1]:
                raise ValueError(
                    'Inconsistent T3Base.\n'
                    + 'Down TT core right rank mismatch at index' + str(ii)
                    + ': D.shape[-1]=' + str(D.shape[-1])
                    + '!= R.shape[-1]=' + str(R.shape[-1])
                )

    def __post_init__(self):
        self.validate()

    def unstack(self):
        """Unstack into an array-like tree.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.corewise as cw
        >>> randnstar = lambda x: np.random.randn(*x)
        >>> ss = (2,3) # not included in variation_shapes.
        >>> up_tucker_cores = (randnstar(ss+(10, 14)), randnstar(ss+(11, 15)), randnstar(ss+(12, 16)))
        >>> down_tt_cores = (randnstar(ss+(1, 9, 4)), randnstar(ss+(2, 8, 5)),randnstar(ss+(3, 7, 1)))
        >>> left_tt_cores = (randnstar(ss+(1, 10, 2)), randnstar(ss+(2, 11, 3)), randnstar(ss+(3,12,5)))
        >>> right_tt_cores = (randnstar(ss+(2, 10, 4)), randnstar(ss+(4, 11, 5)), randnstar(ss+(5, 12, 1)))
        >>> basis = bvf.T3Basis(up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
        >>> S = basis.unstack()
        >>> ii, jj = 1, 2
        >>> Sij = S[ii][jj]
        >>> utk_ij = tuple(x[ii,jj,:,:] for x in up_tucker_cores)
        >>> dtt_ij = tuple(x[ii,jj,:,:,:] for x in down_tt_cores)
        >>> ltt_ij = tuple(x[ii,jj,:,:,:] for x in left_tt_cores)
        >>> rtt_ij = tuple(x[ii,jj,:,:,:] for x in right_tt_cores)
        >>> basis_ij = bvf.T3Basis(utk_ij, dtt_ij, ltt_ij, rtt_ij)
        >>> print(cw.corewise_norm(cw.corewise_sub(basis_ij.data, Sij.data)))
        0.0
        """
        return stacking.apply_func_to_leaf_subtrees(
            stacking.basic_ragged_unstack(self.data, 2),
            lambda x: T3Basis(*x),
            self.data, # leaf_structure
        )

    @staticmethod
    def stack(
            xx, # Array-like tree of T3Basis
    ):
        """Stack array-like tree of T3Basis into a single T3Basis.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.corewise as cw
        >>> randnstar = lambda x: np.random.randn(*x)
        >>> ss = (2,3) # not included in variation_shapes.
        >>> up_tucker_cores = (randnstar(ss+(10, 14)), randnstar(ss+(11, 15)), randnstar(ss+(12, 16)))
        >>> down_tt_cores = (randnstar(ss+(1, 9, 4)), randnstar(ss+(2, 8, 5)),randnstar(ss+(3, 7, 1)))
        >>> left_tt_cores = (randnstar(ss+(1, 10, 2)), randnstar(ss+(2, 11, 3)), randnstar(ss+(3,12,5)))
        >>> right_tt_cores = (randnstar(ss+(2, 10, 4)), randnstar(ss+(4, 11, 5)), randnstar(ss+(5, 12, 1)))
        >>> x = bvf.T3Basis(up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
        >>> xx = x.unstack()
        >>> x2 = bvf.T3Basis.stack(xx)
        >>> print(cw.corewise_norm(cw.corewise_sub(x.data, x2.data)))
        0.0
        """
        xx_tuples = stacking.apply_func_to_leaf_subtrees(
            xx,
            lambda x: x.data,
            None,  # leaf_structure
        )
        result = stacking.basic_ragged_stack(xx_tuples)
        return T3Basis(*result)

    @staticmethod
    def from_t3(x: 't3.TuckerTensorTrain') -> 'T3Basis':
        """Orthogonal representation (basis) of a TuckerTensorTrain ``x`` -- the orthogonal frame at the
        point ``x`` (the basis part of :py:func:`t3_orthogonal_representations`)."""
        return t3_orthogonal_representations(x)[0]

    @staticmethod
    def random_orthogonal(shape, tucker_ranks, tt_ranks, stack_shape=(), use_jax=False) -> 'T3Basis':
        """Orthogonal representation of a *random* T3 -- a genuine random base point (orthogonal,
        consistent), **not** iid-random cores. Equals ``from_t3(TuckerTensorTrain.randn(...))``."""
        x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks, stack_shape=stack_shape, use_jax=use_jax)
        return t3_orthogonal_representations(x)[0]

    @staticmethod
    def random_orthogonal_like(basis: 'T3Basis') -> 'T3Basis':
        """A random orthogonal basis with the same shape/ranks/stack as ``basis``."""
        return T3Basis.random_orthogonal(basis.shape, basis.up_ranks, basis.left_ranks,
                                         stack_shape=basis.stack_shape, use_jax=basis.contains_jax)

    def save(self, file) -> None:
        """Save the basis cores to a ``.npz`` file (load with :py:meth:`load`)."""
        np.savez(file, **{'f%d_%d' % (fi, ci): np.asarray(c)
                          for fi, fam in enumerate(self.data) for ci, c in enumerate(fam)})

    @staticmethod
    def load(file, use_jax: bool = False) -> 'T3Basis':
        """Load a basis saved by :py:meth:`save`."""
        npz = np.load(file)
        def fam(fi):
            ks = sorted((k for k in npz.files if k.startswith('f%d_' % fi)), key=lambda k: int(k.split('_', 1)[1]))
            return tuple(npz[k] for k in ks)
        b = T3Basis(fam(0), fam(1), fam(2), fam(3))
        return b.to_jax() if use_jax else b




@dataclass(frozen=True)
class T3Variations:
    """
    Tuple containing variation cores for basis-variation representations of TuckerTensorTrains.

    *Components*
        - tucker_variations    = (V0, ..., V(d-1)), elm_shape=stack_shape+(nDi, Ni)
        - tt_variations        = (H0, ..., H(d-1)), elm_shape=stack_shape+(rLi, nUi, rRi)

    The variations should fit in the "holes" of a T3Basis.

    See Also
    --------
    T3Basis

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.basis_variations_format as bvf
    >>> ss = (2,3) # stack shape
    >>> up_tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
    >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3,12,5)))
    >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
    >>> down_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
    >>> base = bvf.T3Basis(up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
    >>> print(base.variation_shapes)
    (((9, 14), (8, 15), (7, 16)), ((1, 10, 4), (2, 11, 5), (3, 12, 1)))
    >>> tucker_variations = tuple([np.ones(ss + B_shape) for B_shape in base.variation_shapes[0]])
    >>> tt_variations = tuple([np.ones(ss + G_shape) for G_shape in base.variation_shapes[1]])
    >>> variations = bvf.T3Variations(tucker_variations, tt_variations) # variation that fits with base
    >>> print(variations.variation_shapes) # same as base, except first right tt rank and last left tt rank, which are None
    (((9, 14), (8, 15), (7, 16)), ((1, 10, 4), (2, 11, 5), (3, 12, 1)))
    """
    tucker_variations: typ.Tuple[NDArray,...]  # len=d, elm_shape=stack_shape+(nDi, Ni)
    tt_variations:     typ.Tuple[NDArray,...]  # len=d, elm_shape=stack_shape+(rLi, nUi, rRi)

    @ft.cached_property
    def d(self) -> int:
        return len(self.tucker_variations)

    @ft.cached_property
    def shape(self) -> typ.Tuple[int,...]:
        return tuple([U.shape[-1] for U in self.tucker_variations])

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:   # full leading stack K + C (split-agnostic on its own)
        """The full leading stack ``K + C`` shared by every variation core.

        ``T3Variations`` is **split-agnostic**: the tangent-stack ``K`` vs base-stack ``C`` split is
        fixed only when paired with a :py:class:`T3Basis` (``check_bv_pair`` requires ``C`` to be the
        trailing/inner part of this stack; :py:class:`~t3toolbox.manifold.T3Tangent` then exposes the
        two parts). See ``docs/batching_and_stacking.md``.
        """
        return self.tucker_variations[0].shape[:-2]

    @ft.cached_property
    def variation_shapes(
            self,
    ) -> typ.Tuple[
        typ.Tuple[typ.Tuple[int, ...], ...],  # tucker_variation_shapes. len=d. elm_len=2
        typ.Tuple[typ.Tuple[int, ...], ...],  # tt_variation_shapes. len=d. elm_len=3
    ]:
        '''T3Variations shapes that fit with this T3Basis.

        Shapes of the "holes" in the following tensor diagrams::

            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 U0     U1     U2     U3
                 |      |      |      |

            1 -- L0 -- L1 -- O2 -- R3 -- 1
                 |     |     |     |
                 U0    U1    ( )   U3
                 |     |     |     |
        '''
        tucker_variation_shapes = tuple([B.shape[-2:] for B in self.tucker_variations])
        tt_variation_shapes = tuple([G.shape[-3:] for G in self.tt_variations])
        return tucker_variation_shapes, tt_variation_shapes

    @ft.cached_property
    def data(self) -> typ.Tuple[
        typ.Tuple[NDArray,...], # tucker_variations
        typ.Tuple[NDArray,...], # tt_variations
    ]:
        return self.tucker_variations, self.tt_variations

    def to_jax(self) -> 'T3Variations':
        """Copy with all variation cores converted to jax arrays."""
        return T3Variations(*[tuple(to_jax(c) for c in fam) for fam in self.data])

    def to_numpy(self) -> 'T3Variations':
        """Copy with all variation cores converted to numpy arrays."""
        return T3Variations(*[tuple(to_numpy(c) for c in fam) for fam in self.data])

    def copy(self) -> 'T3Variations':
        """Deep copy (copies every variation core)."""
        return T3Variations(*[tuple(c.copy() for c in fam) for fam in self.data])

    @ft.cached_property
    def contains_jax(self) -> bool:
        """True if any variation core is a jax array."""
        return tree_contains_jax(self.data)

    @ft.cached_property
    def size(self) -> int:
        """Number of elements of the represented dense tensor (``prod(shape)``)."""
        return int(np.prod(self.shape))

    @ft.cached_property
    def data_size(self) -> int:
        """Number of stored core entries (size on disk)."""
        return sum(int(c.size) for fam in self.data for c in fam)

    def __repr__(self) -> str:
        ss = f", stack_shape={self.stack_shape}" if self.stack_shape else ""
        return f"T3Variations(shape={self.shape}{ss})"

    def validate(self) -> None:
        '''Check rank and shape consistency of Tucker tensor train variations (`T3Variations`).

        Parameters
        ----------
        self : T3Variations

        Raises
        ------
        ValueError
            Error raised if the cores of the T3Variations have inconsistent shapes.

        See Also
        --------
        T3Basis
        T3Variations
        '''
        VV, HH = self.data

        d = len(VV)
        if len(HH) != d:
            raise ValueError(
                'Inconsistent T3Variations.\n'
                + 'All core sequences must have length d=' + str(d) + '.\n'
                + 'len(VV)=' + str(len(VV))
                + ', len(HH)=' + str(len(HH))
            )

        for ii, V in enumerate(VV):
            if len(V.shape) < 2:
                raise ValueError(
                    'Inconsistent T3Variations.\n'
                    + 'tucker_cores[' + str(ii) + '] is not a (stacked) matrix. shape=' + str(V.shape)
                )

        for ii, H in enumerate(HH):
            if len(H.shape) < 3:
                raise ValueError(
                    'Inconsistent T3Variations.\n'
                    + 'tt_cores[' + str(ii) + '] is not a (stacked) 3-tensor. '
                    + 'shape=' + str(H.shape)
                )

        tucker_stack_shapes = tuple([B.shape[:-2] for B in self.tucker_variations])
        tt_stack_shapes = tuple([G.shape[:-3] for G in self.tt_variations])

        if not (tucker_stack_shapes == tt_stack_shapes == (self.stack_shape,)*self.d):
            raise ValueError(
                'Inconsistent T3Basis.\n'
                + str(tucker_stack_shapes) + ' = tucker_stack_shapes.\n'
                + str(tt_stack_shapes) + ' = tt_stack_shapes.\n'
            )

    def __post_init__(self):
        self.validate()

    def unstack(self):
        """Unstack into an array-like tree.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.corewise as cw
        >>> rnd = lambda x: np.random.randn(*x)
        >>> ss = (2,3) # not included in variation_shapes.
        >>> NN, nnU, nnD, rrL, rrR = (10,11,12), (8,7,8), (6,7,8), (2,3,4,3), (5,4,6,1)
        >>> tucker_variations = tuple(rnd(ss+(n,N)) for n, N in zip(nnD, NN))
        >>> tt_variations = tuple(rnd(ss+(rL, n, rR)) for rL, n, rR in zip(rrL[:-1],nnU,rrR[1:]))
        >>> V = bvf.T3Variations(tucker_variations, tt_variations)
        >>> VV = V.unstack()
        >>> ii, jj = 1, 2
        >>> V_ij = VV[ii][jj]
        >>> tkv_ij = tuple(x[ii,jj,:,:] for x in tucker_variations)
        >>> ttv_ij = tuple(x[ii,jj,:,:,:] for x in tt_variations)
        >>> V_ij2 = bvf.T3Variations(tkv_ij, ttv_ij)
        >>> print(cw.corewise_norm(cw.corewise_sub(V_ij2.data, V_ij.data)))
        0.0
        """
        return stacking.apply_func_to_leaf_subtrees(
            stacking.basic_ragged_unstack(self.data, 2),
            lambda x: T3Variations(*x),
            self.data, # leaf_structure
        )

    @staticmethod
    def stack(
            xx, # Array-like tree of T3Variations
    ):
        """Stack array-like tree of T3Variations into a single T3Variation.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.basis_variations_format as bvf
        >>> import t3toolbox.corewise as cw
        >>> rnd = lambda x: np.random.randn(*x)
        >>> ss = (2,3) # not included in variation_shapes.
        >>> NN, nnU, nnD, rrL, rrR = (10,11,12), (8,7,8), (6,7,8), (2,3,4,3), (5,4,6,1)
        >>> tucker_variations = tuple(rnd(ss+(n,N)) for n, N in zip(nnD, NN))
        >>> tt_variations = tuple(rnd(ss+(rL, n, rR)) for rL, n, rR in zip(rrL[:-1],nnU,rrR[1:]))
        >>> V = bvf.T3Variations(tucker_variations, tt_variations)
        >>> VV = V.unstack()
        >>> V2 = bvf.T3Variations.stack(VV)
        >>> print(cw.corewise_norm(cw.corewise_sub(V.data, V2.data)))
        0.0
        """
        xx_tuples = stacking.apply_func_to_leaf_subtrees(
            xx,
            lambda x: x.data,
            None,  # leaf_structure
        )
        result = stacking.basic_ragged_stack(xx_tuples)
        return T3Variations(*result)

    @staticmethod
    def zeros(variation_shapes, stack_shape=(), use_jax=False) -> 'T3Variations':
        """Zero variations of the given structure (additive identity).

        ``variation_shapes = (tucker_variation_shapes, tt_variation_shapes)`` -- e.g. a basis's
        :py:attr:`T3Basis.variation_shapes`. (See :py:meth:`zeros_like` to take the structure from an object.)
        """
        tucker_shapes, tt_shapes = variation_shapes
        v = T3Variations(tuple(np.zeros(tuple(stack_shape) + tuple(s)) for s in tucker_shapes),
                         tuple(np.zeros(tuple(stack_shape) + tuple(s)) for s in tt_shapes))
        return v.to_jax() if use_jax else v

    @staticmethod
    def randn(variation_shapes, stack_shape=(), use_jax=False) -> 'T3Variations':
        """Variations with i.i.d. N(0,1) core entries (corewise, ungauged). See :py:meth:`randn_like`."""
        tucker_shapes, tt_shapes = variation_shapes
        v = T3Variations(tuple(np.random.randn(*(tuple(stack_shape) + tuple(s))) for s in tucker_shapes),
                         tuple(np.random.randn(*(tuple(stack_shape) + tuple(s))) for s in tt_shapes))
        return v.to_jax() if use_jax else v

    @staticmethod
    def unit(variation_shapes, index, stack_shape=(), use_jax=False) -> 'T3Variations':
        """Canonical unit variation: zero except a single core entry set to 1.

        ``index = (use_tt_coordinate, i, within_index)`` selects the core (a tt-variation if
        ``use_tt_coordinate`` else a tucker-variation), its position ``i``, and the within-core index.
        These units are the standard basis of the variation cores -- an **overcomplete, non-ambient-
        orthogonal** generating set of the tangent space, not an orthonormal basis.
        """
        use_tt_coordinate, i, within_index = index
        tucker_shapes, tt_shapes = variation_shapes
        tucker = [np.zeros(tuple(stack_shape) + tuple(s)) for s in tucker_shapes]
        tt = [np.zeros(tuple(stack_shape) + tuple(s)) for s in tt_shapes]
        (tt if use_tt_coordinate else tucker)[i][(Ellipsis,) + tuple(within_index)] = 1.0
        v = T3Variations(tuple(tucker), tuple(tt))
        return v.to_jax() if use_jax else v

    @staticmethod
    def zeros_like(x) -> 'T3Variations':
        """Zero variations matching the structure (shapes + stack) of ``x`` (a T3Basis or T3Variations)."""
        return T3Variations.zeros(x.variation_shapes, stack_shape=x.stack_shape, use_jax=x.contains_jax)

    @staticmethod
    def randn_like(x) -> 'T3Variations':
        """Random variations matching the structure (shapes + stack) of ``x`` (a T3Basis or T3Variations)."""
        return T3Variations.randn(x.variation_shapes, stack_shape=x.stack_shape, use_jax=x.contains_jax)

    def to_vector(self) -> NDArray:
        """Flatten the variation cores to a 1D vector (the tangent's degrees of freedom)."""
        return t3_operations.t3_to_vector(self.data)

    @staticmethod
    def from_vector(flat, variation_shapes, stack_shape=()) -> 'T3Variations':
        """Inverse of :py:meth:`to_vector`: rebuild variations of the given structure from a 1D vector.

        ``variation_shapes = (tucker_variation_shapes, tt_variation_shapes)`` (e.g. a basis's
        :py:attr:`T3Basis.variation_shapes`); ``stack_shape`` is the full leading stack ``K + C``.
        """
        xnp, _, _ = get_backend(False, tree_contains_jax(flat))
        flat = xnp.asarray(flat)
        tucker_shapes, tt_shapes = variation_shapes
        full = ([tuple(stack_shape) + tuple(s) for s in tucker_shapes]
                + [tuple(stack_shape) + tuple(s) for s in tt_shapes])
        cores, o = [], 0
        for shp in full:
            n = int(np.prod(shp))
            cores.append(flat[o:o + n].reshape(shp))
            o += n
        nt = len(tucker_shapes)
        return T3Variations(tuple(cores[:nt]), tuple(cores[nt:]))

    def save(self, file) -> None:
        """Save the variation cores to a ``.npz`` file (load with :py:meth:`load`)."""
        np.savez(file, **{'f%d_%d' % (fi, ci): np.asarray(c)
                          for fi, fam in enumerate(self.data) for ci, c in enumerate(fam)})

    @staticmethod
    def load(file, use_jax: bool = False) -> 'T3Variations':
        """Load variations saved by :py:meth:`save`."""
        npz = np.load(file)
        def fam(fi):
            ks = sorted((k for k in npz.files if k.startswith('f%d_' % fi)), key=lambda k: int(k.split('_', 1)[1]))
            return tuple(npz[k] for k in ks)
        v = T3Variations(fam(0), fam(1))
        return v.to_jax() if use_jax else v


def check_bv_pair(base: T3Basis, variations: T3Variations) -> None:
    """Check rank and shape consistency between T3Basis and T3Variations.

    This ensures that the variation cores (V, H) have the correct dimensions to interface with the
    base cores (U, L, R, O), and that their stacks are compatible.

    Stacking: a variation may carry extra *outer* tangent-stack axes -- a batch of tangent vectors
    sharing the same base point -- so its ``stack_shape`` is ``tangent_stack_shape + base_stack_shape``
    (extra axes outermost, base stack innermost). Consistency requires ``base.stack_shape`` to be the
    trailing (inner) part of ``variations.stack_shape``; ``tangent_stack_shape`` may be empty, which
    is the common case.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.basis_variations_format as bvf
    >>> ss = (2,3) # base stack shape
    >>> up_tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
    >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3,12,5)))
    >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
    >>> down_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
    >>> base = bvf.T3Basis(up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
    >>> tucker_variations = tuple([np.ones(ss + B_shape) for B_shape in base.variation_shapes[0]])
    >>> tt_variations = tuple([np.ones(ss + G_shape) for G_shape in base.variation_shapes[1]])
    >>> variations = bvf.T3Variations(tucker_variations, tt_variations)
    >>> bvf.check_bv_pair(base, variations) # does nothing since these are consistent

    A variation may also carry extra outer tangent-stack axes (here ``tangent_stack_shape = (4,)``,
    so the variation stack is ``(4,) + (2,3) = (4, 2, 3)``):

    >>> vss = (4,) + ss # tangent_stack_shape + base_stack_shape
    >>> tucker_variations = tuple([np.ones(vss + B_shape) for B_shape in base.variation_shapes[0]])
    >>> tt_variations = tuple([np.ones(vss + G_shape) for G_shape in base.variation_shapes[1]])
    >>> v_stacked = bvf.T3Variations(tucker_variations, tt_variations)
    >>> bvf.check_bv_pair(base, v_stacked) # also consistent: base stack is the suffix of the variation stack
    """
    base_stack = base.stack_shape
    var_stack = variations.stack_shape
    # The base stack must be the trailing (inner) part of the variation stack; the variation may
    # carry extra *outer* tangent-stack axes (variation stack = tangent_stack_shape + base_stack_shape).
    if var_stack[len(var_stack) - len(base_stack):] != base_stack:
        raise ValueError(
            'Inconsistent (T3Basis, T3Variations) pair.\n'
            'The base stack_shape must be the trailing (inner) part of the variation stack_shape.\n'
            + str(base_stack) + ' = base.stack_shape is not a suffix of '
            + str(var_stack) + ' = variations.stack_shape'
        )

    xVV, xHH = base.variation_shapes
    yVV, yHH = variations.variation_shapes

    for ii, (xV, yV) in enumerate(zip(xVV, yVV)):
        if xV != yV:
            raise ValueError(
                'Inconsistent T3Base - T3Variation pair.\n'
                + str(ii) + '-th Tucker variation shape' + str(yV)
                + ' does not fit base hole ' + str(xV)
            )

    for ii, (xH, yH) in enumerate(zip(xHH, yHH)):
        if xH != yH:
            raise ValueError(
                'Inconsistent T3Base - T3Variation pair.\n'
                + str(ii) + '-th tensor train variation shape' + str(yH)
                + ' does not fit base hole ' + str(xH)
            )


def bv_to_t3(
        index: typ.Tuple[
            bool, # TT core (true) or Tucker core (False)
            int, # number of the non-orthogonal core, 1...d-1
        ],
        basis: T3Basis,
        variations: T3Variations,
) -> t3.TuckerTensorTrain:
    '''Convert basis-variations representation to TuckerTensorTrain.

    If replacement_ind=1, replace_tt=True::

        1 -- L0 --(H1)-- R2 -- R3 -- 1
             |     |     |     |
             U0    U1    U2    U3
             |     |     |     |

    If replacement_ind=2, replace_tt=False::

        1 -- L0 -- L1 -- D2 -- R3 -- 1
             |     |     |     |
             U0    U1   (V2)   U3
             |     |     |     |

    These are the single-core variation terms summed in equation (47), Appendix A.3, of Alger et al.
    (2026), "Tucker Tensor Train Taylor Series" (arXiv:2603.21141).

    Parameters
    ----------
    ii: int
        Index of variation. 0 <= replacement_ind < num_cores
    replace_tt: bool
        Indicates whether to use TT variation (True) or a Tucker variation (False)
    base: T3Basis
        Basis cores
    variations: T3Variations
        Variation cores

    Raises
    ------
    RuntimeError
        - Error raised if the basis and variations do not fit with each other

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.basis_variations_format as bvf
    >>> randn = np.random.randn # shorthand
    >>> (U0,U1,U2) = (randn(10, 14), randn(11, 15), randn(12, 16))
    >>> (L0,L1,L2) = (randn(1, 10, 2), randn(2, 11, 3), randn(3,12,4))
    >>> (R0,R1,R2) = (randn(2,10,4), randn(4, 11, 5), randn(5, 12, 1))
    >>> (D0,D1,D2) = (randn(1, 9, 4), randn(2, 8, 5), randn(3, 7, 1))
    >>> base = bvf.T3Basis((U0,U1,U2), (D0,D1,D2), (L0,L1,L2), (R0,R1,R2))
    >>> (V0,V1,V2) = (randn(9,14), randn(8,15), randn(7,16))
    >>> (H0,H1,H2) = (randn(1,10,4), randn(2,11,5), randn(3,12,1))
    >>> variations = bvf.T3Variations((V0,V1,V2), (H0,H1,H2))
    >>> ((B0, B1, B2), (G0, G1, G2)) = bvf.bv_to_t3((True, 1), base, variations).data # replace index-1 TT-core
    >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,U1,U2), (L0,H1,R2)))
    True
    >>> ((B0, B1, B2), (G0, G1, G2)) = bvf.bv_to_t3((False, 1), base, variations).data # replace index-1 tucker core
    >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,V1,U2), (L0,D1,R2)))
    True
    '''
    check_bv_pair(basis, variations)
    # The term mixes a V+G-stacked variation core with G-stacked base cores (when the variation
    # carries an extra tangent stack K); broadcast all cores to the common K+C stack so the result is
    # a valid (uniform-stack) TuckerTensorTrain. A no-op when there is no tangent stack (K=()).
    cores = t3_operations.broadcast_t3_to_common_stack(
        *bv_conversions.bv_to_t3(index, basis.data, variations.data)
    )
    return t3.TuckerTensorTrain(*cores)


def t3_orthogonal_representations(
        x: t3.TuckerTensorTrain,
        already_left_orthogonal: bool = False,
        squash: bool = True,
) -> typ.Tuple[
    T3Basis,  # orthogonal base
    T3Variations,  # variations
]:
    '''Construct base-variation representations of TuckerTensorTrain with orthogonal base.

    Input TuckerTensorTrain::

                  1 -- G0 -- G1 -- G2 -- G3 -- 1
        X    =         |     |     |     |
                       B0    B1    B2    B3
                       |     |     |     |

    Base-variation representation with non-orthogonal TT-core H1::

                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    U2    U3
                       |     |     |     |

    Base-variation representation with non-orthogonal tucker core V2::

                  1 -- L0 -- L1 -- D2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |

    The input tensor train x is defined by:
        - x_tucker_cores    = (B0, B1, B2, B3)
        - x_tt_cores        = (G0, G1, G2, G3)
    The "base cores" are:
        - tucker_cores      = (U0,U1, U2, U3), up orthogonal
        - down_tt_cores     = (O0, O1, O2, O3), down orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
    The "variation cores" are:
        - tucker_variations  = (V0, V1, V2, V3)
        - tt_variations     = (H0, H1, H2, H3)

    Implements the sweeping orthogonalization (Algorithm 11), producing the representations
    (45)-(46), in Appendix A.3 of Alger et al. (2026), "Tucker Tensor Train Taylor Series"
    (arXiv:2603.21141). NOTE: the left/right orthogonalization sweep order here differs from
    Algorithm 11 (left-then-right vs the paper's right-then-left); the resulting orthogonal
    representations are equivalent.

    Parameters
    ----------
    x: TuckerTensorTrain
        Input TuckerTensorTrain
        x = (x_tucker_cores, x_tt_cores)
        x_tucker_cores = (B0, ..., B(d-1))
        x_tt_cores = (G0, ..., G(d-1))

    Returns
    -------
    T3Base
        Orthogonal base for base-variation representations of x.
    T3Variation
        Variation for base-variation representaions of x.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (3,3,2,1), stack_shape=(2,3))
    >>> base, variations = bvf.t3_orthogonal_representations(x) # Compute orthogonal representations
    >>> up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base.data
    >>> tucker_variations, tt_variations = variations.data
    >>> (U0,U1,U2) = up_tucker_cores
    >>> (D0,D1,D2) = down_tt_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (V0,V1,V2) = tucker_variations
    >>> (H0,H1,H2) = tt_variations
    >>> x2 = t3.TuckerTensorTrain((U0,U1,U2), (L0,H1,R2)) # representation with TT-core variation in index 1
    >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Still represents origional tensor
    4.978421562425667e-12
    >>> x3 = t3.TuckerTensorTrain((U0,V1,U2), (L0,D1,R2)) # representation with tucker core variation in index 1
    >>> print(np.linalg.norm(x.to_dense() - x3.to_dense())) # Still represents origional tensor
    5.4355175448533146e-12
    >>> print(np.linalg.norm(np.einsum('...io,...jo', U1, U1) - np.eye(U1.shape[-2]))) # U: orthogonal
    1.1915111872574236e-15
    >>> print(np.linalg.norm(np.einsum('...iaj,...iak', L1, L1) - np.eye(L1.shape[-1]))) # L: left orthogonal
    9.733823879665448e-16
    >>> print(np.linalg.norm(np.einsum('...iaj,...kaj', R1, R1) - np.eye(R1.shape[-3]))) # R: right orthogonal
    8.027553546330097e-16
    >>> print(np.linalg.norm(np.einsum('...iaj,...ibj', D1, D1) - np.eye(D1.shape[-2]))) # O: outer orthogonal
    1.3870474292323159e-15
    '''
    result = orth_reps.orthogonal_representations(
        x.data, already_left_orthogonal=already_left_orthogonal, squash=squash,
    )
    return T3Basis(*result[0]), T3Variations(*result[1])


if has_jax:
    import jax

    # Register as jax pytrees so they can be jit/vmap/grad-ed. Leaves = the cores (x.data); no aux.
    jax.tree_util.register_pytree_node(
        T3Basis,
        lambda x: (x.data, None),
        lambda aux_data, children: T3Basis(*children),
    )
    jax.tree_util.register_pytree_node(
        T3Variations,
        lambda x: (x.data, None),
        lambda aux_data, children: T3Variations(*children),
    )
