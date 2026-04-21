# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.tucker_tensor_train as t3
from t3toolbox.backend.common import *


__all__ = [
    'T3Basis',
    'T3Variation',
    'BVStructure',
    'BVEdgeWeights',
    'get_base_structure',
    'get_base_hole_shapes',
    'get_variation_shapes',
    'ith_bv_to_t3',
    'check_t3base',
    'check_t3variation',
    'check_t3base_variation_pair',
    't3_orthogonal_representations',
]


@dataclass(frozen=True)
class T3Basis:
    """Basis for basis-coordinates representation of TuckerTensorTrains

    Often, one works with TuckerTensorTrains of the following forms::

        1--(H0)--R1---R2---1    1---L0--(H1)--R2---1    1---L0---L1--(H2)--1
            |    |    |             |    |    |             |    |    |
            U0   U1   U2            U0   U1   U2            U0   U1   U2
            |    |    |             |    |    |             |    |    |

        1---O0---R1---R2---1    1---L0---O1---R2---1    1---L0---L1---O2---1
            |    |    |             |    |    |             |    |    |
           (V0)  U1   U2            U0  (V1)  U2            U0   U1  (V2)
            |    |    |             |    |    |             |    |    |

    In each of these, there is a special "coordinate" core, indicated by parentheses (X), surrounded by basis cores.

    The components of T3Basis are the "basis cores":
        - up_tucker_cores   = (U0, ..., U(d-1)), elm_shape=(nUi, Ni)
        - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))
        - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))
        - outer_tt_cores    = (O0, ..., O(d-1)), elm_shape=(rLi, nOi, rR(i+1))

    The components of T3Coordinates are the "variation cores":
        - tucker_variations = (V0, ..., V(d-1)), elm_shape=(nOi, Ni)
        - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, ni, rRi)

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
        1 ------ L0 ------ O1 ------ ... ------ R(d-1) ------ 1
                 |         |                    |
                 | nU0     | nO1                | nU(d-1)
                 |         |                    |
                 U0       (V1)                   Ud
                 |         |                    |
                 | N0      | N1                 | N(d-1)
                 |         |                    |


    A tangent vector can be written as the sum of all of the tensor diagrams above.
    In this case, the basis cores are representations of the point where the
    tangent space attaches to the manifold, and the coordinate cores define the
    tangent vector with respect to the basis.

    Often, it is desirable for the base cores to be **orthogonal** as follows:
        - up_tucker_cores   = (U0,...,U(d-1)), orthogonal:       U_ia U_ja = delta_ij
        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij
        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal  R_ibc R_jbc = delta_ij
        - outer_tt_cores    = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij

    Often, it is desirable for the variations to satisfy the following **Gauge conditions**:
        - U_ia V_ja = 0    (all V)
        - L_abi H_abj = 0  (all but the last H)

    If these conditions are satisfied, then one can do "dumb" corewise linear algebra backend
    (add, scale, dot product, etc) with the variations, and those backend faithfully correspond
    to linear algebra backend with the N1 x ... x Nd tangent vectors represented by the variations.

    See Also
    --------
    T3Coordinates
    check_t3_base
    orthogonal_representations
    oblique_gauge_projection

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.base_variation_format as bvf
    >>> tucker_cores = (np.ones((10, 14)), np.ones((11, 15)), np.ones((12, 16)))
    >>> left_tt_cores = (np.ones((1, 10, 2)), np.ones((2, 11, 3)), np.ones((3,12,5)))
    >>> right_tt_cores = (np.ones((2, 10, 4)), np.ones((4, 11, 5)), np.ones((5, 12, 1)))
    >>> outer_tt_cores = (np.ones((1, 9, 4)), np.ones((2, 8, 5)), np.ones((3, 7, 1)))
    >>> base = (tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    >>> print(bvf.get_base_structure(base))
    ((14, 15, 16), (10, 11, 12), (9, 8, 7), (1, 2, 3, 5), (2, 4, 5, 1))
    >>> print(bvf.base_hole_shapes(base))
    (((9, 14), (8, 15), (7, 16)), ((1, 10, 2), (2, 11, 4), (3, 12, 5)))
    """

    up_tucker_cores:    typ.Tuple[NDArray,...]  # len=d. B_xo B_yo   = I_xy, B.shape = stack_shape+(n, N)
    left_tt_cores:      typ.Tuple[NDArray,...]  # len=d. P_iax P_iay = I_xy, P.shape = stack_shape+(rL, n, rR)
    right_tt_cores:     typ.Tuple[NDArray,...]  # len=d. Q_xaj Q_yaj = I_xy  Q.shape = stack_shape+(rL, n, rR)
    down_tt_cores:      typ.Tuple[NDArray,...]  # len=d. R_ixj R_iyj = I_xy  R.shape = stack_shape+(rL, n, rR)

    @ft.cached_property
    def d(self) -> int:
        return len(self.up_tucker_cores)

    @ft.cached_property
    def shape(self) -> typ.Tuple[int,...]:
        return tuple([U.shape[-1] for U in self.up_tucker_cores])

    @ft.cached_property
    def up_tucker_ranks(self) -> typ.Tuple[int,...]:
        return tuple([U.shape[-2] for U in self.up_tucker_cores])

    @ft.cached_property
    def down_tucker_ranks(self) -> typ.Tuple[int,...]:
        return tuple([G.shape[-2] for G in self.down_tt_cores])

    @ft.cached_property
    def left_tt_ranks(self) -> typ.Tuple[int,...]:
        return tuple([G.shape[-3] for G in self.left_tt_cores]) + (self.left_tt_cores[-1].shape[-1],)

    @ft.cached_property
    def right_tt_ranks(self) -> typ.Tuple[int, ...]:
        return tuple([G.shape[-3] for G in self.right_tt_cores]) + (self.right_tt_cores[-1].shape[-1],)

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:
        return self.up_tucker_cores[0].shape[:-2]

    @ft.cached_property
    def structure(self) -> typ.Tuple[
        typ.Tuple[int, ...], # shape
        typ.Tuple[int, ...], # up_tucker_ranks
        typ.Tuple[int, ...], # left_tt_ranks
        typ.Tuple[int, ...], # right_tt_ranks
        typ.Tuple[int, ...], # down_tt_ranks
        typ.Tuple[int, ...], # stack_shape
    ]:
        return (
            self.shape, self.up_tucker_ranks,
            self.left_tt_ranks, self.right_tt_ranks, self.down_tucker_ranks,
            self.stack_shape,
        )

    @ft.cached_property
    def data(self) -> typ.Tuple[
        typ.Tuple[NDArray,...], # up_tucker_cores
        typ.Tuple[NDArray,...], # left_tt_cores
        typ.Tuple[NDArray,...], # right_tt_cores
        typ.Tuple[NDArray,...], # down_tt_cores
    ]:
        return self.up_tucker_cores, self.left_tt_cores, self.right_tt_cores, self.down_tt_cores

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
        T3Coordinates
        '''
        UU, LL, RR, OO = self.data

        d = len(UU)
        if not (len(LL) == d and len(RR) == d and len(OO) == d):
            raise ValueError(
                'Inconsistent T3Basis.\n'
                + 'All backend sequences must have length d=' + str(d) + '.\n'
                + 'len(UU)=' + str(len(UU))
                + ', len(LL)=' + str(len(LL))
                + ', len(RR)=' + str(len(RR))
                + ', len(OO)=' + str(len(OO))
            )

        for ii, U in enumerate(UU):
            if len(U.shape) < 2:
                raise ValueError(
                    'Inconsistent T3Basis.\n'
                    + 'tucker_cores[' + str(ii) + '] is not a (stacked) matrix. shape=' + str(U.shape)
                )

        for name, CC in zip(["left_tt", "right_tt", "outer_tt"], [LL, RR, OO]):
            for ii, C in enumerate(CC):
                if len(C.shape) < 3:
                    raise ValueError(
                        'Inconsistent T3Basis.\n'
                        + name + '_cores[' + str(ii) + '] is not a (stacked) 3-tensor. '
                        + 'shape=' + str(C.shape)
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
        if rLl != rLr:
            raise ValueError(
                'Inconsistent T3Basis.\n'
                + str(rRl) + ' = rR_left != rR_right = ' + str(rRr)
            )

        for ii in range(d):
            U, L, R, O = UU[ii], LL[ii], RR[ii], OO[ii]

            if not (U.shape[-2] == L.shape[-2] == R.shape[-2]):
                raise ValueError(
                    'Inconsistent T3Basis.\n'
                    + 'Tucker rank mismatch at index ' + str(ii)
                    + ': U.shape[-2]=' + str(U.shape[0])
                    + ', L.shape[-2]=' + str(L.shape[1])
                    + ', R.shape[-2]=' + str(R.shape[1])
                )

            if O.shape[-3] != L.shape[-3]:
                raise ValueError(
                    'Inconsistent T3Basis.\n'
                    + 'Outer backend left rank mismatch at index' + str(ii)
                    + ': O.shape[-3]=' + str(O.shape[-3])
                    + '!= L.shape[-3]=' + str(L.shape[-3])
                )

            if O.shape[-1] != R.shape[-1]:
                raise ValueError(
                    'Inconsistent T3Base.\n'
                    + 'Outer backend right rank mismatch at index' + str(ii)
                    + ': O.shape[-1]=' + str(O.shape[-1])
                    + '!= R.shape[-1]=' + str(R.shape[-1])
                )




################################################################
########    TuckerTensorTrain base-variation format    #########
################################################################

T3Base = typ.Tuple[
    typ.Sequence[NDArray],  # up_tucker_cores. len=d. B_xo B_yo   = I_xy, B.shape = (n, N)
    typ.Sequence[NDArray],  # left_tt_cores.   len=d. P_iax P_iay = I_xy, P.shape = (rL, n, rR)
    typ.Sequence[NDArray],  # right_tt_cores.  len=d. Q_xaj Q_yaj = I_xy  Q.shape = (rL, n, rR)
    typ.Sequence[NDArray],  # outer_tt_cores.  len=d. R_ixj R_iyj = I_xy  R.shape = (rL, n, rR)
]
"""
Tuple containing base cores for base-variation representation of TuckerTensorTrains

Often, one works with TuckerTensorTrains of the following forms::

    1--(H0)--R1---R2---1    1---L0--(H1)--R2---1    1---L0---L1--(H2)--1
        |    |    |             |    |    |             |    |    |
        U0   U1   U2            U0   U1   U2            U0   U1   U2
        |    |    |             |    |    |             |    |    |

    1---O0---R1---R2---1    1---L0---O1---R2---1    1---L0---L1---O2---1
        |    |    |             |    |    |             |    |    |
       (V0)  U1   U2            U0  (V1)  U2            U0   U1  (V2)
        |    |    |             |    |    |             |    |    |

In each of these, there is a special "variation" backend, indicated by parentheses (X), surrounded by base cores. 

The components of T3Base are the "base cores":
    - up_tucker_cores   = (U0, ..., U(d-1)), elm_shape=(nUi, Ni)
    - left_tt_cores     = (L0, ..., L(d-1)), elm_shape=(rLi, ni, rL(i+1))
    - right_tt_cores    = (R0, ..., R(d-1)), elm_shape=(rRi, ni, rR(i+1))
    - outer_tt_cores    = (O0, ..., O(d-1)), elm_shape=(rLi, nOi, rR(i+1))
    
The components of T3Variations are the "variation cores":
    - tucker_variations = (V0, ..., V(d-1)), elm_shape=(nOi, Ni)
    - tt_variations     = (H0, ..., H(d-1)), elm_shape=(rLi, ni, rRi)

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
    1 ------ L0 ------ O1 ------ ... ------ R(d-1) ------ 1
             |         |                    |
             | nU0     | nO1                | nU(d-1)
             |         |                    |
             U0       (V1)                   Ud
             |         |                    |
             | N0      | N1                 | N(d-1)
             |         |                    |


A tangent vector can be written as the sum of all of the tensor diagrams above. 
In this case, the base cores are representations of the point where the 
tangent space attaches to the manifold, and the variation cores define the 
tangent vector with respect to the base cores. 

Often, it is desirable for the base cores to be **orthogonal** as follows:
    - up_tucker_cores   = (U0,...,U(d-1)), orthogonal:       U_ia U_ja = delta_ij
    - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij
    - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal  R_ibc R_jbc = delta_ij
    - outer_tt_cores    = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij

Often, it is desirable for the variations to satisfy the following **Gauge conditions**:
    - U_ia V_ja = 0    (all V)
    - L_abi H_abj = 0  (all but the last H)

If these conditions are satisfied, then one can do "dumb" corewise linear algebra backend
(add, scale, dot product, etc) with the variations, and those backend faithfully correspond 
to linear algebra backend with the N1 x ... x Nd tangent vectors represented by the variations. 

See Also
--------
T3Variation
check_t3_base
orthogonal_representations
oblique_gauge_projection

Examples
--------
>>> import numpy as np
>>> import t3toolbox.base_variation_format as bvf
>>> tucker_cores = (np.ones((10, 14)), np.ones((11, 15)), np.ones((12, 16)))
>>> left_tt_cores = (np.ones((1, 10, 2)), np.ones((2, 11, 3)), np.ones((3,12,5)))
>>> right_tt_cores = (np.ones((2, 10, 4)), np.ones((4, 11, 5)), np.ones((5, 12, 1)))
>>> outer_tt_cores = (np.ones((1, 9, 4)), np.ones((2, 8, 5)), np.ones((3, 7, 1)))
>>> base = (tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
>>> print(bvf.get_base_structure(base))
((14, 15, 16), (10, 11, 12), (9, 8, 7), (1, 2, 3, 5), (2, 4, 5, 1))
>>> print(bvf.base_hole_shapes(base))
(((9, 14), (8, 15), (7, 16)), ((1, 10, 2), (2, 11, 4), (3, 12, 5)))
"""


T3Variation = typ.Tuple[
    typ.Sequence[NDArray],  # variation_tucker_cores.
    typ.Sequence[NDArray],  # variation_tt_cores.
]
"""
Tuple containing variation cores for base-variation representation of TuckerTensorTrains.

*Components*
    - tucker_variations  = (V0, ..., V(d-1)), elm_shape=(nOi, Ni)
    - tt_variations      = (H0, ..., H(d-1)), elm_shape=(rLi, ni, rRi)

The variation components should fit in the "holes" of a T3Base.

See Also
--------
T3Base
check_t3variation
hole_shapes
check_fit

Examples
--------
>>> import numpy as np
>>> import t3toolbox.base_variation_format as bvf
>>> tucker_cores = (np.ones((10, 14)), np.ones((11, 15)), np.ones((12, 16)))
>>> left_tt_cores = (np.ones((1, 10, 2)), np.ones((2, 11, 3)), np.ones((3,12,5)))
>>> right_tt_cores = (np.ones((2, 10, 4)), np.ones((4, 11, 5)), np.ones((5, 12, 1)))
>>> outer_tt_cores = (np.ones((1, 9, 4)), np.ones((2, 8, 5)), np.ones((3, 7, 1)))
>>> base = (tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
>>> hole_shapes = bvf.base_hole_shapes(base)
>>> print(hole_shapes)
(((9, 14), (8, 15), (7, 16)), ((1, 10, 2), (2, 11, 4), (3, 12, 5)))
>>> var_tucker_cores = [np.ones(s) for s in hole_shapes[0]]
>>> var_tt_cores = [np.ones(s) for s in hole_shapes[1]]
>>> variation = (var_tucker_cores, var_tt_cores) # variation that fits with base
>>> print(bvf.variation_shapes(variation))
(((9, 14), (8, 15), (7, 16)), ((1, 10, 2), (2, 11, 4), (3, 12, 5)))
"""


BVStructure = typ.Tuple[
    typ.Sequence[int], # shape. len=d
    typ.Sequence[int], # up_tucker_ranks. len=d
    typ.Sequence[int], # outer_tucker_ranks. len=d
    typ.Sequence[int], # left_tt_ranks. len=d+1
    typ.Sequence[int], # right_tt_ranks. len=d+1
]
"""Shape and rank structure of a base-variation T3 representation.

*Components*
    - shape:                typ.Sequence[int] = (N0,...,N(d-1)),   len=d
    - up_tucker_ranks:      typ.Sequence[int] = (nU0,...,nU(d-1)), len=d
    - outer_tucker_ranks:   typ.Sequence[int] = (nO0,...,nO(d-1)), len=d
    - left_tt_ranks:        typ.Sequence[int] = (rL0,...,rLd),     len=d+1
    - right_tt_ranks:       typ.Sequence[int] = (rR0,...,rRd),     len=d+1

The variation components should fit in the "holes" of a T3Base.

See Also
--------
T3Base
T3Variation
"""


BVEdgeWeights = typ.Tuple[
    typ.Sequence[NDArray],  # shape_weights, len=d, elm_shape=(Ni,)
    typ.Sequence[NDArray],  # up_tucker_weights, len=d, elm_shape=(nUi,)
    typ.Sequence[NDArray],  # outer_tucker_weights, len=d, elm_shape=(nOi,)
    typ.Sequence[NDArray],  # left_tt_weights, len=d, elm_shape=(rLi,)
    typ.Sequence[NDArray],  # right_tt_weights, len=d, elm_shape=(rRi,)
]
"""Edge weights for base-variation format.

Tensor network diagrams illustrating edge weights::

    1--wL0--L0--wL1--H1--wR2--R2--wR3--1
            |        |        |
            wU0      wU1      wU3
            |        |        |
            U0       U1       U2
            |        |        |
            wS1      wS2      wS3
            |        |        |

and::

    1--wL0--L0--wL1--O1--wR2--R2--wR3--1
            |        |        |
            wU0      wU1      wU3
            |        |        |
            U0       V1       U2
            |        |        |
            wS1      wS2      wS3
            |        |        |

*Components*
    - shape_weights:        typ.Sequence[NDArray] = (wS0,...,wS(d-1)), len=d, elm_shape=(Ni,)
    - up_tucker_weights:    typ.Sequence[NDArray] = (wU0,...,wU(d-1)), len=d, elm_shape=(nUi,)
    - outer_tucker_weights: typ.Sequence[NDArray] = (wO0,...,wO(d-1)), len=d, elm_shape=(nOi,)
    - left_tt_weights:      typ.Sequence[NDArray] = (wL0,...,wL(d-1)), len=d, elm_shape=(rLi,)
    - right_tt_weights:     typ.Sequence[NDArray] = (wR1,...,wRd),     len=d, elm_shape=(rRi,)
    
Note: there are no weights for:
    - The edge between 1--R0
    - The edge between Ld--1 
    
See Also
--------
t3tools.tucker_tensor_train.EdgeWeights
T3Variation
T3Base
"""


def get_base_structure(
        base: T3Base,
) -> BVStructure:
    """Get the edge structure of a base-variation representation of a Tucker tensor train from the base.

    See Also
    --------
    T3Base

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.base_variation_format as bvf
    >>> tucker_cores = (np.ones((10, 14)), np.ones((11, 15)), np.ones((12, 16)))
    >>> left_tt_cores = (np.ones((1, 10, 2)), np.ones((2, 11, 3)), np.ones((1,12,5)))
    >>> right_tt_cores = (np.ones((2,10,4)), np.ones((4, 11, 5)), np.ones((5, 12, 1)))
    >>> outer_tt_cores = (np.ones((1, 9, 4)), np.ones((2, 8, 5)), np.ones((3, 7, 1)))
    >>> base = (tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    >>> print(bvf.get_base_structure(base))
    ((14, 15, 16), (10, 11, 12), (9, 8, 7), (1, 2, 3), (4, 5, 1))
    """
    up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    NN = tuple([U.shape[1] for U in up_tucker_cores])
    nnU = tuple([U.shape[0] for U in up_tucker_cores])
    rrL = tuple([L.shape[0] for L in left_tt_cores]) + (left_tt_cores[-1].shape[2],)
    rrR = tuple([R.shape[0] for R in right_tt_cores]) + (right_tt_cores[-1].shape[2],)
    nnO = tuple([O.shape[1] for O in outer_tt_cores])
    return (NN, nnU, nnO, rrL, rrR)


def get_variation_shapes(
        variation: T3Variation,
) -> typ.Tuple[
    typ.Tuple[int,...], # tucker_var_shapes, len=d
    typ.Tuple[int,...], # tt_var_shapes, len=d
]:
    """Get the shapes of the cores in a variation.
    """
    tucker_var_shapes = tuple([V.shape for V in variation[0]])
    tt_var_shapes = tuple([H.shape for H in variation[1]])
    return tucker_var_shapes, tt_var_shapes


def get_base_hole_shapes(
        base: T3Base,
) -> typ.Tuple[
    typ.Tuple[typ.Tuple[int,...],...], # variation_tucker_shapes. len=d. elm_len=2
    typ.Tuple[typ.Tuple[int,...],...], # variation_tt_shapes. len=d. elm_len=3
]:
    '''T3Variation backend shapes that fit with given T3Base.

    Shapes of the "holes" in the following tensor diagrams::

        1 -- L0 -- ( ) -- R2 -- R3 -- 1
             |      |      |      |
             U0     U1     U2     U3
             |      |      |      |

        1 -- L0 -- L1 -- O2 -- R3 -- 1
             |     |     |     |
             U0    U1    ( )   U3
             |     |     |     |

    Here:
        - tucker_cores      = (U0, U1, U2, U3)
        - left_tt_cores     = (L0, L1, L2, L3)
        - right_tt_cores    = (R0, R1, R2, R3)
        - outer_tt_cores    = (O0, O1, O2, O3)

    Parameters
    ----------
    base: T3Base
        Base cores

    Returns
    -------
    typ.Tuple[int,...]
        Variation Tucker backend shapes. len=d. elm_len=2
    typ.Tuple[int,...]
        Variation TT backend shapes. len=d. elm_len=3

    Raises
    ------
    RuntimeError
        - Error raised if any base_core shapes are inconsistent

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.base_variation_format as bvf
    >>> tucker_cores = (np.ones((10,14)), np.ones((11,15)), np.ones((12,16)))
    >>> left_tt_cores = (np.ones((1,10,2)), np.ones((2,11,3)), np.ones((3,12,1)))
    >>> right_tt_cores = (np.ones((1,10,4)), np.ones((4,11,5)), np.ones((5,12,1)))
    >>> outer_tt_cores = (np.ones((1,9,4)), np.ones((2,8,5)), np.ones((3,7,1)))
    >>> base = (tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    >>> (var_tucker_shapes, var_tt_shapes) = bvf.get_base_hole_shapes(base)
    >>> print(var_tucker_shapes)
    ((9, 14), (8, 15), (7, 16))
    >>> print(var_tt_shapes)
    ((1, 10, 4), (2, 11, 5), (3, 12, 1))
    '''
    NN, nnU, nnO, rrL, rrR = get_base_structure(base)

    var_tucker_shapes = tuple([(nO,N) for nO, N in zip(nnO,NN)])
    var_tt_shapes = tuple([(rL, nU, rR) for rL, nU, rR in zip(rrL[:-1], nnU, rrR[1:])])

    return var_tucker_shapes, var_tt_shapes


def ith_bv_to_t3(
        replacement_ind: int,
        replace_tt: bool, # If True, replace TT-backend. If False, replace tucker_core.
        base: T3Base,
        variation: T3Variation,
) -> t3.TuckerTensorTrain:
    '''Convert base-variation representation to TuckerTensorTrain.

    If replacement_ind=1, replace_tt=True::

        1 -- L0 -- H1 -- R2 -- R3 -- 1
             |     |     |     |
             U0    U1    U2    U3
             |     |     |     |

    If replacement_ind=2, replace_tt=False::

        1 -- L0 -- L1 -- O2 -- R3 -- 1
             |     |     |     |
             U0    U1    V2    U3
             |     |     |     |

    Parameters
    ----------
    replacement_ind: int
        Index of backend to replace. 0 <= replacement_ind < num_cores
    replace_tt: bool
        Indicates whether to replace a TT-backend (True) or a Tucker backend (False)
    base: T3Base
        Base cores
    variation: T3Variation
        Variation cores

    Raises
    ------
    RuntimeError
        - Error raised if the base is internally inconsistent
        - Error raised if the variation is internally incorrect
        - Error raised if the base and variation do not fit with each other

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.base_variation_format as bvf
    >>> randn = np.random.randn # shorthand
    >>> (U0,U1,U2) = (randn(10, 14), randn(11, 15), randn(12, 16))
    >>> (L0,L1,L2) = (randn(1, 10, 2), randn(2, 11, 3), randn(3,12,4))
    >>> (R0,R1,R2) = (randn(2,10,4), randn(4, 11, 5), randn(5, 12, 1))
    >>> (O0,O1,O2) = (randn(1, 9, 4), randn(2, 8, 5), randn(3, 7, 1))
    >>> base = ((U0,U1,U2), (L0,L1,L2), (R0,R1,R2), (O0,O1,O2))
    >>> (V0,V1,V2) = (randn(9,14), randn(8,15), randn(7,16))
    >>> (H0,H1,H2) = (randn(1,10,4), randn(2,11,5), randn(3,12,1))
    >>> variation = ((V0,V1,V2), (H0,H1,H2))
    >>> ((B0, B1, B2), (G0, G1, G2)) = bvf.ith_bv_to_t3(1, True, base, variation) # replace index-1 TT-backend
    >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,U1,U2), (L0,H1,R2)))
    True
    >>> ((B0, B1, B2), (G0, G1, G2)) = bvf.ith_bv_to_t3(1, False, base, variation) # replace index-1 tucker backend
    >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,V1,U2), (L0,O1,R2)))
    True
    '''
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    tucker_vars, tt_vars = variation

    if replace_tt:
        x_tucker_cores = tucker_cores
        x_tt_cores = (
                tuple(left_tt_cores[:replacement_ind]) +
                (tt_vars[replacement_ind],) +
                tuple(right_tt_cores[replacement_ind+1:])
        )
    else:
        x_tucker_cores = (
            tuple(tucker_cores[:replacement_ind]) +
            (tucker_vars[replacement_ind],) +
            tuple(tucker_cores[replacement_ind+1:])
        )
        x_tt_cores = (
                tuple(left_tt_cores[:replacement_ind]) +
                (outer_tt_cores[replacement_ind],) +
                tuple(right_tt_cores[replacement_ind+1:])
        )

    return (x_tucker_cores, x_tt_cores)



    

def check_t3variation(x: T3Variation) -> None:
    '''Check rank and shape consistency of Tucker tensor train base point (`T3Base`).

    Parameters
    ----------
    x : T3Base

    Raises
    ------
    ValueError
        Error raised if the cores of the T3Base have inconsistent shapes.

    See Also
    --------
    T3Base
    T3Variation
    '''
    VV, HH = x

    d = len(VV)
    if len(HH) != d:
        raise ValueError(
            'Inconsistent T3Variation.\n' 
            + 'All backend sequences must have length d=' + str(d) +'.\n'
            + 'len(VV)=' + str(len(VV))
            + ', len(HH)=' + str(len(HH))
        )

    for ii, V in enumerate(VV):
        if len(V.shape) != 2:
            raise ValueError(
                'Inconsistent T3Variation.\n'
                + 'tucker_cores[' + str(ii) + '] is not a matrix. shape=' + str(V.shape)
            )
        
    for ii, H in enumerate(HH):
        if len(H.shape) != 3:
            raise ValueError(
                'Inconsistent T3Variation.\n'
                + 'tt_cores[' + str(ii) + '] is not a 3-tensor. '
                + 'shape=' + str(H.shape)
            )
            

def check_t3bv(x: T3Base, y: T3Variation) -> None:
    """Check rank and shape consistency between T3Base and T3Variation.
    
    This ensures that the variation cores (V, H) have the correct dimensions
     to interface with the base cores (U, L, R, O).
    """
    xVV, xHH = get_base_hole_shapes(x)
    yVV, yHH = get_variation_shapes(y)

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


####

def t3_orthogonal_representations(
        x: t3.TuckerTensorTrain,
        already_left_orthogonal: bool = False,
        squash_tails: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    T3Base,  # orthogonal base
    T3Variation,  # variations
]:
    '''Construct base-variation representations of TuckerTensorTrain with orthogonal base.

    Input TuckerTensorTrain::

                  1 -- G0 -- G1 -- G2 -- G3 -- 1
        X    =         |     |     |     |
                       B0    B1    B2    B3
                       |     |     |     |

    Base-variation representation with non-orthogonal TT-backend H1::

                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    U2    U3
                       |     |     |     |

    Base-variation representation with non-orthogonal tucker backend V2::

                  1 -- L0 -- L1 -- O2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |

    The input tensor train x is defined by:
        - x_tucker_cores     = (B0, B1, B2, B3)
        - x_tt_cores        = (G0, G1, G2, G3)
    The "base cores" are:
        - tucker_cores       = (U0,U1, U2, U3), up orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
        - outer_tt_cores    = (O0, O1, O2, O3), down orthogonal
    The "variation cores" are:
        - tucker_variations  = (V0, V1, V2, V3)
        - tt_variations     = (H0, H1, H2, H3)

    Parameters
    ----------
    x: TuckerTensorTrain
        Input TuckerTensorTrain
        x = (x_tucker_cores, x_tt_cores)
        x_tucker_cores = (B0, ..., B(d-1))
        x_tt_cores = (G0, ..., G(d-1))
    xnp:
        Linear algebra backend. Default: np (numpy)

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
    >>> import t3toolbox.base_variation_format as bvf
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> base, variation = bvf.t3_orthogonal_representations(x) # Compute orthogonal representations
    >>> tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    >>> tucker_vars, tt_vars = variation
    >>> (U0,U1,U2) = tucker_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (O0,O1,O2) = outer_tt_cores
    >>> (V0,V1,V2) = tucker_vars
    >>> (H0,H1,H2) = tt_vars
    >>> x2 = t3.TuckerTensorTrain((U0,U1,U2), (L0,H1,R2)) # representation with TT-backend variation in index 1
    >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Still represents origional tensor
    4.978421562425667e-12
    >>> x3 = t3.TuckerTensorTrain((U0,V1,U2), (L0,O1,R2)) # representation with tucker backend variation in index 1
    >>> print(np.linalg.norm(x.to_dense() - x3.to_dense())) # Still represents origional tensor
    5.4355175448533146e-12
    >>> print(np.linalg.norm(U1 @ U1.T - np.eye(U1.shape[0]))) # U: orthogonal
    1.1915111872574236e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', L1, L1) - np.eye(L1.shape[2]))) # L: left orthogonal
    9.733823879665448e-16
    >>> print(np.linalg.norm(np.einsum('iaj,kaj->ik', R1, R1) - np.eye(R1.shape[0]))) # R: right orthogonal
    8.027553546330097e-16
    >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab', O1, O1) - np.eye(O1.shape[1]))) # O: outer orthogonal
    1.3870474292323159e-15

    Example where r0 and rd are not 1:

    >>> import numpy as np
    >>> from t3toolbox.tucker_tensor_train import TuckerTensorTrain
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2))
    >>> base, variation = bvf.t3_orthogonal_representations(x) # Compute orthogonal representations
    >>> tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    >>> tucker_vars, tt_vars = variation
    >>> (U0,U1,U2) = tucker_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (O0,O1,O2) = outer_tt_cores
    >>> (V0,V1,V2) = tucker_vars
    >>> (H0,H1,H2) = tt_vars
    >>> x2 = TuckerTensorTrain((U0,U1,U2), (L0,H1,R2)) # representation with TT-backend variation in index 1
    >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Still represents origional tensor
    2.5341562994067855e-12
    >>> x3 = TuckerTensorTrain((V0,U1,U2), (O0,R1,R2)) # representation with tucker backend variation in index 0
    >>> print(np.linalg.norm(x.to_dense() - x3.to_dense())) # Still represents origional tensor
    2.9206090606788446e-12
    >>> print(np.linalg.norm(U0 @ U0.T - np.eye(U0.shape[0]))) # U: orthogonal
    1.675264510304594e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', L0, L0) - np.eye(L0.shape[2]))) # L: left orthogonal
    9.046146325204653e-16
    >>> print(np.linalg.norm(np.einsum('iaj,kaj->ik', R2, R2) - np.eye(R2.shape[0]))) # R: right orthogonal
    1.1775693440128312e-16
    >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab', O0, O0) - np.eye(O0.shape[1]))) # O: outer orthogonal
    1.2300840868850519e-15
    '''
    if squash_tails:
        x = x.squash_tails(use_jax=use_jax)

    if not already_left_orthogonal:
        # Orthogonalize Tucker cores upward to get up_tt_cores U
        x = x.up_orthogonalize_tucker_cores(use_jax=use_jax)
        up_tucker_cores = x.tucker_cores

        # Sweep left-to-right, generating left orthogonal tt_cores L
        x = x.left_orthogonalize_tt_cores(use_jax=use_jax)
        left_tt_cores = x.tt_cores
    else:
        up_tucker_cores, left_tt_cores = x.data

    # Sweep right-to-left, generating tt_variations H, and right orthogonal tt_cores R
    x, tt_variations = x.right_orthogonalize_tt_cores(return_variation_cores=True, use_jax=use_jax)
    right_tt_cores = x.tt_cores

    # Orthogonalize TT cores downward to get outer_tt_cores O and tucker_variations V
    x = t3.TuckerTensorTrain(up_tucker_cores, tt_variations)
    x = x.down_orthogonalize_tt_cores(use_jax=use_jax)
    tucker_variations, outer_tt_cores = x.data

    base = (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    variation = (tucker_variations, tt_variations)
    return base, variation