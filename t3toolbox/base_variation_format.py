# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import typing as typ

import t3toolbox.tucker_tensor_train as t3
from t3toolbox.common import *

NDArray = typ.TypeVar('NDArray') # Generic stand-in for np.ndarray, jnp.ndarray, or other array backend

__all__ = [
    'T3Base',
    'T3Variation',
    'BVStructure',
    'BVEdgeWeights',
    'get_base_structure',
    'get_base_hole_shapes',
    'get_variation_shapes',
    'ith_bv_to_t3',
]


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

In each of these, there is a special "variation" core, indicated by parentheses (X), surrounded by base cores. 

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

If these conditions are satisfied, then one can do "dumb" corewise linear algebra operations
(add, scale, dot product, etc) with the variations, and those operations faithfully correspond 
to linear algebra operations with the N1 x ... x Nd tangent vectors represented by the variations. 

See Also
--------
T3Variation
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
"""Shape and rank structore of a base-variation T3 representation.

*Components*
    - shape:                typ.Sequence[int]. len=d
    - up_tucker_ranks:      typ.Sequence[int]. len=d
    - outer_tucker_ranks:   typ.Sequence[int]. len=d
    - left_tt_ranks:        typ.Sequence[int]. len=d+1
    - right_tt_ranks:       typ.Sequence[int]. len=d+1

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

*Components*
    - shape_weights:        typ.Sequence[NDArray], len=d, elm_shape=(Ni,)
    - up_tucker_weights:    typ.Sequence[NDArray], len=d, elm_shape=(nUi,)
    - outer_tucker_weights: typ.Sequence[NDArray], len=d, elm_shape=(nOi,)
    - left_tt_weights:      typ.Sequence[NDArray], len=d, elm_shape=(rLi,)
    - right_tt_weights:     typ.Sequence[NDArray], len=d, elm_shape=(rRi,)
    
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
    '''T3Variation core shapes that fit with given T3Base.

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
        Variation Tucker core shapes. len=d. elm_len=2
    typ.Tuple[int,...]
        Variation TT core shapes. len=d. elm_len=3

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
        replace_tt: bool, # If True, replace TT-core. If False, replace tucker_core.
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
        Index of core to replace. 0 <= replacement_ind < num_cores
    replace_tt: bool
        Indicates whether to replace a TT-core (True) or a Tucker core (False)
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
    >>> ((B0, B1, B2), (G0, G1, G2)) = bvf.ith_bv_to_t3(1, True, base, variation) # replace index-1 TT-core
    >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,U1,U2), (L0,H1,R2)))
    True
    >>> ((B0, B1, B2), (G0, G1, G2)) = bvf.ith_bv_to_t3(1, False, base, variation) # replace index-1 tucker core
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

