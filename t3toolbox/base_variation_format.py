# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import typing as typ

import t3toolbox.tucker_tensor_train as t3

NDArray = typ.TypeVar('NDArray') # Generic stand-in for np.ndarray, jnp.ndarray, or other array backend

__all__ = [
    'T3Base',
    'T3Variation',
    'check_t3base',
    'check_t3variation',
    'hole_shapes',
    'check_fit',
    'ith_bv_to_t3',
]

NDArray = typ.TypeVar('NDArray') # Generic stand-in for np.ndarray, jnp.ndarray, or other array backend


################################################################
########    TuckerTensorTrain base-variation format    #########
################################################################

T3Base = typ.Tuple[
    typ.Sequence[NDArray],  # base_tucker_cores. B_xo B_yo = I_xy    B.shape = (n, N)
    typ.Sequence[NDArray],  # base_left_tt_cores. P_iax P_iay = I_xy, P.shape = (rL, n, rR)
    typ.Sequence[NDArray],  # base_right_tt_cores. Q_xaj Q_yaj = I_xy  Q.shape = (rL, n, rR)
    typ.Sequence[NDArray],  # base_outer_tt_cores. R_ixj R_iyj = I_xy  R.shape = (rL, n, rR)
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
    - tucker_cores      = (U0, ..., Ud), elm_shape=(ni, Ni)
    - left_tt_cores     = (L0, ..., Ld), elm_shape=(rLi, ni, rL(i+1))
    - right_tt_cores    = (R0, ..., Rd), elm_shape=(rRi, ni, rR(i+1))
    - outer_tt_cores    = (O0, ..., Od), elm_shape=(rLi, nOi, rR(i+1))

The components of T3Variations are the "variation cores":
    - tucker_variations = (V0, ..., Vd), elm_shape=(nOi, Ni)
    - tt_variations     = (H0, ..., Hd), elm_shape=(rLi, ni, rRi)

A tangent vector can be written as the sum of all of the tensor diagrams above. 
In this case, the base cores are representations of the point where the 
tangent space attaches to the manifold, and the variation cores define the 
tangent vector with respect to the base cores. 

Often, it is desirable for the base cores to be **orthogonal** as follows:
    - tucker_cores       = (U0,...,Ud), orthogonal:       U_ia U_ja = delta_ij
    - left_tt_cores     = (L0,...,Ld), left-orthogonal:  L_abi L_abj = delta_ij
    - right_tt_cores    = (R0,...,Rd), right-orthogonal  R_ibc R_jbc = delta_ij
    - outer_tt_cores    = (O0,...,Od), outer-orthogonal  O_aib O_ajb = delta_ij

Often, it is desirable for the variations to satisfy the following **Gauge conditions**:
    - U_ia V_ja = 0    (all V)
    - L_abi H_abj = 0  (all but the last H)

If these conditions are satisfied, then one can do "dumb" corewise linear algebra operations
(add, scale, dot product, etc) with the variations, and those operations faithfully correspond 
to linear algebra operations with the N1 x ... x Nd tangent vectors represented by the variations. 

See Also
--------
T3Variation
check_t3base
hole_shapes
check_fit
orthogonal_representations
oblique_gauge_projection


Examples
--------
>>> import numpy as np
>>> import t3toolbox.base_variation_format as bvf
>>> tucker_cores = (np.ones((10, 14)), np.ones((11, 15)), np.ones((12, 16)))
>>> left_tt_cores = (np.ones((1, 10, 2)), np.ones((2, 11, 3)), np.ones((3, 12, 1)))
>>> right_tt_cores = (np.ones((1, 10, 4)), np.ones((4, 11, 5)), np.ones((5, 12, 1)))
>>> outer_tt_cores = (np.ones((1, 9, 4)), np.ones((2, 8, 5)), np.ones((3, 7, 1)))
>>> base = (tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
>>> bvf.check_t3base(base) # Does nothing since base is internally consistent
>>> var_tucker_cores = (np.ones((9,14)), np.ones((8,15)), np.ones((7,16)))
>>> var_tt_cores = (np.ones((1,10,4)), np.ones((2,11,5)), np.ones((3,12,1)))
>>> variation = (var_tucker_cores, var_tt_cores)
>>> bvf.check_t3variation(variation) # Does nothing since variation is internally consistent
>>> bvf.check_fit(base, variation) # Does nothing since variation fits in base
"""

T3Variation = typ.Tuple[
    typ.Sequence[NDArray],  # variation_tucker_cores.
    typ.Sequence[NDArray],  # variation_tt_cores.
]
"""
Tuple containing variation cores for base-variation representation of TuckerTensorTrains.

*Components*
    - tucker_variations  = (V0, ..., Vd), elm_shape=(nOi, Ni)
    - tt_variations     = (H0, ..., Hd), elm_shape=(rLi, ni, rRi)

The variation components should fit in the "holes" of a T3Base.

See Also
--------
T3Base
check_t3variation
hole_shapes
check_fit
"""


def check_t3base(
        base: T3Base,
) -> None:
    '''Check that T3Base core shapes are internally consistent.

    Contractions of the following forms must make sense::

        1 -- L0 -- ( ) -- R2 -- R3 -- 1
             |     |      |     |
             U0    U1     U2    U3
             |     |      |     |

        1 -- L0 -- L1 -- O2 -- R3 -- 1
             |     |     |     |
             U0    U1    ( )   U3
             |     |     |     |


    Here:
        - tucker_cores      = (U0, U1, U2, U3)
        - left_tt_cores     = (L0, L1, L2, L3)
        - right_tt_cores    = (R0, R1, R2, R3)
        - outer_tt_cores    = (O0, O1, O2, O3)

    Raises
    ------
    RuntimeError
        - Error raised if any core shapes are inconsistent
    '''
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base

    num_cores = len(tucker_cores)
    all_num_cores = [len(tucker_cores), len(left_tt_cores), len(right_tt_cores), len(outer_tt_cores)]
    if all_num_cores != [num_cores]*4:
        raise RuntimeError(
            'Orthogonals have different numbers of cores. These should all be equal:\n'
            + '[len(tucker_cores), len(left_tt_cores), len(right_tt_cores), len(outer_tt_cores)]=\n'
            + str(all_num_cores)
        )

    # Check that tucker_cores are matrices
    for ii, B in enumerate(tucker_cores):
        if len(B.shape) != 2:
            raise RuntimeError(
                'tucker_core is not a matrix:\n'
                + 'tucker_cores['+str(ii) + '].shape=' + str(B.shape)
            )

    # Check that outer_tt_cores are 3-tensors
    for ii, G in enumerate(outer_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'outer_tt_core is not a 3-tensor:\n'
                + 'outer_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )

    # Check that left_tt_cores are 3-tensors
    for ii, G in enumerate(left_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'left_tt_core is not a 3-tensor:\n'
                + 'left_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )

    # Check that right_tt_cores are 3-tensors
    for ii, G in enumerate(right_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'right_tt_core is not a 3-tensor:\n'
                + 'right_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )

    # Check outer-left consistency
    for ii in range(1, num_cores):
        GO = outer_tt_cores[ii]
        GL = left_tt_cores[ii-1]
        if GO.shape[0] != GL.shape[2]:
            raise RuntimeError(
                'Inconsistency in outer_tt_core and left_tt_core shapes:\n'
                + str(GO.shape[0]) + ' = GO.shape[0] != GL.shape[2] = ' + str(GL.shape[2]) + '\n'
                + 'left_tt_cores[' + str(ii - 1) + '].shape=' + str(GL.shape) + '\n'
                + 'outer_tt_cores['+str(ii)+'].shape=' + str(GO.shape)
            )

    # Check outer-right consistency
    for ii in range(0, num_cores-1):
        GO = outer_tt_cores[ii]
        GR = right_tt_cores[ii+1]
        if GO.shape[2] != GR.shape[0]:
            raise RuntimeError(
                'Inconsistency in outer_tt_core and right_tt_core shapes:\n'
                + str(GO.shape[2]) + ' = GO.shape[2] != GR.shape[0] = ' + str(GR.shape[0]) + '\n'
                + 'outer_tt_cores['+str(ii)+'].shape=' + str(GO.shape) + '\n'
                + 'right_tt_cores['+str(ii+11)+'].shape=' + str(GR.shape)
            )

    # Check left-left consistency
    for ii in range(1, num_cores):
        GL1 = left_tt_cores[ii-1]
        GL2 = left_tt_cores[ii]
        if GL1.shape[2] != GL2.shape[0]:
            raise RuntimeError(
                'Inconsistency in left_tt_core shapes:\n'
                + str(GL1.shape[2]) + ' = GL1.shape[2] != GL2.shape[0] = ' + str(GL2.shape[0]) + '\n'
                + 'left_tt_cores['+str(ii-1)+'].shape=' + str(GL1.shape) + '\n'
                + 'left_tt_cores['+str(ii)+'].shape=' + str(GL2.shape)
            )

    # Check outer-left consistency
    for ii in range(0, num_cores-1):
        G = left_tt_cores[ii]
        B = tucker_cores[ii]
        if G.shape[1] != B.shape[0]:
            raise RuntimeError(
                'Inconsistency in left_tt_core and tucker_core shapes:\n'
                + str(G.shape[1]) + ' = G.shape[1] != B.shape[0] = ' + str(B.shape[0]) + '\n'
                + 'left_tt_cores['+str(ii)+'].shape=' + str(G.shape) + '\n'
                + 'tucker_cores['+str(ii)+'].shape=' + str(B.shape)
            )

    # Check right-right consistency
    for ii in range(0, num_cores-1):
        GR1 = right_tt_cores[ii]
        GR2 = right_tt_cores[ii+1]
        if GR1.shape[2] != GR2.shape[0]:
            raise RuntimeError(
                'Inconsistency in right_tt_core shapes:\n'
                + str(GR1.shape[2]) + ' = GR1.shape[2] != GR2.shape[0] = ' + str(GR2.shape[0]) + '\n'
                + 'right_tt_cores['+str(ii)+'].shape=' + str(GR1.shape) + '\n'
                + 'right_tt_cores['+str(ii+11)+'].shape=' + str(GR2.shape)
            )

    # Check outer-left consistency
    for ii in range(1, num_cores):
        G = right_tt_cores[ii]
        B = tucker_cores[ii]
        if G.shape[1] != B.shape[0]:
            raise RuntimeError(
                'Inconsistency in right_tt_core and tucker_core shapes:\n'
                + str(G.shape[1]) + ' = G.shape[1] != B.shape[0] = ' + str(B.shape[0]) + '\n'
                + 'right_tt_cores['+str(ii)+'].shape=' + str(G.shape) + '\n'
                + 'tucker_cores['+str(ii)+'].shape=' + str(B.shape)
            )


def check_t3variation(
        variation: T3Variation,
) -> None:
    '''Check that T3Variation core shapes are appropriate.

    Raises
    ------
    RuntimeError
        - Error raised if any variation core shapes are inappropriate
    '''
    var_tucker_cores, var_tt_cores = variation

    if len(var_tucker_cores) != len(var_tt_cores):
        raise RuntimeError(
            str(len(var_tucker_cores)) + ' = len(var_tucker_cores) != len(var_tt_cores) = ' + str(len(var_tt_cores))
        )
    num_cores = len(var_tucker_cores)

    # Check that tucker_cores are matrices
    for ii, B in enumerate(var_tucker_cores):
        if len(B.shape) != 2:
            raise RuntimeError(
                'var_tucker_core is not a matrix:\n'
                + 'var_tucker_cores['+str(ii) + '].shape=' + str(B.shape)
            )

    # Check that outer_tt_cores are 3-tensors
    for ii, G in enumerate(var_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'var_tt_core is not a 3-tensor:\n'
                + 'var_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )


def hole_shapes(
        base: T3Base,
) -> typ.Tuple[
    typ.Tuple[int,...], # variation_tucker_shapes. len=d. elm_len=2
    typ.Tuple[int,...], # variation_tt_shapes. len=d. elm_len=3
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
    >>> (var_tucker_shapes, var_tt_shapes) = bvf.hole_shapes(base)
    >>> print(var_tucker_shapes)
    ((9, 14), (8, 15), (7, 16))
    >>> print(var_tt_shapes)
    ((1, 10, 4), (2, 11, 5), (3, 12, 1))
    '''
    check_t3base(base)
    tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    num_cores = len(tucker_cores)

    variation_tucker_shapes = []
    for ii in range(num_cores):
        n = outer_tt_cores[ii].shape[1]
        N = tucker_cores[ii].shape[1]
        variation_tucker_shapes.append((n,N))

    variation_tt_shapes = []
    for ii in range(num_cores):
        rL = left_tt_cores[ii].shape[0]
        n = tucker_cores[ii].shape[0]
        rR = right_tt_cores[ii].shape[2]
        variation_tt_shapes.append((rL, n, rR))

    return tuple(variation_tucker_shapes), tuple(variation_tt_shapes)


def check_fit(
        variation: T3Variation,
        base: T3Base,
) -> None:
    '''Check that the variation cores fit into the corresponding holes of the base.

    Parameters
    ----------
    variation: T3Variation
        Variation cores
    base: T3Base
        Base cores

    Raises
    ------
    RuntimeError
        - Error raised if the base is internally inconsistent
        - Error raised if the variation is internally incorrect
        - Error raised if the base and variation do not fit with each other

    See Also
    --------
    T3Base
    T3Variation
    check_t3base
    check_t3variation
    '''
    check_t3base(base)
    check_t3variation(variation)

    var_tucker_cores, var_tt_cores = variation
    var_tucker_shapes = tuple([B.shape for B in var_tucker_cores])
    var_tt_shapes = tuple([G.shape for G in var_tt_cores])

    hole_tucker_shapes, hole_tt_shapes = hole_shapes(base)

    if var_tucker_shapes != hole_tucker_shapes:
        raise RuntimeError(
            'Variation Tucker core does do not fit into base:\n'
            + 'var_tucker_shapes=' + str(var_tucker_shapes) + '\n'
            + 'hole_tucker_shapes=' + str(hole_tucker_shapes) + '\n'
        )

    if var_tt_shapes != hole_tt_shapes:
        raise RuntimeError(
            'Variation TT core does do not fit into base:\n'
            + 'var_tt_shapes=' + str(var_tt_shapes) + '\n'
            + 'hole_tt_shapes=' + str(hole_tt_shapes) + '\n'
        )


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
    >>> (L0,L1,L2) = (randn(1, 10, 2), randn(2, 11, 3), randn(3, 12, 1))
    >>> (R0,R1,R2) = (randn(1, 10, 4), randn(4, 11, 5), randn(5, 12, 1))
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
    check_t3base(base)
    check_t3variation(variation)

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

