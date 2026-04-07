# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
import t3tools.tucker_tensor_train as t3

try:
    import jax.numpy as jnp
except:
    print('jax import failed. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]


__all__ = [
    # Base-variation format
    'T3Base',
    'T3Variation',
    'check_t3base',
    'check_t3variation',
    'hole_shapes',
    'check_fit',
    'ith_bv_to_t3',
    'orthogonal_representations',
    # Tangent vectors
    'tangent_to_dense',
    'tangent_to_t3',
    'tangent_zeros',
    'tangent_randn',
    # Projection and retraction
    'orthogonal_gauge_projection',
    'oblique_gauge_projection',
    'project_t3_onto_tangent_space',
    'retract',
]

################################################################
########    TuckerTensorTrain base-variation format    #########
################################################################

T3Base = typ.Tuple[
    typ.Sequence[NDArray],  # base_basis_cores. B_xo B_yo = I_xy    B.shape = (n, N)
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
    - basis_cores       = (U0, ..., Ud), elm_shape=(ni, Ni)
    - left_tt_cores     = (L0, ..., Ld), elm_shape=(rLi, ni, rL(i+1))
    - right_tt_cores    = (R0, ..., Rd), elm_shape=(rRi, ni, rR(i+1))
    - outer_tt_cores    = (O0, ..., Od), elm_shape=(rLi, nOi, rR(i+1))

The components of T3Variations are the "variation cores":
    - basis_variations  = (V0, ..., Vd), elm_shape=(nOi, Ni)
    - tt_variations     = (H0, ..., Hd), elm_shape=(rLi, ni, rRi)

A tangent vector can be written as the sum of all of the tensor diagrams above. 
In this case, the base cores are representations of the point where the 
tangent space attaches to the manifold, and the variation cores define the 
tangent vector with respect to the base cores. 

Often, it is desirable for the basis cores to be **orthogonal** as follows:
    - basis_cores       = (U0,...,Ud), orthogonal:       U_ia U_ja = delta_ij
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
>>> import t3tools.t3_manifold as t3m
>>> basis_cores = (np.ones((10, 14)), np.ones((11, 15)), np.ones((12, 16)))
>>> left_tt_cores = (np.ones((1, 10, 2)), np.ones((2, 11, 3)), np.ones((3, 12, 1)))
>>> right_tt_cores = (np.ones((1, 10, 4)), np.ones((4, 11, 5)), np.ones((5, 12, 1)))
>>> outer_tt_cores = (np.ones((1, 9, 4)), np.ones((2, 8, 5)), np.ones((3, 7, 1)))
>>> base = (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
>>> t3m.check_t3base(base) # Does nothing since base is internally consistent
>>> var_basis_cores = (np.ones((9,14)), np.ones((8,15)), np.ones((7,16)))
>>> var_tt_cores = (np.ones((1,10,4)), np.ones((2,11,5)), np.ones((3,12,1)))
>>> variation = (var_basis_cores, var_tt_cores)
>>> t3m.check_t3variation(variation) # Does nothing since variation is internally consistent
>>> t3m.check_fit(base, variation) # Does nothing since variation fits in base
"""


T3Variation = typ.Tuple[
    typ.Sequence[NDArray],  # variation_basis_cores.
    typ.Sequence[NDArray],  # variation_tt_cores.
]
"""
Tuple containing variation cores for base-variation representation of TuckerTensorTrains.

*Components*
    - basis_variations  = (V0, ..., Vd), elm_shape=(nOi, Ni)
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
        - basis_cores       = (U0, U1, U2, U3)
        - left_tt_cores     = (L0, L1, L2, L3)
        - right_tt_cores    = (R0, R1, R2, R3)
        - outer_tt_cores    = (O0, O1, O2, O3)

    Raises
    ------
    RuntimeError
        - Error raised if any core shapes are inconsistent
    '''
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base

    num_cores = len(basis_cores)
    all_num_cores = [len(basis_cores), len(left_tt_cores), len(right_tt_cores), len(outer_tt_cores)]
    if all_num_cores != [num_cores]*4:
        raise RuntimeError(
            'Orthogonals have different numbers of cores. These should all be equal:\n'
            + '[len(basis_cores), len(left_tt_cores), len(right_tt_cores), len(outer_tt_cores)]=\n'
            + str(all_num_cores)
        )

    # Check that basis_cores are matrices
    for ii, B in enumerate(basis_cores):
        if len(B.shape) != 2:
            raise RuntimeError(
                'basis_core is not a matrix:\n'
                + 'basis_cores['+str(ii) + '].shape=' + str(B.shape)
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
        B = basis_cores[ii]
        if G.shape[1] != B.shape[0]:
            raise RuntimeError(
                'Inconsistency in left_tt_core and basis_core shapes:\n'
                + str(G.shape[1]) + ' = G.shape[1] != B.shape[0] = ' + str(B.shape[0]) + '\n'
                + 'left_tt_cores['+str(ii)+'].shape=' + str(G.shape) + '\n'
                + 'basis_cores['+str(ii)+'].shape=' + str(B.shape)
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
        B = basis_cores[ii]
        if G.shape[1] != B.shape[0]:
            raise RuntimeError(
                'Inconsistency in right_tt_core and basis_core shapes:\n'
                + str(G.shape[1]) + ' = G.shape[1] != B.shape[0] = ' + str(B.shape[0]) + '\n'
                + 'right_tt_cores['+str(ii)+'].shape=' + str(G.shape) + '\n'
                + 'basis_cores['+str(ii)+'].shape=' + str(B.shape)
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
    var_basis_cores, var_tt_cores = variation

    if len(var_basis_cores) != len(var_tt_cores):
        raise RuntimeError(
            str(len(var_basis_cores)) + ' = len(var_basis_cores) != len(var_tt_cores) = ' + str(len(var_tt_cores))
        )
    num_cores = len(var_basis_cores)

    # Check that basis_cores are matrices
    for ii, B in enumerate(var_basis_cores):
        if len(B.shape) != 2:
            raise RuntimeError(
                'var_basis_core is not a matrix:\n'
                + 'var_basis_cores['+str(ii) + '].shape=' + str(B.shape)
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
    typ.Tuple[int,...], # variation_basis_shapes. len=d. elm_len=2
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
        - basis_cores       = (U0, U1, U2, U3)
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
        Variation basis core shapes. len=d. elm_len=2
    typ.Tuple[int,...]
        Variation TT core shapes. len=d. elm_len=3

    Raises
    ------
    RuntimeError
        - Error raised if any base_core shapes are inconsistent

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.manifold as t3m
    >>> basis_cores = (np.ones((10,14)), np.ones((11,15)), np.ones((12,16)))
    >>> left_tt_cores = (np.ones((1,10,2)), np.ones((2,11,3)), np.ones((3,12,1)))
    >>> right_tt_cores = (np.ones((1,10,4)), np.ones((4,11,5)), np.ones((5,12,1)))
    >>> outer_tt_cores = (np.ones((1,9,4)), np.ones((2,8,5)), np.ones((3,7,1)))
    >>> base = (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    >>> (var_basis_shapes, var_tt_shapes) = t3m.hole_shapes(base)
    >>> print(var_basis_shapes)
    ((9, 14), (8, 15), (7, 16))
    >>> print(var_tt_shapes)
    ((1, 10, 4), (2, 11, 5), (3, 12, 1))
    '''
    check_t3base(base)
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    num_cores = len(basis_cores)

    variation_basis_shapes = []
    for ii in range(num_cores):
        n = outer_tt_cores[ii].shape[1]
        N = basis_cores[ii].shape[1]
        variation_basis_shapes.append((n,N))

    variation_tt_shapes = []
    for ii in range(num_cores):
        rL = left_tt_cores[ii].shape[0]
        n = basis_cores[ii].shape[0]
        rR = right_tt_cores[ii].shape[2]
        variation_tt_shapes.append((rL, n, rR))

    return tuple(variation_basis_shapes), tuple(variation_tt_shapes)


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

    var_basis_cores, var_tt_cores = variation
    var_basis_shapes = tuple([B.shape for B in var_basis_cores])
    var_tt_shapes = tuple([G.shape for G in var_tt_cores])

    hole_basis_shapes, hole_tt_shapes = hole_shapes(base)

    if var_basis_shapes != hole_basis_shapes:
        raise RuntimeError(
            'Variation basis does do not fit into base:\n'
            + 'var_basis_shapes=' + str(var_basis_shapes) + '\n'
            + 'hole_basis_shapes=' + str(hole_basis_shapes) + '\n'
        )

    if var_tt_shapes != hole_tt_shapes:
        raise RuntimeError(
            'Variation tt does do not fit into base:\n'
            + 'var_tt_shapes=' + str(var_tt_shapes) + '\n'
            + 'hole_tt_shapes=' + str(hole_tt_shapes) + '\n'
        )


def ith_bv_to_t3(
        replacement_ind: int,
        replace_tt: bool, # If True, replace TT-core. If False, replace basis_core.
        base: T3Base,
        variation: T3Variation,
) -> t3.TuckerTensorTrain:
    '''Convert basis-variation representation to TuckerTensorTrain.

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
        Indicates whether to replace a TT-core (True) or a basis core (False)
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
    >>> import t3tools.manifold as t3m
    >>> randn = np.random.randn # shorthand
    >>> (U0,U1,U2) = (randn(10, 14), randn(11, 15), randn(12, 16))
    >>> (L0,L1,L2) = (randn(1, 10, 2), randn(2, 11, 3), randn(3, 12, 1))
    >>> (R0,R1,R2) = (randn(1, 10, 4), randn(4, 11, 5), randn(5, 12, 1))
    >>> (O0,O1,O2) = (randn(1, 9, 4), randn(2, 8, 5), randn(3, 7, 1))
    >>> base = ((U0,U1,U2), (L0,L1,L2), (R0,R1,R2), (O0,O1,O2))
    >>> (V0,V1,V2) = (randn(9,14), randn(8,15), randn(7,16))
    >>> (H0,H1,H2) = (randn(1,10,4), randn(2,11,5), randn(3,12,1))
    >>> variation = ((V0,V1,V2), (H0,H1,H2))
    >>> ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(1, True, base, variation) # replace index-1 TT-core
    >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,U1,U2), (L0,H1,R2)))
    True
    >>> ((B0, B1, B2), (G0, G1, G2)) = t3m.ith_bv_to_t3(1, False, base, variation) # replace index-1 basis core
    >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,V1,U2), (L0,O1,R2)))
    True
    '''
    check_t3base(base)
    check_t3variation(variation)

    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    basis_vars, tt_vars = variation

    if replace_tt:
        x_basis_cores = basis_cores
        x_tt_cores = (
                tuple(left_tt_cores[:replacement_ind]) +
                (tt_vars[replacement_ind],) +
                tuple(right_tt_cores[replacement_ind+1:])
        )
    else:
        x_basis_cores = (
            tuple(basis_cores[:replacement_ind]) +
            (basis_vars[replacement_ind],) +
            tuple(basis_cores[replacement_ind+1:])
        )
        x_tt_cores = (
                tuple(left_tt_cores[:replacement_ind]) +
                (outer_tt_cores[replacement_ind],) +
                tuple(right_tt_cores[replacement_ind+1:])
        )

    return (x_basis_cores, x_tt_cores)


def orthogonal_representations(
        x: t3.TuckerTensorTrain,
        use_jax: bool = False,
) -> typ.Tuple[
    T3Base, # orthogonal base
    T3Variation, # variations
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

    Base-variation representation with non-orthogonal basis core V2::

                  1 -- L0 -- L1 -- O2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |

    The input tensor train x is defined by:
        - x_basis_cores     = (B0, B1, B2, B3)
        - x_tt_cores        = (G0, G1, G2, G3)
    The "base cores" are:
        - basis_cores       = (U0,U1, U2, U3), up orthogonal
        - left_tt_cores     = (L0, L1, L2, L3), left orthogonal
        - right_tt_cores    = (R0, R1, R2, R3), right orthogonal
        - outer_tt_cores    = (O0, O1, O2, O3), down orthogonal
    The "variation cores" are:
        - basis_variations  = (V0, V1, V2, V3)
        - tt_variations     = (H0, H1, H2, H3)

    Parameters
    ----------
    x: TuckerTensorTrain
        Input TuckerTensorTrain
        x = (x_basis_cores, x_tt_cores)
        x_basis_cores = (B0, ..., Bd)
        x_tt_cores = (G0, ..., Gd)
    use_jax: bool
        If True use jax operations, if False use numpy.

    Returns
    -------
    T3Base
        Orthogonal base for base-variation representations of x.
    T3Variation
        Variation for base-variation representaions of x.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, variation = t3m.orthogonal_representations(x) # Compute orthogonal representations
    >>> basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    >>> basis_vars, tt_vars = variation
    >>> (U0,U1,U2) = basis_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (O0,O1,O2) = outer_tt_cores
    >>> (V0,V1,V2) = basis_vars
    >>> (H0,H1,H2) = tt_vars
    >>> x2 = ((U0,U1,U2), (L0,H1,R2)) # representation with TT-core variation in index 1
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Still represents origional tensor
    4.978421562425667e-12
    >>> x3 = ((U0,V1,U2), (L0,O1,R2)) # representation with basis core variation in index 1
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x3))) # Still represents origional tensor
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> base, variation = t3m.orthogonal_representations(x) # Compute orthogonal representations
    >>> basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    >>> basis_vars, tt_vars = variation
    >>> (U0,U1,U2) = basis_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (O0,O1,O2) = outer_tt_cores
    >>> (V0,V1,V2) = basis_vars
    >>> (H0,H1,H2) = tt_vars
    >>> x2 = ((U0,U1,U2), (L0,H1,R2)) # representation with TT-core variation in index 1
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Still represents origional tensor
    2.5341562994067855e-12
    >>> x3 = ((V0,U1,U2), (O0,R1,R2)) # representation with basis core variation in index 0
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x3))) # Still represents origional tensor
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
    t3.check_t3(x)

    num_cores = len(x[1])

    # Orthogonalize basis matrices
    for ii in range(num_cores):
        x = t3.up_svd_ith_basis_core(ii, x, use_jax=use_jax)[0]
    basis_cores = tuple([U.copy() for U in x[0]])

    # Right orthogonalize
    for ii in range(num_cores-1, 0, -1): # num_cores-1, num_cores-2, ..., 1
        x = t3.right_svd_ith_tt_core(ii, x, use_jax=use_jax)[0]
    right_tt_cores = tuple([G.copy() for G in x[1]])

    basis_variations = []
    tt_variations = []

    left_tt_cores = []
    outer_tt_cores = []
    # Sweep left to right
    for ii in range(num_cores):
        tt_variations.append(x[1][ii])

        tmp = t3.down_svd_ith_tt_core(ii, x, use_jax=use_jax)[0]
        outer_tt_cores.append(tmp[1][ii])
        basis_variations.append(tmp[0][ii])

        if ii < num_cores-1:
            x = t3.left_svd_ith_tt_core(ii, x, use_jax=use_jax)[0]
        left_tt_cores.append(x[1][ii])

    base = (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    variation = (basis_variations, tt_variations)
    return base, variation



####################################################################
##################    Tangent vectors operations  ##################
####################################################################

def tangent_to_dense(
        variation: T3Variation,
        base: T3Base,
        include_shift: bool = False, # False: V. True: P+V. P=base point, V=tangent vector
) -> NDArray:
    """Convert Tangent vector to Tucker tensor train manifold into dense tensor.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> base, _ = t3m.orthogonal_representations(p)
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
    check_fit(variation, base)

    num_cores = len(variation[0])
    basis_terms = [ith_bv_to_t3(ii, False, base, variation) for ii in range(num_cores)]
    tt_terms    = [ith_bv_to_t3(ii, True, base, variation) for ii in range(num_cores)]
    terms = basis_terms + tt_terms
    V = t3.t3_to_dense(terms[0])
    for t in terms[1:]:
        V = V + t3.t3_to_dense(t)

    if include_shift:
        basis_cores, left_tt_cores, _, _ = base
        P = t3.t3_to_dense((basis_cores, left_tt_cores))
        X = P + V
    else:
        X = V

    return X


def tangent_to_t3(
        variation: T3Variation,
        base: T3Base,
        include_shift: bool = False,  # False: v. True: p+v. p=base point, v=tangent vector
        use_jax: bool = False,
) -> t3.TuckerTensorTrain:
    '''Rank 2r Tucker tensor train representation of tangent vector.

    Without shift, we use the formula::

        v(x,y,z,w) = ([dU1(B x) L1(B x)]) ([R2(B y)        0]) ([R3(B z)        0]) ([R4(B w) ])
                     (                  ) ([dU2(B y) L2(B y)]) ([dU3(B z) L3(B z)]) ([dU4(B w)])
                     (         +        ) (         +        ) (        +         ) (    +     )
                     ([O1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
                     (                  ) ([O2(dB y)       0]) ([O3(dB z)       0]) ([O4(dB w)])

    With shift is same as unshifted, except last core modified as follows::

        [R4(B w) ]                  [R4(B w)           ]
        [dU4(B w)]                  [L4(B w) + dU4(B w)]
            +             ->            +
        [0       ]                  [0                 ]
        [O4(dB w)]                  [O4(dB w)          ]

    Parameters
    ----------
    variation: T3Variation,
        Variation representing the tangent vector
    base: T3Base,
        Representation of the base point at which the tangent space attaches to the manifold.
    include_shift: bool
        If False, return tangent vector v only. If True, shift tangent vector so it is attached at the base point, p+v.
    use_jax: bool
        If True, returned TuckerTensorTrain cores are jnp.ndaray. Otherwise, np.ndarray. Default: False

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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> base, _ = t3m.orthogonal_representations(p)
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
    xnp = jnp if use_jax else np

    check_fit(variation, base)
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    basis_vars, tt_vars = variation

    num_cores = len(basis_cores)

    x_basis_cores = []
    for B, dB in zip(basis_cores, basis_vars):
        B2 = xnp.concatenate([B, dB], axis=0)
        x_basis_cores.append(B2)

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
    L = left_tt_cores[-1]
    R = right_tt_cores[-1]
    O = outer_tt_cores[-1]
    Z = xnp.zeros((R.shape[0], O.shape[1], R.shape[2]))
    if include_shift:
        G_top = xnp.concatenate([R, L + dU], axis=0)
    else:
        G_top = xnp.concatenate([R, dU], axis=0)
    G_bot = xnp.concatenate([Z, O], axis=0)
    G = xnp.concatenate([G_top, G_bot], axis=1)
    x_tt_cores.append(G)

    return tuple(x_basis_cores), tuple(x_tt_cores)


def tangent_zeros(
        base: T3Base, # orthogonal base
        use_jax: bool = False,
) -> T3Variation:
    """Construct the zero vector in a Tucker tensor train tangent space.

    Parameters
    ----------
    base: T3Base
        Representations of base point on manifold where tangent space is attached
    use_jax: bool
        If True, return jax arrays, if False return numpy.

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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = t3m.orthogonal_representations(p)
    >>> z = t3m.tangent_zeros(base)
    >>> print(np.linalg.norm(t3m.tangent_to_dense(z, base)))
    0.0
    """
    xnp = jnp if use_jax else np

    check_t3base(base)

    var_basis_shapes, var_tt_shapes = hole_shapes(base)

    basis_vars = tuple([xnp.zeros(s) for s in var_basis_shapes])
    tt_vars = tuple([xnp.zeros(s) for s in var_tt_shapes])

    zero_variation = (basis_vars, tt_vars)
    return zero_variation


def tangent_randn(
        base: T3Base, # orthogonal base
        use_jax: bool = False,
        apply_gauge_projection: bool = True,
) -> T3Variation:
    """Draw a random T3Variation.

    Parameters
    ----------
    orthogonal_base: T3Base
        Representations of base point on manifold where tangent space is attached.

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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, vars0 = t3m.orthogonal_representations(p)
    >>> x = t3m.tangent_randn(base) # Random tangent vector, gauged.

    Don't apply Gauge projection:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, vars0 = t3m.orthogonal_representations(p)
    >>> x = t3m.tangent_randn(base, apply_gauge_projection=False) # Random tangent vector, ungauged
    """
    check_t3base(base)

    var_basis_shapes, var_tt_shapes = hole_shapes(base)

    if use_jax:
        _randn = lambda x: jnp.array(np.random.randn(x))
    else:
        _randn = np.random.randn

    basis_vars0 = tuple([_randn(*s) for s in var_basis_shapes])
    tt_vars0 = tuple([_randn(*s) for s in var_tt_shapes])

    variation = (basis_vars0, tt_vars0)
    if apply_gauge_projection:
        variation = orthogonal_gauge_projection(variation, base)
    return variation


####################################################################
#################    Projection and retraction   ###################
####################################################################

def orthogonal_gauge_projection(
        variation: T3Variation,
        orthogonal_base: T3Base,
        use_jax: bool = False,
) -> T3Variation:
    """Makes tangent variation gauged via orthogonal projection. Changes tangent vector.

    Gauge condition:
        - All variation basis cores Vi are orthogonal to the corresponding base basis cores Ui:
            Ui @ Vi.T = 0    for    i=1,...,d
        - All but the last variation TT-cores H are left-perpendicular to the corresponding base left TT-cores L:
            einsum('iaj,iak->jk', Hi, Li) = 0    for    i=1,...,d-1

    Parameters
    ----------
    variation: T3Variation,
        The variation which will become gauged.
    orthogonal_base: T3Base,
        The base representations. Must be orthogonal for the operation to work properly.
    use_jax: bool
        If True, use jax operations, if False use numpy.

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

    Example
    -------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = t3m.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> proj_variation = t3m.orthogonal_gauge_projection(variation, base) # Make gauged via orthogonal projection
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = proj_variation
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for basis core 1
    3.512073125137391e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-core 1
    1.5807940730805242e-15
    """
    xnp = jnp if use_jax else np

    check_fit(variation, orthogonal_base)
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    basis_vars, tt_vars = variation

    new_tt_variations = []
    for dV, P in zip(tt_vars[:-1], left_tt_cores[:-1]):
        dV2 = dV - xnp.einsum('iaj,jk->iak', P, xnp.einsum('iaj,iak->jk', P, dV))
        new_tt_variations.append(dV2)
    new_tt_variations.append(tt_vars[-1])

    new_basis_variations = []
    for dB, B in zip(basis_vars, basis_cores):
        dB2 = dB - (dB @ B.T) @ B
        new_basis_variations.append(dB2)

    return tuple(new_basis_variations), tuple(new_tt_variations)


def oblique_gauge_projection(
        variation: T3Variation,
        orthogonal_base: T3Base,
        use_jax: bool = False,
) -> T3Variation:
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
    use_jax: bool
        If True, use jax operations, if False use numpy.

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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = t3m.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> proj_variation = t3m.oblique_gauge_projection(variation, base) # Make gauged via oblique projection
    >>> v_dense = t3m.tangent_to_dense(variation, base)
    >>> proj_v_dense = t3m.tangent_to_dense(proj_variation, base)
    >>> print(np.linalg.norm(v_dense - proj_v_dense)) # Zero since projection preserves represented tangent vector
    3.4398319441148304e-15
    >>> (U0,U1,U2), (L0,L1,L2), _, _ = base
    >>> ((V0,V1,V2), (H0,H1,H2)) = proj_variation
    >>> print(np.linalg.norm(V1 @ U1.T)) # Gauge condition for basis core 1
    2.931519226677228e-15
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', H1, L1))) # Gauge condition for TT-core 1
    6.99005312491287e-16

    With minimal ranks, orthogonal bases, and gauged variations, the corewise dot product faithfully represents
    the Hilbert-Schmidt inner product on the ambient space:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> import t3tools.util
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = t3m.orthogonal_representations(p)
    >>> u = t3m.tangent_randn(base, apply_gauge_projection=False)
    >>> v = t3m.tangent_randn(base, apply_gauge_projection=False)
    >>> bad_u_inner_v = util.corewise_dot(u, v) # u and v are ungauged, so this will not give the right answer
    >>> u_dense = t3m.tangent_to_dense(u, base)
    >>> v_dense = t3m.tangent_to_dense(v, base)
    >>> u_inner_v_true = np.sum(u_dense * v_dense)
    >>> print(np.abs(bad_u_inner_v - u_inner_v_true)) # error nonzero because we didn't respect gauge
    6.21838915941413
    >>> u_gauged = t3m.oblique_gauge_projection(u, base) # make them gauged and try again
    >>> v_gauged = t3m.oblique_gauge_projection(v, base)
    >>> u_inner_v = util.corewise_dot(u_gauged, v_gauged)
    >>> print(np.abs(u_inner_v - u_inner_v_true)) # Now the error is numerical zero
    0.0
    """
    xnp = jnp if use_jax else np

    check_fit(variation, orthogonal_base)
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    basis_vars, tt_vars = variation
    num_cores = len(basis_cores)

    tt_vars = list(tt_vars)
    basis_vars = list(basis_vars)

    # Make basis variations left-perpendicular
    for ii in range(num_cores):
        B_io = basis_cores[ii]
        dB_jo = basis_vars[ii]
        R_aib = outer_tt_cores[ii]
        dG_ajb = tt_vars[ii]

        X_ji = dB_jo @ B_io.T
        dB_parallel_jo = X_ji @ B_io
        dB2_jo = dB_jo - dB_parallel_jo # dB_perp
        dG2_ajb = dG_ajb + xnp.einsum('aib,ij->ajb', R_aib, X_ji)

        tt_vars[ii] = dG2_ajb
        basis_vars[ii] = dB2_jo

    # Make tt cores left-perpendicular
    for ii in range(num_cores-1):
        dV1 = tt_vars[ii]
        dV2 = tt_vars[ii+1]

        P1 = left_tt_cores[ii]
        Q2 = right_tt_cores[ii+1]
        X = xnp.einsum('iaj,iak->jk', P1, dV1)
        new_dV1 = dV1 - xnp.einsum('iaj,jk->iak', P1, X)
        new_dV2 = dV2 + xnp.einsum('jk,kbl->jbl', X, Q2)

        tt_vars[ii] = new_dV1
        tt_vars[ii+1] = new_dV2

    return tuple(basis_vars), tuple(tt_vars)


def tt_reverse(cores):
    return tuple([G.swapaxes(0, 2) for G in cores[::-1]])


def tt_zipper_left_to_right(
        coresA: typ.Sequence[NDArray],
        coresB: typ.Sequence[NDArray],
        use_jax: bool = False,
) -> typ.Tuple[NDArray, ...]:  # zipper_matrices. len=num_cores+1
    xnp = jnp if use_jax else np

    zipper_matrices = [xnp.array([[1.0]])]
    for GA, GB in zip(coresA, coresB):
        Z_prev = zipper_matrices[-1]
        Z = xnp.einsum('ij,iak,jal->kl', Z_prev, GA, GB)
        zipper_matrices.append(Z)
    return tuple(zipper_matrices)


def tt_zipper_right_to_left(
        coresA: typ.Sequence[NDArray],
        coresB: typ.Sequence[NDArray],
        use_jax: bool = False,
) -> typ.Tuple[NDArray, ...]:  # zipper_matrices. len=num_cores+1
    return tt_zipper_left_to_right(tt_reverse(coresA), tt_reverse(coresB), use_jax=use_jax)[::-1]


def project_t3_onto_tangent_space(
        x: t3.TuckerTensorTrain, # Tucker tensor train to be projected
        orthogonal_base: T3Base, # Orthogonal representations of base point
        use_jax: bool = False,
) -> T3Variation:
    """Projects TuckerTensorTrain onto tangent space to the manifold of fixed rank TuckerTensorTrains.

    Parameters
    ----------
    x: t3.TuckerTensorTrain
        TuckerTensorTrain to project
    orthogonal_base: T3Base
        Minimal rank orthogonal representations of base point on manifold where tangent space is attached
    use_jax: bool
        If True, use jax operations, if False use numpy.

    Returns
    -------
    T3Variation
        Gauged variation representing the orthogonal projection of x onto the tangent space.

    See Also
    --------
    T3Base
    oblique_gauge_projection
    orthogonal_gauge_projection

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = t3m.orthogonal_representations(p)
    >>> x = t3.t3_corewise_randn(((14,15,16), (7,4,8), (2,5,4,2)))
    >>> proj_x = t3m.project_t3_onto_tangent_space(x, base) # Project x onto tangent space
    >>> P = t3.t3_to_dense(p)
    >>> X = t3.t3_to_dense(x)
    >>> proj_X = t3m.tangent_to_dense(proj_x, base)
    >>> print(np.sum((X - proj_X) * (proj_X - P)) / np.sum(X)) # Check that x was projected orthogonally
    -2.7295025395842007e-13
    """
    xnp = jnp if use_jax else np

    t3.check_t3(x)
    check_t3base(orthogonal_base)

    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = orthogonal_base
    other_basis_cores, other_tt_cores = x

    base_shape = tuple([B.shape[1] for B in basis_cores])
    other_shape = tuple([B.shape[1] for B in other_basis_cores])
    if base_shape != other_shape:
        raise RuntimeError(
            'Attempted to retract TuckerTensorTrain with wrong shape onto tangent space.\n'
            + str(base_shape) + ' = base_shape != other_shape = ' + str(other_shape)
        )

    other_tt_cores2 = []
    for G_other, B_other, B in zip(other_tt_cores, other_basis_cores, basis_cores):
        G_other2 = xnp.einsum('aib,ix->axb', G_other, B_other @ B.T)
        other_tt_cores2.append(G_other2)

    zipper_left2right = tt_zipper_left_to_right(other_tt_cores2, left_tt_cores, use_jax=use_jax)[:-1]
    zipper_right2left = tt_zipper_right_to_left(other_tt_cores2, right_tt_cores, use_jax=use_jax)[1:]

    ungauged_tt_variations = []
    ungauged_basis_variations = []
    for ZL_ax, ZR_by, G_aib, B_io, R0_xjy, B0_jo in zip(
            zipper_left2right, zipper_right2left,
            other_tt_cores, other_basis_cores,
            outer_tt_cores, basis_cores,
    ):
        X_xiy = xnp.einsum('ax,aib,by->xiy', ZL_ax, G_aib, ZR_by)
        dG_xjy = xnp.einsum('xiy,ij->xjy', X_xiy, B_io @ B0_jo.T)
        M_ij = xnp.einsum('xiy,xjy->ij', X_xiy, R0_xjy)
        dB_jo = xnp.einsum('ij,io->jo', M_ij, B_io)

        ungauged_tt_variations.append(dG_xjy)
        ungauged_basis_variations.append(dB_jo)

    ungauged_u = (ungauged_basis_variations, ungauged_tt_variations)
    gauged_u = orthogonal_gauge_projection(ungauged_u, orthogonal_base)
    return gauged_u


def retract(
        variation: T3Variation,
        base: T3Base,
        use_jax: bool = False,
) -> t3.TuckerTensorTrain: # retracted Tucker tensor train
    """Retract Tucker tensor train tangent vector to manifold.

    Parameters
    ----------
    variation: T3Variation,
        Variation representing the tangent vector we wish to retract to the manifold
    base: T3Base,
        Representation of the base point on the manifold where the tangent space is attached.
    use_jax: bool
        If True, use jax operations, if False use numpy.

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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> import t3tools.util
    >>> p = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, _ = t3m.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base) # Random tangent vector
    >>> ret_v = t3m.retract(variation, base) # Retract tangent vector to manifold
    >>> ret_V = t3.t3_to_dense(ret_v)
    >>> V = t3m.tangent_to_dense(variation, base, include_shift=True)
    >>> print(np.linalg.norm(ret_V - V)) # vector changes
    0.14335564543255402
    >>> v2 = util.corewise_scale(variation, 1e-2) # make the tangent vector shorter for smaller retraction
    >>> ret_v2 = t3m.retract(v2, base)
    >>> ret_V2 = t3.t3_to_dense(ret_v2)
    >>> V2 = t3m.tangent_to_dense(v2, base, include_shift=True)
    >>> print(np.linalg.norm(ret_V2 - V2)) # vector changes
    4.9488133126395654e-05
    """
    check_fit(variation, base)

    basis_cores, left_tt_cores, _, _ = base
    _, base_tucker_ranks, base_tt_ranks = t3.structure((basis_cores, left_tt_cores))

    x_t3 = tangent_to_t3(variation, base, include_shift=True)
    retracted_x_t3, _, _ = t3.t3_svd(
        x_t3,
        max_tt_ranks = base_tt_ranks,
        max_tucker_ranks = base_tucker_ranks,
        use_jax=use_jax,
    )
    return retracted_x_t3

