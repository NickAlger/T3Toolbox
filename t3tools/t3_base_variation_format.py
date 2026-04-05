# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
from t3tools.linalg import *
from t3tools.tucker_tensor_train import *

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]

__all__ = [
    'T3Base',
    'T3Variation',
    't3_check_base',
    't3_check_variation',
    't3_base_hole_shapes',
    't3_check_base_variation_fit',
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

    1 -- GL0 -- X1 -- GR2 -- GR3 -- 1
         |      |     |      |
         B0     B1    B2     B3
         |      |     |      |

    1 -- GL0 -- GL1 -- GO2 -- GR3 -- 1
         |      |      |      |
         B0     B1     Y2     B3
         |      |      |      |

The "basis cores" are:
    - basis_cores       = (B0, B1, B2, B3)
    - left_tt_cores     = (GL0, GL1, GL2, GL3)
    - right_tt_cores    = (GR0, GR1, GR2, GR3)
    - outer_tt_cores    = (GO0, GO1, GO2, GO3)
The "variation cores" are:
    - basis_variations  = (Y0, Y1, Y2, Y3)
    - tt_variations     = (X0, X1, X2, X3)

See Also
--------
T3Variation
t3_check_base
t3_check_variation
t3_check_base_variation_fit

Examples
--------
>>> from numpy.random import randn
>>> from t3tools.tucker_tensor_train import *
>>> from t3tools.t3_base_variation_format import *
>>> basis_cores = (randn(10, 14), randn(11, 15), randn(12, 16))
>>> left_tt_cores = (randn(1, 10, 2), randn(2, 11, 3), randn(3, 12, 1))
>>> right_tt_cores = (randn(1, 10, 4), randn(4, 11, 5), randn(5, 12, 1))
>>> outer_tt_cores = (randn(1, 9, 4), randn(2, 8, 5), randn(3, 7, 1))
>>> base = (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
>>> t3_check_base(base) # Does nothing since base is internally consistent
>>> var_basis_cores = (randn(9,14), randn(8,15), randn(7,16))
>>> var_tt_cores = (randn(1,10,4), randn(2,11,5), randn(3,12,1))
>>> variation = (var_basis_cores, var_tt_cores)
>>> t3_check_variation(variation) # Does nothing since variation is internally consistent
>>> t3_check_base_variation_fit(base, variation) # Does nothing since variation fits in base
"""


T3Variation = typ.Tuple[
    typ.Sequence[NDArray],  # variation_basis_cores.
    typ.Sequence[NDArray],  # variation_tt_cores.
]
"""
Tuple containing variation cores for base-variation representation of TuckerTensorTrains

See Also
--------
T3Base
t3_check_base
t3_check_variation
t3_check_base_variation_fit
"""


def t3_check_base(
        base: T3Base,
) -> None:
    '''Check that T3Base core shapes are internally consistent.

    Contractions of the following forms must make sense::

        1 -- GL0 -- ( ) -- GR2 -- GR3 -- 1
             |      |      |      |
             B0     B1     B2     B3
             |      |      |      |

        1 -- GL0 -- GL1 -- GO2 -- GR3 -- 1
             |      |      |     |
             B0     B1     ( )   B3
             |      |      |     |


    Here:
        - basis_cores       = (B0, B1, B2, B3)
        - left_tt_cores     = (GL0, GL1, GL2, GL3)
        - right_tt_cores    = (GR0, GR1, GR2, GR3)
        - outer_tt_cores    = (GO0, GO1, GO2, GO3)

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

    # Check that outer_tt_cores are 3-tensors with leading and trailing 1 dims
    for ii, G in enumerate(outer_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'outer_tt_core is not a 3-tensor:\n'
                + 'outer_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )

    if outer_tt_cores[0].shape[0] != 1:
        raise RuntimeError(
            'First outer_tt_core must have shape (1, . , .).\n'
            + 'outer_tt_cores[0].shape=' + str(outer_tt_cores[0].shape)
        )

    if outer_tt_cores[-1].shape[2] != 1:
        raise RuntimeError(
            'Last outer_tt_core must have shape ( . , . , 1).\n'
            + 'outer_tt_cores[-1].shape=' + str(outer_tt_cores[-1].shape)
        )

    # Check that left_tt_cores are 3-tensors with leading and trailing 1 dims
    for ii, G in enumerate(left_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'left_tt_core is not a 3-tensor:\n'
                + 'left_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )

    if left_tt_cores[0].shape[0] != 1:
        raise RuntimeError(
            'First left_tt_core must have shape (1, . , .).\n'
            + 'left_tt_cores[0].shape=' + str(left_tt_cores[0].shape)
        )

    if left_tt_cores[-1].shape[2] != 1:
        raise RuntimeError(
            'Last left_tt_core must have shape ( . , . , 1).\n'
            + 'left_tt_cores[-1].shape=' + str(left_tt_cores[-1].shape)
        )

    # Check that right_tt_cores are 3-tensors with leading and trailing 1 dims
    for ii, G in enumerate(right_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'right_tt_core is not a 3-tensor:\n'
                + 'right_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )

    if right_tt_cores[0].shape[0] != 1:
        raise RuntimeError(
            'First right_tt_core must have shape (1, . , .).\n'
            + 'right_tt_cores[0].shape=' + str(right_tt_cores[0].shape)
        )

    if right_tt_cores[-1].shape[2] != 1:
        raise RuntimeError(
            'Last right_tt_core must have shape ( . , . , 1).\n'
            + 'right_tt_cores[-1].shape=' + str(right_tt_cores[-1].shape)
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


def t3_check_variation(
        variation: T3Variation,
) -> None:
    '''Check that T3Variation core shapes are appropriate.

    Raises
    ------
    RuntimeError
        - Error raised if any core shapes are inappropriate
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

    # Check that outer_tt_cores are 3-tensors with leading and trailing 1 dims
    for ii, G in enumerate(var_tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'var_tt_core is not a 3-tensor:\n'
                + 'var_tt_cores['+str(ii) + '].shape=' + str(G.shape)
            )


def t3_base_hole_shapes(
        base: T3Base,
) -> typ.Tuple[
    typ.Tuple[int,...], # variation_basis_shapes. len=d. elm_len=2
    typ.Tuple[int,...], # variation_tt_shapes. len=d. elm_len=3
]:
    '''T3Variation core shapes that fit with given T3Base.

    Shapes of the "holes" in the following tensor diagrams::

        1 -- GL0 -- ( ) -- GR2 -- GR3 -- 1
             |      |      |      |
             B0     B1     B2     B3
             |      |      |      |

        1 -- GL0 -- GL1 -- GO2 -- GR3 -- 1
             |      |      |     |
             B0     B1     ( )   B3
             |      |      |     |

    Here:
        - basis_cores       = (B0, B1, B2, B3)
        - left_tt_cores     = (GL0, GL1, GL2, GL3)
        - right_tt_cores    = (GR0, GR1, GR2, GR3)
        - outer_tt_cores    = (GO0, GO1, GO2, GO3)

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
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> from t3tools.t3_base_variation_format import *
    >>> basis_cores = (randn(10,14), randn(11,15), randn(12,16))
    >>> left_tt_cores = (randn(1,10,2), randn(2,11,3), randn(3,12,1))
    >>> right_tt_cores = (randn(1,10,4), randn(4,11,5), randn(5,12,1))
    >>> outer_tt_cores = (randn(1,9,4), randn(2,8,5), randn(3,7,1))
    >>> base = (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    >>> (var_basis_shapes, var_tt_shapes) = t3_base_hole_shapes(base)
    >>> print(var_basis_shapes)
        ((9, 14), (8, 15), (7, 16))
    >>> print(var_tt_shapes)
        ((1, 10, 4), (2, 11, 5), (3, 12, 1))
    '''
    t3_check_base(base)
    basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    num_cores = len(basis_cores)

    variation_basis_shapes = []
    for ii in range(num_cores):
        n = outer_tt_cores[ii].shape[1]
        N = basis_cores[ii].shape[1]
        variation_basis_shapes.append((n,N))

    variation_tt_shapes = []
    for ii in range(num_cores):
        rL = 1 if ii==0 else left_tt_cores[ii-1].shape[2]
        n = basis_cores[ii].shape[0]
        rR = 1 if ii==num_cores-1 else right_tt_cores[ii+1].shape[0]
        variation_tt_shapes.append((rL, n, rR))

    return tuple(variation_basis_shapes), tuple(variation_tt_shapes)


def t3_check_base_variation_fit(
        base: T3Base,
        variation: T3Variation,
) -> None:
    '''Check that the variation cores fit into the corresponding holes of the base.

    Parameters
    ----------
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

    See Also
    --------
    T3Base
    T3Variation
    t3_check_base
    t3_check_variation
    '''
    t3_check_base(base)
    t3_check_variation(variation)

    var_basis_cores, var_tt_cores = variation
    var_basis_shapes = tuple([B.shape for B in var_basis_cores])
    var_tt_shapes = tuple([G.shape for G in var_tt_cores])

    hole_basis_shapes, hole_tt_shapes = t3_base_hole_shapes(base)

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

