# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.core.tucker_tensor_train.ragged.operations as ragged_operations
import t3toolbox.core.tucker_tensor_train.uniform.operations as uniform_operations
import t3toolbox.core.tucker_tensor_train.orthogonalization as orth
import t3toolbox.core.tucker_tensor_train.ragged.orthogonalization as ragged_orth
import t3toolbox.core.tucker_tensor_train.uniform.orthogonalization as uniform_orth
from t3toolbox.common import *

__all__ = [
    'orthogonal_representations',
]

def orthogonal_representations(
        x: typ.Union[
            typ.Tuple[
                typ.Tuple[NDArray,...], # tucker_cores
                typ.Tuple[NDArray,...], # tt_cores
            ], # ragged
            typ.Tuple[
                NDArray, # tucker_supercore
                NDArray, # tt_supercore
            ], # uniform
        ],
        already_left_orthogonal: bool = False,
        squash: bool = True,
        use_jax: bool = False,
) -> typ.Union[
    typ.Tuple[
        typ.Tuple[
            typ.Tuple[NDArray,...], # up_tucker_cores
            typ.Tuple[NDArray,...], # left_tt_cores
            typ.Tuple[NDArray,...], # right_tucker_cores
            typ.Tuple[NDArray,...], # down_tucker_cores
        ],
        typ.Tuple[
            typ.Tuple[NDArray,...], # tucker_variations
            typ.Tuple[NDArray,...], # tt_variations
        ],
    ], # ragged
    typ.Tuple[
        typ.Tuple[
            NDArray,  # up_tucker_supercore
            NDArray,  # left_tt_supercore
            NDArray,  # right_tucker_supercore
            NDArray,  # down_tucker_supercore
        ],
        typ.Tuple[
            NDArray,  # tucker_variations_supercore
            NDArray,  # tt_variations_supercore
        ],
    ],  # uniform
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
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> base, variation = orth.orthogonal_representations(x) # Compute orthogonal representations
    >>> tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    >>> tucker_vars, tt_vars = variation
    >>> (U0,U1,U2) = tucker_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (O0,O1,O2) = outer_tt_cores
    >>> (V0,V1,V2) = tucker_vars
    >>> (H0,H1,H2) = tt_vars
    >>> x2 = ((U0,U1,U2), (L0,H1,R2)) # representation with TT-core variation in index 1
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Still represents origional tensor
    4.978421562425667e-12
    >>> x3 = ((U0,V1,U2), (L0,O1,R2)) # representation with tucker core variation in index 1
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
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.orthogonalization as orth
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> base, variation = orth.orthogonal_representations(x) # Compute orthogonal representations
    >>> tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base
    >>> tucker_vars, tt_vars = variation
    >>> (U0,U1,U2) = tucker_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (O0,O1,O2) = outer_tt_cores
    >>> (V0,V1,V2) = tucker_vars
    >>> (H0,H1,H2) = tt_vars
    >>> x2 = ((U0,U1,U2), (L0,H1,R2)) # representation with TT-core variation in index 1
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Still represents origional tensor
    2.5341562994067855e-12
    >>> x3 = ((V0,U1,U2), (O0,R1,R2)) # representation with tucker core variation in index 0
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
    is_uniform = is_ndarray(x[0])

    if is_uniform:
        squash_tails = lambda tk, tt: (tk, ragged_operations.squash_tt_tails(tt))
        up_orthogonalize_tucker_cores = uniform_orth.up_orthogonalize_uniform_tucker_cores
        down_orthogonalize_tt_cores = uniform_orth.down_orthogonalize_uniform_tt_cores
    else:
        squash_tails = lambda tk, tt: (tk, uniform_operations.squash_utt_tails(tt))
        up_orthogonalize_tucker_cores = ragged_orth.up_orthogonalize_tucker_cores
        down_orthogonalize_tt_cores = ragged_orth.down_orthogonalize_tt_cores

    if squash:
        x = squash_tails(x)

    if not already_left_orthogonal:
        # Orthogonalize Tucker cores upward to get up_tt_cores U
        up_tucker_cores, tt_cores = up_orthogonalize_tucker_cores(
            x, use_jax=use_jax,
        )

        # Sweep left-to-right, generating left orthogonal tt_cores L
        left_tt_cores = orth.left_orthogonalize_tt_cores(
            tt_cores, use_jax=use_jax,
        )
    else:
        up_tucker_cores, left_tt_cores = x

    # Sweep right-to-left, generating tt_variations H, and right orthogonal tt_cores R
    right_tt_cores, tt_variations = orth.right_orthogonalize_tt_cores(
        left_tt_cores, return_variation_cores=True, use_jax=use_jax,
    )

    # Orthogonalize TT cores downward to get outer_tt_cores O and tucker_variations V
    tucker_variations, outer_tt_cores = down_orthogonalize_tt_cores(
        (up_tucker_cores, tt_variations), use_jax=use_jax,
    )

    base = (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    variation = (tucker_variations, tt_variations)
    return base, variation