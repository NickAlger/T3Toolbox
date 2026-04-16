# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.util_linalg
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.base_variation_format as bvf
import t3toolbox.util_linalg as linalg
import t3toolbox.uniform as ut3
import t3toolbox.common as common
from t3toolbox.common import *

__all__ = [
    'up_orthogonalize_tucker_cores',
    'outer_orthogonalize_tt_cores',
    'left_orthogonalize_tt_cores',
    'right_orthogonalize_tt_cores',
    'orthogonal_representations', # Duplicate, need to turn into uniform only version
]




####


def up_orthogonalize_tucker_cores(
        x: typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],
        use_jax: bool = False,
) -> t3.TuckerTensorTrain:
    """Orthogonalize Tucker cores upwards, pushing remainders onto TT cores above.
    """
    is_uniform = not isinstance(x[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #

    if is_uniform:
        B_d_i_o, G_d_a_i_b = x
        B_d_o_i = B_d_i_o.swapaxes(1,2)

        U_d_o_x, ss_d_x, WT_d_x_i = xnp.linalg.svd(B_d_o_i, full_matrices=False)
        R_d_x_i = xnp.einsum('dx,dxi->dxi', ss_d_x, WT_d_x_i)

        # Make sure shape remains the same. Commented out because it' OK for shape to shange now
        # d, N, n = B_d_o_i.shape
        # n2 = ss_d_x.shape[1]
        # U_d_o_x = xnp.concatenate([U_d_o_x, xnp.zeros((d, N, n-n2))], axis=1)
        # R_d_x_i = xnp.concatenate([R_d_x_i, xnp.zeros((d, n-n2, n))], axis=0)

        new_G_d_a_x_b = xnp.einsum('daib,dxi->daxb', G_d_a_i_b, R_d_x_i)
        new_U_d_x_o = U_d_o_x.swapaxes(1,2)
        up_tucker_cores, new_tt_cores = new_U_d_x_o, new_G_d_a_x_b

    else:
        def _up_func(up_func_args):
            Bio, Gaib = up_func_args
            Boi = Bio.T

            Uox, ssx, WTxi = xnp.linalg.svd(Boi, full_matrices=False)
            Rxi = xnp.einsum('x,xi->xi', ssx, WTxi)

            new_Gaxb = xnp.einsum('aib,xi->axb', Gaib, Rxi)
            new_Uxo = Uox.T
            return (new_Uxo, new_Gaxb)

        xs = x
        up_tucker_cores, new_tt_cores = xmap(_up_func, xs)

    return (up_tucker_cores, new_tt_cores)


def outer_orthogonalize_tt_cores(
        x: typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],
        use_jax: bool = False,
):
    """Outer orthogonalize TT cores, pushing remainders downward onto tucker cores below.
    """
    is_uniform = not isinstance(x[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #

    if is_uniform:
        U_d_i_o, H_d_a_i_b = x

        d, rL, n, rR = H_d_a_i_b.shape
        H_d_ab_i = H_d_a_i_b.swapaxes(1,2).reshape((d,rL*rR,n))

        O_d_ab_x, ss_d_x, WT_d_x_i = xnp.linalg.svd(H_d_ab_i, full_matrices=False)
        n2 = ss_d_x.shape[1]
        O_d_a_x_b = O_d_ab_x.reshape((d, rL, rR, n2)).swapaxes(1,2)

        C_d_x_i = ss_d_x.reshape((1, -1, 1)) * WT_d_x_i

        V_d_x_o = np.einsum('dxi,dio->dxo', C_d_x_i, U_d_i_o)
        tucker_variations, outer_tt_cores = V_d_x_o, O_d_a_x_b

    else:
        def _down_func(Uio_Haib):
            Uio, Haib,  = Uio_Haib

            rL, n, rR = Haib.shape
            H_ab_i = Haib.swapaxes(1,2).reshape((rL*rR,n))

            O_ab_x, ssx, WTxi = xnp.linalg.svd(H_ab_i, full_matrices=False)
            n2 = len(ssx)
            Oaxb = O_ab_x.reshape((rL, rR, n2)).swapaxes(1,2)

            Cxi = ssx.reshape((-1, 1)) * WTxi

            Vxo = np.einsum('xi,io->xo', Cxi, Uio)
            return (Vxo, Oaxb)

        tucker_variations, outer_tt_cores = xmap(_down_func, x)

    return tucker_variations, outer_tt_cores


def left_orthogonalize_tt_cores(
        tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(ri,ni,r(i+1))
        return_variation_cores: bool = False,
        use_jax: bool = False,
):
    """Left-orthogonalize a Tensor train (no Tucker).
    """
    is_uniform = not isinstance(tt_cores, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #

    def _left_func(Cxb, left_func_args):
        Gbjc = left_func_args[0]

        Hxjc = xnp.einsum('xb,bjc->xjc', Cxb, Gbjc)

        rL, n, rR = Hxjc.shape
        H_xj_c = Hxjc.reshape((rL*n, rR))
        L_xj_y, ssy, VTyc = xnp.linalg.svd(H_xj_c, full_matrices=False)
        rR2 = len(ssy)
        Lxjy = L_xj_y.reshape((rL,n,rR2))

        Cyc = ssy.reshape((-1, 1)) * VTyc

        return Cyc, (Lxjy, Hxjc)


    init = xnp.eye(tt_cores[0].shape[0])
    xs = (tt_cores[:-1],)

    Cf, (LL, HH) = xscan(_left_func, init, xs)

    # Dealing with the last core as a special case
    Lf = xnp.einsum('xb,bjc->xjc', Cf, tt_cores[-1])
    if is_uniform:
        left_tt_cores = xnp.concatenate([LL, Lf.reshape((1,) + Lf.shape)], axis=0)
        var_tt_cores = xnp.concatenate([HH, Lf.reshape((1,) + Lf.shape)], axis=0)
    else:
        left_tt_cores = tuple(LL) + (Lf,)
        var_tt_cores = tuple(HH) + (Lf,)

    if return_variation_cores:
        return left_tt_cores, var_tt_cores
    else:
        return left_tt_cores


def right_orthogonalize_tt_cores(
        tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(ri,ni,r(i+1))
        return_variation_cores: bool = False,
        use_jax: bool = False,
):
    result = left_orthogonalize_tt_cores(
        t3.reverse_tt(tt_cores), return_variation_cores=return_variation_cores, use_jax=use_jax,
    )
    if return_variation_cores:
        return t3.reverse_tt(result[0]), t3.reverse_tt(result[1])
    else:
        return t3.reverse_tt(result)


def orthogonal_representations(
        x: typ.Union[t3.TuckerTensorTrain, ut3.UniformTuckerTensorTrain],
        already_left_orthogonal: bool = False,
        squash_tails: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    bvf.T3Base, # orthogonal base
    bvf.T3Variation, # variations
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
    is_uniform = not isinstance(x[0], typ.Sequence)

    if squash_tails:
        if is_uniform:
            x = ut3.uniform_squash_tails(x)
        else:
            x = t3.squash_tails(x)

    if not already_left_orthogonal:
        # Orthogonalize Tucker cores upward to get up_tt_cores U
        up_tucker_cores, tt_cores = up_orthogonalize_tucker_cores(
            x, use_jax=use_jax,
        )

        # Sweep left-to-right, generating left orthogonal tt_cores L
        left_tt_cores = left_orthogonalize_tt_cores(
            tt_cores, use_jax=use_jax,
        )
    else:
        up_tucker_cores, left_tt_cores = x

    # Sweep right-to-left, generating tt_variations H, and right orthogonal tt_cores R
    right_tt_cores, tt_variations = right_orthogonalize_tt_cores(
        left_tt_cores, return_variation_cores=True, use_jax=use_jax,
    )

    # Orthogonalize TT cores downward to get outer_tt_cores O and tucker_variations V
    tucker_variations, outer_tt_cores = outer_orthogonalize_tt_cores(
        (up_tucker_cores, tt_variations), use_jax=use_jax,
    )

    base = (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    variation = (tucker_variations, tt_variations)
    return base, variation

