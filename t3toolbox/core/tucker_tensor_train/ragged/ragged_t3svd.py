# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ


import t3toolbox.tucker_tensor_train as t3
import t3toolbox.orthogonalization as orth
import t3toolbox.uniform as ut3
import t3toolbox.util_linalg as linalg
from t3toolbox.common import *

__all__ = [
    't3_svd',
]

def t3_svd(
        x: t3.TuckerTensorTrain,
        min_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        min_tucker_ranks:   typ.Sequence[int] = None,  # len=d
        max_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        max_tucker_ranks:   typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
        squash_tails_first: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    t3.TuckerTensorTrain, # new_x
    typ.Tuple[NDArray,...], # Tucker singular values, len=d
    typ.Tuple[NDArray,...], # TT singular values, len=d+1
]:
    '''Compute (truncated) T3-SVD of TuckerTensorTrain.

    Parameters
    ----------
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_tucker_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation.
    min_tt_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation.
    max_tucker_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation.
    max_tt_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    NDArray
        New TuckerTensorTrain representing the same tensor (or a truncated version), but with modified cores
    typ.Tuple[NDArray,...]
        Singular values associated with edges between Tucker cores and TT-cores
    typ.Tuple[NDArray,...]
        Singular values associated with edges between adjacent TT-cores

    See Also
    --------
    left_svd_3tensor
    right_svd_3tensor
    outer_svd_3tensor
    up_svd_ith_tucker_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    truncated_svd

    Examples
    --------

    T3-SVD with no truncation:
    (ranks may decrease to minimal values, but no approximation error)

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.t3svd as t3svd
    >>> x = t3.t3_corewise_randn((5,6,3), (4,4,3), (1,3,2,1))
    >>> x2, ss_tucker, ss_tt = t3svd.t3_svd(x) # Compute T3-SVD
    >>> x_dense = x.to_dense()
    >>> x2_dense = x2.to_dense()
    >>> print(np.linalg.norm(x_dense - x2_dense)) # Tensor unchanged
    7.556835759880194e-13
    >>> ss_tt1 = np.linalg.svd(x_dense.reshape((5, 6*3)))[1] # Singular values of unfolding 1
    >>> print(ss_tt1); print(ss_tt[1])
    [1.75326490e+02 3.41363029e+01 9.31164204e+00 1.33610061e-14 4.11601708e-15]
    [175.32648969  34.13630287   9.31164204]
    >>> ss_tucker2 = np.linalg.svd(x_dense.transpose([2,0,1]).reshape((3,5*6)))[1] # Singular values of matricization 2
    >>> print(ss_tucker2); print(ss_tucker[2])
    [1.71350937e+02 5.12857505e+01 1.36927051e-14]
    [171.35093708  51.28575045]

    T3-SVD with truncation based on relative tolerance:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.t3svd as t3svd
    >>> B0 = np.random.randn(35,40) @ np.diag(1.0 / np.arange(1, 41)**2) # preconditioned indices
    >>> B1 = np.random.randn(45,50) @ np.diag(1.0 / np.arange(1, 51)**2)
    >>> B2 = np.random.randn(55,60) @ np.diag(1.0 / np.arange(1, 61)**2)
    >>> G0 = np.random.randn(1,35,30)
    >>> G1 = np.random.randn(30,45,40)
    >>> G2 = np.random.randn(40,55,1)
    >>> tucker_cores_x = (B0, B1, B2)
    >>> tt_cores_x = (G0, G1, G2)
    >>> x = t3.TuckerTensorTrain(tucker_cores_x, tt_cores_x) # Tensor has spectral decay due to preconditioning
    >>> x2, ss_tucker, ss_tt = t3svd.t3_svd(x, rtol=1e-2) # Truncate singular values to reduce rank
    >>> print(x.structure)
    ((40, 50, 60), (35, 45, 55), (1, 30, 40, 1))
    >>> print(x2.structure)
    ((40, 50, 60), (6, 6, 5), (1, 6, 5, 1))
    >>> x_dense = x.to_dense()
    >>> x2_dense = x2.to_dense()
    >>> print(np.linalg.norm(x_dense - x2_dense)/np.linalg.norm(x_dense)) # Should be near rtol=1e-2
    0.013078458673911168

    T3-SVD with truncation based on absolute tolerance:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.t3svd as t3svd
    >>> x = t3.t3_corewise_randn((14,15,16), (10,11,12), (1,8,9,1))
    >>> x2, ss_tucker, ss_tt = t3svd.t3_svd(x, max_tucker_ranks=(3,3,3), max_tt_ranks=(1,2,2,1)) # Truncate based on ranks
    >>> print(x.structure)
        ((14, 15, 16), (10, 11, 12), (1, 8, 9, 1))
    >>> print(x2.structure)
        ((14, 15, 16), (3, 3, 2), (1, 2, 2, 1))

    Example where first and last ranks are not ones:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.t3svd as t3svd
    >>> x = t3.t3_corewise_randn((5,6,3), (4,4,3), (2,3,2,2))
    >>> x2, ss_tucker, ss_tt = t3svd.t3_svd(x, squash_tails_first=False) # Compute T3-SVD
    >>> x_dense = x.to_dense(squash_tails=False)
    >>> x2_dense = x2.to_dense(squash_tails=False)
    >>> print(np.linalg.norm(x_dense - x2_dense)) # Tensor unchanged
    5.486408687260824e-13
    >>> ss_tt0 = np.linalg.svd(x_dense.reshape((2,5*6*3*2)))[1] # Singular values of leading unfolding
    >>> print(ss_tt0); print(ss_tt[0])
    [303.0474449   88.85034392]
    [303.0474449   88.85034392]
    >>> ss_tt3 = np.linalg.svd(x_dense.reshape((2*5*6*3,2)))[1] # Singular values of trailing unfolding
    >>> print(ss_tt3); print(ss_tt[3])
    [299.45433768 100.29574828]
    [299.45433768 100.29574828]
    '''
    is_uniform = False
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if x.stack_shape != ():
        raise NotImplementedError(
            'T3-SVD is not implemented for TuckerTensorTrains with vectorized cores. Must do one at a time.'
        )

    # make leading and trailing TT-ranks equal to 1
    if squash_tails_first:
        x = x.squash_tails(use_jax=use_jax) #t3.squash_tails(x, use_jax=use_jax) # This causes a problem for some reason

    x = x.data
    tucker_cores, tt_cores = x

    num_cores = len(tt_cores)

    # Orthogonalize Tucker matrices
    x = orth.up_orthogonalize_tucker_cores(x, use_jax=use_jax)

    # Right orthogonalize
    x = x[0], orth.right_orthogonalize_tt_cores(x[1], use_jax=use_jax)

    G0 = x[1][0]
    _, ss_first, _ = linalg.right_svd_3tensor(G0, use_jax=use_jax)

    # Sweep left to right computing SVDS
    all_ss_tucker = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        min_rank = min_tucker_ranks[ii] if min_tucker_ranks is not None else None
        max_rank = max_tucker_ranks[ii] if max_tucker_ranks is not None else None
        # SVD inbetween TT core and Tucker core
        x, ss_tucker = orth.up_svd_ith_tt_core(
            ii, x, min_rank, max_rank, rtol, atol, use_jax=use_jax,
        )
        all_ss_tucker.append(ss_tucker)

        if ii < num_cores-1:
            min_rank = min_tt_ranks[ii+1] if min_tt_ranks is not None else None
            max_rank = max_tt_ranks[ii+1] if max_tt_ranks is not None else None
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = orth.left_svd_ith_tt_core(
                ii, x, min_rank, max_rank, rtol, atol, use_jax=use_jax,
            )
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = linalg.left_svd_3tensor(Gf, use_jax=use_jax)
        all_ss_tt.append(ss_tt)

    return t3.TuckerTensorTrain(*x), tuple(all_ss_tucker), tuple(all_ss_tt)

