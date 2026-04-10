# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
import t3tools.orthogonalization as orth
import t3tools.uniform as ut3
import t3tools.common as common

xnp = np
scan = common.numpy_scan
NDArray = np.ndarray

__all__ = [
    'ut3_svd',
]

###################################################
##############    Uniform T3-SVD    ###############
###################################################

def ut3_svd(
        cores: ut3.UniformTuckerTensorTrainCores,
        masks: ut3.UniformTuckerTensorTrainMasks,
) -> typ.Tuple[
    ut3.UniformTuckerTensorTrainCores,
    NDArray, # basis_singular_values, shape=(d, n)
    NDArray, # tt_singular_values, shape=(d+1, r)
]:
    """Compute T3-SVD of uniform Tucker tensor train.

    Only guaranteed to give correct results if ranks are minimal.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> import t3tools.t3svd as t3svd
    >>> s0 = ((11,12,13), (6,7,5), (1,3,6,2))
    >>> s = (s0[0],) + t3.compute_minimal_ranks(s0)
    >>> x = t3.t3_corewise_randn(s)
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> ux2, ss_basis_from_ut3, ss_tt_from_ut3 = ut3.ut3_svd(cores, masks) # Uniform T3-SVD
    >>> print(np.linalg.norm(ut3.ut3_to_dense(ux2, masks) - t3.t3_to_dense(x)))
    3.782447238250888e-12
    >>> _, ss_basis, ss_tt = t3svd.t3_svd(x) # Non-uniform T3-SVD
    >>> print(ss_tt[1])
    [980.86624688 624.1067954  159.88424271]
    >>> print(ss_tt_from_ut3[1])
    [980.86624688 624.1067954  159.88424271   0.           0.        ]
    >>> _, tucker_masks, tt_masks = masks
    >>> print(ut3.unpack_edge_tensors(ss_tt_from_ut3, tt_masks)[1])
    [980.86624688 624.1067954  159.88424271]
    >>> ut3.unpack_edge_tensors(ss_basis_from_ut3, tucker_masks)[0] - ss_basis[0]
    array([ 1.13686838e-12, -2.27373675e-13, -1.13686838e-13])

    Non-example with degenerate (unnecessairily large) ranks

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> import t3tools.t3svd as t3svd
    >>> s = ((3,4,3), (4,6,7), (3,5,1,2))
    >>> x = t3.t3_corewise_randn(s)
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> ux2, ss_basis_from_ut3, ss_tt_from_ut3 = ut3.ut3_svd(cores, masks)
    >>> print(np.linalg.norm(ut3.ut3_to_dense(ux2, masks) - t3.t3_to_dense(x))) # OK
    9.404253555983741e-13
    >>> _, ss_basis, ss_tt = t3svd.t3_svd(x) # Non-uniform T3-SVD
    >>> print(ss_tt[1])
    [913.44494453 127.532224    16.08102313]
    >>> print(ss_tt_from_ut3[1]) # Incorrect singular values:
    [417.45514528 401.58448034  72.5343983   22.41273808   0.        ]
    """
    basis_supercore, tt_supercore = cores

    shape_mask, basis_masks, tt_masks = masks

    d, n, N = basis_supercore.shape
    r = tt_supercore.shape[1]

    basis_supercore, tt_supercore = orth.orthogonalize_ut3_basis_cores(basis_supercore, tt_supercore)
    tt_supercore = orth.right_orthogonalize_utt(tt_supercore)

    _, ss_tt00, _ = xnp.linalg.svd(tt_supercore[0].reshape((r, n*r)), full_matrices=False)
    ss_tt0 = xnp.concatenate([ss_tt00, xnp.zeros(r-len(ss_tt00))], axis=0)

    ss_tt0 = ss_tt0 * tt_masks[0]

    def _step(
            carry: NDArray,
            x,
    ):
        Y = carry # shape=(r, r)
        B, G, basis_mask, tt_mask = x

        G = xnp.einsum('ij,jak->iak', Y, G) # shape=(r, n, r)
        # Note: B.shape=(n, N)

        M = G.swapaxes(1,2).reshape((r*r, n))
        U, ss_basis, Vt = xnp.linalg.svd(M, full_matrices=False)
        n2 = len(ss_basis)
        U           = xnp.concatenate([U,           xnp.zeros((r*r, n-n2))],    axis=1)
        ss_basis    = xnp.concatenate([ss_basis,    xnp.zeros((n-n2, ))],       axis=0)
        Vt          = xnp.concatenate([Vt,          xnp.zeros((n-n2, n))],      axis=0)

        U           = U         * basis_mask.reshape((1,-1))
        ss_basis    = ss_basis  * basis_mask
        Vt          = Vt        * basis_mask.reshape((-1,1))

        new_B = xnp.einsum('ij,jk->ik', Vt, B)

        M = xnp.einsum('ij,j->ij', U, ss_basis).reshape((r, r, n)).swapaxes(1,2).reshape((r*n, r))
        U, ss_tt, Vt = xnp.linalg.svd(M, full_matrices=False)

        U       = U     * tt_mask.reshape((1,-1))
        ss_tt   = ss_tt * tt_mask
        Vt      = Vt    * tt_mask.reshape((-1,1))

        new_G = U.reshape((r, n, r))

        Y_next = xnp.einsum('i,ij->ij', ss_tt, Vt)  # shape=(r, r)

        return Y_next, (new_B, new_G, ss_basis, ss_tt)

    Y0 = xnp.eye(r)
    Yf, (new_basis_cores, new_tt_cores, basis_singular_values, tt_singular_values0) = scan(
        _step,
        Y0,
        (basis_supercore, tt_supercore, basis_masks, tt_masks[1:]),
    )

    # G_last = xnp.einsum('diaj,jk->diak', new_tt_cores[-1:], Yf)[:, :, :, :r]
    G_last = xnp.einsum('diaj,jk->diak', new_tt_cores[-1:], Yf)
    new_tt_cores = xnp.concatenate([new_tt_cores[:-1], G_last], axis=0)

    tt_singular_values = xnp.concatenate([ss_tt0.reshape((1, r)), tt_singular_values0], axis=0)
    return (new_basis_cores, new_tt_cores), basis_singular_values, tt_singular_values
