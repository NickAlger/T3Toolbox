# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ


import t3toolbox.tucker_tensor_train as t3
import t3toolbox.core.tucker_tensor_train.orthogonalization as orth
import t3toolbox.core.tucker_tensor_train.uniform.uniform_orthogonalization as uniform_orth
import t3toolbox.core.tucker_tensor_train.uniform.uniform_t3_operations as ut3_ops
# import t3toolbox.OLD_uniform as ut3
import t3toolbox.util_linalg as linalg
from t3toolbox.common import *

__all__ = [
    'uniform_t3_svd',
]


def uniform_t3_svd(
        cores: typ.Tuple[
            NDArray, # tucker_supercore
            NDArray, # tt_supercore
        ],
        masks: typ.Tuple[
            NDArray, # shape_mask
            NDArray, # tucker_edge_mask
            NDArray, # tt_edge_mask
        ], # Can be used to truncate rank. Do not have to be the original masks
        squash_tails_first: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[
        NDArray,  # tucker_supercore
        NDArray,  # tt_supercore
        NDArray,  # shape_mask
        NDArray,  # tucker_edge_mask
        NDArray,  # tt_edge_mask
    ], # new_x
    NDArray, # basis_singular_values, shape=(d, n)
    NDArray, # tt_singular_values, shape=(d+1, r)
]:
    """Compute T3-SVD of uniform Tucker tensor train.

    Only guaranteed to give correct results if ranks are minimal.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.t3svd as t3svd
    >>> s0 = ((11,12,13), (6,7,5), (1,3,6,2))
    >>> s = (s0[0],) + t3.compute_minimal_t3_ranks(s0)
    >>> x = t3.t3_corewise_randn(s)
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> ux2, ss_basis_from_ut3, ss_tt_from_ut3 = t3svd.uniform_t3_svd(cores, masks) # Uniform T3-SVD
    >>> print(np.linalg.norm(ut3.ut3_to_dense(ux2, masks) - t3.t3_to_dense(x)))
    3.782447238250888e-12
    >>> _, ss_basis, ss_tt = t3svd.t3_svd(x) # Non-uniform T3-SVD
    >>> print(ss_tt[1])
    [980.86624688 624.1067954  159.88424271]
    >>> print(ss_tt_from_ut3[1])
    [980.86624688 624.1067954  159.88424271   0.           0.        ]
    >>> _, tucker_masks, tt_masks = masks
    >>> print(ut3.unpack(ss_tt_from_ut3, tt_masks)[1])
    [980.86624688 624.1067954  159.88424271]
    >>> ut3.unpack(ss_basis_from_ut3, tucker_masks)[0] - ss_basis[0]
    array([ 1.13686838e-12, -2.27373675e-13, -1.13686838e-13])

    Uniform example with degenerate (unnecessairily large) ranks. Also using jax to test it

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.t3svd as t3svd
    >>> import jax
    >>> import t3toolbox.corewise as cw
    >>> jax.config.update("jax_enable_x64", True)
    >>> structure = ((3,4,3), (4,6,7), (3,5,1,2))
    >>> x = t3.t3_corewise_randn(structure)
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> inv_masks = cw.corewise_logical_not(masks)
    >>> junk = ut3.uniform_randn(ut3.get_uniform_structure(cores), masks=inv_masks)
    >>> cores = cw.corewise_add(cores, junk) # Add random junk outside the masks
    >>> ux2, ss_basis_from_ut3, ss_tt_from_ut3 = t3svd.uniform_t3_svd(cores, masks, squash_tails_first=False, use_jax=True)
    >>> print(np.linalg.norm(ut3.ut3_to_dense(ux2, masks) - t3.t3_to_dense(x))) # OK
    9.404253555983741e-13
    >>> _, ss_basis, ss_tt = t3svd.t3_svd(x) # Non-uniform T3-SVD
    >>> print(ss_tt[1])
    [913.44494453 127.532224    16.08102313]
    >>> print(ss_tt_from_ut3[1]) # Incorrect singular values:
    [417.45514528 401.58448034  72.5343983   22.41273808   0.        ]
    >>> ux4, ss_basis_from_ut4, ss_tt_from_ut4 = t3svd.uniform_t3_svd(cores, masks, use_jax=True)
    >>> print(ss_basis_from_ut4[1])

    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    #
    basis_supercore, tt_supercore = cores
    shape_mask, basis_masks, tt_masks = masks

    if squash_tails_first:
        tt_supercore = ut3_ops.uniform_squash_tt_tails(tt_supercore, use_jax=use_jax)


    d = basis_supercore.shape[0]
    stack_shape = basis_supercore.shape[1:-2]
    n, N = basis_supercore.shape[-2:]
    r = tt_supercore.shape[-1]

    basis_supercore, tt_supercore = uniform_orth.up_orthogonalize_uniform_tucker_cores(
        basis_supercore, tt_supercore, use_jax=use_jax,
    )
    tt_supercore = orth.right_orthogonalize_tt_cores(tt_supercore, use_jax=use_jax)

    # keep everything the same shape, for consistency with masks
    _, n2, _ = basis_supercore.shape
    basis_supercore = xnp.concatenate([
        basis_supercore, xnp.zeros(stack_shape+(d,)+stack_shape+(n-n2, N))
    ], axis=-2
    )
    tt_supercore    = xnp.concatenate([
        tt_supercore,    xnp.zeros(stack_shape+(d,)+stack_shape+(r, n-n2, r))
    ], axis=-2
    )

    _, ss_tt00, _ = xnp.linalg.svd(
        tt_supercore[0].reshape(stack_shape+(r, n*r)),
        full_matrices=False,
    )
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
    Yf, (new_basis_cores, new_tt_cores, basis_singular_values, tt_singular_values0) = xscan(
        _step,
        Y0,
        (basis_supercore, tt_supercore, basis_masks, tt_masks[1:]),
    )

    # G_last = xnp.einsum('diaj,jk->diak', new_tt_cores[-1:], Yf)[:, :, :, :r]
    G_last = xnp.einsum('diaj,jk->diak', new_tt_cores[-1:], Yf)
    new_tt_cores = xnp.concatenate([new_tt_cores[:-1], G_last], axis=0)

    tt_singular_values = xnp.concatenate([ss_tt0.reshape((1, r)), tt_singular_values0], axis=0)
    return (new_basis_cores, new_tt_cores), basis_singular_values, tt_singular_values
