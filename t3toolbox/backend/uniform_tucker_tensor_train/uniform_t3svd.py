# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.tucker_tensor_train.orthogonalization as orth
import t3toolbox.backend.uniform_tucker_tensor_train.uniform_orthogonalization as uniform_orth
import t3toolbox.backend.uniform_tucker_tensor_train.uniform_t3_operations as ut3_ops
from t3toolbox.backend.common import *

__all__ = [
    'uniform_t3_svd',
]


def uniform_t3_svd(
        cores: typ.Tuple[
            NDArray, # tucker_supercore
            NDArray, # tt_supercore
        ],
        rank_truncation_masks: typ.Tuple[
            NDArray, # shape_mask
            NDArray, # tucker_edge_mask
            NDArray, # tt_edge_mask
        ] = (None, None, None), # Can be used to truncate rank. Do not have to be the original masks
        squash_tails_first: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[
        NDArray,  # tucker_supercore
        NDArray,  # tt_supercore
    ], # new_x
    NDArray, # basis_singular_values, shape=(d, n)
    NDArray, # tt_singular_values, shape=(d+1, r)
]:
    """Compute T3-SVD of uniform Tucker tensor train.

    Only guaranteed to give correct results if ranks are minimal.
    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    #
    basis_supercore, tt_supercore = cores
    shape_mask, basis_masks, tt_masks = rank_truncation_masks

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
    n2 = basis_supercore.shape[-2]
    basis_supercore = xnp.concatenate([
        basis_supercore, xnp.zeros((d,)+stack_shape+(n-n2, N))
    ], axis=-2
    )
    tt_supercore    = xnp.concatenate([
        tt_supercore,    xnp.zeros((d,)+stack_shape+(r, n-n2, r))
    ], axis=-2
    )

    _, ss_tt00, _ = xnp.linalg.svd(
        tt_supercore[0].reshape(stack_shape+(r, n*r)),
        full_matrices=False,
    )
    ss_tt0 = xnp.concatenate([ss_tt00, xnp.zeros(stack_shape+(r-ss_tt00.shape[-1],))], axis=-1) # FIX FOR STACKING

    ss_tt0 = ss_tt0 * tt_masks[0]

    def _step(
            carry: NDArray,
            x,
    ):
        Y = carry # shape=(r, r)
        B, G, basis_mask, tt_mask = x

        # print('1. B.shape=', B.shape)
        # print('2. G.shape=', G.shape)

        G = xnp.einsum('...ij,...jak->...iak', Y, G) # shape=(r, n, r)
        # Note: B.shape=(n, N)

        # print('3. G.shape=', G.shape)

        M = G.swapaxes(-2,-1).reshape(stack_shape+(r*r, n))

        # print('4. M.shape=', M.shape)

        U, ss_basis, Vt = xnp.linalg.svd(M, full_matrices=False)

        # print('5. U.shape=', U.shape)
        # print('6. ss_basis.shape=', ss_basis.shape)
        # print('7. Vt.shape=', Vt.shape)

        n2 = ss_basis.shape[-1]

        # print('8. n=', n, ', n2=', n2)

        U           = xnp.concatenate([U,           xnp.zeros(stack_shape+(r*r, n-n2))],    axis=-1)
        ss_basis    = xnp.concatenate([ss_basis,    xnp.zeros(stack_shape+(n-n2, ))],       axis=-1)
        Vt          = xnp.concatenate([Vt,          xnp.zeros(stack_shape+(n-n2, n))],      axis=-2)

        # print('9. U.shape=', U.shape)
        # print('10. ss_basis.shape=', ss_basis.shape)
        # print('11. Vt.shape=', Vt.shape)

        U           = U         * basis_mask.reshape(stack_shape+(1,-1))
        ss_basis    = ss_basis  * basis_mask
        Vt          = Vt        * basis_mask.reshape(stack_shape+(-1,1))

        # print('12. U.shape=', U.shape)
        # print('13. ss_basis.shape=', ss_basis.shape)
        # print('14. Vt.shape=', Vt.shape)

        new_B = xnp.einsum('...ij,...jk->...ik', Vt, B)

        # print('15. new_B.shape', new_B.shape)

        M = xnp.einsum(
            '...ij,...j->...ij',
            U, ss_basis
        ).reshape(stack_shape+(r, r, n)).swapaxes(-1,-2).reshape(stack_shape+(r*n, r))

        # print('16. M.shape=', M.shape)

        U, ss_tt, Vt = xnp.linalg.svd(M, full_matrices=False)

        # print('17. U.shape=', U.shape)
        # print('18. ss_basis.shape=', ss_basis.shape)
        # print('19. Vt.shape=', Vt.shape)

        U       = U     * tt_mask.reshape(stack_shape+(1,-1))
        ss_tt   = ss_tt * tt_mask
        Vt      = Vt    * tt_mask.reshape(stack_shape+(-1,1))

        # print('20. U.shape=', U.shape)
        # print('21. ss_basis.shape=', ss_basis.shape)
        # print('22. Vt.shape=', Vt.shape)

        new_G = U.reshape(stack_shape+(r, n, r))

        # print('new_G.shape=', new_G.shape)

        Y_next = xnp.einsum('...i,...ij->...ij', ss_tt, Vt)  # shape=(r, r)

        # print('Y_next.shape=', Y_next.shape)

        return Y_next, (new_B, new_G, ss_basis, ss_tt)

    Y0 = xnp.eye(r)
    if stack_shape:
        Y0 = xnp.tensordot(xnp.ones(len(stack_shape)), Y0, axes=[(), ()])

    # print('Y0.shape=', Y0.shape)
    # print('basis_supercore.shape=', basis_supercore.shape)
    # print('tt_supercore.shape=', tt_supercore.shape)
    # print('basis_masks.shape=', basis_masks.shape)
    # print('tt_masks[1:].shape=', tt_masks[1:].shape)

    Yf, (new_basis_cores, new_tt_cores, basis_singular_values, tt_singular_values0) = xscan(
        _step,
        Y0,
        (basis_supercore, tt_supercore, basis_masks, tt_masks[1:]),
    )

    # G_last = xnp.einsum('diaj,jk->diak', new_tt_cores[-1:], Yf)[:, :, :, :r]
    G_last = xnp.einsum('d...iaj,...jk->d...iak', new_tt_cores[-1:], Yf)
    new_tt_cores = xnp.concatenate([
        new_tt_cores[:-1], G_last],
        axis=0,
    )

    tt_singular_values = xnp.concatenate([ss_tt0.reshape((1,)+stack_shape+(r,)), tt_singular_values0], axis=0)
    return (new_basis_cores, new_tt_cores), basis_singular_values, tt_singular_values
