# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ
import t3tools.tucker_tensor_train as t3
import t3tools.common as common

xnp = np
scan = common.numpy_scan
NDArray = np.ndarray

__all__ = [
    'left_orthogonalize_utt',
    'right_orthogonalize_utt',
    'orthogonalize_ut3_basis_cores',
    'construct_ut3_base_representations',
]

if False:

    ###################################################
    #########    Uniform orthogonalization    #########
    ###################################################

    def left_orthogonalize_utt(
            tt_supercore:       NDArray,  # shape=(d, r, n, r)
            use_jax: bool = False,
    ) -> NDArray: # new_tt_cores
        if use_jax:
            xnp = jnp
            scan = jax_scan
        else:
            xnp = np
            scan = numpy_scan

        d, r, n, _ = tt_supercore.shape

        def _orth_one_core(
                prev_R: NDArray, # shape=(r, r)
                G: NDArray, # shape=(r, n, r)
        ):
            G = xnp.einsum('ij,jak->iak', prev_R, G)
            # Q, R = xnp.linalg.qr(G.reshape((r*n, r)), mode='reduced')
            Q, ss, Vt = xnp.linalg.svd(G.reshape((r*n, r)), full_matrices=False)
            R = xnp.einsum('i,ij->ij', ss, Vt)

            G = Q.reshape((r, n, r))
            return R, G

        R0 = xnp.eye(r)
        R, first_new_tt_cores = scan(_orth_one_core, R0, tt_supercore[:-1])
        last_new_tt_core = xnp.einsum('ij,djak->diak', R, tt_supercore[-1:])
        new_tt_supercore = xnp.concatenate([first_new_tt_cores, last_new_tt_core], axis=0)
        return new_tt_supercore


    def right_orthogonalize_utt(
            tt_supercore: NDArray,  # shape=(d, r, n, r)
            use_jax: bool = False,
    ) -> NDArray: # new_tt_cores
        return left_orthogonalize_utt(
            tt_supercore[::-1].swapaxes(1,3), use_jax=use_jax,
        )[::-1].swapaxes(3,1)


    def orthogonalize_ut3_basis_cores(
            basis_supercore: NDArray,  # shape=(d, n, N)
            tt_supercore: NDArray,  # shape=(d, r, n, r)
            use_jax: bool = False,
    ) -> typ.Tuple[
        NDArray, # new_basis_cores
        NDArray, # new_tt_cores
    ]:
        xnp = jnp if use_jax else np

        d, n, N = basis_supercore.shape
        r = tt_supercore.shape[1]

        # QQ, RR = xnp.linalg.qr(basis_cores.swapaxes(1,2), mode='reduced')
        QQ, sss, VVt = xnp.linalg.svd(basis_supercore.swapaxes(1, 2), full_matrices=False) # use SVD because QR sometimes yields nans
        RR = xnp.einsum('da,dab->dab', sss, VVt)

        new_basis_supercore = QQ.swapaxes(2,1)
        new_tt_supercore = xnp.einsum('dab,dibj->diaj', RR, tt_supercore)

        n2 = QQ.shape[-1]
        new_basis_supercore = xnp.concatenate([new_basis_supercore, xnp.zeros((d, n-n2, N))], axis=1)
        new_tt_supercore = xnp.concatenate([new_tt_supercore, xnp.zeros((d, r, n-n2, r))], axis=2)

        return new_basis_supercore, new_tt_supercore



    def construct_ut3_base_representations(
            base_point: typ.Tuple[
                jnp.ndarray, # basis_cores, shape=(d, n, N)
                jnp.ndarray, # tt_cores, shape=(d, r, N, r)
            ],
            masks: typ.Tuple[
                jnp.ndarray,  # basis_cores_mask, shape=(d, n)
                jnp.ndarray,  # tt_cores_mask, shape=(d+1, r)
            ],
    ) -> typ.Tuple[
        jnp.ndarray, # orthogonal_basis_cores,      shape=(d, n, N)
        jnp.ndarray, # left_orthogonal_tt_cores,    shape=(d, r, n, r)
        jnp.ndarray, # right_orthogonal_tt_cores,   shape=(d, r, n, r)
        jnp.ndarray, # up_orthogonal_tt_cores,      shape=(d, r, n, r)
        jnp.ndarray, # nonorthogonal_basis_cores,   shape=(d, n, N)
        jnp.ndarray, # nonorthogonal_tt_cores,      shape=(d, r, n, r)
    ]:
        basis_cores, tt_cores = base_point
        d, n, N = basis_cores.shape
        r = tt_cores.shape[1]

        shape_mask, tucker_mask, tt_mask = masks
        basis_cores_mask = jnp.einsum('da,do->dao', tucker_mask, shape_mask)
        tt_cores_mask = jnp.einsum('di,da,dj->diaj', tt_mask[:-1], tucker_mask, tt_mask[1:])

        QQ, sss, VVt = jnp.linalg.svd(basis_cores.swapaxes(1, 2), full_matrices=False) # use SVD because QR sometimes yields nans
        RR = jnp.einsum('da,dab->dab', sss, VVt)

        orthogonal_basis_cores = QQ.swapaxes(2,1)

        tt_cores = jnp.einsum('dab,dibj->diaj', RR, tt_cores)

        n2 = QQ.shape[-1]
        orthogonal_basis_cores = jnp.concatenate([orthogonal_basis_cores, jnp.zeros((d, n-n2, N))], axis=1)
        tt_cores = jnp.concatenate([tt_cores, jnp.zeros((d, r, n-n2, r))], axis=2)

        orthogonal_basis_cores = orthogonal_basis_cores * basis_cores_mask

        right_orthogonal_tt_cores = right_orthogonalize_utt(tt_cores) * tt_cores_mask

        def _process_one_core(
                prev_R: jnp.ndarray,  # shape=(r, r)
                x: jnp.ndarray,
        ):
            G, B, G_mask, B_mask  = x

            G_tilde = jnp.einsum('ij,jak->iak', prev_R, G)
            G_tilde = G_tilde * G_mask

            Q, ss, Vt = jnp.linalg.svd(G_tilde.swapaxes(1, 2).reshape((r * r, n)), full_matrices=False)
            R = jnp.einsum('a,ab->ab', ss, Vt)
            B_tilde = jnp.einsum('ij,jo->io', R, B)
            B_tilde = B_tilde * B_mask

            n2 = Q.shape[1]
            Q = jnp.concatenate([Q, jnp.zeros((r*r, n-n2))], axis=1)
            G_up = Q.reshape(r,r,n).swapaxes(1,2)
            G_up = G_up * G_mask

            Q, ss, Vt = jnp.linalg.svd(G_tilde.reshape((r*n, r)), full_matrices=False)
            R = jnp.einsum('a,ab->ab', ss, Vt)

            G_left = Q.reshape((r, n, r))
            G_left = G_left * G_mask

            return R, (G_up, G_left, B_tilde, G_tilde)

        R0 = jnp.eye(r)
        R, (up_orthogonal_tt_cores, left_orthogonal_tt_cores0, nonorthogonal_basis_cores, nonorthogonal_tt_cores) = jax.lax.scan(
            _process_one_core, R0,
            (right_orthogonal_tt_cores, orthogonal_basis_cores, tt_cores_mask, basis_cores_mask),
        )

        first_left_orthogonal_tt_cores = left_orthogonal_tt_cores0[:-1]
        last_left_orthogonal_tt_core0 = left_orthogonal_tt_cores0[-1:] * tt_cores_mask[-1:]
        last_left_orthogonal_tt_core = jnp.einsum('diaj,jk->diak', last_left_orthogonal_tt_core0, R)
        left_orthogonal_tt_cores = jnp.concatenate(
            [first_left_orthogonal_tt_cores, last_left_orthogonal_tt_core],
            axis=0,
        )

        return (
            orthogonal_basis_cores,
            left_orthogonal_tt_cores,
            right_orthogonal_tt_cores,
            up_orthogonal_tt_cores,
            nonorthogonal_basis_cores,
            nonorthogonal_tt_cores,
        )




