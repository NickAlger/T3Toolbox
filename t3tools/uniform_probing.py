

# # # # WORK IN PROGRESS DO NOT USE

import jax
import jax.numpy as jnp
import typing as typ
import functools as ft


__all__ = [
    'ut3_actions',
    'ut3_tangent_actions',
    'ut3_tangent_actions_transpose',
    #
    'ut3_mus',
    'ut3_nus',
    'ut3_assemble_actions',
    #
    'ut3_sigmas',
    'ut3_taus',
    'ut3_assemble_tangent_actions',
    #
    'ut3_tau_tildes',
    'ut3_sigma_tildes',
    'ut3_assemble_tangent_tt_core_perturbations',
    'ut3_assemble_tangent_basis_core_perturbations',
    #
    'padded_reshape',
    'compute_result_in_batches',
    'compute_summed_result_in_batches',
]


# Helpers for batching
def padded_reshape(
        A: jnp.ndarray, # shape=(N, ...)
        batch_size: int,
) -> jnp.ndarray: # shape=(num_batches, batch_size, ...)
    N = A.shape[0]
    num_batches = (N // batch_size) + 1
    N_big = num_batches * batch_size

    Z = jnp.zeros((N_big - N,) + A.shape[1:])
    A_big = jnp.concatenate([A, Z], axis=0)
    A_big_reshaped = A_big.reshape((num_batches, batch_size) + A.shape[1:])

    return A_big_reshaped


def compute_result_in_batches(
        func: typ.Callable,
        XX: typ.Sequence[jnp.ndarray],
        batch_size: int,
):
    N = XX[0].shape[0]
    for X in XX:
        assert(X.shape[0] == N)

    if batch_size >= N:
        return func(XX)

    batched_args = tuple([padded_reshape(X, batch_size) for X in XX])
    batched_result = jax.lax.map(func, batched_args)
    num_batches = batched_result.shape[0]
    N_big = num_batches * batch_size
    result_big = batched_result.reshape((N_big,) + batched_result.shape[2:])
    result = result_big[:N]
    return result


def compute_summed_result_in_batches(
        func: typ.Callable,
        XX: typ.Sequence[jnp.ndarray],
        batch_size: int,
):
    N = XX[0].shape[0]
    for X in XX:
        assert(X.shape[0] == N)

    if batch_size >= N:
        return func(XX)

    batched_args = tuple([padded_reshape(X, batch_size) for X in XX])

    last_batch_size = N - (N // batch_size) * batch_size

    batched_args_first  = [X[0] for X in batched_args]
    batched_args_mid    = [X[1:-1] for X in batched_args]
    batched_args_last   = [X[-1][:last_batch_size] for X in batched_args]

    result = func(batched_args_first)
    result, _ = jax.lax.scan(
        lambda carry, x: (tuple([c + fx for c, fx in zip(carry, func(x))]), 0),
        result,
        batched_args_mid,
    )
    result = tuple([c + fx for c, fx in zip(result, func(batched_args_last))])
    return result


########    Tucker tensor train actions    ########

@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_mus(
        tt_cores:   jnp.ndarray, # shape=(d, r, n, r)
        reduced_xx: jnp.ndarray, # shape=(num_samples, d, n)
        batch_size: int = None,
) -> jnp.ndarray: # uniform_left_pushthroughs, shape=(num_samples, d-1, r)
    if batch_size is not None:
        return compute_result_in_batches(
            lambda XX: ut3_mus(tt_cores, XX[0], batch_size=None),
            (reduced_xx,),
            batch_size,
        )

    def _push_left(prev_mu, ith_data):
        G, v = ith_data
        mu = jnp.einsum('zi,za,iaj->zj', prev_mu, v, G)
        return mu, prev_mu

    all_data = (tt_cores, reduced_xx.swapaxes(0,1))

    r = tt_cores.shape[-1]
    num_samples, d, _ = reduced_xx.shape
    init_mu = jnp.outer(jnp.ones(num_samples), jnp.concatenate([jnp.array([1]), jnp.zeros(r-1)]))

    _, mus_swapped = jax.lax.scan(_push_left, init_mu, all_data)
    mus = mus_swapped.swapaxes(0,1)
    return mus


@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_nus(
        tt_cores:   jnp.ndarray,  # shape=(d, r, n, r)
        reduced_xx: jnp.ndarray,  # shape=(num_samples, d, n)
        batch_size: int = None,
) -> jnp.ndarray:  # uniform_left_pushthroughs, shape=(num_samples, d-1, r)
    return ut3_mus(
        tt_cores[::-1, :, :, :].swapaxes(1, 3), # reverse tensor train
        reduced_xx[:, ::-1, :],
        batch_size=batch_size,
    )[:, ::-1, :]


@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_assemble_actions(
        mus:            jnp.ndarray, # shape=(num_samples, d, r)
        nus:            jnp.ndarray, # shape=(num_samples, d, r)
        basis_cores:    jnp.ndarray, # shape=(d, n, N)
        tt_cores:       jnp.ndarray, # shape=(d, r, n, r)
        batch_size: int = None,
):
    if batch_size is not None:
        return compute_result_in_batches(
            lambda XX: ut3_assemble_actions(XX[0], XX[1], basis_cores, tt_cores, batch_size=None),
            (mus, nus),
            batch_size,
        )

    reduced_actions = jnp.einsum('ski,kiaj,skj->ska', mus, tt_cores, nus)
    actions = jnp.einsum('kao,ska->sko', basis_cores, reduced_actions)
    return actions


def ut3_actions(
        basis_cores:    jnp.ndarray,  # shape=(d, n, N)
        tt_cores:       jnp.ndarray,  # shape=(d, r, n, r)
        vectors:        jnp.ndarray,  # shape=(num_samples, d, N)
        batch_size: int = None
) -> jnp.ndarray:  # actions, shape=(num_samples, d, N)
    if batch_size is not None:
        return compute_result_in_batches(
            lambda XX: ut3_actions(basis_cores, tt_cores, XX[0], batch_size=None),
            (vectors,),
            batch_size,
        )

    reduced_vectors = jnp.einsum('kia,ska->ski', basis_cores, vectors)

    mus = ut3_mus(tt_cores, reduced_vectors)
    nus = ut3_nus(tt_cores, reduced_vectors)

    actions = ut3_assemble_actions(mus, nus, basis_cores, tt_cores)
    return actions


########    Uniform Tucker tensor train tangent actions    #######

@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_sigmas(
        tt_variations:              jnp.ndarray,  # shape=(d, r, n, r)
        right_orthogonal_tt_cores:  jnp.ndarray,  # shape=(d, r, n, r)
        up_orthogonal_tt_cores:     jnp.ndarray,  # shape=(d, r, n, r)
        mus:                        jnp.ndarray,  # shape=(num_samples, d, r)
        reduced_xx:                 jnp.ndarray,  # shape=(num_samples, d, n)
        reduced_dxx:                jnp.ndarray,  # shape=(num_samples, d, n)
        batch_size: int = None,
) -> jnp.ndarray:  # sigmas, shape=(num_samples, d, r)
    if batch_size is not None:
        return compute_result_in_batches(
            lambda XX: ut3_sigmas(
                tt_variations, right_orthogonal_tt_cores, up_orthogonal_tt_cores, XX[0], XX[1], XX[2], batch_size=None,
            ),
            (mus, reduced_xx, reduced_dxx),
            batch_size,
        )

    d, r, _, _ = tt_variations.shape
    num_samples = mus.shape[0]

    def _push_left(prev_sigma, ith_data):
        Q, R, dU, x, dx, mu = ith_data
        t1 = jnp.einsum('zi,za,iaj->zj', prev_sigma, x, Q)
        t2 = jnp.einsum('zi,za,iaj->zj', mu, x, dU)
        t3 = jnp.einsum('zi,za,iaj->zj', mu, dx, R)
        sigma = t1 + t2 + t3
        return sigma, prev_sigma

    all_data = (
        right_orthogonal_tt_cores,
        up_orthogonal_tt_cores,
        tt_variations,
        reduced_xx.swapaxes(0,1),
        reduced_dxx.swapaxes(0,1),
        mus.swapaxes(0,1),
    )

    init_sigma = jnp.zeros((num_samples, r))

    _, sigmas_swapped = jax.lax.scan(_push_left, init_sigma, all_data)
    sigmas = sigmas_swapped.swapaxes(0,1)
    return sigmas


@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_taus(
        tt_variations:              jnp.ndarray,  # shape=(d, r, n, r)
        left_orthogonal_tt_cores:   jnp.ndarray,  # shape=(d, r, n, r)
        up_orthogonal_tt_cores:     jnp.ndarray,  # shape=(d, r, n, r)
        nus:                        jnp.ndarray,  # shape=(num_samples, d, r)
        reduced_xx:                 jnp.ndarray,  # shape=(num_samples, d, n)
        reduced_dxx:                jnp.ndarray,  # shape=(num_samples, d, n)
        batch_size: int = None,
) -> jnp.ndarray:  # uniform_right_pushthrough_variants, shape=(num_samples, d, r)
    return ut3_sigmas(
        tt_variations           [::-1, :, :, :].swapaxes(1, 3), # reverse tensor train
        left_orthogonal_tt_cores[::-1, :, :, :].swapaxes(1, 3),
        up_orthogonal_tt_cores  [::-1, :, :, :].swapaxes(1, 3),
        nus         [:, ::-1, :],
        reduced_xx  [:, ::-1, :],
        reduced_dxx [:, ::-1, :],
        batch_size=batch_size,
    )[:, ::-1, :]


@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_assemble_tangent_actions(
        basis_variations:   jnp.ndarray,  # shape=(d, n, N)
        tt_variations:      jnp.ndarray, # shape=(d, r, n, r)
        mus:    jnp.ndarray, # shape=(num_samples, d, r)
        nus:    jnp.ndarray, # shape=(num_samples, d, r)
        sigmas: jnp.ndarray, # shape=(num_samples, d, r)
        taus:   jnp.ndarray, # shape=(num_samples, d, r)
        orthogonal_basis_cores:     jnp.ndarray, # shape=(d, n, N)
        left_orthogonal_tt_cores:   jnp.ndarray, # shape=(d, r, n, r)
        right_orthogonal_tt_cores:  jnp.ndarray, # shape=(d, r, n, r)
        up_orthogonal_tt_cores:     jnp.ndarray, # shape=(d, r, n, r)
        batch_size: int = None,
) -> jnp.ndarray:  # actions, shape=(num_samples, d, N)
    if batch_size is not None:
        return compute_result_in_batches(
            lambda XX: ut3_assemble_tangent_actions(
                basis_variations, tt_variations,
                XX[0], XX[1], XX[2], XX[3],
                orthogonal_basis_cores,
                left_orthogonal_tt_cores,
                right_orthogonal_tt_cores,
                up_orthogonal_tt_cores,
                batch_size=None,
            ),
            (mus, nus, sigmas, taus),
            batch_size,
        )

    B   = orthogonal_basis_cores
    dB  = basis_variations
    dU  = tt_variations
    P   = left_orthogonal_tt_cores
    Q   = right_orthogonal_tt_cores
    R   = up_orthogonal_tt_cores

    t1 = jnp.einsum('zki,kiaj,zkj->zka', sigmas, Q,  nus)
    t2 = jnp.einsum('zki,kiaj,zkj->zka', mus,    P,  taus)
    t3 = jnp.einsum('zki,kiaj,zkj->zka', mus,    dU, nus)
    t4 = jnp.einsum('zki,kiaj,zkj->zka', mus,    R,  nus)

    actions = jnp.einsum('zka,kab->zkb', t1 + t2 + t3, B) + jnp.einsum('zka,kab->zkb', t4, dB)
    return actions


def ut3_tangent_actions(
        basis_variations:           jnp.ndarray, # shape=(d, n, N)
        tt_variations:              jnp.ndarray, # shape=(d, r, n, r)
        vectors:                    jnp.ndarray, # shape=(num_samples, d, N)
        orthogonal_basis_cores:     jnp.ndarray, # shape=(d, n, N)
        left_orthogonal_tt_cores:   jnp.ndarray, # shape=(d, r, n, r)
        right_orthogonal_tt_cores:  jnp.ndarray, # shape=(d, r, n, r)
        up_orthogonal_tt_cores:     jnp.ndarray, # shape=(d, r, n, r)
        batch_size: int = None,
) -> jnp.ndarray:  # actions, shape=(num_samples, d, N)
    if batch_size is not None:
        return compute_result_in_batches(
            lambda XX: ut3_tangent_actions(
                basis_variations, tt_variations,
                XX[0], orthogonal_basis_cores,
                left_orthogonal_tt_cores,
                right_orthogonal_tt_cores,
                up_orthogonal_tt_cores,
                batch_size=None,
            ),
            (vectors,),
            batch_size,
        )

    reduced_xx  = jnp.einsum('kia,ska->ski', orthogonal_basis_cores, vectors)
    reduced_dxx = jnp.einsum('kia,ska->ski', basis_variations, vectors)

    mus = ut3_mus(left_orthogonal_tt_cores, reduced_xx)
    nus = ut3_nus(right_orthogonal_tt_cores, reduced_xx)
    sigmas  = ut3_sigmas(tt_variations, right_orthogonal_tt_cores, up_orthogonal_tt_cores, mus, reduced_xx, reduced_dxx)
    taus    = ut3_taus(tt_variations, left_orthogonal_tt_cores, up_orthogonal_tt_cores, nus, reduced_xx, reduced_dxx)

    tangent_actions = ut3_assemble_tangent_actions(basis_variations, tt_variations, mus, nus, sigmas, taus,
                                                   orthogonal_basis_cores, left_orthogonal_tt_cores,
                                                   right_orthogonal_tt_cores, up_orthogonal_tt_cores)
    return tangent_actions


########    Uniform Tucker tensor train tangent actions transpose    ########

@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_tau_tildes(
        reduced_dyy:                jnp.ndarray, # shape=(num_samples, d, n)
        left_orthogonal_tt_cores:   jnp.ndarray, # shape=(d, r, n, r)
        mus:                        jnp.ndarray, # shape=(num_samples, d, r)
        reduced_xx:                 jnp.ndarray, # shape=(num_samples, d, n)
        batch_size: int = None,
) -> jnp.ndarray: # uniform_tau_tildes, shape=(num_samples, d, r)
    if batch_size is not None:
        return compute_result_in_batches(
            lambda XX: ut3_tau_tildes(XX[0], left_orthogonal_tt_cores, XX[1], XX[2], batch_size=None),
            (reduced_dyy, mus, reduced_xx),
            batch_size,
        )

    def _push_left(prev_tau_tilde, ith_data):
        P, dy, v, mu = ith_data
        t1 = jnp.einsum('zi,za,iaj->zj', prev_tau_tilde, v, P)
        t2 = jnp.einsum('zi,za,iaj->zj', mu, dy, P)
        tau_tilde = t1 + t2
        return tau_tilde, prev_tau_tilde

    all_data = (
        left_orthogonal_tt_cores,
        reduced_dyy.swapaxes(0,1),
        reduced_xx.swapaxes(0,1),
        mus.swapaxes(0,1),
    )

    d, r, _, _ = left_orthogonal_tt_cores.shape
    num_samples = reduced_xx.shape[0]
    init_tau_tilde = jnp.zeros((num_samples, r))

    _, tau_tildes_swapped = jax.lax.scan(_push_left, init_tau_tilde, all_data)
    tau_tildes = tau_tildes_swapped.swapaxes(0,1)
    return tau_tildes


@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_sigma_tildes(
        reduced_dyy:                jnp.ndarray,  # shape=(num_samples, d, n)
        right_orthogonal_tt_cores:  jnp.ndarray,  # shape=(d, r, n, r)
        nus:                        jnp.ndarray,  # shape=(num_samples, d, r)
        reduced_xx:                 jnp.ndarray,  # shape=(num_samples, d, n)
        batch_size: int = None,
) -> jnp.ndarray:  # uniform_sigma_tildes, shape=(num_samples, d, r)
    return ut3_tau_tildes(
        reduced_dyy[:, ::-1, :],
        right_orthogonal_tt_cores[::-1, :, :, :].swapaxes(1, 3), # reverse tensor train
        nus[:, ::-1, :],
        reduced_xx[:, ::-1, :],
        batch_size=batch_size,
    )[:, ::-1, :]


@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_assemble_tangent_tt_core_perturbations(
        mus:                jnp.ndarray, # shape=(num_samples, d, r)
        nus:                jnp.ndarray, # shape=(num_samples, d, r)
        sigma_tildes:       jnp.ndarray, # shape=(num_samples, d, r)
        tau_tildes:         jnp.ndarray, # shape=(num_samples, d, r)
        reduced_xx:         jnp.ndarray, # shape=(num_samples, d, n)
        reduced_dyy:        jnp.ndarray, # shape=(num_samples, d, n)
        batch_size: int = None,
) -> jnp.ndarray: # shape=(num_samples, d, N)
    if batch_size is not None:
        return compute_summed_result_in_batches(
            lambda XX: (ut3_assemble_tangent_tt_core_perturbations(
                XX[0], XX[1], XX[2], XX[3], XX[4], XX[5], batch_size=None,
            ),),
            (mus, nus, sigma_tildes, tau_tildes, reduced_xx, reduced_dyy),
            batch_size,
        )[0]

    t1 = jnp.einsum('sdi,sda,sdj->diaj', mus,           reduced_xx,     sigma_tildes)
    t2 = jnp.einsum('sdi,sda,sdj->diaj', tau_tildes,    reduced_xx,     nus)
    t3 = jnp.einsum('sdi,sda,sdj->diaj', mus,           reduced_dyy,    nus)
    dU = t1 + t2 + t3
    return dU


@ft.partial(jax.jit, static_argnames=['batch_size'])
def ut3_assemble_tangent_basis_core_perturbations(
        mus:                    jnp.ndarray,  # shape=(num_samples, d, r)
        nus:                    jnp.ndarray,  # shape=(num_samples, d, r)
        sigma_tildes:           jnp.ndarray,  # shape=(num_samples, d, r)
        tau_tildes:             jnp.ndarray,  # shape=(num_samples, d, r)
        xx:                     jnp.ndarray,  # shape=(num_samples, d, N)
        dyy:                    jnp.ndarray,  # shape=(num_samples, d, N)
        up_orthogonal_tt_cores: jnp.ndarray,  # shape=(d, r, n, r)
        batch_size: int = None,
) -> jnp.ndarray: # shape=(d, n, N)
    if batch_size is not None:
        return compute_summed_result_in_batches(
            lambda XX: (ut3_assemble_tangent_basis_core_perturbations(
                XX[0], XX[1], XX[2], XX[3], XX[4], XX[5], up_orthogonal_tt_cores, batch_size=None,
            ),),
            (mus, nus, sigma_tildes, tau_tildes, xx, dyy),
            batch_size,
        )[0]

    R = up_orthogonal_tt_cores
    dB_t1 = jnp.einsum('sda,sdx->dax', jnp.einsum('sdi,diaj,sdj->sda', mus,          R, sigma_tildes),   xx)
    dB_t2 = jnp.einsum('sda,sdx->dax', jnp.einsum('sdi,diaj,sdj->sda', tau_tildes,   R, nus),            xx)
    dB_t3 = jnp.einsum('sda,sdx->dax', jnp.einsum('sdi,diaj,sdj->sda', mus,          R, nus),            dyy)
    dB = dB_t1 + dB_t2 + dB_t3
    return dB


def ut3_tangent_actions_transpose(
        action_perturbations:       jnp.ndarray, # shape=(num_samples, d, N)
        vectors:                    jnp.ndarray,  # shape=(num_samples, d, N)
        orthogonal_basis_cores:     jnp.ndarray,  # shape=(d, n, N)
        left_orthogonal_tt_cores:   jnp.ndarray,  # shape=(d, r, n, r)
        right_orthogonal_tt_cores:  jnp.ndarray,  # shape=(d, r, n, r)
        up_orthogonal_tt_cores:     jnp.ndarray,  # shape=(d, r, n, r)
        batch_size:                 int = None,
        batch_size_variations:      int = None,
) -> typ.Tuple[
    jnp.ndarray, # basis_core_perturbations
    jnp.ndarray, # tt_core_perturbations
]:
    if batch_size is not None:
        return compute_summed_result_in_batches(
            lambda XX: ut3_tangent_actions_transpose(
                XX[0], XX[1],
                orthogonal_basis_cores, left_orthogonal_tt_cores, right_orthogonal_tt_cores, up_orthogonal_tt_cores,
                batch_size=None,
                batch_size_variations=batch_size_variations,
            ),
            (action_perturbations, vectors),
            batch_size,
        )

    xx = vectors
    dyy = action_perturbations
    reduced_xx  = jnp.einsum('kia,ska->ski', orthogonal_basis_cores, xx)
    reduced_dyy = jnp.einsum('kia,ska->ski', orthogonal_basis_cores, dyy)
    mus             = ut3_mus(left_orthogonal_tt_cores, reduced_xx)
    nus             = ut3_nus(right_orthogonal_tt_cores, reduced_xx)
    tau_tildes      = ut3_tau_tildes(reduced_dyy, left_orthogonal_tt_cores, mus, reduced_xx)
    sigma_tildes    = ut3_sigma_tildes(reduced_dyy, right_orthogonal_tt_cores, nus, reduced_xx)

    dB = ut3_assemble_tangent_basis_core_perturbations(
        mus, nus, sigma_tildes, tau_tildes, xx, dyy, up_orthogonal_tt_cores, batch_size=batch_size_variations,
    )
    dU = ut3_assemble_tangent_tt_core_perturbations(
        mus, nus, sigma_tildes, tau_tildes, reduced_xx, reduced_dyy, batch_size=batch_size_variations,
    )
    return dB, dU
