# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import math
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.t3_operations as ragged_ops
import t3toolbox.backend.ut3_operations as uniform_ops
from t3toolbox.backend.common import *

__all__ = [
    # Probe a Tucker tensor train
    'probe_t3',
    'compute_xis',
    'compute_mus',
    'compute_nus',
    'compute_etas',
    'assemble_zs',
    # Probe a tangent vector
    'probe_tangent',
    'compute_dxis',
    'compute_sigmas',
    'compute_taus',
    'compute_detas',
    'assemble_tangent_zs',
    # Apply / entries of a tangent vector (all-modes special case of probing)
    'apply_tangent',
    'entries_tangent',
    'apply_tangent_transpose',
    'entries_tangent_transpose',
    # Apply -- base-sweep reuse split (precompute the LEAN (xi,mu) base edge vars once; inject into the
    # bare J / Jᵀ; adjoint-state transpose recomputes the right context as sigma_hat; for fitting.py)
    'precompute_apply_base_sweep',
    'apply_jacobian_from_sweep',
    'apply_transpose_from_sweep',
    'compute_sigma_hats',
    # Entries -- base-sweep reuse split (the fiber-sliced seed; one-hot adjoint-state transpose; for fitting.py)
    'precompute_entries_base_sweep',
    'entries_jacobian_from_sweep',
    'entries_transpose_from_sweep',
    # Probe -- base-sweep reuse split (the FULL (xi,mu,nu,eta) sweep; for fitting.py)
    'precompute_probe_base_sweep',
    'probe_jacobian_from_sweep',
    'probe_transpose_from_sweep',
    # Corewise (non-manifold) transpose -- the tangent transpose with the base's cores in place of the frames
    'apply_corewise_transpose',
    'entries_corewise_transpose',
    # Transpose of map from tangent vector to probes
    'compute_deta_tildes',
    'compute_tau_tildes',
    'compute_sigma_tildes',
    'compute_dxi_tildes',
    'assemble_tucker_variations',
    'assemble_tt_variations',
    'probe_tangent_transpose',
    # Ambient / corewise probe transposes (the plain-probe analogs of the apply/entries transposes)
    'probe_ambient_transpose',
    'probe_corewise_transpose',
    # Probe a dense tensor
    'probe_dense',
]

# NOTE: probing is intentionally UNWEIGHTED. In the typical regime (many probes at once) it is
# cheaper to absorb any edge weights into the cores once, up front, then probe the weighted cores
# with the plain functions below (rather than threading weights through every probe). The up-front
# weighting helper lives with the (deferred) weighted-tensor-network code.


#####################################################
########    Probing a Tucker Tensor Train    ########
#####################################################

def probe_t3(
        ww: typ.Union[typ.Sequence[NDArray],    NDArray],   # len=d, elm_shape=W+(Ni,)
        x:  typ.Union[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # ragged, (tucker_cores, tt_cores)
            typ.Tuple[NDArray, NDArray],  # uniform, (tucker_supercore, tt_supercore)
        ],
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,Ni)
    '''Probe a Tucker tensor train.

    See Section 6.2, particularly Figure 7 and Algorithm 5, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    ww: typ.Sequence[NDArray]
        input vectors to probe with. len=d, elm_shape=(...,Ni)
    x: t3.TuckerTensorTrain.data
        Tucker tensor train to probe, as a (tucker_cores, tt_cores) data tuple.
        structure=((N0,...,N(d-1)),(n0,...,n(d-1)),(1,r1,...,r(d-1),1))

    Returns
    -------
    typ.Tuple[NDArray,...]
        Probes, zz. len=d, elm_shape=(...,Ni)

    See Also
    --------
    probe_tangent
    probe_tangent_transpose
    compute_xis
    compute_mus
    compute_nus
    compute_etas
    assemble_zs

    Examples
    --------
    Probe a T3 with one set of vectors; value-match against the dense reference:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)).data
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.probe_t3(ww, x)
    >>> zz_dense = t3p.probe_dense(ww, t3.TuckerTensorTrain(*x).to_dense())   # dense reference
    >>> print([z.shape for z in zz])        # one probe per mode, elm_shape=(Ni,)
    [(10,), (11,), (12,)]
    >>> print([bool(np.allclose(z, z2)) for z, z2 in zip(zz, zz_dense)])
    [True, True, True]

    Vectorize over probes: a probe stack ``W`` rides through to ``elm_shape = W + (Ni,)``:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)).data
    >>> ww = (np.random.randn(2, 3, 10), np.random.randn(2, 3, 11), np.random.randn(2, 3, 12))
    >>> zz = t3p.probe_t3(ww, x)
    >>> zz_dense = t3p.probe_dense(ww, t3.TuckerTensorTrain(*x).to_dense())
    >>> print([z.shape for z in zz])        # W=(2,3) outer, mode index inner
    [(2, 3, 10), (2, 3, 11), (2, 3, 12)]
    >>> print([bool(np.allclose(z, z2)) for z, z2 in zip(zz, zz_dense)])
    [True, True, True]

    Vectorize over probes AND T3s: both stacks ride through, base-inner ``elm_shape = W + C + (Ni,)``:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1), stack_shape=(4, 5)).data
    >>> ww = (np.random.randn(2, 3, 10), np.random.randn(2, 3, 11), np.random.randn(2, 3, 12))
    >>> zz = t3p.probe_t3(ww, x)
    >>> zz_dense = t3p.probe_dense(ww, t3.TuckerTensorTrain(*x).to_dense())
    >>> print(zz[0].shape)                  # W=(2,3) outer, C=(4,5) inner, then N0=10
    (2, 3, 4, 5, 10)
    >>> print([bool(np.allclose(z, z2)) for z, z2 in zip(zz, zz_dense)])
    [True, True, True]
    '''
    tucker_cores, tt_cores = x

    xis = compute_xis(tucker_cores, ww)

    mus = compute_mus(tt_cores, xis)

    nus = compute_nus(tt_cores, xis)

    etas = compute_etas(tt_cores, mus, nus)

    zs = assemble_zs(tucker_cores, etas)

    return zs


def compute_xis(
        up_tucker_cores:    typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=C+(nUi,Ni)
        ww:                 typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=W+(Ni,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # xis. len=d, elm_shape=(...,nUi)
    '''Compute upward edge variables associated with edges between Tucker cores and adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 6.2, particularly Figure 7 and Algorithm 5, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((up_tucker_cores, ww))
    is_uniform = is_ndarray(up_tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        xis = contractions.dCio_dWo_to_dWCi(up_tucker_cores, ww)
    else:
        def _func(x):
            U, w = x
            return (contractions.Cio_Wo_to_WCi(U, w),)

        (xis,) = xmap(_func, (up_tucker_cores, ww))

    return xis


def compute_mus(
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d-1. elm_shape=C+(rLi,nUi,rL(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=W+C+(nUi,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # mus. len=d, elm_shape=W+C+(rLi,)
    '''Compute leftward edge variables associated with edges between adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 6.2, particularly Figure 7 and Algorithm 5, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((left_tt_cores, xis))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    def _func(mu, x):
        P, xi = x[0], x[1]
        mu_next = contractions.WCa_Caib_WCi_to_WCb(mu, P, xi)
        return mu_next, (mu,)

    # carry has the same leading stack as the edge variables (order-agnostic), plus the left bond
    r0 = left_tt_cores[0].shape[-3]
    init = xnp.ones(xis[0].shape[:-1] + (r0,))

    last_mu, (mus,) = xscan(_func, init, (left_tt_cores, xis))
    return mus


def compute_nus(
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=C+(rRi,nUi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=W+C+(nUi,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # nus. len=d, elm_shape=W+C+(rR(i+1),)
    '''Compute rightward edge variables associated with edges between adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 6.2, particularly Figure 7 and Algorithm 5, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    is_uniform = is_ndarray(right_tt_cores)
    reverse = uniform_ops.reverse_utt if is_uniform else ragged_ops.reverse_tt

    rev_nus = compute_mus(reverse(right_tt_cores), xis[::-1])
    return rev_nus[::-1]


def compute_etas(
        down_tt_cores:         typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=C+(rLi,nOi,rR(i+1))
        mus:                    typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=W+C+(rLi,)
        nus:                    typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=(...,rR(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # etas. len=d, elm_shape=W+C+(nOi,)
    '''Compute downward edge variables associated with edges between Tucker cores and adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 6.2, particularly Figure 7 and Algorithm 5, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((down_tt_cores, mus, nus))
    is_uniform = is_ndarray(down_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        etas = contractions.dWCa_dCaib_dWCb_to_dWCi(mus, down_tt_cores, nus)
    else:
        def _func(x):
            mu, G, nu = x
            return (contractions.WCa_Caib_WCb_to_WCi(mu, G, nu),)

        (etas,) = xmap(_func, (mus, down_tt_cores, nus))

    return etas


def assemble_zs(
        tucker_cores:   typ.Union[typ.Sequence[NDArray], NDArray],  # len=d. elm_shape=C+(ni,Ni)
        etas:           typ.Union[typ.Sequence[NDArray], NDArray],  # len=d. elm_shape=W+C+(ni,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # zs. len=d, elm_shape=W+C+(Ni,)
    '''Assemble probes from downward edge variables.

    See Section 6.2, particularly Figure 7 and Algorithm 5, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((tucker_cores, etas))
    is_uniform = is_ndarray(tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        zs = contractions.dWCi_dCio_to_dWCo(etas, tucker_cores)
    else:
        def _func(x):
            eta, U = x
            return (contractions.WCi_Cio_to_WCo(eta, U),)

        (zs,) = xmap(_func, (etas, tucker_cores))

    return zs


#####################################################
###########    Probing a tangent vector    ##########
#####################################################

def compute_dxis(
        var_tucker_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=(nOi,Ni)
        ww:                     typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=(...,Ni)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # dxis. len=d, elm_shape=(...,nOi)
    '''Compute var-upward edge variables dxi.
    Used for probing a tangent vector.

    Same as compute_xis(), except with var_tucker_cores in place of tucker_cores.

    See Section 6.2.2, particularly Algorithm 7, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_xis
    compute_sigmas
    compute_taus
    compute_detas
    assemble_tangent_zs
    probe_tangent
    '''
    return compute_xis(var_tucker_cores, ww)


def _sigma_step(sigma, Q, O, dG, xi, dxi, mu):
    '''One step of the perturbation-leftward (sigma) recursion (Algorithm 7), shared by
    compute_sigmas (which keeps the per-core sequence, for probing) and apply_tangent/entries_tangent
    (which keep only the terminal carry, for the all-modes contraction).

    Three-group (W probe, K tangent, C base): sigma/dxi carry K, the base edge vars (xi, mu) and base
    cores (Q, O) do not. t1/t3 self-infer the split from the C-only base core; t2's only core is the
    variation core dG (K+C), so len(C) is supplied via n_base (recovered from the C-only Q, the
    n_probe precedent). Reduces to the two-group result when K is empty.
    '''
    n_base = Q.ndim - 3
    t1 = contractions.WKCa_Caib_WCi_to_WKCb(sigma, Q, xi)
    t2 = contractions.WCa_KCaib_WCi_to_WKCb(mu, dG, xi, n_base)
    t3 = contractions.WCa_Caib_WKCi_to_WKCb(mu, O, dxi)
    return t1 + t2 + t3


def compute_sigmas(
        var_tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        down_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi),
        dxis:               typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nOi)
        mus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rLi)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # sigmas. len=d, elm_shape=(...,rR(i+1))
    '''Compute var-leftward edge variables sigma.
    Used for probing a tangent vector.

    See Section 6.2.2, particularly Algorithm 7, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_dxis
    compute_taus
    compute_detas
    assemble_tangent_zs
    probe_tangent
    '''
    use_jax = tree_contains_jax((var_tt_cores, right_tt_cores, down_tt_cores, xis, dxis, mus))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    def _func(sigma, x):
        Q, O, dG, xi, dxi, mu = x
        return _sigma_step(sigma, Q, O, dG, xi, dxi, mu), (sigma,)

    # carry sigma is W+K+C; take the leading stack from dxis (which carries K), not xis (W+C only)
    rR0 = right_tt_cores[0].shape[-3]
    init = xnp.zeros(dxis[0].shape[:-1] + (rR0,))

    last_sigma, (sigmas,) = xscan(_func, init, (right_tt_cores, down_tt_cores, var_tt_cores, xis, dxis, mus))
    return sigmas


def compute_taus(
        var_tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rL(i+1))
        down_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi),
        dxis:               typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nOi)
        nus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # taus. len=d, elm_shape=(...,rL(i+1))
    '''Compute var-rightward edge variables tau.
    Used for probing a tangent vector.

    See Section 6.2.2, particularly Algorithm 7, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_dxis
    compute_sigmas
    compute_detas
    assemble_tangent_zs
    probe_tangent
    '''
    is_uniform = is_ndarray(var_tt_cores)
    reverse = uniform_ops.reverse_utt if is_uniform else ragged_ops.reverse_tt

    rev_taus = compute_sigmas(
        reverse(var_tt_cores), reverse(left_tt_cores), reverse(down_tt_cores),
        xis[::-1], dxis[::-1], nus[::-1],
    )
    return rev_taus[::-1]


def compute_detas(
        var_tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rL(i+1))
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        mus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rLi)
        nus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
        sigmas:             typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
        taus:               typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rL(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # detas. len=d, elm_shape=(...,nUi)
    '''Compute var-downward edge variables deta.
    Used for probing a tangent vector.

    See Section 6.2.2, particularly Algorithm 7, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_dxis
    compute_sigmas
    compute_taus
    assemble_tangent_zs
    probe_tangent
    '''
    use_jax = tree_contains_jax((var_tt_cores, left_tt_cores, right_tt_cores, mus, nus, sigmas, taus))
    is_uniform = not isinstance(mus, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        term1 = xnp.einsum(
            'd...aj,d...j->d...a',
            xnp.einsum('d...i,diaj->d...aj', sigmas, right_tt_cores),
            nus,
        )
        term2 = xnp.einsum(
            'd...aj,d...j->d...a',
            xnp.einsum('d...i,diaj->d...aj', mus, var_tt_cores),
            nus,
        )
        term3 = xnp.einsum(
            'd...aj,d...j->d...a',
            xnp.einsum('d...i,diaj->d...aj', mus, left_tt_cores),
            taus,
        )
        detas = term1 + term2 + term3
    else:
        def _func(x):
            P, Q, dG, mu, nu, sigma, tau = x
            # Three-group contractions (see compute_sigmas): sigma/tau carry K, mu/nu and base cores
            # P/Q do not. term1/term3 self-infer; term2's only core is dG (K+C) -> n_base from Q.
            n_base = Q.ndim - 3
            term1 = contractions.WKCa_Caib_WCb_to_WKCi(sigma, Q, nu)
            term2 = contractions.WCa_KCaib_WCb_to_WKCi(mu, dG, nu, n_base)
            term3 = contractions.WCa_Caib_WKCb_to_WKCi(mu, P, tau)
            return (term1 + term2 + term3,)

        xs = (left_tt_cores, right_tt_cores, var_tt_cores, mus, nus, sigmas, taus)
        (detas,) = xmap(_func, xs)

    return detas


def assemble_tangent_zs(
        tucker_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=(nUi,Ni)
        var_tucker_cores:   typ.Union[typ.Sequence[NDArray], NDArray], # len=d. elm_shape=(nOi,Ni)
        etas:               typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nOi)
        detas:              typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # zs. len=d, elm_shape=(...,Ni)
    '''Assemble tangent vector probes from edge variables.

    See Section 6.2.2, particularly Algorithm 7, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_dxis
    compute_sigmas
    compute_taus
    compute_detas
    probe_tangent
    '''
    use_jax = tree_contains_jax((tucker_cores, var_tucker_cores, etas, detas))
    is_uniform = not isinstance(etas, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        term1 = xnp.einsum('dao,d...a->d...o', tucker_cores, detas)
        term2 = xnp.einsum('dao,d...a->d...o', var_tucker_cores, etas)
        zs = term1 + term2
    else:
        def _func(x):
            B, dB, eta, deta = x
            # Three-group contractions (see compute_sigmas): deta carries K (term1 fuses W+K over the
            # C-only base core B, via the delegator); eta is W+C and dB is the variation core K+C, so
            # term2 needs len(C) -- recovered here from the C-only tucker core B (2 tensor axes).
            n_base = B.ndim - 2
            term1 = contractions.WKCi_Cio_to_WKCo(deta, B)
            term2 = contractions.WCi_KCio_to_WKCo(eta, dB, n_base)
            return (term1 + term2,)

        (zs,) = xmap(_func, (tucker_cores, var_tucker_cores, etas, detas))

    return zs


def probe_jacobian_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],
        ww:         typ.Sequence[NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,) -- for the variation's dxis
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = precompute_probe_base_sweep(base, ww)
) -> typ.Sequence[NDArray]:                 # probes, len=d, elm_shape=W+K+C+(Ni,) (one free mode each)
    '''Forward probe of a tangent vector reusing a precomputed base sweep -- the bare ``𝒥`` (probe) with
    the base edge variables injected. Equivalent to :py:func:`probe_tangent`, but takes
    ``(xis, mus, nus, etas)`` from ``base_sweep`` instead of recomputing them; only the perturbation
    sweep (``dxis``/``sigmas``/``taus``/``detas``) is computed here. Apply and probe **share** the base
    sweep (:py:func:`precompute_probe_base_sweep`). No gauge projector ``Π``.

    See Also
    --------
    precompute_probe_base_sweep
    probe_tangent
    probe_transpose_from_sweep
    '''
    var_tucker_cores, var_tt_cores = variation
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    xis, mus, nus, etas = base_sweep
    dxis   = compute_dxis(var_tucker_cores, ww)
    sigmas = compute_sigmas(var_tt_cores, right_tt_cores, down_tt_cores, xis, dxis, mus)
    taus   = compute_taus(var_tt_cores, left_tt_cores, down_tt_cores, xis, dxis, nus)
    detas  = compute_detas(var_tt_cores, left_tt_cores, right_tt_cores, mus, nus, sigmas, taus)
    return assemble_tangent_zs(up_tucker_cores, var_tucker_cores, etas, detas)


def probe_tangent(
        ww:         typ.Union[typ.Sequence[NDArray],    NDArray],  # input vectors, len=d, elm_shape=(...,Ni)
        variation:  typ.Union[
            typ.Tuple[
                typ.Sequence[NDArray],  # var_tucker_cores. len=d, elm_shape=(nOi,Ni)
                typ.Sequence[NDArray],  # var_tt_cores.     len=d, elm_shape=(rLi,nUi,rRi)
            ],
            typ.Tuple[
                NDArray,  # var_tucker_supercore.
                NDArray,  # var_tt_supercore.
            ],
        ],
        base:       typ.Union[
            typ.Tuple[
                typ.Sequence[NDArray],  # up_tucker_cores. len=d. U_xo U_yo   = I_xy, U.shape = (nU, N)
                typ.Sequence[NDArray],  # down_tt_cores.   len=d. O_ixj O_iyj = I_xy  O.shape = (rL, nO, rR)
                typ.Sequence[NDArray],  # left_tt_cores.   len=d. P_iax P_iay = I_xy, P.shape = (rL, nU, rR)
                typ.Sequence[NDArray],  # right_tt_cores.  len=d. Q_xaj Q_yaj = I_xy  Q.shape = (rL, nU, rR)
            ],
            typ.Tuple[
                NDArray,  # up_tucker_supercore. shape=(d, nU, N),      up orthogonal elements
                NDArray,  # down_tt_supercore.   shape=(d, rL, nO, rR), down orthogonal elements
                NDArray,  # left_tt_supercore.   shape=(d, rL, nU, rR), left orthogonal elements
                NDArray,  # right_tt_supercore.  shape=(d, rL, nU, rR), right orthogonal elements
            ],
        ], # base order = T3Basis.data = (up, down, left, right) = (U, O, P, Q)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,Ni)
    '''Probe a tangent vector. Applies the (single-sample) least-squares Jacobian J^(s).

    Two independent stackings may ride along (handled by the W/C custom contractions in
    ``contractions.py``): the T3 stack ``C`` (the base/variation cores' ``stack_shape``) and the
    probe stack ``W`` (the probing vectors' batch). When both are present the probes are
    double-stacked, ``elm_shape = W + C + (Ni,)`` (probe outer, base inner).

    See Section 6.2.2, particularly Algorithms 6 and 7, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    ww: typ.Sequence[NDArray]
        input vectors to probe with. len=d, elm_shape=(...,Ni)
    variation: bvf.T3Variations.data
        Tangent direction, as a (tucker_variations, tt_variations) data tuple.
    base: (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
        Orthogonal base for the point where the tangent space attaches to the manifold. This is
        exactly ``T3Basis.data`` order (U, O, P, Q) -- pass ``basis.data`` directly, no reorder.

    Returns
    -------
    typ.Tuple[NDArray,...]
        Probes, zz. len=d, elm_shape=(Ni,) or (...,Ni)

    See Also
    --------
    probe_t3
    probe_tangent_transpose
    compute_dxis
    compute_sigmas
    compute_taus
    compute_detas
    assemble_tangent_zs

    Examples
    --------

    Probe a tangent vector with one set of vectors; value-match against the dense reference:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
    >>> base, variations = bvf.t3_orthogonal_representations(x)
    >>> probe_base = base.data  # probing's base order == T3Basis.data, no reorder
    >>> v = t3m.T3Tangent(base, variations)
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.probe_tangent(ww, variations.data, probe_base)
    >>> zz_dense = t3p.probe_dense(ww, v.to_dense())   # dense reference J^(s) v
    >>> print([z.shape for z in zz])        # one probe per mode, elm_shape=(Ni,)
    [(10,), (11,), (12,)]
    >>> print([bool(np.allclose(z, z2)) for z, z2 in zip(zz, zz_dense)])
    [True, True, True]

    Probe with a stack of vectors: the probe stack ``W`` rides through, ``elm_shape = W + (Ni,)``:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
    >>> base, variations = bvf.t3_orthogonal_representations(x)
    >>> v = t3m.T3Tangent(base, variations)
    >>> www = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
    >>> zzz = t3p.probe_tangent(www, variations.data, base.data)
    >>> zzz_dense = t3p.probe_dense(www, v.to_dense())
    >>> print(zzz[0].shape)                 # W=(2,) outer, then N0=10
    (2, 10)
    >>> print([bool(np.allclose(z, z2)) for z, z2 in zip(zzz, zzz_dense)])
    [True, True, True]
    '''
    (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores) = base
    (var_tucker_cores, var_tt_cores) = variation

    xis = compute_xis(up_tucker_cores, ww)

    mus = compute_mus(left_tt_cores, xis)

    nus = compute_nus(right_tt_cores, xis)

    etas = compute_etas(down_tt_cores, mus, nus)

    dxis = compute_dxis(var_tucker_cores, ww)

    sigmas = compute_sigmas(
        var_tt_cores, right_tt_cores, down_tt_cores, xis, dxis, mus,
    )

    taus = compute_taus(
        var_tt_cores, left_tt_cores, down_tt_cores, xis, dxis, nus,
    )

    detas = compute_detas(
        var_tt_cores, left_tt_cores, right_tt_cores, mus, nus, sigmas, taus,
    )

    zz = assemble_tangent_zs(
        up_tucker_cores, var_tucker_cores, etas, detas,
    )

    return zz


#####################################################
#####    Apply / entries of a tangent vector    #####
#####################################################
#
# apply and entries are the all-modes special case of probing (probing leaves ONE mode free; these
# contract EVERY mode). With no free mode the whole computation collapses to a single left-to-right
# pass: the base left sweep mu-hat (via P) feeds the perturbation sweep sigma (via Q, Algorithm 7),
# which is then contracted at the terminal bond. No right (nu) sweep, no central (eta), no per-mode
# assembly -- roughly half of probe_tangent. entries is apply with the up-index xis obtained by
# slicing Tucker-core fibers (no contraction with unit vectors, so no N factor).


def _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores):
    '''Run the perturbation sigma sweep (via Q) to its TERMINAL carry and contract the final bond.

    Shared tail of apply_tangent and entries_tangent (they differ only in how xis/dxis are formed).
    '''
    use_jax = tree_contains_jax((xis, dxis, mus, right_tt_cores))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    def _func(sigma, x):
        Q, O, dG, xi, dxi, mu = x
        return _sigma_step(sigma, Q, O, dG, xi, dxi, mu), (0,)

    # carry sigma is W+K+C; take the leading stack from dxis (carries K), not xis (W+C only).
    rR0 = right_tt_cores[0].shape[-3]
    init = xnp.zeros(dxis[0].shape[:-1] + (rR0,))
    sigma_terminal, _ = xscan(_func, init, (right_tt_cores, down_tt_cores, var_tt_cores, xis, dxis, mus))
    return xnp.sum(sigma_terminal, axis=-1)   # contract the terminal bond -> W + K + C


def _entry_xis(tucker_cores, index):
    '''Up-index edge variables by slicing Tucker-core fibers at ``index`` (no contraction).

    tucker_cores[i].shape = C + (p_i, Ni); index is an int array of shape (d,) + W (index stack W).
    Returns xis with elm_shape = W + C + (p_i,) -- the same layout compute_xis produces, so the
    downstream mu/sigma sweeps are identical to apply.
    '''
    use_jax = tree_contains_jax((tucker_cores,))
    xnp, _, _ = get_backend(False, use_jax)
    index = xnp.array(index)
    n_idx = len(index.shape[1:])                                  # number of index-stack (W) axes
    xis = []
    for i, B in enumerate(tucker_cores):
        xi_CpW = B[..., index[i]]                                 # C + (p,) + W (index batch trails)
        xi_WCp = xnp.moveaxis(xi_CpW, tuple(range(-n_idx, 0)), tuple(range(n_idx)))  # -> W + C + (p,)
        xis.append(xi_WCp)
    return tuple(xis)


def precompute_apply_base_sweep(
        base:   typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # base order = T3Basis.data = (up, down, left, right)
        ww:     typ.Sequence[NDArray],      # apply vectors, len=d, elm_shape=W+(Ni,)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xis. len=d, elm_shape=W+C+(nUi,)
    typ.Sequence[NDArray],  # mus. len=d, elm_shape=W+C+(rLi,)
]:                                          # lean base sweep -- (xis, mus) only
    '''The all-modes apply **base sweep** (lean): the base edge variables ``(xi-hat, mu-hat)`` that
    depend only on the base frame and the apply vectors ``ww`` -- NOT on the tangent or residual.
    Computing them is the expensive, ``W``-scaled part of the apply Jacobian; precomputed **once per
    base** and reused across every ``J`` / ``Jᵀ`` of an inner solve (the reuse hook for ``fitting.py``).

    **Lean ``(xis, mus)`` only** (not the right ``nu`` / down ``eta`` sweeps): the all-modes apply
    forward AND its adjoint-state transpose use only ``(xi, mu)`` -- the transpose recomputes the right
    context as ``sigma_hat`` from the residual rather than storing ``nu``/``eta``, halving the
    ``W``-scaling memory (apply on the manifold, §6.2.2 of Alger et al. (2026)). Probe, which leaves a
    mode free, needs the full sweep -- :py:func:`precompute_probe_base_sweep`.

    See Also
    --------
    apply_jacobian_from_sweep
    apply_transpose_from_sweep
    precompute_probe_base_sweep
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    xis = compute_xis(up_tucker_cores, ww)
    mus = compute_mus(left_tt_cores, xis)
    return xis, mus


def precompute_probe_base_sweep(
        base:   typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # base order = T3Basis.data = (up, down, left, right)
        ww:     typ.Sequence[NDArray],      # probe vectors, len=d, elm_shape=W+(Ni,)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xis.  len=d, elm_shape=W+C+(nUi,)
    typ.Sequence[NDArray],  # mus.  len=d, elm_shape=W+C+(rLi,)
    typ.Sequence[NDArray],  # nus.  len=d, elm_shape=W+C+(rR(i+1),)
    typ.Sequence[NDArray],  # etas. len=d, elm_shape=W+C+(nOi,)
]:                                          # full base sweep -- (xis, mus, nus, etas)
    '''The **probe** base sweep (full): all four base edge variables ``(xi, mu, nu, eta)``. The probe
    leaves one mode free, so its transpose's per-mode (vector) residual must be propagated through both
    the left ``mu`` and right ``nu`` sweeps + the central ``eta`` combine -- it cannot use the
    scalar-residual adjoint-state shortcut that lets apply/entries drop ``nu``/``eta``
    (:py:func:`precompute_apply_base_sweep`). Reused across the probe forward / transpose of an inner
    solve. §6.2.2 of Alger et al. (2026); no gauge projector ``Π``.

    See Also
    --------
    probe_jacobian_from_sweep
    probe_transpose_from_sweep
    precompute_apply_base_sweep
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    xis  = compute_xis(up_tucker_cores, ww)
    mus  = compute_mus(left_tt_cores, xis)
    nus  = compute_nus(right_tt_cores, xis)
    etas = compute_etas(down_tt_cores, mus, nus)
    return xis, mus, nus, etas


def apply_jacobian_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],
        ww:         typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,) -- for the variation's dxis
        base:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d  (unused; for a uniform call signature)
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d  (unused)
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # base order = T3Basis.data = (up, down, left, right)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray],          # xis
            typ.Sequence[NDArray],          # mus
        ],                                  # = precompute_apply_base_sweep(base, ww)  (lean)
) -> NDArray:                               # the scalar apply(v, ww), one per stack element; shape = W + K + C
    '''Forward all-modes apply of a tangent vector reusing a precomputed base sweep -- the bare ``𝒥`` with
    the base edge variables injected. Equivalent to :py:func:`apply_tangent`, but it takes the lean
    ``(xis, mus)`` from ``base_sweep`` instead of recomputing them (the reuse hook for ``fitting.py``).
    Only the variation-dependent ``dxis`` is computed here. No gauge projector ``Π``.

    See Also
    --------
    precompute_apply_base_sweep
    apply_tangent
    '''
    var_tucker_cores, var_tt_cores = variation
    _, down_tt_cores, _, right_tt_cores = base
    xis, mus = base_sweep
    dxis = compute_dxis(var_tucker_cores, ww)     # variation-dependent; not part of the base sweep
    return _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores)


def apply_tangent(
        ww:         typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],
        base:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # base order = T3Basis.data = (up, down, left, right)
) -> NDArray:                               # the scalar apply(v, ww), one per stack element; shape = W + K + C
    '''Apply a tangent vector in all modes: contract the dense tangent with ``ww`` in every index.

    The all-modes special case of probing -- a single left-to-right pass (mu-hat via P, then the
    perturbation sigma via Q), contracted at the terminal bond. No right (nu) / central (eta) sweeps,
    no per-mode assembly. See Section 6.2.2 (Algorithms 6-7) of Alger et al. (2026).

    See Also
    --------
    entries_tangent
    probe_tangent
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    var_tucker_cores, var_tt_cores = variation

    xis  = compute_xis(up_tucker_cores, ww)       # xi-hat_i  = U_i^T w_i
    dxis = compute_dxis(var_tucker_cores, ww)     # delta-xi_i = dU_i^T w_i
    mus  = compute_mus(left_tt_cores, xis)        # base left sweep via P

    return _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores)


def precompute_entries_base_sweep(
        base:   typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # base order = T3Basis.data = (up, down, left, right)
        index:  NDArray,                    # int, shape=(d,)+W -- the grid points
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xis. len=d, elm_shape=W+C+(nUi,) -- the FIBER-SLICED seed (not contracted)
    typ.Sequence[NDArray],  # mus. len=d, elm_shape=W+C+(rLi,)
]:                                          # lean base sweep -- (xis, mus) only (entries seed)
    '''The all-modes **entries** base sweep (lean): identical to :py:func:`precompute_apply_base_sweep`
    but the ``xi-hat`` seed comes from slicing the Tucker-core fibers at ``index`` (``_entry_xis``)
    instead of contracting with probe vectors. Like apply, entries uses the adjoint-state transpose, so
    only ``(xis, mus)`` are needed (no ``nu``/``eta``). Reused by the entries forward/transpose (the
    reuse hook for ``fitting.py``).

    See Also
    --------
    precompute_apply_base_sweep
    entries_jacobian_from_sweep
    entries_transpose_from_sweep
    '''
    up_tucker_cores, _, left_tt_cores, _ = base
    xis = _entry_xis(up_tucker_cores, index)
    mus = compute_mus(left_tt_cores, xis)
    return xis, mus


def entries_jacobian_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],
        index:      NDArray,                # int, shape=(d,)+W -- for the variation's fiber-sliced dxis
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q); uses Q (right) and O (down)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = precompute_entries_base_sweep(base, index)  (lean)
) -> NDArray:                               # entries of the dense tangent at ``index``; shape = W + K + C
    '''Forward all-modes entries of a tangent vector reusing a precomputed base sweep -- the bare ``𝒥``
    (entries) with the base edge variables injected. Equivalent to :py:func:`entries_tangent`, but takes
    the lean ``(xis, mus)`` from ``base_sweep``; only the fiber-sliced ``dxis`` is computed here. No
    gauge projector ``Π``.'''
    var_tucker_cores, var_tt_cores = variation
    _, down_tt_cores, _, right_tt_cores = base
    xis, mus = base_sweep
    dxis = _entry_xis(var_tucker_cores, index)    # fiber slice; variation-dependent, not in the base sweep
    return _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores)


def entries_tangent(
        index:      NDArray,                # int, shape=(d,)+W (index stack W)
        variation:  typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (var_tucker, var_tt)
        base:       typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                              typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (up, down, left, right)
) -> NDArray:                               # entries of the dense tangent at ``index``; shape = W + K + C
    '''Extract entries of the dense tangent at ``index`` (= apply with unit vectors, by slicing).

    Identical to :py:func:`apply_tangent` except the up-index edge variables come from slicing the
    Tucker-core fibers (``U_i[..., index_i]`` and ``dU_i[..., index_i]``) rather than contracting with
    vectors -- so there is no contraction with unit basis vectors and no ``N`` factor.

    See Also
    --------
    apply_tangent
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    var_tucker_cores, var_tt_cores = variation

    xis  = _entry_xis(up_tucker_cores, index)     # xi-hat_i  = U_i[..., index_i]   (fiber slice)
    dxis = _entry_xis(var_tucker_cores, index)    # delta-xi_i = dU_i[..., index_i] (fiber slice)
    mus  = compute_mus(left_tt_cores, xis)

    return _apply_from_xis(xis, dxis, mus, right_tt_cores, down_tt_cores, var_tt_cores)


def _onehot_vectors(index, up_tucker_cores):
    '''Unit vectors e_{index_k} (shape W + (Nk,)) -- the "apply vectors" whose adjoint is the entry
    scatter, so that entries_tangent_transpose is apply_tangent_transpose with these one-hot vectors.'''
    use_jax = tree_contains_jax((index, up_tucker_cores))
    xnp, _, _ = get_backend(False, use_jax)
    index = xnp.array(index)
    return tuple(xnp.eye(B.shape[-1])[index[i]] for i, B in enumerate(up_tucker_cores))


def compute_sigma_hats(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xis:            typ.Sequence[NDArray],  # base up-index edge vars, len=d, elm_shape=W+C+(nUi,)
        c:              NDArray,                # residual (scalar), shape=W+K+C
) -> typ.Sequence[NDArray]:                     # sigma_hats. len=d, elm_shape=W+K+C+(rR(i+1),)
    '''Propagation-only adjoint **reverse** sweep via ``Q``, seeded at the terminal bond by the residual
    ``c`` -- the order-0 (non-jet) analog of :py:func:`t3toolbox.backend.probe_derivatives.compute_sigma_hat_jets`.

    The right context the apply/entries transpose needs, **recomputed** from ``c`` rather than stored:
    this is the low-memory half of the adjoint-state method (no ``nu``/``eta`` precomputed). ``sigma_hats[i]``
    is the adjoint of the after-core-``i`` carry; it carries the tangent stack ``K`` (from ``c``).
    Right-to-left via ``reverse_tt`` (mirroring the forward ``nu`` sweep's reversal).'''
    use_jax = tree_contains_jax((right_tt_cores, xis, c))
    is_uniform = is_ndarray(right_tt_cores)
    reverse = uniform_ops.reverse_utt if is_uniform else ragged_ops.reverse_tt
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    rev_Q = reverse(right_tt_cores)
    rev_xi = xis[::-1]
    # The forward sums the terminal bond (rR_d, not necessarily 1 -- e.g. the corewise base's own cores),
    # so the adjoint BROADCASTS c over it: seed = c (x) 1_{rR_d} -> W+K+C+(rR_d,).
    rR_d = right_tt_cores[-1].shape[-1]
    seed = xnp.broadcast_to(c[..., None], tuple(c.shape) + (rR_d,))

    def _step(carry, data):
        Q, xi = data
        return contractions.WKCa_Caib_WCi_to_WKCb(carry, Q, xi), (carry,)

    _, (rev_sigma_hats,) = xscan(_step, seed, (rev_Q, rev_xi))
    return rev_sigma_hats[::-1]


def _apply_transpose_adjoint(c, ww, xis, mus, down_tt_cores, right_tt_cores, sum_over_probes):
    '''Adjoint-state assembly shared by apply/entries_transpose_from_sweep -- the **low-memory,
    K-aware** transpose (replaces the old scatter). The scalar residual ``c`` seeds one reverse
    ``sigma_hat`` sweep (recomputing the right context, so ``nu``/``eta`` are never stored); then

        dxi-hat_k  = mu-hat_{k-1} . O_k . sigma-hat_k                 # over the down mode nO
        dG-tilde_k = mu-hat_{k-1} (x) xi-hat_k (x) sigma-hat_k        # over (rL, nU, rR)
        dU-tilde_k = dxi-hat_k (x) w_k                               # over (nO, N)

    Uses only the lean base sweep ``(xis, mus)`` (half the memory of the ``(xis, mus, nus, etas)`` the
    scatter stored -- the ``W``-scaling ``nu``/``eta`` are gone), at the cost of the ``sigma_hat`` sweep
    per transpose. Full ``W + K + C``: the residual ``c`` may carry the tangent stack ``K`` (the output
    space of a ``K``-stacked forward ``apply_tangent``), which rides into the variation gradient -- the
    capability the scatter lacked. ``sum_over_probes=True`` sums the probe stack ``W`` (the ``J^T r``
    back-projection); ``False`` keeps it as the output tangent stack. ``K``/``C`` always kept.'''
    n_probe = ww[0].ndim - 1
    sigma_hats = compute_sigma_hats(right_tt_cores, xis, c)
    dxi_hats = tuple(contractions.WCa_Caib_WKCb_to_WKCi(mu, O, sh)        # dxi_hat = mu . O . sigma_hat
                     for mu, O, sh in zip(mus, down_tt_cores, sigma_hats))
    if sum_over_probes:
        dG_tildes = tuple(contractions.WCa_WCi_WKCb_to_KCaib(mu, xi, sh, n_probe)
                          for mu, xi, sh in zip(mus, xis, sigma_hats))
        dU_tildes = tuple(contractions.Wo_WKCa_to_KCao(w, dxh) for w, dxh in zip(ww, dxi_hats))
    else:
        dG_tildes = tuple(contractions.WCa_WCi_WKCb_to_WKCaib(mu, xi, sh, n_probe)
                          for mu, xi, sh in zip(mus, xis, sigma_hats))
        dU_tildes = tuple(contractions.Wo_WKCa_to_WKCao(w, dxh) for w, dxh in zip(ww, dxi_hats))
    return dU_tildes, dG_tildes   # (var_tucker, var_tt) = T3Variations.data


def apply_transpose_from_sweep(
        c:          NDArray,                # residual, shape = W + K + C (K optional)
        ww:         typ.Sequence[NDArray],  # apply vectors (one-hot e_index for entries), len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d  (unused; uniform call signature)
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d  (unused)
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # base order = T3Basis.data = (up, down, left, right)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray],          # xis
            typ.Sequence[NDArray],          # mus
        ],                                  # = precompute_apply_base_sweep(base, ww)  (lean: no nu/eta)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes) = T3Variations.data
    '''Transpose of the all-modes apply reusing a precomputed base sweep -- the bare ``𝒥ᵀ``, by the
    **adjoint-state** method (the scalar residual ``c`` seeds one reverse ``sigma_hat`` sweep; see
    :py:func:`_apply_transpose_adjoint`). Takes the **lean** ``(xis, mus)`` sweep + the base cores ``O,
    Q`` (it recomputes the right context rather than storing ``nu``/``eta`` -- half the memory). Reuse
    hook for ``fitting.py`` (one base sweep feeds the forward and this transpose). Full ``W + K + C``
    (the residual ``c`` may carry the tangent stack ``K``). No gauge projector ``Π``.

    See Also
    --------
    precompute_apply_base_sweep
    apply_tangent_transpose
    '''
    _, down_tt_cores, _, right_tt_cores = base
    xis, mus = base_sweep
    return _apply_transpose_adjoint(c, ww, xis, mus, down_tt_cores, right_tt_cores, sum_over_probes)


def apply_tangent_transpose(
        c:          NDArray,                # residual, shape = W + C (one per probe-set, per base point)
        ww:         typ.Sequence[NDArray],  # the apply vectors, len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                              typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (up, down, left, right)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes) = T3Variations.data
    '''Apply the transpose of :py:func:`apply_tangent` -- back-project a residual ``c`` into a tangent.

    The adjoint of the (linear-in-the-variation) all-modes apply. Needs only the base sweep
    (xi-hat, mu-hat, nu-hat, eta-hat) and a single-term scatter assembly (it skips the adjoint
    perturbation sweep that probe_tangent_transpose runs). With ``sum_over_probes=False`` the probe
    stack W becomes the output tangent stack; with ``True`` it is summed (the ``J^T r`` back-projection).

    See Also
    --------
    apply_tangent
    entries_tangent_transpose
    '''
    base_sweep = precompute_apply_base_sweep(base, ww)
    return apply_transpose_from_sweep(c, ww, base, base_sweep, sum_over_probes)


def entries_transpose_from_sweep(
        c:          NDArray,                # residual, shape = W + C (or W + K + C)
        index:      NDArray,                # int, shape=(d,)+W -- the indices c weights (-> one-hot vectors)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q); uses U (one-hot), O, Q
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = precompute_entries_base_sweep(base, index)  (lean: no nu/eta)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes) = T3Variations.data
    '''Transpose of the all-modes entries reusing a precomputed base sweep -- the bare ``𝒥ᵀ`` (entries),
    by the **adjoint-state** method (see :py:func:`_apply_transpose_adjoint`). Takes the **lean**
    ``(xis, mus)`` sweep (the reuse hook for ``fitting.py``). Identical to
    :py:func:`apply_transpose_from_sweep` with the one-hot vectors ``e_{index}`` as the apply vectors.
    Full ``W + K + C``; no gauge projector ``Π``.'''
    up_tucker_cores, down_tt_cores, _, right_tt_cores = base
    ww = _onehot_vectors(index, up_tucker_cores)
    xis, mus = base_sweep
    return _apply_transpose_adjoint(c, ww, xis, mus, down_tt_cores, right_tt_cores, sum_over_probes)


def entries_tangent_transpose(
        c:          NDArray,                # residual, shape = W + C
        index:      NDArray,                # int, shape=(d,)+W (the indices whose entries c weights)
        base:       typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray],
                              typ.Sequence[NDArray], typ.Sequence[NDArray]],   # (up, down, left, right)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes)
    '''Apply the transpose of :py:func:`entries_tangent` -- scatter a residual ``c`` at ``index`` into a tangent.

    Identical to :py:func:`apply_tangent_transpose` with the up-index ``xi-hat`` from fiber slicing and
    the apply vectors replaced by the unit vectors ``e_{index_k}`` (so the ``dU-tilde`` outer product
    *is* the entry scatter).

    See Also
    --------
    entries_tangent
    apply_tangent_transpose
    '''
    base_sweep = precompute_entries_base_sweep(base, index)
    return entries_transpose_from_sweep(c, index, base, base_sweep, sum_over_probes)


def apply_corewise_transpose(
        c:          NDArray,                # residual, shape=W+C
        ww:         typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
        core_pair:  typ.Tuple[
            typ.Sequence[NDArray],          # tucker_cores, len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],          # tt_cores,     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],
        sum_over_probes: bool = False,      # True: sum the apply stack W (the gradient J^T r)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker-core gradients, same shapes as tucker_cores
    typ.Tuple[NDArray, ...],  # tt-core gradients,     same shapes as tt_cores
]:
    '''Corewise (non-manifold) transpose of :py:func:`tucker_tensor_train_apply`: gradient of the
    measurement w.r.t. the cores of the base ``core_pair``, treated as independent variables.

    The adjoint of the *core parametrization* ``cores -> apply(X(cores), ww)`` at the base point -- the
    gradient a core-wise optimizer (Adam, L-BFGS) needs. Returns gradients shaped exactly like
    ``(tucker_cores, tt_cores)`` -- a gradient, NOT a tensor (so no ``|W|`` blow-up: the apply stack
    collapses into the fixed-size cores). Distinct from the *ambient* transpose (a free CP tensor) and
    the *tangent* transpose (a Riemannian tangent); see ``docs/transposes.md``.

    Implemented by the Section 6.3 ("corewise simplification") substitution into the tangent transpose:
    feed the base's own cores in place of the orthogonal frames (``P, Q, O -> G_i``), with ``U_i`` no
    longer required orthogonal -- i.e. :py:func:`apply_tangent_transpose` at base ``(U, G, G, G)``. No
    orthogonality is required. ``sum_over_probes=True`` sums the apply stack ``W`` (the gradient
    ``J^T r``); ``False`` keeps ``W`` as a stack (one core-gradient set per probe).

    Math reference: Section 6.3, Alger et al. (2026), "Tucker Tensor Train Taylor Series"
    (arXiv:2603.21141).
    '''
    tucker_cores, tt_cores = core_pair
    return apply_tangent_transpose(
        c, ww, (tucker_cores, tt_cores, tt_cores, tt_cores), sum_over_probes=sum_over_probes,
    )


def entries_corewise_transpose(
        c:          NDArray,                # residual, shape=W+C
        index:      NDArray,                # int, shape=(d,)+W
        core_pair:  typ.Tuple[
            typ.Sequence[NDArray],          # tucker_cores, len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],          # tt_cores,     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],
        sum_over_probes: bool = False,      # True: sum the apply stack W (scatter-adds collisions)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker-core gradients, same shapes as tucker_cores
    typ.Tuple[NDArray, ...],  # tt-core gradients,     same shapes as tt_cores
]:
    '''Corewise (non-manifold) transpose of :py:func:`tucker_tensor_train_entries`: gradient of the
    sampled entries w.r.t. the base's cores.

    The ``entries`` counterpart of :py:func:`apply_corewise_transpose` -- the Section 6.3 substitution
    into :py:func:`entries_tangent_transpose`. Needs no ambient ``shape`` argument: the dims come from
    the base ``tucker_cores``. ``sum_over_probes=True`` scatter-adds colliding indices (the gradient
    ``J^T r``).
    '''
    tucker_cores, tt_cores = core_pair
    return entries_tangent_transpose(
        c, index, (tucker_cores, tt_cores, tt_cores, tt_cores), sum_over_probes=sum_over_probes,
    )


###############################################################
###########    Transpose of tangent to probes map    ##########
###############################################################

def compute_deta_tildes(
        up_tucker_cores:    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(nUi,Ni)
        ztildes:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,Ni)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,nUi)
    '''Adjoint-var-upward edge variables deta_tilde.
    Used for computing the transpose of the map from a tangent vector to its probes.

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((up_tucker_cores, ztildes))
    is_uniform = is_ndarray(up_tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        # ztildes carry no separate T3 stack C in the uniform layer, so the outer-product form
        # (same as compute_xis) coincides with the C-batched contraction.
        deta_tildes = contractions.dCio_dWo_to_dWCi(up_tucker_cores, ztildes)
    else:
        def _func(x):
            U, zt = x
            # C (T3 stack) is shared between the core U and the residual zt; W is the probe stack on
            # zt. This is NOT compute_xis (which forms an outer product over the two stacks).
            return (contractions.WCo_Cio_to_WCi(zt, U),)

        (deta_tildes,) = xmap(_func, (up_tucker_cores, ztildes))

    return deta_tildes


def compute_tau_tildes(
        deta_tildes:        typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,nUi)
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rL(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi)
        mus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rLi)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,rLi)
    '''Adjoint-var-rightward edge variables tau_tilde.
    Used for computing the transpose of the map from a tangent vector to its probes.

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((deta_tildes, left_tt_cores, xis, mus))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    def _func(tau_tilde, x):
        P, xi, deta_tilde, mu = x
        # Three-group (W probe, K tangent, C base): tau_tilde/deta_tilde carry K (from the residual),
        # xi/mu and the base core P do not. Both terms self-infer the split (P pins C, xi/mu pin W);
        # reduces to the two-group result when K is empty (no K-stacked residual).
        t1 = contractions.WKCa_Caib_WCi_to_WKCb(tau_tilde, P, xi)
        t2 = contractions.WCa_Caib_WKCi_to_WKCb(mu, P, deta_tilde)
        tau_tilde_next = t1 + t2
        return tau_tilde_next, (tau_tilde,)

    # carry tau_tilde is W+K+C; take the leading stack from deta_tildes (carries K), not mus (W+C).
    init = xnp.zeros(deta_tildes[0].shape[:-1] + (left_tt_cores[0].shape[-3],))

    last_tau_tilde, (tau_tildes,) = xscan(_func, init, (left_tt_cores, xis, deta_tildes, mus))
    return tau_tildes


def compute_sigma_tildes(
        deta_tildes:        typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi)
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi)
        nus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,rR(i+1))
    '''Adjoint-var-leftward edge variables sigma_tilde.
    Used for computing the transpose of the map from a tangent vector to its probes.

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    is_uniform = is_ndarray(deta_tildes)
    reverse = uniform_ops.reverse_utt if is_uniform else ragged_ops.reverse_tt

    return compute_tau_tildes(
        deta_tildes[::-1], reverse(right_tt_cores), xis[::-1], nus[::-1],
    )[::-1]


def compute_dxi_tildes(
        sigma_tildes:           typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rR(i+1))
        tau_tildes:             typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rLi)
        down_tt_cores:         typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        mus:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rLi)
        nus:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rR(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # dxi_tildes. len=d, elm_shape=(...,nOi)
    '''Adjoint-var-downward edge variables dxi_tilde.
    Used for computing the transpose of the map from a tangent vector to its probes.

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((sigma_tildes, tau_tildes, down_tt_cores, mus, nus))
    is_uniform = not isinstance(mus, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        term1 = xnp.einsum(
            'd...aj,d...j->d...a',
            xnp.einsum('d...i,diaj->d...aj', tau_tildes, down_tt_cores),
            nus,
        )
        term2 = xnp.einsum(
            'd...aj,d...j->d...a',
            xnp.einsum('d...i,diaj->d...aj', mus, down_tt_cores),
            sigma_tildes,
        )
        dxi_tildes = term1 + term2
    else:
        def _func(x):
            O, mu, nu, st, tt = x
            # Three-group (see compute_tau_tildes): tt/st carry K, mu/nu and the base core O do not.
            # Both terms self-infer (O pins C, mu/nu pin W).
            term1 = contractions.WKCa_Caib_WCb_to_WKCi(tt, O, nu)
            term2 = contractions.WCa_Caib_WKCb_to_WKCi(mu, O, st)
            return (term1 + term2,)

        xs = (down_tt_cores, mus, nus, sigma_tildes, tau_tildes)
        (dxi_tildes,) = xmap(_func, xs)

    return dxi_tildes


def assemble_tucker_variations(
        ztildes:    typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,Ni)
        dxi_tildes: typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nOi)
        ww:         typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,Ni)
        etas:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nOi)
        sum_over_probes: bool = False,
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # dU_tildes. len=d, elm_shape=(...,nOi,Ni)
    '''Assemble Tucker core variations, delta_U_tilde.
    Used for computing the transpose of the map from a tangent vector to its probes.

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((ztildes, dxi_tildes, ww, etas))
    is_uniform = not isinstance(ww, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        if sum_over_probes:
            dU_tildes = (
                    xnp.einsum('d...o,d...a->dao', ztildes, etas)
                    +
                    xnp.einsum('d...o,d...a->dao', ww, dxi_tildes)
            )
        else:
            dU_tildes = (
                    xnp.einsum('d...o,d...a->d...ao', ztildes, etas)
                    +
                    xnp.einsum('d...o,d...a->d...ao', ww, dxi_tildes)
            )
    else:
        def _func(x):
            z_tilde, eta, w, dxi_tilde = x
            # Three-group (W probe, K tangent, C base): z_tilde/dxi_tilde carry K, eta does not, w is
            # W-only. n_probe = len(W) is recovered locally from the W-only probe vector w (the
            # z_tilde (x) eta term needs it; the w (x) dxi_tilde term self-infers W from w). Output
            # keeps K always; W is summed (K+C) or kept (W+K+C) per sum_over_probes. Reduces to the
            # two-group result when K is empty.
            n_probe = w.ndim - 1
            if sum_over_probes:
                dU_tilde = (
                        contractions.WKCo_WCa_to_KCao(z_tilde, eta, n_probe)
                        +
                        contractions.Wo_WKCa_to_KCao(w, dxi_tilde)
                )
            else:
                dU_tilde = (
                        contractions.WKCo_WCa_to_WKCao(z_tilde, eta, n_probe)
                        +
                        contractions.Wo_WKCa_to_WKCao(w, dxi_tilde)
                )
            return (dU_tilde,)

        (dU_tildes,) = xmap(_func, (ztildes, etas, ww, dxi_tildes))

    return dU_tildes


def assemble_tt_variations(
        sigma_tildes:   typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rR(i+1))
        tau_tildes:     typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rLi)
        deta_tildes:    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,nUi)
        xis:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,nUi)
        mus:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rLi)
        nus:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rR(i+1))
        sum_over_probes: bool = False,
        n_probe: int = 0,  # number of trailing probe-stack axes; only used when sum_over_probes
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # dG_tildes. len=d, elm_shape=(...,rLi,nUi,rRi)
    '''Assemble TT core variations, delta_G_tilde.
    Used for computing the transpose of the map from a tangent vector to its probes.

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((sigma_tildes, tau_tildes, deta_tildes, xis, mus, nus))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        if sum_over_probes:
            dG_tildes = (
                    xnp.einsum(
                        'd...ia,d...j->diaj',
                        xnp.einsum('d...i,d...a->d...ia', mus, xis),
                        sigma_tildes
                    )
                    +
                    xnp.einsum(
                        'd...ia,d...j->diaj',
                        xnp.einsum('d...i,d...a->d...ia', tau_tildes, xis),
                        nus
                    )
                    +
                    xnp.einsum(
                        'd...ia,d...j->diaj',
                        xnp.einsum('d...i,d...a->d...ia', mus, deta_tildes),
                        nus
                    )
            )
        else:
            dG_tildes = (
                    xnp.einsum(
                        'd...ia,d...j->d...iaj',
                        xnp.einsum('d...i,d...a->d...ia', mus, xis),
                        sigma_tildes
                    )
                    +
                    xnp.einsum(
                        'd...ia,d...j->d...iaj',
                        xnp.einsum('d...i,d...a->d...ia', tau_tildes, xis),
                        nus
                    )
                    +
                    xnp.einsum(
                        'd...ia,d...j->d...iaj',
                        xnp.einsum('d...i,d...a->d...ia', mus, deta_tildes),
                        nus
                    )
            )
    else:
        def _func(x):
            xi, mu, nu, sigma_tilde, tau_tilde, deta_tilde = x
            # Three-group (W probe, K tangent, C base): the residual-derived edge vars sigma_tilde /
            # tau_tilde / deta_tilde carry K (on the j / i / a leg respectively), the base edge vars
            # xi/mu/nu do not. No operand here is W-only or C-only, so len(W)=n_probe is supplied;
            # each contraction then derives C from an W+C operand and K from the W+K+C one. Output
            # keeps K always; W is summed (K+C) or kept (W+K+C) per sum_over_probes.
            if sum_over_probes:
                dG_tilde = (
                        contractions.WCi_WCa_WKCj_to_KCiaj(mu, xi, sigma_tilde, n_probe)
                        +
                        contractions.WKCi_WCa_WCj_to_KCiaj(tau_tilde, xi, nu, n_probe)
                        +
                        contractions.WCi_WKCa_WCj_to_KCiaj(mu, deta_tilde, nu, n_probe)
                )
            else:
                dG_tilde = (
                        contractions.WCi_WCa_WKCj_to_WKCiaj(mu, xi, sigma_tilde, n_probe)
                        +
                        contractions.WKCi_WCa_WCj_to_WKCiaj(tau_tilde, xi, nu, n_probe)
                        +
                        contractions.WCi_WKCa_WCj_to_WKCiaj(mu, deta_tilde, nu, n_probe)
                )
            return (dG_tilde,)

        xs = (xis, mus, nus, sigma_tildes, tau_tildes, deta_tildes)
        (dG_tildes,) = xmap(_func, xs)

    return dG_tildes


def probe_transpose_from_sweep(
        ztildes:    typ.Sequence[NDArray],  # probe residuals, len=d, elm_shape=W+K+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors,   len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        base_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = precompute_probe_base_sweep(base, ww)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes) = T3Variations.data
    '''Transpose of the probe reusing a precomputed base sweep -- the bare ``𝒥ᵀ`` (probe) with the base
    edge variables injected. Equivalent to :py:func:`probe_tangent_transpose`, but takes
    ``(xis, mus, nus, etas)`` from ``base_sweep`` (the reuse hook for ``fitting.py``; apply & probe share
    the sweep). No gauge projector ``Π``.

    See Also
    --------
    precompute_probe_base_sweep
    probe_tangent_transpose
    probe_jacobian_from_sweep
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    xis, mus, nus, etas = base_sweep
    deta_tildes  = compute_deta_tildes(up_tucker_cores, ztildes)
    tau_tildes   = compute_tau_tildes(deta_tildes, left_tt_cores, xis, mus)
    sigma_tildes = compute_sigma_tildes(deta_tildes, right_tt_cores, xis, nus)
    dxi_tildes   = compute_dxi_tildes(sigma_tildes, tau_tildes, down_tt_cores, mus, nus)
    n_probe = ww[0].ndim - 1
    dU_tildes = assemble_tucker_variations(ztildes, dxi_tildes, ww, etas, sum_over_probes=sum_over_probes)
    dG_tildes = assemble_tt_variations(sigma_tildes, tau_tildes, deta_tildes, xis, mus, nus,
                                       sum_over_probes=sum_over_probes, n_probe=n_probe)
    return dU_tildes, dG_tildes


def probe_tangent_transpose(
        ztildes:        typ.Union[typ.Sequence[NDArray],    NDArray], # len=d, elm_shape=(...,Ni)
        ww:             typ.Union[typ.Sequence[NDArray],    NDArray], # input vectors, len=d, elm_shape=(...,Ni)
        base:           typ.Union[
            typ.Tuple[
                typ.Sequence[NDArray],  # up_tucker_cores. len=d. U_xo U_yo   = I_xy, U.shape = (nU, N)
                typ.Sequence[NDArray],  # down_tt_cores.   len=d. O_ixj O_iyj = I_xy  O.shape = (rL, nO, rR)
                typ.Sequence[NDArray],  # left_tt_cores.   len=d. P_iax P_iay = I_xy, P.shape = (rL, nU, rR)
                typ.Sequence[NDArray],  # right_tt_cores.  len=d. Q_xaj Q_yaj = I_xy  Q.shape = (rL, nU, rR)
            ],
            typ.Tuple[
                NDArray,  # up_tucker_supercore. shape=(d, nU, N),      up orthogonal elements
                NDArray,  # down_tt_supercore.   shape=(d, rL, nO, rR), down orthogonal elements
                NDArray,  # left_tt_supercore.   shape=(d, rL, nU, rR), left orthogonal elements
                NDArray,  # right_tt_supercore.  shape=(d, rL, nU, rR), right orthogonal elements
            ],
        ], # base order = T3Basis.data = (up, down, left, right) = (U, O, P, Q)
        sum_over_probes: bool = False,
) -> typ.Union[
    typ.Tuple[
        typ.Tuple[NDArray,...], # dU_tildes. len=d, elm_shape=(...,nOi,Ni)
        typ.Tuple[NDArray,...], # dG_tildes. len=d, elm_shape=(...,rLi,nUi,rRi)
    ],
    typ.Tuple[
        NDArray,  # dU_tildes. shape=(d, ..., nOi, Ni)
        NDArray,  # dG_tildes. shape=(d, ..., rLi, nUi, rRi)
    ],
]:
    '''Apply the transpose of the map from a tangent vector to its probes (apply (J^(s))^T to ztildes).

    Stacking (handled by the W/C custom contractions in ``contractions.py``): the residuals
    ``ztildes`` live in the forward probe space, ``elm_shape = W + C + (Ni,)`` (probe stack W
    outermost, T3 stack C innermost -- base-inner), while ``ww`` carries only the probe stack W. With
    ``sum_over_probes=False`` the resulting variations keep both stacks (``W + C + ...``, the probe
    stack W becoming the tangent stack K); with ``sum_over_probes=True`` the probe stack W is summed
    and the T3 stack C is kept (``C + ...``).

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    ztildes: typ.Sequence[NDArray]
        Probe residuals to apply the transpose map to. len=d, elm_shape=(...,Ni)
    ww: typ.Sequence[NDArray]
        input vectors that defined the (forward) probe map. len=d, elm_shape=(...,Ni)
    base: (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
        Orthogonal base for the point where the tangent space attaches to the manifold. This is
        exactly ``T3Basis.data`` order (U, O, P, Q) -- pass ``basis.data`` directly, no reorder.
    sum_over_probes: bool
        Sum results over all probe residuals, rather than returning results for each probe residual.

    Returns
    -------
    (dU_tildes, dG_tildes)
        Tangent variations (a bvf.T3Variations.data tuple) resulting from applying the transpose map.

    See Also
    --------
    probe_t3
    probe_tangent

    Examples
    --------

    Adjoint identity ``<z, J v> = <J^T z, v>`` with one set of probing vectors:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
    >>> base, _ = bvf.t3_orthogonal_representations(x)
    >>> probe_base = base.data  # probing's base order == T3Basis.data, no reorder
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> v = t3m.MANIFOLD.randn(base)
    >>> z = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> Jv  = t3p.probe_tangent(ww, v.variations.data, probe_base)
    >>> JTz = t3p.probe_tangent_transpose(z, ww, probe_base)   # (dU_tildes, dG_tildes)
    >>> lhs = cw.corewise_dot(z, Jv)                  # <z, J v>
    >>> rhs = cw.corewise_dot(JTz, v.variations.data)  # <J^T z, v>
    >>> print(bool(np.allclose(lhs, rhs)))
    True

    With ``sum_over_probes=True`` (the Gauss-Newton ``J^T r``), the adjoint identity still holds
    when a probe stack ``W`` is summed on both sides:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
    >>> base, _ = bvf.t3_orthogonal_representations(x)
    >>> ww = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))  # W=(2,)
    >>> v = t3m.MANIFOLD.randn(base)
    >>> z = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
    >>> Jv  = t3p.probe_tangent(ww, v.variations.data, base.data)        # W-stacked probes
    >>> JTz = t3p.probe_tangent_transpose(z, ww, base.data, sum_over_probes=True)
    >>> lhs = cw.corewise_dot(z, Jv)                  # sum_W <z_W, (J v)_W>
    >>> rhs = cw.corewise_dot(JTz, v.variations.data)  # <sum_W J^T z_W, v>
    >>> print(bool(np.allclose(lhs, rhs)))
    True
    '''
    (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores) = base

    xis = compute_xis(up_tucker_cores, ww)

    mus = compute_mus(left_tt_cores, xis)

    nus = compute_nus(right_tt_cores, xis)

    etas = compute_etas(down_tt_cores, mus, nus)

    #

    deta_tildes = compute_deta_tildes(up_tucker_cores, ztildes)

    tau_tildes = compute_tau_tildes(deta_tildes, left_tt_cores, xis, mus)

    sigma_tildes = compute_sigma_tildes(deta_tildes, right_tt_cores, xis, nus)

    dxi_tildes = compute_dxi_tildes(sigma_tildes, tau_tildes, down_tt_cores, mus, nus)

    #

    # Number of trailing probe-stack (W) axes. Needed by the tt-assemble, whose operands are all W+C
    # or W+K+C (no W-only operand to recover it from); the tucker-assemble recovers it itself from the
    # W-only probe vectors ww. (ragged: ww[0].shape = W + (Ni,).)
    n_probe = ww[0].ndim - 1

    dU_tildes = assemble_tucker_variations(
        ztildes, dxi_tildes, ww, etas,
        sum_over_probes=sum_over_probes,
    )

    dG_tildes = assemble_tt_variations(
        sigma_tildes, tau_tildes, deta_tildes, xis, mus, nus,
        sum_over_probes=sum_over_probes, n_probe=n_probe,
    )

    return dU_tildes, dG_tildes


def probe_ambient_transpose(
        ztildes:    typ.Sequence[NDArray],  # probe residuals, len=d, elm_shape=W+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors,   len=d, elm_shape=W+(Ni,)
        sum_over_probes: bool = False,      # True: W folds into the CP rank
) -> typ.Sequence[NDArray]:  # canonical (CP) factors. len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`probe_t3`: back-project probe residuals into CP factors.

    The *ambient* adjoint -- the transpose of ``probe`` as a linear map on the **full tensor space**.
    Probe returns ``d`` vectors (one free mode each), so the residual ``ztildes`` is ``d`` vectors; the
    back-projection is the rank-``d`` tensor

        sum_i  w0 (x) ... (x) w_{i-1} (x) ztildes_i (x) w_{i+1} (x) ... (x) w_{d-1}

    (term ``i`` has the residual ``ztildes_i`` in slot ``i`` and the probe vectors elsewhere), whose
    natural representation is a **canonical (CP) decomposition** of rank ``d``. Base-free. Distinct from
    the *corewise* transpose (gradient w.r.t. a base's cores) and the *tangent* transpose (Riemannian
    gradient); see ``docs/transposes.md``. The ``apply``/``entries`` analog is the rank-1 (or rank-|W|)
    :py:func:`tucker_tensor_train_apply_ambient_transpose`.

    - ``sum_over_probes=False`` (primary): ``W`` is a passthrough stacking axis -- a ``W (+ C)`` stack
      of rank-``d`` CP tensors.
    - ``sum_over_probes=True``: ``W`` folds into the CP rank -- one rank-``d|W|`` CP tensor
      ``sum_W sum_i (...)``. Cheap as CP (``O(d |W| N)``).

    Returns CP ``factors`` (factor ``k`` has the diagonal structure: rank slot ``k`` = ``ztildes_k``,
    the others = ``ww_k``), in the layout :py:func:`t3_operations.t3_from_canonical` consumes.
    '''
    use_jax = tree_contains_jax((ztildes, ww))
    xnp, _, _ = get_backend(False, use_jax)
    d = len(ww)
    nW = ww[0].ndim - 1
    W  = ww[0].shape[:nW]                 # probe stack
    C  = ztildes[0].shape[nW:-1]          # base stack (ztildes[i] is W + C + (Ni,))
    nC = len(C)
    mW = math.prod(W)

    factors = []
    for k in range(d):
        Nk = ww[k].shape[-1]
        w_bc = xnp.broadcast_to(ww[k].reshape(W + (1,) * nC + (1, Nk)), W + C + (d, Nk))
        diag = (xnp.arange(d) == k).reshape((d, 1))            # rank slot k is the "diagonal"
        Fk = xnp.where(diag, ztildes[k][..., None, :], w_bc)   # W + C + (d, Nk)
        if sum_over_probes:                                    # fold W into the CP rank: rank d -> d|W|
            Fk = xnp.moveaxis(Fk, tuple(range(nW)), tuple(range(nC, nC + nW)))  # C + W + (d, Nk)
            Fk = Fk.reshape(C + (mW * d, Nk))
        factors.append(Fk)
    return tuple(factors)


def probe_corewise_transpose(
        ztildes:    typ.Sequence[NDArray],  # probe residuals, len=d, elm_shape=W+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors,   len=d, elm_shape=W+(Ni,)
        core_pair:  typ.Tuple[
            typ.Sequence[NDArray],          # tucker_cores, len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],          # tt_cores,     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],
        sum_over_probes: bool = False,      # True: sum the probe stack W (the gradient J^T r)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker-core gradients, same shapes as tucker_cores
    typ.Tuple[NDArray, ...],  # tt-core gradients,     same shapes as tt_cores
]:
    '''Corewise (non-manifold) transpose of :py:func:`probe_t3`: gradient of the probes w.r.t. the
    cores of the base ``core_pair``, treated as independent variables.

    The probe analog of :py:func:`apply_corewise_transpose` -- the Section 6.3 substitution
    ``P, Q, O -> G_i`` (``U`` non-orthogonal) into :py:func:`probe_tangent_transpose`, i.e. that
    transpose at base ``(U, G, G, G)``. Returns gradients shaped exactly like ``(tucker_cores,
    tt_cores)`` (a gradient, not a tensor; no ``|W|`` blow-up). For non-manifold optimizers (Adam,
    L-BFGS) fitting from probes. ``sum_over_probes=True`` sums the probe stack ``W`` (the gradient
    ``J^T r``); ``False`` keeps it. Math reference: Section 6.3, Alger et al. (2026) (arXiv:2603.21141).
    '''
    tucker_cores, tt_cores = core_pair
    return probe_tangent_transpose(
        ztildes, ww, (tucker_cores, tt_cores, tt_cores, tt_cores), sum_over_probes=sum_over_probes,
    )


###############################################
##########    Probe dense tensor    ###########
###############################################

def probe_dense(
        vectors: typ.Sequence[NDArray],
        T: NDArray,
) -> typ.Tuple[NDArray]:
    """Probe a dense tensor.

    Parameters
    ----------
    T: NDArray
        Tensor to be probed. shape=C+(N0,...,N(d-1))
    vectors: typ.Sequence[NDArray]
        Probing input vectors.
        len=d.
        elm_shape=W+(Ni,)

    Returns
    -------
    typ.Tuple[NDArray]
        Probes.
        len=d.
        elm_shape=(Ni,) or elm_shape=W+C+(Ni,)

    Examples
    --------

    Probe with one set of vectors; value-match each mode against a hand-written einsum:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> u0, u1, u2 = np.random.randn(10), np.random.randn(11), np.random.randn(12)
    >>> yy = t3p.probe_dense((u0, u1, u2), T)
    >>> y0 = np.einsum('ijk,j,k', T, u1, u2)   # contract all modes but 0
    >>> y1 = np.einsum('ijk,i,k', T, u0, u2)
    >>> y2 = np.einsum('ijk,i,j', T, u0, u1)
    >>> print([y.shape for y in yy])           # one probe per mode, elm_shape=(Ni,)
    [(10,), (11,), (12,)]
    >>> print([bool(np.allclose(y, ref)) for y, ref in zip(yy, (y0, y1, y2))])
    [True, True, True]

    Vectorize over probing vectors: a probe stack ``W`` rides through, ``elm_shape = W + (Ni,)``:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> u0, u1, u2 = np.random.randn(2, 3, 10), np.random.randn(2, 3, 11), np.random.randn(2, 3, 12)
    >>> yy = t3p.probe_dense((u0, u1, u2), T)
    >>> y0 = np.einsum('ijk,uvj,uvk->uvi', T, u1, u2)
    >>> y1 = np.einsum('ijk,uvi,uvk->uvj', T, u0, u2)
    >>> y2 = np.einsum('ijk,uvi,uvj->uvk', T, u0, u1)
    >>> print(yy[0].shape)                      # W=(2,3) outer, then N0=10
    (2, 3, 10)
    >>> print([bool(np.allclose(y, ref)) for y, ref in zip(yy, (y0, y1, y2))])
    [True, True, True]

    Vectorize over probing vectors AND a stacked (big) tensor: base-inner ``elm_shape = W + C + (Ni,)``:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> T = np.random.randn(4, 5, 6, 10, 11, 12)   # C=(4,5,6) stack on the tensor
    >>> u0, u1, u2 = np.random.randn(2, 3, 10), np.random.randn(2, 3, 11), np.random.randn(2, 3, 12)
    >>> yy = t3p.probe_dense((u0, u1, u2), T)
    >>> y0 = np.einsum('xyzijk,uvj,uvk->uvxyzi', T, u1, u2)
    >>> y1 = np.einsum('xyzijk,uvi,uvk->uvxyzj', T, u0, u2)
    >>> y2 = np.einsum('xyzijk,uvi,uvj->uvxyzk', T, u0, u1)
    >>> print(yy[0].shape)                      # W=(2,3) outer, C=(4,5,6) inner, then N0=10
    (2, 3, 4, 5, 6, 10)
    >>> print([bool(np.allclose(y, ref)) for y, ref in zip(yy, (y0, y1, y2))])
    [True, True, True]
    """
    use_jax = tree_contains_jax((vectors, T))
    xnp, _, _ = get_backend(True, use_jax)

    #
    d = len(vectors)
    C = T.shape[:-d]
    shape = T.shape[-d:]
    W = vectors[0].shape[:-1]

    for ii, v in enumerate(vectors):
        assert(v.shape[:-1] == W)
        assert(v.shape[-1] == shape[ii])

    # We are going to construct an einsum string from letters.
    # A dense 2x2x..x2 tensor exhausting these letters would have 4e15 entries
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    C_letters       = letters[:len(C)]
    shape_letters   = letters[len(C):len(C)+len(shape)]
    W_letters       = letters[len(C)+len(shape):len(C)+len(shape)+len(W)]

    vv_letters = []
    for ii in range(d):
        vv_letters.append(W_letters + shape_letters[ii])

    T_letters = C_letters + shape_letters

    zz = []
    for ii in range(d):
        str = T_letters
        for jj in range(ii): # front to back, add weighted slices
            str += ',' + vv_letters[jj]

        for jj in range(d-1,ii,-1): # back to front, contract with each slice
            str += ',' + vv_letters[jj]

        str += '->'

        str += W_letters + C_letters + shape_letters[ii]

        vvi = tuple(vectors[:ii] + vectors[ii+1:][::-1])

        z = xnp.einsum(str, T, *vvi)
        zz.append(z)

    return tuple(zz)
