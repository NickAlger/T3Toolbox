# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
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
    # Transpose of map from tangent vector to probes
    'compute_deta_tildes',
    'compute_tau_tildes',
    'compute_sigma_tildes',
    'compute_dxi_tildes',
    'assemble_tucker_variations',
    'assemble_tt_variations',
    'probe_tangent_transpose',
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
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as t3p
    >>> x = t3.TuckerTensorTrain.randn((10,11,12),(5,6,4),(1,2,3,1)).data
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.probe_t3(ww, x)
    >>> x_dense = t3.TuckerTensorTrain(*x).to_dense()
    >>> zz2 = t3p.probe_dense(ww, x_dense)
    >>> print([float(np.linalg.norm(z - z2)) for z, z2 in zip(zz, zz2)])
    [1.631156050514306e-13, 3.8657704262548816e-13, 6.591432899726004e-13]

    Vectorize over probes:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as t3p
    >>> x = t3.TuckerTensorTrain.randn((10,11,12),(5,6,4),(1,2,3,1)).data
    >>> ww = (np.random.randn(2,3, 10), np.random.randn(2,3, 11), np.random.randn(2,3, 12))
    >>> zz = t3p.probe_t3(ww, x)
    >>> x_dense = t3.TuckerTensorTrain(*x).to_dense()
    >>> zz2 = t3p.probe_dense(ww, x_dense)
    >>> print([float(np.linalg.norm(z - z2)) for z, z2 in zip(zz, zz2)])
    [1.0617919710198539e-12, 1.4936735922499436e-12, 1.1912692019537275e-12]

    Vectorize over probes and T3s:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as t3p
    >>> x = t3.TuckerTensorTrain.randn((10,11,12),(5,6,4),(1,2,3,1), stack_shape=(4,5)).data
    >>> ww = (np.random.randn(2,3, 10), np.random.randn(2,3, 11), np.random.randn(2,3, 12))
    >>> zz = t3p.probe_t3(ww, x)
    >>> x_dense = t3.TuckerTensorTrain(*x).to_dense()
    >>> zz2 = t3p.probe_dense(ww, x_dense)
    >>> print([float(np.linalg.norm(z - z2)) for z, z2 in zip(zz, zz2)])
    [5.7877816775957065e-12, 3.743460951628851e-12, 4.915470050149447e-12]
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
        # Three-group (W probe, K tangent, C base) contractions: sigma/dxi carry K, the base edge
        # vars (xi, mu) and base cores (Q, O) do not. t1/t3 self-infer the split from a C-only base
        # core; t2's only core is the variation core dG (K+C), so len(C) is supplied via n_base
        # (recovered here from the C-only Q, the n_probe precedent). Reduces to the two-group result
        # when K is empty.
        n_base = Q.ndim - 3
        t1 = contractions.WKCa_Caib_WCi_to_WKCb(sigma, Q, xi)
        t2 = contractions.WCa_KCaib_WCi_to_WKCb(mu, dG, xi, n_base)
        t3 = contractions.WCa_Caib_WKCi_to_WKCb(mu, O, dxi)
        sigma_next = t1 + t2 + t3
        return sigma_next, (sigma,)

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

    Probe a tangent vector with one set of vectors, compare against the dense reference:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> x = t3.TuckerTensorTrain.randn((10,11,12),(5,6,4),(1,2,3,1))
    >>> base, variations = bvf.t3_orthogonal_representations(x)
    >>> probe_base = base.data  # probing's base order == T3Basis.data, no reorder
    >>> v = t3m.T3Tangent(base, variations)
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.probe_tangent(ww, variations.data, probe_base)
    >>> zz2 = t3p.probe_dense(ww, v.to_dense())
    >>> print([float(np.linalg.norm(z - z2)) for z, z2 in zip(zz, zz2)])
    [2.802737740769268e-13, 2.1358428881151504e-13, 2.5895846738623505e-13]

    Probe a tangent vector with two sets of vectors:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> x = t3.TuckerTensorTrain.randn((10,11,12),(5,6,4),(1,2,3,1))
    >>> base, variations = bvf.t3_orthogonal_representations(x)
    >>> probe_base = base.data
    >>> v = t3m.T3Tangent(base, variations)
    >>> www = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> zzz = t3p.probe_tangent(www, variations.data, probe_base)
    >>> zzz2 = t3p.probe_dense(www, v.to_dense())
    >>> print([float(np.linalg.norm(zz - zz2)) for zz, zz2 in zip(zzz, zzz2)])
    [9.92987985605743e-12, 6.7500961542780035e-12, 4.080198471837904e-12]
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

    Adjoint identity with one set of probing vectors, <z, J v> = <J^T z, v>:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> x = t3.TuckerTensorTrain.randn((10,11,12),(5,6,4),(1,2,3,1))
    >>> base, _ = bvf.t3_orthogonal_representations(x)
    >>> probe_base = base.data
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> v = t3m.T3Tangent.randn(base)
    >>> z = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> Jv = t3p.probe_tangent(ww, v.variations.data, probe_base)
    >>> JTz = t3p.probe_tangent_transpose(z, ww, probe_base)
    >>> print(float(abs(cw.corewise_dot(z, Jv) - cw.corewise_dot(JTz, v.variations.data))))
    7.105427357601002e-15

    Adjoint identity with two sets of probing vectors:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> x = t3.TuckerTensorTrain.randn((10,11,12),(5,6,4),(1,2,3,1))
    >>> base, _ = bvf.t3_orthogonal_representations(x)
    >>> probe_base = base.data
    >>> ww = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> v = t3m.T3Tangent.randn(base)
    >>> z = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> Jv = t3p.probe_tangent(ww, v.variations.data, probe_base)
    >>> JTz = t3p.probe_tangent_transpose(z, ww, probe_base)
    >>> print(float(abs(cw.corewise_dot(z, Jv) - cw.corewise_dot(JTz, v.variations.data))))
    1.7763568394002505e-15
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

    Probe with one set of vectors:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> T = np.random.randn(10,11,12)
    >>> u0 = np.random.randn(10)
    >>> u1 = np.random.randn(11)
    >>> u2 = np.random.randn(12)
    >>> yy = t3p.probe_dense((u0,u1,u2),T)
    >>> y0 = np.einsum('ijk,j,k', T, u1, u2)
    >>> y1 = np.einsum('ijk,i,k', T, u0, u2)
    >>> y2 = np.einsum('ijk,i,j', T, u0, u1)
    >>> print(float(np.linalg.norm(yy[0] - y0)))
    7.377764055609925e-15
    >>> print(float(np.linalg.norm(yy[1] - y1)))
    0.0
    >>> print(float(np.linalg.norm(yy[2] - y2)))
    0.0

    Vectorize over probing vectors

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> T = np.random.randn(10,11,12)
    >>> u0 = np.random.randn(2,3, 10)
    >>> u1 = np.random.randn(2,3, 11)
    >>> u2 = np.random.randn(2,3, 12)
    >>> yy = t3p.probe_dense((u0,u1,u2),T)
    >>> y0 = np.einsum('ijk,uvj,uvk->uvi', T, u1, u2)
    >>> y1 = np.einsum('ijk,uvi,uvk->uvj', T, u0, u2)
    >>> y2 = np.einsum('ijk,uvi,uvj->uvk', T, u0, u1)
    >>> print(float(np.linalg.norm(yy[0] - y0)))
    1.663149665077564e-14
    >>> print(float(np.linalg.norm(yy[1] - y1)))
    0.0
    >>> print(float(np.linalg.norm(yy[2] - y2)))
    0.0

    Vectorize over probing vectors and big tensor

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> T = np.random.randn(4,5,6, 10,11,12)
    >>> u0 = np.random.randn(2,3, 10)
    >>> u1 = np.random.randn(2,3, 11)
    >>> u2 = np.random.randn(2,3, 12)
    >>> yy = t3p.probe_dense((u0,u1,u2),T)
    >>> y0 = np.einsum('xyzijk,uvj,uvk->uvxyzi', T, u1, u2)
    >>> y1 = np.einsum('xyzijk,uvi,uvk->uvxyzj', T, u0, u2)
    >>> y2 = np.einsum('xyzijk,uvi,uvj->uvxyzk', T, u0, u1)
    >>> print(float(np.linalg.norm(yy[0] - y0)))
    2.4890154384764807e-13
    >>> print(float(np.linalg.norm(yy[1] - y1)))
    0.0
    >>> print(float(np.linalg.norm(yy[2] - y2)))
    0.0
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
