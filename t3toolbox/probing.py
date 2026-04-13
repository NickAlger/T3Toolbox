# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

from networkx.utils.random_sequence import weighted_choice

import t3toolbox.base_variation_format
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.common as common
import t3toolbox.base_variation_format as bvf

__all__ = [
    # Probe a dense tensor
    'probe_dense',
    # Probe a Tucker tensor train
    'probe_t3',
    'absorb_weights_into_cores',
    'compute_weighted_xis',
    'compute_weighted_mus',
    'compute_weighted_nus',
    'compute_weighted_etas',
    'assemble_weighted_zs',
    # Probe a tangent vector
    'probe_tangent',
    'compute_weighted_dxis',
    'compute_weighted_sigmas',
    'compute_weighted_taus',
    'compute_weighted_detas',
    'assemble_weighted_tangent_zs',
    'absorb_weights_into_tangent',
    # Transpose of map from tangent vector to probes
    'compute_weighted_deta_tildes',
    'compute_weighted_tau_tildes',
    'compute_weighted_sigma_tildes',
    'compute_weighted_dxi_tildes',
    'assemble_tucker_variations',
    'assemble_tt_variations',
    'probe_tangent_transpose',
]

NDArray = typ.TypeVar('NDArray') # Generic stand-in for np.ndarray, jnp.ndarray, or other array backend


#####################################################
########    Probing a Tucker Tensor Train    ########
#####################################################

def probe_t3(
        ww: typ.Sequence[NDArray],
        x: t3.TuckerTensorTrain, edge_weights: typ.Tuple[
            typ.Sequence[NDArray],  # shape_weights, len=d, elm_shape=(Ni,)
            typ.Sequence[NDArray],  # tucker_weights, len=d, elm_shape=(ni,)
            typ.Sequence[NDArray],  # tt_weights, len=d+1, elm_shape=(ri,)
        ] = (None, None, None),
        map=common.ragged_map,
        scan=common.ragged_scan,
        xnp=np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(...,Ni)
    '''Probe a Tucker tensor train.

    See Section 5.2, particularly Figure 9 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    x: t3.TuckerTensorTrain
        Tucker tensor train to probe.
        structure=((N1,...,Nd),(n1,...,nd),(1,r1,...,r(d-1),1))
    ww: typ.Sequence[NDArray]
        input vectors to probe with. len=d, elm_shape=(...,Ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

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
    assemble_probes

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.probing as t3p
    >>> x = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.probe_t3(ww, x)
    >>> x_dense = t3.t3_to_dense(x)
    >>> zz2 = t3p.probe_dense(ww, x_dense)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [1.0259410400851746e-12, 1.0909087370186656e-12, 3.620283224238675e-13]

    Using weights:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.probing as t3p
    >>> randn = np.random.randn
    >>> x0 = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> (tucker_cores0, tt_cores0) = x0
    >>> shape_weights = [randn(10), randn(11), randn(12)]
    >>> tucker_weights = [randn(5), randn(6), randn(4)]
    >>> tt_weights = [randn(2), randn(3), randn(4), randn(2)]
    >>> edge_weights = (shape_weights, tucker_weights, tt_weights)
    >>> ww = [np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12)]
    >>> zz = t3p.probe_t3(ww, x0,edge_weights=edge_weights)
    >>> x = t3p.absorb_weights_into_cores(x0, edge_weights)
    >>> zz2 = t3p.probe_t3(ww, x)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [3.372228193172379e-14, 3.826148129405782e-14, 2.294115439089251e-14]
    '''
    shape = t3.get_structure(x)[0]
    assert(len(ww) == len(shape))

    tucker_cores, tt_cores = x

    for B, w in zip(tucker_cores, ww):
        assert(B.shape[1] == w.shape[-1])

    shape_weights, tucker_weights, tt_weights = edge_weights

    left_tt_weights     = tt_weights[:-1]   if tt_weights is not None else None
    right_tt_weights    = tt_weights[1:]    if tt_weights is not None else None

    weighted_ww = _apply_edge_weights(ww, shape_weights, map=map, xnp=xnp) if shape_weights is not None else ww

    weighted_xis = compute_weighted_xis(
        tucker_cores,
        weighted_ww,
        up_tucker_weights=tucker_weights,
        map=map,
        xnp=xnp,
    )

    weighted_mus = compute_weighted_mus(
        tt_cores,
        weighted_xis,
        left_tt_weights=left_tt_weights,
        scan=scan,
        xnp=xnp,
    )

    weighted_nus = compute_weighted_nus(
        tt_cores,
        weighted_xis,
        right_tt_weights=right_tt_weights,
        scan=scan,
        xnp=xnp,
    )

    weighted_etas = compute_weighted_etas(
        tt_cores,
        weighted_mus,
        weighted_nus,
        outer_tucker_weights=tucker_weights,
        map=map,
        xnp=xnp,
    )

    weighted_zs = assemble_weighted_zs(
        tucker_cores,
        weighted_etas,
        shape_weights=shape_weights,
        map=map,
        xnp=xnp,
    )

    return weighted_zs


def absorb_weights_into_cores(
        x0: t3.TuckerTensorTrain,
        weights: typ.Tuple[
            typ.Sequence[NDArray], # shape_weights, len=d, elm_shape=(Ni,)
            typ.Sequence[NDArray], # tucker_weights, len=d, elm_shape=(ni,)
            typ.Sequence[NDArray], # tt_weights, len=d+1, elm_shape=(ri,)
        ]
) -> t3.TuckerTensorTrain:
    tucker_cores0, tt_cores0 = x0
    shape_weights, tucker_weights, tt_weights = weights

    tucker_cores = tuple([
        np.einsum('i,io,o->io', tw, B, sw)
        for tw, B, sw in zip(tucker_weights, tucker_cores0, shape_weights)
    ])

    first_tt_cores = tuple([
        np.einsum('i,iaj->iaj', lw, G)
        for lw, G in zip(tt_weights[:-2], tt_cores0[:-1])
    ])
    last_tt_core = np.einsum('i,iaj,j->iaj', tt_weights[-2], tt_cores0[-1], tt_weights[-1])
    tt_cores = first_tt_cores + (last_tt_core,)

    return tucker_cores, tt_cores


def _apply_edge_weight(edge_variable, edge_weight, xnp=np):
    return xnp.einsum('...i,i->...i', edge_variable, edge_weight)


def _apply_edge_weights(edge_variables, edge_weights, map=common.ragged_map, xnp=np):
    (weighted_edge_variables,) = map(
        lambda v_w: (_apply_edge_weight(v_w[0], v_w[1], xnp=xnp),),
        (edge_variables, edge_weights)
    )
    return weighted_edge_variables


def compute_weighted_xis(
        up_tucker_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nUi,Ni)
        weighted_ww: typ.Sequence[NDArray], # len=d. elm_shape=(...,Ni)
        up_tucker_weights: typ.Sequence[NDArray] = None, # len=d, elm_shape=(nUi,)
        map = common.ragged_map,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # weighted_xis. len=d, elm_shape=(...,nUi)
    '''Compute upward edge variables associated with edges between Tucker cores and adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 5.2, particularly Figure 9 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    up_tucker_cores: typ.Sequence[NDArray]
        Tucker cores for Tucker tensor train.
        len=d. elm_shape=(nUi,Ni)
    ww: typ.Sequence[NDArray]
        input vectors to probe with. len=d, elm_shape=(...,Ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        upward edge variables xi. len=d, elm_shape=(...,nUi)

    See Also
    --------
    probe_t3
    compute_mus
    compute_nus
    compute_etas
    assemble_probes
    '''
    xi_weights = up_tucker_weights

    def _func(x):
        U, weighted_w, ind = x[0], x[1], 2

        xi = xnp.einsum('io,...o->...i', U, weighted_w)

        if xi_weights is not None:
            weight = x[ind]
            weighted_xi = _apply_edge_weight(xi, weight)
        else:
            weighted_xi = xi

        return (weighted_xi,)

    xs = (up_tucker_cores, weighted_ww)
    xs = xs + (xi_weights,) if xi_weights  is not None else xs

    (weighted_xis,) = map(_func, xs)
    return weighted_xis


def compute_weighted_mus(
        left_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(rLi,nUi,rL(i+1))
        weighted_xis: typ.Sequence[NDArray], # len=d. elm_shape=(...,nUi)
        left_tt_weights: typ.Sequence[NDArray] = None, # len=d, elm_shape=(rLi,)
        scan = common.ragged_scan,
        xnp = np,
) -> typ.Sequence[NDArray]: # mus. len=d, elm_shape=(...,rLi)
    '''Compute leftward edge variables associated with edges between adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 5.2, particularly Figure 9 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    left_tt_cores: typ.Sequence[NDArray]
        Left TT-cores for Tucker tensor train.
        len=d. elm_shape=(rLi,nUi,rL(i+1))
    xis: typ.Sequence[NDArray]
        upward edge variables xi. len=d. elm_shape=(...,nUi)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        leftward edge variables mu. len=d, elm_shape=(...,rLi)

    See Also
    --------
    probe_t3
    compute_xis
    compute_nus
    compute_etas
    assemble_probes
    '''
    mu_weights = left_tt_weights

    def _func(mu, x):
        P, weighted_xi, ind = x[0], x[1], 2

        if mu_weights is not None:
            weight = x[ind]
            weighted_mu = _apply_edge_weight(mu, weight)
        else:
            weighted_mu = mu

        mu_next = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', weighted_mu, P),
            weighted_xi,
        )
        return mu_next, (weighted_mu,)

    r0 = left_tt_cores[0].shape[0]
    vectorization_shape = weighted_xis[0].shape[:-1]
    init = xnp.ones(vectorization_shape + (r0,))

    xs = (left_tt_cores, weighted_xis)
    xs = xs + (mu_weights,) if mu_weights is not None else xs

    last_weighted_mu, (weighted_mus,) = scan(_func, init, xs)
    return weighted_mus


def tt_reverse(tt_cores):
    return tuple([G.swapaxes(0, 2) for G in tt_cores[::-1]])


def compute_weighted_nus(
        right_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(rRi,nUi,rR(i+1))
        weighted_xis, # len=d. elm_shape=(...,nUi)
        right_tt_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(rRi,)
        scan = common.ragged_scan,
        xnp = np,
) -> typ.Sequence[NDArray]: # nus. len=d, elm_shape=(...,rR(i+1))
    '''Compute rightward edge variables associated with edges between adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 5.2, particularly Figure 9 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    right_tt_cores: typ.Sequence[NDArray]
        Right TT-cores for Tucker tensor train.
        len=d. elm_shape=(rRi,nUi,rR(i+1))
    xis: typ.Sequence[NDArray]
        upward edge variables xi. len=d. elm_shape=(...,nUi)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        rightward edge variables nu. len=d, elm_shape=(...,rR(i+1))

    See Also
    --------
    probe_t3
    compute_xis
    compute_mus
    compute_etas
    assemble_probes
    '''
    rev_tt_cores = tt_reverse(right_tt_cores)
    rev_weighted_xis = weighted_xis[::-1]
    rev_right_tt_weights  = None if right_tt_weights is None else right_tt_weights[::-1]

    rev_weighted_nus = compute_weighted_mus(
        rev_tt_cores,
        rev_weighted_xis,
        left_tt_weights=rev_right_tt_weights,
        scan=scan,
        xnp=xnp,
    )
    weighted_nus = rev_weighted_nus[::-1]
    return weighted_nus


def compute_weighted_etas(
        outer_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(rLi,nOi,rR(i+1))
        weighted_mus, # len=d. elm_shape=(...,rLi)
        weighted_nus, # len=d. elm_shape=(...,rR(i+1))
        outer_tucker_weights: typ.Sequence[NDArray] = None, # len=d, elm_shape=(nOi)
        map = common.ragged_map,
        xnp = np,
) -> typ.Sequence[NDArray]: # weighted_etas. len=d, elm_shape=(...,nOi)
    '''Compute downward edge variables associated with edges between Tucker cores and adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 5.2, particularly Figure 9 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    outer_tt_cores: typ.Sequence[NDArray]
        Outer TT-cores for Tucker tensor train.
        len=d. elm_shape=(ri,ni,r(i+1))
    mus: typ.Sequence[NDArray]
        leftward edge variables mu. len=d. elm_shape=(num_probes,ri)
    nus: typ.Sequence[NDArray]
        rightward edge variables mu. len=d. elm_shape=(num_probes,r(i+1))
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        downward edge variables eta. len=d+1, elm_shape=(num_probes,ni)

    See Also
    --------
    probe_t3
    compute_xis
    compute_xis
    compute_nus
    assemble_probes
    '''
    eta_weights = outer_tucker_weights

    def _func(x):
        weighted_mu, G, weighted_nu, ind = x[0], x[1], x[2], 3

        eta = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', weighted_mu, G),
            weighted_nu,
        )

        if eta_weights is not None:
            weight = x[ind]
            weighted_eta = _apply_edge_weight(eta, weight)
        else:
            weighted_eta = eta

        return (weighted_eta,)

    xs = (weighted_mus, outer_tt_cores, weighted_nus)
    xs = xs + (eta_weights,) if eta_weights is not None else xs

    (weighted_etas,) = map(_func, xs)
    return weighted_etas


def assemble_weighted_zs(
        tucker_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(ni,Ni)
        weighted_etas,  # len=d. elm_shape=(...,ni)
        shape_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(Ni,)
        map = common.ragged_map,
        xnp = np,
) -> typ.Sequence[NDArray]: # weighted_zs. len=d, elm_shape=(...,Ni)
    '''Assemble probes from downward edge variables.

    See Section 5.2, particularly Figure 9 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    tucker_cores: typ.Sequence[NDArray]
        Tucker cores for Tucker tensor train.
        len=d. elm_shape=(ni,Ni)
    etas: typ.Sequence[NDArray]
        downward edge variables eta. len=d. elm_shape=(...,ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        probes z. len=d, elm_shape=(...,Ni)

    See Also
    --------
    probe_t3
    compute_xis
    compute_mus
    compute_nus
    compute_etas
    '''
    z_weights = shape_weights

    def _func(x):
        weighted_eta, U, ind = x[0], x[1], 2

        z = xnp.einsum('...a,ao->...o', weighted_eta, U)

        if z_weights is not None:
            weight = x[ind]
            weighted_z = _apply_edge_weight(z, weight)
        else:
            weighted_z = z

        return (weighted_z,)

    xs = (weighted_etas, tucker_cores)
    xs = xs + (z_weights,) if z_weights is not None else xs

    (weighted_zs,) = map(_func, xs)
    return weighted_zs


#####################################################
###########    Probing a tangent vector    ##########
#####################################################

def compute_weighted_dxis(
        var_tucker_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        weighted_ww: typ.Sequence[NDArray], # len=d. elm_shape=(...,Ni)
        outer_tucker_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nOi,)
        map=common.ragged_map,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # xis. len=d, elm_shape=(...,nOi)
    '''Compute var-upward edge variables dxi.
    Used for probing a tangent vector.

    Same as t3_compute_dxis(), except with var_tucker_cores in place of tucker_cores.

    See Section 5.2.3, particularly Formula (34), in:
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
    assemble_tangent_probes
    probe_tangent
    '''
    return compute_weighted_xis(
        var_tucker_cores,
        weighted_ww,
        up_tucker_weights=outer_tucker_weights,
        map=map,
        xnp=xnp,
    )


def compute_weighted_sigmas(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        weighted_xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nUi),
        weighted_dxis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nOi)
        weighted_mus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nLi)
        right_tt_weights: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rRi,)
        scan = common.ragged_scan,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # weighted_sigmas. len=d, elm_shape=(...,rR(i+1))
    '''Compute var-leftward edge variables sigma.
    Used for probing a tangent vector.

    See Section 5.2.3, particularly Formula (36), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_dxis
    compute_taus
    compute_detas
    assemble_tangent_probes
    probe_tangent
    '''
    sigma_weights = right_tt_weights

    def _func(weighted_sigma, x):
        Q, O, dG, weighted_xi, weighted_dxi, weighted_mu, ind = x[0], x[1], x[2], x[3], x[4], x[5], 6

        sigma_next_t1 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', weighted_sigma, Q),
            weighted_xi
        )
        sigma_next_t2 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', weighted_mu, dG),
            weighted_xi
        )
        sigma_next_t3 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', weighted_mu, O),
            weighted_dxi
        )
        sigma_next = sigma_next_t1 + sigma_next_t2 + sigma_next_t3

        if sigma_weights is not None:
            weight = x[ind]
            weighted_sigma_next = _apply_edge_weight(sigma_next, weight)
        else:
            weighted_sigma_next = sigma_next

        return weighted_sigma_next, (weighted_sigma,)

    rR0 = right_tt_cores[0].shape[0]
    vectorization_shape = weighted_xis[0].shape[:-1]
    init = xnp.zeros(vectorization_shape + (rR0,))

    xs = (right_tt_cores, outer_tt_cores, var_tt_cores, weighted_xis, weighted_dxis, weighted_mus)
    xs = xs + (sigma_weights,)    if sigma_weights  is not None else xs

    last_weighted_sigma, (weighted_sigmas,) = scan(_func, init, xs)
    return weighted_sigmas


def compute_weighted_taus(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        left_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rL(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        weighted_xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nUi),
        weighted_dxis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nOi)
        weighted_nus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nR(i+1))
        left_tt_weights: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rLi,)
        scan = common.ragged_scan,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # weighted_taus. len=d, elm_shape=(...,rL(i+1))
    '''Compute var-rightward edge variables tau.
    Used for probing a tangent vector.

    See Section 5.2.3, particularly Formula (38), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_dxis
    compute_sigmas
    compute_detas
    assemble_tangent_probes
    probe_tangent
    '''
    rev_var_tt_cores    = tt_reverse(var_tt_cores)
    rev_left_tt_cores   = tt_reverse(left_tt_cores)
    rev_outer_tt_cores  = tt_reverse(outer_tt_cores)
    rev_weighted_xis    = weighted_xis[::-1]
    rev_weighted_dxis   = weighted_dxis[::-1]
    rev_weighted_nus    = weighted_nus[::-1]
    rev_left_tt_weights = None if left_tt_weights is None else left_tt_weights[::-1]

    rev_weighted_taus = compute_weighted_sigmas(
        rev_var_tt_cores,
        rev_left_tt_cores,
        rev_outer_tt_cores,
        rev_weighted_xis,
        rev_weighted_dxis,
        rev_weighted_nus,
        right_tt_weights=rev_left_tt_weights,
        scan=scan,
        xnp=xnp
    )
    weighted_taus = rev_weighted_taus[::-1]
    return weighted_taus


def compute_weighted_detas(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        left_tt_cores: typ.Sequence[NDArray],  # len=d, elm_shape=(rLi,nUi,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        weighted_mus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nLi)
        weighted_nus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nRi)
        weighted_sigmas: typ.Sequence[NDArray], # len=d, elm_shape=(...,rRi)
        weighted_taus: typ.Sequence[NDArray], # len=d, elm_shape=(...,rL(i+1))
        up_tucker_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nUi,)
        map = common.ragged_map,
        xnp = np,
) -> typ.Sequence[NDArray]: # detas. len=d, elm_shape=(...,nUi)
    '''Compute var-downward edge variables deta.
    Used for probing a tangent vector.

    See Section 5.2.3, particularly Formula (40), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_dxis
    compute_sigmas
    compute_taus
    assemble_tangent_probes
    probe_tangent
    '''
    deta_weights  = up_tucker_weights

    def _func(x):
        (P, Q, dG, weighted_mu, weighted_nu,
         weighted_sigma, weighted_tau, ind
         ) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6], 7)

        deta_term1 = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', weighted_sigma, Q),
            weighted_nu,
        )
        deta_term2 = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', weighted_mu, dG),
            weighted_nu,
        )
        deta_term3 = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', weighted_mu, P),
            weighted_tau,
        )
        deta = deta_term1 + deta_term2 + deta_term3

        if deta_weights is not None:
            weight = x[ind]
            weighted_deta = _apply_edge_weight(deta, weight)
        else:
            weighted_deta = deta

        return (weighted_deta,)

    xs = (left_tt_cores, right_tt_cores, var_tt_cores,
          weighted_mus, weighted_nus, weighted_sigmas, weighted_taus)
    xs = xs + (deta_weights,)     if deta_weights   is not None else xs

    detas_tuple = map(_func, xs)
    return detas_tuple[0]


def assemble_weighted_tangent_zs(
        tucker_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(nUi,Ni)
        var_tucker_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        weighted_etas: typ.Sequence[NDArray], # etas. len=d, elm_shape=(...,nUi)
        weighted_detas: typ.Sequence[NDArray], # detas. len=d, elm_shape=(...,nUi)
        shape_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(Ni,)
        map = common.ragged_map,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # probes. len=d, elm_shape=(...,Ni)
    '''Assemble tangent vector probes from edge variables.

    See Section 5.2.3, particularly Formula (41), in:
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
    z_weights = shape_weights

    def _func(x):
        B, dB, weighted_eta, weighted_deta, ind = x[0], x[1], x[2], x[3], 4

        z_term1 = xnp.einsum('ao,...a->...o', B, weighted_deta)
        z_term2 = xnp.einsum('ao,...a->...o', dB, weighted_eta)
        z = z_term1 + z_term2

        if z_weights is not None:
            weight = x[ind]
            weighted_z = _apply_edge_weight(z, weight)
        else:
            weighted_z = z

        return (weighted_z,)

    xs = (tucker_cores, var_tucker_cores, weighted_etas, weighted_detas)
    xs = xs + (z_weights,) if z_weights  is not None else xs

    (weighted_zs,) = map(_func, xs)
    return weighted_zs


def probe_tangent(
        ww: typ.Sequence[NDArray],  # input vectors, len=d, elm_shape=(...,Ni)
        variation: bvf.T3Variation, # tucker_var_shapes=(nOi,Ni), tt_var_shapes=tt_hole_shapes=(rLi,ni,rRi)
        base: bvf.T3Base, # tucker_hole_shapes=(nOi,Ni), tt_hole_shapes=(rLi,ni,rRi)
        edge_weights: typ.Tuple[
            typ.Sequence[NDArray],  # shape_weights, len=d, elm_shape=(Ni,)
            typ.Sequence[NDArray],  # up_tucker_weights, len=d, elm_shape=(nUi,)
            typ.Sequence[NDArray],  # outer_tucker_weights, len=d, elm_shape=(nOi,)
            typ.Sequence[NDArray],  # left_tt_weights, len=d+1, elm_shape=(rLi,)
            typ.Sequence[NDArray],  # right_tt_weights, len=d+1, elm_shape=(rRi,)
        ] = (None, None, None, None, None),
        map = common.ragged_map,
        scan = common.ragged_scan,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(...,Ni)
    '''Probe a tangent vector.

    See Section 5.2.3 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    x: t3m.T3Tangent
        Tangent vector to probe.
        shape=(N1,...,Nd)
    ww: typ.Sequence[NDArray]
        input vectors to probe with. len=d, elm_shape=(...,Ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        Probes, zz. len=d, elm_shape=(Ni,) or (num_probes,Ni)

    See Also
    --------
    probe_t3
    probe_tangent_transpose
    compute_xis
    compute_mus
    compute_nus
    compute_etas
    compute_dxis
    compute_sigmas
    assemble_probes
    compute_detas
    assemble_tangent_probes

    Examples
    --------

    Probe tangent with one set of vectors:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.probe_tangent(ww, variation, base)
    >>> zz2 = t3p.probe_dense(ww, t3m.tangent_to_dense(variation, base))
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [4.6257812371663175e-15, 3.628238740198284e-15, 5.6097341748343224e-15]

    Probe tangent with two sets of vectors:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> www = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> zzz = t3p.probe_tangent(www, variation, base) # Compute probes!
    >>> zzz2 = t3p.probe_dense(www,t3m.tangent_to_dense(variation, base))
    >>> print([np.linalg.norm(zz - zz2) for zz, zz2 in zip(zzz, zzz2)])
    [3.863711710898517e-15, 5.474255194514171e-15, 5.930347504865667e-15]

    Example with weights

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.orthogonalization as orth
    >>> randn = np.random.randn
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,5,4)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> NN = [U.shape[1] for U in base[0]]
    >>> nnU = [U.shape[0] for U in base[0]]
    >>> rrL = [L.shape[0] for L in base[1]]
    >>> rrR = [R.shape[2] for R in base[2]]
    >>> nnO = [O.shape[1] for O in base[3]]
    >>> shape_weights = [randn(N) for N in NN]
    >>> up_tucker_weights = [randn(nU) for nU in nnU]
    >>> outer_tucker_weights = [randn(nO) for nO in nnO]
    >>> left_tt_weights = [randn(rL) for rL in rrL]
    >>> right_tt_weights = [randn(rR) for rR in rrR]
    >>> edge_weights = (shape_weights, up_tucker_weights, outer_tucker_weights, left_tt_weights, right_tt_weights)
    >>> www = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> zzz = t3p.probe_tangent(www, variation, base, edge_weights=edge_weights)
    >>> weighted_variation, weighted_base = t3p.absorb_weights_into_tangent(variation, base, edge_weights)
    >>> zzz2 = t3p.probe_tangent(www, weighted_variation, weighted_base)
    >>> print([np.linalg.norm(zz - zz2) for zz, zz2 in zip(zzz, zzz2)])
    [1.5683512051190777e-15, 4.368484248906507e-15, 1.855735793037041e-15]
    '''
    t3toolbox.base_variation_format.check_fit(variation, base)

    x_shape = tuple([B.shape[1] for B in variation[0]])
    assert(len(ww) == len(x_shape))

    assert(tuple([w.shape[-1] for w in ww]) == x_shape)

    (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores) = base
    (var_tucker_cores, var_tt_cores) = variation

    (shape_weights,
     up_tucker_weights, outer_tucker_weights,
     left_tt_weights, right_tt_weights,
     ) = edge_weights

    weighted_ww = _apply_edge_weights(ww, shape_weights, xnp=xnp) if shape_weights is not None else ww

    weighted_xis = compute_weighted_xis(
        up_tucker_cores,
        weighted_ww,
        up_tucker_weights=up_tucker_weights,
        map=map,
        xnp=xnp,
    )

    weighted_mus = compute_weighted_mus(
        left_tt_cores,
        weighted_xis,
        left_tt_weights=left_tt_weights,
        scan=scan,
        xnp=xnp,
    )

    weighted_nus = compute_weighted_nus(
        right_tt_cores,
        weighted_xis,
        right_tt_weights=right_tt_weights,
        scan=scan,
        xnp=xnp,
    )

    weighted_etas = compute_weighted_etas(
        outer_tt_cores,
        weighted_mus,
        weighted_nus,
        outer_tucker_weights=outer_tucker_weights,
        map=map,
        xnp=xnp,
    )

    weighted_dxis = compute_weighted_dxis(
        var_tucker_cores,
        weighted_ww,
        outer_tucker_weights=outer_tucker_weights,
        xnp=xnp,
    )

    weighted_sigmas = compute_weighted_sigmas(
        var_tt_cores,
        right_tt_cores,
        outer_tt_cores,
        weighted_xis,
        weighted_dxis,
        weighted_mus,
        right_tt_weights=right_tt_weights,
        scan=scan,
        xnp=xnp,
    )

    weighted_taus = compute_weighted_taus(
        var_tt_cores,
        left_tt_cores,
        outer_tt_cores,
        weighted_xis,
        weighted_dxis,
        weighted_nus,
        left_tt_weights=left_tt_weights,
        scan=scan,
        xnp=xnp,
    )

    weighted_detas = compute_weighted_detas(
        var_tt_cores,
        left_tt_cores,
        right_tt_cores,
        weighted_mus,
        weighted_nus,
        weighted_sigmas,
        weighted_taus,
        up_tucker_weights=up_tucker_weights,
        map=map,
        xnp=xnp,
    )

    weighted_zz = assemble_weighted_tangent_zs(
        up_tucker_cores,
        var_tucker_cores,
        weighted_etas,
        weighted_detas,
        shape_weights=shape_weights,
        map=map,
        xnp=xnp,
    )

    return weighted_zz


def absorb_weights_into_tangent(
        variation: bvf.T3Variation,
        base: bvf.T3Base,
        edge_weights: typ.Tuple[
            typ.Sequence[NDArray],  # shape_weights, len=d, elm_shape=(Ni,)
            typ.Sequence[NDArray],  # up_tucker_weights, len=d, elm_shape=(nUi,)
            typ.Sequence[NDArray],  # outer_tucker_weights, len=d, elm_shape=(nOi,)
            typ.Sequence[NDArray],  # left_tt_weights, len=d, elm_shape=(rLi,)
            typ.Sequence[NDArray],  # right_tt_weights, len=d, elm_shape=(rRi,)
        ] = (None, None, None, None),
) -> typ.Tuple[
    bvf.T3Variation, # weighted variation
    bvf.T3Base, # weighted base
]:
    (shape_weights,
     up_tucker_weights, outer_tucker_weights,
     left_tt_weights, right_tt_weights,
     ) = edge_weights

    (up_tucker_cores0, left_tt_cores0, right_tt_cores0, outer_tt_cores0) = base
    (var_tucker_cores0, var_tt_cores0) = variation

    up_tucker_cores = tuple([
        np.einsum('i,io,o->io', tw, U, sw)
        for tw, U, sw in zip(up_tucker_weights, up_tucker_cores0, shape_weights)
    ])

    var_tucker_cores = [
        np.einsum('i,io,o->io', tw, V, sw)
        for tw, V, sw in zip(outer_tucker_weights, var_tucker_cores0, shape_weights)
    ]

    left_tt_cores = tuple([
        np.einsum('i,iaj->iaj', lw, L)
        for lw, L in zip(left_tt_weights, left_tt_cores0)
    ])

    right_tt_cores = tuple([
        np.einsum('iaj,j->iaj', R, rw)
        for rw, R in zip(right_tt_weights, right_tt_cores0)
    ])

    outer_tt_cores = tuple([
        np.einsum('i,iaj,j->iaj', lw, O, rw)
        for lw, rw, O in zip(
            left_tt_weights,
            right_tt_weights,
            outer_tt_cores0,
        )
    ])

    var_tt_cores = tuple([
        np.einsum('i,iaj,j->iaj', lw, H, rw)
        for lw, rw, H in zip(
            left_tt_weights,
            right_tt_weights,
            var_tt_cores0,
        )
    ])

    weighted_base = (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    weighted_variation = (var_tucker_cores, var_tt_cores)

    return weighted_variation, weighted_base


###############################################################
###########    Transpose of tangent to probes map    ##########
###############################################################

def compute_weighted_deta_tildes(
        up_tucker_cores: typ.Sequence[NDArray],  # len=d, elm_shape=(ni,Ni)
        weighted_ztildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,Ni)
        up_tucker_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nUi,)
        map = common.ragged_map,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(...,ni)
    '''Adjoint-var-upward edge variables deta_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (43), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    return compute_weighted_xis(
        up_tucker_cores,
        weighted_ztildes,
        up_tucker_weights=up_tucker_weights,
        map=map,
        xnp=xnp,
    )


def compute_weighted_tau_tildes(
        weighted_deta_tildes,  # len=d+1, elm_shape=(...,ni)
        left_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rL(i+d))
        weighted_xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,ni)
        weighted_mus, # len=d, elm_shape=(...,rLi)
        left_tt_weights: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rLi,)
        scan = common.ragged_scan,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(...,rLi)
    '''Adjoint-var-rightward edge variables tau_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (44), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    tau_tilde_weights = left_tt_weights

    def _func(tau_tilde, x):
        P, weighted_xi, weighted_deta_tilde, weighted_mu, ind = x[0], x[1], x[2], x[3], 4

        if tau_tilde_weights is not None:
            weight = x[ind]
            weighted_tau_tilde = _apply_edge_weight(tau_tilde, weight)
        else:
            weighted_tau_tilde = tau_tilde

        tau_tilde_next_t1 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', weighted_tau_tilde, P),
            weighted_xi
        )
        tau_tilde_next_t2 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', weighted_mu, P),
            weighted_deta_tilde
        )
        tau_tilde_next = tau_tilde_next_t1 + tau_tilde_next_t2

        return tau_tilde_next, (weighted_tau_tilde,)

    init = xnp.zeros(weighted_mus[0].shape[:-1] + (left_tt_cores[0].shape[0],))
    xs = (left_tt_cores, weighted_xis, weighted_deta_tildes, weighted_mus)
    xs = xs + (tau_tilde_weights,) if tau_tilde_weights is not None else xs

    last_weighted_tau_tilde, (weighted_tau_tildes,) = scan(_func, init, xs)
    return weighted_tau_tildes


def compute_weighted_sigma_tildes(
        weighted_deta_tildes,  # len=d, elm_shape=(...,ni)
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,ni,rR(i+d))
        weighted_xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,ni)
        weighted_nus, # len=d, elm_shape=(...,rR(i+1))
        right_tt_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(rRi,)
        scan=common.ragged_scan,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(...,rR(i+1))
    '''Adjoint-var-leftward edge variables sigma_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (45), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    rev_right_tt_weights = right_tt_weights[::-1] if right_tt_weights is not None else None

    return compute_weighted_tau_tildes(
        weighted_deta_tildes[::-1], tt_reverse(right_tt_cores), weighted_xis[::-1], weighted_nus[::-1],
        left_tt_weights = rev_right_tt_weights,
        scan=scan, xnp=xnp,
    )[::-1]


def compute_weighted_dxi_tildes(
        weighted_sigma_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rR(i+1))
        weighted_tau_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rLi)
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        weighted_mus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rLi)
        weighted_nus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rR(i+1))
        outer_tucker_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nOi,)
        map = common.ragged_map,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # dxi_tildes. len=d, elm_shape=(...,nOi)
    '''Adjoint-var-downward edge variables dxi_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (46), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    dxi_tilde_weights = outer_tucker_weights

    def _func(x):
        O, weighted_mu, weighted_nu, weighted_st, weighted_tt, ind = x[0], x[1], x[2], x[3], x[4], 5

        dxi_tilde = (
                xnp.einsum(
                    '...aj,...j->...a',
                    xnp.einsum('...i,iaj->...aj',weighted_tt, O),
                    weighted_nu
                )
                +
                xnp.einsum(
                    '...aj,...j->...a',
                    xnp.einsum('...i,iaj->...aj', weighted_mu, O),
                    weighted_st
                )
        )

        if dxi_tilde_weights is not None:
            weight = x[ind]
            weighted_dxi_tilde = _apply_edge_weight(dxi_tilde, weight)
        else:
            weighted_dxi_tilde = dxi_tilde

        return (weighted_dxi_tilde,)

    xs = (outer_tt_cores, weighted_mus, weighted_nus, weighted_sigma_tildes, weighted_tau_tildes)
    xs = xs + (outer_tucker_weights,) if outer_tucker_weights is not None else xs

    (weighted_dxi_tildes,) = map(_func, xs)
    return weighted_dxi_tildes


def assemble_tucker_variations(
        weighted_ztildes: typ.Sequence[NDArray], # len=d, elm_shape=(...,Ni)
        weighted_dxi_tildes: typ.Sequence[NDArray], #len=d, elm_shape=(...,nOi)
        weighted_ww: typ.Sequence[NDArray],  # input vectors, len=d, elm_shape=(Ni,) or (...,Ni)
        weighted_etas: typ.Sequence[NDArray],  # etas. len=d, elm_shape=(...,ni)
        sum_over_probes: bool = False,
        map = common.ragged_map,
        xnp = np,
):
    '''Assemble Tucker core variations, delta_U_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (47), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    def _func(x):
        weighted_z_tilde, weighted_eta, weighted_w, weighted_dxi_tilde = x
        if sum_over_probes:
            dU_tilde = (
                    xnp.einsum('...o,...a->ao', weighted_z_tilde, weighted_eta)
                    +
                    xnp.einsum('...o,...a->ao', weighted_w, weighted_dxi_tilde)
            )
        else:
            dU_tilde = (
                    xnp.einsum('...o,...a->...ao', weighted_z_tilde, weighted_eta)
                    +
                    xnp.einsum('...o,...a->...ao', weighted_w, weighted_dxi_tilde)
            )
        return [dU_tilde]

    dU_tildes_tuple = map(_func, (weighted_ztildes, weighted_etas, weighted_ww, weighted_dxi_tildes))
    return dU_tildes_tuple[0]


def assemble_tt_variations(
        weighted_sigma_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rR(i+1))
        weighted_tau_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rLi)
        weighted_deta_tildes,  # len=d+1, elm_shape=(...,ni)
        weighted_xis: typ.Sequence[NDArray],  # len=d, elm_shape=(...,ni)
        weighted_mus,  # len=d, elm_shape=(...,rLi)
        weighted_nus,  # len=d, elm_shape=(...,rR(i+1))
        sum_over_probes: bool = False,
        map = common.ragged_map,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(...,rLi,nOi,rRi)
    '''Assemble TT core variations, delta_G_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (48), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    def _func(x):
        weighted_xi, weighted_mu, weighted_nu, weighted_sigma_tilde, weighted_tau_tilde, weighted_deta_tilde = x
        if sum_over_probes:
            dG_tilde = (
                    xnp.einsum(
                        '...ia,...j->iaj',
                        xnp.einsum('...i,...a->...ia', weighted_mu, weighted_xi),
                        weighted_sigma_tilde
                    )
                    +
                    xnp.einsum(
                        '...ia,...j->iaj',
                        xnp.einsum('...i,...a->...ia', weighted_tau_tilde, weighted_xi),
                        weighted_nu
                    )
                    +
                    xnp.einsum(
                        '...ia,...j->iaj',
                        xnp.einsum('...i,...a->...ia', weighted_mu, weighted_deta_tilde),
                        weighted_nu
                    )
            )
        else:
            dG_tilde = (
                    xnp.einsum(
                        '...ia,...j->...iaj',
                        xnp.einsum('...i,...a->...ia', weighted_mu, weighted_xi),
                        weighted_sigma_tilde
                    )
                    +
                    xnp.einsum(
                        '...ia,...j->...iaj',
                        xnp.einsum('...i,...a->...ia', weighted_tau_tilde, weighted_xi),
                        weighted_nu
                    )
                    +
                    xnp.einsum(
                        '...ia,...j->...iaj',
                        xnp.einsum('...i,...a->...ia', weighted_mu, weighted_deta_tilde),
                        weighted_nu
                    )
            )
        return [dG_tilde]

    xs = (weighted_xis, weighted_mus, weighted_nus, weighted_sigma_tildes, weighted_tau_tildes, weighted_deta_tildes)
    dG_tildes_tuple = map(_func, xs)
    return dG_tildes_tuple[0]


def probe_tangent_transpose(
        ztildes: typ.Sequence[NDArray], # len=d, elm_shape=(...,Ni)
        ww: typ.Sequence[NDArray],  # input vectors, len=d, elm_shape=(...,Ni)
        base: t3toolbox.base_variation_format.T3Base, # tucker_hole_shapes=(nOi,Ni), tt_hole_shapes=(rLi,ni,rRi)
        edge_weights: typ.Tuple[
            typ.Sequence[NDArray],  # shape_weights, len=d, elm_shape=(Ni,)
            typ.Sequence[NDArray],  # up_tucker_weights, len=d, elm_shape=(nUi,)
            typ.Sequence[NDArray],  # outer_tucker_weights, len=d, elm_shape=(nOi,)
            typ.Sequence[NDArray],  # left_tt_weights, len=d+1, elm_shape=(rLi,)
            typ.Sequence[NDArray],  # right_tt_weights, len=d+1, elm_shape=(rRi,)
        ] = (None, None, None, None, None),
        sum_over_probes: bool = False,
        xnp = np,
) -> typ.Tuple[
    typ.Tuple[NDArray,...], # dU_tildes. len=d, elm_shape=(...,nOi,Ni)
    typ.Tuple[NDArray,...], # dG_tildes. len=d, elm_shape=(...,rLi,ni,rRi)
]:
    '''Apply the transpose of the map from a T3Tangent to its probes. Apply to ztildes.

    See Section 5.2.4 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    ztildes: typ.Sequence[NDArray]
        Probe residuals to apply the map to
        len=d, elm_shape=(Ni,) or (num_probes,Ni)
    base: t3m.T3Base,
        Orthogonal base for point where the tangent space attaches to the manifold.
        shape=(N1,...,Nd)
    sum_over_probes: bool
        Sum results over all probe residuals, rather than returning results for each probe residual
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    t3m.T3Tangent
        Tangent vector resulting from applying transpose map to ztildes

    See Also
    --------
    probe_t3
    tangent_probes

    Examples
    --------

    Apply transpose map with one set of probing vectors:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.common as common
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> v1 = t3m.tangent_randn(base)
    >>> zz1 = t3p.probe_tangent(ww, v1, base)
    >>> zz2 = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> v2 = t3p.probe_tangent_transpose(zz2, ww, base)
    >>> ipA = cw.corewise_dot(v1, v2)
    >>> print(ipA)
    17.958317927787
    >>> ipB = cw.corewise_dot(zz1, zz2)
    >>> print(ipB)
    17.958317927787

    Apply transpose map with two sets of probing vectors:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.common as common
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> ww = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> apply_J = lambda v: t3p.probe_tangent(ww, v, base)
    >>> apply_Jt = lambda z: t3p.probe_tangent_transpose(z, ww, base)
    >>> v = t3m.tangent_randn(base)
    >>> z = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> print(cw.corewise_dot(z, apply_J(v)) - cw.corewise_dot(apply_Jt(z), v))
    7.105427357601002e-15

    Using weights:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.common as common
    >>> import t3toolbox.orthogonalization as orth
    >>> randn = np.random.randn
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> NN = [U.shape[1] for U in base[0]]
    >>> nnU = [U.shape[0] for U in base[0]]
    >>> rrL = [L.shape[0] for L in base[1]]
    >>> rrR = [R.shape[2] for R in base[2]]
    >>> nnO = [O.shape[1] for O in base[3]]
    >>> shape_weights = [randn(N) for N in NN]
    >>> up_tucker_weights = [randn(nU) for nU in nnU]
    >>> outer_tucker_weights = [randn(nO) for nO in nnO]
    >>> left_tt_weights = [randn(rL) for rL in rrL]
    >>> right_tt_weights = [randn(rR) for rR in rrR]
    >>> edge_weights = (shape_weights, up_tucker_weights, outer_tucker_weights, left_tt_weights, right_tt_weights)
    >>> ww = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> apply_J = lambda v: t3p.probe_tangent(ww, v, base, edge_weights=edge_weights)
    >>> apply_Jt = lambda z: t3p.probe_tangent_transpose(z, ww, base, edge_weights=edge_weights)
    >>> v = t3m.tangent_randn(base)
    >>> z = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> print(cw.corewise_dot(z, apply_J(v)) - cw.corewise_dot(apply_Jt(z), v))
    -1.7763568394002505e-15
    '''
    num_cores = len(ztildes)
    assert(len(ww) == num_cores)

    (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores) = base

    (shape_weights,
     up_tucker_weights, outer_tucker_weights,
     left_tt_weights, right_tt_weights,
     ) = edge_weights

    weighted_ztildes = _apply_edge_weights(ztildes, shape_weights, xnp=xnp) if shape_weights is not None else ztildes
    weighted_ww = _apply_edge_weights(ww, shape_weights, xnp=xnp) if shape_weights is not None else ww

    weighted_xis = compute_weighted_xis(
        up_tucker_cores, weighted_ww,
        up_tucker_weights=up_tucker_weights, xnp=xnp,
    )

    weighted_mus = compute_weighted_mus(
        left_tt_cores, weighted_xis,
        left_tt_weights=left_tt_weights, xnp=xnp,
    )

    weighted_nus = compute_weighted_nus(
        right_tt_cores, weighted_xis,
        right_tt_weights=right_tt_weights, xnp=xnp,
    )

    weighted_etas = compute_weighted_etas(
        outer_tt_cores, weighted_mus, weighted_nus,
        outer_tucker_weights=outer_tucker_weights, xnp=xnp,
    )

    #

    weighted_deta_tildes = compute_weighted_deta_tildes(
        up_tucker_cores, weighted_ztildes,
        up_tucker_weights=up_tucker_weights, xnp=xnp,
    )

    weighted_tau_tildes = compute_weighted_tau_tildes(
        weighted_deta_tildes, left_tt_cores, weighted_xis, weighted_mus,
        left_tt_weights=left_tt_weights, xnp=xnp,
    )

    weighted_sigma_tildes = compute_weighted_sigma_tildes(
        weighted_deta_tildes, right_tt_cores, weighted_xis, weighted_nus,
        right_tt_weights=right_tt_weights, xnp=xnp,
    )

    weighted_dxi_tildes = compute_weighted_dxi_tildes(
        weighted_sigma_tildes, weighted_tau_tildes, outer_tt_cores, weighted_mus, weighted_nus,
        outer_tucker_weights=outer_tucker_weights, xnp=xnp,
    )

    #

    dU_tildes = assemble_tucker_variations(
        weighted_ztildes, weighted_dxi_tildes, weighted_ww, weighted_etas,
        sum_over_probes=sum_over_probes, xnp=xnp,
    )

    dG_tildes = assemble_tt_variations(
        weighted_sigma_tildes, weighted_tau_tildes, weighted_deta_tildes, weighted_xis, weighted_mus, weighted_nus,
        sum_over_probes=sum_over_probes, xnp=xnp,
    )

    return dU_tildes, dG_tildes


###############################################
##########    Probe dense tensor    ###########
###############################################

def probe_dense(
        vectors: typ.Sequence[NDArray],
        T: NDArray,
        xnp=np,
) -> typ.Tuple[NDArray]:
    """Probe a dense tensor.

    Parameters
    ----------
    T: NDArray
        Tensor to be probed. shape=(N1,...,Nd)
    vectors: typ.Sequence[NDArray]
        Probing input vectors.
        len=d.
        elm_shape=(Ni,) or elm_shape=(num_probes, Ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray]
        Probes.
        len=d.
        elm_shape=(Ni,) or elm_shape=(num_probes, Ni)

    Examples
    --------

    Probe with one set of vectors:

    >>> import numpy as np
    >>> import t3toolbox.probing as t3p
    >>> T = np.random.randn(10,11,12)
    >>> u0 = np.random.randn(10)
    >>> u1 = np.random.randn(11)
    >>> u2 = np.random.randn(12)
    >>> yy = t3p.probe_dense((u0,u1,u2),T)
    >>> y0 = np.einsum('ijk,j,k', T, u1, u2)
    >>> y1 = np.einsum('ijk,i,k', T, u0, u2)
    >>> y2 = np.einsum('ijk,i,j', T, u0, u1)
    >>> print(np.linalg.norm(yy[0] - y0))
    2.0928808318295785e-14
    >>> print(np.linalg.norm(yy[1] - y1))
    1.0841599276764049e-14
    >>> print(np.linalg.norm(yy[2] - y2))
    1.2970142174948615e-14

    Probe with two sets of vectors:

    >>> import numpy as np
    >>> import t3toolbox.t3p as t3p
    >>> T = np.random.randn(10,11,12)
    >>> u0, v0 = np.random.randn(10), np.random.randn(10)
    >>> u1, v1 = np.random.randn(11), np.random.randn(11)
    >>> u2, v2 = np.random.randn(12), np.random.randn(12)
    >>> uuu = [np.vstack([u0,v0]), np.vstack([u1,v1]), np.vstack([u2,v2])]
    >>> yyy = t3p.probe_dense(uuu,T)
    >>> yy_u = t3p.probe_dense((u0,u1,u2),T)
    >>> yy_v = t3p.probe_dense((v0,v1,v2),T)
    >>> print(np.linalg.norm(yy_u[0] - yyy[0][0,:]))
    0.0
    >>> print(np.linalg.norm(yy_u[1] - yyy[1][0,:]))
    0.0
    >>> print(np.linalg.norm(yy_u[2] - yyy[2][0,:]))
    0.0
    >>> print(np.linalg.norm(yy_v[0] - yyy[0][1,:]))
    0.0
    >>> print(np.linalg.norm(yy_v[1] - yyy[1][1,:]))
    0.0
    >>> print(np.linalg.norm(yy_v[2] - yyy[2][1,:]))
    0.0
    """
    num_cores = len(T.shape)
    assert(len(vectors) == num_cores)
    if len(vectors[0].shape) == 1:
        vectorized=False
        for ii in range(num_cores):
            assert (len(vectors[ii].shape) == 1)
    elif len(vectors[0].shape) == 2:
        vectorized=True
        for ii in range(num_cores):
            assert (len(vectors[ii].shape) == 2)
    else:
        raise RuntimeError(
            'Wrong vectors[ii] shape in probe_dense. Should be vector or matrix.\n'
            + 'vectors[0].shape=' + str(vectors[0].shape)
        )

    vectors = list(vectors)
    if vectorized == False:
        for ii in range(num_cores):
            vectors[ii] = vectors[ii].reshape((1,-1))

    vector_lengths = tuple([x.shape[1] for x in vectors])
    assert(vector_lengths == T.shape)

    probes = []
    for ii in range(num_cores):
        Ai = T
        for jj in range(ii):
            if jj == 0:
                Ai = xnp.einsum('pi,i...->p...', vectors[jj], Ai)
            else:
                Ai = xnp.einsum('pi,pi...->p...', vectors[jj], Ai)

        for jj in range(num_cores-1, ii, -1):
            if ii==0 and jj==num_cores-1:
                Ai = xnp.einsum('pi,...i->p...', vectors[jj], Ai)
            else:
                Ai = xnp.einsum('pi,p...i->p...', vectors[jj], Ai)
        probes.append(Ai)

    if not vectorized:
        probes = [Ai.reshape(-1) for Ai in probes]

    return tuple(probes)