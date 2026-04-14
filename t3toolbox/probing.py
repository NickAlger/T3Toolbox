# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.base_variation_format
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform as ut3
import t3toolbox.common as common
import t3toolbox.base_variation_format as bvf
from t3toolbox.common import *

__all__ = [
    # Probe a dense tensor
    'probe_dense',
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
]


#####################################################
########    Probing a Tucker Tensor Train    ########
#####################################################

def probe_t3(
        ww:             typ.Union[typ.Sequence[NDArray],    NDArray],
        x:              typ.Union[t3.TuckerTensorTrain,     ut3.UniformTuckerTensorTrainCores],
        edge_weights:   typ.Union[t3.EdgeWeights,           ut3.UniformEdgeWeights] = (None, None, None),
        use_jax: bool = False,
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
    >>> x = t3.absorb_edge_weights_into_cores(x0, edge_weights)
    >>> zz2 = t3p.probe_t3(ww, x)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [3.372228193172379e-14, 3.826148129405782e-14, 2.294115439089251e-14]

    For uniform T3:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.uniform as ut3
    >>> x = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> ww = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> uniform_x, masks = ut3.t3_to_ut3(x)
    >>> uniform_ww = ut3.pack_tensors(ww)
    >>> uniform_zz = t3p.probe_t3(uniform_ww, uniform_x, edge_weights=masks)
    >>> zz = t3p.probe_t3(ww, x)
    >>> uniform_zz2 = ut3.pack_tensors(zz)
    >>> print(np.linalg.norm(uniform_zz - uniform_zz2))
    0.0
    '''
    tucker_cores, tt_cores = x

    shape_weights, tucker_weights, tt_weights = edge_weights

    left_tt_weights     = tt_weights[:-1]   if tt_weights is not None else None
    right_tt_weights    = tt_weights[1:]    if tt_weights is not None else None

    weighted_ww = _apply_edge_weights(ww, shape_weights, use_jax=use_jax) if shape_weights is not None else ww

    xis = compute_xis(
        tucker_cores, weighted_ww, up_tucker_weights=tucker_weights, use_jax=use_jax,
    )

    mus = compute_mus(
        tt_cores, xis, left_tt_weights=left_tt_weights, use_jax=use_jax,
    )

    nus = compute_nus(
        tt_cores, xis, right_tt_weights=right_tt_weights, use_jax=use_jax,
    )

    etas = compute_etas(
        tt_cores, mus, nus, outer_tucker_weights=tucker_weights, use_jax=use_jax,
    )

    zs = assemble_zs(
        tucker_cores, etas, shape_weights=shape_weights, use_jax=use_jax,
    )

    return zs


def _apply_edge_weight(edge_variable, edge_weight, xnp=np):
    return xnp.einsum('...i,i->...i', edge_variable, edge_weight)


def _apply_edge_weights(edge_variables, edge_weights, use_jax: bool=False):
    is_ragged = isinstance(edge_variables, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    (weighted_edge_variables,) = xmap(
        lambda v_w: (_apply_edge_weight(v_w[0], v_w[1], xnp=xnp),),
        (edge_variables, edge_weights)
    )
    return weighted_edge_variables


def compute_xis(
        up_tucker_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nUi,Ni)
        ww: typ.Sequence[NDArray], # len=d. elm_shape=(...,Ni)
        up_tucker_weights: typ.Sequence[NDArray] = None, # len=d, elm_shape=(nUi,)
        use_jax: bool = False,
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
    is_ragged = isinstance(ww, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    xi_weights = up_tucker_weights

    def _func(x):
        U, w, ind = x[0], x[1], 2

        unweighted_xi = xnp.einsum('io,...o->...i', U, w)

        if xi_weights is not None:
            weight = x[ind]
            xi = _apply_edge_weight(unweighted_xi, weight, xnp=xnp)
        else:
            xi = unweighted_xi

        return (xi,)

    xs = (up_tucker_cores, ww)
    xs = xs + (xi_weights,) if xi_weights  is not None else xs

    (xis,) = xmap(_func, xs)
    return xis


def compute_mus(
        left_tt_cores: typ.Sequence[NDArray], # len=d-1. elm_shape=(rLi,nUi,rL(i+1))
        xis: typ.Sequence[NDArray], # len=d. elm_shape=(...,nUi)
        left_tt_weights: typ.Sequence[NDArray] = None, # len=d, elm_shape=(rLi,)
        use_jax: bool = False,
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
    is_ragged = isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    mu_weights = left_tt_weights

    def _func(unweighted_mu, x):
        P, xi, ind = x[0], x[1], 2

        if mu_weights is not None:
            weight = x[ind]
            mu = _apply_edge_weight(unweighted_mu, weight, xnp=xnp)
        else:
            mu = unweighted_mu

        unweighted_mu_next = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', mu, P),
            xi,
        )
        return unweighted_mu_next, (mu,)

    r0 = left_tt_cores[0].shape[0]
    vectorization_shape = xis[0].shape[:-1]
    init = xnp.ones(vectorization_shape + (r0,))

    xs = (left_tt_cores, xis)
    xs = xs + (mu_weights,) if mu_weights is not None else xs

    last_mu, (mus,) = xscan(_func, init, xs)
    return mus


def compute_nus(
        right_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(rRi,nUi,rR(i+1))
        xis, # len=d. elm_shape=(...,nUi)
        right_tt_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(rRi,)
        use_jax: bool = False,
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
    rev_tt_cores = t3.reverse_tt(right_tt_cores)
    rev_xis = xis[::-1]
    rev_right_tt_weights  = None if right_tt_weights is None else right_tt_weights[::-1]

    rev_nus = compute_mus(
        rev_tt_cores,
        rev_xis,
        left_tt_weights=rev_right_tt_weights,
        use_jax=use_jax
    )
    nus = rev_nus[::-1]
    return nus


def compute_etas(
        outer_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(rLi,nOi,rR(i+1))
        mus, # len=d. elm_shape=(...,rLi)
        nus, # len=d. elm_shape=(...,rR(i+1))
        outer_tucker_weights: typ.Sequence[NDArray] = None, # len=d, elm_shape=(nOi)
        use_jax: bool = False,
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
    is_ragged = isinstance(mus, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    eta_weights = outer_tucker_weights

    def _func(x):
        mu, G, nu, ind = x[0], x[1], x[2], 3

        unweighted_eta = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', mu, G),
            nu,
        )

        if eta_weights is not None:
            weight = x[ind]
            eta = _apply_edge_weight(unweighted_eta, weight, xnp=xnp)
        else:
            eta = unweighted_eta

        return (eta,)

    xs = (mus, outer_tt_cores, nus)
    xs = xs + (eta_weights,) if eta_weights is not None else xs

    (etas,) = xmap(_func, xs)
    return etas


def assemble_zs(
        tucker_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(ni,Ni)
        etas,  # len=d. elm_shape=(...,ni)
        shape_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(Ni,)
        use_jax: bool = False,
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
    is_ragged = isinstance(etas, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    z_weights = shape_weights

    def _func(x):
        eta, U, ind = x[0], x[1], 2

        unweighted_z = xnp.einsum('...a,ao->...o', eta, U)

        if z_weights is not None:
            weight = x[ind]
            z = _apply_edge_weight(unweighted_z, weight, xnp=xnp)
        else:
            z = unweighted_z

        return (z,)

    xs = (etas, tucker_cores)
    xs = xs + (z_weights,) if z_weights is not None else xs

    (zs,) = xmap(_func, xs)
    return zs


#####################################################
###########    Probing a tangent vector    ##########
#####################################################

def compute_dxis(
        var_tucker_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        ww: typ.Sequence[NDArray], # len=d. elm_shape=(...,Ni)
        outer_tucker_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nOi,)
        use_jax: bool = False,
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
    return compute_xis(
        var_tucker_cores, ww, up_tucker_weights=outer_tucker_weights, use_jax=use_jax,
    )


def compute_sigmas(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nUi),
        dxis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nOi)
        mus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nLi)
        right_tt_weights: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rRi,)
        use_jax: bool = False,
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
    is_ragged = isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    sigma_weights = right_tt_weights

    def _func(weighted_sigma, x):
        Q, O, dG, xi, dxi, mu, ind = x[0], x[1], x[2], x[3], x[4], x[5], 6

        sigma_next_t1 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', weighted_sigma, Q),
            xi
        )
        sigma_next_t2 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', mu, dG),
            xi
        )
        sigma_next_t3 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', mu, O),
            dxi
        )
        unweighted_sigma_next = sigma_next_t1 + sigma_next_t2 + sigma_next_t3

        if sigma_weights is not None:
            weight = x[ind]
            sigma_next = _apply_edge_weight(unweighted_sigma_next, weight, xnp=xnp)
        else:
            sigma_next = unweighted_sigma_next

        return sigma_next, (weighted_sigma,)

    rR0 = right_tt_cores[0].shape[0]
    vectorization_shape = xis[0].shape[:-1]
    init = xnp.zeros(vectorization_shape + (rR0,))

    xs = (right_tt_cores, outer_tt_cores, var_tt_cores, xis, dxis, mus)
    xs = xs + (sigma_weights,)    if sigma_weights  is not None else xs

    last_sigma, (sigmas,) = xscan(_func, init, xs)
    return sigmas


def compute_taus(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        left_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rL(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nUi),
        dxis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nOi)
        nus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nR(i+1))
        left_tt_weights: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rLi,)
        use_jax: bool = False,
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
    rev_var_tt_cores    = t3.reverse_tt(var_tt_cores)
    rev_left_tt_cores   = t3.reverse_tt(left_tt_cores)
    rev_outer_tt_cores  = t3.reverse_tt(outer_tt_cores)
    rev_xis    = xis[::-1]
    rev_dxis   = dxis[::-1]
    rev_nus    = nus[::-1]
    rev_left_tt_weights = None if left_tt_weights is None else left_tt_weights[::-1]

    rev_taus = compute_sigmas(
        rev_var_tt_cores, rev_left_tt_cores, rev_outer_tt_cores,
        rev_xis, rev_dxis, rev_nus,
        right_tt_weights=rev_left_tt_weights, use_jax=use_jax,
    )
    taus = rev_taus[::-1]
    return taus


def compute_detas(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        left_tt_cores: typ.Sequence[NDArray],  # len=d, elm_shape=(rLi,nUi,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        mus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nLi)
        nus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nRi)
        sigmas: typ.Sequence[NDArray], # len=d, elm_shape=(...,rRi)
        taus: typ.Sequence[NDArray], # len=d, elm_shape=(...,rL(i+1))
        up_tucker_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nUi,)
        use_jax: bool = False,
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
    is_ragged = isinstance(mus, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    deta_weights  = up_tucker_weights

    def _func(x):
        (P, Q, dG, mu, nu, sigma, tau, ind
         ) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6], 7)

        deta_term1 = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', sigma, Q),
            nu,
        )
        deta_term2 = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', mu, dG),
            nu,
        )
        deta_term3 = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', mu, P),
            tau,
        )
        unweighted_deta = deta_term1 + deta_term2 + deta_term3

        if deta_weights is not None:
            weight = x[ind]
            deta = _apply_edge_weight(unweighted_deta, weight, xnp=xnp)
        else:
            deta = unweighted_deta

        return (deta,)

    xs = (left_tt_cores, right_tt_cores, var_tt_cores, mus, nus, sigmas, taus)
    xs = xs + (deta_weights,)     if deta_weights   is not None else xs

    detas_tuple = xmap(_func, xs)
    return detas_tuple[0]


def assemble_tangent_zs(
        tucker_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(nUi,Ni)
        var_tucker_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        etas: typ.Sequence[NDArray], # etas. len=d, elm_shape=(...,nUi)
        detas: typ.Sequence[NDArray], # detas. len=d, elm_shape=(...,nUi)
        shape_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(Ni,)
        use_jax: bool = False,
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
    is_ragged = isinstance(etas, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    z_weights = shape_weights

    def _func(x):
        B, dB, eta, deta, ind = x[0], x[1], x[2], x[3], 4

        z_term1 = xnp.einsum('ao,...a->...o', B, deta)
        z_term2 = xnp.einsum('ao,...a->...o', dB, eta)
        unweighted_z = z_term1 + z_term2

        if z_weights is not None:
            weight = x[ind]
            z = _apply_edge_weight(unweighted_z, weight, xnp=xnp)
        else:
            z = unweighted_z

        return (z,)

    xs = (tucker_cores, var_tucker_cores, etas, detas)
    xs = xs + (z_weights,) if z_weights  is not None else xs

    (zs,) = xmap(_func, xs)
    return zs


def probe_tangent(
        ww:         typ.Union[typ.Sequence[NDArray],    NDArray],  # input vectors, len=d, elm_shape=(...,Ni)
        variation:  typ.Union[bvf.T3Variation,          NDArray], # tucker_var_shapes=(nOi,Ni), tt_var_shapes=tt_hole_shapes=(rLi,ni,rRi)
        base:       typ.Union[bvf.T3Base,               ut3.UniformT3Base], # tucker_hole_shapes=(nOi,Ni), tt_hole_shapes=(rLi,ni,rRi)
        edge_weights: bvf.BVEdgeWeights = (None, None, None, None, None),
        use_jax: bool = False,
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
    >>> weighted_variation, weighted_base = t3m.absorb_weights_into_tangent_cores(variation, base, edge_weights)
    >>> zzz2 = t3p.probe_tangent(www, weighted_variation, weighted_base)
    >>> print([np.linalg.norm(zz - zz2) for zz, zz2 in zip(zzz, zzz2)])
    [1.5683512051190777e-15, 4.368484248906507e-15, 1.855735793037041e-15]

    Uniform tangent

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> www = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> uniform_variation, uniform_base, masks = ut3.bv_to_ubv(variation, base)
    >>> uniform_www = ut3.pack_tensors(www)
    >>> uniform_zzz = t3p.probe_tangent(uniform_www, uniform_variation, uniform_base, edge_weights=masks)
    >>> zzz2 = t3p.probe_tangent(www, variation, base)
    >>> uniform_zzz2 = ut3.pack_tensors(zzz2)
    >>> print(np.linalg.norm(uniform_zzz - uniform_zzz2))
    5.3316719684500096e-15
    '''
    (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores) = base
    (var_tucker_cores, var_tt_cores) = variation

    (shape_weights,
     up_tucker_weights, outer_tucker_weights,
     left_tt_weights, right_tt_weights,
     ) = edge_weights

    weighted_ww = _apply_edge_weights(ww, shape_weights, use_jax=use_jax) if shape_weights is not None else ww

    xis = compute_xis(
        up_tucker_cores, weighted_ww, up_tucker_weights=up_tucker_weights, use_jax=use_jax,
    )

    mus = compute_mus(
        left_tt_cores, xis, left_tt_weights=left_tt_weights, use_jax=use_jax,
    )

    nus = compute_nus(
        right_tt_cores, xis, right_tt_weights=right_tt_weights, use_jax=use_jax,
    )

    etas = compute_etas(
        outer_tt_cores, mus, nus, outer_tucker_weights=outer_tucker_weights, use_jax=use_jax,
    )

    dxis = compute_dxis(
        var_tucker_cores, weighted_ww,
        outer_tucker_weights=outer_tucker_weights, use_jax=use_jax
    )

    sigmas = compute_sigmas(
        var_tt_cores, right_tt_cores, outer_tt_cores, xis, dxis, mus,
        right_tt_weights=right_tt_weights, use_jax=use_jax,
    )

    taus = compute_taus(
        var_tt_cores, left_tt_cores, outer_tt_cores, xis, dxis, nus,
        left_tt_weights=left_tt_weights, use_jax=use_jax,
    )

    detas = compute_detas(
        var_tt_cores, left_tt_cores, right_tt_cores, mus, nus, sigmas, taus,
        up_tucker_weights=up_tucker_weights, use_jax=use_jax,
    )

    zz = assemble_tangent_zs(
        up_tucker_cores, var_tucker_cores, etas, detas,
        shape_weights=shape_weights, use_jax=use_jax,
    )

    return zz


###############################################################
###########    Transpose of tangent to probes map    ##########
###############################################################

def compute_deta_tildes(
        up_tucker_cores: typ.Sequence[NDArray],  # len=d, elm_shape=(ni,Ni)
        ztildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,Ni)
        up_tucker_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nUi,)
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(...,ni)
    '''Adjoint-var-upward edge variables deta_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (43), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    return compute_xis(
        up_tucker_cores, ztildes,
        up_tucker_weights=up_tucker_weights, use_jax=use_jax,
    )


def compute_tau_tildes(
        deta_tildes,  # len=d+1, elm_shape=(...,ni)
        left_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rL(i+d))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,ni)
        mus, # len=d, elm_shape=(...,rLi)
        left_tt_weights: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rLi,)
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(...,rLi)
    '''Adjoint-var-rightward edge variables tau_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (44), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    is_ragged = isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    tau_tilde_weights = left_tt_weights

    def _func(unweighted_tau_tilde, x):
        P, xi, deta_tilde, mu, ind = x[0], x[1], x[2], x[3], 4

        if tau_tilde_weights is not None:
            weight = x[ind]
            tau_tilde = _apply_edge_weight(unweighted_tau_tilde, weight, xnp=xnp)
        else:
            tau_tilde = unweighted_tau_tilde

        tau_tilde_next_t1 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', tau_tilde, P),
            xi
        )
        tau_tilde_next_t2 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', mu, P),
            deta_tilde
        )
        tau_tilde_next = tau_tilde_next_t1 + tau_tilde_next_t2

        return tau_tilde_next, (tau_tilde,)

    init = xnp.zeros(mus[0].shape[:-1] + (left_tt_cores[0].shape[0],))
    xs = (left_tt_cores, xis, deta_tildes, mus)
    xs = xs + (tau_tilde_weights,) if tau_tilde_weights is not None else xs

    last_tau_tilde, (tau_tildes,) = xscan(_func, init, xs)
    return tau_tildes


def compute_sigma_tildes(
        deta_tildes,  # len=d, elm_shape=(...,ni)
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,ni,rR(i+d))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,ni)
        nus, # len=d, elm_shape=(...,rR(i+1))
        right_tt_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(rRi,)
        use_jax: bool = False,
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

    return compute_tau_tildes(
        deta_tildes[::-1], t3.reverse_tt(right_tt_cores), xis[::-1], nus[::-1],
        left_tt_weights = rev_right_tt_weights, use_jax=use_jax,
    )[::-1]


def compute_dxi_tildes(
        sigma_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rR(i+1))
        tau_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rLi)
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        mus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rLi)
        nus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rR(i+1))
        outer_tucker_weights: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nOi,)
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]: # dxi_tildes. len=d, elm_shape=(...,nOi)
    '''Adjoint-var-downward edge variables dxi_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (46), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    is_ragged = isinstance(mus, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    dxi_tilde_weights = outer_tucker_weights

    def _func(x):
        O, mu, nu, st, tt, ind = x[0], x[1], x[2], x[3], x[4], 5

        unweighted_dxi_tilde = (
                xnp.einsum(
                    '...aj,...j->...a',
                    xnp.einsum('...i,iaj->...aj',tt, O),
                    nu
                )
                +
                xnp.einsum(
                    '...aj,...j->...a',
                    xnp.einsum('...i,iaj->...aj', mu, O),
                    st
                )
        )

        if dxi_tilde_weights is not None:
            weight = x[ind]
            dxi_tilde = _apply_edge_weight(unweighted_dxi_tilde, weight, xnp=xnp)
        else:
            dxi_tilde = unweighted_dxi_tilde

        return (dxi_tilde,)

    xs = (outer_tt_cores, mus, nus, sigma_tildes, tau_tildes)
    xs = xs + (outer_tucker_weights,) if outer_tucker_weights is not None else xs

    (dxi_tildes,) = xmap(_func, xs)
    return dxi_tildes


def assemble_tucker_variations(
        ztildes: typ.Sequence[NDArray], # len=d, elm_shape=(...,Ni)
        dxi_tildes: typ.Sequence[NDArray], #len=d, elm_shape=(...,nOi)
        ww: typ.Sequence[NDArray],  # input vectors, len=d, elm_shape=(Ni,) or (...,Ni)
        etas: typ.Sequence[NDArray],  # etas. len=d, elm_shape=(...,ni)
        sum_over_probes: bool = False,
        use_jax: bool = False,
):
    '''Assemble Tucker core variations, delta_U_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (47), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    is_ragged = isinstance(ww, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    def _func(x):
        z_tilde, eta, w, dxi_tilde = x
        if sum_over_probes:
            dU_tilde = (
                    xnp.einsum('...o,...a->ao', z_tilde, eta)
                    +
                    xnp.einsum('...o,...a->ao', w, dxi_tilde)
            )
        else:
            dU_tilde = (
                    xnp.einsum('...o,...a->...ao', z_tilde, eta)
                    +
                    xnp.einsum('...o,...a->...ao', w, dxi_tilde)
            )
        return (dU_tilde,)

    (dU_tildes,) = xmap(_func, (ztildes, etas, ww, dxi_tildes))
    return dU_tildes


def assemble_tt_variations(
        sigma_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rR(i+1))
        tau_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rLi)
        deta_tildes,  # len=d+1, elm_shape=(...,ni)
        xis: typ.Sequence[NDArray],  # len=d, elm_shape=(...,ni)
        mus,  # len=d, elm_shape=(...,rLi)
        nus,  # len=d, elm_shape=(...,rR(i+1))
        sum_over_probes: bool = False,
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(...,rLi,nOi,rRi)
    '''Assemble TT core variations, delta_G_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (48), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    is_ragged = isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_ragged, use_jax)

    #

    def _func(x):
        xi, mu, nu, sigma_tilde, tau_tilde, deta_tilde = x
        if sum_over_probes:
            dG_tilde = (
                    xnp.einsum(
                        '...ia,...j->iaj',
                        xnp.einsum('...i,...a->...ia', mu, xi),
                        sigma_tilde
                    )
                    +
                    xnp.einsum(
                        '...ia,...j->iaj',
                        xnp.einsum('...i,...a->...ia', tau_tilde, xi),
                        nu
                    )
                    +
                    xnp.einsum(
                        '...ia,...j->iaj',
                        xnp.einsum('...i,...a->...ia', mu, deta_tilde),
                        nu
                    )
            )
        else:
            dG_tilde = (
                    xnp.einsum(
                        '...ia,...j->...iaj',
                        xnp.einsum('...i,...a->...ia', mu, xi),
                        sigma_tilde
                    )
                    +
                    xnp.einsum(
                        '...ia,...j->...iaj',
                        xnp.einsum('...i,...a->...ia', tau_tilde, xi),
                        nu
                    )
                    +
                    xnp.einsum(
                        '...ia,...j->...iaj',
                        xnp.einsum('...i,...a->...ia', mu, deta_tilde),
                        nu
                    )
            )
        return (dG_tilde,)

    xs = (xis, mus, nus, sigma_tildes, tau_tildes, deta_tildes)
    (dG_tildes,) = xmap(_func, xs)
    return dG_tildes


def probe_tangent_transpose(
        ztildes:        typ.Union[typ.Sequence[NDArray],    NDArray], # len=d, elm_shape=(...,Ni) OR shape=(d,...,Ni)
        ww:             typ.Union[typ.Sequence[NDArray],    NDArray], # input vectors, len=d, elm_shape=(...,Ni) OR shape=(d,...,Ni)
        base:           typ.Union[bvf.T3Base,               ut3.UniformT3Base], # tucker_hole_shapes=(nOi,Ni), tt_hole_shapes=(rLi,ni,rRi)
        edge_weights:   typ.Union[bvf.BVEdgeWeights,        ut3.UniformEdgeWeights] = (None, None, None, None, None),
        sum_over_probes: bool = False,
        use_jax: bool = False,
) -> typ.Union[
    typ.Tuple[
        typ.Tuple[NDArray,...], # vectorized dU_tildes. len=d, elm_shape=(..., nOi, Ni)
        typ.Tuple[NDArray,...], # vectorized dG_tildes. len=d, elm_shape=(..., rLi, ni, rRi)
    ],
    typ.Tuple[
        NDArray,  # vectorized dU_tildes. shape=(d, ..., nOi, Ni)
        NDArray,  # vectorized dG_tildes. shape=(d, ..., rLi, ni, rRi)
    ],
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
    >>> import t3toolbox.base_variation_format as bvf
    >>> randn = np.random.randn
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> NN, nnU, nnO, rrL, rrR = bvf.get_base_structure(base)
    >>> shape_weights = [randn(N) for N in NN]
    >>> up_tucker_weights = [randn(nU) for nU in nnU]
    >>> outer_tucker_weights = [randn(nO) for nO in nnO]
    >>> left_tt_weights = [randn(rL) for rL in rrL[:-1]]
    >>> right_tt_weights = [randn(rR) for rR in rrR[1:]]
    >>> edge_weights = (shape_weights, up_tucker_weights, outer_tucker_weights, left_tt_weights, right_tt_weights)
    >>> ww = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> apply_J = lambda v: t3p.probe_tangent(ww, v, base, edge_weights=edge_weights)
    >>> apply_Jt = lambda z: t3p.probe_tangent_transpose(z, ww, base, edge_weights=edge_weights)
    >>> v = t3m.tangent_randn(base)
    >>> z = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> print(cw.corewise_dot(z, apply_J(v)) - cw.corewise_dot(apply_Jt(z), v))
    -1.7763568394002505e-15

    Probe uniform T3

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.orthogonalization as orth
    >>> import t3toolbox.corewise as cw
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> www = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> V, B, masks = ut3.bv_to_ubv(variation, base)
    >>> uniform_ww = ut3.pack_tensors(www)
    >>> apply_J = lambda v: t3p.probe_tangent(uniform_ww, v, B, edge_weights=masks)
    >>> apply_Jt = lambda z: t3p.probe_tangent_transpose(z, uniform_ww, B, edge_weights=masks)
    >>> z = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> Z = ut3.pack_tensors(z)
    >>> JV = apply_J(V)
    >>> JTZ = apply_Jt(Z)
    >>> t0a = cw.corewise_dot([x[0,:] for x in Z], [x[0,:] for x in JV])
    >>> t0b = cw.corewise_dot([x[:,0,:,:] for x in JTZ], V)
    >>> print(t0a - t0b)
    -7.105427357601002e-15
    >>> t1a = cw.corewise_dot([x[1,:] for x in Z], [x[1,:] for x in JV])
    >>> t1b = cw.corewise_dot([x[:,1,:,:] for x in JTZ], V)
    >>> print(t1a - t1b)
    -5.329070518200751e-15
    '''
    (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores) = base

    (shape_weights,
     up_tucker_weights, outer_tucker_weights,
     left_tt_weights, right_tt_weights,
     ) = edge_weights

    weighted_ztildes = _apply_edge_weights(ztildes, shape_weights, use_jax=use_jax) if shape_weights is not None else ztildes
    weighted_ww = _apply_edge_weights(ww, shape_weights, use_jax=use_jax) if shape_weights is not None else ww

    xis = compute_xis(
        up_tucker_cores, weighted_ww,
        up_tucker_weights=up_tucker_weights, use_jax=use_jax,
    )

    mus = compute_mus(
        left_tt_cores, xis,
        left_tt_weights=left_tt_weights, use_jax=use_jax,
    )

    nus = compute_nus(
        right_tt_cores, xis,
        right_tt_weights=right_tt_weights, use_jax=use_jax,
    )

    etas = compute_etas(
        outer_tt_cores, mus, nus,
        outer_tucker_weights=outer_tucker_weights, use_jax=use_jax,
    )

    #

    deta_tildes = compute_deta_tildes(
        up_tucker_cores, weighted_ztildes,
        up_tucker_weights=up_tucker_weights, use_jax=use_jax,
    )

    tau_tildes = compute_tau_tildes(
        deta_tildes, left_tt_cores, xis, mus,
        left_tt_weights=left_tt_weights, use_jax=use_jax,
    )

    sigma_tildes = compute_sigma_tildes(
        deta_tildes, right_tt_cores, xis, nus,
        right_tt_weights=right_tt_weights, use_jax=use_jax,
    )

    dxi_tildes = compute_dxi_tildes(
        sigma_tildes, tau_tildes, outer_tt_cores, mus, nus,
        outer_tucker_weights=outer_tucker_weights, use_jax=use_jax,
    )

    #

    dU_tildes = assemble_tucker_variations(
        weighted_ztildes, dxi_tildes, weighted_ww, etas,
        sum_over_probes=sum_over_probes, use_jax=use_jax,
    )

    dG_tildes = assemble_tt_variations(
        sigma_tildes, tau_tildes, deta_tildes, xis, mus, nus,
        sum_over_probes=sum_over_probes, use_jax=use_jax,
    )

    return dU_tildes, dG_tildes


###############################################
##########    Probe dense tensor    ###########
###############################################

def probe_dense(
        vectors: typ.Sequence[NDArray],
        T: NDArray,
        use_jax: bool = False,
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
    xnp, _, _ = get_backend(False, use_jax)

    #

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