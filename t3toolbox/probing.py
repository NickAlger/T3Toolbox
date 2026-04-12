# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.base_variation_format
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.common as common

__all__ = [
    # Probe a dense tensor
    'probe_dense',
    # Probe a Tucker tensor train
    'probe_t3',
    'compute_xis',
    'compute_mus',
    'compute_nus',
    'compute_etas',
    'assemble_probes',
    # Probe a tangent vector
    'probe_tangent',
    'compute_dxis',
    'compute_sigmas',
    'compute_taus',
    'compute_detas',
    'assemble_tangent_probes',
    # Transpose of map from tangent vector to probes
    'compute_deta_tildes',
    'compute_tau_tildes',
    'compute_sigma_tildes',
    'compute_dxi_tildes',
    'assemble_tucker_variations',
    'assemble_tt_variations',
    'probe_tangent_transpose',
]

NDArray = typ.TypeVar('NDArray') # Generic stand-in for np.ndarray, jnp.ndarray, or other array backend


###############################################
##########    Probe dense tensor    ###########
###############################################

def probe_dense(
        T:          NDArray,
        vectors:    typ.Sequence[NDArray], # elm_shape=(...,Ni)
        xnp = np,
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
    >>> yy = t3p.probe_dense(T, (u0,u1,u2))
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
    >>> yyy = t3p.probe_dense(T, uuu)
    >>> yy_u = t3p.probe_dense(T, (u0,u1,u2))
    >>> yy_v = t3p.probe_dense(T, (v0,v1,v2))
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


#####################################################
########    Probing a Tucker Tensor Train    ########
#####################################################

def probe_t3(
        x:  t3.TuckerTensorTrain, # structure=((N1,...,Nd),(n1,...,nd),(1,r1,...,r(d-1),1))
        ww: typ.Sequence[NDArray], # input vectors, len=d, elm_shape=(...,Ni)
        xnp = np,
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
    >>> zz = t3p.probe_t3(x, ww)
    >>> x_dense = t3.t3_to_dense(x)
    >>> zz2 = t3p.probe_dense(x_dense, ww)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [1.0259410400851746e-12, 1.0909087370186656e-12, 3.620283224238675e-13]
    '''
    shape = t3.get_structure(x)[0]
    assert(len(ww) == len(shape))

    tucker_cores, tt_cores = x

    for B, w in zip(tucker_cores, ww):
        assert(B.shape[1] == w.shape[-1])

    xis = compute_xis(tucker_cores, ww, xnp=xnp)
    mus = compute_mus(tt_cores, xis, xnp=xnp)
    nus = compute_nus(tt_cores, xis, xnp=xnp)
    etas = compute_etas(tt_cores, mus, nus, xnp=xnp)
    zz = assemble_probes(tucker_cores, etas, xnp=xnp)

    return zz


def _apply_edge_mask(edge_variable, edge_mask, xnp=np):
    return xnp.einsum('...i,i->...i', edge_variable, edge_mask)


def compute_xis(
        tucker_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ni,Ni)
        ww: typ.Sequence[NDArray], # len=d. elm_shape=(...,Ni)
        shape_masks: typ.Sequence[NDArray] = None, # len=d, elm_shape=(Ni,)
        tucker_masks: typ.Sequence[NDArray] = None, # len=d, elm_shape=(ni,)
        map = common.ragged_map,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # xis. len=d, elm_shape=(...,ni)
    '''Compute upward edge variables associated with edges between Tucker cores and adjacent TT-cores.
    Used for probing a Tucker tensor train.

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
    ww: typ.Sequence[NDArray]
        input vectors to probe with. len=d, elm_shape=(...,Ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        upward edge variables xi. len=d, elm_shape=(...,ni)

    See Also
    --------
    probe_t3
    compute_mus
    compute_nus
    compute_etas
    assemble_probes
    '''
    def _func(U_w_masks):
        U, w = U_w_masks[0], U_w_masks[1]

        if shape_masks is not None:
            shape_mask = U_w_masks[2]
            w = _apply_edge_mask(w, shape_mask)

        xi = xnp.einsum('io,...o->...i', U, w)

        if tucker_masks is not None:
            tucker_mask = U_w_masks[-1]
            xi = _apply_edge_mask(xi, tucker_mask)

        return (xi,)

    xs = (tucker_cores, ww)
    if shape_masks is not None:
        xs = xs + (shape_masks,)
    if tucker_masks is not None:
        xs = xs + (tucker_masks,)

    (xis,) = map(_func, xs)
    return xis


def compute_mus(
        left_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ri,ni,r(i+1))
        xis: typ.Sequence[NDArray], # len=d. elm_shape=(...,ni)
        tucker_masks: typ.Sequence[NDArray] = None, # len=d, elm_shape=(ni,)
        tt_masks: typ.Sequence[NDArray] = None, # len=d+1, elm_shape=(ri,)
        scan = common.ragged_scan,
        xnp = np,
) -> typ.Sequence[NDArray]: # mus. len=d, elm_shape=(...,ri)
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
        len=d. elm_shape=(ri,ni,r(i+1))
    xis: typ.Sequence[NDArray]
        upward edge variables xi. len=d. elm_shape=(...,ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        leftward edge variables mu. len=d, elm_shape=(...,ri)

    See Also
    --------
    probe_t3
    compute_xis
    compute_nus
    compute_etas
    assemble_probes
    '''
    def _func(mu, P_xi_masks):
        P, xi = P_xi_masks[0], P_xi_masks[1]

        if tucker_masks is not None:
            tucker_mask = P_xi_masks[2]
            xi = _apply_edge_mask(xi, tucker_mask)

        if tt_masks is not None:
            tt_mask = P_xi_masks[-1]
            mu = _apply_edge_mask(mu, tt_mask)

        mu_next = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', mu, P),
            xi,
        )
        return mu_next, (mu,)

    r0 = left_tt_cores[0].shape[0]
    vectorization_shape = xis[0].shape[:-1]
    init = xnp.ones(vectorization_shape + (r0,))

    xs = (left_tt_cores, xis)
    if tucker_masks is not None:
        xs = xs + (tucker_masks,)
    if tt_masks is not None:
        xs = xs + (tt_masks[:-1],)

    last_mu, (mus,) = scan(_func, init, xs)
    return mus


def tt_reverse(tt_cores):
    return tuple([G.swapaxes(0, 2) for G in tt_cores[::-1]])


def compute_nus(
        right_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ri,ni,r(i+1))
        xis, # len=d. elm_shape=(...,ni)
        tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(ni,)
        tt_masks: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(ri,)
        xnp = np,
) -> typ.Sequence[NDArray]: # nus. len=d, elm_shape=(...,r(i+1))
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
        len=d. elm_shape=(ri,ni,r(i+1))
    xis: typ.Sequence[NDArray]
        upward edge variables xi. len=d. elm_shape=(...,ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        rightward edge variables nu. len=d, elm_shape=(...,r(i+1))

    See Also
    --------
    probe_t3
    compute_xis
    compute_mus
    compute_etas
    assemble_probes
    '''
    rev_tt_cores = tt_reverse(right_tt_cores)
    rev_xis = xis[::-1]
    rev_tucker_masks    = None if tucker_masks  is None else tucker_masks[::-1]
    rev_tt_masks        = None if tt_masks      is None else tt_masks[::-1]

    rev_nus = compute_mus(
        rev_tt_cores,
        rev_xis,
        tucker_masks=rev_tucker_masks,
        tt_masks=rev_tt_masks,
        xnp=xnp,
    )
    nus = rev_nus[::-1]
    return nus


def compute_etas(
        outer_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ri,ni,r(i+1))
        mus, # len=d. elm_shape=(...,ri)
        nus, # len=d. elm_shape=(...,r(i+1))
        tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(ni,)
        tt_masks: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(ri,)
        map = common.ragged_map,
        xnp = np,
) -> typ.Sequence[NDArray]: # etas. len=d, elm_shape=(...,ni)
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
    def _func(mu_G_nu_masks):
        mu, G, nu = mu_G_nu_masks[0], mu_G_nu_masks[1], mu_G_nu_masks[2]

        if tt_masks is not None:
            left_tt_mask = mu_G_nu_masks[3]
            right_tt_mask = mu_G_nu_masks[4]
            mu = _apply_edge_mask(mu, left_tt_mask)
            nu = _apply_edge_mask(nu, right_tt_mask)

        eta = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', mu, G),
            nu,
        )

        if tucker_masks is not None:
            tucker_mask = mu_G_nu_masks[-1]
            eta = _apply_edge_mask(eta, tucker_mask)

        return (eta,)

    xs = (mus, outer_tt_cores, nus)
    if tt_masks is not None:
        xs = xs + (tt_masks[:-1], tt_masks[1:])
    if tucker_masks is not None:
        xs = xs + (tucker_masks,)

    (etas,) = map(_func, xs)
    return etas


def assemble_probes(
        tucker_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(ni,Ni)
        etas,  # len=d. elm_shape=(...,ni)
        shape_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(Ni,)
        tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(ni,)
        map = common.ragged_map,
        xnp = np,
) -> typ.Sequence[NDArray]: # zz. len=d, elm_shape=(...,Ni)
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
    def _func(eta_U_masks):
        eta, U = eta_U_masks[0], eta_U_masks[1]

        if tucker_masks is not None:
            tucker_mask = eta_U_masks[2]
            eta = _apply_edge_mask(eta, tucker_mask)

        z = xnp.einsum('...a,ao->...o', eta, U)

        if shape_masks is not None:
            shape_mask = eta_U_masks[-1]
            z = _apply_edge_mask(z, shape_mask)

        return (z,)

    xs = (etas, tucker_cores)
    if tucker_masks is not None:
        xs = xs + (tucker_masks,)
    if shape_masks is not None:
        xs = xs + (shape_masks,)

    (zz,) = map(_func, xs)
    return zz


#####################################################
###########    Probing a tangent vector    ##########
#####################################################

def compute_dxis(
        var_tucker_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        ww: typ.Sequence[NDArray], # len=d. elm_shape=(...,Ni)
        shape_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(Ni,)
        outer_tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nOi,)
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
    return compute_xis(var_tucker_cores, ww, shape_masks, outer_tucker_masks, xnp=xnp)


def compute_sigmas(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nUi),
        dxis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nOi)
        mus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nLi)
        up_tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nUi,)
        outer_tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nOi,)
        left_tt_masks: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rLi,)
        right_tt_masks: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rRi,)
        scan = common.ragged_scan,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # sigmas. len=d, elm_shape=(...,rR(i+1))
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
    xi_masks    = up_tucker_masks
    dxi_masks   = outer_tucker_masks
    mu_masks    = left_tt_masks[:-1]    if left_tt_masks    is not None else None
    sigma_masks = right_tt_masks[:-1]   if right_tt_masks   is not None else None  # Yes, [:-1]. init sigma is zero.

    def _func(sigma, Q_O_dG_xi_dxi_mu_masks):
        Q = Q_O_dG_xi_dxi_mu_masks[0]
        O = Q_O_dG_xi_dxi_mu_masks[1]
        dG = Q_O_dG_xi_dxi_mu_masks[2]
        xi = Q_O_dG_xi_dxi_mu_masks[3]
        dxi = Q_O_dG_xi_dxi_mu_masks[4]
        mu = Q_O_dG_xi_dxi_mu_masks[5]

        ind = 6
        if up_tucker_masks is not None:
            xi_mask = Q_O_dG_xi_dxi_mu_masks[ind]
            ind = ind + 1
            xi = _apply_edge_mask(xi, xi_mask)

        if outer_tucker_masks is not None:
            dxi_mask = Q_O_dG_xi_dxi_mu_masks[ind]
            ind = ind + 1
            dxi = _apply_edge_mask(dxi, dxi_mask)

        if left_tt_masks is not None:
            mu_mask = Q_O_dG_xi_dxi_mu_masks[ind]
            ind += 1
            mu = _apply_edge_mask(mu, mu_mask)

        if right_tt_masks is not None:
            sigma_mask = Q_O_dG_xi_dxi_mu_masks[ind]
            ind += 1
            sigma = _apply_edge_mask(sigma, sigma_mask)

        sigma_next_t1 = xnp.einsum(
            '...aj,...a->...j',
            xnp.einsum('...i,iaj->...aj', sigma, Q),
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

        sigma_next = sigma_next_t1 + sigma_next_t2 + sigma_next_t3
        return sigma_next, (sigma,)

    rR0 = right_tt_cores[0].shape[0]
    vectorization_shape = xis[0].shape[:-1]
    init = xnp.zeros(vectorization_shape + (rR0,))

    xs = (right_tt_cores, outer_tt_cores, var_tt_cores, xis, dxis, mus)
    xs = xs + (xi_masks,) if xi_masks is not None else xs
    xs = xs + (dxi_masks,) if dxi_masks is not None else xs
    xs = xs + (mu_masks,) if mu_masks is not None else xs
    xs = xs + (sigma_masks,) if sigma_masks is not None else xs

    last_sigma, (sigmas,) = scan(_func, init, xs)
    return sigmas


def compute_taus(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        left_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nUi,rL(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nUi),
        dxis: typ.Sequence[NDArray], # len=d, elm_shape=(...,nOi)
        nus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,nR(i+1))
        up_tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nUi,)
        outer_tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nOi,)
        left_tt_masks: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rLi,)
        right_tt_masks: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rRi,)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # taus. len=d, elm_shape=(...,rL(i+1))
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
    rev_xis     = xis[::-1]
    rev_dxis    = dxis[::-1]
    rev_nus     = nus[::-1]
    rev_up_tucker_masks     = None if up_tucker_masks       is None else up_tucker_masks[::-1]
    rev_outer_tucker_masks  = None if outer_tucker_masks    is None else outer_tucker_masks[::-1]
    rev_left_tt_masks       = None if left_tt_masks         is None else left_tt_masks[::-1]
    rev_right_tt_masks      = None if right_tt_masks        is None else right_tt_masks[::-1]

    rev_taus = compute_sigmas(
        rev_var_tt_cores,
        rev_left_tt_cores,
        rev_outer_tt_cores,
        rev_xis,
        rev_dxis,
        rev_nus,
        up_tucker_masks=rev_up_tucker_masks,
        outer_tucker_masks=rev_outer_tucker_masks,
        left_tt_masks=rev_left_tt_masks,
        right_tt_masks=rev_right_tt_masks,
        xnp=xnp
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
        outer_tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nOi,)
        left_tt_masks: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rLi,)
        right_tt_masks: typ.Sequence[NDArray] = None,  # len=d+1, elm_shape=(rRi,)
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
    mu_masks    = left_tt_masks[:-1]    if left_tt_masks        is not None else None
    tau_masks   = left_tt_masks[1:]     if left_tt_masks        is not None else None
    nu_masks    = right_tt_masks[1:]    if right_tt_masks       is not None else None
    sigma_masks = right_tt_masks[:-1]   if right_tt_masks       is not None else None
    deta_masks  = outer_tucker_masks

    def _func(x):
        P, Q, dG, mu, nu, sigma, tau = x[0], x[1], x[2], x[3], x[4], x[5], x[6]

        ind = 7
        if mu_masks is not None:
            mu = _apply_edge_mask(mu, x[ind])
            ind += 1

        if tau_masks is not None:
            tau = _apply_edge_mask(tau, x[ind])
            ind += 1

        if nu_masks is not None:
            nu = _apply_edge_mask(nu, x[ind])
            ind += 1

        if sigma_masks is not None:
            sigma = _apply_edge_mask(sigma, x[ind])
            ind += 1

        s1 = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', sigma, Q),
            nu,
        )
        s2 = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', mu, dG),
            nu,
        )
        s3 = xnp.einsum(
            '...aj,...j->...a',
            xnp.einsum('...i,iaj->...aj', mu, P),
            tau,
        )
        deta = s1 + s2 + s3

        if deta_masks is not None:
            deta = _apply_edge_mask(deta, x[ind])
            ind = ind + 1

        return (deta,)

    xs = (left_tt_cores, right_tt_cores, var_tt_cores, mus, nus, sigmas, taus)
    xs = xs + (mu_masks,)       if mu_masks     is not None else xs
    xs = xs + (tau_masks,)      if tau_masks    is not None else xs
    xs = xs + (nu_masks,)       if nu_masks     is not None else xs
    xs = xs + (sigma_masks,)    if sigma_masks  is not None else xs
    xs = xs + (deta_masks,)     if deta_masks   is not None else xs

    detas_tuple = map(_func, xs)
    return detas_tuple[0]


def assemble_tangent_probes(
        tucker_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(nUi,Ni)
        var_tucker_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        etas: typ.Sequence[NDArray], # etas. len=d, elm_shape=(...,nUi)
        detas: typ.Sequence[NDArray], # detas. len=d, elm_shape=(...,nUi)
        shape_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(Ni,)
        up_tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nUi,)
        outer_tucker_masks: typ.Sequence[NDArray] = None,  # len=d, elm_shape=(nOi,)
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
    eta_masks   = up_tucker_masks
    deta_masks  = outer_tucker_masks
    probe_masks = shape_masks

    def _func(x):
        B, dB, eta, deta = x[0], x[1], x[2], x[3]

        ind = 4
        if eta_masks is not None:
            eta = _apply_edge_mask(eta, x[ind])
            ind += 1

        if deta_masks is not None:
            deta = _apply_edge_mask(deta, x[ind])
            ind += 1

        s1 = xnp.einsum('ao,...a->...o', B, deta)
        s2 = xnp.einsum('ao,...a->...o', dB, eta)
        probe = s1 + s2

        if probe_masks is not None:
            probe = _apply_edge_mask(probe, x[ind])
            ind += 1

        return (probe,)

    xs = (tucker_cores, var_tucker_cores, etas, detas)
    xs = xs + (eta_masks,)      if eta_masks    is not None else xs
    xs = xs + (deta_masks,)     if deta_masks   is not None else xs
    xs = xs + (probe_masks,)    if probe_masks  is not None else xs

    (probes,) = map(_func, xs)
    return probes


def probe_tangent(
        variation: t3toolbox.base_variation_format.T3Variation, # tucker_var_shapes=(nOi,Ni), tt_var_shapes=tt_hole_shapes=(rLi,ni,rRi)
        ww:     typ.Sequence[NDArray], # input vectors, len=d, elm_shape=(...,Ni)
        base: t3toolbox.base_variation_format.T3Base, # tucker_hole_shapes=(nOi,Ni), tt_hole_shapes=(rLi,ni,rRi)
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
    >>> zz = t3p.probe_tangent(variation, ww, base)
    >>> zz2 = t3p.probe_dense(t3m.tangent_to_dense(variation, base), ww)
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
    >>> zzz = t3p.probe_tangent(variation, www, base) # Compute probes!
    >>> zzz2 = t3p.probe_dense(t3m.tangent_to_dense(variation, base), www)
    >>> print([np.linalg.norm(zz - zz2, axis=1) for zz, zz2 in zip(zzz, zzz2)])
    [array([3.18560984e-15, 5.06339604e-15]), array([1.74264349e-15, 5.10008230e-15]), array([2.17576097e-15, 2.94156728e-15])]
    '''
    t3toolbox.base_variation_format.check_fit(variation, base)

    x_shape = tuple([B.shape[1] for B in variation[0]])
    assert(len(ww) == len(x_shape))

    assert(tuple([w.shape[-1] for w in ww]) == x_shape)

    (tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores) = base
    (var_tucker_cores, var_tt_cores) = variation

    xis = compute_xis(
        tucker_cores, ww, xnp=xnp,
    )

    mus = compute_mus(
        left_tt_cores, xis, xnp=xnp,
    )

    nus = compute_nus(
        right_tt_cores, xis, xnp=xnp,
    )

    etas = compute_etas(
        outer_tt_cores, mus, nus, xnp=xnp,
    )

    dxis = compute_dxis(
        var_tucker_cores, ww, xnp=xnp,
    )

    sigmas = compute_sigmas(
        var_tt_cores, right_tt_cores, outer_tt_cores,
        xis, dxis, mus,
        xnp=xnp,
    )

    taus = compute_taus(
        var_tt_cores, left_tt_cores, outer_tt_cores,
        xis, dxis, nus,
        xnp=xnp,
    )

    detas = compute_detas(
        var_tt_cores, left_tt_cores, right_tt_cores,
        mus, nus, sigmas, taus,
        xnp=xnp,
    )

    zz = assemble_tangent_probes(
        tucker_cores, var_tucker_cores,
        etas, detas,
        xnp=xnp,
    )

    return zz


###############################################################
###########    Transpose of tangent to probes map    ##########
###############################################################

def compute_deta_tildes(
        ztildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,Ni)
        tucker_cores: typ.Sequence[NDArray], # len=d, elm_shape=(ni,Ni)
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
    def _func(U_zt):
        U, zt = U_zt
        deta_tilde = xnp.einsum('ao,...o->...a', U, zt)
        return [deta_tilde]

    deta_tildes_tuple = map(_func, (tucker_cores, ztildes))
    return deta_tildes_tuple[0]


def compute_tau_tildes(
        deta_tildes,  # len=d+1, elm_shape=(...,ni)
        left_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rL(i+d))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,ni)
        mus, # len=d, elm_shape=(...,rLi)
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
    def _func(tau_tilde, P_xi_deta_tilde_mu):
        P, xi, deta_tilde, mu = P_xi_deta_tilde_mu
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
        return tau_tilde_next, [tau_tilde]

    init = xnp.zeros(mus[0].shape[:-1] + (left_tt_cores[0].shape[0],))
    xs = (left_tt_cores, xis, deta_tildes, mus)
    last_tau_tilde, tau_tildes_tuple = scan(_func, init, xs)
    return tau_tildes_tuple[0]


def compute_sigma_tildes(
        deta_tildes,  # len=d, elm_shape=(...,ni)
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,ni,rR(i+d))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(...,ni)
        nus, # len=d, elm_shape=(...,rR(i+1))
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
    return compute_tau_tildes(
        deta_tildes[::-1], tt_reverse(right_tt_cores), xis[::-1], nus[::-1],
        xnp=xnp,
    )[::-1]


def compute_dxi_tildes(
        sigma_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rR(i+1))
        tau_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rLi)
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        mus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rLi)
        nus: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rR(i+1))
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
    def _func(O_mu_nu_st_tt):
        O, mu, nu, st, tt = O_mu_nu_st_tt
        dxi_tilde = (
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
        return [dxi_tilde]

    dxi_tildes_tuple = map(_func, (outer_tt_cores, mus, nus, sigma_tildes, tau_tildes))
    return dxi_tildes_tuple[0]


def assemble_tucker_variations(
        ztildes: typ.Sequence[NDArray], # len=d, elm_shape=(...,Ni)
        dxi_tildes: typ.Sequence[NDArray], #len=d, elm_shape=(...,nOi)
        ww: typ.Sequence[NDArray],  # input vectors, len=d, elm_shape=(Ni,) or (...,Ni)
        etas: typ.Sequence[NDArray],  # etas. len=d, elm_shape=(...,ni)
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
    def _func(z_tilde_eta_w_dxi_tilde):
        z_tilde, eta, w, dxi_tilde = z_tilde_eta_w_dxi_tilde
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
        return [dU_tilde]

    dU_tildes_tuple = map(_func, (ztildes, etas, ww, dxi_tildes))
    return dU_tildes_tuple[0]


def assemble_tt_variations(
        sigma_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rR(i+1))
        tau_tildes: typ.Sequence[NDArray],  # len=d, elm_shape=(...,rLi)
        deta_tildes,  # len=d+1, elm_shape=(...,ni)
        xis: typ.Sequence[NDArray],  # len=d, elm_shape=(...,ni)
        mus,  # len=d, elm_shape=(...,rLi)
        nus,  # len=d, elm_shape=(...,rR(i+1))
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
    def _func(xi_mu_nu_sigma_tilde_tau_tilde_deta_tilde):
        xi, mu, nu, sigma_tilde, tau_tilde, deta_tilde = xi_mu_nu_sigma_tilde_tau_tilde_deta_tilde
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
        return [dG_tilde]

    dG_tildes_tuple = map(_func, (xis, mus, nus, sigma_tildes, tau_tildes, deta_tildes))
    return dG_tildes_tuple[0]


def probe_tangent_transpose(
        ztildes: typ.Sequence[NDArray], # len=d, elm_shape=(...,Ni)
        ww: typ.Sequence[NDArray],  # input vectors, len=d, elm_shape=(...,Ni)
        base: t3toolbox.base_variation_format.T3Base, # tucker_hole_shapes=(nOi,Ni), tt_hole_shapes=(rLi,ni,rRi)
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

import t3tools.corewise    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.common as common
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> v1 = t3m.tangent_randn(base)
    >>> zz1 = t3p.probe_tangent(v1, ww, base)
    >>> zz2 = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> v2 = t3p.probe_tangent_transpose(zz2, ww, base)
    >>> ipA = t3toolbox.corewise.corewise_dot(v1, v2)
    >>> print(ipA)
    17.958317927787
import t3tools.corewise    >>> ipB = t3tools.corewise.corewise_dot(zz1, zz2)
    >>> print(ipB)
    17.958317927787

    Apply transpose map with two sets of probing vectors:

import t3tools.corewise    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.probing as t3p
    >>> import t3toolbox.common as common
    >>> import t3toolbox.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> ww = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> apply_J = lambda v: t3p.probe_tangent(v, ww, base)
    >>> apply_Jt = lambda z: t3p.probe_tangent_transpose(z, ww, base)
    >>> v = t3m.tangent_randn(base)
    >>> z = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> print(t3toolbox.corewise.corewise_dot(z, apply_J(v)) - t3toolbox.corewise.corewise_dot(apply_Jt(z), v))
    7.105427357601002e-15
    '''
    num_cores = len(ztildes)
    assert(len(ww) == num_cores)

    (tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores) = base

    xis = compute_xis(
        tucker_cores, ww, xnp=xnp,
    )

    mus = compute_mus(
        left_tt_cores, xis, xnp=xnp,
    )

    nus = compute_nus(
        right_tt_cores, xis, xnp=xnp,
    )

    etas = compute_etas(
        outer_tt_cores, mus, nus, xnp=xnp,
    )

    #

    deta_tildes = compute_deta_tildes(
        ztildes, tucker_cores, xnp=xnp,
    )

    tau_tildes = compute_tau_tildes(
        deta_tildes, left_tt_cores, xis, mus, xnp=xnp,
    )

    sigma_tildes = compute_sigma_tildes(
        deta_tildes, right_tt_cores, xis, nus, xnp=xnp,
    )

    dxi_tildes = compute_dxi_tildes(
        sigma_tildes, tau_tildes, outer_tt_cores, mus, nus, xnp=xnp,
    )

    #

    dU_tildes = assemble_tucker_variations(
        ztildes, dxi_tildes, ww, etas,
        sum_over_probes=sum_over_probes, xnp=xnp,
    )

    dG_tildes = assemble_tt_variations(
        sigma_tildes, tau_tildes, deta_tildes, xis, mus, nus,
        sum_over_probes=sum_over_probes, xnp=xnp,
    )

    return dU_tildes, dG_tildes

