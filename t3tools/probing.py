# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ

import t3tools.base_variation_format
import t3tools.tucker_tensor_train as t3
from t3tools.common import NDArray


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
    'assemble_basis_variations',
    'assemble_tt_variations',
    'probe_tangent_transpose',
]


###############################################
##########    Probe dense tensor    ###########
###############################################

def probe_dense(
        T:          NDArray,
        vectors:    typ.Sequence[NDArray],
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
    >>> import t3tools.probing as t3p
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
    >>> import t3tools.t3p as t3p
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
        ww: typ.Sequence[NDArray], # input vectors, len=d, elm_shape=(Ni,) or (num_probes,Ni)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(Ni,) or (num_probes,Ni)
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
        input vectors to probe with. len=d, elm_shape=(Ni,) or (num_probes,Ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        Probes, zz. len=d, elm_shape=(Ni,) or (num_probes,Ni)

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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.probing as t3p
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

    vectorized = True
    if len(ww[0].shape) == 1:
        vectorized = False
        ww = [w.reshape((1,-1)) for w in ww]

    basis_cores, tt_cores = x

    for B, w in zip(basis_cores, ww):
        assert(B.shape[1] == w.shape[1])

    xis = compute_xis(basis_cores, ww, xnp=xnp)
    mus = compute_mus(tt_cores, xis, xnp=xnp)
    nus = compute_nus(tt_cores, xis, xnp=xnp)
    etas = compute_etas(tt_cores, mus, nus, xnp=xnp)
    zz = assemble_probes(basis_cores, etas, xnp=xnp)

    if not vectorized:
        zz = tuple([z.reshape(-1) for z in zz])
    return zz


def compute_xis(
        basis_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ni,Ni)
        ww: typ.Sequence[NDArray], # len=d. elm_shape=(num_probes,Ni)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # xis. len=d, elm_shape=(num_probes,ni)
    '''Compute upward edge variables associated with edges between Tucker cores and adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 5.2, particularly Figure 9 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    basis_cores: typ.Sequence[NDArray]
        Basis cores for Tucker tensor train.
        len=d. elm_shape=(ni,Ni)
    ww: typ.Sequence[NDArray]
        input vectors to probe with. len=d, elm_shape=(Ni,) or (num_probes,Ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        upward edge variables xi. len=d, elm_shape=(num_probes,ni)

    See Also
    --------
    probe_t3
    compute_mus
    compute_nus
    compute_etas
    assemble_probes
    '''
    return tuple([xnp.einsum('io,po->pi', U, w) for U, w in zip(basis_cores, ww)])


def compute_mus(
        left_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ri,ni,r(i+1))
        xis: typ.Sequence[NDArray], # len=d. elm_shape=(num_probes,ni)
        xnp = np,
) -> typ.Sequence[NDArray]: # mus. len=d+1, elm_shape=(num_probes,ri)
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
        upward edge variables xi. len=d. elm_shape=(num_probes,ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        leftward edge variables mu. len=d+1, elm_shape=(num_probes,ri)

    See Also
    --------
    probe_t3
    compute_xis
    compute_nus
    compute_etas
    assemble_probes
    '''
    num_probes = xis[0].shape[0]
    num_cores = len(xis)
    mus = [xnp.ones((num_probes, left_tt_cores[0].shape[0]))]
    for ii in range(num_cores):
        P = left_tt_cores[ii]
        xi = xis[ii]
        mu = mus[-1]
        mu_next = xnp.einsum('pi,iaj,pa->pj', mu, P, xi)
        mus.append(mu_next)
    return tuple(mus)


def tt_reverse(tt_cores):
    return tuple([G.swapaxes(0, 2) for G in tt_cores[::-1]])


def compute_nus(
        right_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ri,ni,r(i+1))
        xis, # len=d. elm_shape=(num_probes,ni)
        xnp = np,
) -> typ.Sequence[NDArray]: # nus. len=d+1, elm_shape=(num_probes,ri)
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
        upward edge variables xi. len=d. elm_shape=(num_probes,ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        rightward edge variables nu. len=d+1, elm_shape=(num_probes,ri)

    See Also
    --------
    probe_t3
    compute_xis
    compute_mus
    compute_etas
    assemble_probes
    '''
    return compute_mus(tt_reverse(right_tt_cores), xis[::-1], xnp=xnp)[::-1]


def compute_etas(
        outer_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ri,ni,r(i+1))
        mus, # len=d. elm_shape=(num_probes,ri)
        nus, # len=d. elm_shape=(num_probes,ri)
        xnp = np,
) -> typ.Sequence[NDArray]: # etas. len=d, elm_shape=(num_probes,ni)
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
        leftward edge variables mu. len=d+1. elm_shape=(num_probes,ri)
    nus: typ.Sequence[NDArray]
        rightward edge variables mu. len=d+1. elm_shape=(num_probes,ri)
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
    return tuple([
        xnp.einsum('pi,iaj,pj->pa', mu, G, nu)
        for mu, G, nu in zip(mus[:-1], outer_tt_cores, nus[1:])
    ])


def assemble_probes(
        basis_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(ni,Ni)
        etas,  # len=d. elm_shape=(num_probes,ni)
        xnp = np,
) -> typ.Sequence[NDArray]: # zz. len=d, elm_shape=(num_probes,Ni)
    '''Assemble probes from downward edge variables.

    See Section 5.2, particularly Figure 9 in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    Parameters
    ----------
    basis_cores: typ.Sequence[NDArray]
        Basis cores for Tucker tensor train.
        len=d. elm_shape=(ni,Ni)
    etas: typ.Sequence[NDArray]
        downward edge variables eta. len=d. elm_shape=(num_probes,ni)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    typ.Tuple[NDArray,...]
        probes z. len=d, elm_shape=(num_probes,Ni)

    See Also
    --------
    probe_t3
    compute_xis
    compute_mus
    compute_nus
    compute_etas
    '''
    return tuple([xnp.einsum('pa,ao->po', eta, U) for U, eta in zip(basis_cores, etas)])


#####################################################
###########    Probing a tangent vector    ##########
#####################################################

def compute_dxis(
        var_basis_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        ww: typ.Sequence[NDArray], # len=d. elm_shape=(num_probes,Ni)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # xis. len=d, elm_shape=(num_probes,nOi)
    '''Compute var-upward edge variables dxi.
    Used for probing a tangent vector.

    Same as t3_compute_dxis(), except with var_basis_cores in place of basis_cores.

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
    return compute_xis(var_basis_cores, ww, xnp=xnp)


def compute_sigmas(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,ni,rR(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,ni),
        dxis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,nOi)
        mus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,nLi)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # sigmas. len=d+1, elm_shape=(num_probes,rR(i+1))
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
    num_cores = len(xis)
    num_probes = xis[0].shape[0]

    sigmas = [xnp.zeros((num_probes, right_tt_cores[0].shape[0]))]
    for ii in range(num_cores):
        Q = right_tt_cores[ii]
        O = outer_tt_cores[ii]
        dG = var_tt_cores[ii]
        xi = xis[ii]
        dxi = dxis[ii]

        mu = mus[ii]
        sigma = sigmas[-1]

        sigma_next_t1   = xnp.einsum('pi,iaj,pa->pj', sigma, Q, xi)
        sigma_next_t2   = xnp.einsum('pi,iaj,pa->pj', mu, dG, xi)
        sigma_next_t3   = xnp.einsum('pi,iaj,pa->pj', mu, O, dxi)

        sigma_next = sigma_next_t1 + sigma_next_t2 + sigma_next_t3
        sigmas.append(sigma_next)
    return tuple(sigmas)


def compute_taus(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rR(i+1))
        left_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rL(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,ni),
        dxis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,nOi)
        nus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,nRi)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # taus. len=d+1, elm_shape=(num_probes,rL(i+1))
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
    return compute_sigmas(
        tt_reverse(var_tt_cores), tt_reverse(left_tt_cores), tt_reverse(outer_tt_cores),
        xis[::-1], dxis[::-1], nus[::-1],
        xnp=xnp
    )[::-1]


def compute_detas(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rR(i+1))
        left_tt_cores: typ.Sequence[NDArray],  # len=d, elm_shape=(rLi,ni,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,ni,rR(i+1))
        mus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,nLi)
        nus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,nRi)
        sigmas: typ.Sequence[NDArray], # len=d+1, elm_shape=(num_probes,rR(i+1))
        taus: typ.Sequence[NDArray], # len=d+1, elm_shape=(num_probes,rL(i+1))
        xnp = np,
) -> typ.Sequence[NDArray]: # detas. len=d, elm_shape=(num_probes,ni)
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
    num_cores = len(var_tt_cores)

    detas = []
    for ii in range(num_cores):
        P = left_tt_cores[ii]
        Q = right_tt_cores[ii]
        dG = var_tt_cores[ii]

        mu = mus[ii]
        nu = nus[ii+1]
        sigma = sigmas[ii]
        tau = taus[ii+1]

        s1 = xnp.einsum('pi,iaj,pj->pa', sigma, Q, nu)
        s2 = xnp.einsum('pi,iaj,pj->pa', mu,    dG, nu)
        s3 = xnp.einsum('pi,iaj,pj->pa', mu,    P, tau)

        deta = s1 + s2 + s3
        detas.append(deta)
    return tuple(detas)


def assemble_tangent_probes(
        basis_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(ni,Ni)
        var_basis_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        etas: typ.Sequence[NDArray], # etas. len=d, elm_shape=(num_probes,ni)
        detas: typ.Sequence[NDArray], # detas. len=d, elm_shape=(num_probes,ni)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # probes. len=d, elm_shape=(num_probes,Ni)
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
    num_cores = len(basis_cores)
    probes = []
    for ii in range(num_cores):
        B = basis_cores[ii]
        dB = var_basis_cores[ii]

        eta = etas[ii]
        deta = detas[ii]

        s1 = xnp.einsum('ao,pa->po', B, deta)
        s2 = xnp.einsum('ao,pa->po', dB, eta)

        probe = s1 + s2
        probes.append(probe)
    return tuple(probes)


def probe_tangent(
        variation: t3tools.base_variation_format.T3Variation, # basis_var_shapes=(nOi,Ni), tt_var_shapes=tt_hole_shapes=(rLi,ni,rRi)
        ww:     typ.Sequence[NDArray], # input vectors, len=d, elm_shape=(Ni,) or (num_probes,Ni)
        base: t3tools.base_variation_format.T3Base, # basis_hole_shapes=(nOi,Ni), tt_hole_shapes=(rLi,ni,rRi)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(Ni,) or (num_probes,Ni)
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
        input vectors to probe with. len=d, elm_shape=(Ni,) or (num_probes,Ni)
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> import t3tools.probing as t3p
    >>> import t3tools.orthogonalization as orth
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> import t3tools.probing as t3p
    >>> import t3tools.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> variation = t3m.tangent_randn(base)
    >>> www = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> zzz = t3p.probe_tangent(variation, www, base) # Compute probes!
    >>> zzz2 = t3p.probe_dense(t3m.tangent_to_dense(variation, base), www)
    >>> print([np.linalg.norm(zz - zz2, axis=1) for zz, zz2 in zip(zzz, zzz2)])
    [array([3.18560984e-15, 5.06339604e-15]), array([1.74264349e-15, 5.10008230e-15]), array([2.17576097e-15, 2.94156728e-15])]
    '''
    t3tools.base_variation_format.check_fit(variation, base)

    x_shape = tuple([B.shape[1] for B in variation[0]])
    assert(len(ww) == len(x_shape))

    vectorized = True
    if len(ww[0].shape) == 1:
        vectorized = False
        ww = [w.reshape((1,-1)) for w in ww]

    assert(tuple([w.shape[1] for w in ww]) == x_shape)

    (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores) = base
    (var_basis_cores, var_tt_cores) = variation

    xis = compute_xis(
        basis_cores, ww, xnp=xnp,
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
        var_basis_cores, ww, xnp=xnp,
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
        basis_cores, var_basis_cores,
        etas, detas,
        xnp=xnp,
    )

    if not vectorized:
        zz = tuple([z.reshape(-1) for z in zz])
    return zz


###############################################################
###########    Transpose of tangent to probes map    ##########
###############################################################

def compute_deta_tildes(
        ztildes: typ.Sequence[NDArray],  # len=d, elm_shape=(num_probes,Ni)
        basis_cores: typ.Sequence[NDArray], # len=d, elm_shape=(ni,Ni)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(num_probes,ni)
    '''Adjoint-var-upward edge variables deta_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (43), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    return tuple([xnp.einsum('ao,po->pa', U, zt) for U, zt in zip(basis_cores, ztildes)])


def compute_tau_tildes(
        deta_tildes,  # len=d+1, elm_shape=(num_probes,ni)
        left_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rL(i+d))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,ni)
        mus, # len=d+1, elm_shape=(num_probes,rLi)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d+1, elm_shape=(num_probes,rLi)
    '''Adjoint-var-rightward edge variables tau_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (44), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    num_cores = len(left_tt_cores)
    num_probes = mus[0].shape[0]

    tau_tildes = [xnp.zeros((num_probes, left_tt_cores[0].shape[0]))]
    for ii in range(num_cores):
        P = left_tt_cores[ii]
        xi = xis[ii]
        deta_tilde = deta_tildes[ii]

        mu = mus[ii]
        tau_tilde = tau_tildes[-1]

        tau_tilde_next_t1   = xnp.einsum('pi,iaj,pa->pj', tau_tilde, P, xi)
        tau_tilde_next_t2   = xnp.einsum('pi,iaj,pa->pj', mu, P, deta_tilde)

        tau_tilde_next = tau_tilde_next_t1 + tau_tilde_next_t2
        tau_tildes.append(tau_tilde_next)
    return tuple(tau_tildes)


def compute_sigma_tildes(
        deta_tildes,  # len=d+1, elm_shape=(num_probes,ni)
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,ni,rR(i+d))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,ni)
        nus, # len=d+1, elm_shape=(num_probes,rRi)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d+1, elm_shape=(num_probes,rRi)
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
        sigma_tildes: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,rRi)
        tau_tildes: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,rLi)
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        mus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,rLi)
        nus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,rRi)
        xnp = np,
) -> typ.Tuple[NDArray,...]: # dxi_tildes. len=d, elm_shape=(num_probes,nOi)
    '''Adjoint-var-downward edge variables dxi_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (46), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    return tuple([
        xnp.einsum('pi,iaj,pj->pa', tt, O, nu) +
        xnp.einsum('pi,iaj,pj->pa', mu, O, st)
        for O, mu, nu, st, tt in
        zip(outer_tt_cores, mus[:-1], nus[1:], sigma_tildes[1:], tau_tildes[:-1])
    ])


def assemble_basis_variations(
        ztildes: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,Ni)
        dxi_tildes: typ.Sequence[NDArray], #len=d, elm_shape=(num_probes,nOi)
        ww: typ.Sequence[NDArray],  # input vectors, len=d, elm_shape=(Ni,) or (num_probes,Ni)
        etas: typ.Sequence[NDArray],  # etas. len=d, elm_shape=(num_probes,ni)
        sum_over_probes: bool = False,
        xnp = np,
):
    '''Assemble basis core variations, delta_U_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (47), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    if sum_over_probes:
        dU_tildes = tuple([
            xnp.einsum('po,pa->ao', z_tilde, eta) +
            xnp.einsum('po,pa->ao', w, dxi_tilde)
            for z_tilde, eta, w, dxi_tilde in
            zip(ztildes, etas, ww, dxi_tildes)
        ])
    else:
        dU_tildes = tuple([
            xnp.einsum('po,pa->pao', z_tilde, eta) +
            xnp.einsum('po,pa->pao', w, dxi_tilde)
            for z_tilde, eta, w, dxi_tilde in
            zip(ztildes, etas, ww, dxi_tildes)
        ])
    return dU_tildes


def assemble_tt_variations(
        sigma_tildes: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,rRi)
        tau_tildes: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,rLi)
        deta_tildes,  # len=d+1, elm_shape=(num_probes,ni)
        xis: typ.Sequence[NDArray],  # len=d, elm_shape=(num_probes,ni)
        mus,  # len=d+1, elm_shape=(num_probes,rLi)
        nus,  # len=d+1, elm_shape=(num_probes,rRi)
        sum_over_probes: bool = False,
        xnp = np,
) -> typ.Tuple[NDArray,...]: # len=d, elm_shape=(rLi,nOi,rRi)
    '''Assemble basis core variations, delta_U_tilde.
    Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.

    See Section 5.2.4, particularly Formula (48), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    if sum_over_probes:
        dG_tildes = tuple([
            xnp.einsum('pi,pa,pj->iaj', mu, xi, sigma_tilde) +
            xnp.einsum('pi,pa,pj->iaj', tau_tilde, xi, nu) +
            xnp.einsum('pi,pa,pj->iaj', mu, deta_tilde, nu)
            for xi, mu, nu, sigma_tilde, tau_tilde, deta_tilde in
            zip(xis, mus[:-1], nus[1:], sigma_tildes[1:], tau_tildes[:-1], deta_tildes)
        ])
    else:
        dG_tildes =  tuple([
            xnp.einsum('pi,pa,pj->piaj', mu,        xi,  sigma_tilde) +
            xnp.einsum('pi,pa,pj->piaj', tau_tilde, xi,  nu) +
            xnp.einsum('pi,pa,pj->piaj', mu,        deta_tilde, nu)
            for xi, mu, nu, sigma_tilde, tau_tilde, deta_tilde in
            zip(xis, mus[:-1], nus[1:], sigma_tildes[1:], tau_tildes[:-1], deta_tildes)
        ])
    return dG_tildes


def probe_tangent_transpose(
        ztildes: typ.Sequence[NDArray], # len=d, elm_shape=(Ni,) or (num_probes,Ni)
        ww: typ.Sequence[NDArray],  # input vectors, len=d, elm_shape=(Ni,) or (num_probes,Ni)
        base: t3tools.base_variation_format.T3Base, # basis_hole_shapes=(nOi,Ni), tt_hole_shapes=(rLi,ni,rRi)
        sum_over_probes: bool = False,
        xnp = np,
) -> typ.Tuple[
    typ.Tuple[NDArray,...], # dU_tildes. len=d, elm_shape=(nOi,Ni)
    typ.Tuple[NDArray,...], # dG_tildes. len=d, elm_shape=(rLi,ni,rRi)
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> import t3tools.probing as t3p
    >>> import t3tools.common as common
    >>> import t3tools.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> v1 = t3m.tangent_randn(base)
    >>> zz1 = t3p.probe_tangent(v1, ww, base)
    >>> zz2 = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> v2 = t3p.probe_tangent_transpose(zz2, ww, base)
    >>> ipA = t3tools.corewise.corewise_dot(v1, v2)
    >>> print(ipA)
    17.958317927787
import t3tools.corewise    >>> ipB = t3tools.corewise.corewise_dot(zz1, zz2)
    >>> print(ipB)
    17.958317927787

    Apply transpose map with two sets of probing vectors:

import t3tools.corewise    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.manifold as t3m
    >>> import t3tools.probing as t3p
    >>> import t3tools.common as common
    >>> import t3tools.orthogonalization as orth
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(2,3,4,2)))
    >>> base, _ = orth.orthogonal_representations(p)
    >>> ww = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> apply_J = lambda v: t3p.probe_tangent(v, ww, base)
    >>> apply_Jt = lambda z: t3p.probe_tangent_transpose(z, ww, base)
    >>> v = t3m.tangent_randn(base)
    >>> z = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> print(t3tools.corewise.corewise_dot(z, apply_J(v)) - t3tools.corewise.corewise_dot(apply_Jt(z), v))
    7.105427357601002e-15
    '''
    num_cores = len(ztildes)
    assert(len(ww) == num_cores)

    if len(ww[0].shape) == 1:
        vectorized = False
        for w in ww:
            assert(len(w.shape) == 1)
        for zt in ztildes:
            assert(len(zt.shape) == 1)

        ww = [w.reshape((1,-1)) for w in ww]
        ztildes = [zt.reshape((1,-1)) for zt in ztildes]
    else:
        vectorized = True
        for w in ww:
            assert(len(w.shape) == 2)
        for zt in ztildes:
            assert(len(zt.shape) == 2)

    (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores) = base
    base_shape = tuple([B.shape[1] for B in basis_cores])
    assert(tuple([w.shape[1] for w in ww]) == base_shape)
    assert(tuple([zt.shape[1] for zt in ztildes]) == base_shape)

    (basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores) = base

    xis = compute_xis(
        basis_cores, ww, xnp=xnp,
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
        ztildes, basis_cores, xnp=xnp,
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

    dU_tildes = assemble_basis_variations(
        ztildes, dxi_tildes, ww, etas,
        sum_over_probes=sum_over_probes, xnp=xnp,
    )

    dG_tildes = assemble_tt_variations(
        sigma_tildes, tau_tildes, deta_tildes, xis, mus, nus,
        sum_over_probes=sum_over_probes, xnp=xnp,
    )

    if not vectorized:
        dU_tildes = tuple([dU_tilde.reshape(dU_tilde.shape[1:]) for dU_tilde in dU_tildes])
        dG_tildes = tuple([dG_tilde.reshape(dG_tilde.shape[1:]) for dG_tilde in dG_tildes])

    return dU_tildes, dG_tildes

