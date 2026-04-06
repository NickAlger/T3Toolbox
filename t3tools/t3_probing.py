# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
import t3tools.tucker_tensor_train as t3
import t3tools.t3_manifold as t3m

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]

__all__ = [
    # Actions of a Tucker tensor train
    't3_probes',
    't3_compute_xis',
    't3_compute_mus',
    't3_compute_nus',
    't3_compute_etas',
    't3_assemble_probes',
    # Actions of a tangent vector
    't3tangent_probes',
    't3_compute_dxis',
    't3_compute_sigmas',
    't3_compute_taus',
    't3_compute_detas',
    't3_assemble_tangent_probes',

    # 't3tangent_probes_transpose',
    # Actions of a tangent vector
    # 't3_assemble_tangent_actions',
    # # Transpose of tangent vector to actions map
    # 't3_compute_tau_tildes',
    # 't3_compute_sigma_tildes',
    # 't3_assemble_core_perturbations',
]

#

def t3_probes(
        x:      t3.TuckerTensorTrain, # structure=((N1,...,Nd),(n1,...,nd),(1,r1,...,r(d-1),1))
        ww:     typ.Sequence[NDArray], # input vectors, len=d, elm_shape=(Ni,) or (num_probes,Ni)
        use_jax: bool = False,
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
    use_jax: bool
        If True, use jax for linear algebra operations. Otherwise, use numpy.

    Returns
    -------
    typ.Tuple[NDArray,...]
        Probes, zz. len=d, elm_shape=(Ni,) or (num_probes,Ni)

    See Also
    --------
    t3tangent_probes
    t3tangent_probes_transpose
    t3_compute_xis
    t3_compute_mus
    t3_compute_nus
    t3_compute_etas
    t3_assemble_probes

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.t3_probing as t3p
    >>> import t3tools.dense as dense
    >>> x = t3.t3_corewise_randn(((10,11,12),(5,6,4),(1,3,4,1)))
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.t3_probes(x, ww)
    >>> x_dense = t3.t3_to_dense(x)
    >>> zz2 = dense.dense_probes(x_dense, ww)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [1.3259763141008934e-13, 2.6471889228499147e-13, 2.4290827307404746e-13]
    '''
    shape = t3.t3_structure(x)[0]
    assert(len(ww) == len(shape))

    xnp = jnp if use_jax else np

    vectorized = True
    if len(ww[0].shape) == 1:
        vectorized = False
        ww = [w.reshape((1,-1)) for w in ww]

    basis_cores, tt_cores = x

    for B, w in zip(basis_cores, ww):
        assert(B.shape[1] == w.shape[1])

    xis = t3_compute_xis(basis_cores, ww, use_jax=use_jax)
    mus = t3_compute_mus(tt_cores, xis, use_jax=use_jax)
    nus = t3_compute_nus(tt_cores, xis, use_jax=use_jax)
    etas = t3_compute_etas(tt_cores, mus, nus, use_jax=use_jax)
    zz = t3_assemble_probes(basis_cores, etas, use_jax=use_jax)

    if not vectorized:
        zz = tuple([z.reshape(-1) for z in zz])
    return zz


def t3_compute_xis(
        basis_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ni,Ni)
        ww: typ.Sequence[NDArray], # len=d. elm_shape=(num_probes,Ni)
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]: # xis. len=d, elm_shape=(num_probes,ni)
    '''Compute upward edge variables associated with edges between Tucker cores and adjacent TT-cores.

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
    use_jax: bool
        If True, use jax for linear algebra operations. Otherwise, use numpy.

    Returns
    -------
    typ.Tuple[NDArray,...]
        upward edge variables xi. len=d, elm_shape=(num_probes,ni)

    See Also
    --------
    t3_probes
    t3_compute_mus
    t3_compute_nus
    t3_compute_etas
    t3_assemble_probes
    '''
    xnp = jnp if use_jax else np
    return tuple([xnp.einsum('io,po->pi', U, w) for U, w in zip(basis_cores, ww)])


def t3_compute_mus(
        left_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ri,ni,r(i+1))
        xis: typ.Sequence[NDArray], # len=d. elm_shape=(num_probes,ni)
        use_jax: bool = False,
) -> typ.Sequence[NDArray]: # mus. len=d+1, elm_shape=(num_probes,ri)
    '''Compute leftward edge variables associated with edges between adjacent TT-cores.

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
    use_jax: bool
        If True, use jax for linear algebra operations. Otherwise, use numpy.

    Returns
    -------
    typ.Tuple[NDArray,...]
        leftward edge variables mu. len=d+1, elm_shape=(num_probes,ri)

    See Also
    --------
    t3_probes
    t3_compute_xis
    t3_compute_nus
    t3_compute_etas
    t3_assemble_probes
    '''
    xnp = jnp if use_jax else np

    num_probes = xis[0].shape[0]
    num_cores = len(xis)
    mus = [xnp.ones((num_probes, 1))]
    for ii in range(num_cores):
        P = left_tt_cores[ii]
        xi = xis[ii]
        mu = mus[-1]
        mu_next = xnp.einsum('pi,iaj,pa->pj', mu, P, xi)
        mus.append(mu_next)
    return tuple(mus)


def tt_reverse(tt_cores):
    return tuple([G.swapaxes(0, 2) for G in tt_cores[::-1]])


def t3_compute_nus(
        right_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ri,ni,r(i+1))
        xis, # len=d. elm_shape=(num_probes,ni)
        use_jax: bool = False,
) -> typ.Sequence[NDArray]: # nus. len=d+1, elm_shape=(num_probes,ri)
    '''Compute rightward edge variables associated with edges between adjacent TT-cores.

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
    use_jax: bool
        If True, use jax for linear algebra operations. Otherwise, use numpy.

    Returns
    -------
    typ.Tuple[NDArray,...]
        rightward edge variables nu. len=d+1, elm_shape=(num_probes,ri)

    See Also
    --------
    t3_probes
    t3_compute_xis
    t3_compute_mus
    t3_compute_etas
    t3_assemble_probes
    '''
    return t3_compute_mus(tt_reverse(right_tt_cores), xis[::-1], use_jax=use_jax)[::-1]


def t3_compute_etas(
        outer_tt_cores: typ.Sequence[NDArray], # len=d. elm_shape=(ri,ni,r(i+1))
        mus, # len=d. elm_shape=(num_probes,ri)
        nus, # len=d. elm_shape=(num_probes,ri)
        use_jax: bool = False,
) -> typ.Sequence[NDArray]: # etas. len=d, elm_shape=(num_probes,ni)
    '''Compute downward edge variables associated with edges between Tucker cores and adjacent TT-cores.

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
    use_jax: bool
        If True, use jax for linear algebra operations. Otherwise, use numpy.

    Returns
    -------
    typ.Tuple[NDArray,...]
        downward edge variables eta. len=d+1, elm_shape=(num_probes,ni)

    See Also
    --------
    t3_probes
    t3_compute_xis
    t3_compute_xis
    t3_compute_nus
    t3_assemble_probes
    '''
    xnp = jnp if use_jax else np
    return tuple([
        xnp.einsum('pi,iaj,pj->pa', mu, G, nu)
        for mu, G, nu in zip(mus[:-1], outer_tt_cores, nus[1:])
    ])


def t3_assemble_probes(
        basis_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(ni,Ni)
        etas,  # len=d. elm_shape=(num_probes,ni)
        use_jax: bool = False,
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
    use_jax: bool
        If True, use jax for linear algebra operations. Otherwise, use numpy.

    Returns
    -------
    typ.Tuple[NDArray,...]
        probes z. len=d, elm_shape=(num_probes,Ni)

    See Also
    --------
    t3_probes
    t3_compute_xis
    t3_compute_mus
    t3_compute_nus
    t3_compute_etas
    '''
    xnp = jnp if use_jax else np
    return tuple([xnp.einsum('pa,ao->po', eta, U) for U, eta in zip(basis_cores, etas)])

###

def t3_compute_dxis(
        var_basis_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        ww: typ.Sequence[NDArray], # len=d. elm_shape=(num_probes,Ni)
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]: # xis. len=d, elm_shape=(num_probes,nOi)
    '''Compute var-upward edge variables dxi.

    Same as t3_compute_dxis(), except with var_basis_cores in place of basis_cores.

    See Section 5.2.3, particularly Formula (34), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    t3_compute_xis
    t3_compute_sigmas
    t3_compute_taus
    t3_compute_detas
    t3_assemble_tangent_probes
    t3tangent_probes
    '''
    return t3_compute_xis(var_basis_cores, ww, use_jax)


def t3_compute_sigmas(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,ni,rR(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,ni),
        dxis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,nOi)
        mus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,nLi)
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]: # sigmas. len=d+1, elm_shape=(num_probes,rR(i+1))
    '''Compute var-leftward edge variables sigma.

    See Section 5.2.3, particularly Formula (36), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    t3_compute_dxis
    t3_compute_taus
    t3_compute_detas
    t3_assemble_tangent_probes
    t3tangent_probes
    '''
    xnp = jnp if use_jax else np

    num_cores = len(xis)
    num_probes = xis[0].shape[0]

    sigmas = [xnp.zeros((num_probes,1))]
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


def t3_compute_taus(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rR(i+1))
        left_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rL(i+1))
        outer_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,ni),
        dxis: typ.Sequence[NDArray], # len=d, elm_shape=(num_probes,nOi)
        nus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,nRi)
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]: # taus. len=d+1, elm_shape=(num_probes,rL(i+1))
    '''Compute var-rightward edge variables tau.

    See Section 5.2.3, particularly Formula (38), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    t3_compute_dxis
    t3_compute_sigmas
    t3_compute_detas
    t3_assemble_tangent_probes
    t3tangent_probes
    '''
    return t3_compute_sigmas(
        tt_reverse(var_tt_cores), tt_reverse(left_tt_cores), tt_reverse(outer_tt_cores),
        xis[::-1], dxis[::-1], nus[::-1],
        use_jax=use_jax
    )[::-1]


def t3_compute_detas(
        var_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rLi,ni,rR(i+1))
        left_tt_cores: typ.Sequence[NDArray],  # len=d, elm_shape=(rLi,ni,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray], # len=d, elm_shape=(rRi,ni,rR(i+1))
        mus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,nLi)
        nus: typ.Sequence[NDArray],  # len=d+1, elm_shape=(num_probes,nRi)
        sigmas: typ.Sequence[NDArray], # len=d+1, elm_shape=(num_probes,rR(i+1))
        taus: typ.Sequence[NDArray], # len=d+1, elm_shape=(num_probes,rL(i+1))
        use_jax: bool = False,
) -> typ.Sequence[NDArray]: # detas. len=d, elm_shape=(num_probes,ni)
    '''Compute var-downward edge variables deta.

    See Section 5.2.3, particularly Formula (40), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    t3_compute_dxis
    t3_compute_sigmas
    t3_compute_taus
    t3_assemble_tangent_probes
    t3tangent_probes
    '''
    xnp = jnp if use_jax else np

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


def t3_assemble_tangent_probes(
        basis_cores: typ.Sequence[NDArray],  # len=d. elm_shape=(ni,Ni)
        var_basis_cores: typ.Sequence[NDArray], # len=d. elm_shape=(nOi,Ni)
        etas: typ.Sequence[NDArray], # etas. len=d, elm_shape=(num_probes,ni)
        detas: typ.Sequence[NDArray], # detas. len=d, elm_shape=(num_probes,ni)
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]: # probes. len=d, elm_shape=(num_probes,Ni)
    '''Assemble tangent vector probes from edge variables.

    See Section 5.2.3, particularly Formula (41), in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    t3_compute_dxis
    t3_compute_sigmas
    t3_compute_taus
    t3_compute_detas
    t3tangent_probes
    '''
    xnp = jnp if use_jax else np

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


def t3tangent_probes(
        x:      t3m.T3Tangent, # shape=(N1,...,Nd)
        ww:     typ.Sequence[NDArray], # input vectors, len=d, elm_shape=(Ni,) or (num_probes,Ni)
        use_jax: bool = False,
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
    use_jax: bool
        If True, use jax for linear algebra operations. Otherwise, use numpy.

    Returns
    -------
    typ.Tuple[NDArray,...]
        Probes, zz. len=d, elm_shape=(Ni,) or (num_probes,Ni)

    See Also
    --------
    t3_probes
    t3tangent_probes_transpose
    t3_compute_xis
    t3_compute_mus
    t3_compute_nus
    t3_compute_etas
    t3_compute_dxis
    t3_compute_sigmas
    t3_assemble_probes
    t3_compute_detas
    t3_assemble_tangent_probes

    Examples
    --------

    Probe tangent with one set of vectors:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.t3_manifold as t3m
    >>> import t3tools.t3_probing as t3p
    >>> import t3tools.dense as dense
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(1,3,4,1)))
    >>> base, _ = t3m.t3_orthogonal_representations(p)
    >>> x = t3m.t3tangent_randn(base)
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.t3tangent_probes(x, ww)
    >>> zz2 = dense.dense_probes(t3m.t3tangent_to_dense(x), ww)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [4.6257812371663175e-15, 3.628238740198284e-15, 5.6097341748343224e-15]

    Probe tangent with two sets of vectors:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.t3_manifold as t3m
    >>> import t3tools.t3_probing as t3p
    >>> import t3tools.dense as dense
    >>> p = t3.t3_corewise_randn(((10,11,12),(5,6,4),(1,3,4,1)))
    >>> base, _ = t3m.t3_orthogonal_representations(p)
    >>> x = t3m.t3tangent_randn(base)
    >>> www = (np.random.randn(2,10), np.random.randn(2,11), np.random.randn(2,12))
    >>> zzz = t3p.t3tangent_probes(x, www) # Compute probes!
    >>> zzz2 = dense.dense_probes(t3m.t3tangent_to_dense(x), www)
    >>> print([np.linalg.norm(zz - zz2, axis=1) for zz, zz2 in zip(zzz, zzz2)])
    [array([3.18560984e-15, 5.06339604e-15]), array([1.74264349e-15, 5.10008230e-15]), array([2.17576097e-15, 2.94156728e-15])]
    '''
    x_shape = tuple([B.shape[1] for B in x[1][0]])
    assert(len(ww) == len(x_shape))

    vectorized = True
    if len(ww[0].shape) == 1:
        vectorized = False
        ww = [w.reshape((1,-1)) for w in ww]

    assert(tuple([w.shape[1] for w in ww]) == x_shape)

    ((basis_cores, left_tt_cores, right_tt_cores, outer_tt_cores),
     (var_basis_cores, var_tt_cores)) = x

    xis = t3_compute_xis(
        basis_cores, ww, use_jax,
    )

    mus = t3_compute_mus(
        left_tt_cores, xis, use_jax=use_jax,
    )

    nus = t3_compute_nus(
        right_tt_cores, xis, use_jax=use_jax,
    )

    etas = t3_compute_etas(
        outer_tt_cores, mus, nus, use_jax=use_jax,
    )

    dxis = t3_compute_dxis(
        var_basis_cores, ww, use_jax=use_jax,
    )

    sigmas = t3_compute_sigmas(
        var_tt_cores, right_tt_cores, outer_tt_cores,
        xis, dxis, mus,
        use_jax=use_jax,
    )

    taus = t3_compute_taus(
        var_tt_cores, left_tt_cores, outer_tt_cores,
        xis, dxis, nus,
        use_jax=use_jax,
    )

    detas = t3_compute_detas(
        var_tt_cores, left_tt_cores, right_tt_cores,
        mus, nus, sigmas, taus,
        use_jax=use_jax,
    )

    zz = t3_assemble_tangent_probes(
        basis_cores, var_basis_cores,
        etas, detas,
        use_jax=use_jax,
    )

    if not vectorized:
        zz = tuple([z.reshape(-1) for z in zz])
    return zz


# def t3_assemble_tangent_actions(
#         TS,
#         tt_variations,
#         basis_variations,
#         mus,
#         nus,
#         sigmas,
#         taus,
# ):
#     '''Form actions of a Tucker tensor train tangent vector from already computed mus, nus, sigmas, taus'''
#     actions = []
#     for ii in range(TS.num_cores):
#         P = TS.left_orthogonal_tt_cores[ii]
#         Q = TS.right_orthogonal_tt_cores[ii]
#         R = TS.up_orthogonal_tt_cores[ii]
#         dU = tt_variations[ii]
#         B = TS.orthogonal_basis_cores[ii]
#         dB = basis_variations[ii]
#
#         mu = mus[ii]
#         nu = nus[ii]
#         sigma = sigmas[ii]
#         tau = taus[ii]
#
#         # Actions given by following formula:
#         # a(z) = [sigma, mu] [Q(B z),                 0] [nu]
#         #                    [dU(B z) + R(dB z), P(B z)] [tau]
#
#         reduced_action_t1 = sigma @ (Q @ nu) #jnp.einsum('i,iaj,j->a', sigma, Q, nu)
#         reduced_action_t2 = mu @ (dU @ nu) #jnp.einsum('i,iaj,j->a', mu,    dU, nu)
#         reduced_action_t3 = mu @ (R @ nu) #jnp.einsum('i,iaj,j->a', mu,    R, nu)
#         reduced_action_t4 = mu @ (P @ tau) #jnp.einsum('i,iaj,j->a', mu,    P, tau)
#
#         action = (reduced_action_t1 + reduced_action_t2 + reduced_action_t4) @ B + reduced_action_t3 @ dB
#         actions.append(action)
#     return tuple(actions)
#
#
# def t3_tangent_actions(
#         u:  T3Variations,
#         TS: T3TangentSpace, # shape=(N1, N2, ..., Nk)
#         input_vectors:  typ.Sequence[jnp.ndarray], # inputs, len=k, elm_shape=(Ni,)
# ) -> typ.Tuple[jnp.ndarray,...]: # len=k, elm_shape=(Ni,)
#     '''Compute actions of a Tucker tensor train tangent vector.
#         u(x,y,z,w) = G1(x) G2(y) G3(z) G4(w)
#         where:
#             G1(x) = [dU1(B x) + R1(dB x), P1(B x)]
#
#             G2(y) = [Q2(B y),                   0]
#                     [dU2(B y) + R2(dB y), P2(B y)]
#
#             G3(z) = [Q3(B z),                   0]
#                     [dU3(B z) + R3(dB z), P3(B z)]
#
#             G4(w) = [Q4(B w) ]
#                     [dU4(B w) + R4(dB w)]
#
#
#     '''
#     TS._check_consistency_with_other_ttt_cores(u)
#     basis_variations, tt_variations = u
#
#     reduced_xx  = [B  @ x for B,  x in zip(TS.orthogonal_basis_cores, input_vectors)]
#     reduced_dxx = [dB @ x for dB, x in zip(basis_variations, input_vectors)]
#
#     mus     = t3_compute_mus(TS, reduced_xx)
#     sigmas  = t3_compute_sigmas(tt_variations, TS, reduced_xx, reduced_dxx, mus)
#     nus     = t3_compute_nus(TS, reduced_xx)
#     taus    = t3_compute_taus(tt_variations, TS, reduced_xx, reduced_dxx, nus)
#
#     actions = t3_assemble_tangent_actions(TS, tt_variations, basis_variations, mus, nus, sigmas, taus)
#     return actions
#
#
# #
#
# def t3_compute_tau_tildes(
#         TS,
#         reduced_xx,
#         reduced_dyy,
#         mus,
# ):
#     '''Adjoints for right-to-left pushthrough partial sums (adjoints go the other way, left to right).
#     Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.
#     '''
#     tau_tildes = [jnp.zeros(1)]
#     for ii in range(TS.num_cores-1):
#         P = TS.left_orthogonal_tt_cores[ii]
#         x = reduced_xx[ii]
#         dy = reduced_dyy[ii]
#
#         mu = mus[ii]
#         tau_tilde = tau_tildes[-1]
#
#         tau_tilde_next_t1   = tau_tilde @ jnp.einsum('iaj,a->ij', P, x)
#         tau_tilde_next_t2   = mu        @ jnp.einsum('iaj,a->ij', P, dy)
#
#         tau_tilde_next = tau_tilde_next_t1 + tau_tilde_next_t2
#         tau_tildes.append(tau_tilde_next)
#     return tuple(tau_tildes)
#
#
# def t3_compute_sigma_tildes(
#         TS,
#         reduced_xx,
#         reduced_dyy,
#         nus,
# ):
#     '''Adjoints for left-to-right pushthrough partial sums (adjoints go the other way, right-to-left).
#     Used for computing transpose of mapping from a Tucker tensor train tangent vector to its actions.
#     '''
#     sigma_tildes_reversed = [jnp.zeros(1)]
#     for ii in range(TS.num_cores-1, 0, -1):
#         Q = TS.right_orthogonal_tt_cores[ii]
#         x = reduced_xx[ii]
#         dy = reduced_dyy[ii]
#
#         nu = nus[ii]
#         sigma_tilde = sigma_tildes_reversed[-1]
#
#         sigma_tilde_prev_t1 = jnp.einsum('iaj,a->ij', Q, x)  @ sigma_tilde
#         sigma_tilde_prev_t2 = jnp.einsum('iaj,a->ij', Q, dy) @ nu
#
#         sigma_tilde_prev = sigma_tilde_prev_t1 + sigma_tilde_prev_t2
#         sigma_tildes_reversed.append(sigma_tilde_prev)
#     sigma_tildes = sigma_tildes_reversed[::-1]
#     return tuple(sigma_tildes)
#
#
# def t3_assemble_core_perturbations(
#         TS,
#         xx,
#         dyy,
#         reduced_xx,
#         reduced_dyy,
#         mus,
#         nus,
#         sigma_tildes,
#         tau_tildes,
# ):
#     '''Apply transpose of mapping from Tucker tensor train tangent vector to its actions,
#     using already computed mus, nus, sigmas, taus.
#     '''
#     tt_core_perturbations = []
#     for ii in range(TS.num_cores):
#         mu = mus[ii]
#         nu = nus[ii]
#         sigma_tilde = sigma_tildes[ii]
#         tau_tilde = tau_tildes[ii]
#         x_hat = reduced_xx[ii]
#         dy_hat = reduced_dyy[ii]
#
#         dU_t1 = jnp.einsum('i,a,j->iaj', mu,        x_hat,  sigma_tilde)
#         dU_t2 = jnp.einsum('i,a,j->iaj', tau_tilde, x_hat,  nu)
#         dU_t3 = jnp.einsum('i,a,j->iaj', mu,        dy_hat, nu)
#
#         dU = dU_t1 + dU_t2 + dU_t3
#         tt_core_perturbations.append(dU)
#
#     basis_core_perturbations = []
#     for ii in range(TS.num_cores):
#         R = TS.up_orthogonal_tt_cores[ii]
#         mu = mus[ii]
#         nu = nus[ii]
#         sigma_tilde = sigma_tildes[ii]
#         tau_tilde = tau_tildes[ii]
#         x = xx[ii]
#         dy = dyy[ii]
#
#         dB_t1 = jnp.outer(jnp.einsum('i,iaj,j->a', mu,          R, sigma_tilde),    x)
#         dB_t2 = jnp.outer(jnp.einsum('i,iaj,j->a', tau_tilde,   R, nu),             x)
#         dB_t3 = jnp.outer(jnp.einsum('i,iaj,j->a', mu,          R, nu),             dy)
#
#         dB = dB_t1 + dB_t2 + dB_t3
#         basis_core_perturbations.append(dB)
#
#     return tuple(tt_core_perturbations), tuple(basis_core_perturbations)
#
#
# def t3_tangent_actions_transpose(
#         action_perturbations: typ.Sequence[jnp.ndarray],  # inputs, len=k, elm_shape=(Ni,)
#         TS: T3TangentSpace, # shape=(N1, N2, ..., Nk)
#         input_vectors:  typ.Sequence[jnp.ndarray], # inputs, len=k, elm_shape=(Ni,)
# ) -> T3Variations:
#     '''Transpose of mapping u -> ttt_tangent_actions(u, TS, input_vectors), where TS, input_vectors are fixed
#
#     '''
#     xx = input_vectors
#     dyy = action_perturbations
#
#     reduced_xx  = [B @ x for B,  x in zip(TS.orthogonal_basis_cores, xx)]
#     reduced_dyy = [B @ dy for B, dy in zip(TS.orthogonal_basis_cores, dyy)]
#
#     mus = t3_compute_mus(TS, reduced_xx)
#     nus = t3_compute_nus(TS, reduced_xx)
#     tau_tildes = t3_compute_tau_tildes(TS, reduced_xx, reduced_dyy, mus)
#     sigma_tildes = t3_compute_sigma_tildes(TS, reduced_xx, reduced_dyy, nus)
#
#     tt_core_perturbations, basis_core_perturbations = t3_assemble_core_perturbations(
#         TS, xx, dyy, reduced_xx, reduced_dyy, mus, nus, sigma_tildes, tau_tildes,
#     )
#
#     return basis_core_perturbations, tt_core_perturbations
#
#
#
