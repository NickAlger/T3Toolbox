# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The ``probe`` sampling type AND the shared sampling machinery it exemplifies.

Probing contracts a T3 with vectors in all but one mode, giving ``d`` vectors (T4S §6). This
module holds the general machinery -- the ``xi/mu/nu/eta`` helper chain, ``assemble_z``, the tv
probe + frame sweeps + transposes -- which ``apply``/``entries`` specialize (probe ⊃ apply ⊃
entries; they import from here, never the reverse). Reference module for the signature style
(``docs/contributor/signature_style.md``).
"""
import math
import numpy as np
import typing as typ

import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.t3_operations as ragged_ops
import t3toolbox.backend.ut3_operations as uniform_ops
from t3toolbox.backend.common import *

__all__ = [
    # Probe a Tucker tensor train
    't3_probe',
    'compute_xi',
    'compute_mu',
    'compute_nu',
    'compute_eta',
    'assemble_z',
    # Probe a tangent vector
    'tv_probe',
    'compute_dxi',
    'compute_sigma',
    'compute_tau',
    'compute_deta',
    'assemble_tangent_z',
    # Apply -- frame-sweep reuse split (precompute the LEAN (xi,mu) frame edge vars once; inject into the
    # bare J / Jᵀ; adjoint-state transpose recomputes the right context as sigma_hat; for fitting.py)
    'compute_sigma_hat',
    # Entries -- frame-sweep reuse split (the fiber-sliced seed; one-hot adjoint-state transpose; for fitting.py)
    # Probe -- frame-sweep reuse split (the FULL (xi,mu,nu,eta) sweep; for fitting.py)
    'tv_precompute_probe_frame_sweep',
    'tv_probe_jacobian_from_sweep',
    'tv_probe_transpose_from_sweep',
    # Corewise (non-manifold) transpose -- the tangent transpose with the frame's cores in place of the frames
    # Transpose of map from tangent vector to probes
    'compute_deta_tilde',
    'compute_tau_tilde',
    'compute_sigma_tilde',
    'compute_dxi_tilde',
    'assemble_tucker_variations',
    'assemble_tt_variations',
    'tv_probe_transpose',
    # Ambient / corewise probe transposes (the plain-probe analogs of the apply/entries transposes)
    't3_probe_ambient_transpose',
    't3_probe_corewise_transpose',
    # Probe a dense tensor
    'dense_probe',
]

# NOTE: probing is intentionally UNWEIGHTED. In the typical regime (many probes at once) it is
# cheaper to absorb any edge weights into the cores once, up front, then probe the weighted cores
# with the plain functions below (rather than threading weights through every probe). The up-front
# weighting helper is ``t3_absorb_weights`` (see ``docs/weighting.md``).


#####################################################
########    Probing a Tucker Tensor Train    ########
#####################################################

def t3_probe(
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
    tv_probe
    tv_probe_transpose
    compute_xi
    compute_mu
    compute_nu
    compute_eta
    assemble_z

    Examples
    --------
    Probe a T3 with one set of vectors; value-match against the dense reference:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1)).data
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.t3_probe(ww, x)
    >>> zz_dense = t3p.dense_probe(ww, t3.TuckerTensorTrain(*x).to_dense())   # dense reference
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
    >>> zz = t3p.t3_probe(ww, x)
    >>> zz_dense = t3p.dense_probe(ww, t3.TuckerTensorTrain(*x).to_dense())
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
    >>> zz = t3p.t3_probe(ww, x)
    >>> zz_dense = t3p.dense_probe(ww, t3.TuckerTensorTrain(*x).to_dense())
    >>> print(zz[0].shape)                  # W=(2,3) outer, C=(4,5) inner, then N0=10
    (2, 3, 4, 5, 10)
    >>> print([bool(np.allclose(z, z2)) for z, z2 in zip(zz, zz_dense)])
    [True, True, True]
    '''
    tucker_cores, tt_cores = x

    xis = compute_xi(tucker_cores, ww)

    mus = compute_mu(tt_cores, xis)

    nus = compute_nu(tt_cores, xis)

    etas = compute_eta(tt_cores, mus, nus)

    zs = assemble_z(tucker_cores, etas)

    return zs


def _xi_step(
        x: typ.Tuple[NDArray, NDArray],   # (U, w): C+(nUi,Ni) ; W+(Ni,)
) -> typ.Tuple[NDArray]:                  # (xi,): W+C+(nUi,)
    '''One core of the ragged map of :py:func:`compute_xi`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    U, w = x
    return (contractions.contract('Cio,Wo->WCi', U, w),)


def compute_xi(
        up_tucker_cores:    typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=C+(nUi,Ni)
        ww:                 typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=W+(Ni,)
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
        xis = contractions.contract('dCio,dWo->dWCi', up_tucker_cores, ww)
    else:
        (xis,) = xmap(_xi_step, (up_tucker_cores, ww))

    return xis


def _mu_step(
        mu: NDArray,                      # carry: W+C+(rLi,)
        x:  typ.Tuple[NDArray, NDArray],  # (P, xi): C+(rLi,nUi,rL(i+1)) ; W+C+(nUi,)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (mu,))
    '''One edge of the leftward sweep of :py:func:`compute_mu`. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    P, xi = x[0], x[1]
    mu_next = contractions.contract('WCa,Caib,WCi->WCb', mu, P, xi)
    return mu_next, (mu,)


def compute_mu(
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d-1, elm_shape=C+(rLi,nUi,rL(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=W+C+(nUi,)
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

    # carry has the same leading stack as the edge variables (order-agnostic), plus the left bond
    r0 = left_tt_cores[0].shape[-3]
    init = xnp.ones(xis[0].shape[:-1] + (r0,))

    last_mu, (mus,) = xscan(_mu_step, init, (left_tt_cores, xis))
    return mus


def compute_nu(
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=W+C+(nUi,)
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
    reverse = tt_operations.tt_reverse if is_uniform else tt_operations.tt_reverse

    rev_nus = compute_mu(reverse(right_tt_cores), xis[::-1])
    return rev_nus[::-1]


def _eta_step(
        x: typ.Tuple[NDArray, NDArray, NDArray],   # (mu, G, nu): W+C+(rLi,) ; C+(rLi,nOi,rR(i+1)) ; W+C+(rR(i+1),)
) -> typ.Tuple[NDArray]:                           # (eta,): W+C+(nOi,)
    '''One core of the ragged map of :py:func:`compute_eta`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    mu, G, nu = x
    return (contractions.contract('WCa,Caib,WCb->WCi', mu, G, nu),)


def compute_eta(
        down_tt_cores:         typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mus:                    typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=W+C+(rLi,)
        nus:                    typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
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
        etas = contractions.contract('dWCa,dCaib,dWCb->dWCi', mus, down_tt_cores, nus)
    else:
        (etas,) = xmap(_eta_step, (mus, down_tt_cores, nus))

    return etas


def _assemble_z_step(
        x: typ.Tuple[NDArray, NDArray],   # (eta, U): W+C+(ni,) ; C+(ni,Ni)
) -> typ.Tuple[NDArray]:                  # (z,): W+C+(Ni,)
    '''One core of the ragged map of :py:func:`assemble_z`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    eta, U = x
    return (contractions.contract('WCi,Cio->WCo', eta, U),)


def assemble_z(
        tucker_cores:   typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=C+(ni,Ni)
        etas:           typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(ni,)
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
        zs = contractions.contract('dWCi,dCio->dWCo', etas, tucker_cores)
    else:
        (zs,) = xmap(_assemble_z_step, (etas, tucker_cores))

    return zs


#####################################################
###########    Probing a tangent vector    ##########
#####################################################

def compute_dxi(
        var_tucker_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(nOi,Ni)
        ww:                     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,Ni)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # dxis. len=d, elm_shape=(...,nOi)
    '''Compute var-upward edge variables dxi.
    Used for probing a tangent vector.

    Same as compute_xi(), except with var_tucker_cores in place of tucker_cores.

    See Section 6.2.2, particularly Algorithm 7, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_xi
    compute_sigma
    compute_tau
    compute_deta
    assemble_tangent_z
    tv_probe
    '''
    return compute_xi(var_tucker_cores, ww)


def _sigma_step(sigma, Q, O, dG, xi, dxi, mu):
    '''One step of the perturbation-leftward (sigma) recursion (Algorithm 7), shared by
    compute_sigma (which keeps the per-core sequence, for probing) and tv_apply/tv_entries
    (which keep only the terminal carry, for the all-modes contraction).

    Three-group (W probe, K tangent, C frame): sigma/dxi carry K, the frame edge vars (xi, mu) and frame
    cores (Q, O) do not. t1/t3 self-infer the split from the C-only frame core; t2's only core is the
    variation core dG (K+C), so len(C) is supplied via n_frame (recovered from the C-only Q, the
    n_probe precedent). Reduces to the two-group result when K is empty.
    '''
    n_frame = Q.ndim - 3
    t1 = contractions.contract('WKCa,Caib,WCi->WKCb', sigma, Q, xi)
    t2 = contractions.contract('WCa,KCaib,WCi->WKCb', mu, dG, xi, len_C=n_frame)
    t3 = contractions.contract('WCa,Caib,WKCi->WKCb', mu, O, dxi)
    return t1 + t2 + t3


def _sigma_sweep_step(
        sigma: NDArray,   # carry: W+K+C+(rRi,)
        x:     typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray],  # (Q, O, dG, xi, dxi, mu)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (sigma,))
    '''One edge of the sweep of :py:func:`compute_sigma`, keeping the per-core sequence. Closure-free
    scan body -- ``docs/contributor/scan_body_principles.md``.'''
    Q, O, dG, xi, dxi, mu = x
    return _sigma_step(sigma, Q, O, dG, xi, dxi, mu), (sigma,)


def compute_sigma(
        var_tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        down_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nOi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi),
        dxis:               typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nOi)
        mus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rLi)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # sigmas. len=d, elm_shape=(...,rRi)
    '''Compute var-leftward edge variables sigma.
    Used for probing a tangent vector.

    See Section 6.2.2, particularly Algorithm 7, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

    See Also
    --------
    compute_dxi
    compute_tau
    compute_deta
    assemble_tangent_z
    tv_probe
    '''
    use_jax = tree_contains_jax((var_tt_cores, right_tt_cores, down_tt_cores, xis, dxis, mus))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    # carry sigma is W+K+C; take the leading stack from dxis (which carries K), not xis (W+C only)
    rR0 = right_tt_cores[0].shape[-3]
    init = xnp.zeros(dxis[0].shape[:-1] + (rR0,))

    last_sigma, (sigmas,) = xscan(_sigma_sweep_step, init, (right_tt_cores, down_tt_cores, var_tt_cores, xis, dxis, mus))
    return sigmas


def compute_tau(
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
    compute_dxi
    compute_sigma
    compute_deta
    assemble_tangent_z
    tv_probe
    '''
    is_uniform = is_ndarray(var_tt_cores)
    reverse = tt_operations.tt_reverse if is_uniform else tt_operations.tt_reverse

    rev_taus = compute_sigma(
        reverse(var_tt_cores), reverse(left_tt_cores), reverse(down_tt_cores),
        xis[::-1], dxis[::-1], nus[::-1],
    )
    return rev_taus[::-1]


def _deta_step(
        x: typ.Tuple[NDArray, NDArray, NDArray,
                     NDArray, NDArray, NDArray, NDArray],   # (P, Q, dG, mu, nu, sigma, tau)
) -> typ.Tuple[NDArray]:                                    # (deta,): W+K+C+(nUi,)
    '''One core of the ragged map of :py:func:`compute_deta`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    P, Q, dG, mu, nu, sigma, tau = x
    # Three-group contractions (see compute_sigma): sigma/tau carry K, mu/nu and frame cores
    # P/Q do not. term1/term3 self-infer; term2's only core is dG (K+C) -> n_frame from Q.
    n_frame = Q.ndim - 3
    term1 = contractions.contract('WKCa,Caib,WCb->WKCi', sigma, Q, nu)
    term2 = contractions.contract('WCa,KCaib,WCb->WKCi', mu, dG, nu, len_C=n_frame)
    term3 = contractions.contract('WCa,Caib,WKCb->WKCi', mu, P, tau)
    return (term1 + term2 + term3,)


def compute_deta(
        var_tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rL(i+1))
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rRi,nUi,rR(i+1))
        mus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rLi)
        nus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
        sigmas:             typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rRi)
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
    compute_dxi
    compute_sigma
    compute_tau
    assemble_tangent_z
    tv_probe
    '''
    use_jax = tree_contains_jax((var_tt_cores, left_tt_cores, right_tt_cores, mus, nus, sigmas, taus))
    is_uniform = not isinstance(mus, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed WKC contractions (3b-6a): sigmas/taus carry K (W+K+C); mus/nus and the frame
        # supercores are W+C / C-only; the variation supercore var_tt_cores is K+C. n_frame = len(C),
        # read off the C-only frame supercore (d,)+C+(rR,nU,rR). The ragged xmap branch below is the oracle.
        n_frame = right_tt_cores.ndim - 4
        term1 = contractions.contract('dWKCa,dCaib,dWCb->dWKCi', sigmas, right_tt_cores, nus)
        term2 = contractions.contract('dWCa,dKCaib,dWCb->dWKCi', mus, var_tt_cores, nus, len_C=n_frame)
        term3 = contractions.contract('dWCa,dCaib,dWKCb->dWKCi', mus, left_tt_cores, taus)
        detas = term1 + term2 + term3
    else:
        xs = (left_tt_cores, right_tt_cores, var_tt_cores, mus, nus, sigmas, taus)
        (detas,) = xmap(_deta_step, xs)

    return detas


def _assemble_tangent_z_step(
        x: typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # (B, dB, eta, deta)
) -> typ.Tuple[NDArray]:                                    # (z,): W+K+C+(Ni,)
    '''One core of the ragged map of :py:func:`assemble_tangent_z`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    B, dB, eta, deta = x
    # Three-group contractions (see compute_sigma): deta carries K (in term1 both W and K ride
    # passively over the C-only frame core B, so no split is needed); eta is W+C and dB is the
    # variation core K+C, so term2 needs len(C) -- recovered here from the C-only core B.
    n_frame = B.ndim - 2
    term1 = contractions.contract('WKCi,Cio->WKCo', deta, B)
    term2 = contractions.contract('WCi,KCio->WKCo', eta, dB, len_C=n_frame)
    return (term1 + term2,)


def assemble_tangent_z(
        tucker_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(nUi,Ni)
        var_tucker_cores:   typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(nOi,Ni)
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
    compute_dxi
    compute_sigma
    compute_tau
    compute_deta
    tv_probe
    '''
    use_jax = tree_contains_jax((tucker_cores, var_tucker_cores, etas, detas))
    is_uniform = not isinstance(etas, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed (3b-6a): lift the edge vars through the tucker supercores. detas is W+K+C, lifted
        # through the C-only frame tucker core (W and K ride passively there, so no split is needed); etas
        # is W+C, lifted through the K+C variation tucker core (n_frame = len(C)). Ragged xmap is the oracle.
        n_frame = tucker_cores.ndim - 3
        term1 = contractions.contract('dWKCi,dCio->dWKCo', detas, tucker_cores)
        term2 = contractions.contract('dWCi,dKCio->dWKCo', etas, var_tucker_cores, len_C=n_frame)
        zs = term1 + term2
    else:
        (zs,) = xmap(_assemble_tangent_z_step, (tucker_cores, var_tucker_cores, etas, detas))

    return zs


def tv_probe_jacobian_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],
        ww:         typ.Sequence[NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,) -- for the variation's dxis
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = tv_precompute_probe_frame_sweep(frame, ww)
) -> typ.Sequence[NDArray]:                 # probes, len=d, elm_shape=W+K+C+(Ni,) (one free mode each)
    '''Forward probe of a tangent vector reusing a precomputed frame sweep -- the bare ``𝒥`` (probe) with
    the frame edge variables injected. Equivalent to :py:func:`tv_probe`, but takes
    ``(xis, mus, nus, etas)`` from ``frame_sweep`` instead of recomputing them; only the perturbation
    sweep (``dxis``/``sigmas``/``taus``/``detas``) is computed here. Apply and probe **share** the frame
    sweep (:py:func:`tv_precompute_probe_frame_sweep`). No gauge projector ``Π``.

    See Also
    --------
    tv_precompute_probe_frame_sweep
    tv_probe
    tv_probe_transpose_from_sweep
    '''
    var_tucker_cores, var_tt_cores = variation
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    xis, mus, nus, etas = frame_sweep
    dxis   = compute_dxi(var_tucker_cores, ww)
    sigmas = compute_sigma(var_tt_cores, right_tt_cores, down_tt_cores, xis, dxis, mus)
    taus   = compute_tau(var_tt_cores, left_tt_cores, down_tt_cores, xis, dxis, nus)
    detas  = compute_deta(var_tt_cores, left_tt_cores, right_tt_cores, mus, nus, sigmas, taus)
    return assemble_tangent_z(up_tucker_cores, var_tucker_cores, etas, detas)


def tv_probe(
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
        frame:       typ.Union[
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
        ], # frame order = T3Frame.data = (up, down, left, right) = (U, O, P, Q)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,Ni)
    '''Probe a tangent vector. Applies the (single-sample) least-squares Jacobian J^(s).

    Two independent stackings may ride along (handled by the W/C custom contractions in
    ``contractions.py``): the T3 stack ``C`` (the frame/variation cores' ``stack_shape``) and the
    probe stack ``W`` (the probing vectors' batch). When both are present the probes are
    double-stacked, ``elm_shape = W + C + (Ni,)`` (probe outer, frame inner).

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
    frame: (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
        Orthogonal frame for the point where the tangent space attaches to the manifold. This is
        exactly ``T3Frame.data`` order (U, O, P, Q) -- pass ``frame.data`` directly, no reorder.

    Returns
    -------
    typ.Tuple[NDArray,...]
        Probes, zz. len=d, elm_shape=(Ni,) or (...,Ni)

    See Also
    --------
    t3_probe
    tv_probe_transpose
    compute_dxi
    compute_sigma
    compute_tau
    compute_deta
    assemble_tangent_z

    Examples
    --------

    Probe a tangent vector with one set of vectors; value-match against the dense reference:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.frame_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
    >>> frame, variations = bvf.t3_orthogonal_representations(x)
    >>> probe_frame = frame.data  # probing's frame order == T3Frame.data, no reorder
    >>> v = t3m.T3Tangent(frame, variations)
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3p.tv_probe(ww, variations.data, probe_frame)
    >>> zz_dense = t3p.dense_probe(ww, v.to_dense())   # dense reference J^(s) v
    >>> print([z.shape for z in zz])        # one probe per mode, elm_shape=(Ni,)
    [(10,), (11,), (12,)]
    >>> print([bool(np.allclose(z, z2)) for z, z2 in zip(zz, zz_dense)])
    [True, True, True]

    Probe with a stack of vectors: the probe stack ``W`` rides through, ``elm_shape = W + (Ni,)``:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.frame_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
    >>> frame, variations = bvf.t3_orthogonal_representations(x)
    >>> v = t3m.T3Tangent(frame, variations)
    >>> www = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
    >>> zzz = t3p.tv_probe(www, variations.data, frame.data)
    >>> zzz_dense = t3p.dense_probe(www, v.to_dense())
    >>> print(zzz[0].shape)                 # W=(2,) outer, then N0=10
    (2, 10)
    >>> print([bool(np.allclose(z, z2)) for z, z2 in zip(zzz, zzz_dense)])
    [True, True, True]
    '''
    (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores) = frame
    (var_tucker_cores, var_tt_cores) = variation

    xis = compute_xi(up_tucker_cores, ww)

    mus = compute_mu(left_tt_cores, xis)

    nus = compute_nu(right_tt_cores, xis)

    etas = compute_eta(down_tt_cores, mus, nus)

    dxis = compute_dxi(var_tucker_cores, ww)

    sigmas = compute_sigma(
        var_tt_cores, right_tt_cores, down_tt_cores, xis, dxis, mus,
    )

    taus = compute_tau(
        var_tt_cores, left_tt_cores, down_tt_cores, xis, dxis, nus,
    )

    detas = compute_deta(
        var_tt_cores, left_tt_cores, right_tt_cores, mus, nus, sigmas, taus,
    )

    zz = assemble_tangent_z(
        up_tucker_cores, var_tucker_cores, etas, detas,
    )

    return zz


#####################################################
#####    Apply / entries of a tangent vector    #####
#####################################################
#
# apply and entries are the all-modes special case of probing (probing leaves ONE mode free; these
# contract EVERY mode). With no free mode the whole computation collapses to a single left-to-right
# pass: the frame left sweep mu-hat (via P) feeds the perturbation sweep sigma (via Q, Algorithm 7),
# which is then contracted at the terminal bond. No right (nu) sweep, no central (eta), no per-mode
# assembly -- roughly half of tv_probe. entries is apply with the up-index xis obtained by
# slicing Tucker-core fibers (no contraction with unit vectors, so no N factor).


def tv_precompute_probe_frame_sweep(
        frame:   typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # frame order = T3Frame.data = (up, down, left, right)
        ww:     typ.Sequence[NDArray],      # probe vectors, len=d, elm_shape=W+(Ni,)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xis.  len=d, elm_shape=W+C+(nUi,)
    typ.Sequence[NDArray],  # mus.  len=d, elm_shape=W+C+(rLi,)
    typ.Sequence[NDArray],  # nus.  len=d, elm_shape=W+C+(rR(i+1),)
    typ.Sequence[NDArray],  # etas. len=d, elm_shape=W+C+(nOi,)
]:                                          # full frame sweep -- (xis, mus, nus, etas)
    '''The **probe** frame sweep (full): all four frame edge variables ``(xi, mu, nu, eta)``. The probe
    leaves one mode free, so its transpose's per-mode (vector) residual must be propagated through both
    the left ``mu`` and right ``nu`` sweeps + the central ``eta`` combine -- it cannot use the
    scalar-residual adjoint-state shortcut that lets apply/entries drop ``nu``/``eta``
    (:py:func:`tv_precompute_apply_frame_sweep`). Reused across the probe forward / transpose of an inner
    solve. §6.2.2 of Alger et al. (2026); no gauge projector ``Π``.

    See Also
    --------
    tv_probe_jacobian_from_sweep
    tv_probe_transpose_from_sweep
    tv_precompute_apply_frame_sweep
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    xis  = compute_xi(up_tucker_cores, ww)
    mus  = compute_mu(left_tt_cores, xis)
    nus  = compute_nu(right_tt_cores, xis)
    etas = compute_eta(down_tt_cores, mus, nus)
    return xis, mus, nus, etas


def _sigma_hat_step(
        carry: NDArray,                      # W+K+C+(rR(i+1),)
        data:  typ.Tuple[NDArray, NDArray],  # (Q, xi): C+(rRi,nUi,rR(i+1)) ; W+C+(nUi,)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (carry,))
    '''One edge of the adjoint reverse sweep of :py:func:`compute_sigma_hat`. Closure-free scan body
    -- ``docs/contributor/scan_body_principles.md``.'''
    Q, xi = data
    return contractions.contract('WKCa,Caib,WCi->WKCb', carry, Q, xi), (carry,)


def compute_sigma_hat(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xis:            typ.Sequence[NDArray],  # frame up-index edge vars, len=d, elm_shape=W+C+(nUi,)
        c:              NDArray,                # residual (scalar), shape=W+K+C
) -> typ.Sequence[NDArray]:                     # sigma_hats. len=d, elm_shape=W+K+C+(rR(i+1),)
    '''Propagation-only adjoint **reverse** sweep via ``Q``, seeded at the terminal bond by the residual
    ``c`` -- the order-0 (non-jet) analog of :py:func:`t3toolbox.backend.sampling_derivatives.compute_sigma_hat_jets`.

    The right context the apply/entries transpose needs, **recomputed** from ``c`` rather than stored:
    this is the low-memory half of the adjoint-state method (no ``nu``/``eta`` precomputed). ``sigma_hats[i]``
    is the adjoint of the after-core-``i`` carry; it carries the tangent stack ``K`` (from ``c``).
    Right-to-left via ``tt_reverse`` (mirroring the forward ``nu`` sweep's reversal).'''
    use_jax = tree_contains_jax((right_tt_cores, xis, c))
    is_uniform = is_ndarray(right_tt_cores)
    reverse = tt_operations.tt_reverse if is_uniform else tt_operations.tt_reverse
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    rev_Q = reverse(right_tt_cores)
    rev_xi = xis[::-1]
    # The forward sums the terminal bond (rR_d, not necessarily 1 -- e.g. the corewise frame's own cores),
    # so the adjoint BROADCASTS c over it: seed = c (x) 1_{rR_d} -> W+K+C+(rR_d,).
    rR_d = right_tt_cores[-1].shape[-1]
    seed = xnp.broadcast_to(c[..., None], tuple(c.shape) + (rR_d,))

    _, (rev_sigma_hats,) = xscan(_sigma_hat_step, seed, (rev_Q, rev_xi))
    return rev_sigma_hats[::-1]


###############################################################
###########    Transpose of tangent to probes map    ##########
###############################################################

def _deta_tilde_step(
        x: typ.Tuple[NDArray, NDArray],   # (U, zt): C+(nUi,Ni) ; W+K+C+(Ni,)
) -> typ.Tuple[NDArray]:                  # (deta_tilde,): W+K+C+(nUi,)
    '''One core of the ragged map of :py:func:`compute_deta_tilde`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    U, zt = x
    # C (T3 stack) is shared between the core U and the residual zt; W is the probe stack on
    # zt. This is NOT compute_xi (which forms an outer product over the two stacks).
    return (contractions.contract('WCo,Cio->WCi', zt, U),)


def compute_deta_tilde(
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
        # C IS a shared batch on both operands (the residual ztildes carries the frame stack C, not only
        # the probe stack W), so this is the SHARED-C contraction dWCo_dCio_to_dWCi -- NOT the outer-product
        # dCio_dWo_to_dWCi (which assumes no shared C and is wrong for a C-stacked tangent). Mirrors the
        # ragged contract('WCo,Cio->WCi', zt, U); the ragged xmap branch below is the oracle.
        deta_tildes = contractions.contract('dWCo,dCio->dWCi', ztildes, up_tucker_cores)
    else:
        (deta_tildes,) = xmap(_deta_tilde_step, (up_tucker_cores, ztildes))

    return deta_tildes


def _tau_tilde_step(
        tau_tilde: NDArray,   # carry: W+K+C+(rLi,)
        x:         typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (P, xi, deta_tilde, mu)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (tau_tilde,))
    '''One edge of the adjoint rightward sweep of :py:func:`compute_tau_tilde`. Closure-free scan
    body -- ``docs/contributor/scan_body_principles.md``.'''
    P, xi, deta_tilde, mu = x
    # Three-group (W probe, K tangent, C frame): tau_tilde/deta_tilde carry K (from the residual),
    # xi/mu and the frame core P do not. Both terms self-infer the split (P pins C, xi/mu pin W);
    # reduces to the two-group result when K is empty (no K-stacked residual).
    t1 = contractions.contract('WKCa,Caib,WCi->WKCb', tau_tilde, P, xi)
    t2 = contractions.contract('WCa,Caib,WKCi->WKCb', mu, P, deta_tilde)
    tau_tilde_next = t1 + t2
    return tau_tilde_next, (tau_tilde,)


def compute_tau_tilde(
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

    # carry tau_tilde is W+K+C; take the leading stack from deta_tildes (carries K), not mus (W+C).
    init = xnp.zeros(deta_tildes[0].shape[:-1] + (left_tt_cores[0].shape[-3],))

    last_tau_tilde, (tau_tildes,) = xscan(_tau_tilde_step, init, (left_tt_cores, xis, deta_tildes, mus))
    return tau_tildes


def compute_sigma_tilde(
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
    reverse = tt_operations.tt_reverse if is_uniform else tt_operations.tt_reverse

    return compute_tau_tilde(
        deta_tildes[::-1], reverse(right_tt_cores), xis[::-1], nus[::-1],
    )[::-1]


def _dxi_tilde_step(
        x: typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray],   # (O, mu, nu, st, tt)
) -> typ.Tuple[NDArray]:                                             # (dxi_tilde,): W+K+C+(nOi,)
    '''One core of the ragged map of :py:func:`compute_dxi_tilde`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    O, mu, nu, st, tt = x
    # Three-group (see compute_tau_tilde): tt/st carry K, mu/nu and the frame core O do not.
    # Both terms self-infer (O pins C, mu/nu pin W).
    term1 = contractions.contract('WKCa,Caib,WCb->WKCi', tt, O, nu)
    term2 = contractions.contract('WCa,Caib,WKCb->WKCi', mu, O, st)
    return (term1 + term2,)


def compute_dxi_tilde(
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
        # d-prefixed WKC (3b-6a): tau_tildes/sigma_tildes carry K (W+K+C); mus/nus are W+C; the frame
        # supercore down_tt_cores (O) is C-only. Mirrors the ragged calls; the ragged xmap is the oracle.
        term1 = contractions.contract('dWKCa,dCaib,dWCb->dWKCi', tau_tildes, down_tt_cores, nus)
        term2 = contractions.contract('dWCa,dCaib,dWKCb->dWKCi', mus, down_tt_cores, sigma_tildes)
        dxi_tildes = term1 + term2
    else:
        xs = (down_tt_cores, mus, nus, sigma_tildes, tau_tildes)
        (dxi_tildes,) = xmap(_dxi_tilde_step, xs)

    return dxi_tildes


def _assemble_tucker_variations_core(
        x:               typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (z_tilde, eta, w, dxi_tilde)
        sum_over_probes: bool,     # caller intent -- no operand carries it; hence the two bodies below
) -> typ.Tuple[NDArray]:           # (dU_tilde,): W+K+C+(nOi,Ni), or K+C+(nOi,Ni) when summed
    '''One core of the ragged map of :py:func:`assemble_tucker_variations`, shared by the two
    closure-free map bodies below -- ``docs/contributor/scan_body_principles.md``.'''
    z_tilde, eta, w, dxi_tilde = x
    # Three-group (W probe, K tangent, C frame): z_tilde/dxi_tilde carry K, eta does not, w is
    # W-only. n_probe = len(W) is recovered locally from the W-only probe vector w (the
    # z_tilde (x) eta term needs it; the w (x) dxi_tilde term self-infers W from w). Output
    # keeps K always; W is summed (K+C) or kept (W+K+C) per sum_over_probes. Reduces to the
    # two-group result when K is empty.
    n_probe = w.ndim - 1
    if sum_over_probes:
        dU_tilde = (
                contractions.contract('WKCo,WCa->KCao', z_tilde, eta, len_W=n_probe)
                +
                contractions.contract('Wo,WKCa->KCao', w, dxi_tilde)
        )
    else:
        dU_tilde = (
                contractions.contract('WKCo,WCa->WKCao', z_tilde, eta, len_W=n_probe)
                +
                contractions.contract('Wo,WKCa->WKCao', w, dxi_tilde)
        )
    return (dU_tilde,)


def _assemble_tucker_variations_step_summed(x):    # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=True`` variant of :py:func:`_assemble_tucker_variations_core`.'''
    return _assemble_tucker_variations_core(x, True)


def _assemble_tucker_variations_step_unsummed(x):  # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=False`` variant of :py:func:`_assemble_tucker_variations_core`.'''
    return _assemble_tucker_variations_core(x, False)


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
        # d-prefixed (3b-6a): ztildes/dxi_tildes carry K; etas is W+C; ww is the W-only packed probe
        # supercore (d,)+W+(N,) -> n_probe = len(W). The ragged xmap below is the oracle.
        n_probe = ww.ndim - 2
        if sum_over_probes:
            dU_tildes = (contractions.contract('dWKCo,dWCa->dKCao', ztildes, etas, len_W=n_probe)
                         + contractions.contract('dWo,dWKCa->dKCao', ww, dxi_tildes))
        else:
            dU_tildes = (contractions.contract('dWKCo,dWCa->dWKCao', ztildes, etas, len_W=n_probe)
                         + contractions.contract('dWo,dWKCa->dWKCao', ww, dxi_tildes))
    else:
        step = (_assemble_tucker_variations_step_summed if sum_over_probes
                else _assemble_tucker_variations_step_unsummed)
        (dU_tildes,) = xmap(step, (ztildes, etas, ww, dxi_tildes))

    return dU_tildes


def _assemble_tt_variations_core(
        x:               typ.Tuple[NDArray, NDArray, NDArray,
                                   NDArray, NDArray, NDArray],  # (xi, mu, nu, sigma_tilde, tau_tilde, deta_tilde)
        n_probe:         int,      # len(W); rides as an extra map operand -- see assemble_tt_variations
        sum_over_probes: bool,     # caller intent -- no operand carries it; hence the two bodies below
) -> typ.Tuple[NDArray]:           # (dG_tilde,): W+K+C+(rLi,nUi,rR(i+1)), or K+C+(...) when summed
    '''One core of the ragged map of :py:func:`assemble_tt_variations`, shared by the two
    closure-free map bodies below -- ``docs/contributor/scan_body_principles.md``.'''
    xi, mu, nu, sigma_tilde, tau_tilde, deta_tilde = x
    # Three-group (W probe, K tangent, C frame): the residual-derived edge vars sigma_tilde /
    # tau_tilde / deta_tilde carry K (on the j / i / a leg respectively), the frame edge vars
    # xi/mu/nu do not. No operand here is W-only or C-only, so len(W)=n_probe is supplied;
    # each contraction then derives C from an W+C operand and K from the W+K+C one. Output
    # keeps K always; W is summed (K+C) or kept (W+K+C) per sum_over_probes.
    if sum_over_probes:
        dG_tilde = (
                contractions.contract('WCi,WCa,WKCj->KCiaj', mu, xi, sigma_tilde, len_W=n_probe)
                +
                contractions.contract('WKCi,WCa,WCj->KCiaj', tau_tilde, xi, nu, len_W=n_probe)
                +
                contractions.contract('WCi,WKCa,WCj->KCiaj', mu, deta_tilde, nu, len_W=n_probe)
        )
    else:
        dG_tilde = (
                contractions.contract('WCi,WCa,WKCj->WKCiaj', mu, xi, sigma_tilde, len_W=n_probe)
                +
                contractions.contract('WKCi,WCa,WCj->WKCiaj', tau_tilde, xi, nu, len_W=n_probe)
                +
                contractions.contract('WCi,WKCa,WCj->WKCiaj', mu, deta_tilde, nu, len_W=n_probe)
        )
    return (dG_tilde,)


def _assemble_tt_variations_step_summed(x):    # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=True`` variant of :py:func:`_assemble_tt_variations_core`; ``n_probe`` rides
    as the last map operand (it is per-call data, not structure).'''
    return _assemble_tt_variations_core(x[:-1], x[-1], True)


def _assemble_tt_variations_step_unsummed(x):  # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=False`` variant of :py:func:`_assemble_tt_variations_core`; ``n_probe`` rides
    as the last map operand (it is per-call data, not structure).'''
    return _assemble_tt_variations_core(x[:-1], x[-1], False)


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
        # d-prefixed WKC triple outer products (3b-6a): K rides on the residual-derived edge var of each
        # term (sigma_tilde on j, tau_tilde on i, deta_tilde on a); the frame edge vars xi/mu/nu are W+C.
        # n_probe = len(W) (supplied). The ragged xmap below is the oracle.
        if sum_over_probes:
            dG_tildes = (contractions.contract('dWCi,dWCa,dWKCj->dKCiaj', mus, xis, sigma_tildes, len_W=n_probe)
                         + contractions.contract('dWKCi,dWCa,dWCj->dKCiaj', tau_tildes, xis, nus, len_W=n_probe)
                         + contractions.contract('dWCi,dWKCa,dWCj->dKCiaj', mus, deta_tildes, nus, len_W=n_probe))
        else:
            dG_tildes = (contractions.contract('dWCi,dWCa,dWKCj->dWKCiaj', mus, xis, sigma_tildes, len_W=n_probe)
                         + contractions.contract('dWKCi,dWCa,dWCj->dWKCiaj', tau_tildes, xis, nus, len_W=n_probe)
                         + contractions.contract('dWCi,dWKCa,dWCj->dWKCiaj', mus, deta_tildes, nus, len_W=n_probe))
    else:
        # n_probe is per-call runtime data with no W-only operand to recover it from, so it rides as
        # an extra per-core map operand (principle 6) rather than being closed over.
        step = (_assemble_tt_variations_step_summed if sum_over_probes
                else _assemble_tt_variations_step_unsummed)
        xs = (xis, mus, nus, sigma_tildes, tau_tildes, deta_tildes, (n_probe,) * len(xis))
        (dG_tildes,) = xmap(step, xs)

    return dG_tildes


def tv_probe_transpose_from_sweep(
        ztildes:    typ.Sequence[NDArray],  # probe residuals, len=d, elm_shape=W+K+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors,   len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = tv_precompute_probe_frame_sweep(frame, ww)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (dU_tildes, dG_tildes) = T3Variations.data
    '''Transpose of the probe reusing a precomputed frame sweep -- the bare ``𝒥ᵀ`` (probe) with the frame
    edge variables injected. Equivalent to :py:func:`tv_probe_transpose`, but takes
    ``(xis, mus, nus, etas)`` from ``frame_sweep`` (the reuse hook for ``fitting.py``; apply & probe share
    the sweep). No gauge projector ``Π``.

    See Also
    --------
    tv_precompute_probe_frame_sweep
    tv_probe_transpose
    tv_probe_jacobian_from_sweep
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    xis, mus, nus, etas = frame_sweep
    deta_tildes  = compute_deta_tilde(up_tucker_cores, ztildes)
    tau_tildes   = compute_tau_tilde(deta_tildes, left_tt_cores, xis, mus)
    sigma_tildes = compute_sigma_tilde(deta_tildes, right_tt_cores, xis, nus)
    dxi_tildes   = compute_dxi_tilde(sigma_tildes, tau_tildes, down_tt_cores, mus, nus)
    n_probe = ww[0].ndim - 1
    dU_tildes = assemble_tucker_variations(ztildes, dxi_tildes, ww, etas, sum_over_probes=sum_over_probes)
    dG_tildes = assemble_tt_variations(sigma_tildes, tau_tildes, deta_tildes, xis, mus, nus,
                                       sum_over_probes=sum_over_probes, n_probe=n_probe)
    return dU_tildes, dG_tildes


def tv_probe_transpose(
        ztildes:        typ.Union[typ.Sequence[NDArray],    NDArray], # len=d, elm_shape=(...,Ni)
        ww:             typ.Union[typ.Sequence[NDArray],    NDArray], # input vectors, len=d, elm_shape=(...,Ni)
        frame:           typ.Union[
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
        ], # frame order = T3Frame.data = (up, down, left, right) = (U, O, P, Q)
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
    frame: (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores)
        Orthogonal frame for the point where the tangent space attaches to the manifold. This is
        exactly ``T3Frame.data`` order (U, O, P, Q) -- pass ``frame.data`` directly, no reorder.
    sum_over_probes: bool
        Sum results over all probe residuals, rather than returning results for each probe residual.

    Returns
    -------
    (dU_tildes, dG_tildes)
        Tangent variations (a bvf.T3Variations.data tuple) resulting from applying the transpose map.

    See Also
    --------
    t3_probe
    tv_probe

    Examples
    --------

    Adjoint identity ``<z, J v> = <J^T z, v>`` with one set of probing vectors:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.frame_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
    >>> frame, _ = bvf.t3_orthogonal_representations(x)
    >>> probe_frame = frame.data  # probing's frame order == T3Frame.data, no reorder
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> v = t3m.MANIFOLD.randn(frame)
    >>> z = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> Jv  = t3p.tv_probe(ww, v.variations.data, probe_frame)
    >>> JTz = t3p.tv_probe_transpose(z, ww, probe_frame)   # (dU_tildes, dG_tildes)
    >>> lhs = cw.corewise_dot(z, Jv)                  # <z, J v>
    >>> rhs = cw.corewise_dot(JTz, v.variations.data)  # <J^T z, v>
    >>> print(bool(np.allclose(lhs, rhs)))
    True

    With ``sum_over_probes=True`` (the Gauss-Newton ``J^T r``), the adjoint identity still holds
    when a probe stack ``W`` is summed on both sides:

    >>> import numpy as np
    >>> import t3toolbox.corewise as cw
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.frame_variations_format as bvf
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
    >>> frame, _ = bvf.t3_orthogonal_representations(x)
    >>> ww = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))  # W=(2,)
    >>> v = t3m.MANIFOLD.randn(frame)
    >>> z = (np.random.randn(2, 10), np.random.randn(2, 11), np.random.randn(2, 12))
    >>> Jv  = t3p.tv_probe(ww, v.variations.data, frame.data)        # W-stacked probes
    >>> JTz = t3p.tv_probe_transpose(z, ww, frame.data, sum_over_probes=True)
    >>> lhs = cw.corewise_dot(z, Jv)                  # sum_W <z_W, (J v)_W>
    >>> rhs = cw.corewise_dot(JTz, v.variations.data)  # <sum_W J^T z_W, v>
    >>> print(bool(np.allclose(lhs, rhs)))
    True
    '''
    (up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores) = frame

    xis = compute_xi(up_tucker_cores, ww)

    mus = compute_mu(left_tt_cores, xis)

    nus = compute_nu(right_tt_cores, xis)

    etas = compute_eta(down_tt_cores, mus, nus)

    #

    deta_tildes = compute_deta_tilde(up_tucker_cores, ztildes)

    tau_tildes = compute_tau_tilde(deta_tildes, left_tt_cores, xis, mus)

    sigma_tildes = compute_sigma_tilde(deta_tildes, right_tt_cores, xis, nus)

    dxi_tildes = compute_dxi_tilde(sigma_tildes, tau_tildes, down_tt_cores, mus, nus)

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


def t3_probe_ambient_transpose(
        ztildes:    typ.Sequence[NDArray],  # probe residuals, len=d, elm_shape=W+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors,   len=d, elm_shape=W+(Ni,)
        sum_over_probes: bool = False,      # True: W folds into the CP rank
) -> typ.Sequence[NDArray]:  # canonical (CP) factors. len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`t3_probe`: back-project probe residuals into CP factors.

    The *ambient* adjoint -- the transpose of ``probe`` as a linear map on the **full tensor space**.
    Probe returns ``d`` vectors (one free mode each), so the residual ``ztildes`` is ``d`` vectors; the
    back-projection is the rank-``d`` tensor

        sum_i  w0 (x) ... (x) w_{i-1} (x) ztildes_i (x) w_{i+1} (x) ... (x) w_{d-1}

    (term ``i`` has the residual ``ztildes_i`` in slot ``i`` and the probe vectors elsewhere), whose
    natural representation is a **canonical (CP) decomposition** of rank ``d``. Frame-free. Distinct from
    the *corewise* transpose (gradient w.r.t. a frame's cores) and the *tangent* transpose (Riemannian
    gradient); see ``docs/transposes.md``. The ``apply``/``entries`` analog is the rank-1 (or rank-``|W|``)
    :py:func:`t3_apply_ambient_transpose`.

    - ``sum_over_probes=False`` (primary): ``W`` is a passthrough stacking axis -- a ``W (+ C)`` stack
      of rank-``d`` CP tensors.
    - ``sum_over_probes=True``: ``W`` folds into the CP rank -- one rank-``d|W|`` CP tensor
      ``sum_W sum_i (...)``. Cheap as CP (``O(d |W| N)``).

    Returns CP ``factors`` (factor ``k`` has the diagonal structure: rank slot ``k`` = ``ztildes_k``,
    the others = ``ww_k``), in the layout :py:func:`t3_conversions.t3_from_canonical` consumes.
    '''
    use_jax = tree_contains_jax((ztildes, ww))
    xnp, _, _ = get_backend(False, use_jax)
    d = len(ww)
    nW = ww[0].ndim - 1
    W  = ww[0].shape[:nW]                 # probe stack
    C  = ztildes[0].shape[nW:-1]          # frame stack (ztildes[i] is W + C + (Ni,))
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


def t3_probe_corewise_transpose(
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
    '''Corewise (non-manifold) transpose of :py:func:`t3_probe`: gradient of the probes w.r.t. the
    cores of the frame ``core_pair``, treated as independent variables.

    The probe analog of :py:func:`t3_apply_corewise_transpose` -- the Section 6.3 substitution
    ``P, Q, O -> G_i`` (``U`` non-orthogonal) into :py:func:`tv_probe_transpose`, i.e. that
    transpose at frame ``(U, G, G, G)``. Returns gradients shaped exactly like ``(tucker_cores,
    tt_cores)`` (a gradient, not a tensor; no ``|W|`` blow-up). For non-manifold optimizers (Adam,
    L-BFGS) fitting from probes. ``sum_over_probes=True`` sums the probe stack ``W`` (the gradient
    ``J^T r``); ``False`` keeps it. Math reference: Section 6.3, Alger et al. (2026) (arXiv:2603.21141).
    '''
    tucker_cores, tt_cores = core_pair
    return tv_probe_transpose(
        ztildes, ww, (tucker_cores, tt_cores, tt_cores, tt_cores), sum_over_probes=sum_over_probes,
    )


###############################################
##########    Probe dense tensor    ###########
###############################################

def dense_probe(
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
    >>> yy = t3p.dense_probe((u0, u1, u2), T)
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
    >>> yy = t3p.dense_probe((u0, u1, u2), T)
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
    >>> yy = t3p.dense_probe((u0, u1, u2), T)
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
