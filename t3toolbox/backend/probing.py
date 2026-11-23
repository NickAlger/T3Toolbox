# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The ``probe`` sampling type AND the shared sampling machinery it exemplifies.

Probing contracts a T3 with vectors in all but one mode, giving ``d`` vectors (T4S §6). This
module holds the general machinery -- the ``xi/mu/nu/eta`` helper chain, ``assemble_z``, the tv
probe + frame sweeps + transposes, and the corewise transpose's derivative in the cores (the exact Hessian's
two-hole) -- which ``apply``/``entries`` specialize (probe ⊃ apply ⊃
entries; they import from here, never the reverse). Reference module for the signature style
(``docs/contributor/signature_style.md``).
"""
import itertools
import math
import numpy as np
import typing as typ

import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.t3_operations as ragged_ops
import t3toolbox.backend.ut3_operations as uniform_ops
import t3toolbox.backend.sampling_variables as sampling_variables
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
    't3_probe_corewise_tangent',
    't3_probe_corewise_transpose',
    # Corewise two-hole: the probe transpose's derivative in the cores (the exact Hessian's sampling term)
    'compute_deta_tilde_tangent',
    'compute_tau_tilde_tangent',
    'compute_sigma_tilde_tangent',
    'compute_dxi_tilde_tangent',
    'assemble_tucker_tangent_variations',
    'assemble_tt_tangent_variations',
    't3_probe_corewise_hessian',
    't3_probe_corewise_hessian_from_sweep',
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
        x:                 typ.Tuple,                    # the point: X, (tucker_cores, tt_cores), elm_shapes
                                                         #   C+(nDi,Ni), C+(ri,nDi,r(i+1))
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> typ.Tuple[NDArray, ...]:               # z: P⊗V_i*, len=d, elm_shape=F+W+C+(Ni,)
    '''Probe a Tucker tensor train: contract every mode but one with the input jets, ``d`` vectors.

    One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``z`` on the point ``x``
    (a corewise call). Taylor coefficients on the jet shape ``order`` (``()`` for none).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> z = probing.t3_probe(x, ww)
    >>> print([zi.shape for zi in z])
    [(3, 4), (3, 5), (3, 6)]
    '''
    (z,) = sampling_variables.sampling_variables('probe', x, ww, ('z',), order=order)
    return z


def _xi_step(
        x: typ.Tuple[NDArray, NDArray],   # (D, w): C+(nDi,Ni) ; W+(Ni,)
) -> typ.Tuple[NDArray]:                  # (xi,): W+C+(nDi,)
    '''One core of the ragged map of :py:func:`compute_xi`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    D, w = x
    return (contractions.contract('Cio,Wo->WCi', D, w),)


def compute_xi(
        down_tucker_cores:  typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=C+(nDi,Ni)
        ww:                 typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=W+(Ni,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # xis. len=d, elm_shape=(...,nDi)
    '''Compute upward edge variables associated with edges between Tucker cores and adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 6.2, particularly Figure 7 and Algorithm 5, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((down_tucker_cores, ww))
    is_uniform = is_ndarray(down_tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        xis = contractions.contract('dCio,dWo->dWCi', down_tucker_cores, ww)
    else:
        (xis,) = xmap(_xi_step, (down_tucker_cores, ww))

    return xis


def _mu_step(
        mu: NDArray,                      # carry: W+C+(rLi,)
        x:  typ.Tuple[NDArray, NDArray],  # (L, xi): C+(rLi,nDi,rL(i+1)) ; W+C+(nDi,)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (mu,))
    '''One edge of the left-to-right sweep of :py:func:`compute_mu`. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    L, xi = x[0], x[1]
    mu_next = contractions.contract('WCa,Caib,WCi->WCb', mu, L, xi)
    return mu_next, (mu,)


def compute_mu(
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=C+(rLi,nDi,rL(i+1)) (the last core is not swept)
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=W+C+(nDi,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # mus. len=d, elm_shape=W+C+(rLi,)
    '''Compute the left edge variables mu (accumulated left to right), associated
    with edges between adjacent TT-cores. Used for probing a Tucker tensor train.

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
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=C+(rRi,nDi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=W+C+(nDi,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # nus. len=d, elm_shape=W+C+(rR(i+1),)
    '''Compute the right edge variables nu (accumulated right to left), associated
    with edges between adjacent TT-cores. Used for probing a Tucker tensor train.

    See Section 6.2, particularly Figure 7 and Algorithm 5, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    is_uniform = is_ndarray(right_tt_cores)
    reverse = tt_operations.tt_reverse

    rev_nus = compute_mu(reverse(right_tt_cores), xis[::-1])
    return rev_nus[::-1]


def _eta_step(
        x: typ.Tuple[NDArray, NDArray, NDArray],   # (mu, G, nu): W+C+(rLi,) ; C+(rLi,nUi,rR(i+1)) ; W+C+(rR(i+1),)
) -> typ.Tuple[NDArray]:                           # (eta,): W+C+(nUi,)
    '''One core of the ragged map of :py:func:`compute_eta`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    mu, G, nu = x
    return (contractions.contract('WCa,Caib,WCb->WCi', mu, G, nu),)


def compute_eta(
        up_tt_cores:            typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        mus:                    typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=W+C+(rLi,)
        nus:                    typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # etas. len=d, elm_shape=W+C+(nUi,)
    '''Compute downward edge variables associated with edges between Tucker cores and adjacent TT-cores.
    Used for probing a Tucker tensor train.

    See Section 6.2, particularly Figure 7 and Algorithm 5, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((up_tt_cores, mus, nus))
    is_uniform = is_ndarray(up_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        etas = contractions.contract('dWCa,dCaib,dWCb->dWCi', mus, up_tt_cores, nus)
    else:
        (etas,) = xmap(_eta_step, (mus, up_tt_cores, nus))

    return etas


def _assemble_z_step(
        x: typ.Tuple[NDArray, NDArray],   # (eta, B): W+C+(ni,) ; C+(ni,Ni)
) -> typ.Tuple[NDArray]:                  # (z,): W+C+(Ni,)
    '''One core of the ragged map of :py:func:`assemble_z`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    eta, B = x
    return (contractions.contract('WCi,Cio->WCo', eta, B),)


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
        var_tucker_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(nUi,Ni)
        ww:                     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,Ni)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # dxis. len=d, elm_shape=(...,nUi)
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


def _sigma_step(sigma, R, U, H, xi, dxi, mu):
    '''One step of the left-side perturbation (sigma) recursion (Algorithm 7), shared by
    compute_sigma (which keeps the per-core sequence, for probing) and tv_apply/tv_entries
    (which keep only the terminal carry, for the all-modes contraction).

    Three-group (W probe, K tangent, C frame): sigma/dxi carry K, the frame edge vars (xi, mu) and frame
    cores (R, U) do not. t1/t3 self-infer the split from the C-only frame core; t2's only core is the
    variation core H (K+C), so len(C) is supplied via n_frame (recovered from the C-only R, the
    n_probe precedent). Reduces to the two-group result when K is empty.
    '''
    n_frame = R.ndim - 3
    t1 = contractions.contract('WKCa,Caib,WCi->WKCb', sigma, R, xi)
    t2 = contractions.contract('WCa,KCaib,WCi->WKCb', mu, H, xi, len_C=n_frame)
    t3 = contractions.contract('WCa,Caib,WKCi->WKCb', mu, U, dxi)
    return t1 + t2 + t3


def _sigma_sweep_step(
        sigma: NDArray,   # carry: W+K+C+(rRi,)
        x:     typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray],  # (R, U, H, xi, dxi, mu)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (sigma,))
    '''One edge of the sweep of :py:func:`compute_sigma`, keeping the per-core sequence. Closure-free
    scan body -- ``docs/contributor/scan_body_principles.md``.'''
    R, U, H, xi, dxi, mu = x
    return _sigma_step(sigma, R, U, H, xi, dxi, mu), (sigma,)


def compute_sigma(
        var_tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nDi,rR(i+1))
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rRi,nDi,rR(i+1))
        up_tt_cores:        typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nDi),
        dxis:               typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi)
        mus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rLi)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # sigmas. len=d, elm_shape=(...,rRi)
    '''Compute the left-side variation edge variables sigma.
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
    use_jax = tree_contains_jax((var_tt_cores, right_tt_cores, up_tt_cores, xis, dxis, mus))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    # carry sigma is W+K+C; take the leading stack from dxis (which carries K), not xis (W+C only)
    rR0 = right_tt_cores[0].shape[-3]
    init = xnp.zeros(dxis[0].shape[:-1] + (rR0,))

    last_sigma, (sigmas,) = xscan(_sigma_sweep_step, init, (right_tt_cores, up_tt_cores, var_tt_cores, xis, dxis, mus))
    return sigmas


def compute_tau(
        var_tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nDi,rR(i+1))
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nDi,rL(i+1))
        up_tt_cores:        typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nUi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nDi),
        dxis:               typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi)
        nus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # taus. len=d, elm_shape=(...,rL(i+1))
    '''Compute the right-side variation edge variables tau.
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
    reverse = tt_operations.tt_reverse

    rev_taus = compute_sigma(
        reverse(var_tt_cores), reverse(left_tt_cores), reverse(up_tt_cores),
        xis[::-1], dxis[::-1], nus[::-1],
    )
    return rev_taus[::-1]


def _deta_step(
        x: typ.Tuple[NDArray, NDArray, NDArray,
                     NDArray, NDArray, NDArray, NDArray],   # (L, R, H, mu, nu, sigma, tau)
) -> typ.Tuple[NDArray]:                                    # (deta,): W+K+C+(nDi,)
    '''One core of the ragged map of :py:func:`compute_deta`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    L, R, H, mu, nu, sigma, tau = x
    # Three-group contractions (see compute_sigma): sigma/tau carry K, mu/nu and frame cores
    # L/R do not. term1/term3 self-infer; term2's only core is H (K+C) -> n_frame from R.
    n_frame = R.ndim - 3
    term1 = contractions.contract('WKCa,Caib,WCb->WKCi', sigma, R, nu)
    term2 = contractions.contract('WCa,KCaib,WCb->WKCi', mu, H, nu, len_C=n_frame)
    term3 = contractions.contract('WCa,Caib,WKCb->WKCi', mu, L, tau)
    return (term1 + term2 + term3,)


def compute_deta(
        var_tt_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nDi,rR(i+1))
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nDi,rL(i+1))
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rRi,nDi,rR(i+1))
        mus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rLi)
        nus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
        sigmas:             typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rRi)
        taus:               typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rL(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # detas. len=d, elm_shape=(...,nDi)
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
        # d-prefixed WKC contractions: sigmas/taus carry K (W+K+C); mus/nus and the frame
        # supercores are W+C / C-only; the variation supercore var_tt_cores is K+C. n_frame = len(C),
        # read off the C-only frame supercore (d,)+C+(rR,nD,rR). The ragged xmap branch below is the oracle.
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
        x: typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # (D, V, eta, deta)
) -> typ.Tuple[NDArray]:                                    # (z,): W+K+C+(Ni,)
    '''One core of the ragged map of :py:func:`assemble_tangent_z`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    D, V, eta, deta = x
    # Three-group contractions (see compute_sigma): deta carries K (in term1 both W and K ride
    # passively over the C-only frame core D, so no split is needed); eta is W+C and V is the
    # variation core K+C, so term2 needs len(C) -- recovered here from the C-only core D.
    n_frame = D.ndim - 2
    term1 = contractions.contract('WKCi,Cio->WKCo', deta, D)
    term2 = contractions.contract('WCi,KCio->WKCo', eta, V, len_C=n_frame)
    return (term1 + term2,)


def assemble_tangent_z(
        tucker_cores:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(nDi,Ni)
        var_tucker_cores:   typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(nUi,Ni)
        etas:               typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi)
        detas:              typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nDi)
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
        # d-prefixed: lift the edge vars through the tucker supercores. detas is W+K+C, lifted
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
            typ.Sequence[NDArray],          # var_tucker_cores. len=d, elm_shape=K+C+(nUi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores.     len=d, elm_shape=K+C+(rLi,nDi,rR(i+1))
        ],
        ww:         typ.Sequence[NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,) -- for the variation's dxis
        frame:      typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (D, U, L, R)
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
    down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores = frame
    xis, mus, nus, etas = frame_sweep
    dxis   = compute_dxi(var_tucker_cores, ww)
    sigmas = compute_sigma(var_tt_cores, right_tt_cores, up_tt_cores, xis, dxis, mus)
    taus   = compute_tau(var_tt_cores, left_tt_cores, up_tt_cores, xis, dxis, nus)
    detas  = compute_deta(var_tt_cores, left_tt_cores, right_tt_cores, mus, nus, sigmas, taus)
    return assemble_tangent_z(down_tucker_cores, var_tucker_cores, etas, detas)


def tv_probe(
        frame:             typ.Tuple,                    # the base point: (D, U, L, R)
        variations:        typ.Tuple,                    # X, (V, H), elm_shapes K+C+(nUi,Ni), K+C+(rLi,nDi,rR(i+1))
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> typ.Tuple[NDArray, ...]:               # Jp: P⊗V_i*, len=d, elm_shape=F+W+K+C+(Ni,)
    '''The probe of a tangent vector: ``J p``, the derivative along the direction ``variations`` at ``frame``.

    One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``Jp``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> Jp = probing.tv_probe(frame, variations, ww)
    >>> print([a.shape for a in Jp])
    [(3, 4), (3, 5), (3, 6)]
    '''
    (Jp,) = sampling_variables.sampling_variables('probe', frame, ww, ('Jp',), order=order, variations=variations)
    return Jp


#####################################################
#####    Apply / entries of a tangent vector    #####
#####################################################
#
# apply and entries are the all-modes special case of probing (probing leaves ONE mode free; these
# contract EVERY mode). With no free mode the whole computation collapses to a single left-to-right
# pass: the frame left sweep mu-hat (via L) feeds the perturbation sweep sigma (via R, Algorithm 7),
# which is then contracted at the terminal bond. No right (nu) sweep, no central (eta), no per-mode
# assembly -- roughly half of tv_probe. entries is apply with the upward xis obtained by
# slicing Tucker-core fibers (no contraction with unit vectors, so no N factor).


def tv_precompute_probe_frame_sweep(
        frame:  typ.Tuple[
            typ.Sequence[NDArray],          # down_tucker_cores  D. len=d
            typ.Sequence[NDArray],          # up_tt_cores        U. len=d
            typ.Sequence[NDArray],          # left_tt_cores      L. len=d
            typ.Sequence[NDArray],          # right_tt_cores     R. len=d
        ],                                  # frame order = T3Frame.data = (down, up, left, right)
        ww:     typ.Sequence[NDArray],      # probe vectors, len=d, elm_shape=W+(Ni,)
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xis.  len=d, elm_shape=W+C+(nDi,)
    typ.Sequence[NDArray],  # mus.  len=d, elm_shape=W+C+(rLi,)
    typ.Sequence[NDArray],  # nus.  len=d, elm_shape=W+C+(rR(i+1),)
    typ.Sequence[NDArray],  # etas. len=d, elm_shape=W+C+(nUi,)
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
    down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores = frame
    xis  = compute_xi(down_tucker_cores, ww)
    mus  = compute_mu(left_tt_cores, xis)
    nus  = compute_nu(right_tt_cores, xis)
    etas = compute_eta(up_tt_cores, mus, nus)
    return xis, mus, nus, etas


def _sigma_hat_step(
        carry: NDArray,                       # W+K+C+(rR(i+1),)
        data:  typ.Tuple[NDArray, NDArray],   # (R, xi): C+(rRi,nDi,rR(i+1)) ; W+C+(nDi,)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:  # (next carry, (carry,))
    '''One edge of the adjoint reverse sweep of :py:func:`compute_sigma_hat`. Closure-free scan body
    -- ``docs/contributor/scan_body_principles.md``.'''
    R, xi = data
    return contractions.contract('WKCa,Caib,WCi->WKCb', carry, R, xi), (carry,)


def compute_sigma_hat(
        right_tt_cores: typ.Sequence[NDArray],  # R.  len=d, elm_shape=C+(rRi,nDi,rR(i+1))
        xis:            typ.Sequence[NDArray],  # frame upward edge vars, len=d, elm_shape=W+C+(nDi,)
        c:              NDArray,                # residual (scalar), shape=W+K+C
) -> typ.Sequence[NDArray]:                     # sigma_hats. len=d, elm_shape=W+K+C+(rR(i+1),)
    '''Propagation-only adjoint **reverse** sweep via ``R``, seeded at the terminal bond by the residual
    ``c`` -- the order-0 (non-jet) analog of :py:func:`t3toolbox.backend.sampling_derivatives.compute_sigma_hat_jets`.

    The right context the apply/entries transpose needs, **recomputed** from ``c`` rather than stored:
    this is the low-memory half of the adjoint-state method (no ``nu``/``eta`` precomputed). ``sigma_hats[i]``
    is the adjoint of the after-core-``i`` carry; it carries the tangent stack ``K`` (from ``c``).
    Right-to-left via ``tt_reverse`` (mirroring the forward ``nu`` sweep's reversal).'''
    use_jax = tree_contains_jax((right_tt_cores, xis, c))
    is_uniform = is_ndarray(right_tt_cores)
    reverse = tt_operations.tt_reverse
    xnp, _, xscan = get_backend(is_uniform, use_jax)

    rev_R = reverse(right_tt_cores)
    rev_xi = xis[::-1]
    # The forward sums the terminal bond (rR_d, not necessarily 1 -- e.g. the corewise frame's own cores),
    # so the adjoint BROADCASTS c over it: seed = c (x) 1_{rR_d} -> W+K+C+(rR_d,).
    rR_d = right_tt_cores[-1].shape[-1]
    c = xnp.asarray(c)          # a bare Python float is the natural W = K = C = () residual (apply/entries)
    seed = xnp.broadcast_to(c[..., None], tuple(c.shape) + (rR_d,))

    _, (rev_sigma_hats,) = xscan(_sigma_hat_step, seed, (rev_R, rev_xi))
    return rev_sigma_hats[::-1]


###############################################################
###########    Transpose of tangent to probes map    ##########
###############################################################

def _deta_tilde_step(
        x: typ.Tuple[NDArray, NDArray],   # (D, zt): C+(nDi,Ni) ; W+K+C+(Ni,)
) -> typ.Tuple[NDArray]:                  # (deta_tilde,): W+K+C+(nDi,)
    '''One core of the ragged map of :py:func:`compute_deta_tilde`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    D, zt = x
    # C (T3 stack) is shared between the core D and the residual zt; W is the probe stack on
    # zt. This is NOT compute_xi (which forms an outer product over the two stacks).
    return (contractions.contract('WCo,Cio->WCi', zt, D),)


def compute_deta_tilde(
        down_tucker_cores:  typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(nDi,Ni)
        ztildes:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,Ni)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,nDi)
    '''Adjoint-var-upward edge variables deta_tilde.
    Used for computing the transpose of the map from a tangent vector to its probes.

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((down_tucker_cores, ztildes))
    is_uniform = is_ndarray(down_tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        # C IS a shared batch on both operands (the residual ztildes carries the frame stack C, not only
        # the probe stack W), so this is the SHARED-C contraction dWCo_dCio_to_dWCi -- NOT the outer-product
        # dCio_dWo_to_dWCi (which assumes no shared C and is wrong for a C-stacked tangent). Mirrors the
        # ragged contract('WCo,Cio->WCi', zt, D); the ragged xmap branch below is the oracle.
        deta_tildes = contractions.contract('dWCo,dCio->dWCi', ztildes, down_tucker_cores)
    else:
        (deta_tildes,) = xmap(_deta_tilde_step, (down_tucker_cores, ztildes))

    return deta_tildes


def _tau_tilde_step(
        tau_tilde: NDArray,   # carry: W+K+C+(rLi,)
        x:         typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (L, xi, deta_tilde, mu)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (tau_tilde,))
    '''One edge of the adjoint left-to-right sweep of :py:func:`compute_tau_tilde`. Closure-free scan
    body -- ``docs/contributor/scan_body_principles.md``.'''
    L, xi, deta_tilde, mu = x
    # Three-group (W probe, K tangent, C frame): tau_tilde/deta_tilde carry K (from the residual),
    # xi/mu and the frame core L do not. Both terms self-infer the split (L pins C, xi/mu pin W);
    # reduces to the two-group result when K is empty (no K-stacked residual).
    t1 = contractions.contract('WKCa,Caib,WCi->WKCb', tau_tilde, L, xi)
    t2 = contractions.contract('WCa,Caib,WKCi->WKCb', mu, L, deta_tilde)
    tau_tilde_next = t1 + t2
    return tau_tilde_next, (tau_tilde,)


def compute_tau_tilde(
        deta_tildes:        typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,nDi)
        left_tt_cores:      typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rLi,nDi,rL(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nDi)
        mus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rLi)
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,rLi)
    '''Adjoint right-side variation edge variables tau_tilde.
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
        deta_tildes:        typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nDi)
        right_tt_cores:     typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(rRi,nDi,rR(i+1))
        xis:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nDi)
        nus:                typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,rR(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # len=d, elm_shape=(...,rR(i+1))
    '''Adjoint left-side variation edge variables sigma_tilde.
    Used for computing the transpose of the map from a tangent vector to its probes.

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    is_uniform = is_ndarray(deta_tildes)
    reverse = tt_operations.tt_reverse

    return compute_tau_tilde(
        deta_tildes[::-1], reverse(right_tt_cores), xis[::-1], nus[::-1],
    )[::-1]


def _dxi_tilde_step(
        x: typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray],   # (U, mu, nu, st, tt)
) -> typ.Tuple[NDArray]:                                             # (dxi_tilde,): W+K+C+(nUi,)
    '''One core of the ragged map of :py:func:`compute_dxi_tilde`. Closure-free map body --
    ``docs/contributor/scan_body_principles.md``.'''
    U, mu, nu, st, tt = x
    # Three-group (see compute_tau_tilde): tt/st carry K, mu/nu and the frame core U do not.
    # Both terms self-infer (U pins C, mu/nu pin W).
    term1 = contractions.contract('WKCa,Caib,WCb->WKCi', tt, U, nu)
    term2 = contractions.contract('WCa,Caib,WKCb->WKCi', mu, U, st)
    return (term1 + term2,)


def compute_dxi_tilde(
        sigma_tildes:           typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rR(i+1))
        tau_tildes:             typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rLi)
        up_tt_cores:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(rLi,nUi,rR(i+1))
        mus:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rLi)
        nus:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rR(i+1))
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # dxi_tildes. len=d, elm_shape=(...,nUi)
    '''Adjoint-var-downward edge variables dxi_tilde.
    Used for computing the transpose of the map from a tangent vector to its probes.

    See Section 6.2.3, particularly Algorithm 8, in:
        Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
        "Tucker Tensor Train Taylor Series."
        arXiv preprint arXiv:2603.21141.
        `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_
    '''
    use_jax = tree_contains_jax((sigma_tildes, tau_tildes, up_tt_cores, mus, nus))
    is_uniform = not isinstance(mus, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed WKC: tau_tildes/sigma_tildes carry K (W+K+C); mus/nus are W+C; the frame
        # supercore up_tt_cores (U) is C-only. Mirrors the ragged calls; the ragged xmap is the oracle.
        term1 = contractions.contract('dWKCa,dCaib,dWCb->dWKCi', tau_tildes, up_tt_cores, nus)
        term2 = contractions.contract('dWCa,dCaib,dWKCb->dWKCi', mus, up_tt_cores, sigma_tildes)
        dxi_tildes = term1 + term2
    else:
        xs = (up_tt_cores, mus, nus, sigma_tildes, tau_tildes)
        (dxi_tildes,) = xmap(_dxi_tilde_step, xs)

    return dxi_tildes


def _assemble_tucker_variations_core(
        x:               typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (z_tilde, eta, w, dxi_tilde)
        sum_over_probes: bool,     # caller intent -- no operand carries it; hence the two bodies below
) -> typ.Tuple[NDArray]:           # (V_tilde,): W+K+C+(nUi,Ni), or K+C+(nUi,Ni) when summed
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
        V_tilde = (
                contractions.contract('WKCo,WCa->KCao', z_tilde, eta, len_W=n_probe)
                +
                contractions.contract('Wo,WKCa->KCao', w, dxi_tilde)
        )
    else:
        V_tilde = (
                contractions.contract('WKCo,WCa->WKCao', z_tilde, eta, len_W=n_probe)
                +
                contractions.contract('Wo,WKCa->WKCao', w, dxi_tilde)
        )
    return (V_tilde,)


def _assemble_tucker_variations_step_summed(x):    # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=True`` variant of :py:func:`_assemble_tucker_variations_core`.'''
    return _assemble_tucker_variations_core(x, True)


def _assemble_tucker_variations_step_unsummed(x):  # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=False`` variant of :py:func:`_assemble_tucker_variations_core`.'''
    return _assemble_tucker_variations_core(x, False)


def assemble_tucker_variations(
        ztildes:    typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,Ni)
        dxi_tildes: typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi)
        ww:         typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,Ni)
        etas:       typ.Union[typ.Sequence[NDArray], NDArray], # len=d, elm_shape=(...,nUi)
        sum_over_probes: bool = False,
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # V_tildes. len=d, elm_shape=(...,nUi,Ni)
    '''Assemble Tucker core variations, V_tilde.
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
        # d-prefixed: ztildes/dxi_tildes carry K; etas is W+C; ww is the W-only packed probe
        # supercore (d,)+W+(N,) -> n_probe = len(W). The ragged xmap below is the oracle.
        n_probe = ww.ndim - 2
        if sum_over_probes:
            V_tildes = (contractions.contract('dWKCo,dWCa->dKCao', ztildes, etas, len_W=n_probe)
                         + contractions.contract('dWo,dWKCa->dKCao', ww, dxi_tildes))
        else:
            V_tildes = (contractions.contract('dWKCo,dWCa->dWKCao', ztildes, etas, len_W=n_probe)
                         + contractions.contract('dWo,dWKCa->dWKCao', ww, dxi_tildes))
    else:
        step = (_assemble_tucker_variations_step_summed if sum_over_probes
                else _assemble_tucker_variations_step_unsummed)
        (V_tildes,) = xmap(step, (ztildes, etas, ww, dxi_tildes))

    return V_tildes


def _assemble_tt_variations_core(
        x:               typ.Tuple[NDArray, NDArray, NDArray,
                                   NDArray, NDArray, NDArray],  # (xi, mu, nu, sigma_tilde, tau_tilde, deta_tilde)
        n_probe:         int,      # len(W); rides as an extra map operand -- see assemble_tt_variations
        sum_over_probes: bool,     # caller intent -- no operand carries it; hence the two bodies below
) -> typ.Tuple[NDArray]:           # (H_tilde,): W+K+C+(rLi,nDi,rR(i+1)), or K+C+(...) when summed
    '''One core of the ragged map of :py:func:`assemble_tt_variations`, shared by the two
    closure-free map bodies below -- ``docs/contributor/scan_body_principles.md``.'''
    xi, mu, nu, sigma_tilde, tau_tilde, deta_tilde = x
    # Three-group (W probe, K tangent, C frame): the residual-derived edge vars sigma_tilde /
    # tau_tilde / deta_tilde carry K (on the j / i / a leg respectively), the frame edge vars
    # xi/mu/nu do not. No operand here is W-only or C-only, so len(W)=n_probe is supplied;
    # each contraction then derives C from an W+C operand and K from the W+K+C one. Output
    # keeps K always; W is summed (K+C) or kept (W+K+C) per sum_over_probes.
    if sum_over_probes:
        H_tilde = (
                contractions.contract('WCi,WCa,WKCj->KCiaj', mu, xi, sigma_tilde, len_W=n_probe)
                +
                contractions.contract('WKCi,WCa,WCj->KCiaj', tau_tilde, xi, nu, len_W=n_probe)
                +
                contractions.contract('WCi,WKCa,WCj->KCiaj', mu, deta_tilde, nu, len_W=n_probe)
        )
    else:
        H_tilde = (
                contractions.contract('WCi,WCa,WKCj->WKCiaj', mu, xi, sigma_tilde, len_W=n_probe)
                +
                contractions.contract('WKCi,WCa,WCj->WKCiaj', tau_tilde, xi, nu, len_W=n_probe)
                +
                contractions.contract('WCi,WKCa,WCj->WKCiaj', mu, deta_tilde, nu, len_W=n_probe)
        )
    return (H_tilde,)


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
        deta_tildes:    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,nDi)
        xis:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,nDi)
        mus:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rLi)
        nus:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=(...,rR(i+1))
        sum_over_probes: bool = False,
        n_probe: int = 0,  # number of trailing probe-stack axes; only used when sum_over_probes
) -> typ.Union[typ.Sequence[NDArray], NDArray]: # H_tildes. len=d, elm_shape=(...,rLi,nDi,rR(i+1))
    '''Assemble TT core variations, H_tilde.
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
        # d-prefixed WKC triple outer products: K rides on the residual-derived edge var of each
        # term (sigma_tilde on j, tau_tilde on i, deta_tilde on a); the frame edge vars xi/mu/nu are W+C.
        # n_probe = len(W) (supplied). The ragged xmap below is the oracle.
        if sum_over_probes:
            H_tildes = (contractions.contract('dWCi,dWCa,dWKCj->dKCiaj', mus, xis, sigma_tildes, len_W=n_probe)
                         + contractions.contract('dWKCi,dWCa,dWCj->dKCiaj', tau_tildes, xis, nus, len_W=n_probe)
                         + contractions.contract('dWCi,dWKCa,dWCj->dKCiaj', mus, deta_tildes, nus, len_W=n_probe))
        else:
            H_tildes = (contractions.contract('dWCi,dWCa,dWKCj->dWKCiaj', mus, xis, sigma_tildes, len_W=n_probe)
                         + contractions.contract('dWKCi,dWCa,dWCj->dWKCiaj', tau_tildes, xis, nus, len_W=n_probe)
                         + contractions.contract('dWCi,dWKCa,dWCj->dWKCiaj', mus, deta_tildes, nus, len_W=n_probe))
    else:
        # n_probe is per-call runtime data with no W-only operand to recover it from, so it rides as
        # an extra per-core map operand (principle 6) rather than being closed over.
        step = (_assemble_tt_variations_step_summed if sum_over_probes
                else _assemble_tt_variations_step_unsummed)
        xs = (xis, mus, nus, sigma_tildes, tau_tildes, deta_tildes, (n_probe,) * len(xis))
        (H_tildes,) = xmap(step, xs)

    return H_tildes


def tv_probe_transpose_from_sweep(
        ztildes:    typ.Sequence[NDArray],  # probe residuals, len=d, elm_shape=W+K+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors,   len=d, elm_shape=W+(Ni,)
        frame:      typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (D, U, L, R)
        frame_sweep: typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = tv_precompute_probe_frame_sweep(frame, ww)
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (V_tildes, H_tildes) = T3Variations.data
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
    down_tucker_cores, up_tt_cores, left_tt_cores, right_tt_cores = frame
    xis, mus, nus, etas = frame_sweep
    deta_tildes  = compute_deta_tilde(down_tucker_cores, ztildes)
    tau_tildes   = compute_tau_tilde(deta_tildes, left_tt_cores, xis, mus)
    sigma_tildes = compute_sigma_tilde(deta_tildes, right_tt_cores, xis, nus)
    dxi_tildes   = compute_dxi_tilde(sigma_tildes, tau_tildes, up_tt_cores, mus, nus)
    n_probe = ww[0].ndim - 1
    V_tildes = assemble_tucker_variations(ztildes, dxi_tildes, ww, etas, sum_over_probes=sum_over_probes)
    H_tildes = assemble_tt_variations(sigma_tildes, tau_tildes, deta_tildes, xis, mus, nus,
                                       sum_over_probes=sum_over_probes, n_probe=n_probe)
    return V_tildes, H_tildes


def tv_probe_transpose(
        frame:             typ.Tuple,                    # the base point: (D, U, L, R)
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        c:                 typ.Sequence[NDArray],        # P*⊗V_i, len=d, elm_shape=F+W+[K]+C+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # V blocks: X*, len=d, elm_shape=[W]+[K]+C+(nUi,Ni)
        typ.Tuple[NDArray, ...],                         # H blocks: X*, len=d, elm_shape=[W]+[K]+C+(rLi,nDi,rR(i+1))
]:
    '''The transpose of :py:func:`tv_probe`: ``J^T`` applied to ``c``, a tangent's blocks at ``frame``.

    ``c`` is an element of ``Y`` (``P*⊗V_i`` per mode per sample; a stack ``K`` of them allowed), paired with ``z`` slot
    by slot. It carries no metric, and none is applied here: for the gradient of a weighted least-squares fit pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` wanting ``JtJp`` with
    ``c`` held as ``Jp`` (the route of a held ``Jp``: ``J^T`` of the vector held; the engine turns ``c`` into its
    dual-jet form inside). The blocks are summed over the samples by default. For repeated calls at one frame (inside an
    iterative solver), call :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the
    frame's sweeps are then computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> Jp = probing.tv_probe(frame, variations, ww)
    >>> V, H = probing.tv_probe_transpose(frame, ww, Jp)
    >>> pair = lambda a, b: sum(float(np.sum(u * v)) for u, v in zip(a[0] + a[1], tuple(b[0]) + tuple(b[1])))
    >>> print(bool(np.isclose(pair((V, H), variations), sum(float(np.sum(a * a)) for a in Jp))))       # <p, J^T J p> = |J p|^2
    True
    '''
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('probe', frame, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


def t3_probe_ambient_transpose(
        ww:                typ.Sequence[NDArray],  # probe vectors: V_i, len=d, elm_shape=W+(Ni,)
        c:                 typ.Sequence[NDArray],  # V_i, len=d, elm_shape=W+C+(Ni,)
        sum_over_samples:  bool = False,           # True: W folds into the CP rank
) -> typ.Sequence[NDArray]:  # canonical (CP) factors: E*, len=d, ith elm_shape=stack_shape+(R, Ni)
    '''Ambient transpose of :py:func:`t3_probe`: back-project ``c`` into CP factors.

    The *ambient* adjoint -- the transpose of ``probe`` as a linear map on the **full tensor space**. Probe returns
    ``d`` vectors (one free mode each), so ``c`` (an element of ``Y``: ``V_i`` per mode and sample, paired with
    ``probe``'s output, with no metric) is ``d`` vectors; the back-projection is the rank-``d`` tensor

        sum_i  w0 (x) ... (x) w_{i-1} (x) c_i (x) w_{i+1} (x) ... (x) w_{d-1}

    (term ``i`` has ``c_i`` in slot ``i`` and the probe vectors elsewhere), whose natural representation is a
    **canonical (CP) decomposition** of rank ``d``. Frame-free. Distinct from the *corewise* transpose (gradient w.r.t.
    a frame's cores) and the *tangent* transpose (Riemannian gradient); see ``docs/transposes.md``. The
    ``apply``/``entries`` analog is the rank-1 (or rank-``|W|``) :py:func:`t3_apply_ambient_transpose`.

    - ``sum_over_samples=False`` (primary): ``W`` is a passthrough stacking axis -- a ``W (+ C)`` stack
      of rank-``d`` CP tensors.
    - ``sum_over_samples=True``: ``W`` folds into the CP rank -- one rank-``d|W|`` CP tensor
      ``sum_W sum_i (...)``. Cheap as CP (``O(d |W| N)``).

    Returns CP ``factors`` (factor ``k`` has the diagonal structure: rank slot ``k`` = ``c_k``,
    the others = ``ww_k``), in the layout :py:func:`t3_conversions.t3_from_canonical` consumes.

    Examples
    --------
    The adjoint of ``probe`` on the full tensor space, ``<J^T c, T> = <c, dense_probe(T, ww)>``. **The argument order**
    (changed in this release: the sample, then ``c``): ``c`` and the sample are both ``d`` vectors, so the old order
    ``(c, ww)`` runs and returns the transpose of a different problem:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> np.random.seed(0)
    >>> T = np.random.randn(4, 5, 6)
    >>> ww, r = ([np.random.randn(3, n) for n in T.shape] for _ in range(2))
    >>> lhs = sum(float(np.sum(a * b)) for a, b in zip(r, probing.dense_probe(T, ww)))
    >>> pair = lambda f: float(np.einsum('rA,rB,rC,ABC->', *f, T))           # <CP(f), T>, the rank axis summed
    >>> right = probing.t3_probe_ambient_transpose(ww, r, sum_over_samples=True)
    >>> wrong = probing.t3_probe_ambient_transpose(r, ww, sum_over_samples=True)   # swapped: runs, and is wrong
    >>> print(abs(lhs - pair(right)) < 1e-12 * abs(lhs), abs(lhs - pair(wrong)) < 1e-12 * abs(lhs))
    True False
    '''
    use_jax = tree_contains_jax((c, ww))
    xnp, _, _ = get_backend(False, use_jax)
    d = len(ww)
    nW = ww[0].ndim - 1
    W  = ww[0].shape[:nW]                 # probe stack
    C  = c[0].shape[nW:-1]          # frame stack (c[i] is W + C + (Ni,))
    nC = len(C)
    mW = math.prod(W)

    factors = []
    for k in range(d):
        Nk = ww[k].shape[-1]
        w_bc = xnp.broadcast_to(ww[k].reshape(W + (1,) * nC + (1, Nk)), W + C + (d, Nk))
        diag = (xnp.arange(d) == k).reshape((d, 1))            # rank slot k is the "diagonal"
        Fk = xnp.where(diag, c[k][..., None, :], w_bc)   # W + C + (d, Nk)
        if sum_over_samples:                                    # fold W into the CP rank: rank d -> d|W|
            Fk = xnp.moveaxis(Fk, tuple(range(nW)), tuple(range(nC, nC + nW)))  # C + W + (d, Nk)
            Fk = Fk.reshape(C + (mW * d, Nk))
        factors.append(Fk)
    return tuple(factors)


def t3_probe_corewise_tangent(
        x:                 typ.Tuple,                    # the point: X, (B, G), whose cores the directions vary
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        core_directions:   typ.Tuple,                    # X, (B_dot, G_dot), the shapes of B, G with K+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
) -> typ.Tuple[NDArray, ...]:               # J x_dot: P⊗V_i*, len=d, elm_shape=F+W+K+C+(Ni,)
    '''The corewise forward derivative: ``J x_dot``, the derivative of ``z(x)`` (:py:func:`t3_probe`) along
    ``core_directions = (B_dot, G_dot)``, of the type of ``z`` with the stack of directions ``K`` after ``W``.

    The trio at a point: this is the tangent, :py:func:`t3_probe_corewise_transpose` its transpose and
    :py:func:`t3_probe_corewise_hessian` the Hessian. Per sample, with no sum over the samples (no
    ``sum_over_samples``). One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the
    corewise point ``x`` wanting ``Jp`` with ``core_directions`` (its raw route), each sweep once; the substitution
    ``tv_probe(t3_corewise_frame(x), core_directions, ...)`` computes the same at the cost of the frame's sweeps on
    ``(B, G, G, G)``. For repeated calls at one point, call
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the point's sweeps are then
    computed once.

    Examples
    --------
    ``z`` is homogeneous of degree ``2d`` in the cores, so the derivative along ``x`` itself is ``2d z`` (Euler):

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> z = probing.t3_probe(x, ww)
    >>> Jx = probing.t3_probe_corewise_tangent(x, ww, x)
    >>> print([a.shape for a in Jx])                                         # W + (Ni,)
    [(3, 4), (3, 5), (3, 6)]
    >>> print(all(np.allclose(a, 6 * b) for a, b in zip(Jx, z)))        # Euler: J x = 2d z
    True
    '''
    (Jp,) = sampling_variables.sampling_variables('probe', x, ww, ('Jp',), order=order,
                                                   core_directions=core_directions)
    return Jp


def t3_probe_corewise_transpose(
        x:                 typ.Tuple,                    # the point: X, (tucker_cores, tt_cores)
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        c:                 typ.Sequence[NDArray],        # P*⊗V_i, len=d, elm_shape=F+W+[K]+C+(Ni,)
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker-core blocks: X*, len=d, elm_shape=[W]+[K]+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT-core blocks:     X*, len=d,
                                                         #   elm_shape=[W]+[K]+C+(ri,nDi,r(i+1))
]:
    '''The corewise transpose: ``J^T`` applied to ``c`` in the cores of the point ``x``, the gradient of
    ``phi(x) = <c, z(x)>`` (the pairing of ``Y`` with ``Y*``, slot by slot: no metric).

    ``c`` is an element of ``Y`` (``P*⊗V_i`` per mode per sample; a stack ``K`` of them allowed), paired with ``z`` slot
    by slot. It carries no metric, and none is applied here: for the gradient of a weighted least-squares fit pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point
    ``x`` wanting ``JtJp`` with ``c`` held as ``Jp``. Summed over the samples by default. For repeated calls at one
    frame (inside an iterative solver), call :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with
    ``held=``: the frame's sweeps are then computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> c = probing.t3_probe(x, ww)
    >>> B, G = probing.t3_probe_corewise_transpose(x, ww, c)
    >>> print([b.shape for b in B], [g.shape for g in G])
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]

    **The argument order** (changed in this release: the point, the sample, then ``c``). ``c`` and the sample are both
    ``d`` vectors, so ``(x, c, ww)`` runs and returns the transpose of a different problem. Checked against the forward
    derivative at the intended sample:

    >>> x_dot = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> r = [np.random.randn(3, n) for n in (4, 5, 6)]
    >>> lhs = sum(float(np.sum(a * b)) for a, b in zip(r, probing.t3_probe_corewise_tangent(x, ww, x_dot)))
    >>> pair = lambda g: sum(float(np.sum(a * b)) for p, q in zip(g, x_dot) for a, b in zip(p, q))
    >>> right = probing.t3_probe_corewise_transpose(x, ww, r)
    >>> wrong = probing.t3_probe_corewise_transpose(x, r, ww)       # c and the sample swapped
    >>> print(abs(lhs - pair(right)) < 1e-12 * abs(lhs), abs(lhs - pair(wrong)) < 1e-12 * abs(lhs))
    True False
    '''
    held = sampling_variables.HeldVariables(Jp=c)
    (blocks,) = sampling_variables.sampling_variables('probe', x, ww, ('JtJp',), order=order, held=held,
                                                       sum_over_samples=sum_over_samples)
    return blocks


###############################################################
######    Corewise two-hole: the probe Hessian           ######
###############################################################
#
# The derivative of t3_probe_corewise_transpose along a direction in the cores. For a fixed residual z~
# (omega^2 applied) the corewise transpose is the gradient of theta -> <z~, probe(Phi(theta))>, so its
# derivative along theta_hat is that function's core-coordinate Hessian applied to theta_hat -- the
# "two-hole" sweep, the sampling half of the exact Riemannian Hessian's curvature term
# K_z theta_hat = d/ds grad_theta <z~, probe(Phi(theta + s theta_hat))>. It is the order-0 case of the
# probe-derivative two-hole (sampling_derivatives.t3_probe_corewise_derivatives_hessian), written in plain
# probing's own kernels: there is no order axis, so no recurrence and no order scan -- every dotted kernel
# is the product rule of the GN kernel it differentiates, in that kernel's form.
#
# Every quantity of the GN transpose's Lagrangian is multilinear in (cores, vectors), so each edge variable
# gains a DOTTED companion carrying the tangent stack K (the directions' stack); the undotted state carries
# W + C only (the residual has no K). At the corewise frame (B, G, G, G) of theta = (B, G):
#
#   1. undotted state: xi, mu (reused from the model's sweep -- identical at the manifold and corewise frames
#      of the base point), nu re-swept through G, and the adjoint tildes deta~, tau~, sigma~;
#   2. dotted forward = the GN tangent forward at U = L = R = G along (Bdot, Gdot): xidot = Bdot w,
#      mudot / nudot = compute_sigma / compute_tau, etadot = compute_deta;
#   3. dotted adjoint-upward lift deta~dot = Bdot z~;
#   4. dotted adjoint sweeps tau~dot / sigma~dot: two _sigma_step calls per step (the product rule of the two
#      trilinear terms of the tau~ step);
#   5. dotted dxi~: two compute_deta calls (the product rule of its two trilinear combines);
#   6. assemblies: Tucker z~ (x) etadot + w (x) dxi~dot; TT factored by the product rule.
#
# No corewise eta and no corewise dxi~ is ever formed: the Hessian needs only their derivatives. Stacks: z~
# and the undotted edge variables W + C, the directions and every dotted quantity K + C (W + K + C on the edge
# variables), the output K + C (W summed). Polymorphic: ragged tuples or uniform supercores (a uniform caller
# masks theta and theta_hat once, packs, delegates here, and re-masks the output -- the kernel is mask-free).

def _deta_tilde_tangent_step(
        x: typ.Tuple[NDArray, NDArray, NDArray],   # (D, Ddot, z~): C+(nDi,Ni) ; K+C+(nDi,Ni) ; W+C+(Ni,)
) -> typ.Tuple[NDArray]:                           # (deta_tilde_dot,): W+K+C+(nDi,)
    '''One core of the ragged map of :py:func:`compute_deta_tilde_tangent`. The C-only ``D`` is passed only to
    recover ``len(C)``. Closure-free map body -- ``docs/contributor/scan_body_principles.md``.'''
    D, Dd, zt = x
    return (contractions.contract('WCo,KCio->WKCi', zt, Dd, len_C=D.ndim - 2),)


def compute_deta_tilde_tangent(
        down_tucker_cores:      typ.Union[typ.Sequence[NDArray], NDArray],  # D.    len=d, elm_shape=C+(nDi,Ni)
        down_tucker_cores_dot:  typ.Union[typ.Sequence[NDArray], NDArray],  # Ddot. len=d, elm_shape=K+C+(nDi,Ni)
        ztildes:                typ.Union[typ.Sequence[NDArray], NDArray],  # residuals, len=d, elm_shape=W+C+(Ni,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]:  # deta_tildes_dot. len=d, elm_shape=W+K+C+(nDi,)
    '''The derivative of :py:func:`compute_deta_tilde` along the Tucker directions, the residual held fixed:
    ``deta_tilde_dot_i = Ddot_i z~_i`` (only the core moves). The tangent stack ``K`` comes from ``Ddot``; ``D``
    only pins ``len(C)``.'''
    use_jax = tree_contains_jax((down_tucker_cores, down_tucker_cores_dot, ztildes))
    is_uniform = is_ndarray(down_tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)
    if is_uniform:
        return contractions.contract('dWCo,dKCio->dWKCi', ztildes, down_tucker_cores_dot,
                                     len_C=down_tucker_cores.ndim - 3)
    (deta_tildes_dot,) = xmap(_deta_tilde_tangent_step, (down_tucker_cores, down_tucker_cores_dot, ztildes))
    return deta_tildes_dot


def _tau_tilde_tangent_step(
        carry_dot:  NDArray,                    # the dotted tilde entering this core, W+K+C+(rLi,)
        x:          typ.Tuple[NDArray, ...],    # (L, L_dot, xi, xi_dot, deta~, deta~_dot, mu, mu_dot, tau~), one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:    # (next dotted tilde, (carry_dot,))
    '''One edge of the dotted adjoint sweep of :py:func:`compute_tau_tilde_tangent`: the product rule of
    :py:func:`_tau_tilde_step`, ``tau~' = tau~ L xi + mu L deta~``. Each term is trilinear, so its derivative is
    exactly one :py:func:`_sigma_step` (``sigma R xi + mu H xi + mu U dxi``, three terms with ``K`` on one factor
    each): ``d(tau~ L xi) = _sigma_step(tau~_dot, L, L, L_dot, xi, xi_dot, tau~)`` and ``d(mu L deta~) =
    _sigma_step(mu_dot, L, L, L_dot, deta~, deta~_dot, mu)``. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    L, Ld, xi, xid, dt, dtd, mu, mud, tt = x
    nxt = (_sigma_step(carry_dot, L, L, Ld, xi, xid, tt)       # K on the carry / the core / xi
           + _sigma_step(mud, L, L, Ld, dt, dtd, mu))          # K on mu / the core / deta~
    return nxt, (carry_dot,)


def compute_tau_tilde_tangent(
        left_tt_cores:          typ.Union[typ.Sequence[NDArray], NDArray],  # L.     len=d, elm_shape=C+(rLi,nDi,rL(i+1))
        left_tt_cores_dot:      typ.Union[typ.Sequence[NDArray], NDArray],  # L_dot. len=d, elm_shape=K+C+(rLi,nDi,rL(i+1))
        xis:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(nDi,)
        xis_dot:                typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(nDi,)
        deta_tildes:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(nDi,)
        deta_tildes_dot:        typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(nDi,)
        mus:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rLi,)
        mus_dot:                typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rLi,)
        tau_tildes:             typ.Union[typ.Sequence[NDArray], NDArray],  # = compute_tau_tilde(deta~, L, xi, mu). len=d, elm_shape=W+C+(rLi,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]:  # tau_tildes_dot. len=d, elm_shape=W+K+C+(rLi,)
    '''The derivative of :py:func:`compute_tau_tilde` along ``(L_dot, xi_dot, deta~_dot, mu_dot)``: the same
    left-to-right sweep, its carry the dotted tilde, two :py:func:`_sigma_step` calls per core
    (:py:func:`_tau_tilde_tangent_step`). The undotted operands carry ``W + C``, the dotted ``W + K + C``; the
    undotted ``tau_tildes`` (the carry entering each core) ride as scan data.'''
    use_jax = tree_contains_jax((left_tt_cores, left_tt_cores_dot, xis, xis_dot, deta_tildes, deta_tildes_dot,
                                 mus, mus_dot, tau_tildes))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    # the dotted carry is W+K+C; deta_tildes_dot carries K, so it supplies the init's leading stack
    dtype = xnp.result_type(*(a[0] for a in (left_tt_cores, left_tt_cores_dot, xis, xis_dot, deta_tildes,
                                             deta_tildes_dot, mus, mus_dot, tau_tildes)))
    init = xnp.zeros(deta_tildes_dot[0].shape[:-1] + (left_tt_cores[0].shape[-3],), dtype)
    _, (tau_tildes_dot,) = xscan(_tau_tilde_tangent_step, init, (left_tt_cores, left_tt_cores_dot, xis, xis_dot,
                                                                 deta_tildes, deta_tildes_dot, mus, mus_dot,
                                                                 tau_tildes))
    return tau_tildes_dot


def compute_sigma_tilde_tangent(
        right_tt_cores:         typ.Union[typ.Sequence[NDArray], NDArray],  # R.     len=d, elm_shape=C+(rRi,nDi,rR(i+1))
        right_tt_cores_dot:     typ.Union[typ.Sequence[NDArray], NDArray],  # R_dot. len=d, elm_shape=K+C+(rRi,nDi,rR(i+1))
        xis:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(nDi,)
        xis_dot:                typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(nDi,)
        deta_tildes:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(nDi,)
        deta_tildes_dot:        typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(nDi,)
        nus:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rR(i+1),)
        nus_dot:                typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rR(i+1),)
        sigma_tildes:           typ.Union[typ.Sequence[NDArray], NDArray],  # = compute_sigma_tilde(deta~, R, xi, nu). len=d, elm_shape=W+C+(rR(i+1),)
) -> typ.Union[typ.Sequence[NDArray], NDArray]:  # sigma_tildes_dot. len=d, elm_shape=W+K+C+(rR(i+1),)
    '''The derivative of :py:func:`compute_sigma_tilde` -- the reverse of :py:func:`compute_tau_tilde_tangent`
    (``tt_reverse`` the cores and their directions, reverse the sequences, run the sweep, reverse the result),
    exactly as :py:func:`compute_sigma_tilde` reverses :py:func:`compute_tau_tilde`.'''
    reverse = tt_operations.tt_reverse
    return compute_tau_tilde_tangent(
        reverse(right_tt_cores), reverse(right_tt_cores_dot), xis[::-1], xis_dot[::-1], deta_tildes[::-1],
        deta_tildes_dot[::-1], nus[::-1], nus_dot[::-1], sigma_tildes[::-1],
    )[::-1]


def compute_dxi_tilde_tangent(
        up_tt_cores:            typ.Union[typ.Sequence[NDArray], NDArray],  # U.     len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        up_tt_cores_dot:        typ.Union[typ.Sequence[NDArray], NDArray],  # U_dot. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        mus:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rLi,)
        mus_dot:                typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rLi,)
        nus:                    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rR(i+1),)
        nus_dot:                typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rR(i+1),)
        sigma_tildes:           typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rR(i+1),)
        sigma_tildes_dot:       typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rR(i+1),)
        tau_tildes:             typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rLi,)
        tau_tildes_dot:         typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rLi,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]:  # dxi_tildes_dot. len=d, elm_shape=W+K+C+(nUi,)
    '''The derivative of :py:func:`compute_dxi_tilde`, ``dxi~ = tau~ U nu + mu U sigma~``: each combine is
    trilinear and :py:func:`compute_deta` is exactly the product rule of one (``sigma R nu + mu H nu + mu L
    tau``), so ``dxi~_dot = compute_deta(U_dot, U, U, tau~, nu, tau~_dot, nu_dot) + compute_deta(U_dot, U, U, mu,
    sigma~, mu_dot, sigma~_dot)`` -- six terms, ``K`` on the dotted factor of each.'''
    from_tau = compute_deta(up_tt_cores_dot, up_tt_cores, up_tt_cores, tau_tildes, nus, tau_tildes_dot,
                            nus_dot)
    from_mu = compute_deta(up_tt_cores_dot, up_tt_cores, up_tt_cores, mus, sigma_tildes, mus_dot,
                           sigma_tildes_dot)
    if is_ndarray(from_tau):
        return from_tau + from_mu
    return tuple(a + b for a, b in zip(from_tau, from_mu))


def _tucker_tangent_variations_step(
        x: typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # (z~, dxi~_dot, eta_dot, w), one core
) -> typ.Tuple[NDArray]:                                    # (dB,): K+C+(ni,Ni), W summed
    '''One core of the ragged map of :py:func:`assemble_tucker_tangent_variations`; ``len(W)`` is read off the
    W-only probe vector ``w``. Closure-free map body -- ``docs/contributor/scan_body_principles.md``.'''
    zt, dxtd, etad, w = x
    n_probe = w.ndim - 1
    return (contractions.contract('WCo,WKCa->KCao', zt, etad, len_W=n_probe)
            + contractions.contract('Wo,WKCa->KCao', w, dxtd),)


def assemble_tucker_tangent_variations(
        ztildes:        typ.Union[typ.Sequence[NDArray], NDArray],  # residuals, len=d, elm_shape=W+C+(Ni,)
        dxi_tildes_dot: typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(ni,)
        ww:             typ.Union[typ.Sequence[NDArray], NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,)
        etas_dot:       typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(ni,)
) -> typ.Union[typ.Sequence[NDArray], NDArray]:  # dB. len=d, elm_shape=K+C+(ni,Ni) (W summed)
    '''The derivative of :py:func:`assemble_tucker_variations` (``sum_over_probes=True``) with the residual and
    the probe vectors fixed: ``sum_W z~ (x) eta_dot + w (x) dxi~_dot``.'''
    use_jax = tree_contains_jax((ztildes, dxi_tildes_dot, ww, etas_dot))
    is_uniform = not isinstance(ww, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)
    if is_uniform:
        n_probe = ww.ndim - 2                      # ww is the W-only packed probe supercore (d,)+W+(N,)
        return (contractions.contract('dWCo,dWKCa->dKCao', ztildes, etas_dot, len_W=n_probe)
                + contractions.contract('dWo,dWKCa->dKCao', ww, dxi_tildes_dot))
    (dB,) = xmap(_tucker_tangent_variations_step, (ztildes, dxi_tildes_dot, etas_dot, ww))
    return dB


def _tt_tangent_variations_step(
        x: typ.Tuple[NDArray, ...],   # (xi, xi_dot, mu, mu_dot, nu, nu_dot, sigma~, sigma~_dot, tau~, tau~_dot,
        #                               deta~, deta~_dot, n_probe), one core
) -> typ.Tuple[NDArray]:              # (dG,): K+C+(rLi,ni,rR(i+1)), W summed
    '''One core of the factored TT tangent assembly (:py:func:`assemble_tt_tangent_variations`). ``n_probe``
    rides as the last map operand (per-call data with no W-only operand to recover it from). Closure-free map
    body -- ``docs/contributor/scan_body_principles.md``.'''
    xi, xid, mu, mud, nu, nud, st, std, tt, ttd, dt, dtd, n_probe = x
    return (_tt_tangent_variations_core(xi, xid, mu, mud, nu, nud, st, std, tt, ttd, dt, dtd, n_probe,
                                        ('WCi,WCb->WCib', 'WKCi,WCb->WKCib', 'WCi,WKCb->WKCib',
                                         'WKCa,WCib->KCaib', 'WCa,WKCib->KCaib')),)


def _tt_tangent_variations_core(xi, xid, mu, mud, nu, nud, st, std, tt, ttd, dt, dtd, n_probe, strings):
    '''The factored TT tangent assembly on one core (ragged) or on supercores (uniform; the ``d``-prefixed
    ``strings``). The GN assembly ``mu (x) xi (x) sigma~ + tau~ (x) xi (x) nu + mu (x) deta~ (x) nu`` groups as
    ``mu . E1 + tau~ . E2`` with the pair intermediates ``E1 = xi (x) sigma~ + deta~ (x) nu`` and ``E2 = xi (x) nu``
    (over the mode and right-bond legs, W + C), so its derivative is ``mu_dot . E1 + mu . E1_dot + tau~_dot . E2 +
    tau~ . E2_dot``: four W-summed finals (the nine-term product rule has nine).'''
    pair, pair_kl, pair_kr, final_kl, final_kr = strings
    kw = dict(len_W=n_probe)
    E1 = contractions.contract(pair, xi, st) + contractions.contract(pair, dt, nu)
    E2 = contractions.contract(pair, xi, nu)
    E1d = (contractions.contract(pair_kl, xid, st, **kw) + contractions.contract(pair_kr, xi, std, **kw)
           + contractions.contract(pair_kl, dtd, nu, **kw) + contractions.contract(pair_kr, dt, nud, **kw))
    E2d = contractions.contract(pair_kl, xid, nu, **kw) + contractions.contract(pair_kr, xi, nud, **kw)
    return (contractions.contract(final_kl, mud, E1, **kw) + contractions.contract(final_kr, mu, E1d, **kw)
            + contractions.contract(final_kl, ttd, E2, **kw) + contractions.contract(final_kr, tt, E2d, **kw))


def assemble_tt_tangent_variations(
        sigma_tildes:       typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rR(i+1),)
        sigma_tildes_dot:   typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rR(i+1),)
        tau_tildes:         typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rLi,)
        tau_tildes_dot:     typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rLi,)
        deta_tildes:        typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(ni,)
        deta_tildes_dot:    typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(ni,)
        xis:                typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(ni,)
        xis_dot:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(ni,)
        mus:                typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rLi,)
        mus_dot:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rLi,)
        nus:                typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+C+(rR(i+1),)
        nus_dot:            typ.Union[typ.Sequence[NDArray], NDArray],  # len=d, elm_shape=W+K+C+(rR(i+1),)
        n_probe:            int,                                        # len(W): no operand here is W-only
) -> typ.Union[typ.Sequence[NDArray], NDArray]:  # dG. len=d, elm_shape=K+C+(rLi,ni,rR(i+1)) (W summed)
    '''The derivative of :py:func:`assemble_tt_variations` (``sum_over_probes=True``), factored by the product
    rule: the three trilinear outer products group as ``mu . (xi (x) sigma~ + deta~ (x) nu) + tau~ . (xi (x) nu)``,
    each pair intermediate and its dotted twin formed once and met by the remaining factor -- four W-summed finals
    instead of the nine-term product rule's nine, equal to it to rounding (the test suite pins it against the
    nine-term form).'''
    use_jax = tree_contains_jax((sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot, deta_tildes,
                                 deta_tildes_dot, xis, xis_dot, mus, mus_dot, nus, nus_dot))
    is_uniform = not isinstance(xis, typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)
    if is_uniform:
        return _tt_tangent_variations_core(
            xis, xis_dot, mus, mus_dot, nus, nus_dot, sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot,
            deta_tildes, deta_tildes_dot, n_probe,
            ('dWCi,dWCb->dWCib', 'dWKCi,dWCb->dWKCib', 'dWCi,dWKCb->dWKCib', 'dWKCa,dWCib->dKCaib',
             'dWCa,dWKCib->dKCaib'))
    xs = (xis, xis_dot, mus, mus_dot, nus, nus_dot, sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot,
          deta_tildes, deta_tildes_dot, (n_probe,) * len(xis))
    (dG,) = xmap(_tt_tangent_variations_step, xs)
    return dG


def t3_probe_corewise_hessian_from_sweep(
        ztildes:            typ.Union[typ.Sequence[NDArray], NDArray],  # residuals (omega^2 applied), len=d, elm_shape=W+C+(Ni,)
        ww:                 typ.Union[typ.Sequence[NDArray], NDArray],  # probe vectors, len=d, elm_shape=W+(Ni,)
        core_pair:          typ.Tuple[
            typ.Sequence[NDArray],                  # tucker_cores of theta. len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],                  # tt_cores of theta.     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],                                          # = TuckerTensorTrain.data
        core_directions:    typ.Tuple[
            typ.Sequence[NDArray],                  # Tucker directions theta_hat. len=d, elm_shape=K+C+(ni,Ni)
            typ.Sequence[NDArray],                  # TT directions theta_hat.     len=d, elm_shape=K+C+(ri,ni,r(i+1))
        ],
        frame_sweep:        typ.Tuple[typ.Sequence[NDArray], ...],  # (xi, mu[, nu, eta]); only xi, mu are read
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                        # Tucker block of K theta_hat. len=d, elm_shape=K+C+(ni,Ni)
    typ.Tuple[NDArray, ...],                        # TT block of K theta_hat.     len=d, elm_shape=K+C+(ri,ni,r(i+1))
]:
    ''':py:func:`t3_probe_corewise_hessian` reusing a precomputed ``frame_sweep``'s ``xi`` and ``mu`` -- the reuse
    hook for a local model, whose probe sweep at the manifold frame ``(D, U, L, R)`` carries exactly the corewise
    frame's ``xi`` (through ``D``) and ``mu`` (through ``L``) when ``core_pair = (D, L)``, the geometry's base point.

    ``frame_sweep`` is :py:func:`tv_precompute_probe_frame_sweep` at that frame (its ``nu`` and ``eta`` belong to
    the manifold frame and are not read: ``nu`` is re-swept here through ``G``), or the lean ``(xi, mu)`` pair of
    :py:func:`~t3toolbox.backend.apply.tv_precompute_apply_frame_sweep` at the corewise frame. Either way its
    ``xi`` / ``mu`` must be ``core_pair``'s own; nothing here checks that.'''
    tucker_cores, tt_cores = core_pair
    tucker_dirs, tt_dirs = core_directions
    xis, mus = frame_sweep[0], frame_sweep[1]
    n_probe = ww[0].ndim - 1   # len(W): ragged ww[0] is W+(N0,), packed ww[0] (mode 0's slice) the same

    # 1. the undotted state at the corewise frame (B, G, G, G): nu re-swept through G; the adjoint tildes.
    #    No corewise eta and no corewise dxi~: the Hessian needs only their derivatives.
    nus = compute_nu(tt_cores, xis)
    deta_tildes = compute_deta_tilde(tucker_cores, ztildes)
    tau_tildes = compute_tau_tilde(deta_tildes, tt_cores, xis, mus)
    sigma_tildes = compute_sigma_tilde(deta_tildes, tt_cores, xis, nus)

    # 2. the dotted forward: the GN tangent forward at U = L = R = G along (Bdot, Gdot) -- its sigma / tau /
    #    deta ARE the derivatives of mu / nu / eta along theta_hat.
    xis_dot = compute_dxi(tucker_dirs, ww)
    mus_dot = compute_sigma(tt_dirs, tt_cores, tt_cores, xis, xis_dot, mus)
    nus_dot = compute_tau(tt_dirs, tt_cores, tt_cores, xis, xis_dot, nus)
    etas_dot = compute_deta(tt_dirs, tt_cores, tt_cores, mus, nus, mus_dot, nus_dot)

    # 3. the dotted adjoint-upward lift; 4. the dotted adjoint sweeps; 5. the dotted dxi~
    deta_tildes_dot = compute_deta_tilde_tangent(tucker_cores, tucker_dirs, ztildes)
    tau_tildes_dot = compute_tau_tilde_tangent(tt_cores, tt_dirs, xis, xis_dot, deta_tildes, deta_tildes_dot,
                                               mus, mus_dot, tau_tildes)
    sigma_tildes_dot = compute_sigma_tilde_tangent(tt_cores, tt_dirs, xis, xis_dot, deta_tildes, deta_tildes_dot,
                                                   nus, nus_dot, sigma_tildes)
    dxi_tildes_dot = compute_dxi_tilde_tangent(tt_cores, tt_dirs, mus, mus_dot, nus, nus_dot, sigma_tildes,
                                               sigma_tildes_dot, tau_tildes, tau_tildes_dot)

    # 6. the assemblies (W summed)
    dB = assemble_tucker_tangent_variations(ztildes, dxi_tildes_dot, ww, etas_dot)
    dG = assemble_tt_tangent_variations(sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot, deta_tildes,
                                        deta_tildes_dot, xis, xis_dot, mus, mus_dot, nus, nus_dot, n_probe)
    return dB, dG


def t3_probe_corewise_hessian(
        x:                 typ.Tuple,                    # the point: X, (B, G), whose cores the directions vary
        ww:                typ.Sequence[NDArray],        # input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        c:                 typ.Sequence[NDArray],        # P*⊗V_i, len=d, elm_shape=F+W+C+(Ni,)
        core_directions:   typ.Tuple,                    # X, (B_dot, G_dot), the shapes of B, G with K+C
        order:             typ.Tuple[int, ...] = (),     # the jet shape; F=tuple(n+1 for n in order)
        sum_over_samples:  bool = True,                  # False: W joins the stack, W+K+C
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                         # Tucker block: X*, len=d, elm_shape=[W]+K+C+(nDi,Ni)
        typ.Tuple[NDArray, ...],                         # TT block:     X*, len=d, elm_shape=[W]+K+C+(ri,nDi,r(i+1))
]:
    '''The corewise Hessian: the second derivative of ``phi(x) = <c, z(x)>`` in the cores, applied to
    ``core_directions`` (the derivative of :py:func:`t3_probe_corewise_transpose` along them).

    ``c`` is an element of ``Y`` (``P*⊗V_i`` per mode per sample), paired with ``z`` slot by slot. It carries no metric,
    and none is applied here: for the curvature term of a weighted least-squares fit (``Kp`` at ``c = M r``), pass
    ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weight and
    no scale. One call of :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` on the corewise point
    ``x`` wanting ``Kp`` with ``c`` held as ``r``. Summed over the samples (``sum_over_samples=False``: per sample,
    ``W`` leading the blocks' stack). For repeated calls at one frame (inside an iterative solver), call
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables` with ``held=``: the frame's sweeps are then
    computed once.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.probing as probing
    >>> from t3toolbox.backend.t3_constructors import t3_corewise_randn
    >>> from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
    >>> np.random.seed(0)
    >>> x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = t3_orthogonal_representations(x)                   # a frame, and a direction at it
    >>> ww = [np.random.randn(3, n) for n in (4, 5, 6)]                          # W = (3,)
    >>> c = probing.t3_probe(x, ww)
    >>> KB, KG = probing.t3_probe_corewise_hessian(x, ww, c, x)    # the direction x itself
    >>> print([b.shape for b in KB], [g.shape for g in KG])
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 2), (2, 2, 1)]
    '''
    held = sampling_variables.HeldVariables(r=c)
    (blocks,) = sampling_variables.sampling_variables('probe', x, ww, ('Kp',), order=order, held=held,
                                                       core_directions=core_directions,
                                                       sum_over_samples=sum_over_samples)
    return blocks


###############################################
##########    Probe dense tensor    ###########
###############################################

def _dense_contract_one_mode(
        A:          NDArray,    # shape=[W]+C+(the modes not yet contracted)
        has_W:      bool,       # does A lead with the sample stack W?
        vector:     NDArray,    # shape=W+(N_mode,): one coefficient of the input jet of the mode
        n_C:        int,        # the number of axes of C
        axis:       int,        # the place of the mode among the modes not yet contracted
) -> NDArray:                   # shape=W+C+(the modes not yet contracted, but this one)
    '''Contract one mode of a dense tensor (partly contracted already) with one vector: one einsum.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((A, vector)))
    n_W = vector.ndim - 1
    n_modes = A.ndim - n_C - (n_W if has_W else 0)
    # We are going to construct an einsum string from letters.
    # A dense 2x2x..x2 tensor exhausting these letters would have 4e15 entries
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    W_letters = letters[:n_W]
    C_letters = letters[n_W:n_W + n_C]
    mode_letters = letters[n_W + n_C:n_W + n_C + n_modes]
    kept = mode_letters[:axis] + mode_letters[axis + 1:]
    subscripts = ((W_letters if has_W else '') + C_letters + mode_letters + ',' + W_letters + mode_letters[axis]
                  + '->' + W_letters + C_letters + kept)
    return xnp.einsum(subscripts, A, vector)


def _dense_jet(
        T:          NDArray,                    # shape=C+(N0,...,N(d-1))
        ww:         typ.Sequence[NDArray],      # len=d, elm_shape=F_in_i+W+(Ni,): the input jets
        order:      typ.Tuple[int, ...],        # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        open_mode:  typ.Optional[int],          # the mode left open, or None: every mode is contracted
        name:       str,                        # the public function, for the error messages
) -> NDArray:                                   # shape=F+W+C+(N_open,), or F+W+C if open_mode is None
    '''The Taylor coefficients, on the box ``F = order + 1``, of ``T`` contracted with the input jets in every mode
    but ``open_mode``. The input jet of mode ``j`` is the polynomial ``w_j(s) = sum_k ww[j][k] s^k`` (``k`` a
    multi-index over its own jet sizes, zero beyond them), and the result is the product of the ``w_j(s)``
    through ``T``, truncated to the box. Computed one mode at a time: the partly contracted tensor is a polynomial
    whose coefficients are kept by multi-index, and contracting mode ``j`` multiplies it by ``w_j(s)``, by explicit
    loops over the pairs of multi-indices whose sum lies in the box, one einsum per pair. Exact: sums of
    contractions, with no step size and no interpolation. The one place the jets of a dense tensor are computed
    (the three public dense functions rest on it), and it shares nothing with the library's sweeps or with
    ``jet_contract``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((T, ww)))
    order = contractions._read_order(order, name)
    m = len(order)
    box = tuple(n + 1 for n in order)
    d = len(ww)
    if T.ndim < d:
        raise ValueError('%s: %d vectors for a tensor with %d axes' % (name, d, T.ndim))
    shape = T.shape[T.ndim - d:]
    n_C = T.ndim - d
    for ii, w in enumerate(ww):
        if w.ndim < m + 1:
            raise ValueError('%s: ww[%d] has %d axes, fewer than the %d jet axes of the jet shape %s and a mode axis'
                             % (name, ii, w.ndim, m, order))
    W = tuple(ww[0].shape[m:-1])
    for ii, w in enumerate(ww):
        if min(w.shape[:m], default=1) < 1:
            raise ValueError('%s: ww[%d] has the jet sizes %s: every jet axis has at least one slot'
                             % (name, ii, tuple(w.shape[:m])))
        if tuple(w.shape[m:-1]) != W:
            raise ValueError('%s: ww[%d] has sample stack %s once its %d jet axes are read, expected W = %s'
                             % (name, ii, tuple(w.shape[m:-1]), m, W))
        if w.shape[-1] != shape[ii]:
            raise ValueError('%s: ww[%d] has length %d, expected N_%d = %d' % (name, ii, w.shape[-1], ii, shape[ii]))

    coefficients = {(0,) * m: T}                # the tensor: a polynomial of degree 0, layout C + modes
    has_W = False                               # after the first contraction: W + C + the modes left
    left = list(range(d))
    for jj in range(d):
        if jj == open_mode:
            continue
        input_slots = list(itertools.product(*(range(min(size, f)) for size, f in zip(ww[jj].shape[:m], box))))
        product = {}
        for r, A in coefficients.items():
            for k in input_slots:               # slots of the input beyond the box are never read
                t = tuple(a + b for a, b in zip(r, k))
                if any(a > n for a, n in zip(t, order)):
                    continue
                term = _dense_contract_one_mode(A, has_W, ww[jj][k], n_C, left.index(jj))
                product[t] = term if t not in product else product[t] + term
        coefficients, has_W = product, True
        left.remove(jj)
    if not has_W:
        # d == 1 and its one mode is open: no vector is contracted, and the result is the tensor itself, broadcast
        # over W (the W letters would otherwise appear only in an einsum output, which numpy rejects)
        coefficients = {r: xnp.broadcast_to(A, W + tuple(A.shape)) for r, A in coefficients.items()}

    first = coefficients[(0,) * m]
    if m == 0:
        return first
    zero = xnp.zeros_like(first)                # a coefficient no product reaches: exactly zero
    slots = [coefficients.get(r, zero) for r in itertools.product(*(range(f) for f in box))]
    return xnp.reshape(xnp.stack(slots), box + tuple(first.shape))


def dense_probe(
        T:      NDArray,                                # E, shape=C+(N0,...,N(d-1))
        ww:     typ.Sequence[NDArray],                  # the input jets: P_in_i⊗V_i, len=d, elm_shape=F_in_i+W+(Ni,)
        order:  typ.Tuple[int, ...] = (),               # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
) -> typ.Tuple[NDArray, ...]:                           # z: P⊗V_i*, len=d, elm_shape=F+W+C+(Ni,)
    """Probe a dense tensor: contract it with a vector in every mode but one, for each mode in turn.

    The dense ground truth of ``probe``: ``z[i]`` is the tensor ``T`` contracted with ``ww[j]`` in every mode
    ``j != i``.

    **Jets.** With a jet shape ``order = (n_1, ..., n_m)`` the inputs are input jets in ``m`` variables
    ``s_1 ... s_m``: ``ww[j]`` has ``m`` leading jet axes of sizes of its own, ``F_in_j``, and its slot at the
    multi-index ``k`` is the Taylor coefficient of ``s_1^k_1 ... s_m^k_m`` of the input vector ``w_j(s)`` (zero
    beyond its sizes). ``z[i]`` is then the polynomial ``T`` contracted with ``w_j(s)`` for ``j != i``, given by
    its Taylor coefficients (**not** derivatives) on the box ``F = (n_1 + 1, ..., n_m + 1)``, on ``m`` leading jet
    axes. Input slots beyond the box are never read. The coefficients are exact: sums of contractions, computed
    one mode at a time. A line ``w + s p`` is the input jet of two slots, :py:func:`~t3toolbox.backend.jets.affine_jet`.
    ``order`` is never read off the inputs: with the default ``order = ()`` an input jet's leading axes are read
    as sample axes.

    In the shape comments ``F`` is the full jet sizes, ``F_in_i`` the jet sizes of the input ``ww[i]``; the
    argument ``T`` is the tensor.

    Parameters
    ----------
    T: NDArray
        The tensor to be probed. shape=C+(N0,...,N(d-1)); ``d`` is the number of vectors.
    ww: typ.Sequence[NDArray]
        The input jets. len=d, elm_shape=F_in_i+W+(Ni,); with ``order = ()``, W+(Ni,).
    order: typ.Tuple[int, ...]
        The jet shape: the maximum order in each jet variable. ``()`` (the default) for no jets.

    Returns
    -------
    typ.Tuple[NDArray, ...]
        The probes ``z``. len=d, elm_shape=F+W+C+(Ni,)

    Raises
    ------
    TypeError
        If ``order`` is not a tuple of integers (an integer is not a jet shape).
    ValueError
        If an entry of ``order`` is negative, or an input does not have the jet axes of the jet shape (each of
        at least one slot), the sample stack of ``ww[0]`` or the length of its mode.

    See Also
    --------
    t3toolbox.backend.apply.dense_apply
    t3toolbox.backend.entries.dense_entries
    t3toolbox.backend.jets.affine_jet

    Examples
    --------

    Probe with one set of vectors; value-match each mode against a hand-written einsum:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> u0, u1, u2 = np.random.randn(10), np.random.randn(11), np.random.randn(12)
    >>> yy = t3p.dense_probe(T, (u0, u1, u2))
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
    >>> yy = t3p.dense_probe(T, (u0, u1, u2))
    >>> y0 = np.einsum('ijk,uvj,uvk->uvi', T, u1, u2)
    >>> y1 = np.einsum('ijk,uvi,uvk->uvj', T, u0, u2)
    >>> y2 = np.einsum('ijk,uvi,uvj->uvk', T, u0, u1)
    >>> print(yy[0].shape)                      # W=(2,3) outer, then N0=10
    (2, 3, 10)
    >>> print([bool(np.allclose(y, ref)) for y, ref in zip(yy, (y0, y1, y2))])
    [True, True, True]

    Vectorize over probing vectors AND a stacked (big) tensor: frame-inner ``elm_shape = W + C + (Ni,)``:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> np.random.seed(0)
    >>> T = np.random.randn(4, 5, 6, 10, 11, 12)   # C=(4,5,6) stack on the tensor
    >>> u0, u1, u2 = np.random.randn(2, 3, 10), np.random.randn(2, 3, 11), np.random.randn(2, 3, 12)
    >>> yy = t3p.dense_probe(T, (u0, u1, u2))
    >>> y0 = np.einsum('xyzijk,uvj,uvk->uvxyzi', T, u1, u2)
    >>> y1 = np.einsum('xyzijk,uvi,uvk->uvxyzj', T, u0, u2)
    >>> y2 = np.einsum('xyzijk,uvi,uvj->uvxyzk', T, u0, u1)
    >>> print(yy[0].shape)                      # W=(2,3) outer, C=(4,5,6) inner, then N0=10
    (2, 3, 4, 5, 6, 10)
    >>> print([bool(np.allclose(y, ref)) for y, ref in zip(yy, (y0, y1, y2))])
    [True, True, True]

    Jets along a line ``ww + s pp``: the input jet of two slots, made by ``affine_jet``. The probes are
    polynomials in ``s`` of degree ``d - 1 = 2``, so slot 1 is the sum over the one mode that takes ``pp``, and
    the three coefficients sum to the probes at ``s = 1``:

    >>> import numpy as np
    >>> import t3toolbox.backend.probing as t3p
    >>> from t3toolbox.backend.jets import affine_jet
    >>> np.random.seed(0)
    >>> T = np.random.randn(10, 11, 12)
    >>> ww = [np.random.randn(2, N) for N in (10, 11, 12)]      # W=(2,)
    >>> pp = [np.random.randn(2, N) for N in (10, 11, 12)]
    >>> zz = t3p.dense_probe(T, affine_jet(ww, pp), order=(2,))
    >>> print(zz[0].shape)                      # F=(3,) outermost, then W=(2,), then N0=10
    (3, 2, 10)
    >>> print(bool(np.allclose(zz[0][0], t3p.dense_probe(T, ww)[0])))             # slot 0 is the plain probe
    True
    >>> print(bool(np.allclose(zz[0][1], np.einsum('ijk,wj,wk->wi', T, pp[1], ww[2])
    ...                                  + np.einsum('ijk,wj,wk->wi', T, ww[1], pp[2]))))   # slot 1: one mode takes pp
    True
    >>> moved = [w + p for w, p in zip(ww, pp)]                                    # the inputs at s = 1
    >>> print(bool(np.allclose(zz[0].sum(axis=0), t3p.dense_probe(T, moved)[0])))  # degree d-1 = 2: nothing above
    True

    An input of three slots, a curve ``w + s p + s^2 q`` in one variable: slot 2 of a probe has the terms where
    one mode takes ``q`` and those where two modes take ``p``:

    >>> qq = [np.random.randn(2, N) for N in (10, 11, 12)]
    >>> curve = [np.stack([w, p, q]) for w, p, q in zip(ww, pp, qq)]               # F_in = (3,): slots 0, 1, 2
    >>> zz = t3p.dense_probe(T, curve, order=(2,))
    >>> slot2 = (np.einsum('ijk,wj,wk->wi', T, qq[1], ww[2]) + np.einsum('ijk,wj,wk->wi', T, ww[1], qq[2])
    ...          + np.einsum('ijk,wj,wk->wi', T, pp[1], pp[2]))
    >>> print(bool(np.allclose(zz[0][2], slot2)))
    True

    Gotcha -- ``order`` is a tuple, the jet shape; an integer is not accepted:

    >>> t3p.dense_probe(T, affine_jet(ww, pp), 2)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: dense_probe: order is the jet shape, a tuple of maximum orders, ...
    """
    return tuple(_dense_jet(T, ww, order, ii, 'dense_probe') for ii in range(len(ww)))
