# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import math
import itertools
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.t3_operations as ragged_ops
from t3toolbox.backend.probing import compute_xis
from t3toolbox.backend.common import *

# Symmetric derivatives of probing.
#
# Probing maps a collection of input vectors X = (x_0, ..., x_{d-1}) to its actions
# Y = (y_0, ..., y_{d-1}), where y_i is the tensor contracted with x_j in every mode j != i (mode i
# left free). This module computes the *symmetric* directional derivatives of that map in a single
# repeated direction P = (p_0, ..., p_{d-1}):
#
#     y_i^(t) := d^t/ds^t y_i(X + s P) |_{s=0},    t = 0, 1, ..., K.
#
# Jet view. Because X + sP is affine in s, each input vector carries a trivial Taylor jet -- value at
# order 0, direction at order 1, zero above -- and every product of jets is a binomial convolution
# driven by the static tensor trs[t,r,s] = C(t,r) if r+s==t (binomial_combine_tensor). We call the
# order-stacked edge variables "jets" (order axis t; index 0 is the ordinary probing edge variable).
# The pushthrough recursion and the combine are then the SAME operation -- a binomial jet-product --
# and reduce to single einsums (the t-contractions in contractions.py):
#   - compute_mu_jets : left  jets mu_i^(t)  via trs_rWCa_Caib_sWCi_to_tWCb (input jet on the mode).
#   - compute_nu_jets : right jets nu_i^(t)  -- the mirror image (reverse_tt).
#   - compute_eta_jets: combine at each free mode via trs_rWCa_Caib_sWCb_to_tWCi (nu jet on the bond).
#   - assemble_z_jets : lift each order through the Tucker cores via tWCi_Cio_to_tWCo (t broadcast).
#
# Provenance: the symmetric-derivative formulation is NOT part of the published T4S paper (it was cut
# long ago). It is reconstructed from an old, unvetted derivation and will be written up afresh in a
# project note; the recursions here have been verified against a dense oracle, not against the paper.
#
# STACKING: two independent stacks, exactly the plain-probing W + C (base-inner).
#   - sample stack S: paired/flat, on the inputs ww/pp -- each sample is a paired (X, P) (duplicate a
#     point across S to sweep many directions at it). This is the probing W block.
#   - base/core stack C: on the cores -- a batch of T3s probed by the same samples.
# Derived edge-variable jets carry order + S + C + bond; outputs are order + S + C + (Ni,), with the
# order axis t OUTERMOST and base-inner (S outer, C inner). Either stack may be empty. The contractions
# self-infer the S/C split from operand shapes, so nothing here threads it.
#
# SCOPE: Euclidean, plain Tucker tensor train (the cores ARE the data, so left = right = down =
# tt_cores). The Riemannian (tangent-vector) case is deferred.

__all__ = [
    # Plain T3 (Euclidean)
    'probe_derivatives_t3',
    'build_input_jets',
    'compute_mu_jets',
    'compute_nu_jets',
    'compute_eta_jets',
    'assemble_z_jets',
    'binomial_combine_tensor',
    # Tangent vector (Riemannian) -- forward
    'probe_tangent_derivatives',
    'compute_sigma_jets',
    'compute_tau_jets',
    'compute_deta_jets',
    'assemble_tangent_z_jets',
    # Tangent vector (Riemannian) -- transpose
    'probe_tangent_derivatives_transpose',
    'compute_deta_tilde_jets',
    'compute_tau_tilde_jets',
    'compute_sigma_tilde_jets',
    'compute_dxi_tilde_jets',
    'assemble_tucker_variation_jets',
    'assemble_tt_variation_jets',
    # Dense oracle
    'probe_derivatives_dense',
]


def probe_derivatives_t3(
        ww:     typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=S+(Ni,)
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=S+(Ni,)
        x:      typ.Tuple[
            typ.Sequence[NDArray],      # tucker_cores. len=d, elm_shape=C+(nUi,Ni)
            typ.Sequence[NDArray],      # tt_cores.     len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        ],                              # = TuckerTensorTrain.data
        order:  int,                    # highest derivative order K
) -> typ.Tuple[NDArray, ...]:           # z_jets. len=d, elm_shape=(K+1,)+S+C+(Ni,)
    '''Symmetric derivatives of probing a Tucker tensor train, in one repeated direction.

    Returns, for each mode ``i``, the stack ``y_i^(t) = d^t/ds^t y_i(X + s P)|_0`` for ``t=0..K``,
    where ``y_i`` is the ``i``-th probing action. Index ``0`` is the ordinary probe ``probe_t3``.

    Two independent stacks ride through, base-inner as in plain probing: a sample stack ``S`` on the
    input vectors (each sample a paired ``(X, P)`` -- repeat a point across ``S`` to sweep many
    directions at it) and a base/core stack ``C`` on the cores (a batch of T3s probed by the same
    samples). Outputs are ``order + S + C + (Ni,)`` (``S`` outer, ``C`` inner); either may be empty.

    The symmetric-derivative formulation is not in the published T4S paper; the recursions are
    verified against :py:func:`probe_derivatives_dense`. (Project write-up in preparation.)

    Parameters
    ----------
    ww: typ.Sequence[NDArray]
        probe vectors X. len=d, elm_shape=S+(Ni,)
    pp: typ.Sequence[NDArray]
        perturbation direction P (the same P fed into every derivative slot). len=d, elm_shape=S+(Ni,)
    x: t3.TuckerTensorTrain.data
        Tucker tensor train, as a (tucker_cores, tt_cores) data tuple.
    order: int
        highest derivative order K.

    Returns
    -------
    typ.Tuple[NDArray,...]
        Probe-derivative jets, z_jets. len=d, elm_shape=(K+1,)+S+C+(Ni,). ``z_jets[i][t]`` is ``y_i^(t)``.

    See Also
    --------
    compute_mu_jets
    compute_nu_jets
    compute_eta_jets
    assemble_z_jets
    probe_derivatives_dense
    t3toolbox.backend.probing.probe_t3
    '''
    tucker_cores, tt_cores = x

    xis  = compute_xis(tucker_cores, ww)      # base projected probes,   U_i x_i
    dxis = compute_xis(tucker_cores, pp)       # projected perturbations, U_i p_i

    xi_jets = build_input_jets(xis, dxis)      # input jet on each mode: (xi, dxi) over order s

    trs = binomial_combine_tensor(order)       # trs[t,r,s] = C(t,r) if r+s==t (shared by all steps)

    mu_jets = compute_mu_jets(tt_cores, xi_jets, trs)

    nu_jets = compute_nu_jets(tt_cores, xi_jets, trs)

    eta_jets = compute_eta_jets(tt_cores, mu_jets, nu_jets, trs)

    z_jets = assemble_z_jets(tucker_cores, eta_jets)

    return z_jets


def binomial_combine_tensor(
        order:  int,        # highest derivative order K
) -> NDArray:               # trs. shape=(K+1,K+1,K+1). trs[t,r,s] = C(t,r) if r+s==t else 0
    '''Binomial tensor driving every jet convolution.

    ``trs[t,r,s] = C(t,r)`` when ``r+s==t`` and ``0`` otherwise, so contracting it against the outer
    product of two jets reproduces ``sum_{r+s=t} C(t,r) (.)^(r) (.)^(s)`` -- the product rule on
    Taylor jets. Used by both the pushthrough (input jet, ``s in {0,1}`` -> slice ``[:, :, :2]``) and
    the combine (full ``s``). Pure structure (exact integer binomials), so always numpy -- it folds
    into the compiled program as a device constant on the jax path, like the uniform masks.
    '''
    K = order
    trs = np.zeros((K + 1, K + 1, K + 1))
    for t in range(K + 1):
        for r in range(t + 1):
            trs[t, r, t - r] = math.comb(t, r)
    return trs


def build_input_jets(
        xis:    typ.Sequence[NDArray],  # base projected probes,        len=d, elm_shape=S+C+(nUi,)
        dxis:   typ.Sequence[NDArray],  # projected perturbation dirs,  len=d, elm_shape=S+C+(nUi,)
) -> typ.Tuple[NDArray, ...]:           # xi_jets. len=d, elm_shape=(2,)+S+C+(nUi,): order 0 = xi, 1 = dxi
    '''Input jets: stack each (value, direction) pair on a leading order axis.

    Since ``x + s p`` is affine in ``s``, an input vector's jet is just ``(x, p, 0, ...)`` -- value at
    order 0, direction at order 1, zero above. Stored at size 2 (orders 0,1); the pushthrough slices
    the binomial tensor to ``s in {0,1}`` accordingly.
    '''
    use_jax = tree_contains_jax((xis, dxis))
    xnp, _, _ = get_backend(False, use_jax)
    return tuple(xnp.stack([xi, dxi], axis=0) for xi, dxi in zip(xis, dxis))


def _init_jet(
        K:            int,                  # highest derivative order
        stack_shape:  typ.Tuple[int, ...],  # full leading batch the jets carry (sample + base, S + C)
        r0:           int,                  # leftmost bond dimension
        xnp,                                # numpy or jax.numpy
) -> NDArray:                               # mu_0 jet, shape=(K+1,)+S+C+(r0,): order 0 = ones, higher = 0
    '''Leftmost left-pushthrough jet mu_0^(t): the empty product (ones) at order 0, zero above.'''
    ones  = xnp.ones((1,) + stack_shape + (r0,))
    zeros = xnp.zeros((K,) + stack_shape + (r0,))
    return xnp.concatenate([ones, zeros], axis=0)


def compute_mu_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+S+C+(nUi,)
        trs:        NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
) -> typ.Tuple[NDArray, ...]:               # mu_jets. len=d, elm_shape=(K+1,)+S+C+(rLi,). mu_jets[i][t]=mu_{i-1}^(t)
    '''Left derivative-pushthrough jets.

    Sweep left-to-right, at each core taking the binomial jet-product of the running left jet with the
    input jet through the core (``trs_rWCa_Caib_sWCi_to_tWCb``). Like :py:func:`probing.compute_mus`,
    ``mu_jets[i]`` is the left edge variable *entering* core ``i`` (``mu_{i-1}``), stacked over orders.
    '''
    use_jax = tree_contains_jax((tt_cores, xi_jets, trs))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    K = trs.shape[0] - 1
    s_size = min(2, K + 1)                    # input jet carries orders {0, 1}, capped at K
    trs_push = trs[:, :, :s_size]

    def _func(mu_jet, data):
        G, xi_jet = data
        return contractions.trs_rWCa_Caib_sWCi_to_tWCb(trs_push, mu_jet, G, xi_jet[:s_size]), (mu_jet,)

    stack_shape = xi_jets[0].shape[1:-1]     # full S + C batch (S outer, C inner); either may be empty
    r0 = tt_cores[0].shape[-3]
    init = _init_jet(K, stack_shape, r0, xnp)

    _, (mu_jets,) = xscan(_func, init, (tt_cores, xi_jets))
    return mu_jets


def compute_nu_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+S+C+(nUi,)
        trs:        NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
) -> typ.Tuple[NDArray, ...]:               # nu_jets. len=d, elm_shape=(K+1,)+S+C+(rR(i+1),). nu_jets[i][t]=nu_i^(t)
    '''Right derivative-pushthrough jets.

    The mirror image of :py:func:`compute_mu_jets`: reverse the tensor train (``reverse_tt`` swaps
    bonds and core order), run the left sweep, reverse the result. ``nu_jets[i]`` is the right edge
    variable entering core ``i`` (``nu_i``), stacked over derivative orders.
    '''
    rev_nu_jets = compute_mu_jets(ragged_ops.reverse_tt(tt_cores), xi_jets[::-1], trs)
    return rev_nu_jets[::-1]


def compute_eta_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+C+(rLi,)
        nu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+C+(rR(i+1),)
        trs:        NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
) -> typ.Tuple[NDArray, ...]:               # eta_jets. len=d, elm_shape=(K+1,)+S+C+(nOi,). eta_jets[i][t]=eta_i^(t)
    '''Combine the left and right jets at each free mode via the binomial jet-product.

    ``eta_i^(t) = sum_{r+s=t} C(t,r) mu_{i-1}^(r) . G_i . nu_i^(s)``, one einsum per core
    (``trs_rWCa_Caib_sWCb_to_tWCi``) -- the same binomial convolution as the pushthrough, with the
    right jet on the bond and mode ``i`` left free.
    '''
    use_jax = tree_contains_jax((tt_cores, mu_jets, nu_jets, trs))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    def _func(data):
        mu_jet, G, nu_jet = data
        return (contractions.trs_rWCa_Caib_sWCb_to_tWCi(trs, mu_jet, G, nu_jet),)

    (eta_jets,) = xmap(_func, (mu_jets, tt_cores, nu_jets))
    return eta_jets


def assemble_z_jets(
        tucker_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(nUi,Ni)
        eta_jets:       typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+C+(nUi,)
) -> typ.Tuple[NDArray, ...]:                   # z_jets. len=d, elm_shape=(K+1,)+S+C+(Ni,)
    '''Lift the combined jets back to the ambient modes through the Tucker cores (order by order).

    The Tucker factor is order-independent, so ``z_i^(t) = U_i eta_i^(t)`` applies per order
    (``tWCi_Cio_to_tWCo``) -- the order axis rides as a leading broadcast batch.
    '''
    use_jax = tree_contains_jax((tucker_cores, eta_jets))
    is_uniform = is_ndarray(tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    def _func(data):
        eta_jet, U = data
        return (contractions.tWCi_Cio_to_tWCo(eta_jet, U),)

    (z_jets,) = xmap(_func, (eta_jets, tucker_cores))
    return z_jets


###############################################################
####    Riemannian: symmetric derivatives of a tangent     ####
###############################################################
#
# probe_tangent (probing.py) is the action of a tangent vector v on the probe vectors: a base sweep
# (xi, mu, nu, eta via the frame cores U, O, P, Q) plus a variation sweep (dxi, sigma, tau, deta via
# the variation cores dU, dG), then z = U.deta + dU.eta. Differentiating that map w.r.t. the probe
# vectors (the same direction P repeated) jet-ifies every edge variable -- and since every term of
# every recursion is itself a pushthrough, combine, or lift, each maps onto the SAME three
# t-contractions with the appropriate (frame or variation) core. The order axis t is uniform across
# ALL edge variables (everything depends on the probe vectors), so there is no K-style asymmetry: the
# plain t-contractions suffice (no n_base threading). Scope: a single tangent vector (no tangent
# stack K); stacks t + S + C as in the plain case. The transpose is a separate slice.


def _zero_jet(
        K:            int,                  # highest derivative order
        stack_shape:  typ.Tuple[int, ...],  # full leading batch the jets carry (sample + base, S + C)
        r:            int,                  # bond dimension
        xnp,                                # numpy or jax.numpy
) -> NDArray:                               # shape=(K+1,)+S+C+(r,): all orders zero
    '''All-zero jet -- the sigma_0 / tau_d boundary of the variation sweeps (no order-0 ones).'''
    return xnp.zeros((K + 1,) + stack_shape + (r,))


def compute_sigma_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+S+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+S+C+(nOi,)
        mu_jets:        typ.Sequence[NDArray],  # base left jets,  len=d, elm_shape=(K+1,)+S+C+(rLi,)
        trs:            NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
) -> typ.Tuple[NDArray, ...]:                   # sigma_jets. len=d, elm_shape=(K+1,)+S+C+(rR(i+1),)
    '''Variation-leftward edge-variable jets sigma (the jet-ified Algorithm-7 sigma recursion).

    ``sigma_i = sigma_{i-1} Q_i(xi_i) + mu_{i-1} dG_i(xi_i) + mu_{i-1} O_i(dxi_i)`` -- three
    pushthroughs (``trs_rWCa_Caib_sWCi_to_tWCb``): the carried sigma jet through Q, and the base mu
    jet through the variation core dG and the down frame O. Boundary ``sigma_0 = 0`` (all orders).
    '''
    use_jax = tree_contains_jax((var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets, trs))
    is_uniform = is_ndarray(var_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    K = trs.shape[0] - 1
    s_size = min(2, K + 1)                    # input jets carry orders {0, 1}, capped at K
    trs_push = trs[:, :, :s_size]
    push = contractions.trs_rWCa_Caib_sWCi_to_tWCb

    def _func(sigma_jet, data):
        Q, O, dG, xi_jet, dxi_jet, mu_jet = data
        t1 = push(trs_push, sigma_jet, Q,  xi_jet[:s_size])
        t2 = push(trs_push, mu_jet,    dG, xi_jet[:s_size])
        t3 = push(trs_push, mu_jet,    O,  dxi_jet[:s_size])
        return t1 + t2 + t3, (sigma_jet,)

    stack_shape = dxi_jets[0].shape[1:-1]     # full S + C batch (S outer, C inner); either may be empty
    rR0 = right_tt_cores[0].shape[-3]
    init = _zero_jet(K, stack_shape, rR0, xnp)

    _, (sigma_jets,) = xscan(_func, init, (right_tt_cores, down_tt_cores, var_tt_cores, xi_jets, dxi_jets, mu_jets))
    return sigma_jets


def compute_tau_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+S+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+S+C+(nOi,)
        nu_jets:        typ.Sequence[NDArray],  # base right jets, len=d, elm_shape=(K+1,)+S+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
) -> typ.Tuple[NDArray, ...]:                   # tau_jets. len=d, elm_shape=(K+1,)+S+C+(rL(i+1),)
    '''Variation-rightward edge-variable jets tau -- the mirror of :py:func:`compute_sigma_jets`.

    Reverse the train (P in the Q-slot, O and dG reversed), run the sigma sweep, reverse the result.
    '''
    rev = compute_sigma_jets(
        ragged_ops.reverse_tt(var_tt_cores), ragged_ops.reverse_tt(left_tt_cores),
        ragged_ops.reverse_tt(down_tt_cores), xi_jets[::-1], dxi_jets[::-1], nu_jets[::-1], trs,
    )
    return rev[::-1]


def compute_deta_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+C+(rR(i+1),)
        sigma_jets:     typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+C+(rR(i+1),)
        tau_jets:       typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+C+(rL(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
) -> typ.Tuple[NDArray, ...]:                   # deta_jets. len=d, elm_shape=(K+1,)+S+C+(nUi,)
    '''Variation-downward edge-variable jets deta (the jet-ified Algorithm-7 deta combine).

    ``deta_i = sigma_{i-1} Q_i nu_i + mu_{i-1} dG_i nu_i + mu_{i-1} P_i tau_i`` -- three combines
    (``trs_rWCa_Caib_sWCb_to_tWCi``), mode ``i`` free.
    '''
    use_jax = tree_contains_jax((var_tt_cores, left_tt_cores, right_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets, trs))
    is_uniform = is_ndarray(var_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    combine = contractions.trs_rWCa_Caib_sWCb_to_tWCi

    def _func(data):
        P, Q, dG, mu_jet, nu_jet, sigma_jet, tau_jet = data
        term1 = combine(trs, sigma_jet, Q,  nu_jet)
        term2 = combine(trs, mu_jet,    dG, nu_jet)
        term3 = combine(trs, mu_jet,    P,  tau_jet)
        return (term1 + term2 + term3,)

    (deta_jets,) = xmap(_func, (left_tt_cores, right_tt_cores, var_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets))
    return deta_jets


def assemble_tangent_z_jets(
        tucker_cores:       typ.Sequence[NDArray],  # U.  len=d, elm_shape=C+(nUi,Ni)
        var_tucker_cores:   typ.Sequence[NDArray],  # dU. len=d, elm_shape=C+(nOi,Ni)
        eta_jets:           typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+C+(nOi,)
        deta_jets:          typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+C+(nUi,)
) -> typ.Tuple[NDArray, ...]:                       # z_jets. len=d, elm_shape=(K+1,)+S+C+(Ni,)
    '''Assemble tangent-probe-derivative jets: ``z_i = U_i deta_i + dU_i eta_i`` -- two lifts
    (``tWCi_Cio_to_tWCo``), the order axis riding as a leading broadcast batch.
    '''
    use_jax = tree_contains_jax((tucker_cores, var_tucker_cores, eta_jets, deta_jets))
    is_uniform = is_ndarray(tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    lift = contractions.tWCi_Cio_to_tWCo

    def _func(data):
        U, dU, eta_jet, deta_jet = data
        return (lift(deta_jet, U) + lift(eta_jet, dU),)

    (z_jets,) = xmap(_func, (tucker_cores, var_tucker_cores, eta_jets, deta_jets))
    return z_jets


def probe_tangent_derivatives(
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=S+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=S+(Ni,)
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores dU. len=d, elm_shape=C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores     dG. len=d, elm_shape=C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data
        base:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # = T3Basis.data = (up, down, left, right) = (U, O, P, Q)
        order:      int,                    # highest derivative order K
) -> typ.Tuple[NDArray, ...]:               # z_jets. len=d, elm_shape=(K+1,)+S+C+(Ni,)
    '''Symmetric derivatives of probing a tangent vector, in one repeated direction (Riemannian J^(s)).

    The probe-derivative analog of :py:func:`probing.probe_tangent`: returns, for each mode ``i``, the
    stack ``y_i^(t) = d^t/ds^t [J^(s) v]_i (X + s P)|_0`` for ``t=0..K``, where ``v`` is the tangent
    vector represented by ``(base, variation)``. Index ``0`` is the ordinary tangent probe.

    Identical structure to :py:func:`probe_derivatives_t3` but on the tangent calculus: a base sweep
    (frame cores) and a variation sweep, every term reusing the three ``t``-contractions. Stacks are
    ``order + S + C`` as in the plain case (single tangent -- no tangent stack ``K`` yet).

    The symmetric-derivative formulation is not in the published T4S paper; verified against
    :py:func:`probe_derivatives_dense` on the densified tangent. (Project write-up in preparation.)

    See Also
    --------
    probe_derivatives_t3
    compute_sigma_jets
    compute_tau_jets
    compute_deta_jets
    assemble_tangent_z_jets
    t3toolbox.backend.probing.probe_tangent
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    var_tucker_cores, var_tt_cores = variation

    xi_jets  = build_input_jets(compute_xis(up_tucker_cores, ww),  compute_xis(up_tucker_cores, pp))
    dxi_jets = build_input_jets(compute_xis(var_tucker_cores, ww), compute_xis(var_tucker_cores, pp))

    trs = binomial_combine_tensor(order)

    # base sweep (frame cores), reusing the plain jet functions
    mu_jets  = compute_mu_jets(left_tt_cores, xi_jets, trs)
    nu_jets  = compute_nu_jets(right_tt_cores, xi_jets, trs)
    eta_jets = compute_eta_jets(down_tt_cores, mu_jets, nu_jets, trs)

    # variation sweep
    sigma_jets = compute_sigma_jets(var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets, trs)
    tau_jets   = compute_tau_jets(var_tt_cores, left_tt_cores, down_tt_cores, xi_jets, dxi_jets, nu_jets, trs)
    deta_jets  = compute_deta_jets(var_tt_cores, left_tt_cores, right_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets, trs)

    z_jets = assemble_tangent_z_jets(up_tucker_cores, var_tucker_cores, eta_jets, deta_jets)

    return z_jets


###############################################################
####    Riemannian transpose (the jet-ified adjoint)       ####
###############################################################
#
# The transpose of probe_tangent_derivatives (linear in the variation): residual jets r -> variation
# gradient (dU_tilde, dG_tilde). Derived as the jet-ified adjoint-state Lagrangian (t4s.pdf Thm 7):
# every forward contraction in the Lagrangian is replaced by its trs version, and stationarity
# d L / d (state, variation) = 0 gives the adjoint sweeps and the gradient assembly -- which is exactly
# the verified non-derivative transpose (probing.compute_*_tildes / assemble_*) with each contraction
# swapped for its ADJOINT-HOOKED trs version (same trs tensor, transposed legs: the multiplier's order
# is summed, the swept order is freed) for the sweeps, and the ORDER-LESS trs (TT, 3 edges) / plain
# order-sum (Tucker, 1 edge) for the assembly. Stacks (this slice): sample stack S only (C deferred);
# sum_over_probes sums S (the J^T r back-projection) else keeps it. Verified against jax.linear_transpose.

def compute_deta_tilde_jets(
        up_tucker_cores:    typ.Sequence[NDArray],  # U.  len=d, elm_shape=(nUi,Ni)
        ztildes:            typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(K+1,)+S+(Ni,)
) -> typ.Tuple[NDArray, ...]:                       # deta_tildes. len=d, elm_shape=(K+1,)+S+(nUi,)
    '''Adjoint-up edge-variable jets: ``deta_tilde_i = U_i r_i`` (contract the ambient mode, order
    diagonal). The 1-internal-edge (Tucker) case -- the order axis just rides through (no trs).'''
    use_jax = tree_contains_jax((up_tucker_cores, ztildes))
    xnp, xmap, _ = get_backend(False, use_jax)

    def _func(data):
        U, zt = data
        return (xnp.einsum('io,t...o->t...i', U, zt),)

    (deta_tildes,) = xmap(_func, (up_tucker_cores, ztildes))
    return deta_tildes


def _adj_sweep(P_cores, xi_jets, deta_tildes, edge_jets, trs):
    '''The jet adjoint sweep shared by compute_tau_tilde_jets / compute_sigma_tilde_jets: a left-to-right scan
    (mirroring probing.compute_tau_tildes) of the adjoint-hooked pushthrough (propagation) plus the
    deta_tilde source. Both terms are the same trs, wired as the transpose (output at the swept order s).'''
    use_jax = tree_contains_jax((P_cores, xi_jets, deta_tildes, edge_jets, trs))
    xnp, xmap, xscan = get_backend(False, use_jax)
    s_size = min(2, trs.shape[0])
    trs_xi = trs[:, :s_size, :]            # input jet (xi) carries orders {0, 1}

    def _step(carry, data):
        P, xi, deta_t, edge = data
        prop = xnp.einsum('tus,t...a,aib,u...i->s...b', trs_xi, carry, P, xi[:s_size])  # propagation
        src  = xnp.einsum('trs,r...a,aib,t...i->s...b', trs,    edge,  P, deta_t)        # deta_tilde source
        return prop + src, (carry,)

    rL0 = P_cores[0].shape[-3]
    init = xnp.zeros((trs.shape[0],) + deta_tildes[0].shape[1:-1] + (rL0,))
    _, (tildes,) = xscan(_step, init, (P_cores, xi_jets, deta_tildes, edge_jets))
    return tildes


def compute_tau_tilde_jets(
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=(rLi,nUi,rL(i+1))
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+S+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(K+1,)+S+(nUi,)
        mu_jets:        typ.Sequence[NDArray],  # base left jets,  len=d, elm_shape=(K+1,)+S+(rLi,)
        trs:            NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
) -> typ.Tuple[NDArray, ...]:                   # tau_tildes. len=d, elm_shape=(K+1,)+S+(rL(i+1),)
    '''Adjoint-var-rightward edge-variable jets (jet-ified probing.compute_tau_tildes).'''
    return _adj_sweep(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)


def compute_sigma_tilde_jets(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=(rRi,nUi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+S+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(K+1,)+S+(nUi,)
        nu_jets:        typ.Sequence[NDArray],  # base right jets, len=d, elm_shape=(K+1,)+S+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
) -> typ.Tuple[NDArray, ...]:                   # sigma_tildes. len=d, elm_shape=(K+1,)+S+(rR(i+1),)
    '''Adjoint-var-leftward edge-variable jets -- the mirror (reverse) of compute_tau_tilde_jets.'''
    rev = _adj_sweep(ragged_ops.reverse_tt(right_tt_cores), xi_jets[::-1],
                     deta_tildes[::-1], nu_jets[::-1], trs)
    return rev[::-1]


def compute_dxi_tilde_jets(
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=(rLi,nOi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+(rR(i+1),)
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+(rL(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
) -> typ.Tuple[NDArray, ...]:                   # dxi_tildes. len=d, elm_shape=(K+1,)+S+(nOi,)
    '''Adjoint-var-down edge-variable jets (jet-ified probing.compute_dxi_tildes): two adjoint-hooked
    combines giving delta-xi-tilde on the mode (output at the order-<=1 leg u).'''
    use_jax = tree_contains_jax((down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes, trs))
    xnp, xmap, _ = get_backend(False, use_jax)

    def _func(data):
        O, mu, nu, st, tt = data
        from_tau = xnp.einsum('tus,t...a,aib,s...b->u...i', trs, tt, O, nu)
        from_sig = xnp.einsum('tru,r...a,aib,t...b->u...i', trs, mu, O, st)
        return (from_tau + from_sig,)

    (dxi_tildes,) = xmap(_func, (down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes))
    return dxi_tildes


def _w_jets(ww, pp, xnp):
    '''The ambient input jets (value, direction) on the raw probe vectors -- the order axis for dU.'''
    return tuple(xnp.stack([w, p], axis=0) for w, p in zip(ww, pp))


def assemble_tucker_variation_jets(
        ztildes:        typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(K+1,)+S+(Ni,)
        dxi_tildes:     typ.Sequence[NDArray],  # adjoint-var-down jets, len=d, elm_shape=(K+1,)+S+(nOi,)
        ww:             typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=S+(Ni,)
        pp:             typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=S+(Ni,)
        etas:           typ.Sequence[NDArray],  # base down jets, len=d, elm_shape=(K+1,)+S+(nOi,)
        sum_over_probes: bool,
) -> typ.Tuple[NDArray, ...]:                   # dU_tildes. len=d, elm_shape=[S+](nOi,Ni)
    '''Assemble Tucker-core variation gradients (the 1-edge, plain-order-sum case):
    ``dU_tilde = sum_t eta^(t) (x) r^(t) + sum_u dxi_tilde^(u) (x) w_jet^(u)``.'''
    use_jax = tree_contains_jax((ztildes, dxi_tildes, ww, pp, etas))
    xnp, xmap, _ = get_backend(False, use_jax)
    w_jets = _w_jets(ww, pp, xnp)
    len_S = etas[0].ndim - 2
    s_size = min(2, etas[0].shape[0])          # the w/dxi input jet carries orders {0, 1}, capped at K

    def _func(data):
        zt, dxt, eta, wj = data
        dU = (xnp.einsum('t...a,t...o->...ao', eta, zt)                       # eta (x) r,         sum t
              + xnp.einsum('u...a,u...o->...ao', dxt[:s_size], wj[:s_size]))  # dxi_tilde (x) w_jet, sum u
        if sum_over_probes:
            dU = xnp.sum(dU, axis=tuple(range(len_S)))
        return (dU,)

    (dU_tildes,) = xmap(_func, (ztildes, dxi_tildes, etas, w_jets))
    return dU_tildes


def assemble_tt_variation_jets(
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+(rL(i+1),)
        deta_tildes:    typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+(nUi,)
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+S+(nUi,)
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(K+1,)+S+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(K+1,K+1,K+1)
        sum_over_probes: bool,
) -> typ.Tuple[NDArray, ...]:                   # dG_tildes. len=d, elm_shape=[S+](rLi,nUi,rRi)
    '''Assemble TT-core variation gradients (the 3-edge, trs case): three order-less trs outer products
    ``mu (x) xi (x) sigma_tilde + tau_tilde (x) xi (x) nu + mu (x) deta_tilde (x) nu`` (the core-adjoints
    of the forward sigma / tau / deta contractions).'''
    use_jax = tree_contains_jax((sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets, trs))
    xnp, xmap, _ = get_backend(False, use_jax)
    s_size = min(2, trs.shape[0])
    len_S = mu_jets[0].ndim - 2

    def _func(data):
        xi, mu, nu, st, tt, dt = data
        t_sig = xnp.einsum('tru,r...a,u...i,t...b->...aib', trs[:, :, :s_size], mu, xi[:s_size], st)
        t_tau = xnp.einsum('tus,t...a,u...i,s...b->...aib', trs[:, :s_size, :], tt, xi[:s_size], nu)
        t_det = xnp.einsum('trs,r...a,t...i,s...b->...aib', trs,                mu, dt,           nu)
        dG = t_sig + t_tau + t_det
        if sum_over_probes:
            dG = xnp.sum(dG, axis=tuple(range(len_S)))
        return (dG,)

    (dG_tildes,) = xmap(_func, (xi_jets, mu_jets, nu_jets, sigma_tildes, tau_tildes, deta_tildes))
    return dG_tildes


def probe_tangent_derivatives_transpose(
        ztildes:    typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(K+1,)+S+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=S+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=S+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        order:      int,                    # highest derivative order K
        sum_over_probes: bool = False,      # True: sum the sample stack S (the J^T r back-projection)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes (Tucker variation gradient)
    typ.Tuple[NDArray, ...],  # dG_tildes (TT variation gradient)
]:                                          # = T3Variations.data
    '''Transpose of :py:func:`probe_tangent_derivatives`: back-project residual jets ``ztildes`` into a
    variation gradient ``(dU_tildes, dG_tildes)``. The jet-ified adjoint-state method (t4s.pdf Thm 7):
    every forward contraction is swapped for its ``trs`` version, then stationarity of the Lagrangian
    gives the adjoint sweeps (``sigma/tau/dxi_tilde`` jets) and the order-less gradient assembly.

    With ``sum_over_probes=False`` the sample stack ``S`` rides through into the variation stack; with
    ``True`` it is summed (the ``J^T r`` back-projection used for fitting). Verified against the dense
    adjoint identity ``<r, J v> = <J^T r, v>``. Single tangent (no tangent stack ``K``); ``S`` only.
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base

    trs = binomial_combine_tensor(order)
    xi_jets = build_input_jets(compute_xis(up_tucker_cores, ww), compute_xis(up_tucker_cores, pp))

    mu_jets  = compute_mu_jets(left_tt_cores, xi_jets, trs)
    nu_jets  = compute_nu_jets(right_tt_cores, xi_jets, trs)
    eta_jets = compute_eta_jets(down_tt_cores, mu_jets, nu_jets, trs)

    deta_tildes  = compute_deta_tilde_jets(up_tucker_cores, ztildes)
    tau_tildes   = compute_tau_tilde_jets(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)
    sigma_tildes = compute_sigma_tilde_jets(right_tt_cores, xi_jets, deta_tildes, nu_jets, trs)
    dxi_tildes   = compute_dxi_tilde_jets(down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes, trs)

    dU_tildes = assemble_tucker_variation_jets(ztildes, dxi_tildes, ww, pp, eta_jets, sum_over_probes)
    dG_tildes = assemble_tt_variation_jets(sigma_tildes, tau_tildes, deta_tildes, xi_jets,
                                       mu_jets, nu_jets, trs, sum_over_probes)

    return dU_tildes, dG_tildes


#####################################################
########    Dense reference (test oracle)    ########
#####################################################

def probe_derivatives_dense(
        ww:     typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=(Ni,)
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=(Ni,)
        T:      NDArray,                # dense tensor, shape=(N0,...,N(d-1))
        order:  int,                    # highest derivative order K
) -> typ.List[NDArray]:                 # z_jets. len=d, elm_shape=(K+1,Ni)
    '''Exact dense symmetric probe derivatives, by the multilinear subset expansion (test oracle).

    Each action ``y_i`` is multilinear in the off-mode vectors, so ``y_i(X+sP)`` is a polynomial in
    ``s`` and the ``t``-th derivative at ``s=0`` is exact:

        ``y_i^(t) = t! * sum_{|S|=t, S subset of modes\\{i}} T contracted with {p_j: j in S, x_j: else}``.

    Enumerates the size-``t`` subsets ``S`` -- only for small ``d``/``t`` (testing), unstacked, no
    rank structure.
    '''
    d = T.ndim
    letters = 'abcdefghijklmnopqrstuvwxyz'
    T_sub = letters[:d]

    z_jets = []
    for i in range(d):
        others = [j for j in range(d) if j != i]
        z_i = np.zeros((order + 1, T.shape[i]))
        for t in range(order + 1):
            acc = np.zeros(T.shape[i])
            for subset in itertools.combinations(others, t):
                operands = [T]
                in_subs = [T_sub]
                for j in others:
                    operands.append(pp[j] if j in subset else ww[j])
                    in_subs.append(letters[j])
                acc = acc + np.einsum(','.join(in_subs) + '->' + letters[i], *operands)
            z_i[t] = math.factorial(t) * acc
        z_jets.append(z_i)
    return z_jets
