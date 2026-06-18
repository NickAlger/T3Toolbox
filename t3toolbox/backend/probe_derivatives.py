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
