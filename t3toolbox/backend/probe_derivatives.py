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
from t3toolbox.backend.probing import compute_xis, _entry_xis
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
# STACKING: the SAME three blocks as plain probing (base-inner W + K + C), plus the convolved order
# axis t. See docs/batching_and_stacking.md.
#   - sample stack W: on the inputs ww/pp -- each sample is a paired (X, P) (duplicate a point across W
#     to sweep many directions at it). This IS the probing probe stack W.
#   - tangent stack K: on the variation cores only -- a batch of tangent vectors sharing one base
#     (Riemannian only). Rides the variation-derived jets (sigma/tau/deta/dxi), not the base jets.
#   - base/core stack C: on the cores -- a batch of T3s.
# The order axis t (the derivative orders, convolved by the binomial tensor trs) is placed OUTERMOST;
# the passive blocks are base-inner W + K + C. So base jets carry order + W + C, variation jets carry
# order + W + K + C, and outputs are order + W + (K) + C + (Ni,). Any block may be empty. The
# contractions self-infer the W/K/C split from operand shapes, so nothing here threads it (a
# variation-core-only term takes n_base = len(C); the n_probe precedent).

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
    # Apply / entries derivatives (all-modes special case) -- forward
    'apply_derivatives_t3',
    'apply_tangent_derivatives',
    'entries_derivatives_t3',
    'entries_tangent_derivatives',
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
    'apply_derivatives_dense',
    'entries_derivatives_dense',
]


def probe_derivatives_t3(
        ww:     typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        x:      typ.Tuple[
            typ.Sequence[NDArray],      # tucker_cores. len=d, elm_shape=C+(nUi,Ni)
            typ.Sequence[NDArray],      # tt_cores.     len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        ],                              # = TuckerTensorTrain.data
        order:  int,                    # highest derivative order K
) -> typ.Tuple[NDArray, ...]:           # z_jets. len=d, elm_shape=(order+1,)+W+C+(Ni,)
    '''Symmetric derivatives of probing a Tucker tensor train, in one repeated direction.

    Returns, for each mode ``i``, the stack ``y_i^(t) = d^t/ds^t y_i(X + s P)|_0`` for ``t=0..K``,
    where ``y_i`` is the ``i``-th probing action. Index ``0`` is the ordinary probe ``probe_t3``.

    Two independent stacks ride through, base-inner as in plain probing: a sample stack ``W`` on the
    input vectors (each sample a paired ``(X, P)`` -- repeat a point across ``W`` to sweep many
    directions at it) and a base/core stack ``C`` on the cores (a batch of T3s probed by the same
    samples). Outputs are ``order + W + C + (Ni,)`` (``W`` outer, ``C`` inner); either may be empty.
    (A plain T3 has no tangent stack ``K``; the Riemannian forwards carry ``order + W + K + C``.)

    The symmetric-derivative formulation is not in the published T4S paper; the recursions are
    verified against :py:func:`probe_derivatives_dense`. (Project write-up in preparation.)

    Parameters
    ----------
    ww: typ.Sequence[NDArray]
        probe vectors X. len=d, elm_shape=W+(Ni,)
    pp: typ.Sequence[NDArray]
        perturbation direction P (the same P fed into every derivative slot). len=d, elm_shape=W+(Ni,)
    x: t3.TuckerTensorTrain.data
        Tucker tensor train, as a (tucker_cores, tt_cores) data tuple.
    order: int
        highest derivative order K.

    Returns
    -------
    typ.Tuple[NDArray,...]
        Probe-derivative jets, z_jets. len=d, elm_shape=(order+1,)+W+C+(Ni,). ``z_jets[i][t]`` is ``y_i^(t)``.

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
) -> NDArray:               # trs. shape=(order+1,order+1,order+1). trs[t,r,s] = C(t,r) if r+s==t else 0
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
        xis:    typ.Sequence[NDArray],  # base projected probes,        len=d, elm_shape=W+C+(nUi,)
        dxis:   typ.Sequence[NDArray],  # projected perturbation dirs,  len=d, elm_shape=W+C+(nUi,)
) -> typ.Tuple[NDArray, ...]:           # xi_jets. len=d, elm_shape=(2,)+W+C+(nUi,): order 0 = xi, 1 = dxi
    '''Input jets: stack each (value, direction) pair on a leading order axis.

    Since ``x + s p`` is affine in ``s``, an input vector's jet is just ``(x, p, 0, ...)`` -- value at
    order 0, direction at order 1, zero above. Stored at size 2 (orders 0,1); the pushthrough slices
    the binomial tensor to ``s in {0,1}`` accordingly.
    '''
    use_jax = tree_contains_jax((xis, dxis))
    xnp, _, _ = get_backend(False, use_jax)
    return tuple(xnp.stack([xi, dxi], axis=0) for xi, dxi in zip(xis, dxis))


def _init_jet(
        order:        int,                  # highest derivative order
        stack_shape:  typ.Tuple[int, ...],  # full leading batch the base jet carries (sample + base, W + C)
        r0:           int,                  # leftmost bond dimension
        xnp,                                # numpy or jax.numpy
) -> NDArray:                               # mu_0 jet, shape=(order+1,)+W+C+(r0,): order 0 = ones, higher = 0
    '''Leftmost left-pushthrough jet mu_0^(t): the empty product (ones) at order 0, zero above.'''
    ones  = xnp.ones((1,) + stack_shape + (r0,))
    zeros = xnp.zeros((order,) + stack_shape + (r0,))
    return xnp.concatenate([ones, zeros], axis=0)


def compute_mu_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        trs:        NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:               # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,). mu_jets[i][t]=mu_{i-1}^(t)
    '''Left derivative-pushthrough jets.

    Sweep left-to-right, at each core taking the binomial jet-product of the running left jet with the
    input jet through the core (``trs_rWCa_Caib_sWCi_to_tWCb``). Like :py:func:`probing.compute_mus`,
    ``mu_jets[i]`` is the left edge variable *entering* core ``i`` (``mu_{i-1}``), stacked over orders.
    '''
    use_jax = tree_contains_jax((tt_cores, xi_jets, trs))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    order = trs.shape[0] - 1
    s_size = min(2, order + 1)                # input jet carries orders {0, 1}, capped at order
    trs_push = trs[:, :, :s_size]

    def _func(mu_jet, data):
        G, xi_jet = data
        return contractions.trs_rWCa_Caib_sWCi_to_tWCb(trs_push, mu_jet, G, xi_jet[:s_size]), (mu_jet,)

    stack_shape = xi_jets[0].shape[1:-1]     # full W + C batch (W outer, C inner); either may be empty
    r0 = tt_cores[0].shape[-3]
    init = _init_jet(order, stack_shape, r0, xnp)

    _, (mu_jets,) = xscan(_func, init, (tt_cores, xi_jets))
    return mu_jets


def compute_nu_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        trs:        NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:               # nu_jets. len=d, elm_shape=(order+1,)+W+C+(rR(i+1),). nu_jets[i][t]=nu_i^(t)
    '''Right derivative-pushthrough jets.

    The mirror image of :py:func:`compute_mu_jets`: reverse the tensor train (``reverse_tt`` swaps
    bonds and core order), run the left sweep, reverse the result. ``nu_jets[i]`` is the right edge
    variable entering core ``i`` (``nu_i``), stacked over derivative orders.
    '''
    rev_nu_jets = compute_mu_jets(ragged_ops.reverse_tt(tt_cores), xi_jets[::-1], trs)
    return rev_nu_jets[::-1]


def compute_eta_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:        NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:               # eta_jets. len=d, elm_shape=(order+1,)+W+C+(nOi,). eta_jets[i][t]=eta_i^(t)
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
        eta_jets:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(nUi,)
) -> typ.Tuple[NDArray, ...]:                   # z_jets. len=d, elm_shape=(order+1,)+W+C+(Ni,)
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
# every recursion is itself a pushthrough, combine, or lift, each maps onto the order-threaded
# contractions with the appropriate (frame or variation) core. The order axis t is uniform across
# ALL edge variables (everything depends on the probe vectors). The base sweep (xi, mu, nu, eta) uses
# the 2-block (W,C) order-threaded contractions; the variation sweep (dxi, sigma, tau, deta) carries
# the tangent stack K, so it uses the order-threaded THREE-block (W,K,C) contractions, exactly as
# probe_tangent's perturbation sweep does (K on the variation cores; n_base = len(C) where the only
# core operand is a variation core). Stacks t + W + K + C, base-inner. The transpose is a separate slice.


def _zero_jet(
        order:        int,                  # highest derivative order
        stack_shape:  typ.Tuple[int, ...],  # full leading batch the variation jets carry (W + K + C)
        r:            int,                  # bond dimension
        xnp,                                # numpy or jax.numpy
) -> NDArray:                               # shape=(order+1,)+W+K+C+(r,): all orders zero
    '''All-zero jet -- the sigma_0 / tau_d boundary of the variation sweeps (no order-0 ones).'''
    return xnp.zeros((order + 1,) + stack_shape + (r,))


def _sigma_jet_step(sigma_jet, Q, O, dG, xi_jet, dxi_jet, mu_jet, trs_push):
    '''One step of the K-aware jet-ified sigma recursion (the perturbation-leftward jets), shared by
    compute_sigma_jets (keeps the per-core sequence) and apply_tangent_derivatives (keeps only the
    terminal carry). Three-group (W sample, K tangent, C base): sigma/dxi carry K, the base edge vars
    (xi, mu) and base cores (Q, O) do not; t2's only core is the variation core dG (K+C), so len(C) is
    supplied via n_base (recovered from the C-only Q). Reduces to the 2-group result when K=().'''
    s_size = trs_push.shape[2]                # input jets carry orders {0, 1}, capped at order
    n_base = Q.ndim - 3
    t1 = contractions.trs_rWKCa_Caib_sWCi_to_tWKCb(trs_push, sigma_jet, Q,  xi_jet[:s_size])
    t2 = contractions.trs_rWCa_KCaib_sWCi_to_tWKCb(trs_push, mu_jet,    dG, xi_jet[:s_size], n_base)
    t3 = contractions.trs_rWCa_Caib_sWKCi_to_tWKCb(trs_push, mu_jet,    O,  dxi_jet[:s_size])
    return t1 + t2 + t3


def compute_sigma_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+C+(nOi,)
        mu_jets:        typ.Sequence[NDArray],  # base left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # sigma_jets. len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
    '''Variation-leftward edge-variable jets sigma (the jet-ified Algorithm-7 sigma recursion).

    ``sigma_i = sigma_{i-1} Q_i(xi_i) + mu_{i-1} dG_i(xi_i) + mu_{i-1} O_i(dxi_i)`` -- three
    pushthroughs (``trs_rWCa_Caib_sWCi_to_tWCb``): the carried sigma jet through Q, and the base mu
    jet through the variation core dG and the down frame O. Boundary ``sigma_0 = 0`` (all orders).
    '''
    use_jax = tree_contains_jax((var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets, trs))
    is_uniform = is_ndarray(var_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    order = trs.shape[0] - 1
    s_size = min(2, order + 1)                # input jets carry orders {0, 1}, capped at order
    trs_push = trs[:, :, :s_size]

    def _func(sigma_jet, data):
        Q, O, dG, xi_jet, dxi_jet, mu_jet = data
        return _sigma_jet_step(sigma_jet, Q, O, dG, xi_jet, dxi_jet, mu_jet, trs_push), (sigma_jet,)

    # carry sigma is W+K+C; take the leading stack from dxi_jets (which carry K), not xi_jets (W+C only)
    stack_shape = dxi_jets[0].shape[1:-1]
    rR0 = right_tt_cores[0].shape[-3]
    init = _zero_jet(order, stack_shape, rR0, xnp)

    _, (sigma_jets,) = xscan(_func, init, (right_tt_cores, down_tt_cores, var_tt_cores, xi_jets, dxi_jets, mu_jets))
    return sigma_jets


def compute_tau_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+C+(nOi,)
        nu_jets:        typ.Sequence[NDArray],  # base right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # tau_jets. len=d, elm_shape=(order+1,)+W+C+(rL(i+1),)
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
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_jets:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        tau_jets:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rL(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # deta_jets. len=d, elm_shape=(order+1,)+W+C+(nUi,)
    '''Variation-downward edge-variable jets deta (the jet-ified Algorithm-7 deta combine).

    ``deta_i = sigma_{i-1} Q_i nu_i + mu_{i-1} dG_i nu_i + mu_{i-1} P_i tau_i`` -- three combines
    (``trs_rWCa_Caib_sWCb_to_tWCi``), mode ``i`` free.
    '''
    use_jax = tree_contains_jax((var_tt_cores, left_tt_cores, right_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets, trs))
    is_uniform = is_ndarray(var_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    def _func(data):
        P, Q, dG, mu_jet, nu_jet, sigma_jet, tau_jet = data
        # Three-group (W,K,C): sigma/tau carry K, mu/nu and base cores P/Q do not; term2's only core
        # is the variation core dG (K+C), so len(C) is supplied via n_base (the C-only Q pins it).
        n_base = Q.ndim - 3
        term1 = contractions.trs_rWKCa_Caib_sWCb_to_tWKCi(trs, sigma_jet, Q,  nu_jet)
        term2 = contractions.trs_rWCa_KCaib_sWCb_to_tWKCi(trs, mu_jet,    dG, nu_jet, n_base)
        term3 = contractions.trs_rWCa_Caib_sWKCb_to_tWKCi(trs, mu_jet,    P,  tau_jet)
        return (term1 + term2 + term3,)

    (deta_jets,) = xmap(_func, (left_tt_cores, right_tt_cores, var_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets))
    return deta_jets


def assemble_tangent_z_jets(
        tucker_cores:       typ.Sequence[NDArray],  # U.  len=d, elm_shape=C+(nUi,Ni)
        var_tucker_cores:   typ.Sequence[NDArray],  # dU. len=d, elm_shape=C+(nOi,Ni)
        eta_jets:           typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(nOi,)
        deta_jets:          typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(nUi,)
) -> typ.Tuple[NDArray, ...]:                       # z_jets. len=d, elm_shape=(order+1,)+W+C+(Ni,)
    '''Assemble tangent-probe-derivative jets: ``z_i = U_i deta_i + dU_i eta_i`` -- two lifts
    (``tWCi_Cio_to_tWCo``), the order axis riding as a leading broadcast batch.
    '''
    use_jax = tree_contains_jax((tucker_cores, var_tucker_cores, eta_jets, deta_jets))
    is_uniform = is_ndarray(tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    def _func(data):
        U, dU, eta_jet, deta_jet = data
        # Three-group (W,K,C): deta carries K (lift via the C-only U fuses W+K); eta is W+C and dU is
        # the variation core (K+C), so the eta-lift needs len(C) -- recovered from the C-only U.
        n_base = U.ndim - 2
        return (contractions.tWKCi_Cio_to_tWKCo(deta_jet, U)
                + contractions.tWCi_KCio_to_tWKCo(eta_jet, dU, n_base),)

    (z_jets,) = xmap(_func, (tucker_cores, var_tucker_cores, eta_jets, deta_jets))
    return z_jets


def probe_tangent_derivatives(
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
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
) -> typ.Tuple[NDArray, ...]:               # z_jets. len=d, elm_shape=(order+1,)+W+C+(Ni,)
    '''Symmetric derivatives of probing a tangent vector, in one repeated direction (Riemannian J^(s)).

    The probe-derivative analog of :py:func:`probing.probe_tangent`: returns, for each mode ``i``, the
    stack ``y_i^(t) = d^t/ds^t [J^(s) v]_i (X + s P)|_0`` for ``t=0..K``, where ``v`` is the tangent
    vector represented by ``(base, variation)``. Index ``0`` is the ordinary tangent probe.

    Identical structure to :py:func:`probe_derivatives_t3` but on the tangent calculus: a base sweep
    (frame cores) and a variation sweep. Full ``order + W + K + C`` stacking (base-inner, mirroring
    probe_tangent): sample stack ``W``, tangent stack ``K`` (a batch of tangents sharing the base, on
    the variation cores), base stack ``C``; any may be empty.

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
####    Apply / entries derivatives (all-modes special)    ####
###############################################################
#
# apply and entries are the all-modes special case of probing (probing leaves ONE mode free; these
# contract EVERY mode). As in probing.py, with no free mode the computation collapses to a single
# left-to-right pass: the base left sweep mu (via P/G) feeds the perturbation sweep sigma (via Q),
# kept only to its TERMINAL carry and contracted at the terminal bond -- no right (nu) / central (eta)
# sweeps, no per-mode assembly. The order axis t and the W/K/C stacks ride exactly as in probing's
# apply (here apply derivatives output order + W + K + C, a scalar-jet per stack element). entries is
# apply with the up-index xis from slicing Tucker-core fibers (deferred to its own step).


def _apply_derivatives_t3_from_xi_jets(xi_jets, tt_cores, trs):
    '''Terminal mu-jet carry of the left sweep (via the cores), bond summed -- the all-modes Euclidean
    tail shared by apply_derivatives_t3 and entries_derivatives_t3 (they differ only in how xi_jets is
    formed). Returns ``(order+1,) + W + C`` (no K for a plain T3).
    '''
    use_jax = tree_contains_jax((tt_cores, xi_jets, trs))
    xnp, xmap, xscan = get_backend(False, use_jax)
    order = trs.shape[0] - 1
    s_size = min(2, order + 1)
    trs_push = trs[:, :, :s_size]

    def _func(mu_jet, data):
        G, xi_jet = data
        return contractions.trs_rWCa_Caib_sWCi_to_tWCb(trs_push, mu_jet, G, xi_jet[:s_size]), (0,)

    stack_shape = xi_jets[0].shape[1:-1]                # W + C
    r0 = tt_cores[0].shape[-3]
    init = _init_jet(order, stack_shape, r0, xnp)
    mu_terminal, _ = xscan(_func, init, (tt_cores, xi_jets))
    return xnp.sum(mu_terminal, axis=-1)               # contract the terminal bond -> (order+1,)+W+C


def apply_derivatives_t3(
        ww:     typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        x:      typ.Tuple[
            typ.Sequence[NDArray],      # tucker_cores. len=d, elm_shape=C+(nUi,Ni)
            typ.Sequence[NDArray],      # tt_cores.     len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        ],                              # = TuckerTensorTrain.data
        order:  int,                    # highest derivative order
) -> NDArray:                           # apply-derivative jets, shape=(order+1,)+W+C
    '''Symmetric derivatives of applying a Tucker tensor train in all modes, in one repeated direction.

    The all-modes Euclidean analog of :py:func:`probe_derivatives_t3`: returns the stack
    ``y^(t) = d^t/ds^t apply(X, W + s P)|_0`` for ``t=0..order``, where ``apply`` contracts the tensor
    with the vectors in every mode (a scalar). Index ``0`` is the ordinary apply. Computed as the
    terminal carry of the left mu-jet sweep (via the cores), bond summed -- no right/central sweeps.
    Stacks ride as ``order + W + C`` (sample stack ``W``, base stack ``C``; no tangent stack here).
    Verified against :py:func:`apply_derivatives_dense`.
    '''
    tucker_cores, tt_cores = x
    xi_jets = build_input_jets(compute_xis(tucker_cores, ww), compute_xis(tucker_cores, pp))
    return _apply_derivatives_t3_from_xi_jets(xi_jets, tt_cores, binomial_combine_tensor(order))


def _apply_derivatives_from_jets(
        xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores, trs,
):
    '''Run the K-aware perturbation sigma-jet sweep to its TERMINAL carry and contract the final bond.

    The jet analog of :py:func:`probing._apply_from_xis`: reuses :py:func:`_sigma_jet_step` (so it is
    `W+K+C`-stacked) but keeps only the terminal carry. Returns ``(order+1,) + W + K + C``.
    '''
    use_jax = tree_contains_jax((xi_jets, dxi_jets, mu_jets, right_tt_cores, trs))
    xnp, xmap, xscan = get_backend(False, use_jax)
    order = trs.shape[0] - 1
    s_size = min(2, order + 1)
    trs_push = trs[:, :, :s_size]

    def _func(sigma_jet, data):
        Q, O, dG, xi_jet, dxi_jet, mu_jet = data
        return _sigma_jet_step(sigma_jet, Q, O, dG, xi_jet, dxi_jet, mu_jet, trs_push), (0,)

    stack_shape = dxi_jets[0].shape[1:-1]              # W + K + C (dxi carries K)
    rR0 = right_tt_cores[0].shape[-3]
    init = _zero_jet(order, stack_shape, rR0, xnp)
    sigma_terminal, _ = xscan(_func, init, (right_tt_cores, down_tt_cores, var_tt_cores, xi_jets, dxi_jets, mu_jets))
    return xnp.sum(sigma_terminal, axis=-1)            # contract the terminal bond -> (order+1,)+W+K+C


def apply_tangent_derivatives(
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores dU. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores     dG. len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data
        base:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # = T3Basis.data = (up, down, left, right) = (U, O, P, Q)
        order:      int,                    # highest derivative order
) -> NDArray:                               # apply-derivative jets, shape=(order+1,)+W+K+C
    '''Symmetric derivatives of applying a tangent vector in all modes, in one repeated direction.

    The all-modes Riemannian analog of :py:func:`probe_tangent_derivatives` (and the derivative analog
    of :py:func:`probing.apply_tangent`): returns ``y^(t) = d^t/ds^t [apply(v, W + s P)]|_0`` for
    ``t=0..order``, where ``v`` is the tangent vector ``(base, variation)``. A single left-to-right pass
    (base mu via P, perturbation sigma via Q) to the terminal carry, bond summed. Stacks ``order + W +
    K + C`` (sample stack ``W``, tangent stack ``K``, base stack ``C``). Verified against
    :py:func:`apply_derivatives_dense` on the densified tangent.

    See Also
    --------
    apply_derivatives_t3
    probe_tangent_derivatives
    probing.apply_tangent
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    var_tucker_cores, var_tt_cores = variation

    xi_jets  = build_input_jets(compute_xis(up_tucker_cores, ww),  compute_xis(up_tucker_cores, pp))
    dxi_jets = build_input_jets(compute_xis(var_tucker_cores, ww), compute_xis(var_tucker_cores, pp))
    trs = binomial_combine_tensor(order)

    mu_jets = compute_mu_jets(left_tt_cores, xi_jets, trs)        # base left sweep via P

    return _apply_derivatives_from_jets(
        xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores, trs)


def entries_derivatives_t3(
        index:  NDArray,                # int, shape=(d,)+W -- the grid points (one multi-index per W sample)
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        x:      typ.Tuple[
            typ.Sequence[NDArray],      # tucker_cores. len=d, elm_shape=C+(nUi,Ni)
            typ.Sequence[NDArray],      # tt_cores.     len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        ],                              # = TuckerTensorTrain.data
        order:  int,                    # highest derivative order
) -> NDArray:                           # entries-derivative jets, shape=(order+1,)+W+C
    '''Symmetric derivatives of an entry of a Tucker tensor train, in one repeated direction.

    The ``entries`` analog of :py:func:`apply_derivatives_t3` -- apply-derivatives with the up-index
    base jet from slicing Tucker-core fibers at ``index`` (order 0) and contracting ``P`` (order 1).
    Returns ``y^(t) = d^t/ds^t apply(X, e_{index} + s P)|_0`` for ``t=0..order``: the Taylor data of
    the tensor's multilinear extension at grid corner ``index``, in direction ``P``. Index ``0`` is the
    ordinary entry ``X[index]``. Stacks ``order + W + C``. Verified vs :py:func:`entries_derivatives_dense`.
    '''
    tucker_cores, tt_cores = x
    xi_jets = build_input_jets(_entry_xis(tucker_cores, index), compute_xis(tucker_cores, pp))
    return _apply_derivatives_t3_from_xi_jets(xi_jets, tt_cores, binomial_combine_tensor(order))


def entries_tangent_derivatives(
        index:      NDArray,                # int, shape=(d,)+W -- the grid points
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores dU. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores     dG. len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data
        base:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # = T3Basis.data = (up, down, left, right) = (U, O, P, Q)
        order:      int,                    # highest derivative order
) -> NDArray:                               # entries-derivative jets, shape=(order+1,)+W+K+C
    '''Symmetric derivatives of an entry of a tangent vector, in one repeated direction.

    The ``entries`` analog of :py:func:`apply_tangent_derivatives` -- identical but with the base/var
    up-index jets from slicing Tucker-core fibers at ``index`` (order 0) + contracting ``P`` (order 1).
    Stacks ``order + W + K + C``. Verified vs :py:func:`entries_derivatives_dense` on the densified tangent.

    See Also
    --------
    entries_derivatives_t3
    apply_tangent_derivatives
    probing.entries_tangent
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    var_tucker_cores, var_tt_cores = variation

    xi_jets  = build_input_jets(_entry_xis(up_tucker_cores, index),  compute_xis(up_tucker_cores, pp))
    dxi_jets = build_input_jets(_entry_xis(var_tucker_cores, index), compute_xis(var_tucker_cores, pp))
    trs = binomial_combine_tensor(order)

    mu_jets = compute_mu_jets(left_tt_cores, xi_jets, trs)

    return _apply_derivatives_from_jets(
        xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores, trs)


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
# order-sum (Tucker, 1 edge) for the assembly. Stacks (base-inner W + C, as in plain probing): sample
# stack W on the inputs, base stack C on the cores; sum_over_probes sums W (the J^T r back-projection)
# else keeps it (W rides into the variation stack). Verified against jax.linear_transpose + the
# adjoint identity. The named contractions live in contractions.py (the *_to_sWCb / *_to_uWCi sweeps,
# the order-less *_to_[W]Caib / *_to_[W]Cao assembly).

def compute_deta_tilde_jets(
        up_tucker_cores:    typ.Sequence[NDArray],  # U.  len=d, elm_shape=C+(nUi,Ni)
        ztildes:            typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+C+(Ni,)
) -> typ.Tuple[NDArray, ...]:                       # deta_tildes. len=d, elm_shape=(order+1,)+W+C+(nUi,)
    '''Adjoint-up edge-variable jets: ``deta_tilde_i = U_i r_i`` (contract the ambient mode, order
    diagonal). The 1-internal-edge (Tucker) case -- the order axis just rides through (no trs).'''
    use_jax = tree_contains_jax((up_tucker_cores, ztildes))
    xnp, xmap, _ = get_backend(False, use_jax)

    def _func(data):
        U, zt = data
        return (contractions.tWCo_Cio_to_tWCi(zt, U),)

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
        prop = contractions.trs_tWCa_Caib_uWCi_to_sWCb(trs_xi, carry, P, xi[:s_size])  # propagation
        src  = contractions.trs_rWCa_Caib_tWCi_to_sWCb(trs,    edge,  P, deta_t)       # deta_tilde source
        return prop + src, (carry,)

    rL0 = P_cores[0].shape[-3]
    init = xnp.zeros((trs.shape[0],) + deta_tildes[0].shape[1:-1] + (rL0,))
    _, (tildes,) = xscan(_step, init, (P_cores, xi_jets, deta_tildes, edge_jets))
    return tildes


def compute_tau_tilde_jets(
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(order+1,)+W+C+(nUi,)
        mu_jets:        typ.Sequence[NDArray],  # base left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # tau_tildes. len=d, elm_shape=(order+1,)+W+C+(rL(i+1),)
    '''Adjoint-var-rightward edge-variable jets (jet-ified probing.compute_tau_tildes).'''
    return _adj_sweep(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)


def compute_sigma_tilde_jets(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(order+1,)+W+C+(nUi,)
        nu_jets:        typ.Sequence[NDArray],  # base right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # sigma_tildes. len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
    '''Adjoint-var-leftward edge-variable jets -- the mirror (reverse) of compute_tau_tilde_jets.'''
    rev = _adj_sweep(ragged_ops.reverse_tt(right_tt_cores), xi_jets[::-1],
                     deta_tildes[::-1], nu_jets[::-1], trs)
    return rev[::-1]


def compute_dxi_tilde_jets(
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rL(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # dxi_tildes. len=d, elm_shape=(order+1,)+W+C+(nOi,)
    '''Adjoint-var-down edge-variable jets (jet-ified probing.compute_dxi_tildes): two adjoint-hooked
    combines giving delta-xi-tilde on the mode (output at the order-<=1 leg u).'''
    use_jax = tree_contains_jax((down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes, trs))
    xnp, xmap, _ = get_backend(False, use_jax)

    def _func(data):
        O, mu, nu, st, tt = data
        from_tau = contractions.trs_tWCa_Caib_sWCb_to_uWCi(trs, tt, O, nu)
        from_sig = contractions.trs_rWCa_Caib_tWCb_to_uWCi(trs, mu, O, st)
        return (from_tau + from_sig,)

    (dxi_tildes,) = xmap(_func, (down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes))
    return dxi_tildes


def _w_jets(ww, pp, xnp):
    '''The ambient input jets (value, direction) on the raw probe vectors -- the order axis for dU.'''
    return tuple(xnp.stack([w, p], axis=0) for w, p in zip(ww, pp))


def assemble_tucker_variation_jets(
        ztildes:        typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+C+(Ni,)
        dxi_tildes:     typ.Sequence[NDArray],  # adjoint-var-down jets, len=d, elm_shape=(order+1,)+W+C+(nOi,)
        ww:             typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:             typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        etas:           typ.Sequence[NDArray],  # base down jets, len=d, elm_shape=(order+1,)+W+C+(nOi,)
        n_probe:        int,                    # number of sample-stack (W) axes
        sum_over_probes: bool,
) -> typ.Tuple[NDArray, ...]:                   # dU_tildes. len=d, elm_shape=[W+]C+(nOi,Ni)
    '''Assemble Tucker-core variation gradients (the 1-edge, plain-order-sum case):
    ``dU_tilde = sum_t eta^(t) (x) r^(t) + sum_u dxi_tilde^(u) (x) w_jet^(u)``.'''
    use_jax = tree_contains_jax((ztildes, dxi_tildes, ww, pp, etas))
    xnp, xmap, _ = get_backend(False, use_jax)
    w_jets = _w_jets(ww, pp, xnp)
    s_size = min(2, etas[0].shape[0])          # the w/dxi input jet carries orders {0, 1}, capped at K
    eta_r = contractions.tWCa_tWCo_to_Cao if sum_over_probes else contractions.tWCa_tWCo_to_WCao
    dxi_w = contractions.uWCa_uWo_to_Cao  if sum_over_probes else contractions.uWCa_uWo_to_WCao

    def _func(data):
        zt, dxt, eta, wj = data
        return (eta_r(eta, zt, n_probe) + dxi_w(dxt[:s_size], wj[:s_size], n_probe),)

    (dU_tildes,) = xmap(_func, (ztildes, dxi_tildes, etas, w_jets))
    return dU_tildes


def assemble_tt_variation_jets(
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rL(i+1),)
        deta_tildes:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(nUi,)
        xi_jets:        typ.Sequence[NDArray],  # base input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
        n_probe:        int,                    # number of sample-stack (W) axes
        sum_over_probes: bool,
) -> typ.Tuple[NDArray, ...]:                   # dG_tildes. len=d, elm_shape=[W+]C+(rLi,nUi,rRi)
    '''Assemble TT-core variation gradients (the 3-edge, trs case): three order-less trs outer products
    ``mu (x) xi (x) sigma_tilde + tau_tilde (x) xi (x) nu + mu (x) deta_tilde (x) nu`` (the core-adjoints
    of the forward sigma / tau / deta contractions).'''
    use_jax = tree_contains_jax((sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets, trs))
    xnp, xmap, _ = get_backend(False, use_jax)
    s_size = min(2, trs.shape[0])
    if sum_over_probes:
        f_sig, f_tau, f_det = (contractions.trs_rWCa_uWCi_tWCb_to_Caib,
                               contractions.trs_tWCa_uWCi_sWCb_to_Caib,
                               contractions.trs_rWCa_tWCi_sWCb_to_Caib)
    else:
        f_sig, f_tau, f_det = (contractions.trs_rWCa_uWCi_tWCb_to_WCaib,
                               contractions.trs_tWCa_uWCi_sWCb_to_WCaib,
                               contractions.trs_rWCa_tWCi_sWCb_to_WCaib)

    def _func(data):
        xi, mu, nu, st, tt, dt = data
        t_sig = f_sig(trs[:, :, :s_size], mu, xi[:s_size], st, n_probe)
        t_tau = f_tau(trs[:, :s_size, :], tt, xi[:s_size], nu, n_probe)
        t_det = f_det(trs,                mu, dt,           nu, n_probe)
        return (t_sig + t_tau + t_det,)

    (dG_tildes,) = xmap(_func, (xi_jets, mu_jets, nu_jets, sigma_tildes, tau_tildes, deta_tildes))
    return dG_tildes


def probe_tangent_derivatives_transpose(
        ztildes:    typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        base:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Basis.data = (U, O, P, Q)
        order:      int,                    # highest derivative order K
        sum_over_probes: bool = False,      # True: sum the sample stack W (the J^T r back-projection)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes (Tucker variation gradient)
    typ.Tuple[NDArray, ...],  # dG_tildes (TT variation gradient)
]:                                          # = T3Variations.data
    '''Transpose of :py:func:`probe_tangent_derivatives`: back-project residual jets ``ztildes`` into a
    variation gradient ``(dU_tildes, dG_tildes)``. The jet-ified adjoint-state method (t4s.pdf Thm 7):
    every forward contraction is swapped for its ``trs`` version, then stationarity of the Lagrangian
    gives the adjoint sweeps (``sigma/tau/dxi_tilde`` jets) and the order-less gradient assembly.

    Two stacks ride through, base-inner as in plain probing: a sample stack ``W`` on the inputs and a
    base/core stack ``C`` on the cores. With ``sum_over_probes=False`` the sample stack ``W`` rides
    through into the variation stack; with ``True`` it is summed (the ``J^T r`` back-projection used for
    fitting), ``C`` always kept. Verified against the dense adjoint identity ``<r, J v> = <J^T r, v>``
    and ``jax.linear_transpose``. Single tangent (no tangent stack ``K``).
    '''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base
    n_probe = ww[0].ndim - 1   # number of sample-stack (W) axes (ww carries W, no C)

    trs = binomial_combine_tensor(order)
    xi_jets = build_input_jets(compute_xis(up_tucker_cores, ww), compute_xis(up_tucker_cores, pp))

    mu_jets  = compute_mu_jets(left_tt_cores, xi_jets, trs)
    nu_jets  = compute_nu_jets(right_tt_cores, xi_jets, trs)
    eta_jets = compute_eta_jets(down_tt_cores, mu_jets, nu_jets, trs)

    deta_tildes  = compute_deta_tilde_jets(up_tucker_cores, ztildes)
    tau_tildes   = compute_tau_tilde_jets(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)
    sigma_tildes = compute_sigma_tilde_jets(right_tt_cores, xi_jets, deta_tildes, nu_jets, trs)
    dxi_tildes   = compute_dxi_tilde_jets(down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes, trs)

    dU_tildes = assemble_tucker_variation_jets(
        ztildes, dxi_tildes, ww, pp, eta_jets, n_probe, sum_over_probes)
    dG_tildes = assemble_tt_variation_jets(
        sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets, trs, n_probe, sum_over_probes)

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


def apply_derivatives_dense(
        ww:     typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=(Ni,)
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=(Ni,)
        T:      NDArray,                # dense tensor, shape=(N0,...,N(d-1))
        order:  int,                    # highest derivative order
) -> NDArray:                           # apply-derivative jets, shape=(order+1,)
    '''Exact dense symmetric apply derivatives, by the all-modes multilinear subset expansion (oracle).

    ``apply(X+sP)`` is a polynomial in ``s`` of degree ``d``, so the ``t``-th derivative at ``s=0`` is

        ``y^(t) = t! * sum_{|S|=t, S subset of {0..d-1}} T contracted with {p_j: j in S, x_j: else}``,

    here contracting **every** mode (no free mode). Enumerates the size-``t`` subsets ``S`` -- testing
    only, unstacked.
    '''
    d = T.ndim
    letters = 'abcdefghijklmnopqrstuvwxyz'
    T_sub = letters[:d]

    y = np.zeros(order + 1)
    for t in range(order + 1):
        acc = 0.0
        for subset in itertools.combinations(range(d), t):
            operands = [T]
            in_subs = [T_sub]
            for j in range(d):
                operands.append(pp[j] if j in subset else ww[j])
                in_subs.append(letters[j])
            acc = acc + np.einsum(','.join(in_subs) + '->', *operands)
        y[t] = math.factorial(t) * acc
    return y


def entries_derivatives_dense(
        index:  typ.Sequence[int],      # the grid point (one int per mode), len=d
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=(Ni,)
        T:      NDArray,                # dense tensor, shape=(N0,...,N(d-1))
        order:  int,                    # highest derivative order
) -> NDArray:                           # entries-derivative jets, shape=(order+1,)
    '''Exact dense symmetric entry derivatives (oracle): apply-derivatives with one-hot base vectors
    ``e_{index}`` (entries = apply with one-hot), via :py:func:`apply_derivatives_dense`. Unstacked.'''
    ww = [np.eye(T.shape[j])[index[j]] for j in range(T.ndim)]   # one-hot e_{index_j}
    return apply_derivatives_dense(ww, pp, T, order)
