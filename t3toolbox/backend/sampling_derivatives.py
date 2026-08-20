# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Jet machinery: symmetric directional derivatives of probe/apply/entries, all in one place.

The ``*_derivatives`` ops strictly generalize the plain sampling ops with a leading order axis
(order 0 == the plain op). The jets of the helper chain (``compute_*_jets``,
``binomial_combine_tensor``) are shared across the three sampling types, which is why the
derivative layer is one module rather than three. Math: ``docs/symmetric_probe_derivatives.tex``;
costs/usage: ``docs/entries_apply_probe.md``.
"""
import math
import functools
import itertools
import numpy as np
import typing as typ

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.t3_operations as ragged_ops
import t3toolbox.backend.ut3_operations as uniform_ops
from t3toolbox.backend.probing import compute_xi
from t3toolbox.backend.entries import _entry_xis, _onehot_vectors
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
#   - compute_mu_jets_trs : left  jets mu_i^(t)  via contract('trs,rWCa,Caib,sWCi->tWCb', ...) (input jet on the mode).
#   - compute_nu_jets_trs : right jets nu_i^(t)  -- the mirror image (tt_reverse).
#   - compute_eta_jets_trs: combine at each free mode via contract('trs,rWCa,Caib,sWCb->tWCi', ...) (nu jet on the bond).
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
#   - tangent stack K: on the variation cores only -- a batch of tangent vectors sharing one frame
#     (Riemannian only). Rides the variation-derived jets (sigma/tau/deta/dxi), not the frame jets.
#   - frame/core stack C: on the cores -- a batch of T3s.
# The order axis t (the derivative orders, convolved by the binomial tensor trs) is placed OUTERMOST;
# the passive blocks are base-inner W + K + C. So frame jets carry order + W + C, variation jets carry
# order + W + K + C, and outputs are order + W + (K) + C + (Ni,). Any block may be empty. The
# contractions self-infer the W/K/C split from operand shapes, so nothing here threads it (a
# variation-core-only term takes n_frame = len(C); the n_probe precedent).

__all__ = [
    # Input validation (X/P sample-stack consistency)
    'check_perturbation_vectors',
    'check_perturbation_index',
    # Plain T3 (Euclidean)
    't3_probe_derivatives',
    'build_input_jets',
    'compute_mu_jets',           # standard (recurrence/scan); *_trs = dense-binomial reference form
    'compute_nu_jets',
    'compute_eta_jets',
    'compute_mu_jets_trs',
    'compute_nu_jets_trs',
    'compute_eta_jets_trs',
    'assemble_z_jets',
    'binomial_combine_tensor',
    # Frame sweep (the reusable, variation-free jets -- the fitting inner-solve reuse hook). Per-kind:
    # apply/entries are lean (xi, mu) -- the adjoint-state forward+transpose use no nu/eta; probe is full.
    'tv_precompute_apply_frame_sweep_jets',
    'tv_precompute_entries_frame_sweep_jets',
    'tv_precompute_probe_frame_sweep_jets',
    # Tangent vector (Riemannian) -- forward
    'tv_probe_derivatives',
    'tv_probe_jacobian_derivatives_from_sweep',
    'compute_sigma_jets',
    'compute_tau_jets',
    'compute_deta_jets',
    'compute_sigma_jets_trs',
    'compute_tau_jets_trs',
    'compute_deta_jets_trs',
    'assemble_tangent_z_jets',
    # Apply / entries derivatives (all-modes special case) -- forward
    't3_apply_derivatives',
    'tv_apply_derivatives',
    'tv_apply_jacobian_derivatives_from_sweep',
    't3_entries_derivatives',
    'tv_entries_derivatives',
    'tv_entries_jacobian_derivatives_from_sweep',
    # Tangent vector (Riemannian) -- transpose
    'tv_probe_derivatives_transpose',
    'tv_probe_transpose_derivatives_from_sweep',
    'estimate_chunk_size',            # eager helpers to choose the transpose chunk_size (docs/chunking.md)
    'max_chunk_size_within',
    'compute_deta_tilde_jets',
    'compute_tau_tilde_jets',
    'compute_sigma_tilde_jets',
    'compute_tau_tilde_jets_trs',
    'compute_sigma_tilde_jets_trs',
    'compute_dxi_tilde_jets',
    'assemble_tucker_variation_jets',
    'assemble_tt_variation_jets',
    'assemble_tucker_variation_jets_trs',
    'assemble_tt_variation_jets_trs',
    # Apply / entries derivatives transpose (adjoint-state)
    'tv_apply_derivatives_transpose',
    'tv_apply_transpose_derivatives_from_sweep',
    'tv_entries_derivatives_transpose',
    'tv_entries_transpose_derivatives_from_sweep',
    'compute_sigma_hat_jets',
    # Corewise (non-manifold) derivative transposes
    't3_probe_corewise_derivatives_transpose',
    't3_apply_corewise_derivatives_transpose',
    't3_entries_corewise_derivatives_transpose',
    # Dense oracle
    'dense_probe_derivatives',
    'dense_apply_derivatives',
    'dense_entries_derivatives',
]


def t3_probe_derivatives(
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
    where ``y_i`` is the ``i``-th probing action. Index ``0`` is the ordinary probe ``t3_probe``.

    Two independent stacks ride through, base-inner as in plain probing: a sample stack ``W`` on the
    input vectors (each sample a paired ``(X, P)`` -- repeat a point across ``W`` to sweep many
    directions at it) and a frame/core stack ``C`` on the cores (a batch of T3s probed by the same
    samples). Outputs are ``order + W + C + (Ni,)`` (``W`` outer, ``C`` inner); either may be empty.
    (A plain T3 has no tangent stack ``K``; the Riemannian forwards carry ``order + W + K + C``.)

    The symmetric-derivative formulation is not in the published T4S paper; the recursions are
    verified against :py:func:`dense_probe_derivatives`. (Project write-up in preparation.)

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
    dense_probe_derivatives
    t3toolbox.backend.probing.t3_probe
    '''
    tucker_cores, tt_cores = x

    xis  = compute_xi(tucker_cores, ww)      # frame projected probes,   U_i x_i
    dxis = compute_xi(tucker_cores, pp)       # projected perturbations, U_i p_i

    xi_jets = build_input_jets(xis, dxis)      # input jet on each mode: (xi, dxi) over order s

    trs = binomial_combine_tensor(order)       # trs[t,r,s] = C(t,r) if r+s==t (shared by all steps)

    mu_jets = compute_mu_jets(tt_cores, xi_jets, trs)

    nu_jets = compute_nu_jets(tt_cores, xi_jets, trs)

    eta_jets = compute_eta_jets(tt_cores, mu_jets, nu_jets, trs)

    z_jets = assemble_z_jets(tucker_cores, eta_jets)

    return z_jets


def check_perturbation_vectors(
        ww: typ.Sequence[NDArray],  # points X,       len=d, elm_shape=W+(Ni,)
        pp: typ.Sequence[NDArray],  # perturbation P, len=d, elm_shape=W+(Ni,)
) -> None:
    '''Structural check (hard error): the perturbation ``P`` (``pp``) shares the sample stack ``W`` and
    mode dims of the points ``X`` (``ww``) -- each sample pairs a point with a direction. Used by the
    derivative-probe/apply frontends; shapes are static, so this is jit-safe.'''
    for i, (w, p) in enumerate(zip(ww, pp)):
        if np.shape(w) != np.shape(p):
            raise ValueError(
                "perturbation P must match the shape (sample stack W and mode dim) of the points X; "
                "mode %d: P %s vs X %s" % (i, np.shape(p), np.shape(w)))


def check_perturbation_index(
        index: NDArray,                          # grid points, int, shape=(d,)+W
        pp:    typ.Sequence[NDArray],            # perturbation P, len=d, elm_shape=W+(Ni,)
        shape: typ.Optional[typ.Sequence[int]]   # ambient mode dims (Ni), len=d -- if given, also check P's mode dim
             = None,
) -> None:
    '''Structural check (hard error): the perturbation ``P`` shares the sample stack ``W`` of the grid
    points ``index`` (shape ``(d,)+W``), and -- when ``shape`` is given (the ambient mode dims ``Ni``,
    which the integer ``index`` does not carry) -- ``P``'s mode dim matches it. Used by the
    derivative-entries frontends; jit-safe.'''
    iW = tuple(np.shape(index)[1:])
    for i, p in enumerate(pp):
        if tuple(np.shape(p)[:-1]) != iW:
            raise ValueError(
                "perturbation P's sample stack W must match index's; mode %d: P %s vs index %s"
                % (i, tuple(np.shape(p)[:-1]), iW))
        if shape is not None and np.shape(p)[-1] != shape[i]:
            raise ValueError(
                "perturbation P's mode dim must match the ambient dim Ni; mode %d: P has %d vs Ni=%d"
                % (i, np.shape(p)[-1], shape[i]))


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
        xis:    typ.Sequence[NDArray],  # frame projected probes,        len=d, elm_shape=W+C+(nUi,)
        dxis:   typ.Sequence[NDArray],  # projected perturbation dirs,  len=d, elm_shape=W+C+(nUi,)
) -> typ.Tuple[NDArray, ...]:           # xi_jets. len=d, elm_shape=(2,)+W+C+(nUi,): order 0 = xi, 1 = dxi
    '''Input jets: stack each (value, direction) pair on a leading order axis.

    Since ``x + s p`` is affine in ``s``, an input vector's jet is just ``(x, p, 0, ...)`` -- value at
    order 0, direction at order 1, zero above. Stored at size 2 (orders 0,1); the pushthrough slices
    the binomial tensor to ``s in {0,1}`` accordingly.
    '''
    use_jax = tree_contains_jax((xis, dxis))
    is_uniform = is_ndarray(xis)
    xnp, _, _ = get_backend(is_uniform, use_jax)
    if is_uniform:
        # Supercore (d,)+W+C+(nU,): stack (value, direction) at axis 1, AFTER the leading core axis d ->
        # (d,)+(2,)+W+C+(nU,), the uniform jet layout (d leads, then the order axis). NOT a per-core Python
        # loop over d (the unroll trap: it "works" by ragged-emulation but unrolls under jit + returns ragged
        # tuples that break the d-prefixed contractions downstream -- the same class as the _entry_xis bug).
        return xnp.stack([xis, dxis], axis=1)
    return tuple(xnp.stack([xi, dxi], axis=0) for xi, dxi in zip(xis, dxis))


def _init_jet(
        order:        int,                  # highest derivative order
        stack_shape:  typ.Tuple[int, ...],  # full leading batch the frame jet carries (sample + frame, W + C)
        r0:           int,                  # leftmost bond dimension
        xnp,                                # numpy or jax.numpy
) -> NDArray:                               # mu_0 jet, shape=(order+1,)+W+C+(r0,): order 0 = ones, higher = 0
    '''Leftmost left-pushthrough jet mu_0^(t): the empty product (ones) at order 0, zero above.'''
    ones  = xnp.ones((1,) + stack_shape + (r0,))
    zeros = xnp.zeros((order,) + stack_shape + (r0,))
    return xnp.concatenate([ones, zeros], axis=0)


def compute_mu_jets_trs(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        trs:        NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:               # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,). mu_jets[i][t]=mu_{i-1}^(t)
    '''Left derivative-pushthrough jets.

    Sweep left-to-right, at each core taking the binomial jet-product of the running left jet with the
    input jet through the core (``'trs,rWCa,Caib,sWCi->tWCb'``). Like :py:func:`probing.compute_mu`,
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
        return contractions.contract('trs,rWCa,Caib,sWCi->tWCb', trs_push, mu_jet, G, xi_jet[:s_size]), (mu_jet,)

    stack_shape = xi_jets[0].shape[1:-1]     # full W + C batch (W outer, C inner); either may be empty
    r0 = tt_cores[0].shape[-3]
    init = _init_jet(order, stack_shape, r0, xnp)

    _, (mu_jets,) = xscan(_func, init, (tt_cores, xi_jets))
    return mu_jets


# ==================================================================================================
# The STANDARD jet contractions -- convolution/recurrence/scan/chunk forms
# ==================================================================================================
# These are the default implementations wired into the sampling-derivative call sites; each has a dense
# `*_trs` twin (above/below) kept as a reference -- numerically equal to tolerance, occasionally faster
# in tiny / memory-abundant regimes, and the oracle in tests/test_jet_recurrence.py. The design question
# that motivates the recurrence form is dev/archive/OPEN_QUESTION_contractions_architecture_RESOLVED_2026-07-17.md.
#
# The idea: a trs binomial tensor is a sparse convolution tensor, so contracting it as a DENSE einsum
# operand is the wrong handling -- it is what makes _pairwise_path degenerate (the trs operand shares
# one index per operand, so it sorts LAST and the intermediate balloons to the union of all indices).
# The right form unrolls the convolution over the order axis into ordinary (non-trs) contractions.
#
# mu/nu pushthrough is the CLEAN case: the input jet is affine, so xi is nonzero only at orders s in
# {0,1} (build_input_jets returns size 2). The binomial sum then has just two surviving terms -- a
# bidiagonal recurrence, no trs tensor, no (order+1)^2 work:
#
#     mu_i^(t)  =  [mu^(t) . G . xi^(0)]  +  t * [mu^(t-1) . G . xi^(1)]
#
# (C(t,t)=1 gives the s=0 term; C(t,t-1)=t gives the s=1 term.) Verified equal to the dense trs
# contraction to 1e-16. eta/deta/tilde are the genuinely-FULL convolutions (nu is a full jet, not
# affine-truncated): the summed order axis is scanned instead (peak drops by the (order+1) factor), and
# the memory win lands on the uniform+jax path where xscan/xmap are real lax.scan/lax.map (sequential).


def compute_mu_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        trs:        NDArray,                # binomial tensor -- ONLY its shape (order) is read here
) -> typ.Tuple[NDArray, ...]:               # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,). mu_jets[i][t]=mu_{i-1}^(t)
    '''Left derivative-pushthrough jets (standard fused-recurrence form).

    ``mu_jets[i]`` is the left edge variable entering core ``i`` (``mu_{i-1}``), stacked over derivative
    orders. The affine input jet (``xi`` nonzero only at orders ``s in {0,1}``) collapses each binomial
    pushthrough to a two-term recurrence ``mu_i^(t) = mu^(t).G.xi^(0) + t * mu^(t-1).G.xi^(1)`` -- no
    dense ``trs`` tensor, no ``(order+1)^2`` work. The two terms are folded into ONE contraction per core:
    stack ``[mu^(t), t * mu^(t-1)]`` on a jet-pair axis ``s`` and contract it together with the bond ``a``
    against ``[G.xi^(0), G.xi^(1)]`` (``'stWCa,sWCab->tWCb'``) -- one larger GEMM for XLA to
    schedule turns the two-einsum form's ~parity with the dense ``trs`` into a win. Equal to the dense
    :py:func:`compute_mu_jets_trs` to tolerance; see it for the binomial-tensor reference form.
    '''
    use_jax = tree_contains_jax((tt_cores, xi_jets, trs))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    order = trs.shape[0] - 1
    s_size = min(2, order + 1)                             # affine input jet: orders {0, 1}
    tvec = xnp.arange(order + 1)                           # the C(t,t-1)=t multipliers

    def _func(mu_jet, data):
        G, xi_jet = data
        Gxi = contractions.contract('Caib,sWCi->sWCab', G, xi_jet[:s_size])     # (s, W, C, a, b), s in {0,1}
        if s_size > 1:                                     # static branch (order >= 1)
            t_bcast = tvec.reshape((order + 1,) + (1,) * (mu_jet.ndim - 1))
            shifted = xnp.concatenate([xnp.zeros_like(mu_jet[:1]), mu_jet[:-1]], axis=0)  # mu^(t-1)
            stacked_mu = xnp.stack([mu_jet, t_bcast * shifted], axis=0)   # (s=2,) + mu jet shape
            next_mu = contractions.contract('stWCa,sWCab->tWCb', stacked_mu, Gxi)
        else:                                              # order 0: only s=0 survives
            next_mu = contractions.contract('tWCa,WCab->tWCb', mu_jet, Gxi[0])
        return next_mu, (mu_jet,)

    stack_shape = xi_jets[0].shape[1:-1]
    r0 = tt_cores[0].shape[-3]
    init = _init_jet(order, stack_shape, r0, xnp)

    _, (mu_jets,) = xscan(_func, init, (tt_cores, xi_jets))
    return mu_jets


def compute_nu_jets_trs(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        trs:        NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:               # nu_jets. len=d, elm_shape=(order+1,)+W+C+(rR(i+1),). nu_jets[i][t]=nu_i^(t)
    '''Right derivative-pushthrough jets.

    The mirror image of :py:func:`compute_mu_jets_trs`: reverse the tensor train (``tt_reverse`` swaps
    bonds and core order), run the left sweep, reverse the result. ``nu_jets[i]`` is the right edge
    variable entering core ``i`` (``nu_i``), stacked over derivative orders.
    '''
    # Polymorphic reverse: the uniform tt_reverse keeps the supercore (tt_operations.tt_reverse would iterate
    # the supercore's d axis -- the unroll trap). The jet slices [::-1] just reverse the leading d axis.
    reverse = tt_operations.tt_reverse
    rev_nu_jets = compute_mu_jets_trs(reverse(tt_cores), xi_jets[::-1], trs)
    return rev_nu_jets[::-1]


def compute_nu_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        trs:        NDArray,                # binomial tensor -- only its shape (order) is read here
) -> typ.Tuple[NDArray, ...]:               # nu_jets. len=d, elm_shape=(order+1,)+W+C+(rR(i+1),). nu_jets[i][t]=nu_i^(t)
    '''Right derivative-pushthrough jets (standard recurrence form).

    The mirror image of :py:func:`compute_mu_jets`: reverse the tensor train (``tt_reverse`` swaps
    bonds and core order), run the left banded-recurrence sweep, reverse the result. ``nu_jets[i]`` is
    the right edge variable entering core ``i`` (``nu_i``), stacked over derivative orders. Equal to the
    dense :py:func:`compute_nu_jets_trs` to tolerance; see it for the binomial-tensor reference form.
    '''
    reverse = tt_operations.tt_reverse
    rev_nu_jets = compute_mu_jets(reverse(tt_cores), xi_jets[::-1], trs)
    return rev_nu_jets[::-1]


def compute_eta_jets_trs(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:        NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:               # eta_jets. len=d, elm_shape=(order+1,)+W+C+(nOi,). eta_jets[i][t]=eta_i^(t)
    '''Combine the left and right jets at each free mode via the binomial jet-product.

    ``eta_i^(t) = sum_{r+s=t} C(t,r) mu_{i-1}^(r) . G_i . nu_i^(s)``, one einsum per core
    (``'trs,rWCa,Caib,sWCb->tWCi'``) -- the same binomial convolution as the pushthrough, with the
    right jet on the bond and mode ``i`` left free.
    '''
    use_jax = tree_contains_jax((tt_cores, mu_jets, nu_jets, trs))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed jet combine (3b-6'a), vectorized over the core index d; the ragged xmap below is the
        # oracle. mu/nu jets are (d,)+(order,)+W+C+(r,); the tt supercore is (d,)+C+(rL,nO,rR) (C-only).
        eta_jets = contractions.contract('trs,drWCa,dCaib,dsWCb->dtWCi', trs, mu_jets, tt_cores, nu_jets)
    else:
        def _func(data):
            mu_jet, G, nu_jet = data
            return (contractions.contract('trs,rWCa,Caib,sWCb->tWCi', trs, mu_jet, G, nu_jet),)

        (eta_jets,) = xmap(_func, (mu_jets, tt_cores, nu_jets))
    return eta_jets


def compute_eta_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:        NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:               # eta_jets. len=d, elm_shape=(order+1,)+W+C+(nOi,)
    '''Combine the left and right jets at each free mode (standard order-scan form).

    Dense reference: :py:func:`compute_eta_jets_trs` (equal to tolerance).

    eta is a FULL convolution (both jets run ``0..order``), so the dense ``trs`` contraction forms the
    spatial product ``mu . G`` with mode AND bond live at every order at once: an ``(order+1)*W*r^2``
    intermediate (and the uniform d-einsum forms it for every core at once -- ``d`` times worse,
    128 GB at r=128, W=32000, order 5). This scans the two small axes instead, so only one slice is
    ever live:

        ``eta^(t) = sum_r C(t,r) [ mu^(r) . G ] . nu^(t-r)``

    - ``xscan`` over the input order ``r`` (the convolution): forms ``mu^(r) . G`` one order at a time
      and accumulates -- removes the ``(order+1)`` factor;
    - ``xmap`` over cores: removes the ``d`` factor.

    **The win requires the UNIFORM path** (a stacked supercore), where ``xmap``/``xscan`` dispatch to the
    real ``jax.lax.map`` / ``jax.lax.scan`` -- both *sequential* (``lax.map`` is a ``scan``, not a
    vectorizing ``vmap``), so only one core+order intermediate is resident. On the RAGGED path
    ``xmap``/``xscan`` are Python loops that unroll under jit and keep everything co-resident (measured
    ~1.2x, i.e. no win). Uniform, both scans: measured **14-28x** smaller XLA peak than the dense
    d-einsum and CONSTANT in order (~4.5 GB vs 64-128 GB at the huge config). The order loop being a
    real scan is what forces this -- an unrolled loop does not. Verified equal to
    :py:func:`compute_eta_jets_trs` to 1e-12 (ragged) and 1e-7 (uniform, float32).
    '''
    use_jax = tree_contains_jax((tt_cores, mu_jets, nu_jets, trs))
    xnp, xmap, xscan = get_backend(is_ndarray(tt_cores), use_jax)
    order = trs.shape[0] - 1
    trs_r = xnp.moveaxis(trs, 1, 0)                        # binomial CONSTANT (order+1)^3; scan leads on input order r -- const-folded, ~free

    def _func(data):
        mu_jet, G, nu_jet = data                          # (T,W,C,a) ; (C,a,i,b) ; (T,W,C,b)
        C_shape = G.shape[:-3]
        i = G.shape[-2]
        W_shape = mu_jet.shape[1:-(len(C_shape) + 1)]

        def _accumulate(eta, xr):
            mu_r, trsr = xr                                                        # (W,C,a) ; (t,s)
            MG_r = contractions.contract('WCa,Caib->WCib', mu_r, G)                # peak: W + C + (i, b)
            MGN_r = contractions.contract('WCib,sWCb->sWCi', MG_r, nu_jet)         # fold in all of nu -> order s
            return eta + contractions.contract('ts,sWCi->tWCi', trsr, MGN_r), ()   # binomial weights over t

        eta0 = xnp.zeros((order + 1,) + W_shape + C_shape + (i,), mu_jet.dtype)
        eta, _ = xscan(_accumulate, eta0, (mu_jet, trs_r))
        return (eta,)

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

    if is_uniform:
        # d-prefixed jet lift (3b-6'a); the ragged xmap below is the oracle. The order axis rides passively
        # through the C-only tucker supercore, keeping its own einsum letter (never folded into W).
        z_jets = contractions.contract('dtWCi,dCio->dtWCo', eta_jets, tucker_cores)
    else:
        def _func(data):
            eta_jet, U = data
            return (contractions.contract('tWCi,Cio->tWCo', eta_jet, U),)

        (z_jets,) = xmap(_func, (eta_jets, tucker_cores))
    return z_jets


###############################################################
####    Frame sweep (the reusable, variation-free jets)     ####
###############################################################
#
# The jet-ified twin of probing.precompute_{apply,probe}_frame_sweep: the frame edge-variable jets depend
# ONLY on the frame frame + the sample vectors (ww/pp or index/pp) + order -- NOT on the tangent direction
# or the residual. They are the expensive W-scaled part of every derivative Jacobian, shared by the
# forward (*_jacobian_derivatives_from_sweep) and the transpose (*_transpose_derivatives_from_sweep). A
# fitting inner solve precomputes them ONCE per frame (the SamplingKind.precompute hook) and reuses them
# across every J / J^T -- exactly as plain probing does.
#
# *** Per-kind, NOT shared. *** apply / entries are the all-modes special case: both their forward AND
# their adjoint-state transpose collapse to a single left pass, so they need ONLY the (xi, mu) jets --
# the right (nu) and central (eta) sweeps never enter the Lagrangian. Only the *probe* derivative (one
# free mode) uses all four. So apply/entries get a LEAN (xi, mu) precompute, probe gets the full
# (xi, mu, nu, eta) -- a real saving (this precompute runs per step in MC-SGD). Plain probing now does
# exactly the same split (apply.tv_precompute_apply_frame_sweep / _probe_frame_sweep, adjoint-state apply
# transpose) -- the regular and derivative apply/entries transposes are the same algorithm, order apart.


def _apply_frame_sweep_jets_from_xi(
        frame:    typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                      # = T3Frame.data = (U, O, P, Q)
        xi_jets: typ.Sequence[NDArray],         # up-index jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        order:   int,
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xi_jets. len=d, elm_shape=(2,)+W+C+(nUi,)
    typ.Sequence[NDArray],  # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,)
]:                                              # lean frame sweep -- (xi, mu) only (apply / entries)
    '''Run the lean frame sweep (only the left ``mu`` jets) from an already-formed ``xi_jets`` -- the
    apply/entries case (the adjoint-state forward + transpose use no ``nu`` / ``eta``). Shared tail of
    :py:func:`tv_precompute_apply_frame_sweep_jets` / :py:func:`tv_precompute_entries_frame_sweep_jets`.'''
    left_tt_cores = frame[2]
    mu_jets = compute_mu_jets(left_tt_cores, xi_jets, binomial_combine_tensor(order))
    return xi_jets, mu_jets


def _probe_frame_sweep_jets_from_xi(
        frame:    typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                      # = T3Frame.data = (U, O, P, Q)
        xi_jets: typ.Sequence[NDArray],         # up-index jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        order:   int,
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xi_jets.  len=d, elm_shape=(2,)+W+C+(nUi,)
    typ.Sequence[NDArray],  # mu_jets.  len=d, elm_shape=(order+1,)+W+C+(rLi,)
    typ.Sequence[NDArray],  # nu_jets.  len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
    typ.Sequence[NDArray],  # eta_jets. len=d, elm_shape=(order+1,)+W+C+(nOi,)
]:                                              # full frame sweep -- (xi, mu, nu, eta) (probe)
    '''Run the full frame mu/nu/eta jet sweeps from an already-formed ``xi_jets`` -- the probe case (one
    free mode needs the right ``nu`` sweep and central ``eta`` combine). The tail of
    :py:func:`tv_precompute_probe_frame_sweep_jets`.'''
    _, down_tt_cores, left_tt_cores, right_tt_cores = frame
    trs = binomial_combine_tensor(order)
    mu_jets  = compute_mu_jets(left_tt_cores, xi_jets, trs)
    nu_jets  = compute_nu_jets(right_tt_cores, xi_jets, trs)
    eta_jets = compute_eta_jets(down_tt_cores, mu_jets, nu_jets, trs)
    return xi_jets, mu_jets, nu_jets, eta_jets


def tv_precompute_apply_frame_sweep_jets(
        frame:   typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                      # = T3Frame.data = (U, O, P, Q)
        ww:     typ.Sequence[NDArray],          # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:     typ.Sequence[NDArray],          # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:  int,                            # highest derivative order
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xi_jets. len=d, elm_shape=(2,)+W+C+(nUi,)
    typ.Sequence[NDArray],  # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,)
]:                                              # lean frame sweep -- (xi, mu) only
    '''The **apply**-derivative frame sweep (lean): the up-index jet ``build_input_jets(U_i x_i, U_i p_i)``
    + the left ``mu`` jets. The all-modes apply forward and its adjoint-state transpose use ONLY
    ``(xi, mu)`` (no ``nu`` / ``eta``), so this skips the right + central sweeps -- a per-step saving in
    a stochastic solve. Reused across the forward / transpose of an inner solve (the
    :py:class:`SamplingKind.precompute` hook).'''
    up_tucker_cores = frame[0]
    xi_jets = build_input_jets(compute_xi(up_tucker_cores, ww), compute_xi(up_tucker_cores, pp))
    return _apply_frame_sweep_jets_from_xi(frame, xi_jets, order)


def tv_precompute_entries_frame_sweep_jets(
        frame:   typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                      # = T3Frame.data = (U, O, P, Q)
        index:  NDArray,                        # int, shape=(d,)+W -- the grid points
        pp:     typ.Sequence[NDArray],          # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:  int,                            # highest derivative order
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xi_jets. len=d, elm_shape=(2,)+W+C+(nUi,)
    typ.Sequence[NDArray],  # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,)
]:                                              # lean frame sweep -- (xi, mu) only
    '''The **entries**-derivative frame sweep (lean): like :py:func:`tv_precompute_apply_frame_sweep_jets`
    but the up-index jet is formed by fiber-slicing the Tucker cores at ``index`` (order 0) + contracting
    ``P`` (order 1), so the variation gradient scatters onto the indexed rows. Also ``(xi, mu)`` only.'''
    up_tucker_cores = frame[0]
    xi_jets = build_input_jets(_entry_xis(up_tucker_cores, index), compute_xi(up_tucker_cores, pp))
    return _apply_frame_sweep_jets_from_xi(frame, xi_jets, order)


def tv_precompute_probe_frame_sweep_jets(
        frame:   typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                      # = T3Frame.data = (U, O, P, Q)
        ww:     typ.Sequence[NDArray],          # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:     typ.Sequence[NDArray],          # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        order:  int,                            # highest derivative order
) -> typ.Tuple[
    typ.Sequence[NDArray],  # xi_jets.  len=d, elm_shape=(2,)+W+C+(nUi,)
    typ.Sequence[NDArray],  # mu_jets.  len=d, elm_shape=(order+1,)+W+C+(rLi,)
    typ.Sequence[NDArray],  # nu_jets.  len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
    typ.Sequence[NDArray],  # eta_jets. len=d, elm_shape=(order+1,)+W+C+(nOi,)
]:                                              # full frame sweep -- (xi, mu, nu, eta)
    '''The **probe**-derivative frame sweep (full): the jet-ified twin of
    :py:func:`t3toolbox.backend.probing.tv_precompute_probe_frame_sweep`. The probe leaves one mode free, so it
    needs all four frame edge-variable jets ``(xi, mu, nu, eta)`` (the right ``nu`` sweep + central
    ``eta`` combine). Reused across the forward / transpose of an inner solve.'''
    up_tucker_cores = frame[0]
    xi_jets = build_input_jets(compute_xi(up_tucker_cores, ww), compute_xi(up_tucker_cores, pp))
    return _probe_frame_sweep_jets_from_xi(frame, xi_jets, order)


###############################################################
####    Riemannian: symmetric derivatives of a tangent     ####
###############################################################
#
# tv_probe (probing.py) is the action of a tangent vector v on the probe vectors: a frame sweep
# (xi, mu, nu, eta via the frame cores U, O, P, Q) plus a variation sweep (dxi, sigma, tau, deta via
# the variation cores dU, dG), then z = U.deta + dU.eta. Differentiating that map w.r.t. the probe
# vectors (the same direction P repeated) jet-ifies every edge variable -- and since every term of
# every recursion is itself a pushthrough, combine, or lift, each maps onto the order-threaded
# contractions with the appropriate (frame or variation) core. The order axis t is uniform across
# ALL edge variables (everything depends on the probe vectors). The frame sweep (xi, mu, nu, eta) uses
# the 2-block (W,C) order-threaded contractions; the variation sweep (dxi, sigma, tau, deta) carries
# the tangent stack K, so it uses the order-threaded THREE-block (W,K,C) contractions, exactly as
# tv_probe's perturbation sweep does (K on the variation cores; n_frame = len(C) where the only
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
    compute_sigma_jets_trs (keeps the per-core sequence) and tv_apply_derivatives (keeps only the
    terminal carry). Three-group (W sample, K tangent, C frame): sigma/dxi carry K, the frame edge vars
    (xi, mu) and frame cores (Q, O) do not; t2's only core is the variation core dG (K+C), so len(C) is
    supplied via n_frame (recovered from the C-only Q). Reduces to the 2-group result when K=().'''
    s_size = trs_push.shape[2]                # input jets carry orders {0, 1}, capped at order
    n_frame = Q.ndim - 3
    t1 = contractions.contract('trs,rWKCa,Caib,sWCi->tWKCb', trs_push, sigma_jet, Q,  xi_jet[:s_size])
    t2 = contractions.contract('trs,rWCa,KCaib,sWCi->tWKCb', trs_push, mu_jet,    dG, xi_jet[:s_size], len_C=n_frame)
    t3 = contractions.contract('trs,rWCa,Caib,sWKCi->tWKCb', trs_push, mu_jet,    O,  dxi_jet[:s_size])
    return t1 + t2 + t3


def compute_sigma_jets_trs(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        mu_jets:        typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # sigma_jets. len=d, elm_shape=(order+1,)+W+K+C+(rRi,)
    '''Variation-leftward edge-variable jets sigma (the jet-ified Algorithm-7 sigma recursion).

    ``sigma_i = sigma_{i-1} Q_i(xi_i) + mu_{i-1} dG_i(xi_i) + mu_{i-1} O_i(dxi_i)`` -- three
    pushthroughs (``'trs,rWCa,Caib,sWCi->tWCb'``): the carried sigma jet through Q, and the frame mu
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


def compute_tau_jets_trs(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        nu_jets:        typ.Sequence[NDArray],  # frame right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # tau_jets. len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
    '''Variation-rightward edge-variable jets tau -- the mirror of :py:func:`compute_sigma_jets_trs`.

    Reverse the train (P in the Q-slot, O and dG reversed), run the sigma sweep, reverse the result.
    '''
    reverse = tt_operations.tt_reverse if is_ndarray(var_tt_cores) else tt_operations.tt_reverse
    rev = compute_sigma_jets_trs(
        reverse(var_tt_cores), reverse(left_tt_cores),
        reverse(down_tt_cores), xi_jets[::-1], dxi_jets[::-1], nu_jets[::-1], trs,
    )
    return rev[::-1]


def compute_deta_jets_trs(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_jets:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rRi,)
        tau_jets:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # deta_jets. len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
    '''Variation-downward edge-variable jets deta (the jet-ified Algorithm-7 deta combine).

    ``deta_i = sigma_{i-1} Q_i nu_i + mu_{i-1} dG_i nu_i + mu_{i-1} P_i tau_i`` -- three combines
    (``'trs,rWCa,Caib,sWCb->tWCi'``), mode ``i`` free.
    '''
    use_jax = tree_contains_jax((var_tt_cores, left_tt_cores, right_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets, trs))
    is_uniform = is_ndarray(var_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed jet combines (3b-6'a), vectorized over d; the ragged xmap below is the oracle. sigma/tau
        # jets carry K (W+K+C); mu/nu are W+C; P/Q are C-only supercores; term2's only core is dG (K+C), so
        # len(C) = n_frame, read off the C-only Q supercore (d,)+C+(rR,nU,rR).
        n_frame = right_tt_cores.ndim - 4
        term1 = contractions.contract('trs,drWKCa,dCaib,dsWCb->dtWKCi', trs, sigma_jets, right_tt_cores, nu_jets)
        term2 = contractions.contract('trs,drWCa,dKCaib,dsWCb->dtWKCi', trs, mu_jets, var_tt_cores, nu_jets, len_C=n_frame)
        term3 = contractions.contract('trs,drWCa,dCaib,dsWKCb->dtWKCi', trs, mu_jets, left_tt_cores, tau_jets)
        deta_jets = term1 + term2 + term3
    else:
        def _func(data):
            P, Q, dG, mu_jet, nu_jet, sigma_jet, tau_jet = data
            # Three-group (W,K,C): sigma/tau carry K, mu/nu and frame cores P/Q do not; term2's only core
            # is the variation core dG (K+C), so len(C) is supplied via n_frame (the C-only Q pins it).
            n_frame = Q.ndim - 3
            term1 = contractions.contract('trs,rWKCa,Caib,sWCb->tWKCi', trs, sigma_jet, Q,  nu_jet)
            term2 = contractions.contract('trs,rWCa,KCaib,sWCb->tWKCi', trs, mu_jet,    dG, nu_jet, len_C=n_frame)
            term3 = contractions.contract('trs,rWCa,Caib,sWKCb->tWKCi', trs, mu_jet,    P,  tau_jet)
            return (term1 + term2 + term3,)

        (deta_jets,) = xmap(_func, (left_tt_cores, right_tt_cores, var_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets))
    return deta_jets


def compute_deta_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_jets:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rRi,)
        tau_jets:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # deta_jets. len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
    '''Tangent combine at each free mode (standard order-scan form).

    Dense reference: :py:func:`compute_deta_jets_trs` (equal to tolerance).

    The three-term tangent analog of :py:func:`compute_eta_jets`: ``deta_i = sigma Q nu +
    mu dG nu + mu P tau``, three FULL convolutions (mode ``i`` free). The dense uniform d-einsum forms
    the ``(order+1)*W*K*r^2`` spatial product for every core at once; this scans the input order ``r``
    (one order slice of ``jetL . core`` live at a time) and folds in all of the right jet + the
    binomial weights -- peak ``W*K*r^2``. The K tangent stack sits on a *different* operand in each
    term (sigma / dG / tau), so several of the grouped contractions need ``len_C`` (the W|C split
    is not pinned by their operands). Memory win needs
    the uniform path (real ``lax.map``/``lax.scan``); see :py:func:`compute_eta_jets`. Verified
    equal to :py:func:`compute_deta_jets_trs` to 1e-12.
    '''
    use_jax = tree_contains_jax((var_tt_cores, left_tt_cores, right_tt_cores,
                                 mu_jets, nu_jets, sigma_jets, tau_jets, trs))
    xnp, xmap, xscan = get_backend(is_ndarray(var_tt_cores), use_jax)
    order = trs.shape[0] - 1
    trs_r = xnp.moveaxis(trs, 1, 0)                   # binomial CONSTANT (order+1)^3; scan leads on left order r -- const-folded, ~free

    def _func(data):
        P, Q, dG, mu_jet, nu_jet, sigma_jet, tau_jet = data
        C_shape = Q.shape[:-3]                        # Q is C+(a,i,b)
        nf = len(C_shape)
        i = Q.shape[-2]                               # mode nU (shared, free); bonds differ per term
        nW = mu_jet.ndim - 2 - nf                     # mu is (order+1,)+W+C+(a,)
        W_shape = mu_jet.shape[1:1 + nW]
        nK = (sigma_jet.ndim - 2) - nW - nf           # sigma is (order+1,)+W+K+C+(a,)
        K_shape = sigma_jet.shape[1 + nW:1 + nW + nK]

        def _step(eta, xr):
            mu_r, sig_r, trsr = xr                                                    # (W,C,a) ; (W,K,C,a) ; (t,s)
            mg1 = contractions.contract('WKCa,Caib->WKCib', sig_r, Q)                 # term1: sigma Q nu  (K on sigma)
            mgn1 = contractions.contract('WKCib,sWCb->sWKCi', mg1, nu_jet, len_C=nf)
            mg2 = contractions.contract('WCa,KCaib->WKCib', mu_r, dG, len_C=nf)       # term2: mu dG nu    (K on core)
            mgn2 = contractions.contract('WKCib,sWCb->sWKCi', mg2, nu_jet, len_C=nf)
            mg3 = contractions.contract('WCa,Caib->WCib', mu_r, P)                    # term3: mu P tau    (K on tau)
            mgn3 = contractions.contract('WCib,sWKCb->sWKCi', mg3, tau_jet, len_C=nf)
            contrib = contractions.contract('ts,sWKCi->tWKCi', trsr, mgn1 + mgn2 + mgn3)
            return eta + contrib, ()

        eta0 = xnp.zeros((order + 1,) + W_shape + K_shape + C_shape + (i,), mu_jet.dtype)
        eta, _ = xscan(_step, eta0, (mu_jet, sigma_jet, trs_r))
        return (eta,)

    (deta_jets,) = xmap(_func, (left_tt_cores, right_tt_cores, var_tt_cores,
                                mu_jets, nu_jets, sigma_jets, tau_jets))
    return deta_jets


def _sigma_banded_step(sigma_jet, Q, O, dG, xi_jet, dxi_jet, mu_jet, s_size, tvec, order, xnp):
    '''One K-aware banded step of the sigma recursion: the three affine pushthroughs of
    :py:func:`_sigma_jet_step` as fused two-term recurrences (no trs). W, K, C ride unflattened
    through the grouped contractions (``len_C`` supplied where the W|C split is unpinned); K rides
    on the carried sigma (t1), the variation core dG (t2), or the var input dxi (t3).'''
    nf = len(Q.shape[:-3])
    xi_s, dxi_s = xi_jet[:s_size], dxi_jet[:s_size]

    Qxi = contractions.contract('Caib,sWCi->sWCab', Q, xi_s)                 # (s,W,C,a,b)   -- no K
    dGxi = contractions.contract('KCaib,sWCi->sWKCab', dG, xi_s, len_C=nf)   # (s,W,K,C,a,b) -- K on core
    Odxi = contractions.contract('Caib,sWKCi->sWKCab', O, dxi_s)             # (s,W,K,C,a,b) -- K on var input

    def stacked(jet):       # (order+1,...,a) -> (s=2,)+... = [jet^(t), t*jet^(t-1)]
        t_b = tvec.reshape((order + 1,) + (1,) * (jet.ndim - 1))
        shifted = xnp.concatenate([xnp.zeros_like(jet[:1]), jet[:-1]], axis=0)
        return xnp.stack([jet, t_b * shifted], axis=0)

    if s_size > 1:
        t1 = contractions.contract('stWKCa,sWCab->tWKCb', stacked(sigma_jet), Qxi, len_C=nf)   # K on sigma
        t2 = contractions.contract('stWCa,sWKCab->tWKCb', stacked(mu_jet), dGxi, len_C=nf)     # K on core
        t3 = contractions.contract('stWCa,sWKCab->tWKCb', stacked(mu_jet), Odxi, len_C=nf)     # K on var input
    else:                                                                    # order 0: only s=0
        t1 = contractions.contract('tWKCa,WCab->tWKCb', sigma_jet, Qxi[0], len_C=nf)
        t2 = contractions.contract('tWCa,WKCab->tWKCb', mu_jet, dGxi[0], len_C=nf)
        t3 = contractions.contract('tWCa,WKCab->tWKCb', mu_jet, Odxi[0], len_C=nf)
    return t1 + t2 + t3


def compute_sigma_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        mu_jets:        typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        trs:            NDArray,                # binomial tensor -- ONLY its shape (order) is read here
) -> typ.Tuple[NDArray, ...]:                   # sigma_jets. len=d, elm_shape=(order+1,)+W+K+C+(rRi,)
    '''Variation-leftward edge-variable jets sigma (standard banded-recurrence form).

    Dense reference: :py:func:`compute_sigma_jets_trs` (equal to tolerance).

    The three-pushthrough tangent analog of :py:func:`compute_mu_jets`. Both input jets
    (``xi``, ``dxi``) are affine (size 2), so each of the three pushthroughs
    (``sigma Q(xi) + mu dG(xi) + mu O(dxi)``) is a fused two-term recurrence rather than a dense ``trs``
    contraction. The K tangent stack rides on the carried sigma / the variation core / the var input.
    Verified equal to :py:func:`compute_sigma_jets_trs` to 1e-12.
    '''
    use_jax = tree_contains_jax((var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets, trs))
    xnp, xmap, xscan = get_backend(is_ndarray(var_tt_cores), use_jax)
    order = trs.shape[0] - 1
    s_size = min(2, order + 1)
    tvec = xnp.arange(order + 1)

    def _func(sigma_jet, data):
        Q, O, dG, xi_jet, dxi_jet, mu_jet = data
        return _sigma_banded_step(sigma_jet, Q, O, dG, xi_jet, dxi_jet, mu_jet,
                                  s_size, tvec, order, xnp), (sigma_jet,)

    stack_shape = dxi_jets[0].shape[1:-1]
    rR0 = right_tt_cores[0].shape[-3]
    init = _zero_jet(order, stack_shape, rR0, xnp)
    _, (sigma_jets,) = xscan(_func, init, (right_tt_cores, down_tt_cores, var_tt_cores, xi_jets, dxi_jets, mu_jets))
    return sigma_jets


def compute_tau_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        nu_jets:        typ.Sequence[NDArray],  # frame right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor -- ONLY its shape (order) is read here
) -> typ.Tuple[NDArray, ...]:                   # tau_jets. len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
    '''Variation-rightward edge-variable jets tau (standard banded-recurrence form) -- sigma-banded on
    the reversed train. Dense reference: :py:func:`compute_tau_jets_trs` (equal to tolerance).'''
    reverse = tt_operations.tt_reverse
    rev = compute_sigma_jets(
        reverse(var_tt_cores), reverse(left_tt_cores),
        reverse(down_tt_cores), xi_jets[::-1], dxi_jets[::-1], nu_jets[::-1], trs,
    )
    return rev[::-1]


def assemble_tangent_z_jets(
        tucker_cores:       typ.Sequence[NDArray],  # U.  len=d, elm_shape=C+(nUi,Ni)
        var_tucker_cores:   typ.Sequence[NDArray],  # dU. len=d, elm_shape=K+C+(nOi,Ni)
        eta_jets:           typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(nOi,)
        deta_jets:          typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
) -> typ.Tuple[NDArray, ...]:                       # z_jets. len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
    '''Assemble tangent-probe-derivative jets: ``z_i = U_i deta_i + dU_i eta_i`` -- two lifts
    (``tWCi_Cio_to_tWCo``), the order axis riding as a leading broadcast batch.
    '''
    use_jax = tree_contains_jax((tucker_cores, var_tucker_cores, eta_jets, deta_jets))
    is_uniform = is_ndarray(tucker_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed jet lifts (3b-6'a); the ragged xmap below is the oracle. deta carries K (lifted via the
        # C-only U supercore, where W and K ride passively); eta is W+C lifted through the K+C variation core dU,
        # so the eta-lift needs len(C) = n_frame, read off the C-only U supercore (d,)+C+(nU,N).
        n_frame = tucker_cores.ndim - 3
        term1 = contractions.contract('dtWKCi,dCio->dtWKCo', deta_jets, tucker_cores)
        term2 = contractions.contract('dtWCi,dKCio->dtWKCo', eta_jets, var_tucker_cores, len_C=n_frame)
        z_jets = term1 + term2
    else:
        def _func(data):
            U, dU, eta_jet, deta_jet = data
            # Three-group (W,K,C): deta carries K (in the lift via the C-only U, W and K ride passively); eta is W+C and dU is
            # the variation core (K+C), so the eta-lift needs len(C) -- recovered from the C-only U.
            n_frame = U.ndim - 2
            return (contractions.contract('tWKCi,Cio->tWKCo', deta_jet, U)
                    + contractions.contract('tWCi,KCio->tWKCo', eta_jet, dU, len_C=n_frame),)

        (z_jets,) = xmap(_func, (tucker_cores, var_tucker_cores, eta_jets, deta_jets))
    return z_jets


def tv_probe_jacobian_derivatives_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores dU. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores     dG. len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        sweep:      typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order)
        order:      int,                    # highest derivative order K
) -> typ.Tuple[NDArray, ...]:               # z_jets. len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
    '''Variation half of :py:func:`tv_probe_derivatives` from a precomputed frame ``sweep``: the
    variation sweep (sigma/tau/deta jets via the variation cores) + the lift, reusing the frame
    ``(xi, mu, nu, eta)_jets``. The reuse hook for a fitting inner solve (frame fixed across J / J^T).'''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    var_tucker_cores, var_tt_cores = variation
    xi_jets, mu_jets, nu_jets, eta_jets = sweep
    trs = binomial_combine_tensor(order)

    dxi_jets = build_input_jets(compute_xi(var_tucker_cores, ww), compute_xi(var_tucker_cores, pp))
    sigma_jets = compute_sigma_jets(var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets, trs)
    tau_jets   = compute_tau_jets(var_tt_cores, left_tt_cores, down_tt_cores, xi_jets, dxi_jets, nu_jets, trs)
    deta_jets  = compute_deta_jets(var_tt_cores, left_tt_cores, right_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets, trs)
    return assemble_tangent_z_jets(up_tucker_cores, var_tucker_cores, eta_jets, deta_jets)


def tv_probe_derivatives(
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores dU. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores     dG. len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data
        frame:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # = T3Frame.data = (up, down, left, right) = (U, O, P, Q)
        order:      int,                    # highest derivative order K
) -> typ.Tuple[NDArray, ...]:               # z_jets. len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
    '''Symmetric derivatives of probing a tangent vector, in one repeated direction (Riemannian J^(s)).

    The probe-derivative analog of :py:func:`probing.tv_probe`: returns, for each mode ``i``, the
    stack ``y_i^(t) = d^t/ds^t [J^(s) v]_i (X + s P)|_0`` for ``t=0..K``, where ``v`` is the tangent
    vector represented by ``(frame, variation)``. Index ``0`` is the ordinary tangent probe.

    Identical structure to :py:func:`t3_probe_derivatives` but on the tangent calculus: a frame sweep
    (frame cores) and a variation sweep. Full ``order + W + K + C`` stacking (base-inner, mirroring
    tv_probe): sample stack ``W``, tangent stack ``K`` (a batch of tangents sharing the frame, on
    the variation cores), frame stack ``C``; any may be empty.

    The symmetric-derivative formulation is not in the published T4S paper; verified against
    :py:func:`dense_probe_derivatives` on the densified tangent. (Project write-up in preparation.)

    See Also
    --------
    t3_probe_derivatives
    compute_sigma_jets
    compute_tau_jets
    compute_deta_jets
    assemble_tangent_z_jets
    t3toolbox.backend.probing.tv_probe
    '''
    sweep = tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order)
    return tv_probe_jacobian_derivatives_from_sweep(variation, ww, pp, frame, sweep, order)


###############################################################
####    Apply / entries derivatives (all-modes special)    ####
###############################################################
#
# apply and entries are the all-modes special case of probing (probing leaves ONE mode free; these
# contract EVERY mode). As in probing.py, with no free mode the computation collapses to a single
# left-to-right pass: the frame left sweep mu (via P/G) feeds the perturbation sweep sigma (via Q),
# kept only to its TERMINAL carry and contracted at the terminal bond -- no right (nu) / central (eta)
# sweeps, no per-mode assembly. The order axis t and the W/K/C stacks ride exactly as in probing's
# apply (here apply derivatives output order + W + K + C, a scalar-jet per stack element). entries is
# apply with the up-index xis from slicing Tucker-core fibers (deferred to its own step).


def _apply_derivatives_t3_from_xi_jets(xi_jets, tt_cores, trs):
    '''Terminal mu-jet carry of the left sweep (via the cores), bond summed -- the all-modes Euclidean
    tail shared by t3_apply_derivatives and t3_entries_derivatives (they differ only in how xi_jets is
    formed). Returns ``(order+1,) + W + C`` (no K for a plain T3).
    '''
    use_jax = tree_contains_jax((tt_cores, xi_jets, trs))
    xnp, xmap, xscan = get_backend(is_ndarray(tt_cores), use_jax)   # scan-style: xscan strips d, trs_* runs per-slice
    order = trs.shape[0] - 1
    s_size = min(2, order + 1)
    trs_push = trs[:, :, :s_size]

    def _func(mu_jet, data):
        G, xi_jet = data
        return contractions.contract('trs,rWCa,Caib,sWCi->tWCb', trs_push, mu_jet, G, xi_jet[:s_size]), (0,)

    stack_shape = xi_jets[0].shape[1:-1]                # W + C
    r0 = tt_cores[0].shape[-3]
    init = _init_jet(order, stack_shape, r0, xnp)
    mu_terminal, _ = xscan(_func, init, (tt_cores, xi_jets))
    return xnp.sum(mu_terminal, axis=-1)               # contract the terminal bond -> (order+1,)+W+C


def t3_apply_derivatives(
        ww:     typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        x:      typ.Tuple[
            typ.Sequence[NDArray],      # tucker_cores. len=d, elm_shape=C+(nUi,Ni)
            typ.Sequence[NDArray],      # tt_cores.     len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        ],                              # = TuckerTensorTrain.data
        order:  int,                    # highest derivative order
) -> NDArray:                           # apply-derivative jets, shape=(order+1,)+W+C
    '''Symmetric derivatives of applying a Tucker tensor train in all modes, in one repeated direction.

    The all-modes Euclidean analog of :py:func:`t3_probe_derivatives`: returns the stack
    ``y^(t) = d^t/ds^t apply(X, W + s P)|_0`` for ``t=0..order``, where ``apply`` contracts the tensor
    with the vectors in every mode (a scalar). Index ``0`` is the ordinary apply. Computed as the
    terminal carry of the left mu-jet sweep (via the cores), bond summed -- no right/central sweeps.
    Stacks ride as ``order + W + C`` (sample stack ``W``, frame stack ``C``; no tangent stack here).
    Verified against :py:func:`dense_apply_derivatives`.
    '''
    tucker_cores, tt_cores = x
    xi_jets = build_input_jets(compute_xi(tucker_cores, ww), compute_xi(tucker_cores, pp))
    return _apply_derivatives_t3_from_xi_jets(xi_jets, tt_cores, binomial_combine_tensor(order))


def _apply_derivatives_from_jets(
        xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores, trs,
):
    '''Run the K-aware perturbation sigma-jet sweep to its TERMINAL carry and contract the final bond.

    The jet analog of :py:func:`apply._apply_from_xis`: reuses :py:func:`_sigma_jet_step` (so it is
    `W+K+C`-stacked) but keeps only the terminal carry. Returns ``(order+1,) + W + K + C``.
    '''
    use_jax = tree_contains_jax((xi_jets, dxi_jets, mu_jets, right_tt_cores, trs))
    xnp, xmap, xscan = get_backend(is_ndarray(right_tt_cores), use_jax)  # scan-style: xscan strips d, per-slice trs_*
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


def tv_apply_jacobian_derivatives_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores dU. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores     dG. len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        sweep:      typ.Tuple[
            typ.Sequence[NDArray],          # xi_jets
            typ.Sequence[NDArray],          # mu_jets
        ],                                  # = tv_precompute_apply_frame_sweep_jets(frame, ww, pp, order)
        order:      int,                    # highest derivative order
) -> NDArray:                               # apply-derivative jets, shape=(order+1,)+W+K+C
    '''Variation half of :py:func:`tv_apply_derivatives` from a precomputed frame ``sweep``: the
    variation input jets + the terminal sigma carry, reusing the frame ``(xi, mu)_jets`` (apply needs no
    ``nu`` / ``eta``). The reuse hook for a fitting inner solve (frame fixed across J / J^T).'''
    _, down_tt_cores, _, right_tt_cores = frame
    var_tucker_cores, var_tt_cores = variation
    xi_jets, mu_jets = sweep
    trs = binomial_combine_tensor(order)
    dxi_jets = build_input_jets(compute_xi(var_tucker_cores, ww), compute_xi(var_tucker_cores, pp))
    return _apply_derivatives_from_jets(
        xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores, trs)


def tv_apply_derivatives(
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores dU. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores     dG. len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data
        frame:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # = T3Frame.data = (up, down, left, right) = (U, O, P, Q)
        order:      int,                    # highest derivative order
) -> NDArray:                               # apply-derivative jets, shape=(order+1,)+W+K+C
    '''Symmetric derivatives of applying a tangent vector in all modes, in one repeated direction.

    The all-modes Riemannian analog of :py:func:`tv_probe_derivatives` (and the derivative analog
    of :py:func:`apply.tv_apply`): returns ``y^(t) = d^t/ds^t [apply(v, W + s P)]|_0`` for
    ``t=0..order``, where ``v`` is the tangent vector ``(frame, variation)``. A single left-to-right pass
    (frame mu via P, perturbation sigma via Q) to the terminal carry, bond summed. Stacks ``order + W +
    K + C`` (sample stack ``W``, tangent stack ``K``, frame stack ``C``). Verified against
    :py:func:`dense_apply_derivatives` on the densified tangent.

    See Also
    --------
    t3_apply_derivatives
    tv_probe_derivatives
    apply.tv_apply
    '''
    sweep = tv_precompute_apply_frame_sweep_jets(frame, ww, pp, order)
    return tv_apply_jacobian_derivatives_from_sweep(variation, ww, pp, frame, sweep, order)


def t3_entries_derivatives(
        index:  NDArray,                # int, shape=(d,)+W -- the grid points (one multi-index per W sample)
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        x:      typ.Tuple[
            typ.Sequence[NDArray],      # tucker_cores. len=d, elm_shape=C+(nUi,Ni)
            typ.Sequence[NDArray],      # tt_cores.     len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        ],                              # = TuckerTensorTrain.data
        order:  int,                    # highest derivative order
) -> NDArray:                           # entries-derivative jets, shape=(order+1,)+W+C
    '''Symmetric derivatives of an entry of a Tucker tensor train, in one repeated direction.

    The ``entries`` analog of :py:func:`t3_apply_derivatives` -- apply-derivatives with the up-index
    frame jet from slicing Tucker-core fibers at ``index`` (order 0) and contracting ``P`` (order 1).
    Returns ``y^(t) = d^t/ds^t apply(X, e_{index} + s P)|_0`` for ``t=0..order``: the Taylor data of
    the tensor's multilinear extension at grid corner ``index``, in direction ``P``. Index ``0`` is the
    ordinary entry ``X[index]``. Stacks ``order + W + C``. Verified vs :py:func:`dense_entries_derivatives`.
    '''
    tucker_cores, tt_cores = x
    xi_jets = build_input_jets(_entry_xis(tucker_cores, index), compute_xi(tucker_cores, pp))
    return _apply_derivatives_t3_from_xi_jets(xi_jets, tt_cores, binomial_combine_tensor(order))


def tv_entries_jacobian_derivatives_from_sweep(
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores dU. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores     dG. len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data
        index:      NDArray,                # int, shape=(d,)+W -- the grid points
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        sweep:      typ.Tuple[
            typ.Sequence[NDArray],          # xi_jets
            typ.Sequence[NDArray],          # mu_jets
        ],                                  # = tv_precompute_entries_frame_sweep_jets(frame, index, pp, order)
        order:      int,                    # highest derivative order
) -> NDArray:                               # entries-derivative jets, shape=(order+1,)+W+K+C
    '''Variation half of :py:func:`tv_entries_derivatives` from a precomputed frame ``sweep``:
    the entries analog of :py:func:`tv_apply_jacobian_derivatives_from_sweep` (variation up-index jet from
    fiber-slicing at ``index`` + ``P``), reusing the frame ``(xi, mu)_jets``.'''
    _, down_tt_cores, _, right_tt_cores = frame
    var_tucker_cores, var_tt_cores = variation
    xi_jets, mu_jets = sweep
    trs = binomial_combine_tensor(order)
    dxi_jets = build_input_jets(_entry_xis(var_tucker_cores, index), compute_xi(var_tucker_cores, pp))
    return _apply_derivatives_from_jets(
        xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores, trs)


def tv_entries_derivatives(
        index:      NDArray,                # int, shape=(d,)+W -- the grid points
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        variation:  typ.Tuple[
            typ.Sequence[NDArray],          # var_tucker_cores dU. len=d, elm_shape=K+C+(nOi,Ni)
            typ.Sequence[NDArray],          # var_tt_cores     dG. len=d, elm_shape=K+C+(rLi,nUi,rRi)
        ],                                  # = T3Variations.data
        frame:       typ.Tuple[
            typ.Sequence[NDArray],          # up_tucker_cores  U. len=d
            typ.Sequence[NDArray],          # down_tt_cores    O. len=d
            typ.Sequence[NDArray],          # left_tt_cores    P. len=d
            typ.Sequence[NDArray],          # right_tt_cores   Q. len=d
        ],                                  # = T3Frame.data = (up, down, left, right) = (U, O, P, Q)
        order:      int,                    # highest derivative order
) -> NDArray:                               # entries-derivative jets, shape=(order+1,)+W+K+C
    '''Symmetric derivatives of an entry of a tangent vector, in one repeated direction.

    The ``entries`` analog of :py:func:`tv_apply_derivatives` -- identical but with the frame/var
    up-index jets from slicing Tucker-core fibers at ``index`` (order 0) + contracting ``P`` (order 1).
    Stacks ``order + W + K + C``. Verified vs :py:func:`dense_entries_derivatives` on the densified tangent.

    See Also
    --------
    t3_entries_derivatives
    tv_apply_derivatives
    entries.tv_entries
    '''
    sweep = tv_precompute_entries_frame_sweep_jets(frame, index, pp, order)
    return tv_entries_jacobian_derivatives_from_sweep(variation, index, pp, frame, sweep, order)


###############################################################
####    Riemannian transpose (the jet-ified adjoint)       ####
###############################################################
#
# The transpose of tv_probe_derivatives (linear in the variation): residual jets r -> variation
# gradient (dU_tilde, dG_tilde). Derived as the jet-ified adjoint-state Lagrangian (t4s.pdf Thm 7):
# every forward contraction in the Lagrangian is replaced by its trs version, and stationarity
# d L / d (state, variation) = 0 gives the adjoint sweeps and the gradient assembly -- which is exactly
# the verified non-derivative transpose (probing.compute_*_tildes / assemble_*) with each contraction
# swapped for its ADJOINT-HOOKED trs version (same trs tensor, transposed legs: the multiplier's order
# is summed, the swept order is freed) for the sweeps, and the ORDER-LESS trs (TT, 3 edges) / plain
# order-sum (Tucker, 1 edge) for the assembly. Stacks (base-inner W + C, as in plain probing): sample
# stack W on the inputs, frame stack C on the cores; sum_over_probes sums W (the J^T r back-projection)
# else keeps it (W rides into the variation stack). Verified against jax.linear_transpose + the
# adjoint identity. The named contractions live in contractions.py (the *_to_sWCb / *_to_uWCi sweeps,
# the order-less *_to_[W]Caib / *_to_[W]Cao assembly).

def compute_deta_tilde_jets(
        up_tucker_cores:    typ.Sequence[NDArray],  # U.  len=d, elm_shape=C+(nUi,Ni)
        ztildes:            typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
) -> typ.Tuple[NDArray, ...]:                       # deta_tildes. len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
    '''Adjoint-up edge-variable jets: ``deta_tilde_i = U_i r_i`` (contract the ambient mode, order
    diagonal). The 1-internal-edge (Tucker) case -- the order axis just rides through (no trs). The
    residual carries the tangent stack K (the forward output's K), which rides through.'''
    use_jax = tree_contains_jax((up_tucker_cores, ztildes))
    is_uniform = is_ndarray(up_tucker_cores)
    xnp, xmap, _ = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed adjoint lift (3b-6'a); the ragged xmap is the oracle. The order axis rides passively
        # through the C-only U supercore, keeping its own einsum letter (t, W and K are never folded).
        deta_tildes = contractions.contract('dtWKCo,dCio->dtWKCi', ztildes, up_tucker_cores)
    else:
        def _func(data):
            U, zt = data
            return (contractions.contract('tWKCo,Cio->tWKCi', zt, U),)

        (deta_tildes,) = xmap(_func, (up_tucker_cores, ztildes))
    return deta_tildes


def _adj_sweep(P_cores, xi_jets, deta_tildes, edge_jets, trs):
    '''The jet adjoint sweep shared by compute_tau_tilde_jets_trs / compute_sigma_tilde_jets_trs: a left-to-right scan
    (mirroring probing.compute_tau_tilde) of the adjoint-hooked pushthrough (propagation) plus the
    deta_tilde source. Both terms are the same trs, wired as the transpose (output at the swept order s).'''
    use_jax = tree_contains_jax((P_cores, xi_jets, deta_tildes, edge_jets, trs))
    xnp, xmap, xscan = get_backend(is_ndarray(P_cores), use_jax)  # scan-style: xscan strips d, per-slice trs_*
    s_size = min(2, trs.shape[0])
    trs_xi = trs[:, :s_size, :]            # input jet (xi) carries orders {0, 1}

    def _step(carry, data):
        P, xi, deta_t, edge = data
        # Three-group (W,K,C): the swept adjoint (carry) and deta_tilde carry K; xi/edge (frame) and P
        # (frame core) do not. Both terms self-infer the split (xi pins W, P pins C, K=remainder).
        prop = contractions.contract('trs,tWKCa,Caib,rWCi->sWKCb', trs_xi, carry, P, xi[:s_size])  # propagation
        src  = contractions.contract('trs,rWCa,Caib,tWKCi->sWKCb', trs,    edge,  P, deta_t)       # deta_tilde source
        return prop + src, (carry,)

    # carry is W+K+C; its leading stack comes from deta_tildes (which carry K), so the init carries K
    rL0 = P_cores[0].shape[-3]
    init = xnp.zeros((trs.shape[0],) + deta_tildes[0].shape[1:-1] + (rL0,))
    _, (tildes,) = xscan(_step, init, (P_cores, xi_jets, deta_tildes, edge_jets))
    return tildes


def compute_tau_tilde_jets_trs(
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        mu_jets:        typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # tau_tildes. len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
    '''Adjoint-var-rightward edge-variable jets (jet-ified probing.compute_tau_tilde).'''
    return _adj_sweep(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)


def compute_sigma_tilde_jets_trs(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        nu_jets:        typ.Sequence[NDArray],  # frame right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # sigma_tildes. len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
    '''Adjoint-var-leftward edge-variable jets -- the mirror (reverse) of compute_tau_tilde_jets_trs.'''
    reverse = tt_operations.tt_reverse if is_ndarray(right_tt_cores) else tt_operations.tt_reverse
    rev = _adj_sweep(reverse(right_tt_cores), xi_jets[::-1],
                     deta_tildes[::-1], nu_jets[::-1], trs)
    return rev[::-1]


def _adj_tilde_step(carry, P, xi, deta_t, edge, s_size, svec, order, xnp, xscan, trs_r):
    '''One K-aware step of the memory-lean adjoint sweep. Two terms, both the TRANSPOSE of a forward
    contraction: **prop** (adjoint of the affine pushthrough) is a two-term REVERSE recurrence -- shifted
    UP (``carry^(s+1)``, weight ``s+1``), fused into one GEMM; **src** (adjoint of the full combine, the
    deta_tilde source) is a full REVERSE convolution, an inner order-scan over the edge order ``r`` with
    peak ``W*r^2`` per slice. W, K, C ride unflattened through the grouped contractions (carry/deta
    carry K; xi/edge/P do not; ``len_C`` supplied where the W|C split is unpinned).'''
    nf = len(P.shape[:-3])
    b = P.shape[-1]
    nW = xi.ndim - 2 - nf
    W_shape = xi.shape[1:1 + nW]
    K_shape = carry.shape[1 + nW:carry.ndim - 1 - nf]
    C_shape = P.shape[:-3]

    # --- prop: two-term reverse recurrence (xi affine). prop^(s) = carry^(s) P xi^(0) + (s+1) carry^(s+1) P xi^(1)
    Pxi = contractions.contract('Caib,kWCi->kWCab', P, xi[:s_size])   # (k=s_size, W, C, a, b)
    if s_size > 1:
        sp1 = (svec + 1).reshape((order + 1,) + (1,) * (carry.ndim - 1))
        up = xnp.concatenate([carry[1:], xnp.zeros_like(carry[:1])], axis=0)   # carry^(s+1)
        stacked = xnp.stack([carry, sp1 * up], axis=0)                # (2,)+carry shape
        prop = contractions.contract('ksWKCa,kWCab->sWKCb', stacked, Pxi, len_C=nf)
    else:
        prop = contractions.contract('sWKCa,WCab->sWKCb', carry, Pxi[0], len_C=nf)

    # --- src: full reverse convolution -- inner scan over the edge order r, peak W*r^2 per slice
    def _src_step(acc, xr):
        edge_r, trsr = xr                                             # (W,C,a) ; (t,s)
        ep = contractions.contract('WCa,Caib->WCib', edge_r, P)       # edge^(r) P over a -- peak W*r^2
        epd = contractions.contract('WCib,tWKCi->tWKCb', ep, deta_t, len_C=nf)   # fold all t (deta full)
        return acc + contractions.contract('ts,tWKCb->sWKCb', trsr, epd), ()

    src0 = xnp.zeros((order + 1,) + W_shape + K_shape + C_shape + (b,), carry.dtype)
    src, _ = xscan(_src_step, src0, (edge, trs_r))

    return prop + src


def _adj_sweep_scanned(P_cores, xi_jets, deta_tildes, edge_jets, trs):
    '''EXPERIMENTAL memory-lean mirror of :py:func:`_adj_sweep` (module-private). Same left-to-right core
    sweep, but each step's two `trs` einsums become the TRANSPOSE of the forward recurrence/convolution:
    a two-term reverse recurrence (prop, affine xi) + an inner order-scan (src, full). The inner scan
    is real (`lax.scan` on the uniform path) so its `W*r^2` slice is the peak, not `(order+1)*W*r^2`.'''
    use_jax = tree_contains_jax((P_cores, xi_jets, deta_tildes, edge_jets, trs))
    xnp, xmap, xscan = get_backend(is_ndarray(P_cores), use_jax)
    order = trs.shape[0] - 1
    s_size = min(2, order + 1)
    svec = xnp.arange(order + 1)
    trs_r = xnp.moveaxis(trs, 1, 0)                             # binomial CONSTANT (order+1)^3; inner scan leads on edge order r -- const-folded, ~free

    def _step(carry, data):
        P, xi, deta_t, edge = data
        new = _adj_tilde_step(carry, P, xi, deta_t, edge, s_size, svec, order, xnp, xscan, trs_r)
        return new, (carry,)

    rL0 = P_cores[0].shape[-3]
    init = xnp.zeros((order + 1,) + deta_tildes[0].shape[1:-1] + (rL0,))
    _, (tildes,) = xscan(_step, init, (P_cores, xi_jets, deta_tildes, edge_jets))
    return tildes


def compute_tau_tilde_jets(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs):
    '''Adjoint-var-rightward edge-variable jets tau_tilde (standard order-scan form). Dense reference:
    :py:func:`compute_tau_tilde_jets_trs` (equal to tolerance).'''
    return _adj_sweep_scanned(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)


def compute_sigma_tilde_jets(right_tt_cores, xi_jets, deta_tildes, nu_jets, trs):
    '''Adjoint-var-leftward edge-variable jets sigma_tilde (standard order-scan form; reverse of
    tau_tilde). Dense reference: :py:func:`compute_sigma_tilde_jets_trs` (equal to tolerance).'''
    rev = _adj_sweep_scanned(tt_operations.tt_reverse(right_tt_cores), xi_jets[::-1],
                             deta_tildes[::-1], nu_jets[::-1], trs)
    return rev[::-1]


def compute_dxi_tilde_jets(
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # dxi_tildes. len=d, elm_shape=(order+1,)+W+K+C+(nOi,)
    '''Adjoint-var-down edge-variable jets (jet-ified probing.compute_dxi_tilde): two adjoint-hooked
    combines giving delta-xi-tilde on the mode (output at the order-<=1 leg u).'''
    use_jax = tree_contains_jax((down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes, trs))
    is_uniform = is_ndarray(down_tt_cores)
    xnp, xmap, _ = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed adjoint combines (3b-6'a); the ragged xmap is the oracle. tau/sigma_tilde carry K; mu/nu
        # are W+C; O is the C-only down supercore. Both self-infer the W/K/C split.
        from_tau = contractions.contract('trs,dtWKCa,dCaib,dsWCb->drWKCi', trs, tau_tildes, down_tt_cores, nu_jets)
        from_sig = contractions.contract('trs,drWCa,dCaib,dtWKCb->dsWKCi', trs, mu_jets, down_tt_cores, sigma_tildes)
        dxi_tildes = from_tau + from_sig
    else:
        def _func(data):
            O, mu, nu, st, tt = data
            # Three-group (W,K,C): tau_tilde (tt) / sigma_tilde (st) carry K, mu/nu (frame) and O (frame core)
            # do not. Both self-infer (mu/nu pin W, O pins C, K=remainder).
            from_tau = contractions.contract('trs,tWKCa,Caib,sWCb->rWKCi', trs, tt, O, nu)
            from_sig = contractions.contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu, O, st)
            return (from_tau + from_sig,)

        (dxi_tildes,) = xmap(_func, (down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes))
    return dxi_tildes


def assemble_tucker_variation_jets_trs(
        ztildes:        typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
        dxi_tildes:     typ.Sequence[NDArray],  # adjoint-var-down jets, len=d, elm_shape=(order+1,)+W+K+C+(nOi,)
        ww:             typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:             typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        etas:           typ.Sequence[NDArray],  # frame down jets, len=d, elm_shape=(order+1,)+W+C+(nOi,)
        n_probe:        int,                    # number of sample-stack (W) axes
        sum_over_probes: bool,
) -> typ.Tuple[NDArray, ...]:                   # dU_tildes. len=d, elm_shape=[W+]C+(nOi,Ni)
    '''Assemble Tucker-core variation gradients (the 1-edge, plain-order-sum case):
    ``dU_tilde = sum_t eta^(t) (x) r^(t) + sum_u dxi_tilde^(u) (x) w_jet^(u)``.'''
    use_jax = tree_contains_jax((ztildes, dxi_tildes, ww, pp, etas))
    is_uniform = is_ndarray(etas)
    _, xmap, _ = get_backend(is_uniform, use_jax)
    w_jets = build_input_jets(ww, pp)          # ambient input jets (value, direction) for dU
    s_size = min(2, etas[0].shape[0])          # the w/dxi input jet carries orders {0, 1}, capped at order
    # Three-group (W,K,C): the residual-derived operands (ztilde, dxi_tilde) carry K, eta is frame; the
    # eta (x) r term takes n_probe (C from eta), the dxi (x) w_jet term self-pins W from the W-only w_jet.
    if is_uniform:
        # d-prefixed assembly (3b-6'a); the order axis is at supercore axis 1, so the w/dxi order-slice is
        # [:, :s_size] (NOT [:s_size], which would slice the leading core axis d). Ragged xmap is the oracle.
        eta_r = 'dtWCa,dtWKCo->dKCao' if sum_over_probes else 'dtWCa,dtWKCo->dWKCao'
        dxi_w = 'duWKCa,duWo->dKCao' if sum_over_probes else 'duWKCa,duWo->dWKCao'
        return (contractions.contract(eta_r, etas, ztildes, len_W=n_probe)
                + contractions.contract(dxi_w, dxi_tildes[:, :s_size], w_jets[:, :s_size]))

    eta_r = 'tWCa,tWKCo->KCao' if sum_over_probes else 'tWCa,tWKCo->WKCao'
    dxi_w = 'uWKCa,uWo->KCao' if sum_over_probes else 'uWKCa,uWo->WKCao'

    def _func(data):
        zt, dxt, eta, wj = data
        return (contractions.contract(eta_r, eta, zt, len_W=n_probe)
                + contractions.contract(dxi_w, dxt[:s_size], wj[:s_size]),)

    (dU_tildes,) = xmap(_func, (ztildes, dxi_tildes, etas, w_jets))
    return dU_tildes


def assemble_tucker_variation_jets(
        ztildes, dxi_tildes, ww, pp, etas, n_probe, sum_over_probes,
        chunk_size: typ.Optional[int] = 100,   # W-chunk size; None (or >= W) -> dense. See docs/chunking.md
) -> NDArray:
    '''Assemble the Tucker variation gradient (standard W-chunked form; dense reference
    :py:func:`assemble_tucker_variation_jets_trs`). The milder assembly (two legs nO,N -- the dense
    gradient is a few GB, not the tt-core's hundreds), chunked over W with the same reducer seam (add if
    summed, concat if kept) for a uniform chunked-assembly interface. The only twist: ``ww``/``pp`` have
    no order axis, so W sits at a per-operand axis (2 for the order-carrying jets, 1 for ww/pp).
    ``chunk_size`` None / ragged / multi-W / small W -> runs the dense assembly directly (no chunking).'''
    ops = (ztildes, dxi_tildes, ww, pp, etas)
    w_axes = (2, 2, 1, 1, 2)                          # per-operand W axis (ww/pp carry no order axis)
    dense = lambda: assemble_tucker_variation_jets_trs(*ops, n_probe, sum_over_probes)
    if chunk_size is None or not is_ndarray(etas) or n_probe != 1:
        return dense()
    use_jax = tree_contains_jax(ops)
    xnp, _, _ = get_backend(True, use_jax)
    W = etas.shape[2]
    if W <= chunk_size:
        return dense()
    return _wchunked_reduce(
        lambda co: assemble_tucker_variation_jets_trs(*co, n_probe, sum_over_probes),
        ops, w_axes, W, chunk_size, sum_over_probes, 1, use_jax, xnp)


def assemble_tt_variation_jets_trs(
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
        deta_tildes:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
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
    is_uniform = is_ndarray(xi_jets)
    xnp, xmap, _ = get_backend(is_uniform, use_jax)
    s_size = min(2, trs.shape[0])
    # Three-group (W,K,C): the residual-derived adjoint vars carry K on the assembled core's leg --
    # sigma_tilde on b (f_sig), tau_tilde on a (f_tau), deta_tilde on i (f_det); the frame xi/mu/nu do
    # not. K kept always; W summed (sum_over_probes -> KCaib) or kept (WKCaib). n_probe = len(W).
    if is_uniform:
        # d-prefixed dG assembly (3b-6'a); the xi order-slice is [:, :s_size] (order at supercore axis 1).
        # trs is shared (no d), so its slices are unchanged. Ragged xmap is the oracle.
        if sum_over_probes:
            f_sig, f_tau, f_det = ('trs,drWCa,dsWCi,dtWKCb->dKCaib',
                                   'trs,dtWKCa,drWCi,dsWCb->dKCaib',
                                   'trs,drWCa,dtWKCi,dsWCb->dKCaib')
        else:
            f_sig, f_tau, f_det = ('trs,drWCa,dsWCi,dtWKCb->dWKCaib',
                                   'trs,dtWKCa,drWCi,dsWCb->dWKCaib',
                                   'trs,drWCa,dtWKCi,dsWCb->dWKCaib')
        t_sig = contractions.contract(f_sig, trs[:, :, :s_size], mu_jets, xi_jets[:, :s_size], sigma_tildes, len_W=n_probe)
        t_tau = contractions.contract(f_tau, trs[:, :s_size, :], tau_tildes, xi_jets[:, :s_size], nu_jets, len_W=n_probe)
        t_det = contractions.contract(f_det, trs,                mu_jets, deta_tildes,           nu_jets, len_W=n_probe)
        return t_sig + t_tau + t_det

    if sum_over_probes:
        f_sig, f_tau, f_det = ('trs,rWCa,sWCi,tWKCb->KCaib',
                               'trs,tWKCa,rWCi,sWCb->KCaib',
                               'trs,rWCa,tWKCi,sWCb->KCaib')
    else:
        f_sig, f_tau, f_det = ('trs,rWCa,sWCi,tWKCb->WKCaib',
                               'trs,tWKCa,rWCi,sWCb->WKCaib',
                               'trs,rWCa,tWKCi,sWCb->WKCaib')

    def _func(data):
        xi, mu, nu, st, tt, dt = data
        t_sig = contractions.contract(f_sig, trs[:, :, :s_size], mu, xi[:s_size], st, len_W=n_probe)
        t_tau = contractions.contract(f_tau, trs[:, :s_size, :], tt, xi[:s_size], nu, len_W=n_probe)
        t_det = contractions.contract(f_det, trs,                mu, dt,           nu, len_W=n_probe)
        return (t_sig + t_tau + t_det,)

    (dG_tildes,) = xmap(_func, (xi_jets, mu_jets, nu_jets, sigma_tildes, tau_tildes, deta_tildes))
    return dG_tildes


def _slice_w(op, wax, start, size, use_dyn):
    '''One W-chunk of ``op`` at axis ``wax`` -- a dynamic slice on jax (traced start), a static slice on
    numpy. Both are views/fusible: NO copy of the operand (unlike pad+reshape+moveaxis).'''
    if use_dyn:
        import jax
        return jax.lax.dynamic_slice_in_dim(op, start, size, axis=wax)
    idx = [slice(None)] * op.ndim
    idx[wax] = slice(start, start + size)
    return op[tuple(idx)]


def _wchunked_reduce(assemble_one, ops, w_axes, W, cs, summed, out_w_axis, use_jax, xnp):
    '''Copy-free W-chunking. Extract each chunk in place and reduce: ADD if W is summed (the gradient)
    or CONCAT along ``out_w_axis`` if W is kept. No pad, no reshape, no transpose of the operands (which
    is what made the old approach duplicate the N-large residual). On jax the summed reduction is a real
    ``lax.scan`` over the full chunks (dynamic slices) so only one chunk is resident; the remainder is a
    static slice; chunk 0 seeds the accumulator (and its shape). On numpy it is an eager loop (which
    frees each chunk anyway).'''
    def _chunk(start, size, dyn):
        return tuple(_slice_w(op, wax, start, size, dyn) for op, wax in zip(ops, w_axes))
    n_full, rem = divmod(W, cs)
    if summed:
        acc = assemble_one(_chunk(0, cs, False))                      # chunk 0: init + output shape
        if use_jax and n_full > 1:
            import jax
            import jax.numpy as jnp

            def _step(a, i):
                return a + assemble_one(tuple(jax.lax.dynamic_slice_in_dim(op, i * cs, cs, wax)
                                              for op, wax in zip(ops, w_axes))), 0
            acc, _ = jax.lax.scan(_step, acc, jnp.arange(1, n_full))
        elif n_full > 1:
            for k in range(1, n_full):
                acc = acc + assemble_one(_chunk(k * cs, cs, False))
        if rem:
            acc = acc + assemble_one(_chunk(n_full * cs, rem, False))
        return acc
    parts = [assemble_one(_chunk(st, min(cs, W - st), False)) for st in range(0, W, cs)]
    return xnp.concatenate(parts, axis=out_w_axis)


def assemble_tt_variation_jets(
        sigma_tildes:   typ.Sequence[NDArray],
        tau_tildes:     typ.Sequence[NDArray],
        deta_tildes:    typ.Sequence[NDArray],
        xi_jets:        typ.Sequence[NDArray],
        mu_jets:        typ.Sequence[NDArray],
        nu_jets:        typ.Sequence[NDArray],
        trs:            NDArray,
        n_probe:        int,
        sum_over_probes: bool,
        chunk_size:     typ.Optional[int] = 100,    # W-chunk size; None (or >= W) -> dense. See docs/chunking.md
) -> NDArray:                                        # dG_tildes supercore, [W+]C+(rLi,nUi,rRi)
    '''Assemble the TT variation gradient (standard W-chunked form; dense reference
    :py:func:`assemble_tt_variation_jets_trs`).

    The dense assembly's peak is exactly LINEAR in the sample stack W (measured: ~5.3 MB / W-row at
    r=128), so it is chunked along W: the dense assembly runs per W-chunk and the partials are combined.
    Peak ~ ``chunk_size * (per-W-row)`` instead of ``W * (per-W-row)``.

    **The reducer is the seam that keeps this from locking into W-only** (Nick, 2026-07-16). A chunked
    batch axis is combined by ADD if it is *summed* (``sum_over_probes`` -> the gradient) or by CONCAT if
    it is *kept* (the ``sum_over_probes=False`` per-probe output, and -- later -- a chunked *frame* stack
    C, which is always kept). The per-chunk assembler is axis-agnostic (the dense assembly), so extending
    to C-chunking is a new slice front + the same reducer, not a rewrite.

    Chunking runs on the uniform path with a single W axis; ragged / multi-W / ``chunk_size`` unset /
    ``W <= chunk_size`` fall back to the dense assembly. The chunk map is a real ``lax.map`` (sequential),
    so only one chunk's intermediate is resident.
    '''
    ops = (sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets)
    dense = lambda: assemble_tt_variation_jets_trs(*ops, trs, n_probe, sum_over_probes)
    if chunk_size is None or not is_ndarray(xi_jets) or n_probe != 1:
        return dense()

    use_jax = tree_contains_jax(ops + (trs,))
    xnp, _, _ = get_backend(True, use_jax)
    W = xi_jets.shape[2]                              # uniform supercore: (d, order, W, ...)
    if W <= chunk_size:
        return dense()
    return _wchunked_reduce(
        lambda co: assemble_tt_variation_jets_trs(*co, trs, n_probe, sum_over_probes),
        ops, (2,) * len(ops), W, chunk_size, sum_over_probes, 1, use_jax, xnp)


# ==================================================================================================
# Choosing chunk_size -- an eager (outside-jit) estimator from the problem shapes
# ==================================================================================================
# The transpose functions take a plain-int chunk_size with a safe fixed default. To pick a *tuned*
# value, call these once outside jit and pass the int down. Two policies (docs/chunking.md):
#   estimate_chunk_size   -- BALANCED: assembly peak ~ the resident edge-jet memory (device-agnostic).
#   max_chunk_size_within -- BUDGET:   the largest chunk whose assembly peak fits an absolute byte cap.
# Both size the LARGER of the two gradient assemblies (TT-core r^2 legs vs Tucker nO/N legs; the latter
# dominates when N >> n), and divide W by n_shards so the chunk sizes each device's shard (docs/chunking.md
# sharding section). Uniform+jax only -- numpy/ragged free eagerly and never chunk.


def _jet_floor_bytes(mode_shapes, tucker_ranks, tt_ranks, order, W, n_tangent, itemsize):
    '''Exact bytes of the resident edge-variable jets the transpose assembly reads -- the necessary
    memory floor (linear in W). Uniform supercores pad ranks to their max, so max ranks are used; the
    tangent stack K rides on the tilde jets and the residual only.'''
    d, op, K = len(mode_shapes), order + 1, n_tangent
    r, nU, N = max(tt_ranks), max(tucker_ranks), max(mode_shapes)
    nO = nU                                         # down-core mode ~ the Tucker rank
    per_core_row = (K * op * (r + r + nU + nO + N)   # sigma_t, tau_t, deta_t, dxi_t, ztildes (carry K)
                    + 2 * nU + op * (r + r + nO)      # xi, mu, nu, eta (no K)
                    + 2 * N)                          # ww, pp
    return d * W * per_core_row * itemsize


@functools.lru_cache(maxsize=None)
def _assembly_per_row_bytes(mode_shapes, tucker_ranks, tt_ranks, order, n_tangent, dtype):
    '''Measured peak scratch (bytes) per W-row of the dense gradient assembly, the MAX over the TT-core
    and Tucker assemblies. Measured, not derived: XLA fuses away most of the naive ``(order+1)^2 r^2``
    intermediate, so an analytic formula is ~20x off. ``memory_analysis().temp_size`` is exactly linear
    in W, so two abstract (``ShapeDtypeStruct``) lowerings isolate the per-row slope with no allocation.
    jax-only (the chunking path is uniform+jax). Cached by shape signature.'''
    import jax
    d, op = len(mode_shapes), order + 1
    r, nU, N = max(tt_ranks), max(tucker_ranks), max(mode_shapes)
    nO = nU
    kax = (n_tangent,) if n_tangent > 1 else ()      # K axis present only when batching tangents

    def _structs(W):
        S = lambda *shape: jax.ShapeDtypeStruct(shape, dtype)
        tt = (S(d, op, W, *kax, r), S(d, op, W, *kax, r), S(d, op, W, *kax, nU),   # sigma_t, tau_t, deta_t
              S(d, 2, W, nU), S(d, op, W, r), S(d, op, W, r), S(op, op, op))         # xi, mu, nu, trs
        tk = (S(d, op, W, *kax, N), S(d, op, W, *kax, nO),                          # ztildes, dxi_t
              S(d, W, N), S(d, W, N), S(d, op, W, nO))                               # ww, pp, eta
        return tt, tk

    def _temp(f, args):
        return jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes

    f_tt = lambda *a: assemble_tt_variation_jets_trs(*a, 1, True)
    f_tk = lambda *a: assemble_tucker_variation_jets_trs(*a, 1, True)
    (tt1, tk1), (tt2, tk2) = _structs(256), _structs(512)
    per_tt = (_temp(f_tt, tt2) - _temp(f_tt, tt1)) / 256.0
    per_tk = (_temp(f_tk, tk2) - _temp(f_tk, tk1)) / 256.0
    return max(per_tt, per_tk, 1.0)


def estimate_chunk_size(
        mode_shapes:    typ.Sequence[int],  # (N_1..N_d)   ambient dims (as passed to TuckerTensorTrain.randn)
        tucker_ranks:   typ.Sequence[int],  # (nU_1..nU_d) Tucker ranks
        tt_ranks:       typ.Sequence[int],  # (r_0..r_d)   the d+1 TT bonds
        order:          int,                # highest derivative order K
        n_probes:       int,                # |W|, the number of probes (global; divided by n_shards)
        *,
        n_tangent:      int = 1,            # tangent stack K (a batch of tangents sharing the frame)
        n_shards:       int = 1,            # W split across this many devices -> chunk sizes the LOCAL shard
        dtype:          typ.Any = None,     # array dtype; default float32 (jax default). float64 under x64
) -> int:                                   # a memory-balanced chunk_size for tv_probe_derivatives_transpose
    '''A memory-**balanced** ``chunk_size`` for the probe-derivative transpose assembly, from the problem
    shapes -- call it once (eagerly, outside ``jit``) and pass the int as ``chunk_size``.

    Picks the largest chunk whose assembly peak is comparable to the edge-variable jets already resident
    (the necessary floor), so the assembly is never the tallest pole -- if the rest of the pipeline fits,
    so does the assembly (total peak ``~2x`` the floor). Device-agnostic: no device-memory query, only the
    shapes. The assembly per-row cost is **measured** via XLA's own scratch accounting (needs jax; a
    ~2 s one-time compile, cached by shape). For an absolute byte cap instead, see
    :py:func:`max_chunk_size_within`; the memory model is in the chunking design note.

    Sized for the larger of the TT-core and Tucker gradients (Tucker wins when ``N >> n``). ``n_shards``
    divides ``W`` so the chunk sizes each device's shard (use ``shard_map``; see the chunking note). Only
    meaningful on the uniform+jax path -- numpy/ragged never chunk.
    '''
    dtype = np.dtype(np.float32 if dtype is None else dtype)
    ms, tr, tt = tuple(mode_shapes), tuple(tucker_ranks), tuple(tt_ranks)
    w_local = max(1, int(n_probes) // max(1, int(n_shards)))
    budget = _jet_floor_bytes(ms, tr, tt, order, w_local, n_tangent, dtype.itemsize)
    per_row = _assembly_per_row_bytes(ms, tr, tt, order, n_tangent, dtype)
    return max(1, min(w_local, int(budget // per_row)))


def max_chunk_size_within(
        mode_shapes:    typ.Sequence[int],  # (N_1..N_d)   ambient dims
        tucker_ranks:   typ.Sequence[int],  # (nU_1..nU_d) Tucker ranks
        tt_ranks:       typ.Sequence[int],  # (r_0..r_d)   the d+1 TT bonds
        order:          int,                # highest derivative order K
        n_probes:       int,                # |W| (global; divided by n_shards)
        target_bytes:   float,              # absolute peak-memory cap for the assembly (per device)
        *,
        n_tangent:      int = 1,
        n_shards:       int = 1,
        dtype:          typ.Any = None,
) -> int:                                   # the largest chunk_size whose assembly peak fits target_bytes
    '''The largest ``chunk_size`` whose gradient-assembly peak stays within ``target_bytes`` (per device)
    -- the "use my whole device" policy: pass an absolute byte budget (e.g. a fraction of device memory).
    Contrast the device-agnostic :py:func:`estimate_chunk_size`. Same measured per-row cost.'''
    dtype = np.dtype(np.float32 if dtype is None else dtype)
    ms, tr, tt = tuple(mode_shapes), tuple(tucker_ranks), tuple(tt_ranks)
    w_local = max(1, int(n_probes) // max(1, int(n_shards)))
    per_row = _assembly_per_row_bytes(ms, tr, tt, order, n_tangent, dtype)
    return max(1, min(w_local, int(float(target_bytes) // per_row)))


def tv_probe_transpose_derivatives_from_sweep(
        ztildes:    typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        sweep:      typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order)
        order:      int,                    # highest derivative order K
        sum_over_probes: bool = False,      # True: sum the sample stack W (the J^T r back-projection)
        chunk_size: typ.Optional[int] = 100,   # W-chunk size for the gradient assembly; None -> dense. docs/chunking.md
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes (Tucker variation gradient)
    typ.Tuple[NDArray, ...],  # dG_tildes (TT variation gradient)
]:                                          # = T3Variations.data
    '''Variation gradient of :py:func:`tv_probe_derivatives_transpose` from a precomputed frame
    ``sweep``: the adjoint sweeps (``sigma/tau/dxi_tilde`` jets) + the order-less gradient assembly,
    reusing the frame ``(xi, mu, nu, eta)_jets``. The reuse hook for a fitting inner solve.'''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    xi_jets, mu_jets, nu_jets, eta_jets = sweep
    trs = binomial_combine_tensor(order)
    n_probe = ww[0].ndim - 1   # number of sample-stack (W) axes (ww carries W, no C)

    deta_tildes  = compute_deta_tilde_jets(up_tucker_cores, ztildes)
    tau_tildes   = compute_tau_tilde_jets(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)
    sigma_tildes = compute_sigma_tilde_jets(right_tt_cores, xi_jets, deta_tildes, nu_jets, trs)
    dxi_tildes   = compute_dxi_tilde_jets(down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes, trs)

    dU_tildes = assemble_tucker_variation_jets(
        ztildes, dxi_tildes, ww, pp, eta_jets, n_probe, sum_over_probes, chunk_size=chunk_size)
    dG_tildes = assemble_tt_variation_jets(
        sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets, trs, n_probe, sum_over_probes,
        chunk_size=chunk_size)
    return dU_tildes, dG_tildes


def tv_probe_derivatives_transpose(
        ztildes:    typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        order:      int,                    # highest derivative order K
        sum_over_probes: bool = False,      # True: sum the sample stack W (the J^T r back-projection)
        chunk_size: typ.Optional[int] = 100,   # W-chunk size for the gradient assembly; None -> dense. docs/chunking.md
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes (Tucker variation gradient)
    typ.Tuple[NDArray, ...],  # dG_tildes (TT variation gradient)
]:                                          # = T3Variations.data
    '''Transpose of :py:func:`tv_probe_derivatives`: back-project residual jets ``ztildes`` into a
    variation gradient ``(dU_tildes, dG_tildes)``. The jet-ified adjoint-state method (t4s.pdf Thm 7):
    every forward contraction is swapped for its ``trs`` version, then stationarity of the Lagrangian
    gives the adjoint sweeps (``sigma/tau/dxi_tilde`` jets) and the order-less gradient assembly.

    Full ``W + K + C`` stacking, base-inner as in plain probing: the residual jets ``ztildes`` carry the
    sample stack ``W``, the tangent stack ``K`` (the forward output's ``K``), and the frame stack ``C``.
    The tangent batch ``K`` always rides through to the variation gradient. With ``sum_over_probes=False``
    the sample stack ``W`` rides through into the variation stack too; with ``True`` it is summed (the
    ``J^T r`` back-projection used for fitting), ``K``/``C`` always kept. Verified against the dense
    adjoint identity ``<r, J v> = <J^T r, v>`` and ``jax.linear_transpose``.

    ``chunk_size`` bounds the peak memory of the (uniform+jax) gradient assembly by processing the
    sample stack ``W`` in slices of that size; ``None`` (or ``>= W``) runs the dense assembly. The
    default is a safe fixed value, tuned only for moderate problems (a fixed ``chunk_size`` bounds the
    chunk *count*, not the bytes); for a memory-balanced value from the problem shapes call
    :py:func:`estimate_chunk_size` once (eagerly). Chunking engages only on the uniform+jax path
    (numpy/ragged free eagerly). See :doc:`/chunking`.
    '''
    sweep = tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order)
    return tv_probe_transpose_derivatives_from_sweep(
        ztildes, ww, pp, frame, sweep, order, sum_over_probes, chunk_size=chunk_size)


###############################################################
####    Apply / entries derivatives transpose (adjoint)    ####
###############################################################
#
# The all-modes special case of the probe transpose. The forward apply/entries derivative is the
# terminal carry of the perturbation sigma sweep (no nu/eta/per-mode assembly), so its transpose is
# the jet-ified adjoint-state Lagrangian with the per-mode residual gone: the scalar residual jet c
# SEEDS one propagation sweep at the terminal bond (compute_sigma_hat_jets) -- a propagation-only
# adjoint sweep via Q, no deta_tilde source. The gradient is then dG_tilde = mu (x) xi (x) sigma_hat
# and dU_tilde = dxi_hat (x) w_jet, with dxi_hat = mu * O * sigma_hat. Half the work of the probe
# transpose (one adjoint sweep, single-term assembly, no nu/eta). Reuses the order-threaded 3-block
# adjoint contractions; full W + K + C, base-inner. Verified vs the adjoint identity + jax.linear_transpose.


def compute_sigma_hat_jets(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        c:              NDArray,                # residual jet (scalar), shape=(order+1)+W+K+C
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # sigma_hats. len=d, elm_shape=(order+1)+W+K+C+(rR(i+1),)
    '''Propagation-only adjoint sweep via Q, seeded at the terminal bond by the residual jet ``c``.

    The apply-transpose analog of :py:func:`compute_sigma_tilde_jets`: no ``deta_tilde`` per-core source,
    and the init carry is the ``c``-seed on the terminal bond (``rR_d = 1``) rather than zeros.
    ``sigma_hats[i]`` is the adjoint of the after-core-``i`` perturbation carry; it carries the tangent
    stack ``K`` (from ``c``). Right-to-left via ``tt_reverse`` (mirroring the ``Q``-sweep there).
    '''
    use_jax = tree_contains_jax((right_tt_cores, xi_jets, c, trs))
    xnp, xmap, xscan = get_backend(is_ndarray(right_tt_cores), use_jax)  # scan-style: xscan strips d, per-slice
    s_size = min(2, trs.shape[0])
    trs_xi = trs[:, :s_size, :]                # input jet (xi) carries orders {0, 1}

    # Polymorphic reverse (uniform tt_reverse keeps the supercore; tt_operations.tt_reverse would iterate d).
    reverse = tt_operations.tt_reverse if is_ndarray(right_tt_cores) else tt_operations.tt_reverse
    rev_Q = reverse(right_tt_cores)
    rev_xi = xi_jets[::-1]
    # The forward sums the terminal bond (rR_d, not necessarily 1 -- e.g. the corewise frame's own
    # cores), so the adjoint BROADCASTS c over it: seed -> (order+1)+W+K+C+(rR_d,).
    rR_d = right_tt_cores[-1].shape[-1]
    seed = xnp.broadcast_to(c[..., None], tuple(c.shape) + (rR_d,))

    def _step(carry, data):
        Q, xi = data
        return contractions.contract('trs,tWKCa,Caib,rWCi->sWKCb', trs_xi, carry, Q, xi[:s_size]), (carry,)

    _, (rev_sigma_hats,) = xscan(_step, seed, (rev_Q, rev_xi))
    return rev_sigma_hats[::-1]


def _apply_derivatives_transpose_from_jets(
        c, xi_jets, mu_jets, w_jets, down_tt_cores, right_tt_cores, trs, n_probe, sum_over_probes,
):
    '''Shared tail of apply/tv_entries_derivatives_transpose (they differ only in how xi_jets /
    w_jets are formed). Runs the seeded sigma_hat sweep, the dxi_hat combine, and the single-term
    gradient assembly (dG = mu (x) xi (x) sigma_hat, dU = dxi_hat (x) w_jet). Returns (dU, dG).'''
    use_jax = tree_contains_jax((c, xi_jets, mu_jets, w_jets, down_tt_cores, right_tt_cores, trs))
    is_uniform = is_ndarray(down_tt_cores)
    xnp, xmap, _ = get_backend(is_uniform, use_jax)
    s_size = min(2, trs.shape[0])
    sigma_hats = compute_sigma_hat_jets(right_tt_cores, xi_jets, c, trs)   # polymorphic

    if is_uniform:
        # d-prefixed adjoint combine + single-term assembly (3b-6'a); the w/dxi order-slice is [:, :s_size]
        # (order at supercore axis 1). trs is shared (no d). Ragged xmap is the oracle.
        dxi_hats = contractions.contract('trs,drWCa,dCaib,dtWKCb->dsWKCi', trs, mu_jets, down_tt_cores, sigma_hats)
        dG = 'trs,drWCa,dsWCi,dtWKCb->dKCaib' if sum_over_probes else 'trs,drWCa,dsWCi,dtWKCb->dWKCaib'
        dU = 'duWKCa,duWo->dKCao' if sum_over_probes else 'duWKCa,duWo->dWKCao'
        dG_tildes = contractions.contract(dG, trs[:, :, :s_size], mu_jets, xi_jets[:, :s_size], sigma_hats, len_W=n_probe)
        dU_tildes = contractions.contract(dU, dxi_hats[:, :s_size], w_jets[:, :s_size])
        return dU_tildes, dG_tildes

    def _dxi_hat(data):
        O, mu, sh = data                        # dxi_hat = mu * O * sigma_hat (mode leg free)
        return (contractions.contract('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu, O, sh),)

    (dxi_hats,) = xmap(_dxi_hat, (down_tt_cores, mu_jets, sigma_hats))

    dG = 'trs,rWCa,sWCi,tWKCb->KCaib' if sum_over_probes else 'trs,rWCa,sWCi,tWKCb->WKCaib'
    dU = 'uWKCa,uWo->KCao' if sum_over_probes else 'uWKCa,uWo->WKCao'

    def _asm(data):
        xi, mu, sh, dxh, wj = data
        dG_t = contractions.contract(dG, trs[:, :, :s_size], mu, xi[:s_size], sh, len_W=n_probe)
        dU_t = contractions.contract(dU, dxh[:s_size], wj[:s_size])
        return (dU_t, dG_t)

    (dU_tildes, dG_tildes) = xmap(_asm, (xi_jets, mu_jets, sigma_hats, dxi_hats, w_jets))
    return dU_tildes, dG_tildes


def tv_apply_transpose_derivatives_from_sweep(
        c:          NDArray,                # residual jet (scalar), shape=(order+1)+W+K+C
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        sweep:      typ.Tuple[
            typ.Sequence[NDArray],          # xi_jets
            typ.Sequence[NDArray],          # mu_jets
        ],                                  # = tv_precompute_apply_frame_sweep_jets(frame, ww, pp, order)
        order:      int,                    # highest derivative order
        sum_over_probes: bool = False,      # True: sum the sample stack W (the J^T r back-projection)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes
    typ.Tuple[NDArray, ...],  # dG_tildes
]:                                          # = T3Variations.data
    '''Variation gradient of :py:func:`tv_apply_derivatives_transpose` from a precomputed frame
    ``sweep``: the ``c``-seeded ``sigma_hat`` sweep + the single-term assembly, reusing the frame
    ``(xi, mu)_jets`` (``w_jets`` is frame-free, recomputed here). The reuse hook for a fitting inner solve.'''
    _, down_tt_cores, _, right_tt_cores = frame
    xi_jets, mu_jets = sweep
    trs = binomial_combine_tensor(order)
    n_probe = ww[0].ndim - 1
    w_jets = build_input_jets(ww, pp)               # ambient input jets (value, direction) for dU
    return _apply_derivatives_transpose_from_jets(
        c, xi_jets, mu_jets, w_jets, down_tt_cores, right_tt_cores, trs, n_probe, sum_over_probes)


def tv_apply_derivatives_transpose(
        c:          NDArray,                # residual jet (scalar), shape=(order+1)+W+K+C
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        order:      int,                    # highest derivative order
        sum_over_probes: bool = False,      # True: sum the sample stack W (the J^T r back-projection)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes (Tucker variation gradient)
    typ.Tuple[NDArray, ...],  # dG_tildes (TT variation gradient)
]:                                          # = T3Variations.data
    '''Transpose of :py:func:`tv_apply_derivatives`: back-project residual jets ``c`` into a
    variation gradient ``(dU_tildes, dG_tildes)``. The adjoint-state apply transpose -- the scalar
    residual jet ``c`` seeds one propagation sweep (no per-mode residual, no nu/eta), so it is about
    half the probe transpose. Full ``W + K + C`` stacking: ``c`` carries the tangent stack ``K``, which
    rides to the variation gradient; ``sum_over_probes`` sums (``True``, the Gauss-Newton ``J^T r``) or
    keeps (``False``, ``W`` rides into the variation stack) the sample stack ``W``; ``K``/``C`` always
    kept. Verified vs the dense adjoint identity ``<c, J v> = <J^T c, v>`` and ``jax.linear_transpose``.

    See Also
    --------
    tv_apply_derivatives
    tv_entries_derivatives_transpose
    tv_probe_derivatives_transpose
    '''
    sweep = tv_precompute_apply_frame_sweep_jets(frame, ww, pp, order)
    return tv_apply_transpose_derivatives_from_sweep(c, ww, pp, frame, sweep, order, sum_over_probes)


def tv_entries_transpose_derivatives_from_sweep(
        c:          NDArray,                # residual jet (scalar), shape=(order+1)+W+K+C
        index:      NDArray,                # int, shape=(d,)+W -- the grid points
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        sweep:      typ.Tuple[
            typ.Sequence[NDArray],          # xi_jets
            typ.Sequence[NDArray],          # mu_jets
        ],                                  # = tv_precompute_entries_frame_sweep_jets(frame, index, pp, order)
        order:      int,                    # highest derivative order
        sum_over_probes: bool = False,      # True: sum the sample stack W (the J^T r back-projection)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes
    typ.Tuple[NDArray, ...],  # dG_tildes
]:                                          # = T3Variations.data
    '''Variation gradient of :py:func:`tv_entries_derivatives_transpose` from a precomputed frame
    ``sweep``: the entries analog of :py:func:`tv_apply_transpose_derivatives_from_sweep` (ambient
    ``w_jets`` from the one-hot ``e_{index}`` + ``P``), reusing the frame ``(xi, mu)_jets``.'''
    up_tucker_cores, down_tt_cores, _, right_tt_cores = frame
    xi_jets, mu_jets = sweep
    trs = binomial_combine_tensor(order)
    ww_onehot = _onehot_vectors(index, up_tucker_cores)   # e_{index}, elm_shape=W+(Ni,)
    n_probe = ww_onehot[0].ndim - 1
    w_jets = build_input_jets(ww_onehot, pp)         # ambient input jets (one-hot, direction) for dU
    return _apply_derivatives_transpose_from_jets(
        c, xi_jets, mu_jets, w_jets, down_tt_cores, right_tt_cores, trs, n_probe, sum_over_probes)


def tv_entries_derivatives_transpose(
        c:          NDArray,                # residual jet (scalar), shape=(order+1)+W+K+C
        index:      NDArray,                # int, shape=(d,)+W -- the grid points
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        frame:       typ.Tuple[
            typ.Sequence[NDArray], typ.Sequence[NDArray],
            typ.Sequence[NDArray], typ.Sequence[NDArray],
        ],                                  # = T3Frame.data = (U, O, P, Q)
        order:      int,                    # highest derivative order
        sum_over_probes: bool = False,      # True: sum the sample stack W (the J^T r back-projection)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes
    typ.Tuple[NDArray, ...],  # dG_tildes
]:                                          # = T3Variations.data
    '''Transpose of :py:func:`tv_entries_derivatives`: scatter residual jets ``c`` at ``index``
    into a variation gradient. Identical to :py:func:`tv_apply_derivatives_transpose` with the
    frame up-index jet from fiber slicing at ``index`` (order 0) + contracting ``P`` (order 1), and the
    ambient ``w_jet`` from the unit vectors ``e_{index}`` (order 0) + ``P`` (order 1) -- so the
    Tucker-variation gradient scatters onto the indexed rows.

    See Also
    --------
    tv_entries_derivatives
    tv_apply_derivatives_transpose
    '''
    sweep = tv_precompute_entries_frame_sweep_jets(frame, index, pp, order)
    return tv_entries_transpose_derivatives_from_sweep(c, index, pp, frame, sweep, order, sum_over_probes)


###############################################################
####    Corewise (non-manifold) derivative transposes      ####
###############################################################
#
# The gradient of a plain-T3 derivative sampling op w.r.t. the cores of the frame (treated as
# independent variables) -- the Section 6.3 "corewise simplification": the tangent derivative
# transpose with the frame's OWN cores in place of the orthogonal frames (P, Q, O -> G_i), U no longer
# required orthogonal. Trivial substitution wrappers (cf. the t3_*_corewise_transpose trio in probing/apply/entries);
# return raw (tucker_grads, tt_grads) shaped like the cores. Verified vs jax.grad of the forward.


def t3_probe_corewise_derivatives_transpose(
        ztildes:    typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1)+W+C+(Ni,)
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        core_pair:  typ.Tuple[
            typ.Sequence[NDArray],          # tucker_cores, len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],          # tt_cores,     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],                                  # = TuckerTensorTrain.data
        order:      int,                    # highest derivative order
        sum_over_probes: bool = False,      # True: sum the sample stack W (the gradient J^T r)
        chunk_size: typ.Optional[int] = 100,   # W-chunk size for the gradient assembly; None -> dense. docs/chunking.md
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker-core gradients, same shapes as tucker_cores
    typ.Tuple[NDArray, ...],  # tt-core gradients,     same shapes as tt_cores
]:
    '''Corewise (non-manifold) transpose of :py:func:`t3_probe_derivatives`: gradient of the
    probe-derivative jets w.r.t. the frame ``core_pair``'s cores, as independent variables (for
    core-wise optimizers). The Section 6.3 substitution ``P,Q,O -> G`` into
    :py:func:`tv_probe_derivatives_transpose` (frame ``(U, G, G, G)``; orthogonality not required).
    Returns gradients shaped like ``(tucker_cores, tt_cores)``. Verified vs ``jax.grad``.
    '''
    tucker_cores, tt_cores = core_pair
    return tv_probe_derivatives_transpose(
        ztildes, ww, pp, (tucker_cores, tt_cores, tt_cores, tt_cores), order,
        sum_over_probes=sum_over_probes, chunk_size=chunk_size)


def t3_apply_corewise_derivatives_transpose(
        c:          NDArray,                # residual jet (scalar), shape=(order+1)+W+C
        ww:         typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        core_pair:  typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # = TuckerTensorTrain.data
        order:      int,                    # highest derivative order
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:  # (tucker_grads, tt_grads)
    '''Corewise transpose of :py:func:`t3_apply_derivatives`: gradient of the apply-derivative jets
    w.r.t. the frame cores (Section 6.3 substitution into :py:func:`tv_apply_derivatives_transpose`).
    '''
    tucker_cores, tt_cores = core_pair
    return tv_apply_derivatives_transpose(
        c, ww, pp, (tucker_cores, tt_cores, tt_cores, tt_cores), order, sum_over_probes=sum_over_probes)


def t3_entries_corewise_derivatives_transpose(
        c:          NDArray,                # residual jet (scalar), shape=(order+1)+W+C
        index:      NDArray,                # int, shape=(d,)+W
        pp:         typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        core_pair:  typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # = TuckerTensorTrain.data
        order:      int,                    # highest derivative order
        sum_over_probes: bool = False,
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:  # (tucker_grads, tt_grads)
    '''Corewise transpose of :py:func:`t3_entries_derivatives`: gradient of the entry-derivative jets
    w.r.t. the frame cores (Section 6.3 substitution into :py:func:`tv_entries_derivatives_transpose`).
    '''
    tucker_cores, tt_cores = core_pair
    return tv_entries_derivatives_transpose(
        c, index, pp, (tucker_cores, tt_cores, tt_cores, tt_cores), order, sum_over_probes=sum_over_probes)


#####################################################
########    Dense reference (test oracle)    ########
#####################################################

def dense_probe_derivatives(
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


def dense_apply_derivatives(
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


def dense_entries_derivatives(
        index:  typ.Sequence[int],      # the grid point (one int per mode), len=d
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=(Ni,)
        T:      NDArray,                # dense tensor, shape=(N0,...,N(d-1))
        order:  int,                    # highest derivative order
) -> NDArray:                           # entries-derivative jets, shape=(order+1,)
    '''Exact dense symmetric entry derivatives (oracle): apply-derivatives with one-hot basis vectors
    ``e_{index}`` (entries = apply with one-hot), via :py:func:`dense_apply_derivatives`. Unstacked.'''
    ww = [np.eye(T.shape[j])[index[j]] for j in range(T.ndim)]   # one-hot e_{index_j}
    return dense_apply_derivatives(ww, pp, T, order)
