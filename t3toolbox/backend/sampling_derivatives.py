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
#   - compute_mu_jets : left  jets mu_i^(t)  via trs_rWCa_Caib_sWCi_to_tWCb (input jet on the mode).
#   - compute_nu_jets : right jets nu_i^(t)  -- the mirror image (tt_reverse).
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
    'compute_mu_jets',
    'compute_nu_jets',
    'compute_eta_jets',
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
    'compute_deta_tilde_jets',
    'compute_tau_tilde_jets',
    'compute_sigma_tilde_jets',
    'compute_dxi_tilde_jets',
    'assemble_tucker_variation_jets',
    'assemble_tt_variation_jets',
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


def compute_mu_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        trs:        NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:               # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,). mu_jets[i][t]=mu_{i-1}^(t)
    '''Left derivative-pushthrough jets.

    Sweep left-to-right, at each core taking the binomial jet-product of the running left jet with the
    input jet through the core (``trs_rWCa_Caib_sWCi_to_tWCb``). Like :py:func:`probing.compute_mu`,
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


# ==================================================================================================
# EXPERIMENTAL -- convolution/recurrence forms of the jet contractions (module-private; workshop)
# ==================================================================================================
# A slow mirror of the trs-based jet functions above, built one at a time and checked to give the
# SAME results (tests/test_jet_recurrence.py), to see whether they run faster / use less memory. If
# they pan out, we swap them into the call sites. NOT public API (no __all__ entry); the design
# question that motivates them is dev/OPEN_QUESTION_contractions_architecture.md.
#
# The idea: a trs binomial tensor is a sparse convolution tensor, so contracting it as a DENSE einsum
# operand is the wrong handling -- it is what makes _pairwise_path degenerate (the trs operand shares
# one index per operand, so it sorts LAST and the intermediate balloons to the union of all indices).
# The right form unrolls the convolution over the order axis into ordinary (non-trs) contractions.
#
# mu/nu pushthrough (this slice) is the CLEAN case: the input jet is affine, so xi is nonzero only at
# orders s in {0,1} (build_input_jets returns size 2). The binomial sum then has just two surviving
# terms -- a bidiagonal recurrence, no trs tensor, no (order+1)^2 work:
#
#     mu_i^(t)  =  [mu^(t) . G . xi^(0)]  +  t * [mu^(t-1) . G . xi^(1)]
#
# (C(t,t)=1 gives the s=0 term; C(t,t-1)=t gives the s=1 term.) Verified equal to the dense trs
# contraction to 1e-16. (eta combine is a DIFFERENT, genuinely-full convolution -- nu is a full jet,
# not affine-truncated -- and is a later slice.)


def _Caib_sWCi_to_sWCab(
        G:       NDArray,  # C + (a, i, b)      -- tt core (C-only frame stack)
        xi_jet:  NDArray,  # s + W + C + (i,)   -- input jet on mode i, s in {0,1}
) -> NDArray:              # s + W + C + (a, b) -- core with the input jet contracted on mode i
    '''Core times input jet, contracting mode ``i``, keeping the order axis ``s``. C is SHARED
    (flattened to one letter); the passive s+W prefix rides as ``'...'``.'''
    use_jax = tree_contains_jax((G, xi_jet))
    xnp, _, _ = get_backend(is_ndarray(G), use_jax)

    C_shape = G.shape[:-3]
    a, i, b = G.shape[-3], G.shape[-2], G.shape[-1]
    sW_shape = xi_jet.shape[:-(len(C_shape) + 1)]      # s + W, ride as '...'
    size_C = math.prod(C_shape)

    G_flat = G.reshape((size_C, a, i, b))
    xi_flat = xi_jet.reshape(sW_shape + (size_C, i))
    out = xnp.einsum('Caib,...Ci->...Cab', G_flat, xi_flat)
    return out.reshape(sW_shape + C_shape + (a, b))


def _tWCa_WCab_to_tWCb(
        mu_jet:  NDArray,  # t + W + C + (a,)    -- left jet at order t (t passive)
        Gxi:     NDArray,  #     W + C + (a, b)  -- core-with-input-jet, one order slice
) -> NDArray:              # t + W + C + (b,)    -- mu^(t) . G . xi, bond a contracted
    '''Contract the left bond ``a``, keeping the order axis ``t`` (passive on ``mu_jet``). W+C is
    shared and contiguous on both operands, so it rides as ``'...'`` -- nothing is flattened.'''
    use_jax = tree_contains_jax((mu_jet, Gxi))
    xnp, _, _ = get_backend(is_ndarray(mu_jet), use_jax)
    return xnp.einsum('t...a,...ab->t...b', mu_jet, Gxi)


def compute_mu_jets_banded(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        trs:        NDArray,                # binomial tensor -- ONLY its shape (order) is read here
) -> typ.Tuple[NDArray, ...]:               # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,). mu_jets[i][t]=mu_{i-1}^(t)
    '''EXPERIMENTAL banded-recurrence mirror of :py:func:`compute_mu_jets` (module-private).

    Same signature and same result, but the affine input jet (``s in {0,1}``) makes each pushthrough a
    two-term recurrence rather than a dense ``trs`` contraction:

        ``mu_i^(t) = mu^(t) . G . xi^(0)  +  t * mu^(t-1) . G . xi^(1)``

    ``trs`` is taken only for its order (``shape[0]-1``), never contracted -- kept in the signature so
    it is a drop-in swap for ``compute_mu_jets`` and the equivalence test can call both identically.
    '''
    use_jax = tree_contains_jax((tt_cores, xi_jets, trs))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    order = trs.shape[0] - 1
    s_size = min(2, order + 1)                             # affine input jet: orders {0, 1}
    tvec = xnp.arange(order + 1)                           # the C(t,t-1)=t multipliers

    def _func(mu_jet, data):
        G, xi_jet = data
        Gxi = _Caib_sWCi_to_sWCab(G, xi_jet[:s_size])     # (s, W, C, a, b), s in {0,1}
        next_mu = _tWCa_WCab_to_tWCb(mu_jet, Gxi[0])      # the s=0 term: mu^(t) . G . xi^(0)
        if s_size > 1:                                     # static branch (order >= 1)
            M1 = _tWCa_WCab_to_tWCb(mu_jet, Gxi[1])       # mu^(t) . G . xi^(1)
            shift = xnp.concatenate([xnp.zeros_like(M1[:1]), M1[:-1]], axis=0)   # mu^(t-1) term
            t_bcast = tvec.reshape((order + 1,) + (1,) * (next_mu.ndim - 1))
            next_mu = next_mu + t_bcast * shift            # + t * mu^(t-1) . G . xi^(1)
        return next_mu, (mu_jet,)

    stack_shape = xi_jets[0].shape[1:-1]     # full W + C batch (W outer, C inner); either may be empty
    r0 = tt_cores[0].shape[-3]
    init = _init_jet(order, stack_shape, r0, xnp)

    _, (mu_jets,) = xscan(_func, init, (tt_cores, xi_jets))
    return mu_jets


def _stWCa_sWCab_to_tWCb(
        stacked_mu:  NDArray,  # s + t + W + C + (a,)    -- jet-pair axis s, order t (both passive on t)
        Gxi:         NDArray,  # s +     W + C + (a, b)  -- core-with-input-jet, both order slices
) -> NDArray:                  #     t + W + C + (b,)    -- the two-term step, in ONE contraction
    '''Fused two-term recurrence step: contract the jet-pair axis ``s`` AND the bond ``a`` in a single
    einsum. With ``stacked_mu[0] = mu^(t)``, ``stacked_mu[1] = t * mu^(t-1)``, and ``Gxi[s] = G . xi^(s)``,
    the ``s`` sum reproduces ``mu^(t) . G . xi^(0) + t * mu^(t-1) . G . xi^(1)``. One larger GEMM instead
    of two smaller ones -- gives XLA a single contraction to schedule (the whole point of the fused form).'''
    use_jax = tree_contains_jax((stacked_mu, Gxi))
    xnp, _, _ = get_backend(is_ndarray(stacked_mu), use_jax)
    return xnp.einsum('st...a,s...ab->t...b', stacked_mu, Gxi)


def compute_mu_jets_banded_fused(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        trs:        NDArray,                # binomial tensor -- ONLY its shape (order) is read here
) -> typ.Tuple[NDArray, ...]:               # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,). mu_jets[i][t]=mu_{i-1}^(t)
    '''EXPERIMENTAL fused variant of :py:func:`compute_mu_jets_banded` (module-private).

    Same two-term recurrence, but folded into ONE contraction per core instead of two einsums + a shift
    + a broadcast: stack ``[mu^(t), t * mu^(t-1)]`` on a jet-pair axis ``s`` and contract it together with
    the bond ``a`` against ``[G.xi^(0), G.xi^(1)]`` (:py:func:`_stWCa_sWCab_to_tWCb`). Motivation: under
    jit the two-einsum form is ~parity with the dense ``trs`` at scale; giving XLA a single larger GEMM
    is the lever to turn parity into a win. The shift (for ``mu^(t-1)``) is still needed, but it now
    feeds the single contraction rather than sitting between two.
    '''
    use_jax = tree_contains_jax((tt_cores, xi_jets, trs))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    order = trs.shape[0] - 1
    s_size = min(2, order + 1)                             # affine input jet: orders {0, 1}
    tvec = xnp.arange(order + 1)                           # the C(t,t-1)=t multipliers

    def _func(mu_jet, data):
        G, xi_jet = data
        Gxi = _Caib_sWCi_to_sWCab(G, xi_jet[:s_size])     # (s, W, C, a, b), s in {0,1}
        if s_size > 1:                                     # static branch (order >= 1)
            t_bcast = tvec.reshape((order + 1,) + (1,) * (mu_jet.ndim - 1))
            shifted = xnp.concatenate([xnp.zeros_like(mu_jet[:1]), mu_jet[:-1]], axis=0)  # mu^(t-1)
            stacked_mu = xnp.stack([mu_jet, t_bcast * shifted], axis=0)   # (s=2,) + mu jet shape
            next_mu = _stWCa_sWCab_to_tWCb(stacked_mu, Gxi)
        else:                                              # order 0: only s=0 survives
            next_mu = _tWCa_WCab_to_tWCb(mu_jet, Gxi[0])
        return next_mu, (mu_jet,)

    stack_shape = xi_jets[0].shape[1:-1]
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

    The mirror image of :py:func:`compute_mu_jets`: reverse the tensor train (``tt_reverse`` swaps
    bonds and core order), run the left sweep, reverse the result. ``nu_jets[i]`` is the right edge
    variable entering core ``i`` (``nu_i``), stacked over derivative orders.
    '''
    # Polymorphic reverse: the uniform tt_reverse keeps the supercore (tt_operations.tt_reverse would iterate
    # the supercore's d axis -- the unroll trap). The jet slices [::-1] just reverse the leading d axis.
    reverse = tt_operations.tt_reverse if is_ndarray(tt_cores) else tt_operations.tt_reverse
    rev_nu_jets = compute_mu_jets(reverse(tt_cores), xi_jets[::-1], trs)
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

    if is_uniform:
        # d-prefixed jet combine (3b-6'a), vectorized over the core index d; the ragged xmap below is the
        # oracle. mu/nu jets are (d,)+(order,)+W+C+(r,); the tt supercore is (d,)+C+(rL,nO,rR) (C-only).
        eta_jets = contractions.trs_drWCa_dCaib_dsWCb_to_dtWCi(trs, mu_jets, tt_cores, nu_jets)
    else:
        def _func(data):
            mu_jet, G, nu_jet = data
            return (contractions.trs_rWCa_Caib_sWCb_to_tWCi(trs, mu_jet, G, nu_jet),)

        (eta_jets,) = xmap(_func, (mu_jets, tt_cores, nu_jets))
    return eta_jets


def compute_eta_jets_scanned(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:        NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:               # eta_jets. len=d, elm_shape=(order+1,)+W+C+(nOi,)
    '''EXPERIMENTAL memory-lean mirror of :py:func:`compute_eta_jets` (module-private).

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
    :py:func:`compute_eta_jets` to 1e-12 (ragged) and 1e-7 (uniform, float32).
    '''
    use_jax = tree_contains_jax((tt_cores, mu_jets, nu_jets, trs))
    xnp, xmap, xscan = get_backend(is_ndarray(tt_cores), use_jax)
    order = trs.shape[0] - 1
    trs_r = xnp.moveaxis(trs, 1, 0)                        # binomial CONSTANT (order+1)^3; scan leads on input order r -- const-folded, ~free

    def _func(data):
        mu_jet, G, nu_jet = data                          # (T,W,C,a) ; (C,a,i,b) ; (T,W,C,b)
        C_shape = G.shape[:-3]
        a, i, b = G.shape[-3], G.shape[-2], G.shape[-1]
        W_shape = mu_jet.shape[1:-(len(C_shape) + 1)]
        size_C = math.prod(C_shape)

        G_f = G.reshape((size_C, a, i, b))
        mu_f = mu_jet.reshape((order + 1,) + W_shape + (size_C, a))
        nu_f = nu_jet.reshape((order + 1,) + W_shape + (size_C, b))

        def _accumulate(eta, xr):
            mu_r, trsr = xr                                          # (W,C,a) ; (t,s)
            MG_r = xnp.einsum('...Ca,Caib->...Cib', mu_r, G_f)       # peak: W + C + (i, b)
            MGN_r = xnp.einsum('...Cib,s...Cb->s...Ci', MG_r, nu_f)  # fold in all of nu -> order s
            return eta + xnp.einsum('ts,s...Ci->t...Ci', trsr, MGN_r), ()   # binomial weights over t

        eta0 = xnp.zeros((order + 1,) + W_shape + (size_C, i), mu_jet.dtype)
        eta, _ = xscan(_accumulate, eta0, (mu_f, trs_r))
        return (eta.reshape((order + 1,) + W_shape + C_shape + (i,)),)

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
        z_jets = contractions.dtWCi_dCio_to_dtWCo(eta_jets, tucker_cores)
    else:
        def _func(data):
            eta_jet, U = data
            return (contractions.tWCi_Cio_to_tWCo(eta_jet, U),)

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
    compute_sigma_jets (keeps the per-core sequence) and tv_apply_derivatives (keeps only the
    terminal carry). Three-group (W sample, K tangent, C frame): sigma/dxi carry K, the frame edge vars
    (xi, mu) and frame cores (Q, O) do not; t2's only core is the variation core dG (K+C), so len(C) is
    supplied via n_frame (recovered from the C-only Q). Reduces to the 2-group result when K=().'''
    s_size = trs_push.shape[2]                # input jets carry orders {0, 1}, capped at order
    n_frame = Q.ndim - 3
    t1 = contractions.trs_rWKCa_Caib_sWCi_to_tWKCb(trs_push, sigma_jet, Q,  xi_jet[:s_size])
    t2 = contractions.trs_rWCa_KCaib_sWCi_to_tWKCb(trs_push, mu_jet,    dG, xi_jet[:s_size], n_frame)
    t3 = contractions.trs_rWCa_Caib_sWKCi_to_tWKCb(trs_push, mu_jet,    O,  dxi_jet[:s_size])
    return t1 + t2 + t3


def compute_sigma_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        mu_jets:        typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # sigma_jets. len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
    '''Variation-leftward edge-variable jets sigma (the jet-ified Algorithm-7 sigma recursion).

    ``sigma_i = sigma_{i-1} Q_i(xi_i) + mu_{i-1} dG_i(xi_i) + mu_{i-1} O_i(dxi_i)`` -- three
    pushthroughs (``trs_rWCa_Caib_sWCi_to_tWCb``): the carried sigma jet through Q, and the frame mu
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
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        nu_jets:        typ.Sequence[NDArray],  # frame right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # tau_jets. len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
    '''Variation-rightward edge-variable jets tau -- the mirror of :py:func:`compute_sigma_jets`.

    Reverse the train (P in the Q-slot, O and dG reversed), run the sigma sweep, reverse the result.
    '''
    reverse = tt_operations.tt_reverse if is_ndarray(var_tt_cores) else tt_operations.tt_reverse
    rev = compute_sigma_jets(
        reverse(var_tt_cores), reverse(left_tt_cores),
        reverse(down_tt_cores), xi_jets[::-1], dxi_jets[::-1], nu_jets[::-1], trs,
    )
    return rev[::-1]


def compute_deta_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_jets:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_jets:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # deta_jets. len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
    '''Variation-downward edge-variable jets deta (the jet-ified Algorithm-7 deta combine).

    ``deta_i = sigma_{i-1} Q_i nu_i + mu_{i-1} dG_i nu_i + mu_{i-1} P_i tau_i`` -- three combines
    (``trs_rWCa_Caib_sWCb_to_tWCi``), mode ``i`` free.
    '''
    use_jax = tree_contains_jax((var_tt_cores, left_tt_cores, right_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets, trs))
    is_uniform = is_ndarray(var_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed jet combines (3b-6'a), vectorized over d; the ragged xmap below is the oracle. sigma/tau
        # jets carry K (W+K+C); mu/nu are W+C; P/Q are C-only supercores; term2's only core is dG (K+C), so
        # len(C) = n_frame, read off the C-only Q supercore (d,)+C+(rR,nU,rR).
        n_frame = right_tt_cores.ndim - 4
        term1 = contractions.trs_drWKCa_dCaib_dsWCb_to_dtWKCi(trs, sigma_jets, right_tt_cores, nu_jets)
        term2 = contractions.trs_drWCa_dKCaib_dsWCb_to_dtWKCi(trs, mu_jets, var_tt_cores, nu_jets, n_frame)
        term3 = contractions.trs_drWCa_dCaib_dsWKCb_to_dtWKCi(trs, mu_jets, left_tt_cores, tau_jets)
        deta_jets = term1 + term2 + term3
    else:
        def _func(data):
            P, Q, dG, mu_jet, nu_jet, sigma_jet, tau_jet = data
            # Three-group (W,K,C): sigma/tau carry K, mu/nu and frame cores P/Q do not; term2's only core
            # is the variation core dG (K+C), so len(C) is supplied via n_frame (the C-only Q pins it).
            n_frame = Q.ndim - 3
            term1 = contractions.trs_rWKCa_Caib_sWCb_to_tWKCi(trs, sigma_jet, Q,  nu_jet)
            term2 = contractions.trs_rWCa_KCaib_sWCb_to_tWKCi(trs, mu_jet,    dG, nu_jet, n_frame)
            term3 = contractions.trs_rWCa_Caib_sWKCb_to_tWKCi(trs, mu_jet,    P,  tau_jet)
            return (term1 + term2 + term3,)

        (deta_jets,) = xmap(_func, (left_tt_cores, right_tt_cores, var_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets))
    return deta_jets


def compute_deta_jets_scanned(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_jets:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_jets:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # deta_jets. len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
    '''EXPERIMENTAL memory-lean mirror of :py:func:`compute_deta_jets` (module-private).

    The three-term tangent analog of :py:func:`compute_eta_jets_scanned`: ``deta_i = sigma Q nu +
    mu dG nu + mu P tau``, three FULL convolutions (mode ``i`` free). The dense uniform d-einsum forms
    the ``(order+1)*W*K*r^2`` spatial product for every core at once; this scans the input order ``r``
    (one order slice of ``jetL . core`` live at a time) and folds in all of the right jet + the
    binomial weights -- peak ``W*K*r^2``. The K tangent stack sits on a *different* operand in each
    term (sigma / dG / tau), so W, K, C are flattened to explicit blocks (K rides). Memory win needs
    the uniform path (real ``lax.map``/``lax.scan``); see :py:func:`compute_eta_jets_scanned`. Verified
    equal to :py:func:`compute_deta_jets` to 1e-12.
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
        a_mu, a_sig = mu_jet.shape[-1], sigma_jet.shape[-1]
        b_nu, b_tau = nu_jet.shape[-1], tau_jet.shape[-1]
        nW = mu_jet.ndim - 2 - nf                     # mu is (order+1,)+W+C+(a,)
        W_shape = mu_jet.shape[1:1 + nW]
        nK = (sigma_jet.ndim - 2) - nW - nf           # sigma is (order+1,)+W+K+C+(a,)
        K_shape = sigma_jet.shape[1 + nW:1 + nW + nK]
        sW, sK, sC = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)

        Qf = Q.reshape((sC, a_sig, i, b_nu))
        Pf = P.reshape((sC, a_mu, i, b_tau))
        dGf = dG.reshape((sK, sC, a_mu, i, b_nu))
        mu_f = mu_jet.reshape((order + 1, sW, sC, a_mu))
        nu_f = nu_jet.reshape((order + 1, sW, sC, b_nu))
        sig_f = sigma_jet.reshape((order + 1, sW, sK, sC, a_sig))
        tau_f = tau_jet.reshape((order + 1, sW, sK, sC, b_tau))

        def _step(eta, xr):
            mu_r, sig_r, trsr = xr                                       # (sW,sC,a) ; (sW,sK,sC,a) ; (t,s)
            mg1 = xnp.einsum('WKCa,Caib->WKCib', sig_r, Qf)              # term1: sigma Q nu  (K on sigma)
            mgn1 = xnp.einsum('WKCib,sWCb->sWKCi', mg1, nu_f)
            mg2 = xnp.einsum('WCa,KCaib->WKCib', mu_r, dGf)             # term2: mu dG nu    (K on core)
            mgn2 = xnp.einsum('WKCib,sWCb->sWKCi', mg2, nu_f)
            mg3 = xnp.einsum('WCa,Caib->WCib', mu_r, Pf)               # term3: mu P tau    (K on tau)
            mgn3 = xnp.einsum('WCib,sWKCb->sWKCi', mg3, tau_f)
            contrib = xnp.einsum('ts,sWKCi->tWKCi', trsr, mgn1 + mgn2 + mgn3)
            return eta + contrib, ()

        eta0 = xnp.zeros((order + 1, sW, sK, sC, i), mu_jet.dtype)
        eta, _ = xscan(_step, eta0, (mu_f, sig_f, trs_r))
        return (eta.reshape((order + 1,) + W_shape + K_shape + C_shape + (i,)),)

    (deta_jets,) = xmap(_func, (left_tt_cores, right_tt_cores, var_tt_cores,
                                mu_jets, nu_jets, sigma_jets, tau_jets))
    return deta_jets


def _sigma_banded_step(sigma_jet, Q, O, dG, xi_jet, dxi_jet, mu_jet, s_size, tvec, order, xnp):
    '''One K-aware banded step of the sigma recursion: the three affine pushthroughs of
    :py:func:`_sigma_jet_step` as fused two-term recurrences (no trs). W, K, C flattened; K rides on
    the carried sigma (t1), the variation core dG (t2), or the var input dxi (t3).'''
    C_shape = Q.shape[:-3]
    nf = len(C_shape)
    nU, nO = Q.shape[-2], O.shape[-2]
    a_sig, a_mu, b = sigma_jet.shape[-1], mu_jet.shape[-1], Q.shape[-1]   # b = output bond rR(i+1)
    nW = xi_jet.ndim - 2 - nf
    W_shape = xi_jet.shape[1:1 + nW]
    nK = dxi_jet.ndim - 2 - nW - nf
    K_shape = dxi_jet.shape[1 + nW:1 + nW + nK]
    sW, sK, sC = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)

    Qf = Q.reshape((sC, a_sig, nU, b))
    dGf = dG.reshape((sK, sC, a_mu, nU, b))
    Of = O.reshape((sC, a_mu, nO, b))
    xi_f = xi_jet[:s_size].reshape((s_size, sW, sC, nU))
    dxi_f = dxi_jet[:s_size].reshape((s_size, sW, sK, sC, nO))
    sig_f = sigma_jet.reshape((order + 1, sW, sK, sC, a_sig))
    mu_f = mu_jet.reshape((order + 1, sW, sC, a_mu))

    Qxi = xnp.einsum('Caib,sWCi->sWCab', Qf, xi_f)          # (s,W,C,a,b)   -- no K
    dGxi = xnp.einsum('KCaib,sWCi->sWKCab', dGf, xi_f)      # (s,W,K,C,a,b) -- K on core
    Odxi = xnp.einsum('Caib,sWKCi->sWKCab', Of, dxi_f)      # (s,W,K,C,a,b) -- K on var input

    def stacked(jet_f):     # (order+1,...,a) -> (s=2,)+... = [jet^(t), t*jet^(t-1)]
        t_b = tvec.reshape((order + 1,) + (1,) * (jet_f.ndim - 1))
        shifted = xnp.concatenate([xnp.zeros_like(jet_f[:1]), jet_f[:-1]], axis=0)
        return xnp.stack([jet_f, t_b * shifted], axis=0)

    if s_size > 1:
        t1 = xnp.einsum('stWKCa,sWCab->tWKCb', stacked(sig_f), Qxi)     # K on sigma
        t2 = xnp.einsum('stWCa,sWKCab->tWKCb', stacked(mu_f), dGxi)     # K on core
        t3 = xnp.einsum('stWCa,sWKCab->tWKCb', stacked(mu_f), Odxi)     # K on var input
    else:                                                              # order 0: only s=0
        t1 = xnp.einsum('tWKCa,WCab->tWKCb', sig_f, Qxi[0])
        t2 = xnp.einsum('tWCa,WKCab->tWKCb', mu_f, dGxi[0])
        t3 = xnp.einsum('tWCa,WKCab->tWKCb', mu_f, Odxi[0])
    return (t1 + t2 + t3).reshape((order + 1,) + W_shape + K_shape + C_shape + (b,))


def compute_sigma_jets_banded(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        mu_jets:        typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        trs:            NDArray,                # binomial tensor -- ONLY its shape (order) is read here
) -> typ.Tuple[NDArray, ...]:                   # sigma_jets. len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
    '''EXPERIMENTAL banded mirror of :py:func:`compute_sigma_jets` (module-private).

    The three-pushthrough tangent analog of :py:func:`compute_mu_jets_banded_fused`. Both input jets
    (``xi``, ``dxi``) are affine (size 2), so each of the three pushthroughs
    (``sigma Q(xi) + mu dG(xi) + mu O(dxi)``) is a fused two-term recurrence rather than a dense ``trs``
    contraction. The K tangent stack rides on the carried sigma / the variation core / the var input.
    Verified equal to :py:func:`compute_sigma_jets` to 1e-12.
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


def compute_tau_jets_banded(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        nu_jets:        typ.Sequence[NDArray],  # frame right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor -- ONLY its shape (order) is read here
) -> typ.Tuple[NDArray, ...]:                   # tau_jets. len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
    '''EXPERIMENTAL banded mirror of :py:func:`compute_tau_jets` -- sigma-banded on the reversed train.'''
    reverse = tt_operations.tt_reverse
    rev = compute_sigma_jets_banded(
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
        term1 = contractions.dtWKCi_dCio_to_dtWKCo(deta_jets, tucker_cores)
        term2 = contractions.dtWCi_dKCio_to_dtWKCo(eta_jets, var_tucker_cores, n_frame)
        z_jets = term1 + term2
    else:
        def _func(data):
            U, dU, eta_jet, deta_jet = data
            # Three-group (W,K,C): deta carries K (in the lift via the C-only U, W and K ride passively); eta is W+C and dU is
            # the variation core (K+C), so the eta-lift needs len(C) -- recovered from the C-only U.
            n_frame = U.ndim - 2
            return (contractions.tWKCi_Cio_to_tWKCo(deta_jet, U)
                    + contractions.tWCi_KCio_to_tWKCo(eta_jet, dU, n_frame),)

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
        return contractions.trs_rWCa_Caib_sWCi_to_tWCb(trs_push, mu_jet, G, xi_jet[:s_size]), (0,)

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
        deta_tildes = contractions.dtWKCo_dCio_to_dtWKCi(ztildes, up_tucker_cores)
    else:
        def _func(data):
            U, zt = data
            return (contractions.tWKCo_Cio_to_tWKCi(zt, U),)

        (deta_tildes,) = xmap(_func, (up_tucker_cores, ztildes))
    return deta_tildes


def _adj_sweep(P_cores, xi_jets, deta_tildes, edge_jets, trs):
    '''The jet adjoint sweep shared by compute_tau_tilde_jets / compute_sigma_tilde_jets: a left-to-right scan
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
        prop = contractions.trs_tWKCa_Caib_rWCi_to_sWKCb(trs_xi, carry, P, xi[:s_size])  # propagation
        src  = contractions.trs_rWCa_Caib_tWKCi_to_sWKCb(trs,    edge,  P, deta_t)       # deta_tilde source
        return prop + src, (carry,)

    # carry is W+K+C; its leading stack comes from deta_tildes (which carry K), so the init carries K
    rL0 = P_cores[0].shape[-3]
    init = xnp.zeros((trs.shape[0],) + deta_tildes[0].shape[1:-1] + (rL0,))
    _, (tildes,) = xscan(_step, init, (P_cores, xi_jets, deta_tildes, edge_jets))
    return tildes


def compute_tau_tilde_jets(
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        mu_jets:        typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # tau_tildes. len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
    '''Adjoint-var-rightward edge-variable jets (jet-ified probing.compute_tau_tilde).'''
    return _adj_sweep(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)


def compute_sigma_tilde_jets(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        nu_jets:        typ.Sequence[NDArray],  # frame right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        trs:            NDArray,                # binomial tensor, shape=(order+1,order+1,order+1)
) -> typ.Tuple[NDArray, ...]:                   # sigma_tildes. len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
    '''Adjoint-var-leftward edge-variable jets -- the mirror (reverse) of compute_tau_tilde_jets.'''
    reverse = tt_operations.tt_reverse if is_ndarray(right_tt_cores) else tt_operations.tt_reverse
    rev = _adj_sweep(reverse(right_tt_cores), xi_jets[::-1],
                     deta_tildes[::-1], nu_jets[::-1], trs)
    return rev[::-1]


def _adj_tilde_step(carry, P, xi, deta_t, edge, s_size, svec, order, xnp, xscan, trs_r):
    '''One K-aware step of the memory-lean adjoint sweep. Two terms, both the TRANSPOSE of a forward
    contraction: **prop** (adjoint of the affine pushthrough) is a two-term REVERSE recurrence -- shifted
    UP (``carry^(s+1)``, weight ``s+1``), fused into one GEMM; **src** (adjoint of the full combine, the
    deta_tilde source) is a full REVERSE convolution, an inner order-scan over the edge order ``r`` with
    peak ``W*r^2`` per slice. W, K, C flattened (carry/deta carry K; xi/edge/P do not).'''
    C_shape = P.shape[:-3]
    nf = len(C_shape)
    a, i, b = P.shape[-3], P.shape[-2], P.shape[-1]
    nW = xi.ndim - 2 - nf
    W_shape = xi.shape[1:1 + nW]
    nK = carry.ndim - 2 - nW - nf
    K_shape = carry.shape[1 + nW:1 + nW + nK]
    sW, sK, sC = math.prod(W_shape), math.prod(K_shape), math.prod(C_shape)

    Pf = P.reshape((sC, a, i, b))
    carry_f = carry.reshape((order + 1, sW, sK, sC, a))
    xi_f = xi[:s_size].reshape((s_size, sW, sC, i))
    edge_f = edge.reshape((order + 1, sW, sC, a))
    deta_f = deta_t.reshape((order + 1, sW, sK, sC, i))

    # --- prop: two-term reverse recurrence (xi affine). prop^(s) = carry^(s) P xi^(0) + (s+1) carry^(s+1) P xi^(1)
    Pxi = xnp.einsum('Caib,kWCi->kWCab', Pf, xi_f)              # (k=s_size, W, C, a, b)
    if s_size > 1:
        sp1 = (svec + 1).reshape((order + 1,) + (1,) * (carry_f.ndim - 1))
        up = xnp.concatenate([carry_f[1:], xnp.zeros_like(carry_f[:1])], axis=0)   # carry^(s+1)
        stacked = xnp.stack([carry_f, sp1 * up], axis=0)        # (2,)+carry shape
        prop = xnp.einsum('ksWKCa,kWCab->sWKCb', stacked, Pxi)
    else:
        prop = xnp.einsum('sWKCa,WCab->sWKCb', carry_f, Pxi[0])

    # --- src: full reverse convolution -- inner scan over the edge order r, peak W*r^2 per slice
    def _src_step(acc, xr):
        edge_r, trsr = xr                                       # (W,C,a) ; (t,s)
        ep = xnp.einsum('WCa,Caib->WCib', edge_r, Pf)           # edge^(r) P over a -- peak W*r^2
        epd = xnp.einsum('WCib,tWKCi->tWKCb', ep, deta_f)       # fold all t (deta full) -> (t,W,K,C,b)
        return acc + xnp.einsum('ts,tWKCb->sWKCb', trsr, epd), ()

    src0 = xnp.zeros((order + 1, sW, sK, sC, b), carry.dtype)
    src, _ = xscan(_src_step, src0, (edge_f, trs_r))

    return (prop + src).reshape((order + 1,) + W_shape + K_shape + C_shape + (b,))


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


def compute_tau_tilde_jets_scanned(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs):
    '''EXPERIMENTAL memory-lean mirror of :py:func:`compute_tau_tilde_jets`.'''
    return _adj_sweep_scanned(left_tt_cores, xi_jets, deta_tildes, mu_jets, trs)


def compute_sigma_tilde_jets_scanned(right_tt_cores, xi_jets, deta_tildes, nu_jets, trs):
    '''EXPERIMENTAL memory-lean mirror of :py:func:`compute_sigma_tilde_jets` (reverse of tau_tilde).'''
    rev = _adj_sweep_scanned(tt_operations.tt_reverse(right_tt_cores), xi_jets[::-1],
                             deta_tildes[::-1], nu_jets[::-1], trs)
    return rev[::-1]


def compute_dxi_tilde_jets(
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
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
        from_tau = contractions.trs_dtWKCa_dCaib_dsWCb_to_drWKCi(trs, tau_tildes, down_tt_cores, nu_jets)
        from_sig = contractions.trs_drWCa_dCaib_dtWKCb_to_dsWKCi(trs, mu_jets, down_tt_cores, sigma_tildes)
        dxi_tildes = from_tau + from_sig
    else:
        def _func(data):
            O, mu, nu, st, tt = data
            # Three-group (W,K,C): tau_tilde (tt) / sigma_tilde (st) carry K, mu/nu (frame) and O (frame core)
            # do not. Both self-infer (mu/nu pin W, O pins C, K=remainder).
            from_tau = contractions.trs_tWKCa_Caib_sWCb_to_rWKCi(trs, tt, O, nu)
            from_sig = contractions.trs_rWCa_Caib_tWKCb_to_sWKCi(trs, mu, O, st)
            return (from_tau + from_sig,)

        (dxi_tildes,) = xmap(_func, (down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes))
    return dxi_tildes


def assemble_tucker_variation_jets(
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
        eta_r = contractions.dtWCa_dtWKCo_to_dKCao if sum_over_probes else contractions.dtWCa_dtWKCo_to_dWKCao
        dxi_w = contractions.duWKCa_duWo_to_dKCao if sum_over_probes else contractions.duWKCa_duWo_to_dWKCao
        return eta_r(etas, ztildes, n_probe) + dxi_w(dxi_tildes[:, :s_size], w_jets[:, :s_size])

    eta_r = contractions.tWCa_tWKCo_to_KCao if sum_over_probes else contractions.tWCa_tWKCo_to_WKCao
    dxi_w = contractions.uWKCa_uWo_to_KCao  if sum_over_probes else contractions.uWKCa_uWo_to_WKCao

    def _func(data):
        zt, dxt, eta, wj = data
        return (eta_r(eta, zt, n_probe) + dxi_w(dxt[:s_size], wj[:s_size]),)

    (dU_tildes,) = xmap(_func, (ztildes, dxi_tildes, etas, w_jets))
    return dU_tildes


def assemble_tucker_variation_jets_scanned(
        ztildes, dxi_tildes, ww, pp, etas, n_probe, sum_over_probes,
        chunk_size: typ.Optional[int] = None,
) -> NDArray:
    '''EXPERIMENTAL W-chunked mirror of :py:func:`assemble_tucker_variation_jets` (module-private,
    uniform). The milder assembly (two legs nO,N -- the dense gradient is a few GB, not the tt-core's
    hundreds), chunked over W with the same reducer seam (add if summed, concat if kept) for a uniform
    chunked-assembly interface. The only twist: ``ww``/``pp`` have no order axis, so W sits at a
    per-operand axis (2 for the order-carrying jets, 1 for ww/pp). chunk_size None / ragged / multi-W /
    small W -> dense.'''
    ops = (ztildes, dxi_tildes, ww, pp, etas)
    w_axes = (2, 2, 1, 1, 2)                          # per-operand W axis (ww/pp carry no order axis)
    dense = lambda: assemble_tucker_variation_jets(*ops, n_probe, sum_over_probes)
    if chunk_size is None or not is_ndarray(etas) or n_probe != 1:
        return dense()
    use_jax = tree_contains_jax(ops)
    xnp, _, _ = get_backend(True, use_jax)
    W = etas.shape[2]
    if W <= chunk_size:
        return dense()
    return _wchunked_reduce(
        lambda co: assemble_tucker_variation_jets(*co, n_probe, sum_over_probes),
        ops, w_axes, W, chunk_size, sum_over_probes, 1, use_jax, xnp)


def assemble_tt_variation_jets(
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
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
            f_sig, f_tau, f_det = (contractions.trs_drWCa_dsWCi_dtWKCb_to_dKCaib,
                                   contractions.trs_dtWKCa_drWCi_dsWCb_to_dKCaib,
                                   contractions.trs_drWCa_dtWKCi_dsWCb_to_dKCaib)
        else:
            f_sig, f_tau, f_det = (contractions.trs_drWCa_dsWCi_dtWKCb_to_dWKCaib,
                                   contractions.trs_dtWKCa_drWCi_dsWCb_to_dWKCaib,
                                   contractions.trs_drWCa_dtWKCi_dsWCb_to_dWKCaib)
        t_sig = f_sig(trs[:, :, :s_size], mu_jets, xi_jets[:, :s_size], sigma_tildes, n_probe)
        t_tau = f_tau(trs[:, :s_size, :], tau_tildes, xi_jets[:, :s_size], nu_jets, n_probe)
        t_det = f_det(trs,                mu_jets, deta_tildes,           nu_jets, n_probe)
        return t_sig + t_tau + t_det

    if sum_over_probes:
        f_sig, f_tau, f_det = (contractions.trs_rWCa_sWCi_tWKCb_to_KCaib,
                               contractions.trs_tWKCa_rWCi_sWCb_to_KCaib,
                               contractions.trs_rWCa_tWKCi_sWCb_to_KCaib)
    else:
        f_sig, f_tau, f_det = (contractions.trs_rWCa_sWCi_tWKCb_to_WKCaib,
                               contractions.trs_tWKCa_rWCi_sWCb_to_WKCaib,
                               contractions.trs_rWCa_tWKCi_sWCb_to_WKCaib)

    def _func(data):
        xi, mu, nu, st, tt, dt = data
        t_sig = f_sig(trs[:, :, :s_size], mu, xi[:s_size], st, n_probe)
        t_tau = f_tau(trs[:, :s_size, :], tt, xi[:s_size], nu, n_probe)
        t_det = f_det(trs,                mu, dt,           nu, n_probe)
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


def assemble_tt_variation_jets_scanned(
        sigma_tildes:   typ.Sequence[NDArray],
        tau_tildes:     typ.Sequence[NDArray],
        deta_tildes:    typ.Sequence[NDArray],
        xi_jets:        typ.Sequence[NDArray],
        mu_jets:        typ.Sequence[NDArray],
        nu_jets:        typ.Sequence[NDArray],
        trs:            NDArray,
        n_probe:        int,
        sum_over_probes: bool,
        chunk_size:     typ.Optional[int] = None,   # W-chunk size; None -> dense (no chunking)
) -> NDArray:                                        # dG_tildes supercore, [W+]C+(rLi,nUi,rRi)
    '''EXPERIMENTAL memory-lean mirror of :py:func:`assemble_tt_variation_jets` (module-private, uniform).

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
    dense = lambda: assemble_tt_variation_jets(*ops, trs, n_probe, sum_over_probes)
    if chunk_size is None or not is_ndarray(xi_jets) or n_probe != 1:
        return dense()

    use_jax = tree_contains_jax(ops + (trs,))
    xnp, _, _ = get_backend(True, use_jax)
    W = xi_jets.shape[2]                              # uniform supercore: (d, order, W, ...)
    if W <= chunk_size:
        return dense()
    return _wchunked_reduce(
        lambda co: assemble_tt_variation_jets(*co, trs, n_probe, sum_over_probes),
        ops, (2,) * len(ops), W, chunk_size, sum_over_probes, 1, use_jax, xnp)


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
        ztildes, dxi_tildes, ww, pp, eta_jets, n_probe, sum_over_probes)
    dG_tildes = assemble_tt_variation_jets(
        sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets, trs, n_probe, sum_over_probes)
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
    '''
    sweep = tv_precompute_probe_frame_sweep_jets(frame, ww, pp, order)
    return tv_probe_transpose_derivatives_from_sweep(ztildes, ww, pp, frame, sweep, order, sum_over_probes)


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
        return contractions.trs_tWKCa_Caib_rWCi_to_sWKCb(trs_xi, carry, Q, xi[:s_size]), (carry,)

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
        dxi_hats = contractions.trs_drWCa_dCaib_dtWKCb_to_dsWKCi(trs, mu_jets, down_tt_cores, sigma_hats)
        dG = contractions.trs_drWCa_dsWCi_dtWKCb_to_dKCaib if sum_over_probes else contractions.trs_drWCa_dsWCi_dtWKCb_to_dWKCaib
        dU = contractions.duWKCa_duWo_to_dKCao if sum_over_probes else contractions.duWKCa_duWo_to_dWKCao
        dG_tildes = dG(trs[:, :, :s_size], mu_jets, xi_jets[:, :s_size], sigma_hats, n_probe)
        dU_tildes = dU(dxi_hats[:, :s_size], w_jets[:, :s_size])
        return dU_tildes, dG_tildes

    def _dxi_hat(data):
        O, mu, sh = data                        # dxi_hat = mu * O * sigma_hat (mode leg free)
        return (contractions.trs_rWCa_Caib_tWKCb_to_sWKCi(trs, mu, O, sh),)

    (dxi_hats,) = xmap(_dxi_hat, (down_tt_cores, mu_jets, sigma_hats))

    dG = contractions.trs_rWCa_sWCi_tWKCb_to_KCaib if sum_over_probes else contractions.trs_rWCa_sWCi_tWKCb_to_WKCaib
    dU = contractions.uWKCa_uWo_to_KCao if sum_over_probes else contractions.uWKCa_uWo_to_WKCao

    def _asm(data):
        xi, mu, sh, dxh, wj = data
        dG_t = dG(trs[:, :, :s_size], mu, xi[:s_size], sh, n_probe)
        dU_t = dU(dxh[:s_size], wj[:s_size])
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
        ztildes, ww, pp, (tucker_cores, tt_cores, tt_cores, tt_cores), order, sum_over_probes=sum_over_probes)


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
