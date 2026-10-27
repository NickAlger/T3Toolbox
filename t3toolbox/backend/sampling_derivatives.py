# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Jet machinery: symmetric directional derivatives of probe/apply/entries, all in one place.

The ``*_derivatives`` ops strictly generalize the plain sampling ops with a leading order axis
(order 0 == the plain op). The jets of the helper chain (``compute_*_jets``) are shared across the
three sampling types, which is why the derivative layer is one module rather than three. Math:
``docs/symmetric_probe_derivatives.tex``; costs/usage: ``docs/entries_apply_probe.md``.
"""
import math
import functools
import itertools
import numpy as np
import typing as typ

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.fv_conversions as fv_conversions
import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.t3_operations as ragged_ops
import t3toolbox.backend.ut3_operations as uniform_ops
from t3toolbox.backend.probing import compute_xi
from t3toolbox.backend.entries import _entry_xis, _onehot_vectors
from t3toolbox.backend.common import *

DEFAULT_CHUNK_SIZE = 100   # default W-chunk size for the chunkable probe gradient assemblies (docs/chunking.md)

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
# order 0, direction at order 1, zero above -- and every product of jets is a binomial convolution,
#     c^(t) = sum_{r+s=t} C(t,r) a^(r) b^(s)      (the product rule on Taylor jets),
# which the derivation writes as a contraction against the binomial tensor trs[t,r,s] = C(t,r)[r+s=t].
# We call the order-stacked edge variables "jets" (order axis t; index 0 is the ordinary probing edge
# variable). The pushthrough recursion and the combine are the SAME operation -- a binomial jet-product:
#   - compute_mu_jets : left  jets mu_i^(t)  (input jet on the mode).
#   - compute_nu_jets : right jets nu_i^(t)  -- the mirror image (tt_reverse).
#   - compute_eta_jets: combine at each free mode (nu jet on the bond).
#   - assemble_z_jets : lift each order through the Tucker cores via tWCi_Cio_to_tWCo (t broadcast).
# No kernel here contracts trs: see "The jet kernels" below for the recurrence and scan forms. The dense
# trs forms are the test suite's reference twins (tests/_jet_trs_twins.py), never imported here.
#
# Provenance: the symmetric-derivative formulation is NOT part of the published T4S paper (it was cut
# long ago). It is reconstructed from an old, unvetted derivation and will be written up afresh in a
# project note; the recursions here have been verified against a dense oracle, not against the paper.
#
# STACKING: the SAME three blocks as plain probing (frame-inner W + K + C), plus the convolved order
# axis t. See docs/batching_and_stacking.md.
#   - sample stack W: on the inputs ww/pp -- each sample is a paired (X, P) (duplicate a point across W
#     to sweep many directions at it). This IS the probing probe stack W.
#   - tangent stack K: on the variation cores only -- a batch of tangent vectors sharing one frame
#     (Riemannian only). Rides the variation-derived jets (sigma/tau/deta/dxi), not the frame jets.
#   - frame/core stack C: on the cores -- a batch of T3s.
# The order axis t (the derivative orders, binomially convolved) is placed OUTERMOST;
# the passive blocks are frame-inner W + K + C. So frame jets carry order + W + C, variation jets carry
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
    'compute_mu_jets',           # recurrence / order-scan forms (the dense-trs twins are test-only)
    'compute_nu_jets',
    'compute_eta_jets',
    'assemble_z_jets',
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
    'estimate_chunk_size',            # eager helpers to choose the transpose chunk_size (docs/chunking.md)
    'max_chunk_size_within',
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
    # Corewise two-hole: the derivative of the corewise transpose along core directions (the exact
    # Hessian's sampling term). Each piece is the derivative of the GN-transpose kernel named after it.
    't3_probe_corewise_derivatives_hessian',
    't3_probe_corewise_derivatives_hessian_from_sweep',
    'compute_deta_tilde_tangent_jets',
    'compute_tau_tilde_tangent_jets',
    'compute_sigma_tilde_tangent_jets',
    'compute_dxi_tilde_tangent_jets',
    'assemble_tucker_tangent_variation_jets',
    'assemble_tt_tangent_variation_jets',
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
        order:  int,                    # highest derivative order
) -> typ.Tuple[NDArray, ...]:           # z_jets. len=d, elm_shape=(order+1,)+W+C+(Ni,)
    '''Symmetric derivatives of probing a Tucker tensor train, in one repeated direction.

    Returns, for each mode ``i``, the stack ``y_i^(t) = d^t/ds^t y_i(X + s P)|_0`` for ``t=0..order``,
    where ``y_i`` is the ``i``-th probing action. Index ``0`` is the ordinary probe ``t3_probe``.

    Two independent stacks ride through, frame-inner as in plain probing: a sample stack ``W`` on the
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

    mu_jets = compute_mu_jets(tt_cores, xi_jets, order)

    nu_jets = compute_nu_jets(tt_cores, xi_jets, order)

    eta_jets = compute_eta_jets(tt_cores, mu_jets, nu_jets)

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


def build_input_jets(
        xis:    typ.Sequence[NDArray],  # frame projected probes,        len=d, elm_shape=W+C+(nUi,)
        dxis:   typ.Sequence[NDArray],  # projected perturbation dirs,  len=d, elm_shape=W+C+(nUi,)
) -> typ.Tuple[NDArray, ...]:           # xi_jets. len=d, elm_shape=(2,)+W+C+(nUi,): order 0 = xi, 1 = dxi
    '''Input jets: stack each (value, direction) pair on a leading order axis.

    Since ``x + s p`` is affine in ``s``, an input vector's jet is just ``(x, p, 0, ...)`` -- value at
    order 0, direction at order 1, zero above. Stored at size 2 (orders 0,1), so every product with it is
    a two-term banded recurrence (:py:func:`_shift_down_pair` / :py:func:`_shift_up_pair`).
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
        dtype,                              # the sweep's dtype (the result type of its operands)
        xnp,                                # numpy or jax.numpy
) -> NDArray:                               # mu_0 jet, shape=(order+1,)+W+C+(r0,): order 0 = ones, higher = 0
    '''Leftmost left-pushthrough jet mu_0^(t): the empty product (ones) at order 0, zero above. Built in the
    operands' dtype -- a default-dtype constant would promote a float32 sweep to float64 under jax x64.'''
    ones  = xnp.ones((1,) + stack_shape + (r0,), dtype)
    zeros = xnp.zeros((order,) + stack_shape + (r0,), dtype)
    return xnp.concatenate([ones, zeros], axis=0)


# ==================================================================================================
# The jet kernels -- banded recurrences and order scans (no binomial tensor)
# ==================================================================================================
# Every jet product is a binomial convolution c^(t) = sum_{r+s=t} C(t,r) a^(r) b^(s), or its adjoint (in
# a), the binomial correlation c^(r) = sum_s C(r+s,r) b^(s) z^(r+s) (docs/symmetric_probe_derivatives.tex,
# "Recurrence and convolution realization"). The derivation writes both as contractions against the
# binomial tensor trs[t,r,s] = C(t,r)[r+s=t]; NO kernel here contracts it. trs is a sparse convolution
# tensor, and as a dense einsum operand it costs flops and peak memory both: _pairwise_path sorts it last
# (it shares one index per operand), so the intermediate balloons to the union of every index, and every
# order of the spatial product is resident at once. (The dense forms survive as the test suite's reference
# twins, tests/_jet_trs_twins.py; the package never imports them.) Two regimes:
#
# * One factor an INPUT jet (affine: orders {0,1}, build_input_jets). Only two terms survive -- a banded
#   recurrence. Forward: c^(t) = a^(t) b^(0) + t a^(t-1) b^(1); reverse: c^(r) = b^(0) z^(r) + (r+1) b^(1)
#   z^(r+1). The two shifted copies of the full jet are stacked on a pair axis (_shift_down_pair /
#   _shift_up_pair) and contracted against [b^(0), b^(1)] in ONE GEMM: mu/nu/sigma/tau, the adjoint
#   propagation, and the TT assembly's sigma~ / tau~ groups. The two-row dxi~ correlation is the same pair
#   (its two output rows are the affine leg) met by a diagonal order scan -- for memory, not for weights.
#
# * Both factors FULL jets. The input order r is scanned (one order slice of the spatial product live at a
#   time: the peak drops by the (order+1) factor), and the binomial weights come from a recurrence on the
#   OTHER factor, carried by the scan. The forward shift D x = [t x^(t-1)]_t (_shift_down) satisfies
#   D^r x = r! C(t,r) x^(t-r), so the convolution is sum_r (a^(r)/r!) (D^r b): the scan carries D^r b,
#   applies D once per step, and its xs are (a, 1/r!). The correlation is the same with the reverse shift
#   U x = [(s+1) x^(s+1)]_s (_shift_up; U^r z = r! C(r+s,r) z^(r+s)). No weight matrix, no traced shift,
#   no dynamic slice: each step is ordinary contractions plus an O(jet) shift -- eta, deta, the adjoint
#   deta~ source, and the TT assembly's deta~ group.
#
# The order axis leads every jet ((order+1,) + W + K + C + legs), so the scans run over it with NO axis
# movement; the bodies touch only the order axis and the core legs, W / K / C ride as grouped letters
# (each sub-axis shardable, docs/grouped_contractions.md). Every scan / map body is a module-level function
# (or a memoized factory keyed on hashable statics) and the loop-invariant core operands ride the carry --
# docs/contributor/scan_body_principles.md. The memory win needs the UNIFORM path, where xmap / xscan are
# the real (sequential) jax.lax.map / jax.lax.scan; see compute_eta_jets.


def _shift_down(
        jet:    NDArray,    # (order+1,)+...; axis 0 is the order axis
        xnp,                # numpy or jax.numpy
) -> NDArray:               # D jet, same shape: (D jet)^(t) = t jet^(t-1), zero at t = 0
    '''The forward Leibniz shift ``(D x)^(t) = t x^(t-1)`` -- the order-``t`` coefficient of the product
    with the affine jet ``(0, 1)``. Iterated, ``(D^r x)^(t) = r! C(t, r) x^(t-r)``: how the order scans
    get their binomial weights.'''
    n = jet.shape[0]
    t = xnp.arange(n, dtype=jet.dtype).reshape((n,) + (1,) * (jet.ndim - 1))   # a weight vector, not a batch axis
    return t * xnp.concatenate([xnp.zeros_like(jet[:1]), jet[:-1]], axis=0)


def _shift_up(
        jet:    NDArray,    # (order+1,)+...; axis 0 is the order axis
        xnp,                # numpy or jax.numpy
) -> NDArray:               # U jet, same shape: (U jet)^(s) = (s+1) jet^(s+1), zero at s = order
    '''The reverse Leibniz shift ``(U x)^(s) = (s+1) x^(s+1)``, the adjoint of :py:func:`_shift_down`.
    Iterated, ``(U^r x)^(s) = r! C(s+r, r) x^(s+r)`` (zero once ``s + r > order``).'''
    n = jet.shape[0]
    s1 = xnp.arange(1, n + 1, dtype=jet.dtype).reshape((n,) + (1,) * (jet.ndim - 1))   # likewise
    return s1 * xnp.concatenate([jet[1:], xnp.zeros_like(jet[:1])], axis=0)


def _shift_down_pair(
        jet:    NDArray,    # (order+1,)+...
        xnp,                # numpy or jax.numpy
        axis:   int = 0,    # where the new pair axis goes: 0 (pair leading) or 1 (after the order axis)
) -> NDArray:               # [jet, D jet] stacked on a new length-2 axis
    '''The forward banded pair ``[x^(t), t x^(t-1)]``: contracted against an affine jet ``[b^(0), b^(1)]``
    it is the whole convolution ``sum_{s<2} C(t, t-s) x^(t-s) b^(s)``.'''
    return xnp.stack([jet, _shift_down(jet, xnp)], axis=axis)


def _shift_up_pair(
        jet:    NDArray,    # (order+1,)+...
        xnp,                # numpy or jax.numpy
        axis:   int = 0,    # where the new pair axis goes: 0 (pair leading) or 1 (after the order axis)
) -> NDArray:               # [jet, U jet] stacked on a new length-2 axis
    '''The reverse banded pair ``[x^(s), (s+1) x^(s+1)]``: contracted against an affine jet it is the whole
    correlation ``sum_{u<2} C(s+u, u) x^(s+u) b^(u)`` (the adjoint of the forward pair).'''
    return xnp.stack([jet, _shift_up(jet, xnp)], axis=axis)


def _inverse_factorials(
        order:  int,        # highest derivative order
        like:   NDArray,    # an array of the target dtype (the scanned jet)
        xnp,                # numpy or jax.numpy
) -> NDArray:               # shape=(order+1,): 1/r!, scanned beside the jet whose order r is summed
    '''``1/r!`` for ``r = 0..order``: the scanned factor's scale in the order scans (the carried factor's
    ``D^r`` / ``U^r`` supplies the other ``r! C(.,r)``). A small folded constant.'''
    return xnp.asarray([1.0 / math.factorial(r) for r in range(order + 1)], dtype=like.dtype)


def _mu_jets_step(
        mu_jet: NDArray,                       # carry: (order+1,)+W+C+(rLi,); axis 0 is the order axis
        data:   typ.Tuple[NDArray, NDArray],   # (G, xi_jet) for one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:   # (next carry, (mu_jet,))
    '''One core of the fused recurrence of :py:func:`compute_mu_jets`. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((mu_jet, data)))   # only xnp; it ignores the flag
    order = mu_jet.shape[0] - 1                            # the carry is stacked over derivative orders
    s_size = min(2, order + 1)                             # affine input jet: orders {0, 1}

    G, xi_jet = data
    Gxi = contractions.contract('Caib,sWCi->sWCab', G, xi_jet[:s_size])     # (s, W, C, a, b), s in {0,1}
    if s_size > 1:                                         # static branch (order >= 1)
        next_mu = contractions.contract('stWCa,sWCab->tWCb', _shift_down_pair(mu_jet, xnp), Gxi)
    else:                                                  # order 0: only s=0 survives
        next_mu = contractions.contract('tWCa,WCab->tWCb', mu_jet, Gxi[0])
    return next_mu, (mu_jet,)


def compute_mu_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        order:      int,                    # highest derivative order (the input jets carry only orders 0, 1)
) -> typ.Tuple[NDArray, ...]:               # mu_jets. len=d, elm_shape=(order+1,)+W+C+(rLi,). mu_jets[i][t]=mu_{i-1}^(t)
    '''Left derivative-pushthrough jets (fused banded recurrence).

    ``mu_jets[i]`` is the left edge variable entering core ``i`` (``mu_{i-1}``), stacked over derivative
    orders. The affine input jet (``xi`` nonzero only at orders ``s in {0,1}``) collapses each binomial
    pushthrough to a two-term recurrence ``mu_i^(t) = mu^(t).G.xi^(0) + t * mu^(t-1).G.xi^(1)`` -- no
    binomial tensor, no ``(order+1)^2`` work. The two terms are folded into ONE contraction per core:
    stack ``[mu^(t), t * mu^(t-1)]`` on a jet-pair axis ``s`` and contract it together with the bond ``a``
    against ``[G.xi^(0), G.xi^(1)]`` (``'stWCa,sWCab->tWCb'``) -- one larger GEMM for XLA to schedule.
    ``order`` is an argument because nothing else carries it (the input jets stop at order 1). The
    per-core step is :py:func:`_mu_jets_step`.
    '''
    use_jax = tree_contains_jax((tt_cores, xi_jets))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    stack_shape = xi_jets[0].shape[1:-1]     # full W + C batch (W outer, C inner); either may be empty
    r0 = tt_cores[0].shape[-3]
    init = _init_jet(order, stack_shape, r0, xnp.result_type(tt_cores[0], xi_jets[0]), xnp)

    _, (mu_jets,) = xscan(_mu_jets_step, init, (tt_cores, xi_jets))
    return mu_jets


def compute_nu_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        xi_jets:    typ.Sequence[NDArray],  # input jets,  len=d, elm_shape=(2,)+W+C+(nUi,)
        order:      int,                    # highest derivative order
) -> typ.Tuple[NDArray, ...]:               # nu_jets. len=d, elm_shape=(order+1,)+W+C+(rR(i+1),). nu_jets[i][t]=nu_i^(t)
    '''Right derivative-pushthrough jets (banded recurrence).

    The mirror image of :py:func:`compute_mu_jets`: reverse the tensor train (``tt_reverse`` swaps
    bonds and core order), run the left banded-recurrence sweep, reverse the result. ``nu_jets[i]`` is
    the right edge variable entering core ``i`` (``nu_i``), stacked over derivative orders.
    '''
    # Polymorphic reverse: the uniform tt_reverse keeps the supercore (tt_operations.tt_reverse would iterate
    # the supercore's d axis -- the unroll trap). The jet slices [::-1] just reverse the leading d axis.
    reverse = tt_operations.tt_reverse
    rev_nu_jets = compute_mu_jets(reverse(tt_cores), xi_jets[::-1], order)
    return rev_nu_jets[::-1]


def _eta_jets_order_step(
        carry:  typ.Tuple[NDArray, NDArray, NDArray],  # (eta, D^r nu, G): (order+1,)+W+C+(nOi,) ; (order+1,)+W+C+(rR(i+1),) ; C+(rLi,nOi,rR(i+1))
        xs:     typ.Tuple[NDArray, NDArray],           # (mu^(r), 1/r!): W+C+(rLi,) ; ()
) -> typ.Tuple[typ.Tuple[NDArray, NDArray, NDArray], typ.Tuple]:   # (next carry, ())
    '''One input order ``r`` of the eta convolution: ``eta += (mu^(r)/r! . G) . D^r nu``, then ``D`` the
    carried right jet. Closure-free scan body (the core ``G`` rides the carry).'''
    eta, nu_r, G = carry
    mu_r, w_r = xs
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, xs)))      # only xnp
    MG_r = contractions.contract('WCa,Caib->WCib', w_r * mu_r, G)      # one order slice: peak W*r^2
    eta = eta + contractions.contract('WCib,sWCb->sWCi', MG_r, nu_r)  # nu_r is indexed by the output order
    return (eta, _shift_down(nu_r, xnp), G), ()


def _eta_jets_core(
        data:       typ.Tuple[NDArray, NDArray, NDArray],  # (mu_jet, G, nu_jet), one core
        is_uniform: bool,                       # inner-scan backend; see compute_eta_jets on the memory win
) -> typ.Tuple[NDArray]:                        # (eta_jet,) elm_shape=(order+1,)+W+C+(nOi,)
    '''One core of the order-scan convolution of :py:func:`compute_eta_jets`.'''
    mu_jet, G, nu_jet = data                              # (T,W,C,a) ; (C,a,i,b) ; (T,W,C,b)
    xnp, _, xscan = get_backend(is_uniform, tree_contains_jax(data))
    order = mu_jet.shape[0] - 1
    eta0 = xnp.zeros((order + 1,) + nu_jet.shape[1:-1] + (G.shape[-2],), xnp.result_type(mu_jet, G, nu_jet))
    (eta, _, _), _ = xscan(_eta_jets_order_step, (eta0, nu_jet, G),
                           (mu_jet, _inverse_factorials(order, mu_jet, xnp)))
    return (eta,)


def _eta_jets_step_uniform(data):   # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Uniform variant of :py:func:`_eta_jets_core`: the inner order loop is a real scan, which is what
    makes the memory win in :py:func:`compute_eta_jets`.'''
    return _eta_jets_core(data, True)


def _eta_jets_step_ragged(data):    # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Ragged variant of :py:func:`_eta_jets_core`: the inner order loop is a Python loop.'''
    return _eta_jets_core(data, False)


def compute_eta_jets(
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
) -> typ.Tuple[NDArray, ...]:               # eta_jets. len=d, elm_shape=(order+1,)+W+C+(nOi,)
    '''Combine the left and right jets at each free mode (order-scan convolution).

    ``eta_i^(t) = sum_{r+s=t} C(t,r) mu_{i-1}^(r) . G_i . nu_i^(s)``. eta is a FULL convolution (both jets
    run ``0..order``): formed all at once, the spatial product ``mu . G`` has mode AND bond live at every
    order -- an ``(order+1)*W*r^2`` intermediate (``d`` times that vectorized over the cores: 128 GB at
    r=128, W=32000, order 5). This scans the two small axes instead, so only one slice is ever live:

        ``eta = sum_r (mu^(r)/r! . G) . D^r nu``,   ``(D x)^(t) = t x^(t-1)``,

    - ``xscan`` over the input order ``r`` (the convolution): forms ``mu^(r) . G`` one order at a time and
      folds in the whole right jet, whose binomial weights the carried ``D^r nu`` supplies (``D^r`` is
      ``r! C(t, r)`` times the shift by ``r``) -- removes the ``(order+1)`` factor;
    - ``xmap`` over cores: removes the ``d`` factor.

    **The win requires the UNIFORM path** (a stacked supercore), where ``xmap``/``xscan`` dispatch to the
    real ``jax.lax.map`` / ``jax.lax.scan`` -- both *sequential* (``lax.map`` is a ``scan``, not a
    vectorizing ``vmap``), so only one core+order intermediate is resident. On the RAGGED path
    ``xmap``/``xscan`` are Python loops that unroll under jit and keep everything co-resident. Uniform, both
    scans: measured **14-28x** smaller XLA peak than the vectorized dense form and CONSTANT in order. The
    order loop being a real scan is what forces this -- an unrolled loop does not. Equal to the dense
    binomial form (the test suite's twin) to ~1e-16.
    '''
    use_jax = tree_contains_jax((tt_cores, mu_jets, nu_jets))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)
    step = _eta_jets_step_uniform if is_uniform else _eta_jets_step_ragged

    (eta_jets,) = xmap(step, (mu_jets, tt_cores, nu_jets))
    return eta_jets


def _z_jets_step(
        data: typ.Tuple[NDArray, NDArray],  # (eta_jet, U) for one core
) -> typ.Tuple[NDArray]:                    # (z_jet,) elm_shape=(order+1,)+W+C+(Ni,)
    '''One core of the Tucker lift of :py:func:`assemble_z_jets` (ragged path only). Closure-free map
    body -- ``docs/contributor/scan_body_principles.md``.'''
    eta_jet, U = data
    return (contractions.contract('tWCi,Cio->tWCo', eta_jet, U),)


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
        # d-prefixed jet lift; the ragged xmap below is the oracle. The order axis rides passively
        # through the C-only tucker supercore, keeping its own einsum letter (never folded into W).
        z_jets = contractions.contract('dtWCi,dCio->dtWCo', eta_jets, tucker_cores)
    else:
        (z_jets,) = xmap(_z_jets_step, (eta_jets, tucker_cores))
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
    mu_jets = compute_mu_jets(left_tt_cores, xi_jets, order)
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
    mu_jets  = compute_mu_jets(left_tt_cores, xi_jets, order)
    nu_jets  = compute_nu_jets(right_tt_cores, xi_jets, order)
    eta_jets = compute_eta_jets(down_tt_cores, mu_jets, nu_jets)
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
# core operand is a variation core). Stacks t + W + K + C, frame-inner. The transpose is a separate slice.


def _zero_jet(
        order:        int,                  # highest derivative order
        stack_shape:  typ.Tuple[int, ...],  # full leading batch the variation jets carry (W + K + C)
        r:            int,                  # bond dimension
        dtype,                              # the sweep's dtype (the result type of its operands)
        xnp,                                # numpy or jax.numpy
) -> NDArray:                               # shape=(order+1,)+W+K+C+(r,): all orders zero
    '''All-zero jet -- the sigma_0 / tau_d boundary of the variation sweeps (no order-0 ones), in the operands'
    dtype (see :py:func:`_init_jet`).'''
    return xnp.zeros((order + 1,) + stack_shape + (r,), dtype)


def _sigma_banded_step(sigma_jet, Q, O, dG, xi_jet, dxi_jet, mu_jet, xnp):
    '''One K-aware banded step of the sigma recursion: the three affine pushthroughs
    ``sigma Q(xi) + mu dG(xi) + mu O(dxi)`` as fused two-term recurrences. W, K, C ride unflattened
    through the grouped contractions (``len_C`` supplied where the W|C split is unpinned); K rides on the
    carried sigma (t1), the variation core dG (t2), or the var input dxi (t3).'''
    order = sigma_jet.shape[0] - 1                # the carry is stacked over derivative orders
    s_size = min(2, order + 1)                    # affine input jets: orders {0, 1}
    nf = len(Q.shape[:-3])
    xi_s, dxi_s = xi_jet[:s_size], dxi_jet[:s_size]

    Qxi = contractions.contract('Caib,sWCi->sWCab', Q, xi_s)                 # (s,W,C,a,b)   -- no K
    dGxi = contractions.contract('KCaib,sWCi->sWKCab', dG, xi_s, len_C=nf)   # (s,W,K,C,a,b) -- K on core
    Odxi = contractions.contract('Caib,sWKCi->sWKCab', O, dxi_s)             # (s,W,K,C,a,b) -- K on var input

    if s_size > 1:
        mu_pair = _shift_down_pair(mu_jet, xnp)
        t1 = contractions.contract('stWKCa,sWCab->tWKCb', _shift_down_pair(sigma_jet, xnp), Qxi, len_C=nf)
        t2 = contractions.contract('stWCa,sWKCab->tWKCb', mu_pair, dGxi, len_C=nf)     # K on core
        t3 = contractions.contract('stWCa,sWKCab->tWKCb', mu_pair, Odxi, len_C=nf)     # K on var input
    else:                                                                    # order 0: only s=0
        t1 = contractions.contract('tWKCa,WCab->tWKCb', sigma_jet, Qxi[0], len_C=nf)
        t2 = contractions.contract('tWCa,WKCab->tWKCb', mu_jet, dGxi[0], len_C=nf)
        t3 = contractions.contract('tWCa,WKCab->tWKCb', mu_jet, Odxi[0], len_C=nf)
    return t1 + t2 + t3


def _sigma_jets_step(
        sigma_jet:  NDArray,                    # carry: (order+1,)+W+K+C+(rRi,); axis 0 is the order axis
        data:       typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray],  # (Q, O, dG, xi_jet, dxi_jet, mu_jet), one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:    # (next carry, (sigma_jet,))
    '''One core of the banded recurrence of :py:func:`compute_sigma_jets`. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((sigma_jet, data)))   # only xnp; it ignores the flag
    Q, O, dG, xi_jet, dxi_jet, mu_jet = data
    return _sigma_banded_step(sigma_jet, Q, O, dG, xi_jet, dxi_jet, mu_jet, xnp), (sigma_jet,)


def compute_sigma_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        mu_jets:        typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
) -> typ.Tuple[NDArray, ...]:                   # sigma_jets. len=d, elm_shape=(order+1,)+W+K+C+(rRi,)
    '''Variation-leftward edge-variable jets sigma (banded recurrence; the jet-ified Algorithm-7 sigma
    recursion).

    ``sigma_i = sigma_{i-1} Q_i(xi_i) + mu_{i-1} dG_i(xi_i) + mu_{i-1} O_i(dxi_i)``, boundary
    ``sigma_0 = 0`` (all orders). The three-pushthrough tangent analog of :py:func:`compute_mu_jets`: both
    input jets (``xi``, ``dxi``) are affine (size 2), so each pushthrough is a fused two-term recurrence.
    The K tangent stack rides on the carried sigma / the variation core / the var input. The order is read
    off ``mu_jets``. The per-core step is :py:func:`_sigma_jets_step`.
    '''
    use_jax = tree_contains_jax((var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets))
    xnp, xmap, xscan = get_backend(is_ndarray(var_tt_cores), use_jax)
    order = mu_jets[0].shape[0] - 1

    # carry sigma is W+K+C; take the leading stack from dxi_jets (which carry K), not xi_jets (W+C only)
    stack_shape = dxi_jets[0].shape[1:-1]
    rR0 = right_tt_cores[0].shape[-3]
    dtype = xnp.result_type(*(a[0] for a in (var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets)))
    init = _zero_jet(order, stack_shape, rR0, dtype, xnp)
    _, (sigma_jets,) = xscan(_sigma_jets_step, init, (right_tt_cores, down_tt_cores, var_tt_cores, xi_jets, dxi_jets, mu_jets))
    return sigma_jets


def compute_tau_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        nu_jets:        typ.Sequence[NDArray],  # frame right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
) -> typ.Tuple[NDArray, ...]:                   # tau_jets. len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
    '''Variation-rightward edge-variable jets tau (banded recurrence) -- the mirror of
    :py:func:`compute_sigma_jets`: reverse the train (P in the Q-slot, O and dG reversed), run the sigma
    sweep, reverse the result.'''
    reverse = tt_operations.tt_reverse
    rev = compute_sigma_jets(
        reverse(var_tt_cores), reverse(left_tt_cores),
        reverse(down_tt_cores), xi_jets[::-1], dxi_jets[::-1], nu_jets[::-1],
    )
    return rev[::-1]


def _deta_jets_order_step(
        carry:  typ.Tuple[NDArray, ...],       # (deta, D^r nu, D^r tau, P, Q, dG); the cores ride unchanged
        xs:     typ.Tuple[NDArray, NDArray, NDArray],  # (mu^(r), sigma^(r), 1/r!): W+C+(rLi,) ; W+K+C+(rRi,) ; ()
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple]:   # (next carry, ())
    '''One input order ``r`` of the deta convolution ``sigma Q nu + mu dG nu + mu P tau``: the left factors'
    order-``r`` slices meet the carried ``D^r nu`` / ``D^r tau``, which are then shifted. The two terms
    sharing ``nu`` are summed before their fold. Closure-free scan body.'''
    deta, nu_r, tau_r, P, Q, dG = carry
    mu_r, sig_r, w_r = xs
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, xs)))      # only xnp
    nf = Q.ndim - 3                                                    # Q is C-only: pins len(C)
    mw = w_r * mu_r
    mg = (contractions.contract('WKCa,Caib->WKCib', w_r * sig_r, Q)            # sigma Q  (K on sigma)
          + contractions.contract('WCa,KCaib->WKCib', mw, dG, len_C=nf))       # mu dG    (K on core)
    mg3 = contractions.contract('WCa,Caib->WCib', mw, P)                       # mu P     (K on tau)
    deta = (deta + contractions.contract('WKCib,sWCb->sWKCi', mg, nu_r, len_C=nf)
            + contractions.contract('WCib,sWKCb->sWKCi', mg3, tau_r, len_C=nf))
    return (deta, _shift_down(nu_r, xnp), _shift_down(tau_r, xnp), P, Q, dG), ()


def _deta_jets_core(
        data:       typ.Tuple[NDArray, NDArray, NDArray, NDArray,
                              NDArray, NDArray, NDArray],  # (P, Q, dG, mu_jet, nu_jet, sigma_jet, tau_jet), one core
        is_uniform: bool,                       # inner-scan backend; see compute_deta_jets on the memory win
) -> typ.Tuple[NDArray]:                        # (deta_jet,) elm_shape=(order+1,)+W+K+C+(nUi,)
    '''One core of the order-scan convolution of :py:func:`compute_deta_jets`.'''
    P, Q, dG, mu_jet, nu_jet, sigma_jet, tau_jet = data
    xnp, _, xscan = get_backend(is_uniform, tree_contains_jax(data))
    order = mu_jet.shape[0] - 1                   # the jets are stacked over derivative orders
    deta0 = xnp.zeros((order + 1,) + sigma_jet.shape[1:-1] + (Q.shape[-2],),                 # W+K+C off sigma
                      xnp.result_type(P, Q, dG, mu_jet, nu_jet, sigma_jet, tau_jet))
    (deta, _, _, _, _, _), _ = xscan(_deta_jets_order_step, (deta0, nu_jet, tau_jet, P, Q, dG),
                                     (mu_jet, sigma_jet, _inverse_factorials(order, mu_jet, xnp)))
    return (deta,)


def _deta_jets_step_uniform(data):  # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Uniform variant of :py:func:`_deta_jets_core`: the inner order loop is a real scan, which is what
    makes the memory win in :py:func:`compute_deta_jets`.'''
    return _deta_jets_core(data, True)


def _deta_jets_step_ragged(data):   # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Ragged variant of :py:func:`_deta_jets_core`: the inner order loop is a Python loop.'''
    return _deta_jets_core(data, False)


def compute_deta_jets(
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_jets:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rRi,)
        tau_jets:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rL(i+1),)
) -> typ.Tuple[NDArray, ...]:                   # deta_jets. len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
    '''Tangent combine at each free mode (order-scan convolution; the jet-ified Algorithm-7 deta combine).

    The three-term tangent analog of :py:func:`compute_eta_jets`: ``deta_i = sigma Q nu + mu dG nu +
    mu P tau``, three FULL convolutions (mode ``i`` free). Formed at once, the spatial product is
    ``(order+1)*W*K*r^2``; this scans the left jets' input order ``r`` (one order slice of ``jetL . core``
    live at a time) against the carried, shifted right jets ``D^r nu`` / ``D^r tau`` -- peak ``W*K*r^2``.
    The K tangent stack sits on a *different* operand in each term (sigma / dG / tau), so several of the
    grouped contractions need ``len_C`` (the W|C split is not pinned by their operands). Memory win needs
    the uniform path (real ``lax.map``/``lax.scan``); see :py:func:`compute_eta_jets`.
    '''
    use_jax = tree_contains_jax((var_tt_cores, left_tt_cores, right_tt_cores,
                                 mu_jets, nu_jets, sigma_jets, tau_jets))
    is_uniform = is_ndarray(var_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)
    step = _deta_jets_step_uniform if is_uniform else _deta_jets_step_ragged

    (deta_jets,) = xmap(step, (left_tt_cores, right_tt_cores, var_tt_cores,
                               mu_jets, nu_jets, sigma_jets, tau_jets))
    return deta_jets


def _tangent_z_jets_step(
        data: typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (U, dU, eta_jet, deta_jet) for one core
) -> typ.Tuple[NDArray]:                                      # (z_jet,) elm_shape=(order+1,)+W+K+C+(Ni,)
    '''One core of the tangent lift of :py:func:`assemble_tangent_z_jets` (ragged path only).
    Closure-free map body -- ``docs/contributor/scan_body_principles.md``.'''
    U, dU, eta_jet, deta_jet = data
    # Three-group (W,K,C): deta carries K (in the lift via the C-only U, W and K ride passively); eta is W+C and dU is
    # the variation core (K+C), so the eta-lift needs len(C) -- recovered from the C-only U.
    n_frame = U.ndim - 2
    return (contractions.contract('tWKCi,Cio->tWKCo', deta_jet, U)
            + contractions.contract('tWCi,KCio->tWKCo', eta_jet, dU, len_C=n_frame),)


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
        # d-prefixed jet lifts; the ragged xmap below is the oracle. deta carries K (lifted via the
        # C-only U supercore, where W and K ride passively); eta is W+C lifted through the K+C variation core dU,
        # so the eta-lift needs len(C) = n_frame, read off the C-only U supercore (d,)+C+(nU,N).
        n_frame = tucker_cores.ndim - 3
        term1 = contractions.contract('dtWKCi,dCio->dtWKCo', deta_jets, tucker_cores)
        term2 = contractions.contract('dtWCi,dKCio->dtWKCo', eta_jets, var_tucker_cores, len_C=n_frame)
        z_jets = term1 + term2
    else:
        (z_jets,) = xmap(_tangent_z_jets_step, (tucker_cores, var_tucker_cores, eta_jets, deta_jets))
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
        order:      int,                    # highest derivative order
) -> typ.Tuple[NDArray, ...]:               # z_jets. len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
    '''Variation half of :py:func:`tv_probe_derivatives` from a precomputed frame ``sweep``: the
    variation sweep (sigma/tau/deta jets via the variation cores) + the lift, reusing the frame
    ``(xi, mu, nu, eta)_jets``. The reuse hook for a fitting inner solve (frame fixed across J / J^T).'''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    var_tucker_cores, var_tt_cores = variation
    xi_jets, mu_jets, nu_jets, eta_jets = sweep

    dxi_jets = build_input_jets(compute_xi(var_tucker_cores, ww), compute_xi(var_tucker_cores, pp))
    sigma_jets = compute_sigma_jets(var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets)
    tau_jets   = compute_tau_jets(var_tt_cores, left_tt_cores, down_tt_cores, xi_jets, dxi_jets, nu_jets)
    deta_jets  = compute_deta_jets(var_tt_cores, left_tt_cores, right_tt_cores, mu_jets, nu_jets, sigma_jets, tau_jets)
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
        order:      int,                    # highest derivative order
) -> typ.Tuple[NDArray, ...]:               # z_jets. len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
    '''Symmetric derivatives of probing a tangent vector, in one repeated direction (Riemannian J^(s)).

    The probe-derivative analog of :py:func:`probing.tv_probe`: returns, for each mode ``i``, the
    stack ``y_i^(t) = d^t/ds^t [J^(s) v]_i (X + s P)|_0`` for ``t=0..order``, where ``v`` is the tangent
    vector represented by ``(frame, variation)``. Index ``0`` is the ordinary tangent probe.

    Identical structure to :py:func:`t3_probe_derivatives` but on the tangent calculus: a frame sweep
    (frame cores) and a variation sweep. Full ``order + W + K + C`` stacking (frame-inner, mirroring
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


def _apply_derivatives_t3_step(
        mu_jet: NDArray,                                # carry: (order+1,)+W+C+(rLi,); axis 0 is the order axis
        data:   typ.Tuple[NDArray, NDArray],            # (G, xi_jet) for one core
) -> typ.Tuple[NDArray, typ.Tuple[int]]:                # (next carry, (0,) -- no per-core output)
    '''One core of the terminal-carry left sweep of :py:func:`_apply_derivatives_t3_from_xi_jets`: the
    banded recurrence of :py:func:`_mu_jets_step`, keeping no per-core output. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    return _mu_jets_step(mu_jet, data)[0], (0,)


def _apply_derivatives_t3_from_xi_jets(
        xi_jets:    typ.Sequence[NDArray],  # input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        tt_cores:   typ.Sequence[NDArray],  # len=d, elm_shape=C+(rLi,nUi,rR(i+1))
        order:      int,                    # highest derivative order
) -> NDArray:                               # apply-derivative jets, shape=(order+1,)+W+C
    '''Terminal mu-jet carry of the left sweep (via the cores), bond summed -- the all-modes Euclidean
    tail shared by t3_apply_derivatives and t3_entries_derivatives (they differ only in how xi_jets is
    formed). Returns ``(order+1,) + W + C`` (no K for a plain T3).
    '''
    use_jax = tree_contains_jax((tt_cores, xi_jets))
    is_uniform = is_ndarray(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)   # scan-style: xscan strips d

    stack_shape = xi_jets[0].shape[1:-1]                # W + C
    r0 = tt_cores[0].shape[-3]
    init = _init_jet(order, stack_shape, r0, xnp.result_type(tt_cores[0], xi_jets[0]), xnp)
    mu_terminal, _ = xscan(_apply_derivatives_t3_step, init, (tt_cores, xi_jets))
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
    return _apply_derivatives_t3_from_xi_jets(xi_jets, tt_cores, order)


def _apply_derivatives_tv_step(
        sigma_jet:  NDArray,                    # carry: (order+1,)+W+K+C+(rRi,); axis 0 is the order axis
        data:       typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray],  # (Q, O, dG, xi_jet, dxi_jet, mu_jet), one core
) -> typ.Tuple[NDArray, typ.Tuple[int]]:        # (next carry, (0,) -- no per-core output)
    '''One core of the terminal-carry sigma sweep of :py:func:`_apply_derivatives_from_jets`: the banded
    recurrence of :py:func:`_sigma_jets_step`, keeping no per-core output. Closure-free scan body --
    ``docs/contributor/scan_body_principles.md``.'''
    return _sigma_jets_step(sigma_jet, data)[0], (0,)


def _apply_derivatives_from_jets(
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        dxi_jets:       typ.Sequence[NDArray],  # var  input jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        mu_jets:        typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        var_tt_cores:   typ.Sequence[NDArray],  # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1))
) -> NDArray:                                   # apply-derivative jets, shape=(order+1,)+W+K+C
    '''Run the K-aware perturbation sigma-jet sweep to its TERMINAL carry and contract the final bond.

    The jet analog of :py:func:`apply._apply_from_xis`: the banded recurrence of
    :py:func:`compute_sigma_jets` (so it is ``W+K+C``-stacked), keeping only the terminal carry; the order
    is read off ``mu_jets``. Returns ``(order+1,) + W + K + C``.
    '''
    use_jax = tree_contains_jax((xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores))
    is_uniform = is_ndarray(right_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)  # scan-style: xscan strips d
    order = mu_jets[0].shape[0] - 1

    stack_shape = dxi_jets[0].shape[1:-1]              # W + K + C (dxi carries K)
    rR0 = right_tt_cores[0].shape[-3]
    dtype = xnp.result_type(*(a[0] for a in (var_tt_cores, right_tt_cores, down_tt_cores, xi_jets, dxi_jets, mu_jets)))
    init = _zero_jet(order, stack_shape, rR0, dtype, xnp)
    sigma_terminal, _ = xscan(_apply_derivatives_tv_step, init,
                              (right_tt_cores, down_tt_cores, var_tt_cores, xi_jets, dxi_jets, mu_jets))
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
    dxi_jets = build_input_jets(compute_xi(var_tucker_cores, ww), compute_xi(var_tucker_cores, pp))
    return _apply_derivatives_from_jets(
        xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores)


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
    return _apply_derivatives_t3_from_xi_jets(xi_jets, tt_cores, order)


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
    dxi_jets = build_input_jets(_entry_xis(var_tucker_cores, index), compute_xi(var_tucker_cores, pp))
    return _apply_derivatives_from_jets(
        xi_jets, dxi_jets, mu_jets, right_tt_cores, down_tt_cores, var_tt_cores)


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
# every forward contraction in the Lagrangian is replaced by its binomial jet convolution, and
# stationarity d L / d (state, variation) = 0 gives the adjoint sweeps and the gradient assembly --
# which is exactly the verified non-derivative transpose (probing.compute_*_tildes / assemble_*) with
# each contraction swapped for its ADJOINT, the binomial CORRELATION (the multiplier's order is summed,
# the swept order is freed) for the sweeps, and the ORDER-LESS three-jet product (TT, 3 edges) / plain
# order-sum (Tucker, 1 edge) for the assembly. Each is realized as in the forward (the jet-kernel note
# above): banded reverse recurrences where one factor is an input jet, order scans with a carried
# up-shifted jet where both are full. Stacks (frame-inner W + C, as in plain probing): sample stack W on
# the inputs, frame stack C on the cores; sum_over_probes sums W (the J^T r back-projection) else keeps
# it (W rides into the variation stack). Verified against the adjoint identity.

def _deta_tilde_jets_step(
        data: typ.Tuple[NDArray, NDArray],  # (U, ztilde) for one core
) -> typ.Tuple[NDArray]:                    # (deta_tilde,) elm_shape=(order+1,)+W+K+C+(nUi,)
    '''One core of the adjoint-up lift of :py:func:`compute_deta_tilde_jets` (ragged path only).
    Closure-free map body -- ``docs/contributor/scan_body_principles.md``.'''
    U, zt = data
    return (contractions.contract('tWKCo,Cio->tWKCi', zt, U),)


def compute_deta_tilde_jets(
        up_tucker_cores:    typ.Sequence[NDArray],  # U.  len=d, elm_shape=C+(nUi,Ni)
        ztildes:            typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
) -> typ.Tuple[NDArray, ...]:                       # deta_tildes. len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
    '''Adjoint-up edge-variable jets: ``deta_tilde_i = U_i r_i`` (contract the ambient mode, order
    diagonal). The 1-internal-edge (Tucker) case -- the order axis just rides through (no binomial weight). The
    residual carries the tangent stack K (the forward output's K), which rides through.'''
    use_jax = tree_contains_jax((up_tucker_cores, ztildes))
    is_uniform = is_ndarray(up_tucker_cores)
    xnp, xmap, _ = get_backend(is_uniform, use_jax)

    if is_uniform:
        # d-prefixed adjoint lift; the ragged xmap is the oracle. The order axis rides passively
        # through the C-only U supercore, keeping its own einsum letter (t, W and K are never folded).
        deta_tildes = contractions.contract('dtWKCo,dCio->dtWKCi', ztildes, up_tucker_cores)
    else:
        (deta_tildes,) = xmap(_deta_tilde_jets_step, (up_tucker_cores, ztildes))
    return deta_tildes


def _adj_prop(
        carry:  NDArray,    # adjoint jet entering the core, (order+1,)+W+K+C+(rLi,)
        P:      NDArray,    # frame core, C+(rLi,nUi,rL(i+1))
        xi:     NDArray,    # input jet,  (2,)+W+C+(nUi,)
        xnp,                # numpy or jax.numpy
) -> NDArray:               # (order+1,)+W+K+C+(rL(i+1),)
    '''The adjoint of the affine pushthrough (the propagation term of every adjoint sweep): the two-term
    REVERSE recurrence ``prop^(s) = carry^(s) P xi^(0) + (s+1) carry^(s+1) P xi^(1)``, fused into one GEMM
    over the reverse banded pair :py:func:`_shift_up_pair`. W, K, C ride unflattened (carry carries K;
    xi/P do not; ``len_C`` read off the C-only ``P``).'''
    order = carry.shape[0] - 1                 # the carry is stacked over derivative orders
    s_size = min(2, order + 1)                 # affine input jet (xi): orders {0, 1}
    nf = P.ndim - 3
    Pxi = contractions.contract('Caib,kWCi->kWCab', P, xi[:s_size])   # (k=s_size, W, C, a, b)
    if s_size > 1:
        return contractions.contract('ksWKCa,kWCab->sWKCb', _shift_up_pair(carry, xnp), Pxi, len_C=nf)
    return contractions.contract('sWKCa,WCab->sWKCb', carry, Pxi[0], len_C=nf)


def _adj_src_order_step(
        carry:  typ.Tuple[NDArray, NDArray, NDArray],  # (src, U^r deta~, P): (order+1,)+W+K+C+(rL(i+1),) ; (order+1,)+W+K+C+(nUi,) ; C+(rLi,nUi,rL(i+1))
        xs:     typ.Tuple[NDArray, NDArray],           # (edge^(r), 1/r!): W+C+(rLi,) ; ()
) -> typ.Tuple[typ.Tuple[NDArray, NDArray, NDArray], typ.Tuple]:   # (next carry, ())
    '''One edge order ``r`` of the adjoint deta~ source, a full binomial CORRELATION
    ``src^(s) = sum_r C(r+s, r) (edge^(r) P) . deta~^(r+s)``: the order-``r`` slice of ``edge P`` (peak
    ``W*r^2``) meets the carried ``U^r deta~``, which is then up-shifted. Closure-free scan body.'''
    src, deta_r, P = carry
    edge_r, w_r = xs
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, xs)))      # only xnp
    nf = P.ndim - 3
    ep = contractions.contract('WCa,Caib->WCib', w_r * edge_r, P)
    src = src + contractions.contract('WCib,tWKCi->tWKCb', ep, deta_r, len_C=nf)   # deta_r indexed by the output order
    return (src, _shift_up(deta_r, xnp), P), ()


def _adj_tilde_step(carry, P, xi, deta_t, edge, is_uniform):
    '''One K-aware step of the memory-lean adjoint sweep (the shared core of
    :py:func:`_adj_sweep_scanned_step_uniform` / :py:func:`_adj_sweep_scanned_step_ragged`). Two terms,
    both the TRANSPOSE of a forward contraction: **prop** (adjoint of the affine pushthrough,
    :py:func:`_adj_prop`) is a two-term REVERSE recurrence, fused into one GEMM; **src** (adjoint of the
    full combine, the deta_tilde source) is a full REVERSE convolution, an inner order-scan over the edge
    order ``r`` (:py:func:`_adj_src_order_step`) with peak ``W*r^2`` per slice. W, K, C ride unflattened
    through the grouped contractions (carry/deta carry K; xi/edge/P do not).'''
    xnp, _, xscan = get_backend(is_uniform, tree_contains_jax((carry, P, xi, deta_t, edge)))
    order = carry.shape[0] - 1                 # the carry is stacked over derivative orders
    src0 = xnp.zeros(carry.shape[:-1] + (P.shape[-1],), xnp.result_type(edge, P, deta_t))   # (order+1,)+W+K+C+(b,)
    (src, _, _), _ = xscan(_adj_src_order_step, (src0, deta_t, P), (edge, _inverse_factorials(order, edge, xnp)))
    return _adj_prop(carry, P, xi, xnp) + src


def _adj_sweep_scanned_step_uniform(
        carry:  NDArray,                                # (order+1,)+W+K+C+(rLi,); axis 0 is the order axis
        data:   typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (P, xi_jet, deta_tilde, edge_jet), one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:            # (next carry, (carry,))
    '''Uniform variant of :py:func:`_adj_tilde_step` as a closure-free scan body: the inner order loop
    is a real scan, which is what makes the memory win in :py:func:`_adj_sweep_scanned`.
    ``docs/contributor/scan_body_principles.md``.'''
    P, xi, deta_t, edge = data
    return _adj_tilde_step(carry, P, xi, deta_t, edge, True), (carry,)


def _adj_sweep_scanned_step_ragged(
        carry:  NDArray,                                # (order+1,)+W+K+C+(rLi,); axis 0 is the order axis
        data:   typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (P, xi_jet, deta_tilde, edge_jet), one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:            # (next carry, (carry,))
    '''Ragged variant of :py:func:`_adj_tilde_step` as a closure-free scan body: the inner order loop is
    a Python loop. ``docs/contributor/scan_body_principles.md``.'''
    P, xi, deta_t, edge = data
    return _adj_tilde_step(carry, P, xi, deta_t, edge, False), (carry,)


def _adj_sweep_scanned(P_cores, xi_jets, deta_tildes, edge_jets):
    '''The jet adjoint sweep (module-private), shared by :py:func:`compute_tau_tilde_jets` /
    :py:func:`compute_sigma_tilde_jets`: a left-to-right core sweep (mirroring probing.compute_tau_tilde)
    whose step is the TRANSPOSE of the forward recurrence/convolution -- a two-term reverse recurrence
    (prop, affine xi) + an inner order-scan (src, full). The inner scan is real (``lax.scan`` on the
    uniform path) so its ``W*r^2`` slice is the peak, not ``(order+1)*W*r^2``.'''
    use_jax = tree_contains_jax((P_cores, xi_jets, deta_tildes, edge_jets))
    is_uniform = is_ndarray(P_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)
    step = _adj_sweep_scanned_step_uniform if is_uniform else _adj_sweep_scanned_step_ragged

    # the carry is W+K+C; its leading stack (and the order) comes from deta_tildes, which carry K
    rL0 = P_cores[0].shape[-3]
    dtype = xnp.result_type(*(a[0] for a in (P_cores, xi_jets, deta_tildes, edge_jets)))   # not the default dtype
    init = xnp.zeros(deta_tildes[0].shape[:-1] + (rL0,), dtype)
    _, (tildes,) = xscan(step, init, (P_cores, xi_jets, deta_tildes, edge_jets))
    return tildes


def compute_tau_tilde_jets(
        left_tt_cores:  typ.Sequence[NDArray],  # P.  len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        mu_jets:        typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
) -> typ.Tuple[NDArray, ...]:                   # tau_tildes. len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
    '''Adjoint-var-rightward edge-variable jets tau_tilde (jet-ified probing.compute_tau_tilde; reverse
    recurrence + order scan, :py:func:`_adj_tilde_step`).'''
    return _adj_sweep_scanned(left_tt_cores, xi_jets, deta_tildes, mu_jets)


def compute_sigma_tilde_jets(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        deta_tildes:    typ.Sequence[NDArray],  # adjoint-up jets, len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        nu_jets:        typ.Sequence[NDArray],  # frame right jets, len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
) -> typ.Tuple[NDArray, ...]:                   # sigma_tildes. len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
    '''Adjoint-var-leftward edge-variable jets sigma_tilde -- the reverse of
    :py:func:`compute_tau_tilde_jets`.'''
    rev = _adj_sweep_scanned(tt_operations.tt_reverse(right_tt_cores), xi_jets[::-1],
                             deta_tildes[::-1], nu_jets[::-1])
    return rev[::-1]


def _dxi_tilde_order_step(
        carry:  typ.Tuple[NDArray, NDArray],   # (dxi~, O): (2,)+W+K+C+(nOi,) ; C+(rLi,nOi,rR(i+1))
        xs:     typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (mu^(j), nu^(j), [U^u sigma~]^(j), [U^u tau~]^(j)): W+C+(rLi,) ; W+C+(rR(i+1),) ; (2,)+W+K+C+(rR(i+1),) ; (2,)+W+K+C+(rLi,)
) -> typ.Tuple[typ.Tuple[NDArray, NDArray], typ.Tuple]:   # (next carry, ())
    '''One order ``j`` of the two-row dxi~ correlation: ``dxi~^(u) += (mu^(j) O) . [U^u sigma~]^(j) +
    [U^u tau~]^(j) . (O nu^(j))`` for ``u in {0, 1}`` -- one order slice of ``mu O`` / ``O nu`` live (peak
    ``W*r*n``). Closure-free scan body (the core ``O`` rides the carry).'''
    dxi, O = carry
    mu_j, nu_j, sig_pair_j, tau_pair_j = xs
    nf = O.ndim - 3                                                    # O is C-only: pins len(C)
    mO = contractions.contract('WCa,Caib->WCib', mu_j, O)
    On = contractions.contract('Caib,WCb->WCai', O, nu_j)
    dxi = (dxi + contractions.contract('WCib,uWKCb->uWKCi', mO, sig_pair_j, len_C=nf)
           + contractions.contract('uWKCa,WCai->uWKCi', tau_pair_j, On, len_C=nf))
    return (dxi, O), ()


def _dxi_tilde_jets_core(
        data:       typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray],  # (O, mu_jet, nu_jet, sigma_tilde, tau_tilde), one core
        is_uniform: bool,                       # inner-scan backend (a real lax.scan on the uniform path)
) -> typ.Tuple[NDArray]:                        # (dxi_tilde,) elm_shape=(2,)+W+K+C+(nOi,)
    '''One core of :py:func:`compute_dxi_tilde_jets`: the two read rows of the adjoint-down combine, as an
    order scan over the frame jets' order with the tildes' reverse banded pairs scanned beside them.'''
    O, mu, nu, st, tt = data
    xnp, _, xscan = get_backend(is_uniform, tree_contains_jax(data))
    dxi0 = xnp.zeros((2,) + st.shape[1:-1] + (O.shape[-2],), xnp.result_type(O, mu, nu, st, tt))   # W+K+C off the tilde
    (dxi, _), _ = xscan(_dxi_tilde_order_step, (dxi0, O),
                        (mu, nu, _shift_up_pair(st, xnp, axis=1), _shift_up_pair(tt, xnp, axis=1)))
    return (dxi,)


def _dxi_tilde_jets_step_uniform(data):  # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Uniform variant of :py:func:`_dxi_tilde_jets_core` (a real inner scan).'''
    return _dxi_tilde_jets_core(data, True)


def _dxi_tilde_jets_step_ragged(data):   # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Ragged variant of :py:func:`_dxi_tilde_jets_core` (the inner order loop is a Python loop).'''
    return _dxi_tilde_jets_core(data, False)


def compute_dxi_tilde_jets(
        down_tt_cores:  typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
) -> typ.Tuple[NDArray, ...]:                   # dxi_tildes. len=d, elm_shape=(2,)+W+K+C+(nOi,): rows u = 0, 1 only
    '''Adjoint-var-down edge-variable jets (jet-ified probing.compute_dxi_tilde): the two adjoint-hooked
    combines ``tau~ O nu`` and ``mu O sigma~`` giving delta-xi-tilde on the mode.

    ``dxi~`` is the adjoint of the affine input jet ``dxi``, so only its rows ``u in {0, 1}`` are ever read
    (the Tucker assembly pairs them with the ambient input jet ``(w, p)``), and only those two are formed:

        ``dxi~^(u) = sum_j (mu^(j) O) . [U^u sigma~]^(j) + [U^u tau~]^(j) . (O nu^(j))``,

    with ``U x = [(s+1) x^(s+1)]_s`` (row 1 of :py:func:`_shift_up_pair`): ``C(j+1, j) = j+1`` is the only
    binomial weight a two-row correlation needs. An order scan over ``j`` (peak one ``W*r*n`` slice per
    term), mapped over cores. At ``order = 0`` row 1 is zero.'''
    use_jax = tree_contains_jax((down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes))
    is_uniform = is_ndarray(down_tt_cores)
    xnp, xmap, _ = get_backend(is_uniform, use_jax)
    step = _dxi_tilde_jets_step_uniform if is_uniform else _dxi_tilde_jets_step_ragged
    (dxi_tildes,) = xmap(step, (down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes))
    return dxi_tildes


def _tucker_variation_jets_core(
        x:               typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (ztilde, dxi_tilde, eta_jet, w_jet), one core
        sum_over_probes: bool,     # caller intent -- no operand carries it; hence the two bodies below
) -> typ.Tuple[NDArray]:           # (dU_tilde,): W+K+C+(nOi,Ni), or K+C+(nOi,Ni) when summed
    '''One core of the ragged map of :py:func:`_assemble_tucker_variation_jets_dense`, shared by the two
    closure-free map bodies below -- ``docs/contributor/scan_body_principles.md``.'''
    zt, dxt, eta, wj = x
    s_size = min(2, eta.shape[0])   # the w/dxi input jet carries orders {0, 1}, capped at order
    # Three-group (W,K,C): the residual-derived operands (ztilde, dxi_tilde) carry K, eta is frame; the
    # eta (x) r term takes n_probe = len(W), recovered locally from the W-only ambient jet w_jet
    # ((2,)+W+(Ni,)); the dxi (x) w_jet term self-pins W from that same w_jet.
    n_probe = wj.ndim - 2
    if sum_over_probes:
        eta_r, dxi_w = 'tWCa,tWKCo->KCao', 'uWKCa,uWo->KCao'
    else:
        eta_r, dxi_w = 'tWCa,tWKCo->WKCao', 'uWKCa,uWo->WKCao'
    return (contractions.contract(eta_r, eta, zt, len_W=n_probe)
            + contractions.contract(dxi_w, dxt[:s_size], wj[:s_size]),)


def _tucker_variation_jets_step_summed(x):    # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=True`` variant of :py:func:`_tucker_variation_jets_core`.'''
    return _tucker_variation_jets_core(x, True)


def _tucker_variation_jets_step_unsummed(x):  # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=False`` variant of :py:func:`_tucker_variation_jets_core`.'''
    return _tucker_variation_jets_core(x, False)


def _assemble_tucker_variation_jets_dense(
        ztildes:        typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
        dxi_tildes:     typ.Sequence[NDArray],  # adjoint-var-down jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        ww:             typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:             typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        etas:           typ.Sequence[NDArray],  # frame down jets, len=d, elm_shape=(order+1,)+W+C+(nOi,)
        n_probe:        int,                    # number of sample-stack (W) axes
        sum_over_probes: bool,
) -> typ.Tuple[NDArray, ...]:                   # dU_tildes. len=d, elm_shape=[W+]C+(nOi,Ni)
    '''Assemble Tucker-core variation gradients, unchunked (the 1-edge, plain-order-sum case; the per-chunk
    kernel of :py:func:`assemble_tucker_variation_jets`):
    ``dU_tilde = sum_t eta^(t) (x) r^(t) + sum_u dxi_tilde^(u) (x) w_jet^(u)`` -- both diagonal in the order,
    so no binomial weight at all.'''
    use_jax = tree_contains_jax((ztildes, dxi_tildes, ww, pp, etas))
    is_uniform = is_ndarray(etas)
    _, xmap, _ = get_backend(is_uniform, use_jax)
    w_jets = build_input_jets(ww, pp)          # ambient input jets (value, direction) for dU
    s_size = min(2, etas[0].shape[0])          # the w/dxi input jet carries orders {0, 1}, capped at order
    # Three-group (W,K,C): the residual-derived operands (ztilde, dxi_tilde) carry K, eta is frame; the
    # eta (x) r term takes n_probe (C from eta), the dxi (x) w_jet term self-pins W from the W-only w_jet.
    if is_uniform:
        # d-prefixed assembly; the order axis is at supercore axis 1, so the w/dxi order-slice is
        # [:, :s_size] (NOT [:s_size], which would slice the leading core axis d). Ragged xmap is the oracle.
        eta_r = 'dtWCa,dtWKCo->dKCao' if sum_over_probes else 'dtWCa,dtWKCo->dWKCao'
        dxi_w = 'duWKCa,duWo->dKCao' if sum_over_probes else 'duWKCa,duWo->dWKCao'
        return (contractions.contract(eta_r, etas, ztildes, len_W=n_probe)
                + contractions.contract(dxi_w, dxi_tildes[:, :s_size], w_jets[:, :s_size]))

    step = (_tucker_variation_jets_step_summed if sum_over_probes
            else _tucker_variation_jets_step_unsummed)
    (dU_tildes,) = xmap(step, (ztildes, dxi_tildes, etas, w_jets))
    return dU_tildes


def assemble_tucker_variation_jets(
        ztildes:        typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+K+C+(Ni,)
        dxi_tildes:     typ.Sequence[NDArray],  # adjoint-var-down jets, len=d, elm_shape=(2,)+W+K+C+(nOi,)
        ww:             typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:             typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        etas:           typ.Sequence[NDArray],  # frame down jets, len=d, elm_shape=(order+1,)+W+C+(nOi,)
        n_probe:        int,                    # number of sample-stack (W) axes
        sum_over_probes: bool,
        chunk_size:     typ.Optional[int] = DEFAULT_CHUNK_SIZE,   # W-chunk size; None (or >= W) -> dense. See docs/chunking.md
) -> typ.Tuple[NDArray, ...]:                   # dU_tildes. len=d, elm_shape=[W+]K+C+(nOi,Ni)
    '''Assemble the Tucker variation gradient (W-chunked; the per-chunk kernel is
    :py:func:`_assemble_tucker_variation_jets_dense`). The milder assembly (two legs nO,N -- the dense
    gradient is a few GB, not the tt-core's hundreds), chunked over W with the same reducer seam (add if
    summed, concat if kept) for a uniform chunked-assembly interface. The only twist: ``ww``/``pp`` have
    no order axis, so W sits at a per-operand axis (2 for the order-carrying jets, 1 for ww/pp).
    ``chunk_size`` None / ragged / multi-W / small W -> runs the dense assembly directly (no chunking).'''
    ops = (ztildes, dxi_tildes, ww, pp, etas)
    w_axes = (2, 2, 1, 1, 2)                          # per-operand W axis (ww/pp carry no order axis)
    dense = lambda: _assemble_tucker_variation_jets_dense(*ops, n_probe, sum_over_probes)
    if chunk_size is None or not is_ndarray(etas):
        return dense()
    if n_probe != 1 and not sum_over_probes:          # merging probe axes is exact only when reduced
        return dense()
    use_jax = tree_contains_jax(ops)
    xnp, _, _ = get_backend(True, use_jax)
    W = math.prod(etas.shape[2:2 + n_probe])
    if W <= chunk_size:
        return dense()
    ops, w_axes = _flatten_w(ops, w_axes, n_probe)     # -> one probe super-axis; n_probe == 1 holds
    if sum_over_probes:
        return _wchunked_sum(_tucker_chunk_summed, None, ops, w_axes, W, chunk_size, use_jax)
    return _wchunked_concat(_tucker_chunk_kept, None, ops, w_axes, W, chunk_size, xnp)


@functools.lru_cache(maxsize=None)
def _tt_det_lag_step(
        n_probe:         int,      # len(W): no operand of the deta~ group is W-only or C-only
        sum_over_probes: bool,     # W summed (KCaib) or kept (WKCaib)
):                                 # -> the scan body (carry, xs) -> (carry, ())
    '''The inner scan body of the TT assembly's deta~ group, ``sum_{r,s} C(r+s, r) mu^(r) (x) deta~^(r+s) (x)
    nu^(s)`` -- a full two-sided binomial correlation, scanned over the lag ``r``. The carry is
    ``(acc, U^r deta~, nu)`` (``nu`` rides unchanged), the xs ``(mu^(r), 1/r!)``: per step, fold the shifted
    ``deta~`` against the whole ``nu`` (peak ``W*K*n*r``), add the outer product with ``mu^(r)``, up-shift.

    A memoized factory rather than a plain function because ``n_probe`` (the W|C split) and the output
    string are statics no operand carries (``docs/contributor/scan_body_principles.md``, principle 4): it
    closes over two hashable scalars, never an array.'''
    final = 'WCa,WKCib->KCaib' if sum_over_probes else 'WCa,WKCib->WKCaib'

    def _step(carry, xs):
        acc, deta_r, nu = carry
        mu_r, w_r = xs
        xnp, _, _ = get_backend(True, tree_contains_jax((carry, xs)))      # only xnp
        fold = contractions.contract('sWKCi,sWCb->WKCib', deta_r, nu, len_W=n_probe)
        acc = acc + contractions.contract(final, w_r * mu_r, fold, len_W=n_probe)
        return (acc, _shift_up(deta_r, xnp), nu), ()
    return _step


@functools.lru_cache(maxsize=None)
def _tt_variation_jets_step(
        n_probe:         int,      # len(W); no operand here is W-only or C-only
        sum_over_probes: bool,     # W summed (KCaib) or kept (WKCaib)
        is_uniform:      bool,     # inner-scan backend (a real lax.scan on the uniform path)
):                                 # -> the map body x -> (dG_tilde,)
    '''The per-core map body of :py:func:`_assemble_tt_variation_jets_dense`: one core's TT variation
    gradient ``mu (x) xi (x) sigma~ + tau~ (x) xi (x) nu + mu (x) deta~ (x) nu`` (the core-adjoints of the
    forward sigma / tau / deta contractions), every order summed.

    - **sigma~ and tau~ groups** (the input jet ``xi`` is affine): banded. ``sum_{r, s<2} C(r+s, r)
      mu^(r) (x) xi^(s) (x) sigma~^(r+s)`` is ``sum_k xi^(k) (x) [sum_r mu^(r) (x) U^k sigma~^(r)]``, so
      form the pair ``sum_r mu^(r) (x) [U^k sigma~]^(r)`` (``k in {0,1}``; :py:func:`_shift_up_pair`) and
      its tau~ twin, add them (both meet ``xi^(k)``), and contract with ``xi`` once.
    - **deta~ group** (both ``mu`` and ``nu`` full): an order scan over the lag, :py:func:`_tt_det_lag_step`.

    Memoized on its statics (``n_probe``, the output layout, the inner-scan backend) -- hashable scalars
    only (``docs/contributor/scan_body_principles.md``, principle 4).'''
    final = 'kWKCab,kWCi->KCaib' if sum_over_probes else 'kWKCab,kWCi->WKCaib'
    lag_step = _tt_det_lag_step(n_probe, sum_over_probes)

    def _step(x):
        xi, mu, nu, st, tt, dt = x
        xnp, _, xscan = get_backend(is_uniform, tree_contains_jax(x))
        order = mu.shape[0] - 1
        pair = (contractions.contract('krWKCb,rWCa->kWKCab', _shift_up_pair(st, xnp), mu, len_W=n_probe)
                + contractions.contract('ksWKCa,sWCb->kWKCab', _shift_up_pair(tt, xnp), nu, len_W=n_probe))
        banded = contractions.contract(final, pair, xi, len_W=n_probe)        # xi: (2,)+W+C+(nU,), all orders
        (det, _, _), _ = xscan(lag_step, (xnp.zeros(banded.shape, xnp.result_type(banded, mu, dt, nu)), dt, nu),
                               (mu, _inverse_factorials(order, mu, xnp)))
        return (banded + det,)
    return _step


def _assemble_tt_variation_jets_dense(
        sigma_tildes:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_tildes:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
        deta_tildes:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        mu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        nu_jets:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        n_probe:        int,                    # number of sample-stack (W) axes
        sum_over_probes: bool,
) -> typ.Tuple[NDArray, ...]:                   # dG_tildes. len=d, elm_shape=[W+]K+C+(rLi,nUi,rRi)
    '''Assemble TT-core variation gradients, unchunked (the per-chunk kernel of
    :py:func:`assemble_tt_variation_jets`): mapped over cores (``lax.map`` on the uniform path, so one core's
    intermediates are resident), the per-core body :py:func:`_tt_variation_jets_step`. The residual-derived
    adjoint vars carry K on the assembled core's leg -- sigma~ on b, tau~ on a, deta~ on i; the frame
    xi/mu/nu do not. K kept always; W summed (``sum_over_probes`` -> KCaib) or kept (WKCaib).'''
    use_jax = tree_contains_jax((sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets))
    is_uniform = is_ndarray(xi_jets)
    _, xmap, _ = get_backend(is_uniform, use_jax)
    step = _tt_variation_jets_step(n_probe, sum_over_probes, is_uniform)
    (dG_tildes,) = xmap(step, (xi_jets, mu_jets, nu_jets, sigma_tildes, tau_tildes, deta_tildes))
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


def _tucker_chunk_summed(co, extra):    # extra unused; kept for a common chunk-fn signature
    '''One W-chunk of the Tucker-variation assembly at the chunked path's pinned statics.'''
    return _assemble_tucker_variation_jets_dense(*co, 1, True)


def _tucker_chunk_kept(co, extra):
    '''As :py:func:`_tucker_chunk_summed`, keeping the probe stack.'''
    return _assemble_tucker_variation_jets_dense(*co, 1, False)


def _tt_chunk_summed(co, extra):        # extra unused; the common chunk-fn signature
    '''One W-chunk of the TT-variation assembly at the chunked path's pinned ``n_probe = 1``.'''
    return _assemble_tt_variation_jets_dense(*co, 1, True)


def _tt_chunk_kept(co, extra):
    '''As :py:func:`_tt_chunk_summed`, keeping the probe stack.'''
    return _assemble_tt_variation_jets_dense(*co, 1, False)


@functools.lru_cache(maxsize=None)
def _wchunked_sum_step(
        chunk_fn,               # one of the module-level _*_chunk_summed functions
        cs:      int,           # chunk size -- the dynamic_slice SIZE, so necessarily static
        w_axes:  typ.Tuple[int, ...],   # per-operand W axis
):                              # -> the lax.scan body (carry, i) -> (carry, 0)
    '''The scan body of :py:func:`_wchunked_sum`: one stable object per ``(assembler, cs, w_axes)``.

    A memoized factory rather than a plain module-level function because ``cs`` is the slice size and
    must stay static, and it is user-tunable -- a stable body closing over it would silently reuse a
    stale jaxpr (``docs/contributor/scan_body_principles.md``, principle 5). All three keys are
    hashable non-arrays, per principle 4's rule; the operands (and any array ``extra``) ride the carry
    instead.'''
    import jax

    def _step(carry, i):
        acc, ops, extra = carry
        chunk = tuple(jax.lax.dynamic_slice_in_dim(o, i * cs, cs, w) for o, w in zip(ops, w_axes))
        return (acc + chunk_fn(chunk, extra), ops, extra), 0
    return _step


def _flatten_w(
        ops:     typ.Sequence[NDArray],  # operands, each with n_probe consecutive W axes at w_axes[i]
        w_axes:  typ.Tuple[int, ...],    # per-operand W start axis
        n_probe: int,                    # number of probe axes to merge
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[int, ...]]:   # (flattened ops, unchanged w_axes)
    '''Merge each operand's probe axes into ONE super-axis of size ``prod(W)``.

    W is a contiguous axis group in every operand, so this is an axis merge (metadata, not a copy).
    It makes ``n_probe == 1`` an invariant of the chunked path rather than a precondition -- so
    chunking applies at any probe rank -- and it removes the question of which probe axis to slice.
    Exact only where the probe stack is REDUCED (the summed path); a kept stack would need
    unflattening, so the caller does not come here in that case with ``n_probe > 1``.'''
    if n_probe == 1:
        return tuple(ops), w_axes
    flat = []
    for o, w in zip(ops, w_axes):
        sh = tuple(o.shape)
        flat.append(o.reshape(sh[:w] + (math.prod(sh[w:w + n_probe]),) + sh[w + n_probe:]))
    return tuple(flat), w_axes


def _wchunked_sum(chunk_fn, extra, ops, w_axes, W, cs, use_jax):
    '''W-chunked assembly, ADDING the chunks (the summed gradient). Copy-free: each chunk is extracted
    in place -- no pad, no reshape, no transpose of the operands. On jax the full chunks are a real
    ``lax.scan`` over dynamic slices, so only one chunk is resident; the remainder is a static slice;
    chunk 0 seeds the accumulator and its shape. On numpy it is an eager loop, which frees each chunk
    anyway. The operands and ``extra`` ride the CARRY (loop-invariant) so the scan body can be a stable
    object -- see :py:func:`_wchunked_sum_step`.'''
    def _chunk(start, size):
        return tuple(_slice_w(o, w, start, size, False) for o, w in zip(ops, w_axes))
    n_full, rem = divmod(W, cs)
    acc = chunk_fn(_chunk(0, cs), extra)
    if use_jax and n_full > 1:
        import jax
        import jax.numpy as jnp
        step = _wchunked_sum_step(chunk_fn, cs, tuple(w_axes))
        (acc, _, _), _ = jax.lax.scan(step, (acc, tuple(ops), extra), jnp.arange(1, n_full))
    elif n_full > 1:
        for k in range(1, n_full):
            acc = acc + chunk_fn(_chunk(k * cs, cs), extra)
    if rem:
        acc = acc + chunk_fn(_chunk(n_full * cs, rem), extra)
    return acc


def _wchunked_concat(chunk_fn, extra, ops, w_axes, W, cs, xnp):
    '''W-chunked assembly, CONCATENATING the chunks (the probe stack is kept). No scan here -- the
    parts must all be materialized to concatenate them -- so no loop body and none of the
    identity-cache concerns of :py:func:`_wchunked_sum`.'''
    def _chunk(start, size):
        return tuple(_slice_w(o, w, start, size, False) for o, w in zip(ops, w_axes))
    parts = [chunk_fn(_chunk(st, min(cs, W - st)), extra) for st in range(0, W, cs)]
    return xnp.concatenate(parts, axis=1)


def assemble_tt_variation_jets(
        sigma_tildes:   typ.Sequence[NDArray],
        tau_tildes:     typ.Sequence[NDArray],
        deta_tildes:    typ.Sequence[NDArray],
        xi_jets:        typ.Sequence[NDArray],
        mu_jets:        typ.Sequence[NDArray],
        nu_jets:        typ.Sequence[NDArray],
        n_probe:        int,
        sum_over_probes: bool,
        chunk_size:     typ.Optional[int] = DEFAULT_CHUNK_SIZE,    # W-chunk size; None (or >= W) -> dense. See docs/chunking.md
) -> NDArray:                                        # dG_tildes supercore, [W+]K+C+(rLi,nUi,rRi)
    '''Assemble the TT variation gradient (W-chunked; the per-chunk kernel is
    :py:func:`_assemble_tt_variation_jets_dense`: banded sigma~ / tau~ groups and an order scan for the deta~
    group, no binomial tensor).

    The unchunked assembly's peak is exactly LINEAR in the sample stack W, so it is chunked along W: the
    unchunked assembly runs per W-chunk and the partials are combined. Peak ~ ``chunk_size * (per-W-row)``
    instead of ``W * (per-W-row)``.

    **The reducer is the seam that keeps this from locking into W-only** (Nick, 2026-07-16). A chunked
    batch axis is combined by ADD if it is *summed* (``sum_over_probes`` -> the gradient) or by CONCAT if
    it is *kept* (the ``sum_over_probes=False`` per-probe output, and -- later -- a chunked *frame* stack
    C, which is always kept). The per-chunk assembler is axis-agnostic (the unchunked assembly), so extending
    to C-chunking is a new slice front + the same reducer, not a rewrite.

    Chunking runs on the uniform path with a single W axis; ragged / multi-W / ``chunk_size`` unset /
    ``W <= chunk_size`` fall back to the unchunked assembly. The chunk loop is a real ``lax.scan``
    (sequential), so only one chunk's intermediate is resident.
    '''
    ops = (sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets)
    w_axes = (2,) * len(ops)
    dense = lambda: _assemble_tt_variation_jets_dense(*ops, n_probe, sum_over_probes)
    if chunk_size is None or not is_ndarray(xi_jets):
        return dense()
    if n_probe != 1 and not sum_over_probes:          # merging probe axes is exact only when reduced
        return dense()

    use_jax = tree_contains_jax(ops)
    xnp, _, _ = get_backend(True, use_jax)
    W = math.prod(xi_jets.shape[2:2 + n_probe])       # uniform supercore: (d, order, W..., ...)
    if W <= chunk_size:
        return dense()
    ops, w_axes = _flatten_w(ops, w_axes, n_probe)     # -> one probe super-axis; n_probe == 1 holds
    if sum_over_probes:
        return _wchunked_sum(_tt_chunk_summed, None, ops, w_axes, W, chunk_size, use_jax)
    return _wchunked_concat(_tt_chunk_kept, None, ops, w_axes, W, chunk_size, xnp)


# ==================================================================================================
# Choosing chunk_size -- an eager (outside-jit) estimator from the problem shapes
# ==================================================================================================
# The transpose functions take a plain-int chunk_size with a safe fixed default. To pick a *tuned*
# value, call these once outside jit and pass the int down. Two policies (docs/chunking.md):
#   estimate_chunk_size   -- BALANCED: assembly peak ~ the resident edge-jet memory (device-agnostic).
#   max_chunk_size_within -- BUDGET:   the largest chunk whose assembly peak fits an absolute byte cap.
# Both measure the per-row cost IN SITU -- the slope in chunk_size of the compiled chunked transpose (both
# gradient assemblies, TT-core r^2 legs and Tucker nO/N legs, plus the chunk's slices of their inputs) -- and
# divide W by n_shards so the chunk sizes each device's shard (docs/chunking.md sharding section). Uniform+jax
# only -- numpy/ragged free eagerly and never chunk. hessian='exact' also sizes the exact Hessian's
# probe-derivative two-hole, which shares the chunk_size and costs more per row.


_HESSIAN_MODES = ('gauss_newton', 'exact')


def _check_hessian_mode(
        hessian:  str,          # 'gauss_newton' | 'exact' (case-insensitive) -- the solver's own vocabulary
        who:      str,          # the function name, for the message
) -> str:                       # the lower-cased mode
    '''Validate a ``hessian`` mode selector (structural: anything else raises, a ``bool`` included -- the
    keyword means the solver's string mode everywhere, so ``True`` / ``False`` is a misuse, not a shorthand).'''
    if not isinstance(hessian, str) or hessian.lower() not in _HESSIAN_MODES:
        raise ValueError("%s: hessian must be 'gauss_newton' or 'exact' (case-insensitive); got %r"
                         % (who, hessian))
    return hessian.lower()


def _jet_floor_bytes(mode_shapes, tucker_ranks, tt_ranks, order, W, n_tangent, itemsize,
                     stack_shape=()):
    '''Exact bytes of the resident edge-variable jets the transpose assembly reads -- the necessary
    memory floor (linear in W). Uniform supercores pad ranks to their max, so max ranks are used; the
    tangent stack K rides on the tilde jets and the residual only.

    The frame stack ``C`` multiplies every *jet*, because each jet carries ``W + [K +] C``; it does
    NOT multiply ``ww``/``pp``, which are probe vectors of shape ``W + (Ni,)`` with no frame stack.
    Scaling the whole row by ``prod(C)`` would therefore over-count.'''
    d, op, K = len(mode_shapes), order + 1, n_tangent
    r, nU, N = max(tt_ranks), max(tucker_ranks), max(mode_shapes)
    nO = nU                                         # down-core mode ~ the Tucker rank
    C = math.prod(tuple(stack_shape))               # frame stack: on the jets only
    per_core_row = (C * (K * (op * (r + r + nU + N)        # sigma_t, tau_t, deta_t, ztildes (K+C)
                              + 2 * nO)                    # dxi_t: its two read rows (K+C)
                         + 2 * nU + op * (r + r + nO))     # xi, mu, nu, eta (C, no K)
                    + 2 * N)                               # ww, pp -- probe vectors, W only
    return d * W * per_core_row * itemsize


_SLOPE_CHUNKS = (128, 256)    # the two chunk sizes whose peak difference is the per-row slope (W = 4x the larger)


def _chunk_slope(
        make:   typ.Callable,           # chunk_size -> the function to lower (its arguments: ``args``)
        args:   typ.Tuple,              # jax.ShapeDtypeStruct pytrees, the sample stack W = 4 * _SLOPE_CHUNKS[1]
) -> float:                             # bytes of compiled scratch per unit of chunk_size
    '''``d(temp)/d(chunk_size)`` of a compiled program: lower it abstractly (no allocation) at the two
    :py:data:`_SLOPE_CHUNKS` with ``W`` fixed and a multiple of both, and difference XLA's
    ``memory_analysis().temp_size_in_bytes``. Everything the chunk size scales is in the slope; everything
    it does not (the resident jets, the accumulator) cancels.'''
    import jax
    c1, c2 = _SLOPE_CHUNKS
    temp = lambda cs: jax.jit(make(cs)).lower(*args).compile().memory_analysis().temp_size_in_bytes
    return (temp(c2) - temp(c1)) / float(c2 - c1)


@functools.lru_cache(maxsize=None)
def _assembly_per_row_bytes(mode_shapes, tucker_ranks, tt_ranks, order, n_tangent, dtype,
                            stack_shape=()):
    '''Measured peak scratch (bytes) per W-row of the chunked probe-derivative transpose -- the slope of the
    compiled :py:func:`tv_probe_derivatives_transpose` (uniform, ``sum_over_probes=True``) in its
    ``chunk_size`` (:py:func:`_chunk_slope`), i.e. what one more probe in the chunk costs **in the program
    that runs**: both gradient assemblies' scratch, the chunk's slices of their input jets, and whatever
    rematerialization XLA's fusion decides. Measured in situ, not on an isolated kernel: an isolated kernel
    sees its inputs as parameters, so it misses the chunk-sized slices the chunked program materializes
    (under-estimating small-rank problems by up to ~1.5x) while counting unchunked intermediates the chunked
    program never forms (over-estimating large ones ~4x). An analytic formula is further off still. jax-only
    (the chunking path is uniform+jax); a few seconds of abstract compiles, cached by shape signature.

    The frame stack ``C`` is put into the operand *shapes* rather than multiplied onto the result,
    because the scaling is not always ``prod(C)``: it is exactly linear when the peak is driven by
    jets (which all carry ``C``) and sub-linear when the C-free ``ww``/``pp`` dominate. Measuring is
    the whole point of this function -- don't layer an assumption on top of it.'''
    import jax
    d, op = len(mode_shapes), order + 1
    r, nU, N = max(tt_ranks), max(tucker_ranks), max(mode_shapes)
    kax = (n_tangent,) if n_tangent > 1 else ()      # K axis present only when batching tangents
    cax = tuple(stack_shape)                         # C axis/axes -- frame-inner, so after K
    S = lambda *shape: jax.ShapeDtypeStruct(shape, dtype)
    W = 4 * _SLOPE_CHUNKS[1]
    frame = (S(d, *cax, nU, N), S(d, *cax, r, nU, r), S(d, *cax, r, nU, r), S(d, *cax, r, nU, r))   # (U, O, P, Q)
    args = (S(d, op, W, *kax, *cax, N), S(d, W, N), S(d, W, N), frame)                            # ztildes, ww, pp
    make = lambda cs: (lambda z, w, p, f: tv_probe_derivatives_transpose(
        z, w, p, f, order, sum_over_probes=True, chunk_size=cs))
    return max(_chunk_slope(make, args), 1.0)


def _two_hole_jet_floor_bytes(mode_shapes, tucker_ranks, tt_ranks, order, W, n_tangent, itemsize,
                              stack_shape=()):
    '''The probe-derivative two-hole's analog of :py:func:`_jet_floor_bytes`: the resident jets its assemblies
    read -- the undotted state (``xi``, ``mu``, the re-swept ``nu``, the three tildes and the residual, all
    without ``K``) and the dotted companions (``K``), including ``eta_dot`` and ``dxi~_dot`` (two rows).'''
    d, op, K = len(mode_shapes), order + 1, n_tangent
    r, nU, N = max(tt_ranks), max(tucker_ranks), max(mode_shapes)
    nO = nU                                         # down-core mode ~ the Tucker rank
    C = math.prod(tuple(stack_shape))               # frame stack: on the jets only
    per_core_row = (C * (2 * nU + op * (4 * r + nU + N)              # xi ; mu, nu, sigma~, tau~ ; deta~ ; z~
                         + K * (2 * nU + op * (4 * r + nU + nO)        # their dots ; eta_dot
                                + 2 * nO))                                   # dxi~_dot: its two read rows
                    + 2 * N)                                           # ww, pp -- probe vectors, W only
    return d * W * per_core_row * itemsize


@functools.lru_cache(maxsize=None)
def _two_hole_assembly_per_row_bytes(mode_shapes, tucker_ranks, tt_ranks, order, n_tangent, dtype,
                                     stack_shape=()):
    '''Measured peak scratch (bytes) per W-row of the chunked probe-derivative two-hole -- the
    :py:func:`_assembly_per_row_bytes` measurement applied to :py:func:`t3_probe_corewise_derivatives_hessian`
    (directions stacked ``K``, the residual jets ``W + C``). jax-only; cached by shape signature.'''
    import jax
    d, op = len(mode_shapes), order + 1
    r, nU, N = max(tt_ranks), max(tucker_ranks), max(mode_shapes)
    kax = (n_tangent,) if n_tangent > 1 else ()      # K axis present only when batching directions
    cax = tuple(stack_shape)
    S = lambda *shape: jax.ShapeDtypeStruct(shape, dtype)
    W = 4 * _SLOPE_CHUNKS[1]
    theta = (S(d, *cax, nU, N), S(d, *cax, r, nU, r))
    dirs = (S(d, *kax, *cax, nU, N), S(d, *kax, *cax, r, nU, r))
    args = (S(d, op, W, *cax, N), S(d, W, N), S(d, W, N), theta, dirs)                    # z~ (no K), ww, pp
    make = lambda cs: (lambda z, w, p, th, dd: t3_probe_corewise_derivatives_hessian(
        z, w, p, th, dd, order, chunk_size=cs))
    return max(_chunk_slope(make, args), 1.0)


def estimate_chunk_size(
        mode_shapes:    typ.Sequence[int],  # (N_1..N_d)   ambient dims (as passed to TuckerTensorTrain.randn)
        tucker_ranks:   typ.Sequence[int],  # (nU_1..nU_d) Tucker ranks
        tt_ranks:       typ.Sequence[int],  # (r_0..r_d)   the d+1 TT bonds
        order:          int,                # highest derivative order
        n_probes:       int,                # |W|, the number of probes (global; divided by n_shards)
        *,
        n_tangent:      int = 1,            # tangent stack K (a batch of tangents sharing the frame)
        stack_shape:    typ.Sequence[int] = (),  # frame stack C (a batch of base points); () = unstacked
        n_shards:       int = 1,            # W split across this many devices -> chunk sizes the LOCAL shard
        dtype:          typ.Any = None,     # array dtype; default float32 (jax default). float64 under x64
        hessian:        str = 'gauss_newton',  # or 'exact': also bound the exact Hessian's two-hole assemblies
) -> int:                                   # a memory-balanced chunk_size for tv_probe_derivatives_transpose
    '''A memory-**balanced** ``chunk_size`` for the probe-derivative transpose assembly, from the problem
    shapes -- call it once (eagerly, outside ``jit``) and pass the int as ``chunk_size``.

    Picks the largest chunk whose assembly peak is comparable to the edge-variable jets already resident
    (the necessary floor), so the assembly is never the tallest pole -- if the rest of the pipeline fits,
    so does the assembly (total peak ``~2x`` the floor). Device-agnostic: no device-memory query, only the
    shapes. The assembly per-row cost is **measured** in the compiled chunked transpose itself, via XLA's own
    scratch accounting (needs jax; a ~2 s one-time abstract compile, cached by shape). For an absolute byte cap
    instead, see :py:func:`max_chunk_size_within`; the memory model is in the chunking design note.

    Sized by the whole transpose, so both the TT-core and the Tucker gradients count (Tucker wins when
    ``N >> n``). When the assemblies cost less per row than the resident jets -- the usual case since they
    became recurrences -- the balance admits the whole ``W`` and this returns ``W``. ``n_shards``
    divides ``W`` so the chunk sizes each device's shard (use ``shard_map``; see the chunking note). Only
    meaningful on the uniform+jax path -- numpy/ragged never chunk.

    Pass ``stack_shape`` for a **batch of base points** (the frame stack ``C``): every edge-variable jet
    carries it, so it multiplies both sides of the balance. This policy is therefore only mildly
    sensitive to it -- the probe vectors ``ww``/``pp`` carry no ``C``, so the two sides do not scale
    quite together -- but :py:func:`max_chunk_size_within` is *very* sensitive, since its budget is an
    absolute byte count that does not scale at all.

    Pass ``hessian='exact'`` (the solver's mode vocabulary, case-insensitive; ``'gauss_newton'`` is the
    default) when the same ``chunk_size`` also drives the exact Hessian's probe-derivative two-hole
    (:py:func:`t3_probe_corewise_derivatives_hessian`). Its chunk costs ``~1.3-2.2x`` the transpose's per
    row (the dotted companion of every pair; measured in its own compiled program), against a jet floor
    ``~1.1-1.6x`` the transpose's; the result is the smaller of the two balanced chunks, so both programs stay
    within their floors. ``n_tangent`` is then also the two-hole's direction stack (one in a
    CG solve).
    '''
    exact = _check_hessian_mode(hessian, 'estimate_chunk_size') == 'exact'
    dtype = np.dtype(np.float32 if dtype is None else dtype)
    ms, tr, tt = tuple(mode_shapes), tuple(tucker_ranks), tuple(tt_ranks)
    cs = tuple(int(c) for c in stack_shape)
    w_local = max(1, int(n_probes) // max(1, int(n_shards)))
    budget = _jet_floor_bytes(ms, tr, tt, order, w_local, n_tangent, dtype.itemsize, cs)
    per_row = _assembly_per_row_bytes(ms, tr, tt, order, n_tangent, dtype, cs)
    chunk = int(budget // per_row)
    if exact:
        budget_h = _two_hole_jet_floor_bytes(ms, tr, tt, order, w_local, n_tangent, dtype.itemsize, cs)
        per_row_h = _two_hole_assembly_per_row_bytes(ms, tr, tt, order, n_tangent, dtype, cs)
        chunk = min(chunk, int(budget_h // per_row_h))
    return max(1, min(w_local, chunk))


def max_chunk_size_within(
        mode_shapes:    typ.Sequence[int],  # (N_1..N_d)   ambient dims
        tucker_ranks:   typ.Sequence[int],  # (nU_1..nU_d) Tucker ranks
        tt_ranks:       typ.Sequence[int],  # (r_0..r_d)   the d+1 TT bonds
        order:          int,                # highest derivative order
        n_probes:       int,                # |W| (global; divided by n_shards)
        target_bytes:   float,              # absolute peak-memory cap for the assembly (per device)
        *,
        n_tangent:      int = 1,
        stack_shape:    typ.Sequence[int] = (),  # frame stack C (a batch of base points); () = unstacked
        n_shards:       int = 1,
        dtype:          typ.Any = None,
        hessian:        str = 'gauss_newton',  # or 'exact': the exact Hessian's two-hole assemblies must fit too
) -> int:                                   # the largest chunk_size whose assembly peak fits target_bytes
    '''The largest ``chunk_size`` whose gradient-assembly peak stays within ``target_bytes`` (per device)
    -- the "use my whole device" policy: pass an absolute byte budget (e.g. a fraction of device memory).
    Contrast the device-agnostic :py:func:`estimate_chunk_size`. Same measured per-row cost.

    **Pass ``stack_shape`` if the frame is a batch of base points** (the frame stack ``C``). The budget
    is an absolute byte count and does not scale with ``C``, while the assembly does -- so omitting it
    on a stacked frame returns a chunk up to ``prod(C)`` times too large, which is exactly the
    out-of-memory this policy exists to prevent.

    ``hessian='exact'`` (the same ``chunk_size`` also drives the exact Hessian's probe-derivative two-hole): the
    cap must also hold for the two-hole's assemblies, whose per-row scratch is the larger (see
    :py:func:`estimate_chunk_size`), so it sizes by the larger of the two.'''
    exact = _check_hessian_mode(hessian, 'max_chunk_size_within') == 'exact'
    dtype = np.dtype(np.float32 if dtype is None else dtype)
    ms, tr, tt = tuple(mode_shapes), tuple(tucker_ranks), tuple(tt_ranks)
    cs = tuple(int(c) for c in stack_shape)
    w_local = max(1, int(n_probes) // max(1, int(n_shards)))
    per_row = _assembly_per_row_bytes(ms, tr, tt, order, n_tangent, dtype, cs)
    if exact:
        per_row = max(per_row, _two_hole_assembly_per_row_bytes(ms, tr, tt, order, n_tangent, dtype, cs))
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
        order:      int,                    # highest derivative order
        sum_over_probes: bool = False,      # True: sum the sample stack W (the J^T r back-projection)
        chunk_size: typ.Optional[int] = DEFAULT_CHUNK_SIZE,   # W-chunk size for the gradient assembly; None -> dense. docs/chunking.md
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes (Tucker variation gradient)
    typ.Tuple[NDArray, ...],  # dG_tildes (TT variation gradient)
]:                                          # = T3Variations.data
    '''Variation gradient of :py:func:`tv_probe_derivatives_transpose` from a precomputed frame
    ``sweep``: the adjoint sweeps (``sigma/tau/dxi_tilde`` jets) + the order-less gradient assembly,
    reusing the frame ``(xi, mu, nu, eta)_jets``. The reuse hook for a fitting inner solve.'''
    up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = frame
    xi_jets, mu_jets, nu_jets, eta_jets = sweep
    n_probe = ww[0].ndim - 1   # number of sample-stack (W) axes (ww carries W, no C)

    deta_tildes  = compute_deta_tilde_jets(up_tucker_cores, ztildes)
    tau_tildes   = compute_tau_tilde_jets(left_tt_cores, xi_jets, deta_tildes, mu_jets)
    sigma_tildes = compute_sigma_tilde_jets(right_tt_cores, xi_jets, deta_tildes, nu_jets)
    dxi_tildes   = compute_dxi_tilde_jets(down_tt_cores, mu_jets, nu_jets, sigma_tildes, tau_tildes)

    dU_tildes = assemble_tucker_variation_jets(
        ztildes, dxi_tildes, ww, pp, eta_jets, n_probe, sum_over_probes, chunk_size=chunk_size)
    dG_tildes = assemble_tt_variation_jets(
        sigma_tildes, tau_tildes, deta_tildes, xi_jets, mu_jets, nu_jets, n_probe, sum_over_probes,
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
        order:      int,                    # highest derivative order
        sum_over_probes: bool = False,      # True: sum the sample stack W (the J^T r back-projection)
        chunk_size: typ.Optional[int] = DEFAULT_CHUNK_SIZE,   # W-chunk size for the gradient assembly; None -> dense. docs/chunking.md
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # dU_tildes (Tucker variation gradient)
    typ.Tuple[NDArray, ...],  # dG_tildes (TT variation gradient)
]:                                          # = T3Variations.data
    '''Transpose of :py:func:`tv_probe_derivatives`: back-project residual jets ``ztildes`` into a
    variation gradient ``(dU_tildes, dG_tildes)``. The jet-ified adjoint-state method (t4s.pdf Thm 7):
    every forward contraction is swapped for its binomial jet convolution, then stationarity of the
    Lagrangian gives the adjoint sweeps (``sigma/tau/dxi_tilde`` jets) and the order-less gradient
    assembly -- banded recurrences and order scans throughout, no binomial tensor.

    Full ``W + K + C`` stacking, frame-inner as in plain probing: the residual jets ``ztildes`` carry the
    sample stack ``W``, the tangent stack ``K`` (the forward output's ``K``), and the frame stack ``C``.
    The tangent batch ``K`` always rides through to the variation gradient. With ``sum_over_probes=False``
    the sample stack ``W`` rides through into the variation stack too; with ``True`` it is summed (the
    ``J^T r`` back-projection used for fitting), ``K``/``C`` always kept. Verified against the dense
    adjoint identity ``<r, J v> = <J^T r, v>``.

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
# and dU_tilde = dxi_hat (x) w_jet, with dxi_hat = mu * O * sigma_hat (its two read rows). Half the work of
# the probe transpose (one adjoint sweep, single-term assembly, no nu/eta). Reuses the probe transpose's
# reverse recurrence and two-row correlation; full W + K + C, frame-inner. Verified vs the adjoint identity.


def _sigma_hat_jets_step(
        carry:  NDArray,                                # (order+1,)+W+K+C+(rR(i+1),); axis 0 is the order axis
        data:   typ.Tuple[NDArray, NDArray],            # (Q, xi_jet) for one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:            # (next carry, (carry,))
    '''One core of the seeded adjoint reverse sweep of :py:func:`compute_sigma_hat_jets`: the two-term
    reverse recurrence :py:func:`_adj_prop` (the adjoint of the affine pushthrough). Closure-free scan body
    -- ``docs/contributor/scan_body_principles.md``.'''
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, data)))   # only xnp; it ignores the flag
    Q, xi = data
    return _adj_prop(carry, Q, xi, xnp), (carry,)


def compute_sigma_hat_jets(
        right_tt_cores: typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        xi_jets:        typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        c:              NDArray,                # residual jet (scalar), shape=(order+1)+W+K+C
) -> typ.Tuple[NDArray, ...]:                   # sigma_hats. len=d, elm_shape=(order+1)+W+K+C+(rR(i+1),)
    '''Propagation-only adjoint sweep via Q, seeded at the terminal bond by the residual jet ``c``.

    The apply-transpose analog of :py:func:`compute_sigma_tilde_jets`: no ``deta_tilde`` per-core source,
    and the init carry is the ``c``-seed on the terminal bond (``rR_d = 1``) rather than zeros.
    ``sigma_hats[i]`` is the adjoint of the after-core-``i`` perturbation carry; it carries the tangent
    stack ``K`` (from ``c``). Right-to-left via ``tt_reverse`` (mirroring the ``Q``-sweep there); each step
    is the two-term reverse recurrence of the affine input jet.
    '''
    use_jax = tree_contains_jax((right_tt_cores, xi_jets, c))
    is_uniform = is_ndarray(right_tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)  # scan-style: xscan strips d

    # Polymorphic reverse (uniform tt_reverse keeps the supercore; tt_operations.tt_reverse would iterate d).
    rev_Q = tt_operations.tt_reverse(right_tt_cores)
    rev_xi = xi_jets[::-1]
    # The forward sums the terminal bond (rR_d, not necessarily 1 -- e.g. the corewise frame's own
    # cores), so the adjoint BROADCASTS c over it: seed -> (order+1)+W+K+C+(rR_d,).
    rR_d = right_tt_cores[-1].shape[-1]
    dtype = xnp.result_type(c, right_tt_cores[0], xi_jets[0])    # the sweep's dtype, so the scan carry is invariant
    seed = xnp.broadcast_to(c[..., None].astype(dtype), tuple(c.shape) + (rR_d,))

    _, (rev_sigma_hats,) = xscan(_sigma_hat_jets_step, seed, (rev_Q, rev_xi))
    return rev_sigma_hats[::-1]


def _apply_transpose_assemble_core(
        x:               typ.Tuple[NDArray, NDArray, NDArray,
                                   NDArray, NDArray],  # (xi_jet, mu_jet, sigma_hat, dxi_hat, w_jet), one core
        sum_over_probes: bool,     # caller intent -- no operand carries it; hence the two bodies below
) -> typ.Tuple[NDArray, NDArray]:  # (dU_tilde, dG_tilde); W summed (K+C+...) or kept (W+K+C+...)
    '''One core of the gradient-assembly map of :py:func:`_apply_derivatives_transpose_from_jets`
    (ragged and uniform alike), shared by the two closure-free map bodies below --
    ``docs/contributor/scan_body_principles.md``. ``dG = sum_{r, s<2} C(r+s, r) mu^(r) (x) xi^(s) (x)
    sigma^^(r+s)`` is banded in the affine ``xi``: the pair ``sum_r mu^(r) (x) [U^k sigma^]^(r)``
    (:py:func:`_shift_up_pair`) meets ``xi^(k)``.'''
    xi, mu, sh, dxh, wj = x
    xnp, _, _ = get_backend(True, tree_contains_jax(x))      # only xnp
    s_size = min(2, mu.shape[0])    # the w / dxi input jets carry orders {0, 1}, capped at order
    n_probe = wj.ndim - 2           # len(W), from the W-only ambient jet w_jet ((2,)+W+(Ni,))
    if sum_over_probes:
        dG, dU = 'kWKCab,kWCi->KCaib', 'uWKCa,uWo->KCao'
    else:
        dG, dU = 'kWKCab,kWCi->WKCaib', 'uWKCa,uWo->WKCao'
    pair = contractions.contract('krWKCb,rWCa->kWKCab', _shift_up_pair(sh, xnp), mu, len_W=n_probe)
    dG_t = contractions.contract(dG, pair, xi, len_W=n_probe)
    dU_t = contractions.contract(dU, dxh[:s_size], wj[:s_size])
    return (dU_t, dG_t)


def _apply_transpose_assemble_step_summed(x):    # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=True`` variant of :py:func:`_apply_transpose_assemble_core`.'''
    return _apply_transpose_assemble_core(x, True)


def _apply_transpose_assemble_step_unsummed(x):  # closure-free map body -- docs/contributor/scan_body_principles.md
    '''``sum_over_probes=False`` variant of :py:func:`_apply_transpose_assemble_core`.'''
    return _apply_transpose_assemble_core(x, False)


def _dxi_hat_order_step(
        carry:  typ.Tuple[NDArray, NDArray],   # (dxi^, O): (2,)+W+K+C+(nOi,) ; C+(rLi,nOi,rR(i+1))
        xs:     typ.Tuple[NDArray, NDArray],   # (mu^(j), [U^u sigma^]^(j)): W+C+(rLi,) ; (2,)+W+K+C+(rR(i+1),)
) -> typ.Tuple[typ.Tuple[NDArray, NDArray], typ.Tuple]:   # (next carry, ())
    '''One order ``j`` of the two-row dxi^ correlation ``dxi^^(u) += (mu^(j) O) . [U^u sigma^]^(j)`` -- the
    ``mu O sigma~`` half of :py:func:`_dxi_tilde_order_step`. Closure-free scan body.'''
    dxh, O = carry
    mu_j, sh_pair_j = xs
    mO = contractions.contract('WCa,Caib->WCib', mu_j, O)
    return (dxh + contractions.contract('WCib,uWKCb->uWKCi', mO, sh_pair_j, len_C=O.ndim - 3), O), ()


def _dxi_hat_jets_core(
        data:       typ.Tuple[NDArray, NDArray, NDArray],  # (O, mu_jet, sigma_hat), one core
        is_uniform: bool,                       # inner-scan backend (a real lax.scan on the uniform path)
) -> typ.Tuple[NDArray]:                        # (dxi_hat,) elm_shape=(2,)+W+K+C+(nOi,)
    '''One core of the adjoint-down combine ``dxi^ = mu O sigma^`` of
    :py:func:`_apply_derivatives_transpose_from_jets`: only its two read rows, as an order scan.'''
    O, mu, sh = data
    xnp, _, xscan = get_backend(is_uniform, tree_contains_jax(data))
    dxh0 = xnp.zeros((2,) + sh.shape[1:-1] + (O.shape[-2],), xnp.result_type(O, mu, sh))   # W+K+C off sigma^
    (dxh, _), _ = xscan(_dxi_hat_order_step, (dxh0, O), (mu, _shift_up_pair(sh, xnp, axis=1)))
    return (dxh,)


def _dxi_hat_jets_step_uniform(data):  # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Uniform variant of :py:func:`_dxi_hat_jets_core` (a real inner scan).'''
    return _dxi_hat_jets_core(data, True)


def _dxi_hat_jets_step_ragged(data):   # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Ragged variant of :py:func:`_dxi_hat_jets_core` (the inner order loop is a Python loop).'''
    return _dxi_hat_jets_core(data, False)


def _apply_derivatives_transpose_from_jets(
        c:                  NDArray,                # residual jet (scalar), shape=(order+1)+W+K+C
        xi_jets:            typ.Sequence[NDArray],  # frame input jets, len=d, elm_shape=(2,)+W+C+(nUi,)
        mu_jets:            typ.Sequence[NDArray],  # frame left jets,  len=d, elm_shape=(order+1,)+W+C+(rLi,)
        w_jets:             typ.Sequence[NDArray],  # ambient input jets, len=d, elm_shape=(2,)+W+(Ni,)
        down_tt_cores:      typ.Sequence[NDArray],  # O.  len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        right_tt_cores:     typ.Sequence[NDArray],  # Q.  len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        sum_over_probes:    bool,
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],    # dU_tildes. len=d, elm_shape=[W+]K+C+(nOi,Ni)
    typ.Tuple[NDArray, ...],    # dG_tildes. len=d, elm_shape=[W+]K+C+(rLi,nUi,rRi)
]:
    '''Shared tail of apply/tv_entries_derivatives_transpose (they differ only in how xi_jets /
    w_jets are formed). Runs the seeded sigma_hat sweep, the two-row dxi_hat combine, and the single-term
    gradient assembly (dG = mu (x) xi (x) sigma_hat, dU = dxi_hat (x) w_jet), mapped over cores on both
    layouts (``len(W)`` is read off the W-only ``w_jets``). Returns (dU, dG).'''
    use_jax = tree_contains_jax((c, xi_jets, mu_jets, w_jets, down_tt_cores, right_tt_cores))
    is_uniform = is_ndarray(down_tt_cores)
    xnp, xmap, _ = get_backend(is_uniform, use_jax)
    sigma_hats = compute_sigma_hat_jets(right_tt_cores, xi_jets, c)   # polymorphic

    dxh_step = _dxi_hat_jets_step_uniform if is_uniform else _dxi_hat_jets_step_ragged
    (dxi_hats,) = xmap(dxh_step, (down_tt_cores, mu_jets, sigma_hats))

    step = (_apply_transpose_assemble_step_summed if sum_over_probes
            else _apply_transpose_assemble_step_unsummed)
    (dU_tildes, dG_tildes) = xmap(step, (xi_jets, mu_jets, sigma_hats, dxi_hats, w_jets))
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
    w_jets = build_input_jets(ww, pp)               # ambient input jets (value, direction) for dU
    return _apply_derivatives_transpose_from_jets(
        c, xi_jets, mu_jets, w_jets, down_tt_cores, right_tt_cores, sum_over_probes)


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
    kept. Verified vs the dense adjoint identity ``<c, J v> = <J^T c, v>``.

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
    ww_onehot = _onehot_vectors(index, up_tucker_cores)   # e_{index}, elm_shape=W+(Ni,)
    xnp, _, _ = get_backend(True, tree_contains_jax((up_tucker_cores, pp)))     # only xnp
    dtype = xnp.result_type(up_tucker_cores[0], pp[0])      # the one-hots are built in the default dtype
    ww_onehot = (ww_onehot.astype(dtype) if is_ndarray(ww_onehot) else tuple(w.astype(dtype) for w in ww_onehot))
    w_jets = build_input_jets(ww_onehot, pp)         # ambient input jets (one-hot, direction) for dU
    return _apply_derivatives_transpose_from_jets(
        c, xi_jets, mu_jets, w_jets, down_tt_cores, right_tt_cores, sum_over_probes)


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
        chunk_size: typ.Optional[int] = DEFAULT_CHUNK_SIZE,   # W-chunk size for the gradient assembly; None -> dense. docs/chunking.md
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
    return tv_probe_derivatives_transpose(
        ztildes, ww, pp, fv_conversions.t3_corewise_frame(core_pair), order,
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
    return tv_apply_derivatives_transpose(
        c, ww, pp, fv_conversions.t3_corewise_frame(core_pair), order, sum_over_probes=sum_over_probes)


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
    return tv_entries_derivatives_transpose(
        c, index, pp, fv_conversions.t3_corewise_frame(core_pair), order, sum_over_probes=sum_over_probes)


###############################################################
####    Corewise two-hole: the probe-derivative Hessian    ####
###############################################################
#
# The derivative of the corewise transpose above along a direction in the cores. For a fixed residual
# jet z~ (omega^2 applied) the corewise transpose is the gradient of theta -> <z~, S(Phi(theta))>, so its
# derivative along theta_hat is that function's core-coordinate Hessian applied to theta_hat -- the
# "two-hole" sweep (two cores replaced at once), the sampling half of the exact Riemannian Hessian's
# curvature term K_z theta_hat = d/ds grad_theta <z~, S(Phi(theta + s theta_hat))>.
#
# Every quantity of the GN transpose's Lagrangian is multilinear in (cores, vectors), so its derivative
# along theta_hat is the product rule over its factors: each edge variable gains a DOTTED companion
# carrying the tangent stack K (the directions' stack), while the undotted state carries W + C only
# (the residual has no K). At the corewise frame (U, G, G, G) of theta = (U, G):
#
#   1. undotted state: xi, mu (reused from the model's sweep -- identical at the manifold and corewise
#      frames of the base point), nu re-swept through G, and the adjoint tildes deta~, tau~, sigma~;
#   2. dotted forward = the GN tangent forward at Q = O = P = G along (Udot, Gdot): xidot = Udot (w, p),
#      mudot / nudot = the sigma / tau jets, etadot = the deta jets;
#   3. dotted adjoint-up lift deta~dot = Udot z~;
#   4. dotted adjoint sweeps tau~dot / sigma~dot: the product rule of _adj_tilde_step;
#   5. dotted dxi~: the product rule of _dxi_tilde_order_step (its two read rows);
#   6. assemblies: Tucker etadot (x) z~ + dxi~dot (x) w; TT factored by the product rule.
#
# No corewise eta and no corewise dxi~ is ever formed: the Hessian needs only their derivatives.
# Each dotted kernel is the product rule of the GN kernel it differentiates, in its form: a fused banded
# recurrence where a factor is an input jet, an order scan with a carried shifted jet where the convolution
# is full (the jet-kernel note near the top of this module). No kernel contracts the binomial tensor; the
# one-dense-contraction-per-term twins exist only as test references
# (tests/backend/_curvature_oracles.py). Stacks: z~ and the undotted jets W + C, the directions and every
# dotted quantity K + C (W + K + C on the jets), the output K + C (W summed). Polymorphic: ragged tuples or
# uniform supercores (a uniform caller masks theta and theta_hat once, packs, delegates here, and re-masks
# the output -- the kernel itself is mask-free).


def _deta_tilde_tangent_jets_step(
        data: typ.Tuple[NDArray, NDArray, NDArray],  # (U, Udot, ztilde) for one core
) -> typ.Tuple[NDArray]:                             # (deta_tilde_dot,) elm_shape=(order+1,)+W+K+C+(nUi,)
    '''One core of :py:func:`compute_deta_tilde_tangent_jets` (ragged path only). The C-only ``U`` is
    passed only to recover ``len(C)``. Closure-free map body -- ``docs/contributor/scan_body_principles.md``.'''
    U, Ud, zt = data
    return (contractions.contract('tWCo,KCio->tWKCi', zt, Ud, len_C=U.ndim - 2),)


def compute_deta_tilde_tangent_jets(
        up_tucker_cores:        typ.Sequence[NDArray],  # U.    len=d, elm_shape=C+(nUi,Ni)
        up_tucker_cores_dot:    typ.Sequence[NDArray],  # Udot. len=d, elm_shape=K+C+(nUi,Ni)
        ztildes:                typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+C+(Ni,)
) -> typ.Tuple[NDArray, ...]:                           # deta_tildes_dot. len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
    '''The derivative of :py:func:`compute_deta_tilde_jets` along the Tucker directions:
    ``deta_tilde_dot_i = Udot_i z~_i`` (the residual is fixed, so only the core moves). The order axis
    rides through; the tangent stack ``K`` comes from ``Udot``.'''
    use_jax = tree_contains_jax((up_tucker_cores, up_tucker_cores_dot, ztildes))
    is_uniform = is_ndarray(up_tucker_cores)
    xnp, xmap, _ = get_backend(is_uniform, use_jax)
    if is_uniform:
        return contractions.contract('dtWCo,dKCio->dtWKCi', ztildes, up_tucker_cores_dot,
                                     len_C=up_tucker_cores.ndim - 3)
    (deta_tildes_dot,) = xmap(_deta_tilde_tangent_jets_step, (up_tucker_cores, up_tucker_cores_dot, ztildes))
    return deta_tildes_dot


def _adj_tangent_src_order_step(
        carry:  typ.Tuple[NDArray, ...],       # (src, U^r deta~, U^r deta~_dot, P, P_dot); the cores ride unchanged
        xs:     typ.Tuple[NDArray, NDArray, NDArray],  # (edge^(r), edge_dot^(r), 1/r!): W+C+(rLi,) ; W+K+C+(rLi,) ; ()
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple]:   # (next carry, ())
    '''One edge order ``r`` of the dotted adjoint source: the product rule of :py:func:`_adj_src_order_step`,
    ``src_dot^(s) += d(edge^(r) P) . [U^r deta~]^(s) + (edge^(r) P) . [U^r deta~_dot]^(s)`` with
    ``d(edge P) = edge_dot P + edge P_dot`` (live slice ``W*K*r^2``), then up-shift both carried jets.
    Closure-free scan body.'''
    src, deta_r, deta_dot_r, P, P_dot = carry
    edge_r, edge_dot_r, w_r = xs
    xnp, _, _ = get_backend(True, tree_contains_jax((carry, xs)))      # only xnp
    nf = P.ndim - 3                                                    # P is C-only: pins len(C)
    ew, edw = w_r * edge_r, w_r * edge_dot_r
    ep = contractions.contract('WCa,Caib->WCib', ew, P)                           # edge^(r) P
    dep = (contractions.contract('WKCa,Caib->WKCib', edw, P)                      # d(edge^(r) P): K on the edge
           + contractions.contract('WCa,KCaib->WKCib', ew, P_dot, len_C=nf))      # ... and on the core
    src = (src + contractions.contract('WKCib,tWCi->tWKCb', dep, deta_r, len_C=nf)
           + contractions.contract('WCib,tWKCi->tWKCb', ep, deta_dot_r, len_C=nf))       # K on deta
    return (src, _shift_up(deta_r, xnp), _shift_up(deta_dot_r, xnp), P, P_dot), ()


def _adj_tilde_tangent_step(
        carry_dot:      NDArray,    # dotted tilde entering this core, (order+1,)+W+K+C+(rLi,)
        carry:          NDArray,    # undotted tilde entering this core, (order+1,)+W+C+(rLi,)
        P:              NDArray,    # frame core,        C+(rLi,nUi,rL(i+1))
        P_dot:          NDArray,    # its direction,   K+C+(rLi,nUi,rL(i+1))
        xi:             NDArray,    # input jet,        (2,)+W+C+(nUi,)
        xi_dot:         NDArray,    # its derivative,   (2,)+W+K+C+(nUi,)
        deta_t:         NDArray,    # adjoint-up jet,   (order+1,)+W+C+(nUi,)
        deta_t_dot:     NDArray,    # its derivative,   (order+1,)+W+K+C+(nUi,)
        edge:           NDArray,    # frame edge jet,   (order+1,)+W+C+(rLi,)
        edge_dot:       NDArray,    # its derivative,   (order+1,)+W+K+C+(rLi,)
        n_frame:        int,        # len(C): no operand of the K-on-core terms pins it
        is_uniform:     bool,       # inner-scan backend (a real lax.scan on the uniform path)
) -> NDArray:                       # next dotted tilde, (order+1,)+W+K+C+(rL(i+1),)
    '''One step of the dotted adjoint sweep: the product rule of :py:func:`_adj_tilde_step`, in its form.

    ``_adj_tilde_step`` is ``prop(carry; P xi) + src(edge P; deta)``, each bilinear in a pair of factors,
    so its derivative is ``prop(carry_dot; P xi) + prop(carry; d(P xi)) + src(d(edge P); deta) +
    src(edge P; deta_dot)`` with ``d(P xi) = P_dot xi + P xi_dot`` and ``d(edge P) = edge_dot P + edge
    P_dot`` -- the six dotted terms (K on the carry / core / xi ; edge / core / deta), paired by shared
    factor:

    - **prop** stays the two-term REVERSE recurrence of the affine input jet, fused into one GEMM per
      factor pair (two GEMMs: the dotted carry against ``P xi``; the undotted carry against ``d(P xi)``);
    - **src** stays the full reverse convolution, an inner order scan over the edge order ``r``
      (:py:func:`_adj_tangent_src_order_step`) whose live slice is ``W*K*r^2`` (``_adj_tilde_step``'s
      ``W*r^2``, times ``K``).

    ``n_frame`` is an argument because the K-on-core contractions have no C-only operand to pin it, and the
    production step's own inference (from a C-only ``P``) cannot be reused term by term. The callers (the
    two scan-body variants) read it off the C-only frame core.'''
    xnp, _, xscan = get_backend(is_uniform, tree_contains_jax(
        (carry_dot, carry, P, P_dot, xi, xi_dot, deta_t, deta_t_dot, edge, edge_dot)))
    order = carry_dot.shape[0] - 1             # the carries are stacked over derivative orders
    s_size = min(2, order + 1)                 # affine input jet (xi): orders {0, 1}
    nf = n_frame

    # --- prop: two-term reverse recurrence per factor pair. prop^(s) = c^(s) . Pxi^(0) + (s+1) c^(s+1) . Pxi^(1)
    Pxi = contractions.contract('Caib,kWCi->kWCab', P, xi[:s_size])                         # (k, W, C, a, b)
    dPxi = (contractions.contract('KCaib,kWCi->kWKCab', P_dot, xi[:s_size], len_C=nf)         # K on the core
            + contractions.contract('Caib,kWKCi->kWKCab', P, xi_dot[:s_size]))                # K on the input
    if s_size > 1:
        prop = (contractions.contract('ksWKCa,kWCab->sWKCb', _shift_up_pair(carry_dot, xnp), Pxi, len_C=nf)
                + contractions.contract('ksWCa,kWKCab->sWKCb', _shift_up_pair(carry, xnp), dPxi, len_C=nf))
    else:                                       # order 0: only k = 0 survives
        prop = (contractions.contract('sWKCa,WCab->sWKCb', carry_dot, Pxi[0], len_C=nf)
                + contractions.contract('sWCa,WKCab->sWKCb', carry, dPxi[0], len_C=nf))

    # --- src: full reverse convolution -- an inner scan over the edge order r, live slice W*K*r^2.
    src0 = xnp.zeros(carry_dot.shape[:-1] + (P.shape[-1],),                         # (order+1,)+W+K+C+(b,)
                     xnp.result_type(P, P_dot, deta_t, deta_t_dot, edge, edge_dot))
    (src, _, _, _, _), _ = xscan(_adj_tangent_src_order_step, (src0, deta_t, deta_t_dot, P, P_dot),
                                 (edge, edge_dot, _inverse_factorials(order, edge, xnp)))

    return prop + src


def _adj_tangent_sweep_scanned_step_uniform(
        carry_dot:  NDArray,        # (order+1,)+W+K+C+(rLi,); axis 0 is the order axis
        data:       typ.Tuple[NDArray, ...],    # (P, P_dot, xi, xi_dot, deta_t, deta_t_dot, edge, edge_dot, carry), one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:    # (next carry, (carry_dot,))
    '''Uniform variant of :py:func:`_adj_tilde_tangent_step` as a closure-free scan body: the inner order
    loop is a real scan. ``len(C)`` is read off the C-only frame core.
    ``docs/contributor/scan_body_principles.md``.'''
    P, P_dot, xi, xi_dot, deta_t, deta_t_dot, edge, edge_dot, carry = data
    return _adj_tilde_tangent_step(carry_dot, carry, P, P_dot, xi, xi_dot, deta_t, deta_t_dot, edge, edge_dot,
                                   P.ndim - 3, True), (carry_dot,)


def _adj_tangent_sweep_scanned_step_ragged(
        carry_dot:  NDArray,        # (order+1,)+W+K+C+(rLi,); axis 0 is the order axis
        data:       typ.Tuple[NDArray, ...],    # (P, P_dot, xi, xi_dot, deta_t, deta_t_dot, edge, edge_dot, carry), one core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray]]:    # (next carry, (carry_dot,))
    '''Ragged variant of :py:func:`_adj_tilde_tangent_step` as a closure-free scan body: the inner order
    loop is a Python loop. ``docs/contributor/scan_body_principles.md``.'''
    P, P_dot, xi, xi_dot, deta_t, deta_t_dot, edge, edge_dot, carry = data
    return _adj_tilde_tangent_step(carry_dot, carry, P, P_dot, xi, xi_dot, deta_t, deta_t_dot, edge, edge_dot,
                                   P.ndim - 3, False), (carry_dot,)


def _adj_tangent_sweep_scanned(P_cores, P_cores_dot, xi_jets, xi_jets_dot, deta_tildes, deta_tildes_dot,
                               edge_jets, edge_jets_dot, tildes):
    '''The derivative of :py:func:`_adj_sweep_scanned` along the dotted operands (module-private): the same
    left-to-right core sweep, its carry the dotted tilde, with the undotted ``tildes`` (the carry entering
    each core) riding as scan data. Shared by :py:func:`compute_tau_tilde_tangent_jets` /
    :py:func:`compute_sigma_tilde_tangent_jets`.'''
    use_jax = tree_contains_jax((P_cores, P_cores_dot, xi_jets, xi_jets_dot, deta_tildes, deta_tildes_dot,
                                 edge_jets, edge_jets_dot, tildes))
    is_uniform = is_ndarray(P_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)
    step = _adj_tangent_sweep_scanned_step_uniform if is_uniform else _adj_tangent_sweep_scanned_step_ragged

    # the dotted carry is W+K+C; deta_tildes_dot carries K, so it supplies the init's leading stack (and order)
    rL0 = P_cores[0].shape[-3]
    dtype = xnp.result_type(*(a[0] for a in (P_cores, P_cores_dot, xi_jets, xi_jets_dot, deta_tildes, deta_tildes_dot,
                                             edge_jets, edge_jets_dot, tildes)))
    init = xnp.zeros(deta_tildes_dot[0].shape[:-1] + (rL0,), dtype)
    _, (tildes_dot,) = xscan(step, init, (P_cores, P_cores_dot, xi_jets, xi_jets_dot, deta_tildes,
                                          deta_tildes_dot, edge_jets, edge_jets_dot, tildes))
    return tildes_dot


def compute_tau_tilde_tangent_jets(
        left_tt_cores:      typ.Sequence[NDArray],  # P.        len=d, elm_shape=C+(rLi,nUi,rL(i+1))
        left_tt_cores_dot:  typ.Sequence[NDArray],  # P_dot.    len=d, elm_shape=K+C+(rLi,nUi,rL(i+1))
        xi_jets:            typ.Sequence[NDArray],  # len=d, elm_shape=(2,)+W+C+(nUi,)
        xi_jets_dot:        typ.Sequence[NDArray],  # len=d, elm_shape=(2,)+W+K+C+(nUi,)
        deta_tildes:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(nUi,)
        deta_tildes_dot:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        mu_jets:            typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        mu_jets_dot:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
        tau_tildes:         typ.Sequence[NDArray],  # = compute_tau_tilde_jets(P, xi, deta~, mu). len=d, elm_shape=(order+1,)+W+C+(rLi,)
) -> typ.Tuple[NDArray, ...]:                       # tau_tildes_dot. len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
    '''The derivative of :py:func:`compute_tau_tilde_jets` along ``(P_dot, xi_dot, deta~_dot, mu_dot)``, in
    its form (the product rule of the scanned adjoint step: a fused reverse recurrence plus an inner order
    scan; see :py:func:`_adj_tilde_tangent_step`). The undotted operands carry ``W + C``, the dotted
    ``W + K + C``. Dense reference: the six-term dense binomial twin in the test helper (equal to ~1e-15).'''
    return _adj_tangent_sweep_scanned(left_tt_cores, left_tt_cores_dot, xi_jets, xi_jets_dot, deta_tildes,
                                      deta_tildes_dot, mu_jets, mu_jets_dot, tau_tildes)


def compute_sigma_tilde_tangent_jets(
        right_tt_cores:     typ.Sequence[NDArray],  # Q.        len=d, elm_shape=C+(rRi,nUi,rR(i+1))
        right_tt_cores_dot: typ.Sequence[NDArray],  # Q_dot.    len=d, elm_shape=K+C+(rRi,nUi,rR(i+1))
        xi_jets:            typ.Sequence[NDArray],  # len=d, elm_shape=(2,)+W+C+(nUi,)
        xi_jets_dot:        typ.Sequence[NDArray],  # len=d, elm_shape=(2,)+W+K+C+(nUi,)
        deta_tildes:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(nUi,)
        deta_tildes_dot:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        nu_jets:            typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        nu_jets_dot:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        sigma_tildes:       typ.Sequence[NDArray],  # = compute_sigma_tilde_jets(Q, xi, deta~, nu). len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
) -> typ.Tuple[NDArray, ...]:                       # sigma_tildes_dot. len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
    '''The derivative of :py:func:`compute_sigma_tilde_jets` -- the reverse of
    :py:func:`compute_tau_tilde_tangent_jets`: ``tt_reverse`` the cores and their directions, reverse the
    jet sequences, run the sweep, reverse the result.'''
    reverse = tt_operations.tt_reverse
    rev = _adj_tangent_sweep_scanned(reverse(right_tt_cores), reverse(right_tt_cores_dot), xi_jets[::-1],
                                     xi_jets_dot[::-1], deta_tildes[::-1], deta_tildes_dot[::-1], nu_jets[::-1],
                                     nu_jets_dot[::-1], sigma_tildes[::-1])
    return rev[::-1]


def _dxi_tilde_tangent_order_step(
        carry:  typ.Tuple[NDArray, NDArray, NDArray],  # (dxi~_dot, O, O_dot): (2,)+W+K+C+(nOi,) ; C+(...) ; K+C+(...)
        xs:     typ.Tuple[NDArray, ...],               # order j of (mu, mu_dot, nu, nu_dot) and of the reverse pairs of (sigma~, sigma~_dot, tau~, tau~_dot)
) -> typ.Tuple[typ.Tuple[NDArray, NDArray, NDArray], typ.Tuple]:   # (next carry, ())
    '''One order ``j`` of the dotted two-row dxi~ correlation: the product rule of
    :py:func:`_dxi_tilde_order_step`, six terms with ``K`` on the dotted factor of each, grouped by shared
    factor -- ``d(mu O) . [U^u sigma~] + (mu O) . [U^u sigma~_dot] + [U^u tau~_dot] . (O nu) + [U^u tau~] .
    d(O nu)``. Closure-free scan body (the cores ride the carry).'''
    dxi, O, Od = carry
    mu_j, mud_j, nu_j, nud_j, sp_j, spd_j, tp_j, tpd_j = xs
    nf = O.ndim - 3                                                    # O is C-only: pins len(C)
    mO = contractions.contract('WCa,Caib->WCib', mu_j, O)
    dmO = (contractions.contract('WKCa,Caib->WKCib', mud_j, O)                    # K on mu_dot
           + contractions.contract('WCa,KCaib->WKCib', mu_j, Od, len_C=nf))       # K on the core
    On = contractions.contract('Caib,WCb->WCai', O, nu_j)
    dOn = (contractions.contract('KCaib,WCb->WKCai', Od, nu_j, len_C=nf)          # K on the core
           + contractions.contract('Caib,WKCb->WKCai', O, nud_j))                 # K on nu_dot
    dxi = (dxi + contractions.contract('WKCib,uWCb->uWKCi', dmO, sp_j, len_C=nf)
           + contractions.contract('WCib,uWKCb->uWKCi', mO, spd_j, len_C=nf)
           + contractions.contract('uWKCa,WCai->uWKCi', tpd_j, On, len_C=nf)
           + contractions.contract('uWCa,WKCai->uWKCi', tp_j, dOn, len_C=nf))
    return (dxi, O, Od), ()


def _dxi_tilde_tangent_jets_core(
        data:       typ.Tuple[NDArray, ...],   # (O, O_dot, mu, mu_dot, nu, nu_dot, sigma~, sigma~_dot, tau~, tau~_dot), one core
        is_uniform: bool,                      # inner-scan backend (a real lax.scan on the uniform path)
) -> typ.Tuple[NDArray]:                       # (dxi_tilde_dot,) elm_shape=(2,)+W+K+C+(nOi,)
    '''One core of :py:func:`compute_dxi_tilde_tangent_jets`: the order scan of
    :py:func:`_dxi_tilde_tangent_order_step`, the tildes' reverse banded pairs scanned beside the jets.'''
    O, Od, mu, mud, nu, nud, st, std, tt, ttd = data
    xnp, _, xscan = get_backend(is_uniform, tree_contains_jax(data))
    dxi0 = xnp.zeros((2,) + std.shape[1:-1] + (O.shape[-2],), xnp.result_type(*data))   # W+K+C off a dotted tilde
    pairs = tuple(_shift_up_pair(j, xnp, axis=1) for j in (st, std, tt, ttd))
    (dxi, _, _), _ = xscan(_dxi_tilde_tangent_order_step, (dxi0, O, Od), (mu, mud, nu, nud) + pairs)
    return (dxi,)


def _dxi_tilde_tangent_jets_step_uniform(data):  # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Uniform variant of :py:func:`_dxi_tilde_tangent_jets_core` (a real inner scan).'''
    return _dxi_tilde_tangent_jets_core(data, True)


def _dxi_tilde_tangent_jets_step_ragged(data):   # closure-free map body -- docs/contributor/scan_body_principles.md
    '''Ragged variant of :py:func:`_dxi_tilde_tangent_jets_core` (the inner order loop is a Python loop).'''
    return _dxi_tilde_tangent_jets_core(data, False)


def compute_dxi_tilde_tangent_jets(
        down_tt_cores:      typ.Sequence[NDArray],  # O.        len=d, elm_shape=C+(rLi,nOi,rR(i+1))
        down_tt_cores_dot:  typ.Sequence[NDArray],  # O_dot.    len=d, elm_shape=K+C+(rLi,nOi,rR(i+1))
        mu_jets:            typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        mu_jets_dot:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
        nu_jets:            typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        nu_jets_dot:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        sigma_tildes:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_tildes_dot:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_tildes:         typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        tau_tildes_dot:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
) -> typ.Tuple[NDArray, ...]:                       # dxi_tildes_dot. len=d, elm_shape=(2,)+W+K+C+(nOi,): rows u = 0, 1 only
    '''The derivative of :py:func:`compute_dxi_tilde_jets`: the product rule of its two adjoint-hooked
    combines (``tau~ O nu`` and ``mu O sigma~``), six terms with ``K`` on the dotted factor of each, in its
    form -- only the two read rows ``u in {0, 1}``, an order scan mapped over cores.'''
    use_jax = tree_contains_jax((down_tt_cores, down_tt_cores_dot, mu_jets, mu_jets_dot, nu_jets, nu_jets_dot,
                                 sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot))
    is_uniform = is_ndarray(down_tt_cores)
    xnp, xmap, _ = get_backend(is_uniform, use_jax)
    step = _dxi_tilde_tangent_jets_step_uniform if is_uniform else _dxi_tilde_tangent_jets_step_ragged
    (dxi_tildes_dot,) = xmap(step, (down_tt_cores, down_tt_cores_dot, mu_jets, mu_jets_dot, nu_jets, nu_jets_dot,
                                    sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot))
    return dxi_tildes_dot


def _tucker_tangent_variation_jets_step(
        x: typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (ztilde, dxi_tilde_dot, eta_dot, w_jet), one core
) -> typ.Tuple[NDArray]:                                   # (dU,) K+C+(nOi,Ni), W summed
    '''One core of :py:func:`_assemble_tucker_tangent_variation_jets_dense` (ragged path only).
    Closure-free map body -- ``docs/contributor/scan_body_principles.md``.'''
    zt, dxtd, etad, wj = x
    s_size = min(2, etad.shape[0])          # the w input jet carries orders {0, 1}, capped at order
    n_probe = wj.ndim - 2                   # len(W), off the W-only ambient jet (2,)+W+(Ni,)
    return (contractions.contract('tWKCa,tWCo->KCao', etad, zt, len_W=n_probe)
            + contractions.contract('uWKCa,uWo->KCao', dxtd[:s_size], wj[:s_size]),)


def _assemble_tucker_tangent_variation_jets_dense(ztildes, dxi_tildes_dot, ww, pp, eta_jets_dot, n_probe):
    '''The dense (unchunked) Tucker assembly of :py:func:`assemble_tucker_tangent_variation_jets`, W summed:
    the per-chunk kernel of the chunked path (``n_probe = 1`` there).'''
    use_jax = tree_contains_jax((ztildes, dxi_tildes_dot, ww, pp, eta_jets_dot))
    is_uniform = is_ndarray(eta_jets_dot)
    _, xmap, _ = get_backend(is_uniform, use_jax)
    w_jets = build_input_jets(ww, pp)          # ambient input jets (value, direction): the dU leg
    s_size = min(2, eta_jets_dot[0].shape[0])
    if is_uniform:
        # d-prefixed; the order axis sits at supercore axis 1, so the order-<=1 slice is [:, :s_size]
        return (contractions.contract('dtWKCa,dtWCo->dKCao', eta_jets_dot, ztildes, len_W=n_probe)
                + contractions.contract('duWKCa,duWo->dKCao', dxi_tildes_dot[:, :s_size], w_jets[:, :s_size]))
    (dU,) = xmap(_tucker_tangent_variation_jets_step, (ztildes, dxi_tildes_dot, eta_jets_dot, w_jets))
    return dU


def _tucker_tangent_chunk_summed(co, extra):    # extra unused; the common chunk-fn signature
    '''One W-chunk of the Tucker tangent assembly at the chunked path's pinned ``n_probe = 1``.'''
    return _assemble_tucker_tangent_variation_jets_dense(*co, 1)


def assemble_tucker_tangent_variation_jets(
        ztildes:            typ.Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+C+(Ni,)
        dxi_tildes_dot:     typ.Sequence[NDArray],  # len=d, elm_shape=(2,)+W+K+C+(nOi,): the two read rows
        ww:                 typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:                 typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        eta_jets_dot:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(nOi,)
        n_probe:            int,                    # len(W)
        chunk_size:         typ.Optional[int] = DEFAULT_CHUNK_SIZE,   # W-chunk size; None (or >= W) -> dense
) -> typ.Tuple[NDArray, ...]:                       # dU. len=d, elm_shape=K+C+(nOi,Ni) (W summed)
    '''The derivative of the Tucker variation-gradient assembly (W summed).

    ``sum_t etadot^(t) (x) z~^(t) + sum_u dxi~dot^(u) (x) w^(u)`` (the residual and the ambient input jet are
    fixed; the autosummary first line above stays literal-free on purpose). W-chunked exactly as
    :py:func:`assemble_tucker_variation_jets` (on the uniform path -- a ``lax.scan`` on jax, where the memory
    win is, an eager loop on numpy; ragged, ``None`` and ``W <= chunk_size`` run dense); see :doc:`/chunking`.'''
    ops = (ztildes, dxi_tildes_dot, ww, pp, eta_jets_dot)
    w_axes = (2, 2, 1, 1, 2)                          # per-operand W axis (ww/pp carry no order axis)
    dense = lambda: _assemble_tucker_tangent_variation_jets_dense(*ops, n_probe)
    if chunk_size is None or not is_ndarray(eta_jets_dot):
        return dense()
    use_jax = tree_contains_jax(ops)
    W = math.prod(eta_jets_dot.shape[2:2 + n_probe])
    if W <= chunk_size:
        return dense()
    ops, w_axes = _flatten_w(ops, w_axes, n_probe)     # -> one probe super-axis; n_probe == 1 holds
    return _wchunked_sum(_tucker_tangent_chunk_summed, None, ops, w_axes, W, chunk_size, use_jax)


@functools.lru_cache(maxsize=None)
def _tt_tangent_det_lag_step(
        n_probe:    int,            # len(W): no operand of the deta~ group is W-only or C-only
):                                  # -> the scan body (carry, xs) -> (carry, ())
    '''The inner scan body of the factored TT tangent assembly's deta~ group: the product rule of
    :py:func:`_tt_det_lag_step` (W summed). The carry is ``(acc, U^r deta~, U^r deta~_dot, nu, nu_dot)``, the xs
    ``(mu^(r), mu_dot^(r), 1/r!)``; per step the shifted deta~ pair folds against ``nu`` / ``nu_dot`` once
    (undotted and dotted folds) and meets ``mu^(r)`` / ``mu_dot^(r)``. Memoized on its one static
    (``docs/contributor/scan_body_principles.md``, principle 4).'''
    def _step(carry, xs):
        acc, deta_r, deta_dot_r, nu, nu_dot = carry
        mu_r, mu_dot_r, w_r = xs
        xnp, _, _ = get_backend(True, tree_contains_jax((carry, xs)))      # only xnp
        fold = contractions.contract('sWCi,sWCb->WCib', deta_r, nu)
        fold_d = (contractions.contract('sWKCi,sWCb->WKCib', deta_dot_r, nu, len_W=n_probe)
                  + contractions.contract('sWCi,sWKCb->WKCib', deta_r, nu_dot, len_W=n_probe))
        acc = (acc + contractions.contract('WKCa,WCib->KCaib', w_r * mu_dot_r, fold, len_W=n_probe)
               + contractions.contract('WCa,WKCib->KCaib', w_r * mu_r, fold_d, len_W=n_probe))
        return (acc, _shift_up(deta_r, xnp), _shift_up(deta_dot_r, xnp), nu, nu_dot), ()
    return _step


@functools.lru_cache(maxsize=None)
def _tt_tangent_variation_jets_step(
        n_probe:    int,            # len(W); no operand is W-only or C-only
        is_uniform: bool,           # inner-scan backend (a real lax.scan on the uniform path)
):                                  # -> the map body x -> (dG,)
    '''The per-core map body of the factored TT tangent assembly (W summed), the derivative of
    :py:func:`_tt_variation_jets_step`.

    The GN assembly is three trilinear outer products ``f(a, b, c)`` -- ``mu (x) xi (x) sigma~``,
    ``tau~ (x) xi (x) nu`` and ``mu (x) deta~ (x) nu`` -- so its derivative is, per group, ``d(a.b) (x) c +
    (a.b) (x) c_dot``, in the GN assembly's forms:

    - **sigma~ / tau~ groups** (``xi`` affine): the banded pairs ``sum_r mu^(r) (x) [U^k sigma~]^(r) +
      sum_s [U^k tau~]^(s) (x) nu^(s)`` and their dotted twins (four terms: K on ``mu_dot`` / ``sigma~_dot`` /
      ``tau~_dot`` / ``nu_dot``), met by ``xi`` and ``xi_dot`` -- two W-summed finals;
    - **deta~ group** (both full): an order scan over the lag, :py:func:`_tt_tangent_det_lag_step`.

    Memoized on its statics -- hashable scalars only (``docs/contributor/scan_body_principles.md``).'''
    nW = n_probe
    lag_step = _tt_tangent_det_lag_step(n_probe)

    def _step(x):
        xi, xid, mu, mud, nu, nud, st, std, tt, ttd, dt, dtd = x
        xnp, _, xscan = get_backend(is_uniform, tree_contains_jax(x))
        order = mu.shape[0] - 1
        pst, pstd, ptt, pttd = (_shift_up_pair(j, xnp) for j in (st, std, tt, ttd))
        pair = (contractions.contract('krWCb,rWCa->kWCab', pst, mu)
                + contractions.contract('ksWCa,sWCb->kWCab', ptt, nu))
        pair_d = (contractions.contract('krWCb,rWKCa->kWKCab', pst, mud, len_W=nW)          # K on mu_dot
                  + contractions.contract('krWKCb,rWCa->kWKCab', pstd, mu, len_W=nW)        # K on sigma~_dot
                  + contractions.contract('ksWKCa,sWCb->kWKCab', pttd, nu, len_W=nW)        # K on tau~_dot
                  + contractions.contract('ksWCa,sWKCb->kWKCab', ptt, nud, len_W=nW))       # K on nu_dot
        banded = (contractions.contract('kWKCab,kWCi->KCaib', pair_d, xi, len_W=nW)
                  + contractions.contract('kWCab,kWKCi->KCaib', pair, xid, len_W=nW))
        (det, _, _, _, _), _ = xscan(lag_step, (xnp.zeros(banded.shape, xnp.result_type(banded, mu, mud, dt, dtd, nu, nud)),
                                                dt, dtd, nu, nud),
                                     (mu, mud, _inverse_factorials(order, mu, xnp)))
        return (banded + det,)
    return _step


def _assemble_tt_tangent_variation_jets_dense(sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot,
                                              deta_tildes, deta_tildes_dot, xi_jets, xi_jets_dot, mu_jets,
                                              mu_jets_dot, nu_jets, nu_jets_dot, n_probe):
    '''The unchunked factored TT assembly of :py:func:`assemble_tt_tangent_variation_jets`, W summed: the
    per-chunk kernel of the chunked path (``n_probe = 1`` there), mapped over cores on both layouts (the
    per-core body :py:func:`_tt_tangent_variation_jets_step`).'''
    ops = (xi_jets, xi_jets_dot, mu_jets, mu_jets_dot, nu_jets, nu_jets_dot, sigma_tildes, sigma_tildes_dot,
           tau_tildes, tau_tildes_dot, deta_tildes, deta_tildes_dot)
    use_jax = tree_contains_jax(ops)
    is_uniform = is_ndarray(xi_jets)
    _, xmap, _ = get_backend(is_uniform, use_jax)
    (dG,) = xmap(_tt_tangent_variation_jets_step(n_probe, is_uniform), ops)
    return dG


def _tt_tangent_chunk_summed(co, extra):    # extra unused; the common chunk-fn signature
    '''One W-chunk of the factored TT tangent assembly at the chunked path's pinned ``n_probe = 1``.'''
    return _assemble_tt_tangent_variation_jets_dense(*co, 1)


def assemble_tt_tangent_variation_jets(
        sigma_tildes:       typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        sigma_tildes_dot:   typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        tau_tildes:         typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        tau_tildes_dot:     typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
        deta_tildes:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(nUi,)
        deta_tildes_dot:    typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(nUi,)
        xi_jets:            typ.Sequence[NDArray],  # len=d, elm_shape=(2,)+W+C+(nUi,)
        xi_jets_dot:        typ.Sequence[NDArray],  # len=d, elm_shape=(2,)+W+K+C+(nUi,)
        mu_jets:            typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rLi,)
        mu_jets_dot:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rLi,)
        nu_jets:            typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+C+(rR(i+1),)
        nu_jets_dot:        typ.Sequence[NDArray],  # len=d, elm_shape=(order+1,)+W+K+C+(rR(i+1),)
        n_probe:            int,                    # len(W)
        chunk_size:         typ.Optional[int] = DEFAULT_CHUNK_SIZE,   # W-chunk size; None (or >= W) -> dense
) -> typ.Tuple[NDArray, ...]:                       # dG. len=d, elm_shape=K+C+(rLi,nUi,rR(i+1)) (W summed)
    '''The derivative of the TT variation-gradient assembly (W summed), factored by the product rule.

    The GN assembly (:py:func:`assemble_tt_variation_jets`) sums three trilinear outer products; differentiating
    each factor gives nine terms. Here each group's undotted pair and its dotted twin are formed once and met by
    the remaining factor: the sigma~ / tau~ groups as banded pairs (both meet ``xi``, so they share their two
    finals), the deta~ group as an order scan over the lag; see :py:func:`_tt_tangent_variation_jets_step`.
    Equal to the nine-term dense form to ~1e-15. W-chunked exactly as the GN assembly (on the uniform path --
    a ``lax.scan`` on jax, an eager loop on numpy; ragged, ``None`` and ``W <= chunk_size`` run unchunked); see
    :doc:`/chunking`.'''
    ops = (sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot, deta_tildes, deta_tildes_dot,
           xi_jets, xi_jets_dot, mu_jets, mu_jets_dot, nu_jets, nu_jets_dot)
    w_axes = (2,) * len(ops)
    dense = lambda: _assemble_tt_tangent_variation_jets_dense(*ops, n_probe)
    if chunk_size is None or not is_ndarray(xi_jets):
        return dense()
    use_jax = tree_contains_jax(ops)
    W = math.prod(xi_jets.shape[2:2 + n_probe])       # uniform supercore: (d, order, W..., ...)
    if W <= chunk_size:
        return dense()
    ops, w_axes = _flatten_w(ops, w_axes, n_probe)     # -> one probe super-axis; n_probe == 1 holds
    return _wchunked_sum(_tt_tangent_chunk_summed, None, ops, w_axes, W, chunk_size, use_jax)


def t3_probe_corewise_derivatives_hessian_from_sweep(
        ztildes:            typ.Sequence[NDArray],  # residual jets (omega^2 applied), len=d, elm_shape=(order+1,)+W+C+(Ni,)
        ww:                 typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:                 typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        core_pair:          typ.Tuple[
            typ.Sequence[NDArray],                  # tucker_cores of theta. len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],                  # tt_cores of theta.     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],                                          # = TuckerTensorTrain.data
        core_directions:    typ.Tuple[
            typ.Sequence[NDArray],                  # Tucker directions theta_hat. len=d, elm_shape=K+C+(ni,Ni)
            typ.Sequence[NDArray],                  # TT directions theta_hat.     len=d, elm_shape=K+C+(ri,ni,r(i+1))
        ],
        order:              int,                    # highest derivative order
        frame_sweep:        typ.Tuple[typ.Sequence[NDArray], ...],  # (xi, mu[, nu, eta]) jets; only xi, mu are read
        chunk_size:         typ.Optional[int] = DEFAULT_CHUNK_SIZE,  # W-chunk size of the assemblies; None -> dense
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                        # Tucker block of K theta_hat. len=d, elm_shape=K+C+(ni,Ni)
    typ.Tuple[NDArray, ...],                        # TT block of K theta_hat.     len=d, elm_shape=K+C+(ri,ni,r(i+1))
]:
    ''':py:func:`t3_probe_corewise_derivatives_hessian` reusing a precomputed ``frame_sweep``'s ``xi`` and
    ``mu`` jets -- the reuse hook for a local model, whose probe sweep at the manifold frame ``(U, O, L, R)``
    carries exactly the corewise frame's ``xi`` (through ``U``) and ``mu`` (through ``L``) when
    ``core_pair = (U, L)``, the geometry's base point.

    ``frame_sweep`` is :py:func:`tv_precompute_probe_frame_sweep_jets` at that frame (its ``nu`` and ``eta``
    belong to the manifold frame and are not read: ``nu`` is re-swept here through ``G``), or the lean
    ``(xi, mu)`` pair of :py:func:`tv_precompute_apply_frame_sweep_jets` at the corewise frame. Either way its
    ``xi``/``mu`` must be ``core_pair``'s own; nothing here checks that.'''
    tucker_cores, tt_cores = core_pair
    tucker_dirs, tt_dirs = core_directions
    xi_jets, mu_jets = frame_sweep[0], frame_sweep[1]
    n_probe = ww[0].ndim - 1   # number of sample-stack (W) axes (ww carries W, no C)

    # 1. the undotted state at the corewise frame (U, G, G, G): nu re-swept through G; the adjoint tildes.
    #    No corewise eta and no corewise dxi~: the Hessian needs only their derivatives.
    nu_jets = compute_nu_jets(tt_cores, xi_jets, order)
    deta_tildes = compute_deta_tilde_jets(tucker_cores, ztildes)
    tau_tildes = compute_tau_tilde_jets(tt_cores, xi_jets, deta_tildes, mu_jets)
    sigma_tildes = compute_sigma_tilde_jets(tt_cores, xi_jets, deta_tildes, nu_jets)

    # 2. the dotted forward: the GN tangent forward at Q = O = P = G along (Udot, Gdot) -- its sigma / tau /
    #    deta jets ARE the derivatives of mu / nu / eta along theta_hat.
    xi_jets_dot = build_input_jets(compute_xi(tucker_dirs, ww), compute_xi(tucker_dirs, pp))
    mu_jets_dot = compute_sigma_jets(tt_dirs, tt_cores, tt_cores, xi_jets, xi_jets_dot, mu_jets)
    nu_jets_dot = compute_tau_jets(tt_dirs, tt_cores, tt_cores, xi_jets, xi_jets_dot, nu_jets)
    eta_jets_dot = compute_deta_jets(tt_dirs, tt_cores, tt_cores, mu_jets, nu_jets, mu_jets_dot, nu_jets_dot)

    # 3. the dotted adjoint-up lift; 4. the dotted adjoint sweeps; 5. the dotted dxi~
    deta_tildes_dot = compute_deta_tilde_tangent_jets(tucker_cores, tucker_dirs, ztildes)
    tau_tildes_dot = compute_tau_tilde_tangent_jets(tt_cores, tt_dirs, xi_jets, xi_jets_dot, deta_tildes,
                                                    deta_tildes_dot, mu_jets, mu_jets_dot, tau_tildes)
    sigma_tildes_dot = compute_sigma_tilde_tangent_jets(tt_cores, tt_dirs, xi_jets, xi_jets_dot, deta_tildes,
                                                        deta_tildes_dot, nu_jets, nu_jets_dot, sigma_tildes)
    dxi_tildes_dot = compute_dxi_tilde_tangent_jets(tt_cores, tt_dirs, mu_jets, mu_jets_dot, nu_jets, nu_jets_dot,
                                                    sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot)

    # 6. the assemblies (W summed, chunked)
    dU = assemble_tucker_tangent_variation_jets(ztildes, dxi_tildes_dot, ww, pp, eta_jets_dot, n_probe,
                                                chunk_size=chunk_size)
    dG = assemble_tt_tangent_variation_jets(sigma_tildes, sigma_tildes_dot, tau_tildes, tau_tildes_dot,
                                            deta_tildes, deta_tildes_dot, xi_jets, xi_jets_dot, mu_jets,
                                            mu_jets_dot, nu_jets, nu_jets_dot, n_probe, chunk_size=chunk_size)
    return dU, dG


def t3_probe_corewise_derivatives_hessian(
        ztildes:            typ.Sequence[NDArray],  # residual jets (omega^2 applied), len=d, elm_shape=(order+1,)+W+C+(Ni,)
        ww:                 typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
        pp:                 typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
        core_pair:          typ.Tuple[
            typ.Sequence[NDArray],                  # tucker_cores of theta. len=d, elm_shape=C+(ni,Ni)
            typ.Sequence[NDArray],                  # tt_cores of theta.     len=d, elm_shape=C+(ri,ni,r(i+1))
        ],                                          # = TuckerTensorTrain.data
        core_directions:    typ.Tuple[
            typ.Sequence[NDArray],                  # Tucker directions theta_hat. len=d, elm_shape=K+C+(ni,Ni)
            typ.Sequence[NDArray],                  # TT directions theta_hat.     len=d, elm_shape=K+C+(ri,ni,r(i+1))
        ],
        order:              int,                    # highest derivative order
        chunk_size:         typ.Optional[int] = DEFAULT_CHUNK_SIZE,  # W-chunk size of the assemblies; None -> dense
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],                        # Tucker block of K theta_hat. len=d, elm_shape=K+C+(ni,Ni)
    typ.Tuple[NDArray, ...],                        # TT block of K theta_hat.     len=d, elm_shape=K+C+(ri,ni,r(i+1))
]:
    '''The derivative of :py:func:`t3_probe_corewise_derivatives_transpose` along core directions: the
    core-coordinate Hessian of ``theta -> <z~, S(Phi(theta))>`` applied to ``theta_hat``,

        ``K_z theta_hat = d/ds grad_theta <z~, S(Phi(theta + s theta_hat))> |_{s=0}``,

    for ``S`` the probe-derivative jets (orders ``0..order``) and fixed residual jets ``z~``. This is the
    sampling half of the exact Riemannian Hessian's curvature term (with ``z~ = omega^2 r``, the weighted
    residual the Gauss-Newton transpose receives), by a hand-written "two-hole" sweep -- two cores replaced
    at once -- rather than automatic differentiation; it is exact (no finite differences) and symmetric.

    Stacks: the residual jets and ``core_pair`` carry ``W + C`` / ``C``; the directions carry a tangent
    stack ``K`` (``K + C``), and the output is raw core variations shaped like the directions, the sample
    stack ``W`` summed. Polymorphic over ragged tuples and uniform supercores (packed ``ztildes``/``ww``/
    ``pp``). ``chunk_size`` W-chunks the assemblies on the uniform path (bounding their peak memory on jax)
    exactly as in :py:func:`tv_probe_derivatives_transpose` (:doc:`/chunking`).

    **Mask-free by contract.** It assumes ``theta`` and ``theta_hat`` arrive with their padded slots already
    zero: a uniform caller masks both once up front with ``theta``'s masks, so the shared algorithm runs
    mask-free. A second derivative multiplies a direction's padded slot into real slots, so an unmasked
    ``theta_hat`` corrupts the real block; masking is the uniform wrapper's job.

    The pieces: :py:func:`compute_deta_tilde_tangent_jets`, :py:func:`compute_tau_tilde_tangent_jets`,
    :py:func:`compute_sigma_tilde_tangent_jets`, :py:func:`compute_dxi_tilde_tangent_jets`,
    :py:func:`assemble_tucker_tangent_variation_jets`, :py:func:`assemble_tt_tangent_variation_jets`; the
    precomputed-sweep variant is :py:func:`t3_probe_corewise_derivatives_hessian_from_sweep`.

    Examples
    --------
    The two-hole is the directional derivative of the corewise transpose (a polynomial in ``s``, so a central
    difference is accurate to ``O(h^2)``), and a symmetric operator:

    >>> import numpy as np
    >>> import t3toolbox.backend.t3_constructors as t3k
    >>> import t3toolbox.backend.sampling_derivatives as sd
    >>> np.random.seed(0)                                                  # t3_corewise_randn draws from it
    >>> rng = np.random.default_rng(0)
    >>> shape, order, W = (4, 5, 3), 2, 6
    >>> theta = t3k.t3_corewise_randn(shape, (2, 3, 2), (1, 2, 2, 1))
    >>> ww = [rng.standard_normal((W, N)) for N in shape]
    >>> pp = [rng.standard_normal((W, N)) for N in shape]
    >>> zt = [rng.standard_normal((order + 1, W, N)) for N in shape]       # residual jets, (order+1)+W+(Ni,)
    >>> dirs = (tuple(rng.standard_normal((2,) + u.shape) for u in theta[0]),   # a tangent stack K = (2,)
    ...         tuple(rng.standard_normal((2,) + g.shape) for g in theta[1]))
    >>> KU, KG = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, theta, dirs, order)
    >>> [k.shape for k in KG]                                              # K + the TT core shapes
    [(2, 1, 2, 2), (2, 2, 3, 2), (2, 2, 2, 1)]
    >>> one = lambda cores: sd.t3_probe_corewise_derivatives_transpose(zt, ww, pp, cores, order,
    ...                                                                sum_over_probes=True)
    >>> shift = lambda h: tuple(tuple(c + h * dc[0] for c, dc in zip(cs, dcs)) for cs, dcs in zip(theta, dirs))
    >>> fd = [(a - b) / 2e-4 for a, b in zip(sum(one(shift(1e-4)), ()), sum(one(shift(-1e-4)), ()))]
    >>> got = [k[0] for k in KU + KG]
    >>> bool(max(np.linalg.norm(g - f) / np.linalg.norm(f) for g, f in zip(got, fd)) < 1e-6)
    True
    >>> gram = sum(a.reshape(2, -1) @ b.reshape(2, -1).T for a, b in zip(dirs[0] + dirs[1], KU + KG))
    >>> bool(abs(gram[0, 1] - gram[1, 0]) < 1e-12 * abs(gram).max())     # <a, K b> = <b, K a>
    True
    '''
    frame_sweep = tv_precompute_apply_frame_sweep_jets(fv_conversions.t3_corewise_frame(core_pair), ww, pp, order)
    return t3_probe_corewise_derivatives_hessian_from_sweep(
        ztildes, ww, pp, core_pair, core_directions, order, frame_sweep, chunk_size=chunk_size)


#####################################################
########    Dense reference (test oracle)    ########
#####################################################

def dense_probe_derivatives(
        ww:     typ.Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=(Ni,)
        pp:     typ.Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=(Ni,)
        T:      NDArray,                # dense tensor, shape=(N0,...,N(d-1))
        order:  int,                    # highest derivative order
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
