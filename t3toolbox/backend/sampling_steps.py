# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The step functions of the sampling engine, and the start values of its sweeps. Private.

A **step function** is one step of the recurrence of a sampling variable (the recurrences are in
:py:mod:`t3toolbox.backend.sampling_definitions`), written for ONE site's arrays. It is named by the letter and the
grade of what it makes, and serves both bases: the basis of its result follows from the cores it is given
(``nu_step`` given the ``R`` cores makes ``nu``, given the ``L`` cores ``nu_L``). A step knows nothing of the layout,
the graph or the other sites: no loop over the sites, no mask, no slicing of a bond. The engine
(:py:mod:`t3toolbox.backend.sampling_variables`) runs the steps over the
sites, through ``xscan`` for a sweep and ``xmap`` for a map, and decides which arrays each step is given.

**Every term is one** ``contractions.jet_contract('<literal>', ...)`` **call**, the string written out at the call
site, passed the whole record of axis counts ``lens``, the jet shape of its output, and the form over the orders.
The strings are the final ones of the library (the ``C`` group written, and ``K`` on the operands that carry it),
so the stacks change the record, not the strings. The call goes through the module attribute
``contractions.jet_contract``, so that a test can record every call.

**The jet shape of each output** is the declared one (the table of
:py:mod:`t3toolbox.backend.sampling_variables`): a full jet ``F`` (``order``) for every variable but the leaves, whose
jet shape is the input's, clipped to ``F`` axis by axis. A leaf step reads the input's jet sizes off the operand's
static shape, at each call, so nothing per site is closed over.

A step takes its arrays positionally, the carry first for a sweep, and the statics by keyword:

- ``lens``: the ``GroupLengths`` of the node (``J``, ``W``, ``K``, ``C``);
- ``order``: the jet shape of the call, a tuple; the box is ``F = tuple(n + 1 for n in order)``;
- ``form``: ``'loop'`` or ``'scan'``, how ``jet_contract`` takes a full partner one coefficient at a time.

The start values: ``start_ones``, the vector of ones that closes an end bond, as a constant polynomial on the full
box; ``start_zeros``, the start of a sweep whose variable has nothing at its end. Each row of the graph declares which.

**The tilde steps** carry ``K`` on their tilde operands in their strings (``'JWKCa,Caib,JWCi->JWKCb'``, ...): with a
record whose ``K`` is zero (the residual's adjoint) they are the plain strings, and with the call's ``K`` they serve
the adjoint seeded by a stack of directions. **The readouts** (the blocks ``V_tilde``, ``H_tilde``) are
``jet_contract`` calls with no ``J`` in the output: the coefficient at the corner of the product, the corner being the
jet shape passed.

The **whole steps** (``r_step``, ``f_step``, ``r_block_sumsq_step``) act on a whole variable, one array per mode
for the probe kind, and take the same statics.

Every name here is private: the module is the engine's, and its ``__all__`` is empty.
"""
import typing as typ

import numpy as np

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.jets as jets
from t3toolbox.backend.common import *

__all__ = []


def _leaf_order(
        w_JWo:  NDArray,                    # an input jet: m leading jet axes of sizes of its own
        order:  typ.Tuple[int, ...],        # the jet shape of the call, len=m
) -> typ.Tuple[int, ...]:                   # the leaf's jet shape: the input's jet sizes clipped to F, minus one
    return tuple(min(a, n + 1) - 1 for a, n in zip(np.shape(w_JWo)[:len(order)], order))


###############################################
########    The plain grade (5.1)    ##########
###############################################

def xi_step(
        B_Cio:  NDArray,                    # a Tucker core, C+(n, N): D[i] for xi
        w_JWo:  NDArray,                    # the input jet of the site, J_in+W+(N,)
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],        # the jet shape of the call; the leaf's is the input's, clipped to F
        form:   str,                        # 'loop' | 'scan'
) -> NDArray:                               # xi[i]: the leaf. min(J_in, F)+W+C+(n,)
    '''``xi[i] = xi_term(D[i], w[i])``: the Tucker core applied to the input jet. A ride: exact, of the input's
    degree, with the input's slots beyond the box ``F`` dropped.'''
    return contractions.jet_contract('JWo,Cio->JWCi', w_JWo, B_Cio,
                                     lens=lens, order=_leaf_order(w_JWo, order), over_orders=form)


def mu_step(
        mu_JWCa:  NDArray,                  # the carry: the left of bond i, F+W+C+(ra,)
        G_Caib:   NDArray,                  # a TT core, C+(ra, n, rb): L[i] for mu
        xi_JWCi:  NDArray,                  # the leaf of site i, min(J_in, F)+W+C+(n,)
        *,
        lens:     contractions.GroupLengths,
        order:    typ.Tuple[int, ...],
        form:     str,
) -> NDArray:                               # the left of bond i+1, F+W+C+(rb,)
    '''``mu[i+1] = mu_term(mu[i], L[i], xi[i])``.'''
    return contractions.jet_contract('JWCa,Caib,JWCi->JWCb', mu_JWCa, G_Caib, xi_JWCi,
                                     lens=lens, order=order, over_orders=form)


def nu_step(
        nu_JWCb:  NDArray,                  # the carry: the right of bond i+1, F+W+C+(rb,)
        G_Caib:   NDArray,                  # a TT core, C+(ra, n, rb): R[i] for nu, L[i] for nu_L
        xi_JWCi:  NDArray,                  # the leaf of site i, min(J_in, F)+W+C+(n,)
        *,
        lens:     contractions.GroupLengths,
        order:    typ.Tuple[int, ...],
        form:     str,
) -> NDArray:                               # the right of bond i, F+W+C+(ra,)
    '''``nu[i] = nu_term(R[i], xi[i], nu[i+1])``; with the ``L`` cores, ``nu_L``.'''
    return contractions.jet_contract('Caib,JWCi,JWCb->JWCa', G_Caib, xi_JWCi, nu_JWCb,
                                     lens=lens, order=order, over_orders=form)


def eta_step(
        mu_JWCa:  NDArray,                  # the left of bond i, F+W+C+(ra,)
        G_Caib:   NDArray,                  # a TT core, C+(ra, n, rb): U[i] for eta, L[i] for eta_D
        nu_JWCb:  NDArray,                  # the right of bond i+1, F+W+C+(rb,)
        *,
        lens:     contractions.GroupLengths,
        order:    typ.Tuple[int, ...],
        form:     str,
) -> NDArray:                               # the trunk above Tucker edge i, F+W+C+(n,)
    '''``eta[i] = eta_term(mu[i], U[i], nu[i+1])``; with ``L[i]`` and ``nu_L``, ``eta_D``.'''
    return contractions.jet_contract('JWCa,Caib,JWCb->JWCi', mu_JWCa, G_Caib, nu_JWCb,
                                     lens=lens, order=order, over_orders=form)


def z_step(
        eta_JWCi:  NDArray,                 # a trunk, F+W+C+(n,): eta_D[i], or eta[i] with the Tucker center
        B_Cio:     NDArray,                 # a Tucker core, C+(n, N): D[i], or Vc[i] with eta[i]
        *,
        lens:      contractions.GroupLengths,
        order:     typ.Tuple[int, ...],
        form:      str,
) -> NDArray:                               # z[i]: F+W+C+(N,)
    '''``z[i] = z_term(eta_D[i], D[i])``, or ``z_term(eta[i], Vc[i])`` with the Tucker centers.'''
    return contractions.jet_contract('JWCi,Cio->JWCo', eta_JWCi, B_Cio,
                                     lens=lens, order=order, over_orders=form)


###############################################
########    The tilde grade (5.3)    ##########
###############################################

def xi_tilde_step(
        B_Cio:          NDArray,            # a Tucker core, C+(n, N): D[i]
        w_tilde_JWCo:   NDArray,            # the seed of mode i, F+W+[K]+C+(N,), stored turned
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # xi_tilde[i]: F+W+[K]+C+(n,)
    '''``xi_tilde[i] = xi_term(D[i], w_tilde[i])``: the Tucker core applied to the seed, a full dual jet.'''
    return contractions.jet_contract('JWKCo,Cio->JWKCi', w_tilde_JWCo, B_Cio,
                                     lens=lens, order=order, over_orders=form)


def mu_tilde_step(
        mu_tilde_JWCa:  NDArray,            # the carry: mu_tilde at bond i, F+W+[K]+C+(ra,)
        G_Caib:         NDArray,            # L[i], C+(ra, n, rb)
        mu_JWCa:        NDArray,            # mu at bond i, F+W+C+(ra,)
        xi_JWCi:        NDArray,            # the leaf of site i, min(J_in, F)+W+C+(n,)
        xi_tilde_JWCi:  NDArray,            # xi_tilde[i], F+W+[K]+C+(n,)
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # mu_tilde at bond i+1, F+W+[K]+C+(rb,)
    '''``mu_tilde[i+1] = mu_term(mu_tilde[i], L[i], xi[i]) + mu_term(mu[i], L[i], xi_tilde[i])``.'''
    return (contractions.jet_contract('JWKCa,Caib,JWCi->JWKCb', mu_tilde_JWCa, G_Caib, xi_JWCi,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('JWCa,Caib,JWKCi->JWKCb', mu_JWCa, G_Caib, xi_tilde_JWCi,
                                        lens=lens, order=order, over_orders=form))


def nu_tilde_step(
        nu_tilde_JWCb:  NDArray,            # the carry: nu_tilde at bond i+1, F+W+[K]+C+(rb,)
        G_Caib:         NDArray,            # R[i], C+(ra, n, rb)
        nu_JWCb:        NDArray,            # nu at bond i+1, F+W+C+(rb,)
        xi_JWCi:        NDArray,            # the leaf of site i
        xi_tilde_JWCi:  NDArray,            # xi_tilde[i], F+W+[K]+C+(n,)
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # nu_tilde at bond i, F+W+[K]+C+(ra,)
    '''``nu_tilde[i] = nu_term(R[i], xi[i], nu_tilde[i+1]) + nu_term(R[i], xi_tilde[i], nu[i+1])``.'''
    return (contractions.jet_contract('Caib,JWCi,JWKCb->JWKCa', G_Caib, xi_JWCi, nu_tilde_JWCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('Caib,JWKCi,JWCb->JWKCa', G_Caib, xi_tilde_JWCi, nu_JWCb,
                                        lens=lens, order=order, over_orders=form))


def eta_tilde_step(
        mu_tilde_JWCa:  NDArray,            # mu_tilde at bond i, F+W+[K]+C+(ra,)
        mu_JWCa:        NDArray,            # mu at bond i
        G_Caib:         NDArray,            # U[i], C+(ra, n, rb)
        nu_tilde_JWCb:  NDArray,            # nu_tilde at bond i+1, F+W+[K]+C+(rb,)
        nu_JWCb:        NDArray,            # nu at bond i+1
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # eta_tilde[i], F+W+[K]+C+(n,)
    '''``eta_tilde[i] = eta_term(mu_tilde[i], U[i], nu[i+1]) + eta_term(mu[i], U[i], nu_tilde[i+1])``, every slot.'''
    return (contractions.jet_contract('JWKCa,Caib,JWCb->JWKCi', mu_tilde_JWCa, G_Caib, nu_JWCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('JWCa,Caib,JWKCb->JWKCi', mu_JWCa, G_Caib, nu_tilde_JWCb,
                                        lens=lens, order=order, over_orders=form))


def eta_tilde_top_step(
        mu_tilde_JWCa:  NDArray,            # as for eta_tilde_step
        mu_JWCa:        NDArray,
        G_Caib:         NDArray,
        nu_tilde_JWCb:  NDArray,
        nu_JWCb:        NDArray,
        w_JWo:          NDArray,            # the input jet of site i: only its jet sizes are read
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # the top slots of eta_tilde[i]: a+W+[K]+C+(n,), a = min(J_in, F)
    '''The short form of ``eta_tilde[i]``: only the slots ``V_tilde`` reads, ``eta_tilde[i][c - a + 1 : c + 1]`` on
    every jet axis (``c`` the corner, ``a`` the input's jet sizes clipped to the box). Each slot is the coefficient of
    the two terms of ``eta_tilde`` at that slot: two readouts at a lowered corner.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((mu_tilde_JWCa, mu_JWCa, G_Caib, nu_tilde_JWCb, nu_JWCb)))
    a = tuple(f + 1 for f in _leaf_order(w_JWo, order))
    slots = []
    for j in np.ndindex(*a):
        corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))
        slots.append(contractions.jet_contract('JWKCa,Caib,JWCb->WKCi', mu_tilde_JWCa, G_Caib, nu_JWCb,
                                               lens=lens, order=corner, over_orders=form)
                     + contractions.jet_contract('JWCa,Caib,JWKCb->WKCi', mu_JWCa, G_Caib, nu_tilde_JWCb,
                                                 lens=lens, order=corner, over_orders=form))
    return xnp.stack(slots).reshape(a + slots[0].shape)


def V_tilde_step(
        top_JWCi:       NDArray,            # the top slots of eta_tilde[i], a+W+[K]+C+(n,)
        eta_JWCi:       NDArray,            # eta[i], F+W+C+(n,)
        w_JWo:          NDArray,            # the input jet of site i, J_in+W+(N,)
        w_tilde_JWCo:   NDArray,            # the seed of mode i, F+W+[K]+C+(N,)
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # V_tilde[i]: [K]+C+(n, N), summed over the samples
    '''``V_tilde[i]``, the corner of ``V_term(eta_tilde[i], w[i]) + V_term(eta[i], w_tilde[i])``. The first term
    from the top slots of ``eta_tilde``: with ``S`` those slots, the corner of ``eta_tilde * w`` is the coefficient
    ``a - 1`` of ``S * w``, so it is a readout at the corner ``a - 1``, the leaf's jet shape.'''
    return (contractions.jet_contract('JWKCi,JWo->KCio', top_JWCi, w_JWo,
                                      lens=lens, order=_leaf_order(w_JWo, order), over_orders=form)
            + contractions.jet_contract('JWCi,JWKCo->KCio', eta_JWCi, w_tilde_JWCo,
                                        lens=lens, order=order, over_orders=form))


def H_tilde_step(
        mu_tilde_JWCa:  NDArray,            # mu_tilde at bond i
        mu_JWCa:        NDArray,            # mu at bond i
        xi_tilde_JWCi:  NDArray,            # xi_tilde[i]
        xi_JWCi:        NDArray,            # xi[i], the leaf
        nu_tilde_JWCb:  NDArray,            # nu_tilde at bond i+1
        nu_JWCb:        NDArray,            # nu at bond i+1
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # H_tilde[i]: [K]+C+(ra, n, rb), summed over the samples
    '''``H_tilde[i]``, the corner of ``H_term(mu_tilde, xi, nu) + H_term(mu, xi_tilde, nu) + H_term(mu, xi,
    nu_tilde)`` at site ``i``: three readouts, the tilde on each of the three in turn.'''
    return (contractions.jet_contract('JWKCa,JWCi,JWCb->KCaib', mu_tilde_JWCa, xi_JWCi, nu_JWCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('JWCa,JWKCi,JWCb->KCaib', mu_JWCa, xi_tilde_JWCi, nu_JWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('JWCa,JWCi,JWKCb->KCaib', mu_JWCa, xi_JWCi, nu_tilde_JWCb,
                                        lens=lens, order=order, over_orders=form))


###############################################
########    The whole steps    ################
###############################################

def w_tilde_step(
        r:       typ.Sequence[NDArray],     # len=d, elm_shape=F+W+C+(Ni,)
        weight:  typ.Optional[NDArray],     # (d,)+F; None = ones
        *,
        lens:    contractions.GroupLengths,
        order:   typ.Tuple[int, ...],
        form:    str,
) -> typ.Tuple[NDArray, ...]:               # the seed. len=d, elm_shape=F+W+C+(Ni,), stored turned
    '''The seed of the probe kind's adjoint: the residual made a dual jet, ``jet_to_dual_jet``: the weight squared,
    then every jet axis flipped.'''
    return jets.jet_to_dual_jet('probe', order, r, weight)


def g_step(
        V_tilde:  typ.Sequence[NDArray],    # len=d, elm_shape=C+(nUi, Ni)
        H_tilde:  typ.Sequence[NDArray],    # len=d, elm_shape=C+(rLi, nDi, rR(i+1))
        *,
        lens:     contractions.GroupLengths,
        order:    typ.Tuple[int, ...],
        form:     str,
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:   # g = (V_tilde, H_tilde)
    '''The gradient: the two families of blocks, paired.'''
    return tuple(V_tilde), tuple(H_tilde)


def r_step(
        z:      typ.Sequence[NDArray],      # len=d, elm_shape=F+W+C+(Ni,)
        y:      typ.Sequence[NDArray],      # len=d, elm_shape=F+W+(Ni,): the data, broadcast over C
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],
        form:   str,
) -> typ.Tuple[NDArray, ...]:               # r = z - y. len=d, elm_shape=F+W+C+(Ni,)
    '''The residual of the probe kind, the library's sign: ``r = z - y``, mode by mode.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((z, y)))
    out = []
    for z_FWCo, y_FWo in zip(z, y):
        y_FWo = xnp.asarray(y_FWo)
        y_FW1o = y_FWo.reshape(y_FWo.shape[:-1] + (1,) * lens.C + y_FWo.shape[-1:])
        out.append(z_FWCo - y_FW1o)
    return tuple(out)


def f_step(
        r:       typ.Sequence[NDArray],     # len=d, elm_shape=F+W+C+(Ni,)
        weight:  typ.Optional[NDArray],     # (d,)+F; None = ones
        *,
        lens:    contractions.GroupLengths,
        order:   typ.Tuple[int, ...],
        form:    str,
) -> NDArray:                               # f = 1/2 <r, weight**2 r>, keeping C. shape=C
    '''The misfit of the probe kind: half the sum, over the modes, the slots, the samples and the mode edge, of the
    squared weighted residual. The stack ``C`` is kept.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((r, weight)))
    m, n_W = lens.J, lens.W
    total = None
    for i, r_FWCo in enumerate(r):
        r_FWCo = xnp.asarray(r_FWCo)
        squares = r_FWCo * r_FWCo
        if weight is not None:
            weight_F = xnp.asarray(weight)[i]
            squares = (weight_F * weight_F).reshape(weight_F.shape + (1,) * (r_FWCo.ndim - m)) * squares
        axes = tuple(range(m + n_W)) + (r_FWCo.ndim - 1,)
        summed = xnp.sum(squares, axis=axes)
        total = summed if total is None else total + summed
    half = xnp.asarray(0.5, dtype=xnp.asarray(total).dtype)    # numpy 1.x: a Python float lifts a float32 sum
    return xnp.asarray(half * total)                            # an array, not a numpy scalar


def r_block_sumsq_step(
        r:      typ.Sequence[NDArray],      # len=d, elm_shape=F+W+C+(Ni,)
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],
        form:   str,
) -> NDArray:                               # the unweighted sum of squares per mode and slot. shape=(d,)+F
    '''The sums of squares of the residual of the probe kind, unweighted, per mode and slot: the shape of the
    weight, ``(d,) + F``. Summed over the samples, the stack ``C`` and the mode edge.'''
    xnp, _, _ = get_backend(False, tree_contains_jax(r))
    m = lens.J
    out = []
    for r_FWCo in r:
        r_FWCo = xnp.asarray(r_FWCo)
        out.append(xnp.sum(r_FWCo * r_FWCo, axis=tuple(range(m, r_FWCo.ndim))))
    return xnp.stack(out)


###############################################
########    The start values    ###############
###############################################

def start_ones(
        box:    typ.Tuple[int, ...],        # F, the full jet sizes
        W:      typ.Tuple[int, ...],        # the sample stack
        C:      typ.Tuple[int, ...],        # the stack of base points
        size:   int,                        # the end bond, in the basis of the sweep's own store
        dtype:  typ.Any,
        xnp,
) -> NDArray:                               # F+W+C+(size,): ones in slot (0, ..., 0), zero in every other slot
    '''The start of a sweep of the plain grade: the vector of ones that closes an end bond, as a constant
    polynomial on the full box (a scan carry has one shape at every site).'''
    corner = np.zeros(box)                                         # host: 1 in slot (0, ..., 0)
    corner[(0,) * len(box)] = 1.0
    slots_F = xnp.asarray(corner, dtype=dtype).reshape(box + (1,) * (len(W) + len(C) + 1))
    return slots_F * xnp.ones(W + C + (size,), dtype=dtype)


def start_zeros(
        box:    typ.Tuple[int, ...],        # F
        W:      typ.Tuple[int, ...],
        C:      typ.Tuple[int, ...],
        size:   int,
        dtype:  typ.Any,
        xnp,
) -> NDArray:                               # F+W+C+(size,), all zero
    '''The start of a sweep whose variable is zero at its end: a tilde sweep of the probe kind.'''
    return xnp.zeros(box + W + C + (size,), dtype=dtype)
