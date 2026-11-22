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

- ``lens``: the ``GroupLengths`` of the node (``P``, ``W``, ``K``, ``C``);
- ``order``: the jet shape of the call, a tuple; the box is ``F = tuple(n + 1 for n in order)``;
- ``form``: ``'loop'`` or ``'scan'``, how ``jet_contract`` takes a full partner one coefficient at a time.

The start values: ``start_ones``, the vector of ones that closes an end bond, as a constant polynomial on the full
box; ``start_zeros``, the start of a sweep whose variable has nothing at its end; ``start_node``, a node of the graph
times the ones that close the end bond (the apply kind's seed, on bond ``d``). Each row of the graph declares which.

**The tilde steps** carry ``K`` on their tilde operands in their strings (``'PWKCa,Caib,PWCi->PWKCb'``, ...): with a
record whose ``K`` is zero (the residual's adjoint) they are the plain strings, and with the call's ``K`` they serve
the adjoint seeded by a stack of directions. **The readouts** (the blocks ``V_tilde``, ``H_tilde``) are
``jet_contract`` calls with no ``P`` in the output: the coefficient at the corner of the product, the corner being the
jet shape passed.

The **whole steps** (``r_step``, ``f_step``, ``close_step``, ...) act on a whole variable, of the type of ``z``
where it is the kind's output (one array per mode with a mode edge for the probe kind, one array for apply), and take
one static more, ``kind``.

**The apply kind** reuses the steps of the plain and dot grades for its adjoint: the seed sits on the ones that close
bond ``d``, so a right side or a trunk is the plain recurrence started from the seed, and its tilde-dot the dot
recurrence with the tilde forms in the plain ones' places. Its own steps are readouts of one term (``eta_top_step``,
``eta_dot_top_step``, ``V_top_step``, ``H_readout_step``) and ``close_step``. ``nu_step`` and ``eta_step`` carry ``K``
on the right side, for the adjoint seeded by ``Jp``.

Every name here is private: the module is the engine's, and its ``__all__`` is empty.
"""
import typing as typ

import numpy as np

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.jets as jets
from t3toolbox.backend.common import *

__all__ = []


def _leaf_order(
        w_PWo:  NDArray,                    # an input jet: m leading jet axes of sizes of its own
        order:  typ.Tuple[int, ...],        # the jet shape of the call, len=m
) -> typ.Tuple[int, ...]:                   # the leaf's jet shape: the input's jet sizes clipped to F, minus one
    return tuple(min(a, n + 1) - 1 for a, n in zip(np.shape(w_PWo)[:len(order)], order))


###############################################
########    The plain grade (5.1)    ##########
###############################################

def xi_step(
        B_Cio:  NDArray,                    # a Tucker core, C+(n, N): D[i] for xi
        w_PWo:  NDArray,                    # the input jet of the site, F_in+W+(N,)
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],        # the jet shape of the call; the leaf's is the input's, clipped to F
        form:   str,                        # 'loop' | 'scan'
) -> NDArray:                               # xi[i]: the leaf. min(F_in, F)+W+C+(n,)
    '''``xi[i] = xi_term(D[i], w[i])``: the Tucker core applied to the input jet. A ride: exact, of the input's
    degree, with the input's slots beyond the box ``F`` dropped.'''
    return contractions.jet_contract('PWo,Cio->PWCi', w_PWo, B_Cio,
                                     lens=lens, order=_leaf_order(w_PWo, order), over_orders=form)


def mu_step(
        mu_PWCa:  NDArray,                  # the carry: the left of bond i, F+W+C+(ra,)
        G_Caib:   NDArray,                  # a TT core, C+(ra, n, rb): L[i] for mu
        xi_PWCi:  NDArray,                  # the leaf of site i, min(F_in, F)+W+C+(n,)
        *,
        lens:     contractions.GroupLengths,
        order:    typ.Tuple[int, ...],
        form:     str,
) -> NDArray:                               # the left of bond i+1, F+W+C+(rb,)
    '''``mu[i+1] = mu_term(mu[i], L[i], xi[i])``.'''
    return contractions.jet_contract('PWCa,Caib,PWCi->PWCb', mu_PWCa, G_Caib, xi_PWCi,
                                     lens=lens, order=order, over_orders=form)


def nu_step(
        nu_PWKCb:  NDArray,                 # the carry: the right of bond i+1, F+W+[K]+C+(rb,)
        G_Caib:    NDArray,                 # a TT core, C+(ra, n, rb): R[i] for nu, L[i] for nu_L
        xi_PWCi:   NDArray,                 # the leaf of site i, min(F_in, F)+W+C+(n,)
        *,
        lens:      contractions.GroupLengths,
        order:     typ.Tuple[int, ...],
        form:      str,
) -> NDArray:                               # the right of bond i, F+W+[K]+C+(ra,)
    '''``nu[i] = nu_term(R[i], xi[i], nu[i+1])``; with the ``L`` cores, ``nu_L``. Started from the seed instead of
    ones it is the apply kind's ``nu_tilde`` (or ``nu_L_tilde``), the seed sitting on the ones that close bond
    ``d``; the carry has ``K`` there in the adjoint seeded by ``Jp``.'''
    return contractions.jet_contract('Caib,PWCi,PWKCb->PWKCa', G_Caib, xi_PWCi, nu_PWKCb,
                                     lens=lens, order=order, over_orders=form)


def eta_step(
        mu_PWCa:   NDArray,                 # the left of bond i, F+W+C+(ra,)
        G_Caib:    NDArray,                 # a TT core, C+(ra, n, rb): U[i] for eta, L[i] for eta_D
        nu_PWKCb:  NDArray,                 # the right of bond i+1, F+W+[K]+C+(rb,)
        *,
        lens:      contractions.GroupLengths,
        order:     typ.Tuple[int, ...],
        form:      str,
) -> NDArray:                               # the trunk above Tucker edge i, F+W+[K]+C+(n,)
    '''``eta[i] = eta_term(mu[i], U[i], nu[i+1])``; with ``L[i]`` and ``nu_L``, ``eta_D``. With the apply kind's
    ``nu_tilde`` (``K`` on it in the adjoint seeded by ``Jp``), the apply kind's ``eta_tilde``.'''
    return contractions.jet_contract('PWCa,Caib,PWKCb->PWKCi', mu_PWCa, G_Caib, nu_PWKCb,
                                     lens=lens, order=order, over_orders=form)


def z_step(
        eta_PWCi:  NDArray,                 # a trunk, F+W+C+(n,): eta_D[i], or eta[i] with the Tucker center
        B_Cio:     NDArray,                 # a Tucker core, C+(n, N): D[i], or Vc[i] with eta[i]
        *,
        lens:      contractions.GroupLengths,
        order:     typ.Tuple[int, ...],
        form:      str,
) -> NDArray:                               # z[i]: F+W+C+(N,)
    '''``z[i] = z_term(eta_D[i], D[i])``, or ``z_term(eta[i], Vc[i])`` with the Tucker centers.'''
    return contractions.jet_contract('PWCi,Cio->PWCo', eta_PWCi, B_Cio,
                                     lens=lens, order=order, over_orders=form)


###############################################
########    The tilde grade (5.3)    ##########
###############################################

def xi_tilde_step(
        B_Cio:          NDArray,            # a Tucker core, C+(n, N): D[i]
        w_tilde_PWCo:   NDArray,            # the seed of mode i, F+W+[K]+C+(N,), stored turned
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # xi_tilde[i]: F+W+[K]+C+(n,)
    '''``xi_tilde[i] = xi_term(D[i], w_tilde[i])``: the Tucker core applied to the seed, a full dual jet.'''
    return contractions.jet_contract('PWKCo,Cio->PWKCi', w_tilde_PWCo, B_Cio,
                                     lens=lens, order=order, over_orders=form)


def mu_tilde_step(
        mu_tilde_PWCa:  NDArray,            # the carry: mu_tilde at bond i, F+W+[K]+C+(ra,)
        G_Caib:         NDArray,            # L[i], C+(ra, n, rb)
        mu_PWCa:        NDArray,            # mu at bond i, F+W+C+(ra,)
        xi_PWCi:        NDArray,            # the leaf of site i, min(F_in, F)+W+C+(n,)
        xi_tilde_PWCi:  NDArray,            # xi_tilde[i], F+W+[K]+C+(n,)
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # mu_tilde at bond i+1, F+W+[K]+C+(rb,)
    '''``mu_tilde[i+1] = mu_term(mu_tilde[i], L[i], xi[i]) + mu_term(mu[i], L[i], xi_tilde[i])``.'''
    return (contractions.jet_contract('PWKCa,Caib,PWCi->PWKCb', mu_tilde_PWCa, G_Caib, xi_PWCi,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,Caib,PWKCi->PWKCb', mu_PWCa, G_Caib, xi_tilde_PWCi,
                                        lens=lens, order=order, over_orders=form))


def nu_tilde_step(
        nu_tilde_PWCb:  NDArray,            # the carry: nu_tilde at bond i+1, F+W+[K]+C+(rb,)
        G_Caib:         NDArray,            # R[i], C+(ra, n, rb)
        nu_PWCb:        NDArray,            # nu at bond i+1, F+W+C+(rb,)
        xi_PWCi:        NDArray,            # the leaf of site i
        xi_tilde_PWCi:  NDArray,            # xi_tilde[i], F+W+[K]+C+(n,)
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # nu_tilde at bond i, F+W+[K]+C+(ra,)
    '''``nu_tilde[i] = nu_term(R[i], xi[i], nu_tilde[i+1]) + nu_term(R[i], xi_tilde[i], nu[i+1])``.'''
    return (contractions.jet_contract('Caib,PWCi,PWKCb->PWKCa', G_Caib, xi_PWCi, nu_tilde_PWCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('Caib,PWKCi,PWCb->PWKCa', G_Caib, xi_tilde_PWCi, nu_PWCb,
                                        lens=lens, order=order, over_orders=form))


def eta_tilde_step(
        mu_tilde_PWCa:  NDArray,            # mu_tilde at bond i, F+W+[K]+C+(ra,)
        mu_PWCa:        NDArray,            # mu at bond i
        G_Caib:         NDArray,            # U[i], C+(ra, n, rb)
        nu_tilde_PWCb:  NDArray,            # nu_tilde at bond i+1, F+W+[K]+C+(rb,)
        nu_PWCb:        NDArray,            # nu at bond i+1
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # eta_tilde[i], F+W+[K]+C+(n,)
    '''``eta_tilde[i] = eta_term(mu_tilde[i], U[i], nu[i+1]) + eta_term(mu[i], U[i], nu_tilde[i+1])``, every slot.'''
    return (contractions.jet_contract('PWKCa,Caib,PWCb->PWKCi', mu_tilde_PWCa, G_Caib, nu_PWCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,Caib,PWKCb->PWKCi', mu_PWCa, G_Caib, nu_tilde_PWCb,
                                        lens=lens, order=order, over_orders=form))


def eta_tilde_top_step(
        mu_tilde_PWCa:  NDArray,            # as for eta_tilde_step
        mu_PWCa:        NDArray,
        G_Caib:         NDArray,
        nu_tilde_PWCb:  NDArray,
        nu_PWCb:        NDArray,
        w_PWo:          NDArray,            # the input jet of site i: only its jet sizes are read
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # the top slots of eta_tilde[i]: a+W+[K]+C+(n,), a = min(F_in, F)
    '''The short form of ``eta_tilde[i]``: only the slots ``V_tilde`` reads, ``eta_tilde[i][c - a + 1 : c + 1]`` on
    every jet axis (``c`` the corner, ``a`` the input's jet sizes clipped to the box). Each slot is the coefficient of
    the two terms of ``eta_tilde`` at that slot: two readouts at a lowered corner.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((mu_tilde_PWCa, mu_PWCa, G_Caib, nu_tilde_PWCb, nu_PWCb)))
    a = tuple(f + 1 for f in _leaf_order(w_PWo, order))
    slots = []
    for j in np.ndindex(*a):
        corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))
        slots.append(contractions.jet_contract('PWKCa,Caib,PWCb->WKCi', mu_tilde_PWCa, G_Caib, nu_PWCb,
                                               lens=lens, order=corner, over_orders=form)
                     + contractions.jet_contract('PWCa,Caib,PWKCb->WKCi', mu_PWCa, G_Caib, nu_tilde_PWCb,
                                                 lens=lens, order=corner, over_orders=form))
    return xnp.stack(slots).reshape(a + slots[0].shape)


def V_tilde_step(
        top_PWCi:       NDArray,            # the top slots of eta_tilde[i], a+W+[K]+C+(n,)
        eta_PWCi:       NDArray,            # eta[i], F+W+C+(n,)
        w_PWo:          NDArray,            # the input jet of site i, F_in+W+(N,)
        w_tilde_PWCo:   NDArray,            # the seed of mode i, F+W+[K]+C+(N,)
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
        keep_W:         bool = False,       # True: W kept, one block per sample (sum_over_samples=False)
) -> NDArray:                               # V_tilde[i]: [K]+C+(n, N), summed over W ([W]+...: keep_W)
    '''``V_tilde[i]``, the corner of ``V_term(eta_tilde[i], w[i]) + V_term(eta[i], w_tilde[i])``. The first term
    from the top slots of ``eta_tilde``: with ``S`` those slots, the corner of ``eta_tilde * w`` is the coefficient
    ``a - 1`` of ``S * w``, so it is a readout at the corner ``a - 1``, the leaf's jet shape.'''
    if keep_W:                              # W kept: the readouts with W first
        return (contractions.jet_contract('PWKCi,PWo->WKCio', top_PWCi, w_PWo,
                                          lens=lens, order=_leaf_order(w_PWo, order), over_orders=form)
                + contractions.jet_contract('PWCi,PWKCo->WKCio', eta_PWCi, w_tilde_PWCo,
                                            lens=lens, order=order, over_orders=form))
    return (contractions.jet_contract('PWKCi,PWo->KCio', top_PWCi, w_PWo,
                                      lens=lens, order=_leaf_order(w_PWo, order), over_orders=form)
            + contractions.jet_contract('PWCi,PWKCo->KCio', eta_PWCi, w_tilde_PWCo,
                                        lens=lens, order=order, over_orders=form))


def H_tilde_step(
        mu_tilde_PWCa:  NDArray,            # mu_tilde at bond i
        mu_PWCa:        NDArray,            # mu at bond i
        xi_tilde_PWCi:  NDArray,            # xi_tilde[i]
        xi_PWCi:        NDArray,            # xi[i], the leaf
        nu_tilde_PWCb:  NDArray,            # nu_tilde at bond i+1
        nu_PWCb:        NDArray,            # nu at bond i+1
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
        keep_W:         bool = False,       # True: W kept, one block per sample (sum_over_samples=False)
) -> NDArray:                               # H_tilde[i]: [K]+C+(ra, n, rb), summed over W ([W]+...: keep_W)
    '''``H_tilde[i]``, the corner of ``H_term(mu_tilde, xi, nu) + H_term(mu, xi_tilde, nu) + H_term(mu, xi,
    nu_tilde)`` at site ``i``: three readouts, the tilde on each of the three in turn.'''
    if keep_W:                              # W kept: the readouts with W first
        return (contractions.jet_contract('PWKCa,PWCi,PWCb->WKCaib', mu_tilde_PWCa, xi_PWCi, nu_PWCb,
                                          lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWCa,PWKCi,PWCb->WKCaib', mu_PWCa, xi_tilde_PWCi, nu_PWCb,
                                            lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWCa,PWCi,PWKCb->WKCaib', mu_PWCa, xi_PWCi, nu_tilde_PWCb,
                                            lens=lens, order=order, over_orders=form))


    ###############################################
    ########    The dot grade (5.2)    ############
    ###############################################
    # A dot step serves both forms of the direction: the frame's (V, H), and P8's raw (D_dot, L_dot) at the base point.
    # The direction's core carries K in the strings; with the frame's form the core over the leaf of the direction's site
    # is U (over xi_U_dot), with the raw form L (over xi_dot).

    def xi_dot_step(
            B_dot_KCio:  NDArray,               # the direction's Tucker core at site i: V[i], K+C+(n, N)
            w_PWo:       NDArray,               # the input jet of the site
            *,
            lens:        contractions.GroupLengths,
            order:       typ.Tuple[int, ...],
            form:        str,
    ) -> NDArray:                               # xi_U_dot[i]: min(F_in, F)+W+K+C+(n,), a leaf
        '''``xi_U_dot[i] = xi_term(V[i], w[i])``: the direction's Tucker core applied to the input jet, a leaf.'''
        return contractions.jet_contract('PWo,KCio->PWKCi', w_PWo, B_dot_KCio,
                                         lens=lens, order=_leaf_order(w_PWo, order), over_orders=form)
    return (contractions.jet_contract('PWKCa,PWCi,PWCb->KCaib', mu_tilde_PWCa, xi_PWCi, nu_PWCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,PWKCi,PWCb->KCaib', mu_PWCa, xi_tilde_PWCi, nu_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,PWCi,PWKCb->KCaib', mu_PWCa, xi_PWCi, nu_tilde_PWCb,
                                        lens=lens, order=order, over_orders=form))


###############################################
########    The dot grade (5.2)    ############
###############################################
# A dot step serves both forms of the direction: the frame's (V, H), and P8's raw (D_dot, L_dot) at the base point.
# The direction's core carries K in the strings; with the frame's form the core over the leaf of the direction's site
# is U (over xi_U_dot), with the raw form L (over xi_dot).

def xi_dot_step(
        B_dot_KCio:  NDArray,               # the direction's Tucker core at site i: V[i], K+C+(n, N)
        w_PWo:       NDArray,               # the input jet of the site
        *,
        lens:        contractions.GroupLengths,
        order:       typ.Tuple[int, ...],
        form:        str,
) -> NDArray:                               # xi_U_dot[i]: min(F_in, F)+W+K+C+(n,), a leaf
    '''``xi_U_dot[i] = xi_term(V[i], w[i])``: the direction's Tucker core applied to the input jet, a leaf.'''
    return contractions.jet_contract('PWo,KCio->PWKCi', w_PWo, B_dot_KCio,
                                     lens=lens, order=_leaf_order(w_PWo, order), over_orders=form)


def mu_dot_step(
        mu_dot_PWKCa:  NDArray,             # the carry at bond i: the direction at one of the sites 0 .. i-1
        G_Caib:        NDArray,             # the TT core the carry passes: R[i] (frame form)
        mu_PWCa:       NDArray,             # mu at bond i
        xi_PWCi:       NDArray,             # the leaf of site i
        xi_dot_PWKCi:  NDArray,             # the direction's leaf at site i: xi_U_dot[i]
        G_up_Caib:     NDArray,             # the TT core over it: U[i]
        H_KCaib:       NDArray,             # the direction's TT core at site i: H[i]
        *,
        lens:          contractions.GroupLengths,
        order:         typ.Tuple[int, ...],
        form:          str,
) -> NDArray:                               # the carry at bond i+1, F+W+K+C+(rb,)
    '''``mu_R_dot[i+1] = mu_term(mu_R_dot[i], R[i], xi[i]) + mu_term(mu[i], H[i], xi[i])
    + mu_term(mu[i], U[i], xi_U_dot[i])``: the carry, or the direction at site ``i`` (``H``, or ``U`` over ``V``).'''
    return (contractions.jet_contract('PWKCa,Caib,PWCi->PWKCb', mu_dot_PWKCa, G_Caib, xi_PWCi,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,KCaib,PWCi->PWKCb', mu_PWCa, H_KCaib, xi_PWCi,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,Caib,PWKCi->PWKCb', mu_PWCa, G_up_Caib, xi_dot_PWKCi,
                                        lens=lens, order=order, over_orders=form))


def nu_dot_step(
        nu_dot_PWKCb:  NDArray,             # the carry at bond i+1: the direction at one of the sites i+1 .. d-1
        G_Caib:        NDArray,             # the TT core the carry passes: L[i] (frame form)
        nu_PWCb:       NDArray,             # nu at bond i+1
        xi_PWCi:       NDArray,             # the leaf of site i
        xi_dot_PWKCi:  NDArray,             # xi_U_dot[i]
        G_up_Caib:     NDArray,             # U[i]
        H_KCaib:       NDArray,             # H[i]
        *,
        lens:          contractions.GroupLengths,
        order:         typ.Tuple[int, ...],
        form:          str,
) -> NDArray:                               # the carry at bond i, F+W+K+C+(ra,)
    '''``nu_L_dot[i] = nu_term(L[i], xi[i], nu_L_dot[i+1]) + nu_term(H[i], xi[i], nu[i+1])
    + nu_term(U[i], xi_U_dot[i], nu[i+1])``.'''
    return (contractions.jet_contract('Caib,PWCi,PWKCb->PWKCa', G_Caib, xi_PWCi, nu_dot_PWKCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('KCaib,PWCi,PWCb->PWKCa', H_KCaib, xi_PWCi, nu_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('Caib,PWKCi,PWCb->PWKCa', G_up_Caib, xi_dot_PWKCi, nu_PWCb,
                                        lens=lens, order=order, over_orders=form))


def eta_dot_step(
        mu_dot_PWKCa:  NDArray,             # mu_R_dot at bond i
        G_left_Caib:   NDArray,             # R[i]: the core of site i when the direction is to its left
        mu_PWCa:       NDArray,             # mu at bond i
        H_KCaib:       NDArray,             # H[i]: the direction at site i itself
        G_right_Caib:  NDArray,             # L[i]: the core of site i when the direction is to its right
        nu_dot_PWKCb:  NDArray,             # nu_L_dot at bond i+1
        nu_PWCb:       NDArray,             # nu at bond i+1
        *,
        lens:          contractions.GroupLengths,
        order:         typ.Tuple[int, ...],
        form:          str,
) -> NDArray:                               # eta_D_dot[i], F+W+K+C+(n,)
    '''``eta_D_dot[i] = eta_term(mu_R_dot[i], R[i], nu[i+1]) + eta_term(mu[i], H[i], nu[i+1])
    + eta_term(mu[i], L[i], nu_L_dot[i+1])``.'''
    return (contractions.jet_contract('PWKCa,Caib,PWCb->PWKCi', mu_dot_PWKCa, G_left_Caib, nu_PWCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,KCaib,PWCb->PWKCi', mu_PWCa, H_KCaib, nu_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,Caib,PWKCb->PWKCi', mu_PWCa, G_right_Caib, nu_dot_PWKCb,
                                        lens=lens, order=order, over_orders=form))


def Jp_step(
        eta_dot_PWKCi:  NDArray,            # eta_D_dot[i], F+W+K+C+(nD,)
        B_Cio:          NDArray,            # D[i]
        eta_PWCi:       NDArray,            # eta[i], F+W+C+(nU,)
        V_KCio:         NDArray,            # V[i], K+C+(nU, N)
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # Jp[i], F+W+K+C+(N,)
    '''``Jp[i] = z_term(eta_D_dot[i], D[i]) + z_term(eta[i], V[i])``: the direction's output.'''
    return (contractions.jet_contract('PWKCi,Cio->PWKCo', eta_dot_PWKCi, B_Cio,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCi,KCio->PWKCo', eta_PWCi, V_KCio,
                                        lens=lens, order=order, over_orders=form))


###############################################
########    The tilde-dot grade (5.4)    ######
###############################################
# At the base point, with the raw form of the direction (D_dot, L_dot): the direction at one site and the seed in place
# of one input, summed over both. K sits on the dotted operand; the residual's seed and its tilde variables carry none.

def xi_tilde_dot_step(
        B_dot_KCio:     NDArray,            # the direction's Tucker core at site i: D_dot[i], K+C+(n, N)
        w_tilde_PWCo:   NDArray,            # the residual's seed of mode i, F+W+C+(N,)
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
) -> NDArray:                               # xi_tilde_dot[i], F+W+K+C+(n,)
    '''``xi_tilde_dot[i] = xi_term(D_dot[i], w_tilde[i])``: the direction's Tucker core applied to the seed.'''
    return contractions.jet_contract('PWCo,KCio->PWKCi', w_tilde_PWCo, B_dot_KCio,
                                     lens=lens, order=order, over_orders=form)


def mu_tilde_dot_step(
        mu_tilde_dot_PWKCa:  NDArray,       # the carry at bond i
        G_Caib:              NDArray,       # L[i]
        mu_dot_PWKCa:        NDArray,       # mu_dot at bond i
        mu_tilde_PWCa:       NDArray,       # mu_tilde at bond i
        mu_PWCa:             NDArray,       # mu at bond i
        xi_tilde_dot_PWKCi:  NDArray,       # xi_tilde_dot[i]
        xi_dot_PWKCi:        NDArray,       # xi_dot[i]
        xi_tilde_PWCi:       NDArray,       # xi_tilde[i]
        xi_PWCi:             NDArray,       # xi[i]
        G_dot_KCaib:         NDArray,       # L_dot[i]
        *,
        lens:                contractions.GroupLengths,
        order:               typ.Tuple[int, ...],
        form:                str,
) -> NDArray:                               # the carry at bond i+1, F+W+K+C+(rb,)
    '''``mu_tilde_dot[i+1]``: six terms, the direction in the carry, the core or the leaf of site ``i``, and the seed
    in the carry or the leaf: ``mu_term`` of ``(mu_tilde_dot, L, xi)``, ``(mu_dot, L, xi_tilde)``, ``(mu_tilde, L,
    xi_dot)``, ``(mu, L, xi_tilde_dot)``, ``(mu_tilde, L_dot, xi)``, ``(mu, L_dot, xi_tilde)``.'''
    return (contractions.jet_contract('PWKCa,Caib,PWCi->PWKCb', mu_tilde_dot_PWKCa, G_Caib, xi_PWCi,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWKCa,Caib,PWCi->PWKCb', mu_dot_PWKCa, G_Caib, xi_tilde_PWCi,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,Caib,PWKCi->PWKCb', mu_tilde_PWCa, G_Caib, xi_dot_PWKCi,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,Caib,PWKCi->PWKCb', mu_PWCa, G_Caib, xi_tilde_dot_PWKCi,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,KCaib,PWCi->PWKCb', mu_tilde_PWCa, G_dot_KCaib, xi_PWCi,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,KCaib,PWCi->PWKCb', mu_PWCa, G_dot_KCaib, xi_tilde_PWCi,
                                        lens=lens, order=order, over_orders=form))


def nu_tilde_dot_step(
        nu_tilde_dot_PWKCb:  NDArray,       # the carry at bond i+1
        G_Caib:              NDArray,       # L[i]
        nu_dot_PWKCb:        NDArray,       # the raw nu_L_dot at bond i+1
        nu_tilde_PWCb:       NDArray,       # nu_L_tilde at bond i+1
        nu_PWCb:             NDArray,       # nu_L at bond i+1
        xi_tilde_dot_PWKCi:  NDArray,       # xi_tilde_dot[i]
        xi_dot_PWKCi:        NDArray,       # xi_dot[i]
        xi_tilde_PWCi:       NDArray,       # xi_tilde[i]
        xi_PWCi:             NDArray,       # xi[i]
        G_dot_KCaib:         NDArray,       # L_dot[i]
        *,
        lens:                contractions.GroupLengths,
        order:               typ.Tuple[int, ...],
        form:                str,
) -> NDArray:                               # the carry at bond i, F+W+K+C+(ra,)
    '''``nu_L_tilde_dot[i]``: the six mirror terms, ``nu_term`` of ``(L, xi, nu_L_tilde_dot)``, ``(L, xi_tilde,
    nu_L_dot)``, ``(L, xi_dot, nu_L_tilde)``, ``(L, xi_tilde_dot, nu_L)``, ``(L_dot, xi, nu_L_tilde)``, ``(L_dot,
    xi_tilde, nu_L)``.'''
    return (contractions.jet_contract('Caib,PWCi,PWKCb->PWKCa', G_Caib, xi_PWCi, nu_tilde_dot_PWKCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('Caib,PWCi,PWKCb->PWKCa', G_Caib, xi_tilde_PWCi, nu_dot_PWKCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('Caib,PWKCi,PWCb->PWKCa', G_Caib, xi_dot_PWKCi, nu_tilde_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('Caib,PWKCi,PWCb->PWKCa', G_Caib, xi_tilde_dot_PWKCi, nu_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('KCaib,PWCi,PWCb->PWKCa', G_dot_KCaib, xi_PWCi, nu_tilde_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('KCaib,PWCi,PWCb->PWKCa', G_dot_KCaib, xi_tilde_PWCi, nu_PWCb,
                                        lens=lens, order=order, over_orders=form))


def eta_tilde_dot_step(
        mu_tilde_dot_PWKCa:  NDArray,       # mu_tilde_dot at bond i
        mu_dot_PWKCa:        NDArray,       # mu_dot at bond i
        mu_tilde_PWCa:       NDArray,       # mu_tilde at bond i
        mu_PWCa:             NDArray,       # mu at bond i
        G_Caib:              NDArray,       # L[i]
        nu_tilde_dot_PWKCb:  NDArray,       # nu_L_tilde_dot at bond i+1
        nu_dot_PWKCb:        NDArray,       # the raw nu_L_dot at bond i+1
        nu_tilde_PWCb:       NDArray,       # nu_L_tilde at bond i+1
        nu_PWCb:             NDArray,       # nu_L at bond i+1
        G_dot_KCaib:         NDArray,       # L_dot[i]
        *,
        lens:                contractions.GroupLengths,
        order:               typ.Tuple[int, ...],
        form:                str,
) -> NDArray:                               # eta_D_tilde_dot[i], F+W+K+C+(n,), every slot
    '''``eta_D_tilde_dot[i]``: six ``eta_term`` of ``(mu_tilde_dot, L, nu_L)``, ``(mu_dot, L, nu_L_tilde)``,
    ``(mu_tilde, L, nu_L_dot)``, ``(mu, L, nu_L_tilde_dot)``, ``(mu_tilde, L_dot, nu_L)``, ``(mu, L_dot,
    nu_L_tilde)``.'''
    return (contractions.jet_contract('PWKCa,Caib,PWCb->PWKCi', mu_tilde_dot_PWKCa, G_Caib, nu_PWCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWKCa,Caib,PWCb->PWKCi', mu_dot_PWKCa, G_Caib, nu_tilde_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,Caib,PWKCb->PWKCi', mu_tilde_PWCa, G_Caib, nu_dot_PWKCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,Caib,PWKCb->PWKCi', mu_PWCa, G_Caib, nu_tilde_dot_PWKCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,KCaib,PWCb->PWKCi', mu_tilde_PWCa, G_dot_KCaib, nu_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,KCaib,PWCb->PWKCi', mu_PWCa, G_dot_KCaib, nu_tilde_PWCb,
                                        lens=lens, order=order, over_orders=form))


def eta_tilde_dot_top_step(
        mu_tilde_dot_PWKCa:  NDArray,       # as for eta_tilde_dot_step
        mu_dot_PWKCa:        NDArray,
        mu_tilde_PWCa:       NDArray,
        mu_PWCa:             NDArray,
        G_Caib:              NDArray,
        nu_tilde_dot_PWKCb:  NDArray,
        nu_dot_PWKCb:        NDArray,
        nu_tilde_PWCb:       NDArray,
        nu_PWCb:             NDArray,
        G_dot_KCaib:         NDArray,
        w_PWo:               NDArray,       # the input jet of site i: only its jet sizes are read
        *,
        lens:                contractions.GroupLengths,
        order:               typ.Tuple[int, ...],
        form:                str,
) -> NDArray:                               # the top slots of eta_D_tilde_dot[i]: a+W+K+C+(n,), a = min(F_in, F)
    '''The short form of ``eta_D_tilde_dot[i]``: the slots ``V_tilde_dot`` reads, ``[c - a + 1 : c + 1]`` on every
    jet axis, each the six terms read at that slot (readouts at a lowered corner).'''
    xnp, _, _ = get_backend(False, tree_contains_jax((mu_tilde_dot_PWKCa, mu_dot_PWKCa, mu_tilde_PWCa, mu_PWCa,
                                                      G_Caib, nu_tilde_dot_PWKCb, nu_dot_PWKCb, nu_tilde_PWCb,
                                                      nu_PWCb, G_dot_KCaib)))
    a = tuple(f + 1 for f in _leaf_order(w_PWo, order))
    slots = []
    for j in np.ndindex(*a):
        corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))
        slots.append(contractions.jet_contract('PWKCa,Caib,PWCb->WKCi', mu_tilde_dot_PWKCa, G_Caib, nu_PWCb,
                                               lens=lens, order=corner, over_orders=form)
                     + contractions.jet_contract('PWKCa,Caib,PWCb->WKCi', mu_dot_PWKCa, G_Caib, nu_tilde_PWCb,
                                                 lens=lens, order=corner, over_orders=form)
                     + contractions.jet_contract('PWCa,Caib,PWKCb->WKCi', mu_tilde_PWCa, G_Caib, nu_dot_PWKCb,
                                                 lens=lens, order=corner, over_orders=form)
                     + contractions.jet_contract('PWCa,Caib,PWKCb->WKCi', mu_PWCa, G_Caib, nu_tilde_dot_PWKCb,
                                                 lens=lens, order=corner, over_orders=form)
                     + contractions.jet_contract('PWCa,KCaib,PWCb->WKCi', mu_tilde_PWCa, G_dot_KCaib, nu_PWCb,
                                                 lens=lens, order=corner, over_orders=form)
                     + contractions.jet_contract('PWCa,KCaib,PWCb->WKCi', mu_PWCa, G_dot_KCaib, nu_tilde_PWCb,
                                                 lens=lens, order=corner, over_orders=form))
    return xnp.stack(slots).reshape(a + slots[0].shape)


def V_tilde_dot_step(
        top_PWKCi:      NDArray,            # the top slots of eta_D_tilde_dot[i], a+W+K+C+(nD,)
        eta_dot_PWKCi:  NDArray,            # the raw eta_D_dot[i], F+W+K+C+(nD,)
        w_PWo:          NDArray,            # the input jet of site i
        w_tilde_PWCo:   NDArray,            # the residual's seed of mode i, F+W+C+(N,)
        *,
        lens:           contractions.GroupLengths,
        order:          typ.Tuple[int, ...],
        form:           str,
        keep_W:         bool = False,       # True: W kept, one block per sample (sum_over_samples=False)
) -> NDArray:                               # V_tilde_dot[i]: K+C+(nD, N), summed over W ([W]+...: keep_W)
    '''``V_tilde_dot[i]``, the corner of ``V_term(eta_D_tilde_dot[i], w[i]) + V_term(eta_D_dot[i], w_tilde[i])``,
    the first from the top slots at the corner ``a - 1``.'''
    if keep_W:                              # W kept: the readouts with W first
        return (contractions.jet_contract('PWKCi,PWo->WKCio', top_PWKCi, w_PWo,
                                          lens=lens, order=_leaf_order(w_PWo, order), over_orders=form)
                + contractions.jet_contract('PWKCi,PWCo->WKCio', eta_dot_PWKCi, w_tilde_PWCo,
                                            lens=lens, order=order, over_orders=form))
    return (contractions.jet_contract('PWKCi,PWo->KCio', top_PWKCi, w_PWo,
                                      lens=lens, order=_leaf_order(w_PWo, order), over_orders=form)
            + contractions.jet_contract('PWKCi,PWCo->KCio', eta_dot_PWKCi, w_tilde_PWCo,
                                        lens=lens, order=order, over_orders=form))


def H_tilde_dot_step(
        mu_tilde_dot_PWKCa:  NDArray,       # at bond i
        mu_dot_PWKCa:        NDArray,
        mu_tilde_PWCa:       NDArray,
        mu_PWCa:             NDArray,
        xi_tilde_dot_PWKCi:  NDArray,       # at site i
        xi_dot_PWKCi:        NDArray,
        xi_tilde_PWCi:       NDArray,
        xi_PWCi:             NDArray,
        nu_tilde_dot_PWKCb:  NDArray,       # at bond i+1, the L basis
        nu_dot_PWKCb:        NDArray,
        nu_tilde_PWCb:       NDArray,
        nu_PWCb:             NDArray,
        *,
        lens:                contractions.GroupLengths,
        order:               typ.Tuple[int, ...],
        form:                str,
        keep_W:              bool = False,  # True: W kept, one block per sample (sum_over_samples=False)
) -> NDArray:                               # H_tilde_dot[i]: K+C+(rL_i,nD_i,rL_(i+1)), summed over W ([W]+...: keep_W)
    '''``H_tilde_dot[i]``, nine readouts of ``(mu, xi, nu_L)``: the tilde on one and the dot on one, both on one
    (its tilde-dot) in three, on two different ones in six.'''
    if keep_W:                              # W kept: the readouts with W first
        return (contractions.jet_contract('PWKCa,PWCi,PWCb->WKCaib', mu_tilde_dot_PWKCa, xi_PWCi, nu_PWCb,
                                          lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWCa,PWKCi,PWCb->WKCaib', mu_PWCa, xi_tilde_dot_PWKCi, nu_PWCb,
                                            lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWCa,PWCi,PWKCb->WKCaib', mu_PWCa, xi_PWCi, nu_tilde_dot_PWKCb,
                                            lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWKCa,PWCi,PWCb->WKCaib', mu_dot_PWKCa, xi_tilde_PWCi, nu_PWCb,
                                            lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWKCa,PWCi,PWCb->WKCaib', mu_dot_PWKCa, xi_PWCi, nu_tilde_PWCb,
                                            lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWCa,PWKCi,PWCb->WKCaib', mu_tilde_PWCa, xi_dot_PWKCi, nu_PWCb,
                                            lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWCa,PWKCi,PWCb->WKCaib', mu_PWCa, xi_dot_PWKCi, nu_tilde_PWCb,
                                            lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWCa,PWCi,PWKCb->WKCaib', mu_tilde_PWCa, xi_PWCi, nu_dot_PWKCb,
                                            lens=lens, order=order, over_orders=form)
                + contractions.jet_contract('PWCa,PWCi,PWKCb->WKCaib', mu_PWCa, xi_tilde_PWCi, nu_dot_PWKCb,
                                            lens=lens, order=order, over_orders=form))


    #####################################################
    ########    The adjoint of apply (5.3)    ###########
    #####################################################
    # For the apply kind the seed sits on the ones that close bond d: a right side or a trunk is the plain recurrence
    # started from the seed (nu_step, eta_step; the tilde-dot: nu_dot_step, eta_dot_step and H_tilde_step with the tilde
    # forms in the plain ones' places), and a left side or a leaf has no tilde. What is new are the readouts of one
    # term: the short forms of eta_step and eta_dot_step, and the blocks with no seed at a leaf.

    def eta_top_step(
            mu_PWCa:   NDArray,                 # mu at bond i, F+W+C+(ra,)
            G_Caib:    NDArray,                 # U[i], or L[i] for eta_D_tilde
            nu_PWKCb:  NDArray,                 # the apply kind's nu_tilde at bond i+1, F+W+[K]+C+(rb,)
            w_PWo:     NDArray,                 # the input jet of site i: only its jet sizes are read
            *,
            lens:      contractions.GroupLengths,
            order:     typ.Tuple[int, ...],
            form:      str,
    ) -> NDArray:                               # the top slots of eta_tilde[i]: a+W+[K]+C+(n,), a = min(F_in, F)
        '''The short form of ``eta_step``: the slots ``[c - a + 1 : c + 1]`` of ``eta_term(mu[i], G[i], nu[i+1])`` on
        every jet axis (``c`` the corner, ``a`` the input's jet sizes clipped to the box), each a readout at a lowered
        corner. With the apply kind's ``nu_tilde`` it is the short form of its ``eta_tilde``.'''
        xnp, _, _ = get_backend(False, tree_contains_jax((mu_PWCa, G_Caib, nu_PWKCb)))
        a = tuple(f + 1 for f in _leaf_order(w_PWo, order))
        slots = []
        for j in np.ndindex(*a):
            corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))
            slots.append(contractions.jet_contract('PWCa,Caib,PWKCb->WKCi', mu_PWCa, G_Caib, nu_PWKCb,
                                                   lens=lens, order=corner, over_orders=form))
        return xnp.stack(slots).reshape(a + slots[0].shape)
    return (contractions.jet_contract('PWKCa,PWCi,PWCb->KCaib', mu_tilde_dot_PWKCa, xi_PWCi, nu_PWCb,
                                      lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,PWKCi,PWCb->KCaib', mu_PWCa, xi_tilde_dot_PWKCi, nu_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,PWCi,PWKCb->KCaib', mu_PWCa, xi_PWCi, nu_tilde_dot_PWKCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWKCa,PWCi,PWCb->KCaib', mu_dot_PWKCa, xi_tilde_PWCi, nu_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWKCa,PWCi,PWCb->KCaib', mu_dot_PWKCa, xi_PWCi, nu_tilde_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,PWKCi,PWCb->KCaib', mu_tilde_PWCa, xi_dot_PWKCi, nu_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,PWKCi,PWCb->KCaib', mu_PWCa, xi_dot_PWKCi, nu_tilde_PWCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,PWCi,PWKCb->KCaib', mu_tilde_PWCa, xi_PWCi, nu_dot_PWKCb,
                                        lens=lens, order=order, over_orders=form)
            + contractions.jet_contract('PWCa,PWCi,PWKCb->KCaib', mu_PWCa, xi_tilde_PWCi, nu_dot_PWKCb,
                                        lens=lens, order=order, over_orders=form))


#####################################################
########    The adjoint of apply (5.3)    ###########
#####################################################
# For the apply kind the seed sits on the ones that close bond d: a right side or a trunk is the plain recurrence
# started from the seed (nu_step, eta_step; the tilde-dot: nu_dot_step, eta_dot_step and H_tilde_step with the tilde
# forms in the plain ones' places), and a left side or a leaf has no tilde. What is new are the readouts of one
# term: the short forms of eta_step and eta_dot_step, and the blocks with no seed at a leaf.

def eta_top_step(
        mu_PWCa:   NDArray,                 # mu at bond i, F+W+C+(ra,)
        G_Caib:    NDArray,                 # U[i], or L[i] for eta_D_tilde
        nu_PWKCb:  NDArray,                 # the apply kind's nu_tilde at bond i+1, F+W+[K]+C+(rb,)
        w_PWo:     NDArray,                 # the input jet of site i: only its jet sizes are read
        *,
        lens:      contractions.GroupLengths,
        order:     typ.Tuple[int, ...],
        form:      str,
) -> NDArray:                               # the top slots of eta_tilde[i]: a+W+[K]+C+(n,), a = min(F_in, F)
    '''The short form of ``eta_step``: the slots ``[c - a + 1 : c + 1]`` of ``eta_term(mu[i], G[i], nu[i+1])`` on
    every jet axis (``c`` the corner, ``a`` the input's jet sizes clipped to the box), each a readout at a lowered
    corner. With the apply kind's ``nu_tilde`` it is the short form of its ``eta_tilde``.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((mu_PWCa, G_Caib, nu_PWKCb)))
    a = tuple(f + 1 for f in _leaf_order(w_PWo, order))
    slots = []
    for j in np.ndindex(*a):
        corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))
        slots.append(contractions.jet_contract('PWCa,Caib,PWKCb->WKCi', mu_PWCa, G_Caib, nu_PWKCb,
                                               lens=lens, order=corner, over_orders=form))
    return xnp.stack(slots).reshape(a + slots[0].shape)


def eta_dot_top_step(
        mu_dot_PWKCa:  NDArray,             # as for eta_dot_step
        G_left_Caib:   NDArray,
        mu_PWCa:       NDArray,
        H_KCaib:       NDArray,
        G_right_Caib:  NDArray,
        nu_dot_PWKCb:  NDArray,
        nu_PWCb:       NDArray,
        w_PWo:         NDArray,             # the input jet of site i: only its jet sizes are read
        *,
        lens:          contractions.GroupLengths,
        order:         typ.Tuple[int, ...],
        form:          str,
) -> NDArray:                               # the top slots: a+W+K+C+(n,), a = min(F_in, F)
    '''The short form of ``eta_dot_step``: its three terms read at the slots ``[c - a + 1 : c + 1]``. With the
    apply kind's ``nu_L_tilde`` and ``nu_L_tilde_dot`` in the places of ``nu`` and ``nu_dot``, the short form of its
    ``eta_D_tilde_dot``.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((mu_dot_PWKCa, G_left_Caib, mu_PWCa, H_KCaib, G_right_Caib,
                                                      nu_dot_PWKCb, nu_PWCb)))
    a = tuple(f + 1 for f in _leaf_order(w_PWo, order))
    slots = []
    for j in np.ndindex(*a):
        corner = tuple(n - a_k + 1 + j_k for n, a_k, j_k in zip(order, a, j))
        slots.append(contractions.jet_contract('PWKCa,Caib,PWCb->WKCi', mu_dot_PWKCa, G_left_Caib, nu_PWCb,
                                               lens=lens, order=corner, over_orders=form)
                     + contractions.jet_contract('PWCa,KCaib,PWCb->WKCi', mu_PWCa, H_KCaib, nu_PWCb,
                                                 lens=lens, order=corner, over_orders=form)
                     + contractions.jet_contract('PWCa,Caib,PWKCb->WKCi', mu_PWCa, G_right_Caib, nu_dot_PWKCb,
                                                 lens=lens, order=corner, over_orders=form))
    return xnp.stack(slots).reshape(a + slots[0].shape)


def V_top_step(
        top_PWKCi:  NDArray,                # the top slots of a trunk's tilde, a+W+[K]+C+(n,)
        w_PWo:      NDArray,                # the input jet of site i, F_in+W+(N,)
        *,
        lens:       contractions.GroupLengths,
        order:      typ.Tuple[int, ...],
        form:       str,
        keep_W:     bool = False,           # True: W kept, one block per sample (sum_over_samples=False)
) -> NDArray:                               # the block: [K]+C+(n, N), summed over W ([W]+...: keep_W)
    '''The block ``V_term`` of a trunk's tilde and the input, read at the corner: from the top slots, a readout at
    the corner ``a - 1``. The apply kind's ``V_tilde`` and ``V_tilde_dot`` (no seed at a leaf: one term).'''
    if keep_W:                              # W kept: the readouts with W first
        return contractions.jet_contract('PWKCi,PWo->WKCio', top_PWKCi, w_PWo,
                                         lens=lens, order=_leaf_order(w_PWo, order), over_orders=form)
    return contractions.jet_contract('PWKCi,PWo->KCio', top_PWKCi, w_PWo,
                                     lens=lens, order=_leaf_order(w_PWo, order), over_orders=form)


def H_readout_step(
        mu_PWCa:   NDArray,                 # mu at bond i
        xi_PWCi:   NDArray,                 # xi[i], the leaf
        nu_PWKCb:  NDArray,                 # the apply kind's nu_tilde at bond i+1, F+W+[K]+C+(rb,)
        *,
        lens:      contractions.GroupLengths,
        order:     typ.Tuple[int, ...],
        form:      str,
        keep_W:    bool = False,            # True: W kept, one block per sample (sum_over_samples=False)
) -> NDArray:                               # H_tilde[i]: [K]+C+(ra, n, rb), summed over W ([W]+...: keep_W)
    '''The apply kind's ``H_tilde[i]``, the corner of ``H_term(mu[i], xi[i], nu_tilde[i+1])``: one readout, the
    seed being only on the right.'''
    if keep_W:                              # W kept: the readouts with W first
        return contractions.jet_contract('PWCa,PWCi,PWKCb->WKCaib', mu_PWCa, xi_PWCi, nu_PWKCb,
                                         lens=lens, order=order, over_orders=form)


    #####################################################
    ########    The entries kind's leaf (5.1, 6)    #####
    #####################################################
    # The entries kind's input at a site is the unit vector e_index: the leaf is a gather (a slice of the Tucker core,
    # not a product), and a block of the Tucker core's shape is a scatter-add of the trunk's tilde into the columns
    # index. No jets: the entries kind has order () only.

    def xi_gather_step(
            B_Cio:     NDArray,                 # a Tucker core, or a direction in its shape: [K]+C+(n, N)
            index_W:   NDArray,                 # the grid points' coordinate at this site, int, shape=W
            *,
            lens:      contractions.GroupLengths,
            order:     typ.Tuple[int, ...],     # () (the entries kind has no jets)
            form:      str,
    ) -> NDArray:                               # the leaf: W+[K]+C+(n,)
        '''The entries kind's leaf, ``xi[i] = D[i][..., index[i]]`` moved to ``W + C + (n,)``: the columns of the core at
        the grid points (a gather, no contraction). Of ``V`` it is ``xi_U_dot``, of ``D_dot`` the raw ``xi_dot``.'''
        xnp, _, _ = get_backend(False, tree_contains_jax((B_Cio, index_W)))
        n_W = np.ndim(index_W)
        taken = B_Cio[..., index_W]                                     # [K]+C+(n,)+W
        return xnp.moveaxis(taken, tuple(range(taken.ndim - n_W, taken.ndim)), tuple(range(n_W)))
    return contractions.jet_contract('PWCa,PWCi,PWKCb->KCaib', mu_PWCa, xi_PWCi, nu_PWKCb,
                                     lens=lens, order=order, over_orders=form)


#####################################################
########    The entries kind's leaf (5.1, 6)    #####
#####################################################
# The entries kind's input at a site is the unit vector e_index: the leaf is a gather (a slice of the Tucker core,
# not a product), and a block of the Tucker core's shape is a scatter-add of the trunk's tilde into the columns
# index. No jets: the entries kind has order () only.

def xi_gather_step(
        B_Cio:     NDArray,                 # a Tucker core, or a direction in its shape: [K]+C+(n, N)
        index_W:   NDArray,                 # the grid points' coordinate at this site, int, shape=W
        *,
        lens:      contractions.GroupLengths,
        order:     typ.Tuple[int, ...],     # () (the entries kind has no jets)
        form:      str,
) -> NDArray:                               # the leaf: W+[K]+C+(n,)
    '''The entries kind's leaf, ``xi[i] = D[i][..., index[i]]`` moved to ``W + C + (n,)``: the columns of the core at
    the grid points (a gather, no contraction). Of ``V`` it is ``xi_U_dot``, of ``D_dot`` the raw ``xi_dot``.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((B_Cio, index_W)))
    n_W = np.ndim(index_W)
    taken = B_Cio[..., index_W]                                     # [K]+C+(n,)+W
    return xnp.moveaxis(taken, tuple(range(taken.ndim - n_W, taken.ndim)), tuple(range(n_W)))


def w_one_hot_step(
        index:  NDArray,                    # the grid points, int, shape=(d,)+W
        D:      typ.Sequence[NDArray],      # the Tucker cores, read for the mode sizes N_i and the dtype
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],        # ()
        form:   str,
        kind:   str,
) -> typ.Tuple[NDArray, ...]:               # w = e_index. len=d, elm_shape=W+(Ni,), in D's dtype
    '''The entries kind's input vectors, by their definition: the one-hot vectors ``e_index[i]``
    (:py:func:`~t3toolbox.backend.jets.index_to_one_hot`, so its index semantics: a numpy index outside ``[-N_i, N_i)``
    raises an ``IndexError``, as the gather does), in the dtype of ``D`` (the result dtype, ``D`` being cast). Made only
    when ``'w'`` is wanted: nothing else reads it.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((index, D)))
    vectors = jets.index_to_one_hot(index, [np.shape(B)[-1] for B in D])
    if is_ndarray(D):                                   # the uniform layout: the padded N, packed (the index is real)
        return xnp.stack([xnp.asarray(e, dtype=D.dtype) for e in vectors])
    return tuple(xnp.asarray(e, dtype=D[0].dtype) for e in vectors)


def V_scatter_step(
        eta_WKCi:  NDArray,                 # a trunk's tilde at site i: W+[K]+C+(n,)
        index_W:   NDArray,                 # the grid points' coordinate at this site, int, shape=W
        B_Cio:     NDArray,                 # D[i]: read for its mode size N only
        *,
        lens:      contractions.GroupLengths,
        order:     typ.Tuple[int, ...],     # ()
        form:      str,
        keep_W:    bool = False,            # True: one block per sample, W+[K]+C+(n, N) (P10's sum_over_samples=False)
) -> NDArray:                               # the block: [K]+C+(n, N), summed over the samples
    '''The entries kind's ``V_tilde`` (and ``V_tilde_dot``): ``V_term(eta_tilde[i], e_index[i])``, the trunk's tilde
    added into the column ``index[i]`` of each sample. A scatter-add, ``W x n`` in memory (not the one-hot product's
    ``W x N``); duplicate indices add up (``.at[].add`` on jax, ``np.add.at`` on numpy; a plain ``+=`` would keep
    one of them). With ``keep_W`` each sample has its own block, so nothing collides and it is a scatter.'''
    use_jax = tree_contains_jax((eta_WKCi, index_W, B_Cio))
    xnp, _, _ = get_backend(False, use_jax)
    n_W, N = np.ndim(index_W), np.shape(B_Cio)[-1]
    rest = tuple(np.shape(eta_WKCi)[n_W:])                         # [K]+C+(n,)
    if keep_W:
        W = tuple(np.shape(eta_WKCi)[:n_W])
        grids = tuple(xnp.arange(size).reshape((size,) + (1,) * (n_W - 1 - k)) for k, size in enumerate(W))
        where = grids + (xnp.asarray(index_W),)
        out_WN = xnp.zeros(W + (N,) + rest, dtype=eta_WKCi.dtype)
        out_WN = out_WN.at[where].set(eta_WKCi) if use_jax else _numpy_set(out_WN, where, eta_WKCi)
        return xnp.moveaxis(out_WN, n_W, -1)                       # W+[K]+C+(n, N)
    out_N = xnp.zeros((N,) + rest, dtype=eta_WKCi.dtype)
    if use_jax:
        out_N = out_N.at[index_W].add(eta_WKCi)
    else:
        np.add.at(out_N, np.asarray(index_W), eta_WKCi)
    return xnp.moveaxis(out_N, 0, -1)                              # [K]+C+(n, N)


def _numpy_set(out, where, values):
    out[where] = values
    return out


###############################################
########    The whole steps    ################
###############################################

# A whole step takes the kind as a static too, a string: the probe kind's type of z is one array per mode with a mode
# edge, the apply kind's one array with neither. _modes gives the arrays of a value of the type of z, each with the
# weight's row it is read with and the number of mode-edge axes after its stacks.

def _modes(
        kind:    str,                       # 'probe' | 'apply'
        value:   typ.Any,                   # of the type of z: probe len=d, elm_shape=F+W+[K]+C+(Ni,); apply F+W+[K]+C
        weight:  typ.Optional[NDArray],     # probe (d,)+F; apply F; None = ones
        xnp,
) -> typ.List[typ.Tuple[NDArray, typ.Optional[NDArray], int]]:     # (array, its weight F or None, mode-edge axes)
    if kind == 'probe':
        return [(xnp.asarray(array), None if weight is None else xnp.asarray(weight)[i], 1)
                for i, array in enumerate(value)]
    return [(xnp.asarray(value), None if weight is None else xnp.asarray(weight), 0)]


def identity_step(
        value:  typ.Any,                    # a whole variable
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],
        form:   str,
        kind:   str,
) -> typ.Any:                               # the same object
    '''A route row over a private node: the public name of a variable whose value another node makes (the frame form
    of ``nu_L_dot`` and ``eta_D_dot``).'''
    return value


def close_step(
        bonds:  typ.Sequence[NDArray],      # a left side, len=d+1, elm_shape=F+W+[K]+C+(r_i,)
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],
        form:   str,
        kind:   str,
) -> NDArray:                               # bond d closed by the ones: F+W+[K]+C
    '''The apply kind's output: the left side at bond ``d`` closed by the vector of ones of its end bond, a SUM over
    the last axis (with tails the end bond is larger than one). Of ``mu`` it is ``z``, closed with ``rL_d`` ones; of
    ``mu_R_dot`` (or ``mu_dot``) it is ``Jp``, closed with the ones of the end bond of the store the sweep ends in.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((bonds[-1],)))
    return xnp.sum(bonds[-1], axis=-1)


def Jp_norm_sq_step(
        Jp:      typ.Any,                   # of the type of z: probe len=d, elm_shape=F+W+K+C+(Ni,); apply F+W+K+C
        weight:  typ.Optional[NDArray],     # probe (d,)+F; apply F; None = ones
        *,
        lens:    contractions.GroupLengths,
        order:   typ.Tuple[int, ...],
        form:    str,
        kind:    str,
) -> NDArray:                               # <Jp, weight**2 Jp>, keeping K + C. shape=K+C
    '''The weighted sum of squares of the direction's output: ``<Jp, weight**2 Jp>``, summed over the modes, the
    slots, the samples and the mode edge, keeping ``K + C`` (no half).'''
    xnp, _, _ = get_backend(False, tree_contains_jax((Jp, weight)))
    m, n_W = lens.P, lens.W
    total = None
    for Jp_FWKCo, weight_F, edge in _modes(kind, Jp, weight, xnp):
        squares = Jp_FWKCo * Jp_FWKCo
        if weight_F is not None:
            squares = (weight_F * weight_F).reshape(weight_F.shape + (1,) * (Jp_FWKCo.ndim - m)) * squares
        summed = xnp.sum(squares, axis=tuple(range(m + n_W)) + tuple(range(Jp_FWKCo.ndim - edge, Jp_FWKCo.ndim)))
        total = summed if total is None else total + summed
    return xnp.asarray(total)


def w_tilde_step(
        r:       typ.Any,                   # of the type of z: probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C
        weight:  typ.Optional[NDArray],     # probe (d,)+F; apply F; None = ones
        *,
        lens:    contractions.GroupLengths,
        order:   typ.Tuple[int, ...],
        form:    str,
        kind:    str,
) -> typ.Any:                               # the seed, of the type of z, stored turned
    '''The seed of the adjoint: the residual made a dual jet, ``jet_to_dual_jet``: the weight squared, then every jet
    axis flipped. The probe kind's is one array per mode; the apply kind's one number per sample and slot, which sits
    on the ones that close bond ``d``.'''
    seed = jets.jet_to_dual_jet(kind, order, r, weight)
    if kind == 'probe' and is_ndarray(r):               # the uniform layout: packed in, packed out
        xnp, _, _ = get_backend(False, tree_contains_jax(seed))
        return xnp.stack(seed)
    return seed


def g_step(
        V_tilde:  typ.Sequence[NDArray],    # len=d, elm_shape=C+(nUi, Ni)
        H_tilde:  typ.Sequence[NDArray],    # len=d, elm_shape=C+(rLi, nDi, rR(i+1))
        *,
        lens:     contractions.GroupLengths,
        order:    typ.Tuple[int, ...],
        form:     str,
        kind:     str,
) -> typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]:   # g = (V_tilde, H_tilde)
    '''The gradient: the two families of blocks, paired (a tuple of ``d`` blocks each on ragged, a supercore each on
    the uniform layout, as given).'''
    if is_ndarray(V_tilde):
        return V_tilde, H_tilde
    return tuple(V_tilde), tuple(H_tilde)


def r_step(
        z:      typ.Any,                    # of the type of z: probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C
        y:      typ.Any,                    # the data, broadcast over C: probe len=d, elm_shape=F+W+(Ni,); apply F+W
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],
        form:   str,
        kind:   str,
) -> typ.Any:                               # r = z - y, of the type of z
    '''The residual, the library's sign: ``r = z - y``, mode by mode for the probe kind.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((z, y)))
    if kind != 'probe':
        y_FW = xnp.asarray(y)
        return xnp.asarray(z) - y_FW.reshape(y_FW.shape + (1,) * lens.C)
    if is_ndarray(z):                                   # the uniform layout: (d,)+F+W+C+(N,) and (d,)+F+W+(N,)
        y_dFWo = xnp.asarray(y)
        return z - y_dFWo.reshape(y_dFWo.shape[:-1] + (1,) * lens.C + y_dFWo.shape[-1:])
    out = []
    for z_FWCo, y_FWo in zip(z, y):
        y_FWo = xnp.asarray(y_FWo)
        y_FW1o = y_FWo.reshape(y_FWo.shape[:-1] + (1,) * lens.C + y_FWo.shape[-1:])
        out.append(z_FWCo - y_FW1o)
    return tuple(out)


def f_step(
        r:       typ.Any,                   # of the type of z: probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C
        weight:  typ.Optional[NDArray],     # probe (d,)+F; apply F; None = ones
        *,
        lens:    contractions.GroupLengths,
        order:   typ.Tuple[int, ...],
        form:    str,
        kind:    str,
) -> NDArray:                               # f = 1/2 <r, weight**2 r>, keeping C. shape=C
    '''The misfit: half the sum, over the modes, the slots, the samples and the mode edge, of the squared weighted
    residual. The stack ``C`` is kept.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((r, weight)))
    m, n_W = lens.P, lens.W
    total = None
    for r_FWCo, weight_F, edge in _modes(kind, r, weight, xnp):
        squares = r_FWCo * r_FWCo
        if weight_F is not None:
            squares = (weight_F * weight_F).reshape(weight_F.shape + (1,) * (r_FWCo.ndim - m)) * squares
        axes = tuple(range(m + n_W)) + tuple(range(r_FWCo.ndim - edge, r_FWCo.ndim))
        summed = xnp.sum(squares, axis=axes)
        total = summed if total is None else total + summed
    half = xnp.asarray(0.5, dtype=xnp.asarray(total).dtype)    # numpy 1.x: a Python float lifts a float32 sum
    return xnp.asarray(half * total)                            # an array, not a numpy scalar


def r_block_sumsq_step(
        r:      typ.Any,                    # of the type of z: probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C
        *,
        lens:   contractions.GroupLengths,
        order:  typ.Tuple[int, ...],
        form:   str,
        kind:   str,
) -> NDArray:                               # the unweighted sums of squares per block: the weight's shape
    '''The sums of squares of the residual, unweighted, per block: per mode and slot for the probe kind (the shape
    ``(d,) + F``), per slot for the apply kind (``F``). Summed over the samples, the stack ``C`` and the mode edge.'''
    xnp, _, _ = get_backend(False, tree_contains_jax(r))
    m = lens.P
    sums = [xnp.sum(r_FWCo * r_FWCo, axis=tuple(range(m, r_FWCo.ndim))) for r_FWCo, _, _ in _modes(kind, r, None, xnp)]
    return xnp.stack(sums) if kind == 'probe' else sums[0]


###############################################
########    The start values    ###############
###############################################

def start_ones(
        box:    typ.Tuple[int, ...],        # F, the full jet sizes
        W:      typ.Tuple[int, ...],        # the sample stack
        KC:     typ.Tuple[int, ...],        # the stacks after W: K + C for a row that carries K, else C
        size:   int,                        # the end bond, in the basis of the sweep's own store
        dtype:  typ.Any,
        xnp,
) -> NDArray:                               # F+W+[K]+C+(size,): ones in slot (0, ..., 0), zero in every other slot
    '''The start of a sweep of the plain grade: the vector of ones that closes an end bond, as a constant
    polynomial on the full box (a scan carry has one shape at every site).'''
    corner = np.zeros(box)                                         # host: 1 in slot (0, ..., 0)
    corner[(0,) * len(box)] = 1.0
    slots_F = xnp.asarray(corner, dtype=dtype).reshape(box + (1,) * (len(W) + len(KC) + 1))
    return slots_F * xnp.ones(W + KC + (size,), dtype=dtype)


def start_zeros(
        box:    typ.Tuple[int, ...],        # F
        W:      typ.Tuple[int, ...],
        KC:     typ.Tuple[int, ...],        # K + C for a row that carries K, else C
        size:   int,
        dtype:  typ.Any,
        xnp,
) -> NDArray:                               # F+W+[K]+C+(size,), all zero
    '''The start of a sweep whose variable is zero at its end: a tilde sweep of the probe kind, a dot sweep (the
    direction is at one of the sites the carry has passed).'''
    return xnp.zeros(box + W + KC + (size,), dtype=dtype)


def start_node(
        node:   NDArray,                    # a whole variable of the graph: the apply kind's seed, F+W+[K]+C
        size:   int,                        # the end bond, in the basis of the sweep's own store
        xnp,
) -> NDArray:                               # F+W+[K]+C+(size,)
    '''The start of a sweep that names a node: the node times the vector of ones that closes the end bond. The apply
    kind's seed sits there, so its right sides start from it.'''
    return node[..., None] * xnp.ones((size,), dtype=node.dtype)
