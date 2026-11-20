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
box. (The start of the other grades, zero, comes with them.)

The **whole steps** (``r_step``, ``f_step``, ``r_block_sumsq_step``) act on a whole variable, one array per mode
for the probe kind, and take the same statics.

Every name here is private: the module is the engine's, and its ``__all__`` is empty.
"""
import typing as typ

import numpy as np

import t3toolbox.backend.contractions as contractions
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
########    The whole steps    ################
###############################################

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
