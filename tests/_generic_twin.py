# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Test-only: the generic twin of the sampling engine, a second check of the written-out terms.

The prototype ``dev/graded_messages/scripts/graded_messages_prototype.py`` brought to the decided names and forms:
the whole probe kind as ONE message-passing program over a graded algebra. A message is a dict ``{grade: jet}``, the
grade ``''`` (plain) or ``'z'`` (the seed put in once: the tilde grade). A core is a ``PointedCore``: which plain core
it is depends on where the hole lies (the gauge of the frame). Seven leg rules, multiplied as truncated polynomials in
any number of jet variables (``order = (n_1, ..., n_m)``), make every variable; a block is the coefficient at the
corner of its product.

It shares no code with the engine: only ``numpy`` and ``itertools`` are imported (a test walks its source), the
truncated product is explicit loops over the multi-indices of the box, and the seed is made here (the residual
weighted, squared, then every jet axis flipped). Every jet is lifted to the full box ``F`` (an input of fewer slots
padded with zeros, one of more cut), so ``V_tilde`` is read from the FULL ``eta_tilde``: the long way round the
engine's short form.

The sample stack ``W`` is one axis here; ``twin`` flattens it on the way in and restores it on the way out.
"""
import itertools

import numpy as np

UP, RIGHT, LEFT, DOWN, OUT = 'io,o->i', 'a,aib,i->b', 'aib,i,b->a', 'a,aib,b->i', 'i,io->o'
GRAD_G, GRAD_B = 'a,i,b->aib', 'i,o->io'


class PointedCore:
    """A core whose plain part depends on where the hole is: ``toward[port]`` is the plain core used when the hole
    lies beyond that port of it (no hole: beyond the output port)."""

    def __init__(self, toward):
        self.toward = toward


def gmul(rule, ops, box, want=None):
    """The graded multilinear rule: ``rule`` an einsum over the legs only. Returns ``{grade: jet}`` for every grade
    the operands reach (each defect used at most once), the jets multiplied as truncated polynomials on ``box``."""
    ins, out = rule.split('->')
    ins = ins.split(',')
    is_msg = [isinstance(o, dict) for o in ops]
    msgs = [k for k in range(len(ops)) if is_msg[k]]
    subs = ','.join(('w' + s) if is_msg[k] else s for k, s in enumerate(ins)) + '->w' + out
    slots = list(itertools.product(*[range(f) for f in box]))
    result = {}
    for grades in itertools.product(*[list(o.keys()) if m else [''] for o, m in zip(ops, is_msg)]):
        total = ''.join(sorted(''.join(grades)))
        if len(total) != len(set(total)) or (want is not None and total not in want):
            continue
        arrays = []
        for k, (o, g) in enumerate(zip(ops, grades)):
            if is_msg[k]:
                arrays.append(o[g])
            else:
                port = [c for c in out if c in ins[k]][0]           # no dot grade here: the hole is beyond the output
                arrays.append(o.toward[port])
        for ks in itertools.product(slots, repeat=len(msgs)):
            t = tuple(sum(k[j] for k in ks) for j in range(len(box)))
            if any(a >= f for a, f in zip(t, box)):
                continue
            sliced = [arrays[k][ks[msgs.index(k)]] if is_msg[k] else arrays[k] for k in range(len(ops))]
            value = np.einsum(subs, *sliced)
            if total not in result:
                result[total] = np.zeros(box + value.shape)
            result[total][t] += value
    return result


def lift(jet, box):
    """A jet on the full box: cut to it, and padded with zeros up to it."""
    m = len(box)
    jet = np.asarray(jet)[tuple(slice(0, f) for f in box)]
    return np.pad(jet, [(0, f - s) for f, s in zip(box, jet.shape[:m])] + [(0, 0)] * (jet.ndim - m))


def ones_jet(box, W, size):
    out = np.zeros(box + (W, size))
    out[(0,) * len(box)] = 1.0
    return out


def run(B, G, w, left, right, box):
    """The program: the leaves, the two sweeps, the trunks, the outputs (where the Tucker cores have a variant for
    the mode edge), the blocks."""
    d = len(G)
    X = [gmul(UP, [B[i], w[i]], box) for i in range(d)]
    M = [left]
    for i in range(d):
        M.append(gmul(RIGHT, [M[i], G[i], X[i]], box))
    N = [None] * d + [right]
    for i in reversed(range(d)):
        N[i] = gmul(LEFT, [G[i], X[i], N[i + 1]], box)
    E = [gmul(DOWN, [M[i], G[i], N[i + 1]], box) for i in range(d)]
    Z = [gmul(OUT, [E[i], B[i]], box, ('',)) if 'o' in B[i].toward else None for i in range(d)]
    dG = [gmul(GRAD_G, [M[i], X[i], N[i + 1]], box, ('z',)) for i in range(d)]
    dB = [gmul(GRAD_B, [E[i], w[i]], box, ('z',)) for i in range(d)]
    return dict(X=X, M=M, N=N, E=E, Z=Z, dG=dG, dB=dB)


def twin(frame, ww, order, y, weight):
    """The probe kind's variables at a manifold frame ``(D, U, L, R)`` and input jets ``ww`` (``F_in_i + W +
    (N_i,)``), the jet shape ``order``, the data ``y`` (``F + W + (N_i,)`` per mode) and the weight ``(d,) + F``.

    Returns a dict of tuples over the index (the engine's shapes): ``z``, ``r``, ``w_tilde``, ``xi``, ``mu``, ``nu``,
    ``eta``, ``xi_tilde``, ``mu_tilde``, ``nu_tilde``, ``eta_tilde`` (in full), ``V_tilde``, ``H_tilde``."""
    D, U, L, R = frame
    d, m = len(D), len(order)
    box = tuple(n + 1 for n in order)
    corner = tuple(order)
    W = ww[0].shape[m:-1]
    nW = int(np.prod(W))
    flat = lambda x, lead: x.reshape(x.shape[:lead] + (nW,) + x.shape[lead + len(W):])
    unflat = lambda x, lead: x.reshape(x.shape[:lead] + W + x.shape[lead + 1:])
    inputs = [lift(flat(np.asarray(w), m), box) for w in ww]

    # the value at the base point (D, L): the corewise frame (D, L, L, L), the Tucker core open on its mode edge
    Bz = [PointedCore({'i': D[i], 'o': D[i]}) for i in range(d)]
    Gz = [PointedCore({'a': L[i], 'b': L[i], 'i': L[i]}) for i in range(d)]
    plain = [{'': inputs[i]} for i in range(d)]
    out = run(Bz, Gz, plain, {'': ones_jet(box, nW, L[0].shape[0])}, {'': ones_jet(box, nW, L[-1].shape[-1])}, box)
    z = [out['Z'][i][''] for i in range(d)]
    r = [z[i] - flat(np.asarray(y[i]), m) for i in range(d)]
    flip = (slice(None, None, -1),) * m
    w_tilde = [((np.asarray(weight[i]) ** 2).reshape(box + (1, 1)) * r[i])[flip] for i in range(d)]

    # the residual's adjoint at the manifold frame: the seed beside each input
    B = [PointedCore({'i': D[i]}) for i in range(d)]                # (no Tucker core for the mode edge: no Z)
    G = [PointedCore({'a': R[i], 'b': L[i], 'i': U[i]}) for i in range(d)]
    seeded = [{'': inputs[i], 'z': w_tilde[i]} for i in range(d)]
    out = run(B, G, seeded, {'': ones_jet(box, nW, L[0].shape[0])}, {'': ones_jet(box, nW, R[-1].shape[-1])}, box)
    zero = lambda x: x if 'z' in x else dict(x, z=np.zeros_like(x['']))
    M, N = [zero(x) for x in out['M']], [zero(x) for x in out['N']]
    V_tilde = [out['dB'][i]['z'][corner].sum(axis=0) for i in range(d)]
    H_tilde = [out['dG'][i]['z'][corner].sum(axis=0) for i in range(d)]
    as_engine = lambda xs: tuple(unflat(x, m) for x in xs)
    return dict(
        z=as_engine(z), r=as_engine(r), w_tilde=as_engine(w_tilde),
        xi=as_engine(x[''] for x in out['X']), mu=as_engine(x[''] for x in M), nu=as_engine(x[''] for x in N),
        eta=as_engine(x[''] for x in out['E']),
        xi_tilde=as_engine(x['z'] for x in out['X']), mu_tilde=as_engine(x['z'] for x in M),
        nu_tilde=as_engine(x['z'] for x in N), eta_tilde=as_engine(x['z'] for x in out['E']),
        V_tilde=tuple(V_tilde), H_tilde=tuple(H_tilde))
