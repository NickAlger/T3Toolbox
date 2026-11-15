"""Prototype: the whole T3 sampling layer as ONE message-passing program over a graded algebra.

This is a reconnaissance sketch, NOT library code. It is deliberately slow and unoptimized (python
loops over jet orders, no W/K/C stack groups beyond a single sample axis W, numpy only). Its purpose
is to check one claim numerically against T3Toolbox itself:

    every sampling operation in entries.py / apply.py / probing.py / sampling_derivatives.py is a
    coefficient of the SAME seven site rules, evaluated on operands that carry "grades".

Grades (a subset of {'t', 'z'}, plus a Taylor-jet axis):
    ''    plain                         (xi, mu, nu, eta)
    't'   one theta-variation           (dxi, sigma, tau, deta      -- the "dot" quantities)
    'z'   one residual insertion        (deta~, tau~, sigma~, dxi~  -- the "tilde"/"hat" quantities)
    'tz'  both                          (the "tilde-dot" two-hole quantities)
    jet   leading axis of length order+1, TAYLOR-normalized (coefficient of s^t, not the t-th derivative).
          Residual jets are stored REVERSED, so that the adjoint (binomial correlation) becomes the
          ordinary truncated product and <z~, y> is the top (s^order) coefficient.

Not covered here (boundary/driver concerns, not new mathematics): the K and C stacks, sum_over_probes=False,
the uniform (supercore) layout, the ambient (CP) transposes, memory-lean order scans and W-chunking.

Usage:  python3 graded_messages_prototype.py /path/to/T3Toolbox
"""
import itertools
import math
import sys

import numpy as np


# ------------------------------------------------------------------------------------------------
# 1. Graded operands
# ------------------------------------------------------------------------------------------------
# A MESSAGE is a dict {grade: array}, array.shape = (P, W, leg)  [P = order + 1 Taylor coefficients].
# A CORE is a PointedCore: its plain part depends on WHERE THE HOLE IS relative to this core
# (the gauge of the frame); its 't' part is the variation core.

class PointedCore:
    def __init__(self, toward, dot=None):
        self.toward = toward     # {port letter: plain core used when the hole lies beyond that port}
        self.dot = dot           # the variation core (H or V), or None

    def grades(self):
        return ['', 't'] if self.dot is not None else ['']


def gmul(rule, ops, P, want=None):
    """Graded multilinear rule. `rule` is an einsum over core legs only, e.g. 'a,aib,i->b'.
    Returns {grade: array} for every grade reachable from the operands' grades (product rule:
    one term per way of distributing the output grade over the factors; jets: truncated Cauchy product)."""
    ins, out = rule.split('->')
    ins = ins.split(',')
    is_msg = [isinstance(o, dict) for o in ops]
    msgs = [k for k in range(len(ops)) if is_msg[k]]
    subs = ','.join(('w' + s) if is_msg[k] else s for k, s in enumerate(ins)) + '->w' + out
    result = {}
    for gs in itertools.product(*[list(o.keys()) if m else o.grades() for o, m in zip(ops, is_msg)]):
        if sum(len(g) for g in gs) != len(set(''.join(gs))):     # each defect may be used at most once
            continue
        total = ''.join(sorted(''.join(gs)))
        if want is not None and total not in want:                # only the requested coefficients
            continue
        arrays = []
        for k, (o, g) in enumerate(zip(ops, gs)):
            if is_msg[k]:
                arrays.append(o[g])
            elif 't' in g:
                arrays.append(o.dot)                              # the hole is this core
            else:                                                 # which way is the hole?
                port = [ins[k2] for k2 in msgs if 't' in gs[k2]]  # beyond the port a 't' message came in on,
                port = port[0] if port else [c for c in out if c in ins[k]][0]   # else beyond the output port
                arrays.append(o.toward.get(port))                # None: this frame has no such variant
        if any(a is None for a in arrays):                        # (e.g. a manifold frame has no plain probe)
            continue
        for orders in itertools.product(range(P), repeat=len(msgs)):   # truncated product of Taylor jets
            t = sum(orders)
            if t >= P:
                continue
            sl = [arrays[k][orders[msgs.index(k)]] if is_msg[k] else arrays[k] for k in range(len(ops))]
            if any(not np.any(x) for x in sl):                    # skip structurally-zero jet coefficients
                continue
            val = np.einsum(subs, *sl)
            if total not in result:
                result[total] = np.zeros((P,) + val.shape, val.dtype)
            result[total][t] += val
    return result


# ------------------------------------------------------------------------------------------------
# 2. The program: seven site rules + closure. This is the ENTIRE tensor-network content.
# ------------------------------------------------------------------------------------------------
UP, RIGHT, LEFT, DOWN, OUT = 'io,o->i', 'a,aib,i->b', 'aib,i,b->a', 'a,aib,b->i', 'i,io->o'
GRAD_G, GRAD_B, CLOSE = 'a,i,b->aib', 'i,o->io', 'a,a->'


def run(B, G, w, left, right, P, grad=('z', 'tz')):
    """B, G: lists of PointedCore. w: list of leaf messages. left/right: boundary messages."""
    d = len(G)
    X = [gmul(UP, [B[i], w[i]], P) for i in range(d)]                    # xi family   (mode edge, up)
    M = [left]                                                            # mu family   (bond, rightward)
    for i in range(d):
        M.append(gmul(RIGHT, [M[i], G[i], X[i]], P))
    N = [None] * d + [right]                                              # nu family   (bond, leftward)
    for i in reversed(range(d)):
        N[i] = gmul(LEFT, [G[i], X[i], N[i + 1]], P)
    E = [gmul(DOWN, [M[i], G[i], N[i + 1]], P) for i in range(d)]         # eta family  (mode edge, down)
    Z = [gmul(OUT, [E[i], B[i]], P) for i in range(d)]                    # probes      (ambient edge, out)
    dG = [gmul(GRAD_G, [M[i], X[i], N[i + 1]], P, grad) for i in range(d)]      # TT core gradients
    dB = [gmul(GRAD_B, [E[i], w[i]], P, grad) for i in range(d)]                # Tucker core gradients
    Phi = gmul(CLOSE, [M[d], N[d]], P)                                    # apply
    return dict(X=X, M=M, N=N, E=E, Z=Z, dG=dG, dB=dB, Phi=Phi)


# ------------------------------------------------------------------------------------------------
# 3. Boundary data (the ONLY thing that distinguishes entries / apply / probe / jets / transposes)
# ------------------------------------------------------------------------------------------------
def jet(P, *coeffs):
    """Taylor jet (P, W, n) from its leading coefficients; the rest are zero."""
    out = np.zeros((P,) + coeffs[0].shape, coeffs[0].dtype)
    for t, c in enumerate(coeffs[:P]):
        out[t] = c
    return out


def dual_jet(P, r):
    """Residual jet r (derivative-normalized, shape (P, W, ...)) -> reversed, Taylor-weighted:
    zeta[P-1-t] = t! r[t], so that  sum_t <r[t], d^t y/ds^t>  =  [s^(P-1)] zeta(s) * yhat(s)."""
    fact = np.array([math.factorial(t) for t in range(P)]).reshape((P,) + (1,) * (r.ndim - 1))
    return (fact * r)[::-1]


def ones_boundary(P, W):
    return {'': jet(P, np.ones((W, 1)))}


def manifold(frame, variation=None, plain_out=False):
    D, U, L, R = frame
    V, H = variation if variation is not None else ([None] * len(D), [None] * len(D))
    # A manifold frame has no Tucker variant for "hole beyond the ambient leg" (no plain probe output);
    # the corewise frame does (it is just B).
    B = [PointedCore({'i': D[i], 'o': D[i]} if plain_out else {'i': D[i]}, V[i]) for i in range(len(D))]
    G = [PointedCore({'a': R[i], 'b': L[i], 'i': U[i]}, H[i]) for i in range(len(D))]
    return B, G


def corewise(core_pair, directions=None):
    Bc, Gc = core_pair
    return manifold((Bc, Gc, Gc, Gc), directions, plain_out=True)


def derivs(x):
    """Taylor coefficients (P, ...) -> derivatives."""
    P = x.shape[0]
    return x * np.array([math.factorial(t) for t in range(P)]).reshape((P,) + (1,) * (x.ndim - 1))


# ------------------------------------------------------------------------------------------------
# 4. Check against T3Toolbox
# ------------------------------------------------------------------------------------------------
def main(repo):
    sys.path.append(repo)
    import t3toolbox.backend.probing as probing
    import t3toolbox.backend.apply as apply
    import t3toolbox.backend.entries as entries
    import t3toolbox.backend.sampling_derivatives as sd

    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
    rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
    d, W = len(Ns), 5

    D = [rn((nD[i], Ns[i])) for i in range(d)]
    U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
    L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]
    R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
    V = [rn((nU[i], Ns[i])) for i in range(d)]
    H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    frame, variation = (D, U, L, R), (V, H)
    Bc, Gc = D, L                                         # a plain T3 (for the corewise ops)
    Bd = [rn(b.shape) for b in Bc]
    Gd = [rn(g.shape) for g in Gc]
    ww = [rn((W, N)) for N in Ns]
    pp = [rn((W, N)) for N in Ns]
    index = np.stack([rng.integers(0, N, size=W) for N in Ns])
    onehot = [np.eye(N)[index[i]] for i, N in enumerate(Ns)]

    worst = [0.0]

    def check(name, mine, theirs):
        mine = mine if isinstance(mine, (list, tuple)) else [mine]
        theirs = theirs if isinstance(theirs, (list, tuple)) else [theirs]
        err = max(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(np.asarray(b)), 1e-300)
                  for a, b in zip(mine, theirs))
        worst[0] = max(worst[0], err)
        print('  %-58s rel err %.1e' % (name, err))

    top = lambda xs, g: [x[g][-1].sum(axis=0) for x in xs]          # top jet coefficient, W summed

    for order in (0, 3):
        P = order + 1
        print('order = %d' % order)
        for kind, vecs in (('apply/probe', ww), ('entries', onehot)):
            leaf = lambda i: jet(P, vecs[i], pp[i])                   # w + s p
            plain = [{'': leaf(i)} for i in range(d)]
            bnd = ones_boundary(P, W)

            # --- forward, plain T3:  grade ''
            out = run(*corewise((Bc, Gc)), plain, bnd, bnd, P)
            if kind == 'entries':
                check('t3_entries_derivatives', derivs(out['Phi']['']),
                      sd.t3_entries_derivatives(index, pp, (Bc, Gc), order))
            else:
                check('t3_probe_derivatives', [derivs(z['']) for z in out['Z']],
                      sd.t3_probe_derivatives(ww, pp, (Bc, Gc), order))
                check('t3_apply_derivatives', derivs(out['Phi']['']),
                      sd.t3_apply_derivatives(ww, pp, (Bc, Gc), order))
                if order == 0:
                    check('t3_probe', [z[''][0] for z in out['Z']], probing.t3_probe(ww, (Bc, Gc)))
                    check('t3_apply', out['Phi'][''][0], apply.t3_apply((Bc, Gc), ww))

            # --- forward, tangent (the Jacobian):  grade 't'
            out = run(*manifold(frame, variation), plain, bnd, bnd, P)
            if kind == 'entries':
                check('tv_entries_derivatives', derivs(out['Phi']['t']),
                      sd.tv_entries_derivatives(index, pp, variation, frame, order))
                if order == 0:
                    check('tv_entries', out['Phi']['t'][0], entries.tv_entries(index, variation, frame))
            else:
                check('tv_probe_derivatives', [derivs(z['t']) for z in out['Z']],
                      sd.tv_probe_derivatives(ww, pp, variation, frame, order))
                check('tv_apply_derivatives', derivs(out['Phi']['t']),
                      sd.tv_apply_derivatives(ww, pp, variation, frame, order))
                if order == 0:
                    check('tv_probe', [z['t'][0] for z in out['Z']], probing.tv_probe(ww, variation, frame))
                    check('tv_apply', out['Phi']['t'][0], apply.tv_apply(ww, variation, frame))

            # --- transposes with a SCALAR residual (apply / entries):  the residual seeds the closure
            c = rn((P, W))
            seeded = {'': bnd[''], 'z': dual_jet(P, c)[..., None]}
            out = run(*manifold(frame), plain, bnd, seeded, P)
            mine = (top(out['dB'], 'z'), top(out['dG'], 'z'))
            if kind == 'entries':
                theirs = sd.tv_entries_derivatives_transpose(c, index, pp, frame, order, sum_over_probes=True)
                check('tv_entries_derivatives_transpose', mine[0] + mine[1], list(theirs[0]) + list(theirs[1]))
                if order == 0:
                    theirs = entries.tv_entries_transpose(c[0], index, frame, sum_over_probes=True)
                    check('tv_entries_transpose', mine[0] + mine[1], list(theirs[0]) + list(theirs[1]))
            else:
                theirs = sd.tv_apply_derivatives_transpose(c, ww, pp, frame, order, sum_over_probes=True)
                check('tv_apply_derivatives_transpose', mine[0] + mine[1], list(theirs[0]) + list(theirs[1]))
                if order == 0:
                    theirs = apply.tv_apply_transpose(c[0], ww, frame, sum_over_probes=True)
                    check('tv_apply_transpose', mine[0] + mine[1], list(theirs[0]) + list(theirs[1]))

            # --- exact-Hessian sampling term, scalar residual:  grade 'tz', corewise frame
            out = run(*corewise((Bc, Gc), (Bd, Gd)), plain, bnd, seeded, P)
            mine = top(out['dB'], 'tz') + top(out['dG'], 'tz')
            if kind == 'entries':
                theirs = sd.t3_entries_corewise_derivatives_hessian(c, index, pp, (Bc, Gc), (Bd, Gd), order)
                check('t3_entries_corewise_derivatives_hessian', mine, list(theirs[0]) + list(theirs[1]))
            else:
                theirs = sd.t3_apply_corewise_derivatives_hessian(c, ww, pp, (Bc, Gc), (Bd, Gd), order)
                check('t3_apply_corewise_derivatives_hessian', mine, list(theirs[0]) + list(theirs[1]))
                if order == 0:
                    theirs = apply.t3_apply_corewise_hessian(c[0], ww, (Bc, Gc), (Bd, Gd))
                    check('t3_apply_corewise_hessian', mine, list(theirs[0]) + list(theirs[1]))

            if kind == 'entries':
                continue

            # --- transposes with a VECTOR residual (probe):  the residual sits on the leaves
            zt = [rn((P, W, N)) for N in Ns]
            resid = [{'': leaf(i), 'z': dual_jet(P, zt[i])} for i in range(d)]
            out = run(*manifold(frame), resid, bnd, bnd, P)
            mine = top(out['dB'], 'z') + top(out['dG'], 'z')
            theirs = sd.tv_probe_derivatives_transpose(zt, ww, pp, frame, order, sum_over_probes=True,
                                                       chunk_size=None)
            check('tv_probe_derivatives_transpose', mine, list(theirs[0]) + list(theirs[1]))
            if order == 0:
                theirs = probing.tv_probe_transpose([z[0] for z in zt], ww, frame, sum_over_probes=True)
                check('tv_probe_transpose', mine, list(theirs[0]) + list(theirs[1]))

            # --- exact-Hessian sampling term (the "two-hole"), probe:  grade 'tz', corewise frame
            out = run(*corewise((Bc, Gc), (Bd, Gd)), resid, bnd, bnd, P)
            mine = top(out['dB'], 'tz') + top(out['dG'], 'tz')
            theirs = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, (Bc, Gc), (Bd, Gd), order,
                                                              chunk_size=None)
            check('t3_probe_corewise_derivatives_hessian', mine, list(theirs[0]) + list(theirs[1]))
            if order == 0:
                theirs = probing.t3_probe_corewise_hessian([z[0] for z in zt], ww, (Bc, Gc), (Bd, Gd))
                check('t3_probe_corewise_hessian', mine, list(theirs[0]) + list(theirs[1]))

    print('worst relative error over all checks: %.1e' % worst[0])


if __name__ == '__main__':
    main(sys.argv[1])
