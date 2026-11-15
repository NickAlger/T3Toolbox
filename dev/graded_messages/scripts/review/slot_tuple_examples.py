"""Concrete examples for review finding F2: fixed-slot tuples in place of dicts and operand objects.

Not library code. It re-expresses the graded prototype (dicts keyed by grade, a PointedCore class) with
plain tuples, checks the two against each other and against T3Toolbox, and shows what `evaluate` looks
like with names and values as aligned tuples.

Usage:  python3 -I slot_tuple_examples.py /path/to/T3Toolbox /path/to/handoff/scripts
"""
import itertools
import sys

import numpy as np

sys.path.append(sys.argv[2])
import graded_messages_prototype as proto              # the dict version: gmul, run, manifold, corewise
import plan_mockup as mock                              # the dict version of evaluate

# ----------------------------------------------------------------------------------------------------
# 1. A graded message is a 4-tuple.  The slot index is a bitmask: bit 0 = tangent, bit 1 = residual.
# ----------------------------------------------------------------------------------------------------
PLAIN, T, Z, TZ = 0, 1, 2, 3


def message(plain=None, t=None, z=None, tz=None):
    return (plain, t, z, tz)


# ----------------------------------------------------------------------------------------------------
# 2. A pointed core is a tuple too: one plain variant per leg of the core, in the rule's leg order,
#    then the variation core.  For a TT core with legs 'aib' that is (R, U, L, H); for a Tucker core
#    with legs 'io' it is (D, D-or-None, V).
# ----------------------------------------------------------------------------------------------------
def manifold(frame, variation=None, plain_out=False):
    D, U, L, R = frame
    V, H = variation if variation is not None else ([None] * len(D), [None] * len(D))
    B = tuple((D[i], D[i] if plain_out else None, V[i]) for i in range(len(D)))
    G = tuple((R[i], U[i], L[i], H[i]) for i in range(len(D)))
    return B, G


def corewise(core_pair, directions=None):
    Bc, Gc = core_pair
    return manifold((Bc, Gc, Gc, Gc), directions, plain_out=True)


# ----------------------------------------------------------------------------------------------------
# 3. A rule states which operand is the core.  With dicts the code asked isinstance(operand, dict);
#    with plain tuples the role is part of the rule, which is static.
# ----------------------------------------------------------------------------------------------------
UP, RIGHT, LEFT = ('io,o->i', 0), ('a,aib,i->b', 1), ('aib,i,b->a', 0)
DOWN, OUT = ('a,aib,b->i', 1), ('i,io->o', 1)
GRAD_G, GRAD_B, CLOSE = ('a,i,b->aib', None), ('i,o->io', None), ('a,a->', None)


def gmul(rule, ops, P, want=None):
    """The graded rule on fixed-slot operands.  Returns a message (plain, t, z, tz)."""
    subscripts, core = rule
    ins, out = subscripts.split('->')
    ins = ins.split(',')
    msgs = [k for k in range(len(ops)) if k != core]
    subs = ','.join(s if k == core else 'w' + s for k, s in enumerate(ins)) + '->w' + out
    slots = [((PLAIN, T) if o[-1] is not None else (PLAIN,)) if k == core
             else [g for g in (PLAIN, T, Z, TZ) if o[g] is not None] for k, o in enumerate(ops)]
    result = [None, None, None, None]
    for gs in itertools.product(*slots):
        total = 0
        for g in gs:
            if total & g:                                         # a defect may be used at most once
                break
            total |= g
        else:
            if want is not None and total not in want:
                continue
            arrays = []
            for k, (o, g) in enumerate(zip(ops, gs)):
                if k != core:
                    arrays.append(o[g])
                elif g == T:
                    arrays.append(o[-1])                          # the hole is this core
                else:                                             # which way is the hole?
                    port = [ins[k2] for k2 in msgs if gs[k2] & T]
                    port = port[0] if port else [c for c in out if c in ins[k]][0]
                    arrays.append(o[ins[k].index(port)])
            if any(a is None for a in arrays):
                continue
            for orders in itertools.product(range(P), repeat=len(msgs)):
                t = sum(orders)
                if t >= P:
                    continue
                sl = [arrays[k] if k == core else arrays[k][orders[msgs.index(k)]] for k in range(len(ops))]
                if any(not np.any(x) for x in sl):
                    continue
                val = np.einsum(subs, *sl)
                if result[total] is None:
                    result[total] = np.zeros((P,) + val.shape, val.dtype)
                result[total][t] += val
    return tuple(result)


def run(B, G, w, left, right, P, grad=(Z, TZ)):
    d = len(G)
    X = tuple(gmul(UP, (B[i], w[i]), P) for i in range(d))
    M = [left]
    for i in range(d):
        M.append(gmul(RIGHT, (M[i], G[i], X[i]), P))
    N = [None] * d + [right]
    for i in reversed(range(d)):
        N[i] = gmul(LEFT, (G[i], X[i], N[i + 1]), P)
    E = tuple(gmul(DOWN, (M[i], G[i], N[i + 1]), P) for i in range(d))
    Zs = tuple(gmul(OUT, (E[i], B[i]), P) for i in range(d))
    dG = tuple(gmul(GRAD_G, (M[i], X[i], N[i + 1]), P, grad) for i in range(d))
    dB = tuple(gmul(GRAD_B, (E[i], w[i]), P, grad) for i in range(d))
    Phi = gmul(CLOSE, (M[d], N[d]), P)
    return X, tuple(M), tuple(N), E, Zs, dG, dB, Phi           # fixed order, fixed length


# ----------------------------------------------------------------------------------------------------
# 4. evaluate: names are a static tuple of strings, values a tuple aligned with it.  (An adapter over
#    the mockup's dict version, to show the call sites; the evaluator itself is not rewritten here.)
# ----------------------------------------------------------------------------------------------------
def evaluate(frame, centers, w, plan, want, have=(), have_values=(), y=None, p=None):
    out = mock.evaluate(frame, centers, w, plan, want, have=dict(zip(have, have_values)), y=y, p=p)
    return tuple(out[name] for name in want)


# ----------------------------------------------------------------------------------------------------
def main(repo):
    sys.path.append(repo)
    import t3toolbox.backend.sampling_derivatives as sd
    import t3toolbox.backend.common as common

    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
    rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
    d, W, order = len(Ns), 5, 2
    P = order + 1
    D = [rn((nD[i], Ns[i])) for i in range(d)]
    U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
    L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]
    R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
    V = [rn((nU[i], Ns[i])) for i in range(d)]
    H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    frame, variation = (D, U, L, R), (V, H)
    Bd, Gd = [rn(b.shape) for b in D], [rn(g.shape) for g in L]
    ww, pp = [rn((W, N)) for N in Ns], [rn((W, N)) for N in Ns]
    zt = [rn((P, W, N)) for N in Ns]
    leaf = lambda i: proto.jet(P, ww[i], pp[i])
    one = proto.jet(P, np.ones((W, 1)))

    # ---- the same program on dicts and on tuples ------------------------------------------------
    as_dict = lambda m: {k: m[s] for k, s in (('', PLAIN), ('t', T), ('z', Z), ('tz', TZ)) if m[s] is not None}
    worst = 0.0
    plain_d = [{'': leaf(i)} for i in range(d)]
    plain_t = tuple(message(plain=leaf(i)) for i in range(d))
    resid_d = [{'': leaf(i), 'z': proto.dual_jet(P, zt[i])} for i in range(d)]
    resid_t = tuple(message(plain=leaf(i), z=proto.dual_jet(P, zt[i])) for i in range(d))
    for name, cores_d, cores_t, leaves_d, leaves_t in (
            ('manifold frame, tangent grade', proto.manifold(frame, variation), manifold(frame, variation),
             plain_d, plain_t),
            ('manifold frame, residual grade', proto.manifold(frame), manifold(frame), resid_d, resid_t),
            ('corewise frame, two-hole', proto.corewise((D, L), (Bd, Gd)), corewise((D, L), (Bd, Gd)),
             resid_d, resid_t)):
        out_d = proto.run(*cores_d, leaves_d, {'': one}, {'': one}, P)
        out_t = run(*cores_t, leaves_t, message(plain=one), message(plain=one), P)
        diff = 0.0
        for key, fam in zip(('X', 'M', 'N', 'E', 'Z', 'dG', 'dB'), out_t):
            for md, mt in zip(out_d[key], fam):
                mt = as_dict(mt)
                assert set(mt) == set(md), (key, set(mt), set(md))
                diff = max([diff] + [float(np.linalg.norm(mt[g] - md[g]) / np.linalg.norm(md[g])) for g in md])
        worst = max(worst, diff)
        print('%-40s tuples vs dicts, every family and grade: max rel diff %.1e' % (name, diff))

    X, M, N, E, Zs, dG, dB, Phi = run(*corewise((D, L), (Bd, Gd)), resid_t, message(plain=one),
                                      message(plain=one), P)
    mine = [m[TZ][-1].sum(axis=0) for m in dB] + [m[TZ][-1].sum(axis=0) for m in dG]
    theirs = sd.t3_probe_corewise_derivatives_hessian(zt, ww, pp, (D, L), (Bd, Gd), order, chunk_size=None)
    theirs = list(theirs[0]) + list(theirs[1])
    err = max(np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(mine, theirs))
    print('%-40s tuples vs library two-hole (order %d): rel err %.1e' % ('', order, err))

    print('\nwhat the values look like')
    show = lambda m: '(' + ', '.join('None' if a is None else str(a.shape) for a in m) + ')'
    print('  xi at site 1, (plain, t, z, tz):   ', show(X[1]))
    print('  mu after site 1:                   ', show(M[2]))
    print('  TT core gradient at site 1:        ', show(dG[1]))
    print('  a TT pointed core (R, U, L, H):    ', show(corewise((D, L), (Bd, Gd))[1][1]))
    print('  a Tucker pointed core (D, D, V):   ', show(corewise((D, L), (Bd, Gd))[0][1]))

    dyn, skel = common.partition_static((X, (6, 5, 7, 4)))
    print('  static split of (xi family, shape): %d dynamic leaves, %d of them arrays; round trip ok: %s'
          % (len(dyn), sum(a is not None for a in dyn),
             common.rebuild_static(dyn, skel)[1] == (6, 5, 7, 4)))

    # ---- evaluate with aligned tuples -----------------------------------------------------------
    x, w, y, fr, centers, p, _, (f0, g0, Hp0) = mock.user_example()
    plan = mock.Plan(samples=512)

    f, = evaluate(fr, centers, w, plan, want=('f',), y=y)
    f, g, xi = evaluate(fr, centers, w, plan, want=('f', 'g', 'xi'), y=y)
    Hp, = evaluate(fr, centers, w, plan, want=('Hp',), have=('xi',), have_values=(xi,), p=p)
    r, = evaluate(fr, centers, w, plan, want=('r',), have=('xi',), have_values=(xi,), y=y)

    hold = ('xi', 'mu', 'nu', 'eta')                                 # a level's held set: names in the plan,
    sweep = evaluate(fr, centers, w, plan, want=hold)                # values in a tuple of the same length
    Hp2, = evaluate(fr, centers, w, plan, want=('Hp',), have=hold, have_values=sweep, p=p)

    flat = lambda v: np.concatenate([np.ravel(a) for part in v for a in part])
    rel = lambda a, b: np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b))
    print('\nevaluate with aligned tuples, against the dict version')
    print('  f   %.1e   g   %.1e   Hp (xi handed in)  %.1e   Hp (sweep handed in)  %.1e'
          % (abs(f - f0) / abs(f0), rel(g, g0), rel(Hp, Hp0), rel(Hp2, Hp0)))
    print('  held sweep: a tuple of %d components; r: a tuple of %d arrays, the first %s'
          % (len(sweep), len(r), r[0].shape))


if __name__ == '__main__':
    main(sys.argv[1])
