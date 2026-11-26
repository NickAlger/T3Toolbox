"""Reviewer 4's dense pins of the reference's weighted variables (K1 part 1), written before the built reference.

    PYTHONPATH=<tree> python -B ref_dense_pins.py --api old     # 78a34d77 and before: one weight (validates the model)
    PYTHONPATH=<tree> python -B ref_dense_pins.py --api new     # part 1 on: block_weight= and full_weight=

What is computed here, with no t3toolbox code (numpy only, explicit loops over every index):

- the dense tensor of a network (TT cores over Tucker cores, end bonds closed with ones) and the dense tangent tensor
  of a direction (V, H) in the frame's form (H[j] over D[j], or U[j] over V[j]; L before j, R after);
- z and Jp of the three kinds at input jets, by explicit sums over the slots of every mode (clipped to the box F,
  truncated to the jet shape), mode by mode;
- the metric M = b**2 f**2 entry by entry (b the block row of the mode read at the slot; f the full weight read at
  the entry, index 0 on a size-1 axis), and from it: w_tilde (turned), f, r_block_sumsq, Jp_norm_sq, g and JtJp. g
  and JtJp are whole: each entry of each block is the pairing of M r (or M Jp(p)) with the sampled dense tangent of
  that unit direction.

Then the reference (t3toolbox.backend.sampling_definitions, from PYTHONPATH) is compared with these, and:

- every seeded function (the 15 that read y and the weights through the seed) with (y, weights) equals the same
  function with w_tilde = THIS module's seed (so each function's own passing of the weights is pinned);
- the 16 functions that do not read a weight give the same result with and without weights;
- (--api new) the refusals of the note's rule, and a few informational probes (what the reference does with
  modes whose full weights disagree in their size-1 W pattern).

Sizes coincide, so a misplaced axis cannot fail by shape: d = 3 = n + 1 at order (2,), F = (3,), W = (3,), every
N_i = 3; at (2, 1), F = (3, 2) and W = (3, 2). One degenerate configuration (order (0,), a mode of size 1, W = (1,))
checks that "exact" means "z's size", which is 1 there. Every table is random (asymmetric on every axis). The old API gets
the product b f written out to z's shape (old weight = one array of the type of z, squared inside).
"""
import argparse
import itertools
import sys

import numpy as np

import t3toolbox.backend.sampling_definitions as sdef

TOL = 1e-10
FAILS = []
COUNT = [0]


def report(label, ok, detail=''):
    COUNT[0] += 1
    if not ok:
        FAILS.append(label)
        print('FAIL', label, detail)


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.inf
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0))


def at(a, idx):
    """a at idx, index 0 on every size-1 axis of a (the meaning of a 1 on a stack axis), by hand."""
    a = np.asarray(a)
    return a[tuple(0 if s == 1 else j for s, j in zip(a.shape, idx))]


# --------------------------------------------------------------------------------------------------------------------
# the dense model
# --------------------------------------------------------------------------------------------------------------------

def dense(tt, tucker):
    """Every TT core over its Tucker core, the end bonds closed with ones."""
    X = np.ones(tt[0].shape[0])
    for G, B in zip(tt, tucker):
        core = np.zeros((G.shape[0], B.shape[1], G.shape[2]))
        for a, n, b, x in itertools.product(range(G.shape[0]), range(G.shape[1]), range(G.shape[2]),
                                            range(B.shape[1])):
            core[a, x, b] += G[a, n, b] * B[n, x]
        X = np.tensordot(X, core, axes=([-1], [0]))
    return np.tensordot(X, np.ones(tt[-1].shape[2]), axes=([-1], [0]))


def contract(T, vecs, open_mode):
    """T contracted with vecs[j] in every mode j != open_mode (from the last mode down, so lower axes stay)."""
    X = T
    for j in reversed(range(T.ndim)):
        if j != open_mode:
            X = np.tensordot(X, vecs[j], axes=([j], [0]))
    return X


def sample(kind, T, ww, order, index):
    """z of a dense tensor, of the type of z (probe: a list of F+W+(Ni,); apply F+W; entries W)."""
    d = T.ndim
    if kind == 'entries':
        W = index.shape[1:]
        z = np.zeros(W)
        for w in np.ndindex(*W):
            z[w] = T[tuple(index[(j,) + w] for j in range(d))]
        return z
    m = len(order)
    F = tuple(n + 1 for n in order)
    W = ww[0].shape[m:-1]
    slots = [[k for k in np.ndindex(*w.shape[:m]) if all(a < b for a, b in zip(k, F))] for w in ww]
    opens = range(d) if kind == 'probe' else [None]
    out = []
    for i in opens:
        shape = F + W + ((T.shape[i],) if i is not None else ())
        z = np.zeros(shape)
        others = [j for j in range(d) if j != i]
        for w in np.ndindex(*W):
            for combo in itertools.product(*(slots[j] for j in others)):
                k = tuple(sum(c[a] for c in combo) for a in range(m))
                if any(a > n for a, n in zip(k, order)):
                    continue
                vecs = [None] * d
                for j, kj in zip(others, combo):
                    vecs[j] = ww[j][kj + w]
                z[k + w] += contract(T, vecs, i)
        out.append(z)
    return out if kind == 'probe' else out[0]


def tangent(frame, V, H):
    D, U, L, R = frame
    d = len(D)
    T = 0.0
    for j in range(d):
        T = T + dense(L[:j] + [H[j]] + R[j + 1:], D)
        T = T + dense(L[:j] + [U[j]] + R[j + 1:], D[:j] + [V[j]] + D[j + 1:])
    return T


def unit_tangents(frame):
    """The dense tangent of every unit direction: [('V', i, idx, T), ...] then H."""
    D, U, L, R = frame
    d = len(D)
    out = []
    for i in range(d):
        for idx in np.ndindex(U[i].shape[1], D[i].shape[1]):
            E = np.zeros((U[i].shape[1], D[i].shape[1]))
            E[idx] = 1.0
            out.append(('V', i, idx, dense(L[:i] + [U[i]] + R[i + 1:], D[:i] + [E] + D[i + 1:])))
    for i in range(d):
        for idx in np.ndindex(L[i].shape[0], D[i].shape[0], R[i].shape[2]):
            E = np.zeros((L[i].shape[0], D[i].shape[0], R[i].shape[2]))
            E[idx] = 1.0
            out.append(('H', i, idx, dense(L[:i] + [E] + R[i + 1:], D)))
    return out


def modes(kind, v):
    return list(v) if kind == 'probe' else [v]


def metric(kind, z_like, m, b, f):
    """M = b**2 f**2 entry by entry, of the type of z (a list over the modes here)."""
    out = []
    for i, zi in enumerate(modes(kind, z_like)):
        Mi = np.zeros(zi.shape)
        for idx in np.ndindex(*zi.shape):
            bb = 1.0 if b is None else (float(np.asarray(b)[(i,) + idx[:m]]) if kind == 'probe' else
                                        float(np.asarray(b)[idx[:m]]))
            ff = 1.0 if f is None else float(at(modes(kind, f)[i], idx))
            Mi[idx] = bb ** 2 * ff ** 2
        out.append(Mi)
    return out


def brute(kind, z, y, Jp_p, units_sampled, order, b, f):
    m = len(order)
    zs, ys, Jps = modes(kind, z), modes(kind, y), modes(kind, Jp_p)
    Ms = metric(kind, z, m, b, f)
    rs = [zi - yi for zi, yi in zip(zs, ys)]
    out = {}
    seeds = []
    for Mi, ri in zip(Ms, rs):
        si = np.zeros(ri.shape)
        for idx in np.ndindex(*ri.shape):
            turned = tuple(n - k for n, k in zip(order, idx[:m])) + idx[m:]
            si[idx] = Mi[turned] * ri[turned]
        seeds.append(si)
    out['w_tilde'] = seeds
    out['f'] = 0.5 * sum(float(np.sum(Mi * ri * ri)) for Mi, ri in zip(Ms, rs))
    F = tuple(n + 1 for n in order)
    blocks = np.zeros((len(rs),) + F)
    for i, (Mi, ri) in enumerate(zip(Ms, rs)):
        for k in np.ndindex(*F):
            blocks[(i,) + k] = float(np.sum(Mi[k] * ri[k] ** 2))
    out['r_block_sumsq'] = blocks if kind == 'probe' else blocks[0]
    out['Jp_norm_sq'] = sum(float(np.sum(Mi * Ji * Ji)) for Mi, Ji in zip(Ms, Jps))
    g, JtJp = {}, {}
    for (fam, i, idx, Ze) in units_sampled:
        Zs = modes(kind, Ze)
        g[(fam, i, idx)] = sum(float(np.sum(Mi * ri * Zi)) for Mi, ri, Zi in zip(Ms, rs, Zs))
        JtJp[(fam, i, idx)] = sum(float(np.sum(Mi * Ji * Zi)) for Mi, Ji, Zi in zip(Ms, Jps, Zs))
    out['g'], out['JtJp'] = g, JtJp
    return out


def blocks_of(result, keys_like):
    """The reference's (V_blocks, H_blocks) as a dict keyed like `brute`'s g."""
    V, H = result
    out = {}
    for fam, i, idx, _ in keys_like:
        out[(fam, i, idx)] = float((V if fam == 'V' else H)[i][idx])
    return out


# --------------------------------------------------------------------------------------------------------------------
# the problems and the weights
# --------------------------------------------------------------------------------------------------------------------

def problem(seed=0, N=(3, 3, 3)):
    rng = np.random.default_rng(seed)
    d = 3
    nD, nU = (2, 3, 2), (3, 2, 2)
    rL, rR = (2, 2, 3, 1), (1, 3, 2, 2)                   # tails on both ends: the ones matter
    D = [rng.standard_normal((nD[i], N[i])) for i in range(d)]
    U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(d)]
    L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(d)]
    R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(d)]
    V = [rng.standard_normal((nU[i], N[i])) for i in range(d)]
    H = [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    Dd = [rng.standard_normal(B.shape) for B in D]
    Ld = [rng.standard_normal(G.shape) for G in L]
    return dict(rng=rng, d=d, N=N, frame=(D, U, L, R), V=V, H=H, cd=(Dd, Ld))


# (order, N, W, F_in): the coinciding sizes, then the degenerate ones where z itself has axes of size 1 (an order
# (0,) has F = (1,), a mode of size 1 has an edge of 1, W = (1,)): there "exact" is 1, and a weight of z's own shape
# must be accepted (a rule written as "no 1 on a structural axis" would refuse it).
CONFIGS = [
    ((), (3, 3, 3), (3,), [(), (), ()]),
    ((2,), (3, 3, 3), (3,), [(2,), (3,), (4,)]),               # slot 3 of mode 2 is beyond the box: never read
    ((2, 1), (3, 3, 3), (3, 2), [(2, 2), (3, 1), (1, 3)]),     # F = (3, 2) = W; mode 2's slot 2 on axis 1 is beyond
    ((0,), (3, 1, 2), (1,), [(1,), (2,), (1,)]),               # degenerate: F = (1,), N_1 = 1, W = (1,)
]


def weight_cases(kind, order, d, W, zshapes, rng):
    """name -> (b, f), f of the type of z with each W axis 1 or full, every other axis exact."""
    m = len(order)
    F = tuple(n + 1 for n in order)
    bshape = (d,) + F if kind == 'probe' else F
    b = 0.5 + rng.random(bshape)

    def full(w_one):
        out = [0.5 + rng.random(tuple(1 if (m <= a < m + len(W) and (a - m) in w_one) else n
                                      for a, n in enumerate(sh))) for sh in zshapes]
        return out if kind == 'probe' else out[0]
    cases = {'none': (None, None), 'block': (b, None), 'full': (None, full(set())),
             'full W of 1': (None, full(set(range(len(W))))), 'both': (b, full(set())),
             'both, W of 1': (b, full(set(range(len(W)))))}
    if len(W) > 1:
        cases['full, one W axis of 1'] = (None, full({0}))
        cases['both, one W axis of 1'] = (b, full({1}))
    if kind == 'entries':
        cases['block, a Python float'] = (float(0.5 + rng.random()), None)
        cases['both, a Python float'] = (float(0.5 + rng.random()), full(set()))
    return cases


def old_weight(kind, zshapes, m, b, f):
    """The product b f written out to z's shape by loops (the one weight of the old API)."""
    if b is None and f is None:
        return None
    out = []
    for i, sh in enumerate(zshapes):
        w = np.zeros(sh)
        for idx in np.ndindex(*sh):
            bb = 1.0 if b is None else (float(np.asarray(b)[(i,) + idx[:m]]) if kind == 'probe' else
                                        float(np.asarray(b)[idx[:m]]))
            ff = 1.0 if f is None else float(at(modes(kind, f)[i], idx))
            w[idx] = bb * ff
        out.append(w)
    return out if kind == 'probe' else out[0]


SEEDED = ('xi_tilde', 'mu_tilde', 'nu_tilde', 'eta_tilde', 'nu_L_tilde', 'eta_D_tilde', 'xi_tilde_dot', 'mu_tilde_dot',
          'nu_L_tilde_dot', 'eta_D_tilde_dot', 'g', 'Kp', 'V_tilde', 'H_tilde', 'V_tilde_dot', 'H_tilde_dot')
UNREAD = ('w', 'xi', 'mu', 'nu', 'eta', 'nu_L', 'eta_D', 'xi_U_dot', 'mu_R_dot', 'nu_L_dot', 'eta_D_dot', 'xi_dot',
          'mu_dot', 'z', 'r', 'Jp')


def leaves(x):
    if isinstance(x, (tuple, list)):
        return [a for item in x for a in leaves(item)]
    return [np.asarray(x)]


def same(a, b):
    la, lb = leaves(a), leaves(b)
    return len(la) == len(lb) and all(rel(p, q) <= TOL for p, q in zip(la, lb))


def run(api, only_kind=None):
    for kind in ('probe', 'apply', 'entries'):
        if only_kind and kind != only_kind:
            continue
        for c, (order, N, W, F_in) in enumerate(CONFIGS):
            if kind == 'entries' and order != ():
                continue
            P = problem(7 + c + 10 * ('probe', 'apply', 'entries').index(kind), N)
            rng, d, frame = P['rng'], P['d'], P['frame']
            m = len(order)
            if kind == 'entries':
                index = np.stack([rng.integers(0, n, size=W) for n in P['N']])
                ww, smp = None, index
            else:
                index = None
                ww = [rng.standard_normal(tuple(F_in[i]) + W + (P['N'][i],)) for i in range(d)]
                smp = ww
            D, U, L, R = frame
            T = dense(L, D)
            z = sample(kind, T, ww, order, index)
            zref = sdef.z(kind, frame, smp, None, order)
            report('z %s %s (the dense model)' % (kind, order), same(z, zref), rel(leaves(z)[0], leaves(zref)[0]))
            Jp_p = sample(kind, tangent(frame, P['V'], P['H']), ww, order, index)
            units = [(fam, i, idx, sample(kind, Te, ww, order, index)) for fam, i, idx, Te in unit_tangents(frame)]
            zshapes = [a.shape for a in modes(kind, z)]
            y = [rng.standard_normal(sh) for sh in zshapes]
            y = y if kind == 'probe' else y[0]
            common = dict(variations=(P['V'], P['H']), core_directions=P['cd'])
            for name, (b, f) in weight_cases(kind, order, d, W, zshapes, rng).items():
                label = '%s %s [%s]' % (kind, order, name)
                if api == 'new':
                    wargs = {}
                    if b is not None:
                        wargs['block_weight'] = b
                    if f is not None:
                        wargs['full_weight'] = f
                else:
                    ow = old_weight(kind, zshapes, m, b, f)
                    wargs = {} if ow is None else dict(weight=ow)
                ex = brute(kind, z, y, Jp_p, units, order, b, f)
                ref = lambda nm, **kw: getattr(sdef, nm)(kind, frame, smp, None, order, **common, **kw)
                wt = ref('w_tilde', y=y, **wargs)
                report(label + ' w_tilde', same(modes(kind, wt), ex['w_tilde']))
                report(label + ' f', rel(ref('f', y=y, **wargs), ex['f']) <= TOL)
                report(label + ' r_block_sumsq', rel(ref('r_block_sumsq', y=y, **wargs), ex['r_block_sumsq']) <= TOL)
                report(label + ' Jp_norm_sq', rel(ref('Jp_norm_sq', **wargs), ex['Jp_norm_sq']) <= TOL)
                gg = blocks_of(ref('g', y=y, **wargs), units)
                report(label + ' g (every entry)', max(abs(gg[k] - ex['g'][k]) for k in gg) <=
                       TOL * max(1.0, max(abs(v) for v in ex['g'].values())))
                JJ = blocks_of(ref('JtJp', **wargs), units)
                report(label + ' JtJp (every entry)', max(abs(JJ[k] - ex['JtJp'][k]) for k in JJ) <=
                       TOL * max(1.0, max(abs(v) for v in ex['JtJp'].values())))
                my_seed = ex['w_tilde'] if kind == 'probe' else ex['w_tilde'][0]
                for nm in SEEDED:
                    report(label + ' %s(y, weights) == %s(w_tilde = the brute seed)' % (nm, nm),
                           same(ref(nm, y=y, **wargs), ref(nm, w_tilde=my_seed)))
                if name == 'both':
                    for nm in UNREAD:
                        report(label + ' %s ignores the weights' % nm, same(ref(nm, y=y, **wargs), ref(nm, y=y)))
            if api == 'new' and 1 not in W:
                refusals(kind, order, frame, smp, y, zshapes, d, W, common)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception as e:                      # noqa: BLE001
        return 'wrong exception %s: %s' % (type(e).__name__, e)
    return False


def refusals(kind, order, frame, smp, y, zshapes, d, W, common):
    m = len(order)
    F = tuple(n + 1 for n in order)
    ref = lambda nm, **kw: getattr(sdef, nm)(kind, frame, smp, None, order, **common, **kw)
    readers = ('f', 'r_block_sumsq', 'Jp_norm_sq', 'w_tilde', 'g', 'JtJp')
    bad_block, bad_full, bad_y = {}, {}, {}
    if kind == 'probe':
        if m:
            bad_block['(d,) only'] = np.ones((d,))                                # E3b at order (n,), d = n + 1
        bad_block['F only'] = np.ones(F) if m else np.ones((d, 1))
        bad_block['(d,)+F+(1,)'] = np.ones((d,) + F + (1,))
        bad_block['the old blocks_to_weight form'] = np.ones((d,) + F + (1,) * len(W) + (1,))
    elif kind == 'apply':
        bad_block['F+(1,)'] = np.ones(F + (1,))
        bad_block['(d,)+F'] = np.ones((d,) + F)
    else:
        bad_block['(1,)'] = np.ones((1,))
        bad_block['W'] = np.ones(W)
    table = np.ones((d,) + F if kind == 'probe' else F)
    type_block = {'a tuple': tuple(table) if table.ndim else (1.0,), 'a list': list(table) if table.ndim else [1.0]}

    def per_mode(fn):
        out = [fn(sh) for sh in zshapes]
        return out if kind == 'probe' else out[0]
    one = lambda sh, axes: np.ones(tuple(1 if a in axes else n for a, n in enumerate(sh)))
    if kind == 'probe':
        bad_full['edge of 1'] = per_mode(lambda sh: one(sh, {len(sh) - 1}))
        bad_full['mode-leading, mode axis of 1'] = np.ones((1,) + zshapes[0])
        bad_full['d - 1 arrays'] = [np.ones(sh) for sh in zshapes[:d - 1]]
        bad_y['edge of 1'] = per_mode(lambda sh: one(sh, {len(sh) - 1}) + 0.0)
    if m:
        bad_full['first jet axis of 1'] = per_mode(lambda sh: one(sh, {0}))
        bad_full['every structural axis 1 (old per mode)'] = per_mode(
            lambda sh: one(sh, set(range(m)) | ({len(sh) - 1} if kind == 'probe' else set())))
        bad_y['first jet axis of 1'] = per_mode(lambda sh: one(sh, {0}))
    bad_full['rank too high'] = per_mode(lambda sh: np.ones(sh + (1,)))
    bad_full['rank too low'] = per_mode(lambda sh: np.ones(sh[1:]))
    bad_y['W of 1'] = per_mode(lambda sh: one(sh, set(range(m, m + len(W)))))
    bad_y['one W axis of 1'] = per_mode(lambda sh: one(sh, {m}))
    if kind == 'entries':
        bad_full['a scalar weight (old per-all form)'] = np.ones(())
    for label, bw in bad_block.items():
        for nm in readers:
            r = raises(lambda: ref(nm, y=y, block_weight=bw), ValueError)
            report('refuse %s %s block %s in %s' % (kind, order, label, nm), r is True, r)
    for label, bw in type_block.items():
        for nm in readers:
            r = raises(lambda: ref(nm, y=y, block_weight=bw), TypeError)
            report('refuse %s %s block as %s in %s (TypeError)' % (kind, order, label, nm), r is True, r)
    for label, fw in bad_full.items():
        for nm in readers:
            r = raises(lambda: ref(nm, y=y, full_weight=fw), ValueError)
            report('refuse %s %s full weight %s in %s' % (kind, order, label, nm), r is True, r)
    for label, yy in bad_y.items():
        for nm in ('f', 'r_block_sumsq', 'w_tilde', 'g', 'r'):
            r = raises(lambda: ref(nm, y=yy), ValueError)
            report('refuse %s %s y %s in %s' % (kind, order, label, nm), r is True, r)
    # informational: the modes of a probe full weight with different size-1 W patterns (the engine refuses these
    # from part 2; the note does not say whether the reference does)
    if kind == 'probe':
        mixed = [one(sh, set(range(m, m + len(W))) if i == 0 else set()) for i, sh in enumerate(zshapes)]
        r = raises(lambda: ref('f', y=y, full_weight=mixed), ValueError)
        print('INFO %s %s: modes of a full weight disagreeing in their W pattern: %s' % (
            kind, order, 'refused' if r is True else 'accepted (read mode by mode)' if r is False else r))


def jax_check(api):
    """jax in, jax out, the same numbers: the weights as jax arrays (and a numpy block with a jax full weight)."""
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    P = problem(99)
    rng, d, frame = P['rng'], P['d'], P['frame']
    order, W = (2,), (3,)
    ww = [rng.standard_normal((3,) + W + (n,)) for n in P['N']]
    z = sdef.z('probe', frame, ww, None, order)
    y = [rng.standard_normal(a.shape) for a in z]
    b = 0.5 + rng.random((d, 3))
    f = [0.5 + rng.random((3, 1, a.shape[-1])) for a in z]
    common = dict(variations=(P['V'], P['H']))
    jf = [jnp.asarray(a) for a in f]
    if api == 'new':
        variants = {'numpy': dict(block_weight=b, full_weight=f), 'jax both': dict(block_weight=jnp.asarray(b),
                    full_weight=jf), 'numpy block, jax full': dict(block_weight=b, full_weight=jf),
                    'jax block only': dict(block_weight=jnp.asarray(b))}
    else:
        ow = old_weight('probe', [a.shape for a in z], 1, b, f)
        variants = {'numpy': dict(weight=ow), 'jax both': dict(weight=[jnp.asarray(a) for a in ow])}
    base = None
    for label, wargs in variants.items():
        for nm in ('f', 'w_tilde', 'r_block_sumsq', 'Jp_norm_sq', 'JtJp'):
            kw = dict(common, **wargs)
            if nm != 'Jp_norm_sq' and nm != 'JtJp':
                kw['y'] = [jnp.asarray(a) for a in y] if label != 'numpy' else y
            out = getattr(sdef, nm)('probe', frame, ww, None, order, **kw)
            is_jax = all(isinstance(a, jax.Array) for a in leaves_raw(out))
            if label != 'numpy':
                report('jax %s %s: jax out' % (label, nm), is_jax)
            if label == 'numpy' or (label == 'jax block only'):
                continue
            ref = getattr(sdef, nm)('probe', frame, ww, None, order, **dict(common, **variants['numpy'],
                                    **({} if nm in ('Jp_norm_sq', 'JtJp') else dict(y=y))))
            report('jax %s %s: the numbers' % (label, nm), same(out, ref))


def leaves_raw(x):
    if isinstance(x, (tuple, list)):
        return [a for item in x for a in leaves_raw(item)]
    return [x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', choices=('old', 'new'), required=True)
    ap.add_argument('--kind')
    ap.add_argument('--jax', action='store_true', help='only the jax dispatch check')
    args = ap.parse_args()
    if args.jax:
        jax_check(args.api)
    else:
        run(args.api, args.kind)
    print('%d checks, %d failed' % (COUNT[0], len(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
