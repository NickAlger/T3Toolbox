"""Read-only cost survey of the production dense-trs contractions in sampling_derivatives.py.

For each site: the production grouped string, expanded at |W|=1 (letter w), |K|=0 or 1 (letter y),
|C|=0; the numpy path production uses (contractions._pairwise_path on the GROUPED string) and the jax
path (opt_einsum 'auto', what jnp.einsum uses), both costed by opt_einsum on shapes; versus the
sparse/banded alternative written as explicit 2-operand steps (+ O(size) shift/scale, not counted).
Each alternative is also checked numerically against the dense contraction on small random arrays.
"""
import math
import sys
import numpy as np
import opt_einsum as oe

sys.path.insert(0, '/home/claude/t3toolbox')
import t3toolbox.backend.contractions as ct
import t3toolbox.backend.sampling_derivatives as sd


def expand(g, K):
    return g.replace('W', 'w').replace('K', 'y' if K else '').replace('C', '')


def cost(sub, shapes, path='auto'):
    p, info = oe.contract_path(sub, *shapes, shapes=True, optimize=path)
    return int(info.opt_cost), int(info.largest_intermediate)


def np_path(grouped):
    return [tuple(x) for x in ct._pairwise_path(grouped)[1:]]


def pair_rev(j):      # [j^(s), (s+1) j^(s+1)]  -- the reverse shift of _adj_tilde_step / pair()
    o1 = j.shape[0]
    up = np.concatenate([j[1:], np.zeros_like(j[:1])], 0)
    w = (np.arange(o1) + 1).reshape((o1,) + (1,) * (j.ndim - 1))
    return np.stack([j, w * up], 0)


def pair_fwd(j):      # [j^(t), t j^(t-1)]    -- the forward shift of _mu_jets_step / stacked()
    o1 = j.shape[0]
    dn = np.concatenate([np.zeros_like(j[:1]), j[:-1]], 0)
    w = np.arange(o1).reshape((o1,) + (1,) * (j.ndim - 1))
    return np.stack([j, w * dn], 0)


def run(o, W, r, n, K):
    trs = sd.binomial_combine_tensor(o)
    o1 = o + 1
    Kd = (K,) if K else ()
    rows = []

    def site(name, g, ops_shapes, alt_steps, check=None):
        sub = expand(g, K)
        npc = cost(sub, ops_shapes, np_path(g))
        jxc = cost(sub, ops_shapes, 'auto')
        alt = [cost(s, sh, 'auto') for s, sh in alt_steps]
        altc = (sum(a for a, _ in alt), max(b for _, b in alt))
        rows.append((name, ops_shapes[0], npc, jxc, altc, check))

    # P1: affine forward pushthrough (L311/L1197; _sigma_jet_step L761-763 is this x3)
    site('P1 fwd push (L1197; L761-3 x3)', 'trs,rWCa,Caib,sWCi->tWCb',
         [(o1, o1, 2), (o1, W, r), (r, n, r), (2, W, n)],
         [('aib,swi->swab', [(r, n, r), (2, W, n)]), ('stwa,swab->twb', [(2, o1, W, r), (2, W, r, r)])])
    # P2: affine reverse propagation (L2334 _sigma_hat_jets_step)
    site('P2 rev prop (L2334)', 'trs,tWKCa,Caib,rWCi->sWKCb',
         [(o1, 2, o1), (o1, W) + Kd + (r,), (r, n, r), (2, W, n)],
         [('aib,kwi->kwab', [(r, n, r), (2, W, n)]),
          (expand('ksWKa,kWab->sWKb', K), [(2, o1, W) + Kd + (r,), (2, W, r, r)])])
    # P4: dxi-type combine, full trs, only output rows {0,1} consumed (L1637/1659, L2407/2425, S3 L2865-2904 x3 each)
    site('P4 dxi from_tau (L1637)', 'trs,tWKCa,Caib,sWCb->rWKCi',
         [(o1, o1, o1), (o1, W) + Kd + (r,), (r, n, r), (o1, W, r)],
         [('aib,swb->swai', [(r, n, r), (o1, W, r)]),
          (expand('usWKa,sWai->uWKi', K), [(2, o1, W) + Kd + (r,), (o1, W, r, n)])])
    site('P4 dxi from_sig (L1638/L2407)', 'trs,rWCa,Caib,tWKCb->sWKCi',
         [(o1, o1, o1), (o1, W, r), (r, n, r), (o1, W) + Kd + (r,)],
         [('rwa,aib->rwib', [(o1, W, r), (r, n, r)]),
          (expand('rWib,urWKb->uWKi', K), [(o1, W, n, r), (2, o1, W) + Kd + (r,)])])
    # P5: TT assembly (L1779-1781 / L1829-1831; L2386/L2428 is t_sig only)
    site('P5 t_sig (L1779; L2386)', 'trs,rWCa,sWCi,tWKCb->KCaib',
         [(o1, o1, 2), (o1, W, r), (2, W, n), (o1, W) + Kd + (r,)],
         [(expand('krWKb,rWa->kWKab', K), [(2, o1, W) + Kd + (r,), (o1, W, r)]),
          (expand('kWKab,kWi->Kaib', K), [(2, W) + Kd + (r, r), (2, W, n)])])
    site('P5 t_tau (L1780)', 'trs,tWKCa,rWCi,sWCb->KCaib',
         [(o1, 2, o1), (o1, W) + Kd + (r,), (2, W, n), (o1, W, r)],
         [(expand('ksWKa,sWb->kWKab', K), [(2, o1, W) + Kd + (r,), (o1, W, r)]),
          (expand('kWKab,kWi->Kaib', K), [(2, W) + Kd + (r, r), (2, W, n)])])
    # t_det alternative: order scan over the lag r (o1 steps), each with a full-length shifted dt slice
    site('P5 t_det (L1781)', 'trs,rWCa,tWKCi,sWCb->KCaib',
         [(o1, o1, o1), (o1, W, r), (o1, W) + Kd + (n,), (o1, W, r)],
         [(expand('sWKi,sWb->WKib', K), [(o1, W) + Kd + (n,), (o1, W, r)])] * o1
         + [(expand('Wa,WKib->Kaib', K), [(W, r), (W,) + Kd + (n, r)])] * o1)
    # S3 factored assembly: the two trs builds (L2998 M full trs; L3000 N trs[:, :2, :])
    site('S3 M build (L2998/L2999)', 'trs,rWCa->tsWCa', [(o1, o1, o1), (o1, W, r)],
         [('kta,ta->kta', [(2, o1, W * r), (o1, W * r)])])      # stand-in: the shift-scale is O(2 o1 W r)
    site('S3 N build (L3000/L3001)', 'trs,tWCa->rsWCa', [(o1, 2, o1), (o1, W, r)],
         [('kta,ta->kta', [(2, o1, W * r), (o1, W * r)])])
    return rows


def check(o=4, W=7, r=3, n=5, K=2):
    rng = np.random.default_rng(0)
    trs = sd.binomial_combine_tensor(o)
    o1 = o + 1
    R = lambda *s: rng.standard_normal(s)
    c = ct.contract
    mu, nu, tt, st = R(o1, W, r), R(o1, W, r), R(o1, W, K, r), R(o1, W, K, r)
    G, xi, dt = R(r, n, r), R(2, W, n), R(o1, W, K, n)
    out = {}
    # P1
    d = c('trs,rWCa,Caib,sWCi->tWCb', trs[:, :, :2], mu, G, xi)
    a = c('stWCa,sWCab->tWCb', pair_fwd(mu), c('Caib,sWCi->sWCab', G, xi))
    out['P1'] = np.abs(d - a).max()
    # P2
    d = c('trs,tWKCa,Caib,rWCi->sWKCb', trs[:, :2, :], tt, G, xi)
    a = c('ksWKCa,kWCab->sWKCb', pair_rev(tt), c('Caib,kWCi->kWCab', G, xi), len_C=0)
    out['P2'] = np.abs(d - a).max()
    # P4 (rows {0,1} only)
    d = c('trs,tWKCa,Caib,sWCb->rWKCi', trs, tt, G, nu)[:2]
    a = c('usWKCa,sWCai->uWKCi', pair_rev(tt), c('Caib,sWCb->sWCai', G, nu), len_C=0)
    out['P4 from_tau'] = np.abs(d - a).max()
    d = c('trs,rWCa,Caib,tWKCb->sWKCi', trs, mu, G, st)[:2]
    a = c('rWCib,urWKCb->uWKCi', c('rWCa,Caib->rWCib', mu, G), pair_rev(st), len_C=0)
    out['P4 from_sig'] = np.abs(d - a).max()
    # P5
    d = c('trs,rWCa,sWCi,tWKCb->KCaib', trs[:, :, :2], mu, xi, st, len_W=1)
    a = c('kWKCab,kWCi->KCaib', c('krWKCb,rWCa->kWKCab', pair_rev(st), mu, len_C=0), xi, len_W=1)
    out['P5 t_sig'] = np.abs(d - a).max()
    d = c('trs,tWKCa,rWCi,sWCb->KCaib', trs[:, :2, :], tt, xi, nu, len_W=1)
    a = c('kWKCab,kWCi->KCaib', c('ksWKCa,sWCb->kWKCab', pair_rev(tt), nu, len_C=0), xi, len_W=1)
    out['P5 t_tau'] = np.abs(d - a).max()
    d = c('trs,rWCa,tWKCi,sWCb->KCaib', trs, mu, dt, nu, len_W=1)
    acc = 0
    for lag in range(o1):       # sum_r mu^(r) (x) sum_s C(r+s, r) dt^(r+s) (x) nu^(s)
        w = np.array([math.comb(lag + s, lag) if lag + s <= o else 0 for s in range(o1)], float)
        sh = np.concatenate([dt[lag:], np.zeros_like(dt[:lag])], 0) * w.reshape(-1, 1, 1, 1)
        acc = acc + c('WCa,WKCib->KCaib', mu[lag], c('sWKCi,sWCb->WKCib', sh, nu, len_C=0), len_W=1)
    out['P5 t_det'] = np.abs(d - acc).max()
    # P3: the per-r 'ts' weight == shift by r, scale by C(t, r)
    X = R(o1, W, K, n)
    errs = []
    trs_r = np.moveaxis(trs, 1, 0)
    for lag in range(o1):
        d = c('ts,sWKCi->tWKCi', trs_r[lag], X, len_C=0)
        w = np.array([math.comb(t, lag) if t >= lag else 0 for t in range(o1)], float)
        a = np.concatenate([np.zeros_like(X[:lag]), X[:o1 - lag]], 0) * w.reshape(-1, 1, 1, 1)
        errs.append(np.abs(d - a).max())
    out['P3 fwd ts'] = max(errs)
    errs = []
    for lag in range(o1):       # the adjoint form 'ts,tWKCb->sWKCb': out^(s) = C(s+lag, lag) X^(s+lag)
        d = c('ts,tWKCb->sWKCb', trs_r[lag], X, len_C=0)
        w = np.array([math.comb(s + lag, lag) if s + lag <= o else 0 for s in range(o1)], float)
        a = np.concatenate([X[lag:], np.zeros_like(X[:lag])], 0) * w.reshape(-1, 1, 1, 1)
        errs.append(np.abs(d - a).max())
    out['P3 adj ts'] = max(errs)
    # dxi rows >= 2 really are dead: show they are nonzero (so the dense form computes them)
    out['P4 rows>=2 nonzero (max abs)'] = np.abs(c('trs,tWKCa,Caib,sWCb->rWKCi', trs, tt, G, nu)[2:]).max()
    return out


if __name__ == '__main__':
    print('numerical equality of each banded/scan alternative vs the dense contraction:')
    for k, v in check().items():
        print('  %-30s %.2e' % (k, v))
    for (W, r, n) in [(2000, 16, 16), (2000, 64, 32)]:
        for K in (0, 3):
            for o in (2, 3, 5, 8):
                print('\n=== o=%d W=%d r=%d n=%d K=%s ===' % (o, W, r, n, K or '()'))
                print('  %-32s %-12s %22s %22s %22s %7s %7s' % ('site', 'trs shape', 'numpy flops / peak',
                      'jax flops / peak', 'alt flops / peak', 'np/alt', 'jx/alt'))
                for name, tshape, npc, jxc, altc, _ in run(o, W, r, n, K):
                    print('  %-32s %-12s %12.3g / %-8.2g %12.3g / %-8.2g %12.3g / %-8.2g %7.2f %7.2f' % (
                        name, str(tshape), npc[0], npc[1], jxc[0], jxc[1], altc[0], altc[1],
                        npc[0] / altc[0], jxc[0] / altc[0]))
