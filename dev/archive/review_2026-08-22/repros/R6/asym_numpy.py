"""R6: forward jets (plain + tangent), standard vs _trs, dense oracle, polynomial-exact oracle,
adjoint identities -- at ASYMMETRIC shapes (nD != nU, rL != rR, distinct N, d in 1..4), orders 0..4,
with W / K / C stacks. numpy float64."""
import itertools
import sys
import numpy as np

import t3toolbox.backend.sampling_derivatives as pd
import t3toolbox.backend.probing as t3p
import t3toolbox.backend.apply as t3a
import t3toolbox.backend.entries as t3e
import t3toolbox.backend.tt_operations as tto
from common_r6 import *

TOL = 1e-9
fails = []
def check(tag, err, tol=TOL):
    if not (err <= tol):
        fails.append((tag, err))
        print('FAIL', tag, err)

rng = np.random.default_rng(123)
R = lambda *s: rng.standard_normal(s)

# ------------------------------------------------------------------ A. plain T3 forward, asymmetric
for d in (1, 2, 3, 4):
    N = ASYM[d][0]
    for C in [(), (2,)]:
        x = make_t3(d, C, rng)
        T = t3_dense(x)
        for W in [(), (3,), (2, 2)]:
            ww = [R(*W, n) for n in N]; pp = [R(*W, n) for n in N]
            for order in range(5):
                z = pd.t3_probe_derivatives(ww, pp, x, order)
                y = pd.t3_apply_derivatives(ww, pp, x, order)
                idx = np.stack([rng.integers(0, n, size=W) for n in N], axis=0)
                e = pd.t3_entries_derivatives(idx, pp, x, order)
                assert np.asarray(z[0]).shape == (order + 1,) + W + C + (N[0],), np.asarray(z[0]).shape
                assert y.shape == (order + 1,) + W + C
                for w in itertools.product(*[range(n) for n in W]):
                    for c in itertools.product(*[range(n) for n in C]):
                        wws = [a[w] for a in ww]; pps = [a[w] for a in pp]
                        zd = pd.dense_probe_derivatives(wws, pps, T[c], order)
                        for i in range(d):
                            check(f'A probe d={d} C={C} W={W} o={order} i={i}',
                                  relerr(np.asarray(z[i])[(slice(None),) + w + c], zd[i]))
                        check(f'A apply d={d} C={C} W={W} o={order}',
                              relerr(y[(slice(None),) + w + c], pd.dense_apply_derivatives(wws, pps, T[c], order)))
                        ii = [int(idx[(j,) + w]) for j in range(d)]
                        check(f'A entries d={d} C={C} W={W} o={order}',
                              relerr(e[(slice(None),) + w + c], pd.dense_entries_derivatives(ii, pps, T[c], order)))
                # polynomial-exact oracle, independent of dense_* (checks the t! convention at order>=2)
                if not C and not W:
                    zp = poly_jets(lambda v: np.stack(t3p.t3_probe(v, x)[0:1], 0)[0], ww, pp, order, d - 1 if d > 1 else 0)
                    check(f'A poly probe d={d} o={order}', relerr(np.asarray(z[0]), zp))
                    yp = poly_jets(lambda v: t3a.t3_apply(x, v), ww, pp, order, d)
                    check(f'A poly apply d={d} o={order}', relerr(y, yp))
print('A done')

# ------------------------------------------------------------------ B. tangent forward, asymmetric
for d in (1, 2, 3, 4):
    N = ASYM[d][0]
    for C in [(), (2,)]:
        frame = make_frame(d, C, rng)
        for K in [(), (2,)]:
            var = make_var(d, K, C, rng)
            Vd = tangent_dense(frame, var)
            for W in [(), (3,), (2, 2)]:
                ww = [R(*W, n) for n in N]; pp = [R(*W, n) for n in N]
                idx = np.stack([rng.integers(0, n, size=W) for n in N], axis=0)
                for order in range(5):
                    z = pd.tv_probe_derivatives(ww, pp, var, frame, order)
                    y = pd.tv_apply_derivatives(ww, pp, var, frame, order)
                    e = pd.tv_entries_derivatives(idx, pp, var, frame, order)
                    assert np.asarray(z[0]).shape == (order + 1,) + W + K + C + (N[0],)
                    assert y.shape == (order + 1,) + W + K + C, (y.shape, W, K, C)
                    for w in itertools.product(*[range(n) for n in W]):
                        wws = [a[w] for a in ww]; pps = [a[w] for a in pp]
                        ii = [int(idx[(j,) + w]) for j in range(d)]
                        for k in itertools.product(*[range(n) for n in K]):
                            for c in itertools.product(*[range(n) for n in C]):
                                sel = (slice(None),) + w + k + c
                                zd = pd.dense_probe_derivatives(wws, pps, Vd[k + c], order)
                                for i in range(d):
                                    check(f'B probe d={d} C={C} K={K} W={W} o={order} i={i}',
                                          relerr(np.asarray(z[i])[sel], zd[i]))
                                check(f'B apply d={d} C={C} K={K} W={W} o={order}',
                                      relerr(y[sel], pd.dense_apply_derivatives(wws, pps, Vd[k + c], order)))
                                check(f'B entries d={d} C={C} K={K} W={W} o={order}',
                                      relerr(e[sel], pd.dense_entries_derivatives(ii, pps, Vd[k + c], order)))
                    if not C and not W and not K:
                        zp = poly_jets(lambda v: np.asarray(t3p.tv_probe(v, var, frame)[d - 1]), ww, pp, order, max(d - 1, 0))
                        check(f'B poly probe d={d} o={order}', relerr(np.asarray(z[d - 1]), zp))
                        yp = poly_jets(lambda v: t3a.tv_apply(v, var, frame), ww, pp, order, d)
                        check(f'B poly apply d={d} o={order}', relerr(y, yp))
print('B done')

# ------------------------------------------------------------------ C. standard vs _trs, asymmetric
for d in (1, 2, 3, 4):
    N = ASYM[d][0]
    for W, K, C in [((), (), ()), ((3,), (2,), ()), ((2, 2), (2,), (2,)), ((), (), (2,))]:
        frame = make_frame(d, C, rng)
        up, down, left, right = frame
        var = make_var(d, K, C, rng)
        dU, dG = var
        ww = [R(*W, n) for n in N]; pp = [R(*W, n) for n in N]
        for order in range(5):
            trs = pd.binomial_combine_tensor(order)
            xi = pd.build_input_jets(pd.compute_xi(up, ww), pd.compute_xi(up, pp))
            dxi = pd.build_input_jets(pd.compute_xi(dU, ww), pd.compute_xi(dU, pp))
            mu_t = pd.compute_mu_jets_trs(left, xi, trs); mu = pd.compute_mu_jets(left, xi, trs)
            nu_t = pd.compute_nu_jets_trs(right, xi, trs); nu = pd.compute_nu_jets(right, xi, trs)
            eta_t = pd.compute_eta_jets_trs(down, mu_t, nu_t, trs); eta = pd.compute_eta_jets(down, mu, nu, trs)
            sg_t = pd.compute_sigma_jets_trs(dG, right, down, xi, dxi, mu_t, trs)
            sg = pd.compute_sigma_jets(dG, right, down, xi, dxi, mu, trs)
            ta_t = pd.compute_tau_jets_trs(dG, left, down, xi, dxi, nu_t, trs)
            ta = pd.compute_tau_jets(dG, left, down, xi, dxi, nu, trs)
            de_t = pd.compute_deta_jets_trs(dG, left, right, mu_t, nu_t, sg_t, ta_t, trs)
            de = pd.compute_deta_jets(dG, left, right, mu, nu, sg, ta, trs)
            zt = [R(order + 1, *W, *K, *C, n) for n in N]
            dett = pd.compute_deta_tilde_jets(up, zt)
            tt_t = pd.compute_tau_tilde_jets_trs(left, xi, dett, mu_t, trs); tt_ = pd.compute_tau_tilde_jets(left, xi, dett, mu, trs)
            st_t = pd.compute_sigma_tilde_jets_trs(right, xi, dett, nu_t, trs); st_ = pd.compute_sigma_tilde_jets(right, xi, dett, nu, trs)
            dxt = pd.compute_dxi_tilde_jets(down, mu, nu, st_, tt_, trs)
            for tag, a, b in [('mu', mu, mu_t), ('nu', nu, nu_t), ('eta', eta, eta_t), ('sigma', sg, sg_t),
                              ('tau', ta, ta_t), ('deta', de, de_t), ('tau_tilde', tt_, tt_t), ('sigma_tilde', st_, st_t)]:
                for i in range(d):
                    assert np.asarray(a[i]).shape == np.asarray(b[i]).shape, (tag, np.asarray(a[i]).shape, np.asarray(b[i]).shape)
                    check(f'C {tag} d={d} W={W} K={K} C={C} o={order} i={i}', relerr(a[i], b[i]), 1e-11)
            n_probe = len(W)
            for sop in (True, False):
                a = pd.assemble_tucker_variation_jets(zt, dxt, ww, pp, eta, n_probe, sop, chunk_size=2)
                b = pd.assemble_tucker_variation_jets_trs(zt, dxt, ww, pp, eta, n_probe, sop)
                for i in range(d):
                    check(f'C asmU d={d} W={W} K={K} C={C} o={order} sop={sop} i={i}', relerr(a[i], b[i]), 1e-11)
                a = pd.assemble_tt_variation_jets(st_, tt_, dett, xi, mu, nu, trs, n_probe, sop, chunk_size=2)
                b = pd.assemble_tt_variation_jets_trs(st_, tt_, dett, xi, mu, nu, trs, n_probe, sop)
                for i in range(d):
                    check(f'C asmG d={d} W={W} K={K} C={C} o={order} sop={sop} i={i}', relerr(a[i], b[i]), 1e-11)
print('C done')

# ------------------------------------------------------------------ D. adjoint identities, asymmetric
for d in (1, 2, 3, 4):
    N = ASYM[d][0]
    for W, K, C in [((), (), ()), ((3,), (), ()), ((3,), (2,), ()), ((2, 2), (2,), (2,)), ((), (2,), (2,))]:
        frame = make_frame(d, C, rng)
        var = make_var(d, K, C, rng)
        dU, dG = var
        ww = [R(*W, n) for n in N]; pp = [R(*W, n) for n in N]
        idx = np.stack([rng.integers(0, n, size=W) for n in N], axis=0)
        for order in range(5):
            for kind in ('probe', 'apply', 'entries'):
                if kind == 'probe':
                    Jv = pd.tv_probe_derivatives(ww, pp, var, frame, order)
                    r = [R(*np.asarray(z).shape) for z in Jv]
                    lhs = sum(np.sum(r[i] * np.asarray(Jv[i])) for i in range(d))
                    T = lambda sop: pd.tv_probe_derivatives_transpose(r, ww, pp, frame, order, sum_over_probes=sop, chunk_size=2)
                elif kind == 'apply':
                    Jv = pd.tv_apply_derivatives(ww, pp, var, frame, order)
                    r = R(*Jv.shape)
                    lhs = np.sum(r * Jv)
                    T = lambda sop: pd.tv_apply_derivatives_transpose(r, ww, pp, frame, order, sum_over_probes=sop)
                else:
                    Jv = pd.tv_entries_derivatives(idx, pp, var, frame, order)
                    r = R(*Jv.shape)
                    lhs = np.sum(r * Jv)
                    T = lambda sop: pd.tv_entries_derivatives_transpose(r, idx, pp, frame, order, sum_over_probes=sop)
                gU, gG = T(True)
                for i in range(d):
                    assert np.asarray(gU[i]).shape == dU[i].shape, (kind, np.asarray(gU[i]).shape, dU[i].shape)
                    assert np.asarray(gG[i]).shape == dG[i].shape, (kind, np.asarray(gG[i]).shape, dG[i].shape)
                rhs = sum(np.sum(np.asarray(gU[i]) * dU[i]) for i in range(d)) + sum(np.sum(np.asarray(gG[i]) * dG[i]) for i in range(d))
                check(f'D adj {kind} d={d} W={W} K={K} C={C} o={order}', abs(lhs - rhs) / max(abs(lhs), 1e-300))
                if W:
                    gU0, gG0 = T(False)
                    ax = tuple(range(len(W)))
                    for i in range(d):
                        assert np.asarray(gU0[i]).shape == W + dU[i].shape
                        check(f'D kept {kind} U d={d} W={W} K={K} C={C} o={order}', relerr(np.sum(np.asarray(gU0[i]), axis=ax), gU[i]))
                        check(f'D kept {kind} G d={d} W={W} K={K} C={C} o={order}', relerr(np.sum(np.asarray(gG0[i]), axis=ax), gG[i]))
print('D done')

# ------------------------------------------------------------------ E. corewise transposes vs FD at asym plain T3
for d in (1, 2, 3, 4):
    N = ASYM[d][0]
    x = make_t3(d, (), rng)
    W = (2,)
    ww = [R(*W, n) for n in N]; pp = [R(*W, n) for n in N]
    idx = np.stack([rng.integers(0, n, size=W) for n in N], axis=0)
    for order in (1, 3):
        for kind in ('probe', 'apply', 'entries'):
            if kind == 'probe':
                fwd = lambda data: pd.t3_probe_derivatives(ww, pp, data, order)
                r = [R(*np.asarray(z).shape) for z in fwd(x)]
                dot = lambda data: sum(np.sum(r[i] * np.asarray(fwd(data)[i])) for i in range(d))
                g = pd.t3_probe_corewise_derivatives_transpose(r, ww, pp, x, order, sum_over_probes=True)
            elif kind == 'apply':
                fwd = lambda data: pd.t3_apply_derivatives(ww, pp, data, order)
                c = R(*fwd(x).shape); dot = lambda data: float(np.sum(c * fwd(data)))
                g = pd.t3_apply_corewise_derivatives_transpose(c, ww, pp, x, order, sum_over_probes=True)
            else:
                fwd = lambda data: pd.t3_entries_derivatives(idx, pp, data, order)
                c = R(*fwd(x).shape); dot = lambda data: float(np.sum(c * fwd(data)))
                g = pd.t3_entries_corewise_derivatives_transpose(c, idx, pp, x, order, sum_over_probes=True)
            gU, gG = g
            dUs = [R(*u.shape) for u in x[0]]; dGs = [R(*gg.shape) for gg in x[1]]
            inner = sum(np.sum(np.asarray(gU[i]) * dUs[i]) for i in range(d)) + sum(np.sum(np.asarray(gG[i]) * dGs[i]) for i in range(d))
            eps = 1e-6
            plus = ([u + eps * du for u, du in zip(x[0], dUs)], [gg + eps * dg for gg, dg in zip(x[1], dGs)])
            minus = ([u - eps * du for u, du in zip(x[0], dUs)], [gg - eps * dg for gg, dg in zip(x[1], dGs)])
            fd = (dot(plus) - dot(minus)) / (2 * eps)
            check(f'E corewise {kind} d={d} o={order}', abs(inner - fd) / max(abs(fd), 1e-30), 1e-5)
print('E done')

print('TOTAL FAILS:', len(fails))
for f in fails[:40]:
    print(f)
