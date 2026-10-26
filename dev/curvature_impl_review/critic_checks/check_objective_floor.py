"""Critic experiment (read-only): exact-Hessian Steihaug trust-region Newton from a ZERO-PADDED continuation restart,
vs GN, with several lift-damping rules. The exact hvp is assembled from EXISTING library kernels + the block-sum
oracle (the writer's c4 construction); the GN parts, gradient, retraction and objective are the library's backend
LocalModel. Nothing here touches the repo."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import math
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt
import t3toolbox.corewise as cw
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.apply as ap
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.optimizers as bopt
from t3toolbox.backend.common import get_backend


def blocksum(gY, d, tr, ttr):
    gYU, gYG = gY
    gU = tuple(gYU[i][:tr[i]] for i in range(d))
    gG = []
    for i in range(d):
        r, n, rn = ttr[i], tr[i], ttr[i + 1]
        g = gYG[i]
        if i == 0:   gG.append(g[:, :n, rn:] + g[:, n:, :rn])
        elif i == d - 1: gG.append(g[:r, :n, :] + g[r:, n:, :])
        else:        gG.append(g[:r, :n, :rn] + g[r:, :n, rn:] + g[r:, n:, :rn])
    return gU, tuple(gG)


def ungauged_inner_grad(th, z):
    up, down, left, right = fvc.t3_corewise_frame(th)
    ztk, ztt = z
    ztt = tto.tt_squash_tails(ztt)
    _, xmap, _ = get_backend(False, False)
    (o2,) = xmap(tv._project_t3_rebasis_step, (ztt, ztk, up))
    zl = tto.tt_zipper_left_to_right(o2[:-1], left[:-1])
    zr = tto.tt_zipper_right_to_left(o2[1:], right[1:])
    dG, dB = xmap(tv._project_t3_variations_step, (zl, zr, ztt, ztk, down, up))
    return (tuple(dB), tuple(dG))


def filt(s, eps):
    keep = s > 1e-13 * max(float(s[:1].max(initial=0.0)), 1e-300)
    return np.where(keep, s / np.where(keep, s * s + eps * eps, 1.0), 0.0)


class Lift:
    def __init__(self, frame, eps):
        U, O, L, R = frame
        self.d = len(U)
        Z = tto.tt_zipper_right_to_left(L, R)
        self.SP, self.TP, self.smin = [], [], []
        for i in range(self.d):
            Hc = np.einsum('aib,bc->aic', L[i], Z[i + 1])
            ST = np.einsum('axb,aub->xu', O[i], Hc)
            u, s, vt = np.linalg.svd(ST, full_matrices=False)
            self.SP.append(vt.T @ np.diag(filt(s, eps)) @ u.T)            # (nU, nD)
            self.smin.append(float(s.min()))
            if i < self.d - 1:
                u, s, vt = np.linalg.svd(Z[i + 1], full_matrices=False)
                self.TP.append(vt.T @ np.diag(filt(s, eps)) @ u.T)        # (rR, rL)
                self.smin.append(float(s.min()))

    def lift(self, v):
        V, H = v
        Ud = tuple(self.SP[i] @ V[i] for i in range(self.d))
        Gd = tuple(np.einsum('aib,bc->aic', H[i], self.TP[i]) if i < self.d - 1 else H[i] for i in range(self.d))
        return Ud, Gd

    def lift_T(self, y):
        YU, YG = y
        V = tuple(self.SP[i].T @ YU[i] for i in range(self.d))
        H = tuple(np.einsum('aic,bc->aib', YG[i], self.TP[i]) if i < self.d - 1 else YG[i] for i in range(self.d))
        return V, H


def exact_curv(lm, ww, lift_obj, g_misfit):
    frame = lm.frame
    U, O, L, R = frame
    d = len(U)
    tr = [u.shape[0] for u in U]
    ttr = [L[0].shape[0]] + [l.shape[2] for l in L]
    cframe = fvc.t3_corewise_frame((tuple(U), tuple(L)))
    Zg = tv.tv_to_t3(frame, g_misfit)

    def curv(p):
        Pp = tv.tv_orthogonal_gauge_projection(frame, p)
        th = lift_obj.lift(Pp)
        Y = tv.tv_to_t3(cframe, th)
        Kr = blocksum(ap.t3_apply_corewise_transpose(lm.residual, ww, Y, sum_over_probes=True), d, tr, ttr)
        Kg = blocksum(ungauged_inner_grad(Y, Zg), d, tr, ttr)
        K = cw.corewise_sub(Kr, Kg)
        return tv.tv_orthogonal_gauge_projection(frame, lift_obj.lift_T(K))
    return curv


def steihaug(hvp, rhs, tol, maxiter, radius):
    p = cw.corewise_zeros_like(rhs); r = rhs; dd = rhs; rs = cw.corewise_dot(r, r)
    it = 0; hit = False; negc = False
    while rs > tol * tol and it < maxiter:
        Hd = hvp(dd); dHd = cw.corewise_dot(dd, Hd)
        pp, pd, d2 = cw.corewise_dot(p, p), cw.corewise_dot(p, dd), cw.corewise_dot(dd, dd)
        if dHd <= 0 or (pp + 2 * (rs / dHd) * pd + (rs / dHd) ** 2 * d2 >= radius * radius):
            tau = (-pd + math.sqrt(max(pd * pd + d2 * (radius * radius - pp), 0.0))) / d2
            p = cw.corewise_add(p, cw.corewise_scale(dd, tau)); hit = True; negc = dHd <= 0; it += 1
            break
        a = rs / dHd
        p = cw.corewise_add(p, cw.corewise_scale(dd, a)); r = cw.corewise_sub(r, cw.corewise_scale(Hd, a))
        rs_new = cw.corewise_dot(r, r); dd = cw.corewise_add(r, cw.corewise_scale(dd, rs_new / rs)); rs = rs_new
        it += 1
    return p, it, hit, negc


def run(problem, ww, x, mode, eps_rule, max_outer=60, gtol=1e-8, g0=None, cg_max=80, tag=''):
    radius = None; n_rej = 0; n_hvp = 0; it = 0; lm = None; acc_last = True; hist = []
    while it < max_outer:
        if acc_last:
            lm = problem.local_model(x)
            g = lm.gradient
            gnorm = math.sqrt(cw.corewise_dot(g, g))
            if g0 is None:
                g0 = gnorm
            f = float(lm.misfit)
            Xn = math.sqrt(float(cw.corewise_dot(lm.frame[2][-1], lm.frame[2][-1])))
        if gnorm <= gtol * g0:
            break
        if radius is None:
            radius = gnorm ** 3 / max(float(lm.gn_quadratic(g)), 1e-300)
        if mode == 'exact':
            eps = {'zero': 0.0, 'rel1e-6': 1e-6 * Xn, 'rel1e-2': 1e-2 * Xn, 'delta': radius}[eps_rule]
            lift_obj = Lift(lm.frame, eps)
            curv = exact_curv(lm, ww, lift_obj, g)
            hvp = lambda p: cw.corewise_add(lm.hvp(p), curv(p))
            smin = min(lift_obj.smin)
        else:
            hvp = lm.hvp; smin = float('nan')
        cnt = [0]
        def hvp_c(p):
            cnt[0] += 1; return hvp(p)
        eta = min(0.5, (gnorm / g0) ** (1.0 if mode == 'exact' else 0.5))
        neg_g = cw.corewise_scale(g, -1.0)
        p, cgi, hit, negc = steihaug(hvp_c, neg_g, eta * gnorm, cg_max, radius)
        n_hvp += cnt[0]
        pHp = cw.corewise_dot(p, hvp(p))
        slope = cw.corewise_dot(g, p)
        pred = slope + 0.5 * pHp
        x_trial = lm.retract(p)
        f_new = float(problem.objective(x_trial))
        rho = (f_new - f) / pred if pred < 0 else float('nan')
        pn = math.sqrt(cw.corewise_dot(p, p))
        accept = math.isfinite(rho) and rho > 0.1
        hist.append((it, gnorm / g0, radius, rho, accept, cgi, hit, negc, smin))
        if (not math.isfinite(rho)) or rho < 0.25:
            radius = 0.25 * pn
        elif rho > 0.75 and hit:
            radius = 2 * radius
        if accept:
            x = x_trial; acc_last = True
        else:
            n_rej += 1; acc_last = False
        it += 1
        if radius < 1e-14 * max(Xn, 1.0):
            hist.append(('COLLAPSE',)); break
    conv = gnorm <= gtol * g0
    return dict(iters=it, rejected=n_rej, hvps=n_hvp, converged=conv, final_rel_g=gnorm / g0, hist=hist)



def scenario(label, target_tk, target_tt, lvl1_tk, lvl1_tt, lvl2_tk, lvl2_tt, noise, seed=7):
    rng = np.random.default_rng(seed)
    shape = (6, 6, 6)
    np.random.seed(seed)
    target = t3.TuckerTensorTrain.randn(shape, target_tk, target_tt)
    W = 400
    ww = [rng.standard_normal((W, N)) for N in shape]
    clean = np.asarray(target.apply(ww))
    data = clean + noise * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
    x1 = t3.TuckerTensorTrain.randn(shape, lvl1_tk, lvl1_tt)
    x1, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x1, max_newton=80, gtol_rel=1e-10)
    x2 = x1.resize(shape, lvl2_tk, lvl2_tt)
    problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
    print(f"\n== {label}: target {target_tk}/{target_tt}, level-1 {lvl1_tk}, level-2 {lvl2_tk}/{lvl2_tt}, noise {noise}")
    _, st = bopt.newton_cg(problem, x2.data, max_newton=80, gtol_rel=1e-8)
    print(f"  GN + Armijo (library): newton iters {st['newton']}, final rel g {st['history'][-1]['gnorm'] / st['history'][-1]['g0norm']:.1e}")
    out = {}
    for mode, rule in (('gn', None), ('exact', 'rel1e-6'), ('exact', 'rel1e-2')):
        r = run(problem, ww, x2.data, mode, rule, max_outer=80)
        name = 'GN + TR' if mode == 'gn' else f'exact + TR, eps={rule}'
        hs = [h for h in r['hist'] if h[0] != 'COLLAPSE']
        sm = [h[8] for h in hs if math.isfinite(h[8])]
        negs = sum(1 for h in hs if h[7])
        print(f"  {name:28s}: outer {r['iters']:3d} (rejected {r['rejected']:2d}), hvps {r['hvps']:5d}, "
              f"converged {r['converged']}, final rel g {r['final_rel_g']:.1e}, neg-curv stops {negs}"
              + (f", min sigma over run {min(sm[1:]) if len(sm) > 1 else float('nan'):.1e} (|X|~{float(np.linalg.norm(x2.to_dense())):.1e})" if sm else ''))
        out[name] = r
    return out



rng = np.random.default_rng(7)
shape = (6, 6, 6)
np.random.seed(7)
target = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
W = 400
ww = [rng.standard_normal((W, N)) for N in shape]
data = np.asarray(target.apply(ww))
x1 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
x1, st1 = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x1, max_newton=80, gtol_rel=1e-10)
x2 = x1.resize(shape, (3, 3, 3), (1, 3, 3, 1))
problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
lm = problem.local_model(x2.data)
f = float(lm.misfit); g = lm.gradient; gnorm = math.sqrt(cw.corewise_dot(g, g))
gq = float(lm.gn_quadratic(g))
radius = gnorm ** 3 / gq
p = cw.corewise_scale(g, -radius / gnorm)
pred = cw.corewise_dot(g, p) + 0.5 * float(lm.gn_quadratic(p))
eps = np.finfo(float).eps
r = np.asarray(lm.residual); y = np.asarray(data)
Xn = math.sqrt(float(cw.corewise_dot(lm.frame[2][-1], lm.frame[2][-1])))
print(f"noise-free over-ranked restart: f = {f:.3e}, |g| = {gnorm:.3e}, Delta0 = {radius:.3e}, |X|_HS = {Xn:.3e}")
print(f"  plan's floor 1e-12*max(|X|, Delta0) = {1e-12 * max(Xn, radius):.3e}  -> Delta0 below floor: {radius < 1e-12 * max(Xn, radius)}")
print(f"  |pred| at the Cauchy step = {abs(pred):.3e}")
print(f"  plan's objective_floor test 16*eps*|f| = {16 * eps * abs(f):.3e}  -> classified objective_floor: {abs(pred) <= 16 * eps * abs(f)}")
print(f"  cancellation scale of f (eps * |y| * |r|) = {eps * np.linalg.norm(y) * np.linalg.norm(r):.3e}")
