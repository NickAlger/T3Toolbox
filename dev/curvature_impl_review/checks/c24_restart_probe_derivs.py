"""Read-only plan check (v3): E2's zero-padded-restart experiment on T3Polynomial's kind -- probe_derivatives
(order 2, omega[mode, order] weight) -- with the exact curvature from the v3 recipe (c22.two_hole_v3, fed by
the manifold model's stored sweep, nu = T nu) + the T3-T3 block-sum, through the critic's Steihaug/TR loop logic.
GN parts, gradient, retraction, objective: the library's backend LocalModel."""
import sys, os, math, io, contextlib
sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(SCR, 'critic_checks', 'check_restart_tr2.py')).read()
ns = {}
exec(compile(src[:src.index('def scenario(')], 'check_restart_tr2_defs', 'exec'), ns)
np, t3, t3m, topt, bopt, bgeo, bfit, cw, tv, fvc = (ns[k] for k in ('np', 't3', 't3m', 'topt', 'bopt', 'bgeo', 'bfit', 'cw', 'tv', 'fvc'))
Lift, steihaug, blocksum, ungauged = ns['Lift'], ns['steihaug'], ns['blocksum'], ns['ungauged_inner_grad']
import t3toolbox.backend.sampling_derivatives as sd
from c22_probe_derivs_v3_recipe import two_hole_v3

ORDER = 2
def curvature(lm, g_mis, lift_obj, kind):
    frame = lm.frame; U, O, L, R = frame; d = len(U)
    tr = [u.shape[0] for u in U]; ttr = [L[0].shape[0]] + [l.shape[2] for l in L]
    cframe = fvc.t3_corewise_frame((tuple(U), tuple(L))); Zg = tv.tv_to_t3(frame, g_mis)
    ww, pp = lm.sample
    zt = kind._apply_weight(lm.residual, 2)                     # omega^2 r, exactly as the kind's transpose
    def curv(p):
        Pp = tv.tv_orthogonal_gauge_projection(frame, p)
        th = lift_obj.lift(Pp)
        Kr = two_hole_v3(frame, lm.sweep, tuple(a[None] for a in th[0]), tuple(a[None] for a in th[1]),
                         ww, pp, zt, ORDER, (1,), ())
        Kr = (tuple(a[0] for a in Kr[0]), tuple(a[0] for a in Kr[1]))
        Kg = blocksum(ungauged(tv.tv_to_t3(cframe, th), Zg), d, tr, ttr)
        return tv.tv_orthogonal_gauge_projection(frame, lift_obj.lift_T(cw.corewise_sub(Kr, Kg)))
    return curv

def run(problem, x, mode, eps_rel, max_outer=80, gtol=1e-8):
    radius = None; n_rej = n_hvp = it = 0; acc_last = True; g0 = None
    while it < max_outer:
        if acc_last:
            lm = problem.local_model(x); g = lm.gradient; gnorm = math.sqrt(cw.corewise_dot(g, g))
            g0 = gnorm if g0 is None else g0; f = float(lm.objective)
            Xn = math.sqrt(float(cw.corewise_dot(lm.frame[2][-1], lm.frame[2][-1])))
            if mode == 'exact':
                curv = curvature(lm, g, Lift(lm.frame, eps_rel * Xn), problem.kind)
                hvp = lambda p: cw.corewise_add(lm.hvp(p), curv(p))
            else:
                hvp = lm.hvp
        if gnorm <= gtol * g0:
            break
        if radius is None:
            radius = gnorm ** 3 / max(float(lm.gn_quadratic(g)), 1e-300); r0 = radius
        cnt = [0]
        def hvp_c(p):
            cnt[0] += 1; return hvp(p)
        eta = min(0.5, (gnorm / g0) ** (1.0 if mode == 'exact' else 0.5))
        p, cgi, hit, negc = steihaug(hvp_c, cw.corewise_scale(g, -1.0), eta * gnorm, 80, radius)
        n_hvp += cnt[0]
        pred = cw.corewise_dot(g, p) + 0.5 * cw.corewise_dot(p, hvp(p))
        xt = lm.retract(p); fn = float(problem.objective(xt))
        rho = (fn - f) / pred if pred < 0 else float('nan'); pn = math.sqrt(cw.corewise_dot(p, p))
        accept = math.isfinite(rho) and rho > 0.1
        if (not math.isfinite(rho)) or rho < 0.25:
            radius = 0.25 * pn
        elif rho > 0.75 and hit:
            radius = min(2 * radius, 1e4 * r0)
        it += 1
        if accept:
            x = xt; acc_last = True
        else:
            n_rej += 1; acc_last = False
    return dict(it=it, rej=n_rej, hvps=n_hvp, conv=gnorm <= gtol * g0, rel=gnorm / g0)

rng = np.random.default_rng(7); np.random.seed(7)
shape, W, noise = (6, 6, 6), 120, 0.05
target = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
ww = [rng.standard_normal((W, N)) for N in shape]; pp = [rng.standard_normal((W, N)) for N in shape]
kind = bfit.probe_derivatives_kind(ORDER, np.array([1.0, 0.5, 0.25]))
clean = kind.point_forward(target.data, (ww, pp))
data = [z + noise * np.linalg.norm(z) / math.sqrt(z.size) * rng.standard_normal(z.shape) for z in clean]
x1 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
with contextlib.redirect_stdout(io.StringIO()):
    x1, _ = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), data, x1, order=ORDER,
                           weight=np.array([1.0, 0.5, 0.25]), max_newton=80, gtol_rel=1e-10)
x2 = x1.resize(shape, (3, 3, 3), (1, 3, 3, 1))
problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), kind, (ww, pp), data)
_, st = bopt.newton_cg(problem, x2.data, max_newton=80, gtol_rel=1e-8)
print(f"probe_derivatives order {ORDER}, zero-padded restart rank 2 -> 3 (target 4), W={W}, 5% noise")
print(f"  GN + Armijo (library): newton {st['newton']}, final rel g {st['history'][-1]['gnorm'] / st['history'][-1]['g0norm']:.1e}")
for mode, eps in (('gn', None), ('exact', 0.0), ('exact', 1e-6), ('exact', 1e-2)):
    r = run(problem, x2.data, mode, eps)
    print(f"  {mode + ('' if eps is None else f' eps={eps:g}*|X|'):22s}: outer {r['it']:3d} (rejected {r['rej']:2d}), hvps {r['hvps']:4d}, converged {r['conv']}, final rel g {r['rel']:.1e}")
