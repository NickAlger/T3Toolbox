"""Read-only plan check (v5, I-7): T16's problem (c20: (6,6,6), target (4,4,4), fit rank 2, apply, W = 200, 1% noise,
start = 15 GN steps) with the exact Hessian + Steihaug/TR at cg_forcing_power 0.5 (v5's single default) vs 1.0 (the
documented choice for a q-quadratic tail). Reports e_k = |g_k|/|g_0| on accepted steps and the rate constants
e_{k+1}/e_k^2 and e_{k+1}/e_k^1.5 inside [1e-8, 1e-1]."""
import sys, os, math, io, contextlib
sys.path.insert(0, __import__("os").getcwd())
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(SCR, 'check_restart_tr2.py')).read()
ns = {}
exec(compile(src[:src.index('def scenario(')], 'check_restart_tr2_defs', 'exec'), ns)
np, t3, t3m, topt, bopt, bgeo, bfit, cw = (ns[k] for k in ('np', 't3', 't3m', 'topt', 'bopt', 'bgeo', 'bfit', 'cw'))
Lift, exact_curv, steihaug = ns['Lift'], ns['exact_curv'], ns['steihaug']


def run(problem, ww, x, power, g0, gtol=1e-12, max_outer=30):
    radius = None; acc_last = True; es = []; hv = 0; rej = 0
    for _ in range(max_outer):
        if acc_last:
            lm = problem.local_model(x); g = lm.gradient; gnorm = math.sqrt(cw.corewise_dot(g, g)); f = float(lm.misfit)
            Xn = math.sqrt(float(cw.corewise_dot(lm.frame[2][-1], lm.frame[2][-1])))
            curv = exact_curv(lm, ww, Lift(lm.frame, 1e-6 * Xn), g)
            hvp = lambda p: cw.corewise_add(lm.hvp(p), curv(p))
            es.append(gnorm / g0)
        if gnorm <= gtol * g0:
            break
        if radius is None:
            radius = gnorm ** 3 / max(float(lm.gn_quadratic(g)), 1e-300); r0 = radius
        cnt = [0]
        def hc(p):
            cnt[0] += 1; return hvp(p)
        eta = min(0.5, (gnorm / g0) ** power)
        p, _, hit, _ = steihaug(hc, cw.corewise_scale(g, -1.0), eta * gnorm, 80, radius); hv += cnt[0]
        pred = cw.corewise_dot(g, p) + 0.5 * cw.corewise_dot(p, hvp(p))
        xt = lm.retract(p); fn = float(problem.objective(xt))
        rho = (fn - f) / pred if pred < 0 else float('nan'); pn = math.sqrt(cw.corewise_dot(p, p))
        if (not math.isfinite(rho)) or rho < 0.25:
            radius = 0.25 * pn
        elif rho > 0.75 and hit:
            radius = min(2 * radius, 1e4 * r0)
        acc_last = math.isfinite(rho) and rho > 0.1
        if acc_last:
            x = xt
        else:
            rej += 1
    return es, hv, rej


for seed in (1, 2):
    rng = np.random.default_rng(seed); np.random.seed(seed)
    shape, W = (6, 6, 6), 200
    tgt = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    clean = np.asarray(tgt.apply(ww)); data = clean + 0.01 * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
    x0 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        x15, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=15, gtol_rel=0.0)
    problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
    lm = problem.local_model(x15.data); g = lm.gradient; g0 = math.sqrt(float(cw.corewise_dot(g, g)))
    for power in (1.0, 0.5):
        es, hv, rej = run(problem, ww, x15.data, power, g0)
        win = [(a, b) for a, b in zip(es, es[1:]) if 1e-8 < a < 1e-1]
        print(f"seed {seed} power {power}: steps {len(es) - 1}, rejected {rej}, hvps {hv}; e_k " + ' '.join('%.1e' % v for v in es))
        print('    e_{k+1}/e_k^2  : ' + ' '.join('%.2e' % (b / a ** 2) for a, b in win))
        print('    e_{k+1}/e_k^1.5: ' + ' '.join('%.2e' % (b / a ** 1.5) for a, b in win))
