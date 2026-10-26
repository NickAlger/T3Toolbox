"""Read-only plan check (v6, n19): T16 at power 0.5, seeds 1 and 2 -- per accepted step, e_k, |pred|/res and the
rate constant e_{k+1}/e_k^1.5: does the n15 criterion (|pred| > 1e3*res) separate the resolution-floor steps?"""
import sys, os, math, io, contextlib
sys.path.insert(0, __import__("os").getcwd())
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(SCR, 'check_restart_tr2.py')).read()
ns = {}
exec(compile(src[:src.index('def scenario(')], 'check_restart_tr2_defs', 'exec'), ns)
np, t3, t3m, topt, bopt, bgeo, bfit, cw = (ns[k] for k in ('np', 't3', 't3m', 'topt', 'bopt', 'bgeo', 'bfit', 'cw'))
Lift, exact_curv, steihaug = ns['Lift'], ns['exact_curv'], ns['steihaug']
power = 0.5
for seed in (1, 2):
    print('seed', seed); prev_e = None
    rng = np.random.default_rng(seed); np.random.seed(seed)
    shape, W = (6, 6, 6), 200
    tgt = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    clean = np.asarray(tgt.apply(ww)); data = clean + 0.01 * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
    x0 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        x, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=15, gtol_rel=0.0)
    x = x.data
    problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
    ynorm = float(np.linalg.norm(data)); eps = np.finfo(float).eps
    g0 = None; radius = None; acc = True
    acc_prev = True
    for it in range(14):
        if acc:
            lm = problem.local_model(x); g = lm.gradient; gn = math.sqrt(cw.corewise_dot(g, g)); f = float(lm.misfit)
            g0 = gn if g0 is None else g0
            Xn = math.sqrt(float(cw.corewise_dot(lm.frame[2][-1], lm.frame[2][-1])))
            curv = exact_curv(lm, ww, Lift(lm.frame, 1e-6 * Xn), g); hvp = lambda p: cw.corewise_add(lm.hvp(p), curv(p))
        if radius is None:
            radius = gn ** 3 / max(float(lm.gn_quadratic(g)), 1e-300); r0 = radius
        eta = min(0.5, (gn / g0) ** power)
        p, cgi, hit, negc = steihaug(hvp, cw.corewise_scale(g, -1.0), eta * gn, 80, radius)
        pred = cw.corewise_dot(g, p) + 0.5 * cw.corewise_dot(p, hvp(p))
        xt = lm.retract(p); fn = float(problem.objective(xt)); rho = (fn - f) / pred if pred < 0 else float('nan')
        res = eps * (ynorm * math.sqrt(2 * f) + abs(f)); pn = math.sqrt(cw.corewise_dot(p, p))
        acc = math.isfinite(rho) and rho > 0.1
        rate = (gn / g0) / prev_e ** 1.5 if (prev_e is not None and acc_prev) else float('nan')
        print(f"it {it:2d} e {gn / g0:.1e} |pred|/res {abs(pred) / res:.1e} rho {rho:.3f} {'ACC' if acc else 'rej'}  rate-to-here e_k/e_(k-1)^1.5 {rate:.2e}")
        prev_e, acc_prev = gn / g0, acc
        if (not math.isfinite(rho)) or rho < 0.25:
            radius = 0.25 * pn
        elif rho > 0.75 and hit:
            radius = min(2 * radius, 1e4 * r0)
        if acc:
            x = xt
        elif abs(pred) <= 10 * res:
            print('  -> objective_floor'); break
