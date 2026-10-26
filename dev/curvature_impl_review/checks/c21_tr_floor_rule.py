"""Read-only plan check (v3, m22): the v3 trust-region floor rule on the two degenerate cases.
v3 rule, tested ONLY on the rejection branch (after Delta was shrunk): (1) 'objective_floor' iff
|pred| <= C_RES * eps_mach * ||w*y|| * ||w*r|| (f's evaluation resolution, from kind.sumsq -- geometry/T3-agnostic);
(2) else 'trust_region_floor' iff Delta < 1e-12 * Delta0 (the run's own initial scale; Delta and Delta0 share units). Reports |pred|/res at every rejection to calibrate c.
Cases: (A) E2's noise-free over-ranked zero-padded restart (f ~ 1e-22); (B) a near-stationary misspecified start
(rank-2 fit of a rank-3 target after 15 GN steps, |g|/|g_ref| ~ 1e-6, f ~ 4e2)."""
import sys, os, math, io, contextlib
sys.path.insert(0, __import__("os").getcwd())
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(SCR, 'check_restart_tr2.py')).read()
ns = {}
exec(compile(src[:src.index('def scenario(')], 'check_restart_tr2_defs', 'exec'), ns)
np, t3, t3m, topt, bopt, bgeo, bfit, cw = ns['np'], ns['t3'], ns['t3m'], ns['topt'], ns['bopt'], ns['bgeo'], ns['bfit'], ns['cw']
Lift, exact_curv, steihaug = ns['Lift'], ns['exact_curv'], ns['steihaug']
EPS = np.finfo(float).eps
C_RES = 10.0

def run_v3(problem, ww, x, max_outer=40, gtol=1e-12, g0=None, floor_rel=1e-12):
    kind = problem.kind; n_w = kind.w_axes(problem.sample)
    y_norm = math.sqrt(float(kind.sumsq(problem.data, n_w)))
    radius = None; acc_last = True; it = 0; rej = 0; log = []; stop = 'max_newton'
    while it < max_outer:
        if acc_last:
            lm = problem.local_model(x); g = lm.gradient; gnorm = math.sqrt(cw.corewise_dot(g, g))
            g0 = gnorm if g0 is None else g0
            f = float(lm.misfit); r_norm = math.sqrt(2 * f)
            Xn = math.sqrt(float(cw.corewise_dot(lm.frame[2][-1], lm.frame[2][-1])))
            res = EPS * y_norm * r_norm
            lift_obj = Lift(lm.frame, 1e-6 * Xn); curv = exact_curv(lm, ww, lift_obj, g)
            hvp = lambda p: cw.corewise_add(lm.hvp(p), curv(p))
        if gnorm <= gtol * g0:
            stop = 'gradient'; break
        if radius is None:
            radius = gnorm ** 3 / max(float(lm.gn_quadratic(g)), 1e-300); radius0 = radius
        eta = min(0.5, gnorm / g0)
        p, cgi, hit, negc = steihaug(hvp, cw.corewise_scale(g, -1.0), eta * gnorm, 80, radius)
        pred = cw.corewise_dot(g, p) + 0.5 * cw.corewise_dot(p, hvp(p))
        f_new = float(problem.objective(lm.retract(p)))
        rho = (f_new - f) / pred if pred < 0 else float('nan')
        pn = math.sqrt(cw.corewise_dot(p, p))
        accept = math.isfinite(rho) and rho > 0.1
        log.append((it, gnorm / g0, radius, rho, accept, abs(pred) / res))
        if (not math.isfinite(rho)) or rho < 0.25:
            radius = 0.25 * pn
        elif rho > 0.75 and hit:
            radius = min(2 * radius, 1e4 * radius0)
        it += 1
        if accept:
            x = lm.retract(p); acc_last = True
        else:
            rej += 1; acc_last = False
            if abs(pred) <= C_RES * res:                                    # tested ONLY after a rejection:
                stop = 'objective_floor'; break                             #   the model change is below f's resolution
            if radius < floor_rel * radius0:                                # below the run's own initial scale (HS units)
                stop = 'trust_region_floor'; break
    return dict(stop=stop, it=it, rej=rej, log=log, final=gnorm / g0, f=f)

def show(label, r):
    print(f"{label}: stop={r['stop']} outer={r['it']} rejected={r['rej']} final |g|/g0={r['final']:.1e} f={r['f']:.3e}")
    for it, e, rad, rho, acc, pr in r['log']:
        print(f"    it {it:2d} e {e:.1e} Delta {rad:.1e} rho {rho:9.3f} {'acc' if acc else 'REJ'} |pred|/res {pr:.1e}")

# (A) E2's degenerate case (critic's check_objective_floor.py set-up)
rng = np.random.default_rng(7); np.random.seed(7)
shape = (6, 6, 6)
target = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
W = 400
ww = [rng.standard_normal((W, N)) for N in shape]
data = np.asarray(target.apply(ww))
x1 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
with contextlib.redirect_stdout(io.StringIO()):
    x1, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x1, max_newton=80, gtol_rel=1e-10)
x2 = x1.resize(shape, (3, 3, 3), (1, 3, 3, 1))
prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
show('(A) noise-free over-ranked restart', run_v3(prob, ww, x2.data, gtol=1e-8))

# (B) near-stationary misspecified start
rng = np.random.default_rng(1); np.random.seed(1)
tgt = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
W = 200
ww = [rng.standard_normal((W, N)) for N in shape]
clean = np.asarray(tgt.apply(ww)); data = clean + 0.01 * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
x0 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
with contextlib.redirect_stdout(io.StringIO()):
    x15, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=15, gtol_rel=0.0)
prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
show('(B) near-stationary misspecified start', run_v3(prob, ww, x15.data, gtol=1e-12))
