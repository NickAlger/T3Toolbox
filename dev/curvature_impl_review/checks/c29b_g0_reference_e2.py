"""Read-only plan check (v5): the g0_reference interplay on E2's three apply restart scenarios (critic's
check_restart_tr2 harness, exact eps = 1e-6|X|, forcing power 1.0 as in E2): zero-tensor reference vs warm start."""
import sys, os, math, io, contextlib
sys.path.insert(0, __import__("os").getcwd())
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(SCR, 'check_restart_tr2.py')).read()
ns = {}
exec(compile(src[:src.index('def scenario(')], 'check_restart_tr2_defs', 'exec'), ns)
np, t3, t3m, topt, bopt, bgeo, bfit, cw, run = (ns[k] for k in ('np', 't3', 't3m', 'topt', 'bopt', 'bgeo', 'bfit', 'cw', 'run'))
for label, ttk, ttt, l1k, l1t, l2k, l2t, noise in (
        ('A above target', (2, 2, 2), (1, 2, 2, 1), (2, 2, 2), (1, 2, 2, 1), (3, 3, 3), (1, 3, 3, 1), 0.05),
        ('B noise-free', (2, 2, 2), (1, 2, 2, 1), (2, 2, 2), (1, 2, 2, 1), (3, 3, 3), (1, 3, 3, 1), 0.0),
        ('C below target', (4, 4, 4), (1, 4, 4, 1), (2, 2, 2), (1, 2, 2, 1), (3, 3, 3), (1, 3, 3, 1), 0.3)):
    rng = np.random.default_rng(7); np.random.seed(7)
    shape, W = (6, 6, 6), 400
    target = t3.TuckerTensorTrain.randn(shape, ttk, ttt)
    ww = [rng.standard_normal((W, N)) for N in shape]
    clean = np.asarray(target.apply(ww))
    data = clean + noise * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
    x1 = t3.TuckerTensorTrain.randn(shape, l1k, l1t)
    with contextlib.redirect_stdout(io.StringIO()):
        x1, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x1, max_newton=80, gtol_rel=1e-10)
    x2 = x1.resize(shape, l2k, l2t)
    geom = bgeo.ManifoldGeometryOps()
    problem = bopt.least_squares_problem(geom, bfit.APPLY, ww, data)
    gz = problem.local_model(cw.corewise_zeros_like(x2.data)).gradient; zref = float(geom.inner(gz, gz)) ** 0.5
    gw = problem.local_model(x2.data).gradient; wref = float(geom.inner(gw, gw)) ** 0.5
    print(f"E2 {label}: zero ref {zref:.3e}, warm ref {wref:.3e} (warm/zero {wref / zref:.1e})")
    for name, ref in (('warm', None), ('zero', zref)):
        for gtol in (1e-8, 1e-3):
            _, st = bopt.newton_cg(problem, x2.data, max_newton=80, gtol_rel=gtol, g0norm_newton=ref)
            r = run(problem, ww, x2.data, 'exact', 'rel1e-6', max_outer=80, gtol=gtol, g0=ref)
            print(f"   ref {name} gtol {gtol:g}: GN+Armijo newton {st['newton']:3d} conv {st['history'][-1]['converged']} | "
                  f"exact+TR outer {r['iters']:3d} rej {r['rejected']:2d} hvps {r['hvps']:4d} conv {r['converged']}")
