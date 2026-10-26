"""Read-only plan check (v3, m28): calibrate T16 on T16's OWN problem. Reuses the critic's exact-TR harness
(SCR/check_restart_tr2.py: library GN pieces + block-sum exact curvature + Steihaug TR) by exec'ing its definitions.
Problem: (6,6,6), fit ranks (2,2,2)/(1,2,2,1), apply, W = 200, data = target + 1% noise, start = 15 GN steps.
We vary the target rank so the residual is nonzero (misspecified), and report per-step e_k = |g_k|/|g_0|."""
import sys, os, math, io, contextlib
sys.path.insert(0, __import__("os").getcwd())
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(SCR, 'check_restart_tr2.py')).read()
ns = {}
exec(compile(src[:src.index('def scenario(')], 'check_restart_tr2_defs', 'exec'), ns)
np, t3, t3m, topt, bopt, bgeo, bfit, run = ns['np'], ns['t3'], ns['t3m'], ns['topt'], ns['bopt'], ns['bgeo'], ns['bfit'], ns['run']

def gn_tail(problem, x, n=40, gtol=1e-12, g0=None):
    with contextlib.redirect_stdout(io.StringIO()):
        _, st = topt.newton_cg(t3m.MANIFOLD, 'apply', problem.sample, problem.data, x, max_newton=n, gtol_rel=gtol,
                               g0norm_newton=g0)
    return [h['gnorm'] for h in st['history']]

for target_rank, seed in [((3, 3, 3), 1), ((4, 4, 4), 1), ((4, 4, 4), 2)]:
    rng = np.random.default_rng(seed); np.random.seed(seed)
    shape, W = (6, 6, 6), 200
    tgt = t3.TuckerTensorTrain.randn(shape, target_rank, (1,) + target_rank[:2] + (1,))
    ww = [rng.standard_normal((W, N)) for N in shape]
    clean = np.asarray(tgt.apply(ww)); data = clean + 0.01 * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
    x0 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        x15, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=15, gtol_rel=0.0)
    problem = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
    lm = problem.local_model(x15.data); g = lm.gradient; g0 = math.sqrt(float(ns['cw'].corewise_dot(g, g)))
    print(f'target {target_rank} seed {seed}: f(x15) = {float(lm.misfit):.3e}, |g0| = {g0:.3e}')
    gn = gn_tail(problem, x15, g0=g0)
    e_gn = [v / g0 for v in gn]
    rat = [b / a for a, b in zip(e_gn, e_gn[1:]) if 1e-10 < a < 1e-1]
    print('   GN+Armijo e_k:', ' '.join('%.1e' % v for v in e_gn[:14]), '...')
    print('   GN linear ratios in window:', ' '.join('%.2f' % v for v in rat[:12]))
    with contextlib.redirect_stdout(io.StringIO()):
        r = run(problem, ww, x15.data, 'exact', 'rel1e-6', max_outer=30, gtol=1e-12, g0=g0)
    e_ex = [h[1] for h in r['hist'] if h[0] != 'COLLAPSE' and h[4]] + [r['final_rel_g']]
    acc = [h for h in r['hist'] if h[0] != 'COLLAPSE']
    print('   exact+TR e_k (accepted):', ' '.join('%.1e' % v for v in e_ex), f"| outer {r['iters']} rejected {r['rejected']}")
    q = [b / a ** 2 for a, b in zip(e_ex, e_ex[1:]) if 1e-10 < a < 1e-1]
    print('   exact quadratic constants e_{k+1}/e_k^2 in window:', ' '.join('%.2e' % v for v in q),
          '| rho on those steps:', ' '.join('%.3f' % h[3] for h in acc if 1e-10 < h[1] < 1e-1))
