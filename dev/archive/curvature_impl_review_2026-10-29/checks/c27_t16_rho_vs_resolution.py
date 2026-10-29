"""Read-only plan check (v4, n15): on T16's own problem, the per-step rho and |pred|/res (res = eps(dtype) *
(||w*y|| ||w*r|| + |f|)), to decide the step filter for T16's rho assertion. Reuses c21's v3 TR loop (run_v3)."""
import sys, os, math, io, contextlib
sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import c21_tr_floor_rule as c21           # runs its two demo cases silently on import
np, t3, t3m, topt, bopt, bgeo, bfit = c21.np, c21.t3, c21.t3m, c21.topt, c21.bopt, c21.bgeo, c21.bfit
for seed in (1, 2):
    rng = np.random.default_rng(seed); np.random.seed(seed)
    shape, W = (6, 6, 6), 200
    tgt = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
    ww = [rng.standard_normal((W, N)) for N in shape]
    clean = np.asarray(tgt.apply(ww)); data = clean + 0.01 * np.linalg.norm(clean) / math.sqrt(W) * rng.standard_normal(W)
    x0 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        x15, _ = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=15, gtol_rel=0.0)
    prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
    r = c21.run_v3(prob, ww, x15.data, gtol=1e-12)
    print(f"seed {seed}: stop={r['stop']}")
    for it, e, rad, rho, acc, pr in r['log']:
        print(f"    it {it:2d} e {e:.1e} rho {rho:8.3f} {'acc' if acc else 'REJ'} |pred|/res {pr:.1e}")
