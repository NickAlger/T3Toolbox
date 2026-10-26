"""Read-only plan check (v5, I-18a): over several seeds/kinds, how often do per-layer zero references (ragged vs
uniform frames at 0 differ) change the CG iteration counts, hence the trajectory, relative to a SHARED reference?"""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m, t3toolbox.uniform_manifold as ut3m
import t3toolbox.optimizers as topt, t3toolbox.corewise as cw
def zero_ref(geom, kind, sample, data, x):
    problem, init, _ = topt._setup(geom, kind, sample, data, x)
    g = problem.local_model(cw.corewise_zeros_like(init)).gradient
    return float(problem.geom.inner(g, g)) ** 0.5
n_div = 0; n = 0
for seed in range(8):
    for kind in ('apply', 'probe'):
        rng = np.random.default_rng(seed); np.random.seed(seed)
        shape, tk, tt, W = (5, 6, 7), (2, 3, 2), (1, 2, 2, 1), 40
        ww = [rng.standard_normal((W, N)) for N in shape]
        tgt = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
        data = np.asarray(tgt.apply(ww)) if kind == 'apply' else [np.asarray(z) for z in tgt.probe(ww)]
        x0 = t3.TuckerTensorTrain.randn(shape, tk, tt); ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)
        zr = zero_ref(t3m.MANIFOLD, kind, ww, data, x0); zu = zero_ref(ut3m.UNIFORM_MANIFOLD, kind, ww, data, ux0)
        _, sr = topt.newton_cg(t3m.MANIFOLD, kind, ww, data, x0, max_newton=6, g0norm_newton=zr)
        _, su = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, kind, ww, data, ux0, max_newton=6, g0norm_newton=zu)
        lr, lu = np.array(sr['losses']), np.array(su['losses'])
        m = min(len(lr), len(lu)); rel = float(np.max(np.abs(lr[:m] - lu[:m]) / np.maximum(np.abs(lr[:m]), 1e-300)))
        cgr = [h['cg_iters'] for h in sr['history']]; cgu = [h['cg_iters'] for h in su['history']]
        n += 1; div = cgr != cgu or rel > 1e-7; n_div += div
        if div:
            print(f"seed {seed} {kind}: zero ref ratio u/r {zu / zr:.3f}; losses rel diff {rel:.1e}; cg {cgr} vs {cgu}")
print(f"{n_div}/{n} cases diverge (cg counts or losses > 1e-7) under per-layer zero references")
