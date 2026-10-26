"""Read-only plan check (v5, I-18a): does a default g0_reference='zero' break the ragged == uniform TRAJECTORY
equivalence the O2 oracle sweep (1e-7) and test_optimizers_frontend U7a assert? Simulated with today's code by
passing each layer's OWN zero-tensor reference as g0norm_newton (which CG inherits). Same nonzero start both layers."""
import sys, io, contextlib; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m, t3toolbox.uniform_manifold as ut3m
import t3toolbox.optimizers as topt, t3toolbox.corewise as cw
rng = np.random.default_rng(5); np.random.seed(5)
shape, tk, tt, W = (5, 6, 7), (2, 3, 2), (1, 2, 2, 1), 60
ww = [rng.standard_normal((W, N)) for N in shape]
data = np.asarray(t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1)).apply(ww))
x0 = t3.TuckerTensorTrain.randn(shape, tk, tt)
ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)
def zero_ref(geom, x):
    problem, init, _ = topt._setup(geom, 'apply', ww, data, x)
    g = problem.local_model(cw.corewise_zeros_like(init)).gradient
    return float(problem.geom.inner(g, g)) ** 0.5
zr, zu = zero_ref(t3m.MANIFOLD, x0), zero_ref(ut3m.UNIFORM_MANIFOLD, ux0)
print(f"zero reference: ragged {zr:.4e}, uniform {zu:.4e} (ratio {zu / zr:.3f})")
for label, kr, ku in (('warm start (today)', {}, {}), ('zero (own layer)', dict(g0norm_newton=zr), dict(g0norm_newton=zu))):
    _, sr = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, data, x0, max_newton=3, **kr)
    _, su = topt.newton_cg(ut3m.UNIFORM_MANIFOLD, 'apply', ww, data, ux0, max_newton=3, **ku)
    lr, lu = np.array(sr['losses']), np.array(su['losses'])
    cgr = [h['cg_iters'] for h in sr['history']]; cgu = [h['cg_iters'] for h in su['history']]
    print(f"  {label:20s}: losses rel diff {np.max(np.abs(lr - lu) / np.abs(lr)):.1e}; cg iters ragged {cgr} uniform {cgu}")
