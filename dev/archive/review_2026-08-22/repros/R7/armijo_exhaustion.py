"""R7: Armijo loop exhausting its backtrack budget and accepting x_trial silently (newton_cg: 40; gradient_descent: 50).
Supported path: a warm-started / continuation run whose Newton stop is pinned loose (g0norm_newton large, or a tiny
gtol_rel) keeps iterating at the machine-precision floor."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit

np.random.seed(0)
shape, tr, rr = (4, 5, 6), (2, 2, 2), (1, 2, 2, 1)
A = t3.TuckerTensorTrain.randn(shape, tr, rr)
ww = [np.random.randn(80, N) for N in shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
b = A.apply(ww)
x0 = t3.TuckerTensorTrain.zeros(shape, tr, rr)
P = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b)
x_conv, st0 = bopt.newton_cg(P, x0.data, max_newton=20)          # converge first (a 'warm start')
print("warm start: converged in %d Newton steps, final gnorm %.2e" % (st0['newton'], st0['history'][-1]['gnorm']))

n_obj = {'n': 0}
orig_obj = bopt.Problem.objective
def counting_objective(self, *a, **k):
    n_obj['n'] += 1
    return orig_obj(self, *a, **k)
bopt.Problem.objective = counting_objective
x, st = bopt.newton_cg(P, x0.data, max_newton=16, gtol_rel=1e-30)   # a stop that never triggers -> runs into the precision floor
bopt.Problem.objective = orig_obj
print("newton_cg (gtol_rel=1e-30): %d Newton iterations, %d Problem.objective evaluations" % (st['newton'], n_obj['n']))
print(" it  ls_steps      alpha      delta_f        gnorm   cg_iters  converged")
for h in st['history']:
    print(" %2d  %8s  %9s  %11s  %11.3e  %8s   %s" % (h['iteration'], h['ls_steps'],
          ('%.3e' % h['alpha']) if h['alpha'] is not None else '-',
          ('%+.3e' % h['delta_f']) if h['delta_f'] is not None else '-', h['gnorm'], h['cg_iters'], h['converged']))
exhausted = [h for h in st['history'] if h['ls_steps'] == 40]
print("iterations with ls_steps == 40 (budget exhausted; x_trial at alpha=2^-40 accepted anyway): %d of %d" % (len(exhausted), st['newton']))
print("accepted an objective INCREASE (delta_f > 0) on an exhausted iteration: %s" % any(h['delta_f'] > 0 for h in exhausted))
print("NewtonInfo has no line-search-failure field; only the tell is ls_steps==40 / alpha=%.1e" % (2.0 ** -40))

n_obj['n'] = 0
bopt.Problem.objective = counting_objective
xg, stg = bopt.gradient_descent(P, x, n_iter=4, gtol_rel=1e-30)   # from the Newton-floored point
bopt.Problem.objective = orig_obj
print("\ngradient_descent (warm start, gtol_rel=1e-30): n_iter=%d, %d Problem.objective evaluations (= %d per iteration)"
      % (stg['n_iter'], n_obj['n'], n_obj['n'] // stg['n_iter']))
print("losses: %s ; stats keys %s -- no record of the exhausted line search at all" % (['%.3e' % v for v in stg['losses']], sorted(stg)))
