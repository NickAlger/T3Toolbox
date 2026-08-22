"""R7: characterize the known finding -- regularized Problem.objective at a raw (non-left-orthogonal) x.
Which geometries / paths / optimizers are affected, and what the user sees."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.optimizers as topt
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf
import t3toolbox.backend.tv_operations as tops
from t3toolbox.backend.t3_orthogonalization import t3_orthogonality_residual
import t3toolbox.shared_geometry as sg

np.random.seed(0)
shape, tr, rr = (4, 5, 6), (2, 3, 2), (1, 2, 3, 1)   # asymmetric
A = t3.TuckerTensorTrain.randn(shape, tr, rr)
ww = [np.random.randn(40, N) for N in shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
b = A.apply(ww)
lam = 0.3
reg = bopt.IdentityRegularizer(lam)

def truth(x, geom_kind):
    r = x.apply(ww) - b
    misfit = 0.5 * float(np.sum(r ** 2))
    if geom_kind == 'manifold':
        rho = 0.5 * lam * float(np.sum(x.to_dense() ** 2))
    else:
        rho = 0.5 * lam * float(sum(np.sum(c ** 2) for c in x.data[0]) + sum(np.sum(c ** 2) for c in x.data[1]))
    return misfit + rho, misfit, rho

x_raw = t3.TuckerTensorTrain.randn(shape, tr, rr) * 3.0   # raw randn: NOT left-orthogonal
print("x_raw left-orth residual:", float(t3_orthogonality_residual(x_raw.data, 'left')))

# (a) ragged manifold: Problem.objective at the raw point
P = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b, regularizer=reg)
got = float(P.objective(x_raw.data))
exp, mis, rho = truth(x_raw, 'manifold')
print("\n[ragged MANIFOLD] Problem.objective(raw x0) = %.6f   truth = %.6f   (misfit %.6f, rho %.6f)  WRONG=%s"
      % (got, exp, mis, rho, not np.isclose(got, exp)))
print("   reg term the code used: %.6f  (= 0.5*lam*||last TT core||^2 = %.6f)"
      % (got - mis, 0.5 * lam * float(np.sum(x_raw.data[1][-1] ** 2))))
lm = P.local_model(x_raw.data)
print("   LocalModel.objective at the same raw point (via the frame) = %.6f  -> correct=%s"
      % (float(lm.objective), np.isclose(float(lm.objective), exp)))

# (b) ragged corewise: exact for any cores
Pc = bopt.least_squares_problem(bgeo.CorewiseGeometryOps(), bfit.APPLY, ww, b, regularizer=reg)
got = float(Pc.objective(x_raw.data)); exp = truth(x_raw, 'corewise')[0]
print("[ragged COREWISE] Problem.objective(raw x0) = %.6f truth = %.6f  correct=%s" % (got, exp, np.isclose(got, exp)))

# (c) uniform manifold: orthogonalizes first -> correct
ux = uf.uniform_minimal(ut3.UniformTuckerTensorTrain.from_t3(x_raw))
Pu = uf.uniform_least_squares_problem('manifold', 'apply', ux, ww, b, regularizer=reg)
got = float(Pu.objective((ux.tucker_supercore, ux.tt_supercore))); exp = truth(x_raw, 'manifold')[0]
print("[uniform MANIFOLD] Problem.objective(raw x0) = %.6f truth = %.6f  correct=%s" % (got, exp, np.isclose(got, exp)))

# (d) after a retraction (the line-search points): left-orthogonal -> correct?
frame = P.geom.frame(x_raw.data)
v = t3m.MANIFOLD.randn(t3m.MANIFOLD.frame(x_raw)).variations.data
x_ret = P.geom.retract(frame, v)
print("\n[retract output] left-orth residual = %.2e" % float(t3_orthogonality_residual(x_ret, 'left')))
got = float(P.objective(x_ret)); exp = truth(t3.TuckerTensorTrain(*x_ret), 'manifold')[0]
print("   Problem.objective(retract out) = %.6f truth = %.6f  correct=%s" % (got, exp, np.isclose(got, exp)))

# (e) SHARED manifold retraction output: is it left-orthogonal too?
sharing = (0, 0, 1)
As = t3.TuckerTensorTrain.randn((5, 5, 4), (2, 2, 2), (1, 2, 2, 1)).share(sharing)
wws = [np.random.randn(40, N) for N in As.shape]
bs = As.apply(wws)
gs = bgeo.ManifoldGeometryOps().with_sharing(sharing, As.shape)
Ps = bopt.least_squares_problem(gs, bfit.APPLY, wws, bs, regularizer=reg)
xs0 = t3.TuckerTensorTrain.randn((5, 5, 4), (2, 2, 2), (1, 2, 2, 1)).share(sharing)
fr = gs.frame(xs0.data); aux = gs.precompute(fr)
vs = gs.project(fr, t3m.MANIFOLD.randn(t3m.MANIFOLD.frame(xs0)).variations.data, aux=aux)
xs_ret = gs.retract(fr, vs, aux=aux)
print("[SHARED manifold retract output] left-orth residual = %.2e" % float(t3_orthogonality_residual(xs_ret, 'left')))
xs_ret_t3 = t3.TuckerTensorTrain(*xs_ret)
got = float(Ps.objective(xs_ret)); r = xs_ret_t3.apply(wws) - bs
exp = 0.5 * float(np.sum(r ** 2)) + 0.5 * lam * float(np.sum(xs_ret_t3.to_dense() ** 2))
print("   Problem.objective(shared retract out) = %.6f truth = %.6f  correct=%s" % (got, exp, np.isclose(got, exp)))

# (f) which OPTIMIZERS record a wrong number from a raw x0?
print("\n--- optimizers from the raw x0 (ragged MANIFOLD + IdentityRegularizer) ---")
exp0 = truth(x_raw, 'manifold')[0]
xo, st = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, b, x_raw, regularizer=topt.IdentityRegularizer(lam), max_newton=3)
print("newton_cg  history[0].objective = %.6f  truth %.6f  correct=%s" % (st['history'][0]['objective'], exp0, np.isclose(st['history'][0]['objective'], exp0)))
xo, st = topt.gradient_descent(t3m.MANIFOLD, 'apply', ww, b, x_raw, regularizer=topt.IdentityRegularizer(lam), n_iter=3)
print("grad_desc  losses[0]            = %.6f  truth %.6f  correct=%s" % (st['losses'][0], exp0, np.isclose(st['losses'][0], exp0)))
rng = np.random.default_rng(0)
xo, st = topt.mc_sgd(t3m.MANIFOLD, 'apply', ww, b, x_raw, rng, 10, regularizer=topt.IdentityRegularizer(lam), max_iter=4, check_every=2)
# losses[0] is at x after 2 retractions -> left-orthogonal
exp_mc = truth(xo, 'manifold')[0]
print("mc_sgd     losses[-1] (at returned x) = %.6f  truth-at-returned %.6f  correct=%s" % (st['losses'][-1], exp_mc, np.isclose(st['losses'][-1], exp_mc)))
# But: the mc_sgd stop check objective is the smoothed value; compare raw last check
print("   (mc_sgd 'losses' are EMA-smoothed; len=%d, values=%s)" % (len(st['losses']), st['losses']))

# (g) frontend GaussNewtonModel objective_value at the raw point (frame-based)
import t3toolbox.fitting as fitting
m = fitting.apply_model(t3m.MANIFOLD, x_raw, ww, x_raw.apply(ww) - b, regularizer=topt.IdentityRegularizer(lam))
print("frontend GaussNewtonModel.objective_value(raw x0) = %.6f truth %.6f correct=%s" % (float(m.objective_value), exp0, np.isclose(float(m.objective_value), exp0)))
