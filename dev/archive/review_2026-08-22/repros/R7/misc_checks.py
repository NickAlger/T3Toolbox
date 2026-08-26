"""R7 misc: (a) mc_sgd stats['losses'] are EMA-smoothed, not the raw full-batch losses; (b) uniform newton_cg step_rel
uses a mask-unaware corewise_norm -- does the retracted supercore carry nonzero padding?"""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as um
import t3toolbox.optimizers as topt
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.corewise as cw

np.random.seed(0)
shape, tr, rr = (4, 5, 6), (2, 2, 2), (1, 2, 2, 1)
A = t3.TuckerTensorTrain.randn(shape, tr, rr)
ww = [np.random.randn(60, N) for N in shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
b = A.apply(ww)
x0 = t3.TuckerTensorTrain.randn(shape, tr, rr)
P = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b)

# (a) reconstruct the raw full-batch losses at each check by re-running the same seed and intercepting Problem.objective
raw = []
orig = bopt.Problem.objective
def rec(self, *a, **k):
    v = orig(self, *a, **k); raw.append(float(v)); return v
bopt.Problem.objective = rec
x, st = bopt.mc_sgd(P, x0.data, np.random.default_rng(0), 10, max_iter=60, check_every=10, smooth_tau=2.0)
bopt.Problem.objective = orig
print("mc_sgd stats['losses'] (smoothed): %s" % ['%.4f' % v for v in st['losses']])
print("raw full-batch objective at the checks: %s" % ['%.4f' % v for v in raw])
print("losses[0]==raw[0]: %s ; losses[1:]==raw[1:]: %s" % (np.isclose(st['losses'][0], raw[0]), np.allclose(st['losses'][1:], raw[1:len(st['losses'])])))

# (b) uniform padding after a retraction, and what step_rel sees
ux0 = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1)))   # nU padded to 3
A2 = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1)); b2 = A2.apply(ww)
x, st = topt.newton_cg(um.UNIFORM_MANIFOLD, 'apply', ww, b2, ux0, max_newton=3)
sc = (x.tucker_supercore, x.tt_supercore)
unmasked = float(cw.corewise_norm(sc))
masked = float(cw.corewise_norm(x.to_t3().data))
print("\nuniform result after 3 Newton steps: corewise_norm(supercores) = %.6f ; corewise_norm(masked real cores) = %.6f ; padding nonzero: %s"
      % (unmasked, masked, not np.isclose(unmasked, masked)))
print("   -> history[-1].step_rel (uses the unmasked norm) = %s" % st['history'][-1]['step_rel'])
