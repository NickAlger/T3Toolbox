import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m, t3toolbox.uniform_manifold as ut3m, t3toolbox.fitting as fitting, t3toolbox.optimizers as topt
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
ww = [np.random.randn(12, N) for N in (4, 5, 6)]
r = [np.random.randn(12, N) for N in (4, 5, 6)]
# (1) GaussNewtonModel.jacobian docstring: "probe: len=d, elm_shape=W+C+(Ni,)" -- uniform returns PACKED
m = fitting.probe_model(t3m.MANIFOLD, x, ww, r)
um = fitting.probe_model(ut3m.UNIFORM_MANIFOLD, ut3.UniformTuckerTensorTrain.from_t3(x), ww, r)
J = m.jacobian(t3m.MANIFOLD.randn(m.frame)); uJ = um.jacobian(ut3m.UNIFORM_MANIFOLD.randn(um.frame))
print('ragged jacobian (probe):', type(J).__name__, [a.shape for a in J])
print('uniform jacobian (probe):', type(uJ).__name__, uJ.shape, ' (docstring: len=d, elm_shape=W+C+(Ni,))')
print('uniform residual field:', type(um.residual).__name__, um.residual.shape, '(field comment: len=d, W+C+(Ni,) (probe))')
# (2) gradient_descent has no use_jit
try:
    topt.gradient_descent(t3m.MANIFOLD, 'apply', ww, x.apply(ww), x, use_jit=True, n_iter=2)
except TypeError as e:
    print('gradient_descent(use_jit=True):', type(e).__name__, str(e)[:100])
