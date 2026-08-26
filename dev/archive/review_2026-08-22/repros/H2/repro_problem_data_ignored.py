"""Problem.objective / local_model: `data=` is silently replaced by self.data when `sample` is omitted."""
import numpy as np, t3toolbox as t3t
import t3toolbox.backend.optimizers as bopt, t3toolbox.backend.fitting as bfit, t3toolbox.backend.geometry as bgeo
np.random.seed(0)
shape = (5, 6, 7)
x = t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
ww = [np.random.randn(8, N) for N in shape]
data = x.apply(ww) + 0.1 * np.random.randn(8)
prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, data)
x0 = t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
x0 = x0.left_orthogonalize() if hasattr(x0, 'left_orthogonalize') else x0
other = data + 5.0
f_train = float(prob.objective(x0.data))
f_other_expected = 0.5 * float(np.sum((x0.apply(ww) - other) ** 2))
f_other_actual = float(prob.objective(x0.data, data=other))
print('objective on training data        :', f_train)
print('objective(x, data=other) expected :', f_other_expected)
print('objective(x, data=other) actual   :', f_other_actual, '  <- equals training value?', np.isclose(f_other_actual, f_train))
lm = prob.local_model(x0.data, data=other)
print('local_model(x, data=other).objective:', float(lm.objective), ' (expected', f_other_expected, ')')
print('--- sample given, data omitted:')
try:
    prob.objective(x0.data, sample=ww)
except Exception as e:
    print(type(e).__name__, ':', str(e)[:120])
