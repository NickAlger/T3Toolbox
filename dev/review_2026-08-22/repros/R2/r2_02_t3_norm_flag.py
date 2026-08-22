"""t3_norm(use_orthogonalization=False) still orthogonalizes: it calls t3_inner_product with the default True."""
import numpy as np, t3toolbox as t3
import t3toolbox.backend.t3_linalg as L
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 3, 1), stack_shape=(2,))
calls = {'n': 0}
orig = L.ragged_orth.t3_left_orthogonalize
def counting(*a, **k):
    calls['n'] += 1
    return orig(*a, **k)
L.ragged_orth.t3_left_orthogonalize = counting
for flag in (True, False):
    calls['n'] = 0
    n = L.t3_norm(x.data, use_orthogonalization=flag)
    print('t3_norm(use_orthogonalization=%s): t3_left_orthogonalize called %d time(s); value %s' % (flag, calls['n'], n.round(6)))
calls['n'] = 0
L.t3_inner_product(x.data, x.data, use_orthogonalization=False)
print('t3_inner_product(use_orthogonalization=False): called %d time(s)' % calls['n'])
calls['n'] = 0
x.norm(use_orthogonalization=False)
print('frontend x.norm(use_orthogonalization=False): called %d time(s)' % calls['n'])
print('dense norm:', np.linalg.norm(x.to_dense().reshape(2, -1), axis=1).round(6))
