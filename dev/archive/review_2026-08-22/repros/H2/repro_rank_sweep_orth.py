"""rank_adjustment_sweep(direction) -- is the output orthogonal as documented? ragged and uniform."""
import numpy as np, t3toolbox as t3t, t3toolbox.uniform_tucker_tensor_train as ut3
np.random.seed(0)
shape = (5, 6, 7)
x = t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)) + t3t.TuckerTensorTrain.randn(shape, (2, 2, 3), (1, 3, 2, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x); xd = x.to_dense()
for dr in ('right_to_left', 'left_to_right'):
    a = x.rank_adjustment_sweep(dr); u = ux.rank_adjustment_sweep(dr)
    print(dr, 'ragged : L-orth', bool(a.is_left_orthogonal()), 'R-orth', bool(a.is_right_orthogonal()), 'ranks', a.tucker_ranks, a.tt_ranks, 'err vs dense %.1e' % (np.linalg.norm(a.to_dense() - xd) / np.linalg.norm(xd)))
    print(dr, 'uniform: L-orth', bool(u.is_left_orthogonal()), 'R-orth', bool(u.is_right_orthogonal()), 'ranks', tuple(int(v) for v in u.tucker_ranks), tuple(int(v) for v in u.tt_ranks), 'err vs dense %.1e' % (np.linalg.norm(u.to_dense() - xd) / np.linalg.norm(xd)))
s = x.t3svd()[0]
print('t3svd() result L-orth', bool(s.is_left_orthogonal()), 'R-orth', bool(s.is_right_orthogonal()))
s2 = s.rank_adjustment_sweep('right_to_left')
print('t3svd().rank_adjustment_sweep(r2l): L-orth', bool(s2.is_left_orthogonal()), 'R-orth', bool(s2.is_right_orthogonal()), 'minimal', bool(s2.has_minimal_ranks) if not callable(s2.has_minimal_ranks) else bool(s2.has_minimal_ranks()))
import inspect
print(inspect.signature(t3t.TuckerTensorTrain.is_right_orthogonal))
