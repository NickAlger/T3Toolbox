import numpy as np, t3toolbox as t3t, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.t3_orthogonalization as bo
np.random.seed(0)
shape = (5, 6, 7)
for label, x in (('randn', t3t.TuckerTensorTrain.randn(shape, (3, 4, 3), (1, 3, 3, 1))),
                 ('sum  ', t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)) + t3t.TuckerTensorTrain.randn(shape, (2, 2, 3), (1, 3, 2, 1)))):
    for dr, side in (('right_to_left', 'right'), ('left_to_right', 'left')):
        a = x.rank_adjustment_sweep(dr)
        res = bo.t3_orthogonality_residual(a.data, side)
        print(label, dr, 'is_%s_orthogonal' % side, bool(getattr(a, 'is_%s_orthogonal' % side)()), 'residual(%s) =' % side, np.asarray(res).round(3).tolist() if np.ndim(res) else float(res), 'ranks', a.tucker_ranks, a.tt_ranks, 'in ranks', x.tucker_ranks, x.tt_ranks)
        ua = ut3.UniformTuckerTensorTrain.from_t3(x).rank_adjustment_sweep(dr)
        print(label, dr, 'uniform is_%s_orthogonal' % side, bool(getattr(ua, 'is_%s_orthogonal' % side)()), 'err vs dense %.1e' % (np.linalg.norm(ua.to_dense() - x.to_dense()) / np.linalg.norm(x.to_dense())))
