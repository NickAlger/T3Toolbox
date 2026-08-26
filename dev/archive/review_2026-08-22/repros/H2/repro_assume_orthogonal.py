"""Is t3svd(assume_orthogonal=) honoured? Truncate a NON-orthogonal input with True vs False."""
import numpy as np, t3toolbox as t3t, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.t3_svd as bsvd, t3toolbox.backend.ut3_svd as busvd
np.random.seed(0)
shape = (5, 6, 7)
x = t3t.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1)) + t3t.TuckerTensorTrain.randn(shape, (2, 2, 3), (1, 3, 2, 1))
x = x * 1.0 if not hasattr(x, 'scale') else x
print('input right-orthogonal?', bool(x.is_right_orthogonal()), ' left?', bool(x.is_left_orthogonal()))
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
xd = x.to_dense()
for kw in (dict(max_tt_ranks=2), dict(max_tucker_ranks=2), dict()):
    aT, sT, tT = x.t3svd(assume_orthogonal=True, **kw); aF, sF, tF = x.t3svd(assume_orthogonal=False, **kw)
    uT, usT, utT = ux.t3svd(assume_orthogonal=True, **kw); uF, usF, utF = ux.t3svd(assume_orthogonal=False, **kw)
    print(kw, 'ragged: True-vs-False dense diff %.1e ; err vs dense True %.1e False %.1e ; tt svals equal? %s' % (
        np.linalg.norm(aT.to_dense() - aF.to_dense()), np.linalg.norm(aT.to_dense() - xd) / np.linalg.norm(xd), np.linalg.norm(aF.to_dense() - xd) / np.linalg.norm(xd),
        all(np.allclose(a, b) for a, b in zip(tT, tF))))
    print(kw, 'uniform: True-vs-False dense diff %.1e ; err vs dense True %.1e False %.1e ; tt svals equal? %s' % (
        np.linalg.norm(uT.to_dense() - uF.to_dense()), np.linalg.norm(uT.to_dense() - xd) / np.linalg.norm(xd), np.linalg.norm(uF.to_dense() - xd) / np.linalg.norm(xd),
        np.allclose(np.asarray(utT), np.asarray(utF))))
# Backend directly
import inspect
print('backend t3svd sig:', inspect.signature(bsvd.t3svd))
src = inspect.getsource(bsvd.t3svd)
import re
print('assume_orthogonal referenced in backend t3svd body:', [l.strip() for l in src.splitlines() if 'assume_orthogonal' in l and 'def' not in l][:6])
srcu = inspect.getsource(busvd.ut3svd)
print('assume_orthogonal referenced in backend ut3svd body:', [l.strip() for l in srcu.splitlines() if 'assume_orthogonal' in l][:6])
