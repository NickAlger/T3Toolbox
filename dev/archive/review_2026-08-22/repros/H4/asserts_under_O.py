"""Structural-error cases that are guarded only by `assert` -- run with and without python -O."""
import sys, numpy as np
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3
import t3toolbox.corewise as cw
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.t3_linalg as t3_linalg
import t3toolbox.backend.t3_operations as t3_ops
print('python -O active (asserts stripped):', not __debug__)

def case(name, f):
    try:
        r = f()
        print('%-40s -> RETURNED %s' % (name, (getattr(r, 'shape', None), type(r).__name__) if not isinstance(r, tuple) else 'tuple len %d: %s' % (len(r), [getattr(a,'shape',a) for a in r][:4])))
    except Exception as e:
        print('%-40s -> %s: %s' % (name, type(e).__name__, (str(e).splitlines() or [''])[0][:100]))

np.random.seed(0)
x = T3.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
y = T3.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1), stack_shape=(3,))
# 1. corewise_add with length-mismatched trees (zip truncates under -O)
case('corewise_add len mismatch', lambda: cw.corewise_add((np.ones(2), np.ones(2), np.ones(2)), (np.ones(2), np.ones(2))))
case('corewise_sub len mismatch', lambda: cw.corewise_sub((np.ones(2), np.ones(2), np.ones(2)), (np.ones(2), np.ones(2))))
case('corewise_dot len mismatch', lambda: cw.corewise_dot((np.ones(2), np.ones(2), np.ones(2)), (np.ones(2), np.ones(2))))
# 2. compute_minimal_ranks with a wrong-length tt_ranks
case('compute_minimal_ranks bad tt len', lambda: ranks.compute_minimal_ranks((4, 5, 6), (2, 2, 2), (1, 2, 1)))
case('compute_continuation_ranks bad len', lambda: ranks.compute_continuation_ranks((4, 5, 6), (np.ones(2), np.ones(2), np.ones(2)), (np.ones(1), np.ones(2), np.ones(1))))
# 3. t3_add with mismatched stack shapes
case('t3_add stack mismatch', lambda: t3_linalg.t3_add(x.data, y.data))
case('T3 + T3 stack mismatch (frontend)', lambda: x + y)
# 4. T3 * dense array of the wrong shape
case('x * ndarray wrong shape', lambda: x * np.ones((4, 5, 1)))
case('x * ndarray wrong shape (2)', lambda: x * np.ones((4, 5)))
# 5. t3_sum axis out of range
case('t3_sum axis out of range', lambda: t3_ops.t3_sum(x.data, axis=5))
# 6. dense_probe with inconsistent vectors
import t3toolbox.backend.probing as probing
case('dense_probe inconsistent W', lambda: probing.dense_probe(x.to_dense(), (np.ones((2, 4)), np.ones((3, 5)), np.ones((2, 6)))))
