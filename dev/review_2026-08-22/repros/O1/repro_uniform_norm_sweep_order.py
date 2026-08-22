"""Mimic the sweep's exact call sequence for d2, C=() and print ux.norm() at each step."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from o1_common import *
shape, tr, ttr = STRUCTS['d2']; d = 2; sh = None
for C in [(), (2,)]:
    np.random.seed(1)
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C); X = np.asarray(x.to_dense())
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C); Y = np.asarray(y.to_dense())
    uy = ut3.UniformTuckerTensorTrain.from_t3(y)
    ref = np.linalg.norm(X.reshape(C + (-1,)), -1)
    print('C=%s  ux.norm()=%s  ref=%s  x.norm()=%s' % (C, np.asarray(ux.norm()).round(3).tolist(), ref.round(3).tolist(), np.asarray(x.norm()).round(3).tolist()))
    print('      ux.inner(uy)=%s  ref=%s' % (np.asarray(ux.inner(uy)).round(3).tolist(), np.sum((X * Y).reshape(C + (-1,)), -1).round(3).tolist()))
    print('      ux.norm(False)=%s  sqrt(ux.inner(ux))=%s' % (np.asarray(ux.norm(use_orthogonalization=False)).round(3).tolist(), np.sqrt(np.asarray(ux.inner(ux))).round(3).tolist()))
    frame, var = ubv.ut3_orthogonal_representations(ux)
    v = ut3m.UNIFORM_MANIFOLD.randn(frame); Vd = np.asarray(v.to_dense())
    print('      UNIFORM_MANIFOLD.norm(v)=%s  dense=%s' % (np.asarray(ut3m.UNIFORM_MANIFOLD.norm(v)).round(3).tolist(), np.linalg.norm(Vd.reshape(C + (-1,)), -1).round(3).tolist()))
    uv = ut3m.UNIFORM_MANIFOLD.randn(frame, stack_shape=(2,))
    try:
        t = ut3m.UNIFORM_MANIFOLD.transport(uv, frame); print('      transport(K-stacked) OK', t.tangent_stack_shape)
    except Exception as e:
        import traceback; print('      transport(K-stacked) RAISES', type(e).__name__, str(e)[:90]); print('      ' + '\n      '.join(traceback.format_exc().splitlines()[-8:-1]))
