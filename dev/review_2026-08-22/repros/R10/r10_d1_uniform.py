"""R10: d=1 uniform norm / weighted_norm."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
x = t3.TuckerTensorTrain.randn((9,), (1,), (1, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
for label, f in (('ux.norm()', lambda: ux.norm()), ('ux.inner(ux)', lambda: ux.inner(ux) if hasattr(ux, 'inner') else ux.inner_product(ux)),
                 ('ut3_weighted_norm', lambda: ut3.ut3_weighted_norm(ux, ut3.UT3Weights.from_ut3svd(ux))),
                 ('ut3_weighted_norm(no orth)', lambda: ut3.ut3_weighted_norm(ux, ut3.UT3Weights.from_ut3svd(ux), use_orthogonalization=False)),
                 ('ux.t3svd()', lambda: ux.t3svd()[0].n), ('ragged x.norm()', lambda: x.norm())):
    try: print('%-30s OK ->' % label, f())
    except Exception as e: print('%-30s RAISED %s: %s' % (label, type(e).__name__, str(e).splitlines()[0][:110]))
# padded-above d=1
ux2 = ut3.UniformTuckerTensorTrain.from_t3(x, n=3, r=2)
for label, f in (('padded ux2.norm()', lambda: ux2.norm()), ('padded ut3_weighted_norm', lambda: ut3.ut3_weighted_norm(ux2, ut3.UT3Weights.from_t3weights(t3.T3Weights.from_t3svd(x), n=3, r=2)))):
    try: print('%-30s OK ->' % label, f())
    except Exception as e: print('%-30s RAISED %s: %s' % (label, type(e).__name__, str(e).splitlines()[0][:110]))
