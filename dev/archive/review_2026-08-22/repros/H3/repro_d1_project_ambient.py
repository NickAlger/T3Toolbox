"""H3-2: MANIFOLD.project_ambient(frame, T3 grad) and transport crash for d = 1 (IndexError in tt_zipper on an empty core tuple)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((6,), (3,), (1, 1)); g = t3.TuckerTensorTrain.randn((6,), (3,), (1, 1))
frame = bvf.t3_orthogonal_representations(x)[0]
print('d=1 project_ambient(frame, dense grad) ->', t3m.MANIFOLD.project_ambient(frame, g.to_dense()).to_dense().shape)
for nm, fn in [('project_ambient(frame, T3 grad)', lambda: t3m.MANIFOLD.project_ambient(frame, g)),
               ('transport(v, frame)', lambda: t3m.MANIFOLD.transport(t3m.MANIFOLD.randn(frame), frame)),
               ('retract', lambda: t3m.MANIFOLD.retract(t3m.MANIFOLD.randn(frame)))]:
    try:
        r = fn(); print('d=1', nm, 'OK')
    except Exception as e:
        import traceback; print('d=1', nm, 'RAISES', type(e).__name__, e); traceback.print_exc(limit=-2)
# stacked d=1 too
xs = t3.TuckerTensorTrain.randn((6,), (3,), (1, 1), stack_shape=(3,))
fs = bvf.t3_orthogonal_representations(xs)[0]
try:
    t3m.MANIFOLD.project_ambient(fs, xs); print('d=1 stacked OK')
except Exception as e:
    print('d=1 stacked C=(3,) project_ambient RAISES', type(e).__name__, e)
