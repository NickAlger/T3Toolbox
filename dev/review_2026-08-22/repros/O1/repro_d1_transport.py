import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5,), (3,), (1, 1))
frame, _ = bvf.t3_orthogonal_representations(x)
v = t3m.MANIFOLD.randn(frame)
g = t3.TuckerTensorTrain.randn((5,), (4,), (1, 1))
for name, fn in [('project_ambient(frame, T3)', lambda: t3m.MANIFOLD.project_ambient(frame, g)),
                 ('project_ambient(frame, dense)', lambda: t3m.MANIFOLD.project_ambient(frame, np.random.randn(5))),
                 ('transport(v, frame)', lambda: t3m.MANIFOLD.transport(v, frame)),
                 ('v.to_t3()', lambda: v.to_t3()),
                 ('retract(v)', lambda: t3m.MANIFOLD.retract(v))]:
    try:
        r = fn(); print(name, 'OK', type(r).__name__)
    except Exception as e:
        print(name, 'RAISES'); traceback.print_exc(limit=4)
