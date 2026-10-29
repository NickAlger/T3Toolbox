"""Read-only: the tangent forward sweep at the corewise frame (U,G,G,G) is the core-direction
derivative of the point op (the one-hole companion), for apply and probe."""
import sys; sys.path.insert(0, __import__('os').getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.apply as ba, t3toolbox.backend.probing as bp
import t3toolbox.backend.fv_conversions as fvc
np.random.seed(1)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
th = x.data
dth = (tuple(np.random.randn(*c.shape) for c in th[0]), tuple(np.random.randn(*c.shape) for c in th[1]))
ww = [np.random.randn(4, N) for N in (5, 6, 7)]
cf = fvc.t3_corewise_frame(th)
f = lambda s: ba.t3_apply((tuple(a + s*b for a, b in zip(th[0], dth[0])), tuple(a + s*b for a, b in zip(th[1], dth[1]))), ww)
h = 1e-6
fd = (f(h) - f(-h)) / (2*h)
print('apply: tv_apply at corewise frame == d/ds t3_apply:', np.allclose(ba.tv_apply(ww, dth, cf), fd, rtol=1e-6))
g = lambda s: bp.t3_probe(ww, (tuple(a + s*b for a, b in zip(th[0], dth[0])), tuple(a + s*b for a, b in zip(th[1], dth[1]))))
fdp = [(a - b) / (2*h) for a, b in zip(g(h), g(-h))]
print('probe: tv_probe at corewise frame == d/ds t3_probe:', all(np.allclose(a, b, rtol=1e-6) for a, b in zip(bp.tv_probe(ww, dth, cf), fdp)))
