"""R7: make_newton_display / newton_cg(verbose=True) with val_data but no val_sample."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.optimizer_display as bdisp
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit

np.random.seed(0)
shape, tr, rr = (4, 5, 6), (2, 2, 2), (1, 2, 2, 1)
A = t3.TuckerTensorTrain.randn(shape, tr, rr)
ww = [np.random.randn(40, N) for N in shape]
b = A.apply(ww)
vww = [np.random.randn(10, N) for N in shape]
vb = A.apply(vww)
x0 = t3.TuckerTensorTrain.zeros(shape, tr, rr)
P = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b)
for label, fn in [("backend make_newton_display(val_data only)", lambda: bdisp.make_newton_display(P, val_data=vb, print_fn=None)),
                  ("backend make_newton_display(val_sample only)", lambda: bdisp.make_newton_display(P, val_sample=vww, print_fn=None)),
                  ("frontend newton_cg(verbose=True, val_data only)", lambda: topt.newton_cg(t3m.MANIFOLD, 'apply', ww, b, x0, verbose=True, val_data=vb, max_newton=1))]:
    try:
        fn(); print(label, "-> no error")
    except Exception as e:
        print(label, "->", type(e).__name__, ":", str(e)[:120])
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print("     raised at %s:%d  %s" % (tb.filename.split('/')[-1], tb.lineno, tb.line))
