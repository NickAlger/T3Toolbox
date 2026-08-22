"""newton_cg(verbose=True, val_sample=...) without val_data is silently ignored; val_data alone crashes."""
import numpy as np, io, contextlib, t3toolbox as t3t
import t3toolbox.optimizers as opt, t3toolbox.manifold as t3m
import t3toolbox.backend.optimizer_display as bdisp, t3toolbox.backend.optimizers as bopt, t3toolbox.backend.fitting as bfit, t3toolbox.backend.geometry as bgeo
np.random.seed(0)
shape = (5, 6, 7)
xt = t3t.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
ww = [np.random.randn(30, N) for N in shape]; b = xt.apply(ww)
wv = [np.random.randn(6, N) for N in shape]; bv = xt.apply(wv)
x0 = t3t.TuckerTensorTrain.zeros(shape, (2, 2, 2), (1, 2, 2, 1))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    _, stats = opt.newton_cg(t3m.MANIFOLD, 'apply', ww, b, x0, max_newton=2, verbose=True, val_sample=wv)
out = buf.getvalue()
print('val_sample only -> raised? no.  "val" in display:', 'val' in out.lower(), '; val_err recorded:', any('val_err' in r for r in stats['diagnostics']))
try:
    with contextlib.redirect_stdout(io.StringIO()):
        opt.newton_cg(t3m.MANIFOLD, 'apply', ww, b, x0, max_newton=2, verbose=True, val_data=bv)
    print('val_data only -> no error')
except Exception as e:
    print('val_data only ->', type(e).__name__, ':', str(e)[:100])
# backend display directly
prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b)
cb, recs = bdisp.make_newton_display(prob, val_sample=wv, print_fn=None)
print('backend make_newton_display(val_sample=wv) only: accepted silently =', cb is not None)
