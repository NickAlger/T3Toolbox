"""Safe-mode GAUGE precondition: the `rtol` is applied as an ABSOLUTE bound on the max-abs gauge residual,
so a large-magnitude tangent false-fails even in numpy float64."""
import numpy as np
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, T3Tangent, MANIFOLD, COREWISE
from t3toolbox.frame_variations_format import t3_orthogonal_representations
np.random.seed(0)
x = T3.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
f, v = t3_orthogonal_representations(x)
for scale in [1.0, 1e2, 1e4, 1e6]:
    t = T3Tangent(f, v) * scale
    p = MANIFOLD.project(t)
    print('scale %g: |p| = %.3g, gauge_residual = %.3g (atol used = %g), is_gauged() = %s' % (
        scale, float(p.corewise_norm()), float(p.gauge_residual), tb.safety.effective_rtol(p.data), bool(p.is_gauged(atol=tb.safety.effective_rtol(p.data)))))
    try:
        n = MANIFOLD.norm(p); print('   MANIFOLD.norm OK = %.6g' % float(n))
    except ValueError as e:
        print('   MANIFOLD.norm RAISED:', str(e).splitlines()[0][:90])
    try:
        n = MANIFOLD.inner(p, p); print('   MANIFOLD.inner OK')
    except ValueError as e:
        print('   MANIFOLD.inner RAISED:', str(e).splitlines()[0][:90])
# the same with a large base point and the gradient of a fit (the ordinary path): project_ambient of a large T3
xb = x * 1e4
fb, vb = t3_orthogonal_representations(xb)
g = MANIFOLD.project_ambient(fb, xb)    # Riemannian gradient-like projection of a large ambient tensor
print('project_ambient of |x|~%.2g: gauge_residual = %.3g' % (float(xb.norm()), float(g.gauge_residual)))
try:
    MANIFOLD.norm(g); print('   MANIFOLD.norm OK')
except ValueError as e:
    print('   MANIFOLD.norm RAISED:', str(e).splitlines()[0][:90])
