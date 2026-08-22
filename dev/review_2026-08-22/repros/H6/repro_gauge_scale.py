"""H6: GAUGE precondition thresholds an ABSOLUTE residual against a RELATIVE tolerance -> scale dependent."""
import numpy as np
import t3toolbox.safety as safety
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m

np.random.seed(0)
frame = bvf.T3Frame.random_orthogonal((5, 4, 6), (2, 3, 2), (1, 2, 3, 1))
v = t3m.MANIFOLD.randn(frame)            # gauged, unit-ish scale
print('gauge residual of a projected tangent at scale 1 :', float(v.gauge_residual))
for s in [1e3, 1e6, 1e8, 1e10]:
    w = v * s                              # scaling preserves the gauge EXACTLY (linear conditions)
    r = float(w.gauge_residual)
    try:
        n = t3m.MANIFOLD.norm(w); msg = 'OK  norm=%.3e' % float(n)
    except ValueError as e:
        msg = 'RAISES ' + str(e).splitlines()[0][:60]
    print('scale %8.0e: gauge_residual=%.2e  (rel=%.1e)  MANIFOLD.norm -> %s' % (s, r, r / float(w.corewise_norm()), msg))

# the converse: a TINY ungauged tangent passes the gauge check although it is NOT gauged
raw = t3m.T3Tangent(frame, bvf.T3Variations.randn(frame.variation_shapes, (), False))   # ungauged
tiny = raw * 1e-10
print('ungauged raw tangent: is_gauged =', bool(raw.is_gauged()), ' residual=%.2e' % float(raw.gauge_residual))
print('same tangent * 1e-10: is_gauged =', bool(tiny.is_gauged()), ' residual=%.2e' % float(tiny.gauge_residual))
hs = float(t3m.MANIFOLD.norm(tiny))                 # passes safe mode
true_hs = float(np.linalg.norm(tiny.to_dense()))
print('MANIFOLD.norm(tiny ungauged) = %.6e ; true HS norm of the realized tensor = %.6e ; ratio = %.4f'
      % (hs, true_hs, hs / true_hs))

# ORTH residual for comparison is scale-free (gram vs identity), TIED residual is relative by construction.
print('orthogonality residual:', float(frame.orthogonality_residual))
