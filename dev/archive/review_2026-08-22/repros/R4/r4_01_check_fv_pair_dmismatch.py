"""R4-1: check_fv_pair zips hole shapes with no length check -> a d-mismatched pair passes silently."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m

np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 2, 4), (1, 2, 3, 1))
frame, variations = bvf.t3_orthogonal_representations(x)          # d = 3
print('frame.d =', frame.d)

# variations with only the first TWO holes filled (d = 2), shapes matching holes 0 and 1 exactly
tucker_shapes, tt_shapes = frame.variation_shapes
v2 = bvf.T3Variations(tuple(np.random.randn(*s) for s in tucker_shapes[:2]),
                      tuple(np.random.randn(*s) for s in tt_shapes[:2]))
print('variations.d =', v2.d)

print('check_fv_pair(frame d=3, variations d=2) ->', bvf.check_fv_pair(frame, v2))   # expected: raise
t = t3m.T3Tangent(frame, v2)                                             # constructs without error
print('T3Tangent constructed:', t)
for name, f in [
    ('to_dense().shape', lambda: t.to_dense().shape),
    ('corewise_norm()', lambda: float(t.corewise_norm())),
    ('is_gauged()', lambda: bool(t.is_gauged())),
    ('gauge_residual', lambda: float(t.gauge_residual)),
    ('MANIFOLD.project(t).variations.d', lambda: t3m.MANIFOLD.project(t).variations.d),
    ('MANIFOLD.norm(t)', lambda: float(t3m.MANIFOLD.norm(t3m.MANIFOLD.project(t)))),
    ('to_t3()', lambda: t.to_t3()),
    ('to_vector().shape', lambda: t.to_vector().shape),
    ('apply(ww)', lambda: float(t.apply(tuple(np.random.randn(N) for N in (5, 6, 7))))),
    ('MANIFOLD.retract(t)', lambda: t3m.MANIFOLD.retract(t)),
]:
    try:
        print(f'  {name:40s} ->', f())
    except Exception as e:
        print(f'  {name:40s} -> {type(e).__name__}: {str(e).splitlines()[0][:110]}')

# the reverse mismatch: variations with MORE cores than the frame
v4 = bvf.T3Variations(tuple(np.random.randn(*s) for s in tucker_shapes) + (np.random.randn(2, 9),),
                      tuple(np.random.randn(*s) for s in tt_shapes) + (np.random.randn(1, 2, 1),))
print('check_fv_pair(frame d=3, variations d=4) ->', bvf.check_fv_pair(frame, v4))
t4 = t3m.T3Tangent(frame, v4)
try:
    print('  to_dense shape', t4.to_dense().shape)
except Exception as e:
    print('  to_dense ->', type(e).__name__, str(e)[:120])
