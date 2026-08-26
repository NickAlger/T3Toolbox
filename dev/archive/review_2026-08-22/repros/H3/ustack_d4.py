import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m
np.random.seed(0)
for (shape, nn, rr) in [((5,6,7,3),(2,3,4,2),(1,2,3,2,1)), ((5,6,7),(2,3,4),(1,2,3,1)), ((5,6,7,3),(2,3,3,2),(1,2,3,2,1))]:
  for C, K in [((), (2,)), ((3,), (2,)), ((3,), ()), ((2,3), (4,))]:
    x = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
    frame = bvf.t3_orthogonal_representations(x)[0]
    v = t3m.MANIFOLD.randn(frame, stack_shape=K); uv = ut3m.UT3Tangent.from_t3tangent(v)
    print('d=%d C=%s K=%s frame up/down/left/right ranks %s %s %s %s' % (len(shape), C, K, frame.up_ranks, frame.down_ranks, frame.left_ranks, frame.right_ranks))
    for nm, fn in [('stack_tangents(unstack_tangents)', lambda: ut3m.UT3Tangent.stack_tangents(uv.unstack_tangents())),
                   ('stack_frame(unstack_frame)', lambda: ut3m.UT3Tangent.stack_frame(uv.unstack_frame())),
                   ('UT3Variations.stack(unstack)', lambda: ubv.UT3Variations.stack(uv.variations.unstack())),
                   ('UT3Frame.stack(unstack)', lambda: ubv.UT3Frame.stack(uv.frame.unstack())),
                   ('ut3.stack(unstack)', lambda: ut3.UniformTuckerTensorTrain.stack(ut3.UniformTuckerTensorTrain.from_t3(x).unstack())),
                   ('sum_tangents', lambda: uv.sum_tangents()), ('to_t3tangent', lambda: uv.to_t3tangent())]:
        if not K and 'tangents' in nm: continue
        if not C and 'frame' in nm.lower() and 'stack_frame' in nm: continue
        try:
            r = fn()
            ok = np.allclose(r.to_dense(), (v.to_dense() if 'sum' not in nm else v.sum_tangents().to_dense())) if hasattr(r, 'to_dense') and 'Variations' not in nm and 'Frame.stack' not in nm and 'to_t3' not in nm else True
            print('   %-36s OK match=%s' % (nm, ok))
        except Exception as e:
            print('   %-36s RAISES %s' % (nm, repr(e)[:160])); traceback.print_exc(limit=-4)
