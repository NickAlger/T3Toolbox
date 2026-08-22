import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m
np.random.seed(0)
for (shape, nn, rr) in [((5,6,7,3),(2,3,4,2),(1,2,3,2,1)), ((5,6,7),(2,3,4),(1,2,3,1)), ((5,6,7),(2,3,3),(1,2,3,1))]:
  for C, K in [((), ()), ((), (2,)), ((3,), (2,)), ((3,), ()), ((2,3), (4,))]:
    x = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C); y = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
    frame = bvf.t3_orthogonal_representations(x)[0]; nframe = bvf.t3_orthogonal_representations(y)[0]
    uframe = ubv.UT3Frame.from_t3frame(frame); unframe = ubv.UT3Frame.from_t3frame(nframe)
    v = t3m.MANIFOLD.randn(frame, stack_shape=K); uv = ut3m.UT3Tangent.from_t3tangent(v)
    g = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
    gk = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=K + C)
    print('d=%d C=%s K=%s' % (len(shape), C, K))
    for nm, fn, ref in [('project_ambient(C grad)', lambda: ut3m.UNIFORM_MANIFOLD.project_ambient(uframe, ut3.UniformTuckerTensorTrain.from_t3(g)), lambda: t3m.MANIFOLD.project_ambient(frame, g)),
                        ('project_ambient(K+C grad)', lambda: ut3m.UNIFORM_MANIFOLD.project_ambient(uframe, ut3.UniformTuckerTensorTrain.from_t3(gk)), lambda: t3m.MANIFOLD.project_ambient(frame, gk)),
                        ('v.to_ut3()', lambda: uv.to_ut3(), lambda: v.to_t3()),
                        ('project_ambient(frame, v.to_ut3())', lambda: ut3m.UNIFORM_MANIFOLD.project_ambient(unframe, uv.to_ut3()), lambda: t3m.MANIFOLD.project_ambient(nframe, v.to_t3())),
                        ('transport', lambda: ut3m.UNIFORM_MANIFOLD.transport(uv, unframe), lambda: t3m.MANIFOLD.transport(v, nframe))]:
        try:
            r = fn()
            try:
                rr_ = ref(); ok = np.allclose(r.to_dense(), rr_.to_dense()); rs = 'ragged OK'
            except Exception as e2:
                ok = None; rs = 'ragged RAISES %s' % repr(e2)[:100]
            print('   %-38s OK match=%s (%s)' % (nm, ok, rs))
        except Exception as e:
            try:
                ref(); rs = 'ragged OK'
            except Exception as e2:
                rs = 'ragged RAISES %s' % repr(e2)[:100]
            print('   %-38s RAISES %s   (%s)' % (nm, repr(e)[:130], rs)); traceback.print_exc(limit=-3)
