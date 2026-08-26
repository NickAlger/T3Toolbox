import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m
np.random.seed(0)
for (shape, nn, rr) in [((5,6,7,3),(2,3,4,2),(1,2,3,2,1)), ((5,6,7),(2,3,4),(1,2,3,1))]:
  for C, K in [((), ()), ((), (2,)), ((3,), (2,)), ((3,), ())]:
    x = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
    xsv = x.t3svd()[0]; frame2 = bvf.t3_orthogonal_representations(xsv)[0]
    fw = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(xsv))
    ufw = ubv.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(ut3.UniformTuckerTensorTrain.from_t3(xsv)))
    ufw2 = ubv.UT3FrameWeights.from_t3frameweights(fw)
    v2 = t3m.MANIFOLD.randn(frame2, stack_shape=K)
    uv2 = ut3m.UT3Tangent.from_t3tangent(v2)
    print('d=%d C=%s K=%s' % (len(shape), C, K), 'ufw stack', ufw.stack_shape, 'uv2 stacks', uv2.frame_stack_shape, uv2.tangent_stack_shape, 'frame n/r', uv2.frame.uniform_structure if hasattr(uv2.frame,'uniform_structure') else '')
    for nm, fn, ref in [('absorb_weights', lambda w: uv2.absorb_weights(w).to_dense(), v2.absorb_weights(fw).to_dense()),
                        ('weighted_norm', lambda w: uv2.weighted_norm(w), v2.weighted_norm(fw)),
                        ('weighted_inner', lambda w: uv2.weighted_inner(uv2 * 1.5, w), v2.weighted_inner(v2 * 1.5, fw))]:
        for wn, w in [('from_ut3weights', ufw), ('from_t3frameweights', ufw2)]:
            try:
                got = np.asarray(fn(w)); print('   %s[%s] shape %s match=%s' % (nm, wn, got.shape, got.shape == np.shape(ref) and np.allclose(got, ref)))
            except Exception as e:
                print('   %s[%s] RAISES %s' % (nm, wn, repr(e)[:120])); traceback.print_exc(limit=-3)
