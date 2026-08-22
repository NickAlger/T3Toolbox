"""UNIFORM_MANIFOLD.transport / project_ambient with a K-stacked tangent (or K-stacked UT3 gradient): crashes;
the ragged MANIFOLD twin handles the same K-stacked inputs."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m
np.random.seed(1)
shape, tr, ttr = (3, 5, 4), (2, 3, 2), (1, 2, 2, 1)      # minimal ranks
K = (2,)
for C in [(), (2,)]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
    frame, _ = bvf.t3_orthogonal_representations(x)
    v = t3m.MANIFOLD.randn(frame, stack_shape=K)
    vt = t3m.MANIFOLD.transport(v, frame)
    print('C=%s ragged : transport(K-stacked v, frame) OK, tangent stack %s, relerr to v %.1e' % (C, vt.tangent_stack_shape,
          np.linalg.norm(np.asarray(vt.to_dense()) - np.asarray(v.to_dense())) / np.linalg.norm(np.asarray(v.to_dense()))))
    ux = ut3.UniformTuckerTensorTrain.from_t3(x); uframe, _ = ubv.ut3_orthogonal_representations(ux)
    uv = ut3m.UNIFORM_MANIFOLD.randn(uframe, stack_shape=K)
    print('C=%s uniform: K-stacked tangent built OK (tangent stack %s, frame stack %s); probe/apply/inner on it work: %s' % (
        C, uv.tangent_stack_shape, uv.frame_stack_shape, np.asarray(ut3m.UNIFORM_MANIFOLD.norm(uv)).shape))
    for name, fn in [('UNIFORM_MANIFOLD.transport(K-stacked uv, uframe)', lambda: ut3m.UNIFORM_MANIFOLD.transport(uv, uframe)),
                     ('UNIFORM_MANIFOLD.project_ambient(uframe, K-stacked UT3)', lambda: ut3m.UNIFORM_MANIFOLD.project_ambient(uframe, uv.to_ut3())),
                     ('UNIFORM_MANIFOLD.transport(unstacked uv, uframe)', lambda: ut3m.UNIFORM_MANIFOLD.transport(ut3m.UNIFORM_MANIFOLD.randn(uframe), uframe))]:
        try:
            r = fn(); print('   %-55s OK -> tangent stack %s' % (name, r.tangent_stack_shape))
        except Exception as e:
            tb = traceback.format_exc().splitlines()
            print('   %-55s RAISES %s: %s\n      at %s' % (name, type(e).__name__, str(e)[:70], tb[-3].strip()))
