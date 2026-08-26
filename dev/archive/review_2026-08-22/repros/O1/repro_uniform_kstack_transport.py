"""UNIFORM_MANIFOLD.transport / project_ambient with a K-stacked tangent / K-stacked gradient, vs the ragged twin."""
import numpy as np, traceback
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m
np.random.seed(1)
shape, tr, ttr = (3, 5, 4), (2, 3, 2), (1, 2, 3, 1)
x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
frame, _ = bvf.t3_orthogonal_representations(x)
K = (2,)
v = t3m.MANIFOLD.randn(frame, stack_shape=K)
print('ragged: transport(K-stacked v, frame) ->', t3m.MANIFOLD.transport(v, frame).tangent_stack_shape, '| relerr to v:',
      np.linalg.norm(np.asarray(t3m.MANIFOLD.transport(v, frame).to_dense()) - np.asarray(v.to_dense())) / np.linalg.norm(np.asarray(v.to_dense())))
g = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=K)
try:
    print('ragged: project_ambient(frame, K-stacked T3 grad) ->', t3m.MANIFOLD.project_ambient(frame, g).tangent_stack_shape)
except Exception as e:
    print('ragged: project_ambient(frame, K-stacked T3 grad) RAISES', type(e).__name__, e)
print('ragged: project_ambient(frame, K-stacked dense grad) ->', t3m.MANIFOLD.project_ambient(frame, np.asarray(g.to_dense())).tangent_stack_shape)
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
uframe, _ = ubv.ut3_orthogonal_representations(ux)
uv = ut3m.UNIFORM_MANIFOLD.randn(uframe, stack_shape=K)
print('uniform: K-stacked tangent stack', uv.tangent_stack_shape, uv.frame_stack_shape)
for name, fn in [('UNIFORM_MANIFOLD.transport(K-stacked uv, uframe)', lambda: ut3m.UNIFORM_MANIFOLD.transport(uv, uframe)),
                 ('UNIFORM_MANIFOLD.project_ambient(uframe, uv.to_ut3())', lambda: ut3m.UNIFORM_MANIFOLD.project_ambient(uframe, uv.to_ut3())),
                 ('UNIFORM_MANIFOLD.project_ambient(uframe, K-stacked UT3 grad)', lambda: ut3m.UNIFORM_MANIFOLD.project_ambient(uframe, ut3.UniformTuckerTensorTrain.from_t3(g)))]:
    try:
        r = fn(); print(name, 'OK -> tangent stack', r.tangent_stack_shape)
    except Exception as e:
        print(name, 'RAISES', type(e).__name__, str(e)[:100]); tb = traceback.format_exc().splitlines(); print('   ', '\n    '.join(tb[-6:-1]))
