"""H3-1: UNIFORM_MANIFOLD.transport / project_ambient crash for a tangent-stacked (K != ()) UT3Tangent;
the ragged twin handles the same K+C-vs-C broadcast."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m
np.random.seed(0)
shape, nn, rr = (5, 6, 7), (2, 3, 3), (1, 2, 3, 1)
x = t3.TuckerTensorTrain.randn(shape, nn, rr); y = t3.TuckerTensorTrain.randn(shape, nn, rr)
frame = bvf.t3_orthogonal_representations(x)[0]; new_frame = bvf.t3_orthogonal_representations(y)[0]
v = t3m.MANIFOLD.randn(frame, stack_shape=(2,))            # K = (2,): two tangents at one frame
print('ragged: tangent_stack_shape', v.tangent_stack_shape, '-> transport dense shape', t3m.MANIFOLD.transport(v, new_frame).to_dense().shape)
uv = ut3m.UT3Tangent.from_t3tangent(v); unew = ubv.UT3Frame.from_t3frame(new_frame)
print('uniform: tangent_stack_shape', uv.tangent_stack_shape)
try:
    print('uniform transport ->', ut3m.UNIFORM_MANIFOLD.transport(uv, unew).to_dense().shape)
except Exception as e:
    print('uniform transport RAISES', type(e).__name__, e)
g = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=(2,))     # a K-stacked ambient gradient at an unstacked frame
print('ragged project_ambient(frame C=(), grad K+C=(2,)) ->', t3m.MANIFOLD.project_ambient(frame, g).stack_shape)
try:
    print('uniform project_ambient ->', ut3m.UNIFORM_MANIFOLD.project_ambient(ubv.UT3Frame.from_t3frame(frame), ut3.UniformTuckerTensorTrain.from_t3(g)).stack_shape)
except Exception as e:
    print('uniform project_ambient RAISES', type(e).__name__, e)
# unstacked K works
print('uniform transport with K=() ->', ut3m.UNIFORM_MANIFOLD.transport(ut3m.UT3Tangent.from_t3tangent(t3m.MANIFOLD.randn(frame)), unew).to_dense().shape)
