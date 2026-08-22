"""H3 misc: obscure errors / asymmetries found while sweeping stacks."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf, t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.geometry as bgeo
np.random.seed(0)
shape, nn, rr = (5, 6, 7), (2, 3, 3), (1, 2, 3, 1)
x3 = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=(3,)); x0 = t3.TuckerTensorTrain.randn(shape, nn, rr)
def attempt(name, fn):
    try: print('%-55s -> %s' % (name, fn()))
    except Exception as e: print('%-55s -> RAISES %s: %r' % (name, type(e).__name__, str(e)[:120]))
attempt('x3 * x0.to_dense()  (stack mismatch)', lambda: (x3 * x0.to_dense()).shape)
attempt('x3.has_numerically_minimal_ranks()', lambda: x3.has_numerically_minimal_ranks())
attempt('x0.has_numerically_minimal_ranks()', lambda: x0.has_numerically_minimal_ranks())
fr = bvf.t3_orthogonal_representations(x3)[0]
attempt('frame(C=3).has_numerically_minimal_ranks()', lambda: fr.has_numerically_minimal_ranks())
v3 = t3m.MANIFOLD.randn(fr); vk3 = t3m.MANIFOLD.randn(fr, stack_shape=(2,))
attempt('T3Tangent.stack_tangents([v(K=()), v(K=(2,))])', lambda: t3m.T3Tangent.stack_tangents([v3, vk3]).tangent_stack_shape)
ux = ut3.UniformTuckerTensorTrain.from_t3(x0); uy = ut3.UniformTuckerTensorTrain.from_t3(x0 * 2.0)
attempt('UniformTuckerTensorTrain * UniformTuckerTensorTrain', lambda: (ux * uy).stack_shape)
# COREWISE.retract on a MANIFOLD-frame tangent with slack (nU != nD)
xs = t3.TuckerTensorTrain.randn(shape, (2, 3, 4), rr)          # mode-2 Tucker rank 4 > TT can carry (3): slack frame
fr_s = bvf.t3_orthogonal_representations(xs)[0]
print('slack frame up_ranks', fr_s.up_ranks, 'down_ranks', fr_s.down_ranks)
attempt('MANIFOLD.retract(v at slack frame)', lambda: t3m.MANIFOLD.retract(t3m.MANIFOLD.randn(fr_s)).ranks)
attempt('COREWISE.retract(v at slack MANIFOLD frame)', lambda: t3m.COREWISE.retract(t3m.MANIFOLD.randn(fr_s)).ranks)
# backend geometry inner collapses C; uniform twin keeps C
geom = bgeo.ManifoldGeometryOps(); fr3 = geom.frame(x3.data); v = t3m.MANIFOLD.randn(bvf.T3Frame(*fr3))
print('ragged  ManifoldGeometryOps.inner on C=(3,) -> shape', np.shape(geom.inner(v.variations.data, v.variations.data)))
print('ragged  ManifoldGeometryOps.point_norm_sq on C=(3,) -> shape', np.shape(geom.point_norm_sq(x3.t3svd()[0].data)))
ux3 = ut3.UniformTuckerTensorTrain.from_t3(x3); ug = bgeo.UniformManifoldGeometryOps.from_point(ux3.data, None)
uf = ug.frame((ux3.tucker_supercore, ux3.tt_supercore)); uvv = ug.project(uf, (np.random.randn(*uf[0].shape), np.random.randn(*uf[2].shape)))
print('uniform UniformManifoldGeometryOps.inner on C=(3,) -> shape', np.shape(ug.inner(uvv, uvv)))
print('uniform UniformManifoldGeometryOps.point_norm_sq on C=(3,) -> shape', np.shape(ug.point_norm_sq((ux3.tucker_supercore, ux3.tt_supercore))))
print('frontend MANIFOLD.inner on C=(3,) -> shape', np.shape(t3m.MANIFOLD.inner(v, v)))
