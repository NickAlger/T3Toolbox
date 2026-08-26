"""R10: differentiability of the weighted norms (the GK regularizer is the intended consumer)."""
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_frame_variations_format as ubvf
import t3toolbox.uniform_manifold as ut3m
J = lambda o: jax.tree_util.tree_map(jnp.asarray, o)
fin = lambda g: all(np.isfinite(np.asarray(l)).all() for l in jax.tree_util.tree_leaves(g))
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1))
W = t3.T3Weights.from_t3svd(x)
print('ragged  grad t3_weighted_norm wrt x (orth default):', fin(jax.grad(lambda a: t3.t3_weighted_norm(a, J(W)))(J(x))))
print('ragged  grad t3_weighted_norm wrt W:', fin(jax.grad(lambda w: t3.t3_weighted_norm(J(x), w))(J(W))))
print('ragged  grad x.norm() wrt x:', fin(jax.grad(lambda a: a.norm())(J(x))))
for n, r, tag in ((3, 3, 'tight'), (4, 4, 'padded')):
    ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=n, r=r)
    UW = ut3.UT3Weights.from_t3weights(W, n=n, r=r)
    print('uniform[%s] grad ux.norm() wrt ux:' % tag, fin(jax.grad(lambda a: a.norm())(J(ux))))
    print('uniform[%s] grad ut3_weighted_norm(orth=True)  wrt ux:' % tag, fin(jax.grad(lambda a: ut3.ut3_weighted_norm(a, J(UW)))(J(ux))),
          '| wrt W:', fin(jax.grad(lambda w: ut3.ut3_weighted_norm(J(ux), w))(J(UW))))
    print('uniform[%s] grad ut3_weighted_norm(orth=False) wrt ux:' % tag, fin(jax.grad(lambda a: ut3.ut3_weighted_norm(a, J(UW), use_orthogonalization=False))(J(ux))),
          '| wrt W:', fin(jax.grad(lambda w: ut3.ut3_weighted_norm(J(ux), w, use_orthogonalization=False))(J(UW))))
    print('uniform[%s] grad ut3_weighted_norm(orth=False) wrt W.reciprocal path:' % tag,
          fin(jax.grad(lambda w: ut3.ut3_weighted_norm(J(ux), w.reciprocal(), use_orthogonalization=False))(J(UW))))
    print('uniform[%s] grad ut3_weighted_inner(orth=True) wrt ux:' % tag, fin(jax.grad(lambda a: ut3.ut3_weighted_inner(a, J(UW), J(ux), J(UW)))(J(ux))))
# tangent-level metric norms (the actual GK-regularizer primitive) -- ragged and uniform
frame, _ = bvf.t3_orthogonal_representations(x); v = t3m.COREWISE.randn(frame); FW = bvf.T3FrameWeights.from_t3weights(W).reciprocal()
print('ragged  grad T3Tangent.weighted_norm wrt tangent:', fin(jax.grad(lambda t: t.weighted_norm(J(FW)))(J(v))), '| wrt W:', fin(jax.grad(lambda w: J(v).weighted_norm(w))(J(FW))))
ux = ut3.UniformTuckerTensorTrain.from_t3(x, n=4, r=4); uframe, _ = ubvf.ut3_orthogonal_representations(ux); uv = ut3m.UNIFORM_COREWISE.randn(uframe)
UFW = ubvf.UT3FrameWeights.from_t3frameweights(FW, nU=4, nD=4, rL=4, rR=4)
print('uniform[padded] grad UT3Tangent.weighted_norm wrt tangent:', fin(jax.grad(lambda t: t.weighted_norm(J(UFW)))(J(uv))), '| wrt W:', fin(jax.grad(lambda w: J(uv).weighted_norm(w))(J(UFW))),
      '| wrt W through reciprocal:', fin(jax.grad(lambda w: J(uv).weighted_norm(w.reciprocal()))(J(UFW))))
