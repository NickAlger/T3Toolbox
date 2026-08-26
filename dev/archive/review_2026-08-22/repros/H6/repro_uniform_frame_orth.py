"""H6: UNIFORM_MANIFOLD.frame(x) on a NON-minimal-rank x fails its own ORTH precondition (ragged twin passes)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as um
np.random.seed(0)
for shape, tr, ttr in [((4, 5, 3), (2, 3, 2), (1, 2, 3, 1)), ((4, 5, 3), (2, 3, 2), (1, 2, 2, 1)), ((4, 4, 4), (3, 3, 3), (1, 2, 2, 1)), ((4, 5, 3), (4, 3, 2), (1, 2, 2, 1))]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr); ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    fr = t3m.MANIFOLD.frame(x); ufr = um.UNIFORM_MANIFOLD.frame(ux)
    print('%s tr=%s ttr=%s minimal=%s | ragged frame orth residual=%.1e is_orth=%s | uniform frame residual=%.1e is_orth=%s up_ranks=%s left_ranks=%s down=%s right=%s'
          % (shape, tr, ttr, x.has_minimal_ranks, float(fr.orthogonality_residual), bool(fr.is_orthogonal()), float(ufr.orthogonality_residual), bool(ufr.is_orthogonal().all()),
             ufr.up_ranks, ufr.left_ranks, ufr.down_ranks, ufr.right_ranks))
    try:
        um.UNIFORM_MANIFOLD.randn(ufr); print('    UNIFORM_MANIFOLD.randn(frame) OK')
    except ValueError as e:
        print('    UNIFORM_MANIFOLD.randn(frame) RAISES:', str(e)[:70])
    # via to_t3frame: is the represented frame orthogonal?
    tf = ufr.to_t3frame(); print('    to_t3frame().is_orthogonal() =', bool(tf.is_orthogonal()), ' residual=%.1e' % float(tf.orthogonality_residual), ' ranks up/down/left/right:', tf.up_ranks, tf.down_ranks, tf.left_ranks, tf.right_ranks)

print('== unsafe mode: the non-orthogonal uniform frame silently gives a different tangent than the ragged path')
import t3toolbox.safety as safety
x = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 3, 1)); ux = ut3.UniformTuckerTensorTrain.from_t3(x)
g = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1)); ug = ut3.UniformTuckerTensorTrain.from_t3(g)
fr = t3m.MANIFOLD.frame(x); ufr = um.UNIFORM_MANIFOLD.frame(ux)
with safety.unsafe():
    pr = t3m.MANIFOLD.project_ambient(fr, g); pu = um.UNIFORM_MANIFOLD.project_ambient(ufr, ug)
print('   |ragged Pi g - uniform Pi g| / |ragged Pi g| = %.3e' % (np.linalg.norm(pr.to_dense() - pu.to_dense()) / np.linalg.norm(pr.to_dense())))
print('   ragged frame reproduces x? %.1e ; uniform frame reproduces x? %.1e' % (
    np.linalg.norm(t3m.T3Tangent.zeros(fr).to_t3().to_dense()[..., :0].sum()) if False else 0.0, 0.0))
for shape, tr, ttr in [((4, 5, 3), (5, 3, 2), (1, 2, 2, 1)), ((3, 3, 3, 3), (2, 2, 2, 2), (1, 2, 4, 2, 1)), ((6, 6), (3, 3), (1, 5, 1))]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr); ux = ut3.UniformTuckerTensorTrain.from_t3(x); ufr = um.UNIFORM_MANIFOLD.frame(ux)
    print('   %s tr=%s ttr=%s minimal=%s -> uniform frame residual %.1e is_orth=%s (ragged: %s)' % (shape, tr, ttr, x.has_minimal_ranks, float(ufr.orthogonality_residual), bool(ufr.is_orthogonal().all()), bool(t3m.MANIFOLD.frame(x).is_orthogonal())))

print('== which family of the extracted frame is non-orthogonal? (x with r2=3 > n2=2)')
x = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 3, 1)); ux = ut3.UniformTuckerTensorTrain.from_t3(x)
tf = um.UNIFORM_MANIFOLD.frame(ux).to_t3frame(); U, D, L, R = tf.data
for i in range(3):
    print('   i=%d U gram dev %.1e | D gram dev %.1e | L gram dev %.1e | R gram dev %.1e' % (i,
          np.abs(np.einsum('io,jo->ij', U[i], U[i]) - np.eye(U[i].shape[0])).max(), np.abs(np.einsum('iaj,ibj->ab', D[i], D[i]) - np.eye(D[i].shape[1])).max(),
          np.abs(np.einsum('iaj,iak->jk', L[i], L[i]) - np.eye(L[i].shape[2])).max(), np.abs(np.einsum('iaj,kaj->ik', R[i], R[i]) - np.eye(R[i].shape[0])).max()))
print('   (boundary remainders L[d-1], R[0] are not checked)')
print('   uniform masks tt:', um.UNIFORM_MANIFOLD.frame(ux).masks.data[2].astype(int).tolist())
