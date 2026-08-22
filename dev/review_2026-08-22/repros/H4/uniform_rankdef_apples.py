"""Apples-to-apples: project the same ambient g onto the SAME frame (uniform frame converted to ragged)
with the ragged backend vs the uniform backend; and locate the zero column in the uniform frame."""
import numpy as np
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, T3Tangent, MANIFOLD, T3Frame
from t3toolbox.frame_variations_format import t3_orthogonal_representations
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3
from t3toolbox.uniform_frame_variations_format import ut3_orthogonal_representations, UT3Frame
from t3toolbox.uniform_manifold import UT3Tangent, UNIFORM_MANIFOLD as UM
np.random.seed(0)

def apples(label, z):
    uz = UT3.from_t3(z)
    fu, vu = ut3_orthogonal_representations(uz)
    fr_from_u = fu.to_t3frame()                       # the SAME frame, ragged
    g = T3.randn(z.shape, z.tucker_ranks, z.tt_ranks)
    with tb.safety.unsafe():
        pu = UM.project_ambient(fu, UT3.from_t3(g)).to_dense()
        pr = MANIFOLD.project_ambient(fr_from_u, g).to_dense()
        # also: the uniform frame converted to ragged, is it orthogonal?
        print('=== %s | uniform frame residual %.1e | same-frame ragged residual %.1e' % (label, float(np.max(fu.orthogonality_residual)), float(np.max(fr_from_u.orthogonality_residual))))
        print('   same frame, project_ambient: |ragged - uniform|/|ragged| = %.2e' % (np.linalg.norm(pr - pu) / np.linalg.norm(pr)))
        # idempotence of each projection: P(P g) == P g  (project the dense result again)
        pu2 = UM.project_ambient(fu, UT3.from_t3(UT3Tangent(fu, UM.project_ambient(fu, UT3.from_t3(g)).variations).to_ut3().to_t3() if False else g)).to_dense()
    # self-adjointness / tangency check vs dense: pr must equal the orthogonal projection of g onto span of the tangent basis
    # build the tangent basis from unit variations (ragged frame) and project densely
    t0 = T3Tangent.zeros(fr_from_u)
    vec0 = t0.to_vector()
    n = vec0.size
    basis = []
    for k in range(n):
        e = np.zeros(n); e[k] = 1.0
        basis.append(T3Tangent.from_vector(e, fr_from_u).to_dense().ravel())
    B = np.stack(basis, axis=1)                       # (N^d, n) -- spans the tangent space (rank-deficient possible)
    gd = g.to_dense().ravel()
    Pg, *_ = np.linalg.lstsq(B, gd, rcond=None)
    proj_dense = B @ Pg
    print('   vs dense least-squares projection onto span(tangent basis): ragged err %.2e, uniform err %.2e' % (
        np.linalg.norm(pr.ravel() - proj_dense) / np.linalg.norm(proj_dense), np.linalg.norm(pu.ravel() - proj_dense) / np.linalg.norm(proj_dense)))

y = T3.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
apples('z = y + y (structurally minimal, numerically rank-deficient)', y + y)
y2 = T3.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
apples('z = y2 + y2', y2 + y2)
x = T3.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
apples('x.resize zero-padded (3,3,3)/(1,3,3,1)', x.resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1)))
apples('x full rank (control)', x)
# locate the zero column
z = x.resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))
fu, vu = ut3_orthogonal_representations(UT3.from_t3(z))
fr = fu.to_t3frame()
for name, cores in [('up', fr.up_tucker_cores), ('down', fr.down_tt_cores), ('left', fr.left_tt_cores), ('right', fr.right_tt_cores)]:
    for i, c in enumerate(cores):
        print('   %s[%d] shape %s  |core| = %.3g  zero rows/cols: %s' % (name, i, c.shape, np.linalg.norm(c), [ax for ax in range(c.ndim) if np.any(np.all(c == 0, axis=tuple(a for a in range(c.ndim) if a != ax)))]))
