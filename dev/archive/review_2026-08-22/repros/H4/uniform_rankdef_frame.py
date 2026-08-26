"""Uniform orthogonal frame on a numerically rank-deficient (but structurally minimal) train:
non-orthogonal frame -> safe mode raises; unsafe mode (the jit/optimizer regime) silently mis-projects."""
import numpy as np
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, T3Tangent, MANIFOLD
from t3toolbox.frame_variations_format import t3_orthogonal_representations
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3
from t3toolbox.uniform_frame_variations_format import ut3_orthogonal_representations
from t3toolbox.uniform_manifold import UT3Tangent, UNIFORM_MANIFOLD as UM
np.random.seed(0)

def report(label, z):
    uz = UT3.from_t3(z)
    print('=== %s: shape %s tucker %s tt %s | has_minimal_ranks(ragged)=%s (uniform)=%s' % (label, z.shape, z.tucker_ranks, z.tt_ranks, z.has_minimal_ranks, np.all(uz.has_minimal_ranks)))
    fr, vr = t3_orthogonal_representations(z)
    fu, vu = ut3_orthogonal_representations(uz)
    print('   ragged  frame residual %.1e ranks up=%s down=%s left=%s' % (float(np.max(fr.orthogonality_residual)), fr.up_ranks, fr.down_ranks, fr.left_ranks))
    print('   uniform frame residual %.1e ranks up=%s down=%s left=%s' % (float(np.max(fu.orthogonality_residual)), fu.up_ranks.tolist(), fu.down_ranks.tolist(), fu.left_ranks.tolist()))
    # the Riemannian gradient of the same ambient tensor g, ragged vs uniform, in UNSAFE mode
    g = T3.randn(z.shape, z.tucker_ranks, z.tt_ranks)
    with tb.safety.unsafe():
        pr = MANIFOLD.project_ambient(fr, g).to_dense()
        pu = UM.project_ambient(fu, UT3.from_t3(g)).to_dense()
    print('   unsafe project_ambient: |ragged - uniform| / |ragged| = %.2e' % (np.linalg.norm(pr - pu) / np.linalg.norm(pr)))
    try:
        UM.project_ambient(fu, UT3.from_t3(g)); print('   safe mode: project_ambient OK')
    except ValueError as e:
        print('   safe mode: RAISES', str(e).splitlines()[0][:70])
    # the projection must be idempotent and reproduce a tangent's own dense: P(v) == v for v in the tangent space
    with tb.safety.unsafe():
        v = UM.project(UT3Tangent(fu, vu))          # a tangent at fu (gauged)
        vd = v.to_dense()
        v2 = UM.project_ambient(fu, UT3.from_t3(T3Tangent(fr, vr).to_t3()))  # irrelevant path; skip
    return fr, fu

y = T3.randn((8, 9, 10), (2, 2, 2), (1, 2, 2, 1))
report('z = y + y (structurally minimal, numerically rank-deficient)', y + y)
y2 = T3.randn((8, 9, 10), (2, 3, 2), (1, 2, 2, 1))
report('z = y2 + y2 (also Tucker rank 6 < N)', y2 + y2)
x = T3.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))
report('x.resize zero-padded to tucker (3,3,3) tt (1,3,3,1) (continuation-style)', x.resize((6, 7, 8), (3, 3, 3), (1, 3, 3, 1)))
report('x full rank (control)', x)
xs = T3.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2,))
report('stacked (2,): xs + xs', xs + xs)
