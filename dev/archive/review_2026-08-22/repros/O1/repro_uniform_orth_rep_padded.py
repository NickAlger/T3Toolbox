"""ut3_orthogonal_representations on a padded UT3 whose TT tail rank is non-minimal: is the frame orthogonal?"""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv, t3toolbox.uniform_manifold as ut3m
np.random.seed(1)
cases = [('d3 nonmin tail r2=3 > n2*r3=2', (3, 5, 4), (2, 3, 2), (1, 2, 3, 1)),
         ('d3 minimal',                   (3, 5, 4), (2, 3, 2), (1, 2, 2, 1)),
         ('nonmin r1=4 > n0*r0=2',        (3, 5, 4), (2, 3, 2), (1, 4, 3, 1)),
         ('d2 minimal',                   (3, 5),    (2, 3),    (1, 2, 1))]
for name, shape, tr, ttr in cases:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr)
    rframe, _ = bvf.t3_orthogonal_representations(x)
    for pad in [{}, dict(N=8, n=5, r=5)]:
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, **pad)
        uframe, uvar = ubv.ut3_orthogonal_representations(ux)
        res = np.asarray(uframe.orthogonality_residual)
        # masked content vs ragged frame: compare represented tensors & per-family real-part orthogonality
        fr = uframe.to_t3frame()
        e_dense = np.linalg.norm(np.asarray(uframe.to_dense()) - np.asarray(x.to_dense())) / np.linalg.norm(np.asarray(x.to_dense()))
        print('%-30s pad=%-25s is_orthogonal=%s residual=%.2e | to_t3frame().is_orthogonal=%s | frame dense relerr %.1e | uniform ranks up=%s left=%s (ragged frame up=%s left=%s)'
              % (name, pad, bool(np.all(uframe.is_orthogonal())), float(np.max(res)), bool(np.all(fr.is_orthogonal())), e_dense,
                 uframe.up_ranks.tolist(), uframe.left_ranks.tolist(), rframe.up_ranks, rframe.left_ranks))
        try:
            ut3m.UNIFORM_MANIFOLD.randn(uframe); print('      UNIFORM_MANIFOLD.randn OK')
        except Exception as e:
            print('      UNIFORM_MANIFOLD.randn RAISES:', str(e)[:70])
