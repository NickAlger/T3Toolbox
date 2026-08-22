"""Which non-minimal TT ranks make UNIFORM_MANIFOLD.frame non-orthogonal? (ragged frame always orthogonal)"""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.manifold as t3m, t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_manifold as um, t3toolbox.safety as safety
np.random.seed(5)
for shape, tr, ttr, why in [((4, 5, 3), (2, 3, 2), (1, 2, 3, 1), 'r2=3 > n2=2 (last core right-deficient)'),
                            ((4, 5, 3), (2, 3, 2), (1, 3, 2, 1), 'r1=3 > n0=2 (first core left-deficient)'),
                            ((3, 3, 3, 3), (2, 2, 2, 2), (1, 2, 5, 2, 1), 'r2=5 > r1*n1=4 (middle, left unfolding deficient)'),
                            ((3, 3, 3, 3), (2, 2, 2, 2), (1, 2, 2, 3, 1), 'r3=3 > n3*r4=2 (right unfolding deficient)'),
                            ((4, 5, 3), (2, 3, 2), (1, 2, 2, 1), 'minimal')]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr); ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    fr = t3m.MANIFOLD.frame(x); ufr = um.UNIFORM_MANIFOLD.frame(ux)
    g = t3.TuckerTensorTrain.randn(shape, tuple(2 for _ in shape), (1,) + (2,) * (len(shape) - 1) + (1,)); ug = ut3.UniformTuckerTensorTrain.from_t3(g)
    with safety.unsafe():
        pr = t3m.MANIFOLD.project_ambient(fr, g); pu = um.UNIFORM_MANIFOLD.project_ambient(ufr, ug)
    print('%-48s ragged orth %.0e | uniform orth residual %.1e is_orth=%-5s | unsafe project_ambient ragged-vs-uniform rel diff %.1e | frame reproduces x: ragged %.0e uniform %.0e'
          % (why, float(fr.orthogonality_residual), float(ufr.orthogonality_residual), bool(ufr.is_orthogonal().all()),
             np.linalg.norm(pr.to_dense() - pu.to_dense()) / np.linalg.norm(pr.to_dense()),
             np.linalg.norm(t3m.T3Tangent.zeros(fr).to_t3().to_dense() - x.to_dense()) / np.linalg.norm(x.to_dense()) if hasattr(t3m.T3Tangent.zeros(fr), 'to_t3') else -1,
             np.linalg.norm(um.UT3Tangent.zeros(ufr).to_dense() - x.to_dense()) / np.linalg.norm(x.to_dense()) if False else float('nan')))
