"""H6 side-check: is the shared projection actually restricting to the tied tangent subspace? Rank of many projected samples."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.shared_geometry as sg
np.random.seed(2)
M = t3m.MANIFOLD
for shape, tr, ttr, sh in [((4, 4), (2, 2), (1, 2, 1), (0, 0)), ((5, 5, 4), (2, 2, 2), (1, 2, 2, 1), (0, 0, 1)), ((5, 5, 5), (2, 2, 2), (1, 3, 3, 1), (0, 0, 0)),
                           ((6, 6, 3, 3), (3, 3, 2, 2), (1, 3, 4, 2, 1), (0, 0, 1, 1))]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr).share(sh); S = sg.shared_manifold(sh); f = S.frame(x); sd = S.shared_frame_data(f)
    full = t3m.manifold_dim((shape, tr, ttr)); shared = t3m.manifold_dim((shape, tr, ttr), sharing=sh)
    n = full + 6
    A = np.stack([M.randn(f).to_dense().ravel() for _ in range(n)])
    P = np.stack([S.project(M.randn(f)).to_dense().ravel() for _ in range(n)])
    y = S.retract(S.project(M.randn(f)))
    print('%s sh=%s: manifold_dim full=%d shared=%d | rank(full tangents)=%d rank(S.project tangents)=%d | svd_s=%s | retract tied? %s | down_ranks=%s'
          % (shape, sh, full, shared, np.linalg.matrix_rank(A, tol=1e-8), np.linalg.matrix_rank(P, tol=1e-8),
             [np.round(s, 2) for s in sd.svd_s], bool(y.has_shared_tucker_factors(sh)), f.down_ranks))
