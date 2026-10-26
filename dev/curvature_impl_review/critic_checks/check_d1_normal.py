"""Critic check (read-only): at d = 1 the manifold tangent space is all of R^N (X = U^T g, X != 0), so the
normal part N = A^T r - grad f vanishes and the manifold curvature term K_N is identically zero."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.apply as ap

rng = np.random.default_rng(0)
for N, n in ((7, 1), (7, 3)):
    U = np.linalg.qr(rng.standard_normal((N, n)))[0].T          # (n, N) orthonormal rows
    G = rng.standard_normal((1, n, 1))
    x = ((U,), (G,))
    frame, _ = fvc.t3_orthogonal_representations(x)
    W = 40
    ww = [rng.standard_normal((W, N))]
    r = rng.standard_normal(W)
    g = tv.tv_orthogonal_gauge_projection(frame, ap.tv_apply_transpose(r, ww, frame, sum_over_probes=True))
    grad_dense = tv.tv_to_dense(frame, g)
    Atr = np.einsum('w,wn->n', r, ww[0])
    print(f"d=1 N={N} n={n} frame ranks up={[u.shape for u in frame[0]]}: "
          f"|A^T r - grad f| / |A^T r| = {np.linalg.norm(Atr - grad_dense) / np.linalg.norm(Atr):.2e}")
