import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ufvf, t3toolbox.backend.ut3_masking as um, t3toolbox.uniform_manifold as ut3m
rng = np.random.default_rng(0)
def masked_rank_deficit(us):
    mtk, mtt = um.ut3_apply_masks(us.data); tkm, ttm = us.masks.data; d = mtk.shape[0]; deficit = 0
    for i in range(d):
        deficit += int(tkm[i].sum()) - np.linalg.matrix_rank(np.asarray(mtk[i]), tol=1e-10)
        if i < d - 1:
            G = np.asarray(mtt[i]); deficit += int(ttm[i+1].sum()) - np.linalg.matrix_rank(G.reshape(-1, G.shape[-1]), tol=1e-10)
    return deficit
n_cases = 60; svd_bad = svd_tensor_bad = frame_bad = retract_bad = 0
for trial in range(n_cases):
    d = int(rng.integers(2, 5)); shape = tuple(int(v) for v in rng.integers(3, 8, size=d))
    tk0 = tuple(int(v) for v in rng.integers(1, 3, size=d)); tt0 = (1,) + tuple(int(v) for v in rng.integers(1, 3, size=d-1)) + (1,)
    grow_t = tuple(a + int(rng.integers(1, 3)) for a in tk0); grow_r = (1,) + tuple(a + int(rng.integers(1, 3)) for a in tt0[1:-1]) + (1,)
    grow_t = tuple(min(g, N) for g, N in zip(grow_t, shape))
    np.random.seed(trial)
    x = t3.TuckerTensorTrain.randn(shape, tk0, tt0).resize(shape, grow_t, grow_r)     # zero-padded continuation start
    ux = ut3.UniformTuckerTensorTrain.from_t3(x); xd = np.asarray(x.to_dense())
    us, _, _ = ux.t3svd()
    if np.linalg.norm(np.asarray(us.to_dense()) - xd) > 1e-9 * np.linalg.norm(xd): svd_tensor_bad += 1
    if not bool(np.all(us.is_left_orthogonal())) or masked_rank_deficit(us) > 0: svd_bad += 1
    uf = ufvf.UT3Frame.from_ut3(ux)
    if float(np.max(uf.orthogonality_residual)) > 1e-8: frame_bad += 1
    import t3toolbox.safety as safety
    fr = ut3m.UNIFORM_MANIFOLD.frame(ux)
    with safety.unsafe():
        y = ut3m.UNIFORM_MANIFOLD.retract(ut3m.UNIFORM_MANIFOLD.randn(fr) * 0.0)
    if np.linalg.norm(np.asarray(y.to_dense()) - xd) > 1e-9 * np.linalg.norm(xd): retract_bad += 1
print('%d zero-padded continuation starts (d=2..4, random shapes/ranks):' % n_cases)
print('   uniform t3svd: tensor wrong in %d, output not left-orthogonal / rank-deficient masked block in %d' % (svd_tensor_bad, svd_bad))
print('   uniform FRAME (ut3_orthogonal_representations): not orthonormal in %d' % frame_bad)
print('   retract(zero tangent) at the deficient point: tensor wrong in %d' % retract_bad)
