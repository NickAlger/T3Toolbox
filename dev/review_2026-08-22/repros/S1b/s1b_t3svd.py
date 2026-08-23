import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ufvf, t3toolbox.backend.ut3_masking as um
import t3toolbox.uniform_manifold as ut3m
np.random.seed(0)
x0 = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
x = x0.resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))          # case B: structurally minimal, numerically deficient
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
xd = np.asarray(x.to_dense())
for label, caps in [('uniform t3svd()', {}), ('uniform t3svd(max ranks = the padded ranks 3/3)', dict(max_tucker_ranks=3, max_tt_ranks=3))]:
    us, st, stt = ux.t3svd(**caps)
    print('%s:' % label)
    print('   tensor error %.1e | ranks tucker %s tt %s | is_left_orthogonal %s' % (
        np.linalg.norm(np.asarray(us.to_dense()) - xd) / np.linalg.norm(xd), tuple(int(v) for v in us.tucker_ranks), tuple(int(v) for v in us.tt_ranks), bool(np.all(us.is_left_orthogonal()))))
    mtk, mtt = um.ut3_apply_masks(us.data)
    tkm, ttm = us.masks.data
    for i in range(3):
        U = np.asarray(mtk[i]); r_mask = int(tkm[i].sum()); r_num = np.linalg.matrix_rank(U, tol=1e-10)
        G = np.asarray(mtt[i]); M = G.reshape(-1, G.shape[-1]); rb = int(ttm[i+1].sum()); rn = np.linalg.matrix_rank(M, tol=1e-10)
        print('   core %d: Tucker mask rank %d, numerical %d | TT right-bond mask rank %d, numerical %d | Tucker svals %s' % (i, r_mask, r_num, rb, rn, np.round(np.asarray(st[i]), 3)))
# the frame built FROM the t3svd output (what the optimizer does next)
us, _, _ = ux.t3svd()
uf = ufvf.UT3Frame.from_ut3(us)
print('frame from the uniform t3svd output: residual %.1e' % float(np.max(uf.orthogonality_residual)))
# and the retraction at this point: a zero tangent step
fr = ut3m.UNIFORM_MANIFOLD.frame(ux)
y = ut3m.UNIFORM_MANIFOLD.retract(ut3m.UNIFORM_MANIFOLD.zeros(fr))
print('retract(zero tangent) at the deficient point: tensor error %.1e | ranks %s %s | is_left_orthogonal %s' % (
    np.linalg.norm(np.asarray(y.to_dense()) - xd) / np.linalg.norm(xd), tuple(int(v) for v in y.tucker_ranks), tuple(int(v) for v in y.tt_ranks), bool(np.all(y.is_left_orthogonal()))))
# ragged comparison
xs, _, _ = x.t3svd()
print('ragged t3svd(): ranks %s %s | is_left_orthogonal %s' % (xs.tucker_ranks, xs.tt_ranks, bool(xs.is_left_orthogonal())))
