"""Uniform layer on a share()-entered point: t3svd(sharing) / orthogonal representations / uniform_minimal."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.backend.uniform_fitting as uf
np.random.seed(1)
for sh, shape, tr, ttr in [((0, 0, 0), (4, 4, 4), (2, 2, 2), (1, 2, 3, 1)), ((0, 0, 1), (4, 4, 5), (2, 2, 3), (1, 2, 3, 1))]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr).share(sh); X = np.asarray(x.to_dense())
    rf, _ = bvf.t3_orthogonal_representations(x)
    print('sharing=%s x ranks %s %s | ragged frame orthogonal=%s minimal=%s' % (sh, x.tucker_ranks, x.tt_ranks, bool(rf.is_orthogonal()), rf.has_minimal_ranks))
    for pad in [{}, dict(N=8, n=5, r=5)]:
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, **pad)
        xs, _, _ = ux.t3svd(sharing=sh)
        print('  pad=%-24s ut3.t3svd(sharing): dense relerr %.1e | left_orth=%s | tied=%s | ranks %s %s' % (pad, np.linalg.norm(np.asarray(xs.to_dense()) - X) / np.linalg.norm(X),
              bool(np.all(xs.is_left_orthogonal())), bool(np.all(xs.has_shared_tucker_factors(sh))), xs.tucker_ranks.tolist(), xs.tt_ranks.tolist()))
        uf0, _ = ubv.ut3_orthogonal_representations(ux)
        um = uf.uniform_minimal(ux, sharing=sh)
        uf1, _ = ubv.ut3_orthogonal_representations(um)
        print('  pad=%-24s orth-rep(ux) orthogonal=%s (res %.1e) | uniform_minimal ranks %s %s -> orth-rep orthogonal=%s (res %.1e)' % (pad,
              bool(np.all(uf0.is_orthogonal())), float(np.max(np.asarray(uf0.orthogonality_residual))), um.tucker_ranks.tolist(), um.tt_ranks.tolist(),
              bool(np.all(uf1.is_orthogonal())), float(np.max(np.asarray(uf1.orthogonality_residual)))))
