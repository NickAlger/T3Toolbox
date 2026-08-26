"""MANIFOLD.retract on a 2-D frame stack C=(2,3): per-element vs unstacked retract."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
from t3toolbox.backend import tv_operations
np.random.seed(0)
shape, tr, ttr = (3, 5, 4), (2, 3, 2), (1, 2, 3, 1)
for C in [(), (2,), (6,), (2, 3), (1, 2), (2, 1)]:
    x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
    frame, _ = bvf.t3_orthogonal_representations(x)
    v = t3m.MANIFOLD.randn(frame) * 0.1
    R = np.asarray(t3m.MANIFOLD.retract(v).to_dense())         # stacked retract, shape C + shape
    # per-element reference: unstack the tangent over C and retract each alone
    errs = []
    if C:
        leaves = v.unstack_frame()
        import t3toolbox.backend.stacking as stacking
        flat = []
        def walk(t):
            if isinstance(t, t3m.T3Tangent): flat.append(t)
            else:
                for s in t: walk(s)
        walk(leaves)
        for k, leaf in enumerate(flat):
            idx = np.unravel_index(k, C)
            ref = np.asarray(t3m.MANIFOLD.retract(leaf).to_dense())
            errs.append(np.linalg.norm(R[idx] - ref) / np.linalg.norm(ref))
    # also: shifted embedding and implicit t3svd pieces
    emb = v.to_t3(include_shift=True)
    e_emb = np.linalg.norm(np.asarray(emb.to_dense()) - (np.asarray(frame.to_dense()) + np.asarray(v.to_dense()))) / np.linalg.norm(np.asarray(frame.to_dense()))
    xs, _, _ = emb.t3svd(max_tucker_ranks=tuple(frame.up_ranks), max_tt_ranks=tuple(frame.left_ranks))
    e_svd = np.linalg.norm(np.asarray(xs.to_dense()) - R) / np.linalg.norm(R)
    print('C=%-7s max per-element err vs unstacked retract: %s | embedding err %.1e | retract vs explicit t3svd(embedding) %.1e'
          % (C, ('%.1e' % max(errs)) if errs else 'n/a', e_emb, e_svd))
