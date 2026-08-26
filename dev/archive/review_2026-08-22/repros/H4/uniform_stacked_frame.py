"""Stacked uniform orthogonal representations: which family is non-orthogonal, and is it mask-related?"""
import numpy as np
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, T3Frame
from t3toolbox.frame_variations_format import t3_orthogonal_representations
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3
from t3toolbox.uniform_frame_variations_format import ut3_orthogonal_representations, UT3Frame
import t3toolbox.backend.ufv_operations as ufv_ops
np.random.seed(0)
def check(label, y):
    uy = UT3.from_t3(y) if isinstance(y, T3) else y
    fu, vu = ut3_orthogonal_representations(uy)
    res = fu.orthogonality_residual
    print('%-60s residual per element = %s' % (label, np.array2string(np.asarray(res), precision=2)))
    if np.max(np.asarray(res)) > 1e-8:
        # per-element ragged conversion and per-family check
        tree = fu.to_t3frame()
        elems = tree if isinstance(tree, (tuple, list)) else [tree]
        for k, fr in enumerate(elems):
            print('    elem %d: ragged-converted frame is_orthogonal=%s ranks up=%s down=%s left=%s right=%s' % (k, bool(fr.is_orthogonal()), fr.up_ranks, fr.down_ranks, fr.left_ranks, fr.right_ranks))
            U = fr.up_tucker_cores; D = fr.down_tt_cores; L = fr.left_tt_cores; R = fr.right_tt_cores
            ru = [float(np.max(np.abs(np.einsum('io,jo->ij', u, u) - np.eye(u.shape[0])))) for u in U]
            rd = [float(np.max(np.abs(np.einsum('iaj,ibj->ab', g, g) - np.eye(g.shape[1])))) for g in D]
            rl = [float(np.max(np.abs(np.einsum('iaj,iak->jk', g, g) - np.eye(g.shape[2])))) for g in L[:-1]]
            rr = [float(np.max(np.abs(np.einsum('iaj,kaj->ik', g, g) - np.eye(g.shape[0])))) for g in R[1:]]
            print('      residuals: up %s down %s left %s right %s' % (np.round(ru, 3), np.round(rd, 3), np.round(rl, 3), np.round(rr, 3)))
        print('    masks up:', fu.masks.up_mask.astype(int).tolist())
        print('    masks down:', fu.masks.down_mask.astype(int).tolist())
        print('    masks left:', fu.masks.frame_left_mask.astype(int).tolist())
        print('    masks right:', fu.masks.frame_right_mask.astype(int).tolist())
        # per-element ragged reference
        for k, ye in enumerate(y.unstack() if isinstance(y, T3) else []):
            fr, _ = t3_orthogonal_representations(ye)
            print('    ragged per-element %d: up=%s down=%s left=%s right=%s' % (k, fr.up_ranks, fr.down_ranks, fr.left_ranks, fr.right_ranks))
x = T3.randn((4, 6), (2, 3), (1, 2, 1), stack_shape=(2,))
check('d2 (2,) x + x (rank-deficient, Tucker 6 == N1)', x + x)
check('d2 (2,) x (full rank)', x)
x2 = T3.randn((4, 6), (2, 3), (1, 2, 1))
check('d2 () x + x', x2 + x2)
x3 = T3.randn((4, 6), (4, 6), (1, 2, 1), stack_shape=(2,))
check('d2 (2,) square Tucker factors, full rank', x3)
x4 = T3.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2,))
check('d3 (2,) x + x', x4 + x4)
x5 = T3.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1), stack_shape=(3,))
check('d3 (3,) x', x5)
x6 = T3.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2,)).resize((4, 5, 6), (3, 3, 3), (1, 2, 2, 1))
check('d3 (2,) resized (zero-padded Tucker ranks 3, non-minimal)', x6)
x7 = T3.randn((4, 5, 6), (2, 2, 2), (1, 4, 4, 1), stack_shape=(2,))
check('d3 (2,) non-minimal TT ranks (1,4,4,1)', x7)
