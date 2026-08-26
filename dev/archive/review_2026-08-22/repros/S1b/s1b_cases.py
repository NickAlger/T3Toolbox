"""S1b follow-up (1)/(2): which combinations of structural / numerical rank deficiency break the uniform
frame, and is the broken frame merely non-orthogonal or actually rank-deficient (lost directions)?"""
import numpy as np, sys
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ufvf
import t3toolbox.frame_variations_format as fvf
import t3toolbox.backend.ufv_masking as ufm
np.random.seed(0)

def real_block_ranks(uf):
    """Per core: (mask rank, numerical rank of the MASKED real block) for the four families."""
    mup, mdown, mleft, mright = ufm.ufv_apply_frame_masks(uf.data)
    um, dm, lm, rm = uf.data[5]
    out = {}
    d = mup.shape[0]
    out['up']    = [(int(um[i].sum()),  np.linalg.matrix_rank(np.asarray(mup[i]),  tol=1e-10)) for i in range(d)]       # rows over N
    out['down']  = [(int(dm[i].sum()),  np.linalg.matrix_rank(np.moveaxis(np.asarray(mdown[i]), 1, 0).reshape(mdown.shape[-2], -1), tol=1e-10)) for i in range(d)]
    out['left']  = [(int(lm[i+1].sum()), np.linalg.matrix_rank(np.asarray(mleft[i]).reshape(-1, mleft.shape[-1]), tol=1e-10)) for i in range(d-1)]
    out['right'] = [(int(rm[i].sum()),   np.linalg.matrix_rank(np.asarray(mright[i]).reshape(mright.shape[-3], -1), tol=1e-10)) for i in range(1, d)]
    return out

x0 = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))
cases = {
 'A  struct minimal,   numerically FULL   (control: randn (3,3,3)/(1,3,3,1))':      t3.TuckerTensorTrain.randn((5, 6, 7), (3, 3, 3), (1, 3, 3, 1)),
 'B  struct minimal,   numerically DEFICIENT (resize zero-pad (2,2,2)->(3,3,3))':  x0.resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1)),
 'C  struct NON-min,   numerically FULL   (randn (2,3,2)/(1,2,3,1): r2=3 > n2*r3=2)': t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 3, 1)),
 'D  struct NON-min,   numerically DEFICIENT (x0 + x0: (4,4,4)/(1,4,4,1))':         x0 + x0,
 'E  struct minimal,   numerically DEFICIENT, Tucker only (resize (2,2,2)->(3,3,3), tt unchanged)': x0.resize((5, 6, 7), (3, 3, 3), (1, 2, 2, 1)),
 'F  struct minimal,   numerically DEFICIENT, TT only (resize tt (1,2,2,1)->(1,3,3,1))':           x0.resize((5, 6, 7), (2, 2, 2), (1, 3, 3, 1)),
}
for name, x in cases.items():
    rf = fvf.T3Frame.from_t3(x)
    uf = ufvf.UT3Frame.from_ut3(ut3.UniformTuckerTensorTrain.from_t3(x))
    rr = real_block_ranks(uf)
    lost = sum(mr - nr for fam in rr.values() for (mr, nr) in fam)
    print('%s\n   has_minimal_ranks=%s | ragged frame residual %.1e | uniform frame residual %.1e | uniform LOST directions (mask rank - numerical rank of the masked block, summed over cores): %d'
          % (name, bool(x.has_minimal_ranks), float(np.max(rf.orthogonality_residual)), float(np.max(uf.orthogonality_residual)), lost))
    if lost:
        print('      per family (mask rank, masked-block rank):', {k: v for k, v in rr.items() if any(m != n for m, n in v)})
