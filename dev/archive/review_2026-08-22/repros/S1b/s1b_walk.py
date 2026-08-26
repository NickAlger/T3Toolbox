"""The walkthrough: the LAST core's down step (TT up-orthogonalization) on the zero-padded resize start.
Rows of the unfolding = (rL slot, rR slot) pairs; the real rR rank is 1 but the supercore pads rR to 3."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ut3_masking as um, t3toolbox.backend.ut3_orthogonalization as uo
import t3toolbox.backend.tt_orthogonalization as orth
np.set_printoptions(precision=3, suppress=True, linewidth=140)
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)).resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
mtk, mtt = um.ut3_apply_masks(ux.data)
uU, uG = uo.down_orthogonalize_tucker_supercores(mtk, mtt)     # step 1
uL = orth.tt_left_orthogonalize(uG)                            # step 2
uR, uH = orth.tt_right_orthogonalize(uL, return_variation_cores=True)   # step 3: H = the centers
H_last = np.asarray(uH[-1])                                     # (rL=3, n=3, rR=3), real rR = 1
print('H_last (rL=3, n=3, rR=3); real rR slot = 0 only. Its unfolding over (rL, rR) rows x n columns:')
M = H_last.swapaxes(-1, -2).reshape(9, 3)                      # rows (a, b) = (rL slot a, rR slot b)
labels = ['(a=%d,b=%d)%s' % (a, b, '' if b == 0 else ' PAD') for a in range(3) for b in range(3)]
for lab, row in zip(labels, M): print('  %-14s %s' % (lab, row))
O, ss, WT = np.linalg.svd(M, full_matrices=False)
print('singular values:', ss, '  <- the third is ZERO: mode slot 2 of the Tucker rank was zero-padded by resize')
print('left singular vectors O (9 x 3): column 3 is the null-space completion LAPACK chose:')
for lab, row in zip(labels, O): print('  %-14s %s' % (lab, row))
print('mass of column 3 in REAL rows (b=0): %.3f   in PADDED rows (b=1,2): %.3f' % (np.linalg.norm(O[0::3, 2]), np.linalg.norm(O[[1,2,4,5,7,8], 2])))
print()
print('RAGGED twin: the same step on the real slice only (rR = 1, so the unfolding is 3 x 3):')
Mr = M[0::3, :]
Or, ssr, _ = np.linalg.svd(Mr, full_matrices=False)
print('  singular values:', ssr, ' column 3 =', Or[:, 2], ' (no padded rows exist, so the completion is necessarily real)')
