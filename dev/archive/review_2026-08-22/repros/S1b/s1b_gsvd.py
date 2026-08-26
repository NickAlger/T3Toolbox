"""GSVD of (M^T, I_real) vs the augmented SVD of [M | eps I_real] on the walkthrough's 9x3 unfolding."""
import numpy as np, scipy.linalg as sla
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ut3_masking as um, t3toolbox.backend.ut3_orthogonalization as uo, t3toolbox.backend.tt_orthogonalization as orth
np.set_printoptions(precision=4, suppress=True, linewidth=150)
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)).resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))
mtk, mtt = um.ut3_apply_masks(ut3.UniformTuckerTensorTrain.from_t3(x).data)
uU, uG = uo.down_orthogonalize_tucker_supercores(mtk, mtt); uL = orth.tt_left_orthogonalize(uG)
_, uH = orth.tt_right_orthogonalize(uL, return_variation_cores=True)
M = np.asarray(uH[-1]).swapaxes(-1, -2).reshape(9, 3)            # rows (a,b); real rows: b == 0
real = np.array([b == 0 for a in range(3) for b in range(3)])
labels = ['(a=%d,b=%d)%s' % (a, b, '' if b == 0 else ' PAD') for a in range(3) for b in range(3)]
U0, s0, _ = np.linalg.svd(M, full_matrices=False)

# --- (1) augmented SVD: [M | eps * I_real]  (I_real = the real coordinate vectors as columns)
eps = 1e-6 * s0[0]
E = np.eye(9)[:, real]                                             # 9 x 3 (three real rows)
Ua, sa, _ = np.linalg.svd(np.concatenate([M, eps * E], axis=1), full_matrices=False)
print('augmented SVD singular values:', sa, ' (sqrt(s^2+eps^2), sqrt(s^2+eps^2), eps, 0, 0, 0)')
print('  |<U_aug[:,k], U_plain[:,k]>| for the two sigma>0 columns:', np.abs(np.sum(Ua[:, :2] * U0[:, :2], axis=0)), ' (exact: the pair commutes)')
print('  column 3 (the completion):'); [print('    %-14s %+.3f' % (lab, v)) for lab, v in zip(labels, Ua[:, 2])]
print('  mass of column 3 in real rows: %.3f, in padded rows: %.3f' % (np.linalg.norm(Ua[real, 2]), np.linalg.norm(Ua[~real, 2])))

# --- (2) the GSVD of the pair (A, B) = (M^T, E^T): LAPACK dggsvd3 if SciPy exposes it
A, B = M.T.copy(), E.T.copy()                                      # A: 3 x 9, B: 3 x 9 -- both with 9 columns (the R index)
try:
    ggsvd = sla.lapack.dggsvd3
    alpha, beta, u, v, q, work, k, l, iwork, info = ggsvd(A, B)[:10] if False else (None,)*10
except Exception as e:
    ggsvd = None
if ggsvd is None or alpha is None:
    # scipy's wrapper signature varies; fall back to the textbook route: the CS decomposition via a QR of the stacked
    # matrix [A; B] (this IS how the GSVD is computed) -- the generalized singular pairs (c_i, s_i) and the common basis Q.
    S = np.concatenate([A, B], axis=0)                               # (3+3) x 9
    # Paige-Saunders preprocessing: an orthonormal basis Q1 of the row space of the stacked [A; B] (rank-revealing --
    # an unpivoted QR would fill a zero column with an arbitrary vector, the very trap under discussion)
    _, ss, Vt_s = np.linalg.svd(S, full_matrices=False)
    r = int(np.sum(ss > 1e-10 * ss[0]))
    print('\nGSVD route (CS decomposition of the stacked [A; B], rank %d of 9): the orthogonal complement of the row space = null(A) ∩ null(B) = the PADDED rows' % r)
    Q1 = Vt_s[:r].T
    # within the row space, the CS decomposition of (A Q1, B Q1) splits it into sigma>0 directions (c>0) and B-only directions (c=0, s=1)
    Ua2, c, Vh = np.linalg.svd(A @ Q1, full_matrices=False)
    W = Q1 @ Vh.T                                                    # the generalized singular directions, sorted by c = sigma(A) on the row space
    print('  generalized singular values c (A-side):', c, ' -> two sigma>0 directions, then the B-only (real complement) direction')
    print('  |<W[:,k], U_plain[:,k]>| for k < 2:', np.abs(np.sum(W[:, :2] * U0[:, :2], axis=0)))
    print('  B-only direction (c = 0): mass in real rows %.3f, padded rows %.3f' % (np.linalg.norm(W[real, 2]), np.linalg.norm(W[~real, 2])))
    print('  |<W[:,2], U_aug[:,2]>| (GSVD completion vs augmented-SVD completion): %.3f' % abs(W[:, 2] @ Ua[:, 2]))
