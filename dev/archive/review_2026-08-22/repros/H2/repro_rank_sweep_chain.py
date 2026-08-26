"""rank_adjustment_sweep('right_to_left') is documented to return a RIGHT-ORTHOGONAL T3 (and to be the
verification step before t3svd(assume_orthogonal=True)). On a generic input it is not; the documented
chain then truncates wrongly, silently."""
import numpy as np, t3toolbox as t3t
np.random.seed(0)
shape = (5, 6, 7)
x = t3t.TuckerTensorTrain.randn(shape, (3, 4, 3), (1, 3, 3, 1))
def core_res(y):
    out = []
    for i, B in enumerate(y.tucker_cores):          # (n_i, N_i)
        out.append('B%d: BBt %.1e BtB %.1e' % (i, np.linalg.norm(B @ B.T - np.eye(B.shape[0])), np.linalg.norm(B.T @ B - np.eye(B.shape[1]))))
    for i, G in enumerate(y.tt_cores):              # (rL, n, rR)
        M = G.reshape(G.shape[0], -1)               # right-orth: M M^T = I
        L = G.reshape(-1, G.shape[-1])              # left-orth:  L^T L = I
        out.append('G%d: right %.1e left %.1e' % (i, np.linalg.norm(M @ M.T - np.eye(M.shape[0])), np.linalg.norm(L.T @ L - np.eye(L.shape[1]))))
    return out
y = x.rank_adjustment_sweep('right_to_left')
print('generic x -> rank_adjustment_sweep(right_to_left):')
print('  is_right_orthogonal():', bool(y.is_right_orthogonal()))
for s in core_res(y): print('  ', s)
z = x.t3svd()[0].rank_adjustment_sweep('right_to_left')
print('left-orth x -> rank_adjustment_sweep(right_to_left): is_right_orthogonal():', bool(z.is_right_orthogonal()))
# the documented chain
xd = x.to_dense()
good = x.t3svd(max_tt_ranks=2)[0]
chain = y.t3svd(max_tt_ranks=2, assume_orthogonal=True)[0]
print('truncation err  x.t3svd(max_tt_ranks=2)                                     : %.4f' % (np.linalg.norm(good.to_dense() - xd) / np.linalg.norm(xd)))
print('truncation err  x.rank_adjustment_sweep(r2l).t3svd(max_tt_ranks=2, assume_orthogonal=True): %.4f' % (np.linalg.norm(chain.to_dense() - xd) / np.linalg.norm(xd)))
print('(same ranks both:', good.tt_ranks, chain.tt_ranks, ')')
