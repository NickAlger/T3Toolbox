"""Directional SVDs and the four *_svd_pair at non-square / rank-deficient / rank-1 / stacked inputs; drift between the four."""
import numpy as np
import t3toolbox.backend.linalg as L
np.random.seed(0)
bad = []
def chk(name, cond, info=''):
    if not cond: bad.append((name, info))
for (ni, na, nj) in [(1, 1, 5), (5, 1, 1), (2, 7, 3), (6, 2, 1), (1, 4, 3), (3, 2, 9)]:
    for ss in [(), (2,), (2, 3)]:
        G = np.random.randn(*ss, ni, na, nj)
        U, s, Vt = L.left_svd(G);  chk('left_svd recon', np.allclose(np.einsum('...iax,...x,...xj->...iaj', U, s, Vt), G), (ni, na, nj, ss))
        chk('left_svd rank', s.shape[-1] == min(ni * na, nj), (s.shape, ni, na, nj))
        U, s, Vt = L.right_svd(G); chk('right_svd recon', np.allclose(np.einsum('...ix,...x,...xaj->...iaj', U, s, Vt), G), (ni, na, nj, ss))
        chk('right_svd rank', s.shape[-1] == min(ni, na * nj), (s.shape, ni, na, nj))
        U, s, Vt = L.up_svd(G);    chk('up_svd recon', np.allclose(np.einsum('...ixj,...x,...xa->...iaj', U, s, Vt), G), (ni, na, nj, ss))
        chk('up_svd rank', s.shape[-1] == min(ni * nj, na), (s.shape, ni, na, nj))
        # rank-deficient: G built from rank-1 factors -> truncated with rtol keeps 1 (unstacked only)
        if ss == ():
            Gr = np.einsum('i,a,j->iaj', np.random.randn(ni), np.random.randn(na), np.random.randn(nj))
            for f, sub in [(L.left_svd, '...iax,...x,...xj->...iaj'), (L.right_svd, '...ix,...x,...xaj->...iaj'), (L.up_svd, '...ixj,...x,...xa->...iaj')]:
                U, s, Vt = f(Gr, rtol=1e-10); chk('rank-1 rtol keeps 1 ' + f.__name__, s.shape[-1] == 1, s.shape)
                chk('rank-1 recon ' + f.__name__, np.allclose(np.einsum(sub, U, s, Vt), Gr))
        # pairs
        nb, nk, N = 3, 2, 4
        G1 = np.random.randn(*ss, nj, nb, nk); B = np.random.randn(*ss, na, N)
        for mr in [None, 1, 2]:
            nG0, nG1, s = L.left_svd_pair(G, G1, max_rank=mr)
            chk('left_pair prod', (mr is None or mr >= min(ni*na, nj)) <= np.allclose(np.einsum('...iax,...xbk->...iabk', nG0, nG1), np.einsum('...iaj,...jbk->...iabk', G, G1)), (ni, na, nj, ss, mr))
            chk('left_pair orth', np.allclose(np.einsum('...iax,...iay->...xy', nG0, nG0), np.eye(nG0.shape[-1])), (ni, na, nj, ss, mr))
            chk('left_pair shapes', nG0.shape == ss + (ni, na, s.shape[-1]) and nG1.shape == ss + (s.shape[-1], nb, nk), (nG0.shape, nG1.shape))
            nG0, nG1, s = L.right_svd_pair(G, G1, max_rank=mr)
            chk('right_pair prod', (mr is None or mr >= min(nj, nb*nk)) <= np.allclose(np.einsum('...iax,...xbk->...iabk', nG0, nG1), np.einsum('...iaj,...jbk->...iabk', G, G1)), (ni, na, nj, ss, mr))
            chk('right_pair orth', np.allclose(np.einsum('...xbk,...ybk->...xy', nG1, nG1), np.eye(nG1.shape[-3])), (ni, na, nj, ss, mr))
            nG, nB, s = L.up_svd_pair(G, B, max_rank=mr)
            chk('up_pair prod', (mr is None or mr >= min(ni*nj, na)) <= np.allclose(np.einsum('...ixj,...xo->...ijo', nG, nB), np.einsum('...iaj,...ao->...ijo', G, B)), (ni, na, nj, ss, mr))
            chk('up_pair orth', np.allclose(np.einsum('...ixj,...iyj->...xy', nG, nG), np.eye(nG.shape[-2])), (ni, na, nj, ss, mr))
            nG, nB, s = L.down_svd_pair(G, B, max_rank=mr)
            chk('down_pair prod', (mr is None or mr >= min(na, N)) <= np.allclose(np.einsum('...ixj,...xo->...ijo', nG, nB), np.einsum('...iaj,...ao->...ijo', G, B)), (ni, na, nj, ss, mr))
            chk('down_pair orth', np.allclose(np.einsum('...xo,...yo->...xy', nB, nB), np.eye(nB.shape[-2])), (ni, na, nj, ss, mr))
print('FAILURES:', bad if bad else 'none')
# rank-1 / degenerate
print('truncated_svd zero matrix rtol=1e-3 kept rank:', L.truncated_svd(np.zeros((3, 4)), rtol=1e-3)[1].shape)
print('truncated_svd zero matrix atol=1e-3 kept rank:', L.truncated_svd(np.zeros((3, 4)), atol=1e-3)[1].shape)
print('truncated_svd max_rank=0 kept rank:', L.truncated_svd(np.random.randn(3, 4), max_rank=0)[1].shape)
print('truncated_svd min_rank=3 > max_rank=1 kept rank:', L.truncated_svd(np.random.randn(3, 4), min_rank=3, max_rank=1)[1].shape, '(min_rank wins)')
print('pad_or_truncate (2,3) -> [(0,-3),(0,0)] (truncate whole axis):', L.pad_or_truncate(np.ones((2, 3)), [(0, -3), (0, 0)]).shape)
